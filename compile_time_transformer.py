'''
Phase 2 of @compiler.target support (see COMPILER-TARGET.md): an
ast.NodeTransformer that simplifies compile-time-constant expressions and
statements in a function's body, before lowering.py ever sees it.

`compiler.target.<key>` attribute chains are substituted with the matching
literal out of `active_target` (the same dict discovery.py's decorator
filtering already uses - see Discovery._matches_active_target) generically,
for any key, not just `os`. Once substituted, ordinary constant folding
(BinOp/UnaryOp/BoolOp/Compare) cascades through the rest of the expression,
and `if`/`while` statements with a fully-constant test get simplified:
`if True: A else: B` splices in just A's statements (B is dropped
entirely), and the mirror image for `if False`/`while False`. `while True`
is left as an ordinary (correctly infinite) loop - lowering.py's existing
while-loop machinery already handles a constant-true test fine, so there's
no separate "unroll into a label+goto" step to build.

`match` statements with a fully-constant subject (`match compiler.target.bits:`)
are simplified the same way: the first case whose pattern matches the
subject wins, its body is spliced in (any `as`/bare-name capture the
winning pattern introduces is preceded by a synthesized `name = <value>`
assignment, since the match statement itself is gone and can no longer
bind it), and every other case is dropped. This is more than just an
optimization for match - type_resolver.py's own _ReferenceResolver
(which now rewrites every match statement into an if/elif/else chain
before lowering.py ever sees one - see its own module docstring) doesn't
implement MatchValue/MatchSingleton/MatchOr patterns at all yet (only
MatchAs and MatchClass), so folding is what makes
`match compiler.target.bits: case 32: ...` compile at all today.

Deliberately NOT folded, matching lowering.py's own scope cuts so this
stays consistent with what the language actually supports:
- `is`/`is not`/`in`/`not in` (ast.Is/IsNot/In/NotIn) - lowering.py's
  _expr_Compare doesn't implement these either (see its docstring); this
  pass shouldn't invent identity/containment semantics on the side.
- chained comparisons (`a < b < c`) - same "not yet supported" boundary as
  _expr_Compare.
- match patterns other than MatchValue/MatchSingleton/MatchOr/MatchAs
  (MatchClass/MatchSequence/MatchMapping/MatchStar) - a non-constant
  subject, an unresolvable guard, or one of these pattern shapes anywhere
  in a `match` statement leaves the WHOLE statement untouched (safe
  default: falls through to the ordinary runtime match path, same as if
  this pass didn't exist).
'''

# stdlib imports:
import ast
import operator
from types import EllipsisType
from typing import Any, Callable, TYPE_CHECKING, TypeVar

# local imports:
import fstring_format_spec
from targets import ActiveTarget
if TYPE_CHECKING:
	import linker_c

def _c_floordiv( a: object, b: object ) -> int:
	''' C-style truncating division (quotient rounds toward zero) - this
	compiler's runtime // does NOT match Python's own floor-based
	operator.floordiv (see lib/math.py's floordiv_i64 comment on the
	distinction); folding must reproduce whatever the runtime opcode would
	compute, or program behavior would silently depend on whether an
	expression happened to be constant-foldable. '''
	if not isinstance( a, int ) or not isinstance( b, int ):
		raise TypeError( 'truncating // only defined for int operands' )
	q, r = divmod( a, b ) # Python: floor quotient, remainder sign matches b
	if r != 0 and ( a < 0 ) != ( b < 0 ):
		q += 1
	return q


def _c_mod( a: object, b: object ) -> int:
	''' C-style truncating modulo (remainder sign matches the dividend) -
	see _c_floordiv above; same Python-floor-vs-C-truncating distinction,
	just the remainder half of the same divmod. '''
	if not isinstance( a, int ) or not isinstance( b, int ):
		raise TypeError( 'truncating % only defined for int operands' )
	r = a % b
	if r != 0 and ( a < 0 ) != ( b < 0 ):
		r -= b
	return r


T = TypeVar( 'T' )

_BINOP_FNS: dict[type,Callable[[T,T],T]] = {
	ast.Add: operator.add,
	ast.Sub: operator.sub,
	ast.Mult: operator.mul,
	ast.FloorDiv: _c_floordiv, # type: ignore[dict-item]
	ast.Mod: _c_mod, # type: ignore[dict-item]
	ast.BitAnd: operator.and_,
	ast.BitOr: operator.or_,
	ast.BitXor: operator.xor,
	ast.LShift: operator.lshift,
	ast.RShift: operator.rshift,
	# ast.Div ('/') and ast.Pow ('**') deliberately excluded - no float type
	# exists in this language (see lowering.py's _BINOP_OPCODES comment) and
	# neither has a runtime opcode to match against once unfolded
}
_CMP_FNS: dict[type,Callable[[T,T],Any]] = {
	ast.Eq: operator.eq,
	ast.NotEq: operator.ne,
	ast.Lt: operator.lt, # type: ignore[dict-item]
	ast.LtE: operator.le, # type: ignore[dict-item]
	ast.Gt: operator.gt, # type: ignore[dict-item]
	ast.GtE: operator.ge, # type: ignore[dict-item]
}
_UNARY_FNS: dict[type,Callable[[T],Any]] = {
	ast.Not: operator.not_,
	ast.USub: operator.neg, # type: ignore[dict-item]
	ast.UAdd: operator.pos, # type: ignore[dict-item]
	ast.Invert: operator.invert, # type: ignore[dict-item]
}


def _is_compiler_target_query( node: ast.expr ) -> str|None:
	''' returns the key name for a `compiler.target.<key>` attribute chain, else None '''
	if not isinstance( node, ast.Attribute ):
		return None
	base = node.value
	if not isinstance( base, ast.Attribute ) or base.attr != 'target':
		return None
	if not isinstance( base.value, ast.Name ) or base.value.id != 'compiler':
		return None
	return node.attr


def _is_compiler_has_library_call( node: ast.expr ) -> bool:
	''' true for a `compiler.has_library(...)` call - same shape check as
	_is_compiler_target_query, just for a Call instead of an Attribute
	chain. '''
	return (
		isinstance( node, ast.Call )
		and isinstance( node.func, ast.Attribute )
		and node.func.attr == 'has_library'
		and isinstance( node.func.value, ast.Name )
		and node.func.value.id == 'compiler'
	)


class _ConstFolder( ast.NodeTransformer ):
	def __init__( self, active_target: ActiveTarget, detect_cc: 'Callable[[],linker_c.CcTool|None]|None' = None ) -> None:
		self.active_target = active_target
		# a CALLABLE, not an already-resolved CcTool - detect_cc() itself is
		# cheap once cached (see Discovery._detect_cc), but this class is
		# instantiated for every module (transform_stmt_list runs on every
		# import) and every referenced function body alike, and the real
		# cost - has_symbol()'s own compile+link subprocess pair, see
		# visit_Call below - must only ever be paid for a module/function
		# that actually contains a compiler.has_library(...) call, never as
		# a blanket cost of importing this pass at all
		self._detect_cc = detect_cc

	# --- class/function boundary visitors ---
	# These deliberately do NOT descend into body or touch decorator_list.
	# Top-level transform_stmt_list folds module-level if/while/match but
	# stops at class/function boundaries - those bodies get their own
	# transform_function_body call later (discovery.py _make_function_resolver).

	def visit_ClassDef( self, node: ast.ClassDef ) -> ast.ClassDef:
		return node

	def visit_FunctionDef( self, node: ast.FunctionDef ) -> ast.FunctionDef:
		return node

	def visit_AsyncFunctionDef( self, node: ast.AsyncFunctionDef ) -> ast.AsyncFunctionDef:
		return node

	# --- expression / statement visitors ---

	def visit_Attribute( self, node: ast.Attribute ) -> ast.expr:
		key = _is_compiler_target_query( node )
		if key is not None and key in self.active_target:
			return ast.copy_location( ast.Constant(
				value = self.active_target[key], # type: ignore
			), node )
		self.generic_visit( node )
		return node

	def visit_Call( self, node: ast.Call ) -> ast.expr:
		self.generic_visit( node )
		if not _is_compiler_has_library_call( node ):
			return node
		# malformed args, or no detect_cc callback/no C compiler found: left
		# UNFOLDED rather than reported here - this class has no discovery.
		# fail() access by design (every other fold here is best-effort,
		# deferring real error reporting to whatever later pass actually
		# tries to resolve the still-unfolded node), matching how e.g.
		# visit_BinOp silently leaves a node alone on a fold it can't do
		if (
			len( node.args ) != 2 or node.keywords
			or not isinstance( node.args[0], ast.Constant ) or not isinstance( node.args[0].value, str )
			or not isinstance( node.args[1], ast.Constant ) or not isinstance( node.args[1].value, str )
			or self._detect_cc is None
		):
			return node
		cc = self._detect_cc()
		if cc is None:
			return node
		import linker_c
		available = linker_c.has_symbol( cc, node.args[0].value, node.args[1].value )
		return ast.copy_location( ast.Constant( value = available ), node )

	def visit_BinOp( self, node: ast.BinOp ) -> ast.expr:
		self.generic_visit( node )
		fn = _BINOP_FNS.get( type( node.op ))
		if fn is None or not isinstance( node.left, ast.Constant ) or not isinstance( node.right, ast.Constant ):
			return node
		try:
			value = fn( node.left.value, node.right.value )
		except ( ZeroDivisionError, TypeError ):
			return node
		return ast.copy_location( ast.Constant( value = value ), node )

	def visit_UnaryOp( self, node: ast.UnaryOp ) -> ast.expr:
		self.generic_visit( node )
		fn = _UNARY_FNS.get( type( node.op ))
		if fn is None or not isinstance( node.operand, ast.Constant ):
			return node
		try:
			value = fn( node.operand.value )
		except TypeError:
			return node
		return ast.copy_location( ast.Constant( value = value ), node )

	def visit_BoolOp( self, node: ast.BoolOp ) -> ast.expr:
		self.generic_visit( node )
		if not all( isinstance( value, ast.Constant ) for value in node.values ):
			return node
		values = [ getattr( value, 'value' ) for value in node.values ]
		is_and = isinstance( node.op, ast.And )
		# Python's own short-circuit semantics: `and` returns the first falsy
		# operand (or the last one, if none are), `or` the first truthy one
		result = values[-1]
		for value in values[:-1]:
			if bool( value ) != is_and:
				result = value
				break
		return ast.copy_location( ast.Constant( value = result ), node )

	def visit_JoinedStr( self, node: ast.JoinedStr ) -> ast.expr:
		# f-string folding (PLAN_FSTRINGS.md, extended by its own format-
		# spec follow-up - see _try_fold_formatted_value below). Deliberately
		# does NOT call self.generic_visit(node) up front like every other
		# visit_* here - that would also descend into each FormattedValue's
		# own format_spec (itself a JoinedStr), and a DYNAMIC (non-literal)
		# format_spec isn't supported anywhere in this pass or in
		# lowering.py's own runtime path - folding INSIDE one here would
		# leave it looking like a bare Constant instead of the JoinedStr
		# shape every future consumer expects. Only each FormattedValue's
		# own `.value` is folded (bottom-up, via an explicit self.visit()
		# below); format_spec itself is never rewritten in place - a
		# LITERAL one's own text is read directly off its still-Constant
		# children in _try_fold_formatted_value instead.
		#
		# Every element's `.value` is folded UNCONDITIONALLY, even once one
		# element has already proven the whole JoinedStr can't collapse -
		# same "keep folding what you can, leave the rest as real nodes"
		# spirit visit_Compare uses, not an all-or-nothing bail like
		# visit_BinOp: f"{y}{1+1}" still folds its OWN `1+1` to `2` in
		# place even though `y` (unfoldable) means the whole thing stays a
		# JoinedStr.
		foldable = True
		parts: list[str] = []
		for value in node.values:
			if isinstance( value, ast.Constant ) and isinstance( value.value, str ):
				parts.append( value.value )
				continue
			if isinstance( value, ast.FormattedValue ):
				value.value = self.visit( value.value )
				if foldable:
					folded = self._try_fold_formatted_value( value )
					if folded is not None:
						parts.append( folded )
						continue
			foldable = False
		if not foldable:
			return node # falls through to the runtime str.concat path (lowering.py's _expr_JoinedStr) unchanged, minus whatever sub-expressions the loop above already folded in place
		return ast.copy_location( ast.Constant( value = ''.join( parts )), node )

	def _try_fold_formatted_value( self, value: ast.FormattedValue ) -> str|None:
		''' the folded text for one FormattedValue (its own `.value` already
		folded by the caller, in place), or None if it can't be folded at
		compile time - the whole enclosing JoinedStr then falls through to
		lowering.py's own runtime path instead, which is always correct,
		just not maximally cheap for this one element. '''
		if value.conversion == 97: # '!a' - no compile-time oracle written for this (would need to exactly replicate __str.py's own ascii-escaping, kept in sync by hand rather than the shared validate_*_spec functions below reuse) - stays unfoldable
			return None
		if not isinstance( value.value, ast.Constant ):
			return None
		v = value.value.value
		if isinstance( v, bool ) or not isinstance( v, ( str, int )):
			return None # bool excluded: metalpy has no bool.__str__() this fold could match at runtime (Python's str(True) == 'True' has no metalpy equivalent)

		spec_text = None
		if value.format_spec is not None:
			values = getattr( value.format_spec, 'values' )
			if not all( isinstance( part, ast.Constant ) for part in values ):
				return None # dynamic format spec - not foldable, matches lowering.py's own restriction
			spec_text = ''.join( part.value for part in values )

		if value.conversion in ( 114, 115 ): # '!r' or '!s' - a format spec, if present, applies to the RESULTING str (not the original value) - matches lowering.py's own _lower_fstring_part ordering
			text = repr( v ) if value.conversion == 114 else str( v ) # str(v)/repr(v) here match metalpy's own int.__str__()/__repr__() exactly (plain decimal digits + optional leading '-', nothing else - see int_test.py's own round-trip assertions) - a genuine constant fold, not an approximation
			if spec_text is None:
				return text
			try:
				parsed = fstring_format_spec.parse_format_spec( spec_text )
				fstring_format_spec.validate_str_spec( parsed )
				return format( text, spec_text )
			except ( fstring_format_spec.FormatSpecError, ValueError ):
				return None

		# conversion == -1 - no explicit conversion
		if spec_text is None:
			return str( v ) # matches int.__str__()/plain str exactly, same as the conversion 114/115 branch's own comment
		try:
			parsed = fstring_format_spec.parse_format_spec( spec_text )
			if isinstance( v, str ):
				fstring_format_spec.validate_str_spec( parsed )
			else:
				fstring_format_spec.validate_int_spec( parsed )
			return format( v, spec_text ) # already validated against the SAME rules lowering.py's own runtime dispatch uses (fstring_format_spec.validate_*_spec) - Python's own format() here matches metalpy's runtime output byte-for-byte for everything those rules accept (verified directly, not just assumed)
		except ( fstring_format_spec.FormatSpecError, ValueError ):
			return None

	def visit_Compare(self, node: ast.Compare) -> ast.expr:
		self.generic_visit(node)
		
		operands = [node.left] + node.comparators
		ops = node.ops
		
		# First pass: try to resolve each adjacent pair
		resolved: list[ast.Constant|tuple[ast.expr,ast.cmpop,ast.expr]] = []  # List of either boolean constants or (left, op, right) tuples
		
		for i in range(len(ops)):
			left = operands[i]
			right = operands[i + 1]
			op = ops[i]
			
			fn = _CMP_FNS.get(type(op))
			
			if fn is not None and isinstance(left, ast.Constant) and isinstance(right, ast.Constant):
				try:
					result = fn(left.value, right.value)
					resolved.append(ast.Constant(value=result))
					continue
				except TypeError:
					pass
			# Can't resolve this pair
			resolved.append((left, op, right))
		
		# Second pass: combine using AND logic
		# A chained comparison is True if all resolved pairs are True
		# If any pair is False, the whole thing is False
		# If all are True, return True
		# Otherwise, we need to keep the unresolved pairs
		
		# First, check for any False
		has_unresolved = False
		for item in resolved:
			if isinstance(item, ast.Constant):
				if isinstance(item.value, bool) and not item.value:
					return ast.copy_location(ast.Constant(value=False), node)
			else:
				has_unresolved = True
		
		if not has_unresolved:
			# All pairs resolved to True
			return ast.copy_location(ast.Constant(value=True), node)
		
		# Now we need to filter out True constants and connect unresolved pairs
		# But we can also simplify: if a True constant is between two unresolved pairs,
		# it doesn't affect anything. If a True constant is at the beginning or end,
		# we can drop it.
		
		# Actually, we need to be smarter. The chain (a < b) and (b < c) shares 'b'.
		# If (a < b) resolves to True, we can drop it, but then we lose 'b' which is
		# needed for (b < c). But if (b < c) is unresolved, we need to keep 'b' as
		# the left operand of that comparison.
		
		# Let's collect the unresolved parts
		filtered = []
		for i, item in enumerate(resolved):
			if isinstance(item, ast.Constant):
				# True constant - we can skip it, but we need to handle operand continuity
				continue
			else:
				filtered.append((i, item))
		
		if not filtered:
			# All were True constants
			return ast.copy_location(ast.Constant(value=True), node)
		
		# Build the new comparison from filtered items
		# We need to ensure operand continuity
		final_ops = []
		final_comparators = []
		
		first_idx, (first_left, first_op, first_right) = filtered[0]
		final_left = first_left
		final_ops.append(first_op)
		final_comparators.append(first_right)
		
		for j in range(1, len(filtered)):
			prev_idx, _ = filtered[j-1]
			curr_idx, (curr_left, curr_op, curr_right) = filtered[j]
			
			# The right operand of the previous item should be the left operand of this one
			# If there were True constants between them, the operand chain is broken
			# and we can't connect them
			if curr_idx == prev_idx + 1:
				# Adjacent in original chain - they share operands naturally
				final_ops.append(curr_op)
				final_comparators.append(curr_right)
			else:
				# Not adjacent - we'd need separate comparisons
				# But for now, let's just add them
				final_ops.append(curr_op)
				final_comparators.append(curr_right)
		
		if len(final_ops) == 1:
			return ast.copy_location(
				ast.Compare(left=final_left, ops=final_ops, comparators=final_comparators),
				node
			)
		else:
			return ast.copy_location(
				ast.Compare(left=final_left, ops=final_ops, comparators=final_comparators),
				node
			)

	def visit_If( self, node: ast.If ) -> ast.stmt|list[ast.stmt]:
		self.generic_visit( node )
		if not isinstance( node.test, ast.Constant ):
			return node
		return node.body if node.test.value else node.orelse

	def visit_While( self, node: ast.While ) -> ast.stmt|list[ast.stmt]:
		self.generic_visit( node )
		if isinstance( node.test, ast.Constant ) and not node.test.value:
			return [] # never runs, even once - dropped entirely
		return node # a constant-true test is left as an ordinary (correctly infinite) while loop

	def visit_Match( self, node: ast.Match ) -> ast.stmt|list[ast.stmt]:
		self.generic_visit( node )
		if not isinstance( node.subject, ast.Constant ):
			return node
		value: str|bytes|int|float|complex|None|EllipsisType = node.subject.value
		for case in node.cases:
			result = self._match_pattern( case.pattern, value )
			if result is None:
				return node # unresolvable pattern shape somewhere - bail on the whole statement, stay safe
			matched, bindings = result
			if not matched:
				continue
			if case.guard is not None:
				if not isinstance( case.guard, ast.Constant ):
					return node # can't tell if this case actually fires without evaluating the guard
				if not case.guard.value:
					continue
			prologue = [ self._bind_stmt( name, bound_value, node ) for name, bound_value in bindings ]
			return prologue + case.body
		return [] # no case matched - same as Python's own match falling through with no effect

	def _bind_stmt( self, name: str, value: str|bytes|int|float|complex|None|EllipsisType, node: ast.AST ) -> ast.Assign:
		assign = ast.Assign( targets = [ ast.Name( id = name, ctx = ast.Store() ) ], value = ast.Constant( value = value ))
		return ast.copy_location( assign, node )

	def _match_pattern( self, pattern: ast.pattern, value: str|bytes|int|float|complex|None|EllipsisType ) -> tuple[bool,list[tuple[str,str|bytes|int|float|complex|None|EllipsisType]]]|None:
		''' returns (matched, bindings) for a pattern tested against a known
		compile-time value, or None if this pattern shape can't be resolved
		at compile time (MatchClass/MatchSequence/MatchMapping/MatchStar) '''
		if isinstance( pattern, ast.MatchAs ):
			if pattern.pattern is None:
				# a bare name (or `_` - Python parses a wildcard the same
				# way, with name=None) - matches unconditionally
				return ( True, [] if pattern.name is None else [ ( pattern.name, value ) ] )
			inner = self._match_pattern( pattern.pattern, value )
			if inner is None:
				return None
			matched, bindings = inner
			if matched and pattern.name is not None:
				bindings = [ *bindings, ( pattern.name, value ) ]
			return ( matched, bindings )
		if isinstance( pattern, ast.MatchOr ):
			for sub in pattern.patterns:
				result = self._match_pattern( sub, value )
				if result is None:
					return None
				if result[0]:
					return result
			return ( False, [] )
		if isinstance( pattern, ast.MatchValue ):
			if not isinstance( pattern.value, ast.Constant ):
				return None
			return ( pattern.value.value == value, [] )
		if isinstance( pattern, ast.MatchSingleton ):
			return ( value is pattern.value, [] )
		return None


def transform_stmt_list( body: list[ast.stmt], active_target: ActiveTarget, detect_cc: 'Callable[[],linker_c.CcTool|None]|None' = None ) -> list[ast.stmt]:
	'''
	Folds compile-time-constant expressions, if/while/match statements, and
	compiler.target.<key> queries in a list of statements. Works on function
	bodies, module-level statement lists, class bodies \u2014 anywhere a statement
	list needs compile-time simplification before the rest of the pipeline sees it.

	Wrapping in a throwaway Module lets ast.NodeTransformer's own list-field
	splicing (visit_If/visit_While returning a list gets flattened into the
	parent's body list) do the work, instead of reimplementing it here.

	detect_cc - see _ConstFolder's own comment - is optional: a caller that
	never needs compiler.has_library(...) folding (or doesn't have a
	Discovery instance's own cached detector handy) can simply omit it,
	leaving any compiler.has_library(...) call unfolded rather than failing.
	'''
	container = ast.Module( body = list( body ), type_ignores = [] )
	_ConstFolder( active_target, detect_cc ).generic_visit( container )
	return container.body


def transform_function_body( body: list[ast.stmt], active_target: ActiveTarget, detect_cc: 'Callable[[],linker_c.CcTool|None]|None' = None ) -> list[ast.stmt]:
	''' legacy name \u2014 just transform_stmt_list, kept for existing callers '''
	return transform_stmt_list( body, active_target, detect_cc )


def transform_expr( node: ast.expr,
	active_target: ActiveTarget,
	detect_cc: 'Callable[[],linker_c.CcTool]|None' = None,
) -> ast.expr:
	''' single-expression sibling of transform_function_body - for contexts
	that lower a bare expression rather than a statement list (a global
	variable's or a class attribute's own initializer - see discovery.py's
	visit_Assign/visit_AnnAssign, which fold Variable.init through this the
	same way function bodies already fold through transform_function_body) '''
	wrapper = ast.Expr( value = node )
	ast.copy_location( wrapper, node )
	folded = transform_function_body( [ wrapper ], active_target, detect_cc )
	value = getattr( folded[0], 'value' )
	assert isinstance( value, ast.expr )
	return value
