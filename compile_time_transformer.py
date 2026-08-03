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
optimization for match - lowering.py's own _stmt_Match doesn't implement
MatchValue/MatchSingleton/MatchOr patterns at all yet (only MatchAs and
MatchClass - see its docstring), so folding is what makes
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

_BINOP_FNS: dict[type,object] = {
	ast.Add: operator.add,
	ast.Sub: operator.sub,
	ast.Mult: operator.mul,
	ast.FloorDiv: operator.floordiv,
	ast.Mod: operator.mod,
	ast.BitAnd: operator.and_,
	ast.BitOr: operator.or_,
	ast.BitXor: operator.xor,
	ast.LShift: operator.lshift,
	ast.RShift: operator.rshift,
	# ast.Div ('/') and ast.Pow ('**') deliberately excluded - no float type
	# exists in this language (see lowering.py's _BINOP_OPCODES comment) and
	# neither has a runtime opcode to match against once unfolded
}
_CMP_FNS: dict[type,object] = {
	ast.Eq: operator.eq,
	ast.NotEq: operator.ne,
	ast.Lt: operator.lt,
	ast.LtE: operator.le,
	ast.Gt: operator.gt,
	ast.GtE: operator.ge,
}
_UNARY_FNS: dict[type,object] = {
	ast.Not: operator.not_,
	ast.USub: operator.neg,
	ast.UAdd: operator.pos,
	ast.Invert: operator.invert,
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


class _ConstFolder( ast.NodeTransformer ):
	def __init__( self, active_target: dict[str,object] ) -> None:
		self.active_target = active_target

	def visit_Attribute( self, node: ast.Attribute ) -> ast.expr:
		key = _is_compiler_target_query( node )
		if key is not None and key in self.active_target:
			return ast.copy_location( ast.Constant( value = self.active_target[key] ), node )
		self.generic_visit( node )
		return node

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
		values = [ value.value for value in node.values ]
		is_and = isinstance( node.op, ast.And )
		# Python's own short-circuit semantics: `and` returns the first falsy
		# operand (or the last one, if none are), `or` the first truthy one
		result = values[-1]
		for value in values[:-1]:
			if bool( value ) != is_and:
				result = value
				break
		return ast.copy_location( ast.Constant( value = result ), node )

	def visit_Compare( self, node: ast.Compare ) -> ast.expr:
		self.generic_visit( node )
		if len( node.ops ) != 1 or len( node.comparators ) != 1:
			return node
		fn = _CMP_FNS.get( type( node.ops[0] ))
		if fn is None or not isinstance( node.left, ast.Constant ) or not isinstance( node.comparators[0], ast.Constant ):
			return node
		try:
			value = fn( node.left.value, node.comparators[0].value )
		except TypeError:
			return node
		return ast.copy_location( ast.Constant( value = value ), node )

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
		value = node.subject.value
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

	def _bind_stmt( self, name: str, value: object, node: ast.AST ) -> ast.Assign:
		assign = ast.Assign( targets = [ ast.Name( id = name, ctx = ast.Store() ) ], value = ast.Constant( value = value ))
		return ast.copy_location( assign, node )

	def _match_pattern( self, pattern: ast.pattern, value: object ) -> tuple[bool,list[tuple[str,object]]]|None:
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


def transform_function_body( body: list[ast.stmt], active_target: dict[str,object] ) -> list[ast.stmt]:
	# wrapping in a throwaway Module lets ast.NodeTransformer's own list-field
	# splicing (visit_If/visit_While returning a list gets flattened into the
	# parent's body list) do the work, instead of reimplementing it here
	container = ast.Module( body = list( body ), type_ignores = [] )
	_ConstFolder( active_target ).generic_visit( container )
	return container.body
