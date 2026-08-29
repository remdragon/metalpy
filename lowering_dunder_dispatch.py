# stdlib imports:
import ast

# local imports:
import arithmetic_mode
import cfg
import ir
from mpy_types import (
	Name, Type, Variable, Parameter, Function, Overload, Specialization, TaggedUnion, CStruct, CUnion, TypeVar, ConditionalDispatch, RCClass, Scalar,
)
import overload_resolution

from lowering_shared import _BINOP_DUNDER, _REFLECTED_BINOP_DUNDER, _IPLACE_BINOP_DUNDER, _MODE_DUNDER_PREFIX, _LeafPairEq, _LeafPairBinop, _dedup_types

class DunderDispatchLoweringMixin:
	''' eq/binop dunder dispatch-tree machinery for tagged unions and leaf-pair operators - mixed into FunctionLowering (lowering.py), which
	see for the shared instance state (self._instructions, self._cfg, self.lowering,
	etc.) every method here reads and writes. Never instantiated on its own;
	split out of lowering.py purely to keep that file to a manageable size - see
	lowering.py's own class docstring and FunctionLowering's base-class list for
	the full set of sibling mixins this one is composed with. '''


	def _mode_qualified_dunder_names( self, base_name: str ) -> list[str]:
		# see _MODE_DUNDER_PREFIX's own module-level comment for the full
		# rationale - the candidate dunder name(s) to try, in order, for
		# binop dispatch given the CURRENT ambient arithmetic mode
		prefix = _MODE_DUNDER_PREFIX.get( type( self._arithmetic_mode[-1] ))
		if prefix is None:
			return [ base_name ]
		return [ f'__{prefix}_{base_name.strip( "_" )}__', base_name ]

	def _emit_binop_dunder_call( self, node: 'ast.BinOp|ast.AugAssign', method: Function, receiver: ir.Operand, arg: ir.Operand, expected_type: Type|None ) -> ir.Operand:
		# shared tail for the forward/reflected dunder-call cases in
		# _lower_binop_values above - a thin, binop-shaped wrapper over
		# _emit_fallible_method_call (a two-operand call: receiver + one
		# arg), which does the real work and is reused by any OTHER call
		# needing the identical "resolve+schedule, splice-or-Call, consume
		# via ambient mode if @fallible_arithmetic" treatment (e.g. .to_T()
		# conversion dispatch - a one-operand call, no second arg)
		return self._emit_fallible_method_call( node, method, receiver, [ arg ], expected_type )

	def _emit_fallible_method_call( self, node: ast.AST, method: Function, receiver: ir.Operand|None, args: list[ir.Operand], expected_type: Type|None ) -> ir.Operand:
		# handles a plain class method (int.__add__, ...) and a
		# scalar-registered one (i32.__add__ = ..., see lib/builtins)
		# identically, since the caller's own dunder/method lookup already
		# resolved both the same way. `method.cls is None` means a genuine
		# free function was registered onto a Scalar (never had `self`
		# stripped by discovery) - same fix _lower_method_call/the general
		# _lower_call already apply: the receiver becomes a plain leading
		# positional arg instead of ir.Call.receiver.
		#
		# is_inline is checked BEFORE is_fallible_arithmetic, not instead of it: @inline
		# splicing (_lower_inline_call) never auto-consumes anything on its
		# own - arithmetic modes don't translate through a function call
		# boundary just because it happens to be inlined away (that's a
		# deliberate design choice, not a gap - see compiler.checked_add's
		# own comment). A @fallible_arithmetic+@inline method's spliced trailing return
		# (typically a compiler.checked_add/wrapped_add/saturated_add/
		# checked_convert intrinsic call) hands back its raw, real return
		# value - for is_fallible_arithmetic methods that's a genuine,
		# unconsumed Result[T,E], exactly matching the declared signature.
		# So is_fallible_arithmetic consumption below runs uniformly on
		# whatever came back, whether that value was produced by a real
		# ir.Call or by a splice - this is what makes `with compiler.
		# panic_arithmetic(...): a // b` (a, b: int) auto-panic, and
		# default-mode `a // b` auto-propagate, exactly like a bare scalar
		# `+` already does.
		# _resolve_call_target, NOT _ensure_resolved - the latter
		# unconditionally schedules its target as a real compile unit as a
		# side effect (see its own docstring), which for an @inline target
		# means compiling it as real, dead, never-called code (confirmed by
		# a real repro - see _resolve_call_target's own identical carve-out
		# and comment, already relied on by the general _lower_call path;
		# this is the same fix, needed again here since dunder-dispatch
		# resolves its own target independently rather than going through
		# that shared path)
		self.lowering._resolve_call_target( method )
		self.lowering.schedule( method.return_type )
		for p in ( method.parameters or [] ):
			self.lowering.schedule( p.type )
		call_receiver = None if method.cls is None else receiver
		call_args = ( [ receiver ] + args ) if method.cls is None else args
		if method.is_inline:
			result = self._lower_inline_call( node, method, call_receiver, call_args, {}, method.return_type, True )
			assert result is not None # want_result=True above guarantees this
		else:
			# expected_type describes the FINAL, post-consumption value (e.g.
			# `q1: int = a // b`'s expected_type is the success type `int`,
			# not the intermediate Result[int,E]) - for an is_fallible_
			# arithmetic method the dest here is that raw, unconsumed Result,
			# so it must always be typed as method.return_type exactly, never
			# expected_type. Using expected_type here silently mistyped the
			# Call's dest, which downstream Unwrap/OrJump lowering then
			# tried to treat as the wrong Result shape - a real, confirmed
			# bug (an assertion in _result_tag_data_names, whose own error-
			# message formatting then hit an unrelated circular-repr hang
			# instead of failing cleanly).
			dest_type = method.return_type if method.is_fallible_arithmetic else ( expected_type or method.return_type )
			dest = self._new_temp( dest_type )
			self._emit( ir.Call( dest = dest, target = method, receiver = call_receiver, args = call_args, kwargs = {} ))
			result = dest
		if not method.is_fallible_arithmetic:
			return result
		shape = self.lowering._type_resolver._tagged_union_shape( method.return_type )
		assert shape is not None and len( shape[1] ) == 2, f'@fallible_arithmetic {method.qualname} must declare a Result[T,E] return type'
		success_type = shape[1][0].type
		mode = self._arithmetic_mode[-1]
		extra = mode.extra if isinstance( mode, arithmetic_mode.ArithmeticPanic ) else None
		if extra is None:
			# Check mode (the default): `result` just flows out here, raw and
			# unconsumed - the general case-1 (discarded statement)/case-2
			# (flowing into a T-typed context) auto-or_throw() rule picks it
			# up downstream, exactly like checked arithmetic's own Check-mode
			# opcodes now do (see _lower_arithmetic_op). This is how //, %,
			# and fallible comparison dunders gained or_throw()/try-except
			# support, not just or_return()'s old unconditional-propagate
			# behavior.
			return result
		# Panic mode (explicit user opt-in via `with compiler.
		# panic_arithmetic(...):`) - unaffected by this whole file's auto-
		# or_throw() generalization: still an immediate Unwrap-panic, still
		# consumed eagerly right here, never left for a downstream hook.
		return self._consume_checked_result( node, result, success_type, extra )

	def _find_iplace_dunder( self, receiver_type: Type|None, op: type, arg_type: Type ) -> Function|None:
		''' __iadd__-family lookup for _stmt_AugAssign's Name/Attribute/
		Subscript target branches. Gated on is_rc_pointer(): only an
		RC-class receiver's `self` is a single shared pointer whose in-place
		mutation is externally visible after the call returns - a plain
		CStruct/Scalar receiver's `self` is passed BY VALUE (see
		_maybe_deref_arrow_receiver's own comment on receiver-passing
		conventions), so an __iadd__ found on one would silently mutate a
		throwaway copy the caller never sees. Never even worth looking there.
		No reflected variant - Python's own data model has none either. '''
		if receiver_type is None:
			return None
		receiver_type = self.lowering._ensure_resolved( receiver_type )
		if not receiver_type.is_rc_pointer():
			return None
		name = _IPLACE_BINOP_DUNDER.get( op )
		if name is None:
			return None
		return self._find_dunder_for_arg( receiver_type, name, arg_type )

	def _emit_iplace_dunder_call( self, node: ast.AugAssign, method: Function, receiver: ir.Operand, arg: ir.Operand ) -> None:
		''' Calls an __iadd__-family method purely for its mutating side
		effect - unlike Python's own convention (an in-place dunder returns
		a value the caller reassigns, typically `self`), this protocol
		requires a plain None return: every caller here only ever found this
		method because the receiver is RC (_find_iplace_dunder's own gate),
		so mutation is already visible through the shared pointer and there
		is nothing meaningful to reassign. A method found here that declares
		a non-None return is almost certainly a signature mistake (someone
		followed Python's own __iadd__ convention by habit) - fail loudly
		right here rather than silently discard the return value, which
		would otherwise mask a real mistake behind a confusing, unrelated
		error somewhere downstream. '''
		none_type = self.lowering.discovery.get_none_type()
		if method.return_type is not none_type:
			self.lowering.discovery.fail(
				f'{method.qualname} must return None to be usable as an in-place operator '
				f'(expected to mutate its receiver directly, not return a value to reassign): {ast.unparse(node)}',
				node,
			)
		self.lowering._resolve_call_target( method )
		self.lowering.schedule( method.return_type )
		for p in ( method.parameters or [] ):
			self.lowering.schedule( p.type )
		call_receiver = None if method.cls is None else receiver
		call_args = ( [ receiver, arg ] ) if method.cls is None else [ arg ]
		if method.is_inline:
			self._lower_inline_call( node, method, call_receiver, call_args, {}, None, False )
		else:
			self._emit( ir.Call( dest = None, target = method, receiver = call_receiver, args = call_args, kwargs = {} ))

	def _find_dunder_for_arg( self, owner_type: Type|None, name: str, arg_type: Type ) -> Function|None:
		''' like self.lowering._find_method, but Overload-aware: if `name`
		resolves to a real Overload group on owner_type (multiple defs
		sharing the name - e.g. int.__eq__(other: int) alongside a second
		int.__eq__(other: i32) cross-dunder overload), picks the ONE
		implementation whose single declared parameter type exactly matches
		arg_type, rather than silently treating the whole group as "no such
		method" the way a bare _find_method does (a real, confirmed gap:
		_find_method's own `isinstance(found, Function)` check returns None
		for an Overload - before this fix, adding a second int.__eq__
		overload SILENTLY broke the pre-existing int==int comparison too,
		since it fell through to comparing by raw pointer identity instead
		of calling __eq__ at all, confirmed via a real repro).

		Deliberately narrow, not a general replacement for _find_method
		everywhere: every caller here already knows the exact concrete
		argument type it wants to match against (comparison dispatch and
		binop/reflected-binop dispatch, not a call site needing real
		runtime dispatch across multiple candidate argument shapes), so a
		simple single-parameter-type scan over the Overload's own
		implementations suffices - no need for overload_resolution.py's
		own general ConditionalDispatch machinery. Used by: _lower_eq_or_ne
		and _classify_leaf_pair_eq (==/!=, both directions),
		_expr_Compare's </>/<=/>= dispatch, _lower_operand_compare, and
		_lower_binop_values' forward/reflected dunder dispatch (+-*//%|&^ and
		their __r<op>__ counterparts) - every one of these already forces
		(or already knows) the argument operand's exact type before dispatch
		is even reached, so the same "caller already knows the wanted arg
		type" precondition holds throughout. _find_method's other ~20
		remaining call sites elsewhere in this file (container-protocol
		dunders like __getitem__/__contains__/__next__, unary dispatch, ...)
		are unrelated and stay untouched - unary in particular can't
		meaningfully be Overloaded on argument type at all (no second
		operand to disambiguate against) - the rest are a separate,
		wider-scoped Overload-blindness gap, not fixed here. '''
		owner_type = self.lowering._ensure_resolved( owner_type )
		if isinstance( owner_type, ( CStruct, RCClass ) ):
			found = owner_type.chain_lookup( name )
		else:
			names = getattr( owner_type, 'names', None )
			found = names.get( name ) if isinstance( names, dict ) else None
		found = self.lowering._resolve_scalar_name( found )
		# a plain (non-Overload) Function is checked against arg_type here
		# too, NOT returned unconditionally the way a bare _find_method
		# would - a real bug caught during development: str only has ONE
		# __eq__(other: str), so this branch used to hand it back for ANY
		# arg_type (even i32), and the caller then emitted a Call passing
		# a mismatched scalar where struct builtins$str* was expected
		candidates = found.implementations if isinstance( found, Overload ) else ( [ found ] if isinstance( found, Function ) else [] )
		for impl in candidates:
			if impl.resolve is not None:
				impl.resolve()
			params = impl.parameters or []
			# a Scalar-registered dunder (impl.cls is None - a free function
			# whose receiver was never stripped by discovery, unlike a real
			# class method) declares its receiver as an ORDINARY leading
			# parameter (`def i32__add__i32(value: i32, other: i32)`), so
			# the operand to match against arg_type is params[1], not
			# params[0] - a real, confirmed bug found via a real repro
			# (`return a + b` from a function declared to return exactly
			# Result[i32,OverflowError] double-wrapped the Result, because
			# this check unconditionally required exactly ONE parameter and
			# so NEVER matched any Scalar-registered dunder at all, silently
			# falling through to the older, pre-dunder-dispatch direct-
			# opcode path below instead - which mistypes result_type as the
			# outer expected_type instead of falling back to left.type,
			# specifically when expected_type happens to already BE the
			# checked-Result shape the binop's own dunder dispatch was
			# supposed to produce). The existing test suite never caught
			# this because every existing test assigns the binop to an
			# unannotated local first (`c = a + b; return Result.Ok(c)`),
			# never returning the binop expression directly.
			arg_index = 1 if impl.cls is None else 0
			if len( params ) != arg_index + 1:
				continue
			param_type = params[arg_index].type
			# either an exact match (the ordinary case - e.g. usize against
			# a `other: usize` param), or a wildcard match against a still-
			# generic candidate's OWN type-param-typed parameter (e.g.
			# ptr_sub_dist[T]'s `other: Ptr[T]` against a concrete Ptr[i32]
			# arg_type - _same_type can't structurally match an unbound
			# TypeVar, so this is a separate, narrower check: same base,
			# and the param's own type arg is one of impl's own type_params -
			# any Ptr[whatever] arg_type counts, since T gets bound from the
			# RECEIVER below via _resolve_receiver_generic_dunder anyway,
			# which is what actually pins this parameter's concrete type).
			is_wildcard = (
				isinstance( param_type, Specialization ) and isinstance( arg_type, Specialization )
				and param_type.base is arg_type.base
				and any( isinstance( a, TypeVar ) and any( a is tv for tv in impl.type_params or [] ) for a in param_type.args )
			)
			if is_wildcard or self.lowering._type_resolver._same_type( param_type, arg_type ):
				return self.lowering._resolve_receiver_generic_dunder( impl, owner_type )
		return None

	def _peek_single_dunder_param_type( self, owner_type: Type|None, name: str ) -> Type|None:
		''' best-effort hint for a bare literal about to be lowered as a
		binary/in-place dunder's own operand - needed now that a bare
		literal's natural type is builtins.int (an RCClass), not a Scalar:
		an RCClass receiver's __add__/__iadd__/__radd__/etc almost always
		declares a concrete Scalar 'other' param (e.g. Counter.__iadd__(self,
		v: i32)), and without this hint the literal locks in as int before
		_find_dunder_for_arg/_find_iplace_dunder ever run, missing every
		such method outright (confirmed via a real repro: `v + 10` on a
		class only declaring __add__(other: i32) started failing the moment
		bare literals stopped defaulting to i32).

		Unlike _find_dunder_for_arg, this runs BEFORE the literal is
		lowered, so there's no real arg_type yet to filter an Overload
		group down with the usual "exact match" rule - instead this only
		considers each candidate's own PARAM KIND: a literal can only ever
		plausibly become a Scalar (or None/bool/str/bytes/...) anyway, never
		an arbitrary class (_expr_Constant's own literal-compatibility
		check already rejects that outright) - so candidates whose param
		isn't a Scalar are simply not real hint targets and get filtered
		out, exactly like `v + 10` needing Vector.__add__(other: i32), not
		Vector's OTHER __add__(other: Vector) overload for `v + w`. Returns
		a hint only when EXACTLY ONE Scalar-typed candidate remains -
		still-ambiguous (two Scalar-typed overloads) or no match at all
		returns None, same as a missing method, and the caller falls back
		to the literal's own natural-type inference exactly as it did
		before this hint existed. '''
		if owner_type is None:
			return None
		owner_type = self.lowering._ensure_resolved( owner_type )
		if isinstance( owner_type, ( CStruct, RCClass )):
			found = owner_type.chain_lookup( name )
		else:
			names = getattr( owner_type, 'names', None )
			found = names.get( name ) if isinstance( names, dict ) else None
		found = self.lowering._resolve_scalar_name( found )
		candidates = found.implementations if isinstance( found, Overload ) else ( [ found ] if isinstance( found, Function ) else [] )
		scalar_param_types: list[Type] = []
		for impl in candidates:
			if impl.resolve is not None:
				impl.resolve()
			params = impl.parameters or []
			# same receiver-stripping asymmetry _find_dunder_for_arg's own
			# candidate loop documents: a Scalar-registered dunder keeps its
			# receiver as an ordinary leading parameter, a real class
			# method has already had it stripped
			arg_index = 1 if impl.cls is None else 0
			if len( params ) != arg_index + 1:
				continue
			param_type = params[arg_index].type
			if isinstance( param_type, Scalar ):
				scalar_param_types.append( param_type )
		if len( scalar_param_types ) != 1:
			return None
		return scalar_param_types[0]

	def _find_indexlike_getitem( self, owner_type: Type|None, index_node: ast.expr|None = None ) -> Function|None:
		''' like self.lowering._find_method(owner_type, '__getitem__'), but
		Overload-aware for the ordinary x[i] (non-slice) subscript path -
		once a type gains a second __getitem__ overload for slice syntax
		(a compound range-descriptor argument, e.g. slice), this picks the
		leaf whose single parameter is a plain Scalar rather than silently
		treating the whole Overload group as "no such method" the way a bare
		_find_method does (same real gap _find_dunder_for_arg's own docstring
		describes, but __getitem__ is explicitly out of that helper's scope).

		Unlike _find_dunder_for_arg, this can't match against a caller-known
		concrete arg_type: an ordinary index's own type is normally INFERRED
		FROM getitem_fn's declared parameter type (a bare literal `0` needs
		that hint to pick i32/usize/whatever a given type's __getitem__
		actually declares - confirmed via a real repro, a @cstruct with
		def __getitem__(self, i: i32) that broke when this was first written
		to require an exact usize match) - so there's no arg_type to match
		against yet at the point this needs to run. Every subscript index is
		numeric regardless of which concrete Scalar a type picks, and no
		slice-descriptor argument is ever itself a bare Scalar, so "prefer
		the Scalar-typed leaf" is a structurally sound, arg-type-agnostic
		way to pick the index leaf over the slice leaf.

		A type can now ALSO declare a SECOND Scalar-typed leaf (e.g. an
		isize sibling alongside the usize leaf, for real Python-style
		negative indexing - see _resolve_index's own comment) - `index_node`,
		when given, disambiguates: a literal negative index (`x[-1]`, folded
		to a bare ast.Constant by compile_time_transformer by the time this
		runs) prefers an isize-typed leaf if one exists; every other index
		shape (a Name, a Call, a non-negative literal, ...) keeps the
		historical behavior and prefers usize - deliberately NOT probing the
		index expression's own inferred type here (that would risk double-
		lowering/double-evaluating a side-effecting index expression - the
		exact hazard this file's own comments elsewhere already flag
		repeatedly), so a variable already holding a genuinely negative
		isize still needs an explicit `.__getitem__(...)` call rather than
		bare subscript syntax - a real, narrower gap than a bare negative
		literal, left unaddressed here. '''
		owner_type = self.lowering._ensure_resolved( owner_type )
		if isinstance( owner_type, ( CStruct, RCClass )):
			found = owner_type.chain_lookup( '__getitem__' )
		else:
			names = getattr( owner_type, 'names', None )
			found = names.get( '__getitem__' ) if isinstance( names, dict ) else None
		found = self.lowering._resolve_scalar_name( found )
		if isinstance( found, Function ):
			return found
		if not isinstance( found, Overload ):
			return None
		wants_isize = (
			isinstance( index_node, ast.Constant ) and type( index_node.value ) is int
			and not isinstance( index_node.value, bool ) and index_node.value < 0
		)
		scalar_candidates: list[Function] = []
		for impl in found.implementations:
			if impl.resolve is not None:
				impl.resolve()
			params = impl.parameters or []
			arg_index = 1 if impl.cls is None else 0
			if len( params ) != arg_index + 1:
				continue
			param_type = self.lowering._ensure_resolved( params[arg_index].type )
			if isinstance( param_type, Scalar ):
				scalar_candidates.append( impl )
		if not scalar_candidates:
			return None
		def _param_stem( impl: Function ) -> str:
			p = ( impl.parameters or [] )[ 1 if impl.cls is None else 0 ]
			return self.lowering._ensure_resolved( p.type ).stem
		if wants_isize:
			for impl in scalar_candidates:
				if _param_stem( impl ) == 'isize':
					return self.lowering._resolve_receiver_generic_dunder( impl, owner_type )
		for impl in scalar_candidates:
			if _param_stem( impl ) != 'isize':
				return self.lowering._resolve_receiver_generic_dunder( impl, owner_type )
		# every candidate was isize (no plain usize leaf at all) - fall back
		# to whichever scalar leaf exists, same as the original single-
		# candidate behavior
		return self.lowering._resolve_receiver_generic_dunder( scalar_candidates[0], owner_type )

	def _find_indexlike_setitem( self, owner_type: Type|None, index_node: ast.expr|None = None ) -> Function|None:
		''' the __setitem__ counterpart of _find_indexlike_getitem - see its
		own docstring for the full reasoning (Overload-aware lookup, usize-
		vs-isize disambiguation from a literal negative index). The one
		real difference: __setitem__ takes TWO parameters (index, value),
		not one, so "the index candidate" here means the FIRST parameter is
		a plain Scalar, not "the whole parameter list is one Scalar". '''
		owner_type = self.lowering._ensure_resolved( owner_type )
		if isinstance( owner_type, ( CStruct, RCClass ) ):
			found = owner_type.chain_lookup( '__setitem__' )
		else:
			names = getattr( owner_type, 'names', None )
			found = names.get( '__setitem__' ) if isinstance( names, dict ) else None
		found = self.lowering._resolve_scalar_name( found )
		if isinstance( found, Function ):
			return found
		if not isinstance( found, Overload ):
			return None
		wants_isize = (
			isinstance( index_node, ast.Constant ) and type( index_node.value ) is int
			and not isinstance( index_node.value, bool ) and index_node.value < 0
		)
		scalar_candidates: list[Function] = []
		for impl in found.implementations:
			if impl.resolve is not None:
				impl.resolve()
			params = impl.parameters or []
			index_arg = 1 if impl.cls is None else 0
			if len( params ) <= index_arg:
				continue
			index_param_type = self.lowering._ensure_resolved( params[index_arg].type )
			if isinstance( index_param_type, Scalar ):
				scalar_candidates.append( impl )
		if not scalar_candidates:
			return None
		def _index_param_stem( impl: Function ) -> str:
			index_arg = 1 if impl.cls is None else 0
			return self.lowering._ensure_resolved( ( impl.parameters or [] )[index_arg].type ).stem
		if wants_isize:
			for impl in scalar_candidates:
				if _index_param_stem( impl ) == 'isize':
					return self.lowering._resolve_receiver_generic_dunder( impl, owner_type )
		for impl in scalar_candidates:
			if _index_param_stem( impl ) != 'isize':
				return self.lowering._resolve_receiver_generic_dunder( impl, owner_type )
		return self.lowering._resolve_receiver_generic_dunder( scalar_candidates[0], owner_type )

	def _lower_eq_or_ne( self, node: ast.Compare, left: ir.Operand, expected_type: Type|None, negate: bool ) -> ir.Operand:
		''' `==`/`!=`, for ANY left operand (scalar or not) - unlike every
		other comparison operator, Eq/NotEq are the one shape a union can
		ever meaningfully participate in (see _build_union_leaf_eq), and a
		union can show up on either side regardless of the OTHER side's own
		scalar-ness, so this doesn't share the non-scalar-left gate the rest
		of _expr_Compare's dunder dispatch still uses.
		Lowers node.comparators[0] EXACTLY ONCE (hinted toward left.type,
		non-strict - see the strict=False comment below), then tries, in
		order:
		  1. a real user __eq__/__ne__ on left's own type, when the
		     comparator's own natural type already matches left.type (the
		     ordinary/common case - unchanged behavior, byte-for-byte the
		     same dunder Call this used to emit before this method existed);
		  2. left is a union and the comparator is one of its own leaves
		     (`x == "hi"` where x: str|None - union on the LEFT);
		  3. the comparator is ITSELF a union containing left.type as a
		     member (`"hi" == x` or `None == x` - union on the RIGHT, the
		     mirror image of (2); NoneType in particular has no __eq__ of
		     its own at all, so a bare `None == x` never even reaches a
		     dunder lookup, and a bare int literal defaults to i32 - also
		     Scalar - so neither shape can be caught by gating on "left is
		     non-scalar" the way every other operator still does; only this
		     unified method, entered unconditionally for Eq/NotEq regardless
		     of left's own scalar-ness, sees both operands together and can
		     recognize the shape);
		  4. neither a matching dunder nor a recognized union - the
		     ORIGINAL pre-union-support behavior: a safe scalar widening
		     (i32->i64, ...) if one applies, else flat Cmp (pointer
		     comparison, or an ordinary same-type scalar compare) when the
		     comparator's natural type already matches left.type, or a
		     genuine type-mismatch compile error otherwise - reported via
		     _check_assignable the same way strict=True used to reject it
		     (before this method existed, that rejection - and the scalar
		     widening - ran INSIDE the right-hand _lower_expr call itself,
		     earlier than the dunder-vs-flat-Cmp fork could even be reached;
		     both are reinstated explicitly here since strict=False below
		     skips them). '''
		method_name = '__ne__' if negate else '__eq__'
		# _find_dunder_for_arg, not the plain _find_method: this is the
		# "receiver and argument end up the SAME type" fast path (right,
		# once lowered below, is hinted toward left.type when left isn't a
		# union - the common case), so the wanted implementation is
		# whichever one declares its own parameter as exactly left.type -
		# see _find_dunder_for_arg's own docstring for why a plain
		# _find_method silently breaks this once a class ever declares a
		# SECOND __eq__/__ne__ overload (e.g. int.__eq__(other: i32)
		# alongside the pre-existing int.__eq__(other: int)). NOT gated on
		# isinstance(left.type, Scalar) anymore - a Scalar-registered
		# __eq__/__ne__ (i32.__eq__ = ...; see lib/builtins/
		# __scalar_dunders.py) resolves identically here, same precedent
		# as _lower_binop_values' own arithmetic dispatch. NoneType (also
		# Scalar) has none registered, so `None == x` below still misses
		# and falls through to the union-recognition/flat-Cmp tail exactly
		# as before.
		# Result[T,E] is itself a @union - reject an unconsumed one up front,
		# on both operands, before either can reach the union-leaf dispatch
		# below and get silently decomposed into a per-leaf (T, E) compare
		# (see _lower_binop_values' own identical guard/comment)
		left = self._reject_unconsumed_result_operand( node, left )

		method = self._find_dunder_for_arg( left.type, method_name, left.type )
		left_shape = self.lowering._type_resolver._tagged_union_shape( left.type )
		# the hint handed to the comparator's own lowering below: left.type,
		# EXCEPT when left.type is ITSELF a union - hinting a plain leaf
		# comparator toward a union expected_type would trigger _coerce_or_
		# check_operand's own (unconditional, not strict-gated) union-WRAP
		# coercion, turning a bare `"hi"` into a FULLY WRAPPED str|None
		# value before this method ever gets a look at it - defeating the
		# whole point of the union checks below (right.type would already
		# equal left.type by then, both union structs, and the code would
		# fall straight through to a flat Cmp comparing two STRUCTS
		# directly - confirmed by a real regression while developing this
		# fix). None here (natural inference only) matches exactly what the
		# union-on-the-left case always did, pre-unification.
		right_hint = None if left_shape is not None else left.type
		# strict=False: skips _coerce_or_check_operand's final
		# _check_assignable rejection AND its scalar-widening coercion
		# (both gated on strict) specifically so a still-mismatched right
		# (after every OTHER, unconditional coercion - union-wrap, RCClass
		# upcast, pointer cast - already had its chance) can be checked
		# HERE, against BOTH operands' own shapes, instead of failing (or
		# silently widening) blind - both are reinstated manually below in
		# the same order _coerce_or_check_operand itself would try them.
		right = self._lower_expr( node.comparators[0], right_hint, strict = False )
		right = self._reject_unconsumed_result_operand( node, right )
		if right.type is not left.type:
			if self._is_safe_scalar_widening( right.type, left.type ):
				widened = self._new_temp( left.type )
				self._emit( ir.CastWrap( dest = widened, operand = right ))
				right = widened
			else:
				right_shape = self.lowering._type_resolver._tagged_union_shape( right.type )
				return self._lower_eq_dispatch( node, left, left_shape, right, right_shape, negate )
		if method is not None:
			# _emit_fallible_method_call, not a hand-rolled ir.Call - see
			# _expr_Compare's own identical comment on why (Scalar-
			# registered dunder receiver threading)
			return self._emit_fallible_method_call( node, method, left, [ right ], expected_type )
		if left_shape is not None:
			# right.type IS left.type (the fast-path check above), and both
			# are the exact SAME union - genuinely no dunder of its own was
			# found. A flat Cmp below would compare two STRUCTS directly (C
			# rejects this) even though both operands are already known to
			# be the identical union type - route through the general
			# dispatch instead of assuming "same type -> flat Cmp is safe",
			# which only holds for scalars/pointers, never for a TaggedUnion
			return self._lower_eq_dispatch( node, left, left_shape, right, left_shape, negate )
		# no matching dunder - a hard compile error, not a silent flat-Cmp
		# fallback - see _expr_Compare's own identical comment for the full
		# rationale (confirmed with the user: no sensible default exists for
		# comparing two arbitrary values)
		type_name = left.type.qualname if left.type is not None else '?'
		self.lowering.discovery.fail(
			f'{type_name} has no {method_name}() defined - comparison requires an explicit dunder: {ast.unparse(node)}',
			node,
		)
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		dest = self._new_temp( bool_cls )
		self._emit( ir.Cmp( dest = dest, op = ir.CmpOp.NE if negate else ir.CmpOp.EQ, left = left, right = right ))
		return dest

	def _lower_eq_dispatch(
		self, node: ast.Compare, left: ir.Operand, left_shape: tuple[TaggedUnion,list[Variable]]|None,
		right: ir.Operand, right_shape: tuple[TaggedUnion,list[Variable]]|None, negate: bool,
	) -> ir.Operand:
		''' `==`/`!=` once the ordinary fast path (a real dunder whose
		declared parameter already matches right's natural type, or a safe
		scalar widening) has already been tried and didn't apply - the general
		case, covering all three arities uniformly: neither operand is a
		union, exactly one is, or both are. A non-union operand is treated as
		a degenerate single-leaf "shape" (its own type, no tag dispatch
		needed) - this is what lets one mechanism replace what used to be two
		separate ones (a leaf-vs-union binary tag test, and a plain
		_check_assignable rejection for two ordinary non-union types), not
		just generalize a new third case alongside them.

		Builds a full (left leaf type x right leaf type) grid via
		_classify_leaf_pair_eq - EVERY cell, not just whichever member happens
		to be "the matching one": comparing a union's CURRENTLY-INACTIVE
		member against the other side is not automatically "not equal" just
		because the tag doesn't match right now - it still needs the exact
		same same-type/cross-dunder/error classification as any other pairing
		(confirmed as a real gap in the narrower predecessor of this method,
		which only ever tested T|None-shaped unions - there, the "wrong"
		member was always NoneType, so its own hardcoded "wrong tag = not
		equal" shortcut happened to coincide with the correct answer by luck,
		not by construction; a union with two non-None members would have
		gotten this wrong).

		Per-cell rule (final, confirmed): both leaves NoneType -> trivially
		equal. Exactly one is NoneType -> trivially not equal (comparing
		anything against None is always well-defined - the Optional-check
		idiom - never an error, regardless of whether the OTHER side's
		declared type actually includes None as a possible member). Same
		concrete non-None type on both sides -> the existing
		_lower_operand_compare (dunder-or-flat-Cmp), unchanged. Different
		concrete non-None types -> Python's real equality protocol: try
		left_type's own __eq__/__ne__ first, then the REFLECTED call
		(right_type's own __eq__/__ne__, receiver/arg swapped - see
		_LeafPairEq's own docstring for why this isn't the same asymmetry as
		__radd__). Neither applies -> TypeError.

		Deliberately NOT gated on whether a union is actually involved: ANY
		'error' cell makes the whole comparison's result Result[bool,
		TypeError] uniformly, even when NEITHER side is a union at all (a
		single, statically-certain mismatch, e.g. two unrelated classes) -
		the user's own call: in practice a bare `if a == b:` without
		explicitly consuming the Result still hits an ordinary "expected bool,
		got Result[...]" mismatch either way, so nothing is lost by not
		special-casing the no-union case into a bespoke, immediate compile
		error the way the narrower predecessor of this method did - and the
		user gains the ability to explicitly consume/inspect a genuine
		type-confusion at runtime if they actually want to (`(a == b).unwrap_or(...)`,
		a match, ...), using the SAME Result[T,E] machinery every other
		fallible operation already provides, no special-casing needed.

		Deliberately does NOT reuse arithmetic's/subscript's own
		_maybe_consume_result auto-`.or_return()` consumption - explicitly
		rejected (no new "compiler binop mode" concept wanted): a fallible
		comparison's Result[bool,TypeError] is simply the expression's own
		real value, returned as-is. '''
		left_types = [ m.type for m in left_shape[1] ] if left_shape is not None else [ left.type ]
		right_types = [ m.type for m in right_shape[1] ] if right_shape is not None else [ right.type ]
		method_name = '__ne__' if negate else '__eq__'
		grid = [
			[ self._classify_leaf_pair_eq( lt, rt, node, method_name ) for rt in right_types ]
			for lt in left_types
		]
		fallible = any( cell.kind == 'error' for row in grid for cell in row )
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		if fallible:
			result_cls = self.lowering.discovery.find_name( 'Result', node )
			type_error_cls = self.lowering.discovery.find_name( 'TypeError', node )
			# result_union (the monomorphized, real TaggedUnion with concrete
			# .attributes) is what _coerce_into_union needs to find Ok/Err's own
			# synthesized member constructors, AND what dest itself must be
			# typed as - unlike _emit_checked_op's own Result[T,E] (used only
			# as a Check-mode opcode's dest, never directly compared against a
			# function's own declared return type by IDENTITY), a fully-
			# concrete generic annotation like `-> Result[bool,TypeError]`
			# (every type arg already concrete, no TypeVars left to bind) gets
			# EAGERLY monomorphized by the time _current_fn.return_type is
			# read (confirmed via a real repro) - _stmt_Return's own identity
			# check then requires dest.type to be that SAME monomorphized
			# object, not the bare Specialization
			check_type = self.lowering.discovery._get_or_create_specialization( result_cls, [ bool_cls, type_error_cls ] )
			self.lowering.schedule( check_type )
			result_union = self.lowering.monomorphize_class( check_type )
			self.lowering._union_storage.get( result_union )
			dest = self._new_temp( result_union )
		else:
			result_union = None
			dest = self._new_temp( bool_cls )
		self._emit_eq_dispatch_tree( node, left, left_shape, right, right_shape, grid, dest, negate, bool_cls, result_union )
		return dest

	def _classify_leaf_pair_eq( self, left_type: Type, right_type: Type, node: ast.AST, method_name: str ) -> _LeafPairEq:
		''' one grid cell of _lower_eq_dispatch's own classification - see
		that method's docstring for the full rule. Pure type-level, no IR. '''
		none_type = self.lowering.discovery.get_none_type()
		left_is_none = left_type is none_type
		right_is_none = right_type is none_type
		if left_is_none and right_is_none:
			return _LeafPairEq( 'none_true' )
		if left_is_none or right_is_none:
			return _LeafPairEq( 'none_false' )
		if self.lowering._type_resolver._same_type( left_type, right_type ):
			return _LeafPairEq( 'same_type' )
		# _find_dunder_for_arg, not the plain _find_method - see its own
		# docstring: a class declaring TWO __eq__/__ne__ signatures (the
		# same-type one plus a genuine cross-type one, e.g. int.__eq__
		# (other: i32) alongside int.__eq__(other: int)) registers as a
		# real Overload, which a bare _find_method silently treats as "no
		# such method" - the exact shape this whole 'cross_dunder' branch
		# exists to use. NOT gated on isinstance(Scalar) anymore - same
		# precedent as everywhere else this file's comparison/binop
		# dispatch dropped that gate; in practice this cross-type branch
		# still never matches for two DIFFERENT Scalar types today (no
		# cross-type scalar comparison dunders are registered, only same-
		# type ones - see lib/builtins/__scalar_dunders.py), so dropping
		# the gate is a no-op for Scalar pairs right now, not a behavior
		# change - just no longer special-cased for no reason.
		method = self._find_dunder_for_arg( left_type, method_name, right_type )
		if method is not None:
			return _LeafPairEq( 'cross_dunder', method = method, reflected = False )
		reflected_method = self._find_dunder_for_arg( right_type, method_name, left_type )
		if reflected_method is not None:
			return _LeafPairEq( 'cross_dunder', method = reflected_method, reflected = True )
		return _LeafPairEq( 'error' )

	def _emit_eq_dispatch_tree(
		self, node: ast.Compare, left: ir.Operand, left_shape: tuple[TaggedUnion,list[Variable]]|None,
		right: ir.Operand, right_shape: tuple[TaggedUnion,list[Variable]]|None,
		grid: list[list[_LeafPairEq]], dest: ir.Temp, negate: bool, bool_cls: Type, result_union: TaggedUnion|None,
	) -> None:
		''' nested 2-level tag dispatch shared by every arity _lower_eq_
		dispatch handles - disambiguates LEFT's active member first (skipped
		entirely when left_shape is None - a non-union operand is used
		directly, no tag test, matching a real union's own "last candidate
		needs no test either" shape), then WITHIN each left branch,
		disambiguates RIGHT's the same way. Reuses the identical
		union_storage.get(base) -> (tag_attr, data_attr, payload_cls, tags)
		primitive and GetAttr(tag)+Cmp EQ+JumpIfFalse / GetAttr(data)+
		GetAttr(v_<member>) IR shape already used identically in
		_lower_dispatch_tests/_maybe_unwrap_union_arg/_match_union_member
		(type_resolver.py) - not reinvented here. Union members are never
		themselves further unions (nested unions are flattened at discovery
		time), so this recursion is bounded to exactly 2 levels regardless of
		how many members either side has. '''
		none_type = self.lowering.discovery.get_none_type()
		end_label = self._new_label( 'eq_dispatch_end' )
		if left_shape is not None:
			left_base, left_members = left_shape
			left_tag_attr, left_data_attr, left_payload_cls, left_tags = self.lowering._union_storage.get( left_base )
			n_left = len( left_members )
		else:
			n_left = 1
		if right_shape is not None:
			right_base, right_members = right_shape
			right_tag_attr, right_data_attr, right_payload_cls, right_tags = self.lowering._union_storage.get( right_base )
			n_right = len( right_members )
		else:
			n_right = 1

		for i in range( n_left ):
			is_last_left = ( i == n_left - 1 )
			if left_shape is not None:
				lm = left_members[i]
				if not is_last_left:
					next_left_label = self._new_label( 'eq_dispatch_left_next' )
					tag_dest = self._new_temp( left_tag_attr.type )
					self._emit( ir.GetAttr( dest = tag_dest, obj = left, attr = left_tag_attr.stem ))
					match = self._new_temp( bool_cls )
					self._emit( ir.Cmp( dest = match, op = ir.CmpOp.EQ, left = tag_dest, right = ir.Const( type = left_tag_attr.type, value = left_tags[lm.stem] )))
					self._emit( ir.JumpIfFalse( cond = match, target = next_left_label ))
				# _emit_leaf_pair_eq_value only ever reads narrowed_left/
				# narrowed_right for a 'same_type'/'cross_dunder' cell - an
				# 'error' cell builds a TypeError instance instead, and a
				# 'none_true'/'none_false' cell returns a bare Const, neither
				# ever touching the operand at all. Extracting the payload
				# regardless is dead code (a real -Wunused-but-set-variable,
				# confirmed suite-wide). Skip it whenever NO cell in this row
				# can ever read it.
				row_has_reader_cell = any( c.kind in ( 'same_type', 'cross_dunder' ) for c in grid[i] )
				if lm.type is none_type or not row_has_reader_cell:
					narrowed_left = left   # never read
				else:
					narrowed_left = self._extract_union_payload( left, left_data_attr, left_payload_cls, lm )
			else:
				narrowed_left = left

			for j in range( n_right ):
				is_last_right = ( j == n_right - 1 )
				if right_shape is not None:
					rm = right_members[j]
					if not is_last_right:
						next_right_label = self._new_label( 'eq_dispatch_right_next' )
						tag_dest2 = self._new_temp( right_tag_attr.type )
						self._emit( ir.GetAttr( dest = tag_dest2, obj = right, attr = right_tag_attr.stem ))
						match2 = self._new_temp( bool_cls )
						self._emit( ir.Cmp( dest = match2, op = ir.CmpOp.EQ, left = tag_dest2, right = ir.Const( type = right_tag_attr.type, value = right_tags[rm.stem] )))
						self._emit( ir.JumpIfFalse( cond = match2, target = next_right_label ))
					# same reasoning as narrowed_left above, but per-cell: this
					# one cell (i,j) is the ONLY reader of narrowed_right
					if rm.type is none_type or grid[i][j].kind not in ( 'same_type', 'cross_dunder' ):
						narrowed_right = right
					else:
						narrowed_right = self._extract_union_payload( right, right_data_attr, right_payload_cls, rm )
				else:
					narrowed_right = right

				# cell_start brackets this ONE cell's own intermediate temps
				# (error_instance, and _coerce_into_union's own Call result
				# for `value` when result_union is set) - this whole per-cell
				# block is only ONE branch of a larger dispatch tree, and the
				# enclosing statement's natural end-of-statement flush fires
				# unconditionally for EVERY temp still tracked regardless of
				# which cell actually ran at runtime (first found via a real
				# ASAN SEGV - release_object() on an uninitialized C local
				# from a cell that was never taken; then a real ASAN LEAK
				# from an earlier fix that untracked with no decref at all).
				# _flush_branch_temps below is the general form of the fix
				# this cell used to apply by hand (see _lower_binary_branch's
				# own identical use for the same reason) - flushes every
				# temp created since cell_start, keeping only dest/value
				cell_start = len( self._pending_temps )
				cell = grid[i][j]
				if cell.kind == 'error':
					value: ir.Operand = self._build_type_error_instance( node )
				else:
					value = self._emit_leaf_pair_eq_value( node, narrowed_left, narrowed_right, cell, negate, bool_cls )
				if result_union is not None:
					value = self._coerce_into_union( value, result_union, node )
				self._flush_branch_temps( cell_start, dest, value )
				self._emit( ir.Assign( dest = dest, src = value ))
				self._emit( ir.Jump( target = end_label ))
				if right_shape is not None and not is_last_right:
					self._emit( ir.Label( name = next_right_label ))
			if left_shape is not None and not is_last_left:
				self._emit( ir.Label( name = next_left_label ))
		self._emit( ir.Label( name = end_label ))
		self._cfg.fresh_temp( dest, dest.type )

	def _extract_union_payload( self, union_operand: ir.Operand, data_attr: Variable, payload_cls: CUnion, member: Variable ) -> ir.Temp:
		''' the two-GetAttr "read one member's own payload out of a union's
		data storage" shape _build_union_leaf_eq/_maybe_unwrap_union_arg both
		used to duplicate independently - factored out here since
		_emit_eq_dispatch_tree now needs it on both axes. Both dests are bare
		GetAttr reads, never fresh_temp()-registered (see _emit's own comment
		- only Call/Allocate results are), so callers need no incref/decref
		bookkeeping around the returned value. '''
		payload_dest = self._new_temp( payload_cls )
		self._emit( ir.GetAttr( dest = payload_dest, obj = union_operand, attr = data_attr.stem ))
		narrowed = self._new_temp( member.type )
		self._emit( ir.GetAttr( dest = narrowed, obj = payload_dest, attr = f'v_{member.stem}' ))
		return narrowed

	def _emit_leaf_pair_eq_value( self, node: ast.AST, narrowed_left: ir.Operand, narrowed_right: ir.Operand, cell: _LeafPairEq, negate: bool, bool_cls: Type ) -> ir.Operand:
		''' produces a plain bool operand for one grid cell whose kind isn't
		'error' (Result-wrapping, if any, is _emit_eq_dispatch_tree's own job,
		kept out of here so this stays usable for both the infallible and
		fallible codegen shapes unchanged). '''
		if cell.kind == 'none_true':
			return ir.Const( type = bool_cls, value = not negate )
		if cell.kind == 'none_false':
			return ir.Const( type = bool_cls, value = negate )
		if cell.kind == 'same_type':
			return self._lower_operand_compare( narrowed_left, narrowed_right, negate, node )
		assert cell.kind == 'cross_dunder' and cell.method is not None
		method = cell.method
		receiver, arg = ( narrowed_right, narrowed_left ) if cell.reflected else ( narrowed_left, narrowed_right )
		self.lowering._ensure_resolved( method )
		self.lowering.schedule( method.return_type )
		for p in ( method.parameters or [] ):
			self.lowering.schedule( p.type )
		dest = self._new_temp( method.return_type )
		self._emit( ir.Call( dest = dest, target = method, receiver = receiver, args = [ arg ], kwargs = {} ))
		return dest

	def _build_type_error_instance( self, node: ast.AST ) -> ir.Temp:
		''' constructs a bare TypeError() instance - the first internal
		(non-AST-driven) construction site for a trivial marker-error class
		anywhere in this file (every existing one, OverflowError() etc., is
		only ever written in real library .py source). Mirrors the general
		class-construction tail's own RCClass branch
		(_schedule_rcclass_construction + bare ir.Allocate), simplified since
		TypeError is guaranteed zero-field. dest is a fresh, Allocate-
		registered temp (see _emit's own fresh_temp() rule) - immediately
		consumed by the caller's own _coerce_into_union, whose synthesized
		member-ctor Call increfs it into the Result's Err payload; the
		ordinary end-of-scope decref of dest itself brings the refcount back
		down to the single reference the Result now owns - same fresh-temp +
		union-ctor-incref shape any ordinary Result.Err(SomeClass()) already
		uses, no new RC mechanism. '''
		type_error_cls = self.lowering.discovery.find_name( 'TypeError', node )
		self.lowering._ensure_resolved( type_error_cls )
		self.lowering.schedule( type_error_cls )
		dest = self._new_temp( type_error_cls )
		assert isinstance( type_error_cls, RCClass )
		self.lowering._schedule_rcclass_construction( type_error_cls, dest.type )
		self._emit( ir.Allocate( dest = dest, cls = type_error_cls, fields = {} ))
		return dest

	def _lower_binop_dispatch(
		self, node: 'ast.BinOp|ast.AugAssign', left: ir.Operand, left_shape: tuple[TaggedUnion,list[Variable]]|None,
		right: ir.Operand, right_shape: tuple[TaggedUnion,list[Variable]]|None,
	) -> ir.Operand:
		''' +-*//%|&^ once at least one operand is union-typed - the
		arithmetic counterpart of _lower_eq_dispatch, covering all three
		arities the same way (a non-union operand is a degenerate
		single-leaf "shape", no tag dispatch needed on that axis). Builds a
		full (left leaf x right leaf) grid via _classify_leaf_pair_binop,
		then SYNTHESIZES the expression's own result type from whatever the
		grid actually produces - unlike equality (always plain bool),
		different leaf pairs here can produce genuinely different concrete
		types (i32+i32 -> i32 vs Vector+Vector -> Vector), and not every
		call site has an expected_type to coerce into, so a fresh union of
		the DISTINCT success types is synthesized via discovery.
		_get_or_create_union - collapsing to a single plain type when every
		reachable pair happens to agree (e.g. int|i32 + int where both
		int.__add__(int) and int.__radd__(i32) return plain int - must NOT
		become a degenerate 1-member union).

		Three independent sources of fallibility all fold into ONE
		synthesized error-type union the same way: (1) a leaf pair with no
		valid operation at all -> TypeError (mirrors equality's own
		'error' cell); (2) plain scalar arithmetic's own EXISTING checked-
		arithmetic fallibility, respecting the CURRENT arithmetic mode per
		cell (_classify_leaf_pair_binop calls the exact same
		_resolve_checked_error the non-union scalar path already uses -
		wrap_arithmetic/saturate_arithmetic/panic_arithmetic are honored
		exactly as they are today, never bypassed); (3) a resolved
		dunder's own declared return type can ITSELF be Result[T,E] - its
		E folds in too, not just its T used as-is (detected via cfg.
		is_result_type + _tagged_union_shape, the same Result-shape
		detection the rest of the compiler already uses).

		Deliberately does NOT reuse the plain scalar path's own auto-
		`.or_return()` consumption (_consume_checked_result) for a
		fallible cell - same "no compiler binop modes" principle
		_lower_eq_dispatch already settled: a fallible union binop's
		Result[...] is simply the expression's own real value, returned
		as-is (see _emit_binop_fallible_split). '''
		left_types = [ m.type for m in left_shape[1] ] if left_shape is not None else [ left.type ]
		right_types = [ m.type for m in right_shape[1] ] if right_shape is not None else [ right.type ]
		method_name = _BINOP_DUNDER.get( type( node.op ))
		reflected_name = _REFLECTED_BINOP_DUNDER.get( method_name ) if method_name is not None else None
		grid = [
			[ self._classify_leaf_pair_binop( node, method_name, reflected_name, lt, rt ) for rt in right_types ]
			for lt in left_types
		]
		success_types = _dedup_types( cell.success_type for row in grid for cell in row if cell.success_type is not None )
		error_types = _dedup_types( cell.error_type for row in grid for cell in row if cell.error_type is not None )
		fallible = bool( error_types )
		success_type = success_types[0] if len( success_types ) == 1 else self.lowering.discovery._get_or_create_union( success_types )
		self.lowering.schedule( success_type )
		if isinstance( success_type, TaggedUnion ):
			self.lowering._union_storage.get( success_type )
		error_type: Type|None = None
		if fallible:
			error_type = error_types[0] if len( error_types ) == 1 else self.lowering.discovery._get_or_create_union( error_types )
			self.lowering.schedule( error_type )
			if isinstance( error_type, TaggedUnion ):
				self.lowering._union_storage.get( error_type )
			result_cls = self.lowering.discovery.find_name( 'Result', node )
			check_type = self.lowering.discovery._get_or_create_specialization( result_cls, [ success_type, error_type ] )
			self.lowering.schedule( check_type )
			result_union = self.lowering.monomorphize_class( check_type )
			self.lowering._union_storage.get( result_union )
			dest = self._new_temp( result_union )
		else:
			result_union = None
			dest = self._new_temp( success_type )
		self._emit_binop_dispatch_tree( node, left, left_shape, right, right_shape, grid, dest, success_type, error_type, result_union )
		return dest

	def _classify_leaf_pair_binop( self, node: 'ast.BinOp|ast.AugAssign', method_name: str|None, reflected_name: str|None, left_type: Type, right_type: Type ) -> _LeafPairBinop:
		''' one grid cell of _lower_binop_dispatch's own classification -
		see that method's docstring for the full rule. Dunder lookup only,
		mode-qualified exactly like the non-union path's own forward/
		reflected loop in _lower_binop_values - no isinstance(Scalar)
		branch at all: a Scalar operand's own arithmetic is registered as
		a real (if @inline, zero-overhead) dunder now (see lib/builtins/
		__scalar_dunders.py), found via _find_dunder_for_arg identically to
		any class's own method. A leaf pair that can't type-check at all
		(mismatched float types - no matching dunder is ever registered
		for that pairing, so lookup just misses; an operator with no
		dunder mapping; or neither side has a usable method) becomes an
		'error' cell (contributing TypeError) rather than an immediate
		compile failure - mirrors _classify_leaf_pair_eq's own identical
		choice: a single bad pairing doesn't reject the WHOLE union
		expression, it becomes one runtime-checkable branch of it. '''
		type_error_cls = self.lowering.discovery.find_name( 'TypeError', node )
		method: Function|None = None
		reflected = False
		if method_name is not None:
			for candidate in self._mode_qualified_dunder_names( method_name ):
				method = self._find_dunder_for_arg( left_type, candidate, right_type )
				if method is not None:
					break
			if method is None and reflected_name is not None:
				for candidate in self._mode_qualified_dunder_names( reflected_name ):
					method = self._find_dunder_for_arg( right_type, candidate, left_type )
					if method is not None:
						reflected = True
						break
		if method is None:
			return _LeafPairBinop( 'error', error_type = type_error_cls )
		# _resolve_call_target, not _ensure_resolved - the latter
		# unconditionally schedules its target as a real compile unit,
		# which for an @inline scalar-arithmetic dunder means compiling
		# it as real, dead, never-called code - same gotcha
		# _emit_fallible_method_call's own comment documents, needed
		# again here since classification resolves the method before
		# codegen ever reaches that shared call site
		self.lowering._resolve_call_target( method )
		self.lowering.schedule( method.return_type )
		for p in ( method.parameters or [] ):
			self.lowering.schedule( p.type )
		if method.is_fallible_arithmetic:
			# consumed via the ambient arithmetic mode at emission time
			# (_emit_binop_cell), exactly like this SAME dunder already
			# behaves reached from the non-union path - never folds into
			# this expression's own aggregate error union
			shape = self.lowering._type_resolver._tagged_union_shape( method.return_type )
			assert shape is not None and len( shape[1] ) == 2, f'@fallible_arithmetic {method.qualname} must declare a Result[T,E] return type'
			return _LeafPairBinop( 'dunder', success_type = shape[1][0].type, method = method, reflected = reflected )
		if cfg.is_result_type( method.return_type ):
			shape = self.lowering._type_resolver._tagged_union_shape( method.return_type )
			assert shape is not None and len( shape[1] ) == 2
			return _LeafPairBinop( 'dunder', success_type = shape[1][0].type, error_type = shape[1][1].type, method = method, reflected = reflected )
		return _LeafPairBinop( 'dunder', success_type = method.return_type, method = method, reflected = reflected )

	def _emit_binop_dispatch_tree(
		self, node: 'ast.BinOp|ast.AugAssign', left: ir.Operand, left_shape: tuple[TaggedUnion,list[Variable]]|None,
		right: ir.Operand, right_shape: tuple[TaggedUnion,list[Variable]]|None,
		grid: list[list[_LeafPairBinop]], dest: ir.Temp, success_type: Type, error_type: Type|None, result_union: TaggedUnion|None,
	) -> None:
		''' nested 2-level tag dispatch - see _emit_eq_dispatch_tree's own
		docstring, this is the identical shape (same union_storage.get /
		GetAttr+Cmp+JumpIfFalse / _extract_union_payload primitive, N-1
		members tested per axis, last is the untested default). Differs
		only in the per-cell body: a binop cell can itself be
		independently fallible (checked scalar arithmetic under the
		current mode, or a dunder declaring Result[T,E]) - see
		_emit_binop_fallible_split for that inner Ok/Err decomposition,
		reached only from cells that need it. '''
		none_type = self.lowering.discovery.get_none_type()
		end_label = self._new_label( 'binop_dispatch_end' )
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		if left_shape is not None:
			left_base, left_members = left_shape
			# not _union_storage.get(left_base) - see _emit_binop_fallible_
			# split's own comment on why a Specialization's abstract base
			# (e.g. left is itself a Result[T,E]-typed operand) needs the
			# concrete, monomorphized union instead for storage purposes
			left_concrete = self.lowering.monomorphize_class( left.type ) if isinstance( left.type, Specialization ) else left_base
			left_tag_attr, left_data_attr, left_payload_cls, left_tags = self.lowering._union_storage.get( left_concrete )
			n_left = len( left_members )
		else:
			n_left = 1
		if right_shape is not None:
			right_base, right_members = right_shape
			right_concrete = self.lowering.monomorphize_class( right.type ) if isinstance( right.type, Specialization ) else right_base
			right_tag_attr, right_data_attr, right_payload_cls, right_tags = self.lowering._union_storage.get( right_concrete )
			n_right = len( right_members )
		else:
			n_right = 1

		for i in range( n_left ):
			is_last_left = ( i == n_left - 1 )
			if left_shape is not None:
				lm = left_members[i]
				if not is_last_left:
					next_left_label = self._new_label( 'binop_dispatch_left_next' )
					tag_dest = self._new_temp( left_tag_attr.type )
					self._emit( ir.GetAttr( dest = tag_dest, obj = left, attr = left_tag_attr.stem ))
					match = self._new_temp( bool_cls )
					self._emit( ir.Cmp( dest = match, op = ir.CmpOp.EQ, left = tag_dest, right = ir.Const( type = left_tag_attr.type, value = left_tags[lm.stem] )))
					self._emit( ir.JumpIfFalse( cond = match, target = next_left_label ))
				# an 'error' cell's own codegen (_emit_binop_cell) never reads
				# narrowed_left/narrowed_right - only a TypeError instance is
				# built. Extracting the payload anyway is dead code (a real
				# -Wunused-but-set-variable, confirmed suite-wide): skip it
				# whenever NO cell in this row can ever read it, i.e. every
				# right-side pairing for this left leaf is itself 'error'.
				row_has_dunder_cell = any( c.kind != 'error' for c in grid[i] )
				narrowed_left = left if lm.type is none_type or not row_has_dunder_cell else self._extract_union_payload( left, left_data_attr, left_payload_cls, lm )
			else:
				narrowed_left = left

			for j in range( n_right ):
				is_last_right = ( j == n_right - 1 )
				if right_shape is not None:
					rm = right_members[j]
					if not is_last_right:
						next_right_label = self._new_label( 'binop_dispatch_right_next' )
						tag_dest2 = self._new_temp( right_tag_attr.type )
						self._emit( ir.GetAttr( dest = tag_dest2, obj = right, attr = right_tag_attr.stem ))
						match2 = self._new_temp( bool_cls )
						self._emit( ir.Cmp( dest = match2, op = ir.CmpOp.EQ, left = tag_dest2, right = ir.Const( type = right_tag_attr.type, value = right_tags[rm.stem] )))
						self._emit( ir.JumpIfFalse( cond = match2, target = next_right_label ))
					# same reasoning as narrowed_left above, but per-cell: this
					# one cell (i,j) is the ONLY reader of narrowed_right, so
					# an 'error' cell alone is enough to skip it
					narrowed_right = right if rm.type is none_type or grid[i][j].kind == 'error' else self._extract_union_payload( right, right_data_attr, right_payload_cls, rm )
				else:
					narrowed_right = right

				self._emit_binop_cell( node, narrowed_left, narrowed_right, grid[i][j], dest, success_type, error_type, result_union, end_label )

				if right_shape is not None and not is_last_right:
					self._emit( ir.Label( name = next_right_label ))
			if left_shape is not None and not is_last_left:
				self._emit( ir.Label( name = next_left_label ))
		self._emit( ir.Label( name = end_label ))
		self._cfg.fresh_temp( dest, dest.type )

	def _emit_binop_cell(
		self, node: 'ast.BinOp|ast.AugAssign', narrowed_left: ir.Operand, narrowed_right: ir.Operand,
		cell: _LeafPairBinop, dest: ir.Temp, success_type: Type, error_type: Type|None, result_union: TaggedUnion|None, end_label: str,
	) -> None:
		''' one grid cell's codegen - see _lower_binop_dispatch's own
		docstring for what each kind means. An 'error' cell needs a fresh
		TypeError() instance; a 'dunder' cell is emitted via the SAME
		_emit_fallible_method_call the non-union path uses (one source of
		truth for "resolve+schedule, splice-or-Call, consume via ambient
		mode if @fallible_arithmetic") - passing expected_type=None always,
		since a non-fallible-arithmetic dunder's raw return value is what
		THIS dispatch's own _coerce_binop_value/_emit_binop_fallible_split
		need to see, never pre-coerced at the call site. For an
		is_fallible_arithmetic method, _emit_fallible_method_call has
		ALREADY consumed its Result via the ambient mode by the time it
		returns here (cell.error_type is None in that case, by
		construction - see _classify_leaf_pair_binop) - so `value` is
		simply the cell's own final success value either way; only a
		cell with error_type set (a REGULAR dunder whose own declared
		return type is Result[T,E], e.g. int.__add__/Vector.__add__)
		still needs the extra Ok/Err decomposition. branch_start (snapshotted
		here, at this cell's own entry, before anything below) is threaded
		through every tail this cell can reach - _finish_binop_cell's own
		_flush_branch_temps call uses it to release every intermediate this
		ONE cell created (error_instance, _coerce_binop_value's own
		`intermediate`), same reasoning _lower_binary_branch's identical
		snapshot-then-flush already documents. Safe to reuse ONE snapshot
		across a cell's own Ok/Err/nested-unwrap sub-branches too, even
		though those are themselves mutually exclusive at runtime -
		_flush_branch_temps trims self._pending_temps back to the snapshot
		on every call, so whichever sub-branch's flush actually runs first
        (in compile-time emission order) only ever sees temps created SO
        FAR, never a later sub-branch's not-yet-emitted ones. '''
		branch_start = len( self._pending_temps )
		if cell.kind == 'error':
			error_instance = self._build_type_error_instance( node )
			assert error_type is not None and result_union is not None   # an 'error' cell always contributes TypeError, so the whole expression is always fallible whenever one exists
			value = self._coerce_binop_value( error_instance, error_type, result_union, node )
			self._finish_binop_result_branch( branch_start, error_instance, value, dest, end_label )
			return
		assert cell.kind == 'dunder' and cell.method is not None
		receiver, arg = ( narrowed_right, narrowed_left ) if cell.reflected else ( narrowed_left, narrowed_right )
		value = self._emit_fallible_method_call( node, cell.method, receiver, [ arg ], None )
		if cell.error_type is None:
			value = self._coerce_binop_value( value, success_type, result_union, node )
			self._finish_binop_cell( branch_start, dest, value, end_label )
			return
		self._emit_binop_fallible_split( node, branch_start, value, dest, success_type, error_type, result_union, end_label )

	def _coerce_binop_value( self, value: ir.Operand, axis_type: Type, result_union: TaggedUnion|None, node: ast.AST ) -> ir.Operand:
		''' two-step coercion for ONE axis (success or error) of the
		synthesized dispatch: first into that axis's own aggregate type
		(success_type/error_type - itself a TaggedUnion only when more
		than one distinct type is actually possible on this axis; a
		cell's own produced value only ever matches ONE LEAF of it, never
		axis_type itself directly, whenever axis_type genuinely is a
		union), THEN - only when the whole expression is fallible - into
		result_union's own matching Ok/Err slot (a cell's value NEVER
		already matches result_union directly: result_union's own two
		leaves are success_type/error_type as a WHOLE, never one leaf's
		own concrete type - so this second step, unlike the first, is
		never skippable once result_union is present). Callers pass
		axis_type = success_type for a success-axis value, error_type
		for an error-axis value (always non-None whenever reached, since
		producing an error value at all implies the whole expression is
		fallible).

		When BOTH steps run, the intermediate axis_type-wrapped value is
		itself a fresh, independently fresh_temp()-tracked Call result
		(same shape as every other branch-local temp this whole dispatch
		tree produces) - the second _coerce_into_union call makes its OWN
		independent embedded reference via its own ctor's incref (a
		tag-gated copy of whichever member is active, since axis_type is
		itself RC-carrying whenever this path is taken), so the
		intermediate's OWN reference is now redundant. Left tracked and
		pending here deliberately (no inline decref/untrack) - the caller's
		own _finish_binop_cell (reached via _finish_binop_result_branch or
		directly) always flushes everything created since ITS OWN
		branch_start right before returning, which correctly sweeps this up
		either way: when result_union is None `intermediate` becomes the
		return value itself (kept alive - see the `if result_union is None:
		return intermediate` branch below, matches whatever `value`
		_finish_binop_cell was called with); when result_union is set,
		`intermediate` is a genuine throwaway distinct from the SECOND
		coercion's own result, correctly released by that same flush. '''
		if isinstance( axis_type, TaggedUnion ) and not self.lowering._type_resolver._same_type( value.type, axis_type ):
			intermediate = self._coerce_into_union( value, axis_type, node )
			if result_union is None:
				return intermediate
			value = self._coerce_into_union( intermediate, result_union, node )
			return value
		if result_union is not None:
			value = self._coerce_into_union( value, result_union, node )
		return value

	def _finish_binop_cell( self, branch_start: int, dest: ir.Temp, value: ir.Operand, end_label: str ) -> None:
		# mirrors _emit_eq_dispatch_tree's own identical per-cell tail -
		# flush everything this cell (or cell sub-branch, for the fallible
		# split path) created since branch_start, keeping dest/value - see
		# _emit_binop_cell's own docstring for why ONE snapshot correctly
		# scopes every sub-branch a cell can reach, not just the top level
		self._flush_branch_temps( branch_start, dest, value )
		self._emit( ir.Assign( dest = dest, src = value ))
		self._emit( ir.Jump( target = end_label ))

	def _finish_binop_result_branch( self, branch_start: int, raw_temp: ir.Temp, value: ir.Operand, dest: ir.Temp, end_label: str ) -> None:
		''' shared tail for every branch of _emit_binop_fallible_split (and
		the 'error' cell above, whose own error_instance is the identical
		shape) - raw_temp is a branch-local, independently fresh_temp()-
		tracked value (a checked op's check_dest, a dunder Call's own
		dest, or _build_type_error_instance's own Allocate result) whose
		OWN payload `value` was just extracted from (via _coerce_into_
		union, whose synthesized ctor already increfs `value` - see
		_build_type_error_instance's own docstring for why this specific
		incref-then-decref pairing is correctly balanced). raw_temp PRE-
		DATES branch_start (it's created once, shared across every
		Ok/Err/nested-unwrap sub-branch reachable from here, released on
		exactly whichever one actually runs) - _flush_branch_temps' own
		since-a-checkpoint model can't express "release a value that
		already existed before the checkpoint", so this stays a dedicated,
		explicit decref + untrack, unlike everything created AFTER
		branch_start (which _finish_binop_cell's own flush call, below,
		handles generically). '''
		for instr in self._cfg.decref( raw_temp.type, raw_temp ):
			self._emit( instr )
		self._cfg.untrack_temp( raw_temp )
		self._finish_binop_cell( branch_start, dest, value, end_label )

	def _emit_binop_fallible_split(
		self, node: ast.AST, branch_start: int, raw_temp: ir.Temp, dest: ir.Temp, success_type: Type, error_type: Type, result_union: TaggedUnion|None, end_label: str,
	) -> None:
		''' raw_temp is a not-yet-consumed Result[T,E] value (a checked
		scalar op's own check_dest, or a dunder's own Call result whose
		declared return type is itself Result[T,E]) - decomposed via ONE
		more nested Ok/Err tag-check (bounded, always exactly 2 members -
		Result's own shape), coercing whichever branch fires into the
		OUTER dest, deliberately WITHOUT _consume_checked_result's own
		auto-`.or_return()` consumption (see _lower_binop_dispatch's own
		docstring - this fallible value IS the expression's own real
		value here, not propagated to the enclosing function). Extracted
		via _extract_union_payload, which returns a bare, un-incref'd
		BORROW (unlike every other value this whole dispatch tree
		produces, which are always already-owned fresh Call/opcode
		results) - _coerce_binop_value's own _coerce_into_union call(s)
		are what give it a real +1 reference; result_union is guaranteed
		non-None whenever this is reached (a cell only gets here when its
		own error_type is set, which is exactly what makes the WHOLE
		expression fallible). '''
		assert result_union is not None
		shape = self.lowering._type_resolver._tagged_union_shape( raw_temp.type )
		assert shape is not None and len( shape[1] ) == 2
		base, members = shape
		# NOT _union_storage.get(base): base is _tagged_union_shape's own
		# deliberately-ABSTRACT return (shared tag values across every
		# instantiation of the same generic union) - raw_temp.type here is
		# routinely a Result[T,E] SPECIALIZATION (a scalar op's own
		# check_type, or a dunder's declared Result[T,E] return type), and
		# the ABSTRACT Result class's own payload union has no real C
		# definition at all (its fields are bare, unsubstituted T/E
		# TypeVars) - confirmed via a real repro ("incomplete type 'union
		# builtins$Result$data'" from clang). union_storage needs THIS
		# instantiation's own concrete, monomorphized union instead - same
		# "monomorphize_class if Specialization else itself" pattern
		# _coerce_or_check_operand already uses for the identical reason.
		concrete_union = self.lowering.monomorphize_class( raw_temp.type ) if isinstance( raw_temp.type, Specialization ) else raw_temp.type
		tag_attr, data_attr, payload_cls, tags = self.lowering._union_storage.get( concrete_union )
		ok_member, err_member = members[0], members[1]
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		err_label = self._new_label( 'binop_result_err' )
		tag_dest = self._new_temp( tag_attr.type )
		self._emit( ir.GetAttr( dest = tag_dest, obj = raw_temp, attr = tag_attr.stem ))
		match = self._new_temp( bool_cls )
		self._emit( ir.Cmp( dest = match, op = ir.CmpOp.EQ, left = tag_dest, right = ir.Const( type = tag_attr.type, value = tags[ok_member.stem] )))
		self._emit( ir.JumpIfFalse( cond = match, target = err_label ))
		# Ok branch
		ok_payload = self._extract_union_payload( raw_temp, data_attr, payload_cls, ok_member )
		ok_value = self._coerce_binop_value( ok_payload, success_type, result_union, node )
		self._finish_binop_result_branch( branch_start, raw_temp, ok_value, dest, end_label )
		self._emit( ir.Label( name = err_label ))
		# Err branch - err_member's own type might ITSELF be a multi-member
		# ANONYMOUS union (signed Div/Mod's own ZeroDivisionError|
		# OverflowError - see _resolve_checked_error) - one more bounded
		# nested unwrap to reach a concrete leaf class before coercing
		# into the OUTER error union. Gated on file is None (the same
		# "synthesized, not a real declared type" marker
		# discovery._get_or_create_union's own flattening logic already
		# uses) - a NOMINAL @union error type (e.g. a real `@union class
		# IntError: DivideByZero: None; ...`) is just as much an opaque
		# LEAF as any plain marker class, exactly like a nominal @union
		# leaf flowing into a WIDER union elsewhere in this compiler (see
		# _coerce_into_union's own "nominal @union is exactly as valid a
		# member... as any plain leaf type" comment) - unwrapping ITS OWN
		# variants here would be wrong, confirmed via a real repro
		# (int.__add__'s own Result[int,IntError] wrongly tried to
		# decompose IntError's OWN internal DivideByZero/... variants
		# instead of treating the whole IntError value as one leaf)
		err_payload = self._extract_union_payload( raw_temp, data_attr, payload_cls, err_member )
		err_shape = self.lowering._type_resolver._tagged_union_shape( err_payload.type )
		if err_shape is not None and err_shape[0].file is None and len( err_shape[1] ) > 1:
			self._emit_nested_error_unwrap( node, branch_start, err_payload, err_shape, raw_temp, dest, error_type, result_union, end_label )
		else:
			err_value = self._coerce_binop_value( err_payload, error_type, result_union, node )
			self._finish_binop_result_branch( branch_start, raw_temp, err_value, dest, end_label )

	def _emit_nested_error_unwrap(
		self, node: ast.AST, branch_start: int, err_union_operand: ir.Operand, err_shape: tuple[TaggedUnion,list[Variable]],
		raw_temp: ir.Temp, dest: ir.Temp, error_type: Type, result_union: TaggedUnion, end_label: str,
	) -> None:
		''' one leaf pair's own checked error type can itself be a
		multi-member anonymous union (signed Div/Mod's ZeroDivisionError|
		OverflowError) - unwraps it down to the one concrete leaf class
		before coercing into the OUTER (already-flattened, individual-
		classes) error union. Bounded to exactly this one extra level -
		_resolve_checked_error never produces a nested union of unions. '''
		base, members = err_shape
		tag_attr, data_attr, payload_cls, tags = self.lowering._union_storage.get( base )
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		n = len( members )
		for i, member in enumerate( members ):
			is_last = ( i == n - 1 )
			if not is_last:
				next_label = self._new_label( 'binop_error_unwrap_next' )
				tag_dest = self._new_temp( tag_attr.type )
				self._emit( ir.GetAttr( dest = tag_dest, obj = err_union_operand, attr = tag_attr.stem ))
				match = self._new_temp( bool_cls )
				self._emit( ir.Cmp( dest = match, op = ir.CmpOp.EQ, left = tag_dest, right = ir.Const( type = tag_attr.type, value = tags[member.stem] )))
				self._emit( ir.JumpIfFalse( cond = match, target = next_label ))
			concrete = self._extract_union_payload( err_union_operand, data_attr, payload_cls, member )
			value = self._coerce_binop_value( concrete, error_type, result_union, node )
			self._finish_binop_result_branch( branch_start, raw_temp, value, dest, end_label )
			if not is_last:
				self._emit( ir.Label( name = next_label ))

	def _corresponding_leaf_param( self, reference: Function, fn: Function, ref_param: Parameter ) -> Parameter:
		''' the Parameter in `fn`'s own parameter list at the SAME POSITION
		as `ref_param` in `reference`'s - used by _lower_union_receiver_call
		to find each leaf's own declared type for an argument that was
		matched (once, against `reference` only) by _match_call_args.
		Index-based, not name-based: type_resolver.py's own _resolve_union_
		receiver_members already guarantees every leaf has the SAME
		parameter COUNT as reference, but not (yet - a real, smaller,
		separate gap, not attempted here) the same names/kinds at each
		position, so position is the only correspondence available. '''
		index = next( i for i, p in enumerate( reference.parameters ) if p is ref_param )
		return fn.parameters[index]
