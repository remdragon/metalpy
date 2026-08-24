# stdlib imports:
from dataclasses import dataclass
import itertools
import math
from typing import Callable, TypeVar as MyPyTypeVar

# local imports:
from errors import CompileError
from mpy_types import Type, Function, Parameter, ConditionalDispatch, TypeVar

T = MyPyTypeVar( 'T' )

'''
Resolves an overloaded call site to a runtime dispatch plan. Pure function
of types - no ast/ir/Discovery reference anywhere in this module - so it's
fully unit-testable on its own (see overload_resolution_test.py), matching
the same "pure function of types" spirit the code this replaces
(mpy_types.py's old Overload.resolve_call) already had.

@overload-decorated members are tried in declaration order (more specific/
earlier wins) via box subtraction: the same decomposition k-d trees use to
subtract one hyper-rectangle from another, applied per parameter slot. Each
candidate either fully or partially covers whatever's still unresolved of
the call's argument-type space; the covered part is scheduled, and the
uncovered remainder splits into independent, non-overlapping pieces that
keep going through the remaining candidates.

Plain (non-@overload) implementations are simpler by design (see TODO.txt's
own worked-through reasoning): whatever pieces survive the @overload sweep
get resolved by requiring exactly one plain implementation to bind each
concrete leaf combination - unchanged in spirit from the historical
approach, just re-scoped to the leftover pieces instead of the whole
original call, and expressed purely positionally (see _translate_indices).
'''

# Type and its subclasses (mpy_types.py) are plain, non-frozen dataclasses -
# structurally-equal and therefore unhashable. Every per-slot "set of
# possible types" here is a plain tuple, with membership/intersection/
# difference done as a linear scan against a caller-supplied `same_type`
# predicate (defaulting to raw `is`), never a real set/frozenset (which
# would silently fall back to structural equality) - mirrors mpy_types.py's
# own _leaf_is_accepted, which is identity-based BY DEFAULT for exactly the
# same reason (avoiding a wrong/expensive full structural comparison).
#
# `is` alone is not actually sufficient, though its own docstring's claim
# ("the existing dedup caches already guarantee 'same type' is the same
# object") turned out not to hold in general: a bare Specialization and its
# own EAGERLY MONOMORPHIZED form (produced by generic-parameter
# substitution, e.g. a generic function's own list[T] specialized to
# list[i32]) are two different Python objects for the identical
# instantiation - the same duality TypeResolver._same_type exists to
# handle elsewhere (_check_assignable/_unify_type_param/_coerce_into_union/
# ...). Confirmed via a real repro (PLAN_COMPILER_BUG_SWEEP.md): an
# @overload candidate's own freshly-annotated `list[i32]` parameter failed
# to match a generic-substituted `list[i32]` call-site argument, raising
# "no matching overload" for a call that should resolve cleanly.
#
# This module still can't hard-depend on TypeResolver/Monomorphizer
# directly (see the module's own "pure function of types, no Discovery
# reference" docstring, load-bearing for overload_resolution_test.py's own
# isolated unit tests) - so `same_type` is INJECTED by the caller instead,
# defaulting to plain `is` (preserving every existing unit test's behavior
# unchanged, since none of them exercise the generic-substitution
# duality). lowering.py/type_resolver.py, the two real production callers,
# pass TypeResolver._same_type.
def _identity_same_type( a: Type, b: Type ) -> bool:
	return a is b

def _contains( types: tuple[Type,...], t: Type, same_type: Callable[[Type,Type],bool] = _identity_same_type ) -> bool:
	return any( same_type( t, x ) for x in types )

def _intersect( a: tuple[Type,...], b: tuple[Type,...], same_type: Callable[[Type,Type],bool] = _identity_same_type ) -> tuple[Type,...]:
	return tuple( x for x in a if _contains( b, x, same_type ) )

def _subtract( a: tuple[Type,...], b: tuple[Type,...], same_type: Callable[[Type,Type],bool] = _identity_same_type ) -> tuple[Type,...]:
	return tuple( x for x in a if not _contains( b, x, same_type ) )

# pure insurance against the theoretical exponential blowup (each candidate
# can split a state into up to N+1 pieces, N = parameter count) - every real
# lib/ overload group is 1-2 params, 2-4 candidates, comfortably under this.
# a named constant so it's trivially adjustable if a real case ever needs more
_MAX_TRACKED_STATES = 4096

@dataclass( frozen = True, kw_only = True )
class _State:
	''' one still-unresolved slice of the call's argument-type space - one leaf-tuple per call-site slot, in the call's own slot order '''
	slots: tuple[tuple[Type,...],...]

@dataclass( kw_only = True )
class _Candidate:
	member: Function             # the stub/overload-decorated member, or a plain implementation - declares the TYPES this candidate requires
	target: Function              # what actually gets scheduled/called (member.bound_to if a stub, else member itself)
	required: tuple[tuple[Type,...],...]      # per call-slot (aligned via _translate_indices), member's own declared type's leaves
	target_params: tuple[Parameter,...]       # per call-slot, target's own Parameter - used to build runtime conditions/find the call-site operand later (lowering.py's _dispatch_operand_for_param looks a Parameter up by identity within target.parameters)
	# per call-slot, True iff member's own declared parameter type there is a
	# BARE TypeVar (e.g. `x: T` on a generic `def foo[T](x: T)`) - a genuine
	# type, but one whose only .leaves() is the TypeVar object itself, which
	# can never same_type-match a real concrete argument type (see this
	# module's own header comment). Such a slot accepts every leaf by
	# construction (lowering.py monomorphizes T from whatever the real
	# argument type turns out to be - see _lower_overload_generic_call), so
	# it's treated as a wildcard everywhere below rather than as
	# "required type: TypeVar" - the lowest-priority fallback when a concrete
	# candidate also matches, never a source of the
	# "no matching overload"/ambiguity errors on its own.
	wildcard: tuple[bool,...]

@dataclass( kw_only = True )
class _RankedMatch:
	target: Function
	conditions: list[tuple[Parameter,Type]]

def _translate_indices( member: Function, call_slots: list[int|str] ) -> tuple[int,...]|None:
	'''
	For one candidate, independently of every other candidate: which of
	member's own parameters does each of the call's slots (in the call's own
	order - positional args first, then kwargs) correspond to? Returns None
	if member can't be called with this call's shape at all (a positional
	slot beyond its own arity, a kwarg name it doesn't declare, or one of
	its own required parameters never supplied).

	Different overload members can legitimately have different parameter
	names AND a different parameter order from each other (e.g. foo(a:int,
	b:str) alongside foo(b:int, a:Bar)) - this is why the translation is
	built per-candidate, independently, rather than requiring one global
	name->position mapping shared across the whole group.
	'''
	params = member.parameters or []
	indices: list[int] = []
	covered: set[int] = set()
	for key in call_slots:
		if isinstance( key, int ):
			if key >= len( params ):
				return None
			idx: int|None = key
		else:
			idx = next( ( i for i, p in enumerate( params ) if p.stem == key ), None )
		if idx is None:
			return None
		indices.append( idx )
		covered.add( idx )
	if not all( i in covered or p.default is not None for i, p in enumerate( params ) ):
		return None
	return tuple( indices )

def stub_covers_call(
	stub: Function, call_slots: list[int|str], arg_leaves: dict[int|str,tuple[Type,...]],
	same_type: Callable[[Type,Type],bool] = _identity_same_type,
) -> bool:
	'''
	True iff `stub`'s own declared parameter types, at every one of the
	call's actual slots, fully accept every leaf the call supplies there -
	i.e. this exact call could have been dispatched to `stub` on its own.
	Used by lowering.py to decide whether a stub's more specific declared
	return type may be used to narrow a resolved call's result type: sound
	only when the call's own arguments are entirely within the stub's
	declared domain, NOT merely whenever the stub happens to be bound_to
	the resolved plain implementation (a stub is bound to exactly one
	implementation regardless of whether any given call actually matched
	the stub's own narrower signature or fell through to the
	implementation's own wider one - e.g. Result[T,E].unwrap_or's
	`default: T` stub is bound_to the plain `default: T|None = None`
	impl, but a zero-argument call only ever matches the impl's own
	broader signature, never the stub's).
	'''
	indices = _translate_indices( stub, call_slots )
	if indices is None:
		return False
	params = stub.parameters or []
	for slot, idx in zip( call_slots, indices ):
		param_type = params[idx].type
		assert param_type is not None
		required = tuple( param_type.leaves() )
		if not all( _contains( required, leaf, same_type ) for leaf in arg_leaves[slot] ):
			return False
	return True

def _build_candidates( members: list[Function], call_slots: list[int|str], targets: dict[int,Function] ) -> list[_Candidate]:
	# targets maps id(member) -> the Function actually called for that
	# member (bound_to for a stub, else the member itself) - a plain dict
	# keyed by id() since Function isn't hashable either (see the module
	# docstring's note on Type)
	candidates: list[_Candidate] = []
	for member in members:
		indices = _translate_indices( member, call_slots )
		if indices is None:
			continue # this member can't be called with this call's shape at all - excluded before any type analysis
		params = member.parameters or []
		target = targets[ id( member ) ]
		target_params_list = target.parameters or []
		param_types: list[Type] = []
		for i in indices:
			param_type = params[i].type
			assert param_type is not None
			param_types.append( param_type )
		required = tuple( tuple( pt.leaves() ) for pt in param_types )
		wildcard = tuple( isinstance( params[i].type, TypeVar ) for i in indices )
		target_params = tuple(
			target_params_list[i] if i < len( target_params_list ) else params[i]
			for i in indices
		)
		candidates.append( _Candidate( member = member, target = target, required = required, target_params = target_params, wildcard = wildcard ))
	return candidates

def _sweep(
	state: _State, required: tuple[tuple[Type,...],...],
	same_type: Callable[[Type,Type],bool] = _identity_same_type,
	wildcard: tuple[bool,...] = (),
) -> tuple[tuple[tuple[Type,...],...], tuple[bool,...], list[_State]]:
	''' standard box-subtraction sweep, one dimension (call-slot) at a time.
	Returns (matched, needs_check, misses):
	- matched[i] is state.slots[i] ∩ required[i] (state.slots[i], narrowed to what this candidate accepts at slot i)
	- needs_check[i] is True iff state.slots[i] wasn't already fully covered by required[i] (i.e. a real runtime check is needed at this slot for this state to conclude it's this candidate)
	- misses is the list of non-overlapping leftover pieces (state minus this candidate's coverage), each with the same shape as state.slots, already filtered to drop any piece that's empty at some slot (impossible)

	`wildcard[i]` (see _Candidate's own field) short-circuits slot i to a full,
	no-remainder, no-runtime-check match regardless of required[i] - a bare
	TypeVar-typed parameter accepts every leaf, so there's never anything left
	over for a later candidate to claim at that slot. Defaults to all-False
	(every existing caller/test that never passes it keeps the old behavior
	unchanged). '''
	misses: list[_State] = []
	matched: list[tuple[Type,...]] = []
	needs_check: list[bool] = []
	prefix: list[tuple[Type,...]] = []
	for i in range( len( state.slots )):
		if i < len( wildcard ) and wildcard[i]:
			matched.append( state.slots[i] )
			needs_check.append( False )
			prefix.append( state.slots[i] )
			continue
		inter = _intersect( state.slots[i], required[i], same_type )
		matched.append( inter )
		remainder = _subtract( state.slots[i], required[i], same_type )
		needs_check.append( bool( remainder ))
		if remainder:
			piece = tuple( prefix ) + ( remainder, ) + tuple( state.slots[i + 1:] )
			if all( piece ):
				misses.append( _State( slots = piece ))
		prefix.append( inter )
	return tuple( matched ), tuple( needs_check ), misses

def not_none( value: T|None ) -> T:
	assert value is not None, f'invalid {value=}'
	return value

def resolve_call(
	stubs: list[Function],
	implementations: list[Function],
	args: list[Type],
	kwargs: dict[str,Type],
	*,
	qualname: str,
	same_type: Callable[[Type,Type],bool] = _identity_same_type,
) -> tuple[list[ConditionalDispatch],Function]:
	'''
	Resolves an overloaded call site to either a single unconditional target
	(([], fn)) or an ordered list of runtime conditions to check before
	falling through to a trailing default. Raises CompileError, unrecorded
	(no Discovery/AST reference here by design) - the caller (lowering.py's
	_lower_call) records it via Discovery.fail().

	`same_type` - see _identity_same_type's own module-level comment - lets
	the real, production callers (lowering.py, type_resolver.py) pass
	TypeResolver._same_type so a generic-substituted argument type and a
	textually-identical-but-differently-resolved candidate parameter type
	are correctly recognized as the same type, without this module itself
	needing to depend on TypeResolver/Monomorphizer directly.
	'''
	for fn in ( *stubs, *implementations ):
		if fn.resolve is not None:
			fn.resolve()
	# a broken member (its own resolution already failed and was recorded
	# once, at that failure - see mpy_types.Name.broken) simply doesn't
	# participate below, exactly as if it were never defined, rather than
	# either poisoning the whole overload group or crashing later on a
	# None .parameters
	stubs = [ fn for fn in stubs if not fn.broken ]
	implementations = [ fn for fn in implementations if not fn.broken ]

	call_slots: list[int|str] = [ *range( len( args )), *kwargs.keys() ]
	arg_leaves: dict[int|str,tuple[Type,...]] = {
		**{ i: tuple( a.leaves() ) for i, a in enumerate( args ) },
		**{ name: tuple( t.leaves() ) for name, t in kwargs.items() },
	}

	# a member with any bare-TypeVar-typed parameter (see _Candidate.wildcard)
	# always sorts LAST, regardless of its own declaration line - it accepts
	# every leaf at that slot, so trying it before a concrete sibling would
	# swallow the whole remaining state via box subtraction and starve every
	# later candidate, however the user happened to order the defs in source.
	# This is what makes a generic overload behave as a genuine lowest-
	# priority fallback rather than a source-order footgun.
	def _is_wildcard_member( fn: Function ) -> bool:
		return any( isinstance( p.type, TypeVar ) for p in ( fn.parameters or [] ) )

	overload_members = sorted(
		[ *stubs, *( f for f in implementations if f.is_overload ) ],
		key = lambda f: ( _is_wildcard_member( f ), f.line ),
	)
	plains = [ f for f in implementations if not f.is_overload ]
	targets: dict[int,Function] = {
		**{ id( m ): not_none( m.bound_to if m in stubs else m ) for m in overload_members },
		**{ id( p ): not_none(p) for p in plains },
	}

	overload_candidates = _build_candidates( overload_members, call_slots, targets )
	plain_candidates = _build_candidates( plains, call_slots, targets )

	budget = _MAX_TRACKED_STATES

	def spend( n: int ) -> None:
		nonlocal budget
		budget -= n
		if budget < 0:
			raise CompileError(
				f'{qualname}: too complex to analyze (more than {_MAX_TRACKED_STATES} argument-type states tracked) - '
				f'consider narrowing the overloaded parameter types'
			)

	states = [ _State( slots = tuple( arg_leaves[key] for key in call_slots )) ]
	matches: list[_RankedMatch] = []

	for candidate in overload_candidates:
		next_states: list[_State] = []
		for state in states:
			matched, needs_check, misses = _sweep( state, candidate.required, same_type, candidate.wildcard )
			spend( len( misses ))
			next_states.extend( misses )
			if all( matched ):
				conditions: list[tuple[Parameter,Type]] = []
				for i, need in enumerate( needs_check ):
					if not need:
						continue
					leaves = matched[i]
					if len( leaves ) > 1:
						# ConditionalDispatch.conditions is a flat list of
						# (Parameter, single Type) - it has no way to express
						# "this slot's tag is one of {A,B}". Not reachable by
						# anything in lib/ today (every real overload group's
						# candidates fully pin down each slot they check) -
						# fail clearly rather than silently pick one leaf
						raise CompileError(
							f'{qualname}: conditional dispatch requiring a multi-type check on a single parameter '
							f'is not yet supported (slot {i}: {[t.qualname for t in leaves]})'
						)
					conditions.append(( candidate.target_params[i], leaves[0] ))
				matches.append( _RankedMatch( target = candidate.target, conditions = conditions ))
		states = next_states

	unresolved: list[tuple[tuple[Type,...],list[_Candidate]]] = []
	combo_targets: list[tuple[tuple[Type,...],Function]] = []
	for state in states:
		size = math.prod( len( s ) for s in state.slots ) if state.slots else 1
		spend( size )
		for combo in itertools.product( *state.slots ):
			found = [
				c for c in plain_candidates
				if all( c.wildcard[i] or _contains( c.required[i], combo[i], same_type ) for i in range( len( combo )) )
			]
			# a candidate that matched fully WITHOUT relying on any wildcard
			# slot is always preferred over one that only matched because a
			# bare TypeVar parameter accepts anything - mirrors the sweep
			# ordering above (concrete beats generic) for the plain-
			# implementation resolution path, and keeps a genuinely
			# concrete-vs-concrete ambiguity (both len > 1 with no wildcard
			# involved) reported exactly as before this change
			concrete_found = [ c for c in found if not any( c.wildcard ) ]
			if concrete_found:
				found = concrete_found
			if len( found ) != 1:
				unresolved.append(( combo, found ))
				continue
			combo_targets.append(( combo, found[0].target ))

	if unresolved:
		# combo holds each slot's own decomposed LEAF (arg_leaves - every
		# candidate is matched per-leaf, since a leaf can win independently
		# of its siblings), not the slot's own real, undecomposed argument
		# type - fine when the two agree (an ordinary, single-leaf
		# argument), but silently misleading whenever they don't: a bare
		# `Result[bytes,CodecError]` passed directly (never unwrapped) has
		# TWO leaves, Ok(bytes) and Err(CodecError) - if only the Err leaf
		# fails to match anything, the reported combo shows just
		# `codecs.CodecError` with no indication that's a decomposed
		# fragment of a whole `Result` argument, not literally what the
		# caller wrote - confirmed via a real repro (`re.compile(b)`,
		# `b: Result[bytes,CodecError]` from a forgotten `.unwrap()`)
		# reporting `('codecs.CodecError',): no matching overload`, which
		# reads as if a bare CodecError were passed positionally. Naming
		# the real slot type alongside the leaf whenever they differ turns
		# that into `(codecs.CodecError [leaf of argument 1's actual type
		# builtins.Result[bytes,codecs.CodecError]],): no matching overload`.
		original_by_slot: dict[int|str,Type] = { **{ i: a for i, a in enumerate( args ) }, **kwargs }
		def _describe_combo_slot( slot: int|str, leaf: Type ) -> str:
			original = original_by_slot[slot]
			if same_type( original, leaf ):
				return leaf.qualname
			return f'{leaf.qualname} [leaf of argument {slot!r}\'s actual type {original.qualname}]'
		parts = [
			f'{tuple( _describe_combo_slot( call_slots[i], t ) for i, t in enumerate( combo ))}: ' + (
				f'ambiguous - matches {[c.target.qualname for c in found]}' if found else 'no matching overload'
			)
			for combo, found in unresolved
		]
		raise CompileError( f'{qualname}: ' + '; '.join( parts ))

	distinct_targets: list[Function] = []
	for _, target in combo_targets:
		if not any( target is t for t in distinct_targets ):
			distinct_targets.append( target )
	for target in distinct_targets:
		own = [ c for c, t in combo_targets if t is target ]
		others = [ c for c, t in combo_targets if t is not target ]
		conditions = []
		target_candidate = next( c for c in plain_candidates if c.target is target )
		for i in range( len( call_slots )):
			own_leaves = [ c[i] for c in own ]
			if any( not same_type( leaf, own_leaves[0] ) for leaf in own_leaves[1:] ):
				continue # this slot still varies within this branch - can't be a single-value condition
			shared_leaf = own_leaves[0]
			if any( not same_type( o[i], shared_leaf ) for o in others ):
				conditions.append(( target_candidate.target_params[i], shared_leaf ))
		matches.append( _RankedMatch( target = target, conditions = conditions ))

	if not matches:
		raise CompileError( f'{qualname}: no overload matches argument types' )
	if len( matches ) == 1:
		return ( [], matches[0].target )

	default = matches[-1].target
	branches = [ ConditionalDispatch( conditions = m.conditions, function = m.target ) for m in matches[:-1] ]
	return ( branches, default )
