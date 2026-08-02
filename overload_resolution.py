# stdlib imports:
from dataclasses import dataclass
import itertools
import math

# local imports:
from errors import CompileError
from mpy_types import Type, Function, Parameter, ConditionalDispatch

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
# difference done as identity-based linear scans, never a real set/
# frozenset (which would silently fall back to structural equality) -
# mirrors mpy_types.py's own _leaf_is_accepted, which is identity-based for
# exactly the same reason.

def _contains( types: tuple[Type,...], t: Type ) -> bool:
	return any( t is x for x in types )

def _intersect( a: tuple[Type,...], b: tuple[Type,...] ) -> tuple[Type,...]:
	return tuple( x for x in a if _contains( b, x ) )

def _subtract( a: tuple[Type,...], b: tuple[Type,...] ) -> tuple[Type,...]:
	return tuple( x for x in a if not _contains( b, x ) )

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
			idx = key
		else:
			idx = next( ( i for i, p in enumerate( params ) if p.stem == key ), None )
			if idx is None:
				return None
		indices.append( idx )
		covered.add( idx )
	if not all( i in covered or p.default is not None for i, p in enumerate( params ) ):
		return None
	return tuple( indices )

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
		required = tuple( tuple( params[i].type.leaves() ) for i in indices )
		target_params = tuple(
			target_params_list[i] if i < len( target_params_list ) else params[i]
			for i in indices
		)
		candidates.append( _Candidate( member = member, target = target, required = required, target_params = target_params ))
	return candidates

def _sweep( state: _State, required: tuple[tuple[Type,...],...] ) -> tuple[tuple[tuple[Type,...],...], tuple[bool,...], list[_State]]:
	''' standard box-subtraction sweep, one dimension (call-slot) at a time.
	Returns (matched, needs_check, misses):
	- matched[i] is state.slots[i] ∩ required[i] (state.slots[i], narrowed to what this candidate accepts at slot i)
	- needs_check[i] is True iff state.slots[i] wasn't already fully covered by required[i] (i.e. a real runtime check is needed at this slot for this state to conclude it's this candidate)
	- misses is the list of non-overlapping leftover pieces (state minus this candidate's coverage), each with the same shape as state.slots, already filtered to drop any piece that's empty at some slot (impossible) '''
	misses: list[_State] = []
	matched: list[tuple[Type,...]] = []
	needs_check: list[bool] = []
	prefix: list[tuple[Type,...]] = []
	for i in range( len( state.slots )):
		inter = _intersect( state.slots[i], required[i] )
		matched.append( inter )
		remainder = _subtract( state.slots[i], required[i] )
		needs_check.append( bool( remainder ))
		if remainder:
			piece = tuple( prefix ) + ( remainder, ) + tuple( state.slots[i + 1:] )
			if all( piece ):
				misses.append( _State( slots = piece ))
		prefix.append( inter )
	return tuple( matched ), tuple( needs_check ), misses

def resolve_call(
	stubs: list[Function],
	implementations: list[Function],
	args: list[Type],
	kwargs: dict[str,Type],
	*,
	qualname: str,
) -> tuple[list[ConditionalDispatch],Function]:
	'''
	Resolves an overloaded call site to either a single unconditional target
	(([], fn)) or an ordered list of runtime conditions to check before
	falling through to a trailing default. Raises CompileError, unrecorded
	(no Discovery/AST reference here by design) - the caller (lowering.py's
	_lower_call) records it via Discovery.fail().
	'''
	for fn in ( *stubs, *implementations ):
		if fn.resolve is not None:
			fn.resolve()

	call_slots: list[int|str] = [ *range( len( args )), *kwargs.keys() ]
	arg_leaves: dict[int|str,tuple[Type,...]] = {
		**{ i: tuple( a.leaves() ) for i, a in enumerate( args ) },
		**{ name: tuple( t.leaves() ) for name, t in kwargs.items() },
	}

	overload_members = sorted( [ *stubs, *( f for f in implementations if f.is_overload ) ], key = lambda f: f.line )
	plains = [ f for f in implementations if not f.is_overload ]
	targets = {
		**{ id( m ): ( m.bound_to if m in stubs else m ) for m in overload_members },
		**{ id( p ): p for p in plains },
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
			matched, needs_check, misses = _sweep( state, candidate.required )
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
			found = [ c for c in plain_candidates if all( _contains( c.required[i], combo[i] ) for i in range( len( combo )) ) ]
			if len( found ) != 1:
				unresolved.append(( combo, found ))
				continue
			combo_targets.append(( combo, found[0].target ))

	if unresolved:
		parts = [
			f'{tuple( t.qualname for t in combo )}: ' + (
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
		conditions: list[tuple[Parameter,Type]] = []
		target_candidate = next( c for c in plain_candidates if c.target is target )
		for i in range( len( call_slots )):
			own_leaves = [ c[i] for c in own ]
			if any( leaf is not own_leaves[0] for leaf in own_leaves[1:] ):
				continue # this slot still varies within this branch - can't be a single-value condition
			shared_leaf = own_leaves[0]
			if any( o[i] is not shared_leaf for o in others ):
				conditions.append(( target_candidate.target_params[i], shared_leaf ))
		matches.append( _RankedMatch( target = target, conditions = conditions ))

	if not matches:
		raise CompileError( f'{qualname}: no overload matches argument types' )
	if len( matches ) == 1:
		return ( [], matches[0].target )

	default = matches[-1].target
	branches = [ ConditionalDispatch( conditions = m.conditions, function = m.target ) for m in matches[:-1] ]
	return ( branches, default )
