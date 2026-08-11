# stdlib imports:
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

# local imports:
import ir
from errors import CompileError
from mpy_types import Type, Variable, Parameter, Function, RCClass, TaggedUnion, CUnion, Move, Copy, Specialization

'''
Ownership tracking for automatic INCREF/DECREF placement - see TODO.txt's
"CFG:" section and RC MANAGEMENT.md for the design this implements, and the
approved plan (declarative-greeting-parnas.md) for how it was arrived at.

This module is a pure state machine: it never touches ast/Discovery, and it
never emits IR by itself - lowering.py calls into it at the exact points it
already visits things (prologue, each Assign, each Call, IF/ELSE/ENDIF,
loop bodies, break/continue/return, DeleteTemp, del), and each call returns
the list[ir.Instruction] lowering.py should emit at that point (or raises
CompileError, which lowering.py records via discovery.fail exactly like
overload_resolution.py's contract). Three small callbacks (new_temp,
new_label, union_storage) are all lowering.py needs to hand over for cfg.py
to build real IR (temps/labels/union tag-check sequences) without importing
ast/Discovery itself - see CFGState.__init__.

Only top-level bindings (params/locals/self) are tracked via the full
epilogue-stack machinery below; temps use a simpler, separate mechanism
(see _temp_states) since their lifetime is already exactly bracketed by
DeclareTemp/DeleteTemp - no stack/merge logic needed for them. v1
deliberately doesn't reach into struct/union FIELDS (Result[str,E]'s
payload, etc) - a binding whose type has no RC leaves at all (CStruct/
CUnion/CEnum/Scalar, or a TaggedUnion with no RC leaves) is invisible here.

Scope is determined structurally, not via any reference scan (see the
plan): a binding assigned while already tracked (already in .bindings) is
a REPLACE (decref old, no new epilogue entry - see assign()); a binding
that's fresh gets a new epilogue entry, popped either at its own block's
natural end (if/loop - see merge_if/loop_back_edge) or at function exit
(return_). Referencing a binding that's been torn down early (moved,
deleted, or confined to a branch that didn't survive to an if's merge)
is caught the same way any undefined name already is - lowering.py removes
it from fn.names at exactly the same point cfg.py tears down its entry.
'''

class OwnState( Enum ):
	OWNED = 'owned'
	BORROWED = 'borrowed'
	COPY = 'copy'
	MOVED = 'moved'

def is_rc( t: Type ) -> bool:
	# a concrete generic RCClass instantiation (Box[i32]) is a Specialization,
	# not an RCClass instance itself - unwrap first, or every generic-class/
	# generic-union instance method's own `self` (already typed as a
	# Specialization) would wrongly look untracked here
	base = t.base if isinstance( t, Specialization ) else t
	return isinstance( base, RCClass )

def is_result_type( t: Type|None ) -> bool:
	''' True when `t` is a concrete Result[T,E] specialization. '''
	if t is None:
		return False
	base = t.base if isinstance( t, Specialization ) else t
	return isinstance( base, TaggedUnion ) and base.stem == 'Result'

def rc_leaves( t: Type ) -> list[Type]:
	# a TaggedUnion's RC-relevant leaves specifically - str|i32 needs a
	# tag-gated incref (only str); str|int (both RC) needs none of that,
	# unconditional instead
	base = t.base if isinstance( t, Specialization ) else t
	if isinstance( base, TaggedUnion ):
		return [ leaf for leaf in base.leaves() if is_rc( leaf ) ]
	return [ t ] if is_rc( t ) else []

UnionStorage = Callable[[TaggedUnion], tuple[Variable,Variable,CUnion,dict[str,int]]]

@dataclass
class Epilogue:
	''' one pending cleanup action - either a defer/errdefer replay (flag
	set) or a named RC binding's Decref (flag None, fully compile-time-
	precise thanks to definite assignment - see the plan's "Flags" section).
	Stays in _epilogue_stack from the moment it's pushed until the owning
	scope actually unwinds; `cancelled` lets a MOVE/replace/del neutralize
	it from anywhere in the stack (not just the top) without disturbing the
	list's order/length, which is what lets every block's own recorded
	depth stay a stable, plain integer even though entries below the top
	can be cancelled at arbitrary points - see move()/deleted(). '''
	instructions: list[ir.Instruction] # defer/errdefer entries only (flag is not None) - the already-lowered replay body, reused as-is (safe: _replay()'s flag-guarded path mints a fresh skip-label on every call). Plain RC entries leave this empty and use `type` below instead
	name: str # this entry's own jump target - see current_epilogue_label()/build_epilogue_ladder()
	operand: Variable | None = None # the RC binding this entry decrefs - None for defer/errdefer entries. Lets return_() skip decref'ing whatever's actually being returned, by identity
	type: Type | None = None # plain RC entries only - the type to decref `operand` as. Instructions are regenerated fresh from this on every replay (_replay()/unwind_to()) rather than cached: a loop-confined entry can be replayed at more than one emission point (an early return inside the loop while it's still the topmost active entry, or several break/continue in the same loop) before it's ever dropped by restore(), and a cached instruction list would bake in the same tag-gated-decref Label names at every one of those sites - confirmed by a real repro, "redefinition of label" from clang on a loop with two early-return Result checks in a row
	flag: Variable | None = None
	is_err_only: bool = False # errdefer vs plain defer - only meaningful when flag is set
	cancelled: bool = False

	@property
	def is_flag_guarded( self ) -> bool:
		return self.flag is not None

@dataclass
class _Binding:
	operand: Variable # the real Variable/Parameter - never reconstructed, so identity always matches what lowering.py already has
	type: Type # the RC-relevant type - operand.type itself for a plain binding, but the INNER type when operand.type is Move/Copy (the wrapper is compile-time-only bookkeeping, never the runtime value's real type)
	state: OwnState
	entry: Epilogue | None # None only for BORROWED (never needs cleanup)

Bindings = dict[str,_Binding]

@dataclass
class _Snapshot:
	''' captured by snapshot(), consumed by restore() - see the IF/loop
	orchestration lowering.py performs around branches/loop bodies. '''
	bindings: Bindings
	stack_depth: int
	results: set[str]

class CFGState:
	''' one instance per function being lowered. `bindings` is public and
	directly snapshot/restore-able (`dict(state.bindings)` / assignment) -
	lowering.py owns sequencing (when to snapshot before a branch, when to
	restore for `else`), cfg.py only computes what a given transition or
	merge means. '''

	def __init__(
		self,
		fn: Function,
		*,
		bool_type: Type,
		new_temp: Callable[[Type], ir.Temp],
		new_label: Callable[[str], str],
		union_storage: UnionStorage,
	) -> None:
		self.fn = fn
		self._bool_type = bool_type
		self._new_temp = new_temp
		self._new_label = new_label
		self._union_storage = union_storage
		self._epilogue_stack: list[Epilogue] = []
		self._loop_entry_depths: list[int] = [] # see enter_loop()/exit_loop()
		self.bindings: Bindings = {}
		self._unchecked_results: set[str] = set() # names of locals currently holding a Result[T,E] that hasn't been is_ok()/is_err()/or_return()/unwrap()/unwrap_or()'d or match'd yet - independent of RC tracking above, see track_result()/clear_result()
		self._temp_states: dict[int,Type] = {} # ir.Temp.id -> its type, only while OWNED (temps are never BORROWED/COPY/MOVED)
		self.prologue_instructions: list[ir.Instruction] = []
		self._construction_self: Variable | None = None # set by enter_construction() - which self param (if any) is still under construction
		self._construction_required: list[Variable] = [] # __init__'s own attributes that must all be initialized before self can escape/construction can complete
		for param in fn.parameters or []:
			self._enter_parameter( param )

	# --- prologue --------------------------------------------------------------

	def _enter_parameter( self, param: Parameter ) -> None:
		if isinstance( param.type, Move ):
			# the callee now fully owns the incoming reference - MOVED is
			# the CALLER's state at the call site, not the callee's own
			# parameter (see prerequisite #2's move() call-site check,
			# which is what guarantees this parameter really was moved in)
			if rc_leaves( param.type.inner ):
				self._push( param, param.type.inner, OwnState.OWNED )
		elif isinstance( param.type, Copy ):
			# the callee wants its own independent reference - an explicit
			# Incref right here in the prologue, matching Decref at exit
			if rc_leaves( param.type.inner ):
				self.prologue_instructions += self._incref_instructions( param.type.inner, param )
				self._push( param, param.type.inner, OwnState.COPY )
		elif rc_leaves( param.type ):
			self.bindings[param.stem] = _Binding( operand = param, type = param.type, state = OwnState.BORROWED, entry = None )

	def enter_self( self, self_param: Variable, *, is_move: bool ) -> None:
		''' `self` is excluded from fn.parameters entirely (see discovery.py's
		_make_function_resolver) and only synthesized by lowering.py once
		it starts lowering a method body - called separately from __init__
		for exactly that reason. A @move-decorated method's self starts
		OWNED (same reasoning as a move[T] parameter - confirmed by tracing
		bytearray.release(), which never manually frees self; an ordinary
		Decref at its own exit, possibly invoking __del__, is correct
		because __del__ already guards the double-free via the
		BYTEARRAY_INVALID sentinel, unrelated to this). Otherwise BORROWED,
		like any other plain parameter - self is never copy[T]. '''
		if not rc_leaves( self_param.type ):
			return
		if is_move:
			self._push( self_param, self_param.type, OwnState.OWNED )
		else:
			self.bindings[self_param.stem] = _Binding( operand = self_param, type = self_param.type, state = OwnState.BORROWED, entry = None )

	def enter_construction( self, self_param: Variable, required: list[Variable] ) -> None:
		''' called instead of enter_self() when lowering __init__
		specifically (see RCCLASS ATTRIBUTE LIFETIME.md and the approved
		plan) - self is never @move for __init__ (construction transfers
		each ATTRIBUTE's ownership into self as it's set, not self's own
		identity), so this is just enter_self(is_move=False) plus recording
		`required` (the class's own declared attributes - not yet base-
		class-aware, see the plan's "forward-compatibility with
		subclassing" note) for check_self_escape()/complete_construction()
		to consult. Every attribute starts UNTRACKED (absent from bindings)
		until its own first attr_assign() - exactly like an unassigned
		local, no separate "uninitialized" state needed. '''
		self.enter_self( self_param, is_move = False )
		self._construction_self = self_param
		self._construction_required = required

	def _push( self, operand: Variable, type_for_decref: Type, state: OwnState, *, key: str | None = None ) -> Epilogue:
		entry = Epilogue( instructions = [], name = self._new_label( 'epilogue' ), operand = operand, type = type_for_decref )
		self._epilogue_stack.append( entry )
		self.bindings[key if key is not None else operand.stem] = _Binding( operand = operand, type = type_for_decref, state = state, entry = entry )
		return entry

	def push_defer( self, instructions: list[ir.Instruction], flag: Variable, is_err_only: bool ) -> None:
		''' defer/errdefer's own replay, registered at the defer/errdefer
		statement's own position (lowering.py emits the flag's own `= True`
		Assign right after this call - that's what "armed" means at
		runtime). Interleaved into the SAME _epilogue_stack RC bindings use,
		by declaration order - build_epilogue_ladder() replays the whole
		stack together, deepest first. Never touches self.bindings (there's
		no name to look it up by - it's not a variable), so it's immune to
		merge_if()'s dict-based reconciliation entirely; restore() below
		gives it the different treatment it actually needs instead. '''
		self._epilogue_stack.append( Epilogue(
			instructions = instructions, name = self._new_label( 'epilogue' ), flag = flag, is_err_only = is_err_only,
		))

	# --- snapshot/restore, for IF/loop orchestration ----------------------------

	def snapshot( self ) -> _Snapshot:
		return _Snapshot( bindings = dict( self.bindings ), stack_depth = len( self._epilogue_stack ), results = set( self._unchecked_results ))

	def restore( self, snap: _Snapshot ) -> None:
		''' truncates back to the snapshot's own depth for ordinary (RC)
		entries - an if-branch's own locals are genuinely block-scoped, torn
		down at the branch's own exit (merge_if handles that). A defer/
		errdefer entry pushed since the snapshot is different: defer's
		cleanup always runs at the FUNCTION's own shared epilogue, no matter
		which branch (if any) armed it - the flag alone decides whether it
		actually replays - so it has to survive this truncation instead of
		being discarded with the branch's own locals. _unchecked_results is
		reverted the same way bindings is - unconditionally, even for a loop
		body that provably checked something inside it: a while-loop's body
		may run zero times, so anything only checked INSIDE the body can't be
		assumed checked once we're back outside it (see loop_back_edge()). '''
		self.bindings = dict( snap.bindings )
		self._unchecked_results = set( snap.results )
		survivors = [ e for e in self._epilogue_stack[snap.stack_depth:] if e.is_flag_guarded ]
		del self._epilogue_stack[snap.stack_depth:]
		self._epilogue_stack += survivors

	def enter_loop( self, stack_depth: int ) -> None:
		''' called by lowering.py's own _lower_loop_body, bracketing one
		loop body's lowering - stack_depth is the SAME loop_snapshot.
		stack_depth already taken just before it (see restore()'s own
		"an if-branch's own locals are genuinely block-scoped" comment;
		a loop body's locals are exactly as block-scoped, torn down by
		restore() once the body's been lowered, one iteration's worth,
		exactly once - regardless of how many times it actually runs at
		runtime). Tracked here (not just left to lowering.py) so
		current_epilogue_label() can tell "is the topmost live entry
		confined to a loop I'm still inside lowering" without every one of
		its many call sites threading loop context through by hand. '''
		self._loop_entry_depths.append( stack_depth )

	def exit_loop( self ) -> None:
		self._loop_entry_depths.pop()

	# --- unchecked Result tracking ----------------------------------------

	def track_result( self, name: str ) -> None:
		''' called whenever a Result[T,E]-typed value is bound to a local
		(see assign()) - `name` now owes an inspection before it can be
		overwritten, del'd, or the function can exit while it's still live.
		Independent of RC tracking entirely - unlike self.bindings, this
		applies even to a Result whose T/E are both plain scalars/enums (no
		RC leaves at all), which is the common shape. '''
		self._unchecked_results.add( name )

	def clear_result( self, name: str ) -> None:
		''' called wherever a Result binding gets genuinely inspected -
		.is_ok()/.is_err()/.or_return()/.unwrap(msg)/.unwrap_or(default), or
		being the subject of a match. A plain discard - safe to call on a
		name that was never tracked (not a Result, or already checked). '''
		self._unchecked_results.discard( name )

	def is_unchecked( self, name: str ) -> bool:
		return name in self._unchecked_results

	def unchecked_results( self ) -> set[str]:
		''' a defensive copy for lowering.py to snapshot alongside bindings
		around if/loop orchestration (see merge_if()/loop_back_edge()) -
		mirrors dict(self.bindings) being taken at the same call sites. '''
		return set( self._unchecked_results )

	def check_unchecked_results( self, returned_operand: ir.Operand | None ) -> None:
		''' called at every real function-exit point: each `return`
		statement, the function's own fall-off-the-end, and or_return()'s/
		checked-arithmetic's own early-return-on-Err path (see lowering.py's
		_consume_checked_result) - raises if anything is still owed.
		Deliberately NOT folded into return_() itself: most returns route
		through current_epilogue_label()'s shared-ladder Jump instead of
		calling return_() inline (see its own docstring), so a check placed
		only inside return_() would silently skip every return that also has
		an RC decref or defer/errdefer pending - a very common combination.
		returned_operand is excluded by name (not identity - a returned
		Result has no epilogue entry to compare against) when it's a
		Variable: `return r` transfers the obligation to the CALLER, it's
		not this function's to discharge (v1 scope cut, see the plan).

		Always clears _unchecked_results entirely before returning OR
		raising - unlike return_() on the RC side, which deliberately
		leaves self.bindings untouched (no code follows a return on that
		path, so nothing needs it to look any particular way). This one
		DOES need to mutate, unconditionally: it's called at points other
		than a genuine, unnested, function-ending return too - a `return`
		inside a loop body or an if-branch leaves the rest of that lowering
		pass (loop_back_edge()'s own fresh-in-loop check, or this same
		method called again later at the function's own fall-off point)
		still looking at whatever was live right before the return. Without
		clearing on the RAISE path too, the fall-off point's own call would
		re-discover the exact same already-reported problem and raise a
		SECOND time - this one uncaught (it runs after lower_function's own
		per-statement recovery loop has already finished), crashing the
		compiler instead of just recording one clean error. '''
		excluded = returned_operand.stem if isinstance( returned_operand, Variable ) else None
		remaining = self._unchecked_results - ( { excluded } if excluded is not None else set() )
		self._unchecked_results = set()
		if remaining:
			names = ', '.join( repr( n ) for n in sorted( remaining ))
			plural = len( remaining ) > 1
			raise CompileError(
				f"Result value{'s' if plural else ''} {names} {'were' if plural else 'was'} never inspected - "
				f"use .is_ok(), .is_err(), .or_return(), .unwrap(msg), or match"
			)

	# --- IF/ELSE/ENDIF -----------------------------------------------------

	def merge_if(
		self, entry_bindings: Bindings, true_end: Bindings, false_end: Bindings, ctx: str,
		*,
		entry_results: set[str] = frozenset(), true_end_results: set[str] = frozenset(), false_end_results: set[str] = frozenset(),
		true_terminates: bool = False, false_terminates: bool = False,
	) -> tuple[list[ir.Instruction],list[ir.Instruction],list[str]]:
		''' called after lowering.py has already restore()'d back to the
		if's own entry snapshot (so self.bindings/self._epilogue_stack are
		clean of whatever either branch speculatively pushed) - compares
		the two branches' own ending snapshots (false_end is just
		entry_bindings again if there was no `else`) and either raises
		CompileError (a binding in an indeterminate state - foo1) or
		returns (true-branch-only instructions, false-branch-only
		instructions, names to remove from fn.names). The two instruction
		lists are returned SEPARATELY, not combined, and MUST be spliced
		into that one branch's own captured code (before its own exit to
		the join point) - a binding confined to one branch only exists on
		that one path, so its teardown can't run at the shared join point
		reached by both. Re-establishes exactly one epilogue entry per
		surviving OWNED/COPY binding - both branches always push their OWN
		entry when creating the same-named binding fresh, and only one of
		the two ever actually runs, so those speculative entries must
		never both survive onto the real stack.

		true_terminates/false_terminates (return/break/continue as that
		branch's own last statement - lowering.py's call site decides)
		mark a branch that never reaches the join point at all. This
		matters because return_()/unwind_to() deliberately don't mutate
		bindings (each exit is independent, no code follows it on that
		path) - so a terminating branch's OWN _end snapshot still looks
		like a perfectly ordinary, still-live set of bindings, exactly as
		if it had fallen through. Comparing it against the other branch
		here would either raise a bogus indeterminate-state error, or
		(worse, silently) re-push a duplicate epilogue entry for a binding
		that the other, non-terminating branch also still owns - a real
		double-decref at whatever exit runs next. When exactly one branch
		terminates, only the survivor's own _end state can possibly reach
		the join, so it wins outright, no comparison/error-checking
		needed - there's only one live path down to here. When both
		terminate, nothing reaches the join (any code after the if is
		unreachable - full dead-code detection is future work, see
		TODO.txt), so nothing survives either.

		A survivor whose entry is IDENTICAL (by object identity) to
		entry_bindings' own entry for that name is already correctly
		sitting on the stack - lowering.py already restore()'d back to
		entry_snapshot before calling this, and restore() only truncates
		the stack/resets the bindings dict, it never touches the Epilogue
		objects already-live entries point to. assign()'s own discipline
		(reuse the same entry for as long as a name stays continuously
		tracked, even across a move-then-reassign's cancel/uncancel cycle
		- only push genuinely fresh when the name wasn't tracked at all
		beforehand) guarantees that identity check is reliable: pushing
		AGAIN here for an unchanged/merely-replaced survivor would give it
		a second, duplicate live entry - the same double-decref bug as the
		terminates case above, just via the ordinary two-branch path
		instead (an if/else where a local declared before it survives
		untouched, the single most common shape there is). Only a name
		that's genuinely new to the stack (never in entry_bindings, or
		re-pushed after being del'd and reassigned) needs a real push.

		entry_results/true_end_results/false_end_results are the unchecked-
		Result analogue of entry_bindings/true_end/false_end - captured by
		lowering.py via unchecked_results() at the same three points it
		already captures dict(self.bindings) - and are reconciled by
		_merge_results() below, mutating self._unchecked_results directly
		(mirroring reestablish()'s direct self.bindings mutation above). Kept
		as a separate parallel set rather than folded into Bindings/_Binding
		because a Result[i32,SomeEnum] has no RC leaves and so has no entry
		in Bindings at all (see assign()'s own comment) - Bindings is an
		RC-only mechanism by design. '''
		true_instructions: list[ir.Instruction] = []
		false_instructions: list[ir.Instruction] = []
		removed: list[str] = []

		def reestablish( name: str, binding: _Binding, already_live: bool ) -> None:
			if binding.state in ( OwnState.OWNED, OwnState.COPY ):
				if already_live:
					self.bindings[name] = binding
				else:
					self._push( binding.operand, binding.type, binding.state )
			else:
				self.bindings[name] = _Binding( operand = binding.operand, type = binding.type, state = binding.state, entry = None )

		if true_terminates or false_terminates:
			survivor = None
			survivor_results = None
			if true_terminates and not false_terminates:
				survivor = false_end
				survivor_results = false_end_results
			elif false_terminates and not true_terminates:
				survivor = true_end
				survivor_results = true_end_results
			if survivor is not None:
				for name, binding in survivor.items():
					prior = entry_bindings.get( name )
					already_live = prior is not None and prior.entry is binding.entry
					reestablish( name, binding, already_live )
			# both terminate -> nothing reaches the join at all (dead code
			# past here, same reasoning as the RC side above) - empty is the
			# safe choice; one terminates -> only the survivor's own results
			# state can possibly reach the join
			self._unchecked_results = set( survivor_results ) if survivor_results is not None else set()
			return true_instructions, false_instructions, removed
		for name in set( true_end ) | set( false_end ):
			in_true = name in true_end
			in_false = name in false_end
			if in_true and in_false:
				if true_end[name].state != false_end[name].state:
					raise CompileError(
						f"{ctx}: {name!r} is in an indeterminate state after the if - "
						f"{true_end[name].state.value} on one branch, {false_end[name].state.value} on the other"
					)
				prior = entry_bindings.get( name )
				already_live = (
					prior is not None
					and prior.entry is true_end[name].entry
					and prior.entry is false_end[name].entry
				)
				reestablish( name, true_end[name], already_live )
				continue
			if name in entry_bindings:
				raise CompileError(
					f"{ctx}: {name!r} exists on only one branch of the if, but was already defined before it - "
					f"both branches must leave it in the same state"
				)
			# fresh on exactly one branch, never existed before the if -
			# fine (per your clarification: confined to that branch, no
			# matching assignment needed on the other) - tear it down
			# inside THAT branch's own code only
			binding = true_end[name] if in_true else false_end[name]
			decref = self._decref_instructions( binding.type, binding.operand ) if binding.state in ( OwnState.OWNED, OwnState.COPY ) else []
			if in_true:
				true_instructions += decref
			else:
				false_instructions += decref
			removed.append( name )
		self._merge_results( entry_results, true_end_results, false_end_results, ctx )
		return true_instructions, false_instructions, removed

	def _merge_results( self, entry_results: set[str], true_end_results: set[str], false_end_results: set[str], ctx: str ) -> None:
		''' the unchecked-Result analogue of merge_if()'s own binding
		reconciliation, for the neither-branch-terminates case (the
		terminates case is handled directly in merge_if() - only the
		survivor's own results state matters there, same as for bindings).
		Unlike OwnState (4 possible states, needs an equality check),
		"unchecked" is a plain presence/absence - the only two interesting
		outcomes per name are "still unchecked on both branches" (stays
		live) and "unchecked on exactly one branch" (an error, in both
		flavors: a pre-existing Result checked on only one side, per the
		plan's own validation table, and a Result introduced fresh inside
		just one branch and left unchecked at that branch's own join point -
		NOT in the original table, but required by the feature's own goal:
		that branch-confined binding is about to go out of scope right here,
		same as reaching `return` while unchecked. '''
		merged: set[str] = set()
		for name in true_end_results | false_end_results:
			in_true = name in true_end_results
			in_false = name in false_end_results
			if in_true and in_false:
				merged.add( name )
				continue
			if name in entry_results:
				raise CompileError( f"{ctx}: Result {name!r} was inspected on one branch but not the other" )
			raise CompileError(
				f"{ctx}: Result value {name!r} was never inspected before going out of scope at the end of its branch - "
				f"use .is_ok(), .is_err(), .or_return(), .unwrap(msg), or match"
			)
		self._unchecked_results = merged

	# --- loops ---------------------------------------------------------------

	def loop_back_edge( self, entry_bindings: Bindings, ctx: str, *, entry_results: set[str] | None = None ) -> list[ir.Instruction]:
		''' called after lowering the loop body once (hooks mutated
		self.bindings/self._epilogue_stack live throughout) - compares
		entry_bindings (snapshot from before the body) against the current
		self.bindings (the state after one iteration, i.e. "the back
		edge"). A binding that exists in both with the SAME state is
		stable (loops forever without further change - foo4's s2, or any
		pre-existing outer binding untouched or replaced inside the body);
		a mismatch is a compile error (foo3). A binding that's fresh
		during the body and absent from entry_bindings is confined to the
		loop - torn down every iteration, instructions returned to emit
		right before the Jump back to the loop's start. Caller still needs
		to restore() back to the entry snapshot afterward (this method
		doesn't mutate state itself, matching merge_if's split of
		responsibilities).

		entry_results (unchecked-Result analogue of entry_bindings) is
		checked for exactly one thing, deliberately NOT the full stability
		check bindings get above: a Result name absent from entry_results
		but present in self._unchecked_results now (i.e. produced fresh
		SOMEWHERE inside the loop body and still unchecked at the back edge)
		is a real bug - every iteration silently overwrites the previous
		one's unchecked Result via the same, once-lowered assign. The
		REVERSE direction (unchecked entering, checked somewhere in the
		body, never reassigned after) is deliberately NOT flagged: unlike
		OwnState, "checked" has no runtime representation and no generated
		code depends on it being stable across iterations - the only actual
		soundness concern is restore()'s own job (a while-loop may run zero
		times, so nothing checked only inside the body can be assumed
		checked once back outside it), not this method's. break/continue
		early exits aren't covered here either (v1 scope cut - unlike
		bindings, which unwind_to() handles for RC purposes, no equivalent
		exists yet for Results; a Result assigned earlier in a loop body and
		discarded via an early continue/break before ever being checked
		currently slips through uncaught). '''
		back_edge = self.bindings
		instructions: list[ir.Instruction] = []
		for name in set( entry_bindings ) | set( back_edge ):
			in_entry = name in entry_bindings
			in_back = name in back_edge
			if in_entry and in_back:
				if entry_bindings[name].state != back_edge[name].state:
					raise CompileError(
						f"{ctx}: {name!r} is in an indeterminate state across loop iterations - "
						f"{entry_bindings[name].state.value} entering the loop body, {back_edge[name].state.value} by the end of one iteration"
					)
				continue
			if in_entry:
				raise CompileError( f"{ctx}: {name!r} does not exist consistently across loop iterations" )
			binding = back_edge[name]
			if binding.state in ( OwnState.OWNED, OwnState.COPY ):
				instructions += self._decref_instructions( binding.type, binding.operand )
		if entry_results is not None:
			fresh_and_unchecked = self._unchecked_results - entry_results
			if fresh_and_unchecked:
				name = sorted( fresh_and_unchecked )[0]
				raise CompileError(
					f"{ctx}: Result value {name!r} is produced fresh every loop iteration but never inspected "
					f"before the next iteration overwrites it - use .is_ok(), .is_err(), .or_return(), .unwrap(msg), or match"
				)
		return instructions

	def unwind_to( self, snap: _Snapshot ) -> list[ir.Instruction]:
		''' break/continue - unwind everything pushed since `snap` (the
		enclosing loop's own entry snapshot) in LIFO order, without
		mutating state (the loop body may still have more statements after
		this break/continue in source order, on a path that isn't taken -
		lowering.py itself is responsible for not lowering unreachable code
		after an unconditional break/continue/return, same as it already
		is for defer/errdefer's own Jump-based exits). '''
		instructions: list[ir.Instruction] = []
		for entry in reversed( self._epilogue_stack[snap.stack_depth:] ):
			if entry.cancelled or entry.is_flag_guarded:
				continue
			instructions += self._decref_instructions( entry.type, entry.operand ) # regenerated fresh, not entry.instructions - see Epilogue.type's docstring
		return instructions

	def check_loop_exit_unchecked_results( self, entry_results: set[str], ctx: str ) -> None:
		''' break/continue's own unchecked-Result analogue of unwind_to() -
		called alongside it, same call sites. Only a Result introduced
		SINCE the loop's own entry (entry_results, from the same snapshot
		unwind_to() takes) is checked - one that already existed
		(unchecked) before the loop started isn't going out of scope here,
		its obligation is still owed by whatever outer code declared it,
		discharged wherever THAT binding's own scope actually ends (a later
		return, del, overwrite, or an enclosing if/loop's own merge point) -
		mirrors unwind_to()'s own "only entries pushed since snap" scoping
		for RC decrefs. Always clears the confined portion, whether this
		raises or not - same reasoning as check_unchecked_results(): dead
		code may still get lowered after this break/continue (see
		unwind_to()'s own docstring), and loop_back_edge()'s later,
		unconditional call must not re-discover (and potentially re-raise,
		uncaught) the exact same already-reported names. '''
		confined_unchecked = self._unchecked_results - entry_results
		self._unchecked_results -= confined_unchecked
		if confined_unchecked:
			names = ', '.join( repr( n ) for n in sorted( confined_unchecked ))
			plural = len( confined_unchecked ) > 1
			raise CompileError(
				f"{ctx}: Result value{'s' if plural else ''} {names} {'were' if plural else 'was'} never inspected "
				f"before exiting the loop - use .is_ok(), .is_err(), .or_return(), .unwrap(msg), or match"
			)

	# --- return / fall-off-the-end --------------------------------------------

	def return_(
		self, returned_operand: ir.Operand | None, get_is_err_check: 'Callable[[],tuple[list[ir.Instruction],ir.Operand]] | None' = None,
	) -> list[ir.Instruction]:
		''' unwind the ENTIRE current stack (every RC binding still live and
		every defer/errdefer entry still pending, function-wide) - called at
		each return statement and at the function's own fall-off-the-end.
		Skips whichever entry IS the returned value itself (ownership
		transfers to the caller, matched by identity - the same Variable/
		Temp object _lower_expr already returned for the `return`
		expression) - a flag-guarded entry's own operand is always None, so
		this never matches one of those. Doesn't mutate .bindings/the stack
		(lowering.py doesn't need it to - each return is independent, no
		code follows it on that path) EXCEPT for one thing: if the returned
		value is itself a bare fresh temp (`return SomeClass()`, never
		assigned to a name), it untracks that temp from _temp_states -
		otherwise the DeleteTemp _lower_stmt's own wrapper emits for it
		right after this statement would decref the very value we just
		handed to the caller. get_is_err_check is only ever actually called
		if an errdefer entry is genuinely live here - see _replay(). '''
		if isinstance( returned_operand, ir.Temp ):
			self._temp_states.pop( returned_operand.id, None )
		instructions: list[ir.Instruction] = []
		for entry in reversed( self._epilogue_stack ):
			if entry.cancelled:
				continue
			if returned_operand is not None and entry.operand is returned_operand:
				continue
			instructions += self._replay( entry, get_is_err_check )
		return instructions

	def current_epilogue_label( self, returned_operand: ir.Operand | None = None ) -> str | None:
		''' the label a `return` (or the function's own fall-off-the-end)
		should jump to instead of unwinding inline via return_() - the
		topmost still-active entry's own name (skipping only cancelled ones -
		e.g. a completed __init__'s own attribute entries, all cancelled by
		complete_construction(), must never produce a pointless jump/Label
		with nothing behind it; a flag-guarded entry is NOT skipped here,
		unlike return_()'s own inline replay - it's always "active" in the
		sense that something needs to check its flag, even if that check
		then finds it wasn't armed), shared across every return that reaches
		it (see build_epilogue_ladder()). None when nothing's left active (a
		plain ir.Return is correct instead), OR when returned_operand is
		itself one of the still-live RC entries ANYWHERE in the stack, not
		just the top: the shared ladder can't skip just one entry for just
		this one return (that's what return_()'s own "excludes the
		returned binding" already handles) - inlining via return_() is the
		only option there.

		ALSO None whenever the topmost active entry is confined to a loop
		body still being lowered (pushed at or after the innermost active
		enter_loop()'s own depth, and not flag-guarded - see restore()'s
		own comment on why a plain RC entry doesn't survive a loop body's
		exit but a defer/errdefer one does): that entry's own label would
		never actually get emitted anywhere (build_epilogue_ladder() only
		ever walks the stack that SURVIVES to the function's real end -
		restore() silently drops confined entries once the loop body
		lowering that pushed them finishes, well before then), so jumping
		into it here would be a dangling reference to a label that's never
		declared - confirmed by a real repro, not just reasoning (an early
		return from inside a while loop, past a locally-declared RC value,
		nested inside a with-block). return_()'s own full, inline unwind
		(which walks the ENTIRE stack directly, needing no label of its
		own at all) is the only correct option for a loop-confined entry,
		exactly like the returned-operand case just above. '''
		if returned_operand is not None and any(
			not entry.cancelled and entry.operand is returned_operand
			for entry in self._epilogue_stack
		):
			return None
		# None (not 0) when no loop is currently being lowered - the whole
		# confinement check below must be a no-op then (every entry is
		# function-scoped), not "confined below index 0" (which would
		# wrongly treat EVERY entry as confined, since every valid index
		# is >= 0)
		loop_floor = self._loop_entry_depths[-1] if self._loop_entry_depths else None
		for i, entry in reversed( list( enumerate( self._epilogue_stack ))):
			if entry.cancelled:
				continue
			if loop_floor is not None and not entry.is_flag_guarded and i >= loop_floor:
				return None
			return entry.name
		return None

	def build_epilogue_ladder(
		self, get_is_err_check: 'Callable[[],tuple[list[ir.Instruction],ir.Operand]] | None' = None,
	) -> list[ir.Instruction]:
		''' the shared unwind sequence every return that used
		current_epilogue_label() (and the function's own fall-off-the-end)
		jumps into - one Label + that entry's own still-live replay per
		pending entry (RC Decref, or a flag-guarded defer/errdefer replay -
		see _replay()), deepest (most-recently-pushed) first, each falling
		straight through into the next with no Jump needed. Cancelled
		entries still get their own Label (current_epilogue_label() can
		still point straight at one - see its own comment), just no
		instructions. Callers append their own final ir.Return - cfg.py has
		no notion of a function's return type or return-value slot. '''
		instructions: list[ir.Instruction] = []
		for entry in reversed( self._epilogue_stack ):
			instructions.append( ir.Label( name = entry.name ))
			if not entry.cancelled:
				instructions += self._replay( entry, get_is_err_check )
		return instructions

	def _replay( self, entry: Epilogue, get_is_err_check: 'Callable[[],tuple[list[ir.Instruction],ir.Operand]] | None' ) -> list[ir.Instruction]:
		# a plain RC entry's instructions always run unconditionally (their
		# liveness is already compile-time-exact - see the class docstring);
		# a flag-guarded one is runtime-conditional instead - skip_label
		# covers both "never armed" (flag) and, for errdefer specifically,
		# "armed but this isn't the error path" (is_err). get_is_err_check
		# is called FRESH here, inline, rather than once up front and
		# shared - with per-Epilogue labels, different returns can jump
		# into DIFFERENT points of the same ladder, so a check computed
		# once outside any specific entry's own replay wouldn't be reached
		# by every jump that might need it (a jump landing deeper in the
		# ladder skips right past it). Called only when actually needed -
		# is_err() is a real Call, not free
		if not entry.is_flag_guarded:
			return self._decref_instructions( entry.type, entry.operand ) # regenerated fresh, not entry.instructions - see Epilogue.type's docstring
		skip_label = self._new_label( 'defer_skip' )
		instructions = [ ir.JumpIfFalse( cond = entry.flag, target = skip_label ) ]
		if entry.is_err_only:
			is_err_instructions, is_err_temp = get_is_err_check()
			instructions += is_err_instructions
			instructions.append( ir.JumpIfFalse( cond = is_err_temp, target = skip_label ))
		instructions += entry.instructions
		instructions.append( ir.Label( name = skip_label ))
		return instructions

	# --- Incref/Decref emission, union-aware ------------------------------------

	def _incref_instructions( self, t: Type, operand: ir.Operand ) -> list[ir.Instruction]:
		return self._refcount_instructions( t, operand, ir.Incref )

	def _decref_instructions( self, t: Type, operand: ir.Operand ) -> list[ir.Instruction]:
		return self._refcount_instructions( t, operand, ir.Decref )

	def _refcount_instructions( self, t: Type, operand: ir.Operand, op: 'type[ir.Incref]|type[ir.Decref]' ) -> list[ir.Instruction]:
		leaves = rc_leaves( t )
		if not leaves:
			return []
		if not isinstance( t, TaggedUnion ):
			return [ op( value = operand ) ]
		all_leaves = t.leaves()
		if len( leaves ) == len( all_leaves ):
			# every member is RC - no tag check needed: they're all the
			# same union payload memory, and every RC type shares the same
			# header layout/offset, so any one member's accessor works
			instrs, payload = self._extract_payload( t, operand, t.attributes[0] )
			return instrs + [ op( value = payload ) ]
		return self._tag_gated_refcount_instructions( t, operand, leaves, op )

	def _extract_payload( self, t: TaggedUnion, operand: ir.Operand, member: Variable ) -> tuple[list[ir.Instruction],ir.Temp]:
		# mirrors lowering.py's _maybe_unwrap_union_arg: data.v_<member> -
		# two GetAttrs through the synthesized payload CUnion
		tag_attr, data_attr, payload_cls, tags = self._union_storage( t )
		payload_dest = self._new_temp( payload_cls )
		leaf_dest = self._new_temp( member.type )
		return (
			[
				ir.GetAttr( dest = payload_dest, obj = operand, attr = data_attr.stem ),
				ir.GetAttr( dest = leaf_dest, obj = payload_dest, attr = f'v_{member.stem}' ),
			],
			leaf_dest,
		)

	def _tag_gated_refcount_instructions( self, t: TaggedUnion, operand: ir.Operand, leaves: list[Type], op: 'type[ir.Incref]|type[ir.Decref]' ) -> list[ir.Instruction]:
		# unlike overload dispatch, there's no safe "last branch needs no
		# test" shortcut here - an unmatched tag means "this is one of the
		# union's non-RC leaves," which needs no action at all, not the
		# last RC leaf's action run unconditionally. Every RC leaf gets its
		# own Cmp+JumpIfFalse, falling through to a shared end label
		tag_attr, data_attr, payload_cls, tags = self._union_storage( t )
		u8_cls = tag_attr.type
		# identity-based membership deliberately, not `in` - Type dataclasses
		# have structural equality (mpy_types.py's own established
		# discipline, see _leaf_is_accepted's comment)
		members = [ m for m in t.attributes if any( m.type is leaf for leaf in leaves ) ]
		instructions: list[ir.Instruction] = []
		end_label = self._new_label( 'rc_end' )
		for member in members:
			next_label = self._new_label( 'rc_next' )
			tag_dest = self._new_temp( u8_cls )
			cmp_dest = self._new_temp( self._bool_type )
			instructions += [
				ir.GetAttr( dest = tag_dest, obj = operand, attr = tag_attr.stem ),
				ir.Cmp( dest = cmp_dest, op = ir.CmpOp.EQ, left = tag_dest, right = ir.Const( type = u8_cls, value = tags[member.stem] )),
				ir.JumpIfFalse( cond = cmp_dest, target = next_label ),
			]
			extract, payload = self._extract_payload( t, operand, member )
			instructions += extract
			instructions.append( op( value = payload ))
			instructions.append( ir.Jump( target = end_label ))
			instructions.append( ir.Label( name = next_label ))
		instructions.append( ir.Label( name = end_label ))
		return instructions

	# --- assignment: fresh / aliasing / replace, all in one -----------------

	def assign( self, dest: Variable, src: ir.Operand, *, is_alias: bool, track_result: bool = True ) -> list[ir.Instruction]:
		''' called right before lowering.py emits `ir.Assign(dest=dest,
		src=src)` (or the Allocate/Call/GetAttr that IS the fresh value, for
		an AnnAssign's own initializer) - returns instructions to emit
		immediately before that Assign. is_alias=True means `src` is a
		reference to an ALREADY-LIVE binding (a plain Name/GetAttr read) and
		needs an Incref; False means `src` is a freshly-produced value
		(Allocate, or a Call returning an RC type) that's already a fresh
		owned handoff, needing none.

		The unchecked-Result bookkeeping below runs BEFORE the rc_leaves()
		early-exit further down, deliberately: a Result[i32,SomeEnum] (no RC
		leaves at all) must still be tracked, and self.bindings/rc_leaves are
		an RC-only mechanism (see the module docstring - "invisible here").
		track_result=False opts a specific destination out of ever becoming
		a tracked obligation - used for compiler-synthesized locals whose
		Result-ness is scaffolding, not something user code is expected to
		inspect itself (the match-statement subject temp, and the hidden
		locals _emit_fallible_construction threads a fallible __init__'s
		Result through - see their own lowering.py call sites). '''
		if dest.stem in self._unchecked_results:
			raise CompileError(
				f"Result value {dest.stem!r} is discarded - it was never inspected: "
				f"use .is_ok(), .is_err(), .or_return(), .unwrap(msg), or match"
			)
		if track_result and is_result_type( dest.type ):
			self.track_result( dest.stem )
		else:
			self.clear_result( dest.stem )
		if not rc_leaves( dest.type ):
			return []
		instructions: list[ir.Instruction] = []
		if is_alias:
			instructions += self._incref_instructions( dest.type, src ) # bump the new value first - safe even if src and dest already alias the same object
		elif isinstance( src, ir.Temp ):
			# ownership transfers from the temp's own (momentary) tracking
			# into dest, not a second independent owner - untrack it so its
			# own eventual DeleteTemp doesn't ALSO decref the same object
			self._temp_states.pop( src.id, None )
		existing = self.bindings.get( dest.stem )
		if existing is not None and existing.entry is not None:
			if existing.state in ( OwnState.OWNED, OwnState.COPY ):
				instructions += self._decref_instructions( dest.type, dest ) # release whatever dest held before - reads dest's CURRENT value, emitted before the Assign overwrites it
			existing.entry.cancelled = False # dest is getting a real value again, even if it was MOVED/never-decref'd before
			self.bindings[dest.stem] = _Binding( operand = dest, type = dest.type, state = OwnState.OWNED, entry = existing.entry )
		else:
			self._push( dest, dest.type, OwnState.OWNED )
		return instructions

	def attr_assign( self, attr: Variable, src: ir.Operand, *, is_alias: bool ) -> list[ir.Instruction]:
		''' self.<attr> = value, inside __init__ specifically - mirrors
		assign()'s own fresh/replace logic almost exactly, but deliberately
		does NOT early-exit for a non-RC attr.type the way assign() does:
		complete_construction() needs to know an attribute was assigned
		even when it never needs a decref (e.g. a plain `a: int`), so every
		attribute is tracked here regardless of RC-ness - only the
		Incref/Decref emission itself stays gated on rc_leaves(). Tracked
		under a 'self.'-prefixed key (not attr.stem directly) so it can
		never collide with an ordinary local of the same base name (e.g.
		`def __init__(self, a): self.a = a` - a real, common pattern). '''
		key = f'self.{attr.stem}'
		is_rc = bool( rc_leaves( attr.type ))
		instructions: list[ir.Instruction] = []
		if is_rc:
			if is_alias:
				instructions += self._incref_instructions( attr.type, src )
			elif isinstance( src, ir.Temp ):
				self._temp_states.pop( src.id, None )
		existing = self.bindings.get( key )
		if existing is not None and existing.entry is not None:
			if is_rc and existing.state in ( OwnState.OWNED, OwnState.COPY ):
				instructions += self._decref_instructions( attr.type, attr )
			existing.entry.cancelled = False
			self.bindings[key] = _Binding( operand = attr, type = attr.type, state = OwnState.OWNED, entry = existing.entry )
		elif is_rc:
			self._push( attr, attr.type, OwnState.OWNED, key = key )
		else:
			self.bindings[key] = _Binding( operand = attr, type = attr.type, state = OwnState.OWNED, entry = None )
		return instructions

	def attr_replace( self, t: Type, old: ir.Operand, new: ir.Operand, *, is_alias: bool ) -> list[ir.Instruction]:
		''' self.<attr> = value, OUTSIDE __init__ construction - "an
		RCClass is always complete, so setting an attribute is always a
		replace" (RCCLASS ATTRIBUTE LIFETIME.md). Unlike attr_assign(),
		this never touches self.bindings/the epilogue stack - struct/union
		field CONTENTS aren't tracked across statements at all (see the
		module docstring's v1 scope cut), so there's no existing binding
		to look up here. lowering.py always reads the field's CURRENT
		value fresh (a GetAttr) and hands it here as `old`, unconditionally
		decref'd; `new` gets field_value()'s own is_alias treatment. Bumps
		`new` first, same as assign()'s own ordering - safe even if `old`
		and `new` happen to already alias the same object. '''
		instructions: list[ir.Instruction] = []
		if is_alias:
			instructions += self._incref_instructions( t, new )
		elif isinstance( new, ir.Temp ):
			self._temp_states.pop( new.id, None )
		instructions += self._decref_instructions( t, old )
		return instructions

	def fresh_temp( self, temp: ir.Temp, t: Type ) -> None:
		''' called right after lowering.py emits the Allocate/Call that
		produced `temp` holding a fresh RC value - registers it so a later
		DeleteTemp(temp) (if the value is never consumed into a named
		binding first - see assign()'s Temp-untracking branch) knows to
		decref it. '''
		if rc_leaves( t ):
			self._temp_states[temp.id] = t

	def delete_temp( self, temp: ir.Temp ) -> list[ir.Instruction]:
		t = self._temp_states.pop( temp.id, None )
		if t is None:
			return []
		return self._decref_instructions( t, temp )

	# --- struct/union field construction (Allocate) -----------------------------

	def field_value( self, t: Type, operand: ir.Operand, *, is_alias: bool ) -> list[ir.Instruction]:
		''' called for a value being embedded into a freshly-constructed
		struct/union field (Class.__allocate__(...)/bare ClassName(...)/a
		TaggedUnion member construction) - an aliasing reference (an existing
		binding's current value) gets its own independent Incref, since the
		new field is a distinct, possibly longer-lived holder of the same
		reference; a fresh value (Call/Allocate result) is already an owned
		handoff, needing none - mirrors assign()'s own is_alias distinction
		exactly. Deliberately does NOT touch self.bindings/the epilogue stack
		- struct/union FIELDS themselves are still untracked (see the module
		docstring's v1 scope cut: no per-field decref on the container's own
		teardown, since no emitter/codegen consumes Decref for real yet).
		This only prevents the SOURCE binding's own ordinary decref (at its
		own scope exit) from leaving the embedded copy under-refcounted -
		confirmed against bytearray.release()'s `Result.Err(
		OwnershipError.SharedReference( self ))`: this increfs self, then
		release()'s own epilogue decrefs self as usual - net zero, and the
		returned payload's reference is never the "already decremented" one.

		A fresh (is_alias=False) operand that's a Temp gets untracked here,
		mirroring assign()'s own "ownership transfers into dest, not a
		second independent owner" discipline for exactly the same reason:
		fresh_temp()-registered temps (see lowering.py's _emit) otherwise
		still look "pending" to their own statement's DeleteTemp, which
		would decref the very value that was just handed into this field -
		a double-decref alongside whatever (currently nonexistent) teardown
			the receiving struct/union eventually gets. '''
		if is_alias:
			return self._incref_instructions( t, operand )
		if isinstance( operand, ir.Temp ):
			self._temp_states.pop( operand.id, None )
		return []

	# --- move[T] call arguments ------------------------------------------------

	def move( self, operand: ir.Operand, *, target_qualname: str, param_stem: str ) -> list[ir.Instruction]:
		''' called for a Call argument matched against a move[T] parameter
		(already validated at the syntax level by prerequisite #2 - move()
		was actually written at the call site). No Incref/Decref at the
		call site itself either way - ownership transfers as-is. '''
		if isinstance( operand, Variable ):
			binding = self.bindings.get( operand.stem )
			if binding is None:
				return [] # not RC-tracked (non-RC type) - nothing to do
			if binding.state not in ( OwnState.OWNED, OwnState.COPY ):
				raise CompileError(
					f'{target_qualname}: cannot move {operand.stem!r} into parameter {param_stem!r} - '
					f'it is {binding.state.value}, not owned here'
				)
			if binding.entry is not None:
				binding.entry.cancelled = True
			# a fresh _Binding, never mutate the existing one in place - an
			# earlier snapshot() may still hold a reference to it (see
			# assign()'s own "always construct fresh" discipline)
			self.bindings[operand.stem] = _Binding( operand = binding.operand, type = binding.type, state = OwnState.MOVED, entry = binding.entry )
			return []
		if isinstance( operand, ir.Temp ):
			self._temp_states.pop( operand.id, None )
			return []
		return []

	# --- del x -------------------------------------------------------------

	def deleted( self, variable: Variable ) -> list[ir.Instruction]:
		''' called for `del x` (see lowering.py's _stmt_Delete) - returns
		the Decref to emit right there (if x was OWNED/COPY), and
		neutralizes its epilogue entry so it's never decref'd again.
		Independent-of-RC unchecked-Result check first, same reasoning as
		assign()'s own early check - del'ing a still-unchecked Result is
		exactly the "discarded via del" table entry, regardless of whether
		its type has any RC leaves at all. '''
		if variable.stem in self._unchecked_results:
			raise CompileError(
				f"Result value {variable.stem!r} is discarded via del - it was never inspected: "
				f"use .is_ok(), .is_err(), .or_return(), .unwrap(msg), or match"
			)
		binding = self.bindings.pop( variable.stem, None )
		if binding is None or binding.entry is None:
			return []
		instructions: list[ir.Instruction] = []
		if binding.state in ( OwnState.OWNED, OwnState.COPY ):
			instructions = self._decref_instructions( binding.type, variable )
		binding.entry.cancelled = True
		return instructions

	# --- compiler.decref(x) -------------------------------------------------

	def manually_decreffed( self, operand: ir.Operand ) -> None:
		''' called for compiler.decref(x) (see lowering.py's
		_lower_compiler_decref) - x's own explicit Decref is emitted by
		lowering.py right at the call site regardless; this only stops x's
		binding from being auto-decref'd a SECOND time once its own scope
		ends. Without this, a live OWNED/COPY local manually decref'd (the
		established idiom throughout this stdlib for tearing down RC
		elements read out of a container - list.__del__/FastList.__del__/
		dict's own _release_key/_release_value all do `val: T = <read>;
		compiler.decref(val)`) gets decref'd AGAIN by the scope's own
		epilogue, since compiler.decref was never wired into the ownership-
		tracking that epilogue relies on - confirmed with
		AddressSanitizer: a real, always-on (not merely heap-layout-
		dependent) double Decref -> use-after-free -> heap corruption on
		every single call, for exactly this shape. Mirrors move()'s own
		cancellation exactly (same entry.cancelled flag, same transition to
		MOVED so a later reference to x - now potentially freed - is caught
		as a compile error same as using a moved-out value would be), but
		without move()'s own state-mismatch error: compiler.decref(x) on a
		BORROWED binding (an ordinary un-owned parameter, entry is None -
		nothing to cancel) or one already MOVED/decref'd is left to whatever
		lowering.py itself decides to allow, not rejected here. '''
		if isinstance( operand, Variable ):
			binding = self.bindings.get( operand.stem )
			if binding is None or binding.entry is None:
				return
			if binding.state not in ( OwnState.OWNED, OwnState.COPY ):
				return
			binding.entry.cancelled = True
			self.bindings[operand.stem] = _Binding( operand = binding.operand, type = binding.type, state = OwnState.MOVED, entry = binding.entry )
			return
		if isinstance( operand, ir.Temp ):
			self._temp_states.pop( operand.id, None )

	# --- self construction (__init__) -------------------------------------

	def check_self_escape( self, operand: ir.Operand, ctx: str ) -> None:
		''' called (see lowering.py's _emit) for every operand of a newly
		emitted instruction EXCEPT GetAttr.obj/SetAttr.obj - self can only
		be used as the receiver of `self.attr` until every required
		attribute is initialized. Per your answer: no calling methods on
		self, no passing it anywhere else (Call receiver/argument, Assign
		src, Return value, Allocate field value, SetItem value) until then
		- release builds don't zero memory (see sys.alloc's debug-only
		zeroing), so a callee reading an uninitialized field would be
		genuine garbage, not just logically wrong. Flow-sensitive, not a
		one-time flag: re-checked fresh against current bindings every
		call, so self is allowed to escape as soon as every attribute
		happens to be set, even before __init__'s own return is reached.
		A no-op outside __init__ (self._construction_self is None) or for
		any operand that isn't literally self. '''
		if self._construction_self is None or operand is not self._construction_self:
			return
		missing = [ attr.stem for attr in self._construction_required if f'self.{attr.stem}' not in self.bindings ]
		if missing:
			raise CompileError(
				f"{ctx}: self cannot be used here until {', '.join(missing)} "
				f"{'is' if len(missing) == 1 else 'are'} initialized - only self.<attr> is allowed "
				f"inside __init__ before construction completes"
			)

	def complete_construction( self, ctx: str ) -> None:
		''' called on __init__'s SUCCESS path (fall-off, plain return, or
		Result.Ok(...) for a fallible __init__) - BEFORE the ordinary
		return_() unwind that follows. Raises CompileError listing any
		required attribute still missing from bindings; otherwise cancels
		every attribute binding's epilogue entry WITHOUT decref (ownership
		transfers into the now-complete self, mirroring move()'s own
		cancel-without-decref mechanism), so return_() only touches real
		locals afterward, never the attributes that just became part of
		self. Not called on a fallible __init__'s Result.Err(...) path -
		there, incomplete state is expected/legal, and the ordinary
		return_() unwind is exactly the desired cleanup (decref whichever
		attributes WERE set - "clean up any that were initialized"). '''
		missing = [ attr.stem for attr in self._construction_required if f'self.{attr.stem}' not in self.bindings ]
		if missing:
			raise CompileError( f"{ctx}: __init__ must initialize {', '.join(missing)} before returning" )
		for attr in self._construction_required:
			binding = self.bindings[f'self.{attr.stem}']
			if binding.entry is not None:
				binding.entry.cancelled = True
