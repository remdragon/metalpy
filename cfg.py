# stdlib imports:
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

# local imports:
import ir
from errors import CompileError
from mpy_types import Type, Variable, Parameter, Function, RCClass, TaggedUnion, CUnion, Move, Copy

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
	return isinstance( t, RCClass )

def rc_leaves( t: Type ) -> list[Type]:
	# a TaggedUnion's RC-relevant leaves specifically - str|i32 needs a
	# tag-gated incref (only str); str|int (both RC) needs none of that,
	# unconditional instead
	if isinstance( t, TaggedUnion ):
		return [ leaf for leaf in t.leaves() if is_rc( leaf ) ]
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
	instructions: list[ir.Instruction]
	operand: Variable | None = None # the RC binding this entry decrefs - None for defer/errdefer entries. Lets return_() skip decref'ing whatever's actually being returned, by identity
	flag: Variable | None = None
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
		self.bindings: Bindings = {}
		self._temp_states: dict[int,Type] = {} # ir.Temp.id -> its type, only while OWNED (temps are never BORROWED/COPY/MOVED)
		self.prologue_instructions: list[ir.Instruction] = []
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

	def _push( self, operand: Variable, type_for_decref: Type, state: OwnState ) -> Epilogue:
		entry = Epilogue( instructions = self._decref_instructions( type_for_decref, operand ), operand = operand )
		self._epilogue_stack.append( entry )
		self.bindings[operand.stem] = _Binding( operand = operand, type = type_for_decref, state = state, entry = entry )
		return entry

	# --- snapshot/restore, for IF/loop orchestration ----------------------------

	def snapshot( self ) -> _Snapshot:
		return _Snapshot( bindings = dict( self.bindings ), stack_depth = len( self._epilogue_stack ))

	def restore( self, snap: _Snapshot ) -> None:
		self.bindings = dict( snap.bindings )
		del self._epilogue_stack[snap.stack_depth:]

	# --- IF/ELSE/ENDIF -----------------------------------------------------

	def merge_if(
		self, entry_bindings: Bindings, true_end: Bindings, false_end: Bindings, ctx: str,
		*, true_terminates: bool = False, false_terminates: bool = False,
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
		re-pushed after being del'd and reassigned) needs a real push. '''
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
			if true_terminates and not false_terminates:
				survivor = false_end
			elif false_terminates and not true_terminates:
				survivor = true_end
			if survivor is not None:
				for name, binding in survivor.items():
					prior = entry_bindings.get( name )
					already_live = prior is not None and prior.entry is binding.entry
					reestablish( name, binding, already_live )
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
		return true_instructions, false_instructions, removed

	# --- loops ---------------------------------------------------------------

	def loop_back_edge( self, entry_bindings: Bindings, ctx: str ) -> list[ir.Instruction]:
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
		responsibilities). '''
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
			instructions += entry.instructions
		return instructions

	# --- return / fall-off-the-end --------------------------------------------

	def return_( self, returned_operand: ir.Operand | None ) -> list[ir.Instruction]:
		''' unwind the ENTIRE current stack (every RC binding still live,
		function-wide) - called at each return statement and at the
		function's own fall-off-the-end. Skips whichever entry IS the
		returned value itself (ownership transfers to the caller, matched
		by identity - the same Variable/Temp object _lower_expr already
		returned for the `return` expression). Doesn't mutate .bindings/the
		stack (lowering.py doesn't need it to - each return is independent,
		no code follows it on that path) EXCEPT for one thing: if the
		returned value is itself a bare fresh temp (`return SomeClass()`,
		never assigned to a name), it untracks that temp from
		_temp_states - otherwise the DeleteTemp _lower_stmt's own wrapper
		emits for it right after this statement would decref the very
		value we just handed to the caller. Flag-guarded (defer/errdefer)
		entries are handled by lowering.py's existing epilogue machinery,
		not here - see the Integration section of the plan. '''
		if isinstance( returned_operand, ir.Temp ):
			self._temp_states.pop( returned_operand.id, None )
		instructions: list[ir.Instruction] = []
		for entry in reversed( self._epilogue_stack ):
			if entry.cancelled or entry.is_flag_guarded:
				continue
			if returned_operand is not None and entry.operand is returned_operand:
				continue
			instructions += entry.instructions
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

	def assign( self, dest: Variable, src: ir.Operand, *, is_alias: bool ) -> list[ir.Instruction]:
		''' called right before lowering.py emits `ir.Assign(dest=dest,
		src=src)` (or the Allocate/Call/GetAttr that IS the fresh value, for
		an AnnAssign's own initializer) - returns instructions to emit
		immediately before that Assign. is_alias=True means `src` is a
		reference to an ALREADY-LIVE binding (a plain Name/GetAttr read) and
		needs an Incref; False means `src` is a freshly-produced value
		(Allocate, or a Call returning an RC type) that's already a fresh
		owned handoff, needing none. '''
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
		neutralizes its epilogue entry so it's never decref'd again. '''
		binding = self.bindings.pop( variable.stem, None )
		if binding is None or binding.entry is None:
			return []
		instructions: list[ir.Instruction] = []
		if binding.state in ( OwnState.OWNED, OwnState.COPY ):
			instructions = self._decref_instructions( binding.type, variable )
		binding.entry.cancelled = True
		return instructions
