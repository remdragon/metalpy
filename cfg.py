# stdlib imports:
from dataclasses import dataclass, field, replace as _dc_replace
from enum import Enum
from typing import Callable

# local imports:
import ir
from errors import CompileError
from mpy_types import Type, Variable, Parameter, Function, RCClass, TaggedUnion, CUnion, Move, Copy, Specialization, TupleType

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
	OWNED = 'owned' # a real epilogue entry backs this binding - decref'd at scope exit. Covers every provenance: a fresh Call/Allocate result, a move[T]/copy[T] parameter (copy[T] takes its own Incref in the prologue - _enter_parameter - but is otherwise indistinguishable from any other OWNED binding; no consumer anywhere ever needed to tell them apart, so there's no separate COPY state)
	BORROWED = 'borrowed'
	MOVED = 'moved'

# rc_leaves/_is_direct_pointer_rc used to be open-coded isinstance ladders
# right here, which is how the same bug shipped three separate times: a new
# Type kind appeared, this ladder wasn't updated, and the new kind silently
# defaulted to "not RC" (nested-union leaf -> leak; generic union's
# unsubstituted TypeVar leaves -> UAF; unresolved union -> order-dependent
# UAF). Each type kind now answers for itself - see mpy_types.Type's own
# is_rc/is_rc_pointer/rc_leaves, which carry the full history of those bugs.
# These two stay as module-level names purely because their call sites in
# this file and lowering.py already spell them that way. is_result_type
# likewise delegates to mpy_types.Type.is_result_type(), but stays a free
# function (rather than being replaced by direct .is_result_type() calls)
# since every call site passes a Type|None and needs the None-guard.

def is_result_type( t: Type|None ) -> bool:
	''' True when `t` is a concrete Result[T,E] specialization. '''
	return t is not None and t.is_result_type()

def rc_leaves( t: Type ) -> list[Type]:
	return t.rc_leaves()

def _is_direct_pointer_rc( t: Type ) -> bool:
	''' True for an RC leaf whose OWN runtime representation is a single, bare
	pointer - as opposed to a NESTED union leaf (also "RC", but a union's
	runtime shape is a tag+data VALUE STRUCT, not a pointer at all).
	_refcount_instructions' "every member shares the same underlying pointer
	layout, read any ONE member's accessor" shortcut is only safe when EVERY
	leaf satisfies this - it silently produces garbage for a nested-union leaf
	otherwise (reading that leaf's payload accessor as a bare RC pointer). '''
	return t.is_rc_pointer()

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
	can be cancelled at arbitrary points - see move()/deleted().

	Deliberately has NO `captured` field, unlike `cancelled`/`flag` - those
	two are genuinely BRANCH-LOCAL (each independently-lowered sibling
	branch needs its own private answer, which is exactly what made them a
	real, repeated source of double-frees when this class's own instances
	were shared by reference across branches instead of value-copied - see
	CFGState._captured_labels' own docstring for the fix). "Has any
	already-lowered code, anywhere in the function, committed a goto into
	THIS label" is a whole-function, monotonic fact instead - true the
	moment ANY branch's own return commits to it, staying true for the
	rest of the function's lowering regardless of which branch that was,
	and it needs to read the SAME way from every clone of this same
	logical entry, not diverge per clone the way an ordinary field would.
	Tracked in CFGState._captured_labels (keyed by `name`, which is a
	plain, clone-invariant string) instead of on the entry itself. '''
	instructions: list[ir.Instruction] # defer/errdefer entries only (flag is not None) - the already-lowered replay body, reused as-is (safe: _replay()'s flag-guarded path mints a fresh skip-label on every call). Plain RC entries leave this empty and use `type` below instead
	name: str # this entry's own jump target - see current_epilogue_label()/build_epilogue_ladder(); also the stable key CFGState._captured_labels uses, see this class's own docstring
	operand: Variable | None = None # the RC binding this entry decrefs - None for defer/errdefer entries. Lets return_() skip decref'ing whatever's actually being returned, by identity
	type: Type | None = None # plain RC entries only - the type to decref `operand` as. Instructions are regenerated fresh from this on every replay (_replay()/unwind_to()) rather than cached: a loop-confined entry can be replayed at more than one emission point (an early return inside the loop while it's still the topmost active entry, or several break/continue in the same loop) before it's ever dropped by restore(), and a cached instruction list would bake in the same tag-gated-decref Label names at every one of those sites - confirmed by a real repro, "redefinition of label" from clang on a loop with two early-return Result checks in a row
	flag: Variable | None = None
	is_err_only: bool = False # errdefer vs plain defer - only meaningful when flag is set
	cancelled: bool = False
	is_construction_attr: bool = False # a self.<attr> entry pushed by attr_assign()/complete_base_construction() during a fallible __init__ - see current_epilogue_label_for_construction_err()'s own docstring for why these can never share a label the way a defer/errdefer or plain local entry can
	survives_loop_restore: bool = False # set only by promote_borrowed_for_loop() - that entry is pushed WHILE lowering a loop body (so restore() would normally treat it as block-scoped and truncate it away, same as an ordinary loop-local), but it represents a real, permanent ownership promotion of a pre-loop binding that now needs releasing once, at the function's own epilogue - not every loop iteration and not never. See restore()'s own survivors check.

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
class InlineScope:
	''' one active multi-statement @inline splice's own local "epilogue" -
	pushed by push_inline_scope() when lowering.py's _splice_multi_statement_
	inline_body begins lowering a target's pre-return statements, popped once
	it's done. current_epilogue_label()/return_() both stop at boundary_depth
	instead of continuing into the CALLER's own older entries - see their own
	comments. A stack (not a single field) because a spliced body can itself
	call another @inline function - the innermost entry is always the one
	that matters. '''
	boundary_depth: int # len(self._epilogue_stack) at push time - entries below this belong to an outer scope (the caller, or an outer splice) and must never be inspected/replayed from inside this one
	label: str # this scope's own shared-ladder fallback target - see current_epilogue_label()'s own comment
	# two INDEPENDENT captured flags, not one shared bit - lowering.py's own
	# _splice_multi_statement_inline_body builds two labels around this
	# scope (`label` itself, and a separate merge_label it owns directly,
	# not stored here), and an early exit only ever reaches ONE of them: an
	# ordinary .or_return()/checked-arithmetic with nothing else pending
	# jumps straight to `label` via current_epilogue_label() (captured
	# below), after which merge_label is reached only by ordinary
	# fallthrough (from label's own replayed ladder) - NEVER a real goto,
	# UNLESS some other early exit in the SAME splice took the separate
	# inline-unwind bypass path instead (mark_inline_scope_captured() below,
	# called directly by _stmt_Return/_consume_checked_result - see their
	# own comments), which jumps PAST `label` straight to merge_label.
	# Conflating the two into one flag is a real, confirmed bug: it makes
	# merge_label look "used" whenever ANY early exit occurred anywhere in
	# the splice, even one that only ever captured `label` - still a
	# genuine -Wunused-label on merge_label specifically, confirmed by a
	# real repro (a splice with a SINGLE or_return() call, going through
	# `label`'s own capture path, not the bypass one).
	captured: bool = False # `label` has been handed out as a live jump target - see current_epilogue_label()
	merge_captured: bool = False # merge_label has been jumped to DIRECTLY (the inline-unwind bypass) - see mark_inline_scope_captured()

@dataclass
class _Snapshot:
	''' captured by snapshot(), consumed by restore() - see the IF/loop
	orchestration lowering.py performs around branches/loop bodies.
	entry_cancelled/entry_flag record every SURVIVING entry's (index <
	stack_depth) own Epilogue state as of snapshot time; entry_objects
	records the ACTUAL Epilogue objects themselves (same order/indices),
	for restore()'s own identity-preserving swap-back (see its own
	docstring - a static cancel's own REPLACEMENT object must be swapped
	back to the ORIGINAL object identity, not just have its .cancelled
	field flipped, or self.bindings[stem].entry and self._epilogue_stack[i]
	silently diverge into two different objects that both claim to be
	"the" entry for the same name - confirmed by a real repro: a later
	_neutralize() on the reestablished (bindings-side) original object then
	fails its own `self._epilogue_stack[i] = new_entry` identity search
	entirely, since the stack-side slot never got reverted to that same
	object, leaving the earlier restore-reset replacement (cancelled=False)
	sitting there permanently - a real double release once the function's
	own closing ladder gets to it). entry_cancelled/entry_flag are ONLY
	consumed by hard_restore() now (unconditional field reset on whatever's
	CURRENTLY at each index - correct there since a hard_restore() discards
	the whole abandoned attempt, no identity to preserve). No entry_captured
	here - see Epilogue's own docstring for why captured-ness isn't
	per-entry snapshotted state at all. '''
	bindings: Bindings
	stack_depth: int
	results: set[str]
	narrowed: dict[str,Variable]
	live: set[str]
	entry_cancelled: list[bool]
	entry_flag: list['Variable | None']
	entry_objects: list['Epilogue']

class CFGState:
	''' one instance per function being lowered. `bindings` is public and
	directly snapshot/restore-able (`dict(state.bindings)` / assignment) -
	lowering.py owns sequencing (when to snapshot before a branch, when to
	restore for `else`), cfg.py only computes what a given transition or
	merge means. '''

	def __init__(
		self,
		fn: Function | None,
		*,
		bool_type: Type,
		new_temp: Callable[[Type], ir.Temp],
		new_label: Callable[[str], str],
		union_storage: UnionStorage,
		resolve_type: Callable[[object], object] = lambda t: t,
	) -> None:
		self.fn = fn # None for a global Variable's own initializer (lowering.py's FunctionLowering.run_global) - no parameters to enter below, no self, no construction
		self._bool_type = bool_type
		# swaps a bare Specialization (e.g. Result[str,SomeError], still
		# wrapping the ABSTRACT, unmonomorphized Result class) for its real,
		# per-instantiation monomorphized ClassLike (SUBSTITUTED leaf types -
		# see _refcount_instructions' own comment on why this matters and is
		# called from there specifically). lowering.py wires this to its own
		# _ensure_resolved (the same "resolve now + swap a Specialization for
		# its monomorphized form" helper used ~15 other places in that file);
		# defaults to the identity function so cfg_test.py's own bare-Type
		# CFGState construction (no real Monomorphizer around at all) is
		# unaffected - those fixture Types are never Specializations of a
		# still-generic TaggedUnion in the first place.
		self._resolve_type = resolve_type
		self._new_temp = new_temp
		self._new_label = new_label
		self._union_storage = union_storage
		self._epilogue_stack: list[Epilogue] = []
		# whole-function, monotonic, name-keyed - see Epilogue's own
		# docstring for why this lives here instead of as a field on
		# Epilogue itself. Never reset/reverted by restore()/hard_restore()
		# - once a label is captured, that stays true regardless of which
		# branch/clone did it or whether that branch's own code even
		# survives (a discarded loop-retry attempt's own capture is
		# harmless dead data here: the retried attempt re-lowers the same
		# source, so it independently re-captures under a FRESH name of
		# its own - see hard_restore()'s own comment)
		self._captured_labels: set[str] = set()
		self._any_shared_label_used: bool = False # see used_shared_epilogue_label()'s own docstring
		self._cancel_flags: list[Variable] = [] # see _neutralize()/cancel_flags() - minted lazily, only for an entry that turns out to need one
		self._confinement_depths: list[int] = [] # see enter_loop()/exit_loop() and enter_branch()/exit_branch()
		self._inline_scope_stack: list[InlineScope] = [] # see push_inline_scope()/pop_inline_scope()
		self._break_narrowed_stack: list[list[dict[str,list[Variable]]]] = [] # one entry per currently-lowering loop (innermost last) - each entry collects a dict[str,list[Variable]] snapshot per break reached inside THAT loop specifically, see enter_loop()/exit_loop()/record_break_narrowed()/merge_loop_exits()
		self._break_live_stack: list[list[set[str]]] = [] # the definite-assignment analogue of _break_narrowed_stack above - one set[str] snapshot per break, see record_break_live()
		self.bindings: Bindings = {}
		self._live: set[str] = set() # names of locals DEFINITELY ASSIGNED on the current path - independent of RC tracking above (unlike bindings/rc_leaves, tracks EVERY local regardless of type - see assign()/is_live()/_expr_Name's own liveness gate). Parameters/self are always live from entry (seeded below/in enter_self()); a bare AnnAssign's own name is added to fn.names but NOT here until its first real assignment
		self._unchecked_results: set[str] = set() # names of locals currently holding a Result[T,E] that hasn't been is_ok()/is_err()/or_return()/unwrap()/unwrap_or()'d or match'd yet - independent of RC tracking above, see track_result()/clear_result()
		self._narrowed: dict[str,list[Variable]] = {} # name -> the non-empty set of the UNION's own members it could still be (each .type the narrowed leaf, .stem the v_<stem> payload field) - see narrow()/unnarrow()/narrowed_member(). A pure compile-time READ-REWRITE fact, no RC implications at all: the name's own real Variable/storage never changes, this only says "a read of this name, right here, may be rewritten to read through the union's own payload instead", and ONLY when the set has collapsed to exactly one member - see narrowed_member(). A single narrow() call always starts as a one-element list; merge_if's own soft-merge can grow it (two disagreeing-but-both-still-possible branches union together rather than discarding the fact) or drop it (a name narrowed on only SOME surviving paths)
		self._temp_states: dict[int,Type] = {} # ir.Temp.id -> its type, only while OWNED (temps are never BORROWED/MOVED)
		self.prologue_instructions: list[ir.Instruction] = []
		self._construction_self: Variable | None = None # set by enter_construction() - which self param (if any) is still under construction
		self._construction_required: list[Variable] = [] # __init__'s own attributes that must all be initialized before self can escape/construction can complete
		self._possibly_retained: set[str] = set() # stems of OWNED locals passed as an argument into a call whose own errdefer/defer conditionally retains that PARAMETER (Function.errdefer_retained_params, set by push_defer below) - see mark_possibly_retained()/manually_decreffed()'s own use of this
		for param in ( fn.parameters or [] ) if fn is not None else []:
			self._enter_parameter( param )

	# --- prologue --------------------------------------------------------------

	def _enter_parameter( self, param: Parameter ) -> None:
		# param.type is always the real, unwrapped T here (discovery.py's
		# own parameter-construction site already strips move[T]/copy[T]
		# down to T, recording the ownership fact on is_move/is_copy
		# instead - see Parameter's own docstring) - only the OWNERSHIP
		# STATE this prologue sets up differs by which flag is set
		self._live.add( param.stem ) # every parameter is definitely assigned from function entry, RC or not
		if param.is_move:
			# the callee now fully owns the incoming reference - MOVED is
			# the CALLER's state at the call site, not the callee's own
			# parameter (see prerequisite #2's move() call-site check,
			# which is what guarantees this parameter really was moved in)
			if rc_leaves( param.type ):
				self._push( param, param.type, OwnState.OWNED )
		elif param.is_copy:
			# the callee wants its own independent reference - an explicit
			# Incref right here in the prologue, matching Decref at exit
			if rc_leaves( param.type ):
				self.prologue_instructions += self._incref_instructions( param.type, param )
				self._push( param, param.type, OwnState.OWNED )
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
		self._live.add( self_param.stem ) # self is definitely assigned from entry, RC or not - unlike the rc_leaves early-return below, this must run unconditionally
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
		# a COPY, not the caller's own list by reference - the caller passes
		# self_cls.attributes directly (lowering.py), the class's own
		# permanent declared-fields list; complete_base_construction() below
		# appends base attributes onto self._construction_required so
		# complete_construction()'s success-path cancellation loop also
		# covers them (see its own comment) - without this copy, that append
		# would mutate self_cls.attributes itself, permanently duplicating
		# the base attribute into the subclass's own field list (confirmed
		# by a real "duplicate member" C struct compile error while fixing
		# the bug complete_base_construction's own comment describes).
		self._construction_required = list( required )

	def _push( self, operand: Variable, type_for_decref: Type, state: OwnState, *, key: str | None = None, is_construction_attr: bool = False, survives_loop_restore: bool = False ) -> Epilogue:
		entry = Epilogue( instructions = [], name = self._new_label( 'epilogue' ), operand = operand, type = type_for_decref, is_construction_attr = is_construction_attr, survives_loop_restore = survives_loop_restore )
		self._epilogue_stack.append( entry )
		self.bindings[key if key is not None else operand.stem] = _Binding( operand = operand, type = type_for_decref, state = state, entry = entry )
		return entry

	def push_defer( self, instructions: list[ir.Instruction], flag: Variable, is_err_only: bool ) -> Epilogue:
		''' defer/errdefer's own replay, registered at the defer/errdefer
		statement's own position (lowering.py emits the flag's own `= True`
		Assign right after this call - that's what "armed" means at
		runtime). Interleaved into the SAME _epilogue_stack RC bindings use,
		by declaration order - build_epilogue_ladder() replays the whole
		stack together, deepest first. Never touches self.bindings (there's
		no name to look it up by - it's not a variable), so it's immune to
		merge_if()'s dict-based reconciliation entirely; restore() below
		gives it the different treatment it actually needs instead.
		Returns the pushed entry so a caller whose own body always falls
		through (with's __exit__, a for-loop's iterator release, try/
		finally's finalbody) can hand it straight to disarm_defer() below,
		instead of unconditionally emitting a runtime flag reset. '''
		entry = Epilogue(
			instructions = instructions, name = self._new_label( 'epilogue' ), flag = flag, is_err_only = is_err_only,
		)
		self._epilogue_stack.append( entry )
		return entry

	def disarm_defer( self, entry: Epilogue ) -> list[ir.Instruction]:
		''' called right where a defer/errdefer's registering construct
		(with/for/try-finally) reaches its own natural, always-executed exit
		- the caller is about to replay entry's body directly, right here,
		so the function-epilogue ladder must never replay it again. Thin
		wrapper over _neutralize(): the overwhelmingly common case (nothing
		earlier in the function has already committed a goto into entry's
		own shared label) statically cancels it, returning no instructions
		at all - emitting `flag = false` there anyway is always-false-at-
		runtime dead code MSVC's flow analysis proves and warns on (C4702)
		at every one of these call sites. Falls back to the real runtime
		reset only in the rarer captured-label case, exactly like
		_neutralize()'s other callers. Discards the replacement entry
		_neutralize() returns - safe here since push_defer() never records
		a defer entry in self.bindings, so nothing else holds a reference
		to the original that would need updating. '''
		_replacement, instructions = self._neutralize( entry )
		return instructions

	def mark_possibly_retained( self, operand: ir.Operand ) -> None:
		''' called by lowering.py's own _emit() (_mark_errdefer_retained_args)
		for a Call whose target's errdefer_retained_params (an AST-derived
		fact - see Lowering._ensure_errdefer_retained_params) says one of
		THIS call's own arguments may come back with an extra, compiler-
		invisible reference already attached: a bare `errdefer(compiler.
		incref(param))`/`defer(compiler.incref(param))` inside the callee
		conditionally hands param's OWN object an extra reference that
		nothing in the callee's own scope ever releases - by design, it's
		meant for the CALLER to release by hand (compiler.refcount()/
		compiler.decref()). operand is the caller's own argument occupying
		that parameter position. A no-op for anything but a plain Variable
		(a Temp/literal argument has no binding for manually_decreffed() to
		later consult anyway). Sticky for the rest of the binding's
		lifetime, same as OwnState itself - a SECOND call through the same
		parameter position only needs to have set this once. '''
		if isinstance( operand, Variable ):
			self._possibly_retained.add( operand.stem )

	def push_inline_scope( self ) -> str:
		''' called once by lowering.py's own _splice_multi_statement_inline_
		body, right before it starts lowering a target's pre-return
		statements - marks the CURRENT stack depth as this splice's own
		boundary. current_epilogue_label()/return_() both stop here instead
		of continuing into the CALLER's own older entries (see their own
		comments) - this is the entire fix that lets an early `return`/
		`.or_return()`/checked-arithmetic inside a spliced body jump to a
		label that's genuinely local to the splice, never the caller's real
		epilogue. Returns the fresh label current_epilogue_label() falls back
		to once nothing shallower (within this scope) qualifies - the caller
		(lowering.py) emits this as a real ir.Label at the end of the splice,
		right where its own local ladder begins. '''
		scope = InlineScope( boundary_depth = len( self._epilogue_stack ), label = self._new_label( 'inline_epilogue' ))
		self._inline_scope_stack.append( scope )
		return scope.label

	def inline_scope_captured( self ) -> bool:
		''' whether the innermost active scope's own `label` (NOT its
		separate merge_label - see InlineScope's own docstring for why the
		two need independent tracking) has been handed out as a real jump
		target so far. Peeks without popping, so lowering.py can gate its
		scope_label ir.Label BEFORE build_inline_scope_ladder() runs (pop_
		inline_scope() only happens after that, but merge_label's own ir.
		Label is emitted after the pop - see its own return value instead). '''
		return self._inline_scope_stack[-1].captured

	def mark_inline_scope_captured( self ) -> None:
		''' called by lowering.py right before it emits a real jump straight
		to the innermost active scope's own merge_label, bypassing `label`
		entirely (the "inline-unwind return_()/.or_return() already
		replayed everything itself" shape - see _stmt_Return/_consume_
		checked_result's own inline_exit branches) - current_epilogue_label()
		only marks `label` captured when IT hands that one out, so this
		separate bypass path (which never calls it, and targets the OTHER
		label) needs its own explicit signal. '''
		self._inline_scope_stack[-1].merge_captured = True

	def pop_inline_scope( self ) -> bool:
		''' called once the splice's own local ladder has been fully emitted
		(lowering.py's own responsibility - this just stops
		current_epilogue_label()/return_() from consulting this scope's
		boundary any further, restoring the immediately-enclosing scope, if
		any, to visibility - the caller/outer splice's own entries were never
		touched while this scope was active, so there's nothing left to
		reconcile here beyond popping the stack entry itself. Returns
		whether the popped scope's own merge_label was ever captured -
		lowering.py's own merge_label ir.Label is emitted right after this
		call, gated on it (see inline_scope_captured()'s own docstring for
		why `label` itself is peeked separately, before this pop, instead). '''
		return self._inline_scope_stack.pop().merge_captured

	# --- snapshot/restore, for IF/loop orchestration ----------------------------

	def snapshot( self ) -> _Snapshot:
		return _Snapshot(
			bindings = dict( self.bindings ), stack_depth = len( self._epilogue_stack ), results = set( self._unchecked_results ),
			narrowed = dict( self._narrowed ), live = set( self._live ),
			entry_cancelled = [ e.cancelled for e in self._epilogue_stack ],
			entry_flag = [ e.flag for e in self._epilogue_stack ],
			entry_objects = list( self._epilogue_stack ),
		)

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
		assumed checked once we're back outside it (see loop_back_edge()).
		_narrowed reverts the same unconditional way - a union narrowed
		inside an if/match branch (or a loop body) is never assumed to still
		hold once back outside it, same reasoning as _unchecked_results
		above, and (v1 scope, see TODO.txt's own "union disambiguation"
		section) narrowing never survives past its own branch regardless of
		whether that branch could only have been entered when it's true -
		no cross-branch/post-if narrowing tracking is attempted yet.

		_live reverts the same unconditional way, for the identical reason:
		a name only definitely-assigned INSIDE a branch/loop body is never
		assumed definitely-assigned once back outside it - merge_if()/
		merge_loop_exits() are what let a name's liveness survive past the
		construct, via their own explicit reconciliation, same split of
		responsibility as bindings/narrowed above.

		A CAPTURED plain entry also survives, same as a flag-guarded one -
		current_epilogue_label() already handed its .name out as a live goto
		target (this can happen even for a genuinely branch-scoped entry
		when the pushing code itself is UNCONFINED at push time - e.g.
		_stmt_Try's own try body, which never wraps itself in enter_branch()
		since it always runs exactly once when reached, unlike a handler -
		see its own docstring). Dropping it here would leave that goto
		dangling once build_epilogue_ladder() never emits the matching
		Label - confirmed by a real repro (a named Result local, or a
		compiler-synthesized match subject, declared directly inside a
		try body, with a `return`/match arm reachable from inside that same
		body). Safe to keep unconditionally: an entry that's NOT captured
		here is confined-and-never-jumped-to, so its teardown is already
		fully handled by whichever reconciliation call (merge_if()/
		merge_loop_exits()) is about to run instead - and an entry that's
		genuinely confined (pushed at or after a live enter_branch()/
		enter_loop() depth) never reaches captured=True in the first place,
		since current_epilogue_label() refuses to hand out a label for one
		(see its own docstring) - so this can never resurrect an entry that
		was truly meant to be block-scoped.

		Also swaps every SURVIVING pre-branch entry (index < stack_depth)
		that's still NOT flag-guarded back to its ORIGINAL object identity
		(snap.entry_objects), not just a reset .cancelled field:
		_neutralize()'s static-cancel path (move()/deleted()/manually_
		decreffed() on an entry declared before this branch) swaps a
		cancelled=True REPLACEMENT object into self._epilogue_stack IN
		PLACE - self.bindings gets the replacement too, but restore() above
		already resets bindings back to snap's own pristine entry OBJECT, so
		only this raw list is left pointing at the stale replacement.
		Merely flipping the replacement's OWN .cancelled back to False (a
		prior version of this) still leaves self.bindings[stem].entry and
		self._epilogue_stack[i] as two DIFFERENT objects claiming to be the
		same entry - a later _neutralize() on the reestablished
		(bindings-side, original) object then searches self._epilogue_stack
		for `e is entry` and never finds it (the list still holds the
		replacement), so its own `self._epilogue_stack[i] = new_entry` swap
		silently fails to happen at all, leaving the earlier reset
		replacement (cancelled=False) sitting there permanently - confirmed
		by a real repro: a pre-if local del'd on an if's own terminating
		branch, then unconditionally del'd again right after the if -
		the second del's own cancellation never reached the real stack
		slot, a real double release at the function's own closing ladder.
		Swapping the actual object back, not just its field, keeps identity
		consistent for every later lookup. Without reverting this at all,
		the FIRST branch's cancellation permanently leaks into every
		sibling restored back to the SAME snap (confirmed by a real repro:
		`del`/compiler.decref() on a pre-if local in the true branch only,
		then an unconditional release expected in the false branch -
		build_epilogue_ladder() found the entry still cancelled and
		silently dropped the false branch's own release, a real leak) -
		exactly the bug enter_diverging_paths() used to paper over before it
		was removed as "provably redundant". A flag-guarded entry is left
		untouched on purpose (matches hard_restore()'s own carve-out): its
		runtime flag already reconciles both branches correctly regardless
		of which one(s) actually disarmed it, and a NEWLY minted flag from
		THIS branch (entry.flag now set, snap recorded None) must stay live
		for the caller's own already-emitted flag-guarded release. '''
		self.bindings = dict( snap.bindings )
		self._unchecked_results = set( snap.results )
		self._narrowed = dict( snap.narrowed )
		self._live = set( snap.live )
		for i, orig_entry in enumerate( snap.entry_objects ):
			if not self._epilogue_stack[i].is_flag_guarded:
				self._epilogue_stack[i] = orig_entry
		survivors = [ e for e in self._epilogue_stack[snap.stack_depth:] if e.is_flag_guarded or e.name in self._captured_labels or e.survives_loop_restore ]
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
		its many call sites threading loop context through by hand. Shares
		_confinement_depths with enter_branch()/exit_branch() below - from
		current_epilogue_label()'s point of view an if/match arm not yet
		merged and a loop body not yet closed are the same hazard (restore()
		is about to silently drop entries pushed since the same kind of
		snapshot), so one combined stack (read via min(), not just the top -
		see current_epilogue_label()'s own comment) covers both without
		needing to know which kind of scope is which. Also pushes a fresh,
		empty collection list onto _break_narrowed_stack (Phase 8) - every
		`break` reached while lowering THIS loop's own body records a
		narrowed-state snapshot into it, consumed by exit_loop()'s own
		return value once this loop's body is fully lowered. _break_live_
		stack is the definite-assignment analogue, pushed/popped in lockstep
		- see record_break_live()/merge_loop_exits(). '''
		self._confinement_depths.append( stack_depth )
		self._break_narrowed_stack.append( [] )
		self._break_live_stack.append( [] )

	def exit_loop( self ) -> tuple[list[dict[str,list[Variable]]],list[set[str]]]:
		''' pops and returns every narrowed-state/live-state snapshot
		record_break_narrowed()/record_break_live() collected while lowering
		this loop's own body (Phase 8) - the caller (lowering.py's
		_stmt_While/for-loop lowerers) merges these together with whatever
		the loop's own natural exit implies via merge_loop_exits(). '''
		self._confinement_depths.pop()
		return self._break_narrowed_stack.pop(), self._break_live_stack.pop()

	def record_break_narrowed( self ) -> None:
		''' called by lowering.py's _stmt_Break, BEFORE its own unwind_to()
		- captures the CURRENT _narrowed state (whatever was proven true
		at the exact point this break fires) into the innermost currently-
		lowering loop's own collection list. Deliberately NOT called by
		_stmt_Continue (a continue re-enters the loop, never reaches
		whatever follows it) or _stmt_Return (a return exits the FUNCTION,
		never reaches "after the loop" either - see the plan's own
		"Context" section on why this is correct, not an oversight). A
		no-op if called with no loop currently lowering (shouldn't happen
		given _stmt_Break's own "break outside a loop" guard, but matches
		this codebase's existing defensive style elsewhere). '''
		if self._break_narrowed_stack:
			self._break_narrowed_stack[-1].append( dict( self._narrowed ))

	def record_break_live( self ) -> None:
		''' the definite-assignment analogue of record_break_narrowed() -
		called from the same _stmt_Break call site, alongside it. Captures
		the CURRENT _live state at the exact point this break fires, into
		the innermost currently-lowering loop's own collection list -
		consumed by merge_loop_exits() below. Same "not for continue/return"
		reasoning as record_break_narrowed(). '''
		if self._break_live_stack:
			self._break_live_stack[-1].append( set( self._live ))

	def merge_loop_exits(
		self,
		natural_exit_narrowed: dict[str,list[Variable]] | None, break_narrowed: list[dict[str,list[Variable]]],
		natural_exit_live: set[str] | None = None, break_live: list[set[str]] = (),
	) -> None:
		''' called once a loop's own body has been fully lowered (after
		its own restore() back to the loop's entry snapshot) - reconciles
		every way execution can actually reach the code AFTER this loop:
		every `break` record_break_narrowed() collected (Phase 8), plus
		`natural_exit_narrowed` (the loop's own ordinary, condition-false
		exit - None when that path is PROVABLY unreachable, e.g. `while
		True:` with no other exit condition at all - see lowering.py's own
		call sites for how each loop kind computes this). Same soft-merge
		rule as _merge_narrowed_soft/_merge_case_narrowing (Phases 5-6): a
		name survives only if narrowed on EVERY candidate path, and its
		value is the UNION (dedup by identity) of what each one narrowed
		it to - not just an identical-only intersection. No candidates at
		all (an unconditional `while True:` with no break) means nothing
		reaches past the loop - empty is correct (dead code follows).

		natural_exit_live/break_live are the definite-assignment analogue,
		reconciled by plain set INTERSECTION across every candidate (same
		"AND, never an error here" rule as _merge_live_soft) rather than
		narrowed's union-of-possible-members - a name is live past the loop
		only if EVERY way of reaching here leaves it definitely assigned.

		No candidates at all (dead code follows) is handled differently
		here than for narrowed, on purpose: narrowed information can only
		ever cause an over-eager ACCEPT if kept, so wiping it is the safe
		default; but self._live gates whether a read is accepted AT ALL -
		wiping it to empty would make every single name in that dead code
		look uninitialized, including parameters/self (live from function
		entry, unconditionally). This compiler doesn't strip unreachable
		statements - they still get lowered structurally, same as any
		other statement (confirmed by a real repro: `while True: pass`
		with no break, followed by an ordinary `return <a parameter>`,
		fails "not initialized on all code branches" even though the
		parameter obviously IS - PLAN_GENERATORS.md's own generator
		rebuild hit this for real: its $$__next__ body always has more
		code after a user's own while-True-with-no-break loop, namely the
		generator's own tail). Leaving self._live untouched here (already
		restore()'d to the loop's own entry snapshot by every caller
		before this runs) is a safe over-approximation either way: if the
		code really is dead, an over-generous live set just lets reads
		that never execute through harmlessly; if it isn't (a resumable
		generator jumping back in), the loop's own entry liveness is
		exactly the right starting point, since nothing between loop
		entry and here could have invalidated it. '''
		candidates = list( break_narrowed )
		if natural_exit_narrowed is not None:
			candidates.append( natural_exit_narrowed )
		if not candidates:
			self._narrowed = {}
		else:
			merged: dict[str,list[Variable]] = dict( candidates[0] )
			for other in candidates[1:]:
				next_merged: dict[str,list[Variable]] = {}
				for name, members in merged.items():
					if name not in other:
						continue
					combined = list( members )
					for m in other[name]:
						if not any( m is existing for existing in combined ):
							combined.append( m )
					next_merged[name] = combined
				merged = next_merged
			self._narrowed = merged
		live_candidates = list( break_live )
		if natural_exit_live is not None:
			live_candidates.append( natural_exit_live )
		if not live_candidates:
			pass # dead code follows - leave self._live exactly as restore() already set it (the loop's own entry snapshot), see this method's own docstring for why that's the safe choice here, unlike self._narrowed above
		else:
			live_merged = set( live_candidates[0] )
			for other_live in live_candidates[1:]:
				live_merged &= other_live
			self._live = live_merged

	def enter_branch( self, stack_depth: int ) -> None:
		''' called by lowering.py's own _stmt_If, bracketing one if/elif/
		match-arm branch's own lowering - stack_depth is the SAME
		entry_snapshot.stack_depth restore() will truncate back to once this
		branch is fully lowered. Confirmed by a real repro (`return` as a
		match arm's own body: the arm's payload binding - e.g. `case
		Result.Ok(s): return s.byte_len()` - pushes a fresh RC entry that
		current_epilogue_label() would otherwise happily hand out as the
		return's own shared jump target, only for _stmt_If's own restore()
		to silently discard that entry before build_epilogue_ladder() ever
		runs, leaving a `goto` into a label that's never declared). See
		enter_loop()'s own comment for why this shares _confinement_depths
		rather than getting its own separate stack. '''
		self._confinement_depths.append( stack_depth )

	def exit_branch( self ) -> None:
		self._confinement_depths.pop()

	# --- union narrowing (compile-time only - see _narrowed's own comment) -

	def narrow( self, name: str, member: Variable ) -> None:
		''' called on entering a branch that's proven `name` (a TaggedUnion-
		typed binding) currently holds `member` (one of that union's own
		`.attributes` - its `.type` is the narrowed leaf, `.stem` the
		`v_<stem>` payload field) - e.g. the true-branch of a `type(x) is
		T`/`instanceof(x, T)` check, or a `match x: case T(x):` arm. Reads
		of `name` from here until this scope's own restore() (snapshot()
		captures/restore() reverts this exactly like _unchecked_results
		above - confined to the branch unless merge_if's own reconciliation
		lets it survive past it) get rewritten to read through the union's
		own payload - see lowering.py's _expr_Name/narrowed_member(). A
		single narrow() call is always exactly one member - merge_if's own
		soft-merge is what may later widen this to more than one. '''
		self._narrowed[name] = [ member ]

	def narrow_many( self, name: str, members: list[Variable] ) -> None:
		''' like narrow(), but `name` is proven to be ONE OF several possible
		members at once, not exactly one - e.g. a 3+-member T|U|None union's
		`if x is None: return` guard: the surviving (non-None) path knows x
		is T or U, never a single leaf. Reuses the SAME list-of-Variable
		_narrowed representation merge_if's own soft-merge already produces
		when two single-member narrow() calls on different branches
		disagree (_merge_narrowed_soft's own docstring) - snapshot()/
		restore()/merge_if() already handle an arbitrary-length list
		correctly with no changes, since that's exactly the shape they
		already reconcile today. narrowed_member() still only ever returns
		something for the fully-collapsed (len==1) case - a caller wanting
		this multi-member fact back reads narrowed_members() instead;
		_expr_Name's own read-rewrite uses it to build a fresh, properly
		re-tagged temp of the narrower union type on each read (see its own
		comment - unlike the O(1) single-field read the len==1 case gets,
		this is a real per-read dispatch, not free, but correct). '''
		assert len( members ) > 0
		self._narrowed[name] = list( members )

	def narrowed_members( self, name: str ) -> list[Variable] | None:
		''' the full set of members `name` is currently known to be ONE of,
		regardless of length - None if not narrowed at all. Unlike
		narrowed_member() (which only ever answers for the len==1 case),
		this is for a caller that can make USE of a >1-length fact -
		currently only _expr_Name's own multi-member dispatch-read path. '''
		return self._narrowed.get( name )

	def unnarrow( self, name: str ) -> None:
		''' called whenever `name` is reassigned (ordinary Assign/AnnAssign)
		- a fresh value invalidates whatever this name was previously
		proven to hold, same reasoning clear_result() already has for
		Result tracking. Safe to call on a name that was never narrowed.
		Also purges any LONGER key sharing `name` as a '::'-prefix
		(`name::attr`, `name::attr::attr2`, ...) - a field-chain narrowing
		key (type_resolver.py's _narrow_subject_key) is only ever valid
		while every hop between the reassigned target and the narrowed
		leaf still refers to the same object; reassigning a SHORTER prefix
		(`self.a = ...`) silently changes what a LONGER narrowed chain
		hanging off it (`self.a.b.c`) even refers to. One-directional:
		popping a longer key never touches its own shorter ancestor
		prefixes - only reassigning the shorter one invalidates the
		longer, never the reverse. '''
		self._narrowed.pop( name, None )
		prefix = f'{name}::'
		for stale in [ k for k in self._narrowed if k.startswith( prefix ) ]:
			self._narrowed.pop( stale, None )

	def narrowed_member( self, name: str ) -> Variable | None:
		''' the SINGLE member `name` is currently known to hold, or None -
		either because it isn't narrowed at all, or because it's narrowed
		to more than one still-possible member (merge_if's own soft-merge
		unioning two disagreeing-but-both-still-possible branches together
		- see its own docstring) - there's no one payload field to read in
		that case, so _expr_Name correctly falls back to treating it as
		unnarrowed for VALUE-reading purposes (safe, just not maximally
		optimized; the fact that some OTHER member is now excluded simply
		isn't exploited any further than this). '''
		members = self._narrowed.get( name )
		return members[0] if members is not None and len( members ) == 1 else None

	def narrowed_snapshot( self ) -> dict[str,list[Variable]]:
		''' a defensive copy for lowering.py to capture alongside bindings/
		unchecked_results() around if/loop orchestration (see merge_if()'s
		own true_end_narrowed/false_end_narrowed params) - mirrors
		unchecked_results()'s own identical purpose. '''
		return dict( self._narrowed )

	# --- definite-assignment ("liveness") tracking -------------------------

	def live_snapshot( self ) -> set[str]:
		''' a defensive copy for lowering.py to capture alongside bindings/
		narrowed/unchecked_results() around if/loop orchestration (see
		merge_if()'s own true_end_live/false_end_live params) - mirrors
		narrowed_snapshot()'s own identical purpose. '''
		return set( self._live )

	def is_live( self, name: str ) -> bool:
		''' True if `name` is definitely assigned on the CURRENT path -
		called from lowering.py's _expr_Name (every Name read) and
		_stmt_Delete, the two places a local's value is actually consumed.
		Type-independent, unlike self.bindings (RC-only) - a plain scalar/
		struct/enum local is tracked here even though it has no entry in
		bindings at all. '''
		return name in self._live

	def mark_live( self, name: str ) -> None:
		''' marks `name` live directly, bypassing assign()'s own RC/Result
		bookkeeping - for lowering.py plumbing that legitimately bypasses
		_cfg_assign by design (compiler-synthesized loop scaffolding via
		_declare_hidden_local/_bind_loop_target; @inline's own parameter/
		self binding, which is deliberately untracked by cfg.py at all -
		see _lower_inline_call's own "no _cfg_assign/incref here,
		deliberately" comment) but is still unconditionally bound at
		exactly the point this is called - same reasoning as parameters/
		self being seeded live from function entry in __init__/enter_self. '''
		self._live.add( name )

	def declare_exception_bind( self, dest: Variable ) -> None:
		''' registers an except-handler's bind (named `as e`, or the hidden
		raise_value_var every handler gets - see _stmt_Try) as a fresh OWNED
		RC binding, exactly like assign()'s own final "fresh, non-aliasing
		RC value" case (its `else: self._push(...)` branch) - but takes no
		`src` operand and emits no instructions, since the actual C
		assignment happens separately, in _emit_leaf_dispatch_case's own
		ir.ThrowLeaf handling (a raw move out of the Result's Err payload -
		already owned, no incref needed there either).
		Without this, `dest` was only ever mark_live()'d - never given an
		epilogue entry - so a caught exception the handler body doesn't
		re-raise/return/otherwise consume leaked unconditionally: neither a
		fallthrough decref nor merge_if's own branch-confined teardown ever
		fired for it, since bindings had no entry to reconcile. '''
		self._live.add( dest.stem )
		if rc_leaves( dest.type ):
			self._push( dest, dest.type, OwnState.OWNED )

	def unmark_live( self, name: str ) -> None:
		''' the inverse of mark_live() - lets a caller that temporarily
		marks a name live (inline parameter binding, which shadows-and-
		restores fn.names the same way) put the name's liveness back
		exactly as found afterward, rather than leaking a permanent
		liveness fact for a name that wasn't actually live before the
		splice (e.g. an outer local that happens to share a spliced
		function's own parameter name). '''
		self._live.discard( name )

	def set_live( self, live: set[str] ) -> None:
		''' overwrites the ENTIRE live set wholesale - used by @inline's
		own multi-statement splice (lowering.py's _lower_inline_call) to
		fully revert whatever liveness the splice's own pre-return
		statements produced, once the whole splice returns. Safe as a
		blunt full-revert (unlike merge_if/merge_loop_exits' own precise
		reconciliation) because every name the splice's body could mark
		live is either alpha-renamed to a name unique to that one splice
		(never referenced again by the caller) or a self/parameter binding
		that's deliberately reverted rather than leaking past the call -
		mirrors the provisional Function itself being single-use and
		discarded once the splice returns. '''
		self._live = set( live )

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
		true_end_narrowed: dict[str,list[Variable]] | None = None, false_end_narrowed: dict[str,list[Variable]] | None = None,
		true_end_live: set[str] = frozenset(), false_end_live: set[str] = frozenset(),
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
		surviving OWNED binding - both branches always push their OWN
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
		RC-only mechanism by design.

		true_end_narrowed/false_end_narrowed are the narrowing analogue -
		captured by lowering.py via narrowed_snapshot() at the same points
		it captures unchecked_results() - but reconciled DIFFERENTLY: a
		narrowed fact is an optimization/ergonomic convenience, not a
		correctness invariant like OwnState or unchecked-Result tracking,
		so disagreement is never an error, just silently dropped (see
		_merge_narrowed_soft's own docstring). No entry_narrowed parameter
		is needed - unlike bindings (which needs entry state to distinguish
		"already live" from "needs a fresh push") or results (whose own
		error path checks entry_results), a narrowed fact's survival past
		the join depends only on the two end-states.

		true_end_live/false_end_live are the definite-assignment analogue -
		captured by lowering.py via live_snapshot() at the same points it
		captures narrowed_snapshot() - reconciled the same soft way narrowing
		is (see _merge_live_soft), except by INTERSECTION rather than union:
		a name survives only if BOTH branches leave it definitely assigned,
		since ANY disagreement means a later read could hit the not-assigned
		path. Unlike bindings' own hard "exists on only one branch" error
		above, disagreement here is never a CompileError at the merge point -
		it's deferred to the actual read/del (_expr_Name/_stmt_Delete), which
		is where the user-facing "not initialized on all code branches"
		message belongs. This is deliberately independent of Bindings/
		rc_leaves - it covers every local, RC or not (see assign()'s own
		unconditional self._live.add()). '''
		true_instructions: list[ir.Instruction] = []
		false_instructions: list[ir.Instruction] = []
		removed: list[str] = []

		def reestablish( name: str, binding: _Binding, already_live: bool ) -> None:
			if binding.state == OwnState.OWNED:
				if already_live:
					self.bindings[name] = binding
				else:
					# key=name, not the default (binding.operand.stem) - for
					# an ordinary local these are identical, but for a
					# 'self.<attr>'-keyed construction binding (attr_assign's
					# own convention) operand.stem is just the bare attribute
					# name ('a'), not the tracking key ('self.a') -
					# defaulting silently re-keyed the reconciled entry under
					# the wrong name, so complete_construction()'s own
					# f'self.{attr.stem}' membership check never found it
					# again even though both branches genuinely set it
					self._push( binding.operand, binding.type, binding.state, key = name )
			else:
				self.bindings[name] = _Binding( operand = binding.operand, type = binding.type, state = binding.state, entry = None )

		if true_terminates or false_terminates:
			survivor = None
			survivor_results = None
			survivor_narrowed = None
			survivor_live = None
			if true_terminates and not false_terminates:
				survivor = false_end
				survivor_results = false_end_results
				survivor_narrowed = false_end_narrowed
				survivor_live = false_end_live
			elif false_terminates and not true_terminates:
				survivor = true_end
				survivor_results = true_end_results
				survivor_narrowed = true_end_narrowed
				survivor_live = true_end_live
			if survivor is not None:
				for name, binding in survivor.items():
					prior = entry_bindings.get( name )
					# an entry can be "already live" two ways: present (by
					# identity) in entry_bindings (the ordinary case), OR -
					# for a binding declared MID-BODY, after entry_bindings
					# was captured, so never eligible for the first check at
					# all - already sitting in self._epilogue_stack right
					# now because restore() (called by the caller before
					# this method runs) preserved it as a flag-guarded/
					# captured survivor (see restore()'s own "survivors"
					# comment). Missing this second case double-pushes a
					# BRAND NEW entry for the SAME already-tracked object -
					# the new one's own natural release fires unconditionally
					# (never flag-guarded itself), stacking on top of the
					# original's own still-live flag-guarded one - confirmed
					# by a real repro (a name manually compiler.decref()d only
					# on a terminating branch of a construct nested inside an
					# enclosing one - e.g. or_throw(mapper)'s own err_thunk
					# decref'ing its receiver, or a raise inside a nested
					# try's own handler decref'ing an outer local before
					# re-raising outward): a real double release/heap
					# corruption on the OTHER (surviving) path, which never
					# actually touched the entry at all.
					already_live = ( prior is not None and prior.entry is binding.entry ) or (
						binding.entry is not None and any( e is binding.entry for e in self._epilogue_stack )
					)
					if (
						not already_live and binding.state != OwnState.OWNED
						and prior is not None and prior.state == OwnState.OWNED and prior.entry is not None
						and any( e is prior.entry for e in self._epilogue_stack )
					):
						# same stale-entry hazard as the "both branches agree"
						# case below, just reached via the survivor path
						# instead: the SURVIVING branch alone (compiler.decref()/
						# del/move on a pre-if OWNED local, e.g. a discarded
						# match-arm's own subject release) already released
						# prior.entry via its own _neutralize() replacement,
						# never equal to it or to what restore() put back - the
						# terminating branch's own exit never touches prior.entry
						# at all (its own release, if any, runs through return_()/
						# unwind_to() independently). Neutralize the still-live
						# stack slot here, on the survivor's own path only - the
						# terminating branch already took its own exit, it never
						# reaches this join.
						_, neutralize_instructions = self._neutralize( prior.entry )
						if survivor is true_end:
							true_instructions += neutralize_instructions
						else:
							false_instructions += neutralize_instructions
					reestablish( name, binding, already_live )
				# a name present in entry_bindings but ABSENT from survivor was
				# explicitly del'd/moved on the surviving branch itself - but
				# self.bindings/self._epilogue_stack right now still reflect
				# the TERMINATING branch's own restore(entry_snapshot) (called
				# by lowering.py right before this method, once per branch,
				# unconditionally reverting to entry_bindings' own pristine,
				# uncancelled entry - see restore()'s own docstring), which
				# never even looked at what the OTHER, surviving branch did.
				# The loop above only ever RE-establishes what survivor still
				# has - it never visits a name survivor doesn't have at all,
				# so without this, the terminating branch's stale, uncancelled
				# entry leaks straight through untouched into the merged
				# state, and build_epilogue_ladder() releases it a SECOND
				# time (the surviving branch's own del/decref already emitted
				# its own explicit release) - confirmed by a real repro (a
				# local declared before a try, manually del'd on the try
				# body's own non-raising fall-through, with a covered raise
				# dispatching to a handler that unconditionally returns).
				for name, prior_binding in entry_bindings.items():
					if name in survivor or prior_binding.entry is None:
						continue
					current = self.bindings.get( name )
					if current is None or current.entry is not prior_binding.entry:
						continue
					_, neutralize_instructions = self._neutralize( current.entry )
					# any disarm instruction this needs must run on the
					# SURVIVOR's own path - that's the path the deletion
					# actually happened on; the terminating branch never
					# touched this name at all, and never reaches the join
					# to run anything anyway
					if survivor is true_end:
						true_instructions += neutralize_instructions
					else:
						false_instructions += neutralize_instructions
					del self.bindings[name]
			# both terminate -> nothing reaches the join at all (dead code
			# past here, same reasoning as the RC side above) - empty is the
			# safe choice; one terminates -> only the survivor's own results/
			# narrowed/live state can possibly reach the join
			self._unchecked_results = set( survivor_results ) if survivor_results is not None else set()
			self._narrowed = dict( survivor_narrowed ) if survivor_narrowed is not None else {}
			self._live = set( survivor_live ) if survivor_live is not None else set()
			return true_instructions, false_instructions, removed
		for name in set( true_end ) | set( false_end ):
			in_true = name in true_end
			in_false = name in false_end
			true_binding = true_end.get( name )
			false_binding = false_end.get( name )
			if in_true and in_false:
				assert true_binding is not None and false_binding is not None
				if true_binding.state != false_binding.state:
					# ALIVENESS agrees (the variable definitely exists both
					# ways - that's what got it here), only OWNERSHIP
					# disagrees. That's not the hazard the error below exists
					# for (a variable that might not exist at all) - it's the
					# ordinary "fill in a default when still borrowed" idiom
					# (`if x is None: x = Owned(...)`). Only OWNED-vs-BORROWED
					# is safe to reconcile this way (the value is valid either
					# way, only "do we own it" differs) - any OTHER
					# disagreement (MOVED involved, etc) stays a hard error,
					# unchanged.
					owning, borrowed = (
						( true_binding, false_binding ) if true_binding.state == OwnState.OWNED
						else ( false_binding, true_binding )
					)
					if not ( owning.state == OwnState.OWNED and borrowed.state == OwnState.BORROWED ):
						raise CompileError(
							f"{ctx}: {name!r} is in an indeterminate state after the if - "
							f"{true_binding.state.value} on one branch, {false_binding.state.value} on the other"
						)
					# synthesize a runtime flag so the eventual epilogue
					# decides AT RUNTIME whether to decref, instead of
					# requiring the compiler to know statically which branch
					# ran - reuses _mint_cancel_flag()'s own flag-guarded-
					# entry mechanism (_neutralize()'s "disarm instead of
					# statically cancel" pattern) rather than inventing a
					# second one; default-True-at-prologue already gives the
					# owning branch its correct value for free, only the
					# borrowed branch needs an explicit disarm
					flag = self._mint_cancel_flag()
					disarm = [ ir.Assign( dest = flag, src = ir.Const( type = flag.type, value = False )) ]
					if true_binding is borrowed:
						true_instructions += disarm
					else:
						false_instructions += disarm
					entry = self._push( owning.operand, owning.type, owning.state )
					entry.flag = flag
					continue
				prior = entry_bindings.get( name )
				# see the survivor-path's own identical comment above for why
				# a mid-body entry needs this second check too - a shared
				# entry already flag-guard-surviving in self._epilogue_stack
				# right now, not just one present in entry_bindings.
				already_live = (
					prior is not None
					and prior.entry is true_binding.entry
					and prior.entry is false_binding.entry
				) or (
					true_binding.entry is not None and true_binding.entry is false_binding.entry
					and any( e is true_binding.entry for e in self._epilogue_stack )
				)
				if (
					not already_live and true_binding.state != OwnState.OWNED
					and prior is not None and prior.state == OwnState.OWNED and prior.entry is not None
					and any( e is prior.entry for e in self._epilogue_stack )
				):
					# both branches independently released the SAME pre-if
					# OWNED binding (e.g. compiler.decref()/del/move on both
					# arms, reaching the same end state via two unrelated
					# instruction sequences - true_binding.entry/false_binding.
					# entry are each that branch's OWN _neutralize() replacement,
					# never equal to each other or to prior.entry, so
					# already_live's identity checks above can't see this).
					# restore() (called before this method runs, and again
					# between the two branches) always puts prior.entry itself
					# back in self._epilogue_stack, pristine/uncancelled, since
					# it's not flag-guarded - neither branch's own reestablish()
					# ever touches THAT object, only self.bindings. Left alone,
					# build_epilogue_ladder() walks the stack directly and
					# double-releases it. Neutralize it once here, for both
					# paths (either one may be the one that actually ran).
					_, neutralize_instructions = self._neutralize( prior.entry )
					true_instructions += neutralize_instructions
					false_instructions += list( neutralize_instructions )
				reestablish( name, true_binding, already_live )
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
			binding = true_binding if in_true else false_binding
			assert binding is not None
			decref = self._decref_instructions( binding.type, binding.operand ) if binding.state == OwnState.OWNED else []
			if in_true:
				true_instructions += decref
			else:
				false_instructions += decref
			removed.append( name )
		self._merge_results( entry_results, true_end_results, false_end_results, ctx )
		self._merge_narrowed_soft( true_end_narrowed, false_end_narrowed )
		self._merge_live_soft( true_end_live, false_end_live )
		return true_instructions, false_instructions, removed

	def _merge_narrowed_soft( self, true_end_narrowed: dict[str,list[Variable]] | None, false_end_narrowed: dict[str,list[Variable]] | None ) -> None:
		''' the narrowing analogue of _merge_results, for the neither-
		branch-terminates case (the terminates case is handled directly in
		merge_if() - only the survivor's own narrowed state matters there,
		same as for bindings/results). UNLIKE _merge_results (a hard
		CompileError on disagreement) or reestablish() (an indeterminate-
		state error), disagreement here is never an error: a name narrowed
		on only ONE branch is just dropped (nothing to merge), but a name
		narrowed on BOTH branches - even to DIFFERENT members - is UNIONED
		together rather than discarded: if the true branch proves x is int
		and the false branch (also surviving) proves x is str, code past
		the join genuinely could be either - "one of int|str" is real,
		actionable information (rules out every OTHER member, e.g. None),
		not something to throw away just because the two branches disagree
		about WHICH one. Identity comparison (`is`, not structural
		equality) for dedup - the same member Variable object is expected
		back from two independent resolutions of the same union/member
		pair (monomorphize_class's own spec.monomorphized caching, and a
		plain TaggedUnion's stable base.attributes list, both guarantee
		this - see the plan's own Trap 4 regression test). '''
		true_end_narrowed = true_end_narrowed or {}
		false_end_narrowed = false_end_narrowed or {}
		merged: dict[str,list[Variable]] = {}
		for name, true_members in true_end_narrowed.items():
			false_members = false_end_narrowed.get( name )
			if false_members is None:
				continue
			combined = list( true_members )
			for m in false_members:
				if not any( m is existing for existing in combined ):
					combined.append( m )
			merged[name] = combined
		self._narrowed = merged

	def _merge_live_soft( self, true_end_live: set[str], false_end_live: set[str] ) -> None:
		''' the definite-assignment analogue of _merge_narrowed_soft, for the
		neither-branch-terminates case (the terminates case is handled
		directly in merge_if() - only the survivor's own live state matters
		there). Unlike narrowing (union) or bindings (hard error),
		disagreement here is a plain, silent INTERSECTION: a name survives
		as live past the join only if BOTH branches leave it definitely
		assigned - live on only one branch means a path exists where it
		isn't, so it can't be trusted past the join, but that's never an
		error HERE, only at the eventual read/del (see merge_if()'s own
		docstring on true_end_live/false_end_live). '''
		self._live = true_end_live & false_end_live

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
			if binding.state == OwnState.OWNED:
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

	def find_promotable_loop_mismatches( self, entry_bindings: Bindings ) -> set[str]:
		''' loop_back_edge()'s own pre-check, for lowering.py's retry: which
		names hit the SAME safe BORROWED-entering/OWNED-by-back-edge shape
		merge_if() already reconciles for if/else branches (its own owning/
		borrowed check above). A loop body is lowered exactly ONCE and
		reused via the back edge (unlike an if's two independently-lowered
		branches), so this can't be reconciled after the fact the way
		merge_if's runtime flag does - a flag alone doesn't retroactively add
		the decref-before-overwrite each promoted reassignment site now
		needs on iteration 2+. lowering.py instead rolls back the whole
		failed attempt and re-lowers with these names pre-promoted via
		promote_borrowed_for_loop(), so every reassignment site sees the true
		steady-state entry ownership up front. '''
		back_edge = self.bindings
		promotable: set[str] = set()
		for name, entry_binding in entry_bindings.items():
			back_binding = back_edge.get( name )
			if back_binding is None or entry_binding.state == back_binding.state:
				continue
			if entry_binding.state == OwnState.BORROWED and back_binding.state == OwnState.OWNED:
				promotable.add( name )
		return promotable

	def promote_borrowed_for_loop( self, name: str ) -> list[ir.Instruction]:
		''' converts a currently-BORROWED binding to OWNED - a single
		explicit incref before the loop starts (not a per-iteration cost),
		conceptually the same "take my own copy" a copy[T] parameter's own
		prologue takes (_enter_parameter). Called once per name found by
		find_promotable_loop_mismatches(), right before lowering.py re-lowers
		the loop from its own start label - the returned instructions must
		be emitted there, before that label.

		survives_loop_restore=True on the pushed entry: this call happens
		WHILE lowering.py's own retry loop is still inside the loop body's
		lowering (before the loop's own restore() call further up the call
		stack), so without this the entry would be indistinguishable from an
		ordinary loop-body-local push and get silently truncated away by
		that same restore() once the (now-successful) retry attempt
		finishes - releasing the promoted incref never happens, a permanent
		+1 leak of the parameter's final value, confirmed via a real repro
		(grap.mpy's own dump_live_objects(), a for-loop reassigning a
		borrowed str parameter - the leak persisted even when the loop body
		never executed at all, i.e. an empty iterable). '''
		binding = self.bindings[name]
		assert binding.state == OwnState.BORROWED, f'promote_borrowed_for_loop({name!r}): binding is {binding.state}, not BORROWED'
		instructions = self._incref_instructions( binding.type, binding.operand )
		self._push( binding.operand, binding.type, OwnState.OWNED, key = name, survives_loop_restore = True )
		return instructions

	@property
	def cancel_flag_count( self ) -> int:
		''' len(self._cancel_flags) - lowering.py's loop-retry rollback uses
		this to snapshot/truncate cancel flags minted by a failed attempt
		(cancel_flags() itself always returns every flag ever minted, needed
		as-is by _emit_epilogue - see its own docstring). '''
		return len( self._cancel_flags )

	def truncate_cancel_flags( self, count: int ) -> None:
		''' drops every cancel flag minted since `count` (a prior
		cancel_flag_count) - a failed loop-lowering attempt being rolled back
		by lowering.py's retry must not leave its own now-unreferenced flags
		behind, or _emit_epilogue would still splice in a real, always-True,
		never-read local for each one (a guaranteed -Wunused-variable, or
		worse a dead store some compilers might not even tolerate silently). '''
		del self._cancel_flags[count:]

	def hard_restore( self, snap: _Snapshot ) -> None:
		''' like restore(), but discards EVERY entry pushed since the
		snapshot, including flag-guarded/captured ones restore() deliberately
		keeps alive (see its own docstring - a defer registered since the
		snapshot, or an early return already committed to one of its own
		labels). Only safe when the caller is about to fully re-lower that
		exact same source code from scratch, which re-registers a fresh
		replacement for anything genuinely still needed - lowering.py's own
		loop-ownership retry rollback is the one caller (see
		_lower_loop_body_with_ownership_retry): a defer statement textually
		inside the loop body gets re-registered on the retried attempt, so
		the ABANDONED attempt's own registration (and lowering.py's matching
		_defer_flags entry, separately truncated there) can simply be
		dropped rather than kept alive for a function epilogue that will
		never see the abandoned code again. Using ordinary restore() here
		would leave that stale entry referencing a flag lowering.py already
		rolled out of _defer_flags - a dangling reference.

		Also swaps every SURVIVING entry (index < stack_depth, i.e. declared
		BEFORE the loop) back to snap's own ORIGINAL object identity, exactly
		like restore() does for its own non-flag-guarded case (see its own
		docstring) - EXCEPT unconditionally, including flag-guarded/captured
		ones restore() deliberately leaves alone. restore()'s carve-out
		exists because a genuinely surviving sibling branch's own newly-
		minted flag has to stay live; hard_restore()'s whole point is the
		opposite - discarding EVERYTHING about the abandoned attempt,
		entry-internal state included, so an abandoned attempt's own
		.cancelled/.flag mutations must never survive either.

		Merely resetting .cancelled/.flag FIELDS on whatever's CURRENTLY
		sitting in each slot (a prior version of this) isn't enough:
		_neutralize()'s own static-cancel path never mutates a shared entry
		in place, it builds a REPLACEMENT and swaps it into
		self._epilogue_stack's own live slot instead (see its own
		docstring) - so a `del`/compiler.decref() on a pre-loop entry during
		the abandoned attempt leaves that REPLACEMENT object sitting in the
		stack, not snap's own original object. Field-only reset then leaves
		self.bindings (reset to snap's own original object above) and
		self._epilogue_stack pointing at two DIFFERENT objects for the same
		logical entry - exactly restore()'s own "identity divergence"
		hazard (see its own docstring). The retried attempt's own
		_neutralize() call then searches self._epilogue_stack for `e is
		entry` (entry = the bindings-side original) and never finds it, so
		its own replacement swap silently fails to happen - a stale,
		uid-suffixed-for-attempt-1 local from the abandoned attempt is left
		referenced by a later instruction instead of the retried attempt's
		own fresh one, a real "use of undeclared identifier" compile error -
		confirmed by a real repro (a pre-loop local `del`'d inside a loop
		body, early-returned past it too, combined with an unrelated
		borrowed-to-owned promotion that triggers this exact retry). A full
		identity swap sidesteps this for free - and, since it reverts the
		WHOLE object (not just cancelled), also covers the flag-guarded case
		that used to need its own separate field-by-field revert (a captured
		entry manually decref'd within the same abandoned attempt, whose
		.flag mutation must not survive either - the earlier "use of
		undeclared identifier __cancel_flag_0" repro this fixed the first
		time around).

		No .captured to revert here either - CFGState._captured_labels is
		whole-function and monotonic (see Epilogue's own docstring), never
		per-entry state to begin with, so there's nothing hard_restore()
		needs to do about it: the retried attempt re-lowers the identical
		source and, if it reaches the same capturing return, independently
		re-captures under its OWN freshly-minted entry.name (entries pushed
		after the snapshot are never reused - see the del below) - a stale
		name left behind by the abandoned attempt is simply never matched
		by any later live entry again. '''
		self.bindings = dict( snap.bindings )
		self._unchecked_results = set( snap.results )
		self._narrowed = dict( snap.narrowed )
		self._live = set( snap.live )
		# entry_objects alone isn't enough to undo a flag mutation: unlike
		# the static-cancel path (a REPLACEMENT object swapped in - see
		# _neutralize()'s own docstring), the flag-guarded path mutates
		# entry.flag directly IN PLACE on the very object entry_objects
		# holds a live reference to - so if the abandoned attempt captured
		# and then flag-guard-cancelled a pre-loop entry, entry_objects[i]
		# is retroactively "poisoned" by that mutation too (it's the SAME
		# object, not a snapshot of it). entry_cancelled/entry_flag were
		# captured as plain VALUES at snapshot time instead, immune to this -
		# resetting both the identity AND these fields together undoes both
		# hazards at once. Confirmed by a real repro: a pre-loop local
		# captured by an early return and `del`'d (minting a flag) during
		# the abandoned attempt left that same flag sitting on the
		# reverted entry, now referencing a Variable already truncated out
		# of _cancel_flags - the retried attempt reused it instead of
		# minting its own, producing a real, undeclared-at-prologue flag
		# variable used before its own (never-emitted) declaration.
		for i, ( orig_entry, cancelled, flag ) in enumerate( zip(
			snap.entry_objects, snap.entry_cancelled, snap.entry_flag,
		)):
			orig_entry.cancelled = cancelled
			orig_entry.flag = flag
			self._epilogue_stack[i] = orig_entry
		del self._epilogue_stack[snap.stack_depth:]

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

	def unwind_confined( self, floor: int, exclude: 'ir.Operand | None' = None ) -> list[ir.Instruction]:
		''' unwind_to()'s own exclusion-aware sibling - a covered `raise`/
		or_throw() dispatch (lowering.py's _dispatch_leaves_against_try_
		stack/ir.ThrowLeaf.epilogue) needs everything confined to the
		covering try's own body released before its goto into the handler,
		same as unwind_to() already does for break/continue leaving a loop
		- EXCEPT the raised value's own entry, if it's a plain Variable
		(`raise x`): that one's ownership is transferring INTO the handler's
		own bind, not ending here (see ir.ThrowLeaf.epilogue's own
		docstring). `floor` is the covering TryContext's own
		entry_stack_depth, not necessarily the innermost try on the stack -
		an outer handler catching a leaf the inner try doesn't cover has to
		unwind everything back to ITS OWN entry, past the inner try's own
		portion too. '''
		instructions: list[ir.Instruction] = []
		for entry in reversed( self._epilogue_stack[floor:] ):
			if entry.cancelled or entry.is_flag_guarded:
				continue
			if exclude is not None and entry.operand is exclude:
				continue
			instructions += self._decref_instructions( entry.type, entry.operand )
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
		if an errdefer entry is genuinely live here - see _replay().

		Bounded to the innermost active multi-statement @inline splice's own
		boundary_depth when one is active (self._inline_scope_stack - see
		push_inline_scope()) - "the ENTIRE current stack" above means the
		entire stack of the CURRENT scope (the splice, if inside one), never
		reaching down into the caller's (or an outer splice's) own older,
		still-pending entries: those aren't this call's to unwind, they'll
		get their own replay whenever THEIR OWN scope eventually exits. '''
		self.untrack_temp( returned_operand )
		instructions: list[ir.Instruction] = []
		floor = self._inline_scope_stack[-1].boundary_depth if self._inline_scope_stack else 0
		for entry in reversed( self._epilogue_stack[floor:] ):
			if entry.cancelled:
				continue
			if returned_operand is not None and entry.operand is returned_operand:
				continue
			instructions += self._replay( entry, get_is_err_check )
		return instructions

	def has_live_entry( self, operand: ir.Operand | None ) -> bool:
		''' whether `operand`'s identity matches a still-live (non-cancelled)
		epilogue entry - the exact identity test current_epilogue_label()/
		return_() already use to recognize "this really is an ownership move,
		its own eventual decref is already accounted for by matching/skipping
		this entry" rather than a borrow. Used by lowering.py's _stmt_Return
		to decide whether an ALIASING return expression (self.lowering.
		_is_aliasing_expr) needs its own Incref before being handed to the
		caller: an OWNED local or a copy[T]/move[T] parameter has a live
		entry here (a genuine move, no Incref needed - the source's own
		decref is what's being skipped), but a BORROWED parameter/self (never
		pushed - see _enter_parameter()'s own BORROWED branch) and an
		attribute/tuple-element read (a fresh GetAttr temp, never pushed
		either - fields are never separately tracked, see field_value()'s own
		comment) both have NO entry at all here even though _is_aliasing_expr
		says they alias existing state - those need a real Incref, since
		nothing downstream is skipping a decref on their behalf. '''
		return operand is not None and any(
			not entry.cancelled and entry.operand is operand for entry in self._epilogue_stack
		)

	def current_epilogue_label( self, returned_operand: ir.Operand | None = None, *, mark_captured: bool = True ) -> str | None:
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
		body or an if/elif/match-arm branch still being lowered (pushed at
		or after the SHALLOWEST active enter_loop()/enter_branch() depth,
		and not flag-guarded - see restore()'s own comment on why a plain RC
		entry doesn't survive a loop body's or a branch's own exit but a
		defer/errdefer one does): that entry's own label would never
		actually get emitted anywhere (build_epilogue_ladder() only ever
		walks the stack that SURVIVES to the function's real end - restore()
		silently drops confined entries once the loop body/branch lowering
		that pushed them finishes, well before then), so jumping into it
		here would be a dangling reference to a label that's never declared -
		confirmed by real repros, not just reasoning (an early return from
		inside a while loop, past a locally-declared RC value, nested inside
		a with-block; a `return` as a match arm's own body, past that arm's
		own payload binding). return_()'s own full, inline unwind (which
		walks the ENTIRE stack directly, needing no label of its own at all)
		is the only correct option for a confined entry, exactly like the
		returned-operand case just above.

		The shallowest (min(), not just the innermost/top-of-stack) active
		depth is what matters, not just the most-recently-entered scope: an
		entry pushed inside an OUTER not-yet-merged branch/loop, before some
		INNER branch/loop was even entered, is still doomed by the OUTER
		scope's own eventual restore() even though it predates the inner
		one - using only the top of the stack would miss exactly that
		entry and hand out a label for it anyway.

		While a multi-statement @inline splice is active (self.
		_inline_scope_stack non-empty - see push_inline_scope()), this never
		returns None purely for "nothing's left pending": the innermost
		scope's own boundary_depth acts as a hard floor the walk below never
		crosses, falling back to that scope's own label instead of either
		returning None or continuing into the caller's own older entries.
		The confinement-floor and returned-operand-identity None-cases above
		still apply exactly as before, scoped to the splice's own portion of
		the stack the same way they'd apply to a real function's - see
		return_()'s own matching comment for why THOSE cases still need a
		self-contained inline unwind rather than the shared label even
		inside a splice.

		mark_captured=False: the caller only wants to know WHETHER there's
		still something pending (is not None), not to actually commit a
		`goto` using the returned name - e.g. lower_function's own "does
		the fall-off-the-end path need build_epilogue_ladder() at all"
		probe, which relies on placing the ladder immediately after the
		function's own body (pure fallthrough is already correct there, no
		goto needed). The normal call already marks entry.name captured
		(self._captured_labels) unconditionally on the assumption its
		caller is about to emit a real jump to entry.name; a probe that
		never does that would otherwise spuriously mark an entry captured
		with no goto anywhere actually referencing it - a real, confirmed
		-Wunused-label/C4102 (build_epilogue_ladder() gates the Label
		itself on entry.name's own captured-ness - see its own docstring). '''
		if returned_operand is not None and any(
			not entry.cancelled and entry.operand is returned_operand
			for entry in self._epilogue_stack
		):
			return None
		# None (not 0) when no loop/branch is currently being lowered - the
		# whole confinement check below must be a no-op then (every entry is
		# function-scoped), not "confined below index 0" (which would
		# wrongly treat EVERY entry as confined, since every valid index
		# is >= 0)
		confinement_floor = min( self._confinement_depths ) if self._confinement_depths else None
		# the innermost active multi-statement @inline splice, if any (see
		# push_inline_scope()'s own comment) - entries BELOW its own
		# boundary_depth belong to the CALLER (or an outer splice), and must
		# never be inspected here, let alone handed back as this return's own
		# jump target: that's exactly the bug that used to make .or_return()/
		# checked arithmetic unsupported inside a spliced body (it would
		# otherwise silently jump into the caller's own real epilogue,
		# short-circuiting the caller's own remaining code). Once the walk
		# below reaches the scope's own boundary with nothing shallower
		# eligible, its own label is always a valid fallback target - unlike
		# the plain "nothing pending" case (a bare ir.Return is fine there),
		# a splice never gets to just fall through to a caller-level ir.
		# Return; it always needs a real, local landing point
		inline_scope = self._inline_scope_stack[-1] if self._inline_scope_stack else None
		for i, entry in reversed( list( enumerate( self._epilogue_stack ))):
			if inline_scope is not None and i < inline_scope.boundary_depth:
				if mark_captured:
					inline_scope.captured = True
				return inline_scope.label
			if entry.cancelled:
				continue
			if confinement_floor is not None and not entry.is_flag_guarded and i >= confinement_floor:
				return None
			# used_shared_epilogue_label()'s flag is scoped to THIS function's
			# own real closing-brace ladder specifically - only set when
			# inline_scope is None (this entry belongs to the function
			# itself, not to some still-open splice's own segment of the
			# stack: had it been the latter, the `i < boundary_depth` branch
			# above would already have returned first). A splice-local
			# entry's own label is consumed by build_inline_scope_ladder()
			# instead, fully popped off the stack by the time this function's
			# own closing brace is ever reached - marking the flag for it
			# here would wrongly make used_shared_epilogue_label() report
			# true for the OUTER function even though nothing of ITS OWN is
			# actually pending, forcing a spurious extra Return/FuncEnd
			# (confirmed via a real regression: an @inline splice's own
			# internal early return/.or_return() must never manufacture a
			# second real ir.Return in the CALLER).
			if mark_captured:
				if inline_scope is None:
					self._any_shared_label_used = True
				# this jump is now committed to entry.name regardless of what
				# happens to `entry` afterward - a LATER manually_decreffed()/
				# deleted()/move() on this same entry must not silently turn
				# this already-emitted goto into a no-op landing (see their
				# shared _neutralize() helper)
				self._captured_labels.add( entry.name )
			return entry.name
		if inline_scope is not None:
			if mark_captured:
				inline_scope.captured = True
			return inline_scope.label
		return None

	def current_epilogue_label_for_construction_err(
		self, returned_operand: ir.Operand | None,
	) -> tuple[str | None, list[ir.Instruction]]:
		''' current_epilogue_label()'s counterpart for a fallible __init__'s
		own Err-path return (lowering.py's _stmt_Return, construction_err_
		path) - a self.<attr> entry (attr_assign()/complete_base_
		construction(), entry.is_construction_attr) can never be handed out
		as a shared label's own JUMP TARGET: complete_construction() cancels
		every required attribute WITHOUT a per-jump-site record (Epilogue.
		cancelled is one mutable flag, not a snapshot - see its own
		docstring), and an attribute has no runtime flag of its own the way
		defer/errdefer does, so treating an attribute's OWN rung as
		reachable via `goto` is only sound for returns strictly AFTER its
		assignment - never provably true once more than one Err return
		exists (this class's own fix #3, the ORIGINAL regression this whole
		construction_err_path mechanism exists to prevent). Every live
		attribute entry is therefore always decref'd INLINE, right here,
		regardless of where it sits in the stack.

		Everything else pending (defer/errdefer, or a plain non-attribute
		RC local) is never touched by complete_construction() at all -
		exactly as safe to route through the ordinary shared epilogue
		label as in a non-__init__ function, and safe to let an
		attribute's own (labelless, unreachable-via-goto) rung sit ABOVE
		OR BELOW it in the stack: build_epilogue_ladder()/build_inline_
		scope_ladder() unconditionally skip replaying is_construction_attr
		entries (see their own comments) - never just because .cancelled
		happens to be set by ladder-build time (relying on that would
		reintroduce a narrower version of the exact same hazard for a
		fallible __init__ with no textual success-shaped return at all,
		where complete_construction() never runs and an attribute would
		stay .cancelled=False forever) - so an attribute's rung is a
		guaranteed no-op wherever a shared jump happens to fall through
		it, and this method never needs to inspect stack ORDER at all,
		only liveness. Returns (label, inline instructions to emit before
		jumping to it) - label is None when nothing needs a shared jump
		(caller falls back to a plain Return), or when current_epilogue_
		label()'s own bail-outs apply (returned_operand aliasing a live
		entry anywhere in the stack, or a confined loop/branch entry -
		both rare enough in a constructor to not warrant a partial-inline
		treatment here); the caller then falls back to plain return_() for
		the WHOLE stack, exactly as before this method existed - the
		returned instruction list is always [] alongside a None label,
		nothing to double-emit.

		Pure classification first (which indices need an inline decref, and
		whether a shared label is even reachable), THEN - only once that's
		fully decided - a second pass that actually calls _decref_
		instructions() for just the entries being kept. Not merged into one
		pass: _decref_instructions() can mint fresh temps/labels for a
		union-typed attribute (_extract_payload()'s own tag-gated path),
		which land as real DeclareTemp instructions in the CURRENT
		instruction stream as an unconditional side effect the moment
		they're minted (lowering.py's own _new_temp(), passed in as this
		class's new_temp callback) - calling it speculatively for an
		attribute later discarded by a bail-out below would leak a stray,
		never-populated temp declaration into the emitted C even though
		this method's own contract is "never emits anything by itself". '''
		if returned_operand is not None and any(
			not entry.cancelled and entry.operand is returned_operand for entry in self._epilogue_stack
		):
			return None, []
		confinement_floor = min( self._confinement_depths ) if self._confinement_depths else None
		inline_scope = self._inline_scope_stack[-1] if self._inline_scope_stack else None
		floor = inline_scope.boundary_depth if inline_scope is not None else 0
		attr_indices: list[int] = []
		candidate_index: int | None = None
		for i, entry in reversed( list( enumerate( self._epilogue_stack ))):
			if i < floor:
				break
			if entry.cancelled:
				continue
			if confinement_floor is not None and not entry.is_flag_guarded and i >= confinement_floor:
				return None, []
			if entry.is_construction_attr:
				attr_indices.append( i )
				continue
			if candidate_index is None:
				candidate_index = i
		inline_instructions: list[ir.Instruction] = []
		for i in attr_indices:
			entry = self._epilogue_stack[i]
			inline_instructions += self._decref_instructions( entry.type, entry.operand )
		if candidate_index is None:
			if inline_scope is not None:
				inline_scope.captured = True
				return inline_scope.label, inline_instructions
			return None, inline_instructions
		entry = self._epilogue_stack[candidate_index]
		if inline_scope is None:
			self._any_shared_label_used = True
		self._captured_labels.add( entry.name )
		return entry.name, inline_instructions

	def used_shared_epilogue_label( self ) -> bool:
		''' whether some ALREADY-LOWERED return/OrJump actually committed a
		jump into one of this function's own shared epilogue labels (i.e.
		current_epilogue_label() returned non-None at least once so far) -
		DELIBERATELY not the same question current_epilogue_label() answers
		for a hypothetical NEW return right here (which correctly skips a
		cancelled entry, since a fresh return needs no unwind through
		something already consumed).

		Needed because an entry a return jumped into WHILE STILL LIVE can
		since have been cancelled (move()/compiler.decref(x)/del - see
		move()'s own comment) by the time lowering reaches the function's
		own closing brace: current_epilogue_label() then correctly reports
		"nothing NEW needs to unwind here" (None), but that EARLIER goto
		still needs its label actually built by build_epilogue_ladder()
		(which emits one per entry regardless of cancelled, per its own
		docstring), or it's left dangling. Tracking "was a real label ever
		handed out" (rather than just "is the stack non-empty") avoids
		over-triggering for a case that looks superficially similar but
		isn't: a nonfallible __init__ whose only entries are attributes
		complete_construction() cancels on its own single, implicit,
		success-only return path - current_epilogue_label() never once
		returns non-None there (complete_construction() always cancels
		before that return's own current_epilogue_label() call, per its own
		docstring), so no dead, never-jumped-to Label is emitted for it.

		Confirmed via a real repro: `out = bytearray(n); if cond: return
		Result.Err(...); return Result.Ok(bytes.from_bytearray(move(out)))` -
		the earlier `return` DOES call current_epilogue_label() while out's
		entry is still live (setting this flag), then move() cancels that
		same entry - without this check, the goto that earlier return
		already committed to would go undeclared - "use of undeclared
		label" at the C level, a real, general, silent miscompile. '''
		return self._any_shared_label_used

	def build_epilogue_ladder(
		self, get_is_err_check: 'Callable[[],tuple[list[ir.Instruction],ir.Operand]] | None' = None,
	) -> list[ir.Instruction]:
		''' the shared unwind sequence every return that used
		current_epilogue_label() (and the function's own fall-off-the-end)
		jumps into - one Label (only when entry.name is captured - see
		below) + that entry's own still-live replay per pending entry (RC
		Decref, or a flag-guarded defer/errdefer replay - see _replay()),
		deepest (most-recently-pushed) first, each falling straight through
		into the next with no Jump needed. Cancelled entries still get
		their own Label whenever captured (current_epilogue_label() can
		still point straight at one - see its own comment), just no
		instructions. An entry current_epilogue_label() never actually
		handed out as a live jump target (entry.name never added to
		self._captured_labels - no return anywhere in the function needed
		to unwind from exactly that depth) gets NO Label
		either: every OTHER rung still reaches it purely by falling
		through from the one above, so a Label with nothing branching to it
		would be a real, always-on -Wunused-label/C4102 on every compiler.
		Callers append their own final ir.Return - cfg.py has no notion of
		a function's return type or return-value slot.

		entry.is_construction_attr is skipped UNCONDITIONALLY here (never
		just because .cancelled happens to be set) - a self.<attr> entry
		is NEVER a valid ladder rung, full stop: current_epilogue_label_
		for_construction_err() never hands one out as a jump target, so its
		own rung only exists as fallthrough scenery for some OTHER entry's
		jump, and it was already decref'd inline, right at whichever Err
		return actually needed it, by that same method. Gating this on
		.cancelled instead (relying on complete_construction() having
		already flipped it by the time this ladder is built) would still
		be correct for the common case, but not for a fallible __init__
		with no textual success-shaped return at all: complete_
		construction() then never runs, .cancelled stays False forever,
		and this same rung - reached by an unrelated entry's shared jump
		simply falling through it - would double-decref an attribute
		some earlier Err return already handled. '''
		instructions: list[ir.Instruction] = []
		for entry in reversed( self._epilogue_stack ):
			if entry.name in self._captured_labels: # see this method's own docstring
				instructions.append( ir.Label( name = entry.name ))
			if not entry.cancelled and not entry.is_construction_attr:
				instructions += self._replay( entry, get_is_err_check )
		return instructions

	def build_inline_scope_ladder(
		self, get_is_err_check: 'Callable[[],tuple[list[ir.Instruction],ir.Operand]] | None' = None,
	) -> list[ir.Instruction]:
		''' the multi-statement @inline splice analogue of build_epilogue_
		ladder() - same shape (one Label + still-live replay per pending
		entry, deepest first), bounded to just the innermost active
		InlineScope's own segment of the stack (self._epilogue_stack[scope.
		boundary_depth:]) instead of the whole thing - entries belonging to
		the caller (or an outer splice) are never touched, exactly like
		current_epilogue_label()/return_() are now scoped (see their own
		comments). Called once, right after lowering.py finishes lowering a
		splice's own pre-return statements - the caller (lowering.py) is
		responsible for emitting the scope's own leading Label (push_inline_
		scope()'s own return value) itself first; this only emits what
		follows it. Truncates _epilogue_stack back to boundary_depth once
		built - this scope's own entries are now fully consumed, whether by
		this ladder or by an earlier inline-unwind return_() call reached
		during the splice itself (those already removed nothing from the
		stack themselves - see return_()'s own docstring - so this is the
		one place a splice's own entries actually get popped). '''
		scope = self._inline_scope_stack[-1]
		instructions: list[ir.Instruction] = []
		for entry in reversed( self._epilogue_stack[scope.boundary_depth:] ):
			# see build_epilogue_ladder()'s own identical comment, including
			# on why is_construction_attr is skipped unconditionally
			if entry.name in self._captured_labels:
				instructions.append( ir.Label( name = entry.name ))
			if not entry.cancelled and not entry.is_construction_attr:
				instructions += self._replay( entry, get_is_err_check )
		del self._epilogue_stack[scope.boundary_depth:]
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
		# entry.type is set only for a plain RC entry _neutralize() converted
		# to flag-guarded on the fly (see its own docstring) - entry.
		# instructions is always empty for those (never populated the way a
		# real defer/errdefer body is), so regenerating fresh here (like the
		# unguarded branch above) is required, not just consistent
		instructions += self._decref_instructions( entry.type, entry.operand ) if entry.type is not None else entry.instructions
		instructions.append( ir.Label( name = skip_label ))
		return instructions

	# --- Incref/Decref emission, union-aware ------------------------------------

	def incref( self, t: Type, operand: ir.Operand ) -> list[ir.Instruction]:
		''' public entry point for a caller that just extracted/duplicated a
		value from somewhere else (e.g. a checked-Result payload via
		or_return()/checked arithmetic's own OrReturn/OrJump/Unwrap extraction
		- lowering.py's _consume_checked_result) and needs to give it its own
		fresh +1 reference of its own - mirroring what an ordinary Assign's
		own is_alias branch already does internally. A no-op (empty list) for
		a non-RC t, same as everywhere else RC emission is gated - safe to
		call unconditionally regardless of whether t actually turns out RC. '''
		return self._incref_instructions( t, operand )

	def decref( self, t: Type, operand: ir.Operand ) -> list[ir.Instruction]:
		''' public entry point mirroring incref() above, for a caller that's
		fully done with a value and needs to release it explicitly (e.g.
		compiler.decref(x) - lowering.py's _lower_compiler_decref). Union-
		aware exactly like incref(): a TaggedUnion operand gets the real
		tag-gated release sequence (_tag_gated_refcount_instructions), not a
		bare pointer release - calling ir.Decref directly here would be
		correct only for a plain RC pointer, not a tag+data value struct. '''
		return self._decref_instructions( t, operand )

	def _incref_instructions( self, t: Type, operand: ir.Operand ) -> list[ir.Instruction]:
		return self._refcount_instructions( t, operand, ir.Incref )

	def _decref_instructions( self, t: Type, operand: ir.Operand ) -> list[ir.Instruction]:
		return self._refcount_instructions( t, operand, ir.Decref )

	def _refcount_instructions( self, t: Type, operand: ir.Operand, op: 'type[ir.Incref]|type[ir.Decref]' ) -> list[ir.Instruction]:
		# resolve a bare Specialization (e.g. Result[str,SomeError]) to its
		# real, monomorphized ClassLike FIRST - this function's own codegen
		# choices below (plain-pointer Incref/Decref vs. tag-gated payload
		# extraction) and t.leaves()/_extract_payload's own union_storage()
		# lookup all need t's REAL, substituted shape (str/SomeError, not
		# Result's own abstract, still-bare-TypeVar T/E) to be correct.
		# Without this, `isinstance(t, TaggedUnion)` is FALSE for a
		# Specialization wrapper (even though rc_leaves(t), fixed separately
		# to substitute leaves for exactly this reason, correctly says it
		# has RC leaves) - taking the plain-RC-pointer branch below on a
		# plain VALUE struct (a Result is never itself heap-allocated/
		# pointer-shaped) emits a real Incref/Decref that the emitter turns
		# into invalid `(a_value_struct)->$header` C. Cheap even called
		# often: memoized by the Monomorphizer (spec.monomorphized), and a
		# no-op passthrough for anything that isn't a Specialization at all
		# (see resolve_type's own default/wiring).
		t = self._resolve_type( t )
		leaves = rc_leaves( t )
		if not leaves:
			return []
		if not isinstance( t, TaggedUnion ):
			return [ op( value = operand ) ]
		all_leaves = t.leaves()
		if len( leaves ) == len( all_leaves ) and all( _is_direct_pointer_rc( leaf ) for leaf in leaves ):
			# every member is RC AND a bare pointer (never a nested union,
			# whose own runtime shape is a tag+data value struct, not a
			# pointer at all - see _is_direct_pointer_rc) - no tag check
			# needed: they're all the same union payload memory, and every
			# RC POINTER type shares the same header layout/offset, so any
			# one member's accessor works. A nested-union member (now also
			# reported by rc_leaves as "RC", per is_rc()'s own new branch)
			# forces the tag-gated path below even when every leaf is
			# otherwise RC, since there's no single shared pointer layout
			# to read through uniformly in that case
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
			# recurse rather than assuming `payload` is always a bare RC
			# pointer op() can retain/release directly - a NESTED union
			# member's own payload is itself a tag+data value struct
			# (_refcount_instructions' own isinstance(t, TaggedUnion) check
			# correctly routes it into its own, further tag-gated dispatch;
			# a direct RCClass/TupleType member instead degenerates straight
			# back to the plain `[op(value=payload)]` this replaces, so this
			# is behavior-preserving for every leaf shape already handled
			# before this fix)
			instructions += self._refcount_instructions( member.type, payload, op )
			instructions.append( ir.Jump( target = end_label ))
			instructions.append( ir.Label( name = next_label ))
		instructions.append( ir.Label( name = end_label ))
		return instructions

	# --- assignment: fresh / aliasing / replace, all in one -----------------

	def assign( self, dest: Variable, src: ir.Operand, *, is_alias: bool, track_result: bool = True, borrow: bool = False ) -> list[ir.Instruction]:
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
		inspect itself (the match-statement subject temp - see its own
		lowering.py call site).

		borrow=True registers dest as a non-owning alias (BORROWED, like an
		ordinary parameter - see _enter_parameter) instead of taking out its
		own Incref'd copy: used only for the match-statement subject temp
		when the subject itself is a bare Name (is_alias=True) - the ORIGINAL
		name already owns a live reference for at least the duration of the
		match (it's function-scoped, never dropped mid-statement), so the
		synthesized __match_subj_N alias doesn't need an independent
		Incref/epilogue-Decref pair of its own; each arm's own bind (a real,
		separately-tracked extraction) already takes whatever retain IT
		needs before __match_subj_N is ever read again. Skipping this
		previously emitted a real, always-unbalanced-until-function-exit
		Incref with no corresponding use - harmless by construction (release
		of the ORIGINAL still nets it to zero eventually) but inflated every
		compiler.refcount() read taken inside a match arm by one, and every
		match execution paid for a wholly unneeded retain/release pair. '''
		self._live.add( dest.stem ) # unconditional, before every early-return below (borrow/rc_leaves) - liveness is type-independent, unlike bindings/rc_leaves themselves
		if dest.stem in self._unchecked_results:
			raise CompileError(
				f"Result value {dest.stem!r} is discarded - it was never inspected: "
				f"use .is_ok(), .is_err(), .or_return(), .unwrap(msg), or match"
			)
		if track_result and is_result_type( dest.type ):
			self.track_result( dest.stem )
		else:
			self.clear_result( dest.stem )
		if borrow:
			self.bindings[dest.stem] = _Binding( operand = dest, type = dest.type, state = OwnState.BORROWED, entry = None )
			return []
		if not rc_leaves( dest.type ):
			return []
		instructions: list[ir.Instruction] = []
		if is_alias:
			# PLAN_THREAD_SAFE_SHARED_STATE.md Part A: reading a global's
			# CURRENT value (src here) needs the same protection writing one
			# does - _incref_instructions shares _refcount_instructions with
			# _decref_instructions (see that method's own comment), so for a
			# union-typed global (e.g. ZoneInfo|None) this ALSO isn't a bare
			# ir.Incref, it's the same tag-check+extract+retain sequence,
			# which only whoever is CONSTRUCTING it (here) can bound.
			#
			# Opens the critical section here (Acquire only - the matching
			# Release is emitted by lowering.py's _cfg_assign, AFTER the
			# trailing ir.Assign it emits, not here) for the SAME reason the
			# write side's Decref+Assign have to share one critical section:
			# `ir.Assign(dest=b, src=X)` is ITS OWN, SEPARATE textual read of
			# X in the generated C (`b = X;`), not a reuse of whatever value
			# the Incref above just retained - confirmed the hard way, via a
			# real crash under concurrent stress: retain_object(X) protected
			# the INCREF, but the Assign's own independent read of X, right
			# after the lock was released, could observe a DIFFERENT object
			# than the one just retained if a writer swapped X in between -
			# leaking the retained object and under-retaining the one
			# actually bound to `b`. Emitted UNCONDITIONALLY whenever src is
			# simply a global (not gated on src.reassigned_outside_init,
			# unlike the write side's own check in the `dest.is_global`
			# branch below) - a READ can be lowered before the ONE write
			# that will eventually mark this global as needing protection is
			# (functions are lowered off a work queue in whatever order
			# they're scheduled, not necessarily the order a human reads the
			# source in), so reassigned_outside_init's FINAL value isn't
			# reliably known yet at this point. emitter_c.py defers the real
			# "does this end up mattering" decision to emission time instead
			# (after every function has been lowered, so the fact is
			# complete) - these markers become pure no-ops there for a
			# global that turns out to never actually be reassigned
			# anywhere.
			if self.is_fresh_temp( src ):
				# src LOOKS aliasing from the source AST's own shape (is_alias
				# reflects that a Name/Attribute node was read, not what it
				# actually lowered to), but it's already a freshly-owned value
				# here - a narrowed read of a protected global performs its
				# own protected retain up front and registers the result via
				# fresh_temp() (see lowering.py's _expr_Name), specifically so
				# this branch doesn't double-own it. Untrack instead of
				# increffing again, same as the plain-Temp ownership-transfer
				# branch below - this is sound for every OTHER existing caller
				# too, not just this new one: is_fresh_temp can only be True
				# for a Call/Allocate-produced (or now, protected-narrowed-
				# read-produced) temp, and nothing already relies on
				# increffing one of those a second time here.
				self._temp_states.pop( src.id, None )
			else:
				# PLAN_THREAD_SAFE_SHARED_STATE.md Part A: reading a global's
				# CURRENT value (src here) needs the same protection writing one
				# does - _incref_instructions shares _refcount_instructions with
				# _decref_instructions (see that method's own comment), so for a
				# union-typed global (e.g. ZoneInfo|None) this ALSO isn't a bare
				# ir.Incref, it's the same tag-check+extract+retain sequence,
				# which only whoever is CONSTRUCTING it (here) can bound.
				#
				# Opens the critical section here (Acquire only - the matching
				# Release is emitted by lowering.py's _cfg_assign, AFTER the
				# trailing ir.Assign it emits, not here) for the SAME reason the
				# write side's Decref+Assign have to share one critical section:
				# `ir.Assign(dest=b, src=X)` is ITS OWN, SEPARATE textual read of
				# X in the generated C (`b = X;`), not a reuse of whatever value
				# the Incref above just retained - confirmed the hard way, via a
				# real crash under concurrent stress: retain_object(X) protected
				# the INCREF, but the Assign's own independent read of X, right
				# after the lock was released, could observe a DIFFERENT object
				# than the one just retained if a writer swapped X in between -
				# leaking the retained object and under-retaining the one
				# actually bound to `b`. Emitted UNCONDITIONALLY whenever src is
				# simply a global (not gated on src.reassigned_outside_init,
				# unlike the write side's own check in the `dest.is_global`
				# branch below) - a READ can be lowered before the ONE write
				# that will eventually mark this global as needing protection is
				# (functions are lowered off a work queue in whatever order
				# they're scheduled, not necessarily the order a human reads the
				# source in), so reassigned_outside_init's FINAL value isn't
				# reliably known yet at this point. emitter_c.py defers the real
				# "does this end up mattering" decision to emission time instead
				# (after every function has been lowered, so the fact is
				# complete) - these markers become pure no-ops there for a
				# global that turns out to never actually be reassigned
				# anywhere.
				if isinstance( src, Variable ) and src.is_global:
					instructions.append( ir.AcquireGlobalLock( var = src, exclusive = False )) # Cost mitigation #4: a read, safe to run concurrently with other readers
				instructions += self._incref_instructions( dest.type, src ) # bump the new value first - safe even if src and dest already alias the same object
		elif isinstance( src, ir.Temp ):
			# ownership transfers from the temp's own (momentary) tracking
			# into dest, not a second independent owner - untrack it so its
			# own eventual DeleteTemp doesn't ALSO decref the same object
			self._temp_states.pop( src.id, None )
		elif isinstance( src, Variable ):
			# is_alias=False but src is a full Variable (not ir.Temp) - a
			# lowering helper handed back an already-fresh owned value bound
			# to a real Variable rather than an ir.Temp (e.g. @inline
			# splicing's `return self`, str(s)'s own identity-conversion
			# repro: _incref_aliasing_return already gave src its own
			# independent +1). src keeps its own binding/epilogue entry -
			# dest's fresh _push() below establishes an INDEPENDENT second
			# owner, exactly like an ordinary aliasing read would, not a
			# transfer.
			#
			# NOT the same shape as or_throw(mapper)/or_return(mapper)'s own
			# hidden __ot_ok_N/__or_ok_N locals - those are moved out via
			# their own explicit manually_decreffed() call at the point
			# they're produced (see ok_thunk in both), so by the time such a
			# src reaches here its binding is already MOVED, not OWNED; a
			# previous version of this branch additionally neutralized an
			# OWNED src here too, on the assumption that ANY Variable
			# reaching this branch was necessarily one of those hidden,
			# single-use locals - wrong for a real outer-scope local like
			# str(s)'s `s`, which caused a leak (s's own epilogue entry was
			# cancelled, leaving only t's to release one of the two live
			# references). Confirmed via or_throw_mapper_test.py/
			# or_return_mapper_test.py still passing without this.
			pass
		if dest.is_global:
			# a global's storage isn't scoped to THIS function's own
			# epilogue at all - whatever gets stored now must persist for
			# FUTURE reads by other calls, long after this function
			# returns (unlike an ordinary local, whose lifetime genuinely
			# IS bounded by the function). Mirrors attr_replace()'s own
			# model (a struct/union field's contents also aren't function-
			# scoped, never tracked in self.bindings at all) rather than an
			# ordinary local's push-a-fresh-epilogue-entry REPLACE below:
			# release whatever the global currently holds (unconditionally
			# - safe even the very first touch, when it's still whatever
			# its own initializer set, since decref on a non-RC-tagged
			# union member is already a documented no-op), store the new
			# value, and never register a per-function decref obligation
			# for it. self.bindings is never touched for a global here, so
			# it can never reach merge_if's branch-reconciliation logic
			# either - there's no function-scoped ownership state to
			# disagree about in the first place. (A prior version of this
			# fix DID push a function-scoped entry for a global, gated by a
			# runtime ownership flag - that was wrong: decref'ing a global
			# at ITS ASSIGNING FUNCTION's own exit would free the very
			# value the global is supposed to keep alive for the NEXT
			# call, a real use-after-free on the following read.)
			#
			# reaching this branch at all means a real `global X; X = ...`
			# reassignment (lower_global()'s own initializer path emits its
			# ir.Assign directly, never through this method - see
			# Variable.reassigned_outside_init's own comment) - flip the
			# flag PLAN_THREAD_SAFE_SHARED_STATE.md's emitter-side lock
			# insertion (emitter_c.py) keys off of, unconditionally: a
			# global only needs protecting once ANY function reassigns it,
			# regardless of how many do.
			dest.reassigned_outside_init = True
			# ir.AcquireGlobalLock/ReleaseGlobalLock bracket the ONE critical
			# section a protected global's reassignment needs: releasing its
			# CURRENT value and (later, in lowering.py's _cfg_assign, which
			# emits the actual ir.Assign this method itself never does)
			# overwriting it must happen as one atomic-with-respect-to-other-
			# threads unit. Wrapping _decref_instructions' OWN output here -
			# not reconstructing this boundary later by pattern-matching the
			# emitted instructions - matters because that output isn't
			# always a single bare ir.Decref: a union-typed global (e.g.
			# ZoneInfo|None, localtz()'s own type) decrefs via a multi-
			# instruction tag-gated sequence (Cmp/JumpIfFalse/GetAttr/Decref/
			# Jump/Label), which only whoever is CONSTRUCTING it (here) can
			# reliably bound - confirmed the hard way: an earlier, emission-
			# time "look for an adjacent Decref+Assign" version of this
			# mechanism silently never matched a union-typed global at all.
			instructions.append( ir.AcquireGlobalLock( var = dest ))
			instructions += self._decref_instructions( dest.type, dest ) # reads dest's CURRENT (pre-overwrite) value
			return instructions
		existing = self.bindings.get( dest.stem )
		if existing is not None and existing.entry is not None:
			if existing.state == OwnState.OWNED:
				instructions += self._decref_instructions( dest.type, dest ) # release whatever dest held before - reads dest's CURRENT value, emitted before the Assign overwrites it
			existing.entry.cancelled = False # dest is getting a real value again, even if it was MOVED/never-decref'd before
			self.bindings[dest.stem] = _Binding( operand = dest, type = dest.type, state = OwnState.OWNED, entry = existing.entry )
		else:
			self._push( dest, dest.type, OwnState.OWNED )
		return instructions

	def assign_global_initializer( self, dest: Variable ) -> list[ir.Instruction]:
		''' PLAN_THREAD_SAFE_SHARED_STATE.md Part A: the Acquire+decref-
		current-value half of a global's OWN initializing write (lowering.py's
		run_global emits the actual ir.Assign itself, then ir.ReleaseGlobalLock,
		mirroring _cfg_assign's own write-side split for the identical "the
		Assign is its own separate textual read/write in the generated C"
		reason). Structurally IDENTICAL to assign()'s own `dest.is_global`
		write branch above, with ONE deliberate difference: this does NOT
		flip dest.reassigned_outside_init - a global written only by its own
		initializer, never reassigned from a function body, is still provably
		single-write and needs no lock at all (that flag's own comment); this
		method's own Acquire/Release markers become no-ops for exactly that
		case, gated at EMISSION time the same way every other marker already
		is (see the is_alias branch above's identical "final value isn't
		known yet here" reasoning).

		Needed despite that "provably single-write" framing because it was
		never quite true: a global's initializer can call an ordinary
		function, and SYNTAX.md documents that as fully legal ("not
		restricted to a compile-time constant... can call ordinary functions
		at real program-startup time") - if that function spawns a thread
		(joined or not), the spawned thread can reassign THIS global (or read
		it) through the fully-locked ordinary `global X; X = ...` path WHILE
		__metalpy_init() is still running, concurrently with this global's
		own unlocked initializing write - confirmed as a real, reachable
		race, not a theoretical one, once a global with a real initializer
		exists alongside ANY reassignment of it from a thread-reachable
		function. rc_leaves(dest.type) early-return matches assign()'s own -
		Part A (and this fix) is scoped to RC-typed globals only, same as
		everywhere else in this mechanism. '''
		if not rc_leaves( dest.type ):
			return []
		instructions: list[ir.Instruction] = [ ir.AcquireGlobalLock( var = dest ) ]
		instructions += self._decref_instructions( dest.type, dest ) # reads dest's CURRENT (zero-initialized, pre-first-write) value - release_object()'s own NULL check makes this a safe no-op the very first time
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
			if is_rc and existing.state == OwnState.OWNED:
				instructions += self._decref_instructions( attr.type, attr )
			existing.entry.cancelled = False
			self.bindings[key] = _Binding( operand = attr, type = attr.type, state = OwnState.OWNED, entry = existing.entry )
		elif is_rc:
			self._push( attr, attr.type, OwnState.OWNED, key = key, is_construction_attr = True )
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

	def is_fresh_temp( self, operand: ir.Operand ) -> bool:
		''' True when `operand` is a still-tracked, fresh/owned temp (a
		Call/Allocate result registered via fresh_temp() above, not yet
		consumed into a named binding or untracked) - NOT the same as
		`isinstance(operand, ir.Temp)` alone: some Temps are deliberately
		never registered (a bare borrowing cast, e.g. compiler.cast(...) or
		list._read_element's own returned slot - see this class's own
		untrack_temp docstring and lowering.py's matching comments), so
		checking membership in _temp_states is the only reliable signal.
		Used by callers (e.g. _coerce_or_check_operand) that need to
		release/untrack a PRE-coercion operand in place, rather than
		leaving it as a dangling pending-temp obligation for whatever
		later flush would otherwise decref it unconditionally - safe only
		when the operand was genuinely fresh to begin with, never for a
		borrowed one. '''
		return isinstance( operand, ir.Temp ) and operand.id in self._temp_states

	def delete_temp( self, temp: ir.Temp ) -> list[ir.Instruction]:
		t = self._temp_states.pop( temp.id, None )
		if t is None:
			return []
		return self._decref_instructions( t, temp )

	def untrack_temp( self, operand: ir.Operand | None ) -> None:
		''' drops `operand` from _temp_states without emitting a Decref -
		ownership is transferring elsewhere (handed to the caller as a
		`return` value, moved into self._return_value_var across a shared-
		epilogue-label Jump - see lowering.py's _stmt_Return, both of its
		branches) rather than actually ending here. A later delete_temp()
		for the SAME temp (lowering.py's own per-statement pending-temp
		flush - _flush_pending_temps) then correctly becomes a no-op
		instead of decref'ing the very value just handed off. A no-op for
		anything that isn't a Temp (an ordinary Variable/Parameter was
		never in _temp_states to begin with). '''
		if isinstance( operand, ir.Temp ):
			self._temp_states.pop( operand.id, None )

	def snapshot_temp_states( self ) -> dict[int,Type]:
		''' lowering.py's _expr_BoolOp needs this: a non-last operand's
		fate (kept by the decisive branch vs. discarded by the sibling
		"continue" branch) forks into two mutually exclusive RUNTIME paths
		that both get COMPILE-TIME code generated, sharing the SAME
		operand/temp - but fresh_temp()/untrack_temp() mutate this dict
		in place, not scoped per branch. Once the decisive branch (always
		emitted FIRST in compile-time instruction order) calls
		untrack_temp() on the operand it's keeping, that removal is
		permanent from this dict's own perspective - the continue branch's
		later delete_temp() call on the SAME operand would then find
		nothing tracked and silently skip its Decref, even though THAT
		branch is the one that's actually discarding it. Snapshot right
		before the branch split, restore_temp_states() right before the
		continue branch's own cleanup, so it sees its own independent,
		unclaimed view - exactly mirroring _expr_IfExp's true/false
		branches, each of which gets a fresh view for free by construction
		(each lowers its own independent sub-expression from scratch,
		rather than forking off one shared, already-lowered operand). '''
		return dict( self._temp_states )

	def restore_temp_states( self, snap: dict[int,Type] ) -> None:
		self._temp_states = dict( snap )

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
		call site itself either way - ownership transfers as-is; the only
		instructions this can return are _neutralize()'s own flag-disarm,
		for the rare case operand's entry was already captured by an
		earlier return (see its own docstring) - lowering.py must emit
		these at the move's own call site. '''
		if isinstance( operand, Variable ):
			binding = self.bindings.get( operand.stem )
			if binding is None:
				return [] # not RC-tracked (non-RC type) - nothing to do
			if binding.state != OwnState.OWNED:
				raise CompileError(
					f'{target_qualname}: cannot move {operand.stem!r} into parameter {param_stem!r} - '
					f'it is {binding.state.value}, not owned here'
				)
			if binding.entry is not None:
				new_entry, instructions = self._neutralize( binding.entry )
			else:
				new_entry, instructions = None, []
			# a fresh _Binding, never mutate the existing one in place - an
			# earlier snapshot() may still hold a reference to it (see
			# assign()'s own "always construct fresh" discipline) - entry =
			# new_entry (_neutralize()'s own possibly-replaced object), not
			# binding.entry, for the identical reason
			self.bindings[operand.stem] = _Binding( operand = binding.operand, type = binding.type, state = OwnState.MOVED, entry = new_entry )
			return instructions
		if isinstance( operand, ir.Temp ):
			self._temp_states.pop( operand.id, None )
			return []
		return []

	# --- entry cancellation (move/del/compiler.decref) ----------------------

	def _mint_cancel_flag( self ) -> Variable:
		''' a fresh runtime bool for _neutralize()'s flag-guarded branch, OR
		for merge_if()'s own ownership-disagreement reconciliation (an
		OWNED-vs-BORROWED split across an if's two branches - "fill in
		a default when still borrowed") - both share the identical shape, so
		this one minting helper covers both callers. Mirrors push_defer()'s
		own flag exactly (a real Variable, spliced in as a body_start init by
		lowering.py's _emit_epilogue - see cancel_flags()), except armed
		(True) by default instead of disarmed: a defer flag starts False and
		gets armed by the defer statement itself; this one starts True
		(still needs releasing) and gets disarmed by whichever of move()/
		deleted()/manually_decreffed()/merge_if()'s own borrowed-branch case
		actually neutralizes the entry - see _neutralize(). '''
		index = len( self._cancel_flags )
		qualname = f'{self.fn.qualname}.__cancel_flag_{index}' if self.fn is not None else f'__cancel_flag_{index}'
		flag = Variable(
			stem = f'__cancel_flag_{index}', qualname = qualname,
			file = self.fn.file if self.fn is not None else None,
			line = self.fn.line if self.fn is not None else None,
			type = self._bool_type,
		)
		self._cancel_flags.append( flag )
		return flag

	def cancel_flags( self ) -> list[Variable]:
		''' every runtime flag _neutralize() has minted so far - lowering.py's
		_emit_epilogue consults this (alongside self._defer_flags) to build
		the function's own flag_inits, each initialized True (armed), unlike
		a defer flag's False - see _mint_cancel_flag()'s own comment. Empty
		for the overwhelming majority of functions (nothing captured-then-
		cancelled ever happened) - this only ever grows past empty for the
		specific shape _neutralize() documents. '''
		return list( self._cancel_flags )

	def _neutralize( self, entry: Epilogue ) -> tuple[Epilogue, list[ir.Instruction]]:
		''' shared by move()/deleted()/manually_decreffed(): stop entry's own
		pending Decref from firing a SECOND time via the scope's own shared
		epilogue, now that the caller has already emitted (or is about to
		emit) an explicit one of its own for the control-flow path reaching
		THIS statement.

		A plain compile-time `entry.cancelled = True` is only safe when no
		earlier return has already committed a `goto` into entry's own
		shared label (entry.captured - see current_epilogue_label()):
		build_epilogue_ladder() replays every entry's CURRENT (final) state
		once, at the function's own closing brace, for every jump site that
		shares it - not the state each jump site actually saw at the time
		it was emitted. An entry captured by an earlier, still-live return
		and THEN cancelled here would silently turn that earlier return's
		own commit into a no-op landing: the label still gets built (used_
		shared_epilogue_label() sees to that), but empty, permanently
		leaking whatever THAT path was relying on the shared ladder to
		release - confirmed by a real repro: two early-return checks
		around an ordinary call, then an explicit compiler.decref() and a
		final return - both early returns silently stopped releasing their
		own copy/local once the trailing decref cancelled the entry they'd
		already jumped into.

		Once captured, the only correct fix is a genuine runtime
		distinction: mint (or reuse) a flag, default-armed at the
		function's own top (cancel_flags()), and disarm it right here
		instead of statically cancelling - _replay() then only actually
		runs the decref for whichever paths reach the ladder WITHOUT having
		gone through this disarm first, exactly like an errdefer's own
		flag already does for its own replay. Never cancelled in this
		branch (an already-flag-guarded entry must keep being replayed -
		by build_epilogue_ladder()'s own "cancelled entries get no
		instructions" rule, cancelling it too would just silently drop the
		flag check itself).

		Returns (the entry going forward - possibly a NEW object, see
		below - and whatever instructions need emitting). The static-cancel
		branch never mutates `entry` in place - it builds a REPLACEMENT
		(cancelled=True) and swaps it into self._epilogue_stack's own live
		slot instead, exactly mirroring _Binding's own "always construct
		fresh" discipline (see move()'s own comment) - `entry` itself may
		still be referenced by an EARLIER snapshot() (e.g. an enclosing
		if/try's own entry_bindings, or a sibling branch's own end-state
		captured before this call), which must see it stay untouched. The
		caller is responsible for using the RETURNED entry (not the
		original `entry` argument) in whatever _Binding it constructs next
		- see move()/deleted()/manually_decreffed()'s own call sites.

		This replace-don't-mutate discipline is also why a manual
		compiler.decref()/del on an entry declared before an if/try/
		or_throw() dispatch's own sibling-branch split needs no special
		runtime-flag protection (an earlier design here,
		enter_diverging_paths(), minted a flag for exactly that case -
		removed once this made it provably redundant): the branch that
		cancels it gets a fresh REPLACEMENT object swapped into
		self._epilogue_stack's own live slot, so the ORIGINAL entry object
		itself stays untouched - restore()'s own .cancelled revert (see its
		own docstring) then resyncs the sibling branch's copy of that same
		slot back to the pristine, uncancelled state, rather than relying on
		object identity alone (self._epilogue_stack is a flat list of
		mutable slots, not a tree of untouched objects - restore() has to
		actively resync it). '''
		if entry.name not in self._captured_labels:
			new_entry = _dc_replace( entry, cancelled = True )
			for i, e in enumerate( self._epilogue_stack ):
				if e is entry:
					self._epilogue_stack[i] = new_entry
					break
			return new_entry, []
		if entry.flag is None:
			entry.flag = self._mint_cancel_flag()
		return entry, [ ir.Assign( dest = entry.flag, src = ir.Const( type = entry.flag.type, value = False )) ]

	# --- del x -------------------------------------------------------------

	def deleted( self, variable: Variable, ctx: str ) -> list[ir.Instruction]:
		''' called for `del x` (see lowering.py's _stmt_Delete) - returns
		the Decref to emit right there (if x was OWNED), and
		neutralizes its epilogue entry so it's never decref'd again.
		Independent-of-RC unchecked-Result check first, same reasoning as
		assign()'s own early check - del'ing a still-unchecked Result is
		exactly the "discarded via del" table entry, regardless of whether
		its type has any RC leaves at all.

		Liveness check next, same reasoning again - `del` reads/consumes the
		binding before removing it, so it needs the identical definite-
		assignment gate _expr_Name applies to an ordinary read (this is the
		"__del__ a variable that's not provably alive" half of that gate -
		see is_live()'s own docstring). Checked before self.bindings.pop()
		below so an already-live-but-never-RC-bound (scalar/struct/enum)
		variable.stem still gets a real error instead of deleted() silently
		no-op'ing (there was never a self.bindings entry to pop for those in
		the first place). self._live is updated unconditionally afterward,
		error or not - a name that WAS live is no longer live once del'd
		either way (mirrors fn.names' own removal in lowering.py). '''
		if variable.stem in self._unchecked_results:
			raise CompileError(
				f"Result value {variable.stem!r} is discarded via del - it was never inspected: "
				f"use .is_ok(), .is_err(), .or_return(), .unwrap(msg), or match"
			)
		if variable.stem not in self._live:
			raise CompileError( f"{ctx}: {variable.stem!r} is not initialized on all code branches" )
		self._live.discard( variable.stem )
		binding = self.bindings.pop( variable.stem, None )
		if binding is None or binding.entry is None:
			return []
		instructions: list[ir.Instruction] = []
		if binding.state == OwnState.OWNED:
			instructions = self._decref_instructions( binding.type, variable )
		_, neutralize_instructions = self._neutralize( binding.entry ) # `variable` is popped from self.bindings entirely - no new _Binding to update with the returned entry
		instructions += neutralize_instructions
		return instructions

	# --- compiler.decref(x) -------------------------------------------------

	def manually_decreffed( self, operand: ir.Operand ) -> list[ir.Instruction]:
		''' called for compiler.decref(x) (see lowering.py's
		_lower_compiler_decref) - x's own explicit Decref is emitted by
		lowering.py right at the call site regardless; this only stops x's
		binding from being auto-decref'd a SECOND time once its own scope
		ends. Without this, a live OWNED local manually decref'd (the
		established idiom throughout this stdlib for tearing down RC
		elements read out of a container - list.__del__/FastList.__del__/
		dict's own _release_key/_release_value all do `val: T = <read>;
		compiler.decref(val)`) gets decref'd AGAIN by the scope's own
		epilogue, since compiler.decref was never wired into the ownership-
		tracking that epilogue relies on - confirmed with
		AddressSanitizer: a real, always-on (not merely heap-layout-
		dependent) double Decref -> use-after-free -> heap corruption on
		every single call, for exactly this shape. Mirrors move()'s own
		cancellation exactly (both go through the shared _neutralize()
		helper, same transition to MOVED so a later reference to x - now
		potentially freed - is caught as a compile error same as using a
		moved-out value would be), but without move()'s own state-mismatch
		error: compiler.decref(x) on a BORROWED binding (an ordinary
		un-owned parameter, entry is None - nothing to cancel) or one
		already MOVED/decref'd is left to whatever lowering.py itself
		decides to allow, not rejected here. Returns whatever _neutralize()
		itself needs emitted (empty unless x's own entry was already
		captured by an earlier return - see its own docstring) -
		lowering.py must emit these right after its own explicit Decref.

		x's own entry is deliberately left UNCANCELLED (and _neutralize()
		never called) when x is in self._possibly_retained: some earlier
		call passed x into a parameter an errdefer/defer conditionally
		incref's (see push_defer()/mark_possibly_retained()) - x's object
		may therefore carry an extra reference this decref() call is only
		balancing, on top of (not instead of) x's own binding-owned
		reference, which still needs its own ordinary scope-exit release. '''
		if isinstance( operand, Variable ):
			binding = self.bindings.get( operand.stem )
			if binding is None or binding.entry is None:
				return []
			if binding.state != OwnState.OWNED:
				return []
			if operand.stem in self._possibly_retained:
				self.bindings[operand.stem] = _Binding( operand = binding.operand, type = binding.type, state = OwnState.MOVED, entry = binding.entry )
				return []
			new_entry, instructions = self._neutralize( binding.entry )
			self.bindings[operand.stem] = _Binding( operand = binding.operand, type = binding.type, state = OwnState.MOVED, entry = new_entry )
			return instructions
		if isinstance( operand, ir.Temp ):
			self._temp_states.pop( operand.id, None )
		return []

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

	def complete_base_construction( self, base_required: list[Variable] ) -> None:
		''' called by lowering.py right after emitting a subclass's own
		super().__init__(...) call (RCClass single-inheritance, see the
		RCClass-subclassing plan's own Phase 2) - the base's own __init__
		has now REALLY run and (per ITS OWN complete_construction(), a
		completely separate CFGState instance scoped to that other
		function's own lowering) already fully initialized every attribute
		in `base_required` (the WHOLE base chain's own required attributes,
		base-first - mirrors RCClass.flattened_attributes()'s own walk,
		called from the base onward, not including the subclass's own
		attributes). Marks each one initialized in THIS (subclass)
		__init__'s own bindings in one shot, WITHOUT any Incref - ownership
		was already correctly established by the base's own construction;
		this is bookkeeping, not a new reference, mirroring attr_assign's
		own is_rc branch exactly except for that one difference. Pushes a
		REAL, epilogue-tracked OWNED binding for each (same as attr_assign)
		so an early exit from the SUBCLASS's own body after this point
		(before ITS OWN complete_construction() runs) still correctly
		decrefs whichever base attributes are already live - exactly like
		an ordinary attr_assign'd field would, not treated any differently
		just because it came from the base. Never called with an attribute
		already bound (super().__init__() is required to be literally the
		first statement - see lowering.py - so nothing could have touched
		self.<base_attr> before this runs), so unlike attr_assign this
		never needs an "already exists" branch. '''
		for attr in base_required:
			# flattened_attributes() (unlike own_new_virtual_slots() right beside
			# it in mpy_types.py) doesn't resolve anything it returns, so base
			# attributes DO routinely arrive here unresolved - that alone is
			# normal and can't be asserted away (i32 fields do it constantly).
			#
			# What IS load-bearing is that an unresolved one is never RC. Asking
			# the RC question of an unresolved attribute answers "not RC" and
			# takes the else-branch below, pushing NO epilogue entry - so an
			# early exit from the subclass __init__ after super().__init__()
			# would never decref that base field, the leak this method's own
			# docstring says it exists to prevent. Same "unresolved and 'has no
			# RC leaves' are indistinguishable" hazard as
			# TaggedUnion._resolved_leaves (see its docstring).
			#
			# Instrumenting the whole test corpus: only i32 attributes ever
			# arrive unresolved; the one RC base attribute (str) is always
			# already resolved here - plausibly because an RC-typed annotation
			# has to be looked up to be scheduled at all, where an intrinsic
			# scalar doesn't. That's an accident of resolution order rather than
			# anything guaranteed, so it's a tripwire, not an assumption: if an
			# RC attribute ever does arrive unresolved, the correct fix is to
			# resolve it at the source, not to rely on being rescued here.
			arrived_unresolved = attr.resolve is not None
			if arrived_unresolved:
				attr.resolve()
			assert not ( arrived_unresolved and attr.type is not None and attr.type.is_rc() ), (
				f'base attribute self.{attr.stem} ({attr.type.qualname if attr.type else "?"}) '
				f'is RC but arrived unresolved - the RC answer here now depends on compile '
				f'order; resolve it at the source'
			)
			key = f'self.{attr.stem}'
			if rc_leaves( attr.type ):
				self._push( attr, attr.type, OwnState.OWNED, key = key, is_construction_attr = True )
			else:
				self.bindings[key] = _Binding( operand = attr, type = attr.type, state = OwnState.OWNED, entry = None )
			# complete_construction()'s own success-path cancellation loop
			# only walks self._construction_required (the SUBCLASS's own
			# declared attributes - enter_construction()'s own docstring
			# flags this as "not yet base-class-aware"). Without also
			# appending base attrs here, a base-owned RC attribute's
			# epilogue entry (just pushed above) is never cancelled on the
			# subclass __init__'s success path, so an ordinary fall-off-the-
			# end/plain `return` wrongly decrefs a field self now legitimately
			# owns - confirmed by a real compile: any subclass whose base
			# __init__ sets an RC field emits `release_object((ObjectHeader*)
			# (<bare_field_name>))` unconditionally at the end of ITS OWN
			# __init__, referencing a name that was never even declared as a
			# local in the generated C (it's self's own field, not a local).
			# already present in self.bindings (just pushed above), so this
			# can't trip complete_construction()'s "missing" check.
			self._construction_required.append( attr )
