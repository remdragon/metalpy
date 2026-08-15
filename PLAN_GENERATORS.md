Generator functions (`yield`, state-machine transform)

STATUS: v1 + Phase 2 (while loops) + Phase 3 (`for`-loop consumption)
landed and real-compile-and-run tested (emitter_c_test.py's
GeneratorFunctionTests) - PLAN_GENERATORS.md's own motivating example (a
real `range()`-shaped generator: `while i < count: yield i; i += 1`,
consumed the natural way: `for x in counter(5):`) now compiles and runs,
not just textual `range()` sugar. `range()` ITSELF has deliberately NOT
been rewritten into a real generator yet (see Phase 3 note below) - the
sugar path (`_is_range_call`/`_lower_for_range`) is untouched.

`yield` may be either a direct top-level statement of the function body
(Phase 1), or the single yield inside a direct top-level `while` loop
(Phase 2) - a yield nested inside an if/for/with/try, or inside a loop
that has more than one yield or any break/continue, is a clear compile
error, not a silently wrong state machine (see GeneratorFunctionTests'
own three rejection tests). Both unit shapes verified for the RC-
correctness payoff this whole plan is about: a generator dropped mid-
iteration correctly decrefs a captured RC-typed PARAMETER via the
ordinary, completely unmodified $$__destructor__ synthesis - see "The
$$__del__ problem" below for why parameters specifically, not yet
arbitrary locals.

Phase 2 design: a `while` unit occupies TWO states (not-yet-entered /
resuming) rather than one state per iteration - cfg.py's own structured
loop machinery is untouched, no goto/switch needed. Restructured (in
type_resolver.py's `_build_while_unit_guard`) into the standard resumable-
loop idiom: `while True: [on resume only: run the code after the yield,
once]; if not cond: break; [code before the yield]; state = N+1; return
value`. When the loop naturally exhausts, state advances and execution
FALLS THROUGH (no return) into whatever follows - correct, since Python's
own generator semantics don't pause between a loop ending and the code
after it. Confirmed working when mixed with ordinary bare-yield units in
the same generator (bare yield, then a while-unit, then another bare
yield - see the `bare_yield_and_while_unit_mixed_in_one_generator` test
case).

Landed design deviates from the plan's original sketch in three ways,
each because the simpler thing turned out to already be sufficient:

1. No new `ir.Yield` instruction, no emitter changes at all. `yield expr`
   inside a segment decomposes into ordinary, already-existing statements
   (`self.__state = N`, `return expr`) built directly as synthesized AST
   handed to the ordinary statement-lowering pipeline - emitter_c.py was
   never touched.
2. No per-state-gated destructor. v1 restricts a promoted LOCAL (as
   opposed to a captured parameter) to scalar types only (bool/integer) -
   scalars need no decref at all, so the existing unconditional
   `_synthesize_rcclass_destructor` cascade is correct completely
   unchanged. An RC-typed local surviving a yield (needing the state-
   gated cascade this doc originally sketched) is deferred, no forcing
   use case yet.
3. Dispatch is a flat sequence of `if self.__state <= i:` guards, not a
   goto/switch - discovered while implementing that emitter_c.py already
   lowers Jump/Label/JumpIfFalse as flat C goto/label pairs (not
   reconstructed structured control flow), so a real dispatch mechanism
   was available for free, but the recursive-AST-If approach turned out
   simpler to generate correctly and needed zero emitter work either way.
   A YIELD unit's own guard still ends in an unconditional `return`; a
   WHILE unit's guard does NOT (see the Phase 2 design note above) - this
   turned out to still fit the same flat-guard-chain shape with no
   goto/switch needed, confirming the Phase 1 prediction that loops
   wouldn't need the dispatch mechanism to change, only the unit-
   recognition/guard-building logic.

Phase 3 design (lowering.py's `_lower_for_over_iterator`, alongside the
existing `_lower_for_range`/`_lower_for_over_indexable`): `for x in
<expr>:` now recognizes a third shape - `<expr>`'s type has a `__next__()`
returning `T|None` (checked generically, not generator-specific - any
hand-written class implementing `__next__` this way qualifies too, see
`test_for_loop_over_bad_next_shape_is_rejected`). `node.iter` is lowered
exactly ONCE up front, then handed to whichever of the three paths
applies (a pre-existing bug class this incidentally forecloses:
`_lower_for_over_indexable` used to re-lower `node.iter` itself, which
would have double-evaluated/double-constructed an iterable expression
with a side effect - never triggered before because nothing passed to a
`for` loop had a side effect worth noticing until a generator
CONSTRUCTOR call became a realistic `node.iter`).

Corrected finding from the v1 write-up above: reading a `T|None` value
back out in narrowed form is NOT a dead end - it does NOT work via a
plain `if x is None: ... else: ...`-style read (confirmed still broken,
generator-unrelated, not fixed here), but DOES work via `match x: case
T(x): ...` - a class-pattern REUSING the subject's own name as its
capture (confirmed with a real compile-and-run repro: `match a: case
i32(a): ... case None: ...`). This is what `_lower_for_over_iterator`
uses under the hood: cfg.py's own `narrow()`/`narrowed_member()` API,
called directly (bypassing match-statement syntax entirely, since that
goes through a type_resolver.py rewrite pass this hand-built lowering
code never runs through) to mark a hidden local's payload extractable,
then read back via the loop target's own ordinary `_stmt_Assign`. This
is now real, working, verified precedent for whoever picks up general
`is None`-based narrowing next - the payload-extraction primitive itself
works fine; what's actually missing is wiring `_ReferenceResolver`'s
existing is/is-not-None tag-comparison rewrite up to ALSO call `cfg.
narrow()` (today it only produces the tag comparison, never the
narrowing fact - confirmed by reading visit_Compare's own rewrite 1).
Also found and worked around the same way (a synthesized `x is None`
AST node doesn't lower correctly at all outside a real function body,
since that rewrite only runs once, early, over REAL source - not
something built mid-lowering): `_lower_for_over_iterator` builds the
equivalent `.tag == N` comparison directly, by hand, rather than relying
on `is None` syntax.

Not attempted this pass, deliberately: rewriting `range()` itself from
`_is_range_call` textual sugar into a real generator. The sugar has a
real, deliberate performance property (documented in its own comment: no
allocation, unchecked arithmetic proven safe by construction) that a
real generator's heap-allocated backing object would give up, and
`range()` is used pervasively throughout lib/ and the existing test
suite - swapping its implementation is a real risk for a cosmetic-only
win (per this doc's own original framing, "lets range() stop being
special-cased syntax") now that the FUNCTIONAL goal (a real, user-
authored range()-shaped generator working end to end) is already met by
`counter()` in the tests above. Left as an explicit follow-up, not
folded into this pass.

Original planning notes follow, kept for the phases not yet attempted
(fallible generators, generic generators, `for` LOOPS containing yield -
as opposed to `while`, still rejected: a `for` loop's own hidden index/
length bookkeeping was never analyzed for this).

Why

TODO.txt's own "generators:" entry (line 378) already states the two hard
parts correctly: the function body has to become a state machine, and the
function's normal epilogue (RC decref of locals) has to move into the
backing object's `__del__`, because a generator can be abandoned mid-
iteration from ANY suspended point, not just the one linear exit an
ordinary function's epilogue handles. This plan works out a concrete design
for both, in terms of machinery that already exists in this compiler
(cfg.py's epilogue/bindings tracking, the `$$__destructor__`/closure-
trampoline synthesis pattern, `T|None` unions), rather than inventing new
infrastructure where existing infrastructure already almost fits.

Motivating use case, unchanged from TODO.txt:

	def range( count: usize ) -> Iterator[usize]:
		i: usize = 0
		while i < count:
			yield i
			i += 1

No real lib/ code needs this yet (`_is_range_call` in lowering.py is
explicit that `range(...)` is textually-recognized sugar today "because
there's no real range() function... the generator state-machine transform,
which doesn't exist yet"). Landing this plan's Phase 3 (below) is what lets
`range()` stop being special-cased syntax and become an ordinary library
function - closing that TODO.txt item for real.

Representation

A `def foo(...) -> Iterator[T]:` whose body contains a `yield` anywhere
(detected the same way Python itself does - an AST walk over the body that
does NOT recurse into a nested `def`/`lambda`, matching PLAN_LAMBDA.md's
own nested-scope boundary) is lowered completely differently from an
ordinary function. Three real, resolved units get synthesized from the one
source `def`, following the exact pattern `_synthesize_rcclass_destructor`
and `_get_or_create_closure_trampoline` already use (hand-build an AST
FunctionDef + Function, `schedule()` it):

1. A backing RCClass, `foo$$generator` (naming mirrors `$$__destructor__`/
   `$$__closure_trampoline__`). Fields:
   - `__state: u32` - resume discriminant. 0 = "not yet started" (the state
     the constructor leaves it in), one value per `yield` site, and one
     reserved terminal value (`DONE`) meaning exhausted.
   - one field per original parameter, captured at construction time.
   - one field per local variable that's live across at least one `yield`
     (see "Liveness" below) - promoted from an ordinary stack local/Temp to
     a struct field, because each `__next__()` call is a SEPARATE C
     function invocation with its own stack frame; nothing on the C stack
     survives a `return` and a later re-entry, only fields do.

2. `foo(...)` itself becomes a trivial constructor: `ir.Allocate` the
   backing RCClass, store each argument into its matching field, set
   `__state = 0`, return the object. No user code runs yet - matches
   Python's own "calling a generator function doesn't execute the body"
   semantics exactly, for free, because construction and first-`__next__`
   are already two different calls in this design.

3. `__next__(self) -> T|None` (see "Consumption protocol" below) is the
   real translated body: a dispatch on `self.__state` followed by the
   original control flow, rewritten so every promoted local reads/writes
   through `self.<field>` instead of a plain Temp/Variable, and every
   `yield expr` becomes "store expr as the result, set `__state` to this
   yield's own state id, return `Some(expr)`" - with the very next
   instruction after it labeled as that state's resume point.

Liveness ("which locals become fields")

No new dataflow pass is needed. cfg.py already maintains `self.bindings`
(what's alive right here) for its own epilogue/snapshot purposes - the
state-machine pass reads that SAME dict at the exact point each `yield` is
lowered. Any local present in `bindings` at a `yield` is promoted to a
field, unconditionally (not path-sensitive). This over-promotes slightly
(a local that happens to be alive at a yield on paths that never actually
use it again after resuming still gets a field) but never under-promotes,
which is the only direction that would be an actual correctness bug (stack
garbage on resume). Cheap and safe beats precise here, and it reuses
tracking that's already maintained for a different reason rather than
adding a second liveness computation that has to agree with the first one.

Each promoted field also records the lowest state number at which it's
known-initialized (the enclosing yield's own state id, or 0 if it's a
parameter/assigned before the first yield) - this is what the destructor
needs next.

The `$$__del__` problem (why this is harder than the ordinary destructor)

`_synthesize_rcclass_destructor` today builds ONE fixed cascade: run
`self.__del__()` if declared, decref every RC-typed field base-first,
`sys.free(self)`. That's correct for an ordinary object because every
field is unconditionally valid for the object's entire lifetime once
construction finishes.

A generator's backing object does NOT have that property: a promoted local
field is only known-initialized from ITS OWN first-assigning state onward,
and if the generator is dropped (refcount hits zero) while paused at
`__state = N`, decref-ing a field that was promoted at state `N+3` (never
reached) would decref uninitialized/stale memory. So the synthesized
`$$__destructor__` for a generator's backing class needs to become
`state`-aware:

	switch ( self->__state ) {
		case DONE: break;               // nothing live, already fell off the end
		case 4: /* decref fields valid from state 4 onward */ FALLTHROUGH;
		case 3: /* ... */ FALLTHROUGH;
		...
		case 0: break;                  // only params live, handled below unconditionally
	}
	// params: always valid from construction onward, unconditional decref
	sys.free( self );

This is the direct, literal realization of "the function epilogue goes
into `__del__`" from the task description, generalized from "one epilogue"
to "one epilogue per suspend state" - because a generator effectively has
N possible exit points from the caller's point of view (drop while paused
at state 0, 1, 2, ..., or after DONE), where an ordinary function only
ever has the one. The per-state field validity ranges computed during
liveness (above) are exactly the table this switch is built from - no
separate design needed once liveness is in hand.

Consumption protocol

`__next__(self) -> T|None` - deliberately reuses the EXISTING nullable-
union machinery (`T|None` is already a first-class synthesized
TaggedUnion, with view-narrowing `is None` checks) rather than inventing a
bespoke `StopIteration` type. `Iterator[T]` in a return-type annotation is
recognized textually by discovery.py, the same way `Callable[...]`/
`tuple[...]` already are (see `TupleType`/`CallableType`'s own textual-
subscript recognition) - it resolves to "this function's real return type
is `Ptr[foo$$generator]`", and `foo$$generator.__next__`'s real return type
is `T|None`.

`for x in some_call_returning_a_generator():` needs `_stmt_For` to grow a
third branch alongside today's `_lower_for_range`/`_lower_for_over_indexable`
(lowering.py:2562-2570): `_lower_for_over_iterator`, triggered when the
iterated object's type has a `__next__` method whose return type is
`<elem>|None`. Loop body: call `__next__`, `is None` check (existing union-
narrowing) to decide break-vs-continue, bind the narrowed non-None value to
the loop target. This is a new recognizer alongside `_is_range_call`, not a
replacement for the existing `__len__`/`__getitem__` indexable path (both
stay valid, for different callee shapes).

New IR

One new instruction, `ir.Yield`:

	class Yield( Instruction ):
		value: Operand       # the yielded T
		state: int           # this yield's own resume-state id
		resume_label: str    # label emitted immediately after, where state
		                     # `state` resumes into

Deliberately a dedicated instruction rather than composing existing
`SetAttr` + `Return` - `emitter_c.py` needs to recognize "this FuncStart is
a generator's `__next__`" to emit the `switch(self->__state) { case 0: goto
L_start; ... }` dispatch prologue before the ordinary body, the same way it
already special-cases `Function.is_destructor` for the void(void*)-plus-
cast prologue. A new `Function.is_generator_next: bool` flag (same posture
as `is_destructor`) drives this. `ir.Yield` codegen: `self->state = N;
return <Some(value) construction>; L_N:`.

Everything else in a generator's `__next__` body - `if`/`else`, `while`,
`for`, `break`/`continue`, arithmetic, calls - is lowered EXACTLY as it is
today; cfg.py's structured control flow, loop back-edges, and epilogue
ladder are untouched and still scoped correctly, because they only ever
need to reason about a single `__next__()` invocation's own linear
lifetime, never across a suspend. A `goto` landing in the middle of a
`while`-loop body (a resume point inside a loop) is ordinary, valid C11 -
the same technique hand-written C coroutines/protothreads (Duff's device,
Simon Tatham's coroutines) already rely on, not something new being
invented here.

Error handling - the TODO.txt open question, resolved by scoping it out of v1

TODO.txt asks: what does a generator do if an operation inside it can fail
(checked arithmetic, `or_return()`), given there's no exception mechanism
to unwind through? Recommended answer, to keep the first working slice
small and testable (this project's own stated stage-by-stage,
testable-per-stage philosophy - see ARCHITECTURE.md):

**v1 (this plan): `Iterator[T]` generators may not contain any fallible
operation** - no `or_return()`, no unguarded Check-mode arithmetic, no call
to a Result-returning function without `.unwrap()`/`.unwrap_or()`/an
explicit panic path. Enforced the same way defer/errdefer-with-RC-locals is
rejected today: a CompileError during lowering, not silently miscompiled.
The motivating `range()` use case is infallible and is unaffected.

**Follow-up (separate, later plan, sketched here so the v1 shape doesn't
paint it into a corner):** a second, explicit `Generator[T, E]` annotation
(distinct from `Iterator[T]`, which stays sugar for the infallible case)
whose `__next__` returns `Result[T|None, E]` instead of `T|None`.
`or_return()` used inside such a generator's body propagates by making
THIS `__next__()` call return `Err(e)` and permanently setting
`__state = DONE` (iteration ends for good, matching Python's own "a raised
exception inside a generator ends it") - exactly ordinary `or_return()`
semantics, just resolved against `__next__`'s own return slot instead of
the outer function's. This isn't designed further here because there's no
forcing use case in lib/ yet (same reasoning PLAN_LAMBDA.md used to defer
real closures) - noted so whoever picks it up isn't starting from zero.

Other body restrictions for v1

- `return` (bare) inside a generator body ends iteration early: emits the
  same as falling off the end (`__state = DONE`, `__next__` returns
  `None`) - matches Python's bare-`return`-inside-a-generator behavior.
- `return <value>` inside a generator body is a compile error (Python only
  allows this for `StopIteration(value)` interop, which has no analog
  here - simplest to reject rather than half-support).
- `defer`/`errdefer` inside a generator body: rejected outright
  (CompileError), same posture as TODO.txt's existing "defer + any RC
  local" gap - defer's single-shared-epilogue-per-CALL model is already a
  strictly easier version of this problem (one exit point, not N), and
  that easier version is already only partially solved. Not attempted here.
- `yield` inside a nested `def`/`lambda`/generic function, or inside a
  generic enclosing function/method: rejected, mirroring PLAN_LAMBDA.md's
  identical "generic enclosing scope" deferral (sidesteps "which
  monomorphization does this state machine belong to").
- `yield from` (sub-generator delegation) and `.send(value)` (two-way
  communication): out of scope, no forcing use case.

Precedent reused (nothing here is a new architectural pattern)

- `_synthesize_rcclass_destructor` / `_get_or_create_closure_trampoline`
  (lowering.py, type_resolver.py): "compiler hand-builds an AST
  FunctionDef + Function, schedules it exactly like user code" - the
  mechanism for all three synthesized units above.
- `ClosureType(RCClass)` (mpy_types.py): precedent for an RCClass with
  compiler-managed synthetic fields, real RC lifetime, no special-casing
  needed past discovery.py.
- cfg.py's `.bindings` / `.snapshot()` / `_epilogue_stack`: reused
  directly for liveness-at-yield (no new pass) and as the conceptual
  model generalized for the per-state destructor table.
- `_is_range_call`'s textual-sugar recognition, and `TupleType`/
  `CallableType`'s textual-subscript recognition in discovery.py: the
  pattern `Iterator[T]`/`Generator[T,E]` annotation recognition follows.
- `Function.is_destructor`'s "emitter special-cases this FuncStart's
  prologue" flag: the exact precedent for `Function.is_generator_next`.

Phasing

0. `ir.Yield` + generator-body detection (AST walk, no nested-scope
   recursion) + the body restrictions above enforced as CompileErrors.
1. Straight-line/if-else-only infallible generators (no loops yet) -
   smallest slice that proves backing-RCClass synthesis + dispatch switch
   + state-aware destructor end to end.
2. Loops (`while`/`for`) inside generator bodies - proves a resume `goto`
   landing inside a loop body interacts correctly with cfg.py's existing
   `loop_back_edge`.
3. `for x in <generator call>:` consumption (`_lower_for_over_iterator`),
   then rewrite `range()` from `_is_range_call` textual sugar into a real
   lib/ generator function - this is the concrete point where TODO.txt's
   motivating example actually compiles and runs, and where
   `_is_range_call`/`_lower_for_range` can be deleted.
4. RC-correctness verification pass: abandon-mid-iteration-and-check-no-
   leak (the actual payoff of the state-aware destructor), a generator
   with an RC-typed live-across-yield field, nested generators.
5. (separate plan, not this one) `Generator[T,E]` fallible generators.

Testing

Following this project's existing convention (structural IR-shape unit
tests in lowering_test.py, e.g. `test_defer_epilogue_shape`; real compile-
and-run tests in emitter_c_test.py, e.g. the `ListThreadSafetyTests`/
`DictThreadSafetyTests` style of "compile, run, assert real behavior"):

- lowering_test.py: exact IR shape for a single-yield generator (Allocate
  in the constructor, `ir.Yield` placement, field promotion for a variable
  alive across the yield, state numbering); defer/errdefer-in-generator
  rejected; value-carrying `return` rejected; generic-enclosing-scope
  rejected; fallible-operation-in-v1-generator rejected.
- emitter_c_test.py, real compile+run: multi-yield generator driven
  manually via repeated `__next__()` calls and checked against expected
  values; `for x in gen():` consuming one; a loop inside a generator body
  (resume-into-loop-middle); an RC-typed field carried across a yield,
  refcount-checked after each resume; a generator abandoned mid-iteration
  (dropped without reaching DONE) with a leak/double-free check identical
  in spirit to the existing `ListThreadSafetyTests`/`DictThreadSafetyTests`
  RC-element variants; `range()` rewritten as a real generator, exercised
  through an ordinary `for i in range(n):` call site.

Deferred / explicitly out of scope for this whole plan

- `Generator[T,E]` fallible generators (sketched above, not designed).
- `yield from`, `.send()`, `.throw()`, `.close()` beyond ordinary RC drop.
- `defer`/`errdefer` inside a generator body.
- Generic generator functions / generators inside a generic method.
- `async`/`await` - unrelated feature, no forcing use case, not the same
  mechanism (this plan is purely synchronous re-entry via explicit
  `__next__()` calls, no scheduler).
