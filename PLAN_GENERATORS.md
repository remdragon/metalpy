Generator functions (`yield`, state-machine transform)

STATUS: v1 + Phase 2 (while loops) + Phase 3 (`for`-loop consumption) +
Phase 4 (`for x in range(...):` containing yield) + Phase 5 (`for x in
<expr>:` containing yield, over a non-range() indexable OR another
generator) + Phase 6 (yield inside `if`/`if-else`, and yield wrapped in
an arithmetic-mode `with` block) + Phase 7 (generic generator functions,
`def gen[T](x: T) -> Iterator[T]:`, both explicit `gen[i32](...)` and
inferred `gen(...)` instantiation, interim-scoped to reject a body that
references its own type param outside a parameter/return annotation) +
Phase 8 (fallible generators, `Generator[T,E]`, `or_return()` inside a
generator body) landed and real-compile-and-run tested
(emitter_c_test.py's GeneratorFunctionTests). Phase 5 matches the
"remaining phases roadmap" section's own Phase 1, Phase 6 matches that
roadmap's own Phase 2, Phase 7 matches that roadmap's own Phase 3, and
Phase 8 matches that roadmap's own Phase 4 (below) - kept the SEQUENTIAL
landed-phase numbering here (v1, Phase 2, 3, 4, 5, 6, 7, 8) rather than
renaming it, since that roadmap's own 1-5 numbering is a separate, later
scoping pass over what was still left, not a renumbering of what had
already landed; the two schemes overlap in NAME but not in MEANING -
watch for this when
reading older commit messages/comments that say "Phase 1" or "Phase 2"
meaning something other than the roadmap's own numbering.

PLAN_GENERATORS.md's own motivating example now compiles and runs in its
most natural, idiomatic spelling: `for i in range(count): yield i`,
consumed the equally natural way: `for x in counter(5):`. `range()`
ITSELF has deliberately NOT been rewritten into a real generator, and
never will be - this is a permanent design decision, not a gap (see
ARCHITECTURE.md's own "design decision: range() stays a compiler
intrinsic" section, confirmed with the user 2026-08-15) - the sugar path
(`_is_range_call`/`_lower_for_range`) is untouched; Phase 4 only teaches
the generator machinery to RECOGNIZE and desugar a for-loop that happens
to iterate over a range() call, same as a user would write by hand today
outside a generator. Phase 5 does the same for a for-loop over anything
ELSE with a real for-loop shape (list-like, or another generator) - this
is what actually makes generators testable with realistic code (a
generator consuming a real collection, or composing another generator),
which is why the roadmap prioritized it first.

`yield` may be a direct top-level statement of the function body (v1),
the single yield inside a direct top-level `while` loop (Phase 2), or the
single yield inside a direct top-level `for` loop - over range() (Phase
4), a list-like indexable, or another generator's own `__next__()` (both
Phase 5) - the last two desugared to the Phase 2 while shape before
anything else runs, same as range() already was. A yield nested inside
an if/with/try, inside a for-loop over something with neither shape, or
inside a loop that has more than one yield or any USER-written
break/continue, is a clear compile
error, not a silently wrong state machine (see GeneratorFunctionTests'
own five rejection tests). Every unit shape verified for the RC-
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

PERMANENT, confirmed-with-the-user design decision (2026-08-15), not a
deferred follow-up: `range()` itself will NEVER be rewritten from
`_is_range_call` textual sugar into a real generator. The sugar has a
real, deliberate performance property (documented in its own comment: no
allocation, unchecked arithmetic proven safe by construction) that a
real generator's heap-allocated, refcounted backing object would give up
at every single call site - `range()` is the single most common loop-
counting construct in any real program, so this isn't a one-off cost.
This doc's own original framing ("lets range() stop being special-cased
syntax") was wrong to treat that as a win worth pursuing - it's cosmetic
at best, and actively costly given how pervasively range() is used. The
FUNCTIONAL goal this framing was actually chasing (a real, user-authored
range()-shaped generator working end to end, in its natural spelling)
is fully met by Phase 4 below - only the literal builtin `range()` stays
exempt. See ARCHITECTURE.md's own "design decision: range() stays a
compiler intrinsic" section and TODO.txt's "generators:" entry for the
same note in context.

Phase 4 design (type_resolver.py's `_desugar_generator_for_loops`/
`_desugar_range_for`): a top-level `for x in range(...): BODY` containing
exactly one yield is rewritten, IN PLACE, into the exact while-loop
equivalent (`x: usize = start; while x < stop: BODY; with compiler.
wrap_arithmetic: x += 1`) before unit collection ever runs - a pure AST-
to-AST desugaring, zero new state-machine logic, zero changes to any
Phase 2 code. Confirmed the emitted C is structurally IDENTICAL to the
hand-written while-loop version (same instruction numbering, same
`__gen_resuming_N` local) by inspecting it directly.

Phase 5 design (type_resolver.py's `_desugar_general_for`/
`_desugar_indexable_for`/`_desugar_iterator_for`) - the roadmap's own
Phase 1, landed the same session it was scoped in. `node.iter`'s type is
resolved via a NEW `_resolve_expr_type_for_desugar` (a standalone,
`.visit()`-never-called `_ReferenceResolver`, seeded with parameters plus
a permissive AnnAssign scan) reusing `_type_of_expr` - confirmed via
research that this already resolves arbitrary expressions, including
`obj.method()` calls recursing into the receiver, entirely from AST, no
lowering needed (it's what `visit_Match` already uses for a match
subject). `__len__`+`__getitem__` wins the SAME priority tie lowering.py's
own ordinary `_stmt_For` gives it against `__next__`... no wait, the
other way - `__next__` is checked FIRST, matching `_lower_for_over_
iterator`'s own priority over indexable for an ordinary for-loop.

Both desugared shapes hit a REAL fallibility snag the design didn't
originally anticipate: `list[T]`'s own `__getitem__` (and, unusually,
NOT its `__len__`, confirmed by inspecting the emitted C - only
`__getitem__` needed the fix) is genuinely fallible, `Result[T,
IndexError]`, and this is true for an ORDINARY (non-generator) for-loop
too, confirmed via a standalone repro (an ordinary for-loop over `list
[i32]` fails to compile inside a plain, non-Result-returning function
with the exact same error) - not something generators broke. `[]`
subscript syntax hard-codes PROPAGATION (needs the enclosing function to
be Result-shaped), which `$$__next__` never is in v1 - so the desugaring
calls `__getitem__`/`__len__` EXPLICITLY (not via `[]`) and passes each
through a new `_maybe_unwrap_call` helper, which panics via `.unwrap(msg)`
when the return type actually is `Result[T,E]`-shaped (structurally safe:
every call site here has an index strictly less than a just-read length,
same reasoning `_lower_for_range`'s own raw-AddWrap bypass already
relies on) and passes a non-fallible call through unchanged otherwise.

The iterated object itself (`__for_obj_N`) is typically RC-typed (a
list, or another generator) - a real complication the roadmap's own
Phase 1 write-up under-scoped (it flagged the per-ITERATION element as
possibly RC-typed and deferred that to Phase 5's OWN later item, but
missed that the ITERABLE'S OWN storage has the identical problem one
level up). Building the general state-gated destructor that item is
scoped for felt like too much for this pass, so `__for_obj_N` is instead
evaluated EAGERLY, in the generator's own CONSTRUCTOR (alongside its real
parameters - see `_rewrite_generator_constructor`'s own extra_fields
handling), rather than lazily on the first `__next__()` call the way
real Python would defer it. This makes `__for_obj_N` valid unconditionally
from construction onward, exactly like a captured parameter, so the
EXISTING unconditional `$$__destructor__` cascade handles it correctly
with ZERO new destructor machinery - the same "simpler thing turned out
sufficient" pattern as v1's own scalar-only-locals simplification.
The real, accepted semantic gap this leaves: if the iterated expression
has an observable side effect, it now happens at `gen(...)` call time
rather than at the first `.__next__()` call - noted in code (`_new_for_
obj_field`'s own docstring), not silently swallowed. Verified correct
for nested composition specifically (a generator consuming another
generator, itself capturing an RC parameter) - both levels release
correctly when abandoned mid-iteration, confirmed by a real refcount
check (`for_loop_over_nested_generator_releases_both_levels`).

The iterator shape's own payload extraction reuses the SAME `match
subject: case T(subject):` technique Phase 3 discovered, but as REAL
match-statement SOURCE syntax this time (not hand-built IR) - which
surfaced two more real snags: (1) the synthesized `case None: break`
tripped `_validate_while_yield_unit`'s existing break/continue rejection
(meant for genuine USER-written break/continue, which stays rejected) -
fixed via a `compiler_synthesized_break` tag on the ast.Break node
itself, checked before rejecting; (2) the intermediate `__for_next_N =
obj.__next__()` plain assignment tripped `_collect_generator_locals`'s
own "must be explicitly annotated" rule (it's not meant to be a promoted
local at all - it's an ordinary `$$__next__`-scoped temp, like
`_build_while_unit_guard`'s own `__gen_resuming_N`, just built one stage
earlier) - fixed via a matching `compiler_synthesized_for_loop_temp` tag.
Both are the SAME established "tag the AST node, check the tag" escape-
hatch convention this file already uses throughout (`resolved_callee`,
`generator_backing_cls`, etc.), not new mechanisms. The DEEPER lesson
both bugs share: raising early (via `discovery.fail`) from partway
through `ensure_generator_synthesized` leaves `fn.node.body` in a
half-desugared, half-renamed state that then gets INCORRECTLY processed
as an ordinary (non-generator) function body once dequeued later -
producing a confusing CASCADE of unrelated-looking downstream errors,
exactly like the very first walk-order bug found while landing v1. Worth
remembering next time a new failure mode here produces a weird error
list: check whether it's actually just ONE early failure cascading.

Phase 6 design (type_resolver.py's `_build_if_unit_guard`, `_validate_
if_yield_unit`, `_if_yield_nodes`, `_yield_with_wrapper`,
`_arithmetic_mode_with_kind`): landed exactly the "recommended first
cut" scoped in the roadmap below - a single top-level `if`/`if-else`,
at most one yield per branch, no elif chains and no nested loops-inside-
branches (both explicit compile errors, see GeneratorFunctionTests'
`test_if_elif_chain_with_yield_is_rejected` and `test_if_else_with_two_
yields_in_one_branch_is_rejected`). An if-unit occupies TWO states, same
as a while-unit, but does NOT loop - `if state <= start+1: [state ==
start: preamble]; resuming = (state == start+1); if cond: <branch-A>
else: <branch-B>; state = end`, where each branch is itself `if
resuming: <post-yield stmts> else: <pre-yield stmts>; state = start+1;
return value`. Safe to re-evaluate `cond` on resume because nothing but
the generator's OWN code touches its fields between `__next__()` calls -
whichever branch a call's `cond` selected is still the branch that will
be selected on the resuming call, so `resuming` doesn't need to be
per-branch, just one shared flag. A non-yielding branch needs no resume
handling at all - it's structurally unreachable when `state == start+1`
(you can only be resuming a branch that itself yielded to get there).
Verified: yield in both branches with different post-yield code
(`side = 100`, unobservable directly but proves no crash/corruption on
resume), falling through correctly to a shared tail unit after the
if/else, exhaustively calling `.__next__()` through both branches on
separate generator instances (`if_else_yield_resumes_correct_branch_and_
falls_through`).

`with`-wrapped yield (`_yield_with_wrapper`/`_arithmetic_mode_with_kind`,
mirroring lowering.py's own textual arithmetic-mode recognition) turned
out to be exactly as low-risk as scoped: no new unit kind at all - it's
still a `('yield', stmt)` unit, just with `_build_yield_unit_guard`
taught to accept `ast.With` as well as `ast.Expr` and re-wrap the
generated `state = N+1; return value` pair back inside the same `with`
block so the arithmetic mode stays correctly scoped once the segment is
actually lowered (`with_wrapped_yield_units` test - two separate
with-wrapped yields in one generator, each with real i32 arithmetic in
between, confirming the mode wrapping round-trips correctly both times).

Phase 7 design (monomorphize.py's `substitute_type_params` GeneratorType
branch; type_resolver.py's `ensure_resolved` Specialization branch,
`_ReferenceResolver.visit_Call`/`_type_of_expr`, `ensure_generator_
synthesized`'s new `origin_type_param_stems` parameter): landed generic
generator functions - `def gen[T](x: T) -> Iterator[T]:`, both explicit
(`gen[i32](...)`) and inferred (`gen(local_var)`) instantiation, real-
compile-and-run tested including two independent instantiations
coexisting and for-loop consumption of one. The groundwork sketched in
the roadmap below turned out to need MORE fixing than anticipated, all
found via real repros rather than more up-front analysis, per the
roadmap's own recommendation:

1. `substitute_type_params`'s new GeneratorType branch (needed the same
   shape as CallableType/ClosureType - substitute elem_type, rebuild a
   fresh instance since GeneratorType is deliberately never interned)
   required all four of GeneratorType's own `Name`-inherited kw-only
   fields (stem/qualname/file/line), not just elem_type - confirmed by a
   real TypeError, fixed by mirroring discovery.py's own `Iterator[T]`
   construction exactly.

2. `ensure_resolved`'s Specialization branch DOES need `ensure_generator_
   synthesized` applied to the monomorphized Function it returns (as
   scoped) - but this branch turned out to never actually fire for an
   ordinary nested call site (`main`'s own body calling `gen[i32](...)`)
   at all: `_ReferenceResolver.visit_Call`'s own generic-call resolution
   deliberately builds the monomorphized copy directly, bypassing ensure_
   resolved entirely (see its own comment - avoiding double-scheduling).
   Needed the identical `ensure_generator_synthesized` call added there
   too, independently - both call sites are safe to call unconditionally
   since the underlying monomorphized Function is memoized (`spec.
   monomorphized`) and `ensure_generator_synthesized` is itself id(fn)-
   memoized, so whichever path reaches a given instantiation first does
   the real work and the other is a no-op.

3. `ensure_generator_synthesized`'s own `fn.type_params` check had to
   change from a hard rejection to a silent skip (mirroring _schedule_
   rcclass_destructor_deps's identical posture for a generic class's own
   abstract template) - it turned out to be reached constantly and
   harmlessly on the ABSTRACT, still-generic Function itself (not just
   on genuinely-unsupported shapes), e.g. from `_type_of_expr`'s Call
   handling on a path that hadn't yet been taught about generics either
   (next point) - a hard failure there rejected the very first `gen(...)`
   call in the program.

4. `_type_of_expr`'s own Call handling had NO `ast.Subscript` case at all
   (`gen[i32](...)`'s own `node.func`), and its bare-Name fallback
   resolved a call like `gen(...)` to the STILL-ABSTRACT generic Function
   rather than doing generic-call resolution - meaning `g1 = gen[i32](5)`
   never got g1's real (post-synthesis) type tracked for narrowing
   purposes at all, silently breaking every subsequent `g1.__next__() is
   None` check downstream (confirmed by a real miscompile - clang
   rejecting a raw union-struct compared against a bare `0`). Fixed by
   reusing `node.resolved_callee` (already tagged by visit_Call, which
   always runs first - `_type_of_expr` is only ever called after
   `generic_visit` has already visited the same Call node) rather than
   re-deriving anything.

5. The roadmap's own flagged "open, unverified risk" (rewrite-3 ordering)
   turned out to be a REAL bug, not just a theoretical one, confirmed by
   the recommended minimal repro: a generic generator body calling
   another generic function via its own type param (`y: T = identity(x)`
   inside `gen[T]`) fails with a confusing "name 'T' is not defined',
   because `_build_generator_next_function` copies the body's raw
   statements into a FRESH `__next__` method/backing-class scope that
   never inherits the `T -> concrete-arg` substitution recorded only on
   the outer generator Function's own `.names`. Landed the roadmap's own
   recommended interim scope exactly: reject such a body up front with a
   clear message (a raw AST scan of the function's own body statements
   for the origin type param's bare name, threaded down from each of the
   two call sites in point 2 above, which each already have the abstract
   base's own `.type_params` in hand) rather than let it cascade into
   the same confusing downstream errors - see `ensure_generator_
   synthesized`'s own docstring/comment for what lifting this properly
   would need (threading the substitution through the synthesized `__next__`/
   backing class's own names, not attempted here).

Separately confirmed, NOT a generator-specific bug: a bare int LITERAL
argument to an INFERRED (no explicit `[T]`) generic call already fails
type inference in this compiler, even for an ordinary non-generator
generic function (`ident(7)` fails the same way `ident[T](x: T) -> T:`
does) - a typed local works fine. Out of scope here; noted for whoever
next touches generic-call inference.

Phase 8 design (discovery.py's `Generator[T,E]` recognition;
mpy_types.py's `GeneratorType.error_type`; monomorphize.py's
`substitute_type_params` GeneratorType branch, extended;
type_resolver.py's `ensure_generator_synthesized`,
`_pessimistic_done_prefix`, `_wrap_generator_next_returns_in_ok`, and all
three guard builders): landed fallible generators - `Generator[T,E]` is
the fallible sibling of `Iterator[T]` (same textual recognition, one
extra type arg), whose `__next__` returns `Result[elem_type|None,E]`
instead of the bare union. `or_return()`/unguarded checked arithmetic
inside the body engage the EXISTING `_require_result_return` machinery
for FREE - no new Function flag, no special generator-side check at all;
it's purely a consequence of `__next__`'s own declared return type,
identical to how any other ordinary fallible function already works. A
plain `Iterator[T]` generator continues to reject both, unchanged, for
the SAME reason (its `__next__` isn't Result-shaped).

The "no existing analog to reuse" risk flagged in the roadmap below
turned out to have a much simpler answer than the IR-level one
originally sketched (a new OrReturn.epilogue-injected SetAttr, or an
errdefer-style OrJump repurposing) - realized once actually design
before implementing: **set "permanently done" BEFORE running any block
of user code that might fail, not after.** Every unit's own guard
already has exactly one success path (advance state, then yield/fall
through) - inserting `self.__state = <done>` as the FIRST statement of
every block that MIGHT contain a fallible early return means an
or_return()'s own OrReturn (completely UNMODIFIED, zero new IR) already
does the right thing: it returns Err(...) immediately, and self.__state
is ALREADY the permanent-done value at that exact moment, because
nothing runs between the pessimistic write and the point of failure that
could still succeed. The unit's own EXISTING success-path state-advance
(already present, unchanged) simply overwrites the pessimistic value
right before yielding - a pure reordering of existing AST, no new
IR/lowering machinery, no way to distinguish "was this an error exit"
needed at all. The one wrinkle: the real "done" state value isn't known
until AFTER every unit is built (it's `final_state + 1`), but each
unit's own pessimistic assignment has to be inserted WHILE building that
very unit - solved by recording each inserted Assign node in a shared
list (`pending_done_assigns`) and patching every one's `.value` to the
real done_state once it's known, rather than trying to compute it
earlier or share one mutable Constant node across every insertion point.

Every block of user code that can run before a unit's own success path
needed this treatment - more insertion points than the "one obvious
spot" intuition suggests: a bare yield-unit's own preamble (one spot); a
while-unit's `pre` (first entry only), its post-yield resume code, AND
its pre-yield code each iteration (three spots); an if-unit's shared
outer preamble, plus per-branch pre-yield/post-yield code for a yielding
branch OR the whole body for a non-yielding one (up to five spots across
both branches). All covered via one shared helper
(`_pessimistic_done_prefix`), not duplicated per builder.

Every `ast.Return` in the fully-assembled `__next__` body (the DONE
short-circuit, every yield-unit's own return, the tail's, the safety
net's) then needs wrapping in `Result.Ok(...)` - done as ONE uniform
post-process pass over the whole assembled body (`_wrap_generator_next_
returns_in_ok`, a plain `ast.walk` - safe here specifically because a
generator body can never contain a nested def/lambda, unlike
`_walk_generator_body`'s own careful non-recursion elsewhere in this
file) rather than threading Result-wrapping through every individual
guard builder. `Result.Ok(...)`'s own payload argument coerces the
ordinary way (same `_lower_expr(arg,expected_type)` machinery any other
call argument gets - confirmed via a real repro that a bare elem_type
value AND a bare `None` constant both coerce correctly with no special
handling needed here).

Once "permanently done" fires (via error OR normal exhaustion), EVERY
later `.__next__()` call returns `Ok(None)` - it does NOT re-surface the
specific error value again. This is a deliberate simplification, not an
oversight: re-surfacing the same `Err` on every subsequent call would
need an extra stored field (the pending error value) for a purely
cosmetic gain, and the roadmap's own verification bar explicitly accepts
"or is otherwise well-defined" - not strictly "returns Err again" -
which `Ok(None)` forever after satisfies with zero extra machinery,
reusing the exact same done-state short-circuit normal exhaustion
already needed.

Verified via real compile-and-run: `or_return()` propagating a real,
properly-allocated error class through two successful yields then a
failure, then permanent `Ok(None)` on every subsequent call (not a
crash, not a re-run of the failing code); RC correctness (a captured RC
parameter still releases correctly via the ordinary $$__destructor__
cascade when the generator is abandoned after an or_return() error, not
just after normal exhaustion); `Iterator[T]` (infallible) continuing to
reject `or_return()` exactly as before this phase, unchanged.

**Found, but explicitly out of scope, a real PRE-EXISTING bug unrelated
to generators:** unguarded (Check-mode) arithmetic that overflows
produces a `Result[T,OverflowError]` whose Err payload is genuinely
UNINITIALIZED (emitter_c.py's `_emit_check_arith` only ever sets the tag
on overflow, per `_emit_widen_error`'s own "zero-payload marker" design
intent for the built-in arithmetic error classes) - and nothing in the
RC-cleanup codegen honors that assumption: the moment such a Result
value's own scope ends, ordinary decref dereferences the garbage
pointer and crashes (confirmed via TWO minimal repros, both crashing
with STATUS_ACCESS_VIOLATION, NEITHER involving a generator at all - a
plain function returning `Result[i32,OverflowError]` from unguarded
overflow, consumed via `match ... case Result.Err(e):` OR merely
`.is_err()`, both crash once the value's own scope ends). This is why
this phase's own positive tests use `or_return()` with a REAL,
properly-allocated error class (`@union ... Boom: None`) throughout,
never the built-in arithmetic error classes directly - the mechanism
this phase actually built (pessimistic-done + Result.Ok wrapping) is
NOT the cause and structurally works fine either way (confirmed by
tracing the generated C by hand before finding the crash was in
_emit_check_arith, entirely outside anything this phase touched) - this
is a pre-existing gap in the CORE checked-arithmetic-error machinery,
flagged as a separate task, not fixed here.

Remaining phases roadmap (scoped 2026-08-15)

Phase 1: LANDED (same session it was scoped in) - see "Phase 5 design"
above (kept the sequential landed-phase numbering there; this roadmap's
own 1-5 numbering is a separate scoping pass, not a renumbering - see
the STATUS section's own note on why the two schemes overlap in name but
not meaning). `for x in <expr>:` over a non-range() iterable inside a
generator body, both the indexable shape (`__len__`+`__getitem__`) and
the iterator shape (`__next__() -> T|None`, i.e. one generator consuming
another), real-compile-and-run tested including nested RC correctness.

Phase 2: LANDED (same session it was scoped in) - see "Phase 6 design"
above (kept the sequential landed-phase numbering there; see the STATUS
section's own note on why the two schemes overlap in name but not
meaning). yield inside `if`/`with`. ("try" doesn't exist in this
language - no exception handling anywhere in lowering.py's statement
dispatch; the closer analog, `with defer/errdefer:`, stays out of scope,
see below.) Landed exactly the recommended first cut: a single if/else,
at most one yield per branch - elif chains and nested loops-inside-
branches are explicit compile errors, not yet supported.

Phase 3: LANDED (same session it was scoped in) - see "Phase 7 design"
above (kept the sequential landed-phase numbering there; see the STATUS
section's own note on why the two schemes overlap in name but not
meaning). generic generator functions (`def gen[T](x: T) -> Iterator[T]:`),
both explicit and inferred instantiation. Landed exactly the recommended
interim scope: a generic generator body that references its own type
param outside a parameter/return annotation (e.g. calling another
generic function through it) is a clear compile error, not yet
supported - the roadmap's own flagged ordering risk turned out to be a
real bug, confirmed by exactly the minimal repro recommended below.

Phase 4: LANDED (same session it was scoped in) - see "Phase 8 design"
below (kept the sequential landed-phase numbering there; see the STATUS
section's own note on why the two schemes overlap in name but not
meaning). fallible generators (`Generator[T, E]`, TODO.txt's original
open question) - the one piece of this whole roadmap flagged as having
"no existing analog to reuse" turned out to have a MUCH simpler fix than
the IR-level one sketched here originally: a pure AST-level reordering
(see "Phase 8 design"), not a new OrReturn/OrJump/IR primitive at all.

Phase 5 (not explicitly requested, proposed as the most load-bearing
remaining gap): RC-typed locals crossing a yield. v1 restricted promoted
LOCALS to scalar types specifically because a not-yet-initialized RC
field would make the ordinary unconditional $$__destructor__ cascade
decref garbage - the original plan's own state-gated-destructor sketch
(below) is the real fix, computed from the same per-field "valid from
state N onward" data the liveness/promotion step already has. Landing
this lifts the scalar-only restriction everywhere it currently applies,
including Phase 1's own iterator-consumption element type.

Explicitly not planned, no forcing use case: `yield from`; `.send()`/
`.throw()`/`.close()`; defer/errdefer inside a generator body (a real,
already-flagged combinatorial hazard, see "Body restrictions for v1"
below); generator methods (a generator must stay a plain function for
now, same posture as PLAN_CALLABLE.md/PLAN_LAMBDA.md's own deferred
closures); async/await (unrelated mechanism entirely).

Original planning notes follow, kept for historical context and for the
phases not yet attempted (the fallible-generator sketch below predates,
and is superseded in detail by, the Phase 4 roadmap entry just above).

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
