Generator functions (`yield`, state-machine transform)

> **Note (2026-08-19): Phase F, B, and now C (`.send()`) have all been
> REIMPLEMENTED against current master - only the A.4a follow-up (`yield
> from`) has not.** History recap: Phase F/B/C/A.4a were originally
> built, merged to master (a4a1d5d), then silently discarded by the next
> merge (1a89a86) before anyone noticed. By the time that was caught,
> master had diverged too far (150+ commits touching the exact substrate
> Phase F depends on) for the old branch to be reapplied as a patch, so
> Phase F/B were rebuilt FRESH first (worktree `generator-phase-f-
> rebuild`), then Phase C on top of that rebuild (worktree `generator-
> phase-c-send`, same day) - both using the sections below as a design
> REFERENCE, not a diff. Both reach the same destination the original
> branch describes, but internal names differ throughout - see `type_
> resolver.py`'s `_assign_generator_yield_dispatch`/`_build_generator_
> next_function`/`_build_generator_send_wrappers`/`_hoist_yield_from_rc_
> reassignment` and `lowering.py`'s `_lower_generator_yield`/`_expr_
> Yield`/`_emit_generator_dispatch_prologue` for the ACTUAL current
> mechanism; the design sections below are kept for their reasoning, not
> as a literal function-by-function map. Phase B's own nesting/
> multiplicity verification is covered by `emitter_c_test.py`'s `test_
> previously_rejected_shapes_now_compile_and_run`; Phase C's own `.send()`
> tests are `test_send_scalar_and_rc_values`/`test_send_before_first_
> yield_panics`/`test_bare_next_at_captured_yield_panics`.
>
> Four real, generator-unrelated bugs were found (and fixed) across both
> rebuilds, via real compile-and-run testing exactly like the original
> build: (1) a yield reached outside a successfully-synthesized generator
> needs a graceful `discovery.fail()`, not a raw crash, when an earlier,
> unrelated error left synthesis only partially done; (2) `cfg.py`'s
> `merge_loop_exits()` wiped `self._live` to empty (rather than
> preserving the loop's own entry snapshot) whenever a `while True:` loop
> had no `break` at all - harmless for genuinely dead code in an ordinary
> function, but wrong the moment code after such a loop is actually
> reachable (a generator's own synthesized tail, in particular); (3)
> `_lower_generator_yield`/`_expr_Yield` need the same `_cfg.untrack_
> temp(value)` call `_stmt_Return` already makes before its own temp
> flush, or the flush immediately decrefs the very value a union-
> coercion's own constructor just increfed, silently cancelling it out -
> confirmed via a real `compiler.refcount()` repro; (4) a separate,
> unrelated bug found and fixed the same day as the Phase F rebuild:
> negating a value (`-x`) directly into a union return/yield type
> produced invalid C - see `_expr_UnaryOp`'s own `operand_hint` comment.
>
> **The A.4a follow-up (`yield from`) is still NOT implemented** -
> `TypeResolver._reject_generator_yield_from` still rejects it cleanly.
> The A.4a design section further down describes what the original
> branch built and could still guide a future attempt, but nothing there
> reflects current code either.

STATUS: v1 + Phase 2 (while loops) + Phase 3 (`for`-loop consumption) +
Phase 4 (`for x in range(...):` containing yield) + Phase 5 (`for x in
<expr>:` containing yield, over a non-range() indexable OR another
generator) + Phase 6 (yield inside `if`/`if-else`, and yield wrapped in
an arithmetic-mode `with` block) + Phase 7 (generic generator functions,
`def gen[T](x: T) -> Iterator[T]:`, both explicit `gen[i32](...)` and
inferred `gen(...)` instantiation, interim-scoped to reject a body that
references its own type param outside a parameter/return annotation) +
Phase 8 (fallible generators, `Generator[T,E]`, `or_return()` inside a
generator body) + Phase 9 (RC-typed locals crossing a yield - the LAST
item on the original roadmap) landed and real-compile-and-run tested
(emitter_c_test.py's GeneratorFunctionTests). Phase 5 matches the
"remaining phases roadmap" section's own Phase 1, Phase 6 matches that
roadmap's own Phase 2, Phase 7 matches that roadmap's own Phase 3, Phase
8 matches that roadmap's own Phase 4, and Phase 9 matches that roadmap's
own Phase 5 (below) - kept the SEQUENTIAL landed-phase numbering here
(v1, Phase 2, 3, 4, 5, 6, 7, 8, 9) rather than renaming it, since that
roadmap's own 1-5 numbering is a separate, later
scoping pass over what was still left, not a renumbering of what had
already landed; the two schemes overlap in NAME but not in MEANING -
watch for this when
reading older commit messages/comments that say "Phase 1" or "Phase 2"
meaning something other than the roadmap's own numbering.

Past the original roadmap's own 9 phases, `defer`/`errdefer` support
inside a generator body has ALSO landed (own separate mini-plan, not
part of the numbered sequence above - see "defer/errdefer phase design"
below), including a prerequisite fix (a bare `return` inside a generator
body now correctly ends iteration permanently, not just once).

Past THAT, a second major rebuild has landed (reimplemented from scratch
2026-08-19, after the original build was lost to a merge conflict - see
this doc's own top-of-file note): **Phase F** replaced the entire AST-
synthesis dispatch mechanism (the old `_collect_generator_units` and its
per-shape guard builders - the flat-unit-recognition machinery every
phase above this point was built on top of) with a real IR-level
`ir.Yield` + `self.__state` goto/label dispatch, built directly by
lowering.py instead of hand-assembled AST. This is what finally lifted
the structural restrictions every phase above inherited from the unit
model: multiple yields per loop/branch, yield nested at ARBITRARY depth
(if-in-while, while-in-if, three-plus levels), `elif` chains containing
yield, and `break`/`continue` inside a yield-containing loop are all now
ordinary compile-and-run cases, not compile errors - real-compile-and-run
verified via `emitter_c_test.py`'s `test_previously_rejected_shapes_now_
compile_and_run` (the **Phase B** nesting/multiplicity verification bar),
all 3 compilers. See "Phase F design" below for the ORIGINAL build's own
reasoning (still broadly accurate) - its literal internal names describe
the original, now-superseded implementation; see this doc's own top note
for where the CURRENT mechanism actually lives.

**Phase C (`.send()`) has ALSO landed** (reimplemented from scratch,
same day, on top of the Phase F/B rebuild above): `Generator[T,SendType,
E]` (3-arg form - `T`/`E` are the existing `elem_type`/`error_type`,
`SendType` new, inserted in the middle) makes `(yield expr)` usable as
an EXPRESSION, evaluating to plain `SendType`, delivered via `.send(v)`.
Backing-class shape and the overall `__next__()`/`send(v)` wrapper split
over a real `$$__resume__` match the ORIGINAL design almost exactly (see
"Phase C design" below for the reasoning) - `type_resolver.py`'s
`_build_generator_send_wrappers` builds the two thin wrappers, and a new
`_hoist_yield_from_rc_reassignment` pass (this rebuild's own name for
the original's `_build_liveness_guard` restructuring - see bug #1 in
"Phase C design") avoids duplicating a captured yield when
`_apply_live_flag_guards` would otherwise deep-copy it for an RC-typed
promoted local's own first-assignment branch. Verified via real
compile-and-run refcount-delta checks matching the original design's own
bar exactly: an RC-typed `Box` sent through `.send()` into a captured
`held = yield i` ends up with THREE independent owners (caller, `__
send_slot`, `held`), a second `.send()` drops the first back to one and
brings the new one to three, `.send()` before the first yield panics,
and resuming a captured yield via bare `.__next__()` panics too. Unlike
the original build, there is no separate 3-arg-vs-2-arg `SendType`-is-
independent-of-`E` nuance write-up needed here - it carries over
unchanged (see "Phase C design" below).

**The A.4a follow-up (`yield from`) did NOT come back with either
rebuild.** `yield from` is still a clean, explicit rejection
(`TypeResolver._reject_generator_yield_from`). The A.4a section below
describes what the ORIGINAL branch built, kept as a design reference for
whoever picks this back up, not as a description of anything currently
compilable.

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

Phase 9 design (type_resolver.py's `_live_flag_stem`,
`_build_generator_backing_class`'s live-flag fields, `_rename_and_track_
liveness`, `_maybe_route_yield_through_temp`, `_build_generator_
destructor`; lowering.py's `_stmt_Assign`'s new `generator_first_rc_
assign` branch and `_expr_Constant`'s `generator_zero_rc_field`
exemption): landed RC-typed locals crossing a yield - the last item on
the whole roadmap. Lifted `_collect_generator_locals`'s scalar-only
check entirely (any type now allowed for a promoted local); this alone
also lifts Phase 1/5's own for-loop element-type restriction, since the
loop target is just another promoted local by the time it reaches that
check - confirmed via a real for-loop-over-`list[RCClass]`-inside-a-
generator repro, no separate code change needed for that part.

**Follow-up cleanup (post-Phase 9): `_maybe_route_yield_through_temp`
removed entirely.** It existed only to dodge three real, generator-
unrelated RC bugs in how a bare value coerces into a declared union
return type (`return self.<field>`, `return <bare tracked value>`, and
an AnnAssign coercing a field read into a union local, all documented in
its own now-deleted docstring). All three were root-caused and fixed by
048af0f ("Fix double-incref/masked-decref when coercing a value into a
union type" - `lowering.py`'s `_is_aliasing_expr` now checks the actual
coerced operand via `_coerce_into_union`'s own `is_union_coerce_result`
tag, not the pre-coercion ast node), on a branch this generator work
hadn't merged yet. Confirmed safe via a direct repro of the exact
`return self.<field>` shape plus a new real-compile-and-run regression
test (`emitter_c_test.py`'s `yielded_rc_value_increfs_exactly_once_
caller_side`) before removing the routing and the `elem_type`/
`elem_is_rc` plumbing that only ever fed it from `_build_yield_unit_
guard`/`_build_while_unit_guard`/`_build_if_unit_guard`/
`_build_generator_next_function` - a yield's own return value now flows
through unmodified, same as any other union-typed return.

**Design pivot from the original sketch**: the ORIGINAL plan (see below)
sketched a per-STATE validity table (a `switch` on `self.__state`,
decref-ing exactly the fields "valid from state N onward", computed from
CFG liveness). Working through the real cases - specifically, a local
assigned only in a while-unit's POST-yield code (`post_iter_stmts`),
which does NOT run on the very first pass through the loop - surfaced a
real correctness gap the per-state model can't express: `self.__state`
alone can't distinguish "paused right after the first yield, post-yield
code never run yet" from "paused after a later iteration where it HAS
run", since both leave `self.__state` at the exact same value. A
per-FIELD boolean live-flag (`__<stem>_live: bool`, false until the
field's own first real assignment, never reset false again since
ordinary reassignment already correctly decrefs the old value) sidesteps
this entirely - correctness no longer depends on WHICH unit/branch/pre-
or-post-yield-position a local's assignment happens to live in, just
"has this specific field literally been written yet." Simpler to build
and provably correct against every shape tried, at the cost of one extra
bool field per RC-typed promoted local (this codebase's own
`_collect_generator_locals` already deliberately over-promotes rather
than tracking precise liveness, so this fits its own established
philosophy).

**Zero-value placeholder**: a promoted RC-typed field needs SOME value
at construction time (every field is required by the no-`__init__`
construction sugar this generator's own constructor already used, with
no existing way to omit one). A bare `0` fails ordinary type-checking
for an RCClass field (confirmed: "an int literal cannot be used where
X is expected") - `_expr_Constant` gained a narrow, compiler-internal-
only exemption (`generator_zero_rc_field`, checked alongside the
pre-existing `Ptr[T]`/`ConstPtr[T]` literal exemption, but NOT merged
into it - deliberately kept as its own separate tag so ordinary user
code still can't write `b: SomeClass = 0` as a novel "null RCClass"
idiom; RCClass values are never null anywhere else in this language).
This placeholder is NEVER read as a real value (the live-flag guarantees
that), only ever passed to `release_object` - which is itself already
null-safe at runtime (`if (obj && ...)`) - so this part alone would have
been harmless. It wasn't: a SEPARATE bug surfaced downstream (below).

**The real bug found, and its fix**: ordinary in-body reassignment of
the SAME promoted local (its own declaring statement, textually once in
source but re-executed every loop iteration at runtime) unconditionally
reads the field's CURRENT value and decrefs it before storing the new
one - correct once a real prior value exists, but on the field's
DYNAMICALLY first-ever execution (which the live-flag, not source
position, is what actually identifies - confirmed via a real repro that
a naive "textually-first-occurrence" rule breaks starting the second
loop iteration), the field is still the placeholder, and computing
`&(NULL)->$header` to decref it is undefined behavior per the C
standard even though `release_object`'s own runtime check makes it
harmless in practice - confirmed as a real UBSan trap
(`-fsanitize=undefined -fsanitize-trap=undefined`, this project's own
default test build flags) even though a plain non-sanitized build ran
the same path without visibly crashing. Fixed by splitting every
reassignment of an RC-typed promoted local into `if self.__<stem>_live:
<ordinary reassignment> else: <a differently-tagged assign, generator_
first_rc_assign> ; self.__<stem>_live = True` - the tagged branch skips
the old-value read/decref entirely, reproducing only the "adopt a fresh
value" half via the same public `cfg.py` helpers (`incref`/
`untrack_temp`) `_stmt_Return` already uses for an analogous "move
ownership in, no bindings tracking" situation - deliberately NOT routed
through `cfg.attr_assign` (the mechanism `__init__` construction uses
for the identical-looking need), since that pushes a `'self.<attr>'`-
keyed entry onto the epilogue/bindings stack scoped to real constructor
lowering - reusing it inside an ordinary synthesized `if/else` (not
`__init__`) made `merge_if()` see that binding as fresh on only one
branch and raise a real `KeyError` trying to reconcile it, confirmed via
a direct repro.

**Yielding an RC value needs an explicit incref** (the user's own
framing, confirmed exactly right): the caller receives a new, counted
reference while the generator's own field keeps its own - both alive
independently. This surfaced the session's THIRD and FOURTH pre-existing,
generator-unrelated RC bugs (flagged separately, not fixed here): plain
`return self.<field>` doesn't incref at all; and coercing a value into a
declared union return type (`__next__`'s own `elem_type|None` shape)
loses the incref even for an ordinary tracked parameter/local, AND
separately, coercing a value into a union assignment target only
correctly increfs when the source is itself already a tracked binding
(a Name), not an untracked expression like a field read. All three are
sidestepped by chaining two ALREADY-correct steps rather than fixing any
of them: `__yield_raw_N: elem_type = <yielded>` (a bare-typed local from
whatever expression, confirmed to always incref correctly regardless of
source), then `__yield_val_N: elem_type|None = __yield_raw_N` (coercing
a TRACKED LOCAL, not a field read, into the union - confirmed this
specific shape increfs correctly), then `return __yield_val_N` (a plain
move of an already-union-typed tracked local - the one return shape
this compiler already gets right, per every existing test in
or_return_rc_test.py). Verifying this took real trial and error against
precise `compiler.refcount()` deltas - repeated false alarms came from
two easy-to-miss, already-established compiler behaviors: `case T(name):`
match extraction always takes its OWN additional incref on top of
whatever the subject already holds, and an earlier match statement's own
hidden subject temp stays alive until the ENCLOSING FUNCTION's scope
ends, not just past its own match statement - both inflate a naive
before/after count by +1 in ways that look like bugs but aren't. The
project's own established RC-test style (a before/after delta around an
isolated helper call, not precise intermediate counts - see
`dropped_mid_iteration_decrefs_captured_parameter`) sidesteps both
pitfalls and is what this phase's own tests use.

Verified via real compile-and-run: an RC-typed promoted local reassigned
fresh every loop iteration, values read back correctly (not corrupted)
across multiple reassignment cycles; a generator dropped mid-iteration
with its own promoted-local field still holding a live RC value,
confirmed released via the live-flag-gated destructor without touching
an unrelated object's refcount; a captured parameter repeatedly yielded
and still correctly released on drop; a for-loop over `list[RCClass]`
inside a generator, both fully drained (values correct) and refcount-
verified.

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
dispatch; the closer analog, `with defer/errdefer:`, stayed out of scope
at the time this phase landed - see "defer/errdefer phase design"
below for where it later landed.) Landed exactly the recommended first cut: a single if/else,
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

Phase 5: LANDED (same session it was scoped in) - see "Phase 9 design"
below (kept the sequential landed-phase numbering there; see the STATUS
section's own note on why the two schemes overlap in name but not
meaning). RC-typed locals crossing a yield - not explicitly requested,
proposed as the most load-bearing remaining gap, and the LAST item on
this whole roadmap. Landed via a live-FLAG-gated destructor instead of
the state-gated one originally sketched here (see "Phase 9 design" for
why - a per-field boolean turned out simpler and more robust than a
per-state validity table, once the actual edge cases were worked
through). Lifts the scalar-only restriction everywhere it applied,
including Phase 1's own for-loop element type.

Explicitly not planned, no forcing use case: generator methods (a
generator must stay a plain function for now, same posture as
PLAN_CALLABLE.md/PLAN_LAMBDA.md's own deferred closures); async/await
(unrelated mechanism entirely). (`defer`/`errdefer` inside a generator
body WAS in this "not planned" list - it has since landed, own separate
mini-plan below. `yield from`, `.send()`, and `.close()` WERE also in
this list - all three have since landed too, see "Phase C design"/A.4a
follow-up/A.4b below; `.throw()` alone was explicitly, permanently
rejected during Phase C's own scoping - this language has no exception
handling at all, and bolting one on just for generators would contradict
this whole plan's own "reuse existing machinery" discipline. `Generator[
T,SendType,E]`'s own `SendType` declared as `Result[V,Err]` is the
supported alternative for error-injection-shaped needs - see "Phase C
design".)

defer/errdefer phase design (own separate mini-plan, past the original
9-phase roadmap above)

The user's own framing, confirmed exactly right as the mechanism's own
starting point, but only half the story: a generator's synthesized
`$$__destructor__` IS where abandonment-time cleanup belongs, but a
generator's own LOGICAL "call" (for defer purposes) spans MANY
`$$__next__()` invocations, most of which are yield-SUSPENDS, not exits
- naively letting ordinary (non-generator) defer machinery see a yield's
own synthesized `self.__state = N; return expr` would fire the defer on
every suspend, not just a real exit. Landed in two genuinely different
mechanisms, matching the two ways a generator's own call can actually
end:

**Prerequisite, found during scoping, fixed first, own commit**: a bare
`return` inside a generator body compiled but never set `self.__state`
to the DONE sentinel (`_reject_generator_value_return` only ever
rejected a VALUE return) - a later manual `.__next__()` call would
wrongly resume and re-run code. Fixed via `_rewrite_generator_bare_
returns`/`_rewrite_bare_return_stmts` (type_resolver.py): every bare/
explicit-`return None` reachable in the body - including nested inside a
while-unit's own loop body or an if-unit's own branch, which nothing
validated for this shape before - gets rewritten into `self.__state =
<placeholder>; return None`, using the exact `pending_done_assigns`-
style patch-later list `_pessimistic_done_prefix` (Phase 8) already
established, since the real done_state isn't known until every unit is
built. A truly bare `return` (no expression at all) ALSO needed its
`.value` normalized to an explicit `ast.Constant(None)` - a real,
separate finding: it lowers to a void C `return;`, which doesn't compile
against `$$__next__`'s own never-void declared return type (confirmed
via a real repro that failed with "non-void function ... should return
a value").

**Mechanism 1 (normal exits - tail exhaustion, a bare-return exit,
abandonment)**: pure AST synthesis, no lowering.py involvement, exactly
this whole plan's established discipline. `_desugar_generator_defer_
sites` replaces each top-level `with defer:`/`with errdefer:`/
`defer(...)`/`errdefer(...)` site (validated to be exactly that - a
direct top-level statement, i.e. living in a preamble or the tail; never
inside a while-unit's own loop body or an if-unit's own branch,
`_validate_generator_defer_sites` - same "start narrow" posture as
break/continue inside a yield-containing loop) with `self.
__defer_armed_N = True`, capturing the site's own body separately.
`_build_defer_replay_guards` builds `if self.__defer_armed_N: <body>` -
LIFO, and PLAIN `defer` only (never `errdefer` - see Mechanism 2) -
deep-copied fresh per insertion site (mutable per-occurrence lowering
attributes like `resolved_*` would corrupt a shared node otherwise,
same reasoning `_rename_and_track_liveness` already documents), inserted
at the tail's own natural-exhaustion exit, every bare-return exit (once
the prerequisite above lands), and `$$__destructor__` (before its
existing field teardown - safe under the preamble/tail-only restriction,
since a defer site can only ever reference a local/parameter declared
before it in program order).

**A real double-replay bug found and fixed here**: a plain `defer`
replayed via Mechanism 1 (say, at natural exhaustion) pins `self.__state`
to done but does NOT free the generator object itself - whatever
reference the caller still holds keeps it alive until IT drops, at which
point `$$__destructor__` runs and, without a fix, would see the SAME
`self.__defer_armed_N` still `True` and replay the identical body a
SECOND time. Every replay guard (Mechanism 1's own, AND Mechanism 2's,
below) now unsets its own flag (`self.__defer_armed_N = False`) right
after replaying - confirmed via a real repro (`defer_does_not_replay_
again_when_generator_later_dropped`: fully drain a generator with an
armed defer, THEN let it go out of scope, refcount-verify the defer's
own side effect happened exactly once, not twice).

**Mechanism 2 (error exits - `or_return()`/checked-arithmetic failure,
both `defer` AND `errdefer`)**: the one piece needing a real, targeted
lowering.py change, matching this hazard's own nature - "did this exit
happen via an error" is genuine lowering-time information (whether
`or_return()`'s Err branch fired), not derivable from the AST alone, and
the existing fallible-generator `_pessimistic_done_prefix` trick (Phase
8) only works for a VALUE (safely overwritten on success), not a SIDE
EFFECT like a defer replay (would wrongly fire on success too). The key
insight making this tractable: `ir.OrReturn.epilogue` is ALREADY Err-
branch-exclusive by construction (same field ordinary, non-generator
errdefer already uses) - hooking into it directly means errdefer needs
NO separate `is_err()` check at all here, unlike an ordinary function's
own shared epilogue (reached by success AND error alike).

Two-file split: type_resolver.py's `_tag_armed_defer_sites` (called from
the exact same call sites `_pessimistic_done_prefix` already runs at -
every block of user code that might fail) tags each such block's own
top-level statements with `generator_armed_defer_sites`: the PREFIX of
defer_sites armed by that point (tracked via a single shared mutable
`armed_count` cell advanced past each arm-assign crossed, in program
order - arming only ever happens at a top-level statement, so a simple
running count suffices). Tagged with an ALREADY-RENAMED copy
(`rendered_defer_sites`, built once via the same renamer used for
everything else in this `$$__next__` build) - lowering.py's own hook has
no access to `_GeneratorNameRenamer`, so unlike Mechanism 1 (which embeds
the raw body and relies on a LATER bulk rename pass to cover it), the tag
itself has to already be self.<field>-qualified.

lowering.py's `_lower_stmt` pushes/pops `self._generator_armed_defer_
sites` around whatever statement it's currently lowering, based on that
tag (inherits the enclosing context when the current node carries none -
mirrors `_arithmetic_mode`'s own stack semantics, so a fallible operation
nested inside an ordinary if/call within a tagged statement still sees
the right armed set). `_consume_checked_result`'s own `OrReturn`-building
branch calls the new `_build_generator_error_defer_replay`: LIFO over
whatever's currently armed, each site lowered via the SAME swap-the-
instruction-buffer technique `_register_defer_block` already established
(AST-If-wrap + `_lower_stmt`, not hand-built IR - the body is already
self.<field>-qualified, so ordinary statement lowering does the right
thing for free), spliced into the SAME `replay` list `cfg.return_()`
already builds for `OrReturn.epilogue`. `self._generator_armed_defer_
sites` is reset to empty while lowering each site's own body - a defer
body is expected to be simple cleanup, not itself something needing its
OWN error-defer replay; without this a fallible op nested inside a defer
body would recurse into this same method against the SAME still-armed
site, unboundedly.

Verified via real compile-and-run (emitter_c_test.py's
GeneratorFunctionTests): plain `defer` at natural exhaustion, a bare-
return exit, and abandonment (each refcount-checked not-fired-early/
fired-exactly-once/not-refired-after-done); the double-replay-after-
later-drop fix specifically; two `defer` sites replayed LIFO; `errdefer`
firing exactly on an `or_return()` error exit and NOT on a separate,
successful full drain (`errdefer_fires_on_error_exit_only`); `defer` AND
`errdefer` both armed, both firing on the SAME error exit
(`defer_and_errdefer_both_fire_on_same_error_exit`); `with defer:`/
`with errdefer:` directly inside a while-unit's own loop body and an
if-unit's own branch both rejected with a clear message, not silently
wrong.

**Follow-up fix, found in a later session while auditing gaps**: a
`return` inside a generator's own `defer`/`errdefer` body was
unvalidated - ordinary (non-generator) functions already reject this
(lowering.py's `_stmt_Return`, gated on `_in_deferred_body`, set only by
`_register_defer_block`), but a generator's own defer body never routes
through that mechanism at all (both Mechanism 1 and Mechanism 2 lower it
via their own separate AST-If-wrap + `_lower_stmt` technique), so
nothing caught it. Fixed via `_reject_return_inside_generator_defer_body`
(type_resolver.py), called right where each site's body is captured
(`_desugar_generator_defer_sites`) - walks the WHOLE captured body
(`_walk_generator_body`, not just the top-level statement) so a return
nested inside an if/while inside the defer body is caught too, reusing
the exact same error message text as the ordinary check for consistency.
Verified via two rejection tests (`test_return_inside_generator_defer_
body_is_rejected`, and a second confirming the nested-inside-an-if case
specifically).

Phase F design (the AST-synthesis-to-real-dispatch rebuild)

Two independent research passes and a planning pass, cross-checked
against the actual code, converged on the same finding: everything above
this point was built on `_collect_generator_units`, which only ever
TEMPLATE-MATCHES four flat, top-level shapes (bare yield, a while loop
with exactly one yield, an if/else with at most one yield per branch,
each occupying a small FIXED state count). This is why every restriction
above (elif, break/continue, nesting, multiple yields) existed - not
because any of them were individually hard, but because the "two states,
one resume-flag" trick each unit builder relied on is fundamentally
single-suspend-point-per-unit, and generalizing it to N yields at
arbitrary depth means re-deriving a state discriminant RECURSIVELY
inside the AST - unbounded complexity for no benefit over what a real
flat integer state already gives for free. `.send()` (Phase C, below)
independently needed the identical foundation: yield recognized in
EXPRESSION position (never supported - the unit matcher only ever saw
`ast.Expr(ast.Yield(...))`, a bare statement) and a second entry point
sharing the same dispatch as `$$__next__`. Both point at the same
missing piece, so this was landed as one foundational rebuild rather
than patching the old mechanism twice.

The rebuild itself: a generator body is now lowered ONCE, through the
ORDINARY `Lowering._lower_stmt`/`_lower_expr` pipeline every non-
generator function already uses for arbitrary `if`/`while`/`for`
nesting, with one new case - an `ast.Yield` reached during lowering
emits `ir.Yield(value, state, resume_label)` (state assigned in AST-walk
order per textual yield site, `FunctionLowering._emit_generator_
dispatch_prologue`) and lowering continues immediately after. This
confirmed the plan doc's own long-standing observation (see "New IR"
below): `ir.Jump`/`ir.Label`/`ir.JumpIfFalse` ALREADY compile to real
flat C `goto`/label pairs, unchanged since v1 - the hardest
infrastructure piece this rebuild needed was already there and already
proven, just never wired up to a real dispatch table. `ir.Yield` itself
turned out NOT to need to own "state store + return + resume label" as
a single opaque unit the way the plan's own original IR sketch (below)
implied - splitting it into three ordinary, already-existing
instructions (`ir.SetAttr` for the state store, `ir.Yield` for a plain
`return value;`, `ir.Label` for the resume point) reuses two already-
correct codegen paths verbatim and needed zero new emitter_c.py logic
beyond `ir.Yield`'s own trivial `return` case - `Function.is_generator_
next` ended up mirroring `is_destructor`'s own precedent exactly as
originally sketched, just gating a lowering-level dispatch-prologue
builder instead of an emitter-level signature/prologue special-case.

What survived completely unmodified, confirmed dispatch-shape-agnostic
rather than just assumed: field promotion (`_collect_generator_locals`
already walked the whole body, any depth); the RC live-flag FIELD
mechanics (`_live_flag_stem`, the `__<stem>_live` bool, the zero-
placeholder exemption); the destructor's overall shape; both defer/
errdefer mechanisms (Mechanism 1's own AST-synthesized arm/replay flags
don't care about nesting depth; Mechanism 2 was ALREADY lowering-level,
and needed no redesign - see "Phase F: defer/errdefer under real nested
lowering" below for the one thing that DID need re-verifying there).

What got deleted: `_collect_generator_units`, `_split_generator_
segments`, `_build_yield_unit_guard`/`_build_while_unit_guard`/
`_build_if_unit_guard`, `_validate_while_yield_unit`/`_validate_if_
yield_unit`, `_pessimistic_done_prefix`, `_wrap_generator_next_returns_
in_ok`'s old per-unit callers (the method itself survives, see below).
`type_resolver.py`'s own remaining job shrank to: rename locals/
parameters to `self.<field>` and split RC-typed field writes for the
live-flag dance over the WHOLE (unsplit) body in one pass instead of per
flat unit fragment (`_rename_and_track_liveness`, generalized to recurse
into nested if/while/for/with bodies via a new `_recurse_liveness_wrap`
- same shape reused twice more later, see below); tag Mechanism 2's
armed-defer-site prefix over the whole body in one call (confirmed via
`_tag_armed_defer_sites`'s own pre-existing docstring that tagging only
the TOP-LEVEL statement in a slice was already sufficient - lowering.py's
own push/pop keeps a tag active for that whole statement's recursive
lowering, so this generalized with ZERO changes to that method itself);
assemble the DONE-check + tail-exhaustion wrapper around the otherwise-
untouched user body (`_build_generator_next_function`, rewritten).

Fallibility's own pessimistic-done trick (Phase 8's `self.__state = done`
inserted BEFORE any block that might fail) is re-derived at the lowering
level instead of the AST level: `_consume_checked_result` (lowering.py,
the single method shared by both `.or_return()` and checked-arithmetic
under Check mode) now appends a `SetAttr(self.__state, done_state)` into
every `OrReturn`'s own `epilogue` list whenever `self._current_fn.is_
generator_next` and the generator is fallible (`_generator_pessimistic_
done_replay`), reusing the EXACT hook Mechanism 2's own error-defer
replay (`_build_generator_error_defer_replay`) already established at
that same call site, for the identical reason - both need "run this
extra thing on the Err branch, before the return." This naturally
reaches every fallible operation anywhere in the body, any nesting
depth, where the old AST pass could only see "immediately before a
unit's own yield/fall-through."

Two real, non-obvious bugs surfaced via real compile-and-run testing
while landing this (not caught by reasoning alone):

1. `resolve_function_body`'s own `for stmt in fn.node.body:` loop
   captures a generator's ORIGINAL body once, up front - if resolving
   one of the body's OWN later statements needs the generator's own
   return type (confirmed via a real repro: `.or_return()` checks the
   ENCLOSING function's declared error type for propagation
   compatibility), that resolution can reentrantly trigger `ensure_
   generator_synthesized` MID-LOOP, which rewrites `fn.node.body` out
   from under the still-running iteration - the loop's own stale
   reference keeps resolving the OLD statements regardless, and this
   method's own unconditional `fn.node.body = new_body` at the end then
   CLOBBERS the freshly-synthesized constructor-call body with a
   resolved copy of the stale original one. Fixed by synthesizing
   eagerly, at the very top of `resolve_function_body`, before its own
   per-statement loop ever starts - a cheap, idempotent no-op for every
   non-generator function (the same `_function_contains_yield` check
   `Lowering.lower_function`'s own identical safety-net call already
   pays for every function).

2. `_lower_generator_yield`'s own `_flush_pending_temps()` call was
   decref-ing the YIELDED VALUE ITSELF, because it was never `untrack_
   temp`'d first, unlike `_stmt_Return`'s own identical case (see its
   own comment on why this matters) - confirmed via a real repro: a
   `Box` yielded through a bare while-unit came back already released
   (tag intact, payload pointer pointing at freed memory). Fixed by
   adding the missing `untrack_temp` call, mirroring `_stmt_Return`
   exactly.

A yield reached OUTSIDE a successfully-synthesized generator (e.g. one
whose own synthesis failed earlier for an unrelated, already-reported
reason - see bug 1 above for how that state becomes reachable even with
the fix, via a genuinely invalid generator like `return` inside a defer
body) now fails gracefully via `discovery.fail()` instead of a raw
`AssertionError` escaping the compiler's own per-unit recovery
boundaries - a single bad generator no longer risks crashing the whole
compile run.

Phase F's own regression gate is the ENTIRE pre-existing
`GeneratorFunctionTests` suite passing unmodified (behavioral tests, not
IR-shape assertions - exactly what an internal-mechanism swap needs to
prove) - it does, except five rejection tests whose own shapes are now
legitimately supported (multiple yields per loop/branch, yield nested
inside if-in-while, elif chains, break inside a yield-containing while,
two yields in one if branch), each replaced with a positive compile-and-
run test of the same shape.

Phase B design (nesting/multiplicity verification)

Once Phase F landed, this was almost entirely VERIFICATION, not new
mechanism - real compile-and-run coverage for every shape the unit model
used to reject, confirming the dispatch prologue genuinely doesn't care
about nesting shape/depth/multiplicity: `while` nested inside a yield-
containing `if` branch AND the reverse (`if` nested inside `while`),
yield nested THREE levels deep (if-in-if-in-while, resuming the correct
arm), `elif` chains with a yield in each arm, `continue` skipping a
loop's own yield mid-iteration (not just `break`, already covered by
Phase F's own regression-gate replacement tests), and an RC refcount-
delta variant (a captured parameter still torn down correctly when a
generator is dropped mid-iteration with its suspended state nested
inside if/while, not just at the old flat single-yield-in-while depth).

Phase C design (`.send()`)

Landed the plan's own final, simplified design after live back-and-forth
during scoping (see the plan file this session started from for the
full reasoning trail - not reproduced here): `Generator[T, SendType, E]`
- three type parameters. `T`/`E` are the existing `elem_type`/
`error_type`, completely unchanged; `SendType` is new. `(yield expr)`
used as an EXPRESSION evaluates to plain `SendType` - no automatic
Result-wrapping by the compiler. A generator author who wants `.send()`
to be able to inject a failure the body can react to simply declares
`SendType` as `Result[V,Err]` themselves and uses the ALREADY-EXISTING,
already-tested `.or_return()`/`.unwrap_or()`/`match` machinery on it
like any other Result value - confirmed these already work generically
on any properly-typed Result-shaped expression reaching lowering.py, no
special-casing tied to the receiver's origin. `SendType` and `E` are
deliberately independent: a generator can `.unwrap_or()` an injected
error into a default without `E`/`or_return()` ever being involved.
`Iterator[T]` (1 arg) and the existing 2-arg `Generator[T,E]` are
completely unaffected - only the new 3-arg form gets `.send()` at all,
dispatched on tuple arity in discovery.py's own `Generator[...]`
recognition.

Backing-class shape: `__send_slot: SendType` (an ordinary promoted
field - participates in the EXISTING RC live-flag/zero-placeholder/
destructor-teardown machinery whenever `SendType` is RC-typed, no new RC
design needed) and `__send_ready: bool` (armed by `send()`, consumed and
cleared by the next captured-yield resume). `_build_generator_next_
function`'s own body-assembly becomes `$$__resume__` instead of
`$$__next__` whenever `send_type` is set (Iterator[T]/the 2-arg form are
unaffected - `$$__next__` stays the one real method, exactly as every
phase before this one built it); two thin public wrappers delegate into
it - `__next__()` (leaves `__send_ready` untouched) and `send(v)`
(panics via `sys.panic()` if `self.__state == 0`, mirroring Python's own
`TypeError` for sending before the first yield, otherwise arms `__send_
slot`/`__send_ready` - live-flag-guarded via `_build_liveness_guard`
directly, when `SendType` is RC-typed - then resumes).

Yield-as-expression itself is a new `_expr_Yield` case registered the
same way every other `_expr_X` handler is (`_lower_expr`'s own getattr-
based dispatch, no special-casing needed) - `_lower_generator_yield`'s
own suspend-building logic (state store, `ir.Yield`, resume `ir.Label`)
is factored into a shared `_emit_generator_yield_suspend`, used by both
the discarded (statement-position, unchanged since Phase F) and captured
(expression-position, new) cases. After resuming, `_expr_Yield` reads
back `self.__send_ready`: true means `send(v)` armed `__send_slot` since
this suspend - clears the flag and returns an incref'd read of it (a
genuine aliasing read, same category as an ordinary field read); false
means this resume came from a bare `__next__()`/for-loop consumption
instead - panics with a clear message pointing at `.send()`. Because
this is ordinary expression lowering, it composes for FREE with
anything wrapping the yield (`x = yield v`, `x = (yield v).or_return()`,
...) - no special-casing needed for nesting, the exact same reason Phase
F's own dispatch mechanism generalized nesting/multiplicity for free.
`_validate_generator_yield_positions` allows yield anywhere inside a
generator that declared a `SendType` (the old bare-statement-only
restriction stays exactly as before for `Iterator[T]`/the 2-arg form,
which have no `SendType` to ever deliver a captured yield's own value
through).

Two more real, non-obvious bugs surfaced via real compile-and-run
testing of an RC-typed `SendType` specifically (a scalar `SendType`
never exercises either path):

1. `_build_liveness_guard`'s own deep-copy strategy (the "if self.__
   stem_live: ... else: ..." guard around an RC-typed promoted local's
   assignment, unmodified since Phase 9) silently DUPLICATES a yield
   when the value expression contains one (`held = yield i`) - each
   copy becomes its own independent `(state, resume_label)` suspend
   point for what must be ONE textual yield site, corrupting the whole
   dispatch state count. An ordinary value expression is safe to
   duplicate this way (only one of the two branches ever actually
   RUNS for a given dynamic execution, so a Call/constructor appearing
   twice in the compiled C still only executes once) - a `yield`
   specifically breaks that assumption, since it's a real suspend point,
   not a value computation. Fixed by detecting a yield in the value
   expression and restructuring into "capture the value once into an
   ordinary local, branch only on the simple re-store" instead of deep-
   copying the whole statement - the yield now appears exactly once,
   textually and state-wise, regardless of which branch the live-flag
   selects at runtime. (`_build_liveness_guard` now returns `list[ast.
   stmt]` instead of a single `ast.If` for this reason - one of its
   three call sites, inside the `send()` wrapper builder above, had
   been silently producing a corrupt nested-list AST body for an RC-
   typed `SendType` before this was caught.)

2. The new capture-temp local was then getting DOUBLE-incremented:
   `_is_aliasing_expr` didn't recognize a captured yield (reading
   `self.__send_slot`, a field that independently owns its own
   reference) as aliasing, so the capture assignment was treated as
   "fresh" and given its own tracked ownership that never gets balanced
   (a yield's own suspend deliberately skips the ordinary epilogue
   unwind - locals persist across it, nothing to unwind), while the
   SUBSEQUENT re-store from the capture temp added its own separate
   increment on top. Fixed by teaching `_is_aliasing_expr` that a
   captured `ast.Yield` is aliasing (same category `ast.Attribute`
   already is - it reads an existing field), and tagging the capture
   assignment the same way `visit_Match`'s own `__match_subj_N` relay
   already is (`is_alias` -> borrow, no independent tracked ownership) -
   mirrors that existing, proven mechanism exactly rather than inventing
   a new one.

Verified via real compile-and-run refcount-delta checks: an RC-typed
`Box` sent through `.send()` into a captured `held = yield i` correctly
ends up with THREE independent owners (the caller's own binding, `__
send_slot`, and `held`), and a second `.send()` call correctly drops the
first value back to one owner and brings the new one up to three; a
scalar accumulator round-trips real values through repeated `.send()`
calls; `.send()` before the first yield panics; resuming a captured
yield via a bare `.__next__()` instead of `.send()` panics too.

A.4a follow-up design (`yield from` / for-loop-with-yield at non-top-
level positions)

`_desugar_generator_yield_from` and `_desugar_generator_for_loops` both
generalized from top-level-only to recursing into nested if/while/for/
with bodies (`_recurse_desugar_yield_from`/`_recurse_desugar_for_loops`,
the SAME shape as Phase F's own `_recurse_liveness_wrap`, now used a
third time) - a `yield from` (or an ordinary for-loop containing yield)
reachable through `if`/`with` now forwards/desugars correctly at any
nesting depth, not just the top level.

A real, deeper bug surfaced along the way, not just the narrow "nested
positions are unreachable" gap this started as: `_new_for_obj_field`'s
own "evaluate the iterated expression once, at construction" design
(Phase 1, still unchanged) is silently WRONG once the for-loop is
reachable through a while/for loop that can re-enter it - the same
already-exhausted iterated object gets reused on every re-entry instead
of being freshly reconstructed, confirmed via a real repro (`while j <
count: yield from inner(); j += 1` only ever forwarded `inner()`'s own
values during the outer loop's FIRST pass - every later pass silently
forwarded nothing at all). Rather than the larger fix (re-deriving `__
for_obj_N` to re-initialize per loop entry, not just once ever - real
work, no forcing use case yet), landed a new explicit pre-desugar
validator (`_reject_generator_for_or_yield_from_nested_inside_loop`,
run against the ORIGINAL undesugared body, before either desugar pass)
that rejects this specific shape with a clear message: nested inside
`if`/`with` is fine and works; nested inside `while`/`for` (at ANY
depth reachable through one - tracked via a simple `in_loop` flag
propagated through the walk, set once entering any loop and never
cleared by an intervening `if`/`with`) stays a compile error instead of
a silent miscompile.

Phase F: defer/errdefer under real nested lowering

Mechanism 2 (the lowering-level lowering.py hook, `_generator_armed_
defer_sites` push/pop in `_lower_stmt` plus `_build_generator_error_
defer_replay`) needed re-verification under Phase F's real nested
lowering, not a redesign - and didn't need one: `_tag_armed_defer_
sites`'s own pre-existing docstring already established that tagging
only the TOP-LEVEL statement in a slice is sufficient (lowering.py's own
push/pop keeps a tag active for that whole statement's own recursive
lowering, so a fallible operation nested arbitrarily deep inside a
tagged statement still sees the right armed set) - calling it once over
the WHOLE (renamed) body, instead of once per flat unit-fragment
preamble as before, required zero changes to that method itself. The
existing `GeneratorFunctionTests` defer/errdefer suite (armed-then-
fired-exactly-once, LIFO ordering, `defer`/`errdefer` both firing on the
same error exit, the preamble/tail-only positional restriction) passed
unmodified through the whole rebuild, confirmed by Phase F's own
regression gate.

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

(This list is part of the ORIGINAL planning notes above - a snapshot of
v1's own starting scope, kept verbatim for historical context. By the
time this doc reached Phase C, every item below except `.throw()` and
`async`/`await` had landed - see the STATUS section at the top of this
doc for the current, up-to-date picture.)

- `Generator[T,E]` fallible generators (sketched above, not designed).
- `yield from`, `.send()`, `.throw()`, `.close()` beyond ordinary RC drop.
- `defer`/`errdefer` inside a generator body.
- Generic generator functions / generators inside a generic method.
- `async`/`await` - unrelated feature, no forcing use case, not the same
  mechanism (this plan is purely synchronous re-entry via explicit
  `__next__()` calls, no scheduler).
