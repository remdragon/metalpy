# Scope: hunt down other instances of 3 bug shapes found while building lib/re.py

## Context

While building `lib/re.py` (branch `worktree-re-module-phase1`), three compiler bugs
were found, root-caused, fixed, and merged into `master` (commit `ee33d86`, merge
`8200e29`):

1. `lowering.py`'s `_expr_Attribute` tagged every CEnum member reference with its
   underlying scalar type, ignoring the `expected_type` hint the caller passed in -
   broke generic type-param inference for `Result.Err(SomeEnum.Variant)`.
2. `lowering.py`'s `_coerce_into_union` matched a union's leaves via `is` identity
   instead of `_same_type` - rejected a bare `list[Op]` return against a declared
   `list[Op]|None` return type, since the two were different-but-equal
   `Specialization` objects.
3. `type_resolver.py`'s `_try_resolve_namespace` (used to resolve a generic
   subscript's own type *arguments*, e.g. the `T` inside `list[T]`) had no case for
   `X|Y` union syntax, even though `discovery.py`'s ordinary annotation resolver
   already did - `list[str|None]()` was rejected as "argument is not a type".
4. `lowering.py`'s `_stmt_If` decided whether a branch "terminates" (and so its
   narrowing survives past the if-statement) via a purely syntactic
   `isinstance(last_stmt, (Return, Break, Continue))` check, blind to a trailing
   call to a `-> NoReturn` function like `sys.panic(...)` - silently dropped
   narrowing instead of erroring, then produced broken C at emission.

None of these were one-off mistakes - each is one instance of a recognizable
*shape* of bug in the type-checker/lowering machinery. This document scopes a
hunt for other instances of those same four shapes, compiled from a systematic
audit (three parallel read-only search passes, one per shape family below).

**This document is scope only - it does not fix anything.** Each candidate below
still needs: re-locating in a fresh worktree (this doc's line numbers will drift -
`master` here is unusually active, see the note below), a minimal repro to confirm
it's real, and a judgment call on fix risk before anyone touches it.

**A live example of why "re-locate, don't trust the line number" matters:** one
of the candidates identified by this audit's own search pass (Shape 2's
`tuple`/`Iterator`/`Generator`/`move`/`copy` gap, below) turned out to already be
fixed by a *different, concurrent* session by the time this document's candidates
were re-verified against a fresh worktree - a matter of hours after the search
that found it. Treat every "open" item below as "open as of the fresh worktree
this document was written from" (`master` commit `7f838c7` plus this worktree's
own base), not as a permanent fact - re-verify before spending real effort on any
of them.

**Out of scope:** `lib/re.py`'s existing workarounds for bugs 1-4 above. The user
has separately notified that session to remove them now that the underlying
compiler bugs are fixed; this document and its follow-up work do not touch
`lib/re.py`.

## Shape 1 - raw `is` identity used where `_same_type` is needed

Two different Python objects can legitimately represent the *same* mpy type
without being identity-equal - two `Specialization`s for the same instantiation
reached via different resolution paths, or a bare `TupleType` vs. its own
`.backing` RCClass. `TypeResolver._same_type` (`type_resolver.py:2577`) exists
specifically to treat these as equal, and several call sites use it correctly
(`_check_assignable`, `_unify_type_param`). The fixed bug (`_coerce_into_union`,
`lowering.py`) didn't.

**Fixed** (all three, this pass):

- `type_resolver.py`'s union-receiver-dispatch leaf-agreement check (was
  `if fn.return_type is not reference.return_type:`) - confirmed with a real
  repro (two leaves, a generic `Box[T].get_list() -> list[T]` monomorphized
  to `list[i32]` and a concrete `Other.get_list() -> list[i32]` resolved
  fresh, both textually identical, wrongly reported as "disagree"). Fixed via
  `_same_type`; genuine mismatches (verified with a negative-test repro)
  still correctly rejected. Regression tests:
  `UnionReceiverDispatchCoercionTests.test_generic_leaves_with_equal_return_types_do_not_false_positive`
  / `.test_genuinely_disagreeing_leaf_return_types_still_rejected`
  (emitter_c_test.py).
- `lowering.py`'s `_lower_dispatch_tests` and `_maybe_unwrap_union_arg` (both
  `attr.type is leaf_type`/`attr.type is target_type`) - fixed via
  `_same_type` for consistency with the rest of the codebase, but **no
  positive repro could be constructed for either**: both are only ever
  reached through the overload-dispatch mechanism
  (`_lower_conditional_dispatch`), which is gated by `overload_resolution.py`'s
  own SEPARATE identity-based leaf matching (`_leaf_is_accepted`, the
  "awareness only" item below) - a call site that would trigger THIS
  duality gets rejected by THAT earlier check first, before ever reaching
  these two lines. The fix is a strict superset of the old behavior (only
  accepts more correct programs, can never wrongly reject one already
  accepted), so applied anyway for consistency; full suite green regardless.
  No dedicated regression test added for these two specifically, since none
  could be constructed - would need the `overload_resolution.py` item fixed
  first to ever exercise them with a legitimately-mismatched-but-equal leaf.

**Fixed** (both, this pass):

- `lowering.py`'s `_stmt_Return` - the hand-rolled reimplementation of
  `_check_assignable`'s CEnum duality logic turned out to be a real,
  confirmed bug, not just a maintainability smell: it only ever re-derived
  ONE of `_check_assignable`'s two CEnum<->value_type exemption directions
  (`is_cenum_to_underlying` - a CEnum value returned where the function
  declares the underlying scalar). The REVERSE direction (a raw scalar
  returned where the function declares the CEnum) was missing entirely -
  confirmed via a real repro (`return x` where `x: i32` inside a function
  declared `-> Color`, wrongly rejected as "function returns Color, not
  i32"). Fixed by adding the missing `is_underlying_to_cenum` check;
  genuine mismatches (a real repro with an unrelated `str` return) still
  correctly rejected. Regression tests: new
  `CEnumReturnCoercionTests.test_programs_compile_and_run` /
  `.test_genuinely_mismatched_return_type_still_rejected`
  (emitter_c_test.py). Not delegated to `_check_assignable` directly -
  that method's own `strict=False` is deliberate, to let the
  covered-Result-error-widening case get a chance before a stricter check
  would reject it outright (see the method's own comment) - so the fix
  stays as a hand-derived exemption, matching the existing pattern, rather
  than folding in the shared helper.

  **Incidentally found, unrelated, NOT fixed (out of scope, flagged for a
  future pass):** `lowering.py`'s `_expr_Constant`/`emitter_c.py`'s
  `_emit_const` crash with an uncaught Python `NotImplementedError` (not a
  clean `CompileError`) for a kind-mismatched literal returned where a
  CEnum is expected (e.g. `return 'not a color'` from a function declared
  `-> Color`) - the literal gets mistagged with the CEnum's own type by
  `_expr_Constant` somewhere upstream of `_stmt_Return`'s own check (which
  never gets a chance to reject it, since `value.type is fn_type` already
  holds by the time it runs), then crashes at C-emission with a raw
  Python traceback instead of a clean compile error. Confirmed pre-existing
  on `master`, unrelated to this fix (reproduces identically with this
  fix reverted).
- `lowering.py`'s `_expr_Name` escape hatch (`expected_type is not
  name.type`) - fixed via `_same_type` for consistency, but **no repro
  could be constructed** despite several attempts (generic-substituted vs.
  fresh-annotation parameter types; local-variable-annotation vs.
  fresh-annotation parameter types - both patterns that DID trigger other
  Shape 1 candidates). Current best guess, not fully confirmed: unlike a
  bare member-level Specialization, the WHOLE union types being compared
  here (`expected_type`/`name.type`) are both `_get_or_create_union`
  results, which cache by a qualname-TEXT key (see `ARCHITECTURE.md`) -
  insensitive to whether the union's own member Specializations are
  identical objects, so two structurally-identical union annotations seem
  to always land on the same cached object regardless of which resolution
  path produced them. This is a DIFFERENT reason for "unconfirmed" than
  the two `lowering.py` dispatch candidates above (those are gated by a
  separate, known upstream bug) - this one may simply not be reachable at
  all. No dedicated regression test added, for the same reason as those
  two.

**Awareness only - deliberately identity-based by design, per their own
comments. Do not touch without separately confirming the design intent still
holds:**

- [cfg.py:1251](cfg.py:1251) (`_tag_gated_refcount_instructions`) - `members =
  [m for m in t.attributes if any(m.type is leaf for leaf in leaves)]`.
- [overload_resolution.py:41-48](overload_resolution.py:41) (`_contains`/
  `_intersect`/`_subtract`) - relies on the existing dedup/interning caches
  (`_get_or_create_union`/`_specialization`/`_move`) guaranteeing "same type ==
  same object" for the specific universe these functions operate over.

## Shape 2 - parallel type-resolution paths that drifted out of sync

`discovery.py`'s `Discovery.visit_Subscript` (~line 742, the full
annotation-position resolver: `list[T]`, `tuple[...]`, `Callable[[...],T]`,
`move[T]`/`copy[T]`, generic class/function specializations) and
`type_resolver.py`'s `TypeResolver._try_resolve_namespace` (~line 2671, a
narrower resolver used specifically for a generic subscript's own type
*arguments*, e.g. resolving `T` inside `list[T]`) are supposed to accept the
same grammar. The fixed bug was a gap in the latter (`X|Y` union syntax
missing).

**Status: the broader instance of this gap is already fixed on current
`master`, found and confirmed during this document's own verification pass.**
`_try_resolve_namespace` (`type_resolver.py:2723-2742`) now has an explicit
branch for `node.value.id in ('move', 'copy', 'Callable', 'Closure', 'tuple',
'Iterator', 'Generator')` that delegates straight to
`self.discovery.visit_Subscript(node)` - the exact fix this audit would have
recommended, already landed by a different concurrent session. Verified via a
direct repro (`list[tuple[i32,i32]]()` compiles and runs correctly on this
worktree's `master`-derived base). **No action needed here** - listed for
context/history only, so a future re-run of this sweep doesn't waste time
re-discovering it.

**Awareness only - documented, narrow-by-design, not live bugs, but worth a
footnote since they're more parallel implementations of the same kind of
namespace-walk:**

- [type_resolver.py:3373](type_resolver.py:3373) - `_ReferenceResolver`'s own
  `_try_resolve_namespace` (class starts at `type_resolver.py:3100`) - a
  *third* copy of this namespace-walking logic, for match-pattern `Owner`
  resolution. Deliberately Name/Attribute-only per its own comment ("no
  Subscript here... rewrites never need it"). Not broken today, but a third
  parallel implementation of "walk a namespace path AST" is a standing risk for
  future drift if match-pattern resolution ever needs to handle a generic
  `Owner` shape.
- `_ReferenceResolver._try_resolve_callable_namespace` /
  `_try_resolve_generic_call` (same class) - a fourth/fifth variant, explicitly
  documented as best-effort/non-authoritative with a graceful `None` fallback to
  lowering.py's real resolution. Low severity by design.

## Shape 3 - syntactic-only terminator/divergence checks miss `-> NoReturn` calls

The fixed bug: `lowering.py`'s `_stmt_If` decided whether a branch "terminates"
via `isinstance(last_stmt, (ast.Return, ast.Break, ast.Continue))`, blind to a
trailing call to a `-> NoReturn` function (`sys.panic(...)`). Fixed via a new
`_stmt_diverges` helper that also resolves the last statement's callee (via
`_resolve_callee_target`) and checks for a `NoReturn`-typed return.

**Fixed:**

- `type_resolver.py`'s `visit_Match` (`terminates` computation) - added a
  type_resolver.py-level `_stmt_diverges` mirroring `lowering.py`'s own,
  wired into the per-case `terminates` flag. Confirmed with a real repro:
  the observable effect is narrower than `_stmt_If`'s bug turned out to be -
  ordinary post-match expressions are protected by `lowering.py`'s own
  independent, already-correct narrowing over the desugared if-chain
  regardless; the actual break is in `type_resolver.py`'s own
  `_rewrite_type_is_comparison` fold-to-constant optimization for a LATER
  `type(x) is T` check, which assumed `x` was still union-typed and emitted
  an invalid `.tag` access once `lowering.py` had already narrowed it out
  from under that assumption (`intrinsics.usize has no attribute 'tag'`).

  **Caught a second, more serious bug building this first one, already
  landed on `master` before it was caught:** calling `_resolve_callee_target`
  unconditionally on a case arm's last statement crashes the compiler for an
  ordinary receiver call (`self.foo()`) OR a match-pattern-bound receiver
  (`case Result.Ok(w): ... w.close()`) - `discovery.find_name` raises rather
  than returning `None` for a name rooted in a local, and (this cost real
  time to discover) catching the exception isn't enough, since
  `discovery.fail()` permanently records the error message before raising.
  The FIRST fix attempt (a `self.locals` membership pre-check) caught the
  `self.foo()` case but missed the match-bound-name case, since a match
  pattern's own binding is spliced into the output as a bare `ast.Assign`
  that's never routed through `self.visit()`/`visit_Assign`, so it never
  updates `self.locals` - this real regression escaped review and landed on
  `master`, then surfaced as a genuine break in 3 real CSV-module tests
  (`csv_dict_test.py`/`csv_linereader_test.py`/`csv_reader_test.py`, via
  `lib/builtins/__File.py`'s `File.binary_writer`) once a concurrent
  session's new tests happened to exercise the exact shape. Fixed by
  checking `discovery.find_name_or_none` directly instead of a hand-tracked
  "known locals" set. Regression tests:
  `NarrowingSurvivalTests.test_programs_compile_and_run`'s
  `match_arm_sys_panic_narrows_past_the_match` /
  `match_arm_receiver_call_does_not_crash_the_compiler` /
  `match_bound_name_receiver_call_does_not_crash_the_compiler`
  (emitter_c_test.py) - each independently confirmed to fail without its
  fix and pass with it.
- [lowering.py:683](lowering.py:683) (`_body_may_fall_off_the_end`) - `return
  not body or not isinstance(body[-1], ast.Return)`, used to decide whether to
  synthesize an implicit `return None` at a function's close. Lower priority:
  per its own comment (672-682), a false positive here is explicitly argued to
  be harmless (produces dead-but-unreachable code after a real diverging call,
  not a compile break). Its comment ("same 'future work' scope cut as
  `_stmt_If`'s own true_terminates/false_terminates detection") is now
  slightly stale, since `_stmt_If` no longer has that exact scope cut - worth a
  comment update even if the behavior itself is judged low-risk enough to leave
  alone.

**Checked, ruled out - no comparable gap:** `cfg.py`'s `merge_loop_exits`,
`type_resolver.py`'s `visit_While`/`visit_For`, and `lowering.py`'s
loop-lowering methods (`_stmt_While`, `_stmt_For`, `_lower_for_range`,
`_lower_for_over_indexable`, `_lower_for_over_iterator`) - none of them make a
last-statement-shape decision the way `_stmt_If`/`visit_Match` do; loop-exit
narrowing is driven by real `break` sites (`cfg.py`'s `record_break_narrowed`),
not by inspecting the body's trailing statement.

## Shape 4 - `expected_type` accepted but silently ignored in expression lowering

The fixed bug: `lowering.py`'s `_expr_Attribute` took an `expected_type`
parameter but its CEnum-member branch ignored it outright, always defaulting to
the enum's underlying scalar type - a case where the *same value* has two
legitimately different type representations (nominal enum vs. underlying
scalar), and only surrounding context can say which one is wanted.

All 15 `_expr_*` dispatch methods in `lowering.py` were checked. Two candidates
in `_expr_Subscript` were flagged and then downgraded on closer inspection:

**Checked, likely not a bug (downgraded from the original candidate list):**

- [lowering.py:5595](lowering.py:5595) (`_expr_Subscript`) - both the
  tuple-index branch (`dest = self._new_temp(attr_var.type)`, ~5627) and the
  real-`__getitem__` branch (`call_dest = self._new_temp(getitem_fn.return_type)`,
  ~5679) use the accessed field/callee's own declared type rather than
  `expected_type`. Unlike the CEnum case, neither of these has a genuine
  *dual representation* to disambiguate - a tuple field's type and a resolved
  `__getitem__`'s return type are each singular and already concrete by the time
  they're read here, the same "authoritative type from the callee, caller-side
  `_check_assignable` coerces afterward" pattern `_expr_JoinedStr`/
  `_expr_NamedExpr` use deliberately elsewhere. Recorded here so a future sweep
  doesn't re-spend time on it, not carried forward as an open candidate.

No other `_expr_*` method showed this pattern - every other one either
genuinely threads `expected_type` through, or has a documented reason not to
(`_expr_BoolOp`/`_expr_Compare`'s flat-comparison paths always produce `bool`,
which needs no hint).

## Priority order for follow-up work

1. ~~`type_resolver.py:4593` (Shape 3, `visit_Match`)~~ - **fixed**, see above.
2. ~~Shape 1's three high-confidence candidates~~ - **fixed**, see above.
3. ~~Shape 1's two medium-confidence candidates~~ - **fixed**, see above. The
   `_stmt_Return` one turned out to be a real, confirmed bug (not just
   duplicated logic); a new, unrelated, pre-existing crash bug
   (`_expr_Constant`/`_emit_const`, kind-mismatched CEnum-return literals)
   was found incidentally and flagged, not fixed.
4. **`lowering.py:683` (Shape 3, `_body_may_fall_off_the_end`)** - low risk, low
   priority; at minimum update its stale comment. Next up.
5. **New, incidentally-found candidate:** `lowering.py`'s `_expr_Constant`
   (feeding `emitter_c.py`'s `_emit_const`) crashes with an uncaught Python
   `NotImplementedError` instead of a clean `CompileError` for a
   kind-mismatched literal (e.g. a string) returned/assigned where a CEnum
   is expected - confirmed via a real repro, pre-existing on `master`,
   unrelated to any fix in this document. Not investigated further (found
   while verifying the `_stmt_Return` fix, out of scope for that task) -
   worth its own root-cause pass.
6. Everything under "awareness only" - do not fix without first confirming with
   the user that the documented deliberate-design reasoning no longer holds.
   Note: `overload_resolution.py`'s `_leaf_is_accepted`/`_contains` (its own
   identity-based design, documented as relying on the dedup caches) is now
   the more load-bearing of the two "awareness only" items - it's the reason
   two of the `lowering.py` Shape 1 candidates couldn't get a positive repro;
   worth reconsidering whether it should move up in priority.

## Verification plan for any fix made from this list

Mirrors the process used for the original three bugs:

1. Work in a fresh `EnterWorktree` worktree (never reuse this scoping worktree
   or any other named one - see this repo's `CLAUDE.md`).
2. Before fixing: write a minimal repro, confirm it fails today against current
   `master` with the reported symptom.
3. Fix, then re-run the same repro to confirm it now compiles/runs correctly -
   for anything touching narrowing survival (Shape 3 candidates), also verify
   the "still diverges when actually reached" case the way the original
   `sys.panic()` fix was checked (a variant of the repro where the panic path
   *is* taken, confirming it still fires rather than becoming unreachable).
4. Run the full suite (`python tests.py`) after *each* individual fix, not
   batched - this repo's test suite runs in ~10s across 16 shards, so there's no
   reason to batch and lose the ability to attribute a regression to a specific
   change.
5. Commit, then merge into `master` via the shared-checkout exception in
   `CLAUDE.md`.
