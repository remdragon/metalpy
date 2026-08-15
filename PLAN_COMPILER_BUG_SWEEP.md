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

**High confidence - same shape, not yet fixed:**

- [type_resolver.py:2834](type_resolver.py:2834) - union-receiver-dispatch
  leaf-agreement check: `if fn.return_type is not reference.return_type:` inside
  the loop building `per_leaf` (~2810-2843). Compares two *different* leaf
  classes' own independently-resolved method return-type annotations by identity
  before declaring them "disagree" and failing the compile. Two leaves whose
  `-> list[Op]`-shaped (or any generic/tuple-shaped) return annotations are
  structurally identical but resolved via different `Specialization` objects
  would trigger a false-positive "leaf implementations disagree on return type"
  error. Repro sketch: a union with two leaf classes, both declaring a method
  `-> list[SomeClass]`, called through the union receiver.
- [lowering.py:8594](lowering.py:8594) (`_lower_dispatch_tests`) - `member =
  next((attr for attr in members if attr.type is leaf_type), None)`. Same shape
  as the fixed `_coerce_into_union` bug: `leaf_type` comes from
  `overload_resolution.py`'s `ConditionalDispatch.conditions` (derived from
  `Type.leaves()` on call-site argument types), a different resolution path than
  `members` (derived from `_tagged_union_shape`/`monomorphize_class`). If the
  union is a generic instantiation, a genuine-but-non-identical match would
  wrongly report `"{leaf_type} is not a member of {operand.type}"`.
- [lowering.py:8627](lowering.py:8627) (`_maybe_unwrap_union_arg`) - `member =
  next((attr for attr in members if attr.type is target_type), None)`. Same
  shape, different member-lookup site (`target_type` from `Parameter.type`). A
  non-identical-but-equal match here doesn't error - it silently falls through
  to `return operand` unwrapped (line ~8629), which is arguably worse: a wrong
  answer instead of a compile failure.

**Medium confidence:**

- [lowering.py:2061-2062](lowering.py:2061) (`_stmt_Return`) - hand-rolled
  reimplementation of `_check_assignable`'s CEnum/Specialization/TupleType
  duality logic inline (`is_cenum_to_underlying`, then `value.type is not fn_type
  and value.type is not expected_concrete and not is_cenum_to_underlying`)
  instead of delegating to `_check_assignable`/`_same_type` directly. Not
  confirmed broken - the logic looks carefully reasoned - but hand-duplicated
  logic is exactly how the `_coerce_into_union` gap happened in the first place.
  Worth checking whether this can just call the shared helper instead of
  re-deriving it.
- [lowering.py:4281](lowering.py:4281) (`_expr_Name`) - `if member is not None and
  expected_type is not name.type:` - the "caller wants the whole union back, not
  the narrowed payload" escape hatch. Its own comment (4298-4302) already flags
  "Same Specialization gap as `_stmt_Assign`'s own narrowing-bind handling
  above" as a known concern. Could misfire (wrongly unwrap when the whole union
  was wanted) if `expected_type` and `name.type` are two non-identical objects
  for the same generic instantiation.

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

**Confirmed, not yet fixed - same pattern, same fix shape available:**

- [type_resolver.py:4593](type_resolver.py:4593) (`visit_Match`, inside the
  per-case narrowing-merge logic starting ~4585) - `terminates = bool(case.body)
  and isinstance(case.body[-1], (ast.Return, ast.Break, ast.Continue))`. A
  `case ...: sys.panic(...)` arm would have the identical narrowing-survival gap
  `_stmt_If` had. This is the single highest-priority item in this whole
  document: it's the same "silently accepted, breaks at C-emission" failure
  mode as the original Bug 3, not merely a spurious rejection. Repro sketch:
  mirror the original `span()` repro but with the two narrowing checks written
  as a `match`/`case` instead of `if`.
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

1. **`type_resolver.py:4593` (Shape 3, `visit_Match`)** - highest priority,
   identical failure mode to the original highest-priority bug (silently
   accepted, breaks at C emission instead of at type-check time). The fix
   pattern (`_stmt_diverges`) already exists and just needs wiring into
   `visit_Match`'s `terminates` computation the same way it was wired into
   `_stmt_If`.
2. **Shape 1's three high-confidence candidates** (`type_resolver.py:2834`,
   `lowering.py:8594`, `lowering.py:8627`) - same `is`-vs-`_same_type` shape as
   an already-fixed bug, in the same two files, with `_same_type` already
   available to swap in directly.
3. **Shape 1's two medium-confidence candidates** (`lowering.py:2061-2062`,
   `lowering.py:4281`) - worth a repro attempt each; the `_stmt_Return` one may
   turn out to be correct-but-duplicated rather than actually broken.
4. **`lowering.py:683` (Shape 3, `_body_may_fall_off_the_end`)** - low risk, low
   priority; at minimum update its stale comment.
5. Everything under "awareness only" - do not fix without first confirming with
   the user that the documented deliberate-design reasoning no longer holds.

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
