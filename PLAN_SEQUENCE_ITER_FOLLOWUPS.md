# Follow-ups from the `_sequence_iter` simplification attempt (2026-08-25)

**UPDATE 2 (same day, worktree `simplify-sequence-iter`): item 1 is FULLY
ROOT-CAUSED, both halves.** Two independent, real compiler bugs were found
and fixed; a third, narrower gap remains (understood, contained, not a
crash/corruption risk) that blocks `_sequence_iter` itself from being
rewritten with `match` for one specific conformer (`set[T]`). Both fixes
are committed on their own (`_sequence_iter` itself is NOT changed - still
the pre-existing if/is_err()/unwrap() shape on master).

**Fix 1 - crash (`mpy_types.py`):** `Name.__deepcopy__` now returns `self`.
Every `Name`/`Type`/`Function`/`Variable`/`Module`/... instance is an
identity-based singleton; `copy.deepcopy` reaching one via a cached
AST-node tag (`resolved_callee`) must never clone it. Root-caused via a
crashing `unit not in self.tagged_unions` membership check in
`compiler.py`: two non-identical-but-qualname-identical
`Result[i32,IndexError]` `TaggedUnion` objects were being compared,
triggering infinite recursion in dataclass `__eq__`. Traced to
`type_resolver.py`'s `_apply_live_flag_guards` deep-copying a promoted
field's own assignment statement (to build its "first assignment" branch)
and sweeping along a `resolved_callee`-tagged `Function` reference,
cloning its entire return-type graph - including supposedly-singleton
scalars. 2 new regression tests in `mpy_types_test.py`'s
`NameDeepcopyIdentityTestCase`.

**Fix 2 - type leak / reentrancy (`type_resolver.py`):** root cause is
`Monomorphizer.ensure_resolved`'s OWN documented, deliberate reentrancy
fallback (`monomorphize.py` - the exact same case its own comment already
describes for `dict[i32,i32].__iter__` needing `self`'s type while
`dict[i32,i32]` is itself still mid-build): when a generic CLASS's own
method body (e.g. `class Wrap[T](Sequence[T]): def __iter__(self): return
helper(self)`) is being monomorphized, and resolving that method's return
type requires resolving `helper[i32,Wrap[i32]]`'s own match-subject type,
that happens WHILE `Wrap[i32]` is still mid-build - hitting
`ensure_resolved`'s documented fallback to the ABSTRACT, still-TypeVar'd
`Wrap` template. `_reserve_generator_match_subject_fields`/`_reserve_
generator_match_binding_fields` had no way to detect this degraded answer
and PERMANENTLY baked `Wrap.T` (the class template's own internal
TypeVar) into the promoted field's declared type - wrong for every
instantiation, not just the one that happened to trigger the reentrant
path first (confirmed via a real repro: mixing `Wrap[i32]`/`Wrap[i64]` in
one program produced `expected Result[Wrap.T,...], got
Result[i32,...]`/`Result[i64,...]` real compile errors). **Fix:** both
reservation methods now call the already-existing `Monomorphizer.
_is_concrete()` check on the resolved subject type and DECLINE the
reservation (same safe fallback as the pre-existing `subject_type is
None` case) whenever it still contains a free TypeVar. Also added, as a
smaller, independently-useful improvement found along the way: `_type_of_
expr`'s `ast.Call` branch (used by the SAME reservation pass) didn't
substitute a generic receiver's own concrete type args into a found
method's declared return type at all (`receiver.__getitem__(...)`'s
return type came back with the RECEIVER CLASS's own unsubstituted
TypeVar, not the receiver's concrete arg) - fixed via `Monomorphizer.
substitute_type_params`, the same primitive monomorphization itself uses.
1 new regression test in `emitter_c_test.py`'s `GeneratorFunctionTests.
test_match_subject_reentrant_generic_class_resolution_declines_safely`.

**Both fixes verified: full suite clean on clang/MSVC/gcc(WSL), 1831/1831.**

**Remaining, narrower gap (NOT fixed, understood and contained):** fix 2's
"decline" safety net avoids the crash/type-corruption, but for the ONE
conformer that's ITSELF a generic class (`set[T]` - `str`/`bytearray`/
`memoryview`/`mmap` are all concrete, non-generic classes and are NOT
affected), declining the reservation means `set[T]`'s own match-subject
field falls back to the ORIGINAL (pre-`dc409bb`) unpromoted plain-local
shape - reintroducing THAT bug's own uninitialized-read risk, but ONLY
for `set[T]` specifically, and ONLY for an RC-typed element (`set[str]`,
`set[SomeRCClass]` - `set[i32]` and other scalar elements are unaffected,
since a scalar promoted-vs-plain-local distinction is moot, no decref
involved). Confirmed via a real MSVC compile of `set[str]` iteration:
`warning C4700: uninitialized local variable '__match_subj_0' used` /
`'item' used` (both real, matching the ORIGINAL bug's own signature) -
does NOT crash in current testing (matches the original bug's own
"works by luck" pattern, not proof of soundness). **This is why
`_sequence_iter` itself was NOT rewritten with `match` this session** -
doing so would ship a known (if narrow) regression for `set[str]`/
`set[<RCClass>]` iteration specifically. Fixing this properly needs the
SAME "chicken-and-egg" reentrancy problem `ensure_resolved`'s own
docstring already flags as deliberately unsolved (retry the reservation
once the enclosing class's REAL monomorphization finishes, rather than
declining permanently) - a bigger, riskier pipeline-ordering change,
matching the scope PLAN_GENERATORS.md's own "Why NOT fixed in this
session" sections describe for the sibling bugs. Left as a known, safe-
to-defer gap: `set[T]` iteration already has this risk on CURRENT master
too (via `_sequence_iter`'s own EXISTING if/is_err()/unwrap() shape,
which never promotes anything, so it never had this fix's protection to
begin with) - fix 2 does not make `set[T]` iteration any WORSE than it
already is today, it just doesn't make it any BETTER either.

**Where to pick this up (if solving `set[T]` iteration's residual gap, or
finally rewriting `_sequence_iter` with `match`):**
1. Confirm the gap is real and current: `METALPY_CC=msvc python mpy.py
   <a set[str]-iterating program>.py` and check for `C4700` on
   `__match_subj_N`/the arm binding.
2. The real fix is making `ensure_resolved`'s reentrant path RETRY rather
   than permanently accept the degraded answer - e.g. `_reserve_
   generator_match_subject_fields`/`_reserve_generator_match_binding_
   fields` could be re-run (or their result invalidated and recomputed)
   once the enclosing class's monomorphization actually completes, rather
   than running once, early, and being trusted forever. Needs real design
   work on WHEN/HOW to detect "the enclosing class just finished" and
   trigger a recompute - not attempted here.
3. Once that's solid, rewrite `_sequence_iter` with `match` (the ORIGINAL
   ask) and verify with a full multi-conformer suite run (str +
   bytearray + memoryview + mmap + `set[T]` with an RC element, all
   iterated in one compiled program) on all 3 compilers, plus explicit
   MSVC C4700/gcc -Wmaybe-uninitialized warning-absence checks for
   `set[<RCClass>]` specifically (the one shape that was actually broken).

## 2. Subscript-sugar (`x[i]`) not recognized by match-subject-type resolution

**Smaller, separate gap, safe to fix independently of item 1.**
`type_resolver.py`'s `_type_of_expr` (used by `_resolve_expr_type_for_
desugar`, which `_reserve_generator_match_subject_fields` calls) has a
documented, deliberate gap: its `ast.Subscript` branch only resolves a
homogeneous-tuple constant-index read or a str/bytearray slice
(`_byte_slice` auto-unwrap) — "every OTHER subscript shape (list[T]/
dict[K,V]/a user `__getitem__`, ...) is deliberately left unresolved
here." This means a match subject spelled as ordinary `seq[i]` sugar
(as opposed to `seq.__getitem__(i)`) silently declines promotion even
when it needs it, falling back to the original (pre-`dc409bb`) unsound
plain-local behavior for that one shape.

A working point-fix for this alone (independent of item 1) was drafted
and verified this session, then reverted alongside the larger revert
(easy to redo — see the earlier diff in this session's transcript /
worktree `simplify-sequence-iter`'s reflog if not GC'd):

- Added `TypeResolver._find_indexlike_getitem_type_only(owner_type)` —
  a type-only, read-only mirror of `lowering.py`'s own
  `_find_indexlike_getitem` (Overload-aware, prefers the Scalar-typed
  `__getitem__` leaf over a slice leaf), restricted to `owner_type.names`
  DIRECTLY (own declared members only — NOT `chain_lookup`, which walks
  inherited/protocol-conformance bases and can surface an unsubstituted
  Protocol stub method instead of the concrete override).
- Wired into `_type_of_expr`'s `ast.Subscript` branch as a fallback when
  `tuple_type is None`, returning the RAW `Result[T,E]` type (not
  auto-unwrapped — a match subject needs the tagged union itself, unlike
  the `_byte_slice` branch's own auto-consume use case).
- Explicitly guarded to bail out (`return None`) for anything that isn't
  a concrete `CStruct`/`RCClass` (excludes bare `TypeVar`/`Protocol`
  receivers, which surfaced item 1's cross-contamination when not
  excluded — a still-generic body should never resolve a promoted
  field's type off its own abstract Protocol bound).

This point-fix alone is safe and independently useful (str/bytearray/
memoryview/mmap are all concrete, non-generic classes, unaffected by
item 1's remaining `set[T]` gap), but was reverted alongside item 1's own
fixes to keep this session's landed changes minimal and independently
reviewable. **Safe to reapply now** (item 1's crash/type-corruption
halves are fixed) for any NON-generic-class `Sequence[T]` conformer using
`[i]` sugar as a match subject; still gated on item 1's remaining `set[T]`
gap (see above) if the target conformer is itself a generic class.

## Recommended path when picking this up

Item 1's crash and type-leak halves are fixed and merged-ready (see the
top-of-file update). What's left: (a) reapply item 2's subscript-sugar
fix if wanted, (b) solve item 1's remaining `set[T]`-specific reentrancy
gap (see item 1's own "Where to pick this up" above) if `_sequence_iter`
itself is to be rewritten with `match`, verified across the SAME multi-
conformer scenario that originally exposed item 1 (str + set[T] +
bytearray + memoryview + mmap all iterated in one compiled test program,
including a `set[<RCClass>]` case specifically, with explicit MSVC
C4700/gcc -Wmaybe-uninitialized checks).

## 3. `Result.or_return(mapper)` — ergonomic error-type conversion (separate feature, unrelated to items 1/2)

**Motivation:** `.or_return()` always propagates the receiver's OWN Err
payload unchanged. Converting one error type into another today requires
a full `match`/`if is_err()` block, which is noisy for a common need
("turn this `IndexError` into `StopIteration`", generally: adapt a
lower-level error into the caller's own error vocabulary). Wanted:
`.or_return(mapper)` where `mapper` is any callable (plain function,
non-capturing lambda, OR a real capturing closure) with signature
`(E) -> E2`, propagating `mapper(err)` instead of `err` on the Err path.

**Not implemented — scoped as a real language feature, comparable in size
to one of the `PLAN_GENERATORS.md` sub-phases.** Design worked out this
session:

### Why it doesn't fit today's `ir.OrReturn`/`ir.OrJump` shape
`or_return()` today is a SINGLE IR instruction (`ir.OrReturn`/`ir.OrJump`)
representing an entire `if err: widen-and-return else: dest=payload`
branch, built directly as a flat conditional in `emitter_c.py` — no real
CFG branching needed. A mapper call must run CONDITIONALLY (Err path
only) and needs a REAL Operand for the original error payload to pass as
its argument — but `_emit_widen_error`'s existing payload access is a
hand-built raw C string (`(value).data.err`), not a first-class
`ir.Operand`; there is no existing IR op to extract a tagged union
member as an Operand (match-arm binding extraction instead goes through
`cfg.narrow()`/`narrowed_member()`, which requires a real NAMED local,
not an arbitrary expression mid-lowering).

### Recommended implementation shape (design only, not built)
Desugar `<result_expr>.or_return(mapper)` at the AST level (same
"textually recognized, expand to existing primitives" posture
`_lower_or_return`'s own docstring already documents for the no-arg
form) into, roughly:

```python
__or_recv_N: Result[T,E] = <already-lowered receiver, bound via a real
                             ir.Assign — NOT re-evaluating the original
                             AST, to avoid double-evaluating a receiver
                             with side effects>
match __or_recv_N:
	case Result.Err(__or_err_N):
		return Result.Err( mapper( __or_err_N ) )   # reuses value-carrying
		                                             # return + _stmt_Return's
		                                             # EXISTING widening +
		                                             # defer/errdefer + inline-
		                                             # splice + generator
		                                             # pessimistic-done-pin
		                                             # machinery for free
	case Result.Ok(__or_ok_N):
		__or_ok_N   # this expression's own value
```

Key design points:
- Reusing `return Result.Err(...)` (already a fully-supported statement
  shape, including inside generator bodies — PLAN_GENERATORS.md's
  "value-carrying return") sidesteps re-implementing widening/defer/
  errdefer/inline-splice/generator-pessimistic-done handling a second
  time — `_stmt_Return`'s existing machinery already does all of it
  correctly for any Result-returning early exit.
- The mapper call itself (`mapper(__or_err_N)`) should go through the
  ordinary `_lower_call`/`_try_lower_closure_call`/`_try_lower_indirect_
  call` dispatch (already handles both raw `Ptr[Callable]` AND real
  capturing closures) by synthesizing an `ast.Call` node and lowering it
  normally — NOT hand-building `CallIndirect` IR, which would duplicate
  the closure-vs-plain-fn-ptr dispatch logic that already exists and is
  tested.
- Binding the receiver into `__or_recv_N` must reuse the ALREADY-lowered
  `ir.Operand` (never re-lower the original AST expression — this
  codebase is careful everywhere else about not double-evaluating a
  receiver with side effects, e.g. `sock.recv(...).or_return()`). This
  needs the same `cfg.assign()`-style alias-tracking every other
  local-binding path uses, gotten right the first time — this is the
  single highest-risk step (this codebase's own memory log is full of
  incref/decref bugs from exactly this kind of new-binding-of-an-
  existing-operand shape).
- Since this expands to a real `match` + `return`, it works as an
  EXPRESSION (needed for `yield seq[i].or_return(mapper)`-style usage)
  by treating the whole construct the same way ternary/other synthesized-
  statement-sequence-as-expression shapes already do in this codebase:
  build the statements, then reference the Ok arm's own bound value as
  the resulting operand.

### Scope for a first landing
- Support both plain functions/lambdas AND real closures (per explicit
  request) — the ast.Call-based dispatch above gets this for free,
  unlike an earlier (rejected) design that hand-rolled the call and
  would have needed to special-case closures separately.
- Verify: RC correctness (no leak/double-free of the original error
  payload, no leak/double-free of the mapper's own result), interaction
  with `defer`/`errdefer` (goes through ordinary `return`, so should
  compose for free — verify with a real repro), interaction with
  `@inline` splices (same), interaction with generator bodies (same —
  a generator's own fallible `or_return()` already works via this exact
  return-based mechanism per PLAN_GENERATORS.md Phase 8).
- Add real compile+run tests mirroring `or_return_rc_test.py`'s existing
  refcount-delta-checking style, on all 3 compilers.

### Explicitly NOT scoped for a first landing
- Nothing — the closure-supporting design above is the FULL feature as
  requested. No further restriction was agreed; do not narrow scope
  without checking back in.
