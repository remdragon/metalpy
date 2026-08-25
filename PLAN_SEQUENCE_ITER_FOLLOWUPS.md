# Follow-ups from the `_sequence_iter` simplification attempt (2026-08-25)

**DONE (worktree `sequence-iter-tuple-fix`): `_sequence_iter` is now
rewritten with `match`, all four compiler bugs found along the way are
fixed, full suite clean 1856/1856 on clang/MSVC/gcc(WSL).** The tuple gap
UPDATE 3 (below) left open turned out to be one more instance of the SAME
reentrancy-recovery mechanism, just missing @overload support: `type_
resolver.py`'s recovery path only handled a plain (non-overloaded)
`Function` found via `Specialization.names` - `VariadicTuple[T]`
(`tuple[T,...]`'s real backing class)'s own `__getitem__` is `@overload`'d
(usize/slice), so the lookup returned an `Overload` group instead and the
recovery silently declined. Fixed by resolving the `Overload` group's own
winning leaf first (via the SAME arg-matching `_overload_call_return_
type` this method already used elsewhere for an ordinary `Overload`-typed
call target), then substituting that leaf's return type exactly like the
plain-`Function` case. Confirmed fixed via a real MSVC compile of
`tuple[Elem,...]` iteration (`Elem` an RCClass) - the `C4700` warnings on
`__match_subj_0`/`item` are gone, matching the earlier `set[str]` fix.
Verified with the SAME multi-conformer full-suite run this doc's own
"Where to pick this up" sections called for (str + bytes + bytearray +
memoryview + mmap + `set[T]` + `tuple[T,...]`, all exercised by the
existing suite, plus 4 dedicated regression tests). +2 new regression
tests this round (on top of the 2 already added for the `set[T]` fix),
`GeneratorFunctionTests.test_match_subject_reentrant_generic_class_
resolution_overloaded_getitem`.

The rest of this document is the historical trail (three prior updates,
four compiler bugs total) kept for reference - see git history/blame on
`lib/builtins/__init__.py`'s `_sequence_iter` and `type_resolver.py`'s
`_type_of_expr`/`_reserve_generator_match_subject_fields`/`_reserve_
generator_match_binding_fields` for the actual landed code.

**UPDATE 3 (worktree `sequence-iter-match-retry`): the `set[T]` gap is now
ALSO fixed (real root-cause fix, not just decline) - but a THIRD,
DIFFERENT reentrancy variant was found, affecting homogeneous variadic
tuples (`tuple[T,...]`) with an RC element. `_sequence_iter` STILL cannot
be safely rewritten with `match` yet - see "UPDATE 3" further down for
full detail.** Three independent, real compiler bugs found and fixed so
far this multi-session effort; a fourth (tuple-specific) remains, not yet
root-caused. All landed fixes are committed on their own -
`_sequence_iter` itself is UNCHANGED on master, still the pre-existing
if/is_err()/unwrap() shape.

**UPDATE 2 (worktree `simplify-sequence-iter`, superseded by UPDATE 3
above): item 1 was believed fully root-caused at the time** - two
independent real compiler bugs found and fixed, with a third gap believed
narrow and specific to `set[T]`. UPDATE 3 found and fixed that `set[T]`
gap for real, but ALSO found the tuple variant, so `_sequence_iter`
remains unchanged. Kept below for the historical trail.

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

**UPDATE 3 (same day, worktree `sequence-iter-match-retry`): the `set[T]`
gap above is FIXED too** (a real root-cause fix, not just a wider
"decline" net) - **but a THIRD, DIFFERENT reentrancy variant was found
along the way, affecting variadic tuples specifically, still unfixed.**
Net result: `_sequence_iter` STILL cannot be safely rewritten with
`match` this session, but the reason has narrowed from "`set[T]`" to
"homogeneous variadic tuples (`tuple[T,...]`) with an RC element".

**Fix 3 (`type_resolver.py`, on top of fix 2):** rather than accepting
`ensure_resolved`'s degraded-to-abstract answer and declining, `_type_of_
expr`'s `ast.Call` branch now DETECTS the degradation (`ensure_resolved`
returning literally `original_specialization.base`, discarding its own
args - the exact identity signature of the documented reentrancy
fallback) and RECOVERS the real answer directly: looks up the same
method on the abstract base via `Specialization.names` (a cheap
passthrough to `.base.names` - mpy_types.py - that NEVER itself triggers
`Monomorphizer.monomorphize_class`, so it can't retrigger the reentrancy),
then substitutes its declared return type against the ORIGINAL
Specialization's own real args via `Monomorphizer.substitute_type_params`
- the same primitive real monomorphization already uses. Critically, the
recovered `Function` is used ONLY to read its declared return type -
never scheduled/resolved as its own compile unit (an earlier attempt at
this fix let the abstract template Function flow into the method's own
shared "resolve and schedule" handling and crashed elsewhere with
`AttributeError: 'Specialization' object has no attribute 'node'`,
confirming that path assumes a real per-instantiation Function, not the
shared abstract template).

**Verified TWICE, for two different symptoms of the SAME underlying
degradation:**
1. The reentrant-class repro from fix 2 (`Wrap[T]`/`Wrap[i32]`+`Wrap[i64]`)
   now compiles AND RUNS correctly (previously: real compile errors).
2. `set[str]` iteration under `_sequence_iter` rewritten with `match`: the
   `C4700` warnings on `__match_subj_0`/`item` are GONE (previously
   present, matching the original pre-`dc409bb` bug's own signature).

**Caught a real regression while building this, now fixed too:** the
first version of fix 3 ALWAYS preferred the new Specialization-based
lookup, even for the ORDINARY (non-reentrant) case - which broke
`list[T].__iter__`'s own return type (a real synthesized `Generator`
backing class, which the fast lookup can't produce, since it deliberately
never calls `ensure_generator_synthesized`). Fixed by trying the
ORIGINAL `ensure_resolved` path FIRST, as before, and only falling back
to the recovery path when the degradation is actually detected
(`receiver_type is original_receiver_spec.base`) - confirmed via the full
suite going from a real failure (`for loop requires an IteratorProtocol
[T] or Iterable[T] conformer, got Generator[...]` on `enumerate`/`list`)
back to clean.

**Fix 3 alone (WITHOUT rewriting `_sequence_iter`) is verified clean on
clang/MSVC/gcc(WSL), 1852/1852** (only the one pre-existing, unrelated
`deflate_test.py` zlib-hex-parsing failure, confirmed via `git stash` to
already exist on unmodified master, nothing to do with this work).
**Committed on its own, real, independently-useful fix, same posture as
fixes 1/2.**

**THE NEW, THIRD GAP - homogeneous variadic tuples specifically:**
rewriting `_sequence_iter` with `match` ON TOP of fix 3 and running the
full suite surfaced a NEW MSVC-only crash (`STATUS_BREAKPOINT`, an
`/RTC1` uninitialized-variable trap - NOT the same code path fix 3
covers) in `emitter_c_test.py`'s `VariadicTupleTests.test_programs_
compile_and_run`. Isolated to a standalone repro (`tuple[Elem,...]`
iteration, `Elem` an RC class) - real `C4700` on `item`/`__match_subj_0`
under MSVC even with fix 3 applied, confirming fix 3's own reentrancy-
detection does NOT cover this shape. Root cause not yet traced, but the
mechanism is clearly DIFFERENT from fixes 1-3: a variadic tuple's own
`Sequence[T]` conformance is synthesized directly by `tuple_storage.py`'s
`TupleStorage._declare_sequence_conformance` (building the backing
`RCClass` itself, not through the ordinary `class Foo[T]:`/
`Specialization` ordinary-generic-class machinery fix 3's own detection
is keyed on) - `receiver_type` for a tuple's own `seq: S` parameter
inside `_sequence_iter` may never even BE a `Specialization` the way
`Wrap[i32]`/`set[i32]` are, meaning fix 3's `original_receiver_spec is
not None` guard likely just never fires for tuples at all, leaving them
on the OLD, still-buggy `ensure_resolved`-only path. **`_sequence_iter`
still cannot be safely rewritten with `match` until this is understood
and fixed too** - reverted again this session, `_sequence_iter` itself
is UNCHANGED on master (still the if/is_err()/unwrap() shape).

**Where to pick this up:**
1. Minimal standalone repro (already have one, non-generic-class-based):
   ```python
   import compiler
   class Elem: pass
   def main() -> i32:
   	xs: list[Elem] = list[Elem]()
   	xs.append( Elem() ); xs.append( Elem() ); xs.append( Elem() )
   	t: tuple[Elem, ...] = tuple( xs )
   	count: i32 = 0
   	with compiler.wrap_arithmetic:
   		for e in t:
   			count += 1
   	return count - 3  # 0 on success
   ```
   Compile with `METALPY_CC=msvc python mpy.py <file>.py` against a
   `_sequence_iter` rewritten with `match` (temporarily, for testing) and
   check for `C4700` on `item`/`__match_subj_N`.
2. Trace `tuple_storage.py`'s `_declare_sequence_conformance`/`tuple_type_
   for` to find what TYPE OBJECT actually flows as `seq`'s own parameter
   type inside `_sequence_iter[T,S]` when `S` = a variadic tuple's own
   backing class - is it a `Specialization` at all, or something else
   (`TupleType` directly, a plain already-built `RCClass` with no
   Specialization wrapper)? This determines whether fix 3's existing
   detection can be extended to cover it, or whether tuples need their
   OWN separate reentrancy-recovery path.
3. Once understood, extend (or add a sibling to) fix 3's detection in
   `_type_of_expr`'s `ast.Call` branch, verify with the SAME standalone
   repro plus the full `VariadicTupleTests` suite on all 3 compilers,
   THEN attempt the `_sequence_iter` `match` rewrite again with a full
   multi-conformer verification pass (str + bytearray + memoryview + mmap
   + `set[<RCClass>]` + `tuple[<RCClass>,...]`, all iterated in one
   compiled program, explicit MSVC `C4700`/gcc `-Wmaybe-uninitialized`
   absence checks for every RC-element case).

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
