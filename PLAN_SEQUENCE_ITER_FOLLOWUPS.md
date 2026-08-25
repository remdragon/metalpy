# Follow-ups from the `_sequence_iter` simplification attempt (2026-08-25)

**UPDATE (same day, worktree `simplify-sequence-iter`): item 1's CRASH half
is FIXED** — `mpy_types.py`'s `Name` gained a `__deepcopy__` returning `self`
(every `Name`/`Type`/`Function`/`Variable`/`Module`/... instance is an
identity-based singleton; `copy.deepcopy` reaching one via a cached
AST-node tag like `resolved_callee` must never clone it). Root-caused via
`compiler.py`'s crashing `unit not in self.tagged_unions` check: two
non-identical-but-qualname-identical `Result[i32,IndexError]` TaggedUnion
objects were being compared, traced to `type_resolver.py`'s
`_apply_live_flag_guards` deep-copying a promoted field's own assignment
statement (to build its "first assignment" branch) and sweeping along a
`resolved_callee`-tagged `Function` reference, cloning its entire
return-type graph — including supposedly-singleton scalars. **Full suite
verified clean on clang/MSVC/gcc(WSL), 1830/1830, with 2 new regression
tests in `mpy_types_test.py`'s `NameDeepcopyIdentityTestCase`.** Ready to
commit/merge on its own — real, independently-justified fix, unrelated to
whether `_sequence_iter` itself ever gets rewritten.

**Item 1's TYPE-LEAK half is NOT fixed** — seeSection "Remaining work"
below: even with the crash gone, `_sequence_iter` rewritten with `match`
still produces a real compile error (`expected Result[set.T,...], got
Result[i32,...]`) when multiple `Sequence[T]` conformers share it in one
program. This is a SEPARATE bug from the crash (different symptom, not
yet root-caused) — do not assume the deepcopy fix resolves it.

Two more independent pieces of work were scoped but not implemented this
session (item 3, and item 1's remaining type-leak half). Start from a
fresh `EnterWorktree` per CLAUDE.md.

## 1. Cross-instantiation match-subject promotion cache bug (crash FIXED; a type-leak variant remains)

**Symptom:** compiling a program that iterates two DIFFERENT concrete
`Sequence[T]` conformers (e.g. `str` and `set[i32]`) through the SAME
shared generic generator function, where that generator's body contains a
`match <non-Name subject>: ... case Ok(x): yield x ...`, produces bogus
type errors like:

```
expected builtins.Result[builtins.set.T,builtins.IndexError],
got builtins.Result[intrinsics.i32,builtins.IndexError]
```

**Root cause (confirmed via repro, not yet traced to an exact line):**
`type_resolver.py`'s `_reserve_generator_match_subject_fields` (added by
the match-subject-yield-resume fix, commit `dc409bb`, merged) tags the
match statement's own AST node with a promoted field name + resolved
type, so `visit_Match` can build `self.<stem> = ...` instead of an
unpromoted, RC-unsafe plain local. This machinery was implicitly assumed
to run fresh per Function instantiation. It does NOT, for a shared
generic generator body: the type resolved for the FIRST monomorphized
instantiation that reaches this pass appears to stick (via node mutation
or some other cache) for every LATER instantiation sharing the same
underlying `fn.node` — even though `ensure_generator_synthesized` is
supposed to be `id(fn)`-memoized per PLAN_GENERATORS.md, `fn.node` itself
may be a SHARED object across Specializations of one generic function,
not deep-copied per instantiation.

**Reproduced with BOTH spellings** — `match seq[i]:` (subscript sugar)
AND `match seq.__getitem__(i):` (explicit call) — ruling out the
subscript-sugar gap (see item 2 below) as the cause. Confirmed via:

```python
# lib/builtins/__init__.py, _sequence_iter, rewritten to:
def _sequence_iter[T, S: Sequence[T]]( seq: S ) -> Generator[T, StopIteration]:
	i: usize = 0
	while True:
		match seq.__getitem__( i ):
			case Result.Ok( item ):
				yield item
			case _:
				return
		with compiler.panic_arithmetic( '...' ):
			i += 1
```

then running the full test suite (`python tests.py`) — since `_sequence_iter`
is shared by `str`/`bytearray`/`memoryview`/`mmap`/`set[T]`'s own
`__iter__`, multiple concrete instantiations compile in the same program
and collide.

**Why this matters beyond `_sequence_iter`:** ANY shared generic generator
function with a non-Name match subject crossing a yield, instantiated more
than once in the same compiled program, is affected — not hypothetical,
just never previously exercised (existing tests for the subject/binding
promotion fixes used single-instantiation repros).

**Attempts to build a minimal standalone repro for the TYPE-LEAK half (all
tried this session, none reproduce it in isolation — the bug needs
`_sequence_iter`'s REAL shape, not just "some generic generator, called
twice"):**
- A free generic helper `probe[T](v,ok)` called AS the match subject
  inside `gen[T](v)` — does NOT reproduce the type-leak; instead hits a
  SEPARATE, already-documented, deliberately-scoped limitation
  (PLAN_GENERATORS.md Phase 7 point 5: a generic generator body calling
  ANOTHER generic function via its own type param fails cleanly with
  "cannot build a generator zero-placeholder value for ...probe.T"). Not
  useful as a repro — it's a different, known gap.
- A generic class `Box[T]` with its own method `probe(self,ok)->Result[T,
  IndexError]`, called as the match subject inside `gen[T](b: Box[T])`,
  instantiated as `Box[i32]` and `Box[i64]` in one program — compiles AND
  RUNS CLEANLY (exit 0), both before and after the deepcopy fix. Does NOT
  reproduce either half of the bug (not the old crash, not the type-leak).
- Conclusion: whatever's special about `_sequence_iter`'s real trigger
  isn't just "shared generic generator + match-with-yield + 2
  instantiations" — something about ITS SPECIFIC shape matters (two type
  params `[T, S: Sequence[T]]` with a PROTOCOL-parametrized bound, not
  just one plain `[T]`; and/or being reached via ANOTHER generic class's
  OWN method body — `set[T].__iter__` calling `_sequence_iter(self)` — as
  opposed to a plain top-level call). The "set.T" in the real error
  (`expected ...Result[builtins.set.T,...]`) strongly suggests the leak
  specifically involves `set[T].__iter__`'s OWN still-abstract `T`
  bleeding into `_sequence_iter`'s reservation, not just any generic T.
  **Next attempt should start from a generic CLASS whose method calls a
  SEPARATE generic function bound via a protocol-parametrized TypeVar
  (`S: Sequence[T]`)** — closer to matching `_sequence_iter`'s exact
  parametrization — rather than a single-type-param free function.

**Where to start:**
1. Build the closer repro above (generic class method → separate function
   with a protocol-parametrized TypeVar bound) to isolate the type-leak
   half from `_sequence_iter` itself while still reproducing it.
2. Trace whether `fn.node` is actually the SAME Python object across two
   Specializations of one generic Function (`id(fn1.node) == id(fn2.node)`)
   — if so, that's the structural root cause, and either (a) `_reserve_
   generator_match_subject_fields`/`_reserve_generator_match_binding_fields`
   need their own per-instantiation state (not stored on the shared node),
   or (b) `fn.node` needs to be deep-copied per instantiation earlier in
   the pipeline (bigger, riskier change).
3. Check whether `_build_generator_backing_class`/`extra_locals` threading
   has the same issue independently of the AST tagging.
4. Follow the existing methodology for this subsystem (small, incremental,
   real-compile-verified steps, all 3 compilers) — see
   `[[match_subject_yield_resume_uninitialized_read_confirmed]]` memory
   entry for how the sibling bugs were actually root-caused and fixed.

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

This point-fix alone is safe and independently useful, but was reverted
because it doesn't fix item 1 — a program using `[i]` sugar as a match
subject in a shared generic generator would still hit the cross-
instantiation cache bug once fixed to resolve at all. **Land item 1
first** (or verify the specific target generator is never multiply-
instantiated) before reintroducing this.

## Recommended path when picking this up

Fix item 1 first (it's the load-bearing correctness bug — affects code
that ALREADY compiles today, silently). Once verified fixed (multi-
instantiation repro clean on all 3 compilers), item 2 can be reapplied
on top to let `_sequence_iter` (and anything else) use `match subject[i]:`
sugar safely. Only then attempt the original ask: rewrite
`_sequence_iter` using `match`, verified across the SAME multi-conformer
scenario that exposed item 1 (str + set[T] + bytearray + memoryview +
mmap all iterated in one compiled test program).

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
