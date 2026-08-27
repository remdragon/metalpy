# Thread-safe module globals and instance fields — closing the "naive racy code corrupts memory" gap

## Status

**Part A (module globals) implemented and confirmed correct for both
direct and narrowed reads/writes, on both Windows and Linux.** Part B's own
**prerequisite** (class-level `_`/`__` field-visibility enforcement, see
below) is implemented and merged.

**Part B itself (the actual per-object lock) - status update: implemented,
all previously-blocking bugs fixed, verified under real concurrent stress
(see below), full 3-compiler suite (clang/MSVC/WSL-gcc) clean, and merged
to master.**

**Performance follow-ons (the "Cost mitigations" section below) - status
update: staged into 4 sessions, all 4 stages landed and merged (see Cost
mitigations #1/#2/#3/#4's own status updates for the full writeups).**

What's built: `ir.AcquireFieldLock`/`ReleaseFieldLock` markers (ir.py,
mirroring Part A's global-lock markers, keyed on the receiver operand
instead of a Variable); `ObjectHeader` growth (`emitter_c.py`'s
`_object_header_prologue()`, a `void*`/`pthread_mutex_t` lock field picked
at emission time, same A.3 platform asymmetry as Part A - POSIX needs a
real `pthread_mutex_init()` at every construction site, no zero-init
guarantee); `acquire_field_lock`/`release_field_lock` helper functions that
skip locking entirely for an IMMORTAL object (required, not optional - a
compile-time-baked static instance's `.lock` field is never initialized,
so locking it would be real UB); `GetAttr`/`SetAttr` wrapping at every
genuine field chokepoint (`lowering.py`'s `_expr_Attribute`, `_stmt_
Assign`'s plain-write and augmented-assign branches), gated on a new
`_is_real_field_receiver` check (RCClass only - NOT CStruct, which has no
`$header` of its own, embedded by value or, for `@interface`, heap-
allocated without a header at all; NOT a TaggedUnion's own `.tag`/`.data`
storage-view accessors either); a matching fix to `compiler.decref(obj.
field)` (used by the synthesized destructor's own field teardown,
`_build_field_teardown_ast`) to consume the field's own reference directly
under one critical section rather than retain-then-immediately-undo. The
whole-program on/off switch (Cost mitigations #1) was deliberately NOT
built for this pass - Part A itself never had one either (see A.1's own
scoping), and the memory-cost tradeoff didn't seem worth gating on an
as-yet-unbuilt Thread-reachability analysis before landing correctness.

**Four real bugs found and fixed during implementation, each confirmed via
a real compile-and-run repro (not just reasoning), worth recording:**
1. Locking a TaggedUnion's own internal `.data` payload view (indistin-
   guishable in AST shape from a real field, reinterpret-cast to
   ObjectHeader* is nonsense) - fixed via `_is_real_field_receiver`.
2. Same shape for a by-value-embedded (or `@interface`, header-less)
   CStruct field - `_is_real_field_receiver` narrowed to RCClass only,
   not the wider InheritanceChainMixin `_check_field_visibility` uses.
3. `compiler.decref(obj.field)` (the synthesized destructor's own field
   teardown) leaked exactly one reference per torn-down RC field - the
   naive retain-on-read at the GetAttr site created a SEPARATE owned copy
   that `compiler.decref`'s own single explicit Decref then just
   cancelled back out, never releasing the field's own original
   reference. Fixed by lowering `compiler.decref(obj.field)` as its own
   atomic acquire/bare-read/decref/release, bypassing the general
   retain-on-read path entirely for this one intrinsic shape.
4. **The loop-condition leak** (this section's own former blocker): a
   while-loop's own condition (`_lower_truth_test(node.test)`,
   `_stmt_While`) is lowered exactly ONCE at compile time, but the
   resulting C sits physically inside the loop and re-executes once per
   real iteration - a read-side `AcquireFieldLock`/incref emitted there
   fired N times at runtime, while the matching decref (driven by
   `_new_temp`/`fresh_temp`'s ordinary pending-temps flush, which only
   runs once per STATEMENT-level lowering pass) fired exactly once.
   Confirmed via `compiler.refcount()`: a generator's own `while i < b.v:`
   condition (`b` a captured `Box` parameter) leaked 2 references after 2
   iterations before the generator was dropped mid-consumption. Not
   generator-specific - any while-loop condition reading an RC field
   through a chained receiver has the same exposure; loop BODY statements
   were never affected (each already gets its own per-iteration flush via
   the ordinary `_lower_stmt` boundary - only the CONDITION expression,
   lowered once but executed repeatedly, was exempt from that). **Fixed**
   by an explicit `self._flush_pending_temps()` call in `_stmt_While`,
   right after computing the condition and before its own `JumpIfFalse` -
   safe even though the boolean `test` operand itself gets flushed too,
   since `ir.DeleteTemp` is a pure bookkeeping no-op at emission time ("C
   block scoping already handles temp lifetime"). `_lower_for_range`/
   `_lower_for_over_indexable`/`_lower_for_over_iterator` were all audited
   and confirmed NOT to share this exposure - their own per-iteration
   tests are either purely-scalar synthesized comparisons or a real
   `__next__()` Call (always a fresh, non-retained result), never a
   chained field read. Two new regression tests
   (`thread_safe_fields_test.py`) confirm this via `compiler.refcount()`,
   each independently confirmed to fail when the fix is sabotaged.
   A fifth, closely related ordering bug surfaced fixing this one and is
   folded in here rather than given its own number: `_object_header_
   prologue()`'s own `pthread_mutex_t lock;` field silently fell back to
   C's legacy "implicit int" behavior on gcc (no error, wrong field type,
   every `pthread_mutex_init()` call site then rejected as "incompatible
   pointer type") because `<pthread.h>` was only ever included later, by
   emit_c()'s own general required-headers loop - fixed by including it
   directly inside `_object_header_prologue()` itself, ahead of the
   struct definition it protects.

**Real concurrent stress tests - status update: added, confirmed real.**
`thread_safe_fields_test.py` now also covers, mirroring `thread_safe_
globals_test.py`'s own three-way split for Part A:
- **A direct read/write of a plain, non-Optional RC-typed field on a
  shared object** (40 OS threads, 8 reassigning while 32 concurrently
  read, 2000 iterations each - `test_concurrent_field_read_write_stress`).
- **A narrowed read of a union-typed field** (`self.g: Box|None`, `if
  self.g is not None: b: Box = self.g` read from 32 threads while 8
  concurrently reassign it via a plain `self.g = Box(n)` SetAttr, 3000
  iterations each, no manual lock at all -
  `test_narrowed_field_read_concurrent_stress`). Deliberately built
  around a pre-initialized field reassigned to a new value, not the
  `if self.g is None: self.g = compute()` lazy-init shape Part A's own
  equivalent global test uses - that specific "narrow after an in-branch
  assignment" shape is NOT currently supported for a field at all
  (confirmed via a real repro, `self.g: expected Box, got Box|NoneType`),
  a genuine, separate, pre-existing compiler gap unrelated to Part B,
  out of scope for this plan.
- **A plain scalar field** (`test_scalar_field_unaffected` - functional
  only, confirms Part B's own `cfg.rc_leaves`-gated early return still
  correctly skips locking a field with no RC leaves at all, the field-
  shaped counterpart of Part A's own `test_scalar_global_unaffected`).

Both concurrency tests were confirmed to be REAL fixes, not no-ops that
happen to pass: temporarily sabotaging `_is_real_field_receiver` to
always return `False` (disabling Part B's locking entirely, the single
chokepoint every read/write wrap site is gated on) reproduced a genuine
crash (`STATUS_ILLEGAL_INSTRUCTION`, the same signature class this whole
mechanism exists to close) in 15/15 runs of the read/write stress test
and 5/5 of the narrowed-read one; 20/20 runs clean again once restored.
Full 3-compiler suite (clang/MSVC/WSL-gcc, 1749 tests) clean.

**Part B is now considered verified to the same bar Part A was** - no
further blockers are tracked in this document. Whether/when to actually
merge is a separate decision from whether the implementation itself is
sound.

**Field-visibility enforcement (the Prerequisite section below) - status
update: implemented and merged**, not just designed. `Discovery.check_
field_visibility` (discovery.py, mirroring `check_module_visibility`'s own
shape) + `Type.in_protected_scope`/`InheritanceChainMixin.field_owner`
(mpy_types.py, siblings of `in_private_scope`/`chain_lookup`) +
`FunctionLowering._check_field_visibility` (lowering.py, wired into the 4
genuine user-facing `obj.field` chokepoints: ordinary read, plain-assign
write, augmented-assign, `compiler.addrof(x.field)` - deliberately NOT
inside `_attr_lookup` itself, which is also reached by internal synthesized
lookups like a tuple element's `_N` field that must stay unchecked).
Confirmed via the exact repro this section's own text below gives (`b.
__secret = 99` now a compile error) and via a real `lib/` audit: one
genuine violation found and fixed (`datetime.timedelta`'s own `_total_us`
field, read directly by `Date`/`Datetime` arithmetic in the same module but
a different, non-subclass class - renamed to a public `total_us`, the
"legitimately needs cross-class access" resolution this document's own
Cost-mitigation-#2 discussion anticipated for the public/protected tiers).
One real false positive found and fixed along the way, worth recording:
the compiler-synthesized `$$__destructor__` (type_resolver.py's
`_synthesize_rcclass_destructor`) is built with `cls=None` even though its
whole job is decref'ing every field of its own class, public or private -
now explicitly exempted (`Function.is_destructor`) rather than made to
carry a real `.cls` neither its own synthesis nor anything else needed
before now. See `discovery_test.py`'s new `FieldVisibilityEnforcementTests`
for the regression suite. Full 3-compiler test suite (clang/MSVC/WSL-gcc)
verified clean.

What's actually shipped for Part A (`cfg.py`/`lowering.py`/`emitter_c.py`/
`ir.py`/`mpy_types.py`):
- Detection: `Variable.reassigned_outside_init`, flipped by `cfg.py`'s
  `assign()` the moment a `global X; X = ...` reassignment is lowered
  (never for a global's own module-level initializer). A global's own
  initializing write goes through a *separate* method,
  `cfg.py`'s `assign_global_initializer()` (called from `lowering.py`'s
  `run_global()`) - structurally identical to `assign()`'s own
  `dest.is_global` write branch, but deliberately does NOT flip
  `reassigned_outside_init` itself. **Status update:** this used to be a
  real gap - a global's initializer can call an ordinary function
  (SYNTAX.md: initializers aren't restricted to compile-time constants,
  they run real code at program-startup time), and that function can spawn
  a thread which concurrently reassigns the SAME global through the
  fully-locked ordinary path *while* `__metalpy_init()` is still running
  other initializers - a genuine, reachable torn-write-plus-leak race
  (confirmed: 8/20 sabotaged runs leaked a `Box`, 0/25 with the fix -
  `thread_safe_globals_test.py`'s `test_global_init_write_race_stress`).
  Fixed by giving a global's own initializing write the same
  Acquire/decref-current-value/Assign/Release critical section an ordinary
  reassignment gets, gated at emission time the same way every other
  marker already is - so a global that's genuinely never reassigned from a
  function body still costs nothing (the markers become no-ops), while one
  that is gets real protection for its own first write too, not just
  later ones.
- A per-global lock, synthesized only for globals that end up needing
  one - platform-shaped, not a single portable primitive (A.3's own
  documented asymmetry): on Windows, a bare `static void*` holding an
  `SRWLOCK`, zero-init, no separate init function needed (SRWLOCK's
  all-zero state IS a valid unlocked lock, per Win32's own contract); on
  Linux, a real `static pthread_mutex_t`, explicitly initialized via a
  genuine `pthread_mutex_init()` call emitted at the very top of the
  synthesized `__metalpy_init()` (zero-initializing a `pthread_mutex_t` is
  not a portable guarantee, unlike SRWLOCK - see A.3's own note; this
  deliberately does not rely on it even though glibc happens to tolerate
  it). `<pthread.h>` is force-included whenever any global needs this on
  Linux (`compiler.disco.required_headers.add('pthread.h')`,
  `emit_c()`), the same way kernel32's SRWLOCK exports are force-linked on
  Windows - a compiled program gets this regardless of whether it itself
  imports `posix.pthread`/`lib/threading.py` for anything else.
- Real `ir.AcquireGlobalLock`/`ir.ReleaseGlobalLock` marker instructions
  (not emission-time pattern-matching - an earlier version tried
  reconstructing critical-section boundaries by looking for adjacent
  `Decref`+`Assign` instructions at emission time and was confirmed
  unsound: a union-typed global's decref/incref is a multi-instruction
  tag-check+extract sequence, not a bare `Decref`/`Incref`, so the pattern
  never matched the exact shape that caused the original bug).
- **A direct read/write of a plain, non-Optional RC-typed global**:
  confirmed correct under real concurrent stress (40 OS threads, 8 of them
  reassigning while 32 concurrently read, 2000 iterations each - see
  `thread_safe_globals_test.py`'s `test_concurrent_read_write_stress`).
- **A narrowed read of a union-typed global** (`X: SomeClass|None`, `if X
  is None: X = compute(); ...; return X` - **the exact shape
  `lib/datetime.py`'s `localtz()` and `lib/termcolor.py`'s `_codes()`
  themselves use**): also confirmed correct under real concurrent stress
  (64 threads, 3000 iterations each, no manual lock at all - see
  `test_narrowed_read_concurrent_stress`). Narrowing extracts a union's
  payload via a *separate* lowering.py code path (`_expr_Name`'s own
  narrowed-read rewrite, `lowering.py` around the `member is not None`
  branch) that used to return a bare, unprotected view with no lock at
  all - fixed by having that extraction perform its own protected
  extract-and-retain (Acquire, both `GetAttr`s, `Incref`, Release, all in
  one critical section) and register the result via `cfg.py`'s existing
  `fresh_temp()`/`is_fresh_temp()` tracking (the same mechanism an
  ordinary `Call`/`Allocate` result already uses) so that whichever of
  `cfg.assign()`'s `is_alias` branch or `_incref_aliasing_return` (the
  `return X` path specifically - a *different* mechanism from
  `cfg.assign()`, confirmed by reading it directly) consumes the result
  next recognizes it's already owned and doesn't increment it a second
  time. **This fix is scoped to `_expr_Name` only** (module globals and
  locals) - `_expr_Attribute`'s own, separately-duplicated narrowed-read
  rewrite (for a narrowed *field*, e.g. `self._g.field`) is untouched,
  since field-level locking is squarely Part B, not attempted this pass.
- Both stress tests verified on clang, MSVC, AND WSL/gcc (Windows and
  Linux targets alike - `emitter_c.py`'s `_global_lock_supported()`
  returns `True` for `_target_os in ('windows', 'linux')`). The Linux leg
  was confirmed to be a REAL fix, not a no-op that happens to pass:
  temporarily sabotaging `_global_lock_supported()` to exclude `'linux'`
  and re-running the narrowed-read stress binary under WSL/gcc reproduced
  real SIGILL crashes (3/30 runs) with the exact same signature as the
  original bug this mechanism exists to close, immediately restored after
  confirming that. macOS is deliberately still excluded (returns `False`
  there too) - this repo's own verified compiler/target matrix is
  Windows-x64 and Linux-x64 only; claiming macOS support would be
  untested, not just unimplemented, even though pthread_mutex_t exists
  there too.
- **Deterministic, non-timing-dependent regression coverage** alongside
  the (necessarily non-deterministic) stress tests:
  `lowering_test.py`'s exact-IR assertions confirm the lock markers land
  in precisely the right positions with zero threading involved - a
  regression here fails 100% of the time instead of "probably, if the
  race happens to fire." Covers: a protected global's write (Decref+Assign
  share one lock), a protected global's direct read (Incref+Assign share
  one lock), a scalar (non-RC) global getting no markers at all on either
  side, and the narrowed-read case (checked structurally - the presence
  and adjacency of Acquire→[2×GetAttr]→Incref→Release as one block, not a
  full exact-IR match, since the surrounding narrowing control flow is
  incidental to what's being tested and would make the assertion brittle
  to unrelated future changes). Each was confirmed to actually fail
  without its corresponding fix, not just pass trivially.
- **Two real bugs found and fixed along the way, worth knowing about if
  you touch this code:**
  1. Protecting only the `Incref`/`Decref` is NOT enough.
     `ir.Assign(dest=b, src=X)` is its own, independent textual read of
     `X` in the generated C (`b = X;`) - it does not reuse whatever value
     an adjacent `Incref`/`Decref` already touched. An earlier version of
     this implementation closed the read-side lock *before* emitting that
     Assign; under real concurrent load this let a writer swap the global
     in the gap between them, retaining one object while binding to a
     different one - confirmed via an actual crash, not reasoned about in
     the abstract. The fix: the Assign has to be inside the *same*
     critical section as the Incref/Decref, on both the read and write
     sides (see `lowering.py`'s `_cfg_assign`).
  2. `cfg.assign()`'s `is_alias` branch previously increfed
     unconditionally whenever `is_alias` was `True`, trusting the
     *source AST shape* (a bare Name/Attribute "looks aliasing") rather
     than what the expression actually *lowered to*. Once `_expr_Name`'s
     narrowed-read fix started handing back an already-owned, freshly-
     retained temp for exactly this shape, that unconditional incref
     would have silently double-owned it (a leak). Fixed by checking
     `cfg.py`'s own `is_fresh_temp(src)` first and taking the ownership-
     transfer path instead when true - which is also a general
     correctness improvement independent of this document's own
     mechanism, not just a narrow enabler for it.
- **Linux (`pthread_mutex_t`) support added in a follow-up pass, with NO
  changes needed to `cfg.py`/`lowering.py`/`ir.py`/`mpy_types.py` at
  all** - every `ir.AcquireGlobalLock`/`ir.ReleaseGlobalLock` marker is
  already platform-agnostic IR emitted the same way regardless of target;
  only `emitter_c.py` (which already decides, per-target, whether a
  marker becomes a real op or a no-op) needed changes: `_global_lock_
  supported()` widened to include `'linux'`, `_global_lock_acquire`/
  `_global_lock_release` branch to `pthread_mutex_lock`/`_unlock` there,
  the per-global lock's own storage/init differs (see above), and
  `<pthread.h>`/`-lpthread` get force-registered the same way kernel32's
  SRWLOCK exports are on Windows. This is the concrete payoff of this
  mechanism's own original design choice - deciding real-op-vs-no-op at
  EMISSION time, per target, off IR markers that never themselves know or
  care what platform they're compiled for.

**What's confirmed NOT yet covered - do not assume otherwise:**
- macOS - deliberately **poisoned, not silently unimplemented.**
  `_global_lock_supported()` now structurally supports macOS (it reuses
  Linux's exact `pthread_mutex_t` codegen via `_target_uses_pthread_lock()`
  - same storage, same `pthread_mutex_init`/`lock`/`unlock` calls, same
  `<pthread.h>`/`-lpthread` force-registration) but `assert False`s the
  instant it's actually invoked for a real macOS compile that needs the
  lock (i.e. only when a genuinely reassigned/protected global exists on
  that target - every call site is gated on `locked_globals` first, so an
  unrelated macOS compile with no protected globals is entirely
  unaffected - see `emitter_c_test.py`'s `MacosGlobalLockPoisonPillTests`,
  which checks both halves of that directly). This is a deliberately
  LOUDER failure mode than the earlier "silently return `False`, leave a
  real race unprotected with no signal" behavior - a compile that would
  have needed this mechanism on macOS now hard-fails instead of silently
  shipping unsafe output. Why poisoned instead of just enabled: this
  repo's own verified compiler/target matrix is Windows-x64 and Linux-x64
  only (no macOS machine in this dev environment) - the code path is
  believed correct by construction (it's the exact same pthread codegen
  already verified on Linux) but has never actually been compiled,
  linked, or run for real. Whoever next has real macOS hardware should
  delete the assert in `_global_lock_supported()`, run
  `thread_safe_globals_test.py`'s own stress tests for real against a
  macOS target, and update this section - not just delete the assert and
  assume.
- A.2's lock-free CAS publish path - not attempted; A.3's lock is used
  unconditionally for every protected global this pass covers.
- Narrowed reads of a narrowed *field* (`_expr_Attribute`'s own copy of
  the same rewrite) - deliberately out of scope, see above; that's Part B
  territory (a field belongs to an object, not a module).
- The write-once-at-init cost mitigations, `_protected`/`__private` field
  enforcement, and everything else in Part B - unimplemented, as
  originally scoped.
- `lib/datetime.py`'s `localtz()`/`lib/termcolor.py`'s `_codes()` still
  keep their own explicit `threading.FastLock` - this mechanism being
  solid now doesn't obligate removing a working, already-verified guard
  from shipped library code for its own sake; the mechanism's correctness
  for their exact shape is verified independently, via
  `thread_safe_globals_test.py`'s own narrowed-read stress test.

## Context

`lib/datetime.py`'s `localtz()` used to cache the system timezone via a bare
`if __localtz is None: __localtz = ZoneInfo()`, no lock. Under a real
thread-per-connection HTTP server (`lib/tcpserver.py`'s
`ThreadPerConnectionDispatcher`) hitting this on a fresh process with ~150
concurrent connections, every thread saw `None` at once, each independently
constructed its own `ZoneInfo`, and all raced to store into the same global
— a crash (`Illegal instruction`, zero output) confirmed via direct repro,
fixed by wrapping the whole check-then-set in a `threading.FastLock`
(commit `c45b255`, this worktree).

That fix is correct but narrow: it requires the library author to notice
the race and hand-write a lock. The same shape existed independently in
`lib/termcolor.py`'s `_codes()` — found by grepping for the pattern after
the first bug, not because anyone was looking for it. Two independent
authors wrote the identical bug the identical way. This document is about
closing the gap at the language level instead: make it structurally
impossible for this class of code to corrupt memory, whether or not the
author knew to reach for a lock.

## The soundness gap, precisely

`retain_object`/`release_object` (`emitter_c.py`'s `_PROLOGUE_RETAIN`/
`_PROLOGUE_RELEASE`) already use `atomic_fetch_add`/`atomic_fetch_sub` on
`ObjectHeader.ref_count`. That makes it safe for two threads to both hold a
pointer to the *same* object and both incref/decref it concurrently — but
it says nothing about the *variable or field slot* that holds the pointer
in the first place. A global or a field is an ordinary, non-atomic memory
location; reading and writing it from multiple threads with no
synchronization is a plain data race, independent of whether the objects
it ever points to are individually memory-safe.

This isn't hypothetical or specific to hand-written library code — the
compiler's *own* generated sequence for `global X; X = new_value` (an
RC-typed global) already has this exact shape today. `cfg.py:1758-1782`
(`assign()`, in the `if dest.is_global:` branch) emits, in order:

1. `Incref(new_value)` (if not already independently owned)
2. `Decref(X)` — reads `X`'s **current** value first, to release whatever
   was there before the overwrite (`cfg.py:1781`, comment: *"reads dest's
   CURRENT (pre-overwrite) value"*)
3. `Assign(dest=X, src=new_value)` — the actual store

Steps 2 and 3 are two separate, unsynchronized reads/writes of the same raw
C global. Two threads racing this sequence concurrently can double-decref
the same old object, or decref a value the other thread already
overwrote. This is a **general, already-present hazard for every RC-typed
global reassignment in the language**, not something specific to
`localtz()` — `localtz()` is simply the one instance that happened to get
hit hard enough, in a syscall-heavy enough construction path, to
manifest as a real crash.

## Prerequisite: field-visibility enforcement doesn't exist today — and must be added as part of this plan

Cost mitigation #2 (below, the `__private`-field write-once exemption)
reasons that a double-underscore field's assignments are provably
confined to methods textually inside its own defining class, per
`SYNTAX.md:132`'s documented visibility rule. That reasoning is only
sound if the compiler actually *enforces* the rule. It does not, today —
confirmed directly with a minimal repro:

```python
class Box:
	__secret: i32
	def __init__( self ) -> None:
		self.__secret = 42

def main() -> i32:
	b: Box = Box()
	b.__secret = 99          # SYNTAX.md says this should be a compile error
	return b.__secret
```

compiles and runs cleanly — the built executable exits `99`, not `42`.
Nothing in `lowering.py`'s attribute-resolution path checks visibility at
all for an ordinary `GetAttr`/`SetAttr`. The only existing enforcement of
`SYNTAX.md`'s privacy rules is narrowly scoped to `Class.__allocate__()`
calls (`lowering.py:11698`, via `Type.in_private_scope()`,
`mpy_types.py:178`) — a genuinely different code path from ordinary field
access, and one that doesn't generalize to `_protected` fields at all:
`in_private_scope`'s own docstring defines it as "`scope` IS this class
itself" (`mpy_types.py:178-183`) — same-class-only, with no "or a
subclass" variant for the protected tier.

This isn't a pre-existing bug this document happens to notice in
passing — **it's a hole this specific plan would open into a new
soundness gap if left unaddressed.** Cost mitigation #2 tells the
compiler "skip locking a `__private` field, because nothing outside its
own class can write to it" — but nothing outside its own class is
actually stopped from writing to it today. Ship that exemption without
also shipping the enforcement, and a `__field` its own author reasonably
assumes is write-once-after-`__init__` becomes an *unlocked*,
externally-writable race — reintroducing exactly the class of memory
corruption this whole document exists to close, just relocated from
"nobody thought to add a lock" to "the compiler assumed a privacy
boundary that nothing actually enforces."

**Enforcing `SYNTAX.md`'s `_protected`/`__private` field-access rules is
therefore a required, in-scope deliverable of this plan, not a
separately tracked nice-to-have — and it must land before Part B's
write-once exemption ships, not alongside it (the exemption is unsound
without it).** Concretely:

- Extend field-access resolution (wherever `obj.field`/`self.field`
  becomes an `ir.GetAttr`/`ir.SetAttr` — the same chokepoint Part B.3
  already needs to touch for locking) to reject, at compile time, any
  read *or* write of a `__private` field from a method whose own class
  isn't the field's defining class — reusing `Type.in_private_scope()`
  (`mpy_types.py:178`), the exact same check `__allocate__()` already
  performs, just applied to ordinary attribute access instead of one
  special-cased call form.
- Add the missing `_protected` counterpart — "is the accessing method's
  own class the defining class, or a (possibly indirect) subclass of
  it" — as a new sibling method on the same `ScopeMixin`/`Type` machinery
  `in_private_scope` already lives on, not a one-off check bolted onto
  the field-access call site.
- Audit `lib/` for any code that (knowingly or not) already reaches past
  a `_`/`__` boundary the way the repro above does, before turning this
  on — enforcing a rule that was previously unenforced can break existing
  callers; this needs its own pass over the standard library, not an
  assumption that nothing relies on the current gap.

**Enforcement is not itself a thread-safety fix, and must not be read as
one.** It narrows *who can write a field* to "this class's own methods" —
it says nothing about what happens when two of that class's own methods
run concurrently, on the same instance, on two different threads. A
`__private` field mutated by more than one method is exactly as exposed
to a race as a public one unless something actually locks the access:
`__localtz` and `_color_codes` (the two real bugs that motivated this
whole document) were already private-by-convention and that alone did
nothing to protect them — `FastLock` did. Enforcement only pays for one
narrow thing in this plan: making cost mitigation #2 below (the
write-once-after-`__init__` exemption, which applies *only* to a field
never reassigned outside `__init__`) a sound compile-time check instead of
an unsound guess. Any private field written by more than one method still
needs, and — once Part B ships — automatically gets, the exact same
per-object lock a public or protected field gets; visibility tier and
"is this access thread-safe" are orthogonal questions in this design,
before and after enforcement lands.

## Design goal (the bar this proposal has to clear)

Naive, unsynchronized-looking code — `if self.x is None: self.x = Foo()`,
a bare `global X; X = Y()`, `self.count += 1` from two threads — must never
be able to corrupt memory, double-free, or read freed memory, no matter how
many threads touch it concurrently. It **may** still do redundant work
(construct-and-discard extra objects) or produce a logically-surprising
result (a lost update on a `+=`) — those are accepted, explicit
consequences of the "no interprocedural escape analysis" decision below,
not soundness bugs. The bar is *memory safety*, not *linearizability of
arbitrary user logic* — that stays `FastLock`'s/`Lazy[T]`'s job (see
"Relationship to `Lazy[T]`" below).

## Scope

**In scope:**
- Module-level global variables of RC type (`Variable.is_global == True`
  and `variable.type.is_rc()`, per `mpy_types.py:80-87`).
- Instance fields of RC type (`ir.GetAttr`/`ir.SetAttr` targets whose
  `attr.type.is_rc()`).
- Plain scalar globals/fields (`i32`, `bool`, `f64`, …) — lighter
  treatment, not a full lock (Part C).

**Confirmed out of scope — no work needed:**
- **Class-level "shared across every instance" variables do not exist in
  this compiler.** Investigated directly: a class-body assignment
  (`class Foo: shared = None`) goes through the *same* discovery-time
  handler as any instance-attribute declaration
  (`discovery.py:1338-1364`/`1553-1630`) and is appended to
  `scope.attributes` — an ordinary per-instance field whose default is
  **re-evaluated fresh at every single construction**
  (`lowering.py:2271-2288`, `_emit_construction_defaults`), never a shared
  storage location. `emit_rcclass` (`emitter_c.py:3191-3220`) never emits
  a `static` slot for a class-body variable. `ClassName.attr` as an
  expression (the only way real Python would let code read/write a
  genuinely shared class variable) isn't even legal today — a bare class
  name used as a value hits `discovery.fail("... is not a value ...")`
  (`lowering.py:7111-7116`). There is nothing to fix here now; if real
  class-level shared storage is ever added later, it will need the
  identical treatment this document gives module globals.

**Deliberately out of scope:**
- Local variables and function parameters — live on one thread's own stack
  frame, never reachable from another thread except by first escaping
  through a global or a field (covered above).
- Atomicity of multi-statement/multi-field invariants — this proposal
  guarantees single-access memory safety only. A caller who needs "these
  two fields always change together" still needs an explicit `FastLock`.

## Design principle: lock the access, not the statement

Every individual global/field **read** and every individual global/field
**write** is its own, self-contained critical section — acquire, do the
one read-and-retain or decref-and-store, release. Nothing spans more than
one such access. This is the same granularity `localtz()`'s hand-written
fix uses for the write, generalized automatically to every access:

- A compound "check, then construct, then store" sequence (the
  `localtz()` shape) is **not** made atomic as a whole by this proposal —
  multiple threads can still both observe "not yet set" and both
  construct. That's the accepted "degrades to redundant work" outcome.
  What's no longer possible is a torn/racing view of the slot itself, or a
  read racing a free.
- This granularity also avoids the two failure modes a coarser design
  would introduce: locking a whole *method* would make ordinary method
  calls into critical sections users can't reason about (and risks
  deadlock on any reentrant/recursive access to the same object); locking
  a whole *process* (see "Alternatives rejected" below) defeats the actual
  point of this compiler's real-OS-thread model.

## Part A — module-level globals

### A.1 What needs a lock

Not every global — only ones that are (a) RC-typed and (b) ever the
`dest` of an `ir.Assign` from inside a function body (i.e. genuinely
reassigned via `global X; X = ...`, not just initialized once at module
load). A global that's only ever written by its own module-level
initializer (`compiler.py:20-23`'s `LoweredGlobal.instructions`) is
provably single-write, happening before any thread the program spawns
even exists — no Part A lock needed **for the slot itself**. Confirmed
directly against real generated C for exactly this shape (a global with a
non-trivial, heap-allocating constructor, mirroring `reactor.py`'s
`_current_worker: threading.ThreadLocal[Worker] = threading.ThreadLocal[Worker]()`):
its constructor call (`ThreadLocal.__init__`, including the `TlsAlloc()`
syscall) is emitted into its own `__metalpy_init_<qualname>()` function,
called unconditionally from `__metalpy_init()`, itself called at the very
top of `main()` — before any user code runs and before any thread the
program could spawn exists. So `_current_worker` (the *slot*) genuinely
is single-write, exactly as claimed.

**That is not the same claim as "no lock needed, full stop," and this
document must not be read that way.** `_current_worker`/`_thread_deadline`
are safe to use concurrently for a completely different reason that has
nothing to do with being written once: the *slot* holds a reference to a
`ThreadLocal[Worker]` object whose own per-thread payload is set
continuously, throughout the program's whole life, from many different
threads (every `Worker.run_until_idle()` call sets it, on whichever
thread is driving that worker) — not once at startup. That's safe purely
because `TlsGetValue`/`TlsSetValue` (Windows) and
`pthread_getspecific`/`pthread_setspecific` (POSIX) are OS-guaranteed to
be per-calling-thread: no two threads can ever race on the same
underlying storage through these calls, by definition of what thread-
local storage means — a property this document's locking scheme
contributes nothing to and takes no credit for. `socket.py`'s
`_wsa_state`/`_wsa_error` are a cleaner example of the *intended* "no
lock needed" case (single-write slot, and the `Atomic[i32]` payload it
holds is *also* safe on its own terms, already out of scope here) —
listing `_current_worker` alongside it without this caveat conflates two
different safety arguments and risks implying "reachable through a
write-once global" is a general license to skip protecting whatever's on
the other end of that reference, which is false: an ordinary *mutable*
RC object (not one with TLS's own special per-thread isolation) sitting
behind a write-once global still needs Part B's protection for its own
fields, exactly as if it were reached any other way.

This "was this `Variable` ever an `ir.Assign` dest outside its own init
instructions" check does not exist today (confirmed: `_stmt_Global` is a
no-op, `lowering.py:2951-2960`; nothing currently records reassignment
anywhere), but is a straightforward new walk over every lowered
`Function`'s instructions — directly modeled on the *existing*
`_referenced_global_qualnames`/`_transitive_global_reads_by_function`
walkers `emitter_c.py:3799-3957` already uses for global-init dependency
ordering (a `dataclasses.fields()`-driven generic instruction walk), just
checking `ir.Assign.dest is <that Variable>` instead of arbitrary operand
reads.

A third category sits between "written once at init" and "written
repeatedly" and deserves its own treatment, not the same lock the
repeatedly-written case needs: a global written **at most once at
runtime**, via the `if X is None: X = compute()` shape — exactly
`localtz()`'s own bug. Once such a global's first (and only) real write
lands, it never changes again — a materially stronger guarantee than
"reassigned somewhere," and one that admits a genuinely lock-free
mechanism instead of a lock. See A.2 below.

### A.2 The write-once-at-runtime case: a lock-free CAS publish, not a lock

For a global whose only writer is a single `X = compute()` guarded by
`if X is None:` (or any provably-single-winner publish, more generally),
a plain `compare_exchange(expected=NULL, desired=new_value)` on the slot
— no separate lock object at all — is not just viable, it's **strictly
better** than A.3/A.4's lock: it removes the SRWLOCK/`pthread_mutex_t`
memory cost and the POSIX init-order asymmetry (A.3) entirely, and it's
genuinely lock-free rather than merely uncontended-fast. It's sound for a
reason the general repeatedly-written case (A.3/A.4) does not share:
because the slot never changes again after the winning CAS, no reader can
ever race a `retain_object()` against a writer's decref-to-zero-and-free
of the value it just loaded — there's no writer left to race against.
Every losing thread's own redundantly-constructed value is simply
`release_object`'d by that thread itself, never having been visible to
anyone else — the same "degrades to redundant work, never corrupts"
outcome this whole document is built around, achieved here with no lock
at all.

**A concrete representation blocker, confirmed directly, not assumed:**
`ZoneInfo|None` — `localtz()`'s own `__localtz` global's actual declared
type — compiles today to a real tagged struct
(`struct $__u$$intrinsics$NoneType$$...$ZoneInfo { u8 tag; union { ... }
data; }`, confirmed via `-c` inspection of real generated C for the
identical shape: a bare `Box|None` global emits `.tag`/`.data` field
accesses, not a single pointer read), not a bare nullable pointer. A
single machine-word CAS can't atomically swap a multi-field struct. This
is exactly the gap `PLAN_NULLABLE_POINTER_UNION.md` (already in this
repo, unimplemented) proposes closing — a tagless, bare-pointer
representation for a `T|None` union whose one non-`None` member is
pointer-shaped — but that document explicitly scopes itself to
`Ptr[T]`/`ConstPtr[T]` only (its own "Proposed semantics" section), not
to `RCClass|None`. Whether extending it to cover `RCClass|None` too is
easy (an RC reference is already represented as a bare, inherently-
nullable pointer in its own non-`Optional` form — see this document's own
earlier finding that a bare `Box`-typed global emits as a plain
`struct Box* name`) or has its own complications wasn't investigated here
and needs its own pass before this mechanism can ship. The alternative —
a double-width CAS (`cmpxchg16b` on x86-64 / GCC-clang's `_Atomic
__int128` / MSVC's `_InterlockedCompareExchange128`) operating directly
on today's tagged-struct representation — sidesteps needing that other
plan to land first, but its own portability across all three compilers
this repo requires (MSVC's intrinsic is a different API shape from GCC/
clang's `__int128`-based atomics, and this compiler's own `ir.AtomicRMW`/
`AtomicStore` codegen, `emitter_c.py:2906-2919`, wasn't checked for
whether it already supports a 16-byte atomic type at all) is unverified
and is a real open question (see "Open questions" below), not a given.

**Extending `PLAN_NULLABLE_POINTER_UNION.md` to `RCClass|None` is the
better route of the two, but it doesn't make CAS work for every `T|None`
shape — worth being precise about the boundary rather than overclaiming
it.** It closes the gap for exactly the shape that caused the motivating
bug and is almost certainly the common case in practice (`SomeClass|None`
— one non-`None`, pointer-shaped member) by making `None` a real `(void*)
0`, no tag at all — genuinely the more foundational fix, and worth landing
independent of this document, not just as a means to unlock A.2 (see
that other plan's own "Recommendation" section, which already reaches the
same "do this as its own dedicated pass" conclusion for unrelated
reasons: extern-boundary bridging, unconditional tag-byte cost). But it
stays a two-tier representation, not a universal one: a union with *more
than one* non-`None` member (`A|B|None`) has no spare bit pattern to
self-encode "is this None" and stays a genuine tagged struct regardless
of this change (`PLAN_NULLABLE_POINTER_UNION.md`'s own "Proposed
semantics" section is explicit that this is unaffected) — A.2's lock-free
publish only ever applies to a global/field whose type is the single-
pointer-payload shape in the first place, so a `Worker|Reactor|None`-typed
global would still need A.3's lock (or A.2's mechanism only kicks in once
that specific global's own type qualifies, checked per-global, not
assumed globally). Scalar payloads (`i32|None`) are also unaffected — no
natural null sentinel exists for a plain scalar, a separate, still-open
question `PLAN_NULLABLE_POINTER_UNION.md` itself declines to resolve.

**Detection is also its own, separate question from A.1's.** A.1's
"was this ever an `ir.Assign` dest" walk finds every reassignment; this
mechanism additionally needs to recognize the *specific*
"guarded-by-`is None`, single call site" shape (or prove a more general
"assigned at most once, ever, on any path" fact) to know it's safe to use
a lock-free publish instead of A.3/A.4's lock. Scope the first cut to the
easily-recognized syntactic shape (one assignment site, textually guarded
by `if <this var> is None:`) rather than attempting a fully general
once-only proof — the same "don't over-reach past what's cheaply
checkable" posture the rest of this document already takes (see A.1's
own `__private`-vs-`_protected`-vs-public cost tiers).

**The same idea applies symmetrically to Part B fields** — a `__private`
field lazily computed on first *access* (not just in `__init__`, e.g. a
memoizing getter) is the field-shaped version of this exact pattern, and
should get the same lock-free CAS publish once the representation
question above is resolved, rather than Part B's per-object lock.

**`Lazy[T]`/`Once[T]`'s own internal implementation (see "Relationship to
`Lazy[T]`/`Once[T]`" below) should be built on this CAS mechanism
directly, not on `FastLock`** — once the representation question is
resolved, `Lazy[T]` becomes the ergonomic wrapper around exactly this
lock-free publish, not around a mutex.

### A.3 The lock, for the genuinely-repeated-write case: a paired raw primitive, not a `FastLock`

**Why this case can't go lock-free the same way A.2 does — precisely,
not by assertion.** A.2's lock-free publish works because the value being
replaced is always `NULL` — nothing is ever freed by the publishing CAS,
so there's nothing for a concurrent reader to race. The moment a slot is
genuinely reassigned more than once, that stops being true, and a bare
CAS reopens a real gap even with the most natural-looking fix applied.
Concretely: a reader confirming "the slot still holds `X`" via its own
`CAS(&slot, X, X)` immediately before increffing does **not** help —
that confirmation is a fact about the past instant it executed, not a
reservation. There is still a real window between the confirming CAS
succeeding and the reader's subsequent `retain_object(X)` call, and a
writer's entire `CAS(&slot, X, Y)` + `decref(X)` sequence fits inside
that window on real hardware (the reader can be preempted between the
two steps for any duration):

```
slot = X, refcount(X) == 1, owned solely by the slot

Reader R                                Writer W
--------                                --------
R1: ptr = atomic_load(&slot)  → X
R2: CAS(&slot, X, X) succeeds
    (proves "slot == X held at
    this instant" — nothing more)
                                         W1: old = atomic_load(&slot) → X
                                         W2: CAS(&slot, X, Y) succeeds
                                         W3: decref(X) → refcount 1→0
                                             → X is freed
R3: incref(ptr)  →  ptr points at
    memory that no longer exists
```

The structural reason no amount of looping on `slot` alone can close
this: `retain_object` necessarily dereferences `ptr` to touch
`ptr->ref_count`, a *different* memory location than `slot`, and no
atomic operation on `slot` can make a later dereference of `ptr` safe —
the two locations aren't linked. Closing this lock-free needs one of two
genuinely different mechanisms, not a cleverer loop:

- **Defer the free** (hazard pointers / epoch-based reclamation) — a
  writer's decref-to-zero doesn't call the destructor immediately; it
  waits until no reader could still be mid-load.
- **Make the reservation and the pointer read one atomic unit** — pack a
  "readers currently in flight" count *into the same word as the pointer
  itself*, so a reader's "reserve" is a single atomic RMW purely on
  `slot`'s own combined `[pointer, count]` value, never dereferencing
  `ptr` until after the reservation succeeds; the writer must wait for
  that count to reach zero before it's allowed to actually free. This
  needs a double-width CAS (`cmpxchg16b`/`_InterlockedCompareExchange128`,
  128 bits) specifically because the pointer and the reservation count
  have to live in one atomically-swappable unit — this is the actual,
  structural reason wide CAS keeps coming up for this problem, not an
  arbitrary implementation choice. Given this document already needs to
  resolve a representation question for A.2's `RCClass|None` case (see
  A.2 above), a future revision of A.3 built on split reference counting
  is a legitimate lock-free upgrade path once that representation work
  lands — recorded here as a real option, not attempted in this pass.
  **Real availability caveat, not just a compiler-support checkbox:**
  the x86-64 hardware side is close to a non-issue for this repo's actual
  target matrix (`CMPXCHG16B` has been baseline on mainstream x86-64 CPUs
  for a long time, and this compiler targets only Windows-x64/Linux-x64
  across all three verified compilers today, no ARM) — but GCC/Clang
  don't emit `cmpxchg16b` for a 16-byte atomic *by default*; without an
  explicit `-mcx16` compile flag, both silently fall back to a
  libatomic-backed, address-keyed **lock** for 128-bit atomics — i.e. the
  exact thing this mechanism exists to avoid, reintroduced silently by a
  missing build flag, with worse overhead than just using a real mutex
  directly. MSVC's `_InterlockedCompareExchange128` has no equivalent
  missing-flag trap but does require 16-byte alignment of its operand — a
  real layout constraint on the `[pointer, count]` packing, not just a
  performance nicety. None of this has been checked against this
  compiler's own `ir.AtomicRMW`/`AtomicStore` codegen or build flags
  (`linker_c.py`) — confirm `-mcx16` is actually wired into the
  clang/gcc build recipe (and confirm it doesn't regress anything else)
  before trusting a double-width CAS is genuinely lock-free rather than
  quietly locked, on every one of the three compilers this repo requires.

Absent one of those, A.3 uses an ordinary lock. `FastLock`
(`lib/threading.py:35-150`) is itself an `RCClass` — heap
allocated via `sys.alloc`, with its inner OS lock *also* heap-allocated
inside `__init__`. Using one `FastLock` per protected global would mean
two nested heap allocations happening as part of module-global
initialization, before the very system this document is trying to make
safe is itself safe to construct. Don't reuse `FastLock`; emit a bare,
paired synchronization primitive per protected global instead, following
`FastLock`'s own per-platform `LockOpaque` choice
(`lib/threading.py:19-32`: Windows `_SRWLOCK`, POSIX `pthread_mutex_t`).

This is cheaper than it sounds, and mostly falls out of existing
machinery:

- **Windows**: `_SRWLOCK` is a plain `@cstruct` (`lib/windows/kernel32.py`)
  whose all-zero state is already a valid, unlocked SRWLOCK per Win32's
  own documented contract (no `InitializeSRWLock` call exists or is used
  anywhere in this codebase). A module-scope `_lockN: _SRWLOCK` (default
  all-zero-Const construction) hits `emitter_c.py:3745-3789`'s **existing**
  `_global_init_is_all_zero_value_type` fast path — a plain
  `TYPE name = {0};` file-scope declaration with **no init-function call
  at all** (`emitter_c.py:3959-3978`, excluded from the topological
  global-init graph at `emitter_c.py:3920-3923`). This needs zero new
  emitter machinery on Windows — the fast path is already there, just
  unused for this purpose today.
- **POSIX**: `pthread_mutex_t` is *not* a `@cstruct` (it's an opaque
  `compiler.c_type(...)`, `lib/threading.py:25`), and unlike SRWLOCK,
  zero-initializing a `pthread_mutex_t` is not a portable guarantee (glibc
  happens to tolerate it; `FastLock`'s own POSIX `__init__` calls
  `pthread_mutex_init()` explicitly rather than relying on zero-init,
  `lib/threading.py:49-62`). A raw per-global POSIX lock needs to go
  through the *general* callable-global-init path
  (`_emit_global_init_fn`/`_topologically_sort_globals`,
  `emitter_c.py:3980-4010`) so `__metalpy_init()` genuinely calls
  `pthread_mutex_init()` once. This is a real, asymmetric extra cost on
  POSIX that doesn't exist on Windows — flagged as an open question below
  (a lighter, portably-zero-init-safe primitive, e.g. a raw futex-based
  spinlock, may be worth building instead of reusing `pthread_mutex_t`
  specifically for this purpose).

  **Status update:** implemented, but simpler than this bullet's own
  speculation - the `pthread_mutex_init()` call is NOT routed through the
  general `_emit_global_init_fn`/`_topologically_sort_globals` machinery
  (that machinery exists to order one metalpy-level global's initializer
  against another's, keyed off a real `Variable`/`LoweredGlobal` - these
  locks have neither, same as the Windows `SRWLOCK` case just above).
  Instead, every protected global's `pthread_mutex_init()` call is emitted
  directly into the synthesized `__metalpy_init()` function body's own
  text, unconditionally ahead of the topologically-sorted global-init
  calls - safe with no ordering analysis needed at all. **Correction:** a
  global's own init instructions CAN now contain
  `ir.AcquireGlobalLock`/`ReleaseGlobalLock` markers too (see the Status
  section's update above - a global's own initializing write needs the
  same protection an ordinary reassignment does, since its own initializer
  can spawn a thread that reassigns it concurrently). The ordering claim
  here stays true regardless, for a different reason than originally
  stated: every lock's `pthread_mutex_init()` runs first, unconditionally,
  before *any* global initializer runs (this global's own included) - so
  by the time a global's own init function can reach its
  `AcquireGlobalLock`, the lock it acquires is already initialized, same
  as for a later ordinary-function reassignment.

### A.4 Where to insert acquire/release (the A.3 lock case)

`cfg.py:1758-1782`'s existing `if dest.is_global:` branch is the exact,
already-present hook: it already knows it's building the special RC
lifecycle sequence for a global write; wrap steps 2-3 (the "decref old,
overwrite" pair) in `acquire(lockN)` / `release(lockN)`. The read side
(`_emit_operand`'s `Variable`/`is_global` branch, `emitter_c.py:1375-1378`,
reached from every place a global appears as an `Operand`) needs the
paired read-side wrap: acquire, load the pointer, `retain_object`, release,
then use the now-independently-owned value. Because `Incref`/`Decref` take
a generic `Operand` with no special-casing in `ir.py` itself (confirmed:
`ir.py:561-579`), and because a global operand already carries `.is_global`
all the way through instruction selection, this is a matter of teaching
`cfg.py`'s existing global-aware branch and `emitter_c.py`'s existing
`_emit_operand` chokepoint to consult the lock table built in A.1, not
inventing new IR.

## Part B — instance fields

### B.1 One lock per object, not one per field

Embed the lock in `ObjectHeader` itself (`emitter_c.py`'s
`_PROLOGUE_HEADER`, currently `_Atomic int32_t ref_count` +
`const __metalpy_ObjectVtbl* vtable`), shared across every field of that
object. A lock per *field* would mean N locks per object for an N-field
class — more memory, and no real concurrency win, since most contended
cases involve one thread mutating several fields of the same object in
sequence anyway (worse: per-field locks reintroduce the exact
lock-ordering hazard a single per-object lock avoids for same-object
multi-field access). One lock per object also composes with A: every
`ObjectHeader`-bearing global already gets its own dedicated lock from
Part A, so there's no need to *also* embed a lock inside the pointee for
the global case — Part A and Part B protect two different kinds of slot
(the variable/field storage location vs. the object's own fields), not
the same thing twice.

### B.2 Memory cost — the real open question

A Windows `SRWLOCK` is a single pointer-sized zero-init word (per A.3) —
adding one to every `ObjectHeader` is close to free (12-16 bytes today →
20-24). A POSIX `pthread_mutex_t` is up to 40 bytes on glibc — adding
*that* to every single RC object in the language is a large, blanket
memory-size regression, not just a POSIX/Windows asymmetry in
initialization cost (A.3) but now in **steady-state object size** for
every object in the language, always, regardless of whether it's ever
touched by more than one thread. This is the single biggest open decision
in this whole proposal (see "Open questions").

### B.3 Where to insert acquire/release

`ir.GetAttr`/`ir.SetAttr` (`emitter_c.py:2817-2822`) are the sole IR
shapes for field read/write — a small, single chokepoint, structurally
identical in shape to Part A's global chokepoints, just unconditional
today (no `is_global`-style branch exists for fields at all; every
RCClass instance goes through the identical zero-synchronization codegen
regardless of sharing). Wrap each `GetAttr` (for an RC-typed field) in
acquire-the-receiver's-header-lock / read+retain / release; each `SetAttr`
in acquire / decref-old+store-new / release — the same read/write shape as
Part A, just keyed off `instr.obj`'s own embedded lock instead of a
per-global static. This applies uniformly to every RC-typed field
regardless of its visibility tier (public, `_protected`, `__private`) —
visibility controls who can reach a field, not whether concurrent access
to it needs a lock, and those are independent questions (see "Cost
mitigations" #2 below for the one narrow, visibility-dependent exemption:
a field proven never reassigned outside `__init__`).

### B.4 Reentrancy hazard, specific to fields (not globals)

Because field access can appear inside a method that's already executing
*on* the object whose lock it's about to (re-)acquire — e.g. a method that
calls another method on `self`, or a getter called from within a setter —
a naive per-access exclusive lock risks a same-thread deadlock the moment
two accesses to the same object's fields nest inside one call stack
(SRWLOCK and `pthread_mutex_t`'s default type are both **non-reentrant**;
`FastLock` inherits that). Since this proposal locks single
*accesses* (B.3), not whole *methods*, most ordinary code is fine — the
lock is held only for the duration of one field read or write, released
immediately, not across the call into another method. But a getter that
returns `self.field` while a caller already holds this exact object's
lock from an *enclosing* GetAttr/SetAttr sequence being emitted
inline/optimized in a way that widens the critical section would be a
real hazard to rule out explicitly during implementation, not assumed
away — flagged as a required verification item, not resolved here.

## Part C — scalar (non-RC) globals and fields

A plain scalar global/field write is a single store with no companion
retain/release bookkeeping — the read-then-retain-vs-write-then-free
hazard that motivates a full lock for RC types doesn't apply. This
codebase already has the right-sized primitive: `atomic.Atomic[T]`
(`lib/atomic.py`) and the underlying `ir.AtomicLoad`/`ir.AtomicStore`/
`ir.AtomicRMW` instructions (`emitter_c.py:2906-2919`). Route scalar
global/field access through an atomic load/store instead of the Part
A/B lock — cheaper, and sufficient: it rules out torn reads/writes and
compiler-reordering UB, which is all that's needed once there's no
refcount to keep consistent. (A `+=` on a scalar field from two threads
can still race to a lost update — same accepted "degrades to a wrong
but non-corrupting result" bar as the RC case, not a memory-safety
issue.)

## Cost mitigations

1. **Whole-program on/off switch — the biggest lever, and the practical
   answer to "escape analysis is impossible".** Precise per-object escape
   analysis ("does *this* object ever reach a second thread") needs
   interprocedural reasoning this compiler doesn't have and, per this
   document's own conclusion (echoing the discussion that produced it),
   isn't worth building. But a much coarser, **whole-program** question is
   both decidable and cheap: *does this program construct a
   `threading.Thread` anywhere at all, reachable from `main()`?* If not,
   nothing can race, full stop — by definition, not by analysis of any
   individual object. If the compiler's own discovery/scheduling never
   reaches a `Thread.__init__` call site, skip emitting *all* of Part
   A/B/C's locking machinery for the whole program, falling back to
   today's raw codegen everywhere. This is coarse (a program that spawns
   one thread anywhere pays the cost everywhere, not just near that
   thread) but sound, requires no new "does this escape" reasoning at
   all, and directly protects the (likely still-common) single-threaded
   program from paying anything. Confirmed today `Thread`/`CreateThread`/
   `pthread_create` are ordinary opaque `@extern` FFI with zero
   compiler-recognized syntax (`lib/threading.py:174-201`) — this switch
   would need the compiler to specifically recognize a construction of
   `lib/threading.py`'s own `Thread` class (a new, narrow, one-off
   special case, not a general escape-analysis feature).

   **Status update: implemented and merged, with a revised detection
   design.** Rather than recognizing `Thread` construction, a new optional
   `@extern(..., spawns_thread=True)` decorator parameter
   (`discovery.py`'s `_parse_extern_decorator`) tags the actual OS-thread-
   creation syscall boundary itself — `posix.pthread.pthread_create()` and
   `windows.kernel32.CreateThread()`. This is strictly more robust than
   class-construction detection: it catches thread creation through *any*
   path (not just `lib/threading.py`'s own `Thread` wrapper — `ThreadPool`,
   `lib/reactor.py`, `lib/tcpserver.py`'s dispatcher all bottom out at
   these same two syscalls, so no extra per-wrapper detection code is ever
   needed), which also closes Open Question #5 below for the *detection*
   half (a raw `@extern` binding to either syscall is caught the same
   way a user's own hand-written binding to them would be, if one
   existed) — though the escape hatch for a thread reached some *other*
   way (a signal handler, an externally-invoked C callback) is still a
   real, separate need, and is shipped alongside (see below).
   `Compiler.spawns_threads` (a new field, `compiler.py`) flips
   incrementally the moment such a function is actually reached and
   lowered — the exact same pattern already used for `requires_crt`
   (`compiler.py`'s function-lowering branch, right beside
   `if unit.requires_crt: self.requires_crt = True`) — so no whole-program
   scan is needed at all, unlike `has_object_header_alloc`'s own scan
   pattern (which was considered and rejected as unnecessary overhead
   here, since the incremental flip is strictly cheaper and simpler).
   `emitter_c.py`'s `emit_c()` reads `compiler.spawns_threads` into a new
   `_program_uses_threads` module global (mirroring `_target_os`'s own
   established precedent exactly), which then gates the real-vs-no-op
   decision for every `AcquireGlobalLock`/`ReleaseGlobalLock`/
   `AcquireFieldLock`/`ReleaseFieldLock` marker, the per-object and
   per-global `pthread_mutex_init()` call sites, and the SRWLOCK/
   `pthread.h` forward-declaration registration - the identical
   emission-time-decided pattern `_global_lock_supported()` already uses,
   just with one more condition ANDed in. `mpy.py` ships a
   `--assume-threaded` escape hatch (Open Question #5's own remaining
   half) that sets `compiler.spawns_threads = True` directly before
   `emit_c()` runs, for a program that reaches a second OS thread some
   way this compiler can't see. Confirmed load-bearing via a real
   sabotage test (forcing `_program_uses_threads` off unconditionally
   reproduced a real `STATUS_ILLEGAL_INSTRUCTION` crash in 15/15 runs of
   the existing concurrent stress tests, restored immediately after
   confirming that) and new `thread_detection_test.py` coverage (no lock
   codegen at all for a program that never spawns a thread, real codegen
   for one that does, transitive detection through `ThreadPool`, the
   override flag). Full 3-compiler suite clean.

   **A related item (Cost mitigation #5 in the original numbering below,
   "Part B's retain-on-read overhead") was attempted in the SAME pass and
   REVERTED after a real double-free bug was found.** The idea: tag the
   specific `Incref`/`Decref` instructions Part B's own retain-on-read
   protection introduced (e.g. `_expr_Attribute`'s field-read retain) and
   elide them too when `_program_uses_threads` is false, the same way the
   lock markers are elided. This is UNSOUND as designed: `cfg.py`'s
   `fresh_temp()`/`is_fresh_temp()` bookkeeping (which lets a later
   binding-time incref be SKIPPED because the value is "already fresh")
   is decided at LOWERING time, permanently baked into the instruction
   stream by the time emission-time elision would try to also skip the
   retain-on-read Incref itself — eliding the Incref while the downstream
   skip-because-already-fresh decision still stands desyncs the two,
   producing an unbalanced decref at scope exit (a real, reproduced
   premature free, not a theoretical concern). Fully reverted (`ir.py`'s
   `Incref`/`Decref` have no `retain_on_read` field, `cfg.py`'s
   `fresh_temp`/`delete_temp`/`untrack_temp` are unchanged from their Part
   B shape). A future attempt would need to decide "does this program
   spawn threads" at LOWERING time (a cheap syntactic pre-pass over the
   AST for `Thread`/`ThreadPool`/etc. constructions, accepting some
   false-positive imprecision, rather than the current exact-but-only-
   known-at-emission-time incremental flip) so lowering.py itself can
   choose, per-program, whether to emit the retain-on-read Incref/
   fresh_temp pair AT ALL - not something to reattempt with the current
   emission-time-only architecture.

   **Status update: investigated (design-only, no code), recommendation is
   to NOT build this.** The lowering-time pre-pass would have to run and
   produce a final answer before `Compiler.run()` enqueues `main`
   (`compiler.py`, before `_drain()` starts) - the existing `@extern(
   spawns_thread=True)` mechanism can't supply that answer this early,
   since `Compiler.spawns_threads` is only flipped incrementally as
   thread-spawning functions are actually reached and lowered
   (`compiler.py`'s `_lower()`, mirroring `requires_crt`) and is only FINAL
   once the whole reachable call graph has drained - useless as an input
   to a per-instruction lowering-time gate. A separate syntactic pre-pass
   would need its own eager, whole-file AST walk descending into nested
   function/lambda bodies (`Discovery.visit_FunctionDef` only registers
   signatures eagerly, not bodies - a real thread-spawning import can sit
   inside a method, e.g. `lib/threading.py`'s own `Thread.__init__`) plus
   its own conservative reimplementation of `discovery.py`'s import-graph
   resolution to know which files are even reachable - a second, permanent,
   must-stay-conservative-forever detection mechanism, entirely separate
   from and unable to reuse the one already shipped. Worse, to stay sound
   against aliased imports (`from threading import Thread as T`),
   cross-module reexports, and closure/function-pointer indirection (the
   language has first-class closures), it cannot do call-site matching -
   it would have to degrade to "any reachable file imports `threading`/
   `posix.pthread`/`windows.kernel32` at all," a real regression from the
   already-shipped `@extern(spawns_thread=True)` mechanism's own "catches
   every wrapper for free via one syscall choke point" property. Given
   Stage 1 already eliminates the expensive part (lock acquisition,
   syscalls) for non-threaded programs, and what's left is only a couple of
   atomic incref/decref instructions, building a second detection
   mechanism - with a demonstrated history of causing a real double-free
   the one time this exact idea was tried - to shave that residual cost is
   not a good trade. **Not pursued further; revisit only if real profiling
   of an actual non-threaded program shows the residual retain-on-read
   traffic is a measurable hot-path cost, not speculatively.**
2. **Write-once-after-`__init__` exemption — only sound for `__private`
   fields, and only once the "Prerequisite" section above actually ships.**
   A field assigned only in `__init__` and never reassigned by any other
   method is safe to read lock-free forever after construction,
   since construction happens on a single thread before the object can
   have been published anywhere. But whether that check can stay a cheap,
   purely-syntactic, single-class-body scan depends entirely on this
   language's field-visibility tier (`SYNTAX.md:132`): a double-underscore
   `__field` is genuinely private — referenceable, and therefore
   assignable, *only* from methods textually inside that one class body
   (the same restriction `Class.__allocate__()` itself already enforces,
   `SYNTAX.md:559`) — so "never reassigned outside `__init__`" really is a
   one-class-body check there, no interprocedural reasoning needed. A
   single-underscore `_field` (protected) is visible to every subclass,
   which this whole-program AOT compiler *can* enumerate (it discovers the
   full class hierarchy reachable from `main()`), but "never reassigned"
   then means walking every method of every discovered subclass, not one
   class body — a real, bounded, but much bigger check than the private
   case, and one that needs to be redone (or invalidated) if a later
   compilation unit/subclass the checker didn't see could exist, which
   shouldn't happen for a true whole-program build but is worth confirming
   explicitly rather than assuming. A bare `field` (public) needs the
   equivalent check across literally every function the compiler
   discovers, not just a class family — the most expensive tier, and
   plausibly not worth building the exemption for at all versus just
   locking public fields unconditionally. Net: implement this exemption
   for `__private` fields first (cheap, unconditionally sound); treat
   `_protected`/public fields as a separate, larger follow-on decision,
   not the same "purely syntactic" claim. Whether the identical
   private/protected split applies to module *globals* too (i.e. can a
   double-underscore global only ever be reassigned via `global X` from
   within its own defining module, the way a private field can only be
   reassigned from within its own defining class?) wasn't confirmed during
   this investigation and needs checking before assuming A.1's global
   exemption is as unconditionally cheap as stated there.

   **Status update: checked, directly against the codebase. A.1's global
   exemption is already sound, for two independent reasons - not a gap.**
   First, the identical privacy split genuinely does apply to module
   globals: `Discovery.check_module_visibility` (`discovery.py:707-835`) is
   real, enforced infrastructure - it hard-errors the moment an accessing
   module differs from a `__`-prefixed global's defining module, wired into
   `visit_ImportFrom` and both attribute- and value-position resolution
   across `discovery.py`/`lowering.py`. Second, and more fundamentally,
   A.1's `reassigned_outside_init` flag is sound *independent* of whether
   that enforcement exists at all: this is a whole-program AOT compiler
   where each global is exactly one shared `Variable` object for the entire
   compilation (`mpy_types.py`), referenced (never cloned) by every
   importing module's scope. `cfg.py`'s `assign()` flips the flag on that
   one object's identity from every real write path (`_stmt_Assign`,
   `_stmt_AugAssign`, tuple/pattern/for-loop targets all funnel through
   it) - and the only way a different module's code can even reference `X`
   is `from module_a import X`, which binds that identical `Variable`
   object, not a copy. There is no disconnected second representation of
   "the same global" the flip could fail to reach. No counterexample
   program exists; this closes the open question with no follow-up fix
   needed.

   **Status update: implemented and merged, `__private` fields only, exactly
   the scope described above.** `ir.AcquireFieldLock`/`ReleaseFieldLock`
   gained a new `field: Variable|None` operand (the field's own declared
   `Variable`) so emission time can identify WHICH field a given critical
   section is for — every one of `lowering.py`'s ~11 construction sites now
   passes it through. `Variable` gained a new `field_reassigned_outside_init`
   flag (deliberately separate from `reassigned_outside_init`, since that
   one's `is_global`-scoped precondition and this one's `__private`/
   `__init__`-scoped precondition are different questions that happen to
   share a "provably single-writer" shape — conflating them would let one's
   flip silently satisfy the other's very different soundness requirement),
   flipped by `lowering.py`'s non-construction `SetAttr` branches in
   `_stmt_Assign`/`_stmt_AugAssign` (mirroring `cfg.py`'s `assign()` flipping
   `reassigned_outside_init` for a global) the moment a field is written from
   anywhere OTHER than `obj is self._construction_self` — a write to some
   OTHER already-published object, even from inside a *different* object's
   own `__init__`, still flips it, which is exactly the race this flag
   exists to catch. `emitter_c.py`'s new `_field_lock_exempt()` gates the
   `AcquireFieldLock`/`ReleaseFieldLock` no-op decision on `field.stem`
   being genuinely `__private` (leading `__`, not also trailing `__` — the
   same name-mangling test `discovery.check_field_visibility` already uses)
   AND `not field.field_reassigned_outside_init`. Confirmed load-bearing via
   a real sabotage test: forcing the exemption to always report `True`
   reproduced a real "double free/release detected" crash in a new stress
   test (`thread_safe_fields_test.py`'s
   `test_private_field_reassigned_outside_init_stress` — a `__private` field
   reassigned from another private method, not `__init__`), restored
   immediately after confirming that. New `thread_detection_test.py`
   coverage: a `__private` write-once-in-`__init__` field gets no lock
   codegen even in a program that DOES spawn a thread elsewhere (the
   per-field exemption is independent of Cost mitigation #1's whole-program
   flag), and a `__private` field reassigned outside `__init__` still gets
   real lock codegen (confirms the detector checks
   `field_reassigned_outside_init`, not just the `__` name shape). Full
   3-compiler suite clean, including with `METALPY_RUN_LOAD_TESTS=1`.
   `_protected`/public fields remain out of scope, as originally decided
   above — a `__private`-only exemption needed no interprocedural
   reasoning; that follow-on would.
3. **"Pull into a local" is already the idiom, for free.** A field/global
   read into a local already produces an independently owned, freshly
   increfed reference under this compiler's existing convention (verified
   directly: reading a module global into a caller-side local increfs it,
   confirmed via a real `compiler.refcount()` test during this
   investigation). A hot loop that would otherwise re-touch
   `self.field`/a global N times only needs to pay the lock once —
   `let snapshot = self.field` up front, then work off `snapshot` — no new
   language feature, just the existing binding idiom.
4. **Read-write lock semantics (open question, not committed).** Most
   accesses to most shared state are reads. SRWLOCK natively supports
   shared/exclusive acquisition (`AcquireSRWLockShared` alongside the
   exclusive calls `FastLock`/A.3 already use); a POSIX `pthread_rwlock_t`
   equivalent exists too, at the same `pthread_mutex_t`-vs-something-else
   sizing tradeoff as B.2. Worth prototyping once A/B land, not a
   precondition for landing them.

   **Status update: implemented and merged (Stage 4).** `ir.AcquireGlobalLock`/
   `ReleaseGlobalLock`/`AcquireFieldLock`/`ReleaseFieldLock` all gained a new
   `exclusive: bool = True` operand (default True - the SAFE choice for any
   call site not updated, since a write mistakenly marked exclusive is only
   slower, never unsound). Every read-side emission site (cfg.py's `assign()`
   own `is_alias` branch, `lowering.py`'s `_expr_Attribute`, the AugAssign
   "retain old" critical section, `compiler.decref(obj.field)`'s bypass
   lowering, the narrowed-union-global extraction) now passes
   `exclusive = False`; every write-side site (the global/field reassignment
   branches, the AugAssign replace/write critical section) keeps the default.
   Windows: `AcquireSRWLockShared`/`ReleaseSRWLockShared` (real Win32
   exports, natively supported by the same `SRWLOCK` A.3/B.2 already use) -
   close to free, exactly as this question predicted. POSIX: **not**
   `pthread_rwlock_t` (that would reintroduce the real-init-call,
   larger-than-a-plain-word cost Stage 3 just eliminated) - a hand-rolled RW
   spinlock on top of Stage 3's own `_Atomic uint32_t` word instead, bit 31
   as a writer flag, the low 31 bits as a live reader count, CAS-based both
   ways (`_PROLOGUE_POSIX_SPINLOCK` in `emitter_c.py`). No fairness
   guarantee either direction - accepted, matching this item's own
   "worth prototyping... not committed" framing, given how short every
   critical section this protects actually is. `acquire_field_lock`/
   `release_field_lock` (Part B's own one-place functions) gained a `bool
   exclusive` parameter rather than splitting into four separately-named
   functions - the C ternary `exclusive ? f() : g()` (both `void`) is valid
   standard C (C11 6.5.15p3), not a GNU extension.

   **B.4 reentrancy re-examination (required by this stage, not deferred):**
   confirmed against real generated C under real concurrent load, not just
   re-read comments - `thread_safe_fields_test.py`'s new
   `test_field_reentrancy_stress` exercises the exact hazard shape
   (`self.box.a = self.box.get_b()` - a setter whose own value expression
   calls a getter reading a DIFFERENT field on the SAME receiver) under 8
   writer + 16 reader threads, 2000 iterations each, with an explicit short
   timeout (15s, not this file's usual 30s) so a genuine self-deadlock would
   fail the test cleanly instead of hanging the suite. Passes clean - by
   construction, a critical section never spans more than one field access
   (B.3's own "lock the access, not the statement" design, unchanged by this
   stage), so no nested acquire on the same object's `$header.lock` was ever
   possible in the first place; this test is the confirming evidence, not a
   fix.

   **Sabotage-and-confirm:** forcing `__metalpy_spinlock_acquire_shared` to
   silently take no lock at all (while its own paired `_release_shared`
   still unconditionally decrements the reader count) corrupted the shared
   word's low bits into looking permanently writer-held after the first
   reader completed - not a crash, but a genuine, reproduced **hang**
   (`test_concurrent_read_write_stress`/`test_concurrent_field_read_write_
   stress` both timed out at their own 30s bound rather than completing),
   confirming the sabotage is load-bearing; restored immediately after
   confirming that. Full 3-compiler suite clean, including
   `METALPY_RUN_LOAD_TESTS=1` on WSL/gcc.

   With Stage 4 landed, all four items originally staged from this
   document's own "Cost mitigations"/"Open questions" sections are now
   implemented and merged (Cost mitigation #5/item 5's retain-on-read
   elision is the sole exception, investigated twice and explicitly
   NOT pursued - see its own status update above).

## Relationship to `Lazy[T]`/`Once[T]`

This document and the (separately proposed, not yet built) `threading.
Lazy[T]`/`Once[T]` primitive solve different problems and both remain
useful once this lands: this document guarantees a bare
`if x is None: x = Foo()` can't corrupt memory; it does **not** stop
multiple threads from redundantly constructing `Foo()` on first touch.
`Lazy[T]` is what avoids the redundant work, for the (common, but not
universal) "compute once, cache forever" shape specifically. Once Part A
ships, `Lazy[T]`'s own internal check-then-set becomes safe by
construction too — it's just no longer the *only* thing standing between
a lazy-init site and a crash.

## Alternatives considered and rejected

- **Precise per-object escape analysis** ("does this specific object ever
  become reachable from a second OS thread"). Rejected: needs genuinely
  new interprocedural reachability tracking through closures, RC
  assignment, and opaque `@extern` calls (`Thread`/`CreateThread` carry
  zero compiler-recognized semantics today — confirmed directly, see
  "Cost mitigations" #1 above) — a much bigger, more fragile feature than
  the blanket lock it would be trying to avoid, and still wouldn't help
  the actual bug class this document targets (module globals are
  unconditionally "escaped" by construction — there's nothing for an
  escape analysis to determine there). The whole-program on/off switch
  above (cost mitigation #1) captures the one coarse, sound, cheap version
  of "don't pay for single-threaded programs" that's actually buildable.
- **A single process-wide lock (GIL-style), matching CPython's own
  accepted tradeoff.** Rejected specifically *for this compiler*: the
  entire reason `lib/threading.py`'s real `Thread`, `lib/reactor.py`, and
  `lib/tcpserver.py`'s `ThreadPerConnectionDispatcher` exist is to give
  genuinely parallel OS threads — the demo this document's motivating bug
  came from depends on 150 connections truly running in parallel. A
  single global lock would silently take that away for every program that
  ever touches a shared global or field, which given `global` statements
  are rare (7 occurrences across all of `lib/`, vs. 2,330 `self.`
  accesses — see Part B's frequency evidence) but field access is
  everywhere, would in practice serialize nearly all real work. Per-object
  (Part B) and per-global (Part A) locks preserve inter-object parallelism
  instead — two threads each owning their own object never contend at
  all.

## Open questions (need an explicit decision before implementation)

1. **POSIX lock primitive for `ObjectHeader`** — embedding a full
   `pthread_mutex_t` (≤40 bytes on glibc) in every object is a much larger
   per-object cost than Windows' `SRWLOCK` (pointer-sized). Worth
   designing a lighter, portably-zero-init-safe primitive (a raw
   futex-based spinlock/mutex, sized like a plain `_Atomic int`) purpose-
   built for this, rather than reusing `pthread_mutex_t` as-is. This is
   the single biggest cost unknown in the whole proposal.

   **Status update: implemented and merged (Cost mitigation #3), a plain
   CAS spinlock, not a real futex.** `_Atomic uint32_t` (0 = unlocked, 1 =
   locked), covering BOTH Part A's per-global lock and Part B's per-object
   `$header.lock` uniformly (a single shared acquire/release pair,
   `_PROLOGUE_POSIX_SPINLOCK` in `emitter_c.py` - later extended into a real
   exclusive/shared pair each by Cost mitigation #4/Stage 4, see that
   status update below) - deliberately narrower than this question's own
   "futex-based" wording: the doc's own
   sizing target ("like a plain `_Atomic int`") is satisfied by a spinlock
   alone, `ir.AtomicCompareExchange` codegen was already wired end-to-end
   and tested (`lib/atomic.py`'s `Atomic[T]`) so this reuses existing
   machinery rather than adding new syscall-wrapper codegen a real
   `futex(2)` mutex would need, and the critical sections this protects are
   extremely short (one `GetAttr`/`SetAttr`-width access), where a
   spinlock's worst case (busy-wait instead of descheduling, mitigated with
   `sched_yield()` between attempts) matters far less than in the general
   case futexes are built for. A real futex remains a legitimate future
   upgrade if profiling under real contention ever shows spinning is a
   problem. Zero-init-safe exactly the way Windows' `SRWLOCK` already was
   (0 is a valid unlocked spinlock) - this REMOVES `pthread_mutex_init()`
   entirely on POSIX (both Part A's and Part B's call sites), collapsing
   A.3's own documented Windows/POSIX asymmetry rather than merely
   shrinking it, and drops the `-lpthread`/`<pthread.h>` force-link this
   mechanism used to need (a program that doesn't itself import
   `lib/posix/pthread.py`/`lib/threading.py` no longer links pthread at
   all just for Part A/B's own lock). Confirmed via a real compiled-and-run
   `compiler.sizeof()` check (`thread_safe_fields_test.py`'s
   `test_posix_spinlock_is_small`) and a sabotage test (a no-op spinlock
   acquire reproduced real double-free/corruption crashes in the existing
   Part A/B stress tests, restored after confirming that). Full 3-compiler
   suite clean, including `METALPY_RUN_LOAD_TESTS=1` on WSL/gcc (the
   primary leg for this POSIX-only change).
2. **Does the reentrancy hazard in B.4 actually occur** in real generated
   code once GetAttr/SetAttr access is genuinely single-instruction-wide,
   or only in a hypothetical inlined/optimized shape? Needs to be
   confirmed with real generated C, not assumed either way, before this
   is considered safe to ship.

   **Status update: confirmed against real generated C under real
   concurrent load - it does not occur, by construction.** See Cost
   mitigation #4's own status update above (`test_field_reentrancy_stress`)
   for the full writeup: a critical section never spans more than one field
   access (B.3's own design), so no nested acquire on the same object's
   `$header.lock` was ever reachable to begin with.
3. **Lock-ordering deadlock across two different globals/objects** — a
   function that touches global A then global B, racing against another
   thread's function that touches B then A, is a classic ordering
   deadlock independent of anything in this document (true of any
   fine-grained locking scheme). Worth an explicit statement of what
   guarantee (if any) this proposal makes here — likely "none, same as
   any other language with fine-grained locks," but say so.

   **Status update: stated explicitly, as requested - this proposal makes
   NO ordering guarantee.** Confirmed by construction, not just asserted:
   every critical section this mechanism ever opens is scoped to exactly
   ONE global/field access (Part A's own `Acquire.../decref/Assign/
   Release...` sequence, Part B's own "lock the access, not the statement"
   - B.3), so this document never itself holds two locks at once and
   cannot introduce a NEW ordering deadlock on its own. But it also does
   nothing to prevent one a program's own code constructs (two functions
   independently touching global A then B, vs. B then A, each under their
   own single-lock-at-a-time critical sections that can still interleave
   across statements) - same accepted risk as any other language exposing
   fine-grained locks (mutexes, `synchronized` blocks, etc.). Not pursued
   further; a real fix (lock ordering/deadlock detection) is a much larger,
   separate feature this document was never scoped to provide.
4. **Read-write lock semantics** (cost mitigation #4) — prototype and
   measure before committing either way.

   **Status update: implemented and merged - see Cost mitigation #4's own
   status update above for the full writeup** (Windows SRWLOCK shared/
   exclusive, a hand-rolled POSIX RW spinlock, the B.4 reentrancy
   re-examination this stage was also required to close).
5. **Whether the whole-program on/off switch (cost mitigation #1) should
   itself be user-overridable** (e.g. a compiler flag to force it on for
   a program that spawns threads via some path the compiler can't see,
   like a raw `@extern` callback the OS itself invokes on a new thread
   outside of `lib/threading.py`'s own `Thread` entirely) — flag as a
   real gap: this switch is sound only for the "threads are spawned via
   `lib/threading.py`'s `Thread`" path; any other way a program's C code
   could end up running on a second OS thread (a raw signal handler, an
   externally-provided callback invoked from a worker thread inside a
   linked C library) would silently bypass detection. Needs an explicit
   escape hatch/flag, not a silent assumption that `Thread` is the only
   door.

   **Status update: implemented and merged (Stage 1), shipped alongside
   the detection mechanism itself, not as a follow-up.** `mpy.py`'s
   `--assume-threaded` CLI flag sets `compiler.spawns_threads = True`
   directly before `emit_c()` runs, forcing real lock/retain codegen on
   even when no `@extern(spawns_thread=True)`-tagged function is ever
   reached - covered by `thread_detection_test.py`'s
   `test_assume_threaded_override_forces_lock_codegen`.
6. **`-mcx16` (or equivalent) must actually be wired into the GCC/Clang
   build recipe** before A.3's future double-width-CAS upgrade path (see
   A.3 above) can be trusted as genuinely lock-free on those two
   compilers — without it, both silently fall back to a libatomic-backed
   *lock* for 128-bit atomics, defeating the entire point. Not a blocker
   for anything landing in this pass (A.3 uses an ordinary lock today),
   but must be resolved, and verified against `linker_c.py`'s actual
   flag set on all three compilers, before that upgrade is ever attempted.

## Verification plan for any future attempt

1. Work in a fresh `EnterWorktree` worktree (never reuse this one or any
   other named one — see this repo's `CLAUDE.md`).
2. Resolve the open questions above explicitly before writing code,
   especially #1 (POSIX primitive) and #5 (detection escape hatch) —
   both change the shape of the implementation, not just its edges.
3. Before touching `cfg.py`/`emitter_c.py`: write a minimal repro program
   exercising the `Incref(new)/Decref(old)/Assign` global-write race
   described in "The soundness gap, precisely" above under real
   concurrent load (mirroring `datetime_test.py`'s
   `test_localtz_concurrent_init_stress`, this worktree) — confirm it's
   real and reproducible on today's codegen before changing anything,
   the same way this worktree's own investigation did for `localtz()`.
4. Land Part A (globals) first, independently verified, before Part B
   (fields) — Part A is the concrete, already-motivated piece; Part B is
   the larger, costlier generalization and should not block on or be
   entangled with Part A landing. Within Part B, land the "Prerequisite"
   section's `_protected`/`__private` field-access enforcement — including
   the standard-library audit it calls for — *before* the write-once
   exemption (cost mitigation #2); do not ship the exemption first "and
   add enforcement later," since that ordering leaves the exact unlocked
   gap the Prerequisite section describes, even if only briefly.
5. Regression-test the enforcement itself with a negative-compile case
   modeled directly on the repro in the "Prerequisite" section above (a
   `__private` field written from outside its own class must become a
   compile error, not silently succeed) — plus the equivalent case for
   `_protected` access from a non-subclass. Confirm the *existing*,
   narrower `__allocate__()` privacy check (`lowering.py:11698`) still
   passes unchanged, since the new field-access check reuses the same
   `Type.in_private_scope()` machinery and must not regress it.
6. Multi-compiler verification (MSVC, clang, WSL gcc) for every stage —
   this touches `emitter_c.py`'s core object/global emission directly,
   exactly the class of change that has previously shipped regressions
   caught only by a non-default compiler (see this repo's own memory
   notes on `linker_c.py`-adjacent compiler coverage). The POSIX-specific
   lock-primitive decision (open question #1) makes the WSL/gcc leg
   non-optional here, not just routine.
7. Run the full suite (`python tests.py`) after each stage, and loop it
   20-30x — this is exactly the kind of RC/refcounting-adjacent,
   concurrency-adjacent change where one green run proves nothing (see
   this repo's own memory notes on RC/concurrency changes).
8. Real concurrent-load regression coverage specifically for the
   `localtz()`/`termcolor._codes()` shape, migrated to rely on the new
   automatic protection instead of (or in addition to) their own
   hand-written `FastLock`, to prove the general mechanism actually
   subsumes the specific fix that motivated this document.
9. Commit, then merge into `master` via the shared-checkout exception in
   `CLAUDE.md`.
