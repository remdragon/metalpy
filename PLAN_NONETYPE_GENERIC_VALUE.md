# Scope: None/NoneType as a real generic value-type argument (dict[K, None], etc.)

## Context

`set[T]` (`lib/builtins/__set.py`) wraps `dict[T, bool]` internally, needing
some dummy value type for its backing `dict[K,V]`. `None` was the obvious
choice (`dict[T, None]`) but hit a real compile error, so `set[T]` shipped
using `bool` instead (works fine, costs one extra byte per stored element -
see that file's own header comment). This document scopes what it would
actually take to fix `None` as a generic value-type argument, investigated
as a follow-on phase of the broader set[T] work (alongside `in`/`not in`
operator support, `{...}` set literals, and set algebra - all landed on
`master`). Turned out to be three bugs deep, with the third one exposing a
genuine type-system ambiguity rather than a simple fix - this document picks
up where that investigation stopped.

## Fixed (or ready to land) - three real, isolated bugs found and confirmed

Each was independently verified via the full test suite
(`python tests.py`, 1146/1146, zero regressions) even without also fixing
the remaining blocker below - each is safe to keep/land on its own merits,
whether or not anyone ever finishes the rest of this.

**1. `lowering.py`'s `compiler.sizeof(...)` resolution used a truthy check,
not a `None` check.** `if size := getattr(target_type, 'sizeof', None):` -
`NoneType`'s own declared `sizeof` is a legitimate `0`, which is falsy, so
this wrongly fell through to a branch that only accepts
`RCClass`/`CStruct`/`CUnion`/`TaggedUnion`, failing with
`compiler.sizeof(NoneType) is not supported yet`. Fixed: `is not None`
instead of the truthy `:=` check.

**2. `NoneType.sizeof` (`discovery.py`'s `get_none_type()`) was declared
`0`, but `emitter_c.py` gives a `NoneType`-typed VALUE a real 1-byte C
representation** (`typedef unsigned char MetalpyNone;`,
`c_type()`'s `NoneType` branch) - distinct from a `-> None` function's
return, which really is a void with no storage. Once bug 1 was fixed alone,
`compiler.sizeof(None)` folded to `0` while the actual C storage was 1 byte
- any `sys.alloc[u8](compiler.sizeof(V))`-then-write call site (exactly
`UnsafeDict._store_value`/`_store_key`'s shape) would allocate a 0-byte
buffer and write 1 real byte into it: a genuine heap buffer overflow. Fixed:
`sizeof=1` on the `NoneType` singleton, matching `MetalpyNone`'s real size.
(`NoReturn` shares the same `sizeof=0` pattern but maps to real C `void` -
confirmed unaffected, left alone.)

**3. `ir.Call`'s C emission never checked whether the callee was
void-in-C before assigning its result to a dest temp.** This codebase
already has a `_returns_void_in_c()` helper (`emitter_c.py`) specifically
for "a generic method's declared return type resolves to `NoneType` for
THIS monomorphization" (e.g. `Result[None,E].unwrap()`'s
`return self.data.v_Ok`), applied consistently to the function's own C
*prototype* and its own `return;` statement - but not to the CALL SITE.
`UnsafeDict._owned_value` (V=NoneType) has a real `-> V` return type, so
lowering.py creates a real dest temp for its call result, and the emitter
blindly emitted `dest = <call to a void-returning C function>;` - a
straight C type error. Fixed by applying the exact same
`_returns_void_in_c()` check at the `ir.Call` emission site, mirroring
the existing prototype/return-statement rule rather than inventing a new
one.

## The blocker: `Ptr[None]` means two different, incompatible things

After all three fixes above, `dict[str, None]` STILL doesn't compile.
`UnsafeDict`'s non-RC value/key storage (`_owned_value`, `_owned_key`,
`_store_value`, `_store_key` - `lib/builtins/__init__.py`) works by casting
a borrowed `Ptr[None]` to `Ptr[V]` and dereferencing it:
```python
ptr: Ptr[V] = compiler.cast( Ptr[V], value_ptr )
return ptr[0]
```
For `V = NoneType`, this becomes `Ptr[NoneType]` - and `emitter_c.py`'s
`c_type()` **unconditionally** maps `Ptr[NoneType]`/`ConstPtr[NoneType]` to
C `void*`:
```python
if isinstance( inner_type, Scalar ) and inner_type.stem == 'NoneType':
    inner = 'void'
```
This is not a bug in isolation - it's the correct, deliberate, and heavily
relied-upon meaning of `Ptr[None]` *elsewhere* in this codebase: `RawDict`'s
own `key_ptr`/`value_ptr: Ptr[None]` fields (`lib/builtins/__RawDict.py`)
are exactly "an opaque, type-erased pointer" - real, working `void*`
semantics that a huge amount of the generic-container machinery (`RawDict`,
`sys.alloc`-adjacent erasure patterns) depends on.

So `ptr[0]` on a `Ptr[NoneType]` tries to dereference/assign through a
`void*` in the generated C - a real error either way:
```
error: assigning to 'MetalpyNone' (aka 'unsigned char') from incompatible type 'void'
error: incomplete type 'void' is not assignable
```

**The actual problem:** the single `NoneType` `Scalar` object (and its
`Ptr[NoneType]` specialization) is being asked to mean two different
things depending on how it got there:

- **"An opaque, erased pointer"** - `Ptr[None]` written directly, or
  produced by generic erasure code that was never meant to hold a real
  value (`RawDict`'s own fields). Correctly `void*`.
- **"A pointer to a real (if zero-content) value"** - `Ptr[V]` where `V` is
  a generic parameter that happens to have been instantiated with
  `NoneType` (`dict[K, None]`'s own `V`). Needs to behave like a real
  1-byte `MetalpyNone*`, not `void*`.

Nothing in the type object itself currently distinguishes these two cases -
they collide through the exact same `NoneType` Scalar and the exact same
`Ptr[T]`-specialization machinery.

## Why a narrow fix is risky

- `Ptr[None]`-as-`void*` is pervasive and load-bearing (`RawDict`'s own
  fields, and presumably other erasure-shaped code elsewhere in `lib/`) -
  changing `c_type()`'s `Ptr[NoneType]` handling wholesale would require
  auditing every existing use to confirm none of them actually wanted a
  real 1-byte pointee instead of `void*`. Getting this wrong risks a real,
  silent memory-layout regression in already-shipped, tested code
  (`dict[K,V]` itself, `set[T]`, everything built on `RawDict`).
- The two meanings are distinguishable only by *intent/provenance* (was
  this `Ptr[None]` written by hand as a deliberate erasure type, or
  produced by monomorphizing a generic `Ptr[V]` where `V` happened to bind
  to `None`?) - there's no existing signal on the `Type`/`Specialization`
  object itself that records which one applies.

## Candidate approaches (none attempted - each needs real scoping of its own)

- **(a) A distinct representation for "real NoneType value" vs "opaque
  erasure".** E.g. never actually let a generic `Ptr[V]` resolve down to
  the same `Ptr[NoneType]` specialization real hand-written erasure code
  uses - some kind of provenance-tagged variant. Unclear how invasive this
  is without a deeper look at how `Specialization`/`Ptr[T]` interning
  works today (risk of breaking the "same instantiation = same object"
  invariant several other subsystems rely on - see `PLAN_COMPILER_BUG_SWEEP.md`'s
  own Shape 1 for how much currently depends on that).
- **(b) Give `UnsafeDict` (and any future generic container built the same
  way) a dedicated, non-`Ptr[V]`-based storage path specifically for
  `V=NoneType`.** Narrower blast radius (touches only the handful of
  generic containers that go through this exact cast-and-deref pattern),
  but is real, non-trivial special-casing repeated per container, and
  needs `compiler.is_rc(V)`-style compile-time branching to detect
  "V resolved to NoneType specifically" (today's `is_rc`/scalar checks
  don't distinguish NoneType from any other non-RC scalar).
- **(c) Accept the limitation.** Document `None` as fundamentally
  unusable as a generic value-type argument through any `Ptr[V]`-based
  storage design (covers `dict[K,V]` today, and any future container built
  the same way) - keep `bool`/a dedicated 1-byte marker as the standing
  workaround, same as `set[T]` already does.

## Recommendation

Leave `set[T]` on `bool` (already shipped, works, costs one byte). The
three isolated fixes above are worth keeping regardless (real bugs,
zero-regression, no reason to revert), but none of them alone unblocks
`dict[K, None]` - that needs a real design decision on (a) vs (b) vs (c)
above, which is bigger than an implementation task and shouldn't be decided
unilaterally mid-fix. This document exists so a future session doesn't have
to re-derive the same three-layers-deep investigation from scratch - pick
up directly at "The blocker" section above.

## Verification plan for any future attempt

1. Work in a fresh `EnterWorktree` worktree (never reuse this one or any
   other named one - see this repo's `CLAUDE.md`).
2. Before touching anything: write a minimal repro (`dict[str, None]`
   construct/insert/lookup, compiled via `Discovery(import_builtins=True)` +
   `Compiler` + `emitter_c.emit_c(...)`, then actually built and run through
   a real C compiler - compiling alone isn't enough, the whole point is a
   correctness bug that only shows up at the generated-C level) - confirm
   it still fails today with the exact symptom described above.
3. Whichever approach is chosen, audit every existing `Ptr[None]` use in
   `lib/` (`RawDict`'s fields at minimum) to confirm it still behaves as
   opaque `void*` after the change.
4. Run the full suite (`python tests.py`) after the fix - this repo's suite
   runs in ~10s across 16 shards, no reason to skip it.
5. Commit, then merge into `master` via the shared-checkout exception in
   `CLAUDE.md`.
