# Scope: a tagless, null-pointer representation for single-pointer-payload `T|None` unions

## Context

Every `T|None` today becomes a full tag-byte + payload-union C struct,
unconditionally - `discovery.py`'s `_get_or_create_union` (~line 711-783)
builds a plain `TaggedUnion` for any `|` operand list with no shape-based
special-casing, and `union_storage.py`'s `UnionStorage.get()` (~line
138-292) synthesizes the same `{ u8 tag; union { ... } data; }` layout for
every one of them, whether the union has one member or ten, pointer-shaped
or not. This costs at least one byte (typically padded to a full pointer's
width) and a real constructor `ir.Call` for every value, even when the
single non-`None` member is a `Ptr[T]`/`ConstPtr[T]` that's already
inherently nullable on its own.

This is *why* the extern-boundary bridging hack in commit `3e0e325f`
(removed - see this repo's own history around the change that added this
document) had to exist in the first place: a foreign C function returning
"a pointer, or NULL" has no way to express this compiler's tag+payload
struct, so the fix had to special-case-recognize the shape and bridge a raw
pointer into it by hand at the one boundary where the mismatch was fatal.
The real fix isn't a boundary-only bridge - it's giving the language a
representation that never needed a tag in the first place for this one
common shape. `Part 1` of the change that produced this document instead
took the narrower, immediately-actionable route: ban `TaggedUnion` (and
`RCClass`) from `@extern` signatures outright, forcing extern authors onto
the bare-nullable-pointer idiom the codebase already uses correctly
elsewhere (`lib/windows/ws2_32.py`'s `inet_ntop`, `lib/socket.py`'s
`inet_ntop`). That sidesteps the extern-boundary case entirely but leaves
the general problem - `T|None` costs a real tag byte and a real
constructor call everywhere else in the language, not just at extern
boundaries - untouched.

## Proposed semantics

- A union with **exactly one non-`None` member T, where T is a pointer
  type** (`Ptr[T']`/`ConstPtr[T']`): represented as a bare, possibly-null
  pointer. No tag byte, no payload struct. `is None`/`== None` become a
  direct null-pointer comparison; `match None`/`match _` become a direct
  null-pointer branch; construction (`_coerce_into_union`, see below)
  becomes a no-op reinterpretation of the already-pointer-typed operand,
  not a call to a synthesized member constructor.
- A union with **more than one non-`None` member** (`A|B|None`, or any
  `A|B` regardless of whether either is itself nullable): **unaffected** -
  stays a real tagged struct, exactly as today. A multi-member union is a
  genuine value type; it has no spare bit pattern to self-encode "is this
  None" the way a single pointer's own null value can.
- **Open question, not resolved by this proposal:** a union with exactly
  one non-`None` member T where T is *not* a pointer (e.g. `i32|None`) has
  no natural null representation - no sentinel value of a plain `i32` is
  safely reserved as "absent" the way a null pointer is. Whoever picks this
  up needs to make an explicit call: leave that shape as an ordinary tagged
  struct (asymmetric with the pointer case, but simple), or design a real
  sentinel/wrapper scheme for it (a substantially bigger decision), or
  scope this proposal to pointer-only `T|None` permanently and document the
  asymmetry as intentional.

## Where this touches (six independent layers, all currently uniform)

Confirmed via two Explore investigations while designing the immediate
`@extern`-boundary fix (Part 1) - none of these currently special-case any
union shape at all, so each is a real, independent piece of work, not a
single central chokepoint:

- **`discovery.py`'s `_get_or_create_union`** (~711-783) - union-creation-
  time shape recognition. Would need to either (a) build a distinct marker/
  flag on the resulting `TaggedUnion` for this shape, or (b) skip
  `TaggedUnion` entirely for this shape and hand back the bare `Ptr[T']`
  specialization instead (architecturally cleaner if every other layer can
  tolerate a `T|None` annotation resolving to something that isn't a
  `TaggedUnion` at all - needs real investigation, not assumed).
- **`union_storage.py`'s `UnionStorage.get()`** (~138-292) - layout
  synthesis. Needs a null-pointer-only path that skips the tag/data struct
  entirely for the recognized shape.
- **`lowering.py`'s `_coerce_into_union`** (~4514-4554) - construction.
  Currently always emits a real `ir.Call` to a `UnionStorage`-synthesized
  member constructor; this shape would need to short-circuit to "the
  operand already IS the union's own runtime representation, use it
  directly."
- **`type_resolver.py`'s is-None narrowing and `match` desugaring** - three
  separate sites, not one: the `is None`/`is not None` rewrite (~3959-3998,
  ~4025) and `match`'s own `case None:`/`case _:` desugaring
  (`visit_Match`/`_match_pattern`, ~4639-4981). All three currently compare/
  switch on a `.tag` field unconditionally; each needs its own
  shape-recognizing branch to compare the pointer itself instead.
- **`cfg.py`'s tag-gated refcounting** (`_tag_gated_refcount_instructions`,
  ~1196-1260) - Incref/Decref on a union-typed value currently always
  tag-switches. Only matters if the recognized shape's own T can itself be
  RC (e.g. would this proposal ever apply to a pointer to an RC type? -
  worth deciding explicitly, since `Ptr[T]`/`ConstPtr[T]` are raw,
  non-owning pointers everywhere else in this language, which likely makes
  this a non-issue, but confirm rather than assume).
- **`emitter_c.py`** - field/type spelling for the recognized shape becomes
  just `c_type(T)` instead of the synthesized struct name. The two helpers
  Part 1 of the `@extern`-boundary fix removed (`_extern_nullable_pointer_
  leaf` and its two call sites) would become straightforwardly
  re-expressible in terms of whatever general mechanism this document's
  approach lands on, if anyone ever wants `@extern` boundaries to accept
  `Ptr[T]|None` again once it has a real tagless representation - not
  required, just a natural side benefit.

## Recommendation

Do not attempt without a dedicated worktree/session of its own - six
independent subsystems, each with its own currently-uniform, currently-
tested behavior for "every union looks the same," all need real design
decisions (not just mechanical edits) before any code changes. Land Part 1
of the `@extern`-boundary fix first (already done - see this repo's history
for the commit that also added this document). This document exists so a
future session doesn't have to re-derive the six-layer investigation from
scratch, the same way `PLAN_NONETYPE_GENERIC_VALUE.md` already does for its
own, differently-shaped `None`-representation gap.

## Verification plan for any future attempt

1. Work in a fresh `EnterWorktree` worktree (never reuse this one or any
   other named one - see this repo's `CLAUDE.md`).
2. Before touching anything: write a minimal repro (`Ptr[T]|None`
   construct/`is None`/`match`/return, compiled via
   `Discovery(import_builtins=True)` + `Compiler` + `emitter_c.emit_c(...)`,
   then actually built and run through a real C compiler - compiling alone
   isn't enough) - confirm today's actual tagged-struct baseline behavior
   before changing it.
3. Resolve the open question above (non-pointer single-payload `T|None`)
   explicitly before writing any code, not as an afterthought once the
   pointer case is done.
4. Whichever representation is chosen, audit every existing `T|None` use in
   `lib/` and the test suite for the affected shape, confirming each still
   behaves identically from the caller's perspective.
5. Run the full suite (`python tests.py`) after the change, and loop it
   20-30x - this is exactly the kind of RC/refcounting-adjacent
   representational change where one green run proves nothing (see this
   repo's own memory notes on RC/concurrency changes).
6. Multi-compiler verification (MSVC, clang, WSL gcc) - this touches
   `emitter_c.py`'s own struct/type emission directly, the same class of
   change that has previously shipped regressions caught only by a
   non-default compiler (see this repo's own memory notes on
   `linker_c.py`-adjacent compiler coverage).
7. Commit, then merge into `master` via the shared-checkout exception in
   `CLAUDE.md`.
