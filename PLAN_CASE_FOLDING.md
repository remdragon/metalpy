# Plan: `case_folding` — optional, install()-able full Unicode casing

## Goal

`str.upper()`/`str.lower()` currently do correct *simple* (one-codepoint-in,
one-codepoint-out) Unicode case mapping via the OS (`LCMapStringEx` on
Windows, `towupper_l`/`towlower_l` on POSIX — see
`PLAN_STR_UPPER_LOWER.md`). Neither OS API implements Unicode's
`SpecialCasing.txt` rules: one-to-many expansions (`ß` → `SS`) or
context-sensitive rules (Greek final sigma). Only ICU reliably does, and we
decided against linking ICU (version-suffix/symbol-renaming headaches on
Debian/Ubuntu, not worth it — see prior discussion).

Instead: a separate, optional stdlib module, `case_folding`, that ships its
own self-generated Unicode data table (no OS/ICU dependency at all) and,
once explicitly installed by the user's program, upgrades every existing
`.upper()`/`.lower()` call site in the program — no call-site changes
needed. A program that never imports `case_folding` pays nothing: no extra
data, no extra linked code, binary size unaffected.

## The dispatch problem, and why the obvious design doesn't work

The obvious design — `case_folder: CaseFolding = BasicCaseFolding()` as a
global, `case_folding.install()` reassigning it to a different subclass
instance with an overridden `upper()`/`lower()`, `str.upper()` calling
`case_folder.upper()` polymorphically — **does not work with this compiler
as it exists today.** Confirmed by direct inspection:

- `RCClass.base` (inheritance) is stored but nothing ever walks it —
  `TODO.txt` states plainly: *"subclassing isn't a real feature yet at all
  ... `super()`/`super.__init__()` don't exist anywhere."*
- Method-call resolution (`type_resolver.py`'s `_attr_lookup_callable`)
  resolves purely from the receiver's **static declared type**'s own
  `names` dict — it never traverses `.base`, and there is no runtime
  type tag on an object to dispatch through even if it did.
- `emitter_c.py`'s call emission always compiles to a **direct call to one
  fixed, statically-known C function symbol** (`mangle_qualname(...)`) —
  there is no indirect/function-pointer call anywhere in the emitter, and
  no vtable concept in object layout at all.

So `x: Base = Derived(); x.foo()` calls `Base.foo`'s C symbol regardless of
what's actually stored in `x` — there is no virtual dispatch. Building real
polymorphic dispatch (or first-class function pointers, the other way to
get this) is its own multi-part compiler feature, not something to fold
into this plan.

## The design this plan actually uses: swap data, not behavior

`CaseFolding` is **one concrete class**, not a base class meant to be
subclassed. Its `upper()`/`lower()` methods are ordinary, statically-bound
methods that branch on their own **mutable fields**:

```python
@cstruct
class CaseFoldEntry:
    codepoint: u32
    mapped: u32          # 0 for entries covered by the multi-codepoint table instead

class CaseFolding:
    simple_table: Ptr[CaseFoldEntry]   # None until installed
    simple_count: usize
    special_table: Ptr[SpecialCaseEntry]  # one-to-many/context-sensitive entries
    special_count: usize

    def upper( self, s: str ) -> str:
        if self.simple_count == 0:
            return s.upper()   # not installed - fall back to str's own OS-backed path
        # ... table-driven mapping (binary search simple_table, consult
        # special_table for one-to-many/context-sensitive codepoints) ...

    def lower( self, s: str ) -> str:
        # mirror image
        ...

case_folder: CaseFolding = CaseFolding.__allocate__( simple_table = None, simple_count = 0, ... )
```

`str.upper()`/`str.lower()` themselves each gain one extra check at the top,
present in **every** build regardless of whether `case_folding` is ever
imported:

```python
def upper( self ) -> str:
    if case_folder.simple_count != 0:
        return case_folder.upper( self )
    # ... existing OS-backed path, unchanged ...
```

`case_folding.install()` (defined in the separate, optional module) does
the only thing it needs to: populate `case_folder`'s fields with pointers
into `case_folding`'s own embedded data table.

```python
# case_folding.py
def install() -> None:
    case_folder.simple_table = compiler.addrof( _SIMPLE_TABLE )
    case_folder.simple_count = _SIMPLE_TABLE_COUNT
    case_folder.special_table = compiler.addrof( _SPECIAL_TABLE )
    case_folder.special_count = _SPECIAL_TABLE_COUNT
```

**Why this achieves everything the original design wanted, without needing
dispatch:** the *data* is what gets swapped, not the *code*. `str.upper()`'s
table-lookup branch is small, fixed, always-compiled logic — cheap even
when unused. The actual Unicode table (the expensive part, tens/hundreds of
KB) lives entirely inside `case_folding`'s own module and is only
referenced — and therefore only scheduled/linked — by `case_folding.
install()` itself. A program that never imports `case_folding` never
references that data at all, so it's never linked in: binary size is
unaffected by this feature's mere existence. And because `case_folder` is a
real global instance with real mutable fields, `install()` is a genuine
runtime effect (an ordinary field assignment) — it just has to run before
the first `.upper()`/`.lower()` call that should see it (in practice:
called once, near the top of `main()`).

**Needs early verification, not yet confirmed:** can `case_folding.py`
(a different module from `builtins`) actually reassign fields on a global
instance `builtins` owns? Plain attribute assignment on an already-existing
instance is ordinary, already-supported functionality — the only open
question is whether cross-module reference to a *global variable* (not
just a global function/class) resolves the way this needs. First concrete
implementation step should be a small spike proving this before building
anything else on top of it.

## Data acquisition

Per your call: **no version pinning for now.** `case_folding` fetches
`UnicodeData.txt` and `SpecialCasing.txt` from the Unicode Character
Database's `latest` alias (`https://www.unicode.org/Public/UCD/latest/ucd/`)
at build time, the first time a program that imports `case_folding` is
compiled.

- **Where the download happens:** this needs a new compiler primitive,
  parallel to `compiler.cexpr()`/`compiler.has_library()` but for "fetch
  and embed a generated data blob" rather than "run a probe program."
  Concretely: a build-time step (invoked from `case_folding.py`'s own
  source, analogous to how `compiler.cexpr(...)` is invoked from ordinary
  metalpy source) that downloads the two files, parses them, generates the
  `_SIMPLE_TABLE`/`_SPECIAL_TABLE` data, and hands lowering.py something it
  can emit as static const C data. This is real new infrastructure, not a
  small addition — flagged as its own numbered step below.
- **Caching:** mirrors `compiler.cexpr()`'s existing convention — cache
  the downloaded files (and/or the generated table) under
  `%TEMP%/metalpy/case_folding/`, cached indefinitely once fetched (same
  no-expiry philosophy `cexpr`/`has_library` already use). Since there's no
  version pin, the cache key is just "have we ever fetched this" rather
  than being keyed by a version string — the tradeoff you're explicitly
  accepting is that two builds on two different machines (or the same
  machine after a manually-cleared cache) could pick up different Unicode
  versions if the UCD's `latest` has moved on between them.
- **Offline/CI resilience:** add an env var override (`METALPY_UNICODE_DATA_DIR`,
  mirroring `METALPY_CC`'s existing override convention) pointing at a local
  copy of the two UCD files, so a build never *has* to reach unicode.org if
  the files are already on disk.

## Table scope

- `UnicodeData.txt` fields 12/13/14 (simple uppercase/lowercase/titlecase)
  → `simple_table`, binary-searchable by codepoint.
- `SpecialCasing.txt`'s **locale-independent** entries only (no trailing
  language-condition field) → `special_table`, holding the up-to-3-codepoint
  expansions and the context-sensitive rules (final sigma needs a small
  amount of lookahead/lookbehind logic in `upper()`/`lower()` itself, not
  just a flat table — it depends on whether the codepoint is followed by
  another cased letter). Locale-conditional entries (Turkish, Lithuanian,
  Armenian) are out of scope for v1, same "invariant casing" posture the
  OS-backed paths already have.

## Testing, without pinning a version

Per your own suggestion: tests should not hardcode specific mappings that
could plausibly shift between Unicode versions. Instead:

- **Structural invariants**, checked against whatever table actually got
  downloaded: the simple table is sorted/binary-searchable, every mapped
  codepoint is a valid codepoint (≤ U+10FFFF), ASCII a-z/A-Z map exactly
  like the existing simple path already does, round-tripping (`upper()`
  then checking the result is stable under `upper()` again) holds for a
  large swept range of codepoints.
- **Long-stable, decades-old mappings** as the few hardcoded exceptions:
  `ß` → `SS` and Greek final sigma have been in every Unicode version for a
  very long time and are exactly the cases motivating this feature — safe
  to assert directly.
- **A "did case_folding actually get linked in" check** — a real
  compile+link+run test (mirroring `StrUpperLowerTests`' own pattern from
  `PLAN_STR_UPPER_LOWER.md`'s implementation) verifying a program that
  never imports `case_folding` produces an identical binary size /
  identical `.upper()` behavior to before this feature existed, and a
  program that does call `install()` measurably differs.

## Open items to confirm before implementation starts

1. The cross-module global-mutation spike (above) — if it doesn't work the
   way assumed, this whole design needs rethinking.
2. Exact shape of the new "fetch + generate + embed a data blob at build
   time" compiler primitive — this is the single biggest net-new piece of
   infrastructure in this plan and deserves its own focused design pass
   before coding starts, not just this one paragraph.
3. Whether `special_table`'s context-sensitive entries (final sigma) are
   worth the added lookahead complexity for v1, or whether v1 should ship
   with one-to-many expansions only (ß→SS etc.) and defer context-sensitive
   rules.
