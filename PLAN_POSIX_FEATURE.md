POSIX-target compile gaps: walrus operator + slice syntax

Why

Found incidentally while stress-testing the new general assignability check
(`_check_assignable`, PLAN work already landed) against `lib/posix/time.py`
under a synthetic `active_target = {'os': 'linux', ...}`. POSIX-target `lib/`
code has apparently never been compiled end-to-end on a real dev machine here
(the machine this compiler is normally developed on is Windows) - real,
long-standing gaps in it were simply unknown until this incidental probe.

A single `import posix.time; get_local_timezone_name()` compile attempt
produces 5 raw errors:

	lib/posix/errors.py:1: typing
	lib/posix/time.py:12: 'utf8' is not a value, cannot use it as an expression
	lib/posix/time.py:49: name 'target_path' is not defined
	lib/posix/fs.py:24: unsupported expression: :nbytes
	lib/posix/time.py:58: unsupported expression: (f := open('/etc/timezone', 'r').unwrap_or())

Re-tracing each to its real, independent root cause (re-verified live against
this checkout, not just static reading) found these are NOT five independent
problems: two are genuine unimplemented language features (this plan's own
scope, below), one is a real but unrelated compiler diagnostic bug, one is a
cascade artifact of the other two, and the rest are plain authoring mistakes
in `lib/posix/*`/`lib/crt.py` (also documented here, but explicitly NOT this
plan's own implementation scope - separate, much smaller follow-ups).

Root-cause breakdown

1. `lib/posix/errors.py:1: typing` - `from typing import TypeAlias`
   (errors.py:1). `TypeAlias` is a compiler-recognized sigil, never needing a
   real import (see discovery.py's own `_parse_type_alias`: "recognized
   purely by AST shape... same posture as compiler.target/compiler.sizeof").
   No file anywhere else in `lib/` imports it despite several using it
   (fs.py's own `HANDLE: TypeAlias = ...`-style patterns elsewhere in the
   tree). `typing` itself doesn't exist as a real module here, so
   `import_name('typing')` raises `FileNotFoundError('typing')`, and
   `str(FileNotFoundError('typing'))` is literally the string `'typing'` -
   the entire error message. Not a language gap - a one-line authoring
   mistake (delete the import).

2. `lib/posix/time.py:12: 'utf8' is not a value` - MISATTRIBUTED. The real
   bug is `lib/posix/fs.py:12`:
       def readlink( path: str, codec: Codec = utf8 ) -> Result[str,PosixError|CodecError]:
   `utf8` (from `from codecs.utf8 import utf8`) is a CLASS
   (`class utf8( Codec ):`, lib/codecs/utf8.py), not an instance - using the
   bare class as a `Codec`-typed default value is a real usage bug in fs.py
   itself (likely meant `utf8()`), not a language gap.

   Separately, and more interestingly: WHY does this get reported against
   `time.py:12` instead of `fs.py:12`? A real, independent compiler bug in
   `lowering.py`'s `_lower_call_args` (currently line 6138):
       for param in target.parameters or []:
           if param.stem not in given and param.default is not None:
               default_operand = self._lower_expr( param.default, param.type )
   This default-value-lowering path never pushes `discovery.module_context`/
   `scope_context` for `target`'s OWN module before lowering `param.default` -
   unlike every other default-lowering entry point in this file (function-
   body lowering itself, annotation resolution, etc.), which does. Since this
   runs while `get_local_timezone_name` (time.py) is mid-lowering,
   `discovery.module_stack[-1]` is still time.py's own `Module` when the
   error fires, even though the AST node (and its correct line number, 12)
   genuinely belongs to fs.py. A real, if minor, diagnostic-quality bug -
   worth its own separate fix (push `module_context`/`scope_context` for
   `target`'s own module around this loop, mirroring how real function-body
   lowering already does it) - NOT bundled into this plan's own scope.

3. `lib/posix/time.py:49: 'target_path' is not defined` and
   `lib/posix/fs.py:24: unsupported expression: :nbytes` - a CASCADE, not two
   independent bugs. `fs.py:24` is `return codec.decode( buf[:nbytes] )` -
   Python slice syntax, confirmed entirely unimplemented (see Scope below).
   `time.py:47` (`target_path = readlink('/etc/localtime').unwrap_or()`)
   needs `readlink` fully resolved/lowered to succeed, which needs both
   item 2's fix (fs.py:12) AND this item's own slice-syntax fix (fs.py:24) -
   either failure alone already means `target_path` never gets a real
   binding. `time.py:49`'s own `if target_path:` then reports "not defined"
   via the ordinary per-statement recovery boundary (lowering.py's own
   `try: self._lower_stmt(stmt) except CompileError: continue`, confirmed
   present at every statement/block level in this file). Fixing items 2 and
   the slice gap below should make this error disappear on its own, with no
   separate fix of its own needed.

4. `lib/posix/time.py:58: unsupported expression: (f := open(...).unwrap_or())`
   - Python's walrus operator (`ast.NamedExpr`). Confirmed: zero handler
   anywhere in discovery.py or lowering.py (the one `NamedExpr` hit in
   lowering.py, an aside comment inside `_reject_free_variables`, explicitly
   says walrus is NOT special-cased there because there's nothing walrus-
   specific implemented to special-case). No prior planning/deferral found -
   TODO.txt's only `:=` hits (lines 235-237) are a brainstorm about a
   completely different, unrelated hypothetical `@move`-consuming rebind
   operator, not Python's real walrus expression. Two real occurrences in
   `lib/`, both in this same function (time.py:58 and time.py:61 - the
   second never surfaces as its own separate error today because line 58's
   own failure already aborts that whole `if`-block via the same per-
   statement recovery boundary as item 3).

Scope for this pass

In scope - the two genuine, unimplemented language features:

1. Walrus operator (`x := expr`, `ast.NamedExpr`) in ordinary expression
   position (an `if`/`while` condition, or any other value-expression
   context). Comprehension-specific scoping (Python's real walrus semantics
   bind into the ENCLOSING function scope, not a comprehension's own private
   scope) is out of scope for now - metalpy has no comprehensions at all yet,
   so this nuance has no forcing case.
2. Slice syntax (`x[a:b]`, `x[:b]`, `x[a:]`, `ast.Slice` in subscript
   position). Not POSIX-specific at all despite being found here - a general
   subscript-lowering gap that would bite any code anywhere using slice
   syntax on any indexable type.

Deferred (flagged, not attempted this pass - separate, smaller follow-ups):

- The `_lower_call_args` default-value error-misattribution bug (item 2
  above) - a real compiler diagnostic-quality fix, unrelated to either
  language feature.
- `lib/posix/errors.py:1`'s unnecessary `from typing import TypeAlias` -
  one-line deletion.
- `lib/posix/fs.py:12`'s `codec: Codec = utf8` (class, not instance) -
  one-line fix, likely `utf8()`.
- `lib/crt.py:83`: `def strerror( errnum: 32 ) -> ConstPtr[u8]|None:` - a bare
  integer literal (`32`) where a type annotation belongs, clearly a typo for
  `i32`. Confirmed the only place in all of `lib/` where a parameter
  annotation is a bare numeric literal. Currently on a dead-code path only
  (`_posix_strerror`, itself only reachable from `PosixError`'s own
  commented-out `__str__`, see below) - low urgency, but a real bug once that
  code path is ever revived.
- Already self-documented in `lib/posix/errors.py` itself (not new
  discoveries, just noted for completeness): `PosixError`'s own enum members
  can't pull in a named constant as their value yet ("TODO FIXME: pulling in
  a constant as the value is not supported yet" - `NotFound = 2 # ENOENT`
  instead of `NotFound = ENOENT`), and CEnum doesn't support methods yet
  (`__str__` is commented out with its own "TODO FIXME: implement method
  support for enums later").

Precedent reused, not invented fresh

- Walrus: `_stmt_Assign`'s own "first assignment to a name with no prior
  declaration" branch (lowering.py:2217, roughly lines 2243-2283) already
  does almost exactly what a walrus binding needs - lower the RHS with no
  expected type, build a new `Variable` in the current function scope
  (`fn.add_name`), schedule its type, run `_cfg_assign`, emit `ir.Assign`.
  The only real difference: that's a STATEMENT (no return value); a walrus
  is an EXPRESSION and must hand back the assigned operand as its own value
  (so `if f := open(...):` can use it directly as the condition).
- Slice: `_expr_Subscript` (lowering.py:4708) already has precedent for a
  "special-case an unusual subscript shape before the ordinary scalar-index
  path" branch - see its own tuple constant-index handling (`isinstance(
  node.slice, ast.Constant)` checked ahead of the ordinary GetItem
  fallback). A slice fix follows the identical shape: check
  `isinstance(node.slice, ast.Slice)` before whatever currently blindly
  treats `node.slice` as a scalar index expression.

Implementation (sketch only - this pass is a scoping doc, not a build)

1. Walrus (`lowering.py`): new `_expr_NamedExpr` handler on `FunctionLowering`,
   dispatched the same way every other `_expr_X` method already is (via
   `_lower_expr`'s own `getattr(self, f'_expr_{node.__class__.__name__}')`
   lookup - no new dispatch plumbing needed, `NamedExpr` just needs an
   implementation to be found). Lower `node.value` with no expected type
   (mirroring `_stmt_Assign`'s own first-assignment branch), build a real
   `Variable` bound into the CURRENT function scope via `fn.add_name`
   (`node.target` is always a bare `ast.Name` per Python's own grammar - no
   other walrus target shape exists), run the same `_cfg_assign`/`ir.Assign`
   sequence `_stmt_Assign` already uses, then return the operand as this
   expression's own value. Needs real end-to-end verification that the new
   binding is visible to code textually AFTER the walrus expression within
   the same scope (the whole point of `if f := open(...): ... f.close()`) -
   should fall out for free from reusing the identical CFG-binding mechanism
   ordinary assignment already relies on, but confirm directly, not just by
   inspection.
2. Slice (`lowering.py`): a new branch in `_expr_Subscript` checking
   `isinstance(node.slice, ast.Slice)` before the existing scalar-index
   paths, dispatching to whatever the receiver's own real slicing mechanism
   should be (e.g. `bytearray`/`str`/`list[T]`'s own `__getitem__` taking a
   real slice, OR a dedicated `__getslice__`-shaped dunder if the existing
   single-index `__getitem__` protocol can't cleanly express a slice's own
   3-way (start/stop/step) shape - needs a real design decision during
   implementation, not assumed here). `ast.Slice.lower`/`.upper`/`.step` are
   each `expr | None` (any can be omitted, e.g. `x[:n]` has `lower=None`) -
   the lowering needs to pick sensible defaults (0 for start, the
   container's own length for stop, matching Python's real slice semantics)
   for whichever pieces are omitted.

Verification

- Re-run this plan's own repro (`import posix.time; get_local_timezone_name()`
  under `active_target={'os':'linux',...}`) after each piece lands - the
  error list should visibly shrink: fixing the two authoring bugs (fs.py:12,
  errors.py:1) plus slice support should clear items 1-3 entirely; walrus
  support should clear item 4 (both occurrences, since line 61 is only
  reached once line 58 stops aborting the block).
- New unit tests for walrus (`lowering_test.py`): a bare `if x := f():`
  binding visible after the if-statement, a binding used again later in the
  same function, nested walrus inside a boolean expression.
- New unit tests for slice (`lowering_test.py`/`emitter_c_test.py`): `x[:n]`,
  `x[a:]`, `x[a:b]` against at least `str`/`bytearray`, plus a real compile-
  and-run test confirming the sliced VALUES are correct, not just that it
  compiles.
- Full `python tests.py` green throughout, plus a real compile-and-run of
  `lib/posix/time.py`'s `get_local_timezone_name` once all pieces land, as
  the actual forcing case this plan started from.

STATUS: DONE - walrus and slice syntax both landed, and the investigation
kept going several layers deeper than originally scoped, fixing a real
double-free along the way. Every item this doc originally deferred was
later closed out too (round 5, below), plus one new safety rule and a
second real, previously-unknown crash bug. Full history below; `python
tests.py` green throughout (final count: 1070/1070), confirmed via 12
consecutive full-suite runs after round 3's own double-free fix (a real
double-free is exactly the kind of bug that only shows up intermittently
under load - a single clean run doesn't prove it's gone).

**Round 1 - the two features themselves.** `lowering.py`'s `_expr_NamedExpr`
(walrus) and `_lower_slice_subscript`/`bytearray._byte_slice` (slice syntax,
scoped to `str`/`bytearray`) landed. Re-running this doc's own repro
afterward showed the walrus/slice "unsupported expression" errors gone,
replaced by two newly-exposed, then-still-deferred authoring bugs those
gaps had been masking: `fs.py:24`'s `codec.decode(buf[:nbytes])` (a
`bytearray`, but `decode()` only accepted `bytes`) and `time.py:58`'s
`open('/etc/timezone','r')` (`open` was never a real name anywhere in this
codebase). Also found: `if x is not None: use(x)` narrowing is unimplemented
for a plain `if` statement (only `while`/`match`/`type(x) is T` narrow) -
confirmed independent of walrus, real, separate, not attempted.

**Round 2 - chasing a real test for `decode()` down the rabbit hole.**
Widening `decode()` to `bytes|bytearray` (`Codec`/`ascii`/`cp437`/`latin1`/
`utf8`, `lib/codecs/*`) and fixing `utf8.decode`'s own body (it had never
been compiled by anything real either - `b.get_bytes()` isn't a real method,
and it fed a non-terminated buffer into `str.from_cstr`'s terminator-
requiring overload) turned out to need a REAL compile-and-run test to
trust. Constructing a real `utf8()` instance forces its whole vtable to
compile (not just `decode()`), which surfaced three more previously-
unexercised bugs, each fixed in turn:
- `utf8.encode()`'s own `get_ptr()`/`get_const_ptr()` mixups (`lib/codecs/utf8.py`).
- List-literal syntax (`[a, b, c]`, `ast.List`) was entirely unimplemented -
  `utf8.names()`'s own `return ['utf8', 'utf-8', ...]` needed it. New
  `_expr_List` + `_construct_generic_instance` in `lowering.py`, scoped to
  `expected_type` already being a concrete `list[T]` (no element-driven
  inference - every real site, `lib/zoneinfo.py:33` included, already has
  one).
- **A real, root-cause compiler bug**: `move[T]`/`copy[T]` was modeled as a
  distinct wrapper *type* (`Move`/`Copy` in `mpy_types.py`) rather than an
  ownership *status* on a binding - already self-documented as wrong and
  unfixed in `Move`/`Copy`'s own docstrings and `TODO.txt`'s "incref/decref"
  section. In practice this meant a `move[T]`-typed parameter couldn't have
  ANY property read on it inside the function that owns it (confirmed two
  ways: `len(src)` and a direct `src.__len__()` both failed differently).
  Fixed at the root: `Parameter` gained `is_move`/`is_copy` flags;
  `discovery.py`'s own parameter-construction site now unwraps `move[T]`/
  `copy[T]` down to plain `T`, recording ownership on the new flags instead
  - every ordinary consumer (attribute lookup, generic inference,
  assignability) now sees a plain binding like any other. This is what
  finally let `bytes.from_bytearray`/`str.from_cstr`'s own real
  `move[bytearray]` parameters work (previously dead code - nothing in
  `lib/` had ever called either).
- **`bytearray.release()`'s own return-type bug**: `Result[Ptr[u8],
  sys.OwnershipError]` (bare, unspecialized generic) left `T` unbound, so
  `Err(SharedReference(x))`'s `x` never resolved to a real `bytearray`
  anywhere that pattern was matched - fixed to
  `sys.OwnershipError[bytearray]`.

**Round 3 - a real double-free, found via test flakiness, not inspection.**
The real-compile test for the above was intermittently flaky (~10-15%,
never in isolation) - not noise: `@move` on a method (e.g.
`bytearray.release()`) means calling it consumes `self`, but nothing
anywhere transitioned the CALLER's own ownership-tracking state for the
RECEIVER - confirmed by grepping every read site of `Function.is_move` in
`lowering.py` (only the callee's own internal view of `self`, via `cfg.py`'s
`enter_self`, ever consulted it). So a receiver like `src` in `str.from_cstr`
still got an ordinary Decref at its own scope exit on top of `release()`'s
own internal cleanup - a genuine double-free, matching the observed
heap-corruption-flavored intermittent failures exactly. Fixed with one
addition in `lowering.py`'s `_lower_call`: when calling an `@move` method,
run the receiver through `cfg.py`'s own `move()` (already the exact
mechanism used for `move[T]` argument-passing, just never invoked for a
receiver) - confirmed via direct instruction-level inspection (the spurious
`Decref` disappears with the fix, present without it) and via 12 consecutive
clean full-suite runs where before there'd reliably be a failure within
10-15.

**Round 4 - `time.py`'s own rewrite, the original forcing case.**
`_read_etc_timezone_file` rewritten to use the real `File.binary_reader`/
`BinaryReader.read` API (`open(path,mode)` was never real), `match`-based
Result narrowing throughout (per round 1's own narrowing finding), and an
explicit `utf8()` codec instance (not `.decode()`'s own broken `codec: Codec
= utf8` default - the same class of bug as `fs.py:12`, now also found on
`bytearray.decode()`'s identical default, not fixed, out of scope). Along
the way, fixed one more small, real, previously-dead bug it exposed:
`lib/builtins/__File.py`'s `BinaryReader.__del__`/`.close()` both discarded
`close_raw(...)`'s own `Result` (now `.is_ok()`'d explicitly - a destructor
can't propagate failure, but it shouldn't silently go unchecked either).

Re-ran this doc's own full repro (`get_local_timezone_name()`) after all of
the above: `fs.py:24`'s type-mismatch and `time.py:58`'s undefined-`open`
errors are both gone (as they were after round 1/round 2 respectively). One
NEW error now surfaces in their place, reached specifically because
`_read_etc_timezone_file` now compiles far enough to call `str.strip()`:
`crt.py:162`'s `compiler.cexpr('LC_CTYPE_MASK', 'locale.h')` fails to
compile on this (Windows) host - confirmed, via isolated repro, that this
is a genuine, pre-existing, environment-specific cross-compilation
limitation, not a code bug: `str.strip()` needs a real POSIX `<locale.h>`
for `compiler.cexpr`'s own constant-probing to succeed against, which
doesn't exist on Windows; the identical snippet compiles cleanly under the
default (Windows) `active_target`. Out of scope - would need a real Linux
sysroot/cross-toolchain, not a code fix.

**Round 5 - every item this doc had deferred, now closed out**, plus one new
safety rule (per a separate, later plan - `~/.claude/plans/scope-out-the-fix-
splendid-tome.md`), and a real, previously-unknown crash found along the way:

- **A real, general RC-correctness bug, found while implementing the
  `fs.py:12` fix below**: any module-level global constructed from an
  RCClass with NO fields of its own (a stateless singleton, or a fieldless
  subclass) never got its real `__metalpy_init_<name>()` call emitted -
  `emitter_c.py`'s `_global_init_is_all_zero_value_type` vacuously matched
  "every field is zero" on an *empty* fields dict, misclassifying a real
  heap `sys.alloc(...)` construction as a skippable value-type zero-init.
  The global's own C declaration (`= {0}`, i.e. NULL) was then never
  overwritten - reading through it crashed (a real access violation,
  confirmed reliably on any fieldless-subclass-with-a-virtual-method
  global, not a hypothetical). Fixed: that function's `ir.Allocate` branch
  now declines outright whenever `instr.cls` is an `RCClass`, regardless
  of field count.
- `errors.py:1`'s stray `TypeAlias` import - deleted.
- `fs.py:12`'s `utf8`-class-not-instance bug - `utf8` renamed to `Utf8`,
  with a shared `utf8: Utf8 = Utf8()` module-level singleton (stateless,
  confirmed no fields anywhere) taking over the OLD class name, so every
  existing bare reference keeps working unchanged.
- `ascii`/`cp437`/`latin1`'s own dormant `get_ptr()`-on-`bytes` bug in
  their `decode()` bodies - fixed (`get_const_ptr()`).
- `crt.py:83`'s `strerror` param type typo (`errnum: 32` -> `i32`) and
  `crt.py`'s `readlink` return-type signedness (`usize` -> `isize`,
  coordinated with `fs.py`'s own now-real `nbytes < 0` check).
- The `_lower_call_args` module-context misattribution bug - default
  values are now lowered under the CALLEE's own module/scope, not the
  caller's (previously could fail outright for a default referencing a
  name private to the callee's own module, not just mis-locate the error).
- `str.from_cstr(move(b))`'s own overload-resolution gap - `move(...)`
  sugar is now recognized during overload candidate matching too, not just
  against an already-resolved single target.
- The plain-`if` narrowing gap - `if x is not None:`/`if x is None: ...
  else:` now narrows, for the `T|None` (exactly one non-None member) shape.
- New safety rule: calling an `@move`-decorated method through a union-
  typed receiver or an overload group is now a compile error (both were
  silent gaps - ownership neither tracked nor rejected).

Re-ran this doc's own full repro (`get_local_timezone_name()`, synthetic
`linux` `active_target`) after all of the above: every error above this
round is gone. Two NEW, unrelated, separately-tracked issues surfaced in
their place (both pre-existing, only reachable now that everything else
compiles far enough to reach them) - a `Result.unwrap_or()` generic
return-type bug (zero-argument call site resolves to the wrong return
type) and `crt.py:162`'s already-known `locale.h` cross-compilation
limitation (unchanged, still needs a real Linux toolchain). Also found,
separately, while implementing the `move(...)`-in-overload fix: a general
C-symbol-collision bug (two overload candidates that both need real C
bodies mangle to the identical C function name). None of these three are
fixed by this round - each spawned as its own separate, self-contained
follow-up task.

Full `python tests.py` green throughout this round too (final count:
1070/1070).
