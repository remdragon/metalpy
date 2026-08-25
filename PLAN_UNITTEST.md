# `lib/unittest.py` — a metalpy-native unit testing module

STATUS: PAUSED. Blocked on a prerequisite compiler feature (`@compiler.test`,
see below) that doesn't exist yet. This file exists so the design work
already done isn't lost/re-derived from scratch when picked back up.

## Goal

MetalPy's own `lib/` standard library (`json.py`, `csv.py`, `base64.py`,
`logging.py`, etc.) has no way for *metalpy programs themselves* to write and
run unit tests — only the Python-side compiler test suite
(`tests.py`/`test_support.py`) compiles and runs metalpy programs from the
outside. This was a greenfield design at the time of writing: a repo-wide
grep turned up zero prior mentions of `unittest`, `assert_eq`, or a
testing-framework plan anywhere in `PLAN_*.md`/`SYNTAX.md`/`ARCHITECTURE.md`.

The goal is a `lib/unittest.py` module letting a metalpy program define test
functions, assert against expected results without crashing the whole
process on one failure, and get a pass/fail summary plus a process exit code
a CI script (or `test_support.py`'s own `RealCompileMixin`) can check.

## Language constraints that shape the design

- **No exceptions, no reflection.** Only `Result[T,E]` (recoverable) and
  `sys.panic(msg) -> NoReturn` (full-process `exit(1)`, unrecoverable) exist.
  There's no `getattr`/`dir()`/method-enumeration of any kind, so a module
  can never walk *itself* looking for `test_*`-named functions.
- **`assert` already exists but always aborts the whole process** —
  `type_resolver.py`'s `visit_Assert` rewrites it straight to
  `sys._assert(cond, msg)` → `panic` on failure. Unsuitable as the mechanism
  for "record this test as failed, keep running the rest" — this module's
  own assertion helpers must return `Result[None, TestFailure]` instead of
  panicking, so one failing check doesn't kill the whole suite.
- **First-class function values are real but limited** (`PLAN_CALLABLE.md`,
  landed): a bare reference to an already-resolved, non-generic, receiver-less
  free function or `@staticmethod` lowers to `Ptr[Callable[[...], R]]`, usable
  as a function/method parameter type and a local variable type — but
  explicitly *not* confirmed as a struct/class field, deeply nested, or a
  function's own return type. Referencing a *bound instance method* as a
  value is separately, explicitly deferred. This rules out
  `unittest.TestCase`-style classes with test methods; a test case must be a
  plain free function.
- **No varargs/kwargs, no user-defined decorators** beyond a fixed compiler
  whitelist (`@overload`, `@staticmethod`, `@virtual`, `@property`,
  `@compiler.target(...)`, etc.) — rules out a pytest-style `@test`
  auto-registration decorator using *general* user-defined decorator support
  (confirmed out of scope with the user — too large a project to take on
  just for this module). A single new, *purpose-built*, compiler-whitelisted
  decorator is a different, much smaller story — see "Path forward" below.
- **Bare scalar f-string interpolation gap — RESOLVED since this design was
  drafted.** At design time, only `f64`/`f32` had real `__str__`/`__repr__`;
  every other scalar (`i8..i128`, `u8..u128`, `usize`/`isize`, `bool`) needed
  explicit `int(x)` boxing first, and that boxing only even worked for
  `i32` (`int.__init__` only ever accepted an `i32` parameter — confirmed by
  a real compile error while spiking this module: `usize` given to `int(...)`
  failed to compile at all). **This is now fully fixed on master**
  (`3c0ab6c`, "Add bool.__str__/__repr__, closing PLAN_STR_FORMAT.md item
  6" — completed via a task spawned mid-investigation of this module,
  merged via `cd9b7cd`): every fixed-width int scalar, `f32`/`f64`, and now
  `bool` all have real `__str__`/`__repr__`, so bare `f"{x}"` just works
  directly for every scalar type, no per-type boxing workaround needed
  anywhere. This simplifies `assert_eq`'s rich-message design considerably
  (see below) — no more "usize/u64/etc. can't build a rich message" gap.
- **Eager RC-value construction from module-level static-init is safe** —
  confirmed via a real spike (see "What was tried" below), resolving what
  had looked like a live bug in `lib/logging.py:286-298`'s own comment
  (traced to an already-fixed topological-sort gap,
  [[global_init_order_dependency_fix]]). This part of the original concern
  turned out fine; the thing that actually blocked the design was a
  *different*, separate mechanism — see below.
- **Generics are duck-typed per specialization** (`SYNTAX.md`'s own Generics
  section) — a generic function's body only needs to type-check for the
  concrete type actually used at each call site, same as a C++ template. A
  generic `assert_eq[T]` doing `actual == expected` compiles fine for any
  `T` with `__eq__`, and fails to compile (pointing at the call site) for any
  `T` that doesn't — no protocol bound is *required* for this to be safe,
  though protocol-bound TypeVars (`def f[T, S: SomeProtocol[T]](x: S)`,
  landed via `bd69eb1`) could tighten the error message in a future pass.

## The module design (still valid, once test discovery is unblocked)

### `TestFailure`

```metalpy
class TestFailure:
	message: str

	def __init__( self, message: str ) -> None:
		self.message = message
```

Mirrors the existing `Base64Error`/`CsvError` per-module error-class
convention. Every assertion helper returns `Result[None, TestFailure]`.

### Assertion helpers (module-level functions)

- `assert_true(cond: bool, msg: str) -> Result[None, TestFailure]`,
  `assert_false(...)` — the primitives everything else is built on.
- `assert_eq`/`assert_ne` for any scalar or `str` builds a real "expected X,
  got Y" message — now that bare scalar interpolation works universally
  (see above), this no longer needs a pile of per-type overloads just for
  message-richness; a single generic-ish implementation over the builtin
  scalar/`str` types suffices. Whether that's one function per concrete type
  (function overloading, `SYNTAX.md`'s "Function overloads") or something
  leaner is an implementation detail to settle when this resumes.
- `assert_eq[T]`/`assert_ne[T]` — a generic fallback for any other type with
  `__eq__` (duck-typed). Message interpolation of the actual/expected values
  now works whenever `T` also happens to have `__str__` (no longer scalar-
  specific) — still fails to compile at the call site for a `T` with neither,
  same as always.
- `assert_almost_eq(actual: f64, expected: f64, tolerance: f64 = 1e-9, msg: str = '') -> Result[None, TestFailure]`
  — float comparison needs a tolerance, not `==`.
- `assert_is_none[T]`/`assert_is_some[T]` for `T|None`, and
  `assert_ok[T,E]`/`assert_err[T,E]` for `Result[T,E]` — cheap, useful
  additions given how much metalpy code is itself `Result`-returning.

### Registry + `record()` + `finish()`

```metalpy
_passed: usize = 0
_failed: usize = 0

def record( name: str, result: Result[None, TestFailure] ) -> bool:
	match result:
		case Result.Ok( _ ):
			print( f'PASS {name}' )
			global _passed
			with compiler.wrap_arithmetic:
				_passed += 1
		case Result.Err( failure ):
			print( f'FAIL {name}: {failure.message}' )
			global _failed
			with compiler.wrap_arithmetic:
				_failed += 1
	return True

def finish() -> i32:
	with compiler.wrap_arithmetic:
		total: usize = _passed + _failed
	print( f'{_passed}/{total} passed' )
	if _failed > 0:
		return 1
	return 0
```

`finish()`'s `0`-on-success/nonzero-on-failure return matches both this
compiler's own `main() -> i32` convention and the exit-code-only signaling
`test_support.py`'s `RealCompileMixin` already checks for. (Counters shown
as `usize` here, not the `i32` the spike had to fall back to — safe again
now that scalar interpolation works for every width.)

## What was tried, and why it doesn't work: self-registering globals

The original idea (before this got blocked): have each test register its
own outcome via a module-level global initializer, running at static-init
time before `main()` — avoiding the need for any `Ptr[Callable]` value or a
hand-maintained call list in `main()`:

```metalpy
def test_addition() -> Result[None, unittest.TestFailure]:
	unittest.assert_eq( 1 + 1, 2, 'one plus one' ).or_return()
	return Result.Ok( None )

_r_test_addition: bool = unittest.record( 'test_addition', test_addition() )

def main() -> i32:
	return unittest.finish()
```

**This does not work.** Confirmed with a real standalone spike: a global
whose own initializer's return value is never read anywhere reachable from
`main()` gets its *entire initializer expression* dropped — not just
"optimized away" after being generated, but never lowered/emitted into the
generated C at all, confirmed by regenerating C source (`mpy.py -c`) for the
unreferenced-registration case and grepping it: the registration function,
the test function it called, and the dummy global itself appear **nowhere**
in the output. This holds even though the registration call's side effects
mutate a *different* global (`_passed`/`_failed`) that `main()` genuinely
does read — reachability is decided per declared global, not via a
side-effect-aware analysis of what a function body touches.

Confirmed this is exactly what's happening by editing the spike so `main()`
explicitly references the dummy globals (`_r1`, `_r2`) — the registration
calls then ran correctly (`PASS`/`FAIL` printed, counters updated, correct
exit code). But that "fix" just relocates the same boilerplate the whole
design was trying to avoid (`main()` still has to reference something per
test to keep it alive) — and arguably makes it worse, since a forgotten
reference silently drops a test with no visible symptom, unlike a forgotten
call in an explicit list.

This pruning behavior is confirmed **intentional and load-bearing** (not a
bug to fix): it's one of the mechanisms keeping generated-executable size in
check for a language where "import a stdlib module" mustn't drag in
everything that module could possibly do. Changing it was explicitly ruled
out by the user for that reason. The fallback (an explicit, hand-maintained
call list inside `main()`, unblocked and usable today with zero new compiler
work) remains viable but wasn't what this design was aiming for.

## Path forward: `@compiler.test`

Landed on during discussion, not yet scoped or implemented. The idea: a
single new compiler-whitelisted decorator — not general user-programmable
decorators, just one more case in the same fixed, compiler-recognized
mechanism `discovery.py` already uses for `@compiler.target(os=...)`
(see `discovery.py`'s `_target_value_matches`/`_matches_has_library` and
`COMPILER-TARGET.md` for that existing precedent) — with two effects, gated
on whether the compiler is invoked in a new "building for tests" mode
(mirroring `mpy.py`'s existing `--release` flag):

- **Normal (non-test) build**: a `@compiler.test`-decorated function is
  excluded entirely, before lowering ever sees it — same posture as an
  `@compiler.target(os='windows')` overload variant getting excluded on a
  non-Windows build today. Zero cost in production builds; no new
  reachability-pruning semantics needed for the general case, so this
  doesn't touch the load-bearing DCE property above.
- **Test build** (e.g. `mpy.py --test my_module.py`): every
  `@compiler.test`-decorated function becomes an *additional root* — reachable
  regardless of whether anything else calls it, the same way `main()` itself
  is already the one hardcoded root the whole stage-2 walk starts from — AND
  the compiler auto-collects the decorated functions into a synthesized
  dispatch, calling `unittest.record(name, fn())` for each one and
  `unittest.finish()` at the end (synthesizing `main()` itself if the module
  doesn't define one, the same shape `test_support.py`'s own
  `_merge_programs`/`_transform_case` already synthesizes today, just moved
  inside the compiler instead of living as an external Python-side AST
  transform).

This gets real, zero-boilerplate test discovery — no manual registration
call needed per test at all, just the decorator — without touching general
decorator support (no arbitrary wrapping/closure semantics needed, unlike
real Python-style decorators) and without an external preprocessing step
(explicitly ruled out by the user). It mirrors a well-trodden precedent
(Rust's `#[test]`/`cfg(test)`) rather than inventing a novel mechanism.

**Not yet scoped**: exact CLI flag shape, where in `discovery.py`/
`type_resolver.py` the extra-root/auto-collection logic plugs in, how
multiple test *files* compile together (single-file `mpy.py` invocation vs.
some multi-file test-target concept), and how the synthesized dispatch
interacts with `unittest.py`'s own `record`/`finish` API above. This is real,
separate compiler work and needs its own design pass before implementation
starts.

## Files (once unblocked)

- New: `lib/unittest.py` (the module itself, per the design above).
- New: `unittest_test.py` (Python-side real-compile test, following
  `csv_test.py`/`json_test.py`'s existing pattern) — exercises a compiled
  metalpy test module in `--test` mode, checks both the exit code and the
  actual `PASS`/`FAIL` stdout lines, across all 3 available compilers
  (MSVC, clang, gcc via WSL).
