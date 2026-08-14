Wiring global-variable initializers into a single __metalpy_init()

Why

emitter_c.py already has almost everything needed to run a global
variable's real initializer (constructor calls, temporaries, arithmetic -
not just a bare constant) before main() starts. What's missing is the
last wire: the per-global init function it already synthesizes is never
actually called from anywhere. This is a known, explicitly-flagged gap -
TODO.txt lines 435-437 ("linking is entirely out of scope so far: no
wiring of a module's __metalpy_init_<global> functions into the real
process entry point") - and emitter_c_test.py's own
EmitGlobalRCClassRealCompileTests.test_non_trivial_global_compiles calls
itself "the FINAL milestone of the whole C-emitter plan," but only
asserts the output *compiles*, not that the constructor's side effects
are actually observable at runtime, precisely because nothing calls the
init function yet.

Two real stdlib globals already need this: lib/sys.py:40
(`stdout: _Stdout = _Stdout()`) and lib/builtins/__init__.py:1366
(`case_folder: CaseFolding = CaseFolding(...)`). Both are self-contained
RCClass constructions with no cross-global dependency on each other.

Separately, the C prologue currently defines __metalpy_init() twice,
selected via a top-level `#ifdef _WIN32 / #else / #endif` (emitter_c.py
lines 160-173) - one version calls SetConsoleOutputCP(CP_UTF8), the other
is an empty stub. Once __metalpy_init() also needs to run global
initializers, it must run on every target, not just when
`_is_entry_point(...) and active_target['os'] == 'windows'` - so the two
full function bodies collapse into one function with only the
Windows-specific statement gated internally.

Current state (what already works, what doesn't)

- discovery.py's visit_AnnAssign/visit_Assign already accept
  `name: T = <any expression>` at module level and build a real
  `Variable(is_global=True, init=<the expression>)` - this already
  covers "creation of global variables which may involve calling
  constructors" at the *front-end* level; nothing new is needed there.
  (Confirmed by the existing EmitGlobalRCClassTests fixture:
  `g_foo: Foo = Foo.make(1)` already discovers/resolves/lowers cleanly
  today.)
- lowering.py's FunctionLowering.run_global (lines 1271-1296) already
  lowers that initializer expression through the exact same
  `_lower_expr` machinery an ordinary function body uses - a
  multi-instruction sequence (DeclareTemp/Call/Allocate/Assign, ...) for
  a real construction, not just a single Const.
- emitter_c.py's emit_global (lines 1733-1762) already branches on
  `_is_trivial_global_init`: a trivial global becomes a real C static
  initializer (`uint32_t X = -11;`); a non-trivial one gets a `{0}`
  zero-initializer PLUS a private `static void __metalpy_init_<name>(
  void )` function containing the real instruction sequence - already
  emitted into the translation unit today.
- What's missing: nothing ever calls `__metalpy_init_<name>()`. It sits
  in the generated .c file, compiles, and is dead code.
- Separately: __metalpy_init() itself (the OS-setup one, not the
  per-global ones) is only ever called when compiling for the Windows
  target (emitter_c.py lines 1939-1943), and is spelled as two entire
  competing function bodies rather than one function with an internal
  `#ifdef`.

Scope for this pass

In scope:
1. A single `__metalpy_init( void )` function, always defined, with only
   the Windows console-codepage statement gated behind `#ifdef _WIN32`
   inside its body (not two separate function definitions).
2. __metalpy_init() is extended to call every non-trivial global's own
   `__metalpy_init_<name>()`, in the order those globals were lowered
   (i.e. `compiler.globals` order - see "Ordering" below).
3. __metalpy_init() is called unconditionally as the first statement of
   `main()`, on every target OS - not just Windows - since global
   initializers must run everywhere now, not only the Windows
   console-codepage setup.
4. The already-existing `no_crt` custom entry point (Windows-only,
   `mainCRTStartup`) keeps calling __metalpy_init() first, unchanged in
   spirit - it already does this; just needs to keep working against the
   new single-function shape.
5. Each global's lowered init instructions are kept attached to their
   owning Variable object (not just reachable indirectly through a
   separate list) - see Implementation step 1.

Deferred (flagged, not attempted this pass):
- True dependency-ordering between globals (a topological sort so that
  if global B's initializer reads global A, A is guaranteed to run
  first). `compiler.globals` order today is TypeResolver's FIFO
  scheduling order (first-referenced-while-lowering-reachable-code), not
  source/declaration order and not a real dependency analysis. Both real
  motivating cases (lib/sys.py's `stdout`, lib/builtins/__init__.py's
  `case_folder`) are independent of each other, so this doesn't block
  them - flagged as a known limitation for whoever needs cross-global
  initializer dependencies later, not silently ignored.
- Arbitrary top-level statements (bare `if`/`for`/expression-statements
  unrelated to a variable declaration) at module level. This pass is
  scoped to what a global variable's own declaration statement already
  produces (a single expression, however much real computation that
  expression itself lowers to) - not a general "run any statement at
  module scope" feature. discovery.py's visit_Expr already special-cases
  the one existing module-level statement form
  (`compiler.require_header(...)`) and silently ignores anything else;
  that stays as-is.
- Skipping the __metalpy_init() call/definition entirely when there's
  nothing for it to do (no Windows target, no non-trivial globals).
  Always emitting and always calling it is simpler and cannot regress
  correctness (worst case it's an empty function call) - an optimization
  to drop it, if ever wanted, is separable follow-up.

Precedent reused, not invented fresh

- emitter_c.py's own `_is_trivial_global_init` / emit_global split -
  reused as-is to decide which globals need a call at all.
- The existing `.replace( '{\n', '{\n\t__metalpy_init();\n', 1 )` splice
  emitter_c.py already uses to inject a call at the top of main()'s body
  (line 1942) - same technique, just no longer gated on the Windows
  target.
- The existing `no_crt` block's "call __metalpy_init() before main()"
  convention (lines 1945-1958) - unchanged, just now calling into a
  function that does more.

Implementation

1. compiler.py: in `Compiler._lower`'s `Variable` branch (around line
   231-237), after computing `instructions = self.lowering.lower_global(
   unit )`, also stash them directly on the variable:
   `unit.init_instructions = instructions` (new field, see next item),
   THEN build `LoweredGlobal(variable=unit, instructions=instructions)`
   as today (same list object referenced from both places - no
   duplication, no drift risk between the two).
2. mpy_types.py: add `init_instructions: list['ir.Instruction']|None =
   None` to `Variable` (mirrors `Function`'s own lazy-population
   pattern - `None` until lowered, matching "no new abstraction, just
   make the association direct" per the request that initialization
   logic travel with the variable object itself, not only be reachable
   through a side list on Compiler).
3. emitter_c.py:
   a. Remove the two `static void __metalpy_init( void ) { ... }`
      definitions from PROLOGUE (lines 160-173). Keep the fixed,
      target-independent declaration bits only: the `#ifdef _WIN32`
      block still declares `SetConsoleOutputCP` (extern decl +
      `#pragma comment` for MSVC's import-lib alternate-name trick) and
      `#define CP_UTF8 65001` - just no function body.
   b. Factor the existing inline `init_name = f'__metalpy_init_{name}'`
      (emit_global, line 1752) into a small shared helper, e.g.
      `_global_init_fn_name( g: LoweredGlobal ) -> str`, so emit_global
      and the new __metalpy_init synthesis (next item) can't drift on
      the naming scheme.
   c. In `emit_c()`, immediately after the existing
      `for g in compiler.globals: parts.append( emit_global( g ))` loop
      (pass 3, ~line 1923-1924) and before the interface-vtable-instance
      loop, synthesize and append the single, real __metalpy_init():
      ```
      static void __metalpy_init( void ) {
      #ifdef _WIN32
      	SetConsoleOutputCP( CP_UTF8 );
      #endif
      	__metalpy_init___main__$g_foo();
      	...
      }
      ```
      One call line per global in `compiler.globals` for which
      `not _is_trivial_global_init( g.instructions )`, in list order,
      using `_global_init_fn_name(g)`. This placement guarantees every
      per-global init function it calls was already textually defined
      earlier in the same translation unit (the globals loop just
      above), and it itself lands before pass 3's function-bodies loop
      (where main() gets its call spliced in) - both directions of C's
      "declared before use" requirement are satisfied with no forward
      declarations needed.
   d. Change the main()-prepend condition (~line 1939-1943) from
      `if _is_entry_point( lf.function ) and compiler.disco.active_target['os'] == 'windows':`
      to just `if _is_entry_point( lf.function ):` - __metalpy_init() is
      now always defined and always safe (and necessary) to call, on
      every target.
   e. Leave the `no_crt` block (lines 1945-1958) alone structurally - it
      already calls `__metalpy_init()` first; it now transparently picks
      up global-initializer calls too, no changes needed there beyond
      confirming it still compiles.
4. TODO.txt: remove the now-resolved bullet at lines 435-437 ("no wiring
   of a module's __metalpy_init_<global> functions into the real process
   entry point"); keep the sibling bullet about linker pragmas/flags for
   @extern libraries, which remains genuinely out of scope.

Verification

- emitter_c_test.py: new assertions that `emit_c()`'s output contains
  exactly one `static void __metalpy_init( void ) {` definition (not
  two, not `#ifdef`-selected), that its body contains the Windows
  console-codepage call inside `#ifdef _WIN32`, and that it contains a
  call to each non-trivial global's own init function - reusing the
  existing `g_foo: Foo = Foo.make(1)` fixture from
  EmitGlobalRCClassTests.
- New assertion that `main()`'s emitted body calls `__metalpy_init();`
  as its first statement regardless of `compiler.disco.active_target`
  (both a Windows-target and a non-Windows-target Compiler instance).
- New assertion with two independent non-trivial globals confirms both
  init functions are called from __metalpy_init() (order not asserted
  beyond "both present," per the documented ordering limitation above).
- Upgrade the existing "final milestone" real-compile test
  (EmitGlobalRCClassRealCompileTests.test_non_trivial_global_compiles)
  from compile-only to compile-AND-RUN: construct `g_foo: Foo =
  Foo.make(1)` (a fixture whose constructor sets an observable field),
  then in `main()` read that field back and return non-zero if it
  doesn't match the value the constructor should have set - the real
  proof that __metalpy_init() ran the constructor before main()'s own
  body executed, not just that the generated C happens to compile.
- Full `python3 tests.py` green, baseline count recorded when work
  starts.

STATUS: done. Implemented largely as designed (Implementation steps 1-4), with
one addition found during real-build verification (not anticipated by this
plan's own "Current state"/"Scope" sections): a global whose non-trivial
initializer is nonetheless a pure, all-zero value-type construction (e.g.
lib/builtins/__init__.py's own case_folder: CaseFolding = CaseFolding(
upper_table=None, upper_count=0, ...) - a CStruct, not an RCClass) must have
its own separate init function suppressed ENTIRELY (not merely left uncalled)
- an uncalled-but-still-emitted function is not a safe no-op on a no-CRT
target: it still gets compiled, and its own struct-copy-of-an-all-zero-
compound-literal is exactly the shape a C compiler is free to lower into a
real memset/memcpy call, which a no-CRT build has no implementation for.
Confirmed via a real `mpy test_hello.py --release && test_hello` failure
(LNK2019 "unresolved external symbol memset", disassembled directly to a
`callq memset` inside the dead function) - see emitter_c.py's
_global_init_is_all_zero_value_type. A second real-build-only finding: on
Windows with no_crt, mainCRTStartup already calls __metalpy_init() once
before calling main() - main()'s own unconditional prepend (this plan's step
3d) would call it a SECOND time, harmless back when __metalpy_init() only
ever did SetConsoleOutputCP (idempotent), but a real double-construction bug
now that it also builds RCClass globals - guarded with a windows_no_crt
check. Full python3 tests.py green on both Windows and Linux/gcc (905/905
each) after these fixes, plus a real end-to-end verification: `mpy
test_hello.py --release && test_hello` now builds, links, and prints
correctly on Windows (previously failed to even compile - "use of
undeclared identifier 'sys$_Stdout$$vtable'" - a separate, pre-existing pass-
3 ordering bug this work also surfaced and fixed: a global's own init
function could reference an RCClass vtable instance emitted later in the
same translation unit; the vtable-instance loops now run before the globals
loop).

UPDATE (cross-global dependency ordering, and a separate global-construction
crash - both found via targeted testing, not code review): the plan's own
"Deferred: true dependency-ordering between globals" limitation above was
flagged as unverified, not confirmed either way. Built two real tests to
settle it:

1. Circular imports + one global's constructor building an instance of
   ANOTHER (circularly-importing) module's class (`f2: foo2.Foo2 =
   foo2.Foo2()` in foo1.py, `f1: foo1.Foo1 = foo1.Foo1()` in foo2.py, each
   module importing the other) - confirmed fine: compiles, links, and runs
   correctly. Discovery's two-phase design (eager name registration,
   deferred resolution) already handles this with no changes needed.
2. A SINGLE global's own constructor building sibling classes, one of them
   (Foo2) declared textually AFTER both the class that uses it and the
   global itself (`bar: Bar = Bar()`, Bar.__init__ builds Foo1() then
   Foo2()) - this one immediately crashed lowering.py's own _try_lower_
   construct_call assert ("... was not resolved before construction"). Root
   cause: an ordinary function body's construction call always gets pre-
   resolved by TypeResolver.resolve_function_body's _ReferenceResolver
   pass (item 3's Call-visiting branch specifically eagerly resolves a bare
   ClassName(...) construction target's own __init__ signature) BEFORE
   lower_function ever runs - Compiler._lower's Function branch calls it
   explicitly. A global's own initializer never got the same treatment:
   Compiler._lower's Variable branch went straight to lower_global. This
   means ANY global constructed via a bare `ClassName()` (going through a
   real __init__, unlike `ClassName.make(...)`'s own staticmethod path -
   which this plan's own original verification fixture exclusively used,
   so it never caught this) was completely broken, unconditionally - not
   an edge case. Fixed: TypeResolver.resolve_global_init (type_resolver.py)
   - the same _ReferenceResolver pass, adapted for a bare expression instead
   of a statement list (_ReferenceResolver.__init__ now accepts fn=None for
   this case - no parameters/self to seed, mirroring lowering.py's own
   FunctionLowering(self, None) convention for global bodies) - called from
   Compiler._lower's Variable branch before lower_global.

With that crash fixed, a THIRD, genuinely real ordering bug surfaced
immediately: `b: Foo = Foo.make(a.x)` (b's own initializer reads global a's
value) with main() only ever referencing `b` directly schedules b BEFORE a
(main reaches b first in TypeResolver's FIFO queue; a is only discovered as
b's own dependency, mid-lowering, so it's enqueued and lowered strictly
later) - compiler.globals ends up `[b, a]`. This broke TWO things, one
louder than originally predicted:
- emit_c's old single globals loop (declaration + init function as one
  unit per global) emitted b's init function BEFORE a's declaration even
  existed - a straight C "use of undeclared identifier '__main__$a'"
  compile error (confirmed via a real clang -fsyntax-only run), not the
  silent null-pointer-runtime-read originally guessed.
- __metalpy_init() itself would have called b's init before a's even once
  the declaration-ordering half was fixed - a real uninitialized-value read
  at runtime.

Fixed with two independent changes in emitter_c.py: (1) emit_global split
into _emit_global_declaration/_emit_global_init_fn - emit_c now emits EVERY
global's declaration before ANY global's own init function body, which
needs no dependency ordering at all (every declaration is a self-contained
{0}-or-constant, never referencing another global's value); (2) a new
_topologically_sort_globals (Kahn's algorithm, seeded in compiler.globals'
own original order for determinism among unrelated globals), which walks
each global's own init instructions generically (via dataclasses.fields(),
so no per-ir.Instruction-subclass enumeration needed) for references to
OTHER globals, and orders __metalpy_init()'s own call sequence by real
dependency rather than compiler.globals' scheduling order. A genuine CYCLE
(two globals whose own initializers each read the other's value -
fundamentally unorderable, unlike a circular import or circular class
reference, both confirmed fine above) is detected structurally (leftover
non-zero-indegree nodes) and reported as a clean, located CompileError via
compiler.disco.fail_loc, not a hang or an arbitrary silently-wrong order.

Verified: emitter_c_test.py gained GlobalInitOrderingRealCompileTests.
test_global_initializer_reading_another_globals_value_runs_in_dependency_
order (real compile+link+run, asserts the b-before-a scheduling-order
fixture assumption explicitly so the test can't silently stop exercising
the fix) and GlobalInitCycleDetectionTests.test_circular_global_value_
dependency_is_a_clean_compile_error. Full python3 tests.py green
throughout (908/908), no regressions.
