Callable[...] function values

Why

dict[K,V]'s design needs the shared, non-generic RawDict to compare/hash
keys of whatever concrete K a given dict[K,V] instantiation uses, without
RawDict itself ever knowing what K is (RawDict must never branch on RC-ness
or call type-specific code - see PLAN_LIST_T.md's RawList for the established
"compile once, share across every instantiation" pattern this is meant to
follow). The original lib/builtins/__dict.py reached for a fictional
K.__eq_fn__ called through a ConstPtr[None] - neither exists. This compiler
currently has no first-class function values at all: referencing a plain
function by name (`f = some_function`) is rejected outright ('some_function'
is not a value), and Callable[...] (already written into lib/bisect.py's
key parameter, and needed by lib/zoneinfo.py's lambda) resolves to nothing.

Scope for this pass

In scope:
1. Callable[[ArgType, ...], RetType] resolves as a real type.
2. A bare reference to an already-resolved, non-generic free function or
   @staticmethod (no bound receiver) lowers to a Ptr[Callable[...]]-typed
   value.
3. Calling through a Ptr[Callable[...]]-typed value (an indirect call).
4. Using Ptr[Callable[...]] as a function parameter type, a local
   variable type, a struct/RCClass field, and (both closed in later
   passes - see status updates below) a function's own return type and
   a call directly through a field-access expression. Not yet deeply
   nested (e.g. inside another container type).

Deferred (flagged, not attempted this pass):
- Lambda expressions / nested function defs - needed for zoneinfo.py's
  `lambda tran: tran.timestamp` and general ergonomics. A natural follow-on
  once function-references-as-values exist (a non-capturing lambda is sugar
  for "synthesize a top-level function, reference it here"), but still new
  AST surface (ast.Lambda/nested ast.FunctionDef, name synthesis).
- Real closures (captured variables) - needs a representation decision
  (heap-allocated env struct + RC vs. borrowed fat pointer), no forcing use
  case yet.
- Protocol (structural interface typing) - a different, bigger feature;
  nothing in this pass needs it.
- Referencing a bound instance method as a value (needs the receiver
  bundled in - a closure in miniature).

Precedent reused, not invented fresh

- emitter_c.py's _vtable_slot_c_type/emit_interface_vtable_instance already
  build `(RetType (*)(ParamTypes))mangled_name` function-pointer casts for
  @interface vtable dispatch - the exact shape a bare function reference
  needs. This pass reuses that pattern (factored into a shared helper)
  rather than inventing a second one.
- mpy_types.Function already carries parameters/return_type - no new
  per-function metadata needed.
- The _stmt_Assign fix for `obj[i] = v` dispatching to a real __setitem__
  via _find_method is the model for "recognize a shape by type, branch
  lowering accordingly", reused here for indirect calls.

Implementation

1. mpy_types.py: new CallableType(Type) - arg_types: list[Type],
   return_type: Type. Not a ScopeMixin (no members of its own).
2. discovery.py: _get_or_create_callable_type (interning cache, mirrors
   _get_or_create_specialization/_get_or_create_union/_get_or_create_move).
   visit_Subscript gets a special case (ahead of the generic-subscript
   path, same spot as the existing move[T]/copy[T] special case) for the
   Callable[[...], R] AST shape: ast.Subscript(Name('Callable'),
   Tuple([List([...arg annotations...]), ret annotation])).
3. ir.py: FunctionRef(Value): fn: Function, added to the Operand union.
   CallIndirect(Instruction): dest, target: Operand, args: list[Operand].
4. lowering.py: _expr_Name allows a Function reference (not just Variable)
   when it's a plain, non-generic, non-overloaded, receiver-less function -
   produces a FunctionRef operand typed Ptr[CallableType(...)]. _expr_Call/
   _lower_call recognizes a Ptr[CallableType]-typed callee and emits
   ir.CallIndirect instead of resolving a named Function/method.
5. emitter_c.py: FunctionRef emits the vtable-style cast expression (shared
   helper extracted from emit_interface_vtable_instance). CallIndirect
   emits `(target)(args...)` - no explicit deref needed, same as an
   existing vtable slot call. Ptr[Callable[...]] parameter/local
   declarations reuse the same `RetType (*name)(ParamTypes)` shape
   _vtable_slot_c_type already produces for a struct field.
6. Once landed: rework dict[K,V] (lib/builtins/__init__.py) / RawDict
   (lib/builtins/__RawDict.py) to use real Ptr[Callable[[Ptr[None],
   Ptr[None]],bool]] eq_fn/hash_fn parameters instead of the fictional
   K.__eq_fn__/reinterpret_cast, and fix the separately-known bugs
   alongside it: missing KeyError, RawEntry storing dangling caller-owned
   pointers instead of owned copies/increfed handles, the update-path
   silently discarding its write, missing dict.__del__.

Verification

- lowering_test.py: bare function reference -> FunctionRef with the right
  Ptr[CallableType] type; call through a Ptr[Callable]-typed parameter emits
  CallIndirect; referencing a generic/overloaded/bound-method function is
  still rejected.
- emitter_c_test.py real compile-and-run: a free function's address taken
  and called indirectly through a Ptr[Callable[...]] parameter, round-
  tripping a real value; bisect.py itself compiles (it already declares
  exactly this shape).
- Full python tests.py green after each stage (Callable machinery first,
  dict rewrite second), not landed as one big change.
- dict[K,V] real compile-and-run coverage once rebuilt: dict[str,i32] and
  dict[i32,str] (RC and non-RC key) - insert, overwrite-existing-key,
  lookup-miss (KeyError), destruction without leak/double-free.

Status update: a function returning Ptr[Callable[...]] (item 4's original
exclusion, above) turned out to need no new machinery at all -
_function_prototype's own "TYPE NAME" spelling had the exact same bug
_emit_global_declaration independently hit for module-level Ptr[Callable[...]]
globals (see git history around commit 32a5c90): a bare c_type(...) prefix
can't express C's function-pointer declarator, which puts the name INSIDE
the parens. Fixed by reusing _declarator as-is - passing "name( params )" as
its own `name` argument nests the two declarator layers correctly
(`RetType (*name(Params))(InnerParams)`), with no separate code path needed.
Verified: free function, method (self ordering unaffected), and a generic
function monomorphized to K = Ptr[Callable[...]] - real compile-and-run,
MSVC/clang/WSL-gcc all green.

Status update: storing Ptr[Callable[...]] as a plain struct/RCClass field
(declaration/construction/read into a local) already worked - both
_struct_or_union_body and emit_rcclass route every field through
_declarator, same as any parameter/local. The real remaining gap was
CALLING directly through the field-access expression itself (`o.field(...)`)
- _try_lower_indirect_call was deliberately scoped to a bare Name callee
only, per this doc's own original item 4 wording. Closed by extending it to
also recognize an Attribute callee, using a purely static, non-emitting
type lookup (_static_type_of_value_expr / a new non-failing _find_field
probe) to decide the shape applies BEFORE ever lowering the receiver -
required because _resolve_callee's own Attribute fallback lowers the
receiver again on any non-match, so lowering it speculatively here first
would double-evaluate a receiver with side effects. Verified: @cstruct and
RCClass fields, a nested field chain (outer.inner.handler(...)), and a
regression guard that an ordinary same-shaped method call (o.method(...))
still dispatches normally rather than being misrouted - real compile-and-
run, MSVC/clang/WSL-gcc all green, plus lowering_test.py IR-level coverage.

Status update: the LAST remaining callee shape - calling directly through
ANY other expression (get_callback()(...), t[0](...)) - is now closed too.
Considered and rejected: retrofitting CallableType with a real `__call__`
dunder so it could ride the ordinary class-method dispatch machinery every
other call in the language already flows through (the user's own
suggestion, worth recording) - CallableType isn't a ScopeMixin (no members
at all, see Implementation #1 above), so this would mean teaching the
SHARED method-resolution/dispatch core to recognize a synthetic member on a
non-class type, a real but more invasive change than warranted here.
Instead, _try_lower_indirect_call's existing Name/Attribute branches (each
statically type-checked before ever evaluating anything, to stay double-
evaluation-safe against _resolve_callee's own fallback) gained one more,
simpler branch: for any OTHER node.func shape, evaluate it once via the
ordinary _lower_expr and inspect the REAL resulting operand's type. This is
double-evaluation-safe too, for a different reason than the static-check
branches - _resolve_callee's own fallback for a non-Attribute/non-Name
func_node fails IMMEDIATELY today, with zero evaluation attempted, so
nothing downstream ever gets a second chance at the same expression.
One real regression found and fixed while building this: a bare Subscript
node.func isn't always "index a runtime value" - `some_generic_fn[T](...)`/
`compiler.atomic_add[T](...)` is NAMESPACE-RESOLVED generic-call syntax
(the exact same _try_resolve_namespace lookup the construction-sugar
recognizers above already use), and evaluating it as an ordinary expression
broke 11 existing tests (generic-function-call and compiler-intrinsic
tests, plus one exercising `sys` used bare). Fixed by trying
_try_resolve_namespace(node.func) first (a purely static, non-evaluating
lookup) and bailing out untouched whenever it resolves to anything,
BEFORE ever calling _lower_expr - leaving that whole class of call
completely unaffected by this change.
Verified: a call-result callee (get_callback()(...)) and a subscript-result
callee via a custom __getitem__ (isolated from whatever separate, unrelated
gaps a generic container's OWN internals might still have storing a
Ptr[Callable[...]] element - never investigated), plus a regression guard
that a genuinely non-callable call-result still fails with the ordinary
"cannot call ..." diagnostic. Real compile-and-run plus lowering_test.py
IR-level coverage (including the generic-call/compiler-intrinsic
regression tests that caught the Subscript bug), MSVC/clang/WSL-gcc all
green, full test suite clean on all three.
