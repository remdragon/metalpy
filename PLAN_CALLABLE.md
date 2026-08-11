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
4. Using Ptr[Callable[...]] as a function parameter type and a local
   variable type - not yet as a struct field, not deeply nested, and NOT
   as a function's own return type (confirmed: crashes today - a function
   RETURNING a function pointer is C's gnarliest declarator shape,
   `RetType (*name(Params))(InnerParams)`, genuinely different from every
   other declarator _declarator handles. Not needed by dict[K,V] - it only
   ever passes a callback as a parameter, never returns one).

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
