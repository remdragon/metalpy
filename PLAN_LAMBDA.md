Non-capturing nested function defs and lambdas

Why

PLAN_CALLABLE.md (Callable[[Args],Ret], Ptr[Callable[...]] values, bare
function references, indirect calls) deliberately deferred lambdas and
nested function defs - dict[K,V]'s own need (a monomorphized @staticmethod
comparator) never required them. Two real library files still can't
compile without this: lib/bisect.py's key: Callable[[T],K]|None = None
parameter has no way for a caller to actually supply a matching value
without either a named top-level function or a lambda, and
lib/zoneinfo.py's own get_ttinfo already calls bisect_right(self.
transitions, timestamp, key = lambda tran: tran.timestamp) - a lambda
expression, which hits lowering.py's generic "unsupported expression"
fallback today (no _expr_Lambda exists at all). This is the concrete
forcing use case.

Scope for this pass

In scope:
1. Non-capturing nested function defs (def inner(...): ... inside another
   function's body) - synthesized as an independent, fully real Function
   (real parameter/return annotations, resolved normally), scheduled and
   compiled like a top-level function, callable/referenceable by name for
   the rest of the enclosing function's own body (ordinary calls AND a
   bare reference per PLAN_CALLABLE.md's FunctionRef).
2. Non-capturing lambda expressions (lambda args: expr) - synthesized the
   same way, except lambda syntax carries no type annotations, so
   parameter types are inferred from the surrounding expected_type context
   (must already be a Ptr[Callable[[ArgTypes],Ret]] shape).
3. A shared no-capture check: a nested def/lambda may only reference its
   own parameters/locals, module-level names, and builtins - referencing
   anything from the immediately enclosing function's own scope is
   rejected with a clear error, not silently miscompiled.

Deferred / out of scope:
- Real closures (captured variables) - still no forcing use case; needs a
  representation decision (heap env + RC vs. borrowed fat pointer).
- A nested def/lambda inside a GENERIC function or a generic class's own
  method - rejected outright (sidesteps "which monomorphization does this
  belong to" entirely).
- Multiple levels of nesting - not specifically tested or blocked.
- Protocol - unrelated, still not needed by anything.

Precedent reused

- lowering.py's _stmt_AnnAssign already calls self.discovery.visit(node.
  annotation) mid-lowering to resolve an ordinary local variable's type
  annotation - the exact mechanism a nested def's own parameter/return
  annotations need.
- type_resolver.py's _synthesize_rcclass_destructor ($$__destructor__) and
  lowering.py's _get_or_create_closure_trampoline
  ($$__closure_trampoline__) are the established pattern for hand-building
  a synthetic ast.FunctionDef + Function and handing it to schedule().
  Naming: f'{enclosing.qualname}$$nested_{node.name}' for a def,
  f'{enclosing.qualname}$$lambda_{counter}' for a lambda.
- PLAN_CALLABLE.md's own _lower_function_ref/ir.FunctionRef/
  _get_or_create_callable_type/ir.CallIndirect - once a nested def/lambda
  is a real, resolved Function, referencing it bare or calling it directly
  reuses this unchanged.

Implementation

1. lowering.py: _reject_free_variables(body, enclosing_fn, own_locals,
   node) - walks the def/lambda's own body collecting free ast.Name(Load)
   references not in own_locals, fails if any resolves to something in
   enclosing_fn.names. Doesn't recurse into a further nested def/lambda.
2. lowering.py: _stmt_FunctionDef (new) - rejects a generic enclosing
   scope; resolves params/return via discovery.visit(...); runs the
   no-capture check; builds a synthetic Function, registers its name in
   the enclosing function's own scope, schedules it. No IR for the
   statement itself.
3. lowering.py: _expr_Lambda (new) - requires a Ptr[Callable[...]]
   expected_type (via type_resolver._callable_type_of) to infer parameter
   types; runs the no-capture check on the single-expression body; builds
   a synthetic Function wrapping [Return(body)]; returns a FunctionRef via
   a helper shared with _lower_function_ref.
4. Both new methods are picked up automatically by _lower_stmt/_lower_expr's
   existing dispatch-by-name convention.

Verification

- lowering_test.py: nested def called/bare-referenced from its own
  enclosing function; capture rejected; generic-enclosing-scope rejected;
  lambda parameter types inferred from a Callable[...]-typed call argument;
  lambda with no expected-Callable context rejected; lambda capture
  rejected.
- emitter_c_test.py real compile-and-run: nested def called directly and
  via bare reference/indirect call; lambda passed as key= to
  bisect_right/bisect_left producing the right answer; lib/zoneinfo.py's
  own get_ttinfo/utcoffset/abbr compiling and running correctly for real -
  the concrete forcing use case.
- Full python tests.py green throughout; nested-def support committed
  first, lambda support second.
