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

Known limitation (found during implementation, not fixed): a lambda body
that does checked arithmetic (+, -, *) under the default Check mode fails,
since Check mode needs either `with compiler.panic_arithmetic(...):`
around the expression or a Result-returning enclosing function to
propagate into - and lambda syntax forbids `with`/statements entirely, so
there is no way to satisfy either from inside a lambda body today. Doesn't
block the forcing use case (`lambda tran: tran.timestamp` does no
arithmetic at all) - noted for whoever picks up real closures/lambda
ergonomics next.

STATUS: landed and tested for the scoped case (a lambda/nested-def
referenced where the surrounding context already supplies a concrete
Ptr[Callable[[ArgTypes],Ret]] type - a typed parameter/local, or a call to
a non-generic function). The stated forcing use case
(lib/zoneinfo.py's own `bisect_right(self.transitions, timestamp, key =
lambda tran: tran.timestamp)`) turned out to need MORE than this pass
scoped, and is explicitly NOT done - see "found but not fixed" below.
Decided with the user: stop here rather than expand scope further right
now.

Found but not fixed - genuinely separate, larger work:
- bisect_right[T,K]/bisect_left[T,K] are themselves generic, and key's
  own declared type (Callable[[T],K]) still has an unresolved K at the
  point _expr_Lambda needs a concrete CallableType to infer the lambda's
  parameter types from - K is only knowable by lowering the lambda's OWN
  body first (its inferred return type), which the current generic-call-
  argument machinery doesn't do (it assumes every parameter's expected
  type is fully concrete before any argument gets lowered). Confirmed via
  a minimal, zoneinfo.py-independent repro (a generic function taking a
  Callable[[T],K] parameter, called with a lambda). Real closures over a
  captured environment are unrelated - this is about a genuine circular
  type-inference dependency, "the callee needs the lambda's type before
  the lambda can be given a type."
- Separately, and unrelated to lambdas: passing list[T] where bisect.py
  declares slice[T] doesn't infer T either (found earlier this session,
  independent of PLAN_LAMBDA.md's own work) - another thing sitting
  between here and zoneinfo.py actually compiling.
- Whoever picks this up next: either extend generic-call argument
  inference to lower a Callable-typed argument's lambda body first and
  feed its inferred type back into the callee's own type-parameter
  binding (bigger, compiler-side), or reconsider bisect.py's own key=
  design to sidestep the inference order problem (e.g. require an
  already- concretely-typed Ptr[Callable[...]] value at the call site
  instead of inferring one from a bare lambda) - library-side, smaller,
  but changes bisect.py's own ergonomics.

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
  via bare reference/indirect call; a lambda's parameter types inferred
  from a concrete Ptr[Callable[...]] context, called indirectly. NOT
  covered (see "found but not fixed" above): lambda passed as key= to a
  GENERIC bisect_right/bisect_left, and lib/zoneinfo.py's own
  get_ttinfo/utcoffset/abbr - both need the further generic-inference
  work this pass stopped short of.
- Full python tests.py green throughout (746 passing); nested-def support
  committed first, lambda support second.
