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

STATUS: landed and tested for the originally scoped case (a lambda/
nested-def referenced where the surrounding context already supplies a
concrete Ptr[Callable[[ArgTypes],Ret]] type - a typed parameter/local, or
a call to a non-generic function), AND for generic type-param inference
through Callable[...] (both a plain function-reference argument and a
lambda argument, including the eager-lowering case a lambda's own
unbound return type needs - see the two follow-ups below). The stated
forcing use case (lib/zoneinfo.py's own `bisect_right(self.transitions,
timestamp, key = lambda tran: tran.timestamp)`) itself is still not
attempted end to end - the one remaining known gap is the separate,
unrelated list[T]-vs-slice[T] inference issue noted at the bottom.

Follow-up done: substitute_type_params/_unify_type_param stopped
recursing at a CallableType entirely (it's not a Specialization, so the
existing Ptr[T]-vs-Ptr[i32] recursion never looked inside a
Ptr[Callable[[T],K]] parameter's own arg_types/return_type) - this broke
generic type-param inference through ANY Callable[...]-typed parameter,
even for a plain function-reference argument with no lambda involved at
all (confirmed: def apply[T,K](x: T, key: Ptr[Callable[[T],K]]) -> K,
called as apply(5, key=identity_i32), failed to infer K). Fixed - both
functions now recurse into CallableType the same way they already do for
Specialization, rebuilding through the existing
_get_or_create_callable_type interning.

Follow-up done (eager lambda lowering): the LAMBDA-argument case above is
now fixed. When _expr_Lambda's own expected Callable[...] type has a
concrete arg_types but a still-unbound (bare TypeVar) return_type - e.g.
bisect_right[T,K]/bisect_left[T,K]'s key: Callable[[T],K], with T already
known but K only knowable from the lambda's own body - the lambda's body
is now lowered EAGERLY, synchronously, right at the call site (via a new
Compiler._lower backreference, Lowering._compile_now, threaded in
Compiler.__init__) instead of only ever being scheduled onto the work
queue for later. The real inferred return type is read off the
synthesized body's own ir.Return afterward and patched onto the
synthetic Function before building its FunctionRef - the CallableType
recursion in _unify_type_param/substitute_type_params (above) then binds
the caller's own K from that unchanged.

This needed one more fix alongside it, found while testing the actual
apply(5, key=lambda v: v) repro: _lower_inferred_generic_call (a BARE
call to a generic function, e.g. apply(...) with no explicit [T,K]) used
to lower EVERY argument with no expected-type hint at all, then unify
all of them against the callee's declared parameter types only
afterward - so by the time the key= argument's lambda was lowered, T
(bindable from the earlier x=5 argument) was never actually visible to
it. Fixed by interleaving: each argument is now lowered in turn with a
hint built from whatever earlier arguments in the SAME call already
bound, then immediately unified to refine those bindings before the next
argument - letting a later Callable[...]-typed argument's own lambda see
an earlier argument's inferred type params. A bare, still-fully-unbound
type param substitutes to itself with no other information to offer, so
that's now folded back to "no hint" explicitly (a literal argument at
such a position still correctly fails via _expr_Constant's own "cannot
infer" error, unchanged).

Reentrancy (lowering a lambda's body while the ENCLOSING function's own
body is still mid-lowering, exactly what eager lowering needs to do) is
handled by moving all of Lowering's PER-FUNCTION state (the instruction
stream being built, temp/label counters, current CFG, defer/construction
bookkeeping - everything _init_lowering_state used to (re)set on the
shared instance) into a brand-new FunctionLowering class, constructed
FRESH for every lower_function/lower_global call, including a nested one
- so an outer and inner lowering never share mutable state, and there's
no field list to keep in sync by hand the way an explicit save/restore
around a single shared instance would need (the approach originally
proposed, then redirected by the user toward this structural fix
instead, before any code was written for it). Lowering itself keeps only
the genuinely persistent, cross-compile-run state (discovery, the type
resolver, schedule, shared UnionStorage/Monomorphizer, closure-trampoline
cache, lambda counter) and becomes a thin `FunctionLowering(self, fn)
.run()` entry point. Verified with a doubly-nested case (a lambda whose
own body eagerly lowers ANOTHER lambda argument, two FunctionLowering
instances deep) - no counter collisions, both register correctly into
compiler.functions.

Separately, and unrelated to lambdas: passing list[T] where bisect.py
declares slice[T] still doesn't infer T (found earlier this session,
independent of PLAN_LAMBDA.md's own work) - still the one remaining
thing sitting between here and zoneinfo.py/bisect.py actually compiling
end to end; explicitly not attempted in this pass.

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
