Return-only generic type-parameter inference (eager body pre-compile)

Why

Looking at `builtins.len[T](t: T) -> usize: return t.__len__()` (now
@inline, see PLAN_INLINE.md), the return type is hard-coded rather than
derived - it happens to always be `usize` today (every `__len__` in the
codebase returns `usize`, confirmed by exhaustive search, no exceptions -
deliberately, per the user), but nothing actually forces that; it's just
true by coincidence. The real question generalizes past `len()` entirely:
any generic function whose return type depends on what its own body does
with an already-bound type parameter, e.g.

	def make[T,R]( factory: T ) -> R:
		obj = factory.create()
		# ... common initialization on obj, regardless of concrete type ...
		return obj

Here `R` never appears in any PARAMETER position - only `factory`'s type
(T) does. Every inference path this compiler had (`_lower_inferred_generic_
call`'s interleaved argument unification, and its discovery-phase clone in
type_resolver.py) was argument-driven only: it walks declared parameter
types against lowered argument types and never once consults `target.
return_type`. A type param that appears only in the return position was
therefore unconditionally rejected ("cannot infer type parameter(s) R ...
call it explicitly"), even though the real answer is sitting right there in
the body once T is known.

The user's own framing: "since lambdas can do it, I think it makes sense
that it could work for any generic function." Correct, and the core
insight this plan builds on - PLAN_LAMBDA.md's "eager lambda lowering"
already solves a narrower version of exactly this problem (a lambda
argument's own unbound Callable[...] return type, forced concrete by
lowering the lambda's body early and reading the real type back off its
ir.Return). This generalizes that same trick from "an unbound Callable's
own return type" to "a named generic function's own return type."

Scope for this pass

1. A BARE call to a generic function (`make(f)`, no explicit `[T,R]`),
   where after ordinary argument-driven unification some type params
   remain unbound AND every one of them appears only in `target.
   return_type` (never in any `param.type`) - infer them by eagerly,
   synchronously lowering the function's own body (with the other,
   argument-bound type params already substituted) and reading the real
   type off its single `return <expr>`.
2. Both @inline and non-@inline targets. These need two different
   mechanisms (see Implementation) because @inline's own invariant -
   "never a real compiled unit" (PLAN_INLINE.md) - must keep holding. The
   user's own forcing example (`make[T,R]`, multi-statement) is NOT
   @inline-eligible at all, so covering only the cheap @inline path would
   leave the actual motivating case unaddressed.
3. A body-shape restriction for the eager-compile path: the function's
   body, walked as a whole (not just top-level, unlike @inline's
   docstring-strip-then-one-statement check) but not descending into a
   further nested def/lambda, must contain EXACTLY ONE `ast.Return` node,
   with a value. This sidesteps "do all return points agree on the same
   type" entirely - there's only ever one to agree with. Multi-statement
   bodies with locals/branches/loops are fine; multiple RETURN POINTS are
   not.
4. A reentrancy guard (direct or mutual recursion through another
   currently-inferring generic function before its own return type is
   known).
5. Cross-call-site memoization, so two call sites needing the same
   (T, ...) binding for the same function don't redo the eager compile (or,
   for the non-inline case, don't produce two compiled copies).

Deferred / out of scope for this pass

- Explicit partial subscript (`make[Foo](f)`, supplying T but omitting R) -
  type_resolver.py's subscript-arg-count check hard-requires the explicit
  arg count to equal the full type-param count, so no existing spelling
  reaches this gap at all.
- `_lower_class_generic_method_call` (`Result.Ok(5)`-shaped receiver-less
  construction sugar) - a structurally different inference mechanism
  (unifies expected_type against target.return_type from the ASSIGNMENT
  context, never from the callee's own body).
- A return-only type param nested inside a TupleType-shaped return position
  (`-> tuple[R,S]`) - `_unify_type_param` has no TupleType recursion branch
  today (pre-existing gap, unrelated to this plan). Falls through to the
  ordinary "cannot infer" error, not silently wrong.
- A body that needs its own Result-shaped early-exit handling (or_return(),
  checked-arithmetic propagation) before the single `return <expr>` is
  reached, while return_type is deliberately still unresolved during the
  eager pass. Not solved - flagged as a real, narrow gap.
- Interaction with @move/@classmethod/@virtual/@overload/@abstractmethod/
  @extern on a function needing return-only inference - no forcing use
  case, not exercised, not specifically rejected either.
- A return-only type param used in a local variable's own explicit
  annotation before the final return (`obj: R = ...`) - genuinely
  unresolvable at that point in the body.

Precedent reused

- `_expr_Lambda`'s eager-compile shape: `self.lowering._compile_now(
  synthetic)` (wired to Compiler._lower in compiler.py), read the real
  return type off the resulting body's ir.Return, patch it onto the SAME
  Function object afterward (never rebuilt).
- FunctionLowering being constructed fresh per lower_function/lower_global
  call, including nested/reentrant ones (PLAN_LAMBDA.md's "Reentrancy"
  section) - the eager pre-compile goes through this same path, so it gets
  a fresh FunctionLowering for free. It does NOT, by itself, bound
  recursive eager compiles - that's new state this plan adds.
- discovery._get_or_create_specialization's string-qualname interning - a
  bare TypeVar has its own real .qualname (its stem), so this cache works
  unmodified even when used as a "pending" key with R still unbound -
  avoids needing a new cache table for cross-call-site memoization.
- discovery.py's _is_inline_eligible_body - precedent for a body-shape
  validator, but this plan's own check (_is_eager_return_inferable_body)
  is lowering-time and call-site-conditional instead of parse-time-
  unconditional: no decorator here to hang an opt-in check off of, so
  validating unconditionally at definition time would risk rejecting an
  existing library function that happens to have this body shape but is
  never called through the inferring path at all.
- @inline's own _inlining_stack (on FunctionLowering) - precedent for
  shape, not placement (this plan's guard lives one level up, on the
  persistent Lowering instance).

Implementation

1. mpy_types.py - no new persistent fields. "does this type param appear
   only in the return position" is a cheap, per-compile-run derived cache,
   not schema-worthy.

2. monomorphize.py `Monomorphizer.monomorphized_function` - refactored
   (not rewritten): the substitution/copy core extracted into
   `_build_monomorphized_function(base, type_params, args, qualname,
   substituted_cls=None)`, taking those as explicit parameters instead of
   reading them off a Specialization. `monomorphized_function` becomes a
   thin wrapper computing type_params/substituted_cls (the "inherited from
   an enclosing generic class" branch stays here, only exercised by this
   caller) and delegating. Pure extraction, no behavior change for any
   existing caller - the new eager-compile path calls the extracted helper
   directly, since it doesn't have a real (fully-concrete) Specialization
   yet at that point.

3. discovery.py, new `_is_eager_return_inferable_body(body) -> bool`, next
   to `_is_inline_eligible_body`: walks the WHOLE statement tree (if/for/
   while/with/try/match bodies included), collecting every `ast.Return`
   found, but does NOT descend into a nested def/lambda/async def (mirrors
   `_reject_free_variables`'s identical discipline, PLAN_LAMBDA.md). True
   iff exactly one Return was found and it has a value. Called lazily, only
   from the eager-compile helper below - NOT wired into the decorator
   loop, NOT called unconditionally at parse time (no decorator exists
   here to hang an opt-in check off of).

4. lowering.py `Lowering.__init__` - two new fields:
   - `_eager_return_inference_stack: list[int]` - the reentrancy guard, by
     id(target) (the abstract base Function), not by which concrete args.
     Lives on the PERSISTENT Lowering instance, not FunctionLowering:
     the eager pre-compile (non-@inline case) always builds a BRAND NEW
     FunctionLowering for the nested call (PLAN_LAMBDA.md's own
     reentrancy fix), so a guard scoped to one FunctionLowering instance
     would start empty every time, invisible across exactly the boundary
     that needs guarding.
   - `_param_referenced_type_params: dict[int,frozenset[int]]` - per-
     id(target) cache of which of a function's own type params occur
     anywhere in its PARAMETER types (a static signature property).

5. lowering.py, two new helpers on `Lowering`:
   - `_type_mentions_param(t, tv) -> bool` - occurs-check, using the SAME
     structural recursion `_unify_type_param` itself uses (Specialization.
     args, CallableType.arg_types/return_type) - deliberately not the
     broader shape `substitute_type_params` uses (ClosureType/anonymous-
     TaggedUnion too): "does this type CONTAIN tv" has to agree with
     "would _unify_type_param actually BIND tv here", or a type param that's
     structurally present but never actually unified against would be
     wrongly classified as argument-inferable.
   - `_param_referenced_type_params_for(target) -> frozenset[int]` - the
     cached membership set described above, built via `_type_mentions_param`
     over every parameter type.

6. lowering.py `_lower_inferred_generic_call` - the main hook. After the
   existing interleaved argument lower/unify loop, `missing` type params
   are split instead of failing on all of them uniformly:
   - `return_only_missing`: doesn't appear in any parameter type (per the
     cache above), but does appear in `target.return_type` - eligible for
     eager inference.
   - `genuinely_missing`: everything else still unbound. Non-empty ->
     fail with the EXACT same message as before (listing every originally-
     missing name, not just the genuinely-stuck ones - no partial rescue:
     an explicit spelling would need every type param supplied anyway,
     since partial explicit subscripts aren't supported).
   - Otherwise, branch on `target.is_inline` and call one of the two new
     helpers below instead of failing.

7. lowering.py, new `FunctionLowering._infer_return_only_type_params` -
   the non-@inline variant:
   - Build `pending_args` (known args, with each return-only param passed
     through as ITSELF, still bare) and `pending_spec` via the EXISTING
     `_get_or_create_specialization` - the cross-call-site memo key. A
     memo hit (`pending_spec.monomorphized is not None`) just re-runs
     `_unify_type_param` against the already-discovered return type and
     returns the cached Function - no new compile.
   - On a miss: the reentrancy guard, then the body-shape check, then
     build a provisional Function via `_build_monomorphized_function`
     (qualname = the pending spec's own, already unambiguous), with
     `.return_type` forced to None (mirrors `_expr_Lambda`'s
     `return_type_provisional` convention exactly - lets `_stmt_Return`
     lower the return expression with no hint, taking its own natural
     type).
   - `self.lowering._compile_now(provisional)` - this IS the final
     compiled unit (appended to compiler.functions by Compiler._lower's
     plain-Function branch), never rebuilt.
   - Read the real return type off the resulting body's one ir.Return,
     unify it against the abstract `target.return_type` to fill in the
     return-only bindings (also naturally catches a genuine structural
     mismatch via the same helper's existing conflict error).
   - Patch `provisional.return_type`/`.qualname` in place afterward
     (substituted-canonical form, not the raw IR operand type - so a
     Specialization-shaped return type is the same interned object
     anything else referencing it would get), register it under BOTH the
     real (fully concrete) and pending Specialization keys, return it.

8. lowering.py, new `FunctionLowering._infer_return_only_type_params_
   inline` - the @inline variant, cheaper because splicing the body IS the
   eager compile already:
   - Same pending-spec memo lookup. A hit just delegates straight to the
     EXISTING, unchanged `_lower_inline_call` against the cached
     provisional - identical to how an ordinary, already-fully-inferred
     @inline call already works.
   - On a miss: reentrancy guard, build the provisional the same way, but
     only run `resolve_function_body` on it (no `_compile_now` - @inline
     never needs a real compiled unit for its own target, only the
     AST-level generic-call-resolution rewrite a NESTED generic call
     inside the body would need). `_is_eager_return_inferable_body` is NOT
     re-checked here - @inline's own decorator-time `_is_inline_eligible_
     body` (exactly one TOP-LEVEL `return <expr>`) is strictly stronger,
     every @inline-eligible body already satisfies it trivially.
   - Calls the existing `_lower_inline_call` against the provisional,
     forcing `want_result=True` internally (the real operand - and its
     .type - is needed to discover the return-only bindings even when the
     caller's own want_result is False; target.return_type is still None
     at that point so `_lower_inline_call`'s own discard-check is a no-op
     either way) - reads the return type off the result operand's own
     .type instead of an ir.Return. Same unify/patch/dual-key-cache tail
     as the non-inline variant, plus the discard-Result check applied
     AFTERWARD (once the real type is known) if the caller's real
     want_result was False.

9. lowering.py `_emit_generic_call` - new `already_compiled: bool = False`
   keyword parameter; when True (passed only from the non-@inline eager
   path's own call site), skips `self.lowering.schedule(spec)` specifically
   (keeps the return/parameter-type scheduling, which is idempotent). See
   STATUS below for why this was a real, not just theoretical, fix.

10. lib/builtins/__init__.py - unchanged. `len[T]`'s `-> usize` stays as
    is (already correct); dropping it in favor of inferred R would be
    purely cosmetic.

STATUS: landed. Two real bugs found during implementation, both fixed,
neither anticipated by the original design:

1. `_expr_Lambda`'s own `next(instr for instr in lowered.instructions if
   isinstance(instr, ir.Return))` (no default) was the template for reading
   the real return type back - reused as-is in a first draft of
   `_infer_return_only_type_params`. This crashes with an unhandled
   `StopIteration`, not a clean compile error, whenever the ONE statement
   in the eagerly-compiled body itself fails to lower (confirmed via a real
   repro: direct/mutual recursion through this same inference, caught
   correctly by the reentrancy guard one level deeper - but
   `FunctionLowering.run()`'s own per-statement recovery, `try: self.
   _lower_stmt(stmt) except CompileError: continue`, SWALLOWS that failure
   silently rather than propagating it up through `_compile_now`, leaving
   `lowered.instructions` with FuncStart/FuncEnd but no ir.Return at all).
   Fixed with `next(..., None)` plus an explicit `fail()` when None - a
   second, redundant-but-harmless error report, the same accepted pattern
   `resolve_function_body`'s own docstring already documents ("gets
   reported again... so nothing is silently swallowed"). `_expr_Lambda`'s
   own identical-shaped `next(...)` call has the exact same latent crash
   risk (a lambda body that itself fails to lower) - NOT fixed here, out of
   scope for this plan, flagged separately.

2. `_lower_call`'s shared, non-generic tail already special-cased
   `is_inline` in `_resolve_call_target` to skip `_ensure_resolved`'s own
   scheduling side effect - but `_emit_generic_call` (the tail this plan's
   OWN non-inline eager path also reaches, once R is known) had no
   equivalent guard: its unconditional `self.lowering.schedule(spec)` would
   re-enqueue an already eagerly-compiled Function, and `Compiler._lower`'s
   Specialization+Function branch has no "already lowered" check of its
   own - it would unconditionally re-run `resolve_function_body` (which
   mutates `.node.body` in place - a second pass over an already-rewritten
   body) and `lower_function` a second time, producing a duplicate
   `LoweredFunction` entry for the same qualname (a real duplicate-symbol C
   compile error). Fixed with the `already_compiled` parameter (Implementation
   #9) - found by reasoning through the scheduling path before it ever
   actually reproduced as a test failure, unlike bug #1 above.

Also found and worked around, NOT fixed here (out of scope for this pass,
flagged separately via spawn_task - since fixed, see below): constructing a
generic `@cstruct` with no `__init__` (bare `ClassName(field=value, ...)`
sugar) never infers the class's own concrete type args from the field
VALUES themselves - only from the surrounding `expected_type` context (an
explicit annotation, say). `_lower_allocate_fields`'s own `dest =
self._new_temp(expected_type or target_cls)` falls back to the bare
ABSTRACT class when no such context is available - exactly the situation
inside an eagerly-inferred body, whose own `return_type` is deliberately
still None. A generic RCClass WITH a real `__init__` doesn't have this gap
(`_lower_generic_construction_args` does real argument-based inference
there, independent of `expected_type`) - lowering_test.py's own "mixed"
test used that shape instead, with a comment explaining why.

Fixed in a follow-up pass: `_lower_allocate_fields` now infers via a new
`_infer_allocate_type_args` helper, unifying each field's declared type
against that field's own real lowered value type (same `_unify_type_param`
every other generic call site uses), falling back to a clear compile error
when a type param is genuinely unresolvable either way. See
lowering_test.py's `test_bare_construct_infers_type_args_from_field_values_with_no_expected_type`,
`test_allocate_dest_type_infers_type_args_when_no_expected_type`,
`test_allocate_fails_loudly_when_type_args_unresolvable`, and
`test_lambda_eager_lowering_infers_generic_no_init_construction_type_args`
(the identical exposure in `_expr_Lambda`'s own eager-lowering branch,
fixed for free since both route through the same shared function). Chasing
this down also surfaced one more real, pre-existing, unrelated bug it
depended on: `_lower_call`'s own plain-call dest computation
(`expected_type or target_return_type`) blindly trusted a bare, unbound
TypeVar `expected_type` hint over the callee's own concrete, resolved
return type - fixed alongside it.

Verification

- lowering_test.py `ReturnOnlyTypeParamInferenceTests` (12 tests): basic
  inference from a single-statement body; the user's own multi-statement
  shape (locals/branches before the one return); a return type mixing an
  already-argument-bound param with a return-only one (`Pair[T,R]`,
  exercising `_unify_type_param`'s Specialization-args recursion); two
  return points rejected; a bare `return` rejected; direct recursion
  rejected; mutual recursion rejected; two call sites with the same
  concrete T reuse one compiled Function (`is` identity); a type param in
  neither any parameter nor the return type still fails with the existing
  error. Separately for @inline: R inferred and still no real Call/
  FuncStart/FuncEnd for the target itself; two @inline call sites each get
  their own independent splice; discarding a Result-shaped inferred return
  is rejected.
- discovery_test.py `EagerReturnInferableBodyTests` (10 tests):
  `_is_eager_return_inferable_body` exercised directly against bare ASTs -
  accepts a single return nested in if/for/while/with/try/match; rejects
  zero, two, or a bare return; a return inside a nested def/lambda doesn't
  count toward the outer function's own total.
- return_inference_test.py (2 tests, real compile+link+run, skipped when no
  C compiler is found): two differently-typed concrete instantiations both
  produce the correct runtime value and the correct, concrete C return
  type; two call sites with the same concrete T compile to exactly one C
  function, not two.
- Full `python tests.py` green throughout (978 passing).
