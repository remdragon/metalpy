@inline decorator - zero-overhead forwarding functions

Why

TODO.txt already calls this out: lib/builtins/__init__.py's `len[T](t: T) ->
usize: return t.__len__()` is a real Call/FuncStart/FuncEnd today purely to
forward to whatever T's own __len__ is - see the comment right above it
("TODO once @inline exists: this should become @inline so len(x) compiles
down to the same code as x.__len__() directly, no call overhead"). A survey
of lib/ (ast-walking every FunctionDef whose body, after stripping a leading
docstring Expr if present, is exactly one `return <expr>` statement) turns up
~50 more candidates of the same shape: FastList.len/.capacity/.id,
list.len/.capacity, str/bytes/bytearray's get_const_ptr/get_cstr/byte_size,
str's __eq__/__ne__/__lt__/__le__/__gt__/__ge__ (all `return
self.__cmp__(other) <op> 0`), int's identical comparison forwarders,
guid.__ne__ (`return not self.__eq__(other)`), threading.Lock.locked,
atomic.py's load/fetch_add/fetch_sub/exchange/compare_exchange, windows/com's
SUCCEEDED/FAILED, and (see "Deferred" below for why this one needs its own
paragraph) Result[T,E].is_ok/.is_err (`return self.tag == 0/1`) when called
through a receiver. None of these need a real call boundary - they exist
purely so callers don't have to remember which private field or lower-level
method backs a given name.

Scope for this pass

In scope:
1. `@inline` on a function/method whose body, after stripping a single
   leading docstring statement (an `ast.Expr` wrapping a string
   `ast.Constant`, same shape `ast.get_docstring` recognizes), is EXACTLY one
   `return <expr>` statement (expr not None - `return` with no value doesn't
   need inlining, it's not a builtins.len()-shaped forwarder). Anything else
   (multiple statements, a bare `return`, an `if`/loop/`with`/`match`,
   construction, `@move`/errdefer machinery) is a discovery-time error, not a
   silent fallback to a real call - matches this codebase's established
   "reject rather than guess" discipline (e.g. @virtual+@overload above).
2. Free functions (builtins.len), plain instance methods on a non-generic
   class (FastList.len, str.__eq__), and generic FREE functions
   (builtins.len[T] itself) called either bare (`len(x)`) or explicitly
   instantiated (`len[i32](x)`).
3. `@staticmethod @inline` (no `self` to bind).

Deferred / out of scope (no forcing use case yet, and each is a real
independent design question):
- A receiver-LESS call to a method whose genericity is inherited from an
  enclosing generic class, reached only through the bare-class spelling
  (`Result.Ok(y)`, `Result.Err(y)` - construction sugar, not really
  "forwarding" anyway) - this is the one path _lower_class_generic_method_
  call actually handles (its own comment: "only ever reached with NO
  receiver"), and it isn't touched by this pass.
  NOT deferred, and initially assumed to be but isn't: a RECEIVER-based call
  to that same kind of method (`some_result.is_ok()`, `.is_err()`, and any
  other single-`return <expr>` method on a generic class) - traced through
  type_resolver.py's `_attr_lookup_callable`: `ensure_resolved(owner_type)`
  on a concrete receiver type (Result[i32,MyError]) already runs
  monomorphize_class, whose method loop hands back an ALREADY-monomorphized
  Function (concrete .cls, type_params=None) - so `_lower_call`'s dispatch
  chain never routes a receiver-based call through _lower_class_generic_
  method_call at all (confirmed by that branch's own comment), it falls
  straight through to the ordinary plain final tail, the same branch #5
  below already wires up. is_ok/is_err's own `_cfg.clear_result(receiver.
  stem)` bookkeeping (~4672) also runs unconditionally BEFORE any dispatch
  branch, so it's already recorded by the time an inlined body would get
  spliced - inlining `is_ok`/`is_err` needs no extra dispatch-site work,
  just `@inline` on the two defs in lib/builtins/__init__.py once this pass
  lands. (`unwrap`/`unwrap_or`/`or_return` stay out either way - the first
  two are Overload-group stubs, the third has its own dedicated _lower_or_
  return path, both already excluded per @overload above / no forcing case.)
- `@classmethod @inline` (no `cls` substitution mechanism built).
- `@move @inline` (whole-function move semantics interacting with a removed
  call boundary - not reasoned through).
- Multi-statement bodies, control flow, defer/errdefer, construction
  (__init__), NoReturn - all of cfg.py's per-function bookkeeping this would
  need to replicate correctly at a splice point instead of a real
  FuncStart/FuncEnd boundary. builtins.len and every other forcing candidate
  found above is a single `return <expr>` - no reason to build the general
  case speculatively.
- Recognizing @inline through an Overload group, or through a runtime
  union-receiver dispatch (_lower_union_receiver_call) - reject the
  @inline+@overload combination outright (like @virtual+@overload already
  is), same reasoning: which branch's body would even get spliced isn't a
  single well-defined answer without more design.

Precedent reused

- discovery.py's `_parse_function` decorator loop (is_static/is_overload/
  is_virtual/etc., ~line 1458) - `@inline` becomes one more case there,
  is_inline stored on Function (mpy_types.py, next to is_overload) via
  `replace()`-friendly dataclass field, so monomorphize.py's existing
  `replace(base, ...)` calls carry it through to a monomorphized generic
  instantiation with zero changes there.
- monomorphize.py's `monomorphized_function` already gives a generic
  @inline function (len[T]) a concrete, per-instantiation Function (own
  substituted .parameters, own deep-copied .node) memoized by Specialization
  - reused completely unchanged. The only new thing is that this pass does
  NOT `schedule()` that monomorphized Function as a real compiled unit the
  way `_emit_generic_call` does today - it only reaches into
  `.node.body`/`.parameters` for the splice.
- lowering.py's `_lower_call_args` (plain path) and `_lower_inferred_generic_
  call`'s interleaved `lower_and_unify` (generic path) already lower every
  argument against the callee's real parameter types and apply move hooks -
  reused verbatim; only the tail (build+emit an ir.Call) changes.
- `_expr_Name`/`ir.Operand = Union[Temp, Const, Variable, FunctionRef]`:
  since Variable/Parameter double as their own operand, and an already-
  lowered call argument can be ANY of Temp/Const/Variable/FunctionRef (not
  just a bare Variable), name substitution needs a level below
  discovery.find_name's Variable-only registry - see Implementation #3.

Implementation

1. mpy_types.py: add `is_inline: bool = False` to `Function` (next to
   `is_overload`).

2. discovery.py `_parse_function`:
   - add `case 'inline': is_inline = True` to the decorator match (~1458).
   - reject combinations: `@inline` with `@overload`, `@virtual`,
     `@abstractmethod`, `@extern`, `@classmethod`, or `@move` - one `self.
     fail(...)` per combination, same style as the existing `is_virtual and
     is_overload` check just above where `fn = Function(...)` is built.
   - new helper `_is_inline_eligible_body(body: list[ast.stmt]) -> bool`,
     next to `_is_stub_body`: strips a leading docstring Expr (`ast.
     get_docstring`-style check, not an actual call to it since that mutates
     nothing but we only need the boolean shape), then requires exactly one
     remaining statement, `isinstance(stmt, ast.Return)`, `stmt.value is not
     None`. When `is_inline` and this returns False, `self.fail(...)`:
     "@inline {qualname} must have a body of exactly `return <expr>`
     (optionally preceded by a docstring) - got {shape}: not yet supported".
   - pass `is_inline = is_inline` into the `Function(...)` construction.

3. lowering.py, new `FunctionLowering._lower_inline_call( self, node:
   ast.Call, target: Function, receiver: ir.Operand|None, args: list[ir.
   Operand], kwargs: dict[str,ir.Operand], expected_type: Type|None,
   want_result: bool ) -> ir.Operand|None`:
   - reentrancy guard: `self._inlining_stack: list[int]` (new field on
     FunctionLowering, initialized in __init__ next to `_pending_temps`) -
     `id(target)` already present -> `self.lowering.discovery.fail(f'@inline
     {target.qualname}: recursive inlining (directly or through another
     @inline function) is not supported', node)`. Push/pop around the whole
     body (try/finally), covering both direct self-recursion and mutual
     recursion through a chain of currently-inlining targets.
   - same discard check the ordinary tail already does: `if not want_result
     and cfg.is_result_type(target.return_type): self.lowering.discovery.
     fail(...)` (copy the exact message from the plain-call tail, ~4787).
   - build `substitution: dict[str,ir.Operand]`: `{'self': receiver}` if
     `receiver is not None`, then walk `target.parameters` and `args` in
     lockstep (positional args bind to the leading parameters in
     declaration order - this language has no *args/**kwargs, see print()'s
     own comment), falling back to `kwargs[param.stem]` for the rest.
   - push `substitution` onto a new `self._inline_substitutions: list[dict[
     str,ir.Operand]]` stack (new field, next to `_inlining_stack`).
   - `return_expr = _inline_return_expr(target.node.body)` (shared tiny
     helper, strips the same optional leading docstring
     _is_inline_eligible_body already validated exists, returns the single
     Return's `.value`).
   - lower it: `with self.lowering.discovery.module_context(self.lowering.
     _find_module_for(target)): with self.lowering.discovery.scope_context(
     target): result = self._lower_expr(return_expr, expected_type or
     target.return_type)` - scope_context(target) (NOT target.cls) so any
     bare name in the body OTHER than self/a parameter (a sibling module-
     level helper, say) still resolves in the callee's own defining scope;
     self/params are intercepted before ever reaching discovery.find_name at
     all (next bullet), so they don't need - and must not get - a stale
     Parameter placeholder from target.names.
   - pop `_inline_substitutions` and `_inlining_stack` (finally).
   - `return result if want_result else None` (still lowered either way -
     side effects inside the substituted expression, e.g. a nested call,
     must still run when the result is discarded, same as any other
     expression statement).

4. lowering.py `_expr_Name` (~2785): at the top, before calling
   `discovery.find_name`, check `if self._inline_substitutions and node.id in
   self._inline_substitutions[-1]: return self._inline_substitutions[-1][
   node.id]`. Only the TOP frame is ever consulted (correct for nested
   inline calls - by the time a nested @inline call's own body is being
   lowered, its own frame is what's on top; once it returns and pops, the
   outer frame is back on top unchanged). This is the one new hook every
   other design here routes through - it's what lets an already-lowered
   argument OPERAND (Temp/Const/Variable/FunctionRef, not just a bare
   Variable/Parameter the way discovery.find_name's registry requires) stand
   in for `self`/a parameter with no synthesized local, no extra ir.Assign,
   and no CFG/RC bookkeeping of its own - it's a pure alias to something the
   caller already owns (or borrows) exactly as before; inlining doesn't
   change who owns what, only how the value is named while the substituted
   expression runs.

5. lowering.py, three call sites, each adding one check before their
   existing ir.Call-emitting tail:
   - `_lower_call`'s plain final tail (~4766-4799): after `args, kwargs =
     self._lower_call_args(target, node)`, `if target.is_inline: return self.
     _lower_inline_call(node, target, receiver, args, kwargs, expected_type,
     want_result)`.
   - `_lower_generic_function_call` (explicit `Name[T](...)`, ~4369): after
     `monomorphized = self.lowering._monomorphized_function(spec)`, `if
     monomorphized.is_inline: args, kwargs = self._lower_call_args(
     monomorphized, node); return self._lower_inline_call(node,
     monomorphized, receiver, args, kwargs, expected_type, want_result)`
     - note this deliberately skips `self.lowering.schedule(spec)` /
       `schedule(monomorphized)` / `schedule(param.type)` /
       `schedule(return_type)` that `_emit_generic_call` does - an @inline
       generic instantiation is never itself a compiled unit, only whatever
       real calls its spliced body makes are (those get scheduled normally,
       by the recursive `_lower_expr` call itself).
   - `_lower_inferred_generic_call` (bare `len(x)`, ~4378): after `spec =
     ...; monomorphized = self.lowering._monomorphized_function(spec)`
     (args/kwargs already computed by the interleaved lower_and_unify
     above), same `if monomorphized.is_inline: return self._lower_inline_
     call(node, monomorphized, receiver, args, kwargs, expected_type,
     want_result)` before falling through to `_emit_generic_call`.
   - the `resolved_callee`-tagged path (~4603) needs no separate change -
     type_resolver.py's pre-pass hands back an already-monomorphized, plain
     (type_params-stripped) Function, which falls through to the same plain
     final tail as any other non-generic call.

6. lib/builtins/__init__.py: add `@inline` to `len[T]`, delete the "TODO
   once @inline exists" comment (its job is done), keep a short note that
   this is the reason `len(x)` costs exactly what `x.__len__()` does.

STATUS: landed. Implementation #4 above (the `_expr_Name` substitution-stack
hook) turned out not to be viable and was replaced during implementation -
kept here, struck through in spirit, as a record of why the actual design
looks different:

Found by a real repro, not just reasoning: `_expr_Name` is NOT the only place
that resolves a bare `self`/parameter Name while lowering a Call.
`_try_resolve_namespace` (used by the construction-call recognizers -
`_try_lower_allocate_call`/`_try_lower_construct_call` - to speculatively
check whether `X(...)` might be `ClassName.method(...)`-shaped construction
sugar, BEFORE ordinary attribute/method resolution ever runs) calls
`discovery.find_name` DIRECTLY, bypassing `_expr_Name` entirely. An inlined
`self.__len__()` body hit exactly this path (`self.__len__()` LOOKS like it
could be construction sugar until proven otherwise) and failed with "name
'self' is not defined" - the substitution-stack hook never got a chance to
intercept it.

Fix actually landed: instead of an `_expr_Name`-level shortcut, each
binding becomes a REAL entry in `target.names` (save the old value, if any;
restore it in the same `finally` the reentrancy-guard pop already uses) -
this makes EVERY `discovery.find_name`-based path see it correctly, not
just `_expr_Name`. Two follow-on issues this raised, both fixed:
- `discovery.find_name`'s registry requires a `Name` (Variable/Function/...)
  - a Temp or Const argument operand isn't one. When the operand is ALREADY
    a Variable (a bare-name receiver/argument - by far the common real-world
    case, e.g. `b.get_len()`, `some_result.is_ok()`), it's registered
    DIRECTLY, no copy, no extra instruction - true zero overhead, confirmed
    by lowering_test.py's own differential test asserting the spliced
    Call's receiver `is` the SAME Parameter object main already declared.
    Only a genuinely computed operand (a Temp from a sub-expression like
    `make_box().get_len()`, or a Const) gets a synthesized local instead - a
    fresh `Variable` + a plain `ir.Assign` (no `_cfg_assign`/incref, same
    borrowed-no-bookkeeping treatment an ordinary function parameter already
    gets at a real call boundary), both to give it a referenceable name at
    all and to guarantee single evaluation if the body references it more
    than once.
- emitter_c.py declares a C local the first time it sees an `ir.Assign`
  against a Variable, keyed by the Variable's own `.stem` STRING, not by
  object identity. Naming a synthesized binding literally `self` (reusing
  the parameter's own stem) would silently collide with and overwrite the
  ENCLOSING function's own real `self`/parameter of the same name the
  moment one method's `@inline` body gets spliced into another method's
  own body (is_ok's own `self.tag` splicing into a method that ALSO has a
  real `self`, say). Fixed by giving the synthesized Variable a unique
  `.stem` (a monotonic `_inline_binding_id` counter) while still
  registering it in `target.names` under the ORIGINAL key ('self') - the
  dict key and the Variable's own field are independent, so
  `discovery.find_name('self', ...)` still resolves correctly while
  emitter_c.py declares a collision-free C name.

Also found and fixed along the way: `_resolve_call_target` (the plain
call tail's own signature-resolution step) called `_ensure_resolved`
unconditionally, which schedules its target as a real compile unit as a
side effect - for an `@inline` target this defeated the "never a real
compiled unit" goal for exactly the RECEIVER-based generic-class-method
case this plan's own "Deferred" section above corrected itself on
(`some_result.is_ok()`): confirmed by a real repro,
`Result.is_ok[i32,MyError]` showed up as a genuine (dead, never-called)
compiled function until this was fixed. Given the existing `is_virtual`
carve-out in the same function (resolve the signature, skip the
scheduling side effect - vtable dispatch has the identical shape), added
an analogous `is_inline` carve-out right next to it.

Verification

- discovery_test.py: `@inline` accepted on a single `return <expr>` body
  (with and without a leading docstring); rejected on a multi-statement
  body, a bare `return`, `return None` with no value is still a bare
  `return` (rejected); rejected combined with @overload/@virtual/
  @abstractmethod/@extern/@classmethod/@move.
- lowering_test.py: a non-generic `@inline` method call emits no ir.Call/
  FuncStart/FuncEnd for the callee at all - the caller's own instruction
  stream contains exactly what calling the forwarded-to expression directly
  would (e.g. `@inline def get_len(self): return self.__len` compiles the
  same as `self.__len` written inline at the call site); a generic `@inline`
  free function (mirroring builtins.len[T]) called bare, resolves T from the
  argument and splices `t.__len__()` against the ARGUMENT's real type, no
  Specialization ever scheduled as a compiled unit; explicit `fn[i32](x)`
  spelling does the same; a receiver expression with side effects (e.g.
  `get_next_list().len()`) is only evaluated once (proves args are lowered
  BEFORE substitution, not re-lowered per reference in the body); recursive
  @inline (direct and mutual A<->B) rejected with a clear error; discarding
  an @inline call whose return type is Result[_,_] rejected the same as an
  ordinary call; `@inline`-ing Result.is_ok/is_err and calling
  `some_result.is_ok()` still correctly clears the CFG's unchecked-Result
  tracking for `some_result` (proves _cfg.clear_result's stem-based check,
  which runs before dispatch, isn't skipped just because the call ends up
  spliced instead of emitted as ir.Call).
- emitter_c_test.py: real compile-and-run of `len(x)` for a couple of
  concrete T (list[i32], str, FastList) producing identical output AND
  (informally, by reading the generated .c) no separate `len` function
  emitted, no call instruction at the use site - direct inlining of
  `x.__len__()`'s own body.
- Full python tests.py green before/after.

Follow-up done: multi-statement bodies (locals, if/for/while before a
single, final, un-nested `return <expr>` - NOT early/nested return, that's
its own further follow-up, explicitly deferred by the user). The original
`@inline` scope above ("EXACTLY one `return <expr>` statement") is now
just the trivial single-statement special case of a more general shape
check; every previously-accepted body still compiles identically.

The two hazards that make this more than "just splice more statements",
both found by design research BEFORE any code was written (a background
CFG deep-dive, then a Plan agent asked to specifically stress-test the
"no new CFG scope primitive needed" hypothesis) - confirmed real via a
repro once implemented, not just reasoning:

- **Hazard 1**: `_stmt_Assign`/`_stmt_AnnAssign`'s fresh-declaration branch
  registers a new local into `self._current_fn` (a `FunctionLowering`-
  level field, assigned exactly once, in `__init__`, never reassigned
  anywhere else in the file before this) - which, unmodified, is the
  CALLER's own top-level function during a splice, not the callee's.
  `Function.add_name` is an unconditional dict overwrite with no collision
  guard - a pre-return-statement local sharing a name with an existing
  caller-side local would silently overwrite the caller's own entry,
  corrupting every later reference to that name in the caller's own
  hand-written code, from that point in the (unordered) dict onward.
- **Hazard 2**: `_stmt_Assign`'s "is this name already bound" check goes
  through `discovery.find_name_or_none` (walking `discovery.scope_stack`,
  which `module_context`/`scope_context` control) - a SEPARATE mechanism
  from `self._current_fn`. Reassigning an already-alpha-renamed local a
  SECOND time within the same spliced body needs both to agree on where
  the first assignment landed, or the second assignment can't find it and
  creates an independent second binding under the same name instead of a
  replace - `cfg.py`'s own fresh-vs-replace RC bookkeeping depends on
  finding the SAME Variable object both times, so this would silently
  skip a decref (a real leak, not just a bookkeeping oddity).

Both fixed by ONE mechanism: a fresh, per-call-site **provisional
Function** (built via `monomorphize.py`'s `_build_monomorphized_function`
- already precedented for exactly this shape by PLAN_RETURN_INFERENCE.md's
own `_infer_return_only_type_params_inline`), with `self._current_fn`
TEMPORARILY reassigned to it (the first and only place this ever happens
in the file, narrowly scoped, restored in a `finally`) for exactly the
window `discovery.scope_context(provisional)` is also active - unifying
where new locals get registered and where "is this already bound" gets
looked up onto the same object resolves both hazards at once.

A third hazard, not part of the original ask, found during the SAME design
pass: reassigning `self`/a parameter inside a multi-statement body, when
that binding was passed in via the existing "reuse the caller's own bare
Variable directly, no copy" fast path (the common case - a bare-name
receiver/argument), would silently mutate the CALLER's own variable, not a
private copy. Rejected outright at parse time for this pass (a new
discovery.py scanner, `_find_inline_body_reserved_name_reassignment`) -
relaxing it later needs "force a defensive copy binding whenever
reassignment is detected" instead of today's zero-copy reuse optimization.

`defer`/`errdefer` anywhere before the final return are ALSO rejected
outright (`_find_inline_body_early_exit_construct`) - not merely "no
boundary exists" but a genuine timing bug waiting to happen: `defer`'s own
contract is "runs when THIS function returns" (i.e. right after the
return-expression is computed), not whenever the CALLER's own, much later,
real exit eventually fires.

`.or_return()`/checked-arithmetic auto-propagation in a pre-return
statement is rejected at LOWERING time (a new `self._in_inline_splice_
prelude` flag, checked at the top of `_consume_checked_result` - the one
shared choke point for `.or_return()`, checked arithmetic's own opcode
dispatch, AND the `__getitem__`/`__len__` auto-consume path, confirmed by
grep before relying on it) rather than at parse time, since whether a given
op is actually checked depends on operand types not known until lowering.
Left unguarded, this would jump to the CALLER's own real epilogue mid-
splice - silently wrong whenever the caller happens to also satisfy the
Result-return shape, not just an error case. The trailing return-
EXPRESSION itself is unaffected (the flag is restored to False before it's
lowered) - nothing of the splice remains after it to skip past, so jumping
to the caller's own epilogue there is already correct, exactly as before
this pass.

Alpha-renaming itself (every local the pre-return statements declare, via
`Store`-context `ast.Name` collection, excluding self/params) turned out
to need only in-place `ast.Name.id` mutation, no `ast.NodeTransformer` -
the provisional's own `.node` is already a private, per-call-site deep
copy, so mutating it directly is safe, and renaming never restructures the
tree, only a string field.

`resolve_function_body` (type_resolver.py) runs against the provisional
BEFORE alpha-renaming, not after - match-statement desugaring (there is no
`_stmt_Match` anywhere in lowering.py, so a `match` in a spliced body can
only ever lower after this rewrite has run) needs to see the ORIGINAL
names; the alpha-renamer only ever needs to understand plain `ast.Name`,
never a match pattern's own capture-binding shapes.

Found, out of scope, flagged separately (spawn_task, not fixed here):
`.or_return()` fails to compile at all - even for a completely ordinary,
non-inline, non-generic free function - when the `Result[T,E]` class is a
bare `@union` with just `Ok`/`Err` members and no explicit, hand-written
`or_return` method of its own (the shape the real lib/builtins/__init__.py
Result actually has, and the same shape `is_ok`/`is_err`/`match` already
work fine with) - only reproduced once this pass's own tests needed a
synthetic Result class exercising `.or_return()` for the first time in
this codebase's session history; a real, pre-existing gap, unrelated to
`@inline` itself.

Verification (multi-statement addition):

- discovery_test.py (12 new tests in `RCClassVirtualTests`): the
  generalized body-shape check accepts locals/if before a final return;
  rejects a return nested in an if even alongside a different final
  return (two reachable returns); rejects a non-last return; rejects a
  bare `return`; rejects `defer`/`errdefer` (both spellings, including
  nested in an if); rejects self/parameter reassignment (including
  nested); accepts reassignment of a body-declared local.
- lowering_test.py `InlineMultiStatementTests` (9 tests): no real Call/
  FuncStart/FuncEnd for a multi-statement target; a caller-side local
  sharing a name with an inline-body local is NOT corrupted (Hazard 1
  regression, checked by identity/qualname, not just "no error"); a
  second assignment to a body-declared local reuses the SAME Variable
  object, not two independent bindings (Hazard 2 regression); a spliced
  `if` nests correctly inside the caller's own `if`; a `match` statement
  in a pre-return statement lowers correctly; `.or_return()` in a pre-
  return statement rejected via the new lowering-time guard (using an
  inline target that itself declares a Result-shaped return type, so the
  PRE-EXISTING "enclosing function must return Result" check passes and
  the NEW guard is what actually catches it); direct recursion reached
  from a pre-return statement rejected; a bare-call generic multi-
  statement `@inline` function splices with no Specialization ever
  compiled; combined with PLAN_RETURN_INFERENCE.md, a multi-statement
  generic `@inline` function with a return-only type parameter still
  infers correctly.
- inline_multistatement_test.py (new file, `test_support.RealCompileMixin`):
  real compile+link+run - a multi-statement `@inline` method (a local,
  checked arithmetic under `with compiler.wrap_arithmetic:`, a nested
  `if`/reassignment) produces the correct runtime value across two call
  sites with different inputs, a caller-side local sharing the inlined
  body's own local name is unaffected, and no separate C function is ever
  emitted for the inlined target.
- Full python tests.py green throughout (1001 passing).

STATUS: early/nested return + defer/errdefer/.or_return() (2026-08-15)

The three restrictions the multi-statement pass above deliberately left in
place - early/nested `return`, `defer`/`errdefer`, and `.or_return()`/
checked-arithmetic in a pre-return statement - are now supported. All three
turned out to be one piece of work: each is "produce a value, then exit the
inlined function early," and `defer`/`errdefer`'s own "runs when THIS
function returns" contract only has a meaning once that early exit is a
real, addressable point again.

The core primitive: `cfg.py` gained `InlineScope`/`push_inline_scope`/
`pop_inline_scope`/`build_inline_scope_ladder` - a splice-local analogue of
the function-wide `current_epilogue_label`/`build_epilogue_ladder` an
ordinary early return already uses. `current_epilogue_label`/`return_` both
now stop at the innermost active scope's own `boundary_depth` instead of
continuing into the caller's (or an outer splice's) older entries - the
exact bug that made jumping into the caller's real epilogue possible before.
`_stmt_Return`/`_consume_checked_result` redirect into the scope's own
`result_var`/`exited_flag` (armed the same way a defer flag already is)
instead of `self._return_value_var`/a real return whenever a scope is
active; `ir.OrJump`/`ir.OrReturn` gained matching optional fields
(`exited_flag`/`inline_exit`) so `.or_return()`/checked-arithmetic redirect
too, with zero new opcodes. `_splice_multi_statement_inline_body`'s own tail
now emits the scope's ladder, then a flag-gated merge (mirroring
`_expr_IfExp`'s own "shared dest temp, two Assign sites, converge at one
label" ternary shape) between the early-exit value and the trailing
return-expression - only one of which actually runs.

Two real bugs found only by testing beyond clang/gcc (see
`vcvars64_available.md`/`linker_c_validate_all_compilers.md` memory - MSVC's
`/RTC1` catches what clang/gcc silently tolerate):
1. A synthesized local (`result_var`) whose first write could land inside
   `emitter_c.py`'s own hand-emitted `if (...) { }` blocks (`_emit_or_return`/
   `_emit_or_jump`) got block-scoped by C, undeclared everywhere else - fixed
   with a new `ir.DeclareLocal` instruction (the `DeclareTemp` of named
   Variables), emitted flat and unconditional before the splice's own
   pre-return statements even start lowering.
2. The trailing return-expression's own temp never got `untrack_temp`'d
   after its ownership moved into the merge's `result` via a bare
   `ir.Assign` - `_flush_pending_temps` then emitted an RC-cleanup check for
   it unconditionally, outside the "normal path" branch that's the only
   place it was ever actually assigned - a genuine uninitialized-read on the
   early-exit path, not just a redundant decref.

PLAN_RETURN_INFERENCE.md's own `@inline` variant needed one more carve-out:
it reaches `_splice_multi_statement_inline_body` with `target.return_type`
set to Python `None` as a deliberate "not yet known" sentinel (not
`none_type`) - none of the new machinery can run against an unresolved
type, but `_is_eager_return_inferable_body`'s own "exactly one reachable
return" eligibility gate already guarantees no early return can co-occur
with it, so this case simply falls back to the original, pre-this-pass
code path unchanged. `.or_return()`/checked-arithmetic in a pre-return
statement stays rejected in that one narrow combination (no scope exists
to redirect into, same as before this pass for every splice).

Verification (early-return/defer/or_return addition):

- discovery_test.py: early `return` nested in if/for/while now accepted; a
  bare early `return` still rejected (every reachable return needs a
  value); `defer`/`errdefer` (both spellings, including nested in an if)
  no longer rejected; self/parameter reassignment still rejected
  (unrelated hazard, untouched).
- lowering_test.py `InlineMultiStatementTests`: an early return nested in
  a spliced `if` produces exactly one real `ir.Return`/`ir.FuncEnd` for
  the caller (not a second one from the inlined target); `.or_return()`
  in a pre-return statement now works, with a direct regression test for
  the "notably important" requirement - the `OrJump` lands on a real,
  declared label local to the splice, and the caller keeps exactly one
  `ir.Return`/`ir.FuncEnd`, proving no caller-level early return happened;
  a defer registered inside a spliced `if` is replayed exactly once, at
  the splice's own ladder.
- inline_multistatement_test.py: real compile+link+run proof of the same
  "notably important" requirement - an `@inline` method whose pre-return
  statement early-exits via `.or_return()` on an `Err` receiver, called
  from `main()` with real code after the call site that must still run
  and correctly observe the propagated `Err`.
- Full python tests.py green (1035 passing, post-merge) under clang (the
  default), MSVC (`METALPY_CC=msvc`), and gcc 14.2.0 (via WSL -
  `wsl bash -lc "cd /mnt/c/cvs/metalpy && METALPY_CC=gcc python3 tests.py"`,
  see `linker_c_validate_all_compilers.md` memory - gcc isn't on native
  PATH here, but is reachable through WSL, don't infer "unavailable" from
  a native-shell check alone). Both new real-compile tests pass on all
  three, after the two MSVC-only bugs above were found and fixed this
  way. (A handful of unrelated MSVC-only failures were also observed
  in this worktree under `METALPY_CC=msvc python tests.py` - confirmed
  pre-existing and already fixed on master by other, concurrent work that
  landed after this worktree branched, not a regression from this pass -
  see master's `4f94935` and neighboring commits.)
