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
