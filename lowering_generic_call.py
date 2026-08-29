# stdlib imports:
import ast
import itertools
import math
from typing import Callable, Iterable

# local imports:
import cfg
import ir
from errors import CompileError
from mpy_types import (
	Name, Type, Parameter, Function, Overload, Specialization, TaggedUnion, CStruct, TypeVar, ConditionalDispatch, GeneratorType,
)
import overload_resolution

class GenericCallLoweringMixin:
	''' generic/monomorphized call inference and emission - mixed into FunctionLowering (lowering.py), which
	see for the shared instance state (self._instructions, self._cfg, self.lowering,
	etc.) every method here reads and writes. Never instantiated on its own;
	split out of lowering.py purely to keep that file to a manageable size - see
	lowering.py's own class docstring and FunctionLowering's base-class list for
	the full set of sibling mixins this one is composed with. '''


	def _lower_generic_function_call( self, node: ast.Call, spec: Specialization, receiver: ir.Operand|None, expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# sys.alloc[u8](...) - explicit generic instantiation. Matches call
		# args against the MONOMORPHIZED signature (so a literal argument's
		# expected type is already concrete, e.g. usize for alloc[u8]'s
		# count - not the abstract, unsubstituted one)
		# spec.base is always a Function here (see the isinstance dispatch in
		# _lower_call that routes to this method)
		self.lowering._check_type_param_bounds( node, spec.base.type_params or [], spec.args, spec.base.qualname )
		monomorphized = self.lowering._monomorphized_function( spec )
		args, kwargs = self._lower_call_args( monomorphized, node )
		if monomorphized.is_inline:
			return self._lower_inline_call( node, monomorphized, receiver, args, kwargs, expected_type, want_result )
		return self._emit_generic_call( node, spec, monomorphized, receiver, args, kwargs, expected_type, want_result )

	def _lower_inferred_generic_call( self, node: ast.Call, target: Function, receiver: ir.Operand|None, expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# a BARE call to a generic function (mylen(a), no explicit [T]) -
		# unlike _lower_generic_function_call, there's no already-concrete
		# Specialization to match args against yet: T has to be inferred
		# FROM the arguments themselves first. Each argument is lowered in
		# turn (positional, then keyword - same order _lower_and_infer_
		# call_args uses) with an expected-type hint built from whatever
		# bindings EARLIER arguments in this same call already solved (a
		# still-unbound type param just passes its own bare TypeVar through
		# - _substitute_type_params leaves anything it doesn't recognize
		# alone, same as no hint at all), then immediately unified
		# (_unify_type_param) against its declared parameter type to refine
		# bindings before the NEXT argument is lowered - the inverse of
		# _substitute_type_params, which already handles substituting a
		# SOLVED binding through arbitrarily nested Specializations (list[T]
		# etc), so unification mirrors that same recursive shape instead of
		# only handling a bare `t: T` parameter. This interleaving (rather
		# than lowering everything first, then unifying everything after) is
		# what lets a LATER Callable[[T],K]-typed argument's own lambda body
		# see T already bound from an EARLIER plain argument, even though K
		# itself is still only inferable from the lambda's own body
		# (PLAN_LAMBDA.md, "eager lambda lowering") - a bare TypeVar
		# parameter still offers no useful literal hint on its own, so a
		# literal argument at a position nothing's bound yet still correctly
		# fails via _expr_Constant's own "cannot infer" error, same as before
		assert target.resolve is None, f'internal compiler error - {target=} was not fully resolved by the type_resolver module'
		positional, keyword = self.lowering._match_call_args( target, node )
		type_params = target.type_params or []
		bindings: dict[int,Type] = {} # id(TypeVar) -> the concrete Type it was inferred as

		def lower_and_unify( param: Parameter, expr: ast.expr ) -> ir.Operand:
			partial_args = [ bindings.get( id( tv ), tv ) for tv in type_params ]
			hint = self.lowering._substitute_type_params( param.type, type_params, partial_args )
			if isinstance( hint, TypeVar ) and any( hint is tv for tv in type_params ):
				# substitution left the hint as a BARE, still-unbound type
				# param (nothing bound it yet) - not a real hint, same as no
				# hint at all (_expr_Constant's own "cannot infer" error is
				# the correct outcome for a literal here, exactly as before
				# this method started interleaving lower+unify). A hint that
				# came back PARTIALLY substituted (e.g. Ptr[Callable[[i32],K]]
				# - T bound, K still bare) is fine as-is and reaches here
				# unchanged - only a hint that IS one of type_params, bare,
				# needs this fallback
				hint = None
			# strict=False: `hint` can still contain an unbound TypeVar nested
			# inside it (e.g. Ptr[Callable[[i32],K]] with K not yet bound) -
			# it's an inference HINT here, not a validated requirement; the
			# REAL validation/binding is _unify_type_param below, not
			# _lower_expr's own general assignability check
			operand = self._lower_expr( expr, hint, strict = False )
			# strict=False above skips _coerce_or_check_operand's own case-2
			# auto-or_throw() hook - reinstated here, same as _lower_and_
			# infer_call_args' own identical fix, so an unconsumed Result[T,E]
			# argument (e.g. an unannotated `c = a + b` passed to a generic
			# function expecting plain T) unifies against its UNWRAPPED type
			# instead of conflicting with an already/still-inferring binding
			operand = self._auto_consume_hint_arg( node, operand, param.type )
			self.lowering._unify_type_param( type_params, param.type, operand.type, bindings, node, target.qualname )
			return operand

		# argument "signal strength" for type-param inference varies -
		# processing every argument in plain left-to-right order lets a
		# weaker-signal argument bind a type param FIRST, wrongly
		# conflicting with (or starving) a stronger one that comes later:
		#   - a bare literal's own natural type (builtins.int/f64/... with
		#     no context - _expr_Constant's own expected_type-is-None rule)
		#     is WEAK evidence: `apply(5, key=identity_i32)` had `5` bind
		#     T=builtins.int before `key`'s own Ptr[Callable[[i32],i32]]
		#     signature could reveal T should be i32, conflicting with it -
		#     confirmed via a real repro.
		#   - a lambda with no explicit parameter annotations has NO signal
		#     of its own at all (PLAN_LAMBDA.md's "eager lambda lowering" -
		#     it needs a type param ALREADY bound to even infer its own
		#     parameter types) - `apply(5, key=lambda v: v)` needs `5`
		#     processed BEFORE the lambda, the opposite ordering from the
		#     literal-vs-concrete-reference case just above.
		# Three tiers fixes both: ordinary/concrete arguments (a Name,
		# a real Call, a typed function reference, ...) always run FIRST,
		# in their own original relative order; bare literals run next;
		# lambdas run last, once every other argument has contributed
		# whatever binding it can. Safe to reorder purely because neither a
		# literal nor a lambda EXPRESSION ITSELF (as opposed to calling it)
		# has an observable side effect - the emitted Call's own argument
		# order below is entirely separate (list/dict position, not
		# evaluation order) and is unaffected either way.
		args: list[ir.Operand|None] = [ None ] * len( positional )
		kwargs: dict[str,ir.Operand] = {}
		def _arg_tier( expr: ast.expr ) -> int:
			if isinstance( expr, ast.Lambda ):
				return 2
			if isinstance( expr, ast.Constant ):
				return 1
			return 0
		all_args: list[tuple[int|str,Parameter,ast.expr]] = (
			[ ( i, param, expr ) for i, ( param, expr ) in enumerate( positional ) ]
			+ [ ( param.stem, param, expr ) for param, expr in keyword ]
		)
		for key, param, expr in sorted( all_args, key = lambda t: _arg_tier( t[2] ) ):
			result = lower_and_unify( param, expr )
			if isinstance( key, int ):
				args[key] = result
			else:
				kwargs[key] = result
		for ( param, _expr ), operand in zip( positional, args ):
			self._apply_move_hook( param, operand, target.qualname, node )
		for param, _expr in keyword:
			self._apply_move_hook( param, kwargs[param.stem], target.qualname, node )

		return self._finish_generic_call( node, target, type_params, bindings, receiver, args, kwargs, expected_type, want_result )
		# else: this parameter position doesn't mention any of type_params
		# (a concrete parameter, or a nested type whose base doesn't even
		# match the argument's) - nothing to infer here. Not an error by
		# itself: a genuine argument-type mismatch isn't checked anywhere
		# yet (no general type-checking pass exists), same as every other
		# call site in this file today

	def _coerce_generic_call_args( self, monomorphized: Function, args: list[ir.Operand], kwargs: dict[str,ir.Operand], node: ast.Call ) -> tuple[list[ir.Operand],dict[str,ir.Operand]]:
		# a bare inferred call's own interleaved lower+unify (lower_and_
		# unify, above) lowers each argument with strict=False against a
		# partial-substitution HINT, deliberately skipping real coercion
		# (see its own comment) since the hint can still contain an unbound
		# type param mid-inference. Once every type param is bound and
		# `monomorphized`'s own parameter types are fully concrete, an
		# argument whose own natural type is just ONE non-union leaf of a
		# union-typed parameter (e.g. `key: Ptr[Callable[[T],K]]|None`,
		# monomorphized to `Ptr[Callable[[i32],i32]]|None`, called with a
		# bare `Ptr[Callable[[i32],i32]]`-typed argument, never itself
		# wrapped into the union) still needs the SAME real coercion an
		# ordinary (non-generic) call's own _lower_and_infer_call_args
		# already applies - without this, the union-wrapping never
		# happened at all, reaching emitter_c.py's _emit_call_args with a
		# bare function pointer passed where the whole union struct is
		# declared, a real (if less silent) clang type-mismatch error -
		# confirmed by a real repro.
		#
		# Deliberately narrow: only when `param.type` is ITSELF a union
		# (the one shape strict=False's own skip actually leaves unfinished -
		# ordinary same-type/no-op cases are already handled by the operand's
		# own natural type matching exactly) - a blanket _coerce_or_check_
		# operand over every parameter, unconditionally, re-surfaced a
		# separate, pre-existing, genuinely out-of-scope gap early (a
		# Closure[[T],K]-shaped parameter whose K is only resolvable via
		# eager lambda-lowering, not fed back into monomorphized.parameters'
		# own substitution - see lowering_test.py's test_lambda_eager_
		# return_type_inference_with_capture, whose own docstring already
		# documents this as "not fixed here, out of scope") as a hard
		# lowering-time discovery.fail() instead of the deferred, silent-
		# until-emission behavior every OTHER unrelated generic-call shape
		# already had before this fix.
		def _maybe_coerce( operand: ir.Operand, declared: Type|None ) -> ir.Operand:
			if not isinstance( declared, TaggedUnion ):
				return operand
			return self._coerce_or_check_operand( operand, declared, node )
		coerced_args = [
			_maybe_coerce( operand, param.type )
			for param, operand in zip( monomorphized.parameters or [], args )
		]
		coerced_kwargs = dict( kwargs )
		for param in monomorphized.parameters or []:
			if param.stem in coerced_kwargs:
				coerced_kwargs[param.stem] = _maybe_coerce( coerced_kwargs[param.stem], param.type )
		return coerced_args, coerced_kwargs

	def _fill_generic_call_defaults( self, monomorphized: Function, args: list[ir.Operand], kwargs: dict[str,ir.Operand], node: ast.Call ) -> None:
		# _lower_inferred_generic_call/_lower_overload_generic_call's own
		# argument lowering (lower_and_unify, and the Overload group's own
		# _lower_overload_arg) only ever populates args/kwargs from what the
		# CALL SITE actually wrote - unlike the plain (non-generic) call path
		# (_lower_call_args) and the concrete-Overload-candidate path (below
		# in _lower_overload_generic_call's caller), neither of which ever
		# reaches a still-generic target, so omitted-but-defaulted parameters
		# were never filled in here at all. Confirmed via a real repro: a
		# bare generic call omitting a defaulted argument (`def take[S](seq:
		# S, pad: i32 = 99): ...` called as `take(x)`) reached emitter_c.py's
		# _emit_call_args with no 'pad' entry in instr.kwargs, crashing with
		# a bare KeyError. Mirrors _lower_call_args's identical tail, against
		# monomorphized's own already-substituted parameter types (never
		# target's abstract, still-TypeVar ones) so a default expression
		# mentioning a type param (`start: T = 0`) lowers against the real,
		# concrete T.
		given = { p.stem for i, p in enumerate( monomorphized.parameters or [] ) if i < len( args ) }
		given.update( kwargs.keys() )
		for param in monomorphized.parameters or []:
			if param.stem not in given and param.default is not None:
				# see _lower_parameter_default's own docstring for why this
				# is needed - a construction call embedded in the default
				# otherwise never gets its __init__ eagerly pre-resolved
				kwargs[param.stem] = self._lower_parameter_default( monomorphized, param, node )

	def _finish_generic_call( self, node: ast.Call, target: Function, type_params: list[TypeVar], bindings: dict[int,Type], receiver: ir.Operand|None, args: list[ir.Operand], kwargs: dict[str,ir.Operand], expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# shared tail of _lower_inferred_generic_call (extracted verbatim,
		# unchanged) and _lower_overload_generic_call below - once `bindings`
		# holds every type param inferable from the ARGUMENTS alone (however
		# they were obtained: interleaved lower+unify for a bare generic-
		# function call, or unified against already-lowered operands for a
		# resolved Overload-group generic candidate), monomorphizing and
		# emitting the call is identical either way.
		missing = [ tv for tv in target.type_params or [] if id( tv ) not in bindings ]
		if missing:
			# return-only inference: a still-unbound type param that never
			# appears in any PARAMETER type (so ordinary argument unification,
			# above, could never have bound it no matter what was passed) but
			# DOES appear in the function's own return type is inferable by
			# actually lowering the body, once every other (argument-bound)
			# type param is known - see _infer_return_only_type_params/
			# _infer_return_only_type_params_inline. A param appearing in
			# NEITHER any parameter nor the return type is a degenerate,
			# vacuous case - left to fail below like anything else genuinely
			# missing, not silently accepted.
			referenced = self.lowering._param_referenced_type_params_for( target )
			return_only_missing = [
				tv for tv in missing
				if id( tv ) not in referenced and self.lowering._type_mentions_param( target.return_type, tv )
			]
			genuinely_missing = [ tv for tv in missing if tv not in return_only_missing ]
			if genuinely_missing:
				# a call can't be "partially" rescued by the return-only path
				# while some other param remains genuinely stuck from the
				# arguments alone - lists every still-unbound name (not just
				# the genuinely-missing ones), same as before this branch
				# existed at all, since an explicit spelling would need to
				# supply ALL of them anyway (no partial explicit subscript
				# exists - see PLAN_RETURN_INFERENCE's own "Deferred")
				self.lowering.discovery.fail(
					f'{target.qualname}[...]: cannot infer type parameter(s) {", ".join(tv.stem for tv in missing)} from these arguments - '
					f'call it explicitly as {target.qualname}[...](...) instead: {ast.unparse(node)}',
					node,
				)
			if target.is_inline:
				return self._infer_return_only_type_params_inline(
					node, target, type_params, bindings, return_only_missing, receiver, args, kwargs, expected_type, want_result,
				)
			monomorphized = self._infer_return_only_type_params( node, target, type_params, bindings, return_only_missing )
			inferred_args = [ bindings[id(tv)] for tv in target.type_params or [] ]
			self.lowering._check_type_param_bounds( node, target.type_params or [], inferred_args, target.qualname )
			spec = self.lowering.discovery._get_or_create_specialization( target, inferred_args )
			args, kwargs = self._coerce_generic_call_args( monomorphized, args, kwargs, node )
			self._fill_generic_call_defaults( monomorphized, args, kwargs, node )
			return self._emit_generic_call( node, spec, monomorphized, receiver, args, kwargs, expected_type, want_result, already_compiled = True )
		inferred_args = [ bindings[id(tv)] for tv in target.type_params or [] ]
		self.lowering._check_type_param_bounds( node, target.type_params or [], inferred_args, target.qualname )
		spec = self.lowering.discovery._get_or_create_specialization( target, inferred_args )
		monomorphized = self.lowering._monomorphized_function( spec )
		args, kwargs = self._coerce_generic_call_args( monomorphized, args, kwargs, node )
		self._fill_generic_call_defaults( monomorphized, args, kwargs, node )
		if monomorphized.is_inline:
			return self._lower_inline_call( node, monomorphized, receiver, args, kwargs, expected_type, want_result )
		return self._emit_generic_call( node, spec, monomorphized, receiver, args, kwargs, expected_type, want_result )

	def _lower_overload_generic_call( self, node: ast.Call, target: Function, receiver: ir.Operand|None, args: list[ir.Operand], kwargs: dict[str,ir.Operand], expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# an Overload group's own overload_resolution.resolve_call picked a
		# GENERIC candidate (target.type_params truthy - e.g. a `[T](x: T)`
		# alternative sharing a name with one or more concrete overloads) as
		# either the sole unconditional match or the trailing default of a
		# runtime ConditionalDispatch. Unlike _lower_inferred_generic_call,
		# args/kwargs are ALREADY lowered operands here (resolve_call needed
		# their real types before it could even pick this candidate - see the
		# Overload branch's own _lower_overload_arg call, above) - so there's
		# no interleaved lower+unify to do, just unification directly against
		# each already-known operand's own .type, then the identical
		# monomorphize-and-emit tail _lower_inferred_generic_call itself
		# funnels into via _finish_generic_call.
		assert target.resolve is None, f'internal compiler error - {target=} was not fully resolved by overload_resolution.resolve_call'
		type_params = target.type_params or []
		bindings: dict[int,Type] = {}
		for param, operand in zip( target.parameters or [], args ):
			self.lowering._unify_type_param( type_params, param.type, operand.type, bindings, node, target.qualname )
		for param in target.parameters or []:
			if param.stem in kwargs:
				self.lowering._unify_type_param( type_params, param.type, kwargs[param.stem].type, bindings, node, target.qualname )
		return self._finish_generic_call( node, target, type_params, bindings, receiver, args, kwargs, expected_type, want_result )

	# defensive cap on how many distinct per-tag monomorphizations
	# _expand_dispatch_target will synthesize for ONE generic branch -
	# mirrors overload_resolution.py's own _MAX_TRACKED_STATES spirit (a
	# named, trivially-adjustable constant, not a hard architectural limit).
	# Every real lib/ overload group is 1-2 params/2-4 leaves - nowhere near
	# this before it'd be a genuine sign of a mis-scoped overload group
	# rather than a legitimate need for more combos
	_MAX_DISPATCH_COMBOS = 64

	def _dispatch_remaining_leaves(
		self, node: ast.Call, target: Function, param: Parameter,
		known: dict[int,Type], args: list[ir.Operand], kwargs: dict[str,ir.Operand], claimed: dict[int,list[Type]],
	) -> list[Type]:
		# every leaf `param` could still be, once THIS branch's own runtime
		# tag check(s) (if any) have already excluded every other candidate -
		# see _expand_dispatch_target's own comment. `known` (keyed by
		# id(param)) covers whatever this branch's own condition (or an
		# earlier _expand_dispatch_target combo) already pinned down
		# explicitly - returned as the sole element, no further narrowing
		# needed. Every OTHER parameter is narrowed from the real call-site
		# operand's own (possibly still union) type, minus whatever leaves
		# `claimed` (built from every OTHER branch's own conditions - only
		# ever non-empty for the trailing default) already accounts for
		# elsewhere. Exactly one leaf here means this parameter is statically
		# resolvable (_monomorphize_dispatch_target); more than one is what
		# _expand_dispatch_target splits into distinct per-leaf branches.
		if id( param ) in known:
			return [ known[ id( param ) ] ]
		operand = self.lowering._dispatch_operand_for_param( node, target, param, args, kwargs )
		already = claimed.get( id( operand ), [] )
		return [
			leaf for leaf in operand.type.leaves()
			if not any( self.lowering._type_resolver._same_type( leaf, c ) for c in already )
		]

	def _monomorphize_dispatch_target(
		self, node: ast.Call, target: Function,
		known: dict[int,Type], args: list[ir.Operand], kwargs: dict[str,ir.Operand], claimed: dict[int,list[Type]],
	) -> Function:
		# monomorphizes a GENERIC branch/default of a runtime-dispatched
		# Overload call in place, so the rest of _lower_conditional_dispatch
		# (which schedules every branch's target as an ordinary, concrete
		# compile unit) never sees a bare TypeVar parameter. Unlike
		# _lower_overload_generic_call (which unifies T against the FULL,
		# possibly-union call-site operand type, for the "sole unconditional
		# match" case), a ConditionalDispatch branch/default only ever runs
		# once every OTHER branch's own runtime tag check has excluded its own
		# leaf(s) - _dispatch_remaining_leaves resolves the real, narrower
		# type(s) reaching THIS target at each parameter. Caller contract:
		# every parameter must already resolve to EXACTLY one leaf here (see
		# _expand_dispatch_target, the only real caller) - asserted, not
		# re-validated, since by the time this is called any genuine
		# ambiguity has already been split into a separate combo.
		type_params = target.type_params or []
		bindings: dict[int,Type] = {}
		for param in target.parameters or []:
			remaining = self._dispatch_remaining_leaves( node, target, param, known, args, kwargs, claimed )
			assert len( remaining ) == 1, (
				f'internal compiler error - {target.qualname} parameter {param.stem!r} not resolved to exactly '
				f'one leaf before monomorphizing ({len(remaining)} remaining) - _expand_dispatch_target caller contract violated'
			)
			self.lowering._unify_type_param( type_params, param.type, remaining[0], bindings, node, target.qualname )
		missing = [ tv for tv in type_params if id( tv ) not in bindings ]
		if missing:
			# every real lib/ generic overload binds every type param
			# directly off a parameter (see this module's own note on
			# real-world overload group shapes) - return-only inference
			# (_infer_return_only_type_params) is a materially bigger
			# feature to wire through a runtime-dispatched branch (it needs
			# to actually lower the body to infer the return type) and isn't
			# attempted here; fails clearly rather than silently
			self.lowering.discovery.fail(
				f'a generic overload of {target.qualname} cannot be one branch of a runtime-dispatched call '
				f'(type parameter(s) {", ".join(tv.stem for tv in missing)} aren\'t bound by any parameter - '
				f'return-only inference isn\'t supported here yet): {ast.unparse(node)}',
				node,
			)
		inferred_args = [ bindings[id(tv)] for tv in type_params ]
		spec = self.lowering.discovery._get_or_create_specialization( target, inferred_args )
		return self.lowering._monomorphized_function( spec )

	def _remap_conditions(
		self, original_params: list[Parameter], new_params: list[Parameter], conditions: list[tuple[Parameter,Type]],
	) -> list[tuple[Parameter,Type]]:
		# a monomorphized Function has its own, distinct Parameter objects
		# (same count/order as the generic original - substitution never
		# reorders or drops parameters) - a runtime condition built against
		# the ORIGINAL generic function's own Parameter identity (from
		# resolve_call, or from an earlier _expand_dispatch_target combo)
		# has to be re-pointed at the monomorphized function's corresponding
		# one before _lower_dispatch_tests/_dispatch_operand_for_param can
		# find it there (identity lookup, not structural equality - see
		# their own docstrings)
		return [
			( new_params[ next( i for i, op in enumerate( original_params ) if op is p ) ], leaf_type )
			for p, leaf_type in conditions
		]

	def _expand_dispatch_target(
		self, node: ast.Call, target: Function,
		known: dict[int,Type], args: list[ir.Operand], kwargs: dict[str,ir.Operand], claimed: dict[int,list[Type]],
	) -> list[tuple[list[tuple[Parameter,Type]],Function]]:
		'''
		Resolves a GENERIC branch/default of a runtime-dispatched Overload
		call into one or more concrete (extra_conditions, monomorphized
		Function) pairs. Most of the time this is exactly one entry with no
		extra conditions - the EASY case (_monomorphize_dispatch_target's own
		docstring): every parameter's real leaf is already pinned to exactly
		one value here, either explicitly (`known`, from this branch's own
		runtime condition) or by elimination (`claimed`, only ever populated
		for the trailing default).

		When one or more parameters can still legitimately be more than one
		leaf here - the HARD case this dispatch machinery used to reject
		outright - this ONE branch is split into one synthetic entry PER
		combination of those parameters' remaining leaves, each monomorphized
		with its own distinct T binding and given its own extra runtime
		condition(s) pinning exactly that combination. This turns "this one
		generic branch might need any of N different C functions at runtime,
		selected by a tag no single Call target can express" into N ordinary,
		individually-concrete branches - exactly the shape
		_lower_conditional_dispatch already knows how to schedule, just more
		of them. The caller (the Overload branch of _lower_call) is
		responsible for splicing these into the overall branches/default
		list and picking exactly one overall entry to remain the trailing,
		unconditioned default.
		'''
		params = target.parameters or []
		remaining_by_id = {
			id( p ): self._dispatch_remaining_leaves( node, target, p, known, args, kwargs, claimed )
			for p in params
		}
		ambiguous = [ p for p in params if len( remaining_by_id[ id( p ) ] ) != 1 ]
		if not ambiguous:
			return [ ( [], self._monomorphize_dispatch_target( node, target, known, args, kwargs, claimed )) ]
		combo_count = math.prod( len( remaining_by_id[ id( p ) ] ) for p in ambiguous )
		if combo_count > self._MAX_DISPATCH_COMBOS:
			self.lowering.discovery.fail(
				f'a generic overload of {target.qualname} used as a runtime-dispatched branch would need '
				f'{combo_count} separate per-type monomorphizations here (more than {self._MAX_DISPATCH_COMBOS}) - '
				f'narrow the overloaded parameter types: {ast.unparse(node)}',
				node,
			)
		results: list[tuple[list[tuple[Parameter,Type]],Function]] = []
		for combo in itertools.product( *( remaining_by_id[ id( p ) ] for p in ambiguous )):
			combo_known = dict( known )
			combo_known.update({ id( p ): leaf for p, leaf in zip( ambiguous, combo ) })
			monomorphized = self._monomorphize_dispatch_target( node, target, combo_known, args, kwargs, claimed )
			new_params = monomorphized.parameters or []
			extra_conditions = self._remap_conditions( params, new_params, list( zip( ambiguous, combo )))
			results.append(( extra_conditions, monomorphized ))
		return results

	def _infer_return_only_type_params( self, node: ast.Call, target: Function, type_params: list[TypeVar], bindings: dict[int,Type], return_only: list[TypeVar] ) -> Function:
		# PLAN_RETURN_INFERENCE.md - non-@inline variant: a bare generic
		# call left one or more type params unbound after ordinary
		# argument unification, each appearing ONLY in target's own return
		# type (never in any parameter) - determined by actually, eagerly,
		# synchronously compiling the function's body (with every OTHER,
		# argument-bound type param already substituted) and reading the
		# real return type back off it, generalizing _expr_Lambda's own
		# eager-lowering trick (PLAN_LAMBDA.md) from an unbound Callable's
		# own return type to a named generic function's own return type.
		# Mutates `bindings` in place to add the newly-discovered args, and
		# returns the (now fully concrete) monomorphized Function - already
		# a real, compiled unit (appended to compiler.functions by
		# _compile_now), never rebuilt - for the caller to finish emitting
		# a Call against.
		pending_args = [ bindings.get( id( tv ), tv ) for tv in type_params ]
		pending_spec = self.lowering.discovery._get_or_create_specialization( target, pending_args )
		if pending_spec.monomorphized is not None:
			# another call site with the SAME known args already discovered
			# the return-only bindings and compiled this - no new compile,
			# just recover the bindings THIS call site's own `bindings`
			# dict still needs filled in (the caller reads it right after
			# this returns to build its own Specialization args)
			self.lowering._unify_type_param( type_params, target.return_type, pending_spec.monomorphized.return_type, bindings, node, target.qualname )
			return pending_spec.monomorphized

		if id( target ) in self.lowering._eager_return_inference_stack:
			self.lowering.discovery.fail(
				f'{target.qualname}[...]: cannot infer return-only type parameter(s) {", ".join(tv.stem for tv in return_only)} - '
				f'the body (directly, or through another generic function) recursively calls itself before its own return type is '
				f'known - call it explicitly as {target.qualname}[...](...) instead: {ast.unparse(node)}',
				node,
			)
		if not self.lowering.discovery._is_eager_return_inferable_body( target.node.body ):
			self.lowering.discovery.fail(
				f'{target.qualname}[...]: cannot infer return-only type parameter(s) {", ".join(tv.stem for tv in return_only)} - '
				f'its body must have exactly one reachable `return <expr>` for this to work - call it explicitly as '
				f'{target.qualname}[...](...) instead: {ast.unparse(node)}',
				node,
			)

		self.lowering._eager_return_inference_stack.append( id( target ))
		try:
			# qualname deliberately the PENDING spec's own (not target's bare
			# abstract name) - already unambiguous per known-args binding,
			# and gives compile errors reached WHILE eagerly lowering this
			# body a sensible name too, not the shared, unspecialized one
			provisional = self.lowering._monomorphizer._build_monomorphized_function( target, type_params, pending_args, pending_spec.qualname )
			# forces _stmt_Return's own _lower_expr(node.value, self.
			# _current_fn.return_type) to lower the return expression with
			# NO hint, taking its own natural type - _expr_Lambda's
			# identical return_type_provisional convention, reused verbatim
			provisional.return_type = None
			lowered = self.lowering._compile_now( provisional ) # == Compiler._lower - this IS the final compiled unit, never rebuilt
			# NOT a bare next(...) (unlike _expr_Lambda's own identical-
			# looking lookup, PLAN_LAMBDA.md line ~3555 - a real, pre-
			# existing gap there too, not fixed here, out of scope for this
			# plan): the ONE statement in `provisional`'s own body can
			# itself fail to lower (e.g. the reentrancy guard above,
			# triggered one level deeper by a recursive call inside the
			# body) - FunctionLowering.run()'s own per-statement recovery
			# (try/except CompileError: continue) SWALLOWS that failure
			# silently rather than propagating it here, leaving `lowered.
			# instructions` with no ir.Return at all. A bare next(...) would
			# crash with an unhandled StopIteration instead of a clean
			# compile error - confirmed by a real repro (direct/mutual
			# recursion through this same path). The real error is already
			# recorded in discovery.errors by whatever failed deeper in the
			# body (the reentrancy guard's own fail(), most commonly) -
			# this fail() call is a second, redundant-but-harmless report,
			# same accepted pattern resolve_function_body's own docstring
			# already documents ("gets reported again... so nothing is
			# silently swallowed")
			return_instr = next( ( instr for instr in lowered.instructions if isinstance( instr, ir.Return ) ), None )
			if return_instr is None:
				self.lowering.discovery.fail(
					f'{target.qualname}[...]: cannot infer return-only type parameter(s) {", ".join(tv.stem for tv in return_only)} - '
					f'the body failed to compile (see earlier error): {ast.unparse(node)}',
					node,
				)
			actual_return_type = (
				return_instr.value.type if return_instr.value is not None
				else self.lowering.discovery.get_none_type()
			)
			self.lowering._unify_type_param( type_params, target.return_type, actual_return_type, bindings, node, target.qualname )
		finally:
			self.lowering._eager_return_inference_stack.pop()

		still_missing = [ tv.stem for tv in return_only if id( tv ) not in bindings ]
		if still_missing:
			self.lowering.discovery.fail(
				f'{target.qualname}[...]: cannot infer type parameter(s) {", ".join(still_missing)} - the function\'s own declared '
				f'return type does not structurally match what its body actually returns: {ast.unparse(node)}',
				node,
			)
		full_args = [ bindings[id(tv)] for tv in type_params ]
		real_spec = self.lowering.discovery._get_or_create_specialization( target, full_args )
		# patch the SAME object in place afterward, not rebuilt - mirrors
		# _expr_Lambda's own "reused afterward" convention exactly
		provisional.return_type = self.lowering._substitute_type_params( target.return_type, type_params, full_args )
		if isinstance( provisional.return_type, GeneratorType ):
			# target itself is a delegating wrapper (-> Generator[T,E]/
			# Iterator[...], no yield of its own - e.g. a @protocol
			# Iterable[T].__iter__ method, or iter() itself, forwarding to
			# a real generator call) whose element/error types were only
			# return-only-inferable - the substitute_type_params() call just
			# above rebuilds a fresh, still-ABSTRACT GeneratorType from
			# target's own annotation (correct for an ordinary function, but
			# not for one of these: the real return value `provisional`'s
			# body actually produces is a concrete generator backing class,
			# already computed above as actual_return_type, then silently
			# discarded once bindings absorbed it). Re-run the SAME pass-
			# through resolution ensure_generator_synthesized's non-yield
			# branch already does elsewhere, now that `provisional` (id
			# never seen before - freshly built above) has a real body and
			# real, substituted parameter types to resolve its own call
			# against - confirmed necessary by a real repro (a for-loop
			# consuming iter(...)'s result otherwise saw the bare abstract
			# GeneratorType, not a real __next__-bearing class).
			self.lowering._type_resolver.ensure_generator_synthesized( provisional )
		provisional.qualname = real_spec.qualname
		real_spec.monomorphized = provisional # the real key, for a FUTURE fully-concrete lookup (e.g. an explicit foo[Concrete,Other](...) call elsewhere)
		pending_spec.monomorphized = provisional # the pending key, for a FUTURE bare call with the same already-known args
		return provisional

	def _infer_return_only_type_params_inline( self, node: ast.Call, target: Function, type_params: list[TypeVar], bindings: dict[int,Type], return_only: list[TypeVar], receiver: ir.Operand|None, args: list[ir.Operand], kwargs: dict[str,ir.Operand], expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# PLAN_RETURN_INFERENCE.md - @inline variant: cheaper than
		# _infer_return_only_type_params above because splicing the body IS
		# the eager compile already - no separate FuncStart/CFG/real
		# compiled unit needed for target itself here (matches PLAN_INLINE.
		# md's own "never a real compiled unit" invariant), just the
		# AST-level generic-call-resolution rewrite (resolve_function_body)
		# a NESTED generic call inside the body would need, same as the
		# ordinary is_inline path already gets via _monomorphized_function.
		# _is_eager_return_inferable_body is NOT re-checked here - @inline's
		# own decorator-time _is_inline_eligible_body (exactly one TOP-LEVEL
		# `return <expr>`) is strictly stronger; every @inline-eligible body
		# already trivially satisfies it.
		pending_args = [ bindings.get( id( tv ), tv ) for tv in type_params ]
		pending_spec = self.lowering.discovery._get_or_create_specialization( target, pending_args )
		if pending_spec.monomorphized is not None:
			# another call site already discovered R for these known args -
			# splice against the cached provisional directly. `bindings`
			# doesn't need filling in here (unlike the non-inline variant) -
			# this call RETURNS the final operand straight to _lower_call's
			# own caller, nothing downstream reads `bindings` again
			self._fill_generic_call_defaults( pending_spec.monomorphized, args, kwargs, node )
			return self._lower_inline_call( node, pending_spec.monomorphized, receiver, args, kwargs, expected_type, want_result )

		if id( target ) in self.lowering._eager_return_inference_stack:
			self.lowering.discovery.fail(
				f'@inline {target.qualname}[...]: cannot infer return-only type parameter(s) {", ".join(tv.stem for tv in return_only)} - '
				f'the body (directly, or through another generic function) recursively calls itself before its own return type is '
				f'known - call it explicitly as {target.qualname}[...](...) instead: {ast.unparse(node)}',
				node,
			)
		self.lowering._eager_return_inference_stack.append( id( target ))
		try:
			provisional = self.lowering._monomorphizer._build_monomorphized_function( target, type_params, pending_args, pending_spec.qualname )
			provisional.return_type = None
			# @inline never needs a real compiled unit for its own target -
			# only the same AST-rewrite pass the ordinary is_inline path
			# already runs on a monomorphized copy before splicing it, so a
			# nested generic call inside the body resolves against the
			# concrete, already-substituted T
			self.lowering._type_resolver.resolve_function_body( provisional )
			# see _fill_generic_call_defaults's own docstring - provisional's
			# own parameters are already substituted (built via
			# _build_monomorphized_function above), same as any other
			# generic call's monomorphized target
			self._fill_generic_call_defaults( provisional, args, kwargs, node )
			# forced True regardless of the real want_result - the real
			# operand (and its .type) is needed to discover the return-only
			# bindings even when the CALLER's own want_result is False; the
			# discard-Result check _lower_inline_call would otherwise apply
			# internally is skipped by forcing this (target.return_type is
			# still None here, so it would've been a no-op anyway) and
			# re-applied below instead, once the real return type is known
			result = self._lower_inline_call( node, provisional, receiver, args, kwargs, expected_type, want_result = True )
			actual_return_type = result.type if result is not None else self.lowering.discovery.get_none_type()
			self.lowering._unify_type_param( type_params, target.return_type, actual_return_type, bindings, node, target.qualname )
		finally:
			self.lowering._eager_return_inference_stack.pop()

		still_missing = [ tv.stem for tv in return_only if id( tv ) not in bindings ]
		if still_missing:
			self.lowering.discovery.fail(
				f'{target.qualname}[...]: cannot infer type parameter(s) {", ".join(still_missing)} - the function\'s own declared '
				f'return type does not structurally match what its body actually returns: {ast.unparse(node)}',
				node,
			)
		full_args = [ bindings[id(tv)] for tv in type_params ]
		self.lowering._check_type_param_bounds( node, type_params, full_args, target.qualname )
		real_spec = self.lowering.discovery._get_or_create_specialization( target, full_args )
		provisional.return_type = self.lowering._substitute_type_params( target.return_type, type_params, full_args )
		provisional.qualname = real_spec.qualname
		real_spec.monomorphized = provisional # caches the PROVISIONAL BODY for reuse by _lower_inline_call at a future call site - provisional is never independently scheduled/appended to compiler.functions by this variant, matching PLAN_INLINE.md's invariant
		pending_spec.monomorphized = provisional

		# a discarded Result return is auto-consumed here (case 1 of the
		# general auto-or_throw() rule - _finish_call_result), done here
		# instead of inside the _lower_inline_call call above since target.
		# return_type wasn't known yet at that point (want_result was
		# forced True there specifically to skip its own identical check)
		return self._finish_call_result( node, result, want_result )

	def _emit_generic_call( self, node: ast.Call, spec: Specialization, monomorphized: Function, receiver: ir.Operand|None, args: list[ir.Operand], kwargs: dict[str,ir.Operand], expected_type: Type|None, want_result: bool, *, already_compiled: bool = False, return_type: Type|None = None ) -> ir.Operand|None:
		# schedules the Specialization itself as the compile unit (see
		# _monomorphized_function/compiler.py's own handling of it), shared
		# tail for both the explicit Name[T](...) and inferred call paths -
		# and, unlike _lower_call's OWN shared tail (which only ever sees a
		# call type_resolver.py's pre-pass could tag with resolved_callee),
		# the ONLY tail a receiver-based generic method call reaches at all
		# (see _lower_class_generic_method_call's own comment on why that
		# one is always left untagged) - so the discard check needs its own
		# copy here too, not just in _lower_call's
		#
		# already_compiled=True (return-only type-parameter inference's own
		# eager-compile path - see _infer_return_only_type_params) means
		# `monomorphized` was already lowered for real, synchronously, via
		# Lowering._compile_now, and is already sitting in compiler.functions
		# - scheduling `spec` again here would re-enqueue it onto the
		# ordinary work queue, and Compiler._lower's Specialization+Function
		# branch has no "already lowered" check of its own: it would
		# unconditionally re-run resolve_function_body (which mutates
		# monomorphized.node.body IN PLACE - a second pass over an already-
		# AST-rewritten body) and lower_function a second time, producing a
		# duplicate LoweredFunction entry for the same qualname (a real
		# duplicate-symbol C compile error), not just wasted work. Every
		# other type/parameter still gets scheduled normally here - those
		# are idempotent registrations of TYPES, unrelated to re-lowering
		# monomorphized's own body
		if not already_compiled:
			self.lowering.schedule( spec )
		self.lowering.schedule( monomorphized.return_type )
		for param in monomorphized.parameters or []:
			self.lowering.schedule( param.type )
		# the REAL return type - monomorphized.return_type by default, but a
		# caller can override via `return_type` when monomorphized.return_type
		# itself isn't fully concrete (see _lower_class_generic_method_call's
		# own call site: a classmethod constructor's declared return type is
		# the ENCLOSING class's bare TaggedUnion, referencing the class's OWN
		# type params - not substituted by _monomorphized_function, which
		# only substitutes the METHOD's own type params. That caller passes
		# the real concrete Specialization it already computed instead).
		real_return_type = return_type if return_type is not None else monomorphized.return_type
		# a discarded Result return is no longer hard-rejected here - case 1
		# of the general auto-or_throw() rule (_finish_call_result, below)
		# picks it up instead, so this call still needs a real dest to
		# consume even though the CALLER's own want_result is False
		force_result = not want_result and cfg.is_result_type( real_return_type )
		# discarded plain (non-Result) RC return - see _lower_call's own
		# identical discard_rc for why this still needs a real dest
		discard_rc = not want_result and not force_result and real_return_type is not None and real_return_type.is_rc()
		if want_result or force_result or discard_rc:
			# real_return_type, NOT `expected_type or ...` (the convention
			# every OTHER call-emission tail in this file uses) - for a
			# GENERIC call specifically, the type params (and so the real
			# return type) are only resolved from the ARGUMENTS, fully
			# independent of expected_type; nothing before this point ever
			# required them to agree. Typing `dest` as expected_type
			# directly used to make that agreement TRUE BY CONSTRUCTION
			# (skipping this file's own real safety net - _lower_expr's tail
			# calls _coerce_or_check_operand against expected_type right
			# after this returns, but only ever a genuine MISMATCH if
			# operand.type differs from expected_type in the first place) -
			# so a real divergence went undetected, reaching emitter_c.py
			# with `dest` LYING about its own type (a bare literal argument
			# happening to always default to i32, matching whatever the
			# caller expected, is what masked this: confirmed via a real
			# repro once literals started defaulting to builtins.int
			# instead). dest now always carries the REAL type; a genuine
			# mismatch is caught (coerced, or a proper "expected X, got Y"
			# error) by that same outer _coerce_or_check_operand call,
			# exactly as it already does for every other kind of operand.
			dest = self._new_temp( real_return_type )
			self._emit( ir.Call( dest = dest, target = monomorphized, receiver = receiver, args = args, kwargs = kwargs ))
			return self._finish_call_result( node, dest, want_result )
		self._emit( ir.Call( dest = None, target = monomorphized, receiver = receiver, args = args, kwargs = kwargs ))
		return None

	def _lower_class_generic_method_call( self, node: ast.Call, target: Function, receiver: ir.Operand|None, expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# a method whose genericity is inherited from its enclosing class
		# (Result.Ok/.Err/.is_ok/.is_err/... referencing Result's own T,E)
		# rather than declared on the method itself (unlike sys.alloc[T]) -
		# target.type_params is empty, but target.cls.type_params isn't.
		#
		# only ever reached with NO receiver (a static/classmethod reached
		# via bare class name, e.g. Result.Ok(y)) - a receiver whose own
		# type already pins down cls's concrete args never gets here at all:
		# _attr_lookup_callable already hands back an already-substituted
		# Function for that case (see monomorphize.py/_ensure_resolved),
		# whose own .cls is the concrete Specialization, not the abstract
		# cls this dispatch condition (_lower_call) checks .type_params on -
		# confirmed by instrumenting this branch and running the full test
		# suite, not just by this reasoning alone. So the class's own
		# concrete type args always have to be INFERRED here, the same way
		# _lower_inferred_generic_call infers a free function's own type
		# params, with one addition: unify expected_type against the
		# method's still-abstract return type FIRST, before lowering any
		# argument - Result.Ok(val: T) ->
		# Result[T,E] never mentions E in its own parameter list at all
		# (only inferable from context), and even T needs to be known
		# BEFORE a bare literal argument (Result.Ok(5)) can be lowered at
		# all (_expr_Constant needs a real expected type, not a raw
		# TypeVar) - unlike a free generic function, where a literal
		# argument at an inferred position is simply unsupported (see
		# _lower_inferred_generic_call's own comment), the surrounding
		# expected_type is usually enough to resolve every class type
		# param here without needing the arguments' own types at all
		assert target.resolve is None, f'internal compiler error - {target=} was not fully resolved by the type_resolver module'
		cls = target.cls
		class_type_params = cls.type_params or [] if cls is not None else []
		bindings: dict[int,Type] = {}
		if expected_type is not None:
			self.lowering._unify_type_param( class_type_params, target.return_type, expected_type, bindings, node, target.qualname )

		args, kwargs = self._lower_and_infer_call_args( node, target, class_type_params, bindings, target.qualname )

		missing = [ tv.stem for tv in class_type_params if id( tv ) not in bindings ]
		if missing:
			self.lowering.discovery.fail(
				f'{target.qualname}(...): cannot infer {cls.qualname if cls else "?"} type parameter(s) '
				f'{", ".join(missing)} from these arguments or the surrounding expected type: {ast.unparse(node)}',
				node,
			)
		cls_args = [ bindings[id(tv)] for tv in class_type_params ]
		self.lowering._check_type_param_bounds( node, class_type_params, cls_args, cls.qualname if cls else target.qualname )
		method_spec = self.lowering.discovery._get_or_create_specialization( target, cls_args )
		monomorphized = self.lowering._monomorphized_function( method_spec )
		self._fill_generic_call_defaults( monomorphized, args, kwargs, node )
		# monomorphized.return_type is NOT reliable here (see
		# _emit_generic_call's own `return_type` param comment) -
		# _monomorphized_function only substitutes the METHOD's own type
		# params (empty for a classmethod like Result.Ok), never the
		# ENCLOSING class's (class_type_params/cls_args, resolved just
		# above) that target.return_type is actually expressed in terms of
		# (e.g. `Result.Ok(val: T) -> Result[T,E]`, T/E being the class's
		# own, not the method's). expected_type itself is the right value
		# here (falls back to monomorphized.return_type when it's None,
		# same as _emit_generic_call's own default for every other caller)
		# - UNLIKE _lower_inferred_generic_call's own generic-function path,
		# this method already unified expected_type against target.
		# return_type/class_type_params right at its own top (see `bindings`
		# above), so trusting it here is well-founded, not the same
		# "silently diverges from the real return type" hazard that path
		# had (confirmed: substituting target.return_type through cls_args
		# directly instead, the seemingly more "principled" alternative,
		# produces a DIFFERENT type REPRESENTATION than expected_type/
		# _get_or_create_specialization do for a CStruct-based generic - a
		# real regression caught by this file's own IR-shape tests, not
		# just theory).
		return self._emit_generic_call( node, method_spec, monomorphized, receiver, args, kwargs, expected_type, want_result, return_type = expected_type )
