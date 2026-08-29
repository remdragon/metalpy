# stdlib imports:
import ast
from typing import Callable

# local imports:
import cfg
import ir
from errors import RedundantCompilationError
from mpy_types import (
	Name, Type, Variable, Parameter, Function, Specialization, CStruct, TypeVar, RCClass, Scalar, CallableType, ClosureType,
)
from type_resolver import TypeResolver

class ClosureLoweringMixin:
	''' closures, lambdas, nested defs, and closure-trampoline construction - mixed into FunctionLowering (lowering.py), which
	see for the shared instance state (self._instructions, self._cfg, self.lowering,
	etc.) every method here reads and writes. Never instantiated on its own;
	split out of lowering.py purely to keep that file to a manageable size - see
	lowering.py's own class docstring and FunctionLowering's base-class list for
	the full set of sibling mixins this one is composed with. '''


	def _collect_free_variables( self, roots: list[ast.AST], param_names: set[str], node: ast.AST ) -> list[tuple[str,Variable]]:
		# a nested def/lambda's own parameters/locally-assigned names,
		# module-level names, and builtins resolve normally; anything else
		# reaching into the immediately enclosing function's own scope is a
		# CAPTURE - collected and returned here (as (name, Variable) pairs,
		# ready for _build_closure_env_class/the AST rewrite - see the
		# capturing-closures plan) rather than rejected, now that a
		# representation decision has been made (real closures - see
		# ClosureType/PLAN_CALLABLE.md's own bound-method precedent,
		# generalized). `roots` is the def's own body (a list of statements)
		# or a lambda's own body wrapped in a single-element list (a bare
		# expression - lambda syntax forbids assignment statements, but NOT
		# ast.NamedExpr/walrus, which also binds via Name(Store) - the same
		# walk covers both shapes uniformly without special-casing).
		# Doesn't recurse into a FURTHER nested def/lambda's own body - that
		# one gets its own independent capture collection when IT gets
		# synthesized (only its OWN name, if it's a def, becomes a local
		# binding at THIS level, same as an ordinary assignment would); this
		# is also the reason a closure capturing another closure's own
		# capture isn't supported yet (see the plan's "Deferred" section) -
		# by the time an inner nested def/lambda's own free-variable walk
		# runs, THIS level's own rewrite has already turned any name it
		# captured into an Attribute expression, never a resolvable name.
		#
		# `x = x + 1` inside the body never reaches `free` at all: any
		# ast.Store anywhere in the body makes that name local for the
		# WHOLE body (mirroring real Python's own hoisting rule), so this
		# is also what makes captures strictly immutable snapshots (no
		# nonlocal write-back) - a captured-and-reassigned name is simply a
		# local read of its own not-yet-assigned local, not a capture,
		# enforced with no extra checking needed here.
		local_names = set( param_names )
		class _BindingCollector( ast.NodeVisitor ):
			def visit_FunctionDef( self, fd: ast.FunctionDef ) -> None:
				local_names.add( fd.name )
			def visit_Lambda( self, lam: ast.Lambda ) -> None:
				pass
			def visit_Name( self, n: ast.Name ) -> None:
				if isinstance( n.ctx, ast.Store ):
					local_names.add( n.id )
		binder = _BindingCollector()
		for root in roots:
			binder.visit( root )

		free: list[ast.Name] = []
		class _LoadCollector( ast.NodeVisitor ):
			def visit_FunctionDef( self, fd: ast.FunctionDef ) -> None:
				pass
			def visit_Lambda( self, lam: ast.Lambda ) -> None:
				pass
			def visit_Name( self, n: ast.Name ) -> None:
				if isinstance( n.ctx, ast.Load ) and n.id not in local_names:
					free.append( n )
		loader = _LoadCollector()
		for root in roots:
			loader.visit( root )

		enclosing_fn = self._current_fn
		if enclosing_fn is None:
			return []
		captures: list[tuple[str,Variable]] = []
		seen: set[str] = set()
		for free_name in free:
			if free_name.id in seen:
				continue
			resolved = enclosing_fn.names.get( free_name.id )
			if resolved is None:
				continue # genuinely undefined - falls through to the ordinary "not defined" error once the (rewritten) body is actually lowered
			if not isinstance( resolved, Variable ):
				self.lowering.discovery.fail(
					f"{ast.unparse(node)}: cannot capture {free_name.id!r} - only local variables and "
					f"parameters can be captured, not {type(resolved).__name__.lower()}s",
					node,
				)
			seen.add( free_name.id )
			captures.append( ( free_name.id, resolved ) )
		return captures

	def _rewrite_captures_into_env_reads( self, roots: list[ast.AST], env_cls: RCClass, erased_param: str, captured_names: set[str] ) -> list[ast.AST]:
		''' replaces every captured-name ast.Name(Load) reference inside
		`roots` with an inline, never-named env-field read:
		compiler.cast(<env>, erased_param).name - mirrors
		_get_or_create_closure_trampoline's own inline compiler.cast(
		<closure_owner>, erased_self) receiver expression, for the
		identical reason: binding the cast to a named local first would
		make _is_aliasing_expr treat it as a fresh, owned value needing its
		own scope-exit decref, over-releasing the env object this is only
		ever a BORROWED reinterpretation of (the closure's own `self`
		field already owns it - see _construct_capturing_closure). Doesn't
		recurse into a FURTHER nested def/lambda's own body - same posture
		_collect_free_variables's own walk takes, for the same reason (see
		its own comment on why multi-level capture-of-a-capture isn't
		supported yet). Mutates/replaces in place - each occurrence's own
		AST is synthesized and lowered exactly once, never reused for
		anything else afterward, so there's no aliasing hazard in doing so. '''
		def env_field_read( name: str, line: int, col: int ) -> ast.Attribute:
			env_type_ref = ast.Name( id = '<closure_env>', ctx = ast.Load(), lineno = line, col_offset = col )
			env_type_ref.resolved_type = env_cls
			cast_call = ast.Call(
				func = ast.Attribute(
					value = ast.Name( id = 'compiler', ctx = ast.Load(), lineno = line, col_offset = col ),
					attr = 'cast', ctx = ast.Load(), lineno = line, col_offset = col,
				),
				args = [ env_type_ref, ast.Name( id = erased_param, ctx = ast.Load(), lineno = line, col_offset = col ) ],
				keywords = [], lineno = line, col_offset = col,
			)
			return ast.Attribute( value = cast_call, attr = name, ctx = ast.Load(), lineno = line, col_offset = col )

		class _CaptureRewriter( ast.NodeTransformer ):
			def visit_FunctionDef( self, fd: ast.FunctionDef ) -> ast.FunctionDef:
				return fd # don't recurse into a further nested def's own body
			def visit_Lambda( self, lam: ast.Lambda ) -> ast.Lambda:
				return lam # ditto for a further nested lambda
			def visit_Name( self, n: ast.Name ) -> ast.AST:
				if isinstance( n.ctx, ast.Load ) and n.id in captured_names:
					return env_field_read( n.id, n.lineno, n.col_offset )
				return n

		rewriter = _CaptureRewriter()
		return [ rewriter.visit( root ) for root in roots ]

	def _lower_function_ref( self, fn: Function, node: ast.AST ) -> ir.Operand:
		# a bare reference to a function used AS A VALUE, not called - see
		# PLAN_CALLABLE.md. Only a plain, receiver-less, non-generic,
		# non-overloaded function can become a Ptr[Callable[...]] value:
		# a bound instance method or classmethod has an implicit receiver
		# with nowhere to go in a raw C function pointer (that's a closure,
		# deliberately out of scope for now - see the plan doc's own
		# "deferred" list), a generic function has no single fixed
		# signature to point at (which specialization?), and an overload
		# group's own individual Functions are ambiguous by name alone
		if fn.type_params:
			self.lowering.discovery.fail( f'{fn.qualname} is generic - cannot take a bare reference to it: {ast.unparse(node)}', node )
		if fn.is_overload:
			self.lowering.discovery.fail( f'{fn.qualname} is one of several @overload implementations - cannot take a bare reference to it by name alone: {ast.unparse(node)}', node )
		if fn.cls is not None and not fn.is_static:
			self.lowering.discovery.fail( f'{fn.qualname} is an instance method or classmethod - cannot take a bare reference to it (no receiver to bind): {ast.unparse(node)}', node )
		if fn.cls is not None and fn.cls.type_params:
			# a @staticmethod belonging to a GENERIC class, referenced bare
			# from within another method of that SAME class - discovery.
			# find_name only ever returns the abstract template (fn.cls
			# itself, K/V still bare TypeVars: confirmed - _value_spelling
			# crashes on the unresolved TypeVar downstream in emitter_c.py
			# without this). Ordinary calls (self.method(...)) never hit
			# this because _lower_generic_call_with_receiver substitutes
			# the class type args and calls _monomorphized_function BEFORE
			# emitting the Call - a bare reference has no such call site to
			# hang that on, so it's done here instead, using the CURRENT
			# specialization being lowered (self._current_fn.cls) rather
			# than inferring from arguments (there's nothing to infer from
			# for a value reference - Ptr[Callable[...]]'s own shape
			# carries no class-type-param information at all)
			current_cls = self._current_fn.cls if self._current_fn is not None else None
			current_spec = current_cls if isinstance( current_cls, Specialization ) else None
			if current_spec is None or current_spec.base is not fn.cls:
				self.lowering.discovery.fail(
					f"{fn.qualname} belongs to a generic class - a bare reference to it is only resolvable from "
					f"inside one of {fn.cls.qualname}'s own (already-specialized) methods: {ast.unparse(node)}",
					node,
				)
			method_spec = self.lowering.discovery._get_or_create_specialization( fn, current_spec.args )
			fn = self.lowering._monomorphized_function( method_spec )
		self.lowering._ensure_resolved( fn )
		if fn.broken:
			raise RedundantCompilationError() # already reported at the point fn's own resolution failed - see Name.broken
		if fn.parameters is None:
			self.lowering.discovery.fail( f'{fn.qualname} could not be resolved (see earlier error): {ast.unparse(node)}', node )
		return self.lowering._function_ref_operand( fn )

	def _stmt_FunctionDef( self, node: ast.FunctionDef ) -> None:
		# a non-capturing nested function def - see PLAN_LAMBDA.md.
		# Synthesized as an independent, fully real Function (real
		# parameter/return annotations, resolved exactly like an ordinary
		# top-level function via discovery.visit(...) - the same mechanism
		# _stmt_AnnAssign already uses mid-lowering for an ordinary local's
		# own annotation), scheduled and compiled like any other compile
		# unit, and made callable/bare-referenceable by name for the rest
		# of the enclosing function's own body. The statement itself emits
		# no IR - a def only binds a name, same as Python
		enclosing = self._current_fn
		if enclosing is None:
			# neither message here unparses `node` itself - that would dump
			# this nested function's entire BODY into the error (a real
			# repro: a multi-statement nested def produced a multi-line
			# error that buried the actual problem); node.name already
			# identifies which def, and node's own lineno (the location
			# passed to fail) already pinpoints it
			self.lowering.discovery.fail( f'nested function def outside any function: {node.name}', node )
		self.lowering._reject_generic_enclosing_scope( enclosing, node, 'nested function defs' )
		if node.decorator_list:
			self.lowering.discovery.fail(
				f'{node.name}: decorators are not supported on a nested function def: '
				f'{", ".join( "@" + ast.unparse(d) for d in node.decorator_list )}',
				node,
			)

		qualname = f'{enclosing.qualname}$$nested_{node.name}'
		synthetic = Function(
			stem = node.name, qualname = qualname,
			file = enclosing.file, line = node.lineno,
			cls = None, node = node,
			parameters = None, return_type = None,
			resolve = None,
		)

		args = node.args
		parameters: list[Parameter] = []
		def add_param( arg: ast.arg, default: ast.expr|None, **kind: bool ) -> None:
			if arg.annotation is None:
				# not ast.unparse(node) - qualname + the parameter name
				# already identify this exactly; unparsing the whole nested
				# function would dump its entire body into the message
				self.lowering.discovery.fail( f'{qualname} parameter {arg.arg!r} has no type annotation', node )
			param_type = self.lowering.discovery.visit( arg.annotation )
			self.lowering.discovery._reject_bare_interface_value_type( param_type, arg, f'{qualname} parameter {arg.arg!r}' )
			param = Parameter(
				stem = arg.arg, qualname = f'{qualname}.{arg.arg}',
				file = enclosing.file, line = node.lineno,
				type = param_type, default = default, **kind,
			)
			parameters.append( param )
			synthetic.add_name( param.stem, param )

		# `defaults` applies to the trailing N of posonlyargs+args combined
		# (an ast-module quirk) - left-pad with None so every positional
		# param lines up with its own default (or lack of one) - same
		# convention discovery.py's own _make_function_resolver uses
		positional = [ *args.posonlyargs, *args.args ]
		defaults = [ None ] * ( len( positional ) - len( args.defaults )) + list( args.defaults )
		for i, arg in enumerate( positional ):
			add_param( arg, defaults[i], is_posonly = i < len( args.posonlyargs ))
		if args.vararg is not None:
			add_param( args.vararg, None, is_vararg = True )
		for arg, default in zip( args.kwonlyargs, args.kw_defaults ):
			add_param( arg, default, is_kwonly = True )
		if args.kwarg is not None:
			add_param( args.kwarg, None, is_kwarg = True )

		synthetic.parameters = parameters
		if node.returns is not None:
			synthetic.return_type = self.lowering.discovery.visit( node.returns )
			self.lowering.discovery._reject_bare_interface_value_type( synthetic.return_type, node.returns, f'{qualname} return type' )
		else:
			synthetic.return_type = self.lowering.discovery.get_none_type()

		captures = self._collect_free_variables( node.body, { p.stem for p in parameters }, node )

		if not captures:
			# the common, zero-cost case: nothing to capture, so `node.name`
			# resolves straight to a real Function - callers reach it via
			# _lower_function_ref (a bare reference) or ordinary Call
			# resolution, exactly as before this feature existed
			enclosing.add_name( node.name, synthetic )
			self.lowering.schedule( synthetic )
			return

		# a CAPTURING nested def - unlike the non-capturing case above, this
		# genuinely emits IR at the def statement's own position (this IS
		# "closure creation time" for a nested def, the same point Python
		# itself creates the function object each time the statement
		# executes - so one inside a loop correctly rebuilds a fresh env +
		# closure every iteration). `node.name` can no longer resolve to a
		# bare Function: calling it must route through _try_lower_closure_
		# call, which only matches a Variable of ClosureType - see the
		# closures plan
		ptr_cls = self.lowering.discovery.get_intrinsics()['Ptr']
		none_type = self.lowering.discovery.get_none_type()
		ptr_none_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ none_type ] )

		env_cls = self.lowering._build_closure_env_class(
			[ ( name, resolved.type ) for name, resolved in captures ], f'{qualname}$$env', enclosing.file, node.lineno,
		)
		captured_names = { name for name, _ in captures }
		node.body = self._rewrite_captures_into_env_reads( node.body, env_cls, 'erased_env', captured_names )

		erased_env_param = Parameter( stem = 'erased_env', qualname = f'{qualname}.erased_env', file = enclosing.file, line = node.lineno, type = ptr_none_type )
		synthetic.parameters = [ erased_env_param, *parameters ]
		synthetic.add_name( 'erased_env', erased_env_param )

		operand = self._construct_capturing_closure( captures, env_cls, synthetic, node )

		# bind node.name as a real Variable (not a Function) - the same
		# "first assignment to a name with no prior declaration" tail
		# _stmt_Assign's own no-prior-declaration branch uses, minus
		# _declare_local's callback-based lowering (operand is already
		# lowered above). is_alias=False: operand is a fresh Allocate
		# result, same as any other first-time construction
		var = Variable( stem = node.name, qualname = f'{enclosing.qualname}.{node.name}', file = enclosing.file, line = node.lineno, type = operand.type, needs_uid_suffix = self._mark_fresh_local_declared( node.name ))
		enclosing.add_name( var.stem, var )
		self.lowering.schedule( var.type )
		self._cfg_assign( var, operand, is_alias = False, node = node )

	def _expr_Lambda( self, node: ast.Lambda, expected_type: Type|None ) -> ir.Operand:
		# a lambda expression, capturing or not - see PLAN_LAMBDA.md/the
		# closures plan. Lambda syntax carries no type annotations at all,
		# so parameter types are inferred entirely from expected_type -
		# either a Ptr[Callable[[ArgTypes],Ret]] shape (the non-capturing
		# case's own established route - e.g. a `key: Callable[[T],K]`
		# parameter's own declared type) or, now, a bare ClosureType (a
		# capturing lambda's own actual result type - `c: Closure[[Args],
		# Ret] = lambda ...: ...`, the natural way to write one). ClosureType
		# already exposes the identical arg_types/return_type shape
		# CallableType does (see its own docstring), so no wrapper object is
		# needed - just falling back to expected_type itself when it's
		# already the right shape. Deliberately NOT folded into
		# TypeResolver._callable_type_of itself - that function's other
		# callers (_try_lower_indirect_call in particular) mean specifically
		# "a bare function-pointer value", not "anything callable"
		fn_type = self.lowering._type_resolver._callable_type_of( expected_type )
		if fn_type is None and isinstance( expected_type, ClosureType ):
			fn_type = expected_type
		if fn_type is None:
			self.lowering.discovery.fail(
				f'cannot infer lambda parameter types - no expected Callable[...] context: {ast.unparse(node)}',
				node,
			)
		args = node.args
		if args.vararg is not None or args.kwarg is not None or args.kwonlyargs:
			self.lowering.discovery.fail( f'lambda does not support *args/**kwargs/keyword-only parameters yet: {ast.unparse(node)}', node )
		positional = [ *args.posonlyargs, *args.args ]
		if len( positional ) != len( fn_type.arg_types ):
			self.lowering.discovery.fail(
				f'lambda takes {len(positional)} argument(s), the expected Callable[...] type declares {len(fn_type.arg_types)}: {ast.unparse(node)}',
				node,
			)
		if any( isinstance( t, TypeVar ) for t in fn_type.arg_types ):
			# unlike the return type below, a lambda's own PARAMETER types
			# have no body to infer them from - they're needed up front just
			# to lower the body at all (see below), so an unbound arg type
			# here is unrecoverable, not just provisional
			self.lowering.discovery.fail(
				f'cannot infer lambda parameter types - Callable[...] parameter types are not fully concrete: {ast.unparse(node)}',
				node,
			)
		# a lambda passed to a generic function's own Callable[[T],K]-typed
		# parameter (e.g. bisect_right's key=) can have an unbound K here -
		# only knowable from the lambda's OWN body, once lowered (PLAN_LAMBDA.md,
		# "eager lambda lowering"/the circular-inference problem). return_type
		# stays a real Type|None field either way (Function.return_type is
		# already None for any not-yet-resolved function - same convention,
		# see FunctionLowering's own docstring), just resolved eagerly below
		# instead of by the ordinary ast-annotation route
		return_type_provisional = isinstance( fn_type.return_type, TypeVar )

		enclosing = self._current_fn
		if enclosing is None:
			self.lowering.discovery.fail( f'lambda outside any function: {ast.unparse(node)}', node )
		self.lowering._reject_generic_enclosing_scope( enclosing, node, 'lambdas' )

		self.lowering._lambda_counter += 1
		name = f'$$lambda_{self.lowering._lambda_counter}'
		qualname = f'{enclosing.qualname}{name}'
		synthetic_node = ast.FunctionDef(
			name = name,
			args = ast.arguments(
				posonlyargs = [], args = [ ast.arg( arg = p.arg, annotation = None ) for p in positional ],
				vararg = None, kwonlyargs = [], kw_defaults = [], kwarg = None, defaults = [],
			),
			body = [ ast.Return( value = node.body ) ],
			decorator_list = [], returns = None, type_params = [],
			lineno = node.lineno, col_offset = node.col_offset,
		)
		ast.fix_missing_locations( synthetic_node )

		synthetic = Function(
			stem = name, qualname = qualname,
			file = enclosing.file, line = node.lineno,
			cls = None, node = synthetic_node,
			parameters = None, return_type = None if return_type_provisional else fn_type.return_type,
			resolve = None,
		)
		parameters: list[Parameter] = []
		for arg_node, arg_type in zip( positional, fn_type.arg_types ):
			param = Parameter(
				stem = arg_node.arg, qualname = f'{qualname}.{arg_node.arg}',
				file = enclosing.file, line = node.lineno,
				type = arg_type,
			)
			parameters.append( param )
			synthetic.add_name( param.stem, param )
		synthetic.parameters = parameters

		captures = self._collect_free_variables( [ node.body ], { p.arg for p in positional }, node )

		env_cls: RCClass|None = None
		if captures:
			# a capturing lambda - build the env class and rewrite the body
			# BEFORE any eager lowering below, so a still-unbound return
			# type is inferred from the ALREADY-REWRITTEN (fully closed, no
			# free names left) body - see the closures plan's own note on
			# why this ordering is one-directional
			ptr_cls = self.lowering.discovery.get_intrinsics()['Ptr']
			none_type = self.lowering.discovery.get_none_type()
			ptr_none_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ none_type ] )

			env_cls = self.lowering._build_closure_env_class(
				[ ( cap_name, resolved.type ) for cap_name, resolved in captures ], f'{qualname}$$env', enclosing.file, node.lineno,
			)
			captured_names = { cap_name for cap_name, _ in captures }
			synthetic_node.body = self._rewrite_captures_into_env_reads( synthetic_node.body, env_cls, 'erased_env', captured_names )

			erased_env_param = Parameter( stem = 'erased_env', qualname = f'{qualname}.erased_env', file = enclosing.file, line = node.lineno, type = ptr_none_type )
			synthetic.parameters = [ erased_env_param, *parameters ]
			synthetic.add_name( 'erased_env', erased_env_param )

		if return_type_provisional:
			# lower the body RIGHT NOW, synchronously, instead of only ever
			# scheduling it onto the work queue for later - the caller's own
			# generic type-parameter binding (_unify_type_param's CallableType
			# recursion) needs the REAL return type immediately, at this call
			# site, not whenever the queue happens to drain it. Reuses
			# Compiler._lower exactly (resolve_function_body + lower_function
			# + extern_libs bookkeeping + registering into compiler.functions)
			# via the _compile_now backreference threaded in Compiler.__init__ -
			# lower_function itself builds a brand-new FunctionLowering
			# instance for this nested call, so there's no shared mutable
			# state with the lowering already in progress for `enclosing` to
			# save/restore around at all. Post-rewrite (if capturing), the
			# body is already fully closed (every capture is now an
			# ordinary attribute-chain expression rooted at erased_env) -
			# eager lowering genuinely cannot tell a capturing lambda from a
			# hand-written one at this point, so nothing here needs to change
			lowered = self.lowering._compile_now( synthetic )
			return_instr = next( instr for instr in lowered.instructions if isinstance( instr, ir.Return ))
			synthetic.return_type = (
				return_instr.value.type if return_instr.value is not None
				else self.lowering.discovery.get_none_type()
			)
		elif not captures:
			self.lowering.schedule( synthetic )

		if captures:
			# _construct_capturing_closure schedules `synthetic` itself
			# (see its own comment) - not done separately here
			return self._construct_capturing_closure( captures, env_cls, synthetic, node, expected_type )
		return self.lowering._function_ref_operand( synthetic )

	def _lower_method_call( self, receiver: ir.Operand, method_name: str, args: list[ir.Operand], result_type: Type, node: ast.AST ) -> ir.Operand:
		# shared _find_method + resolve/schedule + emit Call boilerplate -
		# every f-string dunder-dispatch/format-spec call site below uses
		# this same shape (receiver already lowered, method looked up by
		# plain name). str/int/f32/f64 are never generic, so this comment
		# used to end there - but Ptr[T]/ConstPtr[T]'s own __str__/__repr__
		# (lib/builtins/__ptr_arith.py) ARE bare generic Functions with an
		# unbound type param T (same registration shape as their __add__/
		# __sub__/comparison dunders - see _resolve_receiver_generic_dunder's
		# own docstring), so _find_method alone isn't enough here anymore:
		# without also resolving T from the receiver's own concrete pointee
		# type, `method` still carries the bare TypeVar, and emitter_c.py's
		# c_type crashes on it at prototype-emission time (confirmed via a
		# real repro building f'{some_ptr}'/some_ptr.__str__()) - same fix
		# _find_dunder_for_arg's own tail already applies for operator-
		# dispatched Ptr dunders, just needed here too for this SEPARATE,
		# plain-method-name dispatch path.
		method = self.lowering._resolve_receiver_generic_dunder(
			self.lowering._find_method( receiver.type, method_name ), receiver.type,
		)
		if method is None:
			type_name = receiver.type.qualname if receiver.type is not None else '?'
			self.lowering.discovery.fail( f'f-string requires {type_name}.{method_name}() to be available: {ast.unparse(node)}', node )
		self.lowering._ensure_resolved( method )
		self.lowering.schedule( method.return_type )
		for p in ( method.parameters or [] ):
			self.lowering.schedule( p.type )
		dest = self._new_temp( result_type )
		if method.cls is None:
			# a Scalar-registered method (`SomeScalar.method = some_free_
			# function` - discovery.py's visit_Assign, e.g. this file's own
			# float format-spec dispatch onto f64._sign_prefix/_fixed_digits,
			# lib/builtins/__float.py) is a genuine free Function, unlike a
			# real CStruct/RCClass method - discovery never strips a "self"
			# off its .parameters the way _make_function_resolver does for
			# an actual class body (there IS no class body here), so
			# emitter_c.py's _emit_call_args (which walks target.parameters
			# assuming it already excludes the receiver) would double-count
			# the receiver against the first declared parameter otherwise -
			# confirmed by a real KeyError crash while wiring this up.
			# ir.Call's own receiver field is for real bound-method calls
			# only; a free function just takes the receiver as an ordinary
			# leading positional argument instead.
			self._emit( ir.Call( dest = dest, target = method, receiver = None, args = [ receiver ] + args, kwargs = {} ))
		else:
			self._emit( ir.Call( dest = dest, target = method, receiver = receiver, args = args, kwargs = {} ))
		return dest

	def _lower_bound_method_closure( self, node: ast.Attribute, obj: ir.Operand, method: Function, expected_type: Type|None ) -> ir.Operand:
		# worker.run used as a VALUE (not called) - a bound-method
		# reference, PLAN_CALLABLE.md's own "closure in miniature" deferred
		# item. Builds a real closure value: {fn: Ptr[None] (the memoized
		# trampoline above), self: Ptr[None] (the receiver, increffed -
		# "creating a closure is by definition creating a new reference")}
		# - a compiler-synthesized RCClass (ClosureType), so every existing
		# RC mechanism (cfg.py's is_rc/rc_leaves/assign/move) applies
		# completely unchanged from here on, no special-casing needed
		# is_bound_method_closure tags this SPECIFIC node so
		# _is_aliasing_expr can tell it apart from an ordinary field read
		# whose declared type just happens to be ClosureType (e.g.
		# `self.data.v_Ok` for a Result[Closure[...],E]) - see
		# _is_aliasing_expr's own comment on why the operand's type alone
		# isn't enough
		node.is_bound_method_closure = True
		self.lowering._ensure_resolved( method )
		if method.broken:
			raise RedundantCompilationError() # already reported at the point method's own resolution failed - see Name.broken
		if method.parameters is None:
			self.lowering.discovery.fail( f'{method.qualname} could not be resolved (see earlier error): {ast.unparse(node)}', node )
		arg_types = [ p.type for p in method.parameters ]
		self.lowering.schedule( method.return_type )
		for t in arg_types:
			self.lowering.schedule( t )

		closure_type = self.lowering.discovery._get_or_create_closure_type( arg_types, method.return_type )
		self.lowering._ensure_resolved( closure_type )
		# same scheduling _lower_allocate_fields/_try_lower_construct_call
		# already do for every OTHER RCClass construction (sys.alloc[cls]/
		# sys.free/__del__ must be real, lowered compile units by the time
		# the emitter sees the ir.Allocate below) - closure_type is already
		# concrete, no Specialization involved, so target_cls == concrete_type
		self.lowering._schedule_rcclass_construction( closure_type, closure_type )
		trampoline = self.lowering._get_or_create_closure_trampoline( method, obj.type )

		ptr_cls = self.lowering.discovery.get_intrinsics()['Ptr']
		none_type = self.lowering.discovery.get_none_type()
		ptr_none_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ none_type ] )
		trampoline_callable_type = self.lowering.discovery._get_or_create_callable_type(
			[ p.type for p in ( trampoline.parameters or [] ) ], trampoline.return_type,
		)
		trampoline_ptr_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ trampoline_callable_type ] )

		fn_ref = ir.FunctionRef( type = trampoline_ptr_type, fn = trampoline )
		fn_erased = self._new_temp( ptr_none_type )
		self._emit( ir.CastWrap( dest = fn_erased, operand = fn_ref ))

		self_erased = self._new_temp( ptr_none_type )
		self._emit( ir.CastWrap( dest = self_erased, operand = obj ))

		self._emit( ir.Incref( value = obj )) # the closure is a new owner of the receiver

		dest = self._new_temp( expected_type or closure_type )
		self._emit( ir.Allocate( dest = dest, cls = closure_type, fields = { 'fn': fn_erased, 'self': self_erased } ))
		return dest

	def _construct_capturing_closure(
		self, captures: list[tuple[str,Variable]], env_cls: RCClass, synthetic: Function,
		node: ast.AST, expected_type: Type|None = None,
	) -> ir.Operand:
		''' builds a real, capturing Closure[[Args],Ret] value - the general
		case of _lower_bound_method_closure just above, generalized from one
		erased receiver field to N real captured fields collapsed behind one
		erased env pointer. Shared by _stmt_FunctionDef (a capturing nested
		def) and _expr_Lambda (a capturing lambda) - see the closures plan.
		`synthetic` IS the trampoline here (unlike the bound-method case's
		separate, shared, memoized-per-(method,owner) trampoline function) -
		a lambda/nested-def's own synthesized body is never called any other
		way than through its own closure's fn field, and never shared across
		construction sites, so there's no benefit to a second indirection
		layer; its own first parameter is already `erased_env: Ptr[None]`. '''
		arg_types = [ p.type for p in synthetic.parameters[1:] ] # skip erased_env
		self.lowering.schedule( synthetic.return_type )
		for t in arg_types:
			self.lowering.schedule( t )

		closure_type = self.lowering.discovery._get_or_create_closure_type( arg_types, synthetic.return_type )
		self.lowering._ensure_resolved( closure_type )
		self.lowering._schedule_rcclass_construction( closure_type, closure_type )
		self.lowering._schedule_rcclass_construction( env_cls, env_cls ) # the env is a real, constructed RCClass too - needs its own sys.alloc/__del__ scheduled, same as any other constructed class

		# each capture's CURRENT value, read in the ENCLOSING function's own
		# scope (an ordinary ast.Name read) - field_value() is the same
		# generic per-field embedding every ordinary SomeClass(field=value)
		# construction already uses (cfg.py, shared by _lower_allocate_fields):
		# an aliasing Name read gets exactly one Incref if its type is RC,
		# nothing otherwise - no hand-rolled ir.Incref needed here, unlike
		# the bound-method case above, precisely BECAUSE these fields keep
		# their real declared types instead of erasing to Ptr[None]
		fields: dict[str,ir.Operand] = {}
		for name, resolved in captures:
			name_node = ast.Name( id = name, ctx = ast.Load(), lineno = node.lineno, col_offset = node.col_offset )
			value = self._lower_expr( name_node, resolved.type )
			is_alias = self.lowering._is_aliasing_expr( name_node, value )
			for instr in self._cfg.field_value( value.type, value, is_alias = is_alias ):
				self._emit( instr )
			fields[name] = value

		env_dest = self._new_temp( env_cls )
		self._emit( ir.Allocate( dest = env_dest, cls = env_cls, fields = fields ))

		ptr_cls = self.lowering.discovery.get_intrinsics()['Ptr']
		none_type = self.lowering.discovery.get_none_type()
		ptr_none_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ none_type ] )

		env_erased = self._new_temp( ptr_none_type )
		self._emit( ir.CastWrap( dest = env_erased, operand = env_dest ))
		# env_dest is a fresh ir.Allocate result, so it's fresh_temp()-
		# tracked as a pending obligation for THIS statement's own cleanup
		# (see _emit) - erasing it via CastWrap doesn't transfer that
		# tracking (a CastWrap's own dest is never fresh_temp()-registered,
		# but its OPERAND's existing tracking is untouched), so without this
		# the per-statement pending-temp flush would decref env_dest right
		# out from under the closure that's about to become its only real
		# owner - a real, confirmed premature free (heap corruption at
		# runtime, not just reasoning). untrack_temp mirrors exactly what
		# cfg.field_value's own is_alias=False branch already does for any
		# other fresh value handed into a new field - ownership transfers
		# into the closure's own `self` field, so the original temp needs
		# no independent decref of its own
		self._cfg.untrack_temp( env_dest )

		trampoline_callable_type = self.lowering.discovery._get_or_create_callable_type(
			[ p.type for p in synthetic.parameters ], synthetic.return_type,
		)
		trampoline_ptr_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ trampoline_callable_type ] )
		fn_ref = ir.FunctionRef( type = trampoline_ptr_type, fn = synthetic )
		fn_erased = self._new_temp( ptr_none_type )
		self._emit( ir.CastWrap( dest = fn_erased, operand = fn_ref ))
		# unlike the bound-method trampoline (scheduled inside _get_or_create_
		# closure_trampoline, which BUILDS it), `synthetic` here is built by
		# the caller (_stmt_FunctionDef/_expr_Lambda) - this is the one place
		# both paths funnel through, so scheduling it here (not at either
		# call site) is what actually makes it a real, emitted function
		self.lowering.schedule( synthetic )

		# expected_type is preferred over closure_type itself ONLY when it's
		# actually the same closure type reached through a different
		# representation (e.g. a generic substitution's own fresh
		# Specialization vs this call's already-monomorphized one - the
		# exact duality _same_type exists for). Blindly trusting ANY
		# expected_type here (the previous behavior) let a genuine mismatch
		# (e.g. a capturing lambda passed where a non-capturing Ptr[Callable
		# [...]]-typed parameter is declared) silently bake the WRONG type
		# onto this Allocate's own dest - emitter_c.py then read that dest
		# type back off (not closure_type) to decide how to build the
		# object header/vtable, and crashed on its own internal
		# `assert isinstance(concrete_cls, RCClass)` instead of failing
		# cleanly. Falling back to closure_type here for any genuine
		# mismatch instead lets the ordinary post-hoc _check_assignable
		# machinery in _lower_expr/_coerce_or_check_operand (which every
		# _expr_Lambda caller already goes through) catch and report it the
		# same clean way the non-capturing (Ptr[Callable[...]]) path
		# already does
		dest_type = (
			expected_type if expected_type is not None and self.lowering._type_resolver._same_type( expected_type, closure_type )
			else closure_type
		)
		dest = self._new_temp( dest_type )
		self._emit( ir.Allocate( dest = dest, cls = closure_type, fields = { 'fn': fn_erased, 'self': env_erased } ))
		return dest

	def _try_lower_closure_call( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand|None:
		# my_closure(a, b) where my_closure: Closure[[A,B],R] - a call
		# THROUGH a closure VALUE (see _lower_bound_method_closure). Same
		# "try a shape, None means try the next one" convention as
		# _try_lower_indirect_call just above. Two callee shapes, same
		# double-evaluation-safety discipline _try_lower_indirect_call's own
		# Name/Attribute split documents: a bare Name (my_closure(...)) and
		# an Attribute (obj.field(...), a Closure-typed FIELD read off obj -
		# this also transparently covers a lambda/nested-def capturing a
		# Closure and calling it directly in its own body, since
		# _rewrite_captures_into_env_reads rewrites that capture into an
		# Attribute read off the synthesized env BEFORE lowering ever sees
		# it) both resolve their closure_type via a purely static lookup
		# before evaluating node.func for real, exactly like
		# _try_lower_indirect_call's own two shapes.
		if isinstance( node.func, ast.Name ):
			name = self.lowering.discovery.find_name_or_none( node.func.id )
			if not isinstance( name, Variable ):
				return None
			self.lowering._ensure_resolved( name )
			closure_type = self._narrowed_type_of_name( node.func.id, name.type )
			if not isinstance( closure_type, ClosureType ):
				return None
		elif isinstance( node.func, ast.Attribute ):
			receiver_type = self._static_type_of_value_expr( node.func.value )
			if receiver_type is None:
				return None
			if self.lowering._find_method( receiver_type, node.func.attr ) is not None:
				return None # a real method exists with this name - an ordinary method call, not a field call
			field = self.lowering._find_field( receiver_type, node.func.attr )
			if field is None:
				return None # no such field either - let _resolve_callee's own Attribute path give the accurate diagnostic
			closure_type = self.lowering._type_resolver._closure_type_of( field.type )
			if closure_type is None:
				return None
		else:
			return None
		self.lowering._ensure_resolved( closure_type )
		if any( isinstance( a, ast.Starred ) for a in node.args ):
			self.lowering.discovery.fail( f'*args not supported yet: {ast.unparse(node)}', node )
		if node.keywords:
			self.lowering.discovery.fail( f'a closure call takes no keyword arguments: {ast.unparse(node)}', node )
		if len( node.args ) != len( closure_type.arg_types ):
			self.lowering.discovery.fail(
				f'{ast.unparse(node.func)}(...) takes {len(closure_type.arg_types)} argument(s), got {len(node.args)}: {ast.unparse(node)}',
				node,
			)
		closure_operand = self._lower_expr( node.func, None )
		args = [ self._lower_expr( arg_node, arg_type ) for arg_node, arg_type in zip( node.args, closure_type.arg_types ) ]

		fn_field = closure_type.get_local( 'fn' )
		self_field = closure_type.get_local( 'self' )
		fn_operand = self._new_temp( fn_field.type )
		self._emit( ir.GetAttr( dest = fn_operand, obj = closure_operand, attr = 'fn' ))
		self_operand = self._new_temp( self_field.type )
		self._emit( ir.GetAttr( dest = self_operand, obj = closure_operand, attr = 'self' ))

		# fn is Ptr[None] (type-erased) on the closure struct itself - cast
		# back to the trampoline's real (Ptr[None], *ArgTypes) -> RetType
		# shape before calling through it, exactly mirroring how it was
		# erased going IN (_lower_bound_method_closure's own ir.CastWrap)
		ptr_cls = self.lowering.discovery.get_intrinsics()['Ptr']
		trampoline_callable_type = self.lowering.discovery._get_or_create_callable_type(
			[ fn_field.type, *closure_type.arg_types ], closure_type.return_type,
		)
		trampoline_ptr_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ trampoline_callable_type ] )
		fn_cast = self._new_temp( trampoline_ptr_type )
		self._emit( ir.CastWrap( dest = fn_cast, operand = fn_operand ))

		return self._emit_call_indirect( fn_cast, [ self_operand, *args ], closure_type.return_type, expected_type )
