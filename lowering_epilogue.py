# stdlib imports:
import ast
import copy

# local imports:
import cfg
import ir
from errors import CompileError
from mpy_types import (
	Name, Type, Variable, Function, ClassLike, Module, Specialization, TaggedUnion, CStruct, CUnion, CEnum, RCClass, Scalar, TupleType,
)

class EpilogueLoweringMixin:
	''' return/epilogue lowering, defer-block registration, and construction-completion helpers - mixed into FunctionLowering (lowering.py), which
	see for the shared instance state (self._instructions, self._cfg, self.lowering,
	etc.) every method here reads and writes. Never instantiated on its own;
	split out of lowering.py purely to keep that file to a manageable size - see
	lowering.py's own class docstring and FunctionLowering's base-class list for
	the full set of sibling mixins this one is composed with. '''


	def _emit_epilogue( self, fn: Function, none_type: Type, body_start: int ) -> None:
		# every return/OrJump/fall-off-the-end that has anything pending
		# (self._cfg.current_epilogue_label() was not None) funnels through
		# here exactly once, at the function's own closing brace -
		# build_epilogue_ladder() replays the WHOLE stack (RC decrefs and
		# defer/errdefer replays interleaved by declaration order, deepest/
		# most-recently-pushed first)
		#
		# flag inits have to run before *any* code that could set them -
		# easiest to guarantee by splicing them in right after FuncStart
		# rather than tracking every branch that could reach a defer statement
		# or a captured-then-cancelled epilogue entry (cancel_flags() - see
		# cfg.py's _neutralize()). A defer flag starts False (disarmed until
		# the defer statement itself runs); a cancel flag starts the other
		# way, True (still needs releasing until whichever of move()/
		# deleted()/manually_decreffed() actually neutralizes its entry runs)
		flag_inits = [
			ir.Assign( dest = flag, src = ir.Const( type = flag.type, value = False ))
			for flag in self._defer_flags
		] + [
			ir.Assign( dest = flag, src = ir.Const( type = flag.type, value = True ))
			for flag in self._cfg.cancel_flags()
		] + [
			# see self._for_obj_null_inits's own comment - default-initialize
			# so a goto that skips this loop's real assignment (an enclosing
			# loop that never runs) leaves a well-defined null, not garbage
			ir.Assign( dest = var, src = ir.Const( type = var.type, value = None ))
			for var in self._for_obj_null_inits
		]
		self._instructions[body_start:body_start] = flag_inits

		self._pending_temps = []
		for instr in self._cfg.build_epilogue_ladder( lambda: self._build_is_err_check( fn.node )):
			self._emit( instr )
		for t in reversed( self._pending_temps ):
			for instr in self._cfg.delete_temp( t ):
				self._emit( instr )
			self._emit( ir.DeleteTemp( temp = t ))
		return_value = self._return_value_var if fn.return_type is not none_type else None
		self._emit( ir.Return( value = return_value ))

	def _build_is_err_check( self, node: ast.AST ) -> tuple[list[ir.Instruction],ir.Temp]:
		''' the DeclareTemp+Call that checks self._return_value_var.is_err(),
		built as plain instructions rather than emitted directly - cfg.py's
		_replay() (via this callback) decides exactly where they land. With
		per-Epilogue labels, a single check computed once up front (the old
		design, back when there was only ever one shared epilogue label)
		wouldn't be reached by every jump that might need it - some land
		deeper in the ladder, skipping past it entirely (see
		build_epilogue_ladder()'s own comment) - so this is called fresh,
		deliberately uncached, every time an errdefer entry's own replay
		actually needs it. '''
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		is_err_fn = self.lowering._attr_lookup_callable( self._return_value_var.type, 'is_err', node )
		# resolve only (NOT _ensure_resolved, which also unconditionally
		# schedules is_err_fn as a compile unit) - is_err's genericity is
		# inherited from Result's own class type params (same as Result.
		# Ok/.Err/.is_ok - see _lower_class_generic_method_call's own
		# identical "receiver already concrete" branch), so the ABSTRACT
		# is_err_fn is never itself the right thing to schedule/call: its
		# synthesized `self` parameter would be typed as bare Result, which
		# has no real C struct body anywhere (only concrete specializations
		# do) - a genuine "incomplete type" compile error confirmed via a
		# real errdefer+clang round trip once _ensure_resolved's own
		# incidental scheduling was scheduling BOTH the abstract AND the
		# correctly-monomorphized version side by side
		assert is_err_fn.resolve is None, f'internal compiler error - {is_err_fn.qualname} was not fully resolved by the type_resolver module'
		return_type = self._return_value_var.type
		if isinstance( return_type, Specialization ) and return_type.base is is_err_fn.cls:
			# the receiver's type (self._return_value_var, always a
			# concrete Result[T,E] specialization by the time this runs)
			# already pins down the concrete args, so this is just an
			# ordinary monomorphization
			method_spec = self.lowering.discovery._get_or_create_specialization( is_err_fn, return_type.args )
			self.lowering.schedule( method_spec )
			is_err_fn = self.lowering._monomorphized_function( method_spec )
		else:
			self.lowering.schedule( is_err_fn )
		temp = ir.Temp( type = bool_cls, id = self._temp_id )
		self._temp_id += 1
		self._pending_temps.append( temp )
		return [
			ir.DeclareTemp( temp = temp ),
			ir.Call( dest = temp, target = is_err_fn, receiver = self._return_value_var, args = [], kwargs = {} ),
		], temp

	def _emit_construction_defaults( self, cls: RCClass, self_param: Variable, module: Module ) -> None:
		''' every defaulted attribute (attr.init is not None) gets an
		unconditional prologue assignment before __init__'s own
		user-written body runs - a later `self.a = ...` in the body (if
		any) then becomes an ordinary attr_assign() replace, decref-ing
		the just-created default. Lowered in the CLASS's own scope, not
		__init__'s - a default expression can reference other class-level
		names, but self isn't in scope for it, matching ordinary Python
		class-body semantics. '''
		for attr in cls.attributes:
			if attr.init is None:
				continue
			with self.lowering.discovery.module_context( module ):
				with self.lowering.discovery.scope_context( cls ):
					default_value = self._lower_expr( attr.init, attr.type )
			for instr in self._cfg.attr_assign( attr, default_value, is_alias = self.lowering._is_aliasing_expr( attr.init, default_value )):
				self._emit( instr )
			self._emit( ir.SetAttr( obj = self_param, attr = attr.stem, value = default_value ))

	def _complete_construction_or_fail( self, fn: Function ) -> None:
		try:
			self._cfg.complete_construction( fn.qualname )
		except CompileError as e:
			self.lowering.discovery.fail_loc( str( e ), fn.file, fn.line )

	# --- super().__init__(...) constructor chaining (RCClass, single inheritance) --

	def _super_init_shape( self, node: ast.expr ) -> tuple[ast.Call,ast.Call|None] | None:
		''' recognizes `super().__init__(...)` (base __init__ infallible) or
		`super().__init__(...).or_return()` (base __init__ fallible) as an
		EXACT textual shape - `super` is never a real registered name
		anywhere in this language (there is no builtin/intrinsic for it),
		so this has to be recognized here, before ordinary call resolution
		ever sees it, the same textual-recognition posture as defer/
		errdefer/compiler.X/or_return() itself already uses throughout this
		file. Returns (init_call, or_return_call) - or_return_call is None
		for the plain (infallible) spelling, otherwise the OUTER .or_return()
		ast.Call (handed to _lower_or_return unchanged, so ITS OWN existing
		checked-result propagation logic - OrReturn/OrJump - is reused
		verbatim rather than reimplemented here). Returns None for anything
		that isn't this exact shape - never a compile error by itself,
		callers decide what "not this shape" means in their own context
		(required-and-missing vs. used somewhere it isn't allowed at all). '''
		or_return_call: ast.Call|None = None
		call = node
		if (
			isinstance( call, ast.Call ) and isinstance( call.func, ast.Attribute ) and call.func.attr == 'or_return'
			and not call.args and not call.keywords
		):
			or_return_call = call
			call = call.func.value
		if not ( isinstance( call, ast.Call ) and isinstance( call.func, ast.Attribute ) and call.func.attr == '__init__' ):
			return None
		receiver = call.func.value
		if not (
			isinstance( receiver, ast.Call ) and isinstance( receiver.func, ast.Name ) and receiver.func.id == 'super'
			and not receiver.args and not receiver.keywords
		):
			return None
		return call, or_return_call

	def _lower_super_init_if_required( self, self_cls: RCClass, self_param: Variable ) -> list[ast.stmt]:
		''' called right before a subclass's own __init__ body is lowered
		(self._construction_self is already set) - if self_cls.base has a
		chained __init__ anywhere in ITS OWN chain (RCClass.chain_lookup,
		Phase 1), THIS __init__ must open with super().__init__(...) (or,
		when the base's own __init__ is fallible,
		super().__init__(...).or_return()) as literally its first statement
		- handled specially here rather than through the ordinary
		_stmt_Expr dispatch (see _super_init_shape's own comment on why).
		Returns the REMAINING statements for the ordinary per-statement
		loop to process - fn.node.body[1:] when this consumed statement 0,
		otherwise fn.node.body unchanged.

		A base with no chained __init__ at all needs no super() call - if
		it also has no fields anywhere in its own chain, there is nothing
		for this __init__ to be responsible for on the base's behalf at
		all (matches a root class's own construction exactly, just with a
		harmless base contributing nothing). If it DOES have fields but no
		__init__ to chain to, declaring a subclass __init__ at all is
		rejected outright - a deliberate Phase 2 scope limit (see the
		RCClass-subclassing plan's own Phase 2 notes): no sugar exists yet
		for filling in a field-only ancestor's fields from inside a
		subclass's own __init__, and inventing one isn't this phase's job. '''
		fn = self._current_fn
		if self_cls.base is None:
			return fn.node.body
		base_init = self_cls.base.chain_lookup( '__init__' )
		if base_init is None:
			if self_cls.base.flattened_attributes():
				self.lowering.discovery.fail(
					f'{fn.qualname}: cannot declare __init__ - base {self_cls.base.qualname} has field(s) but no '
					f'__init__ to chain to via super().__init__() (not supported yet)',
					fn.node,
				)
			return fn.node.body
		# resolve AND schedule - base_init might otherwise never become a
		# real compiled unit if nothing else ever calls it directly (an
		# ordinary call's own target already goes through _ensure_resolved/
		# schedule() somewhere upstream; this call site is entirely our own,
		# so it has to do that itself)
		base_init = self.lowering._ensure_resolved( base_init )
		shape = self._super_init_shape( fn.node.body[0].value ) if fn.node.body and isinstance( fn.node.body[0], ast.Expr ) else None
		if shape is None:
			self.lowering.discovery.fail(
				f'{fn.qualname}: must call super().__init__(...) as its first statement '
				f'({self_cls.base.qualname} has its own __init__ to chain to)',
				fn.node.body[0] if fn.node.body else fn.node,
			)
		init_call, or_return_call = shape
		is_fallible = self.lowering._init_fallibility( base_init )
		if is_fallible and or_return_call is None:
			self.lowering.discovery.fail(
				f'super().__init__(...): {self_cls.base.qualname}.__init__ is fallible - must be consumed via '
				f'.or_return(): {ast.unparse(fn.node.body[0])}',
				fn.node.body[0],
			)
		if not is_fallible and or_return_call is not None:
			self.lowering.discovery.fail(
				f'super().__init__(...).or_return(): {self_cls.base.qualname}.__init__ is not fallible - remove '
				f'.or_return(): {ast.unparse(fn.node.body[0])}',
				fn.node.body[0],
			)
		self.lowering.schedule( base_init.return_type )
		for param in base_init.parameters or []:
			self.lowering.schedule( param.type )
		# see the .or_return() recognizer's own identical snapshot/comment
		# (_expr_Call) - args/kwargs below can themselves retain a Part B
		# field receiver (e.g. `super().__init__(self.some_field)`) that's
		# fully consumed by the call and must be released regardless of
		# which leg fires, same reasoning as any other or_return() receiver
		receiver_pending_start = len( self._pending_temps )
		args, kwargs = self._lower_call_args( base_init, init_call )
		dest = self._new_temp( base_init.return_type ) if is_fallible else None
		call = ir.Call( dest = dest, target = base_init, receiver = self_param, args = args, kwargs = kwargs, is_super_init_call = True )
		self._emit( call )
		if or_return_call is not None:
			assert dest is not None
			self._lower_or_return( or_return_call, dest, want_result = False, receiver_pending_start = receiver_pending_start )
		self._cfg.complete_base_construction( self_cls.base.flattened_attributes() )
		return fn.node.body[1:]

	# --- super().<method>(...) chaining, any method other than __init__ ------

	def _super_call_shape( self, node: ast.expr ) -> str | None:
		''' recognizes `super().<name>(...)` as an EXACT textual shape, same
		posture as _super_init_shape (super is never a real registered name).
		__init__ is deliberately excluded here - it stays statement-position-
		only, handled exclusively by _lower_super_init_if_required, so this
		never lets `super().__init__(...)` bypass that "must be the literal
		first statement" requirement from some other position in the body.
		Returns the method name, or None if this isn't the shape at all. '''
		if not ( isinstance( node, ast.Call ) and isinstance( node.func, ast.Attribute )):
			return None
		if node.func.attr == '__init__':
			return None
		receiver = node.func.value
		if not (
			isinstance( receiver, ast.Call ) and isinstance( receiver.func, ast.Name ) and receiver.func.id == 'super'
			and not receiver.args and not receiver.keywords
		):
			return None
		return node.func.attr

	def _lower_super_call( self, node: ast.Call, method_name: str, expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		''' `super().<method_name>(...)` for any method other than __init__ -
		including __del__ (there is no dispatch-mechanism reason this can't
		work the same way ordinary methods do: __del__ itself is just an
		ordinary @virtual method, only the *synthesized* $$__destructor__
		vtable slot is special - see type_resolver.py's
		_synthesize_rcclass_destructor). Resolves directly to self_cls.base's
		OWN implementation (chain_lookup starting at .base, skipping this
		class's own override - identical primitive _lower_super_init_if_
		required already uses for __init__) and forces a direct call in
		emitter_c.py (ir.Call.is_super_call) - ordinary vtable dispatch keys
		off target.is_virtual alone and would re-enter THIS method's own
		override through the receiver's real runtime vtable instead of
		reaching the base's version at all. '''
		fn = self._current_fn
		self_cls = fn.cls if fn is not None else None
		if not isinstance( self_cls, RCClass ):
			self.lowering.discovery.fail(
				f'super().{method_name}(...) can only be used inside an RCClass instance method: {ast.unparse(node)}', node,
			)
		if self_cls.base is None:
			self.lowering.discovery.fail(
				f'{self_cls.qualname} has no base class - nothing for super().{method_name}(...) to call', node,
			)
		target = self_cls.base.chain_lookup( method_name )
		if not isinstance( target, Function ):
			self.lowering.discovery.fail(
				f'{self_cls.base.qualname} has no method {method_name!r} to call via super()', node,
			)
		# target might otherwise never become a real compiled unit if
		# nothing else calls it directly - same reasoning _lower_super_
		# init_if_required's identical _ensure_resolved call already uses
		target = self.lowering._ensure_resolved( target )
		self.lowering.schedule( target.return_type )
		for param in target.parameters or []:
			self.lowering.schedule( param.type )
		self_node = ast.copy_location( ast.Name( id = 'self', ctx = ast.Load() ), node )
		self_operand = self._lower_expr( self_node, None, strict = False )
		args, kwargs = self._lower_call_args( target, node )
		dest = self._new_temp( target.return_type )
		self._emit( ir.Call( dest = dest, target = target, receiver = self_operand, args = args, kwargs = kwargs, is_super_call = True ))
		if not want_result:
			return None
		if expected_type is None:
			return dest
		return self._coerce_or_check_operand( dest, expected_type, node, strict = False )

	# --- temp/instruction bookkeeping ----------------------------------------

	def _emit_captured( self, instr: ir.Instruction ) -> None:
		# splices ONE instruction from a branch's own true_captured/false_
		# captured list (_stmt_If/_lower_binary_branch's own "lower this
		# branch into a SEPARATE instruction list first, decide true_extra/
		# false_extra via merge_if, THEN splice everything into the real
		# stream" technique) back into self._instructions - deliberately
		# NOT through self._emit() below: that instruction was ALREADY
		# _emit()'d once, when it was first captured (self._instructions
		# was redirected to the branch's own list at the time, but _emit()
		# itself ran, including its own fresh_temp() registration) - _emit()
		# has no way to tell "first time" from "being re-spliced", so
		# calling it a SECOND time here for the same ir.Call/Allocate
		# instruction RE-registers its own dest temp into cfg.py's
		# _temp_states, silently UNDOING whatever untracked it in between
		# (e.g. cfg.assign()'s own "ownership transferred into a named
		# binding, untrack the source temp" branch, if the branch's own
		# body assigned this Call's result into an EXISTING binding, like
		# `if flag: r = make_ok(b) else: r = make_err()` reassigning a
		# pre-declared `r`) - confirmed via a real reference leak (refcount
		# one too high after either branch of exactly that shape ran).
		# Every OTHER instruction kind is unaffected (self._emit()'s own
		# registration is gated on isinstance(instr, (Call, Allocate)), and
		# _check_self_escape_in is idempotent - re-running it on an
		# already-checked instruction is harmless, just redundant), so
		# this only needs to skip the ONE non-idempotent side effect,
		# not reimplement self._emit() from scratch.
		if self._current_fn is not None:
			self._check_self_escape_in( instr )
		self._instructions.append( instr )

	def _stmt_Return( self, node: ast.Return ) -> None:
		if self._in_deferred_body:
			# a defer/errdefer body's code runs later, replayed inline at the
			# epilogue (see _register_defer_block) - a `return` inside it
			# doesn't have a sensible meaning (it's not really executing at
			# this point in the function, and jumping to __epilogue__ from
			# CODE ALREADY INSIDE the epilogue replay is nonsensical). Same
			# check _register_defer_block already applies to nested defer/
			# errdefer, catches nested cases too (return inside an if/while
			# inside the defer body) since _in_deferred_body stays set for
			# the whole capture, not just the top-level statement
			self.lowering.discovery.fail( f'return is not allowed inside a defer/errdefer body: {ast.unparse(node)}', node )
		# strict=False: this method already has its OWN, more complete
		# compatibility check just below (monomorphized-Specialization
		# comparison, CEnum-to-underlying, and _maybe_widen_return_result's
		# error-union widening) - _lower_expr's own general _check_assignable
		# would otherwise fire first and incorrectly reject exactly the
		# widening case this method exists to allow (`return x` where x:
		# Result[T,NarrowE] inside a function declared -> Result[T,WideE]).
		# The pre-existing TaggedUnion/RCClass-upcast coercions this method's
		# own comment below already expects still apply regardless (not
		# gated on strict - see _lower_expr's own comment)
		value = self._lower_expr( node.value, self._current_fn.return_type, strict = False ) if node.value is not None else None
		# an ALIASING return expression (self._is_aliasing_expr - a plain
		# Name/Attribute read, or a tuple-element Subscript) that does NOT
		# correspond to a live, skippable epilogue entry (self._cfg.
		# has_live_entry) needs its own Incref right here, before it's
		# handed off below: `return self.x` (an attribute read) and
		# `return self`/`return some_borrowed_param` (a BORROWED Name,
		# never pushed onto the epilogue stack - see cfg.py's
		# _enter_parameter()) both alias a reference that SOMEONE ELSE
		# still independently owns and will decref on their own schedule,
		# so the caller needs a genuinely separate +1, not a bare pointer
		# copy. An OWNED local (or a copy[T]/move[T] parameter) DOES
		# have a live entry - that's a real move (its own decref is what
		# current_epilogue_label()/return_() skip below, by this same
		# identity), and must NOT also get an Incref here, or the moved-
		# out reference would be permanently over-counted by one.
		# Confirmed by direct compile-and-run testing with compiler.
		# refcount(): `Holder.get(self) -> Box: return self.x` previously
		# hung onto only 2 references (the field + the caller's own new
		# holder of the returned value, double-counted as the SAME
		# reference) where 3 are live once the caller's copy exists,
		# leading to a premature free the moment either one dropped.
		self._incref_aliasing_return( node.value, value )
		# what actually gets returned/assigned into the return-value slot
		# below - defaults to `value` itself, reassigned to a widened temp
		# further down when the covered-Result-error-widening case applies.
		# `value` itself stays UNCHANGED throughout this whole method after
		# this point - every other use of it (check_unchecked_results,
		# current_epilogue_label, return_(), untrack_temp) needs the ORIGINAL
		# operand's own identity, not the widened temp's, to correctly
		# recognize "this tracked binding's own epilogue entry is the one
		# being moved out here" (see ir.WidenResult's own docstring)
		return_value = value
		# self._current_fn.return_type is Python None ONLY as a deliberate
		# sentinel (return-only generic type-param inference and lambda
		# eager-lowering both temporarily set it to None specifically so
		# _lower_expr(node.value, None) lets the return expression take its
		# OWN natural type, unconstrained, which then gets read back as the
		# inferred return type - see _infer_return_only_type_params/
		# _expr_Lambda's own "return_type_provisional" comments). A GENUINE
		# `-> None`-declared (or unannotated) function's own return_type is
		# always the real NoneType Scalar object (discovery.py's own
		# get_none_type()), never Python None - so this check only ever
		# skips during that provisional, not-yet-resolved state, never for
		# an actual declared return type
		if value is not None and self._current_fn.return_type is not None:
			# _lower_expr already applies every coercion it legitimately can
			# (union-leaf-wrap via _coerce_into_union, RCClass base-upcast via
			# _is_rcclass_upcast) - if value.type STILL doesn't match the
			# function's own declared return type afterward, this is a
			# genuine, uncaught mismatch that would otherwise emit a `return`
			# of the wrong C type (confirmed: `return 5` inside a `-> str`
			# function, or `return x` where x: Result[T,NarrowE] inside a
			# function declared -> Result[T,WideE] even though NarrowE is
			# covered by WideE, both previously compiled with zero errors and
			# produced C a real compiler rejects outright). The covered-but-
			# narrower-Result case specifically is a real, intentional gap for
			# now - it's handled by widening instead of rejection, added
			# separately (see the auto-widening this same function grows next
			# to this check).
			fn_type = self._current_fn.return_type
			# a generic ClassLike return type (Result[T,E], list[T], any
			# user generic class/@union) stays the ORIGINAL, un-monomorphized
			# Specialization on fn.return_type forever (it's parsed once,
			# from the function's own annotation, never re-resolved) - but a
			# local Variable's own .type (e.g. `result: list[str] = ...`)
			# DOES get monomorphized to the real, concrete ClassLike at some
			# point before this runs (confirmed: `result.type` here is
			# already a concrete RCClass, not the Specialization wrapping
			# list's abstract template) - so value.type and a bare fn_type
			# can legitimately be the "same type" while being different
			# objects. Monomorphize fn_type the same way before comparing -
			# mirrors _lower_expr's own identical "monomorphize a generic
			# union expected_type before comparing" step, generalized here
			# to every ClassLike kind (not just TaggedUnion), since this
			# same divergence isn't union-specific. Ptr[T]/ConstPtr[T]
			# (Scalar-based generics, not ClassLike) are deliberately
			# excluded - those are already consistently interned via plain
			# Specialization identity, confirmed by direct inspection, and
			# monomorphize_class doesn't apply to them anyway
			expected_concrete = fn_type
			if isinstance( fn_type, Specialization ) and isinstance( fn_type.base, ( RCClass, CStruct, CUnion, TaggedUnion, CEnum )):
				expected_concrete = self.lowering.monomorphize_class( fn_type )
			elif isinstance( fn_type, TupleType ) and fn_type.backing is not None:
				# same divergence, different shape: a tuple LITERAL (`return
				# (a, b, c)`) lowers to its own synthesized backing RCClass
				# directly (tuple_storage.TupleStorage.get()'s own real,
				# constructible representation), not the bare TupleType
				# wrapper the function's own `-> tuple[str,str,str]`
				# annotation stays as
				expected_concrete = fn_type.backing
			# a CEnum value flowing into a context expecting its OWN
			# underlying scalar type is also legitimate - "a CEnum has
			# exactly the same runtime representation as its underlying
			# type" (see the CEnum construction-call comment above), so
			# returning one where the underlying type is declared is a
			# value-preserving reinterpretation, not a mismatch. Both
			# directions, mirroring _check_assignable's own bidirectional
			# CEnum<->value_type exemption (lines ~4169/4171) - this method
			# can't just delegate to _check_assignable itself (see this
			# method's own strict=False comment above, on why that would
			# incorrectly reject the covered-Result-error widening case
			# before it's even attempted), so it has to re-derive every
			# exemption _check_assignable would apply; PLAN_COMPILER_BUG_
			# SWEEP.md's own audit found this had only ever re-derived ONE
			# of the two directions - `return raw_scalar` from a function
			# declared `-> SomeCEnum` (the OTHER direction) was wrongly
			# rejected, confirmed via a real repro
			is_cenum_to_underlying = isinstance( value.type, CEnum ) and value.type.value_type is fn_type
			is_underlying_to_cenum = isinstance( fn_type, CEnum ) and value.type is fn_type.value_type
			# the mirror image of expected_concrete above: value.type can
			# ALSO still be a raw, un-monomorphized Specialization here (a
			# compiler.checked_add(...)-style intrinsic's own check_dest,
			# see _lower_compiler_checked_binop's own comment on why IT
			# can't be pre-monomorphized either - the emitter needs its
			# Specialization .args) - monomorphize it the same way before
			# the identity comparison, rather than requiring every producer
			# of a same-statement-escaping Result value to guess which form
			# the compare side wants
			value_concrete = value.type
			if isinstance( value.type, Specialization ) and isinstance( value.type.base, ( RCClass, CStruct, CUnion, TaggedUnion, CEnum )):
				value_concrete = self.lowering.monomorphize_class( value.type )
			if (
				value.type is not fn_type and value.type is not expected_concrete
				and value_concrete is not fn_type and value_concrete is not expected_concrete
				and not is_cenum_to_underlying and not is_underlying_to_cenum
			):
				widened = self._maybe_widen_return_result( node, value, fn_type )
				if widened is None:
					# case 2 of the general auto-or_throw() rule (see _auto_or_
					# throw's own docstring): `return a + b` inside a function
					# declared to return plain T (not Result[T,E]) - this
					# method lowers `value` with strict=False (see this
					# method's own comment above), so it never reaches
					# _coerce_or_check_operand's own identical hook; reuses
					# the exact same shared probe instead of a second copy of
					# the guard
					consumed = self._maybe_auto_consume_result( node, value, fn_type, self.lowering._AUTO_CONSUME_ALTERNATIVES )
					if consumed is None:
						self.lowering.discovery.fail(
							f'{ast.unparse(node)}: function returns '
							f'{fn_type.qualname if fn_type else "None"}, not {value.type.qualname if value.type else "?"}',
							node,
						)
					widened = consumed
				return_value = widened
		try:
			self._cfg.check_unchecked_results( value )
		except CompileError as e:
			self.lowering.discovery.fail( str( e ), node )
		# a fallible __init__'s own Err-path return, before construction
		# completes - forces the INLINE return_() unwind below rather than
		# ever letting current_epilogue_label() hand out one of the
		# function's shared closing-brace labels. That shared ladder is
		# built ONCE, using self._epilogue_stack's FINAL cancelled-state as
		# of the function's own closing brace - a LATER return in this same
		# __init__ that reaches complete_construction() (this construction's
		# eventual success path, cancelling every attribute entry so
		# ownership transfers cleanly into the now-complete self) would
		# retroactively wipe out the very decref this EARLIER Err return's
		# already-committed jump depends on, since Epilogue.cancelled is one
		# mutable flag shared by every jump into that entry's label, not a
		# per-jump-site snapshot. Confirmed by a real repro: an RC attribute
		# assigned before a later-failing validation, on the Err path,
		# silently stopped being released the moment a later Ok-path return
		# in the same __init__ completed construction - masked as a leak
		# (not a crash) only because the CALL SITE's own release of self
		# used to fall back to the generic per-class destructor's
		# unconditional field cascade, independently releasing the same
		# attribute again; that fallback is gone now (see
		# _lower_compiler_raw_free's own comment (used by the synthesized $$__new__'s Err branch) on why it had to be
		# removed - it also unconditionally touched attributes that were
		# NEVER assigned at all, reading uninitialized memory), so this
		# construction's own inline unwind is now the ONLY place whichever
		# attributes it assigned ever get released on this path
		construction_err_path = False
		if self._construction_self is not None:
			# every return in a non-fallible __init__ is unconditionally
			# success (construction_fallible is False, so the `and` below
			# short-circuits) - no legal way to signal failure. In a
			# fallible one, only a return whose value is textually
			# Result.Err(...) is the failure path (partial init expected/
			# legal there, cleaned up normally by the ordinary return_()
			# unwind below - "clean up any that were initialized"); every
			# other shape (Result.Ok(...), or anything else - deliberately
			# not attempting deeper type-level inference here) requires
			# full initialization
			is_success = not ( self._construction_fallible and self.lowering._is_result_err_call( node.value ) is not None )
			if is_success:
				self._complete_construction_or_fail( self._current_fn )
			else:
				construction_err_path = True
		# a construction_err_path return still shares an ordinary label
		# with defer/errdefer/plain-local entries - only self.<attr>
		# entries (construction_err_inline) are forced inline, since only
		# THOSE are at risk of complete_construction()'s retroactive
		# cancellation - see current_epilogue_label_for_construction_err()'s
		# own docstring
		construction_err_inline: list[ir.Instruction] = []
		if construction_err_path:
			label, construction_err_inline = self._cfg.current_epilogue_label_for_construction_err( value )
		else:
			label = self._cfg.current_epilogue_label( value )
		# the innermost active multi-statement @inline splice, if this
		# return is reached from one of its own pre-return statements (see
		# _splice_multi_statement_inline_body/self._inline_scope_vars' own
		# comment) - value-computation/widening above is already correct
		# unchanged (self._current_fn.return_type is provisional's, i.e.
		# the INLINED function's own declared type), only the TERMINAL
		# emission below needs to redirect: into the scope's own result_var
		# instead of self._return_value_var, arming its exited_flag, and
		# (inline-unwind branch only) jumping to the scope's own merge_label
		# instead of emitting a real ir.Return - this early return must
		# never become the CALLER's own return
		inline_scope = self._inline_scope_vars[-1] if self._in_inline_splice_prelude and self._inline_scope_vars else None
		if label is not None:
			# whatever's still pending (RC decrefs, defer/errdefer replays)
			# gets unwound once, later, by the shared ladder every other
			# return reaching this same label also jumps into
			# (build_epilogue_ladder(), emitted at the function's own
			# closing brace - see _emit_epilogue; or, inside a splice, the
			# scope's own local ladder - see _splice_multi_statement_
			# inline_body) - value has to survive the jump some other way
			# than a direct ir.Return
			if inline_scope is not None:
				result_var, exited_flag, _merge_label = inline_scope
				if result_var is not None and value is not None:
					self._emit( ir.Assign( dest = result_var, src = return_value ))
				self._emit( ir.Assign( dest = exited_flag, src = ir.Const( type = exited_flag.type, value = True )))
			elif self._return_value_var is not None and value is not None:
				self._emit( ir.Assign( dest = self._return_value_var, src = return_value ))
			# value's own ownership (if it's a bare temp - `return
			# SomeConstructor(...)`, never assigned to a name) just
			# transferred into self._return_value_var above via the plain
			# ir.Assign - untrack it so _flush_pending_temps below doesn't
			# ALSO decref it (return_()'s own docstring explains the
			# identical concern for the other branch)
			self._cfg.untrack_temp( value )
			# construction_err_inline's own self.<attr> decrefs (if any) -
			# always safe to run before the jump, never after: `value`
			# itself can never alias one of them (current_epilogue_label_
			# for_construction_err() already bails to a plain return_()
			# fallback whenever `value` aliases ANY live stack entry)
			for instr in construction_err_inline:
				self._emit( instr )
			# flushed HERE, before this branch's own unconditional
			# ir.Jump - not left to _lower_stmt's own post-method flush,
			# which runs strictly after this whole method returns and so
			# would land as dead code following the Jump (see
			# _flush_pending_temps' own docstring for the general shape of
			# this bug: a single-statement function body like `def make()
			# -> Result[str,E]: return Result.Ok('hello'.upper())` used to
			# leave the intermediate str temp's own release permanently
			# unreachable, inflating the returned Result's refcount by one
			# forever)
			self._flush_pending_temps()
			self._emit( ir.Jump( target = label ))
		else:
			# either nothing is pending, or `value` IS itself one of the
			# still-live entries current_epilogue_label() can't route
			# through a shared label (see its own comment) - unwind inline,
			# right here, same as always (bounded to the splice's own
			# portion of the stack when inline_scope is set - see cfg.py's
			# return_() own comment). Still has to replay any pending
			# defer/errdefer entries itself (return_() does this now too -
			# they're just as "pending" as an RC decref from here)
			#
			# self._return_value_var populated here too, ONLY when a live
			# errdefer entry could actually need it: _build_is_err_check's
			# own is_err() probe (invoked lazily, only if return_() below
			# finds one to replay) unconditionally reads self._return_
			# value_var as ITS receiver, with no other way to learn what
			# THIS return's own value even is - previously only assigned on
			# the shared-label branch above. Confirmed missing via a real
			# repro: an errdefer entry sitting beneath a CONFINED entry (an
			# ordinary RC-typed local declared earlier in the same still-
			# open branch, forcing this inline path instead of the shared-
			# label one) silently checked a stale/uninitialized self.
			# _return_value_var instead of this return's real Err value -
			# the errdefer fired 0 times instead of once, no error, no
			# crash, just silently skipped cleanup. Gated on an actual live
			# errdefer entry (not unconditional like the label branch above)
			# - an earlier, unconditional version of this fix broke many
			# unrelated functions ("variable has incomplete type void") by
			# forcing self._return_value_var's own declaration/reference
			# into functions that never otherwise touch it at all.
			if self._return_value_var is not None and value is not None and any(
				e.is_err_only and not e.cancelled for e in self._cfg._epilogue_stack
			):
				self._emit( ir.Assign( dest = self._return_value_var, src = return_value ))
			for instr in self._cfg.return_( value, lambda: self._build_is_err_check( node )):
				self._emit( instr )
			# same reasoning as the label-is-not-None branch above - flush
			# BEFORE this branch's own unconditional terminator, not after
			# (return_() already untracked `value` itself, so this only
			# ever cleans up OTHER still-pending temps - e.g. an
			# intermediate argument consumed into constructing `value`)
			self._flush_pending_temps()
			if inline_scope is not None:
				result_var, exited_flag, merge_label = inline_scope
				if result_var is not None and value is not None:
					self._emit( ir.Assign( dest = result_var, src = return_value ))
				self._emit( ir.Assign( dest = exited_flag, src = ir.Const( type = exited_flag.type, value = True )))
				# jumps PAST the scope's own ladder (already replayed
				# inline, right above - re-entering it via its own label
				# would replay the same entries a second time) straight to
				# where the early-exit-vs-normal-fallthrough merge begins
				self._cfg.mark_inline_scope_captured()
				self._emit( ir.Jump( target = merge_label ))
			else:
				self._emit( ir.Return( value = return_value ))

	def _maybe_widen_return_result( self, node: ast.Return, value: ir.Operand, fn_type: Type ) -> ir.Temp|None:
		''' `return x` where x is Result[T,NarrowE] and this function is
		declared -> Result[T,WideE] - if WideE genuinely COVERS NarrowE (every
		leaf of NarrowE is also a leaf of WideE - the identical leaves-
		containment rule type_resolver._require_result_return already applies
		for .or_return()/checked-arithmetic propagation), emit ir.WidenResult
		and return its dest temp for the caller to actually return/assign
		instead of `value`. Returns None (no widening applies) for every other
		shape of mismatch - the caller's own existing error message covers
		those. '''
		tr = self.lowering._type_resolver
		op_shape = tr._result_shape( value.type )
		fn_shape = tr._result_shape( fn_type )
		if op_shape is None or fn_shape is None:
			return None
		op_t, op_e = op_shape
		fn_t, fn_e = fn_shape
		if op_t is not fn_t or op_e is fn_e:
			return None # different Ok type entirely, or errors already match (not this method's concern)
		fn_e_leaves = tr._atomic_leaves( fn_e )
		if not all( leaf in fn_e_leaves for leaf in tr._atomic_leaves( op_e )):
			return None # op's error isn't covered by fn's - a genuine mismatch, not widenable
		# fn_type is guaranteed Result-shaped here (fn_shape matched), so
		# schedule/monomorphize it the same way _stmt_Return's own caller
		# already does for every OTHER ClassLike return type - dest.type
		# must be the real, concrete Result[T,WideE] the function actually
		# returns in C, not the abstract Specialization
		dest_type = self.lowering.monomorphize_class( fn_type ) if isinstance( fn_type, Specialization ) else fn_type
		self.lowering.schedule( dest_type )
		dest = self._new_temp( dest_type )
		self._emit( ir.WidenResult( dest = dest, src = value ))
		return dest

	def _register_defer_block( self, is_err_only: bool, body: list[ast.stmt], node: ast.AST, *, allow_inside_loop: bool = False ) -> cfg.Epilogue:
		''' allow_inside_loop is set only by _lower_with_context_manager,
		whose own loop-safety is verified by its caller via
		_body_may_break_or_continue_to_enclosing_loop before this runs - see
		that check's own comment for why a with-statement's internal defer registration
		doesn't share defer/errdefer's own "single armed slot" hazard in
		the case it actually uses this override. Returns the pushed
		Epilogue entry - a caller whose own body always falls through to a
		single, always-executed exit point (with/for/try-finally) should
		hand it to cfg.py's disarm_defer() there instead of unconditionally
		emitting a runtime `flag = false` reset. '''
		kind = 'errdefer' if is_err_only else 'defer'
		if self._loop_depth > 0 and not allow_inside_loop:
			self.lowering.discovery.fail( f'{kind} is not allowed inside a loop - call another function and {kind} inside that instead', node )
		if self._in_deferred_body:
			self.lowering.discovery.fail( f'{kind} cannot be nested inside another defer/errdefer', node )

		fn = self._current_fn
		if is_err_only:
			return_type = fn.return_type if fn is not None else None
			ok = fn is not None and self.lowering._type_resolver._result_shape( return_type ) is not None
			if not ok:
				where = f'{fn.qualname} returns {return_type.qualname if return_type else None}' if fn is not None else 'this is not inside a function'
				self.lowering.discovery.fail( f'errdefer requires the enclosing function to return Result[_,_] ({where})', node )

		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		index = len( self._defer_flags )
		flag = Variable(
			stem = f'__defer_flag_{index}',
			qualname = f'{fn.qualname}.__defer_flag_{index}',
			file = fn.file,
			line = getattr( node, 'lineno', None ),
			type = bool_cls,
		)

		# capture the body's instructions instead of emitting them inline -
		# they run later, in the epilogue, not at the with-statement's own
		# position. Lowering happens here, once, right now (not re-lowered at
		# replay time) so identifier resolution and dependency scheduling only
		# ever happen once, same as any other statement
		outer_instructions = self._instructions
		outer_in_deferred_body = self._in_deferred_body
		self._instructions = []
		self._in_deferred_body = True
		try:
			for stmt in body:
				try:
					self._lower_stmt( stmt )
				except CompileError:
					continue
			captured = self._instructions
		finally:
			self._instructions = outer_instructions
			self._in_deferred_body = outer_in_deferred_body

		self._defer_flags.append( flag )
		entry = self._cfg.push_defer( captured, flag, is_err_only )
		# this is what actually runs at the with-statement's/call's position -
		# marks the block "armed" so the epilogue knows to replay it
		self._emit( ir.Assign( dest = flag, src = ir.Const( type = bool_cls, value = True )))
		return entry
