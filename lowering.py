# stdlib imports:
import ast
from contextlib import nullcontext
from dataclasses import replace

# local imports:
import arithmetic_mode
import cfg
import ir
from discovery import Discovery
from errors import CompileError
from mpy_types import (
	Name, Type, Variable, Parameter, Function, Overload, ClassLike, Module,
	Specialization, TaggedUnion, CStruct, CUnion, CEnum, TypeVar, ConditionalDispatch, Move, RCClass, Scalar,
)
import overload_resolution
from type_resolution import TypeResolver
from union_storage import ReceiverDispatch as _ReceiverDispatch

# compile-error "here's what to do instead" text for a Check-mode opcode's
# checked_error (ir.py's BinOp/UnaryOp.checked_error) - keyed by error name
# rather than owned by ir.py itself, since these are lowering-level compiler
# messages, not IR shape
_ALTERNATIVES_BY_ERROR: dict[str,str] = {
	'OverflowError': (
		'wrap this in `with compiler.wrap_arithmetic:`, `with compiler.saturate_arithmetic:`, '
		'or `with compiler.panic_arithmetic(...):` instead'
	),
	# the primary, expected path for division is the same as any other
	# Check-mode op: the enclosing function returns Result[_,
	# ZeroDivisionError] and the Result propagates via OrReturn/OrJump - no
	# panic involved, and this is what happens even inside wrap_arithmetic/
	# saturate_arithmetic (there's no wrapped/saturated variant of division,
	# so those modes don't change division's checked-ness at all). This
	# message only fires when that requirement ISN'T met - panic_arithmetic
	# is the one remaining alternative to changing the return type, not a
	# default
	'ZeroDivisionError': 'wrap this in `with compiler.panic_arithmetic(...):` instead',
}


class Lowering:
	'''
	turns one Function body (or one global Variable's initializer) at a time
	into an ir.py instruction list. Reuses Discovery's scope-chain machinery
	(find_name/module_context/scope_context) for identifier resolution -
	function bodies were deliberately left unvisited in stage 1 specifically
	so this could be reused here; what's new here is only the value-expression
	semantics stage 1 never needed (constant values, operator-to-opcode
	mapping, temp allocation, instruction emission).

	Whenever anything that might be a dependency is discovered - a Function, a
	class, a type (possibly a Specialization like Result[i32,E]), a Variable,
	even a Module reached mid-namespace-lookup - `schedule` is called on it
	immediately at the point of discovery, unconditionally; there's no
	separate dependency-scanning pass, and no filtering here either.
	`schedule` (Compiler._enqueue) is the single place that judges what's
	actually a compile unit worth queuing, what decomposes into more of
	those, and what to just quietly ignore - see its own docstring.

	Errors report through self.discovery.errors, the same collector stage 1
	uses (self.discovery.fail()/fail_loc()) - see _lower_stmt's caller in
	lower_function for the recovery boundary (one bad statement doesn't stop
	the rest of that function's body from being lowered).

	Arithmetic (+/-/*) defaults to Check mode (AddCheck/SubCheck/MulCheck,
	producing Result[T,OverflowError]) everywhere - there is no unchecked
	default. A Check op is immediately followed by an OrReturn (like
	Result.or_return()'s own semantics: propagate the error, continue with
	the unwrapped value), which requires the enclosing function to actually
	return Result[_,OverflowError] - using plain arithmetic in a function
	that can't propagate that error is a compile error, unless one of the
	arithmetic-mode with-blocks below is used instead. self._arithmetic_mode
	is a stack of ArithmeticMode objects, pushed/popped by _stmt_With:
		ArithmeticWrap       - `with compiler.wrap_arithmetic:` - plain
		                       AddWrap/SubWrap/MulWrap, no Result involved
		ArithmeticSaturate   - `with compiler.saturate_arithmetic:` - plain
		                       AddSaturate/SubSaturate/MulSaturate, likewise
		ArithmeticChecked    - the default (see above) - Check + OrReturn
		ArithmeticPanic      - `with compiler.panic_arithmetic(errmsg):` -
		                       still Check-mode ops, but consumed with
		                       Unwrap(errmsg) instead of OrReturn, so (unlike
		                       the bare default) this does NOT require the
		                       enclosing function to return Result[_,
		                       OverflowError] - Unwrap panics, it never
		                       needs anywhere to propagate to

	defer/errdefer (SYNTAX.md section 3, either `defer(expr)`/`errdefer(expr)`
	as a single statement or `with defer:`/`with errdefer:` for several) move
	their body to the function's shared epilogue - each registration
	(_register_defer_block) pushes a REAL cfg.Epilogue entry (cfg.py's
	push_defer(), flag set) onto the exact same _epilogue_stack RC bindings
	use, interleaved by declaration order with whatever locals surround it.
	Every `return`/checked-arithmetic-error-path (self._cfg.current_epilogue_
	label()) and the function's own fall-off-the-end funnel through
	build_epilogue_ladder(), which replays the whole stack in reverse
	(deepest/most-recently-pushed first) - a flag-guarded entry's own bool
	flag (False until control passes its registration point) gates whether
	it actually replays; errdefer entries are additionally guarded by
	calling .is_err() on the function's own stowed return value (always a
	Result wherever errdefer is legal) - not a separate signal, so it also
	covers a plain `return Result.Err(x)`, not just the implicit OrJump
	path. defer/errdefer are rejected inside a loop (self._loop_depth) or
	nested inside each other (self._in_deferred_body) - see
	_register_defer_block. A defer/errdefer registered inside an if-branch
	must still be reachable from the function's own single shared epilogue
	regardless of which branch (if either) actually armed it - cfg.py's
	restore() special-cases flag-guarded entries to survive scope-exit
	truncation for exactly this reason (see its own comment).
	'''

	def __init__( self, discovery: Discovery, type_resolver: 'TypeResolver' ) -> None:
		self.discovery = discovery
		# type_resolver (type_resolution.py) owns the reachable-from-main
		# work queue and the shared UnionStorage/Monomorphizer instances -
		# both already depended on nothing but Discovery and a `schedule`
		# callback, so Lowering just borrows the SAME instances rather than
		# building its own (see TypeResolver's own docstring). schedule/
		# _union_storage/_monomorphizer keep their original names here since
		# they're referenced throughout this file - only construction moved
		self._type_resolver = type_resolver
		self.schedule = type_resolver.schedule
		self._union_storage = type_resolver.union_storage
		self._monomorphizer = type_resolver.monomorphizer

	def _init_lowering_state( self, fn: Function | None ) -> None:
		self._instructions: list[ir.Instruction] = []
		self._temp_id = 0
		self._label_id = 0
		self._pending_temps: list[ir.Temp] = []
		self._current_fn = fn
		self._arithmetic_mode: list[arithmetic_mode.ArithmeticMode] = [ arithmetic_mode.ArithmeticChecked() ]
		self._loop_depth = 0
		self._loop_labels: list[tuple[str,str]] = []
		self._in_deferred_body = False
		self._defer_flags: list[Variable] = []
		self._return_value_var = None

	def lower_function( self, fn: Function ) -> list[ir.Instruction]:
		module = self._find_module_for( fn )
		if fn.extern_lib is not None:
			# @extern(lib, symbol) - a foreign call signature declaration,
			# not a real body to lower (discovery.py already required a
			# stub body - see _is_stub_body). No CFG/epilogue/locals
			# machinery applies here at all - just the bare signature, for
			# a future emitter to declare rather than define. Compiler._lower
			# is what actually registers the library dependency (see its
			# extern_libs bookkeeping) - this only has to emit the shape
			self._instructions: list[ir.Instruction] = []
			self._current_fn = fn
			self._emit( ir.FuncStart( name = fn.qualname, params = fn.parameters or [], return_type = fn.return_type, extern_lib = fn.extern_lib, extern_symbol = fn.extern_symbol ))
			self._emit( ir.FuncEnd( name = fn.qualname ))
			return self._instructions
		self._init_lowering_state( fn )

		with self.discovery.module_context( module ):
			with ( self.discovery.scope_context( fn.cls ) if fn.cls is not None else nullcontext() ):
				with self.discovery.scope_context( fn ):
					# self is deliberately excluded from fn.parameters/fn.names
					# in discovery.py (_make_function_resolver's add_param) so
					# overload matching never has to think about it - but that
					# means it was never made resolvable at all. The method
					# body obviously needs it, so it's synthesized here,
					# lowering-only, the moment we start lowering a method body
					# RCCLASS ATTRIBUTE LIFETIME.md / the approved plan - scoped
					# to non-subclassed RCClasses only (fn.cls.base is None):
					# subclassing/super()/attribute visibility aren't real
					# features yet, independent of this
					self._construction_self: Variable | None = None
					self._construction_fallible = False
					if fn.cls is not None and not fn.is_static and not fn.is_classmethod:
						self_param = Parameter( stem = 'self', qualname = f'{fn.qualname}.self', file = fn.file, line = fn.line, type = fn.cls )
						fn.add_name( 'self', self_param )
						# fn.cls may be a Specialization for a monomorphized
						# generic-class __init__ (see Lowering._lower_generic_
						# construction_args) - unwrap to the real RCClass for
						# the isinstance/.base checks below and the field list
						# construction needs further down. NOTE: RCClass.base
						# means "parent class in an inheritance chain" while
						# Specialization.base means "the generic template" -
						# not the same thing, don't conflate them
						self_cls = self._ensure_resolved( fn.cls ) if isinstance( fn.cls, Specialization ) else fn.cls
						if fn.stem == '__init__' and isinstance( self_cls, RCClass ) and self_cls.base is None:
							self._construction_self = self_param
							self._construction_fallible = self._init_fallibility( fn )

					if '$payload_cls' in fn.names:
						# a synthesized union-member constructor (see
						# union_storage.py's _build_member_constructor).
						# $union_cls stays whatever UnionStorage.get() built
						# at synthesis time - always the ABSTRACT union,
						# which is exactly right: _lower_allocate_fields's
						# own existing substitution (fn_cls.base is
						# target_cls) already handles the OUTER
						# .__allocate__() call correctly once fn.cls is a
						# concrete Specialization, the same way a real
						# hand-written method's body (always textually
						# saying `Result.__allocate__`, never `Result[i32,
						# E].__allocate__`) already relies on. $payload_cls
						# is different: nothing substitutes a bare
						# construct-call's OWN target class, so it's
						# refreshed here to the CONCRETE, correctly-
						# substituted payload class (built by
						# monomorphize_class - see its own "fresh payload_cls
						# per specialization" comment) whenever fn.cls is a
						# Specialization - mirrors how self_cls above is
						# also computed fresh per lowering call rather than
						# baked in once
						if isinstance( fn.cls, Specialization ):
							concrete_union = self.monomorphize_class( fn.cls )
							payload_cls = concrete_union.names['data'].type
							fn.add_name( '$payload_cls', payload_cls )

					for param in fn.parameters or []:
						self.schedule( param.type )
					self.schedule( fn.return_type )

					none_type = self.discovery.get_none_type()
					noreturn_type = self.discovery.get_intrinsics()['NoReturn']
					# eagerly created whenever it COULD be needed (whether it
					# actually ends up referenced depends on whether any
					# return ever routes through current_epilogue_label()/
					# OrJump, only known once the body's actually lowered) -
					# harmless when unused: a synthetic Variable, never added
					# to fn.names, that simply never appears in any emitted
					# instruction if nothing ever needs it
					self._return_value_var = (
						Variable( stem = '__return_value', qualname = f'{fn.qualname}.__return_value', file = fn.file, line = fn.line, type = fn.return_type )
						if fn.return_type not in ( none_type, noreturn_type )
						else None
					)

					self._emit( ir.FuncStart( name = fn.qualname, params = fn.parameters or [], return_type = fn.return_type ))
					# constructed AFTER FuncStart - CFGState's own prologue
					# building (a copy[T] union parameter's tag-gated Incref)
					# can call new_temp, which immediately emits its own
					# DeclareTemp, so FuncStart must already be in the stream
					bool_cls = self.discovery.get_intrinsics()['bool']
					self._cfg = cfg.CFGState(
						fn,
						bool_type = bool_cls,
						new_temp = self._new_temp,
						new_label = self._new_label,
						union_storage = self._union_storage.get,
					)
					if self._construction_self is not None:
						for attr in self_cls.attributes:
							self._ensure_resolved( attr ) # each field's own .type is lazily resolved, separate from the class itself - same as _lower_allocate_fields's identical loop
						self._cfg.enter_construction( self_param, self_cls.attributes )
					elif fn.cls is not None and not fn.is_static and not fn.is_classmethod:
						self._cfg.enter_self( self_param, is_move = fn.is_move )
					for instr in self._cfg.prologue_instructions:
						self._emit( instr )
					if self._construction_self is not None:
						self._emit_construction_defaults( self_cls, self_param, module )
					body_start = len( self._instructions )
					for stmt in fn.node.body:
						# one bad statement doesn't stop the rest of this
						# function's body from being lowered (and error-collected) -
						# mirrors discovery.py's per-.resolve()/per-top-level-statement
						# recovery boundaries
						try:
							self._lower_stmt( stmt )
						except CompileError:
							continue

					# reaching the closing brace with no explicit `return` on
					# this path is __init__'s success path too - every
					# required attribute must already be initialized here.
					# Done BEFORE the branch below is even chosen: it cancels
					# each attribute's own epilogue entry (ownership transfers
					# into the now-complete self), which current_epilogue_label()
					# below has to see already applied - otherwise a
					# construction-only function with nothing else pending
					# would wrongly look like it still has a live entry to
					# jump to. A no-op whenever an explicit return already
					# completed construction on every reachable path (see
					# _stmt_Return's own identical call)
					if self._construction_self is not None:
						self._complete_construction_or_fail( fn )

					if self._cfg.current_epilogue_label() is not None:
						# some return (or OrJump) already jumped into the
						# shared epilogue ladder (_stmt_Return/_consume_checked_
						# result, via current_epilogue_label()), or nothing did
						# but entries are still pending at the function's own
						# closing brace (an implicit `return None`/fall-off
						# reaching them the same way) - either way,
						# build_epilogue_ladder() covers whatever's still
						# pending, RC decrefs and defer/errdefer replays alike
						self._emit_epilogue( fn, none_type, body_start )
					elif fn.return_type is none_type and self._body_may_fall_off_the_end( fn.node.body ):
						# nothing pending to unwind - but falling off the end
						# without an explicit `return` is still a real exit
						# (implicit `return None`, same as Python). Every
						# explicit `return` already does this itself (see
						# _stmt_Return's own else branch) - this only covers
						# the specific case nothing else does: reaching the
						# function's closing brace with no `return` at all
						self._emit( ir.Return( value = None ))

					self._emit( ir.FuncEnd( name = fn.qualname ))

		return self._instructions

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
		flag_inits = [
			ir.Assign( dest = flag, src = ir.Const( type = flag.type, value = False ))
			for flag in self._defer_flags
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
		bool_cls = self.discovery.find_name( 'bool', node )
		is_err_fn = self._attr_lookup_callable( self._return_value_var.type, 'is_err', node )
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
		if is_err_fn.resolve is not None:
			is_err_fn.resolve()
		return_type = self._return_value_var.type
		if isinstance( return_type, Specialization ) and return_type.base is is_err_fn.cls:
			# the receiver's type (self._return_value_var, always a
			# concrete Result[T,E] specialization by the time this runs)
			# already pins down the concrete args, so this is just an
			# ordinary monomorphization
			method_spec = self.discovery._get_or_create_specialization( is_err_fn, return_type.args )
			self.schedule( method_spec )
			is_err_fn = self._monomorphized_function( method_spec )
		else:
			self.schedule( is_err_fn )
		temp = ir.Temp( type = bool_cls, id = self._temp_id )
		self._temp_id += 1
		self._pending_temps.append( temp )
		return [
			ir.DeclareTemp( temp = temp ),
			ir.Call( dest = temp, target = is_err_fn, receiver = self._return_value_var, args = [], kwargs = {} ),
		], temp

	# --- __init__ construction (RCCLASS ATTRIBUTE LIFETIME.md) -----------------

	def _init_fallibility( self, fn: Function ) -> bool:
		''' __init__ must return None (non-fallible) or Result[None,E]
		(fallible - per SYNTAX.md, Foo(...) then returns Result[Foo,E]) -
		anything else is a compile error, checked as soon as __init__
		itself is lowered, independent of whether/where it's ever
		constructed from. '''
		none_type = self.discovery.get_none_type()
		if fn.return_type is none_type:
			return False
		shape = self._type_resolver._result_shape( fn.return_type )
		ok = shape is not None and shape[0] is none_type
		if not ok:
			self.discovery.fail(
				f'{fn.qualname} must return None or Result[None,_], got '
				f'{fn.return_type.qualname if fn.return_type else None}',
				fn.node,
			)
		return True

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
			with self.discovery.module_context( module ):
				with self.discovery.scope_context( cls ):
					default_value = self._lower_expr( attr.init, attr.type )
			for instr in self._cfg.attr_assign( attr, default_value, is_alias = self._is_aliasing_expr( attr.init )):
				self._emit( instr )
			self._emit( ir.SetAttr( obj = self_param, attr = attr.stem, value = default_value ))

	def _complete_construction_or_fail( self, fn: Function ) -> None:
		try:
			self._cfg.complete_construction( fn.qualname )
		except CompileError as e:
			self.discovery.fail_loc( str( e ), fn.file, fn.line )

	def _is_result_err_call( self, node: ast.expr | None ) -> str|None:
		# `return Result.Err(...)` - textually recognized, same spirit as
		# _defer_kind_of_call/_defer_kind_of_with - deliberately not
		# attempting deeper type-level inference (see _stmt_Return's own
		# comment on why anything else defaults to "requires completeness").
		# Returns the discriminant ('Result.Err') rather than a bare bool,
		# matching every other textual recognizer in this file
		if (
			isinstance( node, ast.Call )
			and isinstance( node.func, ast.Attribute )
			and node.func.attr == 'Err'
			and isinstance( node.func.value, ast.Name )
			and node.func.value.id == 'Result'
		):
			return 'Result.Err'
		return None

	def lower_global( self, var: Variable ) -> list[ir.Instruction]:
		module = self._find_module_for( var )
		self._init_lowering_state( None )

		with self.discovery.module_context( module ):
			if var.init is not None:
				self._pending_temps = []
				operand = self._lower_expr( var.init, var.type )
				self._emit( ir.Assign( dest = var, src = operand ))
				for t in reversed( self._pending_temps ):
					self._emit( ir.DeleteTemp( temp = t ))

		return self._instructions

	# --- module lookup ------------------------------------------------------

	def _find_module_for( self, unit: Function|Variable|ClassLike ) -> Module:
		# Function/Variable/ClassLike.file is always set to their owning
		# module's .file (see discovery.py's _parse_function/visit_AnnAssign/
		# visit_Assign/_parse_ClassDef_*) - none of them retain a direct
		# back-reference to the Module itself
		for module in self.discovery.modules.values():
			if module.file == unit.file:
				return module
		self.discovery.fail_loc( f'no module found owning {unit.qualname} (file={unit.file})', unit.file, unit.line )

	# --- temp/instruction bookkeeping ----------------------------------------

	def _emit( self, instr: ir.Instruction ) -> None:
		# a Call/Allocate's dest is always a genuinely fresh, owned value
		# from the caller's perspective (same rule _is_aliasing_expr already
		# encodes for Call; Allocate is fresh by definition) - registering it
		# here, centrally, at the exact moment it's actually emitted, is what
		# guarantees every one of these sites is covered instead of needing
		# individual fresh_temp() calls hunted down at each of the many
		# places that build a Call/Allocate (plain calls, generic calls,
		# conditional dispatch, union-receiver dispatch, struct/union
		# construction, ...). Gated on self._current_fn - lower_global()
		# never constructs a CFGState at all, and self._cfg would otherwise
		# be whatever function was lowered most recently (this Lowering
		# instance is reused across units), a strictly worse outcome than
		# just skipping it for globals
		if self._current_fn is not None and isinstance( instr, ( ir.Call, ir.Allocate )) and isinstance( instr.dest, ir.Temp ):
			self._cfg.fresh_temp( instr.dest, instr.dest.type )
		if self._current_fn is not None:
			self._check_self_escape_in( instr )
		self._instructions.append( instr )

	def _check_self_escape_in( self, instr: ir.Instruction ) -> None:
		# self can only be used as the receiver of `self.attr`
		# (GetAttr.obj/SetAttr.obj, deliberately excluded here) until
		# __init__ finishes constructing it - see cfg.check_self_escape().
		# Checking every OTHER operand field centrally, at emission time,
		# covers every site that could hand self off somewhere it shouldn't
		# without hunting each one down individually - same centralization
		# fresh_temp() already uses above. A no-op outside __init__
		# (check_self_escape() itself short-circuits when nothing's under
		# construction) - the instruction-type gate below also means this
		# never touches self._cfg before it exists (FuncStart, emitted
		# before CFGState is constructed, matches none of these types)
		operands: list[ir.Operand] = []
		if isinstance( instr, ir.Call ):
			if instr.receiver is not None:
				operands.append( instr.receiver )
			operands += instr.args
			operands += instr.kwargs.values()
		elif isinstance( instr, ir.Assign ):
			operands.append( instr.src )
		elif isinstance( instr, ir.Return ):
			if instr.value is not None:
				operands.append( instr.value )
		elif isinstance( instr, ir.SetItem ):
			operands.append( instr.value )
		elif isinstance( instr, ir.Allocate ):
			operands += instr.fields.values()
		for operand in operands:
			try:
				self._cfg.check_self_escape( operand, self._current_fn.qualname )
			except CompileError as e:
				self.discovery.fail_loc( str( e ), self._current_fn.file, self._current_fn.line )

	def _new_temp( self, t: Type ) -> ir.Temp:
		temp = ir.Temp( type = t, id = self._temp_id )
		self._temp_id += 1
		self._pending_temps.append( temp )
		self._emit( ir.DeclareTemp( temp = temp ))
		return temp

	def _new_label( self, prefix: str ) -> str:
		label = f'__{prefix}_{self._label_id}__'
		self._label_id += 1
		return label

	# --- statements ------------------------------------------------------------

	def _lower_stmt( self, node: ast.stmt ) -> None:
		# _pending_temps is shared/mutable rather than passed explicitly, so a
		# statement whose own handler recursively lowers nested statements
		# (currently only _stmt_With) must not let those nested calls' own
		# resets/flushes clobber this call's view of it - save/restore around
		# the whole thing, same idea as scope_context's stack push/pop
		outer_pending = self._pending_temps
		self._pending_temps = []
		try:
			method = getattr( self, f'_stmt_{node.__class__.__name__}', None )
			if method is None:
				self.discovery.fail( f'unsupported statement: {ast.unparse(node)}', node )
			method( node )
			for t in reversed( self._pending_temps ):
				# a temp genuinely fresh_temp()-registered (see _emit) and
				# never consumed by assign()/return_()/move()/field_value()
				# along the way (e.g. `foo( SomeClass() )` where SomeClass()
				# is passed into a plain, non-move[T] parameter - nothing
				# ever untracks it) still needs its own decref right here,
				# at the natural end of the temporary's own expression-scoped
				# lifetime. A no-op for every already-consumed temp (already
				# untracked by whichever hook consumed it) and every non-RC
				# temp (never registered in the first place)
				for instr in self._cfg.delete_temp( t ):
					self._emit( instr )
				self._emit( ir.DeleteTemp( temp = t ))
		finally:
			self._pending_temps = outer_pending

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
			self.discovery.fail( f'return is not allowed inside a defer/errdefer body: {ast.unparse(node)}', node )
		value = self._lower_expr( node.value, self._current_fn.return_type ) if node.value is not None else None
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
			is_success = not ( self._construction_fallible and self._is_result_err_call( node.value ) is not None )
			if is_success:
				self._complete_construction_or_fail( self._current_fn )
		label = self._cfg.current_epilogue_label( value )
		if label is not None:
			# whatever's still pending (RC decrefs, defer/errdefer replays)
			# gets unwound once, later, by the shared ladder every other
			# return reaching this same label also jumps into
			# (build_epilogue_ladder(), emitted at the function's own
			# closing brace - see _emit_epilogue) - value has to survive
			# the jump some other way than a direct ir.Return
			if self._return_value_var is not None and value is not None:
				self._emit( ir.Assign( dest = self._return_value_var, src = value ))
			self._emit( ir.Jump( target = label ))
		else:
			# either nothing is pending, or `value` IS itself one of the
			# still-live entries current_epilogue_label() can't route
			# through a shared label (see its own comment) - unwind inline,
			# right here, same as always. Still has to replay any pending
			# defer/errdefer entries itself (return_() does this now too -
			# they're just as "pending" as an RC decref from here)
			for instr in self._cfg.return_( value, lambda: self._build_is_err_check( node )):
				self._emit( instr )
			self._emit( ir.Return( value = value ))

	def _stmt_Pass( self, node: ast.Pass ) -> None:
		pass

	def _stmt_Global( self, node: ast.Global ) -> None:
		# a no-op: an unannotated Assign to a name that already exists
		# anywhere in the scope chain (local, enclosing, or global) always
		# reassigns that same one - find_name_or_none's scope chain already
		# falls through to the module scope on its own, so there's never a
		# separate shadowing local to opt out of. It only introduces a new
		# local when the name is unbound everywhere in the chain (see
		# _stmt_Assign's inference branch), which by definition has nothing
		# to shadow
		pass

	def _stmt_Delete( self, node: ast.Delete ) -> None:
		# del x - ends a local's lifetime early (see TODO.txt/RC MANAGEMENT.md:
		# a local created inside one arm of an if can be referenced only
		# within that arm unless it's del'd before the arm exits, matching
		# the other arm's "never created it either" state). Only a single
		# bare local name is supported - not del a.b, del a[i], or multiple
		# targets. Removing it from fn.names is enough on its own to make a
		# later reference fail (find_name won't find it) - the actual
		# Decref emission is cfg.py's job, wired in alongside its other hooks
		if len( node.targets ) != 1 or not isinstance( node.targets[0], ast.Name ):
			self.discovery.fail( f'del only supports a single local variable name: {ast.unparse(node)}', node )
		target = node.targets[0]
		fn = self._current_fn
		existing = fn.names.get( target.id )
		if not isinstance( existing, Variable ):
			self.discovery.fail( f'{target.id!r} is not a local variable, cannot del it', node )
		for instr in self._cfg.deleted( existing ):
			self._emit( instr )
		del fn.names[target.id]

	def _stmt_ImportFrom( self, node: ast.ImportFrom ) -> None:
		# local (in-function) form of discovery.py's own visit_ImportFrom -
		# function bodies are deliberately never walked by discovery.py's
		# own visitor (see this class's docstring: "function bodies were
		# deliberately left unvisited in stage 1"), so an import written
		# inside a function body (lib/sys.py's memzero()/_alloc()/etc. -
		# one FFI declaration per @compiler.target(os=...) branch) only
		# ever reaches here, never discovery.py's version. Registered
		# directly into the current function's own scope (add_name, same
		# as a parameter) rather than resolved/scheduled eagerly - an
		# unused import costs nothing, same posture as _expr_Name's lazy
		# resolve-on-use
		parts: list[str] = []
		if node.level:
			parts.extend( self.discovery.module_stack[-1].qualname.split( '.' )[:-node.level] )
			if not parts:
				self.discovery.fail( f'unable to relative import from here: {ast.unparse(node)}', node )
		if node.module:
			parts.append( node.module )
		package = '.'.join( parts )
		try:
			mod = self.discovery.import_name( package )
		except FileNotFoundError as e:
			self.discovery.fail( str( e ), node )
		if not mod:
			self.discovery.fail( f'module {package!r} not found', node )
		for alias in node.names:
			item = mod.names.get( alias.name )
			if item is None:
				self.discovery.fail( f'module {package} does not export {alias.name!r}', node )
			self._current_fn.add_name( alias.asname or alias.name, item )

	def _stmt_Import( self, node: ast.Import ) -> None:
		for alias in node.names:
			try:
				mod = self.discovery.import_name( alias.name )
			except FileNotFoundError as e:
				self.discovery.fail( str( e ), node )
			self._current_fn.add_name( alias.asname or alias.name, mod )

	def _is_aliasing_expr( self, node: ast.expr ) -> bool:
		# does lowering `node` hand back a reference to a value that
		# already exists independently (needing its own Incref if it's
		# stored into a new binding), vs a genuinely fresh value (Allocate,
		# or a Call - always a fresh owned handoff, whether the callee's
		# own body built it via Allocate or received it as an alias
		# itself, since a well-behaved callee already accounts for that on
		# its own side)? Name/Attribute reads are the only currently-
		# supported expression forms that alias existing state -
		# BinOp/BoolOp/Compare/Constant/UnaryOp never produce RC values at
		# all, and Call is always fresh from the caller's perspective.
		# ast.Subscript is deliberately NOT included here even though it
		# looks like a read: _expr_Subscript's dominant path (a real
		# __getitem__) is a Call underneath (fresh), and its other path
		# (raw ir.GetItem, genuinely aliasing a container element) isn't
		# reachable by any real code yet - no indexable container exists
		# yet (list[T]/dict[K,V] are still first-draft/WIP per TODO.txt) -
		# revisit this once one does
		return isinstance( node, ( ast.Name, ast.Attribute ))

	def _stmt_AnnAssign( self, node: ast.AnnAssign ) -> None:
		if not isinstance( node.target, ast.Name ):
			self.discovery.fail( f'unsupported AnnAssign target: {ast.unparse(node)}', node )
		fn = self._current_fn
		var_type = self.discovery.visit( node.annotation )
		var = Variable(
			stem = node.target.id,
			qualname = f'{fn.qualname}.{node.target.id}',
			file = fn.file,
			line = node.lineno,
			type = var_type,
		)
		fn.add_name( var.stem, var )
		self.schedule( var_type )
		if node.value is not None:
			operand = self._lower_expr( node.value, var_type )
			for instr in self._cfg.assign( var, operand, is_alias = self._is_aliasing_expr( node.value )):
				self._emit( instr )
			self._emit( ir.Assign( dest = var, src = operand ))

	def _stmt_Assign( self, node: ast.Assign ) -> None:
		if len( node.targets ) != 1:
			self.discovery.fail( f'multiple assignment targets not supported: {ast.unparse(node)}', node )
		target = node.targets[0]
		if isinstance( target, ast.Name ):
			existing = self.discovery.find_name_or_none( target.id )
			if existing is not None:
				if not isinstance( existing, Variable ):
					self.discovery.fail( f'{target.id!r} is not a variable, cannot assign to it', node )
				operand = self._lower_expr( node.value, existing.type )
				for instr in self._cfg.assign( existing, operand, is_alias = self._is_aliasing_expr( node.value )):
					self._emit( instr )
				self._emit( ir.Assign( dest = existing, src = operand ))
			else:
				# first assignment to a name with no prior declaration - same
				# as an AnnAssign, but the type is inferred from the RHS
				# instead of coming from an explicit annotation
				operand = self._lower_expr( node.value, None )
				fn = self._current_fn
				var = Variable(
					stem = target.id,
					qualname = f'{fn.qualname}.{target.id}',
					file = fn.file,
					line = node.lineno,
					type = operand.type,
				)
				fn.add_name( var.stem, var )
				self.schedule( var.type )
				for instr in self._cfg.assign( var, operand, is_alias = self._is_aliasing_expr( node.value )):
					self._emit( instr )
				self._emit( ir.Assign( dest = var, src = operand ))
		elif isinstance( target, ast.Attribute ):
			obj = self._lower_expr( target.value, None )
			attr_var = self._attr_lookup( obj.type, target.attr, target )
			operand = self._lower_expr( node.value, attr_var.type )
			if self._construction_self is not None and obj is self._construction_self:
				# self.<attr> = value, inside __init__ construction itself -
				# tracked for definite-assignment/self-escape purposes (see
				# RCCLASS ATTRIBUTE LIFETIME.md and cfg.attr_assign())
				for instr in self._cfg.attr_assign( attr_var, operand, is_alias = self._is_aliasing_expr( node.value )):
					self._emit( instr )
			elif cfg.rc_leaves( attr_var.type ):
				# ordinary SetAttr on an already-constructed instance -
				# "an RCClass is always complete, so setting an attribute
				# is always a replace" (RCCLASS ATTRIBUTE LIFETIME.md).
				# cfg.py doesn't track arbitrary struct instances' field
				# CONTENTS across statements (v1 scope cut - see cfg.py's
				# module docstring), so the current value is always read
				# fresh here rather than consulted from any tracked state
				old = self._new_temp( attr_var.type )
				self._emit( ir.GetAttr( dest = old, obj = obj, attr = target.attr ))
				for instr in self._cfg.attr_replace( attr_var.type, old, operand, is_alias = self._is_aliasing_expr( node.value )):
					self._emit( instr )
			self._emit( ir.SetAttr( obj = obj, attr = target.attr, value = operand ))
		elif isinstance( target, ast.Subscript ):
			obj = self._lower_expr( target.value, None )
			index = self._lower_expr( target.slice, None )
			operand = self._lower_expr( node.value, None )
			self._emit( ir.SetItem( obj = obj, index = index, value = operand ))
		else:
			self.discovery.fail( f'unsupported Assign target: {ast.unparse(node)}', node )

	def _stmt_AugAssign( self, node: ast.AugAssign ) -> None:
		# desugars x += y to x = x + y (reusing whatever arithmetic mode is
		# active, exactly like a hand-written x = x + y would) - only for a
		# bare Name target: this reads the target once (via the synthesized
		# BinOp) and writes it once (via the synthesized Assign), which is
		# only safe because a Name lookup has no side effects. An Attribute/
		# Subscript target's object/index expression would need evaluating
		# twice under this same desugaring (once to read, once to resolve
		# the write) - a real correctness risk (e.g. get_obj().x += 1 would
		# call get_obj() twice) - so those are left unsupported for now
		# rather than silently introducing a double-evaluation bug
		if not isinstance( node.target, ast.Name ):
			self.discovery.fail( f'unsupported AugAssign target: {ast.unparse(node)}', node )
		read = ast.Name( id = node.target.id, ctx = ast.Load() )
		ast.copy_location( read, node.target )
		binop = ast.BinOp( left = read, op = node.op, right = node.value )
		ast.copy_location( binop, node )
		assign = ast.Assign( targets = [ node.target ], value = binop )
		ast.copy_location( assign, node )
		self._stmt_Assign( assign )

	def _stmt_Expr( self, node: ast.Expr ) -> None:
		defer_kind = self._defer_kind_of_call( node.value )
		if defer_kind is not None:
			if len( node.value.args ) != 1 or node.value.keywords:
				self.discovery.fail( f'{defer_kind}(...) takes exactly one argument: {ast.unparse(node)}', node )
			single_stmt = ast.Expr( value = node.value.args[0] )
			ast.copy_location( single_stmt, node )
			self._register_defer_block( is_err_only = ( defer_kind == 'errdefer' ), body = [ single_stmt ], node = node )
			return
		if isinstance( node.value, ast.Constant ) and isinstance( node.value.value, str ):
			return # a docstring (or any other bare string literal used as a statement) - a no-op, same as _stmt_Pass
		if self._is_compiler_call( node.value ) == 'early_return':
			self._lower_compiler_early_return( node.value )
			return
		if self._is_compiler_call( node.value ) == 'decref':
			self._lower_compiler_decref( node.value )
			return
		if self._is_compiler_call( node.value ) == 'incref':
			self._lower_compiler_incref( node.value )
			return
		if not isinstance( node.value, ast.Call ):
			self.discovery.fail( f'unsupported expression statement: {ast.unparse(node)}', node )
		self._lower_call( node.value, None, want_result = False )

	def _stmt_With( self, node: ast.With ) -> None:
		if len( node.items ) != 1 or node.items[0].optional_vars is not None:
			self.discovery.fail( f'unsupported with statement: {ast.unparse(node)}', node )
		context_expr = node.items[0].context_expr

		defer_kind = self._defer_kind_of_with( context_expr )
		if defer_kind is not None:
			self._register_defer_block( is_err_only = ( defer_kind == 'errdefer' ), body = node.body, node = node )
			return

		attr = self._is_compiler_attr( context_expr )
		if attr == 'wrap_arithmetic':
			mode: arithmetic_mode.ArithmeticMode = arithmetic_mode.ArithmeticWrap()
		elif attr == 'saturate_arithmetic':
			mode = arithmetic_mode.ArithmeticSaturate()
		elif self._is_compiler_call( context_expr ) == 'panic_arithmetic':
			if len( context_expr.args ) != 1 or context_expr.keywords:
				self.discovery.fail( f'compiler.panic_arithmetic(...) takes exactly one argument: {ast.unparse(node)}', node )
			str_cls = self.discovery.find_name( 'str', node )
			errmsg = self._lower_expr( context_expr.args[0], str_cls )
			mode = arithmetic_mode.ArithmeticPanic( errmsg )
		else:
			self.discovery.fail( f'unsupported with statement: {ast.unparse(node)}', node )

		self._arithmetic_mode.append( mode )
		try:
			for stmt in node.body:
				# same per-statement recovery boundary as the top-level loop
				# in lower_function - one bad statement inside the with-block
				# doesn't stop the rest of it from being lowered
				try:
					self._lower_stmt( stmt )
				except CompileError:
					continue
		finally:
			self._arithmetic_mode.pop()

	def _is_compiler_attr( self, node: ast.expr ) -> str|None:
		# textual recognition, same as discovery.py's _is_compiler_target_call -
		# `compiler` is a special pseudo-module (Discovery.compiler_module),
		# not something with a real .names dict to resolve this through
		if (
			isinstance( node, ast.Attribute )
			and isinstance( node.value, ast.Name )
			and node.value.id == 'compiler'
		):
			return node.attr
		return None

	def _is_compiler_call( self, node: ast.expr ) -> str|None:
		if (
			isinstance( node, ast.Call )
			and isinstance( node.func, ast.Attribute )
			and isinstance( node.func.value, ast.Name )
			and node.func.value.id == 'compiler'
		):
			return node.func.attr
		else:
			return None

	def _lower_compiler_sizeof( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.sizeof(T) is a compile-time constant whenever T is
		# already concrete - it folds directly to an ir.Const, no runtime
		# computation involved. T is a TYPE reference, not a value, so its
		# argument is resolved via _try_resolve_namespace (same as a
		# generic call's own [T] argument), not _lower_expr (which would
		# reject it - "not a value" - since a bare type isn't a Variable)
		if len( node.args ) != 1 or node.keywords:
			self.discovery.fail( f'compiler.sizeof(...) takes exactly one type argument: {ast.unparse(node)}', node )
		target_type = self._try_resolve_namespace( node.args[0] )
		if target_type is None:
			self.discovery.fail( f'compiler.sizeof(...) argument must be a type: {ast.unparse(node)}', node )
		if isinstance( target_type, TypeVar ):
			self.discovery.fail(
				f'compiler.sizeof({target_type.stem}) requires a concrete type - {target_type.qualname} is still an '
				f'unbound generic type parameter here (call the enclosing function through an explicit specialization, e.g. foo[SomeType](...))',
				node,
			)
		usize_cls = self.discovery.get_intrinsics()['usize']
		if size := getattr( target_type, 'sizeof', None ):
			return ir.Const( type = expected_type or usize_cls, value = size )
		# a real class-like type (RCClass/CStruct/CUnion/TaggedUnion, or a
		# concrete Specialization of one) - no field-layout algorithm exists
		# in this compiler (nor should one - that's the C compiler's own
		# job), so unlike an intrinsic scalar's sizeof, this can't fold to a
		# Python int here. Stays a real ir.SizeOf instruction instead - the
		# emitter emits a literal C `sizeof(...)` expression, letting the
		# target C compiler compute the real, layout-dependent size (needed
		# by sys.alloc[T]'s own body, e.g. sys.alloc[SomeRCClass](1) for
		# RCClass construction - see _lower_allocate_fields's RCClass branch)
		base = target_type.base if isinstance( target_type, Specialization ) else target_type
		if not isinstance( base, ( RCClass, CStruct, CUnion, TaggedUnion )):
			self.discovery.fail( f'compiler.sizeof({target_type.qualname}) is not supported yet - only intrinsic scalar types and real classes have a known size', node )
		self.schedule( target_type )
		dest = self._new_temp( expected_type or usize_cls )
		self._emit( ir.SizeOf( dest = dest, type = target_type ))
		return dest

	def _lower_compiler_refcount( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.refcount(x) - unlike compiler.sizeof(T), x is a real
		# VALUE (an RC object), not a type reference, so it's lowered via
		# _lower_expr like any other argument. A genuine runtime read (the
		# header's current count), not a compile-time constant - deliberately
		# opaque at this level (ir.RefCount), same spirit as Incref/Decref;
		# what it actually reads is a codegen/emitter concern, not this pass's
		if len( node.args ) != 1 or node.keywords:
			self.discovery.fail( f'compiler.refcount(...) takes exactly one argument: {ast.unparse(node)}', node )
		value = self._lower_expr( node.args[0], None )
		if not self._type_resolver._is_RC( value.type ):
			self.discovery.fail(
				f'compiler.refcount(...) argument must be a reference-counted value, not '
				f'{value.type.qualname if value.type else "?"}: {ast.unparse(node)}',
				node,
			)
		usize_cls = self.discovery.get_intrinsics()['usize']
		dest = self._new_temp( expected_type or usize_cls )
		self._emit( ir.RefCount( dest = dest, value = value ))
		return dest

	def _lower_compiler_addrof( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.addrof(x) -> Ptr[T], translating directly to C's &x - x
		# must be a bare local variable/parameter name (matches SYNTAX.md's
		# "local variable" wording and C's own lvalue-only restriction on
		# &), not an arbitrary expression. _expr_Name already only ever
		# resolves to a Variable (never a Temp), so requiring the argument's
		# AST shape to be ast.Name is what actually enforces this - lowering
		# it via _lower_expr like any other value would silently accept e.g.
		# compiler.addrof(x.field), which has no address to take here (no
		# field-layout computation exists yet - that's an emitter concern)
		if len( node.args ) != 1 or node.keywords:
			self.discovery.fail( f'compiler.addrof(...) takes exactly one argument: {ast.unparse(node)}', node )
		arg_node = node.args[0]
		if not isinstance( arg_node, ast.Name ):
			self.discovery.fail( f'compiler.addrof(...) argument must be a bare local variable, not {ast.unparse(node)}', node )
		value = self._lower_expr( arg_node, None )
		ptr_cls = self.discovery.get_intrinsics()['Ptr']
		ptr_type = self.discovery._get_or_create_specialization( ptr_cls, [ value.type ] )
		dest = self._new_temp( expected_type or ptr_type )
		self._emit( ir.AddrOf( dest = dest, value = value ))
		return dest

	def _lower_scalar_cast( self, target_type: Scalar, source: ast.expr|ir.Operand, node: ast.AST ) -> ir.Operand:
		# shared by compiler.cast(T, x) and T(x) construction-sugar - the
		# one place the actual Scalar-to-Scalar conversion logic lives.
		# `source` is EITHER an unlowered ast.expr (a bare literal - always
		# succeeds via bit-reinterpretation, decided at compile time, no
		# Result involved - -11 reinterpreted as u32 is exactly the
		# well-defined two's-complement value real WinAPI constants like
		# STD_OUTPUT_HANDLE rely on) OR an already-lowered ir.Operand (a
		# real runtime value, where "does this fit" is a genuine runtime
		# question - respects self._arithmetic_mode exactly like +/-/*
		# already do, reusing the same Check/Wrap/Saturate/panic_arithmetic
		# machinery, not a separate concept)
		if isinstance( source, ast.expr ):
			return self._lower_expr( source, target_type )
		operand = source
		opcode, extra = self._arithmetic_mode[-1].GetCast()
		return self._lower_arithmetic_op( node, opcode, extra, target_type, { 'operand': operand }, 'cast' )

	def _lower_compiler_cast( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.cast(T, x) - T is a TYPE reference (resolved via
		# _try_resolve_namespace, same as compiler.sizeof's argument, not
		# _lower_expr), x is a real value. The call's own target type is
		# always authoritative for the result - unlike an ordinary literal,
		# an explicit cast overrides whatever the ambient expected_type is
		if len( node.args ) != 2 or node.keywords:
			self.discovery.fail( f'compiler.cast(...) takes exactly two arguments: {ast.unparse(node)}', node )
		target_type = self._try_resolve_namespace( node.args[0] )
		if target_type is None:
			self.discovery.fail( f'compiler.cast(...) first argument must be a type: {ast.unparse(node)}', node )
		if isinstance( target_type, TypeVar ):
			self.discovery.fail(
				f'compiler.cast({target_type.stem}, ...) requires a concrete type - {target_type.qualname} is still an '
				f'unbound generic type parameter here (call the enclosing function through an explicit specialization, e.g. foo[SomeType](...))',
				node,
			)
		if not isinstance( target_type, Scalar ):
			self.discovery.fail( f'compiler.cast({target_type.qualname}, ...) is not supported yet - only Scalar-to-Scalar casts are, for now', node )
		value_node = node.args[1]
		if isinstance( value_node, ast.Constant ):
			return self._lower_scalar_cast( target_type, value_node, node )
		value = self._lower_expr( value_node, None )
		if not isinstance( value.type, Scalar ):
			self.discovery.fail(
				f'compiler.cast(...) second argument must be a scalar value, not {value.type.qualname if value.type else "?"}: {ast.unparse(node)}',
				node,
			)
		return self._lower_scalar_cast( target_type, value, node )

	def _lower_compiler_early_return( self, node: ast.Call ) -> None:
		# compiler.early_return(err) - a same-function early bailout: usable
		# anywhere inside a function that itself returns Result[_,_], to
		# return Result.Err(err) immediately without writing the boilerplate
		# out by hand. Desugars to `return Result.Err(err)` and delegates to
		# _stmt_Return so it reuses the epilogue-vs-plain-Return split (and
		# errdefer's is_err() epilogue check, which already treats any
		# Result.Err landing in the return slot uniformly, not just the
		# OrJump path) rather than duplicating either.
		#
		# NOTE: Result.or_return()'s own written body uses this same call
		# (`compiler.early_return(self.data.v_Err)`), but that body is
		# never actually lowered as a real function - it's a spec, not
		# compilable code, because it would need this to trigger a return
		# in ITS CALLER's scope, not or_return()'s own (or_return's declared
		# return type is bare T, not Result[T,E] - `return Result.Err(...)`
		# from inside it could never type-check there). or_return() calls
		# are instead recognized and expanded directly at the call site -
		# see _lower_or_return.
		if len( node.args ) != 1 or node.keywords:
			self.discovery.fail( f'compiler.early_return(...) takes exactly one argument: {ast.unparse(node)}', node )
		fn = self._current_fn
		return_type = fn.return_type if fn is not None else None
		ok = fn is not None and self._type_resolver._result_shape( return_type ) is not None
		if not ok:
			where = f'{fn.qualname} returns {return_type.qualname if return_type else None}' if fn is not None else 'this is not inside a function'
			self.discovery.fail( f'compiler.early_return(...) requires the enclosing function to return Result[_,_] ({where})', node )

		err_call = ast.Call(
			func = ast.Attribute( value = ast.Name( id = 'Result', ctx = ast.Load() ), attr = 'Err', ctx = ast.Load() ),
			args = [ node.args[0] ],
			keywords = [],
		)
		ast.copy_location( err_call, node )
		ast.fix_missing_locations( err_call )
		return_stmt = ast.Return( value = err_call )
		ast.copy_location( return_stmt, node )
		self._stmt_Return( return_stmt )

	def _lower_compiler_decref( self, node: ast.Call ) -> None:
		# compiler.decref(x) — emit an ir.Decref for x. Used inside
		# synthesized destructor bodies to tear down each RC field.
		if len( node.args ) != 1 or node.keywords:
			self.discovery.fail( f'compiler.decref(...) takes exactly one argument: {ast.unparse(node)}', node )
		operand = self._lower_expr( node.args[0], None )
		if operand.type is None or not self._type_resolver._is_RC( operand.type ):
			self.discovery.fail(
				f'compiler.decref(...) argument must be a reference-counted value, not '
				f'{operand.type.qualname if operand.type else "?"}: {ast.unparse(node)}',
				node,
			)
		self._emit( ir.Decref( value = operand ))

	def _lower_compiler_incref( self, node: ast.Call ) -> None:
		# compiler.incref(x) — emit an ir.Incref for x.
		if len( node.args ) != 1 or node.keywords:
			self.discovery.fail( f'compiler.incref(...) takes exactly one argument: {ast.unparse(node)}', node )
		operand = self._lower_expr( node.args[0], None )
		if operand.type is None or not self._type_resolver._is_RC( operand.type ):
			self.discovery.fail(
				f'compiler.incref(...) argument must be a reference-counted value, not '
				f'{operand.type.qualname if operand.type else "?"}: {ast.unparse(node)}',
				node,
			)
		self._emit( ir.Incref( value = operand ))

	# --- defer/errdefer ----------------------------------------------------------

	def _defer_kind_of_with( self, node: ast.expr ) -> str|None:
		# `with defer:` / `with errdefer:` - bare names, unlike the
		# compiler.-prefixed arithmetic-mode context managers
		if isinstance( node, ast.Name ) and node.id in ( 'defer', 'errdefer' ):
			return node.id
		return None

	def _defer_kind_of_call( self, node: ast.expr ) -> str|None:
		# `defer( expr )` / `errdefer( expr )` - the single-statement call form
		if isinstance( node, ast.Call ) and isinstance( node.func, ast.Name ) and node.func.id in ( 'defer', 'errdefer' ):
			return node.func.id
		return None

	def _body_may_fall_off_the_end( self, body: list[ast.stmt] ) -> bool:
		# a simple, deliberately narrow check (not full terminator analysis -
		# same "future work" scope cut as _stmt_If's own true_terminates/
		# false_terminates detection): true whenever the LAST top-level
		# statement isn't itself a `return` (an empty body, or one ending in
		# a plain statement/if/loop/etc, could all still fall through to the
		# function's own closing brace). A body that's actually unreachable
		# past this point (both branches of a trailing if already return,
		# a trailing `while True:` with no break, ...) is a false positive -
		# harmless, since the resulting fall-off unwind+Return is then
		# genuinely dead code, never executed
		return not body or not isinstance( body[-1], ast.Return )

	def _register_defer_block( self, is_err_only: bool, body: list[ast.stmt], node: ast.AST ) -> None:
		kind = 'errdefer' if is_err_only else 'defer'
		if self._loop_depth > 0:
			self.discovery.fail( f'{kind} is not allowed inside a loop - call another function and {kind} inside that instead', node )
		if self._in_deferred_body:
			self.discovery.fail( f'{kind} cannot be nested inside another defer/errdefer', node )

		fn = self._current_fn
		if is_err_only:
			return_type = fn.return_type if fn is not None else None
			ok = fn is not None and self._type_resolver._result_shape( return_type ) is not None
			if not ok:
				where = f'{fn.qualname} returns {return_type.qualname if return_type else None}' if fn is not None else 'this is not inside a function'
				self.discovery.fail( f'errdefer requires the enclosing function to return Result[_,_] ({where})', node )

		bool_cls = self.discovery.find_name( 'bool', node )
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
		self._cfg.push_defer( captured, flag, is_err_only )
		# this is what actually runs at the with-statement's/call's position -
		# marks the block "armed" so the epilogue knows to replay it
		self._emit( ir.Assign( dest = flag, src = ir.Const( type = bool_cls, value = True )))

	# --- loops ---------------------------------------------------------------

	def _stmt_While( self, node: ast.While ) -> None:
		if node.orelse:
			self.discovery.fail( 'while/else is not supported', node )
		bool_cls = self.discovery.find_name( 'bool', node )
		start_label = self._new_label( 'while_start' )
		end_label = self._new_label( 'while_end' )
		# the test is positioned right after start_label (re-lowered here
		# once, but the resulting instructions physically sit inside the
		# repeated block, same as _stmt_If's test) so it's genuinely
		# re-evaluated every time the bottom Jump loops back
		self._emit( ir.Label( name = start_label ))
		test = self._lower_expr( node.test, bool_cls )
		self._emit( ir.JumpIfFalse( cond = test, target = end_label ))
		loop_snapshot = self._cfg.snapshot()
		self._lower_loop_body( node.body, continue_label = start_label, break_label = end_label, loop_snapshot = loop_snapshot )
		try:
			back_edge_instructions = self._cfg.loop_back_edge( loop_snapshot.bindings, self._current_fn.qualname )
		except CompileError as e:
			self.discovery.fail( str( e ), node )
		for instr in back_edge_instructions:
			self._emit( instr )
		self._cfg.restore( loop_snapshot )
		self._emit( ir.Jump( target = start_label ))
		self._emit( ir.Label( name = end_label ))

	def _lower_loop_body( self, body: list[ast.stmt], continue_label: str, break_label: str, loop_snapshot: object ) -> None:
		self._loop_depth += 1
		self._loop_labels.append(( continue_label, break_label, loop_snapshot ))
		try:
			for stmt in body:
				try:
					self._lower_stmt( stmt )
				except CompileError:
					continue
		finally:
			self._loop_labels.pop()
			self._loop_depth -= 1

	def _stmt_Break( self, node: ast.Break ) -> None:
		if not self._loop_labels:
			self.discovery.fail( 'break outside a loop', node )
		_, break_label, loop_snapshot = self._loop_labels[-1]
		for instr in self._cfg.unwind_to( loop_snapshot ):
			self._emit( instr )
		self._emit( ir.Jump( target = break_label ))

	def _stmt_Continue( self, node: ast.Continue ) -> None:
		if not self._loop_labels:
			self.discovery.fail( 'continue outside a loop', node )
		continue_label, _, loop_snapshot = self._loop_labels[-1]
		for instr in self._cfg.unwind_to( loop_snapshot ):
			self._emit( instr )
		self._emit( ir.Jump( target = continue_label ))

	def _synth_name( self, stem: str, node: ast.AST ) -> ast.Name:
		n = ast.Name( id = stem, ctx = ast.Load() )
		ast.copy_location( n, node )
		return n

	def _declare_hidden_local( self, stem: str, type: Type, node: ast.AST ) -> Variable:
		# compiler-synthesized locals (for-loop scaffolding: the once-
		# evaluated iterable, its length, the hidden index counter) - real
		# named Variables (not anonymous Temps) registered into the
		# function's flat names dict, the same way `self` gets synthesized
		# in lower_function, so synthetic ast.Name references to them
		# resolve normally through the existing _expr_Name/_stmt_Assign
		# machinery instead of duplicating it
		fn = self._current_fn
		var = Variable( stem = stem, qualname = f'{fn.qualname}.{stem}', file = fn.file, line = getattr( node, 'lineno', None ), type = type )
		fn.add_name( stem, var )
		self.schedule( type )
		return var

	def _find_method( self, owner_type: Type|None, name: str ) -> Function|None:
		# a non-failing probe, unlike _attr_lookup_callable - "this type has
		# no such method" is a normal, expected outcome for callers here
		# (for loop iterability checks, __getitem__'s raw-GetItem fallback),
		# not a real error to report
		owner_type = self._ensure_resolved( owner_type )
		names = getattr( owner_type, 'names', None )
		found = names.get( name ) if isinstance( names, dict ) else None
		return found if isinstance( found, Function ) else None

	def _maybe_consume_result( self, node: ast.AST, value: ir.Temp, alternatives: str ) -> ir.Operand:
		# if `value` is itself a Result[T,E], auto-consume it via the same
		# OrReturn/OrJump propagation or_return()/checked arithmetic use -
		# unlike _lower_or_return, a non-Result value is passed through
		# unchanged rather than rejected, since not every method this is
		# used for (__getitem__, __len__) is necessarily fallible. Uses
		# find_name_or_none (not find_name) - unlike every other Result
		# lookup in this file, this one runs speculatively for ANY value,
		# so a program that never defines Result at all (or hasn't
		# imported builtins) must not hard-fail here just because this
		# particular value happens not to be Result-shaped
		shape = self._type_resolver._result_shape( value.type )
		if shape is None:
			return value
		result_type, error_cls = shape
		self._type_resolver._require_result_return( node, value.type.base, error_cls, alternatives, fn = self._current_fn )
		return self._consume_checked_result( value, result_type, extra = None )

	def _bind_loop_target( self, target: ast.Name, default_type: Type, value_expr: ast.expr, node: ast.AST ) -> Variable:
		# mirrors _stmt_Assign's Name-target "reuse existing, else infer/
		# declare" rule (`for i in range(count):` reuses `i` if a variable
		# of that name already exists - e.g. str.concat in lib/builtins/
		# __init__.py pre-declares `i: usize = 0` before its own for loop)
		# - except a fresh declaration falls back to `default_type` instead
		# of failing outright, since value_expr may be a bare literal
		# (range()'s implicit start=0) with no type of its own to infer from
		existing = self.discovery.find_name_or_none( target.id )
		if existing is not None and not isinstance( existing, Variable ):
			self.discovery.fail( f'{target.id!r} is not a variable, cannot use it as a for loop target', node )
		expected = existing.type if existing is not None else default_type
		operand = self._lower_expr( value_expr, expected )
		if existing is not None:
			self._emit( ir.Assign( dest = existing, src = operand ))
			return existing
		fn = self._current_fn
		var = Variable( stem = target.id, qualname = f'{fn.qualname}.{target.id}', file = fn.file, line = getattr( node, 'lineno', None ), type = operand.type )
		fn.add_name( var.stem, var )
		self.schedule( var.type )
		self._emit( ir.Assign( dest = var, src = operand ))
		return var

	def _is_range_call( self, node: ast.expr ) -> str|None:
		# range(...) is textually recognized as compiler sugar, same as
		# compiler.wrap_arithmetic/defer/etc. - there's no real range()
		# function (TODO.txt: a real range()/Iterator needs the generator
		# state-machine transform, which doesn't exist yet). This covers
		# exactly the 1-2 arg counting-loop shape real lib/ code already
		# uses (str.concat's `for i in range(count):`). Returns the
		# discriminant ('range') rather than a bare bool, matching every
		# other textual recognizer in this file
		if isinstance( node, ast.Call ) and isinstance( node.func, ast.Name ) and node.func.id == 'range':
			return 'range'
		return None

	_FOR_LOOP_ALTERNATIVES = 'call .__len__()/.__getitem__() directly and consume their Result yourself instead'

	def _stmt_For( self, node: ast.For ) -> None:
		if not isinstance( node.target, ast.Name ):
			self.discovery.fail( f'for loop target must be a plain name: {ast.unparse(node)}', node )
		if node.orelse:
			self.discovery.fail( 'for/else is not supported', node )
		if self._is_range_call( node.iter ) is not None:
			self._lower_for_range( node )
		else:
			self._lower_for_over_indexable( node )

	def _lower_for_range( self, node: ast.For ) -> None:
		call = node.iter
		if call.keywords:
			self.discovery.fail( f'range(...) does not support keyword arguments: {ast.unparse(call)}', call )
		if len( call.args ) == 1:
			start_expr = ast.Constant( value = 0 )
			ast.copy_location( start_expr, call )
			stop_expr = call.args[0]
		elif len( call.args ) == 2:
			start_expr, stop_expr = call.args
		else:
			self.discovery.fail( f'range(...) supports 1 or 2 arguments only (no step yet): {ast.unparse(call)}', call )

		usize_cls = self.discovery.get_intrinsics()['usize']
		bool_cls = self.discovery.find_name( 'bool', node )

		target_var = self._bind_loop_target( node.target, usize_cls, start_expr, node )

		stop_operand = self._lower_expr( stop_expr, usize_cls )
		stop_var = self._declare_hidden_local( f'__for_stop_{self._label_id}', usize_cls, node )
		self._emit( ir.Assign( dest = stop_var, src = stop_operand ))

		start_label = self._new_label( 'for_start' )
		continue_label = self._new_label( 'for_continue' )
		end_label = self._new_label( 'for_end' )

		self._emit( ir.Label( name = start_label ))
		test = ast.Compare( left = self._synth_name( target_var.stem, node ), ops = [ ast.Lt() ], comparators = [ self._synth_name( stop_var.stem, node ) ] )
		ast.copy_location( test, node )
		cond = self._lower_expr( test, bool_cls )
		self._emit( ir.JumpIfFalse( cond = cond, target = end_label ))

		loop_snapshot = self._cfg.snapshot()
		self._lower_loop_body( node.body, continue_label = continue_label, break_label = end_label, loop_snapshot = loop_snapshot )
		try:
			back_edge_instructions = self._cfg.loop_back_edge( loop_snapshot.bindings, self._current_fn.qualname )
		except CompileError as e:
			self.discovery.fail( str( e ), node )
		for instr in back_edge_instructions:
			self._emit( instr )
		self._cfg.restore( loop_snapshot )

		self._emit( ir.Label( name = continue_label ))
		# the increment is a compiler-synthesized implementation detail of
		# the loop, not user-written arithmetic - it's structurally
		# guaranteed safe (target_var < stop_var strictly before every
		# increment), so it bypasses the ambient arithmetic-mode policy
		# entirely (AddWrap directly) rather than imposing a
		# Result[_,OverflowError]/wrap_arithmetic/etc. requirement on
		# ordinary for-loops
		incr = self._new_temp( usize_cls )
		self._emit( ir.AddWrap( dest = incr, left = target_var, right = ir.Const( type = usize_cls, value = 1 ) ))
		self._emit( ir.Assign( dest = target_var, src = incr ))
		self._emit( ir.Jump( target = start_label ))
		self._emit( ir.Label( name = end_label ))

	def _lower_for_over_indexable( self, node: ast.For ) -> None:
		usize_cls = self.discovery.get_intrinsics()['usize']
		bool_cls = self.discovery.find_name( 'bool', node )

		obj = self._lower_expr( node.iter, None )
		len_fn = self._find_method( obj.type, '__len__' )
		getitem_fn = self._find_method( obj.type, '__getitem__' )
		missing = [ name for name, fn in (( '__len__', len_fn ), ( '__getitem__', getitem_fn )) if fn is None ]
		if missing:
			self.discovery.fail( f'for loop needs {" and ".join(missing)} on {obj.type.qualname if obj.type else "?"}: {ast.unparse(node)}', node )

		unique = self._label_id
		obj_var = self._declare_hidden_local( f'__for_obj_{unique}', obj.type, node )
		self._emit( ir.Assign( dest = obj_var, src = obj ))

		self._ensure_resolved( len_fn )
		self.schedule( len_fn.return_type )
		len_dest = self._new_temp( len_fn.return_type )
		self._emit( ir.Call( dest = len_dest, target = len_fn, receiver = obj_var, args = [], kwargs = {} ))
		len_operand = self._maybe_consume_result( node, len_dest, self._FOR_LOOP_ALTERNATIVES )
		len_var = self._declare_hidden_local( f'__for_len_{unique}', len_operand.type, node )
		self._emit( ir.Assign( dest = len_var, src = len_operand ))

		index_var = self._declare_hidden_local( f'__for_index_{unique}', usize_cls, node )
		self._emit( ir.Assign( dest = index_var, src = ir.Const( type = usize_cls, value = 0 ) ))

		start_label = self._new_label( 'for_start' )
		continue_label = self._new_label( 'for_continue' )
		end_label = self._new_label( 'for_end' )

		self._emit( ir.Label( name = start_label ))
		test = ast.Compare( left = self._synth_name( index_var.stem, node ), ops = [ ast.Lt() ], comparators = [ self._synth_name( len_var.stem, node ) ] )
		ast.copy_location( test, node )
		cond = self._lower_expr( test, bool_cls )
		self._emit( ir.JumpIfFalse( cond = cond, target = end_label ))

		# the snapshot is taken here, BEFORE the loop target's own binding -
		# that binding (e.g. `s2 = obj[index]`) happens fresh every
		# iteration, exactly like any other loop-body statement (matches
		# foo4: a value reassigned each iteration is expected to be stable
		# across the back edge, not confined-and-torn-down)
		loop_snapshot = self._cfg.snapshot()
		subscript = ast.Subscript(
			value = self._synth_name( obj_var.stem, node ),
			slice = self._synth_name( index_var.stem, node ),
			ctx = ast.Load(),
		)
		ast.copy_location( subscript, node )
		bind = ast.Assign( targets = [ node.target ], value = subscript )
		ast.copy_location( bind, node )
		self._stmt_Assign( bind )

		self._lower_loop_body( node.body, continue_label = continue_label, break_label = end_label, loop_snapshot = loop_snapshot )
		try:
			back_edge_instructions = self._cfg.loop_back_edge( loop_snapshot.bindings, self._current_fn.qualname )
		except CompileError as e:
			self.discovery.fail( str( e ), node )
		for instr in back_edge_instructions:
			self._emit( instr )
		self._cfg.restore( loop_snapshot )

		self._emit( ir.Label( name = continue_label ))
		incr = self._new_temp( usize_cls )
		self._emit( ir.AddWrap( dest = incr, left = index_var, right = ir.Const( type = usize_cls, value = 1 ) ))
		self._emit( ir.Assign( dest = index_var, src = incr ))
		self._emit( ir.Jump( target = start_label ))
		self._emit( ir.Label( name = end_label ))

	def _stmt_If( self, node: ast.If ) -> None:
		bool_cls = self.discovery.find_name( 'bool', node )
		test = self._lower_expr( node.test, bool_cls )
		else_label = self._new_label( 'if_else' )
		self._emit( ir.JumpIfFalse( cond = test, target = else_label ))

		# each branch is lowered into its OWN captured instruction list
		# (same technique _register_defer_block already uses) rather than
		# appended directly - a local confined to just one branch needs its
		# own Decref spliced into THAT branch's own code specifically
		# (before its own exit to the join point), never at the shared
		# join point both branches reach, since it only exists on that one
		# path. Known ahead of time only after BOTH branches have been
		# explored (merge_if, below), so neither branch's own instructions
		# can be emitted directly as they're lowered
		entry_snapshot = self._cfg.snapshot()
		outer_instructions = self._instructions
		self._instructions = []
		for stmt in node.body:
			try:
				self._lower_stmt( stmt )
			except CompileError:
				continue
		true_captured = self._instructions
		true_end = dict( self._cfg.bindings )
		# return/break/continue as a branch's own last statement means
		# that branch never reaches the if's join point at all - see
		# merge_if()'s own comment on why that has to be treated
		# differently from an ordinary falling-through branch (full
		# terminator/dead-code analysis for anything deeper - nested ifs
		# that both terminate, etc - is future work, not attempted here)
		true_terminates = bool( node.body ) and isinstance( node.body[-1], ( ast.Return, ast.Break, ast.Continue ))

		if node.orelse:
			self._cfg.restore( entry_snapshot )
			self._instructions = []
			for stmt in node.orelse:
				try:
					self._lower_stmt( stmt )
				except CompileError:
					continue
			false_captured = self._instructions
			false_end = dict( self._cfg.bindings )
			false_terminates = bool( node.orelse ) and isinstance( node.orelse[-1], ( ast.Return, ast.Break, ast.Continue ))
		else:
			false_captured = []
			false_end = dict( entry_snapshot.bindings )
			false_terminates = False

		self._cfg.restore( entry_snapshot )
		self._instructions = outer_instructions
		try:
			true_extra, false_extra, removed = self._cfg.merge_if(
				entry_snapshot.bindings, true_end, false_end, self._current_fn.qualname,
				true_terminates = true_terminates, false_terminates = false_terminates,
			)
		except CompileError as e:
			self.discovery.fail( str( e ), node )
		for name in removed:
			del self._current_fn.names[name]

		for instr in true_captured:
			self._emit( instr )
		for instr in true_extra:
			self._emit( instr )
		if node.orelse:
			end_label = self._new_label( 'if_end' )
			self._emit( ir.Jump( target = end_label ))
			self._emit( ir.Label( name = else_label ))
			for instr in false_captured:
				self._emit( instr )
			for instr in false_extra:
				self._emit( instr )
			self._emit( ir.Label( name = end_label ))
		else:
			self._emit( ir.Label( name = else_label ))

	# --- expressions -----------------------------------------------------------

	def _lower_expr( self, node: ast.expr, expected_type: Type|None ) -> ir.Operand:
		method = getattr( self, f'_expr_{node.__class__.__name__}', None )
		if method is None:
			self.discovery.fail( f'unsupported expression: {ast.unparse(node)}', node )
		return method( node, expected_type )

	def _expr_Name( self, node: ast.Name, expected_type: Type|None ) -> ir.Operand:
		name = self.discovery.find_name( node.id, node )
		if not isinstance( name, Variable ):
			self.discovery.fail( f'{node.id!r} is not a value, cannot use it as an expression', node )
		self._ensure_resolved( name )
		# when a pointer-typed local flows into a context expecting a
		# differently-typed pointer (e.g. return ptr where ptr: Ptr[u8]
		# but the function returns Ptr[T]), insert a CastWrap — in C all
		# object pointers have the same representation, so this is safe
		if expected_type is not None and name.type is not expected_type and self._type_resolver._is_ptr_specialization( name.type ) and self._type_resolver._is_ptr_specialization( expected_type ):
			dest = self._new_temp( expected_type )
			self._emit( ir.CastWrap( dest = dest, operand = name ))
			return dest
		return name

	def _expr_Constant( self, node: ast.Constant, expected_type: Type|None ) -> ir.Operand:
		if expected_type is None:
			if isinstance( node.value, bool ):
				expected_type = self.discovery.get_intrinsics()['bool']
			elif isinstance( node.value, int ):
				# integer literals default to i32 when no contextual type is
				# available (bare `x = 1`, generic-call arg inference, etc.)
				# TODO FIXME: for most user code, this should probably be builtins.int and get scheduled as an immortal constant
				expected_type = self.discovery.get_intrinsics()['i32']
			else:
				self.discovery.fail(
					f'cannot infer the type of literal {node.value!r} - no expected type available from context ({ast.unparse(node)})',
					node,
				)
		return ir.Const( type = expected_type, value = node.value )

	def _expr_Attribute( self, node: ast.Attribute, expected_type: Type|None ) -> ir.Operand:
		# CEnum member VALUE expressions (OSError.FileNotFoundError used as
		# a runtime value) — the base is a class, not a runtime value, so
		# the normal _lower_expr path would reject it. Walk the namespace
		# chain through .names dicts (type_resolution.py already resolved
		# and scheduled every link) and fold the member to an ir.Const.
		chain = self.find_name_recursive( node )
		if chain is not None:
			obj, attr = chain
			if isinstance( obj, CEnum ):
				assert obj.resolve is None, (
					f'CEnum {obj.qualname} reached lowering unresolved — '
					f'type_resolution.py visit_Attribute should have resolved it'
				)
				value = obj.members.get( attr )
				if value is not None:
					return ir.Const( type = obj.value_type, value = value )
			# scope-like terminal (Module, RCClass, etc.) — look up the
			# final attribute as a value directly, without recursing into
			# _lower_expr (which would fail for `sys` when the base is a
			# Module, since a Module is not a value expression). Covers
			# `sys.stdout`, `sys.free`, `builtins.int`, etc. — any
			# module-level Variable/Function/RCClass reached by dotted name.
			names = getattr( obj, 'names', None )
			if isinstance( names, dict ):
				name_obj = names.get( attr )
				if isinstance( name_obj, Variable ):
					self._ensure_resolved( name_obj )
					return name_obj
		obj = self._lower_expr( node.value, None )
		attr_var = self._attr_lookup( obj.type, node.attr, node )
		dest = self._new_temp( attr_var.type )
		self._emit( ir.GetAttr( dest = dest, obj = obj, attr = node.attr ))
		return dest

	def find_name_recursive( self, node: ast.Attribute ) -> tuple[object,str]|None:
		''' Resolve a dotted ast.Attribute expression (builtins.OSError.
		FileNotFoundError) to the terminal scope object and the final
		attribute name. Purely walks .names dicts — no ensure_resolved,
		no scheduling, no type-resolving. The chain is assumed to already
		be fully resolved by type_resolution.py before lowering runs.

		Returns (terminal_object, last_attr) on success, None when the
		root isn't an ast.Name or isn't a registered name at all (caller
		falls through to the normal value-lowering path).

		Records a specific error via discovery.fail when an intermediate
		attr is missing so the user gets a clear message rather than the
		generic "not a value" from the value-lowering fallthrough. '''
		# collect attrs right-to-left: builtins.OSError.FileNotFoundError → ['FileNotFoundError', 'OSError']
		attrs: list[str] = []
		cur: ast.expr = node
		while isinstance( cur, ast.Attribute ):
			attrs.append( cur.attr )
			cur = cur.value
		if not isinstance( cur, ast.Name ):
			return None
		root_name = cur.id
		obj: object|None = self.discovery.find_name_or_none( root_name )
		if obj is None:
			return None
		# only walk when the root is a scope-like object (Module, ClassLike,
		# etc.) — a local Variable or bare Function has no .names of its own
		# and should fall through to the normal value-lowering path
		if not isinstance( getattr( obj, 'names', None ), dict ):
			return None
		# walk intermediate scopes (all but the last attr) through .names
		for attr in reversed( attrs[1:] ):
			names = getattr( obj, 'names', None )
			if not isinstance( names, dict ):
				self.discovery.fail( f'{root_name} has no members, cannot look up {attr!r} ({ast.unparse(node)})', node )
				return None
			obj = names.get( attr )
			if obj is None:
				self.discovery.fail( f'{root_name} has no attribute {attr!r} ({ast.unparse(node)})', node )
				return None
		return obj, attrs[0]

	_SUBSCRIPT_ALTERNATIVES = 'call .__getitem__(...) directly and consume its Result yourself instead'

	def _expr_Subscript( self, node: ast.Subscript, expected_type: Type|None ) -> ir.Operand:
		obj = self._lower_expr( node.value, None )
		getitem_fn = self._find_method( obj.type, '__getitem__' )
		if getitem_fn is None:
			# no real __getitem__ declared (raw pointers, or any other type
			# that doesn't define subscript access as a method) - falls
			# back to the flat GetItem opcode, unconditionally
			if expected_type is None:
				self.discovery.fail( f'cannot infer the result type of {ast.unparse(node)} - no expected type available from context', node )
			# pointer subscript indices are always usize (pointer arithmetic
			# is defined in terms of the pointer's own element size, not the
			# index's runtime width) — give the index a concrete type so a
			# bare literal 0 in e.g. `ptr[0]` doesn't fail type inference
			index_type = self.discovery.get_intrinsics()['usize']
			index = self._lower_expr( node.slice, index_type )
			dest = self._new_temp( expected_type )
			self._emit( ir.GetItem( dest = dest, obj = obj, index = index ))
			return dest

		# a real __getitem__ - call it like any other method, then if it
		# returns Result[T,E] (slice.__getitem__'s own real signature, e.g.),
		# auto-consume it exactly like or_return()/checked arithmetic do:
		# `obj[i]` reads as sugar for `obj.__getitem__(i).or_return()`
		# whenever __getitem__ can fail
		self._ensure_resolved( getitem_fn )
		self.schedule( getitem_fn.return_type )
		index = self._lower_expr( node.slice, getitem_fn.parameters[0].type )
		call_dest = self._new_temp( getitem_fn.return_type )
		self._emit( ir.Call( dest = call_dest, target = getitem_fn, receiver = obj, args = [ index ], kwargs = {} ))
		return self._maybe_consume_result( node, call_dest, self._SUBSCRIPT_ALTERNATIVES )

	def _expr_Call( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		return self._lower_call( node, expected_type, want_result = True )

	def _lower_binary_operands( self, left_node: ast.expr, right_node: ast.expr, expected_type: Type|None, *, infer_right_from_left: bool = True ) -> tuple[ir.Operand,ir.Operand]:
		# shared by _expr_BinOp and _expr_Compare: a bare literal constant on
		# either side has no type of its own to offer, so the non-constant
		# side is lowered first and its own inferred type used as the
		# constant's expected_type instead. `infer_right_from_left` captures
		# the one real difference between the two callers when NEITHER side
		# is constant: _expr_BinOp still hints the right operand with the
		# left operand's own inferred type (expected_type or left.type) - but
		# _expr_Compare's expected_type is the comparison's own result type
		# (bool), unrelated to the operands, and never cross-hints one
		# operand from the other outside the constant branches above
		left_is_const = isinstance( left_node, ast.Constant )
		right_is_const = isinstance( right_node, ast.Constant )
		if left_is_const and not right_is_const:
			right = self._lower_expr( right_node, expected_type )
			left = self._lower_expr( left_node, right.type )
		elif right_is_const and not left_is_const:
			left = self._lower_expr( left_node, expected_type )
			right = self._lower_expr( right_node, left.type )
		else:
			left = self._lower_expr( left_node, expected_type )
			right_hint = ( expected_type or left.type ) if infer_right_from_left else expected_type
			right = self._lower_expr( right_node, right_hint )
		return left, right

	def _expr_BinOp( self, node: ast.BinOp, expected_type: Type|None ) -> ir.Operand:
		left, right = self._lower_binary_operands( node.left, node.right, expected_type )

		result_type = expected_type or left.type

		opcode, extra = self._arithmetic_mode[-1].GetBinOp( node )
		return self._lower_arithmetic_op( node, opcode, extra, result_type, { 'left': left, 'right': right }, 'binary' )

	def _lower_arithmetic_op( self, node: ast.AST, opcode: type|None, extra: ir.Operand|None, result_type: Type, operand_kwargs: dict, kind: str ) -> ir.Operand:
		# shared by _lower_scalar_cast/_expr_BinOp/_expr_UnaryOp - each just
		# resolves its own (opcode, extra) via the active ArithmeticMode's
		# GetCast/GetBinOp/GetUnaryOp and hands them here along with its own
		# operand shape (cast/USub take a single `operand`, BinOp takes
		# `left`/`right`). `kind` is only used for the unsupported-operator
		# message below - _lower_scalar_cast's GetCast() never actually
		# returns None (every mode defines a cast opcode), so that branch is
		# unreachable from there, but harmless to share
		if opcode is None:
			self.discovery.fail( f'unsupported {kind} operator: {ast.unparse(node)}', node )
		if not opcode.checked_error:
			# wrap/saturate, or no overflow concept at all (bitwise/Invert)
			dest = self._new_temp( result_type )
			self._emit( opcode( dest = dest, **operand_kwargs ))
			return dest
		# check mode (the default - see the class docstring): the op itself
		# produces Result[result_type,<opcode.checked_error>]. How that
		# Result gets consumed depends on `extra`: the default (extra is
		# None) uses OrReturn, mirroring Result.or_return()'s own semantics,
		# and needs somewhere for the error to propagate to; `with
		# compiler.panic_arithmetic(msg):` (extra is the lowered msg
		# operand) uses Unwrap instead, which panics immediately and so has
		# no such requirement
		result_cls, error_cls = self._type_resolver._lookup_result_and_error_types( node, opcode.checked_error )
		if extra is None:
			# validated before anything gets emitted - a mid-statement
			# failure here must not leave partial instructions behind for
			# the per-statement recovery boundary to silently keep
			self._type_resolver._require_result_return( node, result_cls, error_cls, _ALTERNATIVES_BY_ERROR[opcode.checked_error], fn = self._current_fn )
		return self._emit_checked_op( opcode, operand_kwargs, result_type, result_cls, error_cls, extra )

	def _emit_checked_op( self, opcode: type, operand_kwargs: dict, result_type: Type, result_cls: ClassLike, error_cls: ClassLike, extra: ir.Operand|None ) -> ir.Temp:
		# shared by Check-mode binops (Add/Sub/Mult/Shl/Div/Mod), USub, and
		# scalar casts - operand_kwargs is however the specific opcode names
		# its operand(s) (left/right for a binop, operand for USub/cast)
		check_type = self.discovery._get_or_create_specialization( result_cls, [ result_type, error_cls ] )
		# the emitter declares a local variable of this Result type; the
		# struct definition must exist even though the Check op's result
		# is consumed inline (OrReturn/OrJump/Unwrap) — schedule it now
		# so monomorphize_class emits it into compiler.tagged_unions
		self.schedule( check_type )
		check_dest = self._new_temp( check_type )
		self._emit( opcode( dest = check_dest, **operand_kwargs ))
		return self._consume_checked_result( check_dest, result_type, extra )

	def _consume_checked_result( self, check_dest: ir.Temp, result_type: Type, extra: ir.Operand|None ) -> ir.Temp:
		# shared by both binop (AddCheck/.../Div/Mod) and unary (NegCheck)
		# Check-mode ops - see _expr_BinOp's own comment on the OrReturn/
		# OrJump/Unwrap split
		unwrapped = self._new_temp( result_type )
		if extra is None:
			label = self._cfg.current_epilogue_label()
			if label is not None:
				self._emit( ir.OrJump( dest = unwrapped, value = check_dest, target = label, return_slot = self._return_value_var ))
			else:
				self._emit( ir.OrReturn( dest = unwrapped, value = check_dest ))
		else:
			panic_fn = self._type_resolver._resolve_sys_function( 'panic' )
			self.schedule( panic_fn )
			self._emit( ir.Unwrap( dest = unwrapped, value = check_dest, errmsg = extra, panic = panic_fn ))
		return unwrapped

	def _expr_UnaryOp( self, node: ast.UnaryOp, expected_type: Type|None ) -> ir.Operand:
		if isinstance( node.op, ast.Not ):
			operand = self._lower_expr( node.operand, expected_type )
			dest = self._new_temp( expected_type or operand.type )
			self._emit( ir.Not( dest = dest, operand = operand ))
			return dest
		operand = self._lower_expr( node.operand, expected_type )
		result_type = expected_type or operand.type

		opcode, extra = self._arithmetic_mode[-1].GetUnaryOp( node )
		return self._lower_arithmetic_op( node, opcode, extra, result_type, { 'operand': operand }, 'unary' )

	def _expr_BoolOp( self, node: ast.BoolOp, expected_type: Type|None ) -> ir.Operand:
		# short-circuit and/or: evaluate operands left to right, each into
		# the same dest temp, stopping early (jump to end) as soon as the
		# result is already decided - `and` stops on the first falsy
		# operand, `or` stops on the first truthy one. Needed by match's
		# nested pattern tests (an outer tag check AND, only if that
		# passes, an inner tag check on the payload - reading the payload
		# before confirming the outer tag would be reading the wrong
		# union member's storage)
		bool_cls = self.discovery.find_name( 'bool', node )
		is_and = isinstance( node.op, ast.And )
		end_label = self._new_label( 'booland' if is_and else 'boolor' )
		dest = self._new_temp( bool_cls )
		for i, value_node in enumerate( node.values ):
			operand = self._lower_expr( value_node, bool_cls )
			self._emit( ir.Assign( dest = dest, src = operand ))
			if i < len( node.values ) - 1:
				jump_opcode = ir.JumpIfFalse if is_and else ir.JumpIfTrue
				self._emit( jump_opcode( cond = dest, target = end_label ))
		self._emit( ir.Label( name = end_label ))
		return dest

	_CMP_OPCODES: dict[type,'ir.CmpOp'] = {
		ast.Eq: ir.CmpOp.EQ,
		ast.NotEq: ir.CmpOp.NE,
		ast.Lt: ir.CmpOp.LT,
		ast.LtE: ir.CmpOp.LE,
		ast.Gt: ir.CmpOp.GT,
		ast.GtE: ir.CmpOp.GE,
	}

	def _expr_Compare( self, node: ast.Compare, expected_type: Type|None ) -> ir.Operand:
		# ast.In/NotIn are deliberately not handled here - `in`/`not in`
		# need a real container protocol that doesn't exist yet, guessing
		# would bake in the wrong semantics. ast.Is/IsNot ARE handled (see
		# _lower_is_comparison) - identity happens to coincide with value
		# equality for every value kind this language has today
		if len( node.ops ) != 1 or len( node.comparators ) != 1:
			self.discovery.fail( f'chained comparisons are not yet supported: {ast.unparse(node)}', node )
		if isinstance( node.ops[0], ( ast.Is, ast.IsNot )):
			return self._lower_is_comparison( node, negate = isinstance( node.ops[0], ast.IsNot ))
		cmp_op = self._CMP_OPCODES.get( type( node.ops[0] ))
		if cmp_op is None:
			self.discovery.fail( f'unsupported comparison operator: {ast.unparse(node)}', node )

		right_node = node.comparators[0]
		# unlike _expr_BinOp, expected_type here is the comparison's own
		# result type (bool) - unrelated to what type the operands
		# themselves should be lowered as, so it's never passed to either
		# side, only used (below) as one operand's own type inferred from
		# the other
		left, right = self._lower_binary_operands( node.left, right_node, None, infer_right_from_left = False )

		bool_cls = self.discovery.find_name( 'bool', node )
		dest = self._new_temp( bool_cls )
		self._emit( ir.Cmp( dest = dest, op = cmp_op, left = left, right = right ))
		return dest

	def _lower_is_comparison( self, node: ast.Compare, negate: bool ) -> ir.Operand:
		# `is`/`is not` mean real Python identity - for every value kind
		# this language has today (scalars, pointers, RC handles) identity
		# coincides with value equality, so this is plain Cmp EQ/NE...
		# UNLESS one side is a bare `None` literal being compared against a
		# TaggedUnion-typed value (T|None, e.g. sys._alloc()'s
		# Ptr[u8]|None) - there, "is None" means "the active member is
		# NoneType", which needs a tag check (the same UnionStorage.get
		# machinery match statements/conditional dispatch already use), not
		# a flat Cmp against a synthesized None operand of union type
		# (which wouldn't correspond to any real runtime representation)
		bool_cls = self.discovery.find_name( 'bool', node )
		cmp_op = ir.CmpOp.NE if negate else ir.CmpOp.EQ
		left_node, right_node = node.left, node.comparators[0]
		left_is_none = isinstance( left_node, ast.Constant ) and left_node.value is None
		right_is_none = isinstance( right_node, ast.Constant ) and right_node.value is None

		if left_is_none and right_is_none:
			return ir.Const( type = bool_cls, value = not negate ) # `None is None` / `None is not None` - degenerate, but not a crash

		if left_is_none or right_is_none:
			# a TaggedUnion-typed operand (T|None) never reaches here anymore -
			# type_resolution.py's _ReferenceResolver already rewrote that
			# shape into a plain tag Eq/NotEq Compare before lowering ever
			# saw this statement (see its own visit_Compare). What's left is
			# a flat Cmp against a real None-typed Const - e.g. a raw
			# Ptr[T]|None never actually applies (still a TaggedUnion), so in
			# practice this is for whatever non-union type this language
			# ever allows a bare `is None` against
			other = self._lower_expr( right_node if left_is_none else left_node, None )
			dest = self._new_temp( bool_cls )
			self._emit( ir.Cmp( dest = dest, op = cmp_op, left = other, right = ir.Const( type = other.type, value = None ) ))
			return dest

		left = self._lower_expr( left_node, None )
		right = self._lower_expr( right_node, left.type )
		dest = self._new_temp( bool_cls )
		self._emit( ir.Cmp( dest = dest, op = cmp_op, left = left, right = right ))
		return dest

	# --- shared helpers ----------------------------------------------------------

	def _ensure_resolved( self, obj: object ) -> object:
		# moved to TypeResolver.ensure_resolved (type_resolution.py) - kept
		# here as a thin delegate since this file calls it ~15 times and the
		# behavior (resolve now + unconditionally schedule + swap a
		# Specialization for its monomorphized form) is still exactly what
		# every one of those call sites needs. See TypeResolver's own
		# docstring for why this can't wait for schedule()'s work queue.
		return self._type_resolver.ensure_resolved( obj )

	def _attr_lookup( self, owner_type: Type|None, attr: str, ctx: ast.AST ) -> Variable:
		# _ensure_resolved is the one place a Specialization gets swapped for
		# its real, substituted ClassLike - owner_type past this point is
		# never itself a Specialization, and its .names already has
		# substituted field/method entries (see monomorphize.py), so no
		# separate per-field substitution is needed here anymore
		owner_type = self._ensure_resolved( owner_type )
		if isinstance( owner_type, TaggedUnion ) and attr in ( 'tag', 'data' ) and owner_type.names.get( attr ) is None:
			# tag/data are synthesized lazily, the first time the union is
			# actually constructed or matched against (UnionStorage.get) -
			# only reachable here for a PLAIN (non-generic) union: a
			# Specialization's own monomorphize_class already triggers this
			# itself before anything reads its .names. A method reading
			# self.tag/self.data directly (e.g. Result.is_ok()) could be
			# scheduled/lowered before anything else in THIS compilation
			# ever triggers that synthesis (the work queue has no ordering
			# guarantee) - trigger it here too, lazily, the moment it's
			# actually needed
			self._union_storage.get( owner_type )
		names = getattr( owner_type, 'names', None )
		if not isinstance( names, dict ):
			self.discovery.fail( f'{owner_type!r} has no members, cannot look up {attr!r} ({ast.unparse(ctx)})', ctx )
		found = names.get( attr )
		if not isinstance( found, Variable ):
			self.discovery.fail( f'{owner_type.qualname if owner_type else "?"} has no attribute {attr!r}', ctx )
		self._ensure_resolved( found )
		return found

	def _substituted_field( self, found: Variable, owner_type: Type|None ) -> Variable:
		return self._monomorphizer.substituted_field( found, owner_type )

	def _substitute_type_params( self, t: Type|None, type_params: list[TypeVar], args: list[Type] ) -> Type|None:
		return self._monomorphizer.substitute_type_params( t, type_params, args )

	def _monomorphized_function( self, spec: Specialization ) -> Function:
		return self._monomorphizer.monomorphized_function( spec )

	def monomorphize_class( self, spec: Specialization ) -> ClassLike:
		return self._monomorphizer.monomorphize_class( spec )

	def _try_resolve_namespace( self, node: ast.expr ) -> Name|None:
		# a *silent* probe: is this expression a compile-time-resolvable
		# namespace path (a free function, or Class.staticmethod/classmethod
		# reached by class name)? Mirrors discovery.py's own
		# visit_Name/visit_Attribute (find_name + .names traversal), but
		# deliberately doesn't call self.discovery.fail() for "this base has
		# no .names" - that's an expected, normal outcome here (it means
		# _resolve_callee should fall back to receiver-based resolution, e.g.
		# `some_local.method()`), not a real error to record. A genuinely
		# undefined identifier (find_name failing outright) is still a real
		# error either way, so that's left to report/unwind normally.
		if isinstance( node, ast.Name ):
			return self.discovery.find_name( node.id, node )
		if isinstance( node, ast.Attribute ):
			base = self._try_resolve_namespace( node.value )
			if base is None:
				return None
			self._ensure_resolved( base )
			if isinstance( base, TaggedUnion ):
				# a union member's own constructor (Foo.Bar, the synthesized
				# @staticmethod - see union_storage.py's _build_member_
				# constructor) lives in base.names, same as tag/data - but
				# unlike an ordinary class's real methods (already in .names
				# from discovery.py's own class-body parsing), nothing
				# guarantees UnionStorage.get() has actually run yet by the
				# time a namespace path reaches here (same ordering hazard
				# _attr_lookup's own tag/data trigger defends against below)
				self._union_storage.get( base )
			names = getattr( base, 'names', None )
			if not isinstance( names, dict ):
				return None
			return names.get( node.attr )
		if isinstance( node, ast.Subscript ):
			# Name[T](...) / Attribute[T](...) - explicit generic
			# instantiation of a *function* (sys.alloc[u8]), which
			# monomorphizes (a distinct compiled unit per instantiation -
			# see _monomorphized_function) rather than the anonymous-union
			# runtime-tag-checkable approach _get_or_create_union uses for
			# X|Y. Only meaningful when the base is itself a generic
			# Function; a generic CLASS reached this way (Foo[i32], used as
			# a type annotation, not a call) is handled entirely by
			# discovery.py's own visit_Subscript instead - this method is
			# lowering-only namespace-path resolution
			base = self._try_resolve_namespace( node.value )
			if not isinstance( base, Function ) or not base.type_params:
				return None
			# resolve (populate .parameters/.type_params), but deliberately
			# NOT via _ensure_resolved - that also unconditionally
			# schedules its argument, which would incorrectly compile the
			# shared, unspecialized base function too (T never gets bound
			# there - see _monomorphized_function). Only the Specialization
			# this returns gets scheduled, by the caller (_lower_call)
			if base.resolve is not None:
				base.resolve()
			arg_nodes = node.slice.elts if isinstance( node.slice, ast.Tuple ) else [ node.slice ]
			if len( arg_nodes ) != len( base.type_params ):
				self.discovery.fail(
					f'{base.qualname}[...] expects {len(base.type_params)} type argument(s), got {len(arg_nodes)}: {ast.unparse(node)}',
					node,
				)
			args: list[Type] = []
			for a in arg_nodes:
				resolved = self._try_resolve_namespace( a )
				if not isinstance( resolved, Type ):
					self.discovery.fail( f'{base.qualname}[...] argument is not a type: {ast.unparse(a)}', node )
				args.append( resolved )
			return self.discovery._get_or_create_specialization( base, args )
		return None

	def _resolve_callee( self, func_node: ast.expr ) -> tuple[Function|Overload|Specialization|_ReceiverDispatch,ir.Operand|None]:
		namespace_result = self._try_resolve_namespace( func_node )
		if isinstance( namespace_result, ( Function, Overload )):
			return namespace_result, None
		if isinstance( namespace_result, Specialization ) and isinstance( namespace_result.base, Function ):
			return namespace_result, None

		if not isinstance( func_node, ast.Attribute ):
			self.discovery.fail( f'cannot call {ast.unparse(func_node)}', func_node )
		receiver = self._lower_expr( func_node.value, None )
		shape = self._type_resolver._tagged_union_shape( receiver.type )
		if shape is not None:
			base, members = shape
			self._ensure_resolved( base )
			direct = base.names.get( func_node.attr )
			if not isinstance( direct, ( Function, Overload )):
				return self._resolve_union_receiver_members( base, members, func_node.attr, func_node ), receiver
		target = self._attr_lookup_callable( receiver.type, func_node.attr, func_node )
		return target, receiver

	def _resolve_union_receiver_members( self, union: TaggedUnion, members: list[Variable], attr: str, ctx: ast.AST ) -> _ReceiverDispatch:
		# the union itself has no .names entry for attr (an anonymous X|Y
		# union never does; a real @union class only reaches here if it
		# doesn't declare attr as a real method of its own) - so each leaf
		# type's own, unrelated method under this name has to be looked up
		# individually instead, then dispatched on the receiver's runtime tag.
		# `members` is already substituted against the receiver's own
		# concrete args when it's a generic union (see _tagged_union_shape) -
		# a generic leaf's declared type (e.g. `Some: T`) is a bare TypeVar
		# otherwise, which has no attribute lookup of its own to speak of
		per_leaf: list[tuple[Variable,Function]] = []
		for member in members:
			if member.resolve is not None:
				member.resolve()
			found = self._attr_lookup_callable( member.type, attr, ctx )
			if not isinstance( found, Function ):
				self.discovery.fail(
					f'{member.type.qualname if member.type else "?"}.{attr} is an overload group - calling an overloaded '
					f'method through a union receiver is not supported yet: {ast.unparse(ctx)}',
					ctx,
				)
			self._ensure_resolved( found ) # need .return_type/.parameters populated for the signature-consistency check just below
			per_leaf.append(( member, found ))

		reference = per_leaf[0][1]
		for member, fn in per_leaf[1:]:
			if fn.return_type is not reference.return_type:
				self.discovery.fail(
					f'{union.qualname}.{attr}(...): leaf implementations disagree on return type '
					f'({reference.cls.qualname if reference.cls else "?"}.{attr} -> '
					f'{reference.return_type.qualname if reference.return_type else "None"}, '
					f'{member.type.qualname if member.type else "?"}.{attr} -> '
					f'{fn.return_type.qualname if fn.return_type else "None"})',
					ctx,
				)
			if len( fn.parameters or [] ) != len( reference.parameters or [] ):
				self.discovery.fail(
					f'{union.qualname}.{attr}(...): leaf implementations have differing parameter counts, not supported yet',
					ctx,
				)
		return _ReceiverDispatch( union = union, attr = attr, per_leaf = per_leaf )

	def _attr_lookup_callable( self, owner_type: Type|None, attr: str, ctx: ast.AST ) -> Function|Overload:
		owner_type = self._ensure_resolved( owner_type ) # a Specialization owner is swapped for its real, substituted ClassLike/Function here
		names = getattr( owner_type, 'names', None )
		if not isinstance( names, dict ):
			self.discovery.fail( f'{owner_type!r} has no members, cannot look up {attr!r} ({ast.unparse(ctx)})', ctx )
		found = names.get( attr )
		if not isinstance( found, ( Function, Overload )):
			self.discovery.fail( f'{attr!r} is not callable on {owner_type.qualname if owner_type else "?"}', ctx )
		return found

	def _match_call_args( self, target: Function, call: ast.Call ) -> tuple[list[tuple[Parameter,ast.expr]],list[tuple[Parameter,ast.expr]]]:
		if any( isinstance( a, ast.Starred ) for a in call.args ):
			self.discovery.fail( f'*args not supported yet: {ast.unparse(call)}', call )
		if any( kw.arg is None for kw in call.keywords ):
			self.discovery.fail( f'**kwargs not supported yet: {ast.unparse(call)}', call )
		positional_params = [ p for p in target.parameters if not p.is_vararg and not p.is_kwarg and not p.is_kwonly ]
		if len( call.args ) > len( positional_params ):
			self.discovery.fail( f'too many positional arguments: {ast.unparse(call)}', call )
		positional = list( zip( positional_params, call.args ))
		keyword: list[tuple[Parameter,ast.expr]] = []
		for kw in call.keywords:
			param = next(( p for p in target.parameters if p.stem == kw.arg and not p.is_vararg and not p.is_kwarg ), None )
			if param is None:
				self.discovery.fail( f'{target.qualname} has no parameter {kw.arg!r}', call )
			keyword.append(( param, kw.value ))
		positional = [ ( param, self._check_move_argument( target, param, expr, call )) for param, expr in positional ]
		keyword = [ ( param, self._check_move_argument( target, param, expr, call )) for param, expr in keyword ]
		return positional, keyword

	def _check_move_argument( self, target: Function, param: Parameter, expr: ast.expr, call: ast.Call ) -> ast.expr:
		# both sides of a move[T] parameter must agree, checked here (once,
		# for every _match_call_args caller - plain calls, generic calls,
		# both explicit-subscript and inferred) rather than downstream:
		# move(x) and plain x lower to an identical Operand once past this
		# point, so this is the only place that can still tell them apart.
		# Unwraps a valid move(expr) down to expr - callers only ever see
		# the real argument expression from here on
		is_move_call = isinstance( expr, ast.Call ) and isinstance( expr.func, ast.Name ) and expr.func.id == 'move'
		if isinstance( param.type, Move ):
			if not is_move_call:
				self.discovery.fail(
					f"{target.qualname}: parameter {param.stem!r} is move[{param.type.inner.qualname}] - "
					f"call site must pass move({ast.unparse(expr)}): {ast.unparse(call)}",
					call,
				)
			if len( expr.args ) != 1 or expr.keywords:
				self.discovery.fail( f'move(...) takes exactly one argument: {ast.unparse(expr)}', call )
			return expr.args[0]
		if is_move_call:
			self.discovery.fail(
				f"{target.qualname}: parameter {param.stem!r} is not move[T] - "
				f"call site must not wrap it in move(...): {ast.unparse(call)}",
				call,
			)
		return expr

	def _apply_move_hook( self, param: Parameter, operand: ir.Operand, target_qualname: str ) -> None:
		# the semantic half of move[T] - _check_move_argument (run earlier,
		# inside _match_call_args) already validated the call-site syntax
		# agrees; this is where the argument's OWN ownership state actually
		# transitions, once its real Operand exists (needs the lowered
		# value, not just the AST expr) - shared by every _match_call_args
		# caller (plain calls, both generic call flavors, union-receiver
		# dispatch), called right after each argument is lowered
		if isinstance( param.type, Move ):
			for instr in self._cfg.move( operand, target_qualname = target_qualname, param_stem = param.stem ):
				self._emit( instr )

	# stems of intrinsic types a Python literal of this exact type could
	# plausibly be lowered as - deliberately coarse (no int-range/value
	# validation exists anywhere yet, see _expr_Constant), just enough to
	# rule out a string literal matching an i32 parameter and vice versa.
	# `type(value) is X`, not isinstance - bool is an int subclass in
	# Python, and ast.Constant.value is only ever bool|int|str|bytes|None
	_LITERAL_COMPATIBLE_STEMS: dict[type,tuple[str,...]] = {
		bool: ( 'bool', ),
		int: ( 'i8', 'u8', 'i16', 'u16', 'i32', 'u32', 'i64', 'u64', 'i128', 'u128', 'isize', 'usize' ),
		str: ( 'str', ),
		bytes: ( 'bytes', ),
	}

	def _lower_overload_arg( self, expr: ast.expr, position: int|None, kw_name: str|None, candidates: list[Function], node: ast.AST ) -> ir.Operand:
		if not isinstance( expr, ast.Constant ):
			return self._lower_expr( expr, None )
		compatible_stems = self._LITERAL_COMPATIBLE_STEMS.get( type( expr.value ) )
		if compatible_stems is None:
			return self._lower_expr( expr, None ) # a None literal, or something else - falls through to _expr_Constant's own "cannot infer" error, same as before

		candidate_types: list[Type] = []
		for fn in candidates:
			if fn.parameters is None:
				continue
			param = (
				fn.parameters[position] if position is not None and position < len( fn.parameters ) else
				next( ( p for p in fn.parameters if p.stem == kw_name ), None )
			)
			if param is None or param.type is None or getattr( param.type, 'stem', None ) not in compatible_stems:
				continue
			if not any( t is param.type for t in candidate_types ):
				candidate_types.append( param.type )

		if len( candidate_types ) == 1:
			return self._lower_expr( expr, candidate_types[0] )
		if len( candidate_types ) > 1:
			self.discovery.fail(
				f'ambiguous literal argument {ast.unparse(expr)} - matches more than one overload candidate type '
				f'({", ".join( t.qualname for t in candidate_types )}): {ast.unparse(node)}',
				node,
			)
		return self._lower_expr( expr, None ) # no candidate's parameter type is even plausible for this literal's kind - falls through to the existing error

	def _lower_allocate_fields( self, target_cls: ClassLike, node: ast.Call, expected_type: Type|None, label: str ) -> ir.Temp:
		# shared by both callers of ir.Allocate (Class.__allocate__(...) and
		# bare ClassName(...) sugar for the no-__init__ case) - everything
		# past "which class, and is this call form even allowed here" is
		# identical field-matching/emission logic. `label` is just how the
		# call reads in error messages (".__allocate__(...)" vs "(...)"), so
		# existing callers' error text doesn't change.
		if node.args:
			self.discovery.fail( f'{target_cls.qualname}{label} takes keyword arguments only: {ast.unparse(node)}', node )
		if any( kw.arg is None for kw in node.keywords ):
			self.discovery.fail( f'**kwargs not supported for {target_cls.qualname}{label}: {ast.unparse(node)}', node )

		self._ensure_resolved( target_cls )
		for attr in target_cls.attributes:
			self._ensure_resolved( attr ) # each field's own .type is lazily resolved, separate from the class itself - same as _attr_lookup's found.resolve
		# target_cls is always the ABSTRACT class (resolved via
		# _try_resolve_namespace on the shared, unspecialized AST body's
		# own `SomeGeneric.__allocate__` reference - see
		# _try_lower_allocate_call) even from inside a monomorphized
		# generic-class method, whose self._current_fn.cls IS the concrete
		# specialization - substitute target_cls's own field types against
		# that concrete specialization when one's available, same as
		# _substituted_field already does for ordinary attribute reads
		# (_expr_Attribute), or a generic field's expected type here would
		# stay abstract (its own bare TypeVars) forever
		fn_cls = self._current_fn.cls if self._current_fn is not None else None
		if isinstance( target_cls, TaggedUnion ):
			# a TaggedUnion's REAL storage shape is tag+data (synthesized by
			# UnionStorage.get(), registered in .names) - .attributes is the
			# LOGICAL member list (Ok/Err), a completely different thing
			# (used for .leaves()/type-matching, never for real field
			# layout). .__allocate__(tag=.., data=..) - the shape the
			# synthesized per-member constructor's own body uses (see
			# union_storage.py's _build_member_constructor) - must validate
			# against tag/data, not the logical members
			if isinstance( fn_cls, Specialization ) and fn_cls.base is target_cls:
				# target_cls itself is always the ABSTRACT union (see the
				# comment above on why bodies always reference it that way),
				# so target_cls.names['data'].type would stay the ABSTRACT,
				# unsubstituted payload_cls forever - substitute_field can't
				# help here (a payload_cls is a plain CUnion, not a TypeVar/
				# Specialization/anonymous-union shape it knows how to
				# rebuild), so read tag/data from the MONOMORPHIZED copy
				# instead (_ensure_resolved(fn_cls) is exactly monomorphize_
				# class - see its own "fresh payload_cls per specialization"
				# comment for why that copy's own data field is already
				# correctly substituted)
				concrete_union = self._ensure_resolved( fn_cls )
				tag_field = concrete_union.names.get( 'tag' )
				data_field = concrete_union.names.get( 'data' )
			else:
				tag_field = target_cls.names.get( 'tag' )
				data_field = target_cls.names.get( 'data' )
			assert isinstance( tag_field, Variable ) and isinstance( data_field, Variable ), \
				f'{target_cls.qualname}: UnionStorage.get() has not run yet - no real tag/data storage to allocate'
			declared = { tag_field.stem: tag_field, data_field.stem: data_field }
		elif isinstance( fn_cls, Specialization ) and fn_cls.base is target_cls:
			declared = { attr.stem: self._substituted_field( attr, fn_cls ) for attr in target_cls.attributes }
		else:
			declared = { attr.stem: attr for attr in target_cls.attributes }
		given = { kw.arg for kw in node.keywords }
		missing = declared.keys() - given
		if isinstance( target_cls, CUnion ):
			# a union's whole point - only ONE member is ever meaningfully
			# set at a time (see UnionStorage.get's identical comment
			# on the synthesized TaggedUnion payload CUnion) - "every OTHER
			# field is missing" isn't an error here the way it is for an
			# ordinary struct/class, unlike ResultPayload(ok=val) never
			# giving err. Pre-existing gap, confirmed unrelated to this
			# pass (reproduces on a clean checkout: Result.Ok(...)/
			# Result.Err(...)'s own bodies were never actually exercised
			# through a full Compiler.run() before, so this went unnoticed)
			if len( given ) != 1:
				self.discovery.fail( f'{target_cls.qualname}{label} takes exactly one field (only one union member is ever set): {ast.unparse(node)}', node )
		else:
			truly_missing = sorted( name for name in missing if declared[name].init is None )
			if truly_missing:
				self.discovery.fail( f'{target_cls.qualname}{label} is missing field(s): {", ".join(truly_missing)}', node )
		extra = given - declared.keys()
		if extra:
			self.discovery.fail( f'{target_cls.qualname}{label} has no field(s): {", ".join(sorted(extra))}', node )

		given_by_name = { kw.arg: kw.value for kw in node.keywords }
		# a CUnion only ever builds the ONE given member - the other
		# declared fields aren't "defaulted", they're simply not part of
		# this particular construction at all (unlike an ordinary struct/
		# class, where every field always exists)
		fields_to_build = { name: declared[name] for name in given_by_name } if isinstance( target_cls, CUnion ) else declared
		fields: dict[str,ir.Operand] = {}
		for name, field in fields_to_build.items():
			if name in given_by_name:
				expr = given_by_name[name]
				value = self._lower_expr( expr, field.type )
			else:
				# omitted at the call site, but declared with a default
				# (`field.init`, already confirmed not None by truly_missing
				# above) - lowered in the CLASS's own scope, not the caller's,
				# matching ordinary Python class-body scoping (a default
				# expression can reference other class-level names, but not
				# anything local to whoever's constructing this instance)
				expr = field.init
				with self.discovery.module_context( self._find_module_for( target_cls )):
					with self.discovery.scope_context( target_cls ):
						value = self._lower_expr( expr, field.type )
			# value.type, not field.type: field.type is the FIELD's declared
			# type, which stays an unsubstituted TypeVar for any field whose
			# type depends on a class's own type params (class methods are
			# never monomorphized per-specialization - see compiler.py's
			# _enqueue) - value.type is always the operand's real, concrete
			# type regardless, since only concrete values ever actually get
			# lowered
			for instr in self._cfg.field_value( value.type, value, is_alias = self._is_aliasing_expr( expr )):
				self._emit( instr )
			fields[name] = value

		dest = self._new_temp( expected_type or target_cls )
		# dest.type can be a concrete Specialization (ResultPayload[i32,
		# OverflowError], inferred from the substituted field type this
		# construction call is being assigned into - see the field.type
		# comment above) even though target_cls itself (this call's own
		# bare `ResultPayload` reference) is always the abstract base - the
		# concrete Specialization needs its own explicit schedule() here,
		# same as _emit_generic_call already does for a generic FUNCTION's
		# own monomorphized return type; nothing else would ever schedule it
		self.schedule( dest.type )
		if isinstance( target_cls, RCClass ):
			self._schedule_rcclass_construction( target_cls, dest.type )
		self._emit( ir.Allocate( dest = dest, cls = target_cls, fields = fields ))
		return dest

	def _schedule_rcclass_construction( self, target_cls: RCClass, concrete_type: Type ) -> None:
		# guarantees sys.alloc[concrete_type] is a real, lowered compile unit
		# by the time the emitter sees the resulting ir.Allocate - the
		# emitter independently synthesizes the call to it (mangled
		# qualname, same convention as everything else), so this has to
		# actually exist regardless of whether the user's own program ever
		# wrote `import sys` (same posture as _resolve_sys_function's own
		# doc). Shared by _lower_allocate_fields (the no-__init__/field=value
		# path) and _try_lower_construct_call (the real __init__ path) -
		# both eventually emit an ir.Allocate for a real RCClass and need
		# identical scheduling
		sys_alloc_fn = self._type_resolver._resolve_sys_function( 'alloc' )
		alloc_spec = self.discovery._get_or_create_specialization( sys_alloc_fn, [ concrete_type ])
		self.schedule( alloc_spec )
		# every constructed RCClass needs its own destructor eventually
		# synthesized by the emitter (emit_c walks compiler.rcclasses, one
		# destructor function per entry - see emitter_c.py's Phase 4 work) -
		# that destructor calls sys.free on the object's own backing memory
		# and, if the class declares one, the user's own __del__ - both need
		# to already be real, lowered compile units by the time the emitter
		# needs to call them. Triggered at CONSTRUCTION time (same as
		# sys.alloc above), not merely when the class is referenced as a
		# type - scheduling this for every bare type annotation would drag
		# in sys.free's own transitive dependencies (real HeapFree/crt free
		# externs) for classes that are never actually instantiated
		sys_free_fn = self._type_resolver._resolve_sys_function( 'free' )
		self.schedule( sys_free_fn )
		del_fn = target_cls.get_local( '__del__' ) # target_cls is always the abstract base - methods aren't re-specialized per Specialization (Specialization.names passes through to .base.names)
		if isinstance( del_fn, Function ):
			self.schedule( del_fn )

	def _try_lower_allocate_call( self, node: ast.Call, expected_type: Type|None ) -> ir.Temp|None:
		# Class.__allocate__(field=value, ...) - a compiler-synthesized
		# pseudo-method (SYNTAX.md: "strictly private... can only be called
		# from methods inside the same class"), not a real declared method,
		# so it can never be found via the ordinary _resolve_callee/
		# _attr_lookup_callable path (it's never in any class's .names) -
		# recognized textually here instead, same spirit as defer/errdefer
		# and the arithmetic-mode with-blocks. Returns None (not an error)
		# when this doesn't look like a __allocate__ call at all, so the
		# caller falls through to the normal call path and reports whatever
		# error is actually appropriate (e.g. "not callable")
		if not ( isinstance( node.func, ast.Attribute ) and node.func.attr == '__allocate__' ):
			return None
		target_cls = self._try_resolve_namespace( node.func.value )
		if not isinstance( target_cls, ClassLike ):
			return None

		fn = self._current_fn
		# fn.cls is the CONCRETE class specialization for a monomorphized
		# generic-class method (Result.Ok's own fn.cls is Result[i32,E],
		# not bare Result - see _monomorphized_function's substituted_cls),
		# while target_cls (resolved from the shared, unspecialized AST
		# body's own `Result.__allocate__` reference) is always the
		# abstract base - compare against fn.cls's own base in that case
		fn_cls = fn.cls if fn is not None else None
		fn_base_cls = fn_cls.base if isinstance( fn_cls, Specialization ) else fn_cls
		if fn is None or fn_base_cls is not target_cls:
			self.discovery.fail(
				f'{target_cls.qualname}.__allocate__(...) is private - only callable from a method of {target_cls.qualname} itself',
				node,
			)
		return self._lower_allocate_fields( target_cls, node, expected_type, '.__allocate__(...)' )

	def _try_lower_construct_call( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand|None:
		# bare ClassName(...) - SYNTAX.md sugar. A class with no __init__ at
		# all degrades to exactly __allocate__ (field=value sugar). A class
		# WITH __init__: allocate self uninitialized, call __init__ with
		# the call site's own arguments (__init__'s OWN parameter list, NOT
		# the field=value sugar), wrap the result in Result[Foo,E] when
		# __init__ is fallible, dropping self's own refcount (but not
		# calling __del__) on Err - see RCCLASS ATTRIBUTE LIFETIME.md and
		# the approved plan. Scoped to non-subclassed RCClasses only,
		# matching lower_function's own scope check for __init__ itself.
		target_cls = self._try_resolve_namespace( node.func )

		# CEnum construction: EnumName(value) is a plain cast to the
		# enum's underlying type — no allocation, no refcounting, just
		# reinterpret the raw integer as the enum type. e.g. OSError(ENOENT)
		if isinstance( target_cls, CEnum ):
			if len( node.args ) != 1 or node.keywords:
				self.discovery.fail( f'{target_cls.qualname}(...) takes exactly one positional argument: {ast.unparse(node)}', node )
			self._ensure_resolved( target_cls )
			# lower the argument directly — no arithmetic-mode semantics
			# needed here; a CEnum has exactly the same runtime
			# representation as its underlying type, so OSError(42) is
			# just the value 42 with the enum type
			return self._lower_expr( node.args[0], target_cls )

		if not isinstance( target_cls, ClassLike ):
			return None
		# resolve (populate .names/.attributes) WITHOUT scheduling yet - a
		# generic target_cls must never itself become a real compile unit
		# (see below); only a concrete Specialization should
		if target_cls.resolve is not None:
			target_cls.resolve()
		init = target_cls.names.get( '__init__' )
		if init is None:
			return self._lower_allocate_fields( target_cls, node, expected_type, '(...)' )
		if not isinstance( target_cls, RCClass ):
			# __init__ on a @cstruct/@cunion/@enum - not supported yet
			# (attribute lifetime tracking is scoped to RCClass, matching
			# RCCLASS ATTRIBUTE LIFETIME.md's own title) - falls through to
			# the normal call path, same "not callable" as always
			return None
		if not isinstance( init, Function ):
			self.discovery.fail( f'{target_cls.qualname}.__init__ is overloaded - not supported yet: {ast.unparse(node)}', node )
		if target_cls.base is not None:
			self.discovery.fail(
				f'{target_cls.qualname}(...): __init__ invocation is only supported for classes with no base class yet: {ast.unparse(node)}',
				node,
			)

		if target_cls.type_params:
			self_type, init, args, kwargs = self._lower_generic_construction_args( node, target_cls, init, expected_type )
		else:
			self.schedule( target_cls )
			self._ensure_resolved( init )
			self_type = target_cls
			args, kwargs = self._lower_call_args( init, node )

		# self_temp.type is self_type (target_cls itself, or the
		# Specialization for a generic construction) - NEVER a bare
		# monomorphized ClassLike object floating free of any Specialization
		# wrapper, or scheduling it again anywhere else (e.g. sys.alloc[T]'s
		# own substituted return type, exactly T) would register it a
		# second time outside the Specialization dedup path. ir.Allocate's
		# own `cls`, unlike self_temp.type, is always the ABSTRACT target_cls
		# regardless - the emitter only uses it for an RCClass-vs-not check,
		# never to read field layout (real field VALUES are already in
		# `fields`, and the mangled alloc name comes from dest.type, not cls
		# - see emitter_c.py's own ir.Allocate handling)
		self_temp = self._new_temp( self_type )
		self._schedule_rcclass_construction( target_cls, self_temp.type )
		self._emit( ir.Allocate( dest = self_temp, cls = target_cls, fields = {} ))

		self.schedule( init.return_type )
		for param in init.parameters or []:
			self.schedule( param.type )

		if not self._init_fallibility( init ):
			self._emit( ir.Call( dest = None, target = init, receiver = self_temp, args = args, kwargs = kwargs ))
			return self_temp
		return self._emit_fallible_construction( node, self_type, init, self_temp, args, kwargs, expected_type )

	def _lower_generic_construction_args( self, node: ast.Call, target_cls: RCClass, init: Function, expected_type: Type|None ) -> tuple[RCClass|Specialization,Function,list[ir.Operand],dict[str,ir.Operand]]:
		# Box(...) where Box is generic: target_cls's own concrete type args
		# have to be pinned down before __init__ can be called - same two-
		# phase strategy _lower_class_generic_method_call's own inference
		# branch uses (unify from expected_type first, then refine from the
		# lowered arguments' own types), since a class constructor's type
		# params are exactly as inferable as a generic method's - working
		# against __init__'s ABSTRACT parameter list throughout (substituting
		# per-parameter via _substitute_type_params) because the concrete,
		# monomorphized __init__ isn't available until the args are already
		# resolved
		if init.resolve is not None:
			init.resolve()
		class_type_params = target_cls.type_params or []
		bindings: dict[int,Type] = {}
		# expected_type pins target_cls's own args directly for a non-
		# fallible __init__ (b: Box[i32] = Box(1)) - but for a FALLIBLE one,
		# Box(...) itself becomes Result[Box[i32],E] (SYNTAX.md), so the
		# surrounding annotation is r: Result[Box[i32],MyError], one level
		# removed from target_cls. Peek through a Result[_,_] wrapper via
		# _result_shape, which speculatively no-ops (rather than hard-
		# failing) when this particular construction isn't Result-shaped
		# at all, same posture as _maybe_consume_result
		pinning_type = expected_type
		shape = self._type_resolver._result_shape( expected_type )
		if shape is not None:
			pinning_type = shape[0]
		if isinstance( pinning_type, Specialization ) and pinning_type.base is target_cls:
			for tv, arg in zip( class_type_params, pinning_type.args ):
				bindings[ id( tv ) ] = arg

		positional, keyword = self._match_call_args( init, node )
		partial_args = [ bindings.get( id( tv ), tv ) for tv in class_type_params ]
		args = [
			self._lower_expr( expr, self._substitute_type_params( param.type, class_type_params, partial_args ))
			for param, expr in positional
		]
		kwargs = {
			param.stem: self._lower_expr( expr, self._substitute_type_params( param.type, class_type_params, partial_args ))
			for param, expr in keyword
		}
		for ( param, _expr ), operand in zip( positional, args ):
			self._apply_move_hook( param, operand, init.qualname )
		for param, _expr in keyword:
			self._apply_move_hook( param, kwargs[param.stem], init.qualname )

		for ( param, _expr ), operand in zip( positional, args ):
			self._unify_type_param( class_type_params, param.type, operand.type, bindings, node, target_cls.qualname )
		for param, _expr in keyword:
			self._unify_type_param( class_type_params, param.type, kwargs[param.stem].type, bindings, node, target_cls.qualname )

		missing = [ tv.stem for tv in class_type_params if id( tv ) not in bindings ]
		if missing:
			self.discovery.fail(
				f'{target_cls.qualname}(...): cannot infer type parameter(s) {", ".join(missing)} from these arguments or the surrounding expected type: {ast.unparse(node)}',
				node,
			)
		concrete_args = [ bindings[id(tv)] for tv in class_type_params ]
		cls_spec = self.discovery._get_or_create_specialization( target_cls, concrete_args )
		init_spec = self.discovery._get_or_create_specialization( init, concrete_args )
		self._ensure_resolved( cls_spec ) # also populates init_spec.monomorphized as a side effect - same (init, concrete_args) key monomorphize_class's own method-substitution loop uses
		monomorphized_init = self._ensure_resolved( init_spec )
		return cls_spec, monomorphized_init, args, kwargs

	def _emit_fallible_construction(
		self, node: ast.Call, concrete_cls: RCClass|Specialization, init: Function, self_temp: ir.Temp,
		args: list[ir.Operand], kwargs: dict[str,ir.Operand], expected_type: Type|None,
	) -> ir.Operand:
		# __init__ is fallible (Result[None,E]) - Foo(...) becomes
		# Result[Foo,E] (SYNTAX.md). The actual Ok/Err wrapping reuses REAL
		# Result.Ok/Result.Err call-lowering (via synthesized AST
		# referencing hidden locals - _declare_hidden_local, the same
		# technique the for-loop scaffolding already uses) rather than
		# hand-building ResultPayload's own internal shape here - only the
		# branch structure itself (and self_temp's own decref on Err, not
		# expressible as source syntax) is raw IR, mirroring
		# _lower_conditional_dispatch's own style
		init_result = self._new_temp( init.return_type )
		self._emit( ir.Call( dest = init_result, target = init, receiver = self_temp, args = args, kwargs = kwargs ))

		unique = self._label_id
		self_var = self._declare_hidden_local( f'__ctor_self_{unique}', concrete_cls, node )
		for instr in self._cfg.assign( self_var, self_temp, is_alias = False ):
			self._emit( instr )
		self._emit( ir.Assign( dest = self_var, src = self_temp ))

		result_var = self._declare_hidden_local( f'__ctor_result_{unique}', init.return_type, node )
		for instr in self._cfg.assign( result_var, init_result, is_alias = False ):
			self._emit( instr )
		self._emit( ir.Assign( dest = result_var, src = init_result ))

		error_cls = init.return_type.args[1]
		result_cls = self.discovery.find_name( 'Result', node )
		outer_result_type = expected_type or self.discovery._get_or_create_specialization( result_cls, [ concrete_cls, error_cls ] )
		dest_var = self._declare_hidden_local( f'__ctor_dest_{unique}', outer_result_type, node )

		is_err_fn = self._attr_lookup_callable( init.return_type, 'is_err', node )
		self._ensure_resolved( is_err_fn )
		bool_cls = self.discovery.find_name( 'bool', node )
		is_err_temp = self._new_temp( bool_cls )
		self._emit( ir.Call( dest = is_err_temp, target = is_err_fn, receiver = result_var, args = [], kwargs = {} ))

		err_label = self._new_label( 'ctor_err' )
		end_label = self._new_label( 'ctor_end' )
		self._emit( ir.JumpIfFalse( cond = is_err_temp, target = err_label ))

		# Ok branch: self is fully constructed - hand it off
		ok_expr = ast.Call(
			func = ast.Attribute( value = ast.Name( id = 'Result', ctx = ast.Load() ), attr = 'Ok', ctx = ast.Load() ),
			args = [ ast.Name( id = self_var.stem, ctx = ast.Load() ) ], keywords = [],
		)
		ast.copy_location( ok_expr, node )
		ok_value = self._lower_expr( ok_expr, outer_result_type )
		for instr in self._cfg.assign( dest_var, ok_value, is_alias = False ):
			self._emit( instr )
		self._emit( ir.Assign( dest = dest_var, src = ok_value ))
		self._emit( ir.Jump( target = end_label ))

		# Err branch: self never became valid - drop its own refcount
		# (but __del__ is never invoked on it - SYNTAX.md), propagate the
		# same error, re-wrapped for THIS construction's own Result[Foo,E]
		self._emit( ir.Label( name = err_label ))
		self._emit( ir.Decref( value = self_var ))
		err_expr = ast.Call(
			func = ast.Attribute( value = ast.Name( id = 'Result', ctx = ast.Load() ), attr = 'Err', ctx = ast.Load() ),
			args = [ ast.Attribute(
				value = ast.Attribute( value = ast.Name( id = result_var.stem, ctx = ast.Load() ), attr = 'data', ctx = ast.Load() ),
				attr = 'v_Err', ctx = ast.Load(),
			) ], keywords = [],
		)
		ast.copy_location( err_expr, node )
		err_value = self._lower_expr( err_expr, outer_result_type )
		for instr in self._cfg.assign( dest_var, err_value, is_alias = False ):
			self._emit( instr )
		self._emit( ir.Assign( dest = dest_var, src = err_value ))
		self._emit( ir.Label( name = end_label ))
		return dest_var

	def _try_lower_scalar_construct_call( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand|None:
		# ScalarName(x) - Python's own int(x)/float(x)-style constructor-as-
		# cast idiom. Deliberately NOT routed through _try_lower_construct_call
		# (ClassLike-only: its Allocate/self/RC-fallible-construction machinery
		# is meaningless for a scalar - no self to allocate, no attributes, no
		# refcounting)
		target_cls = self._try_resolve_namespace( node.func )
		if not isinstance( target_cls, Scalar ):
			return None
		if len( node.args ) != 1 or node.keywords:
			self.discovery.fail( f'{target_cls.qualname}(...) takes exactly one argument: {ast.unparse(node)}', node )
		arg_node = node.args[0]
		if isinstance( arg_node, ast.Constant ):
			return self._lower_scalar_cast( target_cls, arg_node, node )
		operand = self._lower_expr( arg_node, None )
		if isinstance( operand.type, Scalar ):
			# the real motivating case (u32(s.byte_len())) - same
			# arithmetic-mode-respecting logic compiler.cast(...) uses, no
			# dunder dispatch needed: one compiler primitive already
			# covers every Scalar-to-Scalar pair uniformly
			return self._lower_scalar_cast( target_cls, operand, node )
		# a non-Scalar source (e.g. an RCClass) - this is where library-
		# authored extensibility (Scalar.names, see discovery.py's
		# visit_Assign) actually earns its keep: a future
		# `SomeClass.__u32__(self) -> u32: ...` is dispatched here exactly
		# like any other method call
		dunder = self._find_method( operand.type, f'__{target_cls.stem}__' )
		if dunder is None:
			self.discovery.fail(
				f'{operand.type.qualname if operand.type else "?"} has no __{target_cls.stem}__ method - cannot convert to {target_cls.qualname}: {ast.unparse(node)}',
				node,
			)
		self._ensure_resolved( dunder )
		self.schedule( dunder.return_type )
		dest = self._new_temp( expected_type or dunder.return_type )
		self._emit( ir.Call( dest = dest, target = dunder, receiver = operand, args = [], kwargs = {} ))
		return dest

	_OR_RETURN_ALTERNATIVES = 'or_return() always propagates the error to the caller - there is no other way for the enclosing function to receive it'

	def _lower_or_return( self, node: ast.Call, receiver: ir.Operand, want_result: bool ) -> ir.Operand|None:
		# <result_expr>.or_return() is recognized textually here rather than
		# ever actually calling Result.or_return's own declared body
		# (`if self.is_err(): compiler.early_return(self.data.v_Err)` /
		# `return self.data.v_Ok`) - that body is written as a spec of the
		# intended behavior, not something literally compilable: it needs to
		# trigger a `return Result.Err(...)` in ITS CALLER's scope, not its
		# own (or_return's own declared return type is bare T, not
		# Result[T,E], so `return Result.Err(...)` from inside it could
		# never type-check there - see compiler.early_return's own comment).
		# This expands directly to the same OrReturn/OrJump primitives
		# checked-arithmetic already uses for exactly the same "propagate
		# the error to the enclosing function, continue with the unwrapped
		# value" shape - no new IR needed, and Result.or_return is never
		# scheduled/lowered as a real function as a result.
		if node.args or node.keywords:
			self.discovery.fail( f'or_return() takes no arguments: {ast.unparse(node)}', node )
		shape = self._type_resolver._result_shape( receiver.type )
		if shape is None:
			self.discovery.fail( f'or_return() receiver must be Result[_,_], got {receiver.type.qualname if receiver.type else "?"}', node )
		result_type, error_cls = shape
		self._type_resolver._require_result_return( node, receiver.type.base, error_cls, self._OR_RETURN_ALTERNATIVES, fn = self._current_fn )
		unwrapped = self._consume_checked_result( receiver, result_type, extra = None )
		return unwrapped if want_result else None

	def _lower_call_args( self, target: Function, node: ast.Call ) -> tuple[list[ir.Operand],dict[str,ir.Operand]]:
		# shared by the plain call path (_lower_call's own else branch) and
		# _lower_generic_function_call: lowers positional/keyword args
		# straight against target's own already-concrete declared parameter
		# types, applying each param's move hook as it goes. NOT reused by
		# _lower_inferred_generic_call or _lower_class_generic_method_call -
		# both of those still need to INFER target's type params before a
		# parameter type is concrete enough to lower an argument against (in
		# _lower_inferred_generic_call's case, args are lowered with no
		# expected type at all, and move hooks apply in a separate pass
		# afterward instead), so forcing them through this helper would
		# change what expected_type each argument actually gets
		positional, keyword = self._match_call_args( target, node )
		args = []
		for param, expr in positional:
			operand = self._lower_expr( expr, param.type )
			self._apply_move_hook( param, operand, target.qualname )
			args.append( operand )
		kwargs = {}
		for param, expr in keyword:
			operand = self._lower_expr( expr, param.type )
			self._apply_move_hook( param, operand, target.qualname )
			kwargs[param.stem] = operand
		# fill in default values for any parameter that was not
		# explicitly provided by the call site (e.g. print(msg,
		# end='\n') called as print('hello') — end gets its
		# default lowered here as if the caller had passed it)
		given = { param.stem for param, _ in positional }
		given.update( kwargs.keys() )
		for param in target.parameters or []:
			if param.stem not in given and param.default is not None:
				default_operand = self._lower_expr( param.default, param.type )
				kwargs[param.stem] = default_operand
		return args, kwargs

	def _lower_generic_function_call( self, node: ast.Call, spec: Specialization, receiver: ir.Operand|None, expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# sys.alloc[u8](...) - explicit generic instantiation. Matches call
		# args against the MONOMORPHIZED signature (so a literal argument's
		# expected type is already concrete, e.g. usize for alloc[u8]'s
		# count - not the abstract, unsubstituted one)
		monomorphized = self._monomorphized_function( spec )
		args, kwargs = self._lower_call_args( monomorphized, node )
		return self._emit_generic_call( spec, monomorphized, receiver, args, kwargs, expected_type, want_result )

	def _lower_inferred_generic_call( self, node: ast.Call, target: Function, receiver: ir.Operand|None, expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# a BARE call to a generic function (mylen(a), no explicit [T]) -
		# unlike _lower_generic_function_call, there's no already-concrete
		# Specialization to match args against yet: T has to be inferred
		# FROM the arguments themselves first. Args are lowered once with no
		# expected type (a bare TypeVar parameter can't offer a real literal
		# hint anyway - a literal argument at such a position correctly
		# fails via _expr_Constant's own "cannot infer" error, same as any
		# other call with no usable expected type), then each declared
		# parameter type is unified against that argument's real lowered
		# type (_unify_type_param) to solve for target's own type params -
		# the inverse of _substitute_type_params, which already handles
		# substituting a SOLVED binding through arbitrarily nested
		# Specializations (list[T] etc), so unification mirrors that same
		# recursive shape instead of only handling a bare `t: T` parameter
		if target.resolve is not None:
			target.resolve()
		positional, keyword = self._match_call_args( target, node )
		args = [ self._lower_expr( expr, None ) for _param, expr in positional ]
		kwargs = { param.stem: self._lower_expr( expr, None ) for param, expr in keyword }
		for ( param, _expr ), operand in zip( positional, args ):
			self._apply_move_hook( param, operand, target.qualname )
		for param, _expr in keyword:
			self._apply_move_hook( param, kwargs[param.stem], target.qualname )

		bindings: dict[int,Type] = {} # id(TypeVar) -> the concrete Type it was inferred as
		for ( param, _expr ), operand in zip( positional, args ):
			self._unify_type_param( target.type_params or [], param.type, operand.type, bindings, node, target.qualname )
		for param, _expr in keyword:
			self._unify_type_param( target.type_params or [], param.type, kwargs[param.stem].type, bindings, node, target.qualname )

		missing = [ tv.stem for tv in target.type_params or [] if id( tv ) not in bindings ]
		if missing:
			self.discovery.fail(
				f'{target.qualname}[...]: cannot infer type parameter(s) {", ".join(missing)} from these arguments - '
				f'call it explicitly as {target.qualname}[...](...) instead: {ast.unparse(node)}',
				node,
			)
		inferred_args = [ bindings[id(tv)] for tv in target.type_params or [] ]
		spec = self.discovery._get_or_create_specialization( target, inferred_args )
		monomorphized = self._monomorphized_function( spec )
		return self._emit_generic_call( spec, monomorphized, receiver, args, kwargs, expected_type, want_result )

	def _unify_type_param( self, type_params: list[TypeVar], declared: Type|None, actual: Type|None, bindings: dict[int,Type], node: ast.AST, context_qualname: str ) -> None:
		# generalized over an explicit type_params list (rather than always
		# reading target.type_params) so this same unification shared by
		# both a generic FREE function's own type params (_lower_inferred_
		# generic_call) and a generic CLASS's type params (_lower_class_
		# generic_method_call - Result.Ok/.Err reached with no receiver to
		# read a concrete Specialization's args from directly)
		if declared is None or actual is None:
			return
		if any( declared is tv for tv in type_params ):
			existing = bindings.get( id( declared ) )
			if existing is not None and existing is not actual:
				self.discovery.fail(
					f'{context_qualname}(...): type parameter {declared.stem!r} is inferred as both '
					f'{existing.qualname} and {actual.qualname} by different arguments: {ast.unparse(node)}',
					node,
				)
			bindings[ id( declared ) ] = actual
			return
		if isinstance( declared, Specialization ) and isinstance( actual, Specialization ) and declared.base is actual.base:
			for d_arg, a_arg in zip( declared.args, actual.args ):
				self._unify_type_param( type_params, d_arg, a_arg, bindings, node, context_qualname )
		# else: this parameter position doesn't mention any of type_params
		# (a concrete parameter, or a nested type whose base doesn't even
		# match the argument's) - nothing to infer here. Not an error by
		# itself: a genuine argument-type mismatch isn't checked anywhere
		# yet (no general type-checking pass exists), same as every other
		# call site in this file today

	def _emit_generic_call( self, spec: Specialization, monomorphized: Function, receiver: ir.Operand|None, args: list[ir.Operand], kwargs: dict[str,ir.Operand], expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# schedules the Specialization itself as the compile unit (see
		# _monomorphized_function/compiler.py's own handling of it), shared
		# tail for both the explicit Name[T](...) and inferred call paths
		self.schedule( spec )
		self.schedule( monomorphized.return_type )
		for param in monomorphized.parameters or []:
			self.schedule( param.type )
		if want_result:
			dest = self._new_temp( expected_type or monomorphized.return_type )
			self._emit( ir.Call( dest = dest, target = monomorphized, receiver = receiver, args = args, kwargs = kwargs ))
			return dest
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
		if target.resolve is not None:
			target.resolve()
		cls = target.cls
		class_type_params = cls.type_params or [] if cls is not None else []
		bindings: dict[int,Type] = {}
		if expected_type is not None:
			self._unify_type_param( class_type_params, target.return_type, expected_type, bindings, node, target.qualname )

		positional, keyword = self._match_call_args( target, node )
		partial_args = [ bindings.get( id( tv ), tv ) for tv in class_type_params ]
		args = [
			self._lower_expr( expr, self._substitute_type_params( param.type, class_type_params, partial_args ))
			for param, expr in positional
		]
		kwargs = {
			param.stem: self._lower_expr( expr, self._substitute_type_params( param.type, class_type_params, partial_args ))
			for param, expr in keyword
		}
		for ( param, _expr ), operand in zip( positional, args ):
			self._apply_move_hook( param, operand, target.qualname )
		for param, _expr in keyword:
			self._apply_move_hook( param, kwargs[param.stem], target.qualname )

		for ( param, _expr ), operand in zip( positional, args ):
			self._unify_type_param( class_type_params, param.type, operand.type, bindings, node, target.qualname )
		for param, _expr in keyword:
			self._unify_type_param( class_type_params, param.type, kwargs[param.stem].type, bindings, node, target.qualname )

		missing = [ tv.stem for tv in class_type_params if id( tv ) not in bindings ]
		if missing:
			self.discovery.fail(
				f'{target.qualname}(...): cannot infer {cls.qualname if cls else "?"} type parameter(s) '
				f'{", ".join(missing)} from these arguments or the surrounding expected type: {ast.unparse(node)}',
				node,
			)
		cls_args = [ bindings[id(tv)] for tv in class_type_params ]
		method_spec = self.discovery._get_or_create_specialization( target, cls_args )
		monomorphized = self._monomorphized_function( method_spec )
		return self._emit_generic_call( method_spec, monomorphized, receiver, args, kwargs, expected_type, want_result )

	def _lower_call( self, node: ast.Call, expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		match self._is_compiler_call( node ):
			case 'sizeof':
				result = self._lower_compiler_sizeof( node, expected_type )
				return result if want_result else None

			case 'refcount':
				result = self._lower_compiler_refcount( node, expected_type )
				return result if want_result else None

			case 'cast':
				result = self._lower_compiler_cast( node, expected_type )
				return result if want_result else None

			case 'addrof':
				result = self._lower_compiler_addrof( node, expected_type )
				return result if want_result else None

		# each recognizer returns None (not an error) when this call doesn't
		# match its own construction-sugar shape at all, falling through to
		# the next; a real error inside a matched shape (e.g. a malformed
		# __allocate__ call) still raises/records normally
		construction_recognizers = (
			self._try_lower_allocate_call,
			self._try_lower_construct_call,
			self._try_lower_scalar_construct_call,
		)
		for recognizer in construction_recognizers:
			allocate_dest = recognizer( node, expected_type )
			if allocate_dest is not None:
				return allocate_dest if want_result else None

		# type_resolution.py's own generic-call resolution
		# (_ReferenceResolver.visit_Call) may already have tagged this call
		# with its resolved, monomorphized callee (an ordinary, concrete
		# Function - never a Specialization) - when present, it's
		# authoritative and skips _resolve_callee/the Specialization
		# branches below entirely, so a call resolved there never touches
		# Specialization on this side at all. Absent (any call that pass
		# left untagged, generic or not) falls through to the exact same
		# resolution this always did
		resolved_callee = getattr( node, 'resolved_callee', None )
		if resolved_callee is not None:
			target, receiver = resolved_callee, None
		else:
			target, receiver = self._resolve_callee( node.func )
		if receiver is not None:
			self.schedule( receiver.type )

		if isinstance( target, _ReceiverDispatch ):
			return self._lower_union_receiver_call( node, target, receiver, expected_type, want_result )

		if isinstance( target, Function ) and target.stem == 'or_return':
			# target.cls is a Specialization, not bare Result, whenever the
			# receiver already pinned concrete args (the common case, e.g.
			# some_result.or_return() where some_result: Result[i32,E]) -
			# unwrap before the identity check, or a concrete receiver's own
			# or_return() would stop being recognized at all and fall
			# through to actually CALLING Result.or_return's literal
			# declared body, which is a spec of the intended behavior, not
			# something literally compilable (see _lower_or_return's own
			# comment)
			target_cls_base = target.cls.base if isinstance( target.cls, Specialization ) else target.cls
			if target_cls_base is self.discovery.find_name( 'Result', node ):
				return self._lower_or_return( node, receiver, want_result )

		if isinstance( target, Specialization ) and isinstance( target.base, Function ):
			return self._lower_generic_function_call( node, target, receiver, expected_type, want_result )

		if isinstance( target, Function ) and target.type_params:
			return self._lower_inferred_generic_call( node, target, receiver, expected_type, want_result )

		# target.cls can legitimately BE a Specialization now (monomorphized_
		# function sets a monomorphized method's own .cls to one) - Specialization
		# has no .type_params of its own, so this must not read it directly;
		# getattr's default (None/falsy) correctly means "not this branch",
		# since a receiver that already pinned down concrete class args (the
		# only way target.cls ends up a Specialization here) already went
		# through _attr_lookup_callable's own substitution - nothing left to
		# infer
		if isinstance( target, Function ) and not target.type_params and target.cls is not None and getattr( target.cls, 'type_params', None ):
			return self._lower_class_generic_method_call( node, target, receiver, expected_type, want_result )

		if isinstance( target, Overload ):
			# a bare literal argument has no type of its own before a
			# specific implementation is chosen - _lower_overload_arg tries
			# each candidate's declared parameter type at that position,
			# using it unambiguously if exactly one is even plausible for
			# the literal's own kind (a string literal never plausibly
			# matches an i32 parameter, etc.) and failing clearly rather
			# than guessing if more than one genuinely could
			candidates = [ *target.stubs, *target.implementations ]
			for fn in candidates:
				if fn.resolve is not None:
					fn.resolve()
			args = [ self._lower_overload_arg( a, i, None, candidates, node ) for i, a in enumerate( node.args ) ]
			if any( kw.arg is None for kw in node.keywords ):
				self.discovery.fail( f'**kwargs not supported yet: {ast.unparse(node)}', node )
			kwargs = { kw.arg: self._lower_overload_arg( kw.value, None, kw.arg, candidates, node ) for kw in node.keywords }
			arg_types = [ op.type for op in args ]
			kwarg_types = { name: op.type for name, op in kwargs.items() }

			# an @overload group declared inside a generic CLASS (e.g.
			# Result[T,E].unwrap_or's `default: T` stub) still carries the
			# class's own bare TypeVars on every member's .parameters -
			# overload_resolution.py is a pure function of types with no
			# substitution logic of its own (see its module docstring), so
			# candidate matching there would otherwise compare a REAL,
			# concrete call-site argument type (i32) against the abstract
			# TypeVar T itself and never match. Substitute member-owned
			# copies (parameters only - matching is all resolve_call needs
			# them for) whenever the receiver's own type already pins a
			# concrete specialization of this group's class, mirroring
			# _lower_class_generic_method_call's identical receiver check
			# for a single (non-overloaded) generic method.
			group_cls = candidates[0].cls if candidates else None
			group_type_params = group_cls.type_params or [] if group_cls is not None else []
			cls_args: list[Type]|None = None
			if group_type_params and receiver is not None and isinstance( receiver.type, Specialization ) and receiver.type.base is group_cls:
				cls_args = receiver.type.args

			def _for_matching( fn: Function ) -> Function:
				if cls_args is None:
					return fn
				substituted_params = [
					replace( p, type = self._substitute_type_params( p.type, group_type_params, cls_args ))
					for p in ( fn.parameters or [] )
				]
				return replace( fn, parameters = substituted_params )

			match_stubs = [ _for_matching( fn ) for fn in target.stubs ]
			match_impls = [ _for_matching( fn ) for fn in target.implementations ]
			# resolve_call's own "targets" dict maps a plain (non-stub)
			# candidate to ITSELF - for a substituted copy, that's the
			# SUBSTITUTED copy, a throwaway replace()'d object, never a real
			# compile unit - map it back to the real, original Function
			# resolve_call actually meant (a stub instead resolves via its
			# own .bound_to, already the original, untouched by _for_matching)
			original_by_id = { id( sub ): orig for orig, sub in zip( target.implementations, match_impls ) }

			def _resolve_original( fn: Function ) -> Function:
				original = original_by_id.get( id( fn ), fn )
				if cls_args is not None and original.cls is group_cls:
					# when a stub won the overload resolution, `fn` is the stub's
					# `bound_to` (the real impl) and won't be in original_by_id -
					# the stub has a more specific return type than the real
					# impl (e.g. T vs T|None), so use the stub's return type
					# while still calling through to the real impl
					winning_stub = next( ( s for s in target.stubs if s.bound_to is original ), None )
					if winning_stub is not None:
						stub_spec = self.discovery._get_or_create_specialization( winning_stub, cls_args )
						stub_mono = self._monomorphized_function( stub_spec )
						method_spec = self.discovery._get_or_create_specialization( original, cls_args )
						impl_mono = self._monomorphized_function( method_spec )
						# the call target is the real impl, but the return type is
						# the stub's (more precise) one - swap it on the caller's
						# side via replace() so schedule() sees the right type
						return replace( impl_mono, return_type = stub_mono.return_type )
					method_spec = self.discovery._get_or_create_specialization( original, cls_args )
					return self._monomorphized_function( method_spec )
				return original

			try:
				branches, resolved = overload_resolution.resolve_call( match_stubs, match_impls, arg_types, kwarg_types, qualname = target.qualname )
			except CompileError as e:
				# resolve_call is a pure function of types with no
				# AST/Discovery reference by design - it raises unrecorded,
				# this is where a location actually gets attached and it
				# lands in the collector
				self.discovery.fail( str( e ), node )
			if branches:
				branches = [ ConditionalDispatch( conditions = b.conditions, function = _resolve_original( b.function )) for b in branches ]
				resolved = _resolve_original( resolved )
				return self._lower_conditional_dispatch( node, branches, resolved, args, kwargs, expected_type, want_result )
			target = _resolve_original( resolved )
			self._ensure_resolved( target ) # resolve_call() already resolved every group member internally - this just schedules the chosen one
		else:
			self._ensure_resolved( target )
			args, kwargs = self._lower_call_args( target, node )

		self.schedule( target.return_type )
		for param in target.parameters or []:
			self.schedule( param.type )

		if want_result:
			dest = self._new_temp( expected_type or target.return_type )
			self._emit( ir.Call( dest = dest, target = target, receiver = receiver, args = args, kwargs = kwargs ))
			return dest
		else:
			self._emit( ir.Call( dest = None, target = target, receiver = receiver, args = args, kwargs = kwargs ))
			return None

	def _lower_conditional_dispatch( self, node: ast.Call, branches: list[ConditionalDispatch], default: Function, args: list[ir.Operand], kwargs: dict[str,ir.Operand], expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# a union-typed argument's runtime tag decides which overload
		# implementation actually runs (e.g. len(copy_from) where
		# copy_from: bytes|bytearray resolves to two candidates, bytes and
		# bytearray). Reuses the same tag/data/v_<member> machinery match
		# statements use (UnionStorage.get) - branches are tried in
		# priority order, falling through to `default` (no test needed -
		# it's whatever's left once every more specific branch is excluded)
		self._ensure_resolved( default )
		for branch in branches:
			self._ensure_resolved( branch.function )

		dest = self._new_temp( expected_type or default.return_type ) if want_result else None
		end_label = self._new_label( 'dispatch_end' )
		for branch in branches:
			next_label = self._new_label( 'dispatch_next' )
			self._lower_dispatch_tests( node, branch.function, branch.conditions, args, kwargs, next_label )
			self._emit_dispatch_call( branch.function, args, kwargs, dest, want_result )
			self._emit( ir.Jump( target = end_label ))
			self._emit( ir.Label( name = next_label ))
		self._emit_dispatch_call( default, args, kwargs, dest, want_result )
		self._emit( ir.Label( name = end_label ))
		return dest

	def _lower_dispatch_tests( self, node: ast.AST, target: Function, conditions: list[tuple[Parameter,Type]], args: list[ir.Operand], kwargs: dict[str,ir.Operand], next_label: str ) -> None:
		# a branch's conditions are ANDed together - emits one Cmp +
		# JumpIfFalse per condition, all targeting next_label, which is
		# already a short-circuit AND with no combined boolean value to
		# build at all (same trick _expr_BoolOp uses, just directly in IR
		# since these operands are already lowered)
		bool_cls = self.discovery.find_name( 'bool', node )
		for param, leaf_type in conditions:
			operand = self._dispatch_operand_for_param( node, target, param, args, kwargs )
			shape = self._type_resolver._tagged_union_shape( operand.type )
			if shape is None:
				self.discovery.fail( f'{target.qualname}: conditional dispatch on a non-union argument: {ast.unparse(node)}', node )
			base, members = shape
			member = next( ( attr for attr in members if attr.type is leaf_type ), None )
			if member is None:
				self.discovery.fail( f'{target.qualname}: {leaf_type.qualname if leaf_type else "?"} is not a member of {operand.type.qualname}', node )
			tag_attr, _data_attr, _payload_cls, tags = self._union_storage.get( base )
			tag_dest = self._new_temp( tag_attr.type )
			self._emit( ir.GetAttr( dest = tag_dest, obj = operand, attr = tag_attr.stem ))
			cmp_dest = self._new_temp( bool_cls )
			self._emit( ir.Cmp( dest = cmp_dest, op = ir.CmpOp.EQ, left = tag_dest, right = ir.Const( type = tag_attr.type, value = tags[member.stem] ) ))
			self._emit( ir.JumpIfFalse( cond = cmp_dest, target = next_label ))

	def _dispatch_operand_for_param( self, node: ast.AST, target: Function, param: Parameter, args: list[ir.Operand], kwargs: dict[str,ir.Operand] ) -> ir.Operand:
		if param.stem in kwargs:
			return kwargs[param.stem]
		index = next( ( i for i, p in enumerate( target.parameters or [] ) if p is param ), None )
		if index is not None and index < len( args ):
			return args[index]
		self.discovery.fail( f'{target.qualname}: cannot locate the call-site argument for parameter {param.stem!r}', node )

	def _emit_dispatch_call( self, target: Function, args: list[ir.Operand], kwargs: dict[str,ir.Operand], dest: ir.Temp|None, want_result: bool ) -> None:
		params = target.parameters or []
		unwrapped_args = [ self._maybe_unwrap_union_arg( a, p.type ) for a, p in zip( args, params ) ]
		unwrapped_kwargs = {
			name: self._maybe_unwrap_union_arg( value, next( p for p in params if p.stem == name ).type )
			for name, value in kwargs.items()
		}
		self.schedule( target.return_type )
		for p in params:
			self.schedule( p.type )
		self._emit( ir.Call( dest = dest if want_result else None, target = target, receiver = None, args = unwrapped_args, kwargs = unwrapped_kwargs ))

	def _maybe_unwrap_union_arg( self, operand: ir.Operand, target_type: Type|None ) -> ir.Operand:
		# a union-typed call-site argument (copy_from: bytes|bytearray)
		# must be unwrapped to the concrete leaf type the chosen branch's
		# parameter actually declares before it can be passed as a real
		# argument - mirrors match's own payload extraction
		if target_type is None or operand.type is target_type:
			return operand
		shape = self._type_resolver._tagged_union_shape( operand.type )
		if shape is None:
			return operand
		base, members = shape
		member = next( ( attr for attr in members if attr.type is target_type ), None )
		if member is None:
			return operand
		tag_attr, data_attr, payload_cls, tags = self._union_storage.get( base )
		payload_dest = self._new_temp( payload_cls )
		self._emit( ir.GetAttr( dest = payload_dest, obj = operand, attr = data_attr.stem ))
		dest = self._new_temp( target_type )
		self._emit( ir.GetAttr( dest = dest, obj = payload_dest, attr = f'v_{member.stem}' ))
		return dest

	def _lower_union_receiver_call( self, node: ast.Call, dispatch: _ReceiverDispatch, receiver: ir.Operand, expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# copy_from.get_const_ptr() where copy_from: bytes|bytearray - unlike
		# _lower_conditional_dispatch (one shared Function, a union-typed
		# ARGUMENT unwrapped per branch), each leaf here has its own
		# unrelated method under this name, so what's dispatched on is the
		# RECEIVER's own tag instead - same tag/data/v_<member> machinery
		# match statements and dispatch already use (UnionStorage.get),
		# just no shared target Function to reuse ConditionalDispatch with
		reference = dispatch.per_leaf[0][1]
		for _member, fn in dispatch.per_leaf:
			self._ensure_resolved( fn )
		positional, keyword = self._match_call_args( reference, node )
		args = []
		for param, expr in positional:
			operand = self._lower_expr( expr, param.type )
			self._apply_move_hook( param, operand, dispatch.union.qualname )
			args.append( operand )
		kwargs = {}
		for param, expr in keyword:
			operand = self._lower_expr( expr, param.type )
			self._apply_move_hook( param, operand, dispatch.union.qualname )
			kwargs[param.stem] = operand

		tag_attr, data_attr, payload_cls, tags = self._union_storage.get( dispatch.union )
		bool_cls = self.discovery.find_name( 'bool', node )
		dest = self._new_temp( expected_type or reference.return_type ) if want_result else None
		end_label = self._new_label( 'recv_dispatch_end' )
		for i, ( member, fn ) in enumerate( dispatch.per_leaf ):
			is_last = i == len( dispatch.per_leaf ) - 1
			if not is_last:
				next_label = self._new_label( 'recv_dispatch_next' )
				tag_dest = self._new_temp( tag_attr.type )
				self._emit( ir.GetAttr( dest = tag_dest, obj = receiver, attr = tag_attr.stem ))
				cmp_dest = self._new_temp( bool_cls )
				self._emit( ir.Cmp( dest = cmp_dest, op = ir.CmpOp.EQ, left = tag_dest, right = ir.Const( type = tag_attr.type, value = tags[member.stem] )))
				self._emit( ir.JumpIfFalse( cond = cmp_dest, target = next_label ))
			payload_dest = self._new_temp( payload_cls )
			self._emit( ir.GetAttr( dest = payload_dest, obj = receiver, attr = data_attr.stem ))
			narrowed = self._new_temp( member.type )
			self._emit( ir.GetAttr( dest = narrowed, obj = payload_dest, attr = f'v_{member.stem}' ))
			self.schedule( fn.return_type )
			for p in fn.parameters or []:
				self.schedule( p.type )
			self._emit( ir.Call( dest = dest, target = fn, receiver = narrowed, args = args, kwargs = kwargs ))
			if not is_last:
				self._emit( ir.Jump( target = end_label ))
				self._emit( ir.Label( name = next_label ))
		self._emit( ir.Label( name = end_label ))
		return dest
