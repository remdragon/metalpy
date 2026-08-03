# stdlib imports:
import ast
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Callable

# local imports:
import cfg
import ir
from discovery import Discovery
from errors import CompileError
from mpy_types import (
	Name, Type, Variable, Parameter, Function, Overload, ClassLike, Module,
	Specialization, TaggedUnion, CUnion, TypeVar, ConditionalDispatch, Move, RCClass,
)
import overload_resolution

@dataclass( kw_only = True )
class _DeferBlock:
	is_err_only: bool # True for errdefer, False for plain defer
	flag: Variable # bool local, False until control passes the defer/errdefer statement
	instructions: list['ir.Instruction'] # captured at registration time, replayed in the epilogue

@dataclass( kw_only = True )
class _ReceiverDispatch:
	''' `_resolve_callee`'s answer when an attribute call's receiver is a
	union type and the attribute isn't found on the union itself (e.g.
	copy_from.get_const_ptr() where copy_from: bytes|bytearray) - each leaf
	type has its own unrelated method under this name, so unlike Overload
	(one shared Function, resolved by argument types) there's no single
	target Function here at all, just one per leaf, picked by the
	RECEIVER's own runtime tag. See _lower_union_receiver_call. '''
	union: TaggedUnion
	attr: str
	per_leaf: list[tuple[Variable,Function]] # (union.attributes member, that leaf's resolved method)

_BINOP_WRAP_OPCODES: dict[type,type] = {
	ast.Add: ir.AddWrap,
	ast.Sub: ir.SubWrap,
	ast.Mult: ir.MulWrap,
	ast.LShift: ir.ShlWrap,
}
_BINOP_CHECK_OPCODES: dict[type,type] = {
	ast.Add: ir.AddCheck,
	ast.Sub: ir.SubCheck,
	ast.Mult: ir.MulCheck,
	ast.LShift: ir.ShlCheck,
}
_BINOP_SATURATE_OPCODES: dict[type,type] = {
	ast.Add: ir.AddSaturate,
	ast.Sub: ir.SubSaturate,
	ast.Mult: ir.MulSaturate,
	ast.LShift: ir.ShlSaturate,
}
# always checked against ZeroDivisionError - unlike Add/Sub/Mult/Shl, there's
# no wrapped/saturated variant of division, so this is independent of the
# active arithmetic mode (see _expr_BinOp)
_DIV_MOD_OPCODES: dict[type,type] = {
	ast.FloorDiv: ir.Div, # no float type exists in this language (see ir.py) - '/' (ast.Div) is deliberately left unsupported rather than guessing what it should mean
	ast.Mod: ir.Mod,
}
# no overflow concept at all - always a single opcode, regardless of the
# active arithmetic mode (see _expr_BinOp)
_BITWISE_OPCODES: dict[type,type] = {
	ast.BitAnd: ir.BitAnd,
	ast.BitOr: ir.BitOr,
	ast.BitXor: ir.BitXor,
	ast.RShift: ir.Shr,
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
	is a stack of (kind, extra) pairs, pushed/popped by _stmt_With:
		('wrap', None)      - `with compiler.wrap_arithmetic:` - plain
		                       AddWrap/SubWrap/MulWrap, no Result involved
		('saturate', None)  - `with compiler.saturate_arithmetic:` - plain
		                       AddSaturate/SubSaturate/MulSaturate, likewise
		('check', None)     - the default (see above) - Check + OrReturn
		('check', errmsg)   - `with compiler.panic_arithmetic(errmsg):` -
		                       still Check-mode ops, but consumed with
		                       Unwrap(errmsg) instead of OrReturn, so (unlike
		                       the bare default) this does NOT require the
		                       enclosing function to return Result[_,
		                       OverflowError] - Unwrap panics, it never
		                       needs anywhere to propagate to

	defer/errdefer (SYNTAX.md section 3, either `defer(expr)`/`errdefer(expr)`
	as a single statement or `with defer:`/`with errdefer:` for several) move
	their body to the function's epilogue - a Label placed right after the
	body, reached either by falling off the end or by every `return`/checked-
	arithmetic-error-path jumping there once any defer/errdefer is active
	(self._needs_epilogue - named generically, since a future decref-insertion
	pass will need to trigger the exact same machinery, not just defer). Each
	block gets a bool flag (False until control passes its registration
	point) and is replayed in reverse registration order, guarded by that
	flag; errdefer blocks are additionally guarded by calling .is_err() on
	the function's own stowed return value (always a Result wherever errdefer
	is legal) - not a separate signal, so it also covers a plain `return
	Result.Err(x)`, not just the implicit OrJump path. defer/errdefer are
	rejected inside a loop (self._loop_depth) or nested inside each other
	(self._in_deferred_body) - see _register_defer_block.
	'''

	def __init__( self, discovery: Discovery, schedule: Callable[[object],None] ) -> None:
		self.discovery = discovery
		self.schedule = schedule
		# synthesized __tag/__payload storage per TaggedUnion, memoized by
		# id() - built lazily the first time a union is constructed/matched
		# against, but shared thereafter so every reference (Allocate,
		# GetAttr, across unrelated functions) points at the same objects.
		# Persists for the whole Lowering instance's lifetime, unlike
		# lower_function's per-function state
		self._union_storage: dict[int,tuple[Variable,Variable,CUnion,dict[str,int]]] = {}
		# monomorphized Function copies (T substituted with a concrete
		# type), memoized by id(Specialization) - discovery._get_or_create_
		# specialization already dedupes the Specialization itself by its
		# qualname key, so every call to the same instantiation (sys.
		# alloc[u8], from anywhere) reuses the SAME monomorphized Function
		# object, not a fresh copy per call site
		self._monomorphized: dict[int,Function] = {}

	def lower_function( self, fn: Function ) -> list[ir.Instruction]:
		module = self._find_module_for( fn )
		self._instructions: list[ir.Instruction] = []
		self._temp_id = 0
		self._label_id = 0
		self._pending_temps: list[ir.Temp] = []
		self._current_fn = fn
		self._arithmetic_mode: list[tuple[str,object]] = [ ( 'check', None ) ]
		self._loop_depth = 0
		self._loop_labels: list[tuple[str,str]] = [] # stack of (continue_label, break_label), innermost last
		self._in_deferred_body = False
		self._defer_blocks: list[_DeferBlock] = []
		self._needs_epilogue = self._function_needs_epilogue( fn.node.body )
		self._epilogue_label = '__epilogue__'

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
						if fn.stem == '__init__' and isinstance( fn.cls, RCClass ) and fn.cls.base is None:
							self._construction_self = self_param
							self._construction_fallible = self._init_fallibility( fn )

					for param in fn.parameters or []:
						self.schedule( param.type )
					self.schedule( fn.return_type )

					none_type = self.discovery.get_none_type()
					noreturn_type = self.discovery.get_intrinsics()['NoReturn']
					self._return_value_var = (
						Variable( stem = '__return_value', qualname = f'{fn.qualname}.__return_value', file = fn.file, line = fn.line, type = fn.return_type )
						if self._needs_epilogue and fn.return_type not in ( none_type, noreturn_type )
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
						union_storage = self._tagged_union_storage,
					)
					if self._construction_self is not None:
						for attr in fn.cls.attributes:
							self._ensure_resolved( attr ) # each field's own .type is lazily resolved, separate from the class itself - same as _lower_allocate_fields's identical loop
						self._cfg.enter_construction( self_param, fn.cls.attributes )
					elif fn.cls is not None and not fn.is_static and not fn.is_classmethod:
						self._cfg.enter_self( self_param, is_move = fn.is_move )
					for instr in self._cfg.prologue_instructions:
						self._emit( instr )
					if self._construction_self is not None:
						self._emit_construction_defaults( fn.cls, self_param, module )
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

					if self._needs_epilogue:
						if self._construction_self is not None:
							self._complete_construction_or_fail( fn )
						self._emit_epilogue( fn, none_type, body_start )
					elif fn.return_type is none_type and self._body_may_fall_off_the_end( fn.node.body ):
						# no defer/errdefer, so _emit_epilogue never runs at all -
						# but falling off the end without an explicit `return` is
						# still a real exit (implicit `return None`, same as
						# Python), and every OWNED/COPY local still live at that
						# point still needs its normal decref. Every explicit
						# `return` already does this itself (see _stmt_Return's
						# own else branch) - this only covers the specific case
						# nothing else does: reaching the function's closing brace
						# with no `return` at all. Falling off the end of a
						# non-fallible __init__ (return_type is always none_type
						# for it) is its success path - every required attribute
						# must be initialized here too
						if self._construction_self is not None:
							self._complete_construction_or_fail( fn )
						for instr in self._cfg.return_( None ):
							self._emit( instr )
						self._emit( ir.Return( value = None ))

					self._emit( ir.FuncEnd( name = fn.qualname ))

		return self._instructions

	def _emit_epilogue( self, fn: Function, none_type: Type, body_start: int ) -> None:
		# flag inits have to run before *any* code that could set them -
		# easiest to guarantee by splicing them in right after FuncStart
		# rather than tracking every branch that could reach a defer statement
		flag_inits = [
			ir.Assign( dest = block.flag, src = ir.Const( type = block.flag.type, value = False ))
			for block in self._defer_blocks
		]
		self._instructions[body_start:body_start] = flag_inits

		# falling off the end of the body (no explicit final `return`) reaches
		# this Label naturally, with no extra jump needed, since it's placed
		# immediately after the body - same for every `return`/OrJump, which
		# jumped here explicitly instead of exiting directly
		self._pending_temps = []
		self._emit( ir.Label( name = self._epilogue_label ))

		is_err_temp = None
		if any( block.is_err_only for block in self._defer_blocks ):
			is_err_temp = self._emit_is_err_check( fn.node )

		for i, block in enumerate( reversed( self._defer_blocks )):
			skip_label = f'__defer_skip_{i}__'
			self._emit( ir.JumpIfFalse( cond = block.flag, target = skip_label ))
			if block.is_err_only:
				self._emit( ir.JumpIfFalse( cond = is_err_temp, target = skip_label ))
			for instr in block.instructions:
				self._emit( instr )
			self._emit( ir.Label( name = skip_label ))

		for t in reversed( self._pending_temps ):
			for instr in self._cfg.delete_temp( t ):
				self._emit( instr )
			self._emit( ir.DeleteTemp( temp = t ))
		# covers the fall-off-the-end path specifically (no explicit return
		# reached this point) - every explicit return already checked this
		# itself in _stmt_Return, before jumping here
		if self._cfg.return_( None ):
			self.discovery.fail_loc(
				'defer/errdefer combined with reference-counted locals/parameters is not supported yet (see TODO.txt)',
				fn.file, fn.line,
			)
		return_value = self._return_value_var if fn.return_type is not none_type else None
		self._emit( ir.Return( value = return_value ))

	def _emit_is_err_check( self, node: ast.AST ) -> ir.Temp:
		bool_cls = self.discovery.find_name( 'bool', node )
		is_err_fn = self._attr_lookup_callable( self._return_value_var.type, 'is_err', node )
		self._ensure_resolved( is_err_fn )
		dest = self._new_temp( bool_cls )
		self._emit( ir.Call( dest = dest, target = is_err_fn, receiver = self._return_value_var, args = [], kwargs = {} ))
		return dest

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
		result_cls = self.discovery.find_name( 'Result', fn.node )
		ok = (
			isinstance( fn.return_type, Specialization )
			and fn.return_type.base is result_cls
			and len( fn.return_type.args ) == 2
			and fn.return_type.args[0] is none_type
		)
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

	def _is_result_err_call( self, node: ast.expr | None ) -> bool:
		# `return Result.Err(...)` - textually recognized, same spirit as
		# _defer_kind_of_call/_defer_kind_of_with - deliberately not
		# attempting deeper type-level inference (see _stmt_Return's own
		# comment on why anything else defaults to "requires completeness")
		return (
			isinstance( node, ast.Call )
			and isinstance( node.func, ast.Attribute )
			and node.func.attr == 'Err'
			and isinstance( node.func.value, ast.Name )
			and node.func.value.id == 'Result'
		)

	def lower_global( self, var: Variable ) -> list[ir.Instruction]:
		module = self._find_module_for( var )
		self._instructions = []
		self._temp_id = 0
		self._label_id = 0
		self._pending_temps = []
		self._current_fn = None
		self._arithmetic_mode = [ ( 'check', None ) ]
		# defer/errdefer/loops can't appear in a global initializer (it's a
		# single expression, not a statement body reachable through
		# _lower_stmt) - reset for consistency/safety only, never touched here
		self._loop_depth = 0
		self._loop_labels = []
		self._in_deferred_body = False
		self._defer_blocks: list[_DeferBlock] = []
		self._needs_epilogue = False
		self._return_value_var = None

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
			is_success = not ( self._construction_fallible and self._is_result_err_call( node.value ))
			if is_success:
				self._complete_construction_or_fail( self._current_fn )
		if self._needs_epilogue:
			# every return in a defer/errdefer-using function funnels
			# through the SAME shared epilogue (_emit_epilogue) - RC decref
			# there necessarily uses whatever's live at the END of the
			# whole function body, not this specific return's own live set
			# (which could genuinely differ - an early return before some
			# later local exists). Rather than risk decref'ing something
			# not actually alive on this path, this combination is
			# rejected outright for now - see TODO.txt. cfg.return_() is
			# still called (not just skipped) so the check is exact: only
			# functions that would ACTUALLY need an RC decref here trip it
			cfg_instructions = self._cfg.return_( value )
			if cfg_instructions:
				self.discovery.fail(
					'defer/errdefer combined with reference-counted locals/parameters is not supported yet '
					f'(see TODO.txt): {ast.unparse(node)}',
					node,
				)
			if self._return_value_var is not None and value is not None:
				self._emit( ir.Assign( dest = self._return_value_var, src = value ))
			self._emit( ir.Jump( target = self._epilogue_label ))
		else:
			for instr in self._cfg.return_( value ):
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
		if self._is_compiler_early_return_call( node.value ):
			self._lower_compiler_early_return( node.value )
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

		if self._is_compiler_attr( context_expr, 'wrap_arithmetic' ):
			mode = ( 'wrap', None )
		elif self._is_compiler_attr( context_expr, 'saturate_arithmetic' ):
			mode = ( 'saturate', None )
		elif self._is_compiler_panic_arithmetic_call( context_expr ):
			if len( context_expr.args ) != 1 or context_expr.keywords:
				self.discovery.fail( f'compiler.panic_arithmetic(...) takes exactly one argument: {ast.unparse(node)}', node )
			str_cls = self.discovery.find_name( 'str', node )
			errmsg = self._lower_expr( context_expr.args[0], str_cls )
			mode = ( 'check', errmsg )
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

	def _is_compiler_attr( self, node: ast.expr, attr: str ) -> bool:
		# textual recognition, same as discovery.py's _is_compiler_target_call -
		# `compiler` is a special pseudo-module (Discovery.compiler_module),
		# not something with a real .names dict to resolve this through
		return (
			isinstance( node, ast.Attribute )
			and node.attr == attr
			and isinstance( node.value, ast.Name )
			and node.value.id == 'compiler'
		)

	def _is_compiler_panic_arithmetic_call( self, node: ast.expr ) -> bool:
		return (
			isinstance( node, ast.Call )
			and isinstance( node.func, ast.Attribute )
			and node.func.attr == 'panic_arithmetic'
			and isinstance( node.func.value, ast.Name )
			and node.func.value.id == 'compiler'
		)

	def _is_compiler_sizeof_call( self, node: ast.expr ) -> bool:
		return (
			isinstance( node, ast.Call )
			and isinstance( node.func, ast.Attribute )
			and node.func.attr == 'sizeof'
			and isinstance( node.func.value, ast.Name )
			and node.func.value.id == 'compiler'
		)

	# byte size for every intrinsic scalar this target model actually has a
	# fixed size for - matches this compiler's own intrinsics (see
	# discovery.py's get_intrinsics()). Real user classes have no known size
	# yet (no field-layout computation exists - that's an emitter concern),
	# so compiler.sizeof(SomeClass) stays unsupported until then
	_INTRINSIC_BYTE_SIZES: dict[str,int] = {
		'i8': 1, 'u8': 1, 'i16': 2, 'u16': 2, 'i32': 4, 'u32': 4,
		'i64': 8, 'u64': 8, 'i128': 16, 'u128': 16,
		'isize': 8, 'usize': 8, 'bool': 1,
		'Ptr': 8, 'ConstPtr': 8,
	}

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
		size = self._INTRINSIC_BYTE_SIZES.get( getattr( target_type, 'stem', None ) )
		if size is None:
			self.discovery.fail( f'compiler.sizeof({target_type.qualname}) is not supported yet - only intrinsic scalar types have a known compile-time size', node )
		usize_cls = self.discovery.get_intrinsics()['usize']
		return ir.Const( type = expected_type or usize_cls, value = size )

	def _is_compiler_refcount_call( self, node: ast.expr ) -> bool:
		return (
			isinstance( node, ast.Call )
			and isinstance( node.func, ast.Attribute )
			and node.func.attr == 'refcount'
			and isinstance( node.func.value, ast.Name )
			and node.func.value.id == 'compiler'
		)

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
		if not isinstance( value.type, RCClass ):
			self.discovery.fail(
				f'compiler.refcount(...) argument must be a reference-counted value, not '
				f'{value.type.qualname if value.type else "?"}: {ast.unparse(node)}',
				node,
			)
		usize_cls = self.discovery.get_intrinsics()['usize']
		dest = self._new_temp( expected_type or usize_cls )
		self._emit( ir.RefCount( dest = dest, value = value ))
		return dest

	def _is_compiler_early_return_call( self, node: ast.expr ) -> bool:
		return (
			isinstance( node, ast.Call )
			and isinstance( node.func, ast.Attribute )
			and node.func.attr == 'early_return'
			and isinstance( node.func.value, ast.Name )
			and node.func.value.id == 'compiler'
		)

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
		# (`compiler.early_return(self._payload.err)`), but that body is
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
		result_cls = self.discovery.find_name( 'Result', node )
		ok = (
			fn is not None
			and isinstance( return_type, Specialization )
			and return_type.base is result_cls
			and len( return_type.args ) == 2
		)
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

	def _function_needs_epilogue( self, body: list[ast.stmt] ) -> bool:
		# a simple AST-level pre-scan (not real lowering) - has to be known
		# before lowering a single statement, since every `return` in the
		# function must behave uniformly (see the class docstring)
		for stmt in body:
			for node in ast.walk( stmt ):
				if isinstance( node, ast.With ) and len( node.items ) == 1 and self._defer_kind_of_with( node.items[0].context_expr ):
					return True
				if isinstance( node, ast.Expr ) and self._defer_kind_of_call( node.value ):
					return True
		return False

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
			result_cls = self.discovery.find_name( 'Result', node )
			return_type = fn.return_type if fn is not None else None
			ok = (
				fn is not None
				and isinstance( return_type, Specialization )
				and return_type.base is result_cls
				and len( return_type.args ) == 2
			)
			if not ok:
				where = f'{fn.qualname} returns {return_type.qualname if return_type else None}' if fn is not None else 'this is not inside a function'
				self.discovery.fail( f'errdefer requires the enclosing function to return Result[_,_] ({where})', node )

		bool_cls = self.discovery.find_name( 'bool', node )
		index = len( self._defer_blocks )
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

		self._defer_blocks.append( _DeferBlock( is_err_only = is_err_only, flag = flag, instructions = captured ))
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
		self._ensure_resolved( owner_type )
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
		result_cls = self.discovery.find_name_or_none( 'Result' )
		if result_cls is None or not ( isinstance( value.type, Specialization ) and value.type.base is result_cls and len( value.type.args ) == 2 ):
			return value
		result_type, error_cls = value.type.args
		self._require_result_return( node, value.type.base, error_cls, alternatives )
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

	def _is_range_call( self, node: ast.expr ) -> bool:
		# range(...) is textually recognized as compiler sugar, same as
		# compiler.wrap_arithmetic/defer/etc. - there's no real range()
		# function (TODO.txt: a real range()/Iterator needs the generator
		# state-machine transform, which doesn't exist yet). This covers
		# exactly the 1-2 arg counting-loop shape real lib/ code already
		# uses (str.concat's `for i in range(count):`)
		return isinstance( node, ast.Call ) and isinstance( node.func, ast.Name ) and node.func.id == 'range'

	_FOR_LOOP_ALTERNATIVES = 'call .__len__()/.__getitem__() directly and consume their Result yourself instead'

	def _stmt_For( self, node: ast.For ) -> None:
		if not isinstance( node.target, ast.Name ):
			self.discovery.fail( f'for loop target must be a plain name: {ast.unparse(node)}', node )
		if node.orelse:
			self.discovery.fail( 'for/else is not supported', node )
		if self._is_range_call( node.iter ):
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

	def _stmt_Match( self, node: ast.Match ) -> None:
		# desugars to a synthesized if/elif chain (one arm per case, in
		# source order), delegating to _stmt_If for the actual branching -
		# each arm's test is built by _match_pattern below. The subject is
		# lowered exactly once into a hidden local (same technique as the
		# for-loop's scaffolding: a real named Variable registered into the
		# function's own scope, so the synthesized per-case AST can
		# reference it by name repeatedly with no re-evaluation risk)
		subj = self._lower_expr( node.subject, None )
		subj_var = self._declare_hidden_local( f'__match_subj_{self._label_id}', subj.type, node )
		self._emit( ir.Assign( dest = subj_var, src = subj ))
		subj_name = self._synth_name( subj_var.stem, node )

		chain: ast.If|None = None
		tail: ast.If|None = None
		for case in node.cases:
			if case.guard is not None:
				self.discovery.fail( f'match guards (case ... if ...) are not yet supported: {ast.unparse(case.pattern)}', node )
			test, binds = self._match_pattern( subj_name, subj_var.type, case.pattern, node )
			arm = ast.If( test = test, body = [ *binds, *case.body ], orelse = [] )
			ast.copy_location( arm, node )
			if chain is None:
				chain = arm
			else:
				tail.orelse = [ arm ]
			tail = arm

		if chain is not None:
			self._stmt_If( chain )

	def _match_pattern( self, subj_expr: ast.expr, subj_type: Type|None, pattern: ast.pattern, node: ast.AST ) -> tuple[ast.expr,list[ast.stmt]]:
		# returns (test_expr, binding_stmts): test_expr is a boolean AST
		# expression (lowered later, via the enclosing synthesized If, with
		# bool_cls as its expected type) that's True iff subj_expr matches
		# pattern; binding_stmts are synthesized Assign statements for
		# whatever names the pattern introduces - only valid once test_expr
		# has evaluated True, so the caller must place them inside the
		# resulting if-body, never unconditionally
		if isinstance( pattern, ast.MatchAs ) and pattern.pattern is None:
			# a bare name (or `_` - Python parses a wildcard the same way,
			# with name=None) - matches anything unconditionally; binds the
			# whole subject if a name was actually given
			test = ast.Constant( value = True )
			ast.copy_location( test, node )
			if pattern.name is None:
				return test, []
			bind = ast.Assign( targets = [ ast.Name( id = pattern.name, ctx = ast.Store() ) ], value = subj_expr )
			ast.copy_location( bind, node )
			return test, [ bind ]

		if not isinstance( pattern, ast.MatchClass ):
			self.discovery.fail( f'unsupported match pattern: {ast.unparse(pattern)}', node )
		if pattern.kwd_patterns or len( pattern.patterns ) != 1:
			self.discovery.fail( f'match patterns support exactly one positional sub-pattern: {ast.unparse(pattern)}', node )
		if not isinstance( pattern.cls, ast.Attribute ):
			self.discovery.fail( f'unsupported match pattern class: {ast.unparse(pattern)}', node )

		owner = self._try_resolve_namespace( pattern.cls.value )
		result_cls = self.discovery.find_name_or_none( 'Result' )
		if result_cls is not None and owner is result_cls and pattern.cls.attr in ( 'Ok', 'Err' ):
			# Result.Ok(...)/Result.Err(...) - Result isn't a real
			# TaggedUnion (it predates @union being scoped - see
			# lowering.py's other Result-specific comments), so this is
			# special-cased the same way .or_return() is: reuses is_ok()/
			# is_err() and the real _payload.ok/_payload.err fields
			# directly instead of routing through _tagged_union_storage
			is_ok = pattern.cls.attr == 'Ok'
			if not ( isinstance( subj_type, Specialization ) and subj_type.base is result_cls and len( subj_type.args ) == 2 ):
				self.discovery.fail( f'{ast.unparse(pattern.cls)}(...) pattern used against a non-Result subject: {ast.unparse(pattern)}', node )
			ok_type, err_type = subj_type.args
			test = ast.Call(
				func = ast.Attribute( value = subj_expr, attr = ( 'is_ok' if is_ok else 'is_err' ), ctx = ast.Load() ),
				args = [], keywords = [],
			)
			ast.copy_location( test, node )
			payload_expr = ast.Attribute(
				value = ast.Attribute( value = subj_expr, attr = '_payload', ctx = ast.Load() ),
				attr = ( 'ok' if is_ok else 'err' ),
				ctx = ast.Load(),
			)
			ast.copy_location( payload_expr, node )
			inner_test, inner_binds = self._match_pattern( payload_expr, ok_type if is_ok else err_type, pattern.patterns[0], node )
			combined = ast.BoolOp( op = ast.And(), values = [ test, inner_test ] )
			ast.copy_location( combined, node )
			return combined, inner_binds

		if isinstance( owner, TaggedUnion ):
			self._ensure_resolved( owner )
			member = next( ( attr for attr in owner.attributes if attr.stem == pattern.cls.attr ), None )
			if member is None:
				self.discovery.fail( f'{owner.qualname} has no member {pattern.cls.attr!r}: {ast.unparse(pattern)}', node )
			self._ensure_resolved( member )
			tag_attr, data_attr, payload_cls, tags = self._tagged_union_storage( owner )
			tag_expr = ast.Attribute( value = subj_expr, attr = tag_attr.stem, ctx = ast.Load() )
			ast.copy_location( tag_expr, node )
			test = ast.Compare( left = tag_expr, ops = [ ast.Eq() ], comparators = [ ast.Constant( value = tags[member.stem] ) ] )
			ast.copy_location( test, node )
			payload_expr = ast.Attribute(
				value = ast.Attribute( value = subj_expr, attr = data_attr.stem, ctx = ast.Load() ),
				attr = f'v_{member.stem}',
				ctx = ast.Load(),
			)
			ast.copy_location( payload_expr, node )
			inner_test, inner_binds = self._match_pattern( payload_expr, member.type, pattern.patterns[0], node )
			combined = ast.BoolOp( op = ast.And(), values = [ test, inner_test ] )
			ast.copy_location( combined, node )
			return combined, inner_binds

		self.discovery.fail( f'unsupported match pattern class: {ast.unparse(pattern)}', node )

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
		self._ensure_resolved( name ) # a global read only by bare name (never via an annotation/attribute chain) still needs its own resolve+schedule - Compiler._enqueue ignores this for a non-global (field/parameter/local) Variable
		return name

	def _expr_Constant( self, node: ast.Constant, expected_type: Type|None ) -> ir.Operand:
		if expected_type is None:
			self.discovery.fail(
				f'cannot infer the type of literal {node.value!r} - no expected type available from context ({ast.unparse(node)})',
				node,
			)
		return ir.Const( type = expected_type, value = node.value )

	def _expr_Attribute( self, node: ast.Attribute, expected_type: Type|None ) -> ir.Operand:
		obj = self._lower_expr( node.value, None )
		attr_var = self._attr_lookup( obj.type, node.attr, node )
		dest = self._new_temp( attr_var.type )
		self._emit( ir.GetAttr( dest = dest, obj = obj, attr = node.attr ))
		return dest

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
			index = self._lower_expr( node.slice, None )
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

	_OPCODES_BY_KIND = {
		'wrap': _BINOP_WRAP_OPCODES,
		'saturate': _BINOP_SATURATE_OPCODES,
		'check': _BINOP_CHECK_OPCODES,
	}
	_UNARY_NEG_OPCODES_BY_KIND = {
		'wrap': ir.NegWrap,
		'saturate': ir.NegSaturate,
		'check': ir.NegCheck,
	}

	_ARITHMETIC_ALTERNATIVES = (
		'wrap this in `with compiler.wrap_arithmetic:`, `with compiler.saturate_arithmetic:`, '
		'or `with compiler.panic_arithmetic(...):` instead'
	)
	# the primary, expected path for division is the same as any other
	# Check-mode op: the enclosing function returns Result[_,
	# ZeroDivisionError] and the Result propagates via OrReturn/OrJump - no
	# panic involved, and this is what happens even inside wrap_arithmetic/
	# saturate_arithmetic (there's no wrapped/saturated division opcode, so
	# those modes don't change division's checked-ness at all). This message
	# only fires when that requirement ISN'T met - panic_arithmetic is the
	# one remaining alternative to changing the return type, not a default
	_DIVISION_ALTERNATIVES = 'wrap this in `with compiler.panic_arithmetic(...):` instead'

	def _expr_BinOp( self, node: ast.BinOp, expected_type: Type|None ) -> ir.Operand:
		op_type = type( node.op )
		kind, extra = self._arithmetic_mode[-1]

		left_is_const = isinstance( node.left, ast.Constant )
		right_is_const = isinstance( node.right, ast.Constant )
		if left_is_const and not right_is_const:
			right = self._lower_expr( node.right, expected_type )
			left = self._lower_expr( node.left, right.type )
		elif right_is_const and not left_is_const:
			left = self._lower_expr( node.left, expected_type )
			right = self._lower_expr( node.right, left.type )
		else:
			left = self._lower_expr( node.left, expected_type )
			right = self._lower_expr( node.right, expected_type or left.type )

		result_type = expected_type or left.type

		if op_type in _BITWISE_OPCODES:
			# no overflow concept - always a single opcode, independent of
			# the active wrap/check/saturate arithmetic mode (that only
			# governs Add/Sub/Mult/Shl)
			dest = self._new_temp( result_type )
			self._emit( _BITWISE_OPCODES[op_type]( dest = dest, left = left, right = right ))
			return dest

		if op_type in _DIV_MOD_OPCODES:
			# always checked against ZeroDivisionError, independent of the
			# active arithmetic mode - but still honors panic_arithmetic's
			# own errmsg (extra) for how the Result gets consumed, exactly
			# like Check-mode Add/Sub/Mult/Shl below
			result_cls, error_cls = self._lookup_result_and_error_types( node, 'ZeroDivisionError' )
			if extra is None:
				self._require_result_return( node, result_cls, error_cls, self._DIVISION_ALTERNATIVES )
			return self._emit_checked_binop( _DIV_MOD_OPCODES[op_type], left, right, result_type, result_cls, error_cls, extra )

		opcode = self._OPCODES_BY_KIND[kind].get( op_type )
		if opcode is None:
			self.discovery.fail( f'unsupported binary operator: {ast.unparse(node)}', node )

		if kind in ( 'wrap', 'saturate' ):
			dest = self._new_temp( result_type )
			self._emit( opcode( dest = dest, left = left, right = right ))
			return dest

		# check mode (the default - see the class docstring): the op itself
		# produces Result[result_type,OverflowError]. How that Result gets
		# consumed depends on `extra`: the default (extra is None) uses
		# OrReturn, mirroring Result.or_return()'s own semantics, and needs
		# somewhere for the error to propagate to; `with
		# compiler.panic_arithmetic(msg):` (extra is the lowered msg operand)
		# uses Unwrap instead, which panics immediately and so has no such
		# requirement
		result_cls, overflow_cls = self._lookup_result_and_error_types( node, 'OverflowError' )
		if extra is None:
			# validated before anything gets emitted - a mid-statement
			# failure here must not leave partial instructions behind for
			# the per-statement recovery boundary to silently keep
			self._require_result_return( node, result_cls, overflow_cls, self._ARITHMETIC_ALTERNATIVES )
		return self._emit_checked_binop( opcode, left, right, result_type, result_cls, overflow_cls, extra )

	def _emit_checked_binop( self, opcode: type, left: ir.Operand, right: ir.Operand, result_type: Type, result_cls: ClassLike, error_cls: ClassLike, extra: ir.Operand|None ) -> ir.Temp:
		check_type = self.discovery._get_or_create_specialization( result_cls, [ result_type, error_cls ] )
		check_dest = self._new_temp( check_type )
		self._emit( opcode( dest = check_dest, left = left, right = right ))
		return self._consume_checked_result( check_dest, result_type, extra )

	def _consume_checked_result( self, check_dest: ir.Temp, result_type: Type, extra: ir.Operand|None ) -> ir.Temp:
		# shared by both binop (AddCheck/.../Div/Mod) and unary (NegCheck)
		# Check-mode ops - see _expr_BinOp's own comment on the OrReturn/
		# OrJump/Unwrap split
		unwrapped = self._new_temp( result_type )
		if extra is None:
			if self._needs_epilogue:
				self._emit( ir.OrJump( dest = unwrapped, value = check_dest, target = self._epilogue_label, return_slot = self._return_value_var ))
			else:
				self._emit( ir.OrReturn( dest = unwrapped, value = check_dest ))
		else:
			self._emit( ir.Unwrap( dest = unwrapped, value = check_dest, errmsg = extra ))
		return unwrapped

	def _lookup_result_and_error_types( self, node: ast.AST, error_name: str ) -> tuple[ClassLike,ClassLike]:
		result_cls = self.discovery.find_name( 'Result', node )
		error_cls = self.discovery.find_name( error_name, node )
		return result_cls, error_cls

	def _require_result_return( self, node: ast.AST, result_cls: ClassLike, error_cls: ClassLike, alternatives: str ) -> None:
		fn = self._current_fn
		return_type = fn.return_type if fn is not None else None
		ok = (
			fn is not None
			and isinstance( return_type, Specialization )
			and return_type.base is result_cls
			and len( return_type.args ) == 2
			and return_type.args[1] is error_cls
		)
		if not ok:
			where = f'{fn.qualname} returns {return_type.qualname if return_type else None}' if fn is not None else 'this is not inside a function'
			self.discovery.fail(
				f'this requires the enclosing function to return Result[_,{error_cls.stem}] ({where}) - {alternatives}',
				node,
			)

	def _expr_UnaryOp( self, node: ast.UnaryOp, expected_type: Type|None ) -> ir.Operand:
		# `not` (ast.Not) is deliberately not handled here - there's no
		# boolean-negation opcode in ir.py yet (unlike Invert/Neg*), and
		# adding one is a real design decision, not just wiring up an
		# existing primitive like the rest of this method does
		operand = self._lower_expr( node.operand, expected_type )
		result_type = expected_type or operand.type

		if isinstance( node.op, ast.Invert ):
			# no overflow concept, same posture as the non-Shl bitwise binops
			dest = self._new_temp( result_type )
			self._emit( ir.Invert( dest = dest, operand = operand ))
			return dest

		if isinstance( node.op, ast.USub ):
			kind, extra = self._arithmetic_mode[-1]
			opcode = self._UNARY_NEG_OPCODES_BY_KIND[kind]
			if kind in ( 'wrap', 'saturate' ):
				dest = self._new_temp( result_type )
				self._emit( opcode( dest = dest, operand = operand ))
				return dest
			result_cls, overflow_cls = self._lookup_result_and_error_types( node, 'OverflowError' )
			if extra is None:
				self._require_result_return( node, result_cls, overflow_cls, self._ARITHMETIC_ALTERNATIVES )
			check_type = self.discovery._get_or_create_specialization( result_cls, [ result_type, overflow_cls ] )
			check_dest = self._new_temp( check_type )
			self._emit( opcode( dest = check_dest, operand = operand ))
			return self._consume_checked_result( check_dest, result_type, extra )

		self.discovery.fail( f'unsupported unary operator: {ast.unparse(node)}', node )

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
		# ast.Is/IsNot/In/NotIn are deliberately not handled here - `is`/
		# `is not` are reserved for eventual tagged-union type narrowing
		# (see TODO.txt's union disambiguation section: `if x is int:`),
		# not a plain identity Cmp, and `in`/`not in` need a real container
		# protocol that doesn't exist yet - guessing at either would bake in
		# the wrong semantics
		if len( node.ops ) != 1 or len( node.comparators ) != 1:
			self.discovery.fail( f'chained comparisons are not yet supported: {ast.unparse(node)}', node )
		cmp_op = self._CMP_OPCODES.get( type( node.ops[0] ))
		if cmp_op is None:
			self.discovery.fail( f'unsupported comparison operator: {ast.unparse(node)}', node )

		right_node = node.comparators[0]
		left_is_const = isinstance( node.left, ast.Constant )
		right_is_const = isinstance( right_node, ast.Constant )
		# unlike _expr_BinOp, expected_type here is the comparison's own
		# result type (bool) - unrelated to what type the operands
		# themselves should be lowered as, so it's never passed to either
		# side, only used (below) as one operand's own type inferred from
		# the other
		if left_is_const and not right_is_const:
			right = self._lower_expr( right_node, None )
			left = self._lower_expr( node.left, right.type )
		elif right_is_const and not left_is_const:
			left = self._lower_expr( node.left, None )
			right = self._lower_expr( right_node, left.type )
		else:
			left = self._lower_expr( node.left, None )
			right = self._lower_expr( right_node, None )

		bool_cls = self.discovery.find_name( 'bool', node )
		dest = self._new_temp( bool_cls )
		self._emit( ir.Cmp( dest = dest, op = cmp_op, left = left, right = right ))
		return dest

	# --- shared helpers ----------------------------------------------------------

	def _ensure_resolved( self, obj: object ) -> None:
		# resolving (populating .names/.parameters/whatever) needs to happen
		# immediately, mid-statement, for whoever's asking - unlike schedule(),
		# which just queues obj for whenever the work queue gets to it, this
		# can't wait.
		#
		# unconditionally hands obj to schedule() too - Compiler._enqueue is
		# the single place that judges what's actually a compile unit worth
		# queuing (Function, ClassLike, a genuinely module-level Variable),
		# what decomposes into more of those (a Specialization's base + each
		# arg), and what to just quietly ignore (a Module walked mid-
		# namespace-lookup, a class field/parameter/local Variable). Nothing
		# here needs to know or duplicate that judgment - see Compiler's own
		# docstring for the full list of what it does with each kind
		resolve = getattr( obj, 'resolve', None )
		if resolve is not None:
			resolve()
		self.schedule( obj )

	def _attr_lookup( self, owner_type: Type|None, attr: str, ctx: ast.AST ) -> Variable:
		self._ensure_resolved( owner_type ) # Specialization.resolve/.names passthrough to .base - no unwrap needed
		names = getattr( owner_type, 'names', None )
		if not isinstance( names, dict ):
			self.discovery.fail( f'{owner_type!r} has no members, cannot look up {attr!r} ({ast.unparse(ctx)})', ctx )
		found = names.get( attr )
		if not isinstance( found, Variable ):
			self.discovery.fail( f'{owner_type.qualname if owner_type else "?"} has no attribute {attr!r}', ctx )
		self._ensure_resolved( found )
		return self._substituted_field( found, owner_type )

	def _substituted_field( self, found: Variable, owner_type: Type|None ) -> Variable:
		# a field declared using its owning generic class's own type params
		# (e.g. Result[T,E]'s `_payload: ResultPayload[T,E]`) is stored ONCE,
		# unsubstituted, on the class itself - accessing it through a
		# concrete Specialization (Result[Ptr[u8],OwnershipError]) must
		# substitute T/E with that Specialization's own args, or every
		# access sees the bare TypeVars regardless of which instantiation it
		# went through (this was invisible before match statements: nothing
		# previously read a generic field's type this way - checked-
		# arithmetic/or_return() consume a Result's payload via a dedicated
		# opcode on the whole Result value, never by synthesizing a literal
		# `.field` AST and lowering it)
		type_params = getattr( getattr( owner_type, 'base', None ), 'type_params', None )
		if not isinstance( owner_type, Specialization ) or not type_params:
			return found
		substituted_type = self._substitute_type_params( found.type, type_params, owner_type.args )
		if substituted_type is found.type:
			return found
		return replace( found, type = substituted_type ) # a shallow copy - `found` is the SAME shared Variable object for every access of this field, regardless of specialization, so this must not mutate it in place

	def _substitute_type_params( self, t: Type|None, type_params: list[TypeVar], args: list[Type] ) -> Type|None:
		if isinstance( t, TypeVar ):
			for param, arg in zip( type_params, args ):
				if t is param:
					return arg
			return t
		if isinstance( t, Specialization ):
			substituted_args = [ self._substitute_type_params( a, type_params, args ) for a in t.args ]
			if all( sa is a for sa, a in zip( substituted_args, t.args )):
				return t
			return self.discovery._get_or_create_specialization( t.base, substituted_args )
		return t

	def _monomorphized_function( self, spec: Specialization ) -> Function:
		# a distinct compiled unit per explicit generic instantiation
		# (sys.alloc[u8] vs sys.alloc[u32] are two separate functions, each
		# with T bound to a concrete type throughout - not a shared
		# unspecialized body the way a generic CLASS's methods stay today).
		# Built by copying the base Function with every generic-facing
		# field substituted: qualname (so FuncStart/Call get sys.alloc[u8],
		# not the shared sys.alloc), parameters/return_type (via
		# _substitute_type_params), and names (T's own entry replaced with
		# the concrete arg, so ordinary name lookups - including
		# compiler.sizeof(T) - resolve it correctly while lowering fn.node.
		# body, which is otherwise untouched/shared AST). Memoized by
		# id(spec) - discovery._get_or_create_specialization already
		# dedupes the Specialization itself, so this only ever builds one
		# copy per distinct instantiation
		cached = self._monomorphized.get( id( spec ) )
		if cached is not None:
			return cached
		base = spec.base
		if base.resolve is not None:
			base.resolve()
		type_params = base.type_params or []
		substituted_params = [
			replace( p, type = self._substitute_type_params( p.type, type_params, spec.args ) )
			for p in ( base.parameters or [] )
		]
		substituted_return = self._substitute_type_params( base.return_type, type_params, spec.args )
		substituted_names = dict( base.names )
		for tv, arg in zip( type_params, spec.args ):
			substituted_names[tv.stem] = arg
		for p in substituted_params:
			substituted_names[p.stem] = p
		monomorphized = replace(
			base,
			qualname = spec.qualname,
			parameters = substituted_params,
			return_type = substituted_return,
			names = substituted_names,
			type_params = None,
			resolve = None,
		)
		self._monomorphized[ id( spec ) ] = monomorphized
		return monomorphized

	def lower_function_specialization( self, spec: Specialization ) -> tuple[Function,list[ir.Instruction]]:
		monomorphized = self._monomorphized_function( spec )
		return monomorphized, self.lower_function( monomorphized )

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
		if isinstance( receiver.type, TaggedUnion ):
			self._ensure_resolved( receiver.type )
			direct = receiver.type.names.get( func_node.attr )
			if not isinstance( direct, ( Function, Overload )):
				return self._resolve_union_receiver_members( receiver.type, func_node.attr, func_node ), receiver
		target = self._attr_lookup_callable( receiver.type, func_node.attr, func_node )
		return target, receiver

	def _resolve_union_receiver_members( self, union: TaggedUnion, attr: str, ctx: ast.AST ) -> _ReceiverDispatch:
		# the union itself has no .names entry for attr (an anonymous X|Y
		# union never does; a real @union class only reaches here if it
		# doesn't declare attr as a real method of its own) - so each leaf
		# type's own, unrelated method under this name has to be looked up
		# individually instead, then dispatched on the receiver's runtime tag
		per_leaf: list[tuple[Variable,Function]] = []
		for member in union.attributes:
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
		self._ensure_resolved( owner_type ) # Specialization.resolve/.names passthrough to .base - no unwrap needed
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
		declared = { attr.stem: attr for attr in target_cls.attributes }
		given = { kw.arg for kw in node.keywords }
		missing = declared.keys() - given
		if isinstance( target_cls, CUnion ):
			# a union's whole point - only ONE member is ever meaningfully
			# set at a time (see _tagged_union_storage's identical comment
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
		self._emit( ir.Allocate( dest = dest, cls = target_cls, fields = fields ))
		return dest

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
		if fn is None or fn.cls is not target_cls:
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
		if not isinstance( target_cls, ClassLike ):
			return None
		self._ensure_resolved( target_cls )
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
		self._ensure_resolved( init )

		self_temp = self._new_temp( target_cls )
		self._emit( ir.Allocate( dest = self_temp, cls = target_cls, fields = {} ))

		positional, keyword = self._match_call_args( init, node )
		args = []
		for param, expr in positional:
			operand = self._lower_expr( expr, param.type )
			self._apply_move_hook( param, operand, init.qualname )
			args.append( operand )
		kwargs = {}
		for param, expr in keyword:
			operand = self._lower_expr( expr, param.type )
			self._apply_move_hook( param, operand, init.qualname )
			kwargs[param.stem] = operand
		self.schedule( init.return_type )
		for param in init.parameters or []:
			self.schedule( param.type )

		if not self._init_fallibility( init ):
			self._emit( ir.Call( dest = None, target = init, receiver = self_temp, args = args, kwargs = kwargs ))
			return self_temp
		return self._emit_fallible_construction( node, target_cls, init, self_temp, args, kwargs, expected_type )

	def _emit_fallible_construction(
		self, node: ast.Call, target_cls: RCClass, init: Function, self_temp: ir.Temp,
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
		self_var = self._declare_hidden_local( f'__ctor_self_{unique}', target_cls, node )
		for instr in self._cfg.assign( self_var, self_temp, is_alias = False ):
			self._emit( instr )
		self._emit( ir.Assign( dest = self_var, src = self_temp ))

		result_var = self._declare_hidden_local( f'__ctor_result_{unique}', init.return_type, node )
		for instr in self._cfg.assign( result_var, init_result, is_alias = False ):
			self._emit( instr )
		self._emit( ir.Assign( dest = result_var, src = init_result ))

		error_cls = init.return_type.args[1]
		result_cls = self.discovery.find_name( 'Result', node )
		outer_result_type = expected_type or self.discovery._get_or_create_specialization( result_cls, [ target_cls, error_cls ] )
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
				value = ast.Attribute( value = ast.Name( id = result_var.stem, ctx = ast.Load() ), attr = '_payload', ctx = ast.Load() ),
				attr = 'err', ctx = ast.Load(),
			) ], keywords = [],
		)
		ast.copy_location( err_expr, node )
		err_value = self._lower_expr( err_expr, outer_result_type )
		for instr in self._cfg.assign( dest_var, err_value, is_alias = False ):
			self._emit( instr )
		self._emit( ir.Assign( dest = dest_var, src = err_value ))
		self._emit( ir.Label( name = end_label ))
		return dest_var

	def _tagged_union_storage( self, union: TaggedUnion ) -> tuple[Variable,Variable,CUnion,dict[str,int]]:
		# every TaggedUnion (a user-declared @union class, or a synthesized
		# anonymous X|Y) gets a real runtime representation synthesized here
		# on first use: `tag: u8` (each member's ordinal, by declaration
		# order) + `data: <synthesized CUnion>` (one v_<member>-prefixed
		# field per member, only one ever meaningfully set at a time - the
		# v_ prefix avoids a member name colliding with something else in
		# that payload struct). This is exactly what builtins.Result already
		# hand-writes (_tag: u8 + _payload: ResultPayload[T,E]) - Result
		# itself stays hand-rolled (it's a @cstruct, not a TaggedUnion, and
		# match/dispatch special-case it directly - see _lower_or_return's
		# own comment on why compiler.early_return couldn't just be reused
		# for or_return() either), this generalizes the same shape for
		# every *real* union instead. Memoized in self._union_storage so
		# every reference (construction, match, dispatch, across unrelated
		# functions) points at the same tag/data/payload-class objects.
		cached = self._union_storage.get( id( union ) )
		if cached is not None:
			return cached
		self._ensure_resolved( union )
		for attr in union.attributes:
			self._ensure_resolved( attr )
		u8_cls = self.discovery.get_intrinsics()['u8']
		tag_attr = Variable( stem = 'tag', qualname = f'{union.qualname}.tag', file = union.file, line = union.line, type = u8_cls )
		payload_fields = [
			Variable( stem = f'v_{attr.stem}', qualname = f'{union.qualname}.data.v_{attr.stem}', file = attr.file, line = attr.line, type = attr.type )
			for attr in union.attributes
		]
		payload_cls = CUnion(
			stem = f'{union.stem}$data',
			qualname = f'{union.qualname}$data',
			file = union.file,
			line = union.line,
			attributes = payload_fields,
			names = { f.stem: f for f in payload_fields },
		)
		data_attr = Variable( stem = 'data', qualname = f'{union.qualname}.data', file = union.file, line = union.line, type = payload_cls )
		# register into union.names (NOT .attributes - that list backs
		# .leaves(), which must still only reflect the real union members
		# for overload/type matching) so ordinary GetAttr resolution
		# (_attr_lookup, used by _expr_Attribute for synthesized `subj.tag`/
		# `subj.data` AST) can actually find them
		for synthesized in ( tag_attr, data_attr ):
			existing = union.names.get( synthesized.stem )
			if existing is not None and existing is not synthesized:
				self.discovery.fail_loc(
					f'{union.qualname} already declares a member named {synthesized.stem!r}, which collides with the compiler-synthesized union storage field of the same name',
					union.file, union.line,
				)
			union.names[synthesized.stem] = synthesized
		tags = { attr.stem: i for i, attr in enumerate( union.attributes ) }
		result = ( tag_attr, data_attr, payload_cls, tags )
		self._union_storage[ id( union ) ] = result
		return result

	def _try_lower_union_construct_call( self, node: ast.Call, expected_type: Type|None ) -> ir.Temp|None:
		# TaggedUnionName.MemberName(value) - a compiler-synthesized
		# pseudo-constructor, same spirit as .__allocate__()/bare
		# ClassName(...): TaggedUnion members are plain Variables in
		# .names (discovery.py parses `SharedReference: T` inside a @union
		# body the same way it parses any CStruct/CUnion field), never real
		# declared Functions, so this can never be found via the ordinary
		# _resolve_callee/_attr_lookup_callable path either. Builds the
		# `tag`/`data` storage from _tagged_union_storage and emits two
		# Allocates: the payload union (one field set - v_<member>), then
		# the union instance itself (tag + data)
		if not isinstance( node.func, ast.Attribute ):
			return None
		union = self._try_resolve_namespace( node.func.value )
		if not isinstance( union, TaggedUnion ):
			return None
		self._ensure_resolved( union )
		member = next( ( attr for attr in union.attributes if attr.stem == node.func.attr ), None )
		if member is None:
			return None # not a real member name - fall through, let the normal call path report whatever error fits (e.g. "not callable")
		if len( node.args ) != 1 or node.keywords:
			self.discovery.fail( f'{union.qualname}.{member.stem}(...) takes exactly one positional argument: {ast.unparse(node)}', node )
		self._ensure_resolved( member )

		tag_attr, data_attr, payload_cls, tags = self._tagged_union_storage( union )
		value = self._lower_expr( node.args[0], member.type )
		# value.type, not member.type - see _lower_allocate_fields's identical comment:
		# a generic union's member type (e.g. OwnershipError[T]'s `SharedReference:
		# T`) stays an unsubstituted TypeVar when the union is referenced bare
		# (sys.OwnershipError, not sys.OwnershipError[bytearray]) - value.type is
		# always the operand's real, concrete type regardless
		for instr in self._cfg.field_value( value.type, value, is_alias = self._is_aliasing_expr( node.args[0] )):
			self._emit( instr )
		payload_dest = self._new_temp( payload_cls )
		self._emit( ir.Allocate( dest = payload_dest, cls = payload_cls, fields = { f'v_{member.stem}': value } ))

		dest = self._new_temp( expected_type or union )
		self._emit( ir.Allocate( dest = dest, cls = union, fields = { tag_attr.stem: ir.Const( type = tag_attr.type, value = tags[member.stem] ), data_attr.stem: payload_dest } ))
		return dest

	_OR_RETURN_ALTERNATIVES = 'or_return() always propagates the error to the caller - there is no other way for the enclosing function to receive it'

	def _lower_or_return( self, node: ast.Call, receiver: ir.Operand, want_result: bool ) -> ir.Operand|None:
		# <result_expr>.or_return() is recognized textually here rather than
		# ever actually calling Result.or_return's own declared body
		# (`if self.is_err(): compiler.early_return(self._payload.err)` /
		# `return self._payload.ok`) - that body is written as a spec of the
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
		if not ( isinstance( receiver.type, Specialization ) and len( receiver.type.args ) == 2 ):
			self.discovery.fail( f'or_return() receiver must be Result[_,_], got {receiver.type.qualname if receiver.type else "?"}', node )
		result_type, error_cls = receiver.type.args
		self._require_result_return( node, receiver.type.base, error_cls, self._OR_RETURN_ALTERNATIVES )
		unwrapped = self._consume_checked_result( receiver, result_type, extra = None )
		return unwrapped if want_result else None

	def _lower_generic_function_call( self, node: ast.Call, spec: Specialization, receiver: ir.Operand|None, expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# sys.alloc[u8](...) - explicit generic instantiation. Matches call
		# args against the MONOMORPHIZED signature (so a literal argument's
		# expected type is already concrete, e.g. usize for alloc[u8]'s
		# count - not the abstract, unsubstituted one)
		monomorphized = self._monomorphized_function( spec )
		positional, keyword = self._match_call_args( monomorphized, node )
		args = []
		for param, expr in positional:
			operand = self._lower_expr( expr, param.type )
			self._apply_move_hook( param, operand, monomorphized.qualname )
			args.append( operand )
		kwargs = {}
		for param, expr in keyword:
			operand = self._lower_expr( expr, param.type )
			self._apply_move_hook( param, operand, monomorphized.qualname )
			kwargs[param.stem] = operand
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
			self._unify_type_param( target, param.type, operand.type, bindings, node )
		for param, _expr in keyword:
			self._unify_type_param( target, param.type, kwargs[param.stem].type, bindings, node )

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

	def _unify_type_param( self, target: Function, declared: Type|None, actual: Type|None, bindings: dict[int,Type], node: ast.AST ) -> None:
		if declared is None or actual is None:
			return
		if any( declared is tv for tv in target.type_params or [] ):
			existing = bindings.get( id( declared ) )
			if existing is not None and existing is not actual:
				self.discovery.fail(
					f'{target.qualname}[...]: type parameter {declared.stem!r} is inferred as both '
					f'{existing.qualname} and {actual.qualname} by different arguments: {ast.unparse(node)}',
					node,
				)
			bindings[ id( declared ) ] = actual
			return
		if isinstance( declared, Specialization ) and isinstance( actual, Specialization ) and declared.base is actual.base:
			for d_arg, a_arg in zip( declared.args, actual.args ):
				self._unify_type_param( target, d_arg, a_arg, bindings, node )
		# else: this parameter position doesn't mention any of target's own
		# type params (a concrete parameter, or a nested type whose base
		# doesn't even match the argument's) - nothing to infer here. Not an
		# error by itself: a genuine argument-type mismatch isn't checked
		# anywhere yet (no general type-checking pass exists), same as
		# every other call site in this file today

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

	def _lower_call( self, node: ast.Call, expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		if self._is_compiler_sizeof_call( node ):
			result = self._lower_compiler_sizeof( node, expected_type )
			return result if want_result else None

		if self._is_compiler_refcount_call( node ):
			result = self._lower_compiler_refcount( node, expected_type )
			return result if want_result else None

		allocate_dest = self._try_lower_allocate_call( node, expected_type )
		if allocate_dest is None:
			allocate_dest = self._try_lower_construct_call( node, expected_type )
		if allocate_dest is None:
			allocate_dest = self._try_lower_union_construct_call( node, expected_type )
		if allocate_dest is not None:
			return allocate_dest if want_result else None

		target, receiver = self._resolve_callee( node.func )
		if receiver is not None:
			self.schedule( receiver.type )

		if isinstance( target, _ReceiverDispatch ):
			return self._lower_union_receiver_call( node, target, receiver, expected_type, want_result )

		if isinstance( target, Function ) and target.stem == 'or_return' and target.cls is self.discovery.find_name( 'Result', node ):
			return self._lower_or_return( node, receiver, want_result )

		if isinstance( target, Specialization ) and isinstance( target.base, Function ):
			return self._lower_generic_function_call( node, target, receiver, expected_type, want_result )

		if isinstance( target, Function ) and target.type_params:
			return self._lower_inferred_generic_call( node, target, receiver, expected_type, want_result )

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
			try:
				branches, resolved = overload_resolution.resolve_call( target.stubs, target.implementations, arg_types, kwarg_types, qualname = target.qualname )
			except CompileError as e:
				# resolve_call is a pure function of types with no
				# AST/Discovery reference by design - it raises unrecorded,
				# this is where a location actually gets attached and it
				# lands in the collector
				self.discovery.fail( str( e ), node )
			if branches:
				return self._lower_conditional_dispatch( node, branches, resolved, args, kwargs, expected_type, want_result )
			target = resolved
			self._ensure_resolved( target ) # resolve_call() already resolved every group member internally - this just schedules the chosen one
		else:
			self._ensure_resolved( target )
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
		# statements use (_tagged_union_storage) - branches are tried in
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
			if not isinstance( operand.type, TaggedUnion ):
				self.discovery.fail( f'{target.qualname}: conditional dispatch on a non-union argument: {ast.unparse(node)}', node )
			member = next( ( attr for attr in operand.type.attributes if attr.type is leaf_type ), None )
			if member is None:
				self.discovery.fail( f'{target.qualname}: {leaf_type.qualname if leaf_type else "?"} is not a member of {operand.type.qualname}', node )
			tag_attr, _data_attr, _payload_cls, tags = self._tagged_union_storage( operand.type )
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
		if target_type is None or operand.type is target_type or not isinstance( operand.type, TaggedUnion ):
			return operand
		member = next( ( attr for attr in operand.type.attributes if attr.type is target_type ), None )
		if member is None:
			return operand
		tag_attr, data_attr, payload_cls, tags = self._tagged_union_storage( operand.type )
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
		# match statements and dispatch already use (_tagged_union_storage),
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

		tag_attr, data_attr, payload_cls, tags = self._tagged_union_storage( dispatch.union )
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
