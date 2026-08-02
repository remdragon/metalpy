# stdlib imports:
import ast
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable

# local imports:
import ir
from discovery import Discovery
from errors import CompileError
from mpy_types import (
	Name, Type, Variable, Parameter, Function, Overload, ClassLike, Module,
	Specialization,
)

@dataclass( kw_only = True )
class _DeferBlock:
	is_err_only: bool # True for errdefer, False for plain defer
	flag: Variable # bool local, False until control passes the defer/errdefer statement
	instructions: list['ir.Instruction'] # captured at registration time, replayed in the epilogue

_BINOP_WRAP_OPCODES: dict[type,type] = {
	ast.Add: ir.AddWrap,
	ast.Sub: ir.SubWrap,
	ast.Mult: ir.MulWrap,
}
_BINOP_CHECK_OPCODES: dict[type,type] = {
	ast.Add: ir.AddCheck,
	ast.Sub: ir.SubCheck,
	ast.Mult: ir.MulCheck,
}
_BINOP_SATURATE_OPCODES: dict[type,type] = {
	ast.Add: ir.AddSaturate,
	ast.Sub: ir.SubSaturate,
	ast.Mult: ir.MulSaturate,
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

	def lower_function( self, fn: Function ) -> list[ir.Instruction]:
		module = self._find_module_for( fn )
		self._instructions: list[ir.Instruction] = []
		self._temp_id = 0
		self._pending_temps: list[ir.Temp] = []
		self._current_fn = fn
		self._arithmetic_mode: list[tuple[str,object]] = [ ( 'check', None ) ]
		self._loop_depth = 0
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
					if fn.cls is not None and not fn.is_static and not fn.is_classmethod:
						self_param = Parameter( stem = 'self', qualname = f'{fn.qualname}.self', file = fn.file, line = fn.line, type = fn.cls )
						fn.add_name( 'self', self_param )

					for param in fn.parameters or []:
						self.schedule( param.type )
					self.schedule( fn.return_type )

					none_type = self.discovery.get_none_type()
					self._return_value_var = (
						Variable( stem = '__return_value', qualname = f'{fn.qualname}.__return_value', file = fn.file, line = fn.line, type = fn.return_type )
						if self._needs_epilogue and fn.return_type is not none_type
						else None
					)

					self._emit( ir.FuncStart( name = fn.qualname, params = fn.parameters or [], return_type = fn.return_type ))
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
						self._emit_epilogue( fn, none_type, body_start )

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
			self._emit( ir.DeleteTemp( temp = t ))
		return_value = self._return_value_var if fn.return_type is not none_type else None
		self._emit( ir.Return( value = return_value ))

	def _emit_is_err_check( self, node: ast.AST ) -> ir.Temp:
		bool_cls = self.discovery.find_name( 'bool', node )
		is_err_fn = self._attr_lookup_callable( self._return_value_var.type, 'is_err', node )
		self._ensure_resolved( is_err_fn )
		dest = self._new_temp( bool_cls )
		self._emit( ir.Call( dest = dest, target = is_err_fn, receiver = self._return_value_var, args = [], kwargs = {} ))
		return dest

	def lower_global( self, var: Variable ) -> list[ir.Instruction]:
		module = self._find_module_for( var )
		self._instructions = []
		self._temp_id = 0
		self._pending_temps = []
		self._current_fn = None
		self._arithmetic_mode = [ ( 'check', None ) ]
		# defer/errdefer/loops can't appear in a global initializer (it's a
		# single expression, not a statement body reachable through
		# _lower_stmt) - reset for consistency/safety only, never touched here
		self._loop_depth = 0
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

	def _find_module_for( self, unit: Function|Variable ) -> Module:
		# Function/Variable.file is always set to their owning module's .file
		# (see discovery.py's _parse_function/visit_AnnAssign/visit_Assign) -
		# neither retains a direct back-reference to the Module itself
		for module in self.discovery.modules.values():
			if module.file == unit.file:
				return module
		self.discovery.fail_loc( f'no module found owning {unit.qualname} (file={unit.file})', unit.file, unit.line )

	# --- temp/instruction bookkeeping ----------------------------------------

	def _emit( self, instr: ir.Instruction ) -> None:
		self._instructions.append( instr )

	def _new_temp( self, t: Type ) -> ir.Temp:
		temp = ir.Temp( type = t, id = self._temp_id )
		self._temp_id += 1
		self._pending_temps.append( temp )
		self._emit( ir.DeclareTemp( temp = temp ))
		return temp

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
				self._emit( ir.DeleteTemp( temp = t ))
		finally:
			self._pending_temps = outer_pending

	def _stmt_Return( self, node: ast.Return ) -> None:
		value = self._lower_expr( node.value, self._current_fn.return_type ) if node.value is not None else None
		if self._needs_epilogue:
			if self._return_value_var is not None and value is not None:
				self._emit( ir.Assign( dest = self._return_value_var, src = value ))
			self._emit( ir.Jump( target = self._epilogue_label ))
		else:
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
				self._emit( ir.Assign( dest = var, src = operand ))
		elif isinstance( target, ast.Attribute ):
			obj = self._lower_expr( target.value, None )
			attr_var = self._attr_lookup( obj.type, target.attr, target )
			operand = self._lower_expr( node.value, attr_var.type )
			self._emit( ir.SetAttr( obj = obj, attr = target.attr, value = operand ))
		elif isinstance( target, ast.Subscript ):
			obj = self._lower_expr( target.value, None )
			index = self._lower_expr( target.slice, None )
			operand = self._lower_expr( node.value, None )
			self._emit( ir.SetItem( obj = obj, index = index, value = operand ))
		else:
			self.discovery.fail( f'unsupported Assign target: {ast.unparse(node)}', node )

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

	# --- loops (recognition only - no control-flow IR yet) -----------------------

	def _stmt_For( self, node: ast.For ) -> None:
		self._lower_loop_body( node.body )
		self.discovery.fail( 'for loop control flow is not yet supported', node )

	def _stmt_While( self, node: ast.While ) -> None:
		self._lower_loop_body( node.body )
		self.discovery.fail( 'while loop control flow is not yet supported', node )

	def _lower_loop_body( self, body: list[ast.stmt] ) -> None:
		# pushed so nested defer/errdefer (at any depth) gets rejected, and
		# the body still lowers through the normal statement machinery - so
		# this is reusable once real loop control-flow IR exists, it's only
		# the loop's own iteration/branching that's missing
		self._loop_depth += 1
		try:
			for stmt in body:
				try:
					self._lower_stmt( stmt )
				except CompileError:
					continue
		finally:
			self._loop_depth -= 1

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

	def _expr_Subscript( self, node: ast.Subscript, expected_type: Type|None ) -> ir.Operand:
		if expected_type is None:
			self.discovery.fail( f'cannot infer the result type of {ast.unparse(node)} - no expected type available from context', node )
		obj = self._lower_expr( node.value, None )
		index = self._lower_expr( node.slice, None )
		dest = self._new_temp( expected_type )
		self._emit( ir.GetItem( dest = dest, obj = obj, index = index ))
		return dest

	def _expr_Call( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		return self._lower_call( node, expected_type, want_result = True )

	_OPCODES_BY_KIND = {
		'wrap': _BINOP_WRAP_OPCODES,
		'saturate': _BINOP_SATURATE_OPCODES,
		'check': _BINOP_CHECK_OPCODES,
	}

	def _expr_BinOp( self, node: ast.BinOp, expected_type: Type|None ) -> ir.Operand:
		kind, extra = self._arithmetic_mode[-1]
		opcode = self._OPCODES_BY_KIND[kind].get( type( node.op ))
		if opcode is None:
			self.discovery.fail( f'unsupported binary operator: {ast.unparse(node)}', node )

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
		result_cls, overflow_cls = self._lookup_result_and_overflow_types( node )
		if extra is None:
			# validated before anything gets emitted - a mid-statement
			# failure here must not leave partial instructions behind for
			# the per-statement recovery boundary to silently keep
			self._require_result_return( node, result_cls, overflow_cls )
		check_type = self.discovery._get_or_create_specialization( result_cls, [ result_type, overflow_cls ] )
		check_dest = self._new_temp( check_type )
		self._emit( opcode( dest = check_dest, left = left, right = right ))
		unwrapped = self._new_temp( result_type )
		if extra is None:
			if self._needs_epilogue:
				self._emit( ir.OrJump( dest = unwrapped, value = check_dest, target = self._epilogue_label, return_slot = self._return_value_var ))
			else:
				self._emit( ir.OrReturn( dest = unwrapped, value = check_dest ))
		else:
			self._emit( ir.Unwrap( dest = unwrapped, value = check_dest, errmsg = extra ))
		return unwrapped

	def _lookup_result_and_overflow_types( self, node: ast.AST ) -> tuple[ClassLike,ClassLike]:
		result_cls = self.discovery.find_name( 'Result', node )
		overflow_cls = self.discovery.find_name( 'OverflowError', node )
		return result_cls, overflow_cls

	def _require_result_return( self, node: ast.AST, result_cls: ClassLike, overflow_cls: ClassLike ) -> None:
		fn = self._current_fn
		return_type = fn.return_type if fn is not None else None
		ok = (
			fn is not None
			and isinstance( return_type, Specialization )
			and return_type.base is result_cls
			and len( return_type.args ) == 2
			and return_type.args[1] is overflow_cls
		)
		if not ok:
			where = f'{fn.qualname} returns {return_type.qualname if return_type else None}' if fn is not None else 'this is not inside a function'
			self.discovery.fail(
				f'checked arithmetic requires the enclosing function to return Result[_,OverflowError] ({where}) - '
				f'wrap this in `with compiler.wrap_arithmetic:`, `with compiler.saturate_arithmetic:`, '
				f'or `with compiler.panic_arithmetic(...):` instead',
				node,
			)

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
		return found

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
		return None

	def _resolve_callee( self, func_node: ast.expr ) -> tuple[Function|Overload,ir.Operand|None]:
		namespace_result = self._try_resolve_namespace( func_node )
		if isinstance( namespace_result, ( Function, Overload )):
			return namespace_result, None

		if not isinstance( func_node, ast.Attribute ):
			self.discovery.fail( f'cannot call {ast.unparse(func_node)}', func_node )
		receiver = self._lower_expr( func_node.value, None )
		target = self._attr_lookup_callable( receiver.type, func_node.attr, func_node )
		return target, receiver

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
		return positional, keyword

	def _lower_call( self, node: ast.Call, expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		target, receiver = self._resolve_callee( node.func )
		if receiver is not None:
			self.schedule( receiver.type )

		if isinstance( target, Overload ):
			# bare literal arguments have no unambiguous expected type before
			# a specific implementation is chosen - unsupported for now (see
			# _expr_Constant), same posture as the multi-branch case below
			args = [ self._lower_expr( a, None ) for a in node.args ]
			if any( kw.arg is None for kw in node.keywords ):
				self.discovery.fail( f'**kwargs not supported yet: {ast.unparse(node)}', node )
			kwargs = { kw.arg: self._lower_expr( kw.value, None ) for kw in node.keywords }
			arg_types = [ op.type for op in args ]
			kwarg_types = { name: op.type for name, op in kwargs.items() }
			try:
				branches, resolved = target.resolve_call( arg_types, kwarg_types )
			except CompileError as e:
				# resolve_call is a pure function of types with no
				# AST/Discovery reference by design - it raises unrecorded,
				# this is where a location actually gets attached and it
				# lands in the collector
				self.discovery.fail( str( e ), node )
			if branches:
				self.discovery.fail(
					f'{target.qualname}: multi-branch overload dispatch is not supported yet ({ast.unparse(node)}) - '
					f'this needs TaggedUnion tag-check IR, which does not exist yet',
					node,
				)
			target = resolved
			self._ensure_resolved( target ) # resolve_call() already resolved every group member internally - this just schedules the chosen one
		else:
			self._ensure_resolved( target )
			positional, keyword = self._match_call_args( target, node )
			args = [ self._lower_expr( expr, param.type ) for param, expr in positional ]
			kwargs = { param.stem: self._lower_expr( expr, param.type ) for param, expr in keyword }

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
