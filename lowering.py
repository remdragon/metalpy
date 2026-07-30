# stdlib imports:
import ast
from contextlib import nullcontext
from typing import Callable

# local imports:
import ir
from discovery import Discovery
from mpy_types import (
	Type, Variable, Parameter, Function, Overload, ClassLike, Module,
	Specialization, TaggedUnion,
)

_BINOP_OPCODES: dict[type,type] = {
	ast.Add: ir.AddWrap,
	ast.Sub: ir.SubWrap,
	ast.Mult: ir.MulWrap,
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

	Whenever a Function, ClassLike, or module-level Variable is discovered as
	a dependency, `schedule` is called immediately at the point of discovery -
	there's no separate dependency-scanning pass.
	'''

	def __init__( self, discovery: Discovery, schedule: Callable[[Function|ClassLike|Variable],None] ) -> None:
		self.discovery = discovery
		self.schedule = schedule

	def lower_function( self, fn: Function ) -> list[ir.Instruction]:
		module = self._find_module_for( fn )
		self._instructions: list[ir.Instruction] = []
		self._temp_id = 0
		self._pending_temps: list[ir.Temp] = []
		self._current_fn = fn

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
						self._schedule_type_deps( param.type )
					self._schedule_type_deps( fn.return_type )

					self._emit( ir.FuncStart( name = fn.qualname, params = fn.parameters or [], return_type = fn.return_type ))
					for stmt in fn.node.body:
						self._lower_stmt( stmt )
					self._emit( ir.FuncEnd( name = fn.qualname ))

		return self._instructions

	def lower_global( self, var: Variable ) -> list[ir.Instruction]:
		module = self._find_module_for( var )
		self._instructions = []
		self._temp_id = 0
		self._pending_temps = []
		self._current_fn = None

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
		assert False, f'no module found owning {unit.qualname} (file={unit.file})'

	# --- dependency scheduling -----------------------------------------------

	def _schedule_type_deps( self, t: Type|None ) -> None:
		if t is None:
			return
		if isinstance( t, ClassLike ):
			self.schedule( t )
		elif isinstance( t, Specialization ):
			self._schedule_type_deps( t.base )
			for arg in t.args:
				self._schedule_type_deps( arg )
		elif isinstance( t, TaggedUnion ):
			for leaf in t.leaves():
				self._schedule_type_deps( leaf )

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
		self._pending_temps = []
		method = getattr( self, f'_stmt_{node.__class__.__name__}', None )
		assert method is not None, f'unsupported statement: {ast.unparse(node)}'
		method( node )
		for t in reversed( self._pending_temps ):
			self._emit( ir.DeleteTemp( temp = t ))

	def _stmt_Return( self, node: ast.Return ) -> None:
		value = self._lower_expr( node.value, self._current_fn.return_type ) if node.value is not None else None
		self._emit( ir.Return( value = value ))

	def _stmt_Pass( self, node: ast.Pass ) -> None:
		pass

	def _stmt_Global( self, node: ast.Global ) -> None:
		# a no-op: this language requires an explicit AnnAssign to introduce a
		# new local, so an unannotated Assign to a name never shadows a global
		# in the first place - find_name's scope chain already falls through
		# to the module scope on its own
		pass

	def _stmt_AnnAssign( self, node: ast.AnnAssign ) -> None:
		assert isinstance( node.target, ast.Name ), f'unsupported AnnAssign target: {ast.unparse(node)}'
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
		self._schedule_type_deps( var_type )
		if node.value is not None:
			operand = self._lower_expr( node.value, var_type )
			self._emit( ir.Assign( dest = var, src = operand ))

	def _stmt_Assign( self, node: ast.Assign ) -> None:
		assert len( node.targets ) == 1, f'multiple assignment targets not supported: {ast.unparse(node)}'
		target = node.targets[0]
		if isinstance( target, ast.Name ):
			existing = self.discovery.find_name( target.id, node )
			assert isinstance( existing, Variable ), f'{target.id!r} is not a variable, cannot assign to it'
			operand = self._lower_expr( node.value, existing.type )
			self._emit( ir.Assign( dest = existing, src = operand ))
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
			assert False, f'unsupported Assign target: {ast.unparse(node)}'

	def _stmt_Expr( self, node: ast.Expr ) -> None:
		assert isinstance( node.value, ast.Call ), f'unsupported expression statement: {ast.unparse(node)}'
		self._lower_call( node.value, None, want_result = False )

	# --- expressions -----------------------------------------------------------

	def _lower_expr( self, node: ast.expr, expected_type: Type|None ) -> ir.Operand:
		method = getattr( self, f'_expr_{node.__class__.__name__}', None )
		assert method is not None, f'unsupported expression: {ast.unparse(node)}'
		return method( node, expected_type )

	def _expr_Name( self, node: ast.Name, expected_type: Type|None ) -> ir.Operand:
		name = self.discovery.find_name( node.id, node )
		assert isinstance( name, Variable ), f'{node.id!r} is not a value, cannot use it as an expression'
		return name

	def _expr_Constant( self, node: ast.Constant, expected_type: Type|None ) -> ir.Operand:
		assert expected_type is not None, (
			f'cannot infer the type of literal {node.value!r} - no expected type available from context ({ast.unparse(node)})'
		)
		return ir.Const( type = expected_type, value = node.value )

	def _expr_Attribute( self, node: ast.Attribute, expected_type: Type|None ) -> ir.Operand:
		obj = self._lower_expr( node.value, None )
		attr_var = self._attr_lookup( obj.type, node.attr, node )
		dest = self._new_temp( attr_var.type )
		self._emit( ir.GetAttr( dest = dest, obj = obj, attr = node.attr ))
		return dest

	def _expr_Subscript( self, node: ast.Subscript, expected_type: Type|None ) -> ir.Operand:
		assert expected_type is not None, f'cannot infer the result type of {ast.unparse(node)} - no expected type available from context'
		obj = self._lower_expr( node.value, None )
		index = self._lower_expr( node.slice, None )
		dest = self._new_temp( expected_type )
		self._emit( ir.GetItem( dest = dest, obj = obj, index = index ))
		return dest

	def _expr_Call( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		return self._lower_call( node, expected_type, want_result = True )

	def _expr_BinOp( self, node: ast.BinOp, expected_type: Type|None ) -> ir.Operand:
		opcode = _BINOP_OPCODES.get( type( node.op ))
		assert opcode is not None, f'unsupported binary operator: {ast.unparse(node)}'

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

		dest = self._new_temp( expected_type or left.type )
		self._emit( opcode( dest = dest, left = left, right = right ))
		return dest

	# --- shared helpers ----------------------------------------------------------

	def _attr_lookup( self, owner_type: Type|None, attr: str, ctx: ast.AST ) -> Variable:
		names = getattr( owner_type, 'names', None )
		assert isinstance( names, dict ), f'{owner_type!r} has no members, cannot look up {attr!r} ({ast.unparse(ctx)})'
		found = names.get( attr )
		assert isinstance( found, Variable ), f'{owner_type.qualname if owner_type else "?"} has no attribute {attr!r}'
		if found.resolve is not None:
			found.resolve()
		return found

	def _resolve_callee( self, func_node: ast.expr ) -> tuple[Function|Overload,ir.Operand|None]:
		# try resolving the whole callee expression as a compile-time namespace
		# path first (a free function, or Class.staticmethod/classmethod
		# reached by class name) - this is exactly discovery.py's own
		# visit_Name/visit_Attribute (find_name + .names traversal), which
		# raises AssertionError the moment it hits something with no .names
		# (a Variable/Parameter - a runtime value, not a namespace)
		try:
			namespace_result = self.discovery.visit( func_node )
		except AssertionError:
			namespace_result = None
		if isinstance( namespace_result, ( Function, Overload )):
			return namespace_result, None

		assert isinstance( func_node, ast.Attribute ), f'cannot call {ast.unparse(func_node)}'
		receiver = self._lower_expr( func_node.value, None )
		target = self._attr_lookup_callable( receiver.type, func_node.attr, func_node )
		return target, receiver

	def _attr_lookup_callable( self, owner_type: Type|None, attr: str, ctx: ast.AST ) -> Function|Overload:
		names = getattr( owner_type, 'names', None )
		assert isinstance( names, dict ), f'{owner_type!r} has no members, cannot look up {attr!r} ({ast.unparse(ctx)})'
		found = names.get( attr )
		assert isinstance( found, ( Function, Overload )), f'{attr!r} is not callable on {owner_type.qualname if owner_type else "?"}'
		return found

	def _match_call_args( self, target: Function, call: ast.Call ) -> tuple[list[tuple[Parameter,ast.expr]],list[tuple[Parameter,ast.expr]]]:
		assert not any( isinstance( a, ast.Starred ) for a in call.args ), f'*args not supported yet: {ast.unparse(call)}'
		assert not any( kw.arg is None for kw in call.keywords ), f'**kwargs not supported yet: {ast.unparse(call)}'
		positional_params = [ p for p in target.parameters if not p.is_vararg and not p.is_kwarg and not p.is_kwonly ]
		assert len( call.args ) <= len( positional_params ), f'too many positional arguments: {ast.unparse(call)}'
		positional = list( zip( positional_params, call.args ))
		keyword: list[tuple[Parameter,ast.expr]] = []
		for kw in call.keywords:
			param = next(( p for p in target.parameters if p.stem == kw.arg and not p.is_vararg and not p.is_kwarg ), None )
			assert param is not None, f'{target.qualname} has no parameter {kw.arg!r}'
			keyword.append(( param, kw.value ))
		return positional, keyword

	def _lower_call( self, node: ast.Call, expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		target, receiver = self._resolve_callee( node.func )
		if receiver is not None:
			self._schedule_type_deps( receiver.type )

		if isinstance( target, Overload ):
			# bare literal arguments have no unambiguous expected type before
			# a specific implementation is chosen - unsupported for now (see
			# _expr_Constant's assertion), same posture as the multi-branch
			# case below
			args = [ self._lower_expr( a, None ) for a in node.args ]
			assert not any( kw.arg is None for kw in node.keywords ), f'**kwargs not supported yet: {ast.unparse(node)}'
			kwargs = { kw.arg: self._lower_expr( kw.value, None ) for kw in node.keywords }
			arg_types = [ op.type for op in args ]
			kwarg_types = { name: op.type for name, op in kwargs.items() }
			branches, resolved = target.resolve_call( arg_types, kwarg_types )
			assert not branches, (
				f'{target.qualname}: multi-branch overload dispatch is not supported yet ({ast.unparse(node)}) - '
				f'this needs TaggedUnion tag-check IR, which does not exist yet'
			)
			target = resolved
		else:
			if target.resolve is not None:
				target.resolve()
			positional, keyword = self._match_call_args( target, node )
			args = [ self._lower_expr( expr, param.type ) for param, expr in positional ]
			kwargs = { param.stem: self._lower_expr( expr, param.type ) for param, expr in keyword }

		self._schedule_type_deps( target.return_type )
		for param in target.parameters or []:
			self._schedule_type_deps( param.type )
		self.schedule( target )

		if want_result:
			dest = self._new_temp( expected_type or target.return_type )
			self._emit( ir.Call( dest = dest, target = target, receiver = receiver, args = args, kwargs = kwargs ))
			return dest
		else:
			self._emit( ir.Call( dest = None, target = target, receiver = receiver, args = args, kwargs = kwargs ))
			return None
