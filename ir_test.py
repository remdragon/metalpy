# stdlib imports:
import ast
import unittest

# local imports:
import ir
from mpy_types import Scalar, RCClass, CStruct, Specialization, Parameter, Variable, Function

def _scalar( stem: str, qualname: str|None = None ) -> Scalar:
	return Scalar( stem = stem, qualname = qualname or f'intrinsics.{stem}', file = None, line = None )

def _fn_node() -> ast.FunctionDef:
	return ast.parse( 'def f(): pass' ).body[0]

def _make_function( stem: str, qualname: str, parameters: list[Parameter], return_type: Scalar ) -> Function:
	return Function(
		cls = None,
		node = _fn_node(),
		stem = stem,
		qualname = qualname,
		file = None,
		line = None,
		parameters = parameters,
		return_type = return_type,
	)

class IRTestCase( unittest.TestCase ):
	def setUp( self ) -> None:
		self.i32 = _scalar( 'i32' )
		self.str_type = _scalar( 'str', 'builtins.str' )
		self.none_type = _scalar( 'NoneType' )
		self.bool_type = _scalar( 'bool', 'builtins.bool' )
		self.overflow_error = RCClass( stem = 'OverflowError', qualname = 'builtins.OverflowError', file = None, line = None )
		self.zero_division_error = RCClass( stem = 'ZeroDivisionError', qualname = 'builtins.ZeroDivisionError', file = None, line = None )
		self.result_base = CStruct( stem = 'Result', qualname = 'builtins.Result', file = None, line = None )
		self.result_i32_overflow = Specialization(
			stem = 'Result',
			qualname = 'builtins.Result[intrinsics.i32,builtins.OverflowError]',
			file = None,
			line = None,
			base = self.result_base,
			args = [ self.i32, self.overflow_error ],
		)
		self.result_i32_zerodiv = Specialization(
			stem = 'Result',
			qualname = 'builtins.Result[intrinsics.i32,builtins.ZeroDivisionError]',
			file = None,
			line = None,
			base = self.result_base,
			args = [ self.i32, self.zero_division_error ],
		)

class ArchitectureExampleTests( IRTestCase ):
	''' hand-builds the exact foo(x: i32) -> None: print(x + 1) sequence from ARCHITECTURE.md '''

	def test_foo_sequence( self ) -> None:
		x = Parameter( stem = 'x', qualname = '__main__.foo.x', file = None, line = None, type = self.i32 )
		print_fn = _make_function( 'print', 'builtins.print', [ Parameter( stem = 'msg', qualname = 'builtins.print.msg', file = None, line = None, type = self.i32 ) ], self.none_type )
		t0 = ir.Temp( type = self.i32, id = 0 )

		sequence = [
			ir.FuncStart( name = 'foo', params = [ x ], return_type = self.none_type ),
			ir.DeclareTemp( temp = t0 ),
			ir.AddWrap( dest = t0, left = x, right = ir.Const( type = self.i32, value = 1 )),
			ir.Call( dest = None, target = print_fn, args = [ t0 ], kwargs = {} ),
			ir.DeleteTemp( temp = t0 ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'foo' ),
		]

		expected = [
			ir.FuncStart( name = 'foo', params = [ x ], return_type = self.none_type ),
			ir.DeclareTemp( temp = ir.Temp( type = self.i32, id = 0 )),
			ir.AddWrap( dest = ir.Temp( type = self.i32, id = 0 ), left = x, right = ir.Const( type = self.i32, value = 1 )),
			ir.Call( dest = None, target = print_fn, args = [ ir.Temp( type = self.i32, id = 0 ) ], kwargs = {} ),
			ir.DeleteTemp( temp = ir.Temp( type = self.i32, id = 0 )),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'foo' ),
		]

		self.assertEqual( sequence, expected )
		self.assertIsInstance( sequence[0], ir.FuncStart )
		self.assertEqual( sequence[0].params, [ x ] )
		self.assertIsInstance( sequence[2], ir.AddWrap )
		self.assertIs( sequence[2].left, x )

class BinOpFamilyTests( IRTestCase ):
	''' every op x mode combination shares the BinOp shape (dest, left, right) - confirm equality is class-sensitive '''

	BINOPS = [
		ir.AddWrap, ir.AddCheck, ir.AddSaturate,
		ir.SubWrap, ir.SubCheck, ir.SubSaturate,
		ir.MulWrap, ir.MulCheck, ir.MulSaturate,
		ir.ShlWrap, ir.ShlCheck, ir.ShlSaturate,
		ir.Div, ir.Mod,
		ir.BitAnd, ir.BitOr, ir.BitXor, ir.Shr,
	]

	def test_same_class_same_fields_equal( self ) -> None:
		for cls in self.BINOPS:
			with self.subTest( cls = cls.__name__ ):
				t0 = ir.Temp( type = self.i32, id = 0 )
				a = cls( dest = t0, left = ir.Const( type = self.i32, value = 1 ), right = ir.Const( type = self.i32, value = 2 ))
				b = cls( dest = ir.Temp( type = self.i32, id = 0 ), left = ir.Const( type = self.i32, value = 1 ), right = ir.Const( type = self.i32, value = 2 ))
				self.assertEqual( a, b )

	def test_different_class_same_fields_not_equal( self ) -> None:
		t0 = ir.Temp( type = self.i32, id = 0 )
		left = ir.Const( type = self.i32, value = 1 )
		right = ir.Const( type = self.i32, value = 2 )
		instances = [ cls( dest = t0, left = left, right = right ) for cls in self.BINOPS ]
		for i, a in enumerate( instances ):
			for j, b in enumerate( instances ):
				if i != j:
					self.assertNotEqual( a, b )

	def test_check_dest_is_overflow_result( self ) -> None:
		for cls in ( ir.AddCheck, ir.SubCheck, ir.MulCheck, ir.ShlCheck ):
			with self.subTest( cls = cls.__name__ ):
				dest = ir.Temp( type = self.result_i32_overflow, id = 0 )
				instr = cls( dest = dest, left = ir.Const( type = self.i32, value = 1 ), right = ir.Const( type = self.i32, value = 2 ))
				self.assertIs( instr.dest.type, self.result_i32_overflow )

	def test_div_mod_dest_is_zero_division_result( self ) -> None:
		# Div/Mod are always checked, but against zero rather than overflow,
		# so they get their own error type matching Python's own naming
		for cls in ( ir.Div, ir.Mod ):
			with self.subTest( cls = cls.__name__ ):
				dest = ir.Temp( type = self.result_i32_zerodiv, id = 0 )
				instr = cls( dest = dest, left = ir.Const( type = self.i32, value = 1 ), right = ir.Const( type = self.i32, value = 2 ))
				self.assertIs( instr.dest.type, self.result_i32_zerodiv )
				self.assertIsNot( instr.dest.type, self.result_i32_overflow )

class UnaryOpFamilyTests( IRTestCase ):
	UNARYOPS = [ ir.Invert, ir.NegWrap, ir.NegCheck, ir.NegSaturate ]

	def test_same_class_same_fields_equal( self ) -> None:
		for cls in self.UNARYOPS:
			with self.subTest( cls = cls.__name__ ):
				a = cls( dest = ir.Temp( type = self.i32, id = 0 ), operand = ir.Const( type = self.i32, value = 1 ))
				b = cls( dest = ir.Temp( type = self.i32, id = 0 ), operand = ir.Const( type = self.i32, value = 1 ))
				self.assertEqual( a, b )

	def test_different_class_same_fields_not_equal( self ) -> None:
		operand = ir.Const( type = self.i32, value = 1 )
		instances = [ cls( dest = ir.Temp( type = self.i32, id = 0 ), operand = operand ) for cls in self.UNARYOPS ]
		for i, a in enumerate( instances ):
			for j, b in enumerate( instances ):
				if i != j:
					self.assertNotEqual( a, b )

class ResultConsumingOpsTests( IRTestCase ):
	''' Check-mode arithmetic produces a Result[T,OverflowError]; OrReturn/Unwrap/UnwrapOr consume it '''

	def test_check_then_consume_sequence( self ) -> None:
		x = Parameter( stem = 'x', qualname = '__main__.foo.x', file = None, line = None, type = self.i32 )
		result_temp = ir.Temp( type = self.result_i32_overflow, id = 0 )
		unwrapped = ir.Temp( type = self.i32, id = 1 )

		add_check = ir.AddCheck( dest = result_temp, left = x, right = ir.Const( type = self.i32, value = 1 ))
		or_return = ir.OrReturn( dest = unwrapped, value = result_temp )
		unwrap = ir.Unwrap( dest = unwrapped, value = result_temp, errmsg = ir.Const( type = self.str_type, value = 'overflow' ))
		unwrap_or = ir.UnwrapOr( dest = unwrapped, value = result_temp, default = ir.Const( type = self.i32, value = 0 ))

		self.assertIs( add_check.dest.type, self.result_i32_overflow )
		self.assertIs( or_return.value, result_temp )
		self.assertIs( or_return.dest, unwrapped )
		self.assertEqual( unwrap.errmsg, ir.Const( type = self.str_type, value = 'overflow' ))
		self.assertEqual( unwrap_or.default, ir.Const( type = self.i32, value = 0 ))

		# two independently-built consuming instructions over the same Result compare equal
		self.assertEqual(
			ir.Unwrap( dest = unwrapped, value = result_temp, errmsg = ir.Const( type = self.str_type, value = 'overflow' )),
			unwrap,
		)

class CmpOpTests( IRTestCase ):
	def test_each_cmp_op_round_trips( self ) -> None:
		for op in ir.CmpOp:
			with self.subTest( op = op ):
				dest = ir.Temp( type = self.bool_type, id = 0 )
				a = ir.Cmp( dest = dest, op = op, left = ir.Const( type = self.i32, value = 1 ), right = ir.Const( type = self.i32, value = 2 ))
				b = ir.Cmp( dest = ir.Temp( type = self.bool_type, id = 0 ), op = op, left = ir.Const( type = self.i32, value = 1 ), right = ir.Const( type = self.i32, value = 2 ))
				self.assertEqual( a, b )
				self.assertIs( a.op, op )

	def test_different_ops_not_equal( self ) -> None:
		dest = ir.Temp( type = self.bool_type, id = 0 )
		left = ir.Const( type = self.i32, value = 1 )
		right = ir.Const( type = self.i32, value = 2 )
		instances = [ ir.Cmp( dest = dest, op = op, left = left, right = right ) for op in ir.CmpOp ]
		for i, a in enumerate( instances ):
			for j, b in enumerate( instances ):
				if i != j:
					self.assertNotEqual( a, b )

class OperandVarietyTests( IRTestCase ):
	''' Temp/Const/Variable all satisfy Operand positionally - nothing about the dataclasses forces one specific kind '''

	def test_temp_operand( self ) -> None:
		dest = ir.Temp( type = self.i32, id = 1 )
		src = ir.Temp( type = self.i32, id = 0 )
		instr = ir.Assign( dest = dest, src = src )
		self.assertIsInstance( instr.src, ir.Temp )

	def test_const_operand( self ) -> None:
		dest = ir.Temp( type = self.i32, id = 0 )
		instr = ir.Assign( dest = dest, src = ir.Const( type = self.i32, value = 42 ))
		self.assertIsInstance( instr.src, ir.Const )

	def test_variable_operand( self ) -> None:
		dest = ir.Temp( type = self.i32, id = 0 )
		var = Variable( stem = 'g', qualname = '__main__.g', file = None, line = None, type = self.i32 )
		instr = ir.Assign( dest = dest, src = var )
		self.assertIsInstance( instr.src, Variable )

class ControlFlowTests( IRTestCase ):
	def test_label_jump_jumpif( self ) -> None:
		cond = ir.Temp( type = self.bool_type, id = 0 )
		sequence = [
			ir.JumpIfFalse( cond = cond, target = 'else' ),
			ir.Jump( target = 'end' ),
			ir.Label( name = 'else' ),
			ir.Label( name = 'end' ),
		]
		self.assertEqual( sequence[0], ir.JumpIfFalse( cond = cond, target = 'else' ))
		self.assertNotEqual( sequence[0], ir.JumpIfTrue( cond = cond, target = 'else' ))

class AllocateRefcountTests( IRTestCase ):
	def test_allocate_and_refcount( self ) -> None:
		cls = CStruct( stem = 'Foo', qualname = '__main__.Foo', file = None, line = None )
		dest = ir.Temp( type = cls, id = 0 )
		alloc = ir.Allocate( dest = dest, cls = cls, fields = { 'x': ir.Const( type = self.i32, value = 1 ) } )
		self.assertEqual( alloc.fields, { 'x': ir.Const( type = self.i32, value = 1 ) } )

		incref = ir.Incref( value = dest )
		decref = ir.Decref( value = dest )
		self.assertNotEqual( incref, decref )
		self.assertIs( incref.value, dest )

if __name__ == '__main__':
	unittest.main()
