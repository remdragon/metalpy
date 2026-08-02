# stdlib imports:
import logging
from pathlib import Path
import unittest

# local imports:
from compiler import Compiler, LoweredFunction
from discovery import Discovery
import ir
from mpy_types import Variable

logger = logging.getLogger( __name__ )

class Tests( unittest.TestCase ):
	''' every test's code-under-test lives inside main() - _test_ir always lowers Discovery.main and nothing else '''
	maxDiff = None

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = False )
		self.compiler = Compiler( self.discovery )

	def _import( self, code: str ):
		return self.compiler.import_code( code, filename = Path( '__test__.py' ))

	def _lower_main( self ) -> LoweredFunction:
		fn = self.compiler._lower( self.discovery.main )
		self.assertEqual( type( fn ), LoweredFunction, f'expecting lower to return a LoweredFunction but got {type(fn)!r}' )
		return fn

	def _assert_ir( self, fn: LoweredFunction, expected: list[ir.Instruction] ) -> None:
		got = [ op.test_repr() for op in fn.instructions ]
		want = [ op.test_repr() for op in expected ]
		self.assertEqual( got, want )

	def _test_ir( self, code: str, expected: list[ir.Instruction] ) -> None:
		self._import( code )

		# TODO FIXME: we need to implement better error handling
		#self.assertEqual( len( analyzer.errors ), 0, f"Compiler errors found: {analyzer.errors}" )

		fn = self._lower_main()
		self._assert_ir( fn, expected )

	# --- return -----------------------------------------------------------

	def test_return_literal( self ) -> None:
		code = '\n'.join([
			'def main() -> i32:',
			'	return 1',
		])
		i32 = self.discovery.get_intrinsics()['i32']
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = i32 ),
			ir.Return( value = ir.Const( type = i32, value = 1 )),
			ir.FuncEnd( name = 'main' ),
		])

	def test_return_nothing( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	return',
		])
		none_type = self.discovery.get_none_type()
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_docstring_statement_is_a_no_op( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			"	'''this is a docstring'''",
			'	return',
		])
		none_type = self.discovery.get_none_type()
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	# --- locals: AnnAssign / Assign ----------------------------------------

	def test_annassign_without_initializer( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	x: i32',
			'	return',
		])
		none_type = self.discovery.get_none_type()
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_annassign_then_reassign( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	x: i32 = 1',
			'	x = 2',
			'	return',
		])
		i32 = self.discovery.get_intrinsics()['i32']
		none_type = self.discovery.get_none_type()
		x = Variable( stem = 'x', qualname = 'main.x', file = Path( '__test__.py' ), line = 2, type = i32 )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = x, src = ir.Const( type = i32, value = 1 )),
			ir.Assign( dest = x, src = ir.Const( type = i32, value = 2 )),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_bare_assign_to_new_name_infers_type_from_rhs( self ) -> None:
		# no annotation at all - x's type comes from y's, same as if it had
		# been written `x: i32 = y`
		code = '\n'.join([
			'def main() -> None:',
			'	y: i32 = 1',
			'	x = y',
			'	return',
		])
		i32 = self.discovery.get_intrinsics()['i32']
		none_type = self.discovery.get_none_type()
		y = Variable( stem = 'y', qualname = 'main.y', file = Path( '__test__.py' ), line = 2, type = i32 )
		x = Variable( stem = 'x', qualname = 'main.x', file = Path( '__test__.py' ), line = 3, type = i32 )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = y, src = ir.Const( type = i32, value = 1 )),
			ir.Assign( dest = x, src = y ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_inferred_local_can_be_reassigned_afterward( self ) -> None:
		# the first bare `x = y` introduces x via inference; the second is a
		# plain reassignment of that same Variable, not a second declaration
		code = '\n'.join([
			'def main() -> None:',
			'	y: i32 = 1',
			'	x = y',
			'	x = y',
			'	return',
		])
		i32 = self.discovery.get_intrinsics()['i32']
		none_type = self.discovery.get_none_type()
		y = Variable( stem = 'y', qualname = 'main.y', file = Path( '__test__.py' ), line = 2, type = i32 )
		x = Variable( stem = 'x', qualname = 'main.x', file = Path( '__test__.py' ), line = 3, type = i32 )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = y, src = ir.Const( type = i32, value = 1 )),
			ir.Assign( dest = x, src = y ),
			ir.Assign( dest = x, src = y ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_bare_assign_of_uninferrable_literal_still_fails( self ) -> None:
		# a bare literal has no type of its own to infer from - same
		# limitation _expr_Constant already has for any other context
		code = '\n'.join([
			'def main() -> None:',
			'	x = 1',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'cannot infer the type', self.discovery.errors.errors[0] )

	# --- arithmetic ---------------------------------------------------------

	def test_binop_without_arithmetic_context_is_a_compile_error( self ) -> None:
		# arithmetic defaults to Check mode (Result[T,OverflowError]) - see
		# the Lowering class docstring - and main() returns None, which can't
		# propagate that error, so plain `a + 1` here is a compile error
		# rather than silently falling back to wrapping
		code = '\n'.join([
			'class OverflowError: pass',
			'',
			'@cstruct',
			'class Result[T,E]:',
			'	pass',
			'',
			'def main() -> None:',
			'	a: i32 = 1',
			'	b: i32 = a + 1',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertIn( 'wrap_arithmetic', self.discovery.errors.errors[0] )
		# lower_function's per-statement recovery boundary skips just the
		# failing statement - b is never assigned, everything else is fine
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertEqual( kinds, [ 'FuncStart', 'Assign', 'Return', 'FuncEnd' ] )

	def test_binop_wrap_arithmetic_context( self ) -> None:
		# with compiler.wrap_arithmetic: switches Add/Sub/Mult back to the
		# plain Wrap opcodes, no Result/OrReturn involved - the with
		# statement itself contributes no instructions of its own
		code = '\n'.join([
			'def main() -> None:',
			'	a: i32 = 1',
			'	with compiler.wrap_arithmetic:',
			'		b: i32 = a + 1',
			'		c: i32 = a - 1',
			'		d: i32 = a * 2',
			'	return',
		])
		i32 = self.discovery.get_intrinsics()['i32']
		none_type = self.discovery.get_none_type()
		a = Variable( stem = 'a', qualname = 'main.a', file = Path( '__test__.py' ), line = 2, type = i32 )
		b = Variable( stem = 'b', qualname = 'main.b', file = Path( '__test__.py' ), line = 4, type = i32 )
		c = Variable( stem = 'c', qualname = 'main.c', file = Path( '__test__.py' ), line = 5, type = i32 )
		d = Variable( stem = 'd', qualname = 'main.d', file = Path( '__test__.py' ), line = 6, type = i32 )
		# temp numbering is per-function (not per-statement), so each new
		# statement's temp continues where the last one left off
		t0 = ir.Temp( type = i32, id = 0 )
		t1 = ir.Temp( type = i32, id = 1 )
		t2 = ir.Temp( type = i32, id = 2 )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = a, src = ir.Const( type = i32, value = 1 )),
			ir.DeclareTemp( temp = t0 ),
			ir.AddWrap( dest = t0, left = a, right = ir.Const( type = i32, value = 1 )),
			ir.Assign( dest = b, src = t0 ),
			ir.DeleteTemp( temp = t0 ),
			ir.DeclareTemp( temp = t1 ),
			ir.SubWrap( dest = t1, left = a, right = ir.Const( type = i32, value = 1 )),
			ir.Assign( dest = c, src = t1 ),
			ir.DeleteTemp( temp = t1 ),
			ir.DeclareTemp( temp = t2 ),
			ir.MulWrap( dest = t2, left = a, right = ir.Const( type = i32, value = 2 )),
			ir.Assign( dest = d, src = t2 ),
			ir.DeleteTemp( temp = t2 ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_binop_check_mode_emits_or_return( self ) -> None:
		# a function OTHER than main (main can never return Result, since it
		# takes no arguments to be called with the error) that returns
		# Result[None,OverflowError] - every Check op is immediately followed
		# by an OrReturn (Result.or_return()'s own semantics), consuming the
		# Result and continuing with the unwrapped i32 value
		code = '\n'.join([
			'class OverflowError: pass',
			'',
			'@cstruct',
			'class Result[T,E]:',
			'	pass',
			'',
			'def checked() -> Result[None,OverflowError]:',
			'	a: i32 = 1',
			'	b: i32 = a + 1',
			'	c: i32 = a - 1',
			'	d: i32 = a * 2',
		])
		mod = self._import( code )
		i32 = self.discovery.get_intrinsics()['i32']

		checked_fn = mod.get_local( 'checked' )
		if checked_fn.resolve is not None:
			checked_fn.resolve()
		overflow_cls = mod.get_local( 'OverflowError' )
		result_cls = mod.get_local( 'Result' )
		if result_cls.resolve is not None:
			result_cls.resolve()
		result_i32_overflow = self.discovery._get_or_create_specialization( result_cls, [ i32, overflow_cls ] )

		a = Variable( stem = 'a', qualname = '__test__.checked.a', file = Path( '__test__.py' ), line = 8, type = i32 )
		b = Variable( stem = 'b', qualname = '__test__.checked.b', file = Path( '__test__.py' ), line = 9, type = i32 )
		c = Variable( stem = 'c', qualname = '__test__.checked.c', file = Path( '__test__.py' ), line = 10, type = i32 )
		d = Variable( stem = 'd', qualname = '__test__.checked.d', file = Path( '__test__.py' ), line = 11, type = i32 )

		t0 = ir.Temp( type = result_i32_overflow, id = 0 ) # AddCheck's Result
		t1 = ir.Temp( type = i32, id = 1 )                 # unwrapped via OrReturn
		t2 = ir.Temp( type = result_i32_overflow, id = 2 ) # SubCheck's Result
		t3 = ir.Temp( type = i32, id = 3 )
		t4 = ir.Temp( type = result_i32_overflow, id = 4 ) # MulCheck's Result
		t5 = ir.Temp( type = i32, id = 5 )

		fn = self.compiler._lower( checked_fn )
		self._assert_ir( fn, [
			ir.FuncStart( name = '__test__.checked', params = [], return_type = checked_fn.return_type ),
			ir.Assign( dest = a, src = ir.Const( type = i32, value = 1 )),
			ir.DeclareTemp( temp = t0 ),
			ir.AddCheck( dest = t0, left = a, right = ir.Const( type = i32, value = 1 )),
			ir.DeclareTemp( temp = t1 ),
			ir.OrReturn( dest = t1, value = t0 ),
			ir.Assign( dest = b, src = t1 ),
			ir.DeleteTemp( temp = t1 ),
			ir.DeleteTemp( temp = t0 ),
			ir.DeclareTemp( temp = t2 ),
			ir.SubCheck( dest = t2, left = a, right = ir.Const( type = i32, value = 1 )),
			ir.DeclareTemp( temp = t3 ),
			ir.OrReturn( dest = t3, value = t2 ),
			ir.Assign( dest = c, src = t3 ),
			ir.DeleteTemp( temp = t3 ),
			ir.DeleteTemp( temp = t2 ),
			ir.DeclareTemp( temp = t4 ),
			ir.MulCheck( dest = t4, left = a, right = ir.Const( type = i32, value = 2 )),
			ir.DeclareTemp( temp = t5 ),
			ir.OrReturn( dest = t5, value = t4 ),
			ir.Assign( dest = d, src = t5 ),
			ir.DeleteTemp( temp = t5 ),
			ir.DeleteTemp( temp = t4 ),
			ir.FuncEnd( name = '__test__.checked' ),
		])

	def test_binop_saturate_arithmetic_context( self ) -> None:
		# with compiler.saturate_arithmetic: - same shape as wrap_arithmetic,
		# just the *Saturate opcodes instead - no Result/OrReturn involved
		# either, so this works fine inside main() too
		code = '\n'.join([
			'def main() -> None:',
			'	a: i32 = 1',
			'	with compiler.saturate_arithmetic:',
			'		b: i32 = a + 1',
			'		c: i32 = a - 1',
			'		d: i32 = a * 2',
			'	return',
		])
		i32 = self.discovery.get_intrinsics()['i32']
		none_type = self.discovery.get_none_type()
		a = Variable( stem = 'a', qualname = 'main.a', file = Path( '__test__.py' ), line = 2, type = i32 )
		b = Variable( stem = 'b', qualname = 'main.b', file = Path( '__test__.py' ), line = 4, type = i32 )
		c = Variable( stem = 'c', qualname = 'main.c', file = Path( '__test__.py' ), line = 5, type = i32 )
		d = Variable( stem = 'd', qualname = 'main.d', file = Path( '__test__.py' ), line = 6, type = i32 )
		t0 = ir.Temp( type = i32, id = 0 )
		t1 = ir.Temp( type = i32, id = 1 )
		t2 = ir.Temp( type = i32, id = 2 )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = a, src = ir.Const( type = i32, value = 1 )),
			ir.DeclareTemp( temp = t0 ),
			ir.AddSaturate( dest = t0, left = a, right = ir.Const( type = i32, value = 1 )),
			ir.Assign( dest = b, src = t0 ),
			ir.DeleteTemp( temp = t0 ),
			ir.DeclareTemp( temp = t1 ),
			ir.SubSaturate( dest = t1, left = a, right = ir.Const( type = i32, value = 1 )),
			ir.Assign( dest = c, src = t1 ),
			ir.DeleteTemp( temp = t1 ),
			ir.DeclareTemp( temp = t2 ),
			ir.MulSaturate( dest = t2, left = a, right = ir.Const( type = i32, value = 2 )),
			ir.Assign( dest = d, src = t2 ),
			ir.DeleteTemp( temp = t2 ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_binop_panic_arithmetic_context( self ) -> None:
		# with compiler.panic_arithmetic(msg): - still Check-mode ops
		# (Result[T,OverflowError]), but consumed with Unwrap(errmsg=msg)
		# instead of OrReturn - unlike the bare default, this does NOT
		# require the enclosing function to return Result[_,OverflowError],
		# since Unwrap panics rather than needing anywhere to propagate to -
		# main() works fine here
		code = '\n'.join([
			'class OverflowError: pass',
			'',
			'@cstruct',
			'class Result[T,E]:',
			'	pass',
			'',
			'class str: pass',
			'',
			'def main() -> None:',
			'	a: i32 = 1',
			"	with compiler.panic_arithmetic( 'bad arithmetic' ):",
			'		b: i32 = a + 1',
			'	return',
		])
		mod = self._import( code )
		i32 = self.discovery.get_intrinsics()['i32']
		none_type = self.discovery.get_none_type()
		str_cls = mod.get_local( 'str' )
		overflow_cls = mod.get_local( 'OverflowError' )
		result_cls = mod.get_local( 'Result' )
		if result_cls.resolve is not None:
			result_cls.resolve()
		result_i32_overflow = self.discovery._get_or_create_specialization( result_cls, [ i32, overflow_cls ] )

		a = Variable( stem = 'a', qualname = 'main.a', file = Path( '__test__.py' ), line = 10, type = i32 )
		b = Variable( stem = 'b', qualname = 'main.b', file = Path( '__test__.py' ), line = 12, type = i32 )
		t0 = ir.Temp( type = result_i32_overflow, id = 0 ) # AddCheck's Result
		t1 = ir.Temp( type = i32, id = 1 )                 # unwrapped via Unwrap

		fn = self._lower_main()
		self._assert_ir( fn, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = a, src = ir.Const( type = i32, value = 1 )),
			ir.DeclareTemp( temp = t0 ),
			ir.AddCheck( dest = t0, left = a, right = ir.Const( type = i32, value = 1 )),
			ir.DeclareTemp( temp = t1 ),
			ir.Unwrap( dest = t1, value = t0, errmsg = ir.Const( type = str_cls, value = 'bad arithmetic' )),
			ir.Assign( dest = b, src = t1 ),
			ir.DeleteTemp( temp = t1 ),
			ir.DeleteTemp( temp = t0 ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_binop_literal_on_left( self ) -> None:
		# expected type flows from whichever side is NOT the bare literal -
		# wrapped in wrap_arithmetic just to sidestep the Check-mode/Result
		# requirement, unrelated to what this test actually checks
		code = '\n'.join([
			'def main() -> None:',
			'	a: i32 = 1',
			'	with compiler.wrap_arithmetic:',
			'		b: i32 = 1 + a',
			'	return',
		])
		i32 = self.discovery.get_intrinsics()['i32']
		none_type = self.discovery.get_none_type()
		a = Variable( stem = 'a', qualname = 'main.a', file = Path( '__test__.py' ), line = 2, type = i32 )
		b = Variable( stem = 'b', qualname = 'main.b', file = Path( '__test__.py' ), line = 4, type = i32 )
		t0 = ir.Temp( type = i32, id = 0 )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = a, src = ir.Const( type = i32, value = 1 )),
			ir.DeclareTemp( temp = t0 ),
			ir.AddWrap( dest = t0, left = ir.Const( type = i32, value = 1 ), right = a ),
			ir.Assign( dest = b, src = t0 ),
			ir.DeleteTemp( temp = t0 ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	# --- calls ---------------------------------------------------------------

	def test_call_free_function_positional_and_keyword( self ) -> None:
		code = '\n'.join([
			'def foo( x: i32, y: i32 = 2 ) -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	foo( 1, y = 3 )',
			'	return',
		])
		mod = self._import( code )
		i32 = self.discovery.get_intrinsics()['i32']
		none_type = self.discovery.get_none_type()
		foo_fn = mod.get_local( 'foo' )
		if foo_fn.resolve is not None:
			foo_fn.resolve()

		fn = self._lower_main()
		self._assert_ir( fn, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Call(
				dest = None,
				target = foo_fn,
				args = [ ir.Const( type = i32, value = 1 ) ],
				kwargs = { 'y': ir.Const( type = i32, value = 3 ) },
			),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_call_result_used_in_assignment( self ) -> None:
		code = '\n'.join([
			'def foo( x: i32 ) -> i32:',
			'	return x',
			'',
			'def main() -> None:',
			'	y: i32 = foo( 1 )',
			'	return',
		])
		mod = self._import( code )
		i32 = self.discovery.get_intrinsics()['i32']
		none_type = self.discovery.get_none_type()
		foo_fn = mod.get_local( 'foo' )
		if foo_fn.resolve is not None:
			foo_fn.resolve()
		y = Variable( stem = 'y', qualname = 'main.y', file = Path( '__test__.py' ), line = 5, type = i32 )
		t0 = ir.Temp( type = i32, id = 0 )

		fn = self._lower_main()
		self._assert_ir( fn, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.DeclareTemp( temp = t0 ),
			ir.Call( dest = t0, target = foo_fn, args = [ ir.Const( type = i32, value = 1 ) ], kwargs = {} ),
			ir.Assign( dest = y, src = t0 ),
			ir.DeleteTemp( temp = t0 ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_call_staticmethod_via_class_name( self ) -> None:
		code = '\n'.join([
			'class Foo:',
			'	@staticmethod',
			'	def make() -> i32:',
			'		return 1',
			'',
			'def main() -> None:',
			'	Foo.make()',
			'	return',
		])
		mod = self._import( code )
		none_type = self.discovery.get_none_type()
		foo_cls = mod.get_local( 'Foo' )
		if foo_cls.resolve is not None:
			foo_cls.resolve()
		make_fn = foo_cls.get_local( 'make' )
		if make_fn.resolve is not None:
			make_fn.resolve()

		fn = self._lower_main()
		self._assert_ir( fn, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Call( dest = None, target = make_fn, receiver = None, args = [], kwargs = {} ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_call_instance_method_via_local_variable( self ) -> None:
		# main() has no `self` of its own - this exercises the *other* path
		# into a bound-instance call: a plain local variable of class type
		code = '\n'.join([
			'class Foo:',
			'	def bump( self ) -> i32:',
			'		return 1',
			'',
			'def main() -> None:',
			'	f: Foo',
			'	f.bump()',
			'	return',
		])
		mod = self._import( code )
		none_type = self.discovery.get_none_type()
		foo_cls = mod.get_local( 'Foo' )
		if foo_cls.resolve is not None:
			foo_cls.resolve()
		bump_fn = foo_cls.get_local( 'bump' )
		if bump_fn.resolve is not None:
			bump_fn.resolve()
		f = Variable( stem = 'f', qualname = 'main.f', file = Path( '__test__.py' ), line = 6, type = foo_cls )

		fn = self._lower_main()
		self._assert_ir( fn, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Call( dest = None, target = bump_fn, receiver = f, args = [], kwargs = {} ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	# --- attributes / subscripts --------------------------------------------

	def test_getattr_setattr( self ) -> None:
		code = '\n'.join([
			'class Foo:',
			'	x: i32',
			'',
			'def main() -> None:',
			'	f: Foo',
			'	f.x = 1',
			'	y: i32 = f.x',
			'	return',
		])
		mod = self._import( code )
		i32 = self.discovery.get_intrinsics()['i32']
		none_type = self.discovery.get_none_type()
		foo_cls = mod.get_local( 'Foo' )
		if foo_cls.resolve is not None:
			foo_cls.resolve()
		f = Variable( stem = 'f', qualname = 'main.f', file = Path( '__test__.py' ), line = 5, type = foo_cls )
		y = Variable( stem = 'y', qualname = 'main.y', file = Path( '__test__.py' ), line = 7, type = i32 )
		t0 = ir.Temp( type = i32, id = 0 )

		fn = self._lower_main()
		self._assert_ir( fn, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.SetAttr( obj = f, attr = 'x', value = ir.Const( type = i32, value = 1 )),
			ir.DeclareTemp( temp = t0 ),
			ir.GetAttr( dest = t0, obj = f, attr = 'x' ),
			ir.Assign( dest = y, src = t0 ),
			ir.DeleteTemp( temp = t0 ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_getitem_setitem( self ) -> None:
		# GetItem/SetItem don't type-check the container/index themselves (no
		# __getitem__ resolution yet) - an already-typed local sidesteps the
		# "no expected type available" rule that a bare literal index/value
		# would hit
		code = '\n'.join([
			'class Container:',
			'	pass',
			'',
			'def main() -> None:',
			'	c: Container',
			'	i: usize = 0',
			'	j: i32 = 5',
			'	c[i] = j',
			'	x: i32 = c[i]',
			'	return',
		])
		mod = self._import( code )
		i32 = self.discovery.get_intrinsics()['i32']
		usize = self.discovery.get_intrinsics()['usize']
		none_type = self.discovery.get_none_type()
		container_cls = mod.get_local( 'Container' )
		c = Variable( stem = 'c', qualname = 'main.c', file = Path( '__test__.py' ), line = 5, type = container_cls )
		i = Variable( stem = 'i', qualname = 'main.i', file = Path( '__test__.py' ), line = 6, type = usize )
		j = Variable( stem = 'j', qualname = 'main.j', file = Path( '__test__.py' ), line = 7, type = i32 )
		x = Variable( stem = 'x', qualname = 'main.x', file = Path( '__test__.py' ), line = 9, type = i32 )
		t0 = ir.Temp( type = i32, id = 0 )

		fn = self._lower_main()
		self._assert_ir( fn, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = i, src = ir.Const( type = usize, value = 0 )),
			ir.Assign( dest = j, src = ir.Const( type = i32, value = 5 )),
			ir.SetItem( obj = c, index = i, value = j ),
			ir.DeclareTemp( temp = t0 ),
			ir.GetItem( dest = t0, obj = c, index = i ),
			ir.Assign( dest = x, src = t0 ),
			ir.DeleteTemp( temp = t0 ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	# --- globals -------------------------------------------------------------

	def test_reads_module_global( self ) -> None:
		code = '\n'.join([
			'G: i32 = 5',
			'',
			'def main() -> None:',
			'	x: i32 = G',
			'	return',
		])
		mod = self._import( code )
		i32 = self.discovery.get_intrinsics()['i32']
		none_type = self.discovery.get_none_type()
		g = mod.get_local( 'G' )
		if g.resolve is not None:
			g.resolve()
		x = Variable( stem = 'x', qualname = 'main.x', file = Path( '__test__.py' ), line = 4, type = i32 )

		fn = self._lower_main()
		self._assert_ir( fn, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = x, src = g ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	# --- overload call sites ---------------------------------------------------

	def test_overload_call_resolves_to_unconditional_target( self ) -> None:
		code = '\n'.join([
			'class int: pass',
			'class str: pass',
			'',
			'@overload',
			'def foo( x: int ) -> None:',
			'	...',
			'',
			'def foo( x: int|None = None ) -> None:',
			'	pass',
			'',
			'def foo( x: str ) -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	x: int',
			'	foo( x )',
			'	return',
		])
		mod = self._import( code )
		none_type = self.discovery.get_none_type()
		int_cls = mod.get_local( 'int' )
		group = mod.get_local( 'foo' )
		int_impl = group.implementations[0]
		if int_impl.resolve is not None:
			int_impl.resolve()
		x = Variable( stem = 'x', qualname = 'main.x', file = Path( '__test__.py' ), line = 15, type = int_cls )

		fn = self._lower_main()
		self._assert_ir( fn, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Call( dest = None, target = int_impl, args = [ x ], kwargs = {} ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	# --- defer/errdefer --------------------------------------------------------

	def test_defer_rejected_inside_a_for_loop( self ) -> None:
		code = '\n'.join([
			'def cleanup() -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	for i in range( 3 ):',
			'		defer( cleanup() )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any( 'not allowed inside a loop' in e for e in self.discovery.errors.errors ))

	def test_errdefer_rejected_inside_a_while_loop( self ) -> None:
		code = '\n'.join([
			'class OverflowError: pass',
			'',
			'@cstruct',
			'class Result[T,E]:',
			'	pass',
			'',
			'def checked() -> Result[None,OverflowError]:',
			'	while True:',
			'		with errdefer:',
			'			pass',
		])
		mod = self._import( code )
		checked_fn = mod.get_local( 'checked' )
		if checked_fn.resolve is not None:
			checked_fn.resolve()
		self.compiler._lower( checked_fn )
		self.assertTrue( any( 'not allowed inside a loop' in e for e in self.discovery.errors.errors ))

	def test_defer_rejected_when_nested_inside_another_defer( self ) -> None:
		code = '\n'.join([
			'class bool: pass',
			'',
			'def cleanup() -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	with defer:',
			'		with defer:',
			'			cleanup()',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any( 'nested inside another defer' in e for e in self.discovery.errors.errors ))

	def test_errdefer_rejected_when_function_does_not_return_result( self ) -> None:
		code = '\n'.join([
			'class bool: pass',
			'class OverflowError: pass',
			'',
			'@cstruct',
			'class Result[T,E]:',
			'	pass',
			'',
			'def main() -> None:', # -> None, not Result[_,_] - errdefer isn't legal here
			'	with errdefer:',
			'		pass',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any( 'requires the enclosing function to return Result' in e for e in self.discovery.errors.errors ))

	def test_defer_epilogue_shape( self ) -> None:
		code = '\n'.join([
			'class bool: pass',
			'',
			'def cleanup() -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	with defer:',
			'		cleanup()',
			'	return',
		])
		mod = self._import( code )
		bool_cls = mod.get_local( 'bool' )
		cleanup_fn = mod.get_local( 'cleanup' )
		if cleanup_fn.resolve is not None:
			cleanup_fn.resolve()
		none_type = self.discovery.get_none_type()
		flag0 = Variable( stem = '__defer_flag_0', qualname = 'main.__defer_flag_0', file = Path( '__test__.py' ), line = 7, type = bool_cls )

		fn = self._lower_main()
		self._assert_ir( fn, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = flag0, src = ir.Const( type = bool_cls, value = False )),
			ir.Assign( dest = flag0, src = ir.Const( type = bool_cls, value = True )),
			ir.Jump( target = '__epilogue__' ),
			ir.Label( name = '__epilogue__' ),
			ir.JumpIfFalse( cond = flag0, target = '__defer_skip_0__' ),
			ir.Call( dest = None, target = cleanup_fn, args = [], kwargs = {} ),
			ir.Label( name = '__defer_skip_0__' ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_errdefer_epilogue_shape_with_no_triggering_error( self ) -> None:
		code = '\n'.join([
			'class bool: pass',
			'class OverflowError: pass',
			'',
			'@cstruct',
			'class Result[T,E]:',
			'	def is_err( self ) -> bool:',
			'		pass',
			'',
			'def checked() -> Result[None,OverflowError]:',
			'	with errdefer:',
			'		pass',
		])
		mod = self._import( code )
		bool_cls = mod.get_local( 'bool' )
		result_cls = mod.get_local( 'Result' )
		if result_cls.resolve is not None:
			result_cls.resolve()
		is_err_fn = result_cls.get_local( 'is_err' )
		if is_err_fn.resolve is not None:
			is_err_fn.resolve()
		checked_fn = mod.get_local( 'checked' )
		if checked_fn.resolve is not None:
			checked_fn.resolve()

		flag0 = Variable( stem = '__defer_flag_0', qualname = '__test__.checked.__defer_flag_0', file = Path( '__test__.py' ), line = 10, type = bool_cls )
		return_value_var = Variable( stem = '__return_value', qualname = '__test__.checked.__return_value', file = Path( '__test__.py' ), line = 9, type = checked_fn.return_type )
		is_err_temp = ir.Temp( type = bool_cls, id = 0 )

		fn = self.compiler._lower( checked_fn )
		self._assert_ir( fn, [
			ir.FuncStart( name = '__test__.checked', params = [], return_type = checked_fn.return_type ),
			ir.Assign( dest = flag0, src = ir.Const( type = bool_cls, value = False )),
			ir.Assign( dest = flag0, src = ir.Const( type = bool_cls, value = True )),
			ir.Label( name = '__epilogue__' ),
			ir.DeclareTemp( temp = is_err_temp ),
			ir.Call( dest = is_err_temp, target = is_err_fn, receiver = return_value_var, args = [], kwargs = {} ),
			ir.JumpIfFalse( cond = flag0, target = '__defer_skip_0__' ),
			ir.JumpIfFalse( cond = is_err_temp, target = '__defer_skip_0__' ),
			ir.Label( name = '__defer_skip_0__' ),
			ir.DeleteTemp( temp = is_err_temp ),
			ir.Return( value = return_value_var ),
			ir.FuncEnd( name = '__test__.checked' ),
		])

	def test_errdefer_with_checked_arithmetic_emits_or_jump( self ) -> None:
		code = '\n'.join([
			'class bool: pass',
			'class OverflowError: pass',
			'',
			'@cstruct',
			'class Result[T,E]:',
			'	def is_err( self ) -> bool:',
			'		pass',
			'',
			'def checked() -> Result[None,OverflowError]:',
			'	with errdefer:',
			'		pass',
			'	a: i32 = 1',
			'	b: i32 = a + 1',
		])
		mod = self._import( code )
		checked_fn = mod.get_local( 'checked' )
		if checked_fn.resolve is not None:
			checked_fn.resolve()
		fn = self.compiler._lower( checked_fn )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertIn( 'AddCheck', kinds )
		self.assertIn( 'OrJump', kinds )
		self.assertNotIn( 'OrReturn', kinds )
		# epilogue's errdefer guard is the two-JumpIfFalse (flag, then is_err()) shape, back to back
		jump_if_false_indices = [ i for i, instr in enumerate( fn.instructions ) if isinstance( instr, ir.JumpIfFalse ) ]
		self.assertEqual( len( jump_if_false_indices ), 2 )
		self.assertEqual( jump_if_false_indices[1], jump_if_false_indices[0] + 1 )

	def test_explicit_err_return_still_stows_and_jumps( self ) -> None:
		# the specific gap the is_err()-based design fixes over a separate
		# error-flag: a plain `return Result.Err(...)` never touches OrJump at
		# all, but still needs to stow+jump so the epilogue's .is_err() check
		# (which inspects the stowed value itself) catches it too
		code = '\n'.join([
			'class bool: pass',
			'class OverflowError: pass',
			'',
			'@cstruct',
			'class Result[T,E]:',
			'	def is_err( self ) -> bool:',
			'		pass',
			'	@staticmethod',
			'	def Err( e: E ) -> Result[T,E]:',
			'		pass',
			'',
			'def checked( e: OverflowError ) -> Result[None,OverflowError]:',
			'	with errdefer:',
			'		pass',
			'	return Result.Err( e )',
		])
		mod = self._import( code )
		checked_fn = mod.get_local( 'checked' )
		if checked_fn.resolve is not None:
			checked_fn.resolve()
		fn = self.compiler._lower( checked_fn )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertEqual( kinds.count( 'Return' ), 1 ) # only the epilogue's, not one at the return statement's own position
		self.assertIn( 'Jump', kinds )
		call_index = kinds.index( 'Call' ) # Result.Err(e)
		jump_index = kinds.index( 'Jump' )
		return_index = kinds.index( 'Return' )
		self.assertLess( call_index, jump_index ) # stowed before jumping
		self.assertLess( jump_index, return_index ) # jumps to, rather than falls into, the epilogue

	def test_two_defers_replay_in_reverse_order( self ) -> None:
		code = '\n'.join([
			'class bool: pass',
			'',
			'def cleanup_a() -> None:',
			'	pass',
			'',
			'def cleanup_b() -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	with defer:',
			'		cleanup_a()',
			'	with defer:',
			'		cleanup_b()',
			'	return',
		])
		mod = self._import( code )
		cleanup_a = mod.get_local( 'cleanup_a' )
		cleanup_b = mod.get_local( 'cleanup_b' )
		fn = self._lower_main()
		calls = [ instr for instr in fn.instructions if isinstance( instr, ir.Call ) ]
		self.assertEqual( len( calls ), 2 )
		self.assertIs( calls[0].target, cleanup_b ) # last-registered runs first
		self.assertIs( calls[1].target, cleanup_a )

	def test_call_form_matches_with_block_form( self ) -> None:
		code = '\n'.join([
			'class bool: pass',
			'class OverflowError: pass',
			'',
			'@cstruct',
			'class Result[T,E]:',
			'	def is_err( self ) -> bool:',
			'		pass',
			'',
			'def cleanup() -> None:',
			'	pass',
			'',
			'def checked_with() -> Result[None,OverflowError]:',
			'	with errdefer:',
			'		cleanup()',
			'',
			'def checked_call() -> Result[None,OverflowError]:',
			'	errdefer( cleanup() )',
		])
		mod = self._import( code )
		fn_with = mod.get_local( 'checked_with' )
		fn_call = mod.get_local( 'checked_call' )
		if fn_with.resolve is not None:
			fn_with.resolve()
		if fn_call.resolve is not None:
			fn_call.resolve()
		lowered_with = self.compiler._lower( fn_with )
		lowered_call = self.compiler._lower( fn_call )
		kinds_with = [ type( i ).__name__ for i in lowered_with.instructions ]
		kinds_call = [ type( i ).__name__ for i in lowered_call.instructions ]
		self.assertEqual( kinds_with, kinds_call )

	def test_no_defer_still_uses_plain_return_and_or_return( self ) -> None:
		# regression check - a function with no defer/errdefer at all still
		# gets the pre-epilogue shape exactly as before, no Jump/Label anywhere
		code = '\n'.join([
			'class OverflowError: pass',
			'',
			'@cstruct',
			'class Result[T,E]:',
			'	pass',
			'',
			'def checked() -> Result[None,OverflowError]:',
			'	a: i32 = 1',
			'	b: i32 = a + 1',
		])
		mod = self._import( code )
		checked_fn = mod.get_local( 'checked' )
		if checked_fn.resolve is not None:
			checked_fn.resolve()
		fn = self.compiler._lower( checked_fn )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertIn( 'OrReturn', kinds )
		self.assertNotIn( 'OrJump', kinds )
		self.assertNotIn( 'Jump', kinds )
		self.assertNotIn( 'Label', kinds )

if __name__ == '__main__':
	logging.basicConfig( level = logging.DEBUG )
	unittest.main()
