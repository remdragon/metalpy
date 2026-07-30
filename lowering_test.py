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

if __name__ == '__main__':
	logging.basicConfig( level = logging.DEBUG )
	unittest.main()
