# stdlib imports:
import logging
from pathlib import Path
import unittest

# local imports:
from compiler import Compiler, LoweredFunction
from discovery import Discovery
import ir
from mpy_types import Variable, Specialization

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

	def test_augassign_desugars_to_binop_and_assign( self ) -> None:
		# x += 1 lowers exactly like a hand-written x = x + 1 would - same
		# AddWrap/Assign shape, honoring the active arithmetic mode
		code = '\n'.join([
			'def main() -> None:',
			'	x: i32 = 1',
			'	with compiler.wrap_arithmetic:',
			'		x += 2',
			'	return',
		])
		i32 = self.discovery.get_intrinsics()['i32']
		none_type = self.discovery.get_none_type()
		x = Variable( stem = 'x', qualname = 'main.x', file = Path( '__test__.py' ), line = 2, type = i32 )
		t0 = ir.Temp( type = i32, id = 0 )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = x, src = ir.Const( type = i32, value = 1 )),
			ir.DeclareTemp( temp = t0 ),
			ir.AddWrap( dest = t0, left = x, right = ir.Const( type = i32, value = 2 )),
			ir.Assign( dest = x, src = t0 ),
			ir.DeleteTemp( temp = t0 ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_augassign_to_undeclared_name_fails( self ) -> None:
		# can't read from something that was never declared - the
		# synthesized BinOp's own Name lookup fails naturally, same as any
		# other read of an undefined name
		code = '\n'.join([
			'def main() -> None:',
			'	x += 1',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( "name 'x' is not defined", self.discovery.errors.errors[0] )

	def test_augassign_attribute_target_unsupported( self ) -> None:
		# left unsupported deliberately - the object expression would need
		# to be evaluated twice under the x = x + y desugaring (once to
		# read the current value, once to resolve the write target), a real
		# correctness risk for anything with side effects
		code = '\n'.join([
			'class Foo:',
			'	x: i32',
			'',
			'def main() -> None:',
			'	f: Foo',
			'	f.x += 1',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'unsupported AugAssign target', self.discovery.errors.errors[0] )

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

	def test_or_return_call_expands_to_or_return_ir_at_call_site( self ) -> None:
		# <result_expr>.or_return() is recognized at the call site and
		# expanded directly to OrReturn - Result.or_return's own declared
		# body (`return self.x` here) is never itself scheduled/lowered as
		# a Call target, since it would need to return from ITS CALLER, not
		# itself (see _lower_or_return's own comment)
		code = '\n'.join([
			'class MyError: pass',
			'',
			'@cstruct',
			'class Result[T,E]:',
			'	x: T',
			'',
			'	def or_return( self ) -> T:',
			'		return self.x',
			'',
			'def get_result() -> Result[i32,MyError]:',
			'	pass',
			'',
			'def foo() -> Result[i32,MyError]:',
			'	v: i32 = get_result().or_return()',
		])
		mod = self._import( code )
		i32 = self.discovery.get_intrinsics()['i32']
		foo_fn = mod.get_local( 'foo' )
		if foo_fn.resolve is not None:
			foo_fn.resolve()
		result_cls = mod.get_local( 'Result' )
		myerror_cls = mod.get_local( 'MyError' )
		if result_cls.resolve is not None:
			result_cls.resolve()
		result_i32_myerror = self.discovery._get_or_create_specialization( result_cls, [ i32, myerror_cls ] )

		v = Variable( stem = 'v', qualname = '__test__.foo.v', file = Path( '__test__.py' ), line = 14, type = i32 )
		t0 = ir.Temp( type = result_i32_myerror, id = 0 ) # get_result()'s Result
		t1 = ir.Temp( type = i32, id = 1 )                # unwrapped via OrReturn

		fn = self.compiler._lower( foo_fn )
		get_result_fn = mod.get_local( 'get_result' )
		self._assert_ir( fn, [
			ir.FuncStart( name = '__test__.foo', params = [], return_type = foo_fn.return_type ),
			ir.DeclareTemp( temp = t0 ),
			ir.Call( dest = t0, target = get_result_fn, receiver = None, args = [], kwargs = {} ),
			ir.DeclareTemp( temp = t1 ),
			ir.OrReturn( dest = t1, value = t0 ),
			ir.Assign( dest = v, src = t1 ),
			ir.DeleteTemp( temp = t1 ),
			ir.DeleteTemp( temp = t0 ),
			ir.FuncEnd( name = '__test__.foo' ),
		])
		self.assertFalse( any( isinstance( i, ir.Call ) and getattr( i.target, 'stem', None ) == 'or_return' for i in fn.instructions ))

	def test_or_return_outside_result_returning_function_is_rejected( self ) -> None:
		code = '\n'.join([
			'class MyError: pass',
			'',
			'@cstruct',
			'class Result[T,E]:',
			'	x: T',
			'',
			'	def or_return( self ) -> T:',
			'		return self.x',
			'',
			'def get_result() -> Result[i32,MyError]:',
			'	pass',
			'',
			'def foo() -> None:',
			'	v: i32 = get_result().or_return()',
			'	return',
		])
		self._import( code )
		foo_fn = self.discovery.modules['__test__'].get_local( 'foo' )
		if foo_fn.resolve is not None:
			foo_fn.resolve()
		self.compiler._lower( foo_fn )
		self.assertIn( 'or_return', self.discovery.errors.errors[0] )

	def test_compiler_early_return_desugars_to_return_result_err( self ) -> None:
		code = '\n'.join([
			'class MyError: pass',
			'',
			'@cstruct',
			'class Result[T,E]:',
			'	x: T',
			'',
			'	@staticmethod',
			'	def Err( e: E ) -> Result[T,E]:',
			'		return Result.__allocate__( x = 0 )',
			'',
			'def foo() -> Result[i32,MyError]:',
			'	compiler.early_return( MyError() )',
		])
		mod = self._import( code )
		i32 = self.discovery.get_intrinsics()['i32']
		foo_fn = mod.get_local( 'foo' )
		if foo_fn.resolve is not None:
			foo_fn.resolve()
		result_cls = mod.get_local( 'Result' )
		myerror_cls = mod.get_local( 'MyError' )
		if result_cls.resolve is not None:
			result_cls.resolve()
		err_fn = result_cls.get_local( 'Err' )
		if err_fn.resolve is not None:
			err_fn.resolve()
		result_i32_myerror = self.discovery._get_or_create_specialization( result_cls, [ i32, myerror_cls ] )

		t0 = ir.Temp( type = err_fn.parameters[0].type, id = 0 ) # MyError()'s Allocate - typed as Err's own `e: E` param (generic monomorphization doesn't exist yet, so this stays the bare TypeVar, not MyError)
		t1 = ir.Temp( type = result_i32_myerror, id = 1 )  # Result.Err(...)'s Call

		fn = self.compiler._lower( foo_fn )
		self._assert_ir( fn, [
			ir.FuncStart( name = '__test__.foo', params = [], return_type = foo_fn.return_type ),
			ir.DeclareTemp( temp = t0 ),
			ir.Allocate( dest = t0, cls = myerror_cls, fields = {} ),
			ir.DeclareTemp( temp = t1 ),
			ir.Call( dest = t1, target = err_fn, receiver = None, args = [ t0 ], kwargs = {} ),
			ir.Return( value = t1 ),
			ir.DeleteTemp( temp = t1 ),
			ir.DeleteTemp( temp = t0 ),
			ir.FuncEnd( name = '__test__.foo' ),
		])

	def test_compiler_early_return_outside_result_returning_function_is_rejected( self ) -> None:
		code = '\n'.join([
			'class MyError: pass',
			'',
			'@cstruct',
			'class Result[T,E]:',
			'	x: T',
			'',
			'def foo() -> None:',
			'	compiler.early_return( MyError() )',
			'	return',
		])
		self._import( code )
		foo_fn = self.discovery.modules['__test__'].get_local( 'foo' )
		if foo_fn.resolve is not None:
			foo_fn.resolve()
		self.compiler._lower( foo_fn )
		self.assertIn( 'compiler.early_return', self.discovery.errors.errors[0] )

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

	def test_binop_bitwise_ops_are_unconditional( self ) -> None:
		# no overflow concept for &/|/^/>> - always a single opcode,
		# independent of arithmetic mode (works fine in main() with no
		# wrap/check/saturate context at all, unlike +/-/*)
		code = '\n'.join([
			'def main() -> None:',
			'	a: i32 = 6',
			'	b: i32 = a & 3',
			'	c: i32 = a | 3',
			'	d: i32 = a ^ 3',
			'	e: i32 = a >> 1',
			'	return',
		])
		i32 = self.discovery.get_intrinsics()['i32']
		none_type = self.discovery.get_none_type()
		a = Variable( stem = 'a', qualname = 'main.a', file = Path( '__test__.py' ), line = 2, type = i32 )
		b = Variable( stem = 'b', qualname = 'main.b', file = Path( '__test__.py' ), line = 3, type = i32 )
		c = Variable( stem = 'c', qualname = 'main.c', file = Path( '__test__.py' ), line = 4, type = i32 )
		d = Variable( stem = 'd', qualname = 'main.d', file = Path( '__test__.py' ), line = 5, type = i32 )
		e = Variable( stem = 'e', qualname = 'main.e', file = Path( '__test__.py' ), line = 6, type = i32 )
		t0 = ir.Temp( type = i32, id = 0 )
		t1 = ir.Temp( type = i32, id = 1 )
		t2 = ir.Temp( type = i32, id = 2 )
		t3 = ir.Temp( type = i32, id = 3 )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = a, src = ir.Const( type = i32, value = 6 )),
			ir.DeclareTemp( temp = t0 ),
			ir.BitAnd( dest = t0, left = a, right = ir.Const( type = i32, value = 3 )),
			ir.Assign( dest = b, src = t0 ),
			ir.DeleteTemp( temp = t0 ),
			ir.DeclareTemp( temp = t1 ),
			ir.BitOr( dest = t1, left = a, right = ir.Const( type = i32, value = 3 )),
			ir.Assign( dest = c, src = t1 ),
			ir.DeleteTemp( temp = t1 ),
			ir.DeclareTemp( temp = t2 ),
			ir.BitXor( dest = t2, left = a, right = ir.Const( type = i32, value = 3 )),
			ir.Assign( dest = d, src = t2 ),
			ir.DeleteTemp( temp = t2 ),
			ir.DeclareTemp( temp = t3 ),
			ir.Shr( dest = t3, left = a, right = ir.Const( type = i32, value = 1 )),
			ir.Assign( dest = e, src = t3 ),
			ir.DeleteTemp( temp = t3 ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_binop_shl_respects_arithmetic_mode( self ) -> None:
		# << shares Add/Sub/Mult's wrap/check/saturate mode split (it CAN
		# overflow, unlike the other bitwise ops) - wrap_arithmetic here just
		# sidesteps the Check-mode/Result requirement, same as
		# test_binop_literal_on_left
		code = '\n'.join([
			'def main() -> None:',
			'	a: i32 = 1',
			'	with compiler.wrap_arithmetic:',
			'		b: i32 = a << 2',
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
			ir.ShlWrap( dest = t0, left = a, right = ir.Const( type = i32, value = 2 )),
			ir.Assign( dest = b, src = t0 ),
			ir.DeleteTemp( temp = t0 ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_binop_floordiv_and_mod_check_mode_emits_or_return( self ) -> None:
		# mirrors test_binop_check_mode_emits_or_return, but against
		# ZeroDivisionError instead of OverflowError, and independent of
		# arithmetic mode (there's no wrapped/saturated division)
		code = '\n'.join([
			'class ZeroDivisionError: pass',
			'',
			'@cstruct',
			'class Result[T,E]:',
			'	pass',
			'',
			'def checked() -> Result[None,ZeroDivisionError]:',
			'	a: i32 = 10',
			'	b: i32 = a // 3',
			'	c: i32 = a % 3',
		])
		mod = self._import( code )
		i32 = self.discovery.get_intrinsics()['i32']

		checked_fn = mod.get_local( 'checked' )
		if checked_fn.resolve is not None:
			checked_fn.resolve()
		zerodiv_cls = mod.get_local( 'ZeroDivisionError' )
		result_cls = mod.get_local( 'Result' )
		if result_cls.resolve is not None:
			result_cls.resolve()
		result_i32_zerodiv = self.discovery._get_or_create_specialization( result_cls, [ i32, zerodiv_cls ] )

		a = Variable( stem = 'a', qualname = '__test__.checked.a', file = Path( '__test__.py' ), line = 8, type = i32 )
		b = Variable( stem = 'b', qualname = '__test__.checked.b', file = Path( '__test__.py' ), line = 9, type = i32 )
		c = Variable( stem = 'c', qualname = '__test__.checked.c', file = Path( '__test__.py' ), line = 10, type = i32 )

		t0 = ir.Temp( type = result_i32_zerodiv, id = 0 ) # Div's Result
		t1 = ir.Temp( type = i32, id = 1 )                # unwrapped via OrReturn
		t2 = ir.Temp( type = result_i32_zerodiv, id = 2 ) # Mod's Result
		t3 = ir.Temp( type = i32, id = 3 )

		fn = self.compiler._lower( checked_fn )
		self._assert_ir( fn, [
			ir.FuncStart( name = '__test__.checked', params = [], return_type = checked_fn.return_type ),
			ir.Assign( dest = a, src = ir.Const( type = i32, value = 10 )),
			ir.DeclareTemp( temp = t0 ),
			ir.Div( dest = t0, left = a, right = ir.Const( type = i32, value = 3 )),
			ir.DeclareTemp( temp = t1 ),
			ir.OrReturn( dest = t1, value = t0 ),
			ir.Assign( dest = b, src = t1 ),
			ir.DeleteTemp( temp = t1 ),
			ir.DeleteTemp( temp = t0 ),
			ir.DeclareTemp( temp = t2 ),
			ir.Mod( dest = t2, left = a, right = ir.Const( type = i32, value = 3 )),
			ir.DeclareTemp( temp = t3 ),
			ir.OrReturn( dest = t3, value = t2 ),
			ir.Assign( dest = c, src = t3 ),
			ir.DeleteTemp( temp = t3 ),
			ir.DeleteTemp( temp = t2 ),
			ir.FuncEnd( name = '__test__.checked' ),
		])

	def test_binop_floordiv_without_zerodivision_result_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'class ZeroDivisionError: pass',
			'',
			'@cstruct',
			'class Result[T,E]:',
			'	pass',
			'',
			'def main() -> None:',
			'	a: i32 = 1',
			'	b: i32 = a // 1',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertIn( 'Result[_,ZeroDivisionError]', self.discovery.errors.errors[0] )
		self.assertIn( 'panic_arithmetic', self.discovery.errors.errors[0] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertEqual( kinds, [ 'FuncStart', 'Assign', 'Return', 'FuncEnd' ] )

	def test_binop_floordiv_inside_wrap_arithmetic_still_requires_result_and_uses_or_return( self ) -> None:
		# there's no wrapped/saturated division - being inside
		# wrap_arithmetic/saturate_arithmetic must NOT silently let division
		# through unchecked, and must NOT silently panic either. It stays a
		# real Result[T,ZeroDivisionError] dependency, caught at compile
		# time if the enclosing function can't propagate it, and consumed
		# via the normal OrReturn/OrJump path - panic only ever happens
		# inside an explicit panic_arithmetic block (see
		# test_binop_floordiv_without_zerodivision_result_is_a_compile_error
		# for the rejection case, and the panic case is covered by
		# test_binop_floordiv_and_mod_check_mode_emits_or_return's sibling
		# panic_arithmetic tests elsewhere in this class)
		code = '\n'.join([
			'class ZeroDivisionError: pass',
			'',
			'@cstruct',
			'class Result[T,E]:',
			'	pass',
			'',
			'def checked() -> Result[None,ZeroDivisionError]:',
			'	a: i32 = 10',
			'	with compiler.wrap_arithmetic:',
			'		b: i32 = a // 3',
		])
		mod = self._import( code )
		i32 = self.discovery.get_intrinsics()['i32']
		checked_fn = mod.get_local( 'checked' )
		if checked_fn.resolve is not None:
			checked_fn.resolve()

		fn = self.compiler._lower( checked_fn )
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertIn( 'Div', kinds )
		self.assertIn( 'OrReturn', kinds )
		self.assertNotIn( 'Unwrap', kinds ) # no panic - wrap_arithmetic doesn't imply panic_arithmetic

	def test_binop_floordiv_is_a_compile_error_in_every_non_panic_mode( self ) -> None:
		# division's Result[_,ZeroDivisionError] dependency can't be
		# sidestepped by any mode except panic_arithmetic - default (no
		# context) is already covered by
		# test_binop_floordiv_without_zerodivision_result_is_a_compile_error;
		# this confirms wrap_arithmetic and saturate_arithmetic don't offer
		# an escape hatch either, since neither has a wrapped/saturated
		# division opcode to fall back to
		for context in ( 'compiler.wrap_arithmetic', 'compiler.saturate_arithmetic' ):
			with self.subTest( context = context ):
				code = '\n'.join([
					'class ZeroDivisionError: pass',
					'',
					'@cstruct',
					'class Result[T,E]:',
					'	pass',
					'',
					'def main() -> None:', # -> None, not Result[_,ZeroDivisionError]
					'	a: i32 = 1',
					f'	with {context}:',
					'		b: i32 = a // 1',
					'	return',
				])
				disco = Discovery( import_builtins = False )
				comp = Compiler( disco )
				comp.import_code( code, filename = Path( '__test__.py' ))
				fn = comp._lower( disco.main )
				self.assertIn( 'Result[_,ZeroDivisionError]', disco.errors.errors[0] )
				self.assertIn( 'panic_arithmetic', disco.errors.errors[0] )
				kinds = [ type( instr ).__name__ for instr in fn.instructions ]
				self.assertEqual( kinds, [ 'FuncStart', 'Assign', 'Return', 'FuncEnd' ] )

	def test_binop_true_div_remains_unsupported( self ) -> None:
		# '/' (ast.Div) is deliberately not mapped to anything - there's no
		# float type in this language, and no real lib/ usage of '/' to
		# infer an intended meaning from (only '//'/'%' are used)
		code = '\n'.join([
			'def main() -> None:',
			'	a: i32 = 1',
			'	b: i32 = a / 1',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'unsupported binary operator', self.discovery.errors.errors[0] )

	def test_unaryop_invert_is_unconditional( self ) -> None:
		# ~ has no overflow concept - always a single opcode, works fine in
		# main() with no arithmetic context at all, same as the non-Shl
		# bitwise binops
		code = '\n'.join([
			'def main() -> None:',
			'	a: i32 = 1',
			'	b: i32 = ~a',
			'	return',
		])
		i32 = self.discovery.get_intrinsics()['i32']
		none_type = self.discovery.get_none_type()
		a = Variable( stem = 'a', qualname = 'main.a', file = Path( '__test__.py' ), line = 2, type = i32 )
		b = Variable( stem = 'b', qualname = 'main.b', file = Path( '__test__.py' ), line = 3, type = i32 )
		t0 = ir.Temp( type = i32, id = 0 )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = a, src = ir.Const( type = i32, value = 1 )),
			ir.DeclareTemp( temp = t0 ),
			ir.Invert( dest = t0, operand = a ),
			ir.Assign( dest = b, src = t0 ),
			ir.DeleteTemp( temp = t0 ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_unaryop_neg_wrap_arithmetic_context( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	a: i32 = 1',
			'	with compiler.wrap_arithmetic:',
			'		b: i32 = -a',
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
			ir.NegWrap( dest = t0, operand = a ),
			ir.Assign( dest = b, src = t0 ),
			ir.DeleteTemp( temp = t0 ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_unaryop_neg_check_mode_emits_or_return( self ) -> None:
		# mirrors test_binop_check_mode_emits_or_return - default (Check)
		# mode negation produces Result[T,OverflowError], immediately
		# consumed via OrReturn
		code = '\n'.join([
			'class OverflowError: pass',
			'',
			'@cstruct',
			'class Result[T,E]:',
			'	pass',
			'',
			'def checked() -> Result[None,OverflowError]:',
			'	a: i32 = 1',
			'	b: i32 = -a',
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

		t0 = ir.Temp( type = result_i32_overflow, id = 0 ) # NegCheck's Result
		t1 = ir.Temp( type = i32, id = 1 )                 # unwrapped via OrReturn

		fn = self.compiler._lower( checked_fn )
		self._assert_ir( fn, [
			ir.FuncStart( name = '__test__.checked', params = [], return_type = checked_fn.return_type ),
			ir.Assign( dest = a, src = ir.Const( type = i32, value = 1 )),
			ir.DeclareTemp( temp = t0 ),
			ir.NegCheck( dest = t0, operand = a ),
			ir.DeclareTemp( temp = t1 ),
			ir.OrReturn( dest = t1, value = t0 ),
			ir.Assign( dest = b, src = t1 ),
			ir.DeleteTemp( temp = t1 ),
			ir.DeleteTemp( temp = t0 ),
			ir.FuncEnd( name = '__test__.checked' ),
		])

	def test_unaryop_neg_without_arithmetic_context_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'class OverflowError: pass',
			'',
			'@cstruct',
			'class Result[T,E]:',
			'	pass',
			'',
			'def main() -> None:',
			'	a: i32 = 1',
			'	b: i32 = -a',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertIn( 'wrap_arithmetic', self.discovery.errors.errors[0] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertEqual( kinds, [ 'FuncStart', 'Assign', 'Return', 'FuncEnd' ] )

	def test_unaryop_not_remains_unsupported( self ) -> None:
		# no boolean-negation opcode exists in ir.py yet - flagged as a
		# separate, real design decision rather than guessed at here
		code = '\n'.join([
			'class bool: pass',
			'',
			'def main() -> None:',
			'	a: bool',
			'	b: bool = not a',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'unsupported unary operator', self.discovery.errors.errors[0] )

	# --- comparisons ---------------------------------------------------------

	def test_compare_eq_emits_cmp( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	a: i32 = 1',
			'	b: bool = a == 1',
			'	return',
		])
		i32 = self.discovery.get_intrinsics()['i32']
		bool_cls = self.discovery.get_intrinsics()['bool']
		none_type = self.discovery.get_none_type()
		a = Variable( stem = 'a', qualname = 'main.a', file = Path( '__test__.py' ), line = 2, type = i32 )
		b = Variable( stem = 'b', qualname = 'main.b', file = Path( '__test__.py' ), line = 3, type = bool_cls )
		t0 = ir.Temp( type = bool_cls, id = 0 )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = a, src = ir.Const( type = i32, value = 1 )),
			ir.DeclareTemp( temp = t0 ),
			ir.Cmp( dest = t0, op = ir.CmpOp.EQ, left = a, right = ir.Const( type = i32, value = 1 )),
			ir.Assign( dest = b, src = t0 ),
			ir.DeleteTemp( temp = t0 ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_compare_all_ops_map_to_the_right_cmpop( self ) -> None:
		cases = [
			( '==', ir.CmpOp.EQ ),
			( '!=', ir.CmpOp.NE ),
			( '<', ir.CmpOp.LT ),
			( '<=', ir.CmpOp.LE ),
			( '>', ir.CmpOp.GT ),
			( '>=', ir.CmpOp.GE ),
		]
		i32 = self.discovery.get_intrinsics()['i32']
		bool_cls = self.discovery.get_intrinsics()['bool']
		for py_op, expected_cmpop in cases:
			with self.subTest( op = py_op ):
				disco = Discovery( import_builtins = False )
				comp = Compiler( disco )
				comp.import_code( '\n'.join([
					'def main() -> None:',
					'	a: i32 = 1',
					f'	b: bool = a {py_op} 1',
					'	return',
				]), filename = Path( '__test__.py' ))
				fn = comp._lower( disco.main )
				cmp_instr = next( i for i in fn.instructions if isinstance( i, ir.Cmp ))
				self.assertEqual( cmp_instr.op, expected_cmpop )

	def test_compare_literal_on_left( self ) -> None:
		# expected type flows from whichever side is NOT the bare literal -
		# mirrors test_binop_literal_on_left
		code = '\n'.join([
			'def main() -> None:',
			'	a: i32 = 1',
			'	b: bool = 1 < a',
			'	return',
		])
		i32 = self.discovery.get_intrinsics()['i32']
		bool_cls = self.discovery.get_intrinsics()['bool']
		none_type = self.discovery.get_none_type()
		a = Variable( stem = 'a', qualname = 'main.a', file = Path( '__test__.py' ), line = 2, type = i32 )
		b = Variable( stem = 'b', qualname = 'main.b', file = Path( '__test__.py' ), line = 3, type = bool_cls )
		t0 = ir.Temp( type = bool_cls, id = 0 )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = a, src = ir.Const( type = i32, value = 1 )),
			ir.DeclareTemp( temp = t0 ),
			ir.Cmp( dest = t0, op = ir.CmpOp.LT, left = ir.Const( type = i32, value = 1 ), right = a ),
			ir.Assign( dest = b, src = t0 ),
			ir.DeleteTemp( temp = t0 ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_compare_chained_is_not_yet_supported( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	a: i32 = 1',
			'	b: bool = 0 < a < 2',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'chained comparisons are not yet supported', self.discovery.errors.errors[0] )

	def test_compare_is_not_supported( self ) -> None:
		# `is`/`is not` are reserved for eventual tagged-union type
		# narrowing (see TODO.txt), not a plain identity comparison -
		# deliberately left unsupported rather than guessed at
		code = '\n'.join([
			'def main() -> None:',
			'	a: i32 = 1',
			'	b: bool = a is None',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'unsupported comparison operator', self.discovery.errors.errors[0] )

	# --- boolean operators (and/or) -------------------------------------------

	def test_boolop_and_shape( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	a: bool',
			'	b: bool',
			'	c: bool = a and b',
			'	return',
		])
		bool_cls = self.discovery.get_intrinsics()['bool']
		none_type = self.discovery.get_none_type()
		a = Variable( stem = 'a', qualname = 'main.a', file = Path( '__test__.py' ), line = 2, type = bool_cls )
		b = Variable( stem = 'b', qualname = 'main.b', file = Path( '__test__.py' ), line = 3, type = bool_cls )
		c = Variable( stem = 'c', qualname = 'main.c', file = Path( '__test__.py' ), line = 4, type = bool_cls )
		t0 = ir.Temp( type = bool_cls, id = 0 )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.DeclareTemp( temp = t0 ),
			ir.Assign( dest = t0, src = a ),
			ir.JumpIfFalse( cond = t0, target = '__booland_0__' ),
			ir.Assign( dest = t0, src = b ),
			ir.Label( name = '__booland_0__' ),
			ir.Assign( dest = c, src = t0 ),
			ir.DeleteTemp( temp = t0 ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_boolop_or_shape( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	a: bool',
			'	b: bool',
			'	c: bool = a or b',
			'	return',
		])
		bool_cls = self.discovery.get_intrinsics()['bool']
		none_type = self.discovery.get_none_type()
		a = Variable( stem = 'a', qualname = 'main.a', file = Path( '__test__.py' ), line = 2, type = bool_cls )
		b = Variable( stem = 'b', qualname = 'main.b', file = Path( '__test__.py' ), line = 3, type = bool_cls )
		c = Variable( stem = 'c', qualname = 'main.c', file = Path( '__test__.py' ), line = 4, type = bool_cls )
		t0 = ir.Temp( type = bool_cls, id = 0 )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.DeclareTemp( temp = t0 ),
			ir.Assign( dest = t0, src = a ),
			ir.JumpIfTrue( cond = t0, target = '__boolor_0__' ),
			ir.Assign( dest = t0, src = b ),
			ir.Label( name = '__boolor_0__' ),
			ir.Assign( dest = c, src = t0 ),
			ir.DeleteTemp( temp = t0 ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_boolop_and_short_circuits_on_first_falsy( self ) -> None:
		# three operands: only the first two should ever be lowered/jumped
		# on if the first is falsy at runtime - but since this is static
		# lowering (not interpretation), what we can actually verify is the
		# STATIC shape: two JumpIfFalse checks (one per non-last operand)
		code = '\n'.join([
			'def main() -> None:',
			'	a: bool',
			'	b: bool',
			'	c: bool',
			'	d: bool = a and b and c',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertEqual( kinds.count( 'JumpIfFalse' ), 2 )
		self.assertEqual( kinds.count( 'Label' ), 1 )

	# --- if statements ---------------------------------------------------------

	def test_if_without_else_shape( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	a: bool',
			'	if a:',
			'		b: i32 = 1',
			'	return',
		])
		bool_cls = self.discovery.get_intrinsics()['bool']
		i32 = self.discovery.get_intrinsics()['i32']
		none_type = self.discovery.get_none_type()
		a = Variable( stem = 'a', qualname = 'main.a', file = Path( '__test__.py' ), line = 2, type = bool_cls )
		b = Variable( stem = 'b', qualname = 'main.b', file = Path( '__test__.py' ), line = 4, type = i32 )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.JumpIfFalse( cond = a, target = '__if_else_0__' ),
			ir.Assign( dest = b, src = ir.Const( type = i32, value = 1 )),
			ir.Label( name = '__if_else_0__' ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_if_with_else_shape( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	a: bool',
			'	if a:',
			'		b: i32 = 1',
			'	else:',
			'		b: i32 = 2',
			'	return',
		])
		bool_cls = self.discovery.get_intrinsics()['bool']
		i32 = self.discovery.get_intrinsics()['i32']
		none_type = self.discovery.get_none_type()
		a = Variable( stem = 'a', qualname = 'main.a', file = Path( '__test__.py' ), line = 2, type = bool_cls )
		b_then = Variable( stem = 'b', qualname = 'main.b', file = Path( '__test__.py' ), line = 4, type = i32 )
		b_else = Variable( stem = 'b', qualname = 'main.b', file = Path( '__test__.py' ), line = 6, type = i32 )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.JumpIfFalse( cond = a, target = '__if_else_0__' ),
			ir.Assign( dest = b_then, src = ir.Const( type = i32, value = 1 )),
			ir.Jump( target = '__if_end_1__' ),
			ir.Label( name = '__if_else_0__' ),
			ir.Assign( dest = b_else, src = ir.Const( type = i32, value = 2 )),
			ir.Label( name = '__if_end_1__' ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_if_elif_else_chains_via_nested_orelse( self ) -> None:
		# elif is just a nested If inside orelse in the AST - confirms it
		# "just works" through the same recursive _lower_stmt dispatch, no
		# special-casing needed
		code = '\n'.join([
			'def main() -> None:',
			'	a: bool',
			'	c: bool',
			'	if a:',
			'		x: i32 = 1',
			'	elif c:',
			'		x: i32 = 2',
			'	else:',
			'		x: i32 = 3',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		# outer if and the nested elif-as-If each have their own orelse (the
		# elif chain, and its own else respectively), so each contributes
		# its own else-Label + end-Label + skip-Jump pair - two JumpIfFalse
		# (one per test), three Assigns (one per branch), two Jumps and
		# four Labels (one else + one end, per level)
		self.assertEqual( kinds.count( 'JumpIfFalse' ), 2 )
		self.assertEqual( kinds.count( 'Jump' ), 2 )
		self.assertEqual( kinds.count( 'Label' ), 4 )
		self.assertEqual( kinds.count( 'Assign' ), 3 )

	# --- match statements ------------------------------------------------------

	def test_match_union_shape( self ) -> None:
		code = '\n'.join([
			'@union',
			'class Foo:',
			'	Bar: i32',
			'	Baz: usize',
			'',
			'def get() -> Foo:',
			'	return Foo.Bar( 5 )',
			'',
			'def main() -> None:',
			'	f: Foo = get()',
			'	match f:',
			'		case Foo.Bar( x ):',
			'			y: i32 = x',
			'		case Foo.Baz( z ):',
			'			w: usize = z',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		# one Cmp per case (tag == ordinal), one GetAttr per case for the
		# payload's `data` field plus one more for the specific `v_member`
		# field (2 each), one JumpIfFalse per case's if, one else-Label,
		# one Jump+Label for the if/elif split
		self.assertEqual( kinds.count( 'Cmp' ), 2 )
		self.assertEqual( kinds.count( 'JumpIfFalse' ), 4 ) # 2 booland short-circuits + 2 if-tests

	def test_match_union_construction_and_extraction_round_trip( self ) -> None:
		# construct with one member, match should take that member's arm
		# and correctly extract its value (verified via the field names/
		# types actually referenced, not by literally executing the IR)
		code = '\n'.join([
			'@union',
			'class Foo:',
			'	Bar: i32',
			'',
			'def main() -> None:',
			'	f: Foo = Foo.Bar( 5 )',
			'	match f:',
			'		case Foo.Bar( x ):',
			'			y: i32 = x',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		getattrs = [ i for i in fn.instructions if isinstance( i, ir.GetAttr ) ]
		self.assertEqual( [ g.attr for g in getattrs ], [ 'tag', 'data', 'v_Bar' ] )

	def test_match_result_ok_err_shape( self ) -> None:
		# Result.Ok/Result.Err are special-cased directly (is_ok()/is_err()
		# + _payload.ok/._payload.err) rather than routed through
		# _tagged_union_storage, since Result predates @union and isn't a
		# real TaggedUnion
		code = '\n'.join([
			'class MyError: pass',
			'',
			'@cunion',
			'class ResultPayload[T,E]:',
			'	ok: T',
			'	err: E',
			'',
			'@cstruct',
			'class Result[T,E]:',
			'	_payload: ResultPayload[T,E]',
			'	_tag: u8',
			'',
			'	@staticmethod',
			'	def Ok( val: T ) -> Result[T,E]:',
			'		return Result.__allocate__( _payload = ResultPayload( ok = val ), _tag = 0 )',
			'',
			'	def is_ok( self ) -> bool:',
			'		return self._tag == 0',
			'',
			'	def is_err( self ) -> bool:',
			'		return self._tag == 1',
			'',
			'def get() -> Result[i32,MyError]:',
			'	return Result.Ok( 1 )',
			'',
			'def main() -> None:',
			'	r: Result[i32,MyError] = get()',
			'	match r:',
			'		case Result.Ok( v ):',
			'			x: i32 = v',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertIn( 'Call', kinds ) # is_ok()
		getattrs = [ i for i in fn.instructions if isinstance( i, ir.GetAttr ) ]
		self.assertEqual( [ g.attr for g in getattrs ], [ '_payload', 'ok' ] )

	def test_match_wildcard_binds_whole_subject( self ) -> None:
		code = '\n'.join([
			'@union',
			'class Foo:',
			'	Bar: i32',
			'',
			'def main() -> None:',
			'	f: Foo = Foo.Bar( 5 )',
			'	match f:',
			'		case whatever:',
			'			pass',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_match_guard_is_not_yet_supported( self ) -> None:
		code = '\n'.join([
			'@union',
			'class Foo:',
			'	Bar: i32',
			'',
			'def main() -> None:',
			'	f: Foo = Foo.Bar( 5 )',
			'	match f:',
			'		case Foo.Bar( x ) if x > 0:',
			'			pass',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'guards', self.discovery.errors.errors[0] )

	def test_match_unknown_member_is_rejected( self ) -> None:
		code = '\n'.join([
			'@union',
			'class Foo:',
			'	Bar: i32',
			'',
			'def main() -> None:',
			'	f: Foo = Foo.Bar( 5 )',
			'	match f:',
			'		case Foo.NotAMember( x ):',
			'			pass',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'has no member', self.discovery.errors.errors[0] )

	# --- generic field-type substitution (_attr_lookup) -----------------------

	def test_attr_lookup_substitutes_generic_field_type_through_specialization( self ) -> None:
		# Result[T,E]'s _payload field is declared using Result's OWN
		# type params (ResultPayload[T,E]) - accessing it through a
		# concrete Result[i32,MyError] must substitute T->i32, E->MyError,
		# not return the bare TypeVars
		code = '\n'.join([
			'class MyError: pass',
			'',
			'@cunion',
			'class Payload[T,E]:',
			'	ok: T',
			'	err: E',
			'',
			'@cstruct',
			'class Holder[T,E]:',
			'	payload: Payload[T,E]',
			'',
			'def main( h: Holder[i32,MyError] ) -> None:',
			'	x: i32 = h.payload.ok',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		i32 = self.discovery.get_intrinsics()['i32']
		getattr_ok = next( i for i in fn.instructions if isinstance( i, ir.GetAttr ) and i.attr == 'ok' )
		self.assertEqual( getattr_ok.dest.type, i32 )

	# --- multi-branch overload dispatch (ConditionalDispatch) -----------------

	def test_conditional_dispatch_shape( self ) -> None:
		# a union-typed argument (x: A|B) makes foo(x) ambiguous at compile
		# time - resolve_call returns real branches, lowered here as a
		# runtime tag check (on the synthesized anonymous union) picking
		# between the two real implementations
		code = '\n'.join([
			'class A: pass',
			'class B: pass',
			'',
			'def foo( v: A ) -> None:',
			'	pass',
			'',
			'def foo( v: B ) -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	x: A|B',
			'	foo( x )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertEqual( kinds.count( 'Cmp' ), 1 )
		self.assertEqual( kinds.count( 'Call' ), 2 ) # one per possible target - only one runs at runtime
		self.assertEqual( kinds.count( 'JumpIfFalse' ), 1 )
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) ]
		self.assertTrue( all( c.target.qualname == '__test__.foo' for c in calls )) # both overload members share the same qualname
		self.assertEqual( len( { id( c.target ) for c in calls } ), 2 ) # but are two DIFFERENT Function objects (distinct implementations)

	def test_conditional_dispatch_unwraps_union_argument_to_concrete_leaf( self ) -> None:
		# the Call emitted for each branch must pass the UNWRAPPED concrete
		# value (via data.v_<leaf>), not the raw union operand - foo(v: A)
		# expects a real A, not an A|B
		code = '\n'.join([
			'class A: pass',
			'class B: pass',
			'',
			'def foo( v: A ) -> None:',
			'	pass',
			'',
			'def foo( v: B ) -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	x: A|B',
			'	foo( x )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) ]
		a_cls = self.discovery.modules['__test__'].get_local( 'A' )
		b_cls = self.discovery.modules['__test__'].get_local( 'B' )
		call_arg_types = [ c.args[0].type for c in calls ]
		self.assertTrue( any( t is a_cls for t in call_arg_types ))
		self.assertTrue( any( t is b_cls for t in call_arg_types )) # not the A|B union type
		getattrs = [ i.attr for i in fn.instructions if isinstance( i, ir.GetAttr ) ]
		self.assertIn( 'v_A', getattrs )
		self.assertIn( 'v_B', getattrs )

	def test_conditional_dispatch_never_considers_a_candidate_unrelated_to_the_argument( self ) -> None:
		# mirrors lib/builtins/__init__.py's real len() shape: three plain
		# candidates (str/bytes/bytearray), called with a NARROWER union
		# (bytes|bytearray) that doesn't include str at all - the emitted
		# IR must never reference len(x:str), not just "correctly not call
		# it at runtime" - it should never even be scheduled/considered
		code = '\n'.join([
			'class strlike: pass',
			'class bytes: pass',
			'class bytearray: pass',
			'',
			'def flen( x: strlike ) -> None:',
			'	pass',
			'',
			'def flen( x: bytes ) -> None:',
			'	pass',
			'',
			'def flen( x: bytearray ) -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	x: bytes|bytearray',
			'	flen( x )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		strlike_cls = self.discovery.modules['__test__'].get_local( 'strlike' )
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( len( calls ), 2 ) # exactly bytes and bytearray - never a third for strlike
		self.assertTrue( all( c.target.parameters[0].type is not strlike_cls for c in calls ))

	# --- bare literal arguments to overloaded calls -----------------------------

	def test_overload_literal_arg_resolves_via_unique_candidate_type( self ) -> None:
		code = '\n'.join([
			'class str: pass',
			'',
			'def foo( x: i32 ) -> None:',
			'	pass',
			'',
			'def foo( x: str ) -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	foo( 5 )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		i32 = self.discovery.get_intrinsics()['i32']
		call = next( i for i in fn.instructions if isinstance( i, ir.Call ) )
		self.assertEqual( call.args, [ ir.Const( type = i32, value = 5 ) ] )

	def test_overload_literal_arg_string_kind_only_matches_str_candidate( self ) -> None:
		code = '\n'.join([
			'class str: pass',
			'',
			'def foo( x: i32 ) -> None:',
			'	pass',
			'',
			'def foo( x: str ) -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	foo( "hi" )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		str_cls = self.discovery.modules['__test__'].get_local( 'str' )
		call = next( i for i in fn.instructions if isinstance( i, ir.Call ) )
		self.assertEqual( call.args, [ ir.Const( type = str_cls, value = 'hi' ) ] )

	def test_overload_literal_arg_ambiguous_between_candidates_is_rejected( self ) -> None:
		code = '\n'.join([
			'def foo( x: i32 ) -> None:',
			'	pass',
			'',
			'def foo( x: u8 ) -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	foo( 5 )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'ambiguous literal argument', self.discovery.errors.errors[0] )

	def test_overload_literal_kwarg_resolves_by_name( self ) -> None:
		code = '\n'.join([
			'class str: pass',
			'',
			'def foo( x: i32 ) -> None:',
			'	pass',
			'',
			'def foo( x: str ) -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	foo( x = 5 )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		i32 = self.discovery.get_intrinsics()['i32']
		call = next( i for i in fn.instructions if isinstance( i, ir.Call ) )
		self.assertEqual( call.kwargs, { 'x': ir.Const( type = i32, value = 5 ) } )

	# --- generic function monomorphization (Name[T](...)) ----------------------

	def test_generic_function_call_monomorphizes( self ) -> None:
		code = '\n'.join([
			'def alloc[T]( count: usize ) -> usize:',
			'	with compiler.wrap_arithmetic:',
			'		return count * compiler.sizeof( T )',
			'',
			'def main() -> None:',
			'	x: usize = alloc[u32]( 10 )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		call = next( i for i in fn.instructions if isinstance( i, ir.Call ) )
		self.assertEqual( call.target.qualname, '__test__.alloc[intrinsics.u32]' )
		usize = self.discovery.get_intrinsics()['usize']
		self.assertEqual( call.target.return_type, usize )

	def test_generic_function_specializations_are_memoized( self ) -> None:
		# two call sites specializing the same [T] the same way must
		# schedule/reference the SAME monomorphized Function object, not a
		# fresh copy each time - otherwise it'd get compiled twice
		code = '\n'.join([
			'def alloc[T]( count: usize ) -> usize:',
			'	with compiler.wrap_arithmetic:',
			'		return count * compiler.sizeof( T )',
			'',
			'def main() -> None:',
			'	a: usize = alloc[u32]( 10 )',
			'	b: usize = alloc[u32]( 20 )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( len( calls ), 2 )
		self.assertIs( calls[0].target, calls[1].target )

	def test_generic_function_call_distinguishes_different_specializations( self ) -> None:
		code = '\n'.join([
			'def alloc[T]( count: usize ) -> usize:',
			'	with compiler.wrap_arithmetic:',
			'		return count * compiler.sizeof( T )',
			'',
			'def main() -> None:',
			'	a: usize = alloc[u32]( 10 )',
			'	b: usize = alloc[u8]( 10 )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( len( calls ), 2 )
		self.assertIsNot( calls[0].target, calls[1].target )
		self.assertEqual( calls[0].target.qualname, '__test__.alloc[intrinsics.u32]' )
		self.assertEqual( calls[1].target.qualname, '__test__.alloc[intrinsics.u8]' )

	def test_generic_function_call_wrong_type_arg_count_rejected( self ) -> None:
		code = '\n'.join([
			'def alloc[T]( count: usize ) -> usize:',
			'	return count',
			'',
			'def main() -> None:',
			'	x: usize = alloc[u32,u8]( 10 )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'expects 1 type argument', self.discovery.errors.errors[0] )

	def test_generic_function_call_non_type_arg_rejected( self ) -> None:
		code = '\n'.join([
			'def alloc[T]( count: usize ) -> usize:',
			'	return count',
			'',
			'def main() -> None:',
			'	y: usize = 1',
			'	x: usize = alloc[y]( 10 )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'is not a type', self.discovery.errors.errors[0] )

	# --- compiler.sizeof(T) ----------------------------------------------------

	def test_compiler_sizeof_folds_to_const( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	x: usize = compiler.sizeof( u32 )',
			'	return',
		])
		usize = self.discovery.get_intrinsics()['usize']
		none_type = self.discovery.get_none_type()
		x = Variable( stem = 'x', qualname = 'main.x', file = Path( '__test__.py' ), line = 2, type = usize )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = x, src = ir.Const( type = usize, value = 4 ) ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_compiler_sizeof_each_intrinsic( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	a: usize = compiler.sizeof( u8 )',
			'	b: usize = compiler.sizeof( i64 )',
			'	c: usize = compiler.sizeof( bool )',
			'	d: usize = compiler.sizeof( Ptr )', # bare, unsubscripted - Ptr[u8] needs generic-subscript resolution (separate item)
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		consts = [ i.src.value for i in fn.instructions if isinstance( i, ir.Assign ) ]
		self.assertEqual( consts, [ 1, 8, 1, 8 ] )

	def test_compiler_sizeof_typevar_is_rejected( self ) -> None:
		# calling foo() unspecialized (never through foo[u8](...)) leaves T
		# an abstract, unbound TypeVar in its own body - sizeof needs a
		# concrete type
		code = '\n'.join([
			'def foo[T]() -> None:',
			'	x: usize = compiler.sizeof( T )',
			'	return',
		])
		self._import( code )
		foo_fn = self.discovery.modules['__test__'].get_local( 'foo' )
		if foo_fn.resolve is not None:
			foo_fn.resolve()
		self.compiler._lower( foo_fn )
		self.assertIn( 'unbound generic type parameter', self.discovery.errors.errors[0] )

	def test_compiler_sizeof_class_is_not_yet_supported( self ) -> None:
		code = '\n'.join([
			'class Foo: pass',
			'',
			'def main() -> None:',
			'	x: usize = compiler.sizeof( Foo )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'is not supported yet', self.discovery.errors.errors[0] )

	# --- loops (while / for / break / continue) -----------------------------

	def test_while_shape( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	a: bool',
			'	while a:',
			'		x: i32 = 1',
			'	return',
		])
		bool_cls = self.discovery.get_intrinsics()['bool']
		i32 = self.discovery.get_intrinsics()['i32']
		none_type = self.discovery.get_none_type()
		a = Variable( stem = 'a', qualname = 'main.a', file = Path( '__test__.py' ), line = 2, type = bool_cls )
		x = Variable( stem = 'x', qualname = 'main.x', file = Path( '__test__.py' ), line = 4, type = i32 )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Label( name = '__while_start_0__' ),
			ir.JumpIfFalse( cond = a, target = '__while_end_1__' ),
			ir.Assign( dest = x, src = ir.Const( type = i32, value = 1 )),
			ir.Jump( target = '__while_start_0__' ),
			ir.Label( name = '__while_end_1__' ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_while_else_is_rejected( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	a: bool',
			'	while a:',
			'		pass',
			'	else:',
			'		pass',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'while/else', self.discovery.errors.errors[0] )

	def test_break_outside_loop_is_rejected( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	break',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'break outside a loop', self.discovery.errors.errors[0] )

	def test_continue_outside_loop_is_rejected( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	continue',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'continue outside a loop', self.discovery.errors.errors[0] )

	def test_break_and_continue_target_the_innermost_loop( self ) -> None:
		# a break/continue inside a nested inner while must target the
		# inner loop's own labels, not the outer loop's - and once the
		# inner loop's lowering finishes, the outer loop's labels become
		# active again for anything after it in the outer body
		code = '\n'.join([
			'def main() -> None:',
			'	a: bool',
			'	b: bool',
			'	while a:',
			'		while b:',
			'			break',
			'			continue',
			'		break',
			'		continue',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		jumps = [ instr for instr in fn.instructions if isinstance( instr, ir.Jump ) ]
		# order in the instruction stream: inner break, inner continue, the
		# inner while's own back-edge, outer break, outer continue, the
		# outer while's own back-edge
		inner_break, inner_continue, _inner_back_edge, outer_break, outer_continue, _outer_back_edge = jumps
		self.assertNotEqual( inner_break.target, outer_break.target )
		self.assertNotEqual( inner_continue.target, outer_continue.target )
		self.assertNotEqual( inner_break.target, inner_continue.target ) # inner break -> inner end label, inner continue -> inner start label

	def test_for_target_must_be_a_plain_name( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	xs: i32',
			'	for xs[0] in range( 3 ):',
			'		pass',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'plain name', self.discovery.errors.errors[0] )

	def test_for_range_single_arg_shape( self ) -> None:
		# for i in range(count): reuses `i` if it already exists (matching
		# lib/builtins/__init__.py's str.concat, which pre-declares
		# `i: usize = 0` before its own for loop), and defaults the implicit
		# start=0 to usize when declaring a fresh target
		code = '\n'.join([
			'def main() -> None:',
			'	count: usize = 5',
			'	for i in range( count ):',
			'		x: usize = i',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertEqual( kinds.count( 'Label' ), 3 ) # start, continue, end
		self.assertEqual( kinds.count( 'Cmp' ), 1 )
		self.assertEqual( kinds.count( 'JumpIfFalse' ), 1 )
		self.assertEqual( kinds.count( 'AddWrap' ), 1 ) # the hidden increment - always AddWrap, regardless of ambient arithmetic mode
		self.assertEqual( kinds.count( 'Jump' ), 1 ) # the back-edge to start
		i_var = self.discovery.modules['__test__'].get_local( 'main' ).get_local( 'i' )
		usize = self.discovery.get_intrinsics()['usize']
		self.assertEqual( i_var.type, usize )

	def test_for_range_two_arg_form( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	a: usize = 1',
			'	b: usize = 5',
			'	for i in range( a, b ):',
			'		pass',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		usize = self.discovery.get_intrinsics()['usize']
		a = Variable( stem = 'a', qualname = 'main.a', file = Path( '__test__.py' ), line = 2, type = usize )
		# the initial bind (`i = a`) is the first Assign after main.b's own Assign
		assigns = [ instr for instr in fn.instructions if isinstance( instr, ir.Assign ) ]
		self.assertEqual( assigns[2].dest.stem, 'i' )
		self.assertEqual( assigns[2].src, a )

	def test_for_range_rejects_three_args( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	for i in range( 0, 5, 2 ):',
			'		pass',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( '1 or 2 arguments', self.discovery.errors.errors[0] )

	def test_for_over_indexable_shape( self ) -> None:
		# for v in <obj>: where obj's type declares both __len__ and
		# __getitem__ desugars to a counter-based while, reusing
		# _expr_Subscript's own __getitem__ resolution (with its Result
		# auto-unwrap) for the per-iteration bind
		code = '\n'.join([
			'@cstruct',
			'class Box:',
			'	_len: usize',
			'',
			'	def __len__( self ) -> usize:',
			'		return self._len',
			'',
			'	def __getitem__( self, i: usize ) -> i32:',
			'		return 1',
			'',
			'def main( b: Box ) -> None:',
			'	for v in b:',
			'		x: i32 = v',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertEqual( kinds.count( 'Call' ), 2 ) # __len__() once, __getitem__(i) once per compiled iteration-body
		self.assertEqual( kinds.count( 'Label' ), 3 )
		self.assertEqual( kinds.count( 'AddWrap' ), 1 )
		# Result.or_return-flavored auto-unwrap only fires when __getitem__
		# actually returns a Result - this Box's __getitem__ returns plain
		# i32, so no OrReturn/OrJump should appear
		self.assertNotIn( 'OrReturn', kinds )
		self.assertNotIn( 'OrJump', kinds )

	def test_for_over_indexable_missing_dunders_is_rejected( self ) -> None:
		code = '\n'.join([
			'class Box:',
			'	pass',
			'',
			'def main( b: Box ) -> None:',
			'	for v in b:',
			'		pass',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( '__len__', self.discovery.errors.errors[0] )
		self.assertIn( '__getitem__', self.discovery.errors.errors[0] )

	def test_subscript_with_getitem_resolves_and_consumes_result( self ) -> None:
		# obj[i] is sugar for obj.__getitem__(i).or_return() whenever
		# __getitem__ can fail (mirrors slice.__getitem__'s real signature,
		# Result[T,IndexError])
		code = '\n'.join([
			'class MyError: pass',
			'',
			'@cstruct',
			'class Result[T,E]:',
			'	x: T',
			'',
			'@cstruct',
			'class Box:',
			'	y: i32',
			'',
			'	def __getitem__( self, i: usize ) -> Result[i32,MyError]:',
			'		return Result.__allocate__( x = self.y )',
			'',
			'def foo( b: Box, i: usize ) -> Result[i32,MyError]:',
			'	v: i32 = b[i]',
			'	return Result.Err( MyError() )',
		])
		self._import( code )
		foo_fn = self.discovery.modules['__test__'].get_local( 'foo' )
		if foo_fn.resolve is not None:
			foo_fn.resolve()
		fn = self.compiler._lower( foo_fn )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertIn( 'OrReturn', kinds )
		# Box.__getitem__'s own real body (calling Box.__allocate__/Result.__allocate__
		# from inside itself) is unaffected - only the *call site* `b[i]` goes
		# through this new path, not Box.__getitem__'s own internals
		getitem_errors = [ e for e in self.discovery.errors.errors if 'b[i]' in e or '__getitem__' in e ]
		self.assertEqual( getitem_errors, [] )

	def test_if_body_recovery_boundary_does_not_stop_orelse( self ) -> None:
		# one bad statement inside the if-body doesn't prevent orelse (or
		# anything after the if) from still being lowered - same recovery
		# boundary as everywhere else
		code = '\n'.join([
			'def main() -> None:',
			'	a: bool',
			'	if a:',
			'		x: i32 = undefined_name',
			'	else:',
			'		b: i32 = 2',
			'	c: i32 = 3',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertTrue( any( "'undefined_name' is not defined" in e for e in self.discovery.errors.errors ))
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertEqual( kinds.count( 'Assign' ), 2 ) # b and c, both still lowered

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

	# --- object construction (Class.__allocate__) ---------------------------

	def test_allocate_emits_allocate_instruction( self ) -> None:
		# Class.__allocate__(...) is a compiler-synthesized pseudo-method,
		# never in any class's .names - recognized textually in
		# _try_lower_allocate_call, same spirit as defer/errdefer
		code = '\n'.join([
			'@cstruct',
			'class Foo:',
			'	x: i32',
			'	y: i32',
			'',
			'	@staticmethod',
			'	def make( v: i32 ) -> Foo:',
			'		return Foo.__allocate__( x = v, y = 2 )',
		])
		mod = self._import( code )
		foo_cls = mod.get_local( 'Foo' )
		foo_cls.resolve()
		make_fn = foo_cls.get_local( 'make' )
		if make_fn.resolve is not None:
			make_fn.resolve()
		i32 = self.discovery.get_intrinsics()['i32']
		v = make_fn.parameters[0]
		t0 = ir.Temp( type = foo_cls, id = 0 )

		fn = self.compiler._lower( make_fn )
		self._assert_ir( fn, [
			ir.FuncStart( name = make_fn.qualname, params = [ v ], return_type = foo_cls ),
			ir.DeclareTemp( temp = t0 ),
			ir.Allocate( dest = t0, cls = foo_cls, fields = { 'x': v, 'y': ir.Const( type = i32, value = 2 ) } ),
			ir.Return( value = t0 ),
			ir.DeleteTemp( temp = t0 ),
			ir.FuncEnd( name = make_fn.qualname ),
		])

	def test_allocate_dest_type_uses_expected_type_when_given( self ) -> None:
		# res: Foo[i32] = Foo.__allocate__(...) - the annotation's
		# specialization is the dest temp's type, not the bare generic class
		code = '\n'.join([
			'@cstruct',
			'class Foo[T]:',
			'	x: T',
			'',
			'	@staticmethod',
			'	def make( v: T ) -> Foo[T]:',
			'		res: Foo[T] = Foo.__allocate__( x = v )',
			'		return res',
		])
		mod = self._import( code )
		foo_cls = mod.get_local( 'Foo' )
		foo_cls.resolve()
		make_fn = foo_cls.get_local( 'make' )
		if make_fn.resolve is not None:
			make_fn.resolve()
		fn = self.compiler._lower( make_fn )
		allocate_instr = next( i for i in fn.instructions if isinstance( i, ir.Allocate ))
		self.assertIsInstance( allocate_instr.dest.type, Specialization )
		self.assertIs( allocate_instr.dest.type.base, foo_cls )

	def test_allocate_external_call_is_rejected( self ) -> None:
		# strictly private per SYNTAX.md - only callable from a method of
		# the same class
		code = '\n'.join([
			'@cstruct',
			'class Foo:',
			'	x: i32',
			'',
			'def main() -> None:',
			'	f: Foo = Foo.__allocate__( x = 1 )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'is private', self.discovery.errors.errors[0] )

	def test_allocate_missing_field_is_rejected( self ) -> None:
		code = '\n'.join([
			'@cstruct',
			'class Foo:',
			'	x: i32',
			'	y: i32',
			'',
			'	@staticmethod',
			'	def make() -> Foo:',
			'		return Foo.__allocate__( x = 1 )',
		])
		mod = self._import( code )
		foo_cls = mod.get_local( 'Foo' )
		foo_cls.resolve()
		make_fn = foo_cls.get_local( 'make' )
		if make_fn.resolve is not None:
			make_fn.resolve()
		self.compiler._lower( make_fn )
		self.assertIn( 'missing field', self.discovery.errors.errors[0] )
		self.assertIn( 'y', self.discovery.errors.errors[0] )

	def test_allocate_extra_field_is_rejected( self ) -> None:
		code = '\n'.join([
			'@cstruct',
			'class Foo:',
			'	x: i32',
			'',
			'	@staticmethod',
			'	def make() -> Foo:',
			'		return Foo.__allocate__( x = 1, z = 2 )',
		])
		mod = self._import( code )
		foo_cls = mod.get_local( 'Foo' )
		foo_cls.resolve()
		make_fn = foo_cls.get_local( 'make' )
		if make_fn.resolve is not None:
			make_fn.resolve()
		self.compiler._lower( make_fn )
		self.assertIn( 'no field', self.discovery.errors.errors[0] )
		self.assertIn( 'z', self.discovery.errors.errors[0] )

	def test_allocate_positional_args_rejected( self ) -> None:
		code = '\n'.join([
			'@cstruct',
			'class Foo:',
			'	x: i32',
			'',
			'	@staticmethod',
			'	def make() -> Foo:',
			'		return Foo.__allocate__( 1 )',
		])
		mod = self._import( code )
		foo_cls = mod.get_local( 'Foo' )
		foo_cls.resolve()
		make_fn = foo_cls.get_local( 'make' )
		if make_fn.resolve is not None:
			make_fn.resolve()
		self.compiler._lower( make_fn )
		self.assertIn( 'keyword arguments only', self.discovery.errors.errors[0] )

	def test_bare_construct_emits_allocate_instruction_when_no_init( self ) -> None:
		# ClassName(field=value, ...) with no __init__ declared degrades to
		# exactly __allocate__ (SYNTAX.md's __init__-invocation/failure-
		# wrapping path is future work) - and unlike .__allocate__(...), it's
		# public: callable from outside the class entirely, not just its
		# own methods.
		code = '\n'.join([
			'@cstruct',
			'class Foo:',
			'	x: i32',
			'	y: i32',
			'',
			'def main() -> None:',
			'	f: Foo = Foo( x = 1, y = 2 )',
			'	return',
		])
		mod = self._import( code )
		fn = self._lower_main()
		foo_cls = mod.get_local( 'Foo' )
		i32 = self.discovery.get_intrinsics()['i32']
		allocate_instr = next( i for i in fn.instructions if isinstance( i, ir.Allocate ))
		self.assertIs( allocate_instr.cls, foo_cls )
		self.assertEqual( allocate_instr.fields, {
			'x': ir.Const( type = i32, value = 1 ),
			'y': ir.Const( type = i32, value = 2 ),
		})

	def test_bare_construct_with_init_declared_is_not_yet_supported( self ) -> None:
		# a class WITH __init__ falls through to the normal call path -
		# construction via __init__ needs Result-wrapping/refcount-on-
		# failure cleanup that doesn't exist yet, so this must NOT silently
		# degrade to a plain __allocate__ (that would skip __init__ entirely)
		code = '\n'.join([
			'@cstruct',
			'class Foo:',
			'	x: i32',
			'',
			'	def __init__( self ) -> None:',
			'		return',
			'',
			'def main() -> None:',
			'	f: Foo = Foo( x = 1 )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'cannot call', self.discovery.errors.errors[0] )

	def test_bare_construct_missing_field_is_rejected( self ) -> None:
		code = '\n'.join([
			'@cstruct',
			'class Foo:',
			'	x: i32',
			'	y: i32',
			'',
			'def main() -> None:',
			'	f: Foo = Foo( x = 1 )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'missing field', self.discovery.errors.errors[0] )
		self.assertIn( 'y', self.discovery.errors.errors[0] )

	# --- TaggedUnion construction (Foo.Member(value)) ------------------------

	def test_union_member_construct_emits_tag_and_payload_allocate( self ) -> None:
		code = '\n'.join([
			'@union',
			'class Foo:',
			'	Bar: i32',
			'	Baz: usize',
			'',
			'def main() -> None:',
			'	f: Foo = Foo.Bar( 5 )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		foo_cls = self.discovery.modules['__test__'].get_local( 'Foo' )
		allocates = [ i for i in fn.instructions if isinstance( i, ir.Allocate ) ]
		self.assertEqual( len( allocates ), 2 )
		payload_alloc, union_alloc = allocates
		self.assertEqual( payload_alloc.cls.stem, 'Foo$data' )
		self.assertEqual( set( payload_alloc.fields.keys() ), { 'v_Bar' } )
		self.assertIs( union_alloc.cls, foo_cls )
		self.assertEqual( set( union_alloc.fields.keys() ), { 'tag', 'data' } )
		self.assertEqual( union_alloc.fields['tag'], ir.Const( type = self.discovery.get_intrinsics()['u8'], value = 0 ))
		self.assertIs( union_alloc.fields['data'], payload_alloc.dest )

	def test_union_member_construct_tag_is_declaration_order( self ) -> None:
		code = '\n'.join([
			'@union',
			'class Foo:',
			'	Bar: i32',
			'	Baz: usize',
			'',
			'def main() -> None:',
			'	f: Foo = Foo.Baz( 7 )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		union_alloc = next( i for i in fn.instructions if isinstance( i, ir.Allocate ) and i.cls.stem == 'Foo' )
		self.assertEqual( union_alloc.fields['tag'], ir.Const( type = self.discovery.get_intrinsics()['u8'], value = 1 ))

	def test_union_member_construct_wrong_arg_count_rejected( self ) -> None:
		code = '\n'.join([
			'@union',
			'class Foo:',
			'	Bar: i32',
			'',
			'def main() -> None:',
			'	f: Foo = Foo.Bar( 1, 2 )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'exactly one positional argument', self.discovery.errors.errors[0] )

	def test_union_storage_is_memoized_across_constructions( self ) -> None:
		# construction elsewhere in the same function (or a different one)
		# must reference the SAME synthesized tag/data/payload-class objects
		code = '\n'.join([
			'@union',
			'class Foo:',
			'	Bar: i32',
			'',
			'def main() -> None:',
			'	a: Foo = Foo.Bar( 1 )',
			'	b: Foo = Foo.Bar( 2 )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		payload_allocs = [ i for i in fn.instructions if isinstance( i, ir.Allocate ) and i.cls.stem == 'Foo$data' ]
		self.assertEqual( len( payload_allocs ), 2 )
		self.assertIs( payload_allocs[0].cls, payload_allocs[1].cls )

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

	def test_noreturn_function_epilogue_has_no_return_value_var( self ) -> None:
		# NoReturn behaves exactly like None for the epilogue's own return-
		# value machinery - no __return_value stowing, plain Return(None)
		code = '\n'.join([
			'class bool: pass',
			'',
			'def cleanup() -> None:',
			'	pass',
			'',
			'def die() -> NoReturn:',
			'	with defer:',
			'		cleanup()',
		])
		mod = self._import( code )
		bool_cls = mod.get_local( 'bool' )
		cleanup_fn = mod.get_local( 'cleanup' )
		if cleanup_fn.resolve is not None:
			cleanup_fn.resolve()
		die_fn = mod.get_local( 'die' )
		if die_fn.resolve is not None:
			die_fn.resolve()
		noreturn_cls = self.discovery.get_intrinsics()['NoReturn']
		flag0 = Variable( stem = '__defer_flag_0', qualname = '__test__.die.__defer_flag_0', file = Path( '__test__.py' ), line = 7, type = bool_cls )

		fn = self.compiler._lower( die_fn )
		self._assert_ir( fn, [
			ir.FuncStart( name = '__test__.die', params = [], return_type = noreturn_cls ),
			ir.Assign( dest = flag0, src = ir.Const( type = bool_cls, value = False )),
			ir.Assign( dest = flag0, src = ir.Const( type = bool_cls, value = True )),
			# falls off the end of the body (no explicit return) straight
			# into the epilogue - no Jump needed, it's placed right after
			ir.Label( name = '__epilogue__' ),
			ir.JumpIfFalse( cond = flag0, target = '__defer_skip_0__' ),
			ir.Call( dest = None, target = cleanup_fn, args = [], kwargs = {} ),
			ir.Label( name = '__defer_skip_0__' ),
			ir.Return( value = None ),
			ir.FuncEnd( name = '__test__.die' ),
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
