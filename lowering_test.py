# stdlib imports:
import logging
from pathlib import Path
import queue as queue_module
import tempfile
import unittest

# local imports:
from compiler import Compiler, LoweredFunction
import discovery
from discovery import Discovery
from errors import CompileError
import ir
from mpy_types import Variable, Specialization, Function, ClosureType, TaggedUnion
import targets

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

	def test_annassign_volatile_sets_flag_and_strips_type( self ) -> None:
		# Volatile[T] resolves transparently to plain T (discovery.py's
		# visit_Subscript) - the Variable itself carries is_volatile=True,
		# not a wrapper type, so it keeps behaving as an ordinary usize
		# everywhere else (see the Volatile[T] design note in _stmt_AnnAssign)
		code = '\n'.join([
			'def main() -> None:',
			'	i: Volatile[usize] = 0',
			'	return',
		])
		usize = self.discovery.get_intrinsics()['usize']
		none_type = self.discovery.get_none_type()
		i = Variable( stem = 'i', qualname = 'main.i', file = Path( '__test__.py' ), line = 2, type = usize, is_volatile = True )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = i, src = ir.Const( type = usize, value = 0 )),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_annassign_volatile_rejects_rc_type( self ) -> None:
		code = '\n'.join([
			'class Box:',
			'	v: i32 = 0',
			'def main() -> None:',
			'	b: Volatile[Box] = Box()',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any( 'Volatile[...] does not support refcounted types' in e for e in self.discovery.errors.errors ))

	def test_augassign_desugars_to_binop_and_assign( self ) -> None:
		# x += 1 lowers exactly like a hand-written x = x + 1 would - same
		# AddWrap/Assign shape (via i32.__wrapped_add__'s inline-spliced
		# body), honoring the active arithmetic mode
		self.discovery.import_name( 'builtins' )
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
		add_i32_fn = i32.names['__wrapped_add__']
		if add_i32_fn.resolve is not None:
			add_i32_fn.resolve()
		inline_other = Variable(
			stem = '$inline0$other', qualname = f'{add_i32_fn.qualname}$$inline0$other',
			file = add_i32_fn.file, line = add_i32_fn.line, type = i32,
		)
		t0 = ir.Temp( type = i32, id = 0 )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = x, src = ir.Const( type = i32, value = 1 )),
			ir.Assign( dest = inline_other, src = ir.Const( type = i32, value = 2 )),
			ir.DeclareTemp( temp = t0 ),
			ir.AddWrap( dest = t0, left = x, right = inline_other ),
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

	def test_augassign_attribute_target( self ) -> None:
		# f.x += 2 reads f.x exactly once (GetAttr into a fresh temp),
		# computes the BinOp against that temp, then writes back exactly
		# once (SetAttr) - f itself is lowered exactly once, shared by both
		# the read and the write, unlike the x = x + y desugaring a bare
		# Name target uses (which would double-evaluate f here)
		code = '\n'.join([
			'class Foo:',
			'	x: i32',
			'',
			'def main( f: Foo ) -> None:',
			'	with compiler.wrap_arithmetic:',
			'		f.x += 2',
			'	return',
		])
		self.discovery.import_name( 'builtins' )
		mod = self._import( code )
		i32 = self.discovery.get_intrinsics()['i32']
		none_type = self.discovery.get_none_type()
		foo_cls = mod.get_local( 'Foo' )
		if foo_cls.resolve is not None:
			foo_cls.resolve()
		if self.discovery.main.resolve is not None:
			self.discovery.main.resolve()
		f = self.discovery.main.parameters[0]
		add_i32_fn = i32.names['__wrapped_add__']
		if add_i32_fn.resolve is not None:
			add_i32_fn.resolve()
		# the receiver here is t0 (a GetAttr's Temp, not already a bare
		# Variable) so it ALSO needs splicing into its own synthesized local,
		# not just the literal `other` operand - shifting other to $inline1$
		inline_value = Variable(
			stem = '$inline0$value', qualname = f'{add_i32_fn.qualname}$$inline0$value',
			file = add_i32_fn.file, line = add_i32_fn.line, type = i32,
		)
		inline_other = Variable(
			stem = '$inline1$other', qualname = f'{add_i32_fn.qualname}$$inline1$other',
			file = add_i32_fn.file, line = add_i32_fn.line, type = i32,
		)
		t0 = ir.Temp( type = i32, id = 0 )
		t1 = ir.Temp( type = i32, id = 1 )

		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_ir( fn, [
			ir.FuncStart( name = 'main', params = [ f ], return_type = none_type ),
			ir.DeclareTemp( temp = t0 ),
			ir.GetAttr( dest = t0, obj = f, attr = 'x' ),
			ir.Assign( dest = inline_value, src = t0 ),
			ir.Assign( dest = inline_other, src = ir.Const( type = i32, value = 2 )),
			ir.DeclareTemp( temp = t1 ),
			ir.AddWrap( dest = t1, left = inline_value, right = inline_other ),
			ir.SetAttr( obj = f, attr = 'x', value = t1 ),
			ir.DeleteTemp( temp = t1 ),
			ir.DeleteTemp( temp = t0 ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_augassign_attribute_target_object_evaluated_once( self ) -> None:
		# get_obj().x += 1 must call get_obj() exactly once - the real
		# correctness risk _stmt_AugAssign's own Attribute branch exists to
		# avoid (the x += y-as-x = x + y desugaring a bare Name target uses
		# would otherwise double-evaluate the object expression)
		code = '\n'.join([
			'class Foo:',
			'	x: i32',
			'',
			'g: Foo',
			'',
			'def get_obj() -> Foo:',
			'	return g',
			'',
			'def main() -> None:',
			'	with compiler.wrap_arithmetic:',
			'		get_obj().x += 1',
			'	return',
		])
		self.discovery.import_name( 'builtins' )
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( len( calls ), 1 )
		self.assertEqual( calls[0].target.qualname, '__test__.get_obj' )

	def test_augassign_subscript_target_raw_pointer_fallback( self ) -> None:
		# no __getitem__/__setitem__ declared (raw pointers) - falls back to
		# the flat GetItem/SetItem opcodes, index/obj lowered exactly once
		code = '\n'.join([
			'import sys',
			'',
			'@cstruct',
			'class Point:',
			'	x: i32',
			'',
			'def main() -> None:',
			'	p: Ptr[i32] = sys.alloc[i32]( 1 )',
			'	with compiler.wrap_arithmetic:',
			'		p[0] += 1',
			'	sys.free( p )',
			'	return',
		])
		self.discovery.import_name( 'builtins' )
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertIn( 'GetItem', kinds )
		self.assertIn( 'SetItem', kinds )
		self.assertIn( 'AddWrap', kinds )

	def test_augassign_subscript_target_with_getitem_setitem_methods( self ) -> None:
		# a real __getitem__/__setitem__ pair - read via __getitem__, add,
		# write back via __setitem__, index lowered exactly once and shared
		# by both calls
		code = '\n'.join([
			'@cstruct',
			'class Box:',
			'	y: i32',
			'',
			'	def __getitem__( self, i: usize ) -> i32:',
			'		return self.y',
			'',
			'	def __setitem__( self, i: usize, v: i32 ) -> None:',
			'		self.y = v',
			'',
			'def main( b: Box ) -> None:',
			'	with compiler.wrap_arithmetic:',
			'		b[0] += 5',
			'	return',
		])
		self.discovery.import_name( 'builtins' )
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertNotIn( 'GetItem', kinds )
		self.assertNotIn( 'SetItem', kinds )
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( [ c.target.qualname for c in calls ], [ '__test__.Box.__getitem__', '__test__.Box.__setitem__' ] )

	def test_augassign_attribute_target_with_iadd_skips_setattr( self ) -> None:
		# a field whose RC-class type defines a matching __iadd__ - old
		# aliases the SAME heap object the field already points to (a plain
		# GetAttr read), so SetAttr is skipped entirely: only the GetAttr (to
		# fetch old) plus one Call (__iadd__ itself) should appear
		code = '\n'.join([
			'class Counter:',
			'	n: i32',
			'',
			'	def __iadd__( self, v: i32 ) -> None:',
			'		self.n = v',
			'',
			'class Foo:',
			'	x: Counter',
			'',
			'def main( f: Foo, v: i32 ) -> None:',
			'	f.x += v',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertNotIn( 'SetAttr', kinds )
		self.assertIn( 'GetAttr', kinds )
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( [ c.target.qualname for c in calls ], [ '__test__.Counter.__iadd__' ] )

	def test_augassign_subscript_target_with_iadd_skips_setitem( self ) -> None:
		# an RC-class element type defining a matching __iadd__ - old
		# already went through __getitem__'s own incref, so it's a genuine
		# extra owned reference to the SAME heap object the container's slot
		# stores; __setitem__ is skipped entirely
		code = '\n'.join([
			'class Counter:',
			'	n: i32',
			'',
			'	def __iadd__( self, v: i32 ) -> None:',
			'		self.n = v',
			'',
			'@cstruct',
			'class Box:',
			'	y: Counter',
			'',
			'	def __getitem__( self, i: usize ) -> Counter:',
			'		return self.y',
			'',
			'	def __setitem__( self, i: usize, v: Counter ) -> None:',
			'		self.y = v',
			'',
			'def main( b: Box, v: i32 ) -> None:',
			'	b[0] += v',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( [ c.target.qualname for c in calls ], [ '__test__.Box.__getitem__', '__test__.Counter.__iadd__' ] )

	def test_augassign_subscript_target_merged_result_coverage_names_both_errors( self ) -> None:
		# no __iadd__ applies (i32 element - not RC) - the fallback get-
		# >combine->set path's two independently-fallible Results (different
		# error types on __getitem__ vs __setitem__) are checked TOGETHER in
		# one message, not one-at-a-time
		code = '\n'.join([
			'class GetError: pass',
			'class SetError: pass',
			'',
			'@cstruct',
			'class Result[T,E]:',
			'	x: T',
			'',
			'@cstruct',
			'class Box:',
			'	y: i32',
			'',
			'	def __getitem__( self, i: usize ) -> Result[i32,GetError]:',
			'		return Result.__allocate__( x = self.y )',
			'',
			'	def __setitem__( self, i: usize, v: i32 ) -> Result[bool,SetError]:',
			'		self.y = v',
			'		return Result.__allocate__( x = True )',
			'',
			'def main( b: Box ) -> None:',
			'	with compiler.wrap_arithmetic:',
			'		b[0] += 5',
			'	return',
		])
		self.discovery.import_name( 'builtins' )
		self._import( code )
		self._lower_main()
		self.assertTrue( self.discovery.errors.errors )
		err = self.discovery.errors.errors[0]
		self.assertIn( 'GetError', err )
		self.assertIn( 'SetError', err )

	def test_augassign_name_target_with_iadd_dispatches_to_method( self ) -> None:
		# a bare RC-class Name target with a matching __iadd__ - dispatches
		# straight to it, no reassignment of the name at all
		code = '\n'.join([
			'class Counter:',
			'	n: i32',
			'',
			'	def __iadd__( self, v: i32 ) -> None:',
			'		self.n = v',
			'',
			'def main( c: Counter, v: i32 ) -> None:',
			'	c += v',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( [ c.target.qualname for c in calls ], [ '__test__.Counter.__iadd__' ] )

	def test_augassign_name_target_rc_without_iadd_falls_back_to_add_and_reassign( self ) -> None:
		# an RC-class target with no matching __iadd__ still needs the
		# ordinary x = x + y behavior - reproduced manually (not via the
		# synthesized-BinOp delegation a non-RC/undeclared target uses),
		# since `right` is already lowered by the time __iadd__'s absence is
		# discovered and re-lowering node.value would double-evaluate it
		code = '\n'.join([
			'class Vector:',
			'	x: i32',
			'',
			'	def __add__( self, other: Vector ) -> Vector:',
			'		return other',
			'',
			'def main( v: Vector, w: Vector ) -> None:',
			'	v += w',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( [ c.target.qualname for c in calls ], [ '__test__.Vector.__add__' ] )
		assigns = [ i for i in fn.instructions if isinstance( i, ir.Assign ) ]
		self.assertTrue( assigns ) # v is reassigned to the __add__ result

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

	def test_bare_assign_ints_default_to_i32( self ) -> None:
		# integer literals default to i32 when no context type is available
		code = '\n'.join([
			'def main() -> i32:',
			'	x = 1',
			'	return x',
		])
		i32 = self.discovery.get_intrinsics()['i32']
		x = Variable( stem = 'x', qualname = 'main.x', file = Path( '__test__.py' ), line = 2, type = i32 )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = i32 ),
			ir.Assign( dest = x, src = ir.Const( type = i32, value = 1 )),
			ir.Return( value = x ),
			ir.FuncEnd( name = 'main' ),
		])

	# --- del statement -----------------------------------------------------

	def test_del_removes_local_from_scope( self ) -> None:
		code = '\n'.join([
			'class Foo: pass',
			'',
			'def main() -> None:',
			'	f: Foo = Foo()',
			'	del f',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_del_then_reference_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'class Foo: pass',
			'',
			'def takeref( x: Foo ) -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	f: Foo = Foo()',
			'	del f',
			'	takeref( f )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( "'f' is not defined", self.discovery.errors.errors[0] )

	def test_del_on_never_initialized_local_is_a_compile_error( self ) -> None:
		# the "__del__ a variable that's not provably alive" half of the
		# new definite-assignment gate (cfg.py's deleted()) - a bare
		# declaration with no assignment on any path is never live, so
		# del'ing it is exactly as much an error as reading it would be
		code = '\n'.join([
			'class Foo: pass',
			'',
			'def main() -> None:',
			'	f: Foo',
			'	del f',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( "'f' is not initialized on all code branches", self.discovery.errors.errors[0] )

	def test_del_nonexistent_name_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	del nonexistent',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'is not a local variable', self.discovery.errors.errors[0] )

	def test_del_multiple_targets_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	a: i32 = 1',
			'	b: i32 = 2',
			'	del a, b',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'single local variable name', self.discovery.errors.errors[0] )

	def test_del_then_redeclare_is_a_fresh_binding( self ) -> None:
		# x = 'foo'; del x; x = 'bar' is two independent bindings that
		# happen to reuse the name - allowed for now (detecting/flagging
		# this as likely-confusing reuse is documented future work)
		code = '\n'.join([
			'def main() -> None:',
			'	x: i32 = 1',
			'	del x',
			'	x: i32 = 2',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_del_then_redeclare_with_a_genuinely_different_type_is_allowed( self ) -> None:
		# the companion case the test above's own name promises but its
		# body doesn't actually exercise (both sides there are i32) - del
		# fully removes the name from fn.names (see _stmt_Delete's own
		# comment: "Removing it from fn.names is enough on its own to make
		# a later reference fail"), so a later `x = ...` finds no existing
		# declaration at all and takes the FRESH-binding path
		# (_declare_local, inferring straight from the RHS, unconstrained
		# by whatever type the deleted binding happened to have) - not the
		# _stmt_Assign reassignment path that would otherwise enforce the
		# OLD type against the new value (see the match-arm-binding-reuse
		# diagnostic tests above, which fire specifically because THOSE
		# reused names were never del'd first).
		code = '\n'.join([
			'import builtins',
			'def main() -> None:',
			'	x: i32 = 1',
			'	del x',
			'	x: builtins.str = "hello"',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_annotated_redeclaration_without_del_is_a_compile_error_same_type( self ) -> None:
		# `x: int; x: int` (equal types, no del, straight-line - no
		# branching at all) is STILL a redeclaration error: an explicit
		# type annotation is only ever given once per variable, full stop
		# - matching types doesn't exempt it, only `del` does. This is
		# the direct counterpart to the del-then-redeclare test above,
		# which is allowed for exactly the reason this isn't: del ends
		# the old binding's lifetime first, this doesn't.
		code = '\n'.join([
			'def main() -> None:',
			'	x: i32 = 1',
			'	x: i32 = 2',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( len( self.discovery.errors.errors ), 1 )
		self.assertIn( "'x' already has a declared type", str( self.discovery.errors.errors[0] ))

	def test_annotated_redeclaration_without_del_is_a_compile_error_different_type( self ) -> None:
		# the exact scenario that motivated this whole diagnostic: no del,
		# genuinely incompatible types - must be rejected, not silently
		# accepted (which is what used to happen before _stmt_AnnAssign
		# started checking for an existing binding at all - see del_
		# reuse_and_emitter_naming_bug)
		code = '\n'.join([
			'import builtins',
			'def main() -> None:',
			'	x: i32 = 1',
			'	x: builtins.str = "hello"',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( len( self.discovery.errors.errors ), 1 )
		self.assertIn( "'x' already has a declared type", str( self.discovery.errors.errors[0] ))

	def test_inline_as_annotated_local_is_rejected( self ) -> None:
		# 'inline' can't just be silently mangled to `_inline` the way the
		# rest of emitter_c.py's own _C_KEYWORDS list safely can - `_inline`
		# is ITSELF a reserved identifier under MSVC-compatible headers (a
		# legacy `#define _inline __inline` compatibility macro), so
		# mangling only trades one collision for another. Rejected here
		# instead, at the metalpy source line, before it ever reaches the
		# emitter.
		code = '\n'.join([
			'def main() -> None:',
			'	inline: i32 = 1',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( len( self.discovery.errors.errors ), 1 )
		self.assertIn( "'inline' is a reserved identifier", str( self.discovery.errors.errors[0] ))

	def test_inline_as_bare_local_is_rejected( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	inline = 1',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( len( self.discovery.errors.errors ), 1 )
		self.assertIn( "'inline' is a reserved identifier", str( self.discovery.errors.errors[0] ))

	def test_inline_as_for_loop_target_is_rejected( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	for inline in range( 3 ):',
			'		pass',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( len( self.discovery.errors.errors ), 1 )
		self.assertIn( "'inline' is a reserved identifier", str( self.discovery.errors.errors[0] ))

	def test_inline_as_function_parameter_is_rejected( self ) -> None:
		code = '\n'.join([
			'def f( inline: i32 ) -> i32:',
			'	return inline',
			'def main() -> None:',
			'	f( 1 )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( len( self.discovery.errors.errors ), 1 )
		self.assertIn( "'inline' is a reserved identifier", str( self.discovery.errors.errors[0] ))

	def test_inline_as_module_global_is_rejected( self ) -> None:
		code = '\n'.join([
			'inline: i32 = 1',
			'def main() -> None:',
			'	return',
		])
		self._import( code )
		self.assertEqual( len( self.discovery.errors.errors ), 1 )
		self.assertIn( "'inline' is a reserved identifier", str( self.discovery.errors.errors[0] ))

	def test_annotated_redeclaration_across_branches_is_a_compile_error( self ) -> None:
		# the if/elif/else-arm variant of the two tests above - even
		# though the two arms are mutually exclusive at runtime (only one
		# ever actually executes), fn.names is function-flat with no
		# block scoping (see cfg.py's own module docstring), so `x` from
		# the first arm is STILL a live, already-declared binding by the
		# time the second arm's own `x: i32 = ...` is reached. This is
		# exactly the pattern lib/builtins/__File.py's own `creation`
		# local used to follow (and had to be rewritten away from - see
		# this fix's own commit message) - the valid replacement is a
		# bare `x: i32` declared once before the chain, then a plain
		# (un-annotated) `x = ...` per arm, covered by emitter_c_test.py's
		# AnnotatedLocalRedeclaredAcrossBranchesRealCompileTests.
		code = '\n'.join([
			'def main( a: bool ) -> None:',
			'	if a:',
			'		x: i32 = 1',
			'	else:',
			'		x: i32 = 2',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( len( self.discovery.errors.errors ), 1 )
		self.assertIn( "'x' already has a declared type", str( self.discovery.errors.errors[0] ))

	def test_bare_reassignment_across_branches_is_still_allowed( self ) -> None:
		# the control case distinguishing "an explicit type annotation is
		# only given once" from "a variable can only ever be assigned
		# once" - they're NOT the same rule. A bare (un-annotated,
		# INFERRED-type) `x = ...` repeated once per arm of an if/else is
		# an ordinary reassignment to the SAME binding (the type is fixed
		# by whichever assignment reaches it first - see _stmt_Assign's
		# own "reuse existing" branch), ordinary and unaffected by the
		# annotated-redeclaration checks the tests above cover.
		code = '\n'.join([
			'def main( a: bool ) -> None:',
			'	x: i32',
			'	if a:',
			'		x = 1',
			'	else:',
			'		x = 2',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_global_reassignment_from_inside_a_function_still_works( self ) -> None:
		# _existing_local_or_none (shared by _stmt_Assign/_stmt_AnnAssign/
		# _expr_NamedExpr/_bind_loop_target) needed to grow a genuine
		# MODULE-scope fallback once _stmt_AnnAssign started calling it at
		# all - fn.names alone (the CURRENT function's own scope) can't
		# see a module-level global at all, and _stmt_Global is
		# deliberately a no-op (see its own comment) that relies entirely
		# on this fallback existing. A real, confirmed regression along
		# the way: restricting the lookup to fn.names alone made `global
		# x; x = value` look like a fresh LOCAL instead of a reassignment,
		# silently shadowing the real global.
		code = '\n'.join([
			'x: i32 = 1',
			'def bump() -> None:',
			'	global x',
			'	x = x + 1',
			'def main() -> i32:',
			'	bump()',
			'	bump()',
			'	return x',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_local_may_shadow_an_enclosing_class_method_of_the_same_name( self ) -> None:
		# this language has no local-shadows-outer-scope semantics for a
		# MODULE-level name (see _stmt_Global's own comment - assigning an
		# already-visible module name always reassigns it, never shadows)
		# - but a method body's own locals are still free to shadow an
		# unrelated CLASS MEMBER of the same name, the same way any
		# nested Python scope shadows an enclosing one. A real, confirmed
		# regression along the way: str._from_owned_cstr's own local
		# named byte_len (now renamed to text_len, but exercised here
		# under its original name against a minimal repro class) started
		# getting rejected as "not a variable, cannot assign to it" once
		# _stmt_AnnAssign began walking the full enclosing-scope chain -
		# find_name_or_none reached the CLASS's own byte_len() method
		# before ever finding "no local yet" and stopping.
		code = '\n'.join([
			'class Box:',
			'	def byte_len( self ) -> i32:',
			'		return 5',
			'	def describe( self ) -> i32:',
			'		byte_len: i32 = 3',
			'		return byte_len',
			'def main() -> i32:',
			'	b: Box = Box()',
			'	return b.describe()',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_del_between_match_statements_allows_reusing_binding_name_with_incompatible_type( self ) -> None:
		# ties directly to test_match_binding_name_reused_with_incompatible_
		# type_gets_a_clear_diagnostic above: THAT test's whole point is
		# that reusing a match-arm binding name across two unrelated match
		# statements with incompatible types is now a clear compile error.
		# del is the existing, already-available way to resolve it
		# deliberately - del e between the two match statements removes e
		# from fn.names entirely, so the second match's own `case
		# Result.Err(e):` binding takes the fresh-declaration path (like
		# the test above), not the reassignment path the diagnostic fires
		# from - confirms this real, useful escape hatch actually works,
		# not just reasoning about _stmt_Delete's own mechanism in isolation.
		code = '\n'.join([
			'import builtins',
			'def get_a() -> builtins.Result[i32, builtins.OverflowError|builtins.IndexError]:',
			'	return builtins.Result.Err( builtins.OverflowError() )',
			'',
			'def get_b() -> builtins.Result[i32, builtins.IndexError|builtins.KeyError]:',
			'	return builtins.Result.Err( builtins.IndexError() )',
			'',
			'def main() -> i32:',
			'	match get_a():',
			'		case builtins.Result.Err( e ):',
			'			pass',
			'		case builtins.Result.Ok( _ ):',
			'			return 1',
			'	del e',
			'	match get_b():',
			'		case builtins.Result.Err( e ):',
			'			pass',
			'		case builtins.Result.Ok( _ ):',
			'			return 2',
			'	return 0',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

	# --- arithmetic ---------------------------------------------------------

	def test_binop_without_arithmetic_context_is_a_compile_error( self ) -> None:
		# arithmetic defaults to Check mode (Result[T,OverflowError]) - see
		# the Lowering class docstring - and main() returns None, which can't
		# propagate that error, so plain `a + 1` here is a compile error
		# rather than silently falling back to wrapping. `a + 1` dispatches
		# through i32.__add__ (a real dunder, needing real builtins.Result/
		# OverflowError - no more hand-rolled local stand-ins for those)
		code = '\n'.join([
			'def main() -> None:',
			'	a: i32 = 1',
			'	b: i32 = a + 1',
			'	return',
		])
		self.discovery.import_name( 'builtins' )
		self._import( code )
		fn = self._lower_main()
		# no longer eagerly consumed inside _lower_arithmetic_op itself - the
		# raw Result now flows out and is auto-.or_throw()'d by
		# _coerce_or_check_operand's own case-2 hook once `b: i32 = ...`
		# tries to assign it, so the "how to avoid this" message is or_throw()'s
		# own generic alternatives text now, not _FALLIBLE_METHOD_ALTERNATIVES
		self.assertIn( 'requires the enclosing function to return Result', self.discovery.errors.errors[0] )
		self.assertIn( 'try/except', self.discovery.errors.errors[0] )
		# lower_function's per-statement recovery boundary skips just the
		# failing statement - b is never assigned. Unlike the old fallback
		# (which validated Result-coverage BEFORE emitting anything), the
		# dunder path's own check only runs at the final Result-consumption
		# step, after the inline splice has already emitted its own
		# instructions - those leak into the recovered function body (dead,
		# since nothing ever reads t0/inline_other, but present)
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertEqual( kinds, [ 'FuncStart', 'Assign', 'Assign', 'DeclareTemp', 'AddCheck', 'Return', 'FuncEnd' ] )

	def test_binop_wrap_arithmetic_context( self ) -> None:
		# with compiler.wrap_arithmetic: switches Add/Sub/Mult back to the
		# plain Wrap opcodes (via i32.__wrapped_add__/__wrapped_sub__/
		# __wrapped_mul__'s inline-spliced bodies), no Result/OrReturn
		# involved - the with statement itself contributes no instructions
		# of its own
		self.discovery.import_name( 'builtins' )
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
		# the $inlineN$ splice counter is shared across every inline call
		# spliced into THIS function body (main), not reset per callee - the
		# three ops here get inline0/1/2 in source order
		def inline_other( fn_name: str, n: int ) -> Variable:
			fn = i32.names[fn_name]
			if fn.resolve is not None:
				fn.resolve()
			return Variable( stem = f'$inline{n}$other', qualname = f'{fn.qualname}$$inline{n}$other', file = fn.file, line = fn.line, type = i32 )
		add_other = inline_other( '__wrapped_add__', 0 )
		sub_other = inline_other( '__wrapped_sub__', 1 )
		mul_other = inline_other( '__wrapped_mul__', 2 )
		# temp numbering is per-function (not per-statement), so each new
		# statement's temp continues where the last one left off
		t0 = ir.Temp( type = i32, id = 0 )
		t1 = ir.Temp( type = i32, id = 1 )
		t2 = ir.Temp( type = i32, id = 2 )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = a, src = ir.Const( type = i32, value = 1 )),
			ir.Assign( dest = add_other, src = ir.Const( type = i32, value = 1 )),
			ir.DeclareTemp( temp = t0 ),
			ir.AddWrap( dest = t0, left = a, right = add_other ),
			ir.Assign( dest = b, src = t0 ),
			ir.DeleteTemp( temp = t0 ),
			ir.Assign( dest = sub_other, src = ir.Const( type = i32, value = 1 )),
			ir.DeclareTemp( temp = t1 ),
			ir.SubWrap( dest = t1, left = a, right = sub_other ),
			ir.Assign( dest = c, src = t1 ),
			ir.DeleteTemp( temp = t1 ),
			ir.Assign( dest = mul_other, src = ir.Const( type = i32, value = 2 )),
			ir.DeclareTemp( temp = t2 ),
			ir.MulWrap( dest = t2, left = a, right = mul_other ),
			ir.Assign( dest = d, src = t2 ),
			ir.DeleteTemp( temp = t2 ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_binop_check_mode_emits_or_return( self ) -> None:
		# a function OTHER than main (main can never return Result, since it
		# takes no arguments to be called with the error) that returns
		# Result[None,OverflowError] - every Check op is immediately followed
		# by an auto-inserted OrThrow (the general auto-or_throw() rule -
		# see lowering.py's _auto_or_throw - which degrades to exactly
		# or_return()'s own semantics with an empty dispatch table whenever
		# there's no enclosing try, as here), consuming the Result and
		# continuing with the unwrapped i32 value. The dunder
		# path's error type is the REAL builtins.OverflowError (i32.__add__'s
		# own, fixed at __scalar_dunders.py's own import time) - checked()'s
		# own return annotation must reference that SAME class for
		# _require_result_return's coverage check to pass, so this imports
		# the real builtins.Result/OverflowError rather than hand-rolling
		# local stand-ins the way pre-dunder tests used to
		self.discovery.import_name( 'builtins' )
		code = '\n'.join([
			'from builtins import Result, OverflowError',
			'',
			'def checked() -> Result[None,OverflowError]:',
			'	a: i32 = 1',
			'	b: i32 = a + 1',
			'	c: i32 = a - 1',
			'	d: i32 = a * 2',
			'	return Result.Ok( None )',
		])
		mod = self._import( code )
		i32 = self.discovery.get_intrinsics()['i32']

		checked_fn = mod.get_local( 'checked' )
		if checked_fn.resolve is not None:
			checked_fn.resolve()
		builtins_mod = self.discovery.modules['builtins']
		overflow_cls = builtins_mod.get_local( 'OverflowError' )
		result_cls = builtins_mod.get_local( 'Result' )
		if result_cls.resolve is not None:
			result_cls.resolve()
		result_i32_overflow = self.discovery._get_or_create_specialization( result_cls, [ i32, overflow_cls ] )
		none_type = self.discovery.get_none_type()
		# union_storage.get(), not a bare get_local('Ok') - Result.Ok/Err
		# start out registered as plain field-annotation Variables (the
		# @union sugar's own source shape, `Ok: T = 1`); the REAL
		# constructor Function only exists once UnionStorage synthesizes
		# it, which overwrites names['Ok'] as a side effect - the same
		# thing _coerce_into_union (lowering.py) always does before its
		# own get_local_or_raise('Ok') call. Skipping this and reading
		# get_local('Ok') directly returns the stale Variable instead,
		# confirmed via a real repro (AttributeError: 'Variable' object
		# has no attribute 'type_params', deep in monomorphize.py).
		self.compiler.type_resolver.union_storage.get( result_cls )
		ok_fn = result_cls.get_local( 'Ok' )
		if ok_fn.resolve is not None:
			ok_fn.resolve()
		monomorphized_ok = self.discovery._get_or_create_specialization( ok_fn, [ none_type, overflow_cls ] )
		result_none_overflow = self.discovery._get_or_create_specialization( result_cls, [ none_type, overflow_cls ] )

		a = Variable( stem = 'a', qualname = '__test__.checked.a', file = Path( '__test__.py' ), line = 4, type = i32 )
		b = Variable( stem = 'b', qualname = '__test__.checked.b', file = Path( '__test__.py' ), line = 5, type = i32 )
		c = Variable( stem = 'c', qualname = '__test__.checked.c', file = Path( '__test__.py' ), line = 6, type = i32 )
		d = Variable( stem = 'd', qualname = '__test__.checked.d', file = Path( '__test__.py' ), line = 7, type = i32 )

		# each op's literal `other` operand is spliced into its own
		# synthesized local - counter shared across the whole function body
		def inline_other( fn_name: str, n: int ) -> Variable:
			fn = i32.names[fn_name]
			if fn.resolve is not None:
				fn.resolve()
			return Variable( stem = f'$inline{n}$other', qualname = f'{fn.qualname}$$inline{n}$other', file = fn.file, line = fn.line, type = i32 )
		add_other = inline_other( '__add__', 0 )
		sub_other = inline_other( '__sub__', 1 )
		mul_other = inline_other( '__mul__', 2 )

		t0 = ir.Temp( type = result_i32_overflow, id = 0 ) # AddCheck's Result
		t1 = ir.Temp( type = i32, id = 1 )                 # unwrapped via OrReturn
		t2 = ir.Temp( type = result_i32_overflow, id = 2 ) # SubCheck's Result
		t3 = ir.Temp( type = i32, id = 3 )
		t4 = ir.Temp( type = result_i32_overflow, id = 4 ) # MulCheck's Result
		t5 = ir.Temp( type = i32, id = 5 )
		t6 = ir.Temp( type = result_none_overflow, id = 6 ) # trailing return Result.Ok( None )'s Call

		fn = self.compiler._lower( checked_fn )
		self._assert_ir( fn, [
			ir.FuncStart( name = '__test__.checked', params = [], return_type = checked_fn.return_type ),
			ir.Assign( dest = a, src = ir.Const( type = i32, value = 1 )),
			ir.Assign( dest = add_other, src = ir.Const( type = i32, value = 1 )),
			ir.DeclareTemp( temp = t0 ),
			ir.AddCheck( dest = t0, left = a, right = add_other ),
			ir.DeclareTemp( temp = t1 ),
			ir.OrThrow( dest = t1, value = t0, dispatch = [] ),
			ir.Assign( dest = b, src = t1 ),
			ir.DeleteTemp( temp = t1 ),
			ir.DeleteTemp( temp = t0 ),
			ir.Assign( dest = sub_other, src = ir.Const( type = i32, value = 1 )),
			ir.DeclareTemp( temp = t2 ),
			ir.SubCheck( dest = t2, left = a, right = sub_other ),
			ir.DeclareTemp( temp = t3 ),
			ir.OrThrow( dest = t3, value = t2, dispatch = [] ),
			ir.Assign( dest = c, src = t3 ),
			ir.DeleteTemp( temp = t3 ),
			ir.DeleteTemp( temp = t2 ),
			ir.Assign( dest = mul_other, src = ir.Const( type = i32, value = 2 )),
			ir.DeclareTemp( temp = t4 ),
			ir.MulCheck( dest = t4, left = a, right = mul_other ),
			ir.DeclareTemp( temp = t5 ),
			ir.OrThrow( dest = t5, value = t4, dispatch = [] ),
			ir.Assign( dest = d, src = t5 ),
			ir.DeleteTemp( temp = t5 ),
			ir.DeleteTemp( temp = t4 ),
			ir.DeclareTemp( temp = t6 ),
			ir.Call( dest = t6, target = self.compiler.lowering._monomorphized_function( monomorphized_ok ), receiver = None, args = [ ir.Const( type = none_type, value = None ) ], kwargs = {} ),
			ir.DeleteTemp( temp = t6 ),
			ir.Return( value = t6 ),
			ir.FuncEnd( name = '__test__.checked' ),
		])

	def test_or_return_call_expands_to_or_return_ir_at_call_site( self ) -> None:
		# <result_expr>.or_return() is recognized at the call site purely by
		# AST shape and expanded directly to OrReturn - it has no declared
		# body at all (a user-written `def or_return(...)` is a discovery-
		# time compile error, see discovery.py's _parse_function), since a
		# real one would need to return from ITS CALLER, not itself (see
		# _lower_or_return's own comment)
		code = '\n'.join([
			'class MyError: pass',
			'',
			'@cstruct',
			'class Result[T,E]:',
			'	x: T',
			'',
			'def get_result() -> Result[i32,MyError]:',
			'	return Result( x = 0 )',
			'',
			'def foo() -> Result[i32,MyError]:',
			'	v: i32 = get_result().or_return()',
			'	return Result( x = v )',
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
		v = Variable( stem = 'v', qualname = '__test__.foo.v', file = Path( '__test__.py' ), line = 11, type = i32 )
		t1 = ir.Temp( type = i32, id = 1 )                # unwrapped via OrReturn

		fn = self.compiler._lower( foo_fn )
		get_result_fn = mod.get_local( 'get_result' )
		# get_result_fn's return type is read AFTER _lower (like foo_fn's
		# return_type below) since resolving foo_fn's call to get_result()
		# eagerly monomorphizes get_result_fn's declared return type from a
		# Specialization to the real RCClass/CStruct - constructing our own
		# Specialization via _get_or_create_specialization here would give a
		# distinct (if structurally equal) object, not what the real temp
		# in the lowered IR now carries
		t0 = ir.Temp( type = get_result_fn.return_type, id = 0 ) # get_result()'s Result
		t2 = ir.Temp( type = foo_fn.return_type, id = 2 ) # trailing return Result( x = v )'s Allocate
		self._assert_ir( fn, [
			ir.FuncStart( name = '__test__.foo', params = [], return_type = foo_fn.return_type ),
			ir.DeclareTemp( temp = t0 ),
			ir.Call( dest = t0, target = get_result_fn, receiver = None, args = [], kwargs = {} ),
			ir.DeclareTemp( temp = t1 ),
			ir.OrReturn( dest = t1, value = t0 ),
			ir.Assign( dest = v, src = t1 ),
			ir.DeleteTemp( temp = t1 ),
			ir.DeleteTemp( temp = t0 ),
			ir.DeclareTemp( temp = t2 ),
			ir.Allocate( dest = t2, cls = result_cls, fields = { 'x': v } ),
			ir.DeleteTemp( temp = t2 ),
			ir.Return( value = t2 ),
			ir.FuncEnd( name = '__test__.foo' ),
		])
		self.assertFalse( any( isinstance( i, ir.Call ) and getattr( i.target, 'stem', None ) == 'or_return' for i in fn.instructions ))

	def test_or_return_on_bare_union_result_with_no_declared_method_still_lowers( self ) -> None:
		# regression: a bare @union Result[T,E] with only Ok/Err members and
		# NO explicit or_return method (matching lib/builtins's own real
		# Result post-fix - see its own comment) used to fail to resolve
		# .or_return() at all ("'or_return' is not callable on ..."), since
		# the old dispatch required first finding a real declared method via
		# ordinary attribute lookup. or_return() never needed one - it's
		# recognized purely by AST shape plus the receiver's own type
		code = '\n'.join([
			'class MyError: pass',
			'',
			'@union',
			'class Result[T,E]:',
			'	Ok: T',
			'	Err: E',
			'',
			'def risky() -> Result[i32,MyError]:',
			'	return Result.Ok( 1 )',
			'',
			'def bad() -> Result[i32,MyError]:',
			'	tmp: i32 = risky().or_return()',
			'	return Result.Ok( tmp )',
		])
		mod = self._import( code )
		bad_fn = mod.get_local( 'bad' )
		if bad_fn.resolve is not None:
			bad_fn.resolve()
		fn = self.compiler._lower( bad_fn )
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertTrue( any( isinstance( i, ir.OrReturn ) for i in fn.instructions ))
		self.assertFalse( any( isinstance( i, ir.Call ) and getattr( i.target, 'stem', None ) == 'or_return' for i in fn.instructions ))

	def test_or_return_outside_result_returning_function_is_rejected( self ) -> None:
		code = '\n'.join([
			'class MyError: pass',
			'',
			'@cstruct',
			'class Result[T,E]:',
			'	x: T',
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

		t0 = ir.Temp( type = myerror_cls, id = 0 ) # MyError()'s Allocate - correctly typed as the concrete MyError now that generic-method monomorphization resolves Err's own `e: E` param, not the bare TypeVar
		t1 = ir.Temp( type = result_i32_myerror, id = 1 )  # Result.Err(...)'s Call
		monomorphized_err = self.discovery._get_or_create_specialization( err_fn, [ i32, myerror_cls ] )

		fn = self.compiler._lower( foo_fn )
		self._assert_ir( fn, [
			ir.FuncStart( name = '__test__.foo', params = [], return_type = foo_fn.return_type ),
			ir.DeclareTemp( temp = t0 ),
			ir.Allocate( dest = t0, cls = myerror_cls, fields = {} ),
			ir.DeclareTemp( temp = t1 ),
			ir.Call( dest = t1, target = self.compiler.lowering._monomorphized_function( monomorphized_err ), receiver = None, args = [ t0 ], kwargs = {} ),
			ir.DeleteTemp( temp = t1 ),
			ir.Decref( value = t0 ), # t0 is genuinely RCClass-typed now, so its cleanup correctly decrefs it - previously invisible to cfg.py while it was mistyped as the bare TypeVar
			ir.DeleteTemp( temp = t0 ),
			ir.Return( value = t1 ),
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
		# just the *Saturate opcodes (via i32.__saturated_add__/_sub__/_mul__)
		# instead - no Result/OrReturn involved either, so this works fine
		# inside main() too
		self.discovery.import_name( 'builtins' )
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
		def inline_other( fn_name: str, n: int ) -> Variable:
			fn = i32.names[fn_name]
			if fn.resolve is not None:
				fn.resolve()
			return Variable( stem = f'$inline{n}$other', qualname = f'{fn.qualname}$$inline{n}$other', file = fn.file, line = fn.line, type = i32 )
		add_other = inline_other( '__saturated_add__', 0 )
		sub_other = inline_other( '__saturated_sub__', 1 )
		mul_other = inline_other( '__saturated_mul__', 2 )
		t0 = ir.Temp( type = i32, id = 0 )
		t1 = ir.Temp( type = i32, id = 1 )
		t2 = ir.Temp( type = i32, id = 2 )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = a, src = ir.Const( type = i32, value = 1 )),
			ir.Assign( dest = add_other, src = ir.Const( type = i32, value = 1 )),
			ir.DeclareTemp( temp = t0 ),
			ir.AddSaturate( dest = t0, left = a, right = add_other ),
			ir.Assign( dest = b, src = t0 ),
			ir.DeleteTemp( temp = t0 ),
			ir.Assign( dest = sub_other, src = ir.Const( type = i32, value = 1 )),
			ir.DeclareTemp( temp = t1 ),
			ir.SubSaturate( dest = t1, left = a, right = sub_other ),
			ir.Assign( dest = c, src = t1 ),
			ir.DeleteTemp( temp = t1 ),
			ir.Assign( dest = mul_other, src = ir.Const( type = i32, value = 2 )),
			ir.DeclareTemp( temp = t2 ),
			ir.MulSaturate( dest = t2, left = a, right = mul_other ),
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
		self.discovery.import_name( 'builtins' )
		code = '\n'.join([
			'def main() -> None:',
			'	a: i32 = 1',
			"	with compiler.panic_arithmetic( 'bad arithmetic' ):",
			'		b: i32 = a + 1',
			'	return',
		])
		mod = self._import( code )
		i32 = self.discovery.get_intrinsics()['i32']
		none_type = self.discovery.get_none_type()
		builtins_mod = self.discovery.modules['builtins']
		str_cls = builtins_mod.get_local( 'str' )
		overflow_cls = builtins_mod.get_local( 'OverflowError' )
		result_cls = builtins_mod.get_local( 'Result' )
		if result_cls.resolve is not None:
			result_cls.resolve()
		result_i32_overflow = self.discovery._get_or_create_specialization( result_cls, [ i32, overflow_cls ] )
		# panic_arithmetic's Unwrap calls the REAL sys.panic - not an
		# emitter-invented hook (see ir.Unwrap.panic / Lowering._resolve_sys_function)
		panic_fn = self.discovery.import_name( 'sys' ).get_local( 'panic' )

		a = Variable( stem = 'a', qualname = 'main.a', file = Path( '__test__.py' ), line = 2, type = i32 )
		b = Variable( stem = 'b', qualname = 'main.b', file = Path( '__test__.py' ), line = 4, type = i32 )
		t0 = ir.Temp( type = result_i32_overflow, id = 0 ) # AddCheck's Result
		t1 = ir.Temp( type = i32, id = 1 )                 # unwrapped via Unwrap

		# `a + 1` dispatches through i32.__add__ (a real, @inline dunder -
		# see lib/builtins/__scalar_dunders.py), NOT a bare AddCheck directly
		# against the literal: the literal `1` isn't already a Variable, so
		# _lower_inline_call's splice synthesizes a fresh local to bind the
		# dunder's own `other` parameter to (same "only a genuinely computed
		# operand needs the synthesized-local fallback" rule that applies to
		# every @inline call, not special to this dunder) - looked up
		# dynamically here (rather than hardcoding __scalar_dunders.py's own
		# file/line) so this test doesn't break if that file moves/changes
		add_i32_fn = i32.names['__add__']
		if add_i32_fn.resolve is not None:
			add_i32_fn.resolve()
		inline_other = Variable(
			stem = '$inline0$other', qualname = f'{add_i32_fn.qualname}$$inline0$other',
			file = add_i32_fn.file, line = add_i32_fn.line, type = i32,
		)

		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_ir( fn, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = a, src = ir.Const( type = i32, value = 1 )),
			ir.Assign( dest = inline_other, src = ir.Const( type = i32, value = 1 )),
			ir.DeclareTemp( temp = t0 ),
			ir.AddCheck( dest = t0, left = a, right = inline_other ),
			ir.DeclareTemp( temp = t1 ),
			ir.Unwrap( dest = t1, value = t0, errmsg = ir.Const( type = str_cls, value = 'bad arithmetic' ), panic = panic_fn ),
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
		self.discovery.import_name( 'builtins' )
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
		add_i32_fn = i32.names['__wrapped_add__']
		if add_i32_fn.resolve is not None:
			add_i32_fn.resolve()
		# the receiver here is the literal `1` (not already a Variable), so
		# IT is what gets spliced into a synthesized local - `a` (already a
		# Variable) passes straight through as `other`
		inline_value = Variable(
			stem = '$inline0$value', qualname = f'{add_i32_fn.qualname}$$inline0$value',
			file = add_i32_fn.file, line = add_i32_fn.line, type = i32,
		)
		t0 = ir.Temp( type = i32, id = 0 )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = a, src = ir.Const( type = i32, value = 1 )),
			ir.Assign( dest = inline_value, src = ir.Const( type = i32, value = 1 )),
			ir.DeclareTemp( temp = t0 ),
			ir.AddWrap( dest = t0, left = inline_value, right = a ),
			ir.Assign( dest = b, src = t0 ),
			ir.DeleteTemp( temp = t0 ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_binop_bitwise_ops_are_unconditional( self ) -> None:
		# no overflow concept for &/|/^/>> - always a single, mode-independent
		# dunder (i32.__and__/__or__/__xor__/__rshift__ - no __wrapped_*__/
		# __saturated_*__ variants), works fine in main() with no wrap/check/
		# saturate context at all, unlike +/-/*
		self.discovery.import_name( 'builtins' )
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
		def inline_other( fn_name: str, n: int ) -> Variable:
			fn = i32.names[fn_name]
			if fn.resolve is not None:
				fn.resolve()
			return Variable( stem = f'$inline{n}$other', qualname = f'{fn.qualname}$$inline{n}$other', file = fn.file, line = fn.line, type = i32 )
		and_other = inline_other( '__and__', 0 )
		or_other = inline_other( '__or__', 1 )
		xor_other = inline_other( '__xor__', 2 )
		rshift_other = inline_other( '__rshift__', 3 )
		t0 = ir.Temp( type = i32, id = 0 )
		t1 = ir.Temp( type = i32, id = 1 )
		t2 = ir.Temp( type = i32, id = 2 )
		t3 = ir.Temp( type = i32, id = 3 )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = a, src = ir.Const( type = i32, value = 6 )),
			ir.Assign( dest = and_other, src = ir.Const( type = i32, value = 3 )),
			ir.DeclareTemp( temp = t0 ),
			ir.BitAnd( dest = t0, left = a, right = and_other ),
			ir.Assign( dest = b, src = t0 ),
			ir.DeleteTemp( temp = t0 ),
			ir.Assign( dest = or_other, src = ir.Const( type = i32, value = 3 )),
			ir.DeclareTemp( temp = t1 ),
			ir.BitOr( dest = t1, left = a, right = or_other ),
			ir.Assign( dest = c, src = t1 ),
			ir.DeleteTemp( temp = t1 ),
			ir.Assign( dest = xor_other, src = ir.Const( type = i32, value = 3 )),
			ir.DeclareTemp( temp = t2 ),
			ir.BitXor( dest = t2, left = a, right = xor_other ),
			ir.Assign( dest = d, src = t2 ),
			ir.DeleteTemp( temp = t2 ),
			ir.Assign( dest = rshift_other, src = ir.Const( type = i32, value = 1 )),
			ir.DeclareTemp( temp = t3 ),
			ir.Shr( dest = t3, left = a, right = rshift_other ),
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
		self.discovery.import_name( 'builtins' )
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
		shl_i32_fn = i32.names['__wrapped_lshift__']
		if shl_i32_fn.resolve is not None:
			shl_i32_fn.resolve()
		inline_other = Variable(
			stem = '$inline0$other', qualname = f'{shl_i32_fn.qualname}$$inline0$other',
			file = shl_i32_fn.file, line = shl_i32_fn.line, type = i32,
		)
		t0 = ir.Temp( type = i32, id = 0 )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = a, src = ir.Const( type = i32, value = 1 )),
			ir.Assign( dest = inline_other, src = ir.Const( type = i32, value = 2 )),
			ir.DeclareTemp( temp = t0 ),
			ir.ShlWrap( dest = t0, left = a, right = inline_other ),
			ir.Assign( dest = b, src = t0 ),
			ir.DeleteTemp( temp = t0 ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_binop_floordiv_and_mod_check_mode_emits_or_return( self ) -> None:
		# mirrors test_binop_check_mode_emits_or_return, but for division:
		# SIGNED checked division can raise EITHER ZeroDivisionError (divisor 0)
		# OR OverflowError (INT_MIN/-1), so its Result error type is the union
		# ZeroDivisionError|OverflowError - the enclosing function must return a
		# Result whose error covers both. `//`/`%` dispatch through i32's real
		# __floordiv__/__mod__ dunders now, needing the REAL builtins Result/
		# ZeroDivisionError/OverflowError (see test_binop_check_mode_emits_or_
		# return's own comment on why a hand-rolled local stand-in no longer works)
		self.discovery.import_name( 'builtins' )
		code = '\n'.join([
			'from builtins import Result, ZeroDivisionError, OverflowError',
			'',
			'def checked() -> Result[None,ZeroDivisionError | OverflowError]:',
			'	a: i32 = 10',
			'	b: i32 = a // 3',
			'	c: i32 = a % 3',
			'	return Result.Ok( None )',
		])
		mod = self._import( code )
		i32 = self.discovery.get_intrinsics()['i32']

		checked_fn = mod.get_local( 'checked' )
		if checked_fn.resolve is not None:
			checked_fn.resolve()
		builtins_mod = self.discovery.modules['builtins']
		zerodiv_cls = builtins_mod.get_local( 'ZeroDivisionError' )
		overflow_cls = builtins_mod.get_local( 'OverflowError' )
		result_cls = builtins_mod.get_local( 'Result' )
		if result_cls.resolve is not None:
			result_cls.resolve()
		error_union = self.discovery._get_or_create_union( [ zerodiv_cls, overflow_cls ] )
		result_i32_err = self.discovery._get_or_create_specialization( result_cls, [ i32, error_union ] )
		none_type = self.discovery.get_none_type()
		self.compiler.type_resolver.union_storage.get( result_cls ) # see test_binop_check_mode_emits_or_return's own comment on why this must run before get_local('Ok')
		ok_fn = result_cls.get_local( 'Ok' )
		if ok_fn.resolve is not None:
			ok_fn.resolve()
		monomorphized_ok = self.discovery._get_or_create_specialization( ok_fn, [ none_type, error_union ] )
		result_none_err = self.discovery._get_or_create_specialization( result_cls, [ none_type, error_union ] )

		a = Variable( stem = 'a', qualname = '__test__.checked.a', file = Path( '__test__.py' ), line = 4, type = i32 )
		b = Variable( stem = 'b', qualname = '__test__.checked.b', file = Path( '__test__.py' ), line = 5, type = i32 )
		c = Variable( stem = 'c', qualname = '__test__.checked.c', file = Path( '__test__.py' ), line = 6, type = i32 )

		def inline_other( fn_name: str, n: int ) -> Variable:
			fn = i32.names[fn_name]
			if fn.resolve is not None:
				fn.resolve()
			return Variable( stem = f'$inline{n}$other', qualname = f'{fn.qualname}$$inline{n}$other', file = fn.file, line = fn.line, type = i32 )
		floordiv_other = inline_other( '__floordiv__', 0 )
		mod_other = inline_other( '__mod__', 1 )

		t0 = ir.Temp( type = result_i32_err, id = 0 ) # Div's Result
		t1 = ir.Temp( type = i32, id = 1 )            # unwrapped via OrReturn
		t2 = ir.Temp( type = result_i32_err, id = 2 ) # Mod's Result
		t3 = ir.Temp( type = i32, id = 3 )
		t4 = ir.Temp( type = result_none_err, id = 4 ) # trailing return Result.Ok( None )'s Call

		fn = self.compiler._lower( checked_fn )
		self._assert_ir( fn, [
			ir.FuncStart( name = '__test__.checked', params = [], return_type = checked_fn.return_type ),
			ir.Assign( dest = a, src = ir.Const( type = i32, value = 10 )),
			ir.Assign( dest = floordiv_other, src = ir.Const( type = i32, value = 3 )),
			ir.DeclareTemp( temp = t0 ),
			ir.Div( dest = t0, left = a, right = floordiv_other ),
			ir.DeclareTemp( temp = t1 ),
			ir.OrThrow( dest = t1, value = t0, dispatch = [] ),
			ir.Assign( dest = b, src = t1 ),
			ir.DeleteTemp( temp = t1 ),
			ir.DeleteTemp( temp = t0 ),
			ir.Assign( dest = mod_other, src = ir.Const( type = i32, value = 3 )),
			ir.DeclareTemp( temp = t2 ),
			ir.Mod( dest = t2, left = a, right = mod_other ),
			ir.DeclareTemp( temp = t3 ),
			ir.OrThrow( dest = t3, value = t2, dispatch = [] ),
			ir.Assign( dest = c, src = t3 ),
			ir.DeleteTemp( temp = t3 ),
			ir.DeleteTemp( temp = t2 ),
			ir.DeclareTemp( temp = t4 ),
			ir.Call( dest = t4, target = self.compiler.lowering._monomorphized_function( monomorphized_ok ), receiver = None, args = [ ir.Const( type = none_type, value = None ) ], kwargs = {} ),
			ir.DeleteTemp( temp = t4 ),
			ir.Return( value = t4 ),
			ir.FuncEnd( name = '__test__.checked' ),
		])

	def test_binop_floordiv_without_zerodivision_result_is_a_compile_error( self ) -> None:
		self.discovery.import_name( 'builtins' )
		code = '\n'.join([
			'def main() -> None:',
			'	a: i32 = 1',
			'	b: i32 = a // 1',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		# signed `//` in the default checked mode can raise ZeroDivisionError OR
		# OverflowError (INT_MIN/-1), so the required error type is their union
		self.assertIn( 'Result[_,OverflowError | ZeroDivisionError]', self.discovery.errors.errors[0] )
		self.assertIn( 'try/except', self.discovery.errors.errors[0] )
		# see test_binop_without_arithmetic_context_is_a_compile_error's own
		# comment - the raw Div Result flows out unconsumed and only gets
		# auto-.or_throw()'d (and its coverage checked) once `b: i32 = ...`
		# tries to assign it - the inline splice has already emitted its
		# own (dead, unread) instructions by then
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertEqual( kinds, [ 'FuncStart', 'Assign', 'Assign', 'DeclareTemp', 'Div', 'Return', 'FuncEnd' ] )

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
		self.discovery.import_name( 'builtins' )
		code = '\n'.join([
			'from builtins import Result, ZeroDivisionError',
			'',
			'def checked() -> Result[None,ZeroDivisionError]:',
			'	a: i32 = 10',
			'	with compiler.wrap_arithmetic:',
			'		b: i32 = a // 3',
			'	return Result.Ok( None )',
		])
		mod = self._import( code )
		i32 = self.discovery.get_intrinsics()['i32']
		checked_fn = mod.get_local( 'checked' )
		if checked_fn.resolve is not None:
			checked_fn.resolve()

		fn = self.compiler._lower( checked_fn )
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertIn( 'DivWrap', kinds ) # wrap mode uses DivWrap (INT_MIN/-1 wraps inline); still zero-checked -> Result[_,ZeroDivisionError]
		self.assertIn( 'OrThrow', kinds ) # auto-inserted (no enclosing try here) - degrades to exactly or_return()'s own semantics
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
					'def main() -> None:', # -> None, not Result[_,ZeroDivisionError]
					'	a: i32 = 1',
					f'	with {context}:',
					'		b: i32 = a // 1',
					'	return',
				])
				disco = Discovery( import_builtins = False )
				disco.import_name( 'builtins' )
				comp = Compiler( disco )
				comp.import_code( code, filename = Path( '__test__.py' ))
				fn = comp._lower( disco.main )
				self.assertIn( 'Result[_,ZeroDivisionError]', disco.errors.errors[0] )
				self.assertIn( 'try/except', disco.errors.errors[0] )
				# see test_binop_without_arithmetic_context_is_a_compile_error's
				# own comment - partial (dead) inline-splice instructions leak
				# into the recovered function body
				kinds = [ type( instr ).__name__ for instr in fn.instructions ]
				self.assertEqual( kinds, [ 'FuncStart', 'Assign', 'Assign', 'DeclareTemp', 'DivWrap' if context == 'compiler.wrap_arithmetic' else 'DivSaturate', 'Return', 'FuncEnd' ] )

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

	def test_binop_with_unconsumed_result_operand_is_rejected( self ) -> None:
		# Result[T,E] is itself a @union - without this guard, an unconsumed
		# b[i] (Result[i32,MyError]) would silently decompose into per-leaf
		# (T, E) arithmetic instead of being auto-.or_throw()'d - see
		# _reject_unconsumed_result_operand/_auto_or_throw. foo() doesn't
		# return Result[_,MyError], so the auto-inserted or_throw() itself
		# fails to compile (nowhere for the error to propagate to) - the
		# SAME observable outcome (a compile error) the old hard rejection
		# produced, just via the general auto-or_throw() rule now instead
		# of a bespoke "consume it first" message
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
			'def foo( b: Box, i: usize ) -> i32:',
			'	v: i32 = b[i] + 1',
			'	return v',
		])
		self._import( code )
		foo_fn = self.discovery.modules['__test__'].get_local( 'foo' )
		if foo_fn.resolve is not None:
			foo_fn.resolve()
		self.compiler._lower( foo_fn )
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'requires the enclosing function to return Result', self.discovery.errors.errors[0] )

	def test_eq_with_unconsumed_result_operand_is_rejected( self ) -> None:
		# same guard, ==/!= path (_lower_eq_or_ne) - a fallible comparison's
		# own Result is left unconsumed on purpose (see its docstring), but
		# an unrelated unconsumed Result flowing INTO a comparison operand
		# still gets auto-.or_throw()'d instead of silently decomposing as
		# a union - see test_binop_with_unconsumed_result_operand_is_
		# rejected's own identical comment
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
			'def foo( b: Box, i: usize ) -> bool:',
			'	return b[i] == b[i]',
		])
		self._import( code )
		foo_fn = self.discovery.modules['__test__'].get_local( 'foo' )
		if foo_fn.resolve is not None:
			foo_fn.resolve()
		self.compiler._lower( foo_fn )
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'requires the enclosing function to return Result', self.discovery.errors.errors[0] )

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
			'	return Result()',
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
		none_type = self.discovery.get_none_type()
		result_none_overflow = self.discovery._get_or_create_specialization( result_cls, [ none_type, overflow_cls ] )

		a = Variable( stem = 'a', qualname = '__test__.checked.a', file = Path( '__test__.py' ), line = 8, type = i32 )
		b = Variable( stem = 'b', qualname = '__test__.checked.b', file = Path( '__test__.py' ), line = 9, type = i32 )

		t0 = ir.Temp( type = result_i32_overflow, id = 0 ) # NegCheck's Result
		t1 = ir.Temp( type = i32, id = 1 )                 # unwrapped via OrReturn
		t2 = ir.Temp( type = result_none_overflow, id = 2 ) # trailing return Result()'s Allocate

		fn = self.compiler._lower( checked_fn )
		self._assert_ir( fn, [
			ir.FuncStart( name = '__test__.checked', params = [], return_type = checked_fn.return_type ),
			ir.Assign( dest = a, src = ir.Const( type = i32, value = 1 )),
			ir.DeclareTemp( temp = t0 ),
			ir.NegCheck( dest = t0, operand = a ),
			ir.DeclareTemp( temp = t1 ),
			ir.OrThrow( dest = t1, value = t0, dispatch = [] ),
			ir.Assign( dest = b, src = t1 ),
			ir.DeleteTemp( temp = t1 ),
			ir.DeleteTemp( temp = t0 ),
			ir.DeclareTemp( temp = t2 ),
			ir.Allocate( dest = t2, cls = result_cls, fields = {} ),
			ir.DeleteTemp( temp = t2 ),
			ir.Return( value = t2 ),
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
		# no longer eagerly consumed inside _lower_arithmetic_op itself (see
		# test_binop_without_arithmetic_context_is_a_compile_error's own
		# identical comment) - the opcode-specific wrap_arithmetic/
		# panic_arithmetic alternatives text only existed on that OLD eager
		# path; the general auto-or_throw() rule that now catches this uses
		# its own generic try/except-or-consume-explicitly message instead
		self.assertIn( 'requires the enclosing function to return Result', self.discovery.errors.errors[0] )
		self.assertIn( 'try/except', self.discovery.errors.errors[0] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertEqual( kinds, [ 'FuncStart', 'Assign', 'DeclareTemp', 'NegCheck', 'Return', 'FuncEnd' ] )

	def test_unaryop_not_emits_not_instruction( self ) -> None:
		# needs builtins for the intrinsic `bool` type
		disco = Discovery( import_builtins = True )
		comp = Compiler( disco )
		code = '\n'.join([
			'def main() -> None:',
			'	a: bool = True',
			'	b: bool = not a',
			'	return',
		])
		comp.import_code( code, filename = Path( '__test__.py' ))
		fn = comp._lower( disco.main )
		self.assertEqual( disco.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertIn( 'Not', kinds )

	# --- comparisons ---------------------------------------------------------

	def test_compare_eq_emits_cmp( self ) -> None:
		# a == 1 dispatches through i32.__eq__ (a real, @inline dunder - see
		# lib/builtins/__scalar_dunders.py's scalar_eq[T]), NOT a bare Cmp
		# directly against the literal - same "receiver/literal-arg splice"
		# shape test_binop_wrap_arithmetic_context already documents for
		# arithmetic; comparisons dropped their own isinstance(Scalar)
		# special-casing to reach parity with that same dunder-dispatch
		# mechanism (see binop_fallback_eliminated), so a bare `==` now
		# needs builtins imported here too, exactly like `+` already did
		self.discovery.import_name( 'builtins' )
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
		fn = i32.names['__eq__']
		if fn.resolve is not None:
			fn.resolve()
		other = Variable( stem = '$inline0$other', qualname = f'{fn.qualname}$$inline0$other', file = fn.file, line = fn.line, type = i32 )
		t0 = ir.Temp( type = bool_cls, id = 0 )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = a, src = ir.Const( type = i32, value = 1 )),
			ir.Assign( dest = other, src = ir.Const( type = i32, value = 1 )),
			ir.DeclareTemp( temp = t0 ),
			ir.Cmp( dest = t0, op = ir.CmpOp.EQ, left = a, right = other ),
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
				disco.import_name( 'builtins' )
				comp = Compiler( disco )
				comp.import_code( '\n'.join([
					'def main() -> None:',
					'	a: i32 = 1',
					f'	b: bool = a {py_op} 1',
					'	return',
				]), filename = Path( '__test__.py' ))
				fn = comp._lower( disco.main )
				# each op's own dunder is spliced (@inline), but the spliced
				# body still bottoms out in exactly one real ir.Cmp with the
				# right op (compiler.cmp_eq/etc - see lowering.py's
				# _lower_compiler_cmp) - same assertion as before the dunder
				# migration, just reached one layer deeper
				cmp_instr = next( i for i in fn.instructions if isinstance( i, ir.Cmp ))
				self.assertEqual( cmp_instr.op, expected_cmpop )

	def test_compare_literal_on_left( self ) -> None:
		# expected type flows from whichever side is NOT the bare literal -
		# mirrors test_binop_literal_on_left. The literal `1` becomes the
		# dunder's own RECEIVER (i32.__lt__'s `value` parameter) - it's not
		# already a Variable, so the inline splice synthesizes a fresh local
		# for it ($inline0$value); `a` (already a Variable) passes straight
		# through as `other` with no synthesized local needed - same "only a
		# genuinely computed operand needs the synthesized-local fallback"
		# rule as any other @inline splice
		self.discovery.import_name( 'builtins' )
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
		fn = i32.names['__lt__']
		if fn.resolve is not None:
			fn.resolve()
		value = Variable( stem = '$inline0$value', qualname = f'{fn.qualname}$$inline0$value', file = fn.file, line = fn.line, type = i32 )
		t0 = ir.Temp( type = bool_cls, id = 0 )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = a, src = ir.Const( type = i32, value = 1 )),
			ir.Assign( dest = value, src = ir.Const( type = i32, value = 1 )),
			ir.DeclareTemp( temp = t0 ),
			ir.Cmp( dest = t0, op = ir.CmpOp.LT, left = value, right = a ),
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

	def test_compare_is_between_plain_values_is_identity_equality( self ) -> None:
		# no distinct object-identity concept exists yet for scalars - `is`
		# coincides with `==` for the value kinds this language has today
		code = '\n'.join([
			'def main() -> None:',
			'	a: i32 = 1',
			'	b: bool = a is 1',
			'	c: bool = a is not 1',
		])
		mod = self._import( code )
		lowered = self.compiler._lower( mod.get_local( 'main' ))
		cmp_instrs = [ i for i in lowered.instructions if isinstance( i, ir.Cmp ) ]
		self.assertEqual( [ c.op for c in cmp_instrs ], [ ir.CmpOp.EQ, ir.CmpOp.NE ] )
		self.assertFalse( any( isinstance( i, ir.GetAttr ) for i in lowered.instructions )) # plain scalars - no tag involved

	def test_compare_is_none_on_tagged_union_emits_a_tag_check( self ) -> None:
		# `x is None` where x: Foo|None (a real TaggedUnion, e.g.
		# sys._alloc()'s Ptr[u8]|None) means "the active member is
		# NoneType" - a tag check via the same _tagged_union_storage
		# machinery match/conditional-dispatch already use, NOT a flat Cmp
		# against a synthesized None operand of union type (which wouldn't
		# correspond to any real runtime representation)
		code = '\n'.join([
			'class Foo: pass',
			'',
			'def get() -> Foo|None:',
			'	return None',
			'',
			'def main() -> None:',
			'	x = get()',
			'	b: bool = x is None',
		])
		mod = self._import( code )
		lowered = self.compiler._lower( mod.get_local( 'main' ))
		get_attr = next( i for i in lowered.instructions if isinstance( i, ir.GetAttr ) and i.attr == 'tag' )
		x_var = mod.get_local( 'main' ).names['x']
		self.assertIs( get_attr.obj, x_var ) # x is lowered exactly once, not re-evaluated
		cmp_instrs = [ i for i in lowered.instructions if isinstance( i, ir.Cmp ) ]
		self.assertTrue( any( c.op == ir.CmpOp.EQ for c in cmp_instrs ))

	def test_compare_is_none_on_a_generic_union_specialization_emits_a_tag_check( self ) -> None:
		# regression test: unlike Foo|None above (a synthesized anonymous
		# union, always a real TaggedUnion), a value typed as a SPECIALIZATION
		# of a user-declared generic @union (Maybe[A]) has no .attributes of
		# its own and isn't a TaggedUnion instance itself - `x is None` here
		# must still see past the wrapper to find the real None member and
		# emit the same tag check, not silently fall through to a bogus flat
		# Cmp against a None-typed Const (see _tagged_union_shape)
		code = '\n'.join([
			'class A: pass',
			'',
			'@union',
			'class Maybe[T]:',
			'	Some: T',
			'	Nothing: None',
			'',
			'def main() -> None:',
			'	x: Maybe[A] = Maybe.Some( A() )',
			'	b: bool = x is None',
		])
		self.discovery.import_name( 'builtins' ) # the tag check is now an ordinary u8.__eq__ dunder call
		mod = self._import( code )
		lowered = self.compiler._lower( mod.get_local( 'main' ))
		self.assertEqual( self.discovery.errors.errors, [] )
		get_attr = next( i for i in lowered.instructions if isinstance( i, ir.GetAttr ) and i.attr == 'tag' )
		x_var = mod.get_local( 'main' ).names['x']
		self.assertIs( get_attr.obj, x_var )
		# get_attr.dest (a Temp, not a Variable) feeds the dunder call's own
		# RECEIVER slot, which the @inline splice copies into a fresh local
		# first (only an already-Variable operand passes straight through
		# unsynthesized - see _lower_inline_call's own "only a genuinely
		# computed operand needs the synthesized-local fallback" rule), so
		# the Cmp's own `left` is that synthesized copy, not get_attr.dest
		# directly - find it via the Assign feeding straight from get_attr.dest
		copy_assign = next( i for i in lowered.instructions if isinstance( i, ir.Assign ) and i.src is get_attr.dest )
		# filtered to the Cmp fed by THIS copy, not just "the only Cmp in
		# the function" - x now holds a real Some(A()) value (definite-
		# assignment requires a real initializer), so the RC leaf inside it
		# also needs its own runtime tag check at scope-exit cleanup, which
		# emits an unrelated second Cmp of its own
		cmp_instrs = [ i for i in lowered.instructions if isinstance( i, ir.Cmp ) and i.left is copy_assign.dest ]
		self.assertEqual( len( cmp_instrs ), 1 )
		self.assertEqual( cmp_instrs[0].op, ir.CmpOp.EQ )
		# the ordinal `1` (a bare literal at the dunder call site) is ALSO
		# synthesized into its own local by the same splice, same reasoning
		# as the receiver above - trace it back to its own feeding Assign
		# rather than expecting a raw ir.Const on the Cmp's own right operand
		right_assign = next( i for i in lowered.instructions if isinstance( i, ir.Assign ) and i.dest is cmp_instrs[0].right )
		self.assertEqual( right_assign.src.value, 1 ) # Nothing is member ordinal 1 (Some is 0)

	def test_compare_is_not_none_on_tagged_union_uses_ne( self ) -> None:
		code = '\n'.join([
			'class Foo: pass',
			'',
			'def get() -> Foo|None:',
			'	return None',
			'',
			'def main() -> None:',
			'	x = get()',
			'	b: bool = x is not None',
		])
		self.discovery.import_name( 'builtins' ) # the tag check is now an ordinary u8.__ne__ dunder call
		mod = self._import( code )
		lowered = self.compiler._lower( mod.get_local( 'main' ))
		cmp_instrs = [ i for i in lowered.instructions if isinstance( i, ir.Cmp ) ]
		self.assertTrue( any( c.op == ir.CmpOp.NE for c in cmp_instrs ))

	# --- in / not in (dispatch to __contains__, receiver/arg order reversed) --

	def test_in_dispatches_to_contains_with_reversed_receiver( self ) -> None:
		# `x in y` means y.__contains__(x) - y (the RIGHT operand) is the
		# receiver, x (the LEFT operand) is the sole argument, the reverse
		# of every _COMP_DUNDER-driven comparison (==, <, ...)
		code = '\n'.join([
			'class Bag:',
			'	def __contains__( self, value: i32 ) -> bool:',
			'		return value == 1',
			'',
			'def main() -> None:',
			'	b = Bag()',
			'	found: bool = 1 in b',
			'	return',
		])
		mod = self._import( code )
		lowered = self.compiler._lower( mod.get_local( 'main' ))
		calls = [ i for i in lowered.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( len( calls ), 1 )
		self.assertEqual( calls[0].target.stem, '__contains__' )
		b_var = mod.get_local( 'main' ).names['b']
		self.assertIs( calls[0].receiver, b_var )
		self.assertEqual( len( calls[0].args ), 1 )
		self.assertEqual( calls[0].args[0].value, 1 )
		self.assertFalse( any( isinstance( i, ir.Not ) for i in lowered.instructions ))

	def test_not_in_negates_contains_result( self ) -> None:
		code = '\n'.join([
			'class Bag:',
			'	def __contains__( self, value: i32 ) -> bool:',
			'		return value == 1',
			'',
			'def main() -> None:',
			'	b = Bag()',
			'	missing: bool = 1 not in b',
			'	return',
		])
		mod = self._import( code )
		lowered = self.compiler._lower( mod.get_local( 'main' ))
		calls = [ i for i in lowered.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( len( calls ), 1 )
		self.assertEqual( calls[0].target.stem, '__contains__' )
		not_instrs = [ i for i in lowered.instructions if isinstance( i, ir.Not ) ]
		self.assertEqual( len( not_instrs ), 1 )
		self.assertIs( not_instrs[0].operand, calls[0].dest )

	def test_in_without_contains_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'class Bar: pass',
			'',
			'def main() -> None:',
			'	b = Bar()',
			'	found: bool = 1 in b',
			'	return',
		])
		mod = self._import( code )
		self.compiler._lower( mod.get_local( 'main' ))
		self.assertTrue( any( '__contains__' in e for e in self.discovery.errors.errors ))

	# --- boolean operators (and/or) -------------------------------------------

	def test_boolop_and_shape( self ) -> None:
		# value-preserving `and`: `a` is only DECISIVE (its own value
		# reaches dest) when falsy - JumpIfTrue skips that decisive block
		# (assign+jump-to-end) whenever `a` is truthy, falling into the
		# continue label where `b` (the last operand) is unconditionally
		# decisive by exhaustion, no truthiness check needed for it at all.
		code = '\n'.join([
			'def main( a: bool, b: bool ) -> None:',
			'	c: bool = a and b',
			'	return',
		])
		self._import( code )
		bool_cls = self.discovery.get_intrinsics()['bool']
		none_type = self.discovery.get_none_type()
		if self.discovery.main.resolve is not None:
			self.discovery.main.resolve()
		a, b = self.discovery.main.parameters
		c = Variable( stem = 'c', qualname = 'main.c', file = Path( '__test__.py' ), line = 2, type = bool_cls )
		t0 = ir.Temp( type = bool_cls, id = 0 )
		fn = self._lower_main()
		self._assert_ir( fn, [
			ir.FuncStart( name = 'main', params = [ a, b ], return_type = none_type ),
			ir.DeclareTemp( temp = t0 ),
			ir.JumpIfTrue( cond = a, target = '__booland_continue_1__' ),
			ir.Assign( dest = t0, src = a ),
			ir.Jump( target = '__booland_0__' ),
			ir.Label( name = '__booland_continue_1__' ),
			ir.Assign( dest = t0, src = b ),
			ir.Label( name = '__booland_0__' ),
			ir.Assign( dest = c, src = t0 ),
			ir.DeleteTemp( temp = t0 ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_boolop_or_shape( self ) -> None:
		# mirror of test_boolop_and_shape above: `a` is decisive (survives
		# into dest) when TRUTHY for `or` - JumpIfFalse skips that block
		# whenever `a` is falsy, falling into `b`'s unconditional (last-
		# operand) decisive assignment instead.
		code = '\n'.join([
			'def main( a: bool, b: bool ) -> None:',
			'	c: bool = a or b',
			'	return',
		])
		self._import( code )
		bool_cls = self.discovery.get_intrinsics()['bool']
		none_type = self.discovery.get_none_type()
		if self.discovery.main.resolve is not None:
			self.discovery.main.resolve()
		a, b = self.discovery.main.parameters
		c = Variable( stem = 'c', qualname = 'main.c', file = Path( '__test__.py' ), line = 2, type = bool_cls )
		t0 = ir.Temp( type = bool_cls, id = 0 )
		fn = self._lower_main()
		self._assert_ir( fn, [
			ir.FuncStart( name = 'main', params = [ a, b ], return_type = none_type ),
			ir.DeclareTemp( temp = t0 ),
			ir.JumpIfFalse( cond = a, target = '__boolor_continue_1__' ),
			ir.Assign( dest = t0, src = a ),
			ir.Jump( target = '__boolor_0__' ),
			ir.Label( name = '__boolor_continue_1__' ),
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
		# STATIC shape: two truthiness checks (one per non-last operand,
		# `and`'s own skip check is JumpIfTrue - see test_boolop_and_shape),
		# and three labels (one continue label per non-last operand, plus
		# the shared end label).
		code = '\n'.join([
			'def main() -> None:',
			'	a: bool = True',
			'	b: bool = True',
			'	c: bool = True',
			'	d: bool = a and b and c',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertEqual( kinds.count( 'JumpIfTrue' ), 2 )
		self.assertEqual( kinds.count( 'Label' ), 3 )

	def test_boolop_discarded_non_decisive_fresh_rc_operand_is_decreffed( self ) -> None:
		# regression: a non-last operand that turns out non-decisive at
		# runtime (skipped by the short-circuit jump) is fully discarded -
		# never merged into dest - but _pending_temps/cfg._temp_states are
		# flat, in-place-mutated bookkeeping shared between the decisive
		# and continue branches, not scoped per branch the way _expr_
		# IfExp's true/false branches are. The decisive branch's own flush
		# (generated FIRST, compile-time-sequentially, right after the
		# skip-jump) used to permanently remove the operand from both
		# structures - by the time the continue branch's own flush ran, it
		# found nothing left to release, so a FRESH RC operand discarded
		# this way (e.g. `base.upper() and (tail + '')` when base.upper()
		# is truthy) never got its Decref emitted in EITHER branch's
		# actual code - a real, silent LEAK, confirmed via direct
		# inspection (not caught by refcount()-based compile+run testing,
		# which only ever inspected the SURVIVING value). Fixed via a
		# _pending_temps/cfg._temp_states snapshot+restore around the
		# branch split (see _expr_BoolOp's own comment). Checked via IR
		# shape (not refcount()) since a pure leak doesn't crash or
		# corrupt anything a compile+run test could otherwise observe.
		self.discovery.import_name( 'builtins' )
		code = '\n'.join([
			'def main() -> None:',
			'	base: str = "x"',
			'	tail: str = "y"',
			'	kept: str = base.upper() and ( tail + "" )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		# base.upper()'s own Decref must be emitted exactly once - in the
		# continue branch's own code, the only path that actually
		# discards it (the decisive branch, generated first, keeps it
		# instead - see _expr_BoolOp's own emit_decisive). isinstance(...,
		# ir.Temp) specifically (not Variable) - only a TEMP's own Decref
		# is what's in question here; base/tail/kept's own epilogue
		# Decrefs (all also str) are a separate, already-correct concern.
		temp_str_decrefs = [
			instr.value for instr in fn.instructions
			if type( instr ).__name__ == 'Decref'
			and isinstance( instr.value, ir.Temp )
			and getattr( instr.value.type, 'stem', None ) == 'str'
		]
		self.assertEqual( len( temp_str_decrefs ), 1 )

	# --- if statements ---------------------------------------------------------

	def test_if_without_else_shape( self ) -> None:
		code = '\n'.join([
			'def main( a: bool ) -> None:',
			'	if a:',
			'		b: i32 = 1',
			'	return',
		])
		self._import( code )
		i32 = self.discovery.get_intrinsics()['i32']
		none_type = self.discovery.get_none_type()
		if self.discovery.main.resolve is not None:
			self.discovery.main.resolve()
		a = self.discovery.main.parameters[0]
		b = Variable( stem = 'b', qualname = 'main.b', file = Path( '__test__.py' ), line = 3, type = i32 )
		fn = self._lower_main()
		self._assert_ir( fn, [
			ir.FuncStart( name = 'main', params = [ a ], return_type = none_type ),
			ir.JumpIfFalse( cond = a, target = '__if_else_0__' ),
			ir.Assign( dest = b, src = ir.Const( type = i32, value = 1 )),
			ir.Label( name = '__if_else_0__' ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_if_with_else_shape( self ) -> None:
		# b declared bare, ONCE, before the if/else - not `b: i32 = 1`/
		# `b: i32 = 2` once per arm (that pattern is now a redeclaration
		# error - an explicit type annotation is only ever given once per
		# variable, even across mutually exclusive branches - see
		# test_annotated_redeclaration_across_branches_is_a_compile_error
		# below). Both arms' own `b = ...` are ordinary, INFERRED-type-
		# already-established reassignments to the SAME b, matching the
		# "type set on first [here, the bare] assignment, even within a
		# branch" rule - so this IR shape has only ONE b Variable object,
		# not two, and the bare declaration itself emits no instruction at
		# all (ir.DeclareLocal is reserved for a narrower, unrelated
		# @inline-splice case - see its own docstring)
		code = '\n'.join([
			'def main( a: bool ) -> None:',
			'	b: i32',
			'	if a:',
			'		b = 1',
			'	else:',
			'		b = 2',
			'	return',
		])
		self._import( code )
		i32 = self.discovery.get_intrinsics()['i32']
		none_type = self.discovery.get_none_type()
		if self.discovery.main.resolve is not None:
			self.discovery.main.resolve()
		a = self.discovery.main.parameters[0]
		b = Variable( stem = 'b', qualname = 'main.b', file = Path( '__test__.py' ), line = 2, type = i32 )
		fn = self._lower_main()
		self._assert_ir( fn, [
			ir.FuncStart( name = 'main', params = [ a ], return_type = none_type ),
			ir.JumpIfFalse( cond = a, target = '__if_else_0__' ),
			ir.Assign( dest = b, src = ir.Const( type = i32, value = 1 )),
			ir.Jump( target = '__if_end_1__' ),
			ir.Label( name = '__if_else_0__' ),
			ir.Assign( dest = b, src = ir.Const( type = i32, value = 2 )),
			ir.Label( name = '__if_end_1__' ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_if_elif_else_chains_via_nested_orelse( self ) -> None:
		# elif is just a nested If inside orelse in the AST - confirms it
		# "just works" through the same recursive _lower_stmt dispatch, no
		# special-casing needed. x declared bare, ONCE, before the chain -
		# not `x: i32 = 1`/`= 2`/`= 3` once per arm (a redeclaration error -
		# an explicit type annotation is only ever given once per variable,
		# even across mutually exclusive branches - see
		# test_annotated_redeclaration_across_branches_is_a_compile_error)
		code = '\n'.join([
			'def main() -> None:',
			'	a: bool = True',
			'	c: bool = True',
			'	x: i32',
			'	if a:',
			'		x = 1',
			'	elif c:',
			'		x = 2',
			'	else:',
			'		x = 3',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		# outer if and the nested elif-as-If each have their own orelse (the
		# elif chain, and its own else respectively), so each contributes
		# its own else-Label + end-Label + skip-Jump pair - two JumpIfFalse
		# (one per test), three Assigns (one per branch) plus a's and c's own
		# real initializers (definite-assignment requires them), two Jumps
		# and four Labels (one else + one end, per level)
		self.assertEqual( kinds.count( 'JumpIfFalse' ), 2 )
		self.assertEqual( kinds.count( 'Jump' ), 2 )
		self.assertEqual( kinds.count( 'Label' ), 4 )
		self.assertEqual( kinds.count( 'Assign' ), 5 )

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
		self.discovery.import_name( 'builtins' ) # match-arm tag dispatch is now an ordinary u8.__eq__ dunder call - see test_construct_and_match_round_trip (emitter_c_test.py)
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		# Foo has exactly 2 members and both are explicitly matched here -
		# type_resolver.py's visit_Match now recognizes this as provably
		# exhaustive (Trap 1's own generalization - see steady-dancing-
		# haven.md) and splices the LAST case's own body in unconditionally
		# instead of chaining it behind a redundant tag check: only ONE
		# real Cmp (case 0's own tag == 0), one booland short-circuit +
		# one if-test check for case 0 - case 1 (Baz) has no test at all
		# anymore, reached unconditionally once case 0's own check fails.
		# The booland's own short-circuit is now a JumpIfTrue (`and` skips
		# its decisive/value-preserving block - see _expr_BoolOp - whenever
		# the first operand is truthy, continuing to the second instead of
		# forcing a bool coercion the way this used to); the if-test itself
		# is still a separate, ordinary JumpIfFalse.
		self.assertEqual( kinds.count( 'Cmp' ), 1 )
		self.assertEqual( kinds.count( 'JumpIfTrue' ), 1 ) # case 0's own booland short-circuit
		self.assertEqual( kinds.count( 'JumpIfFalse' ), 1 ) # case 0's own if-test

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
		# Foo has exactly ONE member (Bar) and this is the match's only
		# case - provably exhaustive (Trap 1's own generalization) with
		# nothing else it could possibly be, so type_resolver.py's
		# visit_Match splices the whole case in unconditionally: no `tag`
		# GetAttr/Cmp at all anymore, just the payload extraction
		# (data.v_Bar) - see test_match_union_shape's own comment
		self.assertEqual( [ g.attr for g in getattrs ], [ 'data', 'v_Bar' ] )

	def test_match_result_ok_err_shape( self ) -> None:
		# Result is a real @union now - Result.Ok(...)/Err(...) match
		# patterns are no longer special-cased at all, they fall through to
		# the exact same generic TaggedUnion branch
		# test_match_union_construction_and_extraction_round_trip already
		# exercises above (tag/data.v_<member>, via _tagged_union_storage)
		code = '\n'.join([
			'class MyError: pass',
			'',
			'@union',
			'class Result[T,E]:',
			'	Ok: T',
			'	Err: E',
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
		self.discovery.import_name( 'builtins' ) # non-exhaustive match's own tag check is now an ordinary u8.__eq__ dunder call
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		# r's own RC tracking (own incref right after construction, own
		# tag-gated decref at scope exit - correct: MyError is RC, so
		# Result[i32,MyError] is a real mixed-leaves union, verified with a
		# real compile+run stress test, not just by this passing) ALSO
		# shows up as its own tag/data/v_Err GetAttr sequences elsewhere in
		# the same function now - this test only cares about the match
		# statement's own tag+payload extraction shape, so check that it
		# appears as a contiguous run, not that it's the only thing here
		attrs = [ i.attr for i in fn.instructions if isinstance( i, ir.GetAttr ) ]
		windows = [ attrs[i:i+3] for i in range( len( attrs ) - 2 ) ]
		self.assertIn( [ 'tag', 'data', 'v_Ok' ], windows )

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

	def test_match_binding_name_reused_with_incompatible_type_gets_a_clear_diagnostic( self ) -> None:
		# every local (including a `case T(name):` match-arm binding) is
		# function-scoped, no per-arm/per-match scoping - reusing a binding
		# name across two SEPARATE, unrelated match statements is ordinary
		# and fine when both sides agree on the payload type (see the
		# companion test below), but a genuine MISMATCH used to produce a
		# bare "expected X, got Y" that never explained where X came from
		# (nothing in the SECOND match's own source mentions the FIRST
		# match's error type at all). Confirms the diagnostic now names the
		# real cause instead.
		code = '\n'.join([
			'import builtins',
			'def get_a() -> builtins.Result[i32, builtins.OverflowError|builtins.IndexError]:',
			'	return builtins.Result.Err( builtins.OverflowError() )',
			'',
			'def get_b() -> builtins.Result[i32, builtins.IndexError|builtins.KeyError]:',
			'	return builtins.Result.Err( builtins.IndexError() )',
			'',
			'def main() -> i32:',
			'	match get_a():',
			'		case builtins.Result.Err( e ):',
			'			pass',
			'		case builtins.Result.Ok( _ ):',
			'			return 1',
			'	match get_b():',
			'		case builtins.Result.Err( e ):',
			'			pass',
			'		case builtins.Result.Ok( _ ):',
			'			return 2',
			'	return 0',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( "'e' is already declared earlier in this function", self.discovery.errors.errors[0] )
		self.assertIn( 'another match', self.discovery.errors.errors[0] )

	def test_match_binding_name_reused_with_same_type_is_allowed( self ) -> None:
		# the companion case: reusing a binding name across two separate
		# match statements, where both sides happen to agree on the
		# payload type, is ordinary and must NOT be rejected - the same
		# posture reusing a loop counter across two separate loops already
		# has.
		code = '\n'.join([
			'import builtins',
			'def get_a() -> builtins.Result[i32, builtins.IndexError|builtins.OverflowError]:',
			'	return builtins.Result.Err( builtins.OverflowError() )',
			'',
			'def get_b() -> builtins.Result[i32, builtins.IndexError|builtins.OverflowError]:',
			'	return builtins.Result.Err( builtins.IndexError() )',
			'',
			'def main() -> i32:',
			'	match get_a():',
			'		case builtins.Result.Err( e ):',
			'			pass',
			'		case builtins.Result.Ok( _ ):',
			'			return 1',
			'	match get_b():',
			'		case builtins.Result.Err( e ):',
			'			pass',
			'		case builtins.Result.Ok( _ ):',
			'			return 2',
			'	return 0',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_match_narrowing_shadow_reuse_is_unaffected_by_binding_reuse_diagnostic( self ) -> None:
		# the one exception the reused-binding-name diagnostic above must
		# NOT fire for: a case pattern that reuses the SUBJECT's own name
		# (`match r: case i32(r):`) narrows r in place (is_narrowing_bind,
		# a genuinely different code path - see _stmt_Assign's own early
		# return for it) rather than rebinding a distinct value, so two
		# SEPARATE such narrowings of two DIFFERENTLY-typed subjects
		# sharing a name (r narrows i32|None, s narrows i32|None here,
		# same underlying leaf type, different subjects) must still work.
		code = '\n'.join([
			'import builtins',
			'def main() -> i32:',
			'	r: i32|None = 5',
			'	match r:',
			'		case None:',
			'			return 1',
			'		case i32( r ):',
			'			pass',
			'	x: i32 = r',
			'	if x != 5:',
			'		return 2',
			'	return 0',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

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
			'	x: A|B = A()',
			'	foo( x )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertEqual( kinds.count( 'Cmp' ), 1 )
		self.assertEqual( kinds.count( 'JumpIfFalse' ), 1 )
		# filtered to foo(...) calls specifically - x's own A() initializer
		# (needed now that x must be definitely-assigned) contributes an
		# unrelated third Call, the A|B union member constructor
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) and i.target.qualname == '__test__.foo' ]
		self.assertEqual( len( calls ), 2 ) # one per possible target - only one runs at runtime
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
			'	x: A|B = A()',
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
			'	x: bytes|bytearray = bytes()',
			'	flen( x )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		strlike_cls = self.discovery.modules['__test__'].get_local( 'strlike' )
		# filtered to flen(...) calls specifically - x's own bytes() initializer
		# (needed now that x must be definitely-assigned) contributes an
		# unrelated third Call, the bytes|bytearray union member constructor
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) and i.target.qualname == '__test__.flen' ]
		self.assertEqual( len( calls ), 2 ) # exactly bytes and bytearray - never a third for strlike
		self.assertTrue( all( c.target.parameters[0].type is not strlike_cls for c in calls ))

	# --- union-typed receiver method calls (receiver narrowing) ---------------

	def test_union_receiver_call_dispatches_per_leaf( self ) -> None:
		# mirrors lib/builtins/__init__.py's real copy_from.get_const_ptr()
		# shape (copy_from: bytes|bytearray) - get_const_ptr isn't an
		# @overload group, each leaf just has its own unrelated method under
		# this name, so it's the RECEIVER's own tag that has to be checked,
		# not any argument's
		code = '\n'.join([
			'class A:',
			'	def get( self ) -> i32:',
			'		return 1',
			'',
			'class B:',
			'	def get( self ) -> i32:',
			'		return 2',
			'',
			'def main() -> None:',
			'	x: A|B = A()',
			'	x.get()',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertEqual( kinds.count( 'Cmp' ), 1 )
		self.assertEqual( kinds.count( 'JumpIfFalse' ), 1 )
		# receiver-bound calls only - x's own A() initializer (needed now
		# that x must be definitely-assigned) contributes an unrelated
		# receiver-less Call, the A|B union member constructor
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) and i.receiver is not None ]
		self.assertEqual( len( calls ), 2 )
		a_cls = self.discovery.modules['__test__'].get_local( 'A' )
		b_cls = self.discovery.modules['__test__'].get_local( 'B' )
		receiver_types = [ c.receiver.type for c in calls ]
		self.assertIn( a_cls, receiver_types )
		self.assertIn( b_cls, receiver_types ) # narrowed to the concrete leaf, never the raw A|B union
		self.assertEqual( len( { id( c.target ) for c in calls } ), 2 ) # A.get and B.get are distinct Functions

	def test_union_receiver_call_dispatches_per_leaf_on_a_generic_union_specialization( self ) -> None:
		# regression test: same shape as test_union_receiver_call_dispatches_
		# per_leaf, but the receiver is a SPECIALIZATION of a user-declared
		# generic @union (Choice[A,B]), not a synthesized anonymous union -
		# Specialization has no .attributes of its own and isn't a
		# TaggedUnion instance itself, so finding "get isn't declared on
		# Choice itself, fall back to each leaf's own get" needs to see past
		# the wrapper (see _tagged_union_shape) rather than mistakenly
		# reporting `get` as not found at all
		code = '\n'.join([
			'class A:',
			'	def get( self ) -> i32:',
			'		return 1',
			'',
			'class B:',
			'	def get( self ) -> i32:',
			'		return 2',
			'',
			'@union',
			'class Choice[T,U]:',
			'	First: T',
			'	Second: U',
			'',
			'def main() -> None:',
			'	x: Choice[A,B] = Choice.First( A() )',
			'	x.get()',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertEqual( kinds.count( 'Cmp' ), 1 )
		self.assertEqual( kinds.count( 'JumpIfFalse' ), 1 )
		# receiver-bound calls only - x's own Choice.First(A()) initializer
		# (needed now that x must be definitely-assigned) contributes two
		# unrelated receiver-less Calls, the A() constructor and the
		# Choice.First member constructor
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) and i.receiver is not None ]
		self.assertEqual( len( calls ), 2 )
		a_cls = self.discovery.modules['__test__'].get_local( 'A' )
		b_cls = self.discovery.modules['__test__'].get_local( 'B' )
		receiver_types = [ c.receiver.type for c in calls ]
		self.assertIn( a_cls, receiver_types )
		self.assertIn( b_cls, receiver_types ) # narrowed to the concrete leaf, never the raw Choice[A,B] specialization
		self.assertEqual( len( { id( c.target ) for c in calls } ), 2 ) # A.get and B.get are distinct Functions

	def test_union_receiver_call_mismatched_return_type_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'class A:',
			'	def get( self ) -> i32:',
			'		return 1',
			'',
			'class B:',
			'	def get( self ) -> bool:',
			'		return True',
			'',
			'def main() -> None:',
			'	x: A|B = A()',
			'	x.get()',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'disagree on return type', self.discovery.errors.errors[0] )

	def test_union_receiver_call_prefers_a_real_method_declared_on_the_union_itself( self ) -> None:
		# a real @union class CAN declare its own real method - that wins
		# outright, with no receiver-narrowing dispatch synthesized at all
		code = '\n'.join([
			'@union',
			'class Foo:',
			'	Bar: i32',
			'	Baz: i32',
			'	def get( self ) -> i32:',
			'		return 0',
			'',
			'def main() -> None:',
			'	f: Foo = Foo.Bar( 5 )',
			'	f.get()',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		# Foo.Bar(5) is now ALSO an ordinary call (to the synthesized
		# constructor, receiver=None) - the real assertion here is that
		# f.get()'s own dispatch is a single, direct, receiver-based call
		# to Foo's own real method, with no per-leaf narrowing synthesized
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) ]
		receiver_calls = [ c for c in calls if c.receiver is not None ]
		self.assertEqual( len( receiver_calls ), 1 )
		self.assertIs( receiver_calls[0].receiver.type, self.discovery.modules['__test__'].get_local( 'Foo' ))

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

	def test_overload_literal_arg_magnitude_disambiguates_candidates( self ) -> None:
		# f(300) between f(x: i8)/f(x: i32) used to be rejected as "ambiguous"
		# purely because both are int-KIND-compatible - 300 obviously can't
		# fit i8, so there's really only one answer
		code = '\n'.join([
			'def foo( x: i8 ) -> None:',
			'	pass',
			'',
			'def foo( x: i32 ) -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	foo( 300 )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		i32 = self.discovery.get_intrinsics()['i32']
		call = next( i for i in fn.instructions if isinstance( i, ir.Call ) )
		self.assertEqual( call.args, [ ir.Const( type = i32, value = 300 ) ] )

	def test_overload_literal_arg_out_of_range_for_every_candidate_still_reports_ambiguous( self ) -> None:
		# f(300) where NEITHER candidate can hold it (i8 max 127, u8 max 255)
		# - magnitude narrowing eliminates every candidate, so candidate_types
		# is left as the original, unnarrowed kind-only list and the existing
		# "ambiguous" error is unchanged (not attempting a better message for
		# this case - out of scope)
		code = '\n'.join([
			'def foo( x: i8 ) -> None:',
			'	pass',
			'',
			'def foo( x: u8 ) -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	foo( 300 )',
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

	# --- bare lambda arguments to overloaded calls ------------------------------

	def test_overload_lambda_arg_resolves_via_unique_callable_candidate( self ) -> None:
		code = '\n'.join([
			'def foo( x: i32 ) -> None:',
			'	pass',
			'',
			'def foo( x: Ptr[Callable[[i32],i32]] ) -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	foo( lambda v: v )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		call = next( i for i in fn.instructions if isinstance( i, ir.Call ) )
		# resolved to the SECOND foo (the Callable-typed candidate), not the
		# first (i32) - a lambda can never plausibly satisfy the i32 one
		group = self.discovery.modules['__test__'].get_local( 'foo' )
		self.assertIs( call.target, group.implementations[1] )
		self.assertIsNotNone( self.compiler.lowering._type_resolver._callable_type_of( call.args[0].type ) )

	def test_overload_lambda_arg_no_callable_candidate_falls_through_to_inference_error( self ) -> None:
		# no candidate at all is callable-shaped - unaffected by the new
		# lambda-matching branch, still reaches _expr_Lambda's own
		# pre-existing "no expected Callable[...] context" error, exactly as
		# it did before a lambda was ever handled specially here
		code = '\n'.join([
			'def foo( x: i32 ) -> None:',
			'	pass',
			'',
			'def foo( x: u8 ) -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	foo( lambda v: v )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'cannot infer lambda parameter types', self.discovery.errors.errors[0] )

	def test_overload_lambda_arg_ambiguous_between_callable_candidates_is_rejected( self ) -> None:
		code = '\n'.join([
			'def foo( x: Ptr[Callable[[i32],i32]] ) -> None:',
			'	pass',
			'',
			'def foo( x: Ptr[Callable[[u8],u8]] ) -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	foo( lambda v: v )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'ambiguous lambda argument', self.discovery.errors.errors[0] )

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

	# --- bare-call generic function type inference ------------------------------

	def test_generic_function_call_infers_type_arg_from_argument( self ) -> None:
		# mylen(a), no explicit [T] - T must be inferred from a's own type
		code = '\n'.join([
			'class A:',
			'	def __len__( self ) -> usize:',
			'		return 5',
			'',
			'def mylen[T]( t: T ) -> usize:',
			'	return t.__len__()',
			'',
			'def main() -> usize:',
			'	a: A = A()',
			'	return mylen( a )',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		call = next( i for i in fn.instructions if isinstance( i, ir.Call ) )
		self.assertEqual( call.target.qualname, '__test__.mylen[__test__.A]' )

	def test_generic_function_call_infers_type_arg_through_one_level_of_nesting( self ) -> None:
		# unwrap(b) where b: Box[i32] - T isn't the parameter's own declared
		# type (that's Box[T], a Specialization), so this has to unify one
		# level deep (same base, pair up args) to find T=i32
		code = '\n'.join([
			'class Box[T]:',
			'	v: T',
			'',
			'def unwrap[T]( b: Box[T] ) -> T:',
			'	return b.v',
			'',
			'def main() -> i32:',
			'	b: Box[i32] = Box( v = 1 )',
			'	return unwrap( b )',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		call = next( i for i in fn.instructions if isinstance( i, ir.Call ) )
		self.assertEqual( call.target.qualname, '__test__.unwrap[intrinsics.i32]' )
		i32 = self.discovery.get_intrinsics()['i32']
		self.assertEqual( call.target.return_type, i32 )

	def test_generic_function_call_cannot_infer_type_arg_is_a_compile_error( self ) -> None:
		# T never appears in any parameter position - nothing to infer it
		# from, and no explicit [T] was given either
		code = '\n'.join([
			'def make[T]() -> usize:',
			'	return 0',
			'',
			'def main() -> usize:',
			'	return make()',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'cannot infer type parameter', self.discovery.errors.errors[0] )

	def test_generic_function_call_conflicting_inference_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'def pair[T]( a: T, b: T ) -> usize:',
			'	return 0',
			'',
			'def main() -> usize:',
			'	x: i32 = 1',
			'	y: u8 = 1',
			'	return pair( x, y )',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'inferred as both', self.discovery.errors.errors[0] )

	# --- move(x) call-site syntax must agree with a move[T] parameter ---------

	def test_move_parameter_without_call_site_wrapper_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'class Foo: pass',
			'',
			'def takeown( x: move[Foo] ) -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	f: Foo',
			'	takeown( f )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'must pass move(f)', self.discovery.errors.errors[0] )

	def test_plain_parameter_with_move_call_site_wrapper_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'class Foo: pass',
			'',
			'def takeown( x: Foo ) -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	f: Foo',
			'	takeown( move( f ))',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'is not move[T]', self.discovery.errors.errors[0] )

	def test_move_parameter_with_matching_call_site_wrapper_lowers_cleanly( self ) -> None:
		code = '\n'.join([
			'class Foo: pass',
			'',
			'def takeown( x: move[Foo] ) -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	f: Foo = Foo()',
			'	takeown( move( f ))',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( len( calls ), 1 )
		f_cls = self.discovery.modules['__test__'].get_local( 'Foo' )
		self.assertIs( calls[0].args[0].type, f_cls ) # move(f) unwraps to the real f, not a leftover call expression

	def test_move_of_a_field_access_is_a_compile_error( self ) -> None:
		# cfg.py's ownership tracking only tracks top-level bindings (params/
		# locals/self), never struct/union fields - move(self.foo) would
		# silently miscompile into a double-free (source field keeps its
		# pointer, destination's destructor frees the same storage again)
		code = '\n'.join([
			'class Foo: pass',
			'',
			'class Holder:',
			'	foo: Foo',
			'	def __init__( self ) -> None:',
			'		self.foo = Foo()',
			'',
			'def takeown( x: move[Foo] ) -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	h: Holder = Holder()',
			'	takeown( move( h.foo ))',
			'	return',
		])
		self._import( code )
		self._lower_main()
		error = self.discovery.errors.errors[0]
		self.assertIn( "cannot move 'h.foo' directly", error )
		self.assertIn( 'assign it to a local variable first', error.lower() )
		self.assertIn( 'field or subscript', error ) # same restriction applies to both shapes

	def test_move_of_a_subscript_is_a_compile_error( self ) -> None:
		# needs builtins for list[T]
		disco = Discovery( import_builtins = True )
		comp = Compiler( disco )
		code = '\n'.join([
			'class Foo: pass',
			'',
			'def takeown( x: move[Foo] ) -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	fs: list[Foo] = [ Foo() ]',
			'	takeown( move( fs[0] ))',
			'	return',
		])
		comp.import_code( code, filename = Path( '__test__.py' ))
		comp._lower( disco.main )
		error = disco.errors.errors[0]
		self.assertIn( "cannot move 'fs[0]' directly", error )
		self.assertIn( 'assign it to a local variable first', error.lower() )
		self.assertIn( 'field or subscript', error ) # same restriction applies to both shapes

	def test_move_of_a_fresh_constructor_call_lowers_cleanly( self ) -> None:
		# a fresh rvalue (e.g. a constructor call) has no other owner to
		# double-free, unlike a field/subscript - this must stay legal (see
		# lib/http/client.py's bytes.from_bytearray(move(bytearray(0))))
		code = '\n'.join([
			'class Foo: pass',
			'',
			'def takeown( x: move[Foo] ) -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	takeown( move( Foo() ))',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

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

	def test_compiler_sizeof_rcclass_emits_sizeof_instruction( self ) -> None:
		# unlike an intrinsic scalar (folds straight to ir.Const - no
		# field-layout algorithm exists in this compiler, nor should one -
		# that's the C compiler's own job), a real class-like type stays a
		# genuine ir.SizeOf instruction, letting the emitter defer to a
		# literal C `sizeof(...)` expression
		code = '\n'.join([
			'class Foo: pass',
			'',
			'def main() -> None:',
			'	x: usize = compiler.sizeof( Foo )',
			'	return',
		])
		mod = self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		sizeofs = [ i for i in fn.instructions if isinstance( i, ir.SizeOf ) ]
		self.assertEqual( len( sizeofs ), 1 )
		foo_cls = mod.get_local( 'Foo' )
		self.assertIs( sizeofs[0].type, foo_cls )
		usize = self.discovery.get_intrinsics()['usize']
		self.assertIs( sizeofs[0].dest.type, usize )

	# --- compiler.sizeof(x) (value argument) ------------------------------------

	def test_compiler_sizeof_of_a_local_variable_folds_to_const( self ) -> None:
		# compiler.sizeof(x) where x is a plain scalar-typed local - same
		# ir.Const fold as compiler.sizeof(u32) itself, just resolved
		# through the value's own static type instead of a type name
		code = '\n'.join([
			'def main() -> None:',
			'	v: u32 = 1',
			'	x: usize = compiler.sizeof( v )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		consts = [ i.src.value for i in fn.instructions if isinstance( i, ir.Assign ) ]
		self.assertEqual( consts, [ 1, 4 ] )

	def test_compiler_sizeof_of_self_emits_sizeof_instruction( self ) -> None:
		# compiler.sizeof(self) inside an ordinary method - same ir.SizeOf
		# shape as compiler.sizeof(Foo) itself, just resolved through
		# self's own static type instead of the class name
		code = '\n'.join([
			'class Foo:',
			'	a: i32',
			'',
			'	def size( self ) -> usize:',
			'		return compiler.sizeof( self )',
			'',
			'def main() -> None:',
			'	return',
		])
		mod = self._import( code )
		fn = self.compiler._lower( self._method( mod, 'Foo', 'size' ))
		self.assertEqual( self.discovery.errors.errors, [] )
		sizeofs = [ i for i in fn.instructions if isinstance( i, ir.SizeOf ) ]
		self.assertEqual( len( sizeofs ), 1 )
		foo_cls = mod.get_local( 'Foo' )
		self.assertIs( sizeofs[0].type, foo_cls )

	def test_compiler_sizeof_of_self_is_self_escape_safe_before_construction_completes( self ) -> None:
		# compiler.sizeof(self) reads only self's static TYPE - it never
		# lowers/evaluates self itself (see _static_type_of_value_expr),
		# so self never becomes an operand of any emitted instruction and
		# this compiles cleanly even before every required attribute is
		# initialized, unlike an ordinary use of self (see the self-escape
		# tests below, e.g. test_self_escape_via_plain_argument_is_a_compile_error)
		code = '\n'.join([
			'class Bar:',
			'	a: i32',
			'',
			'	def __init__( self ) -> None:',
			'		x: usize = compiler.sizeof( self )',
			'		self.a = 1',
			'',
			'def main() -> None:',
			'	return',
		])
		mod = self._import( code )
		fn = self.compiler._lower( self._method( mod, 'Bar', '__init__' ))
		self.assertEqual( self.discovery.errors.errors, [] )
		sizeofs = [ i for i in fn.instructions if isinstance( i, ir.SizeOf ) ]
		self.assertEqual( len( sizeofs ), 1 )
		bar_cls = mod.get_local( 'Bar' )
		self.assertIs( sizeofs[0].type, bar_cls )

	def test_compiler_sizeof_of_an_attribute_chain( self ) -> None:
		# compiler.sizeof(self.field) - resolved via the same non-emitting
		# type lookup, one Attribute hop deeper; no GetAttr instruction is
		# emitted for the read - the value itself is never evaluated
		code = '\n'.join([
			'class Inner:',
			'	v: i32',
			'',
			'class Outer:',
			'	inner: Inner',
			'',
			'	def size( self ) -> usize:',
			'		return compiler.sizeof( self.inner )',
			'',
			'def main() -> None:',
			'	return',
		])
		mod = self._import( code )
		fn = self.compiler._lower( self._method( mod, 'Outer', 'size' ))
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertEqual( [ i for i in fn.instructions if isinstance( i, ir.GetAttr ) ], [] )
		sizeofs = [ i for i in fn.instructions if isinstance( i, ir.SizeOf ) ]
		self.assertEqual( len( sizeofs ), 1 )
		inner_cls = mod.get_local( 'Inner' )
		self.assertIs( sizeofs[0].type, inner_cls )

	def test_compiler_sizeof_of_narrowed_name_uses_narrowed_type( self ) -> None:
		# inside a `case U.A(u):` arm reusing the subject's own name, u is
		# narrowed (cfg.py's narrow(), see _stmt_Assign's is_narrowing_bind
		# handling) to A's own leaf type (u8) - compiler.sizeof(u) should
		# use that narrowed scalar type, not U's own (larger) union type
		code = '\n'.join([
			'@union',
			'class U:',
			'	A: u8',
			'	B: i64',
			'',
			'def main() -> None:',
			'	u: U = U.A( 1 )',
			'	match u:',
			'		case U.A( u ):',
			'			x: usize = compiler.sizeof( u )',
			'		case U.B( u ):',
			'			pass',
			'	return',
		])
		self.discovery.import_name( 'builtins' ) # the match-arm tag dispatch is now an ordinary u8.__eq__ dunder call
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		assigns = { getattr( i.dest, 'stem', None ): i.src for i in fn.instructions if isinstance( i, ir.Assign ) }
		self.assertIsInstance( assigns['x'], ir.Const )
		self.assertEqual( assigns['x'].value, 1 ) # u8's own size, not U's (tag + i64 payload)

	def test_compiler_sizeof_of_a_call_expression_is_a_compile_error( self ) -> None:
		# calling a function just to inspect its return type would require
		# actually evaluating it - not supported; make() itself must never
		# be lowered/called just to answer compiler.sizeof(...)
		code = '\n'.join([
			'def make() -> i32:',
			'	return 1',
			'',
			'def main() -> None:',
			'	x: usize = compiler.sizeof( make() )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertIn( 'argument must be a type or a value with a known type', self.discovery.errors.errors[0] )
		self.assertEqual( [ i for i in fn.instructions if isinstance( i, ir.Call ) ], [] )

	# --- compiler.caller_line() / compiler.caller_file() -----------------------

	def test_compiler_caller_line_folds_to_const_with_callsite_lineno( self ) -> None:
		code = '\n'.join([
			'def f( x: i32 = compiler.caller_line() ) -> i32:',
			'	return x',
			'',
			'def main() -> None:',
			'	a: i32 = f()',
			'	b: i32 = f()',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) and getattr( i.target, 'stem', None ) == 'f' ]
		self.assertEqual( [ c.kwargs['x'].value for c in calls ], [ 5, 6 ] ) # each call's OWN line in main(), not f's definition line (1)

	def test_compiler_caller_file_folds_to_const_with_callsite_file( self ) -> None:
		import tempfile
		with tempfile.TemporaryDirectory() as tmp:
			root = Path( tmp )
			( root / 'a.py' ).write_text( '\n'.join([
				'def f( x: str = compiler.caller_file() ) -> str:',
				'	return x',
			]), encoding = 'utf-8' )
			main_path = root / '__main__.py'
			main_path.write_text( '\n'.join([
				'from a import f',
				'def main() -> None:',
				'	s: str = f()',
				'	return',
			]), encoding = 'utf-8' )
			disco = Discovery( paths = [ root, Path( discovery.__file__ ).parent / 'lib' ], import_builtins = True ) # str is a builtin, not an intrinsic
			compiler = Compiler( disco )
			compiler.import_file( main_path )
			compiler.run()
			self.assertEqual( disco.errors.errors, [] )
			main_fn = next( lf for lf in compiler.functions if lf.function.qualname == 'main' )
			calls = [ i for i in main_fn.instructions if isinstance( i, ir.Call ) ]
			self.assertEqual( len( calls ), 1 )
			# the CALLER's file (__main__.py), not f's defining module (a.py)
			self.assertEqual( calls[0].kwargs['x'].value, str( main_path ) )

	def test_compiler_caller_line_explicit_argument_overrides_default( self ) -> None:
		code = '\n'.join([
			'def f( x: i32 = compiler.caller_line() ) -> i32:',
			'	return x',
			'',
			'def main() -> None:',
			'	a: i32 = f( x = 999 )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) and getattr( i.target, 'stem', None ) == 'f' ]
		self.assertEqual( calls[0].kwargs['x'].value, 999 ) # no folding - explicit argument wins, ordinary default semantics

	def test_compiler_caller_line_used_outside_default_position_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	x: i32 = compiler.caller_line() + 1',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( "only valid as a parameter's default value", self.discovery.errors.errors[0] )

	def test_compiler_caller_line_bare_statement_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	compiler.caller_line()',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( "only valid as a parameter's default value", self.discovery.errors.errors[0] )

	def test_compiler_caller_line_wrong_param_type_is_a_compile_error( self ) -> None:
		disco = Discovery( import_builtins = True ) # str is a builtin, not an intrinsic
		comp = Compiler( disco )
		code = '\n'.join([
			'def f( x: str = compiler.caller_line() ) -> str:',
			'	return x',
			'',
			'def main() -> None:',
			'	s: str = f()',
			'	return',
		])
		comp.import_code( code, filename = Path( '__test__.py' ))
		comp._lower( disco.main )
		self.assertIn( 'can only default an i32 parameter', disco.errors.errors[0] )

	def test_compiler_caller_file_wrong_param_type_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'def f( x: i32 = compiler.caller_file() ) -> i32:',
			'	return x',
			'',
			'def main() -> None:',
			'	s: i32 = f()',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'can only default a str parameter', self.discovery.errors.errors[0] )

	def test_bare_generic_call_fills_in_caller_line_default( self ) -> None:
		# confirms _fill_generic_call_defaults (a separate default-fill site
		# from _lower_call_args) routes through the same shared helper
		code = '\n'.join([
			'def take[S]( seq: S, ln: i32 = compiler.caller_line() ) -> i32:',
			'	return ln',
			'',
			'def main() -> i32:',
			'	t = ( 1, 2, 3 )',
			'	return take( t )',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) and getattr( i.target, 'stem', None ) == 'take' ]
		self.assertEqual( len( calls ), 1 )
		self.assertEqual( calls[0].kwargs['ln'].value, 6 )

	def test_overload_resolved_call_fills_in_caller_line_default( self ) -> None:
		# confirms the resolved-Overload-candidate default-fill site (the
		# third of three) also routes through the shared helper
		disco = Discovery( import_builtins = True ) # str/bytes are builtins, not intrinsics
		comp = Compiler( disco )
		code = '\n'.join([
			'class Foo:',
			'	def match( self, a: str, ln: i32 = compiler.caller_line() ) -> i32:',
			'		return ln',
			'',
			'	def match( self, a: bytes, ln: i32 = compiler.caller_line() ) -> i32:',
			'		return ln',
			'',
			'def main() -> i32:',
			'	f: Foo = Foo()',
			'	return f.match( "hi" )',
		])
		comp.import_code( code, filename = Path( '__test__.py' ))
		fn = comp._lower( disco.main )
		self.assertEqual( disco.errors.errors, [] )
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) and getattr( i.target, 'stem', None ) == 'match' ]
		self.assertEqual( len( calls ), 1 )
		self.assertEqual( calls[0].kwargs['ln'].value, 10 )

	def test_overload_resolved_default_referencing_callee_module_private_name( self ) -> None:
		# regression for a drive-by fix: the resolved-Overload-candidate
		# default-fill site used to skip module_context/scope_context
		# entirely (unlike the other two default-fill sites), so an ordinary
		# default referencing a name private to the callee's own module
		# would fail to resolve under the CALLER's context instead
		import tempfile
		with tempfile.TemporaryDirectory() as tmp:
			root = Path( tmp )
			( root / 'a.py' ).write_text( '\n'.join([
				'SPECIAL: i32 = 5', # never imported by __main__.py below
				'class Foo:',
				'	def match( self, x: str, pad: i32 = SPECIAL ) -> i32:',
				'		return pad',
				'',
				'	def match( self, x: bytes, pad: i32 = SPECIAL ) -> i32:',
				'		return pad',
			]), encoding = 'utf-8' )
			( root / '__main__.py' ).write_text( '\n'.join([
				'from a import Foo',
				'def main() -> i32:',
				'	f: Foo = Foo()',
				'	return f.match( "hi" )', # pad omitted
			]), encoding = 'utf-8' )
			disco = Discovery( paths = [ root, Path( discovery.__file__ ).parent / 'lib' ], import_builtins = True ) # str/bytes are builtins, not intrinsics
			compiler = Compiler( disco )
			compiler.import_file( root / '__main__.py' )
			compiler.run()
			self.assertEqual( disco.errors.errors, [] )

	# --- compiler.is_rc(T/x) -----------------------------------------------------
	# no dedicated tests existed for this intrinsic before - the value-argument
	# gap below (silently miscomputing instead of erroring) went unnoticed
	# for exactly that reason

	def test_compiler_is_rc_true_for_an_rcclass_type_argument( self ) -> None:
		code = '\n'.join([
			'class Foo: pass',
			'',
			'def main() -> None:',
			'	x: bool = compiler.is_rc( Foo )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		assigns = { getattr( i.dest, 'stem', None ): i.src for i in fn.instructions if isinstance( i, ir.Assign ) }
		self.assertIs( assigns['x'].value, True )

	def test_compiler_is_rc_false_for_a_scalar_type_argument( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	x: bool = compiler.is_rc( u32 )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		assigns = { getattr( i.dest, 'stem', None ): i.src for i in fn.instructions if isinstance( i, ir.Assign ) }
		self.assertIs( assigns['x'].value, False )

	def test_compiler_is_rc_typevar_is_rejected( self ) -> None:
		code = '\n'.join([
			'def foo[T]() -> None:',
			'	x: bool = compiler.is_rc( T )',
			'	return',
		])
		self._import( code )
		foo_fn = self.discovery.modules['__test__'].get_local( 'foo' )
		if foo_fn.resolve is not None:
			foo_fn.resolve()
		self.compiler._lower( foo_fn )
		self.assertIn( 'unbound generic type parameter', self.discovery.errors.errors[0] )

	def test_compiler_is_rc_of_a_local_rc_variable_is_true( self ) -> None:
		# regression test: before the value-argument fix, is_rc(x) resolved
		# a bare Name via _try_resolve_namespace same as sizeof(x) used to -
		# but unlike sizeof, it never checked "is this actually a Type",
		# so it silently checked isinstance(<the Variable object>, RCClass)
		# instead of the value's real type, which is always False - this
		# always folded to False regardless of x's real RC-ness, with no
		# compile error at all
		code = '\n'.join([
			'class Foo: pass',
			'',
			'def main() -> None:',
			'	f: Foo = Foo()',
			'	x: bool = compiler.is_rc( f )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		assigns = { getattr( i.dest, 'stem', None ): i.src for i in fn.instructions if isinstance( i, ir.Assign ) }
		self.assertIs( assigns['x'].value, True )

	def test_compiler_is_rc_of_a_non_rc_local_variable_is_false( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	v: u32 = 1',
			'	x: bool = compiler.is_rc( v )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		assigns = { getattr( i.dest, 'stem', None ): i.src for i in fn.instructions if isinstance( i, ir.Assign ) }
		self.assertIs( assigns['x'].value, False )

	def test_compiler_is_rc_of_self_is_self_escape_safe_before_construction_completes( self ) -> None:
		# same property as compiler.sizeof(self) - self is read only via
		# its own static type, never as an instruction operand, so this
		# compiles cleanly even before every required attribute is set
		code = '\n'.join([
			'class Bar:',
			'	a: i32',
			'',
			'	def __init__( self ) -> None:',
			'		x: bool = compiler.is_rc( self )',
			'		self.a = 1',
			'',
			'def main() -> None:',
			'	return',
		])
		mod = self._import( code )
		fn = self.compiler._lower( self._method( mod, 'Bar', '__init__' ))
		self.assertEqual( self.discovery.errors.errors, [] )
		assigns = { getattr( i.dest, 'stem', None ): i.src for i in fn.instructions if isinstance( i, ir.Assign ) }
		self.assertIs( assigns['x'].value, True )

	# --- match type(<Name>): case ConcreteClass(...): ... (generic monomorphization fold) ---
	# type_resolver.py's _try_fold_match_type - compile-time arm selection,
	# once a generic function's own type-parameter-typed value is
	# monomorphized to a concrete, non-union type. See PLAN_MATCH_TYPE_
	# MONOMORPHIZATION.md for the design this implements.

	def test_match_type_folds_to_the_matching_arm_with_no_runtime_branch_left( self ) -> None:
		code = '\n'.join([
			'def describe[T]( x: T ) -> i32:',
			'	match type( x ):',
			'		case i32( n ):',
			'			return n',
			'		case _:',
			'			return -1',
			'',
			'def main() -> i32:',
			'	a: i32 = 5',
			'	return describe( a )',
		])
		self._import( code )
		fn = self._lower_main()
		self.compiler._drain() # describe(a)'s own Specialization is only SCHEDULED while lowering main - draining is what actually resolves+lowers its own body
		self.assertEqual( self.discovery.errors.errors, [] )
		described = next( lf for lf in self.compiler.functions if 'describe' in lf.function.qualname )
		# a real compile-time arm selection, not a runtime tag check - no
		# comparison/branch instruction of any kind should be left behind
		self.assertFalse( any( isinstance( i, ( ir.Cmp, ir.JumpIfTrue, ir.JumpIfFalse )) for i in described.instructions ) )

	def test_match_type_falls_back_to_wildcard_for_an_uncovered_concrete_type( self ) -> None:
		code = '\n'.join([
			'def describe[T]( x: T ) -> i32:',
			'	match type( x ):',
			'		case i32( n ):',
			'			return n',
			'		case _:',
			'			return -1',
			'',
			'def main() -> i32:',
			'	b: bool = True',
			'	return describe( b )',
		])
		self._import( code )
		fn = self._lower_main()
		self.compiler._drain()
		self.assertEqual( self.discovery.errors.errors, [] )
		described = next( lf for lf in self.compiler.functions if 'describe' in lf.function.qualname )
		returns = [ i for i in described.instructions if isinstance( i, ir.Return ) ]
		self.assertEqual( len( returns ), 1 )
		self.assertEqual( returns[0].value.value, -1 ) # only the wildcard arm's own body survived - the i32 arm was pruned entirely for this (bool) instantiation

	def test_match_type_no_covering_arm_and_no_wildcard_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'def describe[T]( x: T ) -> i32:',
			'	match type( x ):',
			'		case i32( n ):',
			'			return n',
			'',
			'def main() -> i32:',
			'	f: f64 = 1.0',
			'	return describe( f )',
		])
		self._import( code )
		self._lower_main()
		self.compiler._drain()
		self.assertTrue( any( 'no arm covers' in e for e in self.discovery.errors.errors ), self.discovery.errors.errors )

	# --- compiler.refcount(x) ---------------------------------------------------

	def test_compiler_refcount_emits_refcount_instruction( self ) -> None:
		code = '\n'.join([
			'class Foo: pass',
			'',
			'def main() -> usize:',
			'	f: Foo = Foo()',
			'	return compiler.refcount( f )',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		refcounts = [ i for i in fn.instructions if isinstance( i, ir.RefCount ) ]
		self.assertEqual( len( refcounts ), 1 )
		usize = self.discovery.get_intrinsics()['usize']
		self.assertEqual( refcounts[0].dest.type, usize )
		f_cls = self.discovery.modules['__test__'].get_local( 'Foo' )
		self.assertIs( refcounts[0].value.type, f_cls )

	def test_compiler_refcount_on_a_generic_rcclass_specialization( self ) -> None:
		# regression test: a generic RCClass's own Specialization (Box[i32])
		# isn't an RCClass instance itself - unwrapping to its abstract
		# .base is required or a genuinely refcounted value is wrongly
		# rejected whenever its declared type happens to be a concrete
		# generic instantiation
		code = '\n'.join([
			'class Box[T]:',
			'	v: T',
			'',
			'def main() -> usize:',
			'	b: Box[i32] = Box( v = 1 )',
			'	return compiler.refcount( b )',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		refcounts = [ i for i in fn.instructions if isinstance( i, ir.RefCount ) ]
		self.assertEqual( len( refcounts ), 1 )

	def test_compiler_refcount_on_non_rc_value_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'def main() -> usize:',
			'	x: i32 = 1',
			'	return compiler.refcount( x )',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'reference-counted value', self.discovery.errors.errors[0] )

	def test_compiler_refcount_wrong_arg_count_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'class Foo: pass',
			'',
			'def main() -> usize:',
			'	f: Foo',
			'	return compiler.refcount()',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'takes exactly one argument', self.discovery.errors.errors[0] )

	# --- compiler.addrof(x) --------------------------------------------------

	def test_compiler_addrof_emits_addrof_instruction( self ) -> None:
		code = '\n'.join([
			'def main() -> Ptr[u32]:',
			'	written: u32 = 0',
			'	return compiler.addrof( written )',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		addrofs = [ i for i in fn.instructions if isinstance( i, ir.AddrOf ) ]
		self.assertEqual( len( addrofs ), 1 )
		u32 = self.discovery.get_intrinsics()['u32']
		ptr_cls = self.discovery.get_intrinsics()['Ptr']
		expected_ptr_type = self.discovery._get_or_create_specialization( ptr_cls, [ u32 ] )
		self.assertIs( addrofs[0].dest.type, expected_ptr_type )
		self.assertEqual( addrofs[0].value.type, u32 )

	def test_compiler_addrof_non_name_argument_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'def main() -> Ptr[u32]:',
			'	written: u32 = 0',
			'	return compiler.addrof( written + 1 )',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'bare local variable', self.discovery.errors.errors[0] )

	def test_compiler_addrof_wrong_arg_count_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'def main() -> Ptr[u32]:',
			'	written: u32 = 0',
			'	return compiler.addrof()',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'takes exactly one argument', self.discovery.errors.errors[0] )

	# --- compiler.atomic_*(ptr, ...) -----------------------------------------

	def test_compiler_atomic_load_emits_atomicload_instruction( self ) -> None:
		code = '\n'.join([
			'import sys',
			'def main() -> i32:',
			'	p: Ptr[i32] = sys.alloc[i32]( 1 )',
			'	return compiler.atomic_load( p )',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		loads = [ i for i in fn.instructions if isinstance( i, ir.AtomicLoad ) ]
		self.assertEqual( len( loads ), 1 )
		i32 = self.discovery.get_intrinsics()['i32']
		self.assertEqual( loads[0].dest.type, i32 )

	def test_compiler_atomic_store_emits_atomicstore_instruction( self ) -> None:
		code = '\n'.join([
			'import sys',
			'def main() -> None:',
			'	p: Ptr[i32] = sys.alloc[i32]( 1 )',
			'	compiler.atomic_store( p, 5 )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		stores = [ i for i in fn.instructions if isinstance( i, ir.AtomicStore ) ]
		self.assertEqual( len( stores ), 1 )
		self.assertEqual( stores[0].value.value, 5 )

	def test_compiler_atomic_add_emits_atomicrmw_add( self ) -> None:
		code = '\n'.join([
			'import sys',
			'def main() -> i32:',
			'	p: Ptr[i32] = sys.alloc[i32]( 1 )',
			'	return compiler.atomic_add( p, 1 )',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		rmws = [ i for i in fn.instructions if isinstance( i, ir.AtomicRMW ) ]
		self.assertEqual( len( rmws ), 1 )
		self.assertEqual( rmws[0].op, ir.AtomicRMWOp.ADD )

	def test_compiler_atomic_sub_and_exchange_use_distinct_ops( self ) -> None:
		code = '\n'.join([
			'import sys',
			'def main() -> i32:',
			'	p: Ptr[i32] = sys.alloc[i32]( 1 )',
			'	a: i32 = compiler.atomic_sub( p, 1 )',
			'	b: i32 = compiler.atomic_exchange( p, 2 )',
			'	return a',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		rmws = [ i for i in fn.instructions if isinstance( i, ir.AtomicRMW ) ]
		self.assertEqual( [ r.op for r in rmws ], [ ir.AtomicRMWOp.SUB, ir.AtomicRMWOp.EXCHANGE ] )

	def test_compiler_atomic_compare_exchange_emits_instruction( self ) -> None:
		code = '\n'.join([
			'import sys',
			'def main() -> bool:',
			'	p: Ptr[i32] = sys.alloc[i32]( 1 )',
			'	expected: Ptr[i32] = sys.alloc[i32]( 1 )',
			'	return compiler.atomic_compare_exchange( p, expected, 5 )',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		cas = [ i for i in fn.instructions if isinstance( i, ir.AtomicCompareExchange ) ]
		self.assertEqual( len( cas ), 1 )
		bool_cls = self.discovery.get_intrinsics()['bool']
		self.assertEqual( cas[0].dest.type, bool_cls )

	def test_compiler_atomic_on_non_pointer_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'def main() -> i32:',
			'	x: i32 = 1',
			'	return compiler.atomic_load( x )',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'must be Ptr[T]', self.discovery.errors.errors[0] )

	def test_compiler_atomic_on_rc_pointee_is_a_compile_error( self ) -> None:
		# the lock-free-RC rabbit hole this deliberately stays out of - see
		# the plan's own Context section
		code = '\n'.join([
			'class Foo: pass',
			'import sys',
			'def main() -> None:',
			'	p: Ptr[Foo] = sys.alloc[Foo]( 1 )',
			'	compiler.atomic_load( p )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'plain scalar', self.discovery.errors.errors[0] )

	def test_compiler_atomic_load_wrong_arg_count_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'def main() -> i32:',
			'	return compiler.atomic_load()',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'takes exactly one argument', self.discovery.errors.errors[0] )

	def test_compiler_incref_emits_incref_instruction( self ) -> None:
		code = '\n'.join([
			'class Foo: pass',
			'',
			'def main() -> None:',
			'	f: Foo = Foo()',
			'	compiler.incref( f )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertIn( 'Incref', kinds )

	def test_compiler_incref_wrong_arg_count_is_rejected( self ) -> None:
		code = '\n'.join([
			'class Foo: pass',
			'',
			'def main() -> None:',
			'	f: Foo',
			'	compiler.incref()',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'takes exactly one argument', self.discovery.errors.errors[0] )

	def test_compiler_incref_non_rc_is_rejected( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	x: i32 = 1',
			'	compiler.incref( x )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'reference-counted value', self.discovery.errors.errors[0] )

	# --- loops (while / for / break / continue) -----------------------------

	def test_while_shape( self ) -> None:
		code = '\n'.join([
			'def main( a: bool ) -> None:',
			'	while a:',
			'		x: i32 = 1',
			'	return',
		])
		self._import( code )
		i32 = self.discovery.get_intrinsics()['i32']
		none_type = self.discovery.get_none_type()
		if self.discovery.main.resolve is not None:
			self.discovery.main.resolve()
		a = self.discovery.main.parameters[0]
		x = Variable( stem = 'x', qualname = 'main.x', file = Path( '__test__.py' ), line = 3, type = i32 )
		fn = self._lower_main()
		self._assert_ir( fn, [
			ir.FuncStart( name = 'main', params = [ a ], return_type = none_type ),
			ir.Label( name = '__while_start_0__' ),
			ir.JumpIfFalse( cond = a, target = '__while_end_1__' ),
			ir.Assign( dest = x, src = ir.Const( type = i32, value = 1 )),
			ir.Jump( target = '__while_start_0__' ),
			ir.Label( name = '__while_end_1__' ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_loop_promotes_borrowed_param_mixed_with_owned_reassignment( self ) -> None:
		# regression test: reassigning a BORROWED parameter to either a
		# borrowed alias or a freshly owned value inside a for-loop body used
		# to hard-error ("... is in an indeterminate state across loop
		# iterations") - the loop is lowered exactly once and reused via the
		# back edge, so its entry state (BORROWED, from the parameter) never
		# matched the back-edge state (OWNED, from merge_if's own if/else
		# reconciliation of the two reassignment branches) - see
		# _lower_loop_body_with_ownership_retry's own docstring for the fix
		# (found via a real repro, grap.py's `r_filespec`/`path` handling)
		code = '\n'.join([
			'class Foo:',
			'	pass',
			'',
			'def main( path: Foo, other: Foo, flag: bool ) -> None:',
			'	items: list[Foo] = [ other ]',
			'	for item in items:',
			'		if flag:',
			'			path = other',
			'		else:',
			'			path = Foo()',
			'	return',
		])
		self.discovery.import_name( 'builtins' )
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_loop_promotion_survives_nested_loop_with_its_own_defer( self ) -> None:
		# regression test: the SAME promotion above, but with the mismatched
		# for-loop nested inside an outer while-loop that itself needs its
		# own retry (path's entry state, BORROWED, is only established
		# before the OUTER loop - the for-loop's own successful promotion
		# still leaves the while-loop's back edge disagreeing the same way).
		# The outer retry's rollback used to be blocked by a defer entry any
		# `for x in <a fresh iterable>:` loop always registers (to release
		# its own iterator) - ordinary restore() deliberately keeps a defer
		# entry alive past a loop's own teardown (it must still fire at the
		# function's real epilogue), but that's wrong for a FAILED retry
		# attempt about to be fully re-lowered from scratch - see
		# cfg.py's hard_restore() docstring
		code = '\n'.join([
			'class Foo:',
			'	pass',
			'',
			'def main( path: Foo, other: Foo, flag: bool ) -> None:',
			'	items: list[Foo] = [ other ]',
			'	i: usize = 0',
			'	while i < 3:',
			'		for item in items:',
			'			if flag:',
			'				path = other',
			'			else:',
			'				path = Foo()',
			'		with compiler.wrap_arithmetic:',
			'			i = i + 1',
			'	return',
		])
		self.discovery.import_name( 'builtins' )
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

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
			'	a: bool = True',
			'	b: bool = True',
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

	def test_for_target_must_be_a_plain_name_error_is_a_single_line( self ) -> None:
		# regression: this diagnostic used to unparse the WHOLE ast.For node
		# (target + entire body) instead of just the target - a real repro
		# (a multi-statement for-loop body) produced a multi-line error
		# message that buried the actual problem (an unsupported target)
		# under the loop's own unrelated body text
		code = '\n'.join([
			'def main() -> None:',
			'	xs: i32',
			'	for xs[0] in range( 3 ):',
			'		xs = 1',
			'		xs = 2',
			'	return',
		])
		self._import( code )
		self._lower_main()
		error = self.discovery.errors.errors[0]
		self.assertIn( 'for loop target must be a plain name', error )
		self.assertNotIn( '\n', error )

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
		self.discovery.import_name( 'builtins' ) # range()'s own hidden bound check (i < count) is now an ordinary usize.__lt__ dunder call
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertEqual( kinds.count( 'Label' ), 2 ) # start, end - continue_label omitted, the body never uses `continue`
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
		self.discovery.import_name( 'builtins' ) # range()'s own hidden bound check is now an ordinary usize.__lt__ dunder call
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

	def test_for_over_iterable_conformer_calls_iter_once( self ) -> None:
		# for v in <obj>: now strictly requires Iterator[T]/Iterable[T]
		# conformance (no more __len__/__getitem__ duck-typing) - an
		# Iterable[T] conformer's own __iter__() is called exactly once, up
		# front, and the resulting generator's __next__() drives the loop -
		# not a direct __getitem__(index) walk any more (that whole
		# mechanism, _lower_for_over_indexable, is gone - the per-iteration
		# bounds-checked-index bind it used to build, and the Unwrap-panic
		# consumption on it, both moved into _sequence_iter's own real
		# generator body, lib/builtins/__init__.py - shared by every
		# Sequence[T] conformer, not reimplemented per for-loop any more)
		code = '\n'.join([
			'class Box( Sequence[i32], Iterable[i32] ):',
			'	_len: usize',
			'',
			'	def __init__( self, n: usize ) -> None:',
			'		self._len = n',
			'',
			'	def __len__( self ) -> usize:',
			'		return self._len',
			'',
			'	def __getitem__( self, i: usize ) -> Result[i32, IndexError]:',
			'		return Result.Ok( 1 )',
			'',
			'	def __iter__( self ) -> Generator[i32, StopIteration]:',
			'		return _sequence_iter( self )',
			'',
			'def main( b: Box ) -> None:',
			'	for v in b:',
			'		x: i32 = v',
			'	return',
		])
		self.discovery.import_name( 'builtins' )
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		# exactly 2 Call INSTRUCTIONS in main's own static IR - one to
		# Box.__iter__() up front, one to the generator's own __next__()
		# (executed repeatedly via the loop's back-edge goto, but still just
		# ONE static Call instruction here - the generator's own internal
		# state-machine logic for what __next__ actually DOES lives in a
		# separate, synthesized function, not inlined into main)
		call_targets = [ instr.target.stem for instr in fn.instructions if type( instr ).__name__ == 'Call' ]
		self.assertEqual( call_targets, [ '__iter__', '__next__' ] )

	def test_for_missing_iterator_or_iterable_conformance_is_rejected( self ) -> None:
		# strict protocol dispatch (confirmed with the user) - a type with
		# matching method NAMES but no DECLARED Iterator[T]/Iterable[T]
		# conformance is rejected outright, not silently duck-typed
		code = '\n'.join([
			'class Box:',
			'	pass',
			'',
			'def main( b: Box ) -> None:',
			'	for v in b:',
			'		pass',
			'	return',
		])
		self.discovery.import_name( 'builtins' ) # Iterator/Iterable themselves live there - any for-loop needs to find them to even attempt the check
		self._import( code )
		self._lower_main()
		self.assertIn( 'for loop requires an IteratorProtocol[T] or Iterable[T] conformer', self.discovery.errors.errors[0] )

	def test_for_over_iterable_conformer_fallible_getitem_does_not_require_result_return( self ) -> None:
		# the actual bug this fixes (originally, before the strict-protocol
		# redesign): a plain `-> i32` main (no Result in sight) iterating an
		# ordinary Sequence[T] conformer used to fail to compile, demanding
		# `main` return Result[_,IndexError] to propagate an error the
		# loop's own bounds check already makes unreachable - that specific
		# consumption now lives inside _sequence_iter's own generator body
		# (a real .unwrap() call with a panic message, not or_return()), but
		# the end-to-end behavior (no Result-returning main required) still
		# needs guarding directly, real repro: `for arg in sys.argv[1:]:
		# print(arg)` inside a plain `-> i32` main
		code = '\n'.join([
			'class Box( Sequence[i32], Iterable[i32] ):',
			'	_len: usize',
			'',
			'	def __init__( self, n: usize ) -> None:',
			'		self._len = n',
			'',
			'	def __len__( self ) -> usize:',
			'		return self._len',
			'',
			'	def __getitem__( self, i: usize ) -> Result[i32, IndexError]:',
			'		return Result.Ok( 1 )',
			'',
			'	def __iter__( self ) -> Generator[i32, StopIteration]:',
			'		return _sequence_iter( self )',
			'',
			'def main( b: Box ) -> i32:',
			'	for v in b:',
			'		x: i32 = v',
			'	return 0',
		])
		self.discovery.import_name( 'builtins' )
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_for_over_indexable_rc_element_decref_stays_inside_loop_body( self ) -> None:
		# a real, confirmed bug found while fixing the two tests above: the
		# per-iteration bind (`bind = ast.Assign(...)`) used to be lowered
		# via a bare self._stmt_Assign(bind) call instead of self._lower_stmt
		# (bind) - skipping _lower_stmt's own pending-temps save/reset/flush
		# wrapper entirely. For an RC-typed __getitem__ payload, the raw
		# Result temp behind node.target then leaked into the ENCLOSING
		# ast.For statement's own pending-temps list instead of getting its
		# own per-iteration flush, and was decref'd exactly ONCE, after the
		# whole loop - reading UNINITIALIZED stack memory as an
		# ObjectHeader* whenever the loop body never ran at all (a real,
		# confirmed crash via `for arg in sys.argv[1:]: print(arg)` with no
		# extra command-line arguments - empty slice, zero iterations).
		#
		# Needs the REAL builtins Result/slice[T] (not a synthetic @cstruct
		# stand-in like the two tests above use) - confirmed empirically
		# that a synthetic, non-union Result shim doesn't reproduce this at
		# all (no separate Decref of the union's own v_Ok arm exists for
		# it in the first place, unlike the real tagged-union Result). A
		# real-compile-and-run reproduction is similarly unreliable (an
		# uninitialized-memory read/an extra decref on a still-referenced
		# object is UB, and doesn't reliably crash or show a wrong refcount
		# under every build configuration/compiler - confirmed empirically
		# that a real repro crashes when built via mpy.py's own CLI but not
		# when built through this test suite's own compile-and-run harness).
		# This checks the actual INVARIANT directly instead: the Result's
		# own v_Ok-arm Decref must appear BEFORE the loop's back-edge Jump
		# (i.e. genuinely inside the loop body, flushed every iteration),
		# not after it.
		code = '\n'.join([
			'def main( xs: list[str] ) -> i32:',
			'	ys = xs[1:]',
			'	for y in ys:',
			'		pass',
			'	return 0',
		])
		self.discovery.import_name( 'builtins' )
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		back_edge_indices = [
			i for i, instr in enumerate( fn.instructions )
			if type( instr ).__name__ == 'Jump' and instr.target.startswith( '__for_start_' )
		]
		self.assertEqual( len( back_edge_indices ), 1, 'expected exactly one for-loop back-edge Jump' )
		back_edge = back_edge_indices[0]
		# the leaked instruction's own shape (matches the real bug's C
		# output exactly): GetAttr(...,attr='data') -> GetAttr(...,
		# attr='v_Ok') -> Decref(value=<that v_Ok temp>) - a real,
		# str-payload Result union being torn down. Filtering on
		# `attr == 'v_Ok'` distinguishes this from the loop's OWN `y`
		# variable's own, always-correctly-scoped Decref (index 70 in a
		# real dump of this exact program) and from slice[str]/list[str]'s
		# own unrelated cleanup Decrefs in the function epilogue.
		ok_temp_ids = {
			instr.dest.id for instr in fn.instructions
			if type( instr ).__name__ == 'GetAttr' and instr.attr == 'v_Ok'
		}
		self.assertTrue( ok_temp_ids, 'expected at least one v_Ok GetAttr extracting the Result payload' )
		v_ok_decref_indices = [
			i for i, instr in enumerate( fn.instructions )
			if type( instr ).__name__ == 'Decref'
			and getattr( instr.value, 'id', None ) in ok_temp_ids
		]
		self.assertTrue( v_ok_decref_indices, 'expected a Decref of the Result\'s own v_Ok payload' )
		self.assertTrue(
			all( d < back_edge for d in v_ok_decref_indices ),
			f'the Result\'s v_Ok Decref landed AFTER the loop back-edge Jump (index {back_edge}) - '
			f'v_ok_decref_indices={v_ok_decref_indices} - this is the leaked-post-loop-flush bug: the '
			f'element temp only gets decref\'d once (after the loop), not per iteration',
		)

	def test_subscript_with_getitem_returning_result_is_now_auto_consumed( self ) -> None:
		# obj[i] is plain sugar for obj.__getitem__(i), nothing more - when
		# __getitem__ is fallible the caller gets the raw Result[T,E] back.
		# PLAN_CHECKED_ARITHMETIC_GAP.md's whole point: `v: i32 = b[i]` used
		# to be a hard type-mismatch (zero sugar at all) - now the general
		# auto-or_throw() rule's case 2 (Result flowing into a context
		# wanting its own T directly) picks it up here too, exactly like a
		# declared-Result-typed checked-arithmetic target already does
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
			'	return Result( x = v )',
		])
		self._import( code )
		foo_fn = self.discovery.modules['__test__'].get_local( 'foo' )
		if foo_fn.resolve is not None:
			foo_fn.resolve()
		fn = self.compiler._lower( foo_fn )
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertIn( 'OrThrow', kinds )

	def test_subscript_with_getitem_returning_result_consumed_explicitly_still_works( self ) -> None:
		# the explicit-consumption escape hatch: b[i].or_return() still
		# works exactly like get_result().or_return() already does, since
		# b[i] now hands back the raw Result for the user to consume
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
			'	v: i32 = b[i].or_return()',
			'	return Result( x = v )',
		])
		self._import( code )
		foo_fn = self.discovery.modules['__test__'].get_local( 'foo' )
		if foo_fn.resolve is not None:
			foo_fn.resolve()
		fn = self.compiler._lower( foo_fn )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertIn( 'OrReturn', kinds )
		getitem_errors = [ e for e in self.discovery.errors.errors if 'b[i]' in e or '__getitem__' in e ]
		self.assertEqual( getitem_errors, [] )

	def test_slice_subscript_with_getitem_returning_result_is_now_auto_consumed( self ) -> None:
		# x[a:b] is the same plain sugar as x[i] - _lower_slice_subscript's
		# own Result now auto-consumes too, mirroring _expr_Subscript's own
		# identical case-2 auto-or_throw() (see
		# test_subscript_with_getitem_returning_result_is_now_auto_consumed's
		# own comment for the full story)
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
			'	def __getitem__( self, s: slice ) -> Result[i32,MyError]:',
			'		return Result.__allocate__( x = self.y )',
			'',
			'def foo( b: Box ) -> Result[i32,MyError]:',
			'	v: i32 = b[0:1]',
			'	return Result( x = v )',
		])
		self.discovery.import_name( 'builtins' )
		self._import( code )
		foo_fn = self.discovery.modules['__test__'].get_local( 'foo' )
		if foo_fn.resolve is not None:
			foo_fn.resolve()
		fn = self.compiler._lower( foo_fn )
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertIn( 'OrThrow', kinds )

	def test_subscript_with_getitem_returning_non_result_two_arg_generic_is_not_consumed( self ) -> None:
		# _maybe_consume_result (Stage 4: now routed through _result_shape)
		# must check the Specialization's own base identity against the
		# real Result class, not just "some Specialization with 2 args" -
		# Pair[T,U] here has the same shape (2 type args) as Result[T,E]
		# but is a completely different class, and must pass through
		# unconsumed rather than wrongly being treated as fallible
		code = '\n'.join([
			'class MyError: pass',
			'',
			'@cstruct',
			'class Result[T,E]:',
			'	x: T',
			'',
			'@cstruct',
			'class Pair[T,U]:',
			'	x: T',
			'	y: U',
			'',
			'@cstruct',
			'class Box:',
			'	y: i32',
			'',
			'	def __getitem__( self, i: usize ) -> Pair[i32,MyError]:',
			'		return Pair.__allocate__( x = self.y, y = MyError() )',
			'',
			'def foo( b: Box, i: usize ) -> Pair[i32,MyError]:',
			'	v: Pair[i32,MyError] = b[i]',
			'	return v',
		])
		self._import( code )
		foo_fn = self.discovery.modules['__test__'].get_local( 'foo' )
		if foo_fn.resolve is not None:
			foo_fn.resolve()
		fn = self.compiler._lower( foo_fn )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertNotIn( 'OrReturn', kinds )
		self.assertNotIn( 'OrJump', kinds )
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_subscript_assign_with_setitem_dispatches_to_method( self ) -> None:
		# obj[i] = v used to unconditionally emit the flat SetItem opcode,
		# even when obj.type declares a real __setitem__ - meaning a
		# user-defined __setitem__ could never actually be invoked through
		# assignment syntax. Now mirrors _expr_Subscript's own __getitem__
		# resolution: a real __setitem__ is called like any other method
		code = '\n'.join([
			'@cstruct',
			'class Box:',
			'	y: i32',
			'',
			'	def __setitem__( self, i: usize, v: i32 ) -> None:',
			'		self.y = v',
			'',
			'def main( b: Box ) -> None:',
			'	b[0] = 5',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertIn( 'Call', kinds )
		self.assertNotIn( 'SetItem', kinds )
		call = next( instr for instr in fn.instructions if isinstance( instr, ir.Call ))
		self.assertEqual( call.target.qualname, '__test__.Box.__setitem__' )

	def test_subscript_assign_without_setitem_falls_back_to_raw_setitem( self ) -> None:
		# no __setitem__ declared (raw pointers, or any other type that
		# doesn't define subscript assignment as a method) - must keep
		# emitting the flat SetItem opcode exactly as before this fix
		code = '\n'.join([
			'import sys',
			'',
			'@cstruct',
			'class Point:',
			'	x: i32',
			'',
			'def main() -> None:',
			'	p: Ptr[Point] = sys.alloc[Point]( 1 )',
			'	p[0] = Point( x = 7 )',
			'	sys.free( p )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertIn( 'SetItem', kinds )
		self.assertEqual( [ type( i ).__name__ for i in fn.instructions if isinstance( i, ir.Call ) and 'setitem' in i.target.qualname.lower() ], [] )

	def test_subscript_assign_with_setitem_resolves_and_consumes_result( self ) -> None:
		# obj[i] = v is sugar for obj.__setitem__(i, v).or_return() whenever
		# __setitem__ can fail - mirrors test_subscript_with_getitem_
		# resolves_and_consumes_result above, for the write side
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
			'	def __setitem__( self, i: usize, v: i32 ) -> Result[None,MyError]:',
			'		self.y = v',
			'		return Result.__allocate__( x = None )',
			'',
			'def foo( b: Box, i: usize ) -> Result[None,MyError]:',
			'	b[i] = 5',
			'	return Result.Err( MyError() )',
		])
		self._import( code )
		foo_fn = self.discovery.modules['__test__'].get_local( 'foo' )
		if foo_fn.resolve is not None:
			foo_fn.resolve()
		fn = self.compiler._lower( foo_fn )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		# auto-inserted (no enclosing try here) - degrades to exactly
		# or_return()'s own semantics, same general auto-or_throw() rule
		# every other auto-consumption site now shares (see _auto_or_throw)
		self.assertIn( 'OrThrow', kinds )
		setitem_errors = [ e for e in self.discovery.errors.errors if 'b[i]' in e or '__setitem__' in e ]
		self.assertEqual( setitem_errors, [] )

	def test_if_body_recovery_boundary_does_not_stop_orelse( self ) -> None:
		# one bad statement inside the if-body doesn't prevent orelse (or
		# anything after the if) from still being lowered - same recovery
		# boundary as everywhere else
		code = '\n'.join([
			'def main() -> None:',
			'	a: bool = True',
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
		self.assertEqual( kinds.count( 'Assign' ), 3 ) # a's own init, b and c, all still lowered

	# --- calls ---------------------------------------------------------------

	def test_call_to_function_with_broken_parameters_fails_cleanly( self ) -> None:
		# helper's own parameter resolution fails (missing annotation on
		# 'x'), which raises CompileError partway through discovery.py's
		# _make_function_resolver body - before fn.parameters is ever
		# assigned, so it's left at its None default. discovery.py's
		# _resolve_guarded swallows that CompileError so helper's own
		# broken definition is reported once, not re-raised, AND marks
		# helper.broken - which used to leave any CALLER of helper
		# crashing with an unhandled TypeError ('NoneType' object is not
		# iterable) inside _match_call_args, then (once that was first
		# fixed) reporting a second, redundant "could not be resolved"
		# message at the call site; now it's a silent
		# RedundantCompilationError instead (see mpy_types.Name.broken) -
		# the real error was already recorded once, at helper's own
		# definition
		code = '\n'.join([
			'def helper( x ) -> i32:',
			'	return x',
			'',
			'def main() -> None:',
			'	helper( 5 )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( len( self.discovery.errors.errors ), 1 )
		self.assertIn( "helper parameter 'x' has no type annotation", self.discovery.errors.errors[0] )

	def test_broken_local_assignment_does_not_cascade_to_later_read( self ) -> None:
		# x's own initializer fails to compile (undefined_fn doesn't exist) -
		# the bare-assignment path used to skip registering x entirely when
		# that happened, so the later `return x` reported its OWN, spurious
		# "name 'x' is not defined" on top of the real error. Now x gets
		# registered BROKEN on failure (see lowering.py's _declare_local),
		# so the later read raises a silent RedundantCompilationError
		# instead (see mpy_types.Name.broken).
		code = '\n'.join([
			'def main() -> i32:',
			'	x = undefined_fn()',
			'	return x',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( len( self.discovery.errors.errors ), 1 )
		self.assertIn( "name 'undefined_fn' is not defined", self.discovery.errors.errors[0] )

	def test_broken_annotated_local_does_not_cascade_to_later_read( self ) -> None:
		# _stmt_AnnAssign registers x BEFORE lowering its initializer
		# (unlike plain Assign) - a failed initializer used to leave a
		# plausible-looking, but never actually initialized, Variable
		# sitting in scope; the later `return x` would proceed as if it
		# had a real value instead of raising cleanly.
		code = '\n'.join([
			'def main() -> i32:',
			'	x: i32 = undefined_fn()',
			'	return x',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( len( self.discovery.errors.errors ), 1 )
		self.assertIn( "name 'undefined_fn' is not defined", self.discovery.errors.errors[0] )

	def test_broken_local_assignment_heals_on_redeclaration( self ) -> None:
		# a later, genuinely fresh assignment to the same name must NOT be
		# blocked by the earlier broken one - free to redeclare cleanly,
		# same as a first assignment, since nothing usable was ever
		# produced for the broken attempt.
		code = '\n'.join([
			'def main() -> i32:',
			'	x = undefined_fn()',
			'	x = 5',
			'	return x',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( len( self.discovery.errors.errors ), 1 )
		self.assertIn( "name 'undefined_fn' is not defined", self.discovery.errors.errors[0] )
		x_var = self.discovery.main.get_local( 'x' )
		self.assertIsNotNone( x_var )
		self.assertFalse( x_var.broken )

	def test_augassign_to_undeclared_name_still_fails_cleanly_not_cascading( self ) -> None:
		# _stmt_AugAssign desugars `x += 1` into `x = x + 1`, threaded
		# through _stmt_Assign's own "no prior declaration" branch - the
		# synthesized RHS reads x itself, so x must NOT be registered
		# until AFTER that RHS is lowered (see _declare_local's own
		# comment), or the self-read would find a freshly-declared, empty
		# x instead of correctly failing "not defined". Guards against
		# regressing that ordering while fixing the broken-local gap above.
		code = '\n'.join([
			'def main() -> None:',
			'	x += 1',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( len( self.discovery.errors.errors ), 1 )
		self.assertIn( "name 'x' is not defined", self.discovery.errors.errors[0] )

	def test_broken_base_class_used_later_does_not_cascade( self ) -> None:
		# Baz's base-class resolution fails eagerly, before Baz.resolve is
		# even assigned (see discovery.py's _parse_ClassDef_RCClass) -
		# without marking Baz broken there, Baz.resolve stays at its
		# dataclass default of None, indistinguishable from "already
		# resolved fine" to the later construction call below.
		code = '\n'.join([
			'@cstruct',
			'class Bar:',
			'	pass',
			'',
			'class Baz( Bar ):',
			'	pass',
			'',
			'def main() -> None:',
			'	b = Baz()',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( len( self.discovery.errors.errors ), 1 )
		self.assertIn( 'cannot subclass', self.discovery.errors.errors[0] )

	def test_broken_generic_function_referenced_later_does_not_cascade( self ) -> None:
		# alloc's own parameter resolution fails - a generic function, so
		# the later alloc[u32](...) call site goes through monomorphize.py's
		# monomorphized_function, which reads base.parameters directly.
		# Without checking base.broken there, a fully-failed base (whose
		# .parameters stays None) would silently substitute ZERO parameters
		# into the monomorphized copy instead of failing at all.
		code = '\n'.join([
			'def alloc[T]( count ) -> usize:',
			'	with compiler.wrap_arithmetic:',
			'		return count * compiler.sizeof( T )',
			'',
			'def main() -> None:',
			'	x: usize = alloc[u32]( 10 )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( len( self.discovery.errors.errors ), 1 )
		self.assertIn( "alloc parameter 'count' has no type annotation", self.discovery.errors.errors[0] )

	def test_broken_unannotated_global_referenced_later_does_not_double_report( self ) -> None:
		# an unannotated global's init expression gets a full discovery-
		# stage type-inference visit (discovery.py's _make_value_resolver,
		# needed to infer G's own .type) IN ADDITION TO its ordinary
		# lowering-stage visit (lowering.py's lower_global, via
		# Compiler._lower's Variable branch) - before this fix, both
		# independently visited the same undefined_fn() call and both
		# reported "not defined", producing the same message twice for one
		# real problem. Compiler._lower now checks G.broken (set by the
		# first, discovery-stage failure) and raises a silent
		# RedundantCompilationError instead of ever reaching the second,
		# redundant visit.
		code = '\n'.join([
			'G = undefined_fn()',
			'',
			'def main() -> i32:',
			'	g: i32 = G',
			'	return g',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( len( self.discovery.errors.errors ), 1 )
		self.assertIn( "name 'undefined_fn' is not defined", self.discovery.errors.errors[0] )

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
			'def main( f: Foo ) -> None:',
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
		if self.discovery.main.resolve is not None:
			self.discovery.main.resolve()
		f = self.discovery.main.parameters[0]

		fn = self._lower_main()
		self._assert_ir( fn, [
			ir.FuncStart( name = 'main', params = [ f ], return_type = none_type ),
			ir.Call( dest = None, target = bump_fn, receiver = f, args = [], kwargs = {} ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_property_access_lowers_to_method_call( self ) -> None:
		# obj.attr (no call parens) for an @property getter calls the
		# underlying method and uses its result - not a GetAttr (there's no
		# real field named 'bar'), and not a bound-method closure the way an
		# ordinary (non-property) method-as-value would build
		code = '\n'.join([
			'class Foo:',
			'	@property',
			'	def bar( self ) -> i32:',
			'		return 1',
			'',
			'def main() -> None:',
			'	f: Foo = Foo()',
			'	x: i32 = f.bar',
			'	return',
		])
		mod = self._import( code )
		foo_cls = mod.get_local( 'Foo' )
		if foo_cls.resolve is not None:
			foo_cls.resolve()
		bar_fn = foo_cls.get_local( 'bar' )
		if bar_fn.resolve is not None:
			bar_fn.resolve()

		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) and i.target is bar_fn ]
		self.assertEqual( len( calls ), 1 )
		self.assertIsNotNone( calls[0].receiver )
		self.assertFalse( any( isinstance( i, ir.GetAttr ) and i.attr == 'bar' for i in fn.instructions ))

	def test_property_result_type_mismatch_is_rejected( self ) -> None:
		# regression guard: _expr_Attribute's is_property branch used to hand
		# expected_type straight through to _lower_method_call's own
		# result_type param, which TYPES the call's dest directly rather than
		# checking anything - a declared local type that didn't actually
		# match the getter's real return type silently compiled into a
		# mismatched C struct assignment (confirmed via a real repro: a
		# list[tuple[i32,i32]] local reading a list[tuple[isize,isize]]-
		# returning property compiled clean, then read garbage values at
		# runtime - the two tuple element types have different widths, so
		# the generated C was a genuinely wrong pointer-type assignment, not
		# just numerically imprecise). An ordinary (non-property) call
		# already rejects this same shape; a property must too.
		code = '\n'.join([
			'class Foo:',
			'	@property',
			'	def bar( self ) -> i64:',
			'		return 1',
			'',
			'def main() -> None:',
			'	f: Foo = Foo()',
			'	x: i32 = f.bar',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertNotEqual( self.discovery.errors.errors, [] )
		self.assertTrue( any( 'expected' in str( e ) and 'i32' in str( e ) and 'i64' in str( e ) for e in self.discovery.errors.errors ))

	def test_plain_field_still_lowers_to_getattr( self ) -> None:
		# regression guard alongside the property test above: a plain
		# (non-@property) field access must keep using GetAttr, not get
		# swept into the new property-call branch in _expr_Attribute
		code = '\n'.join([
			'class Foo:',
			'	bar: i32',
			'',
			'def main() -> None:',
			'	f: Foo = Foo( bar = 1 )',
			'	x: i32 = f.bar',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertFalse( any( isinstance( i, ir.Call ) for i in fn.instructions ))
		self.assertTrue( any( isinstance( i, ir.GetAttr ) and i.attr == 'bar' for i in fn.instructions ))

	# --- Callable[...] function references / indirect calls (PLAN_CALLABLE.md) ---

	def test_bare_function_reference_lowers_to_function_ref( self ) -> None:
		# a plain, receiver-less function name used as a VALUE (not called)
		# lowers to ir.FunctionRef, typed Ptr[Callable[[ArgTypes],RetType]]
		# - this used to fail outright ('not a value, cannot use it as an
		# expression')
		code = '\n'.join([
			'def add_one( x: i32 ) -> i32:',
			'	with compiler.wrap_arithmetic:',
			'		return x + 1',
			'',
			'def main() -> None:',
			'	f: Ptr[Callable[[i32],i32]] = add_one',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		assign = next( instr for instr in fn.instructions if isinstance( instr, ir.Assign ))
		self.assertIsInstance( assign.src, ir.FunctionRef )
		self.assertEqual( assign.src.fn.qualname, '__test__.add_one' )
		ptr_type = assign.src.type
		self.assertIsInstance( ptr_type, Specialization )
		self.assertEqual( ptr_type.base.stem, 'Ptr' )
		fn_type = ptr_type.args[0]
		self.assertEqual( [ t.stem for t in fn_type.arg_types ], [ 'i32' ] )
		self.assertEqual( fn_type.return_type.stem, 'i32' )

	def test_staticmethod_reference_lowers_to_function_ref( self ) -> None:
		# a @staticmethod has no receiver either - same as a free function.
		# Referenced bare/unqualified from a SIBLING method's own body (the
		# shape dict[K,V]'s own generated code actually needs - a
		# monomorphized dict[K,V] method referencing its own @staticmethod
		# helper) - a bare Name lookup from inside a method body already
		# searches the enclosing class's own scope (confirmed separately;
		# not new behavior from this feature). Qualified Class.method
		# attribute access is a DIFFERENT codepath (_expr_Attribute, not
		# _expr_Name) that this pass doesn't touch at all - out of scope,
		# not needed by dict[K,V].
		code = '\n'.join([
			'class Box:',
			'	@staticmethod',
			'	def add_one( x: i32 ) -> i32:',
			'		with compiler.wrap_arithmetic:',
			'			return x + 1',
			'',
			'	def use_it( self ) -> None:',
			'		f: Ptr[Callable[[i32],i32]] = add_one',
			'		return',
		])
		self._import( code )
		box_cls = self.discovery.modules['__test__'].get_local( 'Box' )
		if box_cls.resolve is not None:
			box_cls.resolve()
		use_it_fn = box_cls.get_local( 'use_it' )
		if use_it_fn.resolve is not None:
			use_it_fn.resolve()
		lf = self.compiler._lower( use_it_fn )
		self.assertEqual( self.discovery.errors.errors, [] )
		assign = next( instr for instr in lf.instructions if isinstance( instr, ir.Assign ))
		self.assertIsInstance( assign.src, ir.FunctionRef )
		self.assertEqual( assign.src.fn.qualname, '__test__.Box.add_one' )

	def test_instance_method_reference_is_rejected( self ) -> None:
		# a bound instance method has an implicit receiver with nowhere to
		# go in a raw C function pointer - that's a closure, out of scope
		# (see PLAN_CALLABLE.md's own "deferred" list). Same bare-reference-
		# from-a-sibling-method shape as the staticmethod test above -
		# get() is never actually CALLED anywhere, only referenced as a
		# value, so this must directly lower use_it (not rely on
		# reachability from main(), which would never reach this code at
		# all and silently prove nothing)
		code = '\n'.join([
			'class Box:',
			'	v: i32',
			'	def get( self ) -> i32:',
			'		return self.v',
			'',
			'	def use_it( self ) -> None:',
			'		f: Ptr[Callable[[],i32]] = get',
			'		return',
		])
		self._import( code )
		box_cls = self.discovery.modules['__test__'].get_local( 'Box' )
		if box_cls.resolve is not None:
			box_cls.resolve()
		use_it_fn = box_cls.get_local( 'use_it' )
		if use_it_fn.resolve is not None:
			use_it_fn.resolve()
		self.compiler._lower( use_it_fn )
		self.assertIn( 'no receiver to bind', self.discovery.errors.errors[0] )

	def test_generic_function_reference_is_rejected( self ) -> None:
		# no single fixed signature to point a raw function pointer at -
		# which specialization?
		code = '\n'.join([
			'def identity[T]( x: T ) -> T:',
			'	return x',
			'',
			'def main() -> None:',
			'	f: Ptr[Callable[[i32],i32]] = identity',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'is generic', self.discovery.errors.errors[0] )

	def test_call_through_callable_pointer_emits_call_indirect( self ) -> None:
		# eq_fn(a, b) where eq_fn: Ptr[Callable[[i32,i32],bool]] - a call
		# through a function-pointer VALUE, not a named Function/method
		# lookup (there's no Function object to resolve at all - the
		# parameter's own declared type is the only thing available)
		code = '\n'.join([
			'def call_it( f: Ptr[Callable[[i32,i32],bool]], a: i32, b: i32 ) -> bool:',
			'	return f( a, b )',
		])
		self._import( code )
		call_it_fn = self.discovery.modules['__test__'].get_local( 'call_it' )
		if call_it_fn.resolve is not None:
			call_it_fn.resolve()
		lf = self.compiler._lower( call_it_fn )
		call_indirect = next( instr for instr in lf.instructions if isinstance( instr, ir.CallIndirect ))
		self.assertEqual( call_indirect.target.stem, 'f' )
		self.assertEqual( [ a.stem for a in call_indirect.args ], [ 'a', 'b' ])
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_call_through_callable_pointer_wrong_arg_count_rejected( self ) -> None:
		code = '\n'.join([
			'def call_it( f: Ptr[Callable[[i32,i32],bool]], a: i32 ) -> bool:',
			'	return f( a )',
		])
		self._import( code )
		call_it_fn = self.discovery.modules['__test__'].get_local( 'call_it' )
		if call_it_fn.resolve is not None:
			call_it_fn.resolve()
		self.compiler._lower( call_it_fn )
		self.assertIn( 'takes 2 argument(s), got 1', self.discovery.errors.errors[0] )

	def test_call_through_callable_field_emits_get_attr_then_call_indirect( self ) -> None:
		# obj.field(...) - a call THROUGH a Ptr[Callable[...]]-typed FIELD,
		# not a bare Name. Was a hard "not callable" compile error before -
		# _try_lower_indirect_call only recognized a bare Name callee
		# (PLAN_CALLABLE.md's own deferred item). Confirms the receiver is
		# read via an ordinary ir.GetAttr (reusing the exact same field-read
		# path an ordinary `x = o.field` already uses), immediately followed
		# by ir.CallIndirect targeting that GetAttr's own dest.
		code = '\n'.join([
			'@cstruct',
			'class Ops:',
			'	handler: Ptr[Callable[[i32,i32],bool]]',
			'',
			'def call_it( o: Ops, a: i32, b: i32 ) -> bool:',
			'	return o.handler( a, b )',
		])
		self._import( code )
		call_it_fn = self.discovery.modules['__test__'].get_local( 'call_it' )
		if call_it_fn.resolve is not None:
			call_it_fn.resolve()
		lf = self.compiler._lower( call_it_fn )
		self.assertEqual( self.discovery.errors.errors, [] )
		get_attr = next( instr for instr in lf.instructions if isinstance( instr, ir.GetAttr ))
		self.assertEqual( get_attr.attr, 'handler' )
		call_indirect = next( instr for instr in lf.instructions if isinstance( instr, ir.CallIndirect ))
		self.assertIs( call_indirect.target, get_attr.dest )
		self.assertEqual( [ a.stem for a in call_indirect.args ], [ 'a', 'b' ])

	def test_ordinary_method_call_not_misrouted_through_field_call_recognizer( self ) -> None:
		# regression guard: the SAME dotted-attribute callee shape (o.name(...))
		# must still dispatch as an ordinary method call, not get swallowed by
		# the new field-call recognizer above - it has to bail via the non-
		# failing _find_method check before ever probing for a field named
		# the same thing (a real method emits ir.Call, never ir.CallIndirect/
		# ir.GetAttr for the callee itself).
		code = '\n'.join([
			'@cstruct',
			'class Ops:',
			'	handler: Ptr[Callable[[i32],i32]]',
			'',
			'	def real_method( self, x: i32 ) -> i32:',
			'		return x',
			'',
			'def call_it( o: Ops, x: i32 ) -> i32:',
			'	return o.real_method( x )',
		])
		self._import( code )
		call_it_fn = self.discovery.modules['__test__'].get_local( 'call_it' )
		if call_it_fn.resolve is not None:
			call_it_fn.resolve()
		lf = self.compiler._lower( call_it_fn )
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertFalse( any( isinstance( i, ir.CallIndirect ) for i in lf.instructions ))
		call = next( instr for instr in lf.instructions if isinstance( instr, ir.Call ))
		self.assertEqual( call.target.stem, 'real_method' )

	def test_call_through_call_result_callee_emits_call_indirect( self ) -> None:
		# get_callback()(...) - the callee is itself a CALL result, not a
		# Name/Attribute this recognizer can statically type-check without
		# evaluating. _resolve_callee's own fallback for any non-Attribute
		# func_node fails IMMEDIATELY (zero evaluation attempted), so
		# evaluating node.func here first and checking its real type is
		# double-evaluation-safe: nothing downstream ever gets a second
		# chance at it. Confirms both the inner ir.Call (get_callback) and
		# the outer ir.CallIndirect (through its result) appear, in order.
		code = '\n'.join([
			'def eq( x: i32, y: i32 ) -> bool:',
			'	return x == y',
			'',
			'def get_callback() -> Ptr[Callable[[i32,i32],bool]]:',
			'	return eq',
			'',
			'def call_it( a: i32, b: i32 ) -> bool:',
			'	return get_callback()( a, b )',
		])
		self._import( code )
		call_it_fn = self.discovery.modules['__test__'].get_local( 'call_it' )
		if call_it_fn.resolve is not None:
			call_it_fn.resolve()
		lf = self.compiler._lower( call_it_fn )
		self.assertEqual( self.discovery.errors.errors, [] )
		call = next( instr for instr in lf.instructions if isinstance( instr, ir.Call ))
		self.assertEqual( call.target.stem, 'get_callback' )
		call_indirect = next( instr for instr in lf.instructions if isinstance( instr, ir.CallIndirect ))
		self.assertIs( call_indirect.target, call.dest )
		self.assertEqual( [ a.stem for a in call_indirect.args ], [ 'a', 'b' ])

	def test_call_through_non_callable_call_result_fails_cleanly( self ) -> None:
		# regression guard: a call-result callee that ISN'T actually
		# Ptr[Callable[...]]-typed must still fail with the ordinary "cannot
		# call ..." diagnostic (via _resolve_callee's ordinary fallback,
		# once the new recognizer branch declines) - not crash, not silently
		# accept it.
		code = '\n'.join([
			'def get_number() -> i32:',
			'	return 5',
			'',
			'def call_it() -> i32:',
			'	return get_number()( 1 )',
		])
		self._import( code )
		call_it_fn = self.discovery.modules['__test__'].get_local( 'call_it' )
		if call_it_fn.resolve is not None:
			call_it_fn.resolve()
		self.compiler._lower( call_it_fn )
		self.assertIn( 'cannot call get_number()', self.discovery.errors.errors[0] )

	def test_call_through_none_returning_callable_pointer_omits_dest( self ) -> None:
		# regression test: a NoneType-returning Callable[...] call used to
		# always allocate a dest temp and emit ir.CallIndirect(dest=...),
		# producing `t = (void)(...)` in the generated C (a real,
		# confirmed compile error - void isn't assignable to anything) -
		# found while building Thread's own entry trampoline, which calls
		# through exactly this shape (Closure[[],None])
		code = '\n'.join([
			'def call_it( f: Ptr[Callable[[],None]] ) -> None:',
			'	f()',
			'	return',
		])
		self._import( code )
		call_it_fn = self.discovery.modules['__test__'].get_local( 'call_it' )
		if call_it_fn.resolve is not None:
			call_it_fn.resolve()
		lf = self.compiler._lower( call_it_fn )
		self.assertEqual( self.discovery.errors.errors, [] )
		call_indirects = [ i for i in lf.instructions if isinstance( i, ir.CallIndirect ) ]
		self.assertEqual( len( call_indirects ), 1 )
		self.assertIsNone( call_indirects[0].dest )

	def test_generic_call_infers_type_params_through_callable_parameter( self ) -> None:
		# regression test: _unify_type_param/substitute_type_params used to
		# stop recursing at a CallableType (it's not a Specialization, so
		# the existing Ptr[T]-vs-Ptr[i32] recursion never looked inside a
		# Ptr[Callable[[T],K]] parameter's own arg_types/return_type at
		# all) - even a plain function reference argument (no lambda
		# involved) failed to infer K this way before the fix
		code = '\n'.join([
			'def identity_i32( v: i32 ) -> i32:',
			'	return v',
			'',
			'def apply[T,K]( x: T, key: Ptr[Callable[[T],K]] ) -> K:',
			'	return key( x )',
			'',
			'def main() -> i32:',
			'	return apply( 5, key = identity_i32 )',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		call = next( instr for instr in fn.instructions if isinstance( instr, ir.Call ))
		self.assertEqual( call.target.parameters[0].type.stem, 'i32' ) # x: T -> i32
		self.assertEqual( call.target.parameters[1].type.stem, 'intrinsics.Ptr[Callable[[intrinsics.i32],intrinsics.i32]]' ) # key: Ptr[Callable[[T],K]] -> Ptr[Callable[[i32],i32]], both T and K bound
		self.assertEqual( call.target.return_type.stem, 'i32' ) # K -> i32

	def test_lambda_eagerly_lowered_infers_generic_return_type( self ) -> None:
		# the circular case the CallableType fix above still couldn't solve
		# on its own (PLAN_LAMBDA.md, "eager lambda lowering"): K is only
		# knowable from the LAMBDA's own inferred return type, which means
		# _expr_Lambda has to lower the lambda's body right now, at this
		# call site, instead of only ever deferring it onto the work queue -
		# see FunctionLowering/_compile_now (compiler.py's own backreference)
		code = '\n'.join([
			'def apply[T,K]( x: T, key: Ptr[Callable[[T],K]] ) -> K:',
			'	return key( x )',
			'',
			'def main() -> i32:',
			'	return apply( 5, key = lambda v: v )',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		call = next( instr for instr in fn.instructions if isinstance( instr, ir.Call ))
		self.assertEqual( call.target.parameters[0].type.stem, 'i32' ) # x: T -> i32
		self.assertEqual( call.target.parameters[1].type.stem, 'intrinsics.Ptr[Callable[[intrinsics.i32],intrinsics.i32]]' ) # key: Ptr[Callable[[T],K]] -> Ptr[Callable[[i32],i32]], K inferred from the lambda's OWN body
		self.assertEqual( call.target.return_type.stem, 'i32' ) # K -> i32
		lambda_ref = call.kwargs['key']
		self.assertIsInstance( lambda_ref, ir.FunctionRef )
		self.assertEqual( lambda_ref.fn.return_type.stem, 'i32' ) # the synthesized lambda Function's own return_type was patched after eager lowering, not left None
		lowered_names = { lf.function.qualname for lf in self.compiler.functions }
		self.assertIn( lambda_ref.fn.qualname, lowered_names ) # actually registered into compiler.functions by the eager _compile_now path, not silently dropped

	def test_lambda_eager_lowering_infers_generic_no_init_construction_type_args( self ) -> None:
		# the same eager-lowering exposure as
		# test_lambda_eagerly_lowered_infers_generic_return_type above, but
		# for _lower_allocate_fields's own gap (see
		# test_bare_construct_infers_type_args_from_field_values_with_no_expected_type):
		# the lambda's own synthetic return_type is deliberately still None
		# while its BODY lowers (same reason PLAN_RETURN_INFERENCE.md's eager
		# passes leave return_type None), so a lambda body constructing a
		# generic no-__init__ class hits _lower_allocate_fields with that
		# same None expected_type - confirms _infer_allocate_type_args fixes
		# this shared call site too, not just the free-function one
		code = '\n'.join([
			'@cstruct',
			'class Box[T]:',
			'	val: T',
			'',
			'def apply[T,K]( x: T, key: Ptr[Callable[[T],K]] ) -> K:',
			'	return key( x )',
			'',
			'def main() -> i32:',
			'	b: Box[i32] = apply( 5, key = lambda v: Box( val = v ) )',
			'	return b.val',
		])
		mod = self._import( code )
		box_cls = mod.get_local( 'Box' )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		call = next( instr for instr in fn.instructions if isinstance( instr, ir.Call ))
		lambda_ref = call.kwargs['key']
		self.assertIsInstance( lambda_ref, ir.FunctionRef )
		self.assertIsInstance( lambda_ref.fn.return_type, Specialization )
		self.assertIs( lambda_ref.fn.return_type.base, box_cls )
		i32_cls = self.discovery.get_intrinsics()['i32']
		self.assertEqual( lambda_ref.fn.return_type.args, [ i32_cls ] )

	def test_lambda_still_fails_when_arg_type_uninferable( self ) -> None:
		# unlike the return type, a lambda's own PARAMETER types have no
		# body to infer them from - an unbound arg type in the expected
		# Callable[...] is still unrecoverable, eager lowering or not, and
		# must keep failing with a clear error rather than silently trying
		# to eagerly lower a body it can't even assign parameter types to
		code = '\n'.join([
			'def only_callable[T]( key: Ptr[Callable[[T],T]] ) -> i32:',
			'	return 0',
			'',
			'def main() -> i32:',
			'	return only_callable( key = lambda v: v )',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any(
			'cannot infer lambda parameter types' in e for e in self.discovery.errors.errors
		), self.discovery.errors.errors )

	def test_lambda_eagerly_lowered_nested_inside_another_eager_lambda( self ) -> None:
		# regression guard for the whole point of the FunctionLowering class
		# split (PLAN_LAMBDA.md): eagerly lowering a lambda's body can ITSELF
		# hit another generic call needing ANOTHER lambda eagerly lowered,
		# mid-way through the first - a genuinely reentrant lower_function
		# call. Each level gets its own independent FunctionLowering
		# instance (no shared mutable per-function state to clobber), and
		# _lambda_counter (persistent, on the shared Lowering instance)
		# still hands out unique names across all of them
		code = '\n'.join([
			'def apply[T,K]( x: T, key: Ptr[Callable[[T],K]] ) -> K:',
			'	return key( x )',
			'',
			'def outer[T,K]( x: T, key: Ptr[Callable[[T],K]] ) -> K:',
			'	return key( x )',
			'',
			'def main() -> i32:',
			'	return outer( 5, key = lambda v: apply( v, key = lambda w: w ) )',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		call = next( instr for instr in fn.instructions if isinstance( instr, ir.Call ))
		self.assertEqual( call.target.return_type.stem, 'i32' )
		lowered_names = { lf.function.qualname for lf in self.compiler.functions }
		self.assertEqual(
			{ 'main', 'main$$lambda_1', 'main$$lambda_1$$lambda_2' } & lowered_names,
			{ 'main', 'main$$lambda_1', 'main$$lambda_1$$lambda_2' },
		) # both lambdas got distinct names and were both actually registered - no counter collision, no dropped nested lowering

	# --- non-capturing nested function defs (PLAN_LAMBDA.md) ------------------

	def test_nested_def_called_from_enclosing_function( self ) -> None:
		code = '\n'.join([
			'def outer() -> i32:',
			'	def inner( x: i32 ) -> i32:',
			'		with compiler.wrap_arithmetic:',
			'			return x + 1',
			'	return inner( 5 )',
		])
		self._import( code )
		outer_fn = self.discovery.modules['__test__'].get_local( 'outer' )
		if outer_fn.resolve is not None:
			outer_fn.resolve()
		lf = self.compiler._lower( outer_fn )
		self.assertEqual( self.discovery.errors.errors, [] )
		call = next( instr for instr in lf.instructions if isinstance( instr, ir.Call ))
		self.assertEqual( call.target.qualname, '__test__.outer$$nested_inner' )

	def test_nested_def_bare_reference_lowers_to_function_ref( self ) -> None:
		code = '\n'.join([
			'def outer() -> None:',
			'	def inner( x: i32 ) -> i32:',
			'		with compiler.wrap_arithmetic:',
			'			return x + 1',
			'	f: Ptr[Callable[[i32],i32]] = inner',
			'	return',
		])
		self._import( code )
		outer_fn = self.discovery.modules['__test__'].get_local( 'outer' )
		if outer_fn.resolve is not None:
			outer_fn.resolve()
		lf = self.compiler._lower( outer_fn )
		self.assertEqual( self.discovery.errors.errors, [] )
		assign = next( instr for instr in lf.instructions if isinstance( instr, ir.Assign ))
		self.assertIsInstance( assign.src, ir.FunctionRef )
		self.assertEqual( assign.src.fn.qualname, '__test__.outer$$nested_inner' )

	# --- capturing nested function defs (real closures) -----------------------

	def test_nested_def_capturing_scalar_local_builds_env_then_closure_allocate( self ) -> None:
		code = '\n'.join([
			'def outer( y: i32 ) -> i32:',
			'	def inner( x: i32 ) -> i32:',
			'		with compiler.wrap_arithmetic:',
			'			return x + y',
			'	return inner( 5 )',
		])
		self._import( code )
		outer_fn = self.discovery.modules['__test__'].get_local( 'outer' )
		if outer_fn.resolve is not None:
			outer_fn.resolve()
		lf = self.compiler._lower( outer_fn )
		self.assertEqual( self.discovery.errors.errors, [] )
		allocates = [ i for i in lf.instructions if isinstance( i, ir.Allocate ) ]
		self.assertEqual( len( allocates ), 2 )
		env_alloc, closure_alloc = allocates
		self.assertIsInstance( closure_alloc.cls, ClosureType )
		self.assertNotIsInstance( env_alloc.cls, ClosureType )
		self.assertIn( 'fn', closure_alloc.fields )
		self.assertIn( 'self', closure_alloc.fields )
		# env class's own field keeps the REAL captured type (i32), not
		# erased to Ptr[None] the way ClosureType's own fn/self fields are -
		# this is what lets the env's own destructor be synthesized for free
		self.assertIn( 'y', env_alloc.fields )
		y_attr = next( a for a in env_alloc.cls.attributes if a.stem == 'y' )
		self.assertEqual( y_attr.type.stem, 'i32' )
		# no Incref at all for a scalar capture
		self.assertFalse( any( isinstance( i, ir.Incref ) for i in lf.instructions ))

	def test_nested_def_capturing_binds_closure_typed_variable_not_a_function( self ) -> None:
		# unlike the non-capturing case (test_nested_def_bare_reference_
		# lowers_to_function_ref above), a capturing nested def's own name
		# must resolve to a real Variable of ClosureType - _try_lower_
		# closure_call only ever matches a Variable, never a bare Function
		code = '\n'.join([
			'def outer( y: i32 ) -> i32:',
			'	def inner( x: i32 ) -> i32:',
			'		with compiler.wrap_arithmetic:',
			'			return x + y',
			'	return inner( 5 )',
		])
		self._import( code )
		outer_fn = self.discovery.modules['__test__'].get_local( 'outer' )
		if outer_fn.resolve is not None:
			outer_fn.resolve()
		self.compiler._lower( outer_fn )
		self.assertEqual( self.discovery.errors.errors, [] )
		bound = outer_fn.names['inner']
		self.assertIsInstance( bound, Variable )
		self.assertNotIsInstance( bound, Function )
		self.assertIsInstance( bound.type, ClosureType )

	def test_nested_def_capturing_called_directly_emits_callindirect( self ) -> None:
		code = '\n'.join([
			'def outer( y: i32 ) -> i32:',
			'	def inner( x: i32 ) -> i32:',
			'		with compiler.wrap_arithmetic:',
			'			return x + y',
			'	return inner( 5 )',
		])
		self._import( code )
		outer_fn = self.discovery.modules['__test__'].get_local( 'outer' )
		if outer_fn.resolve is not None:
			outer_fn.resolve()
		lf = self.compiler._lower( outer_fn )
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertFalse( any( isinstance( i, ir.Call ) for i in lf.instructions ))
		call_indirects = [ i for i in lf.instructions if isinstance( i, ir.CallIndirect ) ]
		self.assertEqual( len( call_indirects ), 1 )

	def test_nested_def_capturing_rc_typed_local_increfs_env_field_once( self ) -> None:
		code = '\n'.join([
			'class Box:',
			'	v: i32',
			'	@staticmethod',
			'	def make( v: i32 ) -> Box:',
			'		return Box.__allocate__( v = v )',
			'',
			'def outer( b: Box ) -> i32:',
			'	def inner() -> i32:',
			'		return b.v',
			'	return inner()',
		])
		self._import( code )
		outer_fn = self.discovery.modules['__test__'].get_local( 'outer' )
		if outer_fn.resolve is not None:
			outer_fn.resolve()
		lf = self.compiler._lower( outer_fn )
		self.assertEqual( self.discovery.errors.errors, [] )
		increfs = [ i for i in lf.instructions if isinstance( i, ir.Incref ) ]
		self.assertEqual( len( increfs ), 1 )
		allocates = [ i for i in lf.instructions if isinstance( i, ir.Allocate ) ]
		env_alloc = next( a for a in allocates if not isinstance( a.cls, ClosureType ))
		# the capture's own Incref happens before the env is allocated (the
		# same "aliasing read gets its own Incref before being embedded"
		# ordering cfg.field_value/_lower_allocate_fields already use
		# everywhere else)
		self.assertLess( lf.instructions.index( increfs[0] ), lf.instructions.index( env_alloc ))

	def test_two_nested_def_occurrences_get_independent_env_classes( self ) -> None:
		# each closure called independently, not combined via `+` - this
		# test file's own Discovery(import_builtins=False) setup has no
		# real scalar arithmetic dunder dispatch available (unrelated to
		# closures - confirmed: even a plain `a + 1` fails identically here)
		code = '\n'.join([
			'def outer( a: i32, b: i32 ) -> i32:',
			'	def first() -> i32:',
			'		return a',
			'	def second() -> i32:',
			'		return b',
			'	first()',
			'	return second()',
		])
		self._import( code )
		outer_fn = self.discovery.modules['__test__'].get_local( 'outer' )
		if outer_fn.resolve is not None:
			outer_fn.resolve()
		lf = self.compiler._lower( outer_fn )
		self.assertEqual( self.discovery.errors.errors, [] )
		env_allocs = [ i for i in lf.instructions if isinstance( i, ir.Allocate ) and not isinstance( i.cls, ClosureType ) ]
		self.assertEqual( len( env_allocs ), 2 )
		self.assertIsNot( env_allocs[0].cls, env_allocs[1].cls )

	def test_reassigned_captured_name_is_local_not_captured( self ) -> None:
		# `x = x + 1`-shaped body: any ast.Store anywhere in the body makes
		# that name local for the WHOLE body (mirrors real Python's own
		# hoisting rule) - x is never collected as a capture, so this
		# compiles as an ordinary (here: use-before-first-assignment) error,
		# not a capture of the enclosing y
		code = '\n'.join([
			'def outer( y: i32 ) -> i32:',
			'	def inner() -> i32:',
			'		with compiler.wrap_arithmetic:',
			'			y = y + 1',
			'		return y',
			'	return inner()',
		])
		self._import( code )
		outer_fn = self.discovery.modules['__test__'].get_local( 'outer' )
		if outer_fn.resolve is not None:
			outer_fn.resolve()
		self.compiler._lower( outer_fn )
		self.compiler._drain() # inner's own body (and its own errors) is only lowered once dequeued - _stmt_FunctionDef merely schedule()s it
		# NOT a "captures 'y'" error - y is treated as inner's own local,
		# read before it's ever assigned
		self.assertTrue( self.discovery.errors.errors )
		self.assertNotIn( "captures 'y'", self.discovery.errors.errors[0] )

	def test_nested_def_capturing_undefined_name_fails_not_defined( self ) -> None:
		code = '\n'.join([
			'def outer( y: i32 ) -> i32:',
			'	def inner() -> i32:',
			'		return totally_undefined_name',
			'	return inner()',
		])
		self._import( code )
		outer_fn = self.discovery.modules['__test__'].get_local( 'outer' )
		if outer_fn.resolve is not None:
			outer_fn.resolve()
		self.compiler._lower( outer_fn )
		self.compiler._drain() # inner's own body is only lowered once dequeued - see the previous test's identical comment
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'not defined', self.discovery.errors.errors[0] )

	def test_nested_def_inside_generic_function_is_rejected( self ) -> None:
		code = '\n'.join([
			'def outer[T]( y: T ) -> None:',
			'	def inner() -> None:',
			'		return',
			'	inner()',
			'	return',
		])
		self._import( code )
		outer_fn = self.discovery.modules['__test__'].get_local( 'outer' )
		if outer_fn.resolve is not None:
			outer_fn.resolve()
		spec = self.discovery._get_or_create_specialization( outer_fn, [ self.discovery.get_intrinsics()['i32'] ] )
		self.compiler._lower( spec )
		self.assertIn( 'nested function defs are not supported inside a generic function', self.discovery.errors.errors[0] )

	# --- non-capturing lambdas (PLAN_LAMBDA.md) --------------------------------

	def test_lambda_param_types_inferred_from_expected_callable( self ) -> None:
		code = '\n'.join([
			'def call_it( f: Ptr[Callable[[i32],i32]], v: i32 ) -> i32:',
			'	return f( v )',
			'',
			'def main() -> i32:',
			'	return call_it( lambda x: x, 5 )',
		])
		self._import( code )
		main_fn = self.discovery.modules['__test__'].get_local( 'main' )
		if main_fn.resolve is not None:
			main_fn.resolve()
		lf = self.compiler._lower( main_fn )
		self.assertEqual( self.discovery.errors.errors, [] )
		call = next( instr for instr in lf.instructions if isinstance( instr, ir.Call ))
		lambda_ref = next( a for a in call.args if isinstance( a, ir.FunctionRef ))
		self.assertEqual( lambda_ref.fn.qualname, 'main$$lambda_1' ) # 'main' is reserved as the bare, never module-qualified entry point
		self.assertEqual( lambda_ref.fn.parameters[0].type.stem, 'i32' )
		self.assertEqual( lambda_ref.fn.return_type.stem, 'i32' )

	def test_lambda_without_expected_callable_context_is_rejected( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	f = lambda x: x',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'cannot infer lambda parameter types', self.discovery.errors.errors[0] )

	def test_lambda_wrong_arg_count_is_rejected( self ) -> None:
		code = '\n'.join([
			'def call_it( f: Ptr[Callable[[i32,i32],bool]] ) -> None:',
			'	return',
			'',
			'def main() -> None:',
			'	call_it( lambda x: x )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'lambda takes 1 argument(s)', self.discovery.errors.errors[0] )

	def test_lambda_capturing_enclosing_local_builds_closure( self ) -> None:
		# unlike a non-capturing lambda (always Ptr[Callable[...]]), a
		# capturing lambda's own result is a ClosureType - so it can only be
		# written where a Closure[...]-shaped context (not Ptr[Callable[...]])
		# is available to infer its parameter types from; ClosureType
		# duck-types the same arg_types/return_type shape CallableType does
		# for exactly this reason (see _expr_Lambda's own comment)
		code = '\n'.join([
			'def call_it( f: Closure[[i32],i32], v: i32 ) -> i32:',
			'	return f( v )',
			'',
			'def outer( y: i32 ) -> i32:',
			'	return call_it( lambda x: y, 5 )',
			'',
			'def main() -> i32:',
			'	return outer( 1 )',
		])
		self._import( code )
		outer_fn = self.discovery.modules['__test__'].get_local( 'outer' )
		if outer_fn.resolve is not None:
			outer_fn.resolve()
		lf = self.compiler._lower( outer_fn )
		self.assertEqual( self.discovery.errors.errors, [] )
		allocates = [ i for i in lf.instructions if isinstance( i, ir.Allocate ) ]
		self.assertEqual( len( allocates ), 2 )
		self.assertIsInstance( allocates[1].cls, ClosureType )
		self.assertNotIsInstance( allocates[0].cls, ClosureType )

	def test_lambda_capturing_no_incref_for_scalar_capture( self ) -> None:
		code = '\n'.join([
			'def call_it( f: Closure[[i32],i32], v: i32 ) -> i32:',
			'	return f( v )',
			'',
			'def outer( y: i32 ) -> i32:',
			'	return call_it( lambda x: y, 5 )',
		])
		self._import( code )
		outer_fn = self.discovery.modules['__test__'].get_local( 'outer' )
		if outer_fn.resolve is not None:
			outer_fn.resolve()
		lf = self.compiler._lower( outer_fn )
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertFalse( any( isinstance( i, ir.Incref ) for i in lf.instructions ))

	def test_two_lambda_occurrences_get_independent_env_classes( self ) -> None:
		code = '\n'.join([
			'def call_it( f: Closure[[i32],i32], v: i32 ) -> i32:',
			'	return f( v )',
			'',
			'def outer( a: i32, b: i32 ) -> i32:',
			'	call_it( lambda x: a, 1 )',
			'	return call_it( lambda x: b, 2 )',
		])
		self._import( code )
		outer_fn = self.discovery.modules['__test__'].get_local( 'outer' )
		if outer_fn.resolve is not None:
			outer_fn.resolve()
		lf = self.compiler._lower( outer_fn )
		self.assertEqual( self.discovery.errors.errors, [] )
		env_allocs = [ i for i in lf.instructions if isinstance( i, ir.Allocate ) and not isinstance( i.cls, ClosureType ) ]
		self.assertEqual( len( env_allocs ), 2 )
		self.assertIsNot( env_allocs[0].cls, env_allocs[1].cls )

	def test_lambda_capturing_inside_generic_function_is_rejected( self ) -> None:
		# same restriction as a capturing nested def (test_nested_def_
		# inside_generic_function_is_rejected) - _reject_generic_enclosing_
		# scope runs before capture collection either way
		code = '\n'.join([
			'def outer[T]( y: T, f: Closure[[],T] ) -> None:',
			'	g: Closure[[],T] = lambda: y',
			'	return',
		])
		self._import( code )
		outer_fn = self.discovery.modules['__test__'].get_local( 'outer' )
		if outer_fn.resolve is not None:
			outer_fn.resolve()
		spec = self.discovery._get_or_create_specialization( outer_fn, [ self.discovery.get_intrinsics()['i32'] ] )
		self.compiler._lower( spec )
		self.assertIn( 'lambdas are not supported inside a generic function', self.discovery.errors.errors[0] )

	def test_lambda_eager_return_type_inference_with_capture( self ) -> None:
		# the eager-lowering path (PLAN_LAMBDA.md's own "Follow-up done") -
		# key's own K is still a bare TypeVar until the lambda's body is
		# lowered - composed here with a capture (y). Per the closures
		# plan's own note on ordering: capture collection/env-build/rewrite
		# happens BEFORE _compile_now, so eager lowering sees an already-
		# closed body and infers the real return type correctly regardless.
		# NOTE: checked at the LOWERING level only (never calls emitter_c.
		# emit_c()) - this test only verifies what THIS pass is responsible
		# for (the closure's OWN return type, correctly eagerly inferred
		# despite capturing). The outer generic apply[T,K]'s own return
		# type substitution through a Closure[...]-shaped parameter used to
		# have a separate gap (_unify_type_param/_type_mentions_param had
		# no ClosureType branch, so K was wrongly return-only-inferred
		# instead of bound from `key`'s own real type) - now fixed, see
		# emitter_c_test.py's test_generic_closure_param_return_type_
		# inferred_from_argument for the full compile+run coverage.
		code = '\n'.join([
			'def apply[T,K]( x: T, key: Closure[[T],K] ) -> K:',
			'	return key( x )',
			'',
			'def outer( y: i32 ) -> i32:',
			# assigned to an unannotated local, NOT returned directly -
			# unrelated to the now-fixed Closure gap above, kept as-is since
			# it still exercises this file's own _emit_generic_call typing a
			# generic call's dest as the REAL (monomorphized) return type
			# rather than the outer context's own expected type. i32(5), not
			# a bare 5, still needed - see this file's own bare-literal-
			# default comment elsewhere - i32 is the ONLY other thing that
			# could pin T here.
			'	result = apply( i32( 5 ), key = lambda v: y )',
			'	return y',
		])
		self._import( code )
		outer_fn = self.discovery.modules['__test__'].get_local( 'outer' )
		if outer_fn.resolve is not None:
			outer_fn.resolve()
		lf = self.compiler._lower( outer_fn )
		self.assertEqual( self.discovery.errors.errors, [] )
		closure_allocs = [ i for i in lf.instructions if isinstance( i, ir.Allocate ) and isinstance( i.cls, ClosureType ) ]
		self.assertEqual( len( closure_allocs ), 1 )
		self.assertEqual( closure_allocs[0].cls.return_type.stem, 'i32' )

	# NOTE: a capturing lambda written where the context expects
	# Ptr[Callable[...]] instead of Closure[...] (e.g. the OLD, pre-capture
	# shape of the test above) is NOT caught as a clean type-mismatch error
	# at lowering time - it silently produces a ClosureType-shaped operand
	# in a Ptr[Callable[...]]-declared slot, which later crashes the
	# EMITTER with an internal AssertionError (emitter_c.py's own
	# `assert isinstance(concrete_cls, RCClass)`) instead of a real compile
	# error. Confirmed pre-existing and NOT specific to capturing closures -
	# the identical crash reproduces with today's already-shipped bound-
	# method closures (`w.get` passed where Ptr[Callable[...]] is
	# expected). Out of scope here (a _lower_call_args/argument type-
	# checking gap, unrelated to closure construction itself) - flagged for
	# a follow-up, not fixed in this pass.

	# --- Closure[[...],...] bound-method values -------------------------------

	def test_bound_method_reference_builds_closure_allocate( self ) -> None:
		code = '\n'.join([
			'class Worker:',
			'	x: i32',
			'	@staticmethod',
			'	def make( v: i32 ) -> Worker:',
			'		return Worker.__allocate__( x = v )',
			'	def get( self ) -> i32:',
			'		return self.x',
			'',
			'def main() -> None:',
			'	w: Worker = Worker.make( 1 )',
			'	c: Closure[[], i32] = w.get',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		# incref the receiver exactly once (a new owner - the closure),
		# BEFORE the Allocate that actually constructs the closure value
		increfs = [ i for i in fn.instructions if isinstance( i, ir.Incref ) ]
		self.assertEqual( len( increfs ), 1 )
		allocates = [ i for i in fn.instructions if isinstance( i, ir.Allocate ) and isinstance( i.cls, ClosureType ) ]
		self.assertEqual( len( allocates ), 1 )
		self.assertIn( 'fn', allocates[0].fields )
		self.assertIn( 'self', allocates[0].fields )
		self.assertLess( fn.instructions.index( increfs[0] ), fn.instructions.index( allocates[0] ))

	def test_bound_method_reference_synthesizes_one_trampoline_reused_across_references( self ) -> None:
		# two separate `w.get` references to the SAME method - only one
		# trampoline gets synthesized (memoized by (method, owner_type)),
		# not one per reference
		code = '\n'.join([
			'class Worker:',
			'	x: i32',
			'	@staticmethod',
			'	def make( v: i32 ) -> Worker:',
			'		return Worker.__allocate__( x = v )',
			'	def get( self ) -> i32:',
			'		return self.x',
			'',
			'def main() -> None:',
			'	w: Worker = Worker.make( 1 )',
			'	c1: Closure[[], i32] = w.get',
			'	c2: Closure[[], i32] = w.get',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		fn_refs = [ i.src for i in fn.instructions if isinstance( i, ir.Assign ) and isinstance( i.src, ir.FunctionRef ) ]
		# both closures' own fn field construction goes through the SAME
		# trampoline Function object - never re-synthesized per reference.
		# fn_refs itself won't show this directly (the CastWrap operand is
		# what carries the FunctionRef - see the Allocate fields instead)
		allocates = [ i for i in fn.instructions if isinstance( i, ir.Allocate ) and isinstance( i.cls, ClosureType ) ]
		self.assertEqual( len( allocates ), 2 )
		cast_wraps = [ i for i in fn.instructions if isinstance( i, ir.CastWrap ) and isinstance( i.operand, ir.FunctionRef ) ]
		self.assertEqual( len( cast_wraps ), 2 )
		self.assertIs( cast_wraps[0].operand.fn, cast_wraps[1].operand.fn )

	def test_ordinary_method_call_does_not_construct_a_closure( self ) -> None:
		# w.get() (call parens) must stay an ordinary, direct method call -
		# _lower_call resolves node.func structurally and never routes it
		# through _expr_Attribute's own closure-construction branch at all
		code = '\n'.join([
			'class Worker:',
			'	x: i32',
			'	@staticmethod',
			'	def make( v: i32 ) -> Worker:',
			'		return Worker.__allocate__( x = v )',
			'	def get( self ) -> i32:',
			'		return self.x',
			'',
			'def main() -> i32:',
			'	w: Worker = Worker.make( 1 )',
			'	return w.get()',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertFalse( any( isinstance( i, ir.Allocate ) and isinstance( i.cls, ClosureType ) for i in fn.instructions ))
		self.assertFalse( any( isinstance( i, ir.Incref ) for i in fn.instructions ))
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( [ c.target.qualname for c in calls ], [ '__test__.Worker.make', '__test__.Worker.get' ] )

	def test_closure_call_emits_callindirect_with_self_prepended( self ) -> None:
		code = '\n'.join([
			'class Worker:',
			'	x: i32',
			'	@staticmethod',
			'	def make( v: i32 ) -> Worker:',
			'		return Worker.__allocate__( x = v )',
			'	def add( self, n: i32 ) -> i32:',
			'		with compiler.wrap_arithmetic:',
			'			return self.x + n',
			'',
			'def main() -> i32:',
			'	w: Worker = Worker.make( 1 )',
			'	c: Closure[[i32], i32] = w.add',
			'	return c( 5 )',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		call_indirects = [ i for i in fn.instructions if isinstance( i, ir.CallIndirect ) ]
		self.assertEqual( len( call_indirects ), 1 )
		# 2 args on the wire: the closure's own erased self, then the
		# user-supplied 5 - not just the user-supplied argument alone
		self.assertEqual( len( call_indirects[0].args ), 2 )

	def test_closure_typed_field_called_directly_emits_callindirect_with_self_prepended( self ) -> None:
		# obj.field(...) where field: Closure[[...],...] - previously fell
		# through every recognizer to _attr_lookup_callable's generic
		# "'work' is not callable on ..." diagnostic, since
		# _try_lower_closure_call only ever matched a bare Name callee and
		# _try_lower_indirect_call's own Attribute branch only recognizes
		# Ptr[Callable[...]] fields (type_resolver._callable_type_of), never
		# ClosureType ones. Same self-prepended calling convention as the
		# bare-Name case above.
		code = '\n'.join([
			'class Worker:',
			'	x: i32',
			'	@staticmethod',
			'	def make( v: i32 ) -> Worker:',
			'		return Worker.__allocate__( x = v )',
			'	def add( self, n: i32 ) -> i32:',
			'		with compiler.wrap_arithmetic:',
			'			return self.x + n',
			'',
			'class Job:',
			'	work: Closure[[i32], i32]',
			'	def __init__( self, w: Closure[[i32], i32] ) -> None:',
			'		self.work = w',
			'',
			'def main() -> i32:',
			'	w: Worker = Worker.make( 1 )',
			'	c: Closure[[i32], i32] = w.add',
			'	j: Job = Job( c )',
			'	return j.work( 5 )',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		call_indirects = [ i for i in fn.instructions if isinstance( i, ir.CallIndirect ) ]
		self.assertEqual( len( call_indirects ), 1 )
		self.assertEqual( len( call_indirects[0].args ), 2 )

	def test_non_callable_field_still_rejected_by_closure_call_recognizer( self ) -> None:
		# guards the new Attribute branch above against over-matching: a
		# plain scalar field must still fail with the pre-existing generic
		# diagnostic, not be silently accepted as a closure call
		code = '\n'.join([
			'class Job:',
			'	x: i32',
			'	def __init__( self, x: i32 ) -> None:',
			'		self.x = x',
			'',
			'def main() -> i32:',
			'	j: Job = Job( 5 )',
			'	return j.x( 5 )',
		])
		self._import( code )
		self.compiler._lower( self.discovery.main )
		self.assertTrue(
			any( "'x' is not callable on" in e for e in self.discovery.errors.errors ),
			f'expected a not-callable diagnostic, got: {self.discovery.errors.errors}',
		)

	def test_bound_method_on_monomorphized_generic_class_works( self ) -> None:
		# Box[T].get isn't itself generic (method.type_params is empty -
		# T comes from the CLASS's own specialization, already concrete by
		# the time _find_method resolves it through _ensure_resolved) -
		# only a method that's independently generic (def foo[T](...)) hits
		# _lower_function_ref's own restriction; this is a different case
		# and is expected to just work
		code = '\n'.join([
			'class Box[T]:',
			'	v: T',
			'	def get( self ) -> T:',
			'		return self.v',
			'',
			'def main() -> None:',
			'	b: Box[i32] = Box( v = 1 )',
			'	c: Closure[[], i32] = b.get',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		allocates = [ i for i in fn.instructions if isinstance( i, ir.Allocate ) and isinstance( i.cls, ClosureType ) ]
		self.assertEqual( len( allocates ), 1 )

	def test_bound_method_generic_function_itself_is_rejected( self ) -> None:
		# a method that's independently generic (not just via its owning
		# class) has no single fixed signature to point a trampoline at -
		# a real compile error either way (confirmed: rejected earlier,
		# by _find_method itself failing to resolve a bare generic-method
		# reference at all, before this closure-construction's own
		# type_params check would even run)
		code = '\n'.join([
			'class Foo:',
			'	def get[T]( self, default: T ) -> T:',
			'		return default',
			'',
			'def main() -> None:',
			'	f: Foo = Foo()',
			'	c: Closure[[i32], i32] = f.get',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( self.discovery.errors.errors )

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
			ir.DeleteTemp( temp = t0 ),
			ir.Return( value = t0 ),
			ir.FuncEnd( name = make_fn.qualname ),
		])

	def test_rcclass_allocate_schedules_sys_alloc_specialization( self ) -> None:
		# an RCClass's own memory must come through the SAME allocation path
		# every other real allocation in the language goes through -
		# sys.alloc[T] - not an emitter-invented allocator (explicit user
		# decision, see the plan's Context section). This guarantees
		# sys.alloc[Foo] is a real, schedulable compile unit by the time the
		# emitter needs to independently synthesize a call to it. Unlike
		# test_allocate_emits_allocate_instruction (a @cstruct - no header/
		# allocator involved at all), Foo here is a plain (RCClass) class
		code = '\n'.join([
			'class Foo:',
			'	x: i32',
			'',
			'	@staticmethod',
			'	def make( v: i32 ) -> Foo:',
			'		return Foo.__allocate__( x = v )',
		])
		mod = self._import( code )
		foo_cls = mod.get_local( 'Foo' )
		foo_cls.resolve()
		make_fn = foo_cls.get_local( 'make' )
		if make_fn.resolve is not None:
			make_fn.resolve()
		self.compiler._lower( make_fn )
		self.assertEqual( self.discovery.errors.errors, [] )
		queued = []
		while True:
			try:
				queued.append( self.compiler.queue.get_nowait() )
			except queue_module.Empty:
				break
		alloc_specs = [
			u for u in queued
			if isinstance( u, Specialization ) and isinstance( u.base, Function ) and u.base.qualname == 'sys.alloc'
		]
		self.assertEqual( len( alloc_specs ), 1 )
		self.assertEqual( alloc_specs[0].args, [ foo_cls ] )

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

	def test_allocate_dest_type_infers_type_args_when_no_expected_type( self ) -> None:
		# sibling of test_allocate_dest_type_uses_expected_type_when_given -
		# `res` is bare/unannotated here, so no expected_type reaches this
		# __allocate__ call at all; only the field VALUE's own real type
		# (fields[name].type, computed while lowering `x = v`) is available
		# to infer Foo's own type arg from - the pre-existing gap this fixes
		# (see _infer_allocate_type_args) silently fell back to the bare,
		# unspecialized Foo class here instead
		code = '\n'.join([
			'@cstruct',
			'class Foo[T]:',
			'	x: T',
			'',
			'	@staticmethod',
			'	def make( v: T ) -> Foo[T]:',
			'		res = Foo.__allocate__( x = v )',
			'		return res',
		])
		mod = self._import( code )
		foo_cls = mod.get_local( 'Foo' )
		foo_cls.resolve()
		make_fn = foo_cls.get_local( 'make' )
		if make_fn.resolve is not None:
			make_fn.resolve()
		fn = self.compiler._lower( make_fn )
		self.assertEqual( self.discovery.errors.errors, [] )
		allocate_instr = next( i for i in fn.instructions if isinstance( i, ir.Allocate ))
		self.assertIsInstance( allocate_instr.dest.type, Specialization )
		self.assertIs( allocate_instr.dest.type.base, foo_cls )

	def test_allocate_accepts_explicit_self_subscript_of_own_type_param( self ) -> None:
		# a generic class's own method re-applying its OWN type parameter
		# explicitly - Foo[T].__allocate__(...) - rather than the bare
		# Foo.__allocate__(...) spelling used everywhere above. Both should
		# resolve identically (redundant [T] carries no new information over
		# the in-scope T lookup the bare form already does); previously
		# Foo[T] in this call-target position was lowered as a VALUE
		# expression instead of a type reference, failing with "'Foo' is
		# not a value, cannot use it as an expression"
		code = '\n'.join([
			'@cstruct',
			'class Foo[T]:',
			'	x: T',
			'',
			'	@staticmethod',
			'	def make( v: T ) -> Foo[T]:',
			'		res: Foo[T] = Foo[T].__allocate__( x = v )',
			'		return res',
		])
		mod = self._import( code )
		foo_cls = mod.get_local( 'Foo' )
		foo_cls.resolve()
		make_fn = foo_cls.get_local( 'make' )
		if make_fn.resolve is not None:
			make_fn.resolve()
		fn = self.compiler._lower( make_fn )
		self.assertEqual( self.discovery.errors.errors, [] )
		allocate_instr = next( i for i in fn.instructions if isinstance( i, ir.Allocate ))
		self.assertIsInstance( allocate_instr.dest.type, Specialization )
		self.assertIs( allocate_instr.dest.type.base, foo_cls )

	def test_bare_construct_infers_type_args_from_field_values_with_no_expected_type( self ) -> None:
		# the bug report's own repro shape (also PLAN_RETURN_INFERENCE.md's
		# "Also found and worked around, NOT fixed" note): a generic
		# @cstruct with no __init__, constructed via the bare ClassName(...)
		# sugar (_try_lower_construct_call, not .__allocate__), assigned to
		# an unannotated local - genuinely CONCRETE, differently-typed field
		# values (Widget for `first`, bool for `second`) are the only source
		# of Pair's own type args here
		code = '\n'.join([
			'@cstruct',
			'class Pair[T,R]:',
			'	first: T',
			'	second: R',
			'',
			'@cstruct',
			'class Widget:',
			'	y: i32',
			'	def derive( self ) -> bool:',
			'		return self.y != 0',
			'',
			'def main( w: Widget ) -> bool:',
			'	p = Pair( first = w, second = w.derive() )',
			'	return p.second',
		])
		mod = self._import( code )
		pair_cls = mod.get_local( 'Pair' )
		widget_cls = mod.get_local( 'Widget' )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		allocate_instr = next( i for i in fn.instructions if isinstance( i, ir.Allocate ))
		self.assertIsInstance( allocate_instr.dest.type, Specialization )
		self.assertIs( allocate_instr.dest.type.base, pair_cls )
		bool_cls = self.discovery.get_intrinsics()['bool']
		self.assertEqual( allocate_instr.dest.type.args, [ widget_cls, bool_cls ] )

	def test_allocate_fails_loudly_when_type_args_unresolvable( self ) -> None:
		# S appears in no field at all - genuinely unresolvable, must fail
		# loudly instead of silently building the abstract, unspecialized
		# class (the pre-fix behavior) - mirrors
		# ReturnOnlyTypeParamInferenceTests' own
		# test_vacuous_type_param_alongside_return_only_still_fails
		code = '\n'.join([
			'@cstruct',
			'class Foo[T,S]:',
			'	x: T',
			'',
			'def main( v: i32 ) -> None:',
			'	f = Foo( x = v )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any(
			'cannot infer type parameter(s)' in e and 'S' in e for e in self.discovery.errors.errors
		), self.discovery.errors.errors )

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

	def test_call_dunder_dispatches_instead_of_construction( self ) -> None:
		# T(...) where T defines a static __call__ dispatches there instead
		# of construction - no ir.Allocate at all, an ordinary ir.Call
		# targeting __call__, matching how ClassName.static_method(...)
		# already resolves for e.g. int.from_str(...).
		code = '\n'.join([
			'@cstruct',
			'class Converter:',
			'	@staticmethod',
			'	def __call__( x: i32 ) -> i32:',
			'		return x',
			'',
			'def main() -> i32:',
			'	return Converter( 41 )',
		])
		mod = self._import( code )
		converter_cls = mod.get_local( 'Converter' )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertFalse( any( isinstance( i, ir.Allocate ) for i in fn.instructions ) )
		call_instr = next( i for i in fn.instructions if isinstance( i, ir.Call ))
		self.assertIs( call_instr.target, converter_cls.get_local( '__call__' ))
		self.assertIsNone( call_instr.receiver )

	def test_call_dunder_absent_construction_unchanged( self ) -> None:
		# the same shape as above, but Foo has no __call__ at all - ordinary
		# field=value construction sugar must still fire unchanged (the
		# ~100% common case this pre-pass must stay a no-op for)
		code = '\n'.join([
			'@cstruct',
			'class Foo:',
			'	x: i32',
			'',
			'def main() -> Foo:',
			'	return Foo( x = 1 )',
		])
		mod = self._import( code )
		foo_cls = mod.get_local( 'Foo' )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		allocate_instr = next( i for i in fn.instructions if isinstance( i, ir.Allocate ))
		self.assertIs( allocate_instr.cls, foo_cls )

	def test_call_dunder_must_be_static( self ) -> None:
		# an ordinary (non-static) __call__ can't be used for T(...)
		# dispatch - there's no receiver instance to bind self to yet
		code = '\n'.join([
			'@cstruct',
			'class Bad:',
			'	def __call__( self, x: i32 ) -> i32:',
			'		return x',
			'',
			'def main() -> i32:',
			'	return Bad( 1 )',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any(
			'__call__' in e and 'staticmethod' in e for e in self.discovery.errors.errors
		), self.discovery.errors.errors )

	def test_call_dunder_explicit_attribute_spelling_still_works( self ) -> None:
		# T.__call__(...), spelled out explicitly rather than via T(...)
		# sugar, must resolve identically (exercises the early-return guard
		# for an already-Attribute callee in the rewrite pre-pass)
		code = '\n'.join([
			'@cstruct',
			'class Converter:',
			'	@staticmethod',
			'	def __call__( x: i32 ) -> i32:',
			'		return x',
			'',
			'def main() -> i32:',
			'	return Converter.__call__( 41 )',
		])
		mod = self._import( code )
		converter_cls = mod.get_local( 'Converter' )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		call_instr = next( i for i in fn.instructions if isinstance( i, ir.Call ))
		self.assertIs( call_instr.target, converter_cls.get_local( '__call__' ))

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
		# UnionName.Member(value) is now an ordinary call to a real,
		# synthesized @staticmethod constructor (see union_storage.py's
		# _build_member_constructor) - the tag/data Allocates live in THAT
		# function's own body, not inline at the call site
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
		main_fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		foo_cls = self.discovery.modules['__test__'].get_local( 'Foo' )
		calls = [ i for i in main_fn.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( len( calls ), 1 )
		ctor_lf = self.compiler._lower( calls[0].target )
		allocates = [ i for i in ctor_lf.instructions if isinstance( i, ir.Allocate ) ]
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
		main_fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		calls = [ i for i in main_fn.instructions if isinstance( i, ir.Call ) ]
		ctor_lf = self.compiler._lower( calls[0].target )
		union_alloc = next( i for i in ctor_lf.instructions if isinstance( i, ir.Allocate ) and i.cls.stem == 'Foo' )
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
		self.assertIn( 'too many positional arguments', self.discovery.errors.errors[0] )

	def test_missing_required_positional_argument_rejected( self ) -> None:
		code = '\n'.join([
			'def foo( a: i32, b: i32 ) -> i32:',
			'	return a',
			'',
			'def main() -> None:',
			'	x: i32 = foo( 1 )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'missing required argument', self.discovery.errors.errors[0] )
		self.assertIn( "'b'", self.discovery.errors.errors[0] )

	def test_missing_required_keyword_only_argument_rejected( self ) -> None:
		code = '\n'.join([
			'def foo( a: i32, *, b: i32 ) -> i32:',
			'	return a',
			'',
			'def main() -> None:',
			'	x: i32 = foo( 1 )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'missing required argument', self.discovery.errors.errors[0] )
		self.assertIn( "'b'", self.discovery.errors.errors[0] )

	def test_defaulted_argument_omission_is_not_flagged_as_missing( self ) -> None:
		code = '\n'.join([
			'def foo( a: i32, b: i32 = 2 ) -> i32:',
			'	return a',
			'',
			'def main() -> None:',
			'	x: i32 = foo( 1 )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_union_storage_is_memoized_across_constructions( self ) -> None:
		# construction elsewhere in the same function (or a different one)
		# must reference the SAME synthesized tag/data/payload-class objects
		# - both calls resolve to the exact same constructor Function too
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
		main_fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		calls = [ i for i in main_fn.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( len( calls ), 2 )
		self.assertIs( calls[0].target, calls[1].target )
		ctor_lf = self.compiler._lower( calls[0].target )
		payload_allocs = [ i for i in ctor_lf.instructions if isinstance( i, ir.Allocate ) and i.cls.stem == 'Foo$data' ]
		self.assertEqual( len( payload_allocs ), 1 )

	# --- direct @cunion construction (bare, not through a @union's own
	# synthesized per-member constructor) - a @cunion can never declare
	# __init__, so this always goes through _lower_allocate_fields's own
	# "exactly one field" validation (the isinstance(target_cls, CUnion)
	# branch), reachable from ordinary user code, not just union_storage.py's
	# own internal payload-class synthesis. Previously only exercised
	# indirectly through @union's own constructors (the tests just above) -
	# these hit the validation directly

	def test_cunion_construct_with_one_field_succeeds( self ) -> None:
		code = '\n'.join([
			'@cunion',
			'class Payload:',
			'	a: i32',
			'	b: f32',
			'',
			'def main() -> None:',
			'	p: Payload = Payload( a = 1 )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_cunion_construct_with_two_fields_is_rejected( self ) -> None:
		code = '\n'.join([
			'@cunion',
			'class Payload:',
			'	a: i32',
			'	b: f32',
			'',
			'def main() -> None:',
			'	p: Payload = Payload( a = 1, b = 2.0 )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'takes exactly one field', self.discovery.errors.errors[0] )

	def test_cunion_construct_with_no_fields_is_rejected( self ) -> None:
		code = '\n'.join([
			'@cunion',
			'class Payload:',
			'	a: i32',
			'	b: f32',
			'',
			'def main() -> None:',
			'	p: Payload = Payload()',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'takes exactly one field', self.discovery.errors.errors[0] )

	def test_cunion_construct_with_unknown_field_is_rejected( self ) -> None:
		code = '\n'.join([
			'@cunion',
			'class Payload:',
			'	a: i32',
			'	b: f32',
			'',
			'def main() -> None:',
			'	p: Payload = Payload( c = 1 )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'has no field(s)', self.discovery.errors.errors[0] )
		self.assertIn( 'c', self.discovery.errors.errors[0] )

	# --- attributes / subscripts --------------------------------------------

	def test_getattr_setattr( self ) -> None:
		code = '\n'.join([
			'class Foo:',
			'	x: i32',
			'',
			'def main( f: Foo ) -> None:',
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
		if self.discovery.main.resolve is not None:
			self.discovery.main.resolve()
		f = self.discovery.main.parameters[0]
		y = Variable( stem = 'y', qualname = 'main.y', file = Path( '__test__.py' ), line = 6, type = i32 )
		t0 = ir.Temp( type = i32, id = 0 )

		fn = self._lower_main()
		self._assert_ir( fn, [
			ir.FuncStart( name = 'main', params = [ f ], return_type = none_type ),
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
			'def main( c: Container ) -> None:',
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
		if self.discovery.main.resolve is not None:
			self.discovery.main.resolve()
		c = self.discovery.main.parameters[0]
		i = Variable( stem = 'i', qualname = 'main.i', file = Path( '__test__.py' ), line = 5, type = usize )
		j = Variable( stem = 'j', qualname = 'main.j', file = Path( '__test__.py' ), line = 6, type = i32 )
		x = Variable( stem = 'x', qualname = 'main.x', file = Path( '__test__.py' ), line = 8, type = i32 )
		t0 = ir.Temp( type = i32, id = 0 )

		fn = self._lower_main()
		self._assert_ir( fn, [
			ir.FuncStart( name = 'main', params = [ c ], return_type = none_type ),
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

	# --- tuple[...] value construction / constant-index access (PLAN_TUPLE.md) ---

	def test_tuple_type_interning( self ) -> None:
		# two annotations spelling the same element-type list share one
		# TupleType object - mirrors CallableType's own interning
		# (discovery.py's _get_or_create_callable_type), so identity-based
		# comparisons elsewhere (e.g. tuple_storage's own reverse lookup)
		# work correctly
		i32 = self.discovery.get_intrinsics()['i32']
		bool_cls = self.discovery.get_intrinsics()['bool']
		tt1 = self.discovery._get_or_create_tuple_type( [ i32, bool_cls ] )
		tt2 = self.discovery._get_or_create_tuple_type( [ i32, bool_cls ] )
		self.assertIs( tt1, tt2 )
		tt3 = self.discovery._get_or_create_tuple_type( [ bool_cls, i32 ] )
		self.assertIsNot( tt1, tt3 ) # different element ORDER is a different tuple type

	def test_tuple_literal_lowers_to_allocate_with_positional_fields( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	t: tuple[i32, bool] = ( 1, True )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		alloc = next( instr for instr in fn.instructions if isinstance( instr, ir.Allocate ))
		self.assertEqual( alloc.cls.qualname, 'tuple[intrinsics.i32,intrinsics.bool]' )
		self.assertEqual( set( alloc.fields.keys() ), { '_0', '_1' } )
		self.assertEqual( alloc.fields['_0'].value, 1 )
		self.assertEqual( alloc.fields['_1'].value, True )
		# no synthesized __init__ call anywhere - field=value sugar only,
		# same as any other class with no real __init__ (_lower_allocate_
		# fields' own "no __init__" path)
		self.assertFalse( any( isinstance( instr, ir.Call ) for instr in fn.instructions ))

	def test_tuple_literal_arity_zero_and_one_are_valid( self ) -> None:
		# `()`/`(1,)` are both unambiguous at the AST level - Python's own
		# parser never confuses either with a plain parenthesized expression
		# (only a genuine trailing comma or empty parens produce a real
		# ast.Tuple node at all) - previously rejected outright, now first-
		# class tuple types like any other arity
		code = '\n'.join([
			'def main() -> None:',
			'	e = ()',
			'	one = ( 1, )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_tuple_annotation_arity_zero_and_one_are_valid( self ) -> None:
		# `tuple[()]` (empty parens as the single slice element - still an
		# ast.Tuple(elts=[])) and `tuple[i32]` (no comma at all - node.slice
		# is bare i32 itself, never wrapped in ast.Tuple) are the two
		# distinct AST shapes discovery.py's visit_Subscript now recognizes
		code = '\n'.join([
			'def main() -> None:',
			'	e: tuple[()] = ()',
			'	one: tuple[i32] = ( 1, )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_constant_index_lowers_to_getattr( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	t: tuple[i32, bool] = ( 1, True )',
			'	x: i32 = t[0]',
			'	y: bool = t[1]',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		getattrs = [ instr for instr in fn.instructions if isinstance( instr, ir.GetAttr ) ]
		self.assertEqual( [ g.attr for g in getattrs ], [ '_0', '_1' ] )
		# no __getitem__ call/GetItem opcode - constant-index access is
		# plain attribute access, resolved entirely at lowering time
		self.assertFalse( any( isinstance( instr, ( ir.Call, ir.GetItem )) for instr in fn.instructions ))

	def test_out_of_range_constant_index_is_rejected( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	t: tuple[i32, bool] = ( 1, True )',
			'	x: i32 = t[2]',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'out of range', self.discovery.errors.errors[0] )

	def test_non_constant_index_is_rejected( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	t: tuple[i32, bool] = ( 1, True )',
			'	i: usize = 0',
			'	x: i32 = t[i]',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'compile-time-constant integer index', self.discovery.errors.errors[0] )

	def test_two_distinct_tuple_shapes_never_share_a_backing_class( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	a: tuple[i32, bool] = ( 1, True )',
			'	b: tuple[bool, i32] = ( True, 1 )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		allocs = [ instr for instr in fn.instructions if isinstance( instr, ir.Allocate ) ]
		self.assertEqual( len( allocs ), 2 )
		self.assertIsNot( allocs[0].cls, allocs[1].cls )
		self.assertNotEqual( allocs[0].cls.qualname, allocs[1].cls.qualname )

	def test_tuple_as_nested_explicit_type_argument_to_generic_construction_call( self ) -> None:
		# regression test: tuple[...] (and every other textually-special
		# subscript form - move/copy/Callable/Closure/Iterator/Generator)
		# used to fail with "name 'tuple' is not defined" when it appeared as
		# a NESTED, EXPLICIT type argument to another generic class's own
		# constructor CALL - Box[tuple[i32,i32]](...) - even though the exact
		# same tuple[tuple[i32,i32]] shape resolves fine as a plain
		# ANNOTATION. Root cause: type_resolver.py's own _try_resolve_
		# namespace (the generic-construction-call counterpart to discovery.
		# py's visit_Subscript, used by Lowering._try_lower_construct_call)
		# only special-cased Callable/Closure textually, so a nested tuple[...]
		# argument fell through to the "ordinary generic base" branch and
		# tried (and failed) to find_name('tuple') as if it were a real,
		# registered class - see type_resolver.py's _try_resolve_namespace
		# for the fix, which now delegates every textually-special subscript
		# form to discovery.py's own visit_Subscript directly
		code = '\n'.join([
			'class Box[T]:',
			'	value: T',
			'	def __init__( self, value: T ) -> None:',
			'		self.value = value',
			'',
			'def main() -> None:',
			'	b = Box[tuple[i32,i32]]( ( 1, 2 ) )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

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

	def test_writes_rc_module_global_wraps_decref_and_assign_in_one_lock( self ) -> None:
		# PLAN_THREAD_SAFE_SHARED_STATE.md Part A: `global G; G = Box()`
		# must bracket BOTH the Decref of G's current value AND the Assign
		# that overwrites it inside one AcquireGlobalLock/ReleaseGlobalLock
		# pair, never two separate ones - releasing between them would let
		# a concurrent reader's own Incref interleave in the gap (see that
		# plan's own worked reader-vs-writer interleaving). Also confirms
		# reassigned_outside_init flips True as a side effect - never for a
		# global's own module-level initializer (Box() at `G: Box = Box()`
		# above is lowered by lower_global(), which never touches
		# cfg.assign() at all - see that flag's own comment).
		code = '\n'.join([
			'class Box:',
			'	def __init__( self ) -> None:',
			'		pass',
			'',
			'G: Box = Box()',
			'',
			'def main() -> None:',
			'	global G',
			'	G = Box()',
			'	return',
		])
		mod = self._import( code )
		none_type = self.discovery.get_none_type()
		box_cls = mod.get_local( 'Box' )
		g = mod.get_local( 'G' )
		if g.resolve is not None:
			g.resolve()
		self.assertFalse( g.reassigned_outside_init ) # not yet, before main() itself is lowered

		fn = self._lower_main()
		self.assertTrue( g.reassigned_outside_init )
		# $$__new__ is only synthesized/registered once something actually
		# constructs a Box - can't look it up until after _lower_main()
		new_fn = box_cls.get_local( '$$__new__' )
		t0 = ir.Temp( type = box_cls, id = 0 )
		self._assert_ir( fn, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.DeclareTemp( temp = t0 ),
			ir.Call( dest = t0, target = new_fn, receiver = None, args = [], kwargs = {} ),
			ir.AcquireGlobalLock( var = g ),
			ir.Decref( value = g ),
			ir.Assign( dest = g, src = t0 ),
			ir.ReleaseGlobalLock( var = g ),
			ir.DeleteTemp( temp = t0 ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_reads_rc_module_global_wraps_incref_and_assign_in_one_lock( self ) -> None:
		# the read-side counterpart - `x: Box = G` must bracket BOTH the
		# Incref of G's current value AND the Assign that binds it into x
		# inside one lock pair. Confirmed necessary the hard way: an
		# earlier version closed the lock right after the Incref, before
		# emitting the Assign - under real concurrent stress this let a
		# writer swap G in the gap, so the Assign's own SEPARATE read of G
		# (its own `x = G;` in the generated C) could bind a DIFFERENT
		# object than the one the Incref just retained.
		code = '\n'.join([
			'class Box:',
			'	def __init__( self ) -> None:',
			'		pass',
			'',
			'G: Box = Box()',
			'',
			'def touch() -> None:',
			'	global G',
			'	G = Box()',
			'',
			'def main() -> None:',
			'	x: Box = G',
			'	return',
		])
		mod = self._import( code )
		box_cls = mod.get_local( 'Box' )
		none_type = self.discovery.get_none_type()
		g = mod.get_local( 'G' )
		if g.resolve is not None:
			g.resolve()
		touch = mod.get_local( 'touch' )
		if touch.resolve is not None:
			touch.resolve()
		# force g.reassigned_outside_init True BEFORE lowering main(), the
		# same way it would be if `touch` happened to be lowered first in a
		# real compile (functions are lowered off a work queue, in
		# whatever order they're scheduled - not necessarily the order a
		# human reads the source in; the read side has to emit these
		# markers unconditionally for exactly this reason, see cfg.py's
		# own `is_alias` branch comment)
		self.compiler._lower( touch )
		self.assertTrue( g.reassigned_outside_init )
		x = Variable( stem = 'x', qualname = 'main.x', file = Path( '__test__.py' ), line = 12, type = box_cls )

		fn = self._lower_main()
		self._assert_ir( fn, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.AcquireGlobalLock( var = g ),
			ir.Incref( value = g ),
			ir.Assign( dest = x, src = g ),
			ir.ReleaseGlobalLock( var = g ),
			ir.Jump( target = '__epilogue_0__' ),
			ir.Label( name = '__epilogue_0__' ),
			ir.Decref( value = x ),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_writes_scalar_module_global_gets_no_lock( self ) -> None:
		# regression guard for a real bug found while building this
		# mechanism: cfg.py's assign() early-returns before ever reaching
		# its own is_global/is_alias branches when the destination type
		# has no RC leaves, so no AcquireGlobalLock is ever emitted for a
		# scalar global - an earlier version of the read-side release
		# didn't mirror that early-return and emitted an orphaned
		# ReleaseGlobalLock with no matching Acquire (caught by
		# test_reads_module_global above failing unexpectedly).
		code = '\n'.join([
			'G: i32 = 5',
			'',
			'def main() -> None:',
			'	global G',
			'	G = 6',
			'	return',
		])
		mod = self._import( code )
		i32 = self.discovery.get_intrinsics()['i32']
		none_type = self.discovery.get_none_type()
		g = mod.get_local( 'G' )
		if g.resolve is not None:
			g.resolve()

		fn = self._lower_main()
		self.assertFalse( g.reassigned_outside_init )
		self._assert_ir( fn, [
			ir.FuncStart( name = 'main', params = [], return_type = none_type ),
			ir.Assign( dest = g, src = ir.Const( type = i32, value = 6 )),
			ir.Return( value = None ),
			ir.FuncEnd( name = 'main' ),
		])

	def test_narrowed_read_of_protected_union_global_wraps_extraction_and_incref_in_one_lock( self ) -> None:
		# PLAN_THREAD_SAFE_SHARED_STATE.md Part A's hardest case: once
		# narrowing has proven `G` (a Box|None global) is non-None, reading
		# it (`b: Box = G`) used to extract the payload via a bare GetAttr
		# sequence with NO lock at all - _expr_Name's narrowed-read rewrite
		# is a separate code path from cfg.assign()'s own is_alias branch,
		# which the direct-read tests above exercise instead. Confirmed via
		# a real crash under concurrent stress that protecting only the
		# eventual Incref wasn't enough either - the extraction itself
		# (reading the union's .data/.v_Box fields) has to be inside the
		# SAME critical section as the retain, or a concurrent writer can
		# free the object between the two. This test doesn't hand-construct
		# the exact IR (the narrowed if/else control flow around it is
		# incidental, not what's being tested, and brittle to match
		# exactly) - it checks the one property that actually matters:
		# exactly one Acquire/Release pair for G's write (inside
		# `if G is None:`) and one more for the narrowed read, and that
		# second pair genuinely brackets both GetAttrs AND the Incref, not
		# just the Incref alone.
		self.discovery.import_name( 'builtins' )
		code = '\n'.join([
			'class Box:',
			'	def __init__( self ) -> None:',
			'		pass',
			'',
			'G: Box|None = None',
			'',
			'def touch() -> None:',
			'	global G',
			'	G = Box()',
			'',
			'def main() -> Box:',
			'	global G',
			'	if G is None:',
			'		G = Box()',
			'	b: Box = G',
			'	return b',
		])
		mod = self._import( code )
		g = mod.get_local( 'G' )
		if g.resolve is not None:
			g.resolve()
		touch = mod.get_local( 'touch' )
		if touch.resolve is not None:
			touch.resolve()
		self.compiler._lower( touch ) # force g.reassigned_outside_init True before main - see the direct-read test's own comment for why this matters
		self.assertTrue( g.reassigned_outside_init )

		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		acquire_idxs = [ i for i, k in enumerate( kinds ) if k == 'AcquireGlobalLock' ]
		release_idxs = [ i for i, k in enumerate( kinds ) if k == 'ReleaseGlobalLock' ]
		self.assertEqual( len( acquire_idxs ), 2, kinds ) # one for the write inside `if G is None:`, one for the narrowed read
		self.assertEqual( len( release_idxs ), 2, kinds )
		self.assertTrue( all( fn.instructions[i].var is g for i in acquire_idxs ))
		self.assertTrue( all( fn.instructions[i].var is g for i in release_idxs ))
		last_acquire, last_release = acquire_idxs[-1], release_idxs[-1]
		self.assertLess( last_acquire, last_release, kinds )
		bracketed = kinds[ last_acquire : last_release + 1 ]
		self.assertIn( 'Incref', bracketed, kinds ) # the retain, not just the extraction
		self.assertEqual( bracketed.count( 'GetAttr' ), 2, kinds ) # .data, then .v_Box

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
			'	x: int = int()',
			'	foo( x )',
			'	return',
		])
		mod = self._import( code )
		int_cls = mod.get_local( 'int' )
		group = mod.get_local( 'foo' )
		int_impl = group.implementations[0]
		if int_impl.resolve is not None:
			int_impl.resolve()

		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		# 2 calls, not 1: x's own plain `int` type doesn't match int_impl's
		# real declared parameter type (int|None, a union) - x must first be
		# coerced into it via the union's own synthesized member constructor
		# (mirrors test_leaf_value_coerced_into_union_via_synthesized_
		# constructor's identical shape for an ordinary, non-overloaded
		# call), THEN the real, unconditional call to int_impl runs with the
		# coerced value. Before this fix, this exact case (a non-literal
		# argument whose plain type is a LEAF of an overloaded call's
		# winning target's own union-typed parameter) skipped that coercion
		# entirely - confirmed via a real compile producing a genuine
		# "passing 'int32_t' to parameter of incompatible type 'struct
		# $__u$$...'" C mismatch for the equivalent real-builtins shape
		calls = [ instr for instr in fn.instructions if isinstance( instr, ir.Call ) ]
		self.assertEqual( len( calls ), 2 )
		coerce_call, real_call = calls
		self.assertEqual( coerce_call.target.stem, 'int' )
		self.assertEqual( len( coerce_call.args ), 1 )
		self.assertEqual( coerce_call.args[0].stem, 'x' )
		self.assertIs( coerce_call.args[0].type, int_cls )
		self.assertIs( real_call.target, int_impl )
		self.assertEqual( real_call.args, [ coerce_call.dest ] )

	def test_overload_call_on_generic_class_specialization_substitutes_class_type_params( self ) -> None:
		# regression test: an @overload group declared inside a generic
		# class (e.g. builtins.Result[T,E].unwrap_or's own `default: T`
		# stub) must have the class's own type params substituted before
		# candidate matching - overload_resolution.py's resolve_call is a
		# pure function of types with no substitution logic of its own, so
		# without this a real, concrete call-site argument type (i32) is
		# compared directly against the abstract stub's own bare TypeVar T
		# and never matches, failing with "no matching overload" even
		# though it should resolve cleanly once T is bound to the
		# receiver's own concrete specialization (mirrors
		# _lower_class_generic_method_call's identical receiver-pins-a-
		# specialization check for a single, non-overloaded generic method)
		code = '\n'.join([
			'@union',
			'class Box[T]:',
			'	Some: T',
			'',
			'	@overload',
			'	def get_or( self, default: T ) -> T:',
			'		...',
			'	def get_or( self, default: T ) -> T:',
			'		return self.data.v_Some',
			'',
			'def main() -> None:',
			'	b: Box[i32] = Box.Some( 5 )',
			'	fallback: i32 = -1',
			'	w: i32 = b.get_or( fallback )',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) and getattr( i.target, 'stem', None ) == 'get_or' ]
		self.assertEqual( len( calls ), 1 )
		target = calls[0].target
		i32 = self.discovery.get_intrinsics()['i32']
		self.assertIs( target.parameters[0].type, i32 ) # substituted, not the abstract TypeVar T
		self.assertIs( target.return_type, i32 )

	def test_overload_call_with_no_argument_uses_the_impls_own_wider_return_type( self ) -> None:
		# regression test: a real, confirmed bug in builtins.Result[T,E].
		# unwrap_or() - calling it with NO argument (its own `default: T|
		# None = None` fallback) resolved to the STUB's `-> T` return type
		# instead of the plain implementation's own wider `-> T|None`,
		# because _lower_call's _resolve_original narrowed the return type
		# to whatever stub happened to be bound_to the winning
		# implementation, unconditionally - regardless of whether THIS
		# call's own arguments actually matched the stub's narrower
		# signature. A stub is bound_to its implementation as a static,
		# always-true fact (`default: T` binds to `default: T|None = None`
		# here), but a zero-argument call can only ever satisfy the
		# IMPLEMENTATION's own broader signature - the stub itself requires
		# `default`, so a call passing none of it can never match the
		# stub's own domain at all (see overload_resolution.py's own
		# _translate_indices). Fixed via overload_resolution.
		# stub_covers_call, which re-checks the call's real argument types
		# against the stub before narrowing - a one-argument call (which
		# DOES match the stub) still correctly narrows to T, covered by
		# this same file's test_overload_call_on_generic_class_
		# specialization_substitutes_class_type_params just above.
		code = '\n'.join([
			'@union',
			'class Box[T]:',
			'	Some: T',
			'',
			'	@overload',
			'	def get_or( self, default: T ) -> T:',
			'		...',
			'	def get_or( self, default: T|None = None ) -> T|None:',
			'		return default',
			'',
			'def main() -> None:',
			'	b: Box[i32] = Box.Some( 5 )',
			'	w = b.get_or()',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) and getattr( i.target, 'stem', None ) == 'get_or' ]
		self.assertEqual( len( calls ), 1 )
		target = calls[0].target
		i32 = self.discovery.get_intrinsics()['i32']
		none_type = self.discovery.get_none_type()
		self.assertIsInstance( target.return_type, TaggedUnion ) # T|None, NOT narrowed down to bare T
		leaf_types = [ attr.type for attr in target.return_type.attributes ]
		self.assertEqual( len( leaf_types ), 2 )
		self.assertTrue( any( t is i32 for t in leaf_types ))
		self.assertTrue( any( t is none_type for t in leaf_types ))
		# the call site's own omitted `default` argument must still be
		# filled in with the impl's own None default - a separate gap this
		# same fix closes (the Overload dispatch path never filled in
		# defaults for parameters the call site didn't supply at all,
		# unlike the plain, non-Overload call path's _lower_call_args)
		self.assertIn( 'default', calls[0].kwargs )

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
		self.discovery.import_name( 'builtins' ) # range()'s own hidden bound check is now an ordinary i32.__lt__ dunder call
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

	def test_return_rejected_inside_a_defer_body( self ) -> None:
		code = '\n'.join([
			'class bool: pass',
			'',
			'def main() -> None:',
			'	with defer:',
			'		return',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any( 'return is not allowed inside a defer/errdefer body' in e for e in self.discovery.errors.errors ))

	def test_return_rejected_inside_a_defer_body_even_when_nested( self ) -> None:
		# _in_deferred_body stays set for the whole capture, not just the
		# top-level statement - a return buried inside an if inside the
		# defer body must be caught too
		code = '\n'.join([
			'class bool: pass',
			'',
			'def main( cond: bool ) -> None:',
			'	with defer:',
			'		if cond:',
			'			return',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any( 'return is not allowed inside a defer/errdefer body' in e for e in self.discovery.errors.errors ))

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
			ir.Jump( target = '__epilogue_0__' ),
			ir.Label( name = '__epilogue_0__' ),
			ir.JumpIfFalse( cond = flag0, target = '__defer_skip_1__' ),
			ir.Call( dest = None, target = cleanup_fn, args = [], kwargs = {} ),
			ir.Label( name = '__defer_skip_1__' ),
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
			# into the epilogue - no Jump needed, it's placed right after.
			# No Label either: nothing else in this function ever needs to
			# goto this exact depth, so build_epilogue_ladder() correctly
			# omits it (a Label with nothing branching to it is a real
			# -Wunused-label/C4102 on every C compiler) - see cfg.py's own
			# comment on entry.captured
			ir.JumpIfFalse( cond = flag0, target = '__defer_skip_1__' ),
			ir.Call( dest = None, target = cleanup_fn, args = [], kwargs = {} ),
			ir.Label( name = '__defer_skip_1__' ),
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
			'	return Result()',
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
		# t0: Result()'s own Allocate (the explicit trailing return's own
		# value - MUST be real: a Result-returning function relying purely
		# on implicit fallthrough with no real return value is exactly the
		# uninitialized-__return_value bug class this compiler now rejects
		# outright - see lib/builtins/__list.py's own list.insert() fix and
		# msvc_toolset_c11atomics memory). is_err_temp is id=1, not 0 - it's
		# allocated AFTER t0 has already been declared/assigned/deleted
		t0 = ir.Temp( type = checked_fn.return_type, id = 0 )
		is_err_temp = ir.Temp( type = bool_cls, id = 1 )
		# is_err's genericity is inherited from Result's own class type
		# params (like Result.Ok/.Err) - the receiver's type (Result[None,
		# OverflowError]) already pins down the concrete args by the time
		# _build_is_err_check runs, so the call target is the MONOMORPHIZED
		# copy, not the bare is_err_fn (which has no real C struct body
		# anywhere - only concrete specializations do)
		is_err_spec = self.discovery._get_or_create_specialization( is_err_fn, checked_fn.return_type.args )
		monomorphized_is_err = self.compiler.lowering._monomorphized_function( is_err_spec )

		fn = self.compiler._lower( checked_fn )
		self._assert_ir( fn, [
			ir.FuncStart( name = '__test__.checked', params = [], return_type = checked_fn.return_type ),
			ir.Assign( dest = flag0, src = ir.Const( type = bool_cls, value = False )),
			ir.Assign( dest = flag0, src = ir.Const( type = bool_cls, value = True )),
			ir.DeclareTemp( temp = t0 ),
			ir.Allocate( dest = t0, cls = result_cls, fields = {} ),
			ir.Assign( dest = return_value_var, src = t0 ),
			ir.DeleteTemp( temp = t0 ),
			# an EXPLICIT `return Result()` (needed - see t0's own comment
			# above) reaches the SAME shared epilogue a pure fall-off-the-end
			# would have - unlike that pure-fallthrough case (see
			# test_noreturn_function_epilogue_has_no_return_value_var's
			# identical comment on when no Label/Jump is needed), an
			# explicit return mid-body genuinely needs a real goto to reach
			# it, so both the Jump and its Label are real here
			ir.Jump( target = '__epilogue_0__' ),
			ir.Label( name = '__epilogue_0__' ),
			ir.JumpIfFalse( cond = flag0, target = '__defer_skip_1__' ),
			# the is_err() check is computed fresh, INSIDE the flag guard -
			# with per-Epilogue labels a check computed once up front
			# wouldn't be reached by every jump that might land elsewhere in
			# the ladder (see cfg.py's _replay()), and skipping it entirely
			# when the flag never armed is a nice side benefit
			ir.DeclareTemp( temp = is_err_temp ),
			ir.Call( dest = is_err_temp, target = monomorphized_is_err, receiver = return_value_var, args = [], kwargs = {} ),
			ir.JumpIfFalse( cond = is_err_temp, target = '__defer_skip_1__' ),
			ir.Label( name = '__defer_skip_1__' ),
			# this test's own `class bool: pass` fixture is a plain
			# (RCClass) class, same as any undecorated class - is_err_temp
			# is genuinely fresh_temp()-tracked and gets its own Decref here,
			# unrelated to the real intrinsic bool used everywhere else
			ir.Decref( value = is_err_temp ),
			ir.DeleteTemp( temp = is_err_temp ),
			ir.Return( value = return_value_var ),
			ir.FuncEnd( name = '__test__.checked' ),
		])

	def test_only_use_cfg_epilogue_labels( self ):
		# n (a real Scalar) carries the branch conditions; a (the custom RCClass)
		# is purely for exercising RC-tracked-binding epilogue labels via
		# b = a / c = a below - kept separate because comparing `a` (an RCClass
		# with no __gt__ of its own) directly against a bare int literal would
		# itself be a type mismatch, unrelated to what this test is about
		code = '\n'.join([
			'class int:',
			'	def __init__( self, n: usize ) -> None:',
			'		...',
			'def foo( a: int, n: usize ) -> None:',
			'	if n > 10:',
			'		return', # should be a straight return, no epilogue yet
			'	b = a',
			'	if n > 20:',
			'		return', # should jump to b's decref epilogue label
			'	c = a',
			# fallthough return should jump to c's decref epilogue label
			'',
			'def main() -> None:',
			'	foo( usize( 0 ), usize( 0 ))',
		])
		self.discovery.import_name( 'builtins' ) # n > 10/20 is now an ordinary usize.__gt__ dunder call
		mod = self._import( code )
		foo = mod.get_local( 'foo' )
		lfoo = self.compiler._lower( foo )
		# Label is filtered down to epilogue labels specifically - `if`
		# lowering emits its own '__if_else_N__'/'__if_end_N__' Labels from
		# the SAME shared label counter (interleaved with the epilogue's own
		# '__epilogue_N__' ones), which aren't what this test is about
		got = [
			op for op in lfoo.instructions
			if ( isinstance( op, ir.Label ) and op.name.startswith( '__epilogue' ))
			or isinstance( op, ir.Jump )
			or isinstance( op, ir.Return )
		]
		self.assertEqual( got, [
			ir.Return( value = None ), # straight return, no epilogue yet
			ir.Jump( target = '__epilogue_1__' ), # jumps straight to b's own cleanup, skipping c's (not alive yet on this path)
			# no Label for c's own rung (__epilogue_3__) - nothing ever
			# jumps there (only the fall-off-the-end path reaches it, via
			# pure fallthrough from the Jump target above), so it's
			# correctly omitted - see cfg.py's entry.captured
			ir.Label( name = '__epilogue_1__' ), # clean up b - shared with the early return above
			ir.Return( value = None ), # the function's one real Return, reached by fall-off-the-end
		])

	def test_errdefer_with_checked_arithmetic_emits_or_jump( self ) -> None:
		# real builtins.Result already has its own is_err() - see
		# test_binop_check_mode_emits_or_return's comment on why `a + 1`'s
		# own Result/OverflowError must be the REAL builtins ones, not a
		# hand-rolled local stand-in
		self.discovery.import_name( 'builtins' )
		code = '\n'.join([
			'from builtins import Result, OverflowError',
			'',
			'def checked() -> Result[None,OverflowError]:',
			'	with errdefer:',
			'		pass',
			'	a: i32 = 1',
			'	b: i32 = a + 1',
			'	return Result.Ok( None )',
		])
		mod = self._import( code )
		checked_fn = mod.get_local( 'checked' )
		if checked_fn.resolve is not None:
			checked_fn.resolve()
		fn = self.compiler._lower( checked_fn )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertIn( 'AddCheck', kinds )
		# auto-inserted (no enclosing try here) or_throw() targets the same
		# epilogue label OrJump used to (see ir.OrThrow's own target field) -
		# general auto-or_throw() rule, see _auto_or_throw
		self.assertIn( 'OrThrow', kinds )
		self.assertNotIn( 'OrReturn', kinds )
		# epilogue's errdefer guard is the two-JumpIfFalse (flag, then
		# is_err()) shape - the is_err() check itself (DeclareTemp+Call)
		# sits between them, computed fresh inside the flag guard rather
		# than shared/hoisted (see cfg.py's _replay())
		jump_if_false_indices = [ i for i, instr in enumerate( fn.instructions ) if isinstance( instr, ir.JumpIfFalse ) ]
		self.assertEqual( len( jump_if_false_indices ), 2 )
		between = fn.instructions[jump_if_false_indices[0] + 1:jump_if_false_indices[1]]
		self.assertEqual( [ type( instr ).__name__ for instr in between ], ['DeclareTemp', 'Call'] )

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
			'	return Result()',
			'',
			'def checked_call() -> Result[None,OverflowError]:',
			'	errdefer( cleanup() )',
			'	return Result()',
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
		self.discovery.import_name( 'builtins' )
		code = '\n'.join([
			'from builtins import Result, OverflowError',
			'',
			'def checked() -> Result[None,OverflowError]:',
			'	a: i32 = 1',
			'	b: i32 = a + 1',
			'	return Result.Ok( None )',
		])
		mod = self._import( code )
		checked_fn = mod.get_local( 'checked' )
		if checked_fn.resolve is not None:
			checked_fn.resolve()
		fn = self.compiler._lower( checked_fn )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		# auto-inserted (no enclosing try here) - degrades to exactly
		# or_return()'s own semantics (a real C `return`, target=None) -
		# general auto-or_throw() rule, see _auto_or_throw
		self.assertIn( 'OrThrow', kinds )
		self.assertNotIn( 'OrReturn', kinds )
		self.assertNotIn( 'OrJump', kinds )
		self.assertNotIn( 'Jump', kinds )
		self.assertNotIn( 'Label', kinds )

	# --- fresh RC value leak (fresh_temp()/delete_temp() integration) --------

	def test_fresh_rc_value_passed_as_plain_argument_gets_decrefd( self ) -> None:
		# foo( SomeClass() ) - the fresh temp SomeClass() produces is never
		# assigned to a name, returned, moved, or embedded in a field (use's
		# own parameter is plain, not move[T]) - it still needs its own
		# decref right where its expression-scoped lifetime naturally ends
		code = '\n'.join([
			'class Foo: pass',
			'',
			'def use( x: Foo ) -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	use( Foo() )',
		])
		mod = self._import( code )
		fn = mod.get_local( 'main' )
		if fn.resolve is not None:
			fn.resolve()
		lowered = self.compiler._lower( fn )
		kinds = [ type( instr ).__name__ for instr in lowered.instructions ]
		self.assertEqual( kinds.count( 'Decref' ), 1 )
		# the Decref must land right before the temp's own DeleteTemp -
		# the Call itself (using the still-live temp as an argument) comes
		# first
		decref_i = kinds.index( 'Decref' )
		self.assertEqual( kinds[decref_i - 1], 'Call' )
		self.assertEqual( kinds[decref_i + 1], 'DeleteTemp' )

	def test_fresh_rc_value_assigned_to_a_name_is_not_double_decrefd( self ) -> None:
		code = '\n'.join([
			'class Foo: pass',
			'',
			'def main() -> None:',
			'	x: Foo = Foo()',
			'	print( x )',
		])
		mod = self._import( code )
		fn = mod.get_local( 'main' )
		if fn.resolve is not None:
			fn.resolve()
		lowered = self.compiler._lower( fn )
		kinds = [ type( instr ).__name__ for instr in lowered.instructions ]
		# exactly one Decref (x's own, at the fall-off epilogue) - none for
		# the temp Foo() produced, which assign() already untracks once it's
		# consumed into x
		self.assertEqual( kinds.count( 'Decref' ), 1 )

	def test_fresh_rc_value_embedded_in_a_field_is_not_double_decrefd( self ) -> None:
		code = '\n'.join([
			'class Foo: pass',
			'class Wrapper:',
			'	inner: Foo',
			'',
			'	@staticmethod',
			'	def make() -> None:',
			'		w = Wrapper.__allocate__( inner = Foo() )',
			'		print( w )',
		])
		mod = self._import( code )
		wrapper_cls = mod.get_local( 'Wrapper' )
		if wrapper_cls.resolve is not None:
			wrapper_cls.resolve()
		fn = next( m for m in wrapper_cls.methods if getattr( m, 'stem', None ) == 'make' )
		if fn.resolve is not None:
			fn.resolve()
		lowered = self.compiler._lower( fn )
		kinds = [ type( instr ).__name__ for instr in lowered.instructions ]
		# exactly one Decref (w's own, at the fall-off epilogue) - none for
		# the temp Foo() produced, which field_value() now untracks once
		# it's embedded into inner
		self.assertEqual( kinds.count( 'Decref' ), 1 )

	# --- __init__ construction (RCCLASS ATTRIBUTE LIFETIME.md) ---------------

	def _method( self, mod, cls_name: str, method_name: str ):
		cls = mod.get_local( cls_name )
		if cls.resolve is not None:
			cls.resolve()
		fn = next( m for m in cls.methods if getattr( m, 'stem', None ) == method_name )
		if fn.resolve is not None:
			fn.resolve()
		return fn

	# hand-rolled Result fixture (this file's own Discovery uses
	# import_builtins=False - matches test_match_result_ok_err_shape's own
	# fixture exactly, extended with Err)
	_RESULT_FIXTURE = '\n'.join([
		'class bool: pass',
		'',
		'@union',
		'class Result[T,E]:',
		'	Ok: T',
		'	Err: E',
		'',
		'	def is_ok( self ) -> bool:',
		'		return self.tag == 0',
		'',
		'	def is_err( self ) -> bool:',
		'		return self.tag == 1',
	])

	def test_nonfallible_init_shape( self ) -> None:
		code = '\n'.join([
			'class Foo: pass',
			'class Bar:',
			'	a: Foo',
			'',
			'	def __init__( self, x: Foo ) -> None:',
			'		self.a = x',
			'',
			'def main() -> None:',
			'	b = Bar( Foo() )',
		])
		mod = self._import( code )
		lowered = self.compiler._lower( self._method( mod, 'Bar', '__init__' ))
		kinds = [ type( instr ).__name__ for instr in lowered.instructions ]
		# self.a = x is an aliasing assignment of a plain (non-move)
		# parameter - a real Incref, same as it would be for an ordinary
		# local (mirrors assign()'s own is_alias rule via attr_assign())
		self.assertEqual( kinds, ['FuncStart', 'Incref', 'SetAttr', 'Return', 'FuncEnd'] )
		# Bar(...) itself now collapses to a single call into the
		# synthesized per-class $$__new__ constructor (mirrors the
		# destructor's own single call, though $$__new__ is called
		# directly by name rather than dispatched through the vtable) -
		# no more inline Allocate/header-init/Call at each construction
		# site (see type_resolver.py's _synthesize_rcclass_constructor)
		main_lowered = self.compiler._lower( mod.get_local( 'main' ))
		main_kinds = [ type( instr ).__name__ for instr in main_lowered.instructions ]
		# Foo() still allocates inline (main_kinds legitimately still has
		# ONE Allocate for it - Foo has no own __init__, so it goes
		# through the untouched no-__init__ field=value sugar path,
		# _lower_allocate_fields, not $$__new__ synthesis at all) - only
		# Bar's OWN construction is asserted to have collapsed away
		self.assertFalse( any( type( i ).__name__ == 'Allocate' and i.cls.stem == 'Bar' for i in main_lowered.instructions ))
		self.assertIn( 'Call', main_kinds )
		self.assertNotIn( 'JumpIfFalse', main_kinds ) # no Ok/Err branch for a non-fallible __init__
		# the synthesized $$__new__ itself does the alloc - registered
		# directly into Bar.names (not .methods, unlike a user-declared
		# method), so looked up via get_local rather than _method
		bar_cls = mod.get_local( 'Bar' )
		new_fn = bar_cls.get_local( '$$__new__' )
		new_lowered = self.compiler._lower( new_fn )
		new_kinds = [ type( instr ).__name__ for instr in new_lowered.instructions ]
		self.assertIn( 'Allocate', new_kinds )
		allocate = next( i for i in new_lowered.instructions if type( i ).__name__ == 'Allocate' and i.cls.stem == 'Bar' )
		self.assertEqual( allocate.fields, {} ) # self starts fully uninitialized

	def test_fallible_init_shape_has_ok_err_branches( self ) -> None:
		code = self._RESULT_FIXTURE + '\n' + '\n'.join([
			'class MyError: pass',
			'class Bar:',
			'	a: i32',
			'',
			'	def __init__( self, fail: bool ) -> Result[None,MyError]:',
			'		if fail:',
			'			return Result.Err( MyError() )',
			'		self.a = 1',
			'		return Result.Ok( None )',
			'',
			'def main() -> None:',
			'	r = Bar( True )',
			'	r.is_ok()',
		])
		mod = self._import( code )
		lowered = self.compiler._lower( mod.get_local( 'main' ))
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in lowered.instructions ]
		# Bar(...) itself now collapses to a single call into the
		# synthesized $$__new__ constructor, same as the non-fallible
		# case - main's own instructions no longer contain any Ok/Err
		# branch logic at all, that all moved into $$__new__'s own body
		# (built from ordinary if/return AST now, not hand-spliced raw
		# IR - see type_resolver.py's _synthesize_rcclass_constructor)
		self.assertNotIn( 'JumpIfTrue', kinds )
		self.assertIn( 'Call', kinds )
		bar_cls = mod.get_local( 'Bar' )
		new_fn = bar_cls.get_local( '$$__new__' )
		new_lowered = self.compiler._lower( new_fn )
		new_kinds = [ type( instr ).__name__ for instr in new_lowered.instructions ]
		# ordinary `if result.is_err():` lowers to JumpIfFalse (skip the
		# if-body when false), NOT JumpIfTrue - the polarity the old
		# hand-spliced _emit_fallible_construction used (and, before its
		# own fix, got backwards - see git history) is simply a different
		# implementation detail now that this is ordinary statement
		# lowering, not raw IR
		self.assertIn( 'JumpIfFalse', new_kinds )
		# self decref'd on the OK path (its own original reference dropped
		# once ownership moves into the Ok payload - see
		# _synthesize_rcclass_constructor's own Ok-branch comment). The
		# Err path never decrefs self at all: it frees self's raw
		# allocation directly (compiler.__raw_free__ - CastWrap+Call to
		# sys.free) rather than going through the class's ordinary, shared
		# vtable destructor, since self is only partially constructed there
		self.assertIn( 'Decref', new_kinds )
		self.assertIn( 'CastWrap', new_kinds )

	def test_missing_attribute_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'class Bar:',
			'	a: i32',
			'	b: i32',
			'',
			'	def __init__( self ) -> None:',
			'		self.a = 1',
			'',
			'def main() -> None:',
			'	pass',
		])
		mod = self._import( code )
		with self.assertRaises( CompileError ):
			self.compiler._lower( self._method( mod, 'Bar', '__init__' ))
		self.assertTrue( any( 'must initialize' in e and 'b' in e for e in self.discovery.errors.errors ))

	def test_self_escape_via_method_call_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'class Bar:',
			'	a: i32',
			'',
			'	def helper( self ) -> None:',
			'		pass',
			'',
			'	def __init__( self ) -> None:',
			'		self.helper()',
			'		self.a = 1',
			'',
			'def main() -> None:',
			'	pass',
		])
		mod = self._import( code )
		# unlike complete_construction()'s own CompileError (raised outside
		# the per-statement loop, in lower_function's own fall-off-the-end
		# handling), check_self_escape() fires FROM WITHIN a statement's own
		# lowering (via _emit) - lower_function's per-statement recovery
		# boundary ("one bad statement doesn't stop the rest") catches it,
		# so _lower() itself doesn't raise here - the error is still
		# recorded, just not propagated as an exception
		self.compiler._lower( self._method( mod, 'Bar', '__init__' ))
		self.assertTrue( any( 'self cannot be used here' in e for e in self.discovery.errors.errors ))

	def test_self_escape_via_plain_argument_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'class Bar:',
			'	a: i32',
			'',
			'	def __init__( self ) -> None:',
			'		use( self )',
			'		self.a = 1',
			'',
			'def use( b: Bar ) -> None:',
			'	pass',
			'',
			'def main() -> None:',
			'	pass',
		])
		mod = self._import( code )
		self.compiler._lower( self._method( mod, 'Bar', '__init__' )) # doesn't raise - see the identical comment on test_self_escape_via_method_call_is_a_compile_error
		self.assertTrue( any( 'self cannot be used here' in e for e in self.discovery.errors.errors ))

	def test_default_value_prologue_is_spliced_before_the_body( self ) -> None:
		code = '\n'.join([
			'class Bar:',
			'	a: i32 = 5',
			'	b: i32',
			'',
			'	def __init__( self, x: i32 ) -> None:',
			'		self.b = x',
			'',
			'def main() -> None:',
			'	pass',
		])
		mod = self._import( code )
		lowered = self.compiler._lower( self._method( mod, 'Bar', '__init__' ))
		set_attrs = [ i for i in lowered.instructions if type( i ).__name__ == 'SetAttr' ]
		self.assertEqual( [ i.attr for i in set_attrs ], ['a', 'b'] ) # default prologue first, then the user's own body

	def test_ordinary_post_construction_setattr_is_a_replace( self ) -> None:
		code = '\n'.join([
			'class Foo: pass',
			'class Bar:',
			'	a: Foo',
			'',
			'	def __init__( self, x: Foo ) -> None:',
			'		self.a = x',
			'',
			'	def replace_a( self, y: Foo ) -> None:',
			'		self.a = y',
		])
		mod = self._import( code )
		lowered = self.compiler._lower( self._method( mod, 'Bar', 'replace_a' ))
		kinds = [ type( instr ).__name__ for instr in lowered.instructions ]
		# reads the current value (GetAttr), increfs the new aliasing value,
		# decrefs the old one, then stores - "always a replace" outside __init__
		self.assertEqual(
			[ k for k in kinds if k in ( 'GetAttr', 'Incref', 'Decref', 'SetAttr' ) ],
			['GetAttr', 'Incref', 'Decref', 'SetAttr'],
		)

	def test_result_ok_and_err_lower_cleanly( self ) -> None:
		# regression check for the pre-existing (unrelated to this pass -
		# confirmed via a clean-checkout repro) CUnion "missing field" bug:
		# ResultPayload(ok=val)/ResultPayload(err=err) only ever set ONE
		# member, never both - Result.Ok/Result.Err's own bodies must not
		# require the other
		code = self._RESULT_FIXTURE + '\n' + '\n'.join([
			'class MyError: pass',
			'',
			'def main() -> None:',
			'	ok: Result[i32,MyError] = Result.Ok( 5 )',
			'	err: Result[i32,MyError] = Result.Err( MyError() )',
			'	ok.is_ok()',
			'	err.is_ok()',
		])
		mod = self._import( code )
		self.compiler._lower( mod.get_local( 'main' ))
		self.assertEqual( self.discovery.errors.errors, [] )

	# --- super().__init__(...) constructor chaining (RCClass single inheritance,
	# Phase 2 of the RCClass-subclassing plan) -------------------------------

	def test_super_init_call_shape( self ) -> None:
		code = '\n'.join([
			'class Base:',
			'	x: i32',
			'	def __init__( self, x: i32 ) -> None:',
			'		self.x = x',
			'',
			'class Derived( Base ):',
			'	y: i32',
			'	def __init__( self, x: i32, y: i32 ) -> None:',
			'		super().__init__( x )',
			'		self.y = y',
			'',
			'def main() -> None:',
			'	pass',
		])
		mod = self._import( code )
		lowered = self.compiler._lower( self._method( mod, 'Derived', '__init__' ))
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in lowered.instructions ]
		# the super().__init__(x) call, then self.y = y (SetAttr, no Incref -
		# y: i32 isn't RC-managed) - no Allocate here (self is already
		# allocated by the caller of Derived(...) before __init__ ever
		# runs), just a Call whose receiver is self cast to Base
		self.assertEqual( kinds, [ 'FuncStart', 'Call', 'SetAttr', 'Return', 'FuncEnd' ] )
		call = next( i for i in lowered.instructions if type( i ).__name__ == 'Call' )
		self.assertTrue( call.is_super_init_call )
		self.assertEqual( call.target.stem, '__init__' )
		self.assertEqual( call.target.cls.stem, 'Base' )

	def test_missing_super_init_call_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'class Base:',
			'	x: i32',
			'	def __init__( self, x: i32 ) -> None:',
			'		self.x = x',
			'',
			'class Derived( Base ):',
			'	y: i32',
			'	def __init__( self, x: i32, y: i32 ) -> None:',
			'		self.y = y',
			'',
			'def main() -> None:',
			'	pass',
		])
		mod = self._import( code )
		with self.assertRaises( CompileError ):
			self.compiler._lower( self._method( mod, 'Derived', '__init__' ))
		self.assertTrue( any( 'must call super().__init__' in e for e in self.discovery.errors.errors ))

	def test_super_init_call_not_first_statement_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'class Base:',
			'	x: i32',
			'	def __init__( self, x: i32 ) -> None:',
			'		self.x = x',
			'',
			'class Derived( Base ):',
			'	y: i32',
			'	def __init__( self, x: i32, y: i32 ) -> None:',
			'		self.y = y',
			'		super().__init__( x )',
			'',
			'def main() -> None:',
			'	pass',
		])
		mod = self._import( code )
		with self.assertRaises( CompileError ):
			self.compiler._lower( self._method( mod, 'Derived', '__init__' ))
		self.assertTrue( any( 'must call super().__init__' in e for e in self.discovery.errors.errors ))

	def test_super_init_outside_first_statement_of_init_is_rejected_everywhere( self ) -> None:
		# super().__init__(...) used in a non-__init__ method, or in a root
		# class's own __init__ (no base to chain to at all) - both reach
		# _stmt_Expr's own shape-check (never the special statement-0
		# handling), which rejects unconditionally
		code = '\n'.join([
			'class Root:',
			'	x: i32',
			'	def __init__( self, x: i32 ) -> None:',
			'		super().__init__( x )',
			'		self.x = x',
			'',
			'def main() -> None:',
			'	pass',
		])
		mod = self._import( code )
		# doesn't raise - _stmt_Expr's own check fires from WITHIN this
		# statement's own lowering, caught by lower_function's per-statement
		# recovery boundary, same as test_self_escape_via_method_call_is_a_
		# compile_error's own identical situation
		self.compiler._lower( self._method( mod, 'Root', '__init__' ))
		self.assertTrue( any( 'only allowed as the literal first statement' in e for e in self.discovery.errors.errors ))

	def test_field_only_base_with_no_init_rejects_subclass_init( self ) -> None:
		# a Phase 2 scope limit, not a soundness gap - see
		# _lower_super_init_if_required's own comment: no sugar exists yet
		# for a subclass's own __init__ to fill in a field-only ancestor's
		# fields when there's no base __init__ to chain to at all
		code = '\n'.join([
			'class Base:',
			'	x: i32',
			'',
			'class Derived( Base ):',
			'	y: i32',
			'	def __init__( self, y: i32 ) -> None:',
			'		self.y = y',
			'',
			'def main() -> None:',
			'	pass',
		])
		mod = self._import( code )
		with self.assertRaises( CompileError ):
			self.compiler._lower( self._method( mod, 'Derived', '__init__' ))
		self.assertTrue( any( 'has field(s) but no' in e for e in self.discovery.errors.errors ))

	def test_field_only_base_with_no_fields_needs_no_super_call( self ) -> None:
		# a base with NO __init__ AND no fields at all contributes nothing -
		# no super() call required, ordinary construction-safety applies to
		# just the subclass's own attributes
		code = '\n'.join([
			'class Base:',
			'	def hello( self ) -> i32:',
			'		return 1',
			'',
			'class Derived( Base ):',
			'	y: i32',
			'	def __init__( self, y: i32 ) -> None:',
			'		self.y = y',
			'',
			'def main() -> None:',
			'	pass',
		])
		mod = self._import( code )
		self.compiler._lower( self._method( mod, 'Derived', '__init__' ))
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_fallible_base_init_requires_or_return( self ) -> None:
		code = self._RESULT_FIXTURE + '\n' + '\n'.join([
			'class MyError: pass',
			'class Base:',
			'	x: i32',
			'	def __init__( self, x: i32 ) -> Result[None,MyError]:',
			'		self.x = x',
			'		return Result.Ok( None )',
			'',
			'class Derived( Base ):',
			'	y: i32',
			'	def __init__( self, x: i32, y: i32 ) -> None:',
			'		super().__init__( x )',
			'		self.y = y',
			'',
			'def main() -> None:',
			'	pass',
		])
		mod = self._import( code )
		with self.assertRaises( CompileError ):
			self.compiler._lower( self._method( mod, 'Derived', '__init__' ))
		self.assertTrue( any( 'is fallible - must be consumed via .or_return()' in e for e in self.discovery.errors.errors ))

	def test_infallible_base_init_rejects_or_return( self ) -> None:
		code = '\n'.join([
			'class Base:',
			'	x: i32',
			'	def __init__( self, x: i32 ) -> None:',
			'		self.x = x',
			'',
			'class Derived( Base ):',
			'	y: i32',
			'	def __init__( self, x: i32, y: i32 ) -> None:',
			'		super().__init__( x ).or_return()',
			'		self.y = y',
			'',
			'def main() -> None:',
			'	pass',
		])
		mod = self._import( code )
		with self.assertRaises( CompileError ):
			self.compiler._lower( self._method( mod, 'Derived', '__init__' ))
		self.assertTrue( any( 'is not fallible - remove .or_return()' in e for e in self.discovery.errors.errors ))

	def test_fallible_base_init_with_or_return_shape( self ) -> None:
		code = self._RESULT_FIXTURE + '\n' + '\n'.join([
			'class MyError: pass',
			'class Base:',
			'	x: i32',
			'	def __init__( self, x: i32 ) -> Result[None,MyError]:',
			'		self.x = x',
			'		return Result.Ok( None )',
			'',
			'class Derived( Base ):',
			'	y: i32',
			'	def __init__( self, x: i32, y: i32 ) -> Result[None,MyError]:',
			'		super().__init__( x ).or_return()',
			'		self.y = y',
			'		return Result.Ok( None )',
			'',
			'def main() -> None:',
			'	pass',
		])
		mod = self._import( code )
		lowered = self.compiler._lower( self._method( mod, 'Derived', '__init__' ))
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in lowered.instructions ]
		self.assertIn( 'Call', kinds ) # super().__init__(x)
		self.assertIn( 'OrReturn', kinds ) # or_return()'s own propagation primitive, consuming the super-call's result
		call = next( i for i in lowered.instructions if type( i ).__name__ == 'Call' and i.is_super_init_call )
		self.assertIsNotNone( call.dest ) # fallible - the Result gets a real dest for or_return() to consume

	def test_self_escape_still_enforced_after_super_init_for_own_attributes( self ) -> None:
		# super().__init__(...) satisfies the BASE's own required
		# attributes only - the subclass's own attributes still gate self
		# escape exactly as they would for a root class
		code = '\n'.join([
			'class Base:',
			'	x: i32',
			'	def __init__( self, x: i32 ) -> None:',
			'		self.x = x',
			'',
			'class Derived( Base ):',
			'	y: i32',
			'	def helper( self ) -> None:',
			'		pass',
			'	def __init__( self, x: i32, y: i32 ) -> None:',
			'		super().__init__( x )',
			'		self.helper()',
			'		self.y = y',
			'',
			'def main() -> None:',
			'	pass',
		])
		mod = self._import( code )
		self.compiler._lower( self._method( mod, 'Derived', '__init__' )) # doesn't raise - see test_self_escape_via_method_call_is_a_compile_error's own comment
		self.assertTrue( any( 'self cannot be used here' in e and 'y' in e for e in self.discovery.errors.errors ))

	def test_three_level_chain_flattens_base_attributes( self ) -> None:
		code = '\n'.join([
			'class A:',
			'	a: i32',
			'	def __init__( self, a: i32 ) -> None:',
			'		self.a = a',
			'',
			'class B( A ):',
			'	b: i32',
			'	def __init__( self, a: i32, b: i32 ) -> None:',
			'		super().__init__( a )',
			'		self.b = b',
			'',
			'class C( B ):',
			'	c: i32',
			'	def __init__( self, a: i32, b: i32, c: i32 ) -> None:',
			'		super().__init__( a, b )',
			'		self.c = c',
			'',
			'def main() -> None:',
			'	pass',
		])
		mod = self._import( code )
		lowered = self.compiler._lower( self._method( mod, 'C', '__init__' ))
		self.assertEqual( self.discovery.errors.errors, [] )
		call = next( i for i in lowered.instructions if type( i ).__name__ == 'Call' )
		self.assertTrue( call.is_super_init_call )
		self.assertEqual( call.target.cls.stem, 'B' ) # chains to the IMMEDIATE base's own __init__, not the root's

	# --- local (in-function) imports ---------------------------------------

	def test_from_import_makes_the_name_callable( self ) -> None:
		# discovery.py's own visit_ImportFrom only ever runs at module/class
		# scope (function bodies are deliberately never walked by discovery
		# - see Lowering's own docstring) - this is the in-function form,
		# _stmt_ImportFrom, exercised for real here via a second module
		helper_mod = self.compiler.import_code( 'def helper() -> i32:\n\treturn 42\n', Path( 'helper.py' ))
		code = '\n'.join([
			'def main() -> i32:',
			'	from helper import helper as h',
			'	return h()',
		])
		i32 = self.discovery.get_intrinsics()['i32']
		helper_fn = helper_mod.get_local( 'helper' )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = i32 ),
			ir.DeclareTemp( temp = ir.Temp( type = i32, id = 0 )),
			ir.Call( dest = ir.Temp( type = i32, id = 0 ), target = helper_fn, receiver = None, args = [], kwargs = {} ),
			ir.DeleteTemp( temp = ir.Temp( type = i32, id = 0 )),
			ir.Return( value = ir.Temp( type = i32, id = 0 )),
			ir.FuncEnd( name = 'main' ),
		])

	def test_import_module_makes_it_addressable( self ) -> None:
		helper_mod = self.compiler.import_code( 'def helper() -> i32:\n\treturn 42\n', Path( 'helper.py' ))
		code = '\n'.join([
			'def main() -> i32:',
			'	import helper',
			'	return helper.helper()',
		])
		i32 = self.discovery.get_intrinsics()['i32']
		helper_fn = helper_mod.get_local( 'helper' )
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = i32 ),
			ir.DeclareTemp( temp = ir.Temp( type = i32, id = 0 )),
			ir.Call( dest = ir.Temp( type = i32, id = 0 ), target = helper_fn, receiver = None, args = [], kwargs = {} ),
			ir.DeleteTemp( temp = ir.Temp( type = i32, id = 0 )),
			ir.Return( value = ir.Temp( type = i32, id = 0 )),
			ir.FuncEnd( name = 'main' ),
		])

	def test_from_import_missing_name_is_a_compile_error( self ) -> None:
		# _stmt_ImportFrom's CompileError is raised from within _lower_stmt's
		# own per-statement dispatch, so lower_function's per-statement
		# recovery boundary (try/except CompileError: continue) swallows it
		# rather than propagating - only visible via discovery.errors.errors,
		# same as check_self_escape's errors (see the self-escape tests above)
		self.compiler.import_code( 'def helper() -> i32:\n\treturn 42\n', Path( 'helper.py' ))
		code = '\n'.join([
			'def main() -> None:',
			'	from helper import nope',
		])
		mod = self._import( code )
		self.compiler._lower( mod.get_local( 'main' ))
		self.assertTrue( any( "does not export 'nope'" in e for e in self.discovery.errors.errors ))

	def test_from_import_missing_module_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	from nonexistent_module import foo',
		])
		mod = self._import( code )
		self.compiler._lower( mod.get_local( 'main' ))
		self.assertTrue( any( 'nonexistent_module' in e for e in self.discovery.errors.errors ))

	# --- @extern(lib, symbol) -----------------------------------------------

	def test_extern_function_lowers_to_a_bare_signature( self ) -> None:
		# no body to lower (discovery.py already required a stub - see
		# _is_stub_body) - just FuncStart(extern_lib=...)/FuncEnd, no CFG/
		# epilogue/locals machinery in between
		code = '\n'.join([
			"@extern( 'c', 'malloc' )",
			'def malloc( size: usize ) -> Ptr[u8]:',
			'	...',
			'',
			'def main() -> None:',
			'	p = malloc( 4 )',
		])
		mod = self._import( code )
		lowered = self.compiler._lower( mod.get_local( 'malloc' ))
		self.assertEqual( len( lowered.instructions ), 2 )
		start, end = lowered.instructions
		self.assertIsInstance( start, ir.FuncStart )
		self.assertEqual( start.extern_lib, 'c' )
		self.assertEqual( start.extern_symbol, 'malloc' )
		self.assertIsInstance( end, ir.FuncEnd )

	def test_extern_function_is_callable_like_any_other( self ) -> None:
		code = '\n'.join([
			"@extern( 'c', 'malloc' )",
			'def malloc( size: usize ) -> Ptr[u8]:',
			'	...',
			'',
			'def main() -> None:',
			'	p = malloc( 4 )',
		])
		mod = self._import( code )
		lowered = self.compiler._lower( mod.get_local( 'main' ))
		calls = [ i for i in lowered.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( len( calls ), 1 )
		self.assertIs( calls[0].target, mod.get_local( 'malloc' ))

	# --- Scalar-to-Scalar casts (u32(...) construction-sugar / compiler.cast) --

	def test_negative_literal_cast_is_a_bare_const( self ) -> None:
		# a negative literal specifically (not just a positive one) - u32(-11)
		# parses as UnaryOp(USub, Constant(11)), so this also exercises that
		# compile_time_transformer folds it before lowering.py ever sees it
		code = '\n'.join([
			'def main() -> None:',
			'	x: u32 = u32( -11 )',
		])
		mod = self._import( code )
		lowered = self.compiler._lower( mod.get_local( 'main' ))
		kinds = [ type( i ).__name__ for i in lowered.instructions ]
		self.assertNotIn( 'Call', kinds )
		self.assertNotIn( 'CastWrap', kinds )
		self.assertNotIn( 'CastCheck', kinds )
		assign = next( i for i in lowered.instructions if isinstance( i, ir.Assign ))
		self.assertEqual( assign.src, ir.Const( type = self.discovery.get_intrinsics()['u32'], value = -11 ))

	# --- integer literal range validation ----------------------------------

	def test_out_of_range_unsigned_literal_assignment_fails( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	x: u8 = 300',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( '300 is out of range for intrinsics.u8 (0..255)', self.discovery.errors.errors[0] )

	def test_out_of_range_negative_signed_literal_assignment_fails( self ) -> None:
		# folds via compile_time_transformer before lowering.py sees it, same
		# mechanism as test_negative_literal_cast_is_a_bare_const's -11 - the
		# difference here is the PLAIN assignment (no explicit i8(...) cast),
		# which must NOT get the cast's bit-reinterpretation exemption
		code = '\n'.join([
			'def main() -> None:',
			'	x: i8 = -200',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( '-200 is out of range for intrinsics.i8 (-128..127)', self.discovery.errors.errors[0] )

	def test_out_of_range_negative_literal_into_unsigned_128_fails( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	x: u128 = -1',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( '-1 is out of range for intrinsics.u128', self.discovery.errors.errors[0] )

	def test_narrowing_literal_cast_is_range_checked( self ) -> None:
		# a NARROWING literal cast (u8's target width is narrower than a
		# bare literal's own natural i32 width - unlike u32(-11)'s same-
		# width case above, which stays a pure, unconditional bit-
		# reinterpretation) is no longer exempt from the ordinary range
		# check every other literal-into-scalar-type context already
		# enforces - confirmed with the user: even for an explicit cast,
		# a literal magnitude that can't actually fit the target's real
		# value range is worth catching as a compile error rather than
		# silently reinterpreting (e.g. i8(128), once used to construct
		# i8::MIN via bit-reinterpretation of a "one past max" literal,
		# is exactly the class of bug this is meant to catch - see
		# float_test.py's test_int_min_div_* for the idiom this replaced).
		code = '\n'.join([
			'def main() -> None:',
			'	x: u8 = u8( 300 )',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( '300 is out of range for intrinsics.u8 (0..255)', self.discovery.errors.errors[0] )

	def test_boundary_values_signed_and_unsigned( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	a: i8 = -128',
			'	b: i8 = 127',
			'	c: u8 = 0',
			'	d: u8 = 255',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_boundary_values_i128_u128( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	a: i128 = -170141183460469231731687303715884105728',
			'	b: i128 = 170141183460469231731687303715884105727',
			'	c: u128 = 0',
			'	d: u128 = 340282366920938463463374607431768211455',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_out_of_range_i128_by_one_fails( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			'	a: i128 = 170141183460469231731687303715884105728',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( 'is out of range for intrinsics.i128', self.discovery.errors.errors[0] )

	def test_isize_usize_range_uses_active_target_bits_not_hardcoded_64( self ) -> None:
		# 2**31 is a valid isize value at 64 bits but out of range at 32 -
		# confirms _int_stem_range derives width from the Scalar's own
		# .sizeof (itself resolved from active_target['bits']), not a fixed
		# assumption
		code = '\n'.join([
			'def main() -> None:',
			'	x: isize = 2147483648',
		])
		target32 = dict( targets.detect(), bits = 32 )
		discovery32 = Discovery( import_builtins = False, active_target = target32 )
		compiler32 = Compiler( discovery32 )
		compiler32.import_code( code, filename = Path( '__test__.py' ))
		compiler32._lower( discovery32.main )
		self.assertIn( 'is out of range for intrinsics.isize', discovery32.errors.errors[0] )

		discovery64 = Discovery( import_builtins = False )
		compiler64 = Compiler( discovery64 )
		compiler64.import_code( code, filename = Path( '__test__.py' ))
		compiler64._lower( discovery64.main )
		self.assertEqual( discovery64.errors.errors, [] )

	def test_i128_range_uses_active_target_has_i128_not_hardcoded_true( self ) -> None:
		# a value needing >64-bit magnitude is a valid i128/u128 literal under
		# a real 128-bit __metalpy_wideint/__metalpy_wideuint (has_i128=True,
		# clang/gcc), but out of range under MSVC's 64-bit wideint/wideuint
		# fallback (has_i128=False) - confirms get_intrinsics() derives i128/
		# u128's own .sizeof from active_target['has_i128'], not a fixed 16
		code = '\n'.join([
			'def main() -> None:',
			'	x: u128 = 18446744073709551616', # 2**64, needs 65 bits
		])
		target_no_i128: ActiveTarget = dict( targets.detect(), has_i128 = False )
		discovery_no_i128 = Discovery( import_builtins = False, active_target = target_no_i128 )
		compiler_no_i128 = Compiler( discovery_no_i128 )
		compiler_no_i128.import_code( code, filename = Path( '__test__.py' ))
		compiler_no_i128._lower( discovery_no_i128.main )
		self.assertIn( 'is out of range for intrinsics.u128', discovery_no_i128.errors.errors[0] )

		target_i128: ActiveTarget = dict( targets.detect(), has_i128 = True )
		discovery_i128 = Discovery( import_builtins = False, active_target = target_i128 )
		compiler_i128 = Compiler( discovery_i128 )
		compiler_i128.import_code( code, filename = Path( '__test__.py' ))
		compiler_i128._lower( discovery_i128.main )
		self.assertEqual( discovery_i128.errors.errors, [] )

	def test_cenum_construction_out_of_range_fails( self ) -> None:
		code = '\n'.join([
			'@enum( u8 )',
			'class MyError:',
			'	Ok = 0',
			'',
			'def main() -> None:',
			'	x: MyError = MyError( 300 )',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( '300 is out of range for __test__.MyError (0..255)', self.discovery.errors.errors[0] )

	def test_cenum_member_declaration_out_of_range_fails( self ) -> None:
		# distinct from test_cenum_construction_out_of_range_fails above - this
		# is the ENUM'S OWN member declaration (discovery.py's
		# _register_enum_member), a completely separate mechanism from a
		# construction call (lowering.py's _try_lower_construct_call)
		code = '\n'.join([
			'@enum( u8 )',
			'class MyError:',
			'	Bad = 300',
			'	Ok = 0',
			'',
			'def main() -> None:',
			'	x: MyError = MyError( 0 )',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( '300 is out of range for __test__.MyError (0..255)', self.discovery.errors.errors[0] )

	def test_cenum_member_declaration_boundary_values( self ) -> None:
		# i8's own MIN needs a negative literal (-128) - _register_enum_member
		# now folds node.value through compile_time_transformer.transform_expr
		# before checking for ast.Constant, so UnaryOp(USub, Constant(128))
		# collapses to Constant(-128) same as it already would in a function
		# body
		code = '\n'.join([
			'@enum( i8 )',
			'class MyError:',
			'	Lo = -128',
			'	Hi = 127',
			'',
			'def main() -> None:',
			'	x: MyError = MyError( 0 )',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_cenum_member_declaration_out_of_range_by_one_fails( self ) -> None:
		code = '\n'.join([
			'@enum( i8 )',
			'class MyError:',
			'	Bad = 128',
			'',
			'def main() -> None:',
			'	x: MyError = MyError( 0 )',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( '128 is out of range for __test__.MyError (-128..127)', self.discovery.errors.errors[0] )

	def test_cenum_member_declaration_negative_out_of_range_fails( self ) -> None:
		code = '\n'.join([
			'@enum( i8 )',
			'class MyError:',
			'	Bad = -129',
			'',
			'def main() -> None:',
			'	x: MyError = MyError( 0 )',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( '-129 is out of range for __test__.MyError (-128..127)', self.discovery.errors.errors[0] )

	def test_cenum_member_auto_increment_overflow_fails( self ) -> None:
		# the '_' auto-increment sentinel can ALSO overflow the underlying
		# type's range after enough members - A=254, B='_' auto-fills 255
		# (still in range), C='_' auto-fills 256 (out of range for u8)
		code = '\n'.join([
			'@enum( u8 )',
			'class MyError:',
			'	A = 254',
			'	B = _',
			'	C = _',
			'',
			'def main() -> None:',
			'	x: MyError = MyError( 0 )',
		])
		self._import( code )
		self._lower_main()
		self.assertIn( '256 is out of range for __test__.MyError (0..255)', self.discovery.errors.errors[0] )

	def test_non_literal_cast_default_check_mode( self ) -> None:
		code = self._RESULT_FIXTURE + '\n' + '\n'.join([
			'class OverflowError: pass',
			'',
			'def f( s: usize ) -> Result[u32,OverflowError]:',
			'	x = u32( s )',
			'	return Result.Ok( x )',
		])
		mod = self._import( code )
		lowered = self.compiler._lower( mod.get_local( 'f' ))
		kinds = [ type( i ).__name__ for i in lowered.instructions ]
		self.assertIn( 'CastCheck', kinds )
		# x's own raw CastCheck Result flows unconsumed into Result.Ok(x) as
		# a generic-inferred argument for T - _lower_and_infer_call_args'
		# own auto_consume_hint_arg hook auto-.or_throw()'s it there (case 2
		# of the general auto-or_throw() rule) instead of eagerly at `x = ...`
		self.assertIn( 'OrThrow', kinds )

	def test_non_literal_cast_without_result_return_is_a_compile_error( self ) -> None:
		# x is assigned but never used again at all (no annotation, no
		# later read) - stays a raw, uninspected Result[u32,OverflowError]
		# all the way to function end, where cfg.py's own (separate,
		# already-correct, untouched by this whole change) unchecked-
		# result tracking is what actually catches it now, not an eager
		# Result-coverage check at the assignment itself. That check raises
		# past FunctionLowering.run()'s own per-statement recovery boundary
		# (it isn't a per-statement check), so this needs its own explicit
		# CompileError catch instead of reading self.discovery.errors after
		# an ordinary (non-raising) _lower() call
		code = self._RESULT_FIXTURE + '\n' + '\n'.join([
			'class OverflowError: pass',
			'',
			'def f( s: usize ) -> None:',
			'	x = u32( s )',
		])
		mod = self._import( code )
		with self.assertRaises( CompileError ) as ctx:
			self.compiler._lower( mod.get_local( 'f' ))
		self.assertIn( "Result value 'x' was never inspected", str( ctx.exception ))

	def test_wrap_arithmetic_cast_has_no_result( self ) -> None:
		code = '\n'.join([
			'def f( s: usize ) -> u32:',
			'	with compiler.wrap_arithmetic:',
			'		return u32( s )',
		])
		mod = self._import( code )
		lowered = self.compiler._lower( mod.get_local( 'f' ))
		kinds = [ type( i ).__name__ for i in lowered.instructions ]
		self.assertIn( 'CastWrap', kinds )
		self.assertNotIn( 'CastCheck', kinds )
		self.assertNotIn( 'OrReturn', kinds )

	def test_compiler_cast_shares_the_same_lowering_as_construction_sugar( self ) -> None:
		code = '\n'.join([
			'def f( s: usize ) -> u32:',
			'	with compiler.wrap_arithmetic:',
			'		return compiler.cast( u32, s )',
		])
		mod = self._import( code )
		lowered = self.compiler._lower( mod.get_local( 'f' ))
		kinds = [ type( i ).__name__ for i in lowered.instructions ]
		self.assertIn( 'CastWrap', kinds )

	def test_non_scalar_source_without_dunder_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'class Bar: pass',
			'',
			'def f() -> u32:',
			'	b = Bar()',
			'	return u32( b )',
		])
		mod = self._import( code )
		self.compiler._lower( mod.get_local( 'f' ))
		self.assertTrue( any( 'has no __u32__ method' in e for e in self.discovery.errors.errors ))

	def test_non_scalar_source_with_dunder_dispatches_to_it( self ) -> None:
		code = '\n'.join([
			'class Foo:',
			'	def __u32__( self ) -> u32:',
			'		return 5',
			'',
			'def f() -> u32:',
			'	x = Foo()',
			'	return u32( x )',
		])
		mod = self._import( code )
		lowered = self.compiler._lower( mod.get_local( 'f' ))
		calls = [ i for i in lowered.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( len( calls ), 1 )
		self.assertEqual( calls[0].target.stem, '__u32__' )

	def test_wrong_arity_is_a_compile_error( self ) -> None:
		for call in ( 'u32()', 'u32( 1, 2 )' ):
			with self.subTest( call = call ):
				code = '\n'.join([
					'def main() -> None:',
					f'	x = {call}',
				])
				mod = self._import( code )
				self.compiler._lower( mod.get_local( 'main' ))
				self.assertTrue( any( 'takes exactly one argument' in e for e in self.discovery.errors.errors ))

	def test_cenum_member_value_expression_lowers_to_const( self ) -> None:
		code = '\n'.join([
			#'import compiler',
			'@enum( u32 )',
			'class MyError:',
			'	FileNotFound = 2',
			'	Other = _',
			'',
			'def main() -> u32:',
			'	return MyError.FileNotFound',
		])
		u32 = self.discovery.get_intrinsics()['u32']
		self._test_ir( code, [
			ir.FuncStart( name = 'main', params = [], return_type = u32 ),
			ir.Return( value = ir.Const( type = u32, value = 2 )),
			ir.FuncEnd( name = 'main' ),
		])

	def test_recursive_namespace_cenum_member_value_expression_lowers_to_const( self ) -> None:
		import platform
		match platform.system():
			case 'Windows':
				oserror_type = 'u32'
			case 'Linux':
				oserror_type = 'i32'
			case _:
				assert False, f'unsupported {platform.system()=}'
		code = '\n'.join([
			'import compiler',
			'import builtins',
			f'def main() -> {oserror_type}:',
			'	return builtins.OSError.FileNotFoundError',
		])
		oserror_type = self.discovery.get_intrinsics()[oserror_type]
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_ir( fn, [
			ir.FuncStart( name = 'main', params = [], return_type = oserror_type ),
			ir.Return( value = ir.Const( type = oserror_type, value = 2 )),
			ir.FuncEnd( name = 'main' ),
		])

	def test_cenum_construction_lowers_to_underlying_value( self ) -> None:
		import platform
		match platform.system():
			case 'Windows':
				oserror_type_stem = 'u32'
			case 'Linux':
				oserror_type_stem = 'i32'
			case _:
				assert False, f'unsupported {platform.system()=}'
		code = '\n'.join([
			'import builtins',
			f'def main() -> {oserror_type_stem}:',
			'\tx: builtins.OSError = builtins.OSError( 42 )',
			'\treturn x',
		])
		oserror_type = self.discovery.get_intrinsics()[oserror_type_stem]
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( i ).__name__ for i in fn.instructions ]
		self.assertIn( 'Assign', kinds )
		# verify the assign carries the value 42, typed as the enum itself
		assign = next( i for i in fn.instructions if isinstance( i, ir.Assign ))
		self.assertIsInstance( assign.src, ir.Const )
		self.assertEqual( assign.src.value, 42 )
		from mpy_types import CEnum
		self.assertIsInstance( assign.src.type, CEnum )

	# --- unchecked Result tracking ------------------------------------------

	def test_unchecked_result_at_return_is_a_compile_error( self ) -> None:
		code = self._RESULT_FIXTURE + '\n' + '\n'.join([
			'class MyError: pass',
			'',
			'def get() -> Result[i32,MyError]:',
			'	pass',
			'',
			'def main() -> None:',
			'	r: Result[i32,MyError] = get()',
			'	return',
		])
		self._import( code )
		self._lower_main() # caught by lower_function's own per-statement recovery - doesn't raise, just records
		self.assertTrue( any( "'r'" in e and 'never inspected' in e for e in self.discovery.errors.errors ) )

	def test_returning_the_unchecked_result_itself_is_fine( self ) -> None:
		# "return r transfers the obligation to the caller" - v1 scope cut
		code = self._RESULT_FIXTURE + '\n' + '\n'.join([
			'class MyError: pass',
			'',
			'def get() -> Result[i32,MyError]:',
			'	pass',
			'',
			'def main() -> Result[i32,MyError]:',
			'	r: Result[i32,MyError] = get()',
			'	return r',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_overwriting_unchecked_result_is_a_compile_error( self ) -> None:
		code = self._RESULT_FIXTURE + '\n' + '\n'.join([
			'class MyError: pass',
			'',
			'def get() -> Result[i32,MyError]:',
			'	pass',
			'',
			'def main() -> None:',
			'	r: Result[i32,MyError] = get()',
			'	r = get()',
			'	r.is_ok()',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any( "'r'" in e and 'discarded' in e for e in self.discovery.errors.errors ) )

	def test_del_unchecked_result_is_a_compile_error( self ) -> None:
		code = self._RESULT_FIXTURE + '\n' + '\n'.join([
			'class MyError: pass',
			'',
			'def get() -> Result[i32,MyError]:',
			'	pass',
			'',
			'def main() -> None:',
			'	r: Result[i32,MyError] = get()',
			'	del r',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any( "'r'" in e and 'discarded via del' in e for e in self.discovery.errors.errors ) )

	def test_is_ok_clears_the_result( self ) -> None:
		code = self._RESULT_FIXTURE + '\n' + '\n'.join([
			'class MyError: pass',
			'',
			'def get() -> Result[i32,MyError]:',
			'	pass',
			'',
			'def main() -> None:',
			'	r: Result[i32,MyError] = get()',
			'	r.is_ok()',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_match_clears_the_original_name_not_just_the_synthetic_subject( self ) -> None:
		# regression test for the false-positive the naive match desugaring
		# would otherwise produce: match r: rewrites to __match_subj_N = r;
		# if ... - without the is_match_subject/match_clears_name fix, the
		# synthetic __match_subj_N would itself become a phantom tracked
		# obligation nothing ever clears, AND the original r would never
		# get cleared either (ordinary aliasing assignment doesn't propagate
		# a clear back to its source)
		# NOT _RESULT_FIXTURE as-is: its own `class bool: pass` shadow (kept
		# for other, builtins-less callers of that shared fixture) conflicts
		# with builtins' own real intrinsics.bool now that self.tag == 0/1
		# (is_ok/is_err) and the match's own tag dispatch go through a real
		# u8.__eq__ dunder call - that dunder's own return type is always
		# the real bool, so this test needs the real one too
		self.discovery.import_name( 'builtins' )
		code = '\n'.join([
			'@union',
			'class Result[T,E]:',
			'	Ok: T',
			'	Err: E',
			'',
			'	def is_ok( self ) -> bool:',
			'		return self.tag == 0',
			'',
			'	def is_err( self ) -> bool:',
			'		return self.tag == 1',
			'',
			'class MyError: pass',
			'',
			'def get() -> Result[i32,MyError]:',
			'	pass',
			'',
			'def main() -> None:',
			'	r: Result[i32,MyError] = get()',
			'	match r:',
			'		case Result.Ok( v ):',
			'			x: i32 = v',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_bare_result_returning_call_statement_is_a_compile_error( self ) -> None:
		# `get()` as a bare statement is discarded (case 1 of the general
		# auto-or_throw() rule) - no longer a hard "discarded here" rejection
		# on its own; it auto-.or_throw()'s the call's Result, which THEN
		# fails to compile because main() can't propagate MyError anywhere
		code = self._RESULT_FIXTURE + '\n' + '\n'.join([
			'class MyError: pass',
			'',
			'def get() -> Result[i32,MyError]:',
			'	pass',
			'',
			'def main() -> None:',
			'	get()',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any( 'requires the enclosing function to return Result' in e for e in self.discovery.errors.errors ) )

	def test_one_branch_checks_other_doesnt_is_a_compile_error( self ) -> None:
		code = self._RESULT_FIXTURE + '\n' + '\n'.join([
			'class MyError: pass',
			'',
			'def get() -> Result[i32,MyError]:',
			'	pass',
			'',
			'def main() -> None:',
			'	c: bool = True',
			'	r: Result[i32,MyError] = get()',
			'	if c:',
			'		r.is_ok()',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any( 'inspected on one branch' in e for e in self.discovery.errors.errors ) )

	def test_branch_confined_fresh_result_never_checked_is_a_compile_error( self ) -> None:
		# NOT in the plan's original validation table - required by the
		# feature's own goal: a Result introduced fresh inside just one
		# non-terminating branch and left unchecked at that branch's own
		# join point is about to go out of scope right there
		code = self._RESULT_FIXTURE + '\n' + '\n'.join([
			'class MyError: pass',
			'',
			'def get() -> Result[i32,MyError]:',
			'	pass',
			'',
			'def main() -> None:',
			'	c: bool = True',
			'	if c:',
			'		r: Result[i32,MyError] = get()',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any( "'r'" in e and 'never inspected' in e for e in self.discovery.errors.errors ) )

	def test_result_with_no_rc_leaves_is_still_tracked( self ) -> None:
		# regression test: cfg.py's bindings/rc_leaves machinery is RC-only
		# by design ("invisible here" per the module docstring) - the
		# unchecked-Result set must be tracked independent of that gate, or
		# a Result[i32,SomeScalarError] (the common shape) would never be
		# checked at all
		code = self._RESULT_FIXTURE + '\n' + '\n'.join([
			'@cstruct',
			'class MyError:',
			'	code: i32',
			'',
			'def get() -> Result[i32,MyError]:',
			'	pass',
			'',
			'def main() -> None:',
			'	r: Result[i32,MyError] = get()',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any( "'r'" in e and 'never inspected' in e for e in self.discovery.errors.errors ) )

	def test_result_fresh_every_loop_iteration_never_checked_is_a_compile_error( self ) -> None:
		code = self._RESULT_FIXTURE + '\n' + '\n'.join([
			'class MyError: pass',
			'',
			'def get() -> Result[i32,MyError]:',
			'	pass',
			'',
			'def main() -> None:',
			'	a: bool = True',
			'	while a:',
			'		r: Result[i32,MyError] = get()',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any( "'r'" in e and 'produced fresh every loop iteration' in e for e in self.discovery.errors.errors ) )

	# --- formerly-known v1 gaps -----------------------------------------------
	#
	# Each of these used to document a real hole in the unchecked-Result
	# check (found and deliberately scoped out while building it) by
	# asserting the then-current (wrong) behavior - discovery.errors.errors
	# == [] where the feature's own stated goal said it should NOT be
	# empty. All three have since been closed; these now assert the
	# correct error instead, guarding against regressing back to the gap.

	def test_break_can_no_longer_silently_discard_a_result_the_fallthrough_path_checks( self ) -> None:
		# r is discarded on the break path (never reaches r.is_ok(), which
		# only the non-break path executes) - closed via cfg.py's
		# check_loop_exit_unchecked_results(), called from _stmt_Break/
		# _stmt_Continue alongside the existing unwind_to() call. Only
		# flags Results introduced SINCE the loop's own entry (confined to
		# the loop body, about to be lost) - a pre-existing outer unchecked
		# Result is deliberately not flagged by a break/continue, since its
		# obligation isn't lost, just deferred to wherever its own scope
		# actually ends
		code = self._RESULT_FIXTURE + '\n' + '\n'.join([
			'class MyError: pass',
			'',
			'def get() -> Result[i32,MyError]:',
			'	pass',
			'',
			'def main() -> None:',
			'	a: bool = True',
			'	c: bool = True',
			'	while a:',
			'		r: Result[i32,MyError] = get()',
			'		if c:',
			'			break',
			'		r.is_ok()',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any( "'r'" in e and 'never inspected before exiting the loop' in e for e in self.discovery.errors.errors ) )

	def test_checked_arithmetic_overflow_can_no_longer_silently_discard_an_unrelated_result( self ) -> None:
		# a + b's own overflow-triggered OrReturn (_consume_checked_result,
		# shared with or_return()'s own early-exit) is a second, separate
		# function-exit point exactly like or_return()'s - closed by moving
		# the clear_result( receiver )/check_unchecked_results() pair OUT of
		# _lower_or_return and INTO _consume_checked_result itself (the
		# actual shared choke point), with an ast node now threaded through
		# _emit_checked_op/_consume_checked_result from their real callers
		# for discovery.fail()'s own location
		# uses the REAL builtins.Result (not _RESULT_FIXTURE's local one) -
		# a + b's own dunder needs checked()'s return type to cover the REAL
		# builtins.OverflowError (see test_binop_check_mode_emits_or_return's
		# comment); builtins.Result already has is_ok()/Ok(), so there's no
		# need for a separate local Result just for r's own discard-check shape
		self.discovery.import_name( 'builtins' )
		code = '\n'.join([
			'from builtins import Result, OverflowError',
			'',
			'class MyError: pass',
			'',
			'def get() -> Result[i32,MyError]:',
			'	pass',
			'',
			'def checked( a: i32, b: i32 ) -> Result[i32,OverflowError]:',
			'	r: Result[i32,MyError] = get()',
			'	x: i32 = a + b',
			'	r.is_ok()',
			'	return Result.Ok( x )',
		])
		self._import( code )
		fn = self.discovery.modules['__test__'].get_local( 'checked' )
		if fn.resolve is not None:
			fn.resolve()
		self.compiler._lower( fn )
		self.assertTrue( any( "'r'" in e and 'never inspected' in e for e in self.discovery.errors.errors ) )

	def test_receiver_based_generic_method_call_can_no_longer_bypass_the_discard_check( self ) -> None:
		# a generic METHOD called through a receiver whose type isn't known
		# until lowering (Box's own local `b` here) is left untagged by
		# type_resolver.py's own pre-pass (see type_resolver_test.py's own
		# test_generic_call_on_receiver_local_is_left_untagged) and routes
		# through _lower_class_generic_method_call/_emit_generic_call
		# instead of _lower_call's own shared tail - _emit_generic_call has
		# its own copy of the shared case-1 discard-consumption logic
		# (_finish_call_result), so a discarded Result from THIS path auto-
		# .or_throw()s too, exactly like _lower_call's own shared tail
		code = self._RESULT_FIXTURE + '\n' + '\n'.join([
			'class MyError: pass',
			'',
			'class Box:',
			'	def get[T]( self, x: T ) -> Result[T,MyError]:',
			'		pass',
			'',
			'def main() -> None:',
			'	b: Box = Box()',
			'	n: i32 = 5',
			'	b.get( n )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any( 'requires the enclosing function to return Result' in e for e in self.discovery.errors.errors ) )

	# --- T|None leaf coercion (_lower_expr/_coerce_into_union) -------------

	def test_leaf_value_coerced_into_union_via_synthesized_constructor( self ) -> None:
		# a plain leaf value (i32) flowing into an i32|bool-typed call
		# argument must go through the union's own UnionStorage-synthesized
		# member constructor (an ir.Call), not a bare ir.Const mistyped as
		# the whole union - see TODO.txt's "opportunistic union emission",
		# lowering.py's _lower_expr/_coerce_into_union. Uses intrinsics
		# only (i32/bool), not str - this class's own Discovery is built
		# with import_builtins=False
		code = '\n'.join([
			'def helper( x: i32|bool ) -> i32|bool:',
			'	return x',
			'',
			'def main() -> i32:',
			'	helper( 5 )',
			'	return 0',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		# the FIRST Call is the coercion itself (argument lowering runs
		# before the helper(...) call is emitted) - target is the union's
		# own synthesized 'i32' member constructor, called with the
		# literal's own natural i32-typed Const, not the union type
		call = next( instr for instr in fn.instructions if isinstance( instr, ir.Call ))
		self.assertEqual( call.target.stem, 'i32' )
		self.assertEqual( len( call.args ), 1 )
		self.assertIsInstance( call.args[0], ir.Const )
		self.assertEqual( call.args[0].type.stem, 'i32' )
		self.assertEqual( call.args[0].value, 5 )
		# the call's own dest is the union itself, not the leaf type
		# (canonicalized asciibetically by qualname - 'bool' < 'i32')
		self.assertEqual( call.dest.type.stem, 'intrinsics.bool|intrinsics.i32' )

# --- in / not in against real builtin types --------------------------------

class InOperatorRealBuiltinsTests( unittest.TestCase ):
	''' the reversed-receiver __contains__ dispatch tests above (in the main
	Tests class) use cheap user-defined classes under import_builtins=False.
	str.__contains__ is a real lib/builtins/__init__.py method, so exercising
	`x in some_str` needs the real builtins loaded - same reason
	JoinedStrLoweringTests/WalrusOperatorTests keep their own
	import_builtins=True setUp instead of sharing the main Tests class's. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _import( self, code: str ):
		return self.compiler.import_code( code, filename = Path( '__test__.py' ))

	def test_in_on_str_dispatches_to_str_contains( self ) -> None:
		code = '\n'.join([
			'def main() -> None:',
			"	s: str = 'hello world'",
			"	found: bool = 'world' in s",
			'	return',
		])
		mod = self._import( code )
		lowered = self.compiler._lower( mod.get_local( 'main' ))
		calls = [ i for i in lowered.instructions if isinstance( i, ir.Call ) ]
		self.assertTrue( any( c.target.stem == '__contains__' for c in calls ))

# --- try/except/.or_throw() ------------------------------------------------

class TryExceptOrThrowLoweringTests( unittest.TestCase ):
	''' limited try/except/else/finally + Result.or_throw() (see PLAN in the
	task/commit that added this) - real builtins needed (list[T].__getitem__'s
	real Result[T,IndexError], str's real __str__/f-string support), same
	reason InOperatorRealBuiltinsTests keeps its own import_builtins=True
	setUp instead of sharing the main Tests class's minimal fixture. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _import( self, code: str ):
		return self.compiler.import_code( code, filename = Path( '__test__.py' ))

	def test_or_throw_dispatches_to_matching_except_handler( self ) -> None:
		# list[T].__getitem__ declares -> Result[T,IndexError] - a single
		# OPAQUE leaf (not a union), fully covered by the one except clause
		# below, so this needs no ir.OrReturn/ir.OrJump widening code at all
		# (foo() itself declares no Result return type, and needs none -
		# _require_or_throw_return's own "no requirement when every leaf is
		# covered" contract)
		code = '\n'.join([
			'def foo() -> None:',
			'	ar: list[str] = list[str]()',
			'	try:',
			"		print( ar[0].or_throw() )",
			'	except IndexError as e:',
			'		compiler.decref( e )',
		])
		mod = self._import( code )
		foo_fn = mod.get_local( 'foo' )
		if foo_fn.resolve is not None:
			foo_fn.resolve()
		fn = self.compiler._lower( foo_fn )
		self.assertEqual( self.discovery.errors.errors, [] )

		throws = [ i for i in fn.instructions if isinstance( i, ir.OrThrow ) ]
		self.assertEqual( len( throws ), 1, f'expected exactly one ir.OrThrow, got: {fn.instructions!r}' )
		throw = throws[0]
		# fully covered - no propagate-to-caller fallback shape at all
		self.assertEqual( throw.epilogue, [] )
		self.assertIsNone( throw.target )
		self.assertIsNone( throw.return_slot )
		self.assertEqual( len( throw.dispatch ), 1, f'expected exactly one dispatch entry (IndexError is a single opaque leaf), got: {throw.dispatch!r}' )
		leaf = throw.dispatch[0]
		self.assertEqual( leaf.leaf.qualname, 'builtins.IndexError' )
		self.assertIsNotNone( leaf.bind )
		self.assertEqual( leaf.bind.stem, 'e' )
		self.assertEqual( leaf.bind.type.qualname, 'builtins.IndexError' )

		# the handler's own label is a real Label somewhere in this
		# function's instructions, and leaf.label jumps into it
		labels = { i.name for i in fn.instructions if isinstance( i, ir.Label ) }
		self.assertIn( leaf.label, labels )

		# no or_return()-style unconditional propagation anywhere in this
		# function - or_throw()'s own Err branch is fully handled by the
		# dispatch above, not by a separate OrReturn/OrJump
		self.assertFalse( any( isinstance( i, ( ir.OrReturn, ir.OrJump )) for i in fn.instructions ))

# --- @inline (PLAN_INLINE.md) -------------------------------------------

class InlineTests( unittest.TestCase ):
	''' @inline splices a function's own single `return <expr>` body
	directly at each call site - no real Call/FuncStart/FuncEnd for the
	callee itself. Mirrors builtins.len[T]'s own shape without depending on
	real builtins (import_builtins=False, same as the main Tests class). '''
	maxDiff = None

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = False )
		self.compiler = Compiler( self.discovery )

	def _import( self, code: str ):
		return self.compiler.import_code( code, filename = Path( '__test__.py' ))

	def _lower_main( self ) -> LoweredFunction:
		fn = self.compiler._lower( self.discovery.main )
		self.assertEqual( type( fn ), LoweredFunction )
		return fn

	def _ir_repr( self, fn: LoweredFunction ) -> list[str]:
		return [ op.test_repr() for op in fn.instructions ]

	# compiler.run() force-enqueues windows/_console.py's own console-codepage
	# global on every Windows target, and sys.exit() whenever no_crt (see
	# Compiler.force_reachable's own comment) - real but incidental to what
	# these tests check, and absent entirely on non-Windows targets (or on
	# CRT-linked Windows targets, for sys.exit specifically), so exact-set
	# assertions filter them back out first
	_CONSOLE_INIT_QUALNAMES = frozenset({
		'windows._console._init_console', 'windows.kernel32.SetConsoleOutputCP',
		'sys.exit', 'windows.kernel32.ExitProcess',
		'sys.memset', 'sys.memcpy', 'windows.ntdll.RtlFillMemory', 'windows.ntdll.RtlCopyMemory',
	})

	def _function_qualnames( self ) -> set[str]:
		return { lf.function.qualname for lf in self.compiler.functions } - self._CONSOLE_INIT_QUALNAMES

	def test_inline_method_call_compiles_identically_to_calling_the_body_directly( self ) -> None:
		# @inline def get_len(self): return self.__len__() called as
		# b.get_len() must produce the SAME instruction shape as writing
		# b.__len__() directly at the call site - no Call/FuncStart/FuncEnd
		# of its own for get_len anywhere, and the one real Call (to
		# __len__) receives the SAME receiver (main's own `b` parameter,
		# reused directly - no synthesized alias) either way.
		#
		# Compared structurally (instruction kinds + the one real Call's
		# own target/receiver qualnames), not via a full test_repr() diff -
		# Call.receiver's own .type is the whole Box class, whose own
		# .methods legitimately differs between the two snippets (the
		# `inlined` Box really does have one more method, get_len, than
		# `direct`'s Box does), which would make a byte-for-byte repr
		# comparison spuriously fail despite both compiling to the same
		# real work
		inlined = '\n'.join([
			'@cstruct',
			'class Box:',
			'	y: usize',
			'	def __len__( self ) -> usize:',
			'		return self.y',
			'	@inline',
			'	def get_len( self ) -> usize:',
			'		return self.__len__()',
			'',
			'def main( b: Box ) -> usize:',
			'	return b.get_len()',
		])
		direct = '\n'.join([
			'@cstruct',
			'class Box:',
			'	y: usize',
			'	def __len__( self ) -> usize:',
			'		return self.y',
			'',
			'def main( b: Box ) -> usize:',
			'	return b.__len__()',
		])
		self._import( inlined )
		inlined_fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

		other = Discovery( import_builtins = False )
		other_compiler = Compiler( other )
		other_compiler.import_code( direct, filename = Path( '__test__.py' ))
		direct_fn = other_compiler._lower( other.main )
		self.assertEqual( other.errors.errors, [] )

		self.assertEqual(
			[ type( i ).__name__ for i in inlined_fn.instructions ],
			[ type( i ).__name__ for i in direct_fn.instructions ],
		)
		inlined_calls = [ i for i in inlined_fn.instructions if isinstance( i, ir.Call ) ]
		direct_calls = [ i for i in direct_fn.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( [ c.target.qualname for c in inlined_calls ], [ '__test__.Box.__len__' ] )
		self.assertEqual( [ c.target.qualname for c in direct_calls ], [ '__test__.Box.__len__' ] )
		# the receiver is main's OWN `b` parameter, reused directly - not a
		# freshly synthesized alias (proves the "operand already a
		# Variable -> reuse it, no extra Assign" fast path actually fired)
		self.assertEqual( inlined_calls[0].receiver.qualname, 'main.b' )
		self.assertIs( inlined_calls[0].receiver, inlined_fn.function.parameters[0] )

	def test_inline_bare_generic_call_resolves_t_and_splices_no_specialization_compiled( self ) -> None:
		# mirrors builtins.len[T] exactly: a bare generic @inline free
		# function, T inferred from the argument. After compiler.run()
		# drains the whole queue, the ONLY compiled functions must be
		# main and Box.__len__ - never get_len or get_len[Box], since an
		# @inline generic instantiation is never scheduled as a real unit
		code = '\n'.join([
			'@cstruct',
			'class Box:',
			'	y: usize', # usize directly - no scalar cast in __len__, avoids needing a Result[T,E] class in scope just for checked-cast machinery
			'	def __len__( self ) -> usize:',
			'		return self.y',
			'',
			'@inline',
			'def get_len[T]( t: T ) -> usize:',
			'	return t.__len__()',
			'',
			'def main( b: Box ) -> usize:',
			'	return get_len( b )',
		])
		self._import( code )
		self.compiler.run()
		self.assertEqual( self.discovery.errors.errors, [] )
		qualnames = self._function_qualnames()
		self.assertEqual( qualnames, { 'main', '__test__.Box.__len__' } )
		main_fn = next( lf for lf in self.compiler.functions if lf.function.qualname == 'main' )
		calls = [ i for i in main_fn.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( [ c.target.qualname for c in calls ], [ '__test__.Box.__len__' ] )

	def test_inline_explicit_specialization_call_also_splices( self ) -> None:
		# the explicit get_len[Box](b) spelling goes through a different
		# lowering path (_lower_generic_function_call, not the bare-call
		# inference path) - must be wired up the same way
		code = '\n'.join([
			'@cstruct',
			'class Box:',
			'	y: usize', # usize directly - no scalar cast in __len__, avoids needing a Result[T,E] class in scope just for checked-cast machinery
			'	def __len__( self ) -> usize:',
			'		return self.y',
			'',
			'@inline',
			'def get_len[T]( t: T ) -> usize:',
			'	return t.__len__()',
			'',
			'def main( b: Box ) -> usize:',
			'	return get_len[Box]( b )',
		])
		self._import( code )
		self.compiler.run()
		self.assertEqual( self.discovery.errors.errors, [] )
		qualnames = self._function_qualnames()
		self.assertEqual( qualnames, { 'main', '__test__.Box.__len__' } )

	def test_inline_receiver_with_side_effect_evaluated_once( self ) -> None:
		# the receiver expression is a real call (make_box()) - inlining
		# must not re-lower/re-evaluate it once per reference to `self` in
		# the spliced body; it's lowered ONCE, up front, into an operand
		# that the substitution then just reuses
		code = '\n'.join([
			'@cstruct',
			'class Box:',
			'	y: usize',
			'	def __len__( self ) -> usize:',
			'		return self.y',
			'	@inline',
			'	def get_len( self ) -> usize:',
			'		return self.__len__()',
			'',
			'def make_box() -> Box:',
			'	b: Box',
			'	b.y = 5',
			'	return b',
			'',
			'def main() -> usize:',
			'	return make_box().get_len()',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( [ c.target.qualname for c in calls ], [ '__test__.make_box', '__test__.Box.__len__' ] )

	def test_inline_direct_recursion_rejected( self ) -> None:
		code = '\n'.join([
			'@inline',
			'def foo( x: i32 ) -> i32:',
			'	return foo( x )',
			'',
			'def main() -> i32:',
			'	return foo( 1 )',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any( 'recursive inlining' in e for e in self.discovery.errors.errors ))

	def test_inline_mutual_recursion_rejected( self ) -> None:
		code = '\n'.join([
			'@inline',
			'def ping( x: i32 ) -> i32:',
			'	return pong( x )',
			'',
			'@inline',
			'def pong( x: i32 ) -> i32:',
			'	return ping( x )',
			'',
			'def main() -> i32:',
			'	return ping( 1 )',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any( 'recursive inlining' in e for e in self.discovery.errors.errors ))

	def test_inline_discarding_a_result_returning_call_is_rejected( self ) -> None:
		# `make()` (an @inline call) as a bare statement is discarded (case 1
		# of the general auto-or_throw() rule) - it auto-.or_throw()'s the
		# splice's own trailing Result, which THEN fails to compile because
		# main() can't propagate MyError anywhere
		code = '\n'.join([
			'class MyError: pass',
			'',
			'@union',
			'class Result[T,E]:',
			'	Ok: T',
			'	Err: E',
			'',
			'@inline',
			'def make() -> Result[i32,MyError]:',
			'	return Result.Ok( 1 )',
			'',
			'def main() -> None:',
			'	make()',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any( 'requires the enclosing function to return Result' in e for e in self.discovery.errors.errors ))

	def test_inline_is_ok_still_clears_cfg_unchecked_result_tracking( self ) -> None:
		# a receiver-based generic-class method (is_ok, inherited genericity
		# from Result[T,E]) - PLAN_INLINE.md's own correction: this reaches
		# the ordinary plain final tail (already-monomorphized by
		# _attr_lookup_callable before dispatch), and _cfg.clear_result(...)
		# runs unconditionally before that tail regardless of inlining - so
		# `r` must NOT be reported as an unchecked Result here, and the
		# inlined body must still compile down to the bare `self.tag == 0`
		# comparison, no Call to is_ok anywhere
		code = '\n'.join([
			'class MyError: pass',
			'',
			'@union',
			'class Result[T,E]:',
			'	Ok: T',
			'	Err: E',
			'',
			'	@inline',
			'	def is_ok( self ) -> bool:',
			'		return self.tag == 0',
			'',
			'def checked() -> Result[i32,MyError]:',
			'	return Result.Ok( 1 )',
			'',
			'def main() -> bool:',
			'	r = checked()',
			'	return r.is_ok()',
		])
		self.discovery.import_name( 'builtins' ) # self.tag == 0 is now an ordinary u8.__eq__ dunder call
		self._import( code )
		self.compiler.run()
		self.assertEqual( self.discovery.errors.errors, [] )
		main_fn = next( lf for lf in self.compiler.functions if lf.function.qualname == 'main' )
		kinds = [ type( instr ).__name__ for instr in main_fn.instructions ]
		# exactly one real Call (checked()) - is_ok itself never becomes one;
		# its own body is spliced down to a bare r.tag == 0 (GetAttr + Cmp)
		calls = [ instr for instr in main_fn.instructions if isinstance( instr, ir.Call ) ]
		self.assertEqual( [ c.target.qualname for c in calls ], [ '__test__.checked' ] )
		self.assertIn( 'GetAttr', kinds )
		self.assertIn( 'Cmp', kinds )
		qualnames = { lf.function.qualname for lf in self.compiler.functions }
		self.assertNotIn( '__test__.Result.is_ok[intrinsics.i32,__test__.MyError]', qualnames )

	def test_type_error_inside_inline_body_names_the_real_call_site_and_arg_types( self ) -> None:
		# regression: a type mismatch inside an @inline function's own
		# spliced body (e.g. builtins.len[T]'s `return t.__len__()`) used to
		# report ONLY that body's own source location (lib/builtins/
		# __init__.py, in the real case) - useless for a caller with more
		# than one call site to the same @inline generic, and it never said
		# what T actually was either. _lower_inline_call now appends one
		# additional, purely additive note pointing at THIS call's own real
		# site plus each argument's own concrete type - confirmed via this
		# exact shape (mirrors builtins.len[T] without depending on real
		# builtins, same posture as this whole class' own docstring)
		code = '\n'.join([
			'class Thing:',
			'	def get( self ) -> usize:',
			'		return usize( 3 )',
			'',
			'@inline',
			'def get_it[T]( t: T ) -> usize:',
			'	return t.get()',
			'',
			'def main() -> i32:',
			'	t: Thing = Thing()',
			'	count: i32 = get_it( t )',
			'	return count',
		])
		self._import( code )
		self._lower_main()
		errors = self.discovery.errors.errors
		self.assertEqual( len( errors ), 2 )
		self.assertIn( 't.get()', errors[0] )
		note = errors[1]
		self.assertIn( '__test__.py:11', note ) # the real call site, not get_it's own body
		self.assertIn( 'get_it[__test__.Thing]', note )
		self.assertIn( 't: __test__.Thing', note )

# --- multi-statement @inline bodies (generalization of PLAN_INLINE.md) -------

class InlineMultiStatementTests( unittest.TestCase ):
	''' @inline generalized to accept locals/if/for/while before a single,
	final, un-nested `return <expr>` - see _splice_multi_statement_inline_
	body. import_builtins=False, same as InlineTests. '''
	maxDiff = None

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = False )
		self.compiler = Compiler( self.discovery )

	def _import( self, code: str ):
		return self.compiler.import_code( code, filename = Path( '__test__.py' ))

	def _lower_main( self ) -> LoweredFunction:
		fn = self.compiler._lower( self.discovery.main )
		self.assertEqual( type( fn ), LoweredFunction )
		return fn

	# compiler.run() force-enqueues windows/_console.py's own console-codepage
	# global on every Windows target, and sys.exit() whenever no_crt (see
	# Compiler.force_reachable's own comment) - real but incidental to what
	# these tests check, and absent entirely on non-Windows targets (or on
	# CRT-linked Windows targets, for sys.exit specifically), so exact-set
	# assertions filter them back out first
	_CONSOLE_INIT_QUALNAMES = frozenset({
		'windows._console._init_console', 'windows.kernel32.SetConsoleOutputCP',
		'sys.exit', 'windows.kernel32.ExitProcess',
		'sys.memset', 'sys.memcpy', 'windows.ntdll.RtlFillMemory', 'windows.ntdll.RtlCopyMemory',
	})

	def _function_qualnames( self ) -> set[str]:
		return { lf.function.qualname for lf in self.compiler.functions } - self._CONSOLE_INIT_QUALNAMES

	def test_multistatement_body_emits_no_call_funcstart_funcend_for_target( self ) -> None:
		code = '\n'.join([
			'@cstruct',
			'class Widget:',
			'	y: usize',
			'	@inline',
			'	def doubled( self ) -> usize:',
			'		tmp: usize = self.y',
			'		return tmp',
			'',
			'def main( w: Widget ) -> usize:',
			'	return w.doubled()',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertNotIn( 'Call', kinds ) # doubled() itself never becomes a real call
		self.assertIn( 'GetAttr', kinds ) # ...just self.y spliced directly
		# the alpha-renamed local must be a real, uniquely-named Variable,
		# not the callee's own literal 'tmp'
		assigns = [ i for i in fn.instructions if isinstance( i, ir.Assign ) and isinstance( i.dest, Variable ) ]
		self.assertTrue( any( a.dest.stem.startswith( '$inline' ) and 'tmp' in a.dest.stem for a in assigns ) )

	def test_caller_local_with_same_name_as_inline_local_not_corrupted( self ) -> None:
		# Hazard 1 regression: without alpha-renaming + the provisional-
		# Function fix, the inlined body's own `tmp` would silently
		# overwrite main's OWN `tmp` in the shared names dict
		code = '\n'.join([
			'@cstruct',
			'class Widget:',
			'	y: usize',
			'	@inline',
			'	def doubled( self ) -> usize:',
			'		tmp: usize = self.y',
			'		return tmp',
			'',
			'def main( w: Widget ) -> usize:',
			'	tmp: usize = 100',
			'	result: usize = w.doubled()',
			'	with compiler.wrap_arithmetic:',
			'		return tmp + result',
		])
		self.discovery.import_name( 'builtins' )
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		qualnames = { i.dest.qualname for i in fn.instructions if hasattr( i, 'dest' ) and isinstance( getattr( i, 'dest', None ), Variable ) }
		self.assertIn( 'main.tmp', qualnames )
		# main's own `tmp` must still be the operand referenced by the
		# final AddWrap - not silently replaced by the inlined one
		add = next( i for i in fn.instructions if type( i ).__name__ == 'AddWrap' )
		self.assertEqual( add.left.qualname, 'main.tmp' )

	def test_reassigning_own_local_reuses_the_same_variable( self ) -> None:
		# Hazard 2 regression: a SECOND assignment to an already-alpha-
		# renamed local must find and replace the FIRST binding (one
		# decref-then-replace), not silently create a second, independent
		# Variable under the same stem (a leak - cfg.py's own fresh-vs-
		# replace machinery depends on finding the SAME Variable object
		# both times)
		code = '\n'.join([
			'@cstruct',
			'class Widget:',
			'	y: usize',
			'	def other( self ) -> usize:',
			'		return self.y',
			'	@inline',
			'	def pick( self ) -> usize:',
			'		tmp: usize = self.y',
			'		tmp = self.other()',
			'		return tmp',
			'',
			'def main( w: Widget ) -> usize:',
			'	return w.pick()',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		assigns = [ i for i in fn.instructions if isinstance( i, ir.Assign ) and isinstance( i.dest, Variable ) and 'tmp' in i.dest.stem ]
		self.assertEqual( len( assigns ), 2 )
		self.assertIs( assigns[0].dest, assigns[1].dest ) # SAME Variable object, not two independent bindings
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( [ c.target.qualname for c in calls ], [ '__test__.Widget.other' ] )

	def test_spliced_if_nests_inside_callers_own_if( self ) -> None:
		code = '\n'.join([
			'@cstruct',
			'class Widget:',
			'	y: usize',
			'	@inline',
			'	def doubled( self ) -> usize:',
			'		tmp: usize = self.y',
			'		if tmp == 0:',
			'			tmp = 1',
			'		return tmp',
			'',
			'def main( w: Widget, flag: bool ) -> usize:',
			'	if flag:',
			'		return w.doubled()',
			'	return 0',
		])
		self.discovery.import_name( 'builtins' ) # tmp == 0 is now an ordinary usize.__eq__ dunder call
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_match_statement_in_pre_return_statement( self ) -> None:
		code = '\n'.join([
			'@union',
			'class Choice:',
			'	A: i32',
			'	B: usize',
			'',
			'@cstruct',
			'class Widget:',
			'	c: Choice',
			'	@inline',
			'	def resolve_choice( self ) -> i32:',
			'		result: i32 = 0',
			'		match self.c:',
			'			case Choice.A( x ):',
			'				result = x',
			'			case Choice.B( y ):',
			'				result = 1',
			'		return result',
			'',
			'def main( w: Widget ) -> i32:',
			'	return w.resolve_choice()',
		])
		self.discovery.import_name( 'builtins' ) # the match's own tag dispatch is now an ordinary u8.__eq__ dunder call
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertNotIn( 'Call', kinds ) # resolve_choice() itself never becomes a real call

	def test_or_return_in_pre_return_statement_now_works( self ) -> None:
		# early/nested-return + defer/errdefer/.or_return() generalization -
		# a pre-return statement's own .or_return() early exit now jumps to
		# the SPLICE's own local epilogue (cfg.py's push_inline_scope/
		# current_epilogue_label), never the caller's real one. This is the
		# exact regression the user specifically flagged as "notably
		# important": .or_return() from inside an @inline must NOT trigger
		# a return from the calling function - only jump to the end of the
		# spliced/embedded scope, letting the caller's OWN subsequent code
		# still run. Result is a bare @union with no declared or_return
		# method (matching lib/builtins's own real shape - or_return() is
		# recognized structurally, never a real method - see discovery.py's
		# reserved-name rejection)
		code = '\n'.join([
			'class MyError: pass',
			'',
			'@union',
			'class Result[T,E]:',
			'	Ok: T',
			'	Err: E',
			'',
			'	def is_err( self ) -> bool:',
			'		return self.tag == 1',
			'',
			'@cstruct',
			'class Widget:',
			'	def risky( self ) -> Result[usize,MyError]:',
			'		return Result.Ok( 1 )',
			'	@inline',
			'	def bad( self ) -> Result[usize,MyError]:',
			'		tmp: usize = self.risky().or_return()',
			'		return Result.Ok( tmp )',
			'',
			'def main( w: Widget ) -> Result[usize,MyError]:',
			'	if w.bad().is_err():',
			'		pass',
			'	return Result.Ok( 5 )',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		# exactly one OrJump (bad()'s own risky().or_return()), and its
		# target is a real, declared label local to the splice
		or_jumps = [ i for i in fn.instructions if isinstance( i, ir.OrJump ) ]
		self.assertEqual( len( or_jumps ), 1 )
		labels = { i.name for i in fn.instructions if isinstance( i, ir.Label ) }
		self.assertIn( or_jumps[0].target, labels ) # a real, declared label - not a dangling reference
		# the decisive check: main() must still have exactly ONE real
		# ir.Return and ONE ir.FuncEnd - bad()'s own internal early exit
		# must never produce a SEPARATE return/funcend for the CALLER, and
		# main's own trailing `return Result.Ok( 5 )` must be the only one
		returns = [ i for i in fn.instructions if isinstance( i, ir.Return ) ]
		func_ends = [ i for i in fn.instructions if isinstance( i, ir.FuncEnd ) ]
		self.assertEqual( len( returns ), 1 )
		self.assertEqual( len( func_ends ), 1 )
		self.assertIs( fn.instructions[-1], func_ends[0] ) # main's own real end, not cut short mid-body

	def test_early_return_nested_in_if_jumps_to_local_scope_not_caller( self ) -> None:
		# early/nested-return generalization - a `return` nested inside a
		# spliced if must land at a label local to the splice, not the
		# caller's own shared epilogue label, and must never produce a real
		# ir.Return/ir.FuncEnd mid-body (those belong to the caller alone)
		code = '\n'.join([
			'@cstruct',
			'class Widget:',
			'	y: i32',
			'	@inline',
			'	def clamped( self ) -> i32:',
			'		if self.y < 0:',
			'			return 0',
			'		return self.y',
			'',
			'def main( w: Widget ) -> i32:',
			'	x: i32 = w.clamped()',
			'	y: i32 = x',
			'	return y',
		])
		self.discovery.import_name( 'builtins' ) # self.y < 0 is now an ordinary i32.__lt__ dunder call
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		# exactly one real ir.Return (main's own trailing `return y`) -
		# clamped()'s own internal early `return 0` must never produce a
		# SECOND one - and it's the second-to-last instruction, immediately
		# before ir.FuncEnd, not buried mid-body ahead of dead code
		returns = [ i for i in fn.instructions if isinstance( i, ir.Return ) ]
		func_ends = [ i for i in fn.instructions if isinstance( i, ir.FuncEnd ) ]
		self.assertEqual( len( returns ), 1 )
		self.assertEqual( len( func_ends ), 1 )
		self.assertIs( fn.instructions[-1], func_ends[0] )
		self.assertIs( fn.instructions[-2], returns[0] )
		# main's own trailing statement (x + 1) must actually be reachable/
		# present - not skipped by clamped()'s own internal early return
		self.assertTrue( any(
			isinstance( i, ir.Assign ) and isinstance( i.dest, Variable ) and 'x' in i.dest.stem
			for i in fn.instructions
		))

	def test_defer_in_spliced_body_replayed_once_at_splice_ladder( self ) -> None:
		# defer/errdefer generalization - a defer registered inside a
		# spliced body's own pre-return statements is now allowed, and must
		# be replayed exactly once, at the splice's own local ladder - not
		# at the caller's real epilogue, and not duplicated between an
		# early exit and the normal fallthrough path
		code = '\n'.join([
			'@cstruct',
			'class Widget:',
			'	y: i32',
			'	@inline',
			'	def traced( self ) -> i32:',
			'		result: i32 = self.y',
			'		with defer:',
			'			result = result',
			'		return result',
			'',
			'def main( w: Widget ) -> i32:',
			'	return w.traced()',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		# exactly one defer flag armed (Const True Assign into a
		# $defer_flag-named Variable), matching exactly one defer statement
		flag_arms = [
			i for i in fn.instructions
			if isinstance( i, ir.Assign ) and isinstance( i.dest, Variable ) and 'defer_flag' in i.dest.stem
			and isinstance( i.src, ir.Const ) and i.src.value is True
		]
		self.assertEqual( len( flag_arms ), 1 )

	def test_direct_recursion_in_pre_return_statement_rejected( self ) -> None:
		code = '\n'.join([
			'@inline',
			'def foo( x: i32 ) -> i32:',
			'	tmp: i32 = foo( x )',
			'	return tmp',
			'',
			'def main() -> i32:',
			'	return foo( 1 )',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any( 'recursive inlining' in e for e in self.discovery.errors.errors ))

	def test_generic_multistatement_inline_splices_bare_call( self ) -> None:
		code = '\n'.join([
			'@cstruct',
			'class Widget:',
			'	y: usize',
			'	def get( self ) -> usize:',
			'		return self.y',
			'',
			'@inline',
			'def wrap[T]( t: T ) -> usize:',
			'	tmp: usize = t.get()',
			'	return tmp',
			'',
			'def main( w: Widget ) -> usize:',
			'	return wrap( w )',
		])
		self._import( code )
		self.compiler.run()
		self.assertEqual( self.discovery.errors.errors, [] )
		qualnames = self._function_qualnames()
		self.assertEqual( qualnames, { 'main', '__test__.Widget.get' } ) # never a real wrap[Widget] function

	def test_multistatement_inline_with_return_only_type_param( self ) -> None:
		# composes with PLAN_RETURN_INFERENCE.md - R is inferred from the
		# body's own final return, even though R never appears in any
		# parameter, and the body is multi-statement
		code = '\n'.join([
			'@cstruct',
			'class Widget:',
			'	y: usize',
			'	def get( self ) -> usize:',
			'		return self.y',
			'',
			'@inline',
			'def wrap[T,R]( t: T ) -> R:',
			'	tmp = t.get()',
			'	return tmp',
			'',
			'def main( w: Widget ) -> usize:',
			'	return wrap( w )',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		# t.get() is a real (non-@inline) method - a real Call to it is
		# expected; wrap() ITSELF must never become one
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( [ c.target.qualname for c in calls ], [ '__test__.Widget.get' ] )
		self.assertEqual( fn.function.return_type.stem, 'usize' )

# --- return-only generic type-parameter inference (PLAN_RETURN_INFERENCE.md) -

class ReturnOnlyTypeParamInferenceTests( unittest.TestCase ):
	''' A generic function whose return type is a bare type param appearing
	in NO parameter (only in the return annotation) - inferred by eagerly
	lowering the body once every other, argument-bound type param is known,
	generalizing _expr_Lambda's own eager-lowering trick (PLAN_LAMBDA.md)
	from an unbound Callable's own return type to a named generic
	function's own return type. import_builtins=False, same as InlineTests. '''
	maxDiff = None

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = False )
		self.compiler = Compiler( self.discovery )

	def _import( self, code: str ):
		return self.compiler.import_code( code, filename = Path( '__test__.py' ))

	def _lower_main( self ) -> LoweredFunction:
		fn = self.compiler._lower( self.discovery.main )
		self.assertEqual( type( fn ), LoweredFunction )
		return fn

	def test_return_only_param_inferred_from_body( self ) -> None:
		# def make[T,R](t: T) -> R: return t.derive() - R never appears in
		# any parameter, only knowable by lowering the body once T is bound
		code = '\n'.join([
			'@cstruct',
			'class Widget:',
			'	y: i32',
			'	def derive( self ) -> bool:',
			'		return self.y != 0',
			'',
			'def make[T,R]( t: T ) -> R:',
			'	return t.derive()',
			'',
			'def main( w: Widget ) -> bool:',
			'	return make( w )',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		# main's own instructions show ONE real Call: to the monomorphized
		# make[Widget,bool] itself (non-@inline - not spliced) - the nested
		# call to Widget.derive lives inside make's OWN separately compiled
		# body, not here
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( len( calls ), 1 )
		target = calls[0].target
		self.assertEqual( target.qualname, '__test__.make[__test__.Widget,intrinsics.bool]' )
		# the discovered R must be the real, concrete bool - not a leaked
		# bare TypeVar or some other default
		self.assertEqual( target.return_type.stem, 'bool' )
		self.assertEqual( calls[0].dest.type.stem, 'bool' )

	def test_return_only_param_infers_from_multistatement_body( self ) -> None:
		# the user's own forcing example: locals/branches before the single
		# `return <expr>` - multi-statement bodies are fine, only multiple
		# RETURN POINTS are restricted (see test_two_return_points_rejected)
		code = '\n'.join([
			'@cstruct',
			'class Widget:',
			'	y: i32',
			'	def derive( self ) -> bool:',
			'		return self.y != 0',
			'',
			'def make[T,R]( factory: T ) -> R:',
			'	obj = factory.derive()',
			'	extra: i32 = 1',
			'	if extra == 1:',
			'		obj = obj',
			'	return obj',
			'',
			'def main( w: Widget ) -> bool:',
			'	return make( w )',
		])
		self.discovery.import_name( 'builtins' ) # self.y != 0 / extra == 1 are now ordinary i32.__ne__/__eq__ dunder calls
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_return_only_param_mixed_with_argument_bound_param( self ) -> None:
		# make[T,R](t: T) -> Pair[T,R] - T is ordinarily argument-bound (it
		# occurs in `t: T`), R is return-only - the return type mentions
		# BOTH, exercising _unify_type_param's Specialization-args recursion
		# to fill in R while re-confirming T. Pair has a real __init__
		# (RCClass construction-arg inference, not @cstruct field=value
		# sugar) here - this used to be load-bearing (a generic @cstruct with
		# NO __init__ had a real, pre-existing gap: _lower_allocate_fields's
		# own construction only ever inferred its class's own concrete type
		# args from the surrounding expected_type context, never from the
		# field VALUES themselves), now fixed by _infer_allocate_type_args -
		# see test_bare_construct_infers_type_args_from_field_values_with_no_expected_type
		# for that same shape (`Pair(first=t, second=t.derive())`, no
		# expected_type) covered directly. Kept as __init__-based here since
		# it's still valid, independent coverage of the Specialization-args
		# recursion for THAT path.
		code = '\n'.join([
			'class Pair[T,R]:',
			'	first: T',
			'	second: R',
			'	def __init__( self, first: T, second: R ) -> None:',
			'		self.first = first',
			'		self.second = second',
			'',
			'@cstruct',
			'class Widget:',
			'	y: i32',
			'	def derive( self ) -> bool:',
			'		return self.y != 0',
			'',
			'def make[T,R]( t: T ) -> Pair[T,R]:',
			'	return Pair( t, t.derive() )',
			'',
			'def main( w: Widget ) -> None:',
			'	p: Pair[Widget,bool] = make( w )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_two_return_points_rejected( self ) -> None:
		code = '\n'.join([
			'@cstruct',
			'class Widget:',
			'	y: i32',
			'	def derive( self ) -> bool:',
			'		return self.y != 0',
			'',
			'def make[T,R]( t: T ) -> R:',
			'	if t.derive():',
			'		return t.derive()',
			'	return t.derive()',
			'',
			'def main( w: Widget ) -> bool:',
			'	return make( w )',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any( 'exactly one reachable' in e for e in self.discovery.errors.errors ))

	def test_bare_return_rejected( self ) -> None:
		code = '\n'.join([
			'@cstruct',
			'class Widget:',
			'	y: i32',
			'	def derive( self ) -> bool:',
			'		return self.y != 0',
			'',
			'def make[T,R]( t: T ) -> R:',
			'	return',
			'',
			'def main( w: Widget ) -> bool:',
			'	return make( w )',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any( 'exactly one reachable' in e for e in self.discovery.errors.errors ))

	def test_direct_recursion_rejected( self ) -> None:
		code = '\n'.join([
			'@cstruct',
			'class Widget:',
			'	pass',
			'',
			'def make[T,R]( t: T ) -> R:',
			'	return make( t )',
			'',
			'def main( w: Widget ) -> None:',
			'	make( w )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any( 'recursively calls itself' in e for e in self.discovery.errors.errors ))

	def test_mutual_recursion_rejected( self ) -> None:
		code = '\n'.join([
			'@cstruct',
			'class Widget:',
			'	pass',
			'',
			'def ping[T,R]( t: T ) -> R:',
			'	return pong( t )',
			'',
			'def pong[T,R]( t: T ) -> R:',
			'	return ping( t )',
			'',
			'def main( w: Widget ) -> None:',
			'	ping( w )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any( 'recursively calls itself' in e for e in self.discovery.errors.errors ))

	def test_two_call_sites_same_binding_reuse_one_compiled_function( self ) -> None:
		code = '\n'.join([
			'@cstruct',
			'class Widget:',
			'	y: i32',
			'	def derive( self ) -> bool:',
			'		return self.y != 0',
			'',
			'def make[T,R]( t: T ) -> R:',
			'	return t.derive()',
			'',
			'def main( w1: Widget, w2: Widget ) -> None:',
			'	x: bool = make( w1 )',
			'	y: bool = make( w2 )',
			'	return',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) and i.target.qualname.startswith( '__test__.make' ) ]
		self.assertEqual( len( calls ), 2 )
		self.assertIs( calls[0].target, calls[1].target )

	def test_vacuous_type_param_alongside_return_only_still_fails( self ) -> None:
		# S appears in NEITHER any parameter NOR the return type - a
		# degenerate case, left to fail like anything else genuinely
		# missing, not silently accepted just because R (which DOES appear
		# in the return type) happens to be inferable
		code = '\n'.join([
			'@cstruct',
			'class Widget:',
			'	y: i32',
			'	def derive( self ) -> bool:',
			'		return self.y != 0',
			'',
			'def make[T,R,S]( t: T ) -> R:',
			'	return t.derive()',
			'',
			'def main( w: Widget ) -> bool:',
			'	return make( w )',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any( 'cannot infer type parameter(s)' in e and 'S' in e for e in self.discovery.errors.errors ))

	def test_inline_return_only_param_infers_and_still_splices( self ) -> None:
		# @inline get_len[T,R](t: T) -> R: return t.__len__() - a synthetic
		# T whose __len__ returns i32 (not the usual usize), to actually
		# exercise inference rather than coincidentally matching a
		# hard-coded type. Must still emit no real Call/FuncStart/FuncEnd
		# for get_len itself (PLAN_INLINE.md's own invariant preserved)
		code = '\n'.join([
			'@cstruct',
			'class Box:',
			'	y: i32',
			'	def __len__( self ) -> i32:',
			'		return self.y',
			'',
			'@inline',
			'def get_len[T,R]( t: T ) -> R:',
			'	return t.__len__()',
			'',
			'def main( b: Box ) -> i32:',
			'	return get_len( b )',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( [ c.target.qualname for c in calls ], [ '__test__.Box.__len__' ] )
		self.assertEqual( fn.function.return_type.stem, 'i32' )

	def test_inline_return_only_two_call_sites_each_get_independent_splice( self ) -> None:
		# unlike the non-inline case, @inline is ALWAYS spliced per call
		# site - two call sites must NOT share one compiled unit's worth of
		# Call, they each get their own independent Call to __len__ (the
		# discovered R binding is what's shared/cached, not the splice)
		code = '\n'.join([
			'@cstruct',
			'class Box:',
			'	y: i32',
			'	def __len__( self ) -> i32:',
			'		return self.y',
			'',
			'@inline',
			'def get_len[T,R]( t: T ) -> R:',
			'	return t.__len__()',
			'',
			'def main( b1: Box, b2: Box ) -> i32:',
			'	x: i32 = get_len( b1 )',
			'	y: i32 = get_len( b2 )',
			'	return x',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( [ c.target.qualname for c in calls ], [ '__test__.Box.__len__', '__test__.Box.__len__' ] )

	def test_inline_discarding_return_only_result_type_rejected( self ) -> None:
		code = '\n'.join([
			'class MyError: pass',
			'',
			'@union',
			'class Result[T,E]:',
			'	Ok: T',
			'	Err: E',
			'',
			'@inline',
			'def make[T,R]( t: T ) -> R:',
			'	return t.wrap()',
			'',
			'@cstruct',
			'class Widget:',
			'	def wrap( self ) -> Result[i32,MyError]:',
			'		return Result.Ok( 1 )',
			'',
			'def main( w: Widget ) -> None:',
			'	make( w )',
			'	return',
		])
		self._import( code )
		self._lower_main()
		self.assertTrue( any( 'requires the enclosing function to return Result' in e for e in self.discovery.errors.errors ))

# --- compiler.fetch_unicode_table('upper'|'lower') ---------------------------

class FetchUnicodeTableTests( unittest.TestCase ):
	''' structural invariants only (sorted, no duplicate codepoints, valid
	codepoint range, ASCII agrees with the obvious A<->a relationship) - NOT
	specific mappings pinned to a specific Unicode version, per
	PLAN_CASE_FOLDING.md's own explicit "no version pinning" decision (a
	newer UCD release changing some obscure codepoint's mapping should never
	break this suite). METALPY_UNICODE_DATA_DIR points every test at a
	small, hand-written UnicodeData.txt fixture, so none of this ever
	touches the network or depends on what "latest" resolves to today - see
	_fetch_unicode_data_txt's own env-override support in lowering.py. '''

	_UNICODE_DATA_TXT = '\n'.join([
		'0041;LATIN CAPITAL LETTER A;Lu;0;L;;;;;N;;;;0061;',
		'0061;LATIN SMALL LETTER A;Ll;0;L;;;;;N;;;0041;;',
		'0042;LATIN CAPITAL LETTER B;Lu;0;L;;;;;N;;;;0062;',
		'0062;LATIN SMALL LETTER B;Ll;0;L;;;;;N;;;0042;;',
		'00DF;LATIN SMALL LETTER SHARP S;Ll;0;L;;;;;N;;;;;', # real UCD entry: no simple uppercase (needs SpecialCasing.txt's ß->SS, out of scope) - must be skipped, not crash
		'', # trailing blank line - must be skipped, not crash
	])

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )
		self._env_cleanup()

	def _env_cleanup( self ) -> None:
		import os
		old = os.environ.get( 'METALPY_UNICODE_DATA_DIR' )
		def _restore() -> None:
			if old is None:
				os.environ.pop( 'METALPY_UNICODE_DATA_DIR', None )
			else:
				os.environ['METALPY_UNICODE_DATA_DIR'] = old
		self.addCleanup( _restore )

	def _point_env_at_fixture( self, contents: str ) -> None:
		import os
		import tempfile
		tmp = tempfile.TemporaryDirectory()
		self.addCleanup( tmp.cleanup )
		( Path( tmp.name ) / 'UnicodeData.txt' ).write_text( contents, encoding = 'utf-8' )
		os.environ['METALPY_UNICODE_DATA_DIR'] = tmp.name

	def _fetch( self, which: str ) -> bytes:
		code = '\n'.join([
			'import compiler',
			f"TABLE: bytes = compiler.fetch_unicode_table( {which!r} )",
			'',
			'def main() -> None:',
			'	x: bytes = TABLE',
			'	return',
		])
		self.compiler.import_code( code, Path( '__test__.py' ), scope = None )
		self.compiler.run()
		self.assertEqual( self.discovery.errors.errors, [] )
		g = self.compiler.globals[0]
		# PLAN_THREAD_SAFE_SHARED_STATE.md Part A: an RC-typed global's own
		# init now leads with AcquireGlobalLock/Decref before the Assign
		assigns = [ i for i in g.instructions if isinstance( i, ir.Assign ) ]
		self.assertEqual( len( assigns ), 1 )
		assign = assigns[0]
		const = assign.src
		self.assertIsInstance( const, ir.Const )
		self.assertIsInstance( const.value, bytes )
		return const.value

	def _decode( self, table: bytes ) -> list[tuple[int,int]]:
		self.assertEqual( len( table ) % 8, 0, 'table must be a whole number of 8-byte (codepoint,mapped) entries' )
		return [
			( int.from_bytes( table[i:i+4], 'little' ), int.from_bytes( table[i+4:i+8], 'little' ))
			for i in range( 0, len( table ), 8 )
		]

	def test_upper_table_matches_fixture_exactly( self ) -> None:
		self._point_env_at_fixture( self._UNICODE_DATA_TXT )
		# only 'a'->'A' and 'b'->'B' have a simple uppercase mapping in the
		# fixture ('A'/'B' are already upper, 'ß' has none - see comment above)
		self.assertEqual( self._decode( self._fetch( 'upper' )), [ ( 0x61, 0x41 ), ( 0x62, 0x42 ) ] )

	def test_lower_table_matches_fixture_exactly( self ) -> None:
		self._point_env_at_fixture( self._UNICODE_DATA_TXT )
		self.assertEqual( self._decode( self._fetch( 'lower' )), [ ( 0x41, 0x61 ), ( 0x42, 0x62 ) ] )

	def test_structural_invariants_hold( self ) -> None:
		self._point_env_at_fixture( self._UNICODE_DATA_TXT )
		for which in ( 'upper', 'lower' ):
			entries = self._decode( self._fetch( which ))
			codepoints = [ cp for cp, _mapped in entries ]
			self.assertEqual( codepoints, sorted( codepoints ), f'{which} table must be sorted ascending for binary search' )
			self.assertEqual( len( codepoints ), len( set( codepoints )), f'{which} table must have no duplicate codepoints' )
			for cp, mapped in entries:
				self.assertLessEqual( cp, 0x10FFFF, f'{which}: codepoint {cp:#x} exceeds Unicode\'s own max' )
				self.assertLessEqual( mapped, 0x10FFFF, f'{which}: mapped codepoint {mapped:#x} exceeds Unicode\'s own max' )

	def test_missing_override_file_is_a_compile_error_not_a_crash( self ) -> None:
		import tempfile
		import os
		tmp = tempfile.TemporaryDirectory()
		self.addCleanup( tmp.cleanup )
		os.environ['METALPY_UNICODE_DATA_DIR'] = tmp.name # real dir, but no UnicodeData.txt in it
		code = '\n'.join([
			'import compiler',
			"TABLE: bytes = compiler.fetch_unicode_table( 'upper' )",
			'',
			'def main() -> None:',
			'	x: bytes = TABLE',
			'	return',
		])
		self.compiler.import_code( code, Path( '__test__.py' ), scope = None )
		self.compiler.run()
		self.assertTrue( any( 'does not exist' in e for e in self.discovery.errors.errors ))

class JoinedStrLoweringTests( unittest.TestCase ):
	''' f-string (PEP 498) runtime lowering (PLAN_FSTRINGS.md) -
	_expr_JoinedStr/_lower_fstring_part in lowering.py. Needs real str/
	Result/UnsafeList[T] (the runtime N-part path), same import_builtins=
	True convention emitter_c_test.py's own FStringTests uses - compile_
	time_transformer_test.py's own JoinedStrFoldingTests already covers
	the compile-time-constant fold in isolation, and emitter_c_test.py's
	FStringTests covers real end-to-end compile-and-run behavior; this
	class checks the actual IR SHAPE the runtime path produces (proving
	it's really UnsafeList[str]/str.concat, not N-1 chained str.__add__
	calls) and the conversion/error-reporting rules a pure instruction-
	shape check can't see from emitter_c_test.py alone. '''
	maxDiff = None

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _import( self, code: str ):
		return self.compiler.import_code( code, filename = Path( '__test__.py' ))

	def _lower_main( self ) -> LoweredFunction:
		fn = self.compiler._lower( self.discovery.main )
		self.assertEqual( type( fn ), LoweredFunction )
		return fn

	def _calls_to( self, fn: LoweredFunction, needle: str ) -> list[ir.Call]:
		return [ i for i in fn.instructions if isinstance( i, ir.Call ) and needle in i.target.qualname ]

	def _allocates_of( self, fn: LoweredFunction, needle: str ) -> list[ir.Allocate]:
		return [ i for i in fn.instructions if isinstance( i, ir.Allocate ) and needle in i.cls.qualname ]

	def test_single_formatted_value_short_circuits_no_concat_machinery( self ) -> None:
		# f"{x}" alone (x: str, a real runtime value - not foldable) -
		# len(node.values) == 1, PLAN_FSTRINGS.md's own short-circuit: the
		# FormattedValue's own str-typed operand is used directly, no
		# UnsafeList/str.concat machinery at all
		self._import( '\n'.join([
			'def main( x: str ) -> str:',
			'	return f"{x}"',
		]))
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertEqual( self._allocates_of( fn, 'UnsafeList' ), [] )
		self.assertEqual( self._calls_to( fn, 'concat' ), [] )
		# the Return's own value IS x's own parameter, reused directly - no
		# synthesized alias, no __str__ call (x is already str-typed)
		ret = next( i for i in fn.instructions if isinstance( i, ir.Return ) )
		self.assertIs( ret.value, fn.function.parameters[0] )

	def test_multipart_runtime_path_uses_unsafelist_concat_not_chained_add( self ) -> None:
		# f"{a}{b}" (a, b: str, both real runtime values) - exactly 2
		# parts, so exactly 2 UnsafeList[str].append() calls, exactly 1
		# UnsafeList[str] Allocate, exactly 1 str.concat call taking that
		# SAME buffer directly (str.concat's own parameter type is
		# UnsafeList[str] - no intermediate view/copy step at all anymore,
		# see lowering.py's own _expr_JoinedStr comment) and, the actual
		# point of this whole pass, ZERO calls to str.__add__ (proving this
		# ISN'T N-1 chained string concatenation)
		self._import( '\n'.join([
			'def main( a: str, b: str ) -> str:',
			'	return f"{a}{b}"',
		]))
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertEqual( len( self._allocates_of( fn, 'UnsafeList' )), 1 )
		self.assertEqual( len( self._calls_to( fn, '.append' )), 2 )
		self.assertEqual( len( self._calls_to( fn, '.concat' )), 1 )
		self.assertEqual( self._calls_to( fn, '__add__' ), [] )

	def test_literal_and_runtime_value_mixed( self ) -> None:
		# f"a{x}b" (x: str runtime) - N=3 parts (Constant('a'),
		# FormattedValue(x), Constant('b')), none of them individually
		# foldable together with x, so the whole thing stays a real
		# runtime JoinedStr and takes the N>=2 path with exactly 3 parts
		self._import( '\n'.join([
			'def main( x: str ) -> str:',
			'	return f"a{x}b"',
		]))
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertEqual( len( self._calls_to( fn, '.append' )), 3 )
		self.assertEqual( len( self._calls_to( fn, '.concat' )), 1 )

	def test_no_conversion_uses_str_dunder( self ) -> None:
		self._import( '\n'.join([
			'def main( n: int ) -> str:',
			'	return f"{n}"',
		]))
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		calls = self._calls_to( fn, '__str__' )
		self.assertEqual( len( calls ), 1 )
		self.assertEqual( calls[0].target.stem, '__str__' )

	def test_bang_s_conversion_uses_str_dunder( self ) -> None:
		self._import( '\n'.join([
			'def main( n: int ) -> str:',
			'	return f"{n!s}"',
		]))
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertEqual( len( self._calls_to( fn, '__str__' )), 1 )

	def test_bang_r_conversion_uses_repr_dunder( self ) -> None:
		self._import( '\n'.join([
			'def main( n: int ) -> str:',
			'	return f"{n!r}"',
		]))
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertEqual( len( self._calls_to( fn, '__repr__' )), 1 )
		self.assertEqual( self._calls_to( fn, '__str__' ), [] )

	def test_bang_a_conversion_dispatches_to_repr_then_ascii_escape( self ) -> None:
		# !a (PLAN_FSTRINGS.md follow-up): __repr__() first (same as !r),
		# then str._ascii_escape() on the result - _lower_ascii_escape's
		# own comment on why it does NOT add quotes, unlike Python's real
		# ascii()/repr()
		self._import( '\n'.join([
			'def main( n: int ) -> str:',
			'	return f"{n!a}"',
		]))
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertEqual( len( self._calls_to( fn, '__repr__' )), 1 )
		self.assertEqual( len( self._calls_to( fn, '_ascii_escape' )), 1 )
		self.assertEqual( self._calls_to( fn, '__str__' ), [] )

	def test_dynamic_format_spec_is_a_compile_error( self ) -> None:
		# f"{n:{w}}" - the format_spec itself is an ast.JoinedStr holding a
		# real ast.FormattedValue(w), not just ast.Constant pieces -
		# _is_literal_format_spec rejects it before ever trying to parse
		# any joined text as a spec string
		self._import( '\n'.join([
			'def main( n: int, w: int ) -> str:',
			'	return f"{n:{w}}"',
		]))
		self.compiler._lower( self.discovery.main )
		self.assertTrue(
			any( 'format spec' in e and 'dynamic format specs are not supported yet' in e for e in self.discovery.errors.errors ),
			self.discovery.errors.errors,
		)

	def test_float_type_char_on_int_is_a_compile_error( self ) -> None:
		# f"{n:.2f}" - 'f' parses fine (FORMAT_SPEC_TYPE_CHARS includes the
		# float type chars precisely so this can name them), but
		# validate_int_spec rejects 'f' against an int operand with a
		# dedicated message pointing at the missing float feature, not a
		# generic "not valid for int" message
		self._import( '\n'.join([
			'def main( n: int ) -> str:',
			'	return f"{n:.2f}"',
		]))
		self.compiler._lower( self.discovery.main )
		self.assertTrue(
			any( "needs a real float type with formatting support, which doesn't exist yet" in e for e in self.discovery.errors.errors ),
			self.discovery.errors.errors,
		)

	def test_float_format_spec_dispatches_to_fixed_digits_and_sign_prefix( self ) -> None:
		# f"{x:.1f}" (f64) - dispatches to lib/builtins/__float.py's own
		# _fixed_digits/_sign_prefix (Scalar-registered methods, not
		# __str__/__repr__ - a format spec formats the operand's own type
		# directly, matching Python's format(x, spec) == type(x).
		# __format__(x, spec) semantics, same as int's own dispatch)
		self._import( '\n'.join([
			'def main( x: f64 ) -> str:',
			'	return f"{x:.1f}"',
		]))
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertEqual( len( self._calls_to( fn, '_fixed_digits' )), 1 )
		self.assertEqual( len( self._calls_to( fn, '_sign_prefix' )), 1 )
		self.assertEqual( self._calls_to( fn, '__str__' ), [] )

	def test_float_format_spec_exponential_dispatches_to_fixed_digits( self ) -> None:
		# f"{x:.2e}" - dispatches through the same _fixed_digits as 'f'/'F',
		# just with a different type_char argument (PLAN_STR_FORMAT.md item 4)
		self._import( '\n'.join([
			'def main( x: f64 ) -> str:',
			'	return f"{x:.2e}"',
		]))
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertEqual( len( self._calls_to( fn, '_fixed_digits' )), 1 )
		self.assertEqual( len( self._calls_to( fn, '_sign_prefix' )), 1 )

	def test_float_format_spec_percent_dispatches_to_percent_digits( self ) -> None:
		# f"{x:.2%}" - '%' has no printf equivalent, so it goes through its
		# own dedicated _percent_digits (lib/builtins/__float.py) instead of
		# _fixed_digits - the *100 scaling + 'f' + '%' suffix all happen
		# there, not here
		self._import( '\n'.join([
			'def main( x: f64 ) -> str:',
			'	return f"{x:.2%}"',
		]))
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertEqual( len( self._calls_to( fn, '_percent_digits' )), 1 )
		self.assertEqual( self._calls_to( fn, '_fixed_digits' ), [] )
		self.assertEqual( len( self._calls_to( fn, '_sign_prefix' )), 1 )

	def test_float_format_spec_none_type_with_precision_dispatches_to_none_type_digits( self ) -> None:
		# f"{x:.2}" - a literal spec with a precision but no type char at
		# all goes through its own dedicated _none_type_digits (lib/
		# builtins/__float.py, real Python's own "None" presentation type -
		# closer to 'g' than 'f', see its own comment), not the plain
		# _fixed_digits('f') fallback f"{x:.2f}" itself would dispatch to
		self._import( '\n'.join([
			'def main( x: f64 ) -> str:',
			'	return f"{x:.2}"',
		]))
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertEqual( len( self._calls_to( fn, '_none_type_digits' )), 1 )
		self.assertEqual( self._calls_to( fn, '_fixed_digits' ), [] )
		self.assertEqual( self._calls_to( fn, '_percent_digits' ), [] )
		self.assertEqual( len( self._calls_to( fn, '_sign_prefix' )), 1 )

	def test_bare_float_interpolation_dispatches_to_str_dunder( self ) -> None:
		# f"{x}" with NO format spec at all (not even an empty ":") never
		# reaches _lower_float_format_spec - parsed_spec is None, so this
		# takes the plain __str__/__repr__ dispatch path (same as str/int),
		# which f64/f32 now have real implementations of (the shortest-
		# round-trip repr algorithm, PLAN_STR_FORMAT.md item 4's own later
		# writeup) instead of failing to compile. The dispatched Call's own
		# target.qualname is the underlying def's real name (_f64_str, from
		# `f64.__str__ = _f64_str`'s registration, lib/builtins/__float.py)
		# - NOT the literal string '__str__', which is only ever a KEY in
		# f64.names, never the Function's own identity.
		self._import( '\n'.join([
			'def main( x: f64 ) -> str:',
			'	return f"{x}"',
		]))
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertEqual( len( self._calls_to( fn, '_f64_str' )), 1 )
		self.assertEqual( self._calls_to( fn, '_fixed_digits' ), [] )
		self.assertEqual( self._calls_to( fn, '_none_type_digits' ), [] )

	def test_float_format_spec_no_type_no_precision_dispatches_to_repr_digits( self ) -> None:
		# f"{x:10}" - an explicit spec (width only, no type char, no
		# precision) DOES reach _lower_float_format_spec, which routes this
		# exact combination to _repr_digits (same underlying shortest-
		# round-trip algorithm bare f"{x}" uses, just padded afterward) -
		# distinct from both _fixed_digits (needs a type char or precision)
		# and _none_type_digits (needs a precision)
		self._import( '\n'.join([
			'def main( x: f64 ) -> str:',
			'	return f"{x:10}"',
		]))
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertEqual( len( self._calls_to( fn, '_repr_digits' )), 1 )
		self.assertEqual( self._calls_to( fn, '_fixed_digits' ), [] )
		self.assertEqual( self._calls_to( fn, '_none_type_digits' ), [] )
		self.assertEqual( len( self._calls_to( fn, '_sign_prefix' )), 1 )

	def test_invalid_float_type_char_is_a_compile_error( self ) -> None:
		# f"{x:x}" - 'x' is a valid int type char but not a float one
		self._import( '\n'.join([
			'def main( x: f64 ) -> str:',
			'	return f"{x:x}"',
		]))
		self.compiler._lower( self.discovery.main )
		self.assertTrue(
			any( "'x' is not valid for float" in e for e in self.discovery.errors.errors ),
			self.discovery.errors.errors,
		)

	def test_int_type_char_on_str_is_a_compile_error( self ) -> None:
		# f"{s:x}" - 'x' parses fine but validate_str_spec rejects any
		# type char other than 's'/None for a str operand
		self._import( '\n'.join([
			'def main( s: str ) -> str:',
			'	return f"{s:x}"',
		]))
		self.compiler._lower( self.discovery.main )
		self.assertTrue(
			any( "not valid for str" in e for e in self.discovery.errors.errors ), self.discovery.errors.errors,
		)

	def test_precision_on_int_is_a_compile_error( self ) -> None:
		self._import( '\n'.join([
			'def main( n: int ) -> str:',
			'	return f"{n:.2d}"',
		]))
		self.compiler._lower( self.discovery.main )
		self.assertTrue(
			any( 'precision is not allowed for int' in e for e in self.discovery.errors.errors ), self.discovery.errors.errors,
		)

	def test_literal_str_format_spec_width_dispatches_to_rjust( self ) -> None:
		# f"{s:>10}" - operand is already str-typed, no explicit
		# !conversion, so the spec dispatches straight against s itself
		# (_lower_dispatch_format_spec -> _lower_str_format_spec); '>'
		# align maps to .rjust() via _lower_pad_by_align
		self._import( '\n'.join([
			'def main( s: str ) -> str:',
			'	return f"{s:>10}"',
		]))
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertEqual( len( self._calls_to( fn, 'rjust' )), 1 )
		self.assertEqual( self._calls_to( fn, 'ljust' ), [] )
		self.assertEqual( self._calls_to( fn, 'center' ), [] )

	def test_literal_str_format_spec_center_align_dispatches_to_center( self ) -> None:
		self._import( '\n'.join([
			'def main( s: str ) -> str:',
			'	return f"{s:^10}"',
		]))
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertEqual( len( self._calls_to( fn, 'center' )), 1 )

	def test_literal_str_format_spec_precision_dispatches_to_truncate( self ) -> None:
		self._import( '\n'.join([
			'def main( s: str ) -> str:',
			'	return f"{s:.3}"',
		]))
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertEqual( len( self._calls_to( fn, '_truncate_codepoints' )), 1 )

	def test_bang_r_conversion_plus_format_spec_dispatches_against_the_resulting_str( self ) -> None:
		# f"{n!r:>10}" - an explicit conversion reduces n to str FIRST
		# (via __repr__), and the spec then formats THAT str, not n's own
		# int type - so this dispatches through the str branch (.rjust),
		# never through any int-specific method
		self._import( '\n'.join([
			'def main( n: int ) -> str:',
			'	return f"{n!r:>10}"',
		]))
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertEqual( len( self._calls_to( fn, '__repr__' )), 1 )
		self.assertEqual( len( self._calls_to( fn, 'rjust' )), 1 )
		self.assertEqual( self._calls_to( fn, '_to_radix_digits' ), [] )
		self.assertEqual( self._calls_to( fn, '_decimal_digits_with_grouping' ), [] )

	def test_literal_int_format_spec_decimal_dispatches_to_sign_and_digits( self ) -> None:
		# f"{n:05d}" - decimal path: _decimal_digits (raw, ungrouped) +
		# _sign_prefix, then grouping-aware zero-pad via
		# _pad_and_group_after_prefix (the '=' align implied by the '0'
		# shorthand) rather than a generic ljust/rjust/center call or the
		# plain (non-grouping-aware) _pad_after_prefix - see str._pad_and_
		# group_after_prefix's own comment on why the zero-pad path needs
		# raw digits and grouping-aware padding, not pre-grouped digits
		# plus a naive rjust (PLAN_STR_FORMAT.md item 4's own bugfix note)
		self._import( '\n'.join([
			'def main( n: int ) -> str:',
			'	return f"{n:05d}"',
		]))
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertEqual( len( self._calls_to( fn, '_decimal_digits' )), 1 )
		self.assertEqual( len( self._calls_to( fn, '_sign_prefix' )), 1 )
		self.assertEqual( len( self._calls_to( fn, '_pad_and_group_after_prefix' )), 1 )
		self.assertEqual( self._calls_to( fn, '_decimal_digits_with_grouping' ), [] )
		self.assertEqual( self._calls_to( fn, '_pad_after_prefix' ), [] )
		self.assertEqual( self._calls_to( fn, '_to_radix_digits' ), [] )

	def test_literal_int_format_spec_hex_dispatches_to_radix_digits( self ) -> None:
		self._import( '\n'.join([
			'def main( n: int ) -> str:',
			'	return f"{n:#x}"',
		]))
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertEqual( len( self._calls_to( fn, '_to_radix_digits' )), 1 )
		self.assertEqual( self._calls_to( fn, '_decimal_digits_with_grouping' ), [] )

	def test_literal_int_format_spec_no_width_skips_padding_helpers( self ) -> None:
		# f"{n:x}" - no width at all, so neither _pad_after_prefix nor
		# ljust/rjust/center should be reached; the digits + sign/prefix
		# concat (str.__add__) is the whole story
		self._import( '\n'.join([
			'def main( n: int ) -> str:',
			'	return f"{n:x}"',
		]))
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertEqual( self._calls_to( fn, '_pad_after_prefix' ), [] )
		self.assertEqual( self._calls_to( fn, 'rjust' ), [] )
		self.assertGreaterEqual( len( self._calls_to( fn, '__add__' )), 1 )

	def test_value_with_no_str_or_repr_is_a_compile_error( self ) -> None:
		self._import( '\n'.join([
			'@cstruct',
			'class Foo: pass',
			'',
			'def main( f: Foo ) -> str:',
			'	return f"{f}"',
		]))
		self.compiler._lower( self.discovery.main )
		self.assertTrue(
			any( 'Foo' in e and '__str__' in e for e in self.discovery.errors.errors ), self.discovery.errors.errors,
		)

	def test_class_attribute_access_in_fstring_is_a_clean_compile_error( self ) -> None:
		# `Foo.flag` (accessed through the CLASS, not an instance) used to
		# silently reach emitter_c.py as a bare undeclared identifier -
		# find_name_recursive's scope-terminal branch resolved the chain
		# fine, but nothing ever rejected it. Real-world repro: grap.mpy's
		# `f'... = {Grap.one!r}'`.
		self._import( '\n'.join([
			'class Foo:',
			'	flag: bool = False',
			'',
			'def main() -> str:',
			'	return f"{Foo.flag!r}"',
		]))
		self.compiler._lower( self.discovery.main )
		self.assertTrue(
			any( 'class attribute access is not supported' in e for e in self.discovery.errors.errors ),
			self.discovery.errors.errors,
		)

	def test_class_attribute_access_in_fstring_errors_even_with_a_same_class_instance_in_scope( self ) -> None:
		# the same check above used to be bypassed entirely when an
		# instance of the SAME class also existed as a local: `flag`'s own
		# Variable lives in Foo.names (populated once Foo is resolved via
		# construction), and the scope-terminal branch returned it directly
		# as if it were a real class-level global before ever reaching the
		# class-attribute check - is_global (False for anything class-
		# scoped) is what now excludes it. grap.mpy's real shape.
		self._import( '\n'.join([
			'class Foo:',
			'	flag: bool = False',
			'',
			'def main() -> str:',
			'	foo = Foo()',
			'	return f"{Foo.flag!r}"',
		]))
		self.compiler._lower( self.discovery.main )
		self.assertTrue(
			any( 'class attribute access is not supported' in e for e in self.discovery.errors.errors ),
			self.discovery.errors.errors,
		)

	def test_self_documenting_equals_syntax_needs_no_special_handling( self ) -> None:
		# f"{x=}" - CPython's own parser already expands this into an
		# extra literal Constant('x=') ahead of the FormattedValue before
		# metalpy ever sees the tree (PLAN_FSTRINGS.md's own Context
		# section) - just confirms it compiles and takes the expected
		# 2-part runtime path (Constant('x='), FormattedValue(x))
		self._import( '\n'.join([
			'def main( x: str ) -> str:',
			'	return f"{x=}"',
		]))
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertEqual( len( self._calls_to( fn, '.append' )), 2 )

class AssignabilityCheckTests( unittest.TestCase ):
	''' lowering.py's new general assignability check (_lower_expr's
	_check_assignable, plus the safe-scalar-widening coercion and the
	_expr_Constant literal-kind validation) - closes a previously self-
	documented gap ("a genuine argument-type mismatch isn't checked
	anywhere yet"). import_builtins=True throughout: several cases need
	real str/RCClass, and scalars/Ptr are always available as intrinsics
	regardless, so one setUp covers every case here. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _import( self, code: str ):
		return self.compiler.import_code( code, filename = Path( '__test__.py' ))

	def _lower_main( self ) -> LoweredFunction:
		fn = self.compiler._lower( self.discovery.main )
		self.assertEqual( type( fn ), LoweredFunction )
		return fn

	def _assert_rejected( self, code: str, needle: str = 'expected' ) -> None:
		self._import( code )
		self.compiler._lower( self.discovery.main )
		self.assertTrue(
			any( needle in e for e in self.discovery.errors.errors ),
			f'expected an error containing {needle!r}, got: {self.discovery.errors.errors}',
		)

	def _assert_accepted( self, code: str ) -> LoweredFunction:
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		return fn

	# --- must now reject: the original bug repro, verbatim -------------------

	def test_call_argument_type_mismatch_is_rejected( self ) -> None:
		self._assert_rejected( '\n'.join([
			'def foo( x: str ) -> None: return',
			'def main() -> None:',
			'	y: i32 = 5',
			'	foo( y )',
		]))

	def test_annassign_type_mismatch_is_rejected( self ) -> None:
		self._assert_rejected( '\n'.join([
			'def main() -> None:',
			'	y: i32 = 5',
			'	z: str = y',
		]))

	def test_reassignment_type_mismatch_is_rejected( self ) -> None:
		self._assert_rejected( '\n'.join([
			'def main() -> None:',
			'	y: i32 = 5',
			"	y = 'hello'",
		]))

	# --- must now reject: the remaining call sites ----------------------------

	def test_return_type_mismatch_is_rejected( self ) -> None:
		# _stmt_Return has its OWN dedicated compatibility check (including
		# error-union widening) rather than relying on the general
		# _check_assignable (see lowering.py's own comment on why that
		# call is strict=False) - own error wording, not "expected ..."
		self._assert_rejected( '\n'.join([
			'def main() -> str:',
			'	y: i32 = 5',
			'	return y',
		]), needle = 'function returns' )

	def test_attribute_assignment_type_mismatch_is_rejected( self ) -> None:
		self._assert_rejected( '\n'.join([
			'class Foo:',
			'	s: str',
			'	def __init__( self ) -> None:',
			"		self.s = ''",
			'def main() -> None:',
			'	f: Foo = Foo()',
			'	y: i32 = 5',
			'	f.s = y',
		]))

	def test_augassign_type_mismatch_is_rejected( self ) -> None:
		self._assert_rejected( '\n'.join([
			'def main() -> None:',
			'	y: i32 = 5',
			'	y += True',
		]))

	# --- must now reject: literal-kind mismatches -----------------------------

	def test_int_literal_into_str_is_rejected( self ) -> None:
		self._assert_rejected( "def main() -> None:\n\tz: str = 5" )

	def test_int_literal_into_bool_is_rejected( self ) -> None:
		self._assert_rejected( 'def main() -> None:\n\tflag: bool = 5' )

	def test_bool_literal_into_non_bool_scalar_is_rejected( self ) -> None:
		self._assert_rejected( 'def main() -> None:\n\tx: i32 = True' )

	def test_none_literal_into_non_nullable_rcclass_is_rejected( self ) -> None:
		self._assert_rejected( 'def main() -> None:\n\ts: str = None' )

	# --- must now reject: narrowing / sign-changing / isize-usize scalars -----

	def test_narrowing_scalar_assignment_is_rejected( self ) -> None:
		self._assert_rejected( '\n'.join([
			'def main() -> None:',
			'	big: i64 = 5',
			'	small: i32 = big',
		]))

	def test_sign_changing_scalar_assignment_is_rejected( self ) -> None:
		self._assert_rejected( '\n'.join([
			'def main() -> None:',
			'	s: i32 = 5',
			'	u: u32 = s',
		]))

	def test_usize_excluded_from_widening_is_rejected( self ) -> None:
		self._assert_rejected( '\n'.join([
			'def main() -> None:',
			'	s: usize = 5',
			'	w: i64 = s',
		]))

	# --- must keep working: safe scalar widening ------------------------------

	def test_i32_widens_to_i64_via_real_castwrap( self ) -> None:
		fn = self._assert_accepted( '\n'.join([
			'def main() -> None:',
			'	s: i32 = 5',
			'	w: i64 = s',
		]))
		self.assertTrue( any( isinstance( i, ir.CastWrap ) for i in fn.instructions ) )

	def test_u8_widens_to_u32_via_real_castwrap( self ) -> None:
		fn = self._assert_accepted( '\n'.join([
			'def main() -> None:',
			'	s: u8 = 5',
			'	w: u32 = s',
		]))
		self.assertTrue( any( isinstance( i, ir.CastWrap ) for i in fn.instructions ) )

	def test_f32_widens_to_f64_via_real_castwrap( self ) -> None:
		fn = self._assert_accepted( '\n'.join([
			'def main() -> None:',
			'	s: f32 = 1.5',
			'	w: f64 = s',
		]))
		self.assertTrue( any( isinstance( i, ir.CastWrap ) for i in fn.instructions ) )

	def test_float_operand_widths_still_require_an_explicit_cast_in_arithmetic( self ) -> None:
		# the one deliberate exception to "safe widening is always implicit":
		# _lower_binary_operands' own hint-passing is strict=False specifically
		# so THIS keeps failing via _lower_binop_values' own stricter float-
		# same-type rule, not silently widened by the general mechanism above
		self._assert_rejected( '\n'.join([
			'def main() -> None:',
			'	with compiler.wrap_arithmetic:',
			'		a: f64 = 1.0',
			'		b: f32 = 2.0',
			'		c: f64 = a + b',
		]), needle = 'same type' )

	# --- must keep working: pre-existing coercions, unaffected ----------------

	def test_rcclass_upcast_to_base_stays_implicit( self ) -> None:
		self._assert_accepted( '\n'.join([
			'class Base:',
			'	def __init__( self ) -> None: return',
			'class Derived( Base ):',
			'	def __init__( self ) -> None: return',
			'def foo( b: Base ) -> None: return',
			'def main() -> None:',
			'	d: Derived = Derived()',
			'	foo( d )',
		]))

	def test_tagged_union_member_coercion_stays_implicit( self ) -> None:
		self._assert_accepted( '\n'.join([
			'def main() -> None:',
			'	a: i32|str = 5',
			"	b: i32|str = 'x'",
		]))

	def test_explicit_scalar_cast_still_works( self ) -> None:
		self._assert_accepted( '\n'.join([
			'def main() -> None:',
			'	s: i32 = 5',
			'	with compiler.wrap_arithmetic:',
			'		w: u32 = u32( s )',
		]))

	def test_literal_to_scalar_assignment_still_works( self ) -> None:
		self._assert_accepted( '\n'.join([
			'def main() -> None:',
			'	x: i64 = 5',
			'	y: u32 = 10',
			'	f: f32 = 1.5',
		]))

	def test_nullable_pointer_literal_still_works( self ) -> None:
		self._assert_accepted( 'def main() -> None:\n\tp: Ptr[u8] = None' )

	def test_pointer_coercion_generalized_to_a_call_result_not_just_name_or_attribute( self ) -> None:
		# _maybe_castwrap_pointer used to only be reachable from _expr_Name/
		# _expr_Attribute directly - now a third _lower_expr branch applies
		# it to ANY expression kind. compiler.addrof(x) always produces a
		# fixed Ptr[T] regardless of expected_type (lowering.py's own
		# _lower_compiler_addrof never consults it), dispatched through
		# _expr_Call, not _expr_Name/_expr_Attribute - assigning it straight
		# into a ConstPtr[T]-typed target needs the SAME interchangeable-
		# pointer coercion those two node kinds already got on their own
		fn = self._assert_accepted( '\n'.join([
			'def main() -> None:',
			'	x: u8 = 0',
			'	p: ConstPtr[u8] = compiler.addrof( x )',
		]))
		self.assertTrue( any( isinstance( i, ir.CastWrap ) for i in fn.instructions ) )

	# --- Overload-blindness follow-up: bare method-value reference -----------

	def test_bare_overloaded_method_reference_gets_a_specific_diagnostic( self ) -> None:
		# _find_method's own isinstance(found, Function) check silently
		# discards a real Overload group - _expr_Attribute used to fall
		# through to _attr_lookup's generic "no such attribute" message for
		# this, even though the attribute plainly exists (Result.unwrap_or
		# is a real, shipped Overload - two same-named defs in
		# lib/builtins/__init__.py). An overloaded method genuinely has no
		# single signature to bind as a bare value/closure - this needs a
		# clear message saying THAT, not a wrong "doesn't exist" one. Both
		# paths were already a compile error either way; only the message
		# changes
		self._assert_rejected( '\n'.join([
			'def main() -> i32:',
			'	r: Result[i32,TypeError] = Result.Ok( 5 )',
			'	f = r.unwrap_or',
			'	return 0',
		]), needle = 'overloaded method' )

	def test_calling_an_overloaded_method_normally_still_works( self ) -> None:
		# additive-only: the new bare-value-reference diagnostic above must
		# not affect the ordinary call-with-args path at all, which routes
		# through _lower_method_call (Overload-aware resolution), never
		# _find_method/this new check
		self._assert_accepted( '\n'.join([
			'def main() -> i32:',
			'	r: Result[i32,TypeError] = Result.Ok( 5 )',
			'	v: i32 = r.unwrap_or( 0 )',
			'	if v != 5:',
			'		return 1',
			'	return 0',
		]))


class WalrusOperatorTests( unittest.TestCase ):
	''' _expr_NamedExpr (ast.NamedExpr, `x := expr`) - PLAN_POSIX_FEATURE.md's
	scope. Mirrors _stmt_Assign's own two ast.Name-target branches, but
	returns the operand as this expression's own value. All locals here are
	function-scoped unconditionally (not block-scoped), so a walrus binding
	made inside an if/while condition is expected to stay visible in code
	textually after it, same as an ordinary preceding assignment would be -
	these tests confirm that isn't just true by inspection, but actually
	holds once real CFG/binding machinery runs. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _import( self, code: str ):
		return self.compiler.import_code( code, filename = Path( '__test__.py' ))

	def _assert_accepted( self, code: str ) -> LoweredFunction:
		self._import( code )
		fn = self.compiler._lower( self.discovery.main )
		self.assertEqual( type( fn ), LoweredFunction )
		self.assertEqual( self.discovery.errors.errors, [] )
		return fn

	def test_binding_visible_after_the_if_statement( self ) -> None:
		fn = self._assert_accepted( '\n'.join([
			'def f() -> i32:',
			'	return 5',
			'def main() -> i32:',
			'	if ( x := f() ) != 5:',
			'		return 1',
			'	if x != 5:',
			'		return 2',
			'	return 0',
		]))
		# two real reads of the SAME walrus-bound Variable - one inside the
		# if's own condition, one textually after the if - not two
		# different bindings that happen to share a name
		assigns = [ i for i in fn.instructions if isinstance( i, ir.Assign ) and i.dest.stem == 'x' ]
		self.assertEqual( len( assigns ), 1 )

	def test_binding_reused_later_in_the_same_function( self ) -> None:
		fn = self._assert_accepted( '\n'.join([
			'def f() -> i32:',
			'	return 5',
			'def main() -> i32:',
			'	x: i32 = 0',
			'	if ( x := f() ) != 5:',
			'		return 1',
			'	y: i32 = 0',
			'	with compiler.wrap_arithmetic:',
			'		y = x + 1',
			'	if y != 6:',
			'		return 2',
			'	return 0',
		]))
		self.assertTrue( any( isinstance( i, ir.Assign ) and i.dest.stem == 'x' for i in fn.instructions ) )

	def test_nested_walrus_inside_boolean_expression( self ) -> None:
		fn = self._assert_accepted( '\n'.join([
			'def f() -> i32:',
			'	return 5',
			'def main() -> i32:',
			'	if ( a := f() ) != 5 and ( b := f() ) != 6:',
			'		return 1',
			'	if a != 5 or b != 6:',
			'		return 2',
			'	return 0',
		]))
		names = { i.dest.stem for i in fn.instructions if isinstance( i, ir.Assign ) and isinstance( i.dest, Variable ) }
		self.assertIn( 'a', names )
		self.assertIn( 'b', names )

	def test_walrus_rebinding_an_existing_name( self ) -> None:
		# the reassignment branch (existing = discovery.find_name_or_none(...)
		# is not None) - unlike a fresh declaration, must reuse the SAME
		# Variable object, not create a second one under the same stem
		fn = self._assert_accepted( '\n'.join([
			'def main() -> i32:',
			'	x: i32 = 1',
			'	if ( x := 2 ) != 2:',
			'		return 1',
			'	if x != 2:',
			'		return 2',
			'	return 0',
		]))
		assigns = [ i for i in fn.instructions if isinstance( i, ir.Assign ) and i.dest.stem == 'x' ]
		self.assertEqual( len( assigns ), 2 ) # the initial x: i32 = 1, then the walrus rebind
		self.assertIs( assigns[0].dest, assigns[1].dest )


class ListLiteralTests( unittest.TestCase ):
	''' _expr_List (ast.List, `[a, b, c]`) - PLAN_POSIX_FEATURE.md's follow-up
	scope (found while unblocking utf8.names()'s own list-literal return).
	Requires expected_type to already be a concrete list[T] Specialization -
	no element-driven inference. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _import( self, code: str ):
		return self.compiler.import_code( code, filename = Path( '__test__.py' ))

	def _assert_accepted( self, code: str ) -> LoweredFunction:
		self._import( code )
		fn = self.compiler._lower( self.discovery.main )
		self.assertEqual( type( fn ), LoweredFunction )
		self.assertEqual( self.discovery.errors.errors, [] )
		return fn

	def _assert_rejected( self, code: str, needle: str ) -> None:
		self._import( code )
		self.compiler._lower( self.discovery.main )
		self.assertTrue(
			any( needle in e for e in self.discovery.errors.errors ),
			f'expected an error containing {needle!r}, got: {self.discovery.errors.errors}',
		)

	def test_construction_and_append_shape( self ) -> None:
		fn = self._assert_accepted( '\n'.join([
			'def main() -> None:',
			"	x: list[str] = [ 'a', 'b' ]",
			'	return',
		]))
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) ]
		append_calls = [ c for c in calls if c.target.stem == 'append' ]
		self.assertEqual( len( append_calls ), 2 )
		# both append calls target the SAME constructed list instance
		self.assertIs( append_calls[0].receiver, append_calls[1].receiver )
		# append()'s own return value (None, not a Result) is never assigned
		# to a dest - only its side effect matters, mirroring _expr_Set's
		# own add() handling
		self.assertTrue( all( c.dest is None for c in append_calls ) )

	def test_empty_list_literal_is_construction_only( self ) -> None:
		fn = self._assert_accepted( '\n'.join([
			'def main() -> None:',
			'	x: list[i32] = []',
			'	return',
		]))
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) ]
		self.assertFalse( any( c.target.stem == 'append' for c in calls ) )

	def test_wrong_element_type_is_rejected( self ) -> None:
		self._assert_rejected( '\n'.join([
			'def main() -> None:',
			"	x: list[str] = [ 'a', 5 ]",
			'	return',
		]), needle = 'expected' )

	def test_no_expected_type_is_rejected( self ) -> None:
		self._assert_rejected( '\n'.join([
			'def main() -> None:',
			"	x = [ 'a', 'b' ]",
			'	return',
		]), needle = 'list literal needs a known list[T] target type' )

	def test_list_literal_into_optional_list_annotation( self ) -> None:
		# expected_type here is `list[str]|None` (a TaggedUnion), not a bare
		# list[str] Specialization - _expr_List must narrow through the
		# union to find its own list[T] member rather than failing outright
		fn = self._assert_accepted( '\n'.join([
			'def main() -> i32:',
			"	x: list[str]|None = [ 'a', 'b' ]",
			'	if x is None:',
			'		return 1',
			'	return 0',
		]))
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( len([ c for c in calls if c.target.stem == 'append' ]), 2 )

	def test_list_literal_into_optional_list_call_argument( self ) -> None:
		# the exact reported call shape: a defaulted `list[str]|None`
		# parameter, list literal passed by keyword at the call site
		fn = self._assert_accepted( '\n'.join([
			'def take( xs: list[str]|None = None ) -> i32:',
			'	if xs is None:',
			'		return 0',
			'	return len( xs )',
			'',
			'def main() -> i32:',
			"	return take( xs = [ 'a', 'b' ] )",
		]))
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_list_literal_into_ambiguous_union_is_still_rejected( self ) -> None:
		# two DIFFERENT list[T] members - genuinely ambiguous which one the
		# literal targets, so this must still fail rather than guess
		self._assert_rejected( '\n'.join([
			'def main() -> i32:',
			"	x: list[str]|list[i32] = [ 'a', 'b' ]",
			'	return 0',
		]), needle = 'list literal needs a known list[T] target type' )


class SetLiteralTests( unittest.TestCase ):
	''' _expr_Set (ast.Set, `{a, b, c}`) - mirrors ListLiteralTests above.
	Requires expected_type to already be a concrete set[T] Specialization -
	no element-driven inference, same precedent as list literals. No
	empty-literal test here (unlike ListLiteralTests' own
	test_empty_list_literal_is_construction_only): `{}` is unconditionally
	an empty ast.Dict in Python's own grammar, never an empty ast.Set - the
	defensive `if not node.elts` guard in _expr_Set exists only for a
	synthetically-built AST node, not reachable from real source text. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _import( self, code: str ):
		return self.compiler.import_code( code, filename = Path( '__test__.py' ))

	def _assert_accepted( self, code: str ) -> LoweredFunction:
		self._import( code )
		fn = self.compiler._lower( self.discovery.main )
		self.assertEqual( type( fn ), LoweredFunction )
		self.assertEqual( self.discovery.errors.errors, [] )
		return fn

	def _assert_rejected( self, code: str, needle: str ) -> None:
		self._import( code )
		self.compiler._lower( self.discovery.main )
		self.assertTrue(
			any( needle in e for e in self.discovery.errors.errors ),
			f'expected an error containing {needle!r}, got: {self.discovery.errors.errors}',
		)

	def test_construction_and_add_shape( self ) -> None:
		fn = self._assert_accepted( '\n'.join([
			'def main() -> None:',
			"	x: set[str] = { 'a', 'b' }",
			'	return',
		]))
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) ]
		add_calls = [ c for c in calls if c.target.stem == 'add' ]
		self.assertEqual( len( add_calls ), 2 )
		# both add calls target the SAME constructed set instance
		self.assertIs( add_calls[0].receiver, add_calls[1].receiver )
		# unlike list[T].append, set[T].add returns plain None - no
		# unwrap()/Result dance needed for a set literal
		self.assertFalse( any( c.target.stem == 'unwrap' for c in calls ) )

	def test_wrong_element_type_is_rejected( self ) -> None:
		self._assert_rejected( '\n'.join([
			'def main() -> None:',
			"	x: set[str] = { 'a', 5 }",
			'	return',
		]), needle = 'expected' )

	def test_no_expected_type_is_rejected( self ) -> None:
		self._assert_rejected( '\n'.join([
			'def main() -> None:',
			"	x = { 'a', 'b' }",
			'	return',
		]), needle = 'set literal needs a known set[T] target type' )

	def test_set_literal_into_optional_set_annotation( self ) -> None:
		# mirrors ListLiteralTests.test_list_literal_into_optional_list_annotation
		fn = self._assert_accepted( '\n'.join([
			'def main() -> i32:',
			"	x: set[str]|None = { 'a', 'b' }",
			'	if x is None:',
			'		return 1',
			'	return 0',
		]))
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) ]
		self.assertEqual( len([ c for c in calls if c.target.stem == 'add' ]), 2 )


class MoveParameterTests( unittest.TestCase ):
	''' move[T] is an ownership status on a binding, not a distinct type
	from T (Parameter.is_move/is_copy, not a Move/Copy-wrapped .type -
	discovery.py's own parameter-construction site unwraps it). Before this
	fix, NO property of a move[T]-typed parameter could be read at all
	inside the function that owns it - confirmed via lib/builtins/
	__init__.py's own real bytes.from_bytearray/str.from_cstr, both of
	which read len(src) before consuming src via .release(). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _import( self, code: str ):
		return self.compiler.import_code( code, filename = Path( '__test__.py' ))

	def _assert_accepted( self, code: str ) -> LoweredFunction:
		self._import( code )
		fn = self.compiler._lower( self.discovery.main )
		self.assertEqual( type( fn ), LoweredFunction )
		self.assertEqual( self.discovery.errors.errors, [] )
		return fn

	def test_generic_len_call_on_move_parameter( self ) -> None:
		# the exact reported shape: len[T](t: T) is an ORDINARY (non-move)
		# generic free function - a move[bytearray]-typed argument flowing
		# into its plain T parameter must infer T as the unwrapped
		# bytearray, not the wrapped ownership annotation
		self._assert_accepted( '\n'.join([
			'def consume( src: move[bytearray] ) -> usize:',
			'	n: usize = len( src )',
			'	return n',
			'def main() -> usize:',
			'	b: bytearray = bytearray( 5 )',
			'	return consume( move( b ))',
		]))

	def test_direct_method_call_on_move_parameter( self ) -> None:
		# bypasses the generic len() entirely - an ordinary, direct method
		# call on a move[T]-typed parameter must resolve through the same
		# attribute lookup any other binding's method call would
		self._assert_accepted( '\n'.join([
			'def consume( src: move[bytearray] ) -> usize:',
			'	return src.__len__()',
			'def main() -> usize:',
			'	b: bytearray = bytearray( 5 )',
			'	return consume( move( b ))',
		]))

	def test_move_parameter_type_is_unwrapped( self ) -> None:
		mod = self._import( '\n'.join([
			'def consume( src: move[bytearray] ) -> usize:',
			'	return src.__len__()',
			'def main() -> usize:',
			'	b: bytearray = bytearray( 5 )',
			'	return consume( move( b ))',
		]))
		consume = mod.get_local( 'consume' )
		consume.resolve()
		src_param = consume.parameters[0]
		self.assertTrue( src_param.is_move )
		self.assertFalse( src_param.is_copy )
		self.assertEqual( src_param.type.qualname, 'builtins.bytearray' )

	def test_receiver_not_double_decreffed_after_move_method_call( self ) -> None:
		# @move on a METHOD (bytearray.release()'s own real shape) means
		# calling it consumes/invalidates self - before this fix, nothing
		# transitioned the CALLER's own ownership state for the RECEIVER,
		# so a real Decref still got emitted for b at scope exit on top of
		# release()'s own internal cleanup (a genuine double-free,
		# confirmed via a real intermittent ~10-15% test-suite flake).
		# cfg.py's own move() (already used by _apply_move_hook for move[T]
		# ARGUMENTS) cancels the receiver's own pending epilogue Decref -
		# note this does NOT reject reading b again afterward (confirmed:
		# neither does the pre-existing move[T]-argument mechanism this
		# mirrors) - it only stops the double teardown, which is exactly
		# the bug being fixed here.
		fn = self._assert_accepted( '\n'.join([
			'def main() -> i32:',
			'	b: bytearray = bytearray( 5 )',
			'	match b.release():',
			'		case Result.Ok( ptr ):',
			'			return 0',
			'		case Result.Err( _ ):',
			'			return 1',
		]))
		decrefs_on_b = [
			i for i in fn.instructions
			if isinstance( i, ir.Decref ) and getattr( i.value, 'stem', None ) == 'b'
		]
		self.assertEqual( decrefs_on_b, [] )


class DefaultValueModuleContextTests( unittest.TestCase ):
	''' _lower_call_args' own default-value-lowering loop (the branch that
	fills in a parameter the CALLER omitted) used to lower param.default
	with whatever module/scope context happened to be active - the
	CALLER's own, since that's what's active while lowering the caller's
	body - instead of pushing the callee's own module/scope first, unlike
	every other default-lowering site in this file (field defaults,
	@inline splicing). A default value that references a name private to
	the callee's own module (module-scoped, never imported by the caller)
	would then fail to resolve AT ALL under the caller's own context - not
	just a misattributed error location, a genuine false compile failure
	(confirmed via a real repro: lib/posix/time.py's own default-driven
	'utf8' is not a value, the real bug turned out to be lib/posix/fs.py's
	own default value). '''

	def test_default_value_referencing_a_callee_module_private_name( self ) -> None:
		import tempfile
		with tempfile.TemporaryDirectory() as tmp:
			root = Path( tmp )
			( root / 'a.py' ).write_text( '\n'.join([
				'SPECIAL: i32 = 5', # never imported by __main__.py below -
				# only resolvable if the default is lowered in a.py's own
				# module context, not __main__.py's
				'def f( x: i32 = SPECIAL ) -> i32:',
				'	return x',
			]), encoding = 'utf-8' )
			( root / '__main__.py' ).write_text( '\n'.join([
				'from a import f',
				'def main() -> i32:',
				'	return f()', # x omitted - forces the default to be lowered
			]), encoding = 'utf-8' )
			discovery = Discovery( paths = [ root ], import_builtins = False )
			compiler = Compiler( discovery )
			compiler.import_file( root / '__main__.py' )
			compiler.run()
			self.assertEqual( discovery.errors.errors, [] )

	def test_default_value_error_is_located_in_the_callee_module_not_the_caller( self ) -> None:
		import tempfile
		with tempfile.TemporaryDirectory() as tmp:
			root = Path( tmp )
			( root / 'a.py' ).write_text( '\n'.join([
				'def f( x: i32 = undefined_name ) -> i32:',
				'	return x',
			]), encoding = 'utf-8' )
			( root / '__main__.py' ).write_text( '\n'.join([
				'from a import f',
				'def main() -> i32:',
				'	return f()',
			]), encoding = 'utf-8' )
			discovery = Discovery( paths = [ root ], import_builtins = False )
			compiler = Compiler( discovery )
			compiler.import_file( root / '__main__.py' )
			compiler.run()
			self.assertEqual( len( discovery.errors.errors ), 1 )
			# located in a.py (where `undefined_name` was actually written),
			# not __main__.py (which merely calls f() with x omitted)
			self.assertIn( 'a.py', discovery.errors.errors[0] )
			self.assertNotIn( '__main__.py', discovery.errors.errors[0] )

	def test_default_value_referencing_module_constant_with_two_overloads( self ) -> None:
		''' a module-level constant referenced as a default value on a method
		with 2+ overloads (plain same-name defs, no @overload needed) - a real
		repro reported as a "name X is not defined" pointing at each
		overload's own signature line, worked around in lib/re.py by
		hardcoding every affected `max_steps: usize = 65536` default instead
		of `= DEFAULT_MAX_STEPS` once search()/match()/etc grew str/bytes/
		memoryview overloads. Not reproducible against current lowering.py -
		this pins that down as a regression test rather than leaving the
		shape unverified. '''
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		compiler.import_code( '\n'.join([
			'CONST: usize = 65536',
			'',
			'class Foo:',
			'	def match( self, a: str, max_steps: usize = CONST ) -> usize:',
			'		return max_steps',
			'',
			'	def match( self, a: bytes, max_steps: usize = CONST ) -> usize:',
			'		return max_steps',
			'',
			'def main() -> i32:',
			'	f: Foo = Foo()',
			'	x: usize = f.match( "hi" )', # x omitted - forces the default to be lowered
			'	if x != 65536:',
			'		return 1',
			'	return 0',
		]), filename = Path( '__main__.py' ))
		compiler.run()
		self.assertEqual( discovery.errors.errors, [] )


class DefaultValueConstructionTests( unittest.TestCase ):
	''' a construction call embedded in a parameter's own default value
	(`def f( x: Foo = Foo() ) -> Foo:`) used to crash lowering.py's own
	_try_lower_construct_call assert ("... was not resolved before
	construction") once a caller actually omitted that argument.
	type_resolver.py's own eager __init__-signature pre-resolution
	(_ReferenceResolver, via resolve_function_body) only ever walks
	fn.node.body - a parameter's default lives on fn.node.args instead,
	which resolve_function_body never visits - so the construction call
	inside it reached real lowering without ever having been pre-resolved,
	exactly the same bug shape resolve_global_init was already added to fix
	for a global variable's own construction-call initializer (confirmed
	via a real repro - see resolve_parameter_default's own docstring). '''

	def test_construction_call_in_a_default_value_does_not_crash( self ) -> None:
		import tempfile
		with tempfile.TemporaryDirectory() as tmp:
			root = Path( tmp )
			( root / 'a.py' ).write_text( '\n'.join([
				'class Foo:',
				'	x: i32',
				'	def __init__( self, x: i32 = 1 ) -> None:',
				'		self.x = x',
				'',
				'def f( x: Foo = Foo() ) -> Foo:', # the only construction of
				# Foo anywhere in this program - never a direct Call node
				# inside any function BODY, only inside this default
				'	return x',
			]), encoding = 'utf-8' )
			( root / '__main__.py' ).write_text( '\n'.join([
				'from a import f',
				'def main() -> i32:',
				'	obj = f()', # x omitted - forces the default to be lowered
				'	return obj.x',
			]), encoding = 'utf-8' )
			# real RCClass construction needs sys.alloc, so builtins (plus the
			# real lib/) has to be importable here, unlike the sibling class's
			# plain-i32-default tests just above
			disco = Discovery(
				paths = [ root, Path( discovery.__file__ ).parent / 'lib' ],
				import_builtins = True,
			)
			compiler = Compiler( disco )
			compiler.import_file( root / '__main__.py' )
			compiler.run()
			self.assertEqual( disco.errors.errors, [] )


class GenericCallDefaultParameterTests( unittest.TestCase ):
	''' _lower_call_args' own default-value-filling tail (fills in a
	parameter the CALLER omitted - see DefaultValue*Tests above) only ever
	ran on the PLAIN, non-generic call path - a bare call to a generic
	function (`take(x)`, T inferred from the argument) goes through
	_lower_inferred_generic_call/_finish_generic_call instead, which built
	args/kwargs straight from what the call site actually wrote and never
	filled in an omitted default at all. Confirmed via a real repro:
	emitter_c.py's _emit_call_args crashed with a bare KeyError on the
	omitted parameter's own stem, reachable in practice specifically via an
	unannotated tuple local (tuple literals are one of the few shapes whose
	type doesn't need an annotation to compile, so they're also the easiest
	way for type_resolver.py's own eager _try_resolve_generic_call pre-pass
	to miss the call entirely - see _type_of_expr's own lack of an ast.Tuple
	branch - and fall through to lowering.py's late, unfixed path). The same
	gap existed at three sibling call sites that also monomorphize a generic
	target's parameters without ever filling in an omitted default:
	_lower_class_generic_method_call (a generic method whose genericity is
	inherited from its class, e.g. Result.Ok/.Err), generic class
	construction (Box(...) where Box[T] is generic), and
	_infer_return_only_type_params_inline's own two _lower_inline_call call
	sites (an @inline generic function whose type param is only inferable
	from its own return type). Fixed by adding one shared
	_fill_generic_call_defaults, mirroring _lower_call_args' own tail,
	called from every site once each has its own monomorphized target in
	hand. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _import( self, code: str ):
		return self.compiler.import_code( code, filename = Path( '__test__.py' ))

	def _lower_main( self ):
		return self.compiler._lower( self.discovery.main )

	def test_bare_generic_call_fills_in_omitted_default_bound_to_a_tuple( self ) -> None:
		code = '\n'.join([
			'def take[S]( seq: S, pad: i32 = 99 ) -> i32:',
			'	return pad',
			'',
			'def main() -> i32:',
			# i32(...) elements, not bare literals - a bare int literal's own
			# natural type is now builtins.int (arbitrary-precision), not
			# i32 (see lowering.py's _expr_Constant); `t` itself STAYS
			# unannotated (still infers straight from the tuple literal,
			# unlike list/slice, which need an explicit annotation to even
			# compile - see this class's own docstring for why that's the
			# shape that actually reaches the buggy path in practice)
			'	t = ( i32( 1 ), i32( 2 ), i32( 3 ))',
			'	return take( t )',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) and getattr( i.target, 'stem', None ) == 'take' ]
		self.assertEqual( len( calls ), 1 )
		self.assertIn( 'pad', calls[0].kwargs )

	def test_generic_class_construction_fills_in_omitted_default( self ) -> None:
		code = '\n'.join([
			'class Box[T]:',
			'	val: T',
			'	pad: i32',
			'',
			'	def __init__( self, val: T, pad: i32 = 77 ) -> None:',
			'		self.val = val',
			'		self.pad = pad',
			'',
			'def main() -> i32:',
			'	b = Box( 5 )',
			'	return b.pad',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		# construction lowers to a call against the synthesized $$__new__
		# (not a bare __init__ call - see _try_lower_construct_call)
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) and getattr( i.target, 'stem', None ) == '$$__new__' ]
		self.assertEqual( len( calls ), 1 )
		self.assertIn( 'pad', calls[0].kwargs )

	def test_inline_return_only_inference_fills_in_omitted_default( self ) -> None:
		# _infer_return_only_type_params_inline (an @inline generic function
		# whose only type param is inferable from its own RETURN type, never
		# any parameter - PLAN_RETURN_INFERENCE.md) splices the body directly
		# rather than emitting an ir.Call at all, via its own two
		# _lower_inline_call call sites - both had the identical gap, one
		# call site further down the same crash chain: no ir.Call/emitter_c.py
		# KeyError here, but the exact same shape one level earlier
		# (_lower_inline_call's own `kwargs[param.stem]` bindings lookup,
		# lowering.py) crashed with a bare KeyError on the omitted parameter
		# - confirmed via a real repro before this fix
		code = '\n'.join([
			'@inline',
			'def make[T]( pad: i32 = 5 ) -> T:',
			'	return pad',
			'',
			'def main() -> i32:',
			'	x: i32 = make()', # pad omitted - forces the default to be spliced in
			'	return x',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		# the splice's own local for `pad` must be assigned the real default
		# (5), not left unbound - inspects the spliced Assign directly since
		# @inline never emits an ir.Call to check kwargs on
		assigns = [ i for i in fn.instructions if isinstance( i, ir.Assign ) and i.dest.stem.endswith( '$pad' ) ]
		self.assertEqual( len( assigns ), 1 )
		self.assertEqual( assigns[0].src, ir.Const( type = self.discovery.get_intrinsics()['i32'], value = 5 ))


class GenericCallDestTypeTests( unittest.TestCase ):
	''' _emit_generic_call used to type a generic call's own dest as
	`expected_type or monomorphized.return_type` - trusting the CALLER's
	own expected type outright instead of the REAL, just-monomorphized
	return type. For a free generic function (unlike
	_lower_class_generic_method_call, e.g. Result.Ok, which already
	unifies expected_type against the class's own type params up front),
	nothing before that point ever required the two to agree - type params
	are inferred purely from the ARGUMENTS. Typing dest as expected_type
	regardless made operand.type == expected_type true BY CONSTRUCTION,
	which skips _lower_expr's own real safety net (_coerce_or_check_operand
	only ever catches a genuine mismatch when the two differ) - so a real
	divergence went completely undetected, reaching emitter_c.py with dest
	LYING about its own type (confirmed via a real repro: a `struct
	builtins$int*` assigned through an `int32_t` dest, "incompatible
	pointer to integer conversion"). Only ever surfaced once a bare int
	literal argument stopped always defaulting to i32 (which happened to
	coincidentally match whatever the caller expected, in practice, every
	time). Fixed by typing dest as the real resolved return type instead,
	letting _lower_expr's own outer _coerce_or_check_operand call catch a
	genuine mismatch exactly like it already does for every other kind of
	operand - a real "expected X, got Y" compile error now, not a silent
	miscompile. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _import( self, code: str ):
		return self.compiler.import_code( code, filename = Path( '__test__.py' ))

	def _lower_main( self ):
		return self.compiler._lower( self.discovery.main )

	def test_generic_return_type_mismatch_is_a_real_compile_error_not_a_miscompile( self ) -> None:
		code = '\n'.join([
			'def apply[T,K]( x: T, key: Ptr[Callable[[T],K]] ) -> K:',
			'	return key( x )',
			'',
			'def main() -> i32:',
			# int(5), not a bare 5 - forces T=builtins.int unambiguously,
			# independent of any literal-default behavior. K is only ever
			# inferable from the LAMBDA's own body (`v` - an identity), which
			# has no declared signature of its own to conflict-check against
			# T/K up front the way apply(int(5), key=identity_i32) (a real
			# function reference with its own concrete i32 signature) would -
			# so K quietly resolves to builtins.int too, genuinely
			# incompatible with the i32 the assignment target below
			# declares. Before the fix, this compiled with NO errors,
			# silently emitting a dest typed i32 for a call that actually
			# returns builtins.int (confirmed via a real repro: raw C
			# "incompatible pointer to integer conversion" reaching the
			# emitted output undetected by MetalPy's own type checker).
			'	result: i32 = apply( int( 5 ), key = lambda v: v )',
			'	return result',
		])
		self._import( code )
		self._lower_main()
		errors = [ str( e ) for e in self.discovery.errors.errors ]
		self.assertEqual( len( errors ), 1, errors )
		self.assertIn( 'expected intrinsics.i32, got builtins.int', errors[0] )

	def test_generic_return_type_match_still_compiles_and_dest_is_correctly_typed( self ) -> None:
		code = '\n'.join([
			'def apply[T,K]( x: T, key: Ptr[Callable[[T],K]] ) -> K:',
			'	return key( x )',
			'',
			'def identity_i32( v: i32 ) -> i32:',
			'	return v',
			'',
			'def main() -> i32:',
			'	result: i32 = apply( i32( 5 ), key = identity_i32 )',
			'	return result',
		])
		self._import( code )
		fn = self._lower_main()
		self.assertEqual( self.discovery.errors.errors, [] )
		i32 = self.discovery.get_intrinsics()['i32']
		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) and getattr( i.target, 'stem', None ) == 'apply' ]
		self.assertEqual( len( calls ), 1 )
		self.assertIs( calls[0].dest.type, i32 )


class OverloadMoveResolutionTests( unittest.TestCase ):
	''' move(...) sugar used to only be recognized once a single concrete
	Function target was already chosen (_check_move_argument, reachable
	from _match_call_args) - never for an Overload group, since
	_lower_overload_arg fell through to ordinary name resolution instead,
	where `move` isn't a real registered name anywhere ("name 'move' is
	not defined": confirmed via str.from_cstr(move(b)), which has both a
	(ConstPtr[u8], usize) and a move[bytearray] overload). Fixed in
	_lower_call's own Overload branch: move(...) is peeled before
	candidate type-matching (so the peeled argument's plain type, e.g.
	bytearray, can match the move[bytearray] candidate), then - once
	resolve_call settles on a single concrete winner - validated against
	that winner's own is_move-ness and the real ownership-transfer hook
	(_apply_move_hook, i.e. cfg.move()) is applied, mirroring what
	_lower_call_args already does for a plain, non-overloaded target.
	IR-level (not real-compile) coverage: a real compile-and-run test
	against str.from_cstr specifically is blocked by a separate, general,
	pre-existing bug this investigation also found - two overload
	candidates that both need real C bodies in the same program collide on
	an identical mangled C symbol name (emitter_c.py never disambiguates
	between candidates sharing one qualname) - tracked separately, not
	this fix's own scope. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _import( self, code: str ):
		return self.compiler.import_code( code, filename = Path( '__test__.py' ))

	def test_move_call_dispatches_to_the_move_parameter_overload_and_transfers_ownership( self ) -> None:
		code = '\n'.join([
			'@overload',
			'def make( n: i32 ) -> usize:',
			'	...',
			'',
			'def make( n: i32 = 0 ) -> usize:',
			'	return usize( n )',
			'',
			'def make( src: move[bytearray] ) -> usize:',
			'	return len( src )',
			'',
			'def main() -> usize:',
			'	b: bytearray = bytearray( 5 )',
			'	return make( move( b ))',
		])
		self._import( code )
		fn = self.compiler._lower( self.discovery.main )
		self.assertEqual( self.discovery.errors.errors, [] )

		calls = [ i for i in fn.instructions if isinstance( i, ir.Call ) and i.target.stem == 'make' ]
		self.assertEqual( len( calls ), 1 )
		# dispatched to the move[bytearray] overload, not the i32 default one
		winner = calls[0].target
		self.assertEqual( len( winner.parameters ), 1 )
		self.assertTrue( winner.parameters[0].is_move )
		self.assertEqual( winner.parameters[0].type.qualname, 'builtins.bytearray' )

		# b's own ownership actually transferred (_apply_move_hook ran) -
		# no spurious Decref of b left over at its own scope exit on top
		# of whatever the callee itself does with it (the exact double-
		# free shape this same investigation already found and fixed once
		# for the plain, non-overloaded call path)
		decrefs_on_b = [
			i for i in fn.instructions
			if isinstance( i, ir.Decref ) and getattr( i.value, 'stem', None ) == 'b'
		]
		self.assertEqual( decrefs_on_b, [] )

	def test_move_call_against_a_non_move_overload_candidate_is_a_compile_error( self ) -> None:
		# the mirror-image validation _check_move_argument already does for
		# a plain (non-overloaded) target - move(...) wrapping an argument
		# whose resolved candidate ISN'T move[T] must still be rejected,
		# not silently accepted
		code = '\n'.join([
			'@overload',
			'def make( n: i32 ) -> usize:',
			'	...',
			'',
			'def make( n: i32 = 0 ) -> usize:',
			'	return usize( n )',
			'',
			'def make( src: move[bytearray] ) -> usize:',
			'	return len( src )',
			'',
			'def main() -> usize:',
			'	return make( move( 3 ))',
		])
		self._import( code )
		self.compiler._lower( self.discovery.main )
		self.assertTrue( any( 'is not move[T]' in e for e in self.discovery.errors.errors ) )


class IfIsNotNoneNarrowingTests( unittest.TestCase ):
	''' `if x is not None:`/`if x is None: ... else:` against a union-
	typed, bare-Name x now narrows x for whichever branch is actually
	"live" - type_resolver.py's visit_If, previously with no body-
	narrowing setup at all (unlike visit_While/visit_Match). Verified the
	same way test_compiler_sizeof_of_narrowed_name_uses_narrowed_type
	above verifies match-arm narrowing: compiler.sizeof(x) folds to a
	compile-time constant that only matches the NARROWED leaf's own size
	if x's tracked type was actually narrowed down from the whole union's
	own (larger) size. Post-if survival itself needs no new machinery -
	merge_if/_merge_narrowed_soft (cfg.py) are already fully generic. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = False )
		self.compiler = Compiler( self.discovery )

	def _import( self, code: str ):
		return self.compiler.import_code( code, filename = Path( '__test__.py' ))

	def _sizeof_x_is_narrowed_to_u8( self, code: str ) -> bool:
		# True only when compiler.sizeof(u) folded to the compile-time
		# constant 1 (u8's own size) - proof u was narrowed down from U's
		# own (larger) union size. An un-narrowed compiler.sizeof(u) isn't
		# necessarily a Const at all (a TaggedUnion's own size isn't always
		# foldable the same way a scalar leaf's is) - either shape here
		# just means "not narrowed", which is all the negative tests need.
		# `is`/`is not None` against a TaggedUnion is rewritten into a plain
		# tag Eq/NotEq ast.Compare before lowering ever sees it (type_
		# resolver.py's _ReferenceResolver) - now an ordinary u8.__eq__/
		# __ne__ dunder call like any other scalar comparison, so needs
		# builtins imported
		self.discovery.import_name( 'builtins' )
		self._import( code )
		fn = self.compiler._lower( self.discovery.main )
		self.assertEqual( self.discovery.errors.errors, [] )
		assigns = { getattr( i.dest, 'stem', None ): i.src for i in fn.instructions if isinstance( i, ir.Assign ) }
		src = assigns['x']
		return isinstance( src, ir.Const ) and src.value == 1

	def test_narrows_inside_if_is_not_none_body( self ) -> None:
		is_narrowed = self._sizeof_x_is_narrowed_to_u8( '\n'.join([
			'@union',
			'class U:',
			'	A: u8',
			'	Nothing: None',
			'',
			'def main() -> None:',
			'	u: U = U.A( 1 )',
			'	if u is not None:',
			'		x: usize = compiler.sizeof( u )',
			'	return',
		]))
		self.assertTrue( is_narrowed ) # u8's own size, not U's (tag + payload)

	def test_narrows_inside_if_is_none_else_body( self ) -> None:
		is_narrowed = self._sizeof_x_is_narrowed_to_u8( '\n'.join([
			'@union',
			'class U:',
			'	A: u8',
			'	Nothing: None',
			'',
			'def main() -> None:',
			'	u: U = U.A( 1 )',
			'	if u is None:',
			'		pass',
			'	else:',
			'		x: usize = compiler.sizeof( u )',
			'	return',
		]))
		self.assertTrue( is_narrowed )

	def test_narrowing_does_not_leak_into_the_non_narrowed_branch( self ) -> None:
		# the ELSE of `if x is not None:` (x could still be None there) must
		# NOT be narrowed - compiler.sizeof(u) there uses U's own full size
		is_narrowed = self._sizeof_x_is_narrowed_to_u8( '\n'.join([
			'@union',
			'class U:',
			'	A: u8',
			'	Nothing: None',
			'',
			'def main() -> None:',
			'	u: U = U.A( 1 )',
			'	if u is not None:',
			'		pass',
			'	else:',
			'		x: usize = compiler.sizeof( u )',
			'	return',
		]))
		self.assertFalse( is_narrowed )

	def test_narrowing_survives_past_the_whole_if_when_the_other_branch_returns( self ) -> None:
		# `if x is None: return` - the ONLY way past this statement is
		# already having x is not None, so x is narrowed for the REST of
		# the function too, same survival merge_if already gives match/
		# while (steady-dancing-haven.md's own reasoning)
		is_narrowed = self._sizeof_x_is_narrowed_to_u8( '\n'.join([
			'@union',
			'class U:',
			'	A: u8',
			'	Nothing: None',
			'',
			'def main() -> None:',
			'	u: U = U.A( 1 )',
			'	if u is None:',
			'		return',
			'	x: usize = compiler.sizeof( u )',
			'	return',
		]))
		self.assertTrue( is_narrowed )

	def test_no_narrowing_survival_when_neither_branch_terminates( self ) -> None:
		# neither branch of the if unconditionally exits - nothing proves
		# u is non-None by the time execution reaches past the whole
		# statement, so code after it must NOT be narrowed
		is_narrowed = self._sizeof_x_is_narrowed_to_u8( '\n'.join([
			'@union',
			'class U:',
			'	A: u8',
			'	Nothing: None',
			'',
			'def main() -> None:',
			'	u: U = U.A( 1 )',
			'	if u is not None:',
			'		pass',
			'	x: usize = compiler.sizeof( u )',
			'	return',
		]))
		self.assertFalse( is_narrowed )

	def test_three_member_union_is_not_none_does_not_narrow( self ) -> None:
		# more than one non-None member - which of them x actually IS
		# can't be determined from `is not None` alone, matching
		# _rewrite_tagged_union_truthiness's own identical restriction; no
		# multi-member narrowing-marker support exists (compiler.sizeof(u)
		# still compiles - just against U's own full, un-narrowed size)
		is_narrowed = self._sizeof_x_is_narrowed_to_u8( '\n'.join([
			'@union',
			'class U:',
			'	A: u8',
			'	B: i64',
			'	Nothing: None',
			'',
			'def main() -> None:',
			'	u: U = U.A( 1 )',
			'	if u is not None:',
			'		x: usize = compiler.sizeof( u )',
			'	return',
		]))
		self.assertFalse( is_narrowed )


class RejectMoveThroughUnionOrOverloadTests( unittest.TestCase ):
	''' calling an @move-decorated method through a union-typed receiver
	or an overload group is now a compile error, not a silent gap. Both
	were confirmed silent before this fix: the receiver-move-hook
	(lowering.py's _lower_call, gated on isinstance(target, Function))
	never fires for a ReceiverDispatch OR an Overload target at all - so
	calling an @move method through either shape neither tracked
	ownership correctly nor errored. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _import( self, code: str ):
		return self.compiler.import_code( code, filename = Path( '__test__.py' ))

	def test_move_method_through_union_receiver_is_rejected( self ) -> None:
		code = '\n'.join([
			'class A:',
			'	@move',
			'	def consume( self ) -> i32:',
			'		return 1',
			'',
			'class B:',
			'	@move',
			'	def consume( self ) -> i32:',
			'		return 2',
			'',
			'def main() -> None:',
			'	x: A|B = A()',
			'	x.consume()',
			'	return',
		])
		self._import( code )
		self.compiler._lower( self.discovery.main )
		self.assertTrue( any( '@move' in e and 'union-typed receiver' in e for e in self.discovery.errors.errors ) )

	def test_move_overload_candidate_is_rejected( self ) -> None:
		code = '\n'.join([
			'class Box:',
			'	value: i32',
			'',
			'	def __init__( self, v: i32 ) -> None:',
			'		self.value = v',
			'',
			'	@overload',
			'	@move',
			'	def unwrap( self, default: i32 ) -> i32:',
			'		...',
			'',
			'	@move',
			'	def unwrap( self, default: i32 = 0 ) -> i32:',
			'		return self.value',
			'',
			'def main() -> None:',
			'	b: Box = Box( 1 )',
			'	b.unwrap()',
			'	return',
		])
		self._import( code )
		self.compiler._lower( self.discovery.main )
		self.assertTrue( any( '@move-decorated overload' in e for e in self.discovery.errors.errors ) )


class GenericOverloadDispatchTests( unittest.TestCase ):
	''' an @overload (or plain, no-decorator) group can now mix a concrete
	candidate with a generic `[T]` one sharing the same name - fixed a real
	compiler gap where a TypeVar-typed candidate's own required leaves
	(Type.leaves() returns [itself] for a bare TypeVar) could never
	same_type-match a concrete call-site argument (overload_resolution.py's
	_Candidate.wildcard), and where lowering.py's Overload dispatch branch
	had no monomorphization step for a resolved generic candidate (the
	FIRST fix alone still crashed the emitter with a bare TypeVar parameter
	- see _lower_overload_generic_call/_finish_generic_call). A call whose
	argument is CONCRETE (not itself union-typed) always resolves to a
	single, statically-known branch at compile time either way. A UNION-
	typed argument can force a real runtime ConditionalDispatch instead - a
	generic branch/default there is now ALSO supported, whether whatever
	leaf(s) still reach it are pinned to exactly one at compile time (see
	_monomorphize_dispatch_target) or genuinely span 2+ distinct runtime
	leaves (_expand_dispatch_target splits that one branch into one new,
	individually-concrete, individually-monomorphized branch per leaf, each
	with its own runtime tag check - a real per-tag dispatch table, not
	just a compile-time shortcut). Only return-only type-param inference
	(T appearing solely in the return type, never in any parameter) through
	a runtime-dispatched branch remains unsupported (the last test below). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _import( self, code: str ):
		return self.compiler.import_code( code, filename = Path( '__test__.py' ))

	def test_concrete_call_prefers_concrete_overload_regardless_of_declaration_order( self ) -> None:
		# the generic candidate declared FIRST in source must still lose to
		# the concrete one for a str argument - dispatch priority is a
		# property of each candidate's own shape (wildcard vs concrete),
		# not which one the user happened to write first
		code = '\n'.join([
			'class Box:',
			'	def get[T]( self, x: T ) -> str:',
			'		return "generic"',
			'',
			'	def get( self, x: str ) -> str:',
			'		return x',
			'',
			'def main() -> None:',
			'	b: Box = Box()',
			'	s: str = b.get( "hi" )',
			'	return',
		])
		self._import( code )
		self.compiler._lower( self.discovery.main )
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_runtime_dispatched_union_argument_with_generic_branch_resolving_one_leaf_compiles( self ) -> None:
		# a UNION-typed argument can force overload_resolution.resolve_call
		# to return a real runtime ConditionalDispatch, whose branches
		# (including the trailing default) _lower_conditional_dispatch
		# always schedules as concrete, callable C symbols - a generic
		# branch has no single such symbol UNLESS whatever leaf(s) still
		# reach it are already pinned down to exactly one at compile time
		# (here: the union has exactly 2 leaves, str claimed by the concrete
		# overload, so only i32 can ever reach the generic default) - see
		# _monomorphize_dispatch_target. This used to be rejected outright
		# (same blanket rejection the next test still exercises for the
		# genuinely harder shape) until a real repro showed it doesn't
		# actually need runtime-polymorphic dispatch: T is statically
		# knowable here, same as any other generic call.
		code = '\n'.join([
			'class Box:',
			'	def get( self, x: str ) -> str:',
			'		return x',
			'',
			'	def get[T]( self, x: T ) -> str:',
			'		return "generic"',
			'',
			'def pick( flag: bool ) -> str|i32:',
			'	if flag:',
			'		return "hi"',
			'	return 42',
			'',
			'def main() -> None:',
			'	b: Box = Box()',
			'	u: str|i32 = pick( True )',
			'	b.get( u )',
			'	return',
		])
		self._import( code )
		self.compiler._lower( self.discovery.main )
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_runtime_dispatched_union_argument_with_multi_leaf_generic_branch_compiles( self ) -> None:
		# the genuinely harder shape the previous test's fix alone does NOT
		# cover: a 3-leaf union where only ONE leaf has a concrete overload,
		# so TWO distinct leaves (i32 and bool) both fall through to the
		# SAME generic default - each needs its own distinct
		# monomorphization, selected by a runtime tag. This used to be
		# rejected outright (this compiler has no vtable/runtime-
		# polymorphic dispatch concept to fall back on) until
		# _expand_dispatch_target started splitting the one ambiguous
		# branch into one new, individually-concrete branch per leaf
		# instead - a real per-tag monomorphization dispatch table, not
		# just resolving T statically. See emitter_c_test.py's
		# MultiLeafGenericDispatchRealCompileTests for the real compile+run
		# confirmation this actually calls the RIGHT monomorphization per
		# leaf at runtime, not just that it compiles.
		code = '\n'.join([
			'class Box:',
			'	def get( self, x: str ) -> str:',
			'		return x',
			'',
			'	def get[T]( self, x: T ) -> str:',
			'		return "generic"',
			'',
			'def pick( flag: i32 ) -> str|i32|bool:',
			'	if flag == 0:',
			'		return "hi"',
			'	if flag == 1:',
			'		return 42',
			'	return True',
			'',
			'def main() -> None:',
			'	b: Box = Box()',
			'	u: str|i32|bool = pick( 1 )',
			'	b.get( u )',
			'	return',
		])
		self._import( code )
		self.compiler._lower( self.discovery.main )
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_runtime_dispatched_union_argument_with_return_only_type_param_is_rejected( self ) -> None:
		# the one remaining unsupported shape: a generic branch/default's
		# own type param appears ONLY in its return type, never in any
		# parameter - _monomorphize_dispatch_target's own "missing" check
		# fails cleanly here rather than wiring through
		# _infer_return_only_type_params (a materially bigger feature to
		# thread through a runtime-dispatched branch, since it needs to
		# actually lower the body to infer the return type; every real
		# lib/ overload group binds T directly off a parameter instead).
		code = '\n'.join([
			'class Box:',
			'	def get( self, x: str ) -> str:',
			'		return x',
			'',
			'	def get[T, K]( self, x: T ) -> K:',
			'		return compiler.uninitialized()',
			'',
			'def pick( flag: bool ) -> str|i32:',
			'	if flag:',
			'		return "hi"',
			'	return 42',
			'',
			'def main() -> None:',
			'	b: Box = Box()',
			'	u: str|i32 = pick( True )',
			'	n: i32 = b.get( u )',
			'	return',
		])
		self._import( code )
		self.compiler._lower( self.discovery.main )
		self.assertTrue( any(
			'generic overload' in e and 'runtime-dispatched call' in e
			for e in self.discovery.errors.errors
		))


class InFunctionRelativeImportTests( unittest.TestCase ):
	'''
	function bodies are never walked by discovery.py's visitor, so an import
	written inside a function body reaches only Lowering._stmt_ImportFrom.
	That site used to slice the importing module's qualname directly, without
	discovery.py's own compensation for a module whose qualname is already its
	package - so a relative import inside a function body in a package's
	__init__.py climbed one level too far. Both sites now count from
	Module.package instead.
	'''

	def _build( self, root: Path, files: dict[str,str] ) -> Compiler:
		( root / 'pkg' ).mkdir()
		for name, text in files.items():
			( root / 'pkg' / name ).write_text( text )
		# the fixture package plus the real lib/ - these tests lower actual
		# code, so builtins has to be importable
		disco = Discovery(
			paths = [ root, Path( discovery.__file__ ).parent / 'lib' ],
			import_builtins = True,
		)
		return Compiler( disco )

	def test_relative_import_inside_a_function_body_in_package_init( self ) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			root = Path( tmp )
			compiler = self._build( root, {
				'__init__.py': '\n'.join([
					'def get() -> i32:',
					'	from .impl import VALUE',
					'	return VALUE',
					'',
				]),
				'impl.py': 'VALUE: i32 = 7\n',
			})
			compiler.import_code( '\n'.join([
				'import pkg',
				'',
				'def main() -> i32:',
				'	return pkg.get()',
				'',
			]), Path( '__main__.py' ), scope = None )
			compiler.run()

			self.assertEqual( compiler.disco.errors.errors, [] )

	def test_relative_import_inside_a_function_body_in_private_module( self ) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			root = Path( tmp )
			compiler = self._build( root, {
				'__init__.py': 'from .__impl import get\n',
				'__impl.py': '\n'.join([
					'def get() -> i32:',
					'	from .impl import VALUE',
					'	return VALUE',
					'',
				]),
				'impl.py': 'VALUE: i32 = 7\n',
			})
			compiler.import_code( '\n'.join([
				'import pkg',
				'',
				'def main() -> i32:',
				'	return pkg.get()',
				'',
			]), Path( '__main__.py' ), scope = None )
			compiler.run()

			self.assertEqual( compiler.disco.errors.errors, [] )


class DefiniteAssignmentTests( unittest.TestCase ):
	''' cfg.py's new type-independent liveness tracking (CFGState._live) -
	a bare declaration (`x: T`, no initializer) only assigned on SOME
	code paths must raise "not initialized on all code branches" at the
	actual read/del, not the misleading "is not defined" a branch-confined
	RC local used to get (merge_if() used to delete it from fn.names
	entirely - see cfg.py's own merge_if docstring) nor the silent
	uninitialized-read UB a non-RC local used to get (invisible to
	cfg.py's RC-only bindings entirely, before this feature). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True ) # str is a builtin, not an intrinsic - needed by several fixtures below
		self.compiler = Compiler( self.discovery )

	def _import( self, code: str ):
		return self.compiler.import_code( code, filename = Path( '__test__.py' ))

	def _errors( self, code: str ) -> list[str]:
		self._import( code )
		self.compiler._lower( self.discovery.main )
		return self.discovery.errors.errors

	def test_bare_declare_conditionally_assigned_rc_local_used_after_if_is_flagged( self ) -> None:
		# the exact test_hello.py repro this feature was built to fix
		errors = self._errors( '\n'.join([
			'def main( hello: str, world: str ) -> None:',
			'	hello_world: str',
			'	if hello == hello:',
			'		hello_world = hello',
			'	hello_world.lower()',
			'	return',
		]))
		self.assertTrue( any( "'hello_world' is not initialized on all code branches" in e for e in errors ) )
		self.assertFalse( any( 'is not defined' in e for e in errors ))

	def test_bare_declare_conditionally_assigned_scalar_local_used_after_if_is_flagged( self ) -> None:
		# same shape as above but a scalar (no RC leaves) - previously
		# completely invisible to cfg.py (silent uninitialized read)
		errors = self._errors( '\n'.join([
			'def main( cond: bool ) -> None:',
			'	x: i32',
			'	if cond:',
			'		x = 1',
			'	y: i32 = x',
			'	return',
		]))
		self.assertTrue( any( "'x' is not initialized on all code branches" in e for e in errors ) )

	def test_fresh_rc_local_confined_to_one_branch_with_no_prior_declaration_used_after_if_is_flagged( self ) -> None:
		errors = self._errors( '\n'.join([
			'class Foo: pass',
			'',
			'def main( cond: bool ) -> None:',
			'	if cond:',
			'		z: Foo = Foo()',
			'	y: Foo = z',
			'	return',
		]))
		self.assertTrue( any( "'z' is not initialized on all code branches" in e for e in errors ) )

	def test_fresh_scalar_local_confined_to_one_branch_with_no_prior_declaration_used_after_if_is_flagged( self ) -> None:
		errors = self._errors( '\n'.join([
			'def main( cond: bool ) -> None:',
			'	if cond:',
			'		x: i32 = 1',
			'	y: i32 = x',
			'	return',
		]))
		self.assertTrue( any( "'x' is not initialized on all code branches" in e for e in errors ) )

	def test_del_on_bare_declared_never_assigned_local_is_flagged( self ) -> None:
		errors = self._errors( '\n'.join([
			'def main() -> None:',
			'	x: i32',
			'	del x',
			'	return',
		]))
		self.assertTrue( any( "'x' is not initialized on all code branches" in e for e in errors ) )

	def test_corrected_version_with_else_branch_compiles_clean( self ) -> None:
		# both branches cover hello_world - the fix this feature enables:
		# a real definite-assignment error on the broken version, no error
		# once every path actually assigns it
		errors = self._errors( '\n'.join([
			'def main( hello: str, world: str ) -> None:',
			'	hello_world: str',
			'	if hello == hello:',
			'		hello_world = hello',
			'	else:',
			'		hello_world = world',
			'	hello_world.lower()',
			'	return',
		]))
		self.assertEqual( errors, [] )

	def test_branch_confined_rc_local_never_used_again_decrefs_without_flag_or_error( self ) -> None:
		# regression guard, through the real lowering.py path (not just
		# cfg_test.py's own bare-CFGState unit test): a genuinely-fresh
		# RC local confined to one if-branch, with no prior declaration
		# and never read again, still tears down (Decref) with no error -
		# and does NOT synthesize a _mint_cancel_flag()-guarded runtime
		# bool for this simple case (see the plan's own "known risk" note:
		# the Decref is placed via plain branch-splicing, not a flag)
		code = '\n'.join([
			'class Foo: pass',
			'',
			'def main( cond: bool ) -> None:',
			'	if cond:',
			'		z: Foo = Foo()',
			'	return',
		])
		self._import( code )
		fn = self.compiler._lower( self.discovery.main )
		self.assertEqual( self.discovery.errors.errors, [] )
		kinds = [ type( instr ).__name__ for instr in fn.instructions ]
		self.assertIn( 'Decref', kinds )
		bool_cls = self.discovery.get_intrinsics()['bool']
		flag_disarms = [
			i for i in fn.instructions
			if isinstance( i, ir.Assign ) and isinstance( i.src, ir.Const ) and i.src.type is bool_cls and i.src.value is False
		]
		self.assertEqual( flag_disarms, [] )

	def test_bare_declare_conditionally_assigned_in_loop_body_used_after_loop_is_flagged( self ) -> None:
		errors = self._errors( '\n'.join([
			'def main( cond: bool ) -> None:',
			'	x: i32',
			'	while cond:',
			'		if cond:',
			'			x = 1',
			'		cond = False',
			'	y: i32 = x',
			'	return',
		]))
		self.assertTrue( any( "'x' is not initialized on all code branches" in e for e in errors ) )

	def test_unconditionally_assigned_before_use_inside_loop_body_is_fine( self ) -> None:
		errors = self._errors( '\n'.join([
			'def main( cond: bool ) -> None:',
			'	x: i32',
			'	while cond:',
			'		x = 1',
			'		y: i32 = x',
			'		cond = False',
			'	return',
		]))
		self.assertEqual( errors, [] )

	def test_assigned_before_loop_survives_break_and_natural_exit( self ) -> None:
		errors = self._errors( '\n'.join([
			'def main( cond: bool ) -> None:',
			'	x: i32 = 0',
			'	while cond:',
			'		if cond:',
			'			x = 1',
			'			break',
			'		cond = False',
			'	y: i32 = x',
			'	return',
		]))
		self.assertEqual( errors, [] )

	def test_assigned_only_on_break_path_not_natural_exit_used_after_loop_is_flagged( self ) -> None:
		errors = self._errors( '\n'.join([
			'def main( cond: bool ) -> None:',
			'	x: i32',
			'	while cond:',
			'		if cond:',
			'			x = 1',
			'			break',
			'		cond = False',
			'	y: i32 = x',
			'	return',
		]))
		self.assertTrue( any( "'x' is not initialized on all code branches" in e for e in errors ) )

	def test_nested_if_else_both_terminating_does_not_wipe_liveness_of_unrelated_param( self ) -> None:
		# real bug: _stmt_diverges only recognized a LITERAL Return/Break/
		# Continue as its own last statement, never a nested if/else whose
		# OWN two branches both terminate - so an outer if-branch ending in
		# such a nested if/else was wrongly treated as falling through to
		# the join point. merge_if's ordinary (non-terminating) path then
		# intersected that branch's genuinely-empty post-terminator live
		# set against the other (implicit, no-else) branch's full live set,
		# wiping out even ordinary, always-bound parameters read afterward.
		errors = self._errors( '\n'.join([
			'def main( cond: bool, other: bool ) -> None:',
			'	if cond:',
			'		if other:',
			'			return',
			'		else:',
			'			return',
			'	if other:',
			'		return',
			'	return',
		]))
		self.assertEqual( errors, [] )

	def test_match_with_all_arms_returning_inside_if_does_not_wipe_liveness_of_unrelated_param( self ) -> None:
		# the shape this was actually found in (lib/http/client.py's
		# _encode_body): an exhaustive `match` (every arm returns) as the
		# last statement of an `if`-branch, desugared by type_resolver.py's
		# visit_Match into a chained ast.If (not a literal Return) - same
		# root cause as the plain nested-if case above, via the identical
		# _stmt_diverges blind spot.
		errors = self._errors( '\n'.join([
			'def parse( s: str ) -> Result[str,str]:',
			'	return Result.Ok( s )',
			'',
			'def main( data: str|None, form: str|None, json_value: str|None ) -> str:',
			'	if json_value is not None:',
			'		jv: str = json_value',
			'		match parse( jv ):',
			'			case Result.Ok( text ):',
			'				return text',
			'			case Result.Err( _ ):',
			'				return "err"',
			'	if form is not None:',
			'		return form',
			'	if data is not None:',
			'		return data',
			'	return "none"',
		]))
		self.assertEqual( errors, [] )


if __name__ == '__main__':
	logging.basicConfig( level = logging.DEBUG )
	unittest.main()
