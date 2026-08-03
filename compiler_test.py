# stdlib imports:
from pathlib import Path
import unittest

# local imports:
import ir
from discovery import Discovery
from compiler import Compiler
from mpy_types import Module, Variable, RCClass, Specialization

class CompilerTestCase( unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = False )
		self.compiler = Compiler( self.discovery )

	def _run( self, code: str ) -> None:
		self.compiler.import_code( code, Path( '__main__.py' ), scope = None )
		self.compiler.run()

	def _function_names( self ) -> list[str]:
		return [ f.function.qualname for f in self.compiler.functions ]

	def _instructions_for( self, qualname: str ) -> list[ir.Instruction]:
		for f in self.compiler.functions:
			if f.function.qualname == qualname:
				return f.instructions
		self.fail( f'{qualname!r} was never lowered' )

class ArchitectureExampleTests( CompilerTestCase ):
	''' hand-verifies ARCHITECTURE.md's own shape: FuncStart, DeclareTemp, AddWrap, Call, DeleteTemp, Return, FuncEnd '''

	def test_foo_sequence_and_schedule_order( self ) -> None:
		self._run( '''
def main() -> None:
	foo( 3 )

def foo( x: i32 ) -> None:
	with compiler.wrap_arithmetic:
		echo( x + 1 )
	return

def echo( x: i32 ) -> None:
	pass
''' )
		self.assertEqual( self._function_names(), [ 'main', '__main__.foo', '__main__.echo' ] )

		instructions = self._instructions_for( '__main__.foo' )
		kinds = [ type( instr ) for instr in instructions ]
		self.assertEqual( kinds, [ ir.FuncStart, ir.DeclareTemp, ir.AddWrap, ir.Call, ir.DeleteTemp, ir.Return, ir.FuncEnd ] )

		add_wrap = instructions[2]
		self.assertIsInstance( add_wrap, ir.AddWrap )
		self.assertEqual( add_wrap.right, ir.Const( type = add_wrap.left.type, value = 1 ))

		call = instructions[3]
		self.assertIsInstance( call, ir.Call )
		self.assertEqual( call.target.qualname, '__main__.echo' )
		self.assertIsNone( call.dest )
		self.assertIsNone( call.receiver )
		self.assertEqual( call.args, [ add_wrap.dest ] )

class DeadCodeTests( CompilerTestCase ):
	def test_uncalled_function_is_excluded( self ) -> None:
		self._run( '''
def main() -> None:
	pass

def dead() -> None:
	pass
''' )
		self.assertEqual( self._function_names(), [ 'main' ] )

class MutualRecursionTests( CompilerTestCase ):
	def test_terminates_and_schedules_each_once( self ) -> None:
		self._run( '''
def main() -> None:
	a()

def a() -> None:
	b()

def b() -> None:
	a()
''' )
		names = self._function_names()
		self.assertEqual( sorted( names ), sorted([ 'main', '__main__.a', '__main__.b' ]))
		self.assertEqual( len( names ), len( set( names )))

class ClassDependencyTests( CompilerTestCase ):
	def test_class_referenced_only_via_local_annotation_is_scheduled( self ) -> None:
		self._run( '''
class Base:
	pass

class Foo( Base ):
	pass

def main() -> None:
	f: Foo
''' )
		rcclass_names = [ cls.qualname for cls in self.compiler.rcclasses ]
		self.assertIn( '__main__.Foo', rcclass_names )
		self.assertIn( '__main__.Base', rcclass_names )

	def test_class_referenced_via_parameter_type_is_scheduled( self ) -> None:
		self._run( '''
class Foo:
	x: i32
	def bump( self ) -> None:
		self.x = self.x

def use( f: Foo ) -> None:
	f.bump()

def main() -> None:
	f: Foo
	use( f )
''' )
		self.assertIn( '__main__.use', self._function_names() )
		self.assertIn( '__main__.Foo.bump', self._function_names() )
		self.assertIn( '__main__.Foo', [ cls.qualname for cls in self.compiler.rcclasses ] )

		bump_instructions = self._instructions_for( '__main__.Foo.bump' )
		kinds = [ type( instr ) for instr in bump_instructions ]
		# self.x = self.x -> GetAttr( t0, self, 'x' ), SetAttr( self, 'x', t0 ), DeleteTemp( t0 )
		self.assertIn( ir.GetAttr, kinds )
		self.assertIn( ir.SetAttr, kinds )

		use_instructions = self._instructions_for( '__main__.use' )
		call = next( i for i in use_instructions if isinstance( i, ir.Call ) )
		self.assertEqual( call.target.qualname, '__main__.Foo.bump' )
		self.assertIsNotNone( call.receiver )

	def test_class_reached_only_as_an_intermediate_attribute_chain_link_is_scheduled( self ) -> None:
		# Inner never appears as its own parameter/return/local-variable
		# annotation anywhere - the only way to learn it exists at all is by
		# resolving Outer.inner's field type while chasing o.inner.value.
		# _attr_lookup used to resolve() the owner type (so the lookup itself
		# succeeds) without ever scheduling it, so Inner would silently never
		# make it into compiler.rcclasses even though get_value depends on it
		self._run( '''
class Inner:
	value: i32

class Outer:
	inner: Inner

def get_value( o: Outer ) -> i32:
	return o.inner.value

def main() -> None:
	o: Outer
	x: i32 = get_value( o )
''' )
		rcclass_names = [ cls.qualname for cls in self.compiler.rcclasses ]
		self.assertIn( '__main__.Outer', rcclass_names )
		self.assertIn( '__main__.Inner', rcclass_names )

	def test_class_reached_only_via_a_staticmethod_call_is_scheduled( self ) -> None:
		# same gap, different path: Foo is never used as a parameter/return/
		# local-variable type anywhere - the only way to reach it is resolving
		# it while walking the Foo.make() namespace path in _try_resolve_namespace
		self._run( '''
class Foo:
	@staticmethod
	def make() -> i32:
		return 1

def main() -> None:
	x: i32 = Foo.make()
''' )
		self.assertIn( '__main__.Foo', [ cls.qualname for cls in self.compiler.rcclasses ] )

class FunctionSchedulingViaEnsureResolvedTests( CompilerTestCase ):
	''' _lower_call and _emit_is_err_check both schedule their call target
	through _ensure_resolved rather than a separate explicit self.schedule(...)
	call - these confirm the call target still actually ends up lowered by a
	full Compiler.run(), not just resolved for the immediate lookup '''

	def test_plain_call_target_is_scheduled( self ) -> None:
		self._run( '''
def main() -> None:
	helper()

def helper() -> None:
	pass
''' )
		self.assertEqual( self._function_names(), [ 'main', '__main__.helper' ] )

	def test_errdefer_is_err_call_target_is_scheduled( self ) -> None:
		# _emit_is_err_check resolves+schedules Result.is_err purely as a side
		# effect of building the epilogue - is_err is never called from user
		# source at all, so it's the one call site here self.schedule() can't
		# be reached via the ordinary _lower_call path
		self._run( '''
class bool: pass
class OverflowError: pass

@cstruct
class Result[T,E]:
	def is_err( self ) -> bool:
		pass
	@staticmethod
	def Err( e: E ) -> Result[T,E]:
		pass

def checked( e: OverflowError ) -> Result[None,OverflowError]:
	with errdefer:
		pass
	return Result.Err( e )

def main() -> None:
	e: OverflowError
	checked( e )
''' )
		names = self._function_names()
		self.assertIn( '__main__.checked', names )
		self.assertTrue( any( name.endswith( 'Result.is_err' ) for name in names ), names )

class EnqueueFilteringTests( CompilerTestCase ):
	''' lowering.py's _ensure_resolved hands _enqueue literally anything it
	comes across, unconditionally - these are the direct, white-box checks
	that _enqueue itself is the one drawing the line: real compile units get
	queued, a Specialization decomposes into its schedulable pieces, and
	everything else is silently dropped rather than corrupting compiler.
	globals or crashing on drain '''

	def test_module_is_silently_ignored( self ) -> None:
		fake_module = Module( stem = 'fake', qualname = 'fake', file = None, line = None, intrinsics = {}, builtins = None )
		self.compiler._enqueue( fake_module )
		self.assertTrue( self.compiler.queue.empty())

	def test_none_is_silently_ignored( self ) -> None:
		self.compiler._enqueue( None )
		self.assertTrue( self.compiler.queue.empty())

	def test_non_global_variable_is_silently_ignored( self ) -> None:
		field = Variable( stem = 'x', qualname = 'Foo.x', file = None, line = None, is_global = False )
		self.compiler._enqueue( field )
		self.assertTrue( self.compiler.queue.empty())

	def test_global_variable_is_enqueued_and_lowered( self ) -> None:
		self._run( '''
X: i32 = 1

def main() -> None:
	pass
''' )
		x = self.discovery.modules['__main__'].get_local( 'X' )
		self.assertTrue( x.is_global )
		self.compiler._enqueue( x )
		self.compiler.run()
		self.assertIn( 'X', [ g.variable.stem for g in self.compiler.globals ] )

	def test_global_read_only_by_bare_name_is_still_lowered( self ) -> None:
		# _expr_Name used to just look the Variable up and hand it back
		# without ever calling _ensure_resolved on it - a global reached only
		# by plain reference (never via an annotation/attribute chain
		# elsewhere) would silently never make it into compiler.globals
		self._run( '''
X: i32 = 1

def main() -> None:
	y: i32 = X
''' )
		self.assertIn( 'X', [ g.variable.stem for g in self.compiler.globals ] )

	def test_specialization_decomposes_into_base_and_each_arg( self ) -> None:
		base = RCClass( stem = 'Result', qualname = '__main__.Result', file = None, line = None )
		arg1 = RCClass( stem = 'Ok', qualname = '__main__.Ok', file = None, line = None )
		arg2 = RCClass( stem = 'Err', qualname = '__main__.Err', file = None, line = None )
		spec = Specialization( stem = 'Result[Ok,Err]', qualname = '__main__.Result[Ok,Err]', file = None, line = None, base = base, args = [ arg1, arg2 ] )
		self.compiler._enqueue( spec )
		queued = []
		while not self.compiler.queue.empty():
			queued.append( self.compiler.queue.get_nowait())
		self.assertEqual( len( queued ), 3 )
		self.assertTrue( any( q is base for q in queued ))
		self.assertTrue( any( q is arg1 for q in queued ))
		self.assertTrue( any( q is arg2 for q in queued ))

class OverloadCallSiteTests( CompilerTestCase ):
	def test_len_style_plain_overload_group_dispatches_by_argument_type( self ) -> None:
		# mirrors the real lib/builtins/__init__.py len() shape: a free
		# function with no @overload anywhere, one plain implementation per
		# concrete type, dispatched purely by the call's own argument type
		self._run( '''
class usize: pass

class Foo:
	def __len__( self ) -> usize:
		return 1

class Bar:
	def __len__( self ) -> usize:
		return 2

def len( x: Foo ) -> usize:
	return x.__len__()

def len( x: Bar ) -> usize:
	return x.__len__()

def main() -> None:
	f: Foo
	b: Bar
	x: usize = len( f )
	y: usize = len( b )
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		# each len(...) call site picks a *distinct* implementation (both
		# share the same qualname - they're only distinguishable by object
		# identity/their own parameter type) - confirm both actually got
		# scheduled+lowered, and that each one's own body calls the right
		# receiver's __len__, not the other one's
		len_fns = [ f for f in self.compiler.functions if f.function.stem == 'len' ]
		self.assertEqual( len( len_fns ), 2 )
		receiver_classes = sorted( lf.function.parameters[0].type.stem for lf in len_fns )
		self.assertEqual( receiver_classes, [ 'Bar', 'Foo' ])
		for lf in len_fns:
			expected_receiver_cls = lf.function.parameters[0].type.stem
			call = next( i for i in lf.instructions if isinstance( i, ir.Call ))
			self.assertEqual( call.target.qualname, f'__main__.{expected_receiver_cls}.__len__' )

	def test_unconditional_target_schedules_only_that_target( self ) -> None:
		self._run( '''
class int: pass
class str: pass

@overload
def foo( x: int ) -> None:
	...

def foo( x: int|None = None ) -> None:
	pass

def foo( x: str ) -> None:
	pass

def main() -> None:
	x: int
	foo( x )
''' )
		names = self._function_names()
		self.assertIn( 'main', names )
		# exactly one of the two plain implementations was scheduled, never
		# the whole group and never the stub (stubs have no body to lower)
		self.assertEqual( len( names ), 2 )

	def test_multi_branch_dispatch_resolves_via_runtime_tag_check( self ) -> None:
		# a union-typed argument (x: int|str) makes foo(x) ambiguous at
		# compile time - resolve_call returns real ConditionalDispatch
		# branches, which now lower to a runtime tag check (ir.Cmp against
		# the synthesized anonymous union's tag) rather than failing
		self.compiler.import_code( '''
class int: pass
class str: pass

@overload
def foo( x: int ) -> None:
	...

@overload
def foo( x: str ) -> None:
	...

def foo( x: int ) -> None:
	pass

def foo( x: str ) -> None:
	pass

def main() -> None:
	x: int|str
	foo( x )
''', Path( '__main__.py' ), scope = None )
		self.compiler.run()
		self.assertEqual( self.discovery.errors.errors, [] )
		instructions = self._instructions_for( 'main' )
		kinds = [ type( i ).__name__ for i in instructions ]
		self.assertIn( 'Cmp', kinds )
		self.assertEqual( kinds.count( 'Call' ), 2 ) # one per possible target (the branch + the default) - only one runs at runtime

	def test_generic_function_call_schedules_only_the_specialization( self ) -> None:
		# alloc[u32](...) must compile exactly one function - the
		# monomorphized alloc[u32] - never the shared, unspecialized alloc
		# itself (T never gets bound there, so it can't actually compile)
		self._run( '''
def alloc[T]( count: usize ) -> usize:
	with compiler.wrap_arithmetic:
		return count * compiler.sizeof( T )

def main() -> None:
	x: usize = alloc[u32]( 10 )
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		names = self._function_names()
		self.assertEqual( set( names ), { 'main', '__main__.alloc[intrinsics.u32]' } )

class LocalVariableTests( CompilerTestCase ):
	def test_annassign_then_reassign( self ) -> None:
		self._run( '''
def main() -> None:
	x: i32 = 1
	x = 2
''' )
		instructions = self._instructions_for( 'main' )
		assigns = [ i for i in instructions if isinstance( i, ir.Assign ) ]
		self.assertEqual( len( assigns ), 2 )
		self.assertEqual( assigns[0].dest.stem, 'x' )
		self.assertIs( assigns[0].dest, assigns[1].dest )
		self.assertEqual( assigns[0].src, ir.Const( type = assigns[0].dest.type, value = 1 ))
		self.assertEqual( assigns[1].src, ir.Const( type = assigns[0].dest.type, value = 2 ))

class ErrorRecoveryTests( CompilerTestCase ):
	def test_broken_statement_does_not_block_sibling_statements_or_functions( self ) -> None:
		# 'broken's one statement fails (a bare name expression isn't
		# supported) - lower_function's own per-statement recovery boundary
		# means 'broken' still gets lowered (just missing that statement -
		# FuncStart/[fall-off Return]/FuncEnd, since its -> None body has
		# nothing left that ends in an explicit `return`) rather than being
		# dropped entirely, and 'fine' - unrelated - isn't affected at all
		self._run( '''
def broken() -> None:
	undefined_name

def fine() -> None:
	pass

def main() -> None:
	broken()
	fine()
''' )
		names = self._function_names()
		self.assertIn( 'main', names )
		self.assertIn( '__main__.fine', names )
		self.assertIn( '__main__.broken', names )
		self.assertEqual( [ type( i ).__name__ for i in self._instructions_for( '__main__.broken' ) ], [ 'FuncStart', 'Return', 'FuncEnd' ] )
		self.assertTrue( self.discovery.errors.errors )

if __name__ == '__main__':
	unittest.main()
