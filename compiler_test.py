# stdlib imports:
from pathlib import Path
import unittest

# local imports:
import ir
from discovery import Discovery
from compiler import Compiler

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

class OverloadCallSiteTests( CompilerTestCase ):
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

	def test_multi_branch_dispatch_is_not_yet_supported( self ) -> None:
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
		with self.assertRaises( AssertionError ):
			self.compiler.run()

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

if __name__ == '__main__':
	unittest.main()
