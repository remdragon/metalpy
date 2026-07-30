# stdlib imports
import logging
from pathlib import Path
import tempfile
import unittest

# local imports
import discovery
from mpy_types import (
	Module, RCClass, CStruct, CUnion, CEnum, TaggedUnion, Overload,
	Function, Variable, Specialization, Move,
)

logger = logging.getLogger( __name__ )

class ImportTests( unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = discovery.Discovery( import_builtins = False )

	def test_imports( self ) -> None:
		test_self = self

		class TestComplete( BaseException ):
			pass

		class MockDiscovery( discovery.Discovery ):
			expect_package: str

			@staticmethod
			def new_test( expect_package: str ) -> MockDiscovery:
				disco = MockDiscovery( import_builtins = False )
				disco.expect_package = expect_package
				return disco

			def import_name( self, package: str ) -> discovery.Module:
				test_self.assertEqual( package, self.expect_package )
				raise TestComplete()

			def import_code( self, code: str, filename: Path, scope: str|None = None ) -> Module:
				try:
					super().import_code( code, filename, scope )
				except TestComplete:
					return None

		disco1 = MockDiscovery.new_test( 'codecs' )
		disco1.import_code( 'import codecs', Path( '__irrelevant__.py' ), scope = None )

		disco1 = MockDiscovery.new_test( 'codecs' )
		disco1.import_code( 'import codecs', Path( '__irrelevant__.py' ), scope = 'codecs' )

		disco1 = MockDiscovery.new_test( 'codecs' )
		disco1.import_code( 'import codecs', Path( '__irrelevant__.py' ), scope = 'codecs.utf8' )

		disco1 = MockDiscovery.new_test( 'codecs.utf8' )
		disco1.import_code( 'import codecs.utf8', Path( '__irrelevant__.py' ), scope = None )

		disco1 = MockDiscovery.new_test( 'codecs.utf8' )
		disco1.import_code( 'import codecs.utf8', Path( '__irrelevant__.py' ), scope = 'codecs' )

		disco1 = MockDiscovery.new_test( 'codecs.utf8' )
		disco1.import_code( 'import codecs.utf8', Path( '__irrelevant__.py' ), scope = 'codecs.utf8' )

		disco1 = MockDiscovery.new_test( 'codecs' )
		disco1.import_code( 'from codecs import utf8', Path( '__irrelevant__.py' ), scope = None )

		disco1 = MockDiscovery.new_test( 'codecs' )
		disco1.import_code( 'from codecs import utf8', Path( '__irrelevant__.py' ), scope = 'codecs' )

		disco1 = MockDiscovery.new_test( 'codecs' )
		disco1.import_code( 'from .. import utf8', Path( '__irrelevant__.py' ), scope = 'codecs.utf8' )

		with self.assertRaises( AssertionError ):
			disco1 = MockDiscovery.new_test( 'codecs' )
			disco1.import_code( 'from . import utf8', Path( '__irrelevant__.py' ), scope = None )

		disco1 = MockDiscovery.new_test( 'codecs' )
		disco1.import_code( 'from . import utf8', Path( '__irrelevant__.py' ), scope = 'codecs' )

		disco1 = MockDiscovery.new_test( 'codecs.utf8' )
		disco1.import_code( 'from . import utf8', Path( '__irrelevant__.py' ), scope = 'codecs.utf8' )

	def test_mutable_default_paths_not_shared( self ) -> None:
		# discovery.py:__init__ used to default `paths` to a mutable []
		# literal, so the first instance's lazily-appended default paths
		# leaked into every later instance that also omitted `paths`
		d1 = discovery.Discovery( import_builtins = False )
		d2 = discovery.Discovery( import_builtins = False )
		self.assertEqual( len( d1.paths ), 2 )
		self.assertEqual( len( d2.paths ), 2 )
		d1.paths.append( Path( 'extra' ))
		self.assertEqual( len( d2.paths ), 2 )


class ShallowScanTests( unittest.TestCase ):
	'''
	a module body is scanned immediately: every top-level class/function/
	global is discoverable by name right away. A class's own body (its
	attributes/methods) is a different story - like a function's parameters
	or a variable's type, it stays behind .resolve until something actually
	needs it. exceptions, all parsed eagerly at class-creation time:
		* type_params, since external code subscripting a generic class
		  needs it before that class's own resolve() ever runs
		* a shallow scan for nested inner classes
		* base (single inheritance only - see InheritanceTests), since
		  Python itself requires the base to already exist when the
		  `class Foo(Base):` statement runs, so there's no forward
		  reference to defer
	'''

	def setUp( self ) -> None:
		self.discovery = discovery.Discovery( import_builtins = False )

	def _import( self, code: str ) -> Module:
		return self.discovery.import_code( code, Path( '__main__.py' ), scope = None )

	def test_rcclass( self ) -> None:
		mod = self._import( '''
class Foo:
	pass
''' )
		foo = mod.get_local( 'Foo' )
		self.assertIsInstance( foo, RCClass )
		self.assertEqual( foo.qualname, '__main__.Foo' )
		self.assertEqual( foo.attributes, [] )
		self.assertEqual( foo.methods, [] )

	def test_class_body_deferred_until_resolved( self ) -> None:
		mod = self._import( '''
class Foo:
	x: i32
	def bar( self ) -> i32:
		return self.x
''' )
		foo = mod.get_local( 'Foo' )
		self.assertEqual( foo.attributes, [] ) # body not scanned yet
		self.assertEqual( foo.methods, [] )
		self.assertIsNone( foo.get_local( 'x' ))
		self.assertIsNone( foo.get_local( 'bar' ))
		self.assertIsNotNone( foo.resolve )

		foo.resolve()
		self.assertIsNone( foo.resolve )

		self.assertEqual( len( foo.attributes ), 1 )
		x = foo.attributes[0]
		self.assertEqual( x.stem, 'x' )
		self.assertIsNone( x.type ) # the attribute's own type is still deferred
		self.assertIsNotNone( x.resolve )

		bar = foo.get_local( 'bar' )
		self.assertIsInstance( bar, Function )
		self.assertIsNone( bar.parameters ) # the method's own params are still deferred
		self.assertIsNotNone( bar.resolve )
		self.assertIn( bar, foo.methods )

	def test_nested_class_registered_immediately( self ) -> None:
		# regression: a class's own body was made to scan lazily behind
		# .resolve, which accidentally swept up nested class defs too - but
		# those need to be discoverable by name (Outer.Inner) right away,
		# just like top-level classes, since other code may reference them
		# before Outer.resolve() ever runs
		mod = self._import( '''
class Outer:
	class Inner:
		y: i32
	x: i32
''' )
		outer = mod.get_local( 'Outer' )
		self.assertIsInstance( outer, RCClass )
		self.assertIsNotNone( outer.resolve ) # outer's own body still deferred

		inner = outer.get_local( 'Inner' )
		self.assertIsInstance( inner, RCClass )
		self.assertEqual( inner.qualname, '__main__.Outer.Inner' )
		self.assertIsNotNone( inner.resolve ) # inner's own body deferred too
		self.assertEqual( inner.attributes, [] )

		self.assertIsNone( outer.get_local( 'x' )) # non-class members still deferred

		outer.resolve()
		self.assertIsNone( outer.resolve )
		self.assertEqual( [ a.stem for a in outer.attributes ], [ 'x' ]) # Inner isn't an attribute
		self.assertIs( outer.get_local( 'Inner' ), inner )

		inner.resolve()
		self.assertIsNone( inner.resolve )
		self.assertEqual( [ a.stem for a in inner.attributes ], [ 'y' ])

	def test_cstruct_and_cunion_type_params_available_before_resolve( self ) -> None:
		mod = self._import( '''
@cstruct
class Box[T]:
	value: T

@cunion
class Overlap[T]:
	value: T
''' )
		box = mod.get_local( 'Box' )
		self.assertIsInstance( box, CStruct )
		self.assertIsNotNone( box.resolve ) # body itself is still deferred
		self.assertEqual( len( box.type_params ), 1 )
		self.assertEqual( box.type_params[0].stem, 'T' )
		self.assertIs( box.get_local( 'T' ), box.type_params[0] )

		overlap = mod.get_local( 'Overlap' )
		self.assertIsInstance( overlap, CUnion )
		self.assertEqual( overlap.type_params[0].stem, 'T' )

	def test_cenum_members_deferred_until_resolved( self ) -> None:
		mod = self._import( '''
@enum( i32 )
class Color:
	Red = 0
	Green = 1
''' )
		color = mod.get_local( 'Color' )
		self.assertIsInstance( color, CEnum )
		self.assertEqual( color.members, {} ) # deferred
		self.assertIsNotNone( color.resolve )

		color.resolve()
		self.assertEqual( color.members, { 'Red': 0, 'Green': 1 })
		self.assertIsNone( color.resolve )

	def test_tagged_union_declared( self ) -> None:
		# regression: _parse_ClassDef_TaggedUnion used to not exist at all,
		# so visit_ClassDef's dispatch to it crashed with AttributeError
		mod = self._import( '''
@union
class IntOrSize:
	v_int: i32
	v_size: usize
''' )
		iu = mod.get_local( 'IntOrSize' )
		self.assertIsInstance( iu, TaggedUnion )
		self.assertEqual( iu.attributes, [] ) # deferred
		self.assertIsNotNone( iu.resolve )

		iu.resolve()
		self.assertEqual( [ a.stem for a in iu.attributes ], [ 'v_int', 'v_size' ])
		self.assertIsNone( iu.attributes[0].type ) # the variant's own type is still deferred

	def test_function( self ) -> None:
		mod = self._import( '''
def foo( x: i32 ) -> i32:
	return x
''' )
		foo = mod.get_local( 'foo' )
		self.assertIsInstance( foo, Function )
		self.assertIsNone( foo.parameters )
		self.assertIsNone( foo.return_type )
		self.assertIsNotNone( foo.resolve )

	def test_global_registered_immediately_but_type_deferred( self ) -> None:
		# regression: globals used to be stashed in a separate Module.pending
		# list instead of being registered right away like everything else
		mod = self._import( '''
X: i32 = 1
Y = 2
''' )
		x = mod.get_local( 'X' )
		y = mod.get_local( 'Y' )
		self.assertIsInstance( x, Variable )
		self.assertIsInstance( y, Variable )
		self.assertIsNone( x.type )
		self.assertIsNone( y.type )
		self.assertIsNotNone( x.resolve )
		self.assertIsNotNone( y.resolve )


class DeferredResolutionTests( unittest.TestCase ):
	''' calling .resolve() on a Variable/Function actually populates its type/parameters/return_type and sets .resolve back to None '''

	def setUp( self ) -> None:
		self.discovery = discovery.Discovery( import_builtins = False )

	def _import( self, code: str ) -> Module:
		return self.discovery.import_code( code, Path( '__main__.py' ), scope = None )

	def test_resolve_function_populates_parameters_and_return( self ) -> None:
		mod = self._import( '''
def foo( x: i32, y: usize ) -> i32:
	return x
''' )
		foo = mod.get_local( 'foo' )
		foo.resolve()
		self.assertIsNone( foo.resolve )
		self.assertEqual( [ p.stem for p in foo.parameters ], [ 'x', 'y' ])
		intrinsics = self.discovery.get_intrinsics()
		self.assertIs( foo.parameters[0].type, intrinsics['i32'] )
		self.assertIs( foo.parameters[1].type, intrinsics['usize'] )
		self.assertIs( foo.return_type, intrinsics['i32'] )

	def test_resolve_function_no_return_annotation_is_none_type( self ) -> None:
		mod = self._import( '''
def foo() -> None:
	pass
''' )
		foo = mod.get_local( 'foo' )
		foo.resolve()
		self.assertIs( foo.return_type, self.discovery.get_none_type() )

	def test_resolve_class_reveals_attributes_and_method_skeletons( self ) -> None:
		mod = self._import( '''
class Foo:
	x: i32
	def bar( self ) -> i32:
		return self.x
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		self.assertIsNone( foo.resolve )
		self.assertEqual( [ a.stem for a in foo.attributes ], [ 'x' ])

		bar = foo.get_local( 'bar' )
		self.assertIsInstance( bar, Function )
		self.assertFalse( bar.resolve is None ) # method skeleton only - params stay deferred

		bar.resolve()
		self.assertEqual( bar.parameters, [] ) # self was skipped

	def test_resolve_cenum_body( self ) -> None:
		mod = self._import( '''
@enum( i32 )
class Color:
	Red = 0
	Green = _
	Blue = 5
	Purple = _
''' )
		color = mod.get_local( 'Color' )
		color.resolve()
		self.assertIsNone( color.resolve )
		self.assertEqual( color.members, { 'Red': 0, 'Green': 1, 'Blue': 5, 'Purple': 6 })

	def test_resolve_method_skips_self( self ) -> None:
		mod = self._import( '''
class Foo:
	def bar( self, x: i32 ) -> None:
		pass
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		bar = foo.get_local( 'bar' )
		bar.resolve()
		self.assertEqual( [ p.stem for p in bar.parameters ], [ 'x' ])

	def test_resolve_classmethod_skips_cls( self ) -> None:
		mod = self._import( '''
class Foo:
	@classmethod
	def make( cls, x: i32 ) -> None:
		pass
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		make = foo.get_local( 'make' )
		make.resolve()
		self.assertEqual( [ p.stem for p in make.parameters ], [ 'x' ])

	def test_resolve_attribute_annotation( self ) -> None:
		mod = self._import( '''
class Foo:
	x: i32
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		x = foo.attributes[0]
		x.resolve()
		self.assertIsNone( x.resolve )
		self.assertIs( x.type, self.discovery.get_intrinsics()['i32'] )

	def test_resolve_global_annotated( self ) -> None:
		mod = self._import( '''
X: i32 = 1
''' )
		x = mod.get_local( 'X' )
		x.resolve()
		self.assertIsNone( x.resolve )
		self.assertIs( x.type, self.discovery.get_intrinsics()['i32'] )

	def test_resolve_global_bare_assign_literal_inference( self ) -> None:
		mod = self._import( '''
class int:
	pass

X = 5
''' )
		x = mod.get_local( 'X' )
		x.resolve()
		self.assertIs( x.type, mod.get_local( 'int' ))

	def test_resolve_global_bare_assign_non_literal_errors( self ) -> None:
		mod = self._import( '''
def foo() -> i32:
	return 1

X = foo()
''' )
		x = mod.get_local( 'X' )
		with self.assertRaises( AssertionError ):
			x.resolve()

	def test_resolve_global_bare_assign_references_another_global( self ) -> None:
		# X hasn't been resolved yet when Y.resolve() runs - must resolve
		# transitively rather than copying X.type == None
		mod = self._import( '''
X: i32 = 1
Y = X
''' )
		x = mod.get_local( 'X' )
		y = mod.get_local( 'Y' )
		self.assertIsNotNone( x.resolve ) # not resolved yet
		y.resolve()
		self.assertIs( y.type, self.discovery.get_intrinsics()['i32'] )
		self.assertIsNone( x.resolve ) # resolving Y transitively resolved X too


class GenericFunctionTests( unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = discovery.Discovery( import_builtins = False )

	def _import( self, code: str ) -> Module:
		return self.discovery.import_code( code, Path( '__main__.py' ), scope = None )

	def test_type_param_registered_immediately_and_resolves( self ) -> None:
		# mirrors lib/sys.py's `def alloc[T](count: usize) -> Ptr[T]:`
		mod = self._import( '''
def alloc[T]( count: usize ) -> Ptr[T]:
	pass
''' )
		fn = mod.get_local( 'alloc' )
		self.assertEqual( len( fn.type_params ), 1 )
		self.assertEqual( fn.type_params[0].stem, 'T' )

		fn.resolve()
		self.assertIsInstance( fn.return_type, Specialization )
		self.assertIs( fn.return_type.base, self.discovery.get_intrinsics()['Ptr'] )
		self.assertIs( fn.return_type.args[0], fn.type_params[0] )


class AnonymousUnionTests( unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = discovery.Discovery( import_builtins = False )

	def _import( self, code: str ) -> Module:
		return self.discovery.import_code( code, Path( '__main__.py' ), scope = None )

	def test_dedup_regardless_of_operand_order( self ) -> None:
		mod = self._import( '''
def foo( x: i32 | usize ) -> None:
	pass

def bar( y: usize | i32 ) -> None:
	pass
''' )
		foo = mod.get_local( 'foo' )
		bar = mod.get_local( 'bar' )
		foo.resolve()
		bar.resolve()

		union1 = foo.parameters[0].type
		union2 = bar.parameters[0].type
		self.assertIsInstance( union1, TaggedUnion )
		self.assertIs( union1, union2 )
		self.assertEqual( union1.qualname, 'intrinsics.i32|intrinsics.usize' ) # sorted asciibetically
		self.assertEqual( { a.stem for a in union1.attributes }, { 'i32', 'usize' })

	def test_three_way_union( self ) -> None:
		mod = self._import( '''
def foo( x: i32 | usize | i8 ) -> None:
	pass
''' )
		foo = mod.get_local( 'foo' )
		foo.resolve()
		union = foo.parameters[0].type
		self.assertEqual( { a.stem for a in union.attributes }, { 'i32', 'usize', 'i8' })

	def test_optional_style_union_with_none( self ) -> None:
		mod = self._import( '''
def foo( x: i32 | None ) -> None:
	pass
''' )
		foo = mod.get_local( 'foo' )
		foo.resolve()
		union = foo.parameters[0].type
		self.assertEqual( { a.stem for a in union.attributes }, { 'i32', 'NoneType' })


class GenericsTests( unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = discovery.Discovery( import_builtins = False )

	def _import( self, code: str ) -> Module:
		return self.discovery.import_code( code, Path( '__main__.py' ), scope = None )

	def test_specialization_via_top_level_annotation( self ) -> None:
		# mirrors the real lib.builtins.__init__ pattern:
		# `BYTEARRAY_INVALID: ConstPtr[u8] = b''`
		mod = self._import( '''
@cstruct
class Box[T]:
	value: T

X: Box[i32] = None
''' )
		x = mod.get_local( 'X' )
		x.resolve()
		spec = x.type
		self.assertIsInstance( spec, Specialization )
		box = mod.get_local( 'Box' )
		self.assertIs( spec.base, box )
		self.assertEqual( spec.args, [ self.discovery.get_intrinsics()['i32'] ])

	def test_generic_intrinsic_pointer( self ) -> None:
		mod = self._import( '''
X: ConstPtr[u8] = None
''' )
		x = mod.get_local( 'X' )
		x.resolve()
		spec = x.type
		self.assertIsInstance( spec, Specialization )
		self.assertIs( spec.base, self.discovery.get_intrinsics()['ConstPtr'] )
		self.assertEqual( spec.args, [ self.discovery.get_intrinsics()['u8'] ])

	def test_specialization_two_type_params_like_result( self ) -> None:
		mod = self._import( '''
@cstruct
class Result[T,E]:
	pass

def make_result() -> Result[i32,usize]:
	pass
''' )
		fn = mod.get_local( 'make_result' )
		fn.resolve()
		spec = fn.return_type
		self.assertIsInstance( spec, Specialization )
		result_cls = mod.get_local( 'Result' )
		self.assertIs( spec.base, result_cls )
		intrinsics = self.discovery.get_intrinsics()
		self.assertEqual( spec.args, [ intrinsics['i32'], intrinsics['usize'] ])

		# reuse dedups to the identical object
		spec2 = self.discovery._get_or_create_specialization( result_cls, [ intrinsics['i32'], intrinsics['usize'] ])
		self.assertIs( spec, spec2 )

	def test_wrong_arg_count_errors( self ) -> None:
		mod = self._import( '''
@cstruct
class Result[T,E]:
	pass

def make_result() -> Result[i32]:
	pass
''' )
		fn = mod.get_local( 'make_result' )
		with self.assertRaises( AssertionError ):
			fn.resolve()

	def test_non_generic_subscript_errors( self ) -> None:
		mod = self._import( '''
def foo() -> i32[i32]:
	pass
''' )
		fn = mod.get_local( 'foo' )
		with self.assertRaises( AssertionError ):
			fn.resolve()


class InheritanceTests( unittest.TestCase ):
	''' single inheritance only - RCClass.base is resolved eagerly, at class-creation time, same as type_params (see ShallowScanTests) '''

	def setUp( self ) -> None:
		self.discovery = discovery.Discovery( import_builtins = False )

	def _import( self, code: str ) -> Module:
		return self.discovery.import_code( code, Path( '__main__.py' ), scope = None )

	def test_single_base_resolved_eagerly( self ) -> None:
		mod = self._import( '''
class Base:
	pass

class Derived( Base ):
	pass
''' )
		base = mod.get_local( 'Base' )
		derived = mod.get_local( 'Derived' )
		self.assertIsInstance( derived, RCClass )
		self.assertIs( derived.base, base ) # resolved before derived.resolve() ever runs
		self.assertIsNotNone( derived.resolve ) # but derived's own body is still deferred

	def test_no_base_leaves_base_none( self ) -> None:
		mod = self._import( '''
class Foo:
	pass
''' )
		self.assertIsNone( mod.get_local( 'Foo' ).base )

	def test_multiple_inheritance_errors( self ) -> None:
		with self.assertRaises( AssertionError ):
			self._import( '''
class A:
	pass

class B:
	pass

class C( A, B ):
	pass
''' )

	def test_subclassing_non_rcclass_errors( self ) -> None:
		with self.assertRaises( AssertionError ):
			self._import( '''
@cstruct
class Point:
	x: i32

class Foo( Point ):
	pass
''' )


class FunctionParameterTests( unittest.TestCase ):
	''' full ast.arguments coverage - stage 2 needs the kind flags plus `default` to bind keyword/optional call-site arguments down to positional ones '''

	def setUp( self ) -> None:
		self.discovery = discovery.Discovery( import_builtins = False )

	def _import( self, code: str ) -> Module:
		return self.discovery.import_code( code, Path( '__main__.py' ), scope = None )

	def test_positional_only( self ) -> None:
		mod = self._import( '''
def foo( x: i32, / ) -> None:
	pass
''' )
		foo = mod.get_local( 'foo' )
		foo.resolve()
		self.assertEqual( len( foo.parameters ), 1 )
		p = foo.parameters[0]
		self.assertEqual( p.stem, 'x' )
		self.assertTrue( p.is_posonly )
		self.assertFalse( p.is_kwonly or p.is_vararg or p.is_kwarg )
		self.assertIsNone( p.default )

	def test_default_value_captured_unresolved( self ) -> None:
		mod = self._import( '''
def foo( x: i32 = 1 ) -> None:
	pass
''' )
		foo = mod.get_local( 'foo' )
		foo.resolve()
		p = foo.parameters[0]
		self.assertIsNotNone( p.default ) # raw ast.expr - stage 2's concern to evaluate, same as fn.node's body
		self.assertEqual( p.default.value, 1 )

	def test_keyword_only_with_and_without_default( self ) -> None:
		mod = self._import( '''
def foo( *, x: i32, y: usize = 2 ) -> None:
	pass
''' )
		foo = mod.get_local( 'foo' )
		foo.resolve()
		x, y = foo.parameters
		self.assertTrue( x.is_kwonly )
		self.assertIsNone( x.default )
		self.assertTrue( y.is_kwonly )
		self.assertIsNotNone( y.default )

	def test_vararg_and_kwarg( self ) -> None:
		mod = self._import( '''
def foo( *args: i32, **kwargs: usize ) -> None:
	pass
''' )
		foo = mod.get_local( 'foo' )
		foo.resolve()
		args, kwargs = foo.parameters
		self.assertEqual( args.stem, 'args' )
		self.assertTrue( args.is_vararg )
		self.assertIs( args.type, self.discovery.get_intrinsics()['i32'] )
		self.assertEqual( kwargs.stem, 'kwargs' )
		self.assertTrue( kwargs.is_kwarg )
		self.assertIs( kwargs.type, self.discovery.get_intrinsics()['usize'] )

	def test_mixed_signature_full_shape( self ) -> None:
		mod = self._import( '''
def foo( a: i32, /, b: i32 = 1, *args: i32, c: i32, d: i32 = 2, **kwargs: i32 ) -> None:
	pass
''' )
		foo = mod.get_local( 'foo' )
		foo.resolve()
		self.assertEqual( [ p.stem for p in foo.parameters ], [ 'a', 'b', 'args', 'c', 'd', 'kwargs' ])
		a, b, args, c, d, kwargs = foo.parameters
		self.assertTrue( a.is_posonly )
		self.assertIsNone( a.default )
		self.assertFalse( b.is_posonly )
		self.assertIsNotNone( b.default )
		self.assertTrue( args.is_vararg )
		self.assertTrue( c.is_kwonly )
		self.assertIsNone( c.default )
		self.assertTrue( d.is_kwonly )
		self.assertIsNotNone( d.default )
		self.assertTrue( kwargs.is_kwarg )

	def test_self_and_cls_still_skipped_alongside_other_kinds( self ) -> None:
		mod = self._import( '''
class Foo:
	def bar( self, *, x: i32 ) -> None:
		pass

	@classmethod
	def make( cls, *args: i32 ) -> None:
		pass
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		bar = foo.get_local( 'bar' )
		bar.resolve()
		self.assertEqual( [ p.stem for p in bar.parameters ], [ 'x' ])

		make = foo.get_local( 'make' )
		make.resolve()
		self.assertEqual( [ p.stem for p in make.parameters ], [ 'args' ])


class MoveTypeTests( unittest.TestCase ):
	''' move[T] in annotation position - recognized textually (like @move) rather than resolved through find_name, so it works even though `move` is never a real bound name anywhere '''

	def setUp( self ) -> None:
		self.discovery = discovery.Discovery( import_builtins = False )

	def _import( self, code: str ) -> Module:
		return self.discovery.import_code( code, Path( '__main__.py' ), scope = None )

	def test_move_wraps_inner_type( self ) -> None:
		mod = self._import( '''
class Foo:
	pass

def consume( x: move[Foo] ) -> None:
	pass
''' )
		fn = mod.get_local( 'consume' )
		fn.resolve()
		p = fn.parameters[0]
		self.assertIsInstance( p.type, Move )
		self.assertIs( p.type.inner, mod.get_local( 'Foo' ))

	def test_move_dedups_to_identical_object( self ) -> None:
		mod = self._import( '''
class Foo:
	pass

def consume( x: move[Foo] ) -> None:
	pass

def consume2( y: move[Foo] ) -> None:
	pass
''' )
		consume = mod.get_local( 'consume' )
		consume2 = mod.get_local( 'consume2' )
		consume.resolve()
		consume2.resolve()
		self.assertIs( consume.parameters[0].type, consume2.parameters[0].type )

	def test_move_multiple_args_errors( self ) -> None:
		mod = self._import( '''
class Foo:
	pass

class Bar:
	pass

def consume( x: move[Foo, Bar] ) -> None:
	pass
''' )
		fn = mod.get_local( 'consume' )
		with self.assertRaises( AssertionError ):
			fn.resolve()

	def test_move_decorator_flag_on_function( self ) -> None:
		mod = self._import( '''
class Foo:
	@move
	def release( self ) -> None:
		pass
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		release = foo.get_local( 'release' )
		self.assertTrue( release.is_move )


class UnsupportedDecoratorTests( unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = discovery.Discovery( import_builtins = False )

	def _import( self, code: str ) -> Module:
		return self.discovery.import_code( code, Path( '__main__.py' ), scope = None )

	def test_property_is_not_yet_supported( self ) -> None:
		# intentionally unimplemented for now, deferred until other problems
		# are solved (see Discovery/ARCHITECTURE.md discussion) - this test
		# just locks in that it fails loudly rather than silently doing the
		# wrong thing, so implementing it later is a deliberate decision.
		# @property lives inside the class body, which is itself deferred
		# behind .resolve() (see ShallowScanTests), so the error only
		# surfaces once something actually asks for it
		mod = self._import( '''
class Foo:
	@property
	def bar( self ) -> i32:
		return 1
''' )
		foo = mod.get_local( 'Foo' )
		with self.assertRaises( AssertionError ):
			foo.resolve()


class CircularImportTests( unittest.TestCase ):
	'''
	regression: import_name() used to register a module in self.modules only
	after its body had been fully scanned, so a re-entrant import (A imports
	B imports A) recursed forever instead of resolving back to the (still
	being scanned) module
	'''

	def test_mutually_importing_modules_resolve_without_recursing( self ) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			root = Path( tmp )
			( root / 'a.py' ).write_text( 'import b\nclass A:\n\tpass\n' )
			( root / 'b.py' ).write_text( 'import a\nclass B:\n\tpass\n' )

			disco = discovery.Discovery( paths = [ root ], import_builtins = False )
			mod_a = disco.import_name( 'a' )
			mod_b = disco.modules['b']

			self.assertIsInstance( mod_a.get_local( 'A' ), RCClass )
			self.assertIsInstance( mod_b.get_local( 'B' ), RCClass )
			# each module's `import` of the other resolved back to the same
			# (partially-scanned-at-the-time) Module object, not a duplicate
			self.assertIs( mod_b.get_local( 'a' ), mod_a )
			self.assertIs( mod_a.get_local( 'b' ), mod_b )


class OverloadTests( unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = discovery.Discovery( import_builtins = False )

	def _import( self, code: str ) -> Module:
		return self.discovery.import_code( code, Path( '__main__.py' ), scope = None )

	def test_stub_plus_plain_implementation( self ) -> None:
		# mirrors builtins.Result.unwrap_or: one @overload stub (no real
		# body) declaring the public signature, one plain implementation
		mod = self._import( '''
class Foo:
	@overload
	def unwrap_or( self, default: i32 ) -> i32:
		...
	def unwrap_or( self, default: usize ) -> usize:
		return default
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		group = foo.get_local( 'unwrap_or' )
		self.assertIsInstance( group, Overload )
		self.assertEqual( len( group.stubs ), 1 )
		self.assertEqual( len( group.implementations ), 1 )

		group.stubs[0].resolve()
		group.implementations[0].resolve()
		self.assertIsNone( group.stubs[0].resolve )
		self.assertIsNone( group.implementations[0].resolve )

	def test_two_real_implementations_distinct_params( self ) -> None:
		# mirrors builtins.str.from_cstr: two @overload defs, both with
		# real bodies, no plain fallback - genuinely distinct callables
		mod = self._import( '''
class Foo:
	@overload
	@staticmethod
	def from_thing( buf: i32, length: usize ) -> i32:
		return buf

	@overload
	@staticmethod
	def from_thing( src: usize ) -> i32:
		return 0
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		group = foo.get_local( 'from_thing' )
		self.assertIsInstance( group, Overload )
		self.assertEqual( len( group.stubs ), 0 )
		self.assertEqual( len( group.implementations ), 2 )
		self.assertTrue( all( fn.is_static for fn in group.implementations ))

		for fn in group.implementations:
			fn.resolve()
		self.assertEqual( [ p.stem for p in group.implementations[0].parameters ], [ 'buf', 'length' ])
		self.assertEqual( [ p.stem for p in group.implementations[1].parameters ], [ 'src' ])

	def test_group_registered_in_class_methods( self ) -> None:
		mod = self._import( '''
class Foo:
	@overload
	def bar( self, x: i32 ) -> i32:
		...
	def bar( self, x: usize ) -> usize:
		return x
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		group = foo.get_local( 'bar' )
		self.assertIn( group, foo.methods )
		self.assertEqual( len( foo.methods ), 1 ) # not duplicated per overload member


class CompilerTargetTests( unittest.TestCase ):
	''' @compiler.target(...) filtering - excluded functions never enter the type system at all '''

	def _import( self, code: str, active_target: dict[str,str] ) -> tuple[discovery.Discovery, Module]:
		disco = discovery.Discovery( import_builtins = False, active_target = active_target )
		mod = disco.import_code( code, Path( '__main__.py' ), scope = None )
		return disco, mod

	def test_matching_target_included( self ) -> None:
		disco, mod = self._import( '''
@compiler.target( os = 'windows' )
def foo() -> i32:
	pass
''', { 'os': 'windows' })
		self.assertIsInstance( mod.get_local( 'foo' ), Function )

	def test_non_matching_target_excluded( self ) -> None:
		disco, mod = self._import( '''
@compiler.target( os = 'windows' )
def foo() -> i32:
	pass
''', { 'os': 'linux' })
		self.assertIsNone( mod.get_local( 'foo' ))

	def test_negated_target_value( self ) -> None:
		disco, mod = self._import( '''
@compiler.target( os = not 'windows' )
def foo() -> i32:
	pass
''', { 'os': 'linux' })
		self.assertIsInstance( mod.get_local( 'foo' ), Function )

	def test_same_named_functions_pick_one_winner( self ) -> None:
		# mirrors lib/sys.py's cstrlen pattern: two same-named defs guarded by
		# mutually exclusive targets - only the matching one should ever get
		# registered (the other is skipped before scope.add_name is called)
		disco, mod = self._import( '''
@compiler.target( os = 'windows' )
def cstrlen() -> i32:
	pass

@compiler.target( os = not 'windows' )
def cstrlen() -> i32:
	pass
''', { 'os': 'linux' })
		fn = mod.get_local( 'cstrlen' )
		self.assertIsInstance( fn, Function )
		self.assertNotIsInstance( fn, Overload )

	def test_unmodeled_keyword_is_inert( self ) -> None:
		disco, mod = self._import( '''
@compiler.target( os = 'windows', vendor = 'mingw' )
def foo() -> i32:
	pass
''', { 'os': 'windows' }) # 'vendor' isn't a modeled key at all
		self.assertIsInstance( mod.get_local( 'foo' ), Function )

	def test_matching_method_on_class( self ) -> None:
		disco, mod = self._import( '''
class Stdout:
	@compiler.target( os = 'windows' )
	def write( self ) -> None:
		pass

	@compiler.target( os = not 'windows' )
	def write( self ) -> None:
		pass
''', { 'os': 'windows' })
		stdout = mod.get_local( 'Stdout' )
		stdout.resolve()
		write = stdout.get_local( 'write' )
		self.assertIsInstance( write, Function )
		self.assertEqual( len( stdout.methods ), 1 )


class RealLibSmokeTest( unittest.TestCase ):
	''' confirms discovery no longer crashes on the actual example library, not just synthetic snippets '''

	def setUp( self ) -> None:
		self.discovery = discovery.Discovery( import_builtins = True )
		self.builtins_mod = self.discovery.modules['builtins']

	def test_result_plain_method_resolves( self ) -> None:
		result_cls = self.builtins_mod.get_local( 'Result' )
		self.assertIsInstance( result_cls, CStruct )
		result_cls.resolve()

		ok = result_cls.get_local( 'Ok' )
		self.assertIsInstance( ok, Function )
		ok.resolve()
		self.assertIsNone( ok.resolve )
		self.assertIsInstance( ok.return_type, Specialization )
		self.assertIs( ok.return_type.base, result_cls )

	def test_result_unwrap_or_overload_resolves( self ) -> None:
		result_cls = self.builtins_mod.get_local( 'Result' )
		result_cls.resolve()
		group = result_cls.get_local( 'unwrap_or' )
		self.assertIsInstance( group, Overload )
		for fn in ( *group.stubs, *group.implementations ):
			fn.resolve()

	def test_str_from_cstr_overload_resolves( self ) -> None:
		str_cls = self.builtins_mod.get_local( 'str' )
		self.assertIsInstance( str_cls, RCClass )
		str_cls.resolve()
		group = str_cls.get_local( 'from_cstr' )
		self.assertIsInstance( group, Overload )
		self.assertEqual( len( group.implementations ), 2 )
		group.implementations[0].resolve() # buf/length
		self.assertIsNone( group.implementations[0].resolve )

		group.implementations[1].resolve() # src: move[bytearray]
		self.assertIsNone( group.implementations[1].resolve )
		src = group.implementations[1].parameters[0]
		self.assertIsInstance( src.type, Move )
		self.assertIs( src.type.inner, self.builtins_mod.get_local( 'bytearray' ))

	def test_bytearray_resolves( self ) -> None:
		ba_cls = self.builtins_mod.get_local( 'bytearray' )
		self.assertIsInstance( ba_cls, RCClass )
		ba_cls.resolve()
		# not resolving release(): its return annotation references a bare
		# `OwnershipError` name that lib/builtins/__init__.py never actually
		# imports (only `sys.OwnershipError` exists) - a pre-existing gap in
		# the example library, not something in scope for this pass
		get_ptr = ba_cls.get_local( 'get_ptr' )
		get_ptr.resolve()
		self.assertIsInstance( get_ptr.return_type, Specialization )

	def test_sys_cstrlen_target_filtering( self ) -> None:
		# lib/sys.py declares two `cstrlen` defs guarded by mutually
		# exclusive @compiler.target(os=...) decorators - previously this
		# crashed discovery entirely (unrecognized decorator)
		sys_mod = self.discovery.modules['sys']
		cstrlen = sys_mod.get_local( 'cstrlen' )
		self.assertIsInstance( cstrlen, Function )
		self.assertNotIsInstance( cstrlen, Overload )
		cstrlen.resolve()
		self.assertIsNone( cstrlen.resolve )

	def test_codecs_ascii_subclasses_codec( self ) -> None:
		# lib/codecs/ascii.py (and cp437/latin1/utf8) declare `class ascii(
		# Codec):` - the motivating real-world case for RCClass.base. Not
		# reached by the plain `import_builtins=True` walk (codecs/__init__.py
		# only imports these from inside a function body, which discovery
		# never scans), so import it explicitly.
		ascii_mod = self.discovery.import_name( 'codecs.ascii' )
		codecs_mod = self.discovery.modules['codecs']
		ascii_cls = ascii_mod.get_local( 'ascii' )
		self.assertIsInstance( ascii_cls, RCClass )
		self.assertIs( ascii_cls.base, codecs_mod.get_local( 'Codec' ))


if __name__ == '__main__':
	logging.basicConfig( level = logging.DEBUG, force = True )
	unittest.main()
