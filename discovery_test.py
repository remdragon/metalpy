# stdlib imports
import logging
from pathlib import Path
import tempfile
import unittest

# local imports
import discovery
from errors import CompileError
from mpy_types import (
	Module, RCClass, CStruct, CUnion, CEnum, TaggedUnion, Overload,
	Function, Variable, Specialization, Move, ConditionalDispatch,
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

		disco1 = MockDiscovery.new_test( 'codecs' )
		disco1.import_code( 'from . import utf8', Path( '__irrelevant__.py' ), scope = None )
		self.assertTrue( disco1.errors.errors )

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
		self.assertTrue( x.is_global )
		self.assertTrue( y.is_global )

	def test_class_attribute_is_not_flagged_as_a_global( self ) -> None:
		# Variable.is_global is what lets Compiler._enqueue tell a genuine
		# module-level global apart from a class attribute - both go through
		# the exact same visit_AnnAssign/visit_Assign code, distinguished only
		# by whether scope_stack[-1] is the module itself
		mod = self._import( '''
class Foo:
	x: i32
	y = 2
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		x = foo.get_local( 'x' )
		y = foo.get_local( 'y' )
		self.assertIsInstance( x, Variable )
		self.assertIsInstance( y, Variable )
		self.assertFalse( x.is_global )
		self.assertFalse( y.is_global )


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
		x.resolve()
		self.assertIn( 'add an annotation instead', self.discovery.errors.errors[0] )

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
		fn.resolve()
		self.assertIn( 'expects 2 type argument', self.discovery.errors.errors[0] )

	def test_non_generic_subscript_errors( self ) -> None:
		mod = self._import( '''
def foo() -> i32[i32]:
	pass
''' )
		fn = mod.get_local( 'foo' )
		fn.resolve()
		self.assertIn( 'is not generic', self.discovery.errors.errors[0] )


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
		self._import( '''
class A:
	pass

class B:
	pass

class C( A, B ):
	pass
''' )
		self.assertIn( 'multiple inheritance', self.discovery.errors.errors[0] )

	def test_subclassing_non_rcclass_errors( self ) -> None:
		self._import( '''
@cstruct
class Point:
	x: i32

class Foo( Point ):
	pass
''' )
		self.assertIn( 'cannot subclass', self.discovery.errors.errors[0] )


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
		fn.resolve()
		self.assertIn( 'exactly one type argument', self.discovery.errors.errors[0] )

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
		foo.resolve()
		self.assertIn( 'unsupported function decorator', self.discovery.errors.errors[0] )


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
		# body) declaring the public signature, one plain implementation whose
		# accepted type (i32|usize) covers everything the stub declares (i32)
		mod = self._import( '''
class Foo:
	@overload
	def unwrap_or( self, default: i32 ) -> i32:
		...
	def unwrap_or( self, default: i32|usize ) -> i32|usize:
		return default
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		group = foo.get_local( 'unwrap_or' )
		self.assertIsInstance( group, Overload )
		self.assertEqual( len( group.stubs ), 1 )
		self.assertEqual( len( group.implementations ), 1 )

		group.stubs[0].resolve()
		if group.implementations[0].resolve is not None: # binding may already have cross-resolved it
			group.implementations[0].resolve()
		self.assertIsNone( group.stubs[0].resolve )
		self.assertIsNone( group.implementations[0].resolve )
		self.assertIs( group.stubs[0].bound_to, group.implementations[0] )

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

	def test_two_plain_implementations_with_no_overload_decorator_form_a_group( self ) -> None:
		# mirrors the real builtins.str.from_cstr: neither def is @overload -
		# their distinct arities alone are enough to make them unambiguous, so
		# no @overload is needed anywhere for this to be legal
		mod = self._import( '''
class Foo:
	@staticmethod
	def from_thing( buf: i32, length: usize ) -> i32:
		return buf

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
		self.assertIn( group, foo.methods )
		self.assertEqual( len( foo.methods ), 1 ) # first def's own methods-entry was folded into the group, not left dangling

		for fn in group.implementations:
			if fn.resolve is not None: # resolving one cross-resolves its sibling too, via the ambiguity check
				fn.resolve()
		self.assertEqual( [ p.stem for p in group.implementations[0].parameters ], [ 'buf', 'length' ])
		self.assertEqual( [ p.stem for p in group.implementations[1].parameters ], [ 'src' ])

	def test_module_level_plain_redefinition_forms_a_group( self ) -> None:
		# same as above, but at module scope rather than inside a class - the
		# group-formation path isn't class_obj-specific
		mod = self._import( '''
class usize: pass
class str: pass

def parse( x: usize ) -> usize:
	return x

def parse( x: str ) -> usize:
	return 0
''' )
		group = mod.get_local( 'parse' )
		self.assertIsInstance( group, Overload )
		self.assertEqual( len( group.implementations ), 2 )


class OverloadWellFormednessTests( unittest.TestCase ):
	''' binding (@overload stub -> plain implementation) and the well-formedness checks - all triggered from Function.resolve() '''

	def setUp( self ) -> None:
		self.discovery = discovery.Discovery( import_builtins = False )

	def _import( self, code: str ) -> Module:
		return self.discovery.import_code( code, Path( '__main__.py' ), scope = None )

	def test_worked_example_bindings( self ) -> None:
		# foo#1..foo#4 from the design conversation: two stubs, two plain
		# implementations, each stub binds to exactly one implementation
		mod = self._import( '''
class int: pass
class str: pass
class bytes: pass

@overload
def foo( x: int ) -> None:
	...

@overload
def foo( x: str = '' ) -> None:
	...

def foo( x: int|None = None ) -> None:
	pass

def foo( x: str|bytes ) -> None:
	pass
''' )
		group = mod.get_local( 'foo' )
		self.assertEqual( len( group.stubs ), 2 )
		self.assertEqual( len( group.implementations ), 2 )
		foo1, foo2 = group.stubs
		foo3, foo4 = group.implementations

		foo1.resolve()
		if foo2.resolve is not None:
			foo2.resolve()
		self.assertIs( foo1.bound_to, foo3 )
		self.assertIs( foo2.bound_to, foo4 )

	def test_stub_covered_by_nothing_errors( self ) -> None:
		mod = self._import( '''
class bool: pass
class int: pass
class str: pass

@overload
def foo( x: bool ) -> None:
	...

def foo( x: int ) -> None:
	pass

def foo( x: str ) -> None:
	pass
''' )
		group = mod.get_local( 'foo' )
		group.stubs[0].resolve()
		self.assertIn( 'no implementation covers', self.discovery.errors.errors[0] )

	def test_overlapping_plain_implementations_error( self ) -> None:
		# ambiguity between plain implementations is detected purely by
		# resolving them - no call site involved
		mod = self._import( '''
class int: pass
class str: pass

@overload
def foo( x: int ) -> None:
	...

def foo( x: str ) -> None:
	pass

def foo( x: str ) -> None:
	pass
''' )
		group = mod.get_local( 'foo' )
		group.implementations[0].resolve()
		self.assertIn( 'ambiguous', self.discovery.errors.errors[0] )

	def test_overlapping_plain_implementations_error_with_no_overload_decorator_at_all( self ) -> None:
		# same ambiguity, but with zero @overload decorators anywhere - group
		# formation itself (not just the well-formedness check) must trigger
		# purely off the plain redefinition
		mod = self._import( '''
class str: pass

def foo( x: str ) -> None:
	pass

def foo( x: str ) -> None:
	pass
''' )
		group = mod.get_local( 'foo' )
		self.assertIsInstance( group, Overload )
		group.implementations[0].resolve()
		self.assertIn( 'ambiguous', self.discovery.errors.errors[0] )

	def test_shadowed_stub_errors( self ) -> None:
		# an earlier stub's str|bytes fully covers the later stub's str -
		# the later one can never be reached (first-match always picks the
		# earlier one first)
		mod = self._import( '''
class str: pass
class bytes: pass

@overload
def foo( x: str|bytes ) -> None:
	...

@overload
def foo( x: str ) -> None:
	...

def foo( x: str|bytes ) -> None:
	pass
''' )
		group = mod.get_local( 'foo' )
		group.stubs[1].resolve()
		self.assertIn( 'shadowed', self.discovery.errors.errors[0] )

	def test_shadowed_by_real_bodied_overload_errors( self ) -> None:
		# shadowing applies across stubs and real-bodied @overload members
		# alike, not just stub-vs-stub - both are tried first-match together
		mod = self._import( '''
class str: pass
class bytes: pass

@overload
def foo( x: str|bytes ) -> None:
	pass

@overload
def foo( x: str ) -> None:
	...

def foo( x: str|bytes ) -> None:
	pass
''' )
		group = mod.get_local( 'foo' )
		group.stubs[0].resolve()
		self.assertIn( 'shadowed', self.discovery.errors.errors[0] )

	def test_real_bodied_overload_group_unaffected( self ) -> None:
		# str.from_cstr-style: two @overload arms, different arity, no stub,
		# no plain fallback - regression check that the new checks don't
		# false-positive on this existing pattern (different arity means
		# _is_covered_by/_overlaps are trivially False for this pair)
		mod = self._import( '''
class int: pass
class str: pass

class Foo:
	@overload
	@staticmethod
	def make( a: int, b: str ) -> int:
		return a

	@overload
	@staticmethod
	def make( a: str ) -> int:
		return 0
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		group = foo.get_local( 'make' )
		self.assertEqual( len( group.stubs ), 0 )
		self.assertEqual( len( group.implementations ), 2 )
		for fn in group.implementations:
			fn.resolve()
		self.assertIsNone( group.implementations[0].resolve )
		self.assertIsNone( group.implementations[1].resolve )


class OverloadCallResolutionTests( unittest.TestCase ):
	''' Overload.resolve_call - pure function of types, no AST/call-site involved, so this is testable ahead of stage 2 '''

	def setUp( self ) -> None:
		self.discovery = discovery.Discovery( import_builtins = False )

	def _import( self, code: str ) -> Module:
		return self.discovery.import_code( code, Path( '__main__.py' ), scope = None )

	def _worked_example( self ):
		mod = self._import( '''
class bool: pass
class int: pass
class str: pass
class bytes: pass

@overload
def foo( x: int ) -> None:
	...

@overload
def foo( x: str = '' ) -> None:
	...

def foo( x: int|None = None ) -> None:
	pass

def foo( x: str|bytes ) -> None:
	pass
''' )
		group = mod.get_local( 'foo' )
		foo3, foo4 = group.implementations
		return (
			group, foo3, foo4,
			mod.get_local( 'bool' ), mod.get_local( 'int' ), mod.get_local( 'str' ), mod.get_local( 'bytes' ),
		)

	def test_int_resolves_via_first_stub( self ) -> None:
		group, foo3, foo4, bool_cls, int_cls, str_cls, bytes_cls = self._worked_example()
		branches, default = group.resolve_call( [ int_cls ], {} )
		self.assertEqual( branches, [] )
		self.assertIs( default, foo3 )

	def test_str_resolves_via_second_stub( self ) -> None:
		group, foo3, foo4, bool_cls, int_cls, str_cls, bytes_cls = self._worked_example()
		branches, default = group.resolve_call( [ str_cls ], {} )
		self.assertEqual( branches, [] )
		self.assertIs( default, foo4 )

	def test_none_falls_through_to_unique_implementation( self ) -> None:
		group, foo3, foo4, bool_cls, int_cls, str_cls, bytes_cls = self._worked_example()
		branches, default = group.resolve_call( [ self.discovery.get_none_type() ], {} )
		self.assertEqual( branches, [] )
		self.assertIs( default, foo3 )

	def test_bytes_falls_through_to_unique_implementation( self ) -> None:
		group, foo3, foo4, bool_cls, int_cls, str_cls, bytes_cls = self._worked_example()
		branches, default = group.resolve_call( [ bytes_cls ], {} )
		self.assertEqual( branches, [] )
		self.assertIs( default, foo4 )

	def test_union_argument_produces_conditional_dispatch( self ) -> None:
		group, foo3, foo4, bool_cls, int_cls, str_cls, bytes_cls = self._worked_example()
		union = self.discovery._get_or_create_union( [ int_cls, bytes_cls ] )
		branches, default = group.resolve_call( [ union ], {} )
		self.assertEqual( len( branches ), 1 )
		self.assertIs( default, foo4 )
		self.assertIs( branches[0].function, foo3 )
		self.assertEqual( len( branches[0].conditions ), 1 )
		param, expected = branches[0].conditions[0]
		self.assertIs( param, foo3.parameters[0] )
		self.assertIs( expected, int_cls )

	def test_uncovered_type_errors( self ) -> None:
		# resolve_call is a pure function of types with no Discovery reference
		# by design (see mpy_types.py) - it raises CompileError directly,
		# unrecorded; a real call site (lowering.py) attaches location and
		# records it via Discovery.fail() before it would surface here
		group, foo3, foo4, bool_cls, int_cls, str_cls, bytes_cls = self._worked_example()
		with self.assertRaises( CompileError ):
			group.resolve_call( [ bool_cls ], {} )

	def test_kwargs_match_differently_named_parameters( self ) -> None:
		mod = self._import( '''
class int: pass
class str: pass

@overload
def f( a: int ) -> None:
	...

@overload
def f( b: str ) -> None:
	...

def f( a: int ) -> None:
	pass

def f( b: str ) -> None:
	pass
''' )
		group = mod.get_local( 'f' )
		int_impl, str_impl = group.implementations
		int_cls = mod.get_local( 'int' )
		str_cls = mod.get_local( 'str' )

		branches, default = group.resolve_call( [], { 'a': int_cls } )
		self.assertEqual( branches, [] )
		self.assertIs( default, int_impl )

		branches, default = group.resolve_call( [], { 'b': str_cls } )
		self.assertEqual( branches, [] )
		self.assertIs( default, str_impl )

	def test_omitted_parameter_with_default_still_matches( self ) -> None:
		mod = self._import( '''
class int: pass
class str: pass

@overload
def f( a: int, b: str = '' ) -> None:
	...

def f( a: int, b: str = '' ) -> None:
	pass
''' )
		group = mod.get_local( 'f' )
		impl = group.implementations[0]
		int_cls = mod.get_local( 'int' )

		branches, default = group.resolve_call( [ int_cls ], {} ) # 'b' entirely omitted
		self.assertEqual( branches, [] )
		self.assertIs( default, impl )

	def test_multi_parameter_cartesian_product( self ) -> None:
		mod = self._import( '''
class int: pass
class str: pass

@overload
def pair( a: int, b: int ) -> None:
	...

def pair( a: int, b: int ) -> None:
	pass
def pair( a: int, b: str ) -> None:
	pass
def pair( a: str, b: int ) -> None:
	pass
def pair( a: str, b: str ) -> None:
	pass
''' )
		group = mod.get_local( 'pair' )
		p_ii, p_is, p_si, p_ss = group.implementations
		int_cls = mod.get_local( 'int' )
		str_cls = mod.get_local( 'str' )
		union = self.discovery._get_or_create_union( [ int_cls, str_cls ] )

		branches, default = group.resolve_call( [ union, union ], {} )
		target_ids = { id( b.function ) for b in branches } | { id( default ) }
		self.assertEqual( target_ids, { id( p_ii ), id( p_is ), id( p_si ), id( p_ss ) })


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
			if fn.resolve is not None: # binding/ambiguity checks may already have cross-resolved a sibling
				fn.resolve()
		self.assertIs( group.stubs[0].bound_to, group.implementations[0] )

	def test_str_from_cstr_overload_resolves( self ) -> None:
		str_cls = self.builtins_mod.get_local( 'str' )
		self.assertIsInstance( str_cls, RCClass )
		str_cls.resolve()
		group = str_cls.get_local( 'from_cstr' )
		self.assertIsInstance( group, Overload )
		self.assertEqual( len( group.implementations ), 2 )
		group.implementations[0].resolve() # buf/length; cross-resolves its sibling too, as a side effect of the ambiguity check
		self.assertIsNone( group.implementations[0].resolve )

		if group.implementations[1].resolve is not None: # src: move[bytearray]
			group.implementations[1].resolve()
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

	def test_transitively_imported_module_gets_working_builtins( self ) -> None:
		# regression: Discovery used to keep builtins on a dedicated
		# self.builtins attribute, only assigned after import_name('builtins')
		# fully returned - so codecs/sys/etc (imported from *inside*
		# builtins/__init__.py's own body, while that assignment was still
		# pending) permanently got Module.builtins=None, and any bare
		# builtin-type reference in them (e.g. Codec.encode's `s: str`)
		# couldn't resolve. now builtins is looked up from self.modules,
		# which already has 'builtins' registered by this point (see
		# import_code)
		codecs_mod = self.discovery.modules['codecs']
		self.assertTrue( codecs_mod.builtins )
		codec = codecs_mod.get_local( 'Codec' )
		codec.resolve()
		encode = codec.get_local( 'encode' )
		encode.resolve()
		self.assertIs( encode.parameters[0].type, self.builtins_mod.get_local( 'str' ))


class MainSymbolTests( unittest.TestCase ):
	''' Discovery.main gives stage 2 an easy way to find the entry point (see compiler.py's Compiler.run) '''

	def setUp( self ) -> None:
		self.discovery = discovery.Discovery( import_builtins = False )

	def test_no_main_defined( self ) -> None:
		self.discovery.import_code( '''
def not_main() -> None:
	pass
''', Path( '__main__.py' ), scope = None )
		self.assertIsNone( self.discovery.main )

	def test_main_gets_set( self ) -> None:
		self.discovery.import_code( '''
def main() -> None:
	pass
''', Path( '__main__.py' ), scope = None )
		self.assertIsInstance( self.discovery.main, Function )
		self.assertEqual( self.discovery.main.qualname, 'main' )


class ErrorCollectionTests( unittest.TestCase ):
	''' errors record into Discovery.errors and processing continues at the nearest recovery boundary, instead of raising uncaught '''

	def setUp( self ) -> None:
		self.discovery = discovery.Discovery( import_builtins = False )

	def _import( self, code: str ) -> Module:
		return self.discovery.import_code( code, Path( '__main__.py' ), scope = None )

	def test_two_independent_top_level_errors_both_collected( self ) -> None:
		# each bad decorator fires during the immediate (non-deferred) part of
		# ClassDef scanning - confirms import_code's per-top-level-statement
		# recovery boundary collects both instead of stopping at the first
		mod = self._import( '''
@nonsense_decorator_one
class Foo:
	pass

@nonsense_decorator_two
class Bar:
	pass
''' )
		self.assertEqual( len( self.discovery.errors.errors ), 2 )
		self.assertIn( 'nonsense_decorator_one', self.discovery.errors.errors[0] )
		self.assertIn( 'nonsense_decorator_two', self.discovery.errors.errors[1] )
		# neither class made it far enough to be registered - both decorators
		# are checked before _parse_ClassDef_RCClass's scope.add_name() runs
		self.assertIsNone( mod.get_local( 'Foo' ))
		self.assertIsNone( mod.get_local( 'Bar' ))

	def test_broken_function_does_not_taint_sibling_resolution( self ) -> None:
		mod = self._import( '''
def broken( x ) -> None:
	pass

def fine( x: i32 ) -> i32:
	return x
''' )
		broken = mod.get_local( 'broken' )
		fine = mod.get_local( 'fine' )
		broken.resolve()
		self.assertEqual( len( self.discovery.errors.errors ), 1 )
		self.assertIn( 'no type annotation', self.discovery.errors.errors[0] )

		fine.resolve()
		self.assertEqual( len( self.discovery.errors.errors ), 1 ) # unchanged - fine resolved cleanly
		self.assertIsNone( fine.resolve )
		self.assertEqual( len( fine.parameters ), 1 )


if __name__ == '__main__':
	logging.basicConfig( level = logging.DEBUG, force = True )
	unittest.main()
