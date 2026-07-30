# stdlib imports
import logging
from pathlib import Path
import unittest

# local imports
import discovery
from mpy_types import (
	Module, RCClass, CStruct, CUnion, CEnum, TaggedUnion, Overload,
	Function, Variable, Specialization,
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
	a module or class body is scanned exactly once, immediately: every
	class/function/global/attribute it directly contains is discoverable by
	name right away. What's still deferred is each individual function's
	parameters/return type and each individual variable's type - .resolve is
	not None until something calls it.
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

	def test_class_attribute_and_method_registered_immediately_but_deferred( self ) -> None:
		mod = self._import( '''
class Foo:
	x: i32
	def bar( self ) -> i32:
		return self.x
''' )
		foo = mod.get_local( 'Foo' )
		self.assertEqual( len( foo.attributes ), 1 )
		x = foo.attributes[0]
		self.assertEqual( x.stem, 'x' )
		self.assertIsNone( x.type ) # deferred
		self.assertIsNotNone( x.resolve )

		bar = foo.get_local( 'bar' )
		self.assertIsInstance( bar, Function )
		self.assertIsNone( bar.parameters ) # deferred
		self.assertIsNotNone( bar.resolve )
		self.assertIn( bar, foo.methods )

	def test_cstruct_and_cunion_with_generics( self ) -> None:
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
		self.assertEqual( len( box.type_params ), 1 )
		self.assertEqual( box.type_params[0].stem, 'T' )
		self.assertIs( box.get_local( 'T' ), box.type_params[0] )

		overlap = mod.get_local( 'Overlap' )
		self.assertIsInstance( overlap, CUnion )
		self.assertEqual( overlap.type_params[0].stem, 'T' )

	def test_cenum_resolves_immediately( self ) -> None:
		# enum bodies are self-contained (integer literals / '_') so nothing
		# about them needs to stay deferred
		mod = self._import( '''
@enum( i32 )
class Color:
	Red = 0
	Green = _
	Blue = 5
	Purple = _
''' )
		color = mod.get_local( 'Color' )
		self.assertIsInstance( color, CEnum )
		self.assertEqual( color.members, { 'Red': 0, 'Green': 1, 'Blue': 5, 'Purple': 6 })

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
		self.assertEqual( [ a.stem for a in iu.attributes ], [ 'v_int', 'v_size' ])
		self.assertIsNone( iu.attributes[0].type ) # deferred

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

	def test_resolve_method_skips_self( self ) -> None:
		mod = self._import( '''
class Foo:
	def bar( self, x: i32 ) -> None:
		pass
''' )
		bar = mod.get_local( 'Foo' ).get_local( 'bar' )
		bar.resolve()
		self.assertEqual( [ p.stem for p in bar.parameters ], [ 'x' ])

	def test_resolve_classmethod_skips_cls( self ) -> None:
		mod = self._import( '''
class Foo:
	@classmethod
	def make( cls, x: i32 ) -> None:
		pass
''' )
		make = mod.get_local( 'Foo' ).get_local( 'make' )
		make.resolve()
		self.assertEqual( [ p.stem for p in make.parameters ], [ 'x' ])

	def test_resolve_attribute_annotation( self ) -> None:
		mod = self._import( '''
class Foo:
	x: i32
''' )
		x = mod.get_local( 'Foo' ).attributes[0]
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
		write = mod.get_local( 'Stdout' ).get_local( 'write' )
		self.assertIsInstance( write, Function )
		self.assertEqual( len( mod.get_local( 'Stdout' ).methods ), 1 )


class RealLibSmokeTest( unittest.TestCase ):
	''' confirms discovery no longer crashes on the actual example library, not just synthetic snippets '''

	def setUp( self ) -> None:
		self.discovery = discovery.Discovery( import_builtins = True )
		self.builtins_mod = self.discovery.modules['builtins']

	def test_result_plain_method_resolves( self ) -> None:
		result_cls = self.builtins_mod.get_local( 'Result' )
		self.assertIsInstance( result_cls, CStruct )

		ok = result_cls.get_local( 'Ok' )
		self.assertIsInstance( ok, Function )
		ok.resolve()
		self.assertIsNone( ok.resolve )
		self.assertIsInstance( ok.return_type, Specialization )
		self.assertIs( ok.return_type.base, result_cls )

	def test_result_unwrap_or_overload_resolves( self ) -> None:
		result_cls = self.builtins_mod.get_local( 'Result' )
		group = result_cls.get_local( 'unwrap_or' )
		self.assertIsInstance( group, Overload )
		for fn in ( *group.stubs, *group.implementations ):
			fn.resolve()

	def test_str_from_cstr_overload_resolves( self ) -> None:
		str_cls = self.builtins_mod.get_local( 'str' )
		self.assertIsInstance( str_cls, RCClass )
		group = str_cls.get_local( 'from_cstr' )
		self.assertIsInstance( group, Overload )
		self.assertEqual( len( group.implementations ), 2 )
		# only resolving implementations[0] (buf/length) here: implementations[1]
		# takes `src: move[bytearray]` - `move[T]` as a subscriptable ownership
		# annotation isn't a type the discovery type system models yet, and
		# that's out of scope for this pass
		group.implementations[0].resolve()
		self.assertIsNone( group.implementations[0].resolve )

	def test_bytearray_resolves( self ) -> None:
		ba_cls = self.builtins_mod.get_local( 'bytearray' )
		self.assertIsInstance( ba_cls, RCClass )
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


if __name__ == '__main__':
	logging.basicConfig( level = logging.DEBUG, force = True )
	unittest.main()
