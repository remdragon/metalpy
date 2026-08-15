# stdlib imports
import logging
from pathlib import Path
import tempfile
import unittest

# local imports
import discovery
from errors import CompileError
import linker_c
import test_support
from mpy_types import (
	Module, RCClass, CStruct, CUnion, CEnum, TaggedUnion, Overload,
	Function, Variable, Specialization, Move, Copy, ConditionalDispatch, Scalar,
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


class ImportFileEncodingTests( unittest.TestCase ):
	''' regression test: import_file() used to open() source files with no
	explicit encoding, which defaults to the OS locale encoding - on
	Windows that's the system ANSI codepage (e.g. CP1252), not UTF-8. A
	non-ASCII string literal in a real .py file (not passed as an in-
	memory str via import_code, which never touches this path at all)
	would get silently decoded wrong here, then re-encoded as corrupted
	UTF-8 by emitter_c.py's own string literal emission - invisible on
	Linux/macOS, where the locale encoding is already UTF-8. '''

	def test_non_ascii_source_file_reads_as_real_utf8( self ) -> None:
		import tempfile
		with tempfile.TemporaryDirectory() as tmp:
			tmp_path = Path( tmp )
			( tmp_path / '__main__.py' ).write_text(
				"GREETING: str = 'straße café'\n", encoding = 'utf-8',
			)
			disco = discovery.Discovery( paths = [ tmp_path ], import_builtins = False )
			mod = disco.import_file( tmp_path / '__main__.py', scope = None )
			greeting = mod.get_local( 'GREETING' )
			self.assertEqual( disco.errors.errors, [] )
			# the AST constant itself, not a lowered value - this test is
			# about the SOURCE bytes surviving the read, not full string
			# lowering (see emitter_c_test.py's own real-compile coverage
			# for that)
			self.assertEqual( greeting.init.value, 'straße café' )


class UnsupportedBodyStatementTests( unittest.TestCase ):
	''' regression coverage for a real, previously-crashing gap: any
	unsupported statement at module/class scope containing a Store-context
	ast.Name (a bare `for` loop, `x += 1`) fell through to ast.NodeVisitor's
	own default generic_visit, which blindly recurses into it and hits
	visit_Name's own internal invariant assert ("Load is the only context
	an expression-position Name can have") - an uncaught AssertionError,
	not a clean, reported CompileError. Found while reviewing PLAN_GLOBAL_
	INIT.md's own "Deferred: arbitrary top-level statements" note (that
	note is about a genuinely bigger, unrelated feature - general statement
	EXECUTION at module scope, still correctly deferred, no real use case
	yet); this is a narrower robustness fix: an unsupported statement
	should always fail cleanly, regardless of whether real execution
	support is ever built. Fixed via _SUPPORTED_BODY_STATEMENTS/_check_
	supported_statement (discovery.py), called before self.visit(node) at
	both scan sites (import_code's top-level loop, _make_class_resolver's
	body_fn). '''

	def setUp( self ) -> None:
		self.discovery = discovery.Discovery( import_builtins = False )

	def _import( self, code: str ) -> Module:
		return self.discovery.import_code( code, Path( '__main__.py' ), scope = None )

	def test_for_loop_at_module_level_is_a_clean_compile_error( self ) -> None:
		# before the fix: AssertionError, not caught by import_code's own
		# per-statement `except CompileError: continue` - the module import
		# never even progressed past this line, importing nothing
		self._import( '''
for i in range( 3 ):
	pass
''' )
		self.assertEqual( len( self.discovery.errors.errors ), 1 )
		self.assertIn( 'unsupported statement', self.discovery.errors.errors[0] )

	def test_augmented_assignment_at_module_level_is_a_clean_compile_error( self ) -> None:
		self._import( '''
x: i32 = 1
x += 1
''' )
		self.assertEqual( len( self.discovery.errors.errors ), 1 )
		self.assertIn( 'unsupported statement', self.discovery.errors.errors[0] )

	def test_one_bad_top_level_statement_does_not_stop_the_rest_of_the_module( self ) -> None:
		# matches import_code's own existing per-statement recovery
		# discipline (see its docstring) - a rejected statement doesn't
		# prevent everything AROUND it from still being registered
		mod = self._import( '''
x: i32 = 1
for i in range( 3 ):
	pass
y: i32 = 2
''' )
		self.assertEqual( len( self.discovery.errors.errors ), 1 )
		self.assertIsNotNone( mod.get_local( 'x' ))
		self.assertIsNotNone( mod.get_local( 'y' ))

	def test_for_loop_in_class_body_is_a_clean_compile_error_not_a_crash( self ) -> None:
		mod = self._import( '''
class Foo:
	for i in range( 3 ):
		pass
	x: i32
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve() # class bodies are scanned lazily - see ShallowScanTests
		self.assertEqual( len( self.discovery.errors.errors ), 1 )
		self.assertIn( 'unsupported statement', self.discovery.errors.errors[0] )

	def test_pass_bodied_class_is_still_allowed( self ) -> None:
		# ast.Pass is the one statement kind deliberately allowed to reach
		# generic_visit directly (no fields to recurse into, so it's
		# provably always a no-op) - a plain `pass`-bodied class is a common,
		# legitimate pattern (see ShallowScanTests.test_rcclass) that must
		# keep working, not get swept up by this same fix
		mod = self._import( '''
class Foo:
	pass
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertIsNotNone( mod.get_local( 'Foo' ))


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

	def test_bool_is_an_intrinsic( self ) -> None:
		# not a fixed-width integer, but resolved the exact same way as
		# i32/usize/etc - available in any module with no import, since
		# ir.py's own Const.value already treats bool as a first-class
		# IR-level concept, same tier as the numeric scalars
		mod = self._import( '''
def foo( x: bool ) -> bool:
	return x
''' )
		foo = mod.get_local( 'foo' )
		foo.resolve()
		bool_cls = self.discovery.get_intrinsics()['bool']
		self.assertIs( foo.parameters[0].type, bool_cls )
		self.assertIs( foo.return_type, bool_cls )

	def test_noreturn_is_an_intrinsic_distinct_from_none( self ) -> None:
		# a distinct marker, not an alias for NoneType - functionally
		# identical to None for lowering today (no return value), but kept
		# separate so a future emitter can tell "never returns" (e.g.
		# sys.panic()) apart from "returns nothing", to emit C's own
		# _Noreturn/[[noreturn]] and avoid a false "missing return" warning
		mod = self._import( '''
def foo() -> NoReturn:
	pass
''' )
		foo = mod.get_local( 'foo' )
		foo.resolve()
		noreturn_cls = self.discovery.get_intrinsics()['NoReturn']
		none_type = self.discovery.get_none_type()
		self.assertIs( foo.return_type, noreturn_cls )
		self.assertIsNot( noreturn_cls, none_type )

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

	def test_chain_lookup_finds_inherited_attribute_and_method( self ) -> None:
		# RCClass.chain_lookup (generalized from CStruct's own, see
		# mpy_types.py's shared free functions) - a subclass's own .names
		# never gets an inherited member merged into it directly; chain_
		# lookup is what walks up .base to find one. Phase 1 of the RCClass-
		# subclassing plan (base-chain lookup + attribute shadowing).
		mod = self._import( '''
class Base:
	x: i32
	def hello( self ) -> i32:
		return 42

class Derived( Base ):
	y: i32
''' )
		derived = mod.get_local( 'Derived' )
		derived.resolve()
		self.assertEqual( derived.chain_lookup( 'y' ).qualname, '__main__.Derived.y' )
		self.assertEqual( derived.chain_lookup( 'x' ).qualname, '__main__.Base.x' )
		self.assertEqual( derived.chain_lookup( 'hello' ).qualname, '__main__.Base.hello' )
		self.assertIsNone( derived.chain_lookup( 'nonexistent' ))

	def test_subclass_shadowing_base_field_errors( self ) -> None:
		mod = self._import( '''
class Base:
	x: i32

class Derived( Base ):
	x: i32
''' )
		derived = mod.get_local( 'Derived' )
		derived.resolve()
		self.assertIn( 'shadows', self.discovery.errors.errors[0] )
		self.assertIn( 'Derived.x', self.discovery.errors.errors[0] )
		self.assertIn( 'Base.x', self.discovery.errors.errors[0] )

	def test_subclass_shadowing_base_method_errors( self ) -> None:
		mod = self._import( '''
class Base:
	def get_x( self ) -> i32:
		return 1

class Derived( Base ):
	def get_x( self ) -> i32:
		return 2
''' )
		derived = mod.get_local( 'Derived' )
		derived.resolve()
		self.assertIn( 'shadows', self.discovery.errors.errors[0] )

	def test_subclass_own_init_is_not_shadowing( self ) -> None:
		# __init__ is exempted - a subclass declaring its own __init__ is
		# the ordinary, expected constructor-chaining case (super().
		# __init__() support is a later phase), not shadowing
		mod = self._import( '''
class Base:
	x: i32
	def __init__( self, x: i32 ) -> None:
		self.x = x

class Derived( Base ):
	y: i32
	def __init__( self, x: i32, y: i32 ) -> None:
		self.x = x
		self.y = y
''' )
		derived = mod.get_local( 'Derived' )
		derived.resolve()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_unrelated_names_do_not_shadow( self ) -> None:
		mod = self._import( '''
class Base:
	x: i32
	def hello( self ) -> i32:
		return 1

class Derived( Base ):
	y: i32
	def world( self ) -> i32:
		return 2
''' )
		derived = mod.get_local( 'Derived' )
		derived.resolve()
		self.assertEqual( self.discovery.errors.errors, [] )


class RCClassVirtualTests( unittest.TestCase ):
	''' Phase 4 of the RCClass-subclassing plan: @virtual generalized from
	@interface-CStruct-only to ordinary RCClass. Reuses CStruct's own
	vtable machinery (mpy_types.py's shared chain_lookup/virtual_slots/
	vtbl_owner, compiler.py's _validate_interface_vtable) - this class
	covers the RCClass-specific decorator validation: the widened
	@virtual gate, the new single-signature-only check (metalpy's
	implicit same-name-different-signature overloading, not just
	@overload), @virtual+@staticmethod/@classmethod rejection, and the
	Phase 1 shadowing check's new @virtual-override exemption. '''

	def setUp( self ) -> None:
		self.discovery = discovery.Discovery( import_builtins = False )

	def _import( self, code: str ) -> Module:
		return self.discovery.import_code( code, Path( '__main__.py' ), scope = None )

	def test_virtual_allowed_on_ordinary_rcclass_method( self ) -> None:
		mod = self._import( '''
class Foo:
	@virtual
	def hello( self ) -> i32:
		return 1
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		self.assertEqual( self.discovery.errors.errors, [] )
		hello = foo.chain_lookup( 'hello' )
		self.assertTrue( hello.is_virtual )

	def test_matching_virtual_override_is_not_shadowing( self ) -> None:
		mod = self._import( '''
class Base:
	@virtual
	def hello( self ) -> i32:
		return 1

class Derived( Base ):
	@virtual
	def hello( self ) -> i32:
		return 2
''' )
		derived = mod.get_local( 'Derived' )
		derived.resolve()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_override_missing_virtual_is_still_shadowing( self ) -> None:
		# a subclass overriding an inherited @virtual method MUST repeat
		# @virtual on its own re-declaration - matching an existing slot's
		# name+signature alone is not enough (explicit-at-every-
		# declaration-site, same posture as @interface not inheriting
		# implicitly)
		mod = self._import( '''
class Base:
	@virtual
	def hello( self ) -> i32:
		return 1

class Derived( Base ):
	def hello( self ) -> i32:
		return 2
''' )
		derived = mod.get_local( 'Derived' )
		derived.resolve()
		self.assertIn( 'shadows', self.discovery.errors.errors[0] )

	def test_virtual_staticmethod_rejected( self ) -> None:
		mod = self._import( '''
class Foo:
	@virtual
	@staticmethod
	def hello() -> i32:
		return 1
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		self.assertIn( 'cannot also be @staticmethod/@classmethod', self.discovery.errors.errors[0] )

	def test_virtual_classmethod_rejected( self ) -> None:
		mod = self._import( '''
class Foo:
	@virtual
	@classmethod
	def hello( cls ) -> i32:
		return 1
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		self.assertIn( 'cannot also be @staticmethod/@classmethod', self.discovery.errors.errors[0] )

	def test_inline_with_virtual_rejected( self ) -> None:
		# PLAN_INLINE.md - @inline splices the body at each call site,
		# @virtual dispatches indirectly through a vtable slot - mutually
		# exclusive, and the error should name @virtual specifically (not
		# some other decorator this combo also happens to trip)
		mod = self._import( '''
class Foo:
	@inline
	@virtual
	def hello( self ) -> i32:
		return 1
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		self.assertIn( 'cannot also be @virtual', self.discovery.errors.errors[0] )

	def test_inline_with_abstractmethod_rejected( self ) -> None:
		# @abstractmethod implies is_virtual=True internally (see the "bare
		# abstractmethod implies virtual" test above) - this must still
		# report the @abstractmethod-specific message ("no body to splice"),
		# not misattribute the conflict to @virtual, which was never
		# written here at all
		mod = self._import( '''
class Foo:
	@inline
	@abstractmethod
	def hello( self ) -> i32:
		...
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		self.assertIn( 'cannot also be @abstractmethod', self.discovery.errors.errors[0] )
		self.assertNotIn( 'cannot also be @virtual', self.discovery.errors.errors[0] )

	def test_inline_multistatement_body_accepted( self ) -> None:
		# the multi-statement generalization: locals/branches before a
		# single, final, un-nested return
		mod = self._import( '''
class Foo:
	@inline
	def hello( self, x: i32 ) -> i32:
		y: i32 = x
		if y == 0:
			y = 1
		return y
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_inline_body_with_return_nested_in_if_accepted( self ) -> None:
		# early/nested return generalization - the spliced body now has its
		# own local epilogue to jump into (see lowering.py's _splice_multi_
		# statement_inline_body/cfg.py's push_inline_scope), so a `return`
		# nested inside an if is no longer rejected outright
		mod = self._import( '''
class Foo:
	@inline
	def hello( self, x: i32 ) -> i32:
		if x == 0:
			return 0
		return x
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_inline_body_with_return_not_last_rejected( self ) -> None:
		# unchanged: the body must still structurally END in a `return
		# <expr>` - a return followed by dead-but-still-textually-present
		# code stays rejected, only the ERROR MESSAGE changed to reflect
		# that earlier returns are now otherwise allowed
		mod = self._import( '''
class Foo:
	@inline
	def hello( self, x: i32 ) -> i32:
		return x
		y: i32 = 1
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		self.assertIn( 'must have a body ending in exactly one `return <expr>`', self.discovery.errors.errors[0] )

	def test_inline_body_with_bare_return_rejected( self ) -> None:
		mod = self._import( '''
class Foo:
	@inline
	def hello( self ) -> i32:
		y: i32 = 1
		return
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		self.assertIn( 'must have a body ending in exactly one `return <expr>`', self.discovery.errors.errors[0] )

	def test_inline_body_with_bare_early_return_rejected( self ) -> None:
		# every reachable return needs a value, not just the trailing one -
		# an early bare `return` has no well-defined meaning for an inline
		# function's own overall value
		mod = self._import( '''
class Foo:
	@inline
	def hello( self, x: i32 ) -> i32:
		if x == 0:
			return
		return x
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		self.assertIn( 'must have a body ending in exactly one `return <expr>`', self.discovery.errors.errors[0] )

	def test_inline_body_with_defer_accepted( self ) -> None:
		# defer/errdefer generalization - the spliced body now has a
		# well-defined local boundary of its own to run against (see
		# lowering.py's _splice_multi_statement_inline_body), so it's no
		# longer rejected at parse time. resolve() alone doesn't reach
		# lowering/splicing (that only happens at an actual call site), so
		# this only confirms the DISCOVERY-time rejection is gone
		mod = self._import( '''
class Foo:
	@inline
	def hello( self, x: i32 ) -> i32:
		with defer:
			pass
		return x
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_inline_body_with_errdefer_call_form_accepted( self ) -> None:
		# the OTHER recognized spelling, `errdefer(...)` as a bare call
		# statement, not just `with defer:`
		mod = self._import( '''
class Foo:
	@inline
	def hello( self, x: i32 ) -> i32:
		errdefer( x )
		return x
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_inline_body_with_defer_nested_in_if_accepted( self ) -> None:
		mod = self._import( '''
class Foo:
	@inline
	def hello( self, x: i32 ) -> i32:
		if x == 0:
			with defer:
				pass
		return x
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_inline_body_reassigning_self_rejected( self ) -> None:
		mod = self._import( '''
class Foo:
	@inline
	def hello( self ) -> i32:
		self = self
		return 1
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		self.assertIn( 'reassigning self/a parameter', self.discovery.errors.errors[0] )

	def test_inline_body_reassigning_parameter_rejected( self ) -> None:
		mod = self._import( '''
class Foo:
	@inline
	def hello( self, x: i32 ) -> i32:
		x = x + 1
		return x
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		self.assertIn( 'reassigning self/a parameter', self.discovery.errors.errors[0] )

	def test_inline_body_reassigning_parameter_nested_in_if_rejected( self ) -> None:
		mod = self._import( '''
class Foo:
	@inline
	def hello( self, x: i32 ) -> i32:
		if x == 0:
			x = 1
		return x
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		self.assertIn( 'reassigning self/a parameter', self.discovery.errors.errors[0] )

	def test_inline_body_reassigning_own_local_accepted( self ) -> None:
		# unlike self/a parameter, reassigning a local the BODY ITSELF
		# declared is fine - only self/params are restricted (they might
		# alias the caller's own argument; a fresh local never does)
		mod = self._import( '''
class Foo:
	@inline
	def hello( self, x: i32 ) -> i32:
		y: i32 = x
		y = y + 1
		return y
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_virtual_with_second_plain_signature_is_a_compile_error( self ) -> None:
		# NOT just the already-rejected @virtual+@overload-on-the-SAME-def
		# combo - metalpy also allows multiple PLAIN (non-@overload) defs
		# sharing a name with different signatures, silently forming an
		# Overload group with no @overload in sight. A @virtual method
		# must have exactly one signature regardless of how the extra
		# ones are spelled.
		mod = self._import( '''
class Foo:
	@virtual
	def hello( self ) -> i32:
		return 1
	def hello( self, x: i32 ) -> i32:
		return x
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		# the check lives inside the METHOD's own resolver (it's what
		# guarantees a virtual method that's part of a dead-code Overload
		# group still gets checked - see compiler.py's own eager-resolve
		# loop in _validate_interface_vtable) - at the bare discovery
		# level (no full Compiler pipeline), nothing else forces that, so
		# resolve the group's own members directly, same as
		# DeferredResolutionTests' own explicit .resolve() pattern
		group = foo.names['hello']
		for fn in ( *group.stubs, *group.implementations ):
			if fn.resolve is not None:
				fn.resolve()
		self.assertIn( 'must have exactly one signature', self.discovery.errors.errors[0] )

	def test_virtual_with_second_plain_signature_declared_first( self ) -> None:
		# fires regardless of declaration order - the @virtual member can
		# be either the first or second def sharing the name
		mod = self._import( '''
class Foo:
	def hello( self, x: i32 ) -> i32:
		return x
	@virtual
	def hello( self ) -> i32:
		return 1
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		group = foo.names['hello']
		for fn in ( *group.stubs, *group.implementations ):
			if fn.resolve is not None:
				fn.resolve()
		self.assertIn( 'must have exactly one signature', self.discovery.errors.errors[0] )

	def test_virtual_single_signature_via_overload_group_still_only_needs_one( self ) -> None:
		# a genuinely unique name (no sibling at all) never forms an
		# Overload group in the first place - the common case stays a
		# no-op for this check
		mod = self._import( '''
class Foo:
	@virtual
	def hello( self ) -> i32:
		return 1
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		self.assertEqual( self.discovery.errors.errors, [] )


class RCClassAbstractTests( unittest.TestCase ):
	''' Phase 5 of the RCClass-subclassing plan: @abstractmethod for
	RCClass. Unlike CStruct (whose "abstract-ness" is purely structural -
	a stub-bodied @virtual method IS the unimplemented declaration, no
	separate marker needed), RCClass uses the EXPLICIT is_abstract flag -
	@abstractmethod IMPLIES @virtual (there's no other coherent meaning
	for an unimplemented method to have, unlike @interface's own non-
	inheritance or a @virtual override having to repeat @virtual, both of
	which are genuinely ambiguous without being explicit) and must have a
	stub body (mirroring @extern's own identical stub-body requirement).
	Writing @virtual alongside @abstractmethod is still accepted (harmless
	redundancy, not contradictory). Construction-time enforcement
	(rejecting Foo(...) when any slot in the chain is unfulfilled) lives
	in lowering.py/emitter_c_test.py, not here - this class covers just
	the decorator-combination validation. '''

	def setUp( self ) -> None:
		self.discovery = discovery.Discovery( import_builtins = False )

	def _import( self, code: str ) -> Module:
		return self.discovery.import_code( code, Path( '__main__.py' ), scope = None )

	def test_bare_abstractmethod_implies_virtual( self ) -> None:
		mod = self._import( '''
class Foo:
	@abstractmethod
	def hello( self ) -> i32: ...
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		self.assertEqual( self.discovery.errors.errors, [] )
		hello = foo.chain_lookup( 'hello' )
		self.assertTrue( hello.is_virtual )
		self.assertTrue( hello.is_abstract )

	def test_abstractmethod_requires_stub_body( self ) -> None:
		mod = self._import( '''
class Foo:
	@abstractmethod
	def hello( self ) -> i32:
		return 1
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		self.assertIn( 'must have a stub body', self.discovery.errors.errors[0] )

	def test_abstractmethod_with_redundant_explicit_virtual_is_accepted( self ) -> None:
		mod = self._import( '''
class Foo:
	@abstractmethod
	@virtual
	def hello( self ) -> i32: ...
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		self.assertEqual( self.discovery.errors.errors, [] )
		hello = foo.chain_lookup( 'hello' )
		self.assertTrue( hello.is_virtual )
		self.assertTrue( hello.is_abstract )
		hello = foo.chain_lookup( 'hello' )
		self.assertTrue( hello.is_virtual )
		self.assertTrue( hello.is_abstract )


class InterfaceCStructTests( unittest.TestCase ):
	''' @interface CStructs - see PLAN_SUBCLASSING_VTABLES_COM.md. Single
	inheritance, @interface-ness NOT inherited implicitly (a subclass must
	redeclare it), @virtual only meaningful on an @interface CStruct right
	now (RCClass vtable support is deferred). '''

	def setUp( self ) -> None:
		self.discovery = discovery.Discovery( import_builtins = False )

	def _import( self, code: str ) -> Module:
		return self.discovery.import_code( code, Path( '__main__.py' ), scope = None )

	def test_root_interface_is_a_cstruct_marked_is_interface( self ) -> None:
		mod = self._import( '''
@interface
class IFoo:
	def helper( self ) -> i32: ...
''' )
		ifoo = mod.get_local( 'IFoo' )
		self.assertIsInstance( ifoo, CStruct )
		self.assertTrue( ifoo.is_interface )
		self.assertIsNone( ifoo.base )

	def test_subclass_base_resolved_eagerly( self ) -> None:
		mod = self._import( '''
@interface
class IFoo:
	def helper( self ) -> i32: ...

@interface
class FooImpl( IFoo ):
	x: i32
''' )
		ifoo = mod.get_local( 'IFoo' )
		fooimpl = mod.get_local( 'FooImpl' )
		self.assertIs( fooimpl.base, ifoo )

	def test_plain_cstruct_cannot_have_a_base( self ) -> None:
		self._import( '''
@interface
class IFoo:
	def helper( self ) -> i32: ...

@cstruct
class Bad( IFoo ):
	pass
''' )
		self.assertIn( 'cannot have a base', self.discovery.errors.errors[0] )

	def test_subclass_must_redeclare_interface( self ) -> None:
		# @interface-ness is NOT inherited implicitly - deliberately
		# conservative, see the plan doc's "Subclassing mechanics"
		self._import( '''
@interface
class IFoo:
	def helper( self ) -> i32: ...

class Bad( IFoo ):
	def helper( self ) -> i32:
		return 1
''' )
		self.assertIn( 'cannot subclass', self.discovery.errors.errors[0] )

	def test_interface_cannot_subclass_plain_cstruct( self ) -> None:
		self._import( '''
@cstruct
class Point:
	x: i32

@interface
class Bad( Point ):
	pass
''' )
		self.assertIn( 'cannot subclass', self.discovery.errors.errors[0] )

	def test_interface_multiple_inheritance_errors( self ) -> None:
		self._import( '''
@interface
class A:
	pass

@interface
class B:
	pass

@interface
class C( A, B ):
	pass
''' )
		self.assertIn( 'multiple inheritance', self.discovery.errors.errors[0] )

	def test_virtual_allowed_on_interface_method( self ) -> None:
		mod = self._import( '''
@interface
class IFoo:
	@virtual
	def helper( self ) -> i32: ...
''' )
		ifoo = mod.get_local( 'IFoo' )
		ifoo.resolve()
		helper = ifoo.get_local( 'helper' )
		self.assertTrue( helper.is_virtual )

	def test_virtual_rejected_on_plain_cstruct( self ) -> None:
		mod = self._import( '''
@cstruct
class Foo:
	@virtual
	def helper( self ) -> i32:
		return 1
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		self.assertIn( 'only supported on @interface classes', self.discovery.errors.errors[0] )

	def test_bare_interface_type_rejected_as_parameter( self ) -> None:
		# an @interface CStruct is never a plain value type - self is
		# always Ptr[T] (see PLAN_SUBCLASSING_VTABLES_COM.md's REVISION).
		# Construction never produces a bare value, but a bare-typed
		# annotation was still syntactically legal and reachable through
		# Ptr[T]'s own [0] escape hatch - confirmed to crash the compiler
		# outright rather than fail gracefully, before this check existed.
		mod = self._import( '''
@interface
class IFoo:
	@virtual
	def get_value( self ) -> i32: ...

def call_through_base( f: IFoo ) -> i32:
	return f.get_value()
''' )
		fn = mod.get_local( 'call_through_base' )
		fn.resolve()
		self.assertIn( 'can never be a plain value type', self.discovery.errors.errors[0] )

	def test_bare_interface_type_rejected_as_return_type( self ) -> None:
		mod = self._import( '''
@interface
class IFoo:
	@virtual
	def get_value( self ) -> i32: ...

def make_foo() -> IFoo: ...
''' )
		fn = mod.get_local( 'make_foo' )
		fn.resolve()
		self.assertIn( 'can never be a plain value type', self.discovery.errors.errors[0] )

	def test_bare_interface_type_rejected_as_variable_annotation( self ) -> None:
		mod = self._import( '''
@interface
class IFoo:
	@virtual
	def get_value( self ) -> i32: ...

x: IFoo = None
''' )
		x = mod.get_local( 'x' )
		x.resolve()
		self.assertIn( 'can never be a plain value type', self.discovery.errors.errors[0] )

	def test_ptr_interface_type_still_allowed( self ) -> None:
		mod = self._import( '''
@interface
class IFoo:
	@virtual
	def get_value( self ) -> i32: ...

def make_foo() -> Ptr[IFoo]: ...
y: Ptr[IFoo] = None
''' )
		mod.get_local( 'make_foo' ).resolve()
		mod.get_local( 'y' ).resolve()
		self.assertEqual( self.discovery.errors.errors, [] )


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

class CopyTypeTests( unittest.TestCase ):
	''' copy[T] in annotation position - recognized textually the same way move[T] is, no call-site marker involved (unlike move[T]) '''

	def setUp( self ) -> None:
		self.discovery = discovery.Discovery( import_builtins = False )

	def _import( self, code: str ) -> Module:
		return self.discovery.import_code( code, Path( '__main__.py' ), scope = None )

	def test_copy_wraps_inner_type( self ) -> None:
		mod = self._import( '''
class Foo:
	pass

def consume( x: copy[Foo] ) -> None:
	pass
''' )
		fn = mod.get_local( 'consume' )
		fn.resolve()
		p = fn.parameters[0]
		self.assertIsInstance( p.type, Copy )
		self.assertIs( p.type.inner, mod.get_local( 'Foo' ))

	def test_copy_dedups_to_identical_object( self ) -> None:
		mod = self._import( '''
class Foo:
	pass

def consume( x: copy[Foo] ) -> None:
	pass

def consume2( y: copy[Foo] ) -> None:
	pass
''' )
		consume = mod.get_local( 'consume' )
		consume2 = mod.get_local( 'consume2' )
		consume.resolve()
		consume2.resolve()
		self.assertIs( consume.parameters[0].type, consume2.parameters[0].type )

	def test_copy_multiple_args_errors( self ) -> None:
		mod = self._import( '''
class Foo:
	pass

class Bar:
	pass

def consume( x: copy[Foo, Bar] ) -> None:
	pass
''' )
		fn = mod.get_local( 'consume' )
		fn.resolve()
		self.assertIn( 'exactly one type argument', self.discovery.errors.errors[0] )

	def test_copy_and_move_are_distinct_types( self ) -> None:
		mod = self._import( '''
class Foo:
	pass

def consume_copy( x: copy[Foo] ) -> None:
	pass

def consume_move( x: move[Foo] ) -> None:
	pass
''' )
		consume_copy = mod.get_local( 'consume_copy' )
		consume_move = mod.get_local( 'consume_move' )
		consume_copy.resolve()
		consume_move.resolve()
		self.assertIsInstance( consume_copy.parameters[0].type, Copy )
		self.assertIsInstance( consume_move.parameters[0].type, Move )

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


class OrReturnReservedNameTests( unittest.TestCase ):
	''' 'or_return' is reserved for the compiler's own Result[T,E].or_return()
	- <result_expr>.or_return() is recognized purely by AST shape (lowering.
	py's _lower_call, before ordinary call resolution ever runs), never by
	looking up a real declared method the way is_ok()/is_err()/unwrap()/
	unwrap_or() genuinely are - a user-written `def or_return(...)` could
	never actually run, at any receiver type, so it's rejected outright here
	rather than silently accepted as dead code '''

	def setUp( self ) -> None:
		self.discovery = discovery.Discovery( import_builtins = False )

	def _import( self, code: str ) -> Module:
		return self.discovery.import_code( code, Path( '__main__.py' ), scope = None )

	def test_plain_function_named_or_return_is_rejected( self ) -> None:
		self._import( '''
def or_return() -> i32:
	return 1
''' )
		self.assertIn( "'or_return' is reserved", self.discovery.errors.errors[0] )

	def test_method_named_or_return_is_rejected( self ) -> None:
		mod = self._import( '''
class Foo:
	def or_return( self ) -> i32:
		return 1
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		self.assertIn( "'or_return' is reserved", self.discovery.errors.errors[0] )


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

	def test_matching_target_class_included( self ) -> None:
		disco, mod = self._import( '''
@compiler.target( os = 'windows' )
class Foo:
	a: int
''', { 'os': 'windows' })
		self.assertIsInstance( mod.get_local( 'Foo' ), RCClass )

	def test_non_matching_target_class_excluded( self ) -> None:
		disco, mod = self._import( '''
@compiler.target( os = 'windows' )
class Foo:
	a: int
''', { 'os': 'linux' })
		self.assertIsNone( mod.get_local( 'Foo' ))

	def test_same_named_classes_pick_one_winner( self ) -> None:
		disco, mod = self._import( '''
@compiler.target( os = 'windows' )
class Foo:
	a: int

@compiler.target( os = not 'windows' )
class Foo:
	b: int
''', { 'os': 'linux' })
		foo = mod.get_local( 'Foo' )
		self.assertIsInstance( foo, RCClass )
		foo.resolve()
		self.assertIsNotNone( foo.get_local( 'b' ))
		self.assertIsNone( foo.get_local( 'a' ))

	def test_matching_target_cstruct_included( self ) -> None:
		# the @compiler.target(...) filtering check has to run before
		# visit_ClassDef dispatches on the OTHER decorators (@cstruct/
		# @cunion/@enum/@union) - order in decorator_list shouldn't matter
		disco, mod = self._import( '''
@cstruct
@compiler.target( os = 'windows' )
class Foo:
	a: int
''', { 'os': 'windows' })
		self.assertIsInstance( mod.get_local( 'Foo' ), CStruct )

	def test_bits_int_value_matches( self ) -> None:
		# regression: _target_value_matches used to only accept string
		# constants, so an int literal like `bits = 64` (matching SYNTAX.md's
		# BitsSpec = Literal[16,32,64]) never matched anything
		disco, mod = self._import( '''
@compiler.target( bits = 64 )
def foo() -> i32:
	pass
''', { 'bits': 64 })
		self.assertIsInstance( mod.get_local( 'foo' ), Function )

	def test_posix_bool_value_matches( self ) -> None:
		disco, mod = self._import( '''
@compiler.target( posix = True )
def foo() -> i32:
	pass
''', { 'posix': True })
		self.assertIsInstance( mod.get_local( 'foo' ), Function )

	def test_debug_key_modeled_by_default( self ) -> None:
		disco = discovery.Discovery( import_builtins = False )
		self.assertIn( 'debug', disco.active_target )
		self.assertIs( disco.active_target['debug'], True )

	def test_family_is_unix_not_posix( self ) -> None:
		# family is one of SYNTAX.md's FamilySpec literals ('unix'/'windows'/
		# 'wasm') - 'posix' is a separate bool field, not a family value
		disco = discovery.Discovery( import_builtins = False )
		self.assertIn( disco.active_target['family'], ( 'unix', 'windows' ))


_CC = linker_c.detect_cc()

@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found' )
class CompilerHasLibraryTargetTests( unittest.TestCase ):
	''' @compiler.target(has_library=(lib, symbol)) - eager decorator-level
	filtering, backed by a real compile+link probe (see linker_c.has_symbol) -
	mirrors CompilerTargetTests' own os=/arch= style, but has_library's
	value is the PROBE's own (lib, symbol) arguments, not a lookup against
	active_target (see Discovery._matches_has_library's own comment on why
	it's special-cased ahead of the generic dict-lookup path). '''

	def _import( self, code: str ) -> tuple[discovery.Discovery, Module]:
		disco = discovery.Discovery( import_builtins = False )
		mod = disco.import_code( code, Path( '__main__.py' ), scope = None )
		return disco, mod

	def test_available_symbol_included( self ) -> None:
		disco, mod = self._import( f'''
@compiler.target( has_library = ( '{test_support.KNOWN_LIB}', '{test_support.KNOWN_SYMBOL}' ))
def foo() -> i32:
	pass
''' )
		self.assertIsInstance( mod.get_local( 'foo' ), Function )

	def test_unavailable_symbol_excluded( self ) -> None:
		disco, mod = self._import( '''
@compiler.target( has_library = ( 'kernel32', 'ThisIsNotARealSymbol123' ))
def foo() -> i32:
	pass
''' )
		self.assertIsNone( mod.get_local( 'foo' ))

	def test_negated_has_library( self ) -> None:
		disco, mod = self._import( '''
@compiler.target( has_library = not ( 'kernel32', 'ThisIsNotARealSymbol123' ))
def foo() -> i32:
	pass
''' )
		self.assertIsInstance( mod.get_local( 'foo' ), Function )

	def test_same_named_functions_pick_the_available_one( self ) -> None:
		# mirrors CompilerTargetTests' own os= version of this same shape -
		# two mutually-exclusive has_library-gated defs, only the matching
		# one should ever get registered
		disco, mod = self._import( f'''
@compiler.target( has_library = ( '{test_support.KNOWN_LIB}', '{test_support.KNOWN_SYMBOL}' ))
def get_error() -> i32:
	return 1

@compiler.target( has_library = not ( '{test_support.KNOWN_LIB}', '{test_support.KNOWN_SYMBOL}' ))
def get_error() -> i32:
	return 2
''' )
		fn = mod.get_local( 'get_error' )
		self.assertIsInstance( fn, Function )
		self.assertNotIsInstance( fn, Overload )
		self.assertEqual( fn.node.body[0].value.value, 1 )

	def test_malformed_value_is_a_compile_error( self ) -> None:
		disco, mod = self._import( '''
@compiler.target( has_library = 'kernel32' )
def foo() -> i32:
	pass
''' )
		self.assertTrue( any( 'has_library' in e for e in disco.errors.errors ))


class CompileTimeFoldingIntegrationTests( unittest.TestCase ):
	''' confirms discovery.py actually invokes compile_time_transformer at
	function-resolve time (see _make_function_resolver), not just that the
	transformer works in isolation (compile_time_transformer_test.py) '''

	def test_function_body_folded_on_resolve( self ) -> None:
		disco = discovery.Discovery( import_builtins = False, active_target = { 'os': 'windows' } )
		mod = disco.import_code( '''
def foo() -> None:
	if compiler.target.os == 'windows':
		a = 1
	else:
		a = 2
''', Path( '__main__.py' ), scope = None )
		fn = mod.get_local( 'foo' )
		fn.resolve()
		import ast
		self.assertEqual( len( fn.node.body ), 1 )
		self.assertEqual( ast.unparse( fn.node.body[0] ), 'a = 1' )

	def test_annassign_global_initializer_folded_eagerly( self ) -> None:
		# a global's own initializer is a bare expression, never otherwise
		# passed through compile_time_transformer at all (unlike a function
		# body) - visit_AnnAssign folds it directly, matching the real
		# lib/windows/kernel32.py case (STD_ERROR_HANDLE: u32 = u32(-12))
		disco = discovery.Discovery( import_builtins = False )
		mod = disco.import_code( 'X: i32 = 1 + 1\n', Path( '__main__.py' ), scope = None )
		import ast
		self.assertEqual( ast.unparse( mod.get_local( 'X' ).init ), '2' )

	def test_assign_global_initializer_folded_eagerly( self ) -> None:
		# same, for the un-annotated Assign form (visit_Assign)
		disco = discovery.Discovery( import_builtins = False )
		mod = disco.import_code( 'X = 1 + 1\n', Path( '__main__.py' ), scope = None )
		import ast
		self.assertEqual( ast.unparse( mod.get_local( 'X' ).init ), '2' )

	def test_class_attribute_default_folded_eagerly( self ) -> None:
		# visit_AnnAssign is shared by module globals AND class-body
		# attribute defaults - both benefit from the same fix
		disco = discovery.Discovery( import_builtins = False )
		mod = disco.import_code( '''
class Foo:
	a: i32 = 1 + 1
''', Path( '__main__.py' ), scope = None )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		import ast
		self.assertEqual( ast.unparse( foo.get_local( 'a' ).init ), '2' )


class ExternDecoratorTests( unittest.TestCase ):
	def _import( self, code: str ) -> tuple[discovery.Discovery, Module]:
		disco = discovery.Discovery( import_builtins = False )
		mod = disco.import_code( code, Path( '__main__.py' ), scope = None )
		return disco, mod

	def test_extern_lib_and_symbol_recorded( self ) -> None:
		disco, mod = self._import( '''
@extern( 'c', 'malloc' )
def malloc( size: usize ) -> Ptr[u8]:
	...
''' )
		fn = mod.get_local( 'malloc' )
		fn.resolve()
		self.assertEqual( fn.extern_lib, 'c' )
		self.assertEqual( fn.extern_symbol, 'malloc' )

	def test_ordinary_function_has_no_extern_fields( self ) -> None:
		disco, mod = self._import( '''
def foo() -> None:
	pass
''' )
		fn = mod.get_local( 'foo' )
		self.assertIsNone( fn.extern_lib )
		self.assertIsNone( fn.extern_symbol )

	def test_non_stub_body_is_a_compile_error( self ) -> None:
		disco, mod = self._import( '''
@extern( 'c', 'malloc' )
def malloc( size: usize ) -> Ptr[u8]:
	return None
''' )
		self.assertTrue( any( 'must have a stub body' in e for e in disco.errors.errors ))

	def test_wrong_arg_count_is_a_compile_error( self ) -> None:
		disco, mod = self._import( '''
@extern( 'c' )
def malloc( size: usize ) -> Ptr[u8]:
	...
''' )
		self.assertTrue( any( 'requires 2 or 3 positional arguments' in e for e in disco.errors.errors ))

	def test_non_string_arg_is_a_compile_error( self ) -> None:
		disco, mod = self._import( '''
@extern( 'c', 123 )
def malloc( size: usize ) -> Ptr[u8]:
	...
''' )
		self.assertTrue( any( 'symbol name must be a string literal' in e for e in disco.errors.errors ))

	def test_combines_with_compiler_target_regardless_of_order( self ) -> None:
		disco = discovery.Discovery( import_builtins = False, active_target = { 'os': 'windows' } )
		mod = disco.import_code( '''
@compiler.target( os = 'windows' )
@extern( 'kernel32', 'HeapAlloc' )
def HeapAlloc() -> Ptr[u8]:
	...

@extern( 'ntdll', 'RtlAllocateHeap' )
@compiler.target( os = not 'windows' )
def RtlAllocateHeap() -> Ptr[u8]:
	...
''', Path( '__main__.py' ), scope = None )
		heap_alloc = mod.get_local( 'HeapAlloc' )
		self.assertIsInstance( heap_alloc, Function )
		heap_alloc.resolve()
		self.assertEqual( heap_alloc.extern_lib, 'kernel32' )
		self.assertIsNone( mod.get_local( 'RtlAllocateHeap' )) # excluded by @compiler.target( os = not 'windows' )


class TypeAliasTests( unittest.TestCase ):
	def _import( self, code: str ) -> tuple[discovery.Discovery, Module]:
		disco = discovery.Discovery( import_builtins = False )
		mod = disco.import_code( code, Path( '__main__.py' ), scope = None )
		return disco, mod

	def test_alias_resolves_to_the_same_type_object( self ) -> None:
		disco, mod = self._import( '''
HANDLE: TypeAlias = Ptr[None]

def foo( h: HANDLE ) -> HANDLE:
	return h
''' )
		fn = mod.get_local( 'foo' )
		fn.resolve()
		handle = mod.get_local( 'HANDLE' )
		self.assertIs( handle, fn.parameters[0].type )
		self.assertIs( handle, fn.return_type )

	def test_alias_is_not_a_variable( self ) -> None:
		disco, mod = self._import( '''
HANDLE: TypeAlias = Ptr[None]
''' )
		self.assertNotIsInstance( mod.get_local( 'HANDLE' ), Variable )

	def test_no_value_is_a_compile_error( self ) -> None:
		# AnnAssign with no value (`HANDLE: TypeAlias` alone) is legal
		# Python syntax but meaningless for an alias - there's nothing to
		# alias to
		disco, mod = self._import( 'HANDLE: TypeAlias\n' )
		self.assertTrue( any( 'needs a value' in e for e in disco.errors.errors ))

	def test_non_type_value_is_a_compile_error( self ) -> None:
		disco, mod = self._import( '''
some_var: i32 = 5
HANDLE: TypeAlias = some_var
''' )
		self.assertTrue( any( 'must be a type expression' in e for e in disco.errors.errors ))

	def test_no_import_needed( self ) -> None:
		# TypeAlias is recognized purely by AST shape (like compiler.target/
		# compiler.sizeof) - no `from typing import TypeAlias` required,
		# matching lib/windows/kernel32.py's real, import-free usage
		disco, mod = self._import( 'HANDLE: TypeAlias = Ptr[None]\n' )
		self.assertEqual( disco.errors.errors, [] )


class ScalarMethodRegistrationTests( unittest.TestCase ):
	''' `Scalar.method = some_function` - see visit_Assign - the foundation
	future scalar behavior (e.g. a non-Scalar cast source's __u32__ dunder,
	see lowering.py's _try_lower_scalar_construct_call) is meant to build on '''

	def _import( self, code: str ) -> tuple[discovery.Discovery, Module]:
		disco = discovery.Discovery( import_builtins = False )
		mod = disco.import_code( code, Path( '__main__.py' ), scope = None )
		return disco, mod

	def test_registers_into_the_shared_intrinsic( self ) -> None:
		disco, mod = self._import( '''
def my_func( x: usize ) -> u32:
	return 1

usize.__u32__ = my_func
''' )
		self.assertEqual( disco.errors.errors, [] )
		usize_cls = disco.get_intrinsics()['usize']
		registered = usize_cls.names.get( '__u32__' )
		self.assertIsInstance( registered, Function )
		self.assertEqual( registered.stem, 'my_func' )

	def test_forward_reference_ordering_fails( self ) -> None:
		# same top-to-bottom limitation as TypeAlias - the RHS function must
		# already be def'd earlier in the same file
		disco, mod = self._import( '''
usize.__u32__ = my_func

def my_func( x: usize ) -> u32:
	return 1
''' )
		self.assertTrue( any( "not defined" in e or "cannot resolve" in e for e in disco.errors.errors ))

	def test_non_scalar_attribute_target_unaffected( self ) -> None:
		disco, mod = self._import( '''
class Foo: pass
Foo.bar = 5
''' )
		self.assertTrue( any( 'unsupported Assign target' in e for e in disco.errors.errors ))

	def test_rhs_not_a_function_is_a_compile_error( self ) -> None:
		disco, mod = self._import( '''
some_var: i32 = 5
usize.__u32__ = some_var
''' )
		self.assertTrue( any( 'must assign a function' in e for e in disco.errors.errors ))


class RealLibSmokeTest( unittest.TestCase ):
	''' confirms discovery no longer crashes on the actual example library, not just synthetic snippets '''

	def setUp( self ) -> None:
		self.discovery = discovery.Discovery( import_builtins = True )
		self.builtins_mod = self.discovery.modules['builtins']

	def test_result_plain_method_resolves( self ) -> None:
		# Result is a real @union (Ok/Err are TaggedUnion members, not
		# hand-written staticmethods) - is_ok/is_err/or_return/unwrap/
		# unwrap_or are the real, plain methods to check resolve correctly
		result_cls = self.builtins_mod.get_local( 'Result' )
		self.assertIsInstance( result_cls, TaggedUnion )
		result_cls.resolve()
		self.assertEqual( [ attr.stem for attr in result_cls.attributes ], [ 'Ok', 'Err' ] )

		is_ok = result_cls.get_local( 'is_ok' )
		self.assertIsInstance( is_ok, Function )
		is_ok.resolve()
		self.assertIsNone( is_ok.resolve )
		self.assertIsInstance( is_ok.return_type, Scalar )
		self.assertEqual( is_ok.return_type.stem, 'bool' )

	def test_len_is_a_single_generic_function( self ) -> None:
		# len() used to be 3 concrete overloads (str/bytes/bytearray) -
		# migrated to the single generic def len[T](t: T) once bare-call
		# monomorphization could infer T (see lowering.py's
		# _lower_inferred_generic_call)
		fn = self.builtins_mod.get_local( 'len' )
		self.assertIsInstance( fn, Function )
		if fn.resolve is not None:
			fn.resolve()
		self.assertEqual( len( fn.type_params ), 1 )
		self.assertEqual( fn.type_params[0].stem, 'T' )
		self.assertIs( fn.parameters[0].type, fn.type_params[0] )

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


class EagerReturnInferableBodyTests( unittest.TestCase ):
	''' PLAN_RETURN_INFERENCE.md - Discovery._is_eager_return_inferable_body,
	tested directly against a bare function body (no compile pipeline
	needed - this is a pure AST-shape check). '''

	def setUp( self ) -> None:
		self.discovery = discovery.Discovery( import_builtins = False )

	def _body( self, code: str ) -> list:
		import ast
		module = ast.parse( code )
		fn = module.body[0]
		self.assertIsInstance( fn, ast.FunctionDef )
		return fn.body

	def test_single_top_level_return_accepted( self ) -> None:
		body = self._body( '''
def foo():
	return x
''' )
		self.assertTrue( self.discovery._is_eager_return_inferable_body( body ))

	def test_single_return_nested_in_if_accepted( self ) -> None:
		body = self._body( '''
def foo():
	y = 1
	if y == 1:
		return x
''' )
		self.assertTrue( self.discovery._is_eager_return_inferable_body( body ))

	def test_single_return_nested_in_for_while_with_try_match_accepted( self ) -> None:
		for snippet in (
			'''
def foo():
	for i in y:
		return x
''',
			'''
def foo():
	while y:
		return x
''',
			'''
def foo():
	with y:
		return x
''',
			'''
def foo():
	try:
		return x
	except Exception:
		pass
''',
			'''
def foo():
	match y:
		case 1:
			return x
''',
		):
			with self.subTest( snippet = snippet ):
				self.assertTrue( self.discovery._is_eager_return_inferable_body( self._body( snippet )))

	def test_zero_returns_rejected( self ) -> None:
		body = self._body( '''
def foo():
	y = 1
''' )
		self.assertFalse( self.discovery._is_eager_return_inferable_body( body ))

	def test_two_returns_rejected( self ) -> None:
		body = self._body( '''
def foo():
	if y:
		return x
	return z
''' )
		self.assertFalse( self.discovery._is_eager_return_inferable_body( body ))

	def test_bare_return_rejected( self ) -> None:
		body = self._body( '''
def foo():
	return
''' )
		self.assertFalse( self.discovery._is_eager_return_inferable_body( body ))

	def test_bare_return_mixed_with_real_return_rejected( self ) -> None:
		body = self._body( '''
def foo():
	if y:
		return
	return x
''' )
		self.assertFalse( self.discovery._is_eager_return_inferable_body( body ))

	def test_return_inside_nested_def_not_counted( self ) -> None:
		# a nested def's own `return` belongs to IT, not to the enclosing
		# function - matches _reject_free_variables's identical discipline
		# (PLAN_LAMBDA.md)
		body = self._body( '''
def foo():
	def inner():
		return 1
	return x
''' )
		self.assertTrue( self.discovery._is_eager_return_inferable_body( body ))

	def test_return_inside_nested_lambda_not_counted( self ) -> None:
		body = self._body( '''
def foo():
	f = lambda: 1
	return x
''' )
		self.assertTrue( self.discovery._is_eager_return_inferable_body( body ))

	def test_only_returns_inside_nested_def_rejected( self ) -> None:
		# the OUTER function itself has zero returns of its own here - the
		# one inside `inner` doesn't count
		body = self._body( '''
def foo():
	def inner():
		return 1
''' )
		self.assertFalse( self.discovery._is_eager_return_inferable_body( body ))


if __name__ == '__main__':
	logging.basicConfig( level = logging.DEBUG, force = True )
	unittest.main()
