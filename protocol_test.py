# protocol_test.py — real compile+link+run coverage for @protocol.
#
# Structural interface contracts with EXPLICIT per-class declaration (base-
# class-list syntax: `class Bar(SomeProtocol):` - SomeProtocol is not a real
# base, no vtable/chain_lookup participation - see mpy_types.Protocol's own
# docstring and discovery.py's _validate_protocol_conformance/
# _splice_protocol_default). Conformance is checked once, at the conforming
# class's own definition; a protocol's own default (non-stub) method body
# gets spliced directly into the conforming class's dispatch table, so
# ordinary calls need no new runtime dispatch mechanism at all.

import tempfile
import unittest
from pathlib import Path

import test_support
from compiler import Compiler
from discovery import Discovery

class ProtocolCompileErrorTests( unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _run( self, code: str ) -> None:
		self.compiler.import_code( code, Path( '__main__.py' ), scope = None )
		self.compiler.run()

	def test_missing_required_method_is_a_compile_error( self ) -> None:
		self._run( '''
@protocol
class Greeter:
	def greet( self ) -> i32:
		...

class Silent( Greeter ):
	pass

def main() -> i32:
	s = Silent()
	return 0
''' )
		self.assertNotEqual( self.discovery.errors.errors, [] )
		self.assertIn( "does not implement 'greet'", str( self.discovery.errors.errors[0] ) )

	def test_ambiguous_default_from_two_protocols_is_a_compile_error( self ) -> None:
		self._run( '''
@protocol
class A:
	def foo( self ) -> i32:
		return 1

@protocol
class B:
	def foo( self ) -> i32:
		return 2

class C( A, B ):
	pass

def main() -> i32:
	c = C()
	return 0
''' )
		self.assertNotEqual( self.discovery.errors.errors, [] )
		self.assertIn( 'is ambiguous', str( self.discovery.errors.errors[0] ) )

	def test_bound_violation_bare_generic_call_is_a_compile_error( self ) -> None:
		self._run( '''
@protocol
class Greeter:
	def greet( self ) -> i32:
		...

class Talker( Greeter ):
	def greet( self ) -> i32:
		return 7

class Mute:
	pass

def call_greet[T: Greeter]( x: T ) -> i32:
	return x.greet()

def main() -> i32:
	m = Mute()
	call_greet( m )
	return 0
''' )
		self.assertNotEqual( self.discovery.errors.errors, [] )
		message = str( self.discovery.errors.errors[0] )
		self.assertIn( '__main__.Mute', message )
		self.assertIn( '__main__.Greeter', message )
		self.assertIn( 'does not implement protocol', message )

	def test_bound_violation_bare_generic_call_accepts_conforming_type( self ) -> None:
		self._run( '''
@protocol
class Greeter:
	def greet( self ) -> i32:
		...

class Talker( Greeter ):
	def greet( self ) -> i32:
		return 7

def call_greet[T: Greeter]( x: T ) -> i32:
	return x.greet()

def main() -> i32:
	t = Talker()
	call_greet( t )
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_bound_violation_explicit_subscript_call_is_a_compile_error( self ) -> None:
		self._run( '''
@protocol
class Greeter:
	def greet( self ) -> i32:
		...

class Talker( Greeter ):
	def greet( self ) -> i32:
		return 7

class Mute:
	pass

def make_greeting[T: Greeter]() -> i32:
	return 5

def main() -> i32:
	make_greeting[Mute]()
	return 0
''' )
		self.assertNotEqual( self.discovery.errors.errors, [] )
		message = str( self.discovery.errors.errors[0] )
		self.assertIn( '__main__.Mute', message )
		self.assertIn( '__main__.Greeter', message )
		self.assertIn( 'does not implement protocol', message )

	def test_bound_violation_explicit_subscript_call_accepts_conforming_type( self ) -> None:
		self._run( '''
@protocol
class Greeter:
	def greet( self ) -> i32:
		...

class Talker( Greeter ):
	def greet( self ) -> i32:
		return 7

def make_greeting[T: Greeter]() -> i32:
	return 5

def main() -> i32:
	make_greeting[Talker]()
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_bound_violation_generic_construction_is_a_compile_error( self ) -> None:
		self._run( '''
@protocol
class Greeter:
	def greet( self ) -> i32:
		...

class Talker( Greeter ):
	def greet( self ) -> i32:
		return 7

class Mute:
	pass

class Box[T: Greeter]:
	value: T
	def __init__( self, value: T ) -> None:
		self.value = value

def main() -> i32:
	m = Mute()
	b = Box( m )
	return 0
''' )
		self.assertNotEqual( self.discovery.errors.errors, [] )
		message = str( self.discovery.errors.errors[0] )
		self.assertIn( '__main__.Mute', message )
		self.assertIn( '__main__.Greeter', message )
		self.assertIn( 'does not implement protocol', message )

	def test_bound_violation_generic_construction_accepts_conforming_type( self ) -> None:
		self._run( '''
@protocol
class Greeter:
	def greet( self ) -> i32:
		...

class Talker( Greeter ):
	def greet( self ) -> i32:
		return 7

class Box[T: Greeter]:
	value: T
	def __init__( self, value: T ) -> None:
		self.value = value

def main() -> i32:
	t = Talker()
	b = Box( t )
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_protocol_base_routes_to_protocols_not_base( self ) -> None:
		# a class declaring @protocol conformance is NOT a subclass - the
		# protocol entry must land in .protocols, and the one real RCClass
		# entry (in any position among node.bases) must land in .base, with
		# no "multiple inheritance" false positive. Checked directly at the
		# discovery level (not a full compile+run) - construction through an
		# inherited (not own) __init__ exercises a separate, pre-existing,
		# unrelated compiler gap (a class with no own __init__ whose real
		# base HAS one silently constructs with uninitialized memory - see
		# memory/task tracking a dedicated fix for that), which isn't what
		# this test is about.
		self._run( '''
@protocol
class Foo:
	def foo( self ) -> None:
		pass

class Real:
	x: i32
	def __init__( self, x: i32 ) -> None:
		self.x = x

class Bar( Foo, Real ):
	pass

def main() -> i32:
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		module = self.discovery.modules['__main__']
		bar = module.names['Bar']
		real = module.names['Real']
		foo = module.names['Foo']
		self.assertIs( bar.base, real )
		self.assertEqual( bar.protocols, [ foo ] )


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
class ProtocolCompileRunTests( test_support.RealCompileMixin, unittest.TestCase ):
	def test_default_method_is_spliced_and_callable( self ) -> None:
		compiler = self._compile_source( '''
@protocol
class Greeter:
	def greet( self ) -> i32:
		return 99

class Quiet( Greeter ):
	pass

def main() -> i32:
	q = Quiet()
	if q.greet() == 99:
		return 0
	return 1
''' )
		import emitter_c
		self._assert_compiles_and_runs( emitter_c.emit_c( compiler ), expected_exit = 0, compiler = compiler )

	def test_own_method_wins_over_protocol_default( self ) -> None:
		compiler = self._compile_source( '''
@protocol
class Greeter:
	def greet( self ) -> i32:
		...

class Talker( Greeter ):
	def greet( self ) -> i32:
		return 42

def main() -> i32:
	t = Talker()
	if t.greet() == 42:
		return 0
	return 1
''' )
		import emitter_c
		self._assert_compiles_and_runs( emitter_c.emit_c( compiler ), expected_exit = 0, compiler = compiler )

	def test_default_method_body_can_call_own_class_attributes( self ) -> None:
		# the whole point of splicing (vs. some indirect/vtable dispatch): a
		# spliced default runs in the CONCRETE class's own scope, so self.x
		# resolves against Bar's own fields, not the protocol's (which has
		# none) - proves this isn't just returning a constant.
		compiler = self._compile_source( '''
@protocol
class HasLabel:
	def describe( self ) -> i32:
		with compiler.wrap_arithmetic:
			return self.value * 2

class Box( HasLabel ):
	value: i32
	def __init__( self, value: i32 ) -> None:
		self.value = value

def main() -> i32:
	b = Box( 21 )
	with compiler.wrap_arithmetic:
		if b.describe() == 42:
			return 0
	return 1
''' )
		import emitter_c
		self._assert_compiles_and_runs( emitter_c.emit_c( compiler ), expected_exit = 0, compiler = compiler )

	def test_generic_typevar_bound_to_protocol( self ) -> None:
		# proves TypeVar(bound=SomeProtocol) is accepted and a conforming T
		# monomorphizes/dispatches correctly through it, end to end. The
		# REJECTION side (a non-conforming T) is compile-error-only - see
		# ProtocolCompileErrorTests' test_bound_violation_* cases above,
		# which don't need a real C compiler to verify.
		compiler = self._compile_source( '''
@protocol
class Greeter:
	def greet( self ) -> i32:
		...

class Talker( Greeter ):
	def greet( self ) -> i32:
		return 7

def call_greet[T: Greeter]( x: T ) -> i32:
	return x.greet()

def main() -> i32:
	t = Talker()
	if call_greet( t ) == 7:
		return 0
	return 1
''' )
		import emitter_c
		self._assert_compiles_and_runs( emitter_c.emit_c( compiler ), expected_exit = 0, compiler = compiler )


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
class ProtocolCrossModuleSpliceTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' regression coverage for a real bug: a protocol default method's body
	referencing a qualified name from its OWN module's imports (e.g.
	`fs.SEEK_CUR`) failed to resolve ('name X is not defined') once spliced
	into a conformer declared in a DIFFERENT module - _splice_protocol_default
	was parsing the copied body against the CONFORMER's ambient module_stack
	instead of the protocol's own defining module (lexical scoping: a
	function's free/global names resolve against where it was written, not
	where it's spliced to - only self.-prefixed names are meant to rebind).
	Reproduces with two entirely ordinary modules - not specific to lib/
	builtins/ - see lib/io.py's Seekable.tell()/lib/builtins/__File.py's
	BinaryReader for the original real-world trigger. '''

	def test_default_method_qualified_name_resolves_against_protocol_module( self ) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			root = Path( tmp )
			( root / 'xconst.py' ).write_text( 'ANSWER: i32 = 42\n' )
			( root / 'xproto.py' ).write_text( '''
import xconst

@protocol
class Answerable:
	def raw( self ) -> i32:
		...

	def answer( self ) -> i32:
		with compiler.wrap_arithmetic:
			return self.raw() + xconst.ANSWER
''' )
			( root / 'xconform.py' ).write_text( '''
import compiler
from xproto import Answerable

class Thing( Answerable ):
	def raw( self ) -> i32:
		return 0
''' )
			discovery = Discovery( paths = [ root, Path( 'lib' ).resolve() ], import_builtins = True )
			compiler = Compiler( discovery )
			compiler.import_code( '''
import compiler
from xconform import Thing

def main() -> i32:
	t = Thing()
	if t.answer() == 42:
		return 0
	return 1
''', Path( '__main__.py' ), scope = None )
			compiler.run()
			self.assertEqual( discovery.errors.errors, [],
				'compile errors:\n' + '\n'.join( str( e ) for e in discovery.errors.errors ) )
			import emitter_c
			self._assert_compiles_and_runs( emitter_c.emit_c( compiler ), expected_exit = 0, compiler = compiler )
