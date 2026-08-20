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
		# NOTE: this only proves TypeVar(bound=SomeProtocol) is accepted and
		# a conforming T monomorphizes/dispatches correctly through it - it
		# does NOT prove a NON-conforming T gets rejected. That enforcement
		# (checking the concrete T's own .protocols at generic instantiation
		# time - see discovery.py's _get_or_create_specialization, the right
		# cache/builder to hook, but shared by many unrelated call sites with
		# no AST node/error-location available there) is a deliberate,
		# flagged gap, not yet implemented - see task tracking.
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
