# Tests for implicit class-attribute declaration via __init__: `self.x = expr`
# for an `x` never pre-declared in the class body registers a new attribute,
# typed from a small, permanent allowlist of inferable RHS shapes (see
# discovery.py's _infer_init_attributes/_infer_simple_expr_type). __init__
# only, permanently - an object's attribute set must be fully fixed by the
# time __init__ finishes, so no other method gets this treatment.

import unittest
from pathlib import Path

from compiler import Compiler
import discovery
from discovery import Discovery
from mpy_types import Variable, Scalar
import test_support
from test_support import RealCompileMixin

# --- discovery-level: type correctly inferred, structural checks only -----

class InitAttributeInferenceDiscoveryTests( unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = False )

	def _import( self, code: str ) -> discovery.Module:
		return self.discovery.import_code( code, Path( '__main__.py' ), scope = None )

	def test_infers_attribute_type_from_init_parameter( self ) -> None:
		mod = self._import( '''
class Box:
	def __init__( self, v: i32 ) -> None:
		self.v = v
''' )
		box = mod.get_local( 'Box' )
		box.resolve()
		self.assertEqual( self.discovery.errors.errors, [] )
		v = box.chain_lookup( 'v' )
		self.assertIsInstance( v, Variable )
		self.assertIsInstance( v.type, Scalar )
		self.assertEqual( v.type.stem, 'i32' )

	def test_infers_attribute_type_from_constructor_call( self ) -> None:
		mod = self._import( '''
class Inner:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

class Outer:
	def __init__( self ) -> None:
		self.inner = Inner( v = 7 )
''' )
		outer = mod.get_local( 'Outer' )
		outer.resolve()
		self.assertEqual( self.discovery.errors.errors, [] )
		inner_field = outer.chain_lookup( 'inner' )
		self.assertIsInstance( inner_field, Variable )
		self.assertEqual( inner_field.type.qualname, '__main__.Inner' )

	def test_two_attributes_inferred_from_two_statements( self ) -> None:
		mod = self._import( '''
class Point:
	def __init__( self, x: i32, y: i32 ) -> None:
		self.x = x
		self.y = y
''' )
		point = mod.get_local( 'Point' )
		point.resolve()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertEqual( point.chain_lookup( 'x' ).type.stem, 'i32' )
		self.assertEqual( point.chain_lookup( 'y' ).type.stem, 'i32' )

	def test_chained_self_attribute_reference( self ) -> None:
		# self.b's own initializer reads self.a, already registered earlier
		# in this same __init__ - the ast.Attribute self.X allowlist branch
		mod = self._import( '''
class Chain:
	def __init__( self, a: i32 ) -> None:
		self.a = a
		self.b = self.a
''' )
		chain = mod.get_local( 'Chain' )
		chain.resolve()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertEqual( chain.chain_lookup( 'b' ).type.stem, 'i32' )

	def test_same_type_reassignment_across_if_else_branches( self ) -> None:
		mod = self._import( '''
class Flagged:
	def __init__( self, flag: bool, x: i32, y: i32 ) -> None:
		if flag:
			self.x = x
		else:
			self.x = y
''' )
		flagged = mod.get_local( 'Flagged' )
		flagged.resolve()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertEqual( flagged.chain_lookup( 'x' ).type.stem, 'i32' )

	def test_explicitly_declared_attribute_is_left_alone( self ) -> None:
		# self.x = x in __init__ for an ALREADY-declared attribute is just
		# an ordinary assignment - no second Variable registered, no error
		mod = self._import( '''
class Box:
	x: i32
	def __init__( self, x: i32 ) -> None:
		self.x = x
''' )
		box = mod.get_local( 'Box' )
		box.resolve()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertEqual( len( box.attributes ), 1 )

	def test_collision_with_existing_method_errors( self ) -> None:
		mod = self._import( '''
class Foo:
	def bar( self ) -> i32:
		return 1
	def __init__( self ) -> None:
		self.bar = 2
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'collides', self.discovery.errors.errors[0] )

	def test_expression_outside_allowlist_errors( self ) -> None:
		# self.compute() is a method call - outside the narrow, permanent
		# allowlist by design (see _infer_simple_expr_type's own docstring)
		mod = self._import( '''
class Foo:
	def compute( self ) -> i32:
		return 5
	def __init__( self ) -> None:
		self.x = self.compute()
''' )
		foo = mod.get_local( 'Foo' )
		foo.resolve()
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'cannot infer', self.discovery.errors.errors[0] )
		self.assertIn( "declare 'x:", self.discovery.errors.errors[0] )

	def test_assigning_an_inherited_field_is_not_treated_as_new( self ) -> None:
		# self.foo = foo in Derived's own __init__, where foo is declared on
		# Base and never redeclared on Derived - the ORDINARY inherited-
		# field-assignment case (test_subclass_own_init_is_not_shadowing in
		# discovery_test.py's own InheritanceTests), not a fresh attribute
		# to infer. A real regression found while implementing this feature:
		# the own-class-only `class_obj.names.get(name)` check alone can't
		# see an inherited field, so this used to be misread as brand new
		# and then flagged as shadowing Base.foo - must walk the base chain
		# too before deciding
		mod = self._import( '''
class Base:
	foo: i32

class Derived( Base ):
	def __init__( self, foo: i32 ) -> None:
		self.foo = foo
''' )
		derived = mod.get_local( 'Derived' )
		derived.resolve()
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertEqual( len( derived.attributes ), 0 ) # nothing NEW registered on Derived itself
		self.assertEqual( derived.chain_lookup( 'foo' ).qualname, '__main__.Base.foo' )

	def test_inferred_attribute_colliding_with_base_method_errors( self ) -> None:
		mod = self._import( '''
class Base:
	def foo( self ) -> i32:
		return 1

class Derived( Base ):
	def __init__( self, foo: i32 ) -> None:
		self.foo = foo
''' )
		derived = mod.get_local( 'Derived' )
		derived.resolve()
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'shadows', self.discovery.errors.errors[0] )


# --- lowering-level: needs real Stage 2/3 to run (attribute coercion,
# other-method behavior, AugAssign) - discovery's own class_obj.resolve()
# never lowers a method BODY, only its signature -------------------------

class InitAttributeInferenceLoweringTests( unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _import( self, code: str ):
		return self.compiler.import_code( code, filename = Path( '__test__.py' ))

	def test_conflicting_type_reassignment_is_ordinary_coercion_error( self ) -> None:
		# only the FIRST self.x = ... in __init__ sets the inferred type
		# (int, from the literal 1) - the second goes through the ordinary,
		# unmodified attribute-coercion path and fails there, same as any
		# other type mismatch would
		code = '\n'.join([
			'class Foo:',
			'	def __init__( self ) -> None:',
			'		self.x = 1',
			"		self.x = 'text'",
			'def main() -> None:',
			'	f: Foo = Foo()',
			'	return',
		])
		self._import( code )
		self.compiler.run()
		self.assertTrue( self.discovery.errors.errors )

	def test_undeclared_attribute_in_other_method_still_errors( self ) -> None:
		# the permanent __init__-only boundary: this method never gets the
		# implicit-declaration treatment, regardless of what __init__ does
		code = '\n'.join([
			'class Foo:',
			'	def __init__( self ) -> None:',
			'		pass',
			'	def other( self ) -> None:',
			'		self.x = 1',
			'def main() -> None:',
			'	f: Foo = Foo()',
			'	f.other()',
			'	return',
		])
		self._import( code )
		self.compiler.run()
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'no attribute', self.discovery.errors.errors[0] )

	def test_augassign_on_never_declared_attribute_still_errors( self ) -> None:
		# self.x += 1 is never collected by the implicit-declaration scan
		# (it only looks at ast.Assign) - falls through to the existing
		# "no attribute" error unchanged, same as before this feature
		code = '\n'.join([
			'class Foo:',
			'	def __init__( self ) -> None:',
			'		self.x += 1',
			'def main() -> None:',
			'	f: Foo = Foo()',
			'	return',
		])
		self._import( code )
		self.compiler.run()
		self.assertTrue( self.discovery.errors.errors )


# --- real compile+link+run: proves the RC/coercion machinery treats an
# inferred attribute exactly like a declared one - lowering.py needed zero
# changes, so this is the load-bearing check that claim actually holds -----

_LITERAL_INFERENCE = '''
class Counter:
	def __init__( self ) -> None:
		self.count = 0
	def get( self ) -> int:
		return self.count

def main() -> i32:
	c: Counter = Counter()
	if c.get() != 0:
		return 1
	return 0
'''

_PARAM_INFERENCE = '''
class Box:
	def __init__( self, v: i32 ) -> None:
		self.v = v
	def get( self ) -> i32:
		return self.v

def main() -> i32:
	b: Box = Box( v = 5 )
	if b.get() != 5:
		return 1
	return 0
'''

_CONSTRUCTOR_CALL_INFERENCE_RC = '''
class Inner:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

class Outer:
	def __init__( self ) -> None:
		self.inner = Inner( v = 7 )
	def get( self ) -> i32:
		return self.inner.v

def main() -> i32:
	o: Outer = Outer()
	if o.get() != 7:
		return 1
	return 0
'''

_CHAINED_SELF_ATTRIBUTE_INFERENCE = '''
class Chain:
	def __init__( self, a: i32 ) -> None:
		self.a = a
		self.b = self.a
	def get_b( self ) -> i32:
		return self.b

def main() -> i32:
	c: Chain = Chain( a = 3 )
	if c.get_b() != 3:
		return 1
	return 0
'''

_BRANCH_REASSIGNMENT_INFERENCE = '''
class Flagged:
	def __init__( self, flag: bool ) -> None:
		if flag:
			self.x = 1
		else:
			self.x = 2
	def get( self ) -> int:
		return self.x

def main() -> i32:
	a: Flagged = Flagged( flag = True )
	b: Flagged = Flagged( flag = False )
	if a.get() != 1:
		return 1
	if b.get() != 2:
		return 2
	return 0
'''

# read from a method OTHER than __init__ - the core motivating case: proves
# the attribute is fully registered before ANY unit of the class can be
# lowered, not just whenever __init__ itself happens to be reached
_READ_FROM_OTHER_METHOD_BEFORE_INIT_IN_SOURCE = '''
class Foo:
	def get( self ) -> i32:
		return self.v
	def __init__( self, v: i32 ) -> None:
		self.v = v

def main() -> i32:
	f: Foo = Foo( v = 9 )
	if f.get() != 9:
		return 1
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile tests' )
class InitAttributeInferenceRealCompileTests( RealCompileMixin, unittest.TestCase ):
	def test_inferred_attributes_behave_like_declared_ones( self ) -> None:
		self.assert_programs_run([
			( 'literal_inference', _LITERAL_INFERENCE ),
			( 'param_inference', _PARAM_INFERENCE ),
			( 'constructor_call_inference_rc', _CONSTRUCTOR_CALL_INFERENCE_RC ),
			( 'chained_self_attribute_inference', _CHAINED_SELF_ATTRIBUTE_INFERENCE ),
			( 'branch_reassignment_inference', _BRANCH_REASSIGNMENT_INFERENCE ),
			( 'read_from_other_method_before_init_in_source', _READ_FROM_OTHER_METHOD_BEFORE_INIT_IN_SOURCE ),
		])


if __name__ == '__main__':
	unittest.main()
