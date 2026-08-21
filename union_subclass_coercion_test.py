# Real-compile-and-run regression tests for a compiler bug in the argument-
# coercion path that decides whether an operand's type can flow into a
# declared union type (T|None, or any other union) - lowering.py's
# _coerce_or_check_operand (the outer "does operand match one of the union's
# own leaves" probe) and _coerce_into_union (the leaf lookup + synthesized
# member-constructor call).
#
# Both only recognized a leaf match via TypeResolver._same_type - an EXACT
# type match. Ordinary derived->base substitution (passing a Sub instance
# where a plain Base-typed parameter is declared) is supported everywhere
# else in the codebase via _is_rcclass_upcast, but neither of the two union-
# coercion sites consulted it: a Sub instance flowing into a union leaf
# declared as Sub's OWN base class (Base|None) was rejected outright with
# "expected NoneType|Base, got Sub - these are different types", even though
# passing the exact same Sub() to a plain Base-typed (non-union) parameter
# already worked fine.
#
# Fix: both sites now also accept a leaf match via _is_rcclass_upcast, not
# just _same_type. _coerce_into_union additionally CastWraps the operand up
# to the matched leaf's own declared type before calling the synthesized
# union-member constructor, since that constructor's own parameter is always
# declared as exactly the leaf's type (Base), never whatever subclass (Sub)
# actually flowed in - C rejects passing struct Sub* where struct Base* is
# declared without an explicit reinterpret cast, the same requirement the
# existing plain (non-union) RCClass-upcast branch already handles.

import unittest

import test_support
from test_support import RealCompileMixin

# the original minimal repro: a Sub instance flowing into a Base|None
# parameter - virtual dispatch through the union-unwrapped value must reach
# the SUBCLASS's own override, not silently fall back to Base's, proving the
# coercion preserved the real leaf's identity rather than truncating it to
# Base along the way.
_SUBCLASS_INTO_OPTIONAL_BASE = '''
class Base:
	@virtual
	def tag( self ) -> i32:
		return 1

class Sub( Base ):
	@virtual
	def tag( self ) -> i32:
		return 2

def take( x: Base|None = None ) -> i32:
	if x is None:
		return -1
	return x.tag()

def main() -> i32:
	if take( Sub() ) != 2:
		return 1
	if take( Base() ) != 1:
		return 2
	if take() != -1:
		return 3
	if take( None ) != -1:
		return 4
	return 0
'''

# the upcast has to walk the FULL inheritance chain, not just one level -
# Leaf is Root's grandchild, not its direct child.
_MULTI_LEVEL_SUBCLASS_INTO_OPTIONAL_BASE = '''
class Root:
	@virtual
	def tag( self ) -> i32:
		return 10

class Mid( Root ):
	pass

class Leaf( Mid ):
	@virtual
	def tag( self ) -> i32:
		return 30

def take( x: Root|None = None ) -> i32:
	if x is None:
		return -1
	return x.tag()

def main() -> i32:
	if take( Leaf() ) != 30:
		return 1
	if take( Mid() ) != 10:
		return 2
	return 0
'''

# an already-owned LOCAL (not a fresh call-result temp) coerced through the
# same upcast-into-union path - the CastWrap this fix inserts is a pure
# reinterpret (no incref/decref of its own, see _coerce_or_check_operand's
# own RCClass-upcast comment), so the union constructor's OWN incref should
# be the only refcount change visible to the caller, and it should fully
# unwind once the callee's own union-typed parameter goes out of scope.
_SUBCLASS_LOCAL_INTO_OPTIONAL_BASE_REFCOUNT = '''
class Base:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v
	@virtual
	def tag( self ) -> i32:
		return self.v

class Sub( Base ):
	@virtual
	def tag( self ) -> i32:
		with compiler.wrap_arithmetic:
			return self.v + 1

def take( x: Base|None ) -> i32:
	if x is None:
		return -1
	return x.tag()

def main() -> i32:
	with compiler.wrap_arithmetic:
		s: Sub = Sub( v = 41 )
		before: usize = compiler.refcount( s )
		result: i32 = take( s )
		after: usize = compiler.refcount( s )
		if result != 42:
			return 1
		if after != before:
			return 2
	return 0
'''

# a fresh Sub() constructed directly at the call site (no local binding at
# all) repeated many times - the exact shape of the original minimal repro,
# run in a loop to surface any leak or double-free the single-shot version
# above wouldn't catch.
_SUBCLASS_FRESH_CONSTRUCTION_LOOP_NO_LEAK = '''
class Base:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v
	@virtual
	def tag( self ) -> i32:
		return self.v

class Sub( Base ):
	@virtual
	def tag( self ) -> i32:
		return self.v

def take( x: Base|None = None ) -> i32:
	if x is None:
		return -1
	return x.tag()

def main() -> i32:
	with compiler.wrap_arithmetic:
		i: i32 = 0
		while i < 1000:
			if take( Sub( v = i ) ) != i:
				return 1
			i += 1
		return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile tests' )
class UnionSubclassCoercionTests( RealCompileMixin, unittest.TestCase ):
	def test_subclass_and_plain_base_both_coerce_into_optional_base( self ) -> None:
		self.assert_programs_run([
			( 'subclass_into_optional_base', _SUBCLASS_INTO_OPTIONAL_BASE ),
			( 'multi_level_subclass_into_optional_base', _MULTI_LEVEL_SUBCLASS_INTO_OPTIONAL_BASE ),
		])

	def test_subclass_upcast_into_union_refcount_is_exact( self ) -> None:
		self.assert_programs_run([
			( 'subclass_local_into_optional_base_refcount', _SUBCLASS_LOCAL_INTO_OPTIONAL_BASE_REFCOUNT ),
		])

	def test_repeated_fresh_subclass_construction_into_union_no_leak( self ) -> None:
		self.assert_programs_run([
			( 'subclass_fresh_construction_loop_no_leak', _SUBCLASS_FRESH_CONSTRUCTION_LOOP_NO_LEAK ),
		])


if __name__ == '__main__':
	unittest.main()
