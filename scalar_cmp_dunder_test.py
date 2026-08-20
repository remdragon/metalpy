# Real-compile-and-run behavioral test for the comparison-dunder rollout:
# scalar (i8..i128, u8..u128, f32/f64, bool), Ptr[T]/ConstPtr[T], and CEnum
# ==/!=/</<=/>/>= all now dispatch through a real dunder
# (gen_scalar_dunders.py's scalar_eq/etc, lib/builtins/__ptr_arith.py's
# ptr_eq/etc, discovery.py's _synthesize_cenum_comparison_methods) instead of
# lowering.py hardcoding a flat ir.Cmp whenever the left operand happened to
# be Scalar. The isinstance(Scalar) special-casing that used to gate dunder
# lookup in _expr_Compare/_lower_eq_or_ne/_lower_operand_compare/
# _classify_leaf_pair_eq is gone - comparison dispatch is now unified with
# arithmetic's own dunder-only dispatch (see binop_fallback_eliminated).
#
# The flat-Cmp fallback itself is also gone, for every type - confirmed with
# the user: there's no sensible default for comparing two arbitrary values
# (an RCClass's own == is meaningless unless the class defines it), so a
# type with no matching comparison dunder is now a hard compile error, not a
# silent pointer/identity comparison.
#
# CEnum comparisons are auto-synthesized (comparing the underlying
# value_type) UNLESS the user defines ANY of the 6 comparison dunders
# themselves, in which case none are auto-synthesized (confirmed with the
# user - a single user override opts the whole class out, not just that one
# name) - this also required a real, independent, pre-existing gap fix:
# CEnum had no `.methods` field at all (mpy_types.py), so ANY method on an
# @enum class (auto-synthesized or hand-written) crashed discovery.py's
# _parse_function outright (AttributeError: 'CEnum' object has no attribute
# 'methods') - @enum classes apparently never had a single method written on
# them anywhere in this codebase before.

import unittest

import emitter_c
import test_support
from test_support import RealCompileMixin


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile comparison-dunder tests' )
class ScalarCmpDunderBehaviorTests( RealCompileMixin, unittest.TestCase ):
	def test_scalar_all_six_ops_every_type(self) -> None:
		compiler = self._compile_source( '''
def main() -> i32:
	a: i32 = 5
	a2: i32 = 5 # same value as a, distinct variable - exercises the reflexive
	# (==/!=/<=/>=) cases below without a literal `a == a`-style self-
	# comparison, which is a real, confirmed -Wtautological-compare on the
	# generated C (correctly flagging a tautology - the compiler isn't
	# wrong, this test just needed two equal-but-distinct operands instead)
	b: i32 = 7
	if not ( a == a2 ): return 1
	if a == b: return 2
	if not ( a != b ): return 3
	if a2 != a: return 4
	if not ( a < b ): return 5
	if not ( a <= a2 ): return 6
	if not ( b > a ): return 7
	if not ( a >= a2 ): return 8
	f: f64 = 1.5
	g: f64 = 2.5
	if not ( f < g ): return 9
	if f == g: return 10
	x: bool = True
	y: bool = False
	if x == y: return 11
	if not ( x != y ): return 12
	return 0
''' )
		self._assert_compiles_and_runs( emitter_c.emit_c( compiler ), expected_exit = 0, compiler = compiler )

	def test_scalar_comparison_dunder_is_zero_overhead( self ) -> None:
		# @inline splices - no real scalar_eq[T]/etc function should ever be
		# compiled, same zero-overhead convention every other scalar dunder
		# in this file follows (checked_dunder_test.py's own precedent)
		compiler = self._compile_source( '''
def main() -> i32:
	a: i32 = 5
	if a == 5:
		return 0
	return 1
''' )
		qualnames = { lf.function.qualname for lf in compiler.functions }
		self.assertFalse(
			any( 'scalar_eq' in q for q in qualnames ),
			f'a real scalar comparison function was compiled, @inline should have spliced it instead: {sorted(qualnames)}',
		)

	def test_ptr_comparison_all_six_ops( self ) -> None:
		compiler = self._compile_source( '''
import sys
def main() -> i32:
	a: Ptr[u8] = sys.alloc[u8]( 4 )
	b: Ptr[u8] = a
	c: Ptr[u8] = sys.alloc[u8]( 4 )
	ok: i32 = 0
	if not ( a == b ): ok = 1
	if a == c: ok = 2
	if not ( a != c ): ok = 3
	if not ( ( a < c ) or ( c < a ) ): ok = 4 # distinct allocations - exactly one direction holds
	sys.free( a )
	sys.free( c )
	return ok
''' )
		self._assert_compiles_and_runs( emitter_c.emit_c( compiler ), expected_exit = 0, compiler = compiler )

	def test_cenum_auto_synthesized_comparisons( self ) -> None:
		compiler = self._compile_source( '''
@enum( u8 )
class Color:
	Red = 0
	Green = 1
def main() -> i32:
	a: Color = Color.Red
	b: Color = Color.Red
	c: Color = Color.Green
	if not ( a == b ): return 1
	if a == c: return 2
	if not ( a != c ): return 3
	if not ( a < c ): return 4
	if not ( c > a ): return 5
	if not ( a <= b ): return 6
	if not ( a >= b ): return 7
	return 0
''' )
		self._assert_compiles_and_runs( emitter_c.emit_c( compiler ), expected_exit = 0, compiler = compiler )

	def test_cenum_user_defined_eq_suppresses_auto_synthesis_of_others( self ) -> None:
		# a single user-declared comparison dunder opts the WHOLE class out
		# of auto-synthesis - __ne__ is NOT filled in just because the user
		# only wrote __eq__, so `a != c` below must be a compile error
		self._compile_source_expect_error( '''
@enum( u8 )
class Color:
	Red = 0
	Green = 1
	def __eq__( self, other: Color ) -> bool:
		return True
def main() -> i32:
	a: Color = Color.Red
	c: Color = Color.Green
	if a != c:
		return 1
	return 0
''', 'Color has no __ne__() defined' )

	def test_cenum_user_defined_eq_used_instead_of_synthesized( self ) -> None:
		compiler = self._compile_source( '''
@enum( u8 )
class Color:
	Red = 0
	Green = 1
	def __eq__( self, other: Color ) -> bool:
		return True
def main() -> i32:
	a: Color = Color.Red
	c: Color = Color.Green
	if not ( a == c ): # the user's own override always returns True
		return 1
	return 0
''' )
		self._assert_compiles_and_runs( emitter_c.emit_c( compiler ), expected_exit = 0, compiler = compiler )

	def test_rcclass_without_eq_is_a_compile_error( self ) -> None:
		# confirmed with the user: no sensible default exists for comparing
		# two arbitrary class instances - this must be a hard compile error,
		# not a silent pointer-identity flat Cmp
		self._compile_source_expect_error( '''
class Widget:
	pass
def main() -> i32:
	a = Widget()
	b = Widget()
	if a == b:
		return 1
	return 0
''', 'has no __eq__() defined' )

	def test_rcclass_with_eq_still_works( self ) -> None:
		compiler = self._compile_source( '''
class Widget:
	y: i32
	def __init__( self, y: i32 ) -> None:
		self.y = y
	def __eq__( self, other: Widget ) -> bool:
		return self.y == other.y
def main() -> i32:
	a = Widget( 5 )
	b = Widget( 5 )
	c = Widget( 6 )
	if not ( a == b ): return 1
	if a == c: return 2
	return 0
''' )
		self._assert_compiles_and_runs( emitter_c.emit_c( compiler ), expected_exit = 0, compiler = compiler )

	def _compile_source_expect_error( self, source: str, needle: str ) -> None:
		from pathlib import Path
		from compiler import Compiler
		from discovery import Discovery
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		compiler.import_code( source, Path( '__main__.py' ), scope = None )
		compiler.run()
		joined = '\n'.join( str( e ) for e in discovery.errors.errors )
		self.assertIn( needle, joined, f'expected a compile error containing {needle!r}, got:\n{joined or "(no errors)"}' )


if __name__ == '__main__':
	unittest.main()
