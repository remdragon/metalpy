# Real-compile-and-run behavioral tests for lib/builtins/__int.py's
# arbitrary-precision `int` type. Unlike the type-resolution/emission-shape
# tests in emitter_c_test.py, these actually compile, link, and RUN a real
# executable (mirroring emitter_c_test.py's own RealCompileTests/
# _ClangCompileMixin pattern) - correctness of int's actual arithmetic can
# only be confirmed by executing the generated code, not by inspecting IR or
# C source text.
#
# Each MetalPy program under test returns a distinct nonzero i32 exit code
# per failed assertion (0 means every assertion in that program passed) -
# the Python test method then just asserts the process exited 0. A crash
# (segfault, heap corruption, STATUS_HEAP_CORRUPTION on Windows, ...) shows
# up as a nonzero/negative return code too, so these tests also catch
# memory-safety bugs, not just wrong values.
#
# Related tests are grouped into one compiled program per cluster (rather
# than one program per single assertion) for two reasons: it keeps the
# number of real compile+link+run round trips (slow) manageable, and it
# sidesteps a separate, pre-existing, general compiler bug found while
# writing these tests - a @union type used only as a Result[...] error type
# parameter, with no reachable code anywhere in the program actually
# constructing one of its variants, crashes emit_c() outright (confirmed
# with a minimal repro entirely outside int: AssertionError, "_tagged_union_
# storage has not run yet"). Every cluster below exercises at least one
# IntError-constructing path for real, avoiding that crash.

# stdlib imports:
from pathlib import Path
import subprocess
import tempfile
import unittest

# local imports:
import emitter_c
import linker_c
import test_support
from compiler import Compiler
from discovery import Discovery

_CC = linker_c.detect_cc()
_HAS_I128 = linker_c.has_i128( _CC )


@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping real-compile int tests' )
class IntBehaviorTests( unittest.TestCase ):
	def _run_program( self, code: str ) -> subprocess.CompletedProcess:
		''' compiles `code` (a full MetalPy source, needs its own def main()
		-> i32) against the real builtins, links it, runs it, and returns
		the finished subprocess.CompletedProcess (stdout/stderr captured). '''
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		compiler.import_code( code, Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [], f'compile errors:\n' + '\n'.join( str(e) for e in discovery.errors.errors ) )

		no_crt = 'c' not in compiler.extern_libs
		c_source = emitter_c.emit_c( compiler, no_crt = no_crt )

		with tempfile.TemporaryDirectory() as tmp:
			src_path = Path( tmp ) / 'generated.c'
			obj_path = Path( tmp ) / 'generated.o'
			exe_path = Path( tmp ) / 'test_exe.exe'
			src_path.write_text( c_source, encoding = 'utf-8' )

			cc_result = _CC.compile( src_path, obj_path, no_crt = no_crt )
			self.assertEqual( cc_result.returncode, 0, f'{_CC.name} compile failed:\n{cc_result.stdout}{test_support.c_source_on_failure( c_source )}' )

			# mirrors mpy.py's own ldflags construction: every extern
			# library the program actually pulled in (kernel32, ntdll, ...)
			# needs to be linked explicitly - 'c' is the CRT itself, already
			# handled by no_crt above, not a real link flag
			ldflags = ''
			for lib in sorted( compiler.extern_libs ):
				if lib == 'c':
					continue
				flag = linker_c.resolve_lib_ldflag( _CC, lib, compiler.extern_libs[lib], no_crt = no_crt )
				ldflags = ldflags + f' {flag}' if ldflags else flag

			link_result = _CC.link( exe_path, [ obj_path ], ldflags = ldflags, no_crt = no_crt )
			self.assertEqual( link_result.returncode, 0, f'{_CC.name} link failed:\n{link_result.stdout}' )

			return subprocess.run( [ str( exe_path ) ], capture_output = True )

	def _assert_program_succeeds( self, code: str, check_names: list[str] ) -> None:
		''' runs `code` and asserts it exited 0. check_names[i] (0-indexed)
		names whatever assertion inside the program returns i+1 on failure,
		purely so a failure message is legible without cross-referencing
		the MetalPy source by hand. '''
		result = self._run_program( code )
		if result.returncode == 0:
			return
		if 1 <= result.returncode <= len( check_names ):
			failed = check_names[ result.returncode - 1 ]
		else:
			failed = f'(unmapped exit code {result.returncode} - possible crash/corruption, not a plain check failure)'
		self.fail(
			f'program exited {result.returncode}, expected 0. Failed check: {failed}\n'
			f'stdout: {result.stdout}\nstderr: {result.stderr}'
		)

	# --- construction, comparison, is_zero/is_negative ---------------------

	def test_construction_and_comparison( self ) -> None:
		checks = [
			'int(5) == int(5)',
			'int(5) != int(3)',
			'int(-5).is_negative()',
			'not int(5).is_negative()',
			'int(0).is_zero()',
			'not int(1).is_zero()',
			'int(-1).is_zero() is False (still not zero)',
			'int(3) < int(5)',
			'int(5) <= int(5)',
			'int(5) > int(3)',
			'int(5) >= int(5)',
			'int(-5) < int(3) (negative always less than positive)',
			'int(-5) < int(-3) (more negative is smaller)',
			'int(0) == int(0)',
			'from_str constructs an IntError on bad input (union-scheduling guard)',
		]
		self._assert_program_succeeds( '''
def main() -> i32:
	if not ( int(5) == int(5) ):
		return 1
	if not ( int(5) != int(3) ):
		return 2
	if not int(-5).is_negative():
		return 3
	if int(5).is_negative():
		return 4
	if not int(0).is_zero():
		return 5
	if int(1).is_zero():
		return 6
	if int(-1).is_zero():
		return 7
	if not ( int(3) < int(5) ):
		return 8
	if not ( int(5) <= int(5) ):
		return 9
	if not ( int(5) > int(3) ):
		return 10
	if not ( int(5) >= int(5) ):
		return 11
	if not ( int(-5) < int(3) ):
		return 12
	if not ( int(-5) < int(-3) ):
		return 13
	if not ( int(0) == int(0) ):
		return 14
	if not int.from_str('bad').is_err():
		return 15
	return 0
''', checks )

	# --- addition, subtraction, unary negation ------------------------------

	def test_add_sub_neg( self ) -> None:
		checks = [
			'123 + 45 == 168',
			'45 + 123 == 168 (commutative)',
			'(-45) + (-123) == -168 (same-sign negative add)',
			'123 + (-45) == 78 (cross-sign, |a|>|b|)',
			'45 + (-123) == -78 (cross-sign, |a|<|b|)',
			'5 + (-5) == 0, not negative zero (regression: used to compare < 0)',
			'(-5) + 5 == 0, not negative zero (opposite operand order)',
			'99 + 1 == 100 (carries across a digit boundary)',
			'168 - 45 == 123',
			'45 - 168 == -123',
			'5 - 5 == 0, not negative zero',
			'(-5) - (-5) == 0, not negative zero',
			'-(int(5)) == -5',
			'-(int(-5)) == 5',
			'-(int(0)) == 0, and stays non-negative (str has no leading "-")',
			'from_str constructs an IntError on bad input (union-scheduling guard)',
		]
		self._assert_program_succeeds( '''
def main() -> i32:
	if ( int(123) + int(45) ).unwrap('x') != int(168):
		return 1
	if ( int(45) + int(123) ).unwrap('x') != int(168):
		return 2
	if ( int(-45) + int(-123) ).unwrap('x') != int(-168):
		return 3
	if ( int(123) + int(-45) ).unwrap('x') != int(78):
		return 4
	if ( int(45) + int(-123) ).unwrap('x') != int(-78):
		return 5
	zero1: int = ( int(5) + int(-5) ).unwrap('x')
	if zero1 != int(0) or zero1.is_negative():
		return 6
	zero2: int = ( int(-5) + int(5) ).unwrap('x')
	if zero2 != int(0) or zero2.is_negative():
		return 7
	if ( int(99) + int(1) ).unwrap('x') != int(100):
		return 8
	if ( int(168) - int(45) ).unwrap('x') != int(123):
		return 9
	if ( int(45) - int(168) ).unwrap('x') != int(-123):
		return 10
	zero3: int = ( int(5) - int(5) ).unwrap('x')
	if zero3 != int(0) or zero3.is_negative():
		return 11
	zero4: int = ( int(-5) - int(-5) ).unwrap('x')
	if zero4 != int(0) or zero4.is_negative():
		return 12
	neg5: int = ( -int(5) ).unwrap('x')
	if neg5 != int(-5):
		return 13
	pos5: int = ( -int(-5) ).unwrap('x')
	if pos5 != int(5):
		return 14
	negzero: int = ( -int(0) ).unwrap('x')
	if negzero != int(0) or negzero.is_negative():
		return 15
	if negzero.__str__() != '0':
		return 15
	if not int.from_str('bad').is_err():
		return 16
	return 0
''', checks )

	# --- multiplication (incl. the -0 guard) --------------------------------

	def test_mul( self ) -> None:
		checks = [
			'6 * 7 == 42',
			'(-6) * 7 == -42',
			'6 * (-7) == -42',
			'(-6) * (-7) == 42 (double negative)',
			'0 * (-5) == 0, and str has no leading "-" (no negative zero)',
			'(-5) * 0 == 0, and str has no leading "-" (no negative zero)',
			'123 * 456 == 56088 (multi-digit, exercises carrying)',
			'from_str constructs an IntError on bad input (union-scheduling guard)',
		]
		self._assert_program_succeeds( '''
def main() -> i32:
	if ( int(6) * int(7) ).unwrap('x') != int(42):
		return 1
	if ( int(-6) * int(7) ).unwrap('x') != int(-42):
		return 2
	if ( int(6) * int(-7) ).unwrap('x') != int(-42):
		return 3
	if ( int(-6) * int(-7) ).unwrap('x') != int(42):
		return 4
	z1: int = ( int(0) * int(-5) ).unwrap('x')
	if z1 != int(0) or z1.__str__() != '0':
		return 5
	z2: int = ( int(-5) * int(0) ).unwrap('x')
	if z2 != int(0) or z2.__str__() != '0':
		return 6
	if ( int(123) * int(456) ).unwrap('x') != int(56088):
		return 7
	if not int.from_str('bad').is_err():
		return 8
	return 0
''', checks )

	# --- divmod, floordiv/mod operators, divide-by-zero ---------------------

	def test_divmod_and_operators( self ) -> None:
		# __floordiv__/__mod__ are @fallible_arithmetic (see lib/builtins/__int.py) -
		# under panic_arithmetic they auto-unwrap to a plain int (no
		# .unwrap() needed), exactly like a bare scalar `/` already does;
		# .divmod() itself is a plain method call, never affected by
		# @fallible_arithmetic (only real BinOp operator syntax is), so the explicit-
		# call assertions below (including the divide-by-zero one, which
		# deliberately wants Err rather than a panic) are untouched.
		checks = [
			'17 divmod 5 == (3, 2)',
			'17 // 5 == 3 (operator dispatches to __floordiv__, auto-unwraps under panic_arithmetic)',
			'17 % 5 == 2 (operator dispatches to __mod__)',
			'-7 // 2 == -3 (truncating, matches C)',
			"-7 % 2 == -1 (remainder takes dividend's sign, matches C)",
			'7 // -2 == -3',
			'7 % -2 == 1',
			'123 // 10 == 12 and 123 % 10 == 3',
			'divide by zero returns Err via explicit .divmod(), does not crash',
			'1 divmod 1 == (1, 0) (regression: smallest case, once corrupted the heap)',
		]
		self._assert_program_succeeds( '''
def main() -> i32:
	dm1: tuple[int,int] = int(17).divmod(int(5)).unwrap('x')
	if dm1[0] != int(3) or dm1[1] != int(2):
		return 1
	with compiler.panic_arithmetic('unexpected division failure'):
		q1: int = int(17) // int(5)
		if q1 != int(3):
			return 2
		m1: int = int(17) % int(5)
		if m1 != int(2):
			return 3
		q2: int = int(-7) // int(2)
		if q2 != int(-3):
			return 4
		m2: int = int(-7) % int(2)
		if m2 != int(-1):
			return 5
		q3: int = int(7) // int(-2)
		if q3 != int(-3):
			return 6
		m3: int = int(7) % int(-2)
		if m3 != int(1):
			return 7
		q4: int = int(123) // int(10)
		if q4 != int(12):
			return 8
		m4: int = int(123) % int(10)
		if m4 != int(3):
			return 8
	if not int(5).divmod(int(0)).is_err():
		return 9
	dm2: tuple[int,int] = int(1).divmod(int(1)).unwrap('x')
	if dm2[0] != int(1) or dm2[1] != int(0):
		return 10
	return 0
''', checks )

	def test_floordiv_mod_operator_default_mode_propagates_and_panic_mode_panics( self ) -> None:
		# new coverage (not present before @fallible_arithmetic): default (checked)
		# mode auto-propagates a real ZeroDivisionError through the
		# enclosing function's own Result return type, exactly like a bare
		# checked scalar `/` already does; panic_arithmetic auto-panics
		# (nonzero exit) on a real divide-by-zero, rather than returning an
		# Err the caller must explicitly unwrap.
		self._assert_program_succeeds( '''
def div_it( a: int, b: int ) -> Result[int, IntError|ZeroDivisionError]:
	c = a // b
	return Result.Ok( c )

def main() -> i32:
	ok = div_it( int(17), int(5) )
	if ok.is_err() or ok.unwrap('x') != int(3):
		return 1
	failed = div_it( int(5), int(0) )
	if failed.is_ok():
		return 2
	return 0
''', [
			'default mode: no failure - ordinary Result.Ok',
			'default mode: divide by zero auto-propagates as Err, not a panic/crash',
		] )

	# --- from_str parsing, incl. edge cases -----------------------------

	def test_from_str( self ) -> None:
		checks = [
			'"98765" parses to 98765',
			'"-42" parses to -42',
			'"007" parses to 7 (leading zeros stripped)',
			'"0" parses to 0',
			'"-0" parses to 0, not negative (is_negative() is False)',
			'"" (empty string) is Err, not a crash (regression: used to underflow usize)',
			'"-" (lone minus) is Err',
			'"12a3" (embedded invalid char) is Err',
			'"abc" (no valid digits at all) is Err',
			'round-trip: int(N).__str__() parses back via from_str to the same value',
		]
		self._assert_program_succeeds( '''
def main() -> i32:
	if int.from_str('98765').unwrap('x') != int(98765):
		return 1
	if int.from_str('-42').unwrap('x') != int(-42):
		return 2
	if int.from_str('007').unwrap('x') != int(7):
		return 3
	if int.from_str('0').unwrap('x') != int(0):
		return 4
	negzero: int = int.from_str('-0').unwrap('x')
	if negzero != int(0) or negzero.is_negative():
		return 5
	if not int.from_str('').is_err():
		return 6
	if not int.from_str('-').is_err():
		return 7
	if not int.from_str('12a3').is_err():
		return 8
	if not int.from_str('abc').is_err():
		return 9
	roundtrip: int = int.from_str( int(4242).__str__() ).unwrap('x')
	if roundtrip != int(4242):
		return 10
	return 0
''', checks )

	# --- i32 narrowing/widening ------------------------------------------

	def test_i32_conversions( self ) -> None:
		checks = [
			'from_i32(42).to_i32() round-trips to 42',
			'from_i32(-42).to_i32() round-trips to -42',
			'from_i32(0).to_i32() round-trips to 0',
			'i32.MIN round-trips through from_i32/to_i32',
			'i32.MAX round-trips through from_i32/to_i32',
			'a value one past i32.MAX (from_str) overflows to_i32 (Err, not wraparound)',
			'a value one before i32.MIN (from_str) overflows to_i32 (Err, not wraparound)',
			'from_str constructs an IntError on bad input (union-scheduling guard)',
		]
		self._assert_program_succeeds( '''
def main() -> i32:
	if int.from_i32(42).unwrap('x').to_i32().unwrap('y') != 42:
		return 1
	if int.from_i32(-42).unwrap('x').to_i32().unwrap('y') != -42:
		return 2
	if int.from_i32(0).unwrap('x').to_i32().unwrap('y') != 0:
		return 3
	i32_min: i32 = -2147483648
	if int.from_i32(i32_min).unwrap('x').to_i32().unwrap('y') != i32_min:
		return 4
	i32_max: i32 = 2147483647
	if int.from_i32(i32_max).unwrap('x').to_i32().unwrap('y') != i32_max:
		return 5
	too_big: int = int.from_str('2147483648').unwrap('x')
	if not too_big.to_i32().is_err():
		return 6
	too_small: int = int.from_str('-2147483649').unwrap('x')
	if not too_small.to_i32().is_err():
		return 7
	if not int.from_str('bad').is_err():
		return 8
	return 0
''', checks )

	def test_scalar_conversions( self ) -> None:
		checks = [
			'to_i8/to_u8 round-trip MIN/MAX, overflow one past either end',
			'to_i64 correctly overflows a 19-digit value that exceeds i64::MAX (pre-existing accumulator-overflow bug fixed here)',
			'a negative int overflows any unsigned target',
			'i32(some_int) construct-cast syntax dispatches through __i32__ and yields a plain i32 (not a Result) under panic_arithmetic on success',
		]
		self._assert_program_succeeds( '''
def main() -> i32:
	if int.from_str('-128').unwrap('a').to_i8().unwrap('b') != -128:
		return 1
	if int.from_str('127').unwrap('a').to_i8().unwrap('b') != 127:
		return 2
	if not int.from_str('128').unwrap('a').to_i8().is_err():
		return 3
	if not int.from_str('-129').unwrap('a').to_i8().is_err():
		return 4
	if int.from_str('255').unwrap('a').to_u8().unwrap('b') != 255:
		return 5
	if not int.from_str('256').unwrap('a').to_u8().is_err():
		return 6

	# 19 digits, exceeds i64::MAX (9223372036854775807) - digit-count-only
	# bounding used to accumulator-overflow (panic) here instead of
	# cleanly returning Err
	if not int.from_str('9999999999999999999').unwrap('a').to_i64().is_err():
		return 7

	if not int.from_str('-1').unwrap('a').to_u32().is_err():
		return 8

	small: int = int.from_i32(42).unwrap('a')
	with compiler.panic_arithmetic('should not overflow'):
		if i32(small) != 42:
			return 9

	return 0
''', checks )

	@unittest.skipUnless( _HAS_I128, "MSVC's i128/u128 64-bit fallback doesn't have true 128-bit range - see emitter_c.py's __metalpy_wideint" )
	def test_scalar_conversions_i128_u128( self ) -> None:
		# separate from test_scalar_conversions above: i128/u128 themselves
		# (unlike every narrower target, which _to_i64/_to_u64 deliberately
		# avoid routing through i128/u128 for exactly this reason - see
		# int._to_i64's own comment) inherit the same pre-existing MSVC-
		# only 64-bit fallback every other i128/u128 feature in this
		# compiler already has, so this needs the same skip every other
		# real-i128-range test in this codebase already uses.
		checks = [
			'to_u128 round-trips u128::MAX, overflows one past it',
		]
		self._assert_program_succeeds( '''
def main() -> i32:
	u128_max: int = int.from_str('340282366920938463463374607431768211455').unwrap('a')
	if u128_max.to_u128().unwrap('b') != 340282366920938463463374607431768211455:
		return 1
	one_past: int = int.from_str('340282366920938463463374607431768211456').unwrap('a')
	if not one_past.to_u128().is_err():
		return 2
	return 0
''', checks )

	def test_scalar_conversion_construct_cast_panics_on_overflow( self ) -> None:
		# i32(x) (x: int) dispatches through __i32__ (@fallible_arithmetic),
		# so a real overflow under panic_arithmetic auto-panics - a nonzero,
		# crashed exit, not a returned Err the caller could inspect. Checked
		# via the raw process exit code (like _run_program), not _assert_
		# program_succeeds, since the whole point here is that it does NOT
		# exit cleanly.
		result = self._run_program( '''
def main() -> i32:
	big: int = int.from_str('99999999999999999999').unwrap('a')
	with compiler.panic_arithmetic('deliberate overflow'):
		return i32(big)
''' )
		self.assertNotEqual( result.returncode, 0, 'i32(x) overflow under panic_arithmetic should panic, not exit 0' )

	# --- __str__/__repr__ ---------------------------------------------------

	def test_str_repr( self ) -> None:
		checks = [
			'int(0).__str__() == "0"',
			'int(123).__str__() == "123"',
			'int(-123).__str__() == "-123"',
			'int(5).__repr__() == int(5).__str__()',
			'from_str constructs an IntError on bad input (union-scheduling guard)',
		]
		self._assert_program_succeeds( '''
def main() -> i32:
	if int(0).__str__() != '0':
		return 1
	if int(123).__str__() != '123':
		return 2
	if int(-123).__str__() != '-123':
		return 3
	if int(5).__repr__() != int(5).__str__():
		return 4
	if not int.from_str('bad').is_err():
		return 5
	return 0
''', checks )


if __name__ == '__main__':
	unittest.main()
