# Real-compile-and-run behavioral tests for floating-point support (f32/float,
# f64/double). Like int_test.py (and unlike the type-resolution/emission-shape
# tests in emitter_c_test.py), these actually compile, link, and RUN a real
# executable - floating-point arithmetic, inf/nan handling, and the clamping/
# checked casts can only be confirmed by executing the generated code.
#
# Each MetalPy program under test returns a distinct nonzero i32 exit code per
# failed assertion (0 = every assertion passed); a crash shows up as a
# nonzero/negative code too. The checked-mode FAULT tests instead run under
# `with compiler.panic_arithmetic(...):` and assert the process ABORTS (nonzero
# exit) - the cleanest way to confirm a fault is detected without threading a
# Result return type and .is_err() inspection through every program.
#
# The compile-time-error tests (mixing types, unsupported operators) don't run
# anything - they just assert Discovery collected the expected error.

# stdlib imports:
import os
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


@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping real-compile float tests' )
class FloatBehaviorTests( unittest.TestCase ):
	def _run_program( self, code: str ) -> subprocess.CompletedProcess:
		''' compiles `code` (a full MetalPy source with its own def main() ->
		i32), links it, runs it, and returns the finished CompletedProcess. '''
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		compiler.import_code( code, Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [], 'compile errors:\n' + '\n'.join( str(e) for e in discovery.errors.errors ) )

		no_crt = 'c' not in compiler.extern_libs
		c_source = emitter_c.emit_c( compiler, no_crt = no_crt )

		with tempfile.TemporaryDirectory() as tmp:
			src_path = Path( tmp ) / 'generated.c'
			obj_path = Path( tmp ) / 'generated.o'
			exe_path = Path( tmp ) / 'test_exe.exe'
			src_path.write_text( c_source, encoding = 'utf-8' )

			cc_result = _CC.compile( src_path, obj_path, no_crt = no_crt )
			self.assertEqual( cc_result.returncode, 0, f'{_CC.name} compile failed:\n{cc_result.stdout}{test_support.c_source_on_failure( c_source )}' )

			# every extern library the program pulled in needs an explicit link
			# flag ('c' is the CRT, handled by no_crt). A no-CRT program on
			# Windows still calls SetConsoleOutputCP/ExitProcess, so it needs
			# kernel32 - normally pulled in transitively by a program that uses
			# builtins, but these float-only programs use none, so add it here.
			libs = set( compiler.extern_libs )
			if no_crt and os.name == 'nt':
				libs.add( 'kernel32' )
			ldflags = ''
			for lib in sorted( libs ):
				if lib == 'c':
					continue
				flag = f'{lib}.lib' if _CC.name == 'cl' else f'-l{lib}'
				ldflags = ldflags + f' {flag}' if ldflags else flag

			link_result = _CC.link( exe_path, [ obj_path ], ldflags = ldflags, no_crt = no_crt )
			self.assertEqual( link_result.returncode, 0, f'{_CC.name} link failed:\n{link_result.stdout}' )

			return subprocess.run( [ str( exe_path ) ], capture_output = True )

	def _assert_program_succeeds( self, code: str, check_names: list[str] ) -> None:
		''' runs `code` and asserts it exited 0. check_names[i] (0-indexed)
		names whatever assertion inside the program returns i+1 on failure. '''
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

	def _assert_program_panics( self, code: str ) -> None:
		''' runs `code` (which returns 0 on its own normal path) and asserts it
		did NOT exit 0 - i.e. a checked-mode fault fired sys.panic. '''
		result = self._run_program( code )
		self.assertNotEqual(
			result.returncode, 0,
			f'expected the program to panic (nonzero exit) but it exited 0\nstdout: {result.stdout}\nstderr: {result.stderr}',
		)

	def _assert_compile_error( self, code: str, needle: str ) -> None:
		''' asserts compiling `code` collects at least one error containing
		`needle` (a compile-time rejection, no run). '''
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		try:
			compiler.import_code( code, Path( '__main__.py' ), scope = None )
			compiler.run()
		except Exception:
			pass # some failures raise rather than accumulate - the error text check below still applies via str()
		joined = '\n'.join( str(e) for e in discovery.errors.errors )
		self.assertIn( needle, joined, f'expected a compile error containing {needle!r}, got:\n{joined or "(no errors)"}' )

	def _c_source( self, code: str ) -> str:
		''' compile `code` all the way to generated C (no real cc invocation) -
		for structural assertions on emitter output (e.g. the widening remap). '''
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		compiler.import_code( code, Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [], 'compile errors:\n' + '\n'.join( str(e) for e in discovery.errors.errors ) )
		return emitter_c.emit_c( compiler, no_crt = 'c' not in compiler.extern_libs )

	# --- arithmetic, unary negate, comparisons (wrap mode = raw IEEE) --------

	def test_arithmetic_and_comparisons( self ) -> None:
		checks = [
			'f64 add', 'f64 sub', 'f64 mul', 'f64 div (exact 3/2)', 'f64 unary neg',
			'f32 add', 'f32 mul', 'f32 unary neg',
			'f64 <', 'f64 >=', 'f64 == self', 'f64 != other',
			'chained expression 2*(a+b)',
		]
		self._assert_program_succeeds( '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		a: f64 = 3.0
		b: f64 = 2.0
		if a + b != 5.0:
			return 1
		if a - b != 1.0:
			return 2
		if a * b != 6.0:
			return 3
		if a / b != 1.5:
			return 4
		if -a != -3.0:
			return 5
		c: f32 = 1.5
		d: f32 = 2.5
		if c + d != 4.0:
			return 6
		if c * d != 3.75:
			return 7
		if -c != -1.5:
			return 8
		if not ( a < 4.0 ):
			return 9
		if not ( a >= 3.0 ):
			return 10
		if not ( a == 3.0 ):
			return 11
		if not ( a != 2.0 ):
			return 12
		if 2.0 * ( a + b ) != 10.0:
			return 13
	return 0
''', checks )

	# --- default literal typing + both spellings interchangeable ------------

	def test_default_typing_and_aliases( self ) -> None:
		checks = [
			'bare 3.14 defaults to f64 (assignable to an f64)',
			'`float` spelling equals `f32` (same type, adds fine)',
			'`double` spelling equals `f64`',
			'f32 value flows into an f32-typed slot via the `float` alias',
		]
		self._assert_program_succeeds( '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		x = 3.14
		y: f64 = x
		if y != 3.14:
			return 1
		a: float = 1.5
		b: f32 = 2.5
		if a + b != 4.0:
			return 2
		p: double = 3.0
		q: f64 = 4.0
		if p + q != 7.0:
			return 3
		r: f32 = a
		if r != 1.5:
			return 4
	return 0
''', checks )

	# --- wrap/saturate mode: raw IEEE, inf/nan produced silently ------------

	def test_wrap_mode_inf_and_nan( self ) -> None:
		checks = [
			'x/0.0 yields +inf (no error in wrap mode)',
			'+inf compares greater than a huge finite value',
			'0.0/0.0 yields nan',
			'nan != nan (IEEE)',
			'nan compared with < is false',
		]
		self._assert_program_succeeds( '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		one: f64 = 1.0
		zero: f64 = 0.0
		inf: f64 = one / zero
		if inf != inf:
			return 1
		if not ( inf > 1.0e307 ):
			return 2
		nan: f64 = zero / zero
		if nan == nan:
			return 3
		if not ( nan != nan ):
			return 4
		if nan < 1.0:
			return 5
	return 0
''', checks )

	# --- casts: int<->float, float<->float, clamping float->int -------------

	def test_casts( self ) -> None:
		checks = [
			'int->f64 exact', 'int->f32 exact', 'f64->f32 narrowing',
			'f32->f64 widening', 'in-range f64->i32 truncates toward zero',
			'negative f64->i32 truncates toward zero',
			'wrap-mode clamp: i8(1000.0) == 127 (above range)',
			'wrap-mode clamp: i8(-1000.0) == -128 (below range)',
			'wrap-mode clamp: i32(nan) == 0',
		]
		self._assert_program_succeeds( '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		i: i32 = 7
		if f64(i) != 7.0:
			return 1
		if f32(i) != 7.0:
			return 2
		big: f64 = 3.0
		if f32(big) != 3.0:
			return 3
		small: f32 = 2.0
		if f64(small) != 2.0:
			return 4
		v: f64 = 5.9
		if i32(v) != 5:
			return 5
		n: f64 = -5.9
		if i32(n) != -5:
			return 6
		over: f64 = 1000.0
		if i8(over) != 127:
			return 7
		under: f64 = -1000.0
		if i8(under) != -128:
			return 8
		zero: f64 = 0.0
		nan: f64 = zero / zero
		if i32(nan) != 0:
			return 9
	return 0
''', checks )

	# --- compiler.sizeof ----------------------------------------------------

	def test_sizeof( self ) -> None:
		self._assert_program_succeeds( '''
def main() -> i32:
	if compiler.sizeof(f32) != 4:
		return 1
	if compiler.sizeof(f64) != 8:
		return 2
	if compiler.sizeof(float) != 4:
		return 3
	if compiler.sizeof(double) != 8:
		return 4
	return 0
''', [ 'sizeof(f32)==4', 'sizeof(f64)==8', 'sizeof(float)==4', 'sizeof(double)==8' ] )

	# --- checked/panic-mode arithmetic: correct values, faults abort --------

	def test_checked_mode_valid_values( self ) -> None:
		# panic_arithmetic keeps every op checked (inf/nan -> panic) but never
		# fires here since all results stay finite - confirms the checked
		# opcodes compute the right values, not just that they detect faults
		self._assert_program_succeeds( '''
def main() -> i32:
	with compiler.panic_arithmetic("fp"):
		a: f64 = 3.0
		b: f64 = a + 2.0
		if b != 5.0:
			return 1
		cc: f64 = a * b
		if cc != 15.0:
			return 2
		n: i32 = i32(cc)
		if n != 15:
			return 3
	return 0
''', [ 'checked add value', 'checked mul value', 'checked in-range f64->i32 cast' ] )

	def test_checked_overflow_to_inf_panics( self ) -> None:
		self._assert_program_panics( '''
def main() -> i32:
	with compiler.panic_arithmetic("fp"):
		x: f64 = 1.0e308
		y: f64 = x * x
	return 0
''' )

	def test_checked_division_by_zero_panics( self ) -> None:
		self._assert_program_panics( '''
def main() -> i32:
	with compiler.panic_arithmetic("fp"):
		z: f64 = 0.0
		one: f64 = 1.0
		y: f64 = one / z
	return 0
''' )

	def test_checked_cast_out_of_range_panics( self ) -> None:
		self._assert_program_panics( '''
def main() -> i32:
	with compiler.panic_arithmetic("fp"):
		big: f64 = 1.0e300
		n: i32 = i32(big)
	return 0
''' )

	def test_checked_cast_nan_panics( self ) -> None:
		self._assert_program_panics( '''
def main() -> i32:
	with compiler.panic_arithmetic("fp"):
		z: f64 = 0.0
		nan: f64 = z / z
		n: i32 = i32(nan)
	return 0
''' )

	# --- compile-time rejections --------------------------------------------

	def test_mixed_float_int_variable_is_error( self ) -> None:
		self._assert_compile_error( '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		a: f64 = 1.0
		b: i32 = 2
		c: f64 = a + b
	return 0
''', 'same type' )

	def test_mixed_float_widths_is_error( self ) -> None:
		self._assert_compile_error( '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		a: f64 = 1.0
		b: f32 = 2.0
		c: f64 = a + b
	return 0
''', 'same type' )

	def test_float_floordiv_is_error( self ) -> None:
		self._assert_compile_error( '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		a: f64 = 1.0
		b: f64 = 2.0
		c: f64 = a // b
	return 0
''', 'not supported on floating-point' )

	def test_float_bitwise_is_error( self ) -> None:
		self._assert_compile_error( '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		a: f64 = 1.0
		b: f64 = 2.0
		c: f64 = a & b
	return 0
''', 'not supported on floating-point' )

	def test_float_literal_into_int_is_error( self ) -> None:
		# a float literal hinted to an integer type is a silent-truncation trap
		self._assert_compile_error( '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		i: i32 = 5
		c: i32 = i + 1.5
	return 0
''', 'floating-point literal cannot be used' )

	def test_float_literal_annotated_int_is_error( self ) -> None:
		self._assert_compile_error( '''
def main() -> i32:
	x: i32 = 1.5
	return 0
''', 'floating-point literal cannot be used' )

	def test_float_literal_explicit_cast_truncates( self ) -> None:
		# an EXPLICIT cast of a float literal to an int is allowed (it's the
		# float->int conversion mechanism) and truncates toward zero at compile
		# time - only the IMPLICIT hint (i + 1.5, x: i32 = 1.5) is rejected
		self._assert_program_succeeds( '''
def main() -> i32:
	n: i32 = i32(1.5)
	if n != 1:
		return 1
	m: i32 = i32(-2.9)
	if m != -2:
		return 2
	u: u8 = u8(3.9)
	if u != 3:
		return 3
	return 0
''', [ 'i32(1.5) == 1', 'i32(-2.9) == -2 (trunc toward zero)', 'u8(3.9) == 3' ] )

	def test_float_literal_into_float_is_ok( self ) -> None:
		# the guard must NOT fire for a float target, an int-literal->float, or
		# a bare (defaults-f64) literal
		self._assert_program_succeeds( '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		f: f32 = 1.0
		g: f32 = f + 1.5
		if g != 2.5:
			return 1
		h: f64 = f64(1)
		if h != 1.0:
			return 2
		x = 3.25
		if x != 3.25:
			return 3
	return 0
''', [ 'f32 + 1.5 literal', 'f64(1) int->float', 'bare 3.25 defaults f64' ] )

	# --- error-union widening (the general capability) ----------------------

	def test_widening_end_to_end( self ) -> None:
		# ONE declared error union covers three ops with DIFFERENT narrower
		# errors: float / (ZeroDivisionError|FloatingPointError), float + and a
		# float->int cast (FloatingPointError), and int // (ZeroDivisionError|
		# OverflowError). Each widens into the union at its OrReturn. Proves the
		# coverage check admits the mix AND the emitted C is valid and runs.
		checks = [ 'success path returns the right Ok value', 'the Err arm must not run on all-finite inputs' ]
		self._assert_program_succeeds( '''
def compute( a: f64, b: f64, n: i32, d: i32 ) -> Result[i32, ZeroDivisionError | OverflowError | FloatingPointError]:
	q: f64 = a / b
	s: f64 = q + a
	m: i32 = n // d
	return Result.Ok( m + i32(s) )

def main() -> i32:
	r: Result[i32, ZeroDivisionError | OverflowError | FloatingPointError] = compute( 6.0, 2.0, 10, 5 )
	match r:
		case Result.Ok( v ):
			if v != 11:
				return 1
		case Result.Err( e ):
			return 2
	return 0
''', checks )

	def test_widening_propagates_error( self ) -> None:
		# a divide-by-zero on the float / makes compute() return Err (the
		# ZeroDivisionError widened into the 3-error union); main sees is_err
		self._assert_program_succeeds( '''
def compute( a: f64, b: f64 ) -> Result[i32, ZeroDivisionError | OverflowError | FloatingPointError]:
	q: f64 = a / b
	return Result.Ok( i32(q) )

def main() -> i32:
	r: Result[i32, ZeroDivisionError | OverflowError | FloatingPointError] = compute( 1.0, 0.0 )
	if not r.is_err():
		return 1
	ok: Result[i32, ZeroDivisionError | OverflowError | FloatingPointError] = compute( 8.0, 2.0 )
	if ok.is_err():
		return 2
	return 0
''', [ 'divide-by-zero must widen to Err', 'finite division must be Ok' ] )

	def test_widening_remap_is_emitted( self ) -> None:
		# structural check of _emit_widen_error's TAG remap specifically (the
		# arithmetic errors it exercises here are zero-payload markers, so
		# there's no payload to observe behaviorally through them - see
		# test_widening_preserves_error_payload below for that, using a real
		# user error class instead). A single-error op (float +) widening into
		# the 3-union sets the inner variant tag; a union-error op (float /)
		# widening emits a switch.
		src = self._c_source( '''
def compute( a: f64, b: f64 ) -> Result[f64, ZeroDivisionError | OverflowError | FloatingPointError]:
	s: f64 = a + b
	q: f64 = a / b
	return Result.Ok( s + q )

def main() -> i32:
	r: Result[f64, ZeroDivisionError | OverflowError | FloatingPointError] = compute( 6.0, 2.0 )
	if r.is_err():
		return 1
	return 0
''' )
		# float / produces a 2-member union (ZeroDivisionError|FloatingPointError)
		# widened into the 3-member one -> a runtime tag remap switch
		self.assertIn( 'switch', src )
		# 3-union members are ascii-sorted: FloatingPointError=0, OverflowError=1,
		# ZeroDivisionError=2. The float + raises only FloatingPointError, so its
		# single->union widen sets the inner tag to 0; the / switch remaps its
		# ZeroDivisionError arm to 2 (a non-identity mapping, the whole point)
		self.assertRegex( src, r'v_Err\.tag = 2' )

	def test_widening_preserves_error_payload( self ) -> None:
		# or_return() widening is a GENERAL mechanism, not arithmetic-specific -
		# a user error class can carry real fields, and _emit_widen_error must
		# copy that payload, not just remap the tag (the actual gap this test
		# guards: a tag-only widen would leave the payload field reading
		# uninitialized/zeroed data on the far side of the union boundary).
		# Two levels of widening are exercised: inner()'s single ParseError
		# widens into outer()'s 2-member union (the single->union branch of
		# _emit_widen_error), then outer()'s own Result widens again into
		# outermost()'s 3-member union (the union->union switch branch) -
		# the payload must survive both.
		checks = [
			'level-1 widen (single class -> union): payload survives',
			'level-2 widen (union -> wider union): payload still survives',
			'the Ok path (no widening) is unaffected',
		]
		self._assert_program_succeeds( '''
class ParseError:
	code: i32

class OtherError:
	pass

class ThirdError:
	pass

def inner( bad: bool ) -> Result[i32, ParseError]:
	if bad:
		return Result.Err( ParseError( code = 42 ) )
	return Result.Ok( 7 )

def outer( bad: bool ) -> Result[i32, ParseError | OtherError]:
	v: i32 = inner( bad ).or_return()
	return Result.Ok( v )

def outermost( bad: bool ) -> Result[i32, ParseError | OtherError | ThirdError]:
	v: i32 = outer( bad ).or_return()
	return Result.Ok( v )

def main() -> i32:
	r: Result[i32, ParseError | OtherError] = outer( True )
	match r:
		case Result.Ok( v ):
			return 1
		case Result.Err( e ):
			err: ParseError | OtherError = e
			match err:
				case ParseError( pe ):
					if pe.code != 42:
						return 1
				case OtherError( oe ):
					return 1

	r2: Result[i32, ParseError | OtherError | ThirdError] = outermost( True )
	match r2:
		case Result.Ok( v ):
			return 2
		case Result.Err( e2 ):
			err2: ParseError | OtherError | ThirdError = e2
			match err2:
				case ParseError( pe2 ):
					if pe2.code != 42:
						return 2
				case OtherError( oe2 ):
					return 2
				case ThirdError( te2 ):
					return 2

	r3: Result[i32, ParseError | OtherError] = outer( False )
	match r3:
		case Result.Ok( v ):
			if v != 7:
				return 3
		case Result.Err( e3 ):
			return 3
	return 0
''', checks )

	# --- checked float division catches BOTH errors (item 1 + item 4) -------

	def test_checked_float_div_inf_result_panics( self ) -> None:
		# a finite/finite division that overflows to inf is a FloatingPointError
		# even though the divisor is nonzero (division must check its RESULT)
		self._assert_program_panics( '''
def main() -> i32:
	with compiler.panic_arithmetic("fp"):
		big: f64 = 1.0e308
		small: f64 = 1.0e-308
		q: f64 = big / small
	return 0
''' )

	def test_checked_float_div_nan_operand_panics( self ) -> None:
		# a nan produced in a wrap block, then divided in a checked block: the
		# quotient is nan -> FloatingPointError. Proves checked code does NOT
		# assume its operands are already finite (item 4)
		self._assert_program_panics( '''
def main() -> i32:
	nan: f64 = 0.0
	with compiler.wrap_arithmetic:
		z: f64 = 0.0
		nan = z / z
	with compiler.panic_arithmetic("fp"):
		two: f64 = 2.0
		q: f64 = nan / two
	return 0
''' )

	# --- signed INT_MIN/-1 is defined per mode (item 2) ---------------------

	def test_int_min_div_checked_panics( self ) -> None:
		# checked/panic: INT_MIN / -1 (and INT_MIN % -1) is an OverflowError
		self._assert_program_panics( '''
def main() -> i32:
	with compiler.panic_arithmetic("ov"):
		neg_one: i8 = -1
		mn: i8 = i8(128)
		q: i8 = mn // neg_one
	return 0
''' )
		self._assert_program_panics( '''
def main() -> i32:
	with compiler.panic_arithmetic("ov"):
		neg_one: i8 = -1
		mn: i8 = i8(128)
		m: i8 = mn % neg_one
	return 0
''' )

	def test_int_min_div_wrap_and_saturate( self ) -> None:
		# wrap: INT_MIN/-1 -> INT_MIN (no error); saturate -> INT_MAX; mod -> 0.
		# Division is always zero-checked, so each op still yields a Result the
		# helper propagates - the mode only changes the INT_MIN/-1 value
		self._assert_program_succeeds( '''
def wdiv( a: i8, b: i8 ) -> Result[i8, ZeroDivisionError]:
	with compiler.wrap_arithmetic:
		return Result.Ok( a // b )

def wmod( a: i8, b: i8 ) -> Result[i8, ZeroDivisionError]:
	with compiler.wrap_arithmetic:
		return Result.Ok( a % b )

def sdiv( a: i8, b: i8 ) -> Result[i8, ZeroDivisionError]:
	with compiler.saturate_arithmetic:
		return Result.Ok( a // b )

def main() -> i32:
	mn: i8 = i8(128)
	neg_one: i8 = i8(255)
	wq: Result[i8, ZeroDivisionError] = wdiv( mn, neg_one )
	match wq:
		case Result.Ok( v ):
			if v != mn:
				return 1
		case Result.Err( e ):
			return 2
	wm: Result[i8, ZeroDivisionError] = wmod( mn, neg_one )
	match wm:
		case Result.Ok( v ):
			if v != 0:
				return 3
		case Result.Err( e ):
			return 4
	sq: Result[i8, ZeroDivisionError] = sdiv( mn, neg_one )
	match sq:
		case Result.Ok( v ):
			if v != 127:
				return 5
		case Result.Err( e ):
			return 6
	return 0
''', [ 'wrap INT_MIN//-1 == INT_MIN', 'wrap div Err (unexpected)', 'wrap INT_MIN%-1 == 0', 'wrap mod Err (unexpected)', 'saturate INT_MIN//-1 == INT_MAX', 'saturate div Err (unexpected)' ] )

	def test_unsigned_division_has_no_overflow_error( self ) -> None:
		# unsigned division can only raise ZeroDivisionError (no INT_MIN/-1) -
		# a function returning just Result[u8, ZeroDivisionError] must suffice
		self._assert_program_succeeds( '''
def udiv( a: u8, b: u8 ) -> Result[u8, ZeroDivisionError]:
	return Result.Ok( a // b )

def main() -> i32:
	r: Result[u8, ZeroDivisionError] = udiv( 200, 4 )
	match r:
		case Result.Ok( v ):
			if v != 50:
				return 1
		case Result.Err( e ):
			return 2
	return 0
''', [ 'unsigned 200//4 == 50', 'unsigned div Err (unexpected)' ] )


if __name__ == '__main__':
	unittest.main()
