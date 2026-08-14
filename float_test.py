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
			self.assertEqual( cc_result.returncode, 0, f'{_CC.name} compile failed:\n{cc_result.stdout}\n\n--- generated.c ---\n{c_source}' )

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


if __name__ == '__main__':
	unittest.main()
