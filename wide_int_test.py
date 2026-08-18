# Real-compile-and-run behavioral tests for i128/u128 scalar codegen -
# specifically the gaps found while testing the float<->i128/u128 cast fix in
# a prior plan (see emitter_c.py's __metalpy_wideint/__metalpy_wideuint).
# Like float_test.py/int_test.py, these actually compile, link, and RUN a real
# executable - correctness of 128-bit integer codegen can only be confirmed by
# executing the generated code, not by inspecting IR or C source text.
#
# Each MetalPy program under test returns a distinct nonzero i32 exit code per
# failed assertion (0 = every assertion passed); a crash shows up as a
# nonzero/negative code too.

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
from discovery import Discovery, _detect_active_target

_CC = linker_c.detect_cc()
_HAS_I128 = linker_c.has_i128( _CC )


@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping real-compile wide-int tests' )
class WideIntBehaviorTests( unittest.TestCase ):
	def _run_program( self, code: str ) -> subprocess.CompletedProcess:
		''' compiles `code` (a full MetalPy source with its own def main() ->
		i32), links it, runs it, and returns the finished CompletedProcess. '''
		active_target = _detect_active_target()
		active_target['has_i128'] = _HAS_I128
		discovery = Discovery( import_builtins = True, active_target = active_target )
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

			ldflags = ''
			for lib in sorted( compiler.extern_libs ):
				if lib == 'c':
					continue
				flag = linker_c.resolve_lib_ldflag( _CC, lib, compiler.extern_libs[lib] )
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

	# --- Stage 1: integer-constant emission (_emit_const/_emit_wide_int_const) --

	# `i128(1) << 100` (the literal used DIRECTLY as the shift's own operand,
	# no named-variable indirection) used to generate `(1) << (100)` under the
	# DEFAULT (checked) arithmetic mode - the un-cast literal `1` defaults to
	# plain 32-bit `int` in C, so the shift itself was undefined behavior,
	# which then ALSO corrupted ShlCheck's own overflow detection (comparing
	# the garbage shifted-back-down value against the original), causing a
	# false-positive overflow panic even for a perfectly in-range shift.
	# `with compiler.wrap_arithmetic:` was deliberately NOT used here - its
	# own ShlWrap emitter already applies a different cast unrelated to this
	# fix (see emitter_c.py's _emit_shl), so it wouldn't exercise this bug at
	# all; the default checked mode (ShlCheck) is where it actually lives.
	# Compared against a variable-based shift (`one: i128 = 1; one << 100`),
	# which was already correct before this fix (a variable's own declaration
	# already carries the right C type).
	@unittest.skipUnless( _HAS_I128, "MSVC's i128/u128 64-bit fallback doesn't have true 128-bit range - see emitter_c.py's __metalpy_wideint" )
	def test_shift_literal_operand_uses_correct_width( self ) -> None:
		checks = [ 'i128(1) << 100 == a named i128 variable shifted the same way' ]
		self._assert_program_succeeds( '''
def main() -> i32:
	with compiler.panic_arithmetic("ov"):
		one: i128 = 1
		expected: i128 = one << 100
		got: i128 = i128(1) << 100
		if got != expected:
			return 1
	return 0
''', checks )

	# a source literal beyond 64-bit magnitude (but within u128/i128's own real
	# range) used to fail to compile outright - no C token, cast or not, can
	# spell a >64-bit-magnitude integer constant. Expected values below are
	# independently constructed via shifts of a typed variable (not literals),
	# so this test doesn't depend on the fix under test to compute what
	# "correct" even means.
	@unittest.skipUnless( _HAS_I128, "MSVC's i128/u128 64-bit fallback doesn't have true 128-bit range - see emitter_c.py's __metalpy_wideint" )
	def test_large_magnitude_literals_compile_and_round_trip( self ) -> None:
		checks = [
			'positive >64-bit-magnitude u128 literal round-trips',
			'negative >64-bit-magnitude i128 literal round-trips',
		]
		self._assert_program_succeeds( '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		one: u128 = 1
		expected: u128 = (one << 99) + (one << 50) + 1
		got: u128 = 633825300114115826648258445313
		if got != expected:
			return 1

		neg_one: i128 = 1
		neg_expected: i128 = -((neg_one << 100) + 5)
		neg_got: i128 = -1267650600228229401496703205381
		if neg_got != neg_expected:
			return 2
	return 0
''', checks )

	# _emit_wide_int_const's hi/lo-split reconstruction used to hardcode
	# `<< 64` when combining a >64-bit-magnitude i128/u128 constant's two
	# halves - under MSVC, where __metalpy_wideuint is only 64 bits wide (no
	# native 128-bit type), that's a shift-by-width: undefined behavior in C.
	# Fixing the range validation gap (get_intrinsics' has_i128-aware
	# Scalar.sizeof) closes the COMMON path into this (a plain literal
	# assignment now gets rejected before reaching emission), but an explicit
	# bit-reinterpretation cast (`u128(<huge literal>)`) deliberately bypasses
	# that validation on purpose (same mechanism as `u32(-11)`) and can still
	# reach it - confirmed via direct A/B testing under real MSVC: the OLD
	# code made cl.exe itself emit `warning C4293: '<<': shift count negative
	# or too big, undefined behavior` for this exact program; the fix (basing
	# the shift amount on sizeof(__metalpy_wideuint) instead of a hardcoded
	# 64, mirroring _WIDEINT_TOP_BIT_SHIFT's own technique) eliminates that
	# warning entirely. Only meaningful under real MSVC - __metalpy_wideuint
	# genuinely is 128-bit under clang/gcc, so there's nothing to reproduce
	# there.
	@unittest.skipUnless( _CC is not None and _CC.name == 'cl', "only meaningful under real MSVC, where __metalpy_wideuint's 64-bit fallback makes the old hardcoded <<64 a genuine shift-by-width" )
	def test_wide_int_const_hi_lo_split_has_no_shift_ub_under_msvc( self ) -> None:
		code = '''
def main() -> i32:
	x: u128 = u128(340282366920938463463374607431768211455)
	return 0
'''
		active_target = _detect_active_target()
		active_target['has_i128'] = False
		discovery = Discovery( import_builtins = True, active_target = active_target )
		compiler = Compiler( discovery )
		compiler.import_code( code, Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [] )

		no_crt = 'c' not in compiler.extern_libs
		c_source = emitter_c.emit_c( compiler, no_crt = no_crt )
		with tempfile.TemporaryDirectory() as tmp:
			src_path = Path( tmp ) / 'generated.c'
			obj_path = Path( tmp ) / 'generated.o'
			src_path.write_text( c_source, encoding = 'utf-8' )
			cc_result = _CC.compile( src_path, obj_path, no_crt = no_crt )
			self.assertEqual( cc_result.returncode, 0, f'{_CC.name} compile failed:\n{cc_result.stdout}{test_support.c_source_on_failure( c_source )}' )
			self.assertNotIn( 'C4293', cc_result.stdout, f'shift-count UB warning still present:\n{cc_result.stdout}{test_support.c_source_on_failure( c_source )}' )

	# --- Stage 2: saturating arithmetic on i128/u128 (previously NotImplementedError) --

	@unittest.skipUnless( _HAS_I128, "MSVC's i128/u128 64-bit fallback doesn't have true 128-bit range - see emitter_c.py's __metalpy_wideint" )
	def test_saturating_add_sub_mul_i128( self ) -> None:
		checks = [
			'saturating add near i128 MAX clamps to MAX',
			'saturating sub near i128 MIN clamps to MIN',
			'saturating mul overflow clamps to MAX',
		]
		self._assert_program_succeeds( '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		one: i128 = 1
		i128_max: i128 = (one << 127) - 1
		i128_min: i128 = -(one << 127)
	with compiler.saturate_arithmetic:
		near_max: i128 = i128_max - 5
		if ( near_max + 10 ) != i128_max:
			return 1
		near_min: i128 = i128_min + 5
		if ( near_min - 10 ) != i128_min:
			return 2
		if ( i128_max * 2 ) != i128_max:
			return 3
	return 0
''', checks )

	@unittest.skipUnless( _HAS_I128, "MSVC's i128/u128 64-bit fallback doesn't have true 128-bit range - see emitter_c.py's __metalpy_wideint" )
	def test_saturating_add_sub_mul_u128( self ) -> None:
		checks = [
			'saturating add near u128 MAX clamps to MAX',
			'saturating sub below 0 clamps to 0',
			'saturating mul overflow clamps to MAX',
		]
		self._assert_program_succeeds( '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		one: u128 = 1
		u128_max: u128 = ( one << 127 ) * 2 - 1
	with compiler.saturate_arithmetic:
		near_max: u128 = u128_max - 5
		if ( near_max + 10 ) != u128_max:
			return 1
		small: u128 = 3
		if ( small - 10 ) != 0:
			return 2
		if ( u128_max * 2 ) != u128_max:
			return 3
	return 0
''', checks )

	@unittest.skipUnless( _HAS_I128, "MSVC's i128/u128 64-bit fallback doesn't have true 128-bit range - see emitter_c.py's __metalpy_wideint" )
	def test_saturating_shl_i128_u128( self ) -> None:
		checks = [ 'saturating shl overflow on i128 clamps to MAX', 'saturating shl overflow on u128 clamps to MAX' ]
		self._assert_program_succeeds( '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		one: i128 = 1
		i128_max: i128 = (one << 127) - 1
		one_u: u128 = 1
		u128_max: u128 = ( one_u << 127 ) * 2 - 1
	with compiler.saturate_arithmetic:
		big: i128 = 3
		if ( big << 127 ) != i128_max:
			return 1
		big_u: u128 = 3
		if ( big_u << 127 ) != u128_max:
			return 2
	return 0
''', checks )

	@unittest.skipUnless( _HAS_I128, "MSVC's i128/u128 64-bit fallback doesn't have true 128-bit range - see emitter_c.py's __metalpy_wideint" )
	def test_saturating_negate_i128( self ) -> None:
		# negating i128 MIN is the only way signed negation overflows -
		# should saturate to MAX. An ordinary in-range negation is unaffected.
		checks = [ 'saturating -(i128 MIN) clamps to MAX', 'ordinary saturating negation unaffected' ]
		self._assert_program_succeeds( '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		one: i128 = 1
		i128_max: i128 = (one << 127) - 1
		i128_min: i128 = -(one << 127)
	with compiler.saturate_arithmetic:
		if ( -i128_min ) != i128_max:
			return 1
		five: i128 = 5
		if ( -five ) != -5:
			return 2
	return 0
''', checks )

	@unittest.skipUnless( _HAS_I128, "MSVC's i128/u128 64-bit fallback doesn't have true 128-bit range - see emitter_c.py's __metalpy_wideint" )
	def test_saturate_and_checked_cast_involving_i128_u128( self ) -> None:
		checks = [
			'saturating i32->i128 in-range preserves value',
			'saturating u64->u128 in-range preserves value',
			'checked i32->i128 in-range preserves value',
		]
		self._assert_program_succeeds( '''
def main() -> i32:
	with compiler.saturate_arithmetic:
		x: i32 = -12345
		y: i128 = i128(x)
		if y != -12345:
			return 1
		a: u64 = 999
		b: u128 = u128(a)
		if b != 999:
			return 2
	with compiler.panic_arithmetic("ov"):
		c: i32 = -1
		d: i128 = i128(c)
		if d != -1:
			return 3
	return 0
''', checks )

	# a NEGATIVE signed source cast to a u128 TARGET, via T(x) construct-cast
	# syntax - widening (i32, 4 bytes -> u128, 16 bytes), so per the T(x)
	# redesign (confirmed with the user: "fully unconditional... consistency
	# is important, even with a larger blast-radius") this is now an
	# UNCONDITIONAL bit-reinterpretation in every mode, same as any other
	# widening conversion - no clamp, no panic, matching plain two's-
	# complement sign-extension (-5 -> 2**128-5). This REPLACES the old
	# expectation (saturate clamped to 0, panic mode aborted) - that
	# behavior belongs to x.to_u128() now instead (the value-RANGE-checked
	# conversion, independent of width - see compiler.checked_convert/
	# ir.ConvertCheck), not to T(x) (a pure width/bit-pattern operation).
	# TODO once lib/builtins/__scalar_arith.py's .to_T() methods land
	# (Part 2 of the same rollout): add x.to_u128() coverage here for the
	# "negative can't go into unsigned" case this test used to check via
	# T(x) - to_u128() is a fixed, single check (no wrapped/saturated
	# variant - see compiler.checked_convert's own comment), so its
	# ambient-mode-consumption behavior (propagate under checked/wrap/
	# saturate, panic under panic mode) is what should be asserted there,
	# not a numeric clamp.
	@unittest.skipUnless( _HAS_I128, "MSVC's i128/u128 64-bit fallback doesn't have true 128-bit range - see emitter_c.py's __metalpy_wideint" )
	def test_negative_signed_source_into_u128_target( self ) -> None:
		checks = [
			'widening i32(-5) -> u128 sign-extends and reinterprets under saturate mode',
			'widening i32(-5) -> u128 sign-extends and reinterprets under wrap mode',
			'widening i32(-5) -> u128 sign-extends and reinterprets under panic mode',
			'widening i32(-5) -> u128 sign-extends and reinterprets under default checked mode',
		]
		self._assert_program_succeeds( '''
def main() -> i32:
	x: i32 = -5
	expected: u128 = 340282366920938463463374607431768211451
	with compiler.saturate_arithmetic:
		y: u128 = u128(x)
		if y != expected:
			return 1
	with compiler.wrap_arithmetic:
		y2: u128 = u128(x)
		if y2 != expected:
			return 2
	with compiler.panic_arithmetic("ov"):
		y3: u128 = u128(x)
		if y3 != expected:
			return 3
	y4: u128 = u128(x)
	if y4 != expected:
		return 4
	return 0
''', checks )

	# regression test for the u64/usize-SOURCE promotion bug: the u128-source
	# fix below promoted ONLY a literal u128 source to __metalpy_wideuint -
	# but under MSVC's 64-bit __metalpy_wideint fallback, u64/usize sources
	# have the IDENTICAL problem (their own MAX also doesn't fit a SIGNED
	# 64-bit wideint). Confirmed via a real repro: a near-MAX u64 cast down
	# to i32 silently reported "in range" under MSVC instead of overflowing.
	# Deliberately NOT _HAS_I128-gated - u64/usize's own MAX needs no true
	# 128-bit range to construct (unlike the u128 tests below), so this
	# exercises the bug on every backend, MSVC included.
	def test_cast_from_u64_usize_max_saturates_and_panics_correctly( self ) -> None:
		checks = [
			'saturating u64 MAX -> i32 clamps to i32 MAX (not silently in-range)',
			'saturating usize MAX -> i32 clamps to i32 MAX (not silently in-range)',
		]
		self._assert_program_succeeds( '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		one: u64 = 1
		u64_max: u64 = one - 2
		one_sz: usize = 1
		usize_max: usize = one_sz - 2
	with compiler.saturate_arithmetic:
		clamped: i32 = i32(u64_max)
		if clamped != 2147483647:
			return 1
		clamped_sz: i32 = i32(usize_max)
		if clamped_sz != 2147483647:
			return 2
	return 0
''', checks )
		self._assert_program_panics( '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		one: u64 = 1
		u64_max: u64 = one - 2
	with compiler.panic_arithmetic("ov"):
		bad: i32 = i32(u64_max)
	return 0
''' )
		self._assert_program_panics( '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		one: usize = 1
		usize_max: usize = one - 2
	with compiler.panic_arithmetic("ov"):
		bad: i32 = i32(usize_max)
	return 0
''' )

	# regression test for the u128-SOURCE promotion bug: the old code always
	# promoted the source operand to __metalpy_wideint (signed 128-bit) for
	# the range comparison - reinterpreting u128's own MAX (all 128 bits set)
	# as signed gives -1, which the OLD code would have wrongly treated as
	# "in range" for essentially any signed/unsigned narrower target (both
	# the saturate clamp AND the checked-cast overflow detection would have
	# been silently wrong). Confirmed the fix specifically promotes a u128
	# SOURCE to __metalpy_wideuint instead, where this reinterpretation can't
	# happen.
	@unittest.skipUnless( _HAS_I128, "MSVC's i128/u128 64-bit fallback doesn't have true 128-bit range - see emitter_c.py's __metalpy_wideint" )
	def test_cast_from_u128_max_saturates_and_panics_correctly( self ) -> None:
		checks = [
			'saturating u128 MAX -> i32 clamps to i32 MAX (not -1)',
			'saturating u128 MAX -> u32 clamps to u32 MAX (not -1/wrong)',
		]
		self._assert_program_succeeds( '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		one: u128 = 1
		u128_max: u128 = ( one << 127 ) * 2 - 1
	with compiler.saturate_arithmetic:
		clamped: i32 = i32(u128_max)
		if clamped != 2147483647:
			return 1
		clamped_u: u32 = u32(u128_max)
		if clamped_u != 4294967295:
			return 2
	return 0
''', checks )
		self._assert_program_panics( '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		one: u128 = 1
		u128_max: u128 = ( one << 127 ) * 2 - 1
	with compiler.panic_arithmetic("ov"):
		bad: i32 = i32(u128_max)
	return 0
''' )


if __name__ == '__main__':
	unittest.main()
