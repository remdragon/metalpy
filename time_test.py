# Real-compile-and-run behavioral test for lib/time.py (time.monotonic() /
# time.time()). A high-resolution timer can only be confirmed by executing the
# generated code, so this compiles + links + runs a real executable via the
# shared test_support.RealCompileMixin (no copy-pasted harness).
#
# The program returns 0 when every check passes and a distinct nonzero i32 exit
# code per failed check; RealCompileMixin decodes that back to the failing check.
# print() is deliberately avoided - the stdout global-init path is unrelated
# work-in-progress (see PLAN_GLOBAL_INIT.md), so results come back via exit code.
#
# Which OS backend runs follows the build host (there is no cross-compile):
# QueryPerformanceCounter / GetSystemTimePreciseAsFileTime on Windows,
# clock_gettime on POSIX.

import unittest

import test_support
from test_support import RealCompileMixin

# A failed check returns its 1-based index; test_support decodes it via check
# ordering below. now-bounds make this test inherently time-bounded (as any
# wall-clock assertion is): valid roughly 2020-09 .. ~2096.
_TIME_BEHAVIOR = '''
import time

def main() -> i32:
	start: f64 = time.monotonic()

	acc: i64 = 0
	with compiler.wrap_arithmetic:
		for i in range( 5_000_000 ):
			acc += i

	end: f64 = time.monotonic()
	now: f64 = time.time()

	with compiler.wrap_arithmetic:
		elapsed: f64 = end - start

	if end < start:            # monotonic() ran backwards
		return 1
	if elapsed <= 0.0:         # no measurable elapsed time across real work
		return 2
	if now < 1.6e9:            # wall clock before 2020-09 - implausible
		return 3
	if now > 4.0e9:            # wall clock after ~2096 - implausible
		return 4
	if now < elapsed:          # wall clock dwarfs a sub-second delta (ordering sanity)
		return 5
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile time tests' )
class TimeBehaviorTests( RealCompileMixin, unittest.TestCase ):
	def test_monotonic_and_wall_clock( self ) -> None:
		self.assert_programs_run([ ( 'monotonic_and_wall_clock', _TIME_BEHAVIOR ) ])


if __name__ == '__main__':
	unittest.main()
