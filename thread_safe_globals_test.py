# Real-compile-and-run tests for PLAN_THREAD_SAFE_SHARED_STATE.md's Part A -
# automatic locking for a module-level RC-typed global that's genuinely
# reassigned from inside a function body (`global X; X = ...`).
#
# Scope note: this covers a plain (non-Optional) RC-typed global reassigned
# and read concurrently - CONFIRMED correct under real concurrent stress
# (see test_concurrent_read_write_stress below). A module-level global whose
# type is a union with more than one non-None member (e.g. `X: SomeClass|
# None`) is NOT yet fully covered on the read side once the compiler has
# narrowed a reference to it to the non-None member (`if X is None: ...;
# return X` - the exact shape lib/datetime.py's localtz() itself uses) -
# narrowing extracts the payload via a separate lowering.py code path
# (_expr_Name's own narrowed-read rewrite) that bypasses cfg.py's assign()
# is_alias branch entirely, so it isn't yet bracketed by the same lock.
# lib/datetime.py's localtz()/lib/termcolor.py's _codes() therefore still
# need (and keep) their own explicit threading.FastLock - do not remove it
# on the assumption this mechanism alone now covers them.
#
# The mechanism itself also only guards a Windows target so far
# (emitter_c.py's _global_lock_supported() - see PLAN_THREAD_SAFE_SHARED_
# STATE.md's own A.3 POSIX-asymmetry note) - confirmed via a real crash
# under WSL/gcc while building this file: a plain global reassigned/read
# under real concurrent stress corrupts the heap there exactly like it
# used to on Windows before this session's fix, since nothing protects it
# on that target yet. test_concurrent_read_write_stress is Windows-only
# for this reason, not because the shape doesn't apply to POSIX too.

import sys
import unittest

import test_support
from test_support import RealCompileMixin


# a plain (non-Optional) RC-typed global, reassigned by several writer
# threads while several reader threads concurrently read it - this is
# exactly the shape that crashed (SIGILL/segfault) reliably before this
# session's cfg.py/lowering.py/emitter_c.py changes: the read side's
# ir.Assign(dest=local, src=global) is its own separate textual read of the
# global in the generated C, independent of whatever the paired ir.Incref
# retained - protecting only the Incref (an earlier, incomplete version of
# this mechanism) still let a concurrent writer swap the global in the gap
# between the two, retaining one object while binding to a different one.
_CONCURRENT_READ_WRITE_STRESS = '''
import compiler
import threading

class Box:
	x: i32
	def __init__( self, x: i32 ) -> None:
		self.x = x

_current: Box = Box( -1 )

def swap( n: i32 ) -> None:
	global _current
	_current = Box( n )

class Reader:
	saw_bad: bool
	def __init__( self ) -> None:
		self.saw_bad = False
	def run( self ) -> None:
		i: i32 = 0
		while i < 2000:
			b: Box = _current
			if b.x < -1:
				self.saw_bad = True
			with compiler.wrap_arithmetic:
				i = i + 1

class Writer:
	def run( self ) -> None:
		i: i32 = 0
		while i < 2000:
			swap( i )
			with compiler.wrap_arithmetic:
				i = i + 1

def main() -> i32:
	readers: list[Reader] = list[Reader]()
	threads: list[threading.Thread] = list[threading.Thread]()
	i: i32 = 0
	while i < 32:
		r: Reader = Reader()
		readers.append( r ).unwrap( 'append failed' )
		threads.append( threading.Thread( r.run ) ).unwrap( 'append failed' )
		with compiler.wrap_arithmetic:
			i = i + 1
	i = 0
	while i < 8:
		w: Writer = Writer()
		threads.append( threading.Thread( w.run ) ).unwrap( 'append failed' )
		with compiler.wrap_arithmetic:
			i = i + 1
	k: usize = 0
	nt: usize = threads.__len__()
	while k < nt:
		th: threading.Thread = threads.__getitem__( k ).unwrap( 'index in bounds' )
		th.join()
		with compiler.wrap_arithmetic:
			k += 1
	j: usize = 0
	nr: usize = readers.__len__()
	while j < nr:
		r2: Reader = readers.__getitem__( j ).unwrap( 'index in bounds' )
		if r2.saw_bad:
			return 1
		with compiler.wrap_arithmetic:
			j += 1
	return 0
'''

# a plain scalar global (no RC leaves at all) reassigned via `global` -
# regression guard for a real bug found while building the RC case above:
# cfg.py's assign() early-returns before ever reaching its own is_global/
# is_alias branches when the destination type has no RC leaves, so no
# ir.AcquireGlobalLock is ever emitted for a scalar global's read/write -
# an earlier version of the read-side fix didn't mirror that early-return
# and emitted an orphaned ir.ReleaseGlobalLock with no matching Acquire.
_SCALAR_GLOBAL_UNAFFECTED = '''
import compiler

_count: i32 = 0

def bump() -> None:
	global _count
	with compiler.wrap_arithmetic:
		_count = _count + 1

def main() -> i32:
	i: i32 = 0
	while i < 1000:
		bump()
		with compiler.wrap_arithmetic:
			i = i + 1
	if _count != 1000:
		return 1
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile tests' )
class ThreadSafeGlobalsTests( RealCompileMixin, unittest.TestCase ):
	@unittest.skipUnless( sys.platform == 'win32', 'Part A only guards a Windows target so far - see this file\'s own header comment' )
	def test_concurrent_read_write_stress( self ) -> None:
		# own executable: real OS threads, must not be merged with other
		# cases via assert_programs_run
		self.assert_programs_run([ ( 'concurrent_read_write_stress', _CONCURRENT_READ_WRITE_STRESS ) ], timeout = 30.0 )

	def test_scalar_global_unaffected( self ) -> None:
		self.assert_programs_run([ ( 'scalar_global_unaffected', _SCALAR_GLOBAL_UNAFFECTED ) ])


if __name__ == '__main__':
	unittest.main()
