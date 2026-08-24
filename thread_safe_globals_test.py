# Real-compile-and-run tests for PLAN_THREAD_SAFE_SHARED_STATE.md's Part A -
# automatic locking for a module-level RC-typed global that's genuinely
# reassigned from inside a function body (`global X; X = ...`).
#
# Scope: covers both a plain (non-Optional) RC-typed global (test_
# concurrent_read_write_stress) AND a narrowed read of a union-typed global
# (test_narrowed_read_concurrent_stress) - the exact `if X is None: X =
# compute(); return X` shape lib/datetime.py's localtz()/lib/termcolor.py's
# _codes() themselves use, closed in a follow-up to this mechanism's first
# pass (narrowing extracts a union's payload via a separate lowering.py
# code path - _expr_Name's own narrowed-read rewrite - that used to bypass
# the lock entirely; see lowering.py's own comment on that branch for the
# real crash that motivated fixing it). Both are CONFIRMED correct under
# real concurrent stress. lib/datetime.py/lib/termcolor.py still keep their
# own explicit threading.FastLock regardless - this mechanism being solid
# doesn't obligate removing a working, already-verified guard, and Part B
# (instance fields) remains completely unimplemented, so plenty of other
# racy shapes still need it.
#
# The mechanism now guards both a Windows target (SRWLOCK) and a Linux one
# (pthread_mutex_t, emitter_c.py's _global_lock_supported() - see
# PLAN_THREAD_SAFE_SHARED_STATE.md's own A.3 POSIX-asymmetry note for why
# Linux's own storage/init shape genuinely differs from Windows's, not just
# a platform #ifdef). macOS remains unguarded (and untested - this repo's
# only verified targets are Windows-x64/Linux-x64) - _global_lock_supported()
# deliberately excludes it rather than assuming pthread_mutex_t behaves
# identically there. Both stress tests below now run on sys.platform in
# ('win32', 'linux') - widened from the earlier Windows-only gating, still
# excluding macOS/anything else for the same untested-target reason.
# The Linux leg was confirmed to be a REAL fix, not a no-op that happens to
# pass: temporarily sabotaging _global_lock_supported() to exclude 'linux'
# and re-running under WSL/gcc reproduced real SIGILL crashes (3/30 runs)
# with the exact same signature as the original bug this whole mechanism
# exists to close - restored immediately after confirming that.

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
		readers.append( r )
		threads.append( threading.Thread( r.run ) )
		with compiler.wrap_arithmetic:
			i = i + 1
	i = 0
	while i < 8:
		w: Writer = Writer()
		threads.append( threading.Thread( w.run ) )
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

# the narrowed-read shape - lib/datetime.py's localtz()/lib/termcolor.py's
# _codes() themselves, minus the manual FastLock, to prove the automatic
# mechanism alone now covers it: `if X is None: X = compute()`, then a
# SEPARATE, later read of X (narrowed to non-None) - the exact pattern the
# original crash this whole investigation started from was found in.
_NARROWED_READ_CONCURRENT_STRESS = '''
import compiler
import threading

class Box:
	x: i32
	def __init__( self, x: i32 ) -> None:
		self.x = x

_g: Box|None = None

def localish() -> Box:
	# deliberately no manual FastLock - tests whether Part A alone protects
	# this exact shape with no hand-written lock at all
	global _g
	if _g is None:
		_g = Box( 42 )
	b: Box = _g
	return b

class Worker:
	saw_bad: bool
	def __init__( self ) -> None:
		self.saw_bad = False
	def run( self ) -> None:
		i: i32 = 0
		while i < 3000:
			b: Box = localish()
			if b.x != 42:
				self.saw_bad = True
			with compiler.wrap_arithmetic:
				i = i + 1

def main() -> i32:
	workers: list[Worker] = list[Worker]()
	threads: list[threading.Thread] = list[threading.Thread]()
	i: i32 = 0
	while i < 64:
		w: Worker = Worker()
		workers.append( w )
		threads.append( threading.Thread( w.run ) )
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
	nw: usize = workers.__len__()
	while j < nw:
		w2: Worker = workers.__getitem__( j ).unwrap( 'index in bounds' )
		if w2.saw_bad:
			return 1
		with compiler.wrap_arithmetic:
			j += 1
	return 0
'''


# a global's OWN initializing write racing a thread it spawns itself (via an
# ordinary function call in its own initializer, legal per SYNTAX.md) which
# reassigns that SAME global through the normal locked `global X; X = ...'
# path while __metalpy_init() is still running - closes the gap
# cfg.py's assign_global_initializer/lowering.py's run_global fix for:
# previously _g's own init write bypassed cfg.assign() (and therefore the
# lock) entirely, racing the writer thread's fully-locked writes.
_GLOBAL_INIT_WRITE_RACE_STRESS = '''
import compiler
import threading
from atomic import Atomic

# tracks live Box count - a lost decref (the leak half of the race: neither
# write path decrefs the value the OTHER one just clobbered) shows up here
# as live count > 1 even after only one Box is actually reachable through _g
_live: Atomic[i32] = Atomic[i32]( 0 )

class Box:
	x: i32
	def __init__( self, x: i32 ) -> None:
		self.x = x
		_live.fetch_add( 1 )
	def __del__( self ) -> None:
		_live.fetch_add( -1 )

_writer_thread: threading.Thread|None = None

class Writer:
	def run( self ) -> None:
		global _g
		i: i32 = 0
		while i < 200000:
			_g = Box( i )
			with compiler.wrap_arithmetic:
				i = i + 1

def spawn_writer() -> Box:
	global _writer_thread
	w: Writer = Writer()
	t: threading.Thread = threading.Thread( w.run )
	_writer_thread = t
	# spin to give the writer thread a real chance to be mid-loop by the time
	# this function returns and the caller (_g's own initializing write,
	# outside this function) runs - widens the race window instead of
	# relying on the writer losing the OS scheduler race entirely
	j: i32 = 0
	while j < 50000:
		with compiler.wrap_arithmetic:
			j = j + 1
	return Box( -1 )

_g: Box = spawn_writer()

def main() -> i32:
	if _writer_thread is None:
		return 1
	_writer_thread.join()
	i: i32 = 0
	while i < 2000:
		b: Box = _g
		if b.x < -1:
			return 1
		with compiler.wrap_arithmetic:
			i = i + 1
		# only the Box currently reachable through _g should still be alive -
		# a lost decref from the race leaks the clobbered value, live > 1
		if _live.load() != 1:
			return 2
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile tests' )
class ThreadSafeGlobalsTests( RealCompileMixin, unittest.TestCase ):
	@unittest.skipUnless( sys.platform in ( 'win32', 'linux' ), 'Part A only guards Windows/Linux targets - see this file\'s own header comment' )
	def test_concurrent_read_write_stress( self ) -> None:
		# own executable: real OS threads, must not be merged with other
		# cases via assert_programs_run
		self.assert_programs_run([ ( 'concurrent_read_write_stress', _CONCURRENT_READ_WRITE_STRESS ) ], timeout = 30.0 )

	def test_scalar_global_unaffected( self ) -> None:
		self.assert_programs_run([ ( 'scalar_global_unaffected', _SCALAR_GLOBAL_UNAFFECTED ) ])

	@unittest.skipUnless( sys.platform in ( 'win32', 'linux' ), 'Part A only guards Windows/Linux targets - see this file\'s own header comment' )
	def test_narrowed_read_concurrent_stress( self ) -> None:
		# own executable: real OS threads, must not be merged with other
		# cases via assert_programs_run
		self.assert_programs_run([ ( 'narrowed_read_concurrent_stress', _NARROWED_READ_CONCURRENT_STRESS ) ], timeout = 30.0 )

	@unittest.skipUnless( sys.platform in ( 'win32', 'linux' ), 'Part A only guards Windows/Linux targets - see this file\'s own header comment' )
	def test_global_init_write_race_stress( self ) -> None:
		# own executable: real OS threads, must not be merged with other
		# cases via assert_programs_run
		self.assert_programs_run([ ( 'global_init_write_race_stress', _GLOBAL_INIT_WRITE_RACE_STRESS ) ], timeout = 30.0 )


if __name__ == '__main__':
	unittest.main()
