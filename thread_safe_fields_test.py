# Real-compile-and-run tests for PLAN_THREAD_SAFE_SHARED_STATE.md's Part B -
# a per-object lock (ObjectHeader's own new field) protecting every RC-typed
# instance field's GetAttr/SetAttr the same way Part A already protects
# module globals.
#
# Scope: covers a plain (non-Optional) RC-typed field shared across threads
# via a common object (test_concurrent_field_read_write_stress), a narrowed
# read of a union-typed field - the exact `if self.g is None: self.g =
# compute(); return self.g` lazy-init shape thread_safe_globals_test.py's
# own narrowed-GLOBAL-read test covers, here reached through a field instead
# (test_narrowed_field_read_concurrent_stress) - and a plain scalar field
# (test_scalar_field_unaffected, a functional-only regression guard for the
# identical "no RC leaves -> no lock, ever" early-return Part A's own scalar
# global test already covers). Also covers the two bugs fixed after this
# file's own single-threaded refcount tests below were first added: the
# loop-condition per-iteration leak and the pthread.h header-ordering bug -
# both stress tests below run real while-loop conditions reading a field
# thousands of times per thread, so a regression in either would very likely
# reproduce here too, not just in the narrower single-threaded tests.
#
# Same platform gating as thread_safe_globals_test.py, same reasoning:
# Windows (SRWLOCK) and Linux (pthread_mutex_t) are this repo's only real,
# compiler-verified targets; macOS is poisoned (emitter_c._global_lock_
# supported()'s own docstring) rather than silently assumed correct.

import sys
import unittest

import test_support
from test_support import RealCompileMixin


# a plain (non-Optional) RC-typed field on a SHARED object, reassigned by
# several writer threads while several reader threads concurrently read it -
# the field-shaped analogue of thread_safe_globals_test.py's own
# test_concurrent_read_write_stress, now stressing Part B's per-object lock
# (obj->$header.lock) instead of Part A's per-global one.
_CONCURRENT_FIELD_READ_WRITE_STRESS = '''
import compiler
import threading

class Box:
	x: i32
	def __init__( self, x: i32 ) -> None:
		self.x = x

class Holder:
	current: Box
	def __init__( self, initial: Box ) -> None:
		self.current = initial

def swap( h: Holder, n: i32 ) -> None:
	h.current = Box( n )

class Reader:
	h: Holder
	saw_bad: bool
	def __init__( self, h: Holder ) -> None:
		self.h = h
		self.saw_bad = False
	def run( self ) -> None:
		i: i32 = 0
		while i < 2000:
			b: Box = self.h.current
			if b.x < -1:
				self.saw_bad = True
			with compiler.wrap_arithmetic:
				i = i + 1

class Writer:
	h: Holder
	def __init__( self, h: Holder ) -> None:
		self.h = h
	def run( self ) -> None:
		i: i32 = 0
		while i < 2000:
			swap( self.h, i )
			with compiler.wrap_arithmetic:
				i = i + 1

def main() -> i32:
	h: Holder = Holder( Box( -1 ) )
	readers: list[Reader] = list[Reader]()
	threads: list[threading.Thread] = list[threading.Thread]()
	i: i32 = 0
	while i < 32:
		r: Reader = Reader( h )
		readers.append( r )
		threads.append( threading.Thread( r.run ) )
		with compiler.wrap_arithmetic:
			i = i + 1
	i = 0
	while i < 8:
		w: Writer = Writer( h )
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

# a plain scalar field (no RC leaves at all) written via SetAttr from many
# threads - regression guard for the field-shaped version of Part A's own
# "no RC leaves -> no lock, ever" early return (cfg.rc_leaves(attr_var.type)
# gating both the read and write chokepoints in lowering.py) - functional
# only, no real race to observe here (i32 SetAttr is not made atomic by
# this mechanism, same accepted "may lose an update, never corrupts memory"
# posture as Part A's own scalar case), just confirms nothing about Part B
# breaks a scalar field's own ordinary codegen.
_SCALAR_FIELD_UNAFFECTED = '''
import compiler

class Counter:
	n: i32
	def __init__( self ) -> None:
		self.n = 0

def bump( c: Counter ) -> None:
	with compiler.wrap_arithmetic:
		c.n = c.n + 1

def main() -> i32:
	c: Counter = Counter()
	i: i32 = 0
	while i < 1000:
		bump( c )
		with compiler.wrap_arithmetic:
			i = i + 1
	if c.n != 1000:
		return 1
	return 0
'''

# the narrowed-read shape reached through a FIELD instead of a global -
# `if self.g is not None: b: Box = self.g` (a genuinely-supported field-
# chain narrowed read - "chained narrow SUBJECT shipped" landed this for
# self.a.b at any depth) read repeatedly from many threads while a SEPARATE
# writer thread concurrently reassigns the SAME field via a plain, ordinary
# `self.g = Box(n)` SetAttr - the field-shaped analogue of thread_safe_
# globals_test.py's own test_narrowed_read_concurrent_stress, stressing
# _expr_Attribute's is_narrowed branch (Acquire, GetAttr, extract payload,
# Incref, Release, all one critical section) against a concurrent writer,
# without relying on this compiler's own separate "assign
# inside `if x is None:`, narrow on the merged path afterward" field shape -
# g is never reassigned to None here, so the narrowed branch is
# unconditionally taken on every read; a torn/freed read would surface as
# a garbage/incorrect .x value, caught below.
_NARROWED_FIELD_READ_CONCURRENT_STRESS = '''
import compiler
import threading

class Box:
	x: i32
	def __init__( self, x: i32 ) -> None:
		self.x = x

class Lazy:
	g: Box|None
	def __init__( self ) -> None:
		self.g = Box( 42 )
	def read_if_present( self ) -> i32:
		if self.g is not None:
			b: Box = self.g
			return b.x
		return -1

class Writer:
	lz: Lazy
	def __init__( self, lz: Lazy ) -> None:
		self.lz = lz
	def run( self ) -> None:
		i: i32 = 0
		while i < 3000:
			self.lz.g = Box( 42 )
			with compiler.wrap_arithmetic:
				i = i + 1

class Reader:
	lz: Lazy
	saw_bad: bool
	def __init__( self, lz: Lazy ) -> None:
		self.lz = lz
		self.saw_bad = False
	def run( self ) -> None:
		i: i32 = 0
		while i < 3000:
			v: i32 = self.lz.read_if_present()
			if v != 42:
				self.saw_bad = True
			with compiler.wrap_arithmetic:
				i = i + 1

def main() -> i32:
	lz: Lazy = Lazy()
	readers: list[Reader] = list[Reader]()
	threads: list[threading.Thread] = list[threading.Thread]()
	i: i32 = 0
	while i < 32:
		r: Reader = Reader( lz )
		readers.append( r )
		threads.append( threading.Thread( r.run ) )
		with compiler.wrap_arithmetic:
			i = i + 1
	i = 0
	while i < 8:
		w: Writer = Writer( lz )
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


# regression guard for a real, confirmed leak: a while-loop's own CONDITION
# expression is lowered exactly ONCE (Python-level), but the resulting C
# instructions sit physically inside the loop and re-execute every real
# iteration - Part B's retain-on-read for a chained field receiver
# (self.a.b) emitted its own Incref there, but the matching decref (driven
# by the ordinary once-per-STATEMENT pending-temps flush) only ever fired
# once, after the whole while-statement finished lowering. Confirmed via
# compiler.refcount(): a captured Box read through a generator's own
# `while i < b.v:` condition leaked one reference per resumption before the
# generator was dropped mid-iteration - fixed by flushing the condition's
# own pending temps every iteration (lowering.py's _stmt_While, right
# before its own JumpIfFalse).
_WHILE_CONDITION_FIELD_READ_REFCOUNT_STABLE = '''
class Counter:
	n: usize
	def __init__( self, n: usize ) -> None:
		self.n = n

class Holder:
	c: Counter
	def __init__( self, c: Counter ) -> None:
		self.c = c

def count_up_while_under( h: Holder, limit: usize ) -> usize:
	i: usize = 0
	while i < h.c.n:
		with compiler.wrap_arithmetic:
			i += 1
	return i

def main() -> i32:
	c: Counter = Counter( 50 )
	h: Holder = Holder( c )
	rc0: usize = compiler.refcount( c )
	i: usize = 0
	while i < 20:
		result: usize = count_up_while_under( h, 50 )
		if result != 50:
			return 1
		with compiler.wrap_arithmetic:
			i += 1
	rc1: usize = compiler.refcount( c )
	if rc1 != rc0:
		return 2
	return 0
'''

# the exact generator shape the leak was first found in - a generator's own
# state machine re-checks its while-loop's condition once per __next__()
# resumption, reading a captured/promoted RC field each time, then gets
# dropped (destructed) mid-iteration without ever exhausting the loop.
_GENERATOR_DROPPED_MID_LOOP_CONDITION_REFCOUNT_STABLE = '''
class Box:
	v: usize
	def __init__( self, v: usize ) -> None:
		self.v = v

def gen( b: Box ) -> Iterator[Result[usize, StopIteration]]:
	i: usize = 0
	while i < b.v:
		yield i
		with compiler.wrap_arithmetic:
			i += 1

def make_and_partially_consume( b: Box ) -> None:
	g = gen( b )
	first = g.__next__().is_ok()
	second = g.__next__().is_ok() # b.v is 10 - only 2 of 10 iterations consumed
	if first and second: pass

def main() -> i32:
	with compiler.wrap_arithmetic:
		b: Box = Box( v = 10 )
	rc0: usize = compiler.refcount( b )
	i: usize = 0
	while i < 20:
		make_and_partially_consume( b )
		with compiler.wrap_arithmetic:
			i += 1
	rc1: usize = compiler.refcount( b )
	if rc1 != rc0:
		return 1
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile tests' )
class ThreadSafeFieldsTests( RealCompileMixin, unittest.TestCase ):
	def test_while_condition_field_read_refcount_stable( self ) -> None:
		self.assert_programs_run([
			( 'while_condition_field_read_refcount_stable', _WHILE_CONDITION_FIELD_READ_REFCOUNT_STABLE ),
		])

	def test_generator_dropped_mid_loop_condition_refcount_stable( self ) -> None:
		self.assert_programs_run([
			( 'generator_dropped_mid_loop_condition_refcount_stable', _GENERATOR_DROPPED_MID_LOOP_CONDITION_REFCOUNT_STABLE ),
		])

	@unittest.skipUnless( sys.platform in ( 'win32', 'linux' ), 'Part B only guards Windows/Linux targets - see this file\'s own header comment' )
	@test_support.skip_unless_load_tests
	def test_concurrent_field_read_write_stress( self ) -> None:
		# own executable: real OS threads, must not be merged with other
		# cases via assert_programs_run
		self.assert_programs_run([ ( 'concurrent_field_read_write_stress', _CONCURRENT_FIELD_READ_WRITE_STRESS ) ], timeout = 30.0 )

	def test_scalar_field_unaffected( self ) -> None:
		self.assert_programs_run([ ( 'scalar_field_unaffected', _SCALAR_FIELD_UNAFFECTED ) ])

	@unittest.skipUnless( sys.platform in ( 'win32', 'linux' ), 'Part B only guards Windows/Linux targets - see this file\'s own header comment' )
	@test_support.skip_unless_load_tests
	def test_narrowed_field_read_concurrent_stress( self ) -> None:
		# own executable: real OS threads, must not be merged with other
		# cases via assert_programs_run
		self.assert_programs_run([ ( 'narrowed_field_read_concurrent_stress', _NARROWED_FIELD_READ_CONCURRENT_STRESS ) ], timeout = 30.0 )


if __name__ == '__main__':
	unittest.main()
