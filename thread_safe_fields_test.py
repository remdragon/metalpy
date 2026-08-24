# Real-compile-and-run tests for PLAN_THREAD_SAFE_SHARED_STATE.md's Part B -
# a per-object lock (ObjectHeader's own new field) protecting every RC-typed
# instance field's GetAttr/SetAttr the same way Part A already protects
# module globals.
#
# This file currently covers only the functional refcount-accounting side
# (single-threaded, no real concurrency) - real concurrent stress tests for
# Part B (mirroring thread_safe_globals_test.py's own multi-thread
# read/write races) are still needed before Part B is considered verified
# the way Part A is; see PLAN_THREAD_SAFE_SHARED_STATE.md's own Status
# section.

import sys
import unittest

import test_support
from test_support import RealCompileMixin


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


if __name__ == '__main__':
	unittest.main()
