# mp_reactor_test.py — real compile+link+run coverage for lib/mp_reactor.py.

import unittest
from pathlib import Path

import test_support
from compiler import Compiler
from discovery import Discovery

@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
class ReactorTests( test_support.RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _run( self, code: str ) -> None:
		self.compiler.import_code( code, Path( '__main__.py' ), scope = None )
		self.compiler.run()

	def test_worker_runs_a_simple_task( self ) -> None:
		self._run( '''
import compiler
import mp_reactor
import fiber

class Counter:
	n: i32
	def __init__( self ) -> None:
		self.n = 0

class SimpleTask:
	counter: Counter
	def __init__( self, counter: Counter ) -> None:
		self.counter = counter
	def run( self ) -> None:
		with compiler.wrap_arithmetic:
			self.counter.n = self.counter.n + 1

def main() -> i32:
	fiber.enable_current_thread()
	w = mp_reactor.Worker()
	counter = Counter()
	a = SimpleTask( counter )
	w.schedule( a.run )
	w.run_until_idle()
	if counter.n != 1:
		return 1
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0 )

	def test_worker_parks_and_requeues_across_two_drains( self ) -> None:
		self._run( '''
import compiler
import mp_reactor
import fiber

class Counter:
	n: i32
	def __init__( self ) -> None:
		self.n = 0

class SimpleTask:
	counter: Counter
	def __init__( self, counter: Counter ) -> None:
		self.counter = counter
	def run( self ) -> None:
		with compiler.wrap_arithmetic:
			self.counter.n = self.counter.n + 1

class ParkingTask:
	counter: Counter
	def __init__( self, counter: Counter ) -> None:
		self.counter = counter
	def run( self ) -> None:
		with compiler.wrap_arithmetic:
			self.counter.n = self.counter.n + 1
		fiber.park()
		with compiler.wrap_arithmetic:
			self.counter.n = self.counter.n + 10

def main() -> i32:
	fiber.enable_current_thread()
	w = mp_reactor.Worker()
	counter = Counter()
	a = SimpleTask( counter )
	b = ParkingTask( counter )
	w.schedule( a.run )
	w.schedule( b.run )
	w.run_until_idle()
	if counter.n != 2:
		return 1
	w.run_until_idle()
	if counter.n != 12:
		return 2
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0 )

	def test_reactor_spawns_on_its_own_worker_thread( self ) -> None:
		# num_workers=1 deliberately, not a stand-in for "should be higher" -
		# fiber.py's _current/_thread_fiber_handle are plain globals, correct
		# only because every fiber test so far (including this one) has a
		# single OS thread ever touching them. Reactor.run() DOES put this
		# Worker on its own freshly-spawned thread (a real cross-thread
		# handoff, exercising that path) - just never more than one such
		# thread at once. num_workers>1 needs those globals to become a
		# real ThreadLocal[T] first (not built yet - see fiber.py's own
		# _current comment) to be genuinely safe, not just "didn't crash in
		# this particular test."
		self._run( '''
import compiler
import mp_reactor
import atomic

class CountingTask:
	counter: atomic.Atomic[i32]
	def __init__( self, counter: atomic.Atomic[i32] ) -> None:
		self.counter = counter
	def run( self ) -> None:
		self.counter.fetch_add( 1 )

def main() -> i32:
	counter = atomic.Atomic[i32]( 0 )
	r = mp_reactor.Reactor( 1 )
	i: usize = 0
	while i < 20:
		t = CountingTask( counter )
		r.spawn( t.run )
		with compiler.wrap_arithmetic:
			i = i + 1
	r.run()
	if counter.load() != 20:
		return 1
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0 )

def _emit( compiler: Compiler ) -> str:
	import emitter_c
	return emitter_c.emit_c( compiler )
