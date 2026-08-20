# reactor_test.py — real compile+link+run coverage for lib/reactor.py.

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
import reactor
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
	w = reactor.Worker()
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
import reactor
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
	w = reactor.Worker()
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
		# single-worker: Reactor.run() DOES put this Worker on its own
		# freshly-spawned thread (a real cross-thread handoff, exercising
		# that path) - just never more than one such thread at once. See
		# test_reactor_with_multiple_concurrent_workers below for the
		# num_workers>1 case (fiber.py's _current/_thread_fiber_handle are
		# real ThreadLocal[T] now, not plain globals - safe for that).
		self._run( '''
import compiler
import reactor
import atomic

class CountingTask:
	counter: atomic.Atomic[i32]
	def __init__( self, counter: atomic.Atomic[i32] ) -> None:
		self.counter = counter
	def run( self ) -> None:
		self.counter.fetch_add( 1 )

def main() -> i32:
	counter = atomic.Atomic[i32]( 0 )
	r = reactor.Reactor( 1 )
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

	def test_reactor_with_multiple_concurrent_workers( self ) -> None:
		# num_workers=8, several hundred tasks round-robined across them,
		# each task itself parking once mid-run (forcing a REAL fiber
		# switch on whichever OS thread happens to be running it) - the
		# actual scenario fiber.py's ThreadLocal[T] conversion
		# (_current/_thread_fiber_handle) exists for: 8 real OS threads
		# each independently switching fibers via SwitchToFiber/swapcontext
		# at the same time. Before that conversion, this shape would race
		# on plain globals shared across every thread - atomic.Atomic[i32]
		# for the shared counter (not a plain i32 - the counter itself
		# being thread-safe is a separate, already-established concern
		# from whether fiber-switching itself races).
		self._run( '''
import compiler
import reactor
import fiber
import atomic

class CountingTask:
	counter: atomic.Atomic[i32]
	def __init__( self, counter: atomic.Atomic[i32] ) -> None:
		self.counter = counter
	def run( self ) -> None:
		self.counter.fetch_add( 1 )
		fiber.park()
		self.counter.fetch_add( 10 )

def main() -> i32:
	counter = atomic.Atomic[i32]( 0 )
	r = reactor.Reactor( 8 )
	n: usize = 400
	i: usize = 0
	while i < n:
		t = CountingTask( counter )
		r.spawn( t.run )
		with compiler.wrap_arithmetic:
			i = i + 1
	r.run()
	with compiler.panic_arithmetic( 'unexpected count overflow' ):
		expected: i32 = i32( n ) * 11
	if counter.load() != expected:
		return 1
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0 )

def _emit( compiler: Compiler ) -> str:
	import emitter_c
	return emitter_c.emit_c( compiler )
