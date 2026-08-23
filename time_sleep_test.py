# time_sleep_test.py — real compile+link+run coverage for reactor.sleep(),
# the reactor-aware sleep primitive built on top of reactor.py's own
# timeout()/Signal.Completion machinery, and for time.sleep(), lib/time.py's
# own thin forwarding wrapper around it (local, not top-level, imports -
# see time.py's own sleep() docstring for why: a top-level import there
# closes a real module cycle, time -> reactor -> datetime -> zoneinfo ->
# time - discovery_import_cycle_bug in memory).

import unittest
from pathlib import Path

import test_support
from compiler import Compiler
from discovery import Discovery

@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
class TimeSleepTests( test_support.RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _run( self, code: str ) -> None:
		self.compiler.import_code( code, Path( '__main__.py' ), scope = None )
		self.compiler.run()

	def test_sleep_without_a_reactor_blocks_for_the_duration( self ) -> None:
		self._run( '''
import compiler
import time
import reactor
from datetime import timedelta

def run() -> Result[i32, reactor.WaitError]:
	start: f64 = time.monotonic()
	reactor.sleep( timedelta( milliseconds = 150 )).or_return()
	with compiler.wrap_arithmetic:
		elapsed: f64 = time.monotonic() - start
	if elapsed < 0.12:
		return Result.Ok( 1 )
	return Result.Ok( 0 )

def main() -> i32:
	match run():
		case Result.Ok( code ):
			return code
		case Result.Err( _ ):
			return 90
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0, timeout = 20 )

	def test_sleep_inside_a_reactor_still_takes_at_least_the_duration( self ) -> None:
		self._run( '''
import compiler
import time
import reactor
import atomic
from datetime import timedelta

class Sleeper:
	flag: atomic.Atomic[i32]
	def __init__( self, flag: atomic.Atomic[i32] ) -> None:
		self.flag = flag
	def run( self ) -> None:
		start: f64 = time.monotonic()
		match reactor.sleep( timedelta( milliseconds = 150 )):
			case Result.Ok( _ ):
				with compiler.wrap_arithmetic:
					elapsed: f64 = time.monotonic() - start
				if elapsed >= 0.12:
					self.flag.store( 1 )
				else:
					self.flag.store( 2 )
			case Result.Err( _ ):
				self.flag.store( 3 )

def run() -> i32:
	flag: atomic.Atomic[i32] = atomic.Atomic[i32]( 0 )
	sleeper = Sleeper( flag )
	r: reactor.Reactor = reactor.Reactor( 1 )
	r.spawn( sleeper.run )
	r.run()
	return flag.load()

def main() -> i32:
	code: i32 = run()
	if code != 1:
		return code
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0, timeout = 20 )

	def test_sleep_inside_a_reactor_does_not_block_other_work( self ) -> None:
		''' the actual point of reactor.sleep()'s reactor-aware path: a
		fiber sleeping for a LONG duration must not stall a DIFFERENT
		fiber queued on the SAME single-worker Reactor - if sleep()
		secretly did a real blocking OS sleep even with a reactor present,
		the "busy" fiber below would only ever get scheduled AFTER the
		full sleep duration elapsed. Proven with a timestamp, not just
		"did it eventually run": busy must finish well under half the
		sleep duration after the reactor starts.

		Sleeper is spawned AFTER Busy deliberately - Worker.__pending_tasks
		is drained via list.pop() (LIFO: last spawned runs first), so this
		is what makes Sleeper actually start sleeping BEFORE Busy gets its
		own turn, which is the only way this test can tell "yielded" apart
		from "blocked" at all (spawning them the other way around let Busy
		finish first regardless, since it would run before Sleeper ever
		touched the clock - caught by deliberately breaking sleep() and
		confirming this exact test still passed until the order was fixed). '''
		self._run( '''
import compiler
import time
import reactor
import atomic
from datetime import timedelta

class Sleeper:
	def run( self ) -> None:
		reactor.sleep( timedelta( milliseconds = 400 )).unwrap( 'sleep' )

class Busy:
	start:            f64
	finished_in_time: atomic.Atomic[bool]
	def __init__( self, start: f64, finished_in_time: atomic.Atomic[bool] ) -> None:
		self.start = start
		self.finished_in_time = finished_in_time
	def run( self ) -> None:
		with compiler.wrap_arithmetic:
			elapsed: f64 = time.monotonic() - self.start
		self.finished_in_time.store( elapsed < 0.2 )

def run() -> i32:
	start: f64 = time.monotonic()
	finished_in_time: atomic.Atomic[bool] = atomic.Atomic[bool]( False )
	sleeper = Sleeper()
	busy = Busy( start, finished_in_time )
	r: reactor.Reactor = reactor.Reactor( 1 )
	r.spawn( busy.run )
	r.spawn( sleeper.run )
	r.run()
	if not finished_in_time.load():
		return 1
	return 0

def main() -> i32:
	return run()
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0, timeout = 20 )

	def test_time_sleep_forwards_to_reactor_sleep( self ) -> None:
		''' time.sleep() itself (not reactor.sleep() directly) - proves the
		local-import fix end to end: lib/time.py's own sleep() has local
		`import reactor`/`from datetime import timedelta` satisfying its OWN
		parameter/return-type annotation, which needed a real discovery.py
		fix (local_import_own_signature_annotation) to work at all. '''
		self._run( '''
import compiler
import time
import reactor
from datetime import timedelta

def run() -> Result[i32, reactor.WaitError]:
	start: f64 = time.monotonic()
	time.sleep( timedelta( milliseconds = 150 )).or_return()
	with compiler.wrap_arithmetic:
		elapsed: f64 = time.monotonic() - start
	if elapsed < 0.12:
		return Result.Ok( 1 )
	return Result.Ok( 0 )

def main() -> i32:
	match run():
		case Result.Ok( code ):
			return code
		case Result.Err( _ ):
			return 90
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0, timeout = 20 )

def _emit( compiler: Compiler ) -> str:
	import emitter_c
	return emitter_c.emit_c( compiler )

if __name__ == '__main__':
	unittest.main()
