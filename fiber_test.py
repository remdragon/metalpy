# fiber_test.py — real compile+link+run coverage for lib/fiber.py.
#
# Every case here exercises the actual OS-native context-switch primitives
# (Windows Fiber API / POSIX ucontext) through real compiled MetalPy code -
# not just IR-level "it compiles", per this repo's own verification
# convention (see memory: "verify, don't trust IR-level success").

import unittest
from pathlib import Path

import test_support
from compiler import Compiler
from discovery import Discovery

@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
class FiberTests( test_support.RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _run( self, code: str ) -> None:
		self.compiler.import_code( code, Path( '__main__.py' ), scope = None )
		self.compiler.run()

	def test_single_start_runs_task_to_completion( self ) -> None:
		self._run( '''
import fiber

class Task:
	ran: i32

	def __init__( self ) -> None:
		self.ran = 0

	def run( self ) -> None:
		self.ran = 1

def main() -> i32:
	fiber.enable_current_thread()
	f = fiber.Fiber()
	t = Task()
	f.start( t.run )
	if t.ran == 1:
		return 0
	return 1
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0 )

	def test_park_and_unpark_resumes_exactly_where_it_left_off( self ) -> None:
		self._run( '''
import fiber

class Task:
	step: i32

	def __init__( self ) -> None:
		self.step = 0

	def run( self ) -> None:
		self.step = 1
		fiber.park()
		self.step = 2
		fiber.park()
		self.step = 3

def main() -> i32:
	fiber.enable_current_thread()
	f = fiber.Fiber()
	t = Task()

	f.start( t.run )
	if t.step != 1:
		return 1

	f.unpark()
	if t.step != 2:
		return 2

	f.unpark()
	if t.step != 3:
		return 3

	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0 )

	def test_fiber_starting_another_fiber_tracks_caller_correctly( self ) -> None:
		# the harder case: __caller must be tracked per-call, not
		# hardcoded to "the OS thread that started all this" - fiber A
		# starts fiber B, B parks, A unparks B, B finishes and control
		# must correctly unwind all the way back to A, then to main()
		self._run( '''
import fiber

class Log:
	trace: str

	def __init__( self ) -> None:
		self.trace = ""

class Inner:
	log: Log

	def __init__( self, log: Log ) -> None:
		self.log = log

	def run( self ) -> None:
		self.log.trace = self.log.trace + "B1"
		fiber.park()
		self.log.trace = self.log.trace + "B2"

class Outer:
	log: Log
	inner_fiber: fiber.Fiber

	def __init__( self, log: Log, inner_fiber: fiber.Fiber ) -> None:
		self.log = log
		self.inner_fiber = inner_fiber

	def run( self ) -> None:
		self.log.trace = self.log.trace + "A1"
		inner = Inner( self.log )
		self.inner_fiber.start( inner.run )
		self.log.trace = self.log.trace + "A2"
		self.inner_fiber.unpark()
		self.log.trace = self.log.trace + "A3"

def main() -> i32:
	fiber.enable_current_thread()
	log = Log()
	fiber_a = fiber.Fiber()
	fiber_b = fiber.Fiber()
	outer = Outer( log, fiber_b )

	fiber_a.start( outer.run )

	if log.trace == "A1B1A2B2A3":
		return 0
	return 1
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0 )

def _emit( compiler: Compiler ) -> str:
	import emitter_c
	return emitter_c.emit_c( compiler )
