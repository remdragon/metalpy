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
		# also the regression shape for a real heap-use-after-free found in
		# enable_current_thread()'s own _thread_fiber_handle box (a
		# ThreadLocal[_ThreadFiberHandle] slot): ThreadLocal.set() is
		# deliberately non-owning (a bookmark, correct when the stored
		# value is ALREADY kept alive elsewhere - see current()'s own
		# docstring), but the box had no other owner ANYWHERE, so its own
		# ordinary scope-exit decref freed it the instant enable_current_
		# thread() returned - fixed with a permanent compiler.incref(box),
		# same pattern Fiber.__init__ already uses for self. Silent when
		# nothing reused the freed memory first; a real SwitchToFiber
		# access violation the moment something else's allocation did -
		# Task's own field (forcing a real, differently-sized heap
		# allocation between enable_current_thread() and start()) is what
		# actually exposed it. Do not simplify Task down to a bare
		# no-field class - that would silently stop covering this.
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

	def test_current_returns_the_right_fiber_across_reuse( self ) -> None:
		# regression test for a real heap-use-after-free: current() used to
		# return a BORROWED (non-increfed) alias of the module-level
		# _current bookmark - but this compiler unconditionally treats ANY
		# call's result as a fresh, owned value the instant it's bound to a
		# local, regardless of what the callee's own return statement did.
		# Without current()'s own Incref, a caller like `f = fiber.current()`
		# (exactly park()'s own existing pattern) got its OWN phantom
		# epilogue Decref for `f` with nothing having incremented whatever
		# it pointed at to balance it - confirmed via a real generated-C
		# trace during development (a fresh Call-bound local's own release
		# reading already-freed memory).
		#
		# A precise refcount-delta assertion around this call was tried and
		# dropped: fiber start()/unpark()'s own internal RC bookkeeping
		# (the entry trampoline's one-time self-identification, __switch_in/
		# __switch_out's own _current juggling) turned out to have enough
		# additional, subtle behavior of its own that a hand-derived
		# expected delta kept being wrong in ways unrelated to THIS fix -
		# not confidently resolvable under this investigation's own time
		# budget, flagged separately rather than guessed at here. This test
		# instead exercises the FUNCTIONAL contract current()'s own callers
		# actually depend on - reused across two separate fibers and
		# repeated start() calls (matching how a real Worker's own fiber
		# pool drives things), with no crash and the right identity/count
		# each time being the actual bar.
		self._run( '''
import compiler
import fiber

class Counter:
	n: i32
	def __init__( self ) -> None:
		self.n = 0

class Task:
	tag: i32
	counter: Counter
	def __init__( self, tag: i32, counter: Counter ) -> None:
		self.tag = tag
		self.counter = counter
	def run( self ) -> None:
		cur1 = fiber.current()
		cur2 = fiber.current()
		if cur1 is None or cur2 is None:
			sys.panic( 'no current fiber' )
		with compiler.wrap_arithmetic:
			self.counter.n = self.counter.n + self.tag

def main() -> i32:
	fiber.enable_current_thread()
	counter = Counter()
	fa = fiber.Fiber()
	fb = fiber.Fiber()
	i: usize = 0
	while i < 5:
		ta = Task( 1, counter )
		tb = Task( 10, counter )
		fa.start( ta.run )
		fb.start( tb.run )
		with compiler.wrap_arithmetic:
			i = i + 1
	if counter.n != 55:
		return 1
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0 )

	def test_current_refcount_is_stable_across_repeated_calls( self ) -> None:
		# precise refcount-delta regression, now that current()'s own
		# contract is understood directly (see its own docstring): a bare
		# `return _current` already gets an automatic incref from the same
		# is_alias mechanism `local = _current` would - an EXPLICIT
		# compiler.incref() on top of that (tried once, reverted) was a
		# genuine double-increment. Isolates current() alone (unlike the
		# reuse test above, which exercises the whole start()/unpark()
		# machinery and its own separate RC bookkeeping) - calls it 1000
		# times in a tight loop and checks a SEPARATE, outer-held reference
		# to the same Fiber has an identical refcount before and after: any
		# per-call drift (leak OR under-count) fails this immediately.
		self._run( '''
import compiler
import fiber

class Task:
	held: fiber.Fiber|None
	def __init__( self ) -> None:
		self.held = None
	def run( self ) -> None:
		self.held = fiber.current()
		held = self.held
		if held is None:
			sys.panic( 'no current fiber' )
		rc_before = compiler.refcount( held )
		i: usize = 0
		while i < 1000:
			c = fiber.current()
			if c is None:
				sys.panic( 'no current fiber' )
			with compiler.wrap_arithmetic:
				i = i + 1
		rc_after = compiler.refcount( held )
		if rc_before != rc_after:
			sys.panic( 'fiber.current() refcount drifted across repeated calls' )

def main() -> i32:
	fiber.enable_current_thread()
	f = fiber.Fiber()
	t = Task()
	f.start( t.run )
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0 )

def _emit( compiler: Compiler ) -> str:
	import emitter_c
	return emitter_c.emit_c( compiler )
