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

	def test_current_worker_identifies_the_driving_worker( self ) -> None:
		# None on a thread that hasn't driven any Worker yet; the exact
		# SAME Worker instance (not just "some Worker") once inside a task
		# it's driving - the whole point of current_worker() being an
		# ambient lookup rather than just a bool "am I inside a worker".
		self._run( '''
import reactor
import sys

class IdentityTask:
	worker: reactor.Worker
	saw_self: bool
	def __init__( self, worker: reactor.Worker ) -> None:
		self.worker = worker
		self.saw_self = False
	def run( self ) -> None:
		cw = reactor.current_worker()
		if cw is None:
			sys.panic( 'current_worker() returned None inside a running task' )
		self.saw_self = cw is self.worker

def main() -> i32:
	if reactor.current_worker() is not None:
		return 1
	w = reactor.Worker()
	t = IdentityTask( w )
	w.schedule( t.run )
	w.run_until_idle()
	if not t.saw_self:
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

	def test_signal_wait_resumes_only_after_real_readiness( self ) -> None:
		# real TCP loopback pair - a task parks on reactor.wait_for_signal()
		# for the accepted connection's own fd, must NOT resume before any
		# data is written (first run_until_idle() call only starts+parks
		# it), and must resume within a bounded number of further ticks
		# once the other end sends real data (the first of those ticks
		# notices readiness via Worker.__check_signals's own non-blocking
		# poller peek and moves the fiber to __ready_to_unpark; the NEXT
		# tick actually unparks it - see run_until_idle()'s own docstring
		# on why that's two ticks, not one).
		self._run( '''
import socket
import poller
import reactor

class WaitTask:
	fd: poller.SOCKET
	resumed: bool
	def __init__( self, fd: poller.SOCKET ) -> None:
		self.fd = fd
		self.resumed = False
	def run( self ) -> None:
		sig = reactor.fd_signal( self.fd, True, False )
		reactor.wait_for_signal( sig ).unwrap( 'unexpected shutdown during test' )
		self.resumed = True

def run() -> Result[i32, OSError]:
	server = socket.Socket.tcp().or_return()
	server.bind( '127.0.0.1', u16( 0 )).or_return()
	server.listen().or_return()
	bound = server.getsockname().or_return()

	client = socket.Socket.tcp().or_return()
	client.connect( '127.0.0.1', bound.port() ).or_return()
	( conn, _addr ) = server.accept().or_return()

	poller.set_nonblocking( conn.fileno() ).or_return()

	w = reactor.Worker()
	t = WaitTask( conn.fileno() )
	w.schedule( t.run )
	w.run_until_idle()
	if t.resumed:
		return Result.Ok( 1 )

	msg: bytes = b'hi'
	client.send_all( msg.get_const_ptr(), usize( 2 )).or_return()

	i: usize = 0
	while i < 20:
		w.run_until_idle()
		if t.resumed:
			return Result.Ok( 0 )
		with compiler.wrap_arithmetic:
			i = i + 1
	return Result.Ok( 2 )

def main() -> i32:
	match run():
		case Result.Ok( code ):
			return code
		case Result.Err( _ ):
			return 3
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0 )

	def test_wait_for_signal_blocks_without_a_reactor( self ) -> None:
		# no Worker driving this thread at all - wait_for_signal() must
		# fall back to a real, standalone blocking wait (own throwaway
		# Poller) rather than trying to park a fiber into nothing. A
		# background thread writes to the connection's peer; the main
		# thread's wait_for_signal() call is expected to actually block
		# until that real write lands, then return (both epoll and
		# WSAPoll are level-triggered - already-ready-when-registered and
		# becomes-ready-while-waiting are indistinguishable and both
		# correct here, so this is inherently race-free regardless of
		# which side of that the writer thread happens to land on).
		self._run( '''
import socket
import poller
import reactor
import threading

class Writer:
	sock: socket.Socket
	def __init__( self, sock: socket.Socket ) -> None:
		self.sock = sock
	def run( self ) -> None:
		msg: bytes = b'hi'
		self.sock.send_all( msg.get_const_ptr(), usize( 2 )).unwrap( 'writer send failed' )

def run() -> Result[i32, OSError]:
	server = socket.Socket.tcp().or_return()
	server.bind( '127.0.0.1', u16( 0 )).or_return()
	server.listen().or_return()
	bound = server.getsockname().or_return()

	client = socket.Socket.tcp().or_return()
	client.connect( '127.0.0.1', bound.port() ).or_return()
	( conn, _addr ) = server.accept().or_return()

	poller.set_nonblocking( conn.fileno() ).or_return()

	if reactor.current_worker() is not None:
		return Result.Ok( 1 )

	w = Writer( client )
	t = threading.Thread( w.run )   # constructing already launches it

	sig = reactor.fd_signal( conn.fileno(), True, False )
	reactor.wait_for_signal( sig ).unwrap( 'unexpected shutdown during test' )
	t.join()
	return Result.Ok( 0 )

def main() -> i32:
	match run():
		case Result.Ok( code ):
			return code
		case Result.Err( _ ):
			return 2
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0 )

	def test_spawn_wakes_a_genuinely_blocked_worker( self ) -> None:
		# the crux of the whole blocking-drain_fully() design: a single
		# worker parks Task1 on a signal that's deliberately withheld, so
		# its queues drain to nothing else and it genuinely BLOCKS inside
		# run_until_idle(-1)'s own poller.wait() - real Reactor.run(), not
		# manual ticking. From a SEPARATE thread, after a bounded busy
		# delay (no sleep() primitive exists - see busy_delay()) gives the
		# worker a head start to actually reach that blocking call,
		# Reactor.spawn() enqueues Task2 - if Worker.schedule()'s own
		# wake-pair poke didn't exist (or were broken), Task2 would sit in
		# __pending_tasks completely unprocessed until Task1's own signal
		# ALSO happens to fire (queued work never gets lost, just
		# arbitrarily delayed) - which would make this test pass anyway on
		# a broken implementation via that "eventual, not prompt" path,
		# proving nothing. To rule that out, the driving (main) thread
		# busy-polls counter.load() for Task2's own completion BEFORE ever
		# satisfying Task1's signal - if the wake mechanism genuinely
		# works, this observes counter==1 well before the poll's own
		# bound, while Task1's own signal is still deliberately
		# unsatisfied. Sanity-checked directly during development:
		# temporarily removing Worker.schedule()'s own wake-pair write
		# made this exact test correctly fail (exit code 1, not a silent
		# pass or a hang) - confirms it actually discriminates.
		self._run( '''
import compiler
import socket
import poller
import reactor
import atomic
import threading

def busy_delay() -> None:
	i: usize = 0
	while i < usize( 200000000 ):
		with compiler.wrap_arithmetic:
			i = i + 1

class Task1:
	fd: poller.SOCKET
	counter: atomic.Atomic[i32]
	def __init__( self, fd: poller.SOCKET, counter: atomic.Atomic[i32] ) -> None:
		self.fd = fd
		self.counter = counter
	def run( self ) -> None:
		sig = reactor.fd_signal( self.fd, True, False )
		reactor.wait_for_signal( sig ).unwrap( 'unexpected shutdown during test' )
		self.counter.fetch_add( 100 )

class Task2:
	counter: atomic.Atomic[i32]
	def __init__( self, counter: atomic.Atomic[i32] ) -> None:
		self.counter = counter
	def run( self ) -> None:
		self.counter.fetch_add( 1 )

class LateWork:
	r: reactor.Reactor
	counter: atomic.Atomic[i32]
	def __init__( self, r: reactor.Reactor, counter: atomic.Atomic[i32] ) -> None:
		self.r = r
		self.counter = counter
	def run( self ) -> None:
		busy_delay()
		t2 = Task2( self.counter )
		self.r.spawn( t2.run )

class ReactorRunner:
	r: reactor.Reactor
	def __init__( self, r: reactor.Reactor ) -> None:
		self.r = r
	def run( self ) -> None:
		self.r.run()

def run() -> Result[i32, OSError]:
	server = socket.Socket.tcp().or_return()
	server.bind( '127.0.0.1', u16( 0 )).or_return()
	server.listen().or_return()
	bound = server.getsockname().or_return()
	client = socket.Socket.tcp().or_return()
	client.connect( '127.0.0.1', bound.port() ).or_return()
	( conn, _addr ) = server.accept().or_return()
	poller.set_nonblocking( conn.fileno() ).or_return()

	counter = atomic.Atomic[i32]( 0 )
	r = reactor.Reactor( 1 )
	t1 = Task1( conn.fileno(), counter )
	r.spawn( t1.run )

	runner = ReactorRunner( r )
	t_run = threading.Thread( runner.run )

	late = LateWork( r, counter )
	t_late = threading.Thread( late.run )

	i: usize = 0
	saw_task2: bool = False
	while i < usize( 400000000 ):
		if counter.load() == 1:
			saw_task2 = True
			break
		with compiler.wrap_arithmetic:
			i = i + 1

	msg: bytes = b'go'
	client.send_all( msg.get_const_ptr(), usize( 2 )).or_return()

	t_run.join()
	t_late.join()

	if not saw_task2:
		return Result.Ok( 1 )
	if counter.load() != 101:
		return Result.Ok( 2 )
	return Result.Ok( 0 )

def main() -> i32:
	match run():
		case Result.Ok( code ):
			return code
		case Result.Err( _ ):
			return 3
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0, timeout = 20 )

	def test_shutdown_interrupts_a_genuinely_blocked_worker( self ) -> None:
		# a task parks on a signal that's DELIBERATELY never satisfied (the
		# connection's peer never writes anything) - the single worker
		# genuinely blocks in run_until_idle(-1), same as test_spawn_wakes_
		# a_genuinely_blocked_worker above. A separate thread, after a
		# bounded busy delay (no sleep() primitive - see busy_delay()),
		# calls Reactor.shutdown(). Verifies wait_for_signal() returns
		# Result.Err(ShutdownError) - not Ok, not a hang - and that
		# Reactor.run() itself returns promptly (bounded by this test's
		# own harness timeout as a safety net, in case shutdown is
		# broken).
		self._run( '''
import compiler
import socket
import poller
import reactor
import atomic
import threading

def busy_delay() -> None:
	i: usize = 0
	while i < usize( 200000000 ):
		with compiler.wrap_arithmetic:
			i = i + 1

class ShutdownTask:
	fd: poller.SOCKET
	result_flag: atomic.Atomic[i32]
	def __init__( self, fd: poller.SOCKET, result_flag: atomic.Atomic[i32] ) -> None:
		self.fd = fd
		self.result_flag = result_flag
	def run( self ) -> None:
		sig = reactor.fd_signal( self.fd, True, False )
		match reactor.wait_for_signal( sig ):
			case Result.Ok( _ ):
				self.result_flag.store( 1 )
			case Result.Err( _ ):
				self.result_flag.store( 2 )

class Shutter:
	r: reactor.Reactor
	def __init__( self, r: reactor.Reactor ) -> None:
		self.r = r
	def run( self ) -> None:
		busy_delay()
		self.r.shutdown()

def run() -> Result[i32, OSError]:
	server = socket.Socket.tcp().or_return()
	server.bind( '127.0.0.1', u16( 0 )).or_return()
	server.listen().or_return()
	bound = server.getsockname().or_return()
	client = socket.Socket.tcp().or_return()
	client.connect( '127.0.0.1', bound.port() ).or_return()
	( conn, _addr ) = server.accept().or_return()
	poller.set_nonblocking( conn.fileno() ).or_return()

	result_flag = atomic.Atomic[i32]( 0 )
	r = reactor.Reactor( 1 )
	t = ShutdownTask( conn.fileno(), result_flag )
	r.spawn( t.run )

	shutter = Shutter( r )
	t_shutter = threading.Thread( shutter.run )

	r.run()
	t_shutter.join()

	if result_flag.load() != 2:
		return Result.Ok( 1 )
	return Result.Ok( 0 )

def main() -> i32:
	match run():
		case Result.Ok( code ):
			return code
		case Result.Err( _ ):
			return 3
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0, timeout = 20 )

	def test_shutdown_wakes_every_worker( self ) -> None:
		# num_workers=2, one signal-waiting task per worker (round-robin
		# spawn() lands one on each), both signals deliberately never
		# satisfied - Reactor.shutdown() must wake BOTH workers, not just
		# whichever one happens to be checked "first" internally.
		#
		# NOTE: make_conn_pair() returns (conn, client) and the caller
		# MUST keep both alive for the test's own duration - a real bug
		# surfaced by this exact test during development: Socket.__del__
		# closes the real OS socket, so a client left to go out of scope
		# immediately closes itself, which makes the SERVER-accepted conn
		# on the other end look "readable" (a real EOF/HUP condition, not
		# a fake one) - both tasks came back Result.Ok (not the expected
		# Err(ShutdownError)) until client was kept alive in run()'s own
		# scope for the whole test. Not a reactor.py bug - purely a test-
		# authoring footgun worth documenting so it isn't hit again.
		self._run( '''
import compiler
import socket
import poller
import reactor
import atomic
import threading

def busy_delay() -> None:
	i: usize = 0
	while i < usize( 200000000 ):
		with compiler.wrap_arithmetic:
			i = i + 1

def make_conn_pair() -> Result[tuple[socket.Socket, socket.Socket], OSError]:
	server = socket.Socket.tcp().or_return()
	server.bind( '127.0.0.1', u16( 0 )).or_return()
	server.listen().or_return()
	bound = server.getsockname().or_return()
	client = socket.Socket.tcp().or_return()
	client.connect( '127.0.0.1', bound.port() ).or_return()
	( conn, _addr ) = server.accept().or_return()
	poller.set_nonblocking( conn.fileno() ).or_return()
	return Result.Ok(( conn, client ))

class ShutdownTask:
	fd: poller.SOCKET
	result_flag: atomic.Atomic[i32]
	def __init__( self, fd: poller.SOCKET, result_flag: atomic.Atomic[i32] ) -> None:
		self.fd = fd
		self.result_flag = result_flag
	def run( self ) -> None:
		sig = reactor.fd_signal( self.fd, True, False )
		match reactor.wait_for_signal( sig ):
			case Result.Ok( _ ):
				self.result_flag.store( 1 )
			case Result.Err( _ ):
				self.result_flag.store( 2 )

class Shutter:
	r: reactor.Reactor
	def __init__( self, r: reactor.Reactor ) -> None:
		self.r = r
	def run( self ) -> None:
		busy_delay()
		self.r.shutdown()

def run() -> Result[i32, OSError]:
	( conn_a, client_a ) = make_conn_pair().or_return()
	( conn_b, client_b ) = make_conn_pair().or_return()

	flag_a = atomic.Atomic[i32]( 0 )
	flag_b = atomic.Atomic[i32]( 0 )
	r = reactor.Reactor( 2 )
	ta = ShutdownTask( conn_a.fileno(), flag_a )
	tb = ShutdownTask( conn_b.fileno(), flag_b )
	r.spawn( ta.run )
	r.spawn( tb.run )

	shutter = Shutter( r )
	t_shutter = threading.Thread( shutter.run )

	r.run()
	t_shutter.join()

	if flag_a.load() != 2:
		return Result.Ok( 1 )
	if flag_b.load() != 2:
		return Result.Ok( 2 )
	return Result.Ok( 0 )

def main() -> i32:
	match run():
		case Result.Ok( code ):
			return code
		case Result.Err( _ ):
			return 3
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0, timeout = 20 )

def _emit( compiler: Compiler ) -> str:
	import emitter_c
	return emitter_c.emit_c( compiler )
