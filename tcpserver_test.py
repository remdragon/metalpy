# Real-compile-and-run tests for lib/tcpserver.py + threading.ThreadPool
# (see FUTURE.md's "ThreadPool?" entry). Mirrors http_server_test.py's own
# structure (RealCompileMixin.assert_programs_run over real TCP loopback
# connections). No sleep() primitive exists in this codebase - concurrency
# is coordinated via atomic gates + a bounded busy_delay() spin, the same
# idiom reactor_test.py already establishes for this exact problem.
#
# Four cases:
#   - inline_dispatcher_runs_synchronously: InlineDispatcher.dispatch()
#     has already run the handler by the time it returns - no thread
#     involved at all.
#   - thread_per_connection_dispatcher_uses_multiple_threads: two
#     dispatch() calls, the first parked on a gate, genuinely run on two
#     independent OS threads (the second completes while the first is
#     still blocked).
#   - thread_pool_dispatcher_bounds_concurrency: a 2-worker pool given 3
#     gated jobs only ever runs 2 of them at once - the third stays queued
#     until a worker frees up, proving the pool bounds concurrency to its
#     own worker count rather than spawning one thread per submission.
#   - thread_pool_shutdown_wait_drains_queue: shutdown(wait=True) doesn't
#     return until every already-queued job has actually run.
#   - tcpserver_shutdown_stops_a_live_accept_loop: TcpServer.shutdown(wait=
#     True) interrupts a run() loop genuinely blocked in its own
#     poller.wait() on a different (real OS) thread, and blocks until that
#     thread has actually returned.
#   - tcpserver_shutdown_does_not_disrupt_an_in_flight_dispatch: shutdown()
#     only stops the accept loop - a connection already dispatched before
#     shutdown() was called still runs to completion.
#   - thread_pool_submit_rejects_once_a_worker_queue_is_full: a 1-worker
#     pool with max_queue_depth=1 accepts a blocking job (fills the
#     worker) and one queued job, then rejects a third with
#     Err(QueueFullError).
#   - thread_pool_dispatcher_closes_connection_on_full_queue: a bounded
#     ThreadPoolDispatcher closes an overflowing connection outright
#     rather than propagating the rejection - the peer sees an orderly
#     close.
#   - default_pool_size_matches_the_python_heuristic: threading.
#     default_pool_size() == min(32, cpu_count() + 4), computed dynamically
#     so it doesn't flake across machines with different core counts.
#   - thread_pool_drains_a_backlog_in_fifo_order: a 1-worker pool given 5
#     jobs while the worker is parked on a blocked 6th runs them in
#     submission order - regression test for a real LIFO-drain/starvation
#     bug (_PoolWorker used to drain via list.pop(), i.e. LIFO).

import unittest

import test_support


class TcpServerTests( test_support.RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile tcpserver tests' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'inline_dispatcher_runs_synchronously', '''
import atomic
import tcp
import socket
import tcpserver

class Counter:
	n: atomic.Atomic[i32]
	def __init__( self ) -> None:
		self.n = atomic.Atomic[i32]( 0 )
	def bump( self, conn: tcp.TcpConnection ) -> None:
		self.n.fetch_add( 1 )

def main() -> i32:
	listener: tcp.TcpListener = tcp.TcpListener.bind( '127.0.0.1', u16( 0 )).unwrap( 'bind' )
	addr: socket.SocketAddr = listener.getsockname().unwrap( 'getsockname' )
	client: tcp.TcpConnection = tcp.connect( '127.0.0.1', addr.port ).unwrap( 'connect' )
	conn: tcp.TcpConnection = listener.accept().unwrap( 'accept' )

	counter: Counter = Counter()
	d: tcpserver.InlineDispatcher = tcpserver.InlineDispatcher()
	d.dispatch( conn, counter.bump )
	# already ran on the caller's own thread by the time dispatch()
	# returns - no join/wait of any kind needed
	if counter.n.load() != 1:
		return 1
	return 0
''' ),
			( 'thread_per_connection_dispatcher_uses_multiple_threads', '''
import compiler
import atomic
import tcp
import socket
import tcpserver

class Gate:
	started:   atomic.Atomic[i32]
	release:   atomic.Atomic[bool]
	completed: atomic.Atomic[i32]
	def __init__( self ) -> None:
		self.started = atomic.Atomic[i32]( 0 )
		self.release = atomic.Atomic[bool]( False )
		self.completed = atomic.Atomic[i32]( 0 )
	def blocking( self, conn: tcp.TcpConnection ) -> None:
		self.started.fetch_add( 1 )
		while not self.release.load():
			pass
		self.completed.fetch_add( 1 )
	def quick( self, conn: tcp.TcpConnection ) -> None:
		self.completed.fetch_add( 1 )

def busy_delay() -> None:
	i: usize = 0
	while i < usize( 200000000 ):
		with compiler.wrap_arithmetic:
			i = i + 1

def main() -> i32:
	listener: tcp.TcpListener = tcp.TcpListener.bind( '127.0.0.1', u16( 0 )).unwrap( 'bind' )
	addr: socket.SocketAddr = listener.getsockname().unwrap( 'getsockname' )

	client1: tcp.TcpConnection = tcp.connect( '127.0.0.1', addr.port ).unwrap( 'connect1' )
	conn1: tcp.TcpConnection = listener.accept().unwrap( 'accept1' )
	client2: tcp.TcpConnection = tcp.connect( '127.0.0.1', addr.port ).unwrap( 'connect2' )
	conn2: tcp.TcpConnection = listener.accept().unwrap( 'accept2' )

	gate: Gate = Gate()
	d: tcpserver.ThreadPerConnectionDispatcher = tcpserver.ThreadPerConnectionDispatcher()
	d.dispatch( conn1, gate.blocking )
	busy_delay()   # give conn1's own thread time to actually start and block on the gate
	d.dispatch( conn2, gate.quick )
	busy_delay()   # give conn2's own thread time to actually run to completion

	# conn2 finished WHILE conn1's own thread is still parked on the gate -
	# proves each dispatch() call got its own independent OS thread rather
	# than being serialized onto one
	if gate.started.load() != 1:
		return 1
	if gate.completed.load() != 1:
		return 2

	gate.release.store( True )
	while gate.completed.load() != 2:
		pass
	return 0
''' ),
			( 'thread_pool_dispatcher_bounds_concurrency', '''
import compiler
import atomic
import tcp
import socket
import tcpserver
import threading

class Gate:
	started:   atomic.Atomic[i32]
	release:   atomic.Atomic[bool]
	completed: atomic.Atomic[i32]
	def __init__( self ) -> None:
		self.started = atomic.Atomic[i32]( 0 )
		self.release = atomic.Atomic[bool]( False )
		self.completed = atomic.Atomic[i32]( 0 )
	def blocking( self, conn: tcp.TcpConnection ) -> None:
		self.started.fetch_add( 1 )
		while not self.release.load():
			pass
		self.completed.fetch_add( 1 )

def busy_delay() -> None:
	i: usize = 0
	while i < usize( 200000000 ):
		with compiler.wrap_arithmetic:
			i = i + 1

def main() -> i32:
	listener: tcp.TcpListener = tcp.TcpListener.bind( '127.0.0.1', u16( 0 )).unwrap( 'bind' )
	addr: socket.SocketAddr = listener.getsockname().unwrap( 'getsockname' )

	gate: Gate = Gate()
	pool: threading.ThreadPool = threading.ThreadPool( usize( 2 ))
	d: tcpserver.ThreadPoolDispatcher = tcpserver.ThreadPoolDispatcher( pool )

	i: usize = 0
	while i < usize( 3 ):
		client: tcp.TcpConnection = tcp.connect( '127.0.0.1', addr.port ).unwrap( 'connect' )
		conn: tcp.TcpConnection = listener.accept().unwrap( 'accept' )
		d.dispatch( conn, gate.blocking )
		with compiler.wrap_arithmetic:
			i = i + 1

	busy_delay()   # give the pool's 2 worker threads time to pick up and block on 2 of the 3 jobs

	# exactly pool-size (2) jobs running, the 3rd still queued - the pool
	# bounds concurrency to its own worker count, not one thread per job
	if gate.started.load() != 2:
		return 1

	gate.release.store( True )
	while gate.completed.load() != 3:
		pass
	pool.shutdown( True )
	return 0
''' ),
			( 'thread_pool_shutdown_wait_drains_queue', '''
import compiler
import atomic
import threading

class Counter:
	completed: atomic.Atomic[i32]
	def __init__( self ) -> None:
		self.completed = atomic.Atomic[i32]( 0 )
	def bump( self ) -> None:
		self.completed.fetch_add( 1 )

def main() -> i32:
	counter: Counter = Counter()
	pool: threading.ThreadPool = threading.ThreadPool( usize( 1 ))
	i: usize = 0
	while i < usize( 3 ):
		pool.submit( counter.bump ).unwrap( 'submit' )
		with compiler.wrap_arithmetic:
			i = i + 1
	pool.shutdown( True )
	# shutdown(wait=True) doesn't return until every already-queued job
	# has actually run - no busy-wait needed here, that's the property
	# under test
	if counter.completed.load() != 3:
		return 1
	return 0
''' ),
			( 'tcpserver_shutdown_stops_a_live_accept_loop', '''
import tcp
import threading
import tcpserver

class NullOnConnection:
	def handle( self, conn: tcp.TcpConnection ) -> None:
		pass

def main() -> i32:
	listener: tcp.TcpListener = tcp.TcpListener.bind( '127.0.0.1', u16( 0 )).unwrap( 'bind' )
	h: NullOnConnection = NullOnConnection()
	server: tcpserver.TcpServer = tcpserver.TcpServer( listener, h.handle )
	t: threading.Thread = threading.Thread( server.run )   # starts immediately, genuinely blocks in poller.wait()

	server.shutdown( True )   # blocks until run()'s own thread has actually returned
	t.join()                  # must return promptly - proves run()'s OS thread really exited, not just detached
	return 0
''' ),
			( 'tcpserver_shutdown_does_not_disrupt_an_in_flight_dispatch', '''
import atomic
import tcp
import socket
import threading
import tcpserver

class Counter:
	n: atomic.Atomic[i32]
	def __init__( self ) -> None:
		self.n = atomic.Atomic[i32]( 0 )
	def bump( self, conn: tcp.TcpConnection ) -> None:
		self.n.fetch_add( 1 )

def main() -> i32:
	listener: tcp.TcpListener = tcp.TcpListener.bind( '127.0.0.1', u16( 0 )).unwrap( 'bind' )
	addr: socket.SocketAddr = listener.getsockname().unwrap( 'getsockname' )
	counter: Counter = Counter()
	server: tcpserver.TcpServer = tcpserver.TcpServer( listener, counter.bump )
	t: threading.Thread = threading.Thread( server.run )

	client: tcp.TcpConnection = tcp.connect( '127.0.0.1', addr.port ).unwrap( 'connect' )
	while counter.n.load() != 1:
		pass   # wait for run()'s own accept loop to notice and dispatch it

	server.shutdown( True )
	t.join()
	if counter.n.load() != 1:
		return 1
	return 0
''' ),
			( 'thread_pool_submit_rejects_once_a_worker_queue_is_full', '''
import compiler
import atomic
import threading

class Gate:
	started: atomic.Atomic[i32]
	release: atomic.Atomic[bool]
	def __init__( self ) -> None:
		self.started = atomic.Atomic[i32]( 0 )
		self.release = atomic.Atomic[bool]( False )
	def blocking( self ) -> None:
		self.started.fetch_add( 1 )
		while not self.release.load():
			pass

def busy_delay() -> None:
	i: usize = 0
	while i < usize( 200000000 ):
		with compiler.wrap_arithmetic:
			i = i + 1

def main() -> i32:
	gate: Gate = Gate()
	pool: threading.ThreadPool = threading.ThreadPool( usize( 1 ), usize( 1 ))
	pool.submit( gate.blocking ).unwrap( 'first submit' )   # occupies the one worker
	busy_delay()   # give the worker time to actually start and block on the gate
	if gate.started.load() != 1:
		return 1
	pool.submit( gate.blocking ).unwrap( 'second submit' )   # fills the one queue slot
	match pool.submit( gate.blocking ):
		case Result.Ok( _ ):
			return 2   # the queue was already full - this must have been rejected
		case Result.Err( _ ):
			pass
	gate.release.store( True )
	pool.shutdown( True )
	return 0
''' ),
			( 'thread_pool_dispatcher_closes_connection_on_full_queue', '''
import compiler
import atomic
import tcp
import socket
import tcpserver
import threading

class Gate:
	started: atomic.Atomic[i32]
	release: atomic.Atomic[bool]
	def __init__( self ) -> None:
		self.started = atomic.Atomic[i32]( 0 )
		self.release = atomic.Atomic[bool]( False )
	def blocking( self, conn: tcp.TcpConnection ) -> None:
		self.started.fetch_add( 1 )
		while not self.release.load():
			pass

def busy_delay() -> None:
	i: usize = 0
	while i < usize( 200000000 ):
		with compiler.wrap_arithmetic:
			i = i + 1

def main() -> i32:
	listener: tcp.TcpListener = tcp.TcpListener.bind( '127.0.0.1', u16( 0 )).unwrap( 'bind' )
	addr: socket.SocketAddr = listener.getsockname().unwrap( 'getsockname' )

	gate: Gate = Gate()
	pool: threading.ThreadPool = threading.ThreadPool( usize( 1 ), usize( 1 ))
	d: tcpserver.ThreadPoolDispatcher = tcpserver.ThreadPoolDispatcher( pool )

	client1: tcp.TcpConnection = tcp.connect( '127.0.0.1', addr.port ).unwrap( 'connect1' )
	conn1: tcp.TcpConnection = listener.accept().unwrap( 'accept1' )
	d.dispatch( conn1, gate.blocking )   # occupies the one worker
	busy_delay()
	if gate.started.load() != 1:
		return 1

	client2: tcp.TcpConnection = tcp.connect( '127.0.0.1', addr.port ).unwrap( 'connect2' )
	conn2: tcp.TcpConnection = listener.accept().unwrap( 'accept2' )
	d.dispatch( conn2, gate.blocking )   # fills the one queue slot

	client3: tcp.TcpConnection = tcp.connect( '127.0.0.1', addr.port ).unwrap( 'connect3' )
	conn3: tcp.TcpConnection = listener.accept().unwrap( 'accept3' )
	d.dispatch( conn3, gate.blocking )   # queue already full - dispatch() must close conn3 outright, synchronously

	buf: bytearray = bytearray( usize( 16 ))
	match client3.read( buf.get_ptr(), usize( 16 )):
		case Result.Ok( n ):
			if n != usize( 0 ):
				return 2   # a real close reads back 0 bytes, not data
		case Result.Err( _ ):
			pass   # a reset/aborted read is also an acceptable "closed" signal

	gate.release.store( True )
	pool.shutdown( True )
	return 0
''' ),
			( 'default_pool_size_matches_the_python_heuristic', '''
import compiler
import sys
import threading

def main() -> i32:
	n: u32 = sys.cpu_count()
	with compiler.wrap_arithmetic:
		expected: usize = usize( n ) + usize( 4 )
	if expected > usize( 32 ):
		expected = usize( 32 )
	got: usize = threading.default_pool_size()
	if got != expected:
		return 1
	if got < usize( 1 ) or got > usize( 32 ):
		return 2
	return 0
''' ),
			( 'thread_pool_drains_a_backlog_in_fifo_order', '''
import compiler
import atomic
import threading

class Gate:
	started: atomic.Atomic[i32]
	release: atomic.Atomic[bool]
	def __init__( self ) -> None:
		self.started = atomic.Atomic[i32]( 0 )
		self.release = atomic.Atomic[bool]( False )
	def blocking( self ) -> None:
		self.started.fetch_add( 1 )
		while not self.release.load():
			pass

class Recorder:
	order: list[usize]
	def __init__( self ) -> None:
		self.order = list[usize]()
	def record( self, n: usize ) -> None:
		self.order.append( n )

class Job:
	idx: usize
	rec: Recorder
	def __init__( self, idx: usize, rec: Recorder ) -> None:
		self.idx = idx
		self.rec = rec
	def run( self ) -> None:
		self.rec.record( self.idx )

def busy_delay() -> None:
	i: usize = 0
	while i < usize( 200000000 ):
		with compiler.wrap_arithmetic:
			i = i + 1

def main() -> i32:
	gate: Gate = Gate()
	rec: Recorder = Recorder()
	pool: threading.ThreadPool = threading.ThreadPool( usize( 1 ))   # one worker - forces every job onto the same queue

	pool.submit( gate.blocking ).unwrap( 'submit gate' )   # occupies the one worker, parking it
	busy_delay()   # give the worker time to actually start and block on the gate
	if gate.started.load() != 1:
		return 1

	# queue up a real backlog behind the blocked worker - previously drained
	# LIFO (list.pop() took the LAST element), so job 4 would have run
	# before job 1; Queue[T]'s FIFO drain must preserve submission order
	i: usize = 0
	while i < usize( 5 ):
		job: Job = Job( i, rec )
		pool.submit( job.run ).unwrap( 'submit job' )
		with compiler.wrap_arithmetic:
			i = i + 1

	gate.release.store( True )
	pool.shutdown( True )

	if rec.order.__len__() != usize( 5 ):
		return 2
	i = 0
	while i < usize( 5 ):
		got: usize = rec.order.__getitem__( i ).unwrap( 'order index in bounds by construction' )
		if got != i:
			return 3   # out of submission order - LIFO regression
		with compiler.wrap_arithmetic:
			i = i + 1
	return 0
''' ),
		], timeout = 30 )   # busy-wait loops on gate release - an infinite-spin regression fails instead of hanging the whole suite, matching reactor_test.py's own convention


if __name__ == '__main__':
	unittest.main()
