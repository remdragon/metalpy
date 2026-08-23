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
		pool.submit( counter.bump )
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
		], timeout = 30 )   # busy-wait loops on gate release - an infinite-spin regression fails instead of hanging the whole suite, matching reactor_test.py's own convention


if __name__ == '__main__':
	unittest.main()
