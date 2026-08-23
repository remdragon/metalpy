'''
tcpserver - a generic, protocol-agnostic TCP accept-loop server, modeled on
Python's socketserver module (BaseServer/TCPServer/ThreadingMixIn). NOT
HTTP-specific: any future TCP-based protocol builds its own sync-serving
entry point on top of TcpServer, the same way lib/http/server.py's own
async path already builds on reactor.Reactor - see lib/http/server.py's
serve_sync() for exactly that.

This compiler supports single implementation inheritance only (SYNTAX.md's
"Zero-cost interfaces" section; RCClass.base is a single pointer, not an
MRO list - see mpy_types.py), and @protocol's structural conformance
explicitly refuses to resolve an ambiguous default supplied by two
conformed protocols (protocol_test.py's own
test_ambiguous_default_from_two_protocols_is_a_compile_error) - so there is
no ThreadingMixIn-shaped substitute here. ConnectionDispatcher below is a
real, single-inheritance abstract base (@abstractmethod/@virtual), injected
into TcpServer via its constructor as a strategy object - the same
FileOpsInterface/_SyncFileOps injection shape lib/builtins/__File.py
already uses for reactor-aware file I/O - not a base class TcpServer
itself extends.
'''

import atomic
import compiler
import poller
import reactor
import socket
import sys
import tcp
import threading
from datetime import timedelta


class ConnectionDispatcher:
	''' how a freshly-accepted connection actually gets run against
	on_connection - inline on the accept loop's own thread, one new OS
	thread per connection, or handed to a bounded ThreadPool.

	IMPORTANT for on_connection callbacks that loop for the life of the
	connection (e.g. lib/http/server.py's own handle_connection, which
	keeps handling requests on the SAME connection until it closes -
	HTTP/1.1 keep-alive): the dispatcher's own concurrency unit is one
	CONNECTION, for its ENTIRE lifetime, not one discrete unit of work.
	See ThreadPoolDispatcher's own docstring for why that specifically
	rules it out as a default for a keep-alive protocol. '''
	@abstractmethod
	def dispatch( self, conn: tcp.TcpConnection, on_connection: Closure[[tcp.TcpConnection], None] ) -> None:
		...


class InlineDispatcher( ConnectionDispatcher ):
	''' matches Python's plain (non-Threading) TCPServer: one connection is
	fully handled before accept() is called again. A true single-threaded
	baseline, e.g. for tests that want deterministic ordering. '''
	@virtual
	def dispatch( self, conn: tcp.TcpConnection, on_connection: Closure[[tcp.TcpConnection], None] ) -> None:
		on_connection( conn )


class ThreadPerConnectionDispatcher( ConnectionDispatcher ):
	''' one real, unbounded, fire-and-forget OS thread per accepted
	connection (threading.Thread starts immediately - there is nothing to
	join, this is deliberately fire-and-forget). THE DEFAULT (see
	TcpServer.__init__) - the correct choice for a connection-oriented
	protocol whose handler runs for the connection's whole lifetime (e.g.
	HTTP/1.1 keep-alive): each connection gets its own thread for as long
	as it stays open, so one slow/idle client can never block another.
	Costs a real OS thread per concurrent connection - see
	ThreadPoolDispatcher below for why that's NOT bounded via a fixed
	pool instead. '''
	@virtual
	def dispatch( self, conn: tcp.TcpConnection, on_connection: Closure[[tcp.TcpConnection], None] ) -> None:
		t: threading.Thread = threading.Thread( lambda: on_connection( conn ))


class ThreadPoolDispatcher( ConnectionDispatcher ):
	''' submits each connection onto a bounded threading.ThreadPool
	instead of spawning a thread per connection.

	NOT SAFE as a default for a connection-oriented protocol whose
	handler loops for the connection's lifetime (HTTP/1.1 keep-alive
	included) - confirmed by a real repro, not just reasoning: a pool of
	N workers can only ever have N connections ALIVE at once, because
	each occupies its worker for as long as the connection stays open,
	not just for one request. Every connection beyond N sits queued
	behind whichever N connections happened to arrive first, and stays
	stuck there for as long as those N remain open - under sustained
	concurrent keep-alive load (the exact case a real HTTP client like
	`hey` exercises) this is starvation, not backpressure: excess
	connections get accepted, then time out having never received a
	single byte back, while the first N clients are served indefinitely.
	This is also why Python's own socketserver never shipped a bounded-
	pool mixin - only ThreadingMixIn (unbounded, like
	ThreadPerConnectionDispatcher) or a real event loop are sound for a
	persistent-connection protocol.

	Correct use: a protocol/handler that does a BOUNDED, short-lived unit
	of work per dispatch() call and returns - e.g. request-then-close,
	or any handler that itself hands off to something else and returns
	promptly rather than looping for the connection's own lifetime. '''
	__pool: threading.ThreadPool

	def __init__( self, pool: threading.ThreadPool ) -> None:
		self.__pool = pool

	@virtual
	def dispatch( self, conn: tcp.TcpConnection, on_connection: Closure[[tcp.TcpConnection], None] ) -> None:
		# ConnectionDispatcher.dispatch()'s own abstract signature stays ->
		# None (unchanged) - a full queue closes the connection outright
		# rather than propagating a Result up through every dispatcher
		match self.__pool.submit( lambda: on_connection( conn )):
			case Result.Ok( _ ):
				pass
			case Result.Err( _ ):
				conn.close()


class TcpServer:
	''' a generic accept loop over a tcp.TcpListener: each accepted
	connection is handed to on_connection via this server's own
	ConnectionDispatcher.

	Owns a persistent poller.Poller + wake-pair socket (the same self-pipe
	idiom reactor.Worker uses, standalone - run() is meant to run on a raw
	OS thread, not a reactor fiber, so there's no Worker to piggyback a
	shutdown wake on). tcp.TcpListener.accept()'s own internal retry loop
	is reactor-optional but NOT externally interruptible - with no Worker
	driving this thread, it falls to reactor._blocking_wait_no_reactor,
	which builds a fresh, throwaway, single-fd Poller on every retry with no
	shutdown hook at all. run() uses TcpListener.try_accept() instead so it
	can wait on ITS OWN poller, registered with both the listener fd and
	the wake-read fd, making shutdown() (below) able to interrupt it. '''
	__listener:      tcp.TcpListener
	__on_connection: Closure[[tcp.TcpConnection], None]
	__dispatcher:    ConnectionDispatcher
	__poller:        poller.Poller
	__wake_read:     socket.Socket
	__wake_write:    socket.Socket
	__wake_fd:       poller.SOCKET
	__shutting_down: atomic.Atomic[bool]
	__stopped:       atomic.Atomic[bool]

	def __init__(
		self,
		listener:      tcp.TcpListener,
		on_connection: Closure[[tcp.TcpConnection], None],
		dispatcher:    ConnectionDispatcher|None = None,
	) -> None:
		self.__listener = listener
		self.__on_connection = on_connection
		if dispatcher is None:
			self.__dispatcher = ThreadPerConnectionDispatcher()
		else:
			d: ConnectionDispatcher = dispatcher
			self.__dispatcher = d
		self.__poller = poller.Poller()
		( wake_read, wake_write ) = socket.make_loopback_pair()
		poller.set_nonblocking( wake_read.fileno() ).unwrap( 'TcpServer.__init__: set_nonblocking (wake read side) failed' )
		poller.set_nonblocking( wake_write.fileno() ).unwrap( 'TcpServer.__init__: set_nonblocking (wake write side) failed' )
		self.__wake_read = wake_read
		self.__wake_write = wake_write
		self.__wake_fd = wake_read.fileno()
		self.__shutting_down = atomic.Atomic[bool]( False )
		self.__stopped = atomic.Atomic[bool]( False )
		self.__poller.register( self.__listener.fileno(), True, False ).unwrap( 'TcpServer.__init__: poller register (listener) failed' )
		self.__poller.register( self.__wake_fd, True, False ).unwrap( 'TcpServer.__init__: poller register (wake fd) failed' )

	def run( self ) -> None:
		''' blocks until shutdown() is called (or a real accept error):
		waits on this server's own poller, dispatching a ready listener
		fd via dispatcher.dispatch(conn, on_connection) and draining a
		ready wake fd (its only job is to break the wait) - naming matches
		reactor.Reactor.run() rather than Python's serve_forever(), for
		consistency within this codebase. '''
		while True:
			if self.__shutting_down.load():
				break
			match self.__poller.wait( -1 ):
				case Result.Ok( events ):
					stop_now: bool = False
					n: usize = events.__len__()
					i: usize = 0
					while i < n:
						ev: poller.ReadyEvent = events.__getitem__( i ).unwrap( 'TcpServer.run: event index in bounds by construction' )
						if ev.fd == self.__wake_fd:
							buf: bytearray = bytearray( usize( 64 ))
							self.__wake_read.recv( buf.get_ptr(), usize( 64 )).is_ok() # best-effort drain
						elif ev.readable:
							match self.__listener.try_accept():
								case Result.Ok( conn ):
									self.__dispatcher.dispatch( conn, self.__on_connection )
								case Result.Err( e ):
									if e != OSError.WouldBlock:
										print( f'TcpServer.run: error accepting connection: {e}' )
										stop_now = True
						with compiler.wrap_arithmetic:
							i = i + 1
					if stop_now:
						break
				case Result.Err( e ):
					print( f'TcpServer.run: poller wait failed: {e}' )
					break
		self.__stopped.store( True )

	def shutdown( self, wait: bool = True ) -> None:
		''' requests run()'s accept loop to stop - idempotent, callable from
		any thread, including while run() is blocked in its own poller.wait()
		(possibly on a different thread). Does NOT reject already-dispatched
		connections - matches reactor.Worker.request_shutdown()/threading.
		ThreadPool.shutdown()'s own "request, don't reject in-flight work"
		contract; only the accept loop itself stops. wait=True (default)
		additionally blocks until run() has actually returned (a 1ms busy-
		poll on an atomic flag - robust to any call ordering, e.g. shutdown()
		called before run() ever starts, or after it already returned, both
		just return immediately with no risk of ever blocking forever). '''
		self.__shutting_down.store( True )
		poke: bytes = b'x'
		match self.__wake_write.send( poke.get_const_ptr(), usize( 1 )):
			case Result.Ok( _n ):
				pass
			case Result.Err( e ):
				if e != OSError.WouldBlock:
					sys.panic( 'TcpServer.shutdown: wake-pair write failed unexpectedly' )
		if not wait:
			return
		while not self.__stopped.load():
			reactor.sleep( timedelta( milliseconds = 1 )).is_ok()
