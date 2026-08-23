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

import tcp
import threading


class ConnectionDispatcher:
	''' how a freshly-accepted connection actually gets run against
	on_connection - inline on the accept loop's own thread, one new OS
	thread per connection, or handed to a bounded ThreadPool.

	IMPORTANT for on_connection callbacks that loop for the life of the
	connection (e.g. lib/http/server.py's own _handle_connection, which
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
		self.__pool.submit( lambda: on_connection( conn ))


class TcpServer:
	''' a generic accept loop over a tcp.TcpListener: each accepted
	connection is handed to on_connection via this server's own
	ConnectionDispatcher. '''
	__listener:      tcp.TcpListener
	__on_connection: Closure[[tcp.TcpConnection], None]
	__dispatcher:    ConnectionDispatcher

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

	def run( self ) -> None:
		''' blocks forever: accept() then dispatcher.dispatch(conn,
		on_connection) - naming matches reactor.Reactor.run() rather than
		Python's serve_forever(), for consistency within this codebase. '''
		while True:
			match self.__listener.accept():
				case Result.Ok( conn ):
					self.__dispatcher.dispatch( conn, self.__on_connection )
				case Result.Err( e ):
					print( f'TcpServer.run: error accepting connection: {e}' )
					return
