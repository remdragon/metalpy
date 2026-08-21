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

import compiler
import sys
import tcp
import threading


class ConnectionDispatcher:
	''' how a freshly-accepted connection actually gets run against
	on_connection - inline on the accept loop's own thread, one new OS
	thread per connection, or handed to a bounded ThreadPool. '''
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


# _ConnectionJob - a plain-field-call indirection around a known compiler
# gap: calling a Closure THROUGH another closure's own captured environment
# (`lambda: h(conn)` where `h` is itself a captured Closure local) fails to
# compile ("'h' is not callable on ...$$lambda$$env"). Every existing
# lambda-wrapped call in this codebase (asyncfile.py, http/server.py) only
# ever wraps a call to a plain function/method - never a captured Closure
# variable - so this is genuinely unexercised territory, not a previously-
# fixed case. Calling a Closure stored in an ordinary field from within a
# plain bound method (self.__on_connection(self.__conn), no lambda
# involved) works fine - the same shape _handle_connection's own
# `handler(request)` already relies on - so this class sidesteps the gap
# rather than needing a compiler fix to unblock this module.
class _ConnectionJob:
	__conn:          tcp.TcpConnection
	__on_connection: Closure[[tcp.TcpConnection], None]

	def __init__( self, conn: tcp.TcpConnection, on_connection: Closure[[tcp.TcpConnection], None] ) -> None:
		self.__conn = conn
		self.__on_connection = on_connection

	def run( self ) -> None:
		# extract to a local before calling - a Closure-typed FIELD isn't
		# directly callable either (same gap _PoolWorker.run_forever's own
		# `work: Closure[...] = job.work` extraction already works around,
		# see lib/threading.py and lib/asyncfile.py's identical pattern)
		on_connection: Closure[[tcp.TcpConnection], None] = self.__on_connection
		on_connection( self.__conn )


class ThreadPerConnectionDispatcher( ConnectionDispatcher ):
	''' one real, unbounded, fire-and-forget OS thread per accepted
	connection (threading.Thread starts immediately - there is nothing to
	join, this is deliberately fire-and-forget). An explicit opt-in
	stress-test baseline - NOT the default, since unbounded thread
	creation under a connection flood is exactly what ThreadPoolDispatcher
	exists to bound. '''
	@virtual
	def dispatch( self, conn: tcp.TcpConnection, on_connection: Closure[[tcp.TcpConnection], None] ) -> None:
		job: _ConnectionJob = _ConnectionJob( conn, on_connection )
		t: threading.Thread = threading.Thread( job.run )


class ThreadPoolDispatcher( ConnectionDispatcher ):
	''' the default dispatch strategy (see TcpServer.__init__) - submits
	each connection onto a bounded threading.ThreadPool instead of
	spawning an unbounded thread per connection. '''
	__pool: threading.ThreadPool

	def __init__( self, pool: threading.ThreadPool ) -> None:
		self.__pool = pool

	@virtual
	def dispatch( self, conn: tcp.TcpConnection, on_connection: Closure[[tcp.TcpConnection], None] ) -> None:
		job: _ConnectionJob = _ConnectionJob( conn, on_connection )
		self.__pool.submit( job.run )


def _default_pool_size() -> usize:
	''' cpu_count()*4, not cpu_count() - unlike reactor.Reactor's own
	CPU-sized worker count, these threads block on connection I/O rather
	than doing CPU work, so a larger multiplier is the right heuristic
	here. A starting guess, not a tuned value - pass an explicit
	ThreadPoolDispatcher( threading.ThreadPool( n )) for a different size. '''
	with compiler.wrap_arithmetic:
		n: u32 = sys.cpu_count() * u32( 4 )
	return usize( n )


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
			self.__dispatcher = ThreadPoolDispatcher( threading.ThreadPool( _default_pool_size() ))
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
