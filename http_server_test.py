# Real-compile-and-run tests for lib/http/server.py (see PLAN_HTTP_SERVER.md).
# lib/http/client.py's own already-tested HTTPConnection is used as the
# client side throughout - an independent cross-check of the new server,
# not a hand-rolled test-only client - mirroring http_client_test.py's own
# "hand-roll a raw-socket peer" idea but in the other direction (here the
# NEW code is the server, the peer is the EXISTING, already-tested client).
#
# Two sub-programs, per PLAN_HTTP_SERVER.md's own "exercised both bare-
# thread and inside a Reactor" requirement (matching tcp_test.py's own dual
# coverage of TcpConnection/TcpListener):
#   - bare_thread: calls http.server's own (module-private) _handle_connection
#     directly from a plain threading.Thread, no Reactor anywhere - proves
#     the request-parsing/keep-alive loop itself is reactor-optional, the
#     same property tcp.py's own TcpConnection/TcpListener already have.
#   - reactor: the real public entry point (serve() + reactor.Reactor),
#     covering round-trip, keep-alive across 2 requests on the SAME
#     connection, a non-200 response, a Content-Length request body, and
#     "Connection: close" being honored (verified via the response's own
#     echoed Connection header, not a follow-up read-after-close probe).
#
# Each sub-program's main() -> i32 returns a distinct nonzero code per
# failed assertion (0 = every assertion passed) - test_support.
# assert_programs_run decodes a failure back to the offending case name and
# sub-code.

import unittest

import test_support


class HTTPServerTests( test_support.RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile http.server tests' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'bare_thread_no_reactor_keep_alive', '''
import compiler
import threading
import atomic
import tcp
import socket
from socket import Socket
from http.client import HTTPConnection, _Connection, Response as ClientResponse
from http.server import Request, Response, _handle_connection

class App:
	def handle( self, req: Request ) -> Response:
		if req.path == '/echo':
			return Response.text( 'you asked for ' + req.path )
		return Response.text( 'nope', 404, 'Not Found' )

class BareServer:
	__listener: tcp.TcpListener
	__app:      App
	__flag:     atomic.Atomic[i32]

	def __init__( self, listener: tcp.TcpListener, app: App, flag: atomic.Atomic[i32] ) -> None:
		self.__listener = listener
		self.__app = app
		self.__flag = flag

	def run( self ) -> None:
		match self.__listener.accept():
			case Result.Ok( conn ):
				_handle_connection( conn, self.__app.handle )
				self.__flag.store( 1 )
			case Result.Err( _ ):
				self.__flag.store( 2 )

class BareClient:
	__port: u16
	__flag: atomic.Atomic[i32]

	def __init__( self, port: u16, flag: atomic.Atomic[i32] ) -> None:
		self.__port = port
		self.__flag = flag

	def __try_run( self ) -> Result[None, i32]:
		conn: _Connection[Socket] = HTTPConnection.connect( '127.0.0.1', self.__port ).unwrap( 'client connect' )

		conn.request( 'GET', '/echo', None, None ).unwrap( 'req1' )
		resp1: ClientResponse = conn.getresponse().unwrap( 'resp1' )
		if resp1.status_code != 200:
			return Result.Err( 1 )
		text1: str = resp1.text().unwrap( 'text1' )
		if text1 != 'you asked for /echo':
			return Result.Err( 2 )

		# second request, SAME connection - proves keep-alive works with no
		# Reactor involved at all
		conn.request( 'GET', '/missing', None, None ).unwrap( 'req2' )
		resp2: ClientResponse = conn.getresponse().unwrap( 'resp2' )
		if resp2.status_code != 404:
			return Result.Err( 3 )

		conn.close()
		return Result.Ok( None )

	def run( self ) -> None:
		match self.__try_run():
			case Result.Ok( _ ):
				self.__flag.store( 0 )
			case Result.Err( code ):
				self.__flag.store( code )

def scenario() -> Result[i32, OSError]:
	listener: tcp.TcpListener = tcp.TcpListener.bind( '127.0.0.1', u16( 0 )).or_return()
	addr: socket.SocketAddr = listener.getsockname().or_return()
	app: App = App()

	server_flag: atomic.Atomic[i32] = atomic.Atomic[i32]( 0 )
	client_flag: atomic.Atomic[i32] = atomic.Atomic[i32]( 0 )
	server: BareServer = BareServer( listener, app, server_flag )
	client: BareClient = BareClient( addr.port(), client_flag )
	t_server: threading.Thread = threading.Thread( server.run )
	t_client: threading.Thread = threading.Thread( client.run )
	t_server.join()
	t_client.join()

	with compiler.wrap_arithmetic:
		if server_flag.load() != 1:
			return Result.Ok( 10 + server_flag.load() )
		if client_flag.load() != 0:
			return Result.Ok( 20 + client_flag.load() )
	return Result.Ok( 0 )

def main() -> i32:
	match scenario():
		case Result.Ok( code ):
			return code
		case Result.Err( _ ):
			return 90
''' ),
			( 'reactor_round_trip_keep_alive_body_and_close', '''
import compiler
import threading
import atomic
import reactor
import tcp
import socket
from socket import Socket
from http.client import HTTPConnection, _Connection, Response as ClientResponse, HTTPHeaders
from http.server import Request, Response, serve

class App:
	def handle( self, req: Request ) -> Response:
		if req.path == '/echo':
			return Response.text( 'you asked for ' + req.path )
		if req.path == '/upload':
			n: usize = req.body.__len__()
			return Response.text( 'received ' + n.__str__() + ' bytes' )
		return Response.text( 'nope', 404, 'Not Found' )

class ClientDriver:
	__port:    u16
	__reactor: reactor.Reactor
	__flag:    atomic.Atomic[i32]

	def __init__( self, port: u16, r: reactor.Reactor, flag: atomic.Atomic[i32] ) -> None:
		self.__port = port
		self.__reactor = r
		self.__flag = flag

	def __try_run( self ) -> Result[None, i32]:
		conn: _Connection[Socket] = HTTPConnection.connect( '127.0.0.1', self.__port ).unwrap( 'client connect' )

		conn.request( 'GET', '/echo', None, None ).unwrap( 'req1' )
		resp1: ClientResponse = conn.getresponse().unwrap( 'resp1' )
		if resp1.status_code != 200:
			return Result.Err( 1 )
		text1: str = resp1.text().unwrap( 'text1' )
		if text1 != 'you asked for /echo':
			return Result.Err( 2 )

		# second request, SAME connection - keep-alive under a real Reactor
		conn.request( 'GET', '/missing', None, None ).unwrap( 'req2' )
		resp2: ClientResponse = conn.getresponse().unwrap( 'resp2' )
		if resp2.status_code != 404:
			return Result.Err( 3 )

		# third request, a real body - Content-Length round trip
		body: bytes = 'abcde'.encode().unwrap( 'body encode' )
		conn.request( 'POST', '/upload', None, body ).unwrap( 'req3' )
		resp3: ClientResponse = conn.getresponse().unwrap( 'resp3' )
		if resp3.status_code != 200:
			return Result.Err( 4 )
		text3: str = resp3.text().unwrap( 'text3' )
		if text3 != 'received 5 bytes':
			return Result.Err( 5 )

		# fourth request, explicit Connection: close - server should echo
		# it back on the response (that's what we check, rather than a
		# separate read-after-close probe)
		close_headers: HTTPHeaders = HTTPHeaders()
		close_headers.set( 'Connection', 'close' )
		conn.request( 'GET', '/echo', close_headers, None ).unwrap( 'req4' )
		resp4: ClientResponse = conn.getresponse().unwrap( 'resp4' )
		if resp4.status_code != 200:
			return Result.Err( 6 )
		close_value: str|None = resp4.headers.get( 'Connection' )
		if close_value is None:
			return Result.Err( 7 )
		cv: str = close_value
		if cv.lower() != 'close':
			return Result.Err( 8 )

		conn.close()
		return Result.Ok( None )

	def run( self ) -> None:
		match self.__try_run():
			case Result.Ok( _ ):
				self.__flag.store( 0 )
			case Result.Err( code ):
				self.__flag.store( code )
		self.__reactor.shutdown()

def scenario() -> Result[i32, OSError]:
	listener: tcp.TcpListener = tcp.TcpListener.bind( '127.0.0.1', u16( 0 )).or_return()
	addr: socket.SocketAddr = listener.getsockname().or_return()
	# 1 worker, not 2+ - a confirmed, pre-existing reactor.py bug (task
	# task_0904047c, found while writing this test): a task spawned by a
	# running fiber onto ANOTHER worker that started with an empty queue
	# never runs, because that worker's own drain_fully() already returned
	# (nothing to do, nothing to wait for) before the cross-worker spawn
	# ever happens. Reproduces with a minimal reactor.py-only repro, no
	# http.server involved - not something this module can work around.
	r: reactor.Reactor = reactor.Reactor( 1 )
	app: App = App()
	serve( listener, app.handle, r )

	client_flag: atomic.Atomic[i32] = atomic.Atomic[i32]( 0 )
	client: ClientDriver = ClientDriver( addr.port(), r, client_flag )
	t: threading.Thread = threading.Thread( client.run )
	r.run()
	t.join()
	return Result.Ok( client_flag.load() )

def main() -> i32:
	match scenario():
		case Result.Ok( code ):
			return code
		case Result.Err( _ ):
			return 90
''' ),
		])


if __name__ == '__main__':
	unittest.main()
