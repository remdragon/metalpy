# Real-compile-and-run behavioral tests for lib/poller.py (readiness-based
# I/O demultiplexing: epoll on POSIX/WSL, WSAPoll on Windows) and
# poller.set_nonblocking(). A real OS socket pair is the only way to
# confirm readiness reporting actually reflects real kernel state - see
# socket_test.py's own header comment for why each case gets its own
# compile+link+run rather than the merged assert_programs_run helper
# (Socket.create()'s lazy WSAStartup-once guard).
#
# Every program returns 0 on success, or a distinct nonzero i32 exit code
# per failed check - same convention socket_test.py already established.

import unittest

import emitter_c
import test_support
from test_support import RealCompileMixin


_READINESS_TRACKS_REAL_DATA = '''
import socket
import poller

def run() -> Result[i32, OSError]:
	server: socket.Socket = socket.Socket.tcp().or_return()
	server.bind( '127.0.0.1', u16( 0 )).or_return()
	server.listen().or_return()
	bound: socket.SocketAddr = server.getsockname().or_return()

	client: socket.Socket = socket.Socket.tcp().or_return()
	client.connect( '127.0.0.1', bound.port() ).or_return()
	( conn, _addr ) = server.accept().or_return()

	poller.set_nonblocking( conn.fileno() ).or_return()

	p: poller.Poller = poller.Poller()
	p.register( conn.fileno(), True, False ).or_return()

	before: list[poller.ReadyEvent] = p.wait( i32( 50 )).or_return()
	if before.__len__() != usize( 0 ):
		return Result.Ok( 1 )

	msg: bytes = b'ping!'
	client.send_all( msg.get_const_ptr(), usize( 5 )).or_return()

	after: list[poller.ReadyEvent] = p.wait( i32( 5000 )).or_return()
	if after.__len__() != usize( 1 ):
		return Result.Ok( 2 )
	ev: poller.ReadyEvent = after.__getitem__( usize( 0 )).unwrap( 'index in bounds' )
	if not ev.readable:
		return Result.Ok( 3 )
	if ev.fd != conn.fileno():
		return Result.Ok( 4 )

	conn.close()
	client.close()
	server.close()
	return Result.Ok( 0 )

def main() -> i32:
	match run():
		case Result.Ok( code ):
			return code
		case Result.Err( _ ):
			return 5
'''

_UNREGISTER_STOPS_REPORTING = '''
import socket
import poller

def run() -> Result[i32, OSError]:
	server: socket.Socket = socket.Socket.tcp().or_return()
	server.bind( '127.0.0.1', u16( 0 )).or_return()
	server.listen().or_return()
	bound: socket.SocketAddr = server.getsockname().or_return()

	client: socket.Socket = socket.Socket.tcp().or_return()
	client.connect( '127.0.0.1', bound.port() ).or_return()
	( conn, _addr ) = server.accept().or_return()

	poller.set_nonblocking( conn.fileno() ).or_return()

	p: poller.Poller = poller.Poller()
	p.register( conn.fileno(), True, False ).or_return()

	msg: bytes = b'ping!'
	client.send_all( msg.get_const_ptr(), usize( 5 )).or_return()

	p.unregister( conn.fileno() ).or_return()
	after_unregister: list[poller.ReadyEvent] = p.wait( i32( 50 )).or_return()
	if after_unregister.__len__() != usize( 0 ):
		return Result.Ok( 1 )

	conn.close()
	client.close()
	server.close()
	return Result.Ok( 0 )

def main() -> i32:
	match run():
		case Result.Ok( code ):
			return code
		case Result.Err( _ ):
			return 2
'''

_NONBLOCKING_RECV_RETURNS_WOULDBLOCK = '''
import socket
import poller

def run() -> Result[i32, OSError]:
	server: socket.Socket = socket.Socket.tcp().or_return()
	server.bind( '127.0.0.1', u16( 0 )).or_return()
	server.listen().or_return()
	bound: socket.SocketAddr = server.getsockname().or_return()

	client: socket.Socket = socket.Socket.tcp().or_return()
	client.connect( '127.0.0.1', bound.port() ).or_return()
	( conn, _addr ) = server.accept().or_return()

	poller.set_nonblocking( conn.fileno() ).or_return()

	rbuf: bytearray = bytearray( 16 )
	match conn.recv( rbuf.get_ptr(), usize( 16 )):
		case Result.Ok( _n ):
			return Result.Ok( 1 )   # should never succeed - nothing was sent
		case Result.Err( e ):
			if e != OSError.WouldBlock:
				return Result.Ok( 2 )

	conn.close()
	client.close()
	server.close()
	return Result.Ok( 0 )

def main() -> i32:
	match run():
		case Result.Ok( code ):
			return code
		case Result.Err( _ ):
			return 3
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile poller tests' )
class PollerBehaviorTests( RealCompileMixin, unittest.TestCase ):
	def _run( self, source: str ) -> None:
		compiler = self._compile_source( source )
		c_source = emitter_c.emit_c( compiler )
		self._assert_compiles_and_runs( c_source, expected_exit = 0, compiler = compiler )

	def test_readiness_tracks_real_data( self ) -> None:
		self._run( _READINESS_TRACKS_REAL_DATA )

	def test_unregister_stops_reporting( self ) -> None:
		self._run( _UNREGISTER_STOPS_REPORTING )

	def test_nonblocking_recv_returns_wouldblock( self ) -> None:
		self._run( _NONBLOCKING_RECV_RETURNS_WOULDBLOCK )
