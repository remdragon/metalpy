# tcp_test.py — real compile+link+run coverage for lib/tcp.py's TcpConnection/
# TcpListener, the first Reader/Writer conformers built on
# reactor.wait_for_signal(). Each test's handler code is exercised BOTH
# without a Reactor at all (plain OS threads, genuinely blocking) and inside
# a Reactor (cooperative fibers) - proving the reactor-optional property the
# whole design (see PLAN_NON_BLOCKING_IO.md's design notes) is built around:
# the exact same code must behave identically either way.

import unittest
from pathlib import Path

import test_support
from compiler import Compiler
from discovery import Discovery

@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
class TcpTests( test_support.RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _run( self, code: str ) -> None:
		self.compiler.import_code( code, Path( '__main__.py' ), scope = None )
		self.compiler.run()

	def _echo_harness( self, driver_body: str ) -> str:
		return '''
import compiler
import io
import tcp
import atomic
import threading
import reactor

class Server:
	listener: tcp.TcpListener
	flag:     atomic.Atomic[i32]
	def __init__( self, listener: tcp.TcpListener, flag: atomic.Atomic[i32] ) -> None:
		self.listener = listener
		self.flag = flag
	def run( self ) -> None:
		match self.listener.accept():
			case Result.Ok( conn ):
				match conn.readline():
					case Result.Ok( line ):
						match io.write_all( conn, line.get_const_ptr(), len( line )):
							case Result.Ok( _ ):
								self.flag.store( 1 )
							case Result.Err( _ ):
								self.flag.store( 2 )
					case Result.Err( _ ):
						self.flag.store( 3 )
			case Result.Err( _ ):
				self.flag.store( 4 )

class Client:
	port: u16
	flag: atomic.Atomic[i32]
	def __init__( self, port: u16, flag: atomic.Atomic[i32] ) -> None:
		self.port = port
		self.flag = flag
	def run( self ) -> None:
		match tcp.connect( '127.0.0.1', self.port ):
			case Result.Ok( conn ):
				msg: bytes = b'ping\\n'
				match io.write_all( conn, msg.get_const_ptr(), len( msg )):
					case Result.Ok( _ ):
						pass
					case Result.Err( _ ):
						self.flag.store( 5 )
						return
				match conn.readline():
					case Result.Ok( echoed ):
						if echoed.decode().unwrap( 'decode' ) == 'ping\\n':
							self.flag.store( 1 )
						else:
							self.flag.store( 6 )
					case Result.Err( _ ):
						self.flag.store( 7 )
			case Result.Err( _ ):
				self.flag.store( 8 )

''' + driver_body

	def test_tcp_echo_without_a_reactor( self ) -> None:
		self._run( self._echo_harness( '''
def run() -> Result[i32, OSError]:
	listener: tcp.TcpListener = tcp.TcpListener.bind( '127.0.0.1', u16( 0 )).or_return()
	addr = listener.getsockname().or_return()

	server_flag: atomic.Atomic[i32] = atomic.Atomic[i32]( 0 )
	client_flag: atomic.Atomic[i32] = atomic.Atomic[i32]( 0 )
	server = Server( listener, server_flag )
	client = Client( addr.port, client_flag )
	t_server = threading.Thread( server.run )
	t_client = threading.Thread( client.run )
	t_server.join()
	t_client.join()

	with compiler.wrap_arithmetic:
		if server_flag.load() != 1:
			return Result.Ok( 10 + server_flag.load() )
		if client_flag.load() != 1:
			return Result.Ok( 20 + client_flag.load() )
	return Result.Ok( 0 )

def main() -> i32:
	match run():
		case Result.Ok( code ):
			return code
		case Result.Err( _ ):
			return 90
''' ))
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0, timeout = 20 )

	def test_tcp_echo_inside_a_reactor( self ) -> None:
		self._run( self._echo_harness( '''
def run() -> Result[i32, OSError]:
	listener: tcp.TcpListener = tcp.TcpListener.bind( '127.0.0.1', u16( 0 )).or_return()
	addr = listener.getsockname().or_return()

	server_flag: atomic.Atomic[i32] = atomic.Atomic[i32]( 0 )
	client_flag: atomic.Atomic[i32] = atomic.Atomic[i32]( 0 )
	server = Server( listener, server_flag )
	client = Client( addr.port, client_flag )

	r: reactor.Reactor = reactor.Reactor( 2 )
	r.spawn( server.run )
	r.spawn( client.run )
	r.run()

	with compiler.wrap_arithmetic:
		if server_flag.load() != 1:
			return Result.Ok( 30 + server_flag.load() )
		if client_flag.load() != 1:
			return Result.Ok( 40 + client_flag.load() )
	return Result.Ok( 0 )

def main() -> i32:
	match run():
		case Result.Ok( code ):
			return code
		case Result.Err( _ ):
			return 91
''' ))
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0, timeout = 20 )

def _emit( compiler: Compiler ) -> str:
	import emitter_c
	return emitter_c.emit_c( compiler )

if __name__ == '__main__':
	unittest.main()
