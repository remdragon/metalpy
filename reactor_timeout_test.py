# reactor_timeout_test.py — real compile+link+run coverage for
# reactor.timeout() / the fiber.Fiber.get_deadline()/set_deadline()
# mechanism it's built on. Motivating use case: a server bounding an idle
# connection's read (hangup) or resetting a bound on each keepalive.

import unittest
from pathlib import Path

import test_support
from compiler import Compiler
from discovery import Discovery

@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
class ReactorTimeoutTests( test_support.RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _run( self, code: str ) -> None:
		self.compiler.import_code( code, Path( '__main__.py' ), scope = None )
		self.compiler.run()

	def test_timeout_fires_on_an_idle_connection( self ) -> None:
		self._run( '''
import compiler
import tcp
import reactor
import atomic
from datetime import timedelta

class Server:
	listener: tcp.TcpListener
	flag:     atomic.Atomic[i32]
	def __init__( self, listener: tcp.TcpListener, flag: atomic.Atomic[i32] ) -> None:
		self.listener = listener
		self.flag = flag
	def run( self ) -> None:
		match self.listener.accept():
			case Result.Ok( conn ):
				with reactor.timeout( timedelta( milliseconds = 150 )):
					match conn.readline():
						case Result.Ok( _line ):
							self.flag.store( 1 )
						case Result.Err( e ):
							if e == OSError.TimedOut:
								self.flag.store( 2 )
							else:
								self.flag.store( 3 )
			case Result.Err( _ ):
				self.flag.store( 4 )

class IdleClient:
	port: u16
	def __init__( self, port: u16 ) -> None:
		self.port = port
	def run( self ) -> None:
		match tcp.connect( '127.0.0.1', self.port ):
			case Result.Ok( conn ):
				busy_delay()
			case Result.Err( _ ):
				pass

def busy_delay() -> None:
	i: usize = 0
	while i < usize( 400000000 ):
		with compiler.wrap_arithmetic:
			i = i + 1

def run() -> Result[i32, OSError]:
	listener: tcp.TcpListener = tcp.TcpListener.bind( '127.0.0.1', u16( 0 )).or_return()
	addr = listener.getsockname().or_return()
	flag: atomic.Atomic[i32] = atomic.Atomic[i32]( 0 )
	server = Server( listener, flag )
	client = IdleClient( addr.port() )
	r: reactor.Reactor = reactor.Reactor( 2 )
	r.spawn( server.run )
	r.spawn( client.run )
	r.run()
	if flag.load() != 2:
		return Result.Ok( flag.load() )
	return Result.Ok( 0 )

def main() -> i32:
	match run():
		case Result.Ok( code ):
			return code
		case Result.Err( _ ):
			return 90
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0, timeout = 20 )

	def test_timeout_does_not_fire_when_data_arrives_in_time( self ) -> None:
		self._run( '''
import compiler
import io
import tcp
import reactor
import atomic
from datetime import timedelta

class Server:
	listener: tcp.TcpListener
	flag:     atomic.Atomic[i32]
	def __init__( self, listener: tcp.TcpListener, flag: atomic.Atomic[i32] ) -> None:
		self.listener = listener
		self.flag = flag
	def run( self ) -> None:
		match self.listener.accept():
			case Result.Ok( conn ):
				with reactor.timeout( timedelta( seconds = 10 )):
					match conn.readline():
						case Result.Ok( line ):
							if line.decode().unwrap( 'decode' ) == 'hi\\n':
								self.flag.store( 1 )
							else:
								self.flag.store( 2 )
						case Result.Err( _ ):
							self.flag.store( 3 )
			case Result.Err( _ ):
				self.flag.store( 4 )

class PromptClient:
	port: u16
	def __init__( self, port: u16 ) -> None:
		self.port = port
	def run( self ) -> None:
		match tcp.connect( '127.0.0.1', self.port ):
			case Result.Ok( conn ):
				msg: bytes = b'hi\\n'
				io.write_all( conn, msg.get_const_ptr(), len( msg )).unwrap( 'write_all' )
			case Result.Err( _ ):
				pass

def run() -> Result[i32, OSError]:
	listener: tcp.TcpListener = tcp.TcpListener.bind( '127.0.0.1', u16( 0 )).or_return()
	addr = listener.getsockname().or_return()
	flag: atomic.Atomic[i32] = atomic.Atomic[i32]( 0 )
	server = Server( listener, flag )
	client = PromptClient( addr.port() )
	r: reactor.Reactor = reactor.Reactor( 2 )
	r.spawn( server.run )
	r.spawn( client.run )
	r.run()
	if flag.load() != 1:
		return Result.Ok( flag.load() )
	return Result.Ok( 0 )

def main() -> i32:
	match run():
		case Result.Ok( code ):
			return code
		case Result.Err( _ ):
			return 90
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0, timeout = 20 )

	def test_nested_timeout_narrows_to_the_inner_bound( self ) -> None:
		self._run( '''
import compiler
import tcp
import reactor
import atomic
from datetime import timedelta

class Server:
	listener: tcp.TcpListener
	flag:     atomic.Atomic[i32]
	def __init__( self, listener: tcp.TcpListener, flag: atomic.Atomic[i32] ) -> None:
		self.listener = listener
		self.flag = flag
	def run( self ) -> None:
		match self.listener.accept():
			case Result.Ok( conn ):
				# outer bound is generous (10s) - only the INNER 150ms bound
				# should ever be able to fire during this test's own run
				with reactor.timeout( timedelta( seconds = 10 )):
					with reactor.timeout( timedelta( milliseconds = 150 )):
						match conn.readline():
							case Result.Ok( _line ):
								self.flag.store( 1 )
							case Result.Err( e ):
								if e == OSError.TimedOut:
									self.flag.store( 2 )
								else:
									self.flag.store( 3 )
			case Result.Err( _ ):
				self.flag.store( 4 )

class IdleClient:
	port: u16
	def __init__( self, port: u16 ) -> None:
		self.port = port
	def run( self ) -> None:
		match tcp.connect( '127.0.0.1', self.port ):
			case Result.Ok( conn ):
				busy_delay()
			case Result.Err( _ ):
				pass

def busy_delay() -> None:
	i: usize = 0
	while i < usize( 400000000 ):
		with compiler.wrap_arithmetic:
			i = i + 1

def run() -> Result[i32, OSError]:
	listener: tcp.TcpListener = tcp.TcpListener.bind( '127.0.0.1', u16( 0 )).or_return()
	addr = listener.getsockname().or_return()
	flag: atomic.Atomic[i32] = atomic.Atomic[i32]( 0 )
	server = Server( listener, flag )
	client = IdleClient( addr.port() )
	r: reactor.Reactor = reactor.Reactor( 2 )
	r.spawn( server.run )
	r.spawn( client.run )
	r.run()
	if flag.load() != 2:
		return Result.Ok( flag.load() )
	return Result.Ok( 0 )

def main() -> i32:
	match run():
		case Result.Ok( code ):
			return code
		case Result.Err( _ ):
			return 90
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0, timeout = 20 )

	def test_timeout_works_without_a_reactor( self ) -> None:
		self._run( '''
import compiler
import tcp
import reactor
import threading
import atomic
from datetime import timedelta

class Server:
	listener: tcp.TcpListener
	flag:     atomic.Atomic[i32]
	def __init__( self, listener: tcp.TcpListener, flag: atomic.Atomic[i32] ) -> None:
		self.listener = listener
		self.flag = flag
	def run( self ) -> None:
		match self.listener.accept():
			case Result.Ok( conn ):
				with reactor.timeout( timedelta( milliseconds = 150 )):
					match conn.readline():
						case Result.Ok( _line ):
							self.flag.store( 1 )
						case Result.Err( e ):
							if e == OSError.TimedOut:
								self.flag.store( 2 )
							else:
								self.flag.store( 3 )
			case Result.Err( _ ):
				self.flag.store( 4 )

class IdleClient:
	port: u16
	def __init__( self, port: u16 ) -> None:
		self.port = port
	def run( self ) -> None:
		match tcp.connect( '127.0.0.1', self.port ):
			case Result.Ok( conn ):
				busy_delay()
			case Result.Err( _ ):
				pass

def busy_delay() -> None:
	i: usize = 0
	while i < usize( 400000000 ):
		with compiler.wrap_arithmetic:
			i = i + 1

def run() -> Result[i32, OSError]:
	listener: tcp.TcpListener = tcp.TcpListener.bind( '127.0.0.1', u16( 0 )).or_return()
	addr = listener.getsockname().or_return()
	flag: atomic.Atomic[i32] = atomic.Atomic[i32]( 0 )
	server = Server( listener, flag )
	client = IdleClient( addr.port() )
	# no Reactor anywhere - both sides run as plain OS threads, exercising
	# _blocking_wait_no_reactor's own deadline handling
	t_server = threading.Thread( server.run )
	t_client = threading.Thread( client.run )
	t_server.join()
	t_client.join()
	if flag.load() != 2:
		return Result.Ok( flag.load() )
	return Result.Ok( 0 )

def main() -> i32:
	match run():
		case Result.Ok( code ):
			return code
		case Result.Err( _ ):
			return 90
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0, timeout = 20 )

def _emit( compiler: Compiler ) -> str:
	import emitter_c
	return emitter_c.emit_c( compiler )

if __name__ == '__main__':
	unittest.main()
