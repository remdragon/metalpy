'''
TcpConnection / TcpListener — Reader/Writer-conforming, reactor-optional
wrappers around lib/socket.py's raw Socket. Every underlying fd is put into
non-blocking mode at construction; read()/write()/accept() retry through
reactor.wait_for_signal() on WouldBlock - so the exact same call looks like
an ordinary blocking call whether or not a Reactor is driving this thread
(see reactor.wait_for_signal's own docstring for why).

Known gap, deliberately not built here: connect() itself still blocks the
calling OS thread even inside a Reactor - a genuinely non-blocking connect
needs EINPROGRESS handling + waiting for writability + a getsockopt(SO_ERROR)
check, none of which Socket exposes today. Everything downstream of a
successful connect (read/write/accept) is fully reactor-optional; only the
initial handshake is not, yet.
'''

import compiler
import io
import poller
import reactor
import socket


class TcpConnection( io.Reader, io.Writer ):
	__sock: socket.Socket

	def read( self, buf: Ptr[u8], count: usize ) -> Result[usize, OSError]:
		while True:
			match self.__sock.recv( buf, count ):
				case Result.Ok( n ):
					return Result.Ok( n )
				case Result.Err( e ):
					if e != OSError.WouldBlock:
						return Result.Err( e )
			match reactor.wait_for_signal( reactor.fd_signal( self.__sock.fileno(), True, False )):
				case Result.Ok( _ ):
					pass
				case Result.Err( _ ):
					return Result.Err( OSError.Interrupted )

	def write( self, buf: ConstPtr[u8], count: usize ) -> Result[usize, OSError]:
		while True:
			match self.__sock.send( buf, count ):
				case Result.Ok( n ):
					return Result.Ok( n )
				case Result.Err( e ):
					if e != OSError.WouldBlock:
						return Result.Err( e )
			match reactor.wait_for_signal( reactor.fd_signal( self.__sock.fileno(), False, True )):
				case Result.Ok( _ ):
					pass
				case Result.Err( _ ):
					return Result.Err( OSError.Interrupted )

	def fileno( self ) -> socket.SOCKET:
		return self.__sock.fileno()

	def close( self ) -> None:
		self.__sock.close()

	@private
	@staticmethod
	def _from_socket( sock: socket.Socket ) -> Result[TcpConnection, OSError]:
		poller.set_nonblocking( sock.fileno() ).or_return()
		return Result.Ok( TcpConnection.__allocate__( __sock = sock ))


class TcpListener:
	__sock: socket.Socket

	def fileno( self ) -> socket.SOCKET:
		return self.__sock.fileno()

	def getsockname( self ) -> Result[socket.SocketAddr, OSError]:
		return self.__sock.getsockname()

	def close( self ) -> None:
		self.__sock.close()

	def accept( self ) -> Result[TcpConnection, OSError]:
		while True:
			match self.__sock.accept():
				case Result.Ok( pair ):
					( conn, _addr ) = pair
					return TcpConnection._from_socket( conn )
				case Result.Err( e ):
					if e != OSError.WouldBlock:
						return Result.Err( e )
			match reactor.wait_for_signal( reactor.fd_signal( self.__sock.fileno(), True, False )):
				case Result.Ok( _ ):
					pass
				case Result.Err( _ ):
					return Result.Err( OSError.Interrupted )

	@staticmethod
	def bind( host: str, port: u16, backlog: i32 = 128 ) -> Result[TcpListener, OSError]:
		sock: socket.Socket = socket.Socket.tcp().or_return()
		sock.bind( host, port ).or_return()
		sock.listen( backlog ).or_return()
		poller.set_nonblocking( sock.fileno() ).or_return()
		return Result.Ok( TcpListener.__allocate__( __sock = sock ))


def connect( host: str, port: u16 ) -> Result[TcpConnection, OSError]:
	sock: socket.Socket = socket.Socket.tcp().or_return()
	sock.connect( host, port ).or_return()
	return TcpConnection._from_socket( sock )
