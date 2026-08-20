'''
Reader / Writer / Seekable — structural protocols for blocking-shaped byte
I/O, plus generic helpers (read_exact, write_all) written once against them.

Deliberately NOT "non-blocking" protocols: a conforming read()/write() always
looks blocking to its caller (returns once data is available/written, an
error occurred, or EOF) whether it gets there via a single already-ready OS
call (BinaryReader/BinaryWriter over a regular file - see
lib/builtins/__File.py) or by retrying through reactor.wait_for_signal() when
WouldBlock is possible (TcpConnection - see lib/tcp.py). That difference is
each conformer's own business, not something this module's interface needs
to know about - see reactor.wait_for_signal's own docstring for why this is
what lets the exact same handler code run with or without a Reactor.

Reader and Writer are separate protocols, not one combined interface,
because not everything that can be read from can also be written to (or vice
versa) - a read-only file conforms to Reader alone, a write-only file to
Writer alone, and only something genuinely bidirectional (a socket, or a
file opened read+write) conforms to both.
'''

import compiler
import fs


@protocol
class Reader:
	def read( self, buf: Ptr[u8], count: usize ) -> Result[usize, OSError]:
		...

	def readuntil( self, delim: u8 ) -> Result[bytearray, OSError]:
		''' Reads until a byte equal to `delim` is seen (included in the
		result) or EOF. Byte-at-a-time: @protocol default methods can't hold
		state, so this can never look ahead past what it actually needs -
		which is exactly what makes a single-byte delimiter free to check
		this way, no pushback/internal buffering required. Growth is
		amortized-doubling via bytearray.resize() (not a reallocation per
		byte). Mirrors Python's own file.readline(): EOF with nothing
		collected yet returns Ok(empty), EOF mid-sequence returns whatever
		was collected so far - never treated as an error itself. '''
		cap: usize = usize( 64 )
		out: bytearray = bytearray( cap )
		length: usize = 0
		b: u8 = 0
		while True:
			n: usize = self.read( compiler.addrof( b ), usize( 1 )).or_return()
			if n == 0:
				out.resize( length )
				return Result.Ok( out )
			if length == cap:
				with compiler.wrap_arithmetic:
					cap = cap * usize( 2 )
				out.resize( cap )
			out[length] = b
			with compiler.wrap_arithmetic:
				length += usize( 1 )
			if b == delim:
				out.resize( length )
				return Result.Ok( out )

	def readline( self ) -> Result[bytearray, OSError]:
		return self.readuntil( u8( 10 )) # b'\n'


@protocol
class Writer:
	def write( self, buf: ConstPtr[u8], count: usize ) -> Result[usize, OSError]:
		...


@protocol
class Seekable:
	def seek( self, offset: i64, whence: i32 ) -> Result[i64, OSError]:
		...

	def tell( self ) -> Result[i64, OSError]:
		return self.seek( i64( 0 ), i32( 1 )) # SEEK_CUR == 1 on every supported target (fs.SEEK_CUR)


def read_exact[T: Reader]( src: T, buf: Ptr[u8], count: usize ) -> Result[None, OSError]:
	''' loops read() until buf[0:count) is entirely filled, or EOF/error -
	read() itself can do short reads, same reasoning as write_all below. A
	0-length read before count is reached (EOF) is reported as
	OSError.BrokenPipe, matching Socket.send_all's existing convention for
	"the other side stopped producing/accepting data". '''
	got: usize = 0
	with compiler.panic_arithmetic( 'bounded by count, cannot overflow' ):
		while got < count:
			n: usize = src.read( buf + got, count - got ).or_return()
			if n == 0:
				return Result.Err( OSError.BrokenPipe )
			got += n
	return Result.Ok( None )


def write_all[T: Writer]( dst: T, buf: ConstPtr[u8], count: usize ) -> Result[None, OSError]:
	''' loops write() until every byte in buf[0:count) is sent, or an error
	occurs. Written once against the Writer protocol so every conformer
	(TcpConnection, BinaryWriter, BinaryReadWriter, future implementers)
	shares it, instead of each hand-rolling its own copy (lib/fs.py's own
	write_all and Socket.send_all predate this protocol and stay as
	raw-FD-only / Socket-only conveniences - not worth churning callers of
	either just to delete the duplication). '''
	sent: usize = 0
	with compiler.panic_arithmetic( 'bounded by count, cannot overflow' ):
		while sent < count:
			n: usize = dst.write( buf + sent, count - sent ).or_return()
			if n == 0:
				return Result.Err( OSError.BrokenPipe )
			sent += n
	return Result.Ok( None )
