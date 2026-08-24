# lib/poller.py — readiness-based I/O event demultiplexer.
#
# Wraps epoll (POSIX/WSL) / WSAPoll (Windows) behind one portable API:
# register(fd, want_read, want_write), unregister(fd), wait(timeout_ms) ->
# list[ReadyEvent]. Deliberately NOT wired into Worker/Reactor yet - this
# is the standalone primitive; the Signal abstraction + Worker scheduling
# loop changes needed to actually block-wait on it are a separate,
# follow-up increment (see lib/reactor.py's own header comment).
#
# epoll (persistent, kernel-side registration - register once, wait many
# times) and WSAPoll (stateless - the FULL fd list is re-sent every call,
# no persistent registration) are fundamentally different shapes. This
# module's own register()/unregister() hide that: POSIX drives epoll_ctl
# directly; Windows maintains its own registered-fd list and rebuilds the
# WSAPOLLFD array from it on every wait() call.

import compiler
import sys

if compiler.target.os == 'windows':
	from windows.ws2_32 import WSAPoll, ioctlsocket, FIONBIO, SOCKET, SOCKET_ERROR

	# winsock2.h's poll event bits - WSAPOLLFD.events/.revents are SHORT (i16).
	# Only the bits this module actually sets/checks - not a full binding
	# (e.g. POLLRDBAND/POLLPRI/POLLNVAL are real winsock2.h members, left
	# out since nothing here uses them).
	POLLRDNORM: i16 = 0x0100
	POLLWRNORM: i16 = 0x0010
	POLLERR:    i16 = 0x0001
	POLLHUP:    i16 = 0x0002

	@cstruct
	class WSAPOLLFD:
		fd:      SOCKET = 0
		events:  i16 = 0
		revents: i16 = 0

	# WSAPoll is stateless (the full fd list is re-sent every call, unlike
	# epoll's persistent kernel-side registration) - Poller.__fds below is
	# this module's own tracked registration list, rebuilt into a real
	# WSAPOLLFD array on every wait(). No inherent cap the way a fixed-N
	# @cstruct-of-named-slots would need (see lib/ssl.py's _SecBuffer4 for
	# that alternative pattern) - sys.alloc[WSAPOLLFD](n) for a real n
	# works directly (confirmed via a real compile+run repro).
else:
	from posix.epoll import (
		epoll_create1, epoll_ctl, epoll_wait, epoll_event,
		EPOLLIN, EPOLLOUT, EPOLLERR, EPOLLHUP, EPOLL_CTL_ADD, EPOLL_CTL_DEL,
	)
	from posix.fcntl import fcntl, F_GETFL, F_SETFL, O_NONBLOCK
	from crt import close
	SOCKET: TypeAlias = i32


class ReadyEvent:
	fd:       SOCKET
	readable: bool
	writable: bool
	def __init__( self, fd: SOCKET, readable: bool, writable: bool ) -> None:
		self.fd = fd
		self.readable = readable
		self.writable = writable


def set_nonblocking( fd: SOCKET ) -> Result[None, OSError]:
	''' puts an already-open socket into non-blocking mode - a prerequisite
	for using it with Poller at all (a blocking recv()/send() defeats the
	whole point of waiting for readiness first). Free function, not a
	Socket method, so it stays usable on a bare fd/SOCKET without needing
	lib/socket.py to depend on lib/poller.py. '''
	return _set_nonblocking( fd )

@compiler.target( os = 'windows' )
def _set_nonblocking( fd: SOCKET ) -> Result[None, OSError]:
	from windows.ws2_32 import WSAGetLastError
	mode: u32 = 1   # non-zero = non-blocking, per ioctlsocket(FIONBIO) docs
	if ioctlsocket( fd, FIONBIO, compiler.addrof( mode )) != 0:
		return Result.Err( OSError( WSAGetLastError() ))
	return Result.Ok( None )

@compiler.target( os = not 'windows' )
def _set_nonblocking( fd: SOCKET ) -> Result[None, OSError]:
	from crt import get_errno
	flags: i32 = fcntl( fd, F_GETFL, 0 )
	if flags < 0:
		return Result.Err( OSError( get_errno() ))
	with compiler.wrap_arithmetic:
		new_flags: i32 = flags | O_NONBLOCK
	if fcntl( fd, F_SETFL, new_flags ) < 0:
		return Result.Err( OSError( get_errno() ))
	return Result.Ok( None )


@compiler.target( os = not 'windows' )
class Poller:
	__epfd: i32
	# single reusable 1-slot buffer for both epoll_ctl's `event` argument
	# and epoll_wait's own `events` out-array - epoll_wait is called with
	# maxevents=1 repeatedly (see wait()'s own docstring for why), so one
	# slot is always enough; avoids the array-of-struct/multi-element
	# allocation question entirely for a v1 poller.
	__buf: Ptr[epoll_event]

	def __init__( self ) -> None:
		epfd: i32 = epoll_create1( 0 )
		if epfd < 0:
			sys.panic( 'Poller.__init__: epoll_create1 failed' )
		self.__epfd = epfd
		self.__buf = sys.alloc[epoll_event]( 1 )

	def __del__( self ) -> None:
		close( self.__epfd )
		sys.free( compiler.cast( Ptr[None], self.__buf ))

	def register( self, fd: SOCKET, want_read: bool, want_write: bool ) -> Result[None, OSError]:
		events: u32 = 0
		if want_read:
			events = events | EPOLLIN
		if want_write:
			events = events | EPOLLOUT
		compiler.c_field_set( self.__buf, 'events', events )
		# data is epoll_data_t, a union (void*/int/u32/u64) - only ever
		# used here to carry the registered fd back out on the matching
		# ready event, so a raw i32 write/read through the union's own
		# address (whatever its "real" declared member) is correct: the
		# same bytes come back out the same way they went in, and nothing
		# else ever touches this field.
		data_ptr: Ptr[i32] = compiler.cast( Ptr[i32], compiler.c_field_addr( self.__buf, 'data', Ptr[None] ))
		data_ptr[0] = fd
		if epoll_ctl( self.__epfd, EPOLL_CTL_ADD, fd, self.__buf ) != 0:
			from crt import get_errno
			return Result.Err( OSError( get_errno() ))
		return Result.Ok( None )

	def unregister( self, fd: SOCKET ) -> Result[None, OSError]:
		if epoll_ctl( self.__epfd, EPOLL_CTL_DEL, fd, None ) != 0:
			from crt import get_errno
			return Result.Err( OSError( get_errno() ))
		return Result.Ok( None )

	def wait( self, timeout_ms: i32 ) -> Result[list[ReadyEvent], OSError]:
		out = list[ReadyEvent]()
		n: i32 = epoll_wait( self.__epfd, self.__buf, 1, timeout_ms )
		if n < 0:
			from crt import get_errno
			return Result.Err( OSError( get_errno() ))
		if n == 0:
			return Result.Ok( out )
		ev: u32 = compiler.c_field( self.__buf, 'events', u32 )
		data_ptr: Ptr[i32] = compiler.cast( Ptr[i32], compiler.c_field_addr( self.__buf, 'data', Ptr[None] ))
		fd: SOCKET = data_ptr[0]
		out.append( ReadyEvent(
			fd = fd,
			readable = ( ev & ( EPOLLIN | EPOLLERR | EPOLLHUP )) != 0,
			writable = ( ev & EPOLLOUT ) != 0,
		))
		return Result.Ok( out )


@compiler.target( os = 'windows' )
def _elem_at( buf: Ptr[WSAPOLLFD], i: usize ) -> Ptr[WSAPOLLFD]:
	with compiler.wrap_arithmetic:
		offset: usize = i * compiler.sizeof( WSAPOLLFD )
	return compiler.cast( Ptr[WSAPOLLFD], compiler.wrapped_ptr_add( compiler.cast( Ptr[u8], buf ), offset ))

@compiler.target( os = 'windows' )
class Poller:
	__fds: list[WSAPOLLFD]

	def __init__( self ) -> None:
		self.__fds = list[WSAPOLLFD]()

	def register( self, fd: SOCKET, want_read: bool, want_write: bool ) -> Result[None, OSError]:
		events: i16 = 0
		if want_read:
			events = events | POLLRDNORM
		if want_write:
			events = events | POLLWRNORM
		self.__fds.append( WSAPOLLFD( fd = fd, events = events ))
		return Result.Ok( None )

	def unregister( self, fd: SOCKET ) -> Result[None, OSError]:
		kept = list[WSAPOLLFD]()
		n: usize = self.__fds.__len__()
		i: usize = 0
		while i < n:
			entry: WSAPOLLFD = self.__fds.__getitem__( i ).unwrap( 'Poller.unregister: index in bounds by construction' )
			if entry.fd != fd:
				kept.append( entry )
			with compiler.wrap_arithmetic:
				i = i + 1
		self.__fds = kept
		return Result.Ok( None )

	def wait( self, timeout_ms: i32 ) -> Result[list[ReadyEvent], OSError]:
		n: usize = self.__fds.__len__()
		out = list[ReadyEvent]()
		if n == 0:
			return Result.Ok( out )
		buf: Ptr[WSAPOLLFD] = sys.alloc[WSAPOLLFD]( n )
		i: usize = 0
		while i < n:
			entry: WSAPOLLFD = self.__fds.__getitem__( i ).unwrap( 'Poller.wait: index in bounds by construction' )
			slot: Ptr[WSAPOLLFD] = _elem_at( buf, i )
			slot.fd = entry.fd
			slot.events = entry.events
			slot.revents = 0
			with compiler.wrap_arithmetic:
				i = i + 1
		with compiler.panic_arithmetic( 'Poller.wait: registered-fd count too large for WSAPoll (u32)' ):
			fds_count: u32 = u32( n )
		rc: i32 = WSAPoll( compiler.cast( Ptr[None], buf ), fds_count, timeout_ms )
		if rc == SOCKET_ERROR:
			from windows.ws2_32 import WSAGetLastError
			err: OSError = OSError( WSAGetLastError() )
			sys.free( compiler.cast( Ptr[None], buf ))
			return Result.Err( err )
		if rc > 0:
			i = 0
			while i < n:
				# distinct name from the fill loop's own `slot` above - a
				# variable's type is only ever declared once per function
				ready_slot: Ptr[WSAPOLLFD] = _elem_at( buf, i )
				revents: i16 = ready_slot.revents
				if revents != 0:
					out.append( ReadyEvent(
						fd = ready_slot.fd,
						readable = ( revents & ( POLLRDNORM | POLLERR | POLLHUP )) != 0,
						writable = ( revents & POLLWRNORM ) != 0,
					))
				with compiler.wrap_arithmetic:
					i = i + 1
		sys.free( compiler.cast( Ptr[None], buf ))
		return Result.Ok( out )
