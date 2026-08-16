# lib/socket.py — blocking TCP/UDP sockets over IPv4/IPv6, Python's `socket`
# module scoped down to what a hand-FFI'd systems language can support in one
# pass (see PLAN_SOCKET.md-equivalent design notes in the commit history).
#
# Three-layer structure, mirroring lib/fs.py -> lib/builtins/__File.py exactly:
#   1. raw C/Winsock declarations (below, OS-dispatched by name)
#   2. per-OS `_raw` wrappers returning Result[T, OSError]
#   3. Socket / SocketAddr — the public, portable API
#
# v1 scope (confirmed with the user): AF_INET + AF_INET6, SOCK_STREAM +
# SOCK_DGRAM, blocking calls only. Addresses are host: str, port: u16 with
# host a pre-resolved IPv4/IPv6 literal (no DNS/getaddrinfo - a self-contained
# follow-up). No Unix domain sockets, no raw sockets, no non-blocking/select,
# no generic setsockopt (only a narrow set_reuseaddr()).

import compiler
import sys
from atomic import Atomic

# ---------------------------------------------------------------------------
# Layer 1 — raw declarations, OS-dispatched by name
#
# Winsock and BSD sockets share almost the exact same call shape (same
# params, same order) - the one place they genuinely differ, the address-
# length parameter's type (Winsock: int, POSIX: socklen_t/u32), is absorbed
# inside layer 2's per-OS bodies, not here. So layer 1 just needs ONE name
# per operation, bound to the right implementation per target - confirmed
# workable (a real compile) to declare @extern functions and TypeAlias/const
# bindings directly inside a module-level if/else: compile_time_transformer
# collapses it to exactly one branch, same as lib/fs.py's own FD TypeAlias.
# ---------------------------------------------------------------------------

if compiler.target.os == 'windows':
	from windows.ws2_32 import (
		SOCKET, INVALID_SOCKET,
		WSAStartup, WSAGetLastError,
		socket as _c_socket, closesocket as _c_closesocket,
		bind as _c_bind, listen as _c_listen, accept as _c_accept,
		connect as _c_connect, send as _c_send, recv as _c_recv,
		sendto as _c_sendto, recvfrom as _c_recvfrom,
		shutdown as _c_shutdown, getsockname as _c_getsockname,
		setsockopt as _c_setsockopt,
		inet_pton, inet_ntop,
	)
else:
	SOCKET: TypeAlias = i32
	INVALID_SOCKET: SOCKET = i32( -1 )

	@extern( 'c', 'socket' )
	def _c_socket( domain: i32, type: i32, protocol: i32 ) -> i32:
		...
	@extern( 'c', 'close' )
	def _c_closesocket( fd: i32 ) -> i32:
		...
	@extern( 'c', 'bind' )
	def _c_bind( sockfd: i32, addr: Ptr[None], addrlen: u32 ) -> i32:
		...
	@extern( 'c', 'listen' )
	def _c_listen( sockfd: i32, backlog: i32 ) -> i32:
		...
	@extern( 'c', 'accept' )
	def _c_accept( sockfd: i32, addr: Ptr[None], addrlen: Ptr[u32] ) -> i32:
		...
	@extern( 'c', 'connect' )
	def _c_connect( sockfd: i32, addr: Ptr[None], addrlen: u32 ) -> i32:
		...
	@extern( 'c', 'send' )
	def _c_send( sockfd: i32, buf: ConstPtr[u8], len: usize, flags: i32 ) -> isize:
		...
	@extern( 'c', 'recv' )
	def _c_recv( sockfd: i32, buf: Ptr[u8], len: usize, flags: i32 ) -> isize:
		...
	@extern( 'c', 'sendto' )
	def _c_sendto( sockfd: i32, buf: ConstPtr[u8], len: usize, flags: i32, dest_addr: Ptr[None], addrlen: u32 ) -> isize:
		...
	@extern( 'c', 'recvfrom' )
	def _c_recvfrom( sockfd: i32, buf: Ptr[u8], len: usize, flags: i32, src_addr: Ptr[None], addrlen: Ptr[u32] ) -> isize:
		...
	@extern( 'c', 'shutdown' )
	def _c_shutdown( sockfd: i32, how: i32 ) -> i32:
		...
	@extern( 'c', 'getsockname' )
	def _c_getsockname( sockfd: i32, addr: Ptr[None], addrlen: Ptr[u32] ) -> i32:
		...
	@extern( 'c', 'setsockopt' )
	def _c_setsockopt( sockfd: i32, level: i32, optname: i32, optval: ConstPtr[u8], optlen: u32 ) -> i32:
		...
	@extern( 'c', 'inet_pton' )
	def inet_pton( family: i32, src: ConstPtr[u8], dst: Ptr[None] ) -> i32:
		...
	# Plain ConstPtr[u8] return, not ConstPtr[u8]|None - see
	# lib/windows/ws2_32.py's inet_ntop for why (a real compiler bug with
	# T|None as an @extern return type).
	@extern( 'c', 'inet_ntop' )
	def inet_ntop( family: i32, src: Ptr[None], dst: Ptr[u8], size: u32 ) -> ConstPtr[u8]:
		...


# ---------------------------------------------------------------------------
# Constants — via compiler.cexpr, not hardcoded. Load-bearing (not just
# style) for SOL_SOCKET/SO_REUSEADDR specifically: those genuinely differ
# between Linux and macOS/BSD, and the existing `os = not 'windows'` split
# lumps both into one branch (fine for fs.py's O_RDONLY etc, which *are*
# identical across both - not fine here). cexpr queries the real build
# host's headers, sidestepping a 3-way branch.
# ---------------------------------------------------------------------------

if compiler.target.os == 'windows':
	AF_INET:      i32 = compiler.cexpr( 'AF_INET',      'winsock2.h', i32 )
	AF_INET6:     i32 = compiler.cexpr( 'AF_INET6',     'winsock2.h', i32 )
	SOCK_STREAM:  i32 = compiler.cexpr( 'SOCK_STREAM',  'winsock2.h', i32 )
	SOCK_DGRAM:   i32 = compiler.cexpr( 'SOCK_DGRAM',   'winsock2.h', i32 )
	SOL_SOCKET:   i32 = compiler.cexpr( 'SOL_SOCKET',   'winsock2.h', i32 )
	SO_REUSEADDR: i32 = compiler.cexpr( 'SO_REUSEADDR', 'winsock2.h', i32 )
	SHUT_RD:      i32 = compiler.cexpr( 'SD_RECEIVE',   'winsock2.h', i32 )
	SHUT_WR:      i32 = compiler.cexpr( 'SD_SEND',      'winsock2.h', i32 )
	SHUT_RDWR:    i32 = compiler.cexpr( 'SD_BOTH',      'winsock2.h', i32 )
else:
	AF_INET:      i32 = compiler.cexpr( 'AF_INET',      'sys/socket.h', i32 )
	AF_INET6:     i32 = compiler.cexpr( 'AF_INET6',     'sys/socket.h', i32 )
	SOCK_STREAM:  i32 = compiler.cexpr( 'SOCK_STREAM',  'sys/socket.h', i32 )
	SOCK_DGRAM:   i32 = compiler.cexpr( 'SOCK_DGRAM',   'sys/socket.h', i32 )
	SOL_SOCKET:   i32 = compiler.cexpr( 'SOL_SOCKET',   'sys/socket.h', i32 )
	SO_REUSEADDR: i32 = compiler.cexpr( 'SO_REUSEADDR', 'sys/socket.h', i32 )
	SHUT_RD:      i32 = compiler.cexpr( 'SHUT_RD',      'sys/socket.h', i32 )
	SHUT_WR:      i32 = compiler.cexpr( 'SHUT_WR',      'sys/socket.h', i32 )
	SHUT_RDWR:    i32 = compiler.cexpr( 'SHUT_RDWR',    'sys/socket.h', i32 )


# ---------------------------------------------------------------------------
# SockAddrIn / SockAddrIn6 — textbook-stable BSD sockets ABI, identical on
# Windows/Linux/macOS. sin_zero/sin6_addr are unrolled into individual u8
# fields, NOT a `u8[N]` fixed-size array: SYNTAX.md documents that array
# syntax, but a real compile confirms discovery.py doesn't implement it yet
# ("intrinsics.u8 is not generic, cannot subscript it") - lib/guid.py hit
# the identical gap for GUID's Data4[8] and unrolled the same way; this
# follows that established workaround, not a new one.
# ---------------------------------------------------------------------------

@cstruct
class SockAddrIn:
	sin_family: u16 = 0
	sin_port:   u16 = 0  # network byte order
	sin_addr:   u32 = 0  # network byte order
	sin_zero_0: u8 = 0
	sin_zero_1: u8 = 0
	sin_zero_2: u8 = 0
	sin_zero_3: u8 = 0
	sin_zero_4: u8 = 0
	sin_zero_5: u8 = 0
	sin_zero_6: u8 = 0
	sin_zero_7: u8 = 0

@cstruct
class SockAddrIn6:
	sin6_family:   u16 = 0
	sin6_port:     u16 = 0  # network byte order
	sin6_flowinfo: u32 = 0
	sin6_addr_0:   u8 = 0
	sin6_addr_1:   u8 = 0
	sin6_addr_2:   u8 = 0
	sin6_addr_3:   u8 = 0
	sin6_addr_4:   u8 = 0
	sin6_addr_5:   u8 = 0
	sin6_addr_6:   u8 = 0
	sin6_addr_7:   u8 = 0
	sin6_addr_8:   u8 = 0
	sin6_addr_9:   u8 = 0
	sin6_addr_10:  u8 = 0
	sin6_addr_11:  u8 = 0
	sin6_addr_12:  u8 = 0
	sin6_addr_13:  u8 = 0
	sin6_addr_14:  u8 = 0
	sin6_addr_15:  u8 = 0
	sin6_scope_id: u32 = 0


def _htons( port: u16 ) -> u16:
	''' host to network byte order. Pure bit-twiddling, no FFI needed:
	every supported arch (x64/arm64) is little-endian in practice, network
	byte order is always big-endian, and no big-endian target exists here
	to make this wrong. No htonl equivalent needed - inet_pton/inet_ntop
	already read/write sin_addr/sin6_addr directly in network byte order. '''
	with compiler.wrap_arithmetic:
		return ( port << 8 ) | ( port >> 8 )


# ---------------------------------------------------------------------------
# SocketAddr — a plain RC class, NOT a @struct/@cstruct. Load-bearing: per
# PLAN_TUPLE.md, @struct/@cunion fields aren't tracked by the CFG's RC
# decref walk today, so a struct-typed SocketAddr holding a `str` field
# would silently leak that string every time a value goes out of scope.
# ---------------------------------------------------------------------------

class SocketAddr:
	__host: str
	__port: u16

	def host( self ) -> str:
		return self.__host

	def port( self ) -> u16:
		return self.__port

	@private
	@staticmethod
	def _from_parts( host: str, port: u16 ) -> SocketAddr:
		return SocketAddr.__allocate__( __host = host, __port = port )


# _err_invalid() — a real, pre-existing compiler bug (confirmed with a
# minimal repro outside this file): a bare enum-member reference like
# `OSError.Invalid` passed directly as a Result.Err(...) argument confuses
# generic type inference between the enum and its own u32 backing type
# ("type parameter 'E' is inferred as both builtins.OSError and
# intrinsics.u32"). Staging the same value through an explicitly-typed local
# first avoids it entirely, so every `Result.Err(OSError.Invalid)` in this
# file goes through this one helper instead of the bare form.
def _err_invalid() -> OSError:
	err: OSError = OSError.Invalid
	return err


# ---------------------------------------------------------------------------
# Address helpers — build a sockaddr from (host, port), decode one back to
# a SocketAddr. compiler.addrof() only accepts a bare local variable (not a
# field-access expression like `addr.sin_addr` - confirmed by a real compile
# error), so these always stage the raw address bytes into their own local
# before folding them into a struct via field=value construction, never
# mutate a field in place.
# ---------------------------------------------------------------------------

def _build_sockaddr_in( host: str, port: u16 ) -> Result[SockAddrIn, OSError]:
	ip_addr: u32 = 0
	rc: i32 = inet_pton( AF_INET, host.get_cstr(), compiler.cast( Ptr[None], compiler.addrof( ip_addr )))
	if rc != 1:
		return Result.Err( _err_invalid() )
	with compiler.wrap_arithmetic:
		family: u16 = u16( AF_INET )
	return Result.Ok( SockAddrIn( sin_family = family, sin_port = _htons( port ), sin_addr = ip_addr ))


def _build_sockaddr_in6( host: str, port: u16 ) -> Result[SockAddrIn6, OSError]:
	buf: bytearray = bytearray( 16 )
	rc: i32 = inet_pton( AF_INET6, host.get_cstr(), compiler.cast( Ptr[None], buf.get_ptr() ))
	if rc != 1:
		return Result.Err( _err_invalid() )
	with compiler.wrap_arithmetic:
		family: u16 = u16( AF_INET6 )
	# bytearray has no scalar __getitem__ (only slice syntax, e.g. buf[:n] -
	# confirmed by a real compile: `buf[i]` treats the whole struct as
	# indexable and fails to compile) - .get_const_ptr()[i] is the real,
	# C-array-style indexed access every other raw-byte-poking site in this
	# codebase already uses (e.g. lib/guid.py's `data[offset]` off a
	# ConstPtr[u8]).
	p: ConstPtr[u8] = buf.get_const_ptr()
	return Result.Ok( SockAddrIn6(
		sin6_family = family, sin6_port = _htons( port ),
		sin6_addr_0 = p[0], sin6_addr_1 = p[1], sin6_addr_2 = p[2], sin6_addr_3 = p[3],
		sin6_addr_4 = p[4], sin6_addr_5 = p[5], sin6_addr_6 = p[6], sin6_addr_7 = p[7],
		sin6_addr_8 = p[8], sin6_addr_9 = p[9], sin6_addr_10 = p[10], sin6_addr_11 = p[11],
		sin6_addr_12 = p[12], sin6_addr_13 = p[13], sin6_addr_14 = p[14], sin6_addr_15 = p[15],
	))


# inet_ntop's size parameter is size_t on Windows but socklen_t (u32) on
# POSIX - a genuine per-platform width difference (confirmed against both
# real prototypes), not a style choice, so unlike most of this file these
# two helpers need one body per OS rather than a single shared one: a bare
# inline if/os-branch inside a function body isn't how @compiler.target
# dispatch works elsewhere in this codebase (it's always a separate
# top-level def per branch, e.g. every _raw wrapper above) - matching that
# established shape rather than guessing at an unproven alternative.

@compiler.target( os = 'windows' )
def _sockaddr_in_to_addr( sa: SockAddrIn ) -> Result[SocketAddr, OSError]:
	addr_val: u32 = sa.sin_addr
	strbuf: bytearray = bytearray( 16 )  # "255.255.255.255\0" fits in 16
	res = inet_ntop( AF_INET, compiler.cast( Ptr[None], compiler.addrof( addr_val )), strbuf.get_ptr(), usize( 16 ))
	if res is None:
		return Result.Err( _err_invalid() )
	from crt import strnlen
	slen: usize = strnlen( strbuf.get_const_ptr(), usize( 16 ))
	with compiler.wrap_arithmetic:
		full_len: usize = slen + usize( 1 )
	match str.from_cstr( strbuf.get_const_ptr(), full_len ):
		case Result.Ok( host ):
			return Result.Ok( SocketAddr._from_parts( host, _htons( sa.sin_port )))
		case Result.Err( _ ):
			return Result.Err( _err_invalid() )

@compiler.target( os = not 'windows' )
def _sockaddr_in_to_addr( sa: SockAddrIn ) -> Result[SocketAddr, OSError]:
	addr_val: u32 = sa.sin_addr
	strbuf: bytearray = bytearray( 16 )  # "255.255.255.255\0" fits in 16
	res = inet_ntop( AF_INET, compiler.cast( Ptr[None], compiler.addrof( addr_val )), strbuf.get_ptr(), u32( 16 ))
	if res is None:
		return Result.Err( _err_invalid() )
	from crt import strnlen
	slen: usize = strnlen( strbuf.get_const_ptr(), usize( 16 ))
	with compiler.wrap_arithmetic:
		full_len: usize = slen + usize( 1 )
	match str.from_cstr( strbuf.get_const_ptr(), full_len ):
		case Result.Ok( host ):
			return Result.Ok( SocketAddr._from_parts( host, _htons( sa.sin_port )))
		case Result.Err( _ ):
			return Result.Err( _err_invalid() )


# bytearray has no scalar __setitem__ either (same gap as __getitem__ above)
# - raw[i] = x below is real C-array-style indexing on a Ptr[u8] (which IS
# supported, unlike bytearray's own struct-boxed []), matching the pattern
# _build_sockaddr_in6 above already settled on.

@compiler.target( os = 'windows' )
def _sockaddr_in6_to_addr( sa: SockAddrIn6 ) -> Result[SocketAddr, OSError]:
	raw: Ptr[u8] = sys.alloc[u8]( 16 )
	raw[0] = sa.sin6_addr_0;  raw[1] = sa.sin6_addr_1;  raw[2] = sa.sin6_addr_2;  raw[3] = sa.sin6_addr_3
	raw[4] = sa.sin6_addr_4;  raw[5] = sa.sin6_addr_5;  raw[6] = sa.sin6_addr_6;  raw[7] = sa.sin6_addr_7
	raw[8] = sa.sin6_addr_8;  raw[9] = sa.sin6_addr_9;  raw[10] = sa.sin6_addr_10; raw[11] = sa.sin6_addr_11
	raw[12] = sa.sin6_addr_12; raw[13] = sa.sin6_addr_13; raw[14] = sa.sin6_addr_14; raw[15] = sa.sin6_addr_15
	strbuf: bytearray = bytearray( 46 )  # INET6_ADDRSTRLEN
	res = inet_ntop( AF_INET6, compiler.cast( Ptr[None], raw ), strbuf.get_ptr(), usize( 46 ))
	sys.free( compiler.cast( Ptr[None], raw ))
	if res is None:
		return Result.Err( _err_invalid() )
	from crt import strnlen
	slen: usize = strnlen( strbuf.get_const_ptr(), usize( 46 ))
	with compiler.wrap_arithmetic:
		full_len: usize = slen + usize( 1 )
	match str.from_cstr( strbuf.get_const_ptr(), full_len ):
		case Result.Ok( host ):
			return Result.Ok( SocketAddr._from_parts( host, _htons( sa.sin6_port )))
		case Result.Err( _ ):
			return Result.Err( _err_invalid() )

@compiler.target( os = not 'windows' )
def _sockaddr_in6_to_addr( sa: SockAddrIn6 ) -> Result[SocketAddr, OSError]:
	raw: Ptr[u8] = sys.alloc[u8]( 16 )
	raw[0] = sa.sin6_addr_0;  raw[1] = sa.sin6_addr_1;  raw[2] = sa.sin6_addr_2;  raw[3] = sa.sin6_addr_3
	raw[4] = sa.sin6_addr_4;  raw[5] = sa.sin6_addr_5;  raw[6] = sa.sin6_addr_6;  raw[7] = sa.sin6_addr_7
	raw[8] = sa.sin6_addr_8;  raw[9] = sa.sin6_addr_9;  raw[10] = sa.sin6_addr_10; raw[11] = sa.sin6_addr_11
	raw[12] = sa.sin6_addr_12; raw[13] = sa.sin6_addr_13; raw[14] = sa.sin6_addr_14; raw[15] = sa.sin6_addr_15
	strbuf: bytearray = bytearray( 46 )  # INET6_ADDRSTRLEN
	res = inet_ntop( AF_INET6, compiler.cast( Ptr[None], raw ), strbuf.get_ptr(), u32( 46 ))
	sys.free( compiler.cast( Ptr[None], raw ))
	if res is None:
		return Result.Err( _err_invalid() )
	from crt import strnlen
	slen: usize = strnlen( strbuf.get_const_ptr(), usize( 46 ))
	with compiler.wrap_arithmetic:
		full_len: usize = slen + usize( 1 )
	match str.from_cstr( strbuf.get_const_ptr(), full_len ):
		case Result.Ok( host ):
			return Result.Ok( SocketAddr._from_parts( host, _htons( sa.sin6_port )))
		case Result.Err( _ ):
			return Result.Err( _err_invalid() )


# ---------------------------------------------------------------------------
# Layer 2 — per-OS `_raw` wrappers, Result[T, OSError]. Structured exactly
# like lib/fs.py's write_raw/read_raw/open_raw/close_raw: same outward
# signature on both branches (SOCKET/Ptr[None]/usize throughout - the
# Winsock-int-vs-POSIX-socklen_t addrlen difference is absorbed HERE, inside
# each branch, via a wrap_arithmetic cast, never leaking outward). Windows
# failure paths call WSAGetLastError() - NOT kernel32.GetLastError(), a real,
# easy-to-get-wrong detail: Winsock reports errors through its own API.
# ---------------------------------------------------------------------------

@compiler.target( os = 'windows' )
def _create_raw( family: i32, type: i32 ) -> Result[SOCKET, OSError]:
	sock: SOCKET = _c_socket( family, type, 0 )
	if sock == INVALID_SOCKET:
		return Result.Err( OSError( WSAGetLastError() ))
	return Result.Ok( sock )

@compiler.target( os = not 'windows' )
def _create_raw( family: i32, type: i32 ) -> Result[SOCKET, OSError]:
	from crt import get_errno
	sock: i32 = _c_socket( family, type, 0 )
	if sock < 0:
		return Result.Err( OSError( get_errno() ))
	return Result.Ok( sock )


@compiler.target( os = 'windows' )
def _close_raw( sock: SOCKET ) -> Result[None, OSError]:
	if _c_closesocket( sock ) != 0:
		return Result.Err( OSError( WSAGetLastError() ))
	return Result.Ok( None )

@compiler.target( os = not 'windows' )
def _close_raw( sock: SOCKET ) -> Result[None, OSError]:
	from crt import get_errno
	if _c_closesocket( sock ) < 0:
		return Result.Err( OSError( get_errno() ))
	return Result.Ok( None )


@compiler.target( os = 'windows' )
def _bind_raw( sock: SOCKET, addr_ptr: Ptr[None], addr_len: usize ) -> Result[None, OSError]:
	with compiler.wrap_arithmetic:
		n: i32 = i32( addr_len )
	if _c_bind( sock, addr_ptr, n ) != 0:
		return Result.Err( OSError( WSAGetLastError() ))
	return Result.Ok( None )

@compiler.target( os = not 'windows' )
def _bind_raw( sock: SOCKET, addr_ptr: Ptr[None], addr_len: usize ) -> Result[None, OSError]:
	from crt import get_errno
	with compiler.wrap_arithmetic:
		n: u32 = u32( addr_len )
	if _c_bind( sock, addr_ptr, n ) < 0:
		return Result.Err( OSError( get_errno() ))
	return Result.Ok( None )


@compiler.target( os = 'windows' )
def _listen_raw( sock: SOCKET, backlog: i32 ) -> Result[None, OSError]:
	if _c_listen( sock, backlog ) != 0:
		return Result.Err( OSError( WSAGetLastError() ))
	return Result.Ok( None )

@compiler.target( os = not 'windows' )
def _listen_raw( sock: SOCKET, backlog: i32 ) -> Result[None, OSError]:
	from crt import get_errno
	if _c_listen( sock, backlog ) < 0:
		return Result.Err( OSError( get_errno() ))
	return Result.Ok( None )


@compiler.target( os = 'windows' )
def _accept_raw( sock: SOCKET, addr_ptr: Ptr[None], addr_cap: usize ) -> Result[SOCKET, OSError]:
	with compiler.wrap_arithmetic:
		cap: i32 = i32( addr_cap )
	conn: SOCKET = _c_accept( sock, addr_ptr, compiler.addrof( cap ))
	if conn == INVALID_SOCKET:
		return Result.Err( OSError( WSAGetLastError() ))
	return Result.Ok( conn )

@compiler.target( os = not 'windows' )
def _accept_raw( sock: SOCKET, addr_ptr: Ptr[None], addr_cap: usize ) -> Result[SOCKET, OSError]:
	from crt import get_errno
	with compiler.wrap_arithmetic:
		cap: u32 = u32( addr_cap )
	conn: i32 = _c_accept( sock, addr_ptr, compiler.addrof( cap ))
	if conn < 0:
		return Result.Err( OSError( get_errno() ))
	return Result.Ok( conn )


@compiler.target( os = 'windows' )
def _connect_raw( sock: SOCKET, addr_ptr: Ptr[None], addr_len: usize ) -> Result[None, OSError]:
	with compiler.wrap_arithmetic:
		n: i32 = i32( addr_len )
	if _c_connect( sock, addr_ptr, n ) != 0:
		return Result.Err( OSError( WSAGetLastError() ))
	return Result.Ok( None )

@compiler.target( os = not 'windows' )
def _connect_raw( sock: SOCKET, addr_ptr: Ptr[None], addr_len: usize ) -> Result[None, OSError]:
	from crt import get_errno
	with compiler.wrap_arithmetic:
		n: u32 = u32( addr_len )
	if _c_connect( sock, addr_ptr, n ) < 0:
		return Result.Err( OSError( get_errno() ))
	return Result.Ok( None )


@compiler.target( os = 'windows' )
def _send_raw( sock: SOCKET, buf: ConstPtr[u8], count: usize ) -> Result[usize, OSError]:
	with compiler.saturate_arithmetic:
		n: i32 = i32( count )
	sent: i32 = _c_send( sock, buf, n, 0 )
	if sent < 0:
		return Result.Err( OSError( WSAGetLastError() ))
	with compiler.wrap_arithmetic:
		return Result.Ok( usize( sent ))

@compiler.target( os = not 'windows' )
def _send_raw( sock: SOCKET, buf: ConstPtr[u8], count: usize ) -> Result[usize, OSError]:
	from crt import get_errno
	sent: isize = _c_send( sock, buf, count, 0 )
	if sent < isize( 0 ):
		return Result.Err( OSError( get_errno() ))
	with compiler.wrap_arithmetic:
		return Result.Ok( usize( sent ))


@compiler.target( os = 'windows' )
def _recv_raw( sock: SOCKET, buf: Ptr[u8], count: usize ) -> Result[usize, OSError]:
	with compiler.saturate_arithmetic:
		n: i32 = i32( count )
	received: i32 = _c_recv( sock, buf, n, 0 )
	if received < 0:
		return Result.Err( OSError( WSAGetLastError() ))
	with compiler.wrap_arithmetic:
		return Result.Ok( usize( received ))

@compiler.target( os = not 'windows' )
def _recv_raw( sock: SOCKET, buf: Ptr[u8], count: usize ) -> Result[usize, OSError]:
	from crt import get_errno
	received: isize = _c_recv( sock, buf, count, 0 )
	if received < isize( 0 ):
		return Result.Err( OSError( get_errno() ))
	with compiler.wrap_arithmetic:
		return Result.Ok( usize( received ))


@compiler.target( os = 'windows' )
def _sendto_raw( sock: SOCKET, buf: ConstPtr[u8], count: usize, addr_ptr: Ptr[None], addr_len: usize ) -> Result[usize, OSError]:
	with compiler.saturate_arithmetic:
		n: i32 = i32( count )
	with compiler.wrap_arithmetic:
		alen: i32 = i32( addr_len )
	sent: i32 = _c_sendto( sock, buf, n, 0, addr_ptr, alen )
	if sent < 0:
		return Result.Err( OSError( WSAGetLastError() ))
	with compiler.wrap_arithmetic:
		return Result.Ok( usize( sent ))

@compiler.target( os = not 'windows' )
def _sendto_raw( sock: SOCKET, buf: ConstPtr[u8], count: usize, addr_ptr: Ptr[None], addr_len: usize ) -> Result[usize, OSError]:
	from crt import get_errno
	with compiler.wrap_arithmetic:
		alen: u32 = u32( addr_len )
	sent: isize = _c_sendto( sock, buf, count, 0, addr_ptr, alen )
	if sent < isize( 0 ):
		return Result.Err( OSError( get_errno() ))
	with compiler.wrap_arithmetic:
		return Result.Ok( usize( sent ))


@compiler.target( os = 'windows' )
def _recvfrom_raw( sock: SOCKET, buf: Ptr[u8], count: usize, addr_ptr: Ptr[None], addr_cap: usize ) -> Result[usize, OSError]:
	with compiler.saturate_arithmetic:
		n: i32 = i32( count )
	with compiler.wrap_arithmetic:
		cap: i32 = i32( addr_cap )
	received: i32 = _c_recvfrom( sock, buf, n, 0, addr_ptr, compiler.addrof( cap ))
	if received < 0:
		return Result.Err( OSError( WSAGetLastError() ))
	with compiler.wrap_arithmetic:
		return Result.Ok( usize( received ))

@compiler.target( os = not 'windows' )
def _recvfrom_raw( sock: SOCKET, buf: Ptr[u8], count: usize, addr_ptr: Ptr[None], addr_cap: usize ) -> Result[usize, OSError]:
	from crt import get_errno
	with compiler.wrap_arithmetic:
		cap: u32 = u32( addr_cap )
	received: isize = _c_recvfrom( sock, buf, count, 0, addr_ptr, compiler.addrof( cap ))
	if received < isize( 0 ):
		return Result.Err( OSError( get_errno() ))
	with compiler.wrap_arithmetic:
		return Result.Ok( usize( received ))


@compiler.target( os = 'windows' )
def _shutdown_raw( sock: SOCKET, how: i32 ) -> Result[None, OSError]:
	if _c_shutdown( sock, how ) != 0:
		return Result.Err( OSError( WSAGetLastError() ))
	return Result.Ok( None )

@compiler.target( os = not 'windows' )
def _shutdown_raw( sock: SOCKET, how: i32 ) -> Result[None, OSError]:
	from crt import get_errno
	if _c_shutdown( sock, how ) < 0:
		return Result.Err( OSError( get_errno() ))
	return Result.Ok( None )


@compiler.target( os = 'windows' )
def _getsockname_raw( sock: SOCKET, addr_ptr: Ptr[None], addr_cap: usize ) -> Result[None, OSError]:
	with compiler.wrap_arithmetic:
		cap: i32 = i32( addr_cap )
	if _c_getsockname( sock, addr_ptr, compiler.addrof( cap )) != 0:
		return Result.Err( OSError( WSAGetLastError() ))
	return Result.Ok( None )

@compiler.target( os = not 'windows' )
def _getsockname_raw( sock: SOCKET, addr_ptr: Ptr[None], addr_cap: usize ) -> Result[None, OSError]:
	from crt import get_errno
	with compiler.wrap_arithmetic:
		cap: u32 = u32( addr_cap )
	if _c_getsockname( sock, addr_ptr, compiler.addrof( cap )) < 0:
		return Result.Err( OSError( get_errno() ))
	return Result.Ok( None )


@compiler.target( os = 'windows' )
def _set_reuseaddr_raw( sock: SOCKET, enable: bool ) -> Result[None, OSError]:
	value: i32 = 1 if enable else 0
	if _c_setsockopt( sock, SOL_SOCKET, SO_REUSEADDR, compiler.cast( ConstPtr[u8], compiler.addrof( value )), i32( 4 )) != 0:
		return Result.Err( OSError( WSAGetLastError() ))
	return Result.Ok( None )

@compiler.target( os = not 'windows' )
def _set_reuseaddr_raw( sock: SOCKET, enable: bool ) -> Result[None, OSError]:
	from crt import get_errno
	value: i32 = 1 if enable else 0
	if _c_setsockopt( sock, SOL_SOCKET, SO_REUSEADDR, compiler.cast( ConstPtr[u8], compiler.addrof( value )), u32( 4 )) < 0:
		return Result.Err( OSError( get_errno() ))
	return Result.Ok( None )


# ---------------------------------------------------------------------------
# WSAStartup-once lifecycle — no "run at import" mechanism exists in this
# language (module-level code is declarative, not imperative init-on-first-
# use), so a lazy CAS guard on lib/atomic.py's Atomic[bool] does the job.
# The loser SPINS on the flag rather than racing ahead of an in-flight
# WSAStartup call, closing the narrow window a bare "proceed after losing
# the CAS" would leave open. WSACleanup() is deliberately never called
# (matches CPython's own behavior; no natural process-exit hook here).
# ---------------------------------------------------------------------------

if compiler.target.os == 'windows':
	_wsa_started: Atomic[bool] = Atomic[bool]( False )
else:
	_wsa_started: Atomic[bool] = Atomic[bool]( False )  # unused on POSIX, kept unconditional for a single declaration site

# Two top-level bodies, not a function nested inside the if-block above -
# matches the proven shape every other OS-differentiated function in this
# file already uses (a function definition nested inside a module-level
# if/else is untested beyond simple extern/const declarations elsewhere in
# this codebase).
@compiler.target( os = 'windows' )
def _ensure_wsa_started() -> Result[None, OSError]:
	if _wsa_started.load():
		return Result.Ok( None )
	expected: bool = False
	if _wsa_started.compare_exchange( compiler.addrof( expected ), True ):
		wsa_buf: Ptr[u8] = sys.alloc[u8]( 512 )  # WSADATA is well under 512 bytes on any real Windows
		startup_rc: i32 = WSAStartup( 0x0202, compiler.cast( Ptr[None], wsa_buf ))  # MAKEWORD(2,2)
		sys.free( compiler.cast( Ptr[None], wsa_buf ))
		if startup_rc != 0:
			_wsa_started.store( False )  # allow a later retry
			# `+ 0` is deliberate, not decorative: OSError(startup_rc) alone
			# hits a real, pre-existing compiler bug (confirmed with a
			# minimal repro outside this file) where a bare variable-name
			# argument to an enum constructor is misdiagnosed as a type
			# mismatch ("expected builtins.OSError, got intrinsics.i32"),
			# while any non-bare-Name expression of the same value (a call,
			# or this arithmetic no-op) type-checks fine.
			with compiler.wrap_arithmetic:
				return Result.Err( OSError( startup_rc + 0 ))
		return Result.Ok( None )
	while not _wsa_started.load():
		pass  # loser spins until the winner's WSAStartup call completes
	return Result.Ok( None )

@compiler.target( os = not 'windows' )
def _ensure_wsa_started() -> Result[None, OSError]:
	return Result.Ok( None )


# ---------------------------------------------------------------------------
# Layer 3 — Socket, the public API. Mirrors lib/builtins/__File.py's
# BinaryReadWriter shape: private __sock field, __del__ auto-close
# (ignoring failure - "a destructor can't propagate close() failure"),
# idempotent .close(), a private _from_raw() factory.
# ---------------------------------------------------------------------------

class Socket:
	__sock: SOCKET
	__family: i32  # AF_INET or AF_INET6, remembered from creation so later
	               # calls know which sockaddr variant/inet_pton family a
	               # bare host string implies

	def __del__( self ) -> None:
		if self.__sock != INVALID_SOCKET:
			_close_raw( self.__sock ).is_ok()

	def close( self ) -> None:
		if self.__sock != INVALID_SOCKET:
			_close_raw( self.__sock ).is_ok()
			self.__sock = INVALID_SOCKET

	def bind( self, host: str, port: u16 ) -> Result[None, OSError]:
		if self.__family == AF_INET6:
			addr: SockAddrIn6 = _build_sockaddr_in6( host, port ).or_return()
			return _bind_raw( self.__sock, compiler.cast( Ptr[None], compiler.addrof( addr )), compiler.sizeof( SockAddrIn6 ))
		else:
			addr4: SockAddrIn = _build_sockaddr_in( host, port ).or_return()
			return _bind_raw( self.__sock, compiler.cast( Ptr[None], compiler.addrof( addr4 )), compiler.sizeof( SockAddrIn ))

	def listen( self, backlog: i32 = 128 ) -> Result[None, OSError]:
		return _listen_raw( self.__sock, backlog )

	def accept( self ) -> Result[tuple[Socket, SocketAddr], OSError]:
		if self.__family == AF_INET6:
			peer: SockAddrIn6 = SockAddrIn6()
			conn: SOCKET = _accept_raw( self.__sock, compiler.cast( Ptr[None], compiler.addrof( peer )), compiler.sizeof( SockAddrIn6 )).or_return()
			peer_addr: SocketAddr = _sockaddr_in6_to_addr( peer ).or_return()
		else:
			peer4: SockAddrIn = SockAddrIn()
			conn: SOCKET = _accept_raw( self.__sock, compiler.cast( Ptr[None], compiler.addrof( peer4 )), compiler.sizeof( SockAddrIn )).or_return()
			peer_addr: SocketAddr = _sockaddr_in_to_addr( peer4 ).or_return()
		return Result.Ok(( Socket._from_raw( conn, self.__family ), peer_addr ))

	def connect( self, host: str, port: u16 ) -> Result[None, OSError]:
		if self.__family == AF_INET6:
			addr: SockAddrIn6 = _build_sockaddr_in6( host, port ).or_return()
			return _connect_raw( self.__sock, compiler.cast( Ptr[None], compiler.addrof( addr )), compiler.sizeof( SockAddrIn6 ))
		else:
			addr4: SockAddrIn = _build_sockaddr_in( host, port ).or_return()
			return _connect_raw( self.__sock, compiler.cast( Ptr[None], compiler.addrof( addr4 )), compiler.sizeof( SockAddrIn ))

	def send( self, buf: ConstPtr[u8], count: usize ) -> Result[usize, OSError]:
		return _send_raw( self.__sock, buf, count )

	def recv( self, buf: Ptr[u8], count: usize ) -> Result[usize, OSError]:
		return _recv_raw( self.__sock, buf, count )

	def sendto( self, buf: ConstPtr[u8], count: usize, host: str, port: u16 ) -> Result[usize, OSError]:
		if self.__family == AF_INET6:
			addr: SockAddrIn6 = _build_sockaddr_in6( host, port ).or_return()
			return _sendto_raw( self.__sock, buf, count, compiler.cast( Ptr[None], compiler.addrof( addr )), compiler.sizeof( SockAddrIn6 ))
		else:
			addr4: SockAddrIn = _build_sockaddr_in( host, port ).or_return()
			return _sendto_raw( self.__sock, buf, count, compiler.cast( Ptr[None], compiler.addrof( addr4 )), compiler.sizeof( SockAddrIn ))

	def recvfrom( self, buf: Ptr[u8], count: usize ) -> Result[tuple[usize, SocketAddr], OSError]:
		if self.__family == AF_INET6:
			peer: SockAddrIn6 = SockAddrIn6()
			n: usize = _recvfrom_raw( self.__sock, buf, count, compiler.cast( Ptr[None], compiler.addrof( peer )), compiler.sizeof( SockAddrIn6 )).or_return()
			peer_addr: SocketAddr = _sockaddr_in6_to_addr( peer ).or_return()
		else:
			peer4: SockAddrIn = SockAddrIn()
			n: usize = _recvfrom_raw( self.__sock, buf, count, compiler.cast( Ptr[None], compiler.addrof( peer4 )), compiler.sizeof( SockAddrIn )).or_return()
			peer_addr: SocketAddr = _sockaddr_in_to_addr( peer4 ).or_return()
		return Result.Ok(( n, peer_addr ))

	def shutdown( self, how: i32 ) -> Result[None, OSError]:
		return _shutdown_raw( self.__sock, how )

	def set_reuseaddr( self, enable: bool ) -> Result[None, OSError]:
		return _set_reuseaddr_raw( self.__sock, enable )

	def getsockname( self ) -> Result[SocketAddr, OSError]:
		if self.__family == AF_INET6:
			addr: SockAddrIn6 = SockAddrIn6()
			_getsockname_raw( self.__sock, compiler.cast( Ptr[None], compiler.addrof( addr )), compiler.sizeof( SockAddrIn6 )).or_return()
			return _sockaddr_in6_to_addr( addr )
		else:
			addr4: SockAddrIn = SockAddrIn()
			_getsockname_raw( self.__sock, compiler.cast( Ptr[None], compiler.addrof( addr4 )), compiler.sizeof( SockAddrIn )).or_return()
			return _sockaddr_in_to_addr( addr4 )

	@private
	@staticmethod
	def _from_raw( sock: SOCKET, family: i32 ) -> Socket:
		return Socket.__allocate__( __sock = sock, __family = family )

	@staticmethod
	def create( family: i32, type: i32 ) -> Result[Socket, OSError]:
		_ensure_wsa_started().or_return()
		sock: SOCKET = _create_raw( family, type ).or_return()
		return Result.Ok( Socket._from_raw( sock, family ))

	@staticmethod
	def tcp( family: i32 = AF_INET ) -> Result[Socket, OSError]:
		return Socket.create( family, SOCK_STREAM )

	@staticmethod
	def udp( family: i32 = AF_INET ) -> Result[Socket, OSError]:
		return Socket.create( family, SOCK_DGRAM )
