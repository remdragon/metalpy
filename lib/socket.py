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
		getaddrinfo, freeaddrinfo,
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
	# getaddrinfo/freeaddrinfo - no header=, matching every other extern in
	# this POSIX branch (see windows/ws2_32.py's own copy of this comment for
	# why hints/res stay opaque Ptr[None]/Ptr[Ptr[None]] here too - the same
	# reasoning applies even though POSIX's netdb.h doesn't hit the specific
	# windows.h conflict that rules header= out on the Windows side).
	@extern( 'c', 'getaddrinfo' )
	def getaddrinfo( node: ConstPtr[u8], service: ConstPtr[u8], hints: Ptr[None], res: Ptr[Ptr[None]] ) -> i32:
		...
	@extern( 'c', 'freeaddrinfo' )
	def freeaddrinfo( res: Ptr[None] ) -> None:
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


# ---------------------------------------------------------------------------
# _AddrInfo — struct addrinfo, for hostname resolution (getaddrinfo). Field
# order genuinely differs by OS (confirmed against the real headers: Windows
# SDK's ws2def.h and glibc's netdb.h) - Windows' ADDRINFOA puts ai_canonname
# BEFORE ai_addr and sizes ai_addrlen as size_t; POSIX's addrinfo puts
# ai_addr BEFORE ai_canonname and sizes ai_addrlen as socklen_t (u32). Same
# "textbook ABI, one body per OS" treatment as SockAddrIn/SockAddrIn6 above,
# just field-order divergence instead of a width divergence.
#
# ai_addr/ai_canonname/ai_next stay opaque Ptr[None]/Ptr[u8], never a typed
# struct* field - matching this file's and ws2_32.py's own established
# posture for structured pointers we don't allocate ourselves. ai_addr only
# gets reinterpreted into a real SockAddrIn/SockAddrIn6 pointer at the point
# of use (_resolve_v4/_resolve_v6 below); ai_next only ever gets cast back
# to Ptr[_AddrInfo] to keep walking the linked list of results.
# ---------------------------------------------------------------------------

@compiler.target( os = 'windows' )
@cstruct
class _AddrInfo:
	ai_flags:     i32 = 0
	ai_family:    i32 = 0
	ai_socktype:  i32 = 0
	ai_protocol:  i32 = 0
	ai_addrlen:   usize = usize( 0 )
	ai_canonname: Ptr[u8] = None
	ai_addr:      Ptr[None] = None
	ai_next:      Ptr[None] = None

@compiler.target( os = not 'windows' )
@cstruct
class _AddrInfo:
	ai_flags:     i32 = 0
	ai_family:    i32 = 0
	ai_socktype:  i32 = 0
	ai_protocol:  i32 = 0
	ai_addrlen:   u32 = 0
	ai_addr:      Ptr[None] = None
	ai_canonname: Ptr[u8] = None
	ai_next:      Ptr[None] = None


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

	@property
	def host( self ) -> str:
		return self.__host

	@property
	def port( self ) -> u16:
		return self.__port

	@private
	@staticmethod
	def _from_parts( host: str, port: u16 ) -> SocketAddr:
		return SocketAddr.__allocate__( __host = host, __port = port )


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
		return Result.Err( OSError.Invalid )
	with compiler.wrap_arithmetic:
		family: u16 = u16( AF_INET )
	return Result.Ok( SockAddrIn( sin_family = family, sin_port = _htons( port ), sin_addr = ip_addr ))


def _build_sockaddr_in6( host: str, port: u16 ) -> Result[SockAddrIn6, OSError]:
	buf: bytearray = bytearray( 16 )
	rc: i32 = inet_pton( AF_INET6, host.get_cstr(), compiler.cast( Ptr[None], buf.get_ptr() ))
	if rc != 1:
		return Result.Err( OSError.Invalid )
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


# ---------------------------------------------------------------------------
# Hostname resolution — getaddrinfo() walk, one function per address family
# (mirrors every other family-dispatched pair in this file, e.g. Socket.bind/
# connect's own `if self.__family == AF_INET6: ... else: ...` split).
# hints.ai_family is always set to the family being resolved, so getaddrinfo
# itself filters out any non-matching results (POSIX/Winsock guarantee) - no
# family check is needed while walking. Numeric IP literals ("127.0.0.1",
# "::1") keep working here for free: getaddrinfo recognizes them without any
# extra flag and returns a single result with no real DNS round trip.
#
# service is always NULL; the port is folded into each result's own sockaddr
# by re-constructing it (never mutating a field in place - same rule
# _build_sockaddr_in/6 above follow), rather than passing a numeric-service
# string, which would need its own int->str dependency just for this.
#
# A bogus/unresolvable host returns OSError.NameResolutionFailed, distinct
# from the OSError.Invalid _build_sockaddr_in/6 return for an unparseable IP
# literal - these are genuinely different failures (a syntactically-bad
# address vs. a well-formed hostname that just doesn't resolve) and collapsing
# them into one code would leave callers unable to tell "fix your input" from
# "the network/DNS didn't cooperate". NameResolutionFailed is a real,
# specific OSError member rather than the raw getaddrinfo() return code
# itself: on POSIX that code is EAI_*, a wholly separate namespace from errno
# (NOT safe to feed into OSError's own errno-based construction - confirmed
# against the real getaddrinfo(3) contract), and on Windows it IS a real
# WSAHOST_NOT_FOUND-compatible code but only the single most common failure
# gets a name here, same as every other OSError member.
# ---------------------------------------------------------------------------

def _resolve_v4( host: str, port: u16, socktype: i32 ) -> Result[list[SockAddrIn], OSError]:
	hints: _AddrInfo = _AddrInfo( ai_family = AF_INET, ai_socktype = socktype )
	res_head: Ptr[None] = None
	rc: i32 = getaddrinfo( host.get_cstr(), None, compiler.cast( Ptr[None], compiler.addrof( hints )), compiler.addrof( res_head ))
	if rc != 0:
		return Result.Err( OSError.NameResolutionFailed )
	results: list[SockAddrIn] = list[SockAddrIn]()
	cur: Ptr[None] = res_head
	while cur is not None:
		node: Ptr[_AddrInfo] = compiler.cast( Ptr[_AddrInfo], cur )
		info: _AddrInfo = node[0]
		addr_ptr: Ptr[SockAddrIn] = compiler.cast( Ptr[SockAddrIn], info.ai_addr )
		found: SockAddrIn = addr_ptr[0]
		results.append( SockAddrIn( sin_family = found.sin_family, sin_port = _htons( port ), sin_addr = found.sin_addr )).unwrap( 'resolve v4: append' )
		cur = info.ai_next
	freeaddrinfo( res_head )
	return Result.Ok( results )


def _resolve_v6( host: str, port: u16, socktype: i32 ) -> Result[list[SockAddrIn6], OSError]:
	hints: _AddrInfo = _AddrInfo( ai_family = AF_INET6, ai_socktype = socktype )
	res_head: Ptr[None] = None
	rc: i32 = getaddrinfo( host.get_cstr(), None, compiler.cast( Ptr[None], compiler.addrof( hints )), compiler.addrof( res_head ))
	if rc != 0:
		return Result.Err( OSError.NameResolutionFailed )
	results: list[SockAddrIn6] = list[SockAddrIn6]()
	cur: Ptr[None] = res_head
	while cur is not None:
		node: Ptr[_AddrInfo] = compiler.cast( Ptr[_AddrInfo], cur )
		info: _AddrInfo = node[0]
		addr_ptr: Ptr[SockAddrIn6] = compiler.cast( Ptr[SockAddrIn6], info.ai_addr )
		found: SockAddrIn6 = addr_ptr[0]
		results.append( SockAddrIn6(
			sin6_family = found.sin6_family, sin6_port = _htons( port ), sin6_flowinfo = found.sin6_flowinfo,
			sin6_addr_0 = found.sin6_addr_0, sin6_addr_1 = found.sin6_addr_1, sin6_addr_2 = found.sin6_addr_2, sin6_addr_3 = found.sin6_addr_3,
			sin6_addr_4 = found.sin6_addr_4, sin6_addr_5 = found.sin6_addr_5, sin6_addr_6 = found.sin6_addr_6, sin6_addr_7 = found.sin6_addr_7,
			sin6_addr_8 = found.sin6_addr_8, sin6_addr_9 = found.sin6_addr_9, sin6_addr_10 = found.sin6_addr_10, sin6_addr_11 = found.sin6_addr_11,
			sin6_addr_12 = found.sin6_addr_12, sin6_addr_13 = found.sin6_addr_13, sin6_addr_14 = found.sin6_addr_14, sin6_addr_15 = found.sin6_addr_15,
			sin6_scope_id = found.sin6_scope_id,
		)).unwrap( 'resolve v6: append' )
		cur = info.ai_next
	freeaddrinfo( res_head )
	return Result.Ok( results )


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
		return Result.Err( OSError.Invalid )
	from crt import strnlen
	slen: usize = strnlen( strbuf.get_const_ptr(), usize( 16 ))
	with compiler.wrap_arithmetic:
		full_len: usize = slen + usize( 1 )
	match str.from_cstr( strbuf.get_const_ptr(), full_len ):
		case Result.Ok( host ):
			return Result.Ok( SocketAddr._from_parts( host, _htons( sa.sin_port )))
		case Result.Err( _ ):
			return Result.Err( OSError.Invalid )

@compiler.target( os = not 'windows' )
def _sockaddr_in_to_addr( sa: SockAddrIn ) -> Result[SocketAddr, OSError]:
	addr_val: u32 = sa.sin_addr
	strbuf: bytearray = bytearray( 16 )  # "255.255.255.255\0" fits in 16
	res = inet_ntop( AF_INET, compiler.cast( Ptr[None], compiler.addrof( addr_val )), strbuf.get_ptr(), u32( 16 ))
	if res is None:
		return Result.Err( OSError.Invalid )
	from crt import strnlen
	slen: usize = strnlen( strbuf.get_const_ptr(), usize( 16 ))
	with compiler.wrap_arithmetic:
		full_len: usize = slen + usize( 1 )
	match str.from_cstr( strbuf.get_const_ptr(), full_len ):
		case Result.Ok( host ):
			return Result.Ok( SocketAddr._from_parts( host, _htons( sa.sin_port )))
		case Result.Err( _ ):
			return Result.Err( OSError.Invalid )


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
		return Result.Err( OSError.Invalid )
	from crt import strnlen
	slen: usize = strnlen( strbuf.get_const_ptr(), usize( 46 ))
	with compiler.wrap_arithmetic:
		full_len: usize = slen + usize( 1 )
	match str.from_cstr( strbuf.get_const_ptr(), full_len ):
		case Result.Ok( host ):
			return Result.Ok( SocketAddr._from_parts( host, _htons( sa.sin6_port )))
		case Result.Err( _ ):
			return Result.Err( OSError.Invalid )

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
		return Result.Err( OSError.Invalid )
	from crt import strnlen
	slen: usize = strnlen( strbuf.get_const_ptr(), usize( 46 ))
	with compiler.wrap_arithmetic:
		full_len: usize = slen + usize( 1 )
	match str.from_cstr( strbuf.get_const_ptr(), full_len ):
		case Result.Ok( host ):
			return Result.Ok( SocketAddr._from_parts( host, _htons( sa.sin6_port )))
		case Result.Err( _ ):
			return Result.Err( OSError.Invalid )


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
# use), so a lazy CAS guard does the job. A tri-state (really 4-state)
# Atomic[i32] state machine, NOT a bare Atomic[bool]: a bool CAS can't let a
# spinning loser distinguish "nobody has started yet" from "the winner just
# finished (with either outcome)" - both collapse to the same False value.
# The original bool version's failure path reset the flag back to False "to
# allow a later retry" - but that's exactly the same value a loser is
# spinning to see turn True, so a loser spinning at the moment the winner's
# WSAStartup call failed would wait for True forever (or until some
# unrelated FUTURE caller happened to retry and succeed) instead of ever
# observing the failure. Every transition here is monotonic (NOT_STARTED ->
# IN_PROGRESS -> {DONE_OK, DONE_ERR}, never backwards), so a spinning loser
# is guaranteed to see a terminal state - DONE_ERR is now STICKY (not reset
# back to NOT_STARTED), matching CPython's own socket module, which also
# never retries WSAStartup after a failure. WSACleanup() is deliberately
# never called (matches CPython's own behavior; no natural process-exit
# hook here).
# ---------------------------------------------------------------------------

_WSA_NOT_STARTED: i32 = 0
_WSA_IN_PROGRESS: i32 = 1
_WSA_DONE_OK:     i32 = 2
_WSA_DONE_ERR:    i32 = 3

if compiler.target.os == 'windows':
	_wsa_state: Atomic[i32] = Atomic[i32]( _WSA_NOT_STARTED )
	_wsa_error: Atomic[i32] = Atomic[i32]( 0 )  # valid only once _wsa_state == _WSA_DONE_ERR
else:
	_wsa_state: Atomic[i32] = Atomic[i32]( _WSA_NOT_STARTED )  # unused on POSIX, kept unconditional for a single declaration site
	_wsa_error: Atomic[i32] = Atomic[i32]( 0 )

# Two top-level bodies, not a function nested inside the if-block above -
# matches the proven shape every other OS-differentiated function in this
# file already uses (a function definition nested inside a module-level
# if/else is untested beyond simple extern/const declarations elsewhere in
# this codebase).
@compiler.target( os = 'windows' )
def _ensure_wsa_started() -> Result[None, OSError]:
	while True:
		state: i32 = _wsa_state.load()
		if state == _WSA_DONE_OK:
			return Result.Ok( None )
		if state == _WSA_DONE_ERR:
			# `+ 0` is deliberate, not decorative: OSError(...) alone hits a
			# real, pre-existing compiler bug (confirmed with a minimal
			# repro outside this file) where a bare variable-name argument
			# to an enum constructor is misdiagnosed as a type mismatch
			# ("expected builtins.OSError, got intrinsics.i32"), while any
			# non-bare-Name expression of the same value (a call, or this
			# arithmetic no-op) type-checks fine.
			with compiler.wrap_arithmetic:
				return Result.Err( OSError( _wsa_error.load() + 0 ))
		if state == _WSA_NOT_STARTED:
			expected: i32 = _WSA_NOT_STARTED
			if _wsa_state.compare_exchange( compiler.addrof( expected ), _WSA_IN_PROGRESS ):
				# won the race - the only caller that will ever call
				# WSAStartup for this process
				wsa_buf: Ptr[u8] = sys.alloc[u8]( 512 )  # WSADATA is well under 512 bytes on any real Windows
				startup_rc: i32 = WSAStartup( 0x0202, compiler.cast( Ptr[None], wsa_buf ))  # MAKEWORD(2,2)
				sys.free( compiler.cast( Ptr[None], wsa_buf ))
				if startup_rc != 0:
					_wsa_error.store( startup_rc )
					_wsa_state.store( _WSA_DONE_ERR )
				else:
					_wsa_state.store( _WSA_DONE_OK )
				continue  # loop back around - the DONE_OK/DONE_ERR branch above now returns
			# lost the CAS: someone else is already IN_PROGRESS (or finished
			# between our load and our CAS attempt) - fall through and spin
		# state == _WSA_IN_PROGRESS (either genuinely, or because we just
		# lost the CAS above) - another thread is running WSAStartup right
		# now; spin until it reaches a terminal state
		pass

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
		conn: SOCKET
		peer_addr: SocketAddr
		if self.__family == AF_INET6:
			peer: SockAddrIn6 = SockAddrIn6()
			conn = _accept_raw( self.__sock, compiler.cast( Ptr[None], compiler.addrof( peer )), compiler.sizeof( SockAddrIn6 )).or_return()
			peer_addr = _sockaddr_in6_to_addr( peer ).or_return()
		else:
			peer4: SockAddrIn = SockAddrIn()
			conn = _accept_raw( self.__sock, compiler.cast( Ptr[None], compiler.addrof( peer4 )), compiler.sizeof( SockAddrIn )).or_return()
			peer_addr = _sockaddr_in_to_addr( peer4 ).or_return()
		return Result.Ok(( Socket._from_raw( conn, self.__family ), peer_addr ))

	# Resolves host (a hostname OR a numeric IP literal - getaddrinfo handles
	# both) via _resolve_v4/_resolve_v6, then tries each candidate address in
	# order (happy-eyeballs-lite: a hostname with several A/AAAA records of
	# this socket's own family isn't uncommon) until one connects, returning
	# the last candidate's error if every one of them fails. bind()/sendto()
	# deliberately stay literal-IP-only (unchanged) - PLAN_HTTP_CLIENT.md's
	# own socket contract only calls for connect() to resolve hostnames.
	def connect( self, host: str, port: u16 ) -> Result[None, OSError]:
		if self.__family == AF_INET6:
			candidates: list[SockAddrIn6] = _resolve_v6( host, port, SOCK_STREAM ).or_return()
			last_err: OSError = OSError.Invalid
			for i in range( len( candidates )):
				addr: SockAddrIn6 = candidates.__getitem__( i ).unwrap( 'connect: candidate index' )
				match _connect_raw( self.__sock, compiler.cast( Ptr[None], compiler.addrof( addr )), compiler.sizeof( SockAddrIn6 )):
					case Result.Ok( _ ):
						return Result.Ok( None )
					case Result.Err( e ):
						last_err = e
			return Result.Err( last_err )
		else:
			candidates4: list[SockAddrIn] = _resolve_v4( host, port, SOCK_STREAM ).or_return()
			last_err4: OSError = OSError.Invalid
			for i in range( len( candidates4 )):
				addr4: SockAddrIn = candidates4.__getitem__( i ).unwrap( 'connect: candidate index' )
				match _connect_raw( self.__sock, compiler.cast( Ptr[None], compiler.addrof( addr4 )), compiler.sizeof( SockAddrIn )):
					case Result.Ok( _ ):
						return Result.Ok( None )
					case Result.Err( e ):
						last_err4 = e
			return Result.Err( last_err4 )

	def send( self, buf: ConstPtr[u8], count: usize ) -> Result[usize, OSError]:
		return _send_raw( self.__sock, buf, count )

	def send_all( self, buf: ConstPtr[u8], count: usize ) -> Result[None, OSError]:
		''' loops send() until every byte in buf[0:count) is sent, or an
		error occurs - send() itself can do short writes, so a caller that
		actually needs "all N bytes went out" has to loop (mirrors lib/
		http/client.py's own hand-rolled _send_all, promoted here so future
		Socket consumers - e.g. a hand-rolled HTTP server - don't need
		their own copy). A 0-byte send mid-loop (the peer stopped
		accepting data) is reported as OSError.BrokenPipe, the existing
		OSError member that already names this condition. '''
		sent: usize = 0
		with compiler.panic_arithmetic( 'bounded by count, cannot overflow' ):
			while sent < count:
				n: usize = self.send( buf + sent, count - sent ).or_return()
				if n == 0:
					return Result.Err( OSError.BrokenPipe )
				sent += n
		return Result.Ok( None )

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
		n: usize
		peer_addr: SocketAddr
		if self.__family == AF_INET6:
			peer: SockAddrIn6 = SockAddrIn6()
			n = _recvfrom_raw( self.__sock, buf, count, compiler.cast( Ptr[None], compiler.addrof( peer )), compiler.sizeof( SockAddrIn6 )).or_return()
			peer_addr = _sockaddr_in6_to_addr( peer ).or_return()
		else:
			peer4: SockAddrIn = SockAddrIn()
			n = _recvfrom_raw( self.__sock, buf, count, compiler.cast( Ptr[None], compiler.addrof( peer4 )), compiler.sizeof( SockAddrIn )).or_return()
			peer_addr = _sockaddr_in_to_addr( peer4 ).or_return()
		return Result.Ok(( n, peer_addr ))

	def shutdown( self, how: i32 ) -> Result[None, OSError]:
		return _shutdown_raw( self.__sock, how )

	def set_reuseaddr( self, enable: bool ) -> Result[None, OSError]:
		return _set_reuseaddr_raw( self.__sock, enable )

	def fileno( self ) -> SOCKET:
		''' the raw OS socket handle - POSIX fd (i32) or Windows SOCKET
		(usize), matching this module's own SOCKET type alias per platform.
		Needed by TLS backends (lib/ssl.py's Linux/OpenSSL backend) that hand
		this off directly to a native library rather than driving I/O
		through send()/recv() themselves. '''
		return self.__sock

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


def make_loopback_pair() -> tuple[Socket, Socket]:
	''' a connected TCP loopback (127.0.0.1) pair - returns (accepted_side,
	connected_side). A portable, always-available inter-thread wake-up
	primitive (the classic reactor "self-pipe" trick), used directly (as a
	genuinely blocking recv()) or registered non-blocking with a Poller,
	depending on the caller - see lib/reactor.py's Worker (non-blocking,
	polled) and lib/asyncfile.py's thread pool (blocking, no Poller
	involved at all). NOT a real pipe(2): WSAPoll can only poll actual
	SOCKETs on Windows, and a connected TCP pair works identically on both
	platforms either way, so there's no reason for two implementations. '''
	listener: Socket = Socket.tcp().unwrap( 'make_loopback_pair: listener create failed' )
	listener.bind( '127.0.0.1', u16( 0 )).unwrap( 'make_loopback_pair: bind failed' )
	listener.listen().unwrap( 'make_loopback_pair: listen failed' )
	bound: SocketAddr = listener.getsockname().unwrap( 'make_loopback_pair: getsockname failed' )
	side_b: Socket = Socket.tcp().unwrap( 'make_loopback_pair: connect-side create failed' )
	side_b.connect( '127.0.0.1', bound.port ).unwrap( 'make_loopback_pair: connect failed' )
	( side_a, _addr ) = listener.accept().unwrap( 'make_loopback_pair: accept failed' )
	return ( side_a, side_b )


# ---------------------------------------------------------------------------
# RecvBuffer — accumulates bytes read off a Socket across multiple recv()
# calls. A single recv() may return less than requested, and the total
# message length usually isn't known up front - the exact problem lib/http/
# client.py's own hand-rolled _GrowableBuffer was built to solve (see that
# file's own comment). Promoted here (not HTTP-specific) so a future Socket
# consumer (a raw TCP protocol, a simple line-based server, a hand-rolled
# HTTP server's own request-line parsing, ...) doesn't need to reinvent it.
# Deliberately narrow - accumulate + expose the raw accumulated bytes only,
# NOT a general buffered-reader-with-readline() abstraction (out of scope
# here; HTTP's own header/chunk-framing logic stays in lib/http/client.py,
# which could build on top of this type instead of duplicating the growth/
# fill machinery as a future refactor - not done as part of this change).
# ---------------------------------------------------------------------------

class RecvBuffer:
	__data: Ptr[u8]
	__len: usize
	__cap: usize

	def __init__( self, initial_cap: usize = 4096 ) -> None:
		self.__cap = initial_cap
		self.__data = sys.alloc[u8]( self.__cap )
		self.__len = 0

	def __del__( self ) -> None:
		sys.free( self.__data )

	def len( self ) -> usize:
		return self.__len

	def get_const_ptr( self ) -> ConstPtr[u8]:
		return self.__data

	def _grow( self, min_additional: usize ) -> None:
		with compiler.panic_arithmetic( 'irrational buffer growth' ):
			needed: usize = self.__len + min_additional
		if needed <= self.__cap:
			return
		new_cap: usize = self.__cap
		with compiler.panic_arithmetic( 'irrational buffer growth' ):
			while new_cap < needed:
				new_cap = new_cap * 2
		new_data: Ptr[u8] = sys.alloc[u8]( new_cap )
		sys.memcpy( new_data, self.__data, self.__len )
		sys.free( self.__data )
		self.__data = new_data
		self.__cap = new_cap

	def fill_from( self, sock: Socket, chunk_size: usize = 4096 ) -> Result[usize, OSError]:
		''' one recv() call, appended to the buffer. Returns the number of
		bytes read - 0 means the peer closed the connection (EOF), matching
		Socket.recv()'s own convention; not itself an error. '''
		self._grow( chunk_size )
		with compiler.wrap_arithmetic:
			dest: Ptr[u8] = self.__data + self.__len
			room: usize = self.__cap - self.__len
		n: usize = sock.recv( dest, room ).or_return()
		with compiler.wrap_arithmetic:
			self.__len += n
		return Result.Ok( n )


# ---------------------------------------------------------------------------
# resolve() — a standalone hostname->IP-literal-strings lookup, built on the
# same _resolve_v4/_resolve_v6 machinery Socket.connect() uses internally.
# Not part of the PLAN_HTTP_CLIENT.md socket contract (which only needs
# Socket.connect() to resolve transparently) but a natural, cheap-to-expose
# building block on top of it - useful on its own and gives the resolver a
# directly testable surface independent of a live TCP connect().
#
# port is irrelevant to a pure address lookup, so _resolve_v4/_v6 are called
# with a dummy 0 and the port is dropped again (via SocketAddr.host) rather
# than exposing SocketAddr's own host+port pairing here, which would wrongly
# imply the port means something.
# ---------------------------------------------------------------------------

def resolve( host: str, family: i32 = AF_INET ) -> Result[list[str], OSError]:
	_ensure_wsa_started().or_return()
	out: list[str] = list[str]()
	if family == AF_INET6:
		candidates: list[SockAddrIn6] = _resolve_v6( host, u16( 0 ), SOCK_STREAM ).or_return()
		for i in range( len( candidates )):
			addr: SocketAddr = _sockaddr_in6_to_addr( candidates.__getitem__( i ).unwrap( 'resolve: candidate index' )).or_return()
			out.append( addr.host ).unwrap( 'resolve: append' )
	else:
		candidates4: list[SockAddrIn] = _resolve_v4( host, u16( 0 ), SOCK_STREAM ).or_return()
		for i in range( len( candidates4 )):
			addr4: SocketAddr = _sockaddr_in_to_addr( candidates4.__getitem__( i ).unwrap( 'resolve: candidate index' )).or_return()
			out.append( addr4.host ).unwrap( 'resolve: append' )
	return Result.Ok( out )
