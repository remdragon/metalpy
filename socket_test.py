# Real-compile-and-run behavioral tests for lib/socket.py (blocking TCP/UDP
# over IPv4/IPv6). A real OS socket round trip is the only way to confirm
# this actually works end to end - see the design notes at the top of
# lib/socket.py for the FFI/struct-layout details this exercises.
#
# Each case gets its OWN compile+link+run (test_support.RealCompileMixin's
# _assert_compiles_and_runs, not the merged assert_programs_run):
# lib/socket.py's Socket.create() funnels through a lazy WSAStartup-once
# guard (an Atomic[bool] CAS, see _ensure_wsa_started), and test_support.py's
# own assert_programs_run docstring is explicit that merging programs with
# process-global one-time init is unsafe - so this file intentionally does
# NOT use it, unlike e.g. time_test.py.
#
# Which OS backend runs follows the build host (there is no cross-compile):
# Winsock (ws2_32.dll) on Windows, plain BSD sockets (libc) on POSIX. The
# POSIX branch's compiler.cexpr-derived constants (AF_INET, SO_REUSEADDR,
# etc.) are exercised for real when this suite runs under Linux (confirmed
# via WSL Debian during development - see the socket-library plan/commit
# history) - only macOS/BSD's SOL_SOCKET/SO_REUSEADDR divergence (the actual
# reason cexpr is used over hardcoding) stays reviewed-but-unverified here,
# since no macOS host was available.
#
# Every program returns 0 on success, or a distinct nonzero i32 exit code
# per failed check - print()/stdout is deliberately avoided (see time_test.
# py's own reasoning: exit codes keep this decoupled from a separate,
# unrelated stdout regression).

import unittest

import emitter_c
import test_support
from test_support import RealCompileMixin


# Binds both sides to an OS-assigned ephemeral port (port 0) and reads the
# real port back via getsockname() - a hardcoded port would be flaky in CI
# (may already be in use) and this also exercises getsockname()/SocketAddr
# decoding as a side effect, which is good coverage on its own.
_TCP_LOOPBACK_ECHO = '''
import socket

def main() -> i32:
	server: socket.Socket = socket.Socket.tcp().unwrap( 'server create' )
	server.bind( '127.0.0.1', u16( 0 )).unwrap( 'server bind' )
	server.listen().unwrap( 'server listen' )
	bound: socket.SocketAddr = server.getsockname().unwrap( 'server getsockname' )
	if bound.host != '127.0.0.1':
		return 1

	client: socket.Socket = socket.Socket.tcp().unwrap( 'client create' )
	client.connect( '127.0.0.1', bound.port ).unwrap( 'client connect' )

	match server.accept():
		case Result.Ok( pair ):
			conn: socket.Socket = pair[0]
		case Result.Err( _ ):
			return 2

	msg: bytes = b'ping!'
	sent: usize = client.send( msg.get_const_ptr(), usize( 5 )).unwrap( 'client send' )
	if sent != usize( 5 ):
		return 3
	rbuf: bytearray = bytearray( 16 )
	received: usize = conn.recv( rbuf.get_ptr(), usize( 16 )).unwrap( 'conn recv' )
	if received != usize( 5 ):
		return 4

	reply: bytes = b'pong'
	sent2: usize = conn.send( reply.get_const_ptr(), usize( 4 )).unwrap( 'conn send' )
	if sent2 != usize( 4 ):
		return 5
	rbuf2: bytearray = bytearray( 16 )
	received2: usize = client.recv( rbuf2.get_ptr(), usize( 16 )).unwrap( 'client recv' )
	if received2 != usize( 4 ):
		return 6

	conn.close()
	client.close()
	server.close()
	return 0
'''

# Same shape as _TCP_LOOPBACK_ECHO above, but exercises tuple destructuring
# directly against accept()'s Result[tuple[Socket, SocketAddr], OSError] -
# the actual (conn, addr) = server.accept().or_return() shape a hand-rolled
# server would write, instead of the match/case Result.Ok(pair): conn =
# pair[0] workaround _TCP_LOOPBACK_ECHO above still uses. or_return()
# requires the enclosing function to itself return a compatible Result -
# main() itself can't (its return type is constrained to None/a scalar
# int - see SYNTAX.md), so the destructuring lives in a helper, mirroring
# every other real ".or_return() inside a helper, matched in main()" test
# elsewhere in this suite.
_TCP_LOOPBACK_ECHO_TUPLE_DESTRUCTURE = '''
import socket

def run() -> Result[i32, OSError]:
	server: socket.Socket = socket.Socket.tcp().or_return()
	server.bind( '127.0.0.1', u16( 0 )).or_return()
	server.listen().or_return()
	bound: socket.SocketAddr = server.getsockname().or_return()

	client: socket.Socket = socket.Socket.tcp().or_return()
	client.connect( '127.0.0.1', bound.port ).or_return()

	( conn, addr ) = server.accept().or_return()
	if addr.host != '127.0.0.1':
		return Result.Ok( 1 )

	msg: bytes = b'ping!'
	sent: usize = client.send( msg.get_const_ptr(), usize( 5 )).or_return()
	if sent != usize( 5 ):
		return Result.Ok( 2 )
	rbuf: bytearray = bytearray( 16 )
	received: usize = conn.recv( rbuf.get_ptr(), usize( 16 )).or_return()
	if received != usize( 5 ):
		return Result.Ok( 3 )

	conn.close()
	client.close()
	server.close()
	return Result.Ok( 0 )

def main() -> i32:
	match run():
		case Result.Ok( code ):
			return code
		case Result.Err( _ ):
			return 4
'''

# Socket.send_all() - loops until every byte is sent, unlike a single send()
# which can do a short write. Sends a payload well past what one send() call
# is likely to accept in one go, and the peer loops recv() to reassemble it
# (a real short write isn't reliably forceable over loopback, so this proves
# send_all() delivers the FULL payload rather than proving it looped).
_SEND_ALL_DELIVERS_EVERYTHING = '''
import socket

def run() -> Result[i32, OSError]:
	server: socket.Socket = socket.Socket.tcp().or_return()
	server.bind( '127.0.0.1', u16( 0 )).or_return()
	server.listen().or_return()
	bound: socket.SocketAddr = server.getsockname().or_return()

	client: socket.Socket = socket.Socket.tcp().or_return()
	client.connect( '127.0.0.1', bound.port ).or_return()
	( conn, _addr ) = server.accept().or_return()

	payload: bytearray = bytearray( 200000 )
	i: usize = 0
	with compiler.panic_arithmetic( 'divisor 256 is a nonzero literal, never zero-divides; loop is bounded' ):
		while i < 200000:
			payload[i] = u8( i % 256 )
			i += 1

	client.send_all( payload.get_const_ptr(), usize( 200000 )).or_return()

	rbuf: bytearray = bytearray( 200000 )
	total: usize = 0
	with compiler.wrap_arithmetic:
		while total < 200000:
			dest: Ptr[u8] = rbuf.get_ptr() + total
			room: usize = 200000 - total
			n: usize = conn.recv( dest, room ).or_return()
			if n == 0:
				return Result.Ok( 1 ) # peer closed early - send_all didn't deliver everything
			total += n

	i = 0
	with compiler.panic_arithmetic( 'divisor 256 is a nonzero literal, never zero-divides; loop is bounded' ):
		while i < 200000:
			got: u8 = rbuf.__getitem__( i ).unwrap( 'in bounds by construction' )
			if got != u8( i % 256 ):
				return Result.Ok( 2 )
			i += 1

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

# send_all() on an already-closed socket must return Result.Err, not hang
# or silently "succeed" having sent nothing.
_SEND_ALL_ERROR_PROPAGATES = '''
import socket

def main() -> i32:
	sock: socket.Socket = socket.Socket.tcp().unwrap( 'create' )
	sock.close()
	msg: bytes = b'hello'
	match sock.send_all( msg.get_const_ptr(), usize( 5 )):
		case Result.Ok( _ ):
			return 1 # should have failed - socket is closed
		case Result.Err( _ ):
			return 0
'''

# RecvBuffer.fill_from() accumulates across multiple calls, including a real
# capacity-doubling path (the payload is well past RecvBuffer's default
# initial 4096-byte capacity).
_RECVBUFFER_ACCUMULATES_ACROSS_CALLS = '''
import socket

def run() -> Result[i32, OSError]:
	server: socket.Socket = socket.Socket.tcp().or_return()
	server.bind( '127.0.0.1', u16( 0 )).or_return()
	server.listen().or_return()
	bound: socket.SocketAddr = server.getsockname().or_return()

	client: socket.Socket = socket.Socket.tcp().or_return()
	client.connect( '127.0.0.1', bound.port ).or_return()
	( conn, _addr ) = server.accept().or_return()

	payload: bytearray = bytearray( 10000 )
	i: usize = 0
	with compiler.panic_arithmetic( 'divisor 251 is a nonzero literal, never zero-divides; loop is bounded' ):
		while i < 10000:
			payload[i] = u8( i % 251 )
			i += 1
	client.send_all( payload.get_const_ptr(), usize( 10000 )).or_return()

	buf: socket.RecvBuffer = socket.RecvBuffer()
	with compiler.wrap_arithmetic:
		while buf.len() < 10000:
			n: usize = buf.fill_from( conn ).or_return()
			if n == 0:
				return Result.Ok( 1 ) # peer closed before sending everything

	if buf.len() != usize( 10000 ):
		return Result.Ok( 2 )
	ptr: ConstPtr[u8] = buf.get_const_ptr()
	i = 0
	with compiler.panic_arithmetic( 'divisor 251 is a nonzero literal, never zero-divides; loop is bounded' ):
		while i < 10000:
			if ptr[i] != u8( i % 251 ):
				return Result.Ok( 3 )
			i += 1

	conn.close()
	client.close()
	server.close()
	return Result.Ok( 0 )

def main() -> i32:
	match run():
		case Result.Ok( code ):
			return code
		case Result.Err( _ ):
			return 4
'''

# fill_from() returns Ok(0) on peer EOF, not an error - matches Socket.recv()'s
# own convention (stated explicitly in RecvBuffer.fill_from()'s own docstring).
_RECVBUFFER_FILL_FROM_RETURNS_ZERO_ON_PEER_CLOSE = '''
import socket

def run() -> Result[i32, OSError]:
	server: socket.Socket = socket.Socket.tcp().or_return()
	server.bind( '127.0.0.1', u16( 0 )).or_return()
	server.listen().or_return()
	bound: socket.SocketAddr = server.getsockname().or_return()

	client: socket.Socket = socket.Socket.tcp().or_return()
	client.connect( '127.0.0.1', bound.port ).or_return()
	( conn, _addr ) = server.accept().or_return()
	client.close() # peer closes without ever sending anything

	buf: socket.RecvBuffer = socket.RecvBuffer()
	n: usize = buf.fill_from( conn ).or_return()
	if n != usize( 0 ):
		return Result.Ok( 1 )

	conn.close()
	server.close()
	return Result.Ok( 0 )

def main() -> i32:
	match run():
		case Result.Ok( code ):
			return code
		case Result.Err( _ ):
			return 2
'''

# The real regression test for the WSAStartup race fix: 16 threads all racing
# through _ensure_wsa_started() concurrently via their own first-ever
# Socket.tcp() call in this isolated program (matching this file's own
# "isolated program per test" rule, since it's specifically about first-use
# init) - real CAS contention, real losers spinning while a winner runs
# WSAStartup, all 16 required to independently succeed. This can't exercise
# the DONE_ERR (winner-fails) branch specifically - WSAStartup essentially
# never fails in a real test environment and there's no fault-injection seam
# (it's a real @extern into ws2_32.dll) - that branch's correctness rests on
# the state-machine argument in lib/socket.py's own comment, not an executed
# assertion here (matches this file's own "reviewed-but-unverified" convention
# elsewhere, e.g. macOS SOL_SOCKET). Harmless/trivial on POSIX (where
# _ensure_wsa_started is a no-op), but still gives real concurrent-
# Socket.tcp()-creation coverage cross-platform.
_WSA_STARTUP_CONCURRENT_CALLERS_ALL_SUCCEED = '''
import socket
import threading
import atomic

class Worker:
	ok: atomic.Atomic[usize]

	@staticmethod
	def make() -> Worker:
		return Worker.__allocate__( ok = atomic.Atomic[usize]( 0 ) )

	def run( self ) -> None:
		match socket.Socket.tcp():
			case Result.Ok( s ):
				s.close()
				self.ok.fetch_add( 1 )
			case Result.Err( _ ):
				pass

def main() -> i32:
	w: Worker = Worker.make()
	closure: Closure[[], None] = w.run
	threads: list[threading.Thread] = list[threading.Thread]()
	i: usize = 0
	while i < 16:
		threads.append( threading.Thread( closure ) )
		with compiler.wrap_arithmetic:
			i += 1
	i = 0
	while i < 16:
		t: threading.Thread = threads.__getitem__( i ).unwrap( 'getitem failed' )
		t.join()
		with compiler.wrap_arithmetic:
			i += 1
	if w.ok.load() != usize( 16 ):
		return 1
	return 0
'''

_TCP_IPV6_LOOPBACK = '''
import socket

def main() -> i32:
	server: socket.Socket = socket.Socket.tcp( socket.AF_INET6 ).unwrap( 'server create' )
	server.bind( '::1', u16( 0 )).unwrap( 'server bind' )
	server.listen().unwrap( 'server listen' )
	bound: socket.SocketAddr = server.getsockname().unwrap( 'server getsockname' )
	if bound.host != '::1':
		return 1

	client: socket.Socket = socket.Socket.tcp( socket.AF_INET6 ).unwrap( 'client create' )
	client.connect( '::1', bound.port ).unwrap( 'client connect' )

	match server.accept():
		case Result.Ok( pair ):
			conn: socket.Socket = pair[0]
		case Result.Err( _ ):
			return 2

	msg: bytes = b'v6!'
	client.send( msg.get_const_ptr(), usize( 3 )).unwrap( 'client send' )
	rbuf: bytearray = bytearray( 16 )
	received: usize = conn.recv( rbuf.get_ptr(), usize( 16 )).unwrap( 'conn recv' )
	if received != usize( 3 ):
		return 3

	conn.close()
	client.close()
	server.close()
	return 0
'''

_UDP_ROUND_TRIP = '''
import socket

def main() -> i32:
	u1: socket.Socket = socket.Socket.udp().unwrap( 'u1 create' )
	u1.bind( '127.0.0.1', u16( 0 )).unwrap( 'u1 bind' )
	u2: socket.Socket = socket.Socket.udp().unwrap( 'u2 create' )
	u2.bind( '127.0.0.1', u16( 0 )).unwrap( 'u2 bind' )

	a1: socket.SocketAddr = u1.getsockname().unwrap( 'u1 getsockname' )
	a2: socket.SocketAddr = u2.getsockname().unwrap( 'u2 getsockname' )

	msg: bytes = b'udp!'
	sent: usize = u1.sendto( msg.get_const_ptr(), usize( 4 ), '127.0.0.1', a2.port ).unwrap( 'sendto' )
	if sent != usize( 4 ):
		return 1

	rbuf: bytearray = bytearray( 16 )
	match u2.recvfrom( rbuf.get_ptr(), usize( 16 )):
		case Result.Ok( pair ):
			n: usize = pair[0]
			from_addr: socket.SocketAddr = pair[1]
		case Result.Err( _ ):
			return 2
	if n != usize( 4 ):
		return 3
	if from_addr.port != a1.port:
		return 4
	if from_addr.host != '127.0.0.1':
		return 5

	u1.close()
	u2.close()
	return 0
'''

# Two LIVE sockets bound to the identical address:port, neither with
# SO_REUSEADDR - must fail with AddressInUse. (A THIRD socket then binding
# to the same port WITH SO_REUSEADDR while the first is still live is NOT
# tested here - confirmed during development to be genuinely environment-
# dependent even on real Windows, WSAEACCES rather than success in a
# WSL2/Hyper-V-networked setup; see test_set_reuseaddr_allows_rebind below
# for the reliable, standard rebind-after-close scenario instead.)
_BIND_ADDRESS_IN_USE = '''
import socket

def main() -> i32:
	sA: socket.Socket = socket.Socket.tcp().unwrap( 'sA create' )
	sA.bind( '127.0.0.1', u16( 0 )).unwrap( 'sA bind' )
	pA: socket.SocketAddr = sA.getsockname().unwrap( 'sA getsockname' )

	sB: socket.Socket = socket.Socket.tcp().unwrap( 'sB create' )
	match sB.bind( '127.0.0.1', pA.port ):
		case Result.Ok( _ ):
			return 1  # should have failed - address already in use
		case Result.Err( e ):
			if e != OSError.AddressInUse:
				return 2
	return 0
'''

# The standard SO_REUSEADDR use case: a listening socket closes, and a fresh
# socket immediately rebinds the same port. set_reuseaddr(True) must not
# error and the rebind must succeed.
_SET_REUSEADDR_ALLOWS_REBIND = '''
import socket

def main() -> i32:
	sA: socket.Socket = socket.Socket.tcp().unwrap( 'sA create' )
	sA.bind( '127.0.0.1', u16( 0 )).unwrap( 'sA bind' )
	sA.listen().unwrap( 'sA listen' )
	pA: socket.SocketAddr = sA.getsockname().unwrap( 'sA getsockname' )
	sA.close()

	sB: socket.Socket = socket.Socket.tcp().unwrap( 'sB create' )
	sB.set_reuseaddr( True ).unwrap( 'sB set_reuseaddr' )
	sB.bind( '127.0.0.1', pA.port ).unwrap( 'sB bind after close+reuseaddr' )
	return 0
'''

# bind()+immediate close() to obtain a real closed local port (rather than
# assuming some external port is closed), then connect() to it.
_CONNECT_REFUSED = '''
import socket

def main() -> i32:
	sC: socket.Socket = socket.Socket.tcp().unwrap( 'sC create' )
	sC.bind( '127.0.0.1', u16( 0 )).unwrap( 'sC bind' )
	pC: socket.SocketAddr = sC.getsockname().unwrap( 'sC getsockname' )
	sC.close()

	sD: socket.Socket = socket.Socket.tcp().unwrap( 'sD create' )
	match sD.connect( '127.0.0.1', pC.port ):
		case Result.Ok( _ ):
			return 1  # should have failed - nothing listening on pC's port
		case Result.Err( e ):
			if e != OSError.ConnectionRefused:
				return 2
	return 0
'''

_BIND_INVALID_ADDRESS = '''
import socket

def main() -> i32:
	sE: socket.Socket = socket.Socket.tcp().unwrap( 'sE create' )
	match sE.bind( 'not-an-ip', u16( 0 )):
		case Result.Ok( _ ):
			return 1  # should have failed - not a valid IPv4 literal
		case Result.Err( e ):
			if e != OSError.Invalid:
				return 2
	return 0
'''

# connect() now resolves hostnames via getaddrinfo() instead of requiring a
# pre-resolved IP literal - 'localhost' is the standard hermetic choice
# (every OS resolves it locally, no real DNS round trip). Binds an ephemeral
# loopback listener first (same pattern as _TCP_LOOPBACK_ECHO) so the actual
# connect target is real and live, then connects to it BY HOSTNAME.
_TCP_CONNECT_BY_HOSTNAME = '''
import socket

def main() -> i32:
	server: socket.Socket = socket.Socket.tcp().unwrap( 'server create' )
	server.bind( '127.0.0.1', u16( 0 )).unwrap( 'server bind' )
	server.listen().unwrap( 'server listen' )
	bound: socket.SocketAddr = server.getsockname().unwrap( 'server getsockname' )

	client: socket.Socket = socket.Socket.tcp().unwrap( 'client create' )
	client.connect( 'localhost', bound.port ).unwrap( 'client connect via hostname' )

	match server.accept():
		case Result.Ok( pair ):
			conn: socket.Socket = pair[0]
		case Result.Err( _ ):
			return 1

	conn.close()
	client.close()
	server.close()
	return 0
'''

# Same as above but AF_INET6 - exercises _resolve_v6/hints.ai_family=AF_INET6
# specifically (getaddrinfo's own family filtering), not just the AF_INET path.
_TCP_CONNECT_BY_HOSTNAME_IPV6 = '''
import socket

def main() -> i32:
	server: socket.Socket = socket.Socket.tcp( socket.AF_INET6 ).unwrap( 'server create' )
	server.bind( '::1', u16( 0 )).unwrap( 'server bind' )
	server.listen().unwrap( 'server listen' )
	bound: socket.SocketAddr = server.getsockname().unwrap( 'server getsockname' )

	client: socket.Socket = socket.Socket.tcp( socket.AF_INET6 ).unwrap( 'client create' )
	client.connect( 'localhost', bound.port ).unwrap( 'client connect via hostname' )

	match server.accept():
		case Result.Ok( pair ):
			conn: socket.Socket = pair[0]
		case Result.Err( _ ):
			return 1

	conn.close()
	client.close()
	server.close()
	return 0
'''

# A bogus/unresolvable hostname must fail connect() with
# OSError.NameResolutionFailed - distinct from OSError.Invalid, which
# _build_sockaddr_in/6 still use for a syntactically-unparseable IP literal
# (see lib/socket.py's own _resolve_v4/_resolve_v6 comment for why the two
# are kept separate rather than collapsed into one bucket).
_CONNECT_BOGUS_HOSTNAME = '''
import socket

def main() -> i32:
	client: socket.Socket = socket.Socket.tcp().unwrap( 'client create' )
	match client.connect( 'this-host-should-not-exist.invalid', u16( 80 )):
		case Result.Ok( _ ):
			return 1  # should have failed - not a resolvable hostname
		case Result.Err( e ):
			if e != OSError.NameResolutionFailed:
				return 2
	return 0
'''

# Numeric IP literals must keep working through connect() now that it always
# routes through getaddrinfo() (real getaddrinfo recognizes numeric literals
# without any extra flag, per POSIX/Winsock - this is the regression check).
_CONNECT_BY_IP_LITERAL_STILL_WORKS = '''
import socket

def main() -> i32:
	server: socket.Socket = socket.Socket.tcp().unwrap( 'server create' )
	server.bind( '127.0.0.1', u16( 0 )).unwrap( 'server bind' )
	server.listen().unwrap( 'server listen' )
	bound: socket.SocketAddr = server.getsockname().unwrap( 'server getsockname' )

	client: socket.Socket = socket.Socket.tcp().unwrap( 'client create' )
	client.connect( '127.0.0.1', bound.port ).unwrap( 'client connect via IP literal' )

	match server.accept():
		case Result.Ok( pair ):
			conn: socket.Socket = pair[0]
		case Result.Err( _ ):
			return 1

	conn.close()
	client.close()
	server.close()
	return 0
'''

# socket.resolve() - the standalone hostname->IP-literal-strings lookup built
# on the same _resolve_v4/_resolve_v6 machinery connect() uses internally.
# Checked for both families since resolve() takes an explicit family arg.
_RESOLVE_LOCALHOST = '''
import socket

def main() -> i32:
	v4: list[str] = socket.resolve( 'localhost', socket.AF_INET ).unwrap( 'resolve v4' )
	if len( v4 ) == 0:
		return 1
	found_v4: bool = False
	for i in range( len( v4 )):
		if v4.__getitem__( i ).unwrap( 'v4 idx' ) == '127.0.0.1':
			found_v4 = True
	if not found_v4:
		return 2

	v6: list[str] = socket.resolve( 'localhost', socket.AF_INET6 ).unwrap( 'resolve v6' )
	if len( v6 ) == 0:
		return 3
	found_v6: bool = False
	for i in range( len( v6 )):
		if v6.__getitem__( i ).unwrap( 'v6 idx' ) == '::1':
			found_v6 = True
	if not found_v6:
		return 4

	return 0
'''

# resolve() on a bogus hostname must fail the same way connect() does.
_RESOLVE_BOGUS_HOSTNAME = '''
import socket

def main() -> i32:
	match socket.resolve( 'this-host-should-not-exist.invalid', socket.AF_INET ):
		case Result.Ok( _ ):
			return 1  # should have failed - not a resolvable hostname
		case Result.Err( e ):
			if e != OSError.NameResolutionFailed:
				return 2
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile socket tests' )
class SocketBehaviorTests( RealCompileMixin, unittest.TestCase ):
	def _run( self, source: str ) -> None:
		compiler = self._compile_source( source )
		c_source = emitter_c.emit_c( compiler )
		self._assert_compiles_and_runs( c_source, expected_exit = 0, compiler = compiler )

	def test_tcp_loopback_echo( self ) -> None:
		self._run( _TCP_LOOPBACK_ECHO )

	def test_tcp_loopback_echo_tuple_destructure( self ) -> None:
		self._run( _TCP_LOOPBACK_ECHO_TUPLE_DESTRUCTURE )

	def test_send_all_delivers_everything( self ) -> None:
		self._run( _SEND_ALL_DELIVERS_EVERYTHING )

	def test_send_all_error_propagates( self ) -> None:
		self._run( _SEND_ALL_ERROR_PROPAGATES )

	def test_recvbuffer_accumulates_across_calls( self ) -> None:
		self._run( _RECVBUFFER_ACCUMULATES_ACROSS_CALLS )

	def test_recvbuffer_fill_from_returns_zero_on_peer_close( self ) -> None:
		self._run( _RECVBUFFER_FILL_FROM_RETURNS_ZERO_ON_PEER_CLOSE )

	def test_wsa_startup_concurrent_callers_all_succeed( self ) -> None:
		self._run( _WSA_STARTUP_CONCURRENT_CALLERS_ALL_SUCCEED )

	def test_tcp_ipv6_loopback( self ) -> None:
		self._run( _TCP_IPV6_LOOPBACK )

	def test_udp_round_trip( self ) -> None:
		self._run( _UDP_ROUND_TRIP )

	def test_bind_address_in_use( self ) -> None:
		self._run( _BIND_ADDRESS_IN_USE )

	def test_set_reuseaddr_allows_rebind( self ) -> None:
		self._run( _SET_REUSEADDR_ALLOWS_REBIND )

	def test_connect_refused( self ) -> None:
		self._run( _CONNECT_REFUSED )

	def test_bind_invalid_address( self ) -> None:
		self._run( _BIND_INVALID_ADDRESS )

	def test_tcp_connect_by_hostname( self ) -> None:
		self._run( _TCP_CONNECT_BY_HOSTNAME )

	def test_tcp_connect_by_hostname_ipv6( self ) -> None:
		self._run( _TCP_CONNECT_BY_HOSTNAME_IPV6 )

	def test_connect_bogus_hostname( self ) -> None:
		self._run( _CONNECT_BOGUS_HOSTNAME )

	def test_connect_by_ip_literal_still_works( self ) -> None:
		self._run( _CONNECT_BY_IP_LITERAL_STILL_WORKS )

	def test_resolve_localhost( self ) -> None:
		self._run( _RESOLVE_LOCALHOST )

	def test_resolve_bogus_hostname( self ) -> None:
		self._run( _RESOLVE_BOGUS_HOSTNAME )


if __name__ == '__main__':
	unittest.main()
