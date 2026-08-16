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
	if bound.host() != '127.0.0.1':
		return 1

	client: socket.Socket = socket.Socket.tcp().unwrap( 'client create' )
	client.connect( '127.0.0.1', bound.port() ).unwrap( 'client connect' )

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

_TCP_IPV6_LOOPBACK = '''
import socket

def main() -> i32:
	server: socket.Socket = socket.Socket.tcp( socket.AF_INET6 ).unwrap( 'server create' )
	server.bind( '::1', u16( 0 )).unwrap( 'server bind' )
	server.listen().unwrap( 'server listen' )
	bound: socket.SocketAddr = server.getsockname().unwrap( 'server getsockname' )
	if bound.host() != '::1':
		return 1

	client: socket.Socket = socket.Socket.tcp( socket.AF_INET6 ).unwrap( 'client create' )
	client.connect( '::1', bound.port() ).unwrap( 'client connect' )

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
	sent: usize = u1.sendto( msg.get_const_ptr(), usize( 4 ), '127.0.0.1', a2.port() ).unwrap( 'sendto' )
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
	if from_addr.port() != a1.port():
		return 4
	if from_addr.host() != '127.0.0.1':
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
	match sB.bind( '127.0.0.1', pA.port() ):
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
	sB.bind( '127.0.0.1', pA.port() ).unwrap( 'sB bind after close+reuseaddr' )
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
	match sD.connect( '127.0.0.1', pC.port() ):
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


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile socket tests' )
class SocketBehaviorTests( RealCompileMixin, unittest.TestCase ):
	def _run( self, source: str ) -> None:
		compiler = self._compile_source( source )
		c_source = emitter_c.emit_c( compiler )
		self._assert_compiles_and_runs( c_source, expected_exit = 0, compiler = compiler )

	def test_tcp_loopback_echo( self ) -> None:
		self._run( _TCP_LOOPBACK_ECHO )

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


if __name__ == '__main__':
	unittest.main()
