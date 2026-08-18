# Real-compile-and-run tests for lib/ssl.py (see PLAN_SSL.md).
#
# Two classes:
#   SSLPhase0Tests - the SSLError enum, zero prerequisites, no I/O. Runs
#     everywhere a C compiler is available, same as every other Phase 0 test
#     in this codebase.
#   SSLWindowsHandshakeTests - the real Schannel/SSPI backend (Windows only -
#     lib/ssl.py has no other backend yet, see PLAN_SSL.md). Unlike every
#     other *_test.py in this codebase, this genuinely dials out over the
#     network: there is no local TLS server to loop back against without
#     first implementing TLS *server*-side Schannel (out of scope - lib/ssl.py
#     is client-only), so this drives the compiled MetalPy program against
#     real public endpoints instead - example.com for the success path, and
#     badssl.com's expired/wrong-host/self-signed fixtures (a public service
#     that exists specifically for this kind of TLS client testing) for the
#     three certificate-failure paths. Requires network egress; set
#     METALPY_TEST_NETWORK=0 to skip if that's not available.

# stdlib imports:
import os
import unittest

# local imports:
import test_support


class SSLPhase0Tests( test_support.RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile ssl tests' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'sslerror_values_distinct', '''
from ssl import SSLError

def main() -> i32:
	if SSLError.HandshakeFailed == SSLError.CertificateVerifyFailed:
		return 1
	if SSLError.CertificateExpired == SSLError.HostnameMismatch:
		return 2
	if SSLError.ProtocolError == SSLError.Closed:
		return 3
	if SSLError.Other == SSLError.HandshakeFailed:
		return 4
	return 0
''' ),
			( 'sslerror_match_dispatch', '''
from ssl import SSLError

def classify( e: SSLError ) -> i32:
	match e:
		case SSLError.CertificateVerifyFailed:
			return 100
		case SSLError.CertificateExpired:
			return 101
		case SSLError.HostnameMismatch:
			return 102
		case _:
			return 999

def main() -> i32:
	if classify( SSLError.CertificateVerifyFailed ) != 100:
		return 1
	if classify( SSLError.CertificateExpired ) != 101:
		return 2
	if classify( SSLError.HostnameMismatch ) != 102:
		return 3
	if classify( SSLError.HandshakeFailed ) != 999:
		return 4
	return 0
''' ),
		])


_NETWORK_OK = os.environ.get( 'METALPY_TEST_NETWORK', '1' ) not in ( '0', 'false', 'False' )


@unittest.skipUnless( os.name == 'nt', 'lib/ssl.py only has a Windows (Schannel) backend so far - see PLAN_SSL.md' )
@unittest.skipUnless( _NETWORK_OK, 'set METALPY_TEST_NETWORK=0 to acknowledge - these tests dial out to example.com/badssl.com' )
class SSLWindowsHandshakeTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' Drives the real Schannel backend against real public TLS endpoints -
	both struct layouts (SecBuffer/SecBufferDesc/SCHANNEL_CRED/SecHandle/
	SecPkgContext_StreamSizes) and the full handshake/encrypt/decrypt call
	sequence were independently validated against the real Windows SDK
	headers and a standalone C client before being ported here (see
	PLAN_SSL.md and lib/ssl.py's own header comment) - these tests exercise
	the MetalPy port of that same, already-proven sequence. Kept in its own
	assert_programs_run cluster, separate from SSLPhase0Tests, since these
	do real network I/O rather than pure in-memory checks (same reasoning as
	http_client_test.py's HTTPConnectionLoopbackTests). '''

	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'handshake_and_encrypted_round_trip', '''
import socket
import ssl

def main() -> i32:
	host: str = "example.com"
	sock: socket.Socket = socket.Socket.tcp().unwrap( 'tcp socket' )
	sock.connect( host, u16( 443 )).unwrap( 'connect' )

	ctx: ssl.SSLContext = ssl.SSLContext.create_default_context().unwrap( 'create_default_context' )
	tls: ssl.SSLSocket = ssl.SSLSocket.wrap_socket( ctx, sock, host ).unwrap( 'wrap_socket' )

	req: str = "GET / HTTP/1.1\\r\\nHost: example.com\\r\\nConnection: close\\r\\n\\r\\n"
	tls.send_all( req.get_cstr(), len( req )).unwrap( 'send_all' )

	buf: bytearray = bytearray( 4096 )
	total: usize = 0
	saw_ok_status: bool = False
	iters: i32 = 0
	while True:
		n: usize = tls.recv( buf.get_ptr(), usize( 4096 )).unwrap( 'recv' )
		if n == 0:
			break
		with compiler.wrap_arithmetic:
			total += n
			iters += 1
		if iters > 1000:
			return 1
		if n >= usize( 12 ) and not saw_ok_status:
			# "HTTP/1.1 200" - check the status digits directly (buf isn't
			# null-terminated, so str.from_cstr doesn't apply to a raw
			# network buffer without first copying out a terminated slice)
			p: ConstPtr[u8] = buf.get_const_ptr()
			if p[9] == u8( 50 ) and p[10] == u8( 48 ) and p[11] == u8( 48 ):
				saw_ok_status = True

	if total == 0:
		return 2
	if not saw_ok_status:
		return 3
	return 0
''' ),
			( 'certificate_failure_mapping', '''
import socket
import ssl

def try_host( host: str ) -> ssl.SSLError:
	sock: socket.Socket = socket.Socket.tcp().unwrap( 'tcp socket' )
	sock.connect( host, u16( 443 )).unwrap( 'connect' )
	ctx: ssl.SSLContext = ssl.SSLContext.create_default_context().unwrap( 'create_default_context' )
	match ssl.SSLSocket.wrap_socket( ctx, sock, host ):
		case Result.Ok( _ ):
			return ssl.SSLError.Other  # unexpected - badssl.com's cert should never validate
		case Result.Err( e ):
			return e

def main() -> i32:
	if try_host( "expired.badssl.com" ) != ssl.SSLError.CertificateExpired:
		return 1
	if try_host( "wrong.host.badssl.com" ) != ssl.SSLError.HostnameMismatch:
		return 2
	if try_host( "self-signed.badssl.com" ) != ssl.SSLError.CertificateVerifyFailed:
		return 3
	return 0
''' ),
		], timeout = 60.0 )  # four real external TLS handshakes in one run - generous margin for network jitter
