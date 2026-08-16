# Real-compile-and-run tests for lib/http/client.py's Phase 0 pieces (see
# PLAN_HTTP_CLIENT.md): the zero-prerequisite pure wire-format functions -
# HTTPHeaders, status-line/header-line parsing, percent-encoding, base64
# encoding, and chunked-transfer decoding. None of this depends on the
# socket library (still unimplemented - see the plan), so it's tested here
# entirely against in-memory strings/bytes, the same way lib/codecs/*.py's
# own encode/decode round trips are tested.
#
# Each sub-program's main() -> i32 returns a distinct nonzero code per failed
# assertion (0 = every assertion passed) - test_support.assert_programs_run
# decodes a failure back to the offending case name and sub-code.

# stdlib imports:
import unittest

# local imports:
import test_support


class HTTPClientPhase0Tests( test_support.RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile http.client tests' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'headers_case_insensitive_add_set_get', '''
from http.client import HTTPHeaders

def main() -> i32:
	h: HTTPHeaders = HTTPHeaders()
	h.add( 'Content-Type', 'text/plain' )
	h.add( 'X-Foo', 'a' )
	h.set( 'x-foo', 'b' ) # case-insensitive overwrite of the X-Foo entry above
	v: str|None = h.get( 'X-FOO' )
	match v:
		case None:
			return 1
		case _:
			if v != 'b':
				return 2
	if h.__len__() != 2:
		return 3
	if h.get( 'nonexistent' ) is not None:
		return 4
	return 0
''' ),
			( 'headers_get_all_repeated_names', '''
from http.client import HTTPHeaders

def main() -> i32:
	h: HTTPHeaders = HTTPHeaders()
	h.add( 'Set-Cookie', 'a=1' )
	h.add( 'Set-Cookie', 'b=2' )
	all_values: list[str] = h.get_all( 'set-cookie' )
	if all_values.__len__() != 2:
		return 1
	if all_values.__getitem__( 0 ).unwrap( 'x' ) != 'a=1':
		return 2
	if all_values.__getitem__( 1 ).unwrap( 'x' ) != 'b=2':
		return 3
	return 0
''' ),
			( 'parse_status_line_ok_and_malformed', '''
from http.client import parse_status_line, HTTPError

def main() -> i32:
	st: tuple[str,u16,str] = parse_status_line( 'HTTP/1.1 200 OK' ).unwrap( 'status line' )
	if st[0] != 'HTTP/1.1':
		return 1
	if st[1] != 200:
		return 2
	if st[2] != 'OK':
		return 3

	# multi-word reason phrase - must not be truncated at the first space
	st2: tuple[str,u16,str] = parse_status_line( 'HTTP/1.1 404 Not Found' ).unwrap( 'status line 2' )
	if st2[2] != 'Not Found':
		return 4

	bad: Result[tuple[str,u16,str], HTTPError] = parse_status_line( 'garbage' )
	if bad.is_ok():
		return 5

	non_numeric: Result[tuple[str,u16,str], HTTPError] = parse_status_line( 'HTTP/1.1 abc OK' )
	if non_numeric.is_ok():
		return 6
	return 0
''' ),
			( 'parse_header_line_and_block', '''
from http.client import parse_header_line, parse_headers, HTTPHeaders, HTTPError

def main() -> i32:
	hl: tuple[str,str] = parse_header_line( 'Content-Length: 42' ).unwrap( 'header line' )
	if hl[0] != 'Content-Length':
		return 1
	if hl[1] != '42':
		return 2

	bad: Result[tuple[str,str], HTTPError] = parse_header_line( 'no colon here' )
	if bad.is_ok():
		return 3

	hdrs: HTTPHeaders = parse_headers( 'Content-Type: text/plain\\r\\nX-Foo: bar\\r\\n' ).unwrap( 'headers' )
	ct: str|None = hdrs.get( 'content-type' )
	match ct:
		case None:
			return 4
		case _:
			if ct != 'text/plain':
				return 5
	if hdrs.__len__() != 2:
		return 6
	return 0
''' ),
			( 'percent_encode_reserved_and_unreserved', '''
from http.client import percent_encode

def main() -> i32:
	if percent_encode( 'hello world!' ) != 'hello%20world%21':
		return 1
	# every RFC 3986 unreserved character must pass through unchanged
	if percent_encode( 'abc-._~XYZ019' ) != 'abc-._~XYZ019':
		return 2
	if percent_encode( '' ) != '':
		return 3
	return 0
''' ),
			( 'base64_encode_known_vectors', '''
from http.client import base64_encode

def main() -> i32:
	# RFC 4648 test vectors
	d1: bytes = 'M'.encode().unwrap( 'encode' )
	if base64_encode( d1 ) != 'TQ==':
		return 1
	d2: bytes = 'Ma'.encode().unwrap( 'encode' )
	if base64_encode( d2 ) != 'TWE=':
		return 2
	d3: bytes = 'Man'.encode().unwrap( 'encode' )
	if base64_encode( d3 ) != 'TWFu':
		return 3
	d4: bytes = 'hello world'.encode().unwrap( 'encode' )
	if base64_encode( d4 ) != 'aGVsbG8gd29ybGQ=':
		return 4
	d5: bytes = ''.encode().unwrap( 'encode' )
	if base64_encode( d5 ) != '':
		return 5
	return 0
''' ),
			( 'decode_chunked_wikipedia_example_and_errors', '''
from http.client import decode_chunked, HTTPError

def main() -> i32:
	# the canonical RFC 7230-style chunked example
	chunked_body: bytes = '4\\r\\nWiki\\r\\n5\\r\\npedia\\r\\n0\\r\\n\\r\\n'.encode().unwrap( 'encode' )
	decoded: bytes = decode_chunked( chunked_body ).unwrap( 'chunked' )
	decoded_str: str = decoded.decode().unwrap( 'decode' )
	if decoded_str != 'Wikipedia':
		return 1

	empty_chunked: bytes = '0\\r\\n\\r\\n'.encode().unwrap( 'encode' )
	decoded_empty: bytes = decode_chunked( empty_chunked ).unwrap( 'chunked empty' )
	if decoded_empty.__len__() != 0:
		return 2

	bad_hex: bytes = 'zz\\r\\n'.encode().unwrap( 'encode' )
	bad_result: Result[bytes, HTTPError] = decode_chunked( bad_hex )
	if bad_result.is_ok():
		return 3

	truncated: bytes = '4\\r\\nWik'.encode().unwrap( 'encode' )
	truncated_result: Result[bytes, HTTPError] = decode_chunked( truncated )
	if truncated_result.is_ok():
		return 4

	return 0
''' ),
		])


class HTTPConnectionLoopbackTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' real-compile-and-run tests for HTTPConnection/Response (see
	PLAN_HTTP_CLIENT.md's Phase 3a) over an actual loopback TCP connection -
	a background thread (lib/threading.py) plays a minimal HTTP server
	(bind/listen/accept/recv/send via lib/socket.py directly), the main
	thread is the HTTPConnection client. Kept in a separate assert_programs_
	run cluster from HTTPClientPhase0Tests above (real sockets/threads, not
	pure in-memory string/bytes transforms). '''
	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile http.client tests' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'get_request_content_length_body_and_headers', '''
import threading
from atomic import Atomic
from socket import Socket
from http.client import HTTPConnection, Response, HTTPError

class EchoServer:
	port: u16
	ready: Atomic[bool]
	ok: Atomic[bool]

	def __init__( self, port: u16 ) -> None:
		self.port = port
		self.ready = Atomic[bool]( False )
		self.ok = Atomic[bool]( False )

	def run( self ) -> None:
		listener: Socket = Socket.tcp().unwrap( 'server: tcp' )
		listener.set_reuseaddr( True ).unwrap( 'server: reuseaddr' )
		listener.bind( '127.0.0.1', self.port ).unwrap( 'server: bind' )
		listener.listen( 1 ).unwrap( 'server: listen' )
		self.ready.store( True )
		match listener.accept():
			case Result.Ok( pair ):
				conn: Socket = pair[0]
				buf: bytearray = bytearray( 4096 )
				conn.recv( buf.get_ptr(), 4096 ).unwrap( 'server: recv' )
				response: str = 'HTTP/1.1 200 OK\\r\\nContent-Type: text/plain\\r\\nContent-Length: 5\\r\\n\\r\\nhello'
				rb: bytes = response.encode().unwrap( 'server: encode' )
				conn.send( rb.get_const_ptr(), rb.__len__() ).unwrap( 'server: send' )
				conn.close()
				self.ok.store( True )
			case Result.Err( _ ):
				pass
		listener.close()

def main() -> i32:
	server: EchoServer = EchoServer( u16( 18765 ))
	t: threading.Thread = threading.Thread( server.run )
	while not server.ready.load():
		pass

	conn: HTTPConnection = HTTPConnection.connect( '127.0.0.1', u16( 18765 )).unwrap( 'client connect' )
	conn.request( 'GET', '/', None, None ).unwrap( 'client request' )
	resp: Response = conn.getresponse().unwrap( 'client getresponse' )
	conn.close()
	t.join()

	if not server.ok.load():
		return 1
	if resp.status_code != 200:
		return 2
	body: str = resp.text().unwrap( 'text' )
	if body != 'hello':
		return 3
	ct: str|None = resp.headers.get( 'Content-Type' )
	match ct:
		case None:
			return 4
		case _:
			if ct != 'text/plain':
				return 5
	return 0
''' ),
			( 'get_request_chunked_body', '''
import threading
from atomic import Atomic
from socket import Socket
from http.client import HTTPConnection, Response, HTTPError

class ChunkedServer:
	port: u16
	ready: Atomic[bool]
	ok: Atomic[bool]

	def __init__( self, port: u16 ) -> None:
		self.port = port
		self.ready = Atomic[bool]( False )
		self.ok = Atomic[bool]( False )

	def run( self ) -> None:
		listener: Socket = Socket.tcp().unwrap( 'server: tcp' )
		listener.set_reuseaddr( True ).unwrap( 'server: reuseaddr' )
		listener.bind( '127.0.0.1', self.port ).unwrap( 'server: bind' )
		listener.listen( 1 ).unwrap( 'server: listen' )
		self.ready.store( True )
		match listener.accept():
			case Result.Ok( pair ):
				conn: Socket = pair[0]
				buf: bytearray = bytearray( 4096 )
				conn.recv( buf.get_ptr(), 4096 ).unwrap( 'server: recv' )
				response: str = 'HTTP/1.1 200 OK\\r\\nTransfer-Encoding: chunked\\r\\n\\r\\n4\\r\\nWiki\\r\\n5\\r\\npedia\\r\\n0\\r\\n\\r\\n'
				rb: bytes = response.encode().unwrap( 'server: encode' )
				conn.send( rb.get_const_ptr(), rb.__len__() ).unwrap( 'server: send' )
				conn.close()
				self.ok.store( True )
			case Result.Err( _ ):
				pass
		listener.close()

def main() -> i32:
	server: ChunkedServer = ChunkedServer( u16( 18766 ))
	t: threading.Thread = threading.Thread( server.run )
	while not server.ready.load():
		pass

	conn: HTTPConnection = HTTPConnection.connect( '127.0.0.1', u16( 18766 )).unwrap( 'client connect' )
	conn.request( 'GET', '/', None, None ).unwrap( 'client request' )
	resp: Response = conn.getresponse().unwrap( 'client getresponse' )
	conn.close()
	t.join()

	if not server.ok.load():
		return 1
	if resp.status_code != 200:
		return 2
	body: str = resp.text().unwrap( 'text' )
	if body != 'Wikipedia':
		return 3
	return 0
''' ),
			( 'connect_refused_surfaces_as_err', '''
from http.client import HTTPConnection, HTTPError

def main() -> i32:
	result: Result[HTTPConnection, HTTPError] = HTTPConnection.connect( '127.0.0.1', u16( 18767 )) # nothing listening
	if result.is_ok():
		return 1
	return 0
''' ),
		])


class SessionLoopbackTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' real-compile-and-run tests for Session (see PLAN_HTTP_CLIENT.md's
	Phase 3b) over real loopback TCP - same background-thread-plays-a-
	server approach as HTTPConnectionLoopbackTests above, but exercising
	Session's own cookie jar, redirect-following, params=/form=, and auth=
	handling by inspecting the raw bytes each fake server actually
	received. '''
	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile http.client tests' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'session_cookie_jar_roundtrip', '''
import compiler
import threading
from atomic import Atomic
from socket import Socket
from http.client import Session, Response

class CookieJarServer:
	port: u16
	ready: Atomic[bool]
	cookie_seen: Atomic[bool]

	def __init__( self, port: u16 ) -> None:
		self.port = port
		self.ready = Atomic[bool]( False )
		self.cookie_seen = Atomic[bool]( False )

	def run( self ) -> None:
		listener: Socket = Socket.tcp().unwrap( 'server: tcp' )
		listener.set_reuseaddr( True ).unwrap( 'server: reuseaddr' )
		listener.bind( '127.0.0.1', self.port ).unwrap( 'server: bind' )
		listener.listen( 2 ).unwrap( 'server: listen' )
		self.ready.store( True )

		match listener.accept():
			case Result.Ok( pair1 ):
				conn1: Socket = pair1[0]
				buf1: bytearray = bytearray( 4096 )
				conn1.recv( buf1.get_ptr(), 4096 ).unwrap( 'server: recv 1' )
				resp1: str = 'HTTP/1.1 200 OK\\r\\nSet-Cookie: sid=abc123\\r\\nContent-Length: 2\\r\\n\\r\\nok'
				rb1: bytes = resp1.encode().unwrap( 'server: encode 1' )
				conn1.send( rb1.get_const_ptr(), rb1.__len__() ).unwrap( 'server: send 1' )
				conn1.close()
			case Result.Err( _ ):
				pass

		match listener.accept():
			case Result.Ok( pair2 ):
				conn2: Socket = pair2[0]
				buf2: bytearray = bytearray( 4096 )
				recv_total2: usize = 0
				attempts2: usize = 0
				req2: str = ''
				with compiler.wrap_arithmetic:
					while attempts2 < 50:
						dest2: Ptr[u8] = buf2.get_ptr() + recv_total2
						room2: usize = 4096 - recv_total2
						n2: usize = conn2.recv( dest2, room2 ).unwrap( 'server: recv 2' )
						recv_total2 += n2
						attempts2 += 1
						req2 = buf2.decode().unwrap( 'server: decode 2' )
						if req2.find( 'Cookie: sid=abc123' ).is_ok() or n2 == 0:
							break
				if req2.find( 'Cookie: sid=abc123' ).is_ok():
					self.cookie_seen.store( True )
				resp2: str = 'HTTP/1.1 200 OK\\r\\nContent-Length: 2\\r\\n\\r\\nok'
				rb2: bytes = resp2.encode().unwrap( 'server: encode 2' )
				conn2.send( rb2.get_const_ptr(), rb2.__len__() ).unwrap( 'server: send 2' )
				conn2.close()
			case Result.Err( _ ):
				pass
		listener.close()

def main() -> i32:
	server: CookieJarServer = CookieJarServer( u16( 18770 ))
	t: threading.Thread = threading.Thread( server.run )
	while not server.ready.load():
		pass

	s: Session = Session()
	url: str = 'http://127.0.0.1:18770/'
	r1: Response = s.get( url ).unwrap( 'client request 1' )
	r2: Response = s.get( url ).unwrap( 'client request 2' )
	t.join()

	if r1.status_code != 200:
		return 1
	if r2.status_code != 200:
		return 2
	if not server.cookie_seen.load():
		return 3
	return 0
''' ),
			( 'session_follows_redirect_and_merges_query_params', '''
import compiler
import threading
from atomic import Atomic
from socket import Socket
from http.client import Session, Response

class RedirectServer:
	port: u16
	ready: Atomic[bool]
	saw_query: Atomic[bool]

	def __init__( self, port: u16 ) -> None:
		self.port = port
		self.ready = Atomic[bool]( False )
		self.saw_query = Atomic[bool]( False )

	def run( self ) -> None:
		listener: Socket = Socket.tcp().unwrap( 'server: tcp' )
		listener.set_reuseaddr( True ).unwrap( 'server: reuseaddr' )
		listener.bind( '127.0.0.1', self.port ).unwrap( 'server: bind' )
		listener.listen( 2 ).unwrap( 'server: listen' )
		self.ready.store( True )

		match listener.accept():
			case Result.Ok( pair1 ):
				conn1: Socket = pair1[0]
				buf1: bytearray = bytearray( 4096 )
				recv_total1: usize = 0
				attempts1: usize = 0
				req1: str = ''
				with compiler.wrap_arithmetic:
					while attempts1 < 50:
						dest1: Ptr[u8] = buf1.get_ptr() + recv_total1
						room1: usize = 4096 - recv_total1
						n1: usize = conn1.recv( dest1, room1 ).unwrap( 'server: recv 1' )
						recv_total1 += n1
						attempts1 += 1
						req1 = buf1.decode().unwrap( 'server: decode 1' )
						if req1.find( '/search?q=hello%20world' ).is_ok() or n1 == 0:
							break
				if req1.find( '/search?q=hello%20world' ).is_ok():
					self.saw_query.store( True )
				resp1: str = 'HTTP/1.1 302 Found\\r\\nLocation: http://127.0.0.1:18771/final\\r\\nContent-Length: 0\\r\\n\\r\\n'
				rb1: bytes = resp1.encode().unwrap( 'server: encode 1' )
				conn1.send( rb1.get_const_ptr(), rb1.__len__() ).unwrap( 'server: send 1' )
				conn1.close()
			case Result.Err( _ ):
				pass

		match listener.accept():
			case Result.Ok( pair2 ):
				conn2: Socket = pair2[0]
				buf2: bytearray = bytearray( 4096 )
				conn2.recv( buf2.get_ptr(), 4096 ).unwrap( 'server: recv 2' )
				resp2: str = 'HTTP/1.1 200 OK\\r\\nContent-Length: 5\\r\\n\\r\\nfinal'
				rb2: bytes = resp2.encode().unwrap( 'server: encode 2' )
				conn2.send( rb2.get_const_ptr(), rb2.__len__() ).unwrap( 'server: send 2' )
				conn2.close()
			case Result.Err( _ ):
				pass
		listener.close()

def main() -> i32:
	server: RedirectServer = RedirectServer( u16( 18771 ))
	t: threading.Thread = threading.Thread( server.run )
	while not server.ready.load():
		pass

	s: Session = Session()
	params: dict[str,str] = dict[str,str]()
	params[ 'q' ] = 'hello world'
	r: Response = s.get( 'http://127.0.0.1:18771/search', params = params ).unwrap( 'client request' )
	t.join()

	if not server.saw_query.load():
		return 1
	if r.status_code != 200:
		return 2
	body: str = r.text().unwrap( 'text' )
	if body != 'final':
		return 3
	if r.url != 'http://127.0.0.1:18771/final':
		return 4
	return 0
''' ),
			( 'session_form_post_and_basic_auth', '''
import compiler
import threading
from atomic import Atomic
from socket import Socket
from http.client import Session, Response

class FormAuthServer:
	port: u16
	ready: Atomic[bool]
	ok: Atomic[bool]

	def __init__( self, port: u16 ) -> None:
		self.port = port
		self.ready = Atomic[bool]( False )
		self.ok = Atomic[bool]( False )

	def run( self ) -> None:
		listener: Socket = Socket.tcp().unwrap( 'server: tcp' )
		listener.set_reuseaddr( True ).unwrap( 'server: reuseaddr' )
		listener.bind( '127.0.0.1', self.port ).unwrap( 'server: bind' )
		listener.listen( 1 ).unwrap( 'server: listen' )
		self.ready.store( True )
		match listener.accept():
			case Result.Ok( pair ):
				conn: Socket = pair[0]
				buf: bytearray = bytearray( 4096 )
				recv_total: usize = 0
				attempts: usize = 0
				req: str = ''
				with compiler.wrap_arithmetic:
					while attempts < 50:
						dest: Ptr[u8] = buf.get_ptr() + recv_total
						room: usize = 4096 - recv_total
						n: usize = conn.recv( dest, room ).unwrap( 'server: recv' )
						recv_total += n
						attempts += 1
						req = buf.decode().unwrap( 'server: decode' )
						has_ct0: bool = req.find( 'Content-Type: application/x-www-form-urlencoded' ).is_ok()
						has_body0: bool = req.find( 'a=1&b=2' ).is_ok()
						has_auth0: bool = req.find( 'Authorization: Basic dXNlcjpwYXNz' ).is_ok()
						if ( has_ct0 and has_body0 and has_auth0 ) or n == 0:
							break
				has_ct: bool = req.find( 'Content-Type: application/x-www-form-urlencoded' ).is_ok()
				has_body: bool = req.find( 'a=1&b=2' ).is_ok()
				has_auth: bool = req.find( 'Authorization: Basic dXNlcjpwYXNz' ).is_ok()
				if has_ct and has_body and has_auth:
					self.ok.store( True )
				resp: str = 'HTTP/1.1 200 OK\\r\\nContent-Length: 2\\r\\n\\r\\nok'
				rb: bytes = resp.encode().unwrap( 'server: encode' )
				conn.send( rb.get_const_ptr(), rb.__len__() ).unwrap( 'server: send' )
				conn.close()
			case Result.Err( _ ):
				pass
		listener.close()

def main() -> i32:
	server: FormAuthServer = FormAuthServer( u16( 18772 ))
	t: threading.Thread = threading.Thread( server.run )
	while not server.ready.load():
		pass

	s: Session = Session()
	form: dict[str,str] = dict[str,str]()
	form[ 'a' ] = '1'
	form[ 'b' ] = '2'
	r: Response = s.post( 'http://127.0.0.1:18772/submit', form = form, auth = ( 'user', 'pass' )).unwrap( 'client request' )
	t.join()

	if r.status_code != 200:
		return 1
	if not server.ok.load():
		return 2
	return 0
''' ),
		])


if __name__ == '__main__':
	unittest.main()
