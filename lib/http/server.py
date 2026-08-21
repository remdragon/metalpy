'''
HTTP/1.1 server wire-format + accept loop, built directly on lib/tcp.py's
reactor-aware TcpListener/TcpConnection (NOT lib/socket.py's raw Socket, the
way lib/http/client.py's HTTPConnection is - see PLAN_HTTP_SERVER.md for
why) - so the exact same per-connection handler code runs cooperatively
under a reactor.Reactor, or, with no Reactor at all, as an ordinary blocking
accept loop (the same reactor-optional property tcp.py itself documents).

HTTPError          - error type shared by every function/class below
parse_request_line - "GET /path?q=1 HTTP/1.1" -> (method, target, version)
Request            - method/target/path/query/version/headers/body of one
                      parsed request
Response           - status/headers/body, plus Response.text()/.html()/
                      .json()/.bytes_() constructors and write_to() to
                      serialize+send it
serve              - spawns the accept loop onto a reactor.Reactor: each
                      accepted connection is handed to its own fiber, which
                      loops calling the caller-supplied handler once per
                      request on that connection (HTTP/1.1 keep-alive)

v1 scope: Content-Length request/response bodies, and HTTP/1.1 keep-alive
(honoring an explicit "Connection: close" from either side). Explicitly NOT
here: chunked request/response encoding, gzip, TLS, HTTP/2 - see
PLAN_HTTP_SERVER.md.
'''

import compiler
import sys
import reactor
import tcp
import io
from http.client import HTTPHeaders, parse_headers
from urllib.parse import urlsplit, parse_qsl, SplitResult
from json import dumps, JSONValue


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

# @union, not @enum - see lib/http/client.py's own HTTPError for why (a
# CEnum member referenced bare as a real runtime value hits a documented
# lowering.py gap). A distinct type from http.client's own HTTPError (same
# name, different module) - the two never need to interoperate, so no
# conversion between them is provided.
@union
class HTTPError:
	MalformedRequestLine: None
	MalformedHeader:      None
	UnexpectedEOF:        None
	# a clean peer close BETWEEN keep-alive requests (nothing buffered yet
	# when read() returns 0) - expected, not an error; kept as its own
	# HTTPError member (rather than a Request|None return) so
	# _read_one_request can stay Result[Request, HTTPError] and
	# _handle_connection can tell it apart from UnexpectedEOF (a close
	# mid-request, which IS a real error) via ordinary Result.Err(HTTPError.
	# X(_)) matching - see lib/http/client.py's _read_chunked_body for the
	# same "match a specific Err variant, fall through for the rest" shape.
	ConnectionClosed: None
	Other: None


# ---------------------------------------------------------------------------
# parse_request_line
# ---------------------------------------------------------------------------

def parse_request_line( line: str ) -> Result[tuple[str,str,str], HTTPError]:
	''' "GET /path?q=1 HTTP/1.1" -> (method, target, version). Same
	str.partition()-based shape as http.client's own parse_status_line,
	just three space-separated fields instead of two. '''
	first: tuple[str,str,str] = line.partition( ' ' )
	method: str = first[0]
	remainder: str = first[2]
	if remainder.byte_len() == 0:
		return Result.Err( HTTPError.MalformedRequestLine( None ))
	second: tuple[str,str,str] = remainder.partition( ' ' )
	target: str = second[0]
	version: str = second[2]
	if target.byte_len() == 0 or version.byte_len() == 0:
		return Result.Err( HTTPError.MalformedRequestLine( None ))
	return Result.Ok( ( method, target, version ))


# ---------------------------------------------------------------------------
# _RequestBuffer - shaped like http.client's own _GrowableBuffer, but
# generic over io.Reader (calls .read(), matching TcpConnection) instead of
# .recv() (matching Socket directly - see this module's own header comment
# for why a server can't just reuse that one). Adds compact(): a client only
# ever reads one response per connection, but a keep-alive server must
# retain bytes past one consumed request (a pipelined next request already
# sitting in the same buffer) into the next read loop iteration. Not worth
# factoring the shared parts out into a third module for a POC - the two
# buffers differ in which transport method they call, and duplicating ~80
# lines matches this codebase's own stated preference over a premature
# shared abstraction.
# ---------------------------------------------------------------------------

_CR: u8 = 13
_LF: u8 = 10

class _RequestBuffer:
	__data: Ptr[u8]
	__len:  usize
	__cap:  usize

	def __init__( self ) -> None:
		self.__cap = 4096
		self.__data = sys.alloc[u8]( self.__cap )
		self.__len = 0

	def __del__( self ) -> None:
		sys.free( self.__data )

	def len( self ) -> usize:
		return self.__len

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

	def fill_from[T: io.Reader]( self, transport: T ) -> Result[usize, HTTPError]:
		''' one read() call, appended to the buffer. Returns the number of
		bytes read - 0 means the peer closed the connection. '''
		self._grow( 4096 )
		with compiler.wrap_arithmetic:
			dest: Ptr[u8] = self.__data + self.__len
			room: usize = self.__cap - self.__len
		match transport.read( dest, room ):
			case Result.Ok( n ):
				with compiler.wrap_arithmetic:
					self.__len += n
				return Result.Ok( n )
			case Result.Err( _ ):
				return Result.Err( HTTPError.Other( None ))

	def find_double_crlf( self, start: usize ) -> Result[usize, IndexError]:
		''' offset of the first "\\r\\n\\r\\n" at or after start, or Err if
		not (yet) present - the header/body boundary. '''
		if self.__len < 4:
			return Result.Err( IndexError() )
		with compiler.panic_arithmetic( 'bounded by len, cannot overflow' ):
			last_start: usize = self.__len - 4
		i: usize = start
		with compiler.panic_arithmetic( 'bounded by len, cannot overflow' ):
			while i <= last_start:
				if self.__data[i] == _CR and self.__data[i+1] == _LF and self.__data[i+2] == _CR and self.__data[i+3] == _LF:
					return Result.Ok( i )
				i += 1
		return Result.Err( IndexError() )

	def slice_bytes( self, start: usize, end: usize ) -> bytes:
		with compiler.panic_arithmetic( 'bounded by len, cannot overflow' ):
			n: usize = end - start
			src: ConstPtr[u8] = self.__data + start
		out: bytearray = bytearray( n )
		sys.memcpy( out.get_ptr(), src, n )
		return bytes.from_bytearray( move( out ))

	def slice_str( self, start: usize, end: usize ) -> Result[str, CodecError]:
		with compiler.panic_arithmetic( 'bounded by len, cannot overflow' ):
			n: usize = end - start
			src: ConstPtr[u8] = self.__data + start
			buf_size: usize = n + 1
		out: bytearray = bytearray( buf_size )
		sys.memcpy( out.get_ptr(), src, n )
		return str.from_cstr( move( out ))

	def compact( self, consumed: usize ) -> None:
		''' discards [0, consumed) and shifts any trailing bytes (a
		pipelined next request already sitting in this same buffer) down to
		offset 0, so the next _read_one_request call starts clean - see
		this class's own header comment for why a client-side buffer never
		needs this. memmove, not memcpy - [consumed, len) and [0, len -
		consumed) can genuinely overlap. '''
		if consumed == 0:
			return
		with compiler.panic_arithmetic( 'bounded by len, cannot overflow' ):
			remaining: usize = self.__len - consumed
			if remaining > 0:
				sys.memmove( self.__data, self.__data + consumed, remaining )
		self.__len = remaining


# ---------------------------------------------------------------------------
# small helpers - each written as a small returning function (rather than a
# bare-declared local assigned from inside a match arm) per the compiler's
# own documented "declare `x: T` with no initializer, then assign from a
# match arm" crash - see memory bare_declare_match_arm_crash.md.
# ---------------------------------------------------------------------------

_ASCII_ZERO: u8 = 0x30
_ASCII_NINE: u8 = 0x39

def _usize_from_str( s: str ) -> Result[usize, HTTPError]:
	if s.byte_len() == 0:
		return Result.Err( HTTPError.MalformedHeader( None ))
	cstr: ConstPtr[u8] = s.get_cstr()
	n: usize = s.byte_len()
	value: usize = 0
	i: usize = 0
	with compiler.panic_arithmetic( 'a Content-Length within usize range' ):
		while i < n:
			ch: u8 = cstr[i]
			if ch < _ASCII_ZERO or ch > _ASCII_NINE:
				return Result.Err( HTTPError.MalformedHeader( None ))
			value = value * 10 + usize( ch - _ASCII_ZERO )
			i += 1
	return Result.Ok( value )

def _decode_utf8_or_err( buf: _RequestBuffer, start: usize, end: usize ) -> Result[str, HTTPError]:
	''' unlike http.client's own getresponse() (which .unwrap()s the
	equivalent decode - a server it already trusts), a request head comes
	straight from the network and a malformed one shouldn't be able to
	abort the whole process. '''
	match buf.slice_str( start, end ):
		case Result.Ok( s ):
			return Result.Ok( s )
		case Result.Err( _ ):
			return Result.Err( HTTPError.MalformedRequestLine( None ))

def _parse_headers_or_err( raw_block: str ) -> Result[HTTPHeaders, HTTPError]:
	''' adapts http.client's own parse_headers() - which returns ITS OWN
	HTTPError, a distinct nominal type from this module's - onto this
	module's HTTPError. '''
	match parse_headers( raw_block ):
		case Result.Ok( h ):
			return Result.Ok( h )
		case Result.Err( _ ):
			return Result.Err( HTTPError.MalformedHeader( None ))

def _parse_query_or_empty( qs: str ) -> list[tuple[str,str]]:
	''' malformed percent-encoding in a query string collapses to "no
	query params" rather than failing the whole request - a pragmatic POC
	choice, not a claim that it's the only reasonable one. '''
	match parse_qsl( qs ):
		case Result.Ok( pairs ):
			return pairs
		case Result.Err( _ ):
			return list[tuple[str,str]]()

def _read_content_length_body( conn: tcp.TcpConnection, buf: _RequestBuffer, body_start: usize, content_length: usize ) -> Result[bytes, HTTPError]:
	''' mirrors http.client's own _read_content_length_body exactly, just
	generic over _RequestBuffer/TcpConnection instead of _GrowableBuffer/T. '''
	with compiler.panic_arithmetic( 'a real Content-Length body fits well within usize' ):
		while True:
			with compiler.panic_arithmetic( 'bounded by buf.len(), cannot overflow' ):
				have: usize = buf.len() - body_start
			if have >= content_length:
				break
			n: usize = buf.fill_from( conn ).or_return()
			if n == 0:
				return Result.Err( HTTPError.UnexpectedEOF( None ))
	with compiler.wrap_arithmetic:
		body_end: usize = body_start + content_length
	return Result.Ok( buf.slice_bytes( body_start, body_end ))


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class Request:
	method:  str
	target:  str
	path:    str
	query:   list[tuple[str,str]]
	version: str
	headers: HTTPHeaders
	body:    bytes

	def __init__( self, method: str, target: str, path: str, query: list[tuple[str,str]], version: str, headers: HTTPHeaders, body: bytes ) -> None:
		self.method = method
		self.target = target
		self.path = path
		self.query = query
		self.version = version
		self.headers = headers
		self.body = body


def _read_one_request( conn: tcp.TcpConnection, buf: _RequestBuffer ) -> Result[Request, HTTPError]:
	''' reads and parses exactly one request off conn, buffering through
	buf (which may already hold a pipelined next request's bytes left over
	from a prior call - see _RequestBuffer.compact). '''
	first_fill: bool = buf.len() == 0
	header_end: usize = 0
	with compiler.panic_arithmetic( 'a real HTTP request header block fits well within usize' ):
		while True:
			match buf.find_double_crlf( 0 ):
				case Result.Ok( offset ):
					header_end = offset
					break
				case Result.Err( _ ):
					pass
			n: usize = buf.fill_from( conn ).or_return()
			if n == 0:
				if first_fill:
					return Result.Err( HTTPError.ConnectionClosed( None ))
				return Result.Err( HTTPError.UnexpectedEOF( None ))
			first_fill = False

	head_str: str = _decode_utf8_or_err( buf, 0, header_end ).or_return()
	with compiler.wrap_arithmetic:
		body_start: usize = header_end + 4

	first_line_parts: tuple[str,str,str] = head_str.partition( '\r\n' )
	request_line: str = first_line_parts[0]
	rest_headers: str = first_line_parts[2]

	parsed: tuple[str,str,str] = parse_request_line( request_line ).or_return()
	method: str = parsed[0]
	target: str = parsed[1]
	version: str = parsed[2]

	headers: HTTPHeaders = _parse_headers_or_err( rest_headers ).or_return()

	content_length_str: str|None = headers.get( 'Content-Length' )
	body: bytes = bytes.from_bytearray( move( bytearray( 0 )))
	consumed: usize = body_start
	if content_length_str is not None:
		cl: str = content_length_str
		content_length: usize = _usize_from_str( cl ).or_return()
		body = _read_content_length_body( conn, buf, body_start, content_length ).or_return()
		with compiler.wrap_arithmetic:
			consumed = body_start + content_length

	buf.compact( consumed )

	split: SplitResult = urlsplit( target )
	query: list[tuple[str,str]] = _parse_query_or_empty( split.query )

	return Result.Ok( Request( method, target, split.path, query, version, headers, body ))


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

class Response:
	status_code: u16
	reason:      str
	headers:     HTTPHeaders
	body:        bytes

	def __init__( self, status_code: u16, reason: str, headers: HTTPHeaders, body: bytes ) -> None:
		self.status_code = status_code
		self.reason = reason
		self.headers = headers
		self.body = body

	@staticmethod
	def text( body: str, status_code: u16 = 200, reason: str = 'OK' ) -> Response:
		encoded: bytes = body.encode().unwrap( 'Response.text: unreachable - str is always valid UTF-8' )
		headers: HTTPHeaders = HTTPHeaders()
		headers.set( 'Content-Type', 'text/plain; charset=utf-8' )
		return Response( status_code, reason, headers, encoded )

	@staticmethod
	def html( body: str, status_code: u16 = 200, reason: str = 'OK' ) -> Response:
		encoded: bytes = body.encode().unwrap( 'Response.html: unreachable - str is always valid UTF-8' )
		headers: HTTPHeaders = HTTPHeaders()
		headers.set( 'Content-Type', 'text/html; charset=utf-8' )
		return Response( status_code, reason, headers, encoded )

	@staticmethod
	def bytes_( body: bytes, content_type: str, status_code: u16 = 200, reason: str = 'OK' ) -> Response:
		headers: HTTPHeaders = HTTPHeaders()
		headers.set( 'Content-Type', content_type )
		return Response( status_code, reason, headers, body )

	@staticmethod
	def json( value: JSONValue, status_code: u16 = 200, reason: str = 'OK' ) -> Result[Response, HTTPError]:
		match dumps( value ):
			case Result.Ok( s ):
				encoded: bytes = s.encode().unwrap( 'Response.json: unreachable - dumps output is always valid UTF-8' )
				headers: HTTPHeaders = HTTPHeaders()
				headers.set( 'Content-Type', 'application/json' )
				return Result.Ok( Response( status_code, reason, headers, encoded ))
			case Result.Err( _ ):
				return Result.Err( HTTPError.Other( None ))

	def _head_str( self ) -> str:
		head: str = 'HTTP/1.1 ' + self.status_code.__str__() + ' ' + self.reason + '\r\n'
		n: usize = self.headers.__len__()
		i: usize = 0
		for i in range( n ):
			name: str = self.headers.name_at( i ).unwrap( 'Response._head_str: index in bounds by construction' )
			value: str = self.headers.value_at( i ).unwrap( 'Response._head_str: index in bounds by construction' )
			head = head + name + ': ' + value + '\r\n'
		return head + 'Content-Length: ' + self.body.__len__().__str__() + '\r\n\r\n'

	def write_to[T: io.Writer]( self, dst: T ) -> Result[None, HTTPError]:
		head_bytes: bytes = self._head_str().encode().unwrap( 'Response.write_to: unreachable - head is pure ASCII' )
		match io.write_all( dst, head_bytes.get_const_ptr(), head_bytes.__len__() ):
			case Result.Ok( _ ):
				pass
			case Result.Err( _ ):
				return Result.Err( HTTPError.Other( None ))
		match io.write_all( dst, self.body.get_const_ptr(), self.body.__len__() ):
			case Result.Ok( _ ):
				return Result.Ok( None )
			case Result.Err( _ ):
				return Result.Err( HTTPError.Other( None ))


# ---------------------------------------------------------------------------
# per-connection keep-alive loop + accept loop
# ---------------------------------------------------------------------------

def _wants_close( request: Request, response: Response ) -> bool:
	''' HTTP/1.1 defaults to keep-alive - close only when the request or
	the response explicitly says "Connection: close", or the request isn't
	HTTP/1.1 at all (no HTTP/1.0 keep-alive support in this v1). '''
	req_conn: str|None = request.headers.get( 'Connection' )
	if req_conn is not None:
		rc: str = req_conn
		if rc.lower() == 'close':
			return True
	resp_conn: str|None = response.headers.get( 'Connection' )
	if resp_conn is not None:
		sc: str = resp_conn
		if sc.lower() == 'close':
			return True
	return request.version != 'HTTP/1.1'

def _write_best_effort_error( conn: tcp.TcpConnection, status_code: u16, reason: str ) -> None:
	''' used only when a request couldn't even be parsed - best-effort,
	since the connection is being closed regardless of whether this write
	itself succeeds. '''
	response: Response = Response.text( reason + '\n', status_code, reason )
	match response.write_to( conn ):
		case Result.Ok( _ ):
			pass
		case Result.Err( _ ):
			pass

def _handle_connection( conn: tcp.TcpConnection, handler: Closure[[Request], Response] ) -> None:
	buf: _RequestBuffer = _RequestBuffer()
	while True:
		match _read_one_request( conn, buf ):
			case Result.Ok( request ):
				response: Response = handler( request )
				close_after: bool = _wants_close( request, response )
				if close_after:
					response.headers.set( 'Connection', 'close' )
				write_result: Result[None, HTTPError] = response.write_to( conn )
				match write_result:
					case Result.Ok( _ ):
						pass
					case Result.Err( _ ):
						close_after = True
				if close_after:
					conn.close()
					return
			case Result.Err( HTTPError.ConnectionClosed( _ )):
				conn.close()
				return
			case Result.Err( _ ):
				_write_best_effort_error( conn, 400, 'Bad Request' )
				conn.close()
				return

class _AcceptLoop:
	__listener: tcp.TcpListener
	__handler:  Closure[[Request], Response]
	__reactor:  reactor.Reactor

	def __init__( self, listener: tcp.TcpListener, handler: Closure[[Request], Response], r: reactor.Reactor ) -> None:
		self.__listener = listener
		self.__handler = handler
		self.__reactor = r

	def run( self ) -> None:
		while True:
			match self.__listener.accept():
				case Result.Ok( conn ):
					h: Closure[[Request], Response] = self.__handler
					self.__reactor.spawn( lambda: _handle_connection( conn, h ))
				case Result.Err( _ ):
					return

def serve( listener: tcp.TcpListener, handler: Closure[[Request], Response], r: reactor.Reactor ) -> None:
	''' spawns the accept loop onto r - each connection, once accepted, is
	handed to its own fiber (running handler once per request on that
	connection, HTTP/1.1 keep-alive). Doesn't block: call r.run() once
	every serve()/spawn() call you want driven has been made, mirroring
	tcp_test.py's own spawn-then-run() shape. '''
	loop: _AcceptLoop = _AcceptLoop( listener, handler, r )
	r.spawn( loop.run )
