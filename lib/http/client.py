'''
HTTP wire-format primitives, plus a low-level one-connection-at-a-time client
(HTTPConnection/Response) built on lib/socket.py - see PLAN_HTTP_CLIENT.md.

HTTPError       - error type shared by every function/class below
HTTPHeaders     - ordered, case-insensitive multimap for request/response headers
parse_status_line  - "HTTP/1.1 200 OK" -> (version, status_code, reason)
parse_header_line  - "Name: value" -> (name, value)
parse_headers      - a raw CRLF-joined header block -> HTTPHeaders
base64_encode      - standard (padded) base64 encoding of bytes, for auth=
decode_chunked     - decodes an already-fully-buffered chunked-transfer body
HTTPConnection     - connect/request/getresponse/close over one TCP connection
Response           - status_code/reason/headers/content of a received response
Session            - cookie jar + redirect-following requests.Session-style layer
get/post/put/patch/delete/head/options/request - module-level Session convenience

host may be a real hostname now - lib/socket.py's Socket.connect() resolves it
via getaddrinfo internally (commit f794edb). https:// URLs work too, via
lib/ssl.py (PLAN_SSL.md) - HTTPConnection.connect()/HTTPSConnection.connect()
pick the transport, and Session.request() dispatches on the parsed URL's own
scheme. URL parsing/query encoding is built on lib/urllib/parse.py
(urlsplit/urlencode/parse_qsl/urljoin) rather than hand-rolled here - redirect
Location headers may now be relative, resolved against the request URL via
urljoin(). json= (on post/put/patch) and Response.json() are built on
lib/json.py.
'''

import sys
import compiler
import base64
import ssl
from socket import Socket
from urllib.parse import urlencode, parse_qsl, urlsplit, urljoin, SplitResult
from json import loads, dumps, JSONValue

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

# @union, not @enum: a CEnum member referenced bare as a real runtime value
# (exactly what every Result.Err( HTTPError.X ) below needs) is a documented
# lowering.py gap - see lib/builtins/__int.py's own IntError, which hit this
# directly and adopted the same @union / None-payload-variant / X(None)
# construction shape used here.
@union
class HTTPError:
	MalformedStatusLine: None
	MalformedHeader: None
	UnexpectedEOF: None
	ChunkSizeInvalid: None
	InvalidURL: None
	TooManyRedirects: None
	BadStatus: None
	NameResolutionFailed: None
	InvalidJSON: None
	TLSError: None
	Other: None

# ---------------------------------------------------------------------------
# HTTPHeaders - ordered, case-insensitive multimap
# ---------------------------------------------------------------------------

def _header_name_eq( a: str, b: str ) -> bool:
	return a.lower() == b.lower()

class HTTPHeaders( Sized ):
	# list[tuple[str,str]] rather than a dedicated (name,value) class - this
	# used to fail discovery with "name 'tuple' is not defined" (a nested
	# tuple[...] as an EXPLICIT generic type argument to another generic's
	# own constructor call wasn't resolved the same way an annotation's
	# nested tuple[...] already was), fixed on master in 740b54d ("Fix
	# tuple[...] (and move/copy/Callable/Closure/Iterator/Generator) as a
	# nested explicit type argument to a generic constructor call").
	__entries: list[tuple[str,str]]

	def __init__( self ) -> None:
		self.__entries = list[tuple[str,str]]()

	def __len__( self ) -> usize:
		return self.__entries.__len__()

	def add( self, name: str, value: str ) -> None:
		''' appends a new entry, keeping any existing entry with the same
		(case-insensitive) name - matches HTTP's own "repeated headers are
		combined, not overwritten" semantics for headers like Set-Cookie. '''
		self.__entries.append( ( name, value ))

	def set( self, name: str, value: str ) -> None:
		''' replaces every existing entry with a matching (case-insensitive)
		name with a single new entry at the first matching position, or
		appends if name wasn't present - matches Python requests' dict-like
		header-assignment semantics. list[T] has no in-place remove/insert
		yet (see PLAN_LIST_T.md), so this rebuilds a fresh list rather than
		mutating __entries directly. '''
		rebuilt: list[tuple[str,str]] = list[tuple[str,str]]()
		replaced: bool = False
		n: usize = self.__entries.__len__()
		i: usize = 0
		for i in range( n ):
			entry: tuple[str,str] = self.__entries.__getitem__( i ).unwrap( 'HTTPHeaders.set: index in bounds by construction' )
			if _header_name_eq( entry[0], name ):
				if not replaced:
					rebuilt.append( ( name, value ))
					replaced = True
			else:
				rebuilt.append( entry )
		if not replaced:
			rebuilt.append( ( name, value ))
		self.__entries = rebuilt

	def get( self, name: str ) -> str|None:
		''' the first entry matching name (case-insensitive), or None. '''
		n: usize = self.__entries.__len__()
		i: usize = 0
		for i in range( n ):
			entry: tuple[str,str] = self.__entries.__getitem__( i ).unwrap( 'HTTPHeaders.get: index in bounds by construction' )
			if _header_name_eq( entry[0], name ):
				return entry[1]
		return None

	def get_all( self, name: str ) -> list[str]:
		''' every value for name (case-insensitive), in wire/insertion order. '''
		result: list[str] = list[str]()
		n: usize = self.__entries.__len__()
		i: usize = 0
		for i in range( n ):
			entry: tuple[str,str] = self.__entries.__getitem__( i ).unwrap( 'HTTPHeaders.get_all: index in bounds by construction' )
			if _header_name_eq( entry[0], name ):
				result.append( entry[1] )
		return result

	def name_at( self, index: usize ) -> Result[str, IndexError]:
		entry: tuple[str,str] = self.__entries.__getitem__( index ).or_return()
		return Result.Ok( entry[0] )

	def value_at( self, index: usize ) -> Result[str, IndexError]:
		entry: tuple[str,str] = self.__entries.__getitem__( index ).or_return()
		return Result.Ok( entry[1] )

# ---------------------------------------------------------------------------
# status-line / header-line parsing
# ---------------------------------------------------------------------------

def parse_status_code( s: str ) -> Result[u16, HTTPError]:
	''' parses exactly 3 ASCII digits (the HTTP status-code grammar - no
	leading/trailing junk, no sign). '''
	if s.byte_len() != 3:
		return Result.Err( HTTPError.MalformedStatusLine( None ))
	cstr: ConstPtr[u8] = s.get_cstr()
	value: u16 = 0
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < 3:
			ch: u8 = cstr[i]
			if ch < 0x30 or ch > 0x39: # '0'-'9'
				return Result.Err( HTTPError.MalformedStatusLine( None ))
			value = value * 10 + u16( ch - 0x30 )
			i += 1
	return Result.Ok( value )

def parse_status_line( line: str ) -> Result[tuple[str,u16,str], HTTPError]:
	''' "HTTP/1.1 200 OK" -> (version, status_code, reason). Built entirely
	on str.partition() (PLAN_TUPLE.md) rather than manual byte scanning. '''
	first: tuple[str,str,str] = line.partition( ' ' )
	version: str = first[0]
	remainder: str = first[2]
	if remainder.byte_len() == 0:
		return Result.Err( HTTPError.MalformedStatusLine( None ))
	second: tuple[str,str,str] = remainder.partition( ' ' )
	code_str: str = second[0]
	reason: str = second[2]
	status: u16 = parse_status_code( code_str ).or_return()
	return Result.Ok( ( version, status, reason ))

def parse_header_line( line: str ) -> Result[tuple[str,str], HTTPError]:
	''' "Name: value" -> (name, value), with leading/trailing whitespace
	stripped from value (RFC 7230 OWS around the field-value). A line with
	no ':' is malformed. '''
	parts: tuple[str,str,str] = line.partition( ':' )
	sep: str = parts[1]
	if sep.byte_len() == 0:
		return Result.Err( HTTPError.MalformedHeader( None ))
	name: str = parts[0]
	value: str = parts[2].strip()
	return Result.Ok( ( name, value ))

def parse_headers( raw_block: str ) -> Result[HTTPHeaders, HTTPError]:
	''' parses a raw, CRLF-joined block of header lines (no leading status
	line, no trailing blank line - callers already split those off) into an
	HTTPHeaders. Blank lines within the block are skipped rather than
	rejected, tolerating a trailing "\\r\\n" on the block itself. '''
	headers: HTTPHeaders = HTTPHeaders()
	lines: list[str] = raw_block.split( '\r\n' )
	n: usize = lines.__len__()
	i: usize = 0
	for i in range( n ):
		line: str = lines.__getitem__( i ).unwrap( 'parse_headers: index in bounds by construction' )
		if line.byte_len() == 0:
			continue
		parsed: tuple[str,str] = parse_header_line( line ).or_return()
		headers.add( parsed[0], parsed[1] )
	return Result.Ok( headers )

# ---------------------------------------------------------------------------
# base64 (encode only - v1 only needs it for auth= -> Basic auth header)
# ---------------------------------------------------------------------------

def base64_encode( data: bytes ) -> str:
	''' str-returning wrapper around lib/base64.py's own b64encode() (RFC
	4648 base64/urlsafe/base16 encode+decode, added after this file's own
	hand-rolled version - see PLAN_HTTP_CLIENT.md) - base64.b64encode()
	itself returns bytes, matching Python's own base64 module, while an
	HTTP header value needs a str. '''
	return base64.b64encode( data ).decode().unwrap( 'base64_encode: invalid UTF-8 (unreachable - output is pure ASCII)' )

# ---------------------------------------------------------------------------
# chunked transfer-encoding decode
# ---------------------------------------------------------------------------

_CR: u8 = 0x0D
_LF: u8 = 0x0A
_SEMICOLON: u8 = 0x3B

def _find_crlf( data: ConstPtr[u8], start: usize, length: usize ) -> Result[usize, HTTPError]:
	''' byte offset of the next CRLF in data[start:length), or Err if none -
	used to find each chunk-size line's own terminator. '''
	i: usize = start
	with compiler.panic_arithmetic( 'bounded by length, cannot overflow' ):
		while i + 1 < length:
			if data[i] == _CR and data[i+1] == _LF:
				return Result.Ok( i )
			i += 1
	return Result.Err( HTTPError.UnexpectedEOF( None ))

def _hex_digit_value( b: u8 ) -> Result[u8, HTTPError]:
	if b >= 0x30 and b <= 0x39: # '0'-'9'
		with compiler.wrap_arithmetic:
			return Result.Ok( b - 0x30 )
	if b >= 0x41 and b <= 0x46: # 'A'-'F'
		with compiler.wrap_arithmetic:
			return Result.Ok( b - 0x41 + 10 )
	if b >= 0x61 and b <= 0x66: # 'a'-'f'
		with compiler.wrap_arithmetic:
			return Result.Ok( b - 0x61 + 10 )
	return Result.Err( HTTPError.ChunkSizeInvalid( None ))

def _parse_chunk_size( src: ConstPtr[u8], start: usize, end: usize ) -> Result[usize, HTTPError]:
	''' hex chunk-size, stopping at ';' (a chunk-extension, ignored - not
	needed by a client that only wants the decoded body). '''
	value: usize = 0
	digits: usize = 0
	i: usize = start
	with compiler.panic_arithmetic( 'bounded by end, cannot overflow' ):
		while i < end:
			b: u8 = src[i]
			if b == _SEMICOLON:
				break
			digit: u8 = _hex_digit_value( b ).or_return()
			with compiler.wrap_arithmetic:
				value = value * 16 + usize( digit )
			digits += 1
			i += 1
	if digits == 0:
		return Result.Err( HTTPError.ChunkSizeInvalid( None ))
	return Result.Ok( value )

def decode_chunked( data: bytes ) -> Result[bytes, HTTPError]:
	''' decodes an HTTP/1.1 chunked-transfer-encoded body that has already
	been fully read into memory. Trailing headers after the terminating
	0-length chunk are accepted but discarded - not needed by a client that
	only wants the decoded body (see PLAN_HTTP_CLIENT.md). Two passes (size,
	then fill), the same shape every other stdlib encode/decode in this
	codebase uses (see lib/codecs/latin1.py etc). '''
	src: ConstPtr[u8] = data.get_const_ptr()
	length: usize = data.__len__()

	total: usize = 0
	pos: usize = 0
	line_end: usize = 0
	size: usize = 0
	data_start: usize = 0
	data_end: usize = 0
	with compiler.panic_arithmetic( 'bounded by length, cannot overflow' ):
		while True:
			line_end = _find_crlf( src, pos, length ).or_return()
			size = _parse_chunk_size( src, pos, line_end ).or_return()
			data_start = line_end + 2
			if size == 0:
				break
			data_end = data_start + size
			if data_end + 2 > length:
				return Result.Err( HTTPError.UnexpectedEOF( None ))
			if src[data_end] != _CR or src[data_end+1] != _LF:
				return Result.Err( HTTPError.ChunkSizeInvalid( None ))
			total += size
			pos = data_end + 2

	out: bytearray = bytearray( total )
	out_ptr: Ptr[u8] = out.get_ptr()
	out_off: usize = 0
	pos = 0
	with compiler.panic_arithmetic( 'bounded by length, cannot overflow - re-scan of already-validated input' ):
		while True:
			line_end = _find_crlf( src, pos, length ).unwrap( 'decode_chunked: re-scan after first-pass validation' )
			size = _parse_chunk_size( src, pos, line_end ).unwrap( 'decode_chunked: re-scan after first-pass validation' )
			data_start = line_end + 2
			if size == 0:
				break
			sys.memcpy( out_ptr + out_off, src + data_start, size )
			out_off += size
			pos = data_start + size + 2

	return Result.Ok( bytes.from_bytearray( move( out )))

# ---------------------------------------------------------------------------
# small integer <-> str helper - int (the boxed arbitrary-precision type)
# only converts from i32 (lib/builtins/__int.py), not usize, and a
# Content-Length can legitimately need the full usize range - spelled out
# directly rather than routed through int, matching this codebase's own
# established "ASCII digits by hand" idiom (see lib/builtins/__int.py's own
# _ASCII_ZERO-based from_str/digit-count code).
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

# ---------------------------------------------------------------------------
# transport - a plain Socket or a TLS-wrapped ssl.SSLSocket, so
# Connection[T]/_GrowableBuffer/the request-sending helpers below have one
# thing to call send()/recv()/close() on regardless of http:// vs https://.
#
# A GENERIC type parameter T (monomorphized separately for T=Socket and
# T=ssl.SSLSocket), NOT a @union - this file originally used a
# `@union class _Transport: Plain: Socket; Secure: ssl.SSLSocket`, which
# worked but had a real cost a generic doesn't: every shared method call
# (send/recv/close, in _send_all/_GrowableBuffer.fill_from/_read_*_body/
# Connection itself) needed its own runtime tag-dispatch `match`, sprinkled
# through this whole file instead of a direct `transport.send(...)`-style
# call - a generic function/method calling a named method directly on a
# bare type parameter (no shared base class/interface needed between Socket
# and ssl.SSLSocket) monomorphizes cleanly per instantiation, confirmed via
# a real compile spike before committing to this design. Each
# Connection[T] instance is also sized exactly for whichever T it holds,
# not the union's own tag + larger-of-the-two-payloads layout.
#
# NOT a binary-size win, despite the name "monomorphization" suggesting one:
# tested directly (compiled .exe size + extern_libs, HTTPConnection-only vs
# HTTPSConnection-only program) and both came out byte-identical, both
# linking secur32 (Windows TLS) either way. Importing http.client at all
# schedules the WHOLE MODULE for compilation in this compiler's model, not
# just the specific names a program actually references - HTTPSConnection
# sits in this same file, so it's compiled in regardless of whether a given
# program's main() ever calls it, generic transport or not. Splitting
# HTTPSConnection into its own separately-imported module would be a
# genuine way to make ssl.py opt-in; this generic-vs-union change alone
# isn't that, and doesn't claim to be.
# ---------------------------------------------------------------------------

def _connect_tls_or_http_err( host: str, port: u16, verify: bool = True ) -> Result[ssl.SSLSocket, HTTPError]:
	''' TCP-connects (via _connect_or_http_err below), then TLS-wraps via
	lib/ssl.py. A handshake failure (including a certificate problem -
	lib/ssl.py's create_default_context() already turns on peer
	verification, matching this file's own "secure by default" posture
	elsewhere) collapses to HTTPError.TLSError - a caller wanting the
	specific ssl.SSLError reason would need to use lib/ssl.py directly.
	verify=False switches to create_unverified_context() (mirrors requests'
	verify=False) - for a self-signed/dev server only, never a real
	endpoint. '''
	sock: Socket = _connect_or_http_err( host, port ).or_return()
	ctx_result: Result[ssl.SSLContext, ssl.SSLError] = ssl.SSLContext.create_default_context() if verify else ssl.SSLContext.create_unverified_context()
	match ctx_result:
		case Result.Ok( ctx ):
			match ssl.SSLSocket.wrap_socket( ctx, sock, host ):
				case Result.Ok( tls ):
					return Result.Ok( tls )
				case Result.Err( _ ):
					return Result.Err( HTTPError.TLSError( None ))
		case Result.Err( _ ):
			return Result.Err( HTTPError.TLSError( None ))

def _do_request_response[T]( transport: T, method: str, full_path: str, host: str, headers: HTTPHeaders, body: bytes|None ) -> Result[Response, HTTPError]:
	''' request+getresponse+close over an ALREADY-CONNECTED transport of
	generic type T - shared by Session.request() via
	_perform_request_for_scheme below (the one place a runtime scheme check
	picks which T to instantiate this with; everything downstream of that
	one branch, including this function, is fully generic/dispatch-free). '''
	conn: Connection[T] = Connection[T]._from_transport( transport, host, 0 )
	conn.request( method, full_path, headers, body ).or_return()
	response: Response = conn.getresponse().or_return()
	conn.close()
	return Result.Ok( response )

def _perform_request_for_scheme( scheme: str, host: str, port: u16, method: str, full_path: str, headers: HTTPHeaders, body: bytes|None, verify: bool = True ) -> Result[Response, HTTPError]:
	''' the ONE runtime branch point in this whole file for choosing plain
	vs TLS. Session.request() doesn't know the scheme until it's parsed the
	URL, so SOME runtime decision is unavoidable here - but it's confined to
	exactly this one if/else, not sprinkled through every layer the way the
	old _Transport union's tag-dispatch was. '''
	if scheme == 'https':
		tls: ssl.SSLSocket = _connect_tls_or_http_err( host, port, verify ).or_return()
		return _do_request_response( tls, method, full_path, host, headers, body )
	sock: Socket = _connect_or_http_err( host, port ).or_return()
	return _do_request_response( sock, method, full_path, host, headers, body )

# ---------------------------------------------------------------------------
# _GrowableBuffer - accumulates bytes read off a transport across multiple
# recv() calls. bytearray() itself is fixed-size at construction (see lib/
# builtins/__init__.py) with no append/extend, so this is a small hand-
# rolled doubling buffer, the same "count/allocate-exact/fill" discipline
# used throughout this codebase, just amortized across repeated growth
# instead of computed once up front - the total size isn't known ahead of
# time here (it depends on how many recv() calls a response takes).
# ---------------------------------------------------------------------------

class _GrowableBuffer:
	__data: Ptr[u8]
	__len: usize
	__cap: usize

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

	def fill_from[T]( self, transport: T ) -> Result[usize, HTTPError]:
		''' one recv() call, appended to the buffer. Returns the number of
		bytes read - 0 means the peer closed the connection. A generic
		method (monomorphized per T - Socket or ssl.SSLSocket) calling
		transport.recv(...) directly, not a runtime-dispatched union - see
		this file's own "transport" header comment above for why. '''
		self._grow( 4096 )
		with compiler.wrap_arithmetic:
			dest: Ptr[u8] = self.__data + self.__len
			room: usize = self.__cap - self.__len
		match transport.recv( dest, room ):
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
		# NOT str.from_cstr(ptr, size) - that overload requires the SOURCE
		# buffer to already be null-terminated at size-1 (confirmed directly:
		# it copies `size` bytes as-is and rejects a nonzero last byte), and
		# this buffer's raw network bytes have no such terminator anywhere.
		# bytearray(n+1) is zero-filled by construction (see lib/builtins/
		# __init__.py's bytearray.__init__) and never written at its own
		# last index below, so it's null-terminated by construction instead -
		# same approach quote()/base64_encode above already use.
		with compiler.panic_arithmetic( 'bounded by len, cannot overflow' ):
			n: usize = end - start
			src: ConstPtr[u8] = self.__data + start
			buf_size: usize = n + 1
		out: bytearray = bytearray( buf_size )
		sys.memcpy( out.get_ptr(), src, n )
		return str.from_cstr( move( out ))

# ---------------------------------------------------------------------------
# request building / sending
# ---------------------------------------------------------------------------

def _build_request_head( method: str, path: str, host: str, headers: HTTPHeaders|None, body: bytes|None ) -> str:
	# request_head, not head - this module ALSO has a module-level head()
	# verb helper, and this language has no local-shadows-outer-scope
	# semantics (see _existing_local_or_none's own comment)
	request_head: str = method + ' ' + path + ' HTTP/1.1\r\n' + 'Host: ' + host + '\r\n'
	if headers is not None:
		n: usize = headers.__len__()
		i: usize = 0
		for i in range( n ):
			name: str = headers.name_at( i ).unwrap( '_build_request_head: index in bounds by construction' )
			value: str = headers.value_at( i ).unwrap( '_build_request_head: index in bounds by construction' )
			request_head = request_head + name + ': ' + value + '\r\n'
	if body is not None:
		request_head = request_head + 'Content-Length: ' + body.__len__().__str__() + '\r\n'
	return request_head + '\r\n'

# Every socket-facing call in this file is funneled through one of the
# _*_or_http_err helpers below rather than propagated as a bare OSError via
# .or_return()/Result.Err(e). This used to be a required workaround: widening
# a bare @union error type (HTTPError) - or a value of it, staged through an
# explicitly-typed local, or via .or_return() - into a WIDER union return
# type (OSError|HTTPError) was broken in at least three different ways here
# (ambiguous generic inference on Result.Err(e); "expected
# OSError|HTTPError, got HTTPError" on the staging assignment itself). That
# compiler bug is now FIXED (type_resolver._atomic_leaves, lowering.py's
# _coerce_or_check_operand, emitter_c.py's _emit_widen_error - see
# union_widening_test.py), so propagating a real OSError|HTTPError here is
# an option again, not a compile error. Kept as-is anyway: collapsing every
# Socket-facing OSError into HTTPError.Other() right at the call site is
# still arguably better API design on its own merits - HTTPConnection's own
# public error type stays a single, simple HTTPError instead of leaking
# lib/socket.py's OSError as part of its API - not just a workaround anymore.

def _connect_or_http_err( host: str, port: u16 ) -> Result[Socket, HTTPError]:
	match Socket.tcp():
		case Result.Ok( sock ):
			match sock.connect( host, port ):
				case Result.Ok( _ ):
					return Result.Ok( sock )
				case Result.Err( os_err ):
					# distinguish "couldn't even resolve the hostname" from
					# every other connect() failure (refused, reset, timed
					# out, ...) - the one OSError variant a caller is likely
					# to want to handle differently (e.g. retry vs. give up
					# immediately on a typo'd hostname). Every other OSError
					# still collapses to Other() - see this function's own
					# header comment on why that stays deliberate, not a gap.
					if os_err == OSError.NameResolutionFailed:
						return Result.Err( HTTPError.NameResolutionFailed( None ))
					return Result.Err( HTTPError.Other( None ))
		case Result.Err( _ ):
			return Result.Err( HTTPError.Other( None ))

def _send_all[T]( transport: T, ptr: ConstPtr[u8], length: usize ) -> Result[None, HTTPError]:
	sent: usize = 0
	with compiler.panic_arithmetic( 'bounded by length, cannot overflow' ):
		while sent < length:
			match transport.send( ptr + sent, length - sent ):
				case Result.Ok( n ):
					if n == 0:
						return Result.Err( HTTPError.UnexpectedEOF( None ))
					sent += n
				case Result.Err( _ ):
					return Result.Err( HTTPError.Other( None ))
	return Result.Ok( None )

# ---------------------------------------------------------------------------
# Response - the result of HTTPConnection.getresponse()
# ---------------------------------------------------------------------------

class Response:
	status_code: u16
	reason: str
	headers: HTTPHeaders
	content: bytes
	url: str # the URL actually fetched - the final one, after any redirects

	def __init__( self, status_code: u16, reason: str, headers: HTTPHeaders, content: bytes, url: str ) -> None:
		self.status_code = status_code
		self.reason = reason
		self.headers = headers
		self.content = content
		self.url = url

	def text( self ) -> Result[str, CodecError]:
		return self.content.decode()

	def json( self ) -> Result[JSONValue, HTTPError]:
		''' parses content as JSON (via lib/json.py's loads()) - the body must
		already be valid UTF-8 (content.decode() failing, or the decoded text
		not being valid JSON, both collapse to HTTPError.InvalidJSON). '''
		# capture bound to `decoded`, not `text` - `text` collides with this
		# class's own text() method name (confirmed via a real compile:
		# "'text' is not a variable, cannot assign to it")
		match self.text():
			case Result.Ok( decoded ):
				match loads( decoded ):
					case Result.Ok( value ):
						return Result.Ok( value )
					case Result.Err( _ ):
						return Result.Err( HTTPError.InvalidJSON( None ))
			case Result.Err( _ ):
				return Result.Err( HTTPError.InvalidJSON( None ))

	def ok( self ) -> bool:
		return self.status_code < 400

	def raise_for_status( self ) -> Result[None, HTTPError]:
		if self.status_code >= 400:
			return Result.Err( HTTPError.BadStatus( None ))
		return Result.Ok( None )

# ---------------------------------------------------------------------------
# response body reading - one function per Content-Length/chunked/until-
# close strategy. Split out of getresponse() itself (rather than inlined
# per-branch) so each has a single, straight-line return path - definite-
# assignment tracking for a shared post-if local reassigned from inside a
# nested `while True: ... break` didn't hold up under a real compile.
# ---------------------------------------------------------------------------

def _read_chunked_body[T]( transport: T, buf: _GrowableBuffer, body_start: usize ) -> Result[bytes, HTTPError]:
	with compiler.panic_arithmetic( 'a real chunked body fits well within usize' ):
		while True:
			body_bytes: bytes = buf.slice_bytes( body_start, buf.len() )
			match decode_chunked( body_bytes ):
				case Result.Ok( d ):
					return Result.Ok( d )
				case Result.Err( HTTPError.ChunkSizeInvalid( _ )):
					return Result.Err( HTTPError.ChunkSizeInvalid( None ))
				case Result.Err( _ ):
					pass # UnexpectedEOF - not a full chunked body yet, keep reading
			n: usize = buf.fill_from( transport ).or_return()
			if n == 0:
				return Result.Err( HTTPError.UnexpectedEOF( None ))

def _read_content_length_body[T]( transport: T, buf: _GrowableBuffer, body_start: usize, content_length: usize ) -> Result[bytes, HTTPError]:
	with compiler.panic_arithmetic( 'a real Content-Length body fits well within usize' ):
		while True:
			with compiler.panic_arithmetic( 'bounded by buf.len(), cannot overflow' ):
				have: usize = buf.len() - body_start
			if have >= content_length:
				break
			n: usize = buf.fill_from( transport ).or_return()
			if n == 0:
				return Result.Err( HTTPError.UnexpectedEOF( None ))
	with compiler.wrap_arithmetic:
		body_end: usize = body_start + content_length
	return Result.Ok( buf.slice_bytes( body_start, body_end ))

def _read_until_close_body[T]( transport: T, buf: _GrowableBuffer, body_start: usize ) -> Result[bytes, HTTPError]:
	with compiler.panic_arithmetic( 'a real response body fits well within usize' ):
		while True:
			n: usize = buf.fill_from( transport ).or_return()
			if n == 0:
				break
	return Result.Ok( buf.slice_bytes( body_start, buf.len() ))

# ---------------------------------------------------------------------------
# Connection[T] - one TCP connection, one request/response at a time,
# GENERIC over its own transport type T (Socket for plain HTTP,
# ssl.SSLSocket for HTTPS - see this file's own "transport" header comment
# above for why this is a generic, not the @union this file used to use).
# Mirrors lib/builtins/__File.py's handle shape otherwise: an owned resource
# field (T itself, already RC-managed with its own auto-closing __del__ - no
# Connection.__del__ needed, the field's own teardown cascades), a private
# constructor, ordinary Result-returning methods.
#
# HTTPConnection/HTTPSConnection (below) are thin, NON-generic entry-point
# classes wrapping this - each `connect()` returns a specific instantiation
# (Connection[Socket] / Connection[ssl.SSLSocket]), matching CPython's
# http.client naming without HTTPConnection/HTTPSConnection themselves
# needing to be generic. They can't just be Connection[T] with a generic
# connect() of their own: the plain-TCP and TLS-handshake connect steps
# genuinely differ, and a single generic method can't have a different body
# per instantiation - only the free functions above (_connect_or_http_err /
# _connect_tls_or_http_err) differ per scheme; everything downstream of
# "already have a live transport" is the identical generic code below.
#
# Public (not module-private _Connection) despite HTTPConnection/
# HTTPSConnection being the intended ordinary entry points - this module's
# own white-box test coverage (http_client_test.py) constructs/type-
# annotates it directly to test the shared connection machinery in
# isolation from either scheme's own connect() wrapper.
# ---------------------------------------------------------------------------

class Connection[T]:
	__transport: T
	__host: str
	__port: u16

	def close( self ) -> None:
		self.__transport.close()

	@private
	@staticmethod
	def _from_transport( transport: T, host: str, port: u16 ) -> Connection[T]:
		return Connection.__allocate__( __transport = transport, __host = host, __port = port )

	def request( self, method: str, path: str, headers: HTTPHeaders|None = None, body: bytes|None = None ) -> Result[None, HTTPError]:
		# NOT named `head` - a local variable shadowing a module-level
		# function of the same name (this module has one, http.client.head())
		# spuriously schedules that function (and everything it transitively
		# calls) for compilation, see type_resolver.py's
		# _try_resolve_callable_namespace/ensure_resolved (task pending, see
		# PLAN_HTTP_CLIENT.md). Confirmed independent of generics/unions -
		# a plain name collision bug.
		request_head: str = _build_request_head( method, path, self.__host, headers, body )
		head_bytes: bytes = request_head.encode().unwrap( '_build_request_head: unreachable - pure ASCII output' )
		_send_all( self.__transport, head_bytes.get_const_ptr(), head_bytes.__len__() ).or_return()
		if body is not None:
			content: bytes = body
			_send_all( self.__transport, content.get_const_ptr(), content.__len__() ).or_return()
		return Result.Ok( None )

	def getresponse( self ) -> Result[Response, HTTPError]:
		buf: _GrowableBuffer = _GrowableBuffer()

		# --- read until the status-line+headers block is fully buffered ---
		header_end: usize = 0
		with compiler.panic_arithmetic( 'a real HTTP response header block fits well within usize' ):
			while True:
				found: Result[usize, IndexError] = buf.find_double_crlf( 0 )
				match found:
					case Result.Ok( offset ):
						header_end = offset
						break
					case Result.Err( _ ):
						pass
				n: usize = buf.fill_from( self.__transport ).or_return()
				if n == 0:
					return Result.Err( HTTPError.UnexpectedEOF( None ))

		header_block: str = buf.slice_str( 0, header_end ).unwrap( 'getresponse: invalid UTF-8 in status line/headers' )
		with compiler.wrap_arithmetic:
			body_start: usize = header_end + 4

		first: tuple[str,str,str] = header_block.partition( '\r\n' )
		status_line: str = first[0]
		rest_headers: str = first[2]

		parsed_status: tuple[str,u16,str] = parse_status_line( status_line ).or_return()
		headers: HTTPHeaders = parse_headers( rest_headers ).or_return()

		# --- read the body, per whichever length strategy the headers say ---
		transfer_encoding: str|None = headers.get( 'Transfer-Encoding' )
		is_chunked: bool = False
		if transfer_encoding is not None:
			te: str = transfer_encoding
			is_chunked = te.lower() == 'chunked'

		content: bytes = bytes.from_bytearray( move( bytearray( 0 )))
		if is_chunked:
			content = _read_chunked_body( self.__transport, buf, body_start ).or_return()
		else:
			content_length_str: str|None = headers.get( 'Content-Length' )
			if content_length_str is not None:
				cl: str = content_length_str
				content_length: usize = _usize_from_str( cl ).or_return()
				content = _read_content_length_body( self.__transport, buf, body_start, content_length ).or_return()
			else:
				# no Content-Length, not chunked - read until the peer closes
				content = _read_until_close_body( self.__transport, buf, body_start ).or_return()

		# url left blank here - Connection only knows host/port/path, not
		# the scheme a caller reached it through; Session.request() (the only
		# caller that actually knows the full URL) fills this field in itself
		# right after getresponse() returns.
		return Result.Ok( Response( parsed_status[1], parsed_status[2], headers, content, '' ))

class HTTPConnection:
	''' entry point for plain HTTP - connect() TCP-connects and returns a
	Connection[Socket] already carrying that live transport. '''

	@staticmethod
	def connect( host: str, port: u16 = 80 ) -> Result[Connection[Socket], HTTPError]:
		''' host may be a real hostname - lib/socket.py's own Socket.connect()
		resolves it via getaddrinfo internally. '''
		sock: Socket = _connect_or_http_err( host, port ).or_return()
		return Result.Ok( Connection[Socket]._from_transport( sock, host, port ))

class HTTPSConnection:
	''' entry point for HTTPS - connect() TCP-connects, completes a TLS
	handshake (lib/ssl.py), and returns a Connection[ssl.SSLSocket] already
	carrying that live transport. '''

	@staticmethod
	def connect( host: str, port: u16 = 443, verify: bool = True ) -> Result[Connection[ssl.SSLSocket], HTTPError]:
		tls: ssl.SSLSocket = _connect_tls_or_http_err( host, port, verify ).or_return()
		return Result.Ok( Connection[ssl.SSLSocket]._from_transport( tls, host, port ))

# ---------------------------------------------------------------------------
# URL parsing - http:// and https://, built on lib/urllib/parse.py's
# urlsplit() rather than this file's own hand-rolled scheme/host/port/path
# splitter (see PLAN_HTTP_CLIENT.md for the migration - urlsplit() lands
# host/port splitting and query handling on a real, separately-tested
# general-purpose module instead).
# ---------------------------------------------------------------------------

class ParsedURL:
	scheme: str
	host: str
	port: u16
	path: str  # path only, no query - see _build_request_path for query merging
	query: str # raw query string, no leading '?'

	def __init__( self, scheme: str, host: str, port: u16, path: str, query: str ) -> None:
		self.scheme = scheme
		self.host = host
		self.port = port
		self.path = path
		self.query = query

def _u16_from_str( s: str ) -> Result[u16, HTTPError]:
	n: usize = _usize_from_str( s ).or_return()
	if n > 65535:
		return Result.Err( HTTPError.InvalidURL( None ))
	with compiler.panic_arithmetic( 'bounded by the check above, cannot overflow' ):
		return Result.Ok( u16( n ))

def _parse_url( url: str ) -> Result[ParsedURL, HTTPError]:
	split: SplitResult = urlsplit( url )
	if split.scheme != 'http' and split.scheme != 'https':
		return Result.Err( HTTPError.InvalidURL( None ))
	if split.netloc.byte_len() == 0:
		return Result.Err( HTTPError.InvalidURL( None ))

	hp_split: tuple[str,str,str] = split.netloc.partition( ':' )
	host: str = hp_split[0]
	if host.byte_len() == 0:
		return Result.Err( HTTPError.InvalidURL( None ))
	port: u16 = 443 if split.scheme == 'https' else 80
	if hp_split[1].byte_len() != 0:
		port = _u16_from_str( hp_split[2] ).or_return()

	path: str = split.path
	if path.byte_len() == 0:
		path = '/'

	return Result.Ok( ParsedURL( split.scheme, host, port, path, split.query ))

# ---------------------------------------------------------------------------
# query-string / form encoding - dict[str,str] iterated via key_at/value_at
# (lib/builtins/__init__.py's dict[K,V] positional accessors) into
# list[tuple[str,str]] pairs, then urllib.parse's parse_qsl()/urlencode() do
# the actual percent-encoding (quote_plus - space becomes '+', matching
# application/x-www-form-urlencoded and requests' own params=/data= dict
# encoding, both built on Python's own urlencode()).
# ---------------------------------------------------------------------------

def _dict_to_pairs( d: dict[str,str] ) -> list[tuple[str,str]]:
	pairs: list[tuple[str,str]] = list[tuple[str,str]]()
	n: usize = d.__len__()
	i: usize = 0
	for i in range( n ):
		key: str = d.key_at( i ).unwrap( '_dict_to_pairs: index in bounds by construction' )
		value: str = d.value_at( i ).unwrap( '_dict_to_pairs: index in bounds by construction' )
		pairs.append( ( key, value ))
	return pairs

def _build_request_path( parsed: ParsedURL, params: dict[str,str]|None ) -> Result[str, HTTPError]:
	''' parsed.path, plus parsed's own query merged with params= (params=
	appended after whatever query the URL already carried, matching
	requests' own params= behavior). '''
	pairs: list[tuple[str,str]] = list[tuple[str,str]]()
	if parsed.query.byte_len() != 0:
		match parse_qsl( parsed.query ):
			case Result.Ok( existing ):
				pairs = existing
			case Result.Err( _ ):
				return Result.Err( HTTPError.InvalidURL( None ))
	if params is not None:
		p: dict[str,str] = params
		extra: list[tuple[str,str]] = _dict_to_pairs( p )
		en: usize = extra.__len__()
		ei: usize = 0
		for ei in range( en ):
			pairs.append( extra.__getitem__( ei ).unwrap( '_build_request_path: index in bounds by construction' ))
	if pairs.__len__() == 0:
		return Result.Ok( parsed.path )
	return Result.Ok( parsed.path + '?' + urlencode( pairs ))

def _form_encode( data: dict[str,str] ) -> str:
	return urlencode( _dict_to_pairs( data ))

def _copy_headers( h: HTTPHeaders ) -> HTTPHeaders:
	out: HTTPHeaders = HTTPHeaders()
	n: usize = h.__len__()
	i: usize = 0
	for i in range( n ):
		name: str = h.name_at( i ).unwrap( '_copy_headers: index in bounds by construction' )
		value: str = h.value_at( i ).unwrap( '_copy_headers: index in bounds by construction' )
		out.add( name, value )
	return out

# ---------------------------------------------------------------------------
# Session - cookie jar + redirect-following layer over HTTPConnection.
# Mirrors requests.Session; see PLAN_HTTP_CLIENT.md's Phase 3b. Every request
# opens a fresh HTTPConnection (no keep-alive/connection pooling - see the
# plan's own Phase 4). data is bytes|str|None rather than requests' own
# bytes|str|dict|None - a dict-shaped form body goes through the separate
# form= parameter instead (see _encode_body's own comment on why).
# ---------------------------------------------------------------------------

_MAX_REDIRECTS: usize = 10

def _encode_body( data: bytes|str|None, form: dict[str,str]|None, json_value: JSONValue|None ) -> Result[tuple[bytes|None, str|None], HTTPError]:
	''' -> (body bytes, Content-Type to set if not already present). json_value,
	form, and data are mutually exclusive (checked in that priority order if
	somehow more than one is given - not expected in practice). form/json_value
	are kept as their own separate optional parameters rather than one
	requests-style bytes|str|dict|JSONValue|None union: match-based dispatch
	across a real multi-type union is untested territory in this codebase (the
	only confirmed match-on-union-member precedent, union_coercion_rc_test.py's
	Box|None, is a single real type - see PLAN_HTTP_CLIENT.md's own note on
	this), and dict[str,str]/JSONValue specifically nested inside a wider union
	raises the same generic-argument-in-a-union-position question the
	list[tuple[...]] gap (task_a8b4e7c3's sibling, already fixed once) was
	about. Splitting them out avoids the question entirely rather than
	gambling on untested compiler territory here too. Returns a Result (unlike
	the plain data=/form= paths, which can't fail) because json_value can:
	dumps() rejects a non-finite float anywhere in the value. '''
	if json_value is not None:
		jv: JSONValue = json_value
		match dumps( jv ):
			case Result.Ok( text ):
				body: bytes = text.encode().unwrap( '_encode_body: json.dumps() output is always valid UTF-8' )
				return Result.Ok(( body, 'application/json' ))
			case Result.Err( _ ):
				return Result.Err( HTTPError.InvalidJSON( None ))
	if form is not None:
		f: dict[str,str] = form
		encoded: str = _form_encode( f )
		form_body: bytes = encoded.encode().unwrap( '_encode_body: form encoding is always ASCII' )
		return Result.Ok(( form_body, 'application/x-www-form-urlencoded' ))
	if data is not None:
		match data:
			case bytes( b ):
				return Result.Ok(( b, None ))
			case str( s ):
				sb: bytes = s.encode().unwrap( '_encode_body: request body string must be valid UTF-8' )
				return Result.Ok(( sb, None ))
	return Result.Ok(( None, None ))

def _is_redirect_status( status_code: u16 ) -> bool:
	return status_code == 301 or status_code == 302 or status_code == 303 or status_code == 307 or status_code == 308

def _build_request_headers( session_headers: HTTPHeaders, content_type: str|None, extra_headers: HTTPHeaders|None, cookie_header: str|None, auth: tuple[str,str]|None ) -> HTTPHeaders:
	''' merges session defaults + a computed Content-Type + per-call headers=
	(highest precedence among headers) + the Cookie jar + auth= into one
	HTTPHeaders for a single request. A plain (non-looping) function
	specifically so `is not None` narrowing works - see Session.request()'s
	own comment on why this couldn't just be inlined in its while loop. '''
	request_headers: HTTPHeaders = _copy_headers( session_headers )
	if content_type is not None:
		ct: str = content_type
		request_headers.set( 'Content-Type', ct )
	if extra_headers is not None:
		h: HTTPHeaders = extra_headers
		hn: usize = h.__len__()
		hi: usize = 0
		for hi in range( hn ):
			hname: str = h.name_at( hi ).unwrap( '_build_request_headers: index in bounds by construction' )
			hvalue: str = h.value_at( hi ).unwrap( '_build_request_headers: index in bounds by construction' )
			request_headers.set( hname, hvalue )
	if cookie_header is not None:
		ch: str = cookie_header
		request_headers.set( 'Cookie', ch )
	if auth is not None:
		a: tuple[str,str] = auth
		credentials: str = a[0] + ':' + a[1]
		cred_bytes: bytes = credentials.encode().unwrap( '_build_request_headers: auth= must be valid UTF-8' )
		request_headers.set( 'Authorization', 'Basic ' + base64_encode( cred_bytes ))
	return request_headers

def _next_redirect_url( response: Response, allow_redirects: bool, current_url: str ) -> str:
	''' the URL to redirect to, or '' if this response isn't a redirect that
	should be followed - an empty-string sentinel rather than str|None so
	Session.request()'s own while loop never needs to narrow an Optional
	(see its own comment on why that doesn't work inside that loop). A
	relative Location is resolved against current_url via lib/urllib/parse.py
	's urljoin() (RFC 3986 5.3) - previously only an absolute http://
	Location was followed, treating a relative one as "don't redirect"; that
	restriction is gone now that urljoin() exists. Both http:// and https://
	results are followed now that lib/ssl.py makes https:// reachable too
	(including a redirect that crosses schemes, e.g. http:// -> https://,
	which requests() itself also follows transparently). '''
	if not allow_redirects:
		return ''
	if not _is_redirect_status( response.status_code ):
		return ''
	location: str|None = response.headers.get( 'Location' )
	if location is not None:
		loc: str = location
		resolved: str = urljoin( current_url, loc )
		if resolved.startswith( 'http://' ) or resolved.startswith( 'https://' ):
			return resolved
	return ''

# ---------------------------------------------------------------------------
# request/response logging with secret redaction (PLAN_HTTP_CLIENT.md's
# "Remaining backlog" item 1) - design lifted from a working real-world
# reference (the user's own C:\cvs\itas\incpy\demands.py): redact by VALUE
# (caller-supplied secret strings), not by header name; censor the whole
# serialized wire message in one pass; redact before ever handing text to
# the sink; sink is a plain callable, not a Logger, so each call site can
# log in its own context (or nowhere - print, a bound logger.info method,
# whatever the caller wants).
#
# log is a genuine `Ptr[Callable[[str],None]]|None = None`, narrowed with
# `if log is not None:` before calling - this used to crash ("cannot call
# log" after narrowing; a separate emitter crash storing one as a union
# payload) and briefly shipped as a no-op-default-function workaround
# instead of `None` to dodge both. Fixed for real by 7e0703e (task_92b90a9a)
# - reverted back to the plain, obvious `|None` shape once that landed.
# ---------------------------------------------------------------------------

def _redact( text: str, sensitive_values: list[str]|None ) -> str:
	''' replaces every occurrence of every string in sensitive_values with
	'***CENSORED***', longest value first so a short secret that happens to
	be a substring of a longer one doesn't leave a mangled partial redaction
	behind. A plain str.replace() chain, not a regex - every value here is a
	literal to match verbatim, never a pattern, so a regex library buys
	nothing (per the user directly). list[T] has no sort() yet, hence the
	manual selection sort - fine at the size this ever runs at (a handful of
	caller-supplied secrets per request, not a bulk data path). Copies
	sensitive_values into a local list first rather than sorting it in
	place, so a caller's own list isn't silently reordered as a side effect. '''
	if sensitive_values is None:
		return text
	source: list[str] = sensitive_values
	n: usize = source.__len__()
	values: list[str] = list[str]()
	i: usize = 0
	for i in range( n ):
		v: str = source.__getitem__( i ).unwrap( '_redact: index in bounds by construction' )
		values.append( v )

	with compiler.panic_arithmetic( 'n bounded by a real caller-supplied list, cannot overflow' ):
		for i in range( n ):
			longest: usize = i
			j: usize = i + 1
			for j in range( i + 1, n ):
				a: str = values.__getitem__( j ).unwrap( '_redact: index in bounds by construction' )
				b: str = values.__getitem__( longest ).unwrap( '_redact: index in bounds by construction' )
				if a.__len__() > b.__len__():
					longest = j
			if longest != i:
				vi: str = values.__getitem__( i ).unwrap( '_redact: index in bounds by construction' )
				vl: str = values.__getitem__( longest ).unwrap( '_redact: index in bounds by construction' )
				values.__setitem__( i, vl ).unwrap( '_redact: index in bounds by construction' )
				values.__setitem__( longest, vi ).unwrap( '_redact: index in bounds by construction' )

	redacted: str = text
	for i in range( n ):
		secret: str = values.__getitem__( i ).unwrap( '_redact: index in bounds by construction' )
		if secret.__len__() > 0:
			redacted = redacted.replace( secret, '***CENSORED***' )
	return redacted

def _body_text_for_log( content: bytes ) -> str:
	''' best-effort text rendering of a body for logging - invalid UTF-8
	(a binary body) collapses to a placeholder rather than attempting a
	byte-level escape (CPython's backslashreplace equivalent) that this
	stdlib has no codec support for yet. '''
	if content.__len__() == 0:
		return ''
	match content.decode():
		case Result.Ok( s ):
			return s
		case Result.Err( _ ):
			return '<' + content.__len__().__str__() + ' bytes, not valid UTF-8>'

def _build_request_text( method: str, path: str, host: str, headers: HTTPHeaders, body: bytes|None ) -> str:
	# request_head, not head - see _build_request_head's own comment
	request_head: str = _build_request_head( method, path, host, headers, body )
	if body is None:
		return request_head
	b: bytes = body
	return request_head + _body_text_for_log( b )

def _log_if_present( log: Ptr[Callable[[str],None]]|None, text: str ) -> None:
	''' a plain (non-looping) helper, not `if log is not None: log(text)`
	inlined at each call site - Session.request()'s own while-True loop
	comment explains why `is not None` narrowing doesn't survive a loop
	back-edge here; this sidesteps it the same way every other Optional in
	that loop already does. '''
	if log is not None:
		log( text )

def _build_response_text( response: Response ) -> str:
	# response_head, not head - see _build_request_head's own comment
	response_head: str = 'HTTP/1.1 ' + response.status_code.__str__() + ' ' + response.reason + '\r\n'
	n: usize = response.headers.__len__()
	i: usize = 0
	for i in range( n ):
		name: str = response.headers.name_at( i ).unwrap( '_build_response_text: index in bounds by construction' )
		value: str = response.headers.value_at( i ).unwrap( '_build_response_text: index in bounds by construction' )
		response_head = response_head + name + ': ' + value + '\r\n'
	return response_head + '\r\n' + _body_text_for_log( response.content )

class Session:
	headers: HTTPHeaders
	__cookies: dict[str,str]

	def __init__( self ) -> None:
		self.headers = HTTPHeaders()
		self.__cookies = dict[str,str]()

	def _harvest_cookies( self, response_headers: HTTPHeaders ) -> None:
		set_cookies: list[str] = response_headers.get_all( 'Set-Cookie' )
		n: usize = set_cookies.__len__()
		i: usize = 0
		for i in range( n ):
			raw: str = set_cookies.__getitem__( i ).unwrap( '_harvest_cookies: index in bounds by construction' )
			first_part: tuple[str,str,str] = raw.partition( ';' )
			kv: tuple[str,str,str] = first_part[0].partition( '=' )
			if kv[1].byte_len() == 0:
				continue
			name: str = kv[0].strip()
			value: str = kv[2].strip()
			if name.byte_len() == 0:
				continue
			self.__cookies[ name ] = value

	def _build_cookie_header( self, extra: dict[str,str]|None ) -> str|None:
		n: usize = self.__cookies.__len__()
		extra_n: usize = 0
		if extra is not None:
			e: dict[str,str] = extra
			extra_n = e.__len__()
		if n == 0 and extra_n == 0:
			return None
		parts: list[str] = list[str]()
		i: usize = 0
		for i in range( n ):
			name: str = self.__cookies.key_at( i ).unwrap( '_build_cookie_header: index in bounds by construction' )
			value: str = self.__cookies.value_at( i ).unwrap( '_build_cookie_header: index in bounds by construction' )
			parts.append( name + '=' + value )
		if extra is not None:
			e2: dict[str,str] = extra
			for i in range( extra_n ):
				name = e2.key_at( i ).unwrap( '_build_cookie_header: index in bounds by construction' )
				value = e2.value_at( i ).unwrap( '_build_cookie_header: index in bounds by construction' )
				parts.append( name + '=' + value )
		return '; '.join( parts )

	def request( self, method: str, url: str,
		params: dict[str,str]|None = None,
		data: bytes|str|None = None,
		form: dict[str,str]|None = None,
		json: JSONValue|None = None,
		headers: HTTPHeaders|None = None,
		cookies: dict[str,str]|None = None,
		auth: tuple[str,str]|None = None,
		allow_redirects: bool = True,
		verify: bool = True,
		log: Ptr[Callable[[str],None]]|None = None,
		sensitive_values: list[str]|None = None,
	) -> Result[Response, HTTPError]:
		current_method: str = method
		current_url: str = url
		encoded_body: tuple[bytes|None, str|None] = _encode_body( data, form, json ).or_return()
		current_body: bytes|None = encoded_body[0]
		content_type: str|None = encoded_body[1]

		redirect_count: usize = 0
		with compiler.panic_arithmetic( 'bounded by _MAX_REDIRECTS, cannot overflow' ):
			while True:
				parsed: ParsedURL = _parse_url( current_url ).or_return()
				full_path: str = _build_request_path( parsed, params ).or_return()
				# `x is not None` narrowing doesn't hold up inside a `while
				# True:` loop body here (confirmed by several real compiles:
				# every one of content_type/headers/cookie_header/auth/
				# location failed to narrow inside this loop, not just the
				# ones actually reassigned elsewhere in it - narrowing
				# apparently doesn't survive a loop back-edge at all, not
				# just mutation specifically). Every Optional this loop needs
				# is instead threaded through as a bare parameter into a
				# plain (non-looping) helper function below, where the same
				# narrowing pattern DOES work (matches getresponse()'s own
				# working straight-line narrowing).
				cookie_header: str|None = self._build_cookie_header( cookies )
				request_headers: HTTPHeaders = _build_request_headers( self.headers, content_type, headers, cookie_header, auth )

				_log_if_present( log, _redact( _build_request_text( current_method, full_path, parsed.host, request_headers, current_body ), sensitive_values ))

				response: Response = _perform_request_for_scheme( parsed.scheme, parsed.host, parsed.port, current_method, full_path, request_headers, current_body, verify ).or_return()
				response.url = current_url
				_log_if_present( log, _redact( _build_response_text( response ), sensitive_values ))
				self._harvest_cookies( response.headers )

				# '' is a "don't redirect" sentinel, not Optional - see
				# _next_redirect_url's own comment for why
				next_url: str = _next_redirect_url( response, allow_redirects, current_url )
				if next_url.byte_len() == 0:
					return Result.Ok( response )

				if redirect_count >= _MAX_REDIRECTS:
					return Result.Err( HTTPError.TooManyRedirects( None ))
				redirect_count += 1

				if response.status_code == 301 or response.status_code == 302 or response.status_code == 303:
					current_method = 'GET'
					current_body = None
					content_type = None
				current_url = next_url

	def get( self, url: str, params: dict[str,str]|None = None, headers: HTTPHeaders|None = None,
		cookies: dict[str,str]|None = None, auth: tuple[str,str]|None = None, allow_redirects: bool = True, verify: bool = True,
		log: Ptr[Callable[[str],None]]|None = None, sensitive_values: list[str]|None = None ) -> Result[Response, HTTPError]:
		return self.request( 'GET', url, params = params, headers = headers, cookies = cookies, auth = auth, allow_redirects = allow_redirects, verify = verify, log = log, sensitive_values = sensitive_values )

	def post( self, url: str, data: bytes|str|None = None, form: dict[str,str]|None = None, json: JSONValue|None = None, params: dict[str,str]|None = None,
		headers: HTTPHeaders|None = None, cookies: dict[str,str]|None = None, auth: tuple[str,str]|None = None, allow_redirects: bool = True, verify: bool = True,
		log: Ptr[Callable[[str],None]]|None = None, sensitive_values: list[str]|None = None ) -> Result[Response, HTTPError]:
		return self.request( 'POST', url, params = params, data = data, form = form, json = json, headers = headers, cookies = cookies, auth = auth, allow_redirects = allow_redirects, verify = verify, log = log, sensitive_values = sensitive_values )

	def put( self, url: str, data: bytes|str|None = None, form: dict[str,str]|None = None, json: JSONValue|None = None, params: dict[str,str]|None = None,
		headers: HTTPHeaders|None = None, cookies: dict[str,str]|None = None, auth: tuple[str,str]|None = None, allow_redirects: bool = True, verify: bool = True,
		log: Ptr[Callable[[str],None]]|None = None, sensitive_values: list[str]|None = None ) -> Result[Response, HTTPError]:
		return self.request( 'PUT', url, params = params, data = data, form = form, json = json, headers = headers, cookies = cookies, auth = auth, allow_redirects = allow_redirects, verify = verify, log = log, sensitive_values = sensitive_values )

	def patch( self, url: str, data: bytes|str|None = None, form: dict[str,str]|None = None, json: JSONValue|None = None, params: dict[str,str]|None = None,
		headers: HTTPHeaders|None = None, cookies: dict[str,str]|None = None, auth: tuple[str,str]|None = None, allow_redirects: bool = True, verify: bool = True,
		log: Ptr[Callable[[str],None]]|None = None, sensitive_values: list[str]|None = None ) -> Result[Response, HTTPError]:
		return self.request( 'PATCH', url, params = params, data = data, form = form, json = json, headers = headers, cookies = cookies, auth = auth, allow_redirects = allow_redirects, verify = verify, log = log, sensitive_values = sensitive_values )

	def delete( self, url: str, params: dict[str,str]|None = None, headers: HTTPHeaders|None = None,
		cookies: dict[str,str]|None = None, auth: tuple[str,str]|None = None, allow_redirects: bool = True, verify: bool = True,
		log: Ptr[Callable[[str],None]]|None = None, sensitive_values: list[str]|None = None ) -> Result[Response, HTTPError]:
		return self.request( 'DELETE', url, params = params, headers = headers, cookies = cookies, auth = auth, allow_redirects = allow_redirects, verify = verify, log = log, sensitive_values = sensitive_values )

	def head( self, url: str, params: dict[str,str]|None = None, headers: HTTPHeaders|None = None,
		cookies: dict[str,str]|None = None, auth: tuple[str,str]|None = None, allow_redirects: bool = False, verify: bool = True,
		log: Ptr[Callable[[str],None]]|None = None, sensitive_values: list[str]|None = None ) -> Result[Response, HTTPError]:
		return self.request( 'HEAD', url, params = params, headers = headers, cookies = cookies, auth = auth, allow_redirects = allow_redirects, verify = verify, log = log, sensitive_values = sensitive_values )

	def options( self, url: str, params: dict[str,str]|None = None, headers: HTTPHeaders|None = None,
		cookies: dict[str,str]|None = None, auth: tuple[str,str]|None = None, allow_redirects: bool = True, verify: bool = True,
		log: Ptr[Callable[[str],None]]|None = None, sensitive_values: list[str]|None = None ) -> Result[Response, HTTPError]:
		return self.request( 'OPTIONS', url, params = params, headers = headers, cookies = cookies, auth = auth, allow_redirects = allow_redirects, verify = verify, log = log, sensitive_values = sensitive_values )

# ---------------------------------------------------------------------------
# module-level convenience - each a one-off Session() underneath, matching
# requests.get()/requests.post()/etc. (and the PHP fetch() reference's own
# module-level fetch() wrapping a one-off Session).
# ---------------------------------------------------------------------------

def get( url: str, params: dict[str,str]|None = None, headers: HTTPHeaders|None = None,
	cookies: dict[str,str]|None = None, auth: tuple[str,str]|None = None, allow_redirects: bool = True, verify: bool = True,
	log: Ptr[Callable[[str],None]]|None = None, sensitive_values: list[str]|None = None ) -> Result[Response, HTTPError]:
	return Session().get( url, params = params, headers = headers, cookies = cookies, auth = auth, allow_redirects = allow_redirects, verify = verify, log = log, sensitive_values = sensitive_values )

def post( url: str, data: bytes|str|None = None, form: dict[str,str]|None = None, json: JSONValue|None = None, params: dict[str,str]|None = None,
	headers: HTTPHeaders|None = None, cookies: dict[str,str]|None = None, auth: tuple[str,str]|None = None, allow_redirects: bool = True, verify: bool = True,
	log: Ptr[Callable[[str],None]]|None = None, sensitive_values: list[str]|None = None ) -> Result[Response, HTTPError]:
	return Session().post( url, data = data, form = form, json = json, params = params, headers = headers, cookies = cookies, auth = auth, allow_redirects = allow_redirects, verify = verify, log = log, sensitive_values = sensitive_values )

def put( url: str, data: bytes|str|None = None, form: dict[str,str]|None = None, json: JSONValue|None = None, params: dict[str,str]|None = None,
	headers: HTTPHeaders|None = None, cookies: dict[str,str]|None = None, auth: tuple[str,str]|None = None, allow_redirects: bool = True, verify: bool = True,
	log: Ptr[Callable[[str],None]]|None = None, sensitive_values: list[str]|None = None ) -> Result[Response, HTTPError]:
	return Session().put( url, data = data, form = form, json = json, params = params, headers = headers, cookies = cookies, auth = auth, allow_redirects = allow_redirects, verify = verify, log = log, sensitive_values = sensitive_values )

def patch( url: str, data: bytes|str|None = None, form: dict[str,str]|None = None, json: JSONValue|None = None, params: dict[str,str]|None = None,
	headers: HTTPHeaders|None = None, cookies: dict[str,str]|None = None, auth: tuple[str,str]|None = None, allow_redirects: bool = True, verify: bool = True,
	log: Ptr[Callable[[str],None]]|None = None, sensitive_values: list[str]|None = None ) -> Result[Response, HTTPError]:
	return Session().patch( url, data = data, form = form, json = json, params = params, headers = headers, cookies = cookies, auth = auth, allow_redirects = allow_redirects, verify = verify, log = log, sensitive_values = sensitive_values )

def delete( url: str, params: dict[str,str]|None = None, headers: HTTPHeaders|None = None,
	cookies: dict[str,str]|None = None, auth: tuple[str,str]|None = None, allow_redirects: bool = True, verify: bool = True,
	log: Ptr[Callable[[str],None]]|None = None, sensitive_values: list[str]|None = None ) -> Result[Response, HTTPError]:
	return Session().delete( url, params = params, headers = headers, cookies = cookies, auth = auth, allow_redirects = allow_redirects, verify = verify, log = log, sensitive_values = sensitive_values )

def head( url: str, params: dict[str,str]|None = None, headers: HTTPHeaders|None = None,
	cookies: dict[str,str]|None = None, auth: tuple[str,str]|None = None, allow_redirects: bool = False, verify: bool = True,
	log: Ptr[Callable[[str],None]]|None = None, sensitive_values: list[str]|None = None ) -> Result[Response, HTTPError]:
	return Session().head( url, params = params, headers = headers, cookies = cookies, auth = auth, allow_redirects = allow_redirects, verify = verify, log = log, sensitive_values = sensitive_values )

def options( url: str, params: dict[str,str]|None = None, headers: HTTPHeaders|None = None,
	cookies: dict[str,str]|None = None, auth: tuple[str,str]|None = None, allow_redirects: bool = True, verify: bool = True,
	log: Ptr[Callable[[str],None]]|None = None, sensitive_values: list[str]|None = None ) -> Result[Response, HTTPError]:
	return Session().options( url, params = params, headers = headers, cookies = cookies, auth = auth, allow_redirects = allow_redirects, verify = verify, log = log, sensitive_values = sensitive_values )
