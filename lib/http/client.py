'''
HTTP wire-format primitives, plus a low-level one-connection-at-a-time client
(HTTPConnection/Response) built on lib/socket.py - see PLAN_HTTP_CLIENT.md.

HTTPError       - error type shared by every function/class below
HTTPHeaders     - ordered, case-insensitive multimap for request/response headers
parse_status_line  - "HTTP/1.1 200 OK" -> (version, status_code, reason)
parse_header_line  - "Name: value" -> (name, value)
parse_headers      - a raw CRLF-joined header block -> HTTPHeaders
percent_encode     - RFC 3986 percent-encoding of a str's UTF-8 bytes
base64_encode      - standard (padded) base64 encoding of bytes, for auth=
decode_chunked     - decodes an already-fully-buffered chunked-transfer body
HTTPConnection     - connect/request/getresponse/close over one TCP connection
Response           - status_code/reason/headers/content of a received response

host is an IP literal only for now (lib/socket.py itself has no DNS/
getaddrinfo yet - see that file's own header comment and this plan's own
"Socket surface" section). requests-style Session (cookie jar, redirects,
params=/data=/json=/auth=, module-level get()/post()/...) is a separate,
not-yet-started layer on top of HTTPConnection - see PLAN_HTTP_CLIENT.md's
Phase 3b.
'''

import sys
import compiler
import base64
from socket import Socket

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
	Other: None

# ---------------------------------------------------------------------------
# HTTPHeaders - ordered, case-insensitive multimap
# ---------------------------------------------------------------------------

def _header_name_eq( a: str, b: str ) -> bool:
	return a.lower() == b.lower()

class HTTPHeaders:
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
		self.__entries.append( ( name, value )).unwrap( 'HTTPHeaders.add: append failed' )

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
					rebuilt.append( ( name, value )).unwrap( 'HTTPHeaders.set: append failed' )
					replaced = True
			else:
				rebuilt.append( entry ).unwrap( 'HTTPHeaders.set: append failed' )
		if not replaced:
			rebuilt.append( ( name, value )).unwrap( 'HTTPHeaders.set: append failed' )
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
				result.append( entry[1] ).unwrap( 'HTTPHeaders.get_all: append failed' )
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
# percent-encoding (RFC 3986) - needed by params=/form-encoded data=
# ---------------------------------------------------------------------------

def _is_unreserved_byte( b: u8 ) -> bool:
	if b >= 0x41 and b <= 0x5A: # A-Z
		return True
	if b >= 0x61 and b <= 0x7A: # a-z
		return True
	if b >= 0x30 and b <= 0x39: # 0-9
		return True
	if b == 0x2D or b == 0x5F or b == 0x2E or b == 0x7E: # - _ . ~
		return True
	return False

_HEX_UPPER: str = '0123456789ABCDEF'
_PERCENT: u8 = 0x25 # '%'

def percent_encode( s: str ) -> str:
	''' RFC 3986 percent-encoding, applied to s's own UTF-8 bytes (percent-
	encoding is a byte-level transform, not a codepoint-level one - a
	multi-byte UTF-8 codepoint becomes multiple %XX triplets, matching
	Python's urllib.parse.quote() on a UTF-8-encoded str). '''
	data: ConstPtr[u8] = s.get_const_ptr()
	n: usize = s.byte_len()
	hex_ptr: ConstPtr[u8] = _HEX_UPPER.get_const_ptr()

	out_len: usize = 0
	i: usize = 0
	with compiler.panic_arithmetic( 'irrational string length' ):
		while i < n:
			if _is_unreserved_byte( data[i] ):
				out_len += 1
			else:
				out_len += 3
			i += 1
		buf_size: usize = out_len + 1 # zero terminator

	out: bytearray = bytearray( buf_size )
	out_ptr: Ptr[u8] = out.get_ptr()
	o: usize = 0
	i = 0
	with compiler.wrap_arithmetic:
		while i < n:
			b: u8 = data[i]
			if _is_unreserved_byte( b ):
				out_ptr[o] = b
				o += 1
			else:
				out_ptr[o] = _PERCENT
				out_ptr[o+1] = hex_ptr[ usize( b >> 4 ) ]
				out_ptr[o+2] = hex_ptr[ usize( b & 0x0F ) ]
				o += 3
			i += 1

	return str.from_cstr( move( out )).unwrap( 'percent_encode: invalid UTF-8 (unreachable - output is pure ASCII)' )

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
# small integer <-> str helpers - int (the boxed arbitrary-precision type)
# only converts from/to i32 (lib/builtins/__int.py), not usize, and a
# Content-Length can legitimately need the full usize range - spelled out
# directly rather than routed through int, matching this codebase's own
# established "ASCII digits by hand" idiom (see lib/builtins/__int.py's own
# _ASCII_ZERO-based from_str/digit-count code).
# ---------------------------------------------------------------------------

_ASCII_ZERO: u8 = 0x30
_ASCII_NINE: u8 = 0x39

def _usize_to_str( n: usize ) -> str:
	if n == 0:
		return '0'
	digits: bytearray = bytearray( 20 ) # a u64 fits in at most 20 decimal digits
	d_ptr: Ptr[u8] = digits.get_ptr()
	count: usize = 0
	v: usize = n
	with compiler.panic_arithmetic( 'usize has at most 20 decimal digits, divisor is a nonzero literal' ):
		while v > 0:
			d_ptr[count] = u8( v % 10 ) + _ASCII_ZERO
			v = v // 10
			count += 1
	with compiler.panic_arithmetic( 'count is bounded by 20, cannot overflow' ):
		out: bytearray = bytearray( count + 1 ) # +1 zero terminator
	out_ptr: Ptr[u8] = out.get_ptr()
	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by count, cannot overflow' ):
		while i < count:
			out_ptr[i] = d_ptr[ count - 1 - i ] # digits were built least-significant-first
			i += 1
	return str.from_cstr( move( out )).unwrap( '_usize_to_str: unreachable - pure ASCII digits' )

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
# _GrowableBuffer - accumulates bytes read off a Socket across multiple
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

	def fill_from( self, sock: Socket ) -> Result[usize, OSError]:
		''' one recv() call, appended to the buffer. Returns the number of
		bytes read - 0 means the peer closed the connection. '''
		self._grow( 4096 )
		with compiler.wrap_arithmetic:
			dest: Ptr[u8] = self.__data + self.__len
			room: usize = self.__cap - self.__len
		n: usize = sock.recv( dest, room ).or_return()
		with compiler.wrap_arithmetic:
			self.__len += n
		return Result.Ok( n )

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
		# same approach percent_encode/base64_encode above already use.
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
	head: str = method + ' ' + path + ' HTTP/1.1\r\n' + 'Host: ' + host + '\r\n'
	if headers is not None:
		n: usize = headers.__len__()
		i: usize = 0
		for i in range( n ):
			name: str = headers.name_at( i ).unwrap( '_build_request_head: index in bounds by construction' )
			value: str = headers.value_at( i ).unwrap( '_build_request_head: index in bounds by construction' )
			head = head + name + ': ' + value + '\r\n'
	if body is not None:
		head = head + 'Content-Length: ' + _usize_to_str( body.__len__() ) + '\r\n'
	return head + '\r\n'

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
				case Result.Err( _ ):
					return Result.Err( HTTPError.Other( None ))
		case Result.Err( _ ):
			return Result.Err( HTTPError.Other( None ))

def _send_or_http_err( sock: Socket, ptr: ConstPtr[u8], length: usize ) -> Result[usize, HTTPError]:
	match sock.send( ptr, length ):
		case Result.Ok( n ):
			return Result.Ok( n )
		case Result.Err( _ ):
			return Result.Err( HTTPError.Other( None ))

def _fill_or_http_err( buf: _GrowableBuffer, sock: Socket ) -> Result[usize, HTTPError]:
	match buf.fill_from( sock ):
		case Result.Ok( n ):
			return Result.Ok( n )
		case Result.Err( _ ):
			return Result.Err( HTTPError.Other( None ))

def _send_all( sock: Socket, ptr: ConstPtr[u8], length: usize ) -> Result[None, HTTPError]:
	sent: usize = 0
	with compiler.panic_arithmetic( 'bounded by length, cannot overflow' ):
		while sent < length:
			n: usize = _send_or_http_err( sock, ptr + sent, length - sent ).or_return()
			if n == 0:
				return Result.Err( HTTPError.UnexpectedEOF( None ))
			sent += n
	return Result.Ok( None )

# ---------------------------------------------------------------------------
# Response - the result of HTTPConnection.getresponse()
# ---------------------------------------------------------------------------

class Response:
	status_code: u16
	reason: str
	headers: HTTPHeaders
	content: bytes

	def __init__( self, status_code: u16, reason: str, headers: HTTPHeaders, content: bytes ) -> None:
		self.status_code = status_code
		self.reason = reason
		self.headers = headers
		self.content = content

	def text( self ) -> Result[str, CodecError]:
		return self.content.decode()

	def ok( self ) -> bool:
		return self.status_code < 400

# ---------------------------------------------------------------------------
# response body reading - one function per Content-Length/chunked/until-
# close strategy. Split out of getresponse() itself (rather than inlined
# per-branch) so each has a single, straight-line return path - definite-
# assignment tracking for a shared post-if local reassigned from inside a
# nested `while True: ... break` didn't hold up under a real compile.
# ---------------------------------------------------------------------------

def _read_chunked_body( sock: Socket, buf: _GrowableBuffer, body_start: usize ) -> Result[bytes, HTTPError]:
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
			n: usize = _fill_or_http_err( buf, sock ).or_return()
			if n == 0:
				return Result.Err( HTTPError.UnexpectedEOF( None ))

def _read_content_length_body( sock: Socket, buf: _GrowableBuffer, body_start: usize, content_length: usize ) -> Result[bytes, HTTPError]:
	with compiler.panic_arithmetic( 'a real Content-Length body fits well within usize' ):
		while True:
			with compiler.panic_arithmetic( 'bounded by buf.len(), cannot overflow' ):
				have: usize = buf.len() - body_start
			if have >= content_length:
				break
			n: usize = _fill_or_http_err( buf, sock ).or_return()
			if n == 0:
				return Result.Err( HTTPError.UnexpectedEOF( None ))
	with compiler.wrap_arithmetic:
		body_end: usize = body_start + content_length
	return Result.Ok( buf.slice_bytes( body_start, body_end ))

def _read_until_close_body( sock: Socket, buf: _GrowableBuffer, body_start: usize ) -> Result[bytes, HTTPError]:
	with compiler.panic_arithmetic( 'a real response body fits well within usize' ):
		while True:
			n: usize = _fill_or_http_err( buf, sock ).or_return()
			if n == 0:
				break
	return Result.Ok( buf.slice_bytes( body_start, buf.len() ))

# ---------------------------------------------------------------------------
# HTTPConnection - one TCP connection, one request/response at a time.
# Mirrors lib/builtins/__File.py's handle shape: an owned resource field (a
# Socket, itself already RC-managed with its own auto-closing __del__ - no
# HTTPConnection.__del__ needed, the field's own teardown cascades), a
# private constructor, ordinary Result-returning methods.
# ---------------------------------------------------------------------------

class HTTPConnection:
	__sock: Socket
	__host: str
	__port: u16

	def close( self ) -> None:
		self.__sock.close()

	@staticmethod
	def connect( host: str, port: u16 = 80 ) -> Result[HTTPConnection, HTTPError]:
		''' host must be an IP literal for now - lib/socket.py has no DNS/
		getaddrinfo yet (see this file's own module docstring). '''
		sock: Socket = _connect_or_http_err( host, port ).or_return()
		return Result.Ok( HTTPConnection.__allocate__( __sock = sock, __host = host, __port = port ))

	def request( self, method: str, path: str, headers: HTTPHeaders|None = None, body: bytes|None = None ) -> Result[None, HTTPError]:
		head: str = _build_request_head( method, path, self.__host, headers, body )
		head_bytes: bytes = head.encode().unwrap( '_build_request_head: unreachable - pure ASCII output' )
		_send_all( self.__sock, head_bytes.get_const_ptr(), head_bytes.__len__() ).or_return()
		if body is not None:
			content: bytes = body
			_send_all( self.__sock, content.get_const_ptr(), content.__len__() ).or_return()
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
				n: usize = _fill_or_http_err( buf, self.__sock ).or_return()
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
			content = _read_chunked_body( self.__sock, buf, body_start ).or_return()
		else:
			content_length_str: str|None = headers.get( 'Content-Length' )
			if content_length_str is not None:
				cl: str = content_length_str
				content_length: usize = _usize_from_str( cl ).or_return()
				content = _read_content_length_body( self.__sock, buf, body_start, content_length ).or_return()
			else:
				# no Content-Length, not chunked - read until the peer closes
				content = _read_until_close_body( self.__sock, buf, body_start ).or_return()

		return Result.Ok( Response( parsed_status[1], parsed_status[2], headers, content ))
