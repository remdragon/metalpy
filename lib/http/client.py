'''
HTTP wire-format primitives - the zero-prerequisite "Phase 0" slice of
PLAN_HTTP_CLIENT.md. Everything here is a pure str/bytes transform with no
socket or buffered-I/O dependency, so it can be built and tested before the
socket library (a separate, parallel effort) lands.

HTTPError       - error type shared by every function/class below
HTTPHeaders     - ordered, case-insensitive multimap for request/response headers
parse_status_line  - "HTTP/1.1 200 OK" -> (version, status_code, reason)
parse_header_line  - "Name: value" -> (name, value)
parse_headers      - a raw CRLF-joined header block -> HTTPHeaders
base64_encode      - standard (padded) base64 encoding of bytes, for auth=
decode_chunked     - decodes an already-fully-buffered chunked-transfer body

Session/Response/request() (the socket-dependent pieces - Phase 1-3 of
PLAN_HTTP_CLIENT.md) are not part of this file yet.
'''

import sys
import compiler
import base64

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
