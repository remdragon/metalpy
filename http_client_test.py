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


if __name__ == '__main__':
	unittest.main()
