# Real-compile-and-run tests for lib/urllib/parse.py - quote/quote_plus,
# unquote/unquote_plus, urlencode/parse_qsl, urlsplit/urlunsplit, and
# urljoin. Same shape as http_client_test.py (which this module's
# percent-encoding was absorbed from - see that file's own history):
# test_support.RealCompileMixin compiles every case into one program and
# runs it for real, each main() -> i32 returning a distinct nonzero code per
# failed assertion.

# stdlib imports:
import unittest

# local imports:
import test_support


class UrllibParseTests( test_support.RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile urllib.parse tests' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'quote_reserved_unreserved_and_utf8', '''
from urllib.parse import quote

def main() -> i32:
	if quote( 'hello world!' ) != 'hello%20world%21':
		return 1
	# every RFC 3986 unreserved character must pass through unchanged
	if quote( 'abc-._~XYZ019' ) != 'abc-._~XYZ019':
		return 2
	if quote( '' ) != '':
		return 3
	# multi-byte UTF-8 codepoint -> multiple %XX triplets
	if quote( 'caf\\u00e9' ) != 'caf%C3%A9':
		return 4
	# safe= leaves extra bytes unescaped
	if quote( 'a/b', '/' ) != 'a/b':
		return 5
	if quote( 'a/b' ) != 'a%2Fb':
		return 6
	return 0
''' ),
			( 'quote_plus_space_and_literal_plus', '''
from urllib.parse import quote_plus

def main() -> i32:
	if quote_plus( 'hello world' ) != 'hello+world':
		return 1
	# a literal '+' is not unreserved - must itself be escaped
	if quote_plus( 'a+b' ) != 'a%2Bb':
		return 2
	if quote_plus( '' ) != '':
		return 3
	return 0
''' ),
			( 'unquote_roundtrip_and_malformed', '''
from urllib.parse import unquote

def main() -> i32:
	if unquote( 'hello%20world%21' ).unwrap( 'valid input' ) != 'hello world!':
		return 1
	if unquote( 'abc' ).unwrap( 'valid input' ) != 'abc':
		return 2
	if unquote( 'caf%C3%A9' ).unwrap( 'valid input' ) != 'caf\\u00e9':
		return 3
	if not unquote( '%' ).is_err(): # truncated - no hex digits at all
		return 4
	if not unquote( '%2' ).is_err(): # truncated - only one hex digit
		return 5
	if not unquote( '%zz' ).is_err(): # not hex digits
		return 6
	return 0
''' ),
			( 'unquote_plus_roundtrip', '''
from urllib.parse import quote_plus, unquote_plus

def main() -> i32:
	if unquote_plus( 'a+b' ).unwrap( 'valid input' ) != 'a b':
		return 1
	# a percent-escaped literal '+' must stay '+', not become a space
	if unquote_plus( 'a%2Bb' ).unwrap( 'valid input' ) != 'a+b':
		return 2
	original: str = 'hello world+more'
	if unquote_plus( quote_plus( original )).unwrap( 'roundtrip' ) != original:
		return 3
	return 0
''' ),
			( 'urlencode_and_parse_qsl_roundtrip', '''
from urllib.parse import urlencode, parse_qsl

def main() -> i32:
	pairs: list[tuple[str,str]] = list[tuple[str,str]]()
	pairs.append( ( 'a', '1' ))
	pairs.append( ( 'b', 'hello world' ))
	encoded: str = urlencode( pairs )
	if encoded != 'a=1&b=hello+world':
		return 1

	special: list[tuple[str,str]] = list[tuple[str,str]]()
	special.append( ( 'key', 'a&b=c' ))
	if urlencode( special ) != 'key=a%26b%3Dc':
		return 2

	parsed: list[tuple[str,str]] = parse_qsl( encoded ).unwrap( 'valid qs' )
	if parsed.__len__() != 2:
		return 3
	p0: tuple[str,str] = parsed.__getitem__( 0 ).unwrap( 'index in bounds' )
	if p0[0] != 'a':
		return 4
	if p0[1] != '1':
		return 5
	p1: tuple[str,str] = parsed.__getitem__( 1 ).unwrap( 'index in bounds' )
	if p1[0] != 'b':
		return 6
	if p1[1] != 'hello world':
		return 7

	# leading '?', and blank pairs from a doubled '&', are tolerated/skipped
	skipped: list[tuple[str,str]] = parse_qsl( '?a=1&&b=2' ).unwrap( 'valid qs' )
	if skipped.__len__() != 2:
		return 8

	empty: list[tuple[str,str]] = parse_qsl( '' ).unwrap( 'valid qs' )
	if empty.__len__() != 0:
		return 9
	return 0
''' ),
			( 'urlsplit_full_url_and_edge_cases', '''
from urllib.parse import urlsplit, SplitResult

def main() -> i32:
	full: SplitResult = urlsplit( 'http://user:pass@host.com:8080/path/to/thing?x=1&y=2#frag' )
	if full.scheme != 'http':
		return 1
	if full.netloc != 'user:pass@host.com:8080':
		return 2
	if full.path != '/path/to/thing':
		return 3
	if full.query != 'x=1&y=2':
		return 4
	if full.fragment != 'frag':
		return 5

	no_scheme: SplitResult = urlsplit( '//host/path' )
	if no_scheme.scheme != '':
		return 6
	if no_scheme.netloc != 'host':
		return 7
	if no_scheme.path != '/path':
		return 8

	no_netloc: SplitResult = urlsplit( '/just/a/path?q=1' )
	if no_netloc.scheme != '':
		return 9
	if no_netloc.netloc != '':
		return 10
	if no_netloc.path != '/just/a/path':
		return 11
	if no_netloc.query != 'q=1':
		return 12

	mail: SplitResult = urlsplit( 'mailto:foo@example.com' )
	if mail.scheme != 'mailto':
		return 13
	if mail.netloc != '':
		return 14
	if mail.path != 'foo@example.com':
		return 15

	# scheme is lowercased, matching Python
	upper: SplitResult = urlsplit( 'HTTP://Host/Path' )
	if upper.scheme != 'http':
		return 16
	if upper.netloc != 'Host':
		return 17
	return 0
''' ),
			( 'urlunsplit_roundtrips_urlsplit', '''
from urllib.parse import urlsplit, urlunsplit

def main() -> i32:
	full: str = 'http://user:pass@host.com:8080/path/to/thing?x=1&y=2#frag'
	if urlunsplit( urlsplit( full )) != full:
		return 1
	mail: str = 'mailto:foo@example.com'
	if urlunsplit( urlsplit( mail )) != mail:
		return 2
	relative: str = '/just/a/path?q=1'
	if urlunsplit( urlsplit( relative )) != relative:
		return 3
	return 0
''' ),
			( 'urljoin_rfc3986_worked_examples', '''
from urllib.parse import urljoin

def main() -> i32:
	base: str = 'http://a/b/c/d;p?q'
	# RFC 3986 5.4.1 normal examples
	if urljoin( base, 'g' ) != 'http://a/b/c/g':
		return 1
	if urljoin( base, './g' ) != 'http://a/b/c/g':
		return 2
	if urljoin( base, 'g/' ) != 'http://a/b/c/g/':
		return 3
	if urljoin( base, '/g' ) != 'http://a/g':
		return 4
	if urljoin( base, '//g' ) != 'http://g':
		return 5
	if urljoin( base, '?y' ) != 'http://a/b/c/d;p?y':
		return 6
	if urljoin( base, 'g?y' ) != 'http://a/b/c/g?y':
		return 7
	if urljoin( base, '#s' ) != 'http://a/b/c/d;p?q#s':
		return 8
	if urljoin( base, 'g#s' ) != 'http://a/b/c/g#s':
		return 9
	if urljoin( base, ';x' ) != 'http://a/b/c/;x':
		return 10
	if urljoin( base, '' ) != 'http://a/b/c/d;p?q':
		return 11
	if urljoin( base, '.' ) != 'http://a/b/c/':
		return 12
	if urljoin( base, '..' ) != 'http://a/b/':
		return 13
	if urljoin( base, '../g' ) != 'http://a/b/g':
		return 14
	if urljoin( base, '../..' ) != 'http://a/':
		return 15
	if urljoin( base, '../../g' ) != 'http://a/g':
		return 16

	# RFC 3986 5.4.2 abnormal examples - excess ".." never escapes the root
	if urljoin( base, '../../../g' ) != 'http://a/g':
		return 17
	if urljoin( base, '../../../../g' ) != 'http://a/g':
		return 18
	if urljoin( base, '/./g' ) != 'http://a/g':
		return 19
	if urljoin( base, '/../g' ) != 'http://a/g':
		return 20
	if urljoin( base, 'g.' ) != 'http://a/b/c/g.':
		return 21
	if urljoin( base, 'g/./h' ) != 'http://a/b/c/g/h':
		return 22
	if urljoin( base, 'g/../h' ) != 'http://a/b/c/h':
		return 23
	if urljoin( base, 'g;x=1/../y' ) != 'http://a/b/c/y':
		return 24

	# a reference with its own scheme is returned normalized, not merged
	if urljoin( base, 'g:h' ) != 'g:h':
		return 25
	return 0
''' ),
		])
