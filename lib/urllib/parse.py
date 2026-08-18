'''
URL parsing/encoding, mirroring Python's urllib.parse. Motivated by
PLAN_HTTP_CLIENT.md, which names URL splitting and percent-encoding as
zero-prerequisite pieces needed for params=/form-encoded data= (a minimal
percent_encode() lived directly in lib/http/client.py as a stopgap - it has
been absorbed into this module's quote(), see http/client.py's own history).

v1 scope: quote/quote_plus/unquote/unquote_plus (percent-encoding both
directions), urlencode/parse_qsl (query-string building/parsing as ordered
(name,value) pairs), urlsplit/urlunsplit (SplitResult - the pair Python
itself recommends over urlparse/urlunparse for new code), and urljoin
(RFC 3986 5.3 relative-reference resolution). Deferred: urlparse/urlunparse's
legacy ";params" path-parameter field, parse_qs's dict[str,list[str]]
grouped-value shape (no prior code in this repo uses a generic-typed dict
value - untested territory, and parse_qsl's flat pair-list form is what
Python itself recommends when duplicate keys or key order matter anyway),
urldefrag, and bytes-flavored quote/unquote overloads.

Two deviations from Python's own ergonomics:

1. unquote()/unquote_plus() return Result[str,UrlParseError] and reject a
   malformed "%" not followed by 2 hex digits, rather than Python's lenient
   pass-the-literal-"%"-through behavior - this codebase's established
   "error rather than silently degrade" preference (see base64.py's
   validate= defaulting to True, the opposite of Python's own default).

2. urljoin() can't distinguish "the reference's query is empty" from "the
   reference has no query at all" the way Python's Optional-typed parser
   can, since SplitResult's fields are plain (non-optional) str - a
   reference whose query is empty-but-present (e.g. joining with a bare
   "?") falls back to the base's query instead of clearing it. Not one of
   the RFC 3986 5.4 worked examples this is tested against, and an unusual
   reference to construct on purpose.

SplitResult leaves netloc unparsed (no .hostname/.port/.username/.password
accessors) - a follow-up if a caller needs them; Python itself splits netloc
lazily too.
'''

import compiler

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class UrlParseError:
	message: str

	def __init__( self, message: str ) -> None:
		self.message = message

# ---------------------------------------------------------------------------
# percent-encoding (RFC 3986) - quote()/quote_plus()
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
_PLUS: u8 = 0x2B    # '+'
_SPACE: u8 = 0x20   # ' '

def _is_safe_byte( b: u8, safe: str ) -> bool:
	''' True if b (as a single ASCII byte) appears anywhere in safe - extra
	bytes a caller wants left unescaped (e.g. '/' when quoting a whole
	path), on top of the always-unreserved set. safe is assumed to be pure
	ASCII, matching Python's quote(safe=...) contract; a non-ASCII byte in
	safe simply never matches any encodable byte. '''
	n: usize = safe.byte_len()
	cstr: ConstPtr[u8] = safe.get_const_ptr()
	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by n, cannot overflow' ):
		while i < n:
			if cstr[i] == b:
				return True
			i += 1
	return False

def _quote_impl( s: str, safe: str, space_as_plus: bool ) -> str:
	''' shared two-pass size-then-fill implementation behind quote() and
	quote_plus() - same shape as base64.py's _b64encode() taking an
	alphabet parameter rather than duplicating encode logic per variant.
	Percent-encoding is a byte-level transform, not codepoint-level: a
	multi-byte UTF-8 codepoint becomes multiple %XX triplets, matching
	Python's own quote() applied to a UTF-8-encoded str. '''
	data: ConstPtr[u8] = s.get_const_ptr()
	n: usize = s.byte_len()
	hex_ptr: ConstPtr[u8] = _HEX_UPPER.get_const_ptr()

	out_len: usize = 0
	i: usize = 0
	with compiler.panic_arithmetic( 'irrational string length' ):
		while i < n:
			b: u8 = data[i]
			if space_as_plus and b == _SPACE:
				out_len += 1
			elif _is_unreserved_byte( b ) or _is_safe_byte( b, safe ):
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
			if space_as_plus and b == _SPACE:
				out_ptr[o] = _PLUS
				o += 1
			elif _is_unreserved_byte( b ) or _is_safe_byte( b, safe ):
				out_ptr[o] = b
				o += 1
			else:
				out_ptr[o] = _PERCENT
				out_ptr[o+1] = hex_ptr[ usize( b >> 4 ) ]
				out_ptr[o+2] = hex_ptr[ usize( b & 0x0F ) ]
				o += 3
			i += 1

	return str.from_cstr( move( out )).unwrap( '_quote_impl: invalid UTF-8 (unreachable - output is pure ASCII)' )

def quote( s: str, safe: str = '' ) -> str:
	''' RFC 3986 percent-encoding of s's own UTF-8 bytes - unreserved bytes
	(A-Z a-z 0-9 - _ . ~) and every byte in safe pass through unescaped;
	everything else becomes %XX. '''
	return _quote_impl( s, safe, False )

def quote_plus( s: str, safe: str = '' ) -> str:
	''' like quote(), but additionally encodes space as '+' rather than
	'%20' - the application/x-www-form-urlencoded convention, used by
	urlencode(). '''
	return _quote_impl( s, safe, True )

# ---------------------------------------------------------------------------
# percent-decoding - unquote()/unquote_plus()
# ---------------------------------------------------------------------------

def _hex_digit_value( b: u8 ) -> Result[u8, UrlParseError]:
	if b >= 0x30 and b <= 0x39: # '0'-'9'
		with compiler.wrap_arithmetic:
			return Result.Ok( b - 0x30 )
	if b >= 0x41 and b <= 0x46: # 'A'-'F'
		with compiler.wrap_arithmetic:
			return Result.Ok( b - 0x41 + 10 )
	if b >= 0x61 and b <= 0x66: # 'a'-'f'
		with compiler.wrap_arithmetic:
			return Result.Ok( b - 0x61 + 10 )
	return Result.Err( UrlParseError( 'invalid percent-encoding: not a hex digit' ))

def _unquote_impl( s: str, plus_as_space: bool ) -> Result[str, UrlParseError]:
	''' percent-decodes s's %XX triplets back to raw bytes (optionally
	mapping '+' to space first), then revalidates the result as UTF-8 - the
	exact inverse of _quote_impl's byte-level transform. Two passes (size,
	then fill), the first pass validating every %XX triplet so the second
	pass can decode without re-checking (see decode_chunked in
	lib/http/client.py for the same shape). '''
	data: ConstPtr[u8] = s.get_const_ptr()
	n: usize = s.byte_len()

	out_len: usize = 0
	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by n, cannot overflow' ):
		while i < n:
			b: u8 = data[i]
			if b == _PERCENT:
				if i + 2 >= n:
					return Result.Err( UrlParseError( 'truncated percent-encoding at end of string' ))
				_hex_digit_value( data[i+1] ).or_return()
				_hex_digit_value( data[i+2] ).or_return()
				i += 3
			else:
				i += 1
			out_len += 1
		buf_size: usize = out_len + 1 # zero terminator

	out: bytearray = bytearray( buf_size )
	out_ptr: Ptr[u8] = out.get_ptr()
	o: usize = 0
	i = 0
	with compiler.wrap_arithmetic:
		while i < n:
			b: u8 = data[i]
			if b == _PERCENT:
				hi: u8 = _hex_digit_value( data[i+1] ).unwrap( '_unquote_impl: re-scan after first-pass validation' )
				lo: u8 = _hex_digit_value( data[i+2] ).unwrap( '_unquote_impl: re-scan after first-pass validation' )
				out_ptr[o] = ( hi << 4 ) | lo
				i += 3
			elif plus_as_space and b == _PLUS:
				out_ptr[o] = _SPACE
				i += 1
			else:
				out_ptr[o] = b
				i += 1
			o += 1

	match str.from_cstr( move( out )):
		case Result.Ok( decoded ):
			return Result.Ok( decoded )
		case Result.Err( _ ):
			return Result.Err( UrlParseError( 'invalid UTF-8 after percent-decoding' ))

def unquote( s: str ) -> Result[str, UrlParseError]:
	''' percent-decodes s - the inverse of quote(). '''
	return _unquote_impl( s, False )

def unquote_plus( s: str ) -> Result[str, UrlParseError]:
	''' like unquote(), but also maps '+' back to space first - the inverse
	of quote_plus(), matching application/x-www-form-urlencoded decoding. '''
	return _unquote_impl( s, True )

# ---------------------------------------------------------------------------
# query strings - urlencode()/parse_qsl()
# ---------------------------------------------------------------------------

def urlencode( query: list[tuple[str,str]] ) -> str:
	''' builds an application/x-www-form-urlencoded query string from an
	ordered sequence of (name, value) pairs, joining "name=value" pairs
	with '&' - mirrors Python's urlencode() called on a list of 2-tuples.
	No dict[str,str] overload (see this module's own header docstring) -
	construct list[tuple[str,str]] pairs directly. '''
	parts: list[str] = list[str]()
	n: usize = query.__len__()
	i: usize = 0
	for i in range( n ):
		pair: tuple[str,str] = query.__getitem__( i ).unwrap( 'urlencode: index in bounds by construction' )
		encoded: str = quote_plus( pair[0] ) + '=' + quote_plus( pair[1] )
		parts.append( encoded ).unwrap( 'urlencode: append failed' )
	return '&'.join( parts )

def parse_qsl( qs: str ) -> Result[list[tuple[str,str]], UrlParseError]:
	''' parses an application/x-www-form-urlencoded query string (with or
	without a leading '?') into an ordered list of (name, value) pairs -
	mirrors Python's parse_qsl(). Returns list[tuple[str,str]] rather than
	parse_qs()'s dict[str,list[str]] grouped shape (deferred - see this
	module's header docstring); this is also the shape Python itself
	recommends when duplicate keys or key order matter. Empty pairs
	(a leading/trailing/doubled '&', or an entirely empty qs) are skipped,
	matching Python's default keep_blank_values=False. '''
	trimmed: str = qs.removeprefix( '?' )
	result: list[tuple[str,str]] = list[tuple[str,str]]()
	pieces: list[str] = trimmed.split( '&' )
	n: usize = pieces.__len__()
	i: usize = 0
	for i in range( n ):
		piece: str = pieces.__getitem__( i ).unwrap( 'parse_qsl: index in bounds by construction' )
		if piece.byte_len() == 0:
			continue
		kv: tuple[str,str,str] = piece.partition( '=' )
		key: str = unquote_plus( kv[0] ).or_return()
		value: str = unquote_plus( kv[2] ).or_return()
		result.append( ( key, value )).unwrap( 'parse_qsl: append failed' )
	return Result.Ok( result )

# ---------------------------------------------------------------------------
# URL splitting - SplitResult/urlsplit()/urlunsplit()
# ---------------------------------------------------------------------------

class SplitResult:
	scheme: str
	netloc: str
	path: str
	query: str
	fragment: str

	def __init__( self, scheme: str, netloc: str, path: str, query: str, fragment: str ) -> None:
		self.scheme = scheme
		self.netloc = netloc
		self.path = path
		self.query = query
		self.fragment = fragment

def _is_scheme_char( b: u8 ) -> bool:
	if b >= 0x41 and b <= 0x5A: # A-Z
		return True
	if b >= 0x61 and b <= 0x7A: # a-z
		return True
	if b >= 0x30 and b <= 0x39: # 0-9
		return True
	if b == 0x2B or b == 0x2D or b == 0x2E: # + - .
		return True
	return False

def _is_valid_scheme( s: str ) -> bool:
	''' a scheme token: one letter, then any number of letters/digits/
	'+'/'-'/'.' - matches RFC 3986's scheme grammar (and, deliberately,
	Python's own real quirk of treating anything matching this grammar
	before the first ':' as a scheme, even outside a real "scheme:" URL -
	e.g. "user:pass@host" also parses with scheme='user' in both). '''
	n: usize = s.byte_len()
	if n == 0:
		return False
	cstr: ConstPtr[u8] = s.get_const_ptr()
	first: u8 = cstr[0]
	if not (( first >= 0x41 and first <= 0x5A ) or ( first >= 0x61 and first <= 0x7A )):
		return False
	i: usize = 1
	with compiler.panic_arithmetic( 'bounded by n, cannot overflow' ):
		while i < n:
			if not _is_scheme_char( cstr[i] ):
				return False
			i += 1
	return True

def urlsplit( url: str ) -> SplitResult:
	''' splits url into (scheme, netloc, path, query, fragment), built
	entirely on str.partition()/startswith()/removeprefix() (matching
	lib/http/client.py's parse_status_line/parse_header_line), not manual
	byte scanning. '''
	frag_parts: tuple[str,str,str] = url.partition( '#' )
	rest: str = frag_parts[0]
	fragment: str = frag_parts[2]

	query_parts: tuple[str,str,str] = rest.partition( '?' )
	rest = query_parts[0]
	query: str = query_parts[2]

	scheme: str = ''
	scheme_parts: tuple[str,str,str] = rest.partition( ':' )
	if scheme_parts[1].byte_len() != 0 and _is_valid_scheme( scheme_parts[0] ):
		scheme = scheme_parts[0].lower()
		rest = scheme_parts[2]

	netloc: str = ''
	if rest.startswith( '//' ):
		after_slashes: str = rest.removeprefix( '//' )
		netloc_parts: tuple[str,str,str] = after_slashes.partition( '/' )
		netloc = netloc_parts[0]
		if netloc_parts[1].byte_len() != 0:
			rest = '/' + netloc_parts[2]
		else:
			rest = ''

	return SplitResult( scheme, netloc, rest, query, fragment )

def urlunsplit( parts: SplitResult ) -> str:
	''' reassembles a SplitResult into a URL string - the inverse of
	urlsplit(). Simplified from Python's exact separator rules in one
	respect: this always omits "//" when netloc is empty, even for a
	scheme that conventionally requires it (e.g. an empty-netloc "http"
	SplitResult round-trips as "http:path", not "http:///path") - not a
	case urlsplit() itself ever produces from a real "http://..." URL. '''
	result: str = ''
	if parts.scheme.byte_len() != 0:
		result = result + parts.scheme + ':'
	if parts.netloc.byte_len() != 0:
		result = result + '//' + parts.netloc
	result = result + parts.path
	if parts.query.byte_len() != 0:
		result = result + '?' + parts.query
	if parts.fragment.byte_len() != 0:
		result = result + '#' + parts.fragment
	return result

# ---------------------------------------------------------------------------
# urljoin() - RFC 3986 5.3 relative-reference resolution
# ---------------------------------------------------------------------------

def _remove_dot_segments( path: str ) -> str:
	''' RFC 3986 5.2.4 dot-segment removal. Implemented via split('/') +
	a segment stack + join('/') rather than the RFC's own byte-buffer-with-
	lookahead pseudocode (which needs string slicing str doesn't publicly
	expose here) - the leading '/' of an absolute path and a genuine
	trailing '/' are tracked separately from the segment stack itself
	(is_absolute/trailing_slash below) rather than as poppable stack
	entries, so that excess ".." beyond the root is a no-op instead of
	eating the leading slash - confirmed against every RFC 3986 5.4 worked
	example, including the "abnormal" ../../../g-beyond-root ones. Empty
	segments from a doubled '/' are dropped, same as a real browser/
	CPython's own urljoin. '''
	is_absolute: bool = path.startswith( '/' )
	segments: list[str] = path.split( '/' )
	resolved: list[str] = list[str]()
	n: usize = segments.__len__()
	i: usize = 0
	for i in range( n ):
		seg: str = segments.__getitem__( i ).unwrap( '_remove_dot_segments: index in bounds by construction' )
		if seg.byte_len() == 0:
			continue
		elif seg == '..':
			if resolved.__len__() > 0:
				resolved.pop().unwrap( '_remove_dot_segments: just checked non-empty' )
		elif seg == '.':
			continue
		else:
			resolved.append( seg ).unwrap( '_remove_dot_segments: append failed' )

	trailing_slash: bool = False
	if n > 0:
		with compiler.panic_arithmetic( 'n > 0, just checked above' ):
			last_idx: usize = n - 1
		last: str = segments.__getitem__( last_idx ).unwrap( '_remove_dot_segments: index in bounds by construction' )
		if last.byte_len() == 0 or last == '.' or last == '..':
			trailing_slash = True

	result: str = '/'.join( resolved )
	if is_absolute:
		result = '/' + result
	if trailing_slash and result.byte_len() != 0 and not result.endswith( '/' ):
		result = result + '/'
	if result.byte_len() == 0 and is_absolute:
		result = '/'
	return result

def _merge_paths( base: SplitResult, ref_path: str ) -> str:
	''' RFC 3986 5.3's merge(): combines base's path with a relative ref
	path. When base has a netloc and an empty path (a bare "http://host"
	base), the merged result is "/" + ref_path. Otherwise, everything in
	base's path up to and including its last '/' is kept, with ref_path
	appended in its place - built via rpartition('/') rather than raw
	index slicing, since str has no public slicing operator. '''
	if base.netloc.byte_len() != 0 and base.path.byte_len() == 0:
		return '/' + ref_path
	head_parts: tuple[str,str,str] = base.path.rpartition( '/' )
	if head_parts[1].byte_len() == 0: # no '/' anywhere in base.path
		return ref_path
	return head_parts[0] + '/' + ref_path

def urljoin( base: str, url: str ) -> str:
	''' RFC 3986 5.3 relative-reference resolution: resolves url against
	base, e.g. urljoin('http://a/b/c/d;p?q', '../g') == 'http://a/b/g'.
	Structured to mirror the RFC's own pseudocode nesting directly (see the
	comments on each branch below) rather than Python's internal
	implementation, so each branch is traceable back to the spec. '''
	b: SplitResult = urlsplit( base )
	r: SplitResult = urlsplit( url )

	scheme: str = ''
	netloc: str = ''
	path: str = ''
	query: str = ''

	if r.scheme.byte_len() != 0:
		# reference carries its own scheme: fully absolute, dot-segments
		# in its own path are still normalized (RFC 3986 5.3).
		scheme = r.scheme
		netloc = r.netloc
		path = _remove_dot_segments( r.path )
		query = r.query
	else:
		scheme = b.scheme
		if r.netloc.byte_len() != 0:
			# "//host..." reference: takes over the authority, but not
			# base's path.
			netloc = r.netloc
			path = _remove_dot_segments( r.path )
			query = r.query
		else:
			netloc = b.netloc
			if r.path.byte_len() == 0:
				# fragment-only / query-only / entirely-empty reference:
				# keeps base's path, and base's query unless the
				# reference supplies its own.
				path = b.path
				query = b.query
				if r.query.byte_len() != 0:
					query = r.query
			elif r.path.startswith( '/' ):
				# absolute-path reference: replaces base's path outright.
				path = _remove_dot_segments( r.path )
				query = r.query
			else:
				# relative-path reference: merged against base's path.
				path = _remove_dot_segments( _merge_paths( b, r.path ))
				query = r.query

	return urlunsplit( SplitResult( scheme, netloc, path, query, r.fragment ))
