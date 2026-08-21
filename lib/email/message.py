'''
email.message - metalpy's equivalent of Python's email.message module, with
a minimal bundled email.parser/email.header surface (raw text -> Message,
and RFC 2047 header decoding) folded into this same file rather than split
into separate modules - see PLAN_HTTP_CLIENT.md's own note that
lib/http/client.py's HTTPHeaders was a stand-in for this until a real
email.message landed, and the module docstring convention lib/csv.py set of
bundling reader+writer together.

Scope (v1, deliberately as close to Python's own email.message as this
compiler currently supports):

  - ordered, case-insensitive, duplicate-preserving header multimap
  - Content-Type / Content-Disposition parameter parsing (plain
    key=value / key="value" params only - no RFC 2231 continuations)
  - single-part (str) and multipart (list[Message]) payloads, with
    boundary auto-generation
  - Content-Transfer-Encoding decode (base64, quoted-printable)
  - RFC 2047 encoded-word header decoding (decode_header)
  - message_from_string() parsing (header folding, recursive multipart
    body splitting) and as_string() serialization

Deliberate deviations from Python's own API, each forced by a real compiler
constraint rather than a design preference:

  - list[T]|None (and any other generic-in-union) does not type-check in
    this compiler at all (confirmed - see lib/csv.py's own module
    docstring for the same finding). Python's Message.get_payload(i=None)
    is polymorphic (str body / Message sub-part / decoded bytes,
    optionally None-vs-int i); here that's three separate methods instead
    - get_payload() -> str, get_part(i) -> Result[Message,IndexError],
    get_payload_decoded() -> Result[bytes,EmailError] - so no method needs
    a generic-typed Optional return.
  - walk() returns a fully materialized list[Message], not a lazy
    generator/iterator: a Result-returning __next__ can never be
    for-loop-driven in this language (same restriction lib/csv.py's
    Reader/DictReader hit), and a non-Result generator couldn't signal a
    malformed-structure error either - materializing the whole walk
    sidesteps both problems.
  - Preamble/epilogue text (anything before the first boundary delimiter
    or after the closing one) is parsed but discarded, not preserved -
    real messages essentially never rely on it, and preserving it would
    add a field to every Message for a case v1 doesn't need.
  - Header-line output folding is a simple length-based (78-column) fold,
    not Python's charset-aware email.generator wrapping.
  - RFC 2231 parameter continuations/encoding (filename*0*=, filename*=
    charset''...), email.utils-style address/date parsing, and a
    Policy/compat32 object model are all out of scope for v1.

Multipart boundary detection requires a CRLF immediately before each
"--boundary" delimiter line (matching RFC 5322/2046 exactly); bodies using
bare LF before a boundary delimiter are not recognized - every Message this
module itself produces (as_string()) always uses CRLF, so round-tripping a
message built through this API is unaffected.
'''

import compiler
import sys
import base64
import time
from email._quoprimime import encode as _qp_encode, decode as _qp_decode
from codecs.utf8 import utf8
from codecs.ascii import ascii as _AsciiCodec
from codecs.latin1 import latin1 as _Latin1Codec
from codecs.cp437 import cp437 as _Cp437Codec


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

# @union, not @enum - see lib/http/client.py's own HTTPError comment: a
# CEnum member referenced bare as a real runtime value is a documented
# lowering.py gap, so every variant here is a @union member instead.
@union
class EmailError:
	MalformedHeader: None      # a header line with no ':'
	UnterminatedBoundary: None # multipart body never reached a closing/next boundary
	MissingBoundary: None      # multipart/* Content-Type with no boundary param
	Decode: str                # base64/quoted-printable/charset decode failure (message)


# ---------------------------------------------------------------------------
# _Headers - ordered, case-insensitive, duplicate-preserving multimap.
# Modeled directly on lib/http/client.py's HTTPHeaders (list[tuple[str,str]],
# not dict[K,V] - header order and duplicate entries must survive, and
# dict[K,V] iteration order isn't guaranteed here), kept as its own class in
# this module rather than imported from http.client since email folding
# semantics diverge from HTTP's and the two modules don't otherwise need to
# share state.
# ---------------------------------------------------------------------------

def _header_name_eq( a: str, b: str ) -> bool:
	return a.lower() == b.lower()


class _Headers:
	__entries: list[tuple[str,str]]

	def __init__( self ) -> None:
		self.__entries = list[tuple[str,str]]()

	def __len__( self ) -> usize:
		return self.__entries.__len__()

	def add( self, name: str, value: str ) -> None:
		self.__entries.append( ( name, value )).unwrap( '_Headers.add: append failed' )

	def set( self, name: str, value: str ) -> None:
		''' replaces every existing entry with a matching (case-insensitive)
		name with a single new entry at the first matching position, or
		appends if name wasn't present. '''
		rebuilt: list[tuple[str,str]] = list[tuple[str,str]]()
		replaced: bool = False
		n: usize = self.__entries.__len__()
		i: usize = 0
		for i in range( n ):
			entry: tuple[str,str] = self.__entries.__getitem__( i ).unwrap( '_Headers.set: index in bounds by construction' )
			if _header_name_eq( entry[0], name ):
				if not replaced:
					rebuilt.append( ( name, value )).unwrap( '_Headers.set: append failed' )
					replaced = True
			else:
				rebuilt.append( entry ).unwrap( '_Headers.set: append failed' )
		if not replaced:
			rebuilt.append( ( name, value )).unwrap( '_Headers.set: append failed' )
		self.__entries = rebuilt

	def delete( self, name: str ) -> None:
		''' removes every entry with a matching (case-insensitive) name. '''
		rebuilt: list[tuple[str,str]] = list[tuple[str,str]]()
		n: usize = self.__entries.__len__()
		i: usize = 0
		for i in range( n ):
			entry: tuple[str,str] = self.__entries.__getitem__( i ).unwrap( '_Headers.delete: index in bounds by construction' )
			if not _header_name_eq( entry[0], name ):
				rebuilt.append( entry ).unwrap( '_Headers.delete: append failed' )
		self.__entries = rebuilt

	def get( self, name: str ) -> str|None:
		''' the first entry matching name (case-insensitive), or None. '''
		n: usize = self.__entries.__len__()
		i: usize = 0
		for i in range( n ):
			entry: tuple[str,str] = self.__entries.__getitem__( i ).unwrap( '_Headers.get: index in bounds by construction' )
			if _header_name_eq( entry[0], name ):
				return entry[1]
		return None

	def get_all( self, name: str ) -> list[str]:
		result: list[str] = list[str]()
		n: usize = self.__entries.__len__()
		i: usize = 0
		for i in range( n ):
			entry: tuple[str,str] = self.__entries.__getitem__( i ).unwrap( '_Headers.get_all: index in bounds by construction' )
			if _header_name_eq( entry[0], name ):
				result.append( entry[1] ).unwrap( '_Headers.get_all: append failed' )
		return result

	def contains( self, name: str ) -> bool:
		n: usize = self.__entries.__len__()
		i: usize = 0
		for i in range( n ):
			entry: tuple[str,str] = self.__entries.__getitem__( i ).unwrap( '_Headers.contains: index in bounds by construction' )
			if _header_name_eq( entry[0], name ):
				return True
		return False

	def keys( self ) -> list[str]:
		result: list[str] = list[str]()
		n: usize = self.__entries.__len__()
		i: usize = 0
		for i in range( n ):
			entry: tuple[str,str] = self.__entries.__getitem__( i ).unwrap( '_Headers.keys: index in bounds by construction' )
			result.append( entry[0] ).unwrap( '_Headers.keys: append failed' )
		return result

	def values( self ) -> list[str]:
		result: list[str] = list[str]()
		n: usize = self.__entries.__len__()
		i: usize = 0
		for i in range( n ):
			entry: tuple[str,str] = self.__entries.__getitem__( i ).unwrap( '_Headers.values: index in bounds by construction' )
			result.append( entry[1] ).unwrap( '_Headers.values: append failed' )
		return result

	def items( self ) -> list[tuple[str,str]]:
		result: list[tuple[str,str]] = list[tuple[str,str]]()
		n: usize = self.__entries.__len__()
		i: usize = 0
		for i in range( n ):
			entry: tuple[str,str] = self.__entries.__getitem__( i ).unwrap( '_Headers.items: index in bounds by construction' )
			result.append( entry ).unwrap( '_Headers.items: append failed' )
		return result


# ---------------------------------------------------------------------------
# Content-Type / Content-Disposition parameter parsing
# ---------------------------------------------------------------------------

def _split_unquoted_semicolons( s: str ) -> list[str]:
	''' splits s on ';' that are NOT inside a "..." quoted-string span. '''
	segments: list[str] = list[str]()
	data: ConstPtr[u8] = s.get_const_ptr()
	n: usize = s.byte_len()
	i: usize = 0
	start: usize = 0
	in_quotes: bool = False
	while i < n:
		b: u8 = data[i]
		if b == 0x22: # '"'
			in_quotes = not in_quotes
		elif b == 0x3B and not in_quotes: # ';'
			segments.append( s[start:i] ).unwrap( '_split_unquoted_semicolons: append failed' )
			with compiler.wrap_arithmetic:
				i += 1
			start = i
			continue
		with compiler.wrap_arithmetic:
			i += 1
	segments.append( s[start:n] ).unwrap( '_split_unquoted_semicolons: append failed' )
	return segments


def _strip_quotes( s: str ) -> str:
	t: str = s.strip()
	n: usize = t.byte_len()
	if n >= 2 and t.startswith( '"' ) and t.endswith( '"' ):
		with compiler.wrap_arithmetic:
			last: usize = n - 1
		return t[1:last]
	return t


def _parse_params( value: str ) -> list[tuple[str,str]]:
	''' parses "type/subtype; k=v; k2=\"v2\"" into [(k,v), ...] - the leading
	type/subtype token is skipped, only the "; key=value" parameters are
	returned. One level of quoted-string values is handled (semicolons
	inside quotes don't split); no RFC 2231 continuations/encoding. '''
	segments: list[str] = _split_unquoted_semicolons( value )
	params: list[tuple[str,str]] = list[tuple[str,str]]()
	n: usize = segments.__len__()
	i: usize = 0
	for i in range( 1, n ):
		seg: str = segments.__getitem__( i ).unwrap( '_parse_params: index in bounds by construction' )
		trimmed: str = seg.strip()
		if trimmed.byte_len() == 0:
			continue
		kv: tuple[str,str,str] = trimmed.partition( '=' )
		if kv[1].byte_len() == 0:
			continue # no '=' in this segment - not a real parameter
		key: str = kv[0].strip().lower()
		val: str = _strip_quotes( kv[2] )
		params.append( ( key, val )).unwrap( '_parse_params: append failed' )
	return params


# ---------------------------------------------------------------------------
# boundary generation
# ---------------------------------------------------------------------------

_boundary_counter: u64 = 0

def _generate_boundary() -> str:
	''' a token unique enough for a MIME boundary (not cryptographically
	strong - boundaries don't need that): a per-process monotonic counter
	mixed with the current wall-clock microsecond, hex-encoded. No RNG
	binding exists anywhere in lib/ today. '''
	global _boundary_counter
	with compiler.wrap_arithmetic:
		_boundary_counter += 1
	counter: u64 = _boundary_counter
	t: f64 = time.time()
	with compiler.wrap_arithmetic:
		micros: u64 = u64( t * 1000000.0 )
	buf: bytearray = bytearray( 16 )
	ptr: Ptr[u8] = buf.get_ptr()
	i: usize = 0
	while i < 8:
		with compiler.wrap_arithmetic:
			shift: u64 = u64( i ) * 8
			ptr[i] = u8( ( micros >> shift ) & 0xFF )
			j: usize = i + 8
			ptr[j] = u8( ( counter >> shift ) & 0xFF )
			i += 1
	token: bytes = base64.b16encode( buf )
	token_str: str = token.decode().unwrap( '_generate_boundary: hex token is valid ascii' )
	return '=_MetalPy_' + token_str


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

class Message:
	__headers: _Headers
	__payload: str
	__parts: list[Message]
	__is_multipart: bool

	def __init__( self ) -> None:
		self.__headers = _Headers()
		self.__payload = ''
		self.__parts = list[Message]()
		self.__is_multipart = False

	# --- header access ---------------------------------------------------

	def get( self, name: str ) -> str|None:
		return self.__headers.get( name )

	def get_all( self, name: str ) -> list[str]:
		return self.__headers.get_all( name )

	def __getitem__( self, name: str ) -> str|None:
		return self.__headers.get( name )

	def __setitem__( self, name: str, value: str ) -> None:
		''' ADDS a header - matches Python's own Message.__setitem__, which
		does not overwrite or delete any existing header with the same
		name. Use replace_header() to overwrite. '''
		self.__headers.add( name, value )

	def __delitem__( self, name: str ) -> None:
		self.__headers.delete( name )

	def __contains__( self, name: str ) -> bool:
		return self.__headers.contains( name )

	def add_header( self, name: str, value: str ) -> None:
		self.__headers.add( name, value )

	def replace_header( self, name: str, value: str ) -> None:
		self.__headers.set( name, value )

	def keys( self ) -> list[str]:
		return self.__headers.keys()

	def values( self ) -> list[str]:
		return self.__headers.values()

	def items( self ) -> list[tuple[str,str]]:
		return self.__headers.items()

	# --- Content-Type / Content-Disposition -------------------------------

	def get_content_type( self ) -> str:
		raw: str|None = self.__headers.get( 'Content-Type' )
		if raw is None:
			return 'text/plain'
		parts: tuple[str,str,str] = raw.partition( ';' )
		type_part: str = parts[0].strip().lower()
		if type_part.byte_len() == 0:
			return 'text/plain'
		return type_part

	def get_content_maintype( self ) -> str:
		ct: str = self.get_content_type()
		parts: tuple[str,str,str] = ct.partition( '/' )
		return parts[0]

	def get_content_subtype( self ) -> str:
		ct: str = self.get_content_type()
		parts: tuple[str,str,str] = ct.partition( '/' )
		if parts[1].byte_len() == 0:
			return 'plain'
		return parts[2]

	def get_params( self ) -> list[tuple[str,str]]:
		raw: str|None = self.__headers.get( 'Content-Type' )
		if raw is None:
			empty: list[tuple[str,str]] = list[tuple[str,str]]()
			return empty
		return _parse_params( raw )

	def get_param( self, name: str ) -> str|None:
		params: list[tuple[str,str]] = self.get_params()
		needle: str = name.lower()
		n: usize = params.__len__()
		i: usize = 0
		for i in range( n ):
			p: tuple[str,str] = params.__getitem__( i ).unwrap( 'get_param: index in bounds by construction' )
			if p[0] == needle:
				return p[1]
		return None

	def _get_disposition_param( self, name: str ) -> str|None:
		raw: str|None = self.__headers.get( 'Content-Disposition' )
		if raw is None:
			return None
		params: list[tuple[str,str]] = _parse_params( raw )
		needle: str = name.lower()
		n: usize = params.__len__()
		i: usize = 0
		for i in range( n ):
			p: tuple[str,str] = params.__getitem__( i ).unwrap( '_get_disposition_param: index in bounds by construction' )
			if p[0] == needle:
				return p[1]
		return None

	def get_filename( self ) -> str|None:
		fn: str|None = self._get_disposition_param( 'filename' )
		if fn is not None:
			return fn
		return self.get_param( 'name' ) # Content-Type name= fallback, matches Python

	def get_boundary( self ) -> str|None:
		return self.get_param( 'boundary' )

	def set_boundary( self, boundary: str ) -> None:
		raw: str|None = self.__headers.get( 'Content-Type' )
		base_type: str = 'text/plain'
		if raw is not None:
			parts: tuple[str,str,str] = raw.partition( ';' )
			base_type = parts[0].strip()
		self.__headers.set( 'Content-Type', base_type + '; boundary="' + boundary + '"' )

	def get_charset( self ) -> str|None:
		''' the charset= parameter, lowercased (charset names are
		conventionally case-insensitive - matches Python's own
		Message.get_content_charset(), unlike get_param() itself, which
		preserves a parameter value's original case). '''
		raw: str|None = self.get_param( 'charset' )
		if raw is None:
			return None
		return raw.lower()

	# --- payload / multipart ----------------------------------------------

	def is_multipart( self ) -> bool:
		return self.__is_multipart

	def get_payload( self ) -> str:
		return self.__payload

	def set_payload( self, payload: str ) -> None:
		self.__payload = payload
		self.__is_multipart = False

	def get_part( self, i: usize ) -> Result[Message, IndexError]:
		return self.__parts.__getitem__( i )

	def get_parts( self ) -> list[Message]:
		return self.__parts

	def attach( self, part: Message ) -> None:
		''' appends a sub-message and marks this message multipart. If
		Content-Type isn't already multipart/* yet, defaults it to
		multipart/mixed with a freshly generated boundary - a deliberate
		convenience deviation from Python's own Message.attach (which
		assumes a MIMEMultipart subclass already set Content-Type; this
		module has no such subclass in v1 scope). '''
		self.__parts.append( part ).unwrap( 'Message.attach: append failed' )
		self.__is_multipart = True
		ct: str|None = self.__headers.get( 'Content-Type' )
		is_multipart_ct: bool = False
		if ct is not None:
			if ct.lower().startswith( 'multipart/' ):
				is_multipart_ct = True
		if not is_multipart_ct:
			self.__headers.set( 'Content-Type', 'multipart/mixed; boundary="' + _generate_boundary() + '"' )
		elif self.get_boundary() is None:
			self.set_boundary( _generate_boundary() )

	def walk( self ) -> list[Message]:
		''' self followed by every descendant part, pre-order, fully
		materialized (see the module docstring on why this can't be a lazy
		iterator). '''
		result: list[Message] = list[Message]()
		result.append( self ).unwrap( 'Message.walk: append failed' )
		n: usize = self.__parts.__len__()
		i: usize = 0
		for i in range( n ):
			part: Message = self.__parts.__getitem__( i ).unwrap( 'Message.walk: index in bounds by construction' )
			sub: list[Message] = part.walk()
			m: usize = sub.__len__()
			j: usize = 0
			for j in range( m ):
				item: Message = sub.__getitem__( j ).unwrap( 'Message.walk: sub index in bounds by construction' )
				result.append( item ).unwrap( 'Message.walk: append failed' )
		return result

	def get_payload_decoded( self ) -> Result[bytes, EmailError]:
		''' the payload with Content-Transfer-Encoding undone: base64 or
		quoted-printable decoded to raw bytes, or (identity/unset/unknown
		CTE) the payload's own UTF-8 bytes unchanged. '''
		cte: str|None = self.__headers.get( 'Content-Transfer-Encoding' )
		enc: str = 'identity'
		if cte is not None:
			enc = cte.strip().lower()
		if enc == 'base64':
			raw_bytes: bytes = self.__payload.encode().unwrap( 'get_payload_decoded: payload is not valid utf-8 (unreachable - CTE payload text is always ASCII)' )
			match base64.b64decode( raw_bytes, False ):
				case Result.Ok( decoded ):
					return Result.Ok( decoded )
				case Result.Err( e ):
					return Result.Err( EmailError.Decode( 'base64 decode failed' ))
		elif enc == 'quoted-printable':
			return Result.Ok( _qp_decode( self.__payload ))
		else:
			identity_bytes: bytes = self.__payload.encode().unwrap( 'get_payload_decoded: payload is not valid utf-8' )
			return Result.Ok( identity_bytes )

	# --- serialization ------------------------------------------------

	def as_string( self ) -> str:
		result: str = ''
		names: list[str] = self.__headers.keys()
		values: list[str] = self.__headers.values()
		n: usize = names.__len__()
		i: usize = 0
		for i in range( n ):
			name: str = names.__getitem__( i ).unwrap( 'as_string: name index in bounds by construction' )
			value: str = values.__getitem__( i ).unwrap( 'as_string: value index in bounds by construction' )
			result = result + _fold_header_line( name, value )
		result = result + '\r\n'
		if self.__is_multipart:
			boundary_opt: str|None = self.get_boundary()
			b: str = ''
			if boundary_opt is not None:
				b = boundary_opt
			delim: str = '--' + b
			m: usize = self.__parts.__len__()
			j: usize = 0
			for j in range( m ):
				part: Message = self.__parts.__getitem__( j ).unwrap( 'as_string: part index in bounds by construction' )
				result = result + delim + '\r\n' + part.as_string() + '\r\n'
			result = result + delim + '--\r\n'
		else:
			result = result + self.__payload
		return result


def _fold_header_line( name: str, value: str ) -> str:
	''' "Name: value\\r\\n", folded at whitespace if the line would exceed
	78 columns - a simple length-based fold, not Python's charset-aware
	email.generator wrapping (see module docstring). '''
	line: str = name + ': ' + value
	if line.byte_len() <= 78:
		return line + '\r\n'
	result: str = ''
	remaining: str = line
	while remaining.byte_len() > 78:
		limit: usize = 78
		cut: usize = 0
		found_cut: bool = False
		i: usize = limit
		while i > 0:
			with compiler.wrap_arithmetic:
				prev: usize = i - 1
			ch: str = remaining[prev:i]
			if ch == ' ':
				cut = i
				found_cut = True
				break
			i = prev
		if not found_cut:
			break
		piece: str = remaining[0:cut]
		rest: str = remaining[cut:remaining.byte_len()]
		result = result + piece + '\r\n'
		remaining = ' ' + rest.strip()
	result = result + remaining + '\r\n'
	return result


# ---------------------------------------------------------------------------
# RFC 2047 encoded-word header decoding
# ---------------------------------------------------------------------------

class _EncodedWord:
	ok: bool
	charset: str
	encoding: str # 'B' or 'Q'
	text: str
	end: usize    # index just past the encoded-word's closing "?="

	def __init__( self, ok: bool, charset: str, encoding: str, text: str, end: usize ) -> None:
		self.ok = ok
		self.charset = charset
		self.encoding = encoding
		self.text = text
		self.end = end


def _try_parse_encoded_word( value: str, start: usize ) -> _EncodedWord:
	''' value[start:start+2] is already known to be "=?". Attempts to parse
	=?charset?B-or-Q?encoded-text?= starting there. '''
	n: usize = value.byte_len()
	with compiler.wrap_arithmetic:
		p1: usize = start + 2
	q1_found: isize = value.find( '?', p1 )
	if q1_found == isize( -1 ):
		return _EncodedWord( False, '', '', '', start )
	with compiler.panic_arithmetic( 'find() never returns a negative offset once the -1/not-found case is excluded' ):
		q1: usize = usize( q1_found )
	charset: str = value[p1:q1]
	if charset.byte_len() == 0:
		return _EncodedWord( False, '', '', '', start )
	with compiler.wrap_arithmetic:
		p2: usize = q1 + 1
	if p2 >= n:
		return _EncodedWord( False, '', '', '', start )
	with compiler.wrap_arithmetic:
		p2_end: usize = p2 + 1
	enc_char: str = value[p2:p2_end]
	p3: usize = p2_end
	if p3 >= n:
		return _EncodedWord( False, '', '', '', start )
	with compiler.wrap_arithmetic:
		p3_end: usize = p3 + 1
	if value[p3:p3_end] != '?':
		return _EncodedWord( False, '', '', '', start )
	with compiler.wrap_arithmetic:
		p4: usize = p3 + 1
	term_found: isize = value.find( '?=', p4 )
	if term_found == isize( -1 ):
		return _EncodedWord( False, '', '', '', start )
	with compiler.panic_arithmetic( 'find() never returns a negative offset once the -1/not-found case is excluded' ):
		term: usize = usize( term_found )
	text: str = value[p4:term]
	with compiler.wrap_arithmetic:
		endpos: usize = term + 2
	enc_upper: str = enc_char.upper()
	if enc_upper != 'B' and enc_upper != 'Q':
		return _EncodedWord( False, '', '', '', start )
	return _EncodedWord( True, charset, enc_upper, text, endpos )


def _decode_encoded_word_bytes( ew: _EncodedWord ) -> Result[bytes, EmailError]:
	if ew.encoding == 'B':
		raw: bytes = ew.text.encode().unwrap( '_decode_encoded_word_bytes: not valid utf-8 (unreachable - base64 alphabet is ASCII)' )
		match base64.b64decode( raw, False ):
			case Result.Ok( decoded ):
				return Result.Ok( decoded )
			case Result.Err( e ):
				return Result.Err( EmailError.Decode( 'RFC 2047 base64 decode failed' ))
	else:
		underscored: str = ew.text.replace( '_', ' ' )
		return Result.Ok( _qp_decode( underscored ))


def _decode_charset_bytes( data: bytes, charset: str ) -> str:
	''' decodes data using the named charset - dispatched directly against
	the concrete codec classes (utf8/ascii/latin1/cp437), NOT
	codecs.Codec.get()'s registry: calling an abstract @virtual method
	(names()) through a value statically typed as the abstract Codec base
	- exactly what Codec.get()/register() do internally - hits a real,
	previously-latent compiler crash lowering the abstract method's `...`
	body (confirmed via a minimal repro; not something this module can fix
	at the source level). Every call below is on a concretely-typed codec
	instance instead, which sidesteps it entirely. Falls back to latin1 for
	an unrecognized charset name, matching this function's own
	always-succeeds contract (every byte value is a valid Latin-1
	codepoint, so latin1 decode is infallible). '''
	name: str = charset.lower()
	if name == 'utf-8' or name == 'utf8':
		match utf8.decode( data ):
			case Result.Ok( s ):
				return s
			case Result.Err( e ):
				pass
	elif name == 'us-ascii' or name == 'ascii':
		a: _AsciiCodec = _AsciiCodec()
		match a.decode( data ):
			case Result.Ok( s2 ):
				return s2
			case Result.Err( e2 ):
				pass
	elif name == 'cp437':
		c: _Cp437Codec = _Cp437Codec()
		match c.decode( data ):
			case Result.Ok( s3 ):
				return s3
			case Result.Err( e3 ):
				pass
	l: _Latin1Codec = _Latin1Codec()
	return l.decode( data ).unwrap( '_decode_charset_bytes: latin1 decode is infallible' )


def _is_all_whitespace( s: str ) -> bool:
	data: ConstPtr[u8] = s.get_const_ptr()
	n: usize = s.byte_len()
	i: usize = 0
	while i < n:
		b: u8 = data[i]
		if b != 0x20 and b != 0x09 and b != 0x0D and b != 0x0A:
			return False
		with compiler.wrap_arithmetic:
			i += 1
	return True


def decode_header( value: str ) -> Result[str, EmailError]:
	''' decodes RFC 2047 encoded-words (=?charset?B-or-Q?text?=) in a header
	value, concatenating adjacent encoded-words (separated only by linear
	whitespace, which is dropped) per RFC 2047 section 2. A malformed
	candidate ("=?" with no valid encoded-word after it) is passed through
	as literal text rather than treated as an error. '''
	result: str = ''
	n: usize = value.byte_len()
	i: usize = 0
	prev_was_encoded: bool = False
	while i < n:
		found_signed: isize = value.find( '=?', i )
		has_more: bool = found_signed != isize( -1 )
		found_at: usize = 0
		if has_more:
			with compiler.panic_arithmetic( 'find() never returns a negative offset once the -1/not-found case is excluded' ):
				found_at = usize( found_signed )
		if not has_more:
			result = result + value[i:n]
			break
		if found_at > i:
			gap: str = value[i:found_at]
			if prev_was_encoded and _is_all_whitespace( gap ):
				pass # drop inter-encoded-word whitespace per RFC 2047 sec. 2
			else:
				result = result + gap
				prev_was_encoded = False
		ew: _EncodedWord = _try_parse_encoded_word( value, found_at )
		if not ew.ok:
			result = result + '=?'
			with compiler.wrap_arithmetic:
				i = found_at + 2
			prev_was_encoded = False
			continue
		decoded_bytes: bytes = _decode_encoded_word_bytes( ew ).or_return()
		decoded_text: str = _decode_charset_bytes( decoded_bytes, ew.charset )
		result = result + decoded_text
		prev_was_encoded = True
		i = ew.end
	return Result.Ok( result )


# ---------------------------------------------------------------------------
# parsing - message_from_string
# ---------------------------------------------------------------------------

def _split_lines( block: str ) -> list[str]:
	normalized: str = block.replace( '\r\n', '\n' )
	return normalized.split( '\n' )


def _unfold_headers( lines: list[str] ) -> list[str]:
	''' joins RFC 5322 folded continuation lines (leading space/tab) onto
	the previous header line, collapsing the fold to a single space. '''
	result: list[str] = list[str]()
	n: usize = lines.__len__()
	i: usize = 0
	for i in range( n ):
		line: str = lines.__getitem__( i ).unwrap( '_unfold_headers: index in bounds by construction' )
		is_continuation: bool = False
		if line.byte_len() > 0:
			first: str = line[0:1]
			if first == ' ' or first == '\t':
				is_continuation = True
		if is_continuation and result.__len__() > 0:
			with compiler.wrap_arithmetic:
				last_idx: usize = result.__len__() - 1
			prev: str = result.__getitem__( last_idx ).unwrap( '_unfold_headers: prev index in bounds by construction' )
			folded: str = prev + ' ' + line.strip()
			result.__setitem__( last_idx, folded ).unwrap( '_unfold_headers: setitem failed' )
		else:
			result.append( line ).unwrap( '_unfold_headers: append failed' )
	return result


def _parse_header_line( line: str ) -> Result[tuple[str,str], EmailError]:
	parts: tuple[str,str,str] = line.partition( ':' )
	sep: str = parts[1]
	if sep.byte_len() == 0:
		return Result.Err( EmailError.MalformedHeader( None ))
	name: str = parts[0].strip()
	value: str = parts[2].strip()
	return Result.Ok( ( name, value ))


def _split_multipart( body: str, boundary: str ) -> Result[list[Message], EmailError]:
	delim: str = '--' + boundary
	close_delim: str = delim + '--'
	parts: list[Message] = list[Message]()

	pos: usize = 0
	has_first: bool = False
	if body.startswith( delim ):
		has_first = True
		pos = 0
	else:
		first_signed: isize = body.find( '\r\n' + delim )
		if first_signed != isize( -1 ):
			has_first = True
			with compiler.panic_arithmetic( 'find() never returns a negative offset once the -1/not-found case is excluded' ):
				pos = usize( first_signed ) + 2
	if not has_first:
		return Result.Err( EmailError.UnterminatedBoundary( None ))

	while True:
		if body.startswith( close_delim, pos ):
			return Result.Ok( parts )
		if not body.startswith( delim, pos ):
			return Result.Err( EmailError.UnterminatedBoundary( None ))
		with compiler.wrap_arithmetic:
			line_end: usize = pos + delim.byte_len()
		part_start: usize = 0
		nl_signed: isize = body.find( '\n', line_end )
		if nl_signed == isize( -1 ):
			return Result.Err( EmailError.UnterminatedBoundary( None ))
		with compiler.panic_arithmetic( 'find() never returns a negative offset once the -1/not-found case is excluded' ):
			part_start = usize( nl_signed ) + 1
		next_signed: isize = body.find( '\r\n' + delim, part_start )
		if next_signed == isize( -1 ):
			return Result.Err( EmailError.UnterminatedBoundary( None ))
		with compiler.panic_arithmetic( 'find() never returns a negative offset once the -1/not-found case is excluded' ):
			next_pos: usize = usize( next_signed )
		part_text: str = body[part_start:next_pos]
		part_msg: Message = message_from_string( part_text ).or_return()
		parts.append( part_msg ).unwrap( '_split_multipart: append failed' )
		with compiler.wrap_arithmetic:
			pos = next_pos + 2


def message_from_string( raw: str ) -> Result[Message, EmailError]:
	''' parses raw RFC 5322 message text (headers, blank line, body) into a
	Message, recursively splitting a multipart/* body into attached
	sub-messages. '''
	msg: Message = Message()
	header_block: str = raw
	body: str = ''
	if raw.startswith( '\r\n' ):
		# zero headers - the blank line separator is the very first thing in
		# raw, so there's no LEADING header line to make a "\r\n\r\n" pair
		# out of (this is exactly what a part with no headers of its own
		# serializes to - see Message.as_string()). Handled as its own case
		# rather than folded into the search below.
		header_block = ''
		body = raw[2:raw.byte_len()]
	elif raw.startswith( '\n' ):
		header_block = ''
		body = raw[1:raw.byte_len()]
	else:
		blank_signed: isize = raw.find( '\r\n\r\n' )
		if blank_signed != isize( -1 ):
			with compiler.panic_arithmetic( 'find() never returns a negative offset once the -1/not-found case is excluded' ):
				idx: usize = usize( blank_signed )
			header_block = raw[0:idx]
			with compiler.wrap_arithmetic:
				body_start: usize = idx + 4
			body = raw[body_start:raw.byte_len()]
		else:
			blank_signed2: isize = raw.find( '\n\n' )
			if blank_signed2 != isize( -1 ):
				with compiler.panic_arithmetic( 'find() never returns a negative offset once the -1/not-found case is excluded' ):
					idx2: usize = usize( blank_signed2 )
				header_block = raw[0:idx2]
				with compiler.wrap_arithmetic:
					body_start2: usize = idx2 + 2
				body = raw[body_start2:raw.byte_len()]
			else:
				header_block = raw
				body = ''

	lines: list[str] = _split_lines( header_block )
	unfolded: list[str] = _unfold_headers( lines )
	m: usize = unfolded.__len__()
	i: usize = 0
	for i in range( m ):
		line: str = unfolded.__getitem__( i ).unwrap( 'message_from_string: index in bounds by construction' )
		if line.byte_len() == 0:
			continue
		parsed: tuple[str,str] = _parse_header_line( line ).or_return()
		msg.add_header( parsed[0], parsed[1] )

	ct: str = msg.get_content_type()
	if ct.startswith( 'multipart/' ):
		boundary_opt: str|None = msg.get_boundary()
		if boundary_opt is None:
			return Result.Err( EmailError.MissingBoundary( None ))
		parts: list[Message] = _split_multipart( body, boundary_opt ).or_return()
		nparts: usize = parts.__len__()
		j: usize = 0
		for j in range( nparts ):
			part: Message = parts.__getitem__( j ).unwrap( 'message_from_string: part index in bounds by construction' )
			msg.attach( part )
	else:
		msg.set_payload( body )
	return Result.Ok( msg )
