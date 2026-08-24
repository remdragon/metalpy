# json.py — JSON parsing, construction/manipulation, path access, and
# flatten/unflatten for MetalPy programs (RFC 8259).
#
# JSONValue is a @union (a value type, copied by tag+payload at assignment -
# same shape as Result[T,E]), not an RC class - see cfg.py's own documented
# rule that a TaggedUnion WITH RC leaves (String/Array/String's str, Array's
# list[JSONValue], Object's dict[str,JSONValue]) is correctly incref/decref
# tracked, exactly like Result[str,E] already relies on. Array/Object hold
# list[JSONValue]/dict[str,JSONValue] as SELF-REFERENTIAL generic arguments -
# confirmed to actually compile via a standalone spike (see json_test.py's
# own first test case) before the rest of this module was built on top of
# it; this works because an RC element's own field slot is always pointer-
# width regardless of the field's nominal type (same reasoning UnsafeList's
# own RawList storage relies on), so sizeof(JSONValue) stays finite despite
# the textual self-reference.
#
# Numbers are two separate variants - Int (arbitrary-precision) and Float
# (f64) - rather than one f64-only Number, so a JSON integer round-trips
# through loads()/dumps() without a decimal point and without the ~2^53
# precision loss an f64-only model would have. The parser decides Int vs
# Float purely from the literal's own text: any '.'/'e'/'E' forces Float,
# otherwise Int - so `1e10`, despite being mathematically integral, dumps
# back out as `1e10`/`10000000000.0`-shaped float text, never bare digits.
#
# flatten()/get_path()/set_path() share one path syntax - lodash/JS-style,
# dot-separated object keys with bracket-indexed arrays ("a.b[0].c"). Known
# v1 limitation, deliberate (confirmed with the module's own requester): an
# object key that itself contains '.', '[', or ']' is NOT escaped - flatten()
# still emits a best-effort path for it, but that path is not guaranteed to
# parse back to the original key via get_path()/unflatten().

import builtins
import compiler
import sys

# ---------------------------------------------------------------------------
# ASCII byte constants - the JSON grammar (RFC 8259) is itself ASCII-only
# for every structural token; only string CONTENTS need real UTF-8 handling
# (via builtins.decode_utf8_at/encode_utf8_at/utf8_encoded_len below).
# ---------------------------------------------------------------------------

_ASCII_TAB:       u8 = 0x09
_ASCII_LF:        u8 = 0x0A
_ASCII_CR:        u8 = 0x0D
_ASCII_SPACE:     u8 = 0x20
_ASCII_QUOTE:     u8 = 0x22 # '"'
_ASCII_PLUS:      u8 = 0x2B
_ASCII_COMMA:     u8 = 0x2C
_ASCII_MINUS:     u8 = 0x2D
_ASCII_DOT:       u8 = 0x2E
_ASCII_SLASH:     u8 = 0x2F
_ASCII_ZERO:      u8 = 0x30
_ASCII_NINE:      u8 = 0x39
_ASCII_COLON:     u8 = 0x3A
_ASCII_UPPER_A:   u8 = 0x41
_ASCII_UPPER_E:   u8 = 0x45
_ASCII_UPPER_F:   u8 = 0x46
_ASCII_LBRACKET:  u8 = 0x5B
_ASCII_BACKSLASH: u8 = 0x5C
_ASCII_RBRACKET:  u8 = 0x5D
_ASCII_LOWER_A:   u8 = 0x61
_ASCII_LOWER_B:   u8 = 0x62
_ASCII_LOWER_E:   u8 = 0x65
_ASCII_LOWER_F:   u8 = 0x66
_ASCII_LOWER_L:   u8 = 0x6C
_ASCII_LOWER_N:   u8 = 0x6E
_ASCII_LOWER_R:   u8 = 0x72
_ASCII_LOWER_S:   u8 = 0x73
_ASCII_LOWER_T:   u8 = 0x74
_ASCII_LOWER_U:   u8 = 0x75
_ASCII_LBRACE:    u8 = 0x7B
_ASCII_RBRACE:    u8 = 0x7D

def _is_ascii_digit( c: u8 ) -> bool:
	return c >= _ASCII_ZERO and c <= _ASCII_NINE


# ---------------------------------------------------------------------------
# 1. JSONValue / JSONError
# ---------------------------------------------------------------------------

@union
class JSONValue:
	Null:   None
	Bool:   bool
	Int:    int
	Float:  f64
	String: str
	Array:  list[JSONValue]
	Object: dict[str, JSONValue]

	@staticmethod
	def null() -> JSONValue:
		return JSONValue.Null( None )

	@staticmethod
	def from_bool( value: bool ) -> JSONValue:
		return JSONValue.Bool( value )

	@staticmethod
	def from_int( value: int ) -> JSONValue:
		return JSONValue.Int( value )

	@staticmethod
	def from_float( value: f64 ) -> JSONValue:
		return JSONValue.Float( value )

	@staticmethod
	def from_str( value: str ) -> JSONValue:
		return JSONValue.String( value )

	@staticmethod
	def array() -> JSONValue:
		return JSONValue.Array( list[JSONValue]() )

	@staticmethod
	def object() -> JSONValue:
		return JSONValue.Object( dict[str, JSONValue]() )

	def is_null( self ) -> bool:
		return self.tag == 0

	def as_bool( self ) -> Result[bool, JSONError]:
		match self:
			case JSONValue.Bool( b ):
				return Result.Ok( b )
			case _:
				return Result.Err( JSONError.WrongType( None ))

	def as_int( self ) -> Result[int, JSONError]:
		match self:
			case JSONValue.Int( n ):
				return Result.Ok( n )
			case _:
				return Result.Err( JSONError.WrongType( None ))

	def as_float( self ) -> Result[f64, JSONError]:
		match self:
			case JSONValue.Float( f ):
				return Result.Ok( f )
			case _:
				return Result.Err( JSONError.WrongType( None ))

	def as_str( self ) -> Result[str, JSONError]:
		match self:
			case JSONValue.String( s ):
				return Result.Ok( s )
			case _:
				return Result.Err( JSONError.WrongType( None ))

	def as_array( self ) -> Result[list[JSONValue], JSONError]:
		match self:
			case JSONValue.Array( arr ):
				return Result.Ok( arr )
			case _:
				return Result.Err( JSONError.WrongType( None ))

	def as_object( self ) -> Result[dict[str, JSONValue], JSONError]:
		match self:
			case JSONValue.Object( obj ):
				return Result.Ok( obj )
			case _:
				return Result.Err( JSONError.WrongType( None ))

	def array_len( self ) -> Result[usize, JSONError]:
		arr = self.as_array().or_return()
		return Result.Ok( len( arr ))

	def array_append( self, value: JSONValue ) -> Result[None, JSONError]:
		arr = self.as_array().or_return()
		arr.append( value )
		return Result.Ok( None )

	def array_get( self, index: usize ) -> Result[JSONValue, JSONError]:
		arr = self.as_array().or_return()
		match arr.__getitem__( index ):
			case Result.Ok( v ):
				return Result.Ok( v )
			case Result.Err( _ ):
				return Result.Err( JSONError.IndexOutOfBounds( None ))

	def array_set( self, index: usize, value: JSONValue ) -> Result[None, JSONError]:
		arr = self.as_array().or_return()
		match arr.__setitem__( index, value ):
			case Result.Ok( _ ):
				return Result.Ok( None )
			case Result.Err( _ ):
				return Result.Err( JSONError.IndexOutOfBounds( None ))

	def object_len( self ) -> Result[usize, JSONError]:
		obj = self.as_object().or_return()
		return Result.Ok( len( obj ))

	def object_set( self, key: str, value: JSONValue ) -> Result[None, JSONError]:
		obj = self.as_object().or_return()
		obj.__setitem__( key, value )
		return Result.Ok( None )

	def object_get( self, key: str ) -> Result[JSONValue, JSONError]:
		obj = self.as_object().or_return()
		match obj.__getitem__( key ):
			case Result.Ok( v ):
				return Result.Ok( v )
			case Result.Err( _ ):
				return Result.Err( JSONError.KeyNotFound( None ))

	# --- path access (get_path/set_path share _tokenize_path/PathToken,
	# defined later in this file alongside flatten/unflatten - a method
	# referencing a plain function/type declared later in the same file is
	# fine, same precedent as lib/guid.py's GUID.from_str calling
	# _parse_hex_u32 et al, which are defined after the class too) ---------

	def get_path( self, path: str ) -> Result[JSONValue, JSONError]:
		tokens = _tokenize_path( path ).or_return()
		current: JSONValue = self
		i: usize = 0
		n: usize = len( tokens )
		while i < n:
			tok: PathToken = tokens.__getitem__( i ).unwrap( 'get_path: index in bounds by construction' )
			match tok:
				case PathToken.Key( k ):
					current = current.object_get( k ).or_return()
				case PathToken.Index( idx ):
					current = current.array_get( idx ).or_return()
			with compiler.panic_arithmetic( 'walking a token list of known length index-by-index cannot overflow usize' ):
				i += 1
		return Result.Ok( current )

	# Requires self already be a container (Object for a leading Key token,
	# Array for a leading Index token) - does NOT auto-vivify the ROOT from
	# Null, unlike unflatten()'s own free-function root construction. There's
	# no confirmed precedent anywhere in this codebase of a @union/@struct
	# instance method reassigning its own `self` to a different variant and
	# having that be visible to the caller (every self-mutation example found
	# is on an ordinary RCClass receiver) - sidestepped entirely here, since
	# _set_path_tokens only ever mutates an ALREADY-OWNED list/dict handle in
	# place (see array_get/object_get's own aliasing note), never needing to
	# reassign self. unflatten() needs a genuinely fresh root, so it builds
	# one as a plain local before ever calling _set_path_tokens.
	def set_path( self, path: str, value: JSONValue ) -> Result[None, JSONError]:
		tokens = _tokenize_path( path ).or_return()
		if len( tokens ) == 0:
			return Result.Err( JSONError.InvalidPath( 0 ))
		return _set_path_tokens( self, tokens, 0, value )


@union
class JSONError:
	# construction / accessor errors
	WrongType:        None
	KeyNotFound:       None
	IndexOutOfBounds:  None
	Overflow:          None
	NonFiniteFloat:    None
	# path errors (get_path/set_path/unflatten) - payload is the byte offset
	# INTO THE PATH STRING where the problem was found
	InvalidPath:       usize
	PathTypeMismatch:  usize
	# parse errors (loads()) - payload is the byte offset INTO THE JSON TEXT
	UnexpectedEnd:     usize
	UnexpectedChar:    usize
	InvalidEscape:     usize
	InvalidNumber:     usize
	TrailingGarbage:   usize


# ---------------------------------------------------------------------------
# 2. loads() - recursive-descent parser
#
# Each _parse_X takes the input string plus a byte cursor and returns
# Result[tuple[X,usize], JSONError] - the parsed value/text paired with the
# cursor position just past what it consumed. tuple[T,T] return (rather than
# a Ptr[usize] out-param) mirrors int.py's own divmod()/_to_radix_digits'
# established idiom for "value + new position" returns in this codebase.
# ---------------------------------------------------------------------------

def loads( s: str ) -> Result[JSONValue, JSONError]:
	parsed = _parse_value( s, 0 ).or_return()
	pos: usize = _skip_ws( s, parsed[1] )
	if pos != s.byte_len():
		return Result.Err( JSONError.TrailingGarbage( pos ))
	return Result.Ok( parsed[0] )


def _skip_ws( s: str, pos: usize ) -> usize:
	data: ConstPtr[u8] = s.get_cstr()
	length: usize = s.byte_len()
	with compiler.panic_arithmetic( 'walking a string of known length index-by-index cannot overflow usize' ):
		i: usize = pos
		while i < length:
			ch: u8 = data[i]
			if ch != _ASCII_SPACE and ch != _ASCII_TAB and ch != _ASCII_LF and ch != _ASCII_CR:
				return i
			i += 1
		return i


def _parse_value( s: str, pos: usize ) -> Result[tuple[JSONValue, usize], JSONError]:
	p: usize = _skip_ws( s, pos )
	length: usize = s.byte_len()
	if p >= length:
		return Result.Err( JSONError.UnexpectedEnd( p ))
	data: ConstPtr[u8] = s.get_cstr()
	ch: u8 = data[p]
	if ch == _ASCII_LBRACE:
		return _parse_object( s, p )
	if ch == _ASCII_LBRACKET:
		return _parse_array( s, p )
	if ch == _ASCII_QUOTE:
		decoded = _parse_string( s, p ).or_return()
		return Result.Ok(( JSONValue.String( decoded[0] ), decoded[1] ))
	if ch == _ASCII_MINUS or _is_ascii_digit( ch ):
		return _parse_number( s, p )
	if ch == _ASCII_LOWER_T:
		return _parse_literal( s, p, 'true', JSONValue.Bool( True ))
	if ch == _ASCII_LOWER_F:
		return _parse_literal( s, p, 'false', JSONValue.Bool( False ))
	if ch == _ASCII_LOWER_N:
		return _parse_literal( s, p, 'null', JSONValue.Null( None ))
	return Result.Err( JSONError.UnexpectedChar( p ))


def _parse_literal( s: str, pos: usize, literal: str, value: JSONValue ) -> Result[tuple[JSONValue, usize], JSONError]:
	length: usize = s.byte_len()
	with compiler.panic_arithmetic( 'a fixed-length literal probe bounded by the input length cannot overflow usize' ):
		end: usize = pos + literal.byte_len()
	if end > length:
		return Result.Err( JSONError.UnexpectedEnd( pos ))
	if s[pos:end] != literal:
		return Result.Err( JSONError.UnexpectedChar( pos ))
	return Result.Ok(( value, end ))


def _parse_number( s: str, pos: usize ) -> Result[tuple[JSONValue, usize], JSONError]:
	data: ConstPtr[u8] = s.get_cstr()
	length: usize = s.byte_len()
	start: usize = pos
	p: usize = pos
	if p < length and data[p] == _ASCII_MINUS:
		with compiler.panic_arithmetic( 'advancing one byte past a known-in-bounds minus sign cannot overflow usize' ):
			p += 1
	if p >= length or not _is_ascii_digit( data[p] ):
		return Result.Err( JSONError.InvalidNumber( start ))
	if data[p] == _ASCII_ZERO:
		with compiler.panic_arithmetic( 'advancing one byte past a known-in-bounds digit cannot overflow usize' ):
			p += 1
		if p < length and _is_ascii_digit( data[p] ):
			# a leading zero followed by another digit ("01") is rejected
			# explicitly here, as its own InvalidNumber - otherwise the scan
			# would just stop after the lone "0" and the extra digit(s)
			# would surface one level up as unrelated TrailingGarbage,
			# which is misleading for what's really a malformed literal.
			return Result.Err( JSONError.InvalidNumber( start ))
	else:
		with compiler.panic_arithmetic( 'walking a string of known length index-by-index cannot overflow usize' ):
			while p < length and _is_ascii_digit( data[p] ):
				p += 1
	saw_dot: bool = False
	saw_exp: bool = False
	if p < length and data[p] == _ASCII_DOT:
		saw_dot = True
		with compiler.panic_arithmetic( 'advancing one byte past a known-in-bounds dot cannot overflow usize' ):
			p += 1
		if p >= length or not _is_ascii_digit( data[p] ):
			return Result.Err( JSONError.InvalidNumber( start ))
		with compiler.panic_arithmetic( 'walking a string of known length index-by-index cannot overflow usize' ):
			while p < length and _is_ascii_digit( data[p] ):
				p += 1
	if p < length and ( data[p] == _ASCII_LOWER_E or data[p] == _ASCII_UPPER_E ):
		saw_exp = True
		with compiler.panic_arithmetic( 'advancing one byte past a known-in-bounds exponent marker cannot overflow usize' ):
			p += 1
		if p < length and ( data[p] == _ASCII_PLUS or data[p] == _ASCII_MINUS ):
			with compiler.panic_arithmetic( 'advancing one byte past a known-in-bounds exponent sign cannot overflow usize' ):
				p += 1
		if p >= length or not _is_ascii_digit( data[p] ):
			return Result.Err( JSONError.InvalidNumber( start ))
		with compiler.panic_arithmetic( 'walking a string of known length index-by-index cannot overflow usize' ):
			while p < length and _is_ascii_digit( data[p] ):
				p += 1
	literal: str = s[start:p]
	if saw_dot or saw_exp:
		f: f64 = compiler.parse_f64( literal.get_cstr() )
		return Result.Ok(( JSONValue.Float( f ), p ))
	match int.from_str( literal ):
		case Result.Ok( n ):
			return Result.Ok(( JSONValue.Int( n ), p ))
		case Result.Err( _ ):
			return Result.Err( JSONError.InvalidNumber( start ))


# --- string parsing: two-pass size-then-fill, mirroring
# lib/builtins/__str.py's ascii_escape_width/ascii_escape_one split. ---------

def _parse_string( s: str, pos: usize ) -> Result[tuple[str, usize], JSONError]:
	# pos points AT the opening '"'.
	data: ConstPtr[u8] = s.get_cstr()
	length: usize = s.byte_len()
	measured = _measure_string( data, length, pos ).or_return()
	new_size: usize = measured[0]
	end_pos: usize = measured[1]

	with compiler.panic_arithmetic( 'a measured string content size plus one zero terminator cannot overflow usize' ):
		buf_size: usize = new_size + 1
	buf: Ptr[u8] = sys.alloc[u8]( buf_size )
	_fill_string( data, length, pos, buf )
	buf[new_size] = 0

	built: str = str.from_cstr( compiler.cast( ConstPtr[u8], buf ), buf_size ).unwrap( 'json._parse_string: internal buffer was not valid UTF-8 (unreachable - only input UTF-8 bytes and computed escape codepoints are ever written)' )
	sys.free( buf )
	return Result.Ok(( built, end_pos ))


def _measure_string( data: ConstPtr[u8], length: usize, start: usize ) -> Result[tuple[usize, usize], JSONError]:
	with compiler.panic_arithmetic( 'advancing one byte past a known-in-bounds quote cannot overflow usize' ):
		i: usize = start + 1
	new_size: usize = 0
	while i < length:
		ch: u8 = data[i]
		if ch == _ASCII_QUOTE:
			with compiler.panic_arithmetic( 'advancing one byte past a known-in-bounds quote cannot overflow usize' ):
				return Result.Ok(( new_size, i + 1 ))
		if ch < _ASCII_SPACE:
			return Result.Err( JSONError.InvalidEscape( i ))
		if ch == _ASCII_BACKSLASH:
			with compiler.panic_arithmetic( 'advancing one byte past a known-in-bounds backslash cannot overflow usize' ):
				i += 1
			if i >= length:
				return Result.Err( JSONError.UnexpectedEnd( i ))
			esc: u8 = data[i]
			if ( esc == _ASCII_QUOTE or esc == _ASCII_BACKSLASH or esc == _ASCII_SLASH
					or esc == _ASCII_LOWER_B or esc == _ASCII_LOWER_F or esc == _ASCII_LOWER_N
					or esc == _ASCII_LOWER_R or esc == _ASCII_LOWER_T ):
				with compiler.panic_arithmetic( 'a single-char escape increment cannot overflow usize' ):
					new_size += 1
					i += 1
			elif esc == _ASCII_LOWER_U:
				with compiler.panic_arithmetic( 'advancing one byte past a known-in-bounds u-escape marker cannot overflow usize' ):
					hex_start: usize = i + 1
				decoded = _decode_u_escape( data, length, hex_start ).or_return()
				with compiler.panic_arithmetic( 'a decoded codepoints own utf8 width added to a running usize total cannot overflow' ):
					new_size += builtins.utf8_encoded_len( decoded[0] )
				i = decoded[1]
			else:
				return Result.Err( JSONError.InvalidEscape( i ))
		else:
			with compiler.panic_arithmetic( 'a single-byte passthrough increment cannot overflow usize' ):
				new_size += 1
				i += 1
	return Result.Err( JSONError.UnexpectedEnd( i ))


def _fill_string( data: ConstPtr[u8], length: usize, start: usize, buf: Ptr[u8] ) -> None:
	with compiler.panic_arithmetic( 'advancing one byte past a known-in-bounds quote cannot overflow usize' ):
		i: usize = start + 1
	out: usize = 0
	while i < length:
		ch: u8 = data[i]
		if ch == _ASCII_QUOTE:
			return
		if ch == _ASCII_BACKSLASH:
			with compiler.panic_arithmetic( 'advancing one byte past a known-in-bounds backslash cannot overflow usize' ):
				i += 1
			esc: u8 = data[i]
			if esc == _ASCII_LOWER_U:
				with compiler.panic_arithmetic( 'advancing one byte past a known-in-bounds u-escape marker cannot overflow usize' ):
					hex_start: usize = i + 1
				decoded = _decode_u_escape( data, length, hex_start ).unwrap( '_fill_string: re-decoding a \\uXXXX escape already validated by _measure_string cannot fail' )
				with compiler.panic_arithmetic( 'a decoded codepoints own utf8-encode increment cannot overflow usize' ):
					out += builtins.encode_utf8_at( buf, out, decoded[0] )
				i = decoded[1]
			else:
				short: u8 = _short_escape_byte( esc )
				buf[out] = short
				with compiler.panic_arithmetic( 'a single-char escape increment cannot overflow usize' ):
					out += 1
					i += 1
		else:
			buf[out] = ch
			with compiler.panic_arithmetic( 'a single-byte passthrough increment cannot overflow usize' ):
				out += 1
				i += 1


def _short_escape_byte( esc: u8 ) -> u8:
	# maps a single-char JSON escape letter to its literal byte value -
	# only ever called with a byte already validated by _measure_string's
	# own identical branch, so the final `sys.panic` is unreachable.
	if esc == _ASCII_QUOTE:
		return _ASCII_QUOTE
	if esc == _ASCII_BACKSLASH:
		return _ASCII_BACKSLASH
	if esc == _ASCII_SLASH:
		return _ASCII_SLASH
	if esc == _ASCII_LOWER_B:
		return 0x08
	if esc == _ASCII_LOWER_F:
		return 0x0C
	if esc == _ASCII_LOWER_N:
		return _ASCII_LF
	if esc == _ASCII_LOWER_R:
		return _ASCII_CR
	if esc == _ASCII_LOWER_T:
		return _ASCII_TAB
	sys.panic( '_short_escape_byte: unreachable - _measure_string already validated this escape byte' )


def _decode_hex4( data: ConstPtr[u8], length: usize, pos: usize ) -> Result[tuple[u32, usize], JSONError]:
	with compiler.panic_arithmetic( 'a bounded 4-hex-digit probe cannot overflow usize' ):
		end: usize = pos + 4
	if end > length:
		return Result.Err( JSONError.UnexpectedEnd( pos ))
	value: u32 = 0
	i: usize = pos
	with compiler.panic_arithmetic( 'walking exactly 4 known-in-bounds bytes cannot overflow usize' ):
		while i < end:
			c: u8 = data[i]
			digit: u32 = 0
			if c >= _ASCII_ZERO and c <= _ASCII_NINE:
				digit = u32( c - _ASCII_ZERO )
			elif c >= _ASCII_LOWER_A and c <= _ASCII_LOWER_F:
				digit = u32( c - _ASCII_LOWER_A ) + 10
			elif c >= _ASCII_UPPER_A and c <= _ASCII_UPPER_F:
				digit = u32( c - _ASCII_UPPER_A ) + 10
			else:
				return Result.Err( JSONError.InvalidEscape( pos ))
			value = ( value << 4 ) | digit
			i += 1
	return Result.Ok(( value, end ))


def _decode_u_escape( data: ConstPtr[u8], length: usize, pos: usize ) -> Result[tuple[u32, usize], JSONError]:
	# pos points right after the 'u' of a \uXXXX escape. Combines a UTF-16
	# surrogate pair (\uD800-\uDBFF followed immediately by \uDC00-\uDFFF)
	# into one astral codepoint; a lone/unpaired surrogate is InvalidEscape.
	first = _decode_hex4( data, length, pos ).or_return()
	hi: u32 = first[0]
	next_pos: usize = first[1]
	if hi < 0xD800 or hi > 0xDFFF:
		return Result.Ok(( hi, next_pos ))
	if hi > 0xDBFF:
		return Result.Err( JSONError.InvalidEscape( pos ))
	with compiler.panic_arithmetic( 'a bounded low-surrogate marker probe cannot overflow usize' ):
		marker_pos: usize = next_pos + 1
	if marker_pos >= length or data[next_pos] != _ASCII_BACKSLASH or data[marker_pos] != _ASCII_LOWER_U:
		return Result.Err( JSONError.InvalidEscape( next_pos ))
	with compiler.panic_arithmetic( 'a bounded low-surrogate hex offset cannot overflow usize' ):
		hex_pos: usize = next_pos + 2
	second = _decode_hex4( data, length, hex_pos ).or_return()
	lo: u32 = second[0]
	final_pos: usize = second[1]
	if lo < 0xDC00 or lo > 0xDFFF:
		return Result.Err( JSONError.InvalidEscape( next_pos ))
	with compiler.panic_arithmetic( 'combining two known-in-range surrogate halves into an astral codepoint cannot overflow u32' ):
		cp: u32 = 0x10000 + (( hi - 0xD800 ) << 10 ) + ( lo - 0xDC00 )
	return Result.Ok(( cp, final_pos ))


def _parse_object( s: str, pos: usize ) -> Result[tuple[JSONValue, usize], JSONError]:
	data: ConstPtr[u8] = s.get_cstr()
	length: usize = s.byte_len()
	with compiler.panic_arithmetic( 'advancing one byte past a known-in-bounds brace cannot overflow usize' ):
		p: usize = pos + 1
	obj: JSONValue = JSONValue.object()
	p = _skip_ws( s, p )
	if p >= length:
		return Result.Err( JSONError.UnexpectedEnd( p ))
	if data[p] == _ASCII_RBRACE:
		with compiler.panic_arithmetic( 'advancing one byte past a known-in-bounds brace cannot overflow usize' ):
			return Result.Ok(( obj, p + 1 ))
	while p < length:
		p = _skip_ws( s, p )
		if p >= length:
			return Result.Err( JSONError.UnexpectedEnd( p ))
		if data[p] != _ASCII_QUOTE:
			return Result.Err( JSONError.UnexpectedChar( p ))
		key_parsed = _parse_string( s, p ).or_return()
		key: str = key_parsed[0]
		p = key_parsed[1]
		p = _skip_ws( s, p )
		if p >= length or data[p] != _ASCII_COLON:
			return Result.Err( JSONError.UnexpectedChar( p ))
		with compiler.panic_arithmetic( 'advancing one byte past a known-in-bounds colon cannot overflow usize' ):
			p += 1
		val_parsed = _parse_value( s, p ).or_return()
		obj.object_set( key, val_parsed[0] ).unwrap( '_parse_object: object_set on a freshly constructed JSONValue.object() cannot fail' )
		p = val_parsed[1]
		p = _skip_ws( s, p )
		if p >= length:
			return Result.Err( JSONError.UnexpectedEnd( p ))
		if data[p] == _ASCII_RBRACE:
			with compiler.panic_arithmetic( 'advancing one byte past a known-in-bounds brace cannot overflow usize' ):
				return Result.Ok(( obj, p + 1 ))
		if data[p] != _ASCII_COMMA:
			return Result.Err( JSONError.UnexpectedChar( p ))
		with compiler.panic_arithmetic( 'advancing one byte past a known-in-bounds comma cannot overflow usize' ):
			p += 1
	return Result.Err( JSONError.UnexpectedEnd( p ))


def _parse_array( s: str, pos: usize ) -> Result[tuple[JSONValue, usize], JSONError]:
	data: ConstPtr[u8] = s.get_cstr()
	length: usize = s.byte_len()
	with compiler.panic_arithmetic( 'advancing one byte past a known-in-bounds bracket cannot overflow usize' ):
		p: usize = pos + 1
	arr: JSONValue = JSONValue.array()
	p = _skip_ws( s, p )
	if p >= length:
		return Result.Err( JSONError.UnexpectedEnd( p ))
	if data[p] == _ASCII_RBRACKET:
		with compiler.panic_arithmetic( 'advancing one byte past a known-in-bounds bracket cannot overflow usize' ):
			return Result.Ok(( arr, p + 1 ))
	while p < length:
		val_parsed = _parse_value( s, p ).or_return()
		arr.array_append( val_parsed[0] ).unwrap( '_parse_array: array_append on a freshly constructed JSONValue.array() cannot fail' )
		p = val_parsed[1]
		p = _skip_ws( s, p )
		if p >= length:
			return Result.Err( JSONError.UnexpectedEnd( p ))
		if data[p] == _ASCII_RBRACKET:
			with compiler.panic_arithmetic( 'advancing one byte past a known-in-bounds bracket cannot overflow usize' ):
				return Result.Ok(( arr, p + 1 ))
		if data[p] != _ASCII_COMMA:
			return Result.Err( JSONError.UnexpectedChar( p ))
		with compiler.panic_arithmetic( 'advancing one byte past a known-in-bounds comma cannot overflow usize' ):
			p += 1
	return Result.Err( JSONError.UnexpectedEnd( p ))


# ---------------------------------------------------------------------------
# 3. dumps() - serializer. Builds output as a list[str] of pieces + join
# (int._to_radix_digits' own idiom) rather than a hand-rolled buffer - this
# codebase has no growable StringBuilder/byte-buffer type (bytearray is
# fixed-size at construction, no append()).
# ---------------------------------------------------------------------------

def dumps( value: JSONValue ) -> Result[str, JSONError]:
	return _dump_value( value )


def _dump_value( value: JSONValue ) -> Result[str, JSONError]:
	match value:
		case JSONValue.Null( _ ):
			return Result.Ok( str( 'null' ))
		case JSONValue.Bool( b ):
			return Result.Ok( str( 'true' ) if b else str( 'false' ))
		case JSONValue.Int( n ):
			return Result.Ok( n.__str__() )
		case JSONValue.Float( f ):
			if compiler.is_nan( f ) or compiler.is_inf( f ):
				return Result.Err( JSONError.NonFiniteFloat( None ))
			# f.__str__() crashes emit_c() (KeyError: 'value' in
			# _emit_call_args) - f64.__str__ is attached via post-hoc
			# assignment (lib/builtins/__float.py: `f64.__str__ = _f64_str`)
			# rather than declared inline in the class body, and calling it
			# as an explicit method Call hits a genuine, narrow compiler bug
			# (confirmed with a minimal repro with zero json.py involvement -
			# reproduces even for a bare top-level `f.__str__()`). A bare
			# f-string interpolation with no format spec reaches the exact
			# same underlying _f64_str formatter through different codegen
			# (confirmed working) and is documented to always produce
			# identical text to str(float)/repr(float) - not a behavior
			# change, just a differently-spelled call to the same function.
			return Result.Ok( f'{f}' )
		case JSONValue.String( s ):
			return Result.Ok( _json_escape_string( s ))
		case JSONValue.Array( arr ):
			# distinct names per arm (arr_*/obj_*), not shared ones - a
			# variable's type is only ever declared once per function (no
			# block scoping - see cfg.py's own module docstring), so a
			# SECOND `pieces: list[str] = ...`/`i: usize = ...`/`count:
			# usize = ...` in the Object arm below would be a genuine
			# redeclaration error even though the two arms are mutually
			# exclusive and each returns before the other could ever run.
			# `count`, not `n` for the same underlying reason - `n` is
			# already bound to an `int` by the Int( n ) arm above.
			arr_pieces: list[str] = list[str]()
			arr_i: usize = 0
			arr_count: usize = len( arr )
			while arr_i < arr_count:
				elem: JSONValue = arr.__getitem__( arr_i ).unwrap( 'dumps: index in bounds by construction' )
				piece: str = _dump_value( elem ).or_return()
				arr_pieces.append( piece )
				with compiler.panic_arithmetic( 'walking a list of known length index-by-index cannot overflow usize' ):
					arr_i += 1
			return Result.Ok( str( '[' ) + str( ',' ).join( arr_pieces ) + str( ']' ))
		case JSONValue.Object( obj ):
			obj_pieces: list[str] = list[str]()
			obj_i: usize = 0
			obj_count: usize = len( obj )
			while obj_i < obj_count:
				k: str = obj.key_at( obj_i ).unwrap( 'dumps: index in bounds by construction' )
				v: JSONValue = obj.value_at( obj_i ).unwrap( 'dumps: index in bounds by construction' )
				entry: str = _json_escape_string( k ) + str( ':' ) + _dump_value( v ).or_return()
				obj_pieces.append( entry )
				with compiler.panic_arithmetic( 'walking a dict of known length index-by-index cannot overflow usize' ):
					obj_i += 1
			return Result.Ok( str( '{' ) + str( ',' ).join( obj_pieces ) + str( '}' ))


def _json_escape_string( s: str ) -> str:
	data: ConstPtr[u8] = s.get_cstr()
	length: usize = s.byte_len()
	new_size: usize = _measure_escaped_len( data, length )
	with compiler.panic_arithmetic( 'a measured escape size plus 2 quotes plus one zero terminator cannot overflow usize' ):
		buf_size: usize = new_size + 3
	buf: Ptr[u8] = sys.alloc[u8]( buf_size )
	buf[0] = _ASCII_QUOTE
	out: usize = _fill_escaped( data, length, buf, 1 )
	buf[out] = _ASCII_QUOTE
	with compiler.panic_arithmetic( 'advancing one byte past a known-in-bounds closing quote cannot overflow usize' ):
		term: usize = out + 1
	buf[term] = 0
	built: str = str.from_cstr( compiler.cast( ConstPtr[u8], buf ), buf_size ).unwrap( 'json._json_escape_string: internal buffer was not valid UTF-8 (unreachable - only input UTF-8 bytes and computed escape sequences are ever written)' )
	sys.free( buf )
	return built


def _measure_escaped_len( data: ConstPtr[u8], length: usize ) -> usize:
	total: usize = 0
	i: usize = 0
	while i < length:
		consumed: usize = 0
		cp: u32 = builtins.decode_utf8_at( data, i, compiler.addrof( consumed ))
		if cp == 0x22 or cp == 0x5C or cp == 0x0A or cp == 0x0D or cp == 0x09 or cp == 0x08 or cp == 0x0C:
			with compiler.panic_arithmetic( 'a 2-byte escape increment cannot overflow usize' ):
				total += 2
		elif cp < 0x20:
			with compiler.panic_arithmetic( 'a 6-byte \\u00XX escape increment cannot overflow usize' ):
				total += 6
		else:
			with compiler.panic_arithmetic( 'a codepoints own utf8 width added to a running usize total cannot overflow' ):
				total += builtins.utf8_encoded_len( cp )
		with compiler.panic_arithmetic( 'advancing by a codepoints own consumed byte count cannot overflow usize' ):
			i += consumed
	return total


def _fill_escaped( data: ConstPtr[u8], length: usize, buf: Ptr[u8], out_start: usize ) -> usize:
	i: usize = 0
	out: usize = out_start
	while i < length:
		consumed: usize = 0
		cp: u32 = builtins.decode_utf8_at( data, i, compiler.addrof( consumed ))
		short: u8 = _short_json_escape_for( cp )
		if short != 0:
			buf[out] = _ASCII_BACKSLASH
			with compiler.panic_arithmetic( 'a 2-byte escape fill increment cannot overflow usize' ):
				buf[out + 1] = short
				out += 2
		elif cp < 0x20:
			with compiler.panic_arithmetic( 'writing a fixed 6-byte \\u00XX escape cannot overflow usize' ):
				buf[out] = _ASCII_BACKSLASH
				buf[out + 1] = _ASCII_LOWER_U
				buf[out + 2] = _ASCII_ZERO
				buf[out + 3] = _ASCII_ZERO
				buf[out + 4] = _hex_digit_char( u8( cp ) >> 4 )
				buf[out + 5] = _hex_digit_char( u8( cp ) & 0x0F )
				out += 6
		else:
			with compiler.panic_arithmetic( 'a codepoints own utf8-encode increment cannot overflow usize' ):
				out += builtins.encode_utf8_at( buf, out, cp )
		with compiler.panic_arithmetic( 'advancing by a codepoints own consumed byte count cannot overflow usize' ):
			i += consumed
	return out


def _short_json_escape_for( cp: u32 ) -> u8:
	# 0 is not a valid escape LETTER (it's the ascii NUL codepoint's own
	# value coincidentally, but NUL itself takes the \u00XX path below,
	# never this one) - safe to use as the "no short escape" sentinel.
	if cp == 0x22:
		return _ASCII_QUOTE
	if cp == 0x5C:
		return _ASCII_BACKSLASH
	if cp == 0x0A:
		return _ASCII_LOWER_N
	if cp == 0x0D:
		return _ASCII_LOWER_R
	if cp == 0x09:
		return _ASCII_LOWER_T
	if cp == 0x08:
		return _ASCII_LOWER_B
	if cp == 0x0C:
		return _ASCII_LOWER_F
	return 0


def _hex_digit_char( nibble: u8 ) -> u8:
	if nibble < 10:
		with compiler.panic_arithmetic( 'a nibble 0-9 plus the ascii \'0\' offset cannot overflow u8' ):
			return _ASCII_ZERO + nibble
	with compiler.panic_arithmetic( 'a nibble 10-15 plus the ascii lowercase-a offset cannot overflow u8' ):
		return _ASCII_LOWER_A + ( nibble - 10 )


def _usize_to_str( value: usize ) -> str:
	# self-contained digit-by-division (int.__str__'s own idiom) rather than
	# routing through the arbitrary-precision `int` type, which has no
	# usize-widening constructor - only ever used for array indices here.
	if value == 0:
		return str( '0' )
	count: usize = 0
	with compiler.panic_arithmetic( 'a usize has at most 20 decimal digits, so this counter cannot overflow usize' ):
		v: usize = value
		while v > 0:
			v = v // 10
			count += 1
	with compiler.panic_arithmetic( 'a bounded digit count plus one zero terminator cannot overflow usize' ):
		buf_size: usize = count + 1
	buf: Ptr[u8] = sys.alloc[u8]( buf_size )
	buf[count] = 0
	with compiler.panic_arithmetic( 'writing exactly `count` digits (computed above) into a buffer sized for exactly that many plus a terminator cannot overflow' ):
		pos: usize = count
		v2: usize = value
		while pos > 0:
			pos -= 1
			buf[pos] = _ASCII_ZERO + u8( v2 % 10 )
			v2 = v2 // 10
	built: str = str.from_cstr( compiler.cast( ConstPtr[u8], buf ), buf_size ).unwrap( '_usize_to_str: internal buffer was not valid UTF-8 (unreachable - only ASCII digits are ever written)' )
	sys.free( buf )
	return built


# ---------------------------------------------------------------------------
# 4. Path syntax - shared by flatten()'s own path generation and
# get_path()/set_path()/unflatten()'s parsing. Grammar:
#   path    := '' | segment ( '.' segment | index )*
#   segment := any run of bytes excluding '.', '[', ']'
#   index   := '[' digit+ ']'
# A leading '[' (root array access, e.g. "[0]") is valid. Known v1
# limitation (confirmed with this module's requester): a segment/object key
# containing '.'/'['/']' is not escaped - flatten() still emits a best-effort
# path for it, but that path is not guaranteed to parse back to the same key.
# ---------------------------------------------------------------------------

@union
class PathToken:
	Key:   str
	Index: usize


def _tokenize_path( path: str ) -> Result[list[PathToken], JSONError]:
	tokens: list[PathToken] = list[PathToken]()
	data: ConstPtr[u8] = path.get_cstr()
	n: usize = path.byte_len()
	if n == 0:
		return Result.Ok( tokens )
	i: usize = 0
	while i < n:
		ch: u8 = data[i]
		if ch == _ASCII_LBRACKET:
			with compiler.panic_arithmetic( 'advancing one byte past a known-in-bounds bracket cannot overflow usize' ):
				j: usize = i + 1
			digit_start: usize = j
			with compiler.panic_arithmetic( 'walking a path of known length index-by-index cannot overflow usize' ):
				while j < n and _is_ascii_digit( data[j] ):
					j += 1
			with compiler.panic_arithmetic( 'a forward-scanned digit run length cannot overflow usize' ):
				digit_count: usize = j - digit_start
			if digit_count == 0 or digit_count > 9 or j >= n or data[j] != _ASCII_RBRACKET:
				return Result.Err( JSONError.InvalidPath( i ))
			idx: usize = _parse_bounded_digits( data, digit_start, digit_count )
			tokens.append( PathToken.Index( idx ))
			with compiler.panic_arithmetic( 'advancing one byte past a known-in-bounds bracket cannot overflow usize' ):
				i = j + 1
			# an index token may be directly followed by another '[' (chained
			# indices, "a[0][1]") with no separator, or by '.segment'
			# (consume the dot here - the same way the key branch below
			# consumes ITS trailing dot - since a leading '.' at the top of
			# the loop is otherwise rejected outright), or by end-of-string.
			# Anything else ("a[0]x") is a malformed path.
			if i < n and data[i] == _ASCII_DOT:
				with compiler.panic_arithmetic( 'advancing one byte past a known-in-bounds dot cannot overflow usize' ):
					i += 1
				if i >= n:
					return Result.Err( JSONError.InvalidPath( i ))
			elif i < n and data[i] != _ASCII_LBRACKET:
				return Result.Err( JSONError.InvalidPath( i ))
		elif ch == _ASCII_DOT:
			return Result.Err( JSONError.InvalidPath( i ))
		else:
			start: usize = i
			with compiler.panic_arithmetic( 'walking a path of known length index-by-index cannot overflow usize' ):
				while i < n and data[i] != _ASCII_DOT and data[i] != _ASCII_LBRACKET:
					i += 1
			tokens.append( PathToken.Key( path[start:i] ))
			if i < n and data[i] == _ASCII_DOT:
				with compiler.panic_arithmetic( 'advancing one byte past a known-in-bounds dot cannot overflow usize' ):
					i += 1
				if i >= n:
					return Result.Err( JSONError.InvalidPath( i ))
	return Result.Ok( tokens )


def _parse_bounded_digits( data: ConstPtr[u8], start: usize, count: usize ) -> usize:
	# count is capped at 9 by the caller (_tokenize_path), so the maximum
	# representable value (999999999) safely fits in usize on any real
	# target - this deliberately avoids routing through the arbitrary-
	# precision `int` type (which has no usize-narrowing conversion) for
	# what is, in practice, an array index that will never realistically
	# approach a billion.
	value: usize = 0
	i: usize = 0
	with compiler.panic_arithmetic( 'accumulating at most 9 decimal digits (capped by the caller) cannot overflow usize on any real target' ):
		while i < count:
			value = value * 10 + usize( data[start + i] - _ASCII_ZERO )
			i += 1
	return value


def _child_container_for_token( next_tok: PathToken ) -> JSONValue:
	# a fresh, empty container to auto-vivify a missing intermediate link -
	# Array if the token AFTER it is an index, Object if it's a key.
	match next_tok:
		case PathToken.Index( _ ):
			return JSONValue.array()
		case PathToken.Key( _ ):
			return JSONValue.object()


def _set_path_tokens( container: JSONValue, tokens: list[PathToken], idx: usize, value: JSONValue ) -> Result[None, JSONError]:
	# Every branch below computes its own freshly-declared, once-assigned
	# local and returns directly - deliberately never a bare `x: T`
	# declaration reassigned across separate match/if arms (that shape hit a
	# real compiler crash - KeyError in lowering.py's _stmt_If - during this
	# module's own development; this style sidesteps it entirely).
	tok: PathToken = tokens.__getitem__( idx ).unwrap( '_set_path_tokens: index in bounds by construction' )
	with compiler.panic_arithmetic( 'a token list index plus one, bounded by the token list itself, cannot overflow usize' ):
		next_idx: usize = idx + 1
	is_last: bool = next_idx == len( tokens )
	match tok:
		case PathToken.Key( k ):
			obj = container.as_object().or_return()
			if is_last:
				obj.__setitem__( k, value )
				return Result.Ok( None )
			next_tok: PathToken = tokens.__getitem__( next_idx ).unwrap( '_set_path_tokens: index in bounds by construction' )
			match obj.__getitem__( k ):
				case Result.Ok( existing ):
					return _set_path_tokens( existing, tokens, next_idx, value )
				case Result.Err( _ ):
					pass
			fresh: JSONValue = _child_container_for_token( next_tok )
			obj.__setitem__( k, fresh )
			return _set_path_tokens( fresh, tokens, next_idx, value )
		case PathToken.Index( i ):
			arr = container.as_array().or_return()
			n: usize = len( arr )
			if is_last:
				if i < n:
					match arr.__setitem__( i, value ):
						case Result.Ok( _ ):
							return Result.Ok( None )
						case Result.Err( _ ):
							return Result.Err( JSONError.IndexOutOfBounds( None ))
				elif i == n:
					arr.append( value )
					return Result.Ok( None )
				else:
					return Result.Err( JSONError.IndexOutOfBounds( None ))
			next_tok2: PathToken = tokens.__getitem__( next_idx ).unwrap( '_set_path_tokens: index in bounds by construction' )
			if i < n:
				existing2: JSONValue = arr.__getitem__( i ).unwrap( '_set_path_tokens: index in bounds, already checked' )
				return _set_path_tokens( existing2, tokens, next_idx, value )
			elif i == n:
				fresh2: JSONValue = _child_container_for_token( next_tok2 )
				arr.append( fresh2 )
				return _set_path_tokens( fresh2, tokens, next_idx, value )
			else:
				return Result.Err( JSONError.IndexOutOfBounds( None ))


# ---------------------------------------------------------------------------
# 5. flatten() / unflatten()
# ---------------------------------------------------------------------------

def flatten( value: JSONValue ) -> dict[str, JSONValue]:
	result: dict[str, JSONValue] = dict[str, JSONValue]()
	_flatten_into( value, str( '' ), result )
	return result


def _flatten_into( value: JSONValue, prefix: str, out: dict[str, JSONValue] ) -> None:
	match value:
		case JSONValue.Array( arr ):
			n: usize = len( arr )
			if n == 0:
				out.__setitem__( prefix, value )
				return
			i: usize = 0
			while i < n:
				elem: JSONValue = arr.__getitem__( i ).unwrap( 'flatten: index in bounds by construction' )
				child_path: str = prefix + str( '[' ) + _usize_to_str( i ) + str( ']' )
				_flatten_into( elem, child_path, out )
				with compiler.panic_arithmetic( 'walking a list of known length index-by-index cannot overflow usize' ):
					i += 1
		case JSONValue.Object( obj ):
			# distinct names from the Array arm's own n/i above - a
			# variable's type is only ever declared once per function
			obj_n: usize = len( obj )
			if obj_n == 0:
				out.__setitem__( prefix, value )
				return
			obj_i: usize = 0
			while obj_i < obj_n:
				k: str = obj.key_at( obj_i ).unwrap( 'flatten: index in bounds by construction' )
				v: JSONValue = obj.value_at( obj_i ).unwrap( 'flatten: index in bounds by construction' )
				# distinct name from the Array arm's own child_path above
				obj_child_path: str = k if prefix.byte_len() == 0 else prefix + str( '.' ) + k
				_flatten_into( v, obj_child_path, out )
				with compiler.panic_arithmetic( 'walking a dict of known length index-by-index cannot overflow usize' ):
					obj_i += 1
		case _:
			out.__setitem__( prefix, value )


def unflatten( flat: dict[str, JSONValue] ) -> Result[JSONValue, JSONError]:
	n: usize = len( flat )
	if n == 0:
		return Result.Ok( JSONValue.object() )
	root: JSONValue = JSONValue.null()
	i: usize = 0
	while i < n:
		key: str = flat.key_at( i ).unwrap( 'unflatten: index in bounds by construction' )
		value: JSONValue = flat.value_at( i ).unwrap( 'unflatten: index in bounds by construction' )
		tokens = _tokenize_path( key ).or_return()
		if len( tokens ) == 0:
			if n != 1:
				return Result.Err( JSONError.PathTypeMismatch( 0 ))
			return Result.Ok( value )
		if root.is_null():
			first_tok: PathToken = tokens.__getitem__( 0 ).unwrap( 'unflatten: index in bounds by construction' )
			match first_tok:
				case PathToken.Index( _ ):
					root = JSONValue.array()
				case PathToken.Key( _ ):
					root = JSONValue.object()
		_set_path_tokens( root, tokens, 0, value ).or_return()
		with compiler.panic_arithmetic( 'walking a dict of known length index-by-index cannot overflow usize' ):
			i += 1
	return Result.Ok( root )
