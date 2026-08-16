import compiler
import sys

# GUID/IID - see PLAN_SUBCLASSING_VTABLES_COM.md's "COM specifics" section.
# @cstruct (a plain value type, matching the real win32 GUID ABI - a COM
# API takes/returns one of these by value or by ConstPtr[GUID], never
# behind a refcounted pointer), laid out field-for-field exactly like the
# real Windows GUID struct:
#
#   typedef struct _GUID {
#       unsigned long  Data1;
#       unsigned short Data2;
#       unsigned short Data3;
#       unsigned char  Data4[8];
#   } GUID;
#
# Data4 is 8 separate u8 fields here, not a fixed-size array. SYNTAX.md's
# `u8[8]`-style fixed-size inline array syntax now works for DECLARING a
# field and zero-filling it at construction (`data4: u8[8] = 0`), but that's
# not enough for this class: from_str() needs to set 8 independently-parsed
# hex bytes into 8 specific positions, and __eq__/__ne__ need to compare
# them one at a time - neither is possible yet, since element-level indexed
# access (`f.data4[i]`) is still unimplemented ("u8[8] fields cannot be read
# as a whole value yet (no element-level array access is implemented)",
# confirmed by a real compile). This unrolled-fields form stays until that
# follow-up lands.
@cstruct
class GUID:
	data1: u32
	data2: u16
	data3: u16
	data4_0: u8
	data4_1: u8
	data4_2: u8
	data4_3: u8
	data4_4: u8
	data4_5: u8
	data4_6: u8
	data4_7: u8

	@staticmethod
	def from_str( s: str ) -> GUID:
		''' parse the standard hyphenated hex form - 8-4-4-4-12 hex digits,
		e.g. 'deadbeef-dead-beef-dead-beefdeadbeef' - matching the usual
		IID/CLSID text representation. A plain @cstruct can't have a real
		__init__ (confirmed directly - only RCClass supports one; a
		@cstruct only ever builds via field=value sugar), so this is a
		staticmethod factory instead of GUID(s) - GUID.from_str(s). Panics
		on malformed input (wrong group count/lengths, non-hex digit) -
		matches this codebase's existing "hand-written parsing panics on
		malformed input" convention (see int.from_str's own Result-based
		counterpart for the alternative when a caller genuinely needs to
		recover from bad input instead of treating it as a programmer
		error, e.g. a hardcoded IID literal that's simply wrong). '''
		parts: list[str] = s.split( '-' )
		if len( parts ) != 5:
			sys.panic( 'invalid GUID string: expected 5 hyphen-separated groups' )
		g0: str = parts.__getitem__( 0 ).unwrap( 'invalid GUID string' )
		g1: str = parts.__getitem__( 1 ).unwrap( 'invalid GUID string' )
		g2: str = parts.__getitem__( 2 ).unwrap( 'invalid GUID string' )
		g3: str = parts.__getitem__( 3 ).unwrap( 'invalid GUID string' )
		g4: str = parts.__getitem__( 4 ).unwrap( 'invalid GUID string' )
		if g0.byte_len() != 8 or g1.byte_len() != 4 or g2.byte_len() != 4 or g3.byte_len() != 4 or g4.byte_len() != 12:
			sys.panic( 'invalid GUID string: expected 8-4-4-4-12 hex digits' )
		return GUID(
			data1 = _parse_hex_u32( g0 ),
			data2 = _parse_hex_u16( g1 ),
			data3 = _parse_hex_u16( g2 ),
			data4_0 = _parse_hex_u8( g3, 0 ),
			data4_1 = _parse_hex_u8( g3, 2 ),
			data4_2 = _parse_hex_u8( g4, 0 ),
			data4_3 = _parse_hex_u8( g4, 2 ),
			data4_4 = _parse_hex_u8( g4, 4 ),
			data4_5 = _parse_hex_u8( g4, 6 ),
			data4_6 = _parse_hex_u8( g4, 8 ),
			data4_7 = _parse_hex_u8( g4, 10 ),
		)

	def __eq__( self, other: GUID ) -> bool:
		return (
			self.data1 == other.data1 and self.data2 == other.data2 and self.data3 == other.data3
			and self.data4_0 == other.data4_0 and self.data4_1 == other.data4_1
			and self.data4_2 == other.data4_2 and self.data4_3 == other.data4_3
			and self.data4_4 == other.data4_4 and self.data4_5 == other.data4_5
			and self.data4_6 == other.data4_6 and self.data4_7 == other.data4_7
		)

	def __ne__( self, other: GUID ) -> bool:
		return not self.__eq__( other )

def _hex_digit_value( c: u8 ) -> u8:
	''' '0'-'9'/'a'-'f'/'A'-'F' -> 0-15, panics on anything else - same
	posture as _decode_utf8_at's own "assumes well-formed input, this
	isn't the validation boundary" trust, except here malformed input
	genuinely IS a programmer error (a hardcoded GUID literal), not
	attacker-controlled data, so a panic (not a Result) is the right
	shape - matches Result.unwrap's own "if not ok, panic" fallthrough
	convention: no explicit return needed after sys.panic, it's NoReturn. '''
	if c >= 0x30 and c <= 0x39: # '0'-'9'
		with compiler.wrap_arithmetic:
			return c - 0x30
	if c >= 0x61 and c <= 0x66: # 'a'-'f'
		with compiler.wrap_arithmetic:
			return c - 0x61 + 10
	if c >= 0x41 and c <= 0x46: # 'A'-'F'
		with compiler.wrap_arithmetic:
			return c - 0x41 + 10
	sys.panic( 'invalid GUID string: bad hex digit' )


def _parse_hex_u8( s: str, offset: usize ) -> u8:
	''' 2 hex characters starting at byte offset `offset` in `s`. '''
	data: ConstPtr[u8] = s.get_cstr()
	with compiler.wrap_arithmetic:
		hi: u8 = _hex_digit_value( data[offset] )
		lo: u8 = _hex_digit_value( data[offset + 1] )
		return ( hi << 4 ) | lo


def _parse_hex_u16( s: str ) -> u16:
	''' exactly 4 hex characters, big-endian (textual order == field's
	own byte order, matching how GUID.data2/data3 are always written and
	read as a plain integer, not a byte array - unlike Data4). '''
	data: ConstPtr[u8] = s.get_cstr()
	value: u16 = 0
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < 4:
			digit: u8 = _hex_digit_value( data[i] )
			value = ( value << 4 ) | u16( digit )
			i += 1
	return value


def _parse_hex_u32( s: str ) -> u32:
	''' exactly 8 hex characters - see _parse_hex_u16's own note on why
	this is a plain big-endian-text-to-integer parse, not a byte array. '''
	data: ConstPtr[u8] = s.get_cstr()
	value: u32 = 0
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < 8:
			digit: u8 = _hex_digit_value( data[i] )
			value = ( value << 4 ) | u32( digit )
			i += 1
	return value
