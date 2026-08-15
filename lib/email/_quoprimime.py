'''
Quoted-printable (RFC 2045 section 6.7) encode/decode - the private helper
module backing lib/email/message.py's Content-Transfer-Encoding:
quoted-printable support and RFC 2047 'Q' encoded-words. No precedent exists
anywhere else in this repo (unlike base64, which lib/base64.py already
provides).

decode() is deliberately lenient (matches Python's own email.quoprimime /
quopri modules): a soft line break ("=\\r\\n" or "=\\n") is removed, "=XX"
with two hex digits decodes to that byte, and anything else after a literal
'=' (end of string, or non-hex digits) is passed through unchanged rather
than treated as an error - real-world messages routinely contain a bare '='
that isn't actually an escape.

encode() is a simpler byte-oriented pass (printable ASCII passes through,
everything else - including '=' itself - becomes "=XX" hex, uppercase) with
76-column soft line wrapping. It does not implement RFC 2045's more precise
"don't escape space/tab unless trailing on the line" carve-out, so a
byte-for-byte round trip of encode()->decode() is not guaranteed when the
original bytes contain a trailing space/tab immediately before a line break
- every other byte round-trips exactly.
'''

import compiler
import sys


def _hex_val( b: u8 ) -> i32:
	if b >= 0x30 and b <= 0x39: # '0'-'9'
		with compiler.wrap_arithmetic:
			return i32( b ) - 0x30
	if b >= 0x41 and b <= 0x46: # 'A'-'F'
		with compiler.wrap_arithmetic:
			return i32( b ) - 0x41 + 10
	if b >= 0x61 and b <= 0x66: # 'a'-'f'
		with compiler.wrap_arithmetic:
			return i32( b ) - 0x61 + 10
	return -1


def decode( data: str ) -> bytes:
	''' decodes a quoted-printable str (the wire text is always pure ASCII)
	back to its original bytes. Never fails - see the module docstring on
	leniency. '''
	src: ConstPtr[u8] = data.get_const_ptr()
	n: usize = data.byte_len()
	scratch: bytearray = bytearray( n ) # decoded length is always <= n
	out: Ptr[u8] = scratch.get_ptr()
	i: usize = 0
	o: usize = 0
	while i < n:
		b: u8 = src[i]
		if b == 0x3D: # '='
			with compiler.wrap_arithmetic:
				i1: usize = i + 1
				i2: usize = i + 2
				i3: usize = i + 3
			if i1 < n and src[i1] == 0x0D and i2 < n and src[i2] == 0x0A:
				i = i3 # soft line break "=\r\n" - drop it, emit nothing
				continue
			if i1 < n and src[i1] == 0x0A:
				i = i2 # soft line break "=\n" - drop it, emit nothing
				continue
			if i2 < n:
				hi: i32 = _hex_val( src[i1] )
				lo: i32 = _hex_val( src[i2] )
				if hi >= 0 and lo >= 0:
					with compiler.wrap_arithmetic:
						out[o] = u8( hi * 16 + lo )
						o += 1
					i = i3
					continue
			# malformed/truncated escape - pass the '=' through literally
			out[o] = b
			with compiler.wrap_arithmetic:
				o += 1
				i += 1
		else:
			out[o] = b
			with compiler.wrap_arithmetic:
				o += 1
				i += 1
	trimmed: bytearray = scratch[:o]
	return bytes.from_bytearray( move( trimmed ))


def _is_safe_byte( b: u8 ) -> bool:
	# printable ASCII except '=' (0x3D); space/tab (0x20/0x09) pass through
	# too (real quoted-printable escapes them only when trailing on a
	# line - see the module docstring's documented simplification).
	if b == 0x3D:
		return False
	if b >= 0x21 and b <= 0x7E:
		return True
	if b == 0x20 or b == 0x09:
		return True
	return False


def _hex_digit_char( v: u8 ) -> str:
	if v < 10:
		with compiler.wrap_arithmetic:
			return chr( u32( v ) + 0x30 )
	with compiler.wrap_arithmetic:
		return chr( u32( v ) + 0x37 ) # 10 -> 'A' (0x41 = 10 + 0x37)


def encode( data: bytes|bytearray ) -> str:
	''' quoted-printable-encodes arbitrary bytes into a str, wrapping output
	lines at 76 columns with soft line breaks ("=\\r\\n"). '''
	src: ConstPtr[u8] = data.get_const_ptr()
	n: usize = len( data )
	result: str = ''
	line: str = ''
	line_len: usize = 0
	i: usize = 0
	while i < n:
		b: u8 = src[i]
		piece: str = ''
		piece_len: usize = 0
		if _is_safe_byte( b ):
			with compiler.wrap_arithmetic:
				piece = chr( u32( b ))
			piece_len = 1
		else:
			hi: u8 = b >> 4
			lo: u8 = b & 0x0F
			piece = '=' + _hex_digit_char( hi ) + _hex_digit_char( lo )
			piece_len = 3
		with compiler.panic_arithmetic( 'line_len + a 1-or-3-byte piece cannot overflow usize' ):
			prospective: usize = line_len + piece_len
		if prospective > 76:
			result = result + line + '=\r\n'
			line = ''
			line_len = 0
		line = line + piece
		with compiler.wrap_arithmetic:
			line_len += piece_len
			i += 1
	result = result + line
	return result
