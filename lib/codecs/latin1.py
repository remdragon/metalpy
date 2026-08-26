import sys
from . import Codec, CodecError, DecodeErrors

class latin1( Codec ):
	@virtual
	def names( self ) -> list[str]:
		return [ 'latin1', 'latin-1', 'iso-8859-1', 'iso8859-1', '8859' ]

	@virtual
	def encode( self, s: str ) -> Result[bytes,CodecError]:
		s_len: usize = s.byte_len()
		s_ptr: ConstPtr[u8] = s.get_const_ptr()

		out = bytearray( s_len )
		out_ptr: Ptr[u8] = out.get_ptr()

		out_idx: usize = 0
		in_idx: usize = 0

		with compiler.panic_arithmetic( 'bounded by s_len, cannot overflow' ):
			while in_idx < s_len:
				b0: u8 = s_ptr[in_idx]

				if b0 <= 0x7F:
					out_ptr[out_idx] = b0
					out_idx += 1
					in_idx += 1
				elif ( b0 & 0xE0 ) == 0xC0:
					if in_idx + 1 >= s_len:
						return Result.Err( CodecError( 'latin1',
							'Truncated UTF-8 sequence'
						))

					# Decode 2-byte UTF-8 codepoint (U+0080 .. U+07FF)
					cp: u16 = u16( ((u32(b0) & 0x1F) << 6) | (u32(s_ptr[in_idx + 1]) & 0x3F) )

					if cp > 0xFF:
						return Result.Err( CodecError( 'latin1',
							'Ordinal out of range(256)'
						))

					out_ptr[out_idx] = u8( cp )
					out_idx += 1
					in_idx += 2
				else:
					# Codepoints > U+07FF (3 or 4 byte UTF-8) exceed Latin-1 range
					return Result.Err( CodecError( 'latin1',
						'Ordinal out of range(256)'
					))

		# out was allocated to the worst-case size (s_len, one Latin-1 byte
		# per INPUT byte) but 2-byte UTF-8 sequences collapse to a single
		# output byte, so out_idx can be < s_len - copy down to a final
		# buffer sized to what was actually written. Safe to move() this
		# exactly-sized buffer (unlike out itself) since it's allocated
		# after every early-return above has already resolved - see
		# ascii.py's own encode() for the move()-past-a-still-reachable-
		# return compiler bug this sidesteps.
		final = bytearray( out_idx )
		sys.memcpy( final.get_ptr(), out_ptr, out_idx )
		return Result.Ok( bytes.from_bytearray( move( final )))
	
	@virtual
	def decode( self, b: bytes|bytearray ) -> Result[str,CodecError]:
		b_len: usize = len( b )
		b_ptr: ConstPtr[u8] = b.get_const_ptr()
		
		# Max size: 2 output bytes per 1 input byte (for 0x80..0xFF range)
		with compiler.panic_arithmetic( 'irrational byte length' ):
			out = bytearray( b_len * 2 )
		out_ptr: Ptr[u8] = out.get_ptr()

		out_idx: usize = 0
		in_idx: usize = 0

		with compiler.panic_arithmetic( 'bounded by b_len, cannot overflow' ):
			while in_idx < b_len:
				byte: u8 = b_ptr[in_idx]

				if byte <= 0x7F:
					out_ptr[out_idx] = byte
					out_idx += 1
				else:
					# Convert Latin-1 byte (0x80..0xFF) to 2-byte UTF-8 sequence
					out_ptr[out_idx]     = 0xC0 | u8( byte >> 6 )
					out_ptr[out_idx + 1] = 0x80 | u8( byte & 0x3F )
					out_idx += 2

				in_idx += 1

		# See cp437.py's decode() for why out_ptr/out_idx can't be handed to
		# str.from_cstr directly (its size means size INCLUDING the zero
		# terminator, and out's own worst-case buffer isn't terminated).
		with compiler.panic_arithmetic( 'irrational byte length' ):
			buf_size: usize = out_idx + 1
		new_buf: Ptr[u8] = sys.alloc[u8]( buf_size )
		sys.memcpy( new_buf, out_ptr, out_idx )
		new_buf[out_idx] = 0
		return str._from_owned_cstr( new_buf, buf_size )

	@virtual
	def decode_lossy( self, b: bytes|bytearray, errors: DecodeErrors = DecodeErrors.BackslashReplace ) -> str:
		# every byte 0x00..0xFF is a valid Latin-1 codepoint - decode() can
		# never fail, `errors` has nothing to act on
		return self.decode( b ).unwrap( 'latin1 decode() is infallible: every byte is a valid codepoint' )
