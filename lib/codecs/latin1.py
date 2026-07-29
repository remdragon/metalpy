from . import Codec, CodecError

class latin1( Codec ):
	def names( self ) -> list[str]:
		return [ 'latin1', 'latin-1', 'iso-8859-1', 'iso8859-1', '8859' ]
	
	def encode( self, s: str ) -> Result[bytes,CodecError]:
		s_len: usize = s.byte_len()
		s_ptr: ConstPtr[u8] = s.get_ptr()
		
		out = bytearray( s_len )
		out_ptr: Ptr[u8] = out.get_ptr()
		
		out_idx: usize = 0
		in_idx: usize = 0
		
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
		
		return Result.Ok( bytes.from_bytearray( bytes, move( out )))
	
	def decode( self, b: bytes ) -> Result[str,CodecError]:
		b_len: usize = len( b )
		b_ptr: ConstPtr[u8] = b.get_ptr()
		
		# Max size: 2 output bytes per 1 input byte (for 0x80..0xFF range)
		out = bytearray( b_len * 2 )
		out_ptr: Ptr[u8] = out.get_ptr()
		
		out_idx: usize = 0
		in_idx: usize = 0
		
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
		
		return Result.Ok( str.from_cstr( out_ptr, out_idx ))
