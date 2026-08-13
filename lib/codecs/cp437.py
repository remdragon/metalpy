import sys
from . import Codec, CodecError

DECODE_TABLE: list[u16] = [
	0x00C7, 0x00FC, 0x00E9, 0x00E2, 0x00E4, 0x00E0, 0x00E5, 0x00E7,
	0x00EA, 0x00EB, 0x00E8, 0x00EF, 0x00EE, 0x00EC, 0x00C4, 0x00C5,
	0x00C9, 0x00E6, 0x00C6, 0x00F4, 0x00F6, 0x00F2, 0x00FB, 0x00F9,
	0x00FF, 0x00D6, 0x00DC, 0x00A2, 0x00A3, 0x00A5, 0x20A7, 0x0192,
	0x00E1, 0x00ED, 0x00F3, 0x00FA, 0x00F1, 0x00D1, 0x00AA, 0x00BA,
	0x00BF, 0x2310, 0x00AC, 0x00BD, 0x00BC, 0x00A1, 0x00AB, 0x00BB,
	0x2591, 0x2592, 0x2593, 0x2502, 0x2524, 0x2561, 0x2562, 0x2556,
	0x2555, 0x2563, 0x2550, 0x2557, 0x255D, 0x255C, 0x255B, 0x2510,
	0x2514, 0x2534, 0x252C, 0x251C, 0x2500, 0x253C, 0x255E, 0x255F,
	0x255A, 0x2554, 0x2569, 0x2566, 0x2560, 0x2550, 0x256C, 0x2567,
	0x2568, 0x2564, 0x2565, 0x2559, 0x2558, 0x2552, 0x2553, 0x256B,
	0x256A, 0x2518, 0x250C, 0x2588, 0x2584, 0x258C, 0x2590, 0x2580,
	0x03B1, 0x00DF, 0x0393, 0x03C0, 0x03A3, 0x03C3, 0x03BC, 0x03C4,
	0x03A6, 0x0398, 0x03A9, 0x03B4, 0x221E, 0x03C6, 0x03B5, 0x2229,
	0x2261, 0x00B1, 0x2265, 0x2264, 0x2320, 0x2321, 0x00F7, 0x2248,
	0x00B0, 0x2219, 0x00B7, 0x221A, 0x207F, 0x00B2, 0x25A0, 0x00A0
]

class cp437( Codec ):
	@virtual
	def names( self ) -> list[str]:
		return [ 'cp437', 'ibm437', 'IBM437', 'cspc8codecpage437' ]

	@virtual
	def encode( self, s: str ) -> Result[bytes,CodecError]:
		s_len: usize = s.byte_len()
		s_ptr: ConstPtr[u8] = s.get_ptr()
		
		out = bytearray( s_len )
		out_ptr: Ptr[u8] = out.get_ptr()
		
		out_idx: usize = 0
		in_idx: usize = 0
		
		with compiler.panic_arithmetic: # an overflow should be impossible based on the if checks
			while in_idx < s_len:
				b0: u8 = s_ptr[in_idx]
				
				if b0 <= 0x7F: # fast path: ASCII 0x00..0x7F
					out_ptr[out_idx] = b0
					out_idx += 1
					in_idx += 1
					continue
				
				cp: u16 = 0
				bytes_read: usize = 0
				
				if ( b0 & 0xE0 ) == 0xC0:
					if in_idx + 1 >= s_len:
						return Result.Err( CodecError( 'cp437',
							'Truncated UTF-8 sequence'
						))
					cp = (u16(b0 & 0x1F) << 6) | (s_ptr[in_idx + 1] & 0x3F)
					bytes_read = 2
				elif ( b0 & 0xF0 ) == 0xE0:
					if in_idx + 2 >= s_len:
						return Result.Err( CodecError( 'cp437',
							'Truncated UTF-8 sequence'
						))
					cp = (
						u16(b0 & 0x0F) << 12
						| ( u16(s_ptr[in_idx + 1] & 0x3F) << 6 )
						| (s_ptr[in_idx + 2] & 0x3F)
					)
					bytes_read = 3
				else:
					return Result.Err( CodecError( 'cp437',
						'Character outside CP437 range'
					))
				
				# Scan DECODE_TABLE for matching codepoint
				found: bool = False
				i: usize = 0
				while i < 128:
					if DECODE_TABLE[i] == cp:
						out_ptr[out_idx] = u8( i + 0x80 )
						out_idx += 1
						found = True
						break
					i += 1
				
				if not found:
					return Result.Err( CodecError( 'cp437',
						'Character outside CP437 range'
					))
				
				in_idx += bytes_read
		
		return Result.Ok( bytes.from_bytearray( bytes, move( out )))
	
	@virtual
	def decode( self, b: bytes ) -> Result[str,CodecError]:
		b_len: usize = len( b )
		b_ptr: ConstPtr[u8] = b.get_ptr()
		
		# Worst-case allocation: 3 UTF-8 output bytes per 1 CP437 input byte
		out = bytearray( b_len * 3 )
		out_ptr: Ptr[u8] = out.get_ptr()
		
		out_idx: usize = 0
		in_idx: usize = 0
		
		while in_idx < b_len:
			byte: u8 = b_ptr[in_idx]
			
			if byte <= 0x7F: # Standard ASCII passthrough
				out_ptr[out_idx] = byte
				out_idx += 1
			else:
				# Map extended byte to UTF-8 codepoint via table lookup
				cp = DECODE_TABLE[usize(byte - 0x80)]
				
				if cp <= 0x07FF:
					out_ptr[out_idx]     = 0xC0 | u8( cp >> 6 )
					out_ptr[out_idx+1] = 0x80 | u8( cp & 0x3F )
					out_idx += 2
				else:
					out_ptr[out_idx]     = 0xE0 | u8( cp >> 12 )
					out_ptr[out_idx+1] = 0x80 | u8( (cp >> 6) & 0x3F )
					out_ptr[out_idx+2] = 0x80 | u8( cp & 0x3F )
					out_idx += 3
			
			in_idx += 1
		
		# Yield final string directly using internal raw length
		return Result.Ok(str.from_cstr(out_ptr, out_idx))