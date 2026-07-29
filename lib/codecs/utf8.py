import sys
from . import Codec, CodecError

class utf8( Codec ):
	def names( self ) -> list[str]:
		return [ 'utf8', 'utf-8', 'UTF8', 'UTF-8' ]
	
	def encode( self, s: str ) -> Result[bytes,CodecError]:
		length: usize = s.byte_len()
		out = bytearray( length )
		sys.memcpy( out, s.get_ptr(), length )
		return Result.Ok( bytes.from_bytearray( move( out )))
	
	def decode( self, b: bytes ) -> Result[str,CodecError]:
		length: usize = len( b )
		ptr: ConstPtr[u8] = b.get_bytes()
		i: usize = 0
		
		while i < length:
			byte: u8 = ptr[i]
			
			if byte <= 0x7F: # ASCII fast path:
				i += 1
			
			elif ( byte & 0xE0 ) == 0xC0: # 2-Byte Sequence (110xxxxx 10xxxxxx):
				if byte < 0xC2: # reject over-long encodings
					return Result.Err( CodecError( 'utf-8', 'Invalid UTF-8 lead byte (overlong)' ))
				if i + 1 >= length or ( ptr[i+1] & 0xC0 ) != 0x80:
					return Result.Err( CodecError( 'utf-8', 'Truncated or invalid 2-byte sequence' ))
				i += 2
			
			elif ( byte & 0xF0 ) == 0xE0: # 3-Byte Sequence (1110xxxx 10xxxxxx 10xxxxxx)
				if i + 2 >= length:
					return Result.Err(
						CodecError( 'utf-8', 'Truncated 3-byte sequence' )
					)
				next1: u8 = ptr[i + 1]
				next2: u8 = ptr[i + 2]
				
				if ( next1 & 0xC0 ) != 0x80 or ( next2 & 0xC0 ) != 0x80:
					return Result.Err(
						CodecError( 'utf-8', 'Invalid 3-byte continuation' )
					)
				
				# Reject UTF-16 surrogates (0xED 0xA0..0xBF)
				if byte == 0xED and next1 >= 0xA0:
					return Result.Err(
						CodecError( 'utf-8', 'UTF-16 surrogate range forbidden in UTF-8' )
					)
				
				i += 3
			
			elif (byte & 0xF8) == 0xF0: # 4-Byte Sequence (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
				if byte > 0xF4:  # Beyond Unicode max codepoint 0x10FFFF
					return Result.Err(
						CodecError( 'utf-8', 'Codepoint out of Unicode range' )
					)
				if i + 3 >= length:
					return Result.Err(
						CodecError( 'utf-8', 'Truncated 4-byte sequence' )
					)
					
				next1: u8 = ptr[i+1]
				next2: u8 = ptr[i+2]
				next3: u8 = ptr[i+3]
				
				if (
					(next1 & 0xC0) != 0x80
					or (next2 & 0xC0) != 0x80
					or (next3 & 0xC0) != 0x80
				):
					return Result.Err(
						CodecError( 'utf-8', 'Invalid 4-byte continuation' )
					)
				
				i += 4
			
			else: # Invalid lead byte (e.g. 0x80-0xBF as lead, or 0xF5-0xFF)
				return Result.Err(
					CodecError( 'utf-8', 'Invalid lead byte' )
				)
		
		s = str.from_cstr( ptr, length )
		return Result.Ok( s )
