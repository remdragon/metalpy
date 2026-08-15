from . import Codec, CodecError

class ascii( Codec ):
	@virtual
	def names( self ) -> list[str]:
		return [ 'ascii', 'us-ascii', 'US-ASCC', 'cp646' ]

	@virtual
	def encode( self, s: str ) -> Result[bytes,CodecError]:
		length: usize = s.byte_len()
		out = bytearray( length )
		
		s_ptr: ConstPtr[u8] = s.get_ptr()
		out_ptr: Ptr[u8] = out.get_ptr()
		
		i: usize = 0
		while i < length:
			byte: u8 = s_ptr[i]
			if byte > 0x7F:
				return Result.Err(
					CodecError( 'ascii', 'Ordinal out of range(128)' )
				)
			out_ptr[i] = byte
			i += 1
		
		return Result.Ok( bytes.from_bytearray( move( out )))
	
	@virtual
	def decode( self, b: bytes|bytearray ) -> Result[str,CodecError]:
		length: usize = len( b )
		ptr: ConstPtr[u8] = b.get_ptr()
		
		i: usize = 0
		while i < length:
			if ptr[i] > 0x7F:
				return Result.Err(
					CodecError( 'ascii', 'Ordinal out of range(128)' )
				)
			i += 1
		
		return Result.Ok( str.from_cstr( ptr, length ))
