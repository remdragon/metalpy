import sys
from . import Codec, CodecError

class utf8( Codec ):
	@virtual
	def names( self ) -> list[str]:
		return [ 'utf8', 'utf-8', 'UTF8', 'UTF-8' ]

	@virtual
	def encode( self, s: str ) -> Result[bytes,CodecError]:
		'''
		this function can't actually generate a CodecError because str's underlying data is guaranteed to be utf-8
		However, the base class Codec.encode() function definition requires it
		'''
		length: usize = s.byte_len()
		out = bytearray( length )
		sys.memcpy( out, s.get_ptr(), length )
		return Result.Ok( bytes.from_bytearray( move( out )))

	@virtual
	def decode( self, b: bytes ) -> Result[str,CodecError]:
		# NOTE: there's no need to check for valid utf-8 encoding here, because str.from_cstr() has to do it anyway
		return str.from_cstr( b.get_bytes(), len( b ))
