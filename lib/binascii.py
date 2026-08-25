# binascii - hexlify/unhexlify, mirroring Python's binascii module.
#
# Scope for this pass (per user request): hexlify()/unhexlify() only. No
# a2b_*/b2a_* aliases, no CRC helpers, no sep=/bytes_per_sep= chunking.
#
# Algorithmically identical to lib/base64.py's b16encode/b16decode, but
# with different policy: hexlify always emits LOWERCASE (base64.b16encode
# defaults to uppercase), and unhexlify always accepts BOTH cases
# unconditionally (base64.b16decode rejects lowercase unless
# casefold=True). Since base64.py's hex tables/helpers are private and
# this module's case policy differs anyway, this is a small standalone
# module rather than a wrapper - same idioms, own tables.

class BinasciiError:
	message: str

	def __init__( self, message: str ) -> None:
		self.message = message

	def __str__( self ) -> str:
		return self.message

	def __repr__( self ) -> str:
		return f"BinasciiError('{self.message}')"


_HEX_LOWER: list[u8] = [
	0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
	0x38, 0x39, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66,
]


def hexlify( data: bytes|bytearray ) -> bytes:
	n: usize = len( data )
	in_ptr: ConstPtr[u8] = data.get_const_ptr()

	with compiler.panic_arithmetic( 'output length is exactly double input length' ):
		out_len: usize = n * 2
	out = bytearray( out_len )
	out_ptr: Ptr[u8] = out.get_ptr()

	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by n, cannot overflow' ):
		while i < n:
			b: u8 = in_ptr[i]
			out_ptr[i * 2] = _HEX_LOWER.__getitem__( usize( b >> 4 ) ).unwrap( 'nibble always < 16' )
			out_ptr[i * 2 + 1] = _HEX_LOWER.__getitem__( usize( b & 0x0F ) ).unwrap( 'nibble always < 16' )
			i += 1

	return bytes.from_bytearray( move( out ) )


def unhexlify( data: bytes|bytearray ) -> Result[bytes,BinasciiError]:
	n: usize = len( data )
	in_ptr: ConstPtr[u8] = data.get_const_ptr()

	with compiler.panic_arithmetic( 'divisor is a nonzero literal' ):
		is_odd: bool = n % 2 != 0
	if is_odd:
		return Result.Err( BinasciiError( 'Odd-length string' ) )

	with compiler.panic_arithmetic( 'output length is exactly half input length' ):
		out_len: usize = n // 2
	out = bytearray( out_len )
	out_ptr: Ptr[u8] = out.get_ptr()

	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by n, cannot overflow' ):
		while i < n:
			hi: u8 = _hex_nibble_value( in_ptr[i] ).or_return()
			lo: u8 = _hex_nibble_value( in_ptr[i + 1] ).or_return()
			out_ptr[i // 2] = ( hi << 4 ) | lo
			i += 2

	return Result.Ok( bytes.from_bytearray( move( out ) ) )


def _hex_nibble_value( c: u8 ) -> Result[u8,BinasciiError]:
	''' Unlike base64.py's _hex_nibble_value, both cases are always
	accepted - there's no casefold= gate, matching real binascii's
	always-case-insensitive unhexlify(). '''
	if c >= 0x30 and c <= 0x39: # '0'-'9'
		with compiler.wrap_arithmetic:
			return Result.Ok( c - 0x30 )
	if c >= 0x41 and c <= 0x46: # 'A'-'F'
		with compiler.wrap_arithmetic:
			return Result.Ok( c - 0x41 + 10 )
	if c >= 0x61 and c <= 0x66: # 'a'-'f'
		with compiler.wrap_arithmetic:
			return Result.Ok( c - 0x61 + 10 )
	return Result.Err( BinasciiError( 'Non-hexadecimal digit found' ) )
