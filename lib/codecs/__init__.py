import sys

codec_registry: dict[str,Codec]|None = None

class CodecError: # ( sys.Error ):
	encoding: str
	message: str

	def __init__( self, encoding: str, message: str = '' ) -> None:
		self.encoding = encoding
		self.message = message

	def __str__( self ) -> str:
		return f'{self.encoding} codec error: {self.message}'

	def __repr__( self ) -> str:
		return f"CodecError({self.encoding!r}, {self.message!r})"

@enum( u8 )
class DecodeErrors:
	''' mirrors Python's bytes.decode(errors=...); 'strict' isn't a member
	here since Codec.decode() already IS the strict/fallible form - these
	are only the infallible recovery policies used by decode_lossy(). '''
	Replace = 0
	Ignore = 1
	BackslashReplace = 2

def _emit_lossy_unit( out_ptr: Ptr[u8], out_idx: usize, byte: u8, errors: DecodeErrors ) -> usize:
	''' applies one DecodeErrors policy to a single malformed raw byte and
	writes its (possibly empty) UTF-8 replacement into out_ptr, returning
	the new out_idx. Shared by every codec's decode_lossy() so recovery
	behaves identically regardless of source encoding. A malformed unit
	wider than 1 byte (e.g. a bad UTF-16 code unit) is handled by calling
	this once per raw byte - simpler than grouping, and still well-defined,
	at the cost of not always byte-for-byte matching CPython's own grouping
	in multi-byte error cases. '''
	with compiler.panic_arithmetic( 'out_idx bounded by the caller-sized worst-case buffer' ):
		if errors == DecodeErrors.Ignore:
			return out_idx
		if errors == DecodeErrors.Replace:
			# U+FFFD REPLACEMENT CHARACTER, encoded as UTF-8 (EF BF BD)
			out_ptr[out_idx] = 0xEF
			out_ptr[out_idx + 1] = 0xBF
			out_ptr[out_idx + 2] = 0xBD
			return out_idx + 3
		# BackslashReplace: \xHH
		hex_digits = '0123456789abcdef'
		out_ptr[out_idx] = u8( ord( '\\' ))
		out_ptr[out_idx + 1] = u8( ord( 'x' ))
		hi: str = hex_digits.__getitem__( usize( byte >> 4 )).unwrap( 'nibble is 0..15' )
		lo: str = hex_digits.__getitem__( usize( byte & 0x0F )).unwrap( 'nibble is 0..15' )
		out_ptr[out_idx + 2] = u8( ord( hi ))
		out_ptr[out_idx + 3] = u8( ord( lo ))
		return out_idx + 4

class Codec:
	@abstractmethod
	def encode( self, s: str ) -> Result[bytes,CodecError]:
		...

	@abstractmethod
	def decode( self, b: bytes|bytearray ) -> Result[str,CodecError]:
		...

	# like decode(), but never fails: malformed bytes are handled per
	# `errors` instead of returning Result.Err
	@abstractmethod
	def decode_lossy( self, b: bytes|bytearray, errors: DecodeErrors = DecodeErrors.BackslashReplace ) -> str:
		...

	@abstractmethod
	def names( self ) -> list[str]:
		...
	
	@classmethod
	def get( cls, encoding: str ) -> Result[Codec,CodecError]:
		global codec_registry
		if codec_registry is None:
			_build_registry()
		codec = codec_registry.get( encoding )
		if codec is None:
			return Result.Err( CodecError( encoding, 'codec not found' ))
		return Result.Ok( codec )
	
	def register( self ) -> None:
		global codec_registry
		if codec_registry is None:
			codec_registry = {}
		for name in self.names():
			codec_registry[name] = self


def _build_registry() -> None:
	global codec_registry
	codec_registry = {}
	from .ascii import ascii
	ascii().register()
	from .cp437 import cp437
	cp437().register()
	from .latin1 import latin1
	latin1().register()
	from .utf8 import utf8
	utf8.register()

def encode( s: str, encoding: str ) -> Result[bytes,CodecError]:
	codec = Codec.get( encoding ).or_return()
	return codec.encode( s )

def decode( b: bytes, encoding: str ) -> Result[str,CodecError]:
	codec = Codec.get( encoding ).or_return()
	return codec.decode( b )
	