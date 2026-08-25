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

class Codec:
	@abstractmethod
	def encode( self, s: str ) -> Result[bytes,CodecError]:
		...

	@abstractmethod
	def decode( self, b: bytes|bytearray ) -> Result[str,CodecError]:
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
	