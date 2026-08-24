# src/posix/fs.py

import compiler
from codecs import Codec
from codecs.utf8 import utf8
from .errors import PosixError

_PATH_BUFFER_SIZE: usize = 4096

def readlink(
	path: str,
	codec: Codec = utf8,
) -> Result[str,PosixError|CodecError]:
	from crt import readlink as _crt_readlink, get_errno
	buf = bytearray( _PATH_BUFFER_SIZE )

	nbytes = _crt_readlink(
		compiler.cast( ConstPtr[None], path.get_cstr() ),
		compiler.cast( Ptr[None], buf.get_ptr() ),
		_PATH_BUFFER_SIZE,
	)
	if nbytes < 0:
		return Result.Err( PosixError( get_errno() ))

	with compiler.panic_arithmetic( 'nbytes already proven non-negative above' ):
		n: usize = usize( nbytes )
	return codec.decode( buf[:n] )
