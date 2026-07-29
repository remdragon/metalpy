# src/posix/fs.py

import crt
from codecs import Codec
from codecs.utf8 import utf8
from .errors import PosixError

_PATH_BUFFER_SIZE: usize = 4096

def readlink(
	path: str,
	codec: Codec = utf8,
) -> Result[str,PosixError|CodecError]:
	buf = bytearray( _PATH_BUFFER_SIZE )
	
	nbytes = crt.readlink(
		path.get_cstr(),
		buf.get_ptr(),
		_PATH_BUFFER_SIZE,
	)
	if nbytes < 0:
		return Result.Err( PosixError( crt.get_errno() ))
	
	return codec.decode( buf[:nbytes] )
