from typing import TypeAlias

ENOENT: i32 = 2
EACCESS: i32 = 13
EINVAL: i32 = 22
ECONNREFUSED: i32 = 111

def _posix_strerror( errnum: i32 ) -> str:
	from crt import strerror
	raw_ptr: ConstPtr[u8] = strerror( errnum )
	if raw_ptr is None:
		return 'Unknown Error'
	return str.from_cstr( raw_ptr )

@enum( i32 )
class PosixError:
	# TODO FIXME: pulling in a constant as the value is not supported yet
	NotFound = 2 # ENOENT
	PermissionDenied = 13 # EACCESS
	Invalid = 22 # EINVAL
	ConnectionRefused = 111 # ECONNREFUSED
	Other = _
	
	# TODO FIXME: implement method support for enums later...
	#def __str__( self ) -> str:
	#	return _posix_strerror( self.value )
