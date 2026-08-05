import compiler

@compiler.target( os = 'windows' )
@enum( u32 )
class OSError: # windows version of base class for I/O errors
	FileNotFoundError = 2   # ERROR_FILE_NOT_FOUND
	AccessDenied      = 5   # ERROR_ACCESS_DENIED
	BrokenPipe        = 232 # ERROR_NO_DATA
	Invalid           = 6   # ERROR_INVALID_HANDLE
	Other = _

@compiler.target( os = not 'windows' )
@enum( i32 )
class OSError: # linux version of base class for I/O errors
	FileNotFoundError = 2   # ENOENT
	AccessDenied      = 13  # EACCES
	BrokenPipe        = 32  # EPIPE
	Invalid           = 22  # EINVAL
	Other = _
