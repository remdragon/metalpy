import compiler

@compiler.target( os = 'windows' )
@enum( u32 )
class OSError: # windows version of base class for I/O errors
	FileNotFoundError = 2 # ERROR_FILE_NOT_FOUND
	Other = _

@compiler.target( os = not 'windows' )
@enum( i32 )
class OSError: # linux version of base class for I/O errors
	FileNotFoundError = 2 # ENOENT
	Other = _
