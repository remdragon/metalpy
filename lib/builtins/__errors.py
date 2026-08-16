import compiler

@compiler.target( os = 'windows' )
@enum( u32 )
class OSError: # windows version of base class for I/O errors
	FileNotFoundError = 2     # ERROR_FILE_NOT_FOUND
	AccessDenied      = 5     # ERROR_ACCESS_DENIED
	BrokenPipe        = 232   # ERROR_NO_DATA
	Invalid           = 6     # ERROR_INVALID_HANDLE
	ConnectionRefused = 10061 # WSAECONNREFUSED
	ConnectionReset   = 10054 # WSAECONNRESET
	TimedOut          = 10060 # WSAETIMEDOUT
	AddressInUse      = 10048 # WSAEADDRINUSE
	WouldBlock        = 10035 # WSAEWOULDBLOCK
	Other = _

@compiler.target( os = not 'windows' )
@enum( i32 )
class OSError: # linux version of base class for I/O errors
	FileNotFoundError = 2   # ENOENT
	AccessDenied      = 13  # EACCES
	BrokenPipe        = 32  # EPIPE
	Invalid           = 22  # EINVAL
	ConnectionRefused = 111 # ECONNREFUSED
	ConnectionReset   = 104 # ECONNRESET
	TimedOut          = 110 # ETIMEDOUT
	AddressInUse      = 98  # EADDRINUSE
	WouldBlock        = 11  # EWOULDBLOCK == EAGAIN on Linux
	Other = _
