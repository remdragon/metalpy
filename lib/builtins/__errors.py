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
	Interrupted       = 10004 # WSAEINTR - lib/tcp.py maps a
	                          # reactor.WaitError.Shutdown onto this: the
	                          # wait genuinely was interrupted before the
	                          # I/O it was standing in for completed.
	# getaddrinfo()'s own failures genuinely ARE WSA-error-code-compatible on
	# Windows (unlike POSIX - see the i32 branch below), so this is the real
	# WSAHOST_NOT_FOUND value, not a synthetic placeholder. Only the single
	# most common DNS-failure code gets its own name here, matching how every
	# other member above only covers one named code while everything else
	# (WSATRY_AGAIN, WSANO_RECOVERY, etc.) still falls to Other - see
	# lib/socket.py's _resolve_v4/_resolve_v6 for the only current producer.
	NameResolutionFailed = 11001 # WSAHOST_NOT_FOUND
	AlreadyExists     = 183 # ERROR_ALREADY_EXISTS - os.mkdir() on an existing path
	DirectoryNotEmpty = 145 # ERROR_DIR_NOT_EMPTY - os.rmdir() on a non-empty dir
	NotADirectory     = 267 # ERROR_DIRECTORY - os.listdir() on a non-directory path
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
	Interrupted       = 4   # EINTR - lib/tcp.py maps a
	                        # reactor.WaitError.Shutdown onto this: the wait
	                        # genuinely was interrupted before the I/O it
	                        # was standing in for completed.
	# NOT a real errno - getaddrinfo()'s own failures on POSIX are EAI_*
	# codes, a wholly separate (small, negative) namespace from errno that
	# doesn't map onto this enum's errno-based values at all (see
	# lib/socket.py's _resolve_v4/_resolve_v6, the only current producer of
	# this member). 1000 is a synthetic tag chosen to sit far outside every
	# real errno on any supported POSIX target (Linux's own errno.h tops out
	# under 140) so it can never collide with a genuine OS-reported errno.
	NameResolutionFailed = 1000
	AlreadyExists     = 17 # EEXIST - os.mkdir() on an existing path
	DirectoryNotEmpty = 39 # ENOTEMPTY (Linux value; macOS is 66) - os.rmdir() on a non-empty dir
	NotADirectory     = 20 # ENOTDIR - os.listdir() on a non-directory path
	Other = _
