'''
Portable file I/O primitives.

write_raw: single-syscall thin wrapper around the OS write primitive.
	Returns the number of bytes actually written. On Windows the byte
	count is saturated at u32::MAX (a single WriteFile limitation);
	callers that need to write larger buffers must loop (see write_all).

write_all: loops write_raw until all bytes are written or an error occurs.
	Returns Ok(None) on success — this is what most callers want.
'''

import compiler

# FD is the OS file-descriptor type: a HANDLE (Ptr[None]) on Windows, an
# i32 on POSIX. The top-level if/else collapses at compile time via
# compile_time_transformer.transform_stmt_list, so discovery sees exactly
# one concrete TypeAlias for the active target — write_all declares one
# fd: FD parameter and type inference from the argument handles the rest.
if compiler.target.os == 'windows':
	FD: TypeAlias = Ptr[None]
else:
	FD: TypeAlias = i32


# ---------------------------------------------------------------------------
# write_raw — thin OS wrapper, single syscall
# ---------------------------------------------------------------------------

@compiler.target( os = 'windows' )
def write_raw(
	fd: Ptr[None],
	buf: ConstPtr[u8],
	count: usize,
) -> Result[usize, OSError]:
	''' Windows: single WriteFile call. Byte count is saturated at
	u32::MAX; callers writing >4GB must loop (see write_all). '''
	from windows.kernel32 import GetLastError, WriteFile
	with compiler.saturate_arithmetic:
		to_write: u32 = u32( count )
	written: u32 = 0
	success: bool = WriteFile(
		fd, buf, to_write,
		compiler.addrof( written ),
		None, # lpOverlapped
	)
	if success:
		with compiler.wrap_arithmetic:
			return Result.Ok( usize( written ))
	else:
		return Result.Err( OSError( GetLastError() ))


@compiler.target( os = not 'windows' )
def write_raw(
	fd: i32,
	buf: ConstPtr[u8],
	count: usize,
) -> Result[usize, OSError]:
	''' POSIX: single write(2) call. '''
	from crt import get_errno, write as _crt_write
	written: isize = _crt_write( fd, compiler.cast( ConstPtr[None], buf ), count )
	if written < isize( 0 ):
		return Result.Err( OSError( get_errno() ))
	else:
		# written >= 0, safe to reinterpret as unsigned
		with compiler.wrap_arithmetic:
			return Result.Ok( usize( written ))


# ---------------------------------------------------------------------------
# write_all — loop until done or error
# ---------------------------------------------------------------------------

def write_all(
	fd: FD,
	buf: ConstPtr[u8],
	count: usize,
) -> Result[None, OSError]:
	remaining: usize = count
	offset: ConstPtr[u8] = buf
	while remaining > 0:
		n: usize = write_raw( fd, offset, remaining ).or_return()
		with compiler.wrap_arithmetic:
			offset = offset + n
			remaining -= n
	return Result.Ok( None )


# ---------------------------------------------------------------------------
# Platform sentinels and constants
# ---------------------------------------------------------------------------

if compiler.target.os == 'windows':
	INVALID_FD: FD = -1 # INVALID_HANDLE_VALUE
else:
	INVALID_FD: i32 = i32( -1 )

# POSIX open flags
if compiler.target.os != 'windows':
	O_RDONLY: i32 = 0o00
	O_WRONLY: i32 = 0o01
	O_RDWR: i32 = 0o02
	O_CREAT: i32 = 0o100
	O_EXCL: i32 = 0o200
	O_TRUNC: i32 = 0o1000
	O_APPEND: i32 = 0o2000

# Portable seek-whence constants. Values deliberately match Windows'
# FILE_BEGIN/FILE_CURRENT/FILE_END (0/1/2) too, so seek_raw's Windows variant
# needs nothing more than a cast to accept the same i32 on both platforms -
# lib/io.py's Seekable protocol relies on this to have one signature, not a
# per-platform whence type.
SEEK_SET: i32 = 0
SEEK_CUR: i32 = 1
SEEK_END: i32 = 2

# Windows file API constants
if compiler.target.os == 'windows':
	GENERIC_READ: u32 = 0x80000000
	GENERIC_WRITE: u32 = 0x40000000
	FILE_SHARE_READ: u32 = 0x00000001
	FILE_SHARE_WRITE: u32 = 0x00000002
	CREATE_NEW: u32 = 1
	CREATE_ALWAYS: u32 = 2
	OPEN_EXISTING: u32 = 3
	OPEN_ALWAYS: u32 = 4
	TRUNCATE_EXISTING: u32 = 5
	FILE_ATTRIBUTE_NORMAL: u32 = 0x80
	FILE_BEGIN: u32 = 0
	FILE_CURRENT: u32 = 1
	FILE_END: u32 = 2


# ---------------------------------------------------------------------------
# open_raw — thin OS wrapper, returns an open FD
# ---------------------------------------------------------------------------

@compiler.target( os = 'windows' )
def open_raw(
	path: ConstPtr[u8],
	access: u32,
	creation: u32,
) -> Result[FD, OSError]:
	''' Windows: CreateFileA call. Returns a HANDLE on success.

	NOTE: CreateFileA uses the system's active ANSI code page, not UTF-8.
	Non-ASCII paths will be mangled. For full Unicode support, this should
	use CreateFileW + MultiByteToWideChar (or manual UTF-8→UTF-16 conversion).
	'''
	from windows.kernel32 import CreateFileA, GetLastError, INVALID_HANDLE_VALUE
	handle: FD = CreateFileA(
		path, access,
		FILE_SHARE_READ | FILE_SHARE_WRITE,
		None, creation,
		FILE_ATTRIBUTE_NORMAL,
		INVALID_HANDLE_VALUE,
	)
	if handle == INVALID_HANDLE_VALUE:
		return Result.Err( OSError( GetLastError() ))
	return Result.Ok( handle )


@compiler.target( os = not 'windows' )
def open_raw(
	path: ConstPtr[u8],
	flags: i32,
	mode: i32,
) -> Result[FD, OSError]:
	''' POSIX: open(2) call. Returns a file descriptor on success. '''
	from crt import open, get_errno
	fd: i32 = open( path, flags, mode )
	if fd < 0:
		return Result.Err( OSError( get_errno() ))
	return Result.Ok( fd )


# ---------------------------------------------------------------------------
# close_raw — thin OS wrapper, closes an FD
# ---------------------------------------------------------------------------

@compiler.target( os = 'windows' )
def close_raw(
	fd: FD,
) -> Result[None, OSError]:
	''' Windows: CloseHandle call. '''
	from windows.kernel32 import CloseHandle, GetLastError
	success: bool = CloseHandle( fd )
	if not success:
		return Result.Err( OSError( GetLastError() ))
	return Result.Ok( None )


@compiler.target( os = not 'windows' )
def close_raw(
	fd: i32,
) -> Result[None, OSError]:
	''' POSIX: close(2) call. '''
	from crt import close, get_errno
	rc: i32 = close( fd )
	if rc < 0:
		return Result.Err( OSError( get_errno() ))
	return Result.Ok( None )


# ---------------------------------------------------------------------------
# read_raw — thin OS wrapper, single syscall
# ---------------------------------------------------------------------------

@compiler.target( os = 'windows' )
def read_raw(
	fd: Ptr[None],
	buf: Ptr[u8],
	count: usize,
) -> Result[usize, OSError]:
	''' Windows: single ReadFile call. Byte count is saturated at
	u32::MAX (same constraint as write_raw). '''
	from windows.kernel32 import GetLastError, ReadFile
	with compiler.saturate_arithmetic:
		to_read: u32 = u32( count )
	bytes_read: u32 = 0
	success: bool = ReadFile(
		fd, buf, to_read,
		compiler.addrof( bytes_read ),
		None, # lpOverlapped
	)
	if success:
		with compiler.wrap_arithmetic:
			return Result.Ok( usize( bytes_read ))
	else:
		return Result.Err( OSError( GetLastError() ))


@compiler.target( os = not 'windows' )
def read_raw(
	fd: i32,
	buf: Ptr[u8],
	count: usize,
) -> Result[usize, OSError]:
	''' POSIX: single read(2) call. '''
	from crt import get_errno, read
	nread: isize = read( fd, compiler.cast( Ptr[None], buf ), count )
	if nread < isize( 0 ):
		return Result.Err( OSError( get_errno() ))
	else:
		with compiler.wrap_arithmetic:
			return Result.Ok( usize( nread ))


# ---------------------------------------------------------------------------
# seek_raw — thin OS wrapper, reposition the file offset
# ---------------------------------------------------------------------------

@compiler.target( os = 'windows' )
def seek_raw(
	fd: Ptr[None],
	offset: i64,
	whence: i32,
) -> Result[i64, OSError]:
	''' Windows: SetFilePointerEx call. Returns the resulting absolute file
	position. '''
	from windows.kernel32 import GetLastError, SetFilePointerEx
	new_pos: i64 = 0
	success: bool = SetFilePointerEx( fd, offset, compiler.addrof( new_pos ), u32( whence ))
	if not success:
		return Result.Err( OSError( GetLastError() ))
	return Result.Ok( new_pos )


@compiler.target( os = not 'windows' )
def seek_raw(
	fd: i32,
	offset: i64,
	whence: i32,
) -> Result[i64, OSError]:
	''' POSIX: lseek(2) call. Returns the resulting absolute file position. '''
	from crt import get_errno, lseek
	result: i64 = lseek( fd, offset, whence )
	if result < 0:
		return Result.Err( OSError( get_errno() ))
	return Result.Ok( result )


# ---------------------------------------------------------------------------
# truncate_raw — thin OS wrapper, set file size
# ---------------------------------------------------------------------------

@compiler.target( os = 'windows' )
def truncate_raw(
	fd: Ptr[None],
) -> Result[None, OSError]:
	''' Windows: SetEndOfFile call. Truncates to the current file pointer
	position. Caller must seek to the desired size first. '''
	from windows.kernel32 import GetLastError, SetEndOfFile
	success: bool = SetEndOfFile( fd )
	if not success:
		return Result.Err( OSError( GetLastError() ))
	return Result.Ok( None )


@compiler.target( os = not 'windows' )
def truncate_raw(
	fd: i32,
	length: i64,
) -> Result[None, OSError]:
	''' POSIX: ftruncate(2) call. '''
	from crt import ftruncate, get_errno
	rc: i32 = ftruncate( fd, length )
	if rc < 0:
		return Result.Err( OSError( get_errno() ))
	return Result.Ok( None )
