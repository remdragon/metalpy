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
	written: isize = _crt_write( fd, buf, count )
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
