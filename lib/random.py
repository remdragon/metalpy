'''
CSPRNG - cryptographically secure random bytes.

Windows: BCryptGenRandom (bcrypt.dll), hAlgorithm=NULL +
	BCRYPT_USE_SYSTEM_PREFERRED_RNG - no algorithm handle to open/close,
	Microsoft's own recommended shape for a one-off random fill.
Linux: getrandom(2) as the primary path (has_library-gated - glibc >= 2.25;
	older glibc/musl builds without the symbol fall back to reading
	/dev/urandom instead, same trust level).
macOS: not implemented - poison pill, same convention as lib/ssl.py's own
	macOS section (a real def, not an absent one, so anything that merely
	imports this module still compiles on macOS as long as it never calls in).
'''

import compiler
import fs

if compiler.target.os == 'windows':
	BCRYPT_USE_SYSTEM_PREFERRED_RNG: u32 = 0x00000002


@compiler.target( os = 'windows' )
@extern( 'bcrypt', 'BCryptGenRandom' )
def BCryptGenRandom(
	hAlgorithm: Ptr[None],
	pbBuffer:   Ptr[u8],
	cbBuffer:   u32,
	dwFlags:    u32,
) -> i32:  # NTSTATUS
	...

@compiler.target( os = 'windows' )
def _fill_random_raw( buf: Ptr[u8], count: usize ) -> Result[usize, OSError]:
	with compiler.saturate_arithmetic:
		to_fill: u32 = u32( count )
	status: i32 = BCryptGenRandom( None, buf, to_fill, BCRYPT_USE_SYSTEM_PREFERRED_RNG )
	if status != 0:  # STATUS_SUCCESS
		return Result.Err( OSError.Other )  # NTSTATUS isn't a Win32 error code, nothing better to map to
	with compiler.wrap_arithmetic:
		return Result.Ok( usize( to_fill ))


@compiler.target( os = not ( 'windows', 'macos' ), has_library = ( 'c', 'getrandom' ) )
@extern( 'c', 'getrandom' )
def getrandom( buf: Ptr[None], buflen: usize, flags: u32 ) -> isize:
	...

@compiler.target( os = not ( 'windows', 'macos' ), has_library = ( 'c', 'getrandom' ) )
def _fill_random_raw( buf: Ptr[u8], count: usize ) -> Result[usize, OSError]:
	''' getrandom(2) can do a short read for count > 256 if interrupted -
	caller (fill_random) loops. EINTR itself is retried here rather than
	surfaced, matching write_all/read_exact's own "keep going" posture. '''
	from crt import get_errno
	while True:
		n: isize = getrandom( compiler.cast( Ptr[None], buf ), count, u32( 0 ))
		if n < isize( 0 ):
			err: OSError = OSError( get_errno() )
			if err == OSError.Interrupted:
				continue
			return Result.Err( err )
		with compiler.wrap_arithmetic:
			return Result.Ok( usize( n ))

# no getrandom symbol - fall back to reading /dev/urandom directly.
@compiler.target( os = not ( 'windows', 'macos' ), has_library = not ( 'c', 'getrandom' ) )
def _fill_random_raw( buf: Ptr[u8], count: usize ) -> Result[usize, OSError]:
	path: str = "/dev/urandom"
	fd: fs.FD = fs.open_raw( path.get_cstr(), fs.O_RDONLY, 0 ).or_return()
	result: Result[usize, OSError] = fs.read_raw( fd, buf, count )
	fs.close_raw( fd ).is_ok()  # best-effort - a read that already succeeded still counts
	return result


@compiler.target( os = 'macos' )
def _fill_random_raw( buf: Ptr[u8], count: usize ) -> Result[usize, OSError]:
	return _MACOS_RANDOM_NOT_YET_IMPLEMENTED()


def fill_random( buf: Ptr[u8], count: usize ) -> Result[None, OSError]:
	''' fills buf[0:count) with cryptographically secure random bytes,
	looping the OS primitive since a single call isn't guaranteed to fill
	an arbitrarily large buffer in one shot. '''
	filled: usize = 0
	with compiler.panic_arithmetic( 'bounded by count, cannot overflow' ):
		while filled < count:
			with compiler.wrap_arithmetic:
				dest: Ptr[u8] = buf + filled
				remaining: usize = count - filled
			n: usize = _fill_random_raw( dest, remaining ).or_return()
			if n == 0:
				return Result.Err( OSError.Other )
			with compiler.wrap_arithmetic:
				filled += n
	return Result.Ok( None )


def random_bytes( count: usize ) -> Result[bytearray, OSError]:
	''' count cryptographically secure random bytes, as a fresh bytearray. '''
	out: bytearray = bytearray( count )
	fill_random( out.get_ptr(), count ).or_return()
	return Result.Ok( out )
