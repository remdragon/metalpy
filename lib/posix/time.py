# src/unix/time.py

import compiler

# ---------------------------------------------------------------------------
# High-resolution timing backend — lib/time.py's monotonic() and time().
#
# clock_gettime lives in libc on modern glibc/musl and macOS (>= 10.12).
# CLOCK_MONOTONIC (portable) is used rather than Linux-only CLOCK_MONOTONIC_RAW
# (absent on macOS). The clock-id macro values are NOT ABI-stable across libcs,
# so we fetch the real ones from <time.h> via compiler.cexpr - an isolated
# compile-time probe that does NOT pull <time.h> into the main translation unit
# (same pattern as crt.py's LC_CTYPE_MASK). Keeping <time.h> out of the main TU
# is deliberate: our own `timespec` @cstruct below would otherwise collide with
# the header's `struct timespec`. Its 2x i64 layout matches struct timespec on
# every 64-bit target (active_target.bits is always 64 - there is no 32-bit/x32
# target where time_t/long could be narrower).
# ---------------------------------------------------------------------------

CLOCK_REALTIME:  i32 = compiler.cexpr( 'CLOCK_REALTIME',  'time.h', i32 )
CLOCK_MONOTONIC: i32 = compiler.cexpr( 'CLOCK_MONOTONIC', 'time.h', i32 )

@cstruct
class timespec:
	tv_sec:  i64  # time_t on LP64
	tv_nsec: i64  # long   on LP64

# no header= -> our own prototype is emitted and <time.h> stays out of the main
# TU (see the timespec-collision note above). clockid_t is `int` on every target
# here, matching i32; the real symbol resolves against libc at link time.
@extern( 'c', 'clock_gettime' )
def clock_gettime(
	clockid: i32,
	tp: Ptr[timespec],
) -> i32:
	...


def get_local_timezone_name() -> str:
	# sys_readlink syscall or libc wrapper
	from posix.fs import readlink
	target_path = readlink( '/etc/localtime' ).unwrap_or()
	
	if target_path:
		idx = target_path.find( 'zoneinfo/' )
		if idx != -1:
			return target_path[idx+9:]
	
	return _read_etc_timezone_file()


def _read_etc_timezone_file() -> str:
	if f := open( '/etc/timezone', 'r' ).unwrap_or():
		with defer:
			f.close()
		if s := f.read( 128 ):
			return s.strip()
	
	return 'UTC'
