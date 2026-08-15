# src/unix/time.py

import compiler
from codecs.utf8 import utf8

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

# Field defaults let `timespec()` construct a zero-initialized value (the
# constructor otherwise requires every field) - which is exactly the clean slate
# we want before clock_gettime writes into it. Defaults are metalpy-side only;
# the emitted C struct layout is unchanged (two 64-bit fields = struct timespec
# on LP64).
@cstruct
class timespec:
	tv_sec:  i64 = 0  # time_t on LP64
	tv_nsec: i64 = 0  # long   on LP64

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
	# `open(path, mode)` was never a real name anywhere in this codebase -
	# the real primitives are File.binary_reader()/BinaryReader.read()
	# (lib/builtins/__File.py). match-based Result narrowing throughout
	# (not `.unwrap_or()` + `is not None`): `if x is not None: use(x)`
	# doesn't narrow for a plain if-statement in this compiler (confirmed
	# separately, unrelated to this function) - match is the one mechanism
	# already proven to work, same shape as str.partition's own
	# `match found: case Result.Ok(idx): ... case Result.Err(_): ...`.
	# codec=utf8() passed explicitly (not relying on decode()'s own
	# `codec: Codec = utf8` default) - that default is a real, separate,
	# already-known bug (the `utf8` CLASS used as a default value where an
	# INSTANCE belongs, same pattern as fs.py:12's own readlink() default -
	# PLAN_POSIX_FEATURE.md's own deferred item 2). No explicit reader.
	# close() either - not a correctness requirement (its own destructor
	# already closes the fd), and BinaryReader.close()'s own body has a
	# separate, real, pre-existing bug (an unchecked Result from
	# fs.close_raw(), lib/builtins/__File.py:40) - both sidestepped here
	# rather than fixed, out of scope for this function's own rewrite.
	codec = utf8()
	match File.binary_reader( '/etc/timezone' ):
		case Result.Ok( reader ):
			buf: bytearray = bytearray( 128 )
			match reader.read( buf.get_ptr(), 128 ):
				case Result.Ok( n ):
					if n == 0:
						return 'UTC'
					match codec.decode( buf[:n] ):
						case Result.Ok( s ):
							return s.strip()
						case Result.Err( _ ):
							return 'UTC'
				case Result.Err( _ ):
					return 'UTC'
		case Result.Err( _ ):
			return 'UTC'
