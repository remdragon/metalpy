# lib/posix/fcntl.py — fcntl(2) extern declaration, just enough for
# non-blocking mode (F_GETFL/F_SETFL/O_NONBLOCK) - not a general fcntl
# binding.

import compiler

F_GETFL:    i32 = 3
F_SETFL:    i32 = 4
O_NONBLOCK: i32 = 0x800   # Linux value - see this file's own header note

# fcntl is variadic in the real header (third arg's type depends on cmd) -
# header='fcntl.h' would fail to match a fixed-arity prototype against a
# variadic one (same class of problem this codebase's other header=
# externs avoid by NOT using header= when the real signature can't be
# pinned down - see lib/windows/ws2_32.py's WSAStartup comment). Declared
# here with a plain i32 third argument instead (the only shape this module
# ever calls it with - F_GETFL takes none, F_SETFL takes the new flags as
# an int) - correct for every real call site below, not a general binding.
@extern( 'c', 'fcntl' )
def fcntl(
	fd: i32,
	cmd: i32,
	arg: i32,
) -> i32:
	...
