# lib/posix/dirent.py — opendir/readdir/closedir (dirent.h)
#
# DIR/struct dirent are opaque - internal layout is platform/libc-dependent
# (glibc vs musl vs macOS's own dirent all differ) - only ever touched via
# an opaque pointer + compiler.c_field_addr for the one field this codebase
# actually needs (d_name), same posture as ucontext_t/stack_t in
# lib/posix/pthread.py and lib/fiber.py.
#
# readdir(3) signals BOTH "no more entries" and "error" via a NULL return -
# telling them apart requires clearing errno before the call and checking it
# after. This binding does not do that (a deliberate simplification, not an
# oversight) - a NULL return is always treated as "iteration done", same as
# the common minimal-wrapper posture. Revisit if a caller ever needs real
# directory-read error reporting.

import compiler

DIR = compiler.c_type( 'DIR', header = 'dirent.h' )
dirent = compiler.c_type( 'struct dirent', header = 'dirent.h' )

@extern( 'c', 'opendir', header = 'dirent.h' )
def opendir(
	# ConstPtr[None], not ConstPtr[u8]: the real opendir() takes const
	# char*, and gcc (unlike clang/MSVC) hard-errors on passing
	# unsigned-char* where char* is expected (-Wincompatible-pointer-
	# types) once header= makes the real prototype visible - void*
	# implicitly bridges to/from any pointer type in C, sidestepping the
	# signedness distinction entirely. Callers cast their ConstPtr[u8].
	name: ConstPtr[None],
) -> Ptr[DIR]:
	...

@extern( 'c', 'readdir', header = 'dirent.h' )
def readdir( dirp: Ptr[DIR] ) -> Ptr[dirent]:
	...

@extern( 'c', 'closedir', header = 'dirent.h' )
def closedir( dirp: Ptr[DIR] ) -> i32:
	...
