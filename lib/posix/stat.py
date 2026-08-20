# lib/posix/stat.py — stat(2) (sys/stat.h), just enough for os.path.isdir():
# struct stat's own layout is opaque/libc-dependent (varies by platform and
# even by _FILE_OFFSET_BITS), so this only ever touches st_mode via
# compiler.c_field, same posture as lib/posix/dirent.py.

import compiler

stat_t = compiler.c_type( 'struct stat', header = 'sys/stat.h' )

S_IFMT:  u32 = compiler.cexpr( 'S_IFMT',  'sys/stat.h', u32 )
S_IFDIR: u32 = compiler.cexpr( 'S_IFDIR', 'sys/stat.h', u32 )

@extern( 'c', 'stat', header = 'sys/stat.h' )
def stat(
	# ConstPtr[None], not ConstPtr[u8] - see lib/posix/dirent.py's opendir
	# for why (gcc hard-errors on char*/unsigned-char* mismatches once
	# header= makes the real prototype visible; void* sidesteps it).
	path: ConstPtr[None],
	buf: Ptr[stat_t],
) -> i32:
	...
