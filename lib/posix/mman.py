# lib/posix/mman.py — mmap/mprotect/munmap (sys/mman.h)
#
# Used to give a ucontext_t a real stack with a guard region: mmap a block
# PROT_NONE, then mprotect everything except the lowest page back to
# PROT_READ|PROT_WRITE, leaving the lowest page unreadable/unwritable so an
# overflow into it faults instead of silently corrupting adjacent memory.
#
# header='sys/mman.h' suppresses these prototypes in the generated C - the
# real signatures are used instead, same convention as pthread.py.
#
# PROT_*/MAP_* are NOT portable-stable literals the way Win32's constants
# are (they vary by platform/libc) - fetched via compiler.cexpr against the
# real header, matching lib/socket.py's own AF_INET/SOCK_STREAM pattern.

import compiler

PROT_NONE:      i32 = compiler.cexpr( 'PROT_NONE',      'sys/mman.h', i32 )
PROT_READ:      i32 = compiler.cexpr( 'PROT_READ',      'sys/mman.h', i32 )
PROT_WRITE:     i32 = compiler.cexpr( 'PROT_WRITE',     'sys/mman.h', i32 )
MAP_PRIVATE:    i32 = compiler.cexpr( 'MAP_PRIVATE',    'sys/mman.h', i32 )
MAP_ANONYMOUS:  i32 = compiler.cexpr( 'MAP_ANONYMOUS',  'sys/mman.h', i32 )

# void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t
# offset) - offset modelled as i64 (off_t is 64-bit on every 64-bit POSIX
# target this compiler runs on); this POC always passes 0. Failure returns
# MAP_FAILED ((void*)-1), NOT null - callers must compare against that, not
# against None.
@extern('c', 'mmap', header='sys/mman.h')
def mmap(
	addr: Ptr[None],
	length: usize,
	prot: i32,
	flags: i32,
	fd: i32,
	offset: i64,
) -> Ptr[None]:
	...

@extern('c', 'mprotect', header='sys/mman.h')
def mprotect(
	addr: Ptr[None],
	length: usize,
	prot: i32,
) -> i32:
	...

@extern('c', 'munmap', header='sys/mman.h')
def munmap(
	addr: Ptr[None],
	length: usize,
) -> i32:
	...
