# lib/posix/epoll.py — Linux epoll extern declarations.
#
# struct epoll_event is packed on x86/x86-64 (glibc's own __EPOLL_PACKED),
# unlike a "normal" C struct's natural alignment - a hand-rolled @cstruct
# here would need to get that exactly right to match the kernel ABI. Using
# the opaque compiler.c_type(header=...) pattern (same as lib/posix/
# dirent.py's DIR/lib/posix/stat.py's stat_t) instead sidesteps that
# entirely: the real system header supplies the real layout, read/written
# one field at a time via compiler.c_field*(), and compiler.sizeof()
# resolves the real (packed) size for allocation - no risk of this module
# guessing the packing wrong.

import compiler

epoll_event = compiler.c_type( 'struct epoll_event', header = 'sys/epoll.h' )

EPOLLIN:  u32 = 0x001
EPOLLOUT: u32 = 0x004
EPOLLERR: u32 = 0x008
EPOLLHUP: u32 = 0x010

EPOLL_CTL_ADD: i32 = 1
EPOLL_CTL_DEL: i32 = 2
EPOLL_CTL_MOD: i32 = 3

@extern( 'c', 'epoll_create1', header = 'sys/epoll.h' )
def epoll_create1(
	flags: i32,
) -> i32:
	...

@extern( 'c', 'epoll_ctl', header = 'sys/epoll.h' )
def epoll_ctl(
	epfd: i32,
	op: i32,
	fd: i32,
	event: Ptr[epoll_event],
) -> i32:
	...

@extern( 'c', 'epoll_wait', header = 'sys/epoll.h' )
def epoll_wait(
	epfd: i32,
	events: Ptr[epoll_event],
	maxevents: i32,
	timeout: i32,
) -> i32:
	...
