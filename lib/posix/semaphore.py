# lib/posix/semaphore.py — POSIX unnamed semaphores (librt/libc) extern declarations
#
# header='semaphore.h' suppresses these prototypes in the generated C - the
# real signatures from <semaphore.h> are used instead, and #include
# <semaphore.h> is added automatically. If our parameter types disagree
# with the header, the C compiler catches it - same convention as
# lib/posix/pthread.py's own externs.

import compiler

# sem_t is opaque and platform-specific (glibc's own internal layout isn't
# meant to be relied on) - model it as the real C type, sized via
# compiler.sizeof(), the same way lib/threading.py's own FastLock already
# does for pthread_mutex_t. lib/threading.py's Semaphore reuses this.
sem_t = compiler.c_type( 'sem_t', header = 'semaphore.h' )

@extern( 'c', 'sem_init', header = 'semaphore.h' )
def sem_init(
	sem:    Ptr[None],
	pshared: i32,
	value:  u32,
) -> i32:
	...

@extern( 'c', 'sem_destroy', header = 'semaphore.h' )
def sem_destroy(
	sem: Ptr[None],
) -> i32:
	...

@extern( 'c', 'sem_post', header = 'semaphore.h' )
def sem_post(
	sem: Ptr[None],
) -> i32:
	...

@extern( 'c', 'sem_wait', header = 'semaphore.h' )
def sem_wait(
	sem: Ptr[None],
) -> i32:
	...
