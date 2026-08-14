# lib/posix/pthread.py — POSIX threads (libpthread) extern declarations

# header='pthread.h' suppresses these prototypes in the generated C —
# the real signatures from pthread.h are used instead.  It also triggers
# the #include <pthread.h> automatically (no separate require_header needed).
# If our parameter types disagree with the header, the C compiler will catch it.

import compiler

# pthread_t is opaque and platform-specific (an unsigned long on glibc, an
# opaque struct pointer on macOS) - model it as the real C type so pthread_join
# (which takes it BY VALUE) and pthread_create (which fills a pthread_t*) both
# emit the exact type <pthread.h> expects. lib/threading.py reuses this same
# type for Thread.__handle.
pthread_t = compiler.c_type( 'pthread_t', header = 'pthread.h' )

@extern('pthread', 'pthread_mutex_init', header='pthread.h')
def pthread_mutex_init(
	mutex: Ptr[None],
	attr: Ptr[None],
) -> i32:
	...

@extern('pthread', 'pthread_mutex_destroy', header='pthread.h')
def pthread_mutex_destroy(
	mutex: Ptr[None],
) -> i32:
	...

@extern('pthread', 'pthread_mutex_lock', header='pthread.h')
def pthread_mutex_lock(
	mutex: Ptr[None],
) -> i32:
	...

@extern('pthread', 'pthread_mutex_trylock', header='pthread.h')
def pthread_mutex_trylock(
	mutex: Ptr[None],
) -> i32:
	...

@extern('pthread', 'pthread_mutex_unlock', header='pthread.h')
def pthread_mutex_unlock(
	mutex: Ptr[None],
) -> i32:
	...

# pthread_t is opaque/platform-specific (typically an unsigned long or a
# small struct) - Ptr[None] stands in for pthread_t itself (an opaque
# handle, same as HANDLE on Windows); `thread` here is pthread_t*, an
# OUT-param pthread_create WRITES the new thread's own handle into, so
# it's Ptr[Ptr[None]] - one level more indirection than mutex/attr above.
# header='pthread.h' means the REAL declaration from the header is what
# the C compiler actually sees, this one is only for this compiler's own
# type-checking
@extern('pthread', 'pthread_create', header='pthread.h')
def pthread_create(
	thread: Ptr[pthread_t],
	attr: Ptr[None],
	start_routine: Ptr[Callable[[Ptr[None]], Ptr[None]]],
	arg: Ptr[None],
) -> i32:
	...

# thread is pthread_t itself here (passed by value, not by pointer -
# unlike pthread_create's own out-param above)
@extern('pthread', 'pthread_join', header='pthread.h')
def pthread_join(
	thread: pthread_t,
	retval: Ptr[None],
) -> i32:
	...
