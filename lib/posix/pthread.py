# lib/posix/pthread.py — POSIX threads (libpthread) extern declarations

# header='pthread.h' suppresses these prototypes in the generated C —
# the real signatures from pthread.h are used instead.  It also triggers
# the #include <pthread.h> automatically (no separate require_header needed).
# If our parameter types disagree with the header, the C compiler will catch it.

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
