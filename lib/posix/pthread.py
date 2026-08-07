# lib/posix/pthread.py — POSIX threads (libpthread) extern declarations

# All mutex pointer parameters use Ptr[None] (void*) for ABI compatibility —
# pthread_mutex_t is an opaque type whose size/layout varies by platform
# (typically 40 bytes on glibc x86_64, 48 on musl).  Declaring with void*
# avoids needing to define the struct here while keeping the ABI identical.

@extern('pthread', 'pthread_mutex_init')
def pthread_mutex_init(
	mutex: Ptr[None],
	attr: Ptr[None],
) -> i32:
	...

@extern('pthread', 'pthread_mutex_destroy')
def pthread_mutex_destroy(
	mutex: Ptr[None],
) -> i32:
	...

@extern('pthread', 'pthread_mutex_lock')
def pthread_mutex_lock(
	mutex: Ptr[None],
) -> i32:
	...

@extern('pthread', 'pthread_mutex_trylock')
def pthread_mutex_trylock(
	mutex: Ptr[None],
) -> i32:
	...

@extern('pthread', 'pthread_mutex_unlock')
def pthread_mutex_unlock(
	mutex: Ptr[None],
) -> i32:
	...
