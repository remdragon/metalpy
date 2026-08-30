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
@extern('pthread', 'pthread_create', header='pthread.h', spawns_thread=True)
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

# marks `thread` so its resources are reclaimed automatically on exit
# instead of requiring a join() - same by-value pthread_t as pthread_join
@extern('pthread', 'pthread_detach', header='pthread.h')
def pthread_detach(
	thread: pthread_t,
) -> i32:
	...


# ---------------------------------------------------------------------------
# TLS (thread-local storage) - lib/threading.py's own ThreadLocal[T]. pthread_key_t
# is opaque/platform-specific (an unsigned int on glibc), modelled the same
# way as pthread_t above.
# ---------------------------------------------------------------------------

pthread_key_t = compiler.c_type( 'pthread_key_t', header = 'pthread.h' )

# int pthread_key_create(pthread_key_t *key, void (*destructor)(void*)) -
# destructor is always passed NULL here (Ptr[None]) - see ThreadLocal's own
# docstring for why automatic per-thread cleanup isn't attempted
@extern('pthread', 'pthread_key_create', header='pthread.h')
def pthread_key_create(
	key: Ptr[pthread_key_t],
	destructor: Ptr[None],
) -> i32:
	...

@extern('pthread', 'pthread_key_delete', header='pthread.h')
def pthread_key_delete(
	key: pthread_key_t,
) -> i32:
	...

@extern('pthread', 'pthread_getspecific', header='pthread.h')
def pthread_getspecific(
	key: pthread_key_t,
) -> Ptr[None]:
	...

@extern('pthread', 'pthread_setspecific', header='pthread.h')
def pthread_setspecific(
	key: pthread_key_t,
	value: Ptr[None],
) -> i32:
	...


# ---------------------------------------------------------------------------
# ucontext — user-level context switching (glibc, part of libc itself, no
# separate link library - 'c' matches socket.py's own no-extra-library
# externs). ucontext_t is opaque/platform-specific (its layout differs by
# libc/arch) - modelled the same way as pthread_t above via compiler.c_type,
# only ever touched through Ptr[ucontext_t]. Field access (uc_stack.ss_sp/
# ss_size, uc_link) has no mechanism in this compiler (CType exposes no
# field lowering) - fiber_poc_test.py's own shim fills those fields in real
# hand-written C on our behalf; see its comment for why.
# ---------------------------------------------------------------------------

ucontext_t = compiler.c_type( 'ucontext_t', header = 'ucontext.h' )

@extern('c', 'getcontext', header='ucontext.h')
def getcontext(
	ucp: Ptr[ucontext_t],
) -> i32:
	...

# void makecontext(ucontext_t *ucp, void (*func)(), int argc, ...) - the
# real prototype is variadic (extra int args forwarded to func at resume).
# This POC only ever calls it with argc=0 (zero-arg top-level entry
# functions - see fiber_poc_test.py) - no variadic argument-passing is
# bound here. A real feature built on this would need that; deliberately
# out of scope for this feasibility POC (no variadic @extern precedent
# exists anywhere in this codebase).
@extern('c', 'makecontext', header='ucontext.h')
def makecontext(
	ucp: Ptr[ucontext_t],
	func: Ptr[Callable[[], None]],
	argc: i32,
) -> None:
	...

@extern('c', 'swapcontext', header='ucontext.h')
def swapcontext(
	oucp: Ptr[ucontext_t],
	ucp: Ptr[ucontext_t],
) -> i32:
	...
