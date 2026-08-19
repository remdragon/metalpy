# lib/fiber.py — Fiber: cooperative stack-switching on one real OS thread.
#
# A Fiber is a PERSISTENT, reusable execution context with its own stack -
# unlike Thread (which owns exactly one run and then is gone), a Fiber's
# entry point is an infinite loop that pulls its next task and runs it,
# rather than exiting. This is deliberate, not just an optimization: Windows'
# CreateFiber has no way to hand it caller-supplied stack memory (it always
# allocates its own internally), so there is no portable way to "free a
# fiber's stack and give it to a fresh one" the way a raw memory pool would -
# reusing the SAME Fiber object (and therefore the same underlying stack,
# whichever platform owns it) for every task it ever runs is what makes
# pooling work identically on both platforms.
#
# start(task) switches control INTO the fiber to run `task`, returning once
# that fiber parks (via the free function park(), called from code running
# INSIDE the fiber) or the task finishes. park() may be called from
# arbitrarily deep inside task's own call stack - a real OS stack has no
# "only my own top-level body" restriction the way this compiler's stackless
# generators do. unpark() resumes a PARKED fiber exactly where its own
# park() call left off.
#
# start()/unpark() may themselves be called either from a plain OS thread
# (after enable_current_thread()) OR from inside another already-running
# fiber (nested) - park() always returns to WHOEVER called start()/unpark()
# most recently, tracked per-call via __caller, not hardcoded to "the thread
# that started all this".
#
# The pending task is stored TYPE-ERASED (Ptr[None]), not as
# Closure[[],None]|None: this compiler's closure-call recognizer
# (_try_lower_closure_call) needs the callee to be a plain, already-
# concrete ClosureType local - narrowing an Optional via if/match doesn't
# change what's on record for it. Ptr[None] sidesteps that entirely (null
# is an ordinary pointer value, no union involved), the same erase-for-the-
# handoff/cast-back-to-call idiom lib/threading.py's own Thread already
# uses for its entry closure. No extra incref/decref needed for the
# erasure window here (unlike Thread's genuinely async handoff): start()
# is fully synchronous relative to its own caller, whose own reference to
# `task` stays alive for the entire call by ordinary call-stack lifetime.

import compiler
import sys

if compiler.target.os == 'windows':
	from windows.kernel32 import CreateFiber, ConvertThreadToFiber, SwitchToFiber, DeleteFiber
	FiberHandle: TypeAlias = Ptr[None]   # LPVOID from CreateFiber/ConvertThreadToFiber
else:
	from posix.pthread import ucontext_t, getcontext, makecontext, swapcontext
	from posix.mman import mmap, mprotect, munmap, PROT_NONE, PROT_READ, PROT_WRITE, MAP_PRIVATE, MAP_ANONYMOUS
	FiberHandle: TypeAlias = Ptr[ucontext_t]

	# stack_t (signal.h, also ucontext_t's own uc_stack field type) - opaque,
	# only ever touched one field at a time via compiler.c_field*(), same
	# posture as ucontext_t itself (see lib/posix/pthread.py's own comment)
	stack_t = compiler.c_type( 'stack_t', header = 'signal.h' )

DEFAULT_STACK_SIZE: usize = 262144   # 256 KiB
_PAGE_SIZE: usize = 4096             # guard page size (POSIX only) - matches every real target this compiler runs on today (x86-64 Linux/macOS)


class FiberError:
	pass


# ---------------------------------------------------------------------------
# "which fiber is running right now" - an ambient lookup (mirrors the
# reactor plan's own current_worker() design), NOT yet thread-local. Safe
# for a single OS thread driving fibers (what this module is tested against
# today); becomes a real bug the moment more than one OS thread resumes
# fibers concurrently - MUST become a ThreadLocal[Fiber|None] before the
# multi-worker Reactor is built on top of this. Tracked, not forgotten.
# ---------------------------------------------------------------------------

_current: Fiber|None = None

def current() -> Fiber|None:
	return _current


@enum( i32 )
class FiberState:
	IDLE    = 0   # never started, or previous task ran to completion - start(task) starts a NEW one
	RUNNING = 1   # currently executing (this fiber's own POV while switched in)
	PARKED  = 2   # yielded mid-task via park() - unpark() resumes exactly where it left off


class Fiber:
	__handle: FiberHandle
	__caller: FiberHandle|None       # who to switch back to on park() - set fresh each start()/unpark()
	__pending: Ptr[None]             # type-erased Closure[[], None], or null - see module docstring
	__state: FiberState
	# POSIX only in practice (the guard-paged mmap region backing __handle's
	# ucontext_t stack, freed in __del__) - declared unconditionally and
	# harmlessly set to a null/zero placeholder in the Windows __init__ too,
	# since this compiler requires every declared field definitely assigned
	# in every constructor, and there's no per-platform conditional field
	# declaration inside a class body the way @compiler.target gates methods
	__base: Ptr[None]
	__guard_and_stack: usize

	@compiler.target( os = 'windows' )
	def __init__( self, stack_size: usize = DEFAULT_STACK_SIZE ) -> None:
		compiler.incref( self )   # the running trampoline's own identity reference - see _fiber_trampoline's comment
		arg: Ptr[None] = compiler.cast( Ptr[None], self )
		self.__handle = CreateFiber( stack_size, _fiber_trampoline, arg )
		if self.__handle is None:
			sys.panic( 'Fiber.__init__: CreateFiber failed' )
		self.__pending = None
		self.__caller = None
		self.__state = FiberState.IDLE
		self.__base = None            # unused on Windows - see the field's own comment
		self.__guard_and_stack = 0

	@compiler.target( os = not 'windows' )
	def __init__( self, stack_size: usize = DEFAULT_STACK_SIZE ) -> None:
		self.__handle = sys.alloc[ucontext_t]( 1 )
		if getcontext( self.__handle ) != 0:
			sys.panic( 'Fiber.__init__: getcontext failed' )
		guard_and_stack: usize = 0
		with compiler.panic_arithmetic( 'Fiber.__init__: stack_size too large' ):
			guard_and_stack = stack_size + _PAGE_SIZE
		base: Ptr[None] = mmap( None, guard_and_stack, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0 )
		# TODO: mmap's real failure sentinel is MAP_FAILED == (void*)-1, NOT
		# null (see lib/posix/mman.py's own header comment) - this compiler
		# has no clean pointer<->integer cast yet to compare against that
		# bit pattern (compiler.cast only supports pointer-to-pointer and
		# scalar-to-scalar), so this check is currently incomplete (a real
		# mmap failure would fall through as if base were a valid address,
		# and fault later instead of panicking here with a clear message).
		# Flagging rather than silently treating this as solved - fine for
		# the fiber-primitive validation pass, worth fixing before relying
		# on this in anything real.
		if base is None:
			sys.panic( 'Fiber.__init__: mmap failed' )
		if mprotect( base, _PAGE_SIZE, PROT_NONE ) != 0:
			sys.panic( 'Fiber.__init__: mprotect (guard page) failed' )
		# stack grows DOWN on every target this compiler runs on (x86-64) -
		# the guard page goes at the LOW end, ss_sp points PAST it at the
		# usable region's own start
		usable: Ptr[u8] = compiler.wrapped_ptr_add( compiler.cast( Ptr[u8], base ), _PAGE_SIZE )
		stack_field: Ptr[stack_t] = compiler.c_field_addr( self.__handle, 'uc_stack', Ptr[stack_t] )
		compiler.c_field_set( stack_field, 'ss_sp', compiler.cast( Ptr[None], usable ))
		compiler.c_field_set( stack_field, 'ss_size', stack_size )
		compiler.c_field_set( stack_field, 'ss_flags', i32( 0 ))
		null_link: Ptr[None] = None
		compiler.c_field_set( self.__handle, 'uc_link', null_link )   # trampoline never returns - never consulted
		self.__base = base
		self.__guard_and_stack = guard_and_stack
		makecontext( self.__handle, _fiber_trampoline, 0 )
		self.__pending = None
		self.__caller = None
		self.__state = FiberState.IDLE

	@compiler.target( os = 'windows' )
	def __del__( self ) -> None:
		DeleteFiber( self.__handle )

	@compiler.target( os = not 'windows' )
	def __del__( self ) -> None:
		munmap( self.__base, self.__guard_and_stack )
		sys.free( compiler.cast( Ptr[None], self.__handle ))

	def start( self, task: Closure[[], None] ) -> None:
		''' switch control into this IDLE fiber to run a NEW `task` from
		scratch. Returns once the fiber parks (mid-task, via the free
		function park() - see unpark() to resume from there) or the task
		runs to completion (fiber goes back to IDLE). Panics if the fiber
		is currently RUNNING or PARKED - a parked fiber is mid-task
		already; feeding it a second task here would be silently ignored
		(control resumes inside park()'s own switch, never back at the top
		of _run_loop where __pending is even looked at) - unpark() is the
		only correct way to continue it. '''
		if self.__state != FiberState.IDLE:
			sys.panic( 'Fiber.start: fiber is not idle (RUNNING, or PARKED - use unpark() instead)' )
		self.__pending = compiler.cast( Ptr[None], task )
		self.__state = FiberState.RUNNING
		self.__switch_in()

	def unpark( self ) -> None:
		''' switch control back into this PARKED fiber, resuming exactly
		where its own call to park() left off - no new task, __pending is
		never touched. Returns the same way start() does: once the fiber
		parks again or the task finishes. '''
		if self.__state != FiberState.PARKED:
			sys.panic( 'Fiber.unpark: fiber is not parked' )
		self.__state = FiberState.RUNNING
		self.__switch_in()

	def _switch_out( self, new_state: FiberState ) -> None:
		''' shared tail for both ways a fiber gives up control: the free
		function park() (mid-task, ends PARKED) and _run_loop's own call
		once a task runs to completion (ends IDLE). A real method (not
		logic inlined into the free-function trampolines/park()) so it can
		reach __state/__caller directly - a free function in this module
		can't name-mangle its way to another class's private fields. '''
		self.__state = new_state
		self.__switch_out()

	def _run_loop( self ) -> None:
		''' the trampoline body proper - called once by _fiber_trampoline
		after it's identified `self`; loops forever pulling whatever
		start() most recently stashed in __pending. Only ever reached via
		start() (never unpark(), which switches back into the middle of a
		task's OWN call stack, not here) - by the time control returns to
		the top of this loop, the task just run has gone all the way back
		to completion, so IDLE is always the correct state to leave it in
		before switching back out. '''
		while True:
			if self.__pending is None:
				sys.panic( 'Fiber._run_loop: resumed with no pending task' )
			task: Closure[[], None] = compiler.cast( Closure[[], None], self.__pending )
			self.__pending = None
			task()
			self._switch_out( FiberState.IDLE )

	@compiler.target( os = 'windows' )
	def __switch_in( self ) -> None:
		global _current
		prev: Fiber|None = _current
		if prev is not None:
			self.__caller = prev.__handle
		else:
			self.__caller = _thread_fiber_handle
		_current = self
		SwitchToFiber( self.__handle )
		_current = prev

	@compiler.target( os = not 'windows' )
	def __switch_in( self ) -> None:
		global _current
		prev: Fiber|None = _current
		_current = self
		caller_ctx: Ptr[ucontext_t] = sys.alloc[ucontext_t]( 1 )
		self.__caller = caller_ctx
		if swapcontext( caller_ctx, self.__handle ) != 0:
			sys.panic( 'Fiber.start: swapcontext (into fiber) failed' )
		sys.free( compiler.cast( Ptr[None], caller_ctx ))
		_current = prev

	@compiler.target( os = 'windows' )
	def __switch_out( self ) -> None:
		caller: Ptr[None]|None = self.__caller
		if caller is None:
			sys.panic( 'Fiber._switch_out: no caller recorded (parked before ever being started?)' )
		SwitchToFiber( caller )

	@compiler.target( os = not 'windows' )
	def __switch_out( self ) -> None:
		caller: Ptr[ucontext_t]|None = self.__caller
		if caller is None:
			sys.panic( 'Fiber._switch_out: no caller recorded (parked before ever being started?)' )
		if swapcontext( self.__handle, caller ) != 0:
			sys.panic( 'Fiber._switch_out: swapcontext (back to caller) failed' )


def park() -> None:
	''' called from code running INSIDE a fiber (at any call depth) to yield
	control back to whoever called start()/unpark() on it. That call
	returns once this happens; a later unpark() call on the SAME fiber
	picks up exactly where park() left off. '''
	fiber = current()
	if fiber is None:
		sys.panic( 'fiber.park() called with no fiber currently running (enable_current_thread() not called, or called from the raw OS thread itself)' )
	fiber._switch_out( FiberState.PARKED )


@compiler.target( os = 'windows' )
def _fiber_trampoline( arg: Ptr[None] ) -> None:
	# cast ONCE - self's own identity reference (see __init__'s
	# compiler.incref) is exactly what this cast's implicit ownership
	# corresponds to. _run_loop never returns (infinite loop, no break), so
	# that ownership's own scope-exit decref is never reached here - fine,
	# it's released instead by whatever path eventually calls DeleteFiber
	# and lets this Fiber's last real reference go (see __del__).
	fiber: Fiber = compiler.cast( Fiber, arg )
	fiber._run_loop()

@compiler.target( os = not 'windows' )
def _fiber_trampoline() -> None:
	# no argument on POSIX (makecontext's own zero-arg restriction - see
	# lib/posix/pthread.py's comment) - `current()` already correctly
	# points at this fiber by the time we get here, since start()'s own
	# __switch_in sets it BEFORE swapcontext ever jumps here
	started: Fiber|None = current()
	if started is None:
		sys.panic( 'Fiber trampoline: started with no current fiber set' )
	fiber: Fiber = started
	fiber._run_loop()


# ---------------------------------------------------------------------------
# enable_current_thread() - Windows only needs real work here (a thread must
# be converted before SwitchToFiber will accept it as a target); POSIX's
# swapcontext needs no such conversion (any thread can be the "from" side of
# a context switch), so it's a no-op there.
# ---------------------------------------------------------------------------

_thread_fiber_handle: Ptr[None] = None

@compiler.target( os = 'windows' )
def enable_current_thread() -> None:
	global _thread_fiber_handle
	if _thread_fiber_handle is not None:
		return
	_thread_fiber_handle = ConvertThreadToFiber( None )
	if _thread_fiber_handle is None:
		sys.panic( 'fiber.enable_current_thread: ConvertThreadToFiber failed' )

@compiler.target( os = not 'windows' )
def enable_current_thread() -> None:
	pass
