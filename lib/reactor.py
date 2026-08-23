# lib/reactor.py — Worker/Reactor scaffolding on top of fiber.Fiber.
#
# Was named mp_reactor.py for a while - naming this module "reactor" once
# reliably crashed (real use-after-free/heap corruption) for reasons never
# root-caused at the time. Re-investigated later and confirmed RESOLVED:
# the crash traced to two real, independent UAFs that existed the same day
# this module was first written (Fiber.start()'s missing incref, and a
# list[Closure[...]] aliasing-check bug in lowering.py - see
# list_closure_uaf_fixed in memory) and just happened to manifest under
# the specific memory layout the name "reactor" produced that day - nothing
# about the name itself. Both are long since fixed on master; renamed back
# once that was verified (full suite + repeated real compile/run passes on
# all 3 compilers, debug/--release/--asan, no recurrence).
#
# A Worker owns a pool of fibers and two work queues: __pending_tasks (fresh
# work, needs an idle fiber and a start()) and __ready_to_unpark (fibers that
# previously called fiber.park() mid-task, just need unpark() to continue -
# no new task attached). Its own scheduling loop pops from both queues and
# drives fiber.start()/fiber.unpark() directly; once control returns (the
# fiber either finished or parked again), the fiber goes back into the idle
# pool or the ready-to-unpark queue based on its own reported state() - a
# task calling park() needs no special "yield to scheduler" call, ordinary
# park() is already enough for the Worker to notice and requeue it.
#
# list[T] already has its own internal lock (see lib/builtins/__list.py),
# so these queues are safely shared between whichever thread calls spawn()
# and the Worker's own driving thread with no extra locking here.
#
# run_until_idle() advances exactly ONE tick's worth of work - whatever was
# already queued when it was called, plus a poller check (default: non-
# blocking peek, timeout=0) for any Signal that's become ready - and
# returns whether it did anything; a fiber that parks again mid-tick
# (cooperatively, or via wait_for_signal()) waits for the NEXT call, it is
# not redriven within the same one (see its own docstring for why).
# drain_fully() (and Reactor.run(), built on it) loops run_until_idle()
# with timeout=0 while there's immediate progress, then - once a tick is a
# genuine no-op - either returns (nothing outstanding at all) or calls
# run_until_idle() ONE more time with an INFINITE poller timeout if a
# Signal wait is still outstanding, relying on a per-Worker wake pair
# (Worker.__init__'s own self-pipe-equivalent, poked by schedule()) to
# guarantee that blocking call always has a way back out - either the
# awaited signal fires, or new work arrives from another thread and pokes
# it awake. See drain_fully()'s own docstring for the real, deliberate
# consequence of this: a Worker with a standing, never-satisfied signal
# wait can now legitimately never return (correct "keep serving"
# behavior, not a bug) - UNLESS Reactor.shutdown()/Worker.request_
# shutdown() has been called, which reuses the SAME wake pair to force-
# resume every waiting fiber with a WaitError.Shutdown (wait_for_signal()
# returns Result[None, WaitError], not a bare None) instead of
# leaving it hoping for a signal that may never come - see request_
# shutdown()/__drain_waiting_for_shutdown()'s own docstrings. The same
# WaitError union also carries TimedOut, for an active `with timeout(...):`
# deadline elapsing first - see timeout()'s own docstring.
#
# current_worker() - an ambient lookup so code running INSIDE a fiber (e.g.
# a future NonBlockingIO read()/wait_for()) can find which Worker owns it,
# without needing a Worker threaded through every call - mirrors fiber.py's
# own current(), built the same way (a ThreadLocal[Worker] slot, set by
# whichever thread is driving a Worker's own run_until_idle() loop). Set
# UNCONDITIONALLY on every run_until_idle() call (not idempotently, unlike
# fiber.enable_current_thread()'s one-time OS-level side effect) - cheap,
# and correctly reflects whichever Worker most recently drove this thread
# in the (test-only, not the real pinned-per-worker Reactor.run() case)
# scenario of one thread sequentially driving more than one Worker. Safe
# to store without an extra incref (unlike fiber.py's own _thread_fiber_
# handle box, which needed one - see that fix's own comment): a Worker
# handed to run_until_idle() is ALWAYS kept alive by some other real owner
# for the whole call already (a Reactor's own __workers list, or a test's
# bare local) - current_worker() is a bookmark, not a second owner, same
# as fiber.py's own _current.
#
# Reactor with more than ONE worker is now SAFE at the fiber-switching
# level: fiber.py's own _current/_thread_fiber_handle are real
# ThreadLocal[T] slots now, not plain globals racing across concurrent
# OS threads. Reactor.run() genuinely does hand each Worker its own
# freshly-spawned OS thread (see Worker.run_until_idle's own fiber.
# enable_current_thread() call) - multiple CONCURRENT workers (more than
# one such thread alive and switching fibers at once) no longer race on
# shared fiber-switch bookkeeping. This scaffolding's OWN queues
# (__pending_tasks/__ready_to_unpark/__idle_pool) were already safe
# either way (list[T]'s own internal lock).
#
# A real multi-worker termination bug WAS found once something actually
# exercised the round-robin spawn() pattern a real server needs (an
# accept-loop fiber, itself already running on one worker, spawning a
# fresh per-connection handler task while the Reactor is already mid-run):
# Worker.drain_fully() used to return the instant ITS OWN queues went
# idle, with no way to know a sibling worker's in-flight fiber might still
# call Reactor.spawn() and round-robin fresh work onto it - a worker that
# happened to start (or drain to) empty could exit and its OS thread end
# before ever receiving work spawned onto it later, silently dropping that
# task forever (Reactor.spawn()'s w.schedule() call still succeeds - it
# just enqueues onto a Worker nobody is left driving). Fixed via
# _ReactorState.live_tasks, a Reactor-wide "still live somewhere" counter
# every Worker.schedule()/task-completion keeps balanced - see
# _ReactorState and Worker.drain_fully's own docstrings for the full
# mechanism, and test_reactor_multiworker_spawn_from_inside_a_running_
# fiber below for the regression test that first caught this.

import compiler
import sys
import threading
import fiber
import poller
import socket
import atomic
import time
from datetime import timedelta

_current_worker: threading.ThreadLocal[Worker] = threading.ThreadLocal[Worker]()

def current_worker() -> Worker|None:
	''' the Worker driving fibers on THIS OS thread right now, or None if
	this thread isn't currently (or has never been) inside a Worker's own
	run_until_idle() - e.g. the thread that just calls Reactor.spawn()
	from outside any worker. See this module's own header comment for the
	full ownership reasoning. '''
	return _current_worker.get()


# ---------------------------------------------------------------------------
# timeout() - `with timeout(delta):` bounds every wait_for_signal() call
# inside the block, however deeply nested (e.g. a TcpConnection.read()
# calling wait_for_signal() internally), without threading a deadline
# parameter through read()/readline()/write_all()/accept()/etc.
# ---------------------------------------------------------------------------

class _DeadlineBox:
	''' ThreadLocal[T] requires T to be an RC type - see threading.py's own
	module comment - so the no-fiber fallback slot below needs f64 boxed. '''
	value: f64
	def __init__( self, value: f64 ) -> None:
		self.value = value

_thread_deadline: threading.ThreadLocal[_DeadlineBox] = threading.ThreadLocal[_DeadlineBox]()

def _current_deadline() -> f64:
	''' the currently-active `with timeout(...)` deadline (time.monotonic()
	seconds), or fiber.NO_DEADLINE if none is active. Checked via
	fiber.current() first - a real fiber's own deadline survives its
	park()/unpark() for free, since it's the fiber's own field, not shared
	state a DIFFERENT cooperatively-scheduled fiber running on the same OS
	thread in between would otherwise clobber - falling back to a
	ThreadLocal only when there's no current fiber at all (genuinely
	synchronous, fiber-free code; nothing else could possibly interleave on
	this thread in that case, so a plain ThreadLocal is exactly as safe as
	a Fiber-owned field would be there). '''
	cur: fiber.Fiber|None = fiber.current()
	if cur is not None:
		return cur.get_deadline()
	box: _DeadlineBox|None = _thread_deadline.get()
	if box is None:
		return fiber.NO_DEADLINE
	return box.value

def _set_current_deadline( deadline: f64 ) -> None:
	cur: fiber.Fiber|None = fiber.current()
	if cur is not None:
		cur.set_deadline( deadline )
		return
	if deadline == fiber.NO_DEADLINE:
		_thread_deadline.clear()
	else:
		_thread_deadline.set( _DeadlineBox( deadline ))

def _ms_until( deadline: f64 ) -> i32:
	''' deadline (a time.monotonic() reading) as a millisecond countdown
	from NOW, clamped to [0, 2_000_000_000] - 0 if already past (poller.wait
	should check immediately, not block), the upper clamp so a deadline
	millennia away can't overflow i32. '''
	now: f64 = time.monotonic()
	with compiler.wrap_arithmetic:
		remaining_s: f64 = deadline - now
	if remaining_s <= 0.0:
		return i32( 0 )
	with compiler.wrap_arithmetic:
		ms_f: f64 = remaining_s * 1000.0
	if ms_f > 2000000000.0:
		return i32( 2000000000 )
	with compiler.wrap_arithmetic:
		return i32( ms_f )

class timeout:
	''' `with timeout(delta):` bounds every wait_for_signal() call inside
	the block to at most `delta` from now - a TcpConnection.read() calling
	wait_for_signal() internally is covered with no parameter threading
	needed. Nesting narrows, never widens: an inner timeout can only
	tighten an outer one's deadline (never push it later), matching
	Python's own contextvar-style "innermost wins, but can't escape an
	outer bound" convention - __enter__ computes min(candidate, previous)
	so this holds regardless of nesting order. Always restores the
	previous ambient deadline on __exit__ (with's own defer-based
	scope-exit guarantee covers every exit path, including an early
	.or_return() from inside the block), so a fiber pulled from the idle
	pool for its NEXT, unrelated task never inherits a stale deadline. '''
	__previous: f64
	__delta:    timedelta
	def __init__( self, delta: timedelta ) -> None:
		self.__delta = delta
		self.__previous = fiber.NO_DEADLINE
	def __enter__( self ) -> None:
		self.__previous = _current_deadline()
		with compiler.wrap_arithmetic:
			candidate: f64 = time.monotonic() + self.__delta.total_seconds()
		if self.__previous != fiber.NO_DEADLINE and self.__previous < candidate:
			_set_current_deadline( self.__previous )
		else:
			_set_current_deadline( candidate )
	def __exit__( self ) -> None:
		_set_current_deadline( self.__previous )


class FdReadiness:
	''' payload for Signal.FdReady - wait for a specific fd to become
	ready for read and/or write, per lib/poller.py's own readiness model
	(epoll/WSAPoll). '''
	fd:         poller.SOCKET
	want_read:  bool
	want_write: bool
	def __init__( self, fd: poller.SOCKET, want_read: bool, want_write: bool ) -> None:
		self.fd = fd
		self.want_read = want_read
		self.want_write = want_write


class CompletionHandle:
	''' payload for Signal.Completion - a one-shot result box for an
	operation that runs to completion entirely OFF this fiber's own thread
	(e.g. lib/asyncfile.py's thread pool doing a real blocking file read/
	write), matching IOCP/io_uring's own model: by the time this fires, the
	actual result already exists, nothing left to "try again" the way an
	FdReady wakeup might need. The producer (whichever thread actually runs
	the operation) calls _complete() exactly once; the waiter reads take()
	after wait_for_signal() returns Ok - never before, so take() never
	itself has to block or check is_done(). __done is the one field every
	other field's own visibility depends on: __value/__err/__ok are plain
	(non-atomic) writes made BEFORE the atomic store, safe to read only
	AFTER observing __done via atomic load (Worker.__drain_completed_waits'
	own is_done() check) - standard release-store/acquire-load handoff. '''
	__done:  atomic.Atomic[bool]
	__ok:    bool
	__value: usize
	__err:   OSError

	def __init__( self ) -> None:
		self.__done = atomic.Atomic[bool]( False )
		self.__ok = False
		self.__value = 0
		self.__err = OSError.Other

	def complete( self, result: Result[usize, OSError] ) -> None:
		match result:
			case Result.Ok( v ):
				self.__value = v
				self.__ok = True
			case Result.Err( e ):
				self.__err = e
				self.__ok = False
		self.__done.store( True )

	def is_done( self ) -> bool:
		return self.__done.load()

	def take( self ) -> Result[usize, OSError]:
		if self.__ok:
			return Result.Ok( self.__value )
		return Result.Err( self.__err )


@union
class Signal:
	''' what a fiber is waiting for. A tagged union, not a bare fd+interest
	struct, because "something worth waking a fiber up for" has more than
	one real shape:
	  - FdReady - a poller notices a registered fd's readiness (lib/
	    poller.py, epoll/WSAPoll). This is a READINESS signal: once it
	    fires, the caller still has to actually perform the read/write
	    itself, and might get WouldBlock again (a spurious wakeup) - see
	    lib/tcp.py's own read()/write() retry-loop shape.
	  - Completion - IOCP/io_uring's own model, and what lib/asyncfile.py's
	    thread pool uses today (a real, non-IOCP producer, but the same
	    shape): the wait target IS the completing operation itself, not a
	    readiness check - by the time this fires, the actual result (bytes
	    transferred, or an error) already exists in the CompletionHandle,
	    nothing left to "try again". A fundamentally different shape from
	    FdReady, which is exactly why this needed to become a union rather
	    than growing fields on one struct.
	  - a bare "wake me directly" kind (future, not built yet) - what
	    Queue/Event will need: no fd, no completion object, just "some
	    other fiber/thread called wake() on the specific token I'm holding".
	Worker's own internals (_wait_on_signal, __check_signals, __drain_
	waiting_for_shutdown, __drain_expired_waits, __fd_still_waited_on) are
	the only places that need to know which kinds exist - everything above
	them (wait_for_signal, lib/tcp.py, lib/asyncfile.py) just holds a
	Signal opaquely and waits for it. '''
	FdReady:    FdReadiness
	Completion: CompletionHandle

def fd_signal( fd: poller.SOCKET, want_read: bool, want_write: bool ) -> Signal:
	''' convenience constructor - Signal.FdReady(FdReadiness(...)) spelled
	out at every call site would be pure noise for the one kind that
	exists today. '''
	return Signal.FdReady( FdReadiness( fd, want_read, want_write ))


class _PendingWait:
	# fields/params named `waiting_fiber`, not `fiber` - this codebase has
	# a known, still-open compiler bug where a local/field sharing a name
	# with an imported MODULE (fiber.py itself, imported above) spuriously
	# schedules/links that module's own top-level code - see memory:
	# compiler_name_shadow_scheduling_bug.md.
	signal:        Signal
	waiting_fiber: fiber.Fiber
	# set by Worker.__drain_expired_waits (never by anything else) BEFORE
	# waiting_fiber is moved to __ready_to_unpark - _wait_on_signal reads it
	# straight off this SAME object (the one it appended to __waiting, held
	# in its own local the whole time) once fiber.park() returns, to tell a
	# timeout apart from a real signal/shutdown wakeup.
	timed_out:     bool
	def __init__( self, signal: Signal, waiting_fiber: fiber.Fiber ) -> None:
		self.signal = signal
		self.waiting_fiber = waiting_fiber
		self.timed_out = False


@union
class WaitError:
	''' why wait_for_signal() gave up instead of returning Ok:
	  - Shutdown - the Worker driving this fiber was asked to stop
	    (Reactor.shutdown()/Worker.request_shutdown()). Never produced by
	    the no-reactor blocking path - there's no Worker to ask it to stop.
	  - TimedOut - an enclosing `with timeout(...):` deadline elapsed
	    before the signal fired. Produced by BOTH paths.
	Either way, handler code written with defer/errdefer for cleanup
	behaves identically whether a connection closed normally, errored, hit
	its deadline, or the reactor is shutting down - none of those are a
	special case a caller has to know about up front. '''
	Shutdown:  None
	TimedOut:  None


def wait_for_signal( signal: Signal ) -> Result[None, WaitError]:
	''' the ONE reactor-optional wait primitive every higher-level helper
	(TcpConnection's read()/write(), TcpListener's accept()) calls instead
	of talking to a Worker or a Poller directly - branches on
	current_worker() to either park the calling fiber and wait for
	`signal` via whichever Worker is driving this thread, or perform a
	real, standalone blocking wait if no Worker is driving this thread at
	all. This is what makes the exact same connection-handling code work
	unmodified as ordinary synchronous blocking code with zero Reactor
	setup, or as a cooperative fiber inside a full reactor - including
	`with timeout(...):`, which bounds either path identically (see
	timeout()'s own docstring). '''
	w: Worker|None = current_worker()
	if w is not None:
		return w._wait_on_signal( signal )
	return _blocking_wait_no_reactor( signal )

def _blocking_wait_no_reactor( signal: Signal ) -> Result[None, WaitError]:
	''' no Worker driving this thread - there's nothing to park a fiber
	INTO, so this really does block the calling OS thread, same as an
	ordinary blocking recv() would. A throwaway, single-use Poller
	(register one fd, wait with an infinite timeout, let __del__ clean
	up) rather than any shared/cached instance - this path is expected to
	be rare (real reactor-driven code never takes it) and simplicity
	beats reuse here. Only FdReady is handleable this way (a real OS-
	level blocking wait needs something pollable) - a Completion signal
	should never actually reach here: lib/asyncfile.py's own reader/
	writer types check current_worker() themselves BEFORE ever
	constructing one, calling the blocking syscall directly instead when
	there's no Worker to hand the job's completion back to - see this
	function's own Signal.Completion arm below. Respects
	_current_deadline() the same way the reactor-driven path does (via a
	ThreadLocal fallback, since there's no Worker/fiber bookkeeping to
	lean on here - see _current_deadline's own docstring). '''
	deadline: f64 = _current_deadline()
	timeout_ms: i32 = -1
	if deadline != fiber.NO_DEADLINE:
		timeout_ms = _ms_until( deadline )
	match signal:
		case Signal.FdReady( fdr ):
			p: poller.Poller = poller.Poller()
			p.register( fdr.fd, fdr.want_read, fdr.want_write ).unwrap( 'wait_for_signal: poller register failed (no reactor driving this thread)' )
			events: list[poller.ReadyEvent] = p.wait( timeout_ms ).unwrap( 'wait_for_signal: poller wait failed (no reactor driving this thread)' )
			if events.__len__() == 0 and deadline != fiber.NO_DEADLINE:
				return Result.Err( WaitError.TimedOut( None ))
			return Result.Ok( None )
		case Signal.Completion( _ ):
			sys.panic( '_blocking_wait_no_reactor: Completion signals require a Worker to hand the result back to - callers must check current_worker() before constructing one' )


# ---------------------------------------------------------------------------
# sleep(delta: timedelta) -> Result[None, WaitError]
#
# Deliberately NOT lib/time.py (Python's own time.sleep(), just with a
# timedelta argument instead of a float) - a top-level `import reactor`
# there triggers a real, confirmed discovery bug: reactor.py's own
# pre-existing `from datetime import timedelta` (needed for `timeout`
# above, nothing to do with sleep()) started failing with "module datetime
# does not export 'timedelta'" and cascaded into unrelated failures
# elsewhere (lib/datetime.py's own `from zoneinfo import ZoneInfo`, no
# relation to time.py or reactor.py at all) - some ordering issue in how a
# deeper module cycle gets discovered once time.py also points at
# reactor.py (reactor.py already imports time.py; datetime.py already does
# a function-local `import time` of its own). Confirmed the failure is
# real and caused by that one new edge specifically: reverted, reran the
# exact same previously-passing test, it passed again. Not chased down
# further - worth its own investigation. Living here instead avoids the
# new edge entirely (reactor.py already imports time and datetime.timedelta,
# so `sleep` needs no import this module doesn't already have).
# ---------------------------------------------------------------------------

def sleep( delta: timedelta ) -> Result[None, WaitError]:
	''' Python's time.sleep(seconds: float), but the argument is a
	timedelta (this compiler's own idiom - see lib/datetime.py) and the
	implementation is reactor-aware: with a Worker driving the calling
	fiber (current_worker() is not None), yields to the reactor for
	`delta` instead of blocking its OS thread, so other work already
	queued on that Worker keeps making progress in the meantime - the same
	reactor-optional shape every other wait_for_signal()-based primitive
	in this module already follows. With no Worker driving this thread,
	sleeps the OS thread for real instead (_blocking_sleep below).

	The reactor-driven path reuses timeout()'s own deadline machinery
	rather than needing a new Signal kind: it waits on a Signal.Completion
	whose CompletionHandle is deliberately never completed by anyone,
	wrapped in `with timeout(delta):` - the ONLY way that wait can ever
	resolve is the timeout elapsing (a normal sleep) or the reactor
	shutting down (WaitError.Shutdown, propagated to the caller instead of
	silently swallowed - callers with real cleanup to do on shutdown
	should treat it the same as a signal-driven wait being interrupted,
	not a successful sleep). '''
	w: Worker|None = current_worker()
	if w is None:
		_blocking_sleep( delta )
		return Result.Ok( None )
	with timeout( delta ):
		handle: CompletionHandle = CompletionHandle()
		match wait_for_signal( Signal.Completion( handle )):
			case Result.Ok( _ ):
				sys.panic( 'reactor.sleep: unreachable - a Completion signal fired without ever being completed' )
			case Result.Err( werr ):
				match werr:
					case WaitError.TimedOut( _ ):
						return Result.Ok( None )
					case WaitError.Shutdown( _ ):
						return Result.Err( werr )


@compiler.target( os = 'windows' )
def _blocking_sleep( delta: timedelta ) -> None:
	from windows.kernel32 import Sleep
	Sleep( _ms_from_delta( delta ))

@compiler.target( os = not 'windows' )
def _blocking_sleep( delta: timedelta ) -> None:
	from posix.time import nanosleep, timespec
	secs_f: f64 = delta.total_seconds()
	if secs_f <= 0.0:
		return
	with compiler.wrap_arithmetic:
		sec: i64 = i64( secs_f )
		frac_s: f64 = secs_f - f64( sec )
		nsec: i64 = i64( frac_s * 1.0e9 )
	req = timespec( tv_sec = sec, tv_nsec = nsec )
	rem = timespec()
	nanosleep( compiler.addrof( req ), compiler.addrof( rem ))   # best-effort - EINTR is not retried, see lib/posix/time.py's own comment

def _ms_from_delta( delta: timedelta ) -> u32:
	''' delta.total_seconds() as whole milliseconds, clamped to
	[0, 4_000_000_000] - 0 for a non-positive duration (Sleep(0) just
	yields the rest of this thread's timeslice, close enough to "no wait"
	for a duration that shouldn't have blocked at all), the upper clamp so
	a multi-year duration can't overflow u32. '''
	with compiler.wrap_arithmetic:
		ms_f: f64 = delta.total_seconds() * 1000.0
	if ms_f <= 0.0:
		return u32( 0 )
	if ms_f > 4000000000.0:
		return u32( 4000000000 )
	with compiler.wrap_arithmetic:
		return u32( ms_f )


class _ReactorState:
	''' shared once per Reactor across every one of its Workers (see
	Reactor.__init__) - the ONLY extra state Worker.drain_fully() needs to
	tell "nothing queued on ME" apart from "nothing live ANYWHERE in this
	Reactor". That distinction matters because Reactor.spawn() round-robins:
	a worker that starts out with an empty queue (or drains to empty first)
	must not exit just because IT personally has nothing queued - another
	worker's still-running fiber may yet call Reactor.spawn() and land fresh
	work on it (see this module's own header note - this closes that gap,
	first hit by a real multi-worker HTTP server POC spawning per-connection
	handlers from an already-running accept-loop fiber).

	live_tasks counts every task from its Worker.schedule() call until the
	underlying fiber actually finishes for good (however many park()/
	unpark() cycles that takes in between - a single schedule() call only
	ever produces one +1/-1 pair, the -1 applied by __requeue_by_state's own
	IDLE case once the fiber genuinely runs to completion). A worker treats
	the whole Reactor as quiescent, not just itself, once this hits zero.

	Deliberately holds nothing BUT this counter - no back-reference to the
	Workers themselves. Worker already owns a `_ReactorState` strongly (see
	_attach_reactor_state); a `list[Worker]` here too (e.g. to broadcast-wake
	every worker the instant this hits zero) would make Worker and
	_ReactorState a reference cycle under this codebase's plain refcounting,
	leaking every Reactor for the process's whole lifetime. So instead of a
	wake broadcast, an idle worker with nothing local left discovers global
	quiescence via its own short, bounded periodic recheck - see
	drain_fully's own docstring (_QUIESCENCE_RECHECK_MS). '''
	live_tasks: atomic.Atomic[i64]
	def __init__( self ) -> None:
		self.live_tasks = atomic.Atomic[i64]( 0 )


# how often an idle Reactor-owned Worker (empty queues, no local signal
# wait, but the Reactor's shared live_tasks count is still nonzero) rechecks
# whether the REST of the Reactor has finished - see drain_fully's own
# docstring. Short enough that Reactor.run() converges promptly once every
# worker's own work is genuinely done; long enough that idle workers aren't
# meaningfully busy-spinning while waiting on siblings that still have work.
_QUIESCENCE_RECHECK_MS: i32 = 50


class Worker:
	__pending_tasks: list[Closure[[], None]]
	__ready_to_unpark: list[fiber.Fiber]
	__idle_pool: list[fiber.Fiber]
	__poller: poller.Poller
	__registered_fds: list[poller.SOCKET]
	__waiting: list[_PendingWait]
	__wake_read: socket.Socket
	__wake_write: socket.Socket
	__wake_fd: poller.SOCKET
	__shutting_down: atomic.Atomic[bool]
	__reactor_state: _ReactorState|None

	def __init__( self ) -> None:
		self.__pending_tasks = list[Closure[[], None]]()
		self.__ready_to_unpark = list[fiber.Fiber]()
		self.__idle_pool = list[fiber.Fiber]()
		self.__poller = poller.Poller()
		self.__registered_fds = list[poller.SOCKET]()
		self.__waiting = list[_PendingWait]()
		( wake_read, wake_write ) = socket.make_loopback_pair()
		poller.set_nonblocking( wake_read.fileno() ).unwrap( 'Worker.__init__: set_nonblocking (wake read side) failed' )
		poller.set_nonblocking( wake_write.fileno() ).unwrap( 'Worker.__init__: set_nonblocking (wake write side) failed' )
		self.__wake_read = wake_read
		self.__wake_write = wake_write
		self.__wake_fd = wake_read.fileno()
		self.__shutting_down = atomic.Atomic[bool]( False )
		self.__reactor_state = None
		# registered ONCE, unconditionally, for this Worker's whole
		# lifetime - unlike __registered_fds/__waiting (per-wait, torn
		# down once satisfied), the wake fd is infrastructure, always
		# watched, never removed
		self.__poller.register( self.__wake_fd, True, False ).unwrap( 'Worker.__init__: poller register (wake fd) failed' )

	def _attach_reactor_state( self, state: _ReactorState ) -> None:
		''' internal - Reactor.__init__ calls this once per worker, right
		after constructing it, so drain_fully() can tell "nothing queued on
		ME right now" apart from "nothing live ANYWHERE in this Reactor" -
		see _ReactorState's own docstring for why round-robin spawn() across
		N workers needs that distinction. A bare Worker() (no Reactor) never
		gets this call, leaving __reactor_state None - drain_fully() keeps
		its old immediate-return-once-locally-idle contract unchanged in
		that case, exactly what every existing single-worker test relies
		on. '''
		self.__reactor_state = state

	def __poke_wake( self ) -> None:
		''' interrupts a thread currently BLOCKED inside this Worker's own
		poller.wait() (possibly a different thread than the caller - see
		schedule()/request_shutdown(), the two callers) - shared by both,
		since both need the exact same "make sure the next tick actually
		runs soon" guarantee. Best-effort (WouldBlock on the wake write is
		silently ignored) - one byte already sitting in the wake socket's
		own buffer, undrained, already guarantees the NEXT poller.wait()
		wakes up, so a second poke landing on top of it would be
		redundant, not lost. '''
		poke: bytes = b'x'
		match self.__wake_write.send( poke.get_const_ptr(), usize( 1 )):
			case Result.Ok( _n ):
				pass
			case Result.Err( e ):
				if e != OSError.WouldBlock:
					sys.panic( 'Worker.__poke_wake: wake-pair write failed unexpectedly' )

	def wake_external( self ) -> None:
		''' pokes this Worker's own wake pair from OUTSIDE any Worker/
		Reactor machinery entirely - the public counterpart to
		schedule()/request_shutdown()'s own internal __poke_wake() calls,
		for a producer that isn't part of this module at all: lib/
		asyncfile.py's thread pool calls this once a background file I/O
		job's CompletionHandle is filled in, so the Worker driving the
		waiting fiber notices promptly (via __drain_completed_waits, on its
		own next tick) instead of only finding out whenever something else
		happens to wake its poller. Same underlying mechanism, safe to call
		from any thread for the same reason schedule()/request_shutdown()
		already are (see __poke_wake's own docstring). '''
		self.__poke_wake()

	def schedule( self, task: Closure[[], None] ) -> None:
		''' enqueue a fresh task - picked up by whichever thread next calls
		run_until_idle() on this Worker (may be a different thread than
		the caller, e.g. Reactor.spawn() called from outside any worker).
		Also pokes the wake pair (__poke_wake) so a thread currently
		BLOCKED inside this Worker's own drain_fully() (waiting on some
		other Signal, see its own docstring) notices promptly instead of
		only finding this task once whatever it was already waiting on
		eventually fires. For a Reactor-owned worker, also counts this task
		as live in the shared _ReactorState (see its own docstring) - paired
		with __requeue_by_state's own decrement once the task actually
		finishes. '''
		state: _ReactorState|None = self.__reactor_state
		if state is not None:
			state.live_tasks.fetch_add( 1 )
		self.__pending_tasks.append( task ).unwrap( 'Worker.schedule: queue overflow' )
		self.__poke_wake()

	def request_shutdown( self ) -> None:
		''' callable from ANY thread, including while this Worker's own
		drain_fully() is genuinely blocked on another thread. Marks this
		Worker for shutdown and pokes the wake pair (same mechanism
		schedule() uses) so a blocked poller.wait() notices promptly.
		Every fiber currently parked in wait_for_signal() gets unparked
		with a ShuttingDown error on the NEXT tick (see __drain_waiting_
		for_shutdown) rather than waiting for its own signal, which might
		never come. Idempotent - safe to call more than once (e.g. from
		Reactor.shutdown() iterating every worker, or a caller that wants
		to call it defensively). '''
		self.__shutting_down.store( True )
		self.__poke_wake()

	def __take_idle_fiber( self ) -> fiber.Fiber:
		match self.__idle_pool.pop():
			case Result.Ok( f ):
				return f
			case Result.Err( _ ):
				return fiber.Fiber()

	def __is_waiting_on_signal( self, f: fiber.Fiber ) -> bool:
		n: usize = self.__waiting.__len__()
		i: usize = 0
		while i < n:
			w: _PendingWait = self.__waiting.__getitem__( i ).unwrap( 'Worker.__is_waiting_on_signal: index in bounds by construction' )
			if w.waiting_fiber is f:
				return True
			with compiler.wrap_arithmetic:
				i = i + 1
		return False

	def __requeue_by_state( self, f: fiber.Fiber ) -> None:
		match f.state():
			case fiber.FiberState.IDLE:
				self.__idle_pool.append( f ).unwrap( 'Worker: idle pool overflow' )
				# the task that just ran to completion is no longer live -
				# balances the +1 its own originating schedule() call made,
				# however many park()/unpark() cycles happened in between
				state: _ReactorState|None = self.__reactor_state
				if state is not None:
					state.live_tasks.fetch_sub( 1 )
			case fiber.FiberState.PARKED:
				# a fiber parked via _wait_on_signal is ALREADY tracked in
				# __waiting (appended there before its own fiber.park()
				# call) and must NOT also land in __ready_to_unpark - that
				# would unpark it on the very next tick regardless of
				# whether its signal has actually fired, defeating the
				# whole point of waiting for it. An ordinary cooperative
				# fiber.park() (not signal-driven) isn't in __waiting, so
				# it takes the normal path unchanged.
				if not self.__is_waiting_on_signal( f ):
					self.__ready_to_unpark.append( f ).unwrap( 'Worker: ready-to-unpark queue overflow' )
			case fiber.FiberState.RUNNING:
				sys.panic( 'Worker: fiber reported RUNNING after being switched out of - internal bug' )

	def _wait_on_signal( self, signal: Signal ) -> Result[None, WaitError]:
		''' called from WITHIN a running fiber's own task (via the free
		function wait_for_signal, never directly) - for an FdReady signal,
		registers its fd with this worker's own poller if not already
		watched; records which fiber is waiting for it, and parks. Resumes
		once ONE of three things happens: a later run_until_idle()'s own
		__check_signals() notices the fd became ready (see
		__requeue_by_state's own comment for the other half of how that
		stays exactly-once), __drain_waiting_for_shutdown() force-resumes
		it because this worker was asked to stop, or __drain_expired_waits()
		force-resumes it because an active `with timeout(...):` deadline
		elapsed first - checked in that order (shutdown first, matching
		this method's own pre-park check) both BEFORE parking (an
		already-in-progress shutdown shouldn't register/park at all - see
		its own comment) and AFTER (the only way to tell which of the three
		actually happened - pw is the SAME object appended to __waiting
		below, so a mutation to pw.timed_out by __drain_expired_waits is
		visible here through this same local, even though that code runs
		from a completely different call). '''
		if self.__shutting_down.load():
			return Result.Err( WaitError.Shutdown( None ))
		match signal:
			case Signal.FdReady( fdr ):
				if not self.__is_registered( fdr.fd ):
					self.__poller.register( fdr.fd, fdr.want_read, fdr.want_write ).unwrap( 'Worker._wait_on_signal: poller register failed' )
					self.__registered_fds.append( fdr.fd ).unwrap( 'Worker._wait_on_signal: registered-fd list overflow' )
			case Signal.Completion( _ ):
				pass   # nothing to register - the completing thread pokes wake_external() directly, see __drain_completed_waits
		cur: fiber.Fiber|None = fiber.current()
		if cur is None:
			sys.panic( 'Worker._wait_on_signal: no current fiber - must be called from inside a task this Worker is running' )
		pw: _PendingWait = _PendingWait( signal = signal, waiting_fiber = cur )
		self.__waiting.append( pw ).unwrap( 'Worker._wait_on_signal: waiting-list overflow' )
		fiber.park()
		if self.__shutting_down.load():
			return Result.Err( WaitError.Shutdown( None ))
		if pw.timed_out:
			return Result.Err( WaitError.TimedOut( None ))
		return Result.Ok( None )

	def __soonest_deadline( self ) -> f64:
		''' fiber.NO_DEADLINE if nothing currently in __waiting has an
		active `with timeout(...):` deadline, otherwise the earliest one -
		drives __check_signals' own poller.wait() timeout so a real OS-level
		wakeup happens close to when a timeout should actually fire, rather
		than only whenever some unrelated fd event happens to wake it. '''
		soonest: f64 = fiber.NO_DEADLINE
		n: usize = self.__waiting.__len__()
		i: usize = 0
		while i < n:
			w: _PendingWait = self.__waiting.__getitem__( i ).unwrap( 'Worker.__soonest_deadline: index in bounds by construction' )
			d: f64 = w.waiting_fiber.get_deadline()
			if d != fiber.NO_DEADLINE and ( soonest == fiber.NO_DEADLINE or d < soonest ):
				soonest = d
			with compiler.wrap_arithmetic:
				i = i + 1
		return soonest

	def __fd_still_waited_on( self, fd: poller.SOCKET ) -> bool:
		''' whether some OTHER entry still in __waiting (as of the call
		site's own snapshot) still needs `fd` registered - guards
		__drain_expired_waits against unregistering an fd out from under a
		sibling waiter that shares it and hasn't timed out. '''
		n: usize = self.__waiting.__len__()
		i: usize = 0
		while i < n:
			w: _PendingWait = self.__waiting.__getitem__( i ).unwrap( 'Worker.__fd_still_waited_on: index in bounds by construction' )
			match w.signal:
				case Signal.FdReady( fdr ):
					if fdr.fd == fd:
						return True
				case Signal.Completion( _ ):
					pass
			with compiler.wrap_arithmetic:
				i = i + 1
		return False

	def __drain_expired_waits( self ) -> bool:
		''' sweeps __waiting for every entry whose own fiber's deadline has
		now passed, moving each to __ready_to_unpark with timed_out=True
		set first (see _PendingWait's own comment) - same "requeue for the
		NEXT tick, not this one" discipline __check_signals/__drain_
		waiting_for_shutdown already use. Runs on every run_until_idle tick
		(not just right after a deadline-derived poller.wait() elapses) so
		an already-expired deadline is still caught by a purely non-blocking
		peek (poller_timeout_ms=0) that never touches the poller's own
		timeout math at all. '''
		if self.__waiting.__len__() == 0:
			return False
		now: f64 = time.monotonic()
		expired: list[_PendingWait] = list[_PendingWait]()
		still_waiting: list[_PendingWait] = list[_PendingWait]()
		n: usize = self.__waiting.__len__()
		i: usize = 0
		while i < n:
			w: _PendingWait = self.__waiting.__getitem__( i ).unwrap( 'Worker.__drain_expired_waits: index in bounds by construction' )
			d: f64 = w.waiting_fiber.get_deadline()
			if d != fiber.NO_DEADLINE and now >= d:
				expired.append( w ).unwrap( 'Worker.__drain_expired_waits: expired-list overflow' )
			else:
				still_waiting.append( w ).unwrap( 'Worker.__drain_expired_waits: rebuild overflow' )
			with compiler.wrap_arithmetic:
				i = i + 1
		if expired.__len__() == 0:
			return False
		self.__waiting = still_waiting
		m: usize = expired.__len__()
		j: usize = 0
		while j < m:
			# distinct name from the sweep loop's own `w` above - a
			# variable's type is only ever declared once per function
			ew: _PendingWait = expired.__getitem__( j ).unwrap( 'Worker.__drain_expired_waits: index in bounds by construction' )
			ew.timed_out = True
			self.__ready_to_unpark.append( ew.waiting_fiber ).unwrap( 'Worker.__drain_expired_waits: ready-to-unpark queue overflow' )
			match ew.signal:
				case Signal.FdReady( fdr ):
					if self.__is_registered( fdr.fd ) and not self.__fd_still_waited_on( fdr.fd ):
						self.__poller.unregister( fdr.fd ).unwrap( 'Worker.__drain_expired_waits: poller unregister failed' )
						self.__forget_registered_fd( fdr.fd )
				case Signal.Completion( _ ):
					pass   # the pool job keeps running regardless - its eventual complete() lands on an abandoned handle, harmlessly (see CompletionHandle's own docstring)
			with compiler.wrap_arithmetic:
				j = j + 1
		return True

	def __drain_completed_waits( self ) -> bool:
		''' sweeps __waiting for every Signal.Completion entry whose own
		handle.is_done() is now true, moving each to __ready_to_unpark -
		same "requeue for the NEXT tick, not this one" discipline every
		other drain method here already uses. Nothing to unregister (a
		Completion signal was never registered with this Worker's own
		poller in the first place - see _wait_on_signal's own Completion
		arm), so this is simpler than __drain_expired_waits/__check_signals'
		own sweeps. Runs on every __check_signals call regardless of
		whether THIS tick's own wake was actually caused by a completed
		file I/O job or something unrelated - cheap at the __waiting sizes
		this codebase expects (bounded by one Worker's own concurrent
		connection count), same tradeoff __drain_expired_waits already
		makes. '''
		if self.__waiting.__len__() == 0:
			return False
		still_waiting: list[_PendingWait] = list[_PendingWait]()
		progressed: bool = False
		n: usize = self.__waiting.__len__()
		i: usize = 0
		while i < n:
			w: _PendingWait = self.__waiting.__getitem__( i ).unwrap( 'Worker.__drain_completed_waits: index in bounds by construction' )
			done: bool = False
			match w.signal:
				case Signal.FdReady( _ ):
					pass
				case Signal.Completion( handle ):
					done = handle.is_done()
			if done:
				self.__ready_to_unpark.append( w.waiting_fiber ).unwrap( 'Worker.__drain_completed_waits: ready-to-unpark queue overflow' )
				progressed = True
			else:
				still_waiting.append( w ).unwrap( 'Worker.__drain_completed_waits: rebuild overflow' )
			with compiler.wrap_arithmetic:
				i = i + 1
		self.__waiting = still_waiting
		return progressed

	def __is_registered( self, fd: poller.SOCKET ) -> bool:
		n: usize = self.__registered_fds.__len__()
		i: usize = 0
		while i < n:
			existing: poller.SOCKET = self.__registered_fds.__getitem__( i ).unwrap( 'Worker.__is_registered: index in bounds by construction' )
			if existing == fd:
				return True
			with compiler.wrap_arithmetic:
				i = i + 1
		return False

	def __forget_registered_fd( self, fd: poller.SOCKET ) -> None:
		kept: list[poller.SOCKET] = list[poller.SOCKET]()
		n: usize = self.__registered_fds.__len__()
		i: usize = 0
		while i < n:
			existing: poller.SOCKET = self.__registered_fds.__getitem__( i ).unwrap( 'Worker.__forget_registered_fd: index in bounds by construction' )
			if existing != fd:
				kept.append( existing ).unwrap( 'Worker.__forget_registered_fd: rebuild overflow' )
			with compiler.wrap_arithmetic:
				i = i + 1
		self.__registered_fds = kept

	def __drain_waiting_for_shutdown( self ) -> bool:
		''' once request_shutdown() has been called, every fiber still in
		__waiting needs to be force-resumed with a ShuttingDown error
		rather than left hoping its own signal eventually fires - a
		signal that may never come is exactly why shutdown exists. Moves
		every one of them to __ready_to_unpark (same "requeue for the
		NEXT tick, not this one" discipline __check_signals already
		uses) and unregisters each fd that's still registered (guarded by
		__is_registered - more than one waiter can share an fd, and only
		the FIRST one to arrive here would have registered it - avoids a
		double-unregister). No-op (returns False) once __shutting_down is
		false, or once __waiting is already empty - this runs on EVERY
		run_until_idle() call once shutdown starts, not just once. '''
		if not self.__shutting_down.load():
			return False
		if self.__waiting.__len__() == 0:
			return False
		n: usize = self.__waiting.__len__()
		i: usize = 0
		while i < n:
			w: _PendingWait = self.__waiting.__getitem__( i ).unwrap( 'Worker.__drain_waiting_for_shutdown: index in bounds by construction' )
			self.__ready_to_unpark.append( w.waiting_fiber ).unwrap( 'Worker.__drain_waiting_for_shutdown: ready-to-unpark queue overflow' )
			match w.signal:
				case Signal.FdReady( fdr ):
					if self.__is_registered( fdr.fd ):
						self.__poller.unregister( fdr.fd ).unwrap( 'Worker.__drain_waiting_for_shutdown: poller unregister failed' )
						self.__forget_registered_fd( fdr.fd )
				case Signal.Completion( _ ):
					pass   # same reasoning as __drain_expired_waits' own Completion arm
			with compiler.wrap_arithmetic:
				i = i + 1
		self.__waiting = list[_PendingWait]()
		return True

	def __drain_wake( self ) -> None:
		''' the wake fd is level-triggered and never unregistered (see
		__init__'s own comment) - every byte schedule() ever wrote to it
		MUST be fully drained here, or it would keep reporting ready
		forever (a permanent, spurious "something's ready" busy-spin). '''
		buf: bytearray = bytearray( 64 )
		while True:
			match self.__wake_read.recv( buf.get_ptr(), usize( 64 )):
				case Result.Ok( _n ):
					continue
				case Result.Err( e ):
					if e == OSError.WouldBlock:
						return
					sys.panic( 'Worker.__drain_wake: recv failed unexpectedly' )

	def __check_signals( self, timeout_ms: i32 ) -> bool:
		''' checks this worker's own poller for readiness - timeout_ms=0
		(run_until_idle()'s own default) is a non-blocking peek, matching
		its "advances exactly what's already ready right now" contract;
		drain_fully() passes a real (possibly infinite) timeout instead
		once its other queues are genuinely empty but a signal wait is
		still outstanding - see its own docstring. Any fd reported ready
		gets unregistered immediately (a "register" describes ONE wait,
		not a persistent subscription - a caller that turns out to still
		need more, e.g. a spurious wakeup or a short read, calls
		wait_for_signal() again to re-register) and every fiber waiting on
		that fd moves to __ready_to_unpark, to be unparked on the NEXT
		tick (not this one - same snapshot-then-requeue discipline the
		rest of this method already uses). The wake fd (__init__'s own
		self-pipe-equivalent) is handled separately - drained, not
		unregistered, and never matched against __waiting (nothing is
		ever "waiting on" it in that sense - see __drain_wake).
		Additionally clamps timeout_ms down to __soonest_deadline() (never
		up - a caller-requested SHORTER wait always wins) so a genuinely
		infinite drain_fully() block still wakes up promptly for a `with
		timeout(...):` deadline with nothing else outstanding, then sweeps
		__waiting for any deadline that's passed via __drain_expired_waits
		- covers both "this call's own poller.wait() just elapsed because of
		a deadline" and "an unrelated fd event returned first, but some
		OTHER waiter's deadline had already passed anyway". '''
		effective_timeout_ms: i32 = timeout_ms
		soonest: f64 = self.__soonest_deadline()
		if soonest != fiber.NO_DEADLINE:
			deadline_ms: i32 = _ms_until( soonest )
			if timeout_ms < 0 or deadline_ms < timeout_ms:
				effective_timeout_ms = deadline_ms
		ready: list[poller.ReadyEvent] = self.__poller.wait( effective_timeout_ms ).unwrap( 'Worker.__check_signals: poller wait failed' )
		progressed: bool = False
		n: usize = ready.__len__()
		i: usize = 0
		while i < n:
			ev: poller.ReadyEvent = ready.__getitem__( i ).unwrap( 'Worker.__check_signals: index in bounds by construction' )
			if ev.fd == self.__wake_fd:
				self.__drain_wake()
				with compiler.wrap_arithmetic:
					i = i + 1
				continue
			# every entry reaching this sweep came from the fd-based
			# poller (self.__poller.wait() above), so only FdReady
			# entries can ever match here by construction - a future
			# Completion/bare-wake entry in __waiting would never be
			# found via THIS fd, only via its own separate mechanism.
			still_waiting: list[_PendingWait] = list[_PendingWait]()
			m: usize = self.__waiting.__len__()
			j: usize = 0
			while j < m:
				w: _PendingWait = self.__waiting.__getitem__( j ).unwrap( 'Worker.__check_signals: index in bounds by construction' )
				matched: bool = False
				match w.signal:
					case Signal.FdReady( fdr ):
						matched = fdr.fd == ev.fd
					case Signal.Completion( _ ):
						pass   # never matched via an fd-based poller event - see __drain_completed_waits
				if matched:
					self.__ready_to_unpark.append( w.waiting_fiber ).unwrap( 'Worker.__check_signals: ready-to-unpark queue overflow' )
					progressed = True
				else:
					still_waiting.append( w ).unwrap( 'Worker.__check_signals: rebuild overflow' )
				with compiler.wrap_arithmetic:
					j = j + 1
			self.__waiting = still_waiting
			self.__poller.unregister( ev.fd ).unwrap( 'Worker.__check_signals: poller unregister failed' )
			self.__forget_registered_fd( ev.fd )
			with compiler.wrap_arithmetic:
				i = i + 1
		if self.__drain_expired_waits():
			progressed = True
		if self.__drain_completed_waits():
			progressed = True
		return progressed

	def run_until_idle( self, poller_timeout_ms: i32 = 0 ) -> bool:
		''' drains exactly the work that was already queued when this call
		began - __ready_to_unpark and __pending_tasks, plus a poller check
		for any Signal that's ready (__check_signals) - driving each fiber
		via unpark()/start() once, and returns whether it processed
		anything. Deliberately bounded to a snapshot of each queue's
		length rather than looping until both are empty: a fiber that
		parks again during this same call (cooperatively, or via
		wait_for_signal()) gets requeued for the NEXT call to pick up, not
		immediately redriven here. Safe to call again later once more work
		has been scheduled, a fiber has parked, or a signal has fired - a
		fresh call just picks up wherever things are at that point. Calls
		fiber.enable_current_thread() itself (idempotent) - Reactor.run()
		invokes this on a freshly-spawned OS thread that's never been
		fiber-enabled, and Windows' SwitchToFiber requires that before
		it'll accept the thread as a switch target. Also sets this
		thread's own current_worker() to self - see this module's header
		comment for why that's unconditional, not idempotent.

		poller_timeout_ms defaults to 0 (a non-blocking peek) - this is
		what makes the method safe to use as a single-step probe in tests
		(never blocks, matches its own historical contract exactly).
		drain_fully() below is the only caller that ever passes something
		else, once its OTHER queues are genuinely empty but a signal wait
		is still outstanding - see its own docstring for why blocking only
		makes sense at that specific point, not on every call. '''
		fiber.enable_current_thread()
		_current_worker.set( self )
		progressed: bool = False
		to_unpark: usize = self.__ready_to_unpark.__len__()
		while to_unpark > 0:
			match self.__ready_to_unpark.pop():
				case Result.Ok( f ):
					f.unpark()
					self.__requeue_by_state( f )
					progressed = True
				case Result.Err( _ ):
					pass
			with compiler.wrap_arithmetic:
				to_unpark = to_unpark - 1
		to_start: usize = self.__pending_tasks.__len__()
		while to_start > 0:
			match self.__pending_tasks.pop():
				case Result.Ok( task ):
					# distinct name from the unpark loop's own `f` above -
					# a variable's type is only ever declared once per
					# function, even though these two match-arm bindings
					# are otherwise unrelated
					idle_fiber: fiber.Fiber = self.__take_idle_fiber()
					idle_fiber.start( task )
					self.__requeue_by_state( idle_fiber )
					progressed = True
				case Result.Err( _ ):
					pass
			with compiler.wrap_arithmetic:
				to_start = to_start - 1
		if self.__drain_waiting_for_shutdown():
			progressed = True
		if self.__check_signals( poller_timeout_ms ):
			progressed = True
		return progressed

	def drain_fully( self ) -> None:
		''' runs this worker forever, in two alternating modes: drain
		everything ALREADY ready (non-blocking ticks, exactly like
		before), and once a tick genuinely makes no progress, either
		return (nothing outstanding at all - the original "batch of work,
		then done" contract, unchanged for every existing non-Signal use)
		or BLOCK on the poller (an outstanding Signal wait exists, so
		"idle" doesn't mean "done") until either that signal fires,
		schedule() pokes the wake pair from another thread, or request_
		shutdown() does (__init__'s own self-pipe equivalent - without it,
		blocking here would starve any task scheduled onto an already-
		blocked worker until whatever it WAS waiting on happened to fire
		on its own). An infinite poller timeout is safe specifically
		because that wake pair exists - there is always a way back out of
		the blocking call. This DOES mean drain_fully()/Reactor.run() can
		legitimately never return for a worker with a standing signal
		wait that's never satisfied (e.g. a real, long-lived server
		connection) - the correct "keep serving" behavior, not a bug -
		UNLESS request_shutdown() has been called: once __shutting_down
		is set, run_until_idle()'s own __drain_waiting_for_shutdown()
		force-resumes every waiting fiber with a ShuttingDown error
		instead, so this loop naturally converges to "genuinely nothing
		outstanding" and returns, same as the ordinary batch case.

		For a bare Worker() (no Reactor, __reactor_state is None), "locally
		idle with nothing in __waiting" has always meant "genuinely done" -
		unchanged. For a Reactor-owned worker it does NOT: Reactor.spawn()
		round-robins across workers, so a worker that happens to drain to
		empty first must not exit while a SIBLING worker's still-running
		fiber could yet spawn fresh work directly onto it - it would exit,
		its OS thread would end, and Reactor.spawn()'s later w.schedule()
		call would enqueue a task nobody is left driving (this was a real,
		confirmed bug: a multi-worker Reactor could silently drop work
		spawned mid-run onto an already-"finished" worker). So a locally-
		idle Reactor-owned worker instead checks the shared _ReactorState's
		live_tasks count (see its own docstring) - only genuinely returns
		once that's zero (nothing live ANYWHERE in this Reactor), otherwise
		blocks for a short, bounded interval (_QUIESCENCE_RECHECK_MS, not
		infinite - see _ReactorState's own docstring for why this is a
		periodic recheck rather than a wake broadcast) and loops back to
		pick up either fresh local work or the eventual global-zero. '''
		while True:
			if self.run_until_idle( 0 ):
				continue
			if self.__waiting.__len__() == 0:
				state: _ReactorState|None = self.__reactor_state
				if state is None:
					return
				if state.live_tasks.load() == 0:
					return
				self.run_until_idle( _QUIESCENCE_RECHECK_MS )
				continue
			self.run_until_idle( -1 )


class Reactor:
	__workers: list[Worker]
	__threads: list[threading.Thread]
	__next_worker: usize

	def __init__( self, num_workers: usize ) -> None:
		self.__workers = list[Worker]()
		# every worker shares the SAME _ReactorState (one live_tasks
		# counter for the whole Reactor, not one per worker) - see
		# Worker.drain_fully's own docstring for why a locally-idle worker
		# needs this to tell "nothing queued on me" apart from "nothing
		# live anywhere in this Reactor"
		state: _ReactorState = _ReactorState()
		i: usize = 0
		while i < num_workers:
			w: Worker = Worker()
			w._attach_reactor_state( state )
			self.__workers.append( w ).unwrap( 'Reactor.__init__: worker list overflow' )
			with compiler.wrap_arithmetic:
				i = i + 1
		self.__threads = list[threading.Thread]()
		self.__next_worker = 0

	def spawn( self, task: Closure[[], None] ) -> None:
		''' schedule `task` onto one of this Reactor's workers (plain
		round-robin - no load awareness yet). Safe to call before or after
		run() starts the worker threads. '''
		idx: usize = self.__next_worker
		with compiler.panic_arithmetic( 'Reactor.spawn: worker count is zero' ):
			self.__next_worker = ( idx + 1 ) % self.__workers.__len__()
		w: Worker = self.__workers.__getitem__( idx ).unwrap( 'Reactor.spawn: worker index in bounds by construction' )
		w.schedule( task )

	def run( self ) -> None:
		''' starts each worker's drain_fully() on its own OS thread and
		blocks until every one of them has drained its queues (including
		fibers that parked and got requeued mid-drain - see
		Worker.drain_fully). '''
		n: usize = self.__workers.__len__()
		i: usize = 0
		while i < n:
			w: Worker = self.__workers.__getitem__( i ).unwrap( 'Reactor.run: worker index in bounds by construction' )
			t: threading.Thread = threading.Thread( w.drain_fully )
			self.__threads.append( t ).unwrap( 'Reactor.run: thread list overflow' )
			with compiler.wrap_arithmetic:
				i = i + 1
		n_threads: usize = self.__threads.__len__()
		i = 0
		while i < n_threads:
			# distinct name from the spawn loop's own `t` above - a
			# variable's type is only ever declared once per function (no
			# block scoping), so reusing `t` here would be a redeclaration
			joining: threading.Thread = self.__threads.__getitem__( i ).unwrap( 'Reactor.run: thread index in bounds by construction' )
			joining.join()
			with compiler.wrap_arithmetic:
				i = i + 1

	def shutdown( self ) -> None:
		''' requests every worker to stop - callable from ANY thread,
		including while run() is still blocking on its own .join() calls
		(that's the expected usage: some other thread, e.g. reacting to a
		signal or an admin command, calls this while run() is in
		progress). Each worker notices via its own wake pair (the SAME
		mechanism spawn() uses to interrupt a blocked poller.wait()) and
		unwinds any signal-waiting fiber with a ShuttingDown error rather
		than leaving it hoping for a signal that may never come - see
		Worker.request_shutdown()/__drain_waiting_for_shutdown(). run()
		returns once every worker has actually finished draining (which
		now includes unwinding every shutdown-interrupted fiber, not just
		the ones that already had nothing left to wait for). Does NOT
		reject new schedule()/spawn() calls made after this - a task
		queued during shutdown still runs to completion; there's no
		"stop accepting work" concept here yet, only "stop waiting
		forever for signals that might not come". '''
		n: usize = self.__workers.__len__()
		i: usize = 0
		while i < n:
			w: Worker = self.__workers.__getitem__( i ).unwrap( 'Reactor.shutdown: worker index in bounds by construction' )
			w.request_shutdown()
			with compiler.wrap_arithmetic:
				i = i + 1
