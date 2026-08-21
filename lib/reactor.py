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
# resume every waiting fiber with a ShuttingDown error (wait_for_signal()
# returns Result[None, ShutdownError], not a bare None) instead of
# leaving it hoping for a signal that may never come - see request_
# shutdown()/__drain_waiting_for_shutdown()'s own docstrings.
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
# either way (list[T]'s own internal lock) - not yet stress-tested under
# real multi-worker concurrency here, though (this file's own tests still
# use a single worker) - worth a dedicated multi-worker test once real
# I/O work lands and there's a genuine reason to run more than one.

import compiler
import sys
import threading
import fiber
import poller
import socket
import atomic

_current_worker: threading.ThreadLocal[Worker] = threading.ThreadLocal[Worker]()

def current_worker() -> Worker|None:
	''' the Worker driving fibers on THIS OS thread right now, or None if
	this thread isn't currently (or has never been) inside a Worker's own
	run_until_idle() - e.g. the thread that just calls Reactor.spawn()
	from outside any worker. See this module's own header comment for the
	full ownership reasoning. '''
	return _current_worker.get()


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


@union
class Signal:
	''' what a fiber is waiting for. A tagged union, not a bare fd+interest
	struct, because "something worth waking a fiber up for" has more than
	one real shape:
	  - FdReady (the only variant implemented so far) - a poller notices a
	    registered fd's readiness (lib/poller.py, epoll/WSAPoll). This is
	    a READINESS signal: once it fires, the caller still has to
	    actually perform the read/write itself, and might get WouldBlock
	    again (a spurious wakeup) - see NonBlockingIO's own eventual
	    read()/write() retry-loop shape.
	  - Completion (future, not built yet) - IOCP/io_uring's own model:
	    the wait target IS the completing operation itself, not a
	    readiness check - by the time this fires, the actual result (bytes
	    transferred, or an error) already exists, nothing left to "try
	    again". A fundamentally different shape from FdReady, which is
	    exactly why this needed to become a union rather than growing
	    fields on one struct - Worker's own internals (__check_signals,
	    __drain_waiting_for_shutdown) will need to branch on kind, not
	    just interpret every Signal as "some fd is ready".
	  - a bare "wake me directly" kind (future, not built yet) - what
	    Queue/Event will need: no fd, no completion object, just "some
	    other fiber/thread called wake() on the specific token I'm holding".
	Worker._wait_on_signal/__check_signals/__drain_waiting_for_shutdown
	are the only places that need to know which kinds exist - everything
	above them (wait_for_signal, and eventually NonBlockingIO's own
	read()/write()) just holds a Signal opaquely and waits for it. '''
	FdReady: FdReadiness

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
	def __init__( self, signal: Signal, waiting_fiber: fiber.Fiber ) -> None:
		self.signal = signal
		self.waiting_fiber = waiting_fiber


class ShutdownError:
	''' returned by wait_for_signal() instead of Ok when the Worker
	driving this fiber has been asked to stop (Reactor.shutdown()/
	Worker.request_shutdown()) - lets handler code written with defer/
	errdefer for cleanup behave identically whether a connection closed
	normally, errored, or the reactor is shutting down (this is the whole
	point: a shutdown-interrupted wait looks like just another kind of
	failure to unwind from, not a special case every caller has to know
	about). Never produced by the no-reactor blocking path below - there
	is no Worker to ask it to stop, so that path always succeeds once its
	signal fires. '''
	pass


def wait_for_signal( signal: Signal ) -> Result[None, ShutdownError]:
	''' the ONE reactor-optional wait primitive every higher-level helper
	(a future NonBlockingIO's read()/write_all()/accept()) is meant to
	call instead of talking to a Worker or a Poller directly - branches
	on current_worker() to either park the calling fiber and wait for
	`signal` via whichever Worker is driving this thread, or perform a
	real, standalone blocking wait if no Worker is driving this thread at
	all. This is what makes the exact same connection-handling code work
	unmodified as ordinary synchronous blocking code with zero Reactor
	setup, or as a cooperative fiber inside a full reactor. '''
	w: Worker|None = current_worker()
	if w is not None:
		return w._wait_on_signal( signal )
	_blocking_wait_no_reactor( signal )
	return Result.Ok( None )

def _blocking_wait_no_reactor( signal: Signal ) -> None:
	''' no Worker driving this thread - there's nothing to park a fiber
	INTO, so this really does block the calling OS thread, same as an
	ordinary blocking recv() would. A throwaway, single-use Poller
	(register one fd, wait with an infinite timeout, let __del__ clean
	up) rather than any shared/cached instance - this path is expected to
	be rare (real reactor-driven code never takes it) and simplicity
	beats reuse here. Only FdReady is handleable this way (a real OS-
	level blocking wait needs something pollable) - the match is
	exhaustive today because FdReady is the only variant that exists;
	adding a second kind will force a real decision here, not a silent
	gap. '''
	match signal:
		case Signal.FdReady( fdr ):
			p: poller.Poller = poller.Poller()
			p.register( fdr.fd, fdr.want_read, fdr.want_write ).unwrap( 'wait_for_signal: poller register failed (no reactor driving this thread)' )
			p.wait( -1 ).unwrap( 'wait_for_signal: poller wait failed (no reactor driving this thread)' )


def _make_wake_pair() -> tuple[socket.Socket, socket.Socket]:
	''' a connected loopback TCP pair used purely as a wake-up signal (the
	classic reactor "self-pipe" trick) - NOT a real pipe(2), since WSAPoll
	can only poll actual SOCKETs on Windows, and this codebase already has
	a fully proven, portable TCP loopback pattern (lib/socket.py) rather
	than needing a second, POSIX-only primitive just for this. Returns
	(read_side, write_side) - the read side is registered with a Worker's
	own poller unconditionally (see Worker.__init__), the write side is
	poked by schedule() to interrupt a blocked poller.wait() on whichever
	thread (possibly a different one) is currently driving this Worker. '''
	listener: socket.Socket = socket.Socket.tcp().unwrap( '_make_wake_pair: listener create failed' )
	listener.bind( '127.0.0.1', u16( 0 )).unwrap( '_make_wake_pair: bind failed' )
	listener.listen().unwrap( '_make_wake_pair: listen failed' )
	bound: socket.SocketAddr = listener.getsockname().unwrap( '_make_wake_pair: getsockname failed' )
	write_side: socket.Socket = socket.Socket.tcp().unwrap( '_make_wake_pair: connect-side create failed' )
	write_side.connect( '127.0.0.1', bound.port() ).unwrap( '_make_wake_pair: connect failed' )
	( read_side, _addr ) = listener.accept().unwrap( '_make_wake_pair: accept failed' )
	return ( read_side, write_side )


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

	def __init__( self ) -> None:
		self.__pending_tasks = list[Closure[[], None]]()
		self.__ready_to_unpark = list[fiber.Fiber]()
		self.__idle_pool = list[fiber.Fiber]()
		self.__poller = poller.Poller()
		self.__registered_fds = list[poller.SOCKET]()
		self.__waiting = list[_PendingWait]()
		( wake_read, wake_write ) = _make_wake_pair()
		poller.set_nonblocking( wake_read.fileno() ).unwrap( 'Worker.__init__: set_nonblocking (wake read side) failed' )
		poller.set_nonblocking( wake_write.fileno() ).unwrap( 'Worker.__init__: set_nonblocking (wake write side) failed' )
		self.__wake_read = wake_read
		self.__wake_write = wake_write
		self.__wake_fd = wake_read.fileno()
		self.__shutting_down = atomic.Atomic[bool]( False )
		# registered ONCE, unconditionally, for this Worker's whole
		# lifetime - unlike __registered_fds/__waiting (per-wait, torn
		# down once satisfied), the wake fd is infrastructure, always
		# watched, never removed
		self.__poller.register( self.__wake_fd, True, False ).unwrap( 'Worker.__init__: poller register (wake fd) failed' )

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

	def schedule( self, task: Closure[[], None] ) -> None:
		''' enqueue a fresh task - picked up by whichever thread next calls
		run_until_idle() on this Worker (may be a different thread than
		the caller, e.g. Reactor.spawn() called from outside any worker).
		Also pokes the wake pair (__poke_wake) so a thread currently
		BLOCKED inside this Worker's own drain_fully() (waiting on some
		other Signal, see its own docstring) notices promptly instead of
		only finding this task once whatever it was already waiting on
		eventually fires. '''
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

	def _wait_on_signal( self, signal: Signal ) -> Result[None, ShutdownError]:
		''' called from WITHIN a running fiber's own task (via the free
		function wait_for_signal, never directly) - for an FdReady signal,
		registers its fd with this worker's own poller if not already
		watched; records which fiber is waiting for it, and parks. Resumes
		once EITHER a
		later run_until_idle()'s own __check_signals() notices the fd
		became ready (see __requeue_by_state's own comment for the other
		half of how that stays exactly-once), OR __drain_waiting_for_
		shutdown() force-resumes it because this worker was asked to stop
		- checked both BEFORE parking (a shutdown already in progress
		when this is first called shouldn't register/park at all - see
		its own comment) and AFTER (the only way to tell which of the two
		actually happened). '''
		if self.__shutting_down.load():
			return Result.Err( ShutdownError() )
		match signal:
			case Signal.FdReady( fdr ):
				if not self.__is_registered( fdr.fd ):
					self.__poller.register( fdr.fd, fdr.want_read, fdr.want_write ).unwrap( 'Worker._wait_on_signal: poller register failed' )
					self.__registered_fds.append( fdr.fd ).unwrap( 'Worker._wait_on_signal: registered-fd list overflow' )
		cur: fiber.Fiber|None = fiber.current()
		if cur is None:
			sys.panic( 'Worker._wait_on_signal: no current fiber - must be called from inside a task this Worker is running' )
		self.__waiting.append( _PendingWait( signal = signal, waiting_fiber = cur )).unwrap( 'Worker._wait_on_signal: waiting-list overflow' )
		fiber.park()
		if self.__shutting_down.load():
			return Result.Err( ShutdownError() )
		return Result.Ok( None )

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
		ever "waiting on" it in that sense - see __drain_wake). '''
		ready: list[poller.ReadyEvent] = self.__poller.wait( timeout_ms ).unwrap( 'Worker.__check_signals: poller wait failed' )
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
		outstanding" and returns, same as the ordinary batch case. '''
		while True:
			if self.run_until_idle( 0 ):
				continue
			if self.__waiting.__len__() == 0:
				return
			self.run_until_idle( -1 )


class Reactor:
	__workers: list[Worker]
	__threads: list[threading.Thread]
	__next_worker: usize

	def __init__( self, num_workers: usize ) -> None:
		self.__workers = list[Worker]()
		i: usize = 0
		while i < num_workers:
			self.__workers.append( Worker() ).unwrap( 'Reactor.__init__: worker list overflow' )
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
