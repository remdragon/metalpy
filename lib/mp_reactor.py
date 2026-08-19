# lib/mp_reactor.py — Worker/Reactor scaffolding on top of fiber.Fiber.
#
# NAMED mp_reactor, NOT reactor: confirmed via real compile+run testing that
# naming this module "reactor" (identical content, only the filename/import
# name differed) reliably produces a real crash (use-after-free/heap
# corruption) that "mp_reactor" and every other name tried does not. Ruled
# out: Python-level name collision (no "reactor" module/package installed),
# every known content-hash-keyed compiler cache (%TEMP%/metalpy/cexpr etc -
# none are module-name-keyed). Root cause NOT found within this session's
# time budget - flagged clearly rather than silently worked around. If a
# future session renames this back to reactor.py, re-verify this isn't
# still an issue first.
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
# already queued when it was called - and returns whether it did anything;
# a fiber that parks again mid-tick waits for the NEXT call, it is not
# redriven within the same one (see its own docstring for why). Worker's
# drain_fully() (and Reactor.run(), built on it) loops run_until_idle()
# until a tick is a genuine no-op. None of this blocks waiting for future
# work to show up: there's no I/O source yet to wait ON (no poller, no
# Signal abstraction - see PLAN_NON_BLOCKING_IO), so "nothing left to do
# right now" genuinely means done for now. A long-running reactor will need
# a real blocking-wait-for-work mechanism once real I/O exists; not
# attempted here.
#
# current_worker()-style ambient lookup (so code running INSIDE a fiber can
# find which Worker owns it) is DELIBERATELY NOT built yet - nothing here
# needs it (no I/O registration to do), and it needs a real ThreadLocal[T]
# primitive (doesn't exist yet either) to be safe across more than one
# Worker thread. Build both together when real I/O wiring needs them.
#
# Reactor with more than ONE worker is NOT YET SAFE, for the same
# ThreadLocal[T] reason: fiber.py's own _current/_thread_fiber_handle are
# plain globals (correct only because every real test to date, including
# this file's own, has exactly one OS thread touching them at a time).
# Reactor.run() genuinely does hand each Worker its own freshly-spawned OS
# thread (see Worker.run_until_idle's own fiber.enable_current_thread()
# call), so a single-worker Reactor exercises real cross-thread handoff
# correctly - it's specifically CONCURRENT workers (more than one such
# thread alive and switching fibers at once) that would race on those
# globals. Fix in fiber.py once ThreadLocal[T] exists, not here.

import compiler
import sys
import threading
import fiber

class Worker:
	# __pending_tasks is list[Ptr[None]], NOT list[Closure[[],None]] -
	# deliberately type-erased. Confirmed via a minimal real compile+run
	# probe (isolated from everything else in this file - no Fiber, no
	# incref, just list.append(closure)/pop()/unwrap()) that list[T]'s
	# generic RC storage/retrieval is ITSELF broken for T=Closure today -
	# real heap corruption, not just a missing incref (that was the
	# original diagnosis; too narrow - the actual gap goes deeper than one
	# accessor). Root cause is the same known, already-flagged Closure/
	# Capture RC gap another session is rebuilding at its source (see
	# fiber.py's own __pending field, which already worked around this
	# exact class of problem the same way - Ptr[None] isn't RC-tracked at
	# all, so it never touches list[T]'s broken Closure-specific path).
	# schedule()/run_until_idle() below own the incref/cast/decref manually
	# instead, mirroring Fiber.start()/_run_loop's already-verified-working
	# pattern exactly.
	__pending_tasks: list[Ptr[None]]
	__ready_to_unpark: list[fiber.Fiber]
	__idle_pool: list[fiber.Fiber]

	def __init__( self ) -> None:
		self.__pending_tasks = list[Ptr[None]]()
		self.__ready_to_unpark = list[fiber.Fiber]()
		self.__idle_pool = list[fiber.Fiber]()

	def schedule( self, task: Closure[[], None] ) -> None:
		''' enqueue a fresh task - picked up by whichever thread next calls
		run_until_idle() on this Worker (may be a different thread than
		the caller, e.g. Reactor.spawn() called from outside any worker).
		Increfs before stashing the type-erased pointer - see __pending_tasks'
		own comment for why this doesn't just store `task` directly - so the
		queue holds its own genuinely-owned reference, released back into a
		real Closure local (no extra incref needed there) when popped in
		run_until_idle(). '''
		compiler.incref( task )
		raw: Ptr[None] = compiler.cast( Ptr[None], task )
		self.__pending_tasks.append( raw ).unwrap( 'Worker.schedule: queue overflow' )

	def __take_idle_fiber( self ) -> fiber.Fiber:
		match self.__idle_pool.pop():
			case Result.Ok( f ):
				return f
			case Result.Err( _ ):
				return fiber.Fiber()

	def __requeue_by_state( self, f: fiber.Fiber ) -> None:
		match f.state():
			case fiber.FiberState.IDLE:
				self.__idle_pool.append( f ).unwrap( 'Worker: idle pool overflow' )
			case fiber.FiberState.PARKED:
				self.__ready_to_unpark.append( f ).unwrap( 'Worker: ready-to-unpark queue overflow' )
			case fiber.FiberState.RUNNING:
				sys.panic( 'Worker: fiber reported RUNNING after being switched out of - internal bug' )

	def run_until_idle( self ) -> bool:
		''' drains exactly the work that was already queued when this call
		began - both __ready_to_unpark and __pending_tasks - driving each
		fiber via unpark()/start() once, and returns whether it processed
		anything. Deliberately bounded to a snapshot of each queue's length
		rather than looping until both are empty: a fiber that parks again
		during this same call gets requeued into __ready_to_unpark for the
		NEXT call to pick up, not immediately redriven here. Without that
		bound, a park() with nothing external to wake it would just be
		redrained in the same call, since nothing else in this scaffolding
		(no poller, no Signal) ever marks it "actually ready" - callers that
		want "keep ticking until this worker is genuinely out of work" (e.g.
		Reactor, via drain_fully()) call this repeatedly instead. Safe to
		call again later once more work has been scheduled or a fiber has
		parked - a fresh call just picks up wherever the queues are at that
		point. Calls fiber.enable_current_thread() itself (idempotent) -
		Reactor.run() invokes this on a freshly-spawned OS thread that's
		never been fiber-enabled, and Windows' SwitchToFiber requires that
		before it'll accept the thread as a switch target. '''
		fiber.enable_current_thread()
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
		# __pending_tasks is list[Ptr[None]] (type-erased) - see its own
		# field comment. Ptr[None] isn't RC-tracked, so this pop()/unwrap()
		# is ordinary, uneventful pointer plumbing; the ONLY real reference
		# involved is the one schedule() minted with its own compiler.incref
		# before storing, which the compiler.cast() below reclaims into
		# `task` as this local's own genuinely-owned copy (consumed by
		# `task`'s own scope-exit decref at the end of this loop iteration,
		# same as any other compiler.cast()-derived local in this codebase -
		# see fiber.py's _run_loop for the identical pattern).
		to_start: usize = self.__pending_tasks.__len__()
		while to_start > 0:
			next_raw: Result[Ptr[None], IndexError] = self.__pending_tasks.pop()
			if next_raw.is_ok():
				raw: Ptr[None] = next_raw.unwrap( 'Worker.run_until_idle: just checked is_ok' )
				task: Closure[[], None] = compiler.cast( Closure[[], None], raw )
				f: fiber.Fiber = self.__take_idle_fiber()
				f.start( task )
				self.__requeue_by_state( f )
				progressed = True
			with compiler.wrap_arithmetic:
				to_start = to_start - 1
		return progressed

	def drain_fully( self ) -> None:
		''' keeps calling run_until_idle() until a full tick processes
		nothing at all - i.e. actually idle, including fibers that parked
		and got requeued mid-drain. This is the "block until this worker
		has nothing left to do right now" contract Reactor.run() wants;
		run_until_idle() itself deliberately only advances one tick's worth
		(see its own docstring) so it stays usable as a single-step probe
		in tests. '''
		while self.run_until_idle():
			pass


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
			t: threading.Thread = self.__threads.__getitem__( i ).unwrap( 'Reactor.run: thread index in bounds by construction' )
			t.join()
			with compiler.wrap_arithmetic:
				i = i + 1
