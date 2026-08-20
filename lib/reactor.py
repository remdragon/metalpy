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
# needs it yet (no I/O registration to do). ThreadLocal[T] now exists
# (lib/threading.py) and fiber.py's own _current/_thread_fiber_handle have
# already been converted to use it (real per-OS-thread TLS, not plain
# globals) - build current_worker() the same way once real I/O wiring
# needs it.
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

class Worker:
	__pending_tasks: list[Closure[[], None]]
	__ready_to_unpark: list[fiber.Fiber]
	__idle_pool: list[fiber.Fiber]

	def __init__( self ) -> None:
		self.__pending_tasks = list[Closure[[], None]]()
		self.__ready_to_unpark = list[fiber.Fiber]()
		self.__idle_pool = list[fiber.Fiber]()

	def schedule( self, task: Closure[[], None] ) -> None:
		''' enqueue a fresh task - picked up by whichever thread next calls
		run_until_idle() on this Worker (may be a different thread than
		the caller, e.g. Reactor.spawn() called from outside any worker). '''
		self.__pending_tasks.append( task ).unwrap( 'Worker.schedule: queue overflow' )

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
		to_start: usize = self.__pending_tasks.__len__()
		while to_start > 0:
			match self.__pending_tasks.pop():
				case Result.Ok( task ):
					f: fiber.Fiber = self.__take_idle_fiber()
					f.start( task )
					self.__requeue_by_state( f )
					progressed = True
				case Result.Err( _ ):
					pass
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
