# lib/queue.py — Queue[T]: a thread-safe FIFO queue, matching Python's own
# queue module in name/location (moved out of threading.py, which still owns
# the FastLock/Semaphore primitives this is built on).

import threading


class QueueFullError: pass


# ---------------------------------------------------------------------------
# Queue[T]: put() appends and posts a Semaphore; drain() blocks (via that
# same Semaphore) until at least one item is pending, then hands back EVERY
# item currently queued as one UnsafeList[T], in FIFO order - not a single-
# item get(). Draining in batches, not one item at a time, is what lets a
# consumer (e.g. sys.py's own threaded stdout writer, see
# sys.enable_threaded_stdout()) coalesce many small puts into one larger
# operation - one write() syscall instead of many - instead of paying
# per-item overhead. A plain single-item get() can be added later if a real
# caller needs strict one-at-a-time consumption instead; nothing here
# forecloses it.
#
# Deliberately minimal - no built-in shutdown/lifecycle concept. A caller
# that needs graceful "stop, but only after everything already queued is
# processed" should push its own sentinel value through the SAME queue
# (e.g. Queue[str|None] with None meaning "no more data") - a plain,
# ordinary item that naturally arrives after everything queued before it,
# with no separate out-of-band signal for drain() to special-case. See
# sys.py's own sys.enable_threaded_stdout()/_ThreadedStream._shutdown()
# for a real example of that convention.
#
# Storage swap, not per-item removal: drain() replaces the internal
# UnsafeList wholesale with a fresh empty one and hands back the old one -
# O(1), no per-element shifting the way a loop of UnsafeList.erase_at(0)
# would be (that's O(n) per pop, since erase_at shifts everything after the
# removed slot left by one). Safe because append order within one
# UnsafeList already IS FIFO order - grabbing the whole list at once
# preserves it exactly, no separate reordering step needed.
#
# threading.ThreadPool/_PoolWorker (threading.py) is built on this, not the
# loopback-socket-pair + list.pop() drain it used to hand-roll - that old
# drain took the LAST element first (see list.pop()'s own docstring), i.e.
# LIFO, which could starve an old job behind a stream of newer submissions.
# Queue[T]'s FIFO batch-drain fixes that for free.
#
# Uses threading.Semaphore, NOT the loopback-socket-pair pattern
# threading._PoolWorker/ThreadPool use elsewhere, as its own cross-thread
# wake primitive. That choice is load-bearing, not stylistic: an earlier
# version of Queue[T] used sockets for this, and a real benchmark (a tight
# print()-via-Queue loop) measured it 8-10x SLOWER than not threading output
# AT ALL - sockets route every send()/recv() through the full network stack
# (protocol handling, buffering, driver stack entry), real overhead even for
# a loopback pair, on top of the OS's own inherent cross-thread wake latency.
# A semaphore is a direct, purpose-built kernel object for exactly this
# "block until signaled" job, with none of that protocol overhead - the
# right tool, not a micro-optimization.
# ---------------------------------------------------------------------------

class Queue[T]:
	__items:     UnsafeList[T]
	__lock:      threading.FastLock
	__max_depth: usize|None
	__sem:       threading.Semaphore

	def __init__( self, max_depth: usize|None = None ) -> None:
		''' max_depth caps put()'s own success - None (default) is
		unbounded, matching ThreadPool's own max_queue_depth convention. '''
		self.__items = UnsafeList[T]()
		self.__lock = threading.FastLock()
		self.__max_depth = max_depth
		self.__sem = threading.Semaphore()

	def put( self, item: T ) -> Result[None, QueueFullError]:
		''' appends, then posts the Semaphore to wake a blocked drain()
		call, if any - harmless (just an accumulated, unconsumed count) if
		nothing is currently waiting; see this module's own header comment
		for why a semaphore, specifically, is the wake primitive here.
		Posts unconditionally on every put() (unlike an earlier, socket-
		based version of this that specifically tried to post/wake only on
		an empty-to-non-empty transition to cut down on syscalls) - a
		semaphore post() is a direct, cheap kernel-object operation, not a
		network-stack round-trip, so there's no real syscall-count pressure
		left to optimize away here; keeping every put() unconditional
		keeps the accounting simple; harmless extra counts are drained by
		drain()'s own loop the same way regardless. Rejected
		(QueueFullError) once max_depth items are already pending - the
		caller decides whether to block/drop/apply its own backpressure;
		this never blocks the producer waiting on the CONSUMER itself
		(only ever a brief lock hold shared with other producers/drain()). '''
		with self.__lock:
			if self.__max_depth is not None:
				limit: usize = self.__max_depth
				if self.__items.__len__() >= limit:
					return Result.Err( QueueFullError() )
			self.__items.append( item )
		self.__sem.post()
		return Result.Ok( None )

	def drain( self ) -> UnsafeList[T]:
		''' blocks until at least one item is pending, then returns EVERY
		item currently queued (FIFO order) as one UnsafeList[T], replacing
		the internal storage with a fresh empty one. Always blocks - there
		is no shutdown/timeout concept here at all (see this class's own
		header comment) - a caller that needs to stop should watch for its
		own sentinel value inside the returned batch. '''
		while True:
			with self.__lock:
				if self.__items.__len__() > 0:
					taken: UnsafeList[T] = self.__items
					self.__items = UnsafeList[T]()
					return taken
			self.__sem.wait()
