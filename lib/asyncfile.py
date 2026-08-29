'''
The reactor-aware FileOpsInterface implementation, plus AsyncFile - a thin
namespace that hands back an ordinary BinaryReader/BinaryWriter/
BinaryReadWriter (lib/builtins/__File.py) with this module's own
implementation injected into its __aio hook.

Regular files are always "ready" to epoll/WSAPoll, so the readiness-poller
approach that makes lib/tcp.py non-blocking doesn't apply to file I/O -
this offloads the actual blocking read/write syscall to a small background
thread pool instead, freeing the calling fiber's own Worker thread to keep
serving other connections while disk I/O is in flight (io_uring/IOCP stays
deferred - see reactor.Signal.Completion's own docstring: this pool is a
real, non-IOCP producer of that exact same signal shape).

Because BinaryReader/BinaryWriter/BinaryReadWriter's own read()/write()
already branch on __aio being set, there's no separate Async* class here
at all - AsyncFile.binary_reader()/binary_writer()/binary_read_writer()
just call File's own factories and inject this module's FileOpsInterface
implementation. Plain File.binary_reader() (no reactor dependency at all)
and AsyncFile.binary_reader() (this module) both return the exact same
BinaryReader type - only whichever caller wants reactor-aware dispatch
needs to import this module in the first place.

With no current_worker() (no Reactor driving this thread), read()/write()
skip the pool entirely and call the syscall directly - identical to
File.binary_reader() et al., matching every other reactor-optional type in
this codebase (see reactor.wait_for_signal's own docstring).

Deliberately NOT pooled: open()/seek()/tell() - metadata-ish operations,
typically fast even on real disks/network filesystems, not worth the
thread-hop for a first cut (a future revision could pool these too if a
real workload shows otherwise).

The pool itself is a fixed-size set of daemon threads, started at
module/import time (see pool's own comment).
'''

import compiler
import threading
import queue
import atomic
import fs
import reactor

_POOL_SIZE: usize = usize( 4 )


def _wait_error_to_os_error( werr: reactor.WaitError ) -> OSError:
	match werr:
		case reactor.WaitError.Shutdown( _ ):
			return OSError.Interrupted
		case reactor.WaitError.TimedOut( _ ):
			return OSError.TimedOut


# ---------------------------------------------------------------------------
# thread pool - a fixed set of daemon threads, each with its own
# queue.Queue[Job] (FIFO batch-drain - see Queue[T]'s own header comment
# in lib/queue.py). Previously hand-rolled a list[Job].pop()-based drain
# + loopback wake pair: pop() removes the LAST element, so a job could be
# starved indefinitely behind a steady stream of newer submissions on the
# same worker - a real fairness bug for a reactor-driven pool, since it
# directly means unpredictable tail latency for whichever fiber's I/O
# happened to land behind newer requests. Queue[T] fixes that for free.
#
# Job is public (not module-private) for the same reason `pool` (below) is -
# this module's own white-box concurrency tests construct one directly.
# ---------------------------------------------------------------------------

class Job:
	handle: reactor.CompletionHandle
	work:   Closure[[], Result[usize, OSError]]
	waiter: reactor.Worker
	def __init__(
		self,
		handle: reactor.CompletionHandle,
		work:   Closure[[], Result[usize, OSError]],
		waiter: reactor.Worker,
	) -> None:
		self.handle = handle
		self.work = work
		self.waiter = waiter


class _PoolWorker:
	__queue: queue.Queue[Job]

	def __init__( self ) -> None:
		self.__queue = queue.Queue[Job]()

	def submit( self, job: Job ) -> None:
		self.__queue.put( job ).unwrap( '_PoolWorker.submit: unbounded queue put always succeeds' )

	def run_forever( self ) -> None:
		''' blocks (Queue[T].drain()'s own genuine thread-blocking wait - no
		Poller, this thread has nothing else to do while idle) until a job
		arrives, then runs every job currently queued, in FIFO submission
		order - see this class's own header comment for why that matters. '''
		while True:
			batch: UnsafeList[Job] = self.__queue.drain()
			i: usize = 0
			while i < batch.__len__():
				job: Job = batch.__getitem__( i ).unwrap( '_PoolWorker.run_forever: batch index in bounds by construction' )
				work: Closure[[], Result[usize, OSError]] = job.work
				match work():
					case Result.Ok( v ):
						job.handle.complete( Result.Ok( v ))
					case Result.Err( e ):
						job.handle.complete( Result.Err( e ))
				job.waiter.wake_external()
				with compiler.wrap_arithmetic:
					i = i + 1


class _Pool:
	__workers: list[_PoolWorker]
	__threads: list[threading.Thread]
	__next:    atomic.Atomic[usize]

	def __init__( self, size: usize ) -> None:
		self.__workers = list[_PoolWorker]()
		self.__threads = list[threading.Thread]()
		self.__next = atomic.Atomic[usize]( 0 )
		i: usize = 0
		while i < size:
			w: _PoolWorker = _PoolWorker()
			self.__workers.append( w )
			t: threading.Thread = threading.Thread( w.run_forever )
			self.__threads.append( t )
			with compiler.wrap_arithmetic:
				i = i + 1

	def submit( self, job: Job ) -> None:
		''' round-robin via an atomic fetch_add, not a plain load-then-store
		on a bare usize field - two threads calling submit() concurrently
		could otherwise both read the SAME index before either wrote back,
		silently skipping a worker (a lost round-robin step, which also
		lets more jobs pile onto one worker than round-robin intends - see
		threading.ThreadPool.submit()'s own docstring for the identical bug
		found and fixed there first). '''
		ticket: usize = self.__next.fetch_add( usize( 1 ))
		with compiler.panic_arithmetic( '_Pool.submit: pool size is zero' ):
			idx: usize = ticket % self.__workers.__len__()
		w: _PoolWorker = self.__workers.__getitem__( idx ).unwrap( '_Pool.submit: index in bounds by construction' )
		w.submit( job )


# A top-level binding, constructed at module/import time (global-init
# ordering/cross-thread OS-object construction is sound here - see
# _topologically_sort_globals).
#
# Public (not module-private) despite being an implementation detail
# ordinary AsyncFile callers never touch directly - this module's own
# white-box concurrency tests (asyncfile_test.py) submit jobs to it
# directly, bypassing the higher-level read()/write() API, specifically to
# exercise the pool's own dispatch/concurrency behavior in isolation.
pool: _Pool = _Pool( _POOL_SIZE )


# ---------------------------------------------------------------------------
# _ReactorAsyncFileOps — the FileOpsInterface implementation this whole module
# exists to provide. Stateless (a single shared instance, _ops below) -
# every real per-operation state lives in the CompletionHandle/Job each
# call constructs fresh.
# ---------------------------------------------------------------------------

class _ReactorAsyncFileOps( FileOpsInterface ):
	@virtual
	def do_read( self, fd: fs.FD, buf: Ptr[u8], count: usize ) -> Result[usize, OSError]:
		w: reactor.Worker|None = reactor.current_worker()
		if w is None:
			return fs.read_raw( fd, buf, count )
		handle: reactor.CompletionHandle = reactor.CompletionHandle()
		work: Closure[[], Result[usize, OSError]] = lambda: fs.read_raw( fd, buf, count )
		pool.submit( Job( handle = handle, work = work, waiter = w ))
		match reactor.wait_for_signal( reactor.Signal.Completion( handle )):
			case Result.Ok( _ ):
				pass
			case Result.Err( werr ):
				return Result.Err( _wait_error_to_os_error( werr ))
		return handle.take()

	@virtual
	def do_write( self, fd: fs.FD, buf: ConstPtr[u8], count: usize ) -> Result[usize, OSError]:
		w: reactor.Worker|None = reactor.current_worker()
		if w is None:
			return fs.write_raw( fd, buf, count )
		handle: reactor.CompletionHandle = reactor.CompletionHandle()
		work: Closure[[], Result[usize, OSError]] = lambda: fs.write_raw( fd, buf, count )
		pool.submit( Job( handle = handle, work = work, waiter = w ))
		match reactor.wait_for_signal( reactor.Signal.Completion( handle )):
			case Result.Ok( _ ):
				pass
			case Result.Err( werr ):
				return Result.Err( _wait_error_to_os_error( werr ))
		return handle.take()


_async_ops: _ReactorAsyncFileOps = _ReactorAsyncFileOps()


# ---------------------------------------------------------------------------
# AsyncFile — namespace with static factory methods, mirrors
# lib/builtins/__File.py's own File namespace exactly, just injecting
# _async_ops into the BinaryReader/BinaryWriter/BinaryReadWriter File's own
# factories already build (open() itself stays synchronous - see this
# module's own header comment).
# ---------------------------------------------------------------------------

class AsyncFile:

	@staticmethod
	def binary_reader( path: str ) -> Result[BinaryReader, OSError]:
		r: BinaryReader = File.binary_reader( path ).or_return()
		r._set_ops( _async_ops )
		return Result.Ok( r )

	@staticmethod
	def binary_writer(
		path: str,
		append: bool = False,
		truncate: bool = True,
		exists: bool|None = None,
	) -> Result[BinaryWriter, OSError]:
		w: BinaryWriter = File.binary_writer( path, append, truncate, exists ).or_return()
		w._set_ops( _async_ops )
		return Result.Ok( w )

	@staticmethod
	def binary_read_writer(
		path: str,
		append: bool = False,
		truncate: bool = True,
		exists: bool|None = None,
	) -> Result[BinaryReadWriter, OSError]:
		rw: BinaryReadWriter = File.binary_read_writer( path, append, truncate, exists ).or_return()
		rw._set_ops( _async_ops )
		return Result.Ok( rw )
