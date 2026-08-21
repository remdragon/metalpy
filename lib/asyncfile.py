'''
AsyncBinaryReader / AsyncBinaryWriter / AsyncBinaryReadWriter — Reader/
Writer/Seekable-conforming file handles whose read()/write() offload the
actual blocking syscall to a small background thread pool when a Reactor is
driving the calling fiber, freeing that fiber's own Worker thread to keep
serving other connections while the disk I/O is in flight - unlike
lib/builtins/__File.py's BinaryReader/BinaryWriter/BinaryReadWriter, which
always call the blocking syscall directly on the calling thread (the right
choice for an ordinary synchronous script, wrong for a server that wants
one slow file read to not stall every OTHER connection sharing that
Worker). This is the "thread-pool dispatch" half of the two options
PLAN_NON_BLOCKING_IO.md's own design notes called out for file I/O
(regular files are always "ready" to epoll/WSAPoll, so a readiness poller
can't help here the way it does for sockets) - io_uring/IOCP is the other,
deferred (see reactor.Signal.Completion's own docstring: this pool is a
real, non-IOCP producer of that exact same signal shape).

With no current_worker() (no Reactor driving this thread), read()/write()
skip the pool entirely and call the syscall directly - identical to
BinaryReader/BinaryWriter/BinaryReadWriter, matching every other
reactor-optional type in this codebase (see reactor.wait_for_signal's own
docstring).

Deliberately NOT pooled: open()/seek()/tell() - metadata-ish operations,
typically fast even on real disks/network filesystems, not worth the
thread-hop for a first cut (a future revision could pool these too if a
real workload shows otherwise).

The pool itself is a fixed-size set of daemon threads, lazily started on
first real use (see _ensure_pool's own comment for why NOT eagerly at
module/import time - a real, confirmed compiler bug).
'''

import compiler
import sys
import threading
import socket
import fs
import io
import reactor
from fs import FD, INVALID_FD, SEEK_END

if compiler.target.os == 'windows':
	from fs import (
		GENERIC_READ, GENERIC_WRITE,
		CREATE_NEW, CREATE_ALWAYS, OPEN_EXISTING, OPEN_ALWAYS, TRUNCATE_EXISTING,
	)
else:
	from fs import (
		O_RDONLY, O_WRONLY, O_RDWR,
		O_CREAT, O_EXCL, O_TRUNC, O_APPEND,
	)

_POOL_SIZE: usize = usize( 4 )


def _wait_error_to_os_error( werr: reactor.WaitError ) -> OSError:
	match werr:
		case reactor.WaitError.Shutdown( _ ):
			return OSError.Interrupted
		case reactor.WaitError.TimedOut( _ ):
			return OSError.TimedOut


# ---------------------------------------------------------------------------
# thread pool - a fixed set of daemon threads, each with its own job queue
# and its own loopback wake pair (used directly, genuinely blocking - no
# Poller involved, unlike reactor.Worker's own non-blocking/polled use of
# the identical socket.make_loopback_pair() primitive).
# ---------------------------------------------------------------------------

class _Job:
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
	__jobs:       list[_Job]
	__wake_read:  socket.Socket
	__wake_write: socket.Socket

	def __init__( self ) -> None:
		self.__jobs = list[_Job]()
		( read_side, write_side ) = socket.make_loopback_pair()
		self.__wake_read = read_side
		self.__wake_write = write_side

	def submit( self, job: _Job ) -> None:
		self.__jobs.append( job ).unwrap( '_PoolWorker.submit: queue overflow' )
		poke: bytes = b'x'
		self.__wake_write.send( poke.get_const_ptr(), usize( 1 )).unwrap( '_PoolWorker.submit: wake failed' )

	def run_forever( self ) -> None:
		''' blocks (a genuine, thread-blocking recv - no Poller, this thread
		has nothing else to do while idle) until a poke arrives, then drains
		and runs every job currently queued - possibly more than one poke's
		worth, which is fine, same "at least one byte means check the queue"
		discipline reactor.Worker's own __drain_wake uses. '''
		buf: bytearray = bytearray( usize( 64 ))
		while True:
			self.__wake_read.recv( buf.get_ptr(), usize( 64 )).unwrap( '_PoolWorker.run_forever: wake recv failed' )
			while True:
				match self.__jobs.pop():
					case Result.Ok( job ):
						work: Closure[[], Result[usize, OSError]] = job.work
						match work():
							case Result.Ok( v ):
								job.handle.complete( Result.Ok( v ))
							case Result.Err( e ):
								job.handle.complete( Result.Err( e ))
						job.waiter.wake_external()
					case Result.Err( _ ):
						break


class _Pool:
	__workers: list[_PoolWorker]
	__threads: list[threading.Thread]
	__next:    usize

	def __init__( self, size: usize ) -> None:
		self.__workers = list[_PoolWorker]()
		self.__threads = list[threading.Thread]()
		self.__next = 0
		i: usize = 0
		while i < size:
			w: _PoolWorker = _PoolWorker()
			self.__workers.append( w ).unwrap( '_Pool.__init__: worker list overflow' )
			t: threading.Thread = threading.Thread( w.run_forever )
			self.__threads.append( t ).unwrap( '_Pool.__init__: thread list overflow' )
			with compiler.wrap_arithmetic:
				i = i + 1

	def submit( self, job: _Job ) -> None:
		idx: usize = self.__next
		with compiler.panic_arithmetic( '_Pool.submit: pool size is zero' ):
			self.__next = ( idx + 1 ) % self.__workers.__len__()
		w: _PoolWorker = self.__workers.__getitem__( idx ).unwrap( '_Pool.submit: index in bounds by construction' )
		w.submit( job )


# Lazily constructed on first real use, NOT eagerly at module/static-init
# time - a real, confirmed compiler bug: a Socket (this pool's own
# per-worker wake pair, via socket.make_loopback_pair()) constructed during
# module-level static initialization (before main() ever runs) crashes
# (SIGILL) the first time a DIFFERENT thread later calls .recv() on it,
# even though the exact same construction succeeds and works fine when
# done from inside main() instead. Isolated with a minimal repro (a Socket
# pair built in a top-level `_x: T = T()` binding vs. the identical code
# built inside main()) - not chased down further here; worth its own
# investigation. Double-checked locking (_pool_lock, itself just a plain
# OS mutex - safe to construct at module scope, unlike a Socket) makes
# first-use construction safe under concurrent callers.
_pool: _Pool|None = None
_pool_lock: threading.FastLock = threading.FastLock()

def _ensure_pool() -> _Pool:
	global _pool
	if _pool is not None:
		return _pool
	_pool_lock.acquire().unwrap( '_ensure_pool: lock acquire failed' )
	if _pool is None:
		_pool = _Pool( _POOL_SIZE )
	p: _Pool = _pool
	_pool_lock.release()
	return p

def _submit_read( fd: FD, buf: Ptr[u8], count: usize, w: reactor.Worker ) -> Result[usize, OSError]:
	handle: reactor.CompletionHandle = reactor.CompletionHandle()
	work: Closure[[], Result[usize, OSError]] = lambda: fs.read_raw( fd, buf, count )
	_ensure_pool().submit( _Job( handle = handle, work = work, waiter = w ))
	match reactor.wait_for_signal( reactor.Signal.Completion( handle )):
		case Result.Ok( _ ):
			pass
		case Result.Err( werr ):
			return Result.Err( _wait_error_to_os_error( werr ))
	return handle.take()

def _submit_write( fd: FD, buf: ConstPtr[u8], count: usize, w: reactor.Worker ) -> Result[usize, OSError]:
	handle: reactor.CompletionHandle = reactor.CompletionHandle()
	work: Closure[[], Result[usize, OSError]] = lambda: fs.write_raw( fd, buf, count )
	_ensure_pool().submit( _Job( handle = handle, work = work, waiter = w ))
	match reactor.wait_for_signal( reactor.Signal.Completion( handle )):
		case Result.Ok( _ ):
			pass
		case Result.Err( werr ):
			return Result.Err( _wait_error_to_os_error( werr ))
	return handle.take()


# ---------------------------------------------------------------------------
# AsyncBinaryReader / AsyncBinaryWriter / AsyncBinaryReadWriter
# ---------------------------------------------------------------------------

class AsyncBinaryReader( io.Reader, io.Seekable ):
	__fd: FD

	def __del__( self ) -> None:
		if self.__fd != INVALID_FD:
			fs.close_raw( self.__fd ).is_ok()

	def read( self, buf: Ptr[u8], count: usize ) -> Result[usize, OSError]:
		w: reactor.Worker|None = reactor.current_worker()
		if w is None:
			return fs.read_raw( self.__fd, buf, count )
		return _submit_read( self.__fd, buf, count, w )

	def seek( self, offset: i64, whence: i32 ) -> Result[i64, OSError]:
		return fs.seek_raw( self.__fd, offset, whence )

	def close( self ) -> None:
		if self.__fd != INVALID_FD:
			fs.close_raw( self.__fd ).is_ok()
			self.__fd = INVALID_FD

	@private
	@staticmethod
	def _from_fd( fd: FD ) -> AsyncBinaryReader:
		return AsyncBinaryReader.__allocate__( __fd = fd )


class AsyncBinaryWriter( io.Writer, io.Seekable ):
	__fd: FD

	def __del__( self ) -> None:
		if self.__fd != INVALID_FD:
			fs.close_raw( self.__fd ).is_ok()

	def write( self, buf: ConstPtr[u8], count: usize ) -> Result[usize, OSError]:
		w: reactor.Worker|None = reactor.current_worker()
		if w is None:
			return fs.write_raw( self.__fd, buf, count )
		return _submit_write( self.__fd, buf, count, w )

	def seek( self, offset: i64, whence: i32 ) -> Result[i64, OSError]:
		return fs.seek_raw( self.__fd, offset, whence )

	def close( self ) -> None:
		if self.__fd != INVALID_FD:
			fs.close_raw( self.__fd ).is_ok()
			self.__fd = INVALID_FD

	@private
	@staticmethod
	def _from_fd( fd: FD ) -> AsyncBinaryWriter:
		return AsyncBinaryWriter.__allocate__( __fd = fd )


class AsyncBinaryReadWriter( io.Reader, io.Writer, io.Seekable ):
	__fd: FD

	def __del__( self ) -> None:
		if self.__fd != INVALID_FD:
			fs.close_raw( self.__fd ).is_ok()

	def read( self, buf: Ptr[u8], count: usize ) -> Result[usize, OSError]:
		w: reactor.Worker|None = reactor.current_worker()
		if w is None:
			return fs.read_raw( self.__fd, buf, count )
		return _submit_read( self.__fd, buf, count, w )

	def write( self, buf: ConstPtr[u8], count: usize ) -> Result[usize, OSError]:
		w: reactor.Worker|None = reactor.current_worker()
		if w is None:
			return fs.write_raw( self.__fd, buf, count )
		return _submit_write( self.__fd, buf, count, w )

	def seek( self, offset: i64, whence: i32 ) -> Result[i64, OSError]:
		return fs.seek_raw( self.__fd, offset, whence )

	def close( self ) -> None:
		if self.__fd != INVALID_FD:
			fs.close_raw( self.__fd ).is_ok()
			self.__fd = INVALID_FD

	@private
	@staticmethod
	def _from_fd( fd: FD ) -> AsyncBinaryReadWriter:
		return AsyncBinaryReadWriter.__allocate__( __fd = fd )


# ---------------------------------------------------------------------------
# AsyncFile — namespace with static factory methods, mirrors
# lib/builtins/__File.py's own File namespace exactly (open() itself stays
# synchronous - see this module's own header comment).
# ---------------------------------------------------------------------------

class AsyncFile:

	@compiler.target( os = 'windows' )
	@staticmethod
	def binary_reader( path: str ) -> Result[AsyncBinaryReader, OSError]:
		fd: FD = fs.open_raw( path.get_cstr(), GENERIC_READ, OPEN_EXISTING ).or_return()
		return Result.Ok( AsyncBinaryReader._from_fd( fd ))

	@compiler.target( os = not 'windows' )
	@staticmethod
	def binary_reader( path: str ) -> Result[AsyncBinaryReader, OSError]:
		fd: FD = fs.open_raw( path.get_cstr(), O_RDONLY, 0 ).or_return()
		return Result.Ok( AsyncBinaryReader._from_fd( fd ))

	@compiler.target( os = 'windows' )
	@staticmethod
	def binary_writer(
		path: str,
		append: bool = False,
		truncate: bool = True,
		exists: bool|None = None,
	) -> Result[AsyncBinaryWriter, OSError]:
		access: u32 = GENERIC_WRITE
		if exists is None:
			if truncate:
				creation: u32 = CREATE_ALWAYS
			else:
				creation: u32 = OPEN_ALWAYS
		elif exists:
			if truncate:
				creation: u32 = TRUNCATE_EXISTING
			else:
				creation: u32 = OPEN_EXISTING
		else:
			creation: u32 = CREATE_NEW
		fd: FD = fs.open_raw( path.get_cstr(), access, creation ).or_return()
		if append:
			fs.seek_raw( fd, 0, SEEK_END ).or_return()
		return Result.Ok( AsyncBinaryWriter._from_fd( fd ))

	@compiler.target( os = not 'windows' )
	@staticmethod
	def binary_writer(
		path: str,
		append: bool = False,
		truncate: bool = True,
		exists: bool|None = None,
	) -> Result[AsyncBinaryWriter, OSError]:
		flags: i32 = O_WRONLY
		if exists is None:
			flags |= O_CREAT
			if truncate:
				flags |= O_TRUNC
		elif exists:
			if truncate:
				flags |= O_TRUNC
		else:
			flags |= O_CREAT | O_EXCL
		if append:
			flags |= O_APPEND
		fd: FD = fs.open_raw( path.get_cstr(), flags, 0o644 ).or_return()
		return Result.Ok( AsyncBinaryWriter._from_fd( fd ))

	@compiler.target( os = 'windows' )
	@staticmethod
	def binary_read_writer(
		path: str,
		append: bool = False,
		truncate: bool = True,
		exists: bool|None = None,
	) -> Result[AsyncBinaryReadWriter, OSError]:
		access: u32 = GENERIC_READ | GENERIC_WRITE
		if exists is None:
			if truncate:
				creation: u32 = CREATE_ALWAYS
			else:
				creation: u32 = OPEN_ALWAYS
		elif exists:
			if truncate:
				creation: u32 = TRUNCATE_EXISTING
			else:
				creation: u32 = OPEN_EXISTING
		else:
			creation: u32 = CREATE_NEW
		fd: FD = fs.open_raw( path.get_cstr(), access, creation ).or_return()
		if append:
			fs.seek_raw( fd, 0, SEEK_END ).or_return()
		return Result.Ok( AsyncBinaryReadWriter._from_fd( fd ))

	@compiler.target( os = not 'windows' )
	@staticmethod
	def binary_read_writer(
		path: str,
		append: bool = False,
		truncate: bool = True,
		exists: bool|None = None,
	) -> Result[AsyncBinaryReadWriter, OSError]:
		flags: i32 = O_RDWR
		if exists is None:
			flags |= O_CREAT
			if truncate:
				flags |= O_TRUNC
		elif exists:
			if truncate:
				flags |= O_TRUNC
		else:
			flags |= O_CREAT | O_EXCL
		if append:
			flags |= O_APPEND
		fd: FD = fs.open_raw( path.get_cstr(), flags, 0o644 ).or_return()
		return Result.Ok( AsyncBinaryReadWriter._from_fd( fd ))
