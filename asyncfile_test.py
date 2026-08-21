# asyncfile_test.py — real compile+link+run coverage for lib/asyncfile.py's
# AsyncBinaryReader/AsyncBinaryWriter/AsyncBinaryReadWriter and the thread
# pool they're built on.

import unittest
from pathlib import Path

import test_support
from compiler import Compiler
from discovery import Discovery

@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
class AsyncFileTests( test_support.RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _run( self, code: str ) -> None:
		self.compiler.import_code( code, Path( '__main__.py' ), scope = None )
		self.compiler.run()

	def test_async_reader_writer_round_trip_inside_a_reactor( self ) -> None:
		self._run( '''
import compiler
import io
import asyncfile
import reactor
import atomic

class Reader:
	path: str
	flag: atomic.Atomic[i32]
	def __init__( self, path: str, flag: atomic.Atomic[i32] ) -> None:
		self.path = path
		self.flag = flag
	def run( self ) -> None:
		match asyncfile.AsyncFile.binary_reader( self.path ):
			case Result.Ok( r ):
				match r.readline():
					case Result.Ok( line ):
						if line.decode().unwrap( 'decode' ) == 'hello async world\\n':
							self.flag.store( 1 )
						else:
							self.flag.store( 2 )
					case Result.Err( _ ):
						self.flag.store( 3 )
			case Result.Err( _ ):
				self.flag.store( 4 )

def run() -> Result[i32, OSError]:
	path: str = 'asyncfile_test_tmp.bin'
	msg: bytes = b'hello async world\\n'
	w: asyncfile.AsyncBinaryWriter = asyncfile.AsyncFile.binary_writer( path ).or_return()
	io.write_all( w, msg.get_const_ptr(), len( msg )).or_return()
	w.close()

	flag: atomic.Atomic[i32] = atomic.Atomic[i32]( 0 )
	reader = Reader( path, flag )

	r: reactor.Reactor = reactor.Reactor( 1 )
	r.spawn( reader.run )
	r.run()

	if flag.load() != 1:
		return Result.Ok( flag.load() )
	return Result.Ok( 0 )

def main() -> i32:
	match run():
		case Result.Ok( code ):
			return code
		case Result.Err( _ ):
			return 90
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0, timeout = 20 )

	def test_async_reader_without_a_reactor_reads_directly( self ) -> None:
		self._run( '''
import compiler
import io
import asyncfile

def run() -> Result[i32, OSError]:
	path: str = 'asyncfile_test_noreactor_tmp.bin'
	msg: bytes = b'plain blocking path\\n'
	w: asyncfile.AsyncBinaryWriter = asyncfile.AsyncFile.binary_writer( path ).or_return()
	io.write_all( w, msg.get_const_ptr(), len( msg )).or_return()
	w.close()

	r: asyncfile.AsyncBinaryReader = asyncfile.AsyncFile.binary_reader( path ).or_return()
	line: bytearray = r.readline().or_return()
	if line.decode().unwrap( 'decode' ) != 'plain blocking path\\n':
		return Result.Ok( 1 )
	return Result.Ok( 0 )

def main() -> i32:
	match run():
		case Result.Ok( code ):
			return code
		case Result.Err( _ ):
			return 90
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0 )

	def test_pool_dispatches_jobs_with_genuine_concurrency( self ) -> None:
		''' the actual point of lib/asyncfile.py: a slow job submitted to
		the pool must NOT block another concurrently-submitted job from
		making progress at the same time - if the pool secretly serialized
		everything onto one thread, peak concurrency would never exceed 1.
		Tracks how many jobs are simultaneously "in flight" (a busy delay
		standing in for slow disk I/O, since a real local-file read
		completes too fast to time reliably) and asserts the observed peak
		reaches at least half the pool size (4 workers) - not exactly 4,
		since scheduling isn't guaranteed to line up every job perfectly,
		but well beyond what pure serialization (peak == 1) could ever
		produce. '''
		self._run( '''
import compiler
import asyncfile
import reactor
import atomic

class Tracker:
	current: atomic.Atomic[i32]
	peak:    atomic.Atomic[i32]
	def __init__( self ) -> None:
		self.current = atomic.Atomic[i32]( 0 )
		self.peak = atomic.Atomic[i32]( 0 )
	def enter( self ) -> None:
		with compiler.wrap_arithmetic:
			now: i32 = self.current.fetch_add( 1 ) + 1
		p: i32 = self.peak.load()
		if now > p:
			self.peak.store( now )
	def leave( self ) -> None:
		self.current.fetch_sub( 1 )

def _slow_job( tracker: Tracker ) -> Result[usize, OSError]:
	tracker.enter()
	i: usize = 0
	while i < usize( 30000000 ):
		with compiler.wrap_arithmetic:
			i = i + 1
	tracker.leave()
	return Result.Ok( usize( 0 ))

class Task:
	tracker: Tracker
	def __init__( self, tracker: Tracker ) -> None:
		self.tracker = tracker
	def run( self ) -> None:
		w: reactor.Worker|None = reactor.current_worker()
		if w is None:
			return
		handle: reactor.CompletionHandle = reactor.CompletionHandle()
		tracker: Tracker = self.tracker
		work: Closure[[], Result[usize, OSError]] = lambda: _slow_job( tracker )
		job = asyncfile._Job( handle = handle, work = work, waiter = w )
		asyncfile._ensure_pool().submit( job )
		reactor.wait_for_signal( reactor.Signal.Completion( handle )).unwrap( 'wait_for_signal' )

def run() -> i32:
	tracker = Tracker()
	r: reactor.Reactor = reactor.Reactor( 2 )
	n: usize = 0
	while n < usize( 16 ):
		task = Task( tracker )
		r.spawn( task.run )
		with compiler.wrap_arithmetic:
			n += usize( 1 )
	r.run()
	if tracker.peak.load() < 2:
		return 1
	return 0

def main() -> i32:
	return run()
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0, timeout = 20 )

def _emit( compiler: Compiler ) -> str:
	import emitter_c
	return emitter_c.emit_c( compiler )

if __name__ == '__main__':
	unittest.main()
