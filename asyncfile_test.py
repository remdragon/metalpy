# asyncfile_test.py — real compile+link+run coverage for lib/asyncfile.py:
# AsyncFile's factories inject this module's FileOpsInterface implementation
# into an ordinary BinaryReader/BinaryWriter/BinaryReadWriter (lib/builtins/
# __File.py), plus the thread pool that implementation is built on.

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
	w: BinaryWriter = asyncfile.AsyncFile.binary_writer( path ).or_return()
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

	def test_plain_file_reader_stays_unpooled_inside_a_reactor( self ) -> None:
		''' the injection is opt-in, not automatic: a BinaryReader obtained
		via plain File.binary_reader() (never touching lib/asyncfile.py at
		all) must keep calling read_raw() directly even when constructed
		and used from inside a Reactor-driven fiber - only AsyncFile's own
		factories inject the pool-backed FileOpsInterface. '''
		self._run( '''
import compiler
import reactor
import atomic

class Reader:
	path: str
	flag: atomic.Atomic[i32]
	def __init__( self, path: str, flag: atomic.Atomic[i32] ) -> None:
		self.path = path
		self.flag = flag
	def run( self ) -> None:
		match File.binary_reader( self.path ):
			case Result.Ok( r ):
				match r.readline():
					case Result.Ok( line ):
						if line.decode().unwrap( 'decode' ) == 'plain inside reactor\\n':
							self.flag.store( 1 )
						else:
							self.flag.store( 2 )
					case Result.Err( _ ):
						self.flag.store( 3 )
			case Result.Err( _ ):
				self.flag.store( 4 )

def run() -> Result[i32, OSError]:
	path: str = 'asyncfile_test_plain_tmp.bin'
	msg: bytes = b'plain inside reactor\\n'
	w: BinaryWriter = File.binary_writer( path ).or_return()
	n: usize = w.write( msg.get_const_ptr(), len( msg )).or_return()
	if n != len( msg ):
		return Result.Ok( 5 )
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
	w: BinaryWriter = asyncfile.AsyncFile.binary_writer( path ).or_return()
	io.write_all( w, msg.get_const_ptr(), len( msg )).or_return()
	w.close()

	r: BinaryReader = asyncfile.AsyncFile.binary_reader( path ).or_return()
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
		job = asyncfile.Job( handle = handle, work = work, waiter = w )
		asyncfile.pool.submit( job )
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

	def test_pool_worker_drains_backlog_in_fifo_order( self ) -> None:
		''' regression test for a real bug: _PoolWorker used to drain via
		list.pop() (LAST element first, i.e. LIFO), so a job could be
		starved behind a stream of newer submissions on the same worker.
		Blocks worker 0 on a gate, queues a backlog of 4 tracked jobs onto
		that SAME worker (submissions round-robin across the pool's 4
		workers, so every 4th submission lands back on worker 0 - filler
		jobs sent to the other 3 workers between each tracked one just
		advance the round-robin ticket), releases the gate, and asserts
		the 4 tracked jobs ran in submission order. '''
		self._run( '''
import compiler
import atomic
import asyncfile
import reactor

class Gate:
	started: atomic.Atomic[i32]
	release: atomic.Atomic[bool]
	def __init__( self ) -> None:
		self.started = atomic.Atomic[i32]( 0 )
		self.release = atomic.Atomic[bool]( False )
	def blocking( self ) -> Result[usize, OSError]:
		self.started.fetch_add( 1 )
		while not self.release.load():
			pass
		return Result.Ok( usize( 0 ))

class Recorder:
	order: list[usize]
	def __init__( self ) -> None:
		self.order = list[usize]()
	def record( self, n: usize ) -> None:
		self.order.append( n )

class RealJob:
	idx: usize
	rec: Recorder
	def __init__( self, idx: usize, rec: Recorder ) -> None:
		self.idx = idx
		self.rec = rec
	def run( self ) -> Result[usize, OSError]:
		self.rec.record( self.idx )
		return Result.Ok( usize( 0 ))

def busy_delay() -> None:
	i: usize = 0
	while i < usize( 200000000 ):
		with compiler.wrap_arithmetic:
			i = i + 1

class Runner:
	gate: Gate
	rec:  Recorder
	def __init__( self, gate: Gate, rec: Recorder ) -> None:
		self.gate = gate
		self.rec = rec
	def run( self ) -> None:
		w: reactor.Worker|None = reactor.current_worker()
		if w is None:
			return
		gate: Gate = self.gate
		rec: Recorder = self.rec

		# occupies worker 0 (the first submission, ticket 0) - parks it on
		# the gate so a real backlog can build up behind it
		gate_work: Closure[[], Result[usize, OSError]] = gate.blocking
		asyncfile.pool.submit( asyncfile.Job( handle = reactor.CompletionHandle(), work = gate_work, waiter = w ))
		busy_delay()
		if gate.started.load() != 1:
			return

		# 17 more submissions: ticket%4==0 (mod the ALREADY-consumed ticket
		# 0) lands back on worker 0 - a real tracked job every 4th
		# submission, filler jobs (any other worker) in between
		next_real: usize = 0
		i: usize = 0
		while i < usize( 16 ):
			with compiler.panic_arithmetic( 'test: modulus is nonzero by construction' ):
				rem: usize = i % usize( 4 )
			if rem == usize( 3 ):   # tickets 4, 8, 12, 16 -> worker 0
				job: RealJob = RealJob( next_real, rec )
				real_work: Closure[[], Result[usize, OSError]] = job.run
				asyncfile.pool.submit( asyncfile.Job( handle = reactor.CompletionHandle(), work = real_work, waiter = w ))
				with compiler.wrap_arithmetic:
					next_real = next_real + usize( 1 )
			else:
				# captures `rec` (unused by the lambda body otherwise) so
				# this is a real closure, not a captureless function
				# pointer - Job.work's own field type requires Closure
				filler_work: Closure[[], Result[usize, OSError]] = lambda: Result.Ok( usize( rec.order.__len__()))
				asyncfile.pool.submit( asyncfile.Job( handle = reactor.CompletionHandle(), work = filler_work, waiter = w ))
			with compiler.wrap_arithmetic:
				i = i + 1

		gate.release.store( True )
		while rec.order.__len__() < usize( 4 ):
			pass

def run() -> i32:
	gate = Gate()
	rec = Recorder()
	r: reactor.Reactor = reactor.Reactor( 1 )
	runner = Runner( gate, rec )
	r.spawn( runner.run )
	r.run()

	if rec.order.__len__() != usize( 4 ):
		return 1
	i: usize = 0
	while i < usize( 4 ):
		got: usize = rec.order.__getitem__( i ).unwrap( 'order index in bounds by construction' )
		if got != i:
			return 2   # out of submission order - LIFO regression
		with compiler.wrap_arithmetic:
			i = i + 1
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
