# Real-compile-and-run tests for lib/queue.py's Queue[T] (moved out of
# lib/threading.py to match Python's own queue module location - previously
# only exercised indirectly via threading.ThreadPool/lib/asyncfile.py/
# lib/sys.py's threaded stdout).

# stdlib imports:
import unittest

# local imports:
import test_support


class QueueTests( test_support.RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_put_and_drain_is_fifo( self ) -> None:
		import emitter_c
		self.assert_programs_run([
			( 'fifo_order_preserved', '''
import queue

def main() -> i32:
	q: queue.Queue[i32] = queue.Queue[i32]()
	q.put( 1 ).unwrap( 'put should succeed' )
	q.put( 2 ).unwrap( 'put should succeed' )
	q.put( 3 ).unwrap( 'put should succeed' )
	batch: UnsafeList[i32] = q.drain()
	if batch.__len__() != 3:
		return 1
	if batch.__getitem__( 0 ).unwrap( 'in bounds' ) != 1:
		return 2
	if batch.__getitem__( 1 ).unwrap( 'in bounds' ) != 2:
		return 3
	if batch.__getitem__( 2 ).unwrap( 'in bounds' ) != 3:
		return 4
	return 0
''' ),
			( 'max_depth_rejects_once_full', '''
import queue

def main() -> i32:
	q: queue.Queue[i32] = queue.Queue[i32]( usize( 2 ) )
	q.put( 1 ).unwrap( 'first put under max_depth should succeed' )
	q.put( 2 ).unwrap( 'second put under max_depth should succeed' )
	match q.put( 3 ):
		case Result.Ok( _ ):
			return 1
		case Result.Err( _ ):
			pass
	batch: UnsafeList[i32] = q.drain()
	if batch.__len__() != 2:
		return 2
	return 0
''' ),
		])


if __name__ == '__main__':
	unittest.main()
