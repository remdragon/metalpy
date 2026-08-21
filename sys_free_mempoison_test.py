# Real-compile-and-run regression tests for sys.free()'s debug-only
# mempoison-before-free (lib/sys.py) - added so a use-after-free reads
# 0xCD poison rather than whatever the allocator happened to leave behind,
# on every compiler (not just ones whose own debug heap already does
# this). Needs the real allocation SIZE at the point of free (sys.free()
# only ever receives the raw pointer, not a remembered size), queried via
# a platform primitive (windows.kernel32.HeapSize / crt.malloc_usable_size,
# os-split in crt.py the same way __error/__errno_location already are) -
# these tests exercise that query + mempoison in the exact sequence
# sys.free() itself now uses, immediately BEFORE the real free (so the
# result isn't at the mercy of whatever the underlying allocator does to
# the block's content the instant it's actually freed, which is its own
# implementation detail, not something this feature controls or needs to).
#
# compiler.target(os=...) picks the right branch for whichever host this
# test suite is actually compiled on - there is no portable way to name
# HeapSize/malloc_usable_size directly from a single code path.

import unittest

import test_support
from test_support import RealCompileMixin

_MEMPOISON_BEFORE_FREE_SIZES_THE_BLOCK_CORRECTLY = '''
import sys

@compiler.target( os = "windows" )
def block_size( p: Ptr[u8] ) -> usize:
	from windows.kernel32 import GetProcessHeap, HeapSize, HEAP_SIZE_FAILED
	size: usize = HeapSize( GetProcessHeap(), 0, p )
	if size == HEAP_SIZE_FAILED:
		sys.panic( "HeapSize failed on a live allocation" )
	return size

@compiler.target( os = not "windows" )
def block_size( p: Ptr[u8] ) -> usize:
	from crt import malloc_usable_size
	return malloc_usable_size( p )

def main() -> i32:
	p: Ptr[u8] = sys.alloc[u8]( 16 )
	p[0] = 42
	p[15] = 42

	# the real usable size must cover (at least) what was actually
	# requested - an allocator is free to round up, never down
	size: usize = block_size( p )
	if size < 16:
		return 1

	# the exact sequence sys.free()'s own debug branch runs right before
	# the real free - poisons the WHOLE live block (not just the
	# originally-requested 16 bytes), matching sys.free()'s own use of
	# the queried size rather than a remembered request size
	sys.mempoison( p, size )
	if p[0] != 0xCD:
		return 2
	if p[15] != 0xCD:
		return 3

	sys.free( p )
	return 0
'''

_SYS_FREE_STILL_COMPILES_AND_RUNS_IN_RELEASE = '''
import sys

def main() -> i32:
	# release mode skips mempoison entirely (compiler.target.debug is
	# False) - this just needs to compile and run cleanly either way, on
	# both the debug default and --release (test_support.py only builds
	# debug, but this is still worth pinning down as a real compile+run,
	# not just a static assertion, since sys.free()'s debug branch is now
	# gated by a real runtime `if`, not a compiler.target()-selected
	# overload - a bug there would otherwise only show up under --release)
	p: Ptr[u8] = sys.alloc[u8]( 8 )
	p[0] = 7
	sys.free( p )
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class SysFreeMempoisonTests( RealCompileMixin, unittest.TestCase ):
	def test_mempoison_before_free_sizes_the_block_correctly( self ) -> None:
		self.assert_programs_run([ ( 'mempoison_before_free_sizes_the_block_correctly', _MEMPOISON_BEFORE_FREE_SIZES_THE_BLOCK_CORRECTLY ) ])

	def test_sys_free_still_compiles_and_runs( self ) -> None:
		self.assert_programs_run([ ( 'sys_free_still_compiles_and_runs', _SYS_FREE_STILL_COMPILES_AND_RUNS_IN_RELEASE ) ])


if __name__ == '__main__':
	unittest.main()
