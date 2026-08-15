# Real-compile-and-run behavioral tests for lib/logging.py. Each program's
# own main() returns 0 on success and a distinct nonzero i32 per failed
# check; RealCompileMixin decodes that back to the failing check. print()
# is deliberately avoided (see time_test.py's identical reasoning) - a
# FileHandler test verifies its own output by reading the file BACK inside
# the same compiled program, not by inspecting captured stdout.
#
# Each test method gets its OWN assert_programs_run() call (not batched
# together into one executable) because lib/logging.py has module-level
# singleton state (the logger registry) - test_support.py's own docstring
# warns against merging programs that depend on process-global one-time
# init into a single executable.

import unittest

import test_support
from test_support import RealCompileMixin

_LEVEL_FILTERING = '''
import logging

def main() -> i32:
	if logging.root().getEffectiveLevel() != logging.WARNING:
		return 1
	if logging.root().isEnabledFor( logging.INFO ):
		return 2
	if not logging.root().isEnabledFor( logging.ERROR ):
		return 3

	custom: logging.Logger = logging.getLogger( 'custom' )
	custom.setLevel( logging.DEBUG )
	if not custom.isEnabledFor( logging.DEBUG ):
		return 4
	return 0
'''

_HIERARCHY = '''
import logging

def main() -> i32:
	a: logging.Logger = logging.getLogger( 'pkg' )
	a.setLevel( logging.DEBUG )

	child: logging.Logger = logging.getLogger( 'pkg.mod' )
	if child.getEffectiveLevel() != logging.DEBUG:
		return 1
	if not child.isEnabledFor( logging.DEBUG ):
		return 2

	grandchild: logging.Logger = logging.getLogger( 'pkg.mod.sub' )
	if grandchild.getEffectiveLevel() != logging.DEBUG:
		return 3

	unrelated: logging.Logger = logging.getLogger( 'other' )
	if unrelated.getEffectiveLevel() != logging.WARNING:
		return 4

	child.setLevel( logging.ERROR )
	if grandchild.getEffectiveLevel() != logging.ERROR:
		return 5
	if a.getEffectiveLevel() != logging.DEBUG:
		return 6
	return 0
'''

_CUSTOM_FORMATTER_AND_FILE_HANDLER = '''
import compiler
import sys
import logging

class TagFormatter( logging.Formatter ):
	@virtual
	def format( self, record: logging.LogRecord ) -> str:
		return f'[{record.name}] {record.message}'

def main() -> i32:
	path: str = "logging_test_output.txt"

	r = logging.FileHandler( path )
	if r.is_err():
		return 1
	handler: logging.FileHandler = r.unwrap( 'open failed' )
	handler.setFormatter( TagFormatter() )
	handler.setLevel( logging.INFO )

	logger: logging.Logger = logging.getLogger( 'filetest' )
	logger.setLevel( logging.DEBUG )
	logger.propagate = False
	logger.addHandler( handler )

	logger.debug( 'filtered out by the HANDLER level' )
	logger.info( 'hello from metalpy' )
	handler.close()

	rr = File.binary_reader( path )
	if rr.is_err():
		return 2
	reader = rr.unwrap( 'reopen failed' )
	buf: Ptr[u8] = sys.alloc[u8]( 256 )
	n = reader.read( buf, 255 )
	if n.is_err():
		return 3
	reader.close()
	read_len: usize = n.unwrap( 'read failed' )
	buf[read_len] = 0
	with compiler.panic_arithmetic( 'bounded by the 255-byte read cap above' ):
		buf_size: usize = read_len + 1
	got_r = str.from_cstr( compiler.cast( ConstPtr[u8], buf ), buf_size )
	if got_r.is_err():
		return 4
	got: str = got_r.unwrap( 'utf8 decode failed' )
	sys.free( buf )

	expected: str = "[filetest] hello from metalpy\\n"
	if got != expected:
		return 5
	return 0
'''

_PROPAGATION = '''
import compiler
import logging

class CountingHandler( logging.Handler ):
	count: i32

	def __init__( self ) -> None:
		super().__init__()
		self.count = 0

	@virtual
	def emit( self, record: logging.LogRecord ) -> Result[None, OSError]:
		with compiler.wrap_arithmetic:
			self.count += 1
		return Result.Ok( None )

def main() -> i32:
	parent_handler: CountingHandler = CountingHandler()
	parent: logging.Logger = logging.getLogger( 'propparent' )
	parent.setLevel( logging.DEBUG )
	parent.addHandler( parent_handler )

	child: logging.Logger = logging.getLogger( 'propparent.child' )
	child.info( 'reaches the parent handler' )
	if parent_handler.count != 1:
		return 1

	child.propagate = False
	child_handler: CountingHandler = CountingHandler()
	child.addHandler( child_handler )
	child.info( 'stays local now' )
	if parent_handler.count != 1:
		return 2
	if child_handler.count != 1:
		return 3
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile logging tests' )
class LoggingBehaviorTests( RealCompileMixin, unittest.TestCase ):
	def test_level_filtering( self ) -> None:
		self.assert_programs_run([ ( 'level_filtering', _LEVEL_FILTERING ) ])

	def test_dotted_name_hierarchy( self ) -> None:
		self.assert_programs_run([ ( 'hierarchy', _HIERARCHY ) ])

	def test_custom_formatter_and_file_handler( self ) -> None:
		self.assert_programs_run([ ( 'custom_formatter_and_file_handler', _CUSTOM_FORMATTER_AND_FILE_HANDLER ) ])

	def test_propagation( self ) -> None:
		self.assert_programs_run([ ( 'propagation', _PROPAGATION ) ])


if __name__ == '__main__':
	unittest.main()
