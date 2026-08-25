# Real-compile-and-run tests for Result.or_return(mapper): converts one
# error type into another on the Err path (mapper(err) instead of err
# unchanged), while leaving the no-arg form's own behavior untouched on the
# Ok path. See PLAN_SEQUENCE_ITER_FOLLOWUPS.md item 3 for the design.
#
# Building this surfaced two real, pre-existing, generator-unrelated
# compiler bugs, both fixed alongside this feature (not specific to it):
#   1. lowering.py's Lowering._emit only auto-registered ir.Call/ir.Allocate
#      results as fresh_temp - ir.CallIndirect (a real closure/indirect-call
#      result) was missing, so a closure call's own return value, passed
#      directly into a retaining constructor (Result.Err(some_closure())),
#      leaked one reference every time - reproduces with a bare
#      `return Result.Err(closure())`, no or_return() involved at all.
#   2. _stmt_Return's inline-unwind path (taken whenever the topmost pending
#      epilogue entry is confined to the current branch/loop, forcing
#      current_epilogue_label() to return None) never populated
#      self._return_value_var before replaying pending defer/errdefer
#      entries - errdefer's own is_err() check silently read a stale value
#      and never fired. Reproduces with ordinary hand-written code (an
#      RC-typed local declared inside an if-branch, followed by an
#      errdefer-covered early return) - or_return(mapper) hits this reliably
#      because its own error-payload extraction is itself always such a
#      confined entry.

import unittest

import test_support
from test_support import RealCompileMixin

_PLAIN_FUNCTION_MAPPER = '''
def probe( i: usize ) -> Result[i32, IndexError]:
	if i == 0:
		return Result.Ok( 42 )
	return Result.Err( IndexError() )

def to_stop( e: IndexError ) -> StopIteration:
	return StopIteration()

def inner( i: usize ) -> Result[i32, StopIteration]:
	v: i32 = probe( i ).or_return( to_stop )
	return Result.Ok( v )

def main() -> i32:
	match inner( 0 ):
		case Result.Ok( v ):
			if v != 42:
				return 1
		case Result.Err( _ ):
			return 2
	match inner( 1 ):
		case Result.Ok( _ ):
			return 3
		case Result.Err( _ ):
			pass
	return 0
'''

_NON_CAPTURING_LAMBDA_MAPPER = '''
def probe( i: usize ) -> Result[i32, IndexError]:
	if i == 0:
		return Result.Ok( 42 )
	return Result.Err( IndexError() )

def inner( i: usize ) -> Result[i32, StopIteration]:
	v: i32 = probe( i ).or_return( lambda e: StopIteration() )
	return Result.Ok( v )

def main() -> i32:
	match inner( 1 ):
		case Result.Ok( _ ):
			return 1
		case Result.Err( _ ):
			return 0
'''

# a REAL capturing closure - reads `scale` from the enclosing scope. Also
# checks compiler.refcount() before/after to confirm the mapper's own
# StopIteration() result isn't leaked (bug 1 above) and the extracted
# IndexError payload isn't double-freed.
_CAPTURING_CLOSURE_MAPPER_RC_CORRECT = '''
class Elem:
	pass

def probe( i: usize, e: Elem ) -> Result[Elem, IndexError]:
	if i == 0:
		return Result.Ok( e )
	return Result.Err( IndexError() )

def inner( i: usize, e: Elem, scale: i32 ) -> Result[Elem, StopIteration]:
	def to_stop( err: IndexError ) -> StopIteration:
		with compiler.wrap_arithmetic:
			doubled: i32 = scale + scale
		if doubled < 0:
			return StopIteration()
		return StopIteration()

	w: Elem = probe( i, e ).or_return( to_stop )
	return Result.Ok( w )

def main() -> i32:
	e: Elem = Elem()
	before: usize = compiler.refcount( e )
	match inner( 0, e, 3 ):
		case Result.Ok( _ ):
			pass
		case Result.Err( _ ):
			return 1
	if compiler.refcount( e ) != before:
		return 2

	match inner( 1, e, 3 ):
		case Result.Ok( _ ):
			return 3
		case Result.Err( _ ):
			pass
	if compiler.refcount( e ) != before:
		return 4
	return 0
'''

# errdefer's own is_err() check must still fire correctly even though
# or_return(mapper)'s own error-payload extraction is itself always a
# branch-confined entry (bug 2 above) - verified by observing the
# refcount side effect errdefer performs, not just a print.
_ERRDEFER_STILL_FIRES = '''
class Guard:
	fired: i32

def probe( i: usize ) -> Result[i32, IndexError]:
	if i == 0:
		return Result.Ok( 42 )
	return Result.Err( IndexError() )

def to_stop( e: IndexError ) -> StopIteration:
	return StopIteration()

def inner( i: usize, g: Guard ) -> Result[i32, StopIteration]:
	errdefer( compiler.incref( g ))
	v: i32 = probe( i ).or_return( to_stop )
	return Result.Ok( v )

def main() -> i32:
	g: Guard = Guard( fired = 0 )
	before: usize = compiler.refcount( g )
	match inner( 0, g ):
		case Result.Ok( _ ):
			pass
		case Result.Err( _ ):
			return 1
	if compiler.refcount( g ) != before:
		return 2  # errdefer must NOT fire on the Ok path

	match inner( 1, g ):
		case Result.Ok( _ ):
			return 3
		case Result.Err( _ ):
			pass
	with compiler.wrap_arithmetic:
		expected: usize = before + 1
	if compiler.refcount( g ) != expected:
		return 4  # errdefer MUST fire exactly once on the Err path
	compiler.decref( g )
	return 0
'''

_DEFER_STILL_FIRES_ON_BOTH_PATHS = '''
class Guard:
	fired: i32

def probe( i: usize ) -> Result[i32, IndexError]:
	if i == 0:
		return Result.Ok( 42 )
	return Result.Err( IndexError() )

def to_stop( e: IndexError ) -> StopIteration:
	return StopIteration()

def inner( i: usize, g: Guard ) -> Result[i32, StopIteration]:
	defer( compiler.incref( g ))
	v: i32 = probe( i ).or_return( to_stop )
	return Result.Ok( v )

def main() -> i32:
	g: Guard = Guard( fired = 0 )
	before: usize = compiler.refcount( g )
	with compiler.wrap_arithmetic:
		expected: usize = before + 1
	match inner( 0, g ):
		case Result.Ok( _ ):
			pass
		case Result.Err( _ ):
			return 1
	if compiler.refcount( g ) != expected:
		return 2
	compiler.decref( g )

	match inner( 1, g ):
		case Result.Ok( _ ):
			return 3
		case Result.Err( _ ):
			pass
	if compiler.refcount( g ) != expected:
		return 4
	compiler.decref( g )
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
class OrReturnMapperTests( RealCompileMixin, unittest.TestCase ):
	def test_plain_function_mapper( self ) -> None:
		self.assert_programs_run([ ( 'or_return_mapper_plain_function', _PLAIN_FUNCTION_MAPPER ) ])

	def test_non_capturing_lambda_mapper( self ) -> None:
		self.assert_programs_run([ ( 'or_return_mapper_lambda', _NON_CAPTURING_LAMBDA_MAPPER ) ])

	def test_capturing_closure_mapper_rc_correct( self ) -> None:
		self.assert_programs_run([ ( 'or_return_mapper_closure_rc', _CAPTURING_CLOSURE_MAPPER_RC_CORRECT ) ])

	def test_errdefer_still_fires_around_mapper( self ) -> None:
		self.assert_programs_run([ ( 'or_return_mapper_errdefer', _ERRDEFER_STILL_FIRES ) ])

	def test_defer_still_fires_around_mapper( self ) -> None:
		self.assert_programs_run([ ( 'or_return_mapper_defer', _DEFER_STILL_FIRES_ON_BOTH_PATHS ) ])


class OrReturnMapperRejectionTests( unittest.TestCase ):
	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def test_rejected_inside_generator_body( self ) -> None:
		self.compiler.import_code( '''
def probe( i: usize ) -> Result[i32, IndexError]:
	if i == 0:
		return Result.Ok( 42 )
	return Result.Err( IndexError() )

def to_stop( e: IndexError ) -> StopIteration:
	return StopIteration()

def gen() -> Generator[i32, StopIteration]:
	v: i32 = probe( 1 ).or_return( to_stop )
	yield v

def main() -> i32:
	g = gen()
	return 0
''', __import__( 'pathlib' ).Path( '__main__.py' ), scope = None )
		self.compiler.run()
		self.assertTrue( any( 'or_return(mapper) is not supported inside a generator body yet' in str( e ) for e in self.discovery.errors.errors ) )


if __name__ == '__main__':
	unittest.main()
