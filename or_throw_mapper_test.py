# Real-compile-and-run tests for Result.or_throw(mapper): like or_throw()
# (dispatches the error into an enclosing try's own handler, propagating like
# or_return() when uncovered), but converts the error via mapper(err) first -
# see or_return_mapper_test.py's own docstring for the general mapper-
# handling machinery (shared, via _lower_or_return_with_mapper/
# _lower_or_throw_with_mapper's near-identical shape).
#
# Building this surfaced a real, pre-existing, generator-unrelated compiler
# bug in `raise EXPR` itself (not or_throw()-specific, not RC-tracking-only):
# _stmt_Raise unconditionally released the raised value via the ordinary
# end-of-statement pending-temp flush BEFORE the dispatch below ever read it
# (assigning it into a covered leaf's `except X as e:` bind, or widening it
# into a propagated Result) - a real use-after-free, not just a leak:
# `raise Boom(code=5)` / `except Boom as e: ...e.code...` silently read
# freed memory and returned the wrong value. Fixed by excluding the raised
# value from that flush (cfg.untrack_temp) and letting its ownership
# transfer via the dispatch, same as any other move - see _stmt_Raise's own
# comment. A NAMED local raised from inside or_throw(mapper)'s own confined
# Err branch has no equivalent automatic unwind (a goto-based `raise` never
# revisits branch-confined state), so the mapper's own result is kept a
# bare, never-named temp instead (see _lower_or_throw_with_mapper/
# _raise_value's own comments) - the alternative (a hidden named local,
# manually cancelled via cfg.manually_decreffed()) was confirmed NOT to
# work: cancelling a release that would never fire anyway changes nothing.

import unittest

import test_support
from test_support import RealCompileMixin

_PLAIN_FUNCTION_MAPPER_DISPATCHED_INTO_HANDLER = '''
def probe( i: usize ) -> Result[i32, IndexError]:
	if i == 0:
		return Result.Ok( 42 )
	return Result.Err( IndexError() )

def to_stop( e: IndexError ) -> StopIteration:
	return StopIteration()

def inner( i: usize ) -> Result[i32, StopIteration]:
	try:
		v: i32 = probe( i ).or_throw( to_stop )
	except StopIteration as e:
		compiler.decref( e )
		return Result.Ok( -1 )
	return Result.Ok( v )

def main() -> i32:
	match inner( 0 ):
		case Result.Ok( v ):
			if v != 42:
				return 1
		case Result.Err( _ ):
			return 2
	match inner( 1 ):
		case Result.Ok( v ):
			if v != -1:
				return 3
		case Result.Err( _ ):
			return 4
	return 0
'''

# the mapped payload itself must be READABLE (not freed/corrupted) by the
# handler - the exact shape the raise/except-as-name UAF bug above was
# found through.
_MAPPED_PAYLOAD_FIELD_IS_READABLE_IN_HANDLER = '''
class ParseError:
	code: i32

def probe( i: usize ) -> Result[i32, IndexError]:
	if i == 0:
		return Result.Ok( 42 )
	return Result.Err( IndexError() )

def to_parse_error( e: IndexError ) -> ParseError:
	return ParseError( code = 99 )

def inner( i: usize ) -> Result[i32, ParseError]:
	try:
		v: i32 = probe( i ).or_throw( to_parse_error )
	except ParseError as e:
		code: i32 = e.code
		compiler.decref( e )
		if code != 99:
			return Result.Ok( -2 )
		return Result.Ok( -1 )
	return Result.Ok( v )

def main() -> i32:
	match inner( 1 ):
		case Result.Ok( v ):
			if v != -1:
				return 1
		case Result.Err( _ ):
			return 2
	return 0
'''

_UNCOVERED_MAPPER_PROPAGATES_LIKE_OR_RETURN = '''
def probe( i: usize ) -> Result[i32, IndexError]:
	if i == 0:
		return Result.Ok( 42 )
	return Result.Err( IndexError() )

def to_stop( e: IndexError ) -> StopIteration:
	return StopIteration()

def inner( i: usize ) -> Result[i32, StopIteration]:
	v: i32 = probe( i ).or_throw( to_stop )
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
	try:
		v: i32 = probe( i ).or_throw( lambda e: StopIteration() )
	except StopIteration as e:
		compiler.decref( e )
		return Result.Ok( -1 )
	return Result.Ok( v )

def main() -> i32:
	match inner( 1 ):
		case Result.Ok( v ):
			if v != -1:
				return 1
		case Result.Err( _ ):
			return 2
	return 0
'''

# a REAL capturing closure, same RC-correctness check or_return(mapper)'s
# own test uses - confirms the mapper's own StopIteration() result isn't
# leaked/double-freed and the extracted IndexError payload is handled once.
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

	try:
		w: Elem = probe( i, e ).or_throw( to_stop )
	except StopIteration as ex:
		compiler.decref( ex )
		return Result.Ok( Elem() ) # deliberately NOT `e` - this path must not touch e's own refcount at all
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
			pass
		case Result.Err( _ ):
			return 3
	if compiler.refcount( e ) != before:
		return 4
	return 0
'''

_ERRDEFER_STILL_FIRES_ON_UNCOVERED_PROPAGATION = '''
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
	v: i32 = probe( i ).or_throw( to_stop )
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
		return 4  # errdefer MUST fire exactly once on the uncovered/propagating path
	compiler.decref( g )
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
class OrThrowMapperTests( RealCompileMixin, unittest.TestCase ):
	def test_plain_function_mapper_dispatched_into_handler( self ) -> None:
		self.assert_programs_run([ ( 'or_throw_mapper_plain_function', _PLAIN_FUNCTION_MAPPER_DISPATCHED_INTO_HANDLER ) ])

	def test_mapped_payload_field_is_readable_in_handler( self ) -> None:
		self.assert_programs_run([ ( 'or_throw_mapper_field_readable', _MAPPED_PAYLOAD_FIELD_IS_READABLE_IN_HANDLER ) ])

	def test_uncovered_mapper_propagates_like_or_return( self ) -> None:
		self.assert_programs_run([ ( 'or_throw_mapper_uncovered', _UNCOVERED_MAPPER_PROPAGATES_LIKE_OR_RETURN ) ])

	def test_non_capturing_lambda_mapper( self ) -> None:
		self.assert_programs_run([ ( 'or_throw_mapper_lambda', _NON_CAPTURING_LAMBDA_MAPPER ) ])

	def test_capturing_closure_mapper_rc_correct( self ) -> None:
		self.assert_programs_run([ ( 'or_throw_mapper_closure_rc', _CAPTURING_CLOSURE_MAPPER_RC_CORRECT ) ])

	def test_errdefer_still_fires_on_uncovered_propagation( self ) -> None:
		self.assert_programs_run([ ( 'or_throw_mapper_errdefer', _ERRDEFER_STILL_FIRES_ON_UNCOVERED_PROPAGATION ) ])


class OrThrowMapperRejectionTests( unittest.TestCase ):
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
	v: i32 = probe( 1 ).or_throw( to_stop )
	yield v

def main() -> i32:
	g = gen()
	return 0
''', __import__( 'pathlib' ).Path( '__main__.py' ), scope = None )
		self.compiler.run()
		self.assertTrue( any( 'or_throw(mapper) is not supported inside a generator body yet' in str( e ) for e in self.discovery.errors.errors ) )


if __name__ == '__main__':
	unittest.main()
