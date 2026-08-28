import unittest

import test_support
from compiler import Compiler
from discovery import Discovery


class RandomTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' Real compile-and-run coverage for lib/random.py's CSPRNG (Windows:
	BCryptGenRandom; Linux: getrandom(2)/dev/urandom fallback). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'random_bytes_returns_requested_length', '''
import random

def main() -> i32:
	b: bytearray = random.random_bytes( usize( 32 )).unwrap( 'random_bytes' )
	if len( b ) != usize( 32 ):
		return 1
	return 0
''' ),
			( 'repeated_calls_differ', '''
import random

def bytearrays_equal( a: bytearray, b: bytearray ) -> bool:
	if len( a ) != len( b ):
		return False
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < len( a ):
			if a.__getitem__( i ).unwrap( 'a[i]' ) != b.__getitem__( i ).unwrap( 'b[i]' ):
				return False
			i += usize( 1 )
	return True

def main() -> i32:
	# real correctness check, not just "it compiled" - two independent
	# 32-byte draws colliding would be astronomically unlikely (2^-256) if
	# this is genuinely pulling from the OS CSPRNG rather than returning a
	# fixed/zeroed buffer.
	a: bytearray = random.random_bytes( usize( 32 )).unwrap( 'a' )
	b: bytearray = random.random_bytes( usize( 32 )).unwrap( 'b' )
	if bytearrays_equal( a, b ):
		return 1
	return 0
''' ),
			( 'not_all_zero', '''
import random

def main() -> i32:
	b: bytearray = random.random_bytes( usize( 64 )).unwrap( 'random_bytes' )
	all_zero: bool = True
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < usize( 64 ):
			if b.__getitem__( i ).unwrap( 'b[i]' ) != u8( 0 ):
				all_zero = False
			i += usize( 1 )
	if all_zero:
		return 1
	return 0
''' ),
			( 'many_draws_all_distinct', '''
import random

def bytearrays_equal( a: bytearray, b: bytearray ) -> bool:
	if len( a ) != len( b ):
		return False
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < len( a ):
			if a.__getitem__( i ).unwrap( 'a[i]' ) != b.__getitem__( i ).unwrap( 'b[i]' ):
				return False
			i += usize( 1 )
	return True

def main() -> i32:
	# 20 independent 16-byte draws, all pairwise distinct - would catch a
	# broken RNG that cycles or repeats after a handful of calls (e.g. an
	# unseeded/stuck source), which a single before/after pair could miss.
	draws: list[bytearray] = list[bytearray]()
	n: usize = 0
	with compiler.wrap_arithmetic:
		while n < usize( 20 ):
			draws.append( random.random_bytes( usize( 16 )).unwrap( 'draw' ))
			n += usize( 1 )
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < usize( 20 ):
			j: usize = i + usize( 1 )
			while j < usize( 20 ):
				if bytearrays_equal( draws.__getitem__( i ).unwrap( 'draws[i]' ), draws.__getitem__( j ).unwrap( 'draws[j]' )):
					return 1
				with compiler.wrap_arithmetic:
					j += usize( 1 )
			i += usize( 1 )
	return 0
''' ),
			( 'fill_random_into_existing_buffer', '''
import random

def main() -> i32:
	buf: bytearray = bytearray( usize( 16 ))
	random.fill_random( buf.get_ptr(), usize( 16 )).unwrap( 'fill_random' )
	all_zero: bool = True
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < usize( 16 ):
			if buf.__getitem__( i ).unwrap( 'buf[i]' ) != u8( 0 ):
				all_zero = False
			i += usize( 1 )
	if all_zero:
		return 1
	return 0
''' ),
			( 'zero_length_ok', '''
import random

def main() -> i32:
	b: bytearray = random.random_bytes( usize( 0 )).unwrap( 'zero' )
	if len( b ) != usize( 0 ):
		return 1
	return 0
''' ),
		] )
