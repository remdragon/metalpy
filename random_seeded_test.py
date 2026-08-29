# Real-compile-and-run tests for lib/random.py's seedable Random class +
# module-level convenience API (random.seed/random/uniform/randint/choice/
# shuffle) - the deterministic xoshiro256**/SplitMix64 PRNG, distinct from
# the CSPRNG covered by random_test.py.
#
# match/case on the Result[T,IndexError] returned by choice() is avoided
# throughout - it hits a confirmed, separate compiler bug (method-scoped
# generic T inside a matched Result[T,E] crashes emission); .unwrap()/
# .is_ok()/.is_err() work fine and are used instead.

import unittest

import test_support
from compiler import Compiler
from discovery import Discovery


class RandomSeededTests( test_support.RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'same_seed_same_sequence', '''
import random

def main() -> i32:
	a: random.Random = random.Random( u64( 12345 ))
	b: random.Random = random.Random( u64( 12345 ))
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < usize( 50 ):
			if a.random() != b.random():
				return 1
			if a.randint( i64( -1000 ), i64( 1000 )) != b.randint( i64( -1000 ), i64( 1000 )):
				return 2
			i += usize( 1 )
	return 0
''' ),
			( 'different_seeds_diverge', '''
import random

def main() -> i32:
	a: random.Random = random.Random( u64( 1 ))
	b: random.Random = random.Random( u64( 2 ))
	if a.random() == b.random():
		return 1
	return 0
''' ),
			( 'random_stays_in_zero_one_range', '''
import random

def main() -> i32:
	r: random.Random = random.Random( u64( 7 ))
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < usize( 1000 ):
			v: f64 = r.random()
			if v < 0.0 or v >= 1.0:
				return 1
			i += usize( 1 )
	return 0
''' ),
			( 'randint_bounds_inclusive_both_ends', '''
import random

def main() -> i32:
	r: random.Random = random.Random( u64( 3 ))
	saw_low: bool = False
	saw_high: bool = False
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < usize( 500 ):
			n: i64 = r.randint( i64( 1 ), i64( 3 ))
			if n < i64( 1 ) or n > i64( 3 ):
				return 1
			if n == i64( 1 ):
				saw_low = True
			if n == i64( 3 ):
				saw_high = True
			i += usize( 1 )
	# both endpoints should show up at least once in 500 draws over a
	# 3-value range - would catch an off-by-one that silently excludes an
	# endpoint (e.g. a half-open range masquerading as inclusive).
	if not saw_low or not saw_high:
		return 2
	return 0
''' ),
			( 'randint_negative_range', '''
import random

def main() -> i32:
	r: random.Random = random.Random( u64( 9 ))
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < usize( 200 ):
			n: i64 = r.randint( i64( -10 ), i64( -5 ))
			if n < i64( -10 ) or n > i64( -5 ):
				return 1
			i += usize( 1 )
	return 0
''' ),
			( 'randint_single_value_range', '''
import random

def main() -> i32:
	r: random.Random = random.Random( u64( 11 ))
	if r.randint( i64( 42 ), i64( 42 )) != i64( 42 ):
		return 1
	return 0
''' ),
			( 'uniform_stays_in_range', '''
import random

def main() -> i32:
	r: random.Random = random.Random( u64( 21 ))
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < usize( 200 ):
			v: f64 = r.uniform( 10.0, 20.0 )
			if v < 10.0 or v > 20.0:
				return 1
			i += usize( 1 )
	return 0
''' ),
			( 'shuffle_preserves_multiset', '''
import random

def main() -> i32:
	r: random.Random = random.Random( u64( 55 ))
	original: list[i32] = [1, 2, 3, 4, 5, 6, 7, 8]
	shuffled: list[i32] = [1, 2, 3, 4, 5, 6, 7, 8]
	r.shuffle( shuffled )
	if shuffled.__len__() != original.__len__():
		return 1
	# every original value still present exactly once - a real permutation,
	# not a shuffle that drops/duplicates elements.
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < usize( 8 ):
			target: i32 = original.__getitem__( i ).unwrap( 'i < 8' )
			count: usize = 0
			j: usize = 0
			while j < usize( 8 ):
				if shuffled.__getitem__( j ).unwrap( 'j < 8' ) == target:
					count += usize( 1 )
				j += usize( 1 )
			if count != usize( 1 ):
				return 2
			i += usize( 1 )
	return 0
''' ),
			( 'shuffle_empty_and_single_are_noops', '''
import random

def main() -> i32:
	r: random.Random = random.Random( u64( 1 ))
	empty: list[i32] = list[i32]()
	r.shuffle( empty )
	if empty.__len__() != usize( 0 ):
		return 1
	single: list[i32] = [42]
	r.shuffle( single )
	if single.__getitem__( 0 ).unwrap( 'still has one element' ) != 42:
		return 2
	return 0
''' ),
			( 'choice_returns_element_from_seq', '''
import random

def main() -> i32:
	r: random.Random = random.Random( u64( 8 ))
	seq: list[i32] = [10, 20, 30]
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < usize( 100 ):
			c: i32 = r.choice( seq ).unwrap( 'nonempty' )
			if c != 10 and c != 20 and c != 30:
				return 1
			i += usize( 1 )
	return 0
''' ),
			( 'choice_on_empty_seq_is_err', '''
import random

def main() -> i32:
	r: random.Random = random.Random( u64( 8 ))
	empty: list[i32] = list[i32]()
	if r.choice( empty ).is_ok():
		return 1
	return 0
''' ),
			( 'seed_reseed_reproduces_sequence', '''
import random

def main() -> i32:
	r: random.Random = random.Random( u64( 100 ))
	first: f64 = r.random()
	second: f64 = r.random()
	r.seed( u64( 100 ))
	if r.random() != first:
		return 1
	if r.random() != second:
		return 2
	return 0
''' ),
			( 'module_level_default_instance_works', '''
import random

def main() -> i32:
	random.seed( u64( 999 ))
	a: f64 = random.random()
	random.seed( u64( 999 ))
	b: f64 = random.random()
	if a != b:
		return 1
	n: i64 = random.randint( i64( 1 ), i64( 10 ))
	if n < i64( 1 ) or n > i64( 10 ):
		return 2
	u: f64 = random.uniform( 0.0, 1.0 )
	if u < 0.0 or u > 1.0:
		return 3
	seq: list[i32] = [1, 2, 3]
	random.shuffle( seq )
	if seq.__len__() != usize( 3 ):
		return 4
	c: i32 = random.choice( seq ).unwrap( 'nonempty' )
	return 0
''' ),
		] )


if __name__ == '__main__':
	unittest.main()
