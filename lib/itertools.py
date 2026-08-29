# lib/itertools.py — lazy iteration helpers, ported from Python's itertools
# for the ones that carry their weight in a statically-typed compiler with
# no *args (so chain()/zip()-shaped fan-in stays 2-iterable, not variadic -
# see SYNTAX.md's own print() note for why). product/permutations/
# combinations/groupby deliberately NOT ported yet - add if a real caller
# needs them.

import compiler


def count[I]( start: I = 0, step: I = 1 ) -> Generator[I, StopIteration]:
	with compiler.wrap_arithmetic:
		while True:
			yield start
			start += step


def chain[T, S1: Iterable[T], S2: Iterable[T]]( a: S1, b: S2 ) -> Generator[T, StopIteration]:
	for item in a.__iter__(): # not iter(a)/iter(b) - see builtins' own enumerate() comment
		yield item
	for item in b.__iter__():
		yield item


@overload
def islice[T, S: Iterable[T]]( seq: S, stop: usize ) -> Generator[T, StopIteration]:
	i: usize = 0
	with compiler.wrap_arithmetic:
		for item in seq.__iter__():
			if i >= stop:
				return
			yield item
			i += 1

@overload
def islice[T, S: Iterable[T]]( seq: S, start: usize, stop: usize ) -> Generator[T, StopIteration]:
	i: usize = 0
	with compiler.wrap_arithmetic:
		for item in seq.__iter__():
			if i >= stop:
				return
			if i >= start:
				yield item
			i += 1
