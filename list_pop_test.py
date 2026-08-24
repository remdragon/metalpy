# Real-compile-and-run regression test for list[T].pop()'s three call
# shapes: the original zero-arg Result[T,IndexError]-returning form, an
# @overload stub `pop(default: T) -> T` (a concrete, non-None default
# narrows the return type to plain T), and the real `pop(default: T|None)
# -> T|None` implementation the stub binds to - mirrors Result.unwrap_or's
# own stub+impl split (see unwrap_or_overload_default_test.py).

import unittest

import test_support
from test_support import RealCompileMixin

_LIST_POP_THREE_VARIANTS = '''
def main() -> i32:
	xs: list[str] = [ 'a', 'b' ]

	# concrete-default form (the @overload stub, narrows to plain T) -
	# non-empty list never touches the default
	if xs.pop( '' ) != 'b':
		return 1
	if xs.pop( '' ) != 'a':
		return 2
	# now empty - falls back to the given default
	if xs.pop( '' ) != '':
		return 3

	# zero-arg form is untouched, still Result[T,IndexError]
	r: Result[str, IndexError] = xs.pop()
	if not r.is_err():
		return 4

	# default: T|None form
	ys: list[str] = [ 'x' ]
	v1: str|None = ys.pop( None )
	if v1 is None or v1 != 'x':
		return 5
	v2: str|None = ys.pop( None )
	if v2 is not None:
		return 6

	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile list.pop() tests' )
class ListPopOverloadTests( RealCompileMixin, unittest.TestCase ):
	def test_list_pop_three_variants( self ) -> None:
		self.assert_programs_run([ ( 'list_pop_three_variants', _LIST_POP_THREE_VARIANTS ) ])


if __name__ == '__main__':
	unittest.main()
