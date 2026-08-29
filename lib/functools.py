# lib/functools.py — Memoize[K,V]: a caching wrapper around a single-
# argument function, standing in for Python's @functools.lru_cache/@cache.
# Not a decorator - this compiler has no user-created decorators yet (see
# FUTURE.md) - so it's applied explicitly:
#   cache: functools.Memoize[i64, i64] = functools.Memoize[i64,i64]( fib )
#   result: i64 = cache.call( 10 )
# Single-argument only; a multi-argument function can pack its arguments
# into a tuple key. Unbounded (no eviction) - matches @lru_cache(maxsize=
# None)/@cache, not the size-bounded default.
#
# functools.partial deliberately NOT ported: this compiler already has
# closures/lambdas, which cover what partial() is for in Python (`lambda
# x: f( x, fixed )` today) - a separate construct wouldn't earn its keep.

class Memoize[K, V]:
	__fn:    Ptr[Callable[[K],V]]
	__cache: dict[K, V]

	def __init__( self, fn: Ptr[Callable[[K],V]] ) -> None:
		self.__fn = fn
		self.__cache = dict[K, V]()

	def call( self, key: K ) -> V:
		existing: Result[V, KeyError] = self.__cache.__getitem__( key )
		if existing.is_ok():
			return existing.unwrap( 'checked is_ok() above' )
		value: V = self.__fn( key )
		self.__cache[key] = value
		return value

	def cached_count( self ) -> usize:
		return self.__cache.__len__()
