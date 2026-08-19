# bisect_right/bisect_left compare elements DIRECTLY (T against T) - use
# these when arr's own element type is what you're comparing against.
#
# bisect_right_by_key/bisect_left_by_key extract a comparison key via a
# REQUIRED key function (T -> K) instead - use these when T and K are
# different types (e.g. searching a struct array by one numeric field).
#
# Deliberately four functions, not one with an Optional key= parameter:
# an Optional key monomorphizes a SINGLE function body per (T,K), and that
# body's "key is None" branch compares arr[mid] (T) directly against x (K) -
# which only type-checks when T and K are the same type. A caller like
# RawDict (T=RawIndex, K=u64, always passing a key) would still fail to
# compile, because the OTHER, never-taken-at-runtime branch is still
# monomorphized and still needs T and K comparable. Same split Rust's
# binary_search/binary_search_by_key uses for the identical reason.

def bisect_right[T]( arr: slice[T], x: T ) -> usize:
	lo: usize = 0
	hi: usize = len( arr )

	with compiler.panic_arithmetic( 'bisect_right: overflow' ):
		while lo < hi:
			mid: usize = (lo + hi) // 2
			# in bounds by construction - the loop invariant lo <= mid < hi <= len(arr)
			# always holds, same "in bounds by construction" idiom str.concat's own
			# get_unchecked use relies on (see slice.get_unchecked's own docstring)
			if x < arr.get_unchecked( mid ):
				hi = mid
			else:
				lo = mid + 1

	return lo


def bisect_left[T]( arr: slice[T], x: T ) -> usize:
	lo: usize = 0
	hi: usize = len( arr )

	with compiler.panic_arithmetic( 'bisect_left: overflow' ):
		while lo < hi:
			mid: usize = (lo + hi) // 2
			if arr.get_unchecked( mid ) < x:
				lo = mid + 1
			else:
				hi = mid

	return lo


def bisect_right_by_key[T,K]( arr: slice[T], x: K, key: Ptr[Callable[[T],K]] ) -> usize:
	lo: usize = 0
	hi: usize = len( arr )

	with compiler.panic_arithmetic( 'bisect_right_by_key: overflow' ):
		while lo < hi:
			mid: usize = (lo + hi) // 2
			if x < key( arr.get_unchecked( mid )):
				hi = mid
			else:
				lo = mid + 1

	return lo


def bisect_left_by_key[T,K]( arr: slice[T], x: K, key: Ptr[Callable[[T],K]] ) -> usize:
	lo: usize = 0
	hi: usize = len( arr )

	with compiler.panic_arithmetic( 'bisect_left_by_key: overflow' ):
		while lo < hi:
			mid: usize = (lo + hi) // 2
			if key( arr.get_unchecked( mid )) < x:
				lo = mid + 1
			else:
				hi = mid

	return lo
