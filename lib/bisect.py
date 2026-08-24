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
#
# arr: UnsafeList[T] directly (no view/copy type of its own - the old
# slice[T] view class this used to take is gone). Every
# arr.__getitem__(mid).unwrap(...) call below is transient/inline (never
# bound to a name) - that's fine because __getitem__ already returns an
# owned value for RC T, so its own incref and the compiler's automatic
# scope-exit decref on the unnamed temp cancel out net zero, exactly like a
# bare borrow would, with no extra code needed here.

def bisect_right[T]( arr: UnsafeList[T], x: T ) -> usize:
	lo: usize = 0
	hi: usize = len( arr )

	with compiler.panic_arithmetic( 'bisect_right: overflow' ):
		while lo < hi:
			mid: usize = (lo + hi) // 2
			# in bounds by construction - the loop invariant lo <= mid < hi <= len(arr)
			# always holds
			if x < arr.__getitem__( mid ).unwrap( 'bisect_right: index in bounds by construction' ):
				hi = mid
			else:
				lo = mid + 1

	return lo


def bisect_left[T]( arr: UnsafeList[T], x: T ) -> usize:
	lo: usize = 0
	hi: usize = len( arr )

	with compiler.panic_arithmetic( 'bisect_left: overflow' ):
		while lo < hi:
			mid: usize = (lo + hi) // 2
			if arr.__getitem__( mid ).unwrap( 'bisect_left: index in bounds by construction' ) < x:
				lo = mid + 1
			else:
				hi = mid

	return lo


def bisect_right_by_key[T,K]( arr: UnsafeList[T], x: K, key: Ptr[Callable[[T],K]] ) -> usize:
	lo: usize = 0
	hi: usize = len( arr )

	with compiler.panic_arithmetic( 'bisect_right_by_key: overflow' ):
		while lo < hi:
			mid: usize = (lo + hi) // 2
			if x < key( arr.__getitem__( mid ).unwrap( 'bisect_right_by_key: index in bounds by construction' )):
				hi = mid
			else:
				lo = mid + 1

	return lo


def bisect_left_by_key[T,K]( arr: UnsafeList[T], x: K, key: Ptr[Callable[[T],K]] ) -> usize:
	lo: usize = 0
	hi: usize = len( arr )

	with compiler.panic_arithmetic( 'bisect_left_by_key: overflow' ):
		while lo < hi:
			mid: usize = (lo + hi) // 2
			if key( arr.__getitem__( mid ).unwrap( 'bisect_left_by_key: index in bounds by construction' )) < x:
				lo = mid + 1
			else:
				hi = mid

	return lo
