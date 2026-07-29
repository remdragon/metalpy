def bisect_right[T,K](
	arr: slice[T],
	x: K,
	key: Callable[[T],K]|None = None,
) -> usize:
	lo: usize = 0
	hi: usize = len( arr )
	
	if key is not None:
		while lo < hi:
			mid: usize = (lo + hi) // 2
			if x < key( arr[mid] ):
				hi = mid
			else:
				lo = mid + 1
	else:
		while lo < hi:
			mid: usize = (lo + hi) // 2
			if x < arr[mid]:
				hi = mid
			else:
				lo = mid + 1
	
	return lo


def bisect_left[T,K](
	arr: slice[T],
	x: K,
	key: Callable[[T],K]|None = None,
) -> usize:
	lo: usize = 0
	hi: usize = len( arr )
	
	if key is not None:
		while lo < hi:
			mid: usize = (lo + hi) // 2
			if key( arr[mid] ) < x:
				lo = mid + 1
			else:
				hi = mid
	else:
		while lo < hi:
			mid: usize = (lo + hi) // 2
			if arr[mid] < x:
				lo = mid + 1
			else:
				hi = mid
	
	return lo
