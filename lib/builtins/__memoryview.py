'''
memoryview — a read/write VIEW over a bytearray's own buffer (no copy),
supporting slice syntax (mv[a:b]/mv[:b]/mv[a:]) and the `with` context-
manager protocol.

A deliberate extension beyond real Python: real Python's own memoryview
does NOT support `with memoryview(x) as mv:` (its context-manager form
exists, but only for something ALREADY holding one, e.g. re-entering an
existing view - `memoryview(x)` itself isn't normally written inside a
`with`). This one does, so `with memoryview(mm) as mv:` reads naturally
right where the view is created - see grap.mpy (the real-world port this
was built for), whose own usage is exactly that shape.

Holds an OWNED reference to its source (bytearray or mmap - an ordinary
RC field, not a raw pointer alone) specifically so the buffer stays alive
for as long as the view does - real Python's own memoryview does the
analogous thing via its exporter's buffer-refcount protocol.
'''

import compiler
from mmap import mmap

class memoryview:
	__ptr: Ptr[u8]
	__len: usize
	__source: bytearray|mmap

	def __init__( self, source: bytearray|mmap ) -> None:
		# a single union-typed __init__, not two overloads - __init__
		# overloading (unlike ordinary method overloading) isn't supported
		# yet (confirmed directly: "memoryview.__init__ is overloaded -
		# not supported yet"). get_ptr()/__len__() are identically named
		# on both bytearray and mmap, so this works unnarrowed either way.
		self.__ptr = source.get_ptr()
		self.__len = len( source )
		self.__source = source

	def __enter__( self ) -> memoryview:
		return self

	def __exit__( self ) -> None:
		pass

	def __len__( self ) -> usize:
		return self.__len

	@overload
	def __getitem__( self, idx: usize ) -> Result[u8, IndexError]:
		if idx >= self.__len:
			return Result.Err( IndexError() )
		return Result.Ok( self.__ptr[idx] )

	@overload
	def __getitem__( self, s: slice ) -> memoryview:
		''' s[a:b] slice syntax (lowering.py's _lower_slice_subscript) -
		infallible, matching real Python's own slice semantics exactly -
		out-of-range bounds silently clamp rather than raising (see
		_resolve_slice_bounds), unlike single-element s[i] above, which
		DOES error on an out-of-range index. Delegates to _byte_slice below
		for the actual (non-copying) view construction. '''
		( start, stop ) = _resolve_slice_bounds( s, self.__len )
		return self._byte_slice( start, stop )

	def get_ptr( self ) -> Ptr[u8]:
		return self.__ptr

	def get_const_ptr( self ) -> ConstPtr[u8]:
		return self.__ptr

	@private
	def _byte_slice( self, start: usize, end: usize ) -> memoryview:
		''' a VIEW into the SAME underlying buffer (start,end) - unlike
		str/bytearray's own _byte_slice (lib/builtins/__init__.py), this
		does NOT copy, matching real Python's own memoryview slicing
		semantics (a sub-view, not a fresh allocation). Called by
		__getitem__(slice) above once bounds are validated - kept
		separate so trusted internal callers with already-valid bounds
		don't pay for a redundant check. '''
		with compiler.panic_arithmetic( 'memoryview slice: end < start, cannot overflow' ):
			piece_len: usize = end - start
			ptr: Ptr[u8] = self.__ptr + start
		return memoryview.__allocate__( __ptr = ptr, __len = piece_len, __source = self.__source )
