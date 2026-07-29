from codecs import Codec
from codecs.utf8 import utf8
import compiler
import sys
from .__list import list
from .__dict import dict
from .__int import int

@cunion
class ResultPayload[T,E]:
	ok: T
	err: E

@cstruct
class Result[T,E]:
	_payload: ResultPayload[T,E]
	_tag: u8
	
	@staticmethod
	def Ok( val: T ) -> Result[T,E]:
		res: Result[T,E] = Result.__allocate__(
			_payload = ResultPayload( ok = val ),
			_tag = 0,
		)
		return res
	
	@staticmethod
	def Err( err: E ) -> Result[T,E]:
		res: Result[T,E] = Result.__allocate__(
			_payload = ResultPayload( err = err ),
			_tag = 1,
		)
		return res
	
	def is_ok( self ) -> bool:
		return self._tag == 0
	
	def is_err( self ) -> bool:
		return self._tag == 1
	
	def or_return( self ) -> T:
		if self.is_err():
			compiler.early_return( self._payload.err )
		return self._payload.ok
	
	def unwrap( self, errmsg: str ) -> T:
		if self.is_ok():
			return self._payload.ok
		sys.panic( errmsg )
	
	@overload
	def unwrap_or( self, default: T ) -> T:
		...
	
	def unwrap_or( self, default: T|None = None ) -> T|None:
		if self.is_ok():
			return self._payload.ok
		return default

@cstruct
class slice[T]:
	_ptr: ConstPtr[T]
	__len: usize
	
	def len( self ) -> usize:
		return self.__len
	
	def get_unchecked( self, index: usize ) -> T:
		return self._ptr.add( index ).deref()
	
	def __getitem__( self, index: usize ) -> Result[T,IndexError]:
		if index >= self.__len:
			return Result.Err( IndexError )
		
		return Result.Ok( self.get_unchecked( index ))
	
	def get_assert( self, index: usize ) -> T:
		if index >= self.__len:
			sys.panic( 'bad slice index' )
		return self.get_unchecked( index )

class bytes:
	__data: ConstPtr[u8]
	
	__len: usize
	
	def __init__( self, copy_from: bytes|bytearray ) -> None:
		self.__len = len( copy_from )
		data = sys.alloc[u8]( self.__len )
		sys.memcpy( data, copy_from.get_ptr(), self.__len )
		self.__data = data
	
	@staticmethod
	def from_bytearray( src: move[bytearray] ) -> bytes:
		length: usize = len( src )
		match src.release():
			case Result.Ok( ptr ):
				return bytes.__allocate__(
					__data = ptr,
					__len = length,
				)
			case Result.Err( _ ):
				return bytes( src )
	
	def __len__( self ) -> usize:
		return self.__len
	
	def __del__( self ) -> None:
		sys.free( self.__data )
	
	def get_const_ptr( self ) -> ConstPtr[u8]:
		return self.__data
	
	def decode( self, codec: Codec = utf8 ) -> Result[str,CodecError]:
		return utf8.decode( self )

BYTEARRAY_INVALID: ConstPtr[u8] = b'' # this is a sentinel to indicate a bytearray was released

class bytearray:
	__data: Ptr[u8]
	__len: usize
	__cap: usize
	
	def __init__( self, size: usize ) -> None:
		self.__len = size
		self.__cap = size
		self.__data = sys.alloc[u8]( size )
		sys.memzero( self.__data, size )
	
	def __len__( self ) -> usize:
		if compiler.target.debug:
			global BYTEARRAY_INVALID
			assert self.__data != BYTEARRAY_INVALID, 'bytearray.__len__() called after release()'
		return self.__len
	
	def get_ptr( self ) -> Ptr[u8]:
		if compiler.target.debug:
			global BYTEARRAY_INVALID
			assert self.__data != BYTEARRAY_INVALID, 'bytearray.get_ptr() called after release()'
		return self.__data
	
	def get_const_ptr( self ) -> ConstPtr[u8]:
		if compiler.target.debug:
			global BYTEARRAY_INVALID
			assert self.__data != BYTEARRAY_INVALID, 'bytearray.get_const_ptr() called after release()'
		return self.__data
	
	def decode( self, codec: Codec = utf8 ) -> Result[str,CodecError]:
		if compiler.target.debug:
			global BYTEARRAY_INVALID
			assert self.__data != BYTEARRAY_INVALID, 'bytearray.decode() called after release()'
		return codec.decode( self )
	
	@move
	def release( self ) -> Result[Ptr[u8],OwnershipError]:
		global BYTEARRAY_INVALID
		if compiler.refcount( self ) != 1:
			return Result.Err( OwnershipError.SharedReference )
		ptr = self.__data
		self.__len = 0
		self.__cap = 0
		self.__data = BYTEARRAY_INVALID
		return Result.Ok( ptr )
	
	def __del__( self ) -> None:
		global BYTEARRAY_INVALID
		if self.__data != BYTEARRAY_INVALID: # this can happen if release() is called and successful
			sys.free( self.__data )

class str:
	__data: ConstPtr[u8]

	# UTF-8 encoded byte count - NOT what len(s)/__len__ reports (that's the
	# Unicode code point count, matching real Python semantics). Named __cap
	# rather than __len specifically so nothing internal can casually
	# confuse the two again; see byte_len() for the public accessor.
	__cap: usize

	def __init__( self, copy_from: str ) -> None:
		self.__cap = copy_from.__cap
		with compiler.wrap_arithmetic:
			# NOTE: this would only wrap if self.__cap == usize.max which would only happen in memory corruption scenarios
			alloc_size = self.__cap + 1
		data: Ptr[u8] = sys.alloc[u8]( alloc_size )
		sys.memcpy( data, copy_from.__data, alloc_size )
		data[self.__cap] = 0
		self.__data = data

	def __del__( self ) -> None:
		sys.free( self.__data )

	def __add__( self, other: str ) -> Result[str,OverflowError]:
		new_len: usize = self.__cap + other.__cap
		new_buf: Ptr[u8] = sys.alloc[u8]( new_len + 1 )
		sys.memcpy( new_buf, self.__data, self.__cap )
		sys.memcpy( new_buf.add( self.__cap ), other.__data, other.__cap )
		new_buf.add( new_len ).write( 0 )
		
		return Result.Ok( str._from_owned_cstr( new_buf, new_len ))
	
	@staticmethod
	def concat( parts: slice[str] ) -> Result[str,OverflowError]:
		new_len: usize = 0
		i: usize = 0
		count: usize = parts.len()
		for i in range( count ):
			part: str = parts.get_assert( i )
			new_len += part.__cap
		
		new_buf: Ptr[u8] = sys.alloc[u8]( new_len + 1 )
		with errdefer:
			sys.free( new_buf )
		offset: usize = 0
		
		for i in range( count ):
			part: str = parts.get_assert( i )
			part_len: usize = part.__cap
			memcpy( new_buf.add( offset ), part.__data, part_len )
			offset += part_len
		
		new_buf.add( new_len ).write( 0 ) # guarantee null termination
		
		return Result.Ok( str._from_owned_cstr( new_buf, new_len ))
	
	def encode( self, codec: Codec = utf8 ) -> Result[bytes,CodecError]:
		return codec.encode( self )
	
	@overload
	@staticmethod
	def from_cstr( buf: ConstPtr[u8], length: usize ) -> str:
		with compiler.panic_arithmetic( 'invalid str length' ):
			new_buf: Ptr[u8] = sys.alloc[u8]( length + 1 )
		sys.memcpy( new_buf, buf, length )
		new_buf.add( length ).write( 0 ) # guarantee null termination
		
		return Result.Ok( str._from_owned_cstr( new_buf, length ))
	
	@overload
	@staticmethod
	def from_cstr( src: move[bytearray] ) -> str:
		length: usize = len( src )
		match src.release():
			case Result.Ok( ptr ):
				return str.__allocate__(
					__data = ptr,
					__cap = length,
				)
			case Result.Err( _ ):
				# must copy because we don't have exclusive ownership of src:
				return str.from_cstr( src.get_ptr(), length )
	
	def get_const_ptr( self ) -> ConstPtr[u8]:
		return self.__data
	
	def get_cstr( self ) -> ConstPtr[u8]:
		return self.__data
	
	def byte_len( self ) -> usize:
		# The UTF-8 encoded byte count - what most internal stdlib code
		# actually wants (buffer sizing, memcpy counts, ...), as opposed to
		# len(s)/__len__ below (Unicode code point count, matching real
		# Python semantics for len() on a str).
		return self.__cap
	
	def __len__( self ) -> usize:
		# Unicode code point count (real Python len(s) semantics) - counts
		# bytes that are NOT UTF-8 continuation bytes (top two bits != 0b10).
		# count/i are bounded by __cap (an existing buffer's length in bytes,
		# already itself usize-representable), so they can't actually overflow -
		# panic_arithmetic documents that invariant rather than forcing every
		# caller through Result[usize,OverflowError] for something impossible.
		count: usize = 0
		i: usize = 0
		with compiler.panic_arithmetic( 'bounded by __cap, cannot overflow' ):
			while i < self.__cap:
				c: u8 = self.__data[i]
				if ( c & 0xC0 ) != 0x80:
					count += 1
				i += 1
		return count
	
	@private
	@staticmethod
	def _from_owned_cstr( ptr: Ptr[u8], length: usize ) -> str:
		# NOTE: a 0-byte before the end of the string is valid utf-8, so we
		# can't use cstrlen() here, we can only check to make sure the terminating 0 exists where expected
		if ptr[length] != 0:
			sys.panic( 'bad cstr' )
		s: str = str.__allocate__(
			__data = ptr,
			__cap = length,
		)
		return s

def print( msg: str, end: str = '\n' ) -> None:
	# No *args/**kwargs, use f-strings instead (once implemented)
	sys.stdout.write( msg )
	if end:
		sys.stdout.write( end )
