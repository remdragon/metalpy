from codecs import Codec, CodecError
from codecs.utf8 import utf8
import compiler
import sys
from .__dict import dict
from .__errors import OSError
from .__int import int
from .__list import list

# markers with no payload of their own - Check-mode arithmetic (AddCheck/
# SubCheck/MulCheck/...) and Div/Mod produce Result[T,OverflowError]/
# Result[T,ZeroDivisionError] purely as a tag (see ir.py's own comments on
# those opcodes); slice.__getitem__ raises IndexError the same way
class OverflowError: pass
class ZeroDivisionError: pass
class IndexError: pass

@union
class Result[T,E]:
	Ok: T
	Err: E

	def is_ok( self ) -> bool:
		return self.tag == 0

	def is_err( self ) -> bool:
		return self.tag == 1

	def or_return( self ) -> T:
		if self.is_err():
			compiler.early_return( self.data.v_Err )
		return self.data.v_Ok

	def unwrap( self, errmsg: str ) -> T:
		if self.is_ok():
			return self.data.v_Ok
		sys.panic( errmsg )

	@overload
	def unwrap_or( self, default: T ) -> T:
		...

	def unwrap_or( self, default: T|None = None ) -> T|None:
		if self.is_ok():
			return self.data.v_Ok
		return default


@cstruct
class slice[T]:
	_ptr: ConstPtr[T]
	__len: usize
	
	def __len__( self ) -> usize:
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
		sys.memcpy( data, copy_from.get_const_ptr(), self.__len )
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
			case Result.Err( sys.OwnershipError.SharedReference( src2 )):
				return bytes( src2 )
	
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
			assert self.__data != BYTEARRAY_INVALID, 'bytearray.__len__() called after release()'
		return self.__len
	
	def get_ptr( self ) -> Ptr[u8]:
		if compiler.target.debug:
			assert self.__data != BYTEARRAY_INVALID, 'bytearray.get_ptr() called after release()'
		return self.__data
	
	def get_const_ptr( self ) -> ConstPtr[u8]:
		if compiler.target.debug:
			assert self.__data != BYTEARRAY_INVALID, 'bytearray.get_const_ptr() called after release()'
		return self.__data
	
	def decode( self, codec: Codec = utf8 ) -> Result[str,CodecError]:
		if compiler.target.debug:
			assert self.__data != BYTEARRAY_INVALID, 'bytearray.decode() called after release()'
		return codec.decode( self )
	
	@move
	def release( self ) -> Result[Ptr[u8],sys.OwnershipError]:
		if compiler.refcount( self ) != 1:
			return Result.Err( sys.OwnershipError.SharedReference( self ))
		ptr = self.__data
		self.__len = 0
		self.__cap = 0
		self.__data = BYTEARRAY_INVALID
		return Result.Ok( ptr )
	
	def __del__( self ) -> None:
		if self.__data != BYTEARRAY_INVALID: # this can happen if release() is called and successful
			sys.free( self.__data )

class str:
	__data: ConstPtr[u8]
	
	__byte_size: usize # the number of bytes (code units) include the zero-terminater
	
	def __init__( self, copy_from: str ) -> None:
		self.__byte_size = copy_from.__byte_size
		data: Ptr[u8] = sys.alloc[u8]( self.__byte_size )
		sys.memcpy( data, copy_from.__data, self.__byte_size )
		self.__data = data
	
	def __del__( self ) -> None:
		sys.free( self.__data )
	
	def __add__( self, other: str ) -> Result[str,OverflowError]:
		self_len = self.__byte_size - 1
		new_byte_size: usize = self_len + other.__byte_size
		new_buf: Ptr[u8] = sys.alloc[u8]( new_byte_size )
		errdefer( sys.free( new_buf ))

		sys.memcpy( new_buf, self.__data, self_len )
		sys.memcpy( new_buf + self_len, other.__data, other.__byte_size )
		
		return str._from_owned_cstr( new_buf, new_byte_size )
	
	@staticmethod
	def concat( parts: slice[str] ) -> Result[str,OverflowError]:
		new_size: usize = 1 # for the zero terminator
		i: usize = 0
		count: usize = parts.len()
		for i in range( count ):
			part: str = parts.get_unchecked( i )
			new_size += part.__byte_size - 1

		new_buf: Ptr[u8] = sys.alloc[u8]( new_size )
		errdefer( sys.free( new_buf ))
		offset: usize = 0

		for i in range( count ):
			part: str = parts.get_unchecked( i )
			part_len: usize = part.__byte_size - 1
			sys.memcpy( new_buf + offset, part.__data, part_len )
			offset += part_len
		
		new_buf[offset] = 0 # guarantee null termination
		
		return str._from_owned_cstr( new_buf, new_size )
	
	def encode( self, codec: Codec = utf8 ) -> Result[bytes,CodecError]:
		return codec.encode( self )
	
	@staticmethod
	def from_cstr( buf: ConstPtr[u8], size_including_zero_terminator: usize ) -> Result[str,CodecError]:
		'''
		build a str from a raw pointer.
		'''
		with compiler.panic_arithmetic( 'invalid str length' ):
			new_buf: Ptr[u8] = sys.alloc[u8]( size_including_zero_terminator )
		errdefer( sys.free( new_buf ))
		
		sys.memcpy( new_buf, buf, size_including_zero_terminator )
		
		if new_buf[size_including_zero_terminator-1]:
			return Result.Err( CodecError( 'utf-8', 'missing null terminator' ))
		
		return str._from_owned_cstr( new_buf, size_including_zero_terminator )
	
	@staticmethod
	def from_cstr( src: move[bytearray] ) -> Result[str,CodecError]:
		byte_size: usize = len( src )
		
		match src.release():
			case Result.Ok( ptr ):
				return str._from_owned_cstr( ptr, byte_size )
			case Result.Err( sys.OwnershipError.SharedReference( src2 )):
				# must copy because we didn't have exclusive ownership of src
				# but now the release failed and src isn't usable anymore because of @move
				# e is a OwnershipError.SharedReference, which carries the object back to us
				return str.from_cstr( src2.get_const_ptr(), byte_size )
	
	def get_const_ptr( self ) -> ConstPtr[u8]:
		return self.__data
	
	def get_cstr( self ) -> ConstPtr[u8]:
		return self.__data
	
	def byte_size( self ) -> usize:
		# The UTF-8 encoded byte count - what most internal stdlib code
		# actually wants (buffer sizing, memcpy counts, ...), as opposed to
		# len(s)/__len__ below (Unicode code point count, matching real
		# Python semantics for len() on a str).
		# this function returns the size of byte including the zero terminator
		return self.__byte_size
	
	def byte_len( self ) -> usize:
		with compiler.saturate_arithmetic: # __byte_size can't be 0 because an empty str still has '\0'
			return self.__byte_size - 1
	
	def __len__( self ) -> usize:
		# Unicode code point count (real Python len(s) semantics) - counts
		# bytes that are NOT UTF-8 continuation bytes (top two bits != 0b10).
		# count/i are bounded by __byte_len (an existing buffer's length in bytes,
		# already itself usize-representable), so they can't actually overflow -
		# panic_arithmetic documents that invariant rather than forcing every
		# caller through Result[usize,OverflowError] for something impossible.
		count: usize = 0
		i: usize = 0
		with compiler.panic_arithmetic( 'bounded by byte_len, cannot overflow' ):
			assert self.__byte_size > 0, 'byte_size must be > 0'
			byte_len: usize = self.__byte_size - 1
			while i < byte_len:
				c: u8 = self.__data[i]
				if not c:
					return count
				if ( c & 0xC0 ) != 0x80:
					count += 1
				i += 1
		return count
	
	@private
	@staticmethod
	def _from_owned_cstr( ptr: Ptr[u8], byte_size_including_zero_terminator: usize ) -> Result[str,CodecError]:
		# NOTE: a 0-byte before the end of the string is valid utf-8, so we
		# can't use cstrlen() here, we can only check to make sure the terminating 0 exists where expected
		if byte_size_including_zero_terminator == 0:
			return Result.Err( CodecError( 'utf-8', 'empty buffer' ))
		byte_len: usize = byte_size_including_zero_terminator - 1
		if ptr[byte_len] != 0:
			return Result.Err( CodecError( 'utf-8', 'missing 0-terminator' ))
		
		# walk through ptr and confirm valid utf-8 encoding or return CodecError
		i: usize = 0
		with compiler.panic_arithmetic( 'bounded by byte_len, cannot overflow ' ):
			while i < byte_len:
				byte1 = ptr[i]
				
				# 1-byte sequence (ASCII): 0xxxxxxx
				if (byte1 & 0x80) == 0x00:
					i += 1
					continue
				
				# unexpected continuation byte as a leading byte
				if (byte1 & 0xC0) == 0x80:
					return Result.Err( CodecError( 'utf-8', 'Unexpected continuation byte as leading byte' ))
				
				# 2-byte sequence: 110xxxxx 10xxxxxx
				elif (byte1 & 0xE0) == 0xC0:
					if i + 1 >= byte_len:
						return Result.Err( CodecError( 'utf-8', 'Truncated 2-byte sequence' ))
					# Overlong encoding check: code point must be >= U+0080
					if byte1 < 0xC2:
						return Result.Err( CodecError( 'utf-8', 'Overlong 2-byte encoding' ))
					byte2 = ptr[i + 1]
					if (byte2 & 0xC0) != 0x80:
						return Result.Err( CodecError( 'utf-8', 'Invalid continuation byte in 2-byte sequence' ))
					i += 2

				# 3-byte sequence: 1110xxxx 10xxxxxx 10xxxxxx
				elif (byte1 & 0xF0) == 0xE0:
					if i + 2 >= byte_len:
						return Result.Err( CodecError( 'utf-8', 'Truncated 3-byte sequence' ))
					byte2 = ptr[i + 1]
					byte3 = ptr[i + 2]
					# Overlong encoding check: code point must be >= U+0800
					if byte1 == 0xE0 and byte2 < 0xA0:
						return Result.Err( CodecError( 'utf-8', 'Overlong 3-byte encoding' ))
					if (byte2 & 0xC0) != 0x80 or (byte3 & 0xC0) != 0x80:
						return Result.Err( CodecError( 'utf-8', 'Invalid continuation byte in 3-byte sequence' ))
					# Surrogate halves validation: U+D800..U+DFFF are invalid
					if byte1 == 0xED and byte2 >= 0xA0:
						return Result.Err( CodecError( 'utf-8', 'UTF-16 surrogate half' ))
					i += 3

				# 4-byte sequence: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
				elif (byte1 & 0xF8) == 0xF0:
					if i + 3 >= byte_len:
						return Result.Err( CodecError( 'utf-8', 'Truncated 4-byte sequence' ))
					byte2 = ptr[i + 1]
					byte3 = ptr[i + 2]
					byte4 = ptr[i + 3]
					# Overlong encoding check: code point must be >= U+10000
					if byte1 == 0xF0 and byte2 < 0x90:
						return Result.Err( CodecError( 'utf-8', 'Overlong 4-byte encoding' ))
					if (byte2 & 0xC0) != 0x80 or (byte3 & 0xC0) != 0x80 or (byte4 & 0xC0) != 0x80:
						return Result.Err( CodecError( 'utf-8', 'Invalid continuation byte in 4-byte sequence' ))
					# Maximum Unicode code point check: cannot exceed U+10FFFF
					if byte1 == 0xF4 and byte2 >= 0x90:
						return Result.Err( CodecError( 'utf-8', 'Code point exceeds maximum valid Unicode (U+10FFFF)' ))
					if byte1 > 0xF4:
						return Result.Err( CodecError( 'utf-8', 'Code point exceeds maximum valid Unicode (sequence prefix > 0xF4)' ))
					i += 4

				# Invalid leading bytes (0xF5..0xFF)
				else:
					return Result.Err( CodecError( 'utf-8', 'Invalid leading byte' ))
		
		s: str = str.__allocate__(
			__data = ptr,
			__byte_size = byte_size_including_zero_terminator,
		)
		return s

def print( msg: str, end: str = '\n' ) -> None:
	# No *args/**kwargs, use f-strings instead (once implemented)
	sys.stdout.write( msg )
	if end:
		sys.stdout.write( end )

# a single generic function now that bare-call monomorphization can infer T
# from the argument (see lowering.py's _lower_inferred_generic_call) - a real
# Call for now, forwarding to whatever T's own __len__ is; TODO once @inline
# exists (see TODO.txt): this should become @inline so len(x) compiles down
# to the same code as x.__len__() directly, no call overhead.

def len[T]( t: T ) -> usize:
	return t.__len__()

# import this at the end because it depends on str etc to already be pre-parsed:
from .__File import File
