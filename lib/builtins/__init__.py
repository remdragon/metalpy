from codecs import Codec, CodecError
from codecs.utf8 import utf8
import compiler
import sys
from .__errors import OSError
from .__fastlist import FastList
from .__int import int, IntError, DivMod
from .__list import list
from .__RawDict import RawDict

# markers with no payload of their own - Check-mode arithmetic (AddCheck/
# SubCheck/MulCheck/...) and Div/Mod produce Result[T,OverflowError]/
# Result[T,ZeroDivisionError] purely as a tag (see ir.py's own comments on
# those opcodes); slice.__getitem__ raises IndexError the same way
class OverflowError: pass
class ZeroDivisionError: pass
class IndexError: pass
class KeyError: pass

# FNV-1a, 64-bit - a plain, fast, deterministic byte hash. Shared by
# str.__hash__ (below) and dict[K,V]'s own _hash_key (see __init__.py's
# own dict class further down, PLAN_CALLABLE.md) for any non-RC key type's
# generic byte-representation hash.
_FNV_OFFSET_BASIS: u64 = 14695981039346656037
_FNV_PRIME: u64 = 1099511628211

def _fnv1a_hash( data: ConstPtr[u8], length: usize ) -> u64:
	h: u64 = _FNV_OFFSET_BASIS
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < length:
			h = h ^ compiler.cast( u64, data[i] )
			h = h * _FNV_PRIME
			i += 1
	return h

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

BYTEARRAY_INVALID: u32 = 0 # this is a sentinel to indicate a bytearray was released

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
	
	def __add__( self, other: str ) -> str:
		if compiler.target.debug:
			assert self.__byte_size > 0, 'invalid str instance (byte_size must be >0)'
		with compiler.wrap_arithmetic:
			self_len = self.__byte_size - 1
		new_byte_size: usize
		with compiler.panic_arithmetic( 'irrational string length' ):
			new_byte_size = self_len + other.__byte_size
		new_buf: Ptr[u8] = sys.alloc[u8]( new_byte_size )

		sys.memcpy( new_buf, self.__data, self_len )
		with compiler.wrap_arithmetic:
			sys.memcpy( new_buf + self_len, other.__data, other.__byte_size )
		
		return str._from_owned_cstr( new_buf, new_byte_size ).unwrap( 'invalid UTF-8 in str.__add__' )
	
	@staticmethod
	def concat( parts: slice[str] ) -> str:
		new_size: usize = 1 # for the zero terminator
		i: usize = 0
		count: usize = parts.len()
		for i in range( count ):
			part: str = parts.get_unchecked( i )
			with compiler.panic_arithmetic( 'irrational string length' ):
				new_size += part.__byte_size - 1

		new_buf: Ptr[u8] = sys.alloc[u8]( new_size )
		offset: usize = 0

		for i in range( count ):
			part: str = parts.get_unchecked( i )
			with compiler.panic_arithmetic( 'irrational string length' ):
				part_len: usize = part.__byte_size - 1
			with compiler.wrap_arithmetic:
				sys.memcpy( new_buf + offset, part.__data, part_len )
				offset += part_len
		
		new_buf[offset] = 0 # guarantee null termination
		
		return str._from_owned_cstr( new_buf, new_size ).unwrap( 'invalid UTF-8 in concat' )
	
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

	def find( self, sub: str, start: usize = 0 ) -> Result[usize,IndexError]:
		''' byte offset of the first occurrence of sub in self, searching
		from byte offset start onward (default 0 - the whole string; used
		by split() below to resume searching just past each match, without
		its own separate scanning logic). UTF-8-safe at the byte level even
		though the scan itself is pure byte comparison (sys.memcmp): sub is
		itself valid UTF-8 (str's own construction-time invariant - see
		_from_owned_cstr), so a genuine match boundary can never be split
		mid-codepoint - an ASCII byte or a UTF-8 leading/continuation byte
		can only byte-for-byte equal the same kind of byte in sub, never
		straddle one. Empty sub matches at offset start, same as Python's
		str.find(''). '''
		self_len: usize = self.byte_len()
		sub_len: usize = sub.byte_len()
		if start > self_len:
			return Result.Err( IndexError() )
		if sub_len == 0:
			return Result.Ok( start )
		with compiler.wrap_arithmetic: # start <= self_len, just checked above
			remaining: usize = self_len - start
		if sub_len > remaining:
			return Result.Err( IndexError() )
		with compiler.wrap_arithmetic: # sub_len <= self_len, just checked above
			last_start: usize = self_len - sub_len
		i: usize = start
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			while i <= last_start:
				candidate: ConstPtr[u8] = self.__data + i
				if sys.memcmp( candidate, sub.__data, sub_len ) == 0:
					return Result.Ok( i )
				i += 1
		return Result.Err( IndexError() )

	def index( self, sub: str ) -> usize:
		''' like find(), but panics instead of returning Err - matches
		Python's str.index() raising ValueError where str.find() returns
		-1, adapted to this language's panic-not-exceptions convention. '''
		return self.find( sub ).unwrap( 'substring not found' )

	@private
	def _byte_slice( self, start: usize, end: usize ) -> str:
		''' bytes [start, end) of self, as a new, independently-owned str.
		Only ever called (see split(), below) with start/end landing on
		real UTF-8 codepoint boundaries - a byte-exact match of a valid-
		UTF-8 needle always lands there, see find()'s own comment on why -
		but still goes through _from_owned_cstr's own revalidation anyway,
		same "revalidate on construction" consistency str.concat/__add__
		above already keep rather than a separate trust-me bypass path. '''
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			piece_len: usize = end - start
			buf_size: usize = piece_len + 1 # +1 for the zero terminator
		new_buf: Ptr[u8] = sys.alloc[u8]( buf_size )
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			src: ConstPtr[u8] = self.__data + start
		sys.memcpy( new_buf, src, piece_len )
		new_buf[piece_len] = 0
		return str._from_owned_cstr( new_buf, buf_size ).unwrap( 'invalid UTF-8 in _byte_slice' )

	def split( self, sep: str ) -> list[str]:
		''' splits self on every occurrence of sep - Python str.split(sep)
		semantics (a leading/trailing/consecutive separator produces empty
		strings at those positions - no "collapse empty pieces" special
		case, unlike bare .split() with no separator, which isn't
		implemented here). Built entirely on find()/​_byte_slice above, not
		its own separate byte-scanning logic. sep must not be empty
		(Python raises ValueError there; this language panics instead,
		matching its own panic-not-exceptions convention throughout). '''
		if sep.byte_len() == 0:
			sys.panic( 'str.split(...): separator must not be empty' )
		result: list[str] = list[str]()
		self_len: usize = self.byte_len()
		sep_len: usize = sep.byte_len()
		start: usize = 0
		while True:
			found: Result[usize,IndexError] = self.find( sep, start )
			if found.is_err():
				result.append( self._byte_slice( start, self_len )).unwrap( 'str.split: append failed' )
				break
			match_start: usize = found.unwrap( 'unreachable: find() confirmed is_ok' )
			result.append( self._byte_slice( start, match_start )).unwrap( 'str.split: append failed' )
			with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
				start = match_start + sep_len
		return result

	def upper( self ) -> str:
		''' case_folder (see PLAN_CASE_FOLDING.md) is checked first, ahead
		of the OS-backed path - a program that never calls case_folding.
		install() never sets upper_count away from 0, so this is a single
		cheap field read for the overwhelming majority of programs, which
		don't import case_folding at all. Kept as ONE os-independent method
		(rather than duplicating this check into each of _upper_os_native's
		own @compiler.target(os=...) bodies) specifically so it's written
		and checked exactly once. '''
		if case_folder.upper_count != 0:
			return case_folder.upper( self )
		return self._upper_os_native()

	@compiler.target( os = 'windows' )
	@private
	def _upper_os_native( self ) -> str:
		''' Unicode-correct uppercase via LCMapStringEx - see
		PLAN_STR_UPPER_LOWER.md and _case_map_windows' own comment. '''
		from windows.kernel32 import LCMAP_UPPERCASE
		return self._case_map_windows( LCMAP_UPPERCASE )

	@compiler.target( os = not 'windows' )
	@private
	def _upper_os_native( self ) -> str:
		''' Unicode-correct (single-codepoint-mapping) uppercase via
		towupper_l - see PLAN_STR_UPPER_LOWER.md and _case_map_posix's own
		comment for what this does and doesn't cover. '''
		return self._case_map_posix( is_upper = True )

	def __cmp__( self, other: str ) -> i32:
		''' three-way comparison: -1 if self < other, 0 if equal, 1 if self > other '''
		min_len: usize = self.__byte_size if self.__byte_size < other.__byte_size else other.__byte_size
		result: i32 = sys.memcmp( self.__data, other.__data, min_len )
		if result != 0:
			return result
		if self.__byte_size < other.__byte_size:
			return -1
		if self.__byte_size > other.__byte_size:
			return 1
		return 0

	def __eq__( self, other: str ) -> bool:
		return self.__cmp__( other ) == 0

	def __ne__( self, other: str ) -> bool:
		return self.__cmp__( other ) != 0

	def __hash__( self ) -> u64:
		# content-based (never the pointer's own address) - two equal
		# strings must hash equally regardless of where each one lives, or
		# dict[K,V] breaks. RC types can't use the generic byte-hash
		# _fnv1a_hash's other callers rely on (dict[K,V]._hash_key for
		# non-RC K) - the object's own address isn't meaningful content -
		# so this is real, type-specific work only str itself can do
		return _fnv1a_hash( self.get_const_ptr(), self.byte_len() )

	def __lt__( self, other: str ) -> bool:
		return self.__cmp__( other ) < 0

	def __le__( self, other: str ) -> bool:
		return self.__cmp__( other ) <= 0

	def __gt__( self, other: str ) -> bool:
		return self.__cmp__( other ) > 0

	def __ge__( self, other: str ) -> bool:
		return self.__cmp__( other ) >= 0

	def lower( self ) -> str:
		''' mirrors upper()'s own case_folder-first check - see its comment. '''
		if case_folder.lower_count != 0:
			return case_folder.lower( self )
		return self._lower_os_native()

	@compiler.target( os = 'windows' )
	@private
	def _lower_os_native( self ) -> str:
		''' Unicode-correct lowercase via LCMapStringEx - see
		PLAN_STR_UPPER_LOWER.md and _case_map_windows' own comment. '''
		from windows.kernel32 import LCMAP_LOWERCASE
		return self._case_map_windows( LCMAP_LOWERCASE )

	@compiler.target( os = not 'windows' )
	@private
	def _lower_os_native( self ) -> str:
		''' Unicode-correct (single-codepoint-mapping) lowercase via
		towlower_l - see PLAN_STR_UPPER_LOWER.md and _case_map_posix's own
		comment for what this does and doesn't cover. '''
		return self._case_map_posix( is_upper = False )

	@private
	@staticmethod
	def _from_owned_cstr( ptr: Ptr[u8], byte_size_including_zero_terminator: usize ) -> Result[str,CodecError]:
		# NOTE: a 0-byte before the end of the string is valid utf-8, so we
		# can't use cstrlen() here, we can only check to make sure the terminating 0 exists where expected
		if byte_size_including_zero_terminator == 0:
			return Result.Err( CodecError( 'utf-8', 'empty buffer' ))
		byte_len: usize
		with compiler.wrap_arithmetic: # guaranteed to be > 0
			byte_len = byte_size_including_zero_terminator - 1
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
		return Result.Ok( s )

	@private
	@staticmethod
	def _decode_utf8_at( data: ConstPtr[u8], i: usize, consumed: Ptr[usize] ) -> u32:
		''' decodes one codepoint starting at data[i], writing the number of
		bytes consumed (1-4) to *consumed - used by upper()/lower()'s own
		POSIX path (towupper_l/towlower_l take one codepoint at a time,
		unlike Windows' whole-buffer LCMapStringEx). Assumes well-formed
		UTF-8 (str's own construction-time invariant - see this class's
		_from_owned_cstr, right above, which is the one place that's
		actually checked) - not re-validated here, same trust boundary
		byte_len()/__len__ above already rely on. '''
		byte1: u8 = data[i]
		if ( byte1 & 0x80 ) == 0x00:
			consumed[0] = 1
			with compiler.wrap_arithmetic:
				return u32( byte1 )
		if ( byte1 & 0xE0 ) == 0xC0:
			consumed[0] = 2
			with compiler.wrap_arithmetic:
				return ( u32( byte1 & 0x1F ) << 6 ) | u32( data[i+1] & 0x3F )
		if ( byte1 & 0xF0 ) == 0xE0:
			consumed[0] = 3
			with compiler.wrap_arithmetic:
				return ( u32( byte1 & 0x0F ) << 12 ) | ( u32( data[i+1] & 0x3F ) << 6 ) | u32( data[i+2] & 0x3F )
		# 4-byte sequence - the only shape left once 1/2/3-byte are ruled out
		consumed[0] = 4
		with compiler.wrap_arithmetic:
			return ( u32( byte1 & 0x07 ) << 18 ) | ( u32( data[i+1] & 0x3F ) << 12 ) | ( u32( data[i+2] & 0x3F ) << 6 ) | u32( data[i+3] & 0x3F )

	@private
	@staticmethod
	def _utf8_encoded_len( cp: u32 ) -> usize:
		''' how many UTF-8 bytes `cp` would take - the "size" half of the
		size-then-fill two-pass convention str.concat (above) already
		establishes, needed here because towupper_l/towlower_l can move a
		codepoint across a UTF-8 length boundary (e.g. U+00FF -> U+0178 is
		2 bytes -> 2 bytes, but plenty of other codepoints cross a
		boundary), so the output buffer can't just reuse the input's own
		byte size the way the old ASCII-only implementation did. '''
		if cp < 0x80:
			return 1
		if cp < 0x800:
			return 2
		if cp < 0x10000:
			return 3
		return 4

	@private
	@staticmethod
	def _encode_utf8_at( dest: Ptr[u8], i: usize, cp: u32 ) -> usize:
		''' encodes `cp` as UTF-8 into dest starting at dest[i], returns the
		number of bytes written (1-4) - the "fill" half, paired with
		_utf8_encoded_len just above. '''
		if cp < 0x80:
			with compiler.wrap_arithmetic:
				dest[i] = u8( cp )
			return 1
		if cp < 0x800:
			with compiler.wrap_arithmetic:
				dest[i]   = u8( 0xC0 | ( cp >> 6 ))
				dest[i+1] = u8( 0x80 | ( cp & 0x3F ))
			return 2
		if cp < 0x10000:
			with compiler.wrap_arithmetic:
				dest[i]   = u8( 0xE0 | ( cp >> 12 ))
				dest[i+1] = u8( 0x80 | (( cp >> 6 ) & 0x3F ))
				dest[i+2] = u8( 0x80 | ( cp & 0x3F ))
			return 3
		with compiler.wrap_arithmetic:
			dest[i]   = u8( 0xF0 | ( cp >> 18 ))
			dest[i+1] = u8( 0x80 | (( cp >> 12 ) & 0x3F ))
			dest[i+2] = u8( 0x80 | (( cp >> 6 ) & 0x3F ))
			dest[i+3] = u8( 0x80 | ( cp & 0x3F ))
		return 4

	@compiler.target( os = 'windows' )
	@private
	def _case_map_windows( self, flags: u32 ) -> str:
		''' shared by upper()/lower() on Windows - flags is LCMAP_UPPERCASE
		or LCMAP_LOWERCASE. Converts UTF-8 -> UTF-16 (MultiByteToWideChar),
		maps case on the whole UTF-16 buffer at once (LCMapStringEx), then
		converts back (WideCharToMultiByte). Every step follows the
		standard Win32 "call once with a null buffer to get the required
		size, allocate, call again to fill it" idiom - case mapping CAN
		change the byte length in general (surrogate-pair-widening
		codepoints, etc), even though it doesn't for any of the cases this
		implementation actually improves on over the old ASCII-only one.

		Confirmed by an actual compiled-and-run test, not just Win32 docs:
		LCMapStringEx (even with LCMAP_LINGUISTIC_CASING) only does SIMPLE
		(one-codepoint-in, one-codepoint-out) Unicode case mapping, the
		same ceiling _case_map_posix's towupper_l/towlower_l have - 'ß'
		stays 'ß' (not 'SS'), and Greek final sigma isn't applied ('ΣΊΣΥΦΟΣ'
		.lower() ends in plain 'σ', not the contextually-correct 'ς').
		Despite what PLAN_STR_UPPER_LOWER.md's own research suggested,
		there's no evidence LCMAP_LINGUISTIC_CASING implements Unicode's
		SpecialCasing.txt one-to-many/context-sensitive rules at all - only
		ICU reliably does, which this project deliberately isn't linking
		against (see crt.py's own comment on why). What this DOES still
		improve on over the old byte-range-only implementation: full
		simple-mapping coverage across every script Windows' own NLS
		Unicode data covers, not just ASCII 'a'-'z'/'A'-'Z'. The empty
		locale name (a bare zero u16, not NULL/LOCALE_NAME_USER_DEFAULT)
		requests locale-INVARIANT behavior - matching real Python's own
		str.upper()/lower(), which never depend on the process locale (no
		Turkish dotless-i surprises). '''
		from windows.kernel32 import MultiByteToWideChar, WideCharToMultiByte, LCMapStringEx, CP_UTF8, LCMAP_LINGUISTIC_CASING
		self_len: usize = self.byte_len()
		if self_len == 0:
			return str( self )
		src: ConstPtr[u8] = self.get_const_ptr()
		with compiler.panic_arithmetic( 'string too long for a Win32 API call' ):
			src_len: i32 = i32( self_len )

		wide_len: i32 = MultiByteToWideChar( CP_UTF8, 0, src, src_len, None, 0 )
		if wide_len <= 0:
			sys.panic( 'MultiByteToWideChar failed' )
		with compiler.panic_arithmetic( 'string too long for a Win32 API call' ):
			wide_buf: Ptr[u16] = sys.alloc[u16]( usize( wide_len ))
		defer( sys.free( wide_buf ))
		MultiByteToWideChar( CP_UTF8, 0, src, src_len, wide_buf, wide_len )

		empty_locale: u16 = 0
		map_flags: u32 = flags | LCMAP_LINGUISTIC_CASING
		mapped_len: i32 = LCMapStringEx( compiler.addrof( empty_locale ), map_flags, wide_buf, wide_len, None, 0, None, None, 0 )
		if mapped_len <= 0:
			sys.panic( 'LCMapStringEx failed' )
		with compiler.panic_arithmetic( 'string too long for a Win32 API call' ):
			mapped_buf: Ptr[u16] = sys.alloc[u16]( usize( mapped_len ))
		defer( sys.free( mapped_buf ))
		LCMapStringEx( compiler.addrof( empty_locale ), map_flags, wide_buf, wide_len, mapped_buf, mapped_len, None, None, 0 )

		out_len: i32 = WideCharToMultiByte( CP_UTF8, 0, mapped_buf, mapped_len, None, 0, None, None )
		if out_len <= 0:
			sys.panic( 'WideCharToMultiByte failed' )
		with compiler.panic_arithmetic( 'string too long for a Win32 API call' ):
			out_count: usize = usize( out_len )
			out_size: usize = out_count + 1
		out_buf: Ptr[u8] = sys.alloc[u8]( out_size )
		WideCharToMultiByte( CP_UTF8, 0, mapped_buf, mapped_len, out_buf, out_len, None, None )
		out_buf[out_count] = 0
		return str._from_owned_cstr( out_buf, out_size ).unwrap( 'invalid UTF-8 produced by LCMapStringEx' )

	@compiler.target( os = not 'windows' )
	@private
	def _case_map_posix( self, is_upper: bool ) -> str:
		''' shared by upper()/lower() on Linux/macOS/BSD - towupper_l/
		towlower_l only, no ICU (see PLAN_STR_UPPER_LOWER.md and crt.py's
		own comment on why): correctly cased single codepoints, but NOT
		one-to-many expansions (ß stays ß, not SS) or context-sensitive
		rules (Greek final sigma). newlocale('C.UTF-8') is looked up once
		per call (not cached process-wide) specifically to avoid the data
		race a shared/global locale object would need locking around - see
		crt.py's own comment; a NULL result (locale unavailable - possible
		on older systems) falls back to towupper_l/towlower_l's own
		"C"-locale behavior (ASCII-only, same ceiling the old
		implementation already had, everything else passed through
		unchanged) rather than failing outright.

		Two passes over the codepoints (see str.concat's own identical
		shape, above): the first sums the cased codepoints' own encoded
		byte lengths (which can differ from the input's - U+00E9 (2 bytes)
		uppercases to U+00C9 (also 2 bytes), but not everything stays
		put), the second actually encodes into the freshly, exactly sized
		buffer. '''
		from crt import newlocale, freelocale, LC_CTYPE_MASK
		self_len: usize = self.byte_len()
		if self_len == 0:
			return str( self )
		data: ConstPtr[u8] = self.get_const_ptr()
		loc: Ptr[None] = newlocale( LC_CTYPE_MASK, 'C.UTF-8'.get_cstr(), None )
		if loc is not None:
			defer( freelocale( loc ))

		new_size: usize = 1 # zero terminator
		i: usize = 0
		consumed: usize = 0
		while i < self_len:
			cp: u32 = str._decode_utf8_at( data, i, compiler.addrof( consumed ))
			cased: u32 = str._case_codepoint( cp, is_upper, loc )
			with compiler.panic_arithmetic( 'irrational string length' ):
				new_size += str._utf8_encoded_len( cased )
			with compiler.wrap_arithmetic:
				i += consumed

		new_buf: Ptr[u8] = sys.alloc[u8]( new_size )
		out_i: usize = 0
		i = 0
		while i < self_len:
			cp = str._decode_utf8_at( data, i, compiler.addrof( consumed ))
			cased = str._case_codepoint( cp, is_upper, loc )
			with compiler.wrap_arithmetic:
				out_i += str._encode_utf8_at( new_buf, out_i, cased )
				i += consumed
		new_buf[out_i] = 0

		return str._from_owned_cstr( new_buf, new_size ).unwrap( 'invalid UTF-8 produced by towupper_l/towlower_l' )

	@compiler.target( os = not 'windows' )
	@private
	@staticmethod
	def _case_codepoint( cp: u32, is_upper: bool, loc: Ptr[None] ) -> u32:
		''' cases one codepoint via towupper_l/towlower_l, or passes it
		through unchanged if loc is None (newlocale('C.UTF-8') failed -
		see _case_map_posix's own comment on why that's a graceful
		fallback, not a hard error). '''
		if loc is None:
			return cp
		from crt import towupper_l, towlower_l
		with compiler.panic_arithmetic( 'codepoint out of range for wint_t - impossible for valid Unicode (max U+10FFFF)' ):
			wc: i32 = i32( cp )
		cased: i32
		if is_upper:
			cased = towupper_l( wc, loc )
		else:
			cased = towlower_l( wc, loc )
		with compiler.panic_arithmetic( 'towupper_l/towlower_l returned a negative/out-of-range codepoint' ):
			return u32( cased )


# ---------------------------------------------------------------------------
# case_folder: str.upper()/str.lower()'s install()-able full-Unicode-casing
# hook (see PLAN_CASE_FOLDING.md). CaseFolding is a single concrete class,
# not something meant to be subclassed - this compiler has no virtual/
# polymorphic method dispatch (confirmed by direct inspection: method calls
# always resolve from the receiver's static declared type, ir.Call always
# targets one fixed C symbol, no vtable concept exists anywhere), so
# "install() swaps behavior" is realized by swapping DATA on this one
# shared global instance, not by swapping which class's methods run.
# case_folding.py (a separate, optional module nothing here ever imports)
# calls install() to point upper_table/upper_count/lower_table/lower_count
# at its own embedded Unicode table; a program that never imports
# case_folding never references that table at all, so it's never linked
# in - the ONLY cost every program pays is the one `if upper_count != 0`/
# `if lower_count != 0` check in str.upper()/str.lower() above.
# ---------------------------------------------------------------------------

@cstruct
class CaseFolding:
	# a value type (embedded inline in the global below), not an RCClass
	# (heap-allocated, referenced through a pointer) - deliberately, to
	# sidestep a real, documented, pre-existing gap: a global whose
	# initializer needs real computation (any RCClass construction) gets
	# its own __metalpy_init_<name>() function generated, but nothing
	# ever calls it (see emitter_c.py's emit_global, "Wiring this init
	# function into a real process entry point is out of scope... it
	# just needs to exist and compile") - case_folder was silently a null
	# pointer forever, and dereferencing it segfaulted on the very first
	# .upper()/.lower() call. A cstruct global instead gets C's own
	# static {0} zero-initializer for its own fields DIRECTLY (no pointer,
	# no heap allocation, nothing to run before it's valid) - exactly the
	# "not installed" state this needs by default, with no init call
	# required at all.
	#
	# upper_table/lower_table are RAW BYTES (ConstPtr[u8]), not a typed
	# Ptr[SomeEntryStruct] - this compiler has no pointer-reinterpretation
	# primitive (compiler.reinterpret_cast[T] is listed in TODO.txt as
	# never implemented) and no array-literal syntax to construct a typed
	# static table from metalpy source directly. Sidestepped the same way
	# str.upper()'s own UTF-8 decode/encode already does: read individual
	# bytes and reconstruct u32s by hand (_read_u32_le, below) - each
	# table entry is exactly 8 bytes (codepoint: u32 LE, mapped: u32 LE),
	# sorted by codepoint for binary search, *_count is the number of
	# 8-byte entries (not the byte length) - see compiler.
	# fetch_unicode_table()'s own docstring in lowering.py for how the
	# bytes themselves get produced/embedded.
	upper_table: ConstPtr[u8]
	upper_count: usize
	lower_table: ConstPtr[u8]
	lower_count: usize

	def upper( self, s: str ) -> str:
		return self._map( s, self.upper_table, self.upper_count )

	def lower( self, s: str ) -> str:
		return self._map( s, self.lower_table, self.lower_count )

	@private
	def _map( self, s: str, table: ConstPtr[u8], count: usize ) -> str:
		# str.upper()/lower() only ever call upper()/lower() above once
		# their own respective *_count != 0, so table/count here are
		# always real - no "not installed" fallback needed. Otherwise
		# mirrors _case_map_posix's own two-pass (size, then fill) shape
		# almost exactly, just consulting this table instead of towupper_l
		# /towlower_l - see that method's own comment for why two passes
		self_len: usize = s.byte_len()
		if self_len == 0:
			return str( s )
		data: ConstPtr[u8] = s.get_const_ptr()

		new_size: usize = 1 # zero terminator
		i: usize = 0
		consumed: usize = 0
		while i < self_len:
			cp: u32 = str._decode_utf8_at( data, i, compiler.addrof( consumed ))
			mapped: u32 = CaseFolding._lookup( table, count, cp )
			with compiler.panic_arithmetic( 'irrational string length' ):
				new_size += str._utf8_encoded_len( mapped )
			with compiler.wrap_arithmetic:
				i += consumed

		new_buf: Ptr[u8] = sys.alloc[u8]( new_size )
		out_i: usize = 0
		i = 0
		while i < self_len:
			cp = str._decode_utf8_at( data, i, compiler.addrof( consumed ))
			mapped = CaseFolding._lookup( table, count, cp )
			with compiler.wrap_arithmetic:
				out_i += str._encode_utf8_at( new_buf, out_i, mapped )
				i += consumed
		new_buf[out_i] = 0

		return str._from_owned_cstr( new_buf, new_size ).unwrap( 'invalid UTF-8 produced by case_folding table lookup' )

	@private
	@staticmethod
	def _lookup( table: ConstPtr[u8], count: usize, cp: u32 ) -> u32:
		# binary search over the 8-bytes-per-entry (codepoint, mapped)
		# table, sorted ascending by codepoint - a codepoint with no entry
		# (the overwhelming majority - most codepoints have no case at
		# all) passes through unchanged, same posture as the OS-backed
		# _case_map_windows/_case_map_posix paths
		lo: usize = 0
		hi: usize = count
		while lo < hi:
			# // (FloorDiv) is always checked against ZeroDivisionError, in
			# every arithmetic mode (see arithmetic_mode.py's own comment on
			# ArithmeticChecked.GetBinOp) - panic_arithmetic here is that
			# check's escape hatch, not overflow protection; lo < hi (the
			# while condition) already guarantees hi - lo > 0
			with compiler.panic_arithmetic( 'unreachable: hi > lo in binary search' ):
				mid: usize = lo + ( hi - lo ) // 2
				offset: usize = mid * 8
			entry_cp: u32 = CaseFolding._read_u32_le( table, offset )
			if entry_cp == cp:
				with compiler.wrap_arithmetic:
					mapped_offset: usize = offset + 4
				return CaseFolding._read_u32_le( table, mapped_offset )
			if entry_cp < cp:
				with compiler.wrap_arithmetic:
					lo = mid + 1
			else:
				hi = mid
		return cp

	@private
	@staticmethod
	def _read_u32_le( data: ConstPtr[u8], i: usize ) -> u32:
		with compiler.wrap_arithmetic:
			return u32( data[i] ) | ( u32( data[i+1] ) << 8 ) | ( u32( data[i+2] ) << 16 ) | ( u32( data[i+3] ) << 24 )

case_folder: CaseFolding = CaseFolding(
	upper_table = None, upper_count = 0,
	lower_table = None, lower_count = 0,
)

def print( msg: str, end: str = '\n' ) -> None:
	# No *args/**kwargs, use f-strings instead (once implemented)
	sys.stdout.write( msg ).unwrap( 'stdout write failed' )
	if end:
		sys.stdout.write( end ).unwrap( 'stdout write failed' )

# a single generic function now that bare-call monomorphization can infer T
# from the argument (see lowering.py's _lower_inferred_generic_call) - a real
# Call for now, forwarding to whatever T's own __len__ is; TODO once @inline
# exists (see TODO.txt): this should become @inline so len(x) compiles down
# to the same code as x.__len__() directly, no call overhead.

def len[T]( t: T ) -> usize:
	return t.__len__()

def chr( cp: u32 ) -> str:
	''' the inverse of ord() below - encodes a single Unicode code point as
	UTF-8 into a freshly allocated buffer, then constructs a str from it -
	reuses str's own private UTF-8 size/encode helpers (the same ones
	upper()/lower()/case-folding already build on) rather than duplicating
	that logic here. cp must be a real Unicode code point (<= U+10FFFF,
	not a UTF-16 surrogate half) - _from_owned_cstr's own UTF-8
	revalidation (the same one every other str-constructing method here
	already goes through) catches an invalid one, panicking via unwrap()
	rather than returning a Result: Python's real chr() raises ValueError
	for this, and this language panics instead of raising exceptions
	throughout (see str.split's identical reasoning for its own
	non-empty-separator precondition). '''
	encoded_len: usize = str._utf8_encoded_len( cp )
	buf_size: usize
	with compiler.panic_arithmetic( 'irrational string length' ):
		buf_size = encoded_len + 1 # +1 for the zero terminator
	buf: Ptr[u8] = sys.alloc[u8]( buf_size )
	str._encode_utf8_at( buf, 0, cp )
	buf[encoded_len] = 0
	return str._from_owned_cstr( buf, buf_size ).unwrap( 'chr(): not a valid Unicode code point' )

def ord( s: str ) -> u32:
	''' the inverse of chr() above - decodes s's own first (and only) code
	point back to its integer value, reusing str's own private UTF-8
	decode helper. Requires s to be exactly one code point long, matching
	Python's own ord() (TypeError otherwise) - panics instead, same
	panic-not-exceptions convention chr() above follows. '''
	if s.byte_len() == 0:
		sys.panic( 'ord(): expected a string of length 1, got an empty string' )
	consumed: usize = 0
	cp: u32 = str._decode_utf8_at( s.get_const_ptr(), 0, compiler.addrof( consumed ))
	if consumed != s.byte_len():
		sys.panic( 'ord(): expected a string of length 1, got a longer string' )
	return cp

class dict[K, V]:
	''' see PLAN_CALLABLE.md. RawDict (lib/builtins/__RawDict.py) is
	genuinely type-erased - it never decodes a key_ptr/value_ptr back to a
	real K/V, never allocates/frees/increfs/decrefs one, never computes a
	hash. Every method below that branches on compiler.is_rc(K)/
	compiler.is_rc(V) is a @staticmethod for exactly one reason: _key_eq
	needs to be referenced BARE (no receiver - see lowering.py's
	_lower_function_ref) to hand its address to RawDict as a real
	Ptr[Callable[...]] value, and the helpers it calls (_borrow_key) have
	to be static too so a bare reference from within a static method can
	reach them the same way. '''
	__raw: RawDict

	def __init__( self ) -> None:
		self.__raw = RawDict()

	def __len__( self ) -> usize:
		return len( self.__raw )

	def __del__( self ) -> None:
		# RawDict itself owns no K/V-shaped resources (see its own module
		# docstring) - releasing every stored key/value is entirely
		# dict[K,V]'s own job, done here directly (not through a callback -
		# __del__ runs once, in THIS monomorphized class's own code, no
		# erasure boundary to cross)
		i: usize = 0
		with compiler.panic_arithmetic( 'dict.__del__: overflow' ):
			while i < len( self.__raw ):
				self._release_key( self.__raw.key_ptr_at( i ))
				self._release_value( self.__raw.value_ptr_at( i ))
				i += 1

	# --- key/value ownership - only these know K/V's own RC-ness ---------

	@staticmethod
	def _owned_value( value_ptr: Ptr[None] ) -> V:
		# returned OUT to the caller (__getitem__) - an RC value needs its
		# own incref (the dict's own stored reference stays valid too)
		if compiler.is_rc( V ):
			v: V = compiler.cast( V, value_ptr )
			compiler.incref( v )
			return v
		else:
			ptr: Ptr[V] = compiler.cast( Ptr[V], value_ptr )
			return ptr[0]

	@staticmethod
	def _store_key( key: K ) -> Ptr[None]:
		# an OWNED copy for RawEntry to hold onto indefinitely - an RC key
		# just gets increfed (the object is already heap-owned, storing
		# its handle is enough); a value-typed key needs a real heap copy,
		# since RawEntry can't hold its bytes inline (it works on opaque
		# Ptr[None], see its own module docstring)
		if compiler.is_rc( K ):
			compiler.incref( key )
			return compiler.cast( Ptr[None], key )
		else:
			buf: Ptr[None] = compiler.cast( Ptr[None], sys.alloc[u8]( compiler.sizeof( K )))
			ptr: Ptr[K] = compiler.cast( Ptr[K], buf )
			ptr[0] = key
			return buf

	@staticmethod
	def _store_value( value: V ) -> Ptr[None]:
		if compiler.is_rc( V ):
			compiler.incref( value )
			return compiler.cast( Ptr[None], value )
		else:
			buf: Ptr[None] = compiler.cast( Ptr[None], sys.alloc[u8]( compiler.sizeof( V )))
			ptr: Ptr[V] = compiler.cast( Ptr[V], buf )
			ptr[0] = value
			return buf

	@staticmethod
	def _release_key( key_ptr: Ptr[None] ) -> None:
		if compiler.is_rc( K ):
			existing: K = compiler.cast( K, key_ptr )
			compiler.decref( existing )
		else:
			sys.free( compiler.cast( Ptr[u8], key_ptr ))

	@staticmethod
	def _release_value( value_ptr: Ptr[None] ) -> None:
		if compiler.is_rc( V ):
			existing: V = compiler.cast( V, value_ptr )
			compiler.decref( existing )
		else:
			sys.free( compiler.cast( Ptr[u8], value_ptr ))

	# --- hashing/equality - the only crossing into RawDict's own code ----

	@staticmethod
	def _key_eq( a: Ptr[None], b: Ptr[None] ) -> bool:
		# this is the function whose ADDRESS gets handed to RawDict as a
		# real Ptr[Callable[[Ptr[None],Ptr[None]],bool]] value (see
		# __getitem__/__setitem__ below) - RawDict calls it deep inside
		# its own hash-collision scan without ever knowing what K is.
		# Inlined rather than sharing a _borrow_key helper: a bare call to
		# a SIBLING static method of this same generic class, from within
		# another static method (no receiver to pin the class's own type
		# args), hits a separate, unrelated generic-inference gap - see
		# lowering.py's _lower_function_ref for the bare-REFERENCE version
		# of this same class of problem, which IS handled (this method's
		# own address, taken from __getitem__/__setitem__ below, relies on
		# exactly that fix)
		if compiler.is_rc( K ):
			ka: K = compiler.cast( K, a )
			kb: K = compiler.cast( K, b )
			return ka == kb
		else:
			pa: Ptr[K] = compiler.cast( Ptr[K], a )
			pb: Ptr[K] = compiler.cast( Ptr[K], b )
			return pa[0] == pb[0]

	def _hash_key( self, key: K ) -> u64:
		if compiler.is_rc( K ):
			return key.__hash__()
		else:
			# a generic byte hash works uniformly for ANY value-typed K
			# (scalars, or a plain @cstruct of scalars) - equality for
			# those is fundamentally byte/value-based already. An RC key's
			# own address isn't meaningful content, so it can't share this
			# path - see str.__hash__'s own comment
			with compiler.panic_arithmetic( 'dict._hash_key: address-of overflow' ):
				ptr: ConstPtr[u8] = compiler.cast( ConstPtr[u8], compiler.addrof( key ))
			return _fnv1a_hash( ptr, compiler.sizeof( K ))

	def __getitem__( self, key: K ) -> Result[V, KeyError]:
		h: u64 = self._hash_key( key )
		# BORROWED search key - taken directly here (not through a sub-
		# call) so the address stays valid for exactly as long as this
		# synchronous call needs it, same constraint compiler.addrof(...)
		# already documents (bare local variable only)
		key_ptr: Ptr[None] = 0
		if compiler.is_rc( K ):
			key_ptr = compiler.cast( Ptr[None], key )
		else:
			key_ptr = compiler.cast( Ptr[None], compiler.addrof( key ))
		entry_idx: usize = self.__raw._find_entry_idx( h, key_ptr, _key_eq ).or_return()
		return Result.Ok( self._owned_value( self.__raw.value_ptr_at( entry_idx )))

	def __setitem__( self, key: K, value: V ) -> None:
		h: u64 = self._hash_key( key )
		key_ptr: Ptr[None] = 0
		if compiler.is_rc( K ):
			key_ptr = compiler.cast( Ptr[None], key )
		else:
			key_ptr = compiler.cast( Ptr[None], compiler.addrof( key ))
		found: Result[usize, KeyError] = self.__raw._find_entry_idx( h, key_ptr, _key_eq )
		match found:
			case Result.Ok( entry_idx ):
				new_value_ptr: Ptr[None] = self._store_value( value )
				old_value_ptr: Ptr[None] = self.__raw.overwrite_value_at( entry_idx, new_value_ptr )
				self._release_value( old_value_ptr )
			case Result.Err( _ ):
				owned_key_ptr: Ptr[None] = self._store_key( key )
				owned_value_ptr: Ptr[None] = self._store_value( value )
				self.__raw.insert_new( h, owned_key_ptr, owned_value_ptr )

# import this at the end because it depends on str etc to already be pre-parsed:
from .__File import File
