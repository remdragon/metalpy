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

	@compiler.target( os = 'windows' )
	def upper( self ) -> str:
		''' Unicode-correct uppercase via LCMapStringEx - see
		PLAN_STR_UPPER_LOWER.md and _case_map_windows' own comment. '''
		from windows.kernel32 import LCMAP_UPPERCASE
		return self._case_map_windows( LCMAP_UPPERCASE )

	@compiler.target( os = not 'windows' )
	def upper( self ) -> str:
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

	def __lt__( self, other: str ) -> bool:
		return self.__cmp__( other ) < 0

	def __le__( self, other: str ) -> bool:
		return self.__cmp__( other ) <= 0

	def __gt__( self, other: str ) -> bool:
		return self.__cmp__( other ) > 0

	def __ge__( self, other: str ) -> bool:
		return self.__cmp__( other ) >= 0

	@compiler.target( os = 'windows' )
	def lower( self ) -> str:
		''' Unicode-correct lowercase via LCMapStringEx - see
		PLAN_STR_UPPER_LOWER.md and _case_map_windows' own comment. '''
		from windows.kernel32 import LCMAP_LOWERCASE
		return self._case_map_windows( LCMAP_LOWERCASE )

	@compiler.target( os = not 'windows' )
	def lower( self ) -> str:
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

def print( msg: str, end: str = '\n' ) -> None:
	# No *args/**kwargs, use f-strings instead (once implemented)
	# .is_ok() not .unwrap(...) - Result[None,OSError].unwrap() hits an
	# unrelated, pre-existing emitter_c.py bug (a T=None unwrap() compiles
	# to a C function declared void that still tries to `return` a value -
	# separate issue, not touched here); print() swallowing a write failure
	# is no worse than before this session's unchecked-Result check made
	# the OLD silent-discard here a compile error in the first place
	sys.stdout.write( msg ).is_ok()
	if end:
		sys.stdout.write( end ).is_ok()

# a single generic function now that bare-call monomorphization can infer T
# from the argument (see lowering.py's _lower_inferred_generic_call) - a real
# Call for now, forwarding to whatever T's own __len__ is; TODO once @inline
# exists (see TODO.txt): this should become @inline so len(x) compiles down
# to the same code as x.__len__() directly, no call overhead.

def len[T]( t: T ) -> usize:
	return t.__len__()

# import this at the end because it depends on str etc to already be pre-parsed:
from .__File import File
