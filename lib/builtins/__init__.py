from codecs import Codec, CodecError
from codecs.utf8 import utf8
import compiler
import sys
import threading
from .__errors import OSError
from .__fastlist import FastList
from .__int import int, IntError
from .__list import list, UnsafeList
from .__RawDict import RawDict
from .__str import decode_utf8_at, encode_utf8_at, utf8_encoded_len, case_map

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

	def rsplit( self, sep: str ) -> list[str]:
		''' splits self on every occurrence of sep, same semantics as
		split() above - without a maxsplit limit (not implemented here,
		same scope split() itself has), splitting from the right produces
		the exact same list as splitting from the left, just conceptually
		built in the other direction, so this is a direct call-through
		rather than a separate scan. '''
		return self.split( sep )

	def startswith( self, prefix: str, start: usize = 0 ) -> bool:
		''' True if self, starting at byte offset start, begins with
		prefix - mirrors find()'s own start parameter. An empty prefix
		always matches (same vacuously-true convention find('') uses),
		matching Python. '''
		self_len: usize = self.byte_len()
		prefix_len: usize = prefix.byte_len()
		if start > self_len:
			return False
		if prefix_len == 0:
			return True
		with compiler.wrap_arithmetic: # start <= self_len, just checked above
			remaining: usize = self_len - start
		if prefix_len > remaining:
			return False
		with compiler.wrap_arithmetic: # start bounded by self_len above
			candidate: ConstPtr[u8] = self.__data + start
		return sys.memcmp( candidate, prefix.__data, prefix_len ) == 0

	def endswith( self, suffix: str ) -> bool:
		''' True if self ends with suffix. An empty suffix always
		matches, matching Python. '''
		self_len: usize = self.byte_len()
		suffix_len: usize = suffix.byte_len()
		if suffix_len == 0:
			return True
		if suffix_len > self_len:
			return False
		with compiler.wrap_arithmetic: # suffix_len <= self_len, just checked above
			offset: usize = self_len - suffix_len
			candidate: ConstPtr[u8] = self.__data + offset
		return sys.memcmp( candidate, suffix.__data, suffix_len ) == 0

	def removeprefix( self, prefix: str ) -> str:
		''' self with prefix removed if present, else an unchanged copy -
		matches Python's str.removeprefix(). '''
		if not self.startswith( prefix ):
			return str( self )
		return self._byte_slice( prefix.byte_len(), self.byte_len() )

	def removesuffix( self, suffix: str ) -> str:
		''' self with suffix removed if present, else an unchanged copy -
		matches Python's str.removesuffix(). '''
		if not self.endswith( suffix ):
			return str( self )
		with compiler.wrap_arithmetic: # suffix_len <= self_len, endswith() just confirmed it
			end: usize = self.byte_len() - suffix.byte_len()
		return self._byte_slice( 0, end )

	def rfind( self, sub: str ) -> Result[usize,IndexError]:
		''' byte offset of the LAST occurrence of sub in self - mirrors
		find() above exactly, just scanning from the end. Empty sub
		matches at self's own end (self_len), the mirror image of
		find('')'s own vacuous match at offset 0. '''
		self_len: usize = self.byte_len()
		sub_len: usize = sub.byte_len()
		if sub_len == 0:
			return Result.Ok( self_len )
		if sub_len > self_len:
			return Result.Err( IndexError() )
		with compiler.wrap_arithmetic: # sub_len <= self_len, just checked above
			i: usize = self_len - sub_len
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			while True:
				candidate: ConstPtr[u8] = self.__data + i
				if sys.memcmp( candidate, sub.__data, sub_len ) == 0:
					return Result.Ok( i )
				if i == 0:
					break
				i -= 1
		return Result.Err( IndexError() )

	def rindex( self, sub: str ) -> usize:
		''' like rfind(), but panics instead of returning Err - mirrors
		index()'s own relationship to find() above. '''
		return self.rfind( sub ).unwrap( 'substring not found' )

	def replace( self, old: str, new: str ) -> str:
		''' every occurrence of old replaced with new - two-pass (count
		occurrences and compute the exact output size, then fill), the
		same shape __str.py's case_map already uses. old must not be
		empty (same convention split() above already established for an
		empty separator). '''
		if old.byte_len() == 0:
			sys.panic( 'str.replace(...): old must not be empty' )
		self_len: usize = self.byte_len()
		old_len: usize = old.byte_len()
		new_len: usize = new.byte_len()

		occurrences: usize = 0
		start: usize = 0
		while True:
			found: Result[usize,IndexError] = self.find( old, start )
			if found.is_err():
				break
			match_start: usize = found.unwrap( 'unreachable: find() confirmed is_ok' )
			with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
				occurrences += 1
				start = match_start + old_len

		if occurrences == 0:
			return str( self )

		new_size: usize = 1 # zero terminator
		with compiler.panic_arithmetic( 'irrational string length' ):
			removed: usize = occurrences * old_len
			added: usize = occurrences * new_len
		with compiler.wrap_arithmetic: # removed <= self_len - occurrences counted from real non-overlapping matches
			kept: usize = self_len - removed
		with compiler.panic_arithmetic( 'irrational string length' ):
			new_size += kept + added

		buf: Ptr[u8] = sys.alloc[u8]( new_size )
		out: usize = 0
		start = 0
		while True:
			found = self.find( old, start )
			if found.is_err():
				break
			match_start = found.unwrap( 'unreachable: find() confirmed is_ok' )
			with compiler.wrap_arithmetic:
				piece_len: usize = match_start - start
				sys.memcpy( buf + out, self.__data + start, piece_len )
				out += piece_len
				sys.memcpy( buf + out, new.__data, new_len )
				out += new_len
				start = match_start + old_len

		with compiler.wrap_arithmetic:
			tail_len: usize = self_len - start
			sys.memcpy( buf + out, self.__data + start, tail_len )
			out += tail_len
		buf[out] = 0

		return str._from_owned_cstr( buf, new_size ).unwrap( 'invalid UTF-8 in replace' )

	def join( self, parts: list[str] ) -> str:
		''' self inserted between each element of parts - str.concat's own
		two-pass shape, plus self's own bytes between consecutive parts.
		Takes list[str] rather than slice[str] (str.concat's own parameter
		type) - slice[T] has no real construction path from ordinary
		metalpy source anywhere in this codebase yet (no array-literal
		syntax - see CaseFolding's own upper_table comment), while
		list[str] is the container every caller already has a piece of
		text collection in (e.g. split()'s own return type). '''
		count: usize = parts.__len__()
		if count == 0:
			return str( '' )
		self_len: usize = self.byte_len()
		new_size: usize = 1 # zero terminator
		i: usize = 0
		for i in range( count ):
			part: str = parts.__getitem__( i ).unwrap( 'str.join: index in bounds by construction' )
			with compiler.panic_arithmetic( 'irrational string length' ):
				new_size += part.byte_len()
		with compiler.panic_arithmetic( 'irrational string length' ):
			new_size += self_len * ( count - 1 ) # count-1 separators between count parts

		new_buf: Ptr[u8] = sys.alloc[u8]( new_size )
		offset: usize = 0
		for i in range( count ):
			if i != 0:
				with compiler.wrap_arithmetic:
					sys.memcpy( new_buf + offset, self.__data, self_len )
					offset += self_len
			part = parts.__getitem__( i ).unwrap( 'str.join: index in bounds by construction' )
			part_len: usize = part.byte_len()
			with compiler.wrap_arithmetic:
				sys.memcpy( new_buf + offset, part.__data, part_len )
				offset += part_len

		new_buf[offset] = 0
		return str._from_owned_cstr( new_buf, new_size ).unwrap( 'invalid UTF-8 in join' )

	def partition( self, sep: str ) -> tuple[str,str,str]:
		''' splits self at the FIRST occurrence of sep into (before, sep,
		after) - a real tuple[str,str,str] (PLAN_TUPLE.md), matching
		Python's own str.partition() return shape exactly. sep not found:
		(self, '', '') - Python's own no-match convention (the whole
		string stays on the side the search started from). '''
		found: Result[usize,IndexError] = self.find( sep )
		match found:
			case Result.Ok( idx ):
				self_len: usize = self.byte_len()
				with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
					after_start: usize = idx + sep.byte_len()
				return ( self._byte_slice( 0, idx ), sep, self._byte_slice( after_start, self_len ))
			case Result.Err( _ ):
				return ( str( self ), str( '' ), str( '' ))

	def rpartition( self, sep: str ) -> tuple[str,str,str]:
		''' like partition() above, but splits at the LAST occurrence of
		sep. Not found: ('', '', self) - the mirror image of partition()'s
		own no-match convention (rpartition searches from the end, so the
		whole string stays there). '''
		found: Result[usize,IndexError] = self.rfind( sep )
		match found:
			case Result.Ok( idx ):
				self_len: usize = self.byte_len()
				with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
					after_start: usize = idx + sep.byte_len()
				return ( self._byte_slice( 0, idx ), sep, self._byte_slice( after_start, self_len ))
			case Result.Err( _ ):
				return ( str( '' ), str( '' ), str( self ))

	def isascii( self ) -> bool:
		''' True if every byte is < 0x80 - vacuously True for an empty
		string (matches Python). Byte-level only, no codepoint decoding
		needed: ASCII-ness is a UTF-8-byte-level property, not a Unicode
		classification concern (see __str.py's classification primitives,
		used by isalpha()/isdigit()/etc instead). '''
		byte_len: usize = self.byte_len()
		i: usize = 0
		with compiler.panic_arithmetic( 'bounded by byte_len, cannot overflow' ):
			while i < byte_len:
				if self.__data[i] >= 0x80:
					return False
				i += 1
		return True

	def ljust( self, width: usize, fillchar: str = ' ' ) -> str:
		''' pads self on the RIGHT with fillchar until self's own codepoint
		count reaches width - matches Python's str.ljust(). Already-long-
		enough is a no-op (returns an unchanged copy). fillchar must be
		exactly one codepoint, panicking otherwise - Python itself raises
		TypeError for the same len(fillchar) != 1 case. '''
		if len( fillchar ) != 1:
			sys.panic( 'str.ljust(...): fillchar must be exactly one character' )
		self_count: usize = self.__len__()
		if self_count >= width:
			return str( self )
		with compiler.panic_arithmetic( 'bounded by width, cannot overflow' ):
			pad_count: usize = width - self_count
		self_len: usize = self.byte_len()
		fill_len: usize = fillchar.byte_len()
		with compiler.panic_arithmetic( 'irrational string length' ):
			pad_bytes: usize = pad_count * fill_len
			new_size: usize = self_len + pad_bytes + 1
		new_buf: Ptr[u8] = sys.alloc[u8]( new_size )
		sys.memcpy( new_buf, self.__data, self_len )
		offset: usize = self_len
		i: usize = 0
		while i < pad_count:
			with compiler.wrap_arithmetic:
				sys.memcpy( new_buf + offset, fillchar.__data, fill_len )
				offset += fill_len
				i += 1
		new_buf[offset] = 0
		return str._from_owned_cstr( new_buf, new_size ).unwrap( 'invalid UTF-8 in ljust' )

	def rjust( self, width: usize, fillchar: str = ' ' ) -> str:
		''' pads self on the LEFT with fillchar until self's own codepoint
		count reaches width - the mirror image of ljust() above, same
		fillchar/no-op/panic conventions. '''
		if len( fillchar ) != 1:
			sys.panic( 'str.rjust(...): fillchar must be exactly one character' )
		self_count: usize = self.__len__()
		if self_count >= width:
			return str( self )
		with compiler.panic_arithmetic( 'bounded by width, cannot overflow' ):
			pad_count: usize = width - self_count
		self_len: usize = self.byte_len()
		fill_len: usize = fillchar.byte_len()
		with compiler.panic_arithmetic( 'irrational string length' ):
			pad_bytes: usize = pad_count * fill_len
			new_size: usize = self_len + pad_bytes + 1
		new_buf: Ptr[u8] = sys.alloc[u8]( new_size )
		offset: usize = 0
		i: usize = 0
		while i < pad_count:
			with compiler.wrap_arithmetic:
				sys.memcpy( new_buf + offset, fillchar.__data, fill_len )
				offset += fill_len
				i += 1
		with compiler.wrap_arithmetic:
			sys.memcpy( new_buf + offset, self.__data, self_len )
			offset += self_len
		new_buf[offset] = 0
		return str._from_owned_cstr( new_buf, new_size ).unwrap( 'invalid UTF-8 in rjust' )

	def zfill( self, width: usize ) -> str:
		''' like rjust(width, '0'), except a leading '+'/'-' byte stays
		first, with the zero padding inserted right after it - matches
		Python's '-42'.zfill(5) == '-0042'. A leading sign is always a
		single ASCII byte (0x2B/0x2D), never a multi-byte codepoint, so
		checking self.__data[0] directly (rather than decoding a
		codepoint) is exact, not an approximation. '''
		self_len: usize = self.byte_len()
		has_sign: bool = self_len > 0 and ( self.__data[0] == 0x2B or self.__data[0] == 0x2D ) # '+' or '-'
		self_count: usize = self.__len__()
		if self_count >= width:
			return str( self )
		with compiler.panic_arithmetic( 'bounded by width, cannot overflow' ):
			pad_count: usize = width - self_count
		with compiler.panic_arithmetic( 'irrational string length' ):
			new_size: usize = self_len + pad_count + 1
		new_buf: Ptr[u8] = sys.alloc[u8]( new_size )
		offset: usize = 0
		body_start: usize = 0
		if has_sign:
			new_buf[0] = self.__data[0]
			offset = 1
			body_start = 1
		i: usize = 0
		while i < pad_count:
			new_buf[offset] = 0x30 # '0'
			with compiler.wrap_arithmetic:
				offset += 1
				i += 1
		with compiler.wrap_arithmetic:
			rest_len: usize = self_len - body_start
			sys.memcpy( new_buf + offset, self.__data + body_start, rest_len )
			offset += rest_len
		new_buf[offset] = 0
		return str._from_owned_cstr( new_buf, new_size ).unwrap( 'invalid UTF-8 in zfill' )

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

	@private
	def _upper_os_native( self ) -> str:
		''' Unicode-correct uppercase (LCMapStringEx on Windows, towupper_l
		on POSIX) - the actual OS-specific work lives in __str.py's
		case_map (not str-specific, see its own comment there, and its own
		@compiler.target-gated pair of definitions - this method itself
		doesn't need to be OS-gated at all anymore, case_map already
		resolves to whichever one matches the current build); this just
		supplies str's own raw bytes and wraps the returned buffer back
		into a real str. '''
		out_size: usize = 0
		out_buf: Ptr[u8] = case_map( self.get_const_ptr(), self.byte_len(), True, compiler.addrof( out_size ))
		return str._from_owned_cstr( out_buf, out_size ).unwrap( 'invalid UTF-8 produced by case_map' )

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

	@private
	def _lower_os_native( self ) -> str:
		''' Unicode-correct lowercase (LCMapStringEx on Windows, towlower_l
		on POSIX) - the actual OS-specific work lives in __str.py's
		case_map (not str-specific, see its own comment there, and its own
		@compiler.target-gated pair of definitions - this method itself
		doesn't need to be OS-gated at all anymore, case_map already
		resolves to whichever one matches the current build); this just
		supplies str's own raw bytes and wraps the returned buffer back
		into a real str. '''
		out_size: usize = 0
		out_buf: Ptr[u8] = case_map( self.get_const_ptr(), self.byte_len(), False, compiler.addrof( out_size ))
		return str._from_owned_cstr( out_buf, out_size ).unwrap( 'invalid UTF-8 produced by case_map' )

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
		# mirrors __str.py's own POSIX case_map two-pass (size, then fill)
		# shape almost exactly, just consulting this table instead of
		# towupper_l/towlower_l - see that function's own comment for why
		# two passes
		self_len: usize = s.byte_len()
		if self_len == 0:
			return str( s )
		data: ConstPtr[u8] = s.get_const_ptr()

		new_size: usize = 1 # zero terminator
		i: usize = 0
		consumed: usize = 0
		while i < self_len:
			cp: u32 = decode_utf8_at( data, i, compiler.addrof( consumed ))
			mapped: u32 = CaseFolding._lookup( table, count, cp )
			with compiler.panic_arithmetic( 'irrational string length' ):
				new_size += utf8_encoded_len( mapped )
			with compiler.wrap_arithmetic:
				i += consumed

		new_buf: Ptr[u8] = sys.alloc[u8]( new_size )
		out_i: usize = 0
		i = 0
		while i < self_len:
			cp = decode_utf8_at( data, i, compiler.addrof( consumed ))
			mapped = CaseFolding._lookup( table, count, cp )
			with compiler.wrap_arithmetic:
				out_i += encode_utf8_at( new_buf, out_i, mapped )
				i += consumed
		new_buf[out_i] = 0

		return str._from_owned_cstr( new_buf, new_size ).unwrap( 'invalid UTF-8 produced by case_folding table lookup' )

	@private
	@staticmethod
	def _lookup( table: ConstPtr[u8], count: usize, cp: u32 ) -> u32:
		# binary search over the 8-bytes-per-entry (codepoint, mapped)
		# table, sorted ascending by codepoint - a codepoint with no entry
		# (the overwhelming majority - most codepoints have no case at
		# all) passes through unchanged, same posture as __str.py's own
		# OS-backed case_map paths
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
	reuses __str.py's own UTF-8 size/encode helpers (the same ones
	upper()/lower()/case-folding already build on) rather than duplicating
	that logic here. cp must be a real Unicode code point (<= U+10FFFF,
	not a UTF-16 surrogate half) - _from_owned_cstr's own UTF-8
	revalidation (the same one every other str-constructing method here
	already goes through) catches an invalid one, panicking via unwrap()
	rather than returning a Result: Python's real chr() raises ValueError
	for this, and this language panics instead of raising exceptions
	throughout (see str.split's identical reasoning for its own
	non-empty-separator precondition). '''
	encoded_len: usize = utf8_encoded_len( cp )
	buf_size: usize
	with compiler.panic_arithmetic( 'irrational string length' ):
		buf_size = encoded_len + 1 # +1 for the zero terminator
	buf: Ptr[u8] = sys.alloc[u8]( buf_size )
	encode_utf8_at( buf, 0, cp )
	buf[encoded_len] = 0
	return str._from_owned_cstr( buf, buf_size ).unwrap( 'chr(): not a valid Unicode code point' )

def ord( s: str ) -> u32:
	''' the inverse of chr() above - decodes s's own first (and only) code
	point back to its integer value, reusing __str.py's own UTF-8 decode
	helper. Requires s to be exactly one code point long, matching Python's
	own ord() (TypeError otherwise) - panics instead, same panic-not-
	exceptions convention chr() above follows. '''
	if s.byte_len() == 0:
		sys.panic( 'ord(): expected a string of length 1, got an empty string' )
	consumed: usize = 0
	cp: u32 = decode_utf8_at( s.get_const_ptr(), 0, compiler.addrof( consumed ))
	if consumed != s.byte_len():
		sys.panic( 'ord(): expected a string of length 1, got a longer string' )
	return cp

class UnsafeDict[K, V]:
	''' see PLAN_CALLABLE.md. RawDict (lib/builtins/__RawDict.py) is
	genuinely type-erased - it never decodes a key_ptr/value_ptr back to a
	real K/V, never allocates/frees/increfs/decrefs one, never computes a
	hash. Every method below that branches on compiler.is_rc(K)/
	compiler.is_rc(V) is a @staticmethod for exactly one reason: _key_eq
	needs to be referenced BARE (no receiver - see lowering.py's
	_lower_function_ref) to hand its address to RawDict as a real
	Ptr[Callable[...]] value, and the helpers it calls (_borrow_key) have
	to be static too so a bare reference from within a static method can
	reach them the same way.

	UnsafeDict[K,V]: thin type-safe wrapper over RawDict - no locking of
	its own. dict[K,V] (below) is the locked-by-default wrapper around
	this that most code should actually reach for - see __list.py's own
	header comment for why list[T]/UnsafeList[T] made the same split;
	dict[K,V] follows it for the same reason. '''
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

class dict[K, V]:
	''' dict[K,V]: locked-by-default wrapper around UnsafeDict[K,V] - same
	split as list[T]/UnsafeList[T] (see __list.py's own header comment):
	dict[K,V] is the default most people reach for, so every operation
	acquires a real FastLock; UnsafeDict[K,V] is the identical, unlocked
	implementation for callers who already know their dict never crosses
	a thread boundary (e.g. a private, never-escaping field). '''
	__inner: UnsafeDict[K, V]
	__lock:  threading.FastLock

	def __init__( self ) -> None:
		self.__inner = UnsafeDict[K, V]()
		self.__lock  = threading.FastLock()

	def __len__( self ) -> usize:
		self.__lock.acquire().unwrap( 'dict.__len__: lock failed' )
		defer( self.__lock.release() )
		return self.__inner.__len__()

	def __getitem__( self, key: K ) -> Result[V, KeyError]:
		self.__lock.acquire().unwrap( 'dict.__getitem__: lock failed' )
		defer( self.__lock.release() )
		return self.__inner.__getitem__( key )

	def __setitem__( self, key: K, value: V ) -> None:
		self.__lock.acquire().unwrap( 'dict.__setitem__: lock failed' )
		defer( self.__lock.release() )
		self.__inner.__setitem__( key, value )

	def with_lock( self, body: Closure[[UnsafeDict[K, V]], None] ) -> None:
		# unlike list[T]'s own no-arg with_lock, body here takes the raw
		# UnsafeDict[K,V] directly - a compound read-modify-write (the
		# whole reason to reach for with_lock instead of two separate
		# locked calls) needs to touch storage WHILE the lock is held,
		# and __inner is private - body has no other way in. Calling back
		# through self.__getitem__/self.__setitem__ from inside body would
		# try to re-acquire this same (non-reentrant) FastLock and deadlock.
		self.__lock.acquire().unwrap( 'dict.with_lock: lock failed' )
		defer( self.__lock.release() )
		body( self.__inner )

# import this at the end because it depends on str etc to already be pre-parsed:
from .__File import File
