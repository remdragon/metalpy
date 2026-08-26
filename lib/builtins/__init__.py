from codecs import Codec, CodecError
from codecs.utf8 import utf8
import compiler
import sys
import threading

# Sequence[T]/Iterable[T]: the two protocols min(seq)/max(seq)/iter/any/all/
# enumerate/map/reduce/sum are all built on (defined here, before ANY other
# name in this module, since list[T] below needs to declare conformance in
# its own class header, e.g. `class list[T](Sequence[T], Iterable[T]):` - a
# base-class list is resolved EAGERLY, at the base-class expression's own
# parse time, unlike an ordinary annotation reference (Result[T,IndexError]
# below is fine forward-referenced, lazily resolved - only a TypeVar's own
# bound and a class's own base list are eager. list[T] itself is defined in
# a separate file, __list.py, reached via this file's own `from .__list
# import list` below - list[T]'s own class statement is parsed as a side
# effect of THAT import line, so Sequence/Iterable must already be
# registered before it, not merely appear earlier in THIS file).
# Kept as two separate protocols, not one (matching real Python: dict
# conforms to Iterable via key-iteration but isn't a Sequence - __getitem__
# isn't usize-keyed) - see each's own docstring below.
@protocol
class Sized:
	# real Python's own typing.Sized/collections.abc.Sized idiom - len(x)
	# below is bound to this (not left duck-typed against a bare, unbound
	# TypeVar) specifically so a type with no __len__ at all, or one with
	# the wrong RETURN type, fails at the CALL SITE with a clear "does not
	# conform to protocol Sized" diagnostic, instead of a confusing type
	# mismatch reported from deep inside len[T]'s own @inline-spliced body
	# (lib/builtins/__init__.py, not the caller's own file) with no
	# indication of which call site or what T was - confirmed via a real
	# repro (grap.mpy's own `count: i32 = len(some_list)`, which - once
	# _lower_inline_call's own error-enrichment note pointed at it - turned
	# out to be an ordinary i32-vs-usize call-site mismatch, not a real
	# conformance gap, but the duck-typed len[T] had no way to distinguish
	# the two failure modes at all until this protocol existed).
	def __len__( self ) -> usize: ...

@protocol
class Sequence[T]:
	# only __getitem__ is required, not __len__ - reaching the end is
	# signaled by Err(IndexError) itself, so a conformer never needs to
	# separately answer "how many". list[T]'s existing scalar __getitem__
	# already has exactly this shape - conforming needed no method changes
	# there, only the base-class declaration.
	def __getitem__( self, i: usize ) -> Result[T, IndexError]: ...

@protocol
class Iterable[T]:
	# min(seq)/max(seq)/iter/any/all/enumerate/map/reduce/sum all bind on
	# THIS, not Sequence[T] - matching real Python, where those accept any
	# iterable, not just a random-access sequence.
	def __iter__( self ) -> Generator[T, StopIteration]: ...

@protocol
class IteratorProtocol[T]:
	# an ALREADY-in-progress iterator (a real generator object, or any
	# hand-written class with its own __next__) - what for-loops (lowering.
	# py's _stmt_For) actually drive once they have one, whether that came
	# directly from a for-loop's own subject (IteratorProtocol[T]
	# conformance) or via Iterable[T].__iter__() first. Named
	# "IteratorProtocol", not the shorter "Iterator" its real Python
	# namesake uses - "Iterator[...]" is already claimed, permanently, by
	# an unrelated, pre-existing compiler special form (discovery.py's
	# visit_Subscript textually recognizes `Iterator[Result[T,E]]` as
	# sugar for a generator function's own return-type annotation - see
	# Generator[T,E]'s identical treatment right below - long before this
	# protocol existed), so `class Foo(Iterator[T]):` would silently
	# misparse as THAT instead of a real protocol base. Declared
	# conformance is a NAME-only check (discovery.py's _validate_protocol_
	# conformance never inspects __next__'s own signature), so this stub's
	# exact Result[T,StopIteration] shape below doesn't constrain a real
	# generator's own wider error type in any way - __next__'s error type E
	# may be anything AS LONG AS StopIteration is one of its leaves
	# (PLAN_GENERATORS.md's StopIteration reversal - reaching the end is
	# Err(StopIteration()), not a nullable None), which is what a for-
	# loop's own consumption actually requires and already handles (E' = E
	# minus StopIteration - see _lower_for_over_iterator_fallible_bind). A
	# generator's own synthesized backing class (type_resolver.py's
	# ensure_generator_synthesized) declares this conformance itself, the
	# same way TupleStorage._declare_sequence_conformance already does for
	# Sequence[T]/Iterable[T].
	def __next__( self ) -> Result[T, StopIteration]: ...

# the one place a Sequence[T]'s index-walk is written - every conformer's own
# __iter__ just delegates here (a __iter__ method can never itself contain
# yield - see type_resolver.py's ensure_generator_synthesized - so a
# conformer always needs a thin delegating method like this one regardless).
def _sequence_iter[T, S: Sequence[T]]( seq: S ) -> Generator[T, StopIteration]:
	i: usize = 0
	while True:
		match seq.__getitem__( i ):
			case Result.Ok( item ):
				yield item
			case _:
				return
		with compiler.wrap_arithmetic:
			i += 1

from .__errors import OSError
from .__fastlist import FastList
from .__float import _f64_sign_prefix, _f64_fixed_digits
from .__int import int, IntError
from .__scalar_dunders import i_add_checked
from .__ptr_arith import ptr_add_checked
from .__list import list, UnsafeList
from .__vartuple import VariadicTuple, tuple
from .__RawDict import RawDict, RawEntry
from .__set import set
from .__str import decode_utf8_at, encode_utf8_at, utf8_encoded_len, case_map, case_map_one, is_alpha_cp, is_digit_cp, is_space_cp, is_upper_cp, is_lower_cp, is_alnum_cp, is_printable_cp, ascii_escape_width, ascii_escape_one, repr_escape_width, repr_escape_one, nonascii_escape_width, nonascii_escape_one

# markers with no payload of their own - Check-mode arithmetic (AddCheck/
# SubCheck/MulCheck/...) and Div/Mod produce Result[T,OverflowError]/
# Result[T,ZeroDivisionError] purely as a tag (see ir.py's own comments on
# those opcodes); slice.__getitem__ raises IndexError the same way
class OverflowError: pass
class ZeroDivisionError: pass
# checked/panic-mode floating-point arithmetic (FAddCheck/FSubCheck/FMulCheck)
# and float-involving casts (FloatCastCheck) produce Result[T,FloatingPointError]
# when a result is inf/nan or a float->int source is out of range - the float
# analogue of OverflowError (float div-by-zero stays ZeroDivisionError, matching
# integer /). Same empty-marker shape as the others
class FloatingPointError: pass
class IndexError: pass
class KeyError: pass
# structural `==`/`!=` between two operands where at least one (left, right)
# leaf-type pairing has no valid comparison (see lowering.py's
# _lower_eq_dispatch/_classify_leaf_pair_eq) produces Result[bool,TypeError]
# purely as a tag, same empty-marker shape as the others above
class TypeError: pass
# os.path.commonpath()/relpath()'s mismatched-root / mixed-absolute-and-
# relative failure - real Python raises ValueError; Result[str, ValueError]
# instead, per this codebase's Result convention
class ValueError: pass
# not an operation-failure marker like the others above - the generator-
# exhaustion sentinel every Iterator[Result[T,StopIteration]]/Generator[T,E]
# (E always includes this) produces in its own Result's error channel once
# $$__next__ reaches the end of the function, instead of a nullable None
# bundled into the success channel (see PLAN_GENERATORS.md)
class StopIteration: pass

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

	# NOTE: each accessor copies the Ok payload into a LOCAL before returning
	# it, rather than `return self.data.v_Ok` directly. For an RC T that copy
	# is where the incref happens (capturing an attribute into a local
	# increfs; a bare `return self.attr` does not) - so the caller receives a
	# genuinely owned +1 reference. This balances against the receiver
	# Result's OWN payload decref at the end of its expression/scope
	# (whichever of the two applies - a temp receiver's is at end-of-
	# expression, a named receiver's is at scope exit) - see cfg.py's
	# rc_leaves()/_refcount_instructions for the matching fix that makes a
	# Result value's own payload actually get tracked/decref'd at all (a
	# separate, previously-missing half of this same balance). This declared
	# body is what actually runs for unwrap()/unwrap_or() below (real
	# function calls).

	# or_return() is compiler magic, not a real method - deliberately NOT
	# declared here (or anywhere: 'or_return' is a reserved name, rejected
	# outright by discovery.py's _parse_function on ANY class, since a
	# user-written one could never actually run). `<result_expr>.or_return()`
	# is recognized purely by its AST shape (lowering.py's _lower_call,
	# before ordinary call resolution ever runs) and compiled directly to
	# raw OrReturn/OrJump IR (see _lower_or_return) - equivalent to:
	#     if self.is_err():
	#         compiler.early_return( self.data.v_Err )
	#     ok: T = self.data.v_Ok
	#     return ok
	# or_return's own matching incref (see the accessor NOTE above) lives in
	# lowering.py's _consume_checked_result, which every or_return() call
	# actually goes through.

	# or_throw() - same reserved-name/no-real-body/AST-shape-recognized
	# story as or_return() just above (also rejected outright by
	# discovery.py's _parse_function). Like or_return() on the Ok branch;
	# on the Err branch, each leaf of E first checks the INNERMOST
	# enclosing try's own except clauses (only valid textually inside a
	# try body, in the SAME function) and jumps into a matching handler
	# instead of propagating, when one covers that leaf - equivalent to:
	#     if self.is_err():
	#         match self.data.v_Err:                # conceptually - real
	#             case <a covered leaf>: goto <that except clause>
	#             case _: compiler.early_return( self.data.v_Err )  # uncovered leaves only
	#     ok: T = self.data.v_Ok
	#     return ok
	# See lowering.py's _lower_or_throw/_stmt_Try and ir.OrThrow.

	@overload
	def unwrap( self, errmsg: str ) -> T:
		...

	@overload
	def unwrap( self, errmsg: Ptr[Callable[[E], str]] ) -> T:
		...

	# a single real body for BOTH stubs above (not two independent plain
	# implementations) - two Functions sharing the 'unwrap' qualname would
	# otherwise collide in _get_or_create_specialization's own qualname-
	# keyed cache once this generic class is monomorphized (confirmed via a
	# real compile: the second implementation silently got back the FIRST
	# one's already-cached specialization instead of its own), the same
	# collision unwrap_or's own stub+impl split already sidesteps
	def unwrap( self, errmsg: str | Ptr[Callable[[E], str]] ) -> T:
		if self.is_ok():
			ok: T = self.data.v_Ok
			return ok
		if type( errmsg ) is str:
			sys.panic( errmsg )
		else:
			err: E = self.data.v_Err
			sys.panic( errmsg( err ))

	@overload
	def unwrap_or( self, default: T ) -> T:
		...

	def unwrap_or( self, default: T|None = None ) -> T|None:
		if self.is_ok():
			ok: T = self.data.v_Ok
			return ok
		return default


@cstruct
class slice:
	# range descriptor for container[a:b] slice syntax (lowering.py's
	# _lower_slice_subscript) - the __getitem__(slice) overload argument
	# every slice-syntax-supporting type (str, bytearray, memoryview,
	# UnsafeList[T], list[T]) shares, replacing the old hardcoded
	# per-type _byte_slice(start,end)/_SLICE_LENGTH_METHOD dispatch. stop is
	# nullable rather than eagerly resolved against a hardcoded per-type
	# length method (container[a:] omits it) - each type's own overload
	# resolves ITS OWN correct default length in whatever unit is right for
	# it (str's real __len__ is a codepoint count, wrong unit for its
	# byte-offset slicing - see str.byte_len() vs str.__len__()). step is
	# unsupported (rejected at the lowering layer), so no step field. isize,
	# not usize - real Python slice bounds are signed: a negative bound
	# (s[-3:-2]) means "offset from the end", resolved against the
	# container's own real length in _resolve_slice_bounds below.
	start: isize
	stop: isize|None


# Resolves a slice against a container's own real length, matching real
# Python's own slice semantics EXACTLY: a negative bound counts back from
# real_len (clamped to 0 if that still lands negative), an out-of-range
# bound clamps into [0, real_len], and start > stop (after clamping) yields
# an empty range - rather than the stricter Result[T,IndexError] convention
# every other __getitem__ in this codebase uses for single-element access
# (container[i] DOES raise/Result::Err on an out-of-range i). Slicing is a
# deliberately forgiving idiom in real Python, worth preserving faithfully
# rather than picking the stricter convention. Shared by every
# __getitem__(slice) overload (str, bytearray, memoryview, UnsafeList[T],
# list[T]) so the clamping math itself lives in exactly one place.
def _resolve_slice_bounds( s: slice, real_len: usize ) -> tuple[usize,usize]:
	with compiler.panic_arithmetic( 'a real container length always fits isize' ):
		len_i: isize = isize( real_len )
	clamped_start_i: isize = _clamp_slice_bound( s.start, len_i )
	stop_field: isize|None = s.stop
	clamped_stop_i: isize = len_i
	if stop_field is not None:
		clamped_stop_i = _clamp_slice_bound( stop_field, len_i )
	if clamped_start_i > clamped_stop_i:
		clamped_stop_i = clamped_start_i
	with compiler.panic_arithmetic( 'both bounds already clamped into [0, len_i]' ):
		return ( usize( clamped_start_i ), usize( clamped_stop_i ) )


# one bound (start or stop) of _resolve_slice_bounds, resolved against the
# container's own real length (as isize) - negative counts back from the
# end, then both directions clamp into [0, len_i].
def _clamp_slice_bound( bound: isize, len_i: isize ) -> isize:
	resolved: isize = bound
	if resolved < 0:
		with compiler.wrap_arithmetic: # len_i is a real container length, never large enough to overflow isize
			resolved = resolved + len_i
		if resolved < 0:
			resolved = 0
	elif resolved > len_i:
		resolved = len_i
	return resolved



# shared byte-level helpers for bytes.find()/bytearray.find() (etc.) - pure
# sys.memcmp over an explicit length, unlike str.find()'s UTF-8-aware
# reasoning: bytes/bytearray carry no UTF-8-validity or null-terminator
# guarantee, so nothing here infers where a match "safely" lands, length is
# always explicit. Free functions (not methods on either class) so the
# actual scan logic isn't duplicated between bytes and bytearray - mirrors
# how Codec.decode (lib/codecs/utf8.py) already operates generically over a
# bytes|bytearray union via the public len()/get_const_ptr() accessors.
# Same -1-means-not-found convention as str.find() (see str.find() below).
def _bytes_find_at( haystack: bytes|bytearray, needle: bytes|bytearray, start: usize ) -> isize:
	self_len: usize = len( haystack )
	sub_len: usize = len( needle )
	if start > self_len:
		return isize( -1 )
	if sub_len == 0:
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			return isize( start )
	with compiler.wrap_arithmetic: # start <= self_len, just checked above
		remaining: usize = self_len - start
	if sub_len > remaining:
		return isize( -1 )
	with compiler.wrap_arithmetic: # sub_len <= self_len, just checked above
		last_start: usize = self_len - sub_len
	haystack_ptr: ConstPtr[u8] = haystack.get_const_ptr()
	needle_ptr: ConstPtr[u8] = needle.get_const_ptr()
	i: usize = start
	with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
		while i <= last_start:
			if sys.memcmp( haystack_ptr + i, needle_ptr, sub_len ) == 0:
				return isize( i )
			i += 1
	return isize( -1 )

def _bytes_startswith_at( haystack: bytes|bytearray, prefix: bytes|bytearray, start: usize ) -> bool:
	self_len: usize = len( haystack )
	prefix_len: usize = len( prefix )
	if start > self_len:
		return False
	if prefix_len == 0:
		return True
	with compiler.wrap_arithmetic: # start <= self_len, just checked above
		remaining: usize = self_len - start
	if prefix_len > remaining:
		return False
	with compiler.wrap_arithmetic: # start bounded by self_len above
		candidate: ConstPtr[u8] = haystack.get_const_ptr() + start
	return sys.memcmp( candidate, prefix.get_const_ptr(), prefix_len ) == 0

def _bytes_endswith( haystack: bytes|bytearray, suffix: bytes|bytearray ) -> bool:
	self_len: usize = len( haystack )
	suffix_len: usize = len( suffix )
	if suffix_len == 0:
		return True
	if suffix_len > self_len:
		return False
	with compiler.wrap_arithmetic: # suffix_len <= self_len, just checked above
		offset: usize = self_len - suffix_len
		candidate: ConstPtr[u8] = haystack.get_const_ptr() + offset
	return sys.memcmp( candidate, suffix.get_const_ptr(), suffix_len ) == 0


class bytes( Sequence[u8], Iterable[u8], Sized ):
	__data: ConstPtr[u8]

	__len: usize

	def __init__( self, copy_from: bytes|bytearray ) -> None:
		self.__len = len( copy_from )
		data = sys.alloc[u8]( self.__len )
		sys.memcpy( data, copy_from.get_const_ptr(), self.__len )
		self.__data = data

	@staticmethod
	def from_memoryview( src: memoryview ) -> bytes:
		# a NAMED alternative constructor, not a 3rd __init__ union member -
		# __init__ overloading isn't supported yet (see memoryview's own
		# __init__ comment), and widening __init__'s own copy_from to
		# bytes|bytearray|memoryview instead broke every existing bytes|
		# bytearray-typed caller (confirmed via a real repro: this codebase
		# has no general "narrower union coerces into a wider superset
		# union" mechanism - only a single LEAF coercing into a union
		# containing it is supported - so a bytes|bytearray-typed value
		# no longer satisfied a bytes|bytearray|memoryview-typed parameter
		# at all, a straight regression for lib/zipfile.py's own
		# _compress_payload). Matches real Python's own bytes(some_
		# memoryview) support, just spelled as a named constructor here -
		# same shape as from_bytearray just above. get_const_ptr()/
		# __len__() are identically named on all three buffer types, same
		# trick memoryview's own __init__ uses for its bytearray|mmap
		# source.
		length: usize = len( src )
		data = sys.alloc[u8]( length )
		sys.memcpy( data, src.get_const_ptr(), length )
		return bytes.__allocate__( __data = data, __len = length )

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

	def find( self, sub: bytes|bytearray, start: usize = 0 ) -> isize:
		return _bytes_find_at( self, sub, start )

	def startswith( self, prefix: bytes|bytearray, start: usize = 0 ) -> bool:
		return _bytes_startswith_at( self, prefix, start )

	def endswith( self, suffix: bytes|bytearray ) -> bool:
		return _bytes_endswith( self, suffix )

	@overload
	def __getitem__( self, index: usize ) -> Result[u8,IndexError]:
		if index >= self.__len:
			return Result.Err( IndexError() )
		return Result.Ok( self.__data[index] )

	# s[a:b] slice syntax (lowering.py's _lower_slice_subscript) - a new,
	# independently-owned bytes, matching real Python's own slice semantics
	# exactly (out-of-range bounds silently clamp - see
	# _resolve_slice_bounds).
	@overload
	def __getitem__( self, s: slice ) -> bytes:
		( start, stop ) = _resolve_slice_bounds( s, self.__len )
		return self._byte_slice( start, stop )

	def __iter__( self ) -> Generator[u8, StopIteration]:
		return _sequence_iter( self )

	@private
	def _byte_slice( self, start: usize, end: usize ) -> bytes:
		''' bytes [start, end) of self, as a new, independently-owned bytes -
		the bytes-side counterpart to bytearray._byte_slice below. bytes has
		no size-only public constructor (unlike bytearray), so this goes
		through __allocate__ directly, the same way from_bytearray above
		does. '''
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			piece_len: usize = end - start
		new_data: Ptr[u8] = sys.alloc[u8]( piece_len )
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			src: ConstPtr[u8] = self.__data + start
		sys.memcpy( new_data, src, piece_len )
		return bytes.__allocate__( __data = new_data, __len = piece_len )

	def split( self, sep: bytes|bytearray ) -> list[bytes]:
		''' splits self on every occurrence of sep - same semantics as
		str.split() (lib/builtins/__init__.py), built on find()/_byte_slice
		above rather than its own scanning logic. sep must not be empty. '''
		if len( sep ) == 0:
			sys.panic( 'bytes.split(...): separator must not be empty' )
		result: list[bytes] = list[bytes]()
		self_len: usize = self.__len__()
		sep_len: usize = len( sep )
		start: usize = 0
		while True:
			found: isize = self.find( sep, start )
			if found == isize( -1 ):
				result.append( self._byte_slice( start, self_len ))
				break
			with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
				match_start: usize = usize( found )
			result.append( self._byte_slice( start, match_start ))
			with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
				start = match_start + sep_len
		return result

BYTEARRAY_INVALID: Ptr[u8] = 0 # this is a sentinel to indicate a bytearray was released - matches lib/windows/kernel32.py's own INVALID_HANDLE_VALUE convention (a literal assigned directly to its real pointer type, not a same-width integer alias needing its own cast at every comparison site)

class bytearray( Sequence[u8], Iterable[u8], Sized ):
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

	def find( self, sub: bytes|bytearray, start: usize = 0 ) -> isize:
		if compiler.target.debug:
			assert self.__data != BYTEARRAY_INVALID, 'bytearray.find() called after release()'
		return _bytes_find_at( self, sub, start )

	def startswith( self, prefix: bytes|bytearray, start: usize = 0 ) -> bool:
		if compiler.target.debug:
			assert self.__data != BYTEARRAY_INVALID, 'bytearray.startswith() called after release()'
		return _bytes_startswith_at( self, prefix, start )

	def endswith( self, suffix: bytes|bytearray ) -> bool:
		if compiler.target.debug:
			assert self.__data != BYTEARRAY_INVALID, 'bytearray.endswith() called after release()'
		return _bytes_endswith( self, suffix )

	@overload
	def __getitem__( self, index: usize ) -> Result[u8,IndexError]:
		if compiler.target.debug:
			assert self.__data != BYTEARRAY_INVALID, 'bytearray.__getitem__() called after release()'
		if index >= self.__len:
			return Result.Err( IndexError() )
		return Result.Ok( self.__data[index] )

	@overload
	def __getitem__( self, s: slice ) -> bytearray:
		''' s[a:b] slice syntax (lowering.py's _lower_slice_subscript) -
		infallible, matching real Python's own slice semantics exactly -
		out-of-range bounds silently clamp rather than raising (see
		_resolve_slice_bounds), unlike single-element s[i] above, which
		DOES error on an out-of-range index. '''
		if compiler.target.debug:
			assert self.__data != BYTEARRAY_INVALID, 'bytearray.__getitem__() called after release()'
		( start, stop ) = _resolve_slice_bounds( s, self.__len )
		return self._byte_slice( start, stop )

	def __setitem__( self, index: usize, value: u8 ) -> None:
		if compiler.target.debug:
			assert self.__data != BYTEARRAY_INVALID, 'bytearray.__setitem__() called after release()'
			assert index < self.__len, 'bytearray.__setitem__() index out of range'
		self.__data[index] = value

	def __iter__( self ) -> Generator[u8, StopIteration]:
		return _sequence_iter( self )

	def resize( self, new_size: usize ) -> None:
		''' grows or shrinks self to new_size bytes, zero-filling any newly
		exposed bytes (matching __init__'s own zero-init convention, even
		for bytes that were part of an earlier, larger allocation a prior
		shrink left behind - simpler and safer than CPython's own "may
		retain stale bytes there" behavior, at the cost of not matching it
		exactly). Growing within __cap (already-allocated capacity, e.g.
		regrowing after a previous shrink) is a pure length update, no
		reallocation; growing past __cap reallocates to exactly new_size,
		not an amortized/doubled capacity - the motivating caller
		(lib/zipfile.py's own buffer.resize(file_size), one call per zip
		entry, not a tight append loop) doesn't need amortized growth. '''
		if compiler.target.debug:
			assert self.__data != BYTEARRAY_INVALID, 'bytearray.resize() called after release()'
		if new_size <= self.__cap:
			if new_size > self.__len:
				with compiler.panic_arithmetic( 'bounded by __cap, cannot overflow' ):
					grown: usize = new_size - self.__len
					fill_at: Ptr[u8] = self.__data + self.__len
				sys.memzero( fill_at, grown )
			self.__len = new_size
			return
		new_data: Ptr[u8] = sys.alloc[u8]( new_size )
		sys.memcpy( new_data, self.__data, self.__len )
		with compiler.panic_arithmetic( 'new_size > __cap >= __len, cannot overflow' ):
			tail: usize = new_size - self.__len
			new_fill_at: Ptr[u8] = new_data + self.__len
		sys.memzero( new_fill_at, tail )
		sys.free( self.__data )
		self.__data = new_data
		self.__len = new_size
		self.__cap = new_size

	def decode( self, codec: Codec = utf8 ) -> Result[str,CodecError]:
		if compiler.target.debug:
			assert self.__data != BYTEARRAY_INVALID, 'bytearray.decode() called after release()'
		return codec.decode( self )
	
	@move
	def release( self ) -> Result[Ptr[u8],sys.OwnershipError[bytearray]]:
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

	@private
	def _byte_slice( self, start: usize, end: usize ) -> bytearray:
		''' bytes [start, end) of self, as a new, independently-owned
		bytearray - the bytearray-side counterpart to str._byte_slice
		below (same method name deliberately, so slice-syntax lowering
		only needs one method name to look up across both types). No
		zero-terminator/UTF-8 revalidation concern here (unlike str's own
		hand-rolled version), so this just reuses the ordinary public
		constructor. '''
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			piece_len: usize = end - start
		result = bytearray( piece_len )
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			src: ConstPtr[u8] = self.__data + start
		sys.memcpy( result.__data, src, piece_len )
		return result

	def split( self, sep: bytes|bytearray ) -> list[bytearray]:
		''' splits self on every occurrence of sep - same semantics as
		bytes.split()/str.split() above. Each returned piece is an
		independently-owned, freshly-allocated bytearray (_byte_slice
		always allocates+copies, never aliases self's own buffer), so
		mutating one piece afterward cannot affect self or its siblings.
		sep must not be empty. '''
		if compiler.target.debug:
			assert self.__data != BYTEARRAY_INVALID, 'bytearray.split() called after release()'
		if len( sep ) == 0:
			sys.panic( 'bytearray.split(...): separator must not be empty' )
		result: list[bytearray] = list[bytearray]()
		self_len: usize = self.__len__()
		sep_len: usize = len( sep )
		start: usize = 0
		while True:
			found: isize = self.find( sep, start )
			if found == isize( -1 ):
				result.append( self._byte_slice( start, self_len ))
				break
			with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
				match_start: usize = usize( found )
			result.append( self._byte_slice( start, match_start ))
			with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
				start = match_start + sep_len
		return result

class str( Sequence[str], Iterable[str], Sized ):
	__data: ConstPtr[u8]

	__byte_size: usize # the number of bytes (code units) include the zero-terminater

	__char_count: usize # Unicode codepoint count, computed once in _from_owned_cstr

	__index: Ptr[usize] # owned, always-allocated sparse index: one entry per 256
	                     # codepoints, storing the BYTE OFFSET where that group starts.
	                     # entries = (__byte_size >> 8) + 1, a pure function of
	                     # __byte_size so no separate stored length is needed - every
	                     # codepoint takes >= 1 byte, so char_count <= byte_len <=
	                     # byte_size, and idx >> 8 < entries for any valid idx.

	__utf16: ConstPtr[u16] # lazily-computed null-terminated UTF-16LE cache
	                       # (see to_utf16()) - None until first use.
	                       # Published via a real atomic CAS (compiler.
	                       # atomic_compare_exchange on this field's own
	                       # address - see lowering.py's
	                       # _atomic_pointee_type, which allows a raw
	                       # Ptr[T]/ConstPtr[T] pointee same as a plain
	                       # scalar): two threads racing the first
	                       # to_utf16() call may both redundantly encode,
	                       # but exactly one buffer is ever kept - the
	                       # loser frees its own, so there's no leak.

	def __del__( self ) -> None:
		sys.free( self.__data )
		sys.free( self.__index )
		if self.__utf16 is not None:
			sys.free( compiler.cast( Ptr[u8], self.__utf16 ))

	@inline
	def __str__( self ) -> str:
		# str is immutable - str(x) is always just x itself, no copy.
		return self

	def __repr__( self ) -> str:
		''' quotes and backslash-escapes self, matching Python's repr()
		for str: single-quoted, unless self contains a "'" but no '"' (then
		double-quoted instead). Printable non-ASCII codepoints are kept as
		literal UTF-8, matching CPython - see __str.py's repr_escape_width/
		repr_escape_one for the full per-codepoint escaping rule. Two
		passes over the codepoints (first to pick the quote char, since its
		own escape width depends on which one was picked, then the usual
		size-then-fill pass every other string builder here uses). '''
		self_len: usize = self.byte_len()
		has_single: bool = False
		has_double: bool = False
		i: usize = 0
		consumed: usize = 0
		with compiler.panic_arithmetic( 'irrational string length' ):
			while i < self_len:
				cp: u32 = decode_utf8_at( self.__data, i, compiler.addrof( consumed ))
				if cp == 0x27: # '
					has_single = True
				elif cp == 0x22: # "
					has_double = True
				i += consumed

		quote: u8 = u8( 0x22 ) if ( has_single and not has_double ) else u8( 0x27 )

		new_size: usize = 3 # opening + closing quote + zero terminator
		i = 0
		with compiler.panic_arithmetic( 'irrational string length' ):
			while i < self_len:
				cp = decode_utf8_at( self.__data, i, compiler.addrof( consumed ))
				new_size += repr_escape_width( cp, quote )
				i += consumed

		new_buf: Ptr[u8] = sys.alloc[u8]( new_size )
		new_buf[0] = quote
		out: usize = 1
		i = 0
		with compiler.wrap_arithmetic:
			while i < self_len:
				cp = decode_utf8_at( self.__data, i, compiler.addrof( consumed ))
				out += repr_escape_one( new_buf, out, cp, quote )
				i += consumed
			new_buf[out] = quote
			out += 1
			new_buf[out] = 0
		return str._from_owned_cstr( new_buf, new_size ).unwrap( 'invalid UTF-8 in __repr__' )

	@inline
	@staticmethod
	def __call__[T]( x: T ) -> str:
		# str(x) isn't a request to create a new str, but to convert
		# an object into a str, but it requires that object to implement
		# __str__() since we have no way to know how to do that from here.
		return x.__str__()

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
	def concat( parts: UnsafeList[str] ) -> str:
		# __getitem__(i)'s own Incref (bounds-checked, but every i here is
		# < count by construction) plus .unwrap()'s bare extraction already
		# return an owned str, so binding it into a named local and letting
		# that local's normal scope-exit decref fire is the correct release
		# of exactly the reference __getitem__ just gave us - no manual
		# incref needed here (same "bound-local, incref cancels scope-exit
		# decref" idiom list[T].pop() already uses).
		new_size: usize = 1 # for the zero terminator
		i: usize = 0
		count: usize = parts.__len__()
		for i in range( count ):
			part: str = parts.__getitem__( i ).unwrap( 'str.concat: index in bounds by construction' )
			with compiler.panic_arithmetic( 'irrational string length' ):
				new_size += part.__byte_size - 1

		new_buf: Ptr[u8] = sys.alloc[u8]( new_size )
		offset: usize = 0

		for i in range( count ):
			# distinct name from the sizing loop's own `part` above - a
			# variable's type is only ever declared once per function
			copy_part: str = parts.__getitem__( i ).unwrap( 'str.concat: index in bounds by construction' )
			with compiler.panic_arithmetic( 'irrational string length' ):
				part_len: usize = copy_part.__byte_size - 1
			with compiler.wrap_arithmetic:
				sys.memcpy( new_buf + offset, copy_part.__data, part_len )
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

		with compiler.panic_arithmetic( 'invalid str length' ):
			last_index: usize = size_including_zero_terminator - 1
		if new_buf[last_index]:
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

	@staticmethod
	def from_utf16( ptr: ConstPtr[u16], max_len: usize ) -> Result[str,CodecError]:
		'''
		build a str from a null-terminated UTF-16 (native-endianness) buffer
		- e.g. a Win32 LPCWSTR. max_len bounds the terminator scan, in u16
		units, not including it (mirrors sys.cstrlen's own max_length cap
		for u8 C strings).
		'''
		from codecs.utf16 import utf16
		n: usize = 0
		with compiler.wrap_arithmetic:
			while n < max_len and ptr[n] != 0:
				n += 1
		if n == max_len:
			return Result.Err( CodecError( 'utf-16le', 'missing null terminator' ))
		with compiler.panic_arithmetic( 'bounded by max_len, cannot overflow' ):
			byte_len: usize = n * 2
		buf = bytearray( byte_len )
		sys.memcpy( buf.get_ptr(), compiler.cast( ConstPtr[u8], ptr ), byte_len )
		return utf16.decode( buf )

	def to_utf16( self ) -> ConstPtr[u16]:
		'''
		self encoded as null-terminated UTF-16LE - e.g. for a Win32 *W
		call's LPCWSTR argument. Cached after the first call (see __utf16's
		own field comment), so repeated calls against the same str don't
		re-encode. Published with a real atomic CAS - see __utf16's own
		field comment for why a race here is wasted work, never a leak.
		'''
		cached: ConstPtr[u16] = compiler.atomic_load( compiler.addrof( self.__utf16 ))
		if cached is not None:
			return cached
		from codecs.utf16 import utf16
		encoded: bytes = utf16.encode( self ).unwrap( 'str.to_utf16: invalid UTF-8 in str' )
		byte_len: usize = len( encoded )
		with compiler.panic_arithmetic( 'a real string can never be within 2 bytes of usize::MAX' ):
			alloc_size: usize = byte_len + 2
			last_index: usize = alloc_size - 1
		raw: Ptr[u8] = sys.alloc[u8]( alloc_size )
		sys.memcpy( raw, encoded.get_const_ptr(), byte_len )
		raw[byte_len] = 0
		raw[last_index] = 0
		buf: ConstPtr[u16] = compiler.cast( ConstPtr[u16], raw )

		expected: ConstPtr[u16] = None
		if compiler.atomic_compare_exchange( compiler.addrof( self.__utf16 ), compiler.addrof( expected ), buf ):
			return buf
		# someone else already published first - free our redundant buffer
		# and use theirs (CAS failure wrote the actual current value into
		# `expected`)
		sys.free( raw )
		return expected

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
		# Unicode code point count (real Python len(s) semantics) - O(1),
		# precomputed once by _from_owned_cstr during its mandatory UTF-8
		# validation scan (see __char_count's own field comment).
		return self.__char_count

	def __iter__( self ) -> Generator[str, StopIteration]:
		return _sequence_iter( self )

	@overload
	def __getitem__( self, idx: usize ) -> Result[str, IndexError]:
		''' codepoint-indexed access (Python's s[i]) - no negative-index
		support, matching list/bytearray/dict/set.__getitem__ here, none of
		which support negative indices either. Jumps to the nearest <=256-
		codepoint group via __index (O(1)), then decodes forward through at
		most 255 codepoints to reach idx's own start byte offset - same
		decode_utf8_at-forward-through-consumed-bytes shape lstrip/rstrip
		below already use, just bounded to one group instead of the whole
		string. '''
		if idx >= self.__char_count:
			return Result.Err( IndexError() )
		group: usize = idx >> 8
		i: usize = self.__index[group]
		remaining: usize = idx & 0xFF
		consumed: usize = 0
		with compiler.panic_arithmetic( 'bounded by remaining < 256 within a valid group, cannot overflow' ):
			j: usize = 0
			while j < remaining:
				decode_utf8_at( self.__data, i, compiler.addrof( consumed ))
				i += consumed
				j += 1
			decode_utf8_at( self.__data, i, compiler.addrof( consumed ))
			end: usize = i + consumed
		return Result.Ok( self._byte_slice( i, end ))

	@overload
	def __getitem__( self, s: slice ) -> str:
		''' s[a:b] slice syntax (lowering.py's _lower_slice_subscript) -
		BYTE offsets, not this class's own __len__/codepoint-indexed s[i]
		above (deliberate, pre-existing split - see _byte_slice's own
		docstring: the one real caller, lib/posix/time.py's
		target_path[idx+9:], slices from str.find()'s own byte offset).
		Infallible, matching real Python's own slice semantics exactly -
		out-of-range bounds silently clamp rather than raising (see
		_resolve_slice_bounds) - unlike single-element s[i] above, which
		DOES error on an out-of-range index. This also closes what the old
		hardcoded slice-syntax path never checked at all (stop > len() was
		a silent out-of-bounds C-level read) and used to panic on (usize
		underflow on start > stop) - clamping makes both cases well-defined
		instead, on top of matching Python's own behavior. '''
		( start, stop ) = _resolve_slice_bounds( s, self.byte_len() )
		return self._byte_slice( start, stop )

	def __bool__( self ) -> bool:
		# Python-style str truthiness: empty string is falsy. byte_len()
		# (not __len__()'s codepoint-counting UTF-8 scan) is enough here -
		# a string with zero content bytes has zero codepoints and vice
		# versa - so this stays O(1). Lets type_resolver.py's
		# _rewrite_tagged_union_truthiness synthesize a real `.__bool__()`
		# call for a str|None-typed `if x:`/`while x:` (previously the
		# only leaf type it ever reached for was bool, which needs no
		# dunder call at all - see print()'s own end-parameter comment
		# just below for the bare, non-union str truthiness gap this
		# doesn't touch).
		return self.byte_len() != 0

	def find( self, sub: str, start: usize = 0 ) -> isize:
		''' byte offset of the first occurrence of sub in self, searching
		from byte offset start onward (default 0 - the whole string; used
		by split() below to resume searching just past each match, without
		its own separate scanning logic). Returns -1 if not found (Python
		str.find() convention - "not found" is an expected, common outcome
		here, not an error; see index() below for the Result-returning
		variant, for callers that DO want it treated as one). UTF-8-safe at
		the byte level even though the scan itself is pure byte comparison
		(sys.memcmp): sub is itself valid UTF-8 (str's own construction-
		time invariant - see _from_owned_cstr), so a genuine match boundary
		can never be split mid-codepoint - an ASCII byte or a UTF-8
		leading/continuation byte can only byte-for-byte equal the same
		kind of byte in sub, never straddle one. Empty sub matches at
		offset start, same as Python's str.find(''). '''
		self_len: usize = self.byte_len()
		sub_len: usize = sub.byte_len()
		if start > self_len:
			return isize( -1 )
		if sub_len == 0:
			with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
				return isize( start )
		with compiler.wrap_arithmetic: # start <= self_len, just checked above
			remaining: usize = self_len - start
		if sub_len > remaining:
			return isize( -1 )
		with compiler.wrap_arithmetic: # sub_len <= self_len, just checked above
			last_start: usize = self_len - sub_len
		i: usize = start
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			while i <= last_start:
				candidate: ConstPtr[u8] = self.__data + i
				if sys.memcmp( candidate, sub.__data, sub_len ) == 0:
					return isize( i )
				i += 1
		return isize( -1 )

	def index( self, sub: str ) -> Result[usize, IndexError]:
		''' like find(), but returns a Result instead of a -1 sentinel -
		for callers that consider "not found" an error worth propagating
		via match/.or_return()/.is_err(), rather than a plain conditional.
		Never panics - unlike this method's old (backwards) behavior,
		there is deliberately no unwrap()/sys.panic() anywhere in here. '''
		offset: isize = self.find( sub )
		if offset == isize( -1 ):
			return Result.Err( IndexError() )
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			return Result.Ok( usize( offset ) )

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

	@private
	def _truncate_codepoints( self, max_count: usize ) -> str:
		''' keeps only the first max_count codepoints of self, discarding
		the rest - used by f-string format-spec precision on str values
		(f"{s:.5}"), matching Python's own precision-on-str semantics
		(codepoint count, the same unit __len__ itself uses, not byte
		length). Already-short-enough (including max_count >= self's own
		__len__) is a no-op. Same "walk codepoints via the leading-byte
		test" loop shape __len__ already uses, just capturing the BYTE
		OFFSET where the (max_count+1)-th codepoint starts instead of only
		counting - _byte_slice(0, that offset) then keeps exactly
		max_count codepoints. '''
		self_count: usize = self.__len__()
		if self_count <= max_count:
			return self
		count: usize = 0
		i: usize = 0
		with compiler.panic_arithmetic( 'bounded by byte_len, cannot overflow' ):
			byte_len: usize = self.__byte_size - 1
			while i < byte_len:
				c: u8 = self.__data[i]
				if ( c & 0xC0 ) != 0x80:
					if count == max_count:
						return self._byte_slice( 0, i )
					count += 1
				i += 1
		return self._byte_slice( 0, i )

	@private
	def _ascii_escape( self ) -> str:
		''' f-string !a conversion's second pass, run on whatever text
		__repr__() already produced (_lower_fstring_part calls __repr__
		then this, matching Python's own ascii() == escape(repr(x))).
		self is already fully escaped/quoted ASCII except for any
		printable non-ASCII codepoints __repr__ deliberately left as
		literal UTF-8 - so this only escapes THOSE (codepoints >= 0x80),
		leaving every ASCII byte (including the backslashes/quotes
		__repr__ itself inserted) untouched, matching Python's ascii()
		against an already-repr'd string. __str.py's own
		nonascii_escape_width/nonascii_escape_one do the actual
		per-codepoint work; same size-then-fill shape every string
		builder here uses. '''
		self_len: usize = self.byte_len()
		new_size: usize = 1 # zero terminator
		i: usize = 0
		consumed: usize = 0
		with compiler.panic_arithmetic( 'irrational string length' ):
			while i < self_len:
				cp: u32 = decode_utf8_at( self.__data, i, compiler.addrof( consumed ))
				new_size += nonascii_escape_width( cp )
				i += consumed

		new_buf: Ptr[u8] = sys.alloc[u8]( new_size )
		out: usize = 0
		i = 0
		with compiler.wrap_arithmetic:
			while i < self_len:
				cp = decode_utf8_at( self.__data, i, compiler.addrof( consumed ))
				out += nonascii_escape_one( new_buf, out, cp )
				i += consumed
		new_buf[out] = 0
		return str._from_owned_cstr( new_buf, new_size ).unwrap( 'invalid UTF-8 in _ascii_escape' )

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
			found: isize = self.find( sep, start )
			if found == isize( -1 ):
				result.append( self._byte_slice( start, self_len ))
				break
			with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
				match_start: usize = usize( found )
			result.append( self._byte_slice( start, match_start ))
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
			return self
		return self._byte_slice( prefix.byte_len(), self.byte_len() )

	def removesuffix( self, suffix: str ) -> str:
		''' self with suffix removed if present, else an unchanged copy -
		matches Python's str.removesuffix(). '''
		if not self.endswith( suffix ):
			return self
		with compiler.wrap_arithmetic: # suffix_len <= self_len, endswith() just confirmed it
			end: usize = self.byte_len() - suffix.byte_len()
		return self._byte_slice( 0, end )

	def rfind( self, sub: str ) -> isize:
		''' byte offset of the LAST occurrence of sub in self - mirrors
		find() above exactly, just scanning from the end, and shares its
		-1-means-not-found convention (see rindex() below for the Result-
		returning variant). Empty sub matches at self's own end
		(self_len), the mirror image of find('')'s own vacuous match at
		offset 0. '''
		self_len: usize = self.byte_len()
		sub_len: usize = sub.byte_len()
		if sub_len == 0:
			with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
				return isize( self_len )
		if sub_len > self_len:
			return isize( -1 )
		with compiler.wrap_arithmetic: # sub_len <= self_len, just checked above
			i: usize = self_len - sub_len
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			while True:
				candidate: ConstPtr[u8] = self.__data + i
				if sys.memcmp( candidate, sub.__data, sub_len ) == 0:
					return isize( i )
				if i == 0:
					break
				i -= 1
		return isize( -1 )

	def rindex( self, sub: str ) -> Result[usize, IndexError]:
		''' like rfind(), but returns a Result instead of a -1 sentinel -
		mirrors index()'s own relationship to find() above. Never
		panics. '''
		offset: isize = self.rfind( sub )
		if offset == isize( -1 ):
			return Result.Err( IndexError() )
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			return Result.Ok( usize( offset ) )

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
			found: isize = self.find( old, start )
			if found == isize( -1 ):
				break
			with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
				match_start: usize = usize( found )
				occurrences += 1
				start = match_start + old_len

		if occurrences == 0:
			return self

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
			if found == isize( -1 ):
				break
			with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
				match_start = usize( found )
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
		Takes list[str] rather than UnsafeList[str] (str.concat's own
		parameter type) - list[str] is the container every caller already
		has a piece of text collection in (e.g. split()'s own return type),
		and list[T]'s own methods are all lock-guarded per call (see
		__list.py's own header comment), so bridging to str.concat would
		still need an explicit copy into an UnsafeList[str] either way -
		not worth it just to share concat's own loop body. '''
		count: usize = parts.__len__()
		if count == 0:
			return ''
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
		found: isize = self.find( sep )
		if found == isize( -1 ):
			return ( self, '', '' )
		self_len: usize = self.byte_len()
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			idx: usize = usize( found )
			after_start: usize = idx + sep.byte_len()
		return ( self._byte_slice( 0, idx ), sep, self._byte_slice( after_start, self_len ))

	def rpartition( self, sep: str ) -> tuple[str,str,str]:
		''' like partition() above, but splits at the LAST occurrence of
		sep. Not found: ('', '', self) - the mirror image of partition()'s
		own no-match convention (rpartition searches from the end, so the
		whole string stays there). '''
		found: isize = self.rfind( sep )
		if found == isize( -1 ):
			return ( '', '', self )
		self_len: usize = self.byte_len()
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			idx: usize = usize( found )
			after_start: usize = idx + sep.byte_len()
		return ( self._byte_slice( 0, idx ), sep, self._byte_slice( after_start, self_len ))

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
			return self
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
			return self
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

	def center( self, width: usize, fillchar: str = ' ' ) -> str:
		''' pads self on both sides with fillchar until self's own codepoint
		count reaches width - matches Python's str.center() (an odd total
		padding amount puts the extra fill character on the RIGHT, same as
		Python - e.g. 'ab'.center(5) == ' ab  '). Already-long-enough is a
		no-op, same convention ljust/rjust above already use. '''
		if len( fillchar ) != 1:
			sys.panic( 'str.center(...): fillchar must be exactly one character' )
		self_count: usize = self.__len__()
		if self_count >= width:
			return self
		with compiler.panic_arithmetic( 'bounded by width, cannot overflow' ):
			pad_count: usize = width - self_count
		with compiler.panic_arithmetic( 'unreachable: dividing by the literal 2' ):
			left_count: usize = pad_count // 2
		with compiler.panic_arithmetic( 'bounded by pad_count, cannot overflow' ):
			right_count: usize = pad_count - left_count
		self_len: usize = self.byte_len()
		fill_len: usize = fillchar.byte_len()
		with compiler.panic_arithmetic( 'irrational string length' ):
			pad_bytes: usize = pad_count * fill_len
			new_size: usize = self_len + pad_bytes + 1
		new_buf: Ptr[u8] = sys.alloc[u8]( new_size )
		offset: usize = 0
		i: usize = 0
		while i < left_count:
			with compiler.wrap_arithmetic:
				sys.memcpy( new_buf + offset, fillchar.__data, fill_len )
				offset += fill_len
				i += 1
		with compiler.wrap_arithmetic:
			sys.memcpy( new_buf + offset, self.__data, self_len )
			offset += self_len
		i = 0
		while i < right_count:
			with compiler.wrap_arithmetic:
				sys.memcpy( new_buf + offset, fillchar.__data, fill_len )
				offset += fill_len
				i += 1
		new_buf[offset] = 0
		return str._from_owned_cstr( new_buf, new_size ).unwrap( 'invalid UTF-8 in center' )

	@private
	def _pad_after_prefix( self, prefix: str, width: usize, fill: str ) -> str:
		''' self (typically pre-sign-stripped magnitude digits) rjust-
		padded to (width - prefix's own codepoint count), with prefix
		then prepended - the "zero-padding goes BETWEEN a sign+radix-
		prefix and the digits" shape f-string format specs need for e.g.
		f"{-42:#010x}" (prefix="-0x", result "-0x0000002a") - zfill()
		can't express this on its own, since it only ever recognizes a
		bare leading '+'/'-' sign byte, not a multi-character prefix
		standing in front of where the padding needs to go. Deliberately
		saturating (not panicking) if prefix alone already reaches or
		exceeds width - same "the sign/prefix is never truncated, the
		result just ends up longer than the nominal width" leniency
		Python's own str.format() has for the same edge case, not a
		compile-time-provable-impossible situation like most of this
		codebase's own panic_arithmetic call sites. '''
		prefix_count: usize = prefix.__len__()
		with compiler.saturate_arithmetic:
			inner_width: usize = width - prefix_count
		return prefix + self.rjust( inner_width, fill )

	@private
	def _insert_thousands_sep( self, sep: str ) -> str:
		''' self (a plain, ASCII-only digit string - no sign, no radix/
		decimal-point punctuation) with `sep` inserted every 3 digits from
		the right, matching Python's own f"{1234567:,}" == '1,234,567'
		semantics - the shared grouping algorithm _pad_and_group_after_
		prefix below needs applied to an already-zero-padded digit string,
		not just a bare magnitude (int._decimal_digits_with_grouping and
		float's own _group_integer_part, lib/builtins/__float.py, each
		still carry their own earlier, narrower inline copy of this same
		loop shape - written before this method existed - rather than
		being retrofitted onto it here, to avoid touching already-shipped,
		tested code for a pure refactor). sep='' reconstructs the original
		text unchanged (splitting into groups of 3 and joining with
		nothing is a no-op), so callers never need to special-case "no
		grouping requested". '''
		count: usize = self.byte_len() # ASCII-only digit text - byte length is codepoint count here
		if count <= 3:
			# self is a BORROWED parameter (never pushed onto the epilogue
			# stack - cfg.py's _enter_parameter()) - a bare `return self`
			# already gets its own +1 from _stmt_Return's own aliasing-
			# return incref (lowering.py); an explicit compiler.incref(self)
			# here on top of that double-counts and LEAKS (confirmed via
			# ASAN/LeakSanitizer). NOT the same situation as str.concat's
			# own loop-local incref, which guards a fresh LOCAL (its own
			# live epilogue entry) rather than a directly-returned parameter.
			return self
		groups: list[str] = list[str]() # least-significant GROUP first
		end: usize = count
		with compiler.wrap_arithmetic:
			while end > 3:
				with compiler.panic_arithmetic( 'bounded by count, cannot overflow' ):
					start: usize = end - 3
				groups.append( self._byte_slice( start, end ))
				end = start
			groups.append( self._byte_slice( 0, end ))
			group_count: usize = groups.__len__()
			ordered: list[str] = list[str]() # most-significant GROUP first
			i: usize = group_count
			with compiler.panic_arithmetic( 'bounded by group_count, cannot underflow' ):
				while i > 0:
					i -= 1
					ordered.append( groups.__getitem__( i ).unwrap( '_insert_thousands_sep: index in bounds by construction' ))
		return sep.join( ordered )

	@private
	def _pad_and_group_after_prefix( self, prefix: str, width: usize, fill: str, sep: str ) -> str:
		''' the '0' zero-pad shorthand's own behavior when COMBINED with
		grouping (,/_), fixing a real bug: self (RAW, UNGROUPED magnitude
		digits - unlike _pad_after_prefix's own typical caller, which
		hands it an already-grouped string) is padded on the left with
		`fill` until, once the WHOLE padded field is grouped with `sep`
		every 3 digits from the right, it reaches (width - prefix's own
		codepoint count) - then `prefix` is prepended. Matches real Python:
		f"{1234567:015,d}" == '000,001,234,567', where the padding zeros
		themselves pick up their own comma separators too - NOT
		'0000001,234,567' (raw fill characters prepended in front of an
		ALREADY-grouped string), which is what pre-grouping self and then
		calling the plain _pad_after_prefix above would give instead (a
		real, confirmed bug found in exactly that shape - see
		PLAN_STR_FORMAT.md item 4's own writeup for the repro). Deliberately
		saturating/leniently overshooting the nominal width when prefix
		alone already reaches it, or when no padded digit count lands on
		an EXACT grouped-width match (grouping separators don't land at
		every digit count) - same leniency _pad_after_prefix's own comment
		documents, confirmed against real Python as the oracle for this
		"overshoots the nominal width" case too, not just the ordinary
		one. sep='' behaves identically to a plain _pad_after_prefix call
		(see _insert_thousands_sep's own "no grouping requested"
		convention) - callers never need to special-case "no grouping". '''
		prefix_count: usize = prefix.__len__()
		with compiler.saturate_arithmetic:
			inner_width: usize = width - prefix_count
		d: usize = self.__len__()
		sep_len: usize = sep.__len__()
		with compiler.wrap_arithmetic:
			total_count: usize = d
			while True:
				grouped_width: usize = total_count
				if total_count > 0:
					with compiler.panic_arithmetic( 'bounded by total_count, cannot overflow' ):
						grouped_width += ( ( total_count - 1 ) // 3 ) * sep_len
				if grouped_width >= inner_width:
					break
				total_count += 1
		padded: str = self.rjust( total_count, fill )
		return prefix + padded._insert_thousands_sep( sep )

	@private
	def _pad_and_group_before_dot( self, prefix: str, width: usize, fill: str, sep: str ) -> str:
		''' float's own equivalent of _pad_and_group_after_prefix above -
		self may contain a '.' (and, for 'e'/'E'/'g'/'G' text, an exponent
		suffix after that); only the portion BEFORE the first '.' is the
		groupable "integer part" that gets padded+grouped, everything from
		the '.' onward (fractional digits, any exponent) is left
		completely untouched and simply reappended - int has no such
		suffix to protect, hence the separate method rather than one
		shared shape. width already accounts for the untouched suffix's
		own length internally (subtracted before delegating to
		_pad_and_group_after_prefix), so callers pass the SAME nominal
		width the whole result should reach, same as every other format-
		spec width parameter in this codebase (callers wanting to reserve
		room for something this method doesn't know about, like '%'s own
		trailing literal character, subtract that themselves before
		calling - see lowering.py's _lower_float_format_spec). '''
		dot_index: usize = self.byte_len()
		dot_found: isize = self.find( '.' )
		if dot_found != isize( -1 ):
			with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
				dot_index = usize( dot_found )
		int_part: str = self._byte_slice( 0, dot_index )
		rest: str = self._byte_slice( dot_index, self.byte_len() )
		rest_len: usize = rest.__len__()
		with compiler.saturate_arithmetic:
			effective_width: usize = width - rest_len
		return int_part._pad_and_group_after_prefix( prefix, effective_width, fill, sep ) + rest

	@private
	def _pad_maybe_special( self, prefix: str, width: usize, fill: str, sep: str ) -> str:
		''' the '0' zero-pad shorthand's own dispatch for float digit text
		that might be "nan"/"inf" (lib/builtins/__float.py's own _f64_
		fixed_digits_raw/_f64_percent_digits_raw/_f64_none_type_digits_raw,
		special-cased there for non-finite values) instead of real digits.
		Real Python still zero-pads "nan"/"inf" (right-justified with
		fill, sign-aware - f"{inf:015,.1f}" == '000000000000inf'), but
		grouping and any '.'-based dot-splitting never apply to them EVEN
		when requested (no comma ever appears in that padded "inf" - not
		'000,000,000,inf') - _pad_and_group_before_dot's own grouping-
		aware machinery would incorrectly try to treat "nan"/"inf" as
		digits needing exactly that treatment (it has no way to know they
		aren't), so this checks first and routes to the plain (non-
		grouping, non-dot-aware) _pad_after_prefix instead when self isn't
		real digits. Lives here, not in __float.py, despite being float-
		motivated - self is the receiver float's code needs to dispatch
		on, and only a real str method (not a Scalar.names-registered free
		function) can be reached that way. '''
		if self == 'nan' or self == 'inf':
			return self._pad_after_prefix( prefix, width, fill )
		return self._pad_and_group_before_dot( prefix, width, fill, sep )

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
			return self
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

	def isalpha( self ) -> bool:
		''' True if self is non-empty and every codepoint is alphabetic
		(OS-native classification, __str.py's is_alpha_cp) - matches
		Python's str.isalpha() semantics (False for an empty string). '''
		self_len: usize = self.byte_len()
		if self_len == 0:
			return False
		i: usize = 0
		consumed: usize = 0
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			while i < self_len:
				cp: u32 = decode_utf8_at( self.__data, i, compiler.addrof( consumed ))
				if not is_alpha_cp( cp ):
					return False
				i += consumed
		return True

	def isdigit( self ) -> bool:
		''' True if self is non-empty and every codepoint is a digit
		(OS-native classification, __str.py's is_digit_cp) - matches
		Python's str.isdigit() semantics (False for an empty string). '''
		self_len: usize = self.byte_len()
		if self_len == 0:
			return False
		i: usize = 0
		consumed: usize = 0
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			while i < self_len:
				cp: u32 = decode_utf8_at( self.__data, i, compiler.addrof( consumed ))
				if not is_digit_cp( cp ):
					return False
				i += consumed
		return True

	def isdecimal( self ) -> bool:
		''' real Python distinguishes isdecimal() (Unicode category Nd
		only) from isdigit()/isnumeric() (broader) via fine-grained
		Unicode category data no OS classification API exposes directly -
		all three collapse onto is_digit_cp here, a documented
		approximation rather than a precise match (see TODO.txt). '''
		self_len: usize = self.byte_len()
		if self_len == 0:
			return False
		i: usize = 0
		consumed: usize = 0
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			while i < self_len:
				cp: u32 = decode_utf8_at( self.__data, i, compiler.addrof( consumed ))
				if not is_digit_cp( cp ):
					return False
				i += consumed
		return True

	def isnumeric( self ) -> bool:
		''' see isdecimal()'s own comment - the same documented
		approximation applies here too. '''
		return self.isdecimal()

	def isspace( self ) -> bool:
		''' True if self is non-empty and every codepoint is whitespace
		(OS-native classification, __str.py's is_space_cp) - matches
		Python's str.isspace() semantics (False for an empty string). '''
		self_len: usize = self.byte_len()
		if self_len == 0:
			return False
		i: usize = 0
		consumed: usize = 0
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			while i < self_len:
				cp: u32 = decode_utf8_at( self.__data, i, compiler.addrof( consumed ))
				if not is_space_cp( cp ):
					return False
				i += consumed
		return True

	def isupper( self ) -> bool:
		''' True if self is non-empty and every CASED codepoint is
		uppercase (OS-native classification, __str.py's is_upper_cp) -
		matches Python's str.isupper() semantics (False for an empty
		string; uncased codepoints like digits/punctuation don't
		themselves disqualify a match, but at least one cased codepoint
		must be upper for this to differ meaningfully from an all-
		uncased string - approximated here as "no codepoint is lowercase"
		since is_upper_cp already only fires on cased-uppercase
		codepoints, matching real Python closely for the common case). '''
		self_len: usize = self.byte_len()
		if self_len == 0:
			return False
		i: usize = 0
		consumed: usize = 0
		saw_cased: bool = False
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			while i < self_len:
				cp: u32 = decode_utf8_at( self.__data, i, compiler.addrof( consumed ))
				if is_lower_cp( cp ):
					return False
				if is_upper_cp( cp ):
					saw_cased = True
				i += consumed
		return saw_cased

	def islower( self ) -> bool:
		''' the mirror image of isupper() above. '''
		self_len: usize = self.byte_len()
		if self_len == 0:
			return False
		i: usize = 0
		consumed: usize = 0
		saw_cased: bool = False
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			while i < self_len:
				cp: u32 = decode_utf8_at( self.__data, i, compiler.addrof( consumed ))
				if is_upper_cp( cp ):
					return False
				if is_lower_cp( cp ):
					saw_cased = True
				i += consumed
		return saw_cased

	def isalnum( self ) -> bool:
		''' True if self is non-empty and every codepoint is alphabetic
		or a digit (OS-native classification, __str.py's is_alnum_cp) -
		matches Python's str.isalnum() semantics (False for an empty
		string). '''
		self_len: usize = self.byte_len()
		if self_len == 0:
			return False
		i: usize = 0
		consumed: usize = 0
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			while i < self_len:
				cp: u32 = decode_utf8_at( self.__data, i, compiler.addrof( consumed ))
				if not is_alnum_cp( cp ):
					return False
				i += consumed
		return True

	def isprintable( self ) -> bool:
		''' True if EVERY codepoint is printable (OS-native
		classification, __str.py's is_printable_cp) - matches Python's
		str.isprintable() semantics, including its one asymmetry with
		the other is*() methods here: True (vacuously) for an empty
		string. '''
		self_len: usize = self.byte_len()
		i: usize = 0
		consumed: usize = 0
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			while i < self_len:
				cp: u32 = decode_utf8_at( self.__data, i, compiler.addrof( consumed ))
				if not is_printable_cp( cp ):
					return False
				i += consumed
		return True

	def isidentifier( self ) -> bool:
		''' approximates Python's real XID_Start/XID_Continue Unicode
		tables using the alpha/digit classification primitives every
		other is*() method here uses (see TODO.txt) - first codepoint
		must be alphabetic or underscore, every codepoint after that
		alphabetic, a digit, or underscore. False for an empty string,
		matching Python. '''
		self_len: usize = self.byte_len()
		if self_len == 0:
			return False
		i: usize = 0
		consumed: usize = 0
		first: bool = True
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			while i < self_len:
				cp: u32 = decode_utf8_at( self.__data, i, compiler.addrof( consumed ))
				is_underscore: bool = cp == 0x5F # '_'
				if first:
					if not ( is_alpha_cp( cp ) or is_underscore ):
						return False
					first = False
				else:
					if not ( is_alpha_cp( cp ) or is_digit_cp( cp ) or is_underscore ):
						return False
				i += consumed
		return True

	def strip( self, chars: str|None = None ) -> str:
		''' self with leading AND trailing codepoints stripped - whitespace
		codepoints (is_space_cp) when chars is None, matching Python's
		str.strip() with no arguments; otherwise strips only codepoints
		that appear anywhere in chars, matching str.strip(chars). The
		chars= form only became possible once real union narrowing/
		extraction existed (see TODO.txt's own former "union
		disambiguation" blocker note, now resolved) - _should_strip_cp
		below is this method's own real use of it, via `match chars:`.
		Built from lstrip()/rstrip() below. '''
		return self.lstrip( chars ).rstrip( chars )

	def lstrip( self, chars: str|None = None ) -> str:
		''' self with leading codepoints stripped - whitespace
		(is_space_cp) when chars is None, matching Python's str.lstrip()
		with no arguments; otherwise strips only codepoints that appear
		anywhere in chars, matching str.lstrip(chars). '''
		self_len: usize = self.byte_len()
		i: usize = 0
		consumed: usize = 0
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			while i < self_len:
				cp: u32 = decode_utf8_at( self.__data, i, compiler.addrof( consumed ))
				if not self._should_strip_cp( cp, chars ):
					break
				i += consumed
		return self._byte_slice( i, self_len )

	def rstrip( self, chars: str|None = None ) -> str:
		''' self with trailing codepoints stripped - whitespace
		(is_space_cp) when chars is None, matching Python's str.rstrip()
		with no arguments; otherwise strips only codepoints that appear
		anywhere in chars, matching str.rstrip(chars). UTF-8 can only be
		decoded FORWARD, so trimming from the end needs one forward pass
		recording each codepoint's own start offset before walking that
		record backward. '''
		self_len: usize = self.byte_len()
		starts: UnsafeList[usize] = UnsafeList[usize]()
		i: usize = 0
		consumed: usize = 0
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			while i < self_len:
				starts.append( i )
				decode_utf8_at( self.__data, i, compiler.addrof( consumed ))
				i += consumed

		end: usize = self_len
		n: usize = starts.__len__()
		with compiler.panic_arithmetic( 'bounded by n, cannot overflow' ):
			while n > 0:
				start: usize = starts.__getitem__( n - 1 ).unwrap( 'rstrip: index in bounds by construction' )
				cp: u32 = decode_utf8_at( self.__data, start, compiler.addrof( consumed ))
				if not self._should_strip_cp( cp, chars ):
					break
				end = start
				n -= 1
		return self._byte_slice( 0, end )

	@private
	def _should_strip_cp( self, cp: u32, chars: str|None ) -> bool:
		''' shared by lstrip()/rstrip() (and so strip(), built from both):
		whitespace (is_space_cp) when chars is None, matching Python's
		strip()/lstrip()/rstrip() no-argument form; otherwise true iff cp
		appears anywhere in chars, matching Python's own chars= form. '''
		match chars:
			case None:
				return is_space_cp( cp )
			case str( c ):
				c_len: usize = c.byte_len()
				j: usize = 0
				j_consumed: usize = 0
				with compiler.panic_arithmetic( 'bounded by c_len, cannot overflow' ):
					while j < c_len:
						other_cp: u32 = decode_utf8_at( c.__data, j, compiler.addrof( j_consumed ))
						if other_cp == cp:
							return True
						j += j_consumed
				return False

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

	# supports `sub in some_str` (see lowering.py's _lower_in_comparison) -
	# reuses find()'s own byte-level scan rather than duplicating it
	def __contains__( self, sub: str ) -> bool:
		return self.find( sub ) != isize( -1 )

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

	def swapcase( self ) -> str:
		''' every cased codepoint flipped (upper<->lower), every uncased
		codepoint unchanged - matches Python's str.swapcase() for the
		common one-codepoint-in/one-codepoint-out case. A genuine one-to-
		many case expansion (German ß uppercasing to "SS") isn't
		supported: __str.py's case_map_one always maps exactly one
		codepoint to exactly one codepoint (see its own comment) - the
		same ceiling upper()/lower()'s own per-codepoint POSIX path
		already has. Two-pass (size, then fill), same shape case_map
		itself uses. '''
		self_len: usize = self.byte_len()
		if self_len == 0:
			return self
		new_size: usize = 1 # zero terminator
		i: usize = 0
		consumed: usize = 0
		with compiler.panic_arithmetic( 'irrational string length' ):
			while i < self_len:
				cp: u32 = decode_utf8_at( self.__data, i, compiler.addrof( consumed ))
				mapped: u32 = cp
				if is_upper_cp( cp ):
					mapped = case_map_one( cp, False )
				elif is_lower_cp( cp ):
					mapped = case_map_one( cp, True )
				new_size += utf8_encoded_len( mapped )
				i += consumed

		new_buf: Ptr[u8] = sys.alloc[u8]( new_size )
		out_i: usize = 0
		i = 0
		with compiler.wrap_arithmetic:
			while i < self_len:
				cp = decode_utf8_at( self.__data, i, compiler.addrof( consumed ))
				mapped = cp
				if is_upper_cp( cp ):
					mapped = case_map_one( cp, False )
				elif is_lower_cp( cp ):
					mapped = case_map_one( cp, True )
				out_i += encode_utf8_at( new_buf, out_i, mapped )
				i += consumed
		new_buf[out_i] = 0
		return str._from_owned_cstr( new_buf, new_size ).unwrap( 'invalid UTF-8 in swapcase' )

	def title( self ) -> str:
		''' the first codepoint of every alphabetic run uppercased, every
		other alphabetic codepoint lowercased, non-alphabetic codepoints
		unchanged - matches Python's str.title() (word boundaries are
		transitions into/out of an alphabetic run, not whitespace
		specifically - Python's own "they're bill's".title() ==
		"They'Re Bill'S", apostrophes aren't word characters, and neither
		are they here). Same one-to-many case-expansion ceiling
		swapcase() above has. '''
		self_len: usize = self.byte_len()
		if self_len == 0:
			return self
		new_size: usize = 1 # zero terminator
		i: usize = 0
		consumed: usize = 0
		prev_alpha: bool = False
		with compiler.panic_arithmetic( 'irrational string length' ):
			while i < self_len:
				cp: u32 = decode_utf8_at( self.__data, i, compiler.addrof( consumed ))
				mapped: u32 = cp
				if is_alpha_cp( cp ):
					mapped = case_map_one( cp, not prev_alpha )
					prev_alpha = True
				else:
					prev_alpha = False
				new_size += utf8_encoded_len( mapped )
				i += consumed

		new_buf: Ptr[u8] = sys.alloc[u8]( new_size )
		out_i: usize = 0
		i = 0
		prev_alpha = False
		with compiler.wrap_arithmetic:
			while i < self_len:
				cp = decode_utf8_at( self.__data, i, compiler.addrof( consumed ))
				mapped = cp
				if is_alpha_cp( cp ):
					mapped = case_map_one( cp, not prev_alpha )
					prev_alpha = True
				else:
					prev_alpha = False
				out_i += encode_utf8_at( new_buf, out_i, mapped )
				i += consumed
		new_buf[out_i] = 0
		return str._from_owned_cstr( new_buf, new_size ).unwrap( 'invalid UTF-8 in title' )

	def istitle( self ) -> bool:
		''' True if every uppercase codepoint follows an uncased
		codepoint and every lowercase codepoint follows a cased one, with
		at least one cased codepoint present - matches Python's
		str.istitle() (False for an empty string or an all-uncased
		string). '''
		self_len: usize = self.byte_len()
		i: usize = 0
		consumed: usize = 0
		prev_cased: bool = False
		saw_cased: bool = False
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			while i < self_len:
				cp: u32 = decode_utf8_at( self.__data, i, compiler.addrof( consumed ))
				if is_upper_cp( cp ):
					if prev_cased:
						return False
					prev_cased = True
					saw_cased = True
				elif is_lower_cp( cp ):
					if not prev_cased:
						return False
					prev_cased = True
					saw_cased = True
				else:
					prev_cased = False
				i += consumed
		return saw_cased

	@private
	@staticmethod
	def _from_owned_cstr( ptr: Ptr[u8], byte_size_including_zero_terminator: usize ) -> Result[str,CodecError]:
		# NOTE: a 0-byte before the end of the string is valid utf-8, so we
		# can't use cstrlen() here, we can only check to make sure the terminating 0 exists where expected
		if byte_size_including_zero_terminator == 0:
			return Result.Err( CodecError( 'utf-8', 'empty buffer' ))
		text_len: usize
		with compiler.wrap_arithmetic: # guaranteed to be > 0
			text_len = byte_size_including_zero_terminator - 1
		if ptr[text_len] != 0:
			return Result.Err( CodecError( 'utf-8', 'missing 0-terminator' ))

		# sparse codepoint-position index (see __index's own field comment) -
		# built alongside the validation scan below at zero extra passes.
		# Named char_index, not index - str.index() is a real instance
		# method, and this language has no local-shadows-outer-scope
		# semantics (see _existing_local_or_none's own comment).
		with compiler.panic_arithmetic( 'char_index sizing bounded by byte_size, cannot overflow' ):
			entries: usize = ( byte_size_including_zero_terminator >> 8 ) + 1
		char_index: Ptr[usize] = sys.alloc[usize]( entries )
		errdefer( sys.free( char_index ))
		char_count: usize = 0

		# walk through ptr and confirm valid utf-8 encoding or return CodecError
		i: usize = 0
		with compiler.panic_arithmetic( 'bounded by text_len, cannot overflow ' ):
			while i < text_len:
				if ( char_count & 0xFF ) == 0:
					char_index[char_count >> 8] = i
				byte1 = ptr[i]

				# 1-byte sequence (ASCII): 0xxxxxxx
				if (byte1 & 0x80) == 0x00:
					i += 1
					char_count += 1
					continue
				
				# unexpected continuation byte as a leading byte
				if (byte1 & 0xC0) == 0x80:
					return Result.Err( CodecError( 'utf-8', 'Unexpected continuation byte as leading byte' ))
				
				# 2-byte sequence: 110xxxxx 10xxxxxx
				elif (byte1 & 0xE0) == 0xC0:
					if i + 1 >= text_len:
						return Result.Err( CodecError( 'utf-8', 'Truncated 2-byte sequence' ))
					# Overlong encoding check: code point must be >= U+0080
					if byte1 < 0xC2:
						return Result.Err( CodecError( 'utf-8', 'Overlong 2-byte encoding' ))
					byte2 = ptr[i + 1]
					if (byte2 & 0xC0) != 0x80:
						return Result.Err( CodecError( 'utf-8', 'Invalid continuation byte in 2-byte sequence' ))
					i += 2
					char_count += 1

				# 3-byte sequence: 1110xxxx 10xxxxxx 10xxxxxx
				elif (byte1 & 0xF0) == 0xE0:
					if i + 2 >= text_len:
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
					char_count += 1

				# 4-byte sequence: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
				elif (byte1 & 0xF8) == 0xF0:
					if i + 3 >= text_len:
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
					char_count += 1

				# Invalid leading bytes (0xF5..0xFF)
				else:
					return Result.Err( CodecError( 'utf-8', 'Invalid leading byte' ))
		
		s: str = str.__allocate__(
			__data = ptr,
			__byte_size = byte_size_including_zero_terminator,
			__char_count = char_count,
			__index = char_index,
			__utf16 = None,
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
			return s
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
	# `if end:` (Python-style str truthiness - empty string is falsy) was
	# never actually implemented: str has no __bool__/truthiness dunder, so
	# a bare str used as a condition was silently testing its own (always
	# non-null, since a real str is always a valid heap object once
	# constructed) POINTER instead of its content - always true, regardless
	# of whether end was actually empty. Real, previously-latent bug, only
	# surfaced now that a str flowing into a bool-expected context is
	# checked at all. len(end) != 0 is the correct, explicit content check.
	if len( end ) != 0:
		sys.stdout.write( end ).unwrap( 'stdout write failed' )

# a single generic function now that bare-call monomorphization can infer T
# from the argument (see lowering.py's _lower_inferred_generic_call) -
# @inline (PLAN_INLINE.md) splices this straight to whatever T's own
# __len__ is at each call site, so len(x) costs exactly what x.__len__()
# would and no more - never a real Call/FuncStart/FuncEnd of its own.
# T: Sized (not a bare, unbound T) - see Sized's own docstring above for
# why: a real conformance error now surfaces at the call site itself.
@inline
def len[T: Sized]( t: T ) -> usize:
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

@overload
def max[T]( a: T, b: T ) -> T:
	return a if a >= b else b

@overload
def max[T, S: Iterable[T]]( seq: S ) -> T:
	# .__next__() called directly, not through next() - calling next() (an
	# @inline generic function) from a NESTED eager-lowered body (this
	# whole function's own T is return-only-inferred, per the eager
	# provisional-body trick _infer_return_only_type_params uses) hit a
	# separate, real inference gap: an @inline call's own type-param
	# resolution doesn't correctly infer through a Generator[T,E]-shaped
	# parameter the way an ordinary generic call does. Calling __next__()
	# directly sidesteps it entirely (next() itself works fine called
	# directly/non-nested - see its own definition below).
	it = iter( seq )
	value: T = it.__next__().unwrap( 'max(): empty sequence' )
	for t in it:
		value = max( value, t )
	return value

@overload
def min[T]( a: T, b: T ) -> T:
	return a if a <= b else b

@overload
def min[T, S: Iterable[T]]( seq: S ) -> T:
	it = iter( seq )
	value: T = it.__next__().unwrap( 'min(): empty sequence' )
	for t in it:
		value = min( value, t )
	return value

@inline
def next[T]( it: Generator[T, StopIteration] ) -> Result[T, StopIteration]:
	return it.__next__()

def iter[T, S: Iterable[T]]( seq: S ) -> Generator[T, StopIteration]:
	return seq.__iter__()

def any[T, S: Iterable[T]]( seq: S ) -> bool:
	for item in iter( seq ):
		if item:
			return True
	return False

def all[T, S: Iterable[T]]( seq: S ) -> bool:
	for item in iter( seq ):
		if not item:
			return False
	return True

def enumerate[T, S: Iterable[T]]( seq: S, start: isize = 0 ) -> Generator[tuple[isize,T], StopIteration]:
	# seq.__iter__() directly, not iter(seq): a for-loop INSIDE a generator's
	# own body is resolved by a separate, standalone-resolver desugaring pass
	# (deciding the loop's shape before the real generator transform runs),
	# which can't see through a nested bare call to another GENERIC free
	# function (iter[T,S:Iterable[T]]) the way ordinary (non-generator) code
	# can - confirmed by a real repro. seq.__iter__() is a plain attribute
	# call, resolved by receiver type alone, sidestepping that gap entirely.
	with compiler.wrap_arithmetic:
		for item in seq.__iter__():
			yield start, item
			start += 1

def map[T, U, S: Iterable[T]]( fn: Ptr[Callable[[T],U]], seq: S ) -> Generator[U, StopIteration]:
	for item in seq.__iter__(): # not iter(seq) - see enumerate's own comment above
		yield fn( item )

# IMPORTANT NOTE: because `for i in range( ... )` is such a common idiom, the
# compiler doesn't use the range functions below in those cases for performance
# reasons. the implementations below exist when you want to use range() in other
# ways like `it = iter( range( 10 ))`

def range( stop: isize ) -> Iterator[isize,StopIteration]:
	# FYI, see IMPORTANT NOTE above re range()
	i: isize = 0
	while i < stop:
		yield i
		i += 1

def range( start: isize, stop: isize ) -> Iterator[isize,StopIteration]:
	# FYI, see IMPORTANT NOTE above re range()
	i: isize = start
	while i < stop:
		yield i
		i += 1

def range( start: isize, stop: isize, step: isize ) -> Iterator[isize,StopIteration]:
	# FYI, see IMPORTANT NOTE above re range()
	i: isize = start
	while i < stop:
		yield i
		i += step

def reduce[T, S: Iterable[T]]( fn: Ptr[Callable[[T,T],T]], seq: S ) -> T:
	it = iter( seq )
	value1: T = it.__next__().unwrap( 'reduce(): empty sequence' )
	for value2 in it:
		value1 = fn( value1, value2 )
	return value1

def sum[T, S: Iterable[T]]( seq: S, start: T = 0 ) -> T:
	with compiler.wrap_arithmetic:
		for item in iter( seq ):
			start += item
	return start

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

class UnsafeDict[K, V]( Sized ):
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
		# V=NoneType is caught HERE, once, rather than in every method below
		# that casts a borrowed Ptr[None] through Ptr[V] (_owned_value,
		# _store_value, _release_value) - a dict/set can't be used at all
		# without going through this constructor first, so a single guard
		# here covers every one of those call sites for free. See
		# PLAN_NONETYPE_GENERIC_VALUE.md: Ptr[None] is ALSO this codebase's
		# own opaque/type-erased pointer spelling (RawDict's own key_ptr/
		# value_ptr fields, right below), so Ptr[V] with V=NoneType collides
		# with that meaning and compiles to real C void* - dereferencing it
		# (every one of those three methods' non-RC branch) is a genuine C
		# type error, not just a wrong answer. `if type(V) is None:` folds
		# away entirely (dead branch never lowered - see _try_fold_type_is_if,
		# type_resolver.py) for every OTHER V, so this costs nothing and
		# changes nothing for dict[K, str]/dict[K, i32]/etc.
		if type( V ) is None:
			compiler.error( 'dict[K, None] (and set[None]) are not supported - None cannot be a '
				'generic value-storage type (see PLAN_NONETYPE_GENERIC_VALUE.md); use dict[K, bool] instead' )
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
		# own incref (the dict's own stored reference stays valid too).
		# compiler.incref(...) is called UNCONDITIONALLY in both branches,
		# not just the is_rc(V) one - compiler.is_rc(V) is deliberately
		# is_rc_POINTER (true only when V's own runtime representation IS a
		# bare pointer), which answers the STORAGE-LAYOUT question below
		# (handle vs real heap-allocated struct copy) but NOT "does V need
		# RC bookkeeping at all" - a @union V whose RC-carrying leaf is a
		# plain RCClass (e.g. the builtin int) is exactly the is_rc(V)-
		# false-but-still-has-RC-leaves case: it takes the value-typed
		# struct-copy branch below for STORAGE, yet still owns a real RC
		# reference through its tag-gated leaf that must be incref'd here.
		# compiler.incref(...) itself already no-ops for a genuinely non-RC
		# V (a plain scalar/CStruct), so calling it unconditionally is safe
		# for every V - confirmed via a real repro (dict[str,Val] with Val a
		# @union whose RC leaf is int): skipping this incref for the value-
		# typed branch left the dict's own stored value under-retained,
		# heap-corruption-on-free.
		if compiler.is_rc( V ):
			v: V = compiler.cast( V, value_ptr )
			compiler.incref( v )
			return v
		else:
			ptr: Ptr[V] = compiler.cast( Ptr[V], value_ptr )
			v: V = ptr[0]
			compiler.incref( v )
			return v

	@staticmethod
	def _owned_key( key_ptr: Ptr[None] ) -> K:
		# mirrors _owned_value above, for K instead of V - returned OUT to
		# the caller (key_at), so an RC key needs its own incref the same
		# way, unconditionally in both branches (see _owned_value's own
		# comment on why is_rc(K) alone isn't the right gate for this)
		if compiler.is_rc( K ):
			k: K = compiler.cast( K, key_ptr )
			compiler.incref( k )
			return k
		else:
			ptr: Ptr[K] = compiler.cast( Ptr[K], key_ptr )
			k: K = ptr[0]
			compiler.incref( k )
			return k

	@staticmethod
	def _store_key( key: K ) -> Ptr[None]:
		# an OWNED copy for RawEntry to hold onto indefinitely - an RC key
		# just gets increfed (the object is already heap-owned, storing
		# its handle is enough); a value-typed key needs a real heap copy,
		# since RawEntry can't hold its bytes inline (it works on opaque
		# Ptr[None], see its own module docstring). compiler.incref(key) is
		# unconditional here too (see _owned_value's own comment) - the
		# value-typed branch's heap copy still needs its own leaf(s)
		# incref'd if K is itself a union with an RC leaf.
		if compiler.is_rc( K ):
			compiler.incref( key )
			return compiler.cast( Ptr[None], key )
		else:
			compiler.incref( key )
			buf: Ptr[None] = compiler.cast( Ptr[None], sys.alloc[u8]( compiler.sizeof( K )))
			ptr: Ptr[K] = compiler.cast( Ptr[K], buf )
			ptr[0] = key
			return buf

	@staticmethod
	def _store_value( value: V ) -> Ptr[None]:
		# see _store_key's own comment - identical reasoning, mirrored for V.
		if compiler.is_rc( V ):
			compiler.incref( value )
			return compiler.cast( Ptr[None], value )
		else:
			compiler.incref( value )
			buf: Ptr[None] = compiler.cast( Ptr[None], sys.alloc[u8]( compiler.sizeof( V )))
			ptr: Ptr[V] = compiler.cast( Ptr[V], buf )
			ptr[0] = value
			return buf

	@staticmethod
	def _release_key( key_ptr: Ptr[None] ) -> None:
		# compiler.decref(compiler.cast(K, key_ptr)) - NOT `existing: K =
		# compiler.cast(...); compiler.decref(existing)`. Binding the cast
		# result into a named local makes it an ordinary OWNED local (its own
		# scope-exit decref fires unconditionally, same as any other local) -
		# but compiler.cast(...) itself never increfs (it's a bare pointer
		# reinterpret of the SAME already-owned handle, not a fresh
		# allocation), so that auto-decref then fires in ADDITION to the
		# explicit one right here, over-releasing by one. A cast used bare, as
		# an expression (never bound to a name), produces a Temp that is never
		# fresh_temp()-registered - no auto-decref for it at all - so the
		# explicit compiler.decref(...) call is the only release, matching
		# this function's own job: release exactly the ONE reference the dict
		# itself held. Confirmed with a real UAF/AddressSanitizer repro
		# (masked in every existing dict test before this - they all only
		# ever used immortal string literal keys/values, whose release_object
		# is a guarded no-op regardless of how many times it's called).
		#
		# compiler.decref(...) is reached unconditionally now (both
		# branches), not just the is_rc(K) one - see _owned_value's own
		# comment on why is_rc(K) alone under-covers a value-typed K that's
		# still a union with an RC leaf. compiler.decref(...) itself already
		# no-ops for a genuinely non-RC K. The value-typed branch's own
		# decref target is a BARE `compiler.cast(Ptr[K], key_ptr)[0]`
		# expression, deliberately never bound to a name - same "a bound
		# local gets its own auto-decref too, double-releasing" pitfall this
		# comment already explains for the is_rc(K) branch above.
		if compiler.is_rc( K ):
			compiler.decref( compiler.cast( K, key_ptr ))
		else:
			compiler.decref( compiler.cast( Ptr[K], key_ptr )[0] )
			sys.free( compiler.cast( Ptr[u8], key_ptr ))

	@staticmethod
	def _release_value( value_ptr: Ptr[None] ) -> None:
		# see _release_key's own comment - identical reasoning, mirrored for V.
		if compiler.is_rc( V ):
			compiler.decref( compiler.cast( V, value_ptr ))
		else:
			compiler.decref( compiler.cast( Ptr[V], value_ptr )[0] )
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
		#
		# compiler.cast(K, a) == compiler.cast(K, b) - NOT bound into named
		# ka/kb locals first. See _release_key's own comment: a cast bound to
		# a name becomes an ordinary OWNED local with its own unconditional
		# scope-exit decref, but the cast itself never increfs - a's and b's
		# own borrowed handles (one from the dict's stored entry, one from the
		# caller's own lookup key) would each be released here on every
		# single comparison, a guaranteed double-free the moment either key is
		# a real (non-immortal) heap string. Used bare, as an expression, the
		# cast produces an untracked Temp - no auto-decref - matching what a
		# pure equality check actually needs: borrow both, compare, done.
		if compiler.is_rc( K ):
			return compiler.cast( K, a ) == compiler.cast( K, b )
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

	# deliberately does its OWN hash + _find_entry_idx call rather than
	# calling __getitem__ and discarding the Result - that path would
	# incref a found V via _owned_value for no reason, relying on
	# Result[T,E]'s own discarded-payload-decref to balance it back out.
	# A pure membership check never needs to touch _owned_value/_owned_key
	# at all, so there's nothing to balance and nothing to get wrong.
	def __contains__( self, key: K ) -> bool:
		h: u64 = self._hash_key( key )
		key_ptr: Ptr[None] = 0
		if compiler.is_rc( K ):
			key_ptr = compiler.cast( Ptr[None], key )
		else:
			key_ptr = compiler.cast( Ptr[None], compiler.addrof( key ))
		return self.__raw._find_entry_idx( h, key_ptr, _key_eq ).is_ok()

	# del d[key] syntax (lowering.py's _stmt_Delete)
	def __delitem__( self, key: K ) -> Result[None, KeyError]:
		h: u64 = self._hash_key( key )
		key_ptr: Ptr[None] = 0
		if compiler.is_rc( K ):
			key_ptr = compiler.cast( Ptr[None], key )
		else:
			key_ptr = compiler.cast( Ptr[None], compiler.addrof( key ))
		removed: RawEntry = self.__raw.remove_entry( h, key_ptr, _key_eq ).or_return()
		# removed.key_ptr/value_ptr are raw Ptr[None]s straight off RawEntry,
		# handed bare into _release_key/_release_value - same discipline
		# __del__ already follows (see _release_key's own comment: casting
		# a Ptr[None] back to K/V and binding it to a named local would
		# double-decref via the compiler's own scope-exit RC tracking).
		# Binding `removed` itself to a name IS safe - RawEntry is a plain
		# non-RC @cstruct of two Ptr[None] fields, not a K/V-typed value,
		# so that pitfall doesn't apply to it.
		self._release_key( removed.key_ptr )
		self._release_value( removed.value_ptr )
		return Result.Ok( None )

	def key_at( self, index: usize ) -> Result[K, IndexError]:
		# positional access into CURRENT LIVE-ENTRY order - not a fixed
		# mapping for the dict's whole lifetime any more. Still valid for
		# every index in [0, len): RawDict.remove_entry always keeps
		# __entries fully compacted (erase_at shifts left, never leaves a
		# hole), so there's no gap. But a removal renumbers every entry
		# AFTER the removed one down by one slot - key_at(i) can return a
		# different key after a removal than before, even for an i whose
		# own entry was never touched. Relative order among SURVIVING
		# entries is preserved (shift, not reorder).
		if index >= len( self.__raw ):
			return Result.Err( IndexError() )
		return Result.Ok( self._owned_key( self.__raw.key_ptr_at( index )))

	def value_at( self, index: usize ) -> Result[V, IndexError]:
		# see key_at's own comment - same positional access, for V
		if index >= len( self.__raw ):
			return Result.Err( IndexError() )
		return Result.Ok( self._owned_value( self.__raw.value_ptr_at( index )))

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

def _dict_key_iter[K, V]( d: dict[K, V] ) -> Generator[K, StopIteration]:
	# dict.__iter__ walks KEYS (matches real Python) - not routed through
	# the shared _sequence_iter[T,S:Sequence[T]] since dict[K,V] doesn't
	# conform to Sequence[K] itself (its own __getitem__ takes a K key, not
	# a usize index) - key_at(usize) is the index-based accessor instead.
	# Same double-call, no-intermediate-local shape as _sequence_iter et al
	# (see that comment for why).
	i: usize = 0
	while True:
		if d.key_at( i ).is_err():
			return
		yield d.key_at( i ).unwrap( 'dict.__iter__: was just checked is_ok() above' )
		with compiler.wrap_arithmetic:
			i += 1

class dict[K, V]( Iterable[K], Sized ):
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
		with self.__lock:
			return self.__inner.__len__()

	def __bool__( self ) -> bool:
		# Python-style truthiness: an empty dict is falsy - see str.__bool__'s
		# own docstring for why lowering.py's bare (non-union) truthiness
		# testing needs this dunder explicitly rather than defaulting to
		# always-true.
		return self.__len__() != 0

	def __getitem__( self, key: K ) -> Result[V, KeyError]:
		with self.__lock:
			return self.__inner.__getitem__( key )

	def __contains__( self, key: K ) -> bool:
		with self.__lock:
			return self.__inner.__contains__( key )

	def __delitem__( self, key: K ) -> Result[None, KeyError]:
		with self.__lock:
			return self.__inner.__delitem__( key )

	def key_at( self, index: usize ) -> Result[K, IndexError]:
		with self.__lock:
			return self.__inner.key_at( index )

	def value_at( self, index: usize ) -> Result[V, IndexError]:
		with self.__lock:
			return self.__inner.value_at( index )

	def __setitem__( self, key: K, value: V ) -> None:
		with self.__lock:
			self.__inner.__setitem__( key, value )

	def keys( self ) -> Generator[K, StopIteration]:
		return _dict_key_iter( self )

	def __iter__( self ) -> Generator[K, StopIteration]:
		return self.keys()

	def with_lock( self, body: Closure[[UnsafeDict[K, V]], None] ) -> None:
		# unlike list[T]'s own no-arg with_lock, body here takes the raw
		# UnsafeDict[K,V] directly - a compound read-modify-write (the
		# whole reason to reach for with_lock instead of two separate
		# locked calls) needs to touch storage WHILE the lock is held,
		# and __inner is private - body has no other way in. Calling back
		# through self.__getitem__/self.__setitem__ from inside body would
		# try to re-acquire this same (non-reentrant) FastLock and deadlock.
		with self.__lock:
			body( self.__inner )

# import this at the end because it depends on str etc to already be pre-parsed.
# BinaryReader/BinaryWriter/BinaryReadWriter are re-exported alongside File
# (not just File itself) because File's own factory methods hand them back
# to the caller as Result payloads - a caller holding one across multiple
# calls (e.g. a buffered reader keeping a handle alive) needs to be able to
# name the type in a field/parameter annotation, which an unexported name
# does not allow from outside lib/builtins. FileOpsInterface is exported for
# the same reason - lib/asyncfile.py (outside lib/builtins) subclasses it.
from .__File import File, BinaryReader, BinaryWriter, BinaryReadWriter, FileOpsInterface

# same "depends on bytearray already being pre-parsed" reasoning as File
# above - memoryview.__init__ takes a bytearray by name.
from .__memoryview import memoryview
