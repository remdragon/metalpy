# int.mpy
#
# Metal Py's built-in arbitrary-precision `int` type (see SYNTAX.md, Section 2:
# "Arbitrary-Precision Integer: int (Heap-managed, supports standard +, -, *
# operators; no implicit conversion to/from fixed-width integer types)").
#
# This is a native reimplementation of BigInt.c's decimal-digit-array
# algorithm -- not a wrapper around the C library. Every digit array is a
# `Ptr[u8]` owned directly by the `int` instance, and every operation
# (add/subtract/multiply/divide/parse/format) is the same schoolbook
# arithmetic BigInt.c used, ported to Metal Py.
#
# The single biggest difference from the C source: BigInt.c pulls in
# safe_math_impl.h for check_add_uint_uint / check_mul_int_int / etc. and
# calls one of those macros before nearly every arithmetic operation,
# because plain C `+`/`*` overflowing is silent undefined behavior. None of
# that machinery exists here. Bookkeeping arithmetic on array indices and
# digit counts is wrapped in a single `with compiler.panic_arithmetic(...)`
# per method (SYNTAX.md Section 5) instead -- one declarative statement
# instead of a manual checked-call before every `+`. Genuine, expected
# failure conditions (allocation failure, division by zero, malformed input,
# narrowing a value too big for a fixed-width type) still go through
# `Result[T, IntError]` rather than being panics, since those aren't bugs --
# they're normal outcomes a caller needs to handle.

import sys

# ---------------------------------------------------------------------------
# 1. Error model
# ---------------------------------------------------------------------------

# int's arbitrary precision means it never overflows the way a fixed-width
# type does -- the only ways an operation on `int` can actually fail are:
# running out of memory, dividing by zero, parsing a malformed digit
# string, or narrowing a value back down into something like i32.
#
# @union, not @enum: a CEnum member used as a real runtime value (which is
# exactly what every `Result.Err( IntError.X )` below needs) is a pre-
# existing, documented lowering.py gap (see emitter_c_test.py's own
# EmitCEnumTests comment, "CEnum never being referenced by real lowering.py
# output today") - confirmed directly here too (Result's own E type
# parameter came back inferred as BOTH i32 and IntError, from the same
# argument, at every one of this file's `Result.Err( IntError.X )` call
# sites, the moment any of them was actually exercised). sys.py's own
# OwnershipError carries the identical leftover comment for the identical
# reason - @union sidesteps it entirely since union-variant values ARE real,
# lowerable values (Result itself is one). Every variant is a plain no-
# payload tag (`: None`), constructed as `IntError.X( None )` rather than
# referenced bare.
@union
class IntError:
	DivideByZero: None
	InvalidDigit: None
	Overflow: None
	Other: None

# ---------------------------------------------------------------------------
# 2. The `int` class
# ---------------------------------------------------------------------------
#
# Digits are stored least-significant-first (index 0 = ones place), exactly
# as in BigInt.h's own comment: "Greater indices hold more significant
# digits." Each element holds one decimal digit, 0-9.

# ord() does not exist in this language (confirmed by direct inspection -
# no compiler/lib code implements it anywhere else); ASCII byte literals are
# spelled out directly instead, matching guid.py's own established idiom for
# the same thing.
_ASCII_MINUS: u8 = 0x2D # '-'
_ASCII_ZERO: u8 = 0x30 # '0'
_ASCII_NINE: u8 = 0x39 # '9'

# i32.min/i32.max are not real expressions in this language (scalars have no
# such members - confirmed by direct inspection); to_i32()'s own bounds
# check spells the literal range out instead, widened to i64 to match
# value's own type there.
_I32_MIN: i64 = -2147483648
_I32_MAX: i64 = 2147483647

class int:
	__digits: Ptr[u8]
	__num_digits: usize
	__num_allocated: usize
	__is_negative: bool

	# --- construction ---------------------------------------------------

	def __init__( self, value: i32 = 0 ) -> None:
		self.__is_negative = value < 0

		# Widen through i64 before negating so that i32.MIN (whose
		# magnitude doesn't fit in i32) converts correctly. BigInt_construct
		# in the C version just did `-value` on the original `int`, which is
		# undefined behavior for INT_MIN -- a bug this port sidesteps by
		# picking a wide-enough type up front instead of checking for it
		# after the fact.
		magnitude: u32
		with compiler.panic_arithmetic( 'i64 is wide enough to negate any i32 value (including i32.MIN) without overflow' ):
			magnitude = u32( -i64( value ) ) if self.__is_negative else u32( value )
		count: usize = int._digit_count( magnitude )

		digits: Ptr[u8] = sys.alloc[u8]( count )

		self.__digits = digits
		self.__num_allocated = count
		self.__num_digits = count

		with compiler.panic_arithmetic( 'writing at most 10 digits (u32 magnitude) into a buffer sized for exactly that many cannot overflow' ):
			i: usize = 0
			m: u32 = magnitude
			while i < count:
				self.__digits[i] = u8( m % 10 )
				m = m // 10
				i += 1

	@staticmethod
	@private
	def _digit_count( value: u32 ) -> usize:
		# Counts digits by repeated division rather than floor(log10(x))+1
		# the way BigInt_construct/BigInt_assign_int did in C. That
		# float-based approach misjudges exact powers of ten (e.g.
		# log10(1000) can round down to 2.999999... before the floor,
		# under-counting the digits) -- an integer loop has no such
		# rounding hazard.
		if value == 0:
			return 1
		with compiler.panic_arithmetic( 'a u32 has at most 10 decimal digits, so this counter cannot overflow usize' ):
			count: usize = 0
			v: u32 = value
			while v > 0:
				v = v // 10
				count += 1
			return count

	@staticmethod
	def from_str( s: str ) -> Result[int, IntError]:
		cstr: ConstPtr[u8] = s.get_cstr()
		length: usize = s.byte_len()

		if length == 0:
			return Result.Err( IntError.InvalidDigit( None ))

		is_negative: bool = cstr[0] == _ASCII_MINUS
		with compiler.panic_arithmetic( 'a leading sign character is at most one byte within the string\'s own length' ):
			start: usize = 1 if is_negative else 0

			# Skip leading zeros, but always leave at least one digit
			# character behind (so "0" and "-0" still parse to zero
			# instead of being treated as empty).
			pos: usize = start
			while pos < length - 1 and cstr[pos] == _ASCII_ZERO:
				pos += 1

			num_digits: usize = length - pos

		if num_digits == 0:
			return Result.Err( IntError.InvalidDigit( None ))

		digits: Ptr[u8] = sys.alloc[u8]( num_digits )

		with compiler.panic_arithmetic( 'walking a string of known length index-by-index cannot overflow usize bookkeeping' ):
			i: usize = 0
			while i < num_digits:
				ch: u8 = cstr[ length - 1 - i ]
				if ch < _ASCII_ZERO or ch > _ASCII_NINE:
					sys.free( digits )
					return Result.Err( IntError.InvalidDigit( None ))
				digits[i] = ch - _ASCII_ZERO
				i += 1

		result = int.__allocate__(
			__digits = digits,
			__num_digits = num_digits,
			__num_allocated = num_digits,
			__is_negative = False,
		)
		result.__is_negative = is_negative and not result.is_zero()
		return Result.Ok( result )

	def clone( self ) -> int:
		# Never fails (sys.alloc panics on OOM rather than returning an
		# error - see its own definition in lib/sys.py), so this returns a
		# bare `int` rather than Result[int, IntError]: every caller already
		# gets the simplest possible form (`result = self.clone()`, no
		# .or_return() ceremony for an outcome that structurally can't
		# happen), and there's no correctness downside - Result[int,...]
		# .or_return() from within int's own methods works correctly now
		# (see type_resolver.py's _type_of_expr, the 'or_return' branch -
		# fixed a real bug there: a bare `x = <result>.or_return()` with no
		# type annotation used to schedule or_return() itself as a real
		# compiled function while inferring x's type, which fails for ANY
		# same-file receiver type, self-referential or not), so this is a
		# genuine design choice now, not a workaround for that bug.
		capacity: usize = self.__num_allocated if self.__num_allocated > 0 else 1
		new_digits: Ptr[u8] = sys.alloc[u8]( capacity )

		with compiler.panic_arithmetic( 'copying an existing digit array of known length cannot overflow usize bookkeeping' ):
			i: usize = 0
			while i < self.__num_digits:
				new_digits[i] = self.__digits[i]
				i += 1

		return int.__allocate__(
			__digits = new_digits,
			__num_digits = self.__num_digits,
			__num_allocated = capacity,
			__is_negative = self.__is_negative,
		)

	def __del__( self ) -> None:
		sys.free( self.__digits )

	# --- internal digit-array helpers ------------------------------------
	# These operate on magnitudes only and never inspect/set sign; the
	# sign-aware public operators below compose them the same way
	# BigInt_add/BigInt_subtract in the C version composed
	# BigInt_add_digits/BigInt_subtract_digits.

	@private
	def _ensure_digits( self, digits_needed: usize ) -> Result[None, IntError]:
		if self.__num_allocated >= digits_needed:
			return Result.Ok( None )

		new_digits: Ptr[u8] = sys.alloc[u8]( digits_needed )

		with compiler.panic_arithmetic( 'copying the existing digit array into a strictly larger one cannot overflow usize bookkeeping' ):
			i: usize = 0
			while i < self.__num_digits:
				new_digits[i] = self.__digits[i]
				i += 1

		old_digits: Ptr[u8] = self.__digits
		self.__digits = new_digits
		self.__num_allocated = digits_needed
		sys.free( old_digits )
		return Result.Ok( None )

	@staticmethod
	@private
	def _compare_magnitude( a: int, b: int ) -> i8:
		if a.__num_digits > b.__num_digits:
			return 1
		if a.__num_digits < b.__num_digits:
			return -1

		with compiler.panic_arithmetic( 'walking down from an existing digit count to zero cannot underflow usize' ):
			i: usize = a.__num_digits
			while i > 0:
				i -= 1
				da: u8 = a.__digits[i]
				db: u8 = b.__digits[i]
				if da > db:
					return 1
				if da < db:
					return -1
		return 0

	# self += |other|, ignoring sign entirely.
	@private
	def _add_magnitude( self, other: int ) -> Result[None, IntError]:
		with compiler.panic_arithmetic( 'one more digit than the longer of two existing operands is always enough headroom and cannot overflow usize' ):
			longer: usize = self.__num_digits if self.__num_digits > other.__num_digits else other.__num_digits
			digits_needed: usize = longer + 1
		self._ensure_digits( digits_needed ).or_return()

		with compiler.panic_arithmetic( 'per-digit sums (two digits 0-9 plus a carry 0-1) stay far under u8\'s range, and the index/count bookkeeping is bounded by digits_needed' ):
			i: usize = 0
			carry: u8 = 0
			while i < other.__num_digits or carry > 0:
				if i == self.__num_digits:
					self.__digits[i] = 0
					self.__num_digits = self.__num_digits + 1
				other_digit: u8 = other.__digits[i] if i < other.__num_digits else 0
				total: u8 = self.__digits[i] + other_digit + carry
				self.__digits[i] = total % 10
				carry = 1 if total >= 10 else 0
				i += 1
		return Result.Ok( None )

	# self = ||self| - |other||, ignoring sign entirely. Caller is
	# responsible for working out the sign of the result beforehand, since
	# doing so after this call would require the pre-subtraction magnitude
	# comparison anyway (see __add__/__sub__ below).
	@private
	def _subtract_magnitude( self, other: int ) -> Result[None, IntError]:
		with compiler.panic_arithmetic( 'one more digit than the longer of two existing operands is always enough headroom and cannot overflow usize' ):
			longer: usize = self.__num_digits if self.__num_digits > other.__num_digits else other.__num_digits
			digits_needed: usize = longer + 1
		self._ensure_digits( digits_needed ).or_return()

		greater_digits: Ptr[u8]
		greater_num_digits: usize
		smaller_digits: Ptr[u8]
		smaller_num_digits: usize

		if int._compare_magnitude( self, other ) > 0:
			greater_digits = self.__digits
			greater_num_digits = self.__num_digits
			smaller_digits = other.__digits
			smaller_num_digits = other.__num_digits
		else:
			greater_digits = other.__digits
			greater_num_digits = other.__num_digits
			smaller_digits = self.__digits
			smaller_num_digits = self.__num_digits

		with compiler.panic_arithmetic( 'per-digit differences (two digits 0-9 plus a borrow) stay within i8, and the index/count bookkeeping is bounded by digits_needed' ):
			self.__num_digits = 1
			i: usize = 0
			borrow: i8 = 0
			while i < greater_num_digits:
				greater_digit: i8 = i8( greater_digits[i] )
				smaller_digit: i8 = i8( smaller_digits[i] ) if i < smaller_num_digits else i8( 0 )
				new_digit: i8 = greater_digit - smaller_digit - borrow
				if new_digit < 0:
					borrow = 1
					new_digit += 10
				else:
					borrow = 0
				self.__digits[i] = u8( new_digit )
				if new_digit != 0:
					self.__num_digits = i + 1
				i += 1
		return Result.Ok( None )

	# self = self * 10 + digit, done as an in-place digit shift instead of
	# a general multiply. BigInt_divide in the C version built each
	# quotient/remainder digit via a full BigInt_multiply(x, ten) followed
	# by BigInt_add_int(x, digit) -- an O(n*m) general multiply just to
	# multiply by ten. Shifting the digit array up by one slot and writing
	# the new digit into position 0 is the same operation in O(n).
	@private
	def _shift_and_add_digit( self, digit: u8 ) -> Result[None, IntError]:
		with compiler.panic_arithmetic( 'growing by exactly one digit cannot overflow usize' ):
			needed: usize = self.__num_digits + 1
		self._ensure_digits( needed ).or_return()

		with compiler.panic_arithmetic( 'shifting a digit array of known length up by one slot cannot overflow usize bookkeeping' ):
			i: usize = self.__num_digits
			while i > 0:
				self.__digits[i] = self.__digits[i - 1]
				i -= 1
			self.__digits[0] = digit
			self.__num_digits = self.__num_digits + 1

			while self.__num_digits > 1 and self.__digits[self.__num_digits - 1] == 0:
				self.__num_digits = self.__num_digits - 1
		return Result.Ok( None )

	# --- sign-aware comparison --------------------------------------------

	def compare( self, other: int ) -> i8:
		if self.__is_negative and not other.__is_negative:
			return -1
		if not self.__is_negative and other.__is_negative:
			return 1
		# Same sign: for two negatives, the one with the larger magnitude
		# is the smaller number, so the comparison flips.
		if self.__is_negative:
			return int._compare_magnitude( other, self )
		return int._compare_magnitude( self, other )

	def __eq__( self, other: int ) -> bool:
		return self.compare( other ) == 0

	def __ne__( self, other: int ) -> bool:
		return self.compare( other ) != 0

	def __lt__( self, other: int ) -> bool:
		return self.compare( other ) < 0

	def __le__( self, other: int ) -> bool:
		return self.compare( other ) <= 0

	def __gt__( self, other: int ) -> bool:
		return self.compare( other ) > 0

	def __ge__( self, other: int ) -> bool:
		return self.compare( other ) >= 0

	def is_zero( self ) -> bool:
		return self.__num_digits == 1 and self.__digits[0] == 0

	def is_negative( self ) -> bool:
		return self.__is_negative

	# --- arithmetic -------------------------------------------------------
	# Operators never mutate their operands: each clones `self` first and
	# mutates the clone, mirroring how BigInt_add/BigInt_subtract mutated
	# their first argument in place -- except here that argument is always
	# a private clone, since `int` is a value-like RC type rather than one
	# with in-place mutation semantics visible to callers.
	#
	# SYNTAX.md Section 5's arithmetic contexts (unwrapped/wrapped/
	# saturated) retarget +/-/* on *fixed-width* types to
	# checked_add/wrapped_add/saturated_add and friends. Arbitrary-precision
	# `int` has no such family of operations to retarget to -- there's no
	# "wrapping add" for a number with no fixed width -- so __add__/__sub__/
	# __mul__ always return Result[int, IntError] regardless of the
	# enclosing arithmetic context.

	def __add__( self, other: int ) -> Result[int, IntError]:
		result = self.clone()
		if result.__is_negative == other.__is_negative:
			result._add_magnitude( other ).or_return()
		else:
			result_is_negative = result.__is_negative if int._compare_magnitude( result, other ) > 0 else other.__is_negative
			result._subtract_magnitude( other ).or_return()
			# same "negative zero" guard __mul__ already documents on its own
			# sign assignment: opposite-sign operands with EQUAL magnitude
			# (e.g. 5 + (-5)) subtract down to exactly zero, but
			# _compare_magnitude(result, other) > 0 was false going in (the
			# magnitudes were equal, not result>other), so result_is_negative
			# came from other's sign regardless of the actual (zero) outcome
			# - without this guard, 5 + (-5) produces a "negative" zero that
			# compares unequal to (and less than) a plain int(0)
			result.__is_negative = result_is_negative and not result.is_zero()
		return Result.Ok( result )

	def __sub__( self, other: int ) -> Result[int, IntError]:
		result = self.clone()
		result_is_negative: bool = result.compare( other ) < 0
		if result.__is_negative == other.__is_negative:
			result._subtract_magnitude( other ).or_return()
		else:
			result._add_magnitude( other ).or_return()
		result.__is_negative = result_is_negative
		return Result.Ok( result )

	def __neg__( self ) -> Result[int, IntError]:
		result = self.clone()
		result.__is_negative = not result.__is_negative and not result.is_zero()
		return Result.Ok( result )

	def __mul__( self, other: int ) -> Result[int, IntError]:
		result = int( 0 )
		with compiler.panic_arithmetic( 'a product needs at most one more digit than the sum of its operands\' digit counts, which cannot overflow usize' ):
			capacity: usize = self.__num_digits + other.__num_digits + 1
		result._ensure_digits( capacity ).or_return()

		with compiler.panic_arithmetic( 'zero-filling a freshly sized buffer of known length cannot overflow usize bookkeeping' ):
			i: usize = 0
			while i < capacity:
				result.__digits[i] = 0
				i += 1
			result.__num_digits = capacity

		with compiler.panic_arithmetic( 'a single-digit product (<=81) plus an existing digit (<=9) plus a carry stays well within u16, and every index stays within the buffer sized above' ):
			i = 0
			while i < self.__num_digits:
				carry: u16 = 0
				j: usize = 0
				while j < other.__num_digits or carry > 0:
					other_digit: u16 = u16( other.__digits[j] ) if j < other.__num_digits else 0
					idx: usize = i + j
					total: u16 = u16( result.__digits[idx] ) + u16( self.__digits[i] ) * other_digit + carry
					result.__digits[idx] = u8( total % 10 )
					carry = total // 10
					j += 1
				i += 1

			while result.__num_digits > 1 and result.__digits[result.__num_digits - 1] == 0:
				result.__num_digits = result.__num_digits - 1

		# BigInt_multiply in the C version set is_negative unconditionally
		# from the two operands' signs, which produces a "negative zero"
		# whenever either operand is zero and the signs disagree (e.g.
		# 0 * -5). Guarding on is_zero() here fixes that.
		result.__is_negative = ( self.__is_negative != other.__is_negative ) and not result.is_zero()
		return Result.Ok( result )

	# --- division -----------------------------------------------------
	# Ports BigInt_divide's digit-by-digit long division: precompute 1x-9x
	# of |divisor|, then for each dividend digit (most significant first),
	# bring it down into a running remainder and pick the largest multiple
	# that still fits.
	#
	# One behavioral fix versus the original: BigInt_divide operated on
	# digit magnitudes only and never set a sign on its quotient or
	# remainder at all -- negative operands were silently treated as
	# positive. This port applies ordinary truncating-division semantics:
	# the quotient's sign is the xor of the operands' signs, and the
	# remainder takes the dividend's sign (matching C's own `/` and `%`).
	#
	# Returns a real tuple[int,int] (quotient, remainder) - see PLAN_TUPLE.md.
	# Used to return a hand-rolled `@cstruct class DivMod: quotient: int;
	# remainder: int` instead, because this language had no tuple type at
	# all (a bare `(int, int)` return-type annotation used to crash
	# discovery.py outright, AttributeError on a None .qualname). That
	# CStruct's own int fields were never actually torn down by the CFG on
	# scope exit either (cfg.py's v1 ownership tracking deliberately
	# doesn't reach into struct fields - see PLAN_TUPLE.md's own "why
	# RCClass, not CStruct" reasoning) - a real, if narrow, leak this
	# migration fixes as a side effect, not just a workaround removed.
	def divmod( self, divisor: int ) -> Result[tuple[int,int], IntError]:
		if divisor.is_zero():
			return Result.Err( IntError.DivideByZero( None ))

		base = divisor.clone()
		base.__is_negative = False

		# The 9 multiples (1x-9x of |divisor|), precomputed once so the
		# digit-picking loop below can search them per dividend digit
		# instead of recomputing. UnsafeList[int], not list[T] - this is a
		# purely local scratch buffer, never shared, on a hot arithmetic
		# path, so it shouldn't pay list[T]'s own per-call lock cost (see
		# __list.py's own header comment on this split). Originally
		# list[T].__getitem__(idx).unwrap(...) chained directly (T an RC
		# type, receiver never bound to a name) used to double-release the
		# payload: the receiver Temp's own pending cleanup (registered the
		# moment its owning Call/Allocate was emitted - see cfg.py's
		# fresh_temp) was never cancelled when .unwrap()'s return (an alias
		# of the SAME reference, not a fresh incref) got bound to a name
		# instead, undercounting the refcount by one - confirmed with
		# AddressSanitizer, not just reasoning. Fixed in lowering.py's
		# _lower_call (the Temp-receiver branch alongside the existing
		# Variable-receiver one for is_ok/is_err/unwrap/unwrap_or) - that
		# fix applies equally regardless of which list type this is.
		multiples: UnsafeList[int] = UnsafeList[int]()
		multiples.append( base ).unwrap( 'divmod: multiples.append failed' )
		with compiler.panic_arithmetic( 'building exactly eight more multiples of the divisor cannot overflow usize bookkeeping' ):
			k: usize = 1
			while k < 9:
				prev: int = multiples.__getitem__( k - 1 ).unwrap( 'divmod: multiples index in bounds by construction' )
				next_multiple = prev.clone()
				next_multiple._add_magnitude( base ).or_return()
				multiples.append( next_multiple ).unwrap( 'divmod: multiples.append failed' )
				k += 1

		quotient = int( 0 )
		remainder = int( 0 )

		with compiler.panic_arithmetic( 'walking down from an existing digit count to zero, and building up quotient/remainder digits one at a time, cannot overflow usize' ):
			i: usize = self.__num_digits
			while i > 0:
				i -= 1
				remainder._shift_and_add_digit( self.__digits[i] ).or_return()

				new_digit: u8 = 0
				d: usize = 9
				while d >= 1:
					candidate: int = multiples.__getitem__( d - 1 ).unwrap( 'divmod: multiples index in bounds by construction' )
					if int._compare_magnitude( remainder, candidate ) >= 0:
						remainder._subtract_magnitude( candidate ).or_return()
						new_digit = u8( d )
						break
					d -= 1
				quotient._shift_and_add_digit( new_digit ).or_return()

		quotient.__is_negative = ( self.__is_negative != divisor.__is_negative ) and not quotient.is_zero()
		remainder.__is_negative = self.__is_negative and not remainder.is_zero()

		return Result.Ok( ( quotient, remainder ))

	def __floordiv__( self, other: int ) -> Result[int, IntError]:
		result = self.divmod( other ).or_return()
		return Result.Ok( result[0] )

	def __mod__( self, other: int ) -> Result[int, IntError]:
		result = self.divmod( other ).or_return()
		return Result.Ok( result[1] )

	# --- narrowing / widening conversions ------------------------------
	# int never implicitly converts to/from fixed-width types (SYNTAX.md
	# Section 2), so both directions are explicit, fallible calls.
	
	@staticmethod
	def from_i32( value: i32 ) -> Result[int, IntError]:
		return Result.Ok( int( value ))
	
	def to_i32( self ) -> Result[i32, IntError]:
		# A 19-digit number is the most that can possibly fit in an i64
		# accumulator (i64::MAX has 19 digits), so bailing out above that
		# threshold up front means the accumulation loop itself can never
		# overflow -- no need for BigInt_to_int's per-digit
		# check_mul_int_int/check_add_int_int guards.
		if self.__num_digits > 19:
			return Result.Err( IntError.Overflow( None ))
		
		with compiler.panic_arithmetic( 'accumulating at most 19 decimal digits into an i64 cannot overflow, guaranteed by the digit-count check above' ):
			value: i64 = 0
			i: usize = self.__num_digits
			while i > 0:
				i -= 1
				value = value * 10 + i64( self.__digits[i] )
			if self.__is_negative:
				value = -value
		
		if value < _I32_MIN or value > _I32_MAX:
			return Result.Err( IntError.Overflow( None ))
		with compiler.wrap_arithmetic: # already range-checked above; this narrows, never truncates
			return Result.Ok( i32( value ) )
	
	# --- string conversion ----------------------------------------------
	#
	# Builds the digits (most-significant digit first, i.e. reversed from
	# our own internal storage order) directly into a fresh sys.alloc'd
	# buffer, then hands ownership straight to str.from_cstr - bytearray
	# has no zero-argument constructor and no append() method (it's a
	# fixed-size buffer, sized once at construction - see its own
	# definition in lib/builtins/__init__.py), so it can't support this
	# grow-as-you-go usage at all.
	def __str__( self ) -> str:
		with compiler.panic_arithmetic( 'a digit count plus an optional sign byte plus one zero terminator cannot overflow usize' ):
			size: usize = self.__num_digits + 1 # +1 for the zero terminator
			if self.__is_negative:
				size += 1

		buf: Ptr[u8] = sys.alloc[u8]( size )

		pos: usize = 0
		if self.__is_negative:
			buf[0] = _ASCII_MINUS
			pos = 1

		with compiler.panic_arithmetic( 'walking down from an existing digit count to zero cannot underflow usize, and pos stays within the buffer sized above' ):
			i: usize = self.__num_digits
			while i > 0:
				i -= 1
				buf[pos] = self.__digits[i] + _ASCII_ZERO
				pos += 1
			buf[pos] = 0

		return str._from_owned_cstr( buf, size ).unwrap( 'int.__str__: internal buffer was not valid UTF-8 (unreachable -- only ASCII digits and \'-\' are ever written)' )

	def __repr__( self ) -> str:
		return self.__str__()
