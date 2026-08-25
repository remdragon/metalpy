# Ptr[T]/ConstPtr[T] arithmetic dunders - Phase 3 of eliminating lowering.py's
# legacy direct-opcode fallback (see gen_scalar_dunders.py's own header comment
# for the overall design, ARCHITECTURE.md's "design decision" sections for
# the full rationale). Hand-written, not generator-produced - pointer
# arithmetic is always raw BYTE-offset (never sizeof(T)-scaled, confirmed via
# emitter_c.py's own _is_pointer_type/_emit_wrap_arith), so unlike the
# generated int/float file, there's no per-type body variation to templatize
# - one generic body per (op, mode) pair, specialized per pointee type T at
# the CALL SITE (not registration time - see lowering.py's
# _resolve_receiver_generic_dunder for the mechanism this relies on, new for
# this rollout: a bare generic Function registered directly on Ptr/ConstPtr,
# with T bound from the receiver's own concrete pointee type at dispatch
# time, not from an explicit [T] subscript at registration time the way
# every other generic in this codebase works).
#
# Receiver parameter is named `value`, NOT `self` - discovery.py strips a
# parameter literally named `self` as an implicit receiver even in a free
# function (confirmed the hard way - see memory file
# ptr_generic_dunder_spike_findings.md).
#
# Saturating pointer arithmetic is a clean compile-time rejection (confirmed
# with the user: an address isn't a bounded numeric range the way an int is,
# "clamp to min/max" has no coherent meaning) - compiler.saturated_ptr_add/
# sub always fail when actually lowered (discovery.fail is NoReturn), so
# these bodies never really "return" - the declared type just needs to be
# syntactically valid.
#
# Ptr[T] - Ptr[T] -> isize (byte distance, not element-scaled - consistent
# with +/- usize's own byte-offset convention) is a separate shape/dunder
# from Ptr[T] - usize -> Ptr[T], disambiguated by _find_dunder_for_arg's
# existing arg-type matching (same __sub__ name, two Overload-shaped
# implementations - one per arg_type).

@fallible_arithmetic
@inline
def ptr_add_checked[T]( value: Ptr[T], offset: usize ) -> Result[Ptr[T],OverflowError]:
	return compiler.checked_ptr_add( value, offset )

@inline
def ptr_add_wrapped[T]( value: Ptr[T], offset: usize ) -> Ptr[T]:
	return compiler.wrapped_ptr_add( value, offset )

@inline
def ptr_add_saturated[T]( value: Ptr[T], offset: usize ) -> Ptr[T]:
	return compiler.saturated_ptr_add( value, offset )

@fallible_arithmetic
@inline
def ptr_sub_offset_checked[T]( value: Ptr[T], offset: usize ) -> Result[Ptr[T],OverflowError]:
	return compiler.checked_ptr_sub( value, offset )

@inline
def ptr_sub_offset_wrapped[T]( value: Ptr[T], offset: usize ) -> Ptr[T]:
	return compiler.wrapped_ptr_sub( value, offset )

@inline
def ptr_sub_offset_saturated[T]( value: Ptr[T], offset: usize ) -> Ptr[T]:
	return compiler.saturated_ptr_sub( value, offset )

@inline
def ptr_sub_dist[T]( value: Ptr[T], other: Ptr[T] ) -> isize:
	return compiler.ptr_sub_dist( value, other )

@fallible_arithmetic
@inline
def const_ptr_add_checked[T]( value: ConstPtr[T], offset: usize ) -> Result[ConstPtr[T],OverflowError]:
	return compiler.checked_ptr_add( value, offset )

@inline
def const_ptr_add_wrapped[T]( value: ConstPtr[T], offset: usize ) -> ConstPtr[T]:
	return compiler.wrapped_ptr_add( value, offset )

@inline
def const_ptr_add_saturated[T]( value: ConstPtr[T], offset: usize ) -> ConstPtr[T]:
	return compiler.saturated_ptr_add( value, offset )

@fallible_arithmetic
@inline
def const_ptr_sub_offset_checked[T]( value: ConstPtr[T], offset: usize ) -> Result[ConstPtr[T],OverflowError]:
	return compiler.checked_ptr_sub( value, offset )

@inline
def const_ptr_sub_offset_wrapped[T]( value: ConstPtr[T], offset: usize ) -> ConstPtr[T]:
	return compiler.wrapped_ptr_sub( value, offset )

@inline
def const_ptr_sub_offset_saturated[T]( value: ConstPtr[T], offset: usize ) -> ConstPtr[T]:
	return compiler.saturated_ptr_sub( value, offset )

@inline
def const_ptr_sub_dist[T]( value: ConstPtr[T], other: ConstPtr[T] ) -> isize:
	return compiler.ptr_sub_dist( value, other )

# comparisons (==, !=, <, <=, >, >=) - plain address comparison, same
# infallible/no-mode-qualification shape as the scalar comparisons this
# mirrors (gen_scalar_dunders.py's scalar_eq/etc) - a C pointer compares
# natively with the same ir.Cmp opcode a scalar does (see
# _lower_compiler_cmp's own comment), so there's no pointer-specific
# codegen needed here either, just the dunder registration itself. <, <=,
# >, >= on a pointer are a real, meaningful (if less common) operation in
# C - ordering two addresses within the same allocation/array is
# well-defined - included for the same completeness reasons scalars got
# all 6, not just eq/ne.
@inline
def ptr_eq[T]( value: Ptr[T], other: Ptr[T] ) -> bool:
	return compiler.cmp_eq( value, other )

@inline
def ptr_ne[T]( value: Ptr[T], other: Ptr[T] ) -> bool:
	return compiler.cmp_ne( value, other )

@inline
def ptr_lt[T]( value: Ptr[T], other: Ptr[T] ) -> bool:
	return compiler.cmp_lt( value, other )

@inline
def ptr_le[T]( value: Ptr[T], other: Ptr[T] ) -> bool:
	return compiler.cmp_le( value, other )

@inline
def ptr_gt[T]( value: Ptr[T], other: Ptr[T] ) -> bool:
	return compiler.cmp_gt( value, other )

@inline
def ptr_ge[T]( value: Ptr[T], other: Ptr[T] ) -> bool:
	return compiler.cmp_ge( value, other )

@inline
def const_ptr_eq[T]( value: ConstPtr[T], other: ConstPtr[T] ) -> bool:
	return compiler.cmp_eq( value, other )

@inline
def const_ptr_ne[T]( value: ConstPtr[T], other: ConstPtr[T] ) -> bool:
	return compiler.cmp_ne( value, other )

@inline
def const_ptr_lt[T]( value: ConstPtr[T], other: ConstPtr[T] ) -> bool:
	return compiler.cmp_lt( value, other )

@inline
def const_ptr_le[T]( value: ConstPtr[T], other: ConstPtr[T] ) -> bool:
	return compiler.cmp_le( value, other )

@inline
def const_ptr_gt[T]( value: ConstPtr[T], other: ConstPtr[T] ) -> bool:
	return compiler.cmp_gt( value, other )

@inline
def const_ptr_ge[T]( value: ConstPtr[T], other: ConstPtr[T] ) -> bool:
	return compiler.cmp_ge( value, other )

Ptr.__add__ = ptr_add_checked
Ptr.__radd__ = ptr_add_checked          # usize + Ptr[T] - usize.__add__ already misses (arg-type mismatch), falls to reflected
Ptr.__wrapped_add__ = ptr_add_wrapped
Ptr.__wrapped_radd__ = ptr_add_wrapped
Ptr.__saturated_add__ = ptr_add_saturated
Ptr.__saturated_radd__ = ptr_add_saturated
Ptr.__sub__ = ptr_sub_offset_checked
Ptr.__sub__ = ptr_sub_dist               # Ptr[T]-Ptr[T]->isize: second __sub__ registration, merged into
                                          # an Overload by discovery.py; disambiguated from the offset shape
                                          # above by _find_dunder_for_arg's arg-type matching (usize vs
                                          # Ptr[T], the latter via TypeVar-wildcard matching - see that
                                          # method's own comment). Mode-independent: __wrapped_sub__/
                                          # __saturated_sub__ below only cover the offset shape (usize),
                                          # so under wrap/saturate mode a Ptr[T]-Ptr[T] lookup misses there
                                          # and falls back to this base __sub__ Overload, same as any other
                                          # mode-qualified-name-miss fallback.
Ptr.__wrapped_sub__ = ptr_sub_offset_wrapped
Ptr.__saturated_sub__ = ptr_sub_offset_saturated

ConstPtr.__add__ = const_ptr_add_checked
ConstPtr.__radd__ = const_ptr_add_checked
ConstPtr.__wrapped_add__ = const_ptr_add_wrapped
ConstPtr.__wrapped_radd__ = const_ptr_add_wrapped
ConstPtr.__saturated_add__ = const_ptr_add_saturated
ConstPtr.__saturated_radd__ = const_ptr_add_saturated
ConstPtr.__sub__ = const_ptr_sub_offset_checked
ConstPtr.__sub__ = const_ptr_sub_dist    # ConstPtr[T]-ConstPtr[T]->isize - see Ptr.__sub__ above
ConstPtr.__wrapped_sub__ = const_ptr_sub_offset_wrapped
ConstPtr.__saturated_sub__ = const_ptr_sub_offset_saturated

Ptr.__eq__ = ptr_eq
Ptr.__ne__ = ptr_ne
Ptr.__lt__ = ptr_lt
Ptr.__le__ = ptr_le
Ptr.__gt__ = ptr_gt
Ptr.__ge__ = ptr_ge

ConstPtr.__eq__ = const_ptr_eq
ConstPtr.__ne__ = const_ptr_ne
ConstPtr.__lt__ = const_ptr_lt
ConstPtr.__le__ = const_ptr_le
ConstPtr.__gt__ = const_ptr_gt
ConstPtr.__ge__ = const_ptr_ge

# __str__/__repr__ - the natural numeric address, hex, most-significant digit
# first (e.g. "0x7ff6a1234560"), matching how a debugger or C's %p shows a
# pointer - NOT binascii.hexlify() of the pointer's own raw storage bytes,
# which was considered and rejected: that would byte-dump the pointer's
# in-memory representation, giving REVERSED digit order on every little-
# endian target this compiler actually runs on (confirmed with the user).
# Getting the address as a real usize VALUE first (a native typed
# dereference through a reinterpret cast, not a raw memory copy) is what
# makes the result endian-correct - _ptr_addr's own `ConstPtr[usize]` cast
# lets the C compiler interpret those bytes as a usize the same way it
# would any other typed load, sidestepping byte-order entirely.
_HEX_DIGIT_CHARS: str = '0123456789abcdef'

@private
def _ptr_hex_digits( addr: usize ) -> str:
	''' addr's own hex digit text, most-significant digit first - mirrors
	gen_scalar_dunders.py's own scalar_str/i_str_unsigned decimal-digit
	loop, base 16 instead of base 10 (a pointer's own bit width is always a
	power of two, so digit extraction via %16/// never needs a signed-MIN
	special case the way that file's own signed decimal path does). '''
	if addr == 0:
		return '0'
	digits: list[str] = list[str]() # least-significant digit first
	with compiler.panic_arithmetic( 'dividing/moduloing a non-negative value by the literal 16 never zero-divides or overflows' ):
		v: usize = addr
		while v != 0:
			digit: usize = v % 16
			with compiler.panic_arithmetic( 'digit is bounded 0-15, cannot overflow' ):
				digit_end: usize = digit + 1
			digits.append( _HEX_DIGIT_CHARS._byte_slice( digit, digit_end ))
			v = v // 16
	count: usize = digits.__len__()
	ordered: list[str] = list[str]() # most-significant digit first
	i: usize = count
	with compiler.panic_arithmetic( 'bounded by count, cannot underflow' ):
		while i > 0:
			i -= 1
			ordered.append( digits.__getitem__( i ).unwrap( '_ptr_hex_digits: index in bounds by construction' ))
	return ''.join( ordered )

@private
@inline
def _ptr_str[T]( value: Ptr[T] ) -> str:
	# value's own numeric address, read as a real usize (not a raw byte
	# copy) via a reinterpret cast through compiler.addrof - same pointer-
	# reinterpret idiom lib/socket.py's own inet_ntop calls already use
	# (compiler.cast(Ptr[None], compiler.addrof(addr_val))). Inlined directly
	# here (not split into its own _ptr_addr[T] helper) - a nested generic-
	# to-generic call (one @inline generic calling ANOTHER by a bare name,
	# as opposed to via dunder/receiver dispatch) failed to resolve T at
	# emission time in a real repro building this (emitter_c.py's c_type
	# crashing on a still-unresolved TypeVar) - _ptr_hex_digits below is
	# NOT generic, so calling IT from here is an ordinary, unproblematic
	# call.
	addr_ptr: ConstPtr[usize] = compiler.cast( ConstPtr[usize], compiler.addrof( value ))
	addr: usize = addr_ptr[0]
	return '0x' + _ptr_hex_digits( addr )

@private
@inline
def _const_ptr_str[T]( value: ConstPtr[T] ) -> str:
	addr_ptr: ConstPtr[usize] = compiler.cast( ConstPtr[usize], compiler.addrof( value ))
	addr: usize = addr_ptr[0]
	return '0x' + _ptr_hex_digits( addr )

Ptr.__str__ = _ptr_str
Ptr.__repr__ = _ptr_str
ConstPtr.__str__ = _const_ptr_str
ConstPtr.__repr__ = _const_ptr_str
