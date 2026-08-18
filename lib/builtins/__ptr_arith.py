# Ptr[T]/ConstPtr[T] arithmetic dunders - Phase 3 of eliminating lowering.py's
# legacy direct-opcode fallback (see gen_scalar_arith.py's own header comment
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
