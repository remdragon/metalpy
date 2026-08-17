# proof-of-concept: i32's own __add__/__wrapped_add__/__saturated_add__,
# registered via the same `Scalar.method = fn` sigil __float.py already uses
# for f64/f32's __str__/__repr__ (discovery.py's visit_Assign) - not a new
# capability, just a new user of it. Each wraps a new compiler.checked_add/
# wrapped_add/saturated_add intrinsic (lowering.py) and is @inline, so binop
# dispatch's mode-qualified dunder lookup (_lower_binop_values'
# _mode_qualified_dunder_names) resolves straight back to a single opcode at
# each call site - zero call overhead, byte-identical codegen to the direct-
# opcode path this replaces for i32 specifically.
#
# @fallible_arithmetic marks __add__ (the ambient-mode-participating one - see
# _emit_binop_dunder_call): its Result[i32,OverflowError] gets fed through
# the same OrJump/Unwrap consumption a bare checked `+` already used,
# uniformly whether the call ends up inlined or not. __wrapped_add__/
# __saturated_add__ are genuinely infallible (AddWrap/AddSaturate carry no
# checked_errors), so they're plain i32-returning functions, not @fallible_arithmetic -
# under wrap/saturate mode, binop dispatch tries the qualified name first
# and finds these; int (which never registers either) always falls through
# to plain __add__ instead, staying mode-independent with no special-casing.

@fallible_arithmetic
@private
@inline
def i32__add__i32( value: i32, other: i32 ) -> Result[i32,OverflowError]:
	return compiler.checked_add( value, other )

@private
@inline
def i32__wrapped_add__i32( value: i32, other: i32 ) -> i32:
	return compiler.wrapped_add( value, other )

@private
@inline
def i32__saturated_add__i32( value: i32, other: i32 ) -> i32:
	return compiler.saturated_add( value, other )

i32.__add__ = i32__add__i32
i32.__wrapped_add__ = i32__wrapped_add__i32
i32.__saturated_add__ = i32__saturated_add__i32
