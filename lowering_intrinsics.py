# stdlib imports:
import ast
from typing import NoReturn

# local imports:
import cfg
import ir
from mpy_types import (
	Name, Type, Variable, Parameter, Overload, ClassLike, CType, Specialization, TaggedUnion, CStruct, CUnion, CEnum, TypeVar, RCClass, Scalar, FixedArrayType,
)

from lowering_shared import _CHECKED_BINOP_OPCODES, _CHECKED_FLOAT_BINOP_OPCODES
from mpy_types import is_float_scalar as _is_float_scalar

class CompilerIntrinsicsMixin:
	''' `compiler.*()` intrinsic lowering (sizeof, casts, atomics, C interop, debug hooks) - mixed into FunctionLowering (lowering.py), which
	see for the shared instance state (self._instructions, self._cfg, self.lowering,
	etc.) every method here reads and writes. Never instantiated on its own;
	split out of lowering.py purely to keep that file to a manageable size - see
	lowering.py's own class docstring and FunctionLowering's base-class list for
	the full set of sibling mixins this one is composed with. '''


	def _lower_compiler_sizeof( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.sizeof(T) is a compile-time constant whenever T is
		# already concrete - it folds directly to an ir.Const, no runtime
		# computation involved. T is normally a TYPE reference, resolved
		# via _try_resolve_namespace (same as a generic call's own [T]
		# argument); compiler.sizeof(x) also accepts a plain VALUE
		# expression (self, a local, self.field, ...) - _try_resolve_namespace
		# either fails to resolve those (Attribute chains) or resolves to a
		# Variable/Parameter rather than a Type (bare names, since every
		# declared param/local is registered by discovery.find_name too),
		# so falling back to _static_type_of_value_expr's own non-emitting
		# type lookup covers both without ever lowering/evaluating x itself
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.sizeof(...) takes exactly one type argument: {ast.unparse(node)}', node )
		resolved = self.lowering._try_resolve_namespace( node.args[0] )
		target_type = resolved if isinstance( resolved, Type ) else self._static_type_of_value_expr( node.args[0] )
		if target_type is None:
			self.lowering.discovery.fail( f'compiler.sizeof(...) argument must be a type or a value with a known type: {ast.unparse(node)}', node )
		if isinstance( target_type, TypeVar ):
			self.lowering.discovery.fail(
				f'compiler.sizeof({target_type.stem}) requires a concrete type - {target_type.qualname} is still an '
				f'unbound generic type parameter here (call the enclosing function through an explicit specialization, e.g. foo[SomeType](...))',
				node,
			)
		usize_cls = self.lowering.discovery.get_intrinsics()['usize']
		# `is not None`, NOT a truthy `:=` check - NoneType's own sizeof is
		# a legitimate 0 (see discovery.py's get_none_type()), and 0 is
		# falsy, so a truthy check here wrongly fell through to the
		# RCClass/CStruct/CUnion/TaggedUnion-only branch below and failed
		# with "compiler.sizeof(NoneType) is not supported yet" - see that
		# type's own sizeof field for why 0 there is real, not a "missing"
		# sentinel
		sizeof_attr = getattr( target_type, 'sizeof', None )
		if sizeof_attr is not None:
			return ir.Const( type = expected_type or usize_cls, value = sizeof_attr )
		# Ptr[T]/ConstPtr[T] is always exactly one machine pointer wide, whatever
		# T is - fold to the Ptr/ConstPtr intrinsic's own sizeof. A Specialization
		# carries no sizeof of its own, so the plain getattr above misses it;
		# without this, sys.alloc[Ptr[None]]( 1 ) on the POSIX threading path
		# (Thread.__init__'s pthread_create out-slot) fails to compile with
		# "compiler.sizeof(Ptr[None]) is not supported yet".
		if self.lowering._type_resolver._is_ptr_specialization( target_type ):
			return ir.Const( type = expected_type or usize_cls, value = target_type.base.sizeof )
		# a C type declared via compiler.c_type('pthread_mutex_t', ...) -
		# as opaque to this compiler as a real ClassLike; stays a real
		# ir.SizeOf, letting the C compiler itself compute it
		if isinstance( target_type, CType ):
			self.lowering.discovery.required_headers.add( target_type.required_header )
			dest = self._new_temp( expected_type or usize_cls )
			self._emit( ir.SizeOf( dest = dest, type = target_type ))
			return dest

		# a FixedArrayType field (u8[N]/u16[N]-style) - folds to a compile-
		# time constant the same way a plain Scalar's sizeof already does,
		# PROVIDED the element type itself has a plain-int sizeof (true for
		# every real element type this compiler's FixedArrayType is
		# actually exercised with today - u8/u16/etc). An element type
		# without one (a real class-like type, e.g. a hypothetical
		# SomeStruct[N] field) would need the C compiler's own sizeof(...)
		# to size correctly, same as any other class-like type below - not
		# attempted here since no real FixedArrayType field with a non-
		# scalar element type exists anywhere in this codebase yet; falls
		# through to the same "not supported yet" error below rather than
		# silently computing a wrong Python-int size for a type with no
		# sizeof of its own.
		if isinstance( target_type, FixedArrayType ):
			elem_sizeof = getattr( target_type.elem_type, 'sizeof', None )
			if elem_sizeof is not None:
				return ir.Const( type = expected_type or usize_cls, value = elem_sizeof * target_type.count )

		# a real class-like type (RCClass/CStruct/CUnion/TaggedUnion, or a
		# concrete Specialization of one) - no field-layout algorithm exists
		# in this compiler (nor should one - that's the C compiler's own
		# job), so unlike an intrinsic scalar's sizeof, this can't fold to a
		# Python int here. Stays a real ir.SizeOf instruction instead - the
		# emitter emits a literal C `sizeof(...)` expression, letting the
		# target C compiler compute the real, layout-dependent size (needed
		# by sys.alloc[T]'s own body, e.g. sys.alloc[SomeRCClass](1) for
		# RCClass construction - see _lower_allocate_fields's RCClass branch)
		base = target_type.base if isinstance( target_type, Specialization ) else target_type
		if not isinstance( base, ( RCClass, CStruct, CUnion, TaggedUnion )):
			self.lowering.discovery.fail( f'compiler.sizeof({target_type.qualname}) is not supported yet - only intrinsic scalar types and real classes have a known size', node )
		self.lowering.schedule( target_type )
		dest = self._new_temp( expected_type or usize_cls )
		self._emit( ir.SizeOf( dest = dest, type = target_type ))
		return dest

	def _lower_compiler_is_rc( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.is_rc(T) - a compile-time constant bool, true iff T is an
		# RCClass (possibly wrapped in a Specialization). T is always
		# concrete by the time this lowers (same "no unbound TypeVar"
		# requirement as compiler.sizeof), so this always folds directly to
		# an ir.Const - no runtime check, no emitter support needed at all.
		# Lets generic library code (list[T]'s own per-slot storage width -
		# an RCClass value IS a pointer everywhere else in this compiler,
		# but compiler.sizeof(T) deliberately stays the OBJECT's own struct-
		# body size always, for sys.alloc[T]'s sake - see its own docstring)
		# branch on T's own RC-ness without a new kind of type-level
		# reflection existing anywhere else in the language.
		#
		# Like compiler.sizeof(x), T also accepts a plain VALUE expression
		# (self, a local, ...) - moved here (from the outer Lowering class)
		# specifically so it can share compiler.sizeof(x)'s own non-
		# emitting _static_type_of_value_expr fallback, rather than
		# re-implementing a second, narrowing-unaware type lookup. Before
		# this fix, is_rc(x) on a value silently miscomputed instead of
		# erroring: _try_resolve_namespace(x) returns the VARIABLE (not a
		# Type) for a bare Name, and _is_RC(variable) - `variable.base if
		# isinstance(variable, Specialization) else variable` then
		# `isinstance(that, RCClass)` - is always False for a Variable,
		# regardless of the value's real type (compiler.is_rc(some_rc_var)
		# always folded to False, silently)
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.is_rc(...) takes exactly one type argument: {ast.unparse(node)}', node )
		resolved = self.lowering._try_resolve_namespace( node.args[0] )
		target_type = resolved if isinstance( resolved, Type ) else self._static_type_of_value_expr( node.args[0] )
		if target_type is None:
			self.lowering.discovery.fail( f'compiler.is_rc(...) argument must be a type or a value with a known type: {ast.unparse(node)}', node )
		if isinstance( target_type, TypeVar ):
			self.lowering.discovery.fail(
				f'compiler.is_rc({target_type.stem}) requires a concrete type - {target_type.qualname} is still an '
				f'unbound generic type parameter here (call the enclosing function through an explicit specialization, e.g. foo[SomeType](...))',
				node,
			)
		bool_cls = self.lowering.discovery.get_intrinsics()['bool']
		return ir.Const( type = expected_type or bool_cls, value = self.lowering._type_resolver._is_RC( target_type ))

	def _lower_compiler_error( self, node: ast.Call, expected_type: Type|None ) -> NoReturn:
		# compiler.error(msg) - a real compile-time diagnostic library code
		# can raise itself, e.g. a generic container guarding against a type
		# parameter binding it deliberately doesn't support (see
		# PLAN_NONETYPE_GENERIC_VALUE.md - UnsafeDict's own type(V) is None
		# guard is the motivating case). Before this, library code wanting a
		# custom compile-time message had no real mechanism - lib/ssl.py's
		# own comment on _MACOS_SSL_NOT_YET_IMPLEMENTED documents the
		# workaround (an intentionally undefined name, relying on the
		# generic "name is not defined" error) this replaces with a real,
		# purpose-written message. msg must be a literal string constant -
		# same "no runtime computation, compile-time only" posture every
		# other compiler.* intrinsic already has; a non-constant argument
		# would need actual VALUE evaluation to read a message that then
		# only matters if this call is even reachable, which is needlessly
		# more machinery than any real caller needs (every call site wants
		# a fixed, authored message, not a computed one).
		if len( node.args ) != 1 or node.keywords or not isinstance( node.args[0], ast.Constant ) or not isinstance( node.args[0].value, str ):
			self.lowering.discovery.fail( f'compiler.error(...) takes exactly one string-literal argument: {ast.unparse(node)}', node )
		self.lowering.discovery.fail( node.args[0].value, node )

	def _lower_compiler_refcount( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.refcount(x) - unlike compiler.sizeof(T), x is a real
		# VALUE (an RC object), not a type reference, so it's lowered via
		# _lower_expr like any other argument. A genuine runtime read (the
		# header's current count), not a compile-time constant - deliberately
		# opaque at this level (ir.RefCount), same spirit as Incref/Decref;
		# what it actually reads is a codegen/emitter concern, not this pass's
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.refcount(...) takes exactly one argument: {ast.unparse(node)}', node )
		value = self._lower_expr( node.args[0], None )
		if not self.lowering._type_resolver._is_RC( value.type ):
			self.lowering.discovery.fail(
				f'compiler.refcount(...) argument must be a reference-counted value, not '
				f'{value.type.qualname if value.type else "?"}: {ast.unparse(node)}',
				node,
			)
		usize_cls = self.lowering.discovery.get_intrinsics()['usize']
		dest = self._new_temp( expected_type or usize_cls )
		self._emit( ir.RefCount( dest = dest, value = value ))
		return dest

	def _lower_compiler_checked_binop( self, node: ast.Call, intrinsic_name: str, kind: str, mode: str, expected_type: Type|None ) -> ir.Operand:
		# compiler.checked_add(a, b)/wrapped_add(a, b)/saturated_add(a, b)/
		# checked_sub(...)/.../checked_truediv(a, b)/wrapped_truediv(a, b) -
		# FIXED primitives (unlike bare `+`, which reads self._arithmetic_
		# mode[-1] to pick BETWEEN AddCheck/AddWrap/AddSaturate): each of
		# these always resolves to exactly one opcode PER OPERAND TYPE (int
		# vs float pick a different opcode for the same kind/mode pair -
		# see _CHECKED_BINOP_OPCODES/_CHECKED_FLOAT_BINOP_OPCODES below -
		# but which one is picked never depends on ambient arithmetic mode).
		# Intended body for a scalar-registered, @inline'd `__add__`/
		# `__wrapped_add__`/`__saturated_add__`/etc (see lib/builtins/
		# __scalar_dunders.py) - which opcode a given SOURCE `+`/`//`/etc
		# actually gets still comes from ambient mode picking which of
		# these dunders binop dispatch resolves to (mode-qualified name
		# lookup in _lower_binop_values), not from anything read here.
		if len( node.args ) != 2 or node.keywords:
			self.lowering.discovery.fail( f'compiler.{intrinsic_name}(...) takes exactly two positional arguments: {ast.unparse(node)}', node )
		left = self._lower_expr( node.args[0], None )
		right = self._lower_expr( node.args[1], None )
		is_float = _is_float_scalar( left.type )
		opcode = ( _CHECKED_FLOAT_BINOP_OPCODES if is_float else _CHECKED_BINOP_OPCODES ).get(( kind, mode ))
		if not isinstance( left.type, Scalar ) or left.type is not right.type or opcode is None:
			self.lowering.discovery.fail(
				f'compiler.{intrinsic_name}(...) arguments must both be the same scalar type supporting {kind!r} - got '
				f'{left.type.qualname if left.type else "?"} and {right.type.qualname if right.type else "?"}: {ast.unparse(node)}',
				node,
			)
		result_type = expected_type if expected_type is not None and isinstance( expected_type, Scalar ) else left.type
		if not opcode.checked_errors:
			# wrapped_*/saturated_* on add/sub/mul, wrapped_truediv - fully
			# infallible, _lower_arithmetic_op's own infallible branch
			# (single instruction, no Result at all) is exactly right
			return self._lower_arithmetic_op( node, opcode, None, result_type, { 'left': left, 'right': right }, 'binary' )
		# checked_*, and wrapped_/saturated_floordiv/mod (still fallible -
		# ZeroDivisionError persists in every mode, see ir.py's own
		# DivWrap/DivSaturate/ModWrap/ModSaturate comments) - deliberately
		# does NOT auto-consume the way a bare checked `+` would
		# (_lower_arithmetic_op's own check-mode branch always does,
		# regardless of caller context - arithmetic modes don't translate
		# through a function call boundary, inline or not, by design). This
		# returns the RAW Result[result_type,error_type] value instead,
		# matching a @fallible_arithmetic dunder's own declared return type
		# exactly - so whatever calls/inlines this dunder can uniformly
		# consume it via ambient mode at the DISPATCH site
		# (_emit_binop_dunder_call's is_fallible_arithmetic handling), the
		# same way whether this call ends up inlined or not.
		result_cls = self.lowering.discovery.find_name( 'Result', node )
		error_type, _alternatives = self._resolve_checked_error( node, opcode, result_type )
		check_type = self.lowering.discovery._get_or_create_specialization( result_cls, [ result_type, error_type ] )
		self.lowering.schedule( check_type )
		# NOT monomorphized - _emit_check_arith (emitter_c.py) needs this
		# temp's type to stay a Specialization (reads .args[0] for the
		# success type) exactly like _emit_checked_op's identical, older
		# check_dest already relies on. This value deliberately escapes as
		# a real `return` value though (unlike _emit_checked_op's own
		# check_dest, always consumed same-statement via _consume_checked_
		# result) - see _stmt_Return's own comment on why it tries
		# monomorphize_class(value.type) too, not just value.type itself,
		# to match a Specialization-typed return against a function's
		# already-monomorphized declared return type.
		check_dest = self._new_temp( check_type )
		self._emit( opcode( dest = check_dest, left = left, right = right ))
		return check_dest

	def _lower_compiler_checked_convert( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.checked_convert(T, x) - the single fixed-mode intrinsic
		# behind every scalar .to_T() conversion method (lib/builtins/
		# __scalar_dunders.py) - a genuine numeric VALUE-range check against
		# T's own [MIN,MAX], independent of bit width (unlike compiler.
		# cast(T,x)/T(x) construct-cast syntax, which only range-checks a
		# NARROWING conversion - see _lower_scalar_cast). Always fallible
		# (ir.ConvertCheck), unlike checked_add/etc's wrapped_*/saturated_*
		# siblings - there's no "wrapped"/"saturated" variant of a value-
		# range check that means anything different, so .to_T() only ever
		# needs this one intrinsic (see mode-consumption at the DISPATCH
		# site - ambient mode still governs how the Result gets consumed,
		# same as any other @fallible_arithmetic method, just never changes
		# the check itself).
		if len( node.args ) != 2 or node.keywords:
			self.lowering.discovery.fail( f'compiler.checked_convert(...) takes exactly two arguments (target type, value): {ast.unparse(node)}', node )
		target_type = getattr( node.args[0], 'resolved_type', None )
		if target_type is None:
			target_type = self.lowering._try_resolve_namespace( node.args[0] )
		if not isinstance( target_type, Scalar ) or target_type.stem in ( 'f32', 'f64' ):
			self.lowering.discovery.fail(
				f'compiler.checked_convert(...) first argument must be an integer scalar type: {ast.unparse(node)}', node,
			)
		operand = self._lower_expr( node.args[1], None )
		if not isinstance( operand.type, Scalar ) or operand.type.stem in ( 'f32', 'f64' ):
			self.lowering.discovery.fail(
				f'compiler.checked_convert(...) second argument must be an integer scalar value, got '
				f'{operand.type.qualname if operand.type else "?"}: {ast.unparse(node)}',
				node,
			)
		result_cls = self.lowering.discovery.find_name( 'Result', node )
		overflow_cls = self.lowering.discovery.find_name( 'OverflowError', node )
		# NOT monomorphized - same reasoning as _lower_compiler_checked_binop's
		# own check_dest above: _emit_convert_check (emitter_c.py) needs
		# this temp's type to stay a Specialization (reads .args[0] for the
		# target type)
		check_type = self.lowering.discovery._get_or_create_specialization( result_cls, [ target_type, overflow_cls ] )
		self.lowering.schedule( check_type )
		check_dest = self._new_temp( check_type )
		self._emit( ir.ConvertCheck( dest = check_dest, operand = operand ))
		return check_dest

	# (kind, mode) -> opcode for Ptr[T]/ConstPtr[T] +/- usize -> Ptr[T].
	# Deliberately reuses the SAME AddCheck/AddWrap/SubCheck/SubWrap opcodes
	# _CHECKED_BINOP_OPCODES already uses for plain scalar add/sub - both
	# _emit_check_arith and _emit_wrap_arith (emitter_c.py) already branch
	# on a POINTER-typed dest_type correctly (byte-offset uintptr_t round-
	# trip, never sizeof(T)-scaled) - confirmed via Spike B, this already
	# works today via the (about-to-be-retired) fallback path, just needs
	# wiring through dunders now. No 'saturated' entry - saturating pointer
	# arithmetic is a clean compile-time rejection instead (confirmed with
	# the user: an address isn't a bounded numeric range the way an int is,
	# "clamp to min/max" has no coherent meaning) - see
	# _lower_compiler_ptr_binop's own handling of that mode.
	_PTR_BINOP_OPCODES: dict[tuple[str,str],type] = {
		( 'add', 'checked' ): ir.AddCheck, ( 'add', 'wrapped' ): ir.AddWrap,
		( 'sub', 'checked' ): ir.SubCheck, ( 'sub', 'wrapped' ): ir.SubWrap,
	}

	def _lower_compiler_ptr_binop( self, node: ast.Call, intrinsic_name: str, kind: str, mode: str, expected_type: Type|None ) -> ir.Operand:
		# compiler.checked_ptr_add(p, offset)/wrapped_ptr_add(...)/
		# saturated_ptr_add(...)/checked_ptr_sub(...)/wrapped_ptr_sub(...)/
		# saturated_ptr_sub(...) - the fixed-mode intrinsics behind Ptr[T]/
		# ConstPtr[T]'s own __add__/__sub__ dunders (lib/builtins/
		# __ptr_arith.py) for the Ptr[T] +/- usize -> Ptr[T] shape (pointer
		# MINUS pointer, yielding a distance, is a separate shape/dunder -
		# see compiler.ptr_sub_dist). Parallel to, not sharing code with,
		# _lower_compiler_checked_binop - that method hard-requires
		# left.type is right.type, but here the two operand types genuinely
		# differ (Ptr[T], usize).
		if len( node.args ) != 2 or node.keywords:
			self.lowering.discovery.fail( f'compiler.{intrinsic_name}(...) takes exactly two positional arguments: {ast.unparse(node)}', node )
		left = self._lower_expr( node.args[0], None )
		right = self._lower_expr( node.args[1], None )
		if not self.lowering._type_resolver._is_ptr_specialization( left.type ):
			self.lowering.discovery.fail(
				f'compiler.{intrinsic_name}(...) first argument must be a Ptr[T]/ConstPtr[T]: {ast.unparse(node)}', node,
			)
		usize_cls = self.lowering.discovery.get_intrinsics()['usize']
		if right.type is not usize_cls:
			self.lowering.discovery.fail(
				f'compiler.{intrinsic_name}(...) second argument must be usize, got '
				f'{right.type.qualname if right.type else "?"}: {ast.unparse(node)}',
				node,
			)
		if mode == 'saturated':
			self.lowering.discovery.fail(
				f'saturating pointer arithmetic is not supported (an address is not a bounded numeric range - '
				f'use checked or wrap mode instead): {ast.unparse(node)}',
				node,
			)
		opcode = self._PTR_BINOP_OPCODES[( kind, mode )]
		result_type = left.type
		if not opcode.checked_errors:
			return self._lower_arithmetic_op( node, opcode, None, result_type, { 'left': left, 'right': right }, 'binary' )
		result_cls = self.lowering.discovery.find_name( 'Result', node )
		error_type, _alternatives = self._resolve_checked_error( node, opcode, result_type )
		check_type = self.lowering.discovery._get_or_create_specialization( result_cls, [ result_type, error_type ] )
		self.lowering.schedule( check_type )
		check_dest = self._new_temp( check_type )
		self._emit( opcode( dest = check_dest, left = left, right = right ))
		return check_dest

	def _lower_compiler_ptr_diff( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.ptr_sub_dist(a, b) - Ptr[T]/ConstPtr[T] - Ptr[T]/ConstPtr[T]
		# -> isize, the fixed-opcode intrinsic behind Ptr[T]/ConstPtr[T]'s own
		# __sub__ dunder for the pointer-MINUS-pointer shape (ptr_sub_dist[T]
		# in lib/builtins/__ptr_arith.py) - distinct from the Ptr[T]-usize ->
		# Ptr[T] offset-subtraction shape _lower_compiler_ptr_binop handles
		# above (same __sub__ name, disambiguated at the dunder-lookup level
		# by _find_dunder_for_arg's arg-type matching, not here). Infallible,
		# single opcode - see ir.PtrDiff's own comment for why there's no
		# wrap/check/saturate split for a pointer distance.
		if len( node.args ) != 2 or node.keywords:
			self.lowering.discovery.fail( f'compiler.ptr_sub_dist(...) takes exactly two positional arguments: {ast.unparse(node)}', node )
		left = self._lower_expr( node.args[0], None )
		right = self._lower_expr( node.args[1], None )
		if not self.lowering._type_resolver._is_ptr_specialization( left.type ) or left.type is not right.type:
			self.lowering.discovery.fail(
				f'compiler.ptr_sub_dist(...) arguments must both be the same Ptr[T]/ConstPtr[T] type - got '
				f'{left.type.qualname if left.type else "?"} and {right.type.qualname if right.type else "?"}: {ast.unparse(node)}',
				node,
			)
		isize_cls = self.lowering.discovery.get_intrinsics()['isize']
		return self._lower_arithmetic_op( node, ir.PtrDiff, None, isize_cls, { 'left': left, 'right': right }, 'binary' )

	# ir.Shr (>>) joins these deliberately: right-shift by a valid amount is
	# always well-defined (this compiler doesn't check shift-amount-exceeds-
	# width for either direction - a separate, pre-existing, out-of-scope
	# concern), so it's single-opcode/infallible exactly like the bitwise
	# ops - unlike << (Shl), which DOES have real Wrap/Check/Saturate
	# variants (shifting bits out the top is a real, already-modeled
	# concern) and flows through _lower_compiler_checked_binop instead,
	# parallel to add/sub/mul.
	_BITWISE_OPCODES: dict[str,type] = { 'bitand': ir.BitAnd, 'bitor': ir.BitOr, 'bitxor': ir.BitXor, 'rshift': ir.Shr }

	def _lower_compiler_bitwise( self, node: ast.Call, intrinsic_name: str, expected_type: Type|None ) -> ir.Operand:
		# compiler.bitand/bitor/bitxor/rshift(a, b) - the fixed-opcode
		# intrinsics behind every scalar __and__/__or__/__xor__/__rshift__
		# dunder (lib/builtins/__scalar_dunders.py). Unlike checked_add/etc,
		# there is only ONE variant each - ir.BitAnd/BitOr/BitXor/Shr have
		# no Wrap/Check/Saturate forms at all - always infallible, plain
		# T-returning, no Result involved.
		if len( node.args ) != 2 or node.keywords:
			self.lowering.discovery.fail( f'compiler.{intrinsic_name}(...) takes exactly two positional arguments: {ast.unparse(node)}', node )
		left = self._lower_expr( node.args[0], None )
		right = self._lower_expr( node.args[1], None )
		opcode = self._BITWISE_OPCODES[intrinsic_name]
		if not isinstance( left.type, Scalar ) or left.type is not right.type or _is_float_scalar( left.type ):
			self.lowering.discovery.fail(
				f'compiler.{intrinsic_name}(...) arguments must both be the same integer scalar type - got '
				f'{left.type.qualname if left.type else "?"} and {right.type.qualname if right.type else "?"}: {ast.unparse(node)}',
				node,
			)
		result_type = expected_type if expected_type is not None and isinstance( expected_type, Scalar ) else left.type
		return self._lower_arithmetic_op( node, opcode, None, result_type, { 'left': left, 'right': right }, 'binary' )

	# comparison never overflows - single variant each, same "always
	# infallible, plain bool-returning" shape as the bitwise ops above, no
	# ambient arithmetic mode to disambiguate. Backs every scalar comparison
	# dunder (lib/builtins/__scalar_dunders.py's scalar_eq/scalar_ne/
	# scalar_lt/scalar_le/scalar_gt/scalar_ge) - the same fixed-opcode-
	# intrinsic pattern _BITWISE_OPCODES/_lower_compiler_bitwise already use,
	# letting ordinary comparison dispatch (_expr_Compare/_lower_eq_or_ne/
	# _lower_operand_compare/_classify_leaf_pair_eq) find a real dunder for
	# Scalar operands too, instead of hardcoding a flat ir.Cmp as the only
	# possible outcome for a Scalar left operand.
	_CMP_INTRINSIC_OPCODES: dict[str,'ir.CmpOp'] = {
		'cmp_eq': ir.CmpOp.EQ, 'cmp_ne': ir.CmpOp.NE,
		'cmp_lt': ir.CmpOp.LT, 'cmp_le': ir.CmpOp.LE,
		'cmp_gt': ir.CmpOp.GT, 'cmp_ge': ir.CmpOp.GE,
	}

	def _lower_compiler_cmp( self, node: ast.Call, intrinsic_name: str, expected_type: Type|None ) -> ir.Operand:
		# compiler.cmp_eq/cmp_ne/cmp_lt/cmp_le/cmp_gt/cmp_ge(a, b) -> bool -
		# emits the exact same ir.Cmp _expr_Compare's own flat-Cmp fallback
		# does (see its own docstring), just reachable as a real callable
		# intrinsic so a Scalar-, Ptr[T]/ConstPtr[T]-, or CEnum-registered
		# dunder body can delegate to it. Ptr[T]/ConstPtr[T] (a Specialization
		# wrapping a Scalar base, not itself `isinstance(_, Scalar)` - see
		# _is_ptr_specialization) needs the same plain address comparison a
		# bare scalar gets (a C pointer compares natively with ==/!=/</etc,
		# same ir.Cmp opcode, no different codegen) - backs Ptr.__eq__/etc
		# (lib/builtins/__ptr_arith.py). CEnum lowers to a plain C
		# typedef'd int (emitter_c.py's c_type - never struct/union-
		# prefixed), so it compares exactly the same native way too - backs
		# the auto-synthesized CEnum __eq__/etc (type_resolver.py's
		# _synthesize_cenum_comparisons).
		if len( node.args ) != 2 or node.keywords:
			self.lowering.discovery.fail( f'compiler.{intrinsic_name}(...) takes exactly two positional arguments: {ast.unparse(node)}', node )
		left = self._lower_expr( node.args[0], None )
		right = self._lower_expr( node.args[1], None )
		is_ptr = self.lowering._type_resolver._is_ptr_specialization( left.type )
		if not ( isinstance( left.type, ( Scalar, CEnum ) ) or is_ptr ) or left.type is not right.type:
			self.lowering.discovery.fail(
				f'compiler.{intrinsic_name}(...) arguments must both be the same scalar, Ptr[T]/ConstPtr[T], or CEnum type - got '
				f'{left.type.qualname if left.type else "?"} and {right.type.qualname if right.type else "?"}: {ast.unparse(node)}',
				node,
			)
		cmp_op = self._CMP_INTRINSIC_OPCODES[intrinsic_name]
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		dest = self._new_temp( bool_cls )
		self._emit( ir.Cmp( dest = dest, op = cmp_op, left = left, right = right ))
		return dest

	def _lower_compiler_addrof( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.addrof(x) -> Ptr[T], translating directly to C's &x - x
		# must be a bare local variable/parameter name (matches SYNTAX.md's
		# "local variable" wording and C's own lvalue-only restriction on
		# &), OR exactly one level of plain field access ROOTED at one
		# (x.field, e.g. compiler.addrof(addr.sin_addr) - the common "fill
		# one field of a stack struct via a C out-parameter" FFI idiom,
		# e.g. inet_pton(af, str, &addr.sin_addr)). Investigated and
		# confirmed safe for exactly this shape: a bare Name's own operand
		# (_expr_Name) is always a genuine, stable-lifetime Variable (never
		# a Temp) - requiring the CHAIN'S ROOT to be one too is what
		# preserves that same "well-defined, stable lvalue" guarantee one
		# level deeper (a plain field of an already-stable object is itself
		# just as stable/addressable - C's own `&x.field`/`&x->field`).
		# Deliberately NOT extended to a Call/Subscript/deeper chain root
		# (e.g. compiler.addrof(get_foo().field) or compiler.addrof(a.b.c)):
		# a Call's own result is a TEMPORARY whose lifetime this compiler's
		# RC discipline doesn't guarantee outlives the current statement (a
		# real dangling-pointer risk, not just an implementation gap), and a
		# deeper chain is un-analyzed extra scope, not needed by the one
		# real motivating case - both rejected with a clear message rather
		# than silently mishandled.
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.addrof(...) takes exactly one argument: {ast.unparse(node)}', node )
		arg_node = node.args[0]
		if isinstance( arg_node, ast.Subscript ) and isinstance( arg_node.value, ast.Attribute ):
			# compiler.addrof(x.field[i]) where field is a FixedArrayType ->
			# Ptr[ElemType] at element i specifically (unlike the bare
			# compiler.addrof(x.field) case above, which only ever gives
			# element 0 via array-to-pointer decay). Same root-must-be-a-
			# bare-local safety requirement as every other addrof shape
			# here, checked explicitly since _fixed_array_index_target
			# itself doesn't enforce it (by design - GetAttrIndex/
			# SetAttrIndex's own callers only ever need a VALUE, which
			# doesn't share addrof's dangling-pointer concern about the
			# root's lifetime).
			attr_node = arg_node.value
			if not isinstance( attr_node.value, ast.Name ):
				self.lowering.discovery.fail(
					f'compiler.addrof(...) indexed field-access argument must be rooted at a bare local variable, '
					f'not {ast.unparse(node)} (only one level of field access is supported)',
					node,
				)
			fixed = self._fixed_array_index_target( attr_node, arg_node.slice )
			if fixed is None:
				self.lowering.discovery.fail(
					f'compiler.addrof(...) subscript argument must index a fixed-size array field: {ast.unparse(node)}',
					node,
				)
			root, attr, array_type, index = fixed
			ptr_cls = self.lowering.discovery.get_intrinsics()['Ptr']
			elem_ptr_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ array_type.elem_type ] )
			dest = self._new_temp( elem_ptr_type )
			self._emit( ir.AddrOfArrayIndex( dest = dest, obj = root, attr = attr, index = index ))
			return dest
		if isinstance( arg_node, ast.Subscript ) and isinstance( arg_node.value, ast.Name ):
			# compiler.addrof(buf[i]) where buf ITSELF (a bare local/
			# parameter, not a field) is a FixedArrayType - same shape as
			# the field-rooted case just above, attr='' (see
			# _fixed_array_index_target's own docstring).
			fixed = self._fixed_array_index_target( arg_node.value, arg_node.slice )
			if fixed is not None:
				root, attr, array_type, index = fixed
				ptr_cls = self.lowering.discovery.get_intrinsics()['Ptr']
				elem_ptr_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ array_type.elem_type ] )
				dest = self._new_temp( elem_ptr_type )
				self._emit( ir.AddrOfArrayIndex( dest = dest, obj = root, attr = attr, index = index ))
				return dest
		if isinstance( arg_node, ast.Attribute ):
			if not isinstance( arg_node.value, ast.Name ):
				self.lowering.discovery.fail(
					f'compiler.addrof(...) field-access argument must be rooted at a bare local variable, '
					f'not {ast.unparse(node)} (only one level of field access is supported)',
					node,
				)
			root = self._lower_expr( arg_node.value, None )
			attr_var = self.lowering._attr_lookup( root.type, arg_node.attr, arg_node )
			self._check_field_visibility( root.type, attr_var, arg_node.attr, arg_node )
			ptr_cls = self.lowering.discovery.get_intrinsics()['Ptr']
			if isinstance( attr_var.type, FixedArrayType ):
				# compiler.addrof(x.field) where field is ElemType[N] ->
				# Ptr[ElemType], via C's own array-to-pointer decay - NOT
				# &(x.field), which would be a pointer TO the array
				# (ElemType(*)[N]), a different C type than the declared
				# Ptr[ElemType] destination even though the address value
				# is identical. Safe for the same reason ordinary field
				# addrof is: `obj` is a real, stable-lifetime lvalue (a
				# bare local, or one level of field access rooted at one),
				# and a fixed-size array member of a stable object is
				# itself just as stable. See ir.ArrayFieldPtr's own
				# docstring for the emission this builds.
				elem_ptr_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ attr_var.type.elem_type ] )
				dest = self._new_temp( elem_ptr_type )
				self._emit( ir.ArrayFieldPtr( dest = dest, obj = root, attr = arg_node.attr ))
				return dest
			pointee = attr_var.type
			# same RC-pointee-depth rule the bare-Name path below applies -
			# see its own comment for why (Ptr[Foo] already spells `Foo*`,
			# so &-ing an RC-typed FIELD needs the same extra Ptr level)
			if self.lowering._type_resolver._is_RC( pointee ):
				pointee = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ pointee ] )
			ptr_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ pointee ] )
			dest = self._new_temp( ptr_type )
			self._emit( ir.AddrOfField( dest = dest, obj = root, attr = arg_node.attr ))
			return dest
		if not isinstance( arg_node, ast.Name ):
			self.lowering.discovery.fail( f'compiler.addrof(...) argument must be a bare local variable, not {ast.unparse(node)}', node )
		ptr_cls = self.lowering.discovery.get_intrinsics()['Ptr']
		static_type = self._static_field_type_or_none( arg_node )
		if isinstance( static_type, FixedArrayType ):
			# compiler.addrof(buf) where buf ITSELF is a FixedArrayType local
			# -> Ptr[ElemType] via C's own array-to-pointer decay, same as
			# ArrayFieldPtr's field-rooted case (NOT plain AddrOf/&buf, which
			# would give ElemType(*)[N] - pointer-TO-array, a real type
			# mismatch against the declared Ptr[ElemType] destination). obj
			# is the local itself, attr='' (see _fixed_array_index_target's
			# own docstring for the convention). Uses _fixed_array_root_
			# operand, not _lower_expr, since a FixedArrayType has no
			# ordinary whole-value read to fall into (see its own docstring).
			root = self._fixed_array_root_operand( arg_node )
			elem_ptr_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ static_type.elem_type ] )
			dest = self._new_temp( elem_ptr_type )
			self._emit( ir.ArrayFieldPtr( dest = dest, obj = root, attr = '' ))
			return dest
		value = self._lower_expr( arg_node, None )
		# &value's C type is one pointer level deeper than value's OWN storage.
		# For a scalar/CStruct a variable stores the value directly, so that is
		# Ptr[T]. But an RCClass value is itself stored as a pointer (a `Foo`
		# variable holds a `Foo*`), so &value is `Foo**` - which in this type
		# system is Ptr[Ptr[Foo]], because Ptr[Foo] already spells `Foo*` (see
		# emitter_c._value_spelling's "single pointer even when RCClass" rule,
		# which sys.alloc[Foo] -> Ptr[Foo] relies on and must stay). Getting this
		# level right is what makes the emitted `dest = &value` type-check under
		# GCC and Linux-clang; Windows clang/MSVC only warned, so the old
		# Ptr[Foo] result (`Foo*` assigned from `Foo**`) slipped through. The one
		# caller that hits the RC path - list/UnsafeList.append/insert via
		# compiler.cast( Ptr[None], compiler.addrof( val ) ) - casts straight to
		# void*, so the extra level is invisible downstream. (expected_type is
		# intentionally NOT consulted: addrof's result type is determined solely
		# by the operand; callers wanting another pointer type cast explicitly.)
		pointee = value.type
		if self.lowering._type_resolver._is_RC( pointee ):
			pointee = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ pointee ] )
		ptr_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ pointee ] )
		dest = self._new_temp( ptr_type )
		self._emit( ir.AddrOf( dest = dest, value = value ))
		return dest

	def _c_field_name_literal( self, name_node: ast.expr, fn_name: str, node: ast.Call ) -> str:
		# one level of field access ONLY, same restriction compiler.addrof's
		# own field-access shape already has (see its own comment) - a
		# dotted path ('uc_stack.ss_sp') can't reuse GetAttr/SetAttr/
		# AddrOfField's existing 'attr: str' emission unchanged
		# (mangle_qualname corrupts a literal '.' into '$', producing wrong
		# C) - reaching a second-level opaque field means composing two
		# single-level compiler.c_field*() calls instead (see
		# _lower_compiler_c_field_addr's own comment)
		if not ( isinstance( name_node, ast.Constant ) and isinstance( name_node.value, str ) and name_node.value.isidentifier() ):
			self.lowering.discovery.fail(
				f'compiler.{fn_name}(...) field name must be a plain identifier string literal '
				f'(one level of field access, no dots): {ast.unparse(node)}',
				node,
			)
		return name_node.value

	def _require_c_type_pointer( self, ptr_type: Type|None, fn_name: str, node: ast.Call ) -> None:
		if not (
			isinstance( ptr_type, Specialization )
			and isinstance( ptr_type.base, Scalar )
			and ptr_type.base.stem in ( 'Ptr', 'ConstPtr' )
			and isinstance( ptr_type.args[0], CType )
		):
			self.lowering.discovery.fail(
				f'compiler.{fn_name}(...) first argument must be Ptr[T]/ConstPtr[T] where T is a '
				f'compiler.c_type(...): {ast.unparse(node)}',
				node,
			)

	def _lower_compiler_c_field( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.c_field(ptr, 'field_name', T) -> T - reads a field of an
		# OPAQUE compiler.c_type(...) struct through ptr (Ptr[SomeCType]/
		# ConstPtr[SomeCType]), trusting the caller's asserted type T the
		# same way compiler.cexpr's/compiler.sizeof's own type arguments are
		# trusted - nothing here verifies field_name or T against the real
		# header's actual layout; a mismatch is caught by the C compiler
		# once it sees the real struct definition, exactly like an
		# @extern(header=...) signature mismatch already is. See
		# lib/posix/pthread.py's own comment on why ucontext_t needs this:
		# CType has no known field layout at all (unlike a @cstruct/
		# @interface), so ordinary x.field access has nowhere to look the
		# field up (_attr_lookup requires chain_lookup or a .names dict,
		# neither of which CType has).
		if len( node.args ) != 3 or node.keywords:
			self.lowering.discovery.fail( f'compiler.c_field(ptr, field_name, type) takes exactly 3 positional arguments: {ast.unparse(node)}', node )
		ptr_node, name_node, type_node = node.args
		ptr = self._lower_expr( ptr_node, None )
		field_name = self._c_field_name_literal( name_node, 'c_field', node )
		self._require_c_type_pointer( ptr.type, 'c_field', node )
		field_type = self.lowering._try_resolve_namespace( type_node )
		if not isinstance( field_type, Type ):
			self.lowering.discovery.fail( f'compiler.c_field(...) third argument must be a type: {ast.unparse(node)}', node )
		dest = self._new_temp( field_type )
		self._emit( ir.GetAttr( dest = dest, obj = ptr, attr = field_name ))
		return dest

	def _lower_compiler_c_field_set( self, node: ast.Call, expected_type: Type|None ) -> None:
		# compiler.c_field_set(ptr, 'field_name', value) - writes a field of
		# an opaque compiler.c_type(...) struct through ptr. Same trust
		# model as compiler.c_field's own read side (see its comment) -
		# value's own type is whatever it already lowers to, unchecked
		# against the real field's type beyond what the C compiler itself
		# catches.
		if len( node.args ) != 3 or node.keywords:
			self.lowering.discovery.fail( f'compiler.c_field_set(ptr, field_name, value) takes exactly 3 positional arguments: {ast.unparse(node)}', node )
		ptr_node, name_node, value_node = node.args
		ptr = self._lower_expr( ptr_node, None )
		field_name = self._c_field_name_literal( name_node, 'c_field_set', node )
		self._require_c_type_pointer( ptr.type, 'c_field_set', node )
		value = self._lower_expr( value_node, None )
		self._emit( ir.SetAttr( obj = ptr, attr = field_name, value = value ))
		return None

	def _lower_compiler_c_field_addr( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.c_field_addr(ptr, 'field_name', Ptr[T]) -> Ptr[T] -
		# address of a NESTED value-typed field of an opaque c_type struct
		# (e.g. ucontext_t's uc_stack, itself a stack_t VALUE, not a
		# pointer) - lets a second-level opaque type (stack_t, its own
		# compiler.c_type(...)) be reached by composing two single-level
		# accesses (this, then compiler.c_field/c_field_set on the result)
		# rather than needing a dotted field path - see
		# _c_field_name_literal's own comment for why dotted paths aren't
		# supported directly. Third argument is the FULL result type
		# (Ptr[T], not just T) - unlike compiler.addrof, which computes the
		# pointer level itself from a known operand type, everything here is
		# already caller-asserted, so there's no "known type" to compute a
		# level from.
		if len( node.args ) != 3 or node.keywords:
			self.lowering.discovery.fail( f'compiler.c_field_addr(ptr, field_name, type) takes exactly 3 positional arguments: {ast.unparse(node)}', node )
		ptr_node, name_node, type_node = node.args
		ptr = self._lower_expr( ptr_node, None )
		field_name = self._c_field_name_literal( name_node, 'c_field_addr', node )
		self._require_c_type_pointer( ptr.type, 'c_field_addr', node )
		result_type = self.lowering._try_resolve_namespace( type_node )
		if not isinstance( result_type, Type ):
			self.lowering.discovery.fail( f'compiler.c_field_addr(...) third argument must be a type: {ast.unparse(node)}', node )
		dest = self._new_temp( result_type )
		self._emit( ir.AddrOfField( dest = dest, obj = ptr, attr = field_name ))
		return dest

	def _lower_compiler_atomic_load( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.atomic_load(...) takes exactly one argument: {ast.unparse(node)}', node )
		ptr = self._lower_expr( node.args[0], None )
		pointee = self.lowering._atomic_pointee_type( ptr.type, node )
		dest = self._new_temp( expected_type or pointee )
		self._emit( ir.AtomicLoad( dest = dest, ptr = ptr ))
		return dest

	def _lower_compiler_format_f64( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.format_f64(buf, size, precision, type_char, alt, value) ->
		# i32 - writes value's fixed-precision decimal digits (magnitude
		# only, no sign - lib/builtins/__float.py's own callers split the
		# sign out first, the same split int's __str__/_to_radix_digits/
		# _decimal_digits_with_grouping already keep) into buf[0:size),
		# returns the byte count written. type_char is a printf-style
		# conversion character's ASCII code ('f'/'F'/'e'/'E'/'g'/'G' - see
		# fstring_format_spec.FORMAT_SPEC_TYPE_CHARS; '%' is handled entirely
		# in metalpy source instead, by scaling the value and formatting as
		# 'f' - see lib/builtins/__float.py's _percent_digits). alt is the
		# '#' flag (always show the decimal point for 'f'/'F'/'e'/'E', keep
		# trailing zeros for 'g'/'G' - real snprintf's own '#' flag already
		# matches Python's semantics for every one of these exactly, so it's
		# passed straight through rather than needing its own post-
		# processing the way grouping does). Backed by a hand-written C
		# helper in emitter_c.py's PROLOGUE (real snprintf/msvcrt _snprintf,
		# called there with its true variadic prototype) - deliberately NOT
		# an ordinary @extern binding: emitter_c.py's extern codegen only
		# ever emits fixed-arity C prototypes, which is an ABI hazard for a
		# genuinely variadic callee, and tagging this under the 'c' extern
		# lib would flip compiler.extern_libs and break the no-crt Windows
		# build (float_test.py's own no_crt = 'c' not in compiler.
		# extern_libs).
		if len( node.args ) != 6 or node.keywords:
			self.lowering.discovery.fail( f'compiler.format_f64(...) takes exactly 6 arguments (buf, size, precision, type_char, alt, value): {ast.unparse(node)}', node )
		intrinsics = self.lowering.discovery.get_intrinsics()
		ptr_cls = intrinsics['Ptr']
		buf_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ intrinsics['u8'] ] )
		buf = self._lower_expr( node.args[0], buf_type )
		size = self._lower_expr( node.args[1], intrinsics['usize'] )
		precision = self._lower_expr( node.args[2], intrinsics['i32'] )
		type_char = self._lower_expr( node.args[3], intrinsics['i32'] )
		alt = self._lower_expr( node.args[4], intrinsics['bool'] )
		value = self._lower_expr( node.args[5], intrinsics['f64'] )
		dest = self._new_temp( expected_type or intrinsics['i32'] )
		self._emit( ir.FormatFloat( dest = dest, buf = buf, size = size, precision = precision, type_char = type_char, alt = alt, value = value ))
		return dest

	def _lower_compiler_is_nan_or_inf( self, node: ast.Call, expected_type: Type|None, name: str ) -> ir.Operand:
		# compiler.is_nan(x)/compiler.is_inf(x) - x: f32|f64 -> bool. Reuses
		# __metalpy_isnan/__metalpy_isinf (emitter_c.py's PROLOGUE, already
		# there for checked/panic-mode float arithmetic) - exposed directly
		# so f-string format specs can special-case inf/nan display
		# (lib/builtins/__float.py), since real snprintf/msvcrt don't
		# reliably produce "inf"/"nan" text for these themselves (confirmed:
		# legacy msvcrt's own _snprintf gives outright garbage like "1.$"
		# for +infinity, not "inf" - unlike the exponent-padding/missing-'F'
		# quirks found earlier, this one isn't even close to right).
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.{name}(...) takes exactly one argument: {ast.unparse(node)}', node )
		intrinsics = self.lowering.discovery.get_intrinsics()
		value = self._lower_expr( node.args[0], None )
		if value.type is not intrinsics.get( 'f32' ) and value.type is not intrinsics.get( 'f64' ):
			type_name = value.type.qualname if value.type is not None else '?'
			self.lowering.discovery.fail( f'compiler.{name}(...) argument must be f32 or f64, not {type_name}: {ast.unparse(node)}', node )
		dest = self._new_temp( expected_type or intrinsics['bool'] )
		ir_cls = ir.IsNan if name == 'is_nan' else ir.IsInf
		self._emit( ir_cls( dest = dest, value = value ))
		return dest

	def _lower_compiler_parse_f64( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.parse_f64(buf) - buf: ConstPtr[u8] (null-terminated C
		# text) -> f64. The inverse of compiler.format_f64 - needed for the
		# shortest-round-trip repr search (lib/builtins/__float.py's
		# _f64_repr_digits_raw: try increasing precision, re-parse each
		# candidate, stop at the first exact round-trip). Backed by a
		# hand-written C helper in emitter_c.py's PROLOGUE (real strtod on
		# POSIX; msvcrt.dll's own strtod, resolved dynamically via
		# GetModuleHandleA/LoadLibraryA/GetProcAddress, on Windows -
		# verified correct against this system's own msvcrt.dll, unlike
		# some of its other legacy quirks found earlier) - deliberately
		# NOT an ordinary @extern binding even though strtod's own
		# signature is perfectly ordinary (non-variadic, no ABI hazard
		# like compiler.format_f64 has): tagging it under the 'c' extern
		# lib would still wrongly flip compiler.extern_libs and break the
		# no-crt Windows build, the same reason compiler.format_f64 itself
		# isn't a plain @extern binding either.
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.parse_f64(...) takes exactly one argument: {ast.unparse(node)}', node )
		intrinsics = self.lowering.discovery.get_intrinsics()
		ptr_cls = intrinsics['ConstPtr']
		buf_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ intrinsics['u8'] ] )
		buf = self._lower_expr( node.args[0], buf_type )
		dest = self._new_temp( expected_type or intrinsics['f64'] )
		self._emit( ir.ParseFloat( dest = dest, buf = buf ))
		return dest

	# --- debug-mode alloc-site tracking (dump_live_objects) - see
	# i-want-to-investigate-kind-garden.md. lib/sys.py's alloc[T]/free call
	# these directly, always inside a `if compiler.target.debug:` guard (same
	# dead-branch-elimination pattern mempoison already uses - see
	# compile_time_transformer), so a release build's lowering never reaches
	# any of these at all; emitter_c.py's own codegen for the matching ir
	# instructions can assume _target_debug is always True there. ---

	def _lower_compiler_debug_raw_track( self, node: ast.Call ) -> None:
		# compiler.__debug_raw_track__(ptr, size) -> None - statement-only
		# (mirrors compiler.incref/decref/atomic_store); see ir.DebugRawTrack.
		# ptr's own declared type is whatever the caller already has (Ptr[u8]
		# from sys.alloc[T]'s own _alloc call) - no fresh Ptr[u8] type object
		# needed here, this never returns a value of its own.
		if len( node.args ) != 2 or node.keywords:
			self.lowering.discovery.fail( f'compiler.__debug_raw_track__(...) takes exactly two arguments (ptr, size): {ast.unparse(node)}', node )
		intrinsics = self.lowering.discovery.get_intrinsics()
		ptr = self._lower_expr( node.args[0], None )
		size = self._lower_expr( node.args[1], intrinsics['usize'] )
		self._emit( ir.DebugRawTrack( ptr = ptr, size = size ))

	def _lower_compiler_debug_raw_untrack( self, node: ast.Call ) -> None:
		# compiler.__debug_raw_untrack__(ptr) -> None - statement-only,
		# inverse of __debug_raw_track__; see ir.DebugRawUntrack
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.__debug_raw_untrack__(...) takes exactly one argument (ptr): {ast.unparse(node)}', node )
		ptr = self._lower_expr( node.args[0], None )
		self._emit( ir.DebugRawUntrack( ptr = ptr ))

	def _lower_compiler_debug_track_immortal_cache( self, node: ast.Call ) -> None:
		# compiler.__debug_track_immortal_cache__(slot) -> None -
		# statement-only (mirrors compiler.__debug_raw_track__); see
		# ir.DebugTrackImmortalCache. slot's own declared type is whatever
		# the caller already has (Ptr[Ptr[u8]] from lib/sys.py's
		# debug_register_immortal_cache) - no fresh type object needed
		# here, this never returns a value of its own.
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.__debug_track_immortal_cache__(...) takes exactly one argument (slot): {ast.unparse(node)}', node )
		slot = self._lower_expr( node.args[0], None )
		self._emit( ir.DebugTrackImmortalCache( slot = slot ))

	def _lower_compiler_debug_quarantine( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.__debug_quarantine__(ptr) -> same Ptr[T] as ptr - see
		# ir.DebugQuarantine. Returns a value (unlike track/untrack above):
		# the pit's own eviction result, which sys.free()'s debug branch
		# needs to decide what (if anything) to actually free this call.
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.__debug_quarantine__(...) takes exactly one argument (ptr): {ast.unparse(node)}', node )
		ptr = self._lower_expr( node.args[0], None )
		dest = self._new_temp( ptr.type )
		self._emit( ir.DebugQuarantine( dest = dest, ptr = ptr ))
		return dest

	def _lower_compiler_dump_live_objects( self, node: ast.Call ) -> None:
		# compiler.dump_live_objects() - statement-only (mirrors compiler.
		# incref/decref/atomic_store), see ir.DumpLiveObjects. A real compile
		# error outside a debug build (matching the plan's own "compile error
		# if called from a release build" note) - lib/sys.py's own
		# dump_live_objects() is the intended entry point and always guards
		# this behind `if compiler.target.debug:` itself, so reaching here in
		# a release build means user code called the compiler.* intrinsic
		# directly, bypassing that guard.
		if node.args or node.keywords:
			self.lowering.discovery.fail( f'compiler.dump_live_objects() takes no arguments: {ast.unparse(node)}', node )
		if not self.lowering.discovery.active_target['debug']:
			self.lowering.discovery.fail(
				'compiler.dump_live_objects() is only available in a debug build (compiler.target.debug) - '
				'no allocation-site tracking exists in a release build to dump', node,
			)
		self._emit( ir.DumpLiveObjects() )

	def _lower_compiler_atomic_store( self, node: ast.Call ) -> None:
		# statement-only (see _stmt_Expr's own dispatch) - mirrors
		# compiler.incref/decref: no return value, nothing to hand back to
		# an expression context
		if len( node.args ) != 2 or node.keywords:
			self.lowering.discovery.fail( f'compiler.atomic_store(...) takes exactly two arguments: {ast.unparse(node)}', node )
		ptr = self._lower_expr( node.args[0], None )
		pointee = self.lowering._atomic_pointee_type( ptr.type, node )
		value = self._lower_expr( node.args[1], pointee )
		self._emit( ir.AtomicStore( ptr = ptr, value = value ))

	def _lower_compiler_atomic_rmw( self, node: ast.Call, expected_type: Type|None, op: ir.AtomicRMWOp, *, want_result: bool = True ) -> ir.Operand:
		# shared by atomic_add/atomic_sub/atomic_exchange - same shape
		# (ptr, val), dest gets the value from BEFORE the op (C11
		# atomic_fetch_add/sub/exchange's own convention). C11's atomic_fetch_*
		# always returns a value even when called as a bare statement (the
		# common case - most callers only want the side effect), so a
		# want_result=False caller (_stmt_Expr's own bare-statement path)
		# still needs the same dest temp for the IR shape, but marks it
		# unused right after so -Wunused-but-set-variable/C4189 doesn't fire
		# on a temp the source never asked to read
		if len( node.args ) != 2 or node.keywords:
			self.lowering.discovery.fail( f'compiler.atomic_{op.value}(...) takes exactly two arguments: {ast.unparse(node)}', node )
		ptr = self._lower_expr( node.args[0], None )
		pointee = self.lowering._atomic_pointee_type( ptr.type, node )
		value = self._lower_expr( node.args[1], pointee )
		dest = self._new_temp( expected_type or pointee )
		self._emit( ir.AtomicRMW( dest = dest, op = op, ptr = ptr, value = value ))
		if not want_result:
			self._emit( ir.MarkUnused( value = dest ))
		return dest

	def _lower_compiler_atomic_compare_exchange( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# C11 strong CAS: ptr, expected: Ptr[T], desired -> bool. On
		# failure *expected is written with the actual current value - that
		# side effect happens through `expected` itself (an ordinary Ptr[T]
		# the caller already owns), nothing more to hand back for it here
		if len( node.args ) != 3 or node.keywords:
			self.lowering.discovery.fail(
				f'compiler.atomic_compare_exchange(...) takes exactly three arguments (ptr, expected, desired): {ast.unparse(node)}',
				node,
			)
		ptr = self._lower_expr( node.args[0], None )
		pointee = self.lowering._atomic_pointee_type( ptr.type, node )
		expected = self._lower_expr( node.args[1], None )
		expected_pointee = self.lowering._atomic_pointee_type( expected.type, node )
		if expected_pointee is not pointee:
			self.lowering.discovery.fail(
				f'compiler.atomic_compare_exchange(...): ptr and expected must point to the same type: {ast.unparse(node)}',
				node,
			)
		desired = self._lower_expr( node.args[2], pointee )
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		dest = self._new_temp( expected_type or bool_cls )
		self._emit( ir.AtomicCompareExchange( dest = dest, ptr = ptr, expected = expected, desired = desired ))
		return dest

	def _lower_scalar_cast( self, target_type: Scalar, source: ast.expr|ir.Operand, node: ast.AST ) -> ir.Operand:
		# shared by compiler.cast(T, x) and T(x) construction-sugar - the
		# one place the actual Scalar-to-Scalar conversion logic lives.
		# `source` is EITHER an unlowered ast.expr (a bare literal) OR an
		# already-lowered ir.Operand (a real runtime value). For a pure
		# int<->int conversion (float involved is handled separately below,
		# unchanged), the rule is purely a bit-width question, identical for
		# a literal and a runtime value alike (see the width comparison
		# below and its own comment): same-width or widening ALWAYS
		# succeeds, unconditionally, in every arithmetic mode (a pure bit-
		# reinterpretation - -1 reinterpreted as u32 is exactly the well-
		# defined two's-complement value real WinAPI constants like
		# STD_OUTPUT_HANDLE rely on); only a genuinely NARROWING conversion
		# can ever fail, and that stays mode-aware (respects self.
		# _arithmetic_mode exactly like +/-/* already do, reusing the same
		# Check/Wrap/Saturate/panic_arithmetic machinery) - see SYNTAX.md's
		# own T(x) section for the full rationale, including why this is
		# deliberately a DIFFERENT, narrower rule than x.to_T() (a value-
		# range check, independent of width - see compiler.checked_convert).
		if isinstance( source, ast.expr ):
			# an EXPLICIT float-literal cast to a non-float scalar (i32(1.5),
			# compiler.cast(u8, 3.9)) truncates toward zero at compile time -
			# this is the deliberate float->int conversion mechanism, so it
			# bypasses _expr_Constant's implicit-hint guard (which only rejects a
			# float literal being SILENTLY coerced to an int, e.g. `i + 1.5`).
			# A float->float or int/other literal keeps its existing behavior
			# (float value preserved, or int bit-reinterpretation via _lower_expr)
			if isinstance( source, ast.Constant ) and isinstance( source.value, float ) and not _is_float_scalar( target_type ):
				return ir.Const( type = target_type, value = int( source.value ) )
			# a bare int literal's own natural type is i32 (_expr_Constant's
			# own "a literal's OWN Python type always determines its natural
			# type" rule, used whenever no expected_type applies) - applying
			# the SAME same-width/widening-vs-narrowing rule the runtime
			# branch below uses, relative to THAT natural i32 width, is what
			# makes a literal and an equivalent runtime i32 argument behave
			# IDENTICALLY through T(x) - closing the exact inconsistency
			# this whole mechanism exists for (previously: u32(-1) always
			# succeeded via a blanket exemption, but u32(some_i32_var
			# holding -1) went through the real, then still mode-aware,
			# checked-cast path - now both are simply never fallible in the
			# first place, since i32->u32 is same-width). Only a genuinely
			# NARROWING literal (target narrower than i32) still needs
			# range-checking - e.g. u8(-10000) is still a hard, mode-
			# independent compile-time error, matching how `x: u8 = -10000`
			# already behaves everywhere else literals flow into a
			# concrete integer type.
			allow_bit_reinterpret = True
			if isinstance( source, ast.Constant ) and type( source.value ) is int and not _is_float_scalar( target_type ):
				i32 = self.lowering.discovery.get_intrinsics()['i32']
				allow_bit_reinterpret = target_type.sizeof >= i32.sizeof
			prev_allow_bit_reinterpret = self._allow_literal_bit_reinterpret
			self._allow_literal_bit_reinterpret = allow_bit_reinterpret
			try:
				return self._lower_expr( source, target_type )
			finally:
				self._allow_literal_bit_reinterpret = prev_allow_bit_reinterpret
		operand = source
		# a cast that touches a float on either side (int<->float, float<->float)
		# takes the GetFloatCast path: floats have no integer-overflow concept,
		# so checked/panic mode checks the result/source for inf/nan/range and
		# wrap/saturate does a plain-or-clamping conversion - never the unsigned-
		# roundtrip integer cast machinery. A pure int<->int cast is unchanged
		target_is_float = _is_float_scalar( target_type )
		source_is_float = _is_float_scalar( operand.type )
		if target_is_float or source_is_float:
			opcode, extra = self._arithmetic_mode[-1].GetFloatCast( target_is_float = target_is_float, source_is_float = source_is_float )
			return self._lower_arithmetic_op( node, opcode, extra, target_type, { 'operand': operand }, 'cast' )
		if target_type.stem == 'bool':
			# bool(x) for ANY integer x is unconditionally safe, regardless
			# of width - unlike a genuine narrowing int<->int cast (u8(-1)
			# can't represent -1, a real overflow), there is no source value
			# a bool conversion can't represent: C itself defines converting
			# any scalar to _Bool as "0 stays false, any nonzero becomes
			# true" (not a raw truncating bit-cast - confirmed correct via a
			# real repro, bool(4) under wrap_arithmetic mode DID already
			# give True, not a wrongly-truncated False, since the emitted C
			# cast itself carries the right semantics regardless of how
			# THIS compiler's own width bookkeeping treated it). Emitting
			# through the same unconditional CastWrap path a widening cast
			# already uses (rather than the mode-gated GetCast() narrowing
			# path below, previously reached because bool's sizeof is
			# smaller than most sources) means bool(x)/an auto-truthiness
			# conversion never needs an enclosing wrap_arithmetic/
			# panic_arithmetic/Result[_,OverflowError] wrapper - matching
			# Python's own bool(x), which never raises for a plain scalar.
			return self._lower_arithmetic_op( node, ir.CastWrap, None, target_type, { 'operand': operand }, 'cast' )
		if target_type.sizeof >= operand.type.sizeof:
			# same-width or widening int<->int - always succeeds, in every
			# mode, no Result involved: matches CastWrap's own bare-C-cast
			# emitter code (a plain `(ctype)(operand)`), just no longer
			# gated behind wrap-mode specifically. A real, deliberate
			# behavior change from this conversion's own prior semantics
			# (confirmed via a real repro: u32(some_i32_holding_a_negative_
			# value) previously REQUIRED an enclosing Result under checked/
			# panic mode, only wrap mode was unconditional) - see this
			# function's own top-of-method comment and SYNTAX.md.
			return self._lower_arithmetic_op( node, ir.CastWrap, None, target_type, { 'operand': operand }, 'cast' )
		opcode, extra = self._arithmetic_mode[-1].GetCast()
		return self._lower_arithmetic_op( node, opcode, extra, target_type, { 'operand': operand }, 'cast' )

	def _lower_compiler_raw_alloc( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.__raw_alloc__(T) - allocate a fresh RCClass instance with
		# every field left UNINITIALIZED: no field validation, no __init__
		# call. This is the exact alloc _try_lower_construct_call used to
		# emit inline (dest = new temp, schedule sys.alloc[T], ir.Allocate
		# with an empty fields dict) - factored out here so a synthesized
		# $$__new__ body (type_resolver.py's
		# _synthesize_rcclass_constructor) can spell "allocate self, THEN
		# call __init__ myself" as ordinary AST rather than raw IR. Not
		# meant for ordinary user code (there's no field-completeness check
		# at all - the caller is on the hook for calling __init__ or
		# compiler.__raw_free__'ing it before it ever escapes), same
		# internal-only posture as compiler.decref_dynamic.
		#
		# node.args[0].resolved_type, like compiler.cast's first argument,
		# lets compiler-synthesized AST hand over a concrete RCClass object
		# directly (including a monomorphized generic with no user-
		# spellable name), bypassing ordinary namespace resolution.
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.__raw_alloc__(...) takes exactly one argument: {ast.unparse(node)}', node )
		target_type = getattr( node.args[0], 'resolved_type', None )
		if target_type is None:
			target_type = self.lowering._try_resolve_namespace( node.args[0] )
		if not isinstance( target_type, RCClass ):
			self.lowering.discovery.fail( f'compiler.__raw_alloc__(...) argument must be a concrete RCClass: {ast.unparse(node)}', node )
		dest = self._new_temp( target_type )
		self.lowering._schedule_rcclass_construction( target_type, dest.type )
		self._emit( ir.Allocate( dest = dest, cls = target_type, fields = {} ))
		return dest

	def _lower_compiler_cast( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.cast(T, x) - T is a TYPE reference (resolved via
		# _try_resolve_namespace, same as compiler.sizeof's argument, not
		# _lower_expr), x is a real value. The call's own target type is
		# always authoritative for the result - unlike an ordinary literal,
		# an explicit cast overrides whatever the ambient expected_type is.
		# node.args[0].resolved_type, if set, bypasses namespace resolution
		# entirely - mirrors resolved_callee's own established escape hatch
		# (type_resolver.py's _synthesize_rcclass_destructor), for
		# compiler-synthesized AST that already knows its target Type
		# object directly and has no natural resolvable-by-name spelling
		# for it (a closure trampoline's own receiver cast, see
		# _get_or_create_closure_trampoline)
		if len( node.args ) != 2 or node.keywords:
			self.lowering.discovery.fail( f'compiler.cast(...) takes exactly two arguments: {ast.unparse(node)}', node )
		target_type = getattr( node.args[0], 'resolved_type', None )
		if target_type is None:
			target_type = self.lowering._try_resolve_namespace( node.args[0] )
		if target_type is None:
			self.lowering.discovery.fail( f'compiler.cast(...) first argument must be a type: {ast.unparse(node)}', node )
		if isinstance( target_type, TypeVar ):
			self.lowering.discovery.fail(
				f'compiler.cast({target_type.stem}, ...) requires a concrete type - {target_type.qualname} is still an '
				f'unbound generic type parameter here (call the enclosing function through an explicit specialization, e.g. foo[SomeType](...))',
				node,
			)
		if self.lowering._type_resolver._is_pointer_representable( target_type ):
			# Ptr[T]/ConstPtr[T], OR a bare RCClass - both are a single
			# machine pointer's worth of bits, just typed differently (see
			# _is_pointer_representable) - a plain reinterpret cast between
			# any of these (RawList's own byte-buffer indexing scheme needs
			# this: a Ptr[None] slot pointer reinterpreted as Ptr[T], OR
			# reinterpreted directly as a bare RC element T itself, since a
			# T-typed SLOT holds exactly T's own handle, not a Ptr[T] to
			# one - see list[T]'s own read/write helpers). Never fails at
			# runtime (unlike a scalar cast, which can lose bits) - no
			# arithmetic-mode concept applies, so this bypasses
			# _lower_scalar_cast/_lower_arithmetic_op entirely and reuses
			# ir.CastWrap directly, purely for its emitter_c.py shape
			# (`dest = (ctype)(operand);`, no overflow check) - not because
			# this is "wrap mode" in the arithmetic sense
			value = self._lower_expr( node.args[1], None )
			if not self.lowering._type_resolver._is_pointer_representable( value.type ):
				self.lowering.discovery.fail(
					f'compiler.cast({target_type.qualname}, ...) second argument must be a pointer or RC value, not '
					f'{value.type.qualname if value.type else "?"}: {ast.unparse(node)}',
					node,
				)
			dest = self._new_temp( expected_type or target_type )
			self._emit( ir.CastWrap( dest = dest, operand = value ))
			return dest
		if not isinstance( target_type, Scalar ):
			self.lowering.discovery.fail( f'compiler.cast({target_type.qualname}, ...) is not supported yet - only Scalar-to-Scalar and pointer-to-pointer casts are, for now', node )
		value_node = node.args[1]
		if isinstance( value_node, ast.Constant ):
			return self._lower_scalar_cast( target_type, value_node, node )
		value = self._lower_expr( value_node, None )
		if isinstance( value.type, CEnum ) and value.type.value_type is target_type:
			# compiler.cast(u8, some_foo) where Foo is a u8-backed @enum -
			# same bare reinterpret as the T(x) scalar-constructor route
			# (_try_lower_scalar_construct_call) allows for the identical
			# shape - see its own comment for why (a CEnum's runtime
			# representation IS its value_type exactly).
			dest = self._new_temp( expected_type or target_type )
			self._emit( ir.CastWrap( dest = dest, operand = value ))
			return dest
		if not isinstance( value.type, Scalar ):
			self.lowering.discovery.fail(
				f'compiler.cast(...) second argument must be a scalar value, not {value.type.qualname if value.type else "?"}: {ast.unparse(node)}',
				node,
			)
		return self._lower_scalar_cast( target_type, value, node )

	def _lower_compiler_early_return( self, node: ast.Call ) -> None:
		# compiler.early_return(err) - a same-function early bailout: usable
		# anywhere inside a function that itself returns Result[_,_], to
		# return Result.Err(err) immediately without writing the boilerplate
		# out by hand. Desugars to `return Result.Err(err)` and delegates to
		# _stmt_Return so it reuses the epilogue-vs-plain-Return split (and
		# errdefer's is_err() epilogue check, which already treats any
		# Result.Err landing in the return slot uniformly, not just the
		# OrJump path) rather than duplicating either.
		#
		# NOTE: Result.or_return()'s own written body uses this same call
		# (`compiler.early_return(self.data.v_Err)`), but that body is
		# never actually lowered as a real function - it's a spec, not
		# compilable code, because it would need this to trigger a return
		# in ITS CALLER's scope, not or_return()'s own (or_return's declared
		# return type is bare T, not Result[T,E] - `return Result.Err(...)`
		# from inside it could never type-check there). or_return() calls
		# are instead recognized and expanded directly at the call site -
		# see _lower_or_return.
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.early_return(...) takes exactly one argument: {ast.unparse(node)}', node )
		fn = self._current_fn
		return_type = fn.return_type if fn is not None else None
		ok = fn is not None and self.lowering._type_resolver._result_shape( return_type ) is not None
		if not ok:
			where = f'{fn.qualname} returns {return_type.qualname if return_type else None}' if fn is not None else 'this is not inside a function'
			self.lowering.discovery.fail( f'compiler.early_return(...) requires the enclosing function to return Result[_,_] ({where})', node )

		err_call = ast.Call(
			func = ast.Attribute( value = ast.Name( id = 'Result', ctx = ast.Load() ), attr = 'Err', ctx = ast.Load() ),
			args = [ node.args[0] ],
			keywords = [],
		)
		ast.copy_location( err_call, node )
		ast.fix_missing_locations( err_call )
		return_stmt = ast.Return( value = err_call )
		ast.copy_location( return_stmt, node )
		self._stmt_Return( return_stmt )

	def _in_generic_class_method( self ) -> bool:
		# true while lowering a MONOMORPHIZED method of a generic class
		# (list[i32].__del__, ...) - Monomorphizer.monomorphized_function
		# sets a substituted method's own .cls to the concrete class
		# Specialization it was built for (base.cls stays plain/None for
		# an ordinary, non-generic method) - see its own comment on why.
		# Used by compiler.incref/decref to tell "T turned out non-RC this
		# instantiation" (routine, no-op) apart from "this was never RC to
		# begin with" (a real mistake, still rejected) - see their own
		# docstrings
		cls = getattr( self._current_fn, 'cls', None )
		return isinstance( cls, Specialization )

	def _internal_decref_var( self, var: Variable ) -> None:
		''' compiler-internal-only equivalent of compiler.__internal_decref__(x)
		(see its own docstring) for a caller that already has the real
		Variable object in hand, bypassing AST synthesis + re-resolution
		through _lower_expr entirely - required, not just a shortcut, for a
		NARROWED union local (e.g. or_throw(mapper)'s recv_var, narrowed via
		cfg.narrow() to its Err member just before): _lower_expr on a
		synthesized ast.Name reference to it returns a freshly-EXTRACTED
		ir.Temp (the narrowed payload), not the union Variable itself -
		manually_decreffed() then silently takes its ir.Temp branch
		(_temp_states, unrelated to self.bindings/_epilogue_stack) instead of
        ever touching the Variable's own real binding, leaving THAT
		untouched for whatever later unwind (e.g. a covered raise dispatch)
		to redundantly release again. Confirmed via a real repro + trace:
		compiler.__internal_decref__(recv_var) reached this exact class's
		manually_decreffed(), but for the WRONG (Temp) operand - the
		Variable-typed binding it needed to cancel was never touched. '''
		for instr in self._cfg.decref( var.type, var ):
			self._emit( instr )
		for instr in self._cfg.manually_decreffed( var ):
			self._emit( instr )

	def _lower_compiler_internal_decref( self, node: ast.Call ) -> None:
		# compiler.__internal_decref__(x) - NOT reachable from user source
		# (no such name is exposed to user code; only THIS file's own AST
		# synthesis ever spells it) - the compiler-internal-only counterpart
		# of compiler.decref(x), for lowering code that needs to release a
		# hidden local EARLY (before its own natural scope-exit epilogue
		# would fire) and must stop that later epilogue from releasing it a
		# second time: the for-loop iterator (obj_var, see _stmt_For),
		# with-statement context managers (_lower_with_context_manager),
		# or_throw(mapper)/or_return(mapper)'s receiver/extracted-error/
		# mapper locals. Identical body to what compiler.decref(x) itself
		# used to do before cfg.manually_decreffed()'s cancellation was
		# removed from it (see _lower_compiler_decref's own comment on why:
		# "lowering/emitter must not adjust automatic incref/decref behavior
		# in the presence of a [USER-authored] manual compiler.decref()
		# call" - a rule about not letting user source code secretly cancel
		# its own epilogue, not about lowering's own bookkeeping for hidden
		# locals it created and alone is responsible for tearing down).
		# Simplified from compiler.decref(x)'s own version: no field-
		# receiver/generic-class-method carve-outs - every caller here
		# already has a real Variable of a real, already-resolved RC type in
		# hand, never a bare .attr or a not-yet-monomorphized generic T.
		assert len( node.args ) == 1 and isinstance( node.args[0], ast.Name ), f'compiler.__internal_decref__(...) is lowering-internal only, always exactly one ast.Name argument: {ast.unparse(node)}'
		operand = self._lower_expr( node.args[0], None )
		# unlike this intrinsic's older call sites (for-loop iterator,
		# with-statement context manager, or_throw/or_return(mapper)'s own
		# hidden locals - always genuinely OWNED by construction), a match
		# statement's own __match_subj_N can be a BORROWED alias of an
		# existing Name/Attribute (cfg.py's assign() gives it `borrow=True`,
		# no entry - see lowering.py's _stmt_Assign) whenever the match
		# subject is an existing variable/field rather than a fresh call
		# result. Decreffing that unconditionally is a real over-release -
		# the aliased original still owns the only reference, and its own
		# natural epilogue already tears it down. Skip for anything already
		# non-OWNED (covers this, plus a defensive no-op if already MOVED).
		binding = self._cfg.bindings.get( node.args[0].id )
		if binding is not None and binding.state != cfg.OwnState.OWNED:
			return
		for instr in self._cfg.decref( operand.type, operand ):
			self._emit( instr )
		for instr in self._cfg.manually_decreffed( operand ):
			self._emit( instr )

	def _lower_compiler_decref( self, node: ast.Call ) -> None:
		# compiler.decref(x) — emit the real Decref sequence for x, via
		# cfg.py's own union-aware decref() (NOT a bare ir.Decref emitted
		# directly here - that's only correct for a plain RC pointer; a
		# TaggedUnion operand with RC leaves needs the tag-gated release
		# ladder instead, exactly like every other decref site in this
		# compiler - see cfg.py's _refcount_instructions). Used inside
		# synthesized destructor bodies to tear down each RC field, and by
		# generic containers (list[T]) that need to conditionally RC-manage
		# elements whose T may or may not turn out to be an RC type once
		# monomorphized - a silent no-op for a non-RC T (rather than a hard
		# failure) is allowed ONLY inside a monomorphized generic-class
		# method (_in_generic_class_method), so the SAME generic method
		# body stays correct for both list[SomeRCClass] and list[i32]
		# without the class itself branching on whether T is RC - an
		# ordinary, non-generic call site with a genuinely wrong (always
		# non-RC) argument is still rejected, same as before.
		#
		# Gated on cfg.rc_leaves(operand.type), not the narrower
		# type_resolver._is_RC (is_rc_pointer) - _is_RC is False for a
		# TaggedUnion with RC members (its runtime shape is a tag+data value
		# struct, never a bare pointer), which used to make this whole
		# branch treat "T monomorphized to a union with RC leaves" exactly
		# like "T monomorphized to a genuinely non-RC scalar" - a silent
		# no-op inside _in_generic_class_method(), the SAME no-op posture
		# that's actually correct for list[i32]. Confirmed via a real repro
		# (list[T].append/__getitem__ with T a @union whose RC-carrying leaf
		# is a plain RCClass like the builtin int): the missing incref/decref
		# left every such element under-retained by exactly one reference,
		# a real heap-corruption-on-free bug - masked whenever the leaf
		# happened to be an IMMORTAL-refcount value (a string literal),
		# which is why this surfaced as "str leaves work, int leaves crash"
		# rather than an unconditional failure.
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.decref(...) takes exactly one argument: {ast.unparse(node)}', node )
		arg_node = node.args[0]
		if isinstance( arg_node, ast.Attribute ):
			# PLAN_THREAD_SAFE_SHARED_STATE.md Part B: compiler.decref(obj.
			# field) - the synthesized destructor's own field-teardown shape
			# (type_resolver.py's _build_field_teardown_ast, RCClass branch) -
			# means to transfer the FIELD's own single reference directly
			# into this decref. Lowering obj.field through the ordinary
			# _expr_Attribute path (this function's normal `operand =
			# self._lower_expr(...)` below) would go through Part B's own
			# retain-on-read (B.3) first, creating a SEPARATE, independently-
			# owned COPY - decref'ing that copy right back only cancels the
			# extra retain, leaving the field's TRUE original reference
			# permanently unreleased. Confirmed as a real leak, not a
			# theoretical one: compiler.refcount() showed a Box captured by
			# a closure environment gaining one extra reference after the
			# closure's own teardown ran, every single time. A single
			# atomic critical section here instead - acquire, a BARE
			# (unretained) read, decref, release - both fixes the leak and
			# is cheaper than the generic retain-then-immediately-undo
			# roundtrip would have been. Only short-circuits the exact
			# `compiler.decref(<attribute>)` shape - the OTHER established
			# idiom this intrinsic also serves (`val: T = <read>;
			# compiler.decref(val)`, list/dict's own element teardown,
			# manually_decreffed's own docstring) passes an ast.Name, never
			# reaches here, and is unaffected.
			field_obj = self._lower_expr( arg_node.value, None )
			field_var = self.lowering._attr_lookup( field_obj.type, arg_node.attr, arg_node )
			is_real_field = self._is_real_field_receiver( field_obj.type )
			if is_real_field:
				self._check_field_visibility( field_obj.type, field_var, arg_node.attr, arg_node )
			if is_real_field and cfg.rc_leaves( field_var.type ):
				# Cost mitigation #4: a bare read of the field's CURRENT
				# pointer value (the field's own storage is never
				# overwritten here, only its pointee's refcount) - shared is
				# sufficient; two concurrent compiler.decref(obj.field) calls
				# racing each other is a caller-level double-decref trap
				# independent of this lock's mode (both would decref the
				# identical value regardless of exclusivity), same trust
				# model compiler.decref()'s own docstring already documents.
				self._emit( ir.AcquireFieldLock( obj = field_obj, field = field_var, exclusive = False ))
				raw = self._new_temp( field_var.type )
				self._emit( ir.GetAttr( dest = raw, obj = field_obj, attr = arg_node.attr ))
				for instr in self._cfg.decref( field_var.type, raw ):
					self._emit( instr )
				self._emit( ir.ReleaseFieldLock( obj = field_obj, field = field_var, exclusive = False ))
				self._cfg.untrack_temp( raw )
				return
			# either a non-RC field (nothing to decref - falls into the
			# generic-method/fail handling below unchanged) or not a real
			# user field at all (e.g. field_obj.type is a TaggedUnion's own
			# .tag/.data, no locking concept applies there either) - a bare,
			# unretained read straight off the ALREADY-LOWERED field_obj,
			# matching this function's own pre-Part-B behavior exactly,
			# rather than re-lowering arg_node.value a second time (which
			# would double any side effects a non-trivial receiver
			# expression has - confirmed as the right call by _lower_expr's
			# own general "never re-evaluate an already-lowered operand"
			# discipline used throughout this file).
			operand = self._new_temp( field_var.type )
			self._emit( ir.GetAttr( dest = operand, obj = field_obj, attr = arg_node.attr ))
		else:
			operand = self._lower_expr( arg_node, None )
		if operand.type is not None and cfg.rc_leaves( operand.type ):
			for instr in self._cfg.decref( operand.type, operand ):
				self._emit( instr )
			# Suppress operand's own scope-exit epilogue release - without
			# this, a live OWNED local manually decref'd here (the
			# established idiom throughout this stdlib for tearing down RC
			# elements, and for tests verifying teardown by hand, e.g.
			# `h: Holder = Holder(...); ...; compiler.decref(h)`) gets
			# decref'd AGAIN once its scope ends - a real double-free (see
			# cfg.manually_decreffed()'s own docstring). manually_decreffed()
			# only recognizes a plain Variable operand (its Temp branch,
			# for a NARROWED union member _lower_expr extracted above, is a
			# no-op that leaves the union's OWN epilogue entry untouched) -
			# that's deliberate here, not a gap to paper over: reactor.py's
			# _set_current_deadline relies on exactly this narrowed-union
			# non-suppression (see its own `old: _DeadlineBox|None` comment,
			# tuned against a real double-free from trying to suppress
			# there), and reconciling "moved on one branch, owned on the
			# other" after a narrowing if/else is a merge_if() gap, not
			# something to route around here.
			for instr in self._cfg.manually_decreffed( operand ):
				self._emit( instr )
			# operand may be a fresh_temp()-registered Call/Allocate result
			# (e.g. compiler.decref(self._read_element(...)), the "safe
			# inline expression" idiom recommended above) - untrack it so
			# _flush_pending_temps doesn't ALSO decref it at end of statement
			# (a real double-free, confirmed via the debug quarantine
			# detector on list.erase_at). No-op for a Name/GetAttr operand
			# never fresh_temp()-registered in the first place.
			self._cfg.untrack_temp( operand )
			return
		if operand.type is not None and self._in_generic_class_method():
			# a genuine no-op for THIS monomorphization (see the comment
			# above), but operand may have no other reader at all in that
			# case (e.g. list[i32].__del__'s `val`, only ever passed to
			# decref) - mark it read so the no-op doesn't turn its own
			# already-emitted definition into -Wunused-variable/C4189
			self._emit( ir.MarkUsed( operand = operand ))
			return
		self.lowering.discovery.fail(
			f'compiler.decref(...) argument must be a reference-counted value, not '
			f'{operand.type.qualname if operand.type else "?"}: {ast.unparse(node)}',
			node,
		)

	def _lower_compiler_raw_free( self, node: ast.Call ) -> None:
		# compiler.__raw_free__(x) - free a raw, not-yet-fully-alive RCClass
		# allocation (compiler.__raw_alloc__'s own product, once __init__
		# has failed) WITHOUT running the class's real destructor. An
		# ordinary decref-to-zero would call the synthesized
		# $$__destructor__, which reads every field as though __init__ had
		# already populated them - on a raw, not-yet-initialized alloc
		# that's still garbage, a real heap-corruption bug (see
		# type_resolver.py's _synthesize_rcclass_constructor, fallible
		# body, and _emit_fallible_construction's own former Err branch,
		# whose logic this generalizes). Frees the backing storage directly
		# via sys.free - the same thing _synthesize_rcclass_destructor's
		# own step 3 does, skipping its __del__/field-cascade steps 1/2
		# entirely - then cancels x's pending automatic scope-exit release
		# via manually_decreffed, the same pairing compiler.decref(x) uses
		# just above, minus the real Decref that precedes it there.
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.__raw_free__(...) takes exactly one argument: {ast.unparse(node)}', node )
		operand = self._lower_expr( node.args[0], None )
		if not isinstance( operand.type, RCClass ):
			self.lowering.discovery.fail(
				f'compiler.__raw_free__(...) argument must be a bare RCClass value, not '
				f'{operand.type.qualname if operand.type else "?"}: {ast.unparse(node)}',
				node,
			)
		if self.lowering.discovery.active_target['debug']:
			# debug-mode alloc-site tracking (dump_live_objects) - this
			# object's own ir.Allocate already tracked it into the global RC
			# list; sys.free() below frees its storage directly, WITHOUT
			# going through release_object (that's the whole point - see
			# this function's own comment on why the real destructor must
			# never run here), so nothing else will ever untrack it. Must
			# happen before sys.free() runs, not after - see ir.DebugUntrackRC's
			# own comment (a real MSVC-only crash, root-caused via bisection).
			self._emit( ir.DebugUntrackRC( value = operand ))
		sys_module = self.lowering.discovery.modules['sys']
		free_overload = sys_module.get_local( 'free' )
		free_fn = free_overload.implementations[0] if isinstance( free_overload, Overload ) else free_overload
		self.lowering._ensure_resolved( free_fn )
		cast_dest = self._new_temp( free_fn.parameters[0].type )
		self._emit( ir.CastWrap( dest = cast_dest, operand = operand ))
		self._emit( ir.Call( dest = None, target = free_fn, receiver = None, args = [ cast_dest ], kwargs = {} ))
		for instr in self._cfg.manually_decreffed( operand ):
			self._emit( instr )

	def _lower_compiler_incref( self, node: ast.Call ) -> None:
		# compiler.incref(x) — emit the real Incref sequence for x, via
		# cfg.py's own union-aware incref(). Same conditional no-op-for-
		# non-RC-T posture, and the same rc_leaves(...)-vs-_is_RC fix, as
		# _lower_compiler_decref above.
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.incref(...) takes exactly one argument: {ast.unparse(node)}', node )
		operand = self._lower_expr( node.args[0], None )
		if operand.type is not None and cfg.rc_leaves( operand.type ):
			for instr in self._cfg.incref( operand.type, operand ):
				self._emit( instr )
			return
		if operand.type is not None and self._in_generic_class_method():
			# see _lower_compiler_decref's own identical comment
			self._emit( ir.MarkUsed( operand = operand ))
			return
		self.lowering.discovery.fail(
			f'compiler.incref(...) argument must be a reference-counted value, not '
			f'{operand.type.qualname if operand.type else "?"}: {ast.unparse(node)}',
			node,
		)

	def _lower_compiler_decref_dynamic( self, node: ast.Call ) -> None:
		# compiler.decref_dynamic(ptr) - releases a TYPE-ERASED Ptr[None]
		# generically, via release_object, which reads its destructor off
		# the object's own header (see emitter_c.py's ObjectHeader) instead
		# of requiring the concrete type statically, unlike compiler.decref
		# above. Internal machinery for compiler-synthesized code (a
		# closure's own __del__, releasing its captured receiver after
		# type erasure) - not meant for ordinary user code, which always
		# has a real static type and should use compiler.decref instead
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.decref_dynamic(...) takes exactly one argument: {ast.unparse(node)}', node )
		operand = self._lower_expr( node.args[0], None )
		none_type = self.lowering.discovery.get_none_type()
		ptr_cls = self.lowering.discovery.get_intrinsics()['Ptr']
		ptr_none_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ none_type ] )
		if operand.type is not ptr_none_type:
			self.lowering.discovery.fail(
				f'compiler.decref_dynamic(...) argument must be Ptr[None], not '
				f'{operand.type.qualname if operand.type else "?"}: {ast.unparse(node)}',
				node,
			)
		self._emit( ir.DecrefDynamic( value = operand ))
