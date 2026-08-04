# stdlib imports:
import dataclasses
import hashlib
import re

# local imports:
import ir
from compiler import Compiler, LoweredFunction, LoweredGlobal
from mpy_types import (
	CEnum, ClassLike, CStruct, CUnion, Copy, Function, Move,
	RCClass, Scalar, Specialization, TaggedUnion, Type, Variable,
)

# stage 3: turns a fully-lowered Compiler's output into C11 source. Pure
# translation - by the time emit_c() runs, every real dependency (including
# every concrete generic specialization, e.g. Result[i32,OverflowError] -
# see Lowering.monomorphize_class) is already discovered/scheduled/lowered
# (see ARCHITECTURE.md's stage split); this module does no further discovery
# of its own.

# verbatim from C_EMITTER.md - avoids any Windows-CRT (msvcrt) dependency
# from metalpy's own stdlib output; atomic because __del__ can run on any
# thread the moment a refcount hits 0.
PROLOGUE = '''\
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdatomic.h>

#define METALPY_IMMORTAL_REFCOUNT INT32_MAX

typedef struct {
	_Atomic int32_t ref_count;
} ObjectHeader;

static inline void retain_object( ObjectHeader* obj ) {
	if ( obj && obj->ref_count != METALPY_IMMORTAL_REFCOUNT ) {
		atomic_fetch_add( &obj->ref_count, 1 );
	}
}

static inline void release_object( ObjectHeader* obj, void (*destructor)(void*) ) {
	if ( obj && obj->ref_count != METALPY_IMMORTAL_REFCOUNT ) {
		if ( atomic_fetch_sub( &obj->ref_count, 1 ) == 1 ) {
			if ( destructor ) {
				destructor( obj );
			}
		}
	}
}
'''

# NOT part of C_EMITTER.md's own verbatim prologue above - a synthesized
# TaggedUnion payload's None member (e.g. Ptr[u8]|None) needs a real 1-byte
# type (see the plan's type-mapping table); 'void' (c_type(NoneType)'s
# spelling everywhere else) is not a legal struct/union member type
_NONE_PLACEHOLDER_TYPE = 'MetalpyNone'
_NONE_PLACEHOLDER_TYPEDEF = f'typedef unsigned char {_NONE_PLACEHOLDER_TYPE};'

# --- name mangling -----------------------------------------------------------

def mangle_qualname( qualname: str ) -> str:
	''' plain string-level mangling for an already-computed qualname (a
	Function/Variable/ordinary-class qualname, or the already-bracketed
	qualname a Specialization builds for itself in discovery.py's
	_get_or_create_specialization). NOT sufficient on its own for a
	synthesized anonymous union's qualname (builtins.int|builtins.str) -
	see mangle_type(), which is the union-aware entry point everything else
	should call for an actual Type object. '''
	return (
		qualname
		.replace( '.', '$' )
		.replace( '[', '$$g$' )
		.replace( ',', '$$' )
		.replace( ']', '' )
	)

def mangle_type( t: Type ) -> str:
	''' union-aware entry point - call this (not mangle_qualname directly)
	whenever mangling an actual Type object's own name. A synthesized
	anonymous union (TaggedUnion with file is None - a real `@union class
	Foo:` always has file/line set from source, see discovery.py's
	_get_or_create_union vs. a user class's own creation path) gets the
	doc's own worked example, reverse-engineered precisely from
	C_EMITTER.md's `builtins.str|builtins.int -> $__u$$builtins$str$$builtins$int`:
	'$' + '$$'.join(['__u', mangled-member-1, mangled-member-2, ...]).
	Applied here to members in whatever order TaggedUnion.attributes
	actually stores them (ascii-sorted for synthesized unions, per
	discovery.py's _get_or_create_union - NOT the doc's own example order,
	which is str-then-int, the opposite of ascii-sorted int-then-str; the
	doc's example is stale on ordering, the formula itself is verified
	correct against it - do not "fix" this back to match the doc's literal
	example text). Recursive on member types so a union containing another
	union/a Specialization still mangles correctly. '''
	if isinstance( t, TaggedUnion ) and t.file is None:
		parts = [ '__u' ] + [ mangle_type( attr.type ) for attr in t.attributes ]
		return '$' + '$$'.join( parts )
	if isinstance( t, CUnion ) and t.file is None:
		# Lowering._tagged_union_storage's own synthesized payload CUnion
		# for an anonymous TaggedUnion (t.qualname directly embeds the
		# outer union's raw '|'-joined qualname, e.g. 'intrinsics.NoneType|
		# intrinsics.Ptr[intrinsics.u8]$data' - plain mangle_qualname would
		# leave a literal '|' in the output, not a legal C identifier
		# character). t.attributes are the v_<member>-prefixed payload
		# fields, in the SAME order/types as the outer union's own
		# .attributes, so the identical $__u... scheme reconstructs
		# correctly straight from them - no reference to the outer
		# TaggedUnion object needed here (only its type is ever synthesized
		# with file=None; a real @union's own payload always inherits a
		# real file/line, so this can't collide with a genuine user @cunion)
		parts = [ '__u' ] + [ mangle_type( attr.type ) for attr in t.attributes ]
		return '$' + '$$'.join( parts ) + '$data'
	return mangle_qualname( t.qualname )

def _c_label( name: str ) -> str:
	# label names (lowering.py's own generated 'else'/'end'/epilogue labels)
	# are already unique within one function (C's own label scoping is
	# per-function too, so no cross-function collision risk) - just make
	# sure the result is a legal C identifier
	return 'L_' + re.sub( r'[^0-9A-Za-z_]', '_', name )

# --- scalar/type -> C type mapping --------------------------------------------

_SCALAR_C_TYPES: dict[str,str] = {
	'i8': 'int8_t', 'u8': 'uint8_t',
	'i16': 'int16_t', 'u16': 'uint16_t',
	'i32': 'int32_t', 'u32': 'uint32_t',
	'i64': 'int64_t', 'u64': 'uint64_t',
	# GCC/Clang extension, unconditional - no MSVC-compatibility check here;
	# a program using these against a cl.exe backend fails to compile there
	# on its own, same as any other backend-specific limitation (user's own
	# explicit decision - see the plan's Context section)
	'i128': '__int128', 'u128': 'unsigned __int128',
	'isize': 'intptr_t', 'usize': 'uintptr_t',
	'bool': 'bool',
}

# signed stem -> its same-width unsigned counterpart - used to compute
# Wrap-mode arithmetic via the standard defined-behavior idiom (C's signed
# overflow is UB; casting to the unsigned twin, computing there, and casting
# back is well-defined and universally two's-complement in practice)
_SIGNED_TO_UNSIGNED: dict[str,str] = {
	'i8': 'u8', 'i16': 'u16', 'i32': 'u32', 'i64': 'u64', 'i128': 'u128', 'isize': 'usize',
}

def _is_unsigned_stem( stem: str ) -> bool:
	return stem in ( 'u8', 'u16', 'u32', 'u64', 'u128', 'usize', 'bool' )

# MIN/MAX macros for Saturate-mode arithmetic - i128/u128 excluded (stdint.h
# has no INT128_MIN/MAX; there's no portable macro to reach for)
_SATURATE_LIMITS: dict[str,tuple[str,str]] = {
	'i8': ( 'INT8_MIN', 'INT8_MAX' ), 'u8': ( '0', 'UINT8_MAX' ),
	'i16': ( 'INT16_MIN', 'INT16_MAX' ), 'u16': ( '0', 'UINT16_MAX' ),
	'i32': ( 'INT32_MIN', 'INT32_MAX' ), 'u32': ( '0', 'UINT32_MAX' ),
	'i64': ( 'INT64_MIN', 'INT64_MAX' ), 'u64': ( '0', 'UINT64_MAX' ),
	'isize': ( 'INTPTR_MIN', 'INTPTR_MAX' ), 'usize': ( '0', 'UINTPTR_MAX' ),
}

def _class_keyword( base: Type ) -> str:
	# RCClass/CStruct/TaggedUnion all become a C `struct`; only CUnion
	# becomes a C `union`. CEnum is its own typedef, never struct/union-
	# prefixed (see the CEnum branch in c_type below)
	return 'union' if isinstance( base, CUnion ) else 'struct'

def c_type( t: Type|None ) -> str:
	''' maps a metalpy Type to its C spelling - RCClass is always a pointer
	(struct <mangled>*), every other class-like kind is a plain value
	(struct/union <mangled>). Move[T]/Copy[T] unwrap to their inner type
	first (ownership is a compile-time/CFG-only concept, invisible in C). '''
	if t is None:
		return 'void'
	if isinstance( t, ( Move, Copy )):
		return c_type( t.inner )
	if isinstance( t, Specialization ):
		base = t.base
		if isinstance( base, Scalar ) and base.stem in ( 'Ptr', 'ConstPtr' ):
			inner = _value_spelling( t.args[0] )
			return f'{inner}*' if base.stem == 'Ptr' else f'const {inner}*'
		if isinstance( base, RCClass ):
			return f'struct {mangle_type(t)}*'
		if isinstance( base, ( CStruct, CUnion, TaggedUnion )):
			return f'{_class_keyword(base)} {mangle_type(t)}'
		raise NotImplementedError( f'c_type: unsupported Specialization base {base!r}' )
	if isinstance( t, Scalar ):
		if t.stem == 'NoneType':
			return 'void'
		if t.stem == 'NoReturn':
			return 'void'
		if t.stem == 'Ptr': # bare, unsubscripted (rare - see lowering.py's own compiler.sizeof(Ptr) note)
			return 'void*'
		if t.stem == 'ConstPtr':
			return 'const void*'
		mapped = _SCALAR_C_TYPES.get( t.stem )
		if mapped is None:
			raise NotImplementedError( f'c_type: unsupported scalar {t.qualname!r}' )
		return mapped
	if isinstance( t, RCClass ):
		return f'struct {mangle_type(t)}*'
	if isinstance( t, ( CStruct, CUnion, TaggedUnion )):
		return f'{_class_keyword(t)} {mangle_type(t)}'
	if isinstance( t, CEnum ):
		return mangle_type( t ) # the typedef name itself, no struct/union prefix
	raise NotImplementedError( f'c_type: unsupported type {t!r}' )

def _is_noreturn( t: Type|None ) -> bool:
	return isinstance( t, Scalar ) and t.stem == 'NoReturn'

def _value_spelling( t: Type ) -> str:
	''' the C spelling of T's OWN VALUE representation - unlike c_type(),
	which auto-promotes a bare RCClass reference to a pointer (struct Foo*,
	since a variable/field of RCClass type is always a pointer everywhere
	else), this returns the bare struct body type (struct Foo) even for an
	RCClass. Needed wherever C already provides one level of indirection on
	its own and c_type()'s auto-pointering would double it up:
	Ptr[T]/ConstPtr[T]'s inner T (T* must stay a single pointer even when T
	is an RCClass - sys.alloc[Foo]'s own real return type), and sizeof(T)
	(sizeof(struct Foo), never sizeof(struct Foo*) - see ir.SizeOf's
	handling in _emit_instruction). '''
	if isinstance( t, ( Move, Copy )):
		return _value_spelling( t.inner )
	base = t.base if isinstance( t, Specialization ) else t
	if isinstance( base, ( RCClass, CStruct, CUnion, TaggedUnion )):
		return f'{_class_keyword(base)} {mangle_type(t)}'
	return c_type( t ) # scalars/CEnum - value and reference spelling are identical

def _field_type_spelling( t: Type ) -> str:
	if isinstance( t, Scalar ) and t.stem == 'NoneType':
		return _NONE_PLACEHOLDER_TYPE
	return c_type( t )

def _field_name( name: str ) -> str:
	# a REAL struct/class field (x: i32) is already a plain identifier, so
	# mangle_qualname is a no-op there - but a TaggedUnion payload's
	# synthesized field name (Lowering._tagged_union_storage's
	# f'v_{attr.stem}') can be a full type-qualname fragment for a
	# synthesized union member (Ptr[u8]|None's v_intrinsics.Ptr[intrinsics.u8],
	# not a source identifier at all - there's no source identifier to have),
	# which needs exactly the same '.'/'['/','/']' -> valid-C-identifier
	# treatment as any other name this module mangles
	return mangle_qualname( name )

# --- struct/union body emission -------------------------------------------
#
# A concrete generic class specialization (Result[i32,OverflowError]) is a
# real compile unit by the time this module ever sees it - lowering.py's
# Lowering.monomorphize_class (invoked from compiler.py's own _lower
# dispatch whenever it schedules a ClassLike-based Specialization) already
# substituted its .attributes and gave it a concrete qualname, landing it
# directly in compiler.cstructs/.cunions/.tagged_unions/.rcclasses like any
# other class. This module never has to independently rediscover or
# resynthesize one - it just walks those lists (see emit_c below).

def _struct_or_union_body( name: str, keyword: str, attrs: list[tuple[str,Type]] ) -> str:
	lines = [ f'{keyword} {name} {{' ]
	for field_name, field_type in attrs:
		lines.append( f'\t{_field_type_spelling(field_type)} {_field_name(field_name)};' )
	lines.append( '};' )
	return '\n'.join( lines )

# --- functions -----------------------------------------------------------

# discovery.py's visit_FunctionDef reserves the bare (unqualified) name
# 'main' exclusively for the program's entry point ("the name 'main' is
# special, there can be only one" - discovery.py:994-995) - every other
# function gets its usual module-qualified qualname. This is deliberate:
# metalpy's main() is meant to become C's own real `int main(void)` entry
# point directly, no trampoline needed - so it's the one function whose
# declared metalpy return type (-> None) does NOT dictate its C signature.
_ENTRY_POINT_QUALNAME = 'main'

def _is_entry_point( function: Function ) -> bool:
	return function.qualname == _ENTRY_POINT_QUALNAME

def _self_qualname( function: Function ) -> str:
	# mirrors lowering.py's own lowering-only self synthesis EXACTLY
	# (Parameter(stem='self', qualname=f'{fn.qualname}.self', type=fn.cls)
	# - see lowering.py's lower_function) so every body-site GetAttr/SetAttr/
	# Call.receiver operand referencing that same Variable object mangles to
	# the identical C identifier this prototype declares, with no separate
	# self-tracking needed anywhere else in this module
	return f'{function.qualname}.self'

def _has_self( function: Function ) -> bool:
	return function.cls is not None and not function.is_static and not function.is_classmethod

def _function_prototype( function: Function ) -> str:
	params: list[str] = []
	if _has_self( function ):
		params.append( f'{c_type(function.cls)} {mangle_qualname(_self_qualname(function))}' )
	for p in ( function.parameters or [] ):
		params.append( f'{c_type(p.type)} {mangle_qualname(p.qualname)}' )
	params_str = ', '.join( params ) if params else 'void'
	if _is_entry_point( function ):
		return f'int main( {params_str} )'
	noreturn = '_Noreturn ' if _is_noreturn( function.return_type ) else ''
	ret = c_type( function.return_type )
	name = mangle_qualname( function.qualname )
	return f'{noreturn}{ret} {name}( {params_str} )'

def _emit_operand( op: ir.Operand ) -> str:
	if isinstance( op, ir.Const ):
		return _emit_const( op )
	if isinstance( op, ir.Temp ):
		return f't{op.id}'
	if isinstance( op, Variable ):
		return mangle_qualname( op.qualname )
	raise NotImplementedError( f'_emit_operand: unsupported operand {op!r}' )

def _emit_const( c: ir.Const ) -> str:
	if isinstance( c.value, bool ):
		return 'true' if c.value else 'false'
	if isinstance( c.value, int ):
		return str( c.value )
	if c.value is None:
		return '' # NoneType constant - only ever a placeholder operand (e.g. `is None` comparisons), never emitted as a standalone value
	if isinstance( c.value, ( str, bytes )):
		# a str/bytes literal is RCClass-typed (_expr_Constant lowers it
		# directly to ir.Const(type=<builtins.str-or-bytes RCClass>,
		# value=...) - see the plan's grounding facts) - str/bytes are
		# always pointers (like any RCClass), so the operand text here is
		# the ADDRESS of a static, immortal object _emit_string_literals
		# bakes elsewhere in the translation unit, not an inline value
		if not ( isinstance( c.type, RCClass ) and c.type.qualname in _STRING_LITERAL_RCCLASS_QUALNAMES ):
			raise NotImplementedError( f'_emit_const: {c.value!r} needs an RCClass type from {sorted(_STRING_LITERAL_RCCLASS_QUALNAMES)}, got {c.type!r}' )
		return f'&{_string_literal_name(c.type.qualname, c.value)}'
	raise NotImplementedError( f'_emit_const: unsupported constant {c!r}' )

# --- arithmetic -----------------------------------------------------------

_ARITH_BINOP_INFO: dict[type,tuple[str,str]] = {
	ir.AddWrap: ( 'add', 'wrap' ), ir.AddCheck: ( 'add', 'check' ), ir.AddSaturate: ( 'add', 'saturate' ),
	ir.SubWrap: ( 'sub', 'wrap' ), ir.SubCheck: ( 'sub', 'check' ), ir.SubSaturate: ( 'sub', 'saturate' ),
	ir.MulWrap: ( 'mul', 'wrap' ), ir.MulCheck: ( 'mul', 'check' ), ir.MulSaturate: ( 'mul', 'saturate' ),
}
_ARITH_SYMBOL = { 'add': '+', 'sub': '-', 'mul': '*' }
_ARITH_BUILTIN = {
	'add': '__builtin_add_overflow', 'sub': '__builtin_sub_overflow', 'mul': '__builtin_mul_overflow',
}
_PLAIN_BITWISE_SYMBOL = { ir.BitAnd: '&', ir.BitOr: '|', ir.BitXor: '^', ir.Shr: '>>' }

def _emit_wrap_arith( dest: str, left: ir.Operand, right: ir.Operand, symbol: str, dest_type: Type ) -> list[str]:
	# C's signed overflow is UB, not wraparound - cast to the same-width
	# unsigned type (well-defined wraparound there), compute, cast back
	# (implementation-defined but universally two's-complement in practice,
	# same posture as everywhere else this compiler already leans on real-
	# world compiler behavior over strict-ISO-C portability)
	ctype = c_type( dest_type )
	stem = dest_type.stem if isinstance( dest_type, Scalar ) else None
	l, r = _emit_operand( left ), _emit_operand( right )
	if stem in _SIGNED_TO_UNSIGNED:
		uctype = _SCALAR_C_TYPES[_SIGNED_TO_UNSIGNED[stem]]
		return [ f'\t{dest} = ({ctype})(({uctype})({l}) {symbol} ({uctype})({r}));' ]
	return [ f'\t{dest} = ({l}) {symbol} ({r});' ]

def _emit_check_arith( dest_temp_id: int, left: ir.Operand, right: ir.Operand, kind: str, result_spec: Specialization ) -> list[str]:
	ok_type = result_spec.args[0]
	ctype = c_type( ok_type )
	builtin = _ARITH_BUILTIN[kind]
	dest = f't{dest_temp_id}'
	l, r = _emit_operand( left ), _emit_operand( right )
	return [
		'\t{',
		f'\t\t{ctype} __tmp;',
		f'\t\tbool __overflow = {builtin}( {l}, {r}, &__tmp );',
		'\t\tif ( __overflow ) {',
		f'\t\t\t{dest}._tag = 1;',
		'\t\t} else {',
		f'\t\t\t{dest}._tag = 0;',
		f'\t\t\t{dest}._payload.ok = __tmp;',
		'\t\t}',
		'\t}',
	]

def _emit_saturate_arith( dest: str, left: ir.Operand, right: ir.Operand, kind: str, dest_type: Type ) -> list[str]:
	stem = dest_type.stem if isinstance( dest_type, Scalar ) else None
	if stem not in _SATURATE_LIMITS:
		raise NotImplementedError( f'saturating {kind} on {stem!r} is not supported yet (no MIN/MAX for i128/u128)' )
	min_c, max_c = _SATURATE_LIMITS[stem]
	ctype = c_type( dest_type )
	builtin = _ARITH_BUILTIN[kind]
	l, r = _emit_operand( left ), _emit_operand( right )
	if _is_unsigned_stem( stem ):
		clamp_expr = min_c if kind == 'sub' else max_c # only sub can underflow for an unsigned type
	elif kind == 'mul':
		clamp_expr = f'( ( ({l}) >= 0 ) == ( ({r}) >= 0 ) ? {max_c} : {min_c} )' # same sign -> positive overflow, differing sign -> negative
	else: # add/sub: overflow direction always follows the left operand's own sign
		clamp_expr = f'( ({l}) >= 0 ? {max_c} : {min_c} )'
	return [
		'\t{',
		f'\t\t{ctype} __tmp;',
		f'\t\tbool __overflow = {builtin}( {l}, {r}, &__tmp );',
		f'\t\t{dest} = __overflow ? {clamp_expr} : __tmp;',
		'\t}',
	]

def _emit_shl( instr ) -> list[str]:
	# no __builtin_shl_overflow exists - detect overflow by shifting the
	# result back down and comparing to the original value
	l, r = _emit_operand( instr.left ), _emit_operand( instr.right )
	dest_type = instr.dest.type if not isinstance( instr, ir.ShlCheck ) else instr.dest.type.args[0]
	if isinstance( instr, ir.ShlWrap ):
		ctype = c_type( instr.dest.type )
		stem = instr.dest.type.stem if isinstance( instr.dest.type, Scalar ) else None
		dest = _emit_operand( instr.dest )
		if stem in _SIGNED_TO_UNSIGNED:
			uctype = _SCALAR_C_TYPES[_SIGNED_TO_UNSIGNED[stem]]
			return [ f'\t{dest} = ({ctype})(({uctype})({l}) << ({r}));' ]
		return [ f'\t{dest} = ({l}) << ({r});' ]
	if isinstance( instr, ir.ShlCheck ):
		ctype = c_type( dest_type )
		dest = f't{instr.dest.id}'
		return [
			'\t{',
			f'\t\t{ctype} __tmp = ({l}) << ({r});',
			f'\t\tbool __overflow = ( __tmp >> ({r}) ) != ({l});',
			'\t\tif ( __overflow ) {',
			f'\t\t\t{dest}._tag = 1;',
			'\t\t} else {',
			f'\t\t\t{dest}._tag = 0;',
			f'\t\t\t{dest}._payload.ok = __tmp;',
			'\t\t}',
			'\t}',
		]
	# ShlSaturate
	stem = instr.dest.type.stem if isinstance( instr.dest.type, Scalar ) else None
	if stem not in _SATURATE_LIMITS:
		raise NotImplementedError( f'saturating shl on {stem!r} is not supported yet' )
	_min_c, max_c = _SATURATE_LIMITS[stem]
	ctype = c_type( instr.dest.type )
	dest = _emit_operand( instr.dest )
	return [
		'\t{',
		f'\t\t{ctype} __tmp = ({l}) << ({r});',
		f'\t\tbool __overflow = ( __tmp >> ({r}) ) != ({l});',
		f'\t\t{dest} = __overflow ? {max_c} : __tmp;',
		'\t}',
	]

_NEG_MODE = { ir.NegWrap: 'wrap', ir.NegCheck: 'check', ir.NegSaturate: 'saturate' }

def _emit_neg( instr ) -> list[str]:
	mode = _NEG_MODE[type(instr)]
	operand = _emit_operand( instr.operand )
	dest_type = instr.dest.type if mode != 'check' else instr.dest.type.args[0]
	stem = dest_type.stem if isinstance( dest_type, Scalar ) else None
	ctype = c_type( dest_type )
	if mode == 'wrap':
		dest = _emit_operand( instr.dest )
		if stem in _SIGNED_TO_UNSIGNED:
			uctype = _SCALAR_C_TYPES[_SIGNED_TO_UNSIGNED[stem]]
			return [ f'\t{dest} = ({ctype})(-({uctype})({operand}));' ]
		return [ f'\t{dest} = 0;' ] # unsigned wrap-negation: only 0 maps to itself, everything else wraps to (TYPE_MAX - x + 1) - see note below
	if mode == 'saturate':
		dest = _emit_operand( instr.dest )
		if stem not in _SATURATE_LIMITS:
			raise NotImplementedError( f'saturating negation on {stem!r} is not supported yet' )
		if _is_unsigned_stem( stem ):
			return [ f'\t{dest} = 0;' ] # unsigned negation always saturates to 0 (can never go negative)
		_min_c, max_c = _SATURATE_LIMITS[stem]
		return [
			'\t{',
			f'\t\t{ctype} __tmp;',
			f'\t\tbool __overflow = __builtin_sub_overflow( ({ctype})0, ({operand}), &__tmp );',
			f'\t\t{dest} = __overflow ? {max_c} : __tmp;', # negating MIN is the only way signed negation overflows, and it always overflows toward MAX
			'\t}',
		]
	# check
	dest = f't{instr.dest.id}'
	return [
		'\t{',
		f'\t\t{ctype} __tmp;',
		f'\t\tbool __overflow = __builtin_sub_overflow( ({ctype})0, ({operand}), &__tmp );',
		'\t\tif ( __overflow ) {',
		f'\t\t\t{dest}._tag = 1;',
		'\t\t} else {',
		f'\t\t\t{dest}._tag = 0;',
		f'\t\t\t{dest}._payload.ok = __tmp;',
		'\t\t}',
		'\t}',
	]

_CAST_MODE = { ir.CastWrap: 'wrap', ir.CastCheck: 'check', ir.CastSaturate: 'saturate' }

def _emit_cast( instr ) -> list[str]:
	# compiler.cast(T, x) / T(x) construction-sugar (lowering.py's shared
	# _lower_scalar_cast) - a scalar-to-scalar conversion, mode-respecting
	# same as +/-/* already are
	mode = _CAST_MODE[type(instr)]
	operand = _emit_operand( instr.operand )
	target_type = instr.dest.type if mode != 'check' else instr.dest.type.args[0]
	stem = target_type.stem if isinstance( target_type, Scalar ) else None
	ctype = c_type( target_type )
	if mode == 'wrap':
		# C's own integer conversion rules ARE wrap semantics for an
		# out-of-range value (well-defined, no UB, unlike an ARITHMETIC
		# operation on a signed type triggering signed-overflow UB) - a
		# plain cast is all this needs, no unsigned-roundtrip trick required
		dest = _emit_operand( instr.dest )
		return [ f'\t{dest} = ({ctype})({operand});' ]
	if stem not in _SATURATE_LIMITS:
		raise NotImplementedError( f'{mode} cast to {stem!r} is not supported yet (no MIN/MAX for i128/u128)' )
	min_c, max_c = _SATURATE_LIMITS[stem]
	# promoted to __int128 for the range comparison - every scalar width
	# this compiler supports OTHER than i128/u128 themselves (excluded
	# just above) fits inside __int128 without loss, sidestepping the
	# usual signed/unsigned-pairing headache a same-width comparison
	# would otherwise need (source and target can differ in both width
	# AND signedness - e.g. i32 -> u8, or u64 -> i16)
	if mode == 'saturate':
		dest = _emit_operand( instr.dest )
		return [
			'\t{',
			f'\t\t__int128 __wide = (__int128)({operand});',
			f'\t\t{dest} = ( __wide < (__int128)({min_c}) ) ? {min_c} : ( __wide > (__int128)({max_c}) ) ? {max_c} : ({ctype})({operand});',
			'\t}',
		]
	# check
	dest = f't{instr.dest.id}'
	return [
		'\t{',
		f'\t\t__int128 __wide = (__int128)({operand});',
		f'\t\tbool __overflow = ( __wide < (__int128)({min_c}) ) || ( __wide > (__int128)({max_c}) );',
		'\t\tif ( __overflow ) {',
		f'\t\t\t{dest}._tag = 1;',
		'\t\t} else {',
		f'\t\t\t{dest}._tag = 0;',
		f'\t\t\t{dest}._payload.ok = ({ctype})({operand});',
		'\t\t}',
		'\t}',
	]

# --- comparisons / control flow / calls / member access -----------------------

_CMP_SYMBOLS = {
	ir.CmpOp.EQ: '==', ir.CmpOp.NE: '!=', ir.CmpOp.LT: '<', ir.CmpOp.LE: '<=', ir.CmpOp.GT: '>', ir.CmpOp.GE: '>=',
}

def _member_access_operator( obj_type: Type|None ) -> str:
	return '->' if isinstance( obj_type, RCClass ) else '.'

def _emit_call_args( instr: ir.Call ) -> list[str]:
	params = instr.target.parameters or []
	values: list[str] = []
	if instr.receiver is not None:
		values.append( _emit_operand( instr.receiver ))
	positional = list( instr.args )
	for i, param in enumerate( params ):
		if i < len( positional ):
			values.append( _emit_operand( positional[i] ))
		else:
			values.append( _emit_operand( instr.kwargs[param.stem] ))
	return values

# --- function bodies -----------------------------------------------------

def emit_function( fn: LoweredFunction, *, prototype_only: bool = False ) -> str:
	function = fn.function
	proto = _function_prototype( function )
	if prototype_only or function.extern_lib is not None:
		return proto + ';'
	lines = [ proto + ' {' ]
	# locals (x: i32 = 1) have no DeclareTemp-style IR instruction of their
	# own - lowering.py just emits a plain Assign against a Variable that
	# was never separately "declared". C needs a declaration before use, so
	# this module synthesizes one inline at each local's first assignment
	# (see the ir.Assign branch below) - pre-seed with every parameter
	# (already declared via the signature itself, must never be
	# re-declared) so only genuine first-time locals trigger it
	declared: set[str] = set()
	if _has_self( function ):
		declared.add( mangle_qualname( _self_qualname( function )))
	for p in ( function.parameters or [] ):
		declared.add( mangle_qualname( p.qualname ))
	for instr in fn.instructions:
		lines.extend( _emit_instruction( instr, function = function, declared = declared ))
	lines.append( '}' )
	return '\n'.join( lines )

def _emit_instruction( instr: ir.Instruction, *, function: Function|None, declared: set[str] ) -> list[str]:
	# FuncStart/FuncEnd carry no independent C text of their own - the
	# surrounding prototype + braces (built from the LoweredFunction.function
	# object, not these markers) already represent them; see emit_function
	if isinstance( instr, ( ir.FuncStart, ir.FuncEnd )):
		return []

	if isinstance( instr, ir.Return ):
		if instr.value is None:
			# the entry point's declared metalpy return type is -> None, but
			# it compiles to C's real `int main`, which C itself requires a
			# real int return from - 0 is the only sensible synthesized
			# success code here (a real exit-code convention, if one is ever
			# needed, is a later/library concern, not this bare fallback)
			return [ '\treturn 0;' ] if _is_entry_point( function ) else [ '\treturn;' ]
		return [ f'\treturn {_emit_operand(instr.value)};' ]

	if isinstance( instr, ir.DeclareTemp ):
		return [ f'\t{c_type(instr.temp.type)} t{instr.temp.id};' ]
	if isinstance( instr, ir.DeleteTemp ):
		return [] # C block scoping already handles temp lifetime - nothing to emit
	if isinstance( instr, ir.Assign ):
		src = _emit_operand( instr.src )
		if isinstance( instr.dest, Variable ) and not instr.dest.is_global:
			name = mangle_qualname( instr.dest.qualname )
			if name not in declared:
				declared.add( name )
				return [ f'\t{c_type(instr.dest.type)} {name} = {src};' ]
			return [ f'\t{name} = {src};' ]
		# a global Variable is declared separately at file scope (Phase 7 -
		# emit_global) - never re-declared here, only assigned
		return [ f'\t{_emit_operand(instr.dest)} = {src};' ]

	if type( instr ) in _ARITH_BINOP_INFO:
		kind, mode = _ARITH_BINOP_INFO[type(instr)]
		dest_type = instr.dest.type
		if mode == 'wrap':
			return _emit_wrap_arith( _emit_operand( instr.dest ), instr.left, instr.right, _ARITH_SYMBOL[kind], dest_type )
		if mode == 'saturate':
			return _emit_saturate_arith( _emit_operand( instr.dest ), instr.left, instr.right, kind, dest_type )
		return _emit_check_arith( instr.dest.id, instr.left, instr.right, kind, dest_type )

	if isinstance( instr, ( ir.ShlWrap, ir.ShlCheck, ir.ShlSaturate )):
		return _emit_shl( instr )

	if isinstance( instr, ( ir.Div, ir.Mod )):
		# always Check mode against ZeroDivisionError, independent of the
		# active arithmetic mode - see lowering.py's own _expr_BinOp comment.
		# KNOWN GAP: signed INT_MIN / -1 is itself UB in C (the one value
		# that legitimately overflows a division) - not handled, matches
		# nothing exercising it yet
		ok_type = instr.dest.type.args[0]
		dest = f't{instr.dest.id}'
		l, r = _emit_operand( instr.left ), _emit_operand( instr.right )
		symbol = '/' if isinstance( instr, ir.Div ) else '%'
		return [
			f'\tif ( ({r}) == 0 ) {{',
			f'\t\t{dest}._tag = 1;',
			'\t} else {',
			f'\t\t{dest}._tag = 0;',
			f'\t\t{dest}._payload.ok = ({l}) {symbol} ({r});',
			'\t}',
		]

	if type( instr ) in _PLAIN_BITWISE_SYMBOL:
		dest = _emit_operand( instr.dest )
		symbol = _PLAIN_BITWISE_SYMBOL[type(instr)]
		return [ f'\t{dest} = ({_emit_operand(instr.left)}) {symbol} ({_emit_operand(instr.right)});' ]

	if isinstance( instr, ir.Invert ):
		return [ f'\t{_emit_operand(instr.dest)} = ~({_emit_operand(instr.operand)});' ]
	if type( instr ) in _NEG_MODE:
		return _emit_neg( instr )
	if type( instr ) in _CAST_MODE:
		return _emit_cast( instr )

	if isinstance( instr, ir.Cmp ):
		symbol = _CMP_SYMBOLS[instr.op]
		return [ f'\t{_emit_operand(instr.dest)} = ({_emit_operand(instr.left)}) {symbol} ({_emit_operand(instr.right)});' ]

	if isinstance( instr, ir.Label ):
		return [ f'{_c_label(instr.name)}:;' ]
	if isinstance( instr, ir.Jump ):
		return [ f'\tgoto {_c_label(instr.target)};' ]
	if isinstance( instr, ir.JumpIfFalse ):
		return [ f'\tif ( !({_emit_operand(instr.cond)}) ) goto {_c_label(instr.target)};' ]
	if isinstance( instr, ir.JumpIfTrue ):
		return [ f'\tif ( {_emit_operand(instr.cond)} ) goto {_c_label(instr.target)};' ]

	if isinstance( instr, ir.Call ):
		# a method call (instr.receiver is not None) is just an ordinary C
		# function call with self prepended as the first argument - _has_self/
		# _function_prototype already synthesize the matching `self`
		# PARAMETER this way for the callee's own definition (there's no
		# dot-call syntax here, this is C, not C++), and _emit_call_args
		# already prepends instr.receiver the same way
		target_name = mangle_qualname( instr.target.qualname )
		has_args = instr.receiver is not None or instr.args or instr.kwargs
		call_expr = f'{target_name}( {", ".join(_emit_call_args(instr))} )' if has_args else f'{target_name}()'
		if instr.dest is not None:
			return [ f'\t{_emit_operand(instr.dest)} = {call_expr};' ]
		return [ f'\t{call_expr};' ]

	if isinstance( instr, ir.GetAttr ):
		op = _member_access_operator( instr.obj.type )
		return [ f'\t{_emit_operand(instr.dest)} = ({_emit_operand(instr.obj)}){op}{_field_name(instr.attr)};' ]
	if isinstance( instr, ir.SetAttr ):
		op = _member_access_operator( instr.obj.type )
		return [ f'\t({_emit_operand(instr.obj)}){op}{_field_name(instr.attr)} = {_emit_operand(instr.value)};' ]

	if isinstance( instr, ir.GetItem ):
		# only ever reached for a raw pointer with no real __getitem__ (see
		# lowering.py's _expr_Subscript) - Ptr[T]/ConstPtr[T] are plain C
		# pointers (c_type), so C's own subscript operator applies directly
		return [ f'\t{_emit_operand(instr.dest)} = ({_emit_operand(instr.obj)})[{_emit_operand(instr.index)}];' ]
	if isinstance( instr, ir.SetItem ):
		return [ f'\t({_emit_operand(instr.obj)})[{_emit_operand(instr.index)}] = {_emit_operand(instr.value)};' ]

	if isinstance( instr, ir.AddrOf ):
		return [ f'\t{_emit_operand(instr.dest)} = &{_emit_operand(instr.value)};' ]

	if isinstance( instr, ir.SizeOf ):
		# a real class-like type's size is whatever the C compiler itself
		# computes for its struct/union body (sizeof(struct Foo), never
		# sizeof(struct Foo*) - _value_spelling gives the bare body type
		# even for an RCClass) - no field-layout algorithm exists earlier
		# in this compiler, nor should one
		return [ f'\t{_emit_operand(instr.dest)} = sizeof({_value_spelling(instr.type)});' ]

	if isinstance( instr, ir.Incref ):
		return [ f'\tretain_object( &({_emit_operand(instr.value)})->$header );' ]
	if isinstance( instr, ir.Decref ):
		# instr.value.type is always concrete RCClass-typed by the time the
		# emitter sees a bare Decref (cfg.py's own union-handling already
		# expands any TaggedUnion-typed Incref/Decref into a tag-gated
		# GetAttr+Cmp+Jump*+Incref/Decref sequence at the IR level - see the
		# plan's grounding facts) - its own synthesized destructor (see
		# emit_rcclass_destructor) is always the right one to reference
		destructor_name = _rcclass_destructor_name( instr.value.type )
		return [ f'\trelease_object( &({_emit_operand(instr.value)})->$header, {destructor_name} );' ]
	if isinstance( instr, ir.RefCount ):
		return [ f'\t{_emit_operand(instr.dest)} = ({_emit_operand(instr.value)})->$header.ref_count;' ]

	if isinstance( instr, ir.Allocate ):
		if isinstance( instr.cls, RCClass ):
			# routed through sys.alloc[dest.type] - the SAME allocation path
			# every other real allocation in the language goes through, not
			# an emitter-invented allocator (explicit user decision - see
			# the plan's Context section). lowering.py's _lower_allocate_
			# fields already guarantees this exact Specialization is
			# scheduled+lowered (Lowering._resolve_sys_function('alloc')) - its mangled
			# qualname is built the same way discovery._get_or_create_
			# specialization builds every Specialization's own qualname
			# (base.qualname + bracketed, comma-joined arg qualnames), so
			# no lookup is needed here, just the same string formula.
			# Ptr[T]'s c_type mapping uses _value_spelling for its own inner
			# T (see c_type's Ptr/ConstPtr branch), so sys.alloc[Foo]'s real
			# C return type is ALREADY struct Foo* - representationally
			# identical to dest's own type, no cast needed.
			dest_type = instr.dest.type
			alloc_name = mangle_qualname( f'sys.alloc[{dest_type.qualname}]' )
			dest = _emit_operand( instr.dest )
			lines = [ f'\t{dest} = {alloc_name}( 1 );' ]
			# freshly allocated = owned by dest immediately (lowering.py
			# never emits an Incref for the temp an Allocate itself produces)
			# - starting the header at 0 would underflow the very first
			# paired Decref
			lines.append( f'\t({dest})->$header.ref_count = 1;' )
			for name, value in instr.fields.items():
				lines.append( f'\t({dest})->{_field_name(name)} = {_emit_operand(value)};' )
			return lines
		# CStruct/CUnion - plain value construction, no header/no heap
		# allocation at all (see the grounding facts in the plan) - a C11
		# designated-initializer compound literal covers both (a union
		# with more than one field given would be a real error, but
		# nothing here re-validates that - discovery/lowering already did)
		ctype = c_type( instr.dest.type )
		dest = _emit_operand( instr.dest )
		if not instr.fields:
			return [ f'\t{dest} = ({ctype}){{0}};' ] # empty {} isn't valid standard C11
		field_inits = ', '.join( f'.{_field_name(name)} = {_emit_operand(value)}' for name, value in instr.fields.items() )
		return [ f'\t{dest} = ({ctype}){{ {field_inits} }};' ]

	if isinstance( instr, ir.OrReturn ):
		return _emit_or_return( instr, function )
	if isinstance( instr, ir.OrJump ):
		return _emit_or_jump( instr )
	if isinstance( instr, ir.Unwrap ):
		# instr.panic is a real, already-resolved sys.panic Function
		# reference (see ir.Unwrap's own docstring / Lowering._resolve_sys_function)
		# - this module has zero special knowledge of "panic", it just calls
		# whatever Function the IR handed it, the same as any other ir.Call
		value = _emit_operand( instr.value )
		panic_name = mangle_qualname( instr.panic.qualname )
		return [
			f'\tif ( ({value})._tag == 1 ) {{',
			f'\t\t{panic_name}( {_emit_operand(instr.errmsg)} );',
			'\t}',
			f'\t{_emit_operand(instr.dest)} = ({value})._payload.ok;',
		]
	if isinstance( instr, ir.UnwrapOr ):
		value = _emit_operand( instr.value )
		return [ f'\t{_emit_operand(instr.dest)} = ( ({value})._tag == 1 ) ? {_emit_operand(instr.default)} : ({value})._payload.ok;' ]

	raise NotImplementedError( f'_emit_instruction: unsupported instruction {instr!r} (later-phase work)' )

def _emit_or_return( instr: ir.OrReturn, function: Function ) -> list[str]:
	# OrReturn's own IR semantics ARE the branch (see ir.py's docstring:
	# "Err -> return Result::Err(...); Ok -> dest = payload") - this one
	# instruction expands to real conditional C here, not a pre-branched IR
	# sequence. A plain struct-copy return isn't legal C: the enclosing
	# function's own Result[FnT,E] and `value`'s Result[ArithT,E] are
	# DIFFERENT C struct tags even though E (and therefore the whole error
	# payload's byte layout) is identical - only the ._payload.err MEMBER
	# (itself always exactly type E on both sides) is copied across, not
	# the whole struct.
	value = _emit_operand( instr.value )
	dest = _emit_operand( instr.dest )
	return_type = function.return_type
	ret_ctype = c_type( return_type )
	return [
		f'\tif ( ({value})._tag == 1 ) {{',
		f'\t\t{ret_ctype} __err;',
		'\t\t__err._tag = 1;',
		f'\t\t__err._payload.err = ({value})._payload.err;',
		'\t\treturn __err;',
		'\t}',
		f'\t{dest} = ({value})._payload.ok;',
	]

def _emit_or_jump( instr: ir.OrJump ) -> list[str]:
	value = _emit_operand( instr.value )
	dest = _emit_operand( instr.dest )
	lines = [ f'\tif ( ({value})._tag == 1 ) {{' ]
	if instr.return_slot is not None:
		slot = _emit_operand( instr.return_slot )
		lines.append( f'\t\t{slot}._tag = 1;' )
		lines.append( f'\t\t{slot}._payload.err = ({value})._payload.err;' )
	lines.append( f'\t\tgoto {_c_label(instr.target)};' )
	lines.append( '\t}' )
	lines.append( f'\t{dest} = ({value})._payload.ok;' )
	return lines

# --- classes / globals -----------------------------------------------------

def emit_rcclass( cls: RCClass ) -> str:
	# ObjectHeader is the automatic first member of every RCClass C struct
	# (explicit user decision - see the plan's Context section) - this is
	# what lets sys.alloc[Foo]'s own generic byte-count allocation double as
	# the real object allocator: the header is just part of the struct's
	# own layout, sized by the same sizeof(struct Foo) as every other field.
	# Named $header, not header - a real metalpy field can never contain
	# '$' (not a legal character in a Python/metalpy identifier), so this
	# is guaranteed collision-free against a user class that itself
	# declares a field named `header` (mangle_qualname already relies on
	# the same GCC/Clang '$'-in-identifiers extension everywhere else in
	# this module's output, so this is nothing new).
	# Base-class fields (if any) come first, most-derived last - .attributes
	# only ever holds a class's OWN declared fields (discovery.py never
	# merges a base's own attributes in), so the base chain has to be
	# walked and flattened here.
	chain: list[RCClass] = []
	node: RCClass|None = cls
	while node is not None:
		chain.append( node )
		node = node.base
	attrs: list[tuple[str,Type]] = []
	for base_cls in reversed( chain ):
		attrs.extend( ( attr.stem, attr.type ) for attr in base_cls.attributes )
	name = mangle_type( cls )
	lines = [ f'struct {name} {{', '\tObjectHeader $header;' ]
	for field_name, field_type in attrs:
		lines.append( f'\t{_field_type_spelling(field_type)} {_field_name(field_name)};' )
	lines.append( '};' )
	return '\n'.join( lines )

def _rcclass_destructor_name( cls: Type ) -> str:
	# $$ (not a single $) - a single $ is exactly what mangle_qualname's own
	# '.'->'$' rule would ALSO produce for a real user method named e.g.
	# __destructor__ (class Foo: def __destructor__(self): ... mangles to
	# ...Foo$__destructor__, a genuine one-$ collision) - since no real
	# dotted qualname ever mangles to two CONSECUTIVE '$' from a plain
	# (non-generic-bracket) suffix, $$ is what actually guarantees this
	# can't collide with any real declared method, the same spirit as
	# emit_rcclass's own $header
	return f'{mangle_type(cls)}$$__destructor__'

def _type_needs_teardown( t: Type ) -> bool:
	# does tearing down a VALUE of this type require any action at all -
	# generalizes cfg.py's own is_rc()/rc_leaves() to ALSO recurse into
	# CStruct/TaggedUnion fields. cfg.py deliberately doesn't do that for
	# LOCAL VARIABLE tracking (see its own "v1 deliberately doesn't reach
	# into struct/union FIELDS" comment) - that's a separate, still-open
	# limitation of the ownership-tracking model this doesn't touch. A
	# destructor's own cascading walk is different: it already has to
	# visit every field of the RCClass being torn down regardless, so
	# finding RC leaves nested inside a by-value CStruct/TaggedUnion field
	# is no extra structural work, just recursion.
	base = t.base if isinstance( t, Specialization ) else t
	if isinstance( base, RCClass ):
		return True
	if isinstance( base, ( CStruct, TaggedUnion )):
		return any( _type_needs_teardown( attr.type ) for attr in base.attributes )
	# CUnion has no discriminant of its own to safely recurse through (see
	# _emit_field_teardown's own comment) - CEnum/Scalar/Ptr never need
	# teardown at all
	return False

def _emit_field_teardown( self_expr: str, field_type: Type ) -> list[str]:
	# recursively decrefs every RC leaf reachable from a VALUE at
	# self_expr, without needing any external discriminant. RCClass
	# (direct decref), CStruct (every field is always live - safe to walk
	# unconditionally), and TaggedUnion (a tag-gated decref into whichever
	# member's own tag says is live, mirroring cfg.py's own
	# _tag_gated_refcount_instructions at the IR level) are all safe to
	# recurse into this way. A bare CUnion has no discriminant of its own
	# to consult - only the ENCLOSING context (e.g. Result's own hand-
	# rolled _tag+_payload pairing) would know which member is live, and
	# there's no general way to detect that pairing structurally from the
	# union's own type alone - skipped, same "compiles clean, not
	# necessarily leak-free yet" posture Phase 3's own NULL-destructor
	# placeholder already established for a narrower case.
	if not _type_needs_teardown( field_type ):
		return []
	base = field_type.base if isinstance( field_type, Specialization ) else field_type
	if isinstance( base, RCClass ):
		destructor = _rcclass_destructor_name( field_type )
		return [ f'\trelease_object( &({self_expr})->$header, {destructor} );' ]
	if isinstance( base, CStruct ):
		lines: list[str] = []
		for attr in base.attributes:
			lines.extend( _emit_field_teardown( f'({self_expr}).{_field_name(attr.stem)}', attr.type ))
		return lines
	if isinstance( base, TaggedUnion ):
		tag_attr = base.names.get( 'tag' )
		data_attr = base.names.get( 'data' )
		if not ( isinstance( tag_attr, Variable ) and isinstance( data_attr, Variable )):
			# _tagged_union_storage never actually ran for this union (no
			# real construction/match anywhere reached it) - no real
			# runtime storage shape exists to tear down at all
			return []
		lines = [ '\t{', f'\t\tuint8_t __tag = ({self_expr}).{_field_name(tag_attr.stem)};' ]
		for i, member in enumerate( base.attributes ):
			if not _type_needs_teardown( member.type ):
				continue
			member_expr = f'({self_expr}).{_field_name(data_attr.stem)}.{_field_name(f"v_{member.stem}")}'
			lines.append( f'\t\tif ( __tag == {i} ) {{' )
			lines.extend( f'\t{inner}' for inner in _emit_field_teardown( member_expr, member.type ))
			lines.append( '\t\t}' )
		lines.append( '\t}' )
		return lines
	return []

def emit_rcclass_destructor( cls: RCClass ) -> str:
	# the void(*)(void*) release_object needs - pure emitter-side synthesis
	# (no compiler stage recognizes __del__ specially, and there's no IR
	# stream backing a synthesized destructor body - this module has to
	# build the C text directly, unlike every other emit_* function here).
	# Order: the class's own __del__ runs FIRST (as an ordinary function
	# call, not inlined - fields are still fully valid at this point),
	# THEN cascading decref into every RC leaf reachable from a field
	# (base fields before derived, matching emit_rcclass's own field
	# ordering - see _emit_field_teardown for how deeply this recurses),
	# THEN sys.free on the object's own backing memory.
	name = _rcclass_destructor_name( cls )
	ctype = c_type( cls )
	lines = [
		f'static void {name}( void* __obj ) {{',
		f'\t{ctype} self = ({ctype})__obj;',
	]
	del_fn = cls.get_local( '__del__' )
	if isinstance( del_fn, Function ):
		lines.append( f'\t{mangle_qualname(del_fn.qualname)}( self );' )
	chain: list[RCClass] = []
	node: RCClass|None = cls
	while node is not None:
		chain.append( node )
		node = node.base
	for base_cls in reversed( chain ):
		for attr in base_cls.attributes:
			lines.extend( _emit_field_teardown( f'self->{_field_name(attr.stem)}', attr.type ))
	sys_free_name = mangle_qualname( 'sys.free' )
	lines.append( f'\t{sys_free_name}( ( void* )self );' )
	lines.append( '}' )
	return '\n'.join( lines )

# --- string/bytes literal static-baking -----------------------------------
#
# a str/bytes literal is RCClass-typed, like any other instance of those
# classes - but unlike everything else this module bakes into C, it isn't
# built by any real IR (ir.Allocate/sys.alloc/etc.) - _expr_Constant folds
# it directly to ir.Const(type=<RCClass>, value=...) at lowering time (see
# the plan's grounding facts). The emitter special-cases these two specific
# classes (detected by qualname) and bakes a static, immortal object
# (header.ref_count = METALPY_IMMORTAL_REFCOUNT - never freed, matches every
# other string constant's lifetime in a real C program) matching that
# class's REAL field layout, mirroring lib/builtins's own current __data/
# __byte_size (str) and __data/__len (bytes) shape - same posture as
# Result's own hand-rolled ._tag/._payload field names being hardcoded
# throughout lowering.py/cfg.py already, not a new kind of coupling.
_STRING_LITERAL_RCCLASS_QUALNAMES = { 'builtins.str', 'builtins.bytes' }
_STRING_LITERAL_FIELDS = {
	'builtins.str': ( '__data', '__byte_size', True ), # True: __byte_size includes a trailing NUL (lib/builtins's own str.__byte_size comment)
	'builtins.bytes': ( '__data', '__len', False ),
}

def _string_literal_name( qualname: str, value: str|bytes ) -> str:
	# deterministic and content-derived (not a counter/registry) so
	# _emit_const stays a pure function - every reference to the SAME
	# literal (anywhere in the program) independently computes the SAME
	# name, and _emit_string_literals (below) is what actually guarantees
	# each distinct one is only ever DEFINED once
	payload = value if isinstance( value, bytes ) else value.encode( 'utf-8' )
	digest = hashlib.sha256( f'{qualname}:'.encode() + payload ).hexdigest()[:16]
	return f'__literal_{digest}'

def _emit_one_string_literal( qualname: str, value: str|bytes ) -> list[str]:
	data_field, len_field, nul_terminate = _STRING_LITERAL_FIELDS[qualname]
	payload = value.encode( 'utf-8' ) if isinstance( value, str ) else value
	data_bytes = payload + ( b'\x00' if nul_terminate else b'' )
	name = _string_literal_name( qualname, value )
	data_name = f'{name}$data'
	byte_list = ', '.join( str( b ) for b in data_bytes ) if data_bytes else '0'
	struct_name = mangle_qualname( qualname )
	return [
		f'static const uint8_t {data_name}[] = {{ {byte_list} }};',
		f'static struct {struct_name} {name} = {{',
		f'\t.$header = {{ .ref_count = METALPY_IMMORTAL_REFCOUNT }},',
		f'\t.{_field_name(data_field)} = {data_name},',
		f'\t.{_field_name(len_field)} = {len(data_bytes)},',
		'};',
	]

def _emit_string_literals( compiler: Compiler ) -> list[str]:
	# a program-wide collection pass, since C requires each static object
	# defined exactly once - walks every function's instructions looking
	# for a str/bytes-valued ir.Const, deduplicating by (qualname, value)
	# (the same pair _string_literal_name derives its name from, so two
	# occurrences of the identical literal anywhere in the program share
	# one static definition)
	seen: set[tuple[str,str|bytes]] = set()
	parts: list[str] = []
	for lf in compiler.functions:
		for instr in lf.instructions:
			for op in _iter_instruction_operands( instr ):
				if not ( isinstance( op, ir.Const ) and isinstance( op.value, ( str, bytes ) )):
					continue
				if not ( isinstance( op.type, RCClass ) and op.type.qualname in _STRING_LITERAL_RCCLASS_QUALNAMES ):
					continue
				key = ( op.type.qualname, op.value )
				if key in seen:
					continue
				seen.add( key )
				parts.append( '\n'.join( _emit_one_string_literal( *key )))
	return parts

def _iter_instruction_operands( instr: ir.Instruction ) -> list[ir.Operand]:
	# every Operand reachable from any field on this instruction - generic
	# over dataclasses.fields() (rather than a hand-picked list of field
	# names like 'value'/'left'/'right') so this can't silently miss a
	# shape some OTHER instruction kind uses for the same purpose (Assign's
	# src, GetAttr's obj, GetItem's index, ...) - a missed site here would
	# be a real latent bug (_emit_const would reference a static object
	# _emit_string_literals never actually defined)
	operands: list[ir.Operand] = []
	for f in dataclasses.fields( instr ):
		value = getattr( instr, f.name )
		if isinstance( value, ( ir.Const, ir.Temp, Variable )):
			operands.append( value )
		elif isinstance( value, list ):
			operands.extend( v for v in value if isinstance( v, ( ir.Const, ir.Temp, Variable )))
		elif isinstance( value, dict ):
			operands.extend( v for v in value.values() if isinstance( v, ( ir.Const, ir.Temp, Variable )))
	return operands

def emit_cstruct( cls: CStruct ) -> str:
	attrs = [ ( attr.stem, attr.type ) for attr in cls.attributes ]
	return _struct_or_union_body( mangle_type( cls ), 'struct', attrs )

def emit_cunion( cls: CUnion ) -> str:
	attrs = [ ( attr.stem, attr.type ) for attr in cls.attributes ]
	return _struct_or_union_body( mangle_type( cls ), 'union', attrs )

def emit_cenum( cls: CEnum ) -> str:
	# not a real C `enum` - .value_type can be any scalar width (u32/i32 seen
	# in real lib/ code), and C's own `enum` underlying type is
	# implementation-defined/usually int-only (see the plan's type-mapping
	# table) - a typedef of the real value type plus one static const per
	# member reproduces the same "named integer constant" semantics without
	# that portability trap
	name = mangle_type( cls )
	value_ctype = c_type( cls.value_type )
	lines = [ f'typedef {value_ctype} {name};' ]
	for key, value in cls.members.items():
		lines.append( f'static const {name} {name}${key} = {value};' )
	return '\n'.join( lines )

def emit_tagged_union( union: TaggedUnion ) -> str:
	# a TaggedUnion's REAL runtime representation is synthesized lowering-
	# side (Lowering._tagged_union_storage) as `tag: u8` + `data: <payload
	# CUnion>`, registered into union.names - NOT union.attributes, which
	# holds the LOGICAL members (Ok/Err, SharedReference/...) used for
	# type-matching/.leaves(), never the actual storage shape. Guaranteed
	# already populated by the time this runs: a TaggedUnion only ever
	# becomes a real compile unit (lands in compiler.tagged_unions) via a
	# construction or match site that already called _tagged_union_storage.
	tag_attr = union.names.get( 'tag' )
	data_attr = union.names.get( 'data' )
	assert isinstance( tag_attr, Variable ) and isinstance( data_attr, Variable ), \
		f'{union.qualname}: _tagged_union_storage has not run yet - no real storage shape to emit'
	name = mangle_type( union )
	return _struct_or_union_body( name, 'struct', [ ( tag_attr.stem, tag_attr.type ), ( data_attr.stem, data_attr.type ) ] )

def _is_trivial_global_init( instructions: list[ir.Instruction] ) -> bool:
	# Lowering.lower_global always produces a real IR instruction sequence
	# (DeclareTemp/Call/Allocate/... as needed, ending in an Assign of the
	# fully-computed value into the global itself) - "trivial" here means
	# that sequence collapsed to nothing more than the terminal Assign, with
	# a bare Const as its source (u32(-11)'s own bit-reinterpretation
	# already folds to a Const at lowering time - no CastWrap instruction
	# even gets emitted for a literal argument, see _lower_scalar_cast)
	return (
		len( instructions ) == 1
		and isinstance( instructions[0], ir.Assign )
		and isinstance( instructions[0].src, ir.Const )
	)

def emit_global( g: LoweredGlobal ) -> str:
	name = mangle_qualname( g.variable.qualname )
	ctype = c_type( g.variable.type )
	if _is_trivial_global_init( g.instructions ):
		value = _emit_operand( g.instructions[0].src )
		return f'{ctype} {name} = {value};'
	# a non-trivial initializer (anything needing a real computation - an
	# RCClass construction, an arithmetic expression, ...) flattens into a
	# private init function, reusing _emit_instruction exactly like an
	# ordinary function body does (function=None is safe here: lower_global
	# never emits ir.Return/ir.OrReturn, the only two branches that read
	# it - defer/errdefer/loops can't appear in a global initializer at all,
	# see lower_global's own comment). Wiring this init function into a
	# real process entry point is out of scope (linking-adjacent,
	# C_EMITTER.md excludes it) - it just needs to exist and compile. The
	# global itself gets a {0} zero initializer in the meantime - a valid
	# C11 initializer for ANY type alike (ISO C11 6.7.9p11: a scalar
	# initializer may be "optionally enclosed in braces"), matching the
	# same convention ir.Allocate's own empty-fields branch already uses
	init_name = f'__metalpy_init_{name}'
	lines = [
		f'{ctype} {name} = {{0}};',
		'',
		f'static void {init_name}( void ) {{',
	]
	declared: set[str] = set()
	for instr in g.instructions:
		lines.extend( _emit_instruction( instr, function = None, declared = declared ))
	lines.append( '}' )
	return '\n'.join( lines )

def _emit_value_type_bodies( compiler: Compiler ) -> list[str]:
	# CStruct/CUnion/TaggedUnion bodies, topologically sorted on by-value-
	# embedded fields (Result[i32,E] embeds ResultPayload[i32,E] BY VALUE -
	# C requires the payload's full definition before it can be used as a
	# struct member, unlike an RCClass field, which is always a pointer and
	# never forces an ordering - an opaque forward-declared tag is enough
	# for that). Generic (type_params is not None) classes are skipped
	# entirely - only their concrete Specializations (already separate
	# entries in these same lists, see Lowering.monomorphize_class) have a
	# real C representation.
	classes: list[ClassLike] = (
		[ c for c in compiler.cstructs if not c.type_params ]
		+ [ c for c in compiler.cunions if not c.type_params ]
		+ [ c for c in compiler.tagged_unions if not c.type_params ]
	)
	# matched by qualname, not object identity: a field's own type is
	# whatever Specialization object substitution produced (e.g.
	# ResultPayload[i32,OverflowError]), which is a DIFFERENT object from
	# the monomorphized ClassLike copy sitting in compiler.cunions - the
	# two are deliberately given the same qualname (monomorphize_class sets
	# qualname = spec.qualname) precisely so callers can bridge the two
	# this way
	by_qualname = { c.qualname: c for c in classes }
	visited: set[str] = set()
	ordered: list[ClassLike] = []
	def visit( cls: ClassLike ) -> None:
		if cls.qualname in visited:
			return
		visited.add( cls.qualname )
		if isinstance( cls, TaggedUnion ):
			# a TaggedUnion's REAL by-value dependency is its synthesized
			# `data` field (the payload CUnion) - .attributes holds the
			# LOGICAL members (Ok/Err/...) instead, which aren't part of
			# the actual C struct layout at all (see emit_tagged_union)
			data_attr = cls.names.get( 'data' )
			dep_types = [ data_attr.type ] if isinstance( data_attr, Variable ) else []
		else:
			dep_types = [ attr.type for attr in cls.attributes ]
		for dep_type in dep_types:
			dep = by_qualname.get( getattr( dep_type, 'qualname', None ))
			if dep is not None:
				visit( dep )
		ordered.append( cls )
	for cls in classes:
		visit( cls )
	parts = []
	for cls in ordered:
		if isinstance( cls, CUnion ):
			parts.append( emit_cunion( cls ))
		elif isinstance( cls, TaggedUnion ):
			parts.append( emit_tagged_union( cls ))
		else:
			parts.append( emit_cstruct( cls ))
	return parts

# --- whole-program driver ------------------------------------------------

def _rcclass_was_constructed( cls: RCClass, compiler: Compiler ) -> bool:
	# a class landing in compiler.rcclasses does NOT by itself mean an
	# instance was ever actually heap-allocated - a bare parameter/local
	# type annotation (x: Foo) schedules the CLASS the same way construction
	# does (see lowering.py's own var-type scheduling), independent of
	# whether .__allocate__()/sys.alloc[Foo] was ever reached. A class only
	# EVER needs a real destructor if sys.alloc[cls] itself was scheduled
	# (Lowering._lower_allocate_fields's RCClass branch - the one and only
	# place that happens), which is also exactly the trigger that already
	# schedules sys.free/__del__ for it - so this is the precise signal for
	# "will release_object's own function-pointer argument ever actually be
	# invoked for this class." Classes that are only ever baked as an
	# immortal string/bytes literal (never dynamically constructed) are the
	# motivating case: release_object skips an immortal object's destructor
	# call entirely at runtime, so the destructor function itself doesn't
	# need to exist (and its own body's sys.free call, needed unconditionally
	# by every OTHER real destructor, was never scheduled either).
	alloc_qualname = f'sys.alloc[{cls.qualname}]'
	return any( lf.function.qualname == alloc_qualname for lf in compiler.functions )

def emit_c( compiler: Compiler ) -> str:
	''' single C11 translation unit - see the plan's "three-pass emission
	order" decision. Linking is out of scope (C_EMITTER.md); the whole
	program is already collected into one Compiler instance, so there's no
	reason to split output across files. '''
	parts: list[str] = [ PROLOGUE, _NONE_PLACEHOLDER_TYPEDEF ]

	# pass 1: forward declarations (opaque RCClass tags, full CEnum bodies,
	# full CStruct/CUnion/TaggedUnion bodies in dependency order, function
	# prototypes)
	#
	# the opaque RCClass tags have to come FIRST, before anything else -
	# a bare `struct Foo` tag mentioned for the very first time INSIDE a
	# function prototype's PARAMETER LIST gets C's own "function prototype
	# scope" (ISO C11 6.2.1p4), a SEPARATE type from the real file-scope
	# struct Foo{...} defined later in pass 2, even though they're spelled
	# identically - confirmed via a real clang error ("conflicting types
	# for ...", "will not be visible outside of this function") once a
	# class was used as a plain parameter type before its own body was
	# ever emitted (every earlier RCClass milestone happened to dodge this
	# by only ever having a class appear in a RETURN type first, which
	# sits outside the parameter list and doesn't trigger the rule -
	# dumb luck, not a real guarantee). An explicit bare `struct Foo;` at
	# file scope, before any prototype, forces the tag to already be a
	# real file-scope type by the time anything references it.
	for cls in compiler.rcclasses:
		if not cls.type_params:
			parts.append( f'struct {mangle_type(cls)};' )
	for cls in compiler.cenums: # CEnum is never generic - no type_params field exists on it at all
		parts.append( emit_cenum( cls ))
	parts.extend( _emit_value_type_bodies( compiler ))
	for lf in compiler.functions:
		parts.append( emit_function( lf, prototype_only = True ))
	# a destructor is referenced by NAME (a function pointer value passed to
	# release_object), never just called directly - unlike a struct tag,
	# that needs a real prototype in scope first, and unlike an ordinary
	# Function, nothing else already provides one (destructors have no
	# backing Function/LoweredFunction entry at all - pure emitter-side
	# synthesis, see emit_rcclass_destructor)
	for cls in compiler.rcclasses:
		if not cls.type_params and _rcclass_was_constructed( cls, compiler ):
			parts.append( f'static void {_rcclass_destructor_name(cls)}( void* obj );' )

	# pass 2: full RCClass struct bodies (every other tag already exists)
	for cls in compiler.rcclasses:
		if not cls.type_params:
			parts.append( emit_rcclass( cls ))

	# pass 3: string/bytes literal static objects (need str/bytes's own
	# full RCClass body from pass 2 first) and global definitions, then
	# full function bodies (which may reference either by address), then
	# destructor bodies (need the struct's own full definition from pass 2
	# to dereference self->field)
	parts.extend( _emit_string_literals( compiler ))
	for g in compiler.globals:
		parts.append( emit_global( g ))
	for lf in compiler.functions:
		parts.append( emit_function( lf ))
	for cls in compiler.rcclasses:
		if not cls.type_params and _rcclass_was_constructed( cls, compiler ):
			parts.append( emit_rcclass_destructor( cls ))

	return '\n\n'.join( part for part in parts if part ) + '\n'
