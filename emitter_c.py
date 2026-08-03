# stdlib imports:
import dataclasses
import re

# local imports:
import ir
from compiler import Compiler, LoweredFunction, LoweredGlobal
from mpy_types import (
	ClassLike, CEnum, CStruct, CUnion, Copy, Function, Move, Parameter,
	RCClass, Scalar, Specialization, TaggedUnion, Type, TypeVar, Variable,
)

# stage 3: turns a fully-lowered Compiler's output into C11 source. Pure
# translation - by the time emit_c() runs, every real dependency is already
# discovered/scheduled/lowered (see ARCHITECTURE.md's stage split); this
# module does no further discovery of its own, with ONE deliberate exception
# - see _collect_specializations()'s own docstring.

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
			inner = c_type( t.args[0] )
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

# --- generic specialization discovery/synthesis -------------------------------
#
# compiler.py's _enqueue() deliberately never schedules a ClassLike
# Specialization as its own compile unit (only a Function-based one is - see
# _enqueue's own comment) - only the unspecialized generic base (e.g. bare
# `Result`) ends up in compiler.cstructs/.cunions/.tagged_unions/.rcclasses.
# A concrete specialization like Result[i32,OverflowError] therefore never
# appears anywhere in Compiler's own output lists, even though real IR
# (AddCheck's own dest.type, a Call's argument types, ...) references it
# constantly. The emitter has to find every one of these itself by walking
# the already-lowered IR/type graph, then synthesize each one's C struct/
# union body by substituting the generic's own type_params with that
# Specialization's concrete args - this is translation, not new discovery
# (nothing here decides SET of types is used, only surfaces objects the
# rest of the compiler already built and is holding onto).

def _substitute( t: Type, mapping: dict[int,Type], discovery ) -> Type:
	if isinstance( t, TypeVar ):
		return mapping.get( id( t ), t )
	if isinstance( t, Specialization ):
		new_args = [ _substitute( a, mapping, discovery ) for a in t.args ]
		if all( a is b for a, b in zip( new_args, t.args )):
			return t
		return discovery._get_or_create_specialization( t.base, new_args )
	return t # Scalar/RCClass/CStruct/CUnion/TaggedUnion/CEnum/Move/Copy - Move/Copy
	         # substitution isn't needed here: they never appear as a class
	         # attribute's own type, only a Parameter's, which this function
	         # (used solely for attribute-type substitution) never sees

def _substitute_type( t: Type, spec: Specialization, discovery ) -> Type:
	type_params = getattr( spec.base, 'type_params', None ) or []
	mapping = { id( tp ): arg for tp, arg in zip( type_params, spec.args ) }
	return _substitute( t, mapping, discovery )

def _iter_types_in_value( val ):
	if isinstance( val, ( ir.Temp, ir.Const, Variable )):
		if val.type is not None:
			yield val.type
	elif isinstance( val, ( RCClass, CStruct, CUnion, TaggedUnion, CEnum, Specialization, Scalar, Move, Copy )):
		yield val
	elif isinstance( val, dict ):
		for v in val.values():
			yield from _iter_types_in_value( v )
	elif isinstance( val, ( list, tuple )):
		for v in val:
			yield from _iter_types_in_value( v )

def _iter_operand_types_in_instruction( instr: ir.Instruction ):
	for f in dataclasses.fields( instr ):
		yield from _iter_types_in_value( getattr( instr, f.name ))

def _is_concrete( t: Type ) -> bool:
	# a Specialization can be genuinely abstract, not a real instantiation -
	# e.g. a generic class's OWN methods routinely declare things like
	# `-> Result[T,E]` referring to the enclosing class's still-unbound type
	# params (Result.Ok/.Err's own return type annotation is exactly this).
	# discovery.py's _get_or_create_specialization builds a real
	# Specialization object for that self-referential expression too - it
	# just isn't a CONCRETE one, and must never be mistaken for one here
	# (substituting through it would leak an unresolved TypeVar into a real
	# struct field - confirmed via a real repro against this exact fixture)
	if isinstance( t, TypeVar ):
		return False
	if isinstance( t, Specialization ):
		return all( _is_concrete( a ) for a in t.args )
	if isinstance( t, ( Move, Copy )):
		return _is_concrete( t.inner )
	return True

def _collect_specializations( compiler: Compiler ) -> list[Specialization]:
	discovery = compiler.disco
	found: dict[str,Specialization] = {}

	def visit( t: Type|None ) -> None:
		if t is None:
			return
		if isinstance( t, ( Move, Copy )):
			visit( t.inner )
			return
		if isinstance( t, Specialization ):
			if isinstance( t.base, ( CStruct, CUnion, TaggedUnion )) and _is_concrete( t ) and t.qualname not in found:
				found[t.qualname] = t
				for attr in t.base.attributes:
					visit( _substitute_type( attr.type, t, discovery ))
			# RCClass-based Specializations are Phase 3+ work (generic
			# RCClass construction needs the same sys.alloc[T]/header
			# machinery real RCClass construction does) - deliberately not
			# collected here; c_type() still names them correctly (a
			# pointer), just without a body, matching "not exercised by
			# anything yet" rather than silently mis-handling it
			for a in t.args:
				visit( a )
			return
		# plain ClassLike/Scalar/TypeVar/CEnum: nothing further to recurse
		# into here - their own .attributes are walked via the root loop below

	for lf in compiler.functions:
		fn = lf.function
		visit( fn.return_type )
		for p in ( fn.parameters or [] ):
			visit( p.type )
		for instr in lf.instructions:
			for t in _iter_operand_types_in_instruction( instr ):
				visit( t )
	for cls_list in ( compiler.rcclasses, compiler.cstructs, compiler.cunions, compiler.tagged_unions ):
		for cls in cls_list:
			for attr in cls.attributes:
				visit( attr.type )

	return list( found.values() )

def _struct_or_union_body( name: str, keyword: str, attrs: list[tuple[str,Type]] ) -> str:
	lines = [ f'{keyword} {name} {{' ]
	for field_name, field_type in attrs:
		lines.append( f'\t{c_type(field_type)} {field_name};' )
	lines.append( '};' )
	return '\n'.join( lines )

def emit_specialization( spec: Specialization, discovery ) -> str:
	base = spec.base
	attrs = [ ( attr.stem, _substitute_type( attr.type, spec, discovery )) for attr in base.attributes ]
	name = mangle_type( spec )
	keyword = _class_keyword( base )
	return _struct_or_union_body( name, keyword, attrs )

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
	raise NotImplementedError( f'_emit_const: unsupported constant {c!r} - str/bytes literal static-baking is later-phase work' )

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

def _emit_instruction( instr: ir.Instruction, *, function: Function, declared: set[str] ) -> list[str]:
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
		if instr.receiver is not None:
			raise NotImplementedError( '_emit_instruction: method calls (Call.receiver) are Phase 3+ work' )
		target_name = mangle_qualname( instr.target.qualname )
		call_expr = f'{target_name}( {", ".join(_emit_call_args(instr))} )' if instr.args or instr.kwargs else f'{target_name}()'
		if instr.dest is not None:
			return [ f'\t{_emit_operand(instr.dest)} = {call_expr};' ]
		return [ f'\t{call_expr};' ]

	if isinstance( instr, ir.GetAttr ):
		op = _member_access_operator( instr.obj.type )
		return [ f'\t{_emit_operand(instr.dest)} = ({_emit_operand(instr.obj)}){op}{instr.attr};' ]
	if isinstance( instr, ir.SetAttr ):
		op = _member_access_operator( instr.obj.type )
		return [ f'\t({_emit_operand(instr.obj)}){op}{instr.attr} = {_emit_operand(instr.value)};' ]

	if isinstance( instr, ir.OrReturn ):
		return _emit_or_return( instr, function )
	if isinstance( instr, ir.OrJump ):
		return _emit_or_jump( instr )
	if isinstance( instr, ir.Unwrap ):
		# instr.panic is a real, already-resolved sys.panic Function
		# reference (see ir.Unwrap's own docstring / Lowering._resolve_sys_panic)
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
	raise NotImplementedError( 'emit_rcclass: Phase 3 work' )

def emit_cstruct( cls: CStruct ) -> str:
	attrs = [ ( attr.stem, attr.type ) for attr in cls.attributes ]
	return _struct_or_union_body( mangle_type( cls ), 'struct', attrs )

def emit_cunion( cls: CUnion ) -> str:
	attrs = [ ( attr.stem, attr.type ) for attr in cls.attributes ]
	return _struct_or_union_body( mangle_type( cls ), 'union', attrs )

def emit_cenum( cls: CEnum ) -> str:
	raise NotImplementedError( 'emit_cenum: Phase 2 work' )

def emit_tagged_union( union: TaggedUnion ) -> str:
	raise NotImplementedError( 'emit_tagged_union: Phase 5 work' )

def emit_global( g: LoweredGlobal ) -> str:
	raise NotImplementedError( 'emit_global: Phase 7 work' )

# --- whole-program driver ------------------------------------------------

def emit_c( compiler: Compiler ) -> str:
	''' single C11 translation unit - see the plan's "three-pass emission
	order" decision. Linking is out of scope (C_EMITTER.md); the whole
	program is already collected into one Compiler instance, so there's no
	reason to split output across files. '''
	parts: list[str] = [ PROLOGUE ]

	specializations = _collect_specializations( compiler )

	# pass 1: forward declarations (opaque RCClass tags, full CEnum/CStruct/
	# CUnion/TaggedUnion bodies in dependency order, function prototypes).
	# Generic (type_params is not None) classes have no C representation of
	# their own - only their concrete Specializations (collected above) do.
	for cls in compiler.cenums:
		if not cls.type_params:
			parts.append( emit_cenum( cls ))
	for cls in compiler.cstructs:
		if not cls.type_params:
			parts.append( emit_cstruct( cls ))
	for cls in compiler.cunions:
		if not cls.type_params:
			parts.append( emit_cunion( cls ))
	for union in compiler.tagged_unions:
		if not union.type_params:
			parts.append( emit_tagged_union( union ))
	for spec in specializations:
		parts.append( emit_specialization( spec, compiler.disco ))
	for lf in compiler.functions:
		parts.append( emit_function( lf, prototype_only = True ))

	# pass 2: full RCClass struct bodies (every other tag already exists)
	for cls in compiler.rcclasses:
		if not cls.type_params:
			parts.append( emit_rcclass( cls ))

	# pass 3: global definitions, then full function bodies
	for g in compiler.globals:
		parts.append( emit_global( g ))
	for lf in compiler.functions:
		parts.append( emit_function( lf ))

	return '\n\n'.join( part for part in parts if part ) + '\n'
