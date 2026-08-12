# stdlib imports:
import dataclasses
import hashlib
import re

# local imports:
import ir
from compiler import Compiler, LoweredFunction, LoweredGlobal
from discovery import is_stub_body
from mpy_types import (
	CallableType, CEnum, ClassLike, CStruct, CType, CUnion, Copy, Function, Move,
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
	void (*destructor)(void*); // set once at construction (see emit_c's ir.Allocate codegen), read here on every release - adjacent to ref_count in the same cache line the atomic decrement below already touches, not a separate fetch
} ObjectHeader;

static inline void retain_object( ObjectHeader* obj ) {
	if ( obj && obj->ref_count != METALPY_IMMORTAL_REFCOUNT ) {
		atomic_fetch_add( &obj->ref_count, 1 );
	}
}

// the destructor was previously an explicit argument, passed as a compile-
// time literal at every call site - redundant with the header's own
// destructor field (set once at construction), which every caller can
// already reach directly. Reading it here uniformly also means every
// release, not just a type-erased one, is now safe to call through a
// base-typed reference once RCClass subclassing exists (see
// PLAN_SUBCLASSING_VTABLES_COM.md's "Why RCClass is deferred") - without
// making retain_object/the refcount increment/decrement themselves
// virtual, which is the actual hot-path cost that plan deliberately
// avoided paying
static inline void release_object( ObjectHeader* obj ) {
	if ( obj && obj->ref_count != METALPY_IMMORTAL_REFCOUNT ) {
		if ( atomic_fetch_sub( &obj->ref_count, 1 ) == 1 ) {
			if ( obj->destructor ) {
				obj->destructor( obj );
			}
		}
	}
}

// metalpy arithmetic intrinsics — dispatch to compiler builtins (GCC/Clang)
// or manual checks (MSVC). All metalpy scalars are <= 64 bits.
//
// defined(_MSC_VER) && !defined(__clang__), not just _MSC_VER: clang
// targeting Windows defines _MSC_VER too (for MSVC source compatibility)
// even though it fully supports __int128/__builtin_*_overflow, unlike
// real MSVC (cl.exe) - a bare #ifdef _MSC_VER wrongly routed clang-on-
// Windows through the int64_t/uint64_t fallback below, whose
// __metalpy_wideint (signed 64-bit) can't represent usize/u64's full
// unsigned range - a checked cast to usize with ANY valid value greater
// than INT64_MAX would (silently, incorrectly) read as an overflow, and
// in practice even small in-range values misfired once __metalpy_wideint
// end up promoted from a corrupted MAX constant - see the checked-cast
// test this was actually caught by.
#if defined(_MSC_VER) && !defined(__clang__)
typedef int64_t __metalpy_wideint;
typedef uint64_t __metalpy_wideuint;

// MSVC doesn't have __builtin_*_overflow — implement manually.
// The emitter widens operands to int64_t/uint64_t before calling these,
// so we only need per-signedness helpers for the widest type.
static inline bool __metalpy_sadd_overflow(int64_t a, int64_t b, int64_t* r) {
	*r = a + b;
	return (a > 0 && b > 0 && *r < 0) || (a < 0 && b < 0 && *r > 0);
}
static inline bool __metalpy_uadd_overflow(uint64_t a, uint64_t b, uint64_t* r) {
	*r = a + b;
	return *r < a;
}
static inline bool __metalpy_ssub_overflow(int64_t a, int64_t b, int64_t* r) {
	*r = a - b;
	return (a >= 0 && b < 0 && *r < 0) || (a < 0 && b >= 0 && *r > 0);
}
static inline bool __metalpy_usub_overflow(uint64_t a, uint64_t b, uint64_t* r) {
	*r = a - b;
	return a < b;
}
static inline bool __metalpy_smul_overflow(int64_t a, int64_t b, int64_t* r) {
	*r = a * b;
	if (a == 0 || b == 0) return false;
	if (a < 0) { a = -a; b = -b; }
	return a > INT64_MAX / (b < 0 ? -b : b);
}
static inline bool __metalpy_umul_overflow(uint64_t a, uint64_t b, uint64_t* r) {
	*r = a * b;
	if (a == 0) return false;
	return *r / a != b;
}

// _Generic dispatch: the emitter calls __metalpy_add_overflow(a,b,r)
// where *r is a local of the concrete scalar type. This picks the right
// signed/unsigned variant based on the type of *r.
#define __metalpy_add_overflow(a,b,r) \
	_Generic(*(r), \
		uint64_t: __metalpy_uadd_overflow, \
		uint32_t: __metalpy_uadd_overflow, \
		uint16_t: __metalpy_uadd_overflow, \
		uint8_t:  __metalpy_uadd_overflow, \
		int64_t:  __metalpy_sadd_overflow, \
		int32_t:  __metalpy_sadd_overflow, \
		int16_t:  __metalpy_sadd_overflow, \
		int8_t:   __metalpy_sadd_overflow  \
	)(a,b,r)
#define __metalpy_sub_overflow(a,b,r) \
	_Generic(*(r), \
		uint64_t: __metalpy_usub_overflow, \
		uint32_t: __metalpy_usub_overflow, \
		uint16_t: __metalpy_usub_overflow, \
		uint8_t:  __metalpy_usub_overflow, \
		int64_t:  __metalpy_ssub_overflow, \
		int32_t:  __metalpy_ssub_overflow, \
		int16_t:  __metalpy_ssub_overflow, \
		int8_t:   __metalpy_ssub_overflow  \
	)(a,b,r)
#define __metalpy_mul_overflow(a,b,r) \
	_Generic(*(r), \
		uint64_t: __metalpy_umul_overflow, \
		uint32_t: __metalpy_umul_overflow, \
		uint16_t: __metalpy_umul_overflow, \
		uint8_t:  __metalpy_umul_overflow, \
		int64_t:  __metalpy_smul_overflow, \
		int32_t:  __metalpy_smul_overflow, \
		int16_t:  __metalpy_smul_overflow, \
		int8_t:   __metalpy_smul_overflow  \
	)(a,b,r)
#else
// GCC/Clang: use compiler builtins directly
typedef __int128 __metalpy_wideint;
typedef unsigned __int128 __metalpy_wideuint;
#define __metalpy_add_overflow(a,b,r) __builtin_add_overflow(a,b,r)
#define __metalpy_sub_overflow(a,b,r) __builtin_sub_overflow(a,b,r)
#define __metalpy_mul_overflow(a,b,r) __builtin_mul_overflow(a,b,r)
#endif
// Windows: call SetConsoleOutputCP(CP_UTF8) so Unicode print() works.
// Called from main() on every Windows build, and from the custom entry
// point (mainCRTStartup below) when the CRT is not linked.
#ifdef _WIN32
#define CP_UTF8 65001
#ifdef _MSC_VER
int __stdcall SetConsoleOutputCP(unsigned int);
#pragma comment(linker, "/alternatename:__imp__SetConsoleOutputCP=__imp_SetConsoleOutputCP")
#else
int __stdcall SetConsoleOutputCP(unsigned int);
#endif
static void __metalpy_init( void ) {
	SetConsoleOutputCP( CP_UTF8 );
}
#else
static void __metalpy_init( void ) {}
#endif
'''



# NOT part of C_EMITTER.md's own verbatim prologue above - a synthesized
# TaggedUnion payload's None member (e.g. Ptr[u8]|None) needs a real 1-byte
# type (see the plan's type-mapping table); 'void' (c_type(NoneType)'s
# spelling everywhere else) is not a legal struct/union member type
_NONE_PLACEHOLDER_TYPE = 'MetalpyNone'
_NONE_PLACEHOLDER_TYPEDEF = f'typedef unsigned char {_NONE_PLACEHOLDER_TYPE};'

# --- name mangling -----------------------------------------------------------

# C reserved keywords that can't be used as bare local/parameter names.
# `_c_local_name` uses this to safely prefix them with `_` when unmangled.
_C_KEYWORDS: frozenset[str] = frozenset([
	'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
	'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if',
	'int', 'long', 'register', 'return', 'short', 'signed', 'sizeof',
	'static', 'struct', 'switch', 'typedef', 'union', 'unsigned', 'void',
	'volatile', 'while', '_Bool', '_Complex', '_Imaginary', 'inline',
	'restrict', '_Alignas', '_Alignof', '_Atomic', '_Generic', '_Noreturn',
	'_Static_assert', '_Thread_local',
])

def _c_local_name( stem: str ) -> str:
	''' return a C-safe local variable name. C keywords get a leading
	underscore; everything else passes through unchanged. '''
	return f'_{stem}' if stem in _C_KEYWORDS else stem

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
	'i128': '__metalpy_wideint', 'u128': '__metalpy_wideuint',
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
			inner_type = t.args[0]
			# Ptr[None]/ConstPtr[None] is void*/const void* — the
			# pointee is 'nothing', not a real value type
			if isinstance( inner_type, Scalar ) and inner_type.stem == 'NoneType':
				inner = 'void'
			else:
				inner = _value_spelling( inner_type )
			return f'{inner}*' if base.stem == 'Ptr' else f'const {inner}*'
		if isinstance( base, RCClass ):
			return f'struct {mangle_type(t)}*'
		if isinstance( base, ( CStruct, CUnion, TaggedUnion )):
			return f'{_class_keyword(base)} {mangle_type(t)}'
		raise NotImplementedError( f'c_type: unsupported Specialization base {base!r}' )
	if isinstance( t, Scalar ):
		if t.stem == 'NoneType':
			return _NONE_PLACEHOLDER_TYPE
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
	if isinstance( t, CType ):
		return t.c_name
	raise NotImplementedError( f'c_type: unsupported type {t!r}' )

def _is_noreturn( t: Type|None ) -> bool:
	return isinstance( t, Scalar ) and t.stem == 'NoReturn'

def _callable_ptr_type( t: Type|None ) -> CallableType|None:
	''' t's own CallableType if t is Ptr[Callable[...]] (see
	PLAN_CALLABLE.md) - the ptr-vs-bare distinction and the interning both
	live in discovery.py/type_resolver.py already (see TypeResolver.
	_callable_type_of); this is emitter_c.py's own copy of the same
	structural check since this module works on Type objects directly,
	with no TypeResolver instance around to call. '''
	if isinstance( t, Specialization ) and isinstance( t.base, Scalar ) and t.base.stem == 'Ptr':
		inner = t.args[0]
		if isinstance( inner, CallableType ):
			return inner
	return None

def _fn_ptr_cast_type( ret: str, params: list[str] ) -> str:
	''' the C function-pointer TYPE spelling itself (RetType (*)(ParamTypes),
	no name) - shared by emit_interface_vtable_instance (casting a concrete
	implementation's address into a shared vtable slot type) and
	_emit_operand's own FunctionRef branch (spelling a bare function
	reference's cast expression), from whichever (ret, params) tuple the
	caller already has (_vtable_slot_c_type's own self-prepended shape, or
	_function_pointer_c_type's plain one below). '''
	params_str = ', '.join( params ) if params else 'void'
	return f'{ret} (*)( {params_str} )'

def _function_pointer_c_type( fn_type: CallableType ) -> tuple[str,list[str]]:
	''' (return type spelling, param type spellings) for fn_type's own
	signature - the same shape _vtable_slot_c_type already builds for a
	vtable slot's function-pointer field, for a different reason
	(dispatching through a concrete implementation's own address rather
	than an arbitrary Ptr[Callable] value's contents). Shared here rather
	than reimplemented a third time by FunctionRef's own cast expression
	(see _emit_operand) and every Ptr[Callable[...]] declarator (see
	_declarator below). '''
	ret = 'void' if _returns_void_in_c( fn_type.return_type ) else c_type( fn_type.return_type )
	params = [ c_type( a ) for a in fn_type.arg_types ]
	return ret, params

def _declarator( t: Type|None, name: str ) -> str:
	''' "TYPE NAME" for an ordinary parameter/local-variable declaration -
	except when t is Ptr[Callable[...]], where C's function-pointer syntax
	is the one declarator shape that ISN'T "prefix type, then name": the
	name goes INSIDE the parens (RetType (*name)(ParamTypes)), so plain
	string concatenation of c_type(t) and name can't express it. Scoped to
	parameter/local declarations only (see PLAN_CALLABLE.md) - not struct
	fields (nothing needs that yet). '''
	fn_type = _callable_ptr_type( t )
	if fn_type is None:
		return f'{c_type(t)} {name}'
	ret, params = _function_pointer_c_type( fn_type )
	params_str = ', '.join( params ) if params else 'void'
	return f'{ret} (*{name})( {params_str} )'

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
	if isinstance( t, CType ):
		return t.c_name
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

def _result_tag_data_names( result_spec: Type ) -> tuple[str,str,str,str]:
	''' (tag_field, data_field, ok_member_field, err_member_field) for a
	Result[T,E]-shaped value - Result is a real @union like any other
	(TaggedUnion), so this reads the actual field names Lowering.
	_tagged_union_storage synthesized into its own .names['tag']/['data'],
	the SAME shape emit_tagged_union already reads,
	rather than hardcoding '_tag'/'_payload.ok'/'_payload.err' the way this
	module used to when Result predated @union. The Ok=0/Err=1 tag-VALUE
	convention stays a literal assumption at every call site below (that's
	inherent to what Check-mode arithmetic/OrReturn/Unwrap already mean,
	per ir.py's own opcode semantics - not new hardcoding), only the field
	NAMES are looked up dynamically here. '''
	base = result_spec.base if isinstance( result_spec, Specialization ) else result_spec
	assert isinstance( base, TaggedUnion ), f'{base!r}: Result must be a real @union'
	tag_attr = base.names.get( 'tag' )
	data_attr = base.names.get( 'data' )
	assert isinstance( tag_attr, Variable ) and isinstance( data_attr, Variable ), \
		f'{base.qualname}: _tagged_union_storage has not run yet - no real storage shape to read'
	return _field_name( tag_attr.stem ), _field_name( data_attr.stem ), _field_name( 'v_Ok' ), _field_name( 'v_Err' )

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
	if not attrs:
		# MSVC (and pedantic C) reject empty structs/unions:
		lines.append( '\tchar dummy;' )
	else:
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

def _returns_void_in_c( return_type: Type|None ) -> bool:
	''' NoneType/NoReturn are value-less in C - a real void, not a
	MetalpyNone struct with nothing in it. Shared by _function_prototype
	(the C prototype itself) and _emit_instruction's own ir.Return handling
	(the actual `return ...;` statements inside the body) - they have to
	agree, or a generic method monomorphized with T=NoneType (e.g.
	Result[None,E].unwrap(), whose body is `return self.data.v_Ok`) gets a
	void-declared C function whose own body still tries to `return` a real
	(if structurally empty) operand, which every C compiler rejects. '''
	return return_type is None or ( isinstance( return_type, Scalar ) and return_type.stem in ( 'NoneType', 'NoReturn' ))

def _self_c_type( cls: ClassLike ) -> str:
	# an @interface CStruct is never a plain value type (see
	# PLAN_SUBCLASSING_VTABLES_COM.md) - self is Ptr[T], for EVERY method,
	# virtual or not (consistency - see lowering.py's own self_param
	# construction, which types self identically). c_type(cls) itself stays
	# unchanged/untouched (a bare CStruct is still a plain value everywhere
	# else - e.g. Ptr[T]'s own c_type spelling needs the UNqualified value
	# spelling for its inner type) - the pointer-ness is added explicitly
	# here, only for self's own spelling.
	base = c_type( cls )
	return f'{base}*' if isinstance( cls, CStruct ) and cls.is_interface else base

def _function_prototype( function: Function ) -> str:
	if function.is_destructor:
		name = mangle_qualname( function.qualname )
		return f'static void {name}( void* __obj )'
	params: list[str] = []
	if _has_self( function ):
		params.append( f'{_self_c_type(function.cls)} self' )
	for p in ( function.parameters or [] ):
		params.append( _declarator( p.type, _c_local_name( p.stem )))
	params_str = ', '.join( params ) if params else 'void'
	if _is_entry_point( function ):
		return f'int main( {params_str} )'
	noreturn = '_Noreturn ' if _is_noreturn( function.return_type ) else ''
	# NoneType/NoReturn are value-less in C — return void, not MetalpyNone
	if _returns_void_in_c( function.return_type ):
		ret = 'void'
	else:
		ret = c_type( function.return_type )
	# @extern(lib, symbol) functions are declared with their raw C symbol
	# name, not the metalpy-qualified name — the linker resolves the raw
	# symbol from the foreign library, not from this translation unit
	if function.extern_lib is not None:
		name = function.extern_symbol
		# generic @extern monomorphized to different pointer types
		# share the same C symbol — use void* for all object pointers
		# to avoid conflicting prototypes for the same symbol
		if isinstance( ret, str ) and ret.endswith( '*' ):
			ret = 'void*' if not ret.startswith( 'const' ) else 'const void*'
	else:
		name = mangle_qualname( function.qualname )
	return f'{noreturn}{ret} {name}( {params_str} )'

def _emit_operand( op: ir.Operand ) -> str:
	if isinstance( op, ir.Const ):
		return _emit_const( op )
	if isinstance( op, ir.Temp ):
		return f't{op.id}'
	if isinstance( op, ir.FunctionRef ):
		# a bare function reference used as a value - see PLAN_CALLABLE.md.
		# Same cast-expression shape emit_interface_vtable_instance already
		# builds to point a vtable slot at a concrete implementation
		# (a plain pointer-to-pointer function-pointer cast, safe and free
		# at runtime, no wrapper needed) - reused via _function_pointer_c_type
		fn_type = _callable_ptr_type( op.type )
		assert fn_type is not None, f'_emit_operand: FunctionRef with non-Ptr[Callable] type {op.type!r}'
		ret, params = _function_pointer_c_type( fn_type )
		return f'({_fn_ptr_cast_type(ret, params)}){mangle_qualname(op.fn.qualname)}'
	if isinstance( op, Variable ):
		# locals (parameters, stack locals) use bare stem; globals
		# need the full mangled qualname (cross-TU visibility)
		return _c_local_name( op.stem ) if not op.is_global else mangle_qualname( op.qualname )
	raise NotImplementedError( f'_emit_operand: unsupported operand {op!r}' )

def _emit_const( c: ir.Const ) -> str:
	if isinstance( c.value, bool ):
		return 'true' if c.value else 'false'
	if isinstance( c.value, int ):
		# pointer-typed constants (e.g. Ptr[None] = -1) need a cast
		if isinstance( c.type, Specialization ):
			base = c.type.base
			if isinstance( base, Scalar ) and base.stem in ( 'Ptr', 'ConstPtr' ):
				return f'({c_type(c.type)}){c.value}'
		return str( c.value )
	if c.value is None:
		return '0' # NOTE: we would like to put 'nullptr' or 'NULL' here but its causing issues
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
	'add': '__metalpy_add_overflow', 'sub': '__metalpy_sub_overflow', 'mul': '__metalpy_mul_overflow',
}
_PLAIN_BITWISE_SYMBOL = { ir.BitAnd: '&', ir.BitOr: '|', ir.BitXor: '^', ir.Shr: '>>' }

def _is_pointer_type( t: Type|None ) -> bool:
	# Ptr[T]/ConstPtr[T] - a pointer value, not an integer scalar. +/- on
	# one of these means byte-address arithmetic (RawList's whole element-
	# buffer indexing scheme is built on this), computed via a uintptr_t
	# round-trip rather than handed to C's own pointer arithmetic - the
	# latter is undefined on void*/const void* outside GCC/Clang's own
	# extension (this compiler also targets MSVC), and scaled by
	# sizeof(*ptr) even where it IS defined, which is never what this
	# language's own explicit, manually-sized element buffers want
	return isinstance( t, Specialization ) and isinstance( t.base, Scalar ) and t.base.stem in ( 'Ptr', 'ConstPtr' )

def _emit_wrap_arith( dest: str, left: ir.Operand, right: ir.Operand, symbol: str, dest_type: Type ) -> list[str]:
	# C's signed overflow is UB, not wraparound - cast to the same-width
	# unsigned type (well-defined wraparound there), compute, cast back
	# (implementation-defined but universally two's-complement in practice,
	# same posture as everywhere else this compiler already leans on real-
	# world compiler behavior over strict-ISO-C portability)
	ctype = c_type( dest_type )
	l, r = _emit_operand( left ), _emit_operand( right )
	if _is_pointer_type( dest_type ):
		return [ f'\t{dest} = ({ctype})((uintptr_t)({l}) {symbol} (uintptr_t)({r}));' ]
	stem = dest_type.stem if isinstance( dest_type, Scalar ) else None
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
	tag_f, data_f, ok_f, _err_f = _result_tag_data_names( result_spec )
	if _is_pointer_type( ok_type ):
		if kind not in ( 'add', 'sub' ):
			raise NotImplementedError( f'checked pointer {kind} is not supported' )
		return [
			'\t{',
			'\t\tuintptr_t __tmp;',
			f'\t\tbool __overflow = {builtin}( (uintptr_t)({l}), (uintptr_t)({r}), &__tmp );',
			'\t\tif ( __overflow ) {',
			f'\t\t\t{dest}.{tag_f} = 1;',
			'\t\t} else {',
			f'\t\t\t{dest}.{tag_f} = 0;',
			f'\t\t\t{dest}.{data_f}.{ok_f} = ({ctype})__tmp;',
			'\t\t}',
			'\t}',
		]
	return [
		'\t{',
		f'\t\t{ctype} __tmp;',
		f'\t\tbool __overflow = {builtin}( {l}, {r}, &__tmp );',
		'\t\tif ( __overflow ) {',
		f'\t\t\t{dest}.{tag_f} = 1;',
		'\t\t} else {',
		f'\t\t\t{dest}.{tag_f} = 0;',
		f'\t\t\t{dest}.{data_f}.{ok_f} = __tmp;',
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
		tag_f, data_f, ok_f, _err_f = _result_tag_data_names( instr.dest.type )
		return [
			'\t{',
			f'\t\t{ctype} __tmp = ({l}) << ({r});',
			f'\t\tbool __overflow = ( __tmp >> ({r}) ) != ({l});',
			'\t\tif ( __overflow ) {',
			f'\t\t\t{dest}.{tag_f} = 1;',
			'\t\t} else {',
			f'\t\t\t{dest}.{tag_f} = 0;',
			f'\t\t\t{dest}.{data_f}.{ok_f} = __tmp;',
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
			f'\t\tbool __overflow = __metalpy_sub_overflow( ({ctype})0, ({operand}), &__tmp );',
			f'\t\t{dest} = __overflow ? {max_c} : __tmp;', # negating MIN is the only way signed negation overflows, and it always overflows toward MAX
			'\t}',
		]
	# check
	dest = f't{instr.dest.id}'
	tag_f, data_f, ok_f, _err_f = _result_tag_data_names( instr.dest.type )
	return [
		'\t{',
		f'\t\t{ctype} __tmp;',
		f'\t\tbool __overflow = __metalpy_sub_overflow( ({ctype})0, ({operand}), &__tmp );',
		'\t\tif ( __overflow ) {',
		f'\t\t\t{dest}.{tag_f} = 1;',
		'\t\t} else {',
		f'\t\t\t{dest}.{tag_f} = 0;',
		f'\t\t\t{dest}.{data_f}.{ok_f} = __tmp;',
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
	# Ptr[Callable[...]] is the one type c_type() can't spell alone (C's
	# function-pointer syntax needs the (*)(...) shape, not a plain prefix
	# string) - a closure's own fn field cast back to its real trampoline
	# signature (lowering.py's _try_lower_closure_call) is the first real
	# cast-TO-a-function-pointer this emitter ever needed; every existing
	# Ptr[Callable[...]] value came from ir.FunctionRef directly before,
	# never through an explicit cast
	fn_type = _callable_ptr_type( target_type )
	if fn_type is not None:
		ret, params = _function_pointer_c_type( fn_type )
		ctype = _fn_ptr_cast_type( ret, params )
	else:
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
	# promoted to __metalpy_wideint for the range comparison - every scalar width
	# this compiler supports OTHER than i128/u128 themselves (excluded
	# just above) fits inside __metalpy_wideint without loss, sidestepping the
	# usual signed/unsigned-pairing headache a same-width comparison
	# would otherwise need (source and target can differ in both width
	# AND signedness - e.g. i32 -> u8, or u64 -> i16)
	if mode == 'saturate':
		dest = _emit_operand( instr.dest )
		return [
			'\t{',
			f'\t\t__metalpy_wideint __wide = (__metalpy_wideint)({operand});',
			f'\t\t{dest} = ( __wide < (__metalpy_wideint)({min_c}) ) ? {min_c} : ( __wide > (__metalpy_wideint)({max_c}) ) ? {max_c} : ({ctype})({operand});',
			'\t}',
		]
	# check
	dest = f't{instr.dest.id}'
	tag_f, data_f, ok_f, _err_f = _result_tag_data_names( instr.dest.type )
	return [
		'\t{',
		f'\t\t__metalpy_wideint __wide = (__metalpy_wideint)({operand});',
		f'\t\tbool __overflow = ( __wide < (__metalpy_wideint)({min_c}) ) || ( __wide > (__metalpy_wideint)({max_c}) );',
		'\t\tif ( __overflow ) {',
		f'\t\t\t{dest}.{tag_f} = 1;',
		'\t\t} else {',
		f'\t\t\t{dest}.{tag_f} = 0;',
		f'\t\t\t{dest}.{data_f}.{ok_f} = ({ctype})({operand});',
		'\t\t}',
		'\t}',
	]

# --- comparisons / control flow / calls / member access -----------------------

_CMP_SYMBOLS = {
	ir.CmpOp.EQ: '==', ir.CmpOp.NE: '!=', ir.CmpOp.LT: '<', ir.CmpOp.LE: '<=', ir.CmpOp.GT: '>', ir.CmpOp.GE: '>=',
}

def _member_access_operator( obj_type: Type|None ) -> str:
	# a concrete generic RCClass instantiation (Box[i32]) is a Specialization,
	# not an RCClass instance itself - unwrap first, or a monomorphized
	# generic method's own `self` (already typed as a Specialization) would
	# wrongly emit `.` (value access) instead of `->` (every RCClass
	# instance is always accessed through a pointer in C). Ptr[T]/ConstPtr[T]
	# is ALSO always a Specialization (base=the Ptr/ConstPtr intrinsic
	# scalar, args=[T]) - the dot-operator on a raw pointer means arrow too
	# (see lowering.py's _attr_lookup/_attr_lookup_callable, which redirect
	# NAME lookup to the pointee but leave the operand itself, and its
	# type, as the pointer - this is where that pointer-ness actually
	# becomes `->` in the emitted C).
	base = obj_type.base if isinstance( obj_type, Specialization ) else obj_type
	if isinstance( base, RCClass ):
		return '->'
	if isinstance( base, Scalar ) and base.stem in ( 'Ptr', 'ConstPtr' ):
		return '->'
	return '.'

def _emit_self_operand( receiver: ir.Operand, target: Function ) -> str:
	''' an @interface CStruct's self is ALWAYS Ptr[T] (see lowering.py's
	self_param construction) - receiver is therefore always already a
	pointer here, never a plain value, for both an ordinary inherited call
	and a @virtual dispatch call. Either way this is a plain pointer-to-
	pointer reinterpretation - safe and free at runtime, no address-of/
	value-narrowing dance needed (that machinery only ever existed to work
	around self being passed BY VALUE, which is no longer how @interface
	CStruct methods work at all).

	The cast TARGET differs between the two cases though:
	- an ordinary (non-virtual) inherited call always resolves to ONE
	  concrete Function (target.cls, whichever ancestor declared it) - a
	  plain narrowing cast to that ancestor's own type.
	- a @virtual call's target vtable SLOT lives in a Vtbl struct typed
	  uniformly to its owner (see _vtable_slot_c_type) - but which Vtbl
	  struct is "the" one depends on the RECEIVER's own concrete type, not
	  on target.cls (whichever ancestor happened to statically resolve the
	  call - could be any level, an override further down, or the
	  original declaration). receiver's own $vtable field is already typed
	  to receiver's own vtbl_owner() (see emit_cstruct) - the self
	  argument has to match THAT, not target.cls's own root or owner
	  (CStruct.vtbl_owner's own docstring: two different concrete classes
	  sharing an interface can have different vtbl_owner()s even for the
	  SAME inherited slot, e.g. calling an IFoo-declared method through a
	  BarImpl receiver two levels below IFoo needs Ptr[IBar], not
	  Ptr[IFoo], because BarImpl's own $vtable is typed IBarVtbl*). '''
	receiver_text = _emit_operand( receiver )
	target_cls = target.cls
	if isinstance( target_cls, CStruct ) and target_cls.is_interface:
		if target.is_virtual:
			receiver_pointee = receiver.type.args[0] if isinstance( receiver.type, Specialization ) else None
			assert isinstance( receiver_pointee, CStruct ) # every @interface CStruct method's self/receiver is Ptr[T] - see lowering.py's self_param construction
			cast_target = receiver_pointee.vtbl_owner()
		else:
			cast_target = target_cls
		return f'({_self_c_type(cast_target)})({receiver_text})'
	return receiver_text

def _emit_call_args( instr: ir.Call ) -> list[str]:
	params = instr.target.parameters or []
	values: list[str] = []
	if instr.receiver is not None:
		values.append( _emit_self_operand( instr.receiver, instr.target ))
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
	if function.is_destructor:
		# cast void* __obj to the concrete struct Foo* self
		self_type = c_type( function.parameters[0].type ) if function.parameters else 'void*'
		lines.append( f'\t{self_type} self = ({self_type})__obj;' )
	# locals (x: i32 = 1) have no DeclareTemp-style IR instruction of their
	# own - lowering.py just emits a plain Assign against a Variable that
	# was never separately "declared". C needs a declaration before use, so
	# this module synthesizes one inline at each local's first assignment
	# (see the ir.Assign branch below) - pre-seed with every parameter
	# (already declared via the signature itself, must never be
	# re-declared) so only genuine first-time locals trigger it
	declared: set[str] = set()
	if _has_self( function ):
		declared.add( 'self' )
	for p in ( function.parameters or [] ):
		declared.add( _c_local_name( p.stem ))
	# __return_value (ir.OrJump's own return_slot - see Lowering.
	# _return_value_var) is referenced two ways neither of which goes
	# through the ordinary "declare on first Assign" mechanism below: a
	# field-store ({slot}.tag = 1;` in _emit_or_jump, never an ir.Assign),
	# and build_epilogue_ladder()'s own unconditional "fall off the end"
	# `return __return_value;` - emitted whenever this function EVER
	# pushed an epilogue entry at all, even along a path this compiler
	# never proves unreachable (e.g. a while-True loop whose every real
	# exit is an explicit return/break - CFG-wise indistinguishable here
	# from one that might fall through). Either reference can be the
	# FIRST (only) mention of __return_value in the whole function, with
	# no ir.Assign to it anywhere - confirmed by two INDEPENDENT real
	# repros, not just reasoning, and neither one implies the other: an
	# OrJump can reference return_slot without its OWN target label still
	# being live by the time build_epilogue_ladder() walks the (by-then-
	# popped) stack, and a dead "fall off the end" epilogue can exist with
	# no OrJump anywhere in the function at all (a while-True loop whose
	# only exits are return/break, e.g. str.split() below). So this checks
	# for EITHER shape, not just one: any OrJump with a return_slot, OR
	# any Label at all whose name starts with 'epilogue' (cfg.py's
	# Epilogue.name is always 'epilogue_N', from _new_label('epilogue') -
	# a reliable proxy for "an Epilogue entry existed, so build_epilogue_
	# ladder() ran" without replicating cfg.py's own push/cancel
	# bookkeeping here). `declared` then makes any actual ir.Assign to it
	# (from the function's own `return <expr>`) just an ordinary
	# re-assignment, not a second declaration
	needs_return_value = any(
		( isinstance( instr, ir.OrJump ) and instr.return_slot is not None )
		or ( isinstance( instr, ir.Label ) and 'epilogue' in instr.name ) # _new_label('epilogue') -> '__epilogue_N__', not a bare prefix
		for instr in fn.instructions
	)
	if needs_return_value and not _returns_void_in_c( function.return_type ):
		name = _c_local_name( '__return_value' )
		lines.append( f'\t{_declarator( function.return_type, name )};' )
		declared.add( name )
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
		if function is not None and _returns_void_in_c( function.return_type ):
			# instr.value is a real IR operand (not Python None - that's the
			# branch above), but the function's own C return type is void -
			# a generic method monomorphized with T=NoneType still has a
			# real `return <T-typed-expr>;` in its own body (e.g. Result
			# [None,E].unwrap()'s `return self.data.v_Ok`), which would
			# otherwise emit `return t2;` from a function declared void -
			# see _returns_void_in_c's own comment
			return [ '\treturn;' ]
		return [ f'\treturn {_emit_operand(instr.value)};' ]

	if isinstance( instr, ir.DeclareTemp ):
		return [ f'\t{_declarator( instr.temp.type, f"t{instr.temp.id}" )};' ]
	if isinstance( instr, ir.DeleteTemp ):
		return [] # C block scoping already handles temp lifetime - nothing to emit
	if isinstance( instr, ir.Assign ):
		src = _emit_operand( instr.src )
		if isinstance( instr.dest, Variable ) and not instr.dest.is_global:
			name = _c_local_name( instr.dest.stem )
			if name not in declared:
				declared.add( name )
				return [ f'\t{_declarator( instr.dest.type, name )} = {src};' ]
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
		tag_f, data_f, ok_f, _err_f = _result_tag_data_names( instr.dest.type )
		return [
			f'\tif ( ({r}) == 0 ) {{',
			f'\t\t{dest}.{tag_f} = 1;',
			'\t} else {',
			f'\t\t{dest}.{tag_f} = 0;',
			f'\t\t{dest}.{data_f}.{ok_f} = ({l}) {symbol} ({r});',
			'\t}',
		]

	if type( instr ) in _PLAIN_BITWISE_SYMBOL:
		dest = _emit_operand( instr.dest )
		symbol = _PLAIN_BITWISE_SYMBOL[type(instr)]
		return [ f'\t{dest} = ({_emit_operand(instr.left)}) {symbol} ({_emit_operand(instr.right)});' ]

	if isinstance( instr, ir.Invert ):
		return [ f'\t{_emit_operand(instr.dest)} = ~({_emit_operand(instr.operand)});' ]
	if isinstance( instr, ir.Not ):
		return [ f'\t{_emit_operand(instr.dest)} = !({_emit_operand(instr.operand)});' ]
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
		# @extern functions are called by their raw C symbol name
		if instr.target.extern_lib is not None:
			target_name = instr.target.extern_symbol
		else:
			target_name = mangle_qualname( instr.target.qualname )
		# destructor calls sys.free(self) — self is struct Foo*, free takes
		# void*; C needs an explicit cast since the two are different types
		arg_texts = list( _emit_call_args( instr ))
		if function is not None and function.is_destructor and target_name == mangle_qualname( 'sys.free' ):
			for i, a in enumerate( arg_texts ):
				arg_texts[i] = f'(void*)({a})'
		if instr.target.is_virtual:
			# vtable dispatch, not a direct call - _emit_self_operand already
			# put the (possibly re-cast) receiver pointer at arg_texts[0];
			# the receiver's own $vtable field (typed const RootVtbl* - see
			# emit_cstruct) already IS the right pointer type, no cast
			# needed at this call site. receiver is always Ptr[T] now (see
			# lowering.py's self_param construction), so $vtable access is
			# always -> (_member_access_operator), never . directly
			assert instr.receiver is not None # is_virtual only ever set on real instance methods - see discovery.py's _parse_function
			slot_name = _field_name( instr.target.stem )
			vtable_op = _member_access_operator( instr.receiver.type )
			call_expr = f'({_emit_operand(instr.receiver)}){vtable_op}$vtable->{slot_name}( {", ".join(arg_texts)} )'
		else:
			has_args = bool( arg_texts )
			call_expr = f'{target_name}( {", ".join(arg_texts)} )' if has_args else f'{target_name}()'
		if instr.dest is not None:
			return [ f'\t{_emit_operand(instr.dest)} = {call_expr};' ]
		return [ f'\t{call_expr};' ]

	if isinstance( instr, ir.CallIndirect ):
		# calling THROUGH a Ptr[Callable[...]]-typed value - see
		# PLAN_CALLABLE.md. No explicit deref needed (C calls a function
		# pointer directly), same as a vtable slot call above doesn't need
		# one either
		arg_texts = [ _emit_operand( a ) for a in instr.args ]
		call_expr = f'({_emit_operand(instr.target)})( {", ".join(arg_texts)} )'
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
		# release_object reads the destructor off the object's own header
		# now (see ObjectHeader's own comment) - nothing to compute here
		return [ f'\trelease_object( &({_emit_operand(instr.value)})->$header );' ]
	if isinstance( instr, ir.DecrefDynamic ):
		# instr.value is Ptr[None] (type-erased) - $header is always the
		# FIRST member of every RCClass struct (emit_rcclass's own field-
		# flattening), so a pointer to the start of any RCClass instance is
		# always validly reinterpretable as ObjectHeader* directly
		return [ f'\trelease_object( (ObjectHeader*){_emit_operand(instr.value)} );' ]
	if isinstance( instr, ir.RefCount ):
		return [ f'\t{_emit_operand(instr.dest)} = ({_emit_operand(instr.value)})->$header.ref_count;' ]

	if isinstance( instr, ir.AtomicLoad ):
		# cast at the point of use rather than requiring the pointee's own
		# declared storage to be _Atomic-qualified - lowering.py's
		# _atomic_pointee_type already guarantees instr.ptr.type is Ptr[T]
		# with T a plain scalar
		pointee_c_type = c_type( instr.ptr.type.args[0] )
		return [ f'\t{_emit_operand(instr.dest)} = atomic_load((_Atomic({pointee_c_type})*){_emit_operand(instr.ptr)});' ]
	if isinstance( instr, ir.AtomicStore ):
		pointee_c_type = c_type( instr.ptr.type.args[0] )
		return [ f'\tatomic_store((_Atomic({pointee_c_type})*){_emit_operand(instr.ptr)}, {_emit_operand(instr.value)});' ]
	if isinstance( instr, ir.AtomicRMW ):
		pointee_c_type = c_type( instr.ptr.type.args[0] )
		fn_name = { ir.AtomicRMWOp.ADD: 'atomic_fetch_add', ir.AtomicRMWOp.SUB: 'atomic_fetch_sub', ir.AtomicRMWOp.EXCHANGE: 'atomic_exchange' }[instr.op]
		return [ f'\t{_emit_operand(instr.dest)} = {fn_name}((_Atomic({pointee_c_type})*){_emit_operand(instr.ptr)}, {_emit_operand(instr.value)});' ]
	if isinstance( instr, ir.AtomicCompareExchange ):
		# expected stays a plain T* (not _Atomic-cast) - that's what C11's
		# own atomic_compare_exchange_strong signature expects for its
		# second parameter, only the first (the atomic object itself) gets
		# the _Atomic(T)* cast
		pointee_c_type = c_type( instr.ptr.type.args[0] )
		return [
			f'\t{_emit_operand(instr.dest)} = atomic_compare_exchange_strong('
			f'(_Atomic({pointee_c_type})*){_emit_operand(instr.ptr)}, {_emit_operand(instr.expected)}, {_emit_operand(instr.desired)});'
		]

	if isinstance( instr, ir.Allocate ):
		if isinstance( instr.cls, RCClass ):
			# routed through sys.alloc[cls] - the SAME allocation path
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
			# instr.dest.type, NOT instr.cls: instr.cls is always the
			# ABSTRACT class for a generic RCClass construction (Box[i32](...)
			# - see _lower_allocate_fields's own comment), but sys.alloc
			# needs the CONCRETE specialization Box[i32], matching exactly
			# what _schedule_rcclass_construction scheduled.
			alloc_name = mangle_qualname( f'sys.alloc[{instr.dest.type.qualname}]' )
			dest = _emit_operand( instr.dest )
			lines = [ f'\t{dest} = {alloc_name}( 1 );' ]
			# freshly allocated = owned by dest immediately (lowering.py
			# never emits an Incref for the temp an Allocate itself produces)
			# - starting the header at 0 would underflow the very first
			# paired Decref
			lines.append( f'\t({dest})->$header.ref_count = 1;' )
			# set once, here - every release_object call reads it back off
			# the header from here on (see ObjectHeader's own comment). The
			# destructor's own real signature (static void NAME(void*
			# __obj), see _function_prototype's is_destructor branch)
			# already matches ObjectHeader.destructor's declared type
			# exactly, no cast needed
			lines.append( f'\t({dest})->$header.destructor = {_rcclass_destructor_name(instr.dest.type)};' )
			for name, value in instr.fields.items():
				lines.append( f'\t({dest})->{_field_name(name)} = {_emit_operand(value)};' )
			return lines
		if isinstance( instr.cls, CStruct ) and instr.cls.is_interface:
			# an @interface CStruct is never a plain value type (see
			# PLAN_SUBCLASSING_VTABLES_COM.md) - heap-allocated via
			# sys.alloc[cls] same as RCClass above, but NO ObjectHeader/
			# refcount init (no automatic RC for a COM object - AddRef/
			# Release are the user's own virtual methods). Every instance
			# wires $vtable to its OWN class's static instance (never a
			# base's - each concrete class's own vtable instance points at
			# ITS OWN slot implementations, see
			# emit_interface_vtable_instance) - lowering.py's construction-
			# time check already guarantees this class's vtable is fully
			# fulfilled before an Allocate for it is ever emitted, so the
			# symbol referenced here is always real
			alloc_name = mangle_qualname( f'sys.alloc[{instr.cls.qualname}]' )
			dest = _emit_operand( instr.dest )
			lines = [ f'\t{dest} = {alloc_name}( 1 );' ]
			lines.append( f'\t({dest})->$vtable = &{mangle_type(instr.cls)}$$vtable;' )
			for name, value in instr.fields.items():
				lines.append( f'\t({dest})->{_field_name(name)} = {_emit_operand(value)};' )
			return lines
		# plain CStruct/CUnion - stack value construction, no header/no
		# heap allocation at all (see the grounding facts in the plan) - a
		# C11 designated-initializer compound literal covers both (a union
		# with more than one field given would be a real error, but
		# nothing here re-validates that - discovery/lowering already did)
		ctype = c_type( instr.dest.type )
		dest = _emit_operand( instr.dest )
		field_init_strs = [ f'.{_field_name(name)} = {_emit_operand(value)}' for name, value in instr.fields.items() ]
		if not field_init_strs:
			return [ f'\t{dest} = ({ctype}){{0}};' ] # empty {} isn't valid standard C11
		return [ f'\t{dest} = ({ctype}){{ {", ".join(field_init_strs)} }};' ]

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
		tag_f, data_f, ok_f, _err_f = _result_tag_data_names( instr.value.type )
		return [
			f'\tif ( ({value}).{tag_f} == 1 ) {{',
			f'\t\t{panic_name}( {_emit_operand(instr.errmsg)} );',
			'\t}',
			f'\t{_emit_operand(instr.dest)} = ({value}).{data_f}.{ok_f};',
		]
	if isinstance( instr, ir.UnwrapOr ):
		value = _emit_operand( instr.value )
		tag_f, data_f, ok_f, _err_f = _result_tag_data_names( instr.value.type )
		return [ f'\t{_emit_operand(instr.dest)} = ( ({value}).{tag_f} == 1 ) ? {_emit_operand(instr.default)} : ({value}).{data_f}.{ok_f};' ]

	raise NotImplementedError( f'_emit_instruction: unsupported instruction {instr!r} (later-phase work)' )

def _emit_or_return( instr: ir.OrReturn, function: Function ) -> list[str]:
	# OrReturn's own IR semantics ARE the branch (see ir.py's docstring:
	# "Err -> return Result::Err(...); Ok -> dest = payload") - this one
	# instruction expands to real conditional C here, not a pre-branched IR
	# sequence. A plain struct-copy return isn't legal C: the enclosing
	# function's own Result[FnT,E] and `value`'s Result[ArithT,E] are
	# DIFFERENT C struct tags even though E (and therefore the whole error
	# payload's byte layout) is identical - only the err member (itself
	# always exactly type E on both sides) is copied across, not the whole
	# struct.
	value = _emit_operand( instr.value )
	dest = _emit_operand( instr.dest )
	return_type = function.return_type
	ret_ctype = c_type( return_type )
	tag_f, data_f, ok_f, err_f = _result_tag_data_names( instr.value.type )
	return [
		f'\tif ( ({value}).{tag_f} == 1 ) {{',
		f'\t\t{ret_ctype} __err;',
		f'\t\t__err.{tag_f} = 1;',
		f'\t\t__err.{data_f}.{err_f} = ({value}).{data_f}.{err_f};',
		'\t\treturn __err;',
		'\t}',
		f'\t{dest} = ({value}).{data_f}.{ok_f};',
	]

def _emit_or_jump( instr: ir.OrJump ) -> list[str]:
	value = _emit_operand( instr.value )
	dest = _emit_operand( instr.dest )
	tag_f, data_f, ok_f, err_f = _result_tag_data_names( instr.value.type )
	lines = [ f'\tif ( ({value}).{tag_f} == 1 ) {{' ]
	if instr.return_slot is not None:
		slot = _emit_operand( instr.return_slot )
		lines.append( f'\t\t{slot}.{tag_f} = 1;' )
		lines.append( f'\t\t{slot}.{data_f}.{err_f} = ({value}).{data_f}.{err_f};' )
	lines.append( f'\t\tgoto {_c_label(instr.target)};' )
	lines.append( '\t}' )
	lines.append( f'\t{dest} = ({value}).{data_f}.{ok_f};' )
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
# __byte_size (str) and __data/__len (bytes) shape - qualname-detection is
# specific to str/bytes only, unrelated to how Result's own field names are
# now derived dynamically (_result_tag_data_names) rather than hardcoded.
_STRING_LITERAL_RCCLASS_QUALNAMES = { 'builtins.str', 'builtins.bytes' }
_STRING_LITERAL_FIELDS = {
	'builtins.str': ( '__data', '__byte_size', True ), # True: __byte_size includes a trailing NUL (lib/builtins's own str.__byte_size comment)
	'builtins.bytes': ( '__data', '__len', False ),
}

def _c_string_literal( data: bytes ) -> str:
	''' convert raw bytes to a C string literal with proper escapes.
	printable ASCII is emitted as-is; non-printable characters (including
	NUL, backslash, double-quote, control chars, and high bytes) are
	escaped as C escape sequences - octal (\\ooo, always exactly 3 digits),
	not hex: C's \\x escape has NO length limit and keeps consuming hex
	digit CHARACTERS (0-9a-fA-F) for as long as they appear next in the
	literal, so \\x9f immediately followed by the literal printable byte
	'e' (itself just emitted raw, below) silently becomes the single
	escape \\x9fe (0xf9e, out of uint8_t's range - "hex escape sequence
	out of range" from a real non-ASCII string like "straße", or worse,
	an in-range-but-wrong byte value with NO compile error at all for a
	different byte/character combination). Octal escapes don't have this
	problem - the C standard caps them at exactly 3 octal digits
	regardless of what follows, so zero-padding to 3 digits always is
	unambiguous no matter what the next byte is. '''
	parts: list[str] = []
	for b in data:
		if b == 0:
			parts.append( chr(92) + '000' )
		elif b == 34: # double-quote
			parts.append( chr(92) + chr(34) )
		elif b == 92: # backslash
			parts.append( chr(92) + chr(92) )
		elif b == 10: # newline
			parts.append( chr(92) + 'n' )
		elif b == 13: # carriage return
			parts.append( chr(92) + 'r' )
		elif b == 9: # tab
			parts.append( chr(92) + 't' )
		elif 32 <= b <= 126: # printable ASCII
			parts.append( chr( b ))
		else:
			parts.append( chr(92) + f'{b:03o}' )
	return chr(34) + ''.join( parts ) + chr(34)

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
	struct_name = mangle_qualname( qualname )
	return [
		f'static const uint8_t {data_name}[] = {_c_string_literal(data_bytes)};',
		f'static struct {struct_name} {name} = {{',
		f'\t.$header = {{ .ref_count = METALPY_IMMORTAL_REFCOUNT }},',
		f'\t.{_field_name(data_field)} = {data_name},',
		f'\t.{_field_name(len_field)} = {len(data_bytes)},',
		'};',
	]

def _emit_string_literals( compiler: Compiler ) -> list[str]:
	# a program-wide collection pass, since C requires each static object
	# defined exactly once - walks every function's AND every global
	# variable's instructions looking for a str/bytes-valued ir.Const,
	# deduplicating by (qualname, value) (the same pair _string_literal_
	# name derives its name from, so two occurrences of the identical
	# literal anywhere in the program share one static definition).
	# compiler.globals is scanned too, not just compiler.functions - a
	# module-level `X: bytes = <a str/bytes-valued expression>` lowers
	# its own initializer into a LoweredGlobal's own .instructions (see
	# lower_global), completely separate from every ordinary function
	# body; missing this list here left a global's own literal reference
	# dangling (compiles the reference, e.g. `&__literal_...`, but never
	# actually emits the `static const ... __literal_... = ...;` it
	# points at - "use of undeclared identifier" from the C compiler) -
	# found by compiler.fetch_unicode_table()'s own real use (a global
	# bytes constant), not a synthetic case
	seen: set[tuple[str,str|bytes]] = set()
	parts: list[str] = []
	def scan( instructions: list[ir.Instruction] ) -> None:
		for instr in instructions:
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
	for lf in compiler.functions:
		scan( lf.instructions )
	for lg in compiler.globals:
		scan( lg.instructions )
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

def _interface_vtbl_name( cls: CStruct ) -> str:
	# a class's own $vtable field is typed to its vtbl_owner()'s Vtbl type,
	# not necessarily its own (see CStruct.vtbl_owner's own docstring) -
	# only the nearest new-slot-introducing class at or above `cls` gets a
	# real Vtbl type of its own; every class below it that adds nothing
	# new just reuses that same type unchanged (never a per-subclass
	# FooImplVtbl for an ordinary implementation)
	return f'{mangle_type(cls.vtbl_owner())}Vtbl'

def _vtable_slot_c_type( owner: CStruct, slot: Function ) -> tuple[str,list[str]]:
	''' the function-pointer type for one vtable slot in `owner`'s own
	Vtbl struct - self is Ptr[owner] UNIFORMLY for every slot in that one
	struct (including slots owner merely inherited from its own base -
	real COM vtable structs work the same way: IFooVtbl types ALL of its
	slots, including IUnknown's inherited 3, as taking IFoo*, not a mix of
	IUnknown*/IFoo* per slot - see _self_c_type, lowering.py's own
	self_param construction, for the identical "self is Ptr[T]" rule
	every @interface CStruct method already follows). A vtable slot has to
	be ONE C type shared by every override sharing that slot (a function
	pointer field can only ever hold one type), so self can't be typed
	per-override the way an ordinary method's self is - the static vtable
	instance itself casts each concrete implementation's own function
	pointer (self: Ptr[ConcreteClass]) to this shared slot type (see
	emit_interface_vtable_instance) - a plain pointer-to-pointer
	reinterpretation, safe and free at runtime, no wrapper function
	needed. '''
	if slot.resolve is not None:
		slot.resolve()
	ret = 'void' if _returns_void_in_c( slot.return_type ) else c_type( slot.return_type )
	params = [ f'{_self_c_type(owner)} self' ]
	for p in ( slot.parameters or [] ):
		params.append( _declarator( p.type, _c_local_name( p.stem )))
	return ret, params

def emit_interface_vtbl_struct( owner: CStruct ) -> str:
	# the FULL vtable struct body for a class that IS its own vtbl_owner()
	# (see CStruct.vtbl_owner) - one function-pointer field per slot
	# (owner.virtual_slots(), declaration order - see CStruct.virtual_slots'
	# own docstring: every new-slot-introducing ancestor's own slots,
	# root-first, up to and including owner itself - e.g. IUnknown's own 3
	# first, then IFoo's own new ones, if owner is IFoo). Function-pointer
	# fields only ever reference OTHER structs by pointer, never by value,
	# so this needs no forward-declared tags of its own - C's implicit
	# "first pointer mention forward-declares an incomplete struct" rule
	# already covers self's own `const OwnerStruct*` (see
	# PLAN_SUBCLASSING_VTABLES_COM.md's own worked example, which has no
	# separate forward-declare step for this reason either).
	name = _interface_vtbl_name( owner )
	slots = owner.virtual_slots()
	lines = [ f'typedef struct {name} {{' ]
	if not slots:
		lines.append( '\tchar dummy;' ) # MSVC (and pedantic C) reject empty structs - same convention as _struct_or_union_body
	for slot in slots:
		ret, params = _vtable_slot_c_type( owner, slot )
		lines.append( f'\t{ret} (*{_field_name(slot.stem)})( {", ".join(params)} );' )
	lines.append( f'}} {name};' )
	return '\n'.join( lines )

def _interface_fulfilled_slot_impls( cls: CStruct ) -> list[Function]|None:
	''' the ordered per-slot implementing Function for a CONCRETE @interface
	CStruct (nearest override in cls's own chain, or the root's own
	declaration if never overridden) - None if ANY slot is unfulfilled (a
	stub body) anywhere in the chain, meaning cls is a "pure interface"
	role that never gets a static vtable instance at all (matches
	compiler.py's _schedule_interface_vtable_impls and lowering.py's
	construction-time check - all three have to agree on exactly which
	classes are "complete", so this recomputes the identical walk rather
	than trusting a cache that could drift out of sync). '''
	impls: list[Function] = []
	for slot in cls.virtual_slots():
		impl = cls.chain_lookup( slot.stem )
		if not isinstance( impl, Function ):
			return None
		if impl.resolve is not None:
			impl.resolve()
		if is_stub_body( impl.node.body ):
			return None
		impls.append( impl )
	return impls

def emit_interface_vtable_instance( cls: CStruct ) -> str|None:
	''' the static const vtable instance for a fully-fulfilled @interface
	CStruct - each slot is wired DIRECTLY to its real implementing
	function via a function-pointer cast (self: Ptr[ConcreteClass]
	reinterpreted as self: Ptr[root], the shared slot type - see
	_vtable_slot_c_type). No wrapper/trampoline function needed - now that
	self is uniformly Ptr[T] for every @interface method (see
	lowering.py's self_param construction), a plain pointer-to-pointer
	function-pointer cast is all dispatch ever needed; the earlier
	trampoline-based approach only existed to work around self being
	passed by value. None (no instance built) if cls's vtable isn't fully
	fulfilled - see _interface_fulfilled_slot_impls. '''
	slot_impls = _interface_fulfilled_slot_impls( cls )
	if slot_impls is None:
		return None
	owner = cls.vtbl_owner()
	field_inits: list[str] = []
	for slot, impl in zip( cls.virtual_slots(), slot_impls ):
		ret, params = _vtable_slot_c_type( owner, slot )
		field_inits.append( f'.{_field_name(slot.stem)} = ({_fn_ptr_cast_type(ret, params)}){mangle_qualname(impl.qualname)}' )
	vtbl_type = _interface_vtbl_name( cls )
	instance_name = f'{mangle_type(cls)}$$vtable'
	if field_inits:
		return f'static const {vtbl_type} {instance_name} = {{ {", ".join(field_inits)} }};'
	return f'static const {vtbl_type} {instance_name} = {{0}};'

def emit_cstruct( cls: CStruct ) -> str:
	attrs: list[tuple[str,Type]]
	if cls.is_interface:
		# $vtable is the literal first member (COM's one hard ABI
		# requirement) - base-chain flattening mirrors emit_rcclass's own
		# walk, except $vtable itself is a SINGLE inherited field (typed
		# to cls.vtbl_owner()'s own Vtbl type - see _interface_vtbl_name/
		# CStruct.vtbl_owner) rather than each base contributing its own
		# vtable pointer
		chain: list[CStruct] = []
		node: CStruct|None = cls
		while node is not None:
			chain.append( node )
			node = node.base
		own_attrs: list[tuple[str,Type]] = []
		for base_cls in reversed( chain ):
			own_attrs.extend( ( attr.stem, attr.type ) for attr in base_cls.attributes )
		vtbl_name = _interface_vtbl_name( cls )
		lines = [ f'struct {mangle_type(cls)} {{', f'\tconst {vtbl_name}* $vtable;' ]
		for field_name, field_type in own_attrs:
			lines.append( f'\t{_field_type_spelling(field_type)} {_field_name(field_name)};' )
		lines.append( '};' )
		return '\n'.join( lines )
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


def emit_c( compiler: Compiler, *, no_crt: bool = False ) -> str:
	''' single C11 translation unit - see the plan's "three-pass emission
	order" decision. Linking is out of scope (C_EMITTER.md); the whole
	program is already collected into one Compiler instance, so there's no
	reason to split output across files. '''
	parts: list[str] = [ PROLOGUE ]

	# collect #include requirements from all modules whose symbols are
	# compiled into this translation unit
	for h in sorted( compiler.disco.required_headers ):
		parts.append( f'#include <{h}>' )
	parts.append( '' )

	parts.append( _NONE_PLACEHOLDER_TYPEDEF )

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
	# every CStruct/CUnion/TaggedUnion gets the SAME forward tag treatment
	# as RCClass above, for the SAME reason (confirmed by a real clang
	# error, not just theory: "incompatible function pointer types",
	# "will not be visible outside of this function"). This used to be
	# is_interface-only (an @interface CStruct's self/Vtbl slot signatures
	# reference it AS A POINTER before its own full body exists, the
	# original trigger) - but ANY value type can be referenced as a
	# pointer inside an @interface CStruct's OWN vtable slot signature
	# too (e.g. QueryInterface's `riid: ConstPtr[GUID]`, GUID a perfectly
	# ordinary, non-interface @cstruct) - confirmed by a real repro: GUID
	# hit this exact trap the moment it appeared as a Vtbl slot parameter
	# type, well before its own body is emitted later in
	# _emit_value_type_bodies. Forward-tagging every value type
	# unconditionally, not just ones already known to need it, is cheap
	# and always safe (an unused tag is harmless) - simpler and more
	# robust than trying to enumerate exactly which types get referenced
	# by pointer somewhere in a vtable slot signature.
	for cls in compiler.cstructs:
		if not cls.type_params:
			parts.append( f'struct {mangle_type(cls)};' )
	for cls in compiler.cunions:
		if not cls.type_params:
			parts.append( f'union {mangle_type(cls)};' )
	for cls in compiler.tagged_unions:
		if not cls.type_params:
			parts.append( f'struct {mangle_type(cls)};' )
	# @interface CStructs' $vtable field points at the Vtbl struct type
	# (see emit_cstruct/_interface_vtbl_name) - its FULL body (one function-
	# pointer field per slot) is emitted here, early, same "before any
	# prototype/body can reference it" reasoning as the RCClass tags just
	# above. Only each interface's own vtbl_owner() gets a Vtbl type of its
	# own (every class below it that adds nothing new reuses that same
	# type unchanged - see CStruct.vtbl_owner) - dict used as an
	# insertion-ordered dedup set, same convention as elsewhere in this
	# module.
	vtbl_owners: dict[str,CStruct] = {}
	for cls in compiler.cstructs:
		if cls.is_interface and not cls.type_params:
			vtbl_owners[ _interface_vtbl_name( cls ) ] = cls.vtbl_owner()
	for owner in vtbl_owners.values():
		parts.append( emit_interface_vtbl_struct( owner ))
	for cls in compiler.cenums: # CEnum is never generic - no type_params field exists on it at all
		parts.append( emit_cenum( cls ))
	parts.extend( _emit_value_type_bodies( compiler ))
	for lf in compiler.functions:
		# skip @extern prototypes when the header that declares them is
		# already included via compiler.require_header
		fn = lf.function
		if fn.extern_lib is not None and fn.extern_header is not None and fn.extern_header in compiler.disco.required_headers:
			continue
		parts.append( emit_function( lf, prototype_only = True ))

	# pass 2: full RCClass struct bodies (every other tag already exists)

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
	# static vtable instances (+ their own trampolines) - only need every
	# other function's PROTOTYPE (pass 1), not its body, so this can sit
	# anywhere in pass 3; placed before the real function bodies since
	# nothing in a function body depends on a vtable instance's own address
	# existing any earlier than "somewhere in this translation unit"
	for cls in compiler.cstructs:
		if cls.is_interface and not cls.type_params:
			instance_src = emit_interface_vtable_instance( cls )
			if instance_src is not None:
				parts.append( instance_src )
	for lf in compiler.functions:
		# @extern functions have no body (only a ; declaration in pass 1)
		if lf.function.extern_lib is None:
			src = emit_function( lf )
			# Windows: prepend __metalpy_init() to main() so Unicode output
			# is configured before any metalpy code runs
			if _is_entry_point( lf.function ) and compiler.disco.active_target['os'] == 'windows':
				src = src.replace( '{\n', '{\n\t__metalpy_init();\n', 1 )
			parts.append( src )

	# custom entry point when CRT is not linked - the linker expects
	# mainCRTStartup as the /ENTRY, so we provide a thin stub that calls
	# __metalpy_init() then main() and exits cleanly via the process itself
	if no_crt:
		parts.append(
			'#ifdef _WIN32\n'
			'void __stdcall ExitProcess( unsigned int );\n'
			'void mainCRTStartup( void ) {\n'
			'\t__metalpy_init();\n'
			'\tint __result = main();\n'
			'\tExitProcess( (unsigned int)__result );\n'
			'}\n'
			'#endif'
		)
	return '\n\n'.join( part for part in parts if part ) + '\n'
