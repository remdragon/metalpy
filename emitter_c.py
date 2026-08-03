# stdlib imports:
from dataclasses import dataclass

# local imports:
import ir
from compiler import Compiler, LoweredFunction, LoweredGlobal
from mpy_types import (
	ClassLike, CEnum, CStruct, CUnion, Copy, Function, Move, Parameter,
	RCClass, Scalar, Specialization, TaggedUnion, Type, Variable,
)

# stage 3: turns a fully-lowered Compiler's output into C11 source. Pure
# translation - by the time emit_c() runs, every real dependency is already
# discovered/scheduled/lowered (see ARCHITECTURE.md's stage split); this
# module does no further discovery of its own.

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

def emit_function( fn: LoweredFunction, *, prototype_only: bool = False ) -> str:
	function = fn.function
	proto = _function_prototype( function )
	if prototype_only or function.extern_lib is not None:
		return proto + ';'
	lines = [ proto + ' {' ]
	is_entry = _is_entry_point( function )
	for instr in fn.instructions:
		lines.extend( _emit_instruction( instr, is_entry_point = is_entry ))
	lines.append( '}' )
	return '\n'.join( lines )

def _emit_instruction( instr: ir.Instruction, *, is_entry_point: bool = False ) -> list[str]:
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
			return [ '\treturn 0;' ] if is_entry_point else [ '\treturn;' ]
		return [ f'\treturn {_emit_operand(instr.value)};' ]
	raise NotImplementedError( f'_emit_instruction: unsupported instruction {instr!r} (later-phase work)' )

# --- classes / globals (stubs - filled in by later phases) -------------------

def emit_rcclass( cls: RCClass ) -> str:
	raise NotImplementedError( 'emit_rcclass: Phase 3 work' )

def emit_cstruct( cls: CStruct ) -> str:
	raise NotImplementedError( 'emit_cstruct: Phase 2 work' )

def emit_cunion( cls: CUnion ) -> str:
	raise NotImplementedError( 'emit_cunion: Phase 2 work' )

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

	# pass 1: forward declarations (opaque RCClass tags, full CEnum/CStruct/
	# CUnion/TaggedUnion bodies in dependency order, function prototypes)
	for cls in compiler.cenums:
		parts.append( emit_cenum( cls ))
	for cls in compiler.cstructs:
		parts.append( emit_cstruct( cls ))
	for cls in compiler.cunions:
		parts.append( emit_cunion( cls ))
	for union in compiler.tagged_unions:
		parts.append( emit_tagged_union( union ))
	for lf in compiler.functions:
		parts.append( emit_function( lf, prototype_only = True ))

	# pass 2: full RCClass struct bodies (every other tag already exists)
	for cls in compiler.rcclasses:
		parts.append( emit_rcclass( cls ))

	# pass 3: global definitions, then full function bodies
	for g in compiler.globals:
		parts.append( emit_global( g ))
	for lf in compiler.functions:
		parts.append( emit_function( lf ))

	return '\n\n'.join( part for part in parts if part ) + '\n'
