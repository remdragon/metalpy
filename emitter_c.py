# stdlib imports:
import dataclasses
import hashlib
import math
import re

# local imports:
import ir
from compiler import Compiler, LoweredFunction, LoweredGlobal
from discovery import is_stub_body
from mpy_types import (
	CallableType, CEnum, ClassLike, CStruct, CType, CUnion, FixedArrayType, Function, Overload,
	RCClass, Scalar, Specialization, TaggedUnion, Type, TupleType, Variable,
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
# PROLOGUE used to be one monolithic always-emitted blob (verbatim from
# C_EMITTER.md's own proposal). Split into pieces here so emit_c() can leave
# out the ones a given program doesn't need (retain_object/release_object/
# the format_f64+parse_f64 pair) - avoids -Wunused-function on every build
# that doesn't happen to retain/release an RCClass or format/parse a float
# (i.e. most trivial programs). PROLOGUE itself (the full concatenation)
# stays around unchanged for callers that want the whole thing regardless
# (see emitter_c_test.py's own release_object test).
_PROLOGUE_HEADER_FIXED = '''\
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdatomic.h>

#define METALPY_IMMORTAL_REFCOUNT INT32_MAX

// marks a static const that's legitimately unreferenced in SOME compiled
// programs but not others (a CEnum member no compiled code happens to name,
// a class's own vtable instance when nothing constructs it this time round)
// - unlike retain_object/__metalpy_format_f64/.../the format_f64+parse_f64
// pair (see emit_c), these aren't worth conditionally emitting: a CEnum
// member reference always constant-folds away before it ever reaches this
// module (lowering.py's own _expr_Attribute), so "referenced" can only ever
// be judged by matching against ir.Allocate call sites one at a time - real
// dead-code elimination, not a cheap "does the IR use this instruction
// kind anywhere" check. MSVC doesn't warn on an unused static/static const
// at all (confirmed directly - no /W4 diagnostic for it), so this only
// needs to matter to GCC/Clang.
#if defined(_MSC_VER) && !defined(__clang__)
#define __metalpy_maybe_unused
#else
#define __metalpy_maybe_unused __attribute__((unused))
#endif
'''

# the one universal, ALWAYS-leading member of every RCClass's own vtable
# type, whatever else that type goes on to add for its own @virtual
# methods (see emit_rcclass_vtbl_struct's own "destroy-prefixed" comment) -
# this is what lets ObjectHeader's own $vtable field stay typed to this one
# shared, minimal shape and still safely read through ANY concrete class's
# own (possibly larger) real vtable, the same "shared leading layout,
# narrower read through a base-typed lens" trick CStruct's own per-level
# Vtbl types already rely on (see _interface_vtbl_name). slot 0 MUST stay
# `destroy` - every concrete-class vtable-struct generator depends on that
# (emit_rcclass_vtbl_struct, the interface equivalent). debug builds append
# `type_name` AFTER destroy (one pointer per CLASS, not per object - each
# concrete class already gets its own vtable instance) so dump_live_objects
# can label a live object's group without needing a per-object copy.
def _object_vtbl_prologue() -> str:
	type_name_field = '\tconst char* type_name; // debug builds only - see dump_live_objects\n' if _target_debug else ''
	return (
		'typedef struct {\n'
		'\tvoid (*destroy)( void* );\n'
		+ type_name_field +
		'} __metalpy_ObjectVtbl;\n'
	)

_PROLOGUE_HEADER_TAIL = '''\
typedef struct {
	_Atomic int32_t ref_count;
	// set once at construction (see emit_c's ir.Allocate codegen), read
	// here on every release - adjacent to ref_count in the same cache
	// line the atomic decrement below already touches, not a separate
	// fetch. Was a bare `void (*destructor)(void*)` function pointer
	// before RCClass @virtual dispatch existed - unified into a real
	// (if often minimal) vtable pointer instead of keeping destructor
	// dispatch and @virtual dispatch as two separate mechanisms; slot 0
	// of EVERY concrete class's own vtable type is always `destroy`
	// (see __metalpy_ObjectVtbl above), so this field's own declared
	// type never needs to change no matter how many @virtual methods a
	// given class goes on to add. A non-virtual class (still the common
	// case) costs nothing extra for this - it gets a plain
	// __metalpy_ObjectVtbl instance, no synthesized type of its own.
	const __metalpy_ObjectVtbl* vtable;
'''

# PLAN_THREAD_SAFE_SHARED_STATE.md Part B: one lock per object (B.1 - not
# one per field), embedded directly in ObjectHeader so every RCClass's own
# leading $header member carries it for free, no per-class opt-in needed.
# Shipped unconditionally (no whole-program "does this program spawn a
# Thread" gate, unlike the Cost-mitigations section's own aspirational #1 -
# Part A itself was ALSO shipped without that switch, scoped down instead
# by only ever protecting a global that's genuinely reassigned; Part B has
# no equivalent narrower target - every RC object gets one). Same platform
# asymmetry as A.3's per-global lock: a bare `void*` on Windows (SRWLOCK's
# own all-zero state is a valid unlocked lock, so sys.alloc's own
# poison-in-debug/uninitialized-in-release memory just needs an explicit
# zero write at construction, not a real init call - see ir.Allocate's own
# codegen); a real `pthread_mutex_t` on POSIX, needing a genuine
# pthread_mutex_init() call at EVERY construction site (a real per-object
# cost the "revisit the lighter primitive later" decision accepts for now).
_PROLOGUE_HEADER_FIELD_WINDOWS = '\tvoid* lock;\n'
_PROLOGUE_HEADER_FIELD_PTHREAD = '\tpthread_mutex_t lock;\n'

# debug-only alloc-site tracking (dump_live_objects, lib/sys.py): a single
# global intrusive doubly-linked list threaded through every live RC object
# (and, separately, every live sys.alloc[T] raw buffer - see _alloc's own
# debug-mode header in lib/sys.py) so a program can walk everything still
# live without any OS heap-walk API (none is portable across MSVC/clang/gcc).
# No CRT calls (insert-at-head/unlink only) - safe under --no-crt. Debug-only:
# costs nothing in a release build, which never emits this block at all.
#
# Both lists are process-global, mutated from whatever thread happens to
# construct/release an RC object or alloc[T]/free a raw buffer - genuinely
# concurrent in any multi-threaded program (lib/threading.py), so track/
# untrack need a REAL lock around the list-splice, not just the "debug info,
# doesn't need to be exact" posture the rest of this feature gets away with.
# Confirmed necessary by a real regression, not just caution: an earlier,
# unlocked version of this code caused emitter_c_test.py's own 8-thread
# concurrent-dict-insert test to hang/timeout under --debug (multiple
# threads splicing the SAME intrusive list at once corrupts it - a lost
# update or a cycle - which then wedges __metalpy_debug_untrack's own
# linear-scan free path, since a corrupted list can spin forever on a
# pointer that never turns up, or an infinite loop through a self-
# referential node). Reuses the exact same lock PRIMITIVE Part A's per-
# global lock does (SRWLOCK on Windows, pthread_mutex_t on POSIX) - a
# single dedicated lock shared by BOTH lists (RC + raw), not two separate
# ones: this is diagnostic bookkeeping, not a hot path worth finer-grained
# locking for.
def _prologue_debug_list() -> str:
	if _target_uses_pthread_lock():
		# pthread_mutex_t is already in scope (pthread.h is force-included
		# by _object_header_prologue() unconditionally on every POSIX
		# target, regardless of debug mode - see its own comment).
		# PTHREAD_MUTEX_INITIALIZER needs no separate init call, unlike
		# ObjectHeader's own per-object lock field (which has no equivalent
		# static-initializer-at-declaration option since it's a per-object,
		# not per-program, storage location).
		lock_decl = 'static pthread_mutex_t __metalpy_debug_lock = PTHREAD_MUTEX_INITIALIZER;\n'
		lock_acquire = '\tpthread_mutex_lock( &__metalpy_debug_lock );\n'
		lock_release = '\tpthread_mutex_unlock( &__metalpy_debug_lock );\n'
	else:
		# SRWLOCK's own all-zero state is already a valid unlocked lock (Part
		# A/B's own established fact, reused verbatim here) - matches
		# _global_lock_acquire/_global_lock_release's exact call shape and
		# the exact same hand-declared struct tag/prototypes (repeated,
		# COMPATIBLE declarations of the same two symbols are legal C - see
		# _PROLOGUE_GLOBAL_LOCK_WINDOWS's own comment), so this coexists
		# fine whether or not Part A/B's own lock machinery is ALSO active
		# in this same translation unit.
		lock_decl = (
			f'struct {_SRWLOCK_STRUCT_NAME};\n'
			f'void AcquireSRWLockExclusive( struct {_SRWLOCK_STRUCT_NAME}* SRWLock );\n'
			f'void ReleaseSRWLockExclusive( struct {_SRWLOCK_STRUCT_NAME}* SRWLock );\n'
			'static void* __metalpy_debug_lock = 0;\n'
		)
		lock_acquire = f'\tAcquireSRWLockExclusive( (struct {_SRWLOCK_STRUCT_NAME}*)&__metalpy_debug_lock );\n'
		lock_release = f'\tReleaseSRWLockExclusive( (struct {_SRWLOCK_STRUCT_NAME}*)&__metalpy_debug_lock );\n'
	return f'''\
typedef struct __metalpy_debug_link {{
	struct __metalpy_debug_link* prev;
	struct __metalpy_debug_link* next;
}} __metalpy_debug_link;

// two separate lists, not one shared list carrying a runtime type tag: RC
// objects (ObjectHeader-prefixed) and raw sys.alloc[T] buffers (their own
// side-table tracking node, see __metalpy_debug_raw_entry below) have
// different header shapes, so keeping them apart avoids any reinterpret-
// cast-by-tag dance when dump_live_objects walks either one.
static __metalpy_maybe_unused __metalpy_debug_link __metalpy_debug_list_head = {{ &__metalpy_debug_list_head, &__metalpy_debug_list_head }};
static __metalpy_maybe_unused __metalpy_debug_link __metalpy_debug_raw_list_head = {{ &__metalpy_debug_raw_list_head, &__metalpy_debug_raw_list_head }};

{lock_decl}
static inline __metalpy_maybe_unused void __metalpy_debug_lock_acquire( void ) {{
{lock_acquire}}}

static inline __metalpy_maybe_unused void __metalpy_debug_lock_release( void ) {{
{lock_release}}}

// _locked variants assume the caller already holds __metalpy_debug_lock -
// needed by callers whose own critical section spans more than one list
// operation (__metalpy_debug_raw_untrack's scan-then-splice,
// __metalpy_dump_live_objects's whole walk) - taking the lock separately
// for each individual splice there would leave a window between "found it"
// and "removed it" (or between reading two list nodes while walking) for
// another thread to mutate the list first. The plain (self-locking)
// versions below are for the single-splice case (track/RC untrack).
static inline __metalpy_maybe_unused void __metalpy_debug_track_locked( __metalpy_debug_link* head, __metalpy_debug_link* link ) {{
	link->prev = head;
	link->next = head->next;
	head->next->prev = link;
	head->next = link;
}}

static inline __metalpy_maybe_unused void __metalpy_debug_untrack_locked( __metalpy_debug_link* link ) {{
	link->prev->next = link->next;
	link->next->prev = link->prev;
}}

static inline __metalpy_maybe_unused void __metalpy_debug_track( __metalpy_debug_link* head, __metalpy_debug_link* link ) {{
	__metalpy_debug_lock_acquire();
	__metalpy_debug_track_locked( head, link );
	__metalpy_debug_lock_release();
}}

static inline __metalpy_maybe_unused void __metalpy_debug_untrack( __metalpy_debug_link* link ) {{
	__metalpy_debug_lock_acquire();
	__metalpy_debug_untrack_locked( link );
	__metalpy_debug_lock_release();
}}
'''

def _object_header_prologue() -> str:
	''' ObjectHeader's own full text, including Part B's per-object lock
	field - a FUNCTION (not the plain string constant this used to be),
	since the field's own C type depends on _target_os, which is only known
	once emit_c() itself has set it (module-level constants can't see that -
	see _target_os's own docstring on why it's set inside emit_c(), not at
	import time). On POSIX, a genuine `#include <pthread.h>` right here,
	NOT left to emit_c()'s own general required_headers collection loop -
	that loop runs LATER, after this whole prologue (this is the very
	FIRST thing emit_c() assembles) - confirmed as a real bug, not just
	theoretical ordering pedantry: without a real prior declaration in
	scope, `pthread_mutex_t lock;` here silently fell back to C's
	legacy "implicit int" behavior on gcc (no error, just a wrong field
	type - every subsequent `pthread_mutex_init(&obj->$header.lock, ...)`
	call site then failed as "incompatible pointer type", since the field
	was actually a plain int). A repeated `#include <pthread.h>` is always
	safe (real header, standard include guards) even if something else in
	the program ALSO required it independently - never a redundant include
	system's problem to avoid, that's what include guards are for. '''
	field = _PROLOGUE_HEADER_FIELD_PTHREAD if _target_uses_pthread_lock() else _PROLOGUE_HEADER_FIELD_WINDOWS
	header_include = '#include <pthread.h>\n' if _target_uses_pthread_lock() else ''
	debug_list = _prologue_debug_list() if _target_debug else ''
	# alloc_loc/alloc_size/debug_link: debug-only alloc-site tracking
	# (dump_live_objects) - alloc_loc points at a static string literal set
	# once at construction (ir.Allocate codegen, emit_c), which also stamps
	# alloc_size from a plain `sizeof(*dest)` at the call site (the concrete
	# struct type is always known there - simplest available answer to "what
	# does this live object cost", no separate per-class size table needed).
	# debug_link threads this object into the global list _PROLOGUE_DEBUG_LIST
	# declares.
	debug_fields = '\tconst char* alloc_loc;\n\tsize_t alloc_size;\n\t__metalpy_debug_link debug_link;\n' if _target_debug else ''
	return (
		header_include + _PROLOGUE_HEADER_FIXED + debug_list + _object_vtbl_prologue() + _PROLOGUE_HEADER_TAIL
		+ field + debug_fields + '} ObjectHeader;\n'
	)

# debug-only: raw sys.alloc[T] buffer tracking + dump_live_objects itself.
# Comes AFTER _object_header_prologue()'s own text (ObjectHeader must already
# be a complete type - dump_live_objects reads through it via offsetof), so
# this is a separate function emit_c() appends right after that one, not
# folded into _PROLOGUE_DEBUG_LIST above. No CRT calls anywhere in here
# (manual decimal formatting, raw OS write) - safe under --no-crt, and
# deliberately bypasses lib/sys.py's own _Stdout/write_all path (which is
# itself just a thin wrapper over the exact same WriteFile/write(2) primitive
# used here) rather than round-tripping through metalpy-level str/Result
# machinery from inside a hand-written PROLOGUE C function.
def _prologue_debug_ops() -> str:
	if _target_os == 'windows':
		write_decl = 'void* GetStdHandle( uint32_t nStdHandle ); bool WriteFile( void* hFile, const uint8_t* lpBuffer, uint32_t nNumberOfBytesToWrite, uint32_t* lpNumberOfBytesWritten, void* lpOverlapped );'
		write_body = '''\
	void* h = GetStdHandle( (uint32_t)-11 ); // STD_OUTPUT_HANDLE
	if ( h == (void*)(intptr_t)-1 || h == 0 ) return;
	uint32_t written = 0;
	WriteFile( h, (const uint8_t*)s, len, &written, 0 );
'''
		# side-table tracking node allocation - hand-declared HeapAlloc/
		# HeapFree/GetProcessHeap prototypes (same "repeated, compatible
		# extern declaration is legal C" technique the crash handler already
		# uses above), NOT lib/windows/kernel32.py's own @extern bindings:
		# this runs from hand-written PROLOGUE C, which can't reach a
		# metalpy-level import, and must work even in a program that never
		# itself imports windows.kernel32 for anything else.
		# matches windows.kernel32's own @extern-generated signatures
		# EXACTLY (uint8_t*, not void*, for the buffer parameters/return) -
		# a repeated, compatible extern declaration of the same symbol is
		# legal C (same technique the crash handler above already uses),
		# but only if the two declarations really do agree; a program that
		# ALSO imports windows.kernel32 for its own reasons would otherwise
		# see two INCOMPATIBLE declarations of the same symbol (confirmed:
		# a real "conflicting types" error, not just theoretical risk).
		raw_alloc_decl = 'void* GetProcessHeap( void ); void* HeapAlloc( void* hHeap, uint32_t dwFlags, uintptr_t dwBytes ); bool HeapFree( void* hHeap, uint32_t dwFlags, uint8_t* lpMem );'
		raw_alloc_call = '(__metalpy_debug_raw_entry*)HeapAlloc( GetProcessHeap(), 0, sizeof( __metalpy_debug_raw_entry ))'
		raw_free_call = 'HeapFree( GetProcessHeap(), 0, (uint8_t*)e )'
	else:
		write_decl = 'long write( int32_t fd, const void* buf, size_t count );'
		write_body = '\twrite( 1, s, len );\n'
		# POSIX targets always link a real CRT-provided main() in this
		# codebase (see lib/sys.py's own argv comment) - a plain malloc/free
		# call here is never a no-crt hazard the way it would be on Windows
		# matches lib/crt.py's own @extern-generated malloc/free signatures
		# EXACTLY (void* return, uintptr_t param for malloc - confirmed
		# against the real generated C, not assumed: usize maps to
		# uintptr_t, not size_t, in an extern's own parameter position; void*
		# for free's argument, matching libc's real void*-typed signature
		# per that file's own comment) - same "repeated but must stay
		# COMPATIBLE" concern as the Windows branch's HeapAlloc/HeapFree
		# above - confirmed as a REAL conflicting-types error under gcc
		# otherwise (a mismatched hand-declared malloc here vs. crt.py's own
		# extern-generated one, in the same translation unit).
		raw_alloc_decl = 'void* malloc( uintptr_t size ); void free( void* ptr );'
		raw_alloc_call = '(__metalpy_debug_raw_entry*)malloc( sizeof( __metalpy_debug_raw_entry ))'
		raw_free_call = 'free( e )'
	return f'''\
{write_decl}

static void __metalpy_debug_write( const char* s ) {{
	size_t len = 0;
	while ( s[len] ) len++;
{write_body}}}

static void __metalpy_debug_write_udec( size_t v ) {{
	char buf[24];
	int i = 24;
	buf[--i] = 0;
	if ( v == 0 ) buf[--i] = '0';
	while ( v ) {{
		buf[--i] = (char)( '0' + ( v % 10 ));
		v /= 10;
	}}
	__metalpy_debug_write( &buf[i] );
}}

// side-table tracking node for a raw sys.alloc[T] buffer (debug builds only)
// - see lib/sys.py's alloc[T]/free and ir.DebugRawTrack's own comment on why
// this is a SEPARATE node (allocated straight from the OS allocator, never
// through metalpy's own sys.alloc - that would recurse right back into this
// same tracking code) rather than a header prefixed onto the real block:
// sys.alloc[T] must keep returning the EXACT pointer the OS allocator gave
// it, unchanged, since some existing code queries the OS allocator directly
// on that pointer (confirmed by a real regression an earlier prefix-header
// version of this caused - see ir.py's own comment).
typedef struct {{
	__metalpy_debug_link link;
	void* ptr;
	size_t size;
}} __metalpy_debug_raw_entry;

{raw_alloc_decl}

static inline __metalpy_maybe_unused void __metalpy_debug_raw_track( void* ptr, size_t size ) {{
	if ( !ptr ) return;
	__metalpy_debug_raw_entry* e = {raw_alloc_call};
	if ( !e ) return; // tracking is best-effort - must never crash the real allocation path
	e->ptr = ptr;
	e->size = size;
	__metalpy_debug_track( &__metalpy_debug_raw_list_head, &e->link );
}}

static inline __metalpy_maybe_unused void __metalpy_debug_raw_untrack( void* ptr ) {{
	if ( !ptr ) return;
	// the whole scan-then-splice is one critical section, not lock-per-step
	// (see _prologue_debug_list's own comment on why) - find-then-remove
	// under a SEPARATELY acquired lock would leave a window for another
	// thread to mutate the list in between
	__metalpy_debug_lock_acquire();
	__metalpy_debug_link* l = __metalpy_debug_raw_list_head.next;
	while ( l != &__metalpy_debug_raw_list_head ) {{
		__metalpy_debug_raw_entry* e = (__metalpy_debug_raw_entry*)l; // link is this struct's own first member
		if ( e->ptr == ptr ) {{
			__metalpy_debug_untrack_locked( l );
			__metalpy_debug_lock_release();
			{raw_free_call};
			return;
		}}
		l = l->next;
	}}
	__metalpy_debug_lock_release();
}}

// groups RC objects by (vtable, alloc_loc) - both are pointers to static
// storage set once at construction (one literal per Allocate CALL SITE, one
// vtable instance per CLASS), so pointer equality alone already means
// "same class, same source line", no string comparison needed. O(n^2) over
// the live set - a debug/diagnostic tool, not a hot path.
static __metalpy_maybe_unused void __metalpy_dump_live_objects( void ) {{
	// one lock for the WHOLE walk (both lists) - a concurrent track/untrack
	// from another still-running thread must not be allowed to mutate
	// either list mid-walk (see _prologue_debug_list's own comment)
	__metalpy_debug_lock_acquire();
	// the header carries the total live count (not just its own group
	// breakdown below) so a test can assert "(0)" - a real, checkable
	// zero-leaks result - rather than merely tolerating/ignoring whatever
	// this prints, which is what its own presence still had to be treated
	// as before this. A separate, cheap O(n) pass (this diagnostic is
	// already O(n^2) below, one debug/leak-checking dump per program run,
	// not a hot path) - the grouped walk further down still needs its own
	// full traversal regardless, so there's no way to fold this into it
	// without printing the header only after already knowing every group.
	size_t __metalpy_live_rc_total = 0;
	{{
		__metalpy_debug_link* t = __metalpy_debug_list_head.next;
		while ( t != &__metalpy_debug_list_head ) {{ __metalpy_live_rc_total++; t = t->next; }}
	}}
	__metalpy_debug_write( "-- live RC objects (" );
	__metalpy_debug_write_udec( __metalpy_live_rc_total );
	__metalpy_debug_write( ") --\\n" );
	__metalpy_debug_link* l = __metalpy_debug_list_head.next;
	while ( l != &__metalpy_debug_list_head ) {{
		ObjectHeader* h = (ObjectHeader*)( (char*)l - offsetof( ObjectHeader, debug_link ));
		__metalpy_debug_link* p = __metalpy_debug_list_head.next;
		int already_reported = 0;
		while ( p != l ) {{
			ObjectHeader* ph = (ObjectHeader*)( (char*)p - offsetof( ObjectHeader, debug_link ));
			if ( ph->vtable == h->vtable && ph->alloc_loc == h->alloc_loc ) {{ already_reported = 1; break; }}
			p = p->next;
		}}
		if ( !already_reported ) {{
			size_t count = 0, bytes = 0;
			__metalpy_debug_link* q = l;
			while ( q != &__metalpy_debug_list_head ) {{
				ObjectHeader* qh = (ObjectHeader*)( (char*)q - offsetof( ObjectHeader, debug_link ));
				if ( qh->vtable == h->vtable && qh->alloc_loc == h->alloc_loc ) {{ count++; bytes += qh->alloc_size; }}
				q = q->next;
			}}
			__metalpy_debug_write( "  " );
			__metalpy_debug_write( h->vtable && h->vtable->type_name ? h->vtable->type_name : "<unknown type>" );
			__metalpy_debug_write( " @ " );
			__metalpy_debug_write( h->alloc_loc ? h->alloc_loc : "<unknown location>" );
			__metalpy_debug_write( ": count=" );
			__metalpy_debug_write_udec( count );
			__metalpy_debug_write( " bytes=" );
			__metalpy_debug_write_udec( bytes );
			__metalpy_debug_write( "\\n" );
		}}
		l = l->next;
	}}
	// raw sys.alloc[T] buffers: NOT split by (type,site) - see lib/sys.py's
	// alloc[T]/free comment on why per-call-site attribution isn't plumbed
	// through here; still gives a real live-byte total for anything sys.alloc
	// itself allocates (list/dict backing storage, ...).
	{{
		size_t count = 0, bytes = 0;
		__metalpy_debug_link* rl = __metalpy_debug_raw_list_head.next;
		while ( rl != &__metalpy_debug_raw_list_head ) {{
			__metalpy_debug_raw_entry* rh = (__metalpy_debug_raw_entry*)rl; // link is this struct's own first member
			count++; bytes += rh->size;
			rl = rl->next;
		}}
		if ( count ) {{
			__metalpy_debug_write( "-- live raw sys.alloc buffers --\\n  count=" );
			__metalpy_debug_write_udec( count );
			__metalpy_debug_write( " bytes=" );
			__metalpy_debug_write_udec( bytes );
			__metalpy_debug_write( "\\n" );
		}}
	}}
	__metalpy_debug_lock_release();
}}
'''

# only needed where an ir.Incref is actually emitted (see emit_c) - a
# program that only ever gives up references (or never touches an RCClass
# at all) never calls this
_PROLOGUE_RETAIN = '''\
static inline void retain_object( ObjectHeader* obj ) {
	if ( obj && obj->ref_count != METALPY_IMMORTAL_REFCOUNT ) {
		atomic_fetch_add( &obj->ref_count, 1 );
	}
}
'''

# only needed where an ir.Decref/DecrefDynamic is actually emitted (see
# emit_c). A FUNCTION (not a plain string constant), same reason as
# _object_header_prologue: the debug-only untrack call below depends on
# _target_debug, only known once emit_c() has set it.
def _prologue_release() -> str:
	# debug-mode alloc tracking: untrack right when the refcount hits zero,
	# BEFORE vtable->destroy runs (destroy may free the object's own storage)
	untrack = '\t\t\t__metalpy_debug_untrack( &obj->debug_link );\n' if _target_debug else ''
	return (
		'''\
// the destructor was previously an explicit argument, passed as a compile-
// time literal at every call site - redundant with the header's own
// vtable field (set once at construction), which every caller can
// already reach directly. Reading it here uniformly also means every
// release, not just a type-erased one, is safe to call through a
// base-typed reference under RCClass subclassing - without making
// retain_object/the refcount increment/decrement themselves virtual,
// which is the actual hot-path cost that was always the point to avoid
// paying (see the RCClass-subclassing plan's own Design decision 2 - this
// was ORIGINALLY a bare destructor function pointer, unified into a real
// vtable pointer once @virtual dispatch needed one too, rather than
// keeping the two as separate fields/mechanisms)
static inline void release_object( ObjectHeader* obj ) {
	if ( obj && obj->ref_count != METALPY_IMMORTAL_REFCOUNT ) {
		if ( atomic_fetch_sub( &obj->ref_count, 1 ) == 1 ) {
'''
		+ untrack +
		'''\
			if ( obj->vtable && obj->vtable->destroy ) {
				obj->vtable->destroy( obj );
			}
		}
	}
}
'''
	)

_PROLOGUE_ARITH = '''\
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
//
// A real, previously-undiscovered stack buffer overflow lived here: every
// signed/unsigned pair below used to be declared taking ONLY int64_t*/
// uint64_t* (per a stale comment claiming "the emitter widens operands to
// int64_t/uint64_t before calling these" - untrue of the actual call sites,
// which pass &__tmp typed to the REAL narrow destination, e.g. int32_t*),
// while the _Generic dispatch macros below routed EVERY width (i8/i16/i32
// AND i64) to those same 64-bit-only functions. Passing an int32_t* where
// int64_t* is expected is merely a warning in C (C4133), not an error - so
// it compiled, and `*r = a * b;` inside the 64-bit function then wrote a
// full 8 bytes through a pointer to a 4-byte (or narrower) stack local,
// clobbering 4+ bytes of adjacent stack memory on every checked +/-/* on
// anything narrower than i64/u64. Silent in a release build; caught here
// as a genuine /RTC1 STATUS_BREAKPOINT ("__tmp" stack corruption) once a
// debug build actually exercised it - confirmed with cdb: breaking on
// _RTC_StackFailure and reading its own variable-name argument pointed
// straight at "__tmp", and recompiling with warnings visible showed the
// exact int32_t*->int64_t* mismatch at the failing call site.
//
// Fixed with one real function per (width, signedness): the 8/16/32-bit
// versions compute in int64_t/uint64_t (always wide enough to hold the
// exact, non-overflowing result of two <=32-bit operands, add/sub/mul
// alike - only the true 64-bit case needs its own overflow-detection
// technique, kept exactly as it always was) and range-check against the
// real target width's own bounds before narrowing into *r, so *r is only
// ever written its own declared size.
static inline bool __metalpy_sadd_overflow64(int64_t a, int64_t b, int64_t* r) {
	*r = a + b;
	return (a > 0 && b > 0 && *r < 0) || (a < 0 && b < 0 && *r > 0);
}
static inline bool __metalpy_uadd_overflow64(uint64_t a, uint64_t b, uint64_t* r) {
	*r = a + b;
	return *r < a;
}
static inline bool __metalpy_ssub_overflow64(int64_t a, int64_t b, int64_t* r) {
	*r = a - b;
	return (a >= 0 && b < 0 && *r < 0) || (a < 0 && b >= 0 && *r > 0);
}
static inline bool __metalpy_usub_overflow64(uint64_t a, uint64_t b, uint64_t* r) {
	*r = a - b;
	return a < b;
}
static inline bool __metalpy_smul_overflow64(int64_t a, int64_t b, int64_t* r) {
	*r = a * b;
	if (a == 0 || b == 0) return false;
	if (a < 0) { a = -a; b = -b; }
	return a > INT64_MAX / (b < 0 ? -b : b);
}
static inline bool __metalpy_umul_overflow64(uint64_t a, uint64_t b, uint64_t* r) {
	*r = a * b;
	if (a == 0) return false;
	return *r / a != b;
}
static inline bool __metalpy_sadd_overflow32(int32_t a, int32_t b, int32_t* r) {
	int64_t wide = (int64_t)a + (int64_t)b;
	*r = (int32_t)wide;
	return wide < INT32_MIN || wide > INT32_MAX;
}
static inline bool __metalpy_sadd_overflow16(int16_t a, int16_t b, int16_t* r) {
	int64_t wide = (int64_t)a + (int64_t)b;
	*r = (int16_t)wide;
	return wide < INT16_MIN || wide > INT16_MAX;
}
static inline bool __metalpy_sadd_overflow8(int8_t a, int8_t b, int8_t* r) {
	int64_t wide = (int64_t)a + (int64_t)b;
	*r = (int8_t)wide;
	return wide < INT8_MIN || wide > INT8_MAX;
}
static inline bool __metalpy_uadd_overflow32(uint32_t a, uint32_t b, uint32_t* r) {
	uint64_t wide = (uint64_t)a + (uint64_t)b;
	*r = (uint32_t)wide;
	return wide > UINT32_MAX;
}
static inline bool __metalpy_uadd_overflow16(uint16_t a, uint16_t b, uint16_t* r) {
	uint64_t wide = (uint64_t)a + (uint64_t)b;
	*r = (uint16_t)wide;
	return wide > UINT16_MAX;
}
static inline bool __metalpy_uadd_overflow8(uint8_t a, uint8_t b, uint8_t* r) {
	uint64_t wide = (uint64_t)a + (uint64_t)b;
	*r = (uint8_t)wide;
	return wide > UINT8_MAX;
}
static inline bool __metalpy_ssub_overflow32(int32_t a, int32_t b, int32_t* r) {
	int64_t wide = (int64_t)a - (int64_t)b;
	*r = (int32_t)wide;
	return wide < INT32_MIN || wide > INT32_MAX;
}
static inline bool __metalpy_ssub_overflow16(int16_t a, int16_t b, int16_t* r) {
	int64_t wide = (int64_t)a - (int64_t)b;
	*r = (int16_t)wide;
	return wide < INT16_MIN || wide > INT16_MAX;
}
static inline bool __metalpy_ssub_overflow8(int8_t a, int8_t b, int8_t* r) {
	int64_t wide = (int64_t)a - (int64_t)b;
	*r = (int8_t)wide;
	return wide < INT8_MIN || wide > INT8_MAX;
}
static inline bool __metalpy_usub_overflow32(uint32_t a, uint32_t b, uint32_t* r) {
	*r = (uint32_t)(a - b);
	return a < b;
}
static inline bool __metalpy_usub_overflow16(uint16_t a, uint16_t b, uint16_t* r) {
	*r = (uint16_t)(a - b);
	return a < b;
}
static inline bool __metalpy_usub_overflow8(uint8_t a, uint8_t b, uint8_t* r) {
	*r = (uint8_t)(a - b);
	return a < b;
}
static inline bool __metalpy_smul_overflow32(int32_t a, int32_t b, int32_t* r) {
	int64_t wide = (int64_t)a * (int64_t)b;
	*r = (int32_t)wide;
	return wide < INT32_MIN || wide > INT32_MAX;
}
static inline bool __metalpy_smul_overflow16(int16_t a, int16_t b, int16_t* r) {
	int64_t wide = (int64_t)a * (int64_t)b;
	*r = (int16_t)wide;
	return wide < INT16_MIN || wide > INT16_MAX;
}
static inline bool __metalpy_smul_overflow8(int8_t a, int8_t b, int8_t* r) {
	int64_t wide = (int64_t)a * (int64_t)b;
	*r = (int8_t)wide;
	return wide < INT8_MIN || wide > INT8_MAX;
}
static inline bool __metalpy_umul_overflow32(uint32_t a, uint32_t b, uint32_t* r) {
	uint64_t wide = (uint64_t)a * (uint64_t)b;
	*r = (uint32_t)wide;
	return wide > UINT32_MAX;
}
static inline bool __metalpy_umul_overflow16(uint16_t a, uint16_t b, uint16_t* r) {
	uint64_t wide = (uint64_t)a * (uint64_t)b;
	*r = (uint16_t)wide;
	return wide > UINT16_MAX;
}
static inline bool __metalpy_umul_overflow8(uint8_t a, uint8_t b, uint8_t* r) {
	uint64_t wide = (uint64_t)a * (uint64_t)b;
	*r = (uint8_t)wide;
	return wide > UINT8_MAX;
}

// _Generic dispatch: the emitter calls __metalpy_add_overflow(a,b,r)
// where *r is a local of the concrete scalar type. This picks the right
// width-AND-signedness variant based on the type of *r, so *r is always
// written exactly its own declared size (see the real stack-corruption
// bug this replaced, in the comment above).
#define __metalpy_add_overflow(a,b,r) \
	_Generic(*(r), \
		uint64_t: __metalpy_uadd_overflow64, \
		uint32_t: __metalpy_uadd_overflow32, \
		uint16_t: __metalpy_uadd_overflow16, \
		uint8_t:  __metalpy_uadd_overflow8,  \
		int64_t:  __metalpy_sadd_overflow64, \
		int32_t:  __metalpy_sadd_overflow32, \
		int16_t:  __metalpy_sadd_overflow16, \
		int8_t:   __metalpy_sadd_overflow8   \
	)(a,b,r)
#define __metalpy_sub_overflow(a,b,r) \
	_Generic(*(r), \
		uint64_t: __metalpy_usub_overflow64, \
		uint32_t: __metalpy_usub_overflow32, \
		uint16_t: __metalpy_usub_overflow16, \
		uint8_t:  __metalpy_usub_overflow8,  \
		int64_t:  __metalpy_ssub_overflow64, \
		int32_t:  __metalpy_ssub_overflow32, \
		int16_t:  __metalpy_ssub_overflow16, \
		int8_t:   __metalpy_ssub_overflow8   \
	)(a,b,r)
#define __metalpy_mul_overflow(a,b,r) \
	_Generic(*(r), \
		uint64_t: __metalpy_umul_overflow64, \
		uint32_t: __metalpy_umul_overflow32, \
		uint16_t: __metalpy_umul_overflow16, \
		uint8_t:  __metalpy_umul_overflow8,  \
		int64_t:  __metalpy_smul_overflow64, \
		int32_t:  __metalpy_smul_overflow32, \
		int16_t:  __metalpy_smul_overflow16, \
		int8_t:   __metalpy_smul_overflow8   \
	)(a,b,r)
#else
// GCC/Clang: use compiler builtins directly
typedef __int128 __metalpy_wideint;
typedef unsigned __int128 __metalpy_wideuint;
#define __metalpy_add_overflow(a,b,r) __builtin_add_overflow(a,b,r)
#define __metalpy_sub_overflow(a,b,r) __builtin_sub_overflow(a,b,r)
#define __metalpy_mul_overflow(a,b,r) __builtin_mul_overflow(a,b,r)
#endif
// floating-point classification for checked/panic-mode float arithmetic and
// float-involving casts (FAddCheck/.../FloatCastCheck). GCC/Clang expose these
// as builtins (no <math.h> needed, no CRT call). Real MSVC (cl.exe) gets its
// own self-contained, bit-pattern-based implementation instead of <math.h>'s
// isnan/isinf macros: those can lower to a CALL into the CRT's internal
// _dclass/_fdclass classification helper (confirmed: "unresolved external
// symbol _dclass" linking a no_crt build, since this project deliberately
// doesn't link the CRT - see crt.py's own comment on why). Union-based type
// punning is well-defined in C (unlike C++) - no header, no CRT, no function
// call MSVC might not inline needed. INFINITY/NAN (<math.h>, still needed for
// __metalpy_inf[f]/nan[f] below) are themselves compile-time constant
// expressions, not function calls, so they don't share this problem.
#if defined(_MSC_VER) && !defined(__clang__)
#include <math.h>
static inline bool __metalpy_isnan_f32( float x ) {
	union { float f; uint32_t u; } v; v.f = x;
	return ( v.u & 0x7F800000u ) == 0x7F800000u && ( v.u & 0x007FFFFFu ) != 0;
}
static inline bool __metalpy_isnan_f64( double x ) {
	union { double d; uint64_t u; } v; v.d = x;
	return ( v.u & 0x7FF0000000000000ull ) == 0x7FF0000000000000ull && ( v.u & 0x000FFFFFFFFFFFFFull ) != 0;
}
static inline bool __metalpy_isinf_f32( float x ) {
	union { float f; uint32_t u; } v; v.f = x;
	return ( v.u & 0x7FFFFFFFu ) == 0x7F800000u;
}
static inline bool __metalpy_isinf_f64( double x ) {
	union { double d; uint64_t u; } v; v.d = x;
	return ( v.u & 0x7FFFFFFFFFFFFFFFull ) == 0x7FF0000000000000ull;
}
#define __metalpy_isnan(x) _Generic((x), float: __metalpy_isnan_f32, double: __metalpy_isnan_f64)(x)
#define __metalpy_isinf(x) _Generic((x), float: __metalpy_isinf_f32, double: __metalpy_isinf_f64)(x)
#define __metalpy_inff() ((float)INFINITY)
#define __metalpy_inf()  ((double)INFINITY)
#define __metalpy_nanf() ((float)NAN)
#define __metalpy_nan()  ((double)NAN)
#else
#define __metalpy_isnan(x) __builtin_isnan(x)
#define __metalpy_isinf(x) __builtin_isinf(x)
#define __metalpy_inff() __builtin_inff()
#define __metalpy_inf()  __builtin_inf()
#define __metalpy_nanf() __builtin_nanf("")
#define __metalpy_nan()  __builtin_nan("")
#endif
'''

# _PROLOGUE_FLOAT_FORMAT/_PROLOGUE_FLOAT_PARSE - only needed where an
# ir.FormatFloat/ir.ParseFloat is actually emitted (see emit_c), i.e. a
# program that formats a float as text (str(f), f-string float formatting)
# or parses one (float(s)) respectively. Split into two independently-gated
# parts (NOT kept as one combined unit, despite sharing the Windows branch's
# GetModuleHandleA/LoadLibraryA/GetProcAddress declarations and msvcrt
# resolution) so a program using only one direction doesn't pull in a
# genuinely unused static inline function for the other - confirmed via a
# real repro: an f-string-only program (formats, never parses) still showed
# -Wunused-function on __metalpy_parse_f64 when this was one unit. The common
# case (lib/builtins/__float.py's shortest-round-trip repr search) uses both
# anyway, so both parts land together there - this only matters for a
# program that formats-only or parses-only. The 3 extern prototypes are
# duplicated verbatim into BOTH Windows branches rather than factored into a
# shared third part: a repeated, IDENTICAL extern declaration is legal,
# warning-free C on every one of clang/gcc/MSVC, and keeping each part fully
# self-contained is simpler than threading a third always-emitted-if-either-
# part-is dependency through emit_c().
_PROLOGUE_FLOAT_FORMAT = '''\
// backs compiler.format_f64(buf, size, precision, type_char, alt, value)
// (lowering.py's _lower_compiler_format_f64 / ir.FormatFloat) - writes
// value's fixed-precision decimal digits into buf via a dynamically-built
// "%[#].*X" format string (X = type_char, one of 'f'/'F'/'e'/'E'/'g'/'G' -
// see fstring_format_spec.FORMAT_SPEC_TYPE_CHARS; '%' has no printf
// equivalent and is handled entirely in metalpy source instead - see
// lib/builtins/__float.py's _percent_digits; alt is the '#' flag, included
// in the format string when set - real snprintf's own '#' behavior already
// matches Python's semantics for every one of these type chars exactly, so
// it needs no post-processing the way grouping does - see lib/builtins/
// __float.py's _group_integer_part for that), so magnitude only; callers
// split the sign out themselves. Returns the byte count written, or a
// negative value on failure. Always present (like retain_object/
// __metalpy_isnan above) whether or not a given program actually formats a
// float - dead code if unused, same as every other PROLOGUE helper.
//
// Deliberately NOT declared via metalpy's own @extern mechanism: that
// only ever emits a FIXED-arity C prototype (see _function_prototype),
// which is an ABI hazard for a genuinely variadic callee - both the
// Microsoft x64 and SysV x86-64 calling conventions require the CALL
// SITE itself to know it's targeting a variadic function (to duplicate
// float args into the matching integer register / set %al respectively),
// which a fixed-arity declaration never triggers. Real snprintf/_snprintf
// is called here, in hand-written C, with its own true variadic
// prototype, so the real C compiler generates the correct call - metalpy
// itself only ever calls this fixed, ordinary-looking wrapper.
//
// On Windows this calls msvcrt.dll's own exported _snprintf, loaded and
// resolved dynamically (LoadLibraryA/GetProcAddress, both kernel32 - never
// a static `msvcrt.lib` import). msvcrt.dll itself (NOT the redistributable
// Universal CRT/ucrtbase.dll) has shipped as a genuine OS component since
// Windows 2000 - always present, nothing to redistribute - so this reads a
// DLL that's already on the machine rather than linking a CRT the build
// brought with it. This was NOT the first thing tried: ntdll.dll's own
// exported _snprintf (same dynamic-resolution technique, and genuinely
// present per `dumpbin /exports ntdll.dll`) was tried first, on the
// (wrong) assumption that a symbol with the right name and the right
// export table entry would behave the same everywhere - a real functional
// test caught that ntdll's copy silently fails on ANY float conversion
// (confirmed: "%.*f"/"%f"/"%.1f" all just emit a stray "f", 1 byte, no
// digits at all - "%d"/"%s" work fine through it) - consistent with NT's
// kernel-adjacent runtime code traditionally avoiding the FPU altogether.
// msvcrt.dll's own _snprintf was verified correct the same way (real
// compile+link+run, not just symbol presence): "%.1f" of 1.0 -> "1.0",
// "%.3f" of 3.14159 -> "3.142". GetModuleHandleA/GetProcAddress are
// kernel32 exports (emit_c()'s own extern_libs bookkeeping tags this
// 'kernel32', which is either already linked for any real program, or
// already tracked for real via windows._console's/sys.exit's own @extern
// bindings - see compiler.py's Compiler.force_reachable). NOTE: legacy _snprintf
// (unlike C99 snprintf) returns -1 on truncation instead of the would-
// have-been-written length - callers must size buf generously enough
// that truncation never actually happens (a fixed-precision f64 can need
// at most ~309 integer digits + '.' + precision fractional digits + sign
// + NUL).
//
// A SECOND, Windows-only quirk was found while adding 'e'/'E'/'g'/'G':
// legacy msvcrt.dll's exponent is always padded to exactly 3 digits
// ("1.23e+003"), unlike Python/C99 (glibc's real snprintf included - the
// POSIX branch below needs no equivalent fixup), which use the minimum
// digit count with a floor of 2 ("1.23e+03") - confirmed by a real test
// against this system's own msvcrt.dll. __metalpy_fixup_msvcrt_exponent
// strips extra leading zeros from the exponent (down to that 2-digit
// floor) in place, shifting the rest of the buffer left - real f64
// exponents are always <= 3 digits, so there is at most one leading zero
// to strip in practice, but the loop handles more on general principle.
#ifdef _WIN32
static inline void __metalpy_fixup_msvcrt_exponent( char* buf, int* n ) {
	for ( int i = 0; i < *n; i++ ) {
		if ( buf[i] != 'e' && buf[i] != 'E' ) continue;
		int sign_pos = i + 1;
		if ( sign_pos >= *n || ( buf[sign_pos] != '+' && buf[sign_pos] != '-' )) continue;
		int digits_start = sign_pos + 1;
		int digits_end = digits_start;
		while ( digits_end < *n && buf[digits_end] >= '0' && buf[digits_end] <= '9' ) digits_end++;
		int strip = 0;
		while ( ( digits_end - digits_start ) - strip > 2 && buf[digits_start + strip] == '0' ) strip++;
		if ( strip > 0 ) {
			for ( int j = digits_start; j + strip < *n; j++ ) buf[j] = buf[j + strip];
			*n -= strip;
			buf[*n] = 0;
		}
		break; // at most one exponent in a real float conversion
	}
}
// see this module's own _PROLOGUE_FLOAT_FORMAT/_PROLOGUE_FLOAT_PARSE
// comment on why these 3 are duplicated into _PROLOGUE_FLOAT_PARSE too
// rather than factored into a shared part
void* __stdcall GetModuleHandleA( const char* lpModuleName );
void* __stdcall LoadLibraryA( const char* lpLibFileName );
void* __stdcall GetProcAddress( void* hModule, const char* lpProcName );
typedef int ( __cdecl *__metalpy_snprintf_fn )( char*, size_t, const char*, ... );
static inline int __metalpy_format_f64( char* buf, size_t size, int precision, int type_char, int alt, double value ) {
	static __metalpy_snprintf_fn fn = 0;
	if ( !fn ) {
		void* msvcrt = GetModuleHandleA( "msvcrt.dll" );
		if ( !msvcrt ) msvcrt = LoadLibraryA( "msvcrt.dll" );
		fn = msvcrt ? (__metalpy_snprintf_fn)GetProcAddress( msvcrt, "_snprintf" ) : 0;
		if ( !fn ) return -1;
	}
	// legacy msvcrt.dll's _snprintf silently produces NOTHING (n=0, empty
	// buffer) for the uppercase 'F' conversion specifically - confirmed by
	// a real test against this system's own msvcrt.dll ("%.1F" of 1.0 ->
	// empty output), unlike 'E'/'G' (both correctly supported there). This
	// tracks real printf history: %E/%G existed in C89; %F was only added
	// in C99, and legacy msvcrt predates that. 'f'/'F' are otherwise
	// identical for every FINITE value (Python's own docs: 'F' differs
	// from 'f' only in using "INF"/"NAN" instead of "inf"/"nan" - this
	// compiler doesn't yet special-case inf/nan display for either one),
	// so substituting lowercase 'f' here on Windows only costs that one
	// inf/nan-capitalization edge case, not correctness for real numbers.
	if ( type_char == 'F' ) type_char = 'f';
	char fmt[6];
	int fi = 0;
	fmt[fi++] = '%';
	if ( alt ) fmt[fi++] = '#';
	fmt[fi++] = '.';
	fmt[fi++] = '*';
	fmt[fi++] = (char)type_char;
	fmt[fi] = 0;
	int n = fn( buf, size, fmt, precision, value );
	if ( n > 0 ) __metalpy_fixup_msvcrt_exponent( buf, &n );
	return n;
}
#else
#include <stdio.h>
static inline int __metalpy_format_f64( char* buf, size_t size, int precision, int type_char, int alt, double value ) {
	char fmt[6];
	int fi = 0;
	fmt[fi++] = '%';
	if ( alt ) fmt[fi++] = '#';
	fmt[fi++] = '.';
	fmt[fi++] = '*';
	fmt[fi++] = (char)type_char;
	fmt[fi] = 0;
	return snprintf( buf, size, fmt, precision, value );
}
#endif
'''

_PROLOGUE_FLOAT_PARSE = '''\
// backs compiler.parse_f64(buf) - the inverse of compiler.format_f64, needed
// for the shortest-round-trip repr search (lib/builtins/__float.py's
// _f64_repr_digits_raw). msvcrt.dll's own strtod was verified correct
// against this system's own msvcrt.dll (unlike some of its other legacy
// quirks found earlier - _snprintf's own missing 'F'/garbage inf-nan/3-
// digit-exponent issues): 0.1 -> the standard closest-double approximation,
// 5e-324 -> the smallest denormal, the max finite double, all round-tripped
// exactly. strtod is an ordinary (non-variadic) function - no ABI hazard
// like _snprintf has - but resolved the same dynamic way regardless, since
// a plain @extern('c', ...) binding would still wrongly flip the no-crt
// Windows build (same reasoning _PROLOGUE_FLOAT_FORMAT's own
// __metalpy_format_f64 comment documents).
#ifdef _WIN32
// see _PROLOGUE_FLOAT_FORMAT's own identical comment on why these 3 are
// duplicated here rather than factored into a shared part
void* __stdcall GetModuleHandleA( const char* lpModuleName );
void* __stdcall LoadLibraryA( const char* lpLibFileName );
void* __stdcall GetProcAddress( void* hModule, const char* lpProcName );
typedef double ( __cdecl *__metalpy_strtod_fn )( const char*, char** );
static inline double __metalpy_parse_f64( const char* text ) {
	static __metalpy_strtod_fn fn = 0;
	if ( !fn ) {
		void* msvcrt = GetModuleHandleA( "msvcrt.dll" );
		if ( !msvcrt ) msvcrt = LoadLibraryA( "msvcrt.dll" );
		fn = msvcrt ? (__metalpy_strtod_fn)GetProcAddress( msvcrt, "strtod" ) : 0;
		if ( !fn ) return 0.0;
	}
	return fn( text, 0 );
}
#else
#include <stdlib.h>
static inline double __metalpy_parse_f64( const char* text ) {
	return strtod( text, 0 );
}
#endif
'''

# always emitted (unconditionally, like _PROLOGUE_ARITH) - every compiled
# program should report a fault instead of silently vanishing, not just ones
# that happen to reference some other PROLOGUE helper. __metalpy_init()
# installs this before anything else runs (see emit_c's own call to
# __metalpy_install_crash_handler()), so it's in place before global
# initializers, let alone main(), get a chance to run.
#
# Hand-rolled, not routed through sys.panic/_write_stderr_cstr (lib/sys.py):
# a fault handler has to assume the process is in an arbitrary broken state
# (corrupted heap, exhausted stack) - reusing machinery that itself expects a
# healthy runtime isn't trustworthy. Same reasoning as sys.panic's own
# comment on why IT avoids allocation, taken one step further here (avoids
# calling into any metalpy-level code at all).
_PROLOGUE_CRASH_HANDLER = '''\
#ifdef _WIN32
// hand-declared rather than routed through windows.kernel32's own @extern
// bindings (SetUnhandledExceptionFilter/EXCEPTION_POINTERS have no other user
// in this codebase). GetStdHandle/WriteFile/ExitProcess mirror
// windows.kernel32's own @extern-generated signatures exactly (see
// _function_prototype) - a repeated, compatible extern declaration of the
// same symbol is legal C, so this coexists fine whether or not a given
// program separately imports windows.kernel32 for its own reasons.
void* GetStdHandle( uint32_t nStdHandle );
bool WriteFile( void* hFile, const uint8_t* lpBuffer, uint32_t nNumberOfBytesToWrite, uint32_t* lpNumberOfBytesWritten, void* lpOverlapped );
_Noreturn void ExitProcess( uint32_t uExitCode );
// only the fields this handler actually reads - trailing real fields
// (ExceptionInformation[15]) are omitted; safe, since nothing here takes
// sizeof() or reads past NumberParameters, and the OS-owned struct these
// pointers reference is always at least this big.
typedef struct __metalpy_EXCEPTION_RECORD {
	uint32_t ExceptionCode;
	uint32_t ExceptionFlags;
	struct __metalpy_EXCEPTION_RECORD* ExceptionRecord;
	void* ExceptionAddress;
	uint32_t NumberParameters;
} __metalpy_EXCEPTION_RECORD;
typedef struct {
	__metalpy_EXCEPTION_RECORD* ExceptionRecord;
	void* ContextRecord;
} __metalpy_EXCEPTION_POINTERS;
void* SetUnhandledExceptionFilter( int32_t (*lpTopLevelExceptionFilter)( __metalpy_EXCEPTION_POINTERS* ));

// re-entrancy guard - the handler firing again (e.g. a stack-overflow fault
// re-faulting while this handler itself runs on the little extra stack
// Windows grants for exactly that case) exits immediately instead of
// recursing back into WriteFile.
static volatile long __metalpy_in_crash_handler = 0;

static void __metalpy_crash_write( const char* msg ) {
	uint32_t len = 0;
	while ( msg[len] ) len++;
	void* h = GetStdHandle( (uint32_t)-12 ); // STD_ERROR_HANDLE
	if ( h != (void*)(intptr_t)-1 ) {
		uint32_t written = 0;
		WriteFile( h, (const uint8_t*)msg, len, &written, 0 );
	}
}

static void __metalpy_crash_write_hex( uint64_t v ) {
	char buf[] = "0x0000000000000000";
	for ( int i = 17; i >= 2 && v; i-- ) {
		uint32_t nib = (uint32_t)( v & 0xF );
		buf[i] = (char)( nib < 10 ? '0' + nib : 'a' + nib - 10 );
		v >>= 4;
	}
	__metalpy_crash_write( buf );
}

static int32_t __metalpy_crash_filter( __metalpy_EXCEPTION_POINTERS* info ) {
	if ( __metalpy_in_crash_handler ) ExitProcess( 1 );
	__metalpy_in_crash_handler = 1;
	uint32_t code = info && info->ExceptionRecord ? info->ExceptionRecord->ExceptionCode : 0;
	__metalpy_crash_write( "\\nFatal error: unhandled exception " );
	__metalpy_crash_write_hex( code );
	__metalpy_crash_write( " at address " );
	__metalpy_crash_write_hex( (uint64_t)(uintptr_t)( info && info->ExceptionRecord ? info->ExceptionRecord->ExceptionAddress : 0 ));
	__metalpy_crash_write( "\\n" );
	ExitProcess( code ? code : 1 );
}

static void __metalpy_install_crash_handler( void ) {
	SetUnhandledExceptionFilter( __metalpy_crash_filter );
}
#else
// <unistd.h> for the REAL write()/_exit() declarations - safe now that
// lib/crt.py's own write/_exit are split into os='windows'/os=not'windows'
// overloads, with header='unistd.h' only on the POSIX one (see that split's
// own comment in crt.py, and every other unistd.h-shaped extern there,
// for the two real problems this used to hit before the whole file's
// POSIX-facing externs were audited: a single ungated header= reaching a
// Windows build, and this header conflicting with a sibling extern's own
// separately-guessed, incompatible pointee type). Both this file and
// crt.py's own POSIX-branch prototype (skipped whenever unistd.h is
// already required - see emit_c's pass-1 loop) now defer to the one real
// header instead of each hand-declaring their own guess at it.
#include <signal.h>
#include <unistd.h>
static void __metalpy_crash_write( const char* msg ) {
	size_t len = 0;
	while ( msg[len] ) len++;
	ssize_t __metalpy_unused = write( 2, msg, len );
	(void)__metalpy_unused;
}

static const char* __metalpy_signal_name( int sig ) {
	switch ( sig ) {
		case SIGSEGV: return "SIGSEGV (segmentation fault)";
		case SIGABRT: return "SIGABRT (aborted)";
		case SIGFPE:  return "SIGFPE (arithmetic exception)";
		case SIGILL:  return "SIGILL (illegal instruction)";
		case SIGBUS:  return "SIGBUS (bus error)";
		default:      return "unknown signal";
	}
}

// see the Windows branch's own comment on why this needs its own re-entrancy
// guard - sig_atomic_t (not a plain int/bool) is the one type C guarantees
// is safe to read/write from inside a signal handler
static volatile sig_atomic_t __metalpy_in_crash_handler = 0;

static void __metalpy_crash_handler( int sig ) {
	if ( __metalpy_in_crash_handler ) _exit( 128 + sig );
	__metalpy_in_crash_handler = 1;
	__metalpy_crash_write( "\\nFatal error: " );
	__metalpy_crash_write( __metalpy_signal_name( sig ));
	__metalpy_crash_write( "\\n" );
	_exit( 128 + sig );
}

static void __metalpy_install_crash_handler( void ) {
	signal( SIGSEGV, __metalpy_crash_handler );
	signal( SIGABRT, __metalpy_crash_handler );
	signal( SIGFPE, __metalpy_crash_handler );
	signal( SIGILL, __metalpy_crash_handler );
	signal( SIGBUS, __metalpy_crash_handler );
}
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

def _c_local_name( var: Variable ) -> str:
	''' return this local's own C identifier. C keywords get a leading
	underscore. var.needs_uid_suffix - set by lowering.py, only at a
	genuinely fresh declaration that follows an earlier `del` of the same
	stem (see its own docstring) - appends var.uid ('$' can't appear in a
	source identifier, same reasoning as _temp_name) so that Variable
	never collides at the C level with whatever the old, deleted binding
	left behind (see del_reuse_and_emitter_naming_bug). Every other local
	(parameters, an ordinary reassignment reusing the SAME live binding
	across if/elif/else arms, __return_value and other compiler-
	synthesized slots) keeps its bare stem - lowering.py guarantees two
	DIFFERENT Variable objects only ever share a bare, un-suffixed stem
	when they're genuinely meant to (an inferred, non-annotated re-
	assignment reusing an existing binding); a same-scope, no-del
	REdeclaration (a second explicit `x: T = ...` for an already-live x)
	is rejected as a compile error before it ever reaches here - see
	_stmt_AnnAssign's own doc. '''
	base = f'_{var.stem}' if var.stem in _C_KEYWORDS else var.stem
	return f'{base}${var.uid}' if var.needs_uid_suffix else base

def _temp_name( temp_id: int ) -> str:
	''' the C name for a compiler-synthesized ir.Temp, e.g. for id=5, "$t5" -
	the single spot this scheme is defined; every declaration/reference site
	below calls this rather than building the string itself, so they can't
	drift apart. '$'-prefixed for the SAME reason every other purely-
	compiler-internal name in this module is ('$header', mangle_qualname's
	'.'->'$', ...): '$' is not a legal character in a metalpy/Python source
	identifier, so a temp can NEVER collide with a real user local - unlike
	the plain "t5" this used to be, which silently collided with a real
	source local of the exact same name (confirmed by a real repro: `t5:
	i32 = 1` alongside a compiler temp that also happened to be allocated
	id 5 produced a bogus "redefinition"/type-mismatch error in the
	generated C, not a clean "reserved name" diagnostic). Relies on the
	same GCC/Clang/MSVC '$'-in-identifiers extension this module's output
	already depends on throughout. '''
	return f'$t{temp_id}'

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
		# '|' appears when an anonymous union is a type ARGUMENT embedded in a
		# larger generic qualname, e.g. Result[intrinsics.f64, builtins.Zero
		# DivisionError|builtins.FloatingPointError] from checked float division
		# (the union's OWN standalone name never reaches here - mangle_type
		# routes it through the $__u$$... scheme instead). A literal '|' isn't a
		# legal C identifier character; '$or$' is unique ('$' can't appear in a
		# real identifier) and, because every reference to the SAME qualname
		# string - the interned Specialization AND its monomorphized concrete
		# class (which share qualname, see monomorphize_class) - passes through
		# this one function, all mangle to the same name.
		.replace( '|', '$or$' )
	)

# ---------------------------------------------------------------------------
# PLAN_THREAD_SAFE_SHARED_STATE.md Part A - automatic locking for a module-
# level RC-typed global that's genuinely reassigned from inside a function
# body (`global X; X = ...`), closing the same unsynchronized-read-check-
# then-write race lib/datetime.py's localtz()/lib/termcolor.py's _codes()
# used to have to guard by hand with a FastLock (see that plan's own
# "The soundness gap, precisely" section). A global written only by its own
# module-level initializer never needs this - Variable.reassigned_outside_
# init only ever flips True from cfg.py's assign() `dest.is_global` branch,
# which lower_global() (lowering.py) never reaches for a global's own
# initializer (see that flag's own comment).
#
# The lock itself is a bare, paired `void*` global per protected variable -
# not lib/threading.py's FastLock (an RCClass, itself heap-allocated, with
# its OWN inner lock ALSO heap-allocated in __init__ - using one per
# protected global would mean two nested heap allocations happening as
# part of module-global initialization, before the very system this exists
# to protect is itself safe to construct). SRWLOCK is documented by Win32
# as a single opaque, pointer-sized value whose all-zero state IS already a
# valid, unlocked lock (no InitializeSRWLock call exists anywhere in this
# codebase) - a `void*` initialized to 0 is exactly that, no separate init
# # function needed on Windows - Linux's own pthread_mutex_t storage is NOT
# zero-init-safe the same way and needs a real pthread_mutex_init() call;
# see _global_lock_name's own declaration site (in the globals-emission
# loop, emit_c()) for that platform's storage shape and init wiring.
# AcquireSRWLockExclusive/ReleaseSRWLockExclusive take a
# `void**` here (not the real PSRWLOCK type) since SRWLOCK's own real
# layout is just one pointer-sized slot (see windows/kernel32.py's own
# SRWLOCK comment) - void** is ABI-identical and avoids also declaring a
# matching struct type for calls the emitter synthesizes directly, never
# referenced by name from metalpy source (so never routed through the
# ordinary @extern forward-declaration machinery on their own).
def _needs_global_lock( var: Variable ) -> bool:
	return var.is_global and var.reassigned_outside_init

def _global_lock_name( var: Variable ) -> str:
	return mangle_qualname( var.qualname ) + '$lock'

# windows.kernel32.SRWLOCK's own mangled struct name - reused verbatim
# (not an independent void**/opaque type of our own) so that a program
# that ALSO happens to use lib/threading.py's FastLock somewhere (which
# independently triggers a REAL `AcquireSRWLockExclusive(struct windows$
# kernel32$SRWLOCK*)` extern declaration, via the ordinary @extern
# forward-declaration machinery) ends up with two IDENTICAL declarations
# of the same two functions, not two CONFLICTING ones - confirmed as a
# real compile error otherwise (clang: "conflicting types for
# 'AcquireSRWLockExclusive'") the first time this collided with a test
# that used both. Only ever used here as an incomplete (forward-tag-only)
# pointer parameter type, never dereferenced or given a real variable of
# this type - the lock global's own storage stays a plain `void*` (see
# _global_lock_name's own declaration site), cast to this type only at
# the two call sites that need it, so the real struct's full body is
# never required to exist in this translation unit at all. The class was
# renamed windows.kernel32._SRWLOCK -> SRWLOCK (module/package-visibility
# enforcement, SYNTAX.md - the leading underscore was a package-private
# naming choice, not the real Win32 API's own spelling, and lib/
# threading.py genuinely needs cross-package access to it) - this string
# MUST stay in lockstep with that real qualname, or the mangled name
# emitted here silently stops matching the struct's own real declaration.
_SRWLOCK_STRUCT_NAME = mangle_qualname( 'windows.kernel32.SRWLOCK' )

def _global_lock_acquire( lock_name: str ) -> str:
	if _target_uses_pthread_lock():
		# pthread_mutex_lock, not lib/threading.py's own FastLock (see
		# _global_lock_name's storage declaration site for why this needs a
		# REAL pthread_mutex_t, not a bare void* the way SRWLOCK's storage
		# gets away with) - <pthread.h> is force-included whenever any
		# locked global exists on this target (see emit_c()'s own
		# required_headers.add('pthread.h')), so the real prototype is
		# always in scope here, unlike Windows's own hand-declared forward
		# prototypes (SRWLOCK has no portable equivalent to "just include
		# the real header", since that would drag in the whole of
		# windows.h - pthread.h carries no such cost). Reached for macOS
		# only past _global_lock_supported()'s own poison-pill assert - by
		# the time this runs, that's already been survived (or this is a
		# unit test exercising the string-building logic directly without
		# going through the real gate - see this file's own tests).
		return f'\tpthread_mutex_lock( &{lock_name} );'
	return f'\tAcquireSRWLockExclusive( (struct {_SRWLOCK_STRUCT_NAME}*)&{lock_name} );'

def _global_lock_release( lock_name: str ) -> str:
	if _target_uses_pthread_lock():
		return f'\tpthread_mutex_unlock( &{lock_name} );'
	return f'\tReleaseSRWLockExclusive( (struct {_SRWLOCK_STRUCT_NAME}*)&{lock_name} );'

def _field_lock_prologue() -> str:
	''' PLAN_THREAD_SAFE_SHARED_STATE.md Part B: acquire_field_lock/
	release_field_lock wrap the per-object $header.lock the same way
	retain_object/release_object wrap $header.ref_count - real, one-place
	FUNCTIONS (not inline call text at every ir.AcquireFieldLock/
	ReleaseFieldLock site) specifically so the immortal-object skip below
	lives in exactly one place. Required, not just tidy: a compile-time-
	baked static const instance (a string literal, a static vtable
	instance - emit_str_literal and friends) sets $header.ref_count =
	METALPY_IMMORTAL_REFCOUNT directly in a designated-initializer list
	that never mentions .lock at all - C zero-fills that (a real guarantee
	for a static aggregate initializer, unlike sys.alloc's own unions-
	initialized heap memory), which is a valid unlocked SRWLOCK but NOT a
	valid pthread_mutex_t (POSIX gives zero-init-safety to neither by
	contract - A.3's own point, reused here) - locking it directly would be
	real undefined behavior the first time ANY of its fields is ever read
	through an ordinary GetAttr. Skipping locking entirely for an immortal
	object is also simply correct on its own terms, the same reason
	retain_object/release_object already skip refcount work for one: an
	object nothing ever frees needs no exclusion for field access to be
	safe UNLESS something actually reassigns that field concurrently -
	immortal objects in this codebase are exactly the ones nothing does
	that to (compile-time constants), so this stays sound, not just
	convenient. '''
	if _target_uses_pthread_lock():
		lock_expr = 'pthread_mutex_lock( &obj->lock )'
		unlock_expr = 'pthread_mutex_unlock( &obj->lock )'
	else:
		lock_expr = f'AcquireSRWLockExclusive( (struct {_SRWLOCK_STRUCT_NAME}*)&obj->lock )'
		unlock_expr = f'ReleaseSRWLockExclusive( (struct {_SRWLOCK_STRUCT_NAME}*)&obj->lock )'
	return (
		f'static inline void acquire_field_lock( ObjectHeader* obj ) {{\n'
		f'\tif ( obj && obj->ref_count != METALPY_IMMORTAL_REFCOUNT ) {{\n'
		f'\t\t{lock_expr};\n'
		f'\t}}\n'
		f'}}\n'
		f'static inline void release_field_lock( ObjectHeader* obj ) {{\n'
		f'\tif ( obj && obj->ref_count != METALPY_IMMORTAL_REFCOUNT ) {{\n'
		f'\t\t{unlock_expr};\n'
		f'\t}}\n'
		f'}}\n'
	)

# set once, at the very top of emit_c() (which has `compiler` in scope) -
# read from _emit_instruction/_emit_instructions, which do not. This is
# genuinely whole-compilation-unit ambient state (every line emit_c() ever
# produces in ONE call is for the SAME target OS - there is no scenario
# where two DIFFERENT parts of one program compile for two different
# targets), the same sense in which metalpy source's own `compiler.target.
# os` is an ambient, not per-call, fact - a plain module global here
# mirrors that rather than threading a new parameter through emit_function/
# _emit_instruction's own long, already-established call chains for a
# value that never actually varies within a single emit_c() call. Safe
# across tests.py's own parallelism: shards are separate subprocesses (see
# tests.py's own "worker subprocesses" sharding), each with independent
# module state, and emit_c() calls never interleave within one process.
_target_os: str|None = None

# debug-only alloc-site tracking gate (dump_live_objects) - mirrors _target_os
# above: set for real inside emit_c() (compiler.disco.active_target['debug']),
# defaults False at import time so PROLOGUE (built at import time, below)
# reflects a release-shaped ObjectHeader/vtable with no tracking overhead.
_target_debug: bool = False

def _target_uses_pthread_lock() -> bool:
	# Linux AND macOS share the identical pthread_mutex_t codegen (same
	# storage type, same pthread_mutex_init/lock/unlock calls, same
	# <pthread.h>/-lpthread requirement) - both are 'family: unix' per
	# targets.py's own ActiveTarget, and macOS's pthread implementation is
	# every bit as real a POSIX one as Linux's, just never run against a
	# real compiler in this repo's own dev environment (no macOS machine
	# here - see _global_lock_supported's own poison-pill comment). One
	# shared predicate, not two copy-pasted per-OS branches at each of this
	# module's several call sites, so the day someone actually verifies
	# macOS and deletes the poison pill below, there's exactly one already-
	# correct code path waiting, not four to individually double-check.
	return _target_os in ( 'linux', 'macos' )

# the full, unconditional concatenation - kept for callers that want every
# PROLOGUE helper regardless of whether a specific program needs it (e.g.
# emitter_c_test.py's own release_object test). emit_c() itself assembles
# the pieces above selectively instead of using this directly. Placed here
# (not right after _PROLOGUE_FLOAT_PARSE, where it lived before Part B),
# not earlier: _object_header_prologue() needs _target_uses_pthread_lock()
# defined first, since this executes at IMPORT time, in top-to-bottom
# module-body order - _target_os is still None at this point (only ever set
# inside emit_c() itself, per that global's own docstring), so this always
# reflects the Windows-shaped ObjectHeader (a `void* lock`) regardless of
# whatever target a later real emit_c() call actually targets.
PROLOGUE = _object_header_prologue() + _PROLOGUE_RETAIN + _prologue_release() + _PROLOGUE_ARITH + _PROLOGUE_FLOAT_FORMAT + _PROLOGUE_FLOAT_PARSE + _PROLOGUE_CRASH_HANDLER

def _global_lock_supported() -> bool:
	''' PLAN_THREAD_SAFE_SHARED_STATE.md Part A: Windows (SRWLOCK) and Linux
	(pthread_mutex_t, needing a real pthread_mutex_init() call, not
	zero-init-safe the way SRWLOCK is - see that plan's own A.3
	POSIX-asymmetry note, and _global_lock_name's storage-declaration site
	below for how the init call is actually wired in) are both real,
	VERIFIED targets (this repo's own compiler matrix: clang/MSVC on
	Windows, gcc via WSL on Linux). macOS structurally reuses the exact
	same pthread_mutex_t codegen as Linux (_target_uses_pthread_lock) -
	the code path exists and is believed correct by construction, but has
	never actually been compiled, linked, or run on real macOS hardware or
	a real Apple/GNU toolchain (no macOS machine is available in this
	repo's own dev environment - see PLAN_THREAD_SAFE_SHARED_STATE.md's
	own A.3 note on the verified-target matrix). Rather than silently
	claiming untested support (this function returning True with nobody
	the wiser) or silently staying unprotected forever (this function
	returning False, leaving a REAL race in place with no signal that
	anything's missing), this deliberately POISONS the macOS path: every
	call site below is gated on locked_globals (or on a real
	ir.AcquireGlobalLock/ReleaseGlobalLock marker already having been
	emitted for a genuinely reassigned global - see cfg.py's own
	`dest.is_global` branch), so this only ever fires for a compile that
	actually NEEDS this mechanism on macOS, not for every macOS compile
	regardless of relevance. Whoever next has real macOS hardware to
	verify against should delete this assert (and this whole paragraph),
	run thread_safe_globals_test.py's own stress tests for real, and
	update this docstring plus PLAN_THREAD_SAFE_SHARED_STATE.md's own
	status section accordingly - not just delete the assert and assume. '''
	if _target_os == 'macos':
		assert False, (
			'this code is completely untested - macOS Part A global-locking has '
			'structurally identical codegen to the verified Linux (pthread_mutex_t) '
			'path, but has never actually been compiled/linked/run on real macOS '
			'hardware (no macOS machine in this dev environment). Do not remove this '
			'assert without first doing that real verification - see this function\'s '
			'own docstring and PLAN_THREAD_SAFE_SHARED_STATE.md\'s A.3 note.'
		)
	# Checked here, not inside _needs_global_lock itself (which stays a pure
	# "is this global genuinely reassigned" fact, independent of what THIS
	# pass of the emitter can currently do about it) - keeps detection and
	# platform-support-status separate. False here means every one of this
	# global's accesses falls through to today's plain, unwrapped emission -
	# exactly the pre-existing (unprotected, not a regression) behavior for
	# any target this returns False for.
	return _target_os in ( 'windows', 'linux', 'macos' )

# the bare forward tag (no body) is always legal to repeat, even if the
# real struct ALSO gets a full body defined elsewhere in this same
# translation unit (C explicitly allows redeclaring an incomplete tag any
# number of times - only a full body can't be repeated) - safe to emit
# unconditionally here regardless of whether lib/threading.py's FastLock
# is used anywhere else in this program.
_PROLOGUE_GLOBAL_LOCK_WINDOWS = f'''\
struct {_SRWLOCK_STRUCT_NAME};
void AcquireSRWLockExclusive( struct {_SRWLOCK_STRUCT_NAME}* SRWLock );
void ReleaseSRWLockExclusive( struct {_SRWLOCK_STRUCT_NAME}* SRWLock );
'''

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

def _overload_symbol_index( fn: Function, group: Overload ) -> int:
	''' fn's position among the OTHER group.implementations that also need a
	real C body (used to build a disambiguating symbol suffix - see
	mangle_function_qualname). Keyed by (file, line), not object identity or
	`list.index()`: lowering.py's own _resolve_original (the generic-
	specialization/winning-stub path) can hand back a dataclasses.replace()-
	derived COPY of the winning implementation (same file/line/qualname,
	different return_type/object identity) as the actual Call target, so a
	plain identity scan would miss it - and list.index() would fall back to
	Function's auto-generated structural __eq__, which is exactly what
	mpy_types.py's _leaf_is_accepted docstring already warns is wrong/
	expensive for these dataclasses. (file, line) uniquely picks out the
	specific `def` among siblings sharing one qualname. '''
	for i, candidate in enumerate( group.implementations ):
		if candidate.file == fn.file and candidate.line == fn.line:
			return i
	raise AssertionError( f'{fn.qualname}: not found in its own overload_group.implementations by (file, line)' )

def mangle_function_qualname( fn: Function ) -> str:
	''' union-aware entry point for a Function's own symbol name - call
	this (not mangle_qualname directly) for any real function/method
	symbol. Only matters for union_storage.py's own synthesized per-member
	constructor Functions on an ANONYMOUS union (fn.cls is a TaggedUnion
	with file is None - see mangle_type's own comment on why that's the
	anonymous-union signal): fn.qualname there directly embeds the outer
	union's own raw '|'-joined qualname as its prefix (e.g.
	'builtins.str|intrinsics.NoneType.str'), which plain mangle_qualname
	would leave a literal '|' in (not a legal C identifier character) -
	same failure mode mangle_type's own CUnion branch already documents
	for the payload struct. A REAL @union class's own methods (fn.cls.file
	is never None there) mangle exactly as before - this only branches for
	the specific shape nothing scheduled before Lowering._coerce_into_union
	started actually calling these constructors.

	Every member of an @overload group shares its group's own .qualname
	(mpy_types.Overload's docstring: "stands in for a Function when
	multiple defs share a name") - a group with only one real implementation
	(the overwhelmingly common case: signature-only stubs routed to one
	real body) still mangles exactly as before, unsuffixed. Only when more
	than one member of the SAME group actually needs its own real C body
	(distinct implementations for distinct signatures, not just distinct
	stubs) does this append a $$overload<N> suffix, keyed by fn's own
	position among those bodies - see _overload_symbol_index. '''
	if isinstance( fn.cls, TaggedUnion ) and fn.cls.file is None:
		return f'{mangle_type( fn.cls )}${mangle_qualname( fn.stem )}'
	base = mangle_qualname( fn.qualname )
	group = fn.overload_group
	if group is not None and len( group.implementations ) > 1:
		return f'{base}$$overload{_overload_symbol_index( fn, group )}'
	return base

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
	# IEEE 754 single/double precision. Only the canonical stems appear here -
	# `float`/`double` are aliases that resolve to the SAME Scalar object whose
	# .stem is 'f32'/'f64' (see discovery.py's get_intrinsics), so they map
	# through these entries automatically
	'f32': 'float', 'f64': 'double',
}

# the floating-point scalar stems - neither signed nor unsigned integers, so
# they bypass the whole checked/wrap/saturate integer machinery (see the float
# opcode handling below and lowering.py's own _is_float_scalar)
_FLOAT_STEMS: frozenset[str] = frozenset([ 'f32', 'f64' ])

def _is_float_type( t: Type|None ) -> bool:
	return isinstance( t, Scalar ) and t.stem in _FLOAT_STEMS

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
	t = t.unwrap_ownership() # move[T]/copy[T] are compile-time only, invisible in C
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
			if base.stem == 'Ptr':
				return f'{inner}*'
			# a flat, single leading const covers the whole pointer chain
			# in this codebase's model (never a per-level const, e.g. real
			# C's `const T* const*`) - inner_type itself being Ptr[U]/
			# ConstPtr[U] (ConstPtr[ConstPtr[T]] etc) means the recursive
			# _value_spelling/c_type call above already produced that
			# single leading const, so just add this level's own pointer
			# star; re-adding 'const ' here too would double it ("const
			# const T**") - a real, confirmed -Wduplicate-decl-specifier
			# on clang, not just cosmetic pickiness
			return f'{inner}*' if inner.startswith( 'const ' ) else f'const {inner}*'
		if t.is_rc_pointer():
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
	if t.is_rc_pointer():
		# RCClass and TupleType both, in one branch - a bare RC pointer is a
		# bare RC pointer regardless of which kind produced it.
		#
		# PLAN_TUPLE.md, found by a real hang (not anticipated up front): a
		# bare, unresolved TupleType can still reach here even after
		# monomorphize.py's own substitute_type_params fix - a plain LOCAL/
		# PARAMETER/GLOBAL declaration's own annotation type (e.g. `dm1:
		# tuple[int,int] = ...`) is never independently resolved anywhere
		# on the general path, only when it happens to be the destination
		# of a freshly-lowered tuple LITERAL (lowering.py's _expr_Tuple
		# resolves expected_type defensively for exactly that one case).
		# Rather than chase every remaining place resolution could be lost,
		# TupleType gets the exact same treatment Specialization already
		# gets a few lines up: directly emittable via its own mangled
		# qualname, no resolution required - guaranteed to match whatever
		# concrete backing RCClass eventually gets emitted under the SAME
		# mangled name, since TupleType.qualname == backing.qualname by
		# construction (tuple_storage.py's own TupleStorage.get()). Always
		# a pointer, same as an RCClass - a tuple's backing is never
		# anything else, which is what lets both share this branch.
		return f'struct {mangle_type(t)}*'
	if isinstance( t, ( CStruct, CUnion, TaggedUnion )):
		return f'{_class_keyword(t)} {mangle_type(t)}'
	if isinstance( t, CEnum ):
		return mangle_type( t ) # the typedef name itself, no struct/union prefix
	if isinstance( t, CType ):
		return t.c_name
	if isinstance( t, FixedArrayType ):
		# never reached on a legitimate path: a struct/union FIELD of this
		# type is special-cased directly in _struct_or_union_body (C's own
		# discontinuous array declarator, "TYPE NAME[N]", doesn't fit this
		# function's plain "return a type string" shape at all) - discovery.py
		# already rejects every OTHER annotation position (parameter, return
		# type, local/global variable) before this module ever runs, and
		# reading a FixedArrayType field back out as an ordinary value isn't
		# implemented (see FixedArrayType's own docstring) - so reaching this
		# function with one at all means something upstream failed to guard
		# a position that needs its own guard, not a legitimate use.
		raise NotImplementedError(
			f'c_type: {t.qualname} (a fixed-size inline array) cannot be spelled as an ordinary C type - '
			f'it only exists as a @cstruct/@cunion FIELD, handled directly by _struct_or_union_body'
		)
	raise NotImplementedError( f'c_type: unsupported type {t!r}' )

def _is_noreturn( t: Type|None ) -> bool:
	return isinstance( t, Scalar ) and t.stem == 'NoReturn'

def _callable_ptr_type( t: Type|None ) -> tuple[CallableType,int]|None:
	''' (t's own CallableType, indirection depth) if t is N>=1 levels of
	Ptr[Ptr[...[Callable[...]]...]] wrapping a bare CallableType (see
	PLAN_CALLABLE.md) - the ptr-vs-bare distinction and the interning both
	live in discovery.py/type_resolver.py already (see TypeResolver.
	_callable_type_of); this is emitter_c.py's own copy of the same
	structural check since this module works on Type objects directly,
	with no TypeResolver instance around to call.

	depth is almost always 1 (an ordinary Ptr[Callable[...]] parameter/
	local/field/return/global), but can be more: Result[Ptr[Callable[...]],
	E]'s own union-payload storage indirects certain leaf types through an
	EXTRA pointer (UnionStorage's own representation choice), producing
	Ptr[Ptr[Callable[...]]] - real and reachable via any Result/Optional
	whose leaf is itself Ptr[Callable[...]] (e.g. a plain
	list[Ptr[Callable[...]]].__getitem__'s own Result[_,IndexError] -
	confirmed via a real NotImplementedError crash, not just reasoning).
	C's function-pointer declarator generalizes to N indirection levels by
	adding N stars INSIDE the parens (RetType (**name)(Params) for N=2,
	etc) - structurally different from an ordinary object pointer chain
	(T**), which is why every caller below needs the depth, not just a
	yes/no answer. '''
	depth = 0
	cur = t
	while isinstance( cur, Specialization ) and isinstance( cur.base, Scalar ) and cur.base.stem in ( 'Ptr', 'ConstPtr' ):
		depth += 1
		cur = cur.args[0]
	if depth > 0 and isinstance( cur, CallableType ):
		return cur, depth
	return None

def _fn_ptr_cast_type( ret: str, params: list[str], *, stars: int = 1 ) -> str:
	''' the C function-pointer TYPE spelling itself (RetType (*)(ParamTypes)
	for stars=1, RetType (**)(ParamTypes) for stars=2, ...; no name) -
	shared by emit_interface_vtable_instance (casting a concrete
	implementation's address into a shared vtable slot type - always
	stars=1, a vtable slot is never itself indirected) and
	_emit_operand's own FunctionRef branch (spelling a bare function
	reference's cast expression), from whichever (ret, params) tuple the
	caller already has (_vtable_slot_c_type's own self-prepended shape, or
	_function_pointer_c_type's plain one below). '''
	params_str = ', '.join( params ) if params else 'void'
	return f'{ret} ({"*" * stars})( {params_str} )'

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

def _declarator( t: Type|None, name: str, *, volatile: bool = False ) -> str:
	''' "TYPE NAME" for an ordinary parameter/local-variable/struct-or-union-
	field declaration - except when t is Ptr[Callable[...]], where C's
	function-pointer syntax is the one declarator shape that ISN'T "prefix
	type, then name": the name goes INSIDE the parens (RetType (*name)
	(ParamTypes)), so plain string concatenation of c_type(t) and name can't
	express it. `volatile` is for Volatile[T] locals (_stmt_AnnAssign) only -
	never set for a function-pointer declarator or a field. '''
	callable_ptr = _callable_ptr_type( t )
	prefix = 'volatile ' if volatile else ''
	if callable_ptr is None:
		return f'{prefix}{c_type(t)} {name}'
	fn_type, depth = callable_ptr
	ret, params = _function_pointer_c_type( fn_type )
	params_str = ', '.join( params ) if params else 'void'
	return f'{prefix}{ret} ({"*" * depth}{name})( {params_str} )'

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
	t = t.unwrap_ownership()
	base = t.base if isinstance( t, Specialization ) else t
	if isinstance( base, ( RCClass, CStruct, CUnion, TaggedUnion, TupleType )):
		return f'{_class_keyword(base)} {mangle_type(t)}'
	if isinstance( t, CType ):
		return t.c_name
	return c_type( t ) # scalars/CEnum - value and reference spelling are identical

def _mark_used_if_none( operand: ir.Operand ) -> list[str]:
	''' MetalpyNone (see _NONE_PLACEHOLDER_TYPE) carries no real
	information - assigning one is a structurally-required IR shape (e.g.
	`ok: T = self.data.v_Ok` inside Result[T,E].unwrap()'s own generic
	body when T=NoneType, or .or_return()'s own dest when used as a bare
	statement, its value never actually consumed), not necessarily a value
	any caller goes on to read. A genuinely-unused NoneType local was never
	actionable dead code to begin with (there's nothing in it to act on),
	so marking it read here - always AFTER its real assignment, never
	before (an earlier read would be a genuine uninitialized-value bug,
	not just a spurious warning) - is safe in every case and silences
	-Wunused-variable/-Wunused-but-set-variable/C4189 on every such
	monomorphization without risking a false negative on a real,
	non-placeholder type. '''
	if isinstance( operand.type, Scalar ) and operand.type.stem == 'NoneType':
		return [ f'\t(void){_emit_operand(operand)};' ]
	return []

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
	tag_attr = base.get_local_or_raise( 'tag' )
	data_attr = base.get_local_or_raise( 'data' )
	assert isinstance( tag_attr, Variable ) and isinstance( data_attr, Variable ), \
		f'{base.qualname}: _tagged_union_storage has not run yet - no real storage shape to read'
	return _field_name( tag_attr.stem ), _field_name( data_attr.stem ), _field_name( 'v_Ok' ), _field_name( 'v_Err' )

def _union_tag_data_fields( union: TaggedUnion ) -> tuple[str,str]:
	''' (tag_field, data_field) C names for an anonymous error UNION (the E in
	Result[T,E] when E is itself a union like ZeroDivisionError|OverflowError).
	Same UnionStorage-synthesized 'tag'/'data' shape _result_tag_data_names
	reads for the outer Result, just for the inner error union. '''
	tag_attr = union.get_local_or_raise( 'tag' )
	data_attr = union.get_local_or_raise( 'data' )
	assert isinstance( tag_attr, Variable ) and isinstance( data_attr, Variable ), \
		f'{union.qualname}: union storage not synthesized (UnionStorage.get must run before emit)'
	return _field_name( tag_attr.stem ), _field_name( data_attr.stem )

def _union_member( union: TaggedUnion, member_type: Type ) -> tuple[int,Variable]:
	''' (ordinal, attr) for the member of `union` matching `member_type` - the
	ordinal is its index in .attributes (UnionStorage assigns tags in
	attribute order); attr.stem names its payload field (data.v_<attr.stem>).
	Matched by qualname (error classes are interned, but qualname is the
	stable key that survives the Specialization-vs-monomorphized-TaggedUnion
	split - see _result_error_type). '''
	for i, attr in enumerate( union.attributes ):
		if attr.type is not None and attr.type.qualname == member_type.qualname:
			return i, attr
	raise AssertionError( f'{member_type.qualname} is not a member of {union.qualname}' )

def _emit_widen_error( dest_expr: str, e_fn: Type, src_expr: str, e_op: Type ) -> list[str]:
	''' assign the error value `src_expr` (of type e_op) into the error lvalue
	`dest_expr` (of type e_fn), WIDENING when they differ. e_fn is guaranteed
	to cover e_op (type_resolver._require_result_return's leaves-containment
	check ran at lowering).

	The built-in arithmetic error leaves (OverflowError, ZeroDivisionError,
	FloatingPointError) are zero-payload markers, but or_return() widening is
	a GENERAL mechanism - it applies to any user-declared Result[T,E], and a
	user error class can carry real fields (`class ParseError: message: str`).
	Every branch below therefore copies the payload value alongside the tag,
	not just the tag - dropping it would silently discard the error's own
	data on every widening propagation. The payload is a bare pointer for any
	RCClass error (the only kind a bare `class Foo:` ever compiles to -
	discovery.py's own class-parsing rule), so this is a plain pointer copy,
	not a deep copy. RC ownership: this mirrors the pre-existing identical-
	type fast path immediately below EXACTLY (already a raw, no-retain struct
	copy, unchanged by this function) - not a new RC rule invented here, the
	same ownership-transfer semantics that already govern an ordinary
	(non-widened) error value propagating through OrReturn/OrJump. '''
	if e_op is e_fn:
		return [ f'\t\t{dest_expr} = {src_expr};' ] # identical layout - plain struct copy (fast path, copies any payload already)
	assert isinstance( e_fn, TaggedUnion ), f'widening into a non-union error type {e_fn!r}'
	fn_tag, fn_data = _union_tag_data_fields( e_fn )
	# e_op is only genuinely FLATTENABLE into e_fn's own member list when it's
	# itself a synthesized ANONYMOUS union (file is None) - the same
	# distinguishing test discovery.py's _get_or_create_union already uses when
	# flattening a wider union's own operands (only an anonymous operand
	# contributes its own leaves; a real user `@union class Foo:` stays a
	# single opaque member wherever it's nested). A NOMINAL union (e.g.
	# HTTPError, itself one of e_fn's own members verbatim) takes the
	# single-class path below just like any plain class leaf does - remapping
	# ITS OWN internal variants against e_fn's member list would look for e.g.
	# HTTPError's None-payload variant types as members of e_fn, which they
	# never are (type_resolver._atomic_leaves applies this identical
	# distinction to the type-checking side of the same widening, at lowering
	# time - see its own docstring).
	if not ( isinstance( e_op, TaggedUnion ) and e_op.file is None ):
		# single class (or nominal union) -> set the wide union's variant tag AND
		# copy its payload pointer into the matching v_<member> field
		ordinal, fn_attr = _union_member( e_fn, e_op )
		fn_field = _field_name( f'v_{fn_attr.stem}' )
		return [
			f'\t\t{dest_expr}.{fn_tag} = {ordinal};',
			f'\t\t{dest_expr}.{fn_data}.{fn_field} = {src_expr};',
		]
	# e_op is itself an anonymous (narrower) union -> remap each member's tag AND
	# copy its payload at runtime, one case per e_op member
	op_tag, op_data = _union_tag_data_fields( e_op )
	lines = [ f'\t\tswitch ( ({src_expr}).{op_tag} ) {{' ]
	for i, op_attr in enumerate( e_op.attributes ):
		fn_ordinal, fn_attr = _union_member( e_fn, op_attr.type )
		op_field = _field_name( f'v_{op_attr.stem}' )
		fn_field = _field_name( f'v_{fn_attr.stem}' )
		lines.append(
			f'\t\t\tcase {i}: {dest_expr}.{fn_tag} = {fn_ordinal}; '
			f'{dest_expr}.{fn_data}.{fn_field} = ({src_expr}).{op_data}.{op_field}; break;'
		)
	lines.append( '\t\t}' )
	return lines

def _result_error_type( result_type: Type ) -> Type:
	''' the E in a Result[T,E] value. Handles BOTH shapes a Result can take by
	emit time: a Specialization (E is args[1]) OR a concrete monomorphized
	TaggedUnion (Lowering.monomorphize_class gives it a flat qualname and
	substituted Ok/Err members, so E is the 'Err' member's own type). A
	function's return_type in particular is usually the concrete class, not the
	Specialization - the reason _emit_or_return must not assume Specialization.
	Uses .qualname (never !r) in messages - a full type repr recurses. '''
	if isinstance( result_type, Specialization ):
		assert len( result_type.args ) == 2, f'not a Result[T,E]: {result_type.qualname!r}'
		return result_type.args[1]
	assert isinstance( result_type, TaggedUnion ), f'not a Result[T,E]: {getattr(result_type, "qualname", result_type)!r}'
	for attr in result_type.attributes:
		if attr.stem == 'Err':
			assert attr.type is not None, f'{result_type.qualname}: Err member has no resolved type'
			return attr.type
	raise AssertionError( f'{result_type.qualname}: no Err member (not a Result[T,E])' )

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

def _struct_or_union_body( name: str, keyword: str, attrs: list[tuple[str,Type,int|None]], packed: bool = False ) -> str:
	''' packed=True (CStruct.packed/CUnion.packed, from @cstruct(packed=True)/
	@cunion(packed=True)) wraps the WHOLE body in #pragma pack(push,1)/pop -
	confirmed identical layout across MSVC/clang/gcc (a whole-aggregate
	pack directive is portable; see the per-field case just below for why
	that's NOT true of every #pragma pack usage).

	Each attrs entry's own third element is a per-field C alignment
	override (Variable.c_align, from a field declared `Aligned[N, T]`) -
	works in EITHER direction (N below or above the field's own natural
	alignment), independent of `packed`, and NEVER combined with it on the
	same struct (compiler.py's _validate_packed_field_alignment_conflict
	rejects that combination outright, before this ever runs). Deliberately
	NOT emitted as a bare mid-struct #pragma pack(push,N)/pop bracketing
	just that field: confirmed empirically that MSVC honors a pack change
	made between two member declarations (mid-struct) on a PER-FIELD basis,
	but clang/gcc silently ignore it and keep the struct's own natural
	alignment instead - only a pack directive that wraps the ENTIRE
	aggregate is portable on clang/gcc. The portable per-field mechanism is
	instead a #if defined(_MSC_VER) && !defined(__clang__) split:

	- real MSVC: #pragma pack(push,N)/pop (a CAP - can only shrink, never
	  grow, alignment beyond natural) combined with __declspec(align(N))
	  on the field itself (the opposite: __declspec can only GROW
	  alignment, confirmed via a real repro that __declspec(align(4)) on a
	  natural-8-aligned u64 field is silently a no-op, still offset 8, not
	  4). Neither alone covers both directions, but layering both
	  together does: confirmed empirically that pack+declspec combined
	  reproduces the exact pack-alone result when shrinking and the exact
	  declspec-alone result when growing, on real MSVC.
	- clang/gcc: the single GNU __attribute__((packed,aligned(N))) field
	  attribute already covers both directions on its own (confirmed
	  empirically) - `packed` relaxes the field down to its own emitted
	  alignment floor of 1 byte, then `aligned(N)` sets the exact final
	  value, whether that's below or above the field's natural alignment.

	The `!defined(__clang__)` half of the MSVC guard is load-bearing, not
	defensive styling: this machine's own clang targets
	x86_64-pc-windows-msvc and DOES define _MSC_VER (for MSVC-header
	compatibility), so a bare `defined(_MSC_VER)` guard silently routed
	clang down the MSVC branch too - where clang's real pragma-pack
	semantics (the mid-struct case above) do NOT match real MSVC,
	reproducing the exact wrong-size bug this feature exists to prevent.
	Confirmed via a real repro: bare _MSC_VER guard gave sizeof==24 under
	this clang instead of the correct 16 every other compiler (real MSVC,
	gcc) agreed on. '''
	body_lines: list[str] = []
	if not attrs:
		# MSVC (and pedantic C) reject empty structs/unions:
		body_lines.append( '\tchar dummy;' )
	else:
		for field_name, field_type, align in attrs:
			if isinstance( field_type, FixedArrayType ):
				# C's array declarator is discontinuous ("TYPE NAME[N];", not
				# a plain prefix type followed by the name - see
				# FixedArrayType's own docstring and _declarator's identical
				# function-pointer special case) - _declarator's plain
				# "TYPE NAME" concatenation can't express this
				decl = f'{c_type(field_type.elem_type)} {_field_name(field_name)}[{field_type.count}]'
			else:
				decl = _declarator( field_type, _field_name(field_name) )
			if align is None:
				body_lines.append( f'\t{decl};' )
			else:
				body_lines.append( '#if defined(_MSC_VER) && !defined(__clang__)' )
				body_lines.append( f'#pragma pack(push, {align})' )
				body_lines.append( f'\t__declspec(align({align})) {decl};' )
				body_lines.append( '#pragma pack(pop)' )
				body_lines.append( '#else' )
				body_lines.append( f'\t{decl} __attribute__(( packed, aligned({align}) ));' )
				body_lines.append( '#endif' )
	lines: list[str] = []
	if packed:
		lines.append( '#pragma pack(push, 1)' )
	lines.append( f'{keyword} {name} {{' )
	lines.extend( body_lines )
	lines.append( '};' )
	if packed:
		lines.append( '#pragma pack(pop)' )
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

# metalpy's own main() is compiled into this private helper, never the real
# C `main` symbol directly - emit_c() synthesizes a thin, always-present real
# `int main(argc, argv)` that calls __metalpy_init(), then this, then (debug
# builds only) the leak-check epilogue __metalpy_deinit() - see emit_c()'s
# own "whole-program driver" section. Only for the ordinary zero-parameter
# `def main() -> i32:` shape (_function_prototype's own comment) - the rare
# lowering-fixture-only parameterized shape keeps the OLD direct-main-is-the-
# real-entry-point behavior unchanged, since it never reaches this thin-
# wrapper synthesis at all (no test exercises it through emit_c()).
_USER_MAIN_C_NAME = '__metalpy_user_main'

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

def _name_used_as_value( name: str, body_text: str ) -> bool:
	''' emit_function's own unused-parameter (void)-marking check: is `name`
	genuinely referenced as a value somewhere in the body, as opposed to
	merely appearing as a struct FIELD of the same name (e.g. `self` is
	also the Closure struct's own captured-receiver field, so `($t8)->self
	= ...` mentions the literal token `self` without reading the
	parameter) - a bare `\bname\b` search alone can't tell those apart and
	wrongly treats the parameter as used, leaving the real unused
	parameter's own -Wunused-parameter unsilenced '''
	return re.search( rf'(?<!\.)(?<!->)\b{re.escape(name)}\b(?!\$)', body_text ) is not None

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
		name = mangle_function_qualname( function )
		return f'static void {name}( void* __obj )'
	params: list[str] = []
	if _has_self( function ):
		params.append( f'{_self_c_type(function.cls)} self' )
	for p in ( function.parameters or [] ):
		params.append( _declarator( p.type, _c_local_name( p )))
	params_str = ', '.join( params ) if params else 'void'
	if _is_entry_point( function ):
		# ordinary zero-parameter `def main() -> i32:` shape: compiled into
		# the private __metalpy_user_main helper, never the real C `main`
		# symbol - see _USER_MAIN_C_NAME's own comment. No argc/argv here
		# any more (sys.argv's own real-value capture now happens in the
		# synthesized real main() wrapper, before this is ever called - see
		# emit_c()'s own "whole-program driver" section), so plain `void`.
		if not params:
			return f'int {_USER_MAIN_C_NAME}( void )'
		# rare lowering-fixture-only shape: a handful of test fixtures
		# declare a function LITERALLY named main with its own parameter for
		# unrelated reasons (generic dispatch tests, never actually emitted
		# through emit_c()'s real entry-point synthesis) - preserve the old
		# "this literal main() IS the real C entry point" behavior rather
		# than silently dropping a declared parameter from the C signature.
		return f'int main( {params_str} )'
	noreturn = '_Noreturn ' if _is_noreturn( function.return_type ) else ''
	# @extern(lib, symbol) functions are declared with their raw C symbol
	# name, not the metalpy-qualified name — the linker resolves the raw
	# symbol from the foreign library, not from this translation unit
	name = function.extern_symbol if function.extern_lib is not None else mangle_function_qualname( function )
	# NoneType/NoReturn are value-less in C — return void, not MetalpyNone
	if _returns_void_in_c( function.return_type ):
		return f'{noreturn}void {name}( {params_str} )'
	fn_ret_type = _callable_ptr_type( function.return_type )
	if fn_ret_type is not None:
		# a function RETURNING a function pointer is C's gnarliest
		# declarator shape - RetType (*name(Params))(InnerParams) - the one
		# case where the OUTER function's own name+params nest INSIDE the
		# return type's own declarator, rather than the ordinary "prefix
		# type, then name" order every other return type uses (see
		# _declarator's own doc for the analogous parameter/local case).
		# Reuses _declarator as-is: passing "name( params )" as ITS OWN
		# `name` argument makes the two declarator layers nest correctly
		# with no separate logic needed - _declarator wraps whatever string
		# it's given in `(*...)`, and a call expression is a valid thing to
		# wrap.
		return f'{noreturn}{_declarator( function.return_type, f"{name}( {params_str} )" )}'
	ret = c_type( function.return_type )
	# generic @extern monomorphized to different pointer types share the
	# same C symbol — use void* for all object pointers to avoid
	# conflicting prototypes for the same symbol
	if function.extern_lib is not None and ret.endswith( '*' ):
		ret = 'void*' if not ret.startswith( 'const' ) else 'const void*'
	return f'{noreturn}{ret} {name}( {params_str} )'

def _emit_operand( op: ir.Operand ) -> str:
	if isinstance( op, ir.Const ):
		return _emit_const( op )
	if isinstance( op, ir.Temp ):
		return _temp_name( op.id )
	if isinstance( op, ir.FunctionRef ):
		# a bare function reference used as a value - see PLAN_CALLABLE.md.
		# Same cast-expression shape emit_interface_vtable_instance already
		# builds to point a vtable slot at a concrete implementation
		# (a plain pointer-to-pointer function-pointer cast, safe and free
		# at runtime, no wrapper needed) - reused via _function_pointer_c_type
		callable_ptr = _callable_ptr_type( op.type )
		assert callable_ptr is not None, f'_emit_operand: FunctionRef with non-Ptr[Callable] type {op.type!r}'
		fn_type, depth = callable_ptr
		ret, params = _function_pointer_c_type( fn_type )
		return f'({_fn_ptr_cast_type(ret, params, stars = depth)}){mangle_function_qualname(op.fn)}'
	if isinstance( op, Variable ):
		# locals (parameters, stack locals) use _c_local_name; globals
		# need the full mangled qualname (cross-TU visibility)
		return _c_local_name( op ) if not op.is_global else mangle_qualname( op.qualname )
	raise NotImplementedError( f'_emit_operand: unsupported operand {op!r}' )

# stems wider than plain C `int` - a bare, un-cast literal like `1` silently
# defaults to `int` (C's own "usual arithmetic conversions") when used
# directly as an operand inside an expression (as opposed to being a properly
# declared variable's own initializer, which isn't affected the same way) -
# confirmed to cause real UB: `i128(1) << 127` emitted as `(1) << (127)`,
# shifting a plain 32-bit int by 127 bits. i8/i16/i32/u8/u16/u32/bool are
# deliberately excluded - C's own default `int` width already matches or
# covers those, so casting them is unnecessary churn (also would break
# existing exact-string test assertions on ordinary i32 field literals)
_WIDE_INT_STEMS: frozenset[str] = frozenset([ 'i64', 'u64', 'i128', 'u128', 'isize', 'usize' ])

# i128/u128's own top-bit shift amount, as a C sizeof expression rather than
# a hardcoded 127 - __metalpy_wideint is only 64 bits wide under MSVC's
# fallback (see its own typedef comment), where a literal `<< 127` is UB
# (shift amount exceeds the operand's real width). sizeof is evaluated by
# whichever compiler actually processes the emitted C, so this always
# matches __metalpy_wideint's REAL width (64 under MSVC, 128 under
# gcc/clang) - the same technique isize/usize already use for their own
# compiler-dependent width (sizeof(intptr_t)*8 - 1). Confirmed the hardcoded
# 127 broke EVERY i128/u128 float-cast range check under MSVC, even for
# trivially in-range values (see test_checked_cast_i128_u128_in_range).
_WIDEINT_TOP_BIT_SHIFT = 'sizeof(__metalpy_wideint)*8 - 1'

def _emit_wide_int_const( value: int, stem: str ) -> str:
	''' a _WIDE_INT_STEMS constant, cast to its own C type. A bare C integer
	literal token can represent at most 64 bits of magnitude (stdint.h
	guarantees `unsigned long long` is >=64-bit; nothing wider has portable
	literal syntax) - within that budget, a plain cast suffices. Beyond it
	(only reachable for i128/u128 - every other stem's own real range is
	<=64 bits, see _FIXED_INT_BITS), no token can spell the value at all, cast
	or not (confirmed: `(unsigned __int128)999999999999999999999999999999`
	still fails to compile - the LITERAL TOKEN itself is rejected before any
	cast applies) - split into 64-bit hi/lo halves and reconstruct via shift,
	the same bit-pattern technique _int_min_max_bit_pattern/_signed_min_max
	already use for i128 MIN/MAX. '''
	ctype = _SCALAR_C_TYPES[stem]
	if -1 * ( 2**64 - 1 ) <= value <= 2**64 - 1:
		# an explicit ULL suffix on the literal's own MAGNITUDE, not a bare
		# unsuffixed decimal token - confirmed directly that MSVC's cl.exe,
		# even under /std:c11, mistypes an unsuffixed decimal literal outside
		# plain `int`'s own range as `unsigned int` (not the C99+-mandated
		# long/long long promotion): sizeof(-2147483648) is 4 there, and
		# `(int64_t)-2147483648` silently corrupts to a huge positive value
		# instead of the real negative one (caught chasing int_test.py's
		# test_i32_conversions - the exact literal that broke it). Negation
		# is applied to the whole CAST expression afterward, never baked into
		# the literal token itself - this also sidesteps INT64_MIN's own
		# classic "positive magnitude doesn't fit a signed 64-bit literal"
		# problem, since the magnitude is always spelled as unsigned. The
		# negation itself happens in UNSIGNED arithmetic (well-defined modular
		# wraparound in C), with the cast to the signed ctype applied last -
		# NOT `-((ctype)magnitude)`, which is real signed-overflow UB for the
		# exact MIN magnitude of any width (e.g. i64: `-((int64_t)
		# 9223372036854775808ULL)` casts an out-of-range unsigned magnitude to
		# a negative int64_t - implementation-defined but two's-complement in
		# practice, giving INT64_MIN already - then negates THAT, overflowing
		# signed 64-bit a second time). Confirmed via a real crash: `e: i64 =
		# -9223372036854775808` compiled clean but crashed with SIGILL at
		# runtime (a hardware trap from the resulting UB), caught while
		# building fixed-width int __str__ support and needing to construct
		# MIN literals for test coverage - a real, independent, pre-existing
		# bug, not caused by that work. The unsigned intermediate must be
		# stem's OWN same-width unsigned counterpart (_SIGNED_TO_UNSIGNED), not
		# a fixed 64-bit type - i128's ctype is 128-bit __metalpy_wideint, and
		# negating in a narrower 64-bit `unsigned long long` first then
		# widening the cast produces a WRONG positive value (a same-signedness
		# 64->128 widen zero-extends instead of reinterpreting bits) - caught
		# by a real test regression (wide_int_test's own
		# test_saturating_negate_i128) while first drafting this fix.
		#
		# Only SIGNED stems need this uctype detour: a negative `value`
		# reaching here for an UNSIGNED stem (e.g. -1 encoding USIZE_MAX as a
		# two's-complement bit pattern) negates directly in ctype itself,
		# which is already well-defined modular arithmetic with no signed-
		# overflow UB possible - the double-negation bug this fix targets is
		# specific to signed types. _SIGNED_TO_UNSIGNED has no 'usize' (etc.)
		# entry, so routing unsigned stems through it too is a plain KeyError,
		# not just unnecessary - caught by a real regression (socket_test/
		# ssl_test both embed a negative-encoded usize global) while
		# broadening this fix beyond the signed case it was first written for.
		magnitude = abs( value )
		if value < 0:
			# 0-x instead of -x: bitwise identical two's-complement negation
			# (both operands already the same unsigned type), but MSVC's /W4
			# flags a literal unary minus applied to an unsigned type (C4146)
			# even though the result is well-defined - binary subtraction from
			# an unsigned zero sidesteps the warning entirely, clang/gcc emit
			# identical code either way
			if stem in _SIGNED_TO_UNSIGNED:
				uctype = _SCALAR_C_TYPES[_SIGNED_TO_UNSIGNED[stem]]
				return f'(({ctype})(({uctype})0 - ({uctype}){magnitude}ULL))'
			return f'(({ctype})0 - ({ctype}){magnitude}ULL)'
		return f'(({ctype}){magnitude}ULL)'
	magnitude = abs( value )
	hi, lo = magnitude >> 64, magnitude & 0xFFFFFFFFFFFFFFFF
	# the shift amount is derived from __metalpy_wideuint's own real C width
	# rather than hardcoded, the same sizeof()-based technique
	# _WIDEINT_TOP_BIT_SHIFT already uses for i128 MIN/MAX - under MSVC's
	# 64-bit wideint/wideuint fallback this reduces to a safe, well-defined
	# no-op shift (0) instead of a shift-by-width (UB in C). The reconstructed
	# value is still numerically wrong in that case (a >64-bit magnitude
	# can't be represented in a genuinely 64-bit type by any expression -
	# this can only be reached via an explicit bit-reinterpretation cast or a
	# float->int128 literal cast, both of which deliberately bypass this
	# stem's own int_stem_range validation) but at least well-defined, not UB
	unsigned_expr = f'( ( (__metalpy_wideuint){hi}ULL << ( sizeof(__metalpy_wideuint)*8 - 64 ) ) | (__metalpy_wideuint){lo}ULL )'
	if _is_unsigned_stem( stem ):
		return unsigned_expr
	# same unsigned-negate-then-cast fix as the <=64-bit branch above (see its
	# own comment) - i128::MIN is exactly the same double-negation UB, just at
	# 128 bits: `-((__metalpy_wideint)unsigned_expr)` casts the 2**127 bit
	# pattern to a negative __int128 first (already the correct MIN, same
	# implementation-defined-but-relied-upon two's-complement reinterpret this
	# whole function already uses), then negates THAT, overflowing signed
	# __int128 - confirmed via the same real SIGILL crash as i64::MIN above.
	if value < 0:
		# 0-x, not -x - same MSVC C4146 (unary minus on unsigned) dodge as the
		# <=64-bit branch above, same bitwise result
		return f'(__metalpy_wideint)((__metalpy_wideuint)0 - {unsigned_expr})'
	return f'(__metalpy_wideint){unsigned_expr}'

def _emit_const( c: ir.Const ) -> str:
	if isinstance( c.type, FixedArrayType ):
		# the one supported FixedArrayType value (see its own docstring and
		# lowering.py's _expr_Constant fixed-array branch): a bare `0`
		# literal means "zero-fill the whole array" - the one shape a
		# C11 initializer can express for an embedded array field, valid
		# ONLY inside a designated-initializer compound literal (a plain
		# @cstruct's own stack-construction shape - see ir.Allocate's
		# emission), never as an ordinary assignment target
		assert c.value == 0, f'_emit_const: {c.type.qualname} only supports a 0 (zero-fill) constant, got {c.value!r}'
		return '{0}'
	if isinstance( c.value, bool ):
		return 'true' if c.value else 'false'
	if isinstance( c.value, float ) or ( isinstance( c.value, int ) and _is_float_type( c.type )):
		# a float literal, OR an int literal that was hinted to a float type
		# (e.g. the `1` in `f + 1`, typed f32 by _lower_binary_operands's
		# constant-hinting). repr() round-trips a Python float exactly; the
		# `f` suffix on an f32 avoids a double->float narrowing warning and
		# pins the constant to single precision. int values (1 -> "1.0f")
		# are formatted through float() so they always carry a decimal point
		value = float( c.value )
		is_f32 = _is_float_type( c.type ) and c.type.stem == 'f32'
		# a source literal that overflows AT PARSE TIME (Python's ast.parse
		# itself turns e.g. `1e400` into a Constant carrying float('inf'),
		# well before any lowering pass runs) has no valid C spelling via
		# repr() - 'inf'/'-inf'/'nan' aren't C tokens. Use the same
		# MSVC-vs-GCC/Clang-portable macros __metalpy_isnan/__metalpy_isinf
		# already rely on for classification, here for construction.
		if math.isinf( value ):
			sign = '-' if value < 0 else ''
			return f'{sign}__metalpy_inff()' if is_f32 else f'{sign}__metalpy_inf()'
		if math.isnan( value ):
			return '__metalpy_nanf()' if is_f32 else '__metalpy_nan()'
		text = repr( value )
		return text + 'f' if is_f32 else text
	if isinstance( c.value, int ):
		# pointer-typed constants (e.g. Ptr[None] = -1) need a cast.
		# Ptr[Callable[...]] is a special case within that: c_type()
		# doesn't know how to spell a bare function-pointer TYPE at all
		# (see _declarator's own comment - the name goes inside the
		# parens, so it's not an ordinary "prefix type" spelling) -
		# needed for e.g. a null-function-pointer sentinel like SIG_DFL
		# (`sig_dfl: Ptr[Callable[[i32],None]] = 0`)
		callable_ptr = _callable_ptr_type( c.type )
		if callable_ptr is not None:
			fn_type, depth = callable_ptr
			ret, params = _function_pointer_c_type( fn_type )
			return f'({_fn_ptr_cast_type(ret, params, stars = depth)}){c.value}'
		if isinstance( c.type, Specialization ):
			base = c.type.base
			if isinstance( base, Scalar ) and base.stem in ( 'Ptr', 'ConstPtr' ):
				return f'({c_type(c.type)}){c.value}'
		union_base = c.type.base if isinstance( c.type, Specialization ) else c.type
		if c.value == 0 and isinstance( union_base, TaggedUnion ):
			# PLAN_GENERATORS.md - type_resolver.py's generator_zero_rc_field
			# placeholder (a promoted local/field "not assigned yet", never
			# read before its own live-flag gates it) used to only ever
			# target a plain RCClass pointer (0 is a valid null-pointer bit
			# pattern there) or a T|None union (a real `None` Const used
			# instead, see _rewrite_generator_constructor). Result[T,E] broke
			# that: a TaggedUnion with NO None member (tag+data struct, no
			# pointer-shaped representation) assigned a bare `0` - valid to
			# the type checker (lowering.py's _check_assignable exempts every
			# TaggedUnion target from the strict literal-compatibility check
			# generally) but not valid C (assigning to 'struct ...' from
			# incompatible type 'int'). A zero-initialized compound literal
			# is valid C in assignment-RHS position (unlike the
			# FixedArrayType '{0}' above, which is brace-initializer-only)
			# and needs no real tag/payload - this placeholder is never read
			# before being overwritten.
			return f'({c_type(c.type)}){{0}}'
		stem = c.type.stem if isinstance( c.type, Scalar ) else None
		if stem in _WIDE_INT_STEMS:
			return _emit_wide_int_const( c.value, stem )
		if stem is not None and _is_unsigned_stem( stem ) and c.value < 0:
			# a literal conversion like u32(-12) bit-reinterprets straight to a
			# Const at lowering time (_lower_scalar_cast) rather than emitting a
			# CastWrap, so there's no cast-emission path to go through here -
			# cast explicitly or MSVC's /W4 flags the bare negative literal
			# initializing an unsigned type as C4245
			return f'({c_type(c.type)}){c.value}'
		if stem is not None and stem in _FIXED_INT_BITS and not _is_unsigned_stem( stem ) and c.value > ( 2 ** ( _FIXED_INT_BITS[stem] - 1 ) - 1 ):
			# the mirror case: a same-width construct-cast into a SIGNED type
			# (HRESULT(0x80090318)-style winerror.h/SEC_E_ constants - see
			# lib/windows/com/__init__.py's/lib/ssl.py's own matching
			# comments, both already documenting this as deliberate, same-
			# width, infallible bit-reinterpretation, exactly like T(x)'s own
			# general same-width contract) bit-reinterprets straight to a
			# Const here too - the literal's own natural (unsigned/wider)
			# type doesn't fit the target signed stem's positive range even
			# though its BIT PATTERN is exactly the intended value, which
			# clang/MSVC both flag (-Wconstant-conversion/C4309) on a bare,
			# uncast initializer
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
	dest = _temp_name( dest_temp_id )
	l, r = _emit_operand( left ), _emit_operand( right )
	tag_f, data_f, ok_f, err_f = _result_tag_data_names( result_spec )
	# the Err branches below zero the (unused, zero-payload-marker) err_f
	# payload slot alongside the tag, via a compound-literal cast to its own
	# real ctype (not a bare `= 0` - see _emit_set_result_err's own comment:
	# an uninitialized payload here is genuine, confirmed UB, but E isn't
	# always an RC pointer either, so the zero-literal needs E's own ctype)
	err_ctype = c_type( _result_error_type( result_spec ))
	if _is_pointer_type( ok_type ):
		if kind not in ( 'add', 'sub' ):
			raise NotImplementedError( f'checked pointer {kind} is not supported' )
		return [
			'\t{',
			'\t\tuintptr_t __tmp;',
			f'\t\tbool __overflow = {builtin}( (uintptr_t)({l}), (uintptr_t)({r}), &__tmp );',
			'\t\tif ( __overflow ) {',
			f'\t\t\t{dest}.{tag_f} = 1;',
			f'\t\t\t{dest}.{data_f}.{err_f} = ({err_ctype}){{0}};',
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
		f'\t\t\t{dest}.{data_f}.{err_f} = ({err_ctype}){{0}};',
		'\t\t} else {',
		f'\t\t\t{dest}.{tag_f} = 0;',
		f'\t\t\t{dest}.{data_f}.{ok_f} = __tmp;',
		'\t\t}',
		'\t}',
	]

# checked/panic-mode float +,-,* (FAddCheck/FSubCheck/FMulCheck) - the float
# analogue of _emit_check_arith. There's no overflow builtin for floats: compute
# the plain IEEE result, then flag it if it came out inf/nan (which also catches
# a nan/inf operand propagating through). dest.type is Result[float,
# FloatingPointError]. Division is NOT here - checked float / is FloatDivCheck
# (ZeroDivisionError|FloatingPointError union), see _emit_float_div_check.
_FLOAT_CHECK_SYMBOL: dict[type,str] = { ir.FAddCheck: '+', ir.FSubCheck: '-', ir.FMulCheck: '*' }

def _emit_float_check_arith( instr ) -> list[str]:
	symbol = _FLOAT_CHECK_SYMBOL[type(instr)]
	ok_type = instr.dest.type.args[0]
	ctype = c_type( ok_type )
	dest = _temp_name( instr.dest.id )
	l, r = _emit_operand( instr.left ), _emit_operand( instr.right )
	tag_f, data_f, ok_f, err_f = _result_tag_data_names( instr.dest.type )
	err_ctype = c_type( _result_error_type( instr.dest.type ))
	return [
		'\t{',
		f'\t\t{ctype} __tmp = ({l}) {symbol} ({r});',
		'\t\tif ( __metalpy_isinf( __tmp ) || __metalpy_isnan( __tmp ) ) {',
		f'\t\t\t{dest}.{tag_f} = 1;',
		f'\t\t\t{dest}.{data_f}.{err_f} = ({err_ctype}){{0}};', # see _emit_set_result_err's comment
		'\t\t} else {',
		f'\t\t\t{dest}.{tag_f} = 0;',
		f'\t\t\t{dest}.{data_f}.{ok_f} = __tmp;',
		'\t\t}',
		'\t}',
	]

# fixed-width integer scalar stem -> bit width - used only by
# _float_int_range_bounds/_int_min_max_bit_pattern below, for a float<->int
# range check's exact power-of-two boundary. isize/usize are handled
# separately there (their width isn't known to the Python emitter, only to
# the C compiler, via sizeof(intptr_t))
_FIXED_INT_BITS: dict[str,int] = {
	'i8': 8, 'u8': 8, 'i16': 16, 'u16': 16, 'i32': 32, 'u32': 32,
	'i64': 64, 'u64': 64, 'i128': 128, 'u128': 128,
}

def _float_int_range_bounds( stem: str, fctype: str ) -> tuple[str,str]:
	''' (min_expr, exclusive_max_expr): the exact range, as `fctype` C
	expressions, a float source value must satisfy before converting to the
	integer stem `stem` is defined (UB otherwise). exclusive_max_expr is the
	exact power-of-two ONE PAST the stem's real MAX (2**(n-1) signed, 2**n
	unsigned) rather than (fctype)MAX itself: MAX as an int literal rounds UP
	when converted to a float whose mantissa is narrower than n bits (e.g.
	(f64)INT64_MAX rounds to exactly 2**63, silently admitting one
	out-of-range value as in-range) - a power-of-two boundary is always
	exactly representable in a binary float instead, sidestepping the
	rounding entirely and needing no stdint MIN/MAX macro at all (unlike
	_SATURATE_LIMITS, this covers i128/u128 too, with no separate case
	needed). Compare with `< min_expr` / `>= exclusive_max_expr` to reject -
	never `> max_expr`, the imprecise comparison this replaces. '''
	if stem not in _FIXED_INT_BITS and stem not in ( 'isize', 'usize' ):
		raise NotImplementedError( f'float<->int range check: unsupported target stem {stem!r}' )
	if stem in ( 'isize', 'usize' ):
		shift = 'sizeof(intptr_t)*8 - 1'
	elif stem in ( 'i128', 'u128' ):
		shift = _WIDEINT_TOP_BIT_SHIFT  # see its own comment - not a hardcoded 127
	else:
		shift = str( _FIXED_INT_BITS[stem] - 1 )
	half = f'(({fctype})((__metalpy_wideuint)1 << ({shift})))' # exact 2**(n-1)
	if _is_unsigned_stem( stem ):
		return f'({fctype})0', f'(({fctype})2.0 * {half})' # exact 2**n
	return f'(-{half})', half

def _int_min_max_bit_pattern( stem: str ) -> tuple[str,str]:
	''' (MIN, MAX) as C expressions typed for stem's own ctype - covers every
	integer stem including i128/u128 (no stdint MIN/MAX macro exists for
	those; the two's-complement/all-ones bit pattern is used instead,
	matching _signed_min_max's existing technique for i128's own MIN/MAX,
	extended here to the unsigned side too since a float->int CLAMP needs
	both signed and unsigned targets, unlike _signed_min_max's own single
	(division INT_MIN/-1 UB check) caller). '''
	if stem in _SATURATE_LIMITS:
		return _SATURATE_LIMITS[stem]
	if stem == 'i128':
		return ( f'(__metalpy_wideint)((__metalpy_wideuint)1 << ({_WIDEINT_TOP_BIT_SHIFT}))',
			f'(__metalpy_wideint)(((__metalpy_wideuint)1 << ({_WIDEINT_TOP_BIT_SHIFT})) - 1)' )
	if stem == 'u128':
		return ( '(__metalpy_wideuint)0', '(~(__metalpy_wideuint)0)' )
	raise NotImplementedError( f'no MIN/MAX for stem {stem!r}' )

def _emit_float_cast_check( instr ) -> list[str]:
	# checked/panic-mode float-involving cast (FloatCastCheck). dest.type is
	# Result[target,FloatingPointError]. Two directions:
	#  - to-float (int->float, f64->f32): convert, then flag an inf/nan RESULT
	#    (overflow to inf, or a nan source surviving f64->f32).
	#  - float->int: a source that's nan or outside the int's range is UB to
	#    convert in C, so flag it BEFORE converting. Range compared via
	#    _float_int_range_bounds's exact power-of-two boundaries (every
	#    width, i128/u128 included - see its own docstring for why the naive
	#    (fctype)MAX comparison this replaces was imprecise at 64/128-bit
	#    extremes).
	ok_type = instr.dest.type.args[0]
	ctype = c_type( ok_type )
	operand = _emit_operand( instr.operand )
	dest = _temp_name( instr.dest.id )
	tag_f, data_f, ok_f, err_f = _result_tag_data_names( instr.dest.type )
	# the Err branch below only ever sets .tag = 1 - the .data.{err_f} slot
	# is otherwise left uninitialized, harmless in isolation but genuine,
	# confirmed UB once this Result gets copied/widened elsewhere (e.g.
	# _emit_widen_error's identical-layout fast path is a plain whole-struct
	# assignment) - see _emit_set_result_err's own comment. Zero-initialized
	# via a compound literal cast to E's own ctype, not a bare `= 0` - E
	# isn't always an RC pointer (could be a non-RC @cstruct value type)
	err_ctype = c_type( _result_error_type( instr.dest.type ))
	if _is_float_type( ok_type ):
		return [
			'\t{',
			f'\t\t{ctype} __tmp = ({ctype})({operand});',
			'\t\tif ( __metalpy_isinf( __tmp ) || __metalpy_isnan( __tmp ) ) {',
			f'\t\t\t{dest}.{tag_f} = 1;',
			f'\t\t\t{dest}.{data_f}.{err_f} = ({err_ctype}){{0}};',
			'\t\t} else {',
			f'\t\t\t{dest}.{tag_f} = 0;',
			f'\t\t\t{dest}.{data_f}.{ok_f} = __tmp;',
			'\t\t}',
			'\t}',
		]
	# float -> int
	stem = ok_type.stem if isinstance( ok_type, Scalar ) else None
	fctype = c_type( instr.operand.type ) # the source float type (f32/f64)
	min_c, max_c = _float_int_range_bounds( stem, fctype )
	return [
		'\t{',
		f'\t\tif ( __metalpy_isnan( {operand} ) || ({operand}) < {min_c} || ({operand}) >= {max_c} ) {{',
		f'\t\t\t{dest}.{tag_f} = 1;',
		f'\t\t\t{dest}.{data_f}.{err_f} = ({err_ctype}){{0}};',
		'\t\t} else {',
		f'\t\t\t{dest}.{tag_f} = 0;',
		f'\t\t\t{dest}.{data_f}.{ok_f} = ({ctype})({operand});',
		'\t\t}',
		'\t}',
	]

def _emit_float_to_int_clamp( instr ) -> list[str]:
	# wrap/saturate-mode float->int cast (FloatToIntClamp): clamp so it's never
	# UB - nan->0, below-range->MIN, above-range->MAX, else a plain truncating
	# cast. dest.type is the int scalar directly (no Result). Whether a value
	# needs clamping is decided via the exact float-typed boundaries
	# (_float_int_range_bounds); the CLAMPED RESULT itself uses the real
	# integer MIN/MAX (_int_min_max_bit_pattern) - two different types for two
	# different purposes, both exact.
	dest_type = instr.dest.type
	stem = dest_type.stem if isinstance( dest_type, Scalar ) else None
	ctype = c_type( dest_type )
	fctype = c_type( instr.operand.type )
	operand = _emit_operand( instr.operand )
	dest = _emit_operand( instr.dest )
	cmp_min, cmp_max = _float_int_range_bounds( stem, fctype )
	int_min, int_max = _int_min_max_bit_pattern( stem )
	return [
		f'\t{dest} = __metalpy_isnan( {operand} ) ? 0 : '
		f'( ({operand}) < {cmp_min} ? ({ctype}){int_min} : '
		f'( ({operand}) >= {cmp_max} ? ({ctype}){int_max} : ({ctype})({operand}) ) );',
	]

# --- division / modulo (mode-aware, union errors) -------------------------
#
# All division goes through a Result: r==0 -> ZeroDivisionError in every mode.
# The signed INT_MIN/-1 case (UB in C) is where the modes diverge - see
# _emit_int_division. _emit_set_result_err/_ok write into the Result temp,
# setting the inner error-union variant tag when the error type is a union.

def _emit_set_result_ok( dest: str, result_spec: Type, value_expr: str, indent: str ) -> list[str]:
	tag_f, data_f, ok_f, _err_f = _result_tag_data_names( result_spec )
	return [ f'{indent}{dest}.{tag_f} = 0;', f'{indent}{dest}.{data_f}.{ok_f} = {value_expr};' ]

def _emit_set_result_err( dest: str, result_spec: Type, error_stem: str, indent: str ) -> list[str]:
	# set the Err tag; when the error type is a UNION (e.g. ZeroDivisionError|
	# OverflowError), also set the inner union's variant tag for `error_stem`.
	# A single-marker error type has no inner variant to pick.
	#
	# error_stem is always one of the zero-payload arithmetic marker errors
	# (ZeroDivisionError/OverflowError/FloatingPointError - this helper's only
	# 3 call sites, in _emit_int_division/_emit_float_div_check). They carry
	# no real data, but the payload field the union/struct storage still
	# declares for them (a bare RC-pointer slot, like any other error class -
	# see _emit_widen_error's own docstring) was otherwise left uninitialized
	# here. Harmless in isolation (a marker error's payload is never
	# dereferenced), but _emit_widen_error's union-remap switch unconditionally
	# copies EVERY variant's payload alongside its tag (necessarily so, since
	# it's a general mechanism that also serves real user error payloads) -
	# copying that uninitialized pointer value is a genuine, confirmed UB
	# (caught by gcc's UBSan under -O0 as an actual SIGILL, not a false
	# positive - see test_widening_propagates_error). Zero-initializing the
	# payload here, at the one place these markers are ever actually
	# constructed, fixes it at the source: cheap, always well-defined, and
	# makes every later copy of this payload well-defined too.
	tag_f, data_f, _ok_f, err_f = _result_tag_data_names( result_spec )
	lines = [ f'{indent}{dest}.{tag_f} = 1;' ]
	err_type = _result_error_type( result_spec )
	if isinstance( err_type, TaggedUnion ):
		inner_tag, inner_data = _union_tag_data_fields( err_type )
		ordinal, attr = next( ( i, a ) for i, a in enumerate( err_type.attributes ) if a.type is not None and a.type.stem == error_stem )
		lines.append( f'{indent}{dest}.{data_f}.{err_f}.{inner_tag} = {ordinal};' )
		lines.append( f'{indent}{dest}.{data_f}.{err_f}.{inner_data}.{_field_name( f"v_{attr.stem}" )} = 0;' )
	else:
		# a compound-literal zero-init, not a bare `= 0` - err_type is USUALLY
		# one of the built-in RC-pointer markers (NULL-able with a bare 0),
		# but a Result[T,E] can also declare a non-RC (@cstruct-shaped, plain
		# VALUE struct) E, where `= 0` is a real type-mismatch compile error,
		# not just a style nit - (ctype){0} zero-initializes correctly either way
		lines.append( f'{indent}{dest}.{data_f}.{err_f} = ({c_type( err_type )}){{0}};' )
	return lines

def _signed_min_max( stem: str ) -> tuple[str,str]:
	''' (MIN, MAX) C expressions for a signed integer stem. i8..i64/isize use
	stdint macros; i128 has none, so use the two's-complement bit pattern. '''
	if stem in _SATURATE_LIMITS:
		return _SATURATE_LIMITS[stem]
	if stem == 'i128':
		return ( f'(__metalpy_wideint)((__metalpy_wideuint)1 << ({_WIDEINT_TOP_BIT_SHIFT}))',
			f'(__metalpy_wideint)(((__metalpy_wideuint)1 << ({_WIDEINT_TOP_BIT_SHIFT})) - 1)' )
	raise NotImplementedError( f'no MIN/MAX for signed stem {stem!r}' )

def _emit_int_division( instr ) -> list[str]:
	result_spec = instr.dest.type
	ok_type = result_spec.args[0]
	dest = _temp_name( instr.dest.id )
	l, r = _emit_operand( instr.left ), _emit_operand( instr.right )
	is_mod = isinstance( instr, ( ir.Mod, ir.ModWrap, ir.ModSaturate ))
	symbol = '%' if is_mod else '/'
	checked = isinstance( instr, ( ir.Div, ir.Mod ))       # OverflowError on INT_MIN/-1
	wrap = isinstance( instr, ( ir.DivWrap, ir.ModWrap ))  # INT_MIN/-1 inline
	stem = ok_type.stem if isinstance( ok_type, Scalar ) else None
	signed = stem in _SIGNED_TO_UNSIGNED
	ctype = c_type( ok_type )
	lines = [ f'\tif ( ({r}) == 0 ) {{' ]
	lines += _emit_set_result_err( dest, result_spec, 'ZeroDivisionError', '\t\t' )
	if signed:
		# signed INT_MIN / -1 (and INT_MIN % -1) is UB in C - handle it
		# explicitly per mode instead of letting the CPU trap
		min_expr, max_expr = _signed_min_max( stem )
		lines.append( f'\t}} else if ( ({l}) == ({ctype})({min_expr}) && ({r}) == -1 ) {{' )
		if checked:
			lines += _emit_set_result_err( dest, result_spec, 'OverflowError', '\t\t' )
		else:
			# wrap/saturate: a defined inline result. Mod's true answer is 0;
			# Div wraps to INT_MIN, saturates to INT_MAX
			inline = '0' if is_mod else ( min_expr if wrap else max_expr )
			lines += _emit_set_result_ok( dest, result_spec, f'({ctype})({inline})', '\t\t' )
	lines.append( '\t} else {' )
	lines += _emit_set_result_ok( dest, result_spec, f'({l}) {symbol} ({r})', '\t\t' )
	lines.append( '\t}' )
	return lines

def _emit_float_div_check( instr ) -> list[str]:
	# checked/panic float `/`: r==0 -> ZeroDivisionError; else compute and, if
	# the quotient is inf/nan (overflow, or a non-finite operand from a prior
	# wrap-mode block), FloatingPointError. dest.type is Result[T, union].
	result_spec = instr.dest.type
	ok_type = result_spec.args[0]
	ctype = c_type( ok_type )
	dest = _temp_name( instr.dest.id )
	l, r = _emit_operand( instr.left ), _emit_operand( instr.right )
	lines = [ f'\tif ( ({r}) == 0 ) {{' ]
	lines += _emit_set_result_err( dest, result_spec, 'ZeroDivisionError', '\t\t' )
	lines.append( '\t} else {' )
	lines.append( f'\t\t{ctype} __q = ({l}) / ({r});' )
	lines.append( '\t\tif ( __metalpy_isinf( __q ) || __metalpy_isnan( __q ) ) {' )
	lines += _emit_set_result_err( dest, result_spec, 'FloatingPointError', '\t\t\t' )
	lines.append( '\t\t} else {' )
	lines += _emit_set_result_ok( dest, result_spec, '__q', '\t\t\t' )
	lines.append( '\t\t}' )
	lines.append( '\t}' )
	return lines

def _emit_saturate_arith( dest: str, left: ir.Operand, right: ir.Operand, kind: str, dest_type: Type ) -> list[str]:
	stem = dest_type.stem if isinstance( dest_type, Scalar ) else None
	min_c, max_c = _int_min_max_bit_pattern( stem )
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

def _shl_expr( ctype: str, stem: str|None, l: str, r: str ) -> str:
	''' `l << r` computed via an unsigned-cast round-trip when the dest stem is
	signed - a plain signed `l << r` is undefined behavior in C whenever the
	shifted-in bits would overflow the representable range (exactly the case
	ShlCheck/ShlSaturate below need to detect), so the shift itself must happen
	in the unsigned domain (well-defined: modulo wraparound) and get cast back
	to the signed ctype only afterward. Mirrors ShlWrap's own approach just
	below - the same trick, for the same reason. '''
	if stem in _SIGNED_TO_UNSIGNED:
		uctype = _SCALAR_C_TYPES[_SIGNED_TO_UNSIGNED[stem]]
		return f'({ctype})(({uctype})({l}) << ({r}))'
	return f'({l}) << ({r})'

def _emit_shl( instr ) -> list[str]:
	# no __builtin_shl_overflow exists - detect overflow by shifting the
	# result back down and comparing to the original value
	l, r = _emit_operand( instr.left ), _emit_operand( instr.right )
	dest_type = instr.dest.type if not isinstance( instr, ir.ShlCheck ) else instr.dest.type.args[0]
	if isinstance( instr, ir.ShlWrap ):
		ctype = c_type( instr.dest.type )
		stem = instr.dest.type.stem if isinstance( instr.dest.type, Scalar ) else None
		dest = _emit_operand( instr.dest )
		return [ f'\t{dest} = {_shl_expr( ctype, stem, l, r )};' ]
	if isinstance( instr, ir.ShlCheck ):
		ctype = c_type( dest_type )
		stem = dest_type.stem if isinstance( dest_type, Scalar ) else None
		dest = _temp_name( instr.dest.id )
		tag_f, data_f, ok_f, err_f = _result_tag_data_names( instr.dest.type )
		err_ctype = c_type( _result_error_type( instr.dest.type ))
		return [
			'\t{',
			f'\t\t{ctype} __tmp = {_shl_expr( ctype, stem, l, r )};',
			f'\t\tbool __overflow = ( __tmp >> ({r}) ) != ({l});',
			'\t\tif ( __overflow ) {',
			f'\t\t\t{dest}.{tag_f} = 1;',
			f'\t\t\t{dest}.{data_f}.{err_f} = ({err_ctype}){{0}};', # see _emit_set_result_err's comment
			'\t\t} else {',
			f'\t\t\t{dest}.{tag_f} = 0;',
			f'\t\t\t{dest}.{data_f}.{ok_f} = __tmp;',
			'\t\t}',
			'\t}',
		]
	# ShlSaturate
	stem = instr.dest.type.stem if isinstance( instr.dest.type, Scalar ) else None
	_min_c, max_c = _int_min_max_bit_pattern( stem )
	ctype = c_type( instr.dest.type )
	dest = _emit_operand( instr.dest )
	return [
		'\t{',
		f'\t\t{ctype} __tmp = {_shl_expr( ctype, stem, l, r )};',
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
		if _is_float_type( dest_type ):
			return [ f'\t{dest} = -({operand});' ] # IEEE negation: exact, just flips the sign bit (all modes route float `-` through NegWrap)
		if stem in _SIGNED_TO_UNSIGNED:
			uctype = _SCALAR_C_TYPES[_SIGNED_TO_UNSIGNED[stem]]
			return [ f'\t{dest} = ({ctype})(-({uctype})({operand}));' ]
		return [ f'\t{dest} = 0;' ] # unsigned wrap-negation: only 0 maps to itself, everything else wraps to (TYPE_MAX - x + 1) - see note below
	if mode == 'saturate':
		dest = _emit_operand( instr.dest )
		if _is_unsigned_stem( stem ):
			return [ f'\t{dest} = 0;' ] # unsigned negation always saturates to 0 (can never go negative)
		_min_c, max_c = _int_min_max_bit_pattern( stem )
		return [
			'\t{',
			f'\t\t{ctype} __tmp;',
			f'\t\tbool __overflow = __metalpy_sub_overflow( ({ctype})0, ({operand}), &__tmp );',
			f'\t\t{dest} = __overflow ? {max_c} : __tmp;', # negating MIN is the only way signed negation overflows, and it always overflows toward MAX
			'\t}',
		]
	# check
	dest = _temp_name( instr.dest.id )
	tag_f, data_f, ok_f, err_f = _result_tag_data_names( instr.dest.type )
	err_ctype = c_type( _result_error_type( instr.dest.type ))
	return [
		'\t{',
		f'\t\t{ctype} __tmp;',
		f'\t\tbool __overflow = __metalpy_sub_overflow( ({ctype})0, ({operand}), &__tmp );',
		'\t\tif ( __overflow ) {',
		f'\t\t\t{dest}.{tag_f} = 1;',
		f'\t\t\t{dest}.{data_f}.{err_f} = ({err_ctype}){{0}};', # see _emit_set_result_err's comment
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
	callable_ptr = _callable_ptr_type( target_type )
	if callable_ptr is not None:
		fn_type, depth = callable_ptr
		ret, params = _function_pointer_c_type( fn_type )
		ctype = _fn_ptr_cast_type( ret, params, stars = depth )
	else:
		ctype = c_type( target_type )
	if mode == 'wrap':
		# C's own integer conversion rules ARE wrap semantics for an
		# out-of-range value (well-defined, no UB, unlike an ARITHMETIC
		# operation on a signed type triggering signed-overflow UB) - a
		# plain cast is all this needs, no unsigned-roundtrip trick required
		dest = _emit_operand( instr.dest )
		return [ f'\t{dest} = ({ctype})({operand});' ]
	min_c, max_c = _int_min_max_bit_pattern( stem )
	source_stem = instr.operand.type.stem if isinstance( instr.operand.type, Scalar ) else None
	# u128 is the only stem whose own range (0..2**128-1) exceeds
	# __metalpy_wideint's signed positive capacity (0..2**127-1) - as a
	# SOURCE, a large value reinterpreted as signed wraps NEGATIVE,
	# corrupting a wideint-space comparison; as a TARGET, its own MAX
	# (_int_min_max_bit_pattern('u128')'s ~(wideuint)0) reinterpreted as
	# signed becomes -1, equally corrupt. Same-type casts (u128 -> u128)
	# never reach this function, so at most ONE side is ever u128 here.
	if stem == 'u128' and source_stem != 'u128':
		# target is u128, source isn't - no OTHER stem's own range can ever
		# exceed u128's, so the upper bound can never actually fire; only a
		# NEGATIVE source is out of range, checked directly on its own
		# native type (a signed-vs-0 comparison needs no wide promotion at
		# any width) - entirely skipping __metalpy_wide* machinery
		out_of_range = None if source_stem is not None and _is_unsigned_stem( source_stem ) else f'( ({operand}) < 0 )'
		if mode == 'saturate':
			dest = _emit_operand( instr.dest )
			clamp = f'{out_of_range} ? {min_c} : ({ctype})({operand})' if out_of_range else f'({ctype})({operand})'
			return [ f'\t{dest} = {clamp};' ]
		dest = _temp_name( instr.dest.id )
		tag_f, data_f, ok_f, err_f = _result_tag_data_names( instr.dest.type )
		if out_of_range is None:
			return [ f'\t{dest}.{tag_f} = 0;', f'\t{dest}.{data_f}.{ok_f} = ({ctype})({operand});' ]
		err_ctype = c_type( _result_error_type( instr.dest.type ))
		return [
			'\t{',
			f'\t\tbool __overflow = {out_of_range};',
			'\t\tif ( __overflow ) {',
			f'\t\t\t{dest}.{tag_f} = 1;',
			f'\t\t\t{dest}.{data_f}.{err_f} = ({err_ctype}){{0}};', # see _emit_set_result_err's comment
			'\t\t} else {',
			f'\t\t\t{dest}.{tag_f} = 0;',
			f'\t\t\t{dest}.{data_f}.{ok_f} = ({ctype})({operand});',
			'\t\t}',
			'\t}',
		]
	# every other combination (including target u128 paired with source
	# u128, unreachable per the same-type-cast note above) fits safely in
	# __metalpy_wideint, EXCEPT an UNSIGNED source, which needs
	# __metalpy_wideuint instead (non-negative by construction, so its own
	# lower-bound/MIN comparison would misfire if attempted and is skipped
	# entirely - a non-negative source can never actually be "below" any
	# real MIN anyway). Not just u128: any unsigned source's own MAX can
	# exceed __metalpy_wideint's real (backend-dependent - 64-bit under
	# MSVC's fallback, 128-bit under gcc/clang) signed positive capacity,
	# corrupting __wide to negative before any comparison even runs -
	# confirmed for u64/usize specifically (a near-MAX u64 cast down to a
	# narrower signed target silently passed as "in range" under MSVC).
	# _is_unsigned_stem is safe to use unconditionally here even for
	# stems that never actually need it on a given backend (u8/u16/u32,
	# or u64/usize under gcc/clang's true 128-bit wideint) - promoting to
	# wideuint when wideint would have worked anyway gives the identical,
	# correct comparison result either way.
	wide_ctype = '__metalpy_wideuint' if source_stem is not None and _is_unsigned_stem( source_stem ) else '__metalpy_wideint'
	wide_decl = f'{wide_ctype} __wide = ({wide_ctype})({operand});'
	# u64/usize's own MAX (UINT64_MAX/UINTPTR_MAX) doesn't fit as a positive
	# value in a SIGNED __metalpy_wideint once its own width matches the
	# target's - true under MSVC's 64-bit fallback specifically (see the
	# __metalpy_wideint comment near its typedefs), where
	# (__metalpy_wideint)(UINTPTR_MAX) wraps to -1, making a plain
	# "__wide > (wideint)max_c" spuriously true for every non-negative
	# value (confirmed - see lib/builtins/__str.py's case_map, whose own
	# `usize(wide_len)` cast panicked on any string at all this way).
	# Reinterpreting BOTH sides as the unsigned wide type for just the
	# upper-bound comparison sidesteps this: __wide's own bit pattern is
	# unaffected by the reinterpretation, and max_c fits its own unsigned
	# type by definition regardless of width. Scoped to u64/usize only -
	# u128 already has its own, more direct special case above (skips
	# wideint machinery entirely, safe since it's the widest type, nothing
	# else's range can exceed it); u64/usize can't take that same shortcut
	# since a wider i128/u128 SOURCE can legitimately exceed u64/usize's
	# own range and still needs a real overflow check, not just a sign check.
	upper_needs_unsigned_domain = wide_ctype == '__metalpy_wideint' and stem in ( 'u64', 'usize' )
	upper_bound = (
		f'( (__metalpy_wideuint)(__wide) > (__metalpy_wideuint)({max_c}) )'
		if upper_needs_unsigned_domain else
		f'( __wide > ({wide_ctype})({max_c}) )'
	)
	if mode == 'saturate':
		dest = _emit_operand( instr.dest )
		clamp = (
			f'{upper_bound} ? {max_c} : ({ctype})({operand})'
			if wide_ctype == '__metalpy_wideuint' else
			f'( __wide < ({wide_ctype})({min_c}) ) ? {min_c} : {upper_bound} ? {max_c} : ({ctype})({operand})'
		)
		return [
			'\t{',
			f'\t\t{wide_decl}',
			f'\t\t{dest} = {clamp};',
			'\t}',
		]
	# check
	dest = _temp_name( instr.dest.id )
	tag_f, data_f, ok_f, err_f = _result_tag_data_names( instr.dest.type )
	err_ctype = c_type( _result_error_type( instr.dest.type ))
	overflow = (
		upper_bound
		if wide_ctype == '__metalpy_wideuint' else
		f'( __wide < ({wide_ctype})({min_c}) ) || {upper_bound}'
	)
	return [
		'\t{',
		f'\t\t{wide_decl}',
		f'\t\tbool __overflow = {overflow};',
		'\t\tif ( __overflow ) {',
		f'\t\t\t{dest}.{tag_f} = 1;',
		f'\t\t\t{dest}.{data_f}.{err_f} = ({err_ctype}){{0}};', # see _emit_set_result_err's comment
		'\t\t} else {',
		f'\t\t\t{dest}.{tag_f} = 0;',
		f'\t\t\t{dest}.{data_f}.{ok_f} = ({ctype})({operand});',
		'\t\t}',
		'\t}',
	]

def _emit_convert_check( instr: 'ir.ConvertCheck' ) -> list[str]:
	# compiler.checked_convert(T, x) - value-preserving numeric conversion:
	# succeeds iff operand's VALUE fits in target's own [MIN,MAX], entirely
	# independent of bit width. Deliberately NOT the same code path as
	# CastCheck (T(x)/compiler.cast(T,x)'s own check opcode, reached only
	# for a NARROWING conversion - see _lower_scalar_cast's width
	# comparison) - see ir.ConvertCheck's own comment for why these are two
	# separate opcodes despite needing near-identical comparison math. This
	# is a deliberate, close adaptation of _emit_cast's own check-mode
	# branch (same u128/wideint/wideuint/MSVC-fallback edge cases apply
	# here identically - see that function's own extensive comments for
	# the full rationale of each), kept as an independent function rather
	# than a shared helper so the two opcodes' emitters stay independently
	# readable/modifiable.
	target_type = instr.dest.type.args[0]
	operand = _emit_operand( instr.operand )
	stem = target_type.stem if isinstance( target_type, Scalar ) else None
	ctype = c_type( target_type )
	min_c, max_c = _int_min_max_bit_pattern( stem )
	source_stem = instr.operand.type.stem if isinstance( instr.operand.type, Scalar ) else None
	dest = _temp_name( instr.dest.id )
	tag_f, data_f, ok_f, err_f = _result_tag_data_names( instr.dest.type )
	err_ctype = c_type( _result_error_type( instr.dest.type ))

	if stem == 'u128' and source_stem != 'u128':
		out_of_range = None if source_stem is not None and _is_unsigned_stem( source_stem ) else f'( ({operand}) < 0 )'
		if out_of_range is None:
			return [ f'\t{dest}.{tag_f} = 0;', f'\t{dest}.{data_f}.{ok_f} = ({ctype})({operand});' ]
		return [
			'\t{',
			f'\t\tbool __overflow = {out_of_range};',
			'\t\tif ( __overflow ) {',
			f'\t\t\t{dest}.{tag_f} = 1;',
			f'\t\t\t{dest}.{data_f}.{err_f} = ({err_ctype}){{0}};', # see _emit_set_result_err's comment
			'\t\t} else {',
			f'\t\t\t{dest}.{tag_f} = 0;',
			f'\t\t\t{dest}.{data_f}.{ok_f} = ({ctype})({operand});',
			'\t\t}',
			'\t}',
		]

	wide_ctype = '__metalpy_wideuint' if source_stem is not None and _is_unsigned_stem( source_stem ) else '__metalpy_wideint'
	wide_decl = f'{wide_ctype} __wide = ({wide_ctype})({operand});'
	upper_needs_unsigned_domain = wide_ctype == '__metalpy_wideint' and stem in ( 'u64', 'usize' )
	upper_bound = (
		f'( (__metalpy_wideuint)(__wide) > (__metalpy_wideuint)({max_c}) )'
		if upper_needs_unsigned_domain else
		f'( __wide > ({wide_ctype})({max_c}) )'
	)
	overflow = (
		upper_bound
		if wide_ctype == '__metalpy_wideuint' else
		f'( __wide < ({wide_ctype})({min_c}) ) || {upper_bound}'
	)
	return [
		'\t{',
		f'\t\t{wide_decl}',
		f'\t\tbool __overflow = {overflow};',
		'\t\tif ( __overflow ) {',
		f'\t\t\t{dest}.{tag_f} = 1;',
		f'\t\t\t{dest}.{data_f}.{err_f} = ({err_ctype}){{0}};', # see _emit_set_result_err's comment
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
	if obj_type is not None and obj_type.is_rc_pointer(): # PLAN_TUPLE.md: a tuple's backing is always an RCClass, always pointer-accessed
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
	  Ptr[IFoo], because BarImpl's own $vtable is typed IBarVtbl*).

	RCClass (single inheritance, RCClass-subclassing plan Phase 4) follows
	the identical two-case reasoning, just without CStruct's own Ptr[T]
	wrapping - an RCClass receiver's own type IS (a possibly-Specialization-
	wrapped) RCClass directly, never Ptr[T]-of-one, since _self_c_type
	already spells a bare RCClass reference as a pointer in C (see its own
	comment) with no separate metalpy-level Ptr[T] needed to get there. '''
	receiver_text = _emit_operand( receiver )
	target_cls = target.cls
	if isinstance( target_cls, Specialization ) and isinstance( target_cls.base, RCClass ):
		# a monomorphized method whose genericity comes from its own
		# enclosing class (Monomorphizer.monomorphized_function's "class
		# genericity" branch) always carries fn.cls as a Specialization
		# wrapping the ABSTRACT template, never the concrete monomorphized
		# class directly - .monomorphized is that real object, guaranteed
		# already built by now (this exact call's own target was already
		# lowered as a compile unit before this call site could reference
		# it). Without this unwrap, target_cls stayed a bare Specialization
		# here - neither isinstance check below ever matched one, so this
		# fell all the way through to "no cast at all", which is harmless
		# for an ordinary same-class generic call (receiver is already the
		# identical type) but produces an invalid C pointer-type mismatch
		# the moment target_cls and the receiver's own concrete type
		# genuinely differ - e.g. an inherited __init__ found via chain_
		# lookup through a generic ancestor (class Bar[T](Real[T]): pass),
		# where self is a Bar[i32]* but Real[i32].__init__ declares self as
		# Real[i32]* - same idiom as emit_rcclass_instance's own identical
		# unwrap for the same underlying reason.
		concrete = target_cls.monomorphized
		assert isinstance( concrete, RCClass )
		target_cls = concrete
	if isinstance( target_cls, CStruct ) and target_cls.is_interface:
		if target.is_virtual:
			receiver_pointee = receiver.type.args[0] if isinstance( receiver.type, Specialization ) else None
			assert isinstance( receiver_pointee, CStruct ) # every @interface CStruct method's self/receiver is Ptr[T] - see lowering.py's self_param construction
			cast_target = receiver_pointee.vtbl_owner()
		else:
			cast_target = target_cls
		return f'({_self_c_type(cast_target)})({receiver_text})'
	if isinstance( target_cls, RCClass ):
		# unlike the CStruct branch above (always casts, even when
		# redundant - self is ALWAYS Ptr[T] there regardless, so a
		# same-type "cast" is free and every existing test already
		# expects it), RCClass receivers are ordinary values/local
		# variables most of the time, with no subclassing involved at
		# all - skip the cast entirely when it would be a no-op (the
		# receiver's own concrete type already IS cast_target), so an
		# ordinary same-class call stays textually identical to what it
		# always was before RCClass had any cast logic here
		receiver_pointee = receiver.type.base if isinstance( receiver.type, Specialization ) else receiver.type
		assert isinstance( receiver_pointee, RCClass ) # every RCClass receiver is a (possibly Specialization-wrapped) RCClass directly
		cast_target = receiver_pointee.vtbl_owner() if target.is_virtual else target_cls
		if cast_target is receiver_pointee:
			return receiver_text
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
	# re-declared) so only genuine first-time locals trigger it. Keyed by
	# each Variable's own _c_local_name (already uid-suffixed where
	# lowering.py decided that's needed - see Variable.needs_uid_suffix)
	# rather than by bare stem: lowering.py guarantees two DIFFERENT
	# Variable objects only ever render to the SAME name here when
	# they're genuinely meant to share one piece of C storage (an
	# inferred, non-annotated reassignment reusing an existing binding,
	# e.g. across if/elif/else arms) - a same-scope, no-del REdeclaration
	# is rejected as a compile error before it ever reaches emission (see
	# _stmt_AnnAssign's own doc), so this set no longer needs to reason
	# about type compatibility itself.
	declared: set[str] = set()
	if _has_self( function ):
		declared.add( 'self' )
	for p in ( function.parameters or [] ):
		declared.add( _c_local_name( p ))
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
	# for any of THREE shapes: any OrJump with a return_slot; the shared
	# epilogue ladder's own final `ir.Return(value=<__return_value>)`
	# (_emit_epilogue's unconditional tail - the direct, unambiguous
	# signal, not a proxy); OR (kept as a belt-and-suspenders fallback,
	# cheaper to check and still correct whenever it fires) any Label at
	# all whose name starts with 'epilogue' (cfg.py's Epilogue.name is
	# always 'epilogue_N', from _new_label('epilogue')). The Label check
	# ALONE is no longer sufficient on its own - build_epilogue_ladder()
	# now omits an uncaptured entry's own Label entirely (see its own
	# docstring), so a function whose only epilogue entry is reached
	# purely by the fall-off-the-end path with no OTHER return capturing
	# it can have _emit_epilogue() genuinely run (referencing __return_
	# value) while NO Label with 'epilogue' in its name survives anywhere
	# in fn.instructions - confirmed by a real repro (a plain "use of
	# undeclared identifier '__return_value'" compile error) once the
	# Label-omission fix shipped without this. `declared` then makes any
	# actual ir.Assign to it (from the function's own `return <expr>`)
	# just an ordinary re-assignment, not a second declaration
	needs_return_value = any(
		( isinstance( instr, ir.OrJump ) and instr.return_slot is not None )
		or ( isinstance( instr, ir.OrThrow ) and instr.return_slot is not None ) # or_throw()'s own uncovered-leaf fallback, once a defer/errdefer is active - same shape as OrJump
		or ( isinstance( instr, ir.Return ) and isinstance( instr.value, Variable ) and instr.value.stem == '__return_value' )
		or ( isinstance( instr, ir.Label ) and instr.name.startswith( '__epilogue_' ) ) # _new_label('epilogue') -> '__epilogue_N__' - NOT a bare 'epilogue' substring test: _new_label('inline_epilogue') -> '__inline_epilogue_N__' also contains 'epilogue' but is a totally unrelated @inline splice-scope merge label (build_inline_scope_ladder's own, never touches __return_value at all) - a real, confirmed false-positive-triggered -Wunused-variable on a genuinely never-needed __return_value once one of those coexists in the same function with nothing that actually needs the real one
		for instr in fn.instructions
	)
	if needs_return_value and not _returns_void_in_c( function.return_type ):
		name = '__return_value' # bare, not _c_local_name(var) - __return_value's own real Variable object (self._return_value_var) is never independently redeclared/deleted within a function, so needs_uid_suffix is always False for it; a no-Variable-in-hand fallback string is fine here for exactly that reason
		# deliberately NOT zero-initialized (`= {0}`) despite a branch
		# chain compiled from a match/if-elif over every variant of a
		# union (or similarly exhaustive-at-the-metalpy-level shape) being
		# only PROVABLY exhaustive to this compiler's own discovery/type-
		# checking - the emitted C is ordinary if/else-if with no final
		# catch-all else, so clang/MSVC's own (more conservative, per-
		# branch) dataflow analysis can't see that every REACHABLE path
		# already assigned this before the shared "fall off the end"
		# `return __return_value;` ever reads it, and flags -Wsometimes-
		# uninitialized/C4701 - a confirmed false positive (every test in
		# the suite that hits this shape produces the correct, non-zero
		# result). `= {0}` was tried here first and reverted: for a large
		# enough struct/union return type it lowers to a real, CALLED
		# memset() (confirmed via a real link failure - int.__add__'s own
		# Result[i32,OverflowError] triggered it), bypassing this
		# compiler's own extern_libs/no_crt bookkeeping entirely (the C
		# compiler inserts the call on its own, invisibly, well after
		# metalpy's own emission), so a no-CRT build (no memset available
		# at all) fails to link. See CcTool.compile()'s own
		# -Wno-sometimes-uninitialized/-Wno-uninitialized/wd4701 for the
		# actual (diagnostic-suppression, zero behavior-risk) fix instead
		lines.append( f'\t{_declarator( function.return_type, name )};' )
		declared.add( name )
	lines.extend( _emit_instructions( fn.instructions, function = function, declared = declared ))
	if not function.is_destructor:
		# a parameter whose body never reads it (self included - e.g.
		# UnsafeList._read_element, whose is_rc(T) branch only ever touches
		# its slot argument; or an ordinary parameter kept only for a
		# uniform call-site/overload shape) still has to be declared -
		# dropping it from the C signature would make it a different
		# function shape per instantiation/overload, and every call site
		# already passes it uniformly. (void)param silences -Wunused-
		# parameter without an attribute (MSVC doesn't support
		# __attribute__ and doesn't warn on this by default anyway).
		# `(?!\$)` matters now that a genuine local sharing a parameter's
		# stem gets a '$uid' suffix (_c_local_name) - without it, a bare
		# `\bname\b` search would count as "used" merely by matching the
		# unsuffixed PREFIX of that unrelated local's own suffixed
		# occurrence (e.g. parameter `x` against a later `del x; x: T2 =
		# ...`-redeclared local emitted as `x$7` - see del_reuse_and_
		# emitter_naming_bug). Destructors are exempted: their only
		# "parameter" is __obj, never named self in the C signature itself
		# (self is a real local, cast from __obj, just above)
		body_text = '\n'.join( lines[1:] )
		void_marks: list[str] = []
		if _has_self( function ) and not _name_used_as_value( 'self', body_text ):
			void_marks.append( 'self' )
		for p in ( function.parameters or [] ):
			name = _c_local_name( p )
			if not _name_used_as_value( name, body_text ):
				void_marks.append( name )
		for name in reversed( void_marks ):
			lines.insert( 1, f'\t(void){name};' )
	lines.append( '}' )
	return '\n'.join( lines )

def _emit_instructions( instructions: list[ir.Instruction], *, function: Function|None, declared: set[str] ) -> list[str]:
	''' walks a straight-line instruction list, emitting each via
	_emit_instruction. A plain loop is enough - unlike an earlier version of
	this function, PLAN_THREAD_SAFE_SHARED_STATE.md's Part A no longer
	needs to be reconstructed here by pattern-matching adjacent
	instructions: cfg.py's assign() and lowering.py's _cfg_assign now emit
	real ir.AcquireGlobalLock/ir.ReleaseGlobalLock markers directly around
	whatever a protected global's read or write actually expands to
	(NOT always a single bare Incref/Decref - a union-typed global, e.g.
	ZoneInfo|None, decrefs/increfs via a multi-instruction tag-check+
	extract sequence, see cfg.py's own _refcount_instructions), so
	_emit_instruction can just turn each marker into the matching acquire/
	release call directly, with nothing to look ahead for. '''
	lines: list[str] = []
	for instr in instructions:
		lines.extend( _emit_instruction( instr, function = function, declared = declared ))
	return lines

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
			# otherwise emit `return $t2;` from a function declared void -
			# see _returns_void_in_c's own comment. instr.value's own
			# defining instruction (DeclareTemp+GetAttr/Call/whatever, not
			# necessarily an ir.Assign - _mark_used_if_none's other call
			# sites don't cover every shape) already ran; mark it read here
			# so discarding it doesn't turn that already-emitted definition
			# into -Wunused-variable/-Wunused-but-set-variable
			return _mark_used_if_none( instr.value ) + [ '\treturn;' ]
		return [ f'\treturn {_emit_operand(instr.value)};' ]

	if isinstance( instr, ir.Yield ):
		# PLAN_GENERATORS.md Phase F - see ir.Yield's own docstring: a
		# generator's $$__next__ is never void, never the entry point, so
		# this is unconditionally the same shape as ir.Return's own
		# simplest branch - no need to replicate its other special cases
		return [ f'\treturn {_emit_operand(instr.value)};' ]

	if isinstance( instr, ir.DeclareTemp ):
		# dedupe against the SAME `declared` set ir.DeclareLocal/ir.Assign
		# already use below - a defer(...)/errdefer(...) block is lowered
		# ONCE (_register_defer_block) but its captured ir.Instructions,
		# including any ir.DeclareTemp for a temp an implicit cast needed
		# (e.g. sys.free(ptr: Ptr[u8]) needing Ptr[u8]->Ptr[None] on a
		# non-Windows target - see lib/sys.py's two free() overloads), get
		# replayed BY IDENTITY (same ir.Temp object) at every return/unwind
		# point (cfg.py's _replay()/build_epilogue_ladder()) into this
		# function's one flat, goto-based C scope (no per-branch { }
		# blocks) - so a function with 2+ return points can emit the SAME
		# `void* $tN;` declaration multiple times, a genuine gcc compile
		# error ("redeclaration ... with no linkage") that clang/MSVC never
		# hit (Windows's sys.free(Ptr[u8]) overload needs no cast at all,
		# so the temp/DeclareTemp this bug depends on is never created
		# there). Safe to skip a repeat declaration: temp ids come from a
		# monotonic per-function counter never reused for two logically
		# distinct temps, so the ONLY way the same id's DeclareTemp repeats
		# is this replay duplication; the duplicated return/unwind paths
		# are mutually exclusive at runtime (only one executes per call),
		# and each replay's own duplicated cast/assignment re-initializes
		# the shared C variable immediately before its own use.
		name = _temp_name( instr.temp.id )
		if name in declared:
			return []
		declared.add( name )
		return [ f'\t{_declarator( instr.temp.type, name )};' ]
	if isinstance( instr, ir.DeleteTemp ):
		return [] # C block scoping already handles temp lifetime - nothing to emit
	if isinstance( instr, ir.DeclareLocal ):
		# see ir.DeclareLocal's own docstring - a bare declaration, no
		# initializer, emitted flat wherever this instruction itself sits
		# (guaranteed by lowering.py to be a genuinely flat/unconditional
		# point - never nested inside one of THIS module's own hand-emitted
		# C `{ }` blocks)
		name = _c_local_name( instr.variable )
		declared.add( name )
		return [ f'\t{_declarator( instr.variable.type, name, volatile = instr.variable.is_volatile )};' ]
	if isinstance( instr, ir.Assign ):
		src = _emit_operand( instr.src )
		if isinstance( instr.dest, Variable ) and not instr.dest.is_global:
			name = _c_local_name( instr.dest )
			if name not in declared:
				declared.add( name )
				return [ f'\t{_declarator( instr.dest.type, name, volatile = instr.dest.is_volatile )} = {src};' ] + _mark_used_if_none( instr.dest )
			return [ f'\t{name} = {src};' ] + _mark_used_if_none( instr.dest )
		# a global Variable is declared separately at file scope (Phase 7 -
		# emit_global) - never re-declared here, only assigned
		return [ f'\t{_emit_operand(instr.dest)} = {src};' ] + _mark_used_if_none( instr.dest )

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

	if isinstance( instr, ( ir.Div, ir.Mod, ir.DivWrap, ir.DivSaturate, ir.ModWrap, ir.ModSaturate )):
		# integer division/modulo - always zero-checked (ZeroDivisionError);
		# signed INT_MIN/-1 handled per mode (checked -> OverflowError; wrap ->
		# INT_MIN; saturate -> INT_MAX; mod -> 0). See _emit_int_division.
		return _emit_int_division( instr )

	if isinstance( instr, ir.FloatDivCheck ):
		return _emit_float_div_check( instr )

	if isinstance( instr, ir.FloatDiv ):
		# wrap/saturate-mode float `/`: raw IEEE, no zero-check (x/0.0 -> inf,
		# 0.0/0.0 -> nan, produced silently). checked/panic-mode float `/` uses
		# ir.FloatDivCheck above instead (-> ZeroDivisionError|FloatingPointError)
		dest = _emit_operand( instr.dest )
		l, r = _emit_operand( instr.left ), _emit_operand( instr.right )
		return [ f'\t{dest} = ({l}) / ({r});' ]

	if type( instr ) in _FLOAT_CHECK_SYMBOL:
		return _emit_float_check_arith( instr )

	if isinstance( instr, ir.FloatCastCheck ):
		return _emit_float_cast_check( instr )

	if isinstance( instr, ir.FloatToIntClamp ):
		return _emit_float_to_int_clamp( instr )

	if type( instr ) in _PLAIN_BITWISE_SYMBOL:
		dest = _emit_operand( instr.dest )
		symbol = _PLAIN_BITWISE_SYMBOL[type(instr)]
		return [ f'\t{dest} = ({_emit_operand(instr.left)}) {symbol} ({_emit_operand(instr.right)});' ]

	if isinstance( instr, ir.PtrDiff ):
		# raw byte distance, isize - same uintptr_t round-trip _emit_wrap_arith
		# already uses for pointer +/-, just signed and with no dest-type cast
		# back to a pointer type (dest_type is isize here, not Ptr[T])
		dest = _emit_operand( instr.dest )
		l, r = _emit_operand( instr.left ), _emit_operand( instr.right )
		return [ f'\t{dest} = (intptr_t)((uintptr_t)({l}) - (uintptr_t)({r}));' ]

	if isinstance( instr, ir.Invert ):
		return [ f'\t{_emit_operand(instr.dest)} = ~({_emit_operand(instr.operand)});' ]
	if isinstance( instr, ir.Not ):
		return [ f'\t{_emit_operand(instr.dest)} = !({_emit_operand(instr.operand)});' ]
	if isinstance( instr, ir.MarkUsed ):
		return [ f'\t(void){_emit_operand(instr.operand)};' ]
	if type( instr ) in _NEG_MODE:
		return _emit_neg( instr )
	if type( instr ) in _CAST_MODE:
		return _emit_cast( instr )
	if isinstance( instr, ir.ConvertCheck ):
		return _emit_convert_check( instr )

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
			target_name = mangle_function_qualname( instr.target )
		# destructor calls sys.free(self) — self is struct Foo*, free takes
		# void*; C needs an explicit cast since the two are different types
		arg_texts = list( _emit_call_args( instr ))
		if function is not None and function.is_destructor and target_name == mangle_qualname( 'sys.free' ):
			for i, a in enumerate( arg_texts ):
				arg_texts[i] = f'(void*)({a})'
		if instr.target.is_virtual:
			# vtable dispatch, not a direct call - _emit_self_operand already
			# put the (possibly re-cast) receiver pointer at arg_texts[0].
			# receiver is always a pointer (Ptr[T] for @interface CStruct,
			# or an RCClass reference, which is already pointer-shaped in C
			# - see _self_c_type), so vtable access is always ->
			# (_member_access_operator), never . directly. WHERE that field
			# lives differs by class kind: CStruct's own $vtable is a plain
			# top-level struct member, already typed to the exact Vtbl type
			# this call needs (see emit_cstruct/_interface_vtbl_name), no
			# cast needed. RCClass's own vtable pointer lives INSIDE $header
			# (ObjectHeader.vtable - see the PROLOGUE's own comment),
			# reusing the field destructor dispatch already needed rather
			# than adding a second one - but ObjectHeader is the SAME
			# embedded struct in EVERY RCClass, so $header.vtable's own
			# declared C type can only ever be the one shared, minimal
			# __metalpy_ObjectVtbl* (unlike CStruct's own per-class $vtable
			# field, which can be typed differently per class) - reaching a
			# REAL slot (anything past `destroy`) needs an explicit cast
			# back to the receiver's own vtbl_owner()'s real (wider) Vtbl
			# type first, the same cast _emit_self_operand already computes
			# for the self argument, just applied to the vtable pointer too
			assert instr.receiver is not None # is_virtual only ever set on real instance methods - see discovery.py's _parse_function
			slot_name = _field_name( instr.target.stem )
			vtable_op = _member_access_operator( instr.receiver.type )
			# WHERE the vtable pointer lives, not whether there is one - both
			# arms have a vtable. An RCClass reads it out of its ObjectHeader
			# ($header.vtable, reusing the field destructor dispatch already
			# needed); an @interface CStruct has a plain top-level $vtable
			# member instead. has_object_header() is exactly that distinction.
			if instr.target.cls is not None and instr.target.cls.has_object_header():
				receiver_pointee = instr.receiver.type.base if isinstance( instr.receiver.type, Specialization ) else instr.receiver.type
				assert isinstance( receiver_pointee, RCClass )
				vtbl_type = _rcclass_vtbl_type_name( receiver_pointee )
				vtable_expr = f'((const {vtbl_type}*)({_emit_operand(instr.receiver)}){vtable_op}$header.vtable)'
				call_expr = f'{vtable_expr}->{slot_name}( {", ".join(arg_texts)} )'
			else:
				call_expr = f'({_emit_operand(instr.receiver)}){vtable_op}$vtable->{slot_name}( {", ".join(arg_texts)} )'
		else:
			has_args = bool( arg_texts )
			call_expr = f'{target_name}( {", ".join(arg_texts)} )' if has_args else f'{target_name}()'
		# _returns_void_in_c, not just `instr.dest is not None`: a generic
		# method's declared return type can be a real, non-None type T
		# that just happens to RESOLVE to NoneType for THIS
		# monomorphization (e.g. UnsafeDict._owned_value's V, for a
		# dict[K, None]) - lowering.py still creates a real dest temp for
		# it (the declared type isn't literally the bare `None` annotation
		# lowering.py's OWN NoneType-return special-casing checks
		# elsewhere), but the CALLEE's own C function is void
		# (_returns_void_in_c already makes it so, for both its prototype
		# and its own `return;` - see that helper's own comment).
		# Assigning `dest = <call to a void function>;` anyway is a
		# straight C type error. Mirrors that exact existing rule rather
		# than inventing a new one.
		if instr.dest is not None and not _returns_void_in_c( instr.target.return_type ):
			return [ f'\t{_emit_operand(instr.dest)} = {call_expr};' ]
		if instr.dest is not None:
			# dest is a real declared MetalpyNone temp (see the comment above)
			# but the callee's own C function is void, so it can't be assigned
			# FROM the call expression - give it a defined placeholder value
			# separately instead of leaving it genuinely uninitialized. A
			# NoneType temp's bit value never carries information, but it can
			# still be READ later (e.g. _coerce_into_union wrapping it into a
			# Result[None,E].Ok(...) payload) - an uninitialized read there is
			# real UB, not just a harmless value-doesn't-matter case: confirmed
			# via MSVC's /RTC1 uninitialized-variable runtime check aborting
			# (STATUS_BREAKPOINT) on exactly this shape (a void-returning
			# method wrapped to return Result[None,E]).
			return [ f'\t{call_expr};', f'\t{_emit_operand(instr.dest)} = 0;' ]
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

	if isinstance( instr, ir.AddrOfField ):
		# compiler.addrof(x.field) - one flat C expression, &(obj)OP field -
		# see AddrOfField's own docstring for why this is a distinct
		# instruction from AddrOf(GetAttr(...)) (that would take the
		# address of a freshly loaded COPY, not the real field)
		op = _member_access_operator( instr.obj.type )
		return [ f'\t{_emit_operand(instr.dest)} = &({_emit_operand(instr.obj)}){op}{_field_name(instr.attr)};' ]

	if isinstance( instr, ir.ArrayFieldPtr ):
		# compiler.addrof(x.field) where field is a FixedArrayType - one
		# flat C expression, (obj)OP field, deliberately with NO leading &
		# (see ArrayFieldPtr's own docstring: a real C array member decays
		# to a pointer to its first element on use - &-ing it would give a
		# pointer-TO-array instead, a different, mismatched C type)
		op = _member_access_operator( instr.obj.type )
		return [ f'\t{_emit_operand(instr.dest)} = ({_emit_operand(instr.obj)}){op}{_field_name(instr.attr)};' ]

	if isinstance( instr, ir.AddrOfArrayIndex ):
		# compiler.addrof(x.field[i]) - one flat C expression,
		# &(obj)OP field[index] - see AddrOfArrayIndex's own docstring
		op = _member_access_operator( instr.obj.type )
		return [ f'\t{_emit_operand(instr.dest)} = &(({_emit_operand(instr.obj)}){op}{_field_name(instr.attr)}[{_emit_operand(instr.index)}]);' ]

	if isinstance( instr, ir.GetAttrIndex ):
		# f.arr[i] - one flat C expression, (obj)OP field[index] - see
		# GetAttrIndex's own docstring for why this targets the field
		# directly rather than composing GetAttr+GetItem
		op = _member_access_operator( instr.obj.type )
		return [ f'\t{_emit_operand(instr.dest)} = ({_emit_operand(instr.obj)}){op}{_field_name(instr.attr)}[{_emit_operand(instr.index)}];' ]
	if isinstance( instr, ir.SetAttrIndex ):
		op = _member_access_operator( instr.obj.type )
		return [ f'\t({_emit_operand(instr.obj)}){op}{_field_name(instr.attr)}[{_emit_operand(instr.index)}] = {_emit_operand(instr.value)};' ]


	if isinstance( instr, ir.SizeOf ):
		# a real class-like type's size is whatever the C compiler itself
		# computes for its struct/union body (sizeof(struct Foo), never
		# sizeof(struct Foo*) - _value_spelling gives the bare body type
		# even for an RCClass) - no field-layout algorithm exists earlier
		# in this compiler, nor should one
		return [ f'\t{_emit_operand(instr.dest)} = sizeof({_value_spelling(instr.type)});' ]

	# debug-mode alloc-site tracking (dump_live_objects) - see
	# _prologue_debug_ops. These are only ever lowered when compiler.target.
	# debug folds true (lib/sys.py's own `if compiler.target.debug:` guards
	# every call site - see compile_time_transformer), so _target_debug is
	# always True here too; no separate release-mode branch needed.
	if isinstance( instr, ir.DebugRawTrack ):
		return [ f'\t__metalpy_debug_raw_track( (void*){_emit_operand(instr.ptr)}, (size_t){_emit_operand(instr.size)} );' ]
	if isinstance( instr, ir.DebugRawUntrack ):
		return [ f'\t__metalpy_debug_raw_untrack( (void*){_emit_operand(instr.ptr)} );' ]
	if isinstance( instr, ir.DumpLiveObjects ):
		return [ '\t__metalpy_dump_live_objects();' ]
	if isinstance( instr, ir.DebugUntrackRC ):
		# compiler.__raw_free__'s own debug-mode cleanup - see its own
		# comment (Lowering._lower_compiler_raw_free) on why this can't just
		# rely on release_object's normal untrack
		return [ f'\t__metalpy_debug_untrack( &(({_emit_operand(instr.value)})->$header.debug_link) );' ]

	if isinstance( instr, ir.Incref ):
		# a plain cast, not &(value)->$header - $header is always the FIRST
		# member of every RCClass struct (emit_rcclass's own field-flattening,
		# same fact ir.DecrefDynamic below already relies on), so the two are
		# equivalent addresses for a non-null value, but &ptr->field is UB in C
		# when ptr is null (forming an lvalue through a null pointer via `->`,
		# independent of whether it's ever dereferenced afterward) - a real,
		# reachable case here: a "zero-payload marker" RC-typed union leaf
		# (e.g. the built-in ZeroDivisionError/OverflowError/FloatingPointError
		# - see _emit_widen_error's own docstring) is never actually allocated,
		# so its payload pointer is always NULL, and the union-aware tag-gated
		# Incref/Decref emission in cfg.py legitimately reaches this case.
		# retain_object/release_object already null-guard internally, but that
		# guard never gets a chance to matter if computing the argument itself
		# is already UB - confirmed by gcc's UBSan catching a genuine SIGILL
		# here (see test_widening_propagates_error)
		return [ f'\tretain_object( (ObjectHeader*)({_emit_operand(instr.value)}) );' ]
	if isinstance( instr, ir.Decref ):
		# see ir.Incref's own comment just above for why this is a plain cast.
		# Not lock-aware here (or in the Incref branch above) - a protected
		# global's Incref/Decref is always bracketed by real
		# ir.AcquireGlobalLock/ir.ReleaseGlobalLock markers cfg.py/
		# lowering.py emit directly (see those branches below), which is
		# what actually turns into the acquire/release calls; adding a
		# second layer of locking here would double-lock a non-reentrant
		# SRWLOCK (PLAN_THREAD_SAFE_SHARED_STATE.md's B.4).
		return [ f'\trelease_object( (ObjectHeader*)({_emit_operand(instr.value)}) );' ]
	if isinstance( instr, ir.AcquireGlobalLock ):
		if not ( instr.var.reassigned_outside_init and _global_lock_supported() ):
			# either this global turns out to never actually be reassigned
			# anywhere (a read-side marker emitted before that fact was
			# known - see cfg.py's own is_alias branch) or this platform
			# doesn't support the lock yet (POSIX - see
			# _global_lock_supported's own docstring) - a pure no-op either
			# way, not a partial/best-effort lock
			return []
		return [ _global_lock_acquire( _global_lock_name( instr.var ))]
	if isinstance( instr, ir.ReleaseGlobalLock ):
		if not ( instr.var.reassigned_outside_init and _global_lock_supported() ):
			return []
		return [ _global_lock_release( _global_lock_name( instr.var ))]
	if isinstance( instr, ir.AcquireFieldLock ):
		if not _global_lock_supported():
			return [] # unsupported target - see _global_lock_supported's own docstring
		return [ f'\tacquire_field_lock( (ObjectHeader*)({_emit_operand(instr.obj)}) );' ]
	if isinstance( instr, ir.ReleaseFieldLock ):
		if not _global_lock_supported():
			return []
		return [ f'\trelease_field_lock( (ObjectHeader*)({_emit_operand(instr.obj)}) );' ]
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
	if isinstance( instr, ir.MarkUnused ):
		return [ f'\t(void){_emit_operand(instr.value)};' ]
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

	if isinstance( instr, ir.FormatFloat ):
		# __metalpy_format_f64 lives in PROLOGUE, not behind @extern - see
		# ir.FormatFloat's own comment on why (fixed-arity-only extern codegen
		# can't safely reach a genuinely variadic snprintf/_snprintf, and an
		# @extern('c', ...) tag would wrongly flip the no-crt Windows build)
		return [
			f'\t{_emit_operand(instr.dest)} = __metalpy_format_f64('
			f'(char*){_emit_operand(instr.buf)}, {_emit_operand(instr.size)}, {_emit_operand(instr.precision)}, '
			f'{_emit_operand(instr.type_char)}, {_emit_operand(instr.alt)}, {_emit_operand(instr.value)});'
		]

	if isinstance( instr, ir.IsNan ):
		return [ f'\t{_emit_operand(instr.dest)} = __metalpy_isnan( {_emit_operand(instr.value)} );' ]
	if isinstance( instr, ir.IsInf ):
		return [ f'\t{_emit_operand(instr.dest)} = __metalpy_isinf( {_emit_operand(instr.value)} );' ]

	if isinstance( instr, ir.ParseFloat ):
		return [ f'\t{_emit_operand(instr.dest)} = __metalpy_parse_f64( (const char*){_emit_operand(instr.buf)} );' ]

	if isinstance( instr, ir.Allocate ):
		# has_object_header, not is_rc_pointer: this branch writes
		# $header.ref_count and wires $header.vtable, which only exists on a
		# type that actually LEADS with an ObjectHeader. A TupleType is an RC
		# pointer but is never allocated under its own annotation - its
		# synthesized backing RCClass is what reaches here, and that answers
		# True on its own behalf.
		if instr.cls is not None and instr.cls.has_object_header():
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
			# $$vtable instance's own NAME is mangled straight from
			# instr.dest.type (Specialization or not) - same as
			# _rcclass_destructor_name's own identical direct use, since a
			# monomorphized generic RCClass's qualname is deliberately set
			# equal to its own Specialization's qualname (Monomorphizer.
			# monomorphize_class), so mangling either one produces the
			# SAME name. But deciding whether a CAST is needed (does this
			# class have any REAL @virtual slot) needs the actual resolved
			# RCClass object, not the abstract generic template
			# Specialization.base would give - .monomorphized is that
			# object once Monomorphizer has actually built it (guaranteed
			# by now: lowering.py's _lower_allocate_fields already forced
			# this exact construction through monomorphize_class/
			# _ensure_resolved before ever emitting this Allocate)
			concrete_cls = instr.dest.type.monomorphized if isinstance( instr.dest.type, Specialization ) else instr.dest.type
			assert isinstance( concrete_cls, RCClass )
			vtable_ref = f'&{mangle_type(instr.dest.type)}$$vtable'
			if concrete_cls.virtual_slots():
				vtable_ref = f'(const __metalpy_ObjectVtbl*){vtable_ref}'
			lines.append( f'\t({dest})->$header.vtable = {vtable_ref};' )
			# PLAN_THREAD_SAFE_SHARED_STATE.md Part B: sys.alloc's own
			# allocation is never zero-initialized (mempoison in debug,
			# raw malloc in release - same reason $header.ref_count/.vtable
			# above are explicit writes, not relied-on zero-init) - the new
			# per-object lock field needs the identical explicit
			# construction-time init every other ObjectHeader field gets.
			# Windows: an all-zero SRWLOCK IS a valid unlocked lock (A.3's
			# own established fact, reused verbatim here) - a plain zero
			# write suffices, no real init call. POSIX: pthread_mutex_t has
			# no such zero-init guarantee - a genuine pthread_mutex_init()
			# call is required at every single construction site.
			if _global_lock_supported():
				if _target_uses_pthread_lock():
					lines.append( f'\tpthread_mutex_init( &(({dest})->$header.lock), ((void*)0) );' )
				else:
					lines.append( f'\t({dest})->$header.lock = ((void*)0);' )
			if _target_debug:
				# debug-mode alloc-site tracking (dump_live_objects) -
				# instr.loc is stamped centrally by Lowering._emit
				loc = instr.loc or '<unknown>'
				lines.append( f'\t({dest})->$header.alloc_loc = {_c_string_literal(loc.encode("utf-8"))};' )
				lines.append( f'\t({dest})->$header.alloc_size = sizeof(*({dest}));' )
				lines.append( f'\t__metalpy_debug_track( &__metalpy_debug_list_head, &(({dest})->$header.debug_link) );' )
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
		return _emit_or_return( instr, function, declared )
	if isinstance( instr, ir.OrJump ):
		return _emit_or_jump( instr )
	if isinstance( instr, ir.OrThrow ):
		assert function is not None, 'ir.OrThrow requires the enclosing function (for its own uncovered-leaf return-type fallback)'
		return _emit_or_throw( instr, function, declared )
	if isinstance( instr, ir.Raise ):
		assert function is not None, 'ir.Raise requires the enclosing function (for its own uncovered-leaf return-type fallback)'
		return _emit_raise( instr, function, declared )
	if isinstance( instr, ir.WidenResult ):
		return _emit_widen_result( instr )
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
		] + _mark_used_if_none( instr.dest )
	if isinstance( instr, ir.UnwrapOr ):
		value = _emit_operand( instr.value )
		tag_f, data_f, ok_f, _err_f = _result_tag_data_names( instr.value.type )
		return [ f'\t{_emit_operand(instr.dest)} = ( ({value}).{tag_f} == 1 ) ? {_emit_operand(instr.default)} : ({value}).{data_f}.{ok_f};' ] + _mark_used_if_none( instr.dest )

	raise NotImplementedError( f'_emit_instruction: unsupported instruction {instr!r} (later-phase work)' )

def _emit_or_return( instr: ir.OrReturn, function: Function, declared: set[str] ) -> list[str]:
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
	# the error may need WIDENING: value's error type (E_op) can be narrower
	# than the function's declared error union (E_fn) - see _emit_widen_error
	e_op = _result_error_type( instr.value.type )
	e_fn = _result_error_type( return_type )
	# instr.epilogue (see ir.OrReturn's own docstring) is only ever non-empty
	# when `value` is a named, tracked Variable whose own ordinary scope-exit
	# decref must be excluded here (its payload is being moved into __err,
	# unretained, right below) while every OTHER still-live binding/defer/
	# errdefer obligation still needs its normal cleanup on this early-exit
	# path - emitted via the same per-instruction dispatcher as the rest of the
	# function body, so it can contain anything return_() can produce (Decref,
	# tag-gated GetAttr/Cmp/JumpIfFalse/Jump/Label sequences, defer replays)
	epilogue_lines = _emit_instructions( instr.epilogue, function = function, declared = declared )
	if instr.inline_exit is not None:
		# PLAN_INLINE.md early-return generalization - this OrReturn is
		# .or_return()/checked-arithmetic's own inline-unwind path reached
		# from inside a multi-statement @inline splice: `function` here is
		# the CALLER's real, enclosing C function (splicing puts everything
		# in ONE emitted function) - NOT the inlined target - so ret_ctype/
		# e_fn must come from result_var's own type (the target's real
		# return type) instead of function.return_type, or the widened
		# __err would be built with the wrong struct shape/error union
		# entirely. No real `return` here - stow into result_var, arm
		# exited_flag, `goto` merge_label instead (see ir.OrReturn's own
		# inline_exit docstring)
		result_var, exited_flag, merge_label = instr.inline_exit
		inline_ret_ctype = c_type( result_var.type )
		inline_e_fn = _result_error_type( result_var.type )
		result_c = _emit_operand( result_var )
		flag_c = _emit_operand( exited_flag )
		return [
			f'\tif ( ({value}).{tag_f} == 1 ) {{',
			f'\t\t{inline_ret_ctype} __err;',
			f'\t\t__err.{tag_f} = 1;',
			*_emit_widen_error( f'__err.{data_f}.{err_f}', inline_e_fn, f'({value}).{data_f}.{err_f}', e_op ),
			*epilogue_lines,
			f'\t\t{result_c} = __err;',
			f'\t\t{flag_c} = true;',
			f'\t\tgoto {_c_label(merge_label)};',
			'\t}',
			f'\t{dest} = ({value}).{data_f}.{ok_f};',
		] + _mark_used_if_none( instr.dest )
	return [
		f'\tif ( ({value}).{tag_f} == 1 ) {{',
		f'\t\t{ret_ctype} __err;',
		f'\t\t__err.{tag_f} = 1;',
		*_emit_widen_error( f'__err.{data_f}.{err_f}', e_fn, f'({value}).{data_f}.{err_f}', e_op ),
		*epilogue_lines,
		'\t\treturn __err;',
		'\t}',
		f'\t{dest} = ({value}).{data_f}.{ok_f};',
	] + _mark_used_if_none( instr.dest )

def _emit_widen_result( instr: ir.WidenResult ) -> list[str]:
	# a bare `return x` widening x's own Result[T,NarrowE] into dest's wider
	# Result[T,WideE] (see ir.WidenResult's own docstring) - unlike OrReturn
	# (which only ever transforms the Err branch, since its own Ok branch
	# means "continue executing", not "return"), a bare return exits
	# unconditionally on EITHER branch, so BOTH get built into dest here: Ok
	# is a plain field copy (same T on both sides - nothing to widen), Err
	# reuses the exact same _emit_widen_error helper OrReturn/OrJump already
	# use for their own Err branch.
	src = _emit_operand( instr.src )
	dest = _emit_operand( instr.dest )
	tag_f, data_f, ok_f, err_f = _result_tag_data_names( instr.src.type )
	e_op = _result_error_type( instr.src.type )
	e_fn = _result_error_type( instr.dest.type )
	return [
		f'\tif ( ({src}).{tag_f} == 0 ) {{',
		f'\t\t{dest}.{tag_f} = 0;',
		f'\t\t{dest}.{data_f}.{ok_f} = ({src}).{data_f}.{ok_f};',
		'\t} else {',
		f'\t\t{dest}.{tag_f} = 1;',
		*_emit_widen_error( f'{dest}.{data_f}.{err_f}', e_fn, f'({src}).{data_f}.{err_f}', e_op ),
		'\t}',
	]

def _emit_or_jump( instr: ir.OrJump ) -> list[str]:
	value = _emit_operand( instr.value )
	dest = _emit_operand( instr.dest )
	tag_f, data_f, ok_f, err_f = _result_tag_data_names( instr.value.type )
	lines = [ f'\tif ( ({value}).{tag_f} == 1 ) {{' ]
	if instr.return_slot is not None:
		slot = _emit_operand( instr.return_slot )
		# widen value's error (E_op) into the return slot's error union (E_fn)
		# if they differ - same as _emit_or_return
		e_op = _result_error_type( instr.value.type )
		e_fn = _result_error_type( instr.return_slot.type )
		lines.append( f'\t\t{slot}.{tag_f} = 1;' )
		lines.extend( _emit_widen_error( f'{slot}.{data_f}.{err_f}', e_fn, f'({value}).{data_f}.{err_f}', e_op ))
	if instr.exited_flag is not None:
		# PLAN_INLINE.md early-return generalization - see ir.OrJump's own
		# exited_flag docstring: armed alongside return_slot whenever
		# `target` is a multi-statement @inline splice's own local label,
		# so the splice's own ladder tail can tell early exit apart from
		# normal fallthrough
		lines.append( f'\t\t{_emit_operand(instr.exited_flag)} = true;' )
	lines.append( f'\t\tgoto {_c_label(instr.target)};' )
	lines.append( '\t}' )
	lines.append( f'\t{dest} = ({value}).{data_f}.{ok_f};' )
	lines.extend( _mark_used_if_none( instr.dest ))
	return lines

def _emit_leaf_dispatch_case(
	dispatch: 'list[ir.ThrowLeaf]', leaf_type: Type, payload_expr: str,
	epilogue: 'list[ir.Instruction]', target: str|None, return_slot: 'Variable|None',
	function: Function, declared: set[str], inline_exit: 'tuple[Variable,Variable,str]|None' = None,
) -> list[str]:
	''' the body of ONE leaf's own `case`/single-leaf branch inside
	_emit_or_throw/_emit_raise - either dispatches into an except handler
	(a leaf covered by `dispatch`) or falls back to exactly OrReturn's/
	OrJump's own propagate-to-caller shape (an uncovered leaf, target is
	None/a real epilogue label respectively), or (inline_exit set) a multi-
	statement @inline splice's own local result_var/exited_flag/merge_label
	- see ir.OrThrow's and ir.Raise's own docstrings. Matched by qualname,
	same convention _union_member/_atomic_leaves already use. Shared by both
	instructions (extracted from the pre-ir.Raise, OrThrow-only version of
	this function) - `dispatch`/`epilogue`/`target`/`return_slot`/
	`inline_exit` are identically-shaped fields on both.

	Every case body here lives inside its OWN switch-case `{ }` block (see
	_emit_or_throw/_emit_raise) - an except-clause bind Variable is
	therefore declared by the CALLER, before the switch, not here: `except
	(A, B) as e:` produces one dispatch entry per leaf, each landing in a
	DIFFERENT case block, and a C local declared inside one case's own
	block scope isn't visible from another - only a hoisted, shared
	declaration lets both cases assign into the same `e`. '''
	entry = next( ( d for d in dispatch if d.leaf.qualname == leaf_type.qualname ), None )
	if entry is not None:
		lines: list[str] = []
		if entry.bind is not None:
			bind_c = _c_local_name( entry.bind )
			if entry.bind.type.qualname == leaf_type.qualname:
				lines.append( f'\t\t{bind_c} = {payload_expr};' ) # plain leaf, or `except T as e:` with a single covered leaf
			else:
				# `except (A, B) as e:` - e's own type is the anonymous A|B
				# union (see lowering.py's _stmt_Try) - widen this ONE
				# leaf's payload into it, same shape _emit_widen_error's own
				# single-class branch uses
				assert isinstance( entry.bind.type, TaggedUnion ), f'{entry.bind.type!r}: except-clause bind is neither the single leaf nor a union of them'
				ordinal, battr = _union_member( entry.bind.type, leaf_type )
				btag_f, bdata_f = _union_tag_data_fields( entry.bind.type )
				bfield = _field_name( f'v_{battr.stem}' )
				lines.append( f'\t\t{bind_c}.{btag_f} = {ordinal};' )
				lines.append( f'\t\t{bind_c}.{bdata_f}.{bfield} = {payload_expr};' )
		lines.append( f'\t\tgoto {_c_label(entry.label)};' )
		return lines
	# uncovered leaf - propagate to the enclosing function exactly like
	# OrReturn (target is None) or OrJump (target is a real epilogue
	# label) already do, widening just THIS leaf (not the whole error
	# union) into the target error type - see _emit_widen_error's own
	# single-class branch, reached here because leaf_type itself is never
	# a union (each dispatch entry/switch case is already one atomic leaf).
	# Field NAMES (tag_f/data_f/err_f) are the same fixed strings for any
	# Result[_,_] specialization (_result_tag_data_names' own docstring) -
	# reading them off function.return_type/return_slot.type here (rather
	# than instr.value.type, which for ir.Raise isn't even a Result) is
	# equally valid.
	epilogue_lines = _emit_instructions( epilogue, function = function, declared = declared )
	if inline_exit is not None:
		# PLAN_INLINE.md early-return generalization, same shape as
		# _emit_or_return's own inline_exit handling: `function` here is the
		# CALLER's real, enclosing C function (splicing puts everything in
		# ONE emitted function) - result_var's own type (the splice TARGET's
		# real return type), not function.return_type, is what the widened
		# __err must be built as. No real return/goto-to-a-real-epilogue-
		# label here - stow into result_var, arm exited_flag, goto merge_label
		result_var, exited_flag, merge_label = inline_exit
		inline_ret_ctype = c_type( result_var.type )
		inline_e_fn = _result_error_type( result_var.type )
		tag_f, data_f, _ok_f, err_f = _result_tag_data_names( result_var.type )
		result_c = _emit_operand( result_var )
		flag_c = _emit_operand( exited_flag )
		return [
			'\t\t{',
			f'\t\t{inline_ret_ctype} __err;',
			f'\t\t__err.{tag_f} = 1;',
			*_emit_widen_error( f'__err.{data_f}.{err_f}', inline_e_fn, payload_expr, leaf_type ),
			*epilogue_lines,
			f'\t\t{result_c} = __err;',
			f'\t\t{flag_c} = true;',
			f'\t\tgoto {_c_label(merge_label)};',
			'\t\t}',
		]
	if target is None:
		# own `{ }` block (not just the surrounding switch-case's, which
		# _emit_raise's single-class/non-union caller doesn't even have -
		# it emits this bare, with no enclosing `if`/`switch` at all): two
		# separate uncovered `raise`s of the SAME single-class error type
		# reaching the same C scope (e.g. two @inline splices of the same
		# target, or two sibling top-level raises) would otherwise both
		# declare `__err` in that one shared scope - a real "redefinition
		# of '__err'" clang error, confirmed by a real repro (two @inline
		# splice call sites of the same raise() target in one function)
		ret_ctype = c_type( function.return_type )
		tag_f, data_f, _ok_f, err_f = _result_tag_data_names( function.return_type )
		e_fn = _result_error_type( function.return_type )
		return [
			'\t\t{',
			f'\t\t{ret_ctype} __err;',
			f'\t\t__err.{tag_f} = 1;',
			*_emit_widen_error( f'__err.{data_f}.{err_f}', e_fn, payload_expr, leaf_type ),
			*epilogue_lines,
			'\t\treturn __err;',
			'\t\t}',
		]
	assert return_slot is not None
	slot = _emit_operand( return_slot )
	tag_f, data_f, _ok_f, err_f = _result_tag_data_names( return_slot.type )
	e_fn = _result_error_type( return_slot.type )
	return [
		'\t\t{',
		f'\t\t{slot}.{tag_f} = 1;',
		*_emit_widen_error( f'{slot}.{data_f}.{err_f}', e_fn, payload_expr, leaf_type ),
		*epilogue_lines,
		f'\t\tgoto {_c_label(target)};',
		'\t\t}',
	]

def _declare_dispatch_binds( dispatch: 'list[ir.ThrowLeaf]', declared: set[str] ) -> list[str]:
	''' every except-clause bind referenced by `dispatch` is declared ONCE,
	at the function's own FLAT scope (never inside an `if`/switch-case
	block) - the handler body itself (referencing this bind) is emitted
	separately, at a `goto`-reached Label OUTSIDE those nested blocks
	entirely (see lowering.py's _stmt_Try: body/handlers all share one
	flat per-function scope, goto-separated, same convention every other
	construct in this module already follows), so a declaration confined
	to a switch-case's own C block scope would already be out of scope by
	the time the handler body reads it - confirmed by a real "use of
	undeclared identifier" compile error under clang before this hoist.
	Shared by _emit_or_throw and _emit_raise. Deduped by id() (Variable is
	an unhashable plain dataclass) while keeping a stable (first-seen)
	order - a tuple except-clause's several dispatch entries share the
	SAME bind object. '''
	seen_bind_ids: set[int] = set()
	binds = []
	for d in dispatch:
		if d.bind is not None and id( d.bind ) not in seen_bind_ids:
			seen_bind_ids.add( id( d.bind ))
			binds.append( d.bind )
	lines: list[str] = []
	for bind in binds:
		bind_c = _c_local_name( bind )
		if bind_c not in declared:
			declared.add( bind_c )
			lines.append( f'\t{_declarator( bind.type, bind_c )};' )
	return lines

def _emit_or_throw( instr: 'ir.OrThrow', function: Function, declared: set[str] ) -> list[str]:
	# Result.or_throw() - like _emit_or_return/_emit_or_jump's own `if
	# (value.tag==1) {...}` shape, but the Err branch first dispatches per
	# LEAF of the error type (see ir.OrThrow's own docstring): a plain
	# opaque error type (E is not itself a union) has exactly one leaf, no
	# switch needed at all; a real anonymous error union switches on its
	# own runtime tag, one case per member, each independently either
	# jumping into a handler or falling back to ordinary propagation.
	value = _emit_operand( instr.value )
	dest = _emit_operand( instr.dest )
	tag_f, data_f, ok_f, err_f = _result_tag_data_names( instr.value.type )
	e_op = _result_error_type( instr.value.type )
	err_expr = f'({value}).{data_f}.{err_f}'
	lines: list[str] = _declare_dispatch_binds( instr.dispatch, declared )
	lines.append( f'\tif ( ({value}).{tag_f} == 1 ) {{' )
	if isinstance( e_op, TaggedUnion ) and e_op.file is None:
		op_tag, op_data = _union_tag_data_fields( e_op )
		lines.append( f'\t\tswitch ( ({err_expr}).{op_tag} ) {{' )
		for i, op_attr in enumerate( e_op.attributes ):
			payload_expr = f'({err_expr}).{op_data}.{_field_name(f"v_{op_attr.stem}")}'
			lines.append( f'\t\t\tcase {i}: {{' )
			lines.extend( '\t' + l for l in _emit_leaf_dispatch_case(
				instr.dispatch, op_attr.type, payload_expr, instr.epilogue, instr.target, instr.return_slot, function, declared, instr.inline_exit,
			))
			lines.append( '\t\t\t}' ) # every case body above ends in goto/return - no break needed, never falls through
		lines.append( '\t\t}' )
	else:
		lines.extend( _emit_leaf_dispatch_case(
			instr.dispatch, e_op, err_expr, instr.epilogue, instr.target, instr.return_slot, function, declared, instr.inline_exit,
		))
	lines.append( '\t}' )
	lines.append( f'\t{dest} = ({value}).{data_f}.{ok_f};' )
	lines.extend( _mark_used_if_none( instr.dest ))
	return lines

def _emit_raise( instr: 'ir.Raise', function: Function, declared: set[str] ) -> list[str]:
	# `raise EXPR` - same per-LEAF dispatch shape as _emit_or_throw's own
	# Err branch, minus the outer Result tag check and the Ok-arm
	# extraction: `value` here IS the error already, unconditionally taken
	# (see ir.Raise's own docstring) - there's no "if (value.tag==1)"
	# wrapper at all, just the leaf switch (or, for a single-class error
	# type, straight into the one leaf case) directly.
	value = _emit_operand( instr.value )
	e_op = instr.value.type
	lines: list[str] = _declare_dispatch_binds( instr.dispatch, declared )
	if isinstance( e_op, TaggedUnion ) and e_op.file is None:
		op_tag, op_data = _union_tag_data_fields( e_op )
		lines.append( f'\tswitch ( ({value}).{op_tag} ) {{' )
		for i, op_attr in enumerate( e_op.attributes ):
			payload_expr = f'({value}).{op_data}.{_field_name(f"v_{op_attr.stem}")}'
			lines.append( f'\t\tcase {i}: {{' )
			lines.extend( _emit_leaf_dispatch_case(
				instr.dispatch, op_attr.type, payload_expr, instr.epilogue, instr.target, instr.return_slot, function, declared, instr.inline_exit,
			))
			lines.append( '\t\t}' ) # every case body above ends in goto/return - no break needed, never falls through
		lines.append( '\t}' )
	else:
		lines.extend( _emit_leaf_dispatch_case(
			instr.dispatch, e_op, value, instr.epilogue, instr.target, instr.return_slot, function, declared, instr.inline_exit,
		))
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
		lines.append( f'\t{_declarator(field_type, _field_name(field_name))};' )
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
	lines = [
		f'static const uint8_t {data_name}[] = {_c_string_literal(data_bytes)};',
	]
	extra_field_lines: list[str] = []
	if qualname == 'builtins.str':
		# str also caches __char_count/__index (lib/builtins/__init__.py's
		# str._from_owned_cstr) - a literal is baked directly here rather
		# than going through that runtime construction path, so it must
		# independently bake the SAME cached metadata. Computed in Python at
		# compile time instead of C: a Python str's own len()/iteration is
		# already the Unicode codepoint sequence, so no UTF-8 decoding is
		# needed here (unlike the runtime scan, which has to decode). Same
		# entries=(byte_size>>8)+1 sizing formula as the runtime path, so a
		# literal's __index is indistinguishable in shape from a runtime-
		# constructed str's - __getitem__ doesn't know or care which built it.
		assert isinstance( value, str )
		index_name = f'{name}$index'
		byte_size = len( data_bytes )
		entries = ( byte_size >> 8 ) + 1
		offsets = [ 0 ] * entries
		byte_offset = 0
		for i, ch in enumerate( value ):
			if ( i & 0xFF ) == 0:
				offsets[ i >> 8 ] = byte_offset
			byte_offset += len( ch.encode( 'utf-8' ))
		lines.append( f'static const uintptr_t {index_name}[] = {{ {", ".join(str(o) for o in offsets)} }};' )
		extra_field_lines = [
			f'\t.{_field_name("__char_count")} = {len(value)},',
			f'\t.{_field_name("__index")} = (uintptr_t*){index_name},',
		]
	lines += [
		f'static struct {struct_name} {name} = {{',
		f'\t.$header = {{ .ref_count = METALPY_IMMORTAL_REFCOUNT }},',
		f'\t.{_field_name(data_field)} = {data_name},',
		f'\t.{_field_name(len_field)} = {len(data_bytes)},',
		*extra_field_lines,
		'};',
	]
	return lines

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

def _vtable_slot_referenced_classlikes( owner: RCClass|CStruct ) -> list[ClassLike]:
	''' every RCClass/CStruct/CUnion/TaggedUnion referenced (as a return or
	parameter type, unwrapping a Specialization to its own .base) by any of
	owner's own virtual_slots() signatures - in encounter order, deduped.
	Even a still-unfulfilled slot (@abstractmethod, never gets a real
	vtable INSTANCE of its own) still contributes its OWN declared
	signature to owner's shared Vtbl STRUCT TYPE, which is unconditional -
	needed by any future concrete override's own real instance regardless
	of whether one exists yet. A type reachable ONLY through such a slot,
	with no concrete override ever actually compiled anywhere in THIS
	particular program, can otherwise go completely unscheduled - a real,
	confirmed -Wvisibility ("will not be visible outside of this
	function"), since nothing else ever independently forward-tags it
	(compiler.rcclasses/cstructs/cunions/tagged_unions, each unconditionally
	forward-tagged in emit_c's own pass 1, only ever contain types that got
	SCHEDULED as a real compile unit somewhere - a type ONLY ever named in
	an abstract slot's signature never does). '''
	found: list[ClassLike] = []
	for slot in owner.virtual_slots():
		if slot.resolve is not None:
			slot.resolve()
		for t in [ slot.return_type ] + [ p.type for p in ( slot.parameters or [] ) ]:
			base = t.base if isinstance( t, Specialization ) else t
			if isinstance( base, ( RCClass, CStruct, CUnion, TaggedUnion )) and base not in found:
				found.append( base )
	return found

def _vtable_slot_c_type( owner: RCClass|CStruct, slot: Function ) -> tuple[str,list[str]]:
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
		params.append( _declarator( p.type, _c_local_name( p )))
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
	# not every class reachable enough to get a full body emitted is ever
	# actually constructed by THIS program (e.g. only reached through a
	# subclass's own Allocate, or merely type-referenced) - unlike PROLOGUE's
	# retain_object/etc (see emit_c), telling "constructed" from "not" here
	# means matching every ir.Allocate site one at a time, real dead-code
	# elimination rather than a cheap instruction-kind check - not worth it
	# for a warning; see __metalpy_maybe_unused's own comment
	if field_inits:
		return f'__metalpy_maybe_unused static const {vtbl_type} {instance_name} = {{ {", ".join(field_inits)} }};'
	return f'__metalpy_maybe_unused static const {vtbl_type} {instance_name} = {{0}};'

# --- RCClass vtable dispatch (RCClass-subclassing plan, Phase 4) -----------
#
# Generalizes the CStruct machinery just above, but written as its own
# parallel set of functions rather than forcing everything through one
# mega-generalized set of branches - too much of the RCClass shape genuinely
# differs from CStruct's own COM-interface model to share code cleanly:
# - CStruct's own vtable field ($vtable) is a plain top-level struct member;
#   RCClass's own vtable pointer lives INSIDE $header (ObjectHeader.vtable -
#   see the PROLOGUE's own comment), reusing the field that already existed
#   for dynamic destructor dispatch rather than adding a second one.
# - Every @interface CStruct always has at least IUnknown's own 3 real
#   slots, so it always needs its own synthesized Vtbl type. Most RCClasses
#   have NO @virtual methods at all (still the common case) - those need no
#   synthesized type whatsoever, just an instance of the one shared,
#   built-in __metalpy_ObjectVtbl (see the PROLOGUE) - so "does this class
#   need its own Vtbl struct type" is a real, RCClass-specific question
#   CStruct's own functions never had to ask.
# - A class that DOES need its own type always leads with `destroy` (this
#   class's own real, per-class destructor - no CStruct/COM analog), so
#   every RCClass's own real vtable type stays a prefix-compatible superset
#   of __metalpy_ObjectVtbl regardless of how many @virtual methods it adds -
#   this is what lets ObjectHeader.vtable stay declared as the one shared,
#   minimal type and still be read correctly (const __metalpy_ObjectVtbl*)
#   through ANY concrete class's own actual (possibly larger) vtable.
# - Every concrete RCClass gets a real $$vtable instance, unconditionally -
#   unlike CStruct's own "None if this class doesn't build one" (only
#   @interface classes ever get one there), RCClass's own destructor
#   dispatch NEEDS one for every single instance, virtual methods or not.
#   No "unfulfilled slot" (stub-body) skip either - @abstractmethod is
#   Phase 5's job, not this one's; see compiler.py's _schedule_rcclass_
#   vtable_impls, which this stays in lockstep with.

def _rcclass_vtbl_type_name( cls: RCClass ) -> str:
	''' the C type name for cls's own $header.vtable field - the shared,
	global __metalpy_ObjectVtbl when NO @virtual method exists anywhere in
	cls's own chain (the common, zero-extra-synthesis case), otherwise
	cls.vtbl_owner()'s own synthesized (destroy-prefixed) type - mirrors
	_interface_vtbl_name plus this one additional "nothing real to
	dispatch, just use the shared minimal type" case CStruct never needs. '''
	if not cls.virtual_slots():
		return '__metalpy_ObjectVtbl'
	return f'{mangle_type( cls.vtbl_owner() )}Vtbl'

def emit_rcclass_vtbl_struct( owner: RCClass ) -> str:
	''' the full vtable struct body for an RCClass that IS its own
	vtbl_owner() (owner.own_new_virtual_slots() is non-empty) - always
	starts with `destroy` (see this section's own header comment on why),
	then one function-pointer field per REAL @virtual slot
	(owner.virtual_slots(), declaration order) - mirrors
	emit_interface_vtbl_struct plus the destroy prefix. Only ever called
	for a class with at least one real slot; see _rcclass_vtbl_type_name's
	own "shared type" case for a class with none. '''
	name = f'{mangle_type( owner )}Vtbl'
	lines = [ f'typedef struct {name} {{', '\tvoid (*destroy)( void* );' ]
	if _target_debug:
		# must stay a literal PREFIX match with __metalpy_ObjectVtbl's own
		# {destroy, type_name} layout (see that type's own comment) - every
		# concrete class's real vtable is read through a base-typed
		# __metalpy_ObjectVtbl* lens in release_object/dump_live_objects
		lines.append( '\tconst char* type_name;' )
	for slot in owner.virtual_slots():
		ret, params = _vtable_slot_c_type( owner, slot )
		lines.append( f'\t{ret} (*{_field_name(slot.stem)})( {", ".join(params)} );' )
	lines.append( f'}} {name};' )
	return '\n'.join( lines )

def _rcclass_fulfilled_slot_impls( cls: RCClass ) -> list[Function]|None:
	''' the ordered per-slot implementing Function for a CONCRETE RCClass
	(nearest override in cls's own chain, or the root's own declaration
	if never overridden) - None if ANY slot is unfulfilled (@abstractmethod
	- RCClass-subclassing plan Phase 5) anywhere in the chain, meaning cls
	is an abstract role that never gets a static vtable instance at all -
	mirrors CStruct's own _interface_fulfilled_slot_impls exactly, just
	keyed on the explicit is_abstract marker instead of stub-body
	inference (matches compiler.py's _schedule_rcclass_vtable_impls and
	lowering.py's _check_rcclass_fully_implemented construction-time check
	- all three have to agree on exactly which classes are "complete", so
	this recomputes the identical walk rather than trusting a cache that
	could drift out of sync). '''
	impls: list[Function] = []
	for slot in cls.virtual_slots():
		impl = cls.chain_lookup( slot.stem )
		if not isinstance( impl, Function ):
			return None
		if impl.resolve is not None:
			impl.resolve()
		if impl.is_abstract:
			return None
		impls.append( impl )
	return impls

def emit_rcclass_vtable_instance( cls: RCClass ) -> str|None:
	''' the static const vtable instance for a fully-fulfilled concrete
	RCClass - unlike CStruct's own @interface-only opt-in, EVERY non-
	abstract RCClass gets one (destructor dispatch needs one
	unconditionally - see this section's own header comment), not just
	ones with @virtual methods. Slot 0 is always destroy (this class's
	own real, per-class destructor - see _rcclass_destructor_name); any
	further slots are this class's own REAL @virtual slots, each wired
	directly to its actual implementing Function with a function-pointer
	cast, same "no wrapper/trampoline needed" reasoning as
	emit_interface_vtable_instance. None (no instance built) if cls has
	any unfulfilled (@abstractmethod) slot anywhere in its chain - see
	_rcclass_fulfilled_slot_impls; an abstract class can never be directly
	constructed (lowering.py's own construction-time check already
	guarantees this), so nothing ever needs to read its own vtable
	instance - only a concrete, fully-implemented subclass's instance
	ever actually gets wired into $header.vtable. Always typed via
	_rcclass_vtbl_type_name(cls) when an instance IS built. '''
	slot_impls = _rcclass_fulfilled_slot_impls( cls )
	if slot_impls is None:
		return None
	vtbl_type = _rcclass_vtbl_type_name( cls )
	instance_name = f'{mangle_type(cls)}$$vtable'
	field_inits = [ f'.destroy = {_rcclass_destructor_name(cls)}' ]
	if _target_debug:
		field_inits.append( f'.type_name = {_c_string_literal(cls.qualname.encode("utf-8"))}' )
	slots = cls.virtual_slots()
	if slots:
		owner = cls.vtbl_owner()
		for slot, impl in zip( slots, slot_impls ):
			ret, params = _vtable_slot_c_type( owner, slot )
			field_inits.append( f'.{_field_name(slot.stem)} = ({_fn_ptr_cast_type(ret, params)}){mangle_qualname(impl.qualname)}' )
	# see emit_interface_vtable_instance's identical comment: not every
	# concrete RCClass reachable enough to get a full body is actually
	# constructed by this particular program
	return f'__metalpy_maybe_unused static const {vtbl_type} {instance_name} = {{ {", ".join(field_inits)} }};'

def emit_cstruct( cls: CStruct ) -> str:
	attrs: list[tuple[str,Type,int|None]]
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
			lines.append( f'\t{_declarator(field_type, _field_name(field_name))};' )
		lines.append( '};' )
		return '\n'.join( lines )
	attrs = [ ( attr.stem, attr.type, attr.c_align ) for attr in cls.attributes ]
	return _struct_or_union_body( mangle_type( cls ), 'struct', attrs, packed = cls.packed )

def emit_cunion( cls: CUnion ) -> str:
	attrs = [ ( attr.stem, attr.type, attr.c_align ) for attr in cls.attributes ]
	return _struct_or_union_body( mangle_type( cls ), 'union', attrs, packed = cls.packed )

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
		# a member reference (OSError.FileNotFoundError) always constant-
		# folds to a bare ir.Const at lowering time (see lowering.py's
		# _expr_Attribute) - this symbol itself is never referenced by any
		# compiled program, so it's unconditionally -Wunused-const-variable-
		# eligible on GCC/Clang; see __metalpy_maybe_unused's own comment
		lines.append( f'__metalpy_maybe_unused static const {name} {name}${key} = {value};' )
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
	tag_attr = union.get_local_or_raise( 'tag' )
	data_attr = union.get_local_or_raise( 'data' )
	assert isinstance( tag_attr, Variable ) and isinstance( data_attr, Variable ), \
		f'{union.qualname}: _tagged_union_storage has not run yet - no real storage shape to emit'
	name = mangle_type( union )
	return _struct_or_union_body( name, 'struct', [ ( tag_attr.stem, tag_attr.type, None ), ( data_attr.stem, data_attr.type, None ) ] )

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

def _is_zero_const( op: ir.Operand ) -> bool:
	return isinstance( op, ir.Const ) and ( op.value is None or op.value == 0 )

def _global_init_is_all_zero_value_type( instructions: list[ir.Instruction] ) -> bool:
	# a WIDER "doesn't need its init function called at all" check than
	# _is_trivial_global_init above (which only covers a SINGLE bare-Const
	# Assign) - this covers a value-type (CStruct, never RCClass - no
	# sys.alloc/Call/Incref/Decref appears in a pure value-type construction)
	# whose EVERY field is a zero/null Const, e.g. `case_folder: CaseFolding
	# = CaseFolding(upper_table=None, upper_count=0, ...)`. Such an
	# initializer lowers to `$t0 = (struct CaseFolding){0,0,0,0}; global =
	# $t0;` - a struct-copy-of-an-all-zero-literal the C compiler is free to
	# implement via memset/memcpy (confirmed by a REAL LNK2019 "unresolved
	# external symbol memset" failure under a no-CRT Windows build once this
	# global's own init function actually got called - see
	# PLAN_GLOBAL_INIT.md). Skipping the call entirely is exactly correct
	# here, not just a workaround: the global's own C11 `{0}` static
	# zero-initializer (emit_global, above) is ALREADY byte-for-byte
	# equivalent to what this instruction sequence would compute - unlike a
	# global with any NON-zero constant field (still correctly gets a real
	# call) or any RCClass/runtime-computed one (never matches this check at
	# all, since ir.Allocate for an RCClass is preceded by a real
	# sys.alloc(...) ir.Call).
	filtered = [ i for i in instructions if not isinstance( i, ( ir.DeclareTemp, ir.DeleteTemp )) ]
	if not filtered:
		return True
	*rest, last = filtered
	if not ( isinstance( last, ir.Assign ) and isinstance( last.src, ir.Temp ) ):
		return False
	for instr in rest:
		if isinstance( instr, ir.Allocate ):
			# an RCClass Allocate performs a REAL sys.alloc(...) heap
			# allocation at C-emission time (see _emit_instruction's own
			# ir.Allocate/RCClass branch) - never skippable, regardless of
			# how many fields it has (a zero-field or default-constructed
			# RCClass, e.g. a stateless singleton or a fieldless subclass,
			# vacuously satisfies the all-zero-fields check below otherwise,
			# which would wrongly skip the real allocation entirely,
			# leaving the global's own pointer permanently NULL - confirmed
			# via a real access-violation crash, a subclass singleton with
			# no fields of its own calling a virtual method through it)
			if isinstance( instr.cls, RCClass ):
				return False
			if not all( _is_zero_const( v ) for v in instr.fields.values() ):
				return False
		elif isinstance( instr, ir.Assign ) and isinstance( instr.src, ( ir.Const, ir.Temp ) ):
			continue
		else:
			return False
	return True

def _global_init_fn_name( g: LoweredGlobal ) -> str:
	# shared by emit_global (which defines this function) and emit_c's own
	# __metalpy_init() synthesis (which calls it) - factored out so the two
	# can't drift on the naming scheme (see PLAN_GLOBAL_INIT.md)
	return f'__metalpy_init_{mangle_qualname( g.variable.qualname )}'

def _referenced_global_qualnames( instructions: list[ir.Instruction] ) -> set[str]:
	''' every OTHER global Variable's qualname a global's own init
	instructions reference anywhere an ir.Operand can appear (Assign.src,
	Call.args/kwargs/receiver, GetAttr.obj, Allocate.fields, ...one field
	name per ir.Instruction subclass and growing). Walked GENERICALLY via
	dataclasses.fields() rather than hand-enumerating every instruction
	kind's own operand-bearing field(s), so a new instruction shape can
	never silently go unwalked - this only ever looks for Temp/Const/
	Variable/FunctionRef leaves (ir.Operand's own union, mirrored here
	structurally) inside a field, a list, or a dict; anything else (a
	Type, a ClassLike, a Function, a plain str/bool/enum - e.g. Allocate.
	cls, Call.target, SizeOf.type) is neither, so it's inert here, not
	something this needs its own case for. '''
	found: set[str] = set()
	def walk( value: object ) -> None:
		if isinstance( value, Variable ):
			if value.is_global:
				found.add( value.qualname )
			return
		if isinstance( value, ( ir.Temp, ir.Const, ir.FunctionRef )):
			return # real Operand leaves, just not globals - nothing further to walk
		if isinstance( value, list ):
			for item in value:
				walk( item )
		elif isinstance( value, dict ):
			for item in value.values():
				walk( item )
	for instr in instructions:
		for field in dataclasses.fields( instr ):
			walk( getattr( instr, field.name ))
	return found

def _transitive_global_reads_by_function( compiler: Compiler ) -> dict[int,set[str]]:
	''' id(Function) -> every OTHER module global's qualname that function's
	body reads, either directly (_referenced_global_qualnames) or via any
	function it calls, transitively - closing the exact gap
	_referenced_global_qualnames itself can't (see its own docstring: it
	only walks the instructions handed to it, and a global's own init
	instructions are typically just `call _build_value()`, never _build_
	value's OWN body - so `VALUE: T = _build_value()` where _build_value()
	reads another global was invisible to _topologically_sort_globals
	entirely, a real reachable "accepted but miscompiles" bug: the read
	got scheduled before the write, at best silently wrong (a Scalar
	reads its {0}-initialized zero value) and at worst a crash (an RC
	container's own lock/refcount machinery never constructed, dereferenced
	as if it had been - see the bug report this closes).
	Keyed by id(Function), not qualname - Function.qualname isn't unique
	across overloads. A worklist/fixed-point pass (not a single DFS), so
	mutual recursion between called functions converges correctly instead
	of a naive memo-with-recursion-guard silently dropping edges hit while
	still on the call stack. '''
	instructions_by_fn_id = { id( lf.function ): lf.instructions for lf in compiler.functions }
	closure: dict[int,set[str]] = {
		fn_id: _referenced_global_qualnames( instrs ) for fn_id, instrs in instructions_by_fn_id.items()
	}
	calls: dict[int,set[int]] = { fn_id: set() for fn_id in instructions_by_fn_id }
	for fn_id, instrs in instructions_by_fn_id.items():
		for instr in instrs:
			if isinstance( instr, ir.Call ):
				calls[fn_id].add( id( instr.target ))
	changed = True
	while changed:
		changed = False
		for fn_id, callee_ids in calls.items():
			for callee_id in callee_ids:
				callee_reads = closure.get( callee_id )
				if not callee_reads or callee_reads <= closure[fn_id]:
					continue
				closure[fn_id] |= callee_reads
				changed = True
	return closure

def _referenced_global_qualnames_transitive(
	instructions: list[ir.Instruction], fn_closure: dict[int,set[str]],
) -> set[str]:
	# a global's own init instructions PLUS, for every function it calls,
	# that function's own transitive closure (_transitive_global_reads_by_
	# function) - see that function's docstring for why the direct-only
	# walk above isn't enough on its own
	found = set( _referenced_global_qualnames( instructions ))
	for instr in instructions:
		if isinstance( instr, ir.Call ):
			found |= fn_closure.get( id( instr.target ), set())
	return found

def _topologically_sort_globals( compiler: Compiler ) -> list[LoweredGlobal]:
	''' compiler.globals in TypeResolver's own FIFO scheduling order (first-
	referenced-while-lowering-reachable-code) has no relationship to which
	global's own initializer reads which OTHER global's value - confirmed
	as a real, reachable bug (not a theoretical one), found while proving
	out PLAN_GLOBAL_INIT.md's own deferred "true dependency-ordering
	between globals" limitation with a real test: `b: Foo = Foo.make(a.x)`
	with main() only ever referencing `b` schedules b BEFORE a (main
	reaches b first; a is only discovered as b's own dependency, so it's
	enqueued and lowered strictly later) - emit_c's own globals loop would
	then emit b's declaration before a's, a straight C "undeclared
	identifier" compile error (see _emit_global_declaration/_emit_global_
	init_fn's own split, which independently fixes THAT half - declaration
	order needs no dependency sort at all once every declaration is a
	self-contained {0}-or-constant). This function fixes the OTHER half:
	the RUNTIME call order inside __metalpy_init() - b's own init function
	dereferences a's global pointer, which must already be constructed
	(non-null, real fields) by the time b's own init function runs.

	Only globals that actually GET a call at all (excludes both
	_is_trivial_global_init - already fully initialized by C's own static
	initializer semantics before ANY function runs, so it's never a real
	ordering dependency regardless of who references it - and _global_
	init_is_all_zero_value_type, which never gets a call either way) enter
	the graph. A stable Kahn's-algorithm topological sort, seeded in
	compiler.globals' own original order (so two globals with no
	dependency relationship at all keep their original relative order,
	same determinism guarantee an unordered dependency set would otherwise
	lose) - not a DFS-based sort, specifically so an unsatisfiable real
	CYCLE (two globals whose own initializers each read the other's value
	- fundamentally impossible to order, unlike a merely circular IMPORT
	or a circular class reference, both already confirmed fine elsewhere)
	is detected structurally (leftover nodes with no zero-indegree node
	to pick) rather than by recursion-depth crashing or silently emitting
	SOME arbitrary order that compiles but runs one of the two against an
	uninitialized value. '''
	callable_globals = [
		g for g in compiler.globals
		if not _is_trivial_global_init( g.instructions ) and not _global_init_is_all_zero_value_type( g.instructions )
	]
	by_qualname = { g.variable.qualname: g for g in callable_globals }
	# edges[a] = globals that must be called AFTER a (a's own qualname ->
	# the set of dependents reading a's value); indegree[b] counts how many
	# not-yet-called prerequisites b still has
	edges: dict[str,set[str]] = { g.variable.qualname: set() for g in callable_globals }
	indegree: dict[str,int] = { g.variable.qualname: 0 for g in callable_globals }
	fn_closure = _transitive_global_reads_by_function( compiler )
	for g in callable_globals:
		for dep_qualname in _referenced_global_qualnames_transitive( g.instructions, fn_closure ):
			if dep_qualname == g.variable.qualname or dep_qualname not in by_qualname:
				continue # self-reference, or a dependency that never gets a call itself (trivial/all-zero) - no edge needed either way
			if g.variable.qualname not in edges[dep_qualname]:
				edges[dep_qualname].add( g.variable.qualname )
				indegree[g.variable.qualname] += 1
	ready = [ g for g in callable_globals if indegree[g.variable.qualname] == 0 ]
	ordered: list[LoweredGlobal] = []
	while ready:
		g = ready.pop( 0 )
		ordered.append( g )
		for dependent_qualname in edges[g.variable.qualname]:
			indegree[dependent_qualname] -= 1
			if indegree[dependent_qualname] == 0:
				ready.append( by_qualname[dependent_qualname] )
	if len( ordered ) != len( callable_globals ):
		ordered_qualnames = { g.variable.qualname for g in ordered }
		stuck = [ g.variable.qualname for g in callable_globals if g.variable.qualname not in ordered_qualnames ]
		unresolved = by_qualname[ stuck[0] ]
		compiler.disco.fail_loc(
			f'circular global-initializer dependency involving {", ".join(sorted(stuck))} - '
			f'each one\'s own initializer (directly or transitively) reads another\'s value, '
			f'with no valid construction order',
			unresolved.variable.file, unresolved.variable.line,
		)
	return ordered

def _emit_global_declaration( g: LoweredGlobal ) -> str:
	name = mangle_qualname( g.variable.qualname )
	# _declarator, not a plain c_type(...) prefix - a Ptr[Callable[...]]
	# global's name goes INSIDE the C function-pointer declarator's parens
	# (RetType (*name)(ParamTypes)), not after a "TYPE NAME"-shaped prefix
	# (see _declarator's own docstring); c_type() itself doesn't even try
	# to spell a bare CallableType, so a plain f'{c_type(...)} {name}' here
	# hits its NotImplementedError for any function-pointer-typed global.
	declarator = _declarator( g.variable.type, name )
	if _is_trivial_global_init( g.instructions ):
		value = _emit_operand( g.instructions[0].src )
		return f'{declarator} = {value};'
	# a {0} zero initializer either way for a non-trivial global - a valid
	# C11 initializer for ANY type alike (ISO C11 6.7.9p11: a scalar
	# initializer may be "optionally enclosed in braces"), matching the same
	# convention ir.Allocate's own empty-fields branch already uses. The
	# _global_init_is_all_zero_value_type case (see emit_global) needs
	# nothing MORE than this - its own init function is skipped entirely,
	# not just left uncalled.
	return f'{declarator} = {{0}};'

def _emit_global_init_fn( g: LoweredGlobal ) -> str|None:
	''' the private `static void __metalpy_init_<name>(void) { ... }` body
	for a non-trivial global - None for a trivial global (its declaration
	above is already the complete real value, see emit_global) or an
	all-zero value-type one (_global_init_is_all_zero_value_type - see its
	own docstring for why this is skipped entirely, not just left
	uncalled). Split out from emit_global (which still returns both
	pieces joined, for existing direct callers) so emit_c can emit EVERY
	global's own declaration before ANY global's own init function body -
	needed now that one global's init function can reference another
	global BY VALUE (`b: Foo = Foo.make(a.x)`) in an order compiler.globals
	itself doesn't guarantee (see _topologically_sort_globals) - unlike
	declarations, which need no relative ordering among THEMSELVES at all
	(every one is a plain, self-contained {0}-or-constant static
	initializer, never referencing another global's value). '''
	if _is_trivial_global_init( g.instructions ) or _global_init_is_all_zero_value_type( g.instructions ):
		return None
	# reuses _emit_instruction exactly like an ordinary function body does
	# (function=None is safe here: lower_global never emits ir.Return/
	# ir.OrReturn, the only two branches that read it - defer/errdefer/
	# loops can't appear in a global initializer at all, see lower_global's
	# own comment). Called from the synthesized __metalpy_init() (emit_c(),
	# after the globals loop below) - see PLAN_GLOBAL_INIT.md.
	init_name = _global_init_fn_name( g )
	lines = [ f'static void {init_name}( void ) {{' ]
	declared: set[str] = set()
	lines.extend( _emit_instructions( g.instructions, function = None, declared = declared ))
	lines.append( '}' )
	return '\n'.join( lines )

def emit_global( g: LoweredGlobal ) -> str:
	# combined declaration + (if any) init function, for direct callers
	# (tests, mostly) that want one global's own complete emitted text as a
	# single string - emit_c() itself uses _emit_global_declaration/_emit_
	# global_init_fn separately (see their own docstrings for why)
	decl = _emit_global_declaration( g )
	init_fn = _emit_global_init_fn( g )
	return decl if init_fn is None else f'{decl}\n\n{init_fn}'

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
			data_attr = cls.get_local_or_raise( 'data' )
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


def emit_c( compiler: Compiler, *, no_crt: bool = False, leak_check: bool = True ) -> str:
	''' single C11 translation unit - see the plan's "three-pass emission
	order" decision. Linking is out of scope (C_EMITTER.md); the whole
	program is already collected into one Compiler instance, so there's no
	reason to split output across files. leak_check (mpy.py's own
	--no-leak-check) only matters in a debug build - it disables the
	automatic __metalpy_deinit() leak-check epilogue (decref every global,
	then dump_live_objects()) that otherwise runs after main() returns. '''
	global _target_os, _target_debug
	_target_os = compiler.disco.active_target['os']
	_target_debug = bool( compiler.disco.active_target['debug'] )
	locked_globals = [ g for g in compiler.globals if _needs_global_lock( g.variable ) ]
	# PLAN_THREAD_SAFE_SHARED_STATE.md Part B: every constructed RC object
	# now carries its own lock (ObjectHeader's own new field,
	# _object_header_prologue) - the SRWLock exports/pthread.h registration
	# below is needed whenever ANY ir.Allocate constructs one anywhere in
	# the whole program, not just when a global happens to need locking
	# (locked_globals alone). No narrower per-class "does THIS class need
	# it" test worth building - Part B applies uniformly to every RC field.
	has_object_header_alloc = (
		any( isinstance( instr, ir.Allocate ) and instr.cls is not None and instr.cls.has_object_header() for lf in compiler.functions for instr in lf.instructions )
		or any( isinstance( instr, ir.Allocate ) and instr.cls is not None and instr.cls.has_object_header() for g in compiler.globals for instr in g.instructions )
	)
	locks_needed = locked_globals or has_object_header_alloc
	if locks_needed and _global_lock_supported() and _target_os == 'windows':
		# real kernel32.dll exports (SRWLOCK is a genuine Win32 primitive,
		# not something this codebase invents) - registered the same way
		# __metalpy_format_f64's own GetProcAddress dependency is, just
		# below, so the ordinary `for lib in sorted(compiler.extern_libs):
		# ...` linking loop every caller (mpy.py, test_support.py, ...)
		# already runs picks up kernel32.lib/-lkernel32 with no changes
		# needed there, even for a program that never itself references
		# windows.kernel32 for anything else.
		compiler.extern_libs.setdefault( 'kernel32', set() ).update((
			'AcquireSRWLockExclusive', 'ReleaseSRWLockExclusive',
		))
	if locks_needed and _global_lock_supported() and _target_uses_pthread_lock():
		# pthread_mutex_t's real definition (needed below, where this
		# module's own storage declaration is a genuine `pthread_mutex_t`,
		# not an opaque void* the way SRWLOCK gets away with - see
		# _global_lock_name's own storage-declaration comment) only exists
		# if <pthread.h> is actually included in this translation unit.
		# Ordinarily that only happens when a compiled program itself
		# imports lib/posix/pthread.py (its own @extern(header='pthread.h')
		# bindings populate compiler.disco.required_headers - see
		# discovery.py's own extern_header handling) - force it here
		# instead, the same way this whole block force-registers kernel32
		# on Windows even for a program that never itself touches
		# windows.kernel32. Registering 'pthread' the same way (mirroring
		# lib/threading.py's own @extern('pthread', ...) bindings, which
		# already link -lpthread whenever threading.py is used) - harmless
		# even on a modern glibc where pthread symbols live in libc itself
		# (glibc >= 2.34) and -lpthread resolves to an effectively-empty
		# compatibility stub. Same treatment for macOS (_target_uses_pthread_
		# lock covers both) - unverified whether -lpthread is even a
		# meaningful flag there (vs. a no-op the way it's become on modern
		# glibc) since none of this has actually been exercised on real
		# macOS hardware - see _global_lock_supported's own poison-pill
		# comment. Registering it regardless costs nothing if it does turn
		# out to be a no-op there too.
		compiler.disco.required_headers.add( 'pthread.h' )
		compiler.extern_libs.setdefault( 'pthread', set() ).update((
			'pthread_mutex_init', 'pthread_mutex_lock', 'pthread_mutex_unlock',
		))
	# __metalpy_format_f64 (PROLOGUE, always present) resolves ntdll's own
	# exported _snprintf via GetProcAddress on Windows, to avoid linking
	# msvcrt (see its own comment for why not a static ntdll.lib import).
	# But unlike every OTHER Windows call in this codebase, it's reached
	# through the compiler.format_f64(...) intrinsic (lowering.py), not an
	# ordinary @extern binding (a fixed-arity extern can't safely reach a
	# genuinely variadic callee - see ir.FormatFloat's own comment), so it
	# never goes through compiler.py's normal `extern_libs.setdefault(
	# unit.extern_lib, ...)` bookkeeping (compiler.py:184) either.
	# Registering it here instead - tagged 'kernel32' (GetModuleHandleA/
	# GetProcAddress are kernel32 exports), deliberately never 'c'
	# (float_test.py's own no_crt = 'c' not in compiler.extern_libs must
	# stay true on Windows regardless of whether float formatting is used)
	# - keeps every existing caller's `for lib in sorted(compiler.
	# extern_libs): ...` linking loop (mpy.py, test_support.py,
	# float_test.py, ...) picking up the right `kernel32.lib`/`-lkernel32`
	# flag with no changes needed there: they all read compiler.extern_libs
	# AFTER calling emit_c(), so this mutation lands in time. POSIX needs no
	# equivalent entry: linker_c.py's gcc/clang link branch never special-
	# cases no_crt at all - libc (real snprintf, the non-Windows half of
	# __metalpy_format_f64) is always linked there regardless, so the 'c'
	# tag would be pure bookkeeping noise, not a needed flag - and, unlike
	# Windows, adding it would incorrectly flip no_crt for any caller that
	# reads compiler.extern_libs before emit_c().
	uses_format_conv = any( isinstance( instr, ir.FormatFloat ) for lf in compiler.functions for instr in lf.instructions )
	uses_parse_conv = any( isinstance( instr, ir.ParseFloat ) for lf in compiler.functions for instr in lf.instructions )
	if compiler.disco.active_target['os'] == 'windows' and ( uses_format_conv or uses_parse_conv ):
		compiler.extern_libs.setdefault( 'kernel32', set() ).add( 'GetProcAddress' )

	# selective PROLOGUE assembly - _PROLOGUE_HEADER/_PROLOGUE_ARITH are
	# always needed (ObjectHeader/vtable typedefs, arithmetic intrinsics),
	# but retain_object/release_object/format_f64/parse_f64 are real "static
	# inline" FUNCTIONS that trigger -Wunused-function (clang; gcc doesn't
	# warn on unused static inline, MSVC doesn't warn on unused static at
	# all) whenever a program doesn't happen to need them - most commonly a
	# trivial program with no RCClass traffic and no float formatting/
	# parsing at all, or (format_f64/parse_f64 specifically - see their own
	# comment) a program using only one of the two directions. Only
	# emitting what's actually referenced avoids that instead of
	# suppressing the warning after the fact.
	# a global's own init (PLAN_THREAD_SAFE_SHARED_STATE.md Part A: its own
	# initializing write now decrefs its pre-init NULL value too) can be the
	# ONLY source of a Decref/Incref in the whole program - compiler.globals
	# has to be scanned alongside compiler.functions, not just the latter
	uses_incref = any( isinstance( instr, ir.Incref ) for lf in compiler.functions for instr in lf.instructions ) \
		or any( isinstance( instr, ir.Incref ) for g in compiler.globals for instr in g.instructions )
	uses_decref = any( isinstance( instr, ( ir.Decref, ir.DecrefDynamic )) for lf in compiler.functions for instr in lf.instructions ) \
		or any( isinstance( instr, ( ir.Decref, ir.DecrefDynamic )) for g in compiler.globals for instr in g.instructions )
	# PLAN_THREAD_SAFE_SHARED_STATE.md Part B: same -Wunused-function
	# avoidance as uses_incref/uses_decref above, for acquire_field_lock/
	# release_field_lock - a program with no RC field access at all
	# (scalars only, or no RCClass traffic) never gets an AcquireFieldLock/
	# ReleaseFieldLock marker emitted anywhere.
	uses_field_lock = any( isinstance( instr, ( ir.AcquireFieldLock, ir.ReleaseFieldLock )) for lf in compiler.functions for instr in lf.instructions ) \
		or any( isinstance( instr, ( ir.AcquireFieldLock, ir.ReleaseFieldLock )) for g in compiler.globals for instr in g.instructions )
	parts: list[str] = [ _object_header_prologue() ]
	if _target_debug:
		parts.append( _prologue_debug_ops() )
	if uses_incref:
		parts.append( _PROLOGUE_RETAIN )
	if uses_decref:
		parts.append( _prologue_release() )
	if ( locked_globals or uses_field_lock ) and _global_lock_supported() and _target_os == 'windows':
		# needed for Part A's own per-global lock storage (locked_globals)
		# AND Part B's per-object $header.lock (uses_field_lock) - both use
		# the identical SRWLOCK forward declarations/AcquireSRWLockExclusive/
		# ReleaseSRWLockExclusive prototypes, so one shared, unconditionally-
		# repeatable declaration covers whichever (or both) actually apply.
		# Emitted BEFORE _field_lock_prologue() below - acquire_field_lock/
		# release_field_lock's own bodies call AcquireSRWLockExclusive/
		# ReleaseSRWLockExclusive directly, so the forward declaration has
		# to land first in the translation unit (confirmed by a real
		# "conflicting types" error when this was the other way around -
		# clang's own implicit-declaration inference from the CALL SITE
		# disagreed with the real prototype declared afterward).
		parts.append( _PROLOGUE_GLOBAL_LOCK_WINDOWS )
		# Neither Linux nor macOS needs an analogous hand-declared prologue
		# (_target_uses_pthread_lock covers both): pthread_mutex_t's real
		# definition and pthread_mutex_lock/unlock/init's real prototypes
		# come from <pthread.h> itself (force-included just above, in this
		# function's own required_headers.add('pthread.h')), which carries
		# none of SRWLOCK's "don't want to drag in all of windows.h just
		# for one struct" cost - see _global_lock_acquire's own comment.
	if uses_field_lock and _global_lock_supported():
		parts.append( _field_lock_prologue() )
	parts.append( _PROLOGUE_ARITH )
	if uses_format_conv:
		parts.append( _PROLOGUE_FLOAT_FORMAT )
	if uses_parse_conv:
		parts.append( _PROLOGUE_FLOAT_PARSE )
	parts.append( _PROLOGUE_CRASH_HANDLER )

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
	# module. Computed here (rather than immediately before its own
	# emission loop below) so _vtable_slot_referenced_classlikes can walk
	# it for the extra forward-tag pass just below.
	vtbl_owners: dict[str,CStruct] = {}
	for cls in compiler.cstructs:
		if cls.is_interface and not cls.type_params:
			vtbl_owners[ _interface_vtbl_name( cls ) ] = cls.vtbl_owner()
	# RCClass analog (RCClass-subclassing plan, Phase 4) - only classes
	# that actually introduce a REAL @virtual slot need their own
	# synthesized type at all (own_new_virtual_slots() non-empty, via
	# vtbl_owner()); the common case (no @virtual methods anywhere in a
	# class's own chain) needs none - it just uses the shared, built-in
	# __metalpy_ObjectVtbl (already defined in the PROLOGUE), never
	# entering this dict at all. Same insertion-ordered dedup pattern as
	# vtbl_owners above.
	rcclass_vtbl_owners: dict[str,RCClass] = {}
	for cls in compiler.rcclasses:
		if not cls.type_params and cls.virtual_slots():
			rcclass_vtbl_owners[ mangle_type( cls.vtbl_owner() )] = cls.vtbl_owner()
	# a type reachable ONLY through a vtable slot's own signature (see
	# _vtable_slot_referenced_classlikes) isn't guaranteed to appear in any
	# of the four unconditional tag loops just above - typically an
	# abstract slot whose concrete override never happens to get compiled
	# anywhere in THIS particular program (e.g. logging.Handler.emit's own
	# `record: LogRecord` when no concrete Handler subclass is ever
	# constructed). Forward-tag anything the vtable owners below still
	# reference that isn't already covered - same "cheap and always safe"
	# reasoning the four loops above already use.
	already_tagged = {
		mangle_type( c )
		for cls_list in ( compiler.rcclasses, compiler.cstructs, compiler.cunions, compiler.tagged_unions )
		for c in cls_list if not c.type_params
	}
	for owner in list( vtbl_owners.values() ) + list( rcclass_vtbl_owners.values() ):
		for referenced in _vtable_slot_referenced_classlikes( owner ):
			name = mangle_type( referenced )
			if name in already_tagged:
				continue
			already_tagged.add( name )
			keyword = 'union' if isinstance( referenced, CUnion ) else 'struct'
			parts.append( f'{keyword} {name};' )
	for owner in vtbl_owners.values():
		parts.append( emit_interface_vtbl_struct( owner ))
	for owner in rcclass_vtbl_owners.values():
		parts.append( emit_rcclass_vtbl_struct( owner ))
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

	# pass 3: static vtable instances first (+ their own trampolines) - only
	# need every other function's PROTOTYPE (pass 1) and the struct bodies
	# (pass 2), so these can sit anywhere in pass 3 relative to THOSE - but
	# NOT anywhere relative to what follows: a global's own non-trivial init
	# function (emit_global, below) may itself construct an RCClass and set
	# `.vtable = &SomeClass$$vtable` directly in its OWN function body text
	# (unlike an ordinary function body, which is always emitted last, in
	# compiler.functions order, and so never has this problem) - so the
	# vtable instances have to be textually EARLIER than the globals loop, or
	# that reference is to a not-yet-declared identifier. Confirmed by a real
	# compile failure: lib/sys.py's `stdout: _Stdout = _Stdout()` global's
	# own __metalpy_init_sys$stdout(), constructing a _Stdout, referenced
	# sys$_Stdout$$vtable before this reordering, when that vtable instance
	# was still emitted further down, after the globals loop.
	for cls in compiler.cstructs:
		if cls.is_interface and not cls.type_params:
			instance_src = emit_interface_vtable_instance( cls )
			if instance_src is not None:
				parts.append( instance_src )
	# RCClass analog (RCClass-subclassing plan Phase 4, "unfulfilled"/
	# abstract skip added Phase 5) - unlike CStruct, every CONCRETE
	# (fully-implemented) RCClass gets one, not just ones with @virtual
	# methods - see emit_rcclass_vtable_instance's own comment: destructor
	# dispatch needs one for every real instance. None (skipped) for an
	# abstract class - see _rcclass_fulfilled_slot_impls.
	for cls in compiler.rcclasses:
		if not cls.type_params:
			instance_src = emit_rcclass_vtable_instance( cls )
			if instance_src is not None:
				parts.append( instance_src )
	# string/bytes literal static objects (need str/bytes's own full RCClass
	# body from pass 2 first) and global definitions, then full function
	# bodies (which may reference either by address), then destructor bodies
	# (need the struct's own full definition from pass 2 to dereference
	# self->field)
	parts.extend( _emit_string_literals( compiler ))
	# every global's own DECLARATION first, in compiler.globals order (each
	# one is a self-contained {0}-or-constant static initializer, never
	# referencing another global's value - see _emit_global_declaration -
	# so no dependency ordering is needed among these at all), THEN every
	# non-trivial global's own init FUNCTION BODY (which - unlike a
	# declaration - CAN reference another global's already-declared value,
	# e.g. `b: Foo = Foo.make(a.x)`; splitting these two into separate
	# passes means that reference is always to an already-declared symbol
	# regardless of compiler.globals' own order, which is scheduling order,
	# not dependency order - see _emit_global_init_fn's own docstring)
	for g in compiler.globals:
		parts.append( _emit_global_declaration( g ))
	if locked_globals and _global_lock_supported() and _target_os == 'windows':
		# one bare, all-zero-initialized `void*` per protected global -
		# SRWLOCK's own all-zero state is already a valid, unlocked lock
		# (see _PROLOGUE_GLOBAL_LOCK_WINDOWS's own comment), so this needs
		# no init function of its own, same as the existing all-zero
		# value-type global fast path (_global_init_is_all_zero_value_type)
		# these deliberately don't go through - they have no metalpy-level
		# Variable/LoweredGlobal of their own to hang that machinery off.
		for g in locked_globals:
			parts.append( f'static void* {_global_lock_name( g.variable )} = 0;' )
	elif locked_globals and _global_lock_supported() and _target_uses_pthread_lock():
		# a REAL pthread_mutex_t here, not a bare void* - unlike SRWLOCK,
		# zero-initializing a pthread_mutex_t isn't a portable guarantee
		# (PLAN_THREAD_SAFE_SHARED_STATE.md's own A.3 POSIX-asymmetry note;
		# glibc happens to tolerate it, lib/threading.py's own POSIX
		# FastLock.__init__ deliberately does NOT rely on that and calls
		# pthread_mutex_init() explicitly instead, and this mirrors that
		# choice). No initializer here at all (plain tentative-definition
		# zero BSS, same as any other uninitialized static) - the REAL
		# initialization is a genuine pthread_mutex_init() call, emitted
		# unconditionally at the very top of __metalpy_init() below, before
		# any ordinary function body can possibly run (ir.AcquireGlobalLock/
		# ir.ReleaseGlobalLock markers only ever appear inside ordinary
		# function bodies, never inside a global's OWN init instructions -
		# see Variable.reassigned_outside_init's own comment - so it's safe
		# for this to run before, or in any order relative to, the
		# topologically-sorted global-init calls just below it).
		for g in locked_globals:
			parts.append( f'static pthread_mutex_t {_global_lock_name( g.variable )};' )
	for g in compiler.globals:
		init_fn = _emit_global_init_fn( g )
		if init_fn is not None:
			parts.append( init_fn )
	# the single, real __metalpy_init() (PLAN_GLOBAL_INIT.md) - calls every
	# non-trivial global's own init function, in DEPENDENCY order
	# (_topologically_sort_globals - NOT compiler.globals' own scheduling
	# order, which has no relationship to which global's own initializer
	# reads which other global's value; confirmed as a real, reachable bug
	# via a real test - `b: Foo = Foo.make(a.x)` with main() only ever
	# referencing b schedules b before a). Always defined and always called
	# (see main()'s own prepend below) - not just on Windows - since global
	# initializers must run on every target now. The Windows console-
	# codepage setup used to be hardcoded directly in here (two competing
	# #ifdef'd function bodies in PROLOGUE, later folded into one); it's now
	# just an ordinary global - windows/_console.py's _console_init, forced
	# reachable on every Windows target by Compiler.run() - so its own
	# SetConsoleOutputCP call flows through this same init_calls list like
	# any other global, with no special-casing needed here at all. A global
	# whose non-trivial init is nonetheless an all-zero value-type
	# construction (_global_init_is_all_zero_value_type) is skipped here -
	# its own {0} static initializer (_emit_global_declaration, above)
	# already IS that value, so calling it would be a pure no-op at best
	# (and, confirmed by a real link failure, a real problem at worst on a
	# no-CRT target if the C compiler lowers the struct-copy into a memset/
	# memcpy call) - _topologically_sort_globals already excludes it from
	# its own graph for the identical reason (it never gets a call, so it
	# can never be a real dependency edge either).
	init_calls = [
		f'\t{_global_init_fn_name( g )}();'
		for g in _topologically_sort_globals( compiler )
	]
	# pthread_mutex_init() calls for every protected global's real lock
	# storage (see that storage declaration's own comment, just above, for
	# why this can't just be a static zero-initializer the way SRWLOCK's
	# storage is). Emitted first, ahead of init_calls - not because
	# anything actually depends on that relative order (nothing in a
	# global's own init instructions ever acquires one of these locks; see
	# the storage declaration's own comment), just the simplest place to
	# put an unconditional, order-independent one-time setup step.
	lock_init_calls = (
		[ f'\tpthread_mutex_init( &{_global_lock_name( g.variable )}, ((void*)0) );' for g in locked_globals ]
		if locked_globals and _global_lock_supported() and _target_uses_pthread_lock() else []
	)
	parts.append(
		'static void __metalpy_init( void ) {\n'
		# first thing any compiled program does, on every target/entry-point
		# shape (see _PROLOGUE_CRASH_HANDLER's own comment) - before global
		# initializers, let alone main(), get a chance to fault.
		+ '\t__metalpy_install_crash_handler();\n'
		+ ( '\n'.join( lock_init_calls ) + '\n' if lock_init_calls else '' )
		+ ( '\n'.join( init_calls ) + '\n' if init_calls else '' )
		+ '}'
	)
	# automatic debug-mode leak-check epilogue: decref every global RC
	# variable (REVERSE of __metalpy_init()'s own construction order, so a
	# global depended on by another global's destructor - mirroring typical
	# static-destruction ordering - is still valid when its dependent's own
	# decref runs), then dump whatever's still tracked. Only globals that
	# actually get a real init CALL (_topologically_sort_globals - same list
	# __metalpy_init() above already iterates) can hold a real RC value in
	# the first place; a trivial/all-zero-value-type global never does.
	# Gated on debug + leak_check (mpy.py's --no-leak-check), AND only when
	# a real thin main() wrapper is actually going to call it (see main_lf
	# below - the rare parameterized-main lowering-fixture shape gets no
	# wrapper at all) - release builds, --no-leak-check builds, and that
	# rare shape never define this function, rather than defining an unused
	# one (-Wunused-function on clang). Same "define only what's actually
	# referenced" treatment every other debug-only PROLOGUE piece gets (see
	# _prologue_debug_ops).
	main_lf = next( ( lf for lf in compiler.functions if _is_entry_point( lf.function )), None )
	main_has_wrapper = main_lf is not None and not main_lf.function.parameters
	deinit_enabled = _target_debug and leak_check and main_has_wrapper
	if deinit_enabled:
		rc_globals_reverse = [
			g.variable for g in reversed( _topologically_sort_globals( compiler ))
			if g.variable.type.is_rc()
		]
		# reuses cfg.py's own union-aware decref() (via Lowering.
		# lower_deinit_epilogue) - identical dispatch to an ordinary local/
		# field release, so a global wrapping a nested-union RC leaf gets
		# the same runtime tag-check+extract sequence, not a bare pointer
		# release that would be wrong for it.
		deinit_instructions = compiler.lowering.lower_deinit_epilogue( rc_globals_reverse )
		deinit_instructions.append( ir.DumpLiveObjects() )
		parts.append(
			'static void __metalpy_deinit( void ) {\n'
			+ '\n'.join( _emit_instructions( deinit_instructions, function = None, declared = set() ))
			+ '\n}'
		)
	# sys._raw_argc/_raw_argv only actually get DECLARED (see the globals
	# loop above) when compiler.py's Compiler.run() successfully force-
	# reaches sys.argv - which no-ops for a deliberately minimal, fixture-
	# only Discovery whose own paths= doesn't include a real lib/sys.py
	# (see force_reachable's own docstring; several *_test.py files use
	# exactly this shape). Emitting the capture assignment unconditionally
	# would then reference an undeclared identifier for those - gate on
	# whether the global is actually present in THIS program.
	has_argv_globals = any( g.variable.qualname == 'sys._raw_argc' for g in compiler.globals )
	for lf in compiler.functions:
		# @extern functions have no body (only a ; declaration in pass 1)
		if lf.function.extern_lib is None:
			src = emit_function( lf )
			if _is_entry_point( lf.function ) and lf.function.parameters:
				# rare lowering-fixture-only shape (_function_prototype's own
				# comment) - this literal main() IS the real C entry point
				# (never renamed to __metalpy_user_main), so it still needs
				# its own __metalpy_init() prelude the old way; no thin
				# wrapper, no argc/argv capture, no leak-check epilogue for
				# this shape (never exercised through emit_c() for real).
				# windows_no_crt still skips it - mainCRTStartup (below)
				# calls this main() itself and would otherwise double-run
				# every global initializer.
				windows_no_crt = no_crt and compiler.disco.active_target['os'] == 'windows'
				if not windows_no_crt:
					src = src.replace( '{\n', '{\n\t__metalpy_init();\n', 1 )
			parts.append( src )

	if main_has_wrapper:
		# __metalpy_main - the single canonical orchestration function
		# (argv capture -> __metalpy_init() -> __metalpy_user_main() ->
		# __metalpy_deinit() -> return), named like every other PROLOGUE
		# internal (__metalpy_init/__metalpy_deinit/__metalpy_install_crash_
		# handler), NOT the real C `main` symbol itself - reused identically
		# by both real OS-entry shims just below (the ordinary CRT-linked
		# `int main(argc,argv)` forwarder, and no_crt's own mainCRTStartup),
		# rather than overloading `main` itself to serve two roles.
		argv_capture = (
			f'\t{mangle_qualname( "sys._raw_argc" )} = argc;\n'
			f'\t{mangle_qualname( "sys._raw_argv" )} = (uint8_t**)argv;\n'
		) if has_argv_globals else ''
		unused_marker = '\t(void)argc; (void)argv;\n' if not has_argv_globals else ''
		deinit_call = '\t__metalpy_deinit();\n' if deinit_enabled else ''
		parts.append(
			'static int __metalpy_main( int argc, char** argv ) {\n'
			+ unused_marker
			+ argv_capture # BEFORE __metalpy_init() - that's what builds sys.argv itself from these
			+ '\t__metalpy_init();\n'
			+ f'\tint __result = {_USER_MAIN_C_NAME}();\n'
			+ deinit_call
			+ '\treturn __result;\n'
			+ '}'
		)
		# only Windows has a genuine alternate OS entry point (mainCRTStartup,
		# #ifdef _WIN32-guarded below) that can call __metalpy_main directly
		# instead - POSIX always needs a real `main()` symbol regardless of
		# no_crt (no freestanding-entry mechanism exists for it here), same
		# "windows_no_crt" distinction the ordinary-functions loop above
		# already draws.
		windows_no_crt = no_crt and _target_os == 'windows'
		if not windows_no_crt:
			# the ONLY thing literally named `main` - a trivial one-line
			# forwarder to __metalpy_main, needed because the real C `main`
			# symbol can't be __metalpy_user_main directly (init/deinit have
			# to wrap around it) and can't be __metalpy_main directly either
			# under windows_no_crt (see mainCRTStartup below, which calls
			# __metalpy_main itself - no main() symbol at all needed there).
			parts.append(
				'int main( int argc, char** argv ) {\n'
				'\treturn __metalpy_main( argc, argv );\n'
				'}'
			)

	# custom entry point when CRT is not linked - the linker expects
	# mainCRTStartup as the /ENTRY, so we provide a thin stub that calls
	# __metalpy_main directly (no real main() symbol exists at all under
	# no_crt - see main_has_wrapper's own comment) and exits cleanly via the
	# process itself.
	# Terminates via sys.exit()'s own mangled C symbol (mangle_qualname
	# doesn't need a Function object - 'sys.exit' is a known, fixed qualname,
	# same as _global_init_fn_name's approach) rather than a hardcoded raw
	# ExitProcess call - compiler.py's Compiler.run() force-enqueues sys.exit
	# whenever no_crt, so this always resolves to a real, lowered function
	# with its own pass-1 prototype already emitted above.
	if no_crt:
		# rare parameterized-main lowering-fixture shape (main_has_wrapper
		# False - see its own comment): __metalpy_main was never
		# synthesized above, so fall back to calling the literal main()
		# that IS defined in that case (old behavior, preserved unchanged).
		entry_call = '__metalpy_main( 0, (char**)0 )' if main_has_wrapper else 'main( 0, (char**)0 )'
		parts.append(
			'#ifdef _WIN32\n'
			'void mainCRTStartup( void ) {\n'
			# no real argc/argv at a freestanding entry point (the OS loader
			# never hands them to WinMainCRTStartup-shaped entries the way
			# it does the UCRT's own main()) - sys.argv (lib/sys.py) just
			# stays empty here, a known, accepted limitation of no_crt
			# builds specifically, not a bug.
			f'\tint __result = {entry_call};\n'
			f'\t{mangle_qualname( "sys.exit" )}( (uint32_t)__result );\n'
			'}\n'
			'#endif'
		)
		# MSVC's linker requires a _fltused symbol to exist whenever any
		# floating-point instruction is used anywhere in the program,
		# normally provided by the CRT's own startup code (confirmed:
		# "unresolved external symbol _fltused" linking a no_crt build that
		# touches a single float). Since this build deliberately doesn't
		# link the CRT, provide it directly - 0x9875 is MSVC's own
		# documented magic value for this marker. GCC/Clang's no_crt path
		# needs no such marker at all, so this is _MSC_VER-guarded to a
		# no-op there (and never emitted for a CRT-linked build at all,
		# where the CRT's own copy already provides it - defining a second
		# one here would conflict).
		parts.append(
			'#if defined(_MSC_VER) && !defined(__clang__)\n'
			'int _fltused = 0x9875;\n'
			'#endif'
		)
		# clang/gcc's own -O0 codegen lowers ANY nontrivial local zero-init
		# (a bare `struct Foo x = {0};`-shaped compound literal, regardless
		# of struct/array size - confirmed even an 8-byte i32[2] field) to a
		# real `call memset`, and a by-value struct copy above a small size
		# threshold to `call memcpy` - neither is a call MetalPy's own
		# extern-tracking machinery ever sees (it's inserted directly by the
		# C compiler's backend, not lowered from any ir.Call this module
		# emits), so the `no_crt = 'c' not in compiler.extern_libs`
		# auto-detection in mpy.py can never catch it the way an explicit
		# crt.memset()/crt.memcpy() call would (that always flips no_crt
		# off). Confirmed via a real LNK2019 "unresolved external symbol
		# memset" building a @cstruct with an i32[8] field as a plain local.
		# MSVC's own /Od codegen never referenced either symbol in testing
		# (up to a 2KB by-value struct copy, before hitting the separate,
		# still-open __chkstk gap - see msvc_no_crt_missing_chkstk memory) -
		# defined unconditionally here anyway since an unreferenced extern
		# definition is harmless, and cheaper than special-casing per
		# compiler. Thin wrappers around sys.memset/sys.memcpy (lib/sys.py's
		# Windows target already routes both through ntdll's RtlFillMemory/
		# RtlCopyMemory - genuinely no_crt-safe, no CRT dependency) rather
		# than a hand-rolled byte loop: reuses an already-vetted primitive
		# instead of duplicating it, and - unlike a hand-rolled loop - has
		# no risk of loop-idiom recognition folding this very definition
		# back into a self-recursive call to itself under a --release (-O2)
		# no_crt build, since there's no loop in the call chain at all
		# (RtlFillMemory/RtlCopyMemory are opaque extern calls). compiler.py's
		# Compiler.run() force-enqueues sys.memset/sys.memcpy whenever
		# no_crt, mirroring its existing sys.exit force_reachable - so
		# sys$memset/sys$memcpy always resolve here, same guarantee
		# mainCRTStartup's own sys$exit call already relies on.
		parts.append(
			'#ifdef _WIN32\n'
			# MSVC recognizes memset/memcpy as compiler intrinsics under
			# optimization (a --release/-O2 build) and refuses to let a
			# TU define a function with that exact name/signature
			# ("error C2169: 'memset': intrinsic function, cannot be
			# defined") - #pragma function is MSVC's own documented way
			# to say "compile a real call here instead", same idiom
			# freestanding/kernel-mode Windows C code already uses for
			# this. Debug (-Od) builds never hit this, which is why it
			# wasn't caught immediately. clang has no such restriction.
			'#if defined(_MSC_VER) && !defined(__clang__)\n'
			'#pragma function(memset, memcpy)\n'
			'#endif\n'
			f'void* memset( void* dst, int value, size_t n ) {{\n'
			f'\treturn {mangle_qualname( "sys.memset" )}( (uint8_t*)dst, (uint8_t)value, n );\n'
			'}\n'
			f'void* memcpy( void* dst, const void* src, size_t n ) {{\n'
			f'\treturn {mangle_qualname( "sys.memcpy" )}( (uint8_t*)dst, (const uint8_t*)src, n );\n'
			'}\n'
			'#endif'
		)
	return '\n\n'.join( part for part in parts if part ) + '\n'
