import guid

# HRESULT/IUnknown/core COM bootstrapping - see
# PLAN_SUBCLASSING_VTABLES_COM.md's "COM specifics" section. A library
# addition, not a compiler change - matches the plan's own "hand-rolling
# first" decision (QueryInterface/AddRef/Release are ordinary, hand-
# written @virtual methods; no compiler-synthesized IUnknown boilerplate
# here). A real package (not a flat com.py) so individual COM interfaces
# can live in their own files as they're added - see ipersist.py for the
# first one - matching lib/builtins' own package-with-submodules
# convention (__list.py, __dict.py, ...).

HRESULT: TypeAlias = i32

# well-known HRESULT values (winerror.h) - only the handful any hand-
# written QueryInterface/AddRef/Release actually needs; not attempting a
# complete winerror.h transcription
S_OK: HRESULT = 0x00000000
S_FALSE: HRESULT = 0x00000001
# these all have the high bit set (the FAILED() convention - see SUCCEEDED/
# FAILED below) - as an unsigned 32-bit bit pattern they're fine, but they
# all exceed i32's own signed range, so an explicit cast is needed for the
# same reason lib/windows/kernel32.py already casts its own HANDLE/u32
# bit-reinterpreted constants (e.g. STD_OUTPUT_HANDLE: u32 = u32(-11))
E_NOTIMPL: HRESULT = HRESULT( 0x80004001 )
E_NOINTERFACE: HRESULT = HRESULT( 0x80004002 )
E_POINTER: HRESULT = HRESULT( 0x80004003 )
E_ABORT: HRESULT = HRESULT( 0x80004004 )
E_FAIL: HRESULT = HRESULT( 0x80004005 )
E_UNEXPECTED: HRESULT = HRESULT( 0x8000FFFF )
E_ACCESSDENIED: HRESULT = HRESULT( 0x80070005 )
E_OUTOFMEMORY: HRESULT = HRESULT( 0x8007000E )
E_INVALIDARG: HRESULT = HRESULT( 0x80070057 )

def SUCCEEDED( hr: HRESULT ) -> bool:
	return hr >= 0

def FAILED( hr: HRESULT ) -> bool:
	return hr < 0

# every other interface's own base, directly or transitively - single
# inheritance, one vtable per interface (see the plan's own "Decisions
# made so far"), so QueryInterface here walks one linear chain, not full
# multi-interface COM. Every method is @virtual (they're COM ABI
# requirements - always slots 0/1/2, never optional the way a class's own
# additional methods are) and left as stub declarations (no body) - a
# concrete interface subclassing IUnknown must provide real
# implementations for all three (see PLAN_SUBCLASSING_VTABLES_COM.md's
# "Unimplemented @virtual methods" - a stub-bodied @virtual method can
# never be constructed, lowering.py's own construction-time check
# enforces this), exactly matching real COM: there's no default
# QueryInterface/AddRef/Release, every coclass writes its own.
#
# QueryInterface's ppvObject is Ptr[Ptr[None]] (COM's own `void**` -
# genuinely untyped, any interface pointer can be written there) - Ptr[T]
# nesting confirmed to work directly (Ptr[Ptr[Foo]] as a plain type
# annotation, populated via compiler.addrof or an ordinary Ptr[Ptr[T]]
# parameter) even though sys.alloc[Ptr[T]] specifically doesn't
# (irrelevant here - nothing allocates a Ptr[Ptr[T]] slot itself, callers
# already have one to pass in, e.g. compiler.addrof(some_ptr_var)).
@interface
class IUnknown:
	@virtual
	def QueryInterface( self, riid: ConstPtr[guid.GUID], ppvObject: Ptr[Ptr[None]] ) -> HRESULT: ...

	@virtual
	def AddRef( self ) -> u32: ...

	@virtual
	def Release( self ) -> u32: ...

# CoInitializeEx's own dwCoInit values (objbase.h) - only the one this
# library actually recommends (apartment-threaded, the common case for
# code that isn't itself implementing a free-threaded COM server)
COINIT_APARTMENTTHREADED: u32 = 0x2
COINIT_MULTITHREADED: u32 = 0x0

# CoCreateInstance's own dwClsContext values (wtypesbase.h) - only
# in-process, the common case; CLSCTX_ALL/CLSCTX_LOCAL_SERVER/etc. can be
# added here if something actually needs them
CLSCTX_INPROC_SERVER: u32 = 0x1

@extern( 'ole32', 'CoInitializeEx' )
def CoInitializeEx( pvReserved: Ptr[None], dwCoInit: u32 ) -> HRESULT:
	...

@extern( 'ole32', 'CoUninitialize' )
def CoUninitialize() -> None:
	...

# rclsid/riid are REFCLSID/REFIID in the real signature (C++ references),
# which compile down to plain pointers at the C ABI level - Ptr[GUID],
# not GUID by value (verified against Microsoft Learn's own
# CoCreateInstance docs, not just memory - see the plan doc's own note on
# why GUID values/signatures for real COM interop got this scrutiny)
@extern( 'ole32', 'CoCreateInstance' )
def CoCreateInstance(
	rclsid: Ptr[guid.GUID],
	pUnkOuter: Ptr[IUnknown],
	dwClsContext: u32,
	riid: Ptr[guid.GUID],
	ppv: Ptr[Ptr[None]],
) -> HRESULT:
	...
