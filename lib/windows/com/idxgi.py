'''
Minimal DXGI (dxgi.dll/dxgi1_2) surface - just enough to get from a
ID3D11Device to an IDXGIOutputDuplication for desktop-duplication screen
capture. Vtable order and every struct/GUID here verified directly
against this machine's own Windows SDK headers (dxgi.h/dxgi1_2.h under
"Windows Kits\\10\\Include\\...\\shared") - not memory, not a web source -
same scrutiny as windows/com/ipersist.py's own real-COM-interop note, just
with a stronger source available this time.

Every interface below only declares vtable slots up to and including the
last method this project actually calls, in exact real order (inherited
slots first) - unused slots in between are placeholder stubs (`-> None:
...` regardless of the real signature; never called, so only the slot
COUNT/ORDER matters for them - see windows/com/id2d1.py's own header
comment for the identical reasoning already established there).
'''

from windows.com import IUnknown, HRESULT
import guid

def _guid( data1: u32, data2: u16, data3: u16, d0: u8, d1: u8, d2: u8, d3: u8, d4: u8, d5: u8, d6: u8, d7: u8 ) -> guid.GUID:
	g: guid.GUID = guid.GUID( data1 = data1, data2 = data2, data3 = data3, data4 = 0 )
	g.data4[0] = d0
	g.data4[1] = d1
	g.data4[2] = d2
	g.data4[3] = d3
	g.data4[4] = d4
	g.data4[5] = d5
	g.data4[6] = d6
	g.data4[7] = d7
	return g

# verified against this machine's dxgi.h DEFINE_GUID lines directly
IID_IDXGIDEVICE: guid.GUID = _guid( 0x54ec77fa, 0x1377, 0x44e6, 0x8c, 0x32, 0x88, 0xfd, 0x5f, 0x44, 0xc8, 0x4c )
IID_IDXGIADAPTER: guid.GUID = _guid( 0x2411e7e1, 0x12ac, 0x4ccf, 0xbd, 0x14, 0x97, 0x98, 0xe8, 0x53, 0x4d, 0xc0 )
IID_IDXGIOUTPUT1: guid.GUID = _guid( 0x00cddea8, 0x939b, 0x4b83, 0xa3, 0x40, 0xa6, 0x85, 0x22, 0x66, 0x66, 0xcc )

# ---------------------------------------------------------------------------
# plain value structs (dxgi.h/dxgicommon.h/dxgitype.h/dxgi1_2.h)
# ---------------------------------------------------------------------------

@cstruct
class DXGI_RATIONAL:
	numerator: u32 = 0
	denominator: u32 = 0

@cstruct
class DXGI_MODE_DESC:
	width: u32 = 0
	height: u32 = 0
	refresh_rate: DXGI_RATIONAL = DXGI_RATIONAL()
	format: u32 = 0             # DXGI_FORMAT
	scanline_ordering: u32 = 0  # DXGI_MODE_SCANLINE_ORDER
	scaling: u32 = 0            # DXGI_MODE_SCALING

@cstruct
class DXGI_OUTDUPL_DESC:
	mode_desc: DXGI_MODE_DESC = DXGI_MODE_DESC()
	rotation: u32 = 0           # DXGI_MODE_ROTATION
	desktop_image_in_system_memory: i32 = 0 # BOOL

@cstruct
class DXGI_OUTDUPL_POINTER_POSITION:
	x: i32 = 0
	y: i32 = 0
	visible: i32 = 0 # BOOL

@cstruct
class DXGI_OUTDUPL_FRAME_INFO:
	last_present_time: i64 = 0       # LARGE_INTEGER
	last_mouse_update_time: i64 = 0  # LARGE_INTEGER
	accumulated_frames: u32 = 0
	rects_coalesced: i32 = 0             # BOOL
	protected_content_masked_out: i32 = 0 # BOOL
	pointer_position: DXGI_OUTDUPL_POINTER_POSITION = DXGI_OUTDUPL_POINTER_POSITION()
	total_metadata_buffer_size: u32 = 0
	pointer_shape_buffer_size: u32 = 0

@cstruct
class DXGI_RECT: # matches RECT's own layout (LONG left/top/right/bottom) - dxgi's GetFrameDirtyRects fills an array of these
	left: i32 = 0
	top: i32 = 0
	right: i32 = 0
	bottom: i32 = 0


# ---------------------------------------------------------------------------
# IDXGIObject : IUnknown - only GetParent is ever called (to walk
# device -> adapter); SetPrivateData/SetPrivateDataInterface/GetPrivateData
# are unused stubs kept only to hold GetParent at its real vtable offset.
# ---------------------------------------------------------------------------

@interface
class IDXGIObject( IUnknown ):
	@virtual
	def SetPrivateData( self ) -> None: ... # unused stub
	@virtual
	def SetPrivateDataInterface( self ) -> None: ... # unused stub
	@virtual
	def GetPrivateData( self ) -> None: ... # unused stub

	@virtual
	def GetParent( self, riid: ConstPtr[guid.GUID], ppParent: Ptr[Ptr[None]] ) -> HRESULT: ...

# ---------------------------------------------------------------------------
# IDXGIAdapter : IDXGIObject - only EnumOutputs is called.
# ---------------------------------------------------------------------------

@interface
class IDXGIAdapter( IDXGIObject ):
	@virtual
	def EnumOutputs( self, Output: u32, ppOutput: Ptr[Ptr[None]] ) -> HRESULT: ...

# ---------------------------------------------------------------------------
# IDXGIOutput : IDXGIObject - never called through directly (only
# QueryInterface'd, inherited from IUnknown, to get IDXGIOutput1) - still
# has to be a real base class so IDXGIOutput1's own vtable offsets land
# right, but needs none of its OWN 12 methods declared: nothing after
# them in IDXGIOutput1 is skipped over incorrectly as long as the SLOT
# COUNT matches, so they're represented here as one block of stubs.
# ---------------------------------------------------------------------------

@interface
class IDXGIOutput( IDXGIObject ):
	@virtual
	def GetDesc( self ) -> None: ... # unused stub
	@virtual
	def GetDisplayModeList( self ) -> None: ... # unused stub
	@virtual
	def FindClosestMatchingMode( self ) -> None: ... # unused stub
	@virtual
	def WaitForVBlank( self ) -> None: ... # unused stub
	@virtual
	def TakeOwnership( self ) -> None: ... # unused stub
	@virtual
	def ReleaseOwnership( self ) -> None: ... # unused stub
	@virtual
	def GetGammaControlCapabilities( self ) -> None: ... # unused stub
	@virtual
	def SetGammaControl( self ) -> None: ... # unused stub
	@virtual
	def GetGammaControl( self ) -> None: ... # unused stub
	@virtual
	def SetDisplaySurface( self ) -> None: ... # unused stub
	@virtual
	def GetDisplaySurfaceData( self ) -> None: ... # unused stub
	@virtual
	def GetFrameStatistics( self ) -> None: ... # unused stub

# ---------------------------------------------------------------------------
# IDXGIOutput1 : IDXGIOutput - only DuplicateOutput (the last of its own
# 4 new methods) is called.
# ---------------------------------------------------------------------------

@interface
class IDXGIOutput1( IDXGIOutput ):
	@virtual
	def GetDisplayModeList1( self ) -> None: ... # unused stub
	@virtual
	def FindClosestMatchingMode1( self ) -> None: ... # unused stub
	@virtual
	def GetDisplaySurfaceData1( self ) -> None: ... # unused stub

	@virtual
	def DuplicateOutput( self, pDevice: Ptr[None], ppOutputDuplication: Ptr[Ptr[None]] ) -> HRESULT: ...

# ---------------------------------------------------------------------------
# IDXGIOutputDuplication : IDXGIObject - GetDesc/AcquireNextFrame/
# GetFrameDirtyRects/ReleaseFrame are all called; GetFrameMoveRects/
# GetFramePointerShape/MapDesktopSurface/UnMapDesktopSurface (in between,
# before ReleaseFrame) are unused stubs kept only to hold ReleaseFrame at
# its real offset.
# ---------------------------------------------------------------------------

@interface
class IDXGIOutputDuplication( IDXGIObject ):
	@virtual
	def GetDesc( self, pDesc: Ptr[DXGI_OUTDUPL_DESC] ) -> None: ...

	@virtual
	def AcquireNextFrame( self, TimeoutInMilliseconds: u32, pFrameInfo: Ptr[DXGI_OUTDUPL_FRAME_INFO], ppDesktopResource: Ptr[Ptr[None]] ) -> HRESULT: ...

	@virtual
	def GetFrameDirtyRects( self, DirtyRectsBufferSize: u32, pDirtyRectsBuffer: Ptr[DXGI_RECT], pDirtyRectsBufferSizeRequired: Ptr[u32] ) -> HRESULT: ...

	@virtual
	def GetFrameMoveRects( self ) -> None: ... # unused stub
	@virtual
	def GetFramePointerShape( self ) -> None: ... # unused stub
	@virtual
	def MapDesktopSurface( self ) -> None: ... # unused stub
	@virtual
	def UnMapDesktopSurface( self ) -> None: ... # unused stub

	@virtual
	def ReleaseFrame( self ) -> HRESULT: ...
