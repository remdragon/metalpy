'''
Minimal Direct2D (d2d1.dll) surface - just enough to create a factory, an
HWND render target, a solid-color brush, and fill rectangles. Vtable order
for ID2D1RenderTarget/ID2D1Factory verified against Wine's own d2d1.idl (a
faithful ABI reimplementation), not just memory - same scrutiny as
windows/com/ipersist.py's own real-COM-interop note: a wrong vtable slot
here calls into whatever real function actually sits at that offset.

Every slot up to and including the last one this demo actually calls is
declared, in exact real order (inherited slots first) - unused slots in
between are placeholder stubs (`-> None: ...` regardless of the real
signature; never called, so only the slot COUNT/ORDER matters for them,
not their declared parameter types - each vtable slot is one pointer-sized
function pointer regardless). Anything past the last real call
(CreateHwndRenderTarget on ID2D1Factory, EndDraw on ID2D1RenderTarget) is
simply never declared - see PLAN_SUBCLASSING_VTABLES_COM.md: interfaces can
add virtual slots at any level, nothing requires transcribing a REAL
interface's full method list past what a caller actually needs.

ID2D1HwndRenderTarget isn't declared as its own type: every method this
demo calls (BeginDraw/Clear/CreateSolidColorBrush/FillRectangle/EndDraw) is
inherited unchanged from ID2D1RenderTarget, so the pointer
CreateHwndRenderTarget returns is just used as Ptr[ID2D1RenderTarget]
directly. Likewise ID2D1Brush/ID2D1SolidColorBrush aren't declared - the
brush is never called through (no SetColor/SetOpacity needed - its color
is fixed at CreateSolidColorBrush time), so it's carried around as a bare
Ptr[IUnknown] (AddRef/Release/QueryInterface are all it ever needs).
'''

from windows.com import IUnknown, HRESULT
import guid

# ---------------------------------------------------------------------------
# plain value structs (d2d1.h/dcommon.h)
# ---------------------------------------------------------------------------

@cstruct
class D2D1_COLOR_F:
	r: f32 = 0.0
	g: f32 = 0.0
	b: f32 = 0.0
	a: f32 = 1.0

@cstruct
class D2D1_RECT_F:
	left: f32 = 0.0
	top: f32 = 0.0
	right: f32 = 0.0
	bottom: f32 = 0.0

@cstruct
class D2D1_SIZE_U:
	width: u32 = 0
	height: u32 = 0

@cstruct
class D2D1_PIXEL_FORMAT:
	format: u32 = 0    # DXGI_FORMAT_UNKNOWN - let D2D pick
	alphaMode: u32 = 0 # D2D1_ALPHA_MODE_UNKNOWN

@cstruct
class D2D1_RENDER_TARGET_PROPERTIES:
	type: u32 = 0                             # D2D1_RENDER_TARGET_TYPE_DEFAULT
	pixelFormat: D2D1_PIXEL_FORMAT = D2D1_PIXEL_FORMAT()
	dpiX: f32 = 0.0                           # 0 = use the default (96)
	dpiY: f32 = 0.0
	usage: u32 = 0                            # D2D1_RENDER_TARGET_USAGE_NONE
	minLevel: u32 = 0                         # D2D1_FEATURE_LEVEL_DEFAULT

@cstruct
class D2D1_HWND_RENDER_TARGET_PROPERTIES:
	hwnd: Ptr[None] = None
	pixelSize: D2D1_SIZE_U = D2D1_SIZE_U()
	presentOptions: u32 = 0                   # D2D1_PRESENT_OPTIONS_NONE

@cstruct
class D2D1_BITMAP_PROPERTIES:
	pixelFormat: D2D1_PIXEL_FORMAT = D2D1_PIXEL_FORMAT()
	dpiX: f32 = 0.0
	dpiY: f32 = 0.0

# ---------------------------------------------------------------------------
# D2D1CreateFactory - d2d1.dll's own entry point (not CoCreateInstance-based)
# ---------------------------------------------------------------------------

D2D1_FACTORY_TYPE_SINGLE_THREADED: u32 = 0

# {06152247-6f50-465a-9245-118bfd3b6007} - verified against Wine's own
# d2d1.idl, not just memory (see this file's own header note)
IID_ID2D1FACTORY: guid.GUID = guid.GUID.from_str( '06152247-6f50-465a-9245-118bfd3b6007' )

@extern( 'd2d1', 'D2D1CreateFactory' )
def D2D1CreateFactory(
	factoryType: u32,
	riid: Ptr[guid.GUID],
	pFactoryOptions: Ptr[None], # ConstPtr[D2D1_FACTORY_OPTIONS], always None here
	ppIFactory: Ptr[Ptr[None]],
) -> HRESULT:
	...

# ---------------------------------------------------------------------------
# ID2D1RenderTarget : ID2D1Resource : IUnknown
# ---------------------------------------------------------------------------

@interface
class ID2D1RenderTarget( IUnknown ):
	@virtual
	def GetFactory( self ) -> None: ... # ID2D1Resource - unused stub

	@virtual
	def CreateBitmap( self, size: D2D1_SIZE_U, srcData: ConstPtr[None], pitch: u32, bitmapProperties: ConstPtr[D2D1_BITMAP_PROPERTIES], bitmap: Ptr[Ptr[None]] ) -> HRESULT: ...
	@virtual
	def CreateBitmapFromWicBitmap( self ) -> None: ... # unused stub
	@virtual
	def CreateSharedBitmap( self ) -> None: ... # unused stub
	@virtual
	def CreateBitmapBrush( self ) -> None: ... # unused stub

	@virtual
	def CreateSolidColorBrush( self, color: ConstPtr[D2D1_COLOR_F], brushProperties: Ptr[None], solidColorBrush: Ptr[Ptr[None]] ) -> HRESULT: ...

	@virtual
	def CreateGradientStopCollection( self ) -> None: ... # unused stub
	@virtual
	def CreateLinearGradientBrush( self ) -> None: ... # unused stub
	@virtual
	def CreateRadialGradientBrush( self ) -> None: ... # unused stub
	@virtual
	def CreateCompatibleRenderTarget( self ) -> None: ... # unused stub
	@virtual
	def CreateLayer( self ) -> None: ... # unused stub
	@virtual
	def CreateMesh( self ) -> None: ... # unused stub
	@virtual
	def DrawLine( self ) -> None: ... # unused stub
	@virtual
	def DrawRectangle( self ) -> None: ... # unused stub

	@virtual
	def FillRectangle( self, rect: ConstPtr[D2D1_RECT_F], brush: Ptr[None] ) -> None: ...

	@virtual
	def DrawRoundedRectangle( self ) -> None: ... # unused stub
	@virtual
	def FillRoundedRectangle( self ) -> None: ... # unused stub
	@virtual
	def DrawEllipse( self ) -> None: ... # unused stub
	@virtual
	def FillEllipse( self ) -> None: ... # unused stub
	@virtual
	def DrawGeometry( self ) -> None: ... # unused stub
	@virtual
	def FillGeometry( self ) -> None: ... # unused stub
	@virtual
	def FillMesh( self ) -> None: ... # unused stub
	@virtual
	def FillOpacityMask( self ) -> None: ... # unused stub
	@virtual
	def DrawBitmap( self, bitmap: Ptr[None], destinationRectangle: ConstPtr[D2D1_RECT_F], opacity: f32, interpolationMode: u32, sourceRectangle: ConstPtr[D2D1_RECT_F] ) -> None: ...
	@virtual
	def DrawText( self ) -> None: ... # unused stub
	@virtual
	def DrawTextLayout( self ) -> None: ... # unused stub
	@virtual
	def DrawGlyphRun( self ) -> None: ... # unused stub
	@virtual
	def SetTransform( self ) -> None: ... # unused stub
	@virtual
	def GetTransform( self ) -> None: ... # unused stub
	@virtual
	def SetAntialiasMode( self ) -> None: ... # unused stub
	@virtual
	def GetAntialiasMode( self ) -> None: ... # unused stub
	@virtual
	def SetTextAntialiasMode( self ) -> None: ... # unused stub
	@virtual
	def GetTextAntialiasMode( self ) -> None: ... # unused stub
	@virtual
	def SetTextRenderingParams( self ) -> None: ... # unused stub
	@virtual
	def GetTextRenderingParams( self ) -> None: ... # unused stub
	@virtual
	def SetTags( self ) -> None: ... # unused stub
	@virtual
	def GetTags( self ) -> None: ... # unused stub
	@virtual
	def PushLayer( self ) -> None: ... # unused stub
	@virtual
	def PopLayer( self ) -> None: ... # unused stub
	@virtual
	def Flush( self ) -> None: ... # unused stub
	@virtual
	def SaveDrawingState( self ) -> None: ... # unused stub
	@virtual
	def RestoreDrawingState( self ) -> None: ... # unused stub
	@virtual
	def PushAxisAlignedClip( self ) -> None: ... # unused stub
	@virtual
	def PopAxisAlignedClip( self ) -> None: ... # unused stub

	@virtual
	def Clear( self, clearColor: ConstPtr[D2D1_COLOR_F] ) -> None: ...

	@virtual
	def BeginDraw( self ) -> None: ...

	@virtual
	def EndDraw( self, tag1: Ptr[u64], tag2: Ptr[u64] ) -> HRESULT: ...

# ---------------------------------------------------------------------------
# ID2D1Factory : IUnknown
# ---------------------------------------------------------------------------

@interface
class ID2D1Factory( IUnknown ):
	@virtual
	def ReloadSystemMetrics( self ) -> None: ... # unused stub
	@virtual
	def GetDesktopDpi( self ) -> None: ... # unused stub
	@virtual
	def CreateRectangleGeometry( self ) -> None: ... # unused stub
	@virtual
	def CreateRoundedRectangleGeometry( self ) -> None: ... # unused stub
	@virtual
	def CreateEllipseGeometry( self ) -> None: ... # unused stub
	@virtual
	def CreateGeometryGroup( self ) -> None: ... # unused stub
	@virtual
	def CreateTransformedGeometry( self ) -> None: ... # unused stub
	@virtual
	def CreatePathGeometry( self ) -> None: ... # unused stub
	@virtual
	def CreateStrokeStyle( self ) -> None: ... # unused stub
	@virtual
	def CreateDrawingStateBlock( self ) -> None: ... # unused stub
	@virtual
	def CreateWicBitmapRenderTarget( self ) -> None: ... # unused stub

	@virtual
	def CreateHwndRenderTarget(
		self,
		renderTargetProperties: ConstPtr[D2D1_RENDER_TARGET_PROPERTIES],
		hwndRenderTargetProperties: ConstPtr[D2D1_HWND_RENDER_TARGET_PROPERTIES],
		hwndRenderTarget: Ptr[Ptr[None]],
	) -> HRESULT: ...
