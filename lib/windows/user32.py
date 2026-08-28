'''
Minimal user32.dll surface - just enough to register a window class, create
a top-level window, and pump its message loop. Only the ANSI (`A`-suffixed)
entry points are used throughout (matching this file's own class/window
name handling - plain `str.get_cstr()`, no UTF-16 conversion needed), same
posture as every other lib/windows/*.py file: no `windows.h` include (see
lib/windows/ws2_32.py's own comment on why that conflicts with this
compiler's generated prototypes) - every struct/function here is hand-
transcribed from real Win32 headers/docs, not machine-generated.
'''

from windows.kernel32 import HANDLE

HWND: TypeAlias = Ptr[None]
WPARAM: TypeAlias = usize
LPARAM: TypeAlias = isize
LRESULT: TypeAlias = isize
WndProc: TypeAlias = Ptr[Callable[[HWND, u32, WPARAM, LPARAM], LRESULT]]

@cstruct
class POINT:
	x: i32 = 0
	y: i32 = 0

@cstruct
class RECT:
	left: i32 = 0
	top: i32 = 0
	right: i32 = 0
	bottom: i32 = 0

@cstruct
class MSG:
	hwnd: HWND = None
	message: u32 = 0
	wParam: WPARAM = 0
	lParam: LPARAM = 0
	time: u32 = 0
	pt: POINT = POINT()
	# real MSG also has a trailing lPrivate field on current SDKs - never
	# written by GetMessage/PeekMessage for a caller-owned struct at this
	# classic size (kept for decades of ABI back-compat), safe to omit

@cstruct
class PAINTSTRUCT:
	hdc: HANDLE = None
	fErase: i32 = 0
	rcPaint: RECT = RECT()
	fRestore: i32 = 0
	fIncUpdate: i32 = 0
	rgbReserved: u8[32] = 0

@cstruct
class WNDCLASSEXA:
	cbSize: u32 = 0
	style: u32 = 0
	lpfnWndProc: WndProc = None
	cbClsExtra: i32 = 0
	cbWndExtra: i32 = 0
	hInstance: HANDLE = None
	hIcon: HANDLE = None
	hCursor: HANDLE = None
	hbrBackground: HANDLE = None
	lpszMenuName: ConstPtr[u8] = None
	lpszClassName: ConstPtr[u8] = None
	hIconSm: HANDLE = None

# window styles (winuser.h) - only what's needed for one plain overlapped
# top-level window
WS_OVERLAPPED: u32 = 0x00000000
WS_CAPTION: u32 = 0x00C00000
WS_SYSMENU: u32 = 0x00080000
WS_THICKFRAME: u32 = 0x00040000
WS_MINIMIZEBOX: u32 = 0x00020000
WS_MAXIMIZEBOX: u32 = 0x00010000
WS_OVERLAPPEDWINDOW: u32 = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX
CW_USEDEFAULT: i32 = i32( u32( 0x80000000 )) # (int)0x80000000 - avoids MSVC C4146 on a bare i32-min literal

SW_SHOW: i32 = 5

# window messages actually handled/checked by a WndProc in this demo
WM_DESTROY: u32 = 0x0002
WM_PAINT: u32 = 0x000F

@extern( 'user32', 'RegisterClassExA' )
def RegisterClassExA( lpwcx: ConstPtr[WNDCLASSEXA] ) -> u16: # ATOM, 0 on failure
	...

@extern( 'user32', 'CreateWindowExA' )
def CreateWindowExA(
	dwExStyle: u32,
	lpClassName: ConstPtr[u8],
	lpWindowName: ConstPtr[u8],
	dwStyle: u32,
	X: i32,
	Y: i32,
	nWidth: i32,
	nHeight: i32,
	hWndParent: HWND,
	hMenu: HANDLE,
	hInstance: HANDLE,
	lpParam: Ptr[None],
) -> HWND:
	...

@extern( 'user32', 'DefWindowProcA' )
def DefWindowProcA( hWnd: HWND, Msg: u32, wParam: WPARAM, lParam: LPARAM ) -> LRESULT:
	...

# real return contract is a genuine 3-way result (nonzero/0/-1 on WM_QUIT
# vs. an error) - i32, not bool, so the -1 error case survives intact;
# callers loop `while GetMessageA(...) > 0:`
@extern( 'user32', 'GetMessageA' )
def GetMessageA( lpMsg: Ptr[MSG], hWnd: HWND, wMsgFilterMin: u32, wMsgFilterMax: u32 ) -> i32:
	...

@extern( 'user32', 'TranslateMessage' )
def TranslateMessage( lpMsg: ConstPtr[MSG] ) -> bool:
	...

@extern( 'user32', 'DispatchMessageA' )
def DispatchMessageA( lpMsg: ConstPtr[MSG] ) -> LRESULT:
	...

@extern( 'user32', 'PostQuitMessage' )
def PostQuitMessage( nExitCode: i32 ) -> None:
	...

@extern( 'user32', 'ShowWindow' )
def ShowWindow( hWnd: HWND, nCmdShow: i32 ) -> bool:
	...

@extern( 'user32', 'UpdateWindow' )
def UpdateWindow( hWnd: HWND ) -> bool:
	...

@extern( 'user32', 'GetClientRect' )
def GetClientRect( hWnd: HWND, lpRect: Ptr[RECT] ) -> bool:
	...

@extern( 'user32', 'BeginPaint' )
def BeginPaint( hWnd: HWND, lpPaint: Ptr[PAINTSTRUCT] ) -> HANDLE:
	...

@extern( 'user32', 'EndPaint' )
def EndPaint( hWnd: HWND, lpPaint: ConstPtr[PAINTSTRUCT] ) -> bool:
	...
