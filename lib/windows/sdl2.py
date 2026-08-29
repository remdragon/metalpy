# Minimal SDL2 (SDL2.dll) bindings - feasibility spike for calling SDL2's C
# ABI from metalpy-compiled code. Hand-transcribed from the public SDL2 API
# docs (no SDL.h shipped here), same posture as tcl.py/kernel32.py.
#
# SDL2 on Win64 uses the plain C calling convention (SDLCALL == cdecl there),
# matching every other extern in lib/windows/ - no stdcall wrapping needed.
#
# SDL2.lib (link-time import lib, synthesized - see scripts/gen_sdl2_import_
# lib.ps1, no .lib ships with pip-installed SDL2) is found automatically via
# each extern's libdir= pointing at scripts/sdl2_import_lib. SDL2.dll
# (runtime) must still be on PATH at build time - see sdl2_test.py's own
# module docstring for what's needed.

Window: TypeAlias = Ptr[None]
Renderer: TypeAlias = Ptr[None]
Texture: TypeAlias = Ptr[None]

SDL_INIT_VIDEO: u32 = 0x00000020

SDL_WINDOWPOS_UNDEFINED: i32 = 0x1FFF0000

SDL_WINDOW_SHOWN: u32 = 0x00000004

SDL_RENDERER_ACCELERATED: u32 = 0x00000002

# SDL_Event (SDL_events.h) is a tagged union, 56 bytes on a 64-bit build
# (padded by SDL itself for ABI future-proofing - `union { ...; Uint8
# padding[56]; }`). Only `type` is read by this spike; the remaining bytes
# are never interpreted, just reserved so SDL_PollEvent has a full-size
# buffer to write into - same "opaque trailing bytes" posture as
# PAINTSTRUCT.rgbReserved in user32.py.
SDL_QUIT: u32 = 0x100

@cstruct
class Event:
	type: u32 = 0
	_padding: u8[52] = 0

@cstruct
class Rect:
	x: i32 = 0
	y: i32 = 0
	w: i32 = 0
	h: i32 = 0

@extern( 'SDL2', 'SDL_Init', dll = 'SDL2.dll', libdir = '../../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_Init( flags: u32 ) -> i32:
	...

@extern( 'SDL2', 'SDL_Quit', dll = 'SDL2.dll', libdir = '../../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_Quit() -> None:
	...

@extern( 'SDL2', 'SDL_GetError', dll = 'SDL2.dll', libdir = '../../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_GetError() -> ConstPtr[u8]:
	...

@extern( 'SDL2', 'SDL_CreateWindow', dll = 'SDL2.dll', libdir = '../../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_CreateWindow(
	title: ConstPtr[u8],
	x: i32,
	y: i32,
	w: i32,
	h: i32,
	flags: u32,
) -> Window:
	...

@extern( 'SDL2', 'SDL_DestroyWindow', dll = 'SDL2.dll', libdir = '../../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_DestroyWindow( window: Window ) -> None:
	...

@extern( 'SDL2', 'SDL_CreateRenderer', dll = 'SDL2.dll', libdir = '../../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_CreateRenderer(
	window: Window,
	index: i32,
	flags: u32,
) -> Renderer:
	...

@extern( 'SDL2', 'SDL_DestroyRenderer', dll = 'SDL2.dll', libdir = '../../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_DestroyRenderer( renderer: Renderer ) -> None:
	...

@extern( 'SDL2', 'SDL_SetRenderDrawColor', dll = 'SDL2.dll', libdir = '../../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_SetRenderDrawColor(
	renderer: Renderer,
	r: u8,
	g: u8,
	b: u8,
	a: u8,
) -> i32:
	...

@extern( 'SDL2', 'SDL_RenderClear', dll = 'SDL2.dll', libdir = '../../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_RenderClear( renderer: Renderer ) -> i32:
	...

@extern( 'SDL2', 'SDL_RenderPresent', dll = 'SDL2.dll', libdir = '../../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_RenderPresent( renderer: Renderer ) -> None:
	...

@extern( 'SDL2', 'SDL_PollEvent', dll = 'SDL2.dll', libdir = '../../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_PollEvent( event: Ptr[Event] ) -> i32:
	...

@extern( 'SDL2', 'SDL_RenderCopy', dll = 'SDL2.dll', libdir = '../../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_RenderCopy(
	renderer: Renderer,
	texture: Texture,
	srcrect: Ptr[Rect],
	dstrect: Ptr[Rect],
) -> i32:
	...

@extern( 'SDL2', 'SDL_DestroyTexture', dll = 'SDL2.dll', libdir = '../../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_DestroyTexture( texture: Texture ) -> None:
	...

@extern( 'SDL2', 'SDL_Delay', dll = 'SDL2.dll', libdir = '../../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_Delay( ms: u32 ) -> None:
	...
