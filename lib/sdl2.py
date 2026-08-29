import compiler

# Minimal SDL2 bindings - feasibility spike for calling SDL2's C ABI from
# metalpy-compiled code. Hand-transcribed from the public SDL2 API docs (no
# SDL.h shipped here). SDLCALL is cdecl on every target this codebase
# supports (Win64 and SysV) - no stdcall wrapping needed anywhere.
#
# Windows: links against SDL2.dll via a synthesized import lib (no .lib
# ships with pip-installed SDL2) - see scripts/gen_sdl2_import_lib.ps1 and
# each extern's own dll=/libdir=/notice= below. dll= triggers mpy.py's
# post-link step that copies SDL2.dll next to the built exe; libdir= is the
# link-time search path for the synthesized SDL2.lib.
# POSIX: links against the system's libSDL2.so ('apt install libsdl2-dev'
# or equivalent) - plain -lSDL2, no dll=/libdir=/notice= (nothing is bundled,
# same posture as every other system-package extern in this codebase).

Window: TypeAlias = Ptr[None]
Renderer: TypeAlias = Ptr[None]
Texture: TypeAlias = Ptr[None]
Surface: TypeAlias = Ptr[None]

@cstruct
class Color:
	r: u8 = 0
	g: u8 = 0
	b: u8 = 0
	a: u8 = 255

SDL_INIT_VIDEO: u32 = 0x00000020

SDL_WINDOWPOS_UNDEFINED: i32 = 0x1FFF0000

SDL_WINDOW_SHOWN: u32 = 0x00000004

SDL_RENDERER_ACCELERATED: u32 = 0x00000002

# SDL_Event (SDL_events.h) is a tagged union, 56 bytes on a 64-bit build
# (padded by SDL itself for ABI future-proofing - `union { ...; Uint8
# padding[56]; }`). Only `type` is read by this spike; the remaining bytes
# are never interpreted, just reserved so SDL_PollEvent has a full-size
# buffer to write into - same "opaque trailing bytes" posture as
# PAINTSTRUCT.rgbReserved in windows/user32.py.
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

@compiler.target( os = 'windows' )
@extern( 'SDL2', 'SDL_Init', dll = 'SDL2.dll', libdir = '../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_Init( flags: u32 ) -> i32:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2', 'SDL_Init' )
def SDL_Init( flags: u32 ) -> i32:
	...

@compiler.target( os = 'windows' )
@extern( 'SDL2', 'SDL_Quit', dll = 'SDL2.dll', libdir = '../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_Quit() -> None:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2', 'SDL_Quit' )
def SDL_Quit() -> None:
	...

@compiler.target( os = 'windows' )
@extern( 'SDL2', 'SDL_GetError', dll = 'SDL2.dll', libdir = '../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_GetError() -> ConstPtr[u8]:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2', 'SDL_GetError' )
def SDL_GetError() -> ConstPtr[u8]:
	...

@compiler.target( os = 'windows' )
@extern( 'SDL2', 'SDL_CreateWindow', dll = 'SDL2.dll', libdir = '../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_CreateWindow(
	title: ConstPtr[u8],
	x: i32,
	y: i32,
	w: i32,
	h: i32,
	flags: u32,
) -> Window:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2', 'SDL_CreateWindow' )
def SDL_CreateWindow(
	title: ConstPtr[u8],
	x: i32,
	y: i32,
	w: i32,
	h: i32,
	flags: u32,
) -> Window:
	...

@compiler.target( os = 'windows' )
@extern( 'SDL2', 'SDL_DestroyWindow', dll = 'SDL2.dll', libdir = '../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_DestroyWindow( window: Window ) -> None:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2', 'SDL_DestroyWindow' )
def SDL_DestroyWindow( window: Window ) -> None:
	...

@compiler.target( os = 'windows' )
@extern( 'SDL2', 'SDL_CreateRenderer', dll = 'SDL2.dll', libdir = '../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_CreateRenderer(
	window: Window,
	index: i32,
	flags: u32,
) -> Renderer:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2', 'SDL_CreateRenderer' )
def SDL_CreateRenderer(
	window: Window,
	index: i32,
	flags: u32,
) -> Renderer:
	...

@compiler.target( os = 'windows' )
@extern( 'SDL2', 'SDL_DestroyRenderer', dll = 'SDL2.dll', libdir = '../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_DestroyRenderer( renderer: Renderer ) -> None:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2', 'SDL_DestroyRenderer' )
def SDL_DestroyRenderer( renderer: Renderer ) -> None:
	...

@compiler.target( os = 'windows' )
@extern( 'SDL2', 'SDL_SetRenderDrawColor', dll = 'SDL2.dll', libdir = '../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_SetRenderDrawColor(
	renderer: Renderer,
	r: u8,
	g: u8,
	b: u8,
	a: u8,
) -> i32:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2', 'SDL_SetRenderDrawColor' )
def SDL_SetRenderDrawColor(
	renderer: Renderer,
	r: u8,
	g: u8,
	b: u8,
	a: u8,
) -> i32:
	...

@compiler.target( os = 'windows' )
@extern( 'SDL2', 'SDL_RenderClear', dll = 'SDL2.dll', libdir = '../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_RenderClear( renderer: Renderer ) -> i32:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2', 'SDL_RenderClear' )
def SDL_RenderClear( renderer: Renderer ) -> i32:
	...

@compiler.target( os = 'windows' )
@extern( 'SDL2', 'SDL_RenderPresent', dll = 'SDL2.dll', libdir = '../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_RenderPresent( renderer: Renderer ) -> None:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2', 'SDL_RenderPresent' )
def SDL_RenderPresent( renderer: Renderer ) -> None:
	...

@compiler.target( os = 'windows' )
@extern( 'SDL2', 'SDL_PollEvent', dll = 'SDL2.dll', libdir = '../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_PollEvent( event: Ptr[Event] ) -> i32:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2', 'SDL_PollEvent' )
def SDL_PollEvent( event: Ptr[Event] ) -> i32:
	...

@compiler.target( os = 'windows' )
@extern( 'SDL2', 'SDL_RenderCopy', dll = 'SDL2.dll', libdir = '../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_RenderCopy(
	renderer: Renderer,
	texture: Texture,
	srcrect: Ptr[Rect],
	dstrect: Ptr[Rect],
) -> i32:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2', 'SDL_RenderCopy' )
def SDL_RenderCopy(
	renderer: Renderer,
	texture: Texture,
	srcrect: Ptr[Rect],
	dstrect: Ptr[Rect],
) -> i32:
	...

@compiler.target( os = 'windows' )
@extern( 'SDL2', 'SDL_DestroyTexture', dll = 'SDL2.dll', libdir = '../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_DestroyTexture( texture: Texture ) -> None:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2', 'SDL_DestroyTexture' )
def SDL_DestroyTexture( texture: Texture ) -> None:
	...

@compiler.target( os = 'windows' )
@extern( 'SDL2', 'SDL_Delay', dll = 'SDL2.dll', libdir = '../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_Delay( ms: u32 ) -> None:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2', 'SDL_Delay' )
def SDL_Delay( ms: u32 ) -> None:
	...

@compiler.target( os = 'windows' )
@extern( 'SDL2', 'SDL_CreateTextureFromSurface', dll = 'SDL2.dll', libdir = '../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_CreateTextureFromSurface( renderer: Renderer, surface: Surface ) -> Texture:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2', 'SDL_CreateTextureFromSurface' )
def SDL_CreateTextureFromSurface( renderer: Renderer, surface: Surface ) -> Texture:
	...

@compiler.target( os = 'windows' )
@extern( 'SDL2', 'SDL_FreeSurface', dll = 'SDL2.dll', libdir = '../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_FreeSurface( surface: Surface ) -> None:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2', 'SDL_FreeSurface' )
def SDL_FreeSurface( surface: Surface ) -> None:
	...

# SDL_RenderFillRect/SDL_RenderDrawRect were whitelisted into the synthesized
# SDL2.lib/.def but never actually bound here - callers (e.g. mpygame1's
# debug_hud.py) had to re-declare them locally. Bound properly now that
# something else in this file needed to touch the surrounding bindings anyway.

@compiler.target( os = 'windows' )
@extern( 'SDL2', 'SDL_RenderFillRect', dll = 'SDL2.dll', libdir = '../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_RenderFillRect( renderer: Renderer, rect: Ptr[Rect] ) -> i32:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2', 'SDL_RenderFillRect' )
def SDL_RenderFillRect( renderer: Renderer, rect: Ptr[Rect] ) -> i32:
	...

@compiler.target( os = 'windows' )
@extern( 'SDL2', 'SDL_RenderDrawRect', dll = 'SDL2.dll', libdir = '../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_RenderDrawRect( renderer: Renderer, rect: Ptr[Rect] ) -> i32:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2', 'SDL_RenderDrawRect' )
def SDL_RenderDrawRect( renderer: Renderer, rect: Ptr[Rect] ) -> i32:
	...

@compiler.target( os = 'windows' )
@extern( 'SDL2', 'SDL_RenderReadPixels', dll = 'SDL2.dll', libdir = '../scripts/sdl2_import_lib', notice = 'SDL2' )
def SDL_RenderReadPixels( renderer: Renderer, rect: Ptr[Rect], format: u32, pixels: Ptr[u8], pitch: i32 ) -> i32:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2', 'SDL_RenderReadPixels' )
def SDL_RenderReadPixels( renderer: Renderer, rect: Ptr[Rect], format: u32, pixels: Ptr[u8], pitch: i32 ) -> i32:
	...
