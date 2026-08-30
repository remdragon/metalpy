import compiler

# Minimal SDL2_ttf bindings - hand-transcribed from the public SDL_ttf API
# docs, cdecl, no headers shipped here. Same cross-platform structure as
# sdl2.py:
#
# Windows: links against SDL2_ttf.dll via a synthesized import lib (no .lib
# ships with pip-installed SDL2_ttf) - see scripts/gen_sdl2_import_lib.ps1
# and each extern's own dll=/libdir=/notice= below.
# POSIX: links against the system's libSDL2_ttf.so ('apt install
# libsdl2-ttf-dev' or equivalent) - plain -lSDL2_ttf, no dll=/libdir=/notice=.
#
# _Solid (not _Blended) is used by callers that just want fast, correctly-
# antialiased small debug/UI labels - no alpha blending needed for that.

from sdl2 import Color, Surface

Font: TypeAlias = Ptr[None]

@compiler.target( os = 'windows' )
@extern( 'SDL2_ttf', 'TTF_Init', dll = 'SDL2_ttf.dll', libdir = '../scripts/sdl2_import_lib', pip_package = 'sdl2dll', notice = 'SDL2_ttf' )
def TTF_Init() -> i32:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2_ttf', 'TTF_Init' )
def TTF_Init() -> i32:
	...

@compiler.target( os = 'windows' )
@extern( 'SDL2_ttf', 'TTF_Quit', dll = 'SDL2_ttf.dll', libdir = '../scripts/sdl2_import_lib', pip_package = 'sdl2dll', notice = 'SDL2_ttf' )
def TTF_Quit() -> None:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2_ttf', 'TTF_Quit' )
def TTF_Quit() -> None:
	...

@compiler.target( os = 'windows' )
@extern( 'SDL2_ttf', 'TTF_OpenFont', dll = 'SDL2_ttf.dll', libdir = '../scripts/sdl2_import_lib', pip_package = 'sdl2dll', notice = 'SDL2_ttf' )
def TTF_OpenFont( file: ConstPtr[u8], ptsize: i32 ) -> Font:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2_ttf', 'TTF_OpenFont' )
def TTF_OpenFont( file: ConstPtr[u8], ptsize: i32 ) -> Font:
	...

@compiler.target( os = 'windows' )
@extern( 'SDL2_ttf', 'TTF_CloseFont', dll = 'SDL2_ttf.dll', libdir = '../scripts/sdl2_import_lib', pip_package = 'sdl2dll', notice = 'SDL2_ttf' )
def TTF_CloseFont( font: Font ) -> None:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2_ttf', 'TTF_CloseFont' )
def TTF_CloseFont( font: Font ) -> None:
	...

@compiler.target( os = 'windows' )
@extern( 'SDL2_ttf', 'TTF_RenderText_Solid', dll = 'SDL2_ttf.dll', libdir = '../scripts/sdl2_import_lib', pip_package = 'sdl2dll', notice = 'SDL2_ttf' )
def TTF_RenderText_Solid( font: Font, text: ConstPtr[u8], fg: Color ) -> Surface:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2_ttf', 'TTF_RenderText_Solid' )
def TTF_RenderText_Solid( font: Font, text: ConstPtr[u8], fg: Color ) -> Surface:
	...

@compiler.target( os = 'windows' )
@extern( 'SDL2_ttf', 'TTF_SizeText', dll = 'SDL2_ttf.dll', libdir = '../scripts/sdl2_import_lib', pip_package = 'sdl2dll', notice = 'SDL2_ttf' )
def TTF_SizeText( font: Font, text: ConstPtr[u8], w: Ptr[i32], h: Ptr[i32] ) -> i32:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2_ttf', 'TTF_SizeText' )
def TTF_SizeText( font: Font, text: ConstPtr[u8], w: Ptr[i32], h: Ptr[i32] ) -> i32:
	...
