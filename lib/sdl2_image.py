import compiler

# Minimal SDL2_image bindings - hand-transcribed from the public SDL_image
# API docs, cdecl, no headers shipped here. Same cross-platform structure as
# sdl2.py:
#
# Windows: links against SDL2_image.dll via a synthesized import lib (no .lib
# ships with pip-installed SDL2_image) - see scripts/gen_sdl2_import_lib.ps1
# and each extern's own dll=/libdir=/notice= below.
# POSIX: links against the system's libSDL2_image.so ('apt install
# libsdl2-image-dev' or equivalent) - plain -lSDL2_image, no dll=/libdir=/notice=.

from sdl2 import Renderer, Texture

IMG_INIT_PNG: i32 = 0x00000002

@compiler.target( os = 'windows' )
@extern( 'SDL2_image', 'IMG_Init', dll = 'SDL2_image.dll', libdir = '../scripts/sdl2_import_lib', pip_package = 'sdl2dll', notice = 'SDL2_image' )
def IMG_Init( flags: i32 ) -> i32:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2_image', 'IMG_Init' )
def IMG_Init( flags: i32 ) -> i32:
	...

@compiler.target( os = 'windows' )
@extern( 'SDL2_image', 'IMG_Quit', dll = 'SDL2_image.dll', libdir = '../scripts/sdl2_import_lib', pip_package = 'sdl2dll', notice = 'SDL2_image' )
def IMG_Quit() -> None:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2_image', 'IMG_Quit' )
def IMG_Quit() -> None:
	...

@compiler.target( os = 'windows' )
@extern( 'SDL2_image', 'IMG_LoadTexture', dll = 'SDL2_image.dll', libdir = '../scripts/sdl2_import_lib', pip_package = 'sdl2dll', notice = 'SDL2_image' )
def IMG_LoadTexture( renderer: Renderer, filename: ConstPtr[u8] ) -> Texture:
	...

@compiler.target( os = not 'windows' )
@extern( 'SDL2_image', 'IMG_LoadTexture' )
def IMG_LoadTexture( renderer: Renderer, filename: ConstPtr[u8] ) -> Texture:
	...
