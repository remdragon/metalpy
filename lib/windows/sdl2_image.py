# Minimal SDL2_image (SDL2_image.dll) bindings - same posture as sdl2.py:
# hand-transcribed from the public SDL_image API docs, cdecl, no headers
# shipped here.
#
# SDL2_image.lib (link-time) isn't shipped by pysdl2-dll (DLL only) - see
# scripts/gen_sdl2_import_lib.ps1 for how scripts/sdl2_import_lib/SDL2_image.lib
# is synthesized from the DLL's own export table, same as SDL2.lib.

from sdl2 import Renderer, Texture

IMG_INIT_PNG: i32 = 0x00000002

@extern( 'SDL2_image', 'IMG_Init', dll = 'SDL2_image.dll', libdir = '../../scripts/sdl2_import_lib', notice = 'SDL2_image' )
def IMG_Init( flags: i32 ) -> i32:
	...

@extern( 'SDL2_image', 'IMG_Quit', dll = 'SDL2_image.dll', libdir = '../../scripts/sdl2_import_lib', notice = 'SDL2_image' )
def IMG_Quit() -> None:
	...

@extern( 'SDL2_image', 'IMG_LoadTexture', dll = 'SDL2_image.dll', libdir = '../../scripts/sdl2_import_lib', notice = 'SDL2_image' )
def IMG_LoadTexture( renderer: Renderer, filename: ConstPtr[u8] ) -> Texture:
	...
