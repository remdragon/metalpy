# Minimal SDL2_ttf (SDL2_ttf.dll) bindings - same posture as sdl2_image.py:
# hand-transcribed from the public SDL_ttf API docs, cdecl, no headers
# shipped here.
#
# SDL2_ttf.lib (link-time) isn't shipped by pysdl2-dll (DLL only) - see
# scripts/gen_sdl2_import_lib.ps1 for how scripts/sdl2_import_lib/SDL2_ttf.lib
# is synthesized from the DLL's own export table, same as SDL2.lib.
#
# _Solid (not _Blended) is used by callers that just want fast, correctly-
# antialiased small debug/UI labels - no alpha blending needed for that.

from sdl2 import Color, Surface

Font: TypeAlias = Ptr[None]

@extern( 'SDL2_ttf', 'TTF_Init', dll = 'SDL2_ttf.dll', libdir = '../../scripts/sdl2_import_lib', notice = 'SDL2_ttf' )
def TTF_Init() -> i32:
	...

@extern( 'SDL2_ttf', 'TTF_Quit', dll = 'SDL2_ttf.dll', libdir = '../../scripts/sdl2_import_lib', notice = 'SDL2_ttf' )
def TTF_Quit() -> None:
	...

@extern( 'SDL2_ttf', 'TTF_OpenFont', dll = 'SDL2_ttf.dll', libdir = '../../scripts/sdl2_import_lib', notice = 'SDL2_ttf' )
def TTF_OpenFont( file: ConstPtr[u8], ptsize: i32 ) -> Font:
	...

@extern( 'SDL2_ttf', 'TTF_CloseFont', dll = 'SDL2_ttf.dll', libdir = '../../scripts/sdl2_import_lib', notice = 'SDL2_ttf' )
def TTF_CloseFont( font: Font ) -> None:
	...

@extern( 'SDL2_ttf', 'TTF_RenderText_Solid', dll = 'SDL2_ttf.dll', libdir = '../../scripts/sdl2_import_lib', notice = 'SDL2_ttf' )
def TTF_RenderText_Solid( font: Font, text: ConstPtr[u8], fg: Color ) -> Surface:
	...

@extern( 'SDL2_ttf', 'TTF_SizeText', dll = 'SDL2_ttf.dll', libdir = '../../scripts/sdl2_import_lib', notice = 'SDL2_ttf' )
def TTF_SizeText( font: Font, text: ConstPtr[u8], w: Ptr[i32], h: Ptr[i32] ) -> i32:
	...
