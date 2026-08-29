# Feasibility spike for lib/windows/sdl2_ttf.py, same two-tier pattern as
# sdl2_image_test.py:
#
# 1. test_program_compiles_to_c - always runs, no SDL2_ttf install required.
# 2. test_render_text_to_texture - real compile+link+run against pysdl2-dll's
#    bundled SDL2.dll + SDL2_ttf.dll, skipped unless the synthesized import
#    lib is found (see scripts/gen_sdl2_import_lib.ps1) AND a .ttf font is
#    available to open (a Windows system font - no font is shipped in this
#    repo; mpygame1 ships its own CC0 font for its actual game use).

import os
import unittest
from pathlib import Path

import emitter_c
import test_support
from test_support import RealCompileMixin

try:
	import sdl2dll  # pip install pysdl2-dll - also bundles SDL2_ttf.dll
	_SDL2_DLL_DIR: Path | None = Path( sdl2dll.__file__ ).parent / 'dll'
except ImportError:
	_SDL2_DLL_DIR = None

_SDL2_SCRATCH_LIB_DIR = Path( __file__ ).parent / 'scripts' / 'sdl2_import_lib'

# system font used only to exercise a real TTF_OpenFont - not shipped by this
# repo (mpygame1 carries its own CC0 font for the actual game).
_TEST_FONT = Path( r'C:\Windows\Fonts\arial.ttf' )


def _find_sdl2_ttf_install() -> Path | None:
	if ( _SDL2_SCRATCH_LIB_DIR / 'SDL2.lib' ).exists() and ( _SDL2_SCRATCH_LIB_DIR / 'SDL2_ttf.lib' ).exists():
		return _SDL2_SCRATCH_LIB_DIR
	return None


_SDL2_TTF_LIB_DIR = _find_sdl2_ttf_install()


@unittest.skipUnless( os.name == 'nt', 'lib/windows/sdl2_ttf.py is a Windows-only binding (SDL2_ttf.dll via dll=/libdir=) - skipping off Windows' )
class Sdl2TtfTests( RealCompileMixin, unittest.TestCase ):

	def setUp( self ) -> None:
		# runtime deps for the compiled exe: SDL2.dll + SDL2_ttf.dll (and its
		# bundled freetype/harfbuzz DLLs) on PATH - the synthesized .lib files
		# above are link-time only.
		if _SDL2_DLL_DIR is not None:
			os.environ[ 'PATH' ] = os.environ.get( 'PATH', '' ) + os.pathsep + str( _SDL2_DLL_DIR )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_program_compiles_to_c( self ) -> None:
		''' IR-level check only (no link/run) - proves sdl2_ttf.py's @extern
		declarations are accepted and emit valid C, independent of whether
		SDL2_ttf itself is installed. '''
		compiler = self._compile_source( '''
import sdl2
import windows.sdl2_ttf as ttf

def main() -> i32:
	if sdl2.SDL_Init( sdl2.SDL_INIT_VIDEO ) != 0:
		return 1
	if ttf.TTF_Init() != 0:
		return 2

	window: sdl2.Window = sdl2.SDL_CreateWindow(
		"metalpy SDL2_ttf spike".get_cstr(), sdl2.SDL_WINDOWPOS_UNDEFINED, sdl2.SDL_WINDOWPOS_UNDEFINED,
		64, 64, sdl2.SDL_WINDOW_SHOWN )
	if window is None:
		return 3

	renderer: sdl2.Renderer = sdl2.SDL_CreateRenderer( window, -1, sdl2.SDL_RENDERER_ACCELERATED )
	if renderer is None:
		return 4

	font: ttf.Font = ttf.TTF_OpenFont( "font.ttf".get_cstr(), 16 )
	if font is None:
		return 5

	color: sdl2.Color = sdl2.Color()
	color.r = 255
	color.g = 255
	color.b = 255
	color.a = 255
	surface: sdl2.Surface = ttf.TTF_RenderText_Solid( font, "HP".get_cstr(), color )
	if surface is None:
		return 6

	texture: sdl2.Texture = sdl2.SDL_CreateTextureFromSurface( renderer, surface )
	sdl2.SDL_FreeSurface( surface )
	if texture is None:
		return 7

	sdl2.SDL_RenderCopy( renderer, texture, None, None )
	sdl2.SDL_RenderPresent( renderer )

	sdl2.SDL_DestroyTexture( texture )
	ttf.TTF_CloseFont( font )
	sdl2.SDL_DestroyRenderer( renderer )
	sdl2.SDL_DestroyWindow( window )
	ttf.TTF_Quit()
	sdl2.SDL_Quit()
	return 0
''' )
		c_source = emitter_c.emit_c( compiler )
		self.assertIn( 'TTF_RenderText_Solid', c_source )
		self.assertIn( 'SDL_CreateTextureFromSurface', c_source )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	@unittest.skipUnless( _SDL2_TTF_LIB_DIR is not None, 'no SDL2/SDL2_ttf import libs found - see '
		'scripts/gen_sdl2_import_lib.ps1 - skipping real link+run' )
	@unittest.skipUnless( _TEST_FONT.exists(), f'{_TEST_FONT} not found - skipping real link+run' )
	def test_render_text_to_texture( self ) -> None:
		''' real compile+link+run: opens a window, opens a system .ttf font,
		renders "HP" to a surface via TTF_RenderText_Solid, converts it to a
		texture and draws it - checks every step returns success (non-null
		pointers, zero return codes), not just IR-level success.

		SDL_Init(SDL_INIT_VIDEO) failure (no display/video subsystem) is
		reported as a skip rather than a failure - see sdl2_test.py for why. '''
		c_source = self._emit( f'''
import sdl2
import windows.sdl2_ttf as ttf

def main() -> i32:
	if sdl2.SDL_Init( sdl2.SDL_INIT_VIDEO ) != 0:
		return 1
	if ttf.TTF_Init() != 0:
		return 2

	window: sdl2.Window = sdl2.SDL_CreateWindow(
		"metalpy SDL2_ttf spike".get_cstr(), sdl2.SDL_WINDOWPOS_UNDEFINED, sdl2.SDL_WINDOWPOS_UNDEFINED,
		64, 64, sdl2.SDL_WINDOW_SHOWN )
	if window is None:
		return 3

	renderer: sdl2.Renderer = sdl2.SDL_CreateRenderer( window, -1, sdl2.SDL_RENDERER_ACCELERATED )
	if renderer is None:
		return 4

	font: ttf.Font = ttf.TTF_OpenFont( "{_TEST_FONT.as_posix()}".get_cstr(), 16 )
	if font is None:
		return 5

	w: i32 = 0
	h: i32 = 0
	if ttf.TTF_SizeText( font, "HP".get_cstr(), compiler.addrof( w ), compiler.addrof( h )) != 0:
		return 6
	if w <= 0 or h <= 0:
		return 7

	color: sdl2.Color = sdl2.Color()
	color.r = 255
	color.g = 255
	color.b = 255
	color.a = 255
	surface: sdl2.Surface = ttf.TTF_RenderText_Solid( font, "HP".get_cstr(), color )
	if surface is None:
		return 8

	texture: sdl2.Texture = sdl2.SDL_CreateTextureFromSurface( renderer, surface )
	sdl2.SDL_FreeSurface( surface )
	if texture is None:
		return 9

	if sdl2.SDL_RenderCopy( renderer, texture, None, None ) != 0:
		return 10
	sdl2.SDL_RenderPresent( renderer )

	sdl2.SDL_DestroyTexture( texture )
	ttf.TTF_CloseFont( font )
	sdl2.SDL_DestroyRenderer( renderer )
	sdl2.SDL_DestroyWindow( window )
	ttf.TTF_Quit()
	sdl2.SDL_Quit()
	return 0
''' )
		result = self._build_and_run( self.compiler, c_source, timeout = 30 )
		if result.returncode == 1:
			self.skipTest( f'SDL_Init(SDL_INIT_VIDEO) failed - no display/video subsystem available in this '
				f'environment (stderr: {result.stderr})' )
		self.assertEqual( result.returncode, 0,
			f'exe exited {result.returncode}, expected 0 (stderr: {result.stderr})' )

	def _emit( self, source: str ) -> str:
		self.compiler = self._compile_source( source )
		return emitter_c.emit_c( self.compiler )


if __name__ == '__main__':
	unittest.main()
