# Feasibility spike for lib/windows/sdl2_image.py, same two-tier pattern as
# sdl2_test.py:
#
# 1. test_program_compiles_to_c - always runs, no SDL2/SDL2_image install
#    required.
# 2. test_load_and_draw_texture - real compile+link+run against pysdl2-dll's
#    bundled SDL2.dll + SDL2_image.dll, skipped unless both synthesized
#    import libs are found (see scripts/gen_sdl2_import_lib.ps1 - same
#    dumpbin-exports -> hand-written .def -> lib.exe /DEF technique as
#    SDL2.lib, extended to also emit SDL2_image.lib).

import os
import unittest
from pathlib import Path

import emitter_c
import test_support
from test_support import RealCompileMixin

try:
	import sdl2dll  # pip install pysdl2-dll - also bundles SDL2_image.dll + codec DLLs
	_SDL2_DLL_DIR: Path | None = Path( sdl2dll.__file__ ).parent / 'dll'
except ImportError:
	_SDL2_DLL_DIR = None

_SDL2_SCRATCH_LIB_DIR = Path( __file__ ).parent / 'scripts' / 'sdl2_import_lib'
_TEST_PNG = Path( __file__ ).parent / 'testdata' / 'tiny.png'


def _find_sdl2_image_install() -> Path | None:
	if ( _SDL2_SCRATCH_LIB_DIR / 'SDL2.lib' ).exists() and ( _SDL2_SCRATCH_LIB_DIR / 'SDL2_image.lib' ).exists():
		return _SDL2_SCRATCH_LIB_DIR
	return None


_SDL2_IMAGE_LIB_DIR = _find_sdl2_image_install()


@unittest.skipUnless( os.name == 'nt', 'lib/windows/sdl2_image.py is a Windows-only binding (SDL2_image.dll via dll=/libdir=) - skipping off Windows' )
class Sdl2ImageTests( RealCompileMixin, unittest.TestCase ):

	def setUp( self ) -> None:
		# runtime deps for the compiled exe: SDL2.dll + SDL2_image.dll (and its
		# codec DLLs, e.g. libpng via zlib) - the synthesized .lib files above
		# are link-time only.
		if _SDL2_DLL_DIR is not None:
			os.environ[ 'PATH' ] = os.environ.get( 'PATH', '' ) + os.pathsep + str( _SDL2_DLL_DIR )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_program_compiles_to_c( self ) -> None:
		''' IR-level check only (no link/run) - proves sdl2_image.py's @extern
		declarations are accepted and emit valid C, independent of whether
		SDL2_image itself is installed. '''
		compiler = self._compile_source( '''
import sdl2
import windows.sdl2_image as img

def main() -> i32:
	if sdl2.SDL_Init( sdl2.SDL_INIT_VIDEO ) != 0:
		return 1
	if img.IMG_Init( img.IMG_INIT_PNG ) == 0:
		return 2

	window: sdl2.Window = sdl2.SDL_CreateWindow(
		"metalpy SDL2_image spike".get_cstr(), sdl2.SDL_WINDOWPOS_UNDEFINED, sdl2.SDL_WINDOWPOS_UNDEFINED,
		64, 64, sdl2.SDL_WINDOW_SHOWN )
	if window is None:
		return 3

	renderer: sdl2.Renderer = sdl2.SDL_CreateRenderer( window, -1, sdl2.SDL_RENDERER_ACCELERATED )
	if renderer is None:
		return 4

	texture: sdl2.Texture = img.IMG_LoadTexture( renderer, "testdata/tiny.png".get_cstr() )
	if texture is None:
		return 5

	sdl2.SDL_RenderCopy( renderer, texture, None, None )
	sdl2.SDL_RenderPresent( renderer )

	sdl2.SDL_DestroyTexture( texture )
	sdl2.SDL_DestroyRenderer( renderer )
	sdl2.SDL_DestroyWindow( window )
	img.IMG_Quit()
	sdl2.SDL_Quit()
	return 0
''' )
		c_source = emitter_c.emit_c( compiler )
		self.assertIn( 'IMG_LoadTexture', c_source )
		self.assertIn( 'SDL_RenderCopy', c_source )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	@unittest.skipUnless( _SDL2_IMAGE_LIB_DIR is not None, 'no SDL2/SDL2_image import libs found - see '
		'scripts/gen_sdl2_import_lib.ps1 - skipping real link+run' )
	def test_load_and_draw_texture( self ) -> None:
		''' real compile+link+run: opens a window, loads testdata/tiny.png via
		IMG_LoadTexture straight to an SDL_Texture, draws it with
		SDL_RenderCopy, presents once - checks every step returns success
		(non-null pointers, zero return codes), not just IR-level success.

		SDL_Init(SDL_INIT_VIDEO) failure (no display/video subsystem, e.g. a
		headless CI runner) is reported as a skip rather than a failure -
		see sdl2_test.py's test_window_open_draw_close for why. timeout is
		generous (30s) to absorb CPU contention under the parallel test
		harness. '''
		c_source = self._emit( f'''
import sdl2
import windows.sdl2_image as img

def main() -> i32:
	if sdl2.SDL_Init( sdl2.SDL_INIT_VIDEO ) != 0:
		return 1
	if img.IMG_Init( img.IMG_INIT_PNG ) == 0:
		return 2

	window: sdl2.Window = sdl2.SDL_CreateWindow(
		"metalpy SDL2_image spike".get_cstr(), sdl2.SDL_WINDOWPOS_UNDEFINED, sdl2.SDL_WINDOWPOS_UNDEFINED,
		64, 64, sdl2.SDL_WINDOW_SHOWN )
	if window is None:
		return 3

	renderer: sdl2.Renderer = sdl2.SDL_CreateRenderer( window, -1, sdl2.SDL_RENDERER_ACCELERATED )
	if renderer is None:
		return 4

	texture: sdl2.Texture = img.IMG_LoadTexture( renderer, "{_TEST_PNG.as_posix()}".get_cstr() )
	if texture is None:
		return 5

	if sdl2.SDL_RenderCopy( renderer, texture, None, None ) != 0:
		return 6
	sdl2.SDL_RenderPresent( renderer )

	sdl2.SDL_DestroyTexture( texture )
	sdl2.SDL_DestroyRenderer( renderer )
	sdl2.SDL_DestroyWindow( window )
	img.IMG_Quit()
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
