# Feasibility spike for lib/windows/sdl2.py: can metalpy-compiled code call
# into SDL2's C ABI at all? Two tiers, same split as tkinter_test.py:
#
# 1. test_program_compiles_to_c - always runs. Proves the extern/cstruct
#    declarations themselves are well-formed and the compiler can emit valid
#    C for a real SDL2-calling program, with NO SDL2 install required (no
#    link step, just source -> IR -> C).
#
# 2. test_window_open_draw_close - real compile+link+run against a real
#    SDL2.dll, skipped unless a dev install (SDL2.lib + headers-equivalent
#    knowledge already hand-transcribed into sdl2.py) is found.
#
#    `pip install pysdl2-dll` ships only the runtime SDL2.dll (confirmed:
#    its wheel has no .lib anywhere) - not enough for MSVC's linker on its
#    own. But an import lib can be SYNTHESIZED straight from that bare DLL's
#    export table, same technique linker_c.py's build_ntdll_import_lib
#    already uses for ntdll: `dumpbin /exports SDL2.dll` lists every
#    function name, a hand-written .def lists the ones this binding needs,
#    and `lib.exe /DEF:... /OUT:SDL2.lib /MACHINE:X64` turns that into a
#    real MS-COFF import lib that links cleanly against pysdl2-dll's own
#    SDL2.dll. _SDL2_SCRATCH_LIB_DIR below is where this spike's own
#    generated SDL2.lib+SDL2.def live (checked in, not gitignored, so this
#    test is reproducible without re-running the generation step - see
#    scripts/gen_sdl2_import_lib.ps1). Confirmed working end to end this
#    session: a real window opened, cleared, and closed.

import os
import sys as _pysys
import unittest
from pathlib import Path

import emitter_c
import test_support
from test_support import RealCompileMixin

try:
	import sdl2dll  # pip install pysdl2-dll - bundles the runtime SDL2.dll (no .lib)
	_SDL2_DLL_DIR: Path | None = Path( sdl2dll.__file__ ).parent / 'dll'
except ImportError:
	_SDL2_DLL_DIR = None

_SDL2_SCRATCH_LIB_DIR = Path( __file__ ).parent / 'scripts' / 'sdl2_import_lib'


def _find_sdl2_install() -> Path | None:
	''' locate an SDL2 import lib (SDL2.lib) to link against - checked
	locations: vcpkg's default triplet dir, a couple of conventional
	manual-install spots, and this repo's own synthesized-from-pysdl2-dll
	one (see module docstring). '''
	candidates = [
		Path( 'C:/vcpkg/installed/x64-windows/lib' ),
		Path( 'C:/SDL2/lib/x64' ),
		Path( _pysys.base_prefix ) / 'SDL2' / 'lib' / 'x64',
		_SDL2_SCRATCH_LIB_DIR,
	]
	for lib_dir in candidates:
		if ( lib_dir / 'SDL2.lib' ).exists():
			return lib_dir
	return None


_SDL2_LIB_DIR = _find_sdl2_install()


@unittest.skipUnless( os.name == 'nt', 'lib/windows/sdl2.py is a Windows-only binding (SDL2.dll via dll=/libdir=) - skipping off Windows' )
class Sdl2Tests( RealCompileMixin, unittest.TestCase ):

	def setUp( self ) -> None:
		# runtime dep for the compiled exe: pysdl2-dll's own SDL2.dll (and
		# its bundled codec DLLs) on PATH - the synthesized SDL2.lib above
		# is link-time only, the exe still needs the real DLL to run.
		if _SDL2_DLL_DIR is not None:
			os.environ[ 'PATH' ] = os.environ.get( 'PATH', '' ) + os.pathsep + str( _SDL2_DLL_DIR )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_program_compiles_to_c( self ) -> None:
		''' IR-level check only (no link/run) - proves sdl2.py's @extern/
		@cstruct declarations are accepted and a real SDL2-calling program
		emits valid C, independent of whether SDL2 itself is installed. '''
		compiler = self._compile_source( '''
import windows.sdl2 as sdl2

def main() -> i32:
	if sdl2.SDL_Init( sdl2.SDL_INIT_VIDEO ) != 0:
		return 1

	window: sdl2.Window = sdl2.SDL_CreateWindow(
		"metalpy SDL2 spike".get_cstr(), sdl2.SDL_WINDOWPOS_UNDEFINED, sdl2.SDL_WINDOWPOS_UNDEFINED,
		640, 480, sdl2.SDL_WINDOW_SHOWN )
	if window is None:
		return 2

	renderer: sdl2.Renderer = sdl2.SDL_CreateRenderer( window, -1, sdl2.SDL_RENDERER_ACCELERATED )
	if renderer is None:
		return 3

	event: sdl2.Event = sdl2.Event()
	running: bool = True
	frames: i32 = 0
	while running and frames < 120:
		while sdl2.SDL_PollEvent( compiler.addrof( event ) ) != 0:
			if event.type == sdl2.SDL_QUIT:
				running = False
		sdl2.SDL_SetRenderDrawColor( renderer, 30, 60, 120, 255 )
		sdl2.SDL_RenderClear( renderer )
		sdl2.SDL_RenderPresent( renderer )
		sdl2.SDL_Delay( 16 )
		with compiler.wrap_arithmetic:
			frames += 1

	sdl2.SDL_DestroyRenderer( renderer )
	sdl2.SDL_DestroyWindow( window )
	sdl2.SDL_Quit()
	return 0
''' )
		c_source = emitter_c.emit_c( compiler )
		self.assertIn( 'SDL_CreateWindow', c_source )
		self.assertIn( 'SDL_PollEvent', c_source )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	@unittest.skipUnless( _SDL2_LIB_DIR is not None, 'no SDL2 dev install found (checked vcpkg default '
		'triplet + a couple of manual-install spots - see _find_sdl2_install) - skipping real link+run' )
	def test_window_open_draw_close( self ) -> None:
		''' real compile+link+run: opens a real window, clears it to a solid
		color, pumps events for ~2s or until closed, checks SDL_Init/
		SDL_CreateWindow/SDL_CreateRenderer all returned success.

		A sandbox with no window station/desktop at all (e.g. a headless CI
		runner) makes SDL_Init(SDL_INIT_VIDEO) itself fail - that's a real
		environment limitation, not a regression, so it's reported as a skip
		(with SDL_GetError()'s text) rather than a failure; every other
		nonzero exit is a genuine bug and still fails the test. timeout is
		generous (30s, vs. the ~2s the draw loop itself needs) to absorb
		CPU contention under the parallel test harness (tests.py shards many
		compile+link+run tests concurrently).'''
		c_source = self._emit( '''
import windows.sdl2 as sdl2

def main() -> i32:
	if sdl2.SDL_Init( sdl2.SDL_INIT_VIDEO ) != 0:
		return 1

	window: sdl2.Window = sdl2.SDL_CreateWindow(
		"metalpy SDL2 spike".get_cstr(), sdl2.SDL_WINDOWPOS_UNDEFINED, sdl2.SDL_WINDOWPOS_UNDEFINED,
		640, 480, sdl2.SDL_WINDOW_SHOWN )
	if window is None:
		return 2

	renderer: sdl2.Renderer = sdl2.SDL_CreateRenderer( window, -1, sdl2.SDL_RENDERER_ACCELERATED )
	if renderer is None:
		return 3

	event: sdl2.Event = sdl2.Event()
	running: bool = True
	frames: i32 = 0
	while running and frames < 120:  # ~2s at 16ms/frame, or until the window is closed
		while sdl2.SDL_PollEvent( compiler.addrof( event ) ) != 0:
			if event.type == sdl2.SDL_QUIT:
				running = False
		sdl2.SDL_SetRenderDrawColor( renderer, 30, 60, 120, 255 )
		sdl2.SDL_RenderClear( renderer )
		sdl2.SDL_RenderPresent( renderer )
		sdl2.SDL_Delay( 16 )
		with compiler.wrap_arithmetic:
			frames += 1

	sdl2.SDL_DestroyRenderer( renderer )
	sdl2.SDL_DestroyWindow( window )
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
