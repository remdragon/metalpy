# Real compile-and-run coverage for lib/tkinter.py / lib/windows/tcl.py -
# the Tcl/Tk PoC binding. Requires real Tcl/Tk dev libs (tcl86t.lib/
# tk86t.lib) to link against and tcl86t.dll/tk86t.dll + the Tcl/Tk script
# library to run - none of that is bundled with this repo, so this whole
# module is skipped unless it can find them under the running Python
# interpreter's own install (the same one CPython's own tkinter uses).

import os
import sys as _pysys
import unittest
from pathlib import Path

import test_support
from compiler import Compiler
from discovery import Discovery


def _find_tcl_install() -> Path | None:
	''' locate a Tcl/Tk dev+runtime install to build/run against - the
	Python interpreter running this test suite ships one at
	<prefix>/tcl/{tcl86t.lib,tk86t.lib,tcl8.6/,tk8.6/} and
	<prefix>/DLLs/{tcl86t.dll,tk86t.dll} on Windows. Returns the `tcl`
	directory (the .lib/script-library root) if a complete set is found,
	else None. '''
	prefix = Path( _pysys.base_prefix )
	tcl_dir = prefix / 'tcl'
	dlls_dir = prefix / 'DLLs'
	required = [
		tcl_dir / 'tcl86t.lib', tcl_dir / 'tk86t.lib',
		tcl_dir / 'tcl8.6', tcl_dir / 'tk8.6',
		dlls_dir / 'tcl86t.dll', dlls_dir / 'tk86t.dll',
	]
	if all( p.exists() for p in required ):
		return tcl_dir
	return None


_TCL_DIR = _find_tcl_install()


class TkinterTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' Real compile-and-run coverage for lib/tkinter.py's Tcl/Tk PoC:
	window create/configure/destroy, and a Button -command callback firing
	through Tcl_CreateCommand + a Ptr[Callable[...]] fixed trampoline. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )
		# runtime deps for the compiled exe: tcl86t.dll/tk86t.dll on PATH,
		# and TCL_LIBRARY/TK_LIBRARY pointing at the Tcl/Tk script library
		# Tcl_Init()/Tk_Init() load at process startup (see lib/windows/tcl.py's
		# own module docstring - this is not auto-discovered).
		os.environ[ 'PATH' ] = os.environ.get( 'PATH', '' ) + os.pathsep + str( Path( _pysys.base_prefix ) / 'DLLs' )
		os.environ[ 'TCL_LIBRARY' ] = str( _TCL_DIR / 'tcl8.6' )
		os.environ[ 'TK_LIBRARY' ] = str( _TCL_DIR / 'tk8.6' )

	def _extern_ldflags( self, compiler: Compiler ) -> str:
		''' extends RealCompileMixin's default: tcl86t.lib/tk86t.lib live
		under the Python install's own tcl/ dir, not any standard linker
		search path, so a program using them needs an extra library search
		path flag - see lib/windows/tcl.py's own module docstring for why
		this can't just be an env var alone (LIB/LIBRARY_PATH interact with
		each C compiler's own default-library autodetection in ways that
		aren't reliable across clang/cl - confirmed empirically). '''
		flags = super()._extern_ldflags( compiler )
		if 'tcl86t' in compiler.extern_libs or 'tk86t' in compiler.extern_libs:
			is_cl = test_support._CC is not None and test_support._CC.name == 'cl'
			flags += ( f' /LIBPATH:{_TCL_DIR}' if is_cl else f' -L{_TCL_DIR}' )
		return flags

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	@unittest.skipUnless( _TCL_DIR is not None, 'no Tcl/Tk dev install found under the running '
		'Python interpreter (expected <prefix>/tcl/tcl86t.lib etc.) - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'window_lifecycle', '''
import tkinter

def main() -> i32:
	root: tkinter.Tk = tkinter.Tk()
	if not root.title( "metalpy tkinter test" ):
		return 1

	btn: tkinter.Button = tkinter.Button( root, "Hi" )
	if not btn.pack():
		return 2

	# bounded, non-blocking pump - not root.mainloop(), which blocks until
	# the window is closed (interactively verified separately - see the
	# PLAN's verification section, not something this automated test can
	# drive without a real user closing a window).
	root.pump_events( 20 )

	if not root.eval( "destroy ." ):
		return 3
	return 0
''' ),
			( 'button_command_callback', '''
import tkinter

_click_count: i32 = 0

def on_click( clientData: Ptr[None], interp: tkinter.Tcl_Interp, argc: i32, argv: Ptr[ConstPtr[u8]] ) -> i32:
	global _click_count
	with compiler.wrap_arithmetic:
		_click_count += 1
	return tkinter.TCL_OK

def main() -> i32:
	root: tkinter.Tk = tkinter.Tk()
	btn: tkinter.Button = tkinter.Button( root, "Click me" )
	if not btn.pack():
		return 1
	if not btn.bind_command( on_click ):
		return 2

	if not btn.invoke():
		return 3
	if not btn.invoke():
		return 4

	if _click_count != 2:
		return 5

	if not root.eval( "destroy ." ):
		return 6
	return 0
''' ),
		])


if __name__ == '__main__':
	unittest.main()
