# Raw Tcl/Tk C API bindings (Windows only, `tcl86t.dll`/`tk86t.dll`) - the
# opaque-handle + Tcl_Eval-driven surface `lib/tkinter.py` builds on. Tk
# widgets are created and configured entirely through Tcl command strings
# (`Tcl_Eval`) rather than by calling Tk's internal widget-construction C
# functions directly - the same approach CPython's own `_tkinter` uses under
# the hood - so this module never has to lay out any Tk-internal struct
# (Tk_Window etc.); the only opaque handle type it needs is Ptr[None], same
# convention as kernel32.py's HANDLE.
#
# No tcl.h/tk.h are shipped alongside the DLLs this binds against (a plain
# Python install ships the compiled libs/DLLs but not the C headers), so
# every signature below is hand-transcribed from the public Tcl/Tk C API
# docs - same manual-declaration precedent kernel32.py already set for
# Win32 (no headers used there either).
#
# Linking requires the Tcl/Tk import libs (tcl86t.lib/tk86t.lib) to be on
# the linker's search path - they are NOT in a standard system location, so
# the caller must pass an extra library search path (e.g. mpy's --ldflags
# "/LIBPATH:<dir>" for MSVC, or LIBRARY_PATH for gcc/clang) pointing at
# wherever tcl86t.lib/tk86t.lib live (e.g. a Python install's `tcl/`
# subdirectory). Running the resulting exe additionally requires
# tcl86t.dll/tk86t.dll on PATH, and TCL_LIBRARY/TK_LIBRARY environment
# variables pointing at the Tcl/Tk script library (the tcl8.6/tk8.6
# directories of .tcl scripts Tcl_Init()/Tk_Init() load at runtime) - none
# of this is bundled or auto-discovered by this module.

Tcl_Interp: TypeAlias = Ptr[None]

# TCL_OK == 0, the only return code this module's callers check for.
TCL_OK: i32 = 0

# Tcl_DoOneEvent flag bits (tcl.h) - stable, decades-unchanged public
# constants. TCL_DONT_WAIT processes one already-pending event (or returns
# immediately if none), used for a bounded, non-blocking event pump.
# TCL_ALL_EVENTS (~TCL_DONT_WAIT) processes any event class and blocks
# until one arrives, used by a real mainloop.
TCL_DONT_WAIT: i32 = 2
TCL_ALL_EVENTS: i32 = -3

@extern( 'tcl86t', 'Tcl_CreateInterp' )
def Tcl_CreateInterp() -> Tcl_Interp:
	...

@extern( 'tcl86t', 'Tcl_Init' )
def Tcl_Init( interp: Tcl_Interp ) -> i32:
	...

@extern( 'tk86t', 'Tk_Init' )
def Tk_Init( interp: Tcl_Interp ) -> i32:
	...

@extern( 'tcl86t', 'Tcl_Eval' )
def Tcl_Eval( interp: Tcl_Interp, script: ConstPtr[u8] ) -> i32:
	...

@extern( 'tcl86t', 'Tcl_GetStringResult' )
def Tcl_GetStringResult( interp: Tcl_Interp ) -> ConstPtr[u8]:
	...

# proc's signature is (ClientData, Tcl_Interp*, int argc, const char *argv[])
# -> int, the classic (pre-Tcl_Obj) Tcl_CmdProc shape - the simplest command
# callback signature Tcl_CreateCommand accepts, sufficient for a fixed
# trampoline that doesn't need to inspect its arguments.
@extern( 'tcl86t', 'Tcl_CreateCommand' )
def Tcl_CreateCommand(
	interp: Tcl_Interp,
	cmdName: ConstPtr[u8],
	proc: Ptr[Callable[[Ptr[None], Tcl_Interp, i32, Ptr[ConstPtr[u8]]], i32]],
	clientData: Ptr[None],
	deleteProc: Ptr[None],
) -> Ptr[None]:
	...

@extern( 'tcl86t', 'Tcl_DoOneEvent' )
def Tcl_DoOneEvent( flags: i32 ) -> i32:
	...

@extern( 'tcl86t', 'Tcl_DeleteInterp' )
def Tcl_DeleteInterp( interp: Tcl_Interp ) -> None:
	...

# Number of toplevel windows still open - a real Tk_MainLoop-shaped
# mainloop (below, in lib/tkinter.py) runs until this reaches 0, i.e. until
# the user closes the last window, same as real tkinter's mainloop().
@extern( 'tk86t', 'Tk_GetNumMainWindows' )
def Tk_GetNumMainWindows() -> i32:
	...
