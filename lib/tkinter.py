'''
Minimal tkinter-like wrapper over the real Tcl/Tk C library (Windows only,
`tcl86t.dll`/`tk86t.dll` - see lib/windows/tcl.py for the raw bindings and
the linker/runtime environment this requires). Proof-of-concept scope only:

	Tk    — root interpreter + window; .eval()/.title()/.mainloop()
	Button — one widget, constructed and packed via Tcl_Eval, with an
	         optional command callback wired through Tcl_CreateCommand

Deliberately NOT attempted here (see PLAN docs / project history before
extending this file): Entry/Frame/other widgets, grid/place layout, a
StringVar-equivalent, or a general multi-callback registry beyond "each
Button gets its own plain top-level callback function" - real closures
aren't supported by this compiler yet (see PLAN_CALLABLE.md).
'''

import compiler
import sys

if compiler.target.os == 'windows':
	from windows.tcl import (
		Tcl_Interp, TCL_OK, TCL_DONT_WAIT, TCL_ALL_EVENTS,
		Tcl_CreateInterp, Tcl_Init, Tk_Init, Tcl_Eval, Tcl_GetStringResult,
		Tcl_CreateCommand, Tcl_DoOneEvent, Tcl_DeleteInterp,
		Tk_GetNumMainWindows,
	)

# The exact callback shape Tcl_CreateCommand's `proc` parameter requires
# (the classic pre-Tcl_Obj Tcl_CmdProc signature) - a Button's `command`
# must be a bare reference to a plain top-level function (or @staticmethod)
# matching this signature exactly; this compiler has no closures yet, so
# there is no way to adapt an arbitrary zero-arg callback automatically.
TclCommandProc: TypeAlias = Ptr[Callable[[Ptr[None], Tcl_Interp, i32, Ptr[ConstPtr[u8]]], i32]]


class Tk:
	__interp: Tcl_Interp

	def __init__( self ) -> None:
		self.__interp = Tcl_CreateInterp()
		if not self.__interp:
			sys.panic( 'Tcl_CreateInterp failed' )
		if Tcl_Init( self.__interp ) != TCL_OK:
			sys.panic( 'Tcl_Init failed' )
		if Tk_Init( self.__interp ) != TCL_OK:
			sys.panic( 'Tk_Init failed' )

	def __del__( self ) -> None:
		Tcl_DeleteInterp( self.__interp )

	def eval( self, script: str ) -> bool:
		return Tcl_Eval( self.__interp, script.get_cstr() ) == TCL_OK

	def last_error( self ) -> str:
		# NOT sys.cstrlen() - that goes through windows.ntdll.strnlen, which
		# fails to link in a real no-CRT build (Microsoft's own ntdll.lib
		# stub doesn't actually export the undocumented strnlen the DLL
		# itself has - confirmed via a real link failure this session,
		# masked by sys.cstrlen's own has_symbol()-style probes because a
		# probe's default-CRT-linked compile silently resolves strnlen from
		# the CRT instead of ntdll; a real no-CRT program has no such
		# fallback). Flagged separately - counting bytes by hand here avoids
		# depending on it.
		msg: ConstPtr[u8] = Tcl_GetStringResult( self.__interp )
		length: usize = 0
		while length < 8192 and msg[length]:
			with compiler.wrap_arithmetic:
				length += 1
		with compiler.wrap_arithmetic:
			size: usize = length + 1
		return str.from_cstr( msg, size ).unwrap( 'invalid UTF-8 in Tcl error result' )

	def title( self, text: str ) -> bool:
		return self.eval( f'wm title . {{{text}}}' )

	def create_command( self, name: str, proc: TclCommandProc ) -> None:
		Tcl_CreateCommand( self.__interp, name.get_cstr(), proc, None, None )

	def mainloop( self ) -> None:
		''' blocks until the user closes the last toplevel window - same
		"run until Tk_GetNumMainWindows() hits 0" shape real tkinter's
		mainloop() uses. '''
		while Tk_GetNumMainWindows() > 0:
			Tcl_DoOneEvent( TCL_ALL_EVENTS )

	def pump_events( self, max_events: i32 ) -> None:
		''' non-blocking: processes up to `max_events` already-pending
		events without waiting for more. For tests/headless callers that
		need widget state (e.g. after .invoke()) to settle without calling
		the real, blocking mainloop(). '''
		i: i32 = 0
		while i < max_events:
			Tcl_DoOneEvent( TCL_DONT_WAIT )
			with compiler.wrap_arithmetic:
				i += 1

	def interp( self ) -> Tcl_Interp:
		return self.__interp


# module-level counter so multiple Button()s in one program get distinct
# Tcl widget paths (".b0", ".b1", ...) without needing any per-instance
# state beyond a plain string - not thread-safe, fine for this PoC.
_next_button_id: i32 = 0


class Button:
	__root: Tk
	__path: str

	def __init__( self, root: Tk, text: str ) -> None:
		global _next_button_id
		self.__root = root
		my_id: i32 = _next_button_id
		with compiler.wrap_arithmetic:
			_next_button_id += 1
		self.__path = f'.b{int( my_id )}'
		if not root.eval( f'button {self.__path} -text {{{text}}}' ):
			sys.panic( f'failed to create Button: {root.last_error()}' )

	def pack( self ) -> bool:
		return self.__root.eval( f'pack {self.__path}' )

	def bind_command( self, proc: TclCommandProc ) -> bool:
		''' registers `proc` as a Tcl command and wires it as this button's
		-command. `proc` must be a bare top-level function/@staticmethod
		reference matching TclCommandProc's exact signature - see the
		module docstring. '''
		cmd_name: str = f'{self.__path}_cmd'
		self.__root.create_command( cmd_name, proc )
		return self.__root.eval( f'{self.__path} configure -command {cmd_name}' )

	def invoke( self ) -> bool:
		return self.__root.eval( f'{self.__path} invoke' )
