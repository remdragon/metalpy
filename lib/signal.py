'''
signal - process-level signal handling, scoped via signal.context().

metalpy has no exceptions (SYNTAX.md's "Zero Exceptions"), so there is no
try/except KeyboardInterrupt to hang a catch off of. Default behavior
everywhere is metalpy's own normal, deterministic process termination on
SIGINT (Ctrl+C) - exactly as if this module were never imported at all.
signal.context(SIGINT, flag) is an explicit, SCOPED opt-out: while the
with-block is active, Ctrl+C sets `flag` instead of terminating the
process; the moment the block exits (normally, or via defer-style unwind),
default termination is restored.

	flag = signal.Flag()
	with signal.context( signal.SIGINT, flag ):
		while not flag.get():
			compute_chunk()
	# outside the block, Ctrl+C reverts to normal termination

Only SIGINT is wired up - the only signal any current caller (grap.mpy)
needs, and the only one whose value/semantics real Python's own `signal`
module and this module can both rely on being identical across POSIX and
Windows (unlike SIGKILL/SIGTERM/... , which are POSIX-only). Plain ANSI C
signal()/<signal.h> covers it on every target this compiler supports - no
os-split needed, unlike most of this codebase's other OS-facing bindings.

Only one signal.context() may be active at a time (checked, panics on a
second concurrent/nested attempt) - the underlying OS-level handler is a
single process-wide registration, not one per Flag instance, so there is
no well-defined behavior for two contexts (even on two different signals)
racing to install/restore it.
'''

import compiler
import sys

SIGINT: i32 = compiler.cexpr( 'SIGINT', 'signal.h', i32 )

_SignalHandler: TypeAlias = Ptr[Callable[[i32], None]]

@extern( 'c', 'signal', header = 'signal.h' )
def _signal( signum: i32, handler: _SignalHandler ) -> _SignalHandler:
	...


class Flag:
	''' the thing a signal.context() block redirects a signal INTO, instead
	of letting it terminate the process. Owns no OS resources of its own -
	just a plain bool a context() block points the shared C-level handler
	at while it's the active one. '''
	__value: bool

	def __init__( self ) -> None:
		self.__value = False

	def get( self ) -> bool:
		return self.__value

	def _set( self ) -> None:
		self.__value = True

	def _clear( self ) -> None:
		self.__value = False


_active_flag: Flag|None = None

def _handler( sig: i32 ) -> None:
	# kept deliberately trivial (a single bool store) - this runs as a
	# real OS signal handler on POSIX, where only async-signal-safe
	# operations are well-defined; anything heavier belongs in the
	# ordinary code polling Flag.get(), not here
	if _active_flag is not None:
		_active_flag._set()


class context:
	''' with signal.context( SIGINT, flag ): ... - see module docstring. '''
	__sig: i32
	__flag: Flag
	__installed: bool

	def __init__( self, sig: i32, flag: Flag ) -> None:
		self.__sig = sig
		self.__flag = flag
		self.__installed = False

	def __enter__( self ) -> None:
		global _active_flag
		if _active_flag is not None:
			sys.panic( 'signal.context: another context is already active (nesting/concurrent contexts are not supported)' )
		self.__flag._clear()
		_active_flag = self.__flag
		_signal( self.__sig, _handler )
		self.__installed = True

	def __exit__( self ) -> None:
		global _active_flag
		if self.__installed:
			# SIG_DFL is a (void(*)(int))0 macro on every real ANSI C
			# implementation (POSIX and Windows' own CRT both define it
			# this way) - a local, not a module global: a Ptr[Callable[...]]
			# -typed GLOBAL still isn't supported by the emitter
			# (_emit_global_declaration doesn't route through _declarator's
			# own function-pointer-declarator special case; filed as
			# task_c3dd4993), unlike a plain local/parameter of that type,
			# which already works fine
			sig_dfl: _SignalHandler = 0
			_signal( self.__sig, sig_dfl )
			_active_flag = None
			self.__installed = False
