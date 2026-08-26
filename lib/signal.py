'''
signal - process-level signal handling: a permanent, process-wide registry
(signal.signal()) plus a scoped, reentrant override (signal.context()).

metalpy has no exceptions (SYNTAX.md's "Zero Exceptions"), so there is no
try/except KeyboardInterrupt to hang a catch off of. Default behavior for
any signal nothing has registered for is metalpy's own normal,
deterministic process termination - exactly as if this module were never
imported at all.

signal.signal(sig, handler) is a PERMANENT, process-wide registration
(mirrors real Python's own signal.signal) - `handler` stays installed until
changed again by another signal.signal() call, for the life of the process:

	def on_sigint( sig: i32 ) -> None:
		...
	signal.signal( signal.SIGINT, on_sigint )

signal.context(sig, flag) is a SCOPED, per-thread override on top of that:
while the with-block is active on THIS thread, the signal sets `flag`
instead of whatever signal.signal() (or an outer context()) would otherwise
do; the moment the block exits (normally, or via defer-style unwind), the
previous registration - an outer context() on the same thread, or the
process-wide signal.signal() one, or nothing at all - takes back over.
Reentrant/nestable, including nesting the SAME signal on the SAME thread:
each context() only ever remembers and restores whatever ITS OWN thread's
own prior registration for its own signal was.

	flag = signal.Flag()
	with signal.context( signal.SIGINT, flag ):
		while not flag.get():
			compute_chunk()
	# outside the block, this thread's previous registration (signal.signal()'s,
	# or an outer context()'s, or - if neither - real default disposition) resumes

Priority when a signal is actually delivered, checked in this order:
this thread's own context()-scoped registration, then the process-wide
signal.signal() registration, then (if neither exists) real OS default
disposition (SIG_DFL) - restored and re-raised on the spot, so an
unregistered signal behaves exactly as if this module were never imported.

Plain ANSI C signal() has one-shot ("unreliable signal") semantics - the
standard itself specifies a signal's disposition resets to SIG_DFL as soon
as it's delivered, before the handler even runs (confirmed as real,
observed behavior on this codebase's own Windows CRT target too, not just
a standard technicality never actually exercised). The real OS-level
trampoline (_signal_handler below) therefore reinstalls itself as its own
first action, every single time it runs - without that, a SECOND delivery
of the same signal, after the first one already ran, would silently fall
through to real SIG_DFL instead of ever reaching this module again.

Only SIGINT is unit-tested (the only signal whose value/semantics real
Python's own `signal` module and this module can both rely on being
identical across POSIX and Windows, unlike SIGKILL/SIGTERM/... , which are
POSIX-only) - but nothing here is SIGINT-specific; any POSIX signal number
works, e.g. for a Linux-only program that wants SIGHUP/SIGTERM/etc.

Two residual, deliberately-accepted risks, both a consequence of the real
OS-level handler (see _signal_handler below) running as a genuine
async-signal-context callback, where only a small set of operations are
well-defined:
  - the process-wide `_SignalHandler` registry is a locked dict[K,V] (it's
    genuinely shared across threads, so it needs the lock) - a signal
    delivered to a thread that's already mid-signal.signal() call, holding
    that same lock, would deadlock. Accepted, matching this module's
    previous `_handler`'s own aspirational-not-guaranteed framing.
  - the per-thread `_TlsSignalHandler` registry is deliberately a plain,
    UNLOCKED UnsafeDict (not dict[K,V]) specifically to avoid that same
    hazard on the much hotter context() path - it's never touched by any
    thread but its own, so a lock there would buy no real protection.
'''

import compiler
import sys
import threading

SIGINT: i32 = compiler.cexpr( 'SIGINT', 'signal.h', i32 )

_HandlerFn: TypeAlias = Ptr[Callable[[i32], None]]

@extern( 'c', 'signal', header = 'signal.h' )
def _signal( signum: i32, handler: _HandlerFn ) -> _HandlerFn:
	...

@extern( 'c', 'raise', header = 'signal.h' )
def _raise( sig: i32 ) -> i32:
	...


class Flag:
	''' the thing a signal.context() block redirects a signal INTO, instead
	of letting it terminate the process (or run whatever else was
	registered). Owns no OS resources of its own - just a plain bool a
	context() block points the shared C-level trampoline at while it's the
	active registration for its signal, on its thread. '''
	__value: bool

	def __init__( self ) -> None:
		self.__value = False

	def get( self ) -> bool:
		return self.__value

	@inline
	def __bool__( self ) -> bool:
		return self.__value

	def _set( self ) -> None:
		self.__value = True

	def _clear( self ) -> None:
		self.__value = False


# process-wide, PERMANENT - written only by signal.signal(), read by
# _signal_handler as its second-priority fallback (after this thread's own
# context()-scoped override, if any). See module docstring for the
# accepted same-thread-reentrant-lock-deadlock risk this implies.
_SignalHandler: dict[i32, _HandlerFn] = dict[i32, _HandlerFn]()

# one per OS thread, written only by context(): signum -> the Flag that
# thread's own (possibly nested) context() block wants set. Deliberately
# UnsafeDict[i32, Flag] (not the locked dict[K,V], not a Callable/Closure
# value) - see module docstring for why the lock is skipped, and
# context()'s own comment for why Flag (context() only ever needs to do
# ONE thing: flag._set()) beats inventing a closure-capture scheme here.
_TlsSignalHandler: threading.ThreadLocal[UnsafeDict[i32, Flag]] = threading.ThreadLocal[UnsafeDict[i32, Flag]]()


def _signal_handler( sig: i32 ) -> None:
	''' the ONE real OS-level signal handler ever installed (by signal.signal()
	or context.__enter__(), both via a plain, idempotent _signal(sig, _signal_handler)
	call - see their own comments) - never called directly by user code.
	Reinstalls itself first (see module docstring), then looks up whichever
	registration currently owns `sig` (this thread's own context() override
	first, then the process-wide signal.signal() one) and calls it; if
	neither exists, restores real default disposition and re-raises, so an
	unregistered signal behaves exactly as if this module were never
	imported - then reinstalls itself a second time, only reached if that
	re-raise's default action didn't terminate the process (e.g. a POSIX
	signal whose default is Ignore, not Terminate). '''
	_signal( sig, _signal_handler )   # ANSI C signal()'s one-shot semantics - see module docstring
	tls: UnsafeDict[i32, Flag]|None = _TlsSignalHandler.get()
	if tls is not None:
		if sig in tls:
			tls.__getitem__( sig ).unwrap( '_signal_handler: sig just confirmed present in tls' )._set()
			return
	if sig in _SignalHandler:
		_SignalHandler.__getitem__( sig ).unwrap( '_signal_handler: sig just confirmed present in _SignalHandler' )( sig )
		return
	sig_dfl: _HandlerFn = 0   # SIG_DFL == (void(*)(int))0 on every real ANSI C implementation
	_signal( sig, sig_dfl )
	_raise( sig )
	_signal( sig, _signal_handler )   # only reached if sig's default action doesn't terminate the process - the top-of-function reinstall above was just undone by the sig_dfl call two lines up


def signal( sig: i32, handler: _HandlerFn ) -> None:
	''' permanent, process-wide registration - see module docstring, and
	context() for a scoped alternative. Registry write happens BEFORE the
	OS-level install (not after): a real signal delivered in between would
	otherwise find nothing registered yet and fall through to the default-
	and-reraise path, defeating the registration that was mid-flight. '''
	_SignalHandler[sig] = handler
	_signal( sig, _signal_handler )   # idempotent - harmless if sig's trampoline is already installed


class context:
	''' with signal.context( sig, flag ): ... - see module docstring.
	Reentrant/nestable, including nesting the SAME signal on the SAME
	thread: __enter__ saves whatever THIS thread's own registration for
	`sig` was before (an outer context(), or nothing) and __exit__ restores
	it - a signal.signal() registration for the same `sig` is never
	touched (it lives in the separate, process-wide _SignalHandler
	registry, not this thread's own _TlsSignalHandler one), so exiting a
	context() with no outer one on this thread correctly falls back to it. '''
	__sig: i32
	__flag: Flag
	__prev_flag: Flag|None
	__installed: bool

	def __init__( self, sig: i32, flag: Flag ) -> None:
		self.__sig = sig
		self.__flag = flag
		self.__prev_flag = None
		self.__installed = False

	def __enter__( self ) -> None:
		existing: UnsafeDict[i32, Flag]|None = _TlsSignalHandler.get()
		tls: UnsafeDict[i32, Flag]
		if existing is None:
			tls = UnsafeDict[i32, Flag]()
			# ThreadLocal[T].set() is deliberately non-owning (a bare per-
			# thread bookmark, see threading.py's own module comment) - this
			# freshly-constructed dict has no other owner anywhere, so
			# without this explicit permanent identity reference it would be
			# freed the instant this function returns, leaving the TLS slot
			# dangling (the exact heap-use-after-free fiber.py's own
			# _thread_fiber_handle hit before its own equivalent incref).
			compiler.incref( tls )
			_TlsSignalHandler.set( tls )
		else:
			tls = existing
		if self.__sig in tls:
			self.__prev_flag = tls.__getitem__( self.__sig ).unwrap( 'context.__enter__: sig just confirmed present' )
		else:
			self.__prev_flag = None
		self.__flag._clear()
		tls[self.__sig] = self.__flag   # registry write BEFORE the OS-level install - see signal()'s own comment
		_signal( self.__sig, _signal_handler )
		self.__installed = True

	def __exit__( self ) -> None:
		if not self.__installed:
			return
		tls: UnsafeDict[i32, Flag]|None = _TlsSignalHandler.get()
		if tls is None:
			sys.panic( 'signal.context.__exit__: thread-local handler registry unexpectedly missing' )
		prev: Flag|None = self.__prev_flag
		if prev is not None:
			tls[self.__sig] = prev
		else:
			tls.__delitem__( self.__sig ).unwrap( 'context.__exit__: own entry unexpectedly missing' )
			if tls.__len__() == 0:
				# last registration on this thread just went away - release
				# the permanent extra reference __enter__'s first-ever call
				# added (see its own comment) and clear the slot, so this
				# thread leaves no trace once it stops using signal.context()
				# entirely; a later context() on the same thread just
				# recreates it via the same is-None branch.
				_TlsSignalHandler.clear()
				compiler.decref( tls )
		self.__installed = False
