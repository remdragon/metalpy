'''
pty - PTY allocation + attaching a child process to it, for interactive
terminal passthrough (a real shell, not a pipe).

Linux only for now (has_library-gated on POSIX_SPAWN_SETSID, a glibc/Linux
extension since 2.26 - see spawn_attached's own comment for why it's load-
bearing here, not just a nicety). macOS: poison pill, same convention as
lib/random.py's own macOS section - a real def so anything that merely
imports this module still compiles there, just panics if actually called.
Windows ConPTY is a different enough mechanism (no posix_openpt/posix_spawn
at all) that it isn't attempted here - out of scope for this module.

No fork()+exec() anywhere in this file, deliberately: this codebase's own
lib/threading.py wraps real OS threads (pthread_create), and any program
using lib/reactor.py or lib/threading.py's ThreadPool has other live
threads by construction. fork() only clones the calling thread - a lock
another thread held at the instant of fork() (e.g. glibc malloc's own
internal arena lock, taken on every allocation) stays held forever in the
child, with no thread left alive to release it; the next allocation this
runtime's own RC bookkeeping does in the child deadlocks. posix_spawn(2)
never runs any of this process's code in the child at all - only glibc's
own fixed, small file_actions/attr vocabulary runs between clone() and
execve() - so none of that applies.
'''

import compiler
import fs
import sys

posix_spawn_file_actions_t = compiler.c_type( 'posix_spawn_file_actions_t', header = 'spawn.h' )
posix_spawnattr_t = compiler.c_type( 'posix_spawnattr_t', header = 'spawn.h' )
winsize_t = compiler.c_type( 'struct winsize', header = 'sys/ioctl.h' )

# real `char`, distinct from our u8/i8 - needed to exactly match posix_spawn's
# `char *const argv[]`/`const char*`: a single-level void* implicitly
# converts to/from any object pointer type in C (the trick every other
# char*-shaped extern in this codebase uses - see lib/crt.py's mkdir/
# readlink), but that stops working at posix_spawn's argv double
# indirection - void** does NOT implicitly convert to char*const*, only a
# real char** does (a legal one-step qualification-add conversion).
Char = compiler.c_type( 'char', header = 'spawn.h' )

if compiler.target.os != 'windows' and compiler.target.os != 'macos':
	O_RDWR: i32 = compiler.cexpr( 'O_RDWR', 'fcntl.h', i32 )
	O_NOCTTY: i32 = compiler.cexpr( 'O_NOCTTY', 'fcntl.h', i32 )
	POSIX_SPAWN_SETSID: i16 = compiler.cexpr( 'POSIX_SPAWN_SETSID', 'spawn.h', i16 )
	TIOCSWINSZ: u64 = compiler.cexpr( 'TIOCSWINSZ', 'sys/ioctl.h', u64 )


# ---------------------------------------------------------------------------
# posix_openpt/grantpt/unlockpt/ptsname - XSI (_XOPEN_SOURCE>=600), hidden
# under this codebase's plain -std=c11 build the same way lib/crt.py's
# readlink already is - hand-declared without header=, own prototype used
# verbatim rather than fighting glibc's feature-test macros.
# ---------------------------------------------------------------------------

@compiler.target( os = not ( 'windows', 'macos' ))
@extern( 'c', 'posix_openpt' )
def posix_openpt( flags: i32 ) -> i32:
	...

@compiler.target( os = not ( 'windows', 'macos' ))
@extern( 'c', 'grantpt' )
def grantpt( fd: i32 ) -> i32:
	...

@compiler.target( os = not ( 'windows', 'macos' ))
@extern( 'c', 'unlockpt' )
def unlockpt( fd: i32 ) -> i32:
	...

@compiler.target( os = not ( 'windows', 'macos' ))
@extern( 'c', 'ptsname' )
def ptsname( fd: i32 ) -> ConstPtr[u8]:
	...


# ---------------------------------------------------------------------------
# posix_spawn family
# ---------------------------------------------------------------------------

@compiler.target( os = not ( 'windows', 'macos' ))
@extern( 'c', 'posix_spawn_file_actions_init', header = 'spawn.h' )
def _fa_init( fa: Ptr[posix_spawn_file_actions_t] ) -> i32:
	...

@compiler.target( os = not ( 'windows', 'macos' ))
@extern( 'c', 'posix_spawn_file_actions_addopen', header = 'spawn.h' )
def _fa_addopen( fa: Ptr[posix_spawn_file_actions_t], fd: i32, path: ConstPtr[Char], flags: i32, mode: i32 ) -> i32:
	...

@compiler.target( os = not ( 'windows', 'macos' ))
@extern( 'c', 'posix_spawn_file_actions_adddup2', header = 'spawn.h' )
def _fa_adddup2( fa: Ptr[posix_spawn_file_actions_t], fd: i32, newfd: i32 ) -> i32:
	...

@compiler.target( os = not ( 'windows', 'macos' ))
@extern( 'c', 'posix_spawn_file_actions_addclose', header = 'spawn.h' )
def _fa_addclose( fa: Ptr[posix_spawn_file_actions_t], fd: i32 ) -> i32:
	...

@compiler.target( os = not ( 'windows', 'macos' ))
@extern( 'c', 'posix_spawn_file_actions_destroy', header = 'spawn.h' )
def _fa_destroy( fa: Ptr[posix_spawn_file_actions_t] ) -> i32:
	...

@compiler.target( os = not ( 'windows', 'macos' ))
@extern( 'c', 'posix_spawnattr_init', header = 'spawn.h' )
def _attr_init( attr: Ptr[posix_spawnattr_t] ) -> i32:
	...

@compiler.target( os = not ( 'windows', 'macos' ))
@extern( 'c', 'posix_spawnattr_setflags', header = 'spawn.h' )
def _attr_setflags( attr: Ptr[posix_spawnattr_t], flags: i16 ) -> i32:
	...

@compiler.target( os = not ( 'windows', 'macos' ))
@extern( 'c', 'posix_spawnattr_destroy', header = 'spawn.h' )
def _attr_destroy( attr: Ptr[posix_spawnattr_t] ) -> i32:
	...

@compiler.target( os = not ( 'windows', 'macos' ))
@extern( 'c', 'posix_spawn', header = 'spawn.h' )
def _posix_spawn(
	pid: Ptr[i32],
	path: ConstPtr[Char],
	fa: Ptr[posix_spawn_file_actions_t],
	attr: Ptr[posix_spawnattr_t],
	argv: Ptr[Ptr[Char]],
	envp: Ptr[Ptr[Char]],
) -> i32:
	...

@compiler.target( os = not ( 'windows', 'macos' ))
@extern( 'c', 'waitpid', header = 'sys/wait.h' )
def waitpid( pid: i32, status: Ptr[i32], options: i32 ) -> i32:
	...

# not a general ioctl binding - one purpose-built extern for TIOCSWINSZ
# only, same posture as lib/posix/fcntl.py's own fcntl() (fixed third-arg
# type; ioctl's real third arg type depends on the request, header= would
# fail to match a fixed-arity prototype against ioctl's real variadic one).
# header='sys/ioctl.h' (unlike lib/posix/fcntl.py's own fcntl(), which
# deliberately avoids header= since ITS real prototype is fixed-arity-
# incompatible) - here it's needed anyway: winsize_t's own c_type() above
# already pulls sys/ioctl.h in, and a second, un-headered prototype for the
# same real (variadic) ioctl() would conflict with it. The real prototype's
# variadic third argument accepts our Ptr[winsize_t] with no cast needed
# (default argument promotion, no strict vararg type-check in C).
@compiler.target( os = not ( 'windows', 'macos' ))
@extern( 'c', 'ioctl', header = 'sys/ioctl.h' )
def _ioctl_winsize( fd: i32, request: u64, ws: Ptr[winsize_t] ) -> i32:
	...


class PtyPair:
	master: fs.FD
	slave_path: str

	def __init__( self, master: fs.FD, slave_path: str ) -> None:
		self.master = master
		self.slave_path = slave_path


@compiler.target( os = not ( 'windows', 'macos' ))
def open_pty() -> Result[PtyPair, OSError]:
	''' allocates a PTY master/slave pair. The slave is identified by path
	only (not opened here) - spawn_attached opens it fresh inside the
	child, which is what lets the kernel auto-assign it as the child's
	controlling terminal (see spawn_attached's own comment). '''
	from crt import get_errno
	master: i32 = posix_openpt( O_RDWR | O_NOCTTY )
	if master < 0:
		return Result.Err( OSError( get_errno() ))
	if grantpt( master ) != 0:
		err0: OSError = OSError( get_errno() )
		fs.close_raw( master ).is_ok()
		return Result.Err( err0 )
	if unlockpt( master ) != 0:
		err1: OSError = OSError( get_errno() )
		fs.close_raw( master ).is_ok()
		return Result.Err( err1 )
	raw_path: ConstPtr[u8] = ptsname( master )
	if raw_path is None:
		err2: OSError = OSError( get_errno() )
		fs.close_raw( master ).is_ok()
		return Result.Err( err2 )
	with compiler.wrap_arithmetic:
		path_len: usize = sys.cstrlen( raw_path, usize( 256 )) + usize( 1 )
	path: str = str.from_cstr( raw_path, path_len ).unwrap( 'ptsname: not valid UTF-8' )
	return Result.Ok( PtyPair( master, path ))


@compiler.target( os = not ( 'windows', 'macos' ))
def spawn_attached( argv: list[str], slave_path: str ) -> Result[i32, OSError]:
	''' spawns argv[0] (argv used as the full child argument list) attached
	to slave_path as its controlling terminal, via posix_spawn - no
	fork(), see this module's own header comment for why that matters.

	Controlling-terminal trick: posix_spawn never runs any of this
	process's code in the child, so there's no window to call setsid() +
	ioctl(TIOCSCTTY) there the way a fork()+exec() implementation would.
	Instead: POSIX_SPAWN_SETSID (glibc/Linux extension) makes the child
	call setsid() before exec(), and the very next file action opens
	slave_path BY PATH (not a dup of an already-open fd) onto fd 0 - the
	kernel auto-assigns a terminal as the controlling one on the first
	open() of it (without O_NOCTTY) by a session leader that doesn't have
	one yet, so this specific open is exactly that first open. No
	ioctl(TIOCSCTTY) needed at all. Verified against a real WSL gcc build
	(see terminal_passthrough_investigation.md) - a real interactive bash
	came up with working job control this way. '''
	from crt import get_errno

	if len( argv ) == 0:
		return Result.Err( OSError.Invalid )

	fa: Ptr[posix_spawn_file_actions_t] = sys.alloc[posix_spawn_file_actions_t]( 1 )
	_fa_init( fa )
	slave_cstr: ConstPtr[u8] = slave_path.get_cstr()
	_fa_addopen( fa, 0, compiler.cast( ConstPtr[Char], slave_cstr ), O_RDWR, 0 )
	_fa_adddup2( fa, 0, 1 )
	_fa_adddup2( fa, 0, 2 )

	attr: Ptr[posix_spawnattr_t] = sys.alloc[posix_spawnattr_t]( 1 )
	_attr_init( attr )
	_attr_setflags( attr, POSIX_SPAWN_SETSID )

	n: usize = len( argv )
	with compiler.wrap_arithmetic:
		c_argv: Ptr[Ptr[Char]] = compiler.cast( Ptr[Ptr[Char]], sys.alloc[Ptr[u8]]( n + usize( 1 )))
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < n:
			arg: str = argv.__getitem__( i ).unwrap( 'spawn_attached: argv index in bounds by construction' )
			c_argv[i] = compiler.cast( Ptr[Char], compiler.cast( Ptr[u8], arg.get_cstr() ))
			i += usize( 1 )
	c_argv[n] = None

	pid: i32 = 0
	path0: str = argv.__getitem__( usize( 0 )).unwrap( 'spawn_attached: argv non-empty, checked above' )
	rc: i32 = _posix_spawn( compiler.addrof( pid ), compiler.cast( ConstPtr[Char], path0.get_cstr() ), fa, attr, c_argv, None )

	_fa_destroy( fa )
	_attr_destroy( attr )
	sys.free( compiler.cast( Ptr[None], fa ))
	sys.free( compiler.cast( Ptr[None], attr ))
	sys.free( compiler.cast( Ptr[None], c_argv ))

	if rc != 0:
		return Result.Err( OSError( rc ))
	return Result.Ok( pid )


@compiler.target( os = not ( 'windows', 'macos' ))
def resize( master: fs.FD, rows: u16, cols: u16 ) -> Result[None, OSError]:
	''' sets the PTY's window size via TIOCSWINSZ on the master fd. The
	kernel delivers SIGWINCH to the slave's foreground process group as a
	side effect of this ioctl - nothing needed on lib/signal.py's side,
	the shell/child gets it automatically. '''
	from crt import get_errno
	ws: Ptr[winsize_t] = sys.alloc[winsize_t]( 1 )
	compiler.c_field_set( ws, 'ws_row', rows )
	compiler.c_field_set( ws, 'ws_col', cols )
	compiler.c_field_set( ws, 'ws_xpixel', u16( 0 ))
	compiler.c_field_set( ws, 'ws_ypixel', u16( 0 ))
	rc: i32 = _ioctl_winsize( master, TIOCSWINSZ, ws )
	sys.free( compiler.cast( Ptr[None], ws ))
	if rc != 0:
		return Result.Err( OSError( get_errno() ))
	return Result.Ok( None )


@compiler.target( os = not ( 'windows', 'macos' ))
def wait( pid: i32 ) -> Result[i32, OSError]:
	''' blocks until pid exits, returns its raw wait status (caller decodes
	via WIFEXITED/WEXITSTATUS-equivalent if needed - not attempted here,
	no existing binding for those macros in this codebase yet). '''
	from crt import get_errno
	status: i32 = 0
	rc: i32 = waitpid( pid, compiler.addrof( status ), 0 )
	if rc < 0:
		return Result.Err( OSError( get_errno() ))
	return Result.Ok( status )


@compiler.target( os = 'macos' )
def open_pty() -> Result[PtyPair, OSError]:
	return _MACOS_PTY_NOT_YET_IMPLEMENTED()

@compiler.target( os = 'macos' )
def spawn_attached( argv: list[str], slave_path: str ) -> Result[i32, OSError]:
	return _MACOS_PTY_NOT_YET_IMPLEMENTED()

@compiler.target( os = 'macos' )
def resize( master: fs.FD, rows: u16, cols: u16 ) -> Result[None, OSError]:
	return _MACOS_PTY_NOT_YET_IMPLEMENTED()

@compiler.target( os = 'macos' )
def wait( pid: i32 ) -> Result[i32, OSError]:
	return _MACOS_PTY_NOT_YET_IMPLEMENTED()
