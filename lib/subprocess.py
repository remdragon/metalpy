# lib/subprocess.py — spawn a child process, optionally capture its
# stdout/stderr, and wait for its exit code. Mirrors Python's
# subprocess.run()/CompletedProcess for the common case; NOT ported:
# Popen's own streaming/incremental-write API, shell=True, stdin
# redirection (the child always inherits this process' stdin, matching
# subprocess.run()'s own default when stdin= is omitted), env= overrides
# (the child always inherits this process' environment).
#
# Windows: CreateProcessA + anonymous pipes for capture_output.
# POSIX:   posix_spawn + pipe() - same "no fork()" posture as lib/pty.py
#          (see its own header comment for why: a lock another thread held
#          at the instant of fork() stays held forever in the child, with
#          no thread left to release it - posix_spawn never runs any of
#          this process' code in the child, so that risk doesn't apply).
#
# capture_output reads stdout and stderr concurrently (main thread drains
# one, a spawned lib/threading.Thread drains the other) - reading them
# sequentially can deadlock if the child fills one pipe's OS buffer while
# blocked writing to it, while this process is still blocked reading the
# OTHER pipe.

import compiler
import fs
import sys
import threading


class CompletedProcess:
	args:       list[str]
	returncode: i32
	stdout:     bytes
	stderr:     bytes

	def __init__( self, args: list[str], returncode: i32, stdout: bytes, stderr: bytes ) -> None:
		self.args = args
		self.returncode = returncode
		self.stdout = stdout
		self.stderr = stderr


class _RunResult:
	returncode: i32
	stdout:     bytearray
	stderr:     bytearray

	def __init__( self, returncode: i32, stdout: bytearray, stderr: bytearray ) -> None:
		self.returncode = returncode
		self.stdout = stdout
		self.stderr = stderr


def run( args: list[str], capture_output: bool = False, cwd: str|None = None ) -> Result[CompletedProcess, OSError]:
	''' spawns args[0] with args as its full argv, waits for it to exit.
	capture_output=True redirects the child's stdout/stderr into pipes and
	returns their full contents; False (default) leaves them inherited from
	this process (matching real subprocess.run()'s own default). cwd, if
	given, is the child's working directory (this process' own cwd is left
	untouched either way). '''
	if len( args ) == 0:
		return Result.Err( OSError.Invalid )
	result: _RunResult = _spawn_and_wait( args, capture_output, cwd ).or_return()
	return Result.Ok( CompletedProcess( args, result.returncode, bytes( result.stdout ), bytes( result.stderr )))


# ---------------------------------------------------------------------------
# _PipeDrain: reads one pipe to completion on whichever thread runs it - see
# this module's own header comment for why capture_output needs two of
# these running concurrently (one on a spawned Thread, one on the caller's
# own thread) rather than reading stdout then stderr sequentially.
# ---------------------------------------------------------------------------

class _PipeDrain:
	__fd:    fs.FD
	result:  Result[bytearray, OSError]

	def __init__( self, fd: fs.FD ) -> None:
		self.__fd = fd
		self.result = Result.Ok( bytearray( usize( 0 )))

	def run( self ) -> None:
		self.result = _read_all( self.__fd )


def _read_all( fd: fs.FD ) -> Result[bytearray, OSError]:
	''' reads fd to EOF into a freshly-grown bytearray. '''
	out: bytearray = bytearray( usize( 0 ))
	cap: usize = 4096
	buf: Ptr[u8] = sys.alloc[u8]( cap )
	defer( sys.free( compiler.cast( Ptr[None], buf )))
	while True:
		n: usize = _pipe_read_or_eof( fd, buf, cap ).or_return()
		if n == usize( 0 ):
			break
		old_len: usize = out.__len__()
		with compiler.wrap_arithmetic:
			new_len: usize = old_len + n
			out.resize( new_len )
			sys.memcpy( out.get_ptr() + old_len, buf, n )
	return Result.Ok( out )


@compiler.target( os = 'windows' )
def _pipe_read_or_eof( fd: fs.FD, buf: Ptr[u8], cap: usize ) -> Result[usize, OSError]:
	''' ReadFile on an anonymous pipe signals EOF as a FAILED call with
	ERROR_BROKEN_PIPE (109) once the write end closes - NOT a successful
	zero-byte read the way POSIX read(2)/this compiler's own fs.read_raw
	convention does. Translated to Ok(0) here so callers (_read_all) can
	treat both platforms identically. '''
	from windows.kernel32 import ReadFile, GetLastError
	with compiler.saturate_arithmetic:
		to_read: u32 = u32( cap )
	bytes_read: u32 = 0
	success: bool = ReadFile( fd, buf, to_read, compiler.addrof( bytes_read ), None )
	if success:
		with compiler.wrap_arithmetic:
			return Result.Ok( usize( bytes_read ))
	err: u32 = GetLastError()
	if err == u32( 109 ):  # ERROR_BROKEN_PIPE
		return Result.Ok( usize( 0 ))
	return Result.Err( OSError( i32( err )))

@compiler.target( os = not 'windows' )
def _pipe_read_or_eof( fd: fs.FD, buf: Ptr[u8], cap: usize ) -> Result[usize, OSError]:
	''' POSIX read(2) already returns Ok(0) at EOF - fs.read_raw as-is. '''
	return fs.read_raw( fd, buf, cap )


# ---------------------------------------------------------------------------
# Windows backend
# ---------------------------------------------------------------------------

@compiler.target( os = 'windows' )
def _win32_quote_arg( arg: str ) -> str:
	''' one argument's worth of the Microsoft C/C++ command-line quoting
	convention (the algorithm every CRT startup's argv parser assumes,
	including this compiler's own CommandLineToArgvW-based sys.argv - see
	lib/sys.py's _build_argv) - same algorithm as Python's own
	subprocess.list2cmdline, ported per-argument. Backslashes are only
	special immediately before a quote (either an embedded one or the
	closing one this function adds) - literal backslashes elsewhere (e.g.
	a plain Windows path) pass through untouched. '''
	needs_quotes: bool = arg == '' or ' ' in arg or '\t' in arg
	parts: list[str] = list[str]()
	if needs_quotes:
		parts.append( '"' )
	bs_run: usize = 0
	i: usize = 0
	n: usize = arg.__len__()
	with compiler.wrap_arithmetic:
		while i < n:
			c: str = arg.__getitem__( i ).unwrap( 'i < len by construction' )
			if c == '\\':
				bs_run += usize( 1 )
			elif c == '"':
				j: usize = 0
				while j < bs_run:
					parts.append( '\\\\' )
					j += usize( 1 )
				bs_run = usize( 0 )
				parts.append( '\\"' )
			else:
				j2: usize = 0
				while j2 < bs_run:
					parts.append( '\\' )
					j2 += usize( 1 )
				bs_run = usize( 0 )
				parts.append( c )
			i += usize( 1 )
		if needs_quotes:
			# trailing run of backslashes must be doubled before the
			# CLOSING quote too, same as before an embedded one
			j3: usize = 0
			while j3 < bs_run:
				parts.append( '\\\\' )
				j3 += usize( 1 )
			parts.append( '"' )
		else:
			j4: usize = 0
			while j4 < bs_run:
				parts.append( '\\' )
				j4 += usize( 1 )
	return ''.join( parts )

@compiler.target( os = 'windows' )
def _win32_cmdline( args: list[str] ) -> str:
	quoted: list[str] = list[str]()
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < args.__len__():
			quoted.append( _win32_quote_arg( args.__getitem__( i ).unwrap( 'i < len by construction' )))
			i += usize( 1 )
	return ' '.join( quoted )


@compiler.target( os = 'windows' )
def _spawn_and_wait( args: list[str], capture_output: bool, cwd: str|None ) -> Result[_RunResult, OSError]:
	from windows.kernel32 import (
		CreateProcessA, CreatePipe, SetHandleInformation, GetExitCodeProcess,
		WaitForSingleObject, CloseHandle, GetLastError,
		GetStdHandle, STD_INPUT_HANDLE,
		STARTUPINFOA, PROCESS_INFORMATION, SECURITY_ATTRIBUTES,
		STARTF_USESTDHANDLES, HANDLE_FLAG_INHERIT, INFINITE,
	)

	cmdline: str = _win32_cmdline( args )
	cmd_len: usize = cmdline.byte_len()
	with compiler.wrap_arithmetic:
		cmd_buf: Ptr[u8] = sys.alloc[u8]( cmd_len + usize( 1 ))  # mutable - CreateProcessA may rewrite this buffer in place
	sys.memcpy( cmd_buf, cmdline.get_cstr(), cmd_len )
	cmd_buf[cmd_len] = u8( 0 )

	si: STARTUPINFOA = STARTUPINFOA()
	with compiler.wrap_arithmetic:
		si.cb = u32( compiler.sizeof( STARTUPINFOA ))
	pi: PROCESS_INFORMATION = PROCESS_INFORMATION()

	stdout_read:  Ptr[None] = None
	stdout_write: Ptr[None] = None
	stderr_read:  Ptr[None] = None
	stderr_write: Ptr[None] = None

	if capture_output:
		sa: SECURITY_ATTRIBUTES = SECURITY_ATTRIBUTES()
		with compiler.wrap_arithmetic:
			sa.nLength = u32( compiler.sizeof( SECURITY_ATTRIBUTES ))
		sa.bInheritHandle = i32( 1 )

		if not CreatePipe( compiler.addrof( stdout_read ), compiler.addrof( stdout_write ), compiler.addrof( sa ), 0 ):
			return Result.Err( OSError( GetLastError() ))
		# the parent's own end must NOT be inheritable (else it leaks into
		# grandchildren spawned by THIS child, and Windows anonymous pipes
		# only reach real EOF once every write handle everywhere is
		# closed) - only the child's end (already inheritable via sa
		# above) should cross the CreateProcessA boundary.
		SetHandleInformation( stdout_read, HANDLE_FLAG_INHERIT, u32( 0 ))
		if not CreatePipe( compiler.addrof( stderr_read ), compiler.addrof( stderr_write ), compiler.addrof( sa ), 0 ):
			CloseHandle( stdout_read )
			CloseHandle( stdout_write )
			return Result.Err( OSError( GetLastError() ))
		SetHandleInformation( stderr_read, HANDLE_FLAG_INHERIT, u32( 0 ))

		si.dwFlags = STARTF_USESTDHANDLES
		si.hStdOutput = stdout_write
		si.hStdError = stderr_write
		si.hStdInput = GetStdHandle( STD_INPUT_HANDLE )

	cwd_ptr: ConstPtr[u8] = None
	if cwd is not None:
		cwd_str: str = cwd
		cwd_ptr = cwd_str.get_cstr()

	success: bool = CreateProcessA(
		None, cmd_buf, None, None,
		capture_output,  # bInheritHandles - only true when pipe handles need to cross
		0, None, cwd_ptr, compiler.addrof( si ), compiler.addrof( pi ),
	)
	sys.free( compiler.cast( Ptr[None], cmd_buf ))  # no longer needed - CreateProcessA already made its own copy of the command line
	if not success:
		err: OSError = OSError( GetLastError() )
		if capture_output:
			CloseHandle( stdout_read ); CloseHandle( stdout_write )
			CloseHandle( stderr_read ); CloseHandle( stderr_write )
		return Result.Err( err )

	CloseHandle( pi.hThread )

	stdout_bytes: bytearray = bytearray( usize( 0 ))
	stderr_bytes: bytearray = bytearray( usize( 0 ))
	if capture_output:
		# the child owns the write ends now - close this process' copies
		# BEFORE reading, or the read ends never see EOF (a pipe's write
		# side isn't fully closed, for EOF purposes, until every open
		# handle to it - including this parent's own - is closed).
		CloseHandle( stdout_write )
		CloseHandle( stderr_write )
		drain: _PipeDrain = _PipeDrain( stderr_read )
		t: threading.Thread = threading.Thread( drain.run )
		stdout_result: Result[bytearray, OSError] = _read_all( stdout_read )
		t.join()
		CloseHandle( stdout_read )
		CloseHandle( stderr_read )
		stdout_bytes = stdout_result.or_return()
		stderr_bytes = drain.result.or_return()

	WaitForSingleObject( pi.hProcess, INFINITE )
	exit_code: u32 = 0
	GetExitCodeProcess( pi.hProcess, compiler.addrof( exit_code ))
	CloseHandle( pi.hProcess )

	return Result.Ok( _RunResult( i32( exit_code ), stdout_bytes, stderr_bytes ))


# ---------------------------------------------------------------------------
# POSIX backend - posix_spawn, same externs/conventions as lib/pty.py's own
# posix_spawn section (see its header comment for why no fork()).
# ---------------------------------------------------------------------------

# guarded, not unconditional like the @compiler.target-decorated functions
# below: compiler.c_type() pulls in its header eagerly at module-discovery
# time regardless of any @compiler.target gate on the functions that use
# it - spawn.h doesn't exist on Windows, so this would break a Windows
# build the moment this module is merely imported, target or not.
if compiler.target.os != 'windows' and compiler.target.os != 'macos':
	posix_spawn_file_actions_t = compiler.c_type( 'posix_spawn_file_actions_t', header = 'spawn.h' )
	posix_spawnattr_t = compiler.c_type( 'posix_spawnattr_t', header = 'spawn.h' )
	Char = compiler.c_type( 'char', header = 'spawn.h' )

@compiler.target( os = not ( 'windows', 'macos' ))
@extern( 'c', 'pipe' )
def _pipe2( fds: Ptr[i32] ) -> i32:
	...

@compiler.target( os = not ( 'windows', 'macos' ))
@extern( 'c', 'posix_spawn_file_actions_init', header = 'spawn.h' )
def _fa_init( fa: Ptr[posix_spawn_file_actions_t] ) -> i32:
	...

@compiler.target( os = not ( 'windows', 'macos' ))
@extern( 'c', 'posix_spawn_file_actions_adddup2', header = 'spawn.h' )
def _fa_adddup2( fa: Ptr[posix_spawn_file_actions_t], fd: i32, newfd: i32 ) -> i32:
	...

@compiler.target( os = not ( 'windows', 'macos' ))
@extern( 'c', 'posix_spawn_file_actions_addclose', header = 'spawn.h' )
def _fa_addclose( fa: Ptr[posix_spawn_file_actions_t], fd: i32 ) -> i32:
	...

# glibc extension (>= 2.29) - same "Linux/glibc, not portable POSIX" posture
# already accepted by lib/pty.py's own POSIX_SPAWN_SETSID (glibc >= 2.26).
# No header= - same posture as lib/pty.py's own posix_openpt/ptsname:
# spawn.h only exposes this GNU extension under _GNU_SOURCE/
# _DEFAULT_SOURCE, which this codebase's plain -std=c11 build doesn't
# define; hand-declaring the prototype sidesteps the feature-test-macro
# fight entirely.
@compiler.target( os = not ( 'windows', 'macos' ))
@extern( 'c', 'posix_spawn_file_actions_addchdir_np' )
def _fa_addchdir( fa: Ptr[posix_spawn_file_actions_t], path: ConstPtr[Char] ) -> i32:
	...

@compiler.target( os = not ( 'windows', 'macos' ))
@extern( 'c', 'posix_spawn_file_actions_destroy', header = 'spawn.h' )
def _fa_destroy( fa: Ptr[posix_spawn_file_actions_t] ) -> i32:
	...

@compiler.target( os = not ( 'windows', 'macos' ))
@extern( 'c', 'posix_spawn', header = 'spawn.h' )
def _posix_spawn(
	pid:  Ptr[i32],
	path: ConstPtr[Char],
	fa:   Ptr[posix_spawn_file_actions_t],
	attr: Ptr[posix_spawnattr_t],
	argv: Ptr[Ptr[Char]],
	envp: Ptr[Ptr[Char]],
) -> i32:
	...

@compiler.target( os = not ( 'windows', 'macos' ))
@extern( 'c', 'waitpid', header = 'sys/wait.h' )
def _waitpid( pid: i32, status: Ptr[i32], options: i32 ) -> i32:
	...


@compiler.target( os = not ( 'windows', 'macos' ))
def _spawn_and_wait( args: list[str], capture_output: bool, cwd: str|None ) -> Result[_RunResult, OSError]:
	from crt import get_errno

	n: usize = args.__len__()
	with compiler.wrap_arithmetic:
		c_argv: Ptr[Ptr[Char]] = compiler.cast( Ptr[Ptr[Char]], sys.alloc[Ptr[u8]]( n + usize( 1 )))
	defer( sys.free( compiler.cast( Ptr[None], c_argv )))
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < n:
			arg: str = args.__getitem__( i ).unwrap( 'i < len by construction' )
			c_argv[i] = compiler.cast( Ptr[Char], compiler.cast( Ptr[u8], arg.get_cstr() ))
			i += usize( 1 )
	c_argv[n] = None

	fa: Ptr[posix_spawn_file_actions_t] = sys.alloc[posix_spawn_file_actions_t]( 1 )
	defer( sys.free( compiler.cast( Ptr[None], fa )))
	_fa_init( fa )
	defer( _fa_destroy( fa ))

	stdout_fds: Ptr[i32] = sys.alloc[i32]( 2 )
	defer( sys.free( compiler.cast( Ptr[None], stdout_fds )))
	stderr_fds: Ptr[i32] = sys.alloc[i32]( 2 )
	defer( sys.free( compiler.cast( Ptr[None], stderr_fds )))

	if capture_output:
		if _pipe2( stdout_fds ) != 0:
			return Result.Err( OSError( get_errno() ))
		if _pipe2( stderr_fds ) != 0:
			fs.close_raw( stdout_fds[0] ).is_ok(); fs.close_raw( stdout_fds[1] ).is_ok()
			return Result.Err( OSError( get_errno() ))
		# child: its stdout/stderr become the pipes' write ends, then the
		# raw fds it inherited for the pipe plumbing itself (both read
		# ends, and the write ends' original numbers if they land above
		# fd 2) are closed - it should only ever see fd 0/1/2.
		_fa_adddup2( fa, stdout_fds[1], 1 )
		_fa_adddup2( fa, stderr_fds[1], 2 )
		_fa_addclose( fa, stdout_fds[0] )
		_fa_addclose( fa, stdout_fds[1] )
		_fa_addclose( fa, stderr_fds[0] )
		_fa_addclose( fa, stderr_fds[1] )

	if cwd is not None:
		cwd_str: str = cwd
		if _fa_addchdir( fa, compiler.cast( ConstPtr[Char], cwd_str.get_cstr() )) != 0:
			if capture_output:
				fs.close_raw( stdout_fds[0] ).is_ok(); fs.close_raw( stdout_fds[1] ).is_ok()
				fs.close_raw( stderr_fds[0] ).is_ok(); fs.close_raw( stderr_fds[1] ).is_ok()
			return Result.Err( OSError( get_errno() ))

	pid: i32 = 0
	path0: str = args.__getitem__( usize( 0 )).unwrap( 'args non-empty, checked by run()' )
	rc: i32 = _posix_spawn( compiler.addrof( pid ), compiler.cast( ConstPtr[Char], path0.get_cstr() ), fa, None, c_argv, None )

	if capture_output:
		# parent no longer needs the write ends - same "close before
		# read or EOF never arrives" reasoning as the Windows backend.
		fs.close_raw( stdout_fds[1] ).is_ok()
		fs.close_raw( stderr_fds[1] ).is_ok()

	if rc != 0:
		if capture_output:
			fs.close_raw( stdout_fds[0] ).is_ok()
			fs.close_raw( stderr_fds[0] ).is_ok()
		return Result.Err( OSError( rc ))

	stdout_bytes: bytearray = bytearray( usize( 0 ))
	stderr_bytes: bytearray = bytearray( usize( 0 ))
	if capture_output:
		drain: _PipeDrain = _PipeDrain( stderr_fds[0] )
		t: threading.Thread = threading.Thread( drain.run )
		stdout_result: Result[bytearray, OSError] = _read_all( stdout_fds[0] )
		t.join()
		fs.close_raw( stdout_fds[0] ).is_ok()
		fs.close_raw( stderr_fds[0] ).is_ok()
		stdout_bytes = stdout_result.or_return()
		stderr_bytes = drain.result.or_return()

	status: i32 = 0
	if _waitpid( pid, compiler.addrof( status ), 0 ) < 0:
		return Result.Err( OSError( get_errno() ))
	# WEXITSTATUS(status) - no existing binding for the WIF*/WEXITSTATUS
	# macros in this codebase yet (lib/pty.py's own wait() punts on this
	# too, returning the raw status) - inlined here rather than adding a
	# general macro binding for one caller. Only the normal-exit shape is
	# decoded; a signal-killed child's status is returned as-is (matches
	# real subprocess.run()'s own returncode convention of a NEGATIVE
	# number for "killed by signal N", though that specific encoding isn't
	# reproduced here - a real gap if a caller needs to distinguish the two).
	with compiler.wrap_arithmetic:
		exit_code: i32 = ( status >> 8 ) & 0xFF

	return Result.Ok( _RunResult( exit_code, stdout_bytes, stderr_bytes ))


@compiler.target( os = 'macos' )
def _spawn_and_wait( args: list[str], capture_output: bool, cwd: str|None ) -> Result[_RunResult, OSError]:
	return _MACOS_SUBPROCESS_NOT_YET_IMPLEMENTED()
