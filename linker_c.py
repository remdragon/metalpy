'''
stage 6 - linker: C compiler detection and object-file compilation.

Detects a working C compiler (clang, gcc, or MSVC's cl), respecting
the METALPY_CC environment variable override. Provides a CcTool class
that wraps the detected compiler and exposes a compile() method for
turning generated .c source files into .o object files.

See ARCHITECTURE.md line 196.
'''

# stdlib imports:
import os
from pathlib import Path
import shutil
import subprocess
import sys


def atomic_write_cache( cache_file: Path, data: 'bytes|str' ) -> None:
	''' publish a disk-cache entry so a concurrent reader sees either the
	complete previous state or the complete new one, never a half-written file.

	Path.write_text/write_bytes open with 'w', which TRUNCATES first - so
	between the open and the write completing there is a window where the file
	exists but is empty (or short). Every cache reader in this codebase gates
	on `cache_file.is_file()` and then parses the contents, so a reader landing
	in that window doesn't see a cache MISS, it sees a cache HIT with garbage:

	  - linker_c.has_symbol:    ''.strip() == '1' -> False, i.e. "that symbol
	                            isn't available" for a symbol that is
	  - lowering._eval_cexpr:   int('') -> ValueError, a hard crash
	  - lowering's UnicodeData: a truncated table, silently

	Confirmed by a real, if rare, test failure: two mutually-exclusive
	@compiler.target(has_library=X) / (has_library=not X) definitions BOTH
	survived discovery (producing an Overload where exactly one Function was
	expected) because the two probes - separate has_symbol calls, no in-process
	memo between them - read INCONSISTENT values, one before and one during
	another shard's rewrite of the same cache file. tests.py runs 16 shards as
	concurrent subprocesses sharing one cache dir, which is why the suite is
	where this showed up; any two concurrent mpy invocations can hit it.

	The temp file is created in the SAME directory as the target so os.replace
	is a same-filesystem rename, which is atomic on both POSIX and Windows.
	Readers should ALSO treat empty/unparseable content as a miss - this fixes
	new writes, but cannot repair a corrupt file some earlier run left behind.

	PUBLISHING IS BEST-EFFORT, deliberately. On Windows os.replace fails with
	PermissionError (WinError 5) when the destination is currently OPEN - which
	a concurrent reader doing cache_file.read_text() briefly makes it. The first
	version of this raised, which turned the original rare silent-wrong-answer
	into a rare hard crash that aborted the compile - strictly worse, and caught
	by the same test that motivated the fix in the first place.

	Losing that race is harmless: this cache is IDEMPOTENT, every writer for a
	given key computes the same value from the same (lib, symbol, compiler) or
	(expr, header) inputs, so whoever wins publishes the identical bytes. The
	caller already has its own correct value in hand and returns it either way;
	all that's lost is the chance to save the NEXT process a re-probe. A few
	tight retries first, since a reader's handle is only open for microseconds
	and retrying usually wins immediately - but never at the cost of failing. '''
	cache_file.parent.mkdir( parents = True, exist_ok = True )
	tmp = cache_file.with_name( f'{cache_file.name}.{os.getpid()}.tmp' )
	try:
		if isinstance( data, bytes ):
			tmp.write_bytes( data )
		else:
			tmp.write_text( data, encoding = 'utf-8' )
		for attempt in range( 3 ):
			try:
				os.replace( tmp, cache_file )
				return
			except OSError:
				if attempt == 2:
					# give up publishing - see "best-effort" above. NOT an error
					# to report: a failed publish costs a future re-probe, never
					# correctness, and the cache lives in %TEMP% where a full
					# disk / locked file is the user's environment, not a bug in
					# the compile they asked for.
					break
	finally:
		# never leave a stray .tmp behind - on the give-up path above, and on
		# any exception from the writes themselves. Nothing reaps %TEMP%/metalpy
		# the way it eventually reaps the real cache files.
		tmp.unlink( missing_ok = True )


class CcTool:
	'''
	A detected C compiler.

	name: 'clang', 'gcc', or 'cl'
	path: full path (or just 'cl' for MSVC since it's in PATH after vcvars)
	'''

	def __init__( self, name: str, path: str ) -> None:
		self.name = name
		self.path = path

	def compile( self, src: Path, obj: Path, verbose: bool = False, no_crt: bool = False, debug: bool = True, asan: bool = False, cflags: str = '' ) -> subprocess.CompletedProcess[bytes]:
		''' compile a single .c file to a .o object file '''
		# asan forces debug INFO on regardless of debug/release, so a crash
		# report is symbolized - optimization level still follows debug/release
		# normally (asan works fine instrumented+optimized, a common combo for
		# fuzzing performance)
		want_debug_info = debug or asan
		if self.name == 'cl':
			cmd = [ self.path, '/nologo', '/std:c11',
				'/experimental:c11atomics',
				'/W4', '-c', str( src ), f'/Fo:{obj}' ]
			if no_crt:
				cmd += [ '/GS-' ]
			if want_debug_info:
				# /Fd points the PDB at obj's own directory instead of cl's
				# default (a shared vc140.pdb in the CURRENT directory) - since
				# every caller already compiles into its own unique temp dir
				# (mpy.py, test_support.py, etc.), this makes concurrent cl.exe
				# processes (parallel test shards) never share a PDB path in the
				# first place, rather than relying on /FS to merely serialize
				# writes through mspdbsrv.exe (which alone still produced
				# C1041 "cannot open program database" under this project's
				# full parallel test suite - /FS is kept too, since it's still
				# correct/harmless for the rarer case of two compiles that DO
				# legitimately share one obj directory)
				cmd += [ '/Zi', '/FS', f'/Fd:{obj.with_name( "vc140.pdb" )}' ]
			if debug:
				cmd += [ '/Od' ]
				# /RTC1 (stack-frame + uninitialized-variable checks) needs the
				# _RTC_* support routines that live in the CRT - the no_crt
				# freestanding path already passes /NODEFAULTLIB at link time,
				# which would leave those symbols unresolved. Also mutually
				# exclusive with /fsanitize=address (MSVC hard-errors if both
				# are given), so asan wins when both would otherwise apply.
				if not no_crt and not asan:
					cmd += [ '/RTC1' ]
			else:
				cmd += [ '/O2', '/DNDEBUG' ]
			if asan:
				cmd += [ '/fsanitize=address' ]
		else:
			cmd = [ self.path, '-std=c11', '-Wall', '-Wextra', '-c', str( src ), '-o', str( obj ) ]
			if want_debug_info:
				cmd += [ '-g' ]
			if debug:
				# -fsanitize-trap=undefined compiles each UBSan check straight to
				# a trap instruction instead of calling a runtime-library
				# handler, so unlike ASan it has no CRT/allocator dependency and
				# is safe even in the no_crt freestanding path.
				# -fno-sanitize=function: emit_interface_vtable_instance()
				# stores each vtable slot as a function pointer cast from the
				# concrete override's own signature to the interface's base
				# pointer type - the standard C vtable idiom, technically UB by
				# the letter of the standard but load-bearing for every
				# @interface/@virtual call in the language; there's no
				# UB-clean alternative short of a trampoline per override.
				# (the analogous ShlCheck/ShlSaturate false positive on
				# shift-base was fixed at the source instead - see _shl_expr
				# in emitter_c.py - so no exclusion is needed for that one).
				# -fno-sanitize=function is clang-only: GCC never implemented
				# -fsanitize=function (no function-pointer-type check in its
				# own -fsanitize=undefined group at all), so it has nothing to
				# exclude and rejects the flag outright - gcc's vtable dispatch
				# was never going to trip this check in the first place.
				cmd += [ '-O0', '-fsanitize=undefined', '-fsanitize-trap=undefined' ]
				if self.name == 'clang':
					cmd += [ '-fno-sanitize=function' ]
			else:
				cmd += [ '-O2', '-DNDEBUG' ]
			if asan:
				# clang/gcc accumulate multiple -fsanitize= flags (this adds to,
				# not replaces, the debug-mode -fsanitize=undefined above) -
				# -fsanitize-trap=undefined still scopes its trap behavior to
				# only the undefined group, so asan keeps its normal runtime-
				# reporting behavior (it has no trap-mode equivalent)
				cmd += [ '-fsanitize=address', '-fno-omit-frame-pointer' ]
		if cflags:
			cmd += cflags.split()
		if verbose:
			print( ' '.join( cmd ), file = sys.stderr )
		return subprocess.run( cmd,
			stdout = subprocess.PIPE,
			stderr = subprocess.STDOUT,
			text = True,
		)

	def link( self, exe: Path, objs: list[Path], ldflags: str = '', verbose: bool = False, no_crt: bool = False, debug: bool = True, asan: bool = False, strip: bool = False ) -> subprocess.CompletedProcess[bytes]:
		''' link one or more .o files into an executable '''
		obj_args = [ str( o ) for o in objs ]
		extra = ldflags.split() if ldflags else []
		if self.name == 'cl':
			cmd = [ 'link', '/nologo', f'/OUT:{exe}' ] + obj_args + extra
			if no_crt:
				cmd += [ '/NODEFAULTLIB', '/ENTRY:mainCRTStartup' ]
			if debug or asan:
				cmd += [ '/DEBUG' ]
			if strip:
				# PE has no ELF-style embedded symbol table to strip in the
				# first place (a binary built without /DEBUG already carries
				# none) - /OPT:REF /OPT:ICF (dead-code elimination + identical-
				# COMDAT folding) is the closest MSVC analog to what people
				# actually mean by a "stripped" release build
				cmd += [ '/OPT:REF', '/OPT:ICF' ]
			# no /fsanitize=address here: that's a cl.exe compiler-frontend
			# flag, not understood by link.exe directly - cl.exe embeds the
			# necessary /DEFAULTLIB directive for the ASan runtime straight
			# into the .obj itself, so the separate link step needs nothing
			# extra (verified empirically - see plan's verification section)
		else:
			# a program using an f32/f64<->i128/u128 cast needs GCC/Clang's own
			# runtime helpers (__fixdfti/__fixunsdfti/__floattidf/... - see
			# _find_wide_int_runtime_lib) - added as a full PATH positional
			# arg, not a bare -l<name> flag: its own directory can contain
			# spaces (a stock Windows LLVM install does), which `extra`'s
			# plain-whitespace ldflags.split() above can't represent, and
			# clang/gcc both accept a literal .a/.lib path as an ordinary
			# linker input. Harmless to add even when unused - a static
			# archive only pulls in symbols something else in the link
			# actually references
			wide_int_lib = _find_wide_int_runtime_lib( self )
			cmd = [ self.path ] + extra + obj_args + ( [ wide_int_lib ] if wide_int_lib else [] ) + [ '-o', str( exe ) ]
			if asan:
				# clang/gcc's own driver acts as the linker frontend even for
				# an objects-only link, and only links the ASan runtime when
				# -fsanitize=address is present at THIS invocation too, not
				# just at compile time
				cmd += [ '-fsanitize=address' ]
			if strip:
				cmd += [ '-s' ]
		if verbose:
			print( ' '.join( cmd ), file = sys.stderr )
		return subprocess.run( cmd,
			stdout = subprocess.PIPE,
			stderr = subprocess.STDOUT,
			text = True,
		)


def has_i128( cc: CcTool|None ) -> bool:
	'''
	True 128-bit i128/u128 range/semantics, vs MSVC's documented 64-bit
	fallback (see emitter_c.py's __metalpy_wideint/__metalpy_wideuint
	typedefs, `#if defined(_MSC_VER) && !defined(__clang__)`) - this is the
	direct Python-side mirror of that same C-preprocessor condition. `cc is
	None` (no compiler found at all) defaults to True: permissive, and moot
	anyway since an actual build with no compiler dies with a clear error
	regardless of what this said.
	'''
	return cc is None or cc.name != 'cl'


def has_symbol( cc: CcTool, lib: str, symbol: str ) -> bool:
	'''
	Probes whether `symbol` resolves when linked against `lib` - compile
	and link only, no run (unlike lowering.py's compiler.cexpr(), which
	has to execute the probe binary to read its result - this only checks
	that the linker can find the symbol, so it works even cross-compiling).

	Declares `symbol` with a deliberately vague signature (`char symbol();`
	- an old-style K&R declaration whose actual return type doesn't matter,
	since main() never really calls it meaningfully) and takes its return
	value, mirroring autoconf's own AC_CHECK_LIB macro - the standard,
	portable way to ask "does this symbol link" without needing the real
	header/prototype.

	Cached to disk under %TEMP%/metalpy/has_symbol/, keyed by (lib, symbol,
	compiler name) - same spirit and location as compiler.cexpr()'s own
	cache (see lowering.py's _eval_cexpr) - so a real compile+link
	subprocess pair is only ever paid once per (lib, symbol) pair, not on
	every mpy invocation that happens to reference it.
	'''
	import hashlib
	import tempfile

	key = hashlib.sha256( f'{lib}\0{symbol}\0{cc.name}'.encode() ).hexdigest()[:16]
	cache_dir = Path( tempfile.gettempdir() ) / 'metalpy' / 'has_symbol'
	cache_dir.mkdir( parents = True, exist_ok = True )
	cache_file = cache_dir / key
	if cache_file.is_file():
		# an empty/unrecognized body is a TORN or half-written entry, not a
		# real answer - fall through and re-probe rather than reporting "not
		# available" for something that is (see atomic_write_cache). Cheap:
		# the re-probe overwrites it with a good value.
		#
		# OSError is the same story from the other side: on Windows, opening
		# this file fails with PermissionError while another process's
		# os.replace of it is in flight. READING is best-effort for exactly the
		# reason PUBLISHING is - a lost read costs one re-probe, never
		# correctness - so cache contention must never surface as an error on
		# either side.
		try:
			cached = cache_file.read_text().strip()
		except OSError:
			cached = ''
		if cached in ( '0', '1' ):
			return cached == '1'

	c_src = f'char {symbol}();\nint main(void) {{ return {symbol}(); }}\n'
	with tempfile.TemporaryDirectory() as tmp:
		src_path = Path( tmp ) / 'probe.c'
		obj_path = Path( tmp ) / 'probe.o'
		exe_path = Path( tmp ) / 'probe'
		src_path.write_text( c_src, encoding = 'utf-8' )
		compile_result = cc.compile( src_path, obj_path )
		if compile_result.returncode != 0:
			available = False
		else:
			ldflag = f'{lib}.lib' if cc.name == 'cl' else f'-l{lib}'
			link_result = cc.link( exe_path, [ obj_path ], ldflags = ldflag )
			available = link_result.returncode == 0

	atomic_write_cache( cache_file, '1' if available else '0' )
	return available


def _find_wide_int_runtime_lib( cc: CcTool ) -> str|None:
	'''
	Locates the static runtime library providing GCC/Clang's own float<->
	128-bit-int conversion helpers (__fixdfti, __fixunsdfti, __floattidf,
	...) - no hardware instruction does an f32/f64<->i128/u128 conversion
	directly, so GCC/Clang emit a CALL to one of these instead (confirmed:
	linking without this library fails with "unresolved external symbol
	__fixdfti/__fixunsdfti"). MSVC's cl.exe needs none of this - i128/u128
	fall back to plain 64-bit under cl.exe instead (see emitter_c.py's
	__metalpy_wideint typedef and its own comment on why).

	Returns None (best-effort, never fails the build) when not found - only
	a program that actually performs one of these conversions needs the
	symbols at all; every other program links exactly as it did before this
	lookup existed.

	Cached to disk under %TEMP%/metalpy/wide_int_runtime_lib/, same spirit
	and location as has_symbol's own cache, since each lookup pays a real
	subprocess.
	'''
	import hashlib
	import tempfile

	if cc.name not in ( 'clang', 'gcc' ):
		return None

	key = hashlib.sha256( f'{cc.name}\0{cc.path}'.encode() ).hexdigest()[:16]
	cache_dir = Path( tempfile.gettempdir() ) / 'metalpy' / 'wide_int_runtime_lib'
	cache_dir.mkdir( parents = True, exist_ok = True )
	cache_file = cache_dir / key
	if cache_file.is_file():
		return cache_file.read_text( encoding = 'utf-8' ).strip() or None

	found: str|None = None
	if cc.name == 'clang':
		# clang -print-runtime-dir gives compiler-rt's own lib directory
		# (e.g. .../lib/clang/<ver>/lib/windows) - the builtins archive in
		# there is named clang_rt.builtins-<arch>.lib (Windows) or
		# libclang_rt.builtins-<arch>.a (Linux/macOS); glob for it rather
		# than hardcoding <arch>, excluding the "_dynamic"-suffixed DLL-
		# import variant (this needs the STATIC archive, linked directly)
		result = subprocess.run( [ cc.path, '-print-runtime-dir' ], capture_output = True, text = True )
		rt_dir = Path( result.stdout.strip() ) if result.stdout.strip() else None
		if result.returncode == 0 and rt_dir is not None and rt_dir.is_dir():
			candidates = sorted(
				p for p in rt_dir.glob( '*builtins*' )
				if p.suffix in ( '.lib', '.a' ) and '_dynamic' not in p.stem
			)
			if candidates:
				found = str( candidates[0] )
	elif cc.name == 'gcc':
		# the standard, portable way to ask gcc where its own libgcc.a is -
		# gcc's own driver normally auto-links this already, but adding it
		# explicitly here costs nothing and keeps this function's contract
		# (a real path when found) uniform across both compilers
		result = subprocess.run( [ cc.path, '-print-libgcc-file-name' ], capture_output = True, text = True )
		path = Path( result.stdout.strip() ) if result.stdout.strip() else None
		if result.returncode == 0 and path is not None and path.is_file():
			found = str( path )

	cache_file.write_text( found or '', encoding = 'utf-8' )
	return found


def detect_cc() -> CcTool|None:
	'''
	detect a working C compiler. respects METALPY_CC (one of clang/
	gcc/msvc, or empty for auto-detect). returns None if no compiler
	is found.
	'''
	METALPY_CC = os.environ.get( 'METALPY_CC', '' ).strip().lower()

	# --- clang ---
	if METALPY_CC in ( 'clang', '' ):
		path = shutil.which( 'clang' )
		if path:
			return CcTool( 'clang', path )

	# --- gcc ---
	if METALPY_CC in ( 'gcc', '' ):
		path = shutil.which( 'gcc' )
		if path:
			return CcTool( 'gcc', path )

	# --- msvc ---
	if METALPY_CC in ( 'msvc', '' ):
		if os.environ.get( 'VCINSTALLDIR', None ):
			return CcTool( 'cl', 'cl' )
		# vcvars64 has not been run - try to auto-detect it via vswhere
		print( 'WARNING - vcvars64 not run - trying to auto-detect it (this is slow)', file = sys.stderr )
		vswhere = Path( r'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe' )
		if vswhere.is_file():
			args = [ str( vswhere ), '-latest', '-products', '*', '-all', '-find', r'VC\Auxiliary\Build\vcvars64.bat' ]
			result = subprocess.run( args, capture_output = True, text = True )
			vcvars64 = result.stdout.strip()
			if vcvars64:
				print( f'vcvars64 path={vcvars64}', file = sys.stderr )
				output = subprocess.check_output( f'"{vcvars64}" && set', shell = True, text = True )
				for line in output.splitlines():
					if '=' in line:
						k, _, v = line.partition( '=' )
						os.environ[k] = v
				return CcTool( 'cl', 'cl' )

	return None
