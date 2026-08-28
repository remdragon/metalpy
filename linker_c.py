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


def ensure_cache_dir( cache_dir: Path ) -> bool:
	''' best-effort mkdir for one of this codebase's %TEMP%/metalpy/<category>
	disk-cache directories - shared by every cache call site (atomic_write_
	cache below, and each cache_dir.mkdir(...) that gates a read attempt
	before ever reaching it: lowering._eval_cexpr, has_symbol,
	ntdll_import_lib, wide_int_runtime_lib). A cache directory being
	unavailable - most commonly a DIFFERENT user's earlier process having
	left one behind at a restrictive mode (confirmed: root-owned, 0o755,
	blocking every non-root user from creating anything inside it) - must
	never fail the compile the caller actually asked for; it only means
	this call, and every other process sharing the directory, pays the real
	cost the cache exists to avoid. Warns via stderr (NOT silent - unlike
	atomic_write_cache's own publish-race retries, this is a standing
	environment problem worth a human noticing, not a routine microseconds-
	wide contention window) and returns False so the caller skips the
	read/write attempt entirely instead of tripping over a directory that
	still doesn't actually exist.

	On POSIX, ALSO chmods cache_dir and its immediate parent (the shared
	.../metalpy directory itself, but never higher - tempfile.gettempdir()
	is a real system directory, e.g. /tmp, this code must never touch) to
	0o777 whenever it can, regardless of whether THIS call just created
	them: Path.mkdir(mode=...) is filtered through the process umask, so
	passing a permissive mode there is not reliable, and re-asserting 0o777
	on a directory this process (or a past run of this same fixed code)
	already owns is a harmless, self-healing no-op. A directory owned by a
	different user simply raises EPERM here, silently ignored - not ours to
	fix, that's exactly the "unavailable, fall through" case above. '''
	try:
		cache_dir.mkdir( parents = True, exist_ok = True )
	except OSError as e:
		print( f'WARNING - metalpy: cache directory {cache_dir} is unavailable ({e}) - continuing without caching', file = sys.stderr )
		return False
	if os.name == 'posix':
		for d in ( cache_dir, cache_dir.parent ):
			try:
				os.chmod( d, 0o777 )
			except OSError:
				pass
	return True


def atomic_write_cache( cache_file: Path, data: 'bytes|str' ) -> bool:
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

	PUBLISHING IS BEST-EFFORT, deliberately - returns whether it actually
	succeeded (True) or gave up (False), but NEVER raises. On Windows
	os.replace fails with PermissionError (WinError 5) when the destination
	is currently OPEN - which a concurrent reader doing cache_file.
	read_text() briefly makes it. The first version of this raised, which
	turned the original rare silent-wrong-answer into a rare hard crash
	that aborted the compile - strictly worse, and caught by the same test
	that motivated the fix in the first place. The SAME posture now also
	covers the write itself (tmp.write_bytes/write_text) and the directory
	creation (ensure_cache_dir) - a different user's earlier process having
	left the cache directory at a mode this process can't write into hits
	PermissionError right there, before os.replace is even reached, and
	needs the identical "don't crash the caller's compile" treatment.

	Losing that race is harmless: this cache is IDEMPOTENT, every writer for a
	given key computes the same value from the same (lib, symbol, compiler) or
	(expr, header) inputs, so whoever wins publishes the identical bytes. The
	caller already has its own correct value in hand and returns it either way;
	all that's lost is the chance to save the NEXT process a re-probe. A few
	tight retries first, since a reader's handle is only open for microseconds
	and retrying usually wins immediately.

	A give-up IS reported (stderr WARNING, not silent - see ensure_cache_dir's
	own identical reasoning): losing one publish is harmless noise, but a
	caller like ntdll_import_lib whose own contract needs the file to actually
	land on disk checks this return value and needs to know why it came back
	False, and a standing "this directory is unusable" environment problem is
	worth a human noticing even where the immediate caller doesn't care. '''
	if not ensure_cache_dir( cache_file.parent ):
		return False
	tmp = cache_file.with_name( f'{cache_file.name}.{os.getpid()}.tmp' )
	try:
		try:
			if isinstance( data, bytes ):
				tmp.write_bytes( data )
			else:
				tmp.write_text( data, encoding = 'utf-8' )
		except OSError as e:
			print( f'WARNING - metalpy: cannot write cache file {cache_file} ({e}) - continuing without caching', file = sys.stderr )
			return False
		if os.name == 'posix':
			try:
				os.chmod( tmp, 0o666 )
			except OSError:
				pass
		last_e = None
		for attempt in range( 3 ):
			try:
				os.replace( tmp, cache_file )
				return True
			except OSError as e:
				last_e = e
		print( f'WARNING - metalpy: cannot publish cache file {cache_file} ({last_e}) - continuing without caching', file = sys.stderr )
		return False
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

	def compile( self, src: Path, obj: Path, verbose: bool = False,
		no_crt: bool = False,
		debug: bool = True,
		asan: bool = False,
		cflags: str = '',
		warnings: bool = True,
	) -> subprocess.CompletedProcess[str]:
		''' compile a single .c file to a .o object file '''
		# asan forces debug INFO on regardless of debug/release, so a crash
		# report is symbolized - optimization level still follows debug/release
		# normally (asan works fine instrumented+optimized, a common combo for
		# fuzzing performance)
		want_debug_info = debug or asan
		if self.name == 'cl':
			# /experimental:c11atomics - MSVC's own <stdatomic.h> (unconditionally
			# included in every generated .c, for RC refcount atomics) hard-errors
			# with C1189 "C atomic support is not enabled" under plain /std:c11
			# without this flag on newer MSVC toolsets (confirmed on VS 18/
			# BuildTools 14.51.36231) - full C11 atomics support is still gated
			# behind this experimental switch there.
			cmd = [ self.path, '/nologo', '/std:c11', '/experimental:c11atomics' ]
			if warnings:
				cmd += [ '/W4', '/wd4701' ]
			# NOT adding /Gy (function-level linking, MSVC's analog of
			# clang/gcc's -ffunction-sections below) - measured ZERO size
			# benefit from it on this codebase's own generated C: a real
			# release+strip build (link.exe already gets /OPT:REF /OPT:ICF
			# under strip below) came out byte-identical, same exact symbol
			# set, with or without /Gy. Unlike LLVM (see -ffunction-sections'
			# own comment for the real 20% win it unlocks there), MSVC's own
			# backend apparently doesn't emit byte-for-byte identical machine
			# code for structurally-identical generic instantiations even
			# once /Gy gives /OPT:ICF individual COMDATs to compare - so
			# there's nothing here for ICF to actually fold. /Gy also
			# introduced a reproducible CreateProcess "Access is denied"
			# hitting the just-linked exe under this project's own build-
			# then-immediately-run test harness (root-caused to Windows/EDR
			# holding the file slightly longer for /Gy's more fragmented
			# per-function-COMDAT layout, not real corruption - the same exe
			# runs fine seconds later by hand) - moot now that there's no
			# size upside to weigh against it either.
			cmd += [ '-c', str( src ), f'/Fo:{obj}' ]
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
			cmd = [ self.path, '-std=c11' ]
			if warnings:
				cmd += [ '-Wall', '-Wextra' ]
			# -ffunction-sections/-fdata-sections: give every function/global
			# its own section, so the linker's identical-code/data folding
			# (lld-link's /OPT:ICF, or a GNU ld with --icf, see CcTool.link()'s
			# strip branch) has individual chunks to compare and merge at all -
			# without this, a whole .o's functions share one section and
			# nothing is foldable regardless of how many byte-identical
			# monomorphized generic instantiations exist. Free money for
			# release/strip builds.
			cmd += [ '-ffunction-sections', '-fdata-sections' ]
			cmd += [ '-c', str( src ), '-o', str( obj ) ]
			if warnings:
				cmd += [ '-Wno-uninitialized' ]
				cmd += [ '-Wno-sometimes-uninitialized' ] if self.name == 'clang' else [ '-Wno-maybe-uninitialized' ]
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

	def link( self, exe: Path, objs: list[Path], ldflags: str = '', verbose: bool = False, no_crt: bool = False, debug: bool = True, asan: bool = False, strip: bool = False, map_file: Path|None = None ) -> subprocess.CompletedProcess[str]:
		''' link one or more .o files into an executable '''
		obj_args = [ str( o ) for o in objs ]
		extra = ldflags.split() if ldflags else []
		if self.name == 'cl':
			if no_crt:
				# a freestanding MSVC build has no CRT to supply __chkstk (the
				# stack-probe routine cl.exe's own backend silently calls from
				# any function prologue whose frame exceeds one page) - see
				# _build_chkstk_obj's own docstring
				chkstk_obj = _build_chkstk_obj( verbose = verbose )
				if chkstk_obj is not None:
					obj_args = obj_args + [ str( chkstk_obj ) ]
			cmd = [ 'link', '/nologo', f'/OUT:{exe}' ] + obj_args + extra
			if no_crt:
				# /SUBSYSTEM:CONSOLE is required here now too - link.exe can
				# normally infer the subsystem from a `main`-shaped symbol,
				# but emitter_c.py's own __metalpy_main rename means no real
				# `main` symbol exists at all under no_crt any more, so an
				# explicit /ENTRY without this now hits LNK1221 "a subsystem
				# can't be inferred and must be defined" (confirmed via a
				# real link failure) - same fix as the clang/lld-link branch
				# below needed for the identical reason.
				cmd += [ '/NODEFAULTLIB', '/ENTRY:mainCRTStartup', '/SUBSYSTEM:CONSOLE' ]
			if debug or asan:
				cmd += [ '/DEBUG' ]
			if strip:
				# PE has no ELF-style embedded symbol table to strip in the
				# first place (a binary built without /DEBUG already carries
				# none) - /OPT:REF /OPT:ICF (dead-code elimination + identical-
				# COMDAT folding) is the closest MSVC analog to what people
				# actually mean by a "stripped" release build
				cmd += [ '/OPT:REF', '/OPT:ICF' ]
			if map_file is not None:
				cmd += [ f'/MAP:{map_file}' ]
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
			#
			# extra (ldflags, e.g. -lssl) MUST come after obj_args, not
			# before - GNU ld resolves a -l<name> against whatever undefined
			# references are ALREADY pending when it reaches that flag on the
			# command line; a library placed before the objects that need it
			# is a no-op (confirmed by a real repro: `gcc -lssl generated.o`
			# silently fails to resolve SSL_new, `gcc generated.o -lssl`
			# resolves it fine). This was invisible until lib/ssl.py's Linux
			# backend (has_library=('ssl', 'SSL_new')) - every prior has_
			# library/extern_libs use on Linux was libc ('c'), which every
			# compiler driver links implicitly regardless of -l ordering, so
			# the bug never affected a real -l<name> flag before.
			wide_int_lib = _find_wide_int_runtime_lib( self )
			cmd = [ self.path ] + obj_args + ( [ wide_int_lib ] if wide_int_lib else [] ) + extra + [ '-o', str( exe ) ]
			if no_crt and os.name != 'posix':
				# clang on native Windows drives lld-link (MSVC-compatible) -
				# unlike the 'cl' branch above, it never got an explicit
				# /ENTRY flag; it inferred the entry point from a `main`-
				# shaped symbol instead, which a freestanding build no longer
				# defines (emitter_c.py's own mainCRTStartup rename -
				# __metalpy_main is the only orchestrator now, no real
				# `main` symbol exists at all under no_crt). Without this,
				# lld-link fails with LNK1561 "entry point must be defined"
				# (confirmed via a real link failure) - and once the entry
				# is explicit, it also can no longer infer the subsystem
				# from `main`'s presence either (LNK1221, the same fix the
				# 'cl' branch above now needs too), so both need spelling
				# out. gcc's own ELF/WSL target never reaches this
				# branch in practice (mainCRTStartup is #ifdef _WIN32-only,
				# so os.name == 'posix' there) - no GNU-ld equivalent needed.
				cmd += [ '-Wl,-entry:mainCRTStartup', '-Wl,-subsystem:console' ]
			if asan:
				# clang/gcc's own driver acts as the linker frontend even for
				# an objects-only link, and only links the ASan runtime when
				# -fsanitize=address is present at THIS invocation too, not
				# just at compile time
				cmd += [ '-fsanitize=address' ]
			if strip:
				if os.name == 'posix':
					# real GNU ld (gcc, or clang under WSL) - -Wl,-s forwards
					# -s straight to the linker, bypassing the driver's own
					# flag interpretation (plain -s is a compile-stage flag to
					# clang's own driver, silently dropped with "argument
					# unused during compilation" on an objs-only link, which
					# never runs a cc1 compile step to consume it)
					cmd += [ '-Wl,-s' ]
				else:
					# clang on native Windows drives lld-link (MSVC-compatible,
					# see the no_crt branch above) - -s means nothing there
					# either way (LNK4044 "unrecognized option", confirmed via
					# a real repro), same as the 'cl' branch's own comment: PE
					# has no ELF-style symbol table to strip in the first
					# place, /OPT:REF /OPT:ICF (dead-code elim + identical-
					# COMDAT folding) is the closest real analog
					cmd += [ '-Wl,/OPT:REF', '-Wl,/OPT:ICF' ]
			if map_file is not None:
				if os.name == 'posix':
					# real GNU ld (gcc, or clang under WSL) - -Map=<file> is a
					# standard ld option, forwarded straight through via -Wl,
					cmd += [ f'-Wl,-Map={map_file}' ]
				else:
					# clang on native Windows drives lld-link (MSVC-compatible,
					# see the no_crt/strip branches above) - lld-link accepts
					# MSVC link.exe's own /MAP:<file> spelling, not GNU ld's
					cmd += [ f'-Wl,/MAP:{map_file}' ]
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
	cache_file = cache_dir / key
	if ensure_cache_dir( cache_dir ) and cache_file.is_file():
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


def _default_link_provides( cc: CcTool, symbol: str ) -> bool:
	'''
	True if `symbol` resolves through an ORDINARY, CRT-linked build's own
	default linking alone (the compiler's implicit default libraries - e.g.
	ucrt.lib under MSVC/clang) - i.e. with no extra -l/.lib flag at all.
	Distinguishes "ntdll genuinely is the only source of this symbol" from
	"the default C runtime already provides an identically-named,
	functionally-equivalent symbol" - see build_ntdll_import_lib's own
	docstring for why this matters (a synthetic ntdll import entry for a
	name the default CRT libraries ALSO define, e.g. strnlen via ucrt.lib,
	produces a real LNK2005 duplicate-symbol error the moment a build links
	both).

	No no_crt parameter, deliberately: this is only ever meaningful - and
	only ever called (see build_ntdll_import_lib) - for a build that IS
	linking its default C runtime (no_crt=False). A genuinely freestanding
	build never needs it: no_crt=True means /NODEFAULTLIB under MSVC (ucrt.
	lib is explicitly excluded, full stop), and under clang/gcc it means
	emitter_c.py emitted the freestanding program's OWN mainCRTStartup -
	which, confirmed empirically, is what actually keeps the real ucrt/CRT
	default libraries out of a clang/gcc link in the first place (nothing
	else in this codebase ever pulls in the CRT's own startup object, which
	is what would otherwise drag ucrt.lib onto the default library search
	list at all - clang has no unconditional "-defaultlib:ucrt"-style flag
	of its own). A bare `int main(void)` probe - the only shape this
	function could reasonably synthesize - does NOT define its own
	mainCRTStartup, so probing it under a claimed no_crt=True would silently
	answer for the WRONG program shape and could wrongly report a symbol as
	default-linked when the real freestanding build never links it at all.

	Mirrors has_symbol()'s probe shape (compile+link only, no run - see its
	own docstring), but without an extra lib argument.

	Cached to disk under %TEMP%/metalpy/default_link_symbol/, keyed by
	(compiler name, symbol) - same spirit as has_symbol()'s own cache.
	'''
	import hashlib
	import tempfile

	key = hashlib.sha256( f'{cc.name}\0{symbol}'.encode() ).hexdigest()[:16]
	cache_dir = Path( tempfile.gettempdir() ) / 'metalpy' / 'default_link_symbol'
	cache_file = cache_dir / key
	if ensure_cache_dir( cache_dir ) and cache_file.is_file():
		# see has_symbol's identical reasoning: a torn/unreadable cache entry
		# is a miss to re-probe, never a hard error
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
			link_result = cc.link( exe_path, [ obj_path ] )
			available = link_result.returncode == 0

	atomic_write_cache( cache_file, '1' if available else '0' )
	return available


_NTDLL_PATH = Path( os.environ.get( 'SystemRoot', r'C:\Windows' ) ) / 'System32' / 'ntdll.dll'


def _ntdll_toolchain( cc: CcTool ) -> tuple[str,str]:
	'''
	(export-lister, lib-builder) for reading/rebuilding an MS-COFF import
	library - see build_ntdll_import_lib's docstring for why this exists at
	all. Deliberately tied to `cc`, not just "whatever's on PATH": cl.exe
	only runs after vcvars has put its whole VC\\Tools\\...\\bin\\Hostx64\\x64
	directory on PATH, so MSVC's own dumpbin.exe/lib.exe are guaranteed to
	be right there too - but clang needs no such thing (it locates link.exe
	itself via its own internal Visual Studio probing, not PATH), so a build
	using --cc clang without vcvars having ever run would newly require it
	if this reached for dumpbin/lib.exe the same way. LLVM ships its own
	drop-in equivalents (llvm-readobj/llvm-lib, MS-COFF compatible -
	confirmed empirically: a .lib llvm-lib built from a hand-written .def
	links fine against link.exe-produced objects) colocated with clang.exe
	itself, so use those instead when cc is clang.
	'''
	if cc.name == 'cl':
		dumpbin = shutil.which( 'dumpbin' )
		lib_exe = shutil.which( 'lib' )
		if not dumpbin or not lib_exe:
			raise RuntimeError( "can't build a custom ntdll import library: dumpbin.exe/lib.exe not found on PATH "
				"(they normally sit right alongside cl.exe once vcvars has run)" )
		return dumpbin, lib_exe
	if cc.name == 'clang':
		bindir = Path( cc.path ).parent
		readobj = bindir / 'llvm-readobj.exe'
		llvmlib = bindir / 'llvm-lib.exe'
		if not readobj.is_file() or not llvmlib.is_file():
			raise RuntimeError( f"can't build a custom ntdll import library: llvm-readobj.exe/llvm-lib.exe not found alongside {cc.path}" )
		return str( readobj ), str( llvmlib )
	raise RuntimeError( f'generating a custom ntdll import library is not supported for compiler {cc.name!r} '
		'(only cl/clang ever target Windows in this codebase - gcc here is WSL-only, for posix targets)' )


def _parse_dumpbin_exports( text: str ) -> set[str]:
	''' names of every real, named, non-forwarded export in a `dumpbin
	/exports` listing. A forwarder line ("name = OtherDll.OtherName") and
	an ordinal-only "[NONAME]" line both fail this line shape on purpose -
	neither is a symbol @extern('ntdll', ...) could ever bind to directly. '''
	import re
	return set( re.findall( r'^\s*\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+(\S+)\s*$', text, re.MULTILINE ) )


def _parse_llvm_readobj_exports( text: str ) -> set[str]:
	''' same as _parse_dumpbin_exports, for `llvm-readobj --coff-exports`'s
	block-structured "Export { ... }" output. '''
	names: set[str] = set()
	name: str|None = None
	forwarded = False
	for raw in text.splitlines():
		line = raw.strip()
		if line == 'Export {':
			name, forwarded = None, False
		elif line.startswith( 'Name:' ):
			name = line[ len( 'Name:' ): ].strip()
		elif line.startswith( 'ForwardedTo:' ):
			forwarded = True
		elif line == '}':
			if name and not forwarded:
				names.add( name )
			name, forwarded = None, False
	return names


def _real_ntdll_exports( cc: CcTool ) -> set[str]:
	''' the real system ntdll.dll's own export table (NOT the Windows SDK's
	curated ntdll.lib stub - see build_ntdll_import_lib's docstring). '''
	lister, _ = _ntdll_toolchain( cc )
	args = [ lister, '/exports', str( _NTDLL_PATH ) ] if cc.name == 'cl' else [ lister, '--coff-exports', str( _NTDLL_PATH ) ]
	result = subprocess.run( args, capture_output = True, text = True )
	if result.returncode != 0:
		raise RuntimeError( f"failed to read {_NTDLL_PATH}'s export table:\n{result.stdout}{result.stderr}" )
	return _parse_dumpbin_exports( result.stdout ) if cc.name == 'cl' else _parse_llvm_readobj_exports( result.stdout )


def build_ntdll_import_lib( cc: CcTool, symbols: set[str], verbose: bool = False, no_crt: bool = False ) -> Path|None:
	'''
	Builds (and disk-caches) a small MS-COFF import library exposing exactly
	`symbols` from the REAL system ntdll.dll, bypassing the Windows SDK's
	own ntdll.lib import library entirely.

	When `no_crt` is False (this build links its default C runtime for
	real), symbols the default CRT linking already provides (see
	_default_link_provides) are dropped from `symbols` FIRST, before
	anything else below - ntdll.dll and a CRT-linked build's own default
	libraries (e.g. MSVC/clang's ucrt.lib) both genuinely export a real
	`strnlen`, two unrelated functions that happen to share a name and, for
	this exact `size_t(const char*, size_t)` shape, are functionally
	interchangeable to a caller. Synthesizing an ntdll import entry for one
	of those names on top of a build that ALSO links the library already
	providing it produces a real LNK2005 "already defined" - the fix is to
	simply not manufacture a duplicate, and let the symbol resolve through
	the normal default link it was already going to resolve through.
	Returns None (no import library needed at all - `resolve_lib_ldflag`
	then omits the ldflag entirely) if every requested symbol was dropped
	this way. A symbol dropped here is NOT re-validated against ntdll's own
	export table below: the default link already proves it resolves,
	regardless of whether it happens to also be a genuine ntdll export.

	When `no_crt` is True, this filtering is skipped entirely - a
	genuinely freestanding build never links the default CRT at all (see
	_default_link_provides's own docstring for why: MSVC's explicit
	/NODEFAULTLIB, and clang/gcc's own default-CRT-library pull being
	conditioned on nothing here ever defining a competing mainCRTStartup),
	so every requested symbol still genuinely needs its own ntdll import
	entry, exactly as before this whole default-CRT-overlap check existed.
	`no_crt` MUST match whatever this same build will actually pass to
	CcTool.compile()/link() - see resolve_lib_ldflag's own docstring for the
	two ways getting this wrong is unsafe.

	Why this exists: ntdll.dll's actual export table (confirmed via `dumpbin
	/exports`) is far larger than what the SDK's ntdll.lib import library
	exposes - that .lib is a curated, documented-APIs-only subset. strnlen
	is a real, confirmed ntdll export the stub omits, which produces a real
	LNK2019 at link time for any program that needs it despite the DLL
	genuinely providing it (see lib/windows/ntdll.py's own note on its
	strnlen binding - the motivating case for this function). Rather than
	hand-roll a workaround per missing symbol, this generates a *real*
	import library straight from the DLL's own export table, so any
	genuine ntdll export metalpy declares via @extern works - not just
	the SDK-blessed subset.

	Scoped to `symbols` (not all ~2500 of ntdll's exports) rather than a
	wholesale replacement of the SDK's ntdll.lib: those are the only names
	any @extern('ntdll', ...) binding in this build could reference, and
	staying scoped sidesteps having to correctly model data exports/
	forwarders for symbols nothing here ever uses (ntdll's own export table
	happens to have neither today, confirmed by parsing its full dump, but
	nothing guarantees that stays true on every future Windows version).

	Raises RuntimeError if a requested symbol is not actually a real,
	named, non-forwarded export of the system ntdll.dll - a much clearer
	error than the LNK2019 that would otherwise surface deep in the link
	step for a genuine typo/nonexistent-symbol @extern binding.

	Cached to disk under %TEMP%/metalpy/ntdll_import_lib/, keyed by
	(compiler name, POST-filter symbol set) - same spirit as has_symbol()'s
	own cache - so the dumpbin/llvm-readobj probe and the lib.exe/llvm-lib
	build are each only ever paid once per distinct (compiler, symbol set).
	The no_crt=False default-CRT filtering above always runs first
	regardless (it has its own, separate cache), since it decides what the
	effective symbol set even is.
	'''
	import hashlib
	import tempfile

	if not no_crt:
		symbols = { s for s in symbols if not _default_link_provides( cc, s ) }
	if not symbols:
		return None

	key = hashlib.sha256( f'{cc.name}\0{",".join( sorted( symbols ))}'.encode() ).hexdigest()[:16]
	cache_dir = Path( tempfile.gettempdir() ) / 'metalpy' / 'ntdll_import_lib'
	lib_path = cache_dir / f'{key}.lib'
	if ensure_cache_dir( cache_dir ) and lib_path.is_file():
		# cache hit - skip both the export-table probe and the lib.exe/
		# llvm-lib build below entirely, same spirit as has_symbol()'s own
		# cache (this is the whole point of caching: a cache hit must not
		# still pay for the thing being cached)
		return lib_path

	real_exports = _real_ntdll_exports( cc )
	missing = symbols - real_exports
	if missing:
		raise RuntimeError(
			f"ntdll.dll does not export {sorted( missing )} as real, named, non-forwarded "
			f"symbols - check lib/windows/ntdll.py's @extern('ntdll', ...) declarations "
			f"against a real `dumpbin /exports {_NTDLL_PATH}`" )

	_, lib_builder = _ntdll_toolchain( cc )
	with tempfile.TemporaryDirectory() as tmp:
		def_path = Path( tmp ) / 'ntdll.def'
		out_path = Path( tmp ) / 'ntdll.lib'
		def_path.write_text(
			'LIBRARY ntdll.dll\nEXPORTS\n' + '\n'.join( f'\t{s}' for s in sorted( symbols ) ) + '\n',
			encoding = 'utf-8' )
		# x64-only, matching this whole file's existing implicit assumption
		# (compile()/link() above have no arch parameter either)
		cmd = [ lib_builder, f'/def:{def_path}', f'/out:{out_path}', '/machine:x64', '/nologo' ]
		if verbose:
			print( ' '.join( cmd ), file = sys.stderr )
		result = subprocess.run( cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True )
		if result.returncode != 0 or not out_path.is_file():
			raise RuntimeError( f'failed to build a custom ntdll import library:\n{result.stdout}' )
		data = out_path.read_bytes()
		if atomic_write_cache( lib_path, data ):
			return lib_path
		# the shared cache directory is unavailable this run (atomic_write_
		# cache already warned why) - out_path is about to be deleted along
		# with this TemporaryDirectory, but this function's own contract is
		# a real, on-disk .lib path regardless of whether caching it
		# actually worked, so fall back to a private, uncached copy outside
		# the shared tree rather than returning a path that was just
		# confirmed not to exist
		fallback_fd, fallback_name = tempfile.mkstemp( suffix = '.lib', prefix = 'metalpy_ntdll_' )
		with os.fdopen( fallback_fd, 'wb' ) as f:
			f.write( data )
		return Path( fallback_name )


def resolve_lib_ldflag( cc: CcTool, lib: str, symbols: set[str], verbose: bool = False, no_crt: bool = False ) -> str:
	'''
	The linker flag/path for one @extern library dependency, given the set
	of symbol names this build's program actually references from it.
	Every library except 'ntdll' resolves the ordinary way (a plain -l/.lib
	flag, searched against the compiler's own default library path) -
	ntdll is special-cased because the SDK's own ntdll.lib is missing real
	exports it should have (see build_ntdll_import_lib's docstring).

	`no_crt` must match whatever this same build will actually pass to
	CcTool.compile()/link(): build_ntdll_import_lib uses it to decide which
	requested ntdll symbols are already covered by this build's own default
	libraries (and so need no synthetic import entry at all) - passing the
	wrong value here can either wastefully synthesize an entry a no-CRT
	build didn't need, or - the actually unsafe direction - wrongly skip one
	a no-CRT build genuinely does need because a *different*, CRT-linked
	probe found it "already available".

	Returns '' when build_ntdll_import_lib determines no synthetic import
	library is needed at all (every requested ntdll symbol already resolves
	through this build's own default linking) - safe to append as an ldflag,
	same as any other empty/no-op flag.
	'''
	if lib == 'ntdll':
		lib_path = build_ntdll_import_lib( cc, symbols, verbose = verbose, no_crt = no_crt )
		return str( lib_path ) if lib_path is not None else ''
	return f'{lib}.lib' if cc.name == 'cl' else f'-l{lib}'


def find_dll( name: str ) -> Path|None:
	'''
	Locates a runtime DLL by bare filename (e.g. 'tcl86t.dll') for
	bundling into a build's output directory - see mpy.py's post-link
	step, driven by compiler.extern_dlls (populated from
	@extern(..., dll=...) declarations on functions actually reached).

	Searches PATH, in order - the same place a real Windows process
	resolves an unqualified DLL import from, so "found here" is a direct
	stand-in for "the exe would find this DLL too, if PATH weren't
	different at run time" (e.g. on a machine without this build's own
	dev tools installed). Not a general library search (no LIB/
	LIBRARY_PATH, no system directories) - those are for the .lib import
	library at link time, a different file that can live somewhere else
	entirely (see mpy_types.Function.extern_dll's own comment).

	Returns None (best-effort) if not found anywhere on PATH - mpy.py
	warns and continues rather than failing the build over a bundling
	step; the exe already linked successfully.
	'''
	for entry in os.environ.get( 'PATH', '' ).split( os.pathsep ):
		if not entry:
			continue
		candidate = Path( entry ) / name
		if candidate.is_file():
			return candidate
	return None


_CHKSTK_ASM = '''\
; __chkstk - x64 stack-probe support routine, normally supplied by the CRT.
; A freestanding (no_crt) MSVC build has no CRT to provide it, but cl.exe's
; own backend still silently emits a `call __chkstk` in the prologue of any
; function whose local frame exceeds one page (4KB) - see
; msvc_no_crt_missing_chkstk memory. Contract (cl.exe's own, undocumented but
; stable ABI): RAX = requested frame size in bytes; does NOT adjust RSP
; itself (unlike the x86 32-bit _chkstk) - only touches each page from the
; current RSP downward, in descending order, so the OS's guard-page
; mechanism commits/traps in the same order the prologue's own subsequent
; `sub rsp, rax` will actually access them. Skipping pages (e.g. touching
; only the final page) can hit the WRONG guard page and crash with an
; unrecoverable access violation instead of an ordinary, catchable stack
; overflow.
;
; This is the same body (translated to MASM/Intel syntax) as LLVM
; compiler-rt's ___chkstk_ms (x86_64/chkstk.S) - an independently-shipped,
; long-established reimplementation of the same undocumented MSVC ABI, used
; by GCC/Clang's own -mstack-probe. Only preserves RAX/RCX/flags, matching
; what a compiler-inserted call site actually expects to survive.
_TEXT SEGMENT

PUBLIC __chkstk

__chkstk PROC
        push    rcx
        push    rax
        cmp     rax, 1000h
        lea     rcx, [rsp+18h]
        jb      SHORT lastpage
probeloop:
        sub     rcx, 1000h
        test    qword ptr [rcx], rcx
        sub     rax, 1000h
        cmp     rax, 1000h
        ja      probeloop
lastpage:
        sub     rcx, rax
        test    qword ptr [rcx], rcx
        pop     rax
        pop     rcx
        ret
__chkstk ENDP

_TEXT ENDS

END
'''


def _build_chkstk_obj( verbose: bool = False ) -> Path|None:
	'''
	Builds (and disk-caches) __chkstk.obj, assembled from _CHKSTK_ASM via
	ml64.exe - see _CHKSTK_ASM's own comment for why a freestanding MSVC
	build needs this at all. Only ever called for a `cl` build (see link()'s
	own call site) - ml64.exe lives right alongside cl.exe in VC's Hostx64\\x64
	toolchain directory, so it's guaranteed to already be on PATH by the same
	vcvars64 import that put cl.exe there (detect_cc's own MSVC branch).

	Cannot be written as ordinary C: x64 MSVC has neither inline asm nor
	__declspec(naked), and __chkstk's calling convention (RAX in, no
	parameter-register/stack-frame setup, RSP left untouched) isn't
	expressible as a callable C function signature regardless - it has to be
	hand-assembled.

	Returns None (best-effort, never fails the build) if ml64.exe can't be
	found - the resulting build fails exactly as it did before this function
	existed (a LNK2019 for __chkstk, but only for a program whose no_crt
	build actually needs it).

	The assembly source is fixed/constant, so the cache key is just its own
	content hash (changes only if _CHKSTK_ASM itself is ever edited) - same
	spirit as build_ntdll_import_lib's own cache, but with nothing per-build
	to vary on.
	'''
	import hashlib
	import tempfile

	ml64 = shutil.which( 'ml64' )
	if ml64 is None:
		return None

	key = hashlib.sha256( _CHKSTK_ASM.encode() ).hexdigest()[:16]
	cache_dir = Path( tempfile.gettempdir() ) / 'metalpy' / 'chkstk_obj'
	obj_path = cache_dir / f'{key}.obj'
	if ensure_cache_dir( cache_dir ) and obj_path.is_file():
		return obj_path

	with tempfile.TemporaryDirectory() as tmp:
		asm_path = Path( tmp ) / 'chkstk.asm'
		out_path = Path( tmp ) / 'chkstk.obj'
		asm_path.write_text( _CHKSTK_ASM, encoding = 'utf-8' )
		cmd = [ ml64, '/nologo', '/c', f'/Fo{out_path}', str( asm_path ) ]
		if verbose:
			print( ' '.join( cmd ), file = sys.stderr )
		result = subprocess.run( cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True )
		if result.returncode != 0 or not out_path.is_file():
			print( f'WARNING - failed to assemble __chkstk.obj via ml64:\n{result.stdout}', file = sys.stderr )
			return None
		data = out_path.read_bytes()
		if atomic_write_cache( obj_path, data ):
			return obj_path
		# shared cache dir unavailable this run - fall back to a private,
		# uncached copy rather than a path just confirmed not to exist (same
		# fallback shape as build_ntdll_import_lib's own)
		fallback_fd, fallback_name = tempfile.mkstemp( suffix = '.obj', prefix = 'metalpy_chkstk_' )
		with os.fdopen( fallback_fd, 'wb' ) as f:
			f.write( data )
		return Path( fallback_name )


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
	cache_file = cache_dir / key
	if ensure_cache_dir( cache_dir ) and cache_file.is_file():
		# best-effort read, same reasoning as has_symbol's identical guard -
		# a torn/mid-replace read costs one re-probe, never correctness
		try:
			return cache_file.read_text( encoding = 'utf-8' ).strip() or None
		except OSError:
			pass

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

	atomic_write_cache( cache_file, found or '' )
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
		# vcvars64 has not been run - PATH already has it whenever this
		# shell descends from a Developer Command Prompt (VsDevCmd.bat adds
		# VC's own Auxiliary\Build dir), so try that first - it's an
		# ordinary shutil.which(), no subprocess spawn at all. Only fall
		# back to the slow vswhere-based search (a real VS install lookup)
		# when PATH doesn't already have it.
		vcvars64 = shutil.which( 'vcvars64' )
		if vcvars64 is None:
			print( 'WARNING - vcvars64 not on PATH - trying to auto-detect it via vswhere (this is slow)', file = sys.stderr )
			vswhere = Path( r'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe' )
			if vswhere.is_file():
				args = [ str( vswhere ), '-latest', '-products', '*', '-all', '-find', r'VC\Auxiliary\Build\vcvars64.bat' ]
				result = subprocess.run( args, capture_output = True, text = True )
				vcvars64 = result.stdout.strip() or None
		if vcvars64:
			print( f'vcvars64 path={vcvars64}', file = sys.stderr )
			output = subprocess.check_output( f'"{vcvars64}" && set', shell = True, text = True )
			for line in output.splitlines():
				if '=' in line:
					k, _, v = line.partition( '=' )
					os.environ[k] = v
			return CcTool( 'cl', 'cl' )

	return None
