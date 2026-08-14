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


class CcTool:
	'''
	A detected C compiler.

	name: 'clang', 'gcc', or 'cl'
	path: full path (or just 'cl' for MSVC since it's in PATH after vcvars)
	'''

	def __init__( self, name: str, path: str ) -> None:
		self.name = name
		self.path = path

	def compile( self, src: Path, obj: Path, verbose: bool = False, no_crt: bool = False ) -> subprocess.CompletedProcess[bytes]:
		''' compile a single .c file to a .o object file '''
		if self.name == 'cl':
			cmd = [ self.path, '/nologo', '/std:c11',
				'/experimental:c11atomics',
				'/W4', '-c', str( src ), f'/Fo:{obj}' ]
			if no_crt:
				cmd += [ '/GS-' ]
		else:
			cmd = [ self.path, '-std=c11', '-Wall', '-Wextra', '-c', str( src ), '-o', str( obj ) ]
		if verbose:
			print( ' '.join( cmd ), file = sys.stderr )
		return subprocess.run( cmd,
			stdout = subprocess.PIPE,
			stderr = subprocess.STDOUT,
			text = True,
		)

	def link( self, exe: Path, objs: list[Path], ldflags: str = '', verbose: bool = False, no_crt: bool = False ) -> subprocess.CompletedProcess[bytes]:
		''' link one or more .o files into an executable '''
		obj_args = [ str( o ) for o in objs ]
		extra = ldflags.split() if ldflags else []
		if self.name == 'cl':
			cmd = [ 'link', '/nologo', f'/OUT:{exe}' ] + obj_args + extra
			if no_crt:
				cmd += [ '/NODEFAULTLIB', '/ENTRY:mainCRTStartup' ]
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
		if verbose:
			print( ' '.join( cmd ), file = sys.stderr )
		return subprocess.run( cmd,
			stdout = subprocess.PIPE,
			stderr = subprocess.STDOUT,
			text = True,
		)


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
		return cache_file.read_text().strip() == '1'

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

	cache_file.write_text( '1' if available else '0', encoding = 'utf-8' )
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
