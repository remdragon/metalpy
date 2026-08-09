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
			cmd = [ self.path ] + extra + obj_args + [ '-o', str( exe ) ]
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
