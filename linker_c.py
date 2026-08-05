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

	def compile( self, src: Path, obj: Path, verbose: bool = False ) -> subprocess.CompletedProcess[bytes]:
		''' compile a single .c file to a .o object file '''
		if self.name == 'cl':
			cmd = [ self.path, '/nologo', '/std:c11',
				'/experimental:c11atomics',
				'/W4', '-c', str( src ), '/Fo:', str( obj ) ]
		else:
			cmd = [ self.path, '-std=c11', '-Wall', '-Wextra', '-c', str( src ), '-o', str( obj ) ]
		if verbose:
			print( ' '.join( cmd ), file = sys.stderr )
		return subprocess.run( cmd, capture_output = True, text = True )

	def link( self, exe: Path, objs: list[Path], ldflags: str = '', verbose: bool = False ) -> subprocess.CompletedProcess[bytes]:
		''' link one or more .o files into an executable '''
		obj_args = [ str( o ) for o in objs ]
		extra = ldflags.split() if ldflags else []
		if self.name == 'cl':
			cmd = [ 'link', '/nologo', f'/OUT:{exe}' ] + obj_args + extra
		else:
			cmd = [ self.path ] + extra + obj_args + [ '-o', str( exe ) ]
		if verbose:
			print( ' '.join( cmd ), file = sys.stderr )
		return subprocess.run( cmd, capture_output = True, text = True )


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
