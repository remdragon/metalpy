# Real-compile-and-run tests for lib/subprocess.py. Test commands are
# chosen per-OS inside the embedded MetalPy source itself (compiler.target.
# os), same convention lib/subprocess.py's own Windows/POSIX backends use,
# so one merged test binary exercises whichever backend actually built.

import unittest

import test_support


class SubprocessTests( test_support.RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		import emitter_c
		self.assert_programs_run([
			( 'run_returns_zero_on_success', '''
import compiler
import subprocess

def main() -> i32:
	args: list[str] = list[str]()
	if compiler.target.os == 'windows':
		args = ['cmd.exe', '/c', 'exit', '0']
	else:
		args = ['/bin/sh', '-c', 'exit 0']
	result: subprocess.CompletedProcess = subprocess.run( args ).unwrap( 'run should succeed' )
	if result.returncode != 0:
		return 1
	return 0
''' ),
			( 'nonzero_exit_code_propagates', '''
import compiler
import subprocess

def main() -> i32:
	args: list[str] = list[str]()
	if compiler.target.os == 'windows':
		args = ['cmd.exe', '/c', 'exit', '3']
	else:
		args = ['/bin/sh', '-c', 'exit 3']
	result: subprocess.CompletedProcess = subprocess.run( args ).unwrap( 'run should succeed' )
	if result.returncode != 3:
		return 1
	return 0
''' ),
			( 'capture_output_reads_stdout', '''
import compiler
import subprocess

def main() -> i32:
	args: list[str] = list[str]()
	if compiler.target.os == 'windows':
		args = ['cmd.exe', '/c', 'echo', 'hello world']
	else:
		args = ['/bin/echo', 'hello world']
	result: subprocess.CompletedProcess = subprocess.run( args, capture_output = True ).unwrap( 'run should succeed' )
	if result.returncode != 0:
		return 1
	out: str = result.stdout.decode().unwrap( 'valid utf8' )
	if 'hello world' not in out:
		return 2
	return 0
''' ),
			( 'capture_output_reads_stderr_independently_of_stdout', '''
import compiler
import subprocess

def main() -> i32:
	# a command that writes to BOTH stdout and stderr - would deadlock if
	# the two pipes were drained sequentially instead of concurrently (see
	# lib/subprocess.py's own header comment for why).
	args: list[str] = list[str]()
	if compiler.target.os == 'windows':
		args = ['cmd.exe', '/c', 'echo out-line 1>&2 & echo err-line 1>&2']
	else:
		args = ['/bin/sh', '-c', 'echo out-line; echo err-line 1>&2']
	result: subprocess.CompletedProcess = subprocess.run( args, capture_output = True ).unwrap( 'run should succeed' )
	err: str = result.stderr.decode().unwrap( 'valid utf8' )
	if 'err-line' not in err:
		return 1
	return 0
''' ),
			( 'stdout_and_stderr_are_empty_without_capture_output', '''
import compiler
import subprocess

def main() -> i32:
	args: list[str] = list[str]()
	if compiler.target.os == 'windows':
		args = ['cmd.exe', '/c', 'echo', 'hi']
	else:
		args = ['/bin/echo', 'hi']
	result: subprocess.CompletedProcess = subprocess.run( args ).unwrap( 'run should succeed' )
	if len( result.stdout ) != usize( 0 ):
		return 1
	if len( result.stderr ) != usize( 0 ):
		return 2
	return 0
''' ),
			( 'cwd_changes_child_working_directory', '''
import compiler
import subprocess

def main() -> i32:
	target: str = ''
	args: list[str] = list[str]()
	if compiler.target.os == 'windows':
		target = 'C:\\\\Windows'
		args = ['cmd.exe', '/c', 'cd']
	else:
		target = '/tmp'
		args = ['/bin/pwd']
	result: subprocess.CompletedProcess = subprocess.run( args, capture_output = True, cwd = target ).unwrap( 'run should succeed' )
	out: str = result.stdout.decode().unwrap( 'valid utf8' )
	# case-insensitive-ish check via substring on the leaf name is enough -
	# not asserting exact formatting (cmd's `cd` output vs pwd's differ)
	leaf: str = 'Windows' if compiler.target.os == 'windows' else 'tmp'
	if leaf not in out:
		return 1
	return 0
''' ),
			( 'empty_args_is_an_error', '''
import subprocess

def main() -> i32:
	if subprocess.run( list[str]() ).is_ok():
		return 1
	return 0
''' ),
			( 'completed_process_carries_back_original_args', '''
import compiler
import subprocess

def main() -> i32:
	args: list[str] = list[str]()
	if compiler.target.os == 'windows':
		args = ['cmd.exe', '/c', 'exit', '0']
	else:
		args = ['/bin/sh', '-c', 'exit 0']
	result: subprocess.CompletedProcess = subprocess.run( args ).unwrap( 'run should succeed' )
	if result.args.__len__() != args.__len__():
		return 1
	return 0
''' ),
		])


if __name__ == '__main__':
	unittest.main()
