import sys
import unittest

import test_support
from compiler import Compiler
from discovery import Discovery


@unittest.skipUnless( sys.platform.startswith( 'linux' ), 'lib/pty.py only has a Linux backend so far (macOS/Windows are poison-pill/absent - see terminal_passthrough_investigation.md) - unlike os_test.py\'s both-platforms os.rename, pty.py\'s functions have NO Windows definition at all, so a program referencing them doesn\'t even compile for a Windows target, runtime no-op branch or not; same posture as ssl_test.py\'s own Linux-only backend tests' )
class PtyTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' Real compile-and-run coverage for lib/pty.py (PTY allocation +
	posix_spawn-attached shell, no fork()). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'open_pty_gives_a_real_slave_path', '''
import pty

def main() -> i32:
	pair: pty.PtyPair = pty.open_pty().unwrap( 'open_pty' )
	if not pair.slave_path.startswith( '/dev/' ):
		return 1
	return 0
''' ),
			( 'spawn_attached_runs_a_real_command_round_trip', '''
import compiler
import fs
import sys
import pty

def main() -> i32:
	pair: pty.PtyPair = pty.open_pty().unwrap( 'open_pty' )

	argv: list[str] = list[str]()
	argv.append( "/bin/sh" )
	pid: i32 = pty.spawn_attached( argv, pair.slave_path ).unwrap( 'spawn_attached' )
	if pid <= 0:
		return 1

	cmd: str = "echo pty_test_marker\\n"
	fs.write_all( pair.master, cmd.get_cstr(), usize( cmd.__len__() )).unwrap( 'write cmd' )

	# accumulate reads until the marker shows up or we give up - a single
	# blocking read can legitimately return a partial chunk (echo, then
	# the command\\'s own output, can arrive as separate writes) - see
	# terminal_passthrough_investigation.md's own note on this.
	found: bool = False
	attempt: i32 = 0
	buf: Ptr[u8] = sys.alloc[u8]( 4096 )
	while attempt < 20:
		result: Result[usize, OSError] = fs.read_raw( pair.master, buf, usize( 4095 ))
		match result:
			case Result.Ok( n ):
				if n > usize( 0 ):
					buf[n] = 0
					with compiler.wrap_arithmetic:
						nul_len: usize = n + usize( 1 )
					match str.from_cstr( compiler.cast( ConstPtr[u8], buf ), nul_len ):
						case Result.Ok( out ):
							if 'pty_test_marker' in out:
								found = True
						case Result.Err( _ ):
							pass
			case Result.Err( _ ):
				pass
		if found:
			break
		with compiler.wrap_arithmetic:
			attempt = attempt + 1
	sys.free( buf )

	exitcmd: str = "exit\\n"
	fs.write_all( pair.master, exitcmd.get_cstr(), usize( exitcmd.__len__() )).is_ok()
	pty.wait( pid ).unwrap( 'wait' )
	fs.close_raw( pair.master ).is_ok()

	if not found:
		return 2
	return 0
''' ),
			( 'resize_succeeds_on_a_live_pty', '''
import fs
import pty

def main() -> i32:
	pair: pty.PtyPair = pty.open_pty().unwrap( 'open_pty' )
	pty.resize( pair.master, u16( 50 ), u16( 120 )).unwrap( 'resize' )
	fs.close_raw( pair.master ).is_ok()
	return 0
''' ),
		], timeout = 15.0 )


if __name__ == '__main__':
	unittest.main()
