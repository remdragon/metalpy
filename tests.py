# Test runner for the metalpy suite.
#
# The suite is dominated (wall-clock) by tests that compile MetalPy source to a
# real native executable and run it - each such test spawns a C compiler + linker
# and executes the result. Those tests are subprocess-bound, so running them
# serially wastes a multi-core machine. This runner shards the discovered tests
# across worker processes (one `python -m` style re-invocation of this file per
# shard) and aggregates the results. Sharding at test-method granularity (rather
# than per file) matters because emitter_c_test.py alone holds the large majority
# of the compile+link+run round trips - a per-file split would leave one worker
# doing almost all the work.
#
# Usage:
#   python tests.py              # parallel, min(cpu_count, 16) shards
#   python tests.py -j 8         # parallel, 8 shards
#   python tests.py --serial     # single-process (old behavior; for debugging)
#   python tests.py <test.id> …  # run only the named tests, serially

# stdlib imports:
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

_MAX_SHARDS = 16


def _flatten( suite: unittest.TestSuite ):
	''' yields the individual TestCase leaves of a (possibly nested) suite. '''
	for item in suite:
		if isinstance( item, unittest.TestSuite ):
			yield from _flatten( item )
		else:
			yield item


def _discover_ids() -> list[str]:
	''' returns every discoverable test's fully-qualified id, sorted. '''
	loader = unittest.TestLoader()
	suite = loader.discover( start_dir = '.', pattern = '*_test.py' )
	return sorted( t.id() for t in _flatten( suite ) )


def _run_names( names: list[str], verbosity: int ) -> unittest.TestResult:
	''' loads and runs the named tests serially, returning the TestResult.
	Each name is loaded independently so one un-loadable name (e.g. a module
	that fails to import) is reported as an error rather than aborting the
	whole shard. '''
	loader = unittest.TestLoader()
	suite = unittest.TestSuite()
	for name in names:
		try:
			suite.addTests( loader.loadTestsFromName( name ) )
		except Exception:  # noqa: BLE001 - surface the load failure as a fake failing test
			suite.addTest( _LoadErrorTest( name ) )
	return unittest.TextTestRunner( verbosity = verbosity, stream = sys.stderr ).run( suite )


class _LoadErrorTest( unittest.TestCase ):
	''' stands in for a test id that could not be loaded, so the failure is
	reported through the normal result machinery instead of crashing a shard. '''
	def __init__( self, name: str ) -> None:
		super().__init__( 'runTest' )
		self._name = name

	def runTest( self ) -> None:  # noqa: N802 - unittest naming
		import traceback
		self.fail( f'could not load test {self._name!r}:\n{traceback.format_exc()}' )

	def id( self ) -> str:
		return self._name


# --- worker mode -----------------------------------------------------------

def _worker_main( ids_file: str, result_file: str, verbosity: int ) -> int:
	''' runs one shard: reads test ids from ids_file, runs them, writes a JSON
	summary to result_file, and exits 0 iff the shard was fully successful. '''
	names = [ line for line in Path( ids_file ).read_text( encoding = 'utf-8' ).splitlines() if line.strip() ]
	result = _run_names( names, verbosity )
	summary = {
		'testsRun': result.testsRun,
		'failures': len( result.failures ),
		'errors': len( result.errors ),
		'skipped': len( result.skipped ),
		'expectedFailures': len( result.expectedFailures ),
		'unexpectedSuccesses': len( result.unexpectedSuccesses ),
		'success': result.wasSuccessful(),
	}
	Path( result_file ).write_text( json.dumps( summary ), encoding = 'utf-8' )
	return 0 if result.wasSuccessful() else 1


# --- parallel driver -------------------------------------------------------

def _run_parallel( jobs: int | None, verbosity: int ) -> bool:
	''' shards the discovered tests across worker subprocesses and aggregates.
	Returns True iff every shard succeeded. '''
	ids = _discover_ids()
	if not ids:
		print( 'no tests discovered', file = sys.stderr )
		return False

	shard_count = jobs if jobs else min( os.cpu_count() or 1, _MAX_SHARDS )
	shard_count = max( 1, min( shard_count, len( ids ) ) )

	# round-robin so the heavy emitter_c_test.py compile+link+run tests spread
	# evenly across shards instead of clustering in one worker
	shards: list[list[str]] = [ [] for _ in range( shard_count ) ]
	for i, tid in enumerate( ids ):
		shards[ i % shard_count ].append( tid )

	print( f'running {len(ids)} tests across {shard_count} shards...', file = sys.stderr )
	start = time.perf_counter()

	this_file = os.path.abspath( __file__ )
	with tempfile.TemporaryDirectory( prefix = 'metalpy_tests_' ) as tmp:
		procs = []
		for si, shard in enumerate( shards ):
			ids_file = Path( tmp ) / f'shard_{si}.ids'
			result_file = Path( tmp ) / f'shard_{si}.json'
			out_file = Path( tmp ) / f'shard_{si}.out'
			ids_file.write_text( '\n'.join( shard ), encoding = 'utf-8' )
			out_fh = open( out_file, 'wb' )
			proc = subprocess.Popen(
				[ sys.executable, this_file, '--worker', str( ids_file ), str( result_file ), '--verbosity', str( verbosity ) ],
				stdout = out_fh, stderr = subprocess.STDOUT,
			)
			procs.append( ( si, proc, out_fh, result_file, out_file, len( shard ) ) )

		# wait for all shards (they run concurrently; we just collect in order)
		totals = { 'testsRun': 0, 'failures': 0, 'errors': 0, 'skipped': 0, 'expectedFailures': 0, 'unexpectedSuccesses': 0 }
		failed_shards = []
		for si, proc, out_fh, result_file, out_file, n in procs:
			rc = proc.wait()
			out_fh.close()
			if result_file.exists():
				summary = json.loads( result_file.read_text( encoding = 'utf-8' ) )
				for k in totals:
					totals[ k ] += summary.get( k, 0 )
				ok = summary.get( 'success', False )
			else:
				# worker crashed before writing its summary
				ok = False
			if rc != 0 or not ok:
				failed_shards.append( ( si, out_file.read_text( encoding = 'utf-8', errors = 'replace' ) if out_file.exists() else '' ) )

	elapsed = time.perf_counter() - start

	for si, output in failed_shards:
		print( f'\n===== shard {si} output =====', file = sys.stderr )
		print( output, file = sys.stderr )

	success = not failed_shards
	print( '\n' + '-' * 60, file = sys.stderr )
	print(
		f'ran {totals["testsRun"]} tests in {elapsed:.2f}s across {shard_count} shards - '
		f'{"OK" if success else "FAILED"} '
		f'(failures={totals["failures"]}, errors={totals["errors"]}, skipped={totals["skipped"]})',
		file = sys.stderr,
	)
	return success


def main( argv: list[str] ) -> int:
	parser = argparse.ArgumentParser( description = 'metalpy test runner' )
	parser.add_argument( '--worker', nargs = 2, metavar = ( 'IDS_FILE', 'RESULT_FILE' ), help = argparse.SUPPRESS )
	parser.add_argument( '--serial', action = 'store_true', help = 'run everything in a single process (debugging)' )
	parser.add_argument( '-j', '--jobs', type = int, default = None, help = 'number of parallel shards (default: min(cpu_count, 16))' )
	parser.add_argument( '--verbosity', type = int, default = 1, help = 'unittest verbosity (default 1)' )
	parser.add_argument( 'names', nargs = '*', help = 'specific test ids to run (implies serial)' )
	args = parser.parse_args( argv )

	if args.worker:
		return _worker_main( args.worker[ 0 ], args.worker[ 1 ], args.verbosity )

	if args.names:
		result = _run_names( args.names, args.verbosity )
		return 0 if result.wasSuccessful() else 1

	if args.serial:
		loader = unittest.TestLoader()
		suite = loader.discover( start_dir = '.', pattern = '*_test.py' )
		result = unittest.TextTestRunner( verbosity = args.verbosity ).run( suite )
		return 0 if result.wasSuccessful() else 1

	return 0 if _run_parallel( args.jobs, args.verbosity ) else 1


if __name__ == '__main__':
	sys.exit( main( sys.argv[ 1: ] ) )
