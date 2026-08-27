#!/usr/bin/env python3
'''
MetalPy compiler driver — converts a .py source file into a native executable.

Usage:
    mpy.py [flags] <source.py>

The compiler pipeline:
  stage 1 (discovery)       — parse source + imports, register all type-level info
  stage 2 (type resolution) — walk from main(), resolve every reachable symbol
  stage 3 (lowering)        — convert every reachable function/class body to IR
  stage 4 (IR optimization) — currently a no-op (reserved slot)
  stage 5 (emission)        — emit C11 source from IR
  stage 6 (linking)         — detect C compiler, compile .c → .o, link → executable
'''

# stdlib imports:
import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile

# local imports:
from compiler import Compiler
from discovery import ActiveTarget, Discovery
import emitter_c
from errors import CompileError
import linker_c
import targets

def _parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(
		prog = 'mpy',
		description = 'MetalPy compiler — compile a .py source file to a native executable',
	)
	p.add_argument( 'source', type = Path, help = 'MetalPy source file (.py)' )
	p.add_argument( '-o', '--output', type = Path, default = None,
		help = 'output executable path (default: ./dist/<source stem>, created on demand)' )
	p.add_argument( '--release', action = 'store_true',
		help = 'build in release mode (default: debug)' )
	p.add_argument( '--cc', type = str, default = None,
		help = 'C compiler override (clang/gcc/msvc, overrides METALPY_CC)' )
	p.add_argument( '--dep-report', action = 'store_true',
		help = 'print every lowered FQDN after compilation' )
	p.add_argument( '-v', action = 'store_true',
		help = 'verbose print all compiler/linker commands as they run' )
	p.add_argument( '-c', action = 'store_true',
		help = 'emit C source only, do not compile or link' )
	p.add_argument( '--keep-c', action = 'store_true',
		help = 'on compile failure, save generated C to -o path (or source.c) instead of discarding' )
	p.add_argument( '--cflags', type = str, default = '',
		help = 'extra flags passed through to the C compiler' )
	p.add_argument( '--ldflags', type = str, default = '',
		help = 'extra flags passed through to the linker' )
	p.add_argument( '--strip', action = 'store_true',
		help = 'strip symbols / fold identical code for a smaller binary' )
	p.add_argument( '--asan', action = 'store_true',
		help = 'build with AddressSanitizer (requires the C runtime - forces CRT linking for a program that would otherwise build freestanding/no-CRT)' )
	p.add_argument( '--crt', action = 'store_true',
		help = 'force CRT linking even if the program itself never uses a \'c\' extern (normally: no_crt = \'c\' not in compiler.extern_libs) - e.g. to get __chkstk/other CRT-only support routines without adding a throwaway extern call' )
	p.add_argument( '--hide-warnings', action = 'store_true',
		help = 'suppress compiler warnings on an otherwise-successful build (shown by default)' )
	p.add_argument( '--no-leak-check', action = 'store_true',
		help = 'disable the automatic debug-build leak-check epilogue (decref globals + dump_live_objects at exit) - no effect in release builds' )
	p.add_argument( '--assume-threaded', action = 'store_true',
		help = 'force Part A/B\'s locking machinery on even if the program never reaches pthread_create/CreateThread - escape hatch for a program that reaches a second OS thread some other way (a raw signal handler, an externally-invoked C callback) this compiler cannot see (normally: locking is skipped whenever no reachable code ever spawns a thread)' )
	return p.parse_args()

def _die( msg: str ) -> None:
	print( f'mpy: error: {msg}', file = sys.stderr )
	sys.exit( 1 )

_DIST_DIR = Path( 'dist' )

def _default_output_path( source: Path, suffix: str ) -> Path:
	'''
	Default build output location when -o/--output isn't given:
	./dist/<source stem><suffix>, creating dist/ on demand. A real project
	build directory rather than wherever the source file happens to live
	(which could be deep in lib/), so a program that also needs to bundle
	runtime files alongside its exe (DLLs, script libraries, ...) has
	somewhere real to put them. Only used for the DEFAULT - an explicit
	-o/--output path is always respected as given, with no directory
	auto-creation, so it stays predictable for scripts that pass one.
	'''
	_DIST_DIR.mkdir( parents = True, exist_ok = True )
	return _DIST_DIR / ( source.stem + suffix )

def _build_active_target( args: argparse.Namespace, cc: linker_c.CcTool|None ) -> ActiveTarget:
	'''
	Build the active_target dict. Starts from targets.detect() just
	like Discovery does, then overrides debug based on --release and has_i128
	based on the detected C compiler backend (`cc` must already be detected -
	compile_time_transformer folds compiler.target.* eagerly, so this needs a
	concrete answer before Discovery ever starts, not a lazily-resolved one).
	'''
	target = targets.detect()
	target['debug'] = not args.release
	target['has_i128'] = linker_c.has_i128( cc )
	return target

def _print_dep_report( compiler: Compiler ) -> None:
	'''
	Print every FQDN that was lowered, grouped by kind, with the
	FQDN that triggered each one.
	'''
	def _triggered( unit: object ) -> str:
		# unwrap LoweredFunction -> Function, Specialization -> .base/.qualname
		if hasattr( unit, 'function' ):
			unit = unit.function
		u = unit
		if hasattr( u, 'base' ) and hasattr( u, 'qualname' ):
			u = u.base if isinstance( u.base, object ) else u
		t = compiler.type_resolver.triggered_by( unit )
		return f' (via {t})' if t else ''

	print( '\n=== dependency report ===' )
	print( f'functions ({len(compiler.functions)}):' )
	for lf in compiler.functions:
		print( f'  {lf.function.qualname}{_triggered(lf.function)}' )
	if compiler.rcclasses:
		print( f'rcclasses ({len(compiler.rcclasses)}):' )
		for cls in compiler.rcclasses:
			print( f'  {cls.qualname}{_triggered(cls)}' )
	if compiler.cstructs:
		print( f'cstructs ({len(compiler.cstructs)}):' )
		for cls in compiler.cstructs:
			print( f'  {cls.qualname}{_triggered(cls)}' )
	if compiler.cunions:
		print( f'cunions ({len(compiler.cunions)}):' )
		for cls in compiler.cunions:
			print( f'  {cls.qualname}{_triggered(cls)}' )
	if compiler.tagged_unions:
		print( f'tagged_unions ({len(compiler.tagged_unions)}):' )
		for cls in compiler.tagged_unions:
			print( f'  {cls.qualname}{_triggered(cls)}' )
	if compiler.cenums:
		print( f'cenums ({len(compiler.cenums)}):' )
		for cls in compiler.cenums:
			print( f'  {cls.qualname}{_triggered(cls)}' )
	if compiler.globals:
		print( f'globals ({len(compiler.globals)}):' )
		for g in compiler.globals:
			print( f'  {g.variable.qualname}' )
	print()

def main() -> None:
	args = _parse_args()

	# --- validate source file ---
	if not args.source.is_file():
		_die( f'source file not found: {args.source}' )

	# --- compiler override (must happen before detect_cc() below) ---
	if args.cc:
		os.environ['METALPY_CC'] = args.cc

	# --- detect compiler early: active_target['has_i128'] needs a concrete
	# answer before Discovery even starts (see _build_active_target) - reused
	# again at stage 6 below, so this only ever detects once per run ---
	cc = linker_c.detect_cc()

	# --- build active target ---
	active_target = _build_active_target( args, cc )

	# --- stage 1: discovery ---
	disco = Discovery( import_builtins = True, active_target = active_target )
	compiler = Compiler( disco )

	# --- load source ---
	try:
		compiler.import_file( args.source.resolve() )
	except CompileError:
		pass # errors already in disco.errors

	# --- stage 2-3: type resolution + lowering ---
	try:
		compiler.run()
	except CompileError:
		pass

	# --- report discovery/compilation errors ---
	if disco.errors.errors:
		for err in disco.errors.errors:
			print( f'mpy: {err}', file = sys.stderr )
		sys.exit( 1 )

	if not compiler.functions:
		_die( 'no functions were lowered (missing main() maybe?)' )

	# --- dep report ---
	if args.dep_report:
		_print_dep_report( compiler )
		return

	# --- stage 5: emit C ---
	if args.assume_threaded:
		compiler.spawns_threads = True
	no_crt = 'c' not in compiler.extern_libs and not compiler.requires_crt
	if args.crt:
		no_crt = False
	if args.asan and no_crt:
		print( 'WARNING - --asan requires the C runtime - forcing CRT linking (no_crt=True request ignored)', file = sys.stderr )
		no_crt = False
	try:
		c_source = emitter_c.emit_c( compiler, no_crt = no_crt, leak_check = not args.no_leak_check )
	except CompileError:
		pass # errors already in disco.errors - e.g. _topologically_sort_globals' own circular-dependency fail_loc

	# --- report emission errors (e.g. circular global-initializer dependency) ---
	if disco.errors.errors:
		for err in disco.errors.errors:
			print( f'mpy: {err}', file = sys.stderr )
		sys.exit( 1 )

	# --- -c: emit C source only ---
	if args.c:
		c_path = args.output or _default_output_path( args.source, '.c' )
		c_path.write_text( c_source, encoding = 'utf-8' )
		print( f'mpy: wrote {c_path}' )
		return

	# --- stage 6: compiler was already detected above (needed early for has_i128) ---
	if cc is None:
		_die( 'no C compiler found (try --cc or METALPY_CC)' )

	with tempfile.TemporaryDirectory() as tmp:
		src_path = Path( tmp ) / 'generated.c'
		obj_path = Path( tmp ) / 'generated.o'
		src_path.write_text( c_source, encoding = 'utf-8' )

		compile_result = cc.compile( src_path, obj_path, verbose = args.v,
			no_crt = no_crt,
			debug = bool( active_target['debug'] ),
			asan = args.asan,
			cflags = args.cflags,
			warnings = not args.hide_warnings,
		)
		if compile_result.returncode != 0:
			print( f'mpy: {cc.name} compile failed:', file = sys.stderr )
			print( compile_result.stdout, file = sys.stderr )
			if args.keep_c:
				c_path = args.output or _default_output_path( args.source, '.c' )
				src_path.rename( c_path )
				print( f'mpy: generated C kept at {c_path}', file = sys.stderr )
			sys.exit( 1 )
		elif not args.hide_warnings and compile_result.stdout.strip():
			# build succeeded but the compiler still had something to say (e.g.
			# -Wall/-Wextra or /W4 warnings) - shown by default: the shared
			# builtins runtime and every reachable stdlib module now compile
			# warning-free, so a warning here means the USER's own program
			# triggered it. --hide-warnings opts back out to quiet output.
			print( f'mpy: {cc.name} compile warnings:', file = sys.stderr )
			print( compile_result.stdout, file = sys.stderr )

		# --- link .o → executable ---
		exe_path = (args.output or _default_output_path( args.source, '' )).resolve()
		if active_target['os'] == 'windows' and exe_path.suffix != '.exe':
			exe_path = exe_path.with_suffix( exe_path.suffix + '.exe' )
		ldflags = args.ldflags
		for lib in sorted( compiler.extern_libs ):
			if lib == 'c':
				continue
			if lib not in ldflags:
				flag = linker_c.resolve_lib_ldflag( cc, lib, compiler.extern_libs[lib], verbose = args.v, no_crt = no_crt )
				ldflags = ldflags + f' {flag}' if ldflags else flag
		link_result = cc.link( exe_path, [ obj_path ], ldflags = ldflags, verbose = args.v, no_crt = no_crt, debug = bool( active_target['debug'] ), asan = args.asan, strip = args.strip )
		if link_result.returncode != 0:
			print( f'mpy: {cc.name} link failed:', file = sys.stderr )
			print( link_result.stdout, file = sys.stderr )
			if args.keep_c:
				c_path = args.output or _default_output_path( args.source, '.c' )
				src_path.rename( c_path )
				print( f'mpy: generated C kept at {c_path}', file = sys.stderr )
			sys.exit( 1 )
		elif not args.hide_warnings and link_result.stdout.strip():
			print( f'mpy: {cc.name} link warnings:', file = sys.stderr )
			print( link_result.stdout, file = sys.stderr )

		print( f'mpy: built {exe_path}' )

		# --- bundle runtime DLL dependencies declared via @extern(..., dll=...) -
		# driven entirely by compiler.extern_dlls (populated only from functions
		# actually reached/lowered, same reachability gate as extern_libs), never
		# anything hardcoded to a particular library here. Every declared name is
		# something the author explicitly said this program needs at runtime, so
		# unlike a merely-advisory step, failing to find or copy one fails the
		# whole build - a program missing a runtime dependency it's known to need
		# isn't safely deployable, and finding out via mpy's own exit code beats
		# finding out when the shipped exe won't start on another machine. ---
		bundle_errors: list[str] = []
		for dll_name in sorted( compiler.extern_dlls ):
			found = linker_c.find_dll( dll_name )
			if found is None:
				bundle_errors.append( f'{dll_name}: not found on PATH' )
				continue
			dest = exe_path.parent / dll_name
			try:
				shutil.copy2( found, dest )
				print( f'mpy: bundled {dest}' )
			except OSError as e:
				bundle_errors.append( f'{dll_name}: {e}' )

		# --- combine 3rd-party license notices declared via
		# @extern(..., notice=...) into one dist/THIRD-PARTY-LICENSES.txt -
		# driven entirely by compiler.extern_notices (same reachability
		# gate as extern_dlls above, same "explicit, author-listed, fails
		# the build if unresolvable" philosophy). Each identifier resolves
		# to licenses/<NAME>.txt next to this script - a fixed metalpy-
		# installation-relative directory (Path(__file__).parent), the
		# same anchor discovery.py's own Discovery.__init__ uses to find
		# lib/ (licenses/ is source material shipped with metalpy itself,
		# not build output - unlike dist/, which is CWD-relative because
		# it belongs wherever the caller is building). Deliberately
		# separate from extern_dlls: one notice (e.g. 'ZLIB') can cover
		# several otherwise-unrelated DLL dependencies across different
		# libraries, so a program bundling zlib1.dll for a reason unrelated
		# to Tcl/Tk would reference the same file rather than a duplicate
		# copy. ---
		if compiler.extern_notices:
			licenses_dir = Path( __file__ ).parent / 'licenses'
			notice_blocks: list[str] = []
			for name in sorted( compiler.extern_notices ):
				notice_path = licenses_dir / f'{name}.txt'
				try:
					notice_blocks.append( f'{"=" * 20} {name} {"=" * 20}\n{notice_path.read_text( encoding = "utf-8" )}' )
				except OSError:
					bundle_errors.append( f'{name}: license notice not found at {notice_path}' )
			if notice_blocks:
				notices_dest = exe_path.parent / 'THIRD-PARTY-LICENSES.txt'
				notices_dest.write_text( '\n\n'.join( notice_blocks ), encoding = 'utf-8' )
				print( f'mpy: wrote {notices_dest}' )

		if bundle_errors:
			print( 'mpy: error: failed to bundle required runtime DLL(s)/license notice(s):', file = sys.stderr )
			for err in bundle_errors:
				print( f'  {err}', file = sys.stderr )
			sys.exit( 1 )

if __name__ == '__main__':
	main()
