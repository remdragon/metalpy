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
import sys
import tempfile

# local imports:
import emitter_c
import linker_c
from compiler import Compiler
from discovery import Discovery, _detect_active_target
from errors import CompileError

def _parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(
		prog = 'mpy',
		description = 'MetalPy compiler — compile a .py source file to a native executable',
	)
	p.add_argument( 'source', type = Path, help = 'MetalPy source file (.py)' )
	p.add_argument( '-o', '--output', type = Path, default = None,
		help = 'output executable name (default: source stem)' )
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
	return p.parse_args()

def _die( msg: str ) -> None:
	print( f'mpy: error: {msg}', file = sys.stderr )
	sys.exit( 1 )

def _build_active_target( args: argparse.Namespace ) -> dict[str,object]:
	'''
	Build the active_target dict. Starts from _detect_active_target() just
	like Discovery does, then overrides debug based on --release.
	'''
	target = _detect_active_target()
	target['debug'] = not args.release
	return target

def _print_dep_report( compiler: Compiler ) -> None:
	'''
	Print every FQDN that was lowered, grouped by kind.
	Future: track which FQDN triggered each lowering (needs instrumentation
	in TypeResolver/lowering.py to record scheduling provenance).
	'''
	print( '\n=== dependency report ===' )
	print( f'functions ({len(compiler.functions)}):' )
	for lf in compiler.functions:
		print( f'  {lf.function.qualname}' )
	if compiler.rcclasses:
		print( f'rcclasses ({len(compiler.rcclasses)}):' )
		for cls in compiler.rcclasses:
			print( f'  {cls.qualname}' )
	if compiler.cstructs:
		print( f'cstructs ({len(compiler.cstructs)}):' )
		for cls in compiler.cstructs:
			print( f'  {cls.qualname}' )
	if compiler.cunions:
		print( f'cunions ({len(compiler.cunions)}):' )
		for cls in compiler.cunions:
			print( f'  {cls.qualname}' )
	if compiler.tagged_unions:
		print( f'tagged_unions ({len(compiler.tagged_unions)}):' )
		for cls in compiler.tagged_unions:
			print( f'  {cls.qualname}' )
	if compiler.cenums:
		print( f'cenums ({len(compiler.cenums)}):' )
		for cls in compiler.cenums:
			print( f'  {cls.qualname}' )
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

	# --- build active target ---
	active_target = _build_active_target( args )

	# --- compiler override ---
	if args.cc:
		os.environ['METALPY_CC'] = args.cc

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

	# --- stage 5: emit C ---
	c_source = emitter_c.emit_c( compiler )

	# --- -c: emit C source only ---
	if args.c:
		c_path = args.output or args.source.with_suffix( '.c' )
		c_path.write_text( c_source, encoding = 'utf-8' )
		print( f'mpy: wrote {c_path}' )
		return

	# --- stage 6: detect compiler ---
	cc = linker_c.detect_cc()
	if cc is None:
		_die( 'no C compiler found (try --cc or METALPY_CC)' )

	# --- compile C → .o ---
	with tempfile.TemporaryDirectory() as tmp:
		src_path = Path( tmp ) / 'generated.c'
		obj_path = Path( tmp ) / 'generated.o'
		src_path.write_text( c_source, encoding = 'utf-8' )

		result = cc.compile( src_path, obj_path, verbose = args.v )
		if result.returncode != 0:
			print( f'mpy: {cc.name} compile failed:', file = sys.stderr )
			print( result.stdout, file = sys.stderr )
			if args.keep_c:
				c_path = args.output or args.source.with_suffix( '.c' )
				src_path.rename( c_path )
				print( f'mpy: generated C kept at {c_path}', file = sys.stderr )
			sys.exit( 1 )

		# --- link .o → executable ---
		exe_path = (args.output or args.source.with_suffix( '' )).resolve()
		if active_target['os'] == 'windows' and exe_path.suffix != '.exe':
			exe_path = exe_path.with_suffix( exe_path.suffix + '.exe' )
		ldflags = args.ldflags
		# auto-link every @extern library that was actually lowered.
		# 'c' means the platform C runtime, already linked implicitly.
		for lib in sorted( compiler.extern_libs ):
			if lib == 'c':
				continue
			if lib not in ldflags:
				ldflags = ldflags + f' -l{lib}' if ldflags else f'-l{lib}'
		result = cc.link( exe_path, [ obj_path ], ldflags = ldflags, verbose = args.v )
		if result.returncode != 0:
			print( f'mpy: {cc.name} link failed:', file = sys.stderr )
			if result.stdout:
				print( result.stdout, file = sys.stderr )
			if result.stderr:
				print( result.stderr, file = sys.stderr )
			if args.keep_c:
				c_path = args.output or args.source.with_suffix( '.c' )
				src_path.rename( c_path )
				print( f'mpy: generated C kept at {c_path}', file = sys.stderr )
			sys.exit( 1 )

		print( f'mpy: built {exe_path}' )

if __name__ == '__main__':
	main()
