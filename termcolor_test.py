# Real-compile-and-run tests for lib/termcolor.py: ANSI/CSI color helpers
# mirroring the real termcolor package's colored() signature closely enough
# to run existing code written against it (e.g. grap.mpy's cyan/yellow/red/
# grey helpers) unmodified.

# stdlib imports:
import unittest

# local imports:
import test_support


class TermcolorTests( test_support.RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		import emitter_c
		self.assert_programs_run([
			( 'plain_color_no_attrs', '''
import termcolor

def main() -> i32:
	s: str = termcolor.colored( 'hi', 'cyan' )
	if s != '\\x1b[36mhi\\x1b[0m':
		return 1
	return 0
''' ),
			( 'color_with_bold_attr', '''
import termcolor

def main() -> i32:
	# a list LITERAL directly as attrs=[...] hits a real compiler gap
	# (task_37940f20 - list literal rejected when the param's declared
	# type is list[T]|None, not a bare list[T]) - assigning to an
	# explicitly-typed local first works around it.
	bold: list[str] = ['bold']
	s: str = termcolor.colored( 'hi', 'red', attrs = bold )
	if s != '\\x1b[31;1mhi\\x1b[0m':
		return 1
	return 0
''' ),
			( 'grey_maps_to_bright_black', '''
import termcolor

def main() -> i32:
	bold: list[str] = ['bold'] # task_37940f20, see color_with_bold_attr
	s: str = termcolor.colored( 'hi', 'grey', attrs = bold )
	if s != '\\x1b[90;1mhi\\x1b[0m':
		return 1
	return 0
''' ),
			( 'no_color_no_attrs_is_a_passthrough', '''
import termcolor

def main() -> i32:
	s: str = termcolor.colored( 'plain' )
	if s != 'plain':
		return 1
	return 0
''' ),
			( 'bold_only_no_color', '''
import termcolor

def main() -> i32:
	bold: list[str] = ['bold'] # task_37940f20, see color_with_bold_attr
	s: str = termcolor.colored( 'hi', attrs = bold )
	if s != '\\x1b[1mhi\\x1b[0m':
		return 1
	return 0
''' ),
		])

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile termcolor tests' )
	def test_unknown_color_panics( self ) -> None:
		''' a real process-exit check, not merged via assert_programs_run:
		a panicking sub-program's abrupt exit(1) would break the merged
		dispatch's own "each case's main() returns normally" assumption. '''
		from discovery import Discovery
		from compiler import Compiler
		from pathlib import Path
		import emitter_c
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		compiler.import_code( '''
import termcolor

def main() -> i32:
	s: str = termcolor.colored( 'hi', 'not_a_real_color' )
	return 0
''', Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( compiler ), expected_exit = 1, compiler = compiler )


if __name__ == '__main__':
	unittest.main()
