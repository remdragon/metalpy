# stdlib imports:
from pathlib import Path
from typing import NoReturn

class CompileError( Exception ):
	'''
	unwinds to the nearest recovery boundary - the failure is already
	recorded in an ErrorCollector by the time this is raised, so callers
	that catch it need no data from it, just `except CompileError: pass`
	'''

class RedundantCompilationError( CompileError ):
	'''
	raised when a lookup resolves to a name whose creation/resolution
	already failed - the real error was recorded once, at that failure;
	this is deliberately silent (no ErrorCollector.error() call) so it
	doesn't produce a second, confusing diagnostic on top of the first. A
	plain `except CompileError` (the existing recovery-boundary idiom used
	everywhere) catches it for free, since it's just a CompileError
	subclass.
	'''

class ErrorCollector:
	def __init__( self ) -> None:
		self.errors: list[str] = []

	def error( self, message: str, file: Path|None, line: int|None ) -> None:
		location = f'{file}:{line}: ' if file is not None and line is not None else ( f'{file}: ' if file is not None else '' )
		self.errors.append( f'{location}{message}' )

	def fail( self, message: str, file: Path|None, line: int|None ) -> NoReturn:
		self.error( message, file, line )
		raise CompileError( message )
