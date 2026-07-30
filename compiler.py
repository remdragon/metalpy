# stdlib imports:
from pathlib import Path

# local imports:
from discovery import Discovery
from mpy_types import Module

class Compiler:
	def __init__( self, disco: Discovery ) -> None:
		self.disco = disco

	def import_code( self, code: str, filename: Path, scope: str|None = None ) -> Module:
		return self.disco.import_code( code, filename, scope )

	def import_file( self, filename: Path, scope: str|None = None ) -> Module:
		return self.disco.import_file( filename, scope )

if __name__ == '__main__':
	Discovery.log_unhandled = False # enable for discovery debugging

	TROUBLESHOOT_IMPORT = False

	disco = Discovery( import_builtins = not TROUBLESHOOT_IMPORT )
	c = Compiler( disco )

	if TROUBLESHOOT_IMPORT:
		m = c.import_code( '''
#from . import foo
''', Path( '__test__.py' ), scope = 'first.second' )
	
	
	m = c.import_code( '''
x = Foo.foo()

class Foo:
	def foo( self ) -> i32:
		return 0
''', Path( '__main__.py' ), scope = '__main__' )
	print( f'{m.stem=}' )
	print( f'{m.qualname=}' )
	print( f'{m.file=}' )
	print( f'{m.line=}' )
	print( '' )
	print( '' )
	print( 'module names:' )
	for name, obj in m.names.items():
		pending = getattr( obj, 'resolve', None ) is not None
		print( f'	{name!r} -> {obj.stem!r} -> {obj.qualname!r}{" (unresolved)" if pending else ""}' )
