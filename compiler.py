# stdlib imports:
from dataclasses import dataclass
from pathlib import Path
import queue

# local imports:
import ir
from discovery import Discovery
from lowering import Lowering
from mpy_types import Module, Function, Variable, ClassLike, RCClass, CStruct, CUnion, TaggedUnion, CEnum

@dataclass( kw_only = True )
class LoweredFunction:
	function: Function
	instructions: list[ir.Instruction]

@dataclass( kw_only = True )
class LoweredGlobal:
	variable: Variable
	instructions: list[ir.Instruction]

CompileUnit = Function|ClassLike|Variable

class Compiler:
	'''
	stage 2 driver: starting from Discovery.main, lowers one symbol at a
	time, discovering everything that symbol touches and scheduling anything
	not yet lowered, until nothing is left. See ARCHITECTURE.md lines 118-138.

	Scheduling goes through a stdlib queue.Queue (already thread-safe) even
	though nothing here is threaded yet - run() just drains it with a plain
	get_nowait() loop until queue.Empty.

	Compiled objects are organized by concrete kind (functions, rcclasses,
	cstructs, ...) as they're lowered, rather than being collated after the
	fact - stage 3's emitter (which has to output things in a certain order,
	e.g. all class definitions before the functions that use them) can just
	consume these lists directly.
	'''
	def __init__( self, disco: Discovery ) -> None:
		self.disco = disco
		self.lowering = Lowering( disco, schedule = self._enqueue )

		self.queue: queue.Queue = queue.Queue()
		self._seen: set[int] = set()

		self.functions: list[LoweredFunction] = []
		self.rcclasses: list[RCClass] = []
		self.cstructs: list[CStruct] = []
		self.cunions: list[CUnion] = []
		self.tagged_unions: list[TaggedUnion] = []
		self.cenums: list[CEnum] = []
		self.globals: list[LoweredGlobal] = []

	def import_code( self, code: str, filename: Path, scope: str|None = None ) -> Module:
		module = self.disco.import_code( code, filename, scope )
		# entry modules aren't registered in disco.modules on their own (that's
		# keyed by import package name, for nested imports reached via `import
		# X`) - stage 2 needs to be able to find any module by file (see
		# lowering.py's _find_module_for), including this one
		self.disco.modules[module.qualname] = module
		return module

	def import_file( self, filename: Path, scope: str|None = None ) -> Module:
		module = self.disco.import_file( filename, scope )
		self.disco.modules[module.qualname] = module
		return module

	def _enqueue( self, unit: CompileUnit ) -> None:
		if id( unit ) in self._seen:
			return
		self._seen.add( id( unit ))
		self.queue.put( unit )

	def run( self ) -> None:
		assert self.disco.main is not None, 'no main() found'
		self._enqueue( self.disco.main )
		while True:
			try:
				unit = self.queue.get_nowait()
			except queue.Empty:
				break
			self._lower( unit )

	def _lower( self, unit: CompileUnit ) -> None:
		if isinstance( unit, Function ):
			if unit.resolve is not None:
				unit.resolve()
			instructions = self.lowering.lower_function( unit )
			self.functions.append( LoweredFunction( function = unit, instructions = instructions ))
		elif isinstance( unit, RCClass ):
			if unit.resolve is not None:
				unit.resolve()
			if unit.base is not None:
				self._enqueue( unit.base )
			self.rcclasses.append( unit )
		elif isinstance( unit, CStruct ):
			if unit.resolve is not None:
				unit.resolve()
			self.cstructs.append( unit )
		elif isinstance( unit, CUnion ):
			if unit.resolve is not None:
				unit.resolve()
			self.cunions.append( unit )
		elif isinstance( unit, TaggedUnion ):
			if unit.resolve is not None:
				unit.resolve()
			self.tagged_unions.append( unit )
		elif isinstance( unit, CEnum ):
			if unit.resolve is not None:
				unit.resolve()
			self.cenums.append( unit )
		elif isinstance( unit, Variable ):
			instructions = self.lowering.lower_global( unit )
			self.globals.append( LoweredGlobal( variable = unit, instructions = instructions ))
		else:
			assert False, f'unsupported compile unit: {unit!r}'

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
def main() -> None:
	x: i32 = foo( 3 )

def foo( x: i32 ) -> i32:
	return x + 1
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

	print( '' )
	c.run()
	print( 'lowered functions:', [ f.function.qualname for f in c.functions ] )
	print( 'lowered rcclasses:', [ cls.qualname for cls in c.rcclasses ] )
