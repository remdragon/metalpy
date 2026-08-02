# stdlib imports:
from dataclasses import dataclass
from pathlib import Path
import queue
import threading

# local imports:
import ir
from discovery import Discovery
from errors import CompileError
from lowering import Lowering
from mpy_types import Module, Function, Variable, ClassLike, RCClass, CStruct, CUnion, TaggedUnion, CEnum, Specialization

@dataclass( kw_only = True )
class LoweredFunction:
	function: Function
	instructions: list[ir.Instruction]

@dataclass( kw_only = True )
class LoweredGlobal:
	variable: Variable
	instructions: list[ir.Instruction]

CompileUnit = Function|ClassLike|Variable|Specialization # Specialization only ever wraps a generic Function here - a class Specialization never reaches _lower directly, see _enqueue
CompiledUnit = LoweredFunction|ClassLike|LoweredGlobal

class Compiler:
	'''
	stage 2 driver: starting from Discovery.main, lowers one symbol at a
	time, discovering everything that symbol touches and scheduling anything
	not yet lowered, until nothing is left. See ARCHITECTURE.md lines 118-138.

	Scheduling goes through a stdlib queue.Queue (already thread-safe) even
	though nothing here is threaded yet - run() just drains it with a plain
	get_nowait() loop until queue.Empty.

	_enqueue is the single place that decides what a "dependency" actually
	means - lowering.py hands it anything it comes across (a Function, a
	class, a Variable, a Specialization, even a Module walked mid-namespace-
	lookup) without needing to know which of those are real compile units.
	A Specialization decomposes into its base + each type arg (recursively -
	Result[Result[i32,E1],E2] schedules i32/E1/E2 too); anything that isn't a
	Function, a ClassLike, or a genuinely module-level Variable
	(Variable.is_global - the same class also represents class attributes
	and lowering.py's own local variables, neither a standalone unit) is
	silently ignored rather than enqueued.

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
		self._seen_lock = threading.Lock()

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

	def _enqueue( self, unit: object ) -> None:
		if isinstance( unit, Specialization ):
			if isinstance( unit.base, Function ):
				# an explicit generic function instantiation (sys.alloc[u8])
				# monomorphizes - unlike a class Specialization (whose
				# methods stay shared/unspecialized, so only the base class
				# + type args need scheduling), this IS a real, distinct
				# compile unit in its own right: queued directly rather
				# than decomposed
				with self._seen_lock:
					if id( unit ) in self._seen:
						return
					self._seen.add( id( unit ))
				self.queue.put( unit )
				return
			self._enqueue( unit.base )
			for arg in unit.args:
				self._enqueue( arg )
			return
		if not isinstance( unit, ( Function, ClassLike )) and not ( isinstance( unit, Variable ) and unit.is_global ):
			# not a real compile unit: a Module (walked mid-namespace-lookup,
			# e.g. the `sys` in `sys.alloc(...)`), a class field/parameter/
			# local Variable, a bare Scalar/TypeVar, an Overload group itself
			# (only a resolved member is ever actually compiled) - all
			# harmless to just drop here rather than every caller having to
			# know not to pass them in the first place
			return
		with self._seen_lock:
			if id( unit ) in self._seen:
				return
			self._seen.add( id( unit ))
		self.queue.put( unit )

	def run( self ) -> None:
		if self.disco.main is None:
			self.disco.errors.error( 'no main() found', file = None, line = None )
			return
		self._enqueue( self.disco.main )
		while True:
			try:
				unit = self.queue.get_nowait()
			except queue.Empty:
				break
			# one broken symbol doesn't stop the rest of the work queue from
			# draining - mirrors the recovery boundaries in discovery.py/lowering.py
			try:
				self._lower( unit )
			except CompileError:
				continue

	def _lower( self, unit: CompileUnit ) -> CompiledUnit:
		if isinstance( unit, Specialization ) and isinstance( unit.base, Function ):
			monomorphized, instructions = self.lowering.lower_function_specialization( unit )
			lf = LoweredFunction( function = monomorphized, instructions = instructions )
			self.functions.append( lf )
			return lf
		elif isinstance( unit, Function ):
			if unit.resolve is not None:
				unit.resolve()
			instructions = self.lowering.lower_function( unit )
			lf = LoweredFunction( function = unit, instructions = instructions )
			self.functions.append( lf )
			return lf
		elif isinstance( unit, RCClass ):
			if unit.resolve is not None:
				unit.resolve()
			if unit.base is not None:
				self._enqueue( unit.base )
			self.rcclasses.append( unit )
			return unit
		elif isinstance( unit, CStruct ):
			if unit.resolve is not None:
				unit.resolve()
			self.cstructs.append( unit )
			return unit
		elif isinstance( unit, CUnion ):
			if unit.resolve is not None:
				unit.resolve()
			self.cunions.append( unit )
			return unit
		elif isinstance( unit, TaggedUnion ):
			if unit.resolve is not None:
				unit.resolve()
			self.tagged_unions.append( unit )
			return unit
		elif isinstance( unit, CEnum ):
			if unit.resolve is not None:
				unit.resolve()
			self.cenums.append( unit )
			return unit
		elif isinstance( unit, Variable ):
			if unit.resolve is not None:
				unit.resolve()
			instructions = self.lowering.lower_global( unit )
			lg = LoweredGlobal( variable = unit, instructions = instructions )
			self.globals.append( lg )
			return lg
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
