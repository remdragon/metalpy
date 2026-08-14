# stdlib imports:
from dataclasses import dataclass
from pathlib import Path
import queue

# local imports:
import ir
from discovery import Discovery, is_stub_body
from errors import CompileError
from lowering import Lowering
from mpy_types import Module, Function, Overload, Variable, ClassLike, RCClass, CStruct, CUnion, TaggedUnion, CEnum, Specialization
from type_resolver import TypeResolver

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
	time, until type_resolver's work queue is empty. See ARCHITECTURE.md
	lines 118-138.

	The work queue itself (a stdlib queue.Queue, already thread-safe even
	though nothing here is threaded yet) and the "what's actually a
	dependency worth scheduling" judgment both live on type_resolver
	(type_resolver.py) now - lowering.py hands it anything it comes
	across (a Function, a class, a Variable, a Specialization, even a
	Module walked mid-namespace-lookup) without needing to know which of
	those are real compile units; see TypeResolver.schedule's own
	docstring. `queue`/`_enqueue` here are thin delegates kept for existing
	callers/tests.

	Compiled objects are organized by concrete kind (functions, rcclasses,
	cstructs, ...) as they're lowered, rather than being collated after the
	fact - stage 3's emitter (which has to output things in a certain order,
	e.g. all class definitions before the functions that use them) can just
	consume these lists directly.
	'''
	def __init__( self, disco: Discovery ) -> None:
		self.disco = disco
		# TypeResolver (type_resolver.py) owns the reachable-from-main
		# work queue (previously built directly here) and the shared
		# UnionStorage/Monomorphizer instances (previously built inside
		# Lowering.__init__) - see its own docstring
		self.type_resolver = TypeResolver( disco )
		self.lowering = Lowering( disco, self.type_resolver )
		# lets a synthesized function's body be lowered EAGERLY, synchronously,
		# at the call site that needs its result right away - instead of only
		# ever being scheduled onto the work queue for later - see
		# _expr_Lambda's eager return-type inference (PLAN_LAMBDA.md)
		self.lowering._compile_now = self._lower

		self.functions: list[LoweredFunction] = []
		self.rcclasses: list[RCClass] = []
		self.cstructs: list[CStruct] = []
		self.cunions: list[CUnion] = []
		self.tagged_unions: list[TaggedUnion] = []
		self.cenums: list[CEnum] = []
		self.globals: list[LoweredGlobal] = []
		# @extern('lib', 'symbol') dependencies, registered as each extern
		# function is lowered (see _lower's Function branch below) - lib
		# name -> the symbols pulled from it. 'c' means the platform C
		# runtime specifically, not a real .lib/.so on disk (see
		# mpy_types.Function.extern_lib) - a future emitter/linker's call
		# on what to do with that, not this registry's
		self.extern_libs: dict[str,set[str]] = {}

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

	@property
	def queue( self ) -> queue.Queue:
		# thin delegate for existing white-box tests (compiler_test.py's
		# EnqueueFilteringTests) that read the queue directly - the real
		# queue now lives on type_resolver, see its own docstring
		return self.type_resolver.queue

	def _trigger_name( self, unit: CompileUnit ) -> str:
		suffix = ''
		if isinstance( unit, Specialization ):
			suffix = f'[{", ".join( a.qualname for a in unit.args if hasattr( a, "qualname" ))}]'
			unit = unit.base
		# NOT getattr(unit, 'qualname', str(unit)) - Python eagerly evaluates
		# a getattr() default argument regardless of whether the attribute
		# exists, so str(unit) (a dataclass's auto-generated __repr__, which
		# walks every field with no cycle detection) ran unconditionally for
		# EVERY unit here, not just the rare one actually missing .qualname.
		# Harmless most of the time, but a genuine hang the moment any
		# reachable object graph has a real cycle (confirmed: list[str]
		# triggers one - Function.cls <-> its owning class's own .methods
		# list, or similar back-reference, recursing forever through repr).
		qualname = getattr( unit, 'qualname', None )
		return ( qualname if qualname is not None else str( unit ) ) + suffix

	def _enqueue( self, unit: object ) -> None:
		# thin delegate, kept for existing white-box tests and the two
		# internal call sites below - the real scheduling logic ("what's a
		# dependency" judgment) moved to TypeResolver.schedule, see its own
		# docstring
		self.type_resolver.schedule( unit )

	def run( self ) -> None:
		if self.disco.main is None:
			self.disco.errors.error( 'no main() found', file = None, line = None )
			return
		self._enqueue( self.disco.main )
		while True:
			unit = self.type_resolver.next_unit()
			if unit is None:
				break
			# record what's currently being lowered so schedule() can
			# attribute newly-discovered dependencies to this unit
			self.type_resolver._current_trigger = self._trigger_name( unit )
			# one broken symbol doesn't stop the rest of the work queue from
			# draining - mirrors the recovery boundaries in discovery.py/lowering.py
			try:
				self._lower( unit )
			except CompileError:
				continue

	def _lower( self, unit: CompileUnit ) -> CompiledUnit:
		if isinstance( unit, Specialization ) and isinstance( unit.base, Function ):
			if unit.base.resolve is not None:
				unit.base.resolve()
			self.type_resolver.resolve_function_body( unit.base ) # rewrites 1/2 against the abstract, shared-until-now body - see resolve_function_body's own docstring
			monomorphized = self.type_resolver.ensure_resolved( unit ) # swaps the Specialization for its real, substituted Function - own deep-copied body (see Monomorphizer.monomorphized_function)
			self.type_resolver.resolve_function_body( monomorphized ) # rewrite 3 (generic-call resolution) against THIS copy's own body, now that its own type params are concretely bound
			instructions = self.lowering.lower_function( monomorphized )
			lf = LoweredFunction( function = monomorphized, instructions = instructions )
			self.functions.append( lf )
			return lf
		elif isinstance( unit, Specialization ) and isinstance( unit.base, ( RCClass, CStruct, CUnion, TaggedUnion )):
			monomorphized = self.lowering.monomorphize_class( unit )
			if isinstance( monomorphized, RCClass ):
				if monomorphized.base is not None:
					self._enqueue( monomorphized.base )
				if monomorphized not in self.rcclasses:
					self.rcclasses.append( monomorphized )
				self.type_resolver._synthesize_rcclass_destructor( monomorphized )
				self._validate_interface_vtable( monomorphized )
				self._schedule_rcclass_vtable_impls( monomorphized )
			elif isinstance( monomorphized, CStruct ):
				if monomorphized.base is not None:
					self._enqueue( monomorphized.base )
				if monomorphized.is_interface:
					self._validate_interface_vtable( monomorphized )
					self._schedule_interface_vtable_impls( monomorphized )
				if monomorphized not in self.cstructs:
					self.cstructs.append( monomorphized )
			elif isinstance( monomorphized, CUnion ):
				if monomorphized not in self.cunions:
					self.cunions.append( monomorphized )
			elif isinstance( monomorphized, TaggedUnion ):
				if monomorphized not in self.tagged_unions:
					self.tagged_unions.append( monomorphized )
			return monomorphized
		elif isinstance( unit, Function ):
			if unit.resolve is not None:
				unit.resolve()
			self.type_resolver.resolve_function_body( unit )
			instructions = self.lowering.lower_function( unit )
			if unit.extern_lib is not None:
				self.extern_libs.setdefault( unit.extern_lib, set() ).add( unit.extern_symbol )
			lf = LoweredFunction( function = unit, instructions = instructions )
			self.functions.append( lf )
			return lf
		elif isinstance( unit, RCClass ):
			if unit.resolve is not None:
				unit.resolve()
			for attr in unit.attributes: # each field's own .type is lazily resolved, separate from the class itself - same as Lowering._lower_allocate_fields's identical loop; monomorphize_class already does this for the Specialization branch above, but a bare (non-generic) class landing here directly never went through that
				self.lowering._ensure_resolved( attr )
			if unit.base is not None:
				self._enqueue( unit.base )
			if unit not in self.rcclasses:
				self.rcclasses.append( unit )
				self.type_resolver._synthesize_rcclass_destructor( unit )
			self._validate_interface_vtable( unit )
			self._schedule_rcclass_vtable_impls( unit )
			return unit
		elif isinstance( unit, CStruct ):
			if unit.resolve is not None:
				unit.resolve()
			for attr in unit.attributes:
				self.lowering._ensure_resolved( attr )
			if unit.base is not None: # @interface subclass - base interface needs to be a real compile unit too (its Vtbl type is what $vtable actually points to), same as RCClass.base above
				self._enqueue( unit.base )
			if unit.is_interface:
				self._validate_interface_vtable( unit )
				self._schedule_interface_vtable_impls( unit )
			if unit not in self.cstructs:
				self.cstructs.append( unit )
			return unit
		elif isinstance( unit, CUnion ):
			if unit.resolve is not None:
				unit.resolve()
			for attr in unit.attributes:
				self.lowering._ensure_resolved( attr )
			if unit not in self.cunions:
				self.cunions.append( unit )
			return unit
		elif isinstance( unit, TaggedUnion ):
			if unit.resolve is not None:
				unit.resolve()
			for attr in unit.attributes:
				self.lowering._ensure_resolved( attr )
			if unit not in self.tagged_unions:
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
			unit.init_instructions = instructions # same list object as LoweredGlobal.instructions below - no duplication, no drift risk (see PLAN_GLOBAL_INIT.md)
			lg = LoweredGlobal( variable = unit, instructions = instructions )
			self.globals.append( lg )
			return lg
		else:
			assert False, f'unsupported compile unit: {unit!r}'

	def _validate_interface_vtable( self, cls: RCClass|CStruct ) -> None:
		''' every @virtual method on an @interface CStruct (or, generalized
		for RCClass single inheritance - the RCClass-subclassing plan's own
		Phase 4 - an ordinary RCClass with @virtual methods) is either a
		genuinely NEW slot (its name isn't already a slot anywhere in
		cls's own ancestor chain - always allowed, any level can
		introduce new capabilities now, see CStruct.vtbl_owner's own
		docstring on why the original root-only rule was replaced) or an
		OVERRIDE (name collides with an inherited slot - must strict-
		signature-match it, no covariance/contravariance). Runs once per
		real (non-generic) @interface CStruct compile unit, here rather
		than discovery.py, because checking an override's signature needs
		the OVERRIDDEN slot's own parameters/return_type already resolved,
		which isn't guaranteed yet at discovery-time parse order (a
		subclass can be parsed before its base's own methods are
		individually resolved). Also doubles as the ONE place every
		@virtual method on cls is guaranteed to get its own .resolve()
		called even if nothing else in the program ever calls it - a
		virtual method's own resolver is what runs the single-signature-
		only check (discovery.py's _make_function_resolver), so leaving a
		never-called one permanently unresolved would silently skip that
		check for genuinely dead code. This is why the per-method resolve
		loop below runs UNCONDITIONALLY, even when cls.base is None (a root
		class has nothing to validate an override AGAINST, but its own
		@virtual methods still need resolving for this reason alone).

		Walks cls.methods FLATTENED through any Overload group, not just
		cls.methods' own top-level entries - a name with more than one
		signature (whether @overload-decorated or not) is stored as ONE
		Overload group object in cls.methods, never as the individual
		Function objects directly (see discovery.py's _parse_function) -
		without flattening, a @virtual method sharing its name with a
		sibling def would never even be SEEN here, let alone resolved,
		silently skipping both the override-collision check below and the
		single-signature check inside its own resolver. '''
		ancestor_slots = { m.stem: m for m in cls.base.virtual_slots() } if cls.base is not None else {}
		members: list[Function] = []
		for m in cls.methods:
			if isinstance( m, Function ):
				members.append( m )
			elif isinstance( m, Overload ):
				members.extend( m.stubs )
				members.extend( m.implementations )
		for m in members:
			if not m.is_virtual:
				continue
			if m.resolve is not None:
				m.resolve()
			slot = ancestor_slots.get( m.stem )
			if slot is None:
				continue # a genuinely new slot - always fine now
			if slot.resolve is not None:
				slot.resolve()
			if not self._virtual_signatures_match( m, slot ):
				self.disco.fail(
					f"{m.qualname}: @virtual override does not match {slot.qualname}'s signature "
					f'(strict signature matching required - no covariance/contravariance)',
					m.node,
				)

	def _virtual_signatures_match( self, a: Function, b: Function ) -> bool:
		if a.return_type is not b.return_type:
			return False
		a_params = a.parameters or []
		b_params = b.parameters or []
		if len( a_params ) != len( b_params ):
			return False
		return all( ap.type is bp.type for ap, bp in zip( a_params, b_params ))

	def _schedule_interface_vtable_impls( self, cls: CStruct ) -> None:
		# every slot's ACTUAL implementing Function (found by walking cls's
		# own chain, nearest override wins) needs to be a real, lowered
		# compile unit - it might only ever be reached through vtable
		# dispatch (emitter_c.py's own static vtable instance references it
		# by mangled name directly, never through an ordinary call site the
		# ordinary schedule()-on-reference path would already have caught).
		# Skipped entirely if any slot is unfulfilled (a stub body,
		# lowering.py's own construction-time check already rejects
		# actually building one) - matches emitter_c.py's own identical
		# "None if any slot is unfulfilled" gate for whether to emit a
		# static vtable instance for this class at all, so the two stay in
		# lockstep: this schedules exactly the set of functions that
		# emission will end up referencing by name.
		impls: list[Function] = []
		for slot in cls.virtual_slots():
			impl = cls.chain_lookup( slot.stem )
			if not isinstance( impl, Function ):
				return
			if impl.resolve is not None:
				impl.resolve()
			if is_stub_body( impl.node.body ):
				return
			impls.append( impl )
		for impl in impls:
			self._enqueue( impl )

	def _schedule_rcclass_vtable_impls( self, cls: RCClass ) -> None:
		''' RCClass analog of _schedule_interface_vtable_impls - every REAL
		@virtual slot's own implementing Function (cls.chain_lookup,
		nearest override wins) needs to be a real, lowered compile unit,
		since it might only ever be reached through vtable dispatch
		(emitter_c.py's own static vtable instance references it by
		mangled name directly, never through an ordinary call site the
		usual schedule()-on-reference path would already have caught).
		Skipped entirely if any slot is unfulfilled (@abstractmethod, a
		real explicit marker - RCClass-subclassing plan Phase 5) - matches
		emitter_c.py's own identical "None if any slot is unfulfilled"
		gate for whether emit_rcclass_vtable_instance builds a static
		instance for cls at all, so the two stay in lockstep: this
		schedules exactly the set of functions emission will end up
		referencing by name. A no-op when cls has no @virtual methods
		anywhere in its chain (virtual_slots() is empty). '''
		impls: list[Function] = []
		for slot in cls.virtual_slots():
			impl = cls.chain_lookup( slot.stem )
			assert isinstance( impl, Function ) # virtual_slots()'s own entries always exist somewhere in the chain - at minimum the root's own declaration chain_lookup started from
			if impl.resolve is not None:
				impl.resolve()
			if impl.is_abstract:
				return
			impls.append( impl )
		for impl in impls:
			self._enqueue( impl )

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
