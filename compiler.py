# stdlib imports:
from dataclasses import dataclass
from pathlib import Path
import queue

# local imports:
import ir
from discovery import Discovery, is_stub_body
from errors import CompileError, RedundantCompilationError
from lowering import Lowering
from mpy_types import Module, Function, Overload, Variable, ClassLike, RCClass, CStruct, CUnion, TaggedUnion, CEnum, Specialization, by_value_dependency
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
		# id(Function) -> its own already-built LoweredFunction - guards
		# against the SAME underlying Function object being lowered+
		# emitted twice through two different schedule()-tracked unit
		# shapes (a Specialization wrapper vs the bare, already-
		# monomorphized Function) - see _lower's own comment on this
		self._lowered_functions: dict[int,LoweredFunction] = {}
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
		# runtime DLLs declared via @extern(..., dll='<name>'|[...]),
		# registered the same way and at the same point as extern_libs
		# above - only ever populated from functions that were actually
		# reached/lowered, never a static/declared-anywhere set, so a
		# program that never calls into a given vendored library doesn't
		# get its DLL bundled. Bare filenames (e.g. 'tcl86t.dll'), not
		# paths - mpy.py's post-link bundling step is what turns this into
		# actual file copies. See mpy_types.Function.extern_dlls's own
		# comment for why this is independent from extern_lib (different
		# directories on a real machine, in general) and deliberately not
		# auto-derived from scanning a DLL's own import table.
		self.extern_dlls: set[str] = set()

	def import_code( self, code: str, filename: Path, scope: str|None = None ) -> Module:
		# pass the entry module's own eventual qualname through as `package` so
		# disco.import_code registers it in disco.modules BEFORE scanning its
		# body, same as any nested `import X`/`from X import Y` reaches
		# (discovery.py's own import_code comment on the `package is not None`
		# branch) - without this, a self-import inside the entry module itself
		# (`import foo` written in foo.py, the file being compiled) can't find
		# itself here yet, falls through to a fresh file-system lookup, and
		# re-parses the same source as an independent second Module - real
		# "already defined" collisions for every top-level name, further
		# masked into a mismatched-Module-identity cascade downstream
		# (type_resolver.py's _find_module_for) by self.paths' own unresolved
		# relative '.' entry not matching this file's already-absolute path.
		# Mirrors discovery.py's own non-folding qualname formula - an entry
		# file is never a folding (__init__.py-style) module in practice
		package = f'{scope}.{filename.stem}' if scope else filename.stem
		module = self.disco.import_code( code, filename, scope, package = package )
		# entry modules aren't registered in disco.modules on their own (that's
		# keyed by import package name, for nested imports reached via `import
		# X`) - stage 2 needs to be able to find any module by file (see
		# lowering.py's _find_module_for), including this one
		self.disco.modules[module.qualname] = module
		return module

	def import_file( self, filename: Path, scope: str|None = None ) -> Module:
		package = f'{scope}.{filename.stem}' if scope else filename.stem
		module = self.disco.import_file( filename, scope, package = package )
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
		self._drain()
		if self.disco.active_target['os'] == 'windows':
			# 'c' not in self.extern_libs mirrors mpy.py's own no_crt
			# computation ('c' not in compiler.extern_libs) - captured HERE,
			# right after the user's own program has fully drained (above),
			# before any of the forcing below runs, so it reflects exactly
			# what the user's own program needs. Neither force_reachable
			# call below ever touches the 'c' extern either way (Windows
			# console-codepage/exit both live in kernel32), so this doesn't
			# need to be recomputed between them.
			no_crt = 'c' not in self.extern_libs
			# force windows._console's _console_init global to be reachable on
			# EVERY Windows build - nothing in the user's own program
			# necessarily references it, but its own initializer
			# (SetConsoleOutputCP) must still run before main() does. See
			# windows/_console.py's own comment. Enqueued AFTER main's own
			# graph fully drains (not alongside it above) so this always
			# lands at the END of compiler.globals - tests (and any other
			# compiler.globals[0]-style code) that assume the user's own
			# first-declared global is index 0 stay correct; the real CALL
			# order inside __metalpy_init() is dependency order
			# (_topologically_sort_globals in emitter_c.py), not this
			# scheduling order anyway, so appending here has no effect on
			# correctness - only on this list's own enumeration order.
			self.force_reachable( 'windows._console', '_console_init' )
			if no_crt:
				# emitter_c.py's synthesized mainCRTStartup (no-CRT Windows
				# entry point only) calls sys.exit() directly by its own
				# mangled C symbol name to terminate the process - force it
				# reachable so that call always resolves. Unlike
				# _console_init above, this is only needed when no_crt (a
				# CRT-linked Windows build never emits mainCRTStartup at
				# all), so it's gated separately rather than being forced
				# unconditionally on every Windows target.
				self.force_reachable( 'sys', 'exit' )

	def force_reachable( self, module_qualname: str, attr_name: str ) -> None:
		''' resolves module_qualname.attr_name (a Function or global
		Variable) and forces it onto the work queue + drains, even though
		nothing in the user's own program necessarily references it. Used
		for compiler-synthesized C text (raw, hand-written, outside the
		normal IR pipeline - see emitter_c.py's __metalpy_init/
		mainCRTStartup synthesis) that needs to call/reference a real
		metalpy-level symbol by its own mangled C name. Silently no-ops if
		module_qualname can't be resolved, or doesn't define attr_name -
		some Discovery instances (tests) use a deliberately minimal,
		fixture-only `paths=` that doesn't include the real lib/ tree, or
		swaps in a minimal stand-in module missing this particular symbol
		(e.g. emitter_c_test.py's BuiltinsStrTestCase, whose own sys.py
		fixture has no exit()) - there's nothing useful to force-init in
		either case, so skip it rather than hard-failing every compile
		through that harness. '''
		try:
			mod = self.disco.import_name( module_qualname )
		except FileNotFoundError:
			return
		unit = mod.get_local( attr_name )
		if unit is None or unit.broken:
			return
		self._enqueue( unit )
		self._drain()

	def _drain( self ) -> None:
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
			# a monomorphized generic method can be reached through TWO
			# different unit "shapes" that schedule() (type_resolver.py)
			# tracks as unrelated units - this Specialization wrapper
			# (id(unit), scheduled by e.g. lowering.py's own generic-
			# construction inference) AND the bare, already-monomorphized
			# Function itself (id(monomorphized), scheduled by ordinary
			# method-call lowering against an already-concrete receiver -
			# e.g. a synthesized $$__new__ body calling self.__init__(...)
			# once self's own type is already the concrete monomorphized
			# class, no Specialization needed). schedule()'s own _seen
			# dedup is id-based, so it can't catch this - both make it
			# through independently. Guard on the ACTUAL underlying Function
			# object (id(monomorphized), the thing that would actually get
			# lowered+emitted) rather than id(unit), so either shape
			# reaching here first "wins" and the other is a cheap no-op -
			# confirmed by a real repro: a monomorphized RCClass's own
			# __init__ emitted twice (duplicate C symbol) once a
			# synthesized $$__new__ started calling it via an ordinary
			# self.__init__(...) AST statement instead of raw IR
			cached = self._lowered_functions.get( id( monomorphized ))
			if cached is not None:
				assert isinstance( cached, LoweredFunction )
				return cached
			self.type_resolver.resolve_function_body( monomorphized ) # rewrite 3 (generic-call resolution) against THIS copy's own body, now that its own type params are concretely bound
			instructions = self.lowering.lower_function( monomorphized )
			lf = LoweredFunction( function = monomorphized, instructions = instructions )
			self.functions.append( lf )
			self._lowered_functions[ id( monomorphized ) ] = lf
			return lf
		elif isinstance( unit, Specialization ) and isinstance( unit.base, ( RCClass, CStruct, CUnion, TaggedUnion )):
			monomorphized = self.lowering.monomorphize_class( unit )
			if isinstance( monomorphized, RCClass ):
				if monomorphized.base is not None:
					self._enqueue( monomorphized.base )
				if monomorphized not in self.rcclasses:
					self.rcclasses.append( monomorphized )
				self.type_resolver._synthesize_rcclass_destructor( monomorphized )
				# NOT _synthesize_rcclass_constructor here - unlike the
				# destructor (needed for EVERY RCClass, since any instance,
				# however constructed, might need releasing), $$__new__ is
				# only ever looked up from _try_lower_construct_call's own
				# eager, self-sufficient call (lowering.py), which already
				# guarantees its own availability - triggering it here too,
				# unconditionally for every registered class, synthesizes
				# (and thus references - the header.vtable assignment
				# inside it) a constructor for classes NEVER actually
				# constructed by any reachable user code, e.g. an abstract
				# base only ever used polymorphically through a subclass -
				# confirmed by a real regression: it made emitter_c.py
				# start emitting that abstract base's own vtable instance
				# (a real static object, referenced by the unwanted $$__new__),
				# which a dedicated test asserts must never be emitted
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
			# same cross-shape dedup as the Specialization+Function branch
			# above - a bare Function reached here may be the identical
			# underlying object a Specialization wrapper already lowered
			cached = self._lowered_functions.get( id( unit ))
			if cached is not None:
				assert isinstance( cached, LoweredFunction )
				return cached
			if unit.resolve is not None:
				unit.resolve()
			self.type_resolver.resolve_function_body( unit )
			instructions = self.lowering.lower_function( unit )
			if unit.extern_lib is not None:
				self.extern_libs.setdefault( unit.extern_lib, set() ).add( unit.extern_symbol )
				self.extern_dlls.update( unit.extern_dlls )
			lf = LoweredFunction( function = unit, instructions = instructions )
			self.functions.append( lf )
			self._lowered_functions[ id( unit ) ] = lf
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
				# NOT _synthesize_rcclass_constructor here - see the
				# identical comment on the Specialization+RCClass branch
				# above
			self._validate_interface_vtable( unit )
			self._schedule_rcclass_vtable_impls( unit )
			return unit
		elif isinstance( unit, CStruct ):
			if unit.resolve is not None:
				unit.resolve()
			for attr in unit.attributes:
				self.lowering._ensure_resolved( attr )
				# a by-value-embedded field (e.g. SYSTEMTIME nested inside a
				# larger cstruct) is only reachable THROUGH this attribute -
				# unlike a Function/global Variable, merely resolving attr's
				# own .type never schedules attr.type itself (schedule()
				# ignores class-attribute Variables, is_global=False - see its
				# own comment), so a field type referenced ONLY as another
				# struct's own member, never independently constructed/sized/
				# pointed-to anywhere else in the reachable program, would
				# otherwise never land in compiler.cstructs/cunions/
				# tagged_unions at all. _emit_value_type_bodies's topological
				# sort then has nothing to order it against - not a wrong
				# order, a MISSING definition entirely (confirmed directly: a
				# real clang "field has incomplete type" error, task_421ed8be)
				dep = by_value_dependency( attr.type )
				if dep is not None:
					self.lowering._ensure_resolved( attr.type )
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
				dep = by_value_dependency( attr.type ) # see the identical CStruct branch above for why this is needed
				if dep is not None:
					self.lowering._ensure_resolved( attr.type )
			if unit not in self.cunions:
				self.cunions.append( unit )
			return unit
		elif isinstance( unit, TaggedUnion ):
			if unit.resolve is not None:
				unit.resolve()
			for attr in unit.attributes:
				self.lowering._ensure_resolved( attr )
				dep = by_value_dependency( attr.type ) # see the identical CStruct branch above for why this is needed
				if dep is not None:
					self.lowering._ensure_resolved( attr.type )
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
			if unit.broken:
				# unit's own type-resolution (discovery.py's _make_value_
				# resolver) already failed and recorded the error once -
				# resolve_global_init/lower_global below would independently
				# re-visit the SAME init expression and report the identical
				# failure a second time (see resolve_global_init's own
				# comment, which already silences its OWN half of this exact
				# duplicate but explicitly documents lower_global producing
				# the other half)
				raise RedundantCompilationError()
			# a global's init expression needs the same construction-call
			# pre-resolution an ordinary function body gets from resolve_
			# function_body (below, Function branch) before lowering ever
			# reaches it - see TypeResolver.resolve_global_init's own
			# docstring for the real crash this fixes (a bare ClassName()
			# construction, not a ClassName.factory() call, as a global's
			# own initializer)
			self.type_resolver.resolve_global_init( unit )
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
		# _same_type, not raw `is` - an override's own declared type and its
		# base method's own declared type can be two different objects for
		# the identical type (one eagerly monomorphized via some OTHER call
		# reference resolving it first, the other still a bare
		# Specialization) - same duality TypeResolver._same_type exists to
		# handle elsewhere. Confirmed via a real repro: TypeResolver.
		# resolve_declared_types eagerly monomorphizing a plain declared
		# parameter/return type wherever a Function gets resolved for a
		# real call made an @virtual override's own signature-match check
		# here start seeing false positives, since only ONE side of the
		# comparison (whichever method something else happened to call
		# first) had been through that path by the time this runs.
		if not self.type_resolver._same_type( a.return_type, b.return_type ):
			return False
		a_params = a.parameters or []
		b_params = b.parameters or []
		if len( a_params ) != len( b_params ):
			return False
		return all( self.type_resolver._same_type( ap.type, bp.type ) for ap, bp in zip( a_params, b_params ))

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
