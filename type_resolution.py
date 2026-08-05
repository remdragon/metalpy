# stdlib imports:
import queue
import threading

# local imports:
from discovery import Discovery
from monomorphize import Monomorphizer
from mpy_types import (
	CEnum, ClassLike, CStruct, CUnion, Function, RCClass, Specialization, TaggedUnion, Variable,
)
from union_storage import UnionStorage

class TypeResolver:
	'''
	stage 1.5: sits between discovery.py (lazy name-binding + skeleton type
	registry) and lowering.py (pure IR emission). Owns the reachable-from-
	main work queue - previously Compiler._enqueue/.queue/._seen - and the
	shared UnionStorage/Monomorphizer instances - previously constructed
	inside Lowering.__init__ (both already depended on nothing but Discovery
	and an injected `schedule` callback, so moving them here is a pure
	relocation, not a rewrite - see union_storage.py/monomorphize.py's own
	docstrings, and monomorphize_test.py's "never lowering.py itself" test
	discipline, which this class follows too).

	`schedule(...)` is still "the single place that decides what a
	dependency actually means" (see its own docstring below, moved verbatim
	from Compiler._enqueue) - lowering.py hands it anything it comes across
	unconditionally, same as always. Compiler still owns the per-kind
	lowering dispatch and the compiled-object collections (functions/
	rcclasses/...) - this class only decides what's reachable and makes
	sure it's resolved by the time something asks for it.
	'''
	def __init__( self, discovery: Discovery ) -> None:
		self.discovery = discovery
		self.queue: queue.Queue = queue.Queue()
		self._seen: set[int] = set()
		self._seen_lock = threading.Lock()
		self.union_storage = UnionStorage( discovery, self.schedule )
		self.monomorphizer = Monomorphizer( discovery, self.schedule, self.union_storage )

	def schedule( self, unit: object ) -> None:
		# moved verbatim from Compiler._enqueue - lowering.py hands this
		# anything it comes across (a Function, a class, a Variable, a
		# Specialization, even a Module walked mid-namespace-lookup)
		# without needing to know which of those are real compile units.
		# A Specialization decomposes into its base + each type arg
		# (recursively - Result[Result[i32,E1],E2] schedules i32/E1/E2 too);
		# anything that isn't a Function, a ClassLike, or a genuinely
		# module-level Variable (Variable.is_global - the same class also
		# represents class attributes and lowering.py's own local
		# variables, neither a standalone unit) is silently ignored rather
		# than enqueued
		if isinstance( unit, Specialization ):
			if isinstance( unit.base, Function ):
				# an explicit generic function instantiation (sys.alloc[u8])
				# monomorphizes - a real, distinct compile unit in its own
				# right: queued directly rather than decomposed
				with self._seen_lock:
					if id( unit ) in self._seen:
						return
					self._seen.add( id( unit ))
				self.queue.put( unit )
				return
			if isinstance( unit.base, ( RCClass, CStruct, CUnion, TaggedUnion )):
				# a concrete generic CLASS specialization (Result[i32,
				# OverflowError]) - also queued directly (mirroring the
				# Function-based branch above), giving it its own real
				# struct/union layout via Lowering.monomorphize_class (see
				# Compiler._lower), so it lands in compiler.cstructs/.cunions/
				# .tagged_unions/.rcclasses just like any other compile
				# unit - a future emitter never has to independently
				# rediscover/resynthesize a concrete specialization itself.
				# Unlike the Function case, its own concrete type ARGS are
				# also independently scheduled here: a generic function's
				# args get scheduled incidentally via its own body/
				# signature (_emit_generic_call's explicit schedule()
				# calls), but a class specialization's substituted field
				# types (Result[i32,OverflowError]'s own OverflowError, for
				# instance) have no equivalent "body" to walk for that
				with self._seen_lock:
					if id( unit ) in self._seen:
						return
					self._seen.add( id( unit ))
				self.queue.put( unit )
				for arg in unit.args:
					self.schedule( arg )
				return
			self.schedule( unit.base )
			for arg in unit.args:
				self.schedule( arg )
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

	def next_unit( self ) -> object|None:
		try:
			return self.queue.get_nowait()
		except queue.Empty:
			return None

	def ensure_resolved( self, obj: object ) -> object:
		# moved verbatim from Lowering._ensure_resolved - resolving
		# (populating .names/.parameters/whatever) needs to happen
		# immediately, mid-statement, for whoever's asking - unlike
		# schedule(), which just queues obj for whenever the work queue
		# gets to it, this can't wait.
		#
		# unconditionally hands obj to schedule() too - schedule() is the
		# single place that judges what's actually a compile unit worth
		# queuing, what decomposes into more of those, and what to just
		# quietly ignore. Nothing here needs to know or duplicate that
		# judgment.
		#
		# the SINGLE place a Specialization gets swapped for the real,
		# substituted thing it stands in for: every caller MUST use the
		# returned value, not the object passed in, or they see the
		# abstract, unsubstituted base instead (Specialization.names/.resolve
		# are raw passthroughs to it - see mpy_types.py). schedule() still
		# gets the ORIGINAL Specialization (schedule() dispatches on
		# isinstance(unit, Specialization) to actually build/register the
		# real compile unit) - only the return value here is swapped
		resolve = getattr( obj, 'resolve', None )
		if resolve is not None:
			resolve()
		self.schedule( obj )
		if isinstance( obj, Specialization ):
			if isinstance( obj.base, Function ):
				return self.monomorphizer.monomorphized_function( obj )
			if isinstance( obj.base, ( RCClass, CStruct, CUnion, TaggedUnion, CEnum )):
				return self.monomorphizer.monomorphize_class( obj )
			# Scalar (Ptr[T]/ConstPtr[T], the intrinsic generic-pointer
			# scalars - see mpy_types.py's Scalar) has no monomorphization
			# support at all - .names/.resolve stay raw passthroughs to the
			# abstract base, same as always
		return obj
