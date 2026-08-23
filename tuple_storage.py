# stdlib imports:
import ast
from typing import Callable

# local imports:
from discovery import Discovery
from mpy_types import RCClass, Specialization, TupleType, Variable

'''
Synthesizes and memoizes the real runtime representation of every distinct
tuple[T0, T1, ..., Tn] (see PLAN_TUPLE.md) - split out of lowering.py the
same way union_storage.py's UnionStorage is, and for the same reason (no
dependency on statement/expression lowering or CFG state, only on Discovery
for scheduling).

Unlike UnionStorage (one shared tag/data shape per TaggedUnion, itself a
real, already-existing ClassLike parsed from source), a TupleType has no
existing class to attach a synthesized representation to - it's recognized
textually in discovery.py's visit_Subscript (see PLAN_TUPLE.md's own
"Precedent reused" section), so this module builds the backing RCClass from
nothing: fields _0.._n, one per elem_types[i], no synthesized __init__ at
all - tuple literals build one directly via ir.Allocate's own field=value
shape (lowering.py's _expr_Tuple), the exact same "no-__init__ degrades to
field=value sugar" convention _lower_allocate_fields already applies to any
class with no real __init__, and the same direct-Allocate shape
_lower_bound_method_closure already uses to build a ClosureType value with
no synthesized __init__ of its own either. TupleStorage.get() ALSO now
synthesizes real ast.FunctionDef bodies (unlike a union member's constructor,
built entirely from mpy_types objects with no AST at all) for HOMOGENEOUS
tuples only (every elem_types entry equal): __getitem__/__iter__, declaring
Sequence[T]/Iterable[T] conformance - see get()'s own comment for why a
source-text-then-ast.parse approach was used instead of hand-building nodes.
'''

class TupleStorage:
	'''
	Builds and memoizes the backing RCClass for every distinct TupleType, on
	first use - see get()'s own docstring. Persists for the whole Lowering/
	TypeResolver instance's lifetime (one TupleStorage per compile run,
	shared between them - see type_resolver.py's own construction, mirrors
	union_storage/monomorphizer), unlike FunctionLowering's per-function
	state.
	'''

	def __init__( self, discovery: Discovery, schedule: Callable[[object],None] ) -> None:
		self.discovery = discovery
		self.schedule = schedule
		# reverse lookup (backing RCClass identity -> the TupleType it backs)
		# - lowering.py's _expr_Subscript needs this to recognize "obj.type is
		# a tuple-backed RCClass" and recover elem_types for bounds-checking a
		# constant index, without every RCClass needing a field of its own
		# just to carry this
		self._tuple_type_by_backing: dict[int,TupleType] = {}

	def get( self, tt: TupleType ) -> RCClass:
		''' every distinct TupleType (interned by discovery.py's
		_get_or_create_tuple_type, keyed on its own elem_types) gets a real
		backing RCClass synthesized here on first use: fields _0.._n, one per
		elem_types[i], named/typed exactly like an ordinary hand-declared
		class attribute (see mpy_types.Variable) - every existing RC
		mechanism (cfg.py's is_rc/rc_leaves, type_resolver.py's
		_synthesize_rcclass_destructor, emitter_c.py's emit_rcclass) applies
		to it completely unchanged, no special-casing needed anywhere past
		this synthesis. Memoized on TupleType.backing itself (a self-caching
		slot, same spirit as Specialization.monomorphized - see
		mpy_types.py's own comment on that field) rather than a separate
		id(tt)-keyed side table, since a TupleType is already the canonical,
		interned object for its own elem_types (two annotations spelling the
		same element list share one TupleType, and therefore one backing
		class here too - never a fresh one per occurrence). '''
		if tt.backing is not None:
			return tt.backing
		# qualname mirrors how a real generic Specialization spells its own
		# qualname (discovery.py's _get_or_create_specialization: f'{base.
		# qualname}[{args}]', e.g. 'builtins.Result[builtins.i32,builtins.
		# IntError]') - emitter_c.py's mangle_qualname/mangle_type already
		# turn '.', '[', ',', ']' into legal C identifier fragments generically
		# (a flat string transform, not a structural parse), so this needs no
		# emitter-side special case at all, the same way a nested generic
		# element type (tuple[list[i32], str]) mangles correctly today with
		# zero extra work
		qualname = f'tuple[{",".join( t.qualname for t in tt.elem_types )}]'
		# NOT None: type_resolver.py's _synthesize_rcclass_destructor copies
		# cls.file/.line straight onto the synthesized $$__destructor__
		# Function it builds for every concrete RCClass, and lowering.py's
		# _find_module_for looks up the owning module by matching that
		# file exactly - a real RCClass (unlike UnionStorage's own
		# synthesized payload CUnion/anonymous TaggedUnion, neither of
		# which ever needs a destructor looked up this way) can't get away
		# with file=None here. module_stack[-1] mirrors discovery.py's own
		# _get_or_create_closure_type - "whichever module is currently
		# active" is always a real, valid module at every call site this
		# reaches (TypeResolver.ensure_resolved / lowering.py's _expr_Tuple,
		# both always inside an active module_context)
		file = self.discovery.module_stack[-1].file if self.discovery.module_stack else None
		line = self.discovery.module_stack[-1].line if self.discovery.module_stack else None
		attributes = [
			Variable( stem = f'_{i}', qualname = f'{qualname}._{i}', file = file, line = line, type = elem_type )
			for i, elem_type in enumerate( tt.elem_types )
		]
		backing = RCClass(
			stem = qualname,
			qualname = qualname,
			file = file,
			line = line,
			base = None,
			type_params = None,
			attributes = attributes,
			methods = [],
			names = { attr.stem: attr for attr in attributes },
			resolve = None,
		)
		tt.backing = backing
		self._tuple_type_by_backing[ id( backing ) ] = tt
		self._declare_sequence_conformance( tt, backing )
		# scheduled here (not left for lowering.py's own construction-site
		# scheduling to discover) so a tuple[...] type reached ONLY through
		# an annotation - a parameter/return type/field that's never locally
		# constructed in THIS particular function - still becomes a real,
		# emitted struct, exactly like UnionStorage.get()'s own identical
		# schedule(payload_cls) call for the same reason
		self.schedule( backing )
		return backing

	def _declare_sequence_conformance( self, tt: TupleType, backing: RCClass ) -> None:
		''' HOMOGENEOUS tuples only (every elem_types entry the same type) -
		e.g. tuple[i32,i32,i32] conforms to Sequence[i32]/Iterable[i32];
		tuple[i32,str] (heterogeneous - no single element type) conforms to
		neither, and gets no methods synthesized here at all. Compared by
		qualname, not `==`/`is` - two annotation occurrences of "the same"
		type aren't always identity-equal (see TypeResolver._same_type's own
		docstring) and a bare dataclass `==` walks the whole object
		structurally, which is unsafe/wrong for a Type (same reasoning
		TypeVar.bound_satisfied_by's own comment gives).

		Building real ast.FunctionDef bodies via source text + ast.parse,
		not hand-built ast.* node graphs (unlike e.g. type_resolver.py's
		_synthesize_rcclass_destructor) - __getitem__'s own if/return chain
		(one branch per tuple index) is far simpler to express as a small
		Python template than to hand-assemble node-by-node, and the element
		type / this tuple's own backing class (neither has a plain,
		reliably-resolvable-by-name spelling from an arbitrary calling
		module) are attached directly via node.resolved_type - the same
		"synthesized code names a concrete Type object directly, bypassing
		ordinary scope resolution" escape hatch discovery.py's own
		visit_Name already documents. '''
		if not tt.elem_types:
			return
		elem_type = tt.elem_types[0]
		if any( t.qualname != elem_type.qualname for t in tt.elem_types ):
			return
		sequence_protocol = self.discovery.find_name_or_none( 'Sequence' )
		iterable_protocol = self.discovery.find_name_or_none( 'Iterable' )
		if sequence_protocol is None or iterable_protocol is None:
			# builtins.Sequence/Iterable genuinely don't exist in every
			# compile (e.g. import_builtins=False test fixtures) - a
			# homogeneous tuple simply doesn't conform to anything in that
			# world, same as it never did before this feature existed; only
			# __getitem__(usize)/self._N-style constant-index access
			# (unaffected by any of this) still works.
			return
		index_checks = '\n'.join(
			f'\tif i == {i}:\n\t\treturn Result.Ok( self._{i} )' for i in range( len( tt.elem_types ))
		)
		src = (
			f'def __getitem__( self, i: usize ) -> Result[__ElemT__, IndexError]:\n'
			f'{index_checks}\n'
			f'\treturn Result.Err( IndexError() )\n'
			f'def __iter__( self ) -> Generator[__ElemT__, StopIteration]:\n'
			f'\treturn _sequence_iter( self )\n' # bare call - T/S both ordinarily inferable (S from self, T reverse-unified through S's own Sequence[T] bound - see Lowering._unify_type_param)
		)
		module_ast = ast.parse( src )
		resolved_by_placeholder = { '__ElemT__': elem_type }
		for node in ast.walk( module_ast ):
			if isinstance( node, ast.Name ) and node.id in resolved_by_placeholder:
				node.resolved_type = resolved_by_placeholder[ node.id ]
		with self.discovery.scope_context( backing ):
			for fn_node in module_ast.body:
				assert isinstance( fn_node, ast.FunctionDef )
				self.discovery._parse_function( fn_node, backing )
		backing.protocols.append( self.discovery._get_or_create_specialization( sequence_protocol, [ elem_type ] ))
		backing.protocols.append( self.discovery._get_or_create_specialization( iterable_protocol, [ elem_type ] ))

	def tuple_type_for( self, cls: object ) -> TupleType|None:
		''' the reverse of get() - given a (already-resolved, concrete)
		type, the TupleType it's the synthesized backing class for, or None
		if it isn't one. Used by lowering.py's _expr_Subscript to recognize
		a tuple-backed receiver and recover elem_types for constant-index
		bounds-checking. '''
		return self._tuple_type_by_backing.get( id( cls ))
