# stdlib imports:
from dataclasses import dataclass
from typing import Callable

# local imports:
from discovery import Discovery
from mpy_types import Variable, Function, TaggedUnion, CUnion

'''
Synthesizes and memoizes the real runtime representation of every TaggedUnion
(a user-declared @union class, or a synthesized anonymous X|Y) - split out of
lowering.py since it has no dependency on statement/expression lowering or
CFG state, only on Discovery (for resolving/scheduling) - see UnionStorage's
own docstring.
'''

@dataclass( kw_only = True )
class ReceiverDispatch:
	''' `Lowering._resolve_callee`'s answer when an attribute call's receiver
	is a union type and the attribute isn't found on the union itself (e.g.
	copy_from.get_const_ptr() where copy_from: bytes|bytearray) - each leaf
	type has its own unrelated method under this name, so unlike Overload
	(one shared Function, resolved by argument types) there's no single
	target Function here at all, just one per leaf, picked by the
	RECEIVER's own runtime tag. See Lowering._lower_union_receiver_call. '''
	union: TaggedUnion
	attr: str
	per_leaf: list[tuple[Variable,Function]] # (union.attributes member, that leaf's resolved method)

class UnionStorage:
	'''
	Builds and memoizes the `tag: u8` / `data: <synthesized CUnion>` runtime
	representation for every TaggedUnion, on first use - see get()'s own
	docstring. Persists for the whole Lowering instance's lifetime (one
	UnionStorage per Lowering), unlike lower_function's per-function state.
	'''

	def __init__( self, discovery: Discovery, schedule: Callable[[object],None] ) -> None:
		self.discovery = discovery
		self.schedule = schedule
		self._cache: dict[int,tuple[Variable,Variable,CUnion,dict[str,int]]] = {}

	def _ensure_resolved( self, obj: object ) -> None:
		# same discipline as Lowering._ensure_resolved (resolve immediately,
		# then unconditionally hand to schedule - see that method's own
		# docstring for why) - duplicated here rather than depending back on
		# Lowering, since this class is meant to be usable standalone
		resolve = getattr( obj, 'resolve', None )
		if resolve is not None:
			resolve()
		self.schedule( obj )

	def get( self, union: TaggedUnion ) -> tuple[Variable,Variable,CUnion,dict[str,int]]:
		''' every TaggedUnion (a user-declared @union class, or a synthesized
		anonymous X|Y) gets a real runtime representation synthesized here on
		first use: `tag: u8` (each member's ordinal, by declaration order) +
		`data: <synthesized CUnion>` (one v_<member>-prefixed field per
		member, only one ever meaningfully set at a time - the v_ prefix
		avoids a member name colliding with something else in that payload
		struct). builtins.Result is itself an ordinary @union (Ok/Err
		members) and goes through this exact same path - only or_return()
		stays specially recognized (see Lowering._lower_or_return's own
		comment on why compiler.early_return couldn't just be reused for it).
		Memoized so every reference (construction, match, dispatch, across
		unrelated functions) points at the same tag/data/payload-class
		objects. '''
		cached = self._cache.get( id( union ) )
		if cached is not None:
			return cached
		self._ensure_resolved( union )
		for attr in union.attributes:
			self._ensure_resolved( attr )
		u8_cls = self.discovery.get_intrinsics()['u8']
		tag_attr = Variable( stem = 'tag', qualname = f'{union.qualname}.tag', file = union.file, line = union.line, type = u8_cls )
		payload_fields = [
			Variable( stem = f'v_{attr.stem}', qualname = f'{union.qualname}.data.v_{attr.stem}', file = attr.file, line = attr.line, type = attr.type )
			for attr in union.attributes
		]
		payload_cls = CUnion(
			stem = f'{union.stem}$data',
			qualname = f'{union.qualname}$data',
			file = union.file,
			line = union.line,
			attributes = payload_fields,
			names = { f.stem: f for f in payload_fields },
		)
		data_attr = Variable( stem = 'data', qualname = f'{union.qualname}.data', file = union.file, line = union.line, type = payload_cls )
		# register into union.names (NOT .attributes - that list backs
		# .leaves(), which must still only reflect the real union members
		# for overload/type matching) so ordinary GetAttr resolution
		# (Lowering._attr_lookup, used by _expr_Attribute for synthesized
		# `subj.tag`/`subj.data` AST) can actually find them
		for synthesized in ( tag_attr, data_attr ):
			existing = union.names.get( synthesized.stem )
			if existing is not None and existing is not synthesized:
				self.discovery.fail_loc(
					f'{union.qualname} already declares a member named {synthesized.stem!r}, which collides with the compiler-synthesized union storage field of the same name',
					union.file, union.line,
				)
			union.names[synthesized.stem] = synthesized
		tags = { attr.stem: i for i, attr in enumerate( union.attributes ) }
		# payload_cls (the synthesized CUnion backing `data`) needs its own
		# explicit schedule() here - unlike the outer TaggedUnion itself
		# (already scheduled by every caller reaching this point), nothing
		# else would ever schedule payload_cls on its own, since it's never
		# directly named anywhere in user code, only reached through
		# union.names['data'].type - without this, a real TaggedUnion could
		# be scheduled/emitted (the outer struct) while the CUnion its own
		# `data` field embeds BY VALUE never lands in compiler.cunions,
		# leaving that field's type incomplete. Skipped when `union` is
		# itself still generic (type_params set, i.e. this is the abstract
		# base of something like Result[T,E]): payload_fields carry bare
		# TypeVars in that case, not a real emittable C type - only a
		# CONCRETE specialization's own substituted payload_cls (built by
		# Lowering.monomorphize_class) is ever a real compile unit.
		if not union.type_params:
			self.schedule( payload_cls )
		result = ( tag_attr, data_attr, payload_cls, tags )
		self._cache[ id( union ) ] = result
		return result
