# stdlib imports:
import ast
from dataclasses import dataclass
from typing import Callable

# local imports:
from discovery import Discovery
from mpy_types import Variable, Parameter, Function, TaggedUnion, CUnion, Type

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

def build_member_constructor(
	union: TaggedUnion, member: Variable, tag_value: int,
	tag_attr: Variable, data_attr: Variable, payload_cls: CUnion, return_type: Type,
	fn_file: object, fn_line: int|None,
) -> Function:
	''' Not private despite historical convention elsewhere in this file -
	monomorphize.py's own monomorphize_class also calls this directly, to
	rebuild a GENERIC union's own per-member constructors for each concrete
	specialization (Result[str,MyError], not bare Result) - see its own
	comment on why a plain field-copy of the abstract base's already-
	synthesized constructors is wrong there (a real bug this shared helper
	fixes, not just refactors: monomorphize_class used to silently clobber
	them back to plain attribute Variables).

	SYNTAX.md: `@union` expands into a struct wrapping a synthesized
	payload CUnion "plus one @staticmethod constructor per variant"
	(`ClassName.Variant(value)`) - synthesized here as a REAL Function with a
	real AST body, registered into union.names[member.stem] (overwriting the
	plain attribute-declaration Variable that's there from ordinary class-
	body parsing - union.attributes, the logical member list .leaves() reads,
	is untouched). This means `UnionName.Member(value)` is just an ordinary
	call to a real staticmethod, resolved/monomorphized through lowering.py's
	EXISTING generic call-lowering path - no union-specific construction code
	needed there at all (this is what replaces the old, textually-recognized
	`_try_lower_union_construct_call`).

	Body: `return UnionName.__allocate__(tag=N, data=PayloadCls(v_Member=
	value))`. The outer `.__allocate__()` is normally private (only callable
	from a method of the same class - see Lowering._try_lower_allocate_call)
	but is satisfied here because this Function's own `.cls` IS `union`. The
	inner payload construction deliberately uses bare `PayloadCls(...)`
	sugar instead of `.__allocate__()` - a CUnion has no `__init__`, so bare
	construction has no privacy restriction at all (see
	Lowering._try_lower_construct_call) - the exact same shape Result.Ok/
	.Err's own hand-written bodies used before `@union` was scoped (see
	TODO.txt).

	`$union_cls`/`$payload_cls` are synthesized names with no real source
	spelling (a `$` can't appear in a real identifier), registered directly
	into this Function's own `.names` so ordinary namespace resolution
	(Lowering._try_resolve_namespace, via Discovery.find_name) finds them on
	the function's own scope without needing either class to have a
	findable name anywhere - the only thing that matters for an ANONYMOUS
	union (a synthesized X|Y, never spelled in source at all).

	`return_type` is `union` itself when non-generic, or a Specialization of
	`union` against its OWN type_params (e.g. Result[T,E], not bare Result)
	when generic - mirroring exactly what a real hand-written `def Ok(value:
	T) -> Result[T,E]:` would declare. This matters: Lowering._lower_class_
	generic_method_call unifies expected_type against target.return_type to
	solve the class's type params (Result.Ok(x) never mentions E in its own
	parameter list at all) - a bare, unparameterized `union` return type has
	no TypeVar anywhere in it for that unification to bind.

	fn_file/fn_line are deliberately NOT union.file/union.line: for a real
	user `@union class Foo:` they're the same thing, but an anonymous
	synthesized union (X|Y) always has file=None (emitter_c.py's
	mangle_type() relies on exactly that to mangle it via the special
	$__u$$... scheme rather than plain mangle_qualname(), which can't
	handle a qualname containing a literal '|') - yet this Function still
	needs a REAL file the moment anything actually calls it (Lowering.
	_find_module_for looks up "which module owns this" by matching .file
	against a real Module's .file, and fails outright on a bare None,
	unlike the tag/data-field-only paths that never needed a real Function
	here at all before Lowering._coerce_into_union). UnionStorage.get()
	passes "whichever module is currently active" (self.discovery.
	module_stack[-1]) for this reason - same fallback
	_get_or_create_closure_type already established for ClosureType's own
	fn/self fields needing __del__ discoverable. '''
	value_param = Parameter(
		stem = 'value', qualname = f'{union.qualname}.{member.stem}.value',
		file = fn_file, line = fn_line, type = member.type,
	)
	payload_call = ast.Call(
		func = ast.Name( id = '$payload_cls', ctx = ast.Load() ),
		args = [],
		keywords = [ ast.keyword( arg = f'v_{member.stem}', value = ast.Name( id = 'value', ctx = ast.Load() )) ],
	)
	allocate_call = ast.Call(
		func = ast.Attribute( value = ast.Name( id = '$union_cls', ctx = ast.Load() ), attr = '__allocate__', ctx = ast.Load() ),
		args = [],
		keywords = [
			ast.keyword( arg = tag_attr.stem, value = ast.Constant( value = tag_value )),
			ast.keyword( arg = data_attr.stem, value = payload_call ),
		],
	)
	node = ast.FunctionDef(
		name = member.stem,
		args = ast.arguments( posonlyargs = [], args = [], vararg = None, kwonlyargs = [], kw_defaults = [], kwarg = None, defaults = [] ),
		body = [ ast.Return( value = allocate_call ) ],
		decorator_list = [], returns = None, type_params = [],
	)
	node.lineno = fn_line or 1
	node.col_offset = 0
	ast.fix_missing_locations( node )

	fn = Function(
		stem = member.stem, qualname = f'{union.qualname}.{member.stem}',
		file = fn_file, line = fn_line,
		cls = union, node = node, parameters = [ value_param ], return_type = return_type,
		is_static = True, resolve = None,
	)
	fn.add_name( 'value', value_param )
	fn.add_name( '$union_cls', union )
	fn.add_name( '$payload_cls', payload_cls )
	return fn

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
			# scheduling `attr` itself (the union's own member Variable) is
			# NOT enough to get its LEAF TYPE emitted: schedule()'s own
			# guard only enqueues a Function/ClassLike/global Variable, and
			# a union member's own attribute Variable is none of those (not
			# global - it's a class-attribute-shaped Variable, same as an
			# RCClass's own field) - scheduling it is a silent no-op. This
			# left a real, confirmed gap: a union member whose own leaf type
			# (e.g. an RCClass like str) is otherwise NEVER constructed
			# anywhere reachable in the program - only ever flowing through
			# as this union's own unconstructed variant - never got its full
			# struct emitted, even though the union's own generic tag-gated
			# Incref/Decref codegen (cfg.py's _tag_gated_refcount_
			# instructions) always needs to reference EVERY RC-bearing
			# member's full layout, unconditionally, regardless of whether
			# this program ever actually constructs one (confirmed via a
			# real repro: emit_c() emits `release_object(&(t5)->$header)`
			# against a leaf type that was only ever forward-declared,
			# `struct builtins$str` with no member definitions - clang
			# fails with "incomplete definition of type"). Explicitly
			# scheduling attr.type here closes that gap.
			self._ensure_resolved( attr.type )
		# a monomorphized CONCRETE specialization's own .names starts as a
		# shallow copy of its abstract base's .names (Monomorphizer.
		# monomorphize_class calls union_storage.get(base) before building
		# that copy, precisely so 'tag'/'data' already ride along in it) -
		# a caller reaching THIS object directly for the first time (this
		# instance's own _cache, keyed by id(union), has never seen this
		# exact object - cfg.py's own per-temp decref lookups are the
		# confirmed real case) would otherwise immediately collide with
		# its own inherited copy below. Detected via the synthesized
		# payload CUnion's own reserved qualname pattern (f'{union.
		# qualname}$data' - '$' never appears in a real source identifier,
		# so this can't coincidentally match a genuine user-declared field
		# of the same name) - a TRUE collision (a real @union class that
		# happens to declare its own 'tag'/'data' member) still falls
		# through to the ordinary synthesis-and-fail path below
		existing_tag = union.names.get( 'tag' )
		existing_data = union.names.get( 'data' )
		if (
			isinstance( existing_tag, Variable ) and isinstance( existing_data, Variable )
			and isinstance( existing_data.type, CUnion ) and existing_data.type.qualname == f'{union.qualname}$data'
		):
			tags = { attr.stem: i for i, attr in enumerate( union.attributes ) }
			result = ( existing_tag, existing_data, existing_data.type, tags )
			self._cache[ id( union ) ] = result
			return result
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
		ctor_return_type = (
			self.discovery._get_or_create_specialization( union, union.type_params )
			if union.type_params else union
		)
		# union.file is None for a synthesized anonymous union (by design -
		# see build_member_constructor's own comment on why that can't
		# change) but each constructor Function still needs a real file the
		# moment anything actually CALLS it - "whichever module is
		# currently active" is the same fallback discovery.py's own
		# _get_or_create_closure_type already established for the
		# identical reason
		fn_file = union.file
		fn_line = union.line
		if fn_file is None and self.discovery.module_stack:
			fn_file = self.discovery.module_stack[-1].file
			fn_line = self.discovery.module_stack[-1].line
		for tag_value, attr in enumerate( union.attributes ):
			union.names[attr.stem] = build_member_constructor( union, attr, tag_value, tag_attr, data_attr, payload_cls, ctor_return_type, fn_file, fn_line )
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
