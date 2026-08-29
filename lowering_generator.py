# stdlib imports:
import ast
import copy

# local imports:
import cfg
import ir
from mpy_types import (
	Name, Type, Variable, Function, Specialization, TaggedUnion, CStruct, CUnion, CEnum, TypeVar, RCClass, Scalar,
)
from type_resolver import TypeResolver

class GeneratorLoweringMixin:
	''' generator yield/suspend/resume lowering - mixed into FunctionLowering (lowering.py), which
	see for the shared instance state (self._instructions, self._cfg, self.lowering,
	etc.) every method here reads and writes. Never instantiated on its own;
	split out of lowering.py purely to keep that file to a manageable size - see
	lowering.py's own class docstring and FunctionLowering's base-class list for
	the full set of sibling mixins this one is composed with. '''


	def _emit_generator_dispatch_prologue( self, fn: Function ) -> None:
		''' PLAN_GENERATORS.md Phase F - a real state-check-and-goto
		dispatch, built directly as IR rather than synthesized AST like
		everything else in a generator's assembled $$__next__ body still
		is (see type_resolver.py's _build_generator_next_function's own
		docstring for why: Python's ast module has no goto statement to
		spell this with). For every (state, resume_label) TypeResolver.
		_assign_generator_yield_dispatch tagged onto fn.node (cached as
		node.generator_yield_states): `if self.__state == state: goto
		resume_label`. Reuses ordinary comparison lowering (a synthesized
		ast.Compare fed through _lower_expr) rather than hand-building
		the GetAttr/comparison IR directly - the exact same "borrow the
		real expression-lowering pipeline for a tiny synthesized
		snippet" trick this file's other generator hooks already use
		(_build_generator_error_defer_replay, etc.).

		State 0 ("not yet started") and the DONE sentinel (checked
		separately, by the assembled body's own leading ast.If - see
		_build_generator_next_function) both simply fail every check
		here and fall through into the body's own ordinary top, exactly
		as intended - this only ever needs to actively dispatch on a
		real mid-body suspend state. '''
		states = getattr( fn.node, 'generator_yield_states', None )
		if not states:
			return
		bool_cls = self.lowering.discovery.get_intrinsics()['bool']
		for state, resume_label in states:
			self_attr = ast.Attribute( value = ast.Name( id = 'self', ctx = ast.Load() ), attr = '__state', ctx = ast.Load() )
			compare = ast.Compare( left = self_attr, ops = [ ast.Eq() ], comparators = [ ast.Constant( value = state ) ] )
			ast.fix_missing_locations( ast.copy_location( compare, fn.node ) )
			cond = self._lower_expr( compare, bool_cls )
			skip_label = self._new_label( 'gen_dispatch_skip' )
			self._emit( ir.JumpIfFalse( cond = cond, target = skip_label ))
			self._emit( ir.Jump( target = resume_label ))
			self._emit( ir.Label( name = skip_label ))

	def _lower_generator_yield( self, node: ast.Yield ) -> None:
		''' PLAN_GENERATORS.md Phase F - a real `yield` suspend point:
		`self.__state = state` (an ordinary SetAttr - __state is a scalar
		usize field, never RC-typed, so this needs none of _stmt_Assign's
		decref-old-value machinery), the yielded value coerced against
		this function's own declared return type (mirrors _stmt_Return's
		identical coercion + _incref_aliasing_return call, deliberately
		WITHOUT everything else Return does - a yield doesn't unwind or
		jump to any epilogue; locals persist across a suspend by
		construction, see type_resolver.py's own live-flag-field
		mechanism for what actually makes that safe at the value level),
		then ir.Yield itself, then the resume Label.

		Composes for free with arbitrary nesting (if/while/for/with),
		unlike the old AST-synthesis unit-matcher this replaced: nothing
		here touches self._cfg's bindings/live-set at all, so lowering
		the WHOLE generator body once, in ordinary program order (exactly
		like any non-generator function's body), already leaves the
		compiler's own static RC-ownership view exactly where a real
		fall-through would - regardless of how many times a given
		textual point is actually reached at RUNTIME via a resume jump.
		No merge_if-style reconciliation is needed for the dispatch
		prologue's own jumps either, for the same reason: they're
		alternate ENTRY points into one linear lowering pass, not a fork
		requiring two independently-lowered branches to be reconciled.

		state/resume_label can genuinely be missing here (not just a
		theoretical "should never happen"): a generator whose own
		ensure_generator_synthesized run aborted partway through, for an
		unrelated already-reported reason (e.g. a `return` inside a
		defer/errdefer body - _reject_return_inside_generator_defer_body,
		called from _desugar_generator_defer_sites, BEFORE _assign_
		generator_yield_dispatch ever runs), leaves fn.node.body's own
		yields untagged AND fn.return_type never rewritten to the
		synthesized backing class - id(fn) is already memoized as
		"synthesized" by then (see ensure_generator_synthesized's own
		top), so nothing retries it, and the original, still-yield-
		bearing body can still reach real lowering via a later, unrelated
        call site. A graceful discovery.fail() here, not a raw crash - a
		single already-broken generator shouldn't be able to take down
		the whole compile run. '''
		self._emit_generator_yield_suspend( node )

	def _emit_generator_yield_suspend( self, node: ast.Yield ) -> None:
		''' PLAN_GENERATORS.md Phase F/Phase C - the suspend itself, shared
		by both the discarded (statement-position, `_lower_generator_
		yield` above) and captured (expression-position, `_expr_Yield`
		below) cases: `self.__state = state` (an ordinary SetAttr -
		__state is a scalar usize field, never RC-typed, so this needs
		none of _stmt_Assign's decref-old-value machinery), the yielded
		value coerced against this function's own declared return type
		(mirrors _stmt_Return's identical coercion + _incref_aliasing_
		return call, deliberately WITHOUT everything else Return does - a
		yield doesn't unwind or jump to any epilogue; locals persist
		across a suspend by construction, see type_resolver.py's own
		live-flag-field mechanism for what actually makes that safe at
		the value level), then ir.Yield itself, then the resume Label.
		`_expr_Yield` picks up from here to read back whatever `.send()`
		delivered - this method itself has no idea whether that's even
		possible (SendType may not be declared at all), that's entirely
		its own caller's concern.

		Composes for free with arbitrary nesting (if/while/for/with),
		unlike the old AST-synthesis unit-matcher this replaced: nothing
		here touches self._cfg's bindings/live-set at all, so lowering
		the WHOLE generator body once, in ordinary program order (exactly
		like any non-generator function's body), already leaves the
		compiler's own static RC-ownership view exactly where a real
		fall-through would - regardless of how many times a given
		textual point is actually reached at RUNTIME via a resume jump.
		No merge_if-style reconciliation is needed for the dispatch
		prologue's own jumps either, for the same reason: they're
		alternate ENTRY points into one linear lowering pass, not a fork
		requiring two independently-lowered branches to be reconciled.

		state/resume_label can genuinely be missing here (not just a
		theoretical "should never happen"): a generator whose own
		ensure_generator_synthesized run aborted partway through, for an
		unrelated already-reported reason (e.g. a `return` inside a
		defer/errdefer body - _reject_return_inside_generator_defer_body,
		called from _desugar_generator_defer_sites, BEFORE _assign_
		generator_yield_dispatch ever runs), leaves fn.node.body's own
		yields untagged AND fn.return_type never rewritten to the
		synthesized backing class - id(fn) is already memoized as
		"synthesized" by then (see ensure_generator_synthesized's own
		top), so nothing retries it, and the original, still-yield-
		bearing body can still reach real lowering via a later, unrelated
        call site. A graceful discovery.fail() here, not a raw crash - a
		single already-broken generator shouldn't be able to take down
		the whole compile run. '''
		state = getattr( node, 'generator_yield_state', None )
		resume_label = getattr( node, 'generator_resume_label', None )
		if state is None or resume_label is None:
			self.lowering.discovery.fail(
				f'{self._current_fn.qualname}: yield reached outside a successfully-synthesized generator '
				f'(an earlier, already-reported error left this generator only partially built) - see PLAN_GENERATORS.md',
				node,
			)
			return

		self_name = ast.Name( id = 'self', ctx = ast.Load() )
		ast.copy_location( self_name, node )
		self_obj, writeback = self._lower_attr_target_obj( self_name )
		usize_cls = self.lowering.discovery.get_intrinsics()['usize']
		self._emit( ir.SetAttr( obj = self_obj, attr = '__state', value = ir.Const( type = usize_cls, value = state )))
		if writeback is not None:
			writeback( self_obj )

		value_node = node.value if node.value is not None else ast.Constant( value = None )
		value = self._lower_expr( value_node, self._current_fn.return_type, strict = False )
		# strict=False (mirrors _stmt_Return's own identical call) skips
		# _lower_expr's own built-in final rejection - so, same as _stmt_
		# Return, this needs its OWN explicit check afterward: _lower_
		# expr already applies every coercion it legitimately can (union-
		# leaf-wrap via _coerce_into_union, RCClass base-upcast); if
		# value.type STILL doesn't match this function's own declared
		# return type, that's a genuine, uncaught mismatch that would
		# otherwise silently emit a `return` of the wrong C type - a
		# scoped-down version of _stmt_Return's own check (no CEnum
		# bidirectional exemption, no Result-error widening - neither
		# realistically arises for a generator's own elem_type|None/
		# Result[elem_type|None,error_type] return shape), confirmed via
		# a real repro: `for x in range(n): yield x * 10` inside an
		# Iterator[i32] generator (range()'s own loop counter is usize,
		# not i32) compiled with zero errors and produced C a real
		# compiler rejects outright (`return $tN;` returning a bare
		# uintptr_t where the union struct is expected) - found while
		# testing A.4a's own nested-for-loop generalization, but
		# reproduces identically with no nesting involved at all.
		if value.type is not None and self._current_fn.return_type is not None:
			fn_type = self._current_fn.return_type
			expected_concrete = fn_type
			if isinstance( fn_type, Specialization ) and isinstance( fn_type.base, ( RCClass, CStruct, CUnion, TaggedUnion, CEnum )):
				expected_concrete = self.lowering.monomorphize_class( fn_type )
			value_concrete = value.type
			if isinstance( value.type, Specialization ) and isinstance( value.type.base, ( RCClass, CStruct, CUnion, TaggedUnion, CEnum )):
				value_concrete = self.lowering.monomorphize_class( value.type )
			if (
				value.type is not fn_type and value.type is not expected_concrete
				and value_concrete is not fn_type and value_concrete is not expected_concrete
			):
				self.lowering.discovery.fail(
					f'{ast.unparse(node)}: yield produces {fn_type.qualname if fn_type else "None"}, '
					f'not {value.type.qualname if value.type else "?"}',
					node,
				)
		self._incref_aliasing_return( value_node, value )
		# mirrors _stmt_Return's own identical call (its simpler, no-
		# shared-epilogue-label branch - a yield needs none of that
		# branch's OTHER machinery, cfg.return_()'s own epilogue-style
		# unwind of every other still-live local, since a yield's own
		# suspend must NOT decref anything else - locals persist across
		# it by construction): `value`'s own ownership is about to
		# transfer into ir.Yield below (a bare temp, if the yielded
		# expression needed coercing into this function's own union
		# return type - `yield self.<field>`, the common case). Without
		# this, _flush_pending_temps right after ALSO decrefs it,
		# silently cancelling out the coercion's own incref (confirmed
		# via a real refcount() repro: a captured RC parameter yielded
		# back through a match arm read compiler.refcount() one lower
		# than expected).
		self._cfg.untrack_temp( value )
		self._flush_pending_temps()
		self._emit( ir.Yield( value = value, state = state, resume_label = resume_label ))
		self._emit( ir.Label( name = resume_label ))

	def _expr_Yield( self, node: ast.Yield, expected_type: Type|None ) -> ir.Operand:
		''' PLAN_GENERATORS.md Phase C - `(yield expr)` used as an
		EXPRESSION: the suspend itself is identical to the ordinary
		statement-position case (_emit_generator_yield_suspend, shared) -
		what's new is what happens AFTER resuming. Only legal inside a
		generator that declared a SendType (Generator[T,SendType,E]);
		Iterator[T]/the 2-arg Generator[T,E] form have no SendType to
		ever deliver a captured yield's own value through, so this fails
		clearly instead of reaching lowering with nothing to read back
		(mirrors the pre-Phase-C behavior exactly - _lower_expr's own
		getattr-based dispatch already rejected any expression-position
		yield as "unsupported expression" before this method existed at
		all, just with a much less specific message).

		Reads back self.__send_ready: True means send(v) armed __send_
		slot since this exact suspend point - clears the flag (consumed,
		one-shot) and reads __send_slot as this expression's own value; a
		PLAIN field read, no incref of its own - _is_aliasing_expr now
		recognizes a captured ast.Yield as aliasing (same category
		ast.Attribute already is - it reads an existing field, self.
		__send_slot's own reference, independent of whatever __send_slot
		itself keeps), so wherever this operand ultimately lands (`held =
		yield i`, an argument, ...) gets its own Incref through the SAME
		ordinary consumption-time machinery any other field read already
		relies on - no special-casing needed here beyond that one
		recognition fix. False means this resume came from a bare
		__next__()/for-loop consumption instead (send_ready never armed) -
		panics with a message pointing at .send(), mirroring Python's own
		runtime behavior for resuming a captured yield without sending a
		value. Composes for FREE with anything wrapping the yield (`x =
		yield v`, `x = (yield v).or_return()`, ...) since this is ordinary
		expression lowering - no special-casing needed for nesting, same
		reason Phase F's own dispatch mechanism generalized nesting/
		multiplicity for free. '''
		send_type = getattr( self._current_fn.node, 'generator_send_type', None )
		if send_type is None:
			self.lowering.discovery.fail(
				f'{self._current_fn.qualname}: yield can only be used as an expression inside a generator that declares a '
				f'SendType (Generator[T,SendType,E]) - a bare `yield expr` statement is still supported everywhere else - '
				f'see PLAN_GENERATORS.md',
				node,
			)
			send_type = self._current_fn.return_type # keep going with SOME type so the rest of this method still produces a well-typed (if meaningless) operand, same "don't cascade into confusing follow-on errors" posture the rest of this file uses after a reported failure

		self._emit_generator_yield_suspend( node )

		self_name = ast.Name( id = 'self', ctx = ast.Load() )
		ast.copy_location( self_name, node )
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		ready_read = ast.Attribute( value = self_name, attr = '__send_ready', ctx = ast.Load() )
		ast.copy_location( ready_read, node )
		ready = self._lower_expr( ready_read, bool_cls )

		panic_label = self._new_label( 'gen_send_required' )
		end_label = self._new_label( 'gen_send_end' )
		self._emit( ir.JumpIfFalse( cond = ready, target = panic_label ))

		# true branch: a pending send exists - consume it (clear the flag,
		# one-shot) and deliver __send_slot as this expression's value
		self_obj, writeback = self._lower_attr_target_obj( self_name )
		self._emit( ir.SetAttr( obj = self_obj, attr = '__send_ready', value = ir.Const( type = bool_cls, value = False )))
		if writeback is not None:
			writeback( self_obj )
		slot_read = ast.Attribute( value = ast.Name( id = 'self', ctx = ast.Load() ), attr = '__send_slot', ctx = ast.Load() )
		ast.copy_location( slot_read, node )
		ast.copy_location( slot_read.value, node )
		slot_val = self._lower_expr( slot_read, send_type, strict = False )
		dest = self._new_temp( send_type )
		self._emit( ir.Assign( dest = dest, src = slot_val ))
		self._emit( ir.Jump( target = end_label ))

		# false branch: resumed via __next__()/for-loop consumption
		# instead of .send() - panic, same statement-position AST-
		# synthesis + swap-buffer technique _build_generator_error_defer_
		# replay/_build_generator_pessimistic_done_pin already use to
		# reuse ordinary _lower_call/_lower_stmt machinery for a call this
		# file has no other reason to hand-build IR for directly
		self._emit( ir.Label( name = panic_label ))
		panic_call = ast.Call(
			func = ast.Attribute( value = ast.Name( id = 'sys', ctx = ast.Load() ), attr = 'panic', ctx = ast.Load() ),
			args = [ ast.Constant( value = f'{self._current_fn.qualname}: resumed via __next__()/for-loop consumption at a captured yield - use .send() instead' ) ],
			keywords = [],
		)
		panic_stmt = ast.Expr( value = panic_call )
		ast.copy_location( panic_call, node ); ast.copy_location( panic_stmt, node )
		ast.fix_missing_locations( panic_stmt )
		self._lower_stmt( panic_stmt )
		self._emit( ir.Label( name = end_label ))
		return dest

	def _build_generator_zero_value( self, t: Type, node: ast.AST ) -> ir.Operand:
		''' PLAN_GENERATORS.md Phase 5's own "generator_zero_rc_field"
		placeholder mechanism (type_resolver.py's _rewrite_generator_
		constructor) works for a plain scalar or a plain RCClass-typed
		promoted field (a bare `0` reinterpreted as NULL, gated on the
		compiler-internal tag _expr_Constant checks), but a TaggedUnion
		WITHOUT a None member (e.g. Result[Box,IndexError] - no leaf a bare
		int literal can coerce into when NEITHER leaf is itself a plain
		int) has no such placeholder at all - confirmed unreachable before
		anything actually promoted a match statement's own subject (see
		_reserve_generator_match_subject_fields), which routinely needs
		exactly this shape (Result[T,IndexError] for an arbitrary element
		type T, including an RCClass). Building a REAL leaf instance via
		the union's own synthesized Ok/Err-style wrap constructor is unsafe
		here: that constructor's own body unconditionally increfs the leaf
		it wraps (see _coerce_into_union's own comment) - retaining a NULL
		placeholder would deref a null ObjectHeader* and crash immediately
		at CONSTRUCTION time, before the generator is ever iterated once.

		Instead this builds the union's raw tag+data storage directly via
		two nested ir.Allocate ops (mirroring UnionStorage.build_member_
		constructor's own `$union_cls.__allocate__(tag=.., data=$payload_
		cls(v_<member>=value))` shape, just built at the IR level instead
		of via synthesized AST needing a scope to resolve $union_cls/
		$payload_cls names in) - NOT the member's own wrap constructor, so
		no incref ever fires: _lower_allocate_fields's own generic field-
		building loop only increfs an ALIASING value (a bare Name/Attribute
		read of something that already exists), and every value built here
		is a fresh ast.Constant, never aliasing (_is_aliasing_expr's own
		documented rule - "Constant... never produce RC values at all"),
		so it's treated as already-owned, no incref emitted - correct here
		specifically because there is nothing real to own yet.

		Recurses into the union's own FIRST member (arbitrarily - live-
		flag gating means the actual tag/value picked here is never
		observed by anything but this same field's own state/flag-gated
		destructor deciding NOT to decref it), so a leaf that's itself a
		nested union (uncommon, unexercised by anything built so far) gets
		the same treatment. Declines (a clean, actionable compile error,
		not a silent miscompile or a raw crash) for any OTHER leaf shape
		(CStruct, CEnum, ...) - genuinely unneeded for this bug's own
		scope (every case built so far is a scalar or an RCClass leaf),
		and a real "how do you zero THIS" design question if it ever is,
		better raised explicitly than guessed. '''
		base = t.base if isinstance( t, Specialization ) else t
		if isinstance( base, Scalar ):
			const_node = ast.Constant( value = False if base.stem == 'bool' else 0 )
			ast.copy_location( const_node, node )
			return self._lower_expr( const_node, t )
		if isinstance( base, RCClass ):
			const_node = ast.Constant( value = 0 )
			const_node.generator_zero_rc_field = True
			ast.copy_location( const_node, node )
			return self._lower_expr( const_node, t )
		if isinstance( base, TaggedUnion ):
			_abstract_tag_attr, _abstract_data_attr, _abstract_payload_cls, tags = self.lowering._union_storage.get( base )
			# union_storage.get(base)'s own tag/data/payload_cls are for the
			# ABSTRACT union - deliberately never schedule()d by get() itself
			# when the union is generic (see its own comment: "only a
			# CONCRETE specialization's own substituted payload_cls... is
			# ever a real compile unit"), so emitting THAT bare payload_cls
			# directly would bake in still-unbound TypeVars (confirmed by a
			# real repro: NotImplementedError: c_type: unsupported type
			# <TypeVar 'builtins.Result.T'>, from a generic generator's own
			# for-loop-over-iterator desugaring, which synthesizes exactly
			# this shape - _desugar_iterator_for's own internal match,
			# checking .__next__()'s Result). Mirrors _lower_allocate_
			# fields's own identical "concrete_union = ...monomorphize_
			# class(fn_cls)" branch for a TaggedUnion target_cls, just keyed
			# on `t` itself being a Specialization rather than on the
			# current lowering context's own class happening to match -
			# this can be reached from ANY generator, not just from inside
			# one of the union's own synthesized methods. `tags` (tag
			# VALUES, not types) doesn't change under substitution, so it's
			# still read from the abstract union above.
			if isinstance( t, Specialization ):
				concrete_union = self.lowering.monomorphize_class( t )
				tag_attr = concrete_union.get_local_or_raise( 'tag' )
				data_attr = concrete_union.get_local_or_raise( 'data' )
				assert isinstance( tag_attr, Variable ) and isinstance( data_attr, Variable )
				payload_cls = data_attr.type
				assert isinstance( payload_cls, CUnion )
			else:
				tag_attr, data_attr, payload_cls = _abstract_tag_attr, _abstract_data_attr, _abstract_payload_cls
			leaf = self.lowering._substituted_field( base.attributes[0], t )
			leaf_value = self._build_generator_zero_value( leaf.type, node )
			for instr in self._cfg.field_value( leaf_value.type, leaf_value, is_alias = False ):
				self._emit( instr )
			self.lowering.schedule( payload_cls )
			payload_dest = self._new_temp( payload_cls )
			self._emit( ir.Allocate( dest = payload_dest, cls = payload_cls, fields = { f'v_{leaf.stem}': leaf_value } ))
			tag_const_node = ast.Constant( value = tags[ leaf.stem ] )
			ast.copy_location( tag_const_node, node )
			tag_value = self._lower_expr( tag_const_node, tag_attr.type )
			for instr in self._cfg.field_value( tag_value.type, tag_value, is_alias = False ):
				self._emit( instr )
			self.lowering.schedule( t )
			union_dest = self._new_temp( t )
			self._emit( ir.Allocate( dest = union_dest, cls = base, fields = { tag_attr.stem: tag_value, data_attr.stem: payload_dest } ))
			return union_dest
		self.lowering.discovery.fail(
			f'cannot build a generator zero-placeholder value for {t.qualname if t else "?"} yet - see PLAN_GENERATORS.md',
			node,
		)

	def _build_generator_error_defer_replay( self ) -> list[ir.Instruction]:
		''' PLAN_GENERATORS.md's defer/errdefer phase (Mechanism 2) - lowers
		every site in self._generator_armed_defer_sites (LIFO - deepest/
		most-recently-armed first, same convention Mechanism 1's own
		_build_defer_replay_guards uses), wrapped in `if self.
		__defer_armed_N: <body>`, for splicing into an OrReturn's own
		epilogue (see this method's one call site). A no-op (empty list,
		no lowering work at all) whenever the list is empty - true for
		EVERY ordinary, non-generator function, and for a generator body
		with no armed defer/errdefer site reaching this exact position.

		Reuses AST synthesis + the swap-the-instruction-buffer technique
		_register_defer_block already established (lower once into a
		fresh buffer, splice the result) rather than hand-building IR
		directly - the body statements are ALREADY self.<field>-qualified
		(type_resolver.py's _build_generator_next_function renamed them
		once, via the same renamer used for everything else in $$__next__,
		before ever tagging a node with them), so ordinary statement
		lowering already does the right thing with zero new machinery.

		Each site's own body is deep-copied FRESH here (same reasoning as
		_build_defer_replay_guards - lowering attaches mutable per-
		occurrence attributes like resolved_* that would corrupt a shared
		node if two OrReturn sites, or this site and a Mechanism-1 normal-
		exit site, shared one). self._generator_armed_defer_sites is reset
		to empty while lowering each body - a defer/errdefer body is
		expected to be simple cleanup, not itself something needing its
		OWN error-defer replay; without this, a fallible operation nested
		inside a defer body would recurse into this same method against
		the SAME still-armed site, unboundedly.

		Each guard ALSO unsets its own flag right after replaying (self.
		__defer_armed_N = False) - same reasoning as _build_defer_replay_
		guards' own identical unset: this error exit permanently pins
		self.__state to done, but the generator OBJECT itself often isn't
		freed until later (whatever reference the caller still holds), at
		which point $$__destructor__'s own Mechanism-1 replay would
		otherwise see this SAME flag still True and fire the (non-
		errdefer) body a second time. '''
		if not self._generator_armed_defer_sites:
			return []
		instructions: list[ir.Instruction] = []
		outer_armed = self._generator_armed_defer_sites
		self._generator_armed_defer_sites = []
		try:
			for flag_stem, _is_errdefer, body_stmts in reversed( outer_armed ):
				body_copy = [ copy.deepcopy( s ) for s in body_stmts ]
				unset = ast.Assign(
					targets = [ ast.Attribute( value = ast.Name( id = 'self', ctx = ast.Load() ), attr = flag_stem, ctx = ast.Store() ) ],
					value = ast.Constant( value = False ),
				)
				guard = ast.If(
					test = ast.Attribute( value = ast.Name( id = 'self', ctx = ast.Load() ), attr = flag_stem, ctx = ast.Load() ),
					body = ( body_copy or [ ast.Pass() ] ) + [ unset ], orelse = [],
				)
				if body_stmts:
					ast.copy_location( guard, body_stmts[0] )
				else:
					guard.lineno = 1; guard.col_offset = 0
				ast.fix_missing_locations( guard )
				outer_instructions = self._instructions
				self._instructions = []
				try:
					self._lower_stmt( guard )
				finally:
					captured = self._instructions
					self._instructions = outer_instructions
				instructions += captured
		finally:
			self._generator_armed_defer_sites = outer_armed
		return instructions

	def _build_generator_pessimistic_done_pin( self ) -> list[ir.Instruction]:
		''' PLAN_GENERATORS.md Phase 4/Phase F - a fallible generator's
		$$__next__ needs "permanently done" set on ANY early error exit
		(or_return()'s own Err branch, or checked-arithmetic under Check
		mode consumed the same way) - OrReturn's own error exit returns
		directly out of $$__next__ WITHOUT running whatever would
		normally advance self.__state afterward, so without this, self.
		__state stays at whatever it was BEFORE the failing statement,
		and a later .__next__() call would wrongly re-enter and re-run
		the same (possibly already-consumed-a-moved-value) code from
		scratch.

		Phase F re-derives this at the LOWERING level (this hook,
		spliced into ir.OrReturn's own epilogue right alongside
		Mechanism 2's error-defer replay - see this method's one call
		site) instead of the old AST-level pre-write (_pessimistic_
		done_prefix, inserted ahead of every block of user code that
		MIGHT fail, deleted along with the rest of the unit-matcher):
		reaching this exact point during lowering already means an
		early Err-branch exit is really happening, so the pin only ever
		needs building once per OrReturn site, not speculatively ahead
		of every fallible-eligible block regardless of whether it's
		even generator code. A no-op outside a generator
		(self._current_fn.is_generator_next False for every ordinary
		function) - correct, since only a generator's own $$__next__
		has a self.__state field to pin at all. '''
		if not self._current_fn.is_generator_next:
			return []
		done_state = len( self._current_fn.node.generator_yield_states ) + 1
		pin = ast.Assign(
			targets = [ ast.Attribute( value = ast.Name( id = 'self', ctx = ast.Load() ), attr = '__state', ctx = ast.Store() ) ],
			value = ast.Constant( value = done_state ),
		)
		ast.fix_missing_locations( ast.copy_location( pin, self._current_fn.node ) )
		outer_instructions = self._instructions
		self._instructions = []
		try:
			self._lower_stmt( pin )
		finally:
			captured = self._instructions
			self._instructions = outer_instructions
		return captured

	def _try_lower_generator_allocate_call( self, node: ast.Call, expected_type: Type|None ) -> ir.Temp|None:
		# PLAN_GENERATORS.md - Lowering._rewrite_generator_constructor's own
		# synthesized `BackingClass(__state=0, ...)` call, tagged directly
		# with the target RCClass object itself (generator_backing_cls)
		# rather than something resolvable by name through any real scope -
		# same escape-hatch spirit as node.resolved_callee/resolved_
		# construction elsewhere in this file, just for a class instead of
		# a function.
		target_cls = getattr( node, 'generator_backing_cls', None )
		if target_cls is None:
			return None
		return self._lower_allocate_fields( target_cls, node, expected_type, '(...)' )
