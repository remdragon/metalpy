# stdlib imports:
import ast
from typing import Callable, Iterable, NoReturn

# local imports:
import cfg
import ir
from errors import CompileError
from mpy_types import (
	Name, Type, Variable, Function, Specialization, TaggedUnion, CStruct, RCClass, Scalar, GeneratorType,
)

from lowering_shared import _LoopContext

class LoopLoweringMixin:
	''' while/for loop lowering, break/continue, and loop-body control-flow helpers - mixed into FunctionLowering (lowering.py), which
	see for the shared instance state (self._instructions, self._cfg, self.lowering,
	etc.) every method here reads and writes. Never instantiated on its own;
	split out of lowering.py purely to keep that file to a manageable size - see
	lowering.py's own class docstring and FunctionLowering's base-class list for
	the full set of sibling mixins this one is composed with. '''


	# --- loops ---------------------------------------------------------------

	def _lower_truth_test( self, node: ast.expr ) -> ir.Operand:
		''' the value fed to a while/if's own truth test - Python semantics:
		ANY value can be tested for truthiness, not just a real bool
		(0/0.0 -> False, any other scalar -> True; a bool already -> itself
		unchanged; anything else - an RC/union receiver, say - falls
		through to the ordinary strict bool-typed coercion, which already
		handles is-not-None-style union narrowing and correctly rejects a
		genuine non-bool-non-scalar mismatch). strict=False here skips
		_check_assignable's hard rejection so a non-bool SCALAR operand can
		be intercepted and converted, rather than failing outright with
		"expected bool, got i32" - confirmed missing by a real repro (any()/
		all() over a list[i32] needed an explicit bool(item) wrapper to
		compile at all, unlike Python's own `if item:`). Reuses
		_lower_scalar_cast for the actual conversion - the same mechanism
		bool(x) construction-sugar already goes through, now unconditionally
		safe for a bool target (see its own comment) so this never needs an
		enclosing wrap_arithmetic/panic_arithmetic the way a genuine
		narrowing scalar cast would. '''
		# None, not bool_cls: a BoolOp/IfExp dispatch method treats a given
		# expected_type as authoritative for ITS OWN dest (seeding the
		# merge target's type up front, needed for real by the genuine
		# "declared union target" case, e.g. `z: T|U = x or y`) - it has
		# no way to tell "the caller only wants this as loose HINT"
		# (strict=False below) apart from "the caller genuinely requires
		# it" (an ordinary strict assignment/argument/return), since
		# strict itself is never threaded down into the dispatch methods.
		# Forcing bool_cls here would make THAT operand's own natural
		# value (e.g. a bare RCClass with no __bool__, always truthy
		# per _truthiness_of_operand's own default) fail to coerce for a
		# condition like `if f and g:` even though real Python evaluates
		# it fine - the whole point of computing the REAL value first and
		# reducing it to bool via _truthiness_of_operand afterward, same
		# as any other expression kind here. A bare literal (True/False/5/
		# 1.0/...) still self-types correctly without this hint - its own
		# Python value's type (bool vs int vs float) is what _expr_
		# Constant keys off, not expected_type.
		operand = self._lower_expr( node, None, strict = False )
		return self._truthiness_of_operand( operand, node )

	def _truthiness_of_operand( self, operand: ir.Operand, node: ast.AST ) -> ir.Operand:
		''' the tail half of _lower_truth_test above, factored out so a
		caller that already has a LOWERED operand - and, unlike if/while/
		ternary's test position, also needs the operand's own original
		value (_expr_BoolOp, which must return the decisive operand's real
		value, not a forced bool) - can get its truthiness without lowering
		`node` a second time (which would double any side effects, e.g. a
		Call operand). Same rules and same reused conversion machinery
		(_lower_scalar_cast/_coerce_or_check_operand) as _lower_truth_test
		itself - see its docstring for the semantics. '''
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		if operand.type is bool_cls or self.lowering._type_resolver._same_type( operand.type, bool_cls ):
			return operand
		if isinstance( operand.type, Scalar ):
			return self._lower_scalar_cast( bool_cls, operand, node )
		# a non-Scalar type with its own __bool__ (e.g. builtins.str - empty
		# string is falsy, see its own docstring) dispatches to it directly -
		# the bare (non-union) counterpart of _rewrite_tagged_union_
		# truthiness's identical call on a NULLABLE union's non-None leaf.
		# Previously missing entirely: a bare str used as a condition
		# silently tested its own always-true POINTER instead of its
		# content (see builtins.print()'s own end-parameter comment, which
		# worked around it with an explicit len()!=0 check rather than
		# fixing the root cause) - now needed for real by _expr_BoolOp,
		# which requires a genuine truthiness answer for ANY operand type,
		# not just bool/Scalar, to decide a non-last operand's short-circuit
		# jump without forcing its VALUE through bool too.
		bool_method = self.lowering._find_method( operand.type, '__bool__' )
		if bool_method is not None:
			self.lowering._ensure_resolved( bool_method )
			self.lowering.schedule( bool_method.return_type )
			for p in ( bool_method.parameters or [] ):
				self.lowering.schedule( p.type )
			dest = self._new_temp( bool_cls )
			self._emit( ir.Call( dest = dest, target = bool_method, receiver = operand, args = [], kwargs = {} ))
			return dest
		# no __bool__ of its own - a TaggedUnion dispatches per-leaf via its
		# own tag instead: this is the GENERAL counterpart of type_resolver.
		# py's _rewrite_tagged_union_truthiness, which only ever rewrites an
		# if/while/ternary's own TOP-LEVEL test and only for the narrow
		# "exactly one non-None leaf" shape. An operand reached from here
		# (e.g. a BoolOp operand union-coerced into i32|str, which has no
		# None leaf at all) was never visited by that AST-level rewrite in
		# the first place, so it still needs a real answer.
		# EXCEPT a Result[T,E] (or any is_result_type() union): that one
		# must stay a hard error here, not get a silent tag-based truthy/
		# falsy answer - cfg.py's own _unchecked_results tracking already
		# requires a Result be explicitly consumed (.unwrap()/.is_ok()/
		# match) before use, specifically so a fallible call's error case
		# can never be silently ignored; falling into the generic per-leaf
		# dispatch here would treat `if fallible_call():` as implicitly
		# "truthy unless Err", quietly bypassing that whole mechanism.
		shape = self.lowering._type_resolver._tagged_union_shape( operand.type )
		if shape is not None:
			if shape[0].is_result_type():
				# fall all the way through to the ordinary hard-fail below,
				# NOT the bare-type default-true fallback further down -
				# an unchecked Result must stay a real compile error either
				# way, never a silent answer of any kind
				return self._coerce_or_check_operand( operand, bool_cls, node )
			return self._truthiness_of_union_operand( operand, shape, node )
		# default truthiness for any other bare, non-Scalar, non-union,
		# no-__bool__ type (a plain RCClass/CStruct/... instance): only
		# None itself is falsy by default, everything else is truthy -
		# matches real Python's own bool(obj) default. NoneType is itself
		# a Scalar in this compiler (see discovery.get_none_type()), so a
		# bare `None`-typed operand is already handled by the Scalar
		# branch above and never reaches here; this default is reachable
		# only for a real, always-non-null object reference, for which
		# "truthy" is the only sound answer.
		return ir.Const( type = bool_cls, value = True )

	def _truthiness_of_union_operand( self, operand: ir.Operand, shape: 'tuple[TaggedUnion,list[Variable]]', node: ast.AST ) -> ir.Operand:
		''' tag-dispatch truthiness for a bare TaggedUnion operand (already
		resolved to (base, members) by the caller's _tagged_union_shape
		call) - each leaf's OWN truthiness decides the union's, recursively
		(another _truthiness_of_operand call, so a nested union leaf, or a
		leaf with its own __bool__, or a plain Scalar leaf all work exactly
		as they would standalone). A None leaf is hard-coded falsy - the
		only type this compiler treats as unconditionally falsy (see this
		method's caller). Mirrors cfg.py's _tag_gated_refcount_instructions'
		identical per-leaf tag-Cmp-JumpIfFalse-then-Jump-to-a-shared-end
		shape, just computing a bool RESULT per leaf (via ir.Assign into a
		shared dest) instead of an incref/decref side effect. '''
		base, members = shape
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		none_type = self.lowering.discovery.get_none_type()
		tag_attr, _data_attr, _payload_cls, tags = self.lowering._union_storage.get( base )
		dest = self._new_temp( bool_cls )
		end_label = self._new_label( 'union_truth_end' )
		for member in members:
			next_label = self._new_label( 'union_truth_next' )
			tag_dest = self._new_temp( tag_attr.type )
			self._emit( ir.GetAttr( dest = tag_dest, obj = operand, attr = tag_attr.stem ))
			cmp_dest = self._new_temp( bool_cls )
			self._emit( ir.Cmp( dest = cmp_dest, op = ir.CmpOp.EQ, left = tag_dest, right = ir.Const( type = tag_attr.type, value = tags[member.stem] )))
			self._emit( ir.JumpIfFalse( cond = cmp_dest, target = next_label ))
			if member.type is none_type:
				leaf_truth: ir.Operand = ir.Const( type = bool_cls, value = False )
			else:
				# owning=False (default): a transient borrow for the
				# duration of this check only, same as _maybe_unwrap_union_
				# arg's own ordinary call-argument use - `operand` itself
				# still owns the reference and gets its own normal decref
				# wherever it's otherwise flushed/released
				payload = self._maybe_unwrap_union_arg( operand, member.type )
				leaf_truth = self._truthiness_of_operand( payload, node )
				if isinstance( leaf_truth, ir.Const ):
					# _truthiness_of_operand's own default-truthy fallback
					# (a plain RCClass/CStruct/... leaf with no __bool__,
					# not itself Scalar/union) returns Const(True) WITHOUT
					# ever reading payload - real -Wunused-but-set-variable
					# (confirmed via a real repro: `if some_optional_match:`
					# where the non-None leaf is a bare class with no
					# __bool__). Harmless when leaf_truth DOES depend on
					# payload (this branch is only reached for the const
					# case), same reasoning as this file's other MarkUsed
					# call sites.
					self._emit( ir.MarkUsed( operand = payload ))
			self._emit( ir.Assign( dest = dest, src = leaf_truth ))
			self._emit( ir.Jump( target = end_label ))
			self._emit( ir.Label( name = next_label ))
		self._emit( ir.Label( name = end_label ))
		return dest

	def _lower_branch_condition( self, test: ast.expr ) -> ir.Operand:
		''' shared by _stmt_If/_stmt_While: lowers a statement-level branch
		condition, then immediately flushes any temps retained while
		evaluating it (PLAN_THREAD_SAFE_SHARED_STATE.md Part B - e.g. a
		chained field receiver, `self.a.b`) - BEFORE the caller emits its
		own JumpIfFalse/branches, not after. Both callers' own branches can
		diverge early (a while loop's body re-enters this same test every
		iteration; an if's own branch can return/break/continue, skipping
		the enclosing statement's normal once-per-statement flush entirely
		for that path) - either way, deferring the flush past the branch
		point leaves it as dead code on any path that doesn't fall through
		to the shared join point, permanently over-retaining whatever the
		condition retained. Safe to flush even though `test` itself is one
		of the flushed temps: ir.DeleteTemp is a pure bookkeeping no-op at
		emission time ("C block scoping already handles temp lifetime" -
		emitter_c.py's own comment), so `test`'s C variable stays perfectly
		readable immediately after. Not used by _expr_IfExp - a ternary's
		own condition can never contain a return/break/continue (it's an
		expression, not a statement), so its already-existing once-per-
		statement flush never has an early-exit path to go missing on. '''
		cond = self._lower_truth_test( test )
		self._flush_pending_temps()
		return cond

	def _stmt_While( self, node: ast.While ) -> None:
		if node.orelse:
			self.lowering.discovery.fail( 'while/else is not supported', node )
		pre_loop_mark = len( self._instructions ) # see _lower_loop_body_with_ownership_retry's own docstring - a promoted name's one-time incref splices in here, strictly before start_label
		start_label = self._new_label( 'while_start' )
		end_label = self._new_label( 'while_end' )
		# the test is positioned right after start_label (re-lowered here
		# once, but the resulting instructions physically sit inside the
		# repeated block, same as _stmt_If's test) so it's genuinely
		# re-evaluated every time the bottom Jump loops back
		self._emit( ir.Label( name = start_label ))
		# a walrus (`while x := f():`) FRESHLY declared right here in the
		# test is a genuinely different shape than every other loop-carried
		# binding: _expr_NamedExpr's own "no prior declaration" branch
		# (_declare_local, matching an ordinary `x = f()` statement) is
		# correct for the FIRST run through this code, but this exact same
		# compiled Assign then re-executes every later iteration too (via
		# the back-edge Jump below looping back to start_label, BEFORE the
		# test) - a plain declare-shaped Assign never decrefs whatever x
		# already held from the PRIOR iteration before overwriting it,
		# unlike an ordinary reassignment of an already-EXISTING name
		# (_cfg_assign's own decref-old-then-assign, which the test's OTHER
		# branch in _expr_NamedExpr already gets correctly). Detected here
		# by diffing fn.names before/after lowering the test - any NAME
		# that's newly present afterward was freshly declared BY the test
		# itself. Confirmed via a real repro: `while root := paths.pop(
		# ''):` leaked root's own first-iteration value every time the loop
		# body ran a genuine second iteration.
		names_before_test = dict( self._current_fn.names )
		test = self._lower_branch_condition( node.test )
		fresh_rc_walrus_names = [
			v for k, v in self._current_fn.names.items()
			if k not in names_before_test and isinstance( v, Variable ) and cfg.rc_leaves( v.type )
		]
		self._emit( ir.JumpIfFalse( cond = test, target = end_label ))
		loop_snapshot = self._cfg.snapshot()
		# continue_captured unused here - start_label (this loop's own
		# continue target) is always jumped to by the back edge below
		# regardless of whether the body itself ever uses `continue`
		break_narrowed, break_live, _ = self._lower_loop_body_with_ownership_retry(
			node, node.body, continue_label = start_label, break_label = end_label, loop_snapshot = loop_snapshot, pre_loop_mark = pre_loop_mark,
		)
		# Phase 7/8: reconcile every way execution can actually reach
		# end_label - the loop's own natural (condition-false) exit, PLUS
		# every break_narrowed record_break_narrowed() collected while
		# lowering the body above. `while True:` (test still the literal
		# Constant(True) it started as - type_resolver.py's visit_While
		# only ever rewrites it away from that shape when it recognizes a
		# real type(x) is T/is not T condition, never for a bare `True`)
		# has NO real condition-false exit at all - passing None here (not
		# a real candidate) is what keeps merge_loop_exits from wrongly
		# treating that unreachable path as competing with, and
		# suppressing, narrowing that only survives via an explicit break.
		# For an ordinary loop, type_resolver.py's own exit_narrows_name/
		# exit_narrows_member_stem (Phase 7 - the ONLY way to reach
		# end_label is via the condition going false, which for a 2-member
		# union, or unconditionally for the `is not` form, uniquely proves
		# x's own type from here on) overlays loop_snapshot's own narrowed
		# state to build the natural-exit candidate.
		is_while_true = isinstance( node.test, ast.Constant ) and node.test.value is True
		natural_exit_narrowed: dict[str,list[Variable]] | None = None
		natural_exit_live: set[str] | None = None
		if not is_while_true:
			natural_exit_narrowed = dict( loop_snapshot.narrowed )
			exit_name = getattr( node, 'exit_narrows_name', None )
			if exit_name is not None:
				exit_attr_base = getattr( node, 'exit_narrows_attr_base', None )
				if exit_attr_base is not None:
					# field-chain exit-narrowing (`while type(self.field) is
					# T:` etc) - mirrors _stmt_Assign's own is_narrowing_bind
					# attr branch exactly, see its comment
					member = self._resolve_narrow_attr_member( exit_attr_base, node.exit_narrows_attr_hops, node.exit_narrows_member_stem, node )
				else:
					member = self._resolve_narrow_member( exit_name, node.exit_narrows_member_stem, node )
				natural_exit_narrowed[exit_name] = [ member ]
			natural_exit_live = set( loop_snapshot.live )
		self._cfg.merge_loop_exits( natural_exit_narrowed, break_narrowed, natural_exit_live, break_live )
		# release this iteration's own value of any RC-typed walrus target
		# freshly declared by the test (see fresh_rc_walrus_names' own
		# comment above) BEFORE looping back to re-run that same declare-
		# shaped Assign again - it has no way to know it's about to
		# overwrite a real value rather than uninitialized memory. NOTE:
		# this only covers the loop's own NORMAL (body-completed) back
		# edge - an explicit `continue` inside the body jumps to
		# start_label directly (continue_label == start_label for a while
		# loop) and bypasses this, same gap, not yet handled; not
		# reachable by any real code in this codebase today (confirmed:
		# no while-loop-with-walrus-test in lib/ uses continue), flagged
		# here rather than silently left in a comment nobody sees.
		for var in fresh_rc_walrus_names:
			for instr in self._cfg.decref( var.type, var ):
				self._emit( instr )
		self._emit( ir.Jump( target = start_label ))
		self._emit( ir.Label( name = end_label ))

	def _lower_loop_body_with_ownership_retry(
		self, node: ast.AST, body: list[ast.stmt], continue_label: str, break_label: str, loop_snapshot: object, pre_loop_mark: int,
	) -> tuple[list[dict[str,list[Variable]]],list[set[str]],bool]:
		''' _lower_loop_body() + loop_back_edge() (+ emitting the back-edge
		instructions and restore()'ing), run up to twice. A loop body is
		lowered exactly ONCE and reused via the back edge, so a reassignment
		like `path = file` (borrowed) / `path = os.path.join(...)` (owned) -
		already reconciled by merge_if() into a single OWNED-with-flag state
		for the REST of one iteration - still leaves the loop's own ENTRY
		state (whatever `path` was before the loop, e.g. a borrowed
		parameter) disagreeing with that back edge: BORROWED vs OWNED,
		cfg.py's loop_back_edge() own hard error. Unlike merge_if's two
		independently-lowered branches, there's no reconciling this after
		the fact - the reassignment sites inside were already compiled
		assuming the ENTRY state (BORROWED), so neither of them emitted the
		decref-before-overwrite an OWNED entry would need on iteration 2+; a
		bare "promote and hope" would leak one reference per iteration.

		So: on the first CompileError from loop_back_edge, check whether
		it's exactly that safe shape (cfg.py's find_promotable_loop_
		mismatches - the same BORROWED-vs-OWNED case merge_if() reconciles
		for if/else). If so, every side effect of that attempt is
		rolled back - emitted instructions, cfg bindings/epilogue/cancel-flag
		state (via hard_restore(), NOT the ordinary restore() lowering.py
		uses on success - a defer registered while lowering the doomed
		attempt gets fully discarded, not kept alive for a function epilogue
		that will never see that code again, since the retried body below
		re-registers a fresh one), defer-flag registrations, pending temps -
		the mismatched names are then promoted to OWNED via promote_
		borrowed_for_loop() - a ONE-TIME incref spliced in at pre_loop_mark,
		strictly before this loop's own start label so the back-edge jump
		(which targets that label directly) never re-runs it - and the body
		is lowered again, this time with every reassignment site seeing the
		true steady-state entry ownership up front. Any other CompileError,
		or a mismatch still unresolved after that one retry, is reported for
		real via discovery.fail(). '''
		for attempt_number in ( 1, 2 ):
			body_snapshot = self._cfg.snapshot()
			instructions_mark = len( self._instructions )
			defer_flags_mark = len( self._defer_flags )
			for_obj_null_inits_mark = len( self._for_obj_null_inits )
			cancel_flags_mark = self._cfg.cancel_flag_count
			pending_temps_mark = len( self._pending_temps )
			# fn.names is lowering.py's own (not cfg.py's) name->Variable table -
			# hard_restore() below only reverts cfg.py's OWN bindings/epilogue
			# state, it has no notion of this dict at all. A name `del`'d and
			# then reassigned INSIDE the loop body mints a fresh, uid-suffixed
			# Variable (Variable.needs_uid_suffix - see _mark_fresh_local_
			# declared()) and registers it here (fn.add_name) - if attempt 1
			# reaches that shape and then fails/rolls back, this entry is left
			# pointing at attempt 1's own now-abandoned Variable (its own
			# declaration instruction was just discarded above) instead of
			# reverting to whatever fn.names held before this attempt started.
			# The retried attempt's own `del` of that same name then resolves
			# through THIS stale entry (fn.get_local_or_raise - see
			# _stmt_Delete) rather than the true pre-loop original, referencing
			# a C-level local that was never actually declared - confirmed by a
			# real repro ("use of undeclared identifier"), the fn.names
			# analogue of hard_restore()'s own identity-divergence fix above.
			names_snapshot = dict( self._current_fn.names )
			try:
				break_narrowed, break_live, continue_captured = self._lower_loop_body(
					body, continue_label = continue_label, break_label = break_label, loop_snapshot = loop_snapshot,
				)
				back_edge_instructions = self._cfg.loop_back_edge( loop_snapshot.bindings, self._current_fn.qualname, entry_results = loop_snapshot.results )
			except CompileError as e:
				promotable = self._cfg.find_promotable_loop_mismatches( loop_snapshot.bindings ) if attempt_number == 1 else set()
				if not promotable:
					self.lowering.discovery.fail( str( e ), node )
				del self._instructions[instructions_mark:]
				del self._defer_flags[defer_flags_mark:]
				del self._for_obj_null_inits[for_obj_null_inits_mark:]
				self._cfg.truncate_cancel_flags( cancel_flags_mark )
				del self._pending_temps[pending_temps_mark:]
				self._cfg.hard_restore( body_snapshot )
				self._current_fn.names.clear()
				self._current_fn.names.update( names_snapshot )
				promo_instructions: list[ir.Instruction] = []
				for promoted_name in sorted( promotable ):
					promo_instructions += self._cfg.promote_borrowed_for_loop( promoted_name )
					loop_snapshot.bindings[promoted_name] = self._cfg.bindings[promoted_name]
				self._instructions[pre_loop_mark:pre_loop_mark] = promo_instructions
				continue
			for instr in back_edge_instructions:
				self._emit( instr )
			self._cfg.restore( loop_snapshot )
			return break_narrowed, break_live, continue_captured
		assert False, 'unreachable' # the attempt_number==2 branch above always either returns or calls discovery.fail (NoReturn)

	def _lower_loop_body( self, body: list[ast.stmt], continue_label: str, break_label: str, loop_snapshot: object ) -> tuple[list[dict[str,list[Variable]]],list[set[str]],bool]:
		self._loop_depth += 1
		ctx = _LoopContext( continue_label = continue_label, break_label = break_label, loop_snapshot = loop_snapshot )
		self._loop_labels.append( ctx )
		# see cfg.py's CFGState.enter_loop's own docstring: lets
		# current_epilogue_label() recognize an RC entry pushed while
		# lowering THIS body as loop-confined (restore(), called once this
		# body's fully lowered, silently drops it - its own label, if a
		# `return` from inside here ever pointed at it, would never
		# actually get emitted)
		self._cfg.enter_loop( loop_snapshot.stack_depth )
		break_narrowed: list[dict[str,list[Variable]]] = []
		break_live: list[set[str]] = []
		try:
			for stmt in body:
				try:
					self._lower_stmt( stmt )
				except CompileError:
					continue
		finally:
			# Phase 8: every narrowed-state/live-state snapshot recorded by a
			# `break` reached while lowering this body (cfg.py's own
			# record_break_narrowed()/record_break_live(), called from
			# _stmt_Break below) - handed back to the caller (_stmt_While/
			# for-loop lowerers) to merge with the loop's own natural exit
			# via merge_loop_exits()
			break_narrowed, break_live = self._cfg.exit_loop()
			self._loop_labels.pop()
			self._loop_depth -= 1
		return break_narrowed, break_live, ctx.continue_captured

	def _check_loop_exit_unchecked_results( self, loop_snapshot: object, node: ast.AST ) -> None:
		try:
			self._cfg.check_loop_exit_unchecked_results( loop_snapshot.results, self._current_fn.qualname )
		except CompileError as e:
			self.lowering.discovery.fail( str( e ), node )

	def _stmt_Break( self, node: ast.Break ) -> None:
		if not self._loop_labels:
			self.lowering.discovery.fail( 'break outside a loop', node )
		ctx = self._loop_labels[-1]
		break_label, loop_snapshot = ctx.break_label, ctx.loop_snapshot
		self._check_loop_exit_unchecked_results( loop_snapshot, node )
		# Phase 8: capture whatever's narrowed RIGHT HERE, at the exact
		# point this break fires - cfg.py's record_break_narrowed() files
		# it under the innermost currently-lowering loop, to be merged
		# with every other break/the loop's own natural exit once that
		# loop's own body is fully lowered. Before unwind_to() below (which
		# doesn't touch _narrowed at all, but ordering it first here keeps
		# this call sitting right next to unwind_to()'s own snapshot read).
		# record_break_live() is the definite-assignment analogue, captured
		# alongside it for the identical reason.
		self._cfg.record_break_narrowed()
		self._cfg.record_break_live()
		for instr in self._cfg.unwind_to( loop_snapshot ):
			self._emit( instr )
		self._emit( ir.Jump( target = break_label ))

	def _stmt_Continue( self, node: ast.Continue ) -> None:
		if not self._loop_labels:
			self.lowering.discovery.fail( 'continue outside a loop', node )
		ctx = self._loop_labels[-1]
		continue_label, loop_snapshot = ctx.continue_label, ctx.loop_snapshot
		ctx.continue_captured = True
		self._check_loop_exit_unchecked_results( loop_snapshot, node )
		for instr in self._cfg.unwind_to( loop_snapshot ):
			self._emit( instr )
		self._emit( ir.Jump( target = continue_label ))

	def _declare_hidden_local( self, stem: str, type: Type, node: ast.AST, *, user_facing: bool = False ) -> Variable:
		# compiler-synthesized locals (for-loop scaffolding: the once-
		# evaluated iterable, its length, the hidden index counter) - real
		# named Variables (not anonymous Temps) registered into the
		# function's flat names dict, the same way `self` gets synthesized
		# in lower_function, so synthetic ast.Name references to them
		# resolve normally through the existing _expr_Name/_stmt_Assign
		# machinery instead of duplicating it. user_facing defaults False -
		# every caller except the fallible for-loop's own bare loop-target
		# rebind passes an already-unique, '__'-prefixed synthesized stem
		# (a per-loop counter baked directly into the name), so it can
		# never collide with itself and needs no del-tracking; that one
		# caller passes the user's own source-level target name, which can
		# (see _mark_fresh_local_declared/needs_uid_suffix's own docstrings)
		fn = self._current_fn
		needs_uid_suffix = self._mark_fresh_local_declared( stem ) if user_facing else False
		var = Variable( stem = stem, qualname = f'{fn.qualname}.{stem}', file = fn.file, line = getattr( node, 'lineno', None ), type = type, needs_uid_suffix = needs_uid_suffix )
		fn.add_name( stem, var )
		# every caller unconditionally assigns this right after declaring it
		# (no user code runs in between - see each call site's own next
		# line/statement), so it's live from here on, same bucket as a
		# parameter - see cfg.py's mark_live() docstring
		self._cfg.mark_live( stem )
		self.lowering.schedule( type )
		return var

	def _maybe_consume_result( self, node: ast.AST, value: ir.Temp, alternatives: str ) -> ir.Operand:
		# if `value` is itself a Result[T,E], auto-consume it - unlike
		# _lower_or_return, a non-Result value is passed through unchanged
		# rather than rejected, since not every method this is used for
		# (__getitem__, __len__) is necessarily fallible. The old panic_errmsg
		# carve-out (Unwrap-panic instead of or_throw(), for a for-loop's own
		# compiler-proven-safe bounds-checked element read) was removed along
		# with the indexable for-loop lowering path it was the only caller of
		# - see b1fc7f9 "Unify for-loop dispatch to strict IteratorProtocol[T]/
		# Iterable[T] conformance"; for-loops no longer synthesize a raw
		# __getitem__ call needing this, so nothing calls this with a panic
		# message anymore
		result = self._auto_or_throw( node, value, alternatives, want_result = True )
		assert result is not None # want_result=True above guarantees this
		return result

	def _reject_unconsumed_result_operand( self, node: ast.AST, operand: ir.Operand ) -> ir.Operand:
		# an unconsumed Result[T,E] used directly as a binop/comparison
		# operand is (unlike __len__/__getitem__ above) never legitimate as-
		# is - Result is itself a @union, so left unchecked it would
		# silently fall into the ordinary union leaf-pair dispatch and get
		# decomposed into per-leaf (T, E) arithmetic/comparison instead of
		# erroring. Auto-.or_throw()s it instead of hard-rejecting - the
		# general case-1/case-2 rule (see _auto_or_throw) applies here too:
		# `(a + b) + c` needs a + b's own leftover Result auto-consumed
		# before it can become the outer +'s own left operand.
		result = self._auto_or_throw( node, operand, self.lowering._AUTO_CONSUME_ALTERNATIVES, want_result = True )
		assert result is not None # want_result=True above guarantees this
		return result

	def _bind_loop_target( self, target: ast.Name, default_type: Type, value_expr: ast.expr, node: ast.AST ) -> Variable:
		# mirrors _stmt_Assign's Name-target "reuse existing, else infer/
		# declare" rule (`for i in range(count):` reuses `i` if a variable
		# of that name already exists - e.g. str.concat in lib/builtins/
		# __init__.py pre-declares `i: usize = 0` before its own for loop)
		# - except a fresh declaration falls back to `default_type` instead
		# of failing outright, since value_expr may be a bare literal
		# (range()'s implicit start=0) with no type of its own to infer from
		existing = self._existing_local_or_none( target.id, node, 'cannot use it as a for loop target' )
		if existing is not None:
			self.lowering._ensure_resolved( existing ) # see _stmt_Assign's identical call for why
			operand = self._lower_expr( value_expr, existing.type )
			self._emit( ir.Assign( dest = existing, src = operand ))
			self._cfg.mark_live( existing.stem ) # this bypasses _cfg_assign like the rest of this function does (pre-existing, not touched here) - liveness alone still needs marking, since it's unconditionally assigned right here regardless
			return existing
		var, operand = self._declare_local( target.id, node, lambda expected: self._lower_expr( value_expr, expected ), default_type = default_type )
		self._emit( ir.Assign( dest = var, src = operand ))
		self._cfg.mark_live( var.stem )
		return var

	def _stmt_For( self, node: ast.For ) -> None:
		if not isinstance( node.target, ast.Name ):
			# a Tuple target (`for a, b in EXPR:`) is desugared away by
			# type_resolver.py's _ReferenceResolver.visit_For long before
			# this ever runs - reaching here with one would mean that pass
			# was skipped somehow, so this stays a hard failure rather than
			# silently re-attempting the same desugar. NOT ast.unparse(node)
			# - that would dump this for-loop's entire BODY into the message
			# (a real repro: a multi-statement for-body produced a
			# multi-line error that buried the actual problem)
			self.lowering.discovery.fail( f'for loop target must be a plain name: {ast.unparse(node.target)}', node )
		if node.orelse:
			self.lowering.discovery.fail( 'for/else is not supported', node )
		if self.lowering._is_range_call( node.iter ) is not None:
			self._lower_for_range( node )
			return
		# PLAN_GENERATORS.md Phase 3 - node.iter is lowered exactly ONCE here
		# (not once per candidate path) and the resulting operand handed to
		# whichever real consumption path applies, so an iterable expression
		# with a side effect (most commonly: a generator CONSTRUCTOR call)
		# is never evaluated twice - _lower_for_over_iterator's own obj
		# parameter is always this already-lowered operand, never re-derived
		# from node.iter itself.
		#
		# Strict IteratorProtocol[T]/Iterable[T] protocol dispatch, no
		# structural duck-typing (confirmed with the user - a for-loop
		# subject must DECLARE conformance, not merely happen to have
		# matching method names): an object that's ALREADY an iterator (a
		# real generator, or any hand-written IteratorProtocol[T] conformer)
		# drives its own __next__() directly, unchanged from before. An
		# Iterable[T] conformer (list/
		# set/tuple/VariadicTuple/dict/...) gets its __iter__() called ONCE
		# here to obtain a real iterator, which then drives the exact same
		# __next__()-consumption path - "iter() on an iterator returns
		# itself" is subsumed by the FIRST branch already handling a
		# self-iterating object without ever needing to call __iter__() on
		# it at all, rather than needing every IteratorProtocol[T] conformer
		# to also separately implement Iterable[T].__iter__ returning self.
		#
		# This is a real, confirmed correctness fix, not just stricter
		# typing: the old __len__/__getitem__(usize) duck-typing this
		# replaced would have silently misused dict[K,V]'s own __getitem__
		# (key: K) as if it were positional 0..len indexing for `for k in
		# some_dict:` whenever K happened to be usize-compatible - dict
		# never declares __next__ directly, so it always fell to that path.
		obj = self._lower_expr( node.iter, None )
		iterator_protocol = self.lowering.discovery.find_name( 'IteratorProtocol', node )
		iterable_protocol = self.lowering.discovery.find_name( 'Iterable', node )
		if self.lowering._type_conforms_to_protocol( obj.type, iterator_protocol ):
			next_fn = self.lowering._find_iterator_next_method( obj.type )
			assert next_fn is not None, 'internal compiler error: IteratorProtocol[T] conformance declared without a real __next__'
			self._lower_for_over_iterator( node, obj, next_fn )
		elif self.lowering._type_conforms_to_protocol( obj.type, iterable_protocol ):
			iter_fn = self.lowering._find_method( obj.type, '__iter__' )
			assert iter_fn is not None, 'internal compiler error: Iterable[T] conformance declared without a real __iter__'
			self.lowering._ensure_resolved( iter_fn )
			# __iter__'s own declared return type (Generator[T,StopIteration])
			# is still the bare, unresolved GeneratorType annotation until
			# its own passthrough body ("return _sequence_iter(self)"-shaped)
			# gets synthesized - mirrors lower_function's own identical
			# safety-net call for an ordinary generator reached with no
			# earlier caller (PLAN_GENERATORS.md).
			self.lowering._type_resolver.ensure_generator_synthesized( iter_fn )
			self.lowering.schedule( iter_fn.return_type )
			iterator_dest = self._new_temp( iter_fn.return_type )
			self._emit( ir.Call( dest = iterator_dest, target = iter_fn, receiver = obj, args = [], kwargs = {} ))
			# obj's own receiver use ends right here - __iter__() built its
			# own independently-retained iterator (e.g. Part B's own retain-
			# on-read for a chained field receiver, `self.a.b`, or a fresh
			# Call/Allocate result), so if obj was itself a genuinely fresh/
			# owned temp, it needs releasing NOW, not deferred to the natural
			# end of this statement (see _lower_for_over_iterator's own
			# identical reasoning for iterator_dest below) - the loop body
			# below can return/break early, well before this statement's own
			# generic end-of-statement flush would ever run. A bare aliasing
			# read (obj already borrowed from an existing binding, e.g. `for
			# x in some_local_iterable:`) is unaffected: is_fresh_temp is
			# False there, so this is a no-op, matching its existing (already
			# correct) behavior of never releasing a borrowed reference.
			if self._cfg.is_fresh_temp( obj ):
				for instr in self._cfg.delete_temp( obj ):
					self._emit( instr )
				self._emit( ir.DeleteTemp( temp = obj ))
			next_fn = self.lowering._find_iterator_next_method( iterator_dest.type )
			if next_fn is None:
				self.lowering.discovery.fail(
					f'{obj.type.qualname if obj.type else "?"}.__iter__() returned '
					f'{iterator_dest.type.qualname if iterator_dest.type else "?"}, which does not conform to '
					f'IteratorProtocol[T] (missing __next__): {ast.unparse(node.iter)}',
					node,
				)
			self._lower_for_over_iterator( node, iterator_dest, next_fn )
		else:
			self.lowering.discovery.fail(
				f'for loop requires an IteratorProtocol[T] or Iterable[T] conformer, got '
				f'{obj.type.qualname if obj.type else "?"}: {ast.unparse(node.iter)}',
				node,
			)

	def _lower_for_range( self, node: ast.For ) -> None:
		call = node.iter
		if call.keywords:
			self.lowering.discovery.fail( f'range(...) does not support keyword arguments: {ast.unparse(call)}', call )
		if len( call.args ) == 1:
			start_expr = ast.Constant( value = 0 )
			ast.copy_location( start_expr, call )
			stop_expr = call.args[0]
		elif len( call.args ) == 2:
			start_expr, stop_expr = call.args
		else:
			self.lowering.discovery.fail( f'range(...) supports 1 or 2 arguments only (no step yet): {ast.unparse(call)}', call )

		usize_cls = self.lowering.discovery.get_intrinsics()['usize']
		bool_cls = self.lowering.discovery.find_name( 'bool', node )

		target_var = self._bind_loop_target( node.target, usize_cls, start_expr, node )

		stop_operand = self._lower_expr( stop_expr, usize_cls )
		stop_var = self._declare_hidden_local( f'__for_stop_{self._label_id}', usize_cls, node )
		self._emit( ir.Assign( dest = stop_var, src = stop_operand ))

		pre_loop_mark = len( self._instructions ) # see _lower_loop_body_with_ownership_retry's own docstring - a promoted name's one-time incref splices in here, strictly before start_label
		start_label = self._new_label( 'for_start' )
		continue_label = self._new_label( 'for_continue' )
		end_label = self._new_label( 'for_end' )

		self._emit( ir.Label( name = start_label ))
		test = ast.Compare( left = self.lowering._synth_name( target_var.stem, node ), ops = [ ast.Lt() ], comparators = [ self.lowering._synth_name( stop_var.stem, node ) ] )
		ast.copy_location( test, node )
		cond = self._lower_expr( test, bool_cls )
		self._emit( ir.JumpIfFalse( cond = cond, target = end_label ))

		loop_snapshot = self._cfg.snapshot()
		break_narrowed, break_live, continue_captured = self._lower_loop_body_with_ownership_retry(
			node, node.body, continue_label = continue_label, break_label = end_label, loop_snapshot = loop_snapshot, pre_loop_mark = pre_loop_mark,
		)
		# Phase 8: a for-loop has no type(x) is T condition of its own to
		# narrow FROM, but its natural exit (the range simply exhausted,
		# including never having run the body at all - always reachable
		# for any for-loop) is still a real candidate to reconcile against
		# every break_narrowed collected above - same merge_loop_exits()
		# used by _stmt_While
		self._cfg.merge_loop_exits( dict( loop_snapshot.narrowed ), break_narrowed, set( loop_snapshot.live ), break_live )

		# continue_label is purely a fallthrough landing (the increment
		# below) unless some `continue` in the body actually jumped to it -
		# an un-goto'd label triggers -Wunused-label/C4102
		if continue_captured:
			self._emit( ir.Label( name = continue_label ))
		# the increment is a compiler-synthesized implementation detail of
		# the loop, not user-written arithmetic - it's structurally
		# guaranteed safe (target_var < stop_var strictly before every
		# increment), so it bypasses the ambient arithmetic-mode policy
		# entirely (AddWrap directly) rather than imposing a
		# Result[_,OverflowError]/wrap_arithmetic/etc. requirement on
		# ordinary for-loops
		incr = self._new_temp( usize_cls )
		self._emit( ir.AddWrap( dest = incr, left = target_var, right = ir.Const( type = usize_cls, value = 1 ) ))
		self._emit( ir.Assign( dest = target_var, src = incr ))
		self._emit( ir.Jump( target = start_label ))
		self._emit( ir.Label( name = end_label ))

	def _lower_binary_branch( self, cond: ir.Operand, node: ast.AST, true_thunk: 'Callable[[],bool]', false_thunk: 'Callable[[],bool]' ) -> None:
		''' hand-rolled version of _stmt_If's own branch-then-merge
		machinery (JumpIfFalse/snapshot/enter_branch/exit_branch/merge_if/
		splice), for a caller building its own condition and branch bodies
		directly at the IR level rather than lowering a real ast.If -
		_lower_for_over_iterator's own Result[T,E]/Err(StopIteration)-vs-
		real-error dispatch, which has no ast.If to lower (the condition is
		a synthesized tag comparison, and each branch's own body is a
		narrow()+bind pair, not user-written statements).

		Each thunk is called with no arguments inside its own branch-
		confined region (self._cfg.enter_branch/exit_branch, self.
		_instructions redirected to a captured list - identical setup
		_stmt_If uses for node.body/node.orelse), expected to emit
		whatever IR it needs via self._emit/self._cfg directly, and return
		True if it's a dead end that never reaches the merge point (a raw
		jump elsewhere - the same "terminates" concept _stmt_If's own
		true_terminates/false_terminates track for a branch ending in
		return/break/continue, just decided by the thunk itself instead of
		_stmt_diverges since there's no real AST statement to inspect). '''
		else_label = self._new_label( 'branch_else' )
		self._emit( ir.JumpIfFalse( cond = cond, target = else_label ))

		entry_snapshot = self._cfg.snapshot()
		outer_instructions = self._instructions
		self._instructions = []
		# same entry_snapshot/restore() shape _stmt_If/_stmt_Try use for
		# their own sibling branches - a thunk that manually decrefs an
		# entry declared BEFORE this call (e.g. or_throw(mapper)'s own
		# err_thunk explicitly compiler.decref()ing the receiver, since a
		# GOTO-based ir.Raise never unwinds anything on its own) is safe
		# without any special protection: cfg.py's own _neutralize() never
		# mutates a shared Epilogue object in place, so the OTHER thunk -
		# restored back to this same entry snapshot - still sees the
		# untouched original.
		self._cfg.enter_branch( entry_snapshot.stack_depth )
		true_temps_start = len( self._pending_temps )
		try:
			true_terminates = true_thunk()
		finally:
			self._cfg.exit_branch()
		# each thunk's own PURELY INTERMEDIATE temps (e.g. leaf_bind_thunk's
		# E'-union coercion wrapper, built to pass a narrowed leaf value into
		# Result.Err(e: E')) must be flushed HERE, still inside this branch's
		# own captured instruction list - _flush_branch_temps' own comment
		# documents the exact same crash class this recreates otherwise:
		# left pending, they'd survive into the ENCLOSING statement's single
		# unconditional end-of-statement flush (this method builds raw IR,
		# never routes a thunk's own statements through _lower_stmt, so nothing
		# else ever flushes them) and get decref'd there even for whichever
		# branch never ran at runtime - reading tag/payload data off an
		# uninitialized C local (confirmed via a real crash: multi_leaf-style
		# two-leaf error dispatch, MSVC access violation)
		self._flush_branch_temps( true_temps_start )
		true_captured = self._instructions
		true_end = dict( self._cfg.bindings )
		true_end_results = self._cfg.unchecked_results()
		true_end_narrowed = self._cfg.narrowed_snapshot()
		true_end_live = self._cfg.live_snapshot()

		self._cfg.restore( entry_snapshot )
		self._instructions = []
		self._cfg.enter_branch( entry_snapshot.stack_depth )
		false_temps_start = len( self._pending_temps )
		try:
			false_terminates = false_thunk()
		finally:
			self._cfg.exit_branch()
		self._flush_branch_temps( false_temps_start )
		false_captured = self._instructions
		false_end = dict( self._cfg.bindings )
		false_end_results = self._cfg.unchecked_results()
		false_end_narrowed = self._cfg.narrowed_snapshot()
		false_end_live = self._cfg.live_snapshot()

		self._cfg.restore( entry_snapshot )
		self._instructions = outer_instructions
		try:
			true_extra, false_extra, removed = self._cfg.merge_if(
				entry_snapshot.bindings, true_end, false_end, self._current_fn.qualname,
				entry_results = entry_snapshot.results, true_end_results = true_end_results, false_end_results = false_end_results,
				true_terminates = true_terminates, false_terminates = false_terminates,
				true_end_narrowed = true_end_narrowed, false_end_narrowed = false_end_narrowed,
				true_end_live = true_end_live, false_end_live = false_end_live,
			)
		except CompileError as e:
			self.lowering.discovery.fail( str( e ), node )

		for instr in true_captured:
			self._emit_captured( instr )
		for instr in true_extra:
			self._emit( instr )
		end_label = self._new_label( 'branch_end' )
		self._emit( ir.Jump( target = end_label ))
		self._emit( ir.Label( name = else_label ))
		for instr in false_captured:
			self._emit_captured( instr )
		for instr in false_extra:
			self._emit( instr )
		self._emit( ir.Label( name = end_label ))

	def _lower_for_over_iterator( self, node: ast.For, obj: ir.Operand, next_fn: Function ) -> None:
		''' PLAN_GENERATORS.md's StopIteration reversal - `for x in <expr
		with a __next__() returning Result[T,E]>:` (E always including
		StopIteration). Structurally the same shape as _lower_for_over_
		indexable (once-evaluated iterable, start/continue/end labels,
		snapshot-bind-lower_loop_body-back_edge-restore-merge), except the
		"is there another element" test is __next__()'s own Result[T,E]
		tag rather than an index/length comparison, and the element
		binding needs the union's payload extracted rather than a plain
		__getitem__ call.

		Binding rule (matches type_resolver.py's own _desugar_iterator_for,
		the in-generator-body mirror of this method - see its own docstring
		for the full reasoning, confirmed directly with the user): if E is
		JUST StopIteration, x binds to plain T. If E has any OTHER error,
		x binds to Result[T,E'] (E' = E minus StopIteration) - the caller
		handles the real error explicitly inside the loop body (match/
		.is_err()/.or_return()/.unwrap()), no auto-propagation. Unlike the
		in-generator-body desugaring (which builds fresh AST processed by a
		LATER type-checking/desugaring pass with its own narrowing
		machinery), this method lowers directly to IR - constructing a
		NARROWER Result[T,E'] value from a payload read out of the WIDER
		Result[T,E] needs a genuine N-way tag dispatch per E' leaf (cfg.py's
		narrow() only yields a concrete single type when exactly one
		candidate remains - Result.Err(e)'s own generic T/E inference
		disagrees between the assignment target's declared E' and e's own
		un-narrowed E otherwise, confirmed via a real repro identical to
		type_resolver.py's own _desugar_iterator_for hitting the same
		trap), built here via _lower_binary_branch - a hand-rolled
		_stmt_If-style branch-then-merge helper for exactly this situation
		(a condition + branch bodies with no real ast.If to lower).

		The payload extraction reuses cfg.py's REAL narrowing mechanism
		(narrow()/narrowed_member(), the same machinery a `match x: case
		T(x):` arm - reusing the subject's own name - already relies on;
		confirmed working via a standalone repro, since a plain `if x is
		None: ... else: ...`-narrowed read does NOT currently work anywhere
		in this compiler, generator-unrelated - see PLAN_GENERATORS.md's own
		STATUS section) directly at the IR level: __next__()'s result is
		bound into a hidden local, narrow()'d to the union's own Ok leaf,
		then read back through node.target's own ordinary _stmt_Assign -
		_expr_Name's existing narrowed-read branch does the rest (extracts
		through GetAttr(data)/GetAttr(v_<leaf>) automatically), no new
		extraction code needed here at all.

		The "was this an error" test can't be spelled `x.is_err()` in the
		synthesized AST the way user source would (that's a real method
		call, more machinery than needed) - the same direct .tag comparison
		this method has always built by hand (see its own historical
		docstring on why is/is-not-None doesn't work here either - the same
		reasoning applies to any comparison against a TaggedUnion tag). '''
		self.lowering._ensure_resolved( next_fn )
		result_type = next_fn.return_type
		shape = self.lowering._type_resolver._result_shape( result_type )
		stop_iteration_cls = self.lowering.discovery.find_name_or_none( 'StopIteration' )
		if shape is None or stop_iteration_cls is None or stop_iteration_cls not in self.lowering._type_resolver._atomic_leaves( shape[1] ):
			self.lowering.discovery.fail(
				f'for loop needs __next__() to return Result[T,E] (E including StopIteration) on '
				f'{obj.type.qualname if obj.type else "?"}: {ast.unparse(node.iter)}',
				node,
			)
		elem_type, full_error_type = shape
		remaining_leaves = [ leaf for leaf in self.lowering._type_resolver._atomic_leaves( full_error_type ) if leaf is not stop_iteration_cls ]
		remaining_error_type: Type|None
		if not remaining_leaves:
			remaining_error_type = None
		elif len( remaining_leaves ) == 1:
			remaining_error_type = remaining_leaves[0]
		else:
			remaining_error_type = self.lowering.discovery._get_or_create_union( remaining_leaves )
		self.lowering.schedule( result_type )
		# result_type is routinely a Result[T,E] SPECIALIZATION, not a bare
		# TaggedUnion - _tagged_union_shape gives back the abstract base's
		# own member list (substituted for THIS instantiation), and _union_
		# storage.get needs a monomorphized concrete union (the abstract
		# Result class's own payload union has no real C definition, bare
		# unsubstituted T/E TypeVars) - same "monomorphize_class if
		# Specialization else itself" pattern _emit_binop_fallible_check's
		# own identical situation already uses (see its own comment)
		tagged_shape = self.lowering._type_resolver._tagged_union_shape( result_type )
		assert tagged_shape is not None
		_result_base, result_members = tagged_shape
		err_member = next( a for a in result_members if a.stem == 'Err' )
		ok_member = next( a for a in result_members if a.stem == 'Ok' )
		concrete_result_union = self.lowering.monomorphize_class( result_type ) if isinstance( result_type, Specialization ) else result_type
		tag_attr, _data_attr, _payload_cls, tags = self.lowering._union_storage.get( concrete_result_union )
		bool_cls = self.lowering.discovery.find_name( 'bool', node )

		unique = self._label_id
		obj_var = self._declare_hidden_local( f'__for_obj_{unique}', obj.type, node )
		self._emit( ir.Assign( dest = obj_var, src = obj ))
		# obj_var holds the iterator this loop drives via __next__() for its
		# ENTIRE body, unlike _stmt_If/_stmt_While's own condition temps
		# (needed only up to the branch point) - it can't be released the
		# instant it's bound, only once the loop truly ends, on EVERY exit
		# path: normal exhaustion, break (both converge at end_label below),
		# and an early return/continue-out-of-an-enclosing-construct reached
		# from inside the loop body, which skips end_label entirely. Only a
		# genuinely fresh/owned obj (a real Call/Allocate result, or Part B's
		# own retain-on-read for a chained field receiver, `self.a.b`) needs
		# this at all - a bare aliasing read (`for x in some_local_iterator:`)
		# never took an extra reference in the first place, so obj_var stays
		# a plain, unreleased borrow, matching this function's existing
		# (correct) behavior for that case. untrack_temp here hands the
		# release responsibility to the registered defer below instead of
		# leaving it as an ordinary pending temp - this statement's own
		# generic end-of-statement flush would otherwise still try to
		# release it too, once the defer already has (a double-free) or,
		# worse, only ever release it there at all, which is unreachable
		# dead code on any early-return exit (the exact bug this mirrors
		# from _stmt_If's own identical fix).
		def _make_obj_decref_stmt() -> ast.stmt:
			# a FRESH node every call, never reused across the two sites this
			# is called from (defer registration below, and the disarm-and-
			# call-directly site right after end_label) - mirrors
			# _lower_with_context_manager's own identical _make_exit_stmt()
			# and its own comment on why
			call = ast.Call(
				func = ast.Attribute( value = ast.Name( id = 'compiler', ctx = ast.Load() ), attr = '__internal_decref__', ctx = ast.Load() ),
				args = [ self.lowering._synth_name( obj_var.stem, node ) ], keywords = [],
			)
			ast.fix_missing_locations( ast.copy_location( call, node ))
			stmt = ast.Expr( value = call )
			ast.fix_missing_locations( ast.copy_location( stmt, node ))
			return stmt

		obj_needs_release = self._cfg.is_fresh_temp( obj )
		obj_release_entry: cfg.Epilogue|None = None
		if obj_needs_release:
			self._cfg.untrack_temp( obj )
			obj_release_entry = self._register_defer_block( is_err_only = False, body = [ _make_obj_decref_stmt() ], node = node, allow_inside_loop = True )
			self._for_obj_null_inits.append( obj_var )
		next_var = self._declare_hidden_local( f'__for_next_{unique}', result_type, node )

		if remaining_error_type is not None:
			# the loop target's own type is now Result[elem_type,
			# remaining_error_type], not bare elem_type - pre-declared
			# EXPLICITLY (rather than left to _stmt_Assign's own "first
			# assignment infers the type from the RHS" path, which the
			# bare-T case below relies on) because NEITHER Result.Ok(v)
			# nor Result.Err(e) can infer their own OTHER type parameter
			# from their single argument alone (Ok's own call never
			# mentions E at all; Err's never mentions T) - needs an
			# expected_type to resolve against, which _stmt_Assign only
			# supplies for an ALREADY-declared target (existing.type) -
			# confirmed via a real repro otherwise ("type parameter E is
			# inferred as both remaining_error_type and full_error_type").
			result_cls = self.lowering.discovery.find_name( 'Result', node )
			target_type = self.lowering.discovery._get_or_create_specialization( result_cls, [ elem_type, remaining_error_type ] )
			self._declare_hidden_local( node.target.id, target_type, node, user_facing = True )

		pre_loop_mark = len( self._instructions ) # see _lower_loop_body_with_ownership_retry's own docstring - a promoted name's one-time incref splices in here, strictly before start_label
		start_label = self._new_label( 'for_start' )
		continue_label = self._new_label( 'for_continue' )
		end_label = self._new_label( 'for_end' )

		self._emit( ir.Label( name = start_label ))
		# snapshot at the VERY TOP of the loop, before next_var's own
		# per-iteration rebind AND before the loop target's own binding -
		# both bindings are fresh every iteration, not confined-and-torn-
		# down across it - taken here, before ANY of that, so loop_back_
		# edge()'s later reconciliation sees next_var (like the target) as
		# absent from entry/present at the back edge and decrefs its stale
		# value right before jumping back to start_label. A raw ir.Assign
		# for next_var (the original shape here) would never decref what it
		# held from the PRIOR iteration before overwriting it - leaking one
		# reference per iteration for any RC-typed elem_type (confirmed via
		# a real multi-iteration compiler.refcount() repro). Tried gating
		# next_var's own decref through a plain cfg.assign() call instead of
		# loop_snapshot placement first - doesn't work: cfg.assign() only
		# emits a decref-of-the-old-value when a binding ALREADY exists in
		# self.bindings, but next_var's assignment here is lowered exactly
		# ONCE at compile time (this call happens on every RUNTIME
		# iteration via the goto back-edge, but cfg only ever sees it as the
		# textually-first assignment) - loop_back_edge() is the mechanism
		# actually built for "this binding's value is torn down and
		# replaced every iteration", not a second compile-time assign() call
		loop_snapshot = self._cfg.snapshot()
		next_dest = self._new_temp( result_type )
		self._emit( ir.Call( dest = next_dest, target = next_fn, receiver = obj_var, args = [], kwargs = {} ))
		# track_result=False: next_var's own Result-ness is scaffolding
		# inspected via the raw .tag comparison below, not is_ok()/is_err()/
		# match - same reasoning as the match-subject temp's own call site
		for instr in self._cfg.assign( next_var, next_dest, is_alias = False, track_result = False ):
			self._emit( instr )
		self._emit( ir.Assign( dest = next_var, src = next_dest ))

		tag_expr = ast.Attribute( value = self.lowering._synth_name( next_var.stem, node ), attr = tag_attr.stem, ctx = ast.Load() )
		ast.copy_location( tag_expr, node )
		is_err_test = ast.Compare( left = tag_expr, ops = [ ast.Eq() ], comparators = [ ast.Constant( value = tags[ err_member.stem ] ) ] )
		ast.copy_location( is_err_test, node )
		is_err_cond = self._lower_expr( is_err_test, bool_cls )

		# only set (and only reachable) in the remaining_error_type is None
		# branch below - see its own comment at the bottom of this method
		# for why this exit needs its own dedicated label rather than
		# jumping straight to end_label the way it used to
		stop_label: str|None = None
		if remaining_error_type is None:
			stop_label = self._new_label( 'for_stop' )
			self._emit( ir.JumpIfTrue( cond = is_err_cond, target = stop_label ))
			self._cfg.narrow( next_var.stem, ok_member )
			bind = ast.Assign( targets = [ node.target ], value = self.lowering._synth_name( next_var.stem, node ))
			ast.copy_location( bind, node )
			self._stmt_Assign( bind )
		else:
			self._lower_for_over_iterator_fallible_bind(
				node, next_var, err_member, ok_member, is_err_cond, elem_type, full_error_type,
				remaining_leaves, remaining_error_type, stop_iteration_cls, end_label, unique,
			)

		break_narrowed, break_live, continue_captured = self._lower_loop_body_with_ownership_retry(
			node, node.body, continue_label = continue_label, break_label = end_label, loop_snapshot = loop_snapshot, pre_loop_mark = pre_loop_mark,
		)
		# Phase 8 - see _lower_for_range's own identical call/comment
		self._cfg.merge_loop_exits( dict( loop_snapshot.narrowed ), break_narrowed, set( loop_snapshot.live ), break_live )

		# see _lower_for_range's own identical comment on continue_captured
		if continue_captured:
			self._emit( ir.Label( name = continue_label ))
		self._emit( ir.Jump( target = start_label ))
		if stop_label is not None:
			self._emit( ir.Label( name = stop_label ))
			# next_var holds the Err/StopIteration payload __next__() just
			# returned - unlike `break` (which already releases next_var's
			# CURRENT payload itself, via its own existing branch-confined-
			# binding cleanup - see _stmt_Break) this JumpIfTrue above is a
			# bare, hand-emitted jump with none of that machinery behind it,
			# so nothing else ever releases it. The ORDINARY per-iteration
			# release (right before looping back to start_label, paired with
			# next_var's own assignment above) only runs on the NORMAL
			# continuation path - this exit skips straight past it. next_var's
			# own cfg-tracked binding (from cfg.assign() above) is no help
			# either: restore(loop_snapshot) below already drops it back to
			# its pre-loop (nonexistent) state, same as any other loop-
			# confined binding - so this can't reuse that tracking, it has to
			# be a bare, tag-dispatched release keyed on next_var's OWN
			# declared type, exactly like compiler.decref(x)'s own "bare
			# Name" path. A dedicated label (not just falling into end_label,
			# which break ALSO targets) is required specifically so this
			# release never runs on break's own path too - it already has
			# its own, and this would double-release the SAME payload
			# otherwise (confirmed via a real repro: an RC-typed element
			# still `for`-in-progress when `break` fires came back under-
			# counted, not over, until this was scoped to just this label).
			# Confirmed as a real leak via AddressSanitizer for the
			# StopIteration case: a fresh StopIteration object allocated
			# every time a for-loop's iterator is exhausted, permanently
			# unreachable afterward - found only once the pending-temp
			# fixes elsewhere in this method stopped masking it.
			#
			# A NARROWED extraction (GetAttr(data)+GetAttr(v_<Err member>)),
			# not a general cfg.decref(result_type, next_var) - stop_label is
			# ONLY reached via is_err_cond being true, so the tag is
			# statically known Err here; a general tag-dispatched decref
			# would still emit a genuinely dead v_Ok GetAttr+Decref pair
			# alongside the real v_Err one (unreachable at runtime, since
			# this label's only predecessor already proved the tag isn't Ok -
			# but still emitted, unconditionally, as part of decref()'s own
			# generic per-leaf dispatch). That dead pair collided with
			# lowering_test.py's own test_for_over_indexable_rc_element_
			# decref_stays_inside_loop_body, which (rightly) asserts every
			# v_Ok Decref appears before the loop's back-edge Jump, to guard
			# against a DIFFERENT, previously-fixed bug (the per-iteration
			# element decref only firing once, after the whole loop, instead
			# of every iteration) - this narrowed version simply never emits
			# a v_Ok arm at all, so there's nothing to collide with, and it's
			# less code besides.
			data_dest = self._new_temp( _payload_cls )
			self._emit( ir.GetAttr( dest = data_dest, obj = next_var, attr = _data_attr.stem ))
			err_dest = self._new_temp( err_member.type )
			self._emit( ir.GetAttr( dest = err_dest, obj = data_dest, attr = f'v_{err_member.stem}' ))
			for instr in self._cfg.decref( err_member.type, err_dest ):
				self._emit( instr )
		# end_label is jumped to unconditionally from
		# _lower_for_over_iterator_fallible_bind's own StopIteration exit
		# (stop_label is None here - that helper ran instead, see its own
		# comment) but, when stop_label WAS used, only break ever targets
		# end_label (the natural exhaustion exit already fell straight
		# through, right above) - a body with no break then leaves it a
		# goto-less label, -Wunused-label/C4102 (confirmed via a real
		# repro: any `for x in <list/str/...>:` with no break inside)
		if stop_label is None or break_narrowed:
			self._emit( ir.Label( name = end_label ))
		if obj_needs_release:
			# every non-early-return exit converges here (normal exhaustion
			# via stop_label above, and break via break_label=end_label) -
			# obj_var's own lexical extent ends exactly at this label, so
			# disarm the registered defer and release it directly, right
			# here, matching _lower_with_context_manager's own identical
			# "disarm + direct call" pattern (see its own comment for why:
			# without this, obj_var would leak past the for-statement's own
			# end and only ever get released later, at the function's own
			# eventual return/epilogue - too broad, not "when this loop
			# ends"). An early return reached from inside the loop body
			# never reaches this point at all - that path is exactly what
			# the still-armed defer itself covers, replayed by _stmt_Return.
			assert obj_release_entry is not None
			for instr in self._cfg.disarm_defer( obj_release_entry ):
				self._emit( instr )
			self._lower_stmt( _make_obj_decref_stmt() )

	def _lower_for_over_iterator_fallible_bind(
		self, node: ast.For, next_var: Variable, err_member: Variable, ok_member: Variable, is_err_cond: ir.Operand,
		elem_type: Type, full_error_type: Type, remaining_leaves: list[Type], remaining_error_type: Type,
		stop_iteration_cls: Type, end_label: str, unique: int,
	) -> None:
		''' _lower_for_over_iterator's own Result[T,E'] loop-target binding
		(E' = the generator's declared error type minus StopIteration) -
		split out into its own method purely for readability, not reused
		elsewhere. node.target's own type (Result[elem_type,remaining_
		error_type]) is already pre-declared by the caller before this
		runs (see its own comment on why - neither Result.Ok(v) nor
		Result.Err(e) can infer their own OTHER type parameter alone).

		Three-way outcome (Ok / Err-StopIteration / Err-real-error), but
		StopIteration is a pure early exit (jumps straight to end_label,
		contributing nothing to the loop body's own entry state) - so this
		is built as ONE binary branch (Ok vs Err) via _lower_binary_branch,
		with the Err side recursing into its OWN binary branch (StopIteration-
		exit vs real-error-bind), and the real-error side recursing into a
		right-nested CHAIN of one more binary branch PER remaining leaf
		(dispatch, below) when there's more than one - each arm narrow()s
		the extracted error to that ONE concrete leaf (cfg.py's narrow()
		only yields a concrete type when exactly one candidate remains),
		needed for Result.Err(e)'s own generic inference to resolve E
		correctly (matches type_resolver.py's own _desugar_iterator_for,
		which hit the identical trap with a single wildcard arm instead of
		one explicit arm per leaf - see its own comment). '''
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		if full_error_type.resolve is not None:
			full_error_type.resolve()
		for attr in full_error_type.attributes:
			if attr.resolve is not None:
				attr.resolve()
		e_tag_attr, _e_data_attr, _e_payload_cls, e_tags = self.lowering._union_storage.get( full_error_type )
		# identity-keyed, not by value - Type dataclasses aren't all
		# hashable (e.g. RCClass), and every leaf here is interned anyway
		# (the same _atomic_leaves-derived Type object backs both this
		# union's own member and remaining_leaves/stop_iteration_cls)
		e_members = { id( m.type ): m for m in full_error_type.attributes }

		def rewrap_bind( ctor_attr: str, bind_name: str, payload_type: Type ) -> None:
			rewrap = ast.Call(
				func = ast.Attribute( value = ast.Name( id = 'Result', ctx = ast.Load() ), attr = ctor_attr, ctx = ast.Load() ),
				args = [ ast.Name( id = bind_name, ctx = ast.Load() ) ], keywords = [],
			)
			ast.copy_location( rewrap, node ); ast.copy_location( rewrap.func, node ); ast.copy_location( rewrap.func.value, node )
			bind = ast.Assign( targets = [ node.target ], value = rewrap )
			ast.copy_location( bind, node )
			self._stmt_Assign( bind )
			if payload_type.is_rc():
				# bind_name's own extraction (above) is an aliasing-read
				# incref, and Result.Ok(...)/Result.Err(...) increfs its own
				# argument AGAIN when wrapping it - bind_name's own
				# extracted reference is never otherwise consumed (node.
				# target holds the REWRAPPED value's own, independent
				# reference), so without this it's a real per-iteration
				# leak - confirmed via a real repro, matches type_resolver.
				# py's _desugar_iterator_for's identical situation (its own
				# comment has the full reasoning)
				decref_call = ast.Expr( value = ast.Call(
					func = ast.Attribute( value = ast.Name( id = 'compiler', ctx = ast.Load() ), attr = '__internal_decref__', ctx = ast.Load() ),
					args = [ ast.Name( id = bind_name, ctx = ast.Load() ) ], keywords = [],
				))
				ast.copy_location( decref_call, node ); ast.copy_location( decref_call.value, node )
				ast.copy_location( decref_call.value.func, node ); ast.copy_location( decref_call.value.func.value, node )
				self._lower_stmt( decref_call )

		def ok_thunk() -> bool:
			self._cfg.narrow( next_var.stem, ok_member )
			ok_bind_name = f'__for_ok_{unique}'
			self._declare_hidden_local( ok_bind_name, elem_type, node )
			extract = ast.Assign( targets = [ ast.Name( id = ok_bind_name, ctx = ast.Store() ) ], value = self.lowering._synth_name( next_var.stem, node ))
			ast.copy_location( extract, node )
			self._stmt_Assign( extract )
			rewrap_bind( 'Ok', ok_bind_name, elem_type )
			return False

		def err_thunk() -> bool:
			self._cfg.narrow( next_var.stem, err_member )
			err_bind_name = f'__for_err_{unique}'
			self._declare_hidden_local( err_bind_name, full_error_type, node )
			extract = ast.Assign( targets = [ ast.Name( id = err_bind_name, ctx = ast.Store() ) ], value = self.lowering._synth_name( next_var.stem, node ))
			ast.copy_location( extract, node )
			self._stmt_Assign( extract )

			def leaf_test_cond( leaf_type: Type ) -> ir.Operand:
				leaf_member = e_members[ id( leaf_type ) ]
				tag_expr = ast.Attribute( value = self.lowering._synth_name( err_bind_name, node ), attr = e_tag_attr.stem, ctx = ast.Load() )
				ast.copy_location( tag_expr, node )
				test = ast.Compare( left = tag_expr, ops = [ ast.Eq() ], comparators = [ ast.Constant( value = e_tags[ leaf_member.stem ] ) ] )
				ast.copy_location( test, node )
				return self._lower_expr( test, bool_cls )

			def leaf_bind_thunk( leaf_type: Type ) -> bool:
				# narrow()'d reads of err_bind_name aren't cached - EVERY
				# ast.Name(id=err_bind_name) read after this narrow() re-
				# extracts through the payload independently (a fresh
				# aliasing-read incref each time, same mechanism as any
				# other narrowed read), unlike an ordinary plain-variable
				# read which always refers to the SAME already-extracted
				# value. rewrap_bind reads bind_name TWICE (the rewrap's
				# own argument, then the decref) - reading err_bind_name
				# itself directly for both would incref it twice while
				# only ever decref-ing one of those extractions, leaking
				# the other - confirmed via a real repro. Extract ONCE
				# into a fresh, real (non-narrowed) local instead, matching
				# ok_thunk's own identical pattern - both of rewrap_bind's
				# own reads then correctly refer to the SAME extraction.
				leaf_bind_name = f'__for_err_{leaf_type.stem}_{unique}'
				self._declare_hidden_local( leaf_bind_name, leaf_type, node )
				self._cfg.narrow( err_bind_name, e_members[ id( leaf_type ) ] )
				leaf_extract = ast.Assign( targets = [ ast.Name( id = leaf_bind_name, ctx = ast.Store() ) ], value = self.lowering._synth_name( err_bind_name, node ))
				ast.copy_location( leaf_extract, node )
				self._stmt_Assign( leaf_extract )
				rewrap_bind( 'Err', leaf_bind_name, leaf_type )
				return False

			def dispatch_real_error( leaves: list[Type] ) -> bool:
				if len( leaves ) == 1:
					return leaf_bind_thunk( leaves[0] )
				leaf = leaves[0]
				cond = leaf_test_cond( leaf )
				self._lower_binary_branch(
					cond, node,
					lambda: leaf_bind_thunk( leaf ),
					lambda: dispatch_real_error( leaves[1:] ),
				)
				return False

			is_stop_iteration_cond = leaf_test_cond( stop_iteration_cls )

			def stop_iteration_thunk() -> bool:
				self._emit( ir.Jump( target = end_label ))
				return True

			def real_error_thunk() -> bool:
				return dispatch_real_error( remaining_leaves )

			self._lower_binary_branch( is_stop_iteration_cond, node, stop_iteration_thunk, real_error_thunk )
			return False

		self._lower_binary_branch( is_err_cond, node, err_thunk, ok_thunk )

	def _loop_has_reachable_break( self, body: list[ast.stmt] ) -> bool:
		''' does `body` contain a `break` that would actually exit the loop
		it belongs to - used by _stmt_diverges' own `while True:` case, to
		decide whether the loop can really be exited normally at all. Does
		NOT descend into a NESTED ast.While/ast.For's own body - a break
		inside a nested loop exits THAT loop, not this one, so it doesn't
		count here. Does descend into ast.If (both body/orelse - a match
		statement's own desugared chain included, same shape _stmt_diverges
		itself already recurses through) and ast.Try, since a break inside
		either of those still belongs to the SAME enclosing loop. '''
		for stmt in body:
			if isinstance( stmt, ast.Break ):
				return True
			if isinstance( stmt, ( ast.While, ast.For, ast.FunctionDef ) ):
				continue
			for field in ( 'body', 'orelse', 'finalbody', 'handlers' ):
				sub = getattr( stmt, field, None )
				if not isinstance( sub, list ):
					continue
				# ast.Try.handlers is a list[ast.ExceptHandler], not a list[ast.stmt]
				# directly - recurse into each handler's own .body instead
				sub_stmts = [ h.body for h in sub ] if field == 'handlers' else [ sub ]
				for stmts in sub_stmts:
					if isinstance( stmts, list ) and self._loop_has_reachable_break( stmts ):
						return True
		return False

	def _try_diverges( self, node: ast.Try ) -> bool:
		''' 3-way generalization of _stmt_diverges's own ast.If case, for a
		try statement: true iff EVERY reachable path out of body/else/
		handlers already diverges - i.e. nothing can fall through to the
		try statement's own natural end. Does NOT factor in finalbody's own
		divergence - _stmt_diverges' own ast.Try branch checks that
		separately (finally runs on every path, including the ones this
		already counts as diverging, so it can't change the answer here
		either way); _stmt_Try's own fallthrough-inline decision (whether to
		disarm and inline finalbody directly, vs. leave it to fire once from
		the function's real epilogue on every path already diverging) reuses
		this same computation. '''
		tail = node.orelse if node.orelse else node.body
		if not tail or not self._stmt_diverges( tail[-1] ):
			return False
		for h in node.handlers:
			if not h.body or not self._stmt_diverges( h.body[-1] ):
				return False
		return True

	def _stmt_diverges( self, stmt: ast.stmt ) -> bool:
		''' true if `stmt` never falls through to the statement after it -
		either structurally (return/break/continue) or because it's a bare
		call expression to a function declared -> NoReturn (sys.panic, most
		commonly). Used by _stmt_If (true_terminates/false_terminates) to
		decide whether a branch's own ending narrowed/bindings state can
		reach the if's join point at all - see merge_if()'s own docstring.
		Resolved via _resolve_callee_target rather than a full _lower_call -
		this only needs the CALLEE's declared return type, not a real
		lowered call (the statement was already lowered by the caller's own
		loop before this runs), and _resolve_callee_target is a pure lookup
		with no scheduling side effects beyond _resolve_callable's ordinary
		signature-resolution. A receiver call (x.method()) or anything
		_resolve_callee_target can't resolve without a receiver just isn't
		recognized here - NoReturn is overwhelmingly a free-function/sys.*
		shape (sys.panic, sys.exit, ...), and misses just fall back to
		today's existing (safe, if incomplete) behavior. '''
		if isinstance( stmt, ( ast.Return, ast.Break, ast.Continue, ast.Raise )):
			return True
		if isinstance( stmt, ast.With ):
			# a with-block's own exit (__exit__) doesn't change whether
			# control falls through PAST the with statement itself - that's
			# entirely decided by whether its OWN body's last statement
			# diverges, exactly like an ast.If's body above. Extremely
			# common shape in this codebase (`with self.__lock: return
			# ...`, `with compiler.wrap_arithmetic: return ...`) - without
			# this, EVERY lock-wrapped accessor/mutator in lib/builtins/
			# __list.py, __init__.py (dict), etc. was wrongly flagged as
			# non-terminating by the definite-return check built on this.
			return bool( stmt.body ) and self._stmt_diverges( stmt.body[-1] )
		if isinstance( stmt, ast.If ):
			# an if/else BOTH of whose own branches diverge is itself
			# terminating, even though it isn't literally a Return/Break/
			# Continue - the shape a `match` statement desugars to
			# (type_resolver.py's visit_Match, chained ast.If via
			# tail.orelse) whenever every case returns. Recursing through
			# _stmt_diverges itself (rather than a one-level check) handles
			# arbitrarily long desugared case chains, each nested one level
			# deeper than the last. No orelse at all means the false path
			# always falls through, so it can never qualify.
			return (
				bool( stmt.body ) and self._stmt_diverges( stmt.body[-1] )
				and bool( stmt.orelse ) and self._stmt_diverges( stmt.orelse[-1] )
			)
		if isinstance( stmt, ast.Try ):
			# finalbody itself diverging (a bare `return`/panic in a
			# `finally:` block) makes the WHOLE construct diverge
			# regardless of body/else/handlers - it runs unconditionally on
			# every exit path, so nothing downstream is ever reachable
			# either way. Otherwise: see _try_diverges's own docstring
			if stmt.finalbody and self._stmt_diverges( stmt.finalbody[-1] ):
				return True
			return self._try_diverges( stmt )
		if (
			isinstance( stmt, ast.While ) and isinstance( stmt.test, ast.Constant ) and stmt.test.value is True
			and not self._loop_has_reachable_break( stmt.body )
		):
			# `while True:` with no break anywhere in its own body (not
			# counting a break that belongs to a NESTED loop instead - see
			# _loop_has_reachable_break) never falls through to whatever
			# follows it, REGARDLESS of what its body itself ends with -
			# the only way past this statement is a `return`/NoReturn call
			# somewhere inside it, or it runs forever. Needed so a
			# definite-return check built on this can't-fall-through
			# analysis doesn't reject a real, common shape (an event-loop-
			# style function whose only way out is an inner `return`) as
			# "missing return".
			return True
		if not ( isinstance( stmt, ast.Expr ) and isinstance( stmt.value, ast.Call )):
			return False
		if self.lowering._defer_kind_of_call( stmt.value ) is not None:
			# defer(...)/errdefer(...) as an if-branch's LAST statement (a
			# natural, common shape - arm a cleanup right before the branch
			# falls through) is not itself a real call: `defer`/`errdefer`
			# are recognized purely by this AST shape (_stmt_Expr, before
			# ordinary call resolution ever runs - see this file's own
			# module docstring), never registered as an actual Name anywhere.
			# Falling through to _resolve_callee_target below tried to look
			# up 'defer' as an ordinary callable and failed outright ("name
			# 'defer' is not defined") - a real, confirmed compile error on
			# every defer/errdefer that happens to be the last statement of
			# a non-terminating if-branch (found via lib/builtins/__str.py's
			# `if loc is not None: defer(freelocale(loc))`, the only defer
			# call in this codebase shaped that way - every other call site
			# happens to sit at the top level or right after an early-return
			# guard, never as an if-branch's own last statement, which is
			# why this went unnoticed until real POSIX-target compilation
			# actually exercised it). defer/errdefer always falls through
			# (arms a flag, never diverges) - never NoReturn-shaped
			return False
		if self.lowering._is_compiler_call( stmt.value ) == 'early_return':
			# compiler.early_return(err) - a same-function early bailout,
			# lowered as an unconditional jump straight to the epilogue with
			# Result.Err(err) as the return value (_lower_compiler_early_
			# return) - genuinely diverges exactly like an ordinary `return`,
			# but is a compiler intrinsic (recognized textually, same as
			# defer/errdefer above), not an ordinary resolvable Function, so
			# _resolve_callee_target below would never find it - without this
			# case a function whose only "return" is compiler.early_return(...)
			# as a bare statement was wrongly flagged as never returning.
			return True
		target = self.lowering._type_resolver._resolve_callee_target( stmt.value.func )
		fn = target.base if isinstance( target, Specialization ) else target
		if not isinstance( fn, Function ):
			return False
		return isinstance( fn.return_type, Scalar ) and fn.return_type.stem == 'NoReturn'

	def _flush_branch_temps( self, start_idx: int, *keep: ir.Operand ) -> None:
		# shared by _expr_BoolOp and _expr_IfExp: both lower a SEQUENCE of
		# conditionally-skippable sub-expressions (BoolOp operands after a
		# short-circuit jump; the IfExp branch that didn't run) into the
		# same straight-line instruction stream. A sub-expression that
		# chains multiple calls (e.g. `field.find(x)` receiver consumed by
		# `.is_ok()`, or `prefix + str('.') + k`'s two chained __add__s)
		# DeclareTemp's its own intermediate temps via the ordinary
		# self._new_temp() path - every one of those lands in
		# self._pending_temps exactly like any other temp. Neither method
		# ever touches these purely-intermediate temps itself (only the
		# construct's own final value - `operand`/`true_val`/`false_val` -
		# gets special handling), so left alone they'd survive in
		# _pending_temps all the way to the ENCLOSING STATEMENT's own
		# _flush_pending_temps() (e.g. _stmt_Return's, called once after
		# every operand/branch has already merged at end_label) - which
		# then decref's them UNCONDITIONALLY, including for whichever
		# operand/branch never actually ran (skipped by an earlier
		# operand's short-circuit jump, or the branch not taken), reading
		# tag/payload data off an uninitialized C local. Confirmed as a
		# real, reproducible crash in both shapes - not just a leak/UAF:
		# _expr_IfExp's case releases a garbage object pointer through a
		# garbage vtable (a stack-overflow crash); _expr_BoolOp's case
		# reads a garbage union tag and, whenever it happens to look like
		# the RC leaf, releases a garbage pointer straight from the stack
		# (a STATUS_BREAKPOINT crash under MSVC - confirmed with as few as
		# 2 chained `field.find(x).is_ok() or field.find(y).is_ok()`
		# operands, whenever the first one short-circuits).
		#
		# The fix: flush each operand/branch's OWN intermediate temps
		# (added to _pending_temps since `start_idx`, i.e. everything
		# DeclareTemp'd while lowering just this one) right here, before
		# moving on to the next operand or the branch's own Jump/merge -
		# exactly mirroring how the already-correct if/else STATEMENT form
		# gets this right for free (each branch is its own statement, so
		# _lower_stmt's per-statement pending_temps save/flush/restore
		# already scopes it correctly). `keep` (the construct's own
		# dest/final-value temps) is excluded - their ownership is already
		# fully resolved by the incref/untrack_temp() decision made just
		# above each call site (both IfExp's branches and BoolOp's own
		# DECISIVE operand get this same treatment now - see _expr_BoolOp),
		# and dest in particular is still actively in use afterward
		# (assigned into, then read again once every operand/branch
		# merges) so it must not be DeleteTemp'd here even though the
		# actual decref side would already be a safe no-op for it.
		# _expr_BoolOp's own NON-decisive operands (evaluated only for
		# their truthiness, never surviving into dest - see its own
		# comment) are the one case that calls this with `keep` holding
		# ONLY dest: the operand itself is meant to be released here too,
		# same as any other purely-intermediate temp.
		branch_temps = self._pending_temps[ start_idx: ]
		self._pending_temps = self._pending_temps[ : start_idx ]
		keep_ids = { k.id for k in keep if isinstance( k, ir.Temp ) }
		for t in reversed( branch_temps ):
			if t.id in keep_ids:
				continue
			for instr in self._cfg.delete_temp( t ):
				self._emit( instr )
			self._emit( ir.DeleteTemp( temp = t ))
