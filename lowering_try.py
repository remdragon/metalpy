# stdlib imports:
import ast
from typing import Callable

# local imports:
import cfg
import ir
from errors import CompileError
from mpy_types import (
	Name, Type, Variable, Parameter, Specialization, TaggedUnion, CStruct, CUnion, CEnum, TypeVar, RCClass, ClosureType,
)

from lowering_shared import TryHandler, TryContext

class TryLoweringMixin:
	''' try/except/raise lowering and or_return/or_throw propagation machinery - mixed into FunctionLowering (lowering.py), which
	see for the shared instance state (self._instructions, self._cfg, self.lowering,
	etc.) every method here reads and writes. Never instantiated on its own;
	split out of lowering.py purely to keep that file to a manageable size - see
	lowering.py's own class docstring and FunctionLowering's base-class list for
	the full set of sibling mixins this one is composed with. '''


	def _stmt_TryStar( self, node: ast.stmt ) -> None:
		# ast.TryStar (`try: ... except* T:`) reuses ast.ExceptHandler for
		# its own handlers, so it isn't distinguishable from ast.Try by
		# shape alone - dispatched here separately (by class name, see
		# _lower_stmt) purely to reject it with a clear, dedicated message
		# rather than whatever _stmt_Try's own except-clause validation
		# would happen to say about it
		self.lowering.discovery.fail( f'except* (exception groups) is not supported: {ast.unparse(node)}', node )

	def _stmt_Try( self, node: ast.Try ) -> None:
		''' limited try/except/else/finally - not real unwinding: control
		reaches an except handler only via `.or_throw()` (_lower_or_throw)
		or a `raise EXPR` statement (_stmt_Raise), each textually inside
		this try's own body, in this SAME function (self._try_stack,
		pushed/popped around the body's own lowering). Mirrors
		_lower_with_context_manager's own already-debugged shape for
		`finally` - see that method's docstring for the full "why no new
		lexical scope, why register-then-disarm-and-inline" story; `as
		NAME` binds an ordinary, function-scoped local exactly like with's
		own NAME does, just built directly here (there's no AST-level
		expression for "the narrowed Err payload of this already-lowered
		or_throw()/raise dispatch" for an ast.Assign to reference - see the
		Variable construction below).

		Nested try: an uncovered leaf walks the WHOLE enclosing try stack,
		innermost first (_dispatch_leaves_against_try_stack, shared by
		or_throw() and raise) - an inner try's own uncovered leaf DOES
		search every outer try's own handlers before falling back to the
		function-return propagation path. This is sound without any new CFG
		machinery: _stmt_Try is Python-call-stack-recursive, so an outer
		TryContext's own `handlers` list is still live/mutable while an
		inner try's body is being lowered, and every handler (inner or
		outer) is already modeled as diverging from its OWN try's entry
		snapshot (below) - a goto from anywhere inside an inner try's body
		is textually still inside the outer try's own body too, already
		covered by that same conservative model. Labels are function-flat
		and globally unique (_new_label), so a goto reaching an outer
		label works exactly like reaching an inner one.

		Dead-except check (after the try_stack pop, below): an except
		clause whose TryHandler.matched never got set True by any
		or_throw()/raise dispatch anywhere in this try's own body is
		unreachable - a compile error. '''
		# none of the messages below unparse `node`/`h` themselves - each is
		# a compound statement with its own BODY (the whole try block, or
		# a whole except handler), which would dump that entire block into
		# the error message (a real repro: a multi-statement try/except
		# produced a multi-line error that buried the actual problem); the
		# node passed as the location (node/h/te) already pinpoints the
		# right line precisely enough on its own
		if self._current_fn.is_generator_next:
			self.lowering.discovery.fail(
				'try-statement is not supported inside a generator body yet', node,
			)
		if self._loop_depth > 0 and (
			self._body_may_break_or_continue_to_enclosing_loop( node.body )
			or self._body_may_break_or_continue_to_enclosing_loop( node.orelse )
			or any( self._body_may_break_or_continue_to_enclosing_loop( h.body ) for h in node.handlers )
		):
			self.lowering.discovery.fail(
				'try-statement is not allowed inside a loop when its body/else/except-handlers can break/continue out '
				'of that loop - call another function and use try/except inside that instead', node,
			)

		handlers: list[TryHandler] = []
		for h in node.handlers:
			if h.type is None:
				self.lowering.discovery.fail( 'bare `except:` is not supported - name the specific error class(es)', h )
			type_exprs = h.type.elts if isinstance( h.type, ast.Tuple ) else [ h.type ]
			leaves: list[Type] = []
			for te in type_exprs:
				resolved = self.lowering._type_resolver._try_resolve_namespace( te )
				if resolved is None or not isinstance( resolved, ( RCClass, CStruct, CUnion, TaggedUnion, CEnum )):
					self.lowering.discovery.fail( f'except clause must name a class: {ast.unparse(te)}', te )
				if getattr( resolved, 'stem', None ) == 'Exception':
					self.lowering.discovery.fail(
						'bare `except Exception:` is not supported - name the specific error class(es)', h,
					)
				leaves.append( resolved )
			label = self._new_label( 'except' )
			bind_type = leaves[0] if len( leaves ) == 1 else self.lowering.discovery._get_or_create_union( leaves )
			if len( leaves ) > 1:
				self.lowering._union_storage.get( bind_type ) # ensures bind_type's own tag/data storage exists by emit time
			self.lowering.schedule( bind_type )
			fn = self._current_fn
			bind_var: Variable|None = None
			if h.name is not None:
				needs_uid_suffix = self._mark_fresh_local_declared( h.name )
				bind_var = Variable(
					stem = h.name, qualname = f'{fn.qualname}.{h.name}', file = fn.file, line = getattr( h, 'lineno', None ),
					type = bind_type, needs_uid_suffix = needs_uid_suffix,
				)
				fn.add_name( bind_var.stem, bind_var )
				raise_value_var = bind_var
			else:
				# hidden, not user-facing - only bare `raise` inside this
				# handler's own body ever references it (_stmt_Raise), so no
				# uid-suffix collision tracking is needed: `label` is already
				# unique per handler within this function (_new_label).
				hidden_stem = f'__except_value_{label}'
				raise_value_var = Variable(
					stem = hidden_stem, qualname = f'{fn.qualname}.{hidden_stem}', file = fn.file, line = getattr( h, 'lineno', None ),
					type = bind_type, needs_uid_suffix = False,
				)
				fn.add_name( raise_value_var.stem, raise_value_var )
			handlers.append( TryHandler( leaves = leaves, label = label, bind = bind_var, raise_value_var = raise_value_var ))
		end_label = self._new_label( 'try_end' )

		if node.finalbody and self._body_contains_return( node.finalbody ):
			self.lowering.discovery.fail(
				'finally: return statements are not allowed inside a finally block - a return here would silently '
				'discard whatever the try/except was actually about to return', node,
			)

		exit_entry: cfg.Epilogue|None = None
		if node.finalbody:
			# registered BEFORE the body is lowered, exactly like
			# _lower_with_context_manager's own __exit__ - covers every
			# early-exit path reached from inside body/else/any handler
			# (return, or an uncovered or_throw() leaf propagating out)
			exit_entry = self._register_defer_block( is_err_only = False, body = node.finalbody, node = node, allow_inside_loop = True )

		# Handlers are mutually-exclusive alternatives - structurally like
		# if/elif arms - and get the same enter_branch/snapshot/restore/
		# merge_if treatment _stmt_If gives its own branches (see this
		# method's own module-level design notes in the task that produced
		# this code). The try body/else stay UNCONFINED (matching `with`'s
		# own precedent - they always execute exactly once when reached,
		# unlike a handler, which may or may not run at all).
		#
		# Wrinkle: a single handler can be the goto target of multiple
		# .or_throw() call sites scattered through the try body, each with
		# potentially different live/owned/narrowed state. Real per-
		# dispatch-site predecessor merging would need a proper multi-
		# predecessor CFG join - out of scope. Conservative approximation
		# instead: every handler is modeled as diverging from the try's own
		# ENTRY snapshot, never from wherever a specific dispatch site sits.
		# This can only ever be MORE conservative than reality, never
		# unsound.
		entry_snapshot = self._cfg.snapshot()
		outer_instructions = self._instructions
		self._instructions = []
		self._try_stack.append( TryContext( handlers = handlers, end_label = end_label, entry_stack_depth = entry_snapshot.stack_depth ))
		# no special protection needed here for entries that SURVIVE this
		# try (index < entry_snapshot.stack_depth) even though the try body
		# and each handler are independently restore()'d back to this SAME
		# entry_snapshot below - cfg.py's own _neutralize() never mutates a
		# shared Epilogue object in place (see its own docstring), so an
		# ordinary `compiler.decref(g)` on the try body's own fall-through
		# path builds a REPLACEMENT entry rather than touching the one
		# entry_snapshot (and every handler restored from it) still holds.
		try:
			for stmt in node.body:
				try:
					self._lower_stmt( stmt )
				except CompileError:
					continue
		finally:
			self._try_stack.pop()

		for handler, h in zip( handlers, node.handlers ):
			if not handler.matched:
				self.lowering.discovery.fail(
					f'except {ast.unparse(h.type)}: is unreachable - nothing in this try block ever throws it',
					h,
				)

		for stmt in node.orelse:
			try:
				self._lower_stmt( stmt )
			except CompileError:
				continue

		body_captured = self._instructions
		body_end = self._cfg.snapshot()

		tail = node.orelse if node.orelse else node.body
		combined_terminates = bool( tail ) and self._stmt_diverges( tail[-1] )

		combined_groups: list[list[ir.Instruction]] = [ body_captured ]
		combined_end = body_end

		for handler, h in zip( handlers, node.handlers ):
			self._cfg.restore( entry_snapshot )
			self._instructions = []
			self._cfg.enter_branch( entry_snapshot.stack_depth )
			try:
				# the emitter unconditionally assigns handler.raise_value_var's
				# own payload before jumping to this exact label (ir.OrThrow's/
				# ir.Raise's dispatch - see _emit_leaf_dispatch_case) -
				# definitely assigned AND owned on entry here, hence
				# declare_exception_bind (not bare mark_live). Must happen
				# INSIDE this branch-confined window (moved from the old
				# unconfined lowering) so restore() below correctly tears it
				# back down before the next handler. Always runs now (not
				# just `as NAME` clauses) - a hidden hand-off variable exists
				# for every handler, see TryHandler's own docstring.
				self._cfg.declare_exception_bind( handler.raise_value_var )
				if handler.bind is None:
					# anonymous `except Foo:` (no `as name`) - raise_value_var
					# only gets a real read if the body contains a bare
					# `raise` re-raising it (_stmt_Raise); with none, the
					# emitter's own unconditional payload-write above (see
					# this loop's own leading comment) is its only reference -
					# real -Wunused-but-set-variable/C4189, and there's no
					# syntax the user could write to opt out (this variable
					# is entirely hidden from them). Silence unconditionally,
					# same reasoning as this file's other MarkUsed call sites.
					self._emit( ir.MarkUsed( operand = handler.raise_value_var ))
				self._active_raise_values.append( handler.raise_value_var )
				try:
					for stmt in h.body:
						try:
							self._lower_stmt( stmt )
						except CompileError:
							continue
				finally:
					self._active_raise_values.pop()
			finally:
				self._cfg.exit_branch()
			handler_captured = self._instructions
			handler_end = self._cfg.snapshot()
			# same sense as _stmt_If's own true_terminates: last stmt
			# diverges (return/break/continue) -> this branch never reaches
			# the join point at all.
			handler_terminates = bool( h.body ) and self._stmt_diverges( h.body[-1] )

			# restore to the TRY's own entry snapshot (not combined_end) -
			# merge_if's own docstring documents its precondition as
			# "self.bindings is clean of whatever either branch
			# speculatively pushed" (i.e. exactly entry state); combined_end/
			# handler_end are passed through as plain DATA parameters
			# (true_end/false_end) below, entirely independent of whatever
			# self.bindings currently holds. Restoring to combined_end
			# instead (tried first) left a stale binding in self.bindings
			# for any name merge_if's own "fresh on exactly one branch"
			# path drops (it only ever WRITES self.bindings for a name it
			# reestablishes - it never deletes one that was already there
			# but isn't a survivor) - confirmed as a real double-
			# release_object()/uninitialized-read bug via a real MSVC
			# compile+run repro (C4700 "uninitialized local variable 'w'
			# used", then a heap-corruption crash at runtime).
			self._cfg.restore( entry_snapshot )
			# merge_if() below can mint fresh temps (a name dropped on only
			# one side needs its own decref computed here - see its own
			# "fresh on exactly one branch" case) - _new_temp()'s DeclareTemp
			# side effect lands in whatever self._instructions currently is,
			# which must be the real, live outer list (matching _stmt_If's
			# own identical merge_if() call site), NOT handler_captured
			# (still assigned from the just-lowered handler body above) -
			# otherwise the DeclareTemp ends up spliced into the handler's
			# own block while the matching compute/use instructions
			# (returned as plain data, appended into combined_groups/
			# handler_captured below) end up somewhere else entirely -
			# confirmed by a real repro ("use of undeclared identifier
			# '$tN'": a named Result local declared directly inside a try
			# body, dropped by the handler-loop's own restore() above).
			self._instructions = outer_instructions
			try:
				combined_extra, handler_extra, removed = self._cfg.merge_if(
					entry_snapshot.bindings, combined_end.bindings, handler_end.bindings, self._current_fn.qualname,
					entry_results = entry_snapshot.results, true_end_results = combined_end.results, false_end_results = handler_end.results,
					true_terminates = combined_terminates, false_terminates = handler_terminates,
					true_end_narrowed = combined_end.narrowed, false_end_narrowed = handler_end.narrowed,
					true_end_live = combined_end.live, false_end_live = handler_end.live,
				)
			except CompileError as e:
				self.lowering.discovery.fail( str( e ), node )
			# `removed` only drives Decref instructions already spliced into
			# combined_extra/handler_extra above - see _stmt_If's own
			# identical comment on why fn.names must NOT also be touched.

			# combined_extra must land on EVERY physical block folded into
			# the "combined" side so far - either could be the real runtime
			# path (a name confined to just ONE prior handler still needs
			# its own teardown spliced into THAT handler's own block, not
			# just the most recently merged one).
			for group in combined_groups:
				group.extend( combined_extra )
			handler_captured.extend( handler_extra )
			combined_groups.append( handler_captured )

			combined_terminates = combined_terminates and handler_terminates
			combined_end = self._cfg.snapshot()

		self._cfg.restore( combined_end )
		self._instructions = outer_instructions
		reachable = not combined_terminates

		for instr in combined_groups[0]:
			self._emit_captured( instr )
		if reachable:
			self._emit( ir.Jump( target = end_label ))

		for handler_idx, ( handler, h ) in enumerate( zip( handlers, node.handlers )):
			self._emit( ir.Label( name = handler.label ))
			for instr in combined_groups[handler_idx + 1]:
				self._emit_captured( instr )
			if not h.body or not self._stmt_diverges( h.body[-1] ):
				reachable = True
				self._emit( ir.Jump( target = end_label ))

		if reachable:
			self._emit( ir.Label( name = end_label ))
			if node.finalbody:
				# disarm the registered defer and inline finalbody directly
				# here - the SAME already-debugged shape
				# _lower_with_context_manager uses for its own __exit__,
				# just at this construct's own (possibly multi-path)
				# fallthrough merge point instead of a single body's own end
				assert exit_entry is not None
				for instr in self._cfg.disarm_defer( exit_entry ):
					self._emit( instr )
				for stmt in node.finalbody:
					try:
						self._lower_stmt( stmt )
					except CompileError:
						continue

	def _stmt_Raise( self, node: ast.Raise ) -> None:
		''' raise EXPR - a bare goto into a matching except handler of any
		enclosing try (walked innermost-first via
		_dispatch_leaves_against_try_stack, the same helper .or_throw() (
		_emit_or_throw) uses), no Result ever built for that covered case -
		it's a jump, not a value. A leaf uncovered by every enclosing try
		propagates via a REAL function return instead, built from the
		enclosing function's own declared Result[T,E] return type (same
		coverage requirement .or_throw() already enforces via
		_require_or_throw_return) - see ir.Raise's own docstring for the
		emitted shape, which reuses _emit_leaf_dispatch_case's existing
		uncovered-leaf machinery in emitter_c.py.

		Bare `raise` (re-raise) inside an except handler re-raises THAT
		handler's own currently-caught value - reuses this exact same
		dispatch/emission path, just sourced from
		FunctionLowering._active_raise_values's top (the innermost
		enclosing handler's own TryHandler.raise_value_var) instead of a
		freshly-lowered node.exc; a plain `ast.Name` read off it, fed back
		through the ordinary expression pipeline (_lower_expr), gets the
		usual aliasing incref for free - see TryHandler's own docstring.
		Because _stmt_Try pops the current try's own TryContext off
		_try_stack before lowering ANY handler body, this dispatch can
		never re-match a sibling except of the SAME try - it only ever
		walks OUTER enclosing tries, same as it would for a real re-raised
		value written out by hand. Bare `raise` with nothing on
		_active_raise_values (not textually inside a handler's own body at
		all) is a compile error. No `raise ... from ...` (exception
		chaining). Same generator-body rejection as .or_throw()
		(_lower_or_throw) - see its own comment for why. The @inline-
		splice-prelude case is now generalized (ir.Raise.inline_exit)
		rather than rejected - same carve-out _emit_or_throw's own
		uncovered-leaf branch applies, and same PLAN_RETURN_INFERENCE.md
		sentinel-state rejection as _consume_checked_result/
		_emit_or_throw's own. '''
		if node.cause is not None:
			self.lowering.discovery.fail( f'raise ... from ... (exception chaining) is not supported: {ast.unparse(node)}', node )
		if self._current_fn is not None and self._current_fn.is_generator_next:
			self.lowering.discovery.fail(
				f'raise is not supported inside a generator body yet - a generator body is a state machine re-entered '
				f'across multiple send()/next() resumptions, not called once like an ordinary function, so what an '
				f'uncaught error should even mean here (fail just this resumption vs. end the generator entirely) '
				f'needs real design first: {ast.unparse(node)}', node,
			)
		if self._in_inline_splice_prelude and not self._inline_scope_vars:
			self.lowering.discovery.fail(
				f'@inline: or_throw()/raise that could propagate an error is not yet supported before the '
				f'final return of a multi-statement body whose own return type is still being inferred: {ast.unparse(node)}',
				node,
			)

		if node.exc is None:
			if not self._active_raise_values:
				self.lowering.discovery.fail(
					f'bare raise (re-raise) is only valid inside an except handler: {ast.unparse(node)}', node,
				)
			raise_var = self._active_raise_values[-1]
			exc_node = ast.Name( id = raise_var.stem, ctx = ast.Load())
			ast.fix_missing_locations( ast.copy_location( exc_node, node ))
		else:
			exc_node = node.exc
		value = self._lower_expr( exc_node, None )
		self._raise_value( value, node )

	def _raise_value( self, value: ir.Operand, node: ast.AST ) -> None:
		''' the dispatch/emission half of _stmt_Raise, split out so
		or_throw(mapper) (_lower_or_throw_with_mapper) can raise an
		ALREADY-COMPUTED operand directly - it needs the mapped error's
		own value settled (mapper called, its argument decreffed) BEFORE
		dispatch, which means the raised expression can't be a fresh
		ast.Raise(exc=<call>) re-lowered from scratch here (that would
		call the mapper a second time, or read the argument after it's
		already been decreffed - see the caller's own comment). Keeping
		`value` a bare, never-named ir.Temp (returned directly from
		_lower_expr, never bound via ast.Assign) is what lets untrack_temp()
		below correctly hand its ownership off with no separate unwind
		needed - a NAMED Variable raised from inside a confined branch has
		no such automatic path (see _lower_or_throw_with_mapper's own
		manually_decreffed()-was-a-no-op history for why that alternative
		doesn't work). '''
		error_cls = value.type
		if error_cls is None or not isinstance( error_cls, ( RCClass, CStruct, CUnion, TaggedUnion, CEnum )):
			self.lowering.discovery.fail(
				f'raise EXPR must be a class instance, got {error_cls.qualname if error_cls else "?"}: {ast.unparse(node)}', node,
			)

		all_leaves = self.lowering._type_resolver._atomic_leaves( error_cls )
		# a bare Temp's own exclusion is handled below (untrack_temp) - only
		# a NAMED Variable raised directly (`raise x`) needs excluding HERE
		# too, so unwind_confined() doesn't release it out from under the
		# handler's own bind assignment just below it
		dispatch, covered_leaves = self._dispatch_leaves_against_try_stack(
			all_leaves, exclude = value if isinstance( value, Variable ) else None,
		)

		result_cls = self.lowering.discovery.find_name( 'Result', node )
		self.lowering._type_resolver._require_or_throw_return(
			node, result_cls, error_cls, covered_leaves, self.lowering._RAISE_ALTERNATIVES, fn = self._current_fn,
		)

		try:
			self._cfg.check_unchecked_results( value )
		except CompileError as e:
			self.lowering.discovery.fail( str( e ), node )

		# `value`'s own reference always transfers out through the dispatch
		# below - into the handler's raise_value_var (every handler has one,
		# `as NAME` or not - TryHandler's own docstring; its own lifetime is
		# the same manual-decref idiom as any other extracted RC element,
		# e.g. list.__del__/dict's _release_key) or widened into a
		# propagated Result - never simply dropped here, so it must never be
		# released by the ordinary end-of-statement temp flush either.
		# Confirmed by a real repro: `raise Boom(code=5)` / `except Boom as
		# e: ...e.code...` previously read freed memory (release_object($t0)
		# emitted before `e = $t0`) - the flush was unconditionally
		# releasing `value` regardless of whether anything downstream still
		# needed it.
		self._cfg.untrack_temp( value )

		all_covered = len( covered_leaves ) == len( all_leaves )
		if all_covered:
			# every leaf dispatches straight into a handler - no propagation
			# path exists at all, mirrors _emit_or_throw's own identical
			# all_covered short-circuit
			self._flush_pending_temps() # see _stmt_Return's own identical comment: before the terminator, not after
			self._emit( ir.Raise( value = value, dispatch = dispatch ))
			return

		tracked_operand = value if isinstance( value, Variable ) else None
		# see _emit_or_throw's own identical inline_scope carve-out/comment
		inline_scope = self._inline_scope_vars[-1] if self._in_inline_splice_prelude and self._inline_scope_vars else None
		if inline_scope is not None:
			replay = self._cfg.return_( tracked_operand, lambda: self._build_is_err_check( node ))
			self._flush_pending_temps()
			self._cfg.mark_inline_scope_captured()
			self._emit( ir.Raise( value = value, dispatch = dispatch, epilogue = replay, inline_exit = inline_scope ))
			return
		label = self._cfg.current_epilogue_label( tracked_operand )
		if label is not None:
			self._flush_pending_temps()
			self._emit( ir.Raise(
				value = value, dispatch = dispatch, target = label, return_slot = self._return_value_var,
			))
		else:
			replay = self._cfg.return_( tracked_operand, lambda: self._build_is_err_check( node ))
			self._flush_pending_temps()
			self._emit( ir.Raise( value = value, dispatch = dispatch, epilogue = replay ))

	def _lower_or_return(
		self, node: ast.Call, receiver: ir.Operand, want_result: bool, *, receiver_pending_start: int|None = None,
	) -> ir.Operand|None:
		# <result_expr>.or_return() is recognized textually here rather than
		# ever actually calling Result.or_return's own declared body
		# (`if self.is_err(): compiler.early_return(self.data.v_Err)` /
		# `return self.data.v_Ok`) - that body is written as a spec of the
		# intended behavior, not something literally compilable: it needs to
		# trigger a `return Result.Err(...)` in ITS CALLER's scope, not its
		# own (or_return's own declared return type is bare T, not
		# Result[T,E], so `return Result.Err(...)` from inside it could
		# never type-check there - see compiler.early_return's own comment).
		# This expands directly to the same OrReturn/OrJump primitives
		# checked-arithmetic already uses for exactly the same "propagate
		# the error to the enclosing function, continue with the unwrapped
		# value" shape - no new IR needed, and Result.or_return is never
		# scheduled/lowered as a real function as a result.
		#
		# or_return(mapper): a single positional argument is a whole
		# separate shape (see _lower_or_return_with_mapper) - dispatched
		# here, before any of the no-arg-specific validation below runs.
		if len( node.args ) == 1 and not node.keywords:
			return self._lower_or_return_with_mapper( node, receiver, node.args[0], want_result, receiver_pending_start = receiver_pending_start )
		if node.args or node.keywords:
			self.lowering.discovery.fail( f'or_return() takes no arguments, or a single error-mapping callable: {ast.unparse(node)}', node )
		shape = self.lowering._type_resolver._result_shape( receiver.type )
		if shape is None:
			self.lowering.discovery.fail( f'or_return() receiver must be Result[_,_], got {receiver.type.qualname if receiver.type else "?"}', node )
		result_type, error_cls = shape
		# find_name, not receiver.type.base - see _maybe_consume_result's
		# identical comment on why
		result_cls = self.lowering.discovery.find_name( 'Result', node )
		self.lowering._type_resolver._require_result_return( node, result_cls, error_cls, self.lowering._OR_RETURN_ALTERNATIVES, fn = self._current_fn )
		# clearing receiver (when it's a named Variable) and validating that
		# nothing ELSE is still unchecked at this early-exit point both now
		# live in _consume_checked_result itself, shared with checked-
		# arithmetic's own identical OrReturn/OrJump early-exit - see its
		# own comment
		unwrapped = self._consume_checked_result( node, receiver, result_type, extra = None, receiver_pending_start = receiver_pending_start )
		if not want_result:
			# unwrapped's own extraction is bundled into OrReturn/OrJump's IR
			# shape (see _consume_checked_result) - can't be skipped even
			# though a bare `expr.or_return()` statement (validate-only,
			# error propagation is the only wanted effect) never reads it.
			# _flush_pending_temps (via _consume_checked_result's own
			# fresh_temp() registration) covers the RC-leaf release; this
			# MarkUsed just silences a real -Wunused-but-set-variable for the
			# non-RC case (e.g. Result[u8,E]) - confirmed suite-wide
			# (lib/urllib/parse.py's own _unquote_impl first-pass validation
			# scan).
			self._emit( ir.MarkUsed( operand = unwrapped ))
			return None
		return unwrapped

	def _fold_caller_location( self, intrinsic: str, param: Parameter, node: ast.Call ) -> ir.Operand:
		call = param.default
		if call.args or call.keywords:
			self.lowering.discovery.fail( f'compiler.{intrinsic}() takes no arguments', call )
		if intrinsic == 'caller_line':
			i32_cls = self.lowering.discovery.get_intrinsics()['i32']
			if param.type is not i32_cls:
				self.lowering.discovery.fail( f'compiler.caller_line() can only default an i32 parameter, not {param.type.qualname}', call )
			return self._const_i32( node.lineno )
		str_cls = self.lowering.discovery.find_name_or_none( 'str' )
		if param.type is not str_cls:
			self.lowering.discovery.fail( f'compiler.caller_file() can only default a str parameter, not {param.type.qualname}', call )
		caller_file = self.lowering.discovery.module_stack[-1].file if self.lowering.discovery.module_stack else None
		return ir.Const( type = str_cls, value = str( caller_file ) if caller_file is not None else '<unknown>' )

	def _lower_or_return_with_mapper(
		self, node: ast.Call, receiver: ir.Operand, mapper_node: ast.expr, want_result: bool, *, receiver_pending_start: int|None = None,
	) -> ir.Operand|None:
		''' <result_expr>.or_return(mapper) - like the no-arg or_return()
		above, but the Err leg calls `mapper(err)` first and propagates
		ITS result instead of the receiver's own error payload unchanged
		(converting one error type into another, e.g. `IndexError` ->
		`StopIteration`, without a full match/if-is_err() block).

		Unlike the no-arg form (a single flat OrReturn/OrJump instruction,
		since it only ever moves the SAME payload through unmodified),
		this needs a real conditional: the mapper call must run ONLY on
		the Err path, and needs a genuine Operand for the receiver's own
		error payload to pass it - built here as an ordinary binary branch
		(_lower_binary_branch, the same primitive _lower_for_over_iterator
		already uses for its own Ok/Err dispatch) rather than a new IR
		shape. The Err arm ends in a real `return Result.Err(mapper(err))`
		statement (_stmt_Return), deliberately reusing its ALREADY-correct
		widening/defer/errdefer/inline-splice/generator-pessimistic-done
		handling rather than re-implementing any of it for a second time -
		mapper(err)'s own call goes through the ordinary _lower_call
		dispatch too (via a synthesized ast.Call, not hand-built IR),
		which is what makes a real capturing closure work here for free,
		the same as a plain function/non-capturing lambda.

		Deliberately NOT supported yet inside a generator body
		(self._current_fn.is_generator_next) - a generator's own body is
		desugared into its yield/resume state machine by an EARLIER,
		separate pass (type_resolver.py, before lowering.py ever runs),
		which never sees this method's own synthesized branch/return at
		all (it's built directly as IR, here, after that pass has already
		finished) - untested interaction, rejected outright rather than
		risking a silent miscompile. '''
		if self._current_fn is not None and self._current_fn.is_generator_next:
			self.lowering.discovery.fail(
				f'or_return(mapper) is not supported inside a generator body yet: {ast.unparse(node)}', node,
			)
		shape = self.lowering._type_resolver._result_shape( receiver.type )
		if shape is None:
			self.lowering.discovery.fail( f'or_return(mapper) receiver must be Result[_,_], got {receiver.type.qualname if receiver.type else "?"}', node )
		result_ok_type, error_cls = shape
		tagged_shape = self.lowering._type_resolver._tagged_union_shape( receiver.type )
		assert tagged_shape is not None, 'internal compiler error: _result_shape succeeded but _tagged_union_shape did not'
		_result_base, result_members = tagged_shape
		err_member = next( a for a in result_members if a.stem == 'Err' )
		ok_member = next( a for a in result_members if a.stem == 'Ok' )
		concrete_result_union = self.lowering.monomorphize_class( receiver.type ) if isinstance( receiver.type, Specialization ) else receiver.type
		tag_attr, _data_attr, _payload_cls, tags = self.lowering._union_storage.get( concrete_result_union )
		bool_cls = self.lowering.discovery.find_name( 'bool', node )

		is_alias = self.lowering._is_aliasing_expr( node.func.value, receiver )
		if receiver_pending_start is not None:
			self._flush_new_pending_temps( receiver_pending_start, receiver )
		unique = self._label_id
		self._label_id += 1
		recv_var = self._declare_hidden_local( f'__or_recv_{unique}', receiver.type, node )
		# track_result=False: recv_var's own Result-ness is scaffolding
		# inspected via the raw .tag comparison below, not is_ok()/is_err()/
		# match - same reasoning as _lower_for_over_iterator's own next_var
		self._cfg_assign( recv_var, receiver, is_alias = is_alias, node = node, track_result = False )

		# mapper is lowered here, ONCE, unconditionally (matching ordinary
		# Python call-argument evaluation semantics - a Call's own callee
		# expression is always evaluated regardless of what its result is
		# later used for), then bound into its own hidden local so err_
		# thunk below (which only runs conditionally) can reference it by
		# a bare Name - required for TWO reasons, not just consistency:
		# (1) _try_lower_indirect_call/_try_lower_closure_call's own callee
		# dispatch needs a real Name/Attribute AST shape to statically
		# resolve HOW to call it (raw fn ptr vs closure), which an
		# already-computed Operand alone can't provide; (2) an inline
		# lambda (`or_return(lambda e: StopIteration())`) needs a real
		# expected Callable[...] type to infer its own parameter types
		# from (confirmed via a real repro: without this, lowering the raw
		# mapper_node directly as err_thunk's own ast.Call.func failed with
		# "cannot infer lambda parameter types - no expected Callable[...]
		# context") - built here as Ptr[Callable[[error_cls],<TypeVar>]],
		# the SAME provisional-TypeVar-return shape any other generic call
		# site accepting a lambda argument already uses (_expr_Lambda's own
		# return_type_provisional branch infers the real return type
		# eagerly from the lambda's own body either way).
		ptr_cls = self.lowering.discovery.get_intrinsics()['Ptr']
		mapper_ret_placeholder = TypeVar( stem = '_MapperRet', qualname = f'{self._current_fn.qualname}._MapperRet_{unique}', file = None, line = None )
		mapper_callable_type = self.lowering.discovery._get_or_create_callable_type( [ error_cls ], mapper_ret_placeholder )
		mapper_expected_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ mapper_callable_type ] )
		# strict=False: mapper_expected_type carries a PROVISIONAL return
		# type (an unbound TypeVar, needed only so _expr_Lambda has
		# somewhere to infer parameter types from) - the operand actually
		# produced legitimately has a narrower, fully concrete return type
		# once a lambda's own body resolves it, which the ordinary strict
		# _check_assignable tail would otherwise reject as a mismatch
		# against the placeholder type it was never meant to literally match
		mapper_operand = self._lower_expr( mapper_node, mapper_expected_type, strict = False )
		mapper_var = self._declare_hidden_local( f'__or_mapper_{unique}', mapper_operand.type, node )
		mapper_is_alias = self.lowering._is_aliasing_expr( mapper_node, mapper_operand )
		self._cfg_assign( mapper_var, mapper_operand, is_alias = mapper_is_alias, node = node )

		tag_expr = ast.Attribute( value = self.lowering._synth_name( recv_var.stem, node ), attr = tag_attr.stem, ctx = ast.Load() )
		ast.copy_location( tag_expr, node )
		is_err_test = ast.Compare( left = tag_expr, ops = [ ast.Eq() ], comparators = [ ast.Constant( value = tags[ err_member.stem ] ) ] )
		ast.copy_location( is_err_test, node )
		is_err_cond = self._lower_expr( is_err_test, bool_cls )

		ok_var_holder: list[Variable] = []

		def err_thunk() -> bool:
			self._cfg.narrow( recv_var.stem, err_member )
			err_bind_name = f'__or_err_{unique}'
			err_bind_var = self._declare_hidden_local( err_bind_name, error_cls, node )
			extract = ast.Assign(
				targets = [ ast.Name( id = err_bind_name, ctx = ast.Store() ) ], value = self.lowering._synth_name( recv_var.stem, node ),
			)
			ast.copy_location( extract, node )
			self._lower_stmt( extract )
			# recv_var's own payload is already safely aliased into err_bind_
			# name by this point - see or_throw_with_mapper's own identical
			# comment for why releasing it here (rather than leaving it to a
			# later epilogue) is correct, not premature.
			if cfg.rc_leaves( receiver.type ):
				self._internal_decref_var( recv_var )
			# mapped_call is lowered EAGERLY here, as its own statement,
			# rather than embedded unevaluated inside `return Result.Err(...)`
			# the way this used to work - a real `return` is unconditional
			# once lowered (_lower_stmt(ret) below never falls back through
			# to here), so err_bind_var/mapper_var's own release has to
			# happen BETWEEN the mapper call and the return, exactly like
			# or_throw_with_mapper's own err_thunk (see its own comment on
			# why eager evaluation, not embedding, is required once anything
			# needs to run between the call and the terminal statement).
			mapper_callable_type_resolved = self.lowering._type_resolver._callable_type_of( mapper_var.type )
			if mapper_callable_type_resolved is None and isinstance( mapper_var.type, ClosureType ):
				mapper_callable_type_resolved = mapper_var.type
			mapped_call = ast.Call(
				func = self.lowering._synth_name( mapper_var.stem, node ), args = [ ast.Name( id = err_bind_name, ctx = ast.Load() ) ], keywords = [],
			)
			ast.copy_location( mapped_call, node )
			mapped_value = self._lower_expr( mapped_call, mapper_callable_type_resolved.return_type )
			if error_cls.is_rc():
				# fully consumed as the mapper's own argument - see or_throw_
				# with_mapper's own identical comment
				self._internal_decref_var( err_bind_var )
			if mapper_var.type.is_rc():
				# a real capturing closure - see or_throw_with_mapper's own
				# identical comment on why its own epilogue entry needs an
				# explicit release here rather than relying on the (absent,
				# for this confined branch) automatic one
				self._internal_decref_var( mapper_var )
			# mapped_value is an already-lowered Operand, not an AST
			# expression, so it can't be re-embedded into a fresh ast.Call
			# the way the original (pre-eager) code did - bound into its own
			# hidden local instead, purely so `Result.Err(...)` below has a
			# real Name to reference. Ownership transfers into the union via
			# an ordinary retain-on-construction + this local's own release
			# (see ok_thunk's own comment on why THAT extraction instead
			# needed an explicit manually_decreffed() move - Result.Err(...)'s
			# construction here, unlike a bare `w = <expr>` assignment, does
			# its own internal retain, so mapped_var's normal scope-exit
			# release balances against mapped_call's own single returned
			# reference correctly, without any double-release).
			mapped_var = self._declare_hidden_local( f'__or_mapped_{unique}', mapped_value.type, node )
			mapped_is_alias = self.lowering._is_aliasing_expr( mapped_call, mapped_value )
			self._cfg_assign( mapped_var, mapped_value, is_alias = mapped_is_alias, node = node )
			wrap_err = ast.Call(
				func = ast.Attribute( value = ast.Name( id = 'Result', ctx = ast.Load() ), attr = 'Err', ctx = ast.Load() ),
				args = [ self.lowering._synth_name( mapped_var.stem, node ) ], keywords = [],
			)
			ast.copy_location( wrap_err, node ); ast.copy_location( wrap_err.func, node ); ast.copy_location( wrap_err.func.value, node )
			ret = ast.Return( value = wrap_err )
			ast.copy_location( ret, node )
			self._lower_stmt( ret )
			return True

		def ok_thunk() -> bool:
			self._cfg.narrow( recv_var.stem, ok_member )
			ok_bind_name = f'__or_ok_{unique}'
			ok_var = self._declare_hidden_local( ok_bind_name, result_ok_type, node )
			extract = ast.Assign(
				targets = [ ast.Name( id = ok_bind_name, ctx = ast.Store() ) ], value = self.lowering._synth_name( recv_var.stem, node ),
			)
			ast.copy_location( extract, node )
			self._lower_stmt( extract )
			# recv_var's own reference (from whatever originally constructed
			# it, e.g. Result.Ok(...)'s own retain-on-construction) is never
			# released on this arm otherwise - only err_thunk's own mirror
			# release does that, for its own leg. ok_var's own retain just
			# above is a SEPARATE, fresh reference - releasing recv_var here
			# doesn't touch it - see err_thunk's own identical comment on why
			# this is correct, not premature, once the payload's already
			# safely copied out.
			if cfg.rc_leaves( receiver.type ):
				self._internal_decref_var( recv_var )
			# mapper_var was retained ONCE, unconditionally, before this
			# branch even started (see this function's own prefix, mirroring
			# or_throw_with_mapper's identical setup) - its release therefore
			# has to happen on BOTH arms, not just err_thunk's. Confirmed via
			# a real repro that this does NOT happen automatically even
			# though the caller's own eventual `return Result.Ok(w)` is an
			# ordinary, unconfined real return (mapper_var's own binding
			# stayed live, unreleased, all the way to program exit) - unlike
			# or_throw_with_mapper, whose OWN err_thunk is a goto/raise that
			# never merges back into this branch at all, or_return's err_
			# thunk EXPLICITLY moves mapper_var out (see its own comment)
			# before its real `return` - merge_if's own OWNED-vs-MOVED
			# reconciliation across the two arms apparently doesn't leave
			# the merged-back state in a form the later real return's walk
			# still recognizes as needing release either way, so this arm
			# needs the same explicit release, not just a differently-timed
			# one.
			if mapper_var.type.is_rc():
				self._internal_decref_var( mapper_var )
			# ok_var is handed out below as this whole expression's OWN
			# return value (to whatever ordinary assignment/expression
			# context invoked or_return(mapper)) - a move, not a borrow: the
			# caller receives ok_var's single retained reference as-is (no
			# fresh retain of its own - confirmed via emitted C, matching
			# every other "expression already returns an owned value"
			# convention in this file). Without cancelling ok_var's own
			# pending epilogue here, its scope-exit release fires ANYWAY on
			# top of whatever the caller does with the value it was handed -
			# a real double-release, confirmed via a real repro
			# (`w: Elem = probe(...).or_return(to_stop); return Result.Ok(w)`
			# on the Ok path, e's own refcount left one short at the far end
			# of a chain of two retains and two releases). manually_decreffed()
			# alone (no preceding decref - see its own docstring, "mirrors
			# move()'s own cancellation exactly") is exactly the move-out
			# primitive this needs: cancel the binding, emit nothing else,
			# same as move()'s own call-argument transfer.
			for instr in self._cfg.manually_decreffed( ok_var ):
				self._emit( instr )
			ok_var_holder.append( ok_var )
			return False

		self._lower_binary_branch( is_err_cond, node, err_thunk, ok_thunk )
		# err_thunk always terminates (a real `return`, never falls through
		# to here) - ok_var is therefore always live whenever control
		# actually reaches this point, exactly like a local declared in a
		# non-terminating if-branch when the other branch returns.
		ok_operand = ok_var_holder[0]
		if not want_result:
			self._emit( ir.MarkUsed( operand = ok_operand ))
			return None
		return ok_operand

	def _lower_or_throw(
		self, node: ast.Call, receiver: ir.Operand, want_result: bool, *, receiver_pending_start: int|None = None,
	) -> ir.Operand|None:
		''' <result_expr>.or_throw() - like or_return() above (same "no real
		method, recognized by AST shape alone" story - see _lower_or_return's
		own comment). The real per-leaf dispatch/emission logic is shared
		with every AUTO-inserted or_throw() site (checked arithmetic,
		__setitem__/AugAssign, a discarded Result statement, a Result
		flowing into a T-typed context) via _emit_or_throw - see its own
		docstring. Only the checks specific to the explicit `.or_throw()`
		SYNTAX (no-args, generator-body rejection) live here - the @inline-
		splice-prelude case is now generalized (ir.OrThrow.inline_exit),
		same carve-out _emit_or_throw's own uncovered-leaf branch applies.
		or_throw(mapper): a single positional argument is a whole separate
		shape (see _lower_or_throw_with_mapper) - dispatched here, before
		any of the no-arg-specific validation below runs. '''
		if len( node.args ) == 1 and not node.keywords:
			return self._lower_or_throw_with_mapper( node, receiver, node.args[0], want_result, receiver_pending_start = receiver_pending_start )
		if node.args or node.keywords:
			self.lowering.discovery.fail( f'or_throw() takes no arguments, or a single error-mapping callable: {ast.unparse(node)}', node )
		if self._current_fn is not None and self._current_fn.is_generator_next:
			self.lowering.discovery.fail(
				f'or_throw() is not supported inside a generator body yet - a generator body is a state machine '
				f're-entered across multiple send()/next() resumptions, not called once like an ordinary function, so '
				f'what an uncaught error should even mean here (fail just this resumption vs. end the generator '
				f'entirely) needs real design first: {ast.unparse(node)}', node,
			)
		shape = self.lowering._type_resolver._result_shape( receiver.type )
		if shape is None:
			self.lowering.discovery.fail( f'or_throw() receiver must be Result[_,_], got {receiver.type.qualname if receiver.type else "?"}', node )
		return self._emit_or_throw(
			node, receiver, want_result, alternatives = self.lowering._OR_THROW_ALTERNATIVES, receiver_pending_start = receiver_pending_start,
		)

	def _lower_or_throw_with_mapper(
		self, node: ast.Call, receiver: ir.Operand, mapper_node: ast.expr, want_result: bool, *, receiver_pending_start: int|None = None,
	) -> ir.Operand|None:
		''' <result_expr>.or_throw(mapper) - like or_return(mapper) above
		(see its own docstring for the general shape: a real conditional
		branch via _lower_binary_branch, the mapper call going through the
		ordinary _lower_call dispatch via a synthesized ast.Call so a real
		capturing closure works for free), but the Err arm ends in a real
		`raise mapper(err)` statement (_stmt_Raise) instead of `return
		Result.Err(mapper(err))` - dispatching the MAPPED error's own
		leaves against any enclosing try's own handlers (innermost first),
		falling back to or_return(mapper)'s own propagate-to-caller
		behavior for any leaf left uncovered, exactly like a hand-written
		`raise` already does. This is why the mapping happens BEFORE
		dispatch, not after: `.or_throw(mapper)` means "convert this error,
		THEN handle/propagate the converted one" - an enclosing `except
		StopIteration:` next to `seq[i].or_throw(to_stop_iteration)` catches
		the MAPPED type, never the receiver's own original error type,
		which is never visible outside this call at all.

		Same generator-body rejection as the no-arg form above (see its own
		comment) - `raise` itself is independently rejected inside a
		generator body too (_stmt_Raise's own identical check), so this
		only needs its own explicit check to give a clearer, or_throw(mapper)-
		specific message rather than surfacing _stmt_Raise's generic one
		for a call site the user never wrote a literal `raise` at. '''
		if self._current_fn is not None and self._current_fn.is_generator_next:
			self.lowering.discovery.fail(
				f'or_throw(mapper) is not supported inside a generator body yet: {ast.unparse(node)}', node,
			)
		shape = self.lowering._type_resolver._result_shape( receiver.type )
		if shape is None:
			self.lowering.discovery.fail( f'or_throw(mapper) receiver must be Result[_,_], got {receiver.type.qualname if receiver.type else "?"}', node )
		result_ok_type, error_cls = shape
		tagged_shape = self.lowering._type_resolver._tagged_union_shape( receiver.type )
		assert tagged_shape is not None, 'internal compiler error: _result_shape succeeded but _tagged_union_shape did not'
		_result_base, result_members = tagged_shape
		err_member = next( a for a in result_members if a.stem == 'Err' )
		ok_member = next( a for a in result_members if a.stem == 'Ok' )
		concrete_result_union = self.lowering.monomorphize_class( receiver.type ) if isinstance( receiver.type, Specialization ) else receiver.type
		tag_attr, _data_attr, _payload_cls, tags = self.lowering._union_storage.get( concrete_result_union )
		bool_cls = self.lowering.discovery.find_name( 'bool', node )

		is_alias = self.lowering._is_aliasing_expr( node.func.value, receiver )
		if receiver_pending_start is not None:
			self._flush_new_pending_temps( receiver_pending_start, receiver )
		unique = self._label_id
		self._label_id += 1
		recv_var = self._declare_hidden_local( f'__ot_recv_{unique}', receiver.type, node )
		self._cfg_assign( recv_var, receiver, is_alias = is_alias, node = node, track_result = False )

		# see or_return_with_mapper's own identical comment on why the
		# mapper is lowered here, once, into its own hidden local, rather
		# than embedded directly in err_thunk's own synthesized ast.Call
		ptr_cls = self.lowering.discovery.get_intrinsics()['Ptr']
		mapper_ret_placeholder = TypeVar( stem = '_MapperRet', qualname = f'{self._current_fn.qualname}._MapperRet_{unique}', file = None, line = None )
		mapper_callable_type = self.lowering.discovery._get_or_create_callable_type( [ error_cls ], mapper_ret_placeholder )
		mapper_expected_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ mapper_callable_type ] )
		mapper_operand = self._lower_expr( mapper_node, mapper_expected_type, strict = False )
		mapper_var = self._declare_hidden_local( f'__ot_mapper_{unique}', mapper_operand.type, node )
		mapper_is_alias = self.lowering._is_aliasing_expr( mapper_node, mapper_operand )
		self._cfg_assign( mapper_var, mapper_operand, is_alias = mapper_is_alias, node = node )

		tag_expr = ast.Attribute( value = self.lowering._synth_name( recv_var.stem, node ), attr = tag_attr.stem, ctx = ast.Load() )
		ast.copy_location( tag_expr, node )
		is_err_test = ast.Compare( left = tag_expr, ops = [ ast.Eq() ], comparators = [ ast.Constant( value = tags[ err_member.stem ] ) ] )
		ast.copy_location( is_err_test, node )
		is_err_cond = self._lower_expr( is_err_test, bool_cls )

		ok_var_holder: list[Variable] = []

		def err_thunk() -> bool:
			self._cfg.narrow( recv_var.stem, err_member )
			err_bind_name = f'__ot_err_{unique}'
			err_bind_var = self._declare_hidden_local( err_bind_name, error_cls, node )
			extract = ast.Assign(
				targets = [ ast.Name( id = err_bind_name, ctx = ast.Store() ) ], value = self.lowering._synth_name( recv_var.stem, node ),
			)
			ast.copy_location( extract, node )
			self._lower_stmt( extract )
			mapped_call = ast.Call(
				func = self.lowering._synth_name( mapper_var.stem, node ), args = [ ast.Name( id = err_bind_name, ctx = ast.Load() ) ], keywords = [],
			)
			ast.copy_location( mapped_call, node )
			# a GOTO-based ir.Raise (the all_covered case - every leaf
			# dispatches straight into a handler, see _stmt_Raise's own
			# short-circuit) never unwinds anything - unlike a real
			# function-level return/propagation, which walks and replays
			# the ENTIRE pending stack (cfg.return_()), a jump into a
			# handler INSIDE the same function correctly leaves that to
			# whatever scope it lands in. Every hidden local THIS call
			# introduced (err_bind_name below, and recv_var itself - an
			# OUTER local, but one this specific call is the only reason
			# it's still holding a reference by this point) would then
			# never get released on this specific path at all - confirmed
			# via a real repro (both leaked, "-- live RC objects (2) --").
			# recv_var's own payload is already safely aliased into err_
			# bind_name by this point, so releasing recv_var here (its
			# extraction already done, nothing else in err_thunk touches
			# it again) is correct, not premature.
			#
			# _internal_decref_var(recv_var) directly - NOT a synthesized
			# compiler.__internal_decref__(recv_var) ast.Call, which is what
			# this used to be. recv_var was narrowed (cfg.narrow, above) to
			# its Err member just before this point - re-resolving a
			# synthesized ast.Name reference to it through _lower_expr,
			# post-narrowing, returns a freshly-EXTRACTED ir.Temp (the
			# narrowed payload itself), not the union Variable - so
			# manually_decreffed() silently cancelled the wrong thing (an
			# unrelated Temp, via its own _temp_states branch) while
			# recv_var's REAL binding/epilogue-stack entry stayed live,
			# left for the raise below to redundantly release a second
			# time. Confirmed via a real repro + compiler-side trace.
			# _internal_decref_var operates on the Variable object directly,
			# bypassing that re-resolution entirely - see its own docstring.
			#
			# cfg.rc_leaves(receiver.type), NOT receiver.type.is_rc() -
			# receiver.type is Result[_,_], a TaggedUnion struct, never a
			# bare RC pointer itself (is_rc() is narrower than rc_leaves(),
			# same "TaggedUnion with RC members" gap _lower_compiler_decref's
			# own comment documents) - kept for consistency with every other
			# guard of this shape in this file, though not itself what was
			# broken here (both happened to agree on this specific type).
			if cfg.rc_leaves( receiver.type ):
				self._internal_decref_var( recv_var )
			# the mapper call is lowered directly here (never embedded
			# unevaluated inside a fresh ast.Raise, unlike a plain hand-
			# written `raise mapper(err)`) - both err_bind_name and mapper_var
			# themselves need explicit teardown BETWEEN the call and the
			# raise (see below), and _stmt_Raise's own all_covered dispatch
			# is an unconditional goto with no reachable code after it, so
			# anything still needing to run has to happen BEFORE it, not
			# after. Kept mapped_value a bare, never-named operand (never
			# bound via ast.Assign - see _raise_value's own docstring)
			# rather than a hidden Variable: a NAMED local raised from
			# inside this confined Err branch has no natural unwind at all
			# (a GOTO-based ir.Raise never walks back through
			# _lower_binary_branch's own confinement - confirmed via a real
			# repro, the mapper's own result permanently leaking even after
			# trying cfg.manually_decreffed() on it, which only cancels an
			# ALREADY-firing release, not one that would never fire to
			# begin with). untrack_temp() inside _raise_value is what
			# correctly hands a bare temp's ownership off instead.
			#
			# mapper_var.type is EITHER Ptr[Callable[...]] (a plain
			# function/non-capturing lambda) or a bare ClosureType (a real
			# capturing closure) - mirrors _expr_Lambda's own identical
			# two-shape fallback for resolving fn_type
			mapper_callable_type_resolved = self.lowering._type_resolver._callable_type_of( mapper_var.type )
			if mapper_callable_type_resolved is None and isinstance( mapper_var.type, ClosureType ):
				mapper_callable_type_resolved = mapper_var.type
			mapped_value = self._lower_expr( mapped_call, mapper_callable_type_resolved.return_type )
			if error_cls.is_rc():
				# fully consumed as the mapper's own argument - same "read a
				# container-owned value into a local, explicitly decref it"
				# idiom list.__del__/dict's own _release_key/_release_value
				# already use elsewhere in this stdlib
				self._internal_decref_var( err_bind_var )
			if mapper_var.type.is_rc():
				# a real capturing closure (ClosureType is_rc()) - declared
				# in the try body (before this branch even starts), so its
				# own OWNED epilogue entry would normally survive to be
				# released wherever the enclosing function actually returns
				# (return_()'s own whole-stack walk). But _stmt_Try's own
				# handler lowering restores the CFG back to the try's ENTRY
				# snapshot before running ANY handler body (see _stmt_Try's
				# own comment) - silently dropping, not releasing, every
				# entry the try body itself pushed, mapper_var included.
				# Confirmed via a real repro: OrThrowMapperTests.
				# test_capturing_closure_mapper_rc_correct leaked exactly
				# the closure + its own captured-env allocation, on the Err
				# path only (the Ok path's own real function-level `return`
				# still correctly walks and releases it, since it never
				# goes through a handler dispatch at all).
				self._internal_decref_var( mapper_var )
			self._raise_value( mapped_value, node )
			return True

		def ok_thunk() -> bool:
			self._cfg.narrow( recv_var.stem, ok_member )
			ok_bind_name = f'__ot_ok_{unique}'
			ok_var = self._declare_hidden_local( ok_bind_name, result_ok_type, node )
			extract = ast.Assign(
				targets = [ ast.Name( id = ok_bind_name, ctx = ast.Store() ) ], value = self.lowering._synth_name( recv_var.stem, node ),
			)
			ast.copy_location( extract, node )
			self._lower_stmt( extract )
			# see or_return_with_mapper's own identical ok_thunk comment: recv_
			# var's own reference (from whatever originally constructed it) is
			# never released on this arm otherwise - only err_thunk's own
			# mirror release does that, for its own leg.
			if cfg.rc_leaves( receiver.type ):
				self._internal_decref_var( recv_var )
			# this expression hands ok_var's single retained reference
			# straight to its own caller (no fresh retain of its own) - move
			# it out via manually_decreffed() (no preceding decref, per its
			# own "mirrors move()'s own cancellation exactly") so ok_var's
			# own scope-exit release doesn't ALSO fire on top of whatever the
			# caller does with it.
			for instr in self._cfg.manually_decreffed( ok_var ):
				self._emit( instr )
			ok_var_holder.append( ok_var )
			return False

		self._lower_binary_branch( is_err_cond, node, err_thunk, ok_thunk )
		# err_thunk always terminates (a real `raise` - either a goto into
		# a handler, or a real `return` when uncovered - never falls
		# through to here) - ok_var is therefore always live whenever
		# control actually reaches this point, exactly like or_return_
		# with_mapper's own identical reasoning.
		ok_operand = ok_var_holder[0]
		if not want_result:
			self._emit( ir.MarkUsed( operand = ok_operand ))
			return None
		return ok_operand

	def _dispatch_leaves_against_try_stack(
		self, all_leaves: list[Type], *, exclude: 'ir.Operand | None' = None,
	) -> tuple[list[ir.ThrowLeaf],list[Type]]:
		''' matches each of `all_leaves` against every enclosing try's own
		handlers, innermost first (self._try_stack - see TryContext's own
		docstring) - the first handler found across ANY try context on the
		stack wins, not just the innermost try's own handlers; a leaf
		matching nothing anywhere on the stack is left uncovered, for the
		caller (_emit_or_throw / _stmt_Raise) to propagate to the function's
		own return type instead. Shared by BOTH .or_throw() and `raise` so
		this walk - and the matched-handler bookkeeping below (Change 3's
		dead-except check, see TryHandler's own docstring) - lives in
		exactly one place. Sound without any new CFG machinery: see
		_stmt_Try's own docstring for why an outer TryContext is still safe
		to mutate mid-recursion, and why an inner try's goto reaching an
		outer handler's label needs no extra confinement.

		`exclude` is `raise EXPR`'s own raised value, when it's a plain
		Variable (_stmt_Raise's own tracked_operand - None for or_throw(),
		whose payload is a struct field of the receiver, never an
		independently-tracked entry to begin with) - passed straight
		through to each matched leaf's own unwind_confined() call below, so
		its ownership transfers into the handler's bind instead of being
		released twice. Each matched leaf gets its OWN epilogue, bounded to
		the SPECIFIC ctx that covered it (see ir.ThrowLeaf.epilogue's own
		docstring) - a nested try's outer handler catching a leaf the inner
		try doesn't needs a DEEPER unwind (back to the outer try's own
		entry) than a leaf the inner try catches itself. '''
		dispatch: list[ir.ThrowLeaf] = []
		covered_leaves: list[Type] = []
		for leaf in all_leaves:
			handler = None
			covering_ctx = None
			for ctx in reversed( self._try_stack ):
				handler = next( ( h for h in ctx.handlers if leaf in h.leaves ), None )
				if handler is not None:
					covering_ctx = ctx
					break
			if handler is not None:
				handler.matched = True
				epilogue = self._cfg.unwind_confined( covering_ctx.entry_stack_depth, exclude )
				dispatch.append( ir.ThrowLeaf( leaf = leaf, bind = handler.raise_value_var, label = handler.label, epilogue = epilogue ))
				covered_leaves.append( leaf )
		return dispatch, covered_leaves

	def _emit_or_throw(
		self, node: ast.AST, receiver: ir.Operand, want_result: bool, *,
		alternatives: str, pre_checked: bool = False, receiver_pending_start: int|None = None,
	) -> ir.Operand|None:
		''' The real body of .or_throw() (explicit or auto-inserted alike) -
		extracted from _lower_or_throw so every auto-insertion site (see its
		own docstring) can reuse the exact same per-leaf dispatch/emission,
		not just the literal `.or_throw()` call syntax. Each leaf of the Err
		branch is matched against every enclosing try's own handlers,
		innermost first (_dispatch_leaves_against_try_stack), falling back
		to EXACTLY or_return()'s own propagate-to-the-caller behavior for
		any leaf left uncovered by every enclosing try (or every leaf, when
		there's no enclosing try at all - self._try_stack empty).

		`pre_checked`, when True, skips this method's own
		_require_or_throw_return coverage call - for a caller (AugAssign's
		combined get+set subscript chain) that already validated coverage
		up front via _require_chained_result_return, so the same gap isn't
		reported twice under two different messages.

		Assumes receiver.type is already known Result[_,_]-shaped (every
		caller either recognized real `.or_throw()` syntax, which already
		checked this, or is _auto_or_throw, which only reaches here after
		its own _result_shape probe succeeded).

		Can't just reuse _consume_checked_result (it always emits an
		unconditional OrReturn/OrJump - there's no way to thread a per-leaf
		dispatch table through it) - this replays its checked-result
		bookkeeping (unchecked-result clearing, epilogue-label lookup) by
		hand instead, then builds ir.OrThrow directly.

		Uncovered-leaf propagation out of a multi-statement @inline splice's
		pre-return statements mirrors _consume_checked_result's own
		self._inline_scope_vars[-1] carve-out exactly (same PLAN_RETURN_
		INFERENCE.md sentinel-state rejection too - see its own comment). '''
		if self._in_inline_splice_prelude and not self._inline_scope_vars:
			self.lowering.discovery.fail(
				f'@inline: or_throw()/raise that could propagate an error is not yet supported before the '
				f'final return of a multi-statement body whose own return type is still being inferred: {ast.unparse(node)}',
				node,
			)
		shape = self.lowering._type_resolver._result_shape( receiver.type )
		assert shape is not None
		result_type, error_cls = shape
		result_cls = self.lowering.discovery.find_name( 'Result', node )

		all_leaves = self.lowering._type_resolver._atomic_leaves( error_cls )
		dispatch, covered_leaves = self._dispatch_leaves_against_try_stack( all_leaves )

		if not pre_checked:
			self.lowering._type_resolver._require_or_throw_return(
				node, result_cls, error_cls, covered_leaves, alternatives, fn = self._current_fn,
			)

		# same bookkeeping _consume_checked_result runs for or_return() -
		# see its own comment on why both are needed
		if isinstance( receiver, Variable ):
			self._cfg.clear_result( receiver.stem )
		try:
			self._cfg.check_unchecked_results( None )
		except CompileError as e:
			self.lowering.discovery.fail( str( e ), node )

		# see _consume_checked_result's own identical call/comment - same
		# construct (a checked-Result value with an early-exit Err leg),
		# just a different emission path (ir.OrThrow instead of OrReturn/
		# OrJump, since or_throw() needs a per-leaf dispatch table
		# _consume_checked_result has no way to thread through)
		if receiver_pending_start is not None:
			self._flush_new_pending_temps( receiver_pending_start, receiver )
		unwrapped = self._new_temp( result_type )
		# see _consume_checked_result's identical fresh_temp() call/comment -
		# without this, a discarded/bare-operand unwrapped payload (e.g. used
		# only as a `!=` operand, never bound) is never decref'd: a real leak,
		# confirmed via `s[0] != 'h'` on an RC str
		self._cfg.fresh_temp( unwrapped, result_type )
		all_covered = len( covered_leaves ) == len( all_leaves )
		if all_covered:
			# every leaf dispatches straight into a handler - no
			# propagation path exists at all, so the enclosing function's
			# own epilogue/return-type machinery is never touched (also
			# matches _require_or_throw_return's own "no requirement at
			# all" contract above)
			self._emit( ir.OrThrow( dest = unwrapped, value = receiver, dispatch = dispatch ))
		else:
			tracked_operand = receiver if isinstance( receiver, Variable ) else None
			# see _consume_checked_result's own identical inline_scope carve-
			# out/comment - reached from inside a multi-statement @inline
			# splice's pre-return statements, the uncovered-leaf propagation
			# must land in the SPLICE's own result_var/exited_flag/merge_label,
			# never the caller's real return_slot/epilogue - always via the
			# direct return_()-replay shape (mirrors OrReturn's own inline_exit,
			# never OrJump's shared-scope-label optimization - simpler, and
			# just as correct, at the cost of not sharing ladder code across
			# multiple early-exit sites within the same splice)
			inline_scope = self._inline_scope_vars[-1] if self._in_inline_splice_prelude and self._inline_scope_vars else None
			if inline_scope is not None:
				replay = self._cfg.return_( tracked_operand, lambda: self._build_is_err_check( node ))
				self._cfg.mark_inline_scope_captured()
				self._emit( ir.OrThrow( dest = unwrapped, value = receiver, dispatch = dispatch, epilogue = replay, inline_exit = inline_scope ))
			else:
				label = self._cfg.current_epilogue_label( tracked_operand )
				if label is not None:
					self._emit( ir.OrThrow(
						dest = unwrapped, value = receiver, dispatch = dispatch, target = label, return_slot = self._return_value_var,
					))
				else:
					replay = self._cfg.return_( tracked_operand, lambda: self._build_is_err_check( node ))
					self._emit( ir.OrThrow( dest = unwrapped, value = receiver, dispatch = dispatch, epilogue = replay ))

		# same borrow-then-incref rationale as _consume_checked_result's own
		# identical tail (see its own comment) - a no-op for a non-RC
		# result_type
		for instr in self._cfg.incref( unwrapped.type, unwrapped ):
			self._emit( instr )

		if not want_result:
			self._emit( ir.MarkUsed( operand = unwrapped ))
			return None
		return unwrapped

	def _auto_or_throw(
		self, node: ast.AST, value: ir.Operand, alternatives: str, *,
		want_result: bool = True, pre_checked: bool = False, receiver_pending_start: int|None = None,
	) -> ir.Operand|None:
		''' The ONE general rule this whole file's auto-consumption story
		now boils down to: whenever a Result[T,E]-shaped value is (1) a
		discarded statement (want_result=False) or (2) flowing into a
		context that wants T directly rather than the whole Result (see
		_coerce_or_check_operand's own case-2 hook) - insert an implicit
		.or_throw() (NOT .or_return()) exactly as if the user had written
		it. Since or_throw() degrades to or_return()'s own unconditional-
		propagate behavior whenever there's no enclosing try (self.
		_try_stack empty, or every leaf uncovered), this is the single
		mechanism that now backs checked arithmetic, __setitem__/AugAssign,
		a bare discarded fallible call, AND `x: i32 = arr[0]` alike - see
		PLAN_CHECKED_ARITHMETIC_GAP.md (now closed) for the gap this
		unifies away.

		A non-Result `value` passes straight through unchanged - not every
		caller's value is necessarily fallible (mirrors _maybe_consume_
		result's own identical "pass through, don't reject" contract).

		Reached from inside a multi-statement @inline splice's pre-return
		statements exactly like any other call site now (ir.OrThrow.
		inline_exit) - no separate carve-out needed any more; _emit_or_throw
		itself routes the uncovered-leaf fallback into the splice's own
		result_var/exited_flag/merge_label. '''
		shape = self.lowering._type_resolver._result_shape( value.type )
		if shape is None:
			return value if want_result else None
		return self._emit_or_throw(
			node, value, want_result, alternatives = alternatives, pre_checked = pre_checked,
			receiver_pending_start = receiver_pending_start,
		)

	def _maybe_auto_consume_result( self, node: ast.AST, operand: ir.Operand, expected_type: Type|None, alternatives: str, *, context: str|None = None ) -> ir.Operand|None:
		''' case 2 of the general auto-or_throw() rule (see _auto_or_throw's
		own docstring) as a probe, not an unconditional coercion: returns the
		freshly unwrapped-and-re-coerced operand when `operand` is
		Result[T,E]-shaped AND `expected_type` is NOT itself Result-shaped,
		else None (not applicable - the caller keeps its OWN existing
		operand/mismatch handling unchanged). _coerce_or_check_operand uses
		this directly (its ordinary strict=True path); _stmt_Return needs its
		own separate call to the same probe since it deliberately lowers
		with strict=False (see its own comment on why) and so never reaches
		_coerce_or_check_operand's own hook - both must agree on the exact
		same guard, hence one shared helper instead of two copies of it. '''
		if ( expected_type is None
				or self.lowering._type_resolver._result_shape( operand.type ) is None
				or self.lowering._type_resolver._result_shape( expected_type ) is not None ):
			return None
		consumed = self._auto_or_throw( node, operand, alternatives, want_result = True )
		assert consumed is not None # want_result=True above guarantees this
		return self._coerce_or_check_operand( consumed, expected_type, node, context = context )

	def _finish_call_result( self, node: ast.AST, result: ir.Operand|None, want_result: bool ) -> ir.Operand|None:
		''' shared tail for every call-lowering path (_lower_inline_call,
		_infer_return_only_type_params_inline, _emit_generic_call, and
		_lower_call's own shared tail) once the call's real result operand
		is already computed/emitted, regardless of want_result. Case 1 of
		the general auto-or_throw() rule (see _auto_or_throw's own
		docstring): a discarded (want_result=False) Result gets implicitly
		.or_throw()'d instead of the old hard "returns a Result that is
		discarded here" compile error. A non-Result, or an already-wanted,
		result passes straight through unchanged.

		A discarded NON-Result RC value (e.g. `xs.pop().unwrap(msg) -> str`
		called as a bare statement - unwrap() already extracted the Ok
		payload, its own return type is plain str, never Result-shaped) is
		the other half of "produced, immediately discarded, never bound to
		a name": every call-emission tail that widens its own dest-creation
		condition to also cover this case (force_result's sibling,
		`discard_rc` - grep for it) still needs SOMETHING to actually
		release the object once it has a dest at all, or creating that dest
		only moved the leak from "never captured" to "captured, never
		released". Handled centrally here rather than duplicated at each of
		those call sites: release it in place, exactly like any other
		discarded-fresh-temp cleanup elsewhere in this file (e.g.
		_coerce_or_check_operand's own "was_fresh: decref+untrack" pattern).
		Confirmed as a real leak via `xs.pop().unwrap(msg)` as a bare
		statement (also hit for real by lib/os.py's normpath()). '''
		if want_result or result is None:
			return result
		if not cfg.is_result_type( result.type ) and result.type is not None and result.type.is_rc():
			# a discarded plain (non-Result) RC return - the callee already
			# constructed/returned it, nothing else will ever capture or
			# release it otherwise (a straight leak, not an auto-or_throw
			# case - _auto_or_throw itself just passes a non-Result value
			# through unreleased)
			for instr in self._cfg.decref( result.type, result ):
				self._emit( instr )
			self._cfg.untrack_temp( result )
			return None
		# consumed (and reset) here, the ONE place case-1 auto-or_throw
		# actually reaches _emit_or_throw's own early-return propagate path -
		# see self._discarded_call_pending_start's own comment. Only non-None
		# when this call IS the bare discarded-statement call itself (set by
		# _stmt_Expr immediately before lowering it); a nested call reached
		# while building this one's own args/receiver always has want_result
		# = True and returns above before ever reaching here, so it can never
		# steal/clear this out from under the outer call
		pending_start, self._discarded_call_pending_start = self._discarded_call_pending_start, None
		self._auto_or_throw(
			node, result, self.lowering._AUTO_CONSUME_ALTERNATIVES, want_result = False,
			receiver_pending_start = pending_start,
		)
		return None
