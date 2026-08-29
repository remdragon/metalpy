# stdlib imports:
import ast
from typing import Callable

# local imports:
import arithmetic_mode
import cfg
import ir
from discovery import reject_reserved_c_identifier
from errors import CompileError, RedundantCompilationError
from mpy_types import (
	Name, Type, Variable, Function, Module, Specialization, CStruct, TypeVar, RCClass, Scalar, FixedArrayType,
)

from lowering_shared import _IPLACE_BINOP_DUNDER

class StmtLoweringMixin:
	''' core statement dispatch (_stmt_*) and assignment/target-resolution helpers - mixed into FunctionLowering (lowering.py), which
	see for the shared instance state (self._instructions, self._cfg, self.lowering,
	etc.) every method here reads and writes. Never instantiated on its own;
	split out of lowering.py purely to keep that file to a manageable size - see
	lowering.py's own class docstring and FunctionLowering's base-class list for
	the full set of sibling mixins this one is composed with. '''


	def _emit( self, instr: ir.Instruction ) -> None:
		# a Call/CallIndirect/Allocate's dest is always a genuinely fresh,
		# owned value from the caller's perspective (same rule _is_aliasing_
		# expr already encodes for Call; Allocate is fresh by definition;
		# CallIndirect - calling THROUGH a Ptr[Callable[...]]/closure value,
		# e.g. a real capturing closure passed to or_return(mapper) - is the
		# identical "fresh owned handoff" shape as an ordinary Call, just
		# reached through a different ir.Instruction) - registering it here,
		# centrally, at the exact moment it's actually emitted, is what
		# guarantees every one of these sites is covered instead of needing
		# individual fresh_temp() calls hunted down at each of the many
		# places that build a Call/CallIndirect/Allocate (plain calls,
		# generic calls, conditional dispatch, union-receiver dispatch,
		# struct/union construction, ...). CallIndirect was missing from
		# this check entirely until confirmed via a real repro: its own
		# result, passed directly into a retaining constructor (e.g.
		# `Result.Err(some_closure())`), never got queued for the ordinary
		# end-of-statement decref that cancels the constructor's own
		# retain out to net-one - a permanent one-reference-per-call leak
		# for any RC-typed indirect/closure call result, regardless of
		# what consumes it (not specific to or_return(mapper) - the same
		# repro leaks with a bare `return Result.Err(closure())`, no
		# generator/or_return involved at all). Gated on self._cfg, not
		# self._current_fn (STALE reasoning here used to say lower_global()
		# never constructs a CFGState at all, leaving self._cfg pointing at
		# whatever function was lowered most recently - no longer true:
		# run_global()/run_deinit_epilogue() both build their own fresh
		# CFGState up front, same as run() does for a real function, so
		# self._cfg is exactly as reliable a signal there as self._current_
		# fn is for an ordinary function body). The self._current_fn-gated
		# version left every module-level global's own initializer
		# expression completely untracked - any intermediate RC temp NOT
		# equal to the final stored value (e.g. a fallible initializer's own
		# raw Result[T,E], still holding its own internal reference after
		# .unwrap()'s narrowed extraction) was silently never registered for
		# release at all, not merely un-flushed. Confirmed via a real repro:
		# `r_eol: re.Pattern = re.compile(...).unwrap(...)` at module scope
		# leaked the compile()'s own Result permanently, for the whole
		# process lifetime, not just past one statement's end.
		if self._cfg is not None and isinstance( instr, ( ir.Call, ir.CallIndirect, ir.Allocate )) and isinstance( instr.dest, ir.Temp ):
			self._cfg.fresh_temp( instr.dest, instr.dest.type )
		if self._current_fn is not None and isinstance( instr, ir.Allocate ):
			if self._current_fn.stem == '$$__new__':
				# a synthesized, SHARED constructor - every real Foo(...) call
				# site in the program reuses this one compiled function (see
				# type_resolver.py's _synthesize_rcclass_constructor), so
				# self._current_fn.file/self._current_lineno here are just
				# this wrapper's own synthesized (class-definition) location,
				# useless for telling apart which of possibly many real call
				# sites allocated a given live object. The wrapper's own
				# hidden __alloc_loc parameter carries the REAL caller's
				# location instead, baked in as a real per-call-site literal
				# by _try_lower_construct_call - point Allocate.loc at that
				# parameter (a runtime operand, not a static string) instead.
				alloc_loc_param = next( p for p in self._current_fn.parameters or [] if p.stem == '__alloc_loc' )
				instr.loc = alloc_loc_param
			else:
				instr.loc = f'{self._current_fn.file}:{self._current_lineno}'
		if self._current_fn is not None:
			self._check_self_escape_in( instr )
		if self._current_fn is not None and isinstance( instr, ir.Call ):
			self._mark_errdefer_retained_args( instr )
		self._instructions.append( instr )

	def _mark_errdefer_retained_args( self, instr: ir.Call ) -> None:
		# instr.target's own errdefer_retained_params (Lowering.
		# _ensure_errdefer_retained_params, an AST-derived fact - computed
		# here, lazily, since reachability-ordered lowering means the
		# callee itself may not have been lowered yet) says one of ITS
		# parameters gets conditionally incref'd by a defer/errdefer
		# somewhere in its body - if THIS call passes one of the CALLER's
		# own plain locals into that exact parameter position, that local
		# may come back with an extra, compiler-invisible reference
		# attached. Flagging it (cfg.mark_possibly_retained) is what stops
		# a later compiler.decref(...) on it from being mistaken for the
		# release of its own separately-tracked binding - see
		# manually_decreffed()'s own use of this. self/cls receivers are
		# excluded from Function.parameters entirely (see ir.Call's own
		# comment) - not handled here, no test currently needs it.
		retained = self.lowering._ensure_errdefer_retained_params( instr.target )
		if not retained:
			return
		for param, arg in zip( instr.target.parameters or [], instr.args ):
			if param.stem in retained:
				self._cfg.mark_possibly_retained( arg )
		for name, arg in instr.kwargs.items():
			if name in retained:
				self._cfg.mark_possibly_retained( arg )

	def _check_self_escape_in( self, instr: ir.Instruction ) -> None:
		# self can only be used as the receiver of `self.attr`
		# (GetAttr.obj/SetAttr.obj, deliberately excluded here) until
		# __init__ finishes constructing it - see cfg.check_self_escape().
		# Checking every OTHER operand field centrally, at emission time,
		# covers every site that could hand self off somewhere it shouldn't
		# without hunting each one down individually - same centralization
		# fresh_temp() already uses above. A no-op outside __init__
		# (check_self_escape() itself short-circuits when nothing's under
		# construction) - the instruction-type gate below also means this
		# never touches self._cfg before it exists (FuncStart, emitted
		# before CFGState is constructed, matches none of these types)
		operands: list[ir.Operand] = []
		if isinstance( instr, ir.Call ):
			# is_super_init_call's receiver (self, mid-construction) is
			# deliberately excluded too, alongside GetAttr.obj/SetAttr.obj
			# above - see ir.Call.is_super_init_call's own comment. args/
			# kwargs still go through the ordinary check below regardless
			if instr.receiver is not None and not instr.is_super_init_call:
				operands.append( instr.receiver )
			operands += instr.args
			operands += instr.kwargs.values()
		elif isinstance( instr, ir.Assign ):
			operands.append( instr.src )
		elif isinstance( instr, ir.Return ):
			if instr.value is not None:
				operands.append( instr.value )
		elif isinstance( instr, ir.SetItem ):
			operands.append( instr.value )
		elif isinstance( instr, ir.Allocate ):
			operands += instr.fields.values()
		for operand in operands:
			try:
				self._cfg.check_self_escape( operand, self._current_fn.qualname )
			except CompileError as e:
				self.lowering.discovery.fail_loc( str( e ), self._current_fn.file, self._current_fn.line )

	def _new_temp( self, t: Type ) -> ir.Temp:
		temp = ir.Temp( type = t, id = self._temp_id )
		self._temp_id += 1
		self._pending_temps.append( temp )
		self._emit( ir.DeclareTemp( temp = temp ))
		return temp

	def _new_label( self, prefix: str ) -> str:
		label = f'__{prefix}_{self._label_id}__'
		self._label_id += 1
		return label

	def _mark_fresh_local_declared( self, stem: str ) -> bool:
		''' called at every genuinely fresh local declaration - one reached
		with no LIVE binding for `stem` already in fn.names (every call
		site here already confirmed that itself, via _existing_local_or_
		none/_stmt_AnnAssign's own check, before getting here). Returns
		True if `stem` needs its own disambiguated C identifier
		(Variable.needs_uid_suffix) because it was declared before AND has
		since been del'd (only del removes a live name from fn.names
		without also going through here again) - False for a genuinely
		first-ever declaration. '''
		needs_suffix = stem in self._ever_declared_stems
		self._ever_declared_stems.add( stem )
		return needs_suffix

	# --- statements ------------------------------------------------------------

	def _lower_stmt( self, node: ast.stmt ) -> None:
		self._current_lineno = getattr( node, 'lineno', self._current_lineno )
		# _pending_temps is shared/mutable rather than passed explicitly, so a
		# statement whose own handler recursively lowers nested statements
		# (currently only _stmt_With) must not let those nested calls' own
		# resets/flushes clobber this call's view of it - save/restore around
		# the whole thing, same idea as scope_context's stack push/pop
		outer_pending = self._pending_temps
		self._pending_temps = []
		# PLAN_GENERATORS.md's defer/errdefer phase (Mechanism 2) - only
		# OVERRIDES self._generator_armed_defer_sites when `node` itself
		# carries type_resolver.py's own tag (_tag_armed_defer_sites tags
		# only the top-level statement of each preamble/segment slice, not
		# every nested descendant); otherwise this statement's own
		# recursive lowering (e.g. an ordinary nested if's own body)
		# simply inherits whatever an ENCLOSING tagged statement already
		# pushed, exactly like _arithmetic_mode's own stack semantics
		outer_armed = self._generator_armed_defer_sites
		tagged = getattr( node, 'generator_armed_defer_sites', None )
		if tagged is not None:
			self._generator_armed_defer_sites = tagged
		try:
			method = getattr( self, f'_stmt_{node.__class__.__name__}', None )
			if method is None:
				self.lowering.discovery.fail( f'unsupported statement: {ast.unparse(node)}', node )
			method( node )
			# a no-op by the time this runs for a Return (see _stmt_Return's
			# own comment - it flushes _pending_temps ITSELF, before its own
			# ir.Return/ir.Jump, precisely so this generic post-statement
			# flush - unconditionally emitted AFTER method(node) returns,
			# i.e. AFTER any unconditional terminator that statement itself
			# already emitted - never lands as dead code following it)
			self._flush_pending_temps()
		finally:
			self._pending_temps = outer_pending
			self._generator_armed_defer_sites = outer_armed

	def _flush_pending_temps( self ) -> None:
		''' decref+DeleteTemp every still-pending temp (reverse declaration
		order), then clear the list. A temp genuinely fresh_temp()-
		registered (see _emit) and never consumed by assign()/return_()/
		untrack_temp()/move()/field_value() along the way (e.g. `foo(
		SomeClass() )` where SomeClass() is passed into a plain, non-move[T]
		parameter - nothing ever untracks it) still needs its own decref
		right here, at the natural end of the temporary's own expression-
		scoped lifetime. A no-op for every already-consumed temp (already
		untracked by whichever hook consumed it) and every non-RC temp
		(never registered in the first place). Factored out of _lower_stmt
		so _stmt_Return can call it explicitly BEFORE its own terminator
		(ir.Return/ir.Jump) instead of relying on _lower_stmt's own post-
		method call, which - for every OTHER statement kind, fine, since
		none of them emit an unconditional jump/return of their own - would
		otherwise land as unreachable code right after one. '''
		for t in reversed( self._pending_temps ):
			for instr in self._cfg.delete_temp( t ):
				self._emit( instr )
			self._emit( ir.DeleteTemp( temp = t ))
		self._pending_temps = []

	def _flush_new_pending_temps( self, start_index: int, *keep: ir.Operand ) -> None:
		''' shared by _consume_checked_result/_emit_or_throw: flushes every
		pending temp added SINCE `start_index` (an index into
		self._pending_temps, snapshotted by the caller right before
		lowering the checked receiver expression - see _lower_or_return/
		_lower_or_throw's own `receiver_pending_start`), except those in
		`keep` (the checked receiver/result value itself, still needed by
		the OrReturn/OrJump/OrThrow/Unwrap instruction about to be
		emitted). Restores `keep`'s own still-pending entries afterward so
		their normal, later lifecycle (the natural end-of-statement flush,
		same as always) is completely unaffected.

		Deliberately scoped to start_index onward, NOT "everything
		currently pending": an OUTER, still-in-progress expression this
		checked value is only PART of (e.g. the first half of a chained
		`k + str(':') + v.or_return()` concatenation) can have its OWN
		still-needed pending temp sitting BEFORE start_index - flushing
		that too corrupts it out from under the rest of the expression,
		confirmed as a real bug (not hypothetical): an earlier version of
		this fix flushed unconditionally and caused exactly that shape to
        read a freed string's length as its concatenation grew, aborting
		with "out of memory" on the very first nested json.dumps() call in
		the test suite - lib/json.py's own `_json_escape_string(k) + ':' +
		_dump_value(v).or_return()` in its Object dump arm.

		What's actually left to flush after start_index, once `keep` is
		excluded, is exactly the receiver's OWN sub-expression temps -
		most commonly Part B's own retain-on-read for a chained field
		receiver used only to build the checked value, e.g. `self.a.b.
		method(...).or_return()` - which must be released regardless of
		which leg (Ok/Err) actually fires, since neither leg needs it
		afterward. Without this it's dead code exactly like _stmt_If's own
		sibling bug: the checked call's Err leg is a real early exit (a
		genuine `return`/`goto` under the hood, same as or_return()'s own
		declared spec says), so it skips right past the ordinary once-per-
		statement flush that would otherwise release it - confirmed via a
		real compiler.refcount()/AddressSanitizer repro (`self.__raw.
		_ptr_at(idx).or_return()` permanently over-retained self.__raw, a
		leak found only once _stmt_If's own sibling bug was fixed and
		stopped masking it). `keep` uses Temp.id, not Python identity -
		operands round-trip through dataclasses that don't override
		__eq__/__hash__ for this purpose. '''
		keep_ids = { k.id for k in keep if isinstance( k, ir.Temp ) }
		older = self._pending_temps[:start_index]
		newer = self._pending_temps[start_index:]
		kept = [ t for t in newer if t.id in keep_ids ]
		self._pending_temps = [ t for t in newer if t.id not in keep_ids ]
		self._flush_pending_temps()
		self._pending_temps = older + kept

	def _incref_aliasing_return( self, node_expr: ast.expr, value: 'ir.Operand|None', *, force: bool = False, wrap_fresh: bool = False ) -> 'ir.Operand|None':
		''' shared by _stmt_Return and @inline splicing (_lower_inline_call/
		_splice_multi_statement_inline_body): an ALIASING return expression
		(self.lowering._is_aliasing_expr - `return self`/`return self.x`)
		hands back a reference someone else still independently owns, so the
		caller needs its own +1 - regardless of whether the return happens
		through a real call boundary or is spliced in directly. Skipping this
		for the spliced case (confirmed by a real refcount() repro) silently
		drops the Incref an @inline'd `return self` would otherwise get from
		a real, non-inlined call to the same function.

		`force` bypasses the has_live_entry() check below - needed by the
		@inline splice callers specifically: self/a parameter is bound
		zero-copy (SAME Variable identity as whatever the caller passed in -
		see _lower_inline_call's own "no _cfg_assign/incref here, deliberately"
		comment), so has_live_entry(value) would answer "does the CALLER's own
		operand happen to be a live owned local in the OUTER scope" instead of
		"is this splice's self/parameter borrowed" - the wrong question
		whenever the caller's argument was itself a plain owned local (exactly
		the str(s) repro: s has its own live entry in main(), so an unforced
		check wrongly concluded "already a move, no Incref needed"). @inline
		splice callers already know from their own binding loop that self/
		every parameter is always treated as borrowed at the splice boundary
		(same loop, same comment), so they pass force=True for those; a
		multi-statement splice's own pre-return-declared local (a real,
		splice-scoped self._cfg entry, not aliased to any outer identity)
		still needs the ordinary has_live_entry check, so force stays False
		for those.

		`wrap_fresh` (also @inline-splice-only): the Incref emitted below
		attaches to `value`'s own EXISTING identity - a Variable (self/a
		parameter) that already has its own, independent release scheduled
		elsewhere. _stmt_Return/the generator caller are fine with that:
		`value` there flows on into the function's own real return-value
		slot, which is what actually carries this extra unit of ownership
		out to the real caller. An @inline splice has no such slot - its
		"result" IS this call EXPRESSION's own value, used directly wherever
		the call appears (e.g. passed straight into another call's argument
		list). If that use is inline and unbound (never assigned to a fresh
		named local first, which WOULD independently track it), this extra
		Incref has nothing left to track it at all - confirmed via a real
		repro (sink(s.__str__())), str.__str__'s own `return self` inlined
		leaked one str per call; list.__repr__'s own `parts.append(str(val))`
		inside a loop hit the identical shape. Materializing a genuinely fresh
		ir.Temp and registering it via cfg.fresh_temp() here - exactly what a
		real, non-inlined call's own dest already gets - makes the splice's
		result participate in the same pending-temp/fresh-temp release
		machinery an ordinary call result does. '''
		if value is None or not self.lowering._is_aliasing_expr( node_expr, value ):
			return value
		if self._cfg.is_fresh_temp( value ):
			# value LOOKS aliasing from node_expr's own AST shape (a bare
			# Name/Attribute node), but it's already a freshly-owned value -
			# a narrowed read of a protected global performs its own
			# protected retain up front and registers the result via
			# fresh_temp() (see _expr_Name's own comment - this is
			# `return localtz_style_global` after narrowing, the exact
			# shape that motivated this whole mechanism). Increffing again
			# here would double-own it; nothing else to do - it's already
			# tracked the same way an ordinary Call/Allocate result being
			# returned is, and existing ownership-transfer handling covers
			# that case already.
			return value
		if force or not self._cfg.has_live_entry( value ):
			for instr in self._cfg.incref( value.type, value ):
				self._emit( instr )
			# a non-RC value (e.g. a scalar parameter, `return pad`) has no
			# refcount for the Incref above to have touched at all - wrapping
			# it would just be a pointless extra Temp/Assign (and a real
			# regression: callers that scan fn.instructions for the ORIGINAL
			# named Assign, e.g. a default-value splice check, no longer find
			# it as the sole match)
			if wrap_fresh and value.type is not None and value.type.is_rc():
				fresh = self._new_temp( value.type )
				self._emit( ir.Assign( dest = fresh, src = value ))
				self._cfg.fresh_temp( fresh, value.type )
				return fresh
		return value

	def _stmt_Pass( self, node: ast.Pass ) -> None:
		pass

	def _stmt_Global( self, node: ast.Global ) -> None:
		# a no-op: an unannotated Assign to a name that already exists
		# anywhere in the scope chain (local, enclosing, or global) always
		# reassigns that same one - find_name_or_none's scope chain already
		# falls through to the module scope on its own, so there's never a
		# separate shadowing local to opt out of. It only introduces a new
		# local when the name is unbound everywhere in the chain (see
		# _stmt_Assign's inference branch), which by definition has nothing
		# to shadow
		pass

	def _stmt_Delete( self, node: ast.Delete ) -> None:
		# del x / del a[i] - single target only, not del a.b or multiple
		# targets (`del a, b`)
		if len( node.targets ) != 1 or not isinstance( node.targets[0], ( ast.Name, ast.Subscript )):
			self.lowering.discovery.fail( f'del only supports a single local variable name or subscript: {ast.unparse(node)}', node )
		target = node.targets[0]
		if isinstance( target, ast.Subscript ):
			self._stmt_Delete_subscript( node, target )
			return
		# ends a local's lifetime early (see TODO.txt/RC MANAGEMENT.md: a
		# local created inside one arm of an if can be referenced only
		# within that arm unless it's del'd before the arm exits, matching
		# the other arm's "never created it either" state). Removing it
		# from fn.names is enough on its own to make a later reference
		# fail (find_name won't find it) - the actual Decref emission is
		# cfg.py's job, wired in alongside its other hooks
		fn = self._current_fn
		existing = fn.get_local_or_raise( target.id )
		if not isinstance( existing, Variable ):
			self.lowering.discovery.fail( f'{target.id!r} is not a local variable, cannot del it', node )
		try:
			instructions = self._cfg.deleted( existing, self._current_fn.qualname )
		except CompileError as e:
			self.lowering.discovery.fail( str( e ), node )
		for instr in instructions:
			self._emit( instr )
		del fn.names[target.id]

	def _stmt_Delete_subscript( self, node: ast.Delete, target: ast.Subscript ) -> None:
		# del a[i] - dispatches to __delitem__ like `a[i] = v` dispatches to
		# __setitem__ (see _stmt_Assign's Subscript branch); no raw-opcode
		# fallback since there's no DeleteItem IR opcode - every type
		# supporting del a[i] must declare __delitem__ (list/dict/set do)
		obj = self._lower_expr( target.value, None )
		delitem_fn = self.lowering._find_method( obj.type, '__delitem__' )
		if delitem_fn is None:
			self.lowering.discovery.fail( f'{obj.type} has no __delitem__, cannot del {ast.unparse(target)}', node )
			return
		self.lowering._ensure_resolved( delitem_fn )
		self.lowering.schedule( delitem_fn.return_type )
		index = self._lower_expr( target.slice, delitem_fn.parameters[0].type )
		if delitem_fn.return_type is self.lowering.discovery.get_none_type():
			self._emit( ir.Call( dest = None, target = delitem_fn, receiver = obj, args = [ index ], kwargs = {} ))
		else:
			call_dest = self._new_temp( delitem_fn.return_type )
			self._emit( ir.Call( dest = call_dest, target = delitem_fn, receiver = obj, args = [ index ], kwargs = {} ))
			self._maybe_consume_result( node, call_dest, self.lowering._DELITEM_ALTERNATIVES )

	def _stmt_ImportFrom( self, node: ast.ImportFrom ) -> None:
		# local (in-function) form of discovery.py's own visit_ImportFrom -
		# function bodies are deliberately never walked by discovery.py's
		# own visitor (see this class's docstring: "function bodies were
		# deliberately left unvisited in stage 1"), so an import written
		# inside a function body (lib/sys.py's memzero()/_alloc()/etc. -
		# one FFI declaration per @compiler.target(os=...) branch) only
		# ever reaches here, never discovery.py's version. Registered
		# directly into the current function's own scope (add_name, same
		# as a parameter) rather than resolved/scheduled eagerly - an
		# unused import costs nothing, same posture as _expr_Name's lazy
		# resolve-on-use
		parts: list[str] = []
		if node.level:
			# same package-relative counting as discovery.py's own
			# visit_ImportFrom - see the comment there. This site previously
			# sliced the qualname without discovery's compensation for a
			# folded module, so a relative import written inside a function
			# body in a package's __init__.py climbed one level too far
			package = self.lowering.discovery.module_stack[-1].package
			strip = node.level - 1
			parts.extend(( package.split( '.' )[:-strip] if strip else package.split( '.' )) if package else [] )
			if not parts:
				self.lowering.discovery.fail( f'unable to relative import from here: {ast.unparse(node)}', node )
		if node.module:
			parts.append( node.module )
		package = '.'.join( parts )
		try:
			mod = self.lowering.discovery.import_name( package )
		except FileNotFoundError as e:
			self.lowering.discovery.fail( str( e ), node )
		if not mod:
			self.lowering.discovery.fail( f'module {package!r} not found', node )
		for alias in node.names:
			item = mod.get_local_or_raise( alias.name )
			if item is None:
				self.lowering.discovery.fail( f'module {package} does not export {alias.name!r}', node )
			self._current_fn.add_name( alias.asname or alias.name, item )

	def _stmt_Import( self, node: ast.Import ) -> None:
		for alias in node.names:
			try:
				mod = self.lowering.discovery.import_name( alias.name )
			except FileNotFoundError as e:
				self.lowering.discovery.fail( str( e ), node )
			self._current_fn.add_name( alias.asname or alias.name, mod )

	def _cfg_assign( self, dest: Variable, src: ir.Operand, *, is_alias: bool, node: ast.AST, track_result: bool = True, borrow: bool = False ) -> None:
		# wrapper around cfg.assign() - now that it can raise CompileError
		# (see cfg.py's own unchecked-Result overwrite check), every one of
		# its 9 call sites needs the same discovery.fail() conversion
		# _stmt_If/loop_back_edge's own call sites already use, or the
		# raised-but-unrecorded error would just be silently swallowed by
		# the nearest enclosing per-statement `except CompileError:
		# continue` recovery boundary.
		#
		# Also owns emitting the trailing `ir.Assign(dest=dest, src=src)`
		# itself now (every one of the 9 call sites used to emit this
		# immediately after looping over this function's own returned
		# instructions, with nothing interposed - the exact same two lines
		# duplicated 9 times) - centralizing it here isn't just
		# deduplication: it's what lets this method reliably close
		# PLAN_THREAD_SAFE_SHARED_STATE.md's Part A critical section(s) too,
		# on BOTH sides: cfg.py's assign() opens ir.AcquireGlobalLock for a
		# write (`dest.is_global` branch, wrapping whatever _decref_
		# instructions produces) AND for a read (`is_alias` branch, wrapping
		# _incref_instructions) - in both cases the trailing Assign this
		# method emits is ITS OWN separate textual read/write of the global
		# in the generated C, not a reuse of whatever the Incref/Decref
		# already touched, so it has to stay inside the SAME critical
		# section (confirmed necessary the hard way: an earlier version
		# that closed the read-side lock inside cfg.assign() itself, before
		# this Assign, crashed under real concurrent stress - the Assign's
		# own unprotected re-read of the global could see a DIFFERENT
		# object than the one just retained). The matching
		# ir.ReleaseGlobalLock(s) belong here, right after this Assign,
		# since cfg.assign() itself never emits that Assign (see this
		# method's own history).
		try:
			instructions = self._cfg.assign( dest, src, is_alias = is_alias, track_result = track_result, borrow = borrow )
		except CompileError as e:
			self.lowering.discovery.fail( str( e ), node )
		for instr in instructions:
			self._emit( instr )
		self._emit( ir.Assign( dest = dest, src = src ))
		if dest.is_global and dest.reassigned_outside_init:
			# reassigned_outside_init only reads True here if THIS call just
			# went through cfg.assign()'s RC global-write branch (it early-
			# returns before ever reaching that branch for a non-RC dest.type,
			# so the flag stays False in that case, correctly skipping this)
			self._emit( ir.ReleaseGlobalLock( var = dest ))
		if is_alias and isinstance( src, Variable ) and src.is_global and cfg.rc_leaves( dest.type ):
			# the read-side counterpart - closes the critical section
			# cfg.py's assign() opened (its own `is_alias` branch, see that
			# comment) right after THIS Assign, which is its own separate
			# textual read of src in the generated C (`dest = src;`), not a
			# reuse of whatever the Incref inside cfg.assign() already
			# retained - see that branch's own comment for the real crash
			# this closes. Structurally unconditional otherwise here too
			# (mirroring the Acquire) - emitter_c.py decides at emission
			# time, once src.reassigned_outside_init's FINAL value is
			# known, whether this becomes a real release or a no-op.
			#
			# The cfg.rc_leaves(dest.type) check is NOT optional: cfg.assign()
			# itself early-returns `[]` before ever reaching its own
			# `is_alias` branch when dest.type has no RC leaves (a plain
			# scalar global, e.g. `x = G` for `G: i32`) - meaning no Acquire
			# was ever emitted for that call. Without this same check here,
			# a scalar global's read would get an orphaned Release with no
			# matching Acquire - confirmed as a real bug via
			# lowering_test.py's own test_reads_module_global, which
			# expects a plain `Assign`, nothing else, for exactly this case.
			self._emit( ir.ReleaseGlobalLock( var = src, exclusive = False ))

	def _stmt_AnnAssign( self, node: ast.AnnAssign ) -> None:
		if not isinstance( node.target, ast.Name ):
			self.lowering.discovery.fail( f'unsupported AnnAssign target: {ast.unparse(node)}', node )
		reject_reserved_c_identifier( self.lowering.discovery.fail, node.target.id, node )
		fn = self._current_fn
		# Volatile[T] resolves transparently to plain T (discovery.py's
		# visit_Subscript strips it) - detected here, separately, by peeking
		# at the raw annotation AST so this ONE call site (the only thing
		# that needs to know) can tag the resulting Variable's storage.
		is_volatile = ( isinstance( node.annotation, ast.Subscript ) and isinstance( node.annotation.value, ast.Name )
				and node.annotation.value.id == 'Volatile' )
		var_type = self.lowering.discovery.visit( node.annotation )
		if is_volatile and var_type.is_rc():
			self.lowering.discovery.fail( f'Volatile[...] does not support refcounted types: {ast.unparse(node)}', node )
		# var_type starts as whatever discovery.visit() returns - often a
		# bare, un-monomorphized Specialization - and STAYS that way for
		# var's own construction/_lower_expr's expected_type below. Fixed
		# up (see the resolution block after _lower_expr, below) only once
		# the RHS has actually been lowered, and only when node.value is
		# real - see that block's own comment for why both restrictions
		# are load-bearing, not incidental.
		self.lowering.schedule( var_type )
		# an explicit type annotation is a DECLARATION, not just an
		# assignment - a variable's type is only ever given once. Unlike a
		# bare `x = value` (_stmt_Assign, where the type is INFERRED from
		# whichever assignment reaches it first, so reassigning/re-
		# entering it via a different branch is ordinary, expected reuse
		# of the same binding), redeclaring an already-live name via a
		# SECOND `x: T = ...` is always rejected here - regardless of
		# whether T matches the original annotation, and regardless of
		# whether the two occurrences are in mutually exclusive branches
		# (e.g. one `x: T = ...` per arm of an if/elif/else chain - see
		# this fix's own commit message for why: a variable's type is
		# declared exactly once, no matter how many source-level branches
		# happen to reach it). `del x` first is the only way to
		# legitimately redeclare x's type. Confirmed via a real repro: this
		# used to construct a FRESH Variable unconditionally, with no
		# existing-binding check at all, so `x: i32 = 1; x: str = 'hi'`
		# (no del) silently succeeded, each occurrence getting its own
		# independent Variable object sharing a stem - undiagnosed at the
		# source level, and (before Variable.needs_uid_suffix existed)
		# producing outright invalid C at the emitter level too - see
		# del_reuse_and_emitter_naming_bug.
		existing = self._existing_local_or_none( node.target.id, node, 'cannot assign to it' )
		if existing is not None:
			self.lowering.discovery.fail(
				f'{node.target.id!r} already has a declared type - a variable can only be given an '
				f'explicit type annotation once. Use a plain `{node.target.id} = ...` (no annotation) to '
				f'reassign it, or `del {node.target.id}` first to redeclare it with a fresh type',
				node,
			)
		var = Variable(
			stem = node.target.id,
			qualname = f'{fn.qualname}.{node.target.id}',
			file = fn.file,
			line = node.lineno,
			type = var_type,
			is_volatile = is_volatile,
			needs_uid_suffix = self._mark_fresh_local_declared( node.target.id ),
		)
		fn.add_name( var.stem, var ) # scoped to the whole function body regardless of node.value (no block scoping - see cfg.py's own module docstring) - a bare declaration (node.value is None) deliberately does NOT mark it live in self._cfg (see below); a later real assignment does, via assign()'s own unconditional self._live.add()
		if node.value is not None:
			try:
				operand = self._lower_expr( node.value, var_type )
			except CompileError:
				# var is already registered (above) with a plausible
				# declared type but no real value/instructions behind it -
				# mark it broken so a later reference raises
				# RedundantCompilationError instead of using it as if it
				# were genuinely initialized (see Name.broken)
				var.broken = True
				raise
			# Only NOW, after the RHS is fully lowered, swap var.type for
			# its resolved (monomorphized, if a Specialization) form -
			# ensure_resolved()'s own contract: "the SINGLE place a
			# Specialization gets swapped for the real, substituted thing
			# it stands in for - every caller MUST use the returned value,
			# or they see the abstract, unsubstituted base instead"
			# (Specialization.names/.resolve are raw passthroughs to it).
			# var.type staying an unresolved Specialization here is a real
			# gap: cfg.py's rc_leaves() reads a Specialization's ABSTRACT
			# base.attributes (still bare TypeVars for a generic union like
			# Result[T,E]) and silently concludes an annotated local like
			# `x: Result[SomeRCClass,E]` has no RC leaves at all, skipping
			# its own incref/decref entirely - confirmed directly with
			# AddressSanitizer, not just reasoning.
			#
			# Both restrictions below are load-bearing, found by real
			# regressions, not just caution:
			#
			# 1. Resolving BEFORE lowering the RHS (i.e. swapping var_type
			# up front and reusing it for _lower_expr's own expected_type)
			# regressed a real, unrelated bug into existence: for a
			# generic RCClass's own construction (`b: Box[i32] = Box(1)`),
			# eagerly monomorphizing Box[i32] here - before Box(1)'s own
			# construction-call lowering has had a chance to monomorphize
			# Box.__init__[i32] itself the ordinary way - raced it, and
			# the copy built here cached a version of Box.__init__[i32]
			# whose own `self` parameter was left typed as the abstract
			# Box[T] instead of the concrete Box[i32], which then got
			# scheduled as a bogus extra "Box[Box.T]" compile unit
			# (confirmed with a real repro against compiler.rcclasses,
			# not just a hunch). Resolving only after the RHS's own
			# construction-call machinery has already run first sidesteps
			# it - the resolution here then just reads back whatever it
			# already correctly cached (monomorphized_function's own
			# spec.monomorphized memoization), never racing it.
			#
			# 2. Only when node.value is not None (this whole branch) -
			# skipped for a bare declaration (`c: Box[u32]`, assigned via
			# a later, ordinary Assign statement, not this one) since
			# there's no RHS lowering here to resolve after in the first
			# place, and deferring is always safe: whatever later
			# statement actually assigns/uses c triggers its own
			# resolution through the ordinary paths (e.g. _attr_lookup's
			# own _ensure_resolved call), same as it always has.
			if self.lowering._monomorphizer._is_concrete( var_type ):
				var.type = self.lowering._ensure_resolved( var_type )
			self._cfg_assign( var, operand, is_alias = self.lowering._is_aliasing_expr( node.value, operand ), node = node )

	def _lower_attr_target_obj( self, value_node: ast.expr ) -> tuple[ir.Operand, Callable[[ir.Operand],None]|None]:
		''' the object operand for an attribute assignment target
		(target.attr = ...), plus an optional writeback callback the
		caller must invoke (with the SAME operand, now mutated via
		SetAttr) once the ordinary SetAttr has been emitted.

		Ordinarily there's no writeback needed - the object expression's
		own lowered value (self, an already-Ptr[T] variable, ...) IS the
		lvalue SetAttr writes through (the dot-operator already makes a
		bare Ptr[T] receiver work correctly here - see _attr_lookup's own
		pointee-redirect, emitter_c.py's _member_access_operator). But
		`ptr[idx].attr = value` is different: ptr[idx] ALONE (via
		_expr_Subscript's raw-pointer GetItem fallback, the only shape a
		raw pointer's own subscript has - no real __getitem__ method to
		dispatch to) loads a COPY of the pointee into a fresh temp, and
		writing through that copy silently drops the write entirely -
		confirmed by a real repro, not just reasoning. C has no single
		"address of the idx'th pointee, then ->field = value" primitive
		this compiler emits directly (unlike a bare Ptr[T] receiver,
		which is already the pointer itself) - so this reads the WHOLE
		element via ir.GetItem, returns that temp as the object SetAttr
		mutates (reusing every existing RC-tracking/attr-assignment path
		unchanged), and writes the WHOLE element back via ir.SetItem
		afterward - the same read-modify-write shape `ptr[idx] = value`
		(whole-value replacement) already uses one level up. ptr/index
		are lowered exactly once here (not re-lowered inside
		_expr_Subscript AND again for the writeback) - relowering the
		AST a second time would double-evaluate them, a real correctness
		risk if either expression has side effects (matches the same
		concern _stmt_AugAssign's own Attribute/Subscript-target
		restriction is about). '''
		if isinstance( value_node, ast.Subscript ):
			ptr_obj = self._lower_expr( value_node.value, None )
			if (
				self.lowering._find_method( ptr_obj.type, '__getitem__' ) is None
				and self.lowering._type_resolver._is_ptr_specialization( ptr_obj.type )
			):
				index_type = self.lowering.discovery.get_intrinsics()['usize']
				index = self._lower_expr( value_node.slice, index_type )
				elem_type = ptr_obj.type.args[0]
				elem = self._new_temp( elem_type )
				self._emit( ir.GetItem( dest = elem, obj = ptr_obj, index = index ))
				def writeback( updated: ir.Operand ) -> None:
					self._emit( ir.SetItem( obj = ptr_obj, index = index, value = updated ))
				return elem, writeback
		return self._lower_expr( value_node, None ), None

	def _static_field_type_or_none( self, node: ast.expr ) -> Type|None:
		''' like _static_type_of_value_expr, but NEVER calls discovery.fail()
		- any lookup miss (unknown name, a method/property instead of a
		plain field, a scope with no .names, ...) just returns None instead
		of hard-erroring. Needed specifically for _fixed_array_index_target's
		speculative "is this a FixedArrayType field?" peek: unlike sizeof(x)'s
		fallback (always a genuine error if it misses), a MISS here is the
		expected, common case - e.g. `obj.some_list_field[i]` or
		`obj.some_property[i]` (a property returning something indexable)
		must fall through to the ordinary _expr_Attribute/_lower_attr_target_obj
		handling unchanged, not be mistaken for a real error. '''
		if isinstance( node, ast.Name ):
			name = self.lowering.discovery.find_name_or_none( node.id )
			if not isinstance( name, Variable ) or name.broken:
				return None
			member = self._cfg.narrowed_member( node.id )
			return member.type if member is not None else name.type
		if isinstance( node, ast.Attribute ):
			owner_type = self._static_field_type_or_none( node.value )
			if owner_type is None:
				return None
			owner_type = self.lowering._ensure_resolved( owner_type )
			if isinstance( owner_type, Specialization ) and owner_type.pointer_stem() is not None:
				owner_type = self.lowering._ensure_resolved( owner_type.args[0] )
			if isinstance( owner_type, ( CStruct, RCClass )):
				try:
					found = owner_type.chain_lookup( node.attr )
				except RedundantCompilationError:
					# a broken inherited member is exactly as much "not
					# statically known" as a genuine miss, for this
					# speculative-peek contract - never raise here
					return None
			else:
				names = getattr( owner_type, 'names', None )
				found = names.get( node.attr ) if isinstance( names, dict ) else None
			if not isinstance( found, Variable ) or found.broken:
				return None
			# resolve the VARIABLE itself first, same as _attr_lookup's own
			# `self._ensure_resolved( found ); return found` - found.type can
			# still be a lazily-deferred placeholder until found itself is
			# resolved (confirmed by a real repro: `g: Foo = make()` then
			# `g.b[i]`, Foo only reached indirectly through make()'s return
			# type rather than a direct Foo() construction in the same
			# statement, left found.type unresolved here and silently
			# misidentified a genuine FixedArrayType field as "not one")
			self.lowering._ensure_resolved( found )
			return found.type
		return None

	def _fixed_array_index_target( self, attr_node: 'ast.Attribute|ast.Name', index_node: ast.expr ) -> tuple[ir.Operand,str,FixedArrayType,ir.Operand]|None:
		''' `f.arr[i]` where `f.arr` (attr_node) statically resolves to a
		FixedArrayType field, OR `buf[i]` where `buf` (attr_node, a bare
		ast.Name) is ITSELF a local/parameter of FixedArrayType - returns
		(root_obj, attr_name, array_type, index_operand) ready for
		ir.GetAttrIndex/SetAttrIndex, or None if it doesn't (every other
		subscript shape is handled unchanged by the ordinary paths in
		_expr_Subscript/_stmt_Assign). For the bare-Name case, attr_name is
		always '' - emission (see emitter_c.py's GetAttrIndex/SetAttrIndex/
		ArrayFieldPtr/AddrOfArrayIndex handling) treats an empty attr as
		"root_obj IS the array itself", spelling a flat `root[index]`
		instead of `(root).attr[index]`. Checked via
		_static_field_type_or_none FIRST, before lowering attr_node.value
		for real - that helper emits no IR, never fails, and never evaluates
		attr_node, so a non-match here doesn't double-evaluate the root
		object (the same "lower it exactly once" concern
		_lower_attr_target_obj's own docstring explains) and doesn't
		misfire a spurious error for some other legitimate attribute shape
		(property, method-value, ...). This is also why deeper/non-Name
		roots (e.g. `make().arr[i]`) fall through unrecognized rather than
		being specially rejected here: the helper only walks Name/Attribute
		chains, so anything else just resolves to None and reaches the
		ordinary whole-value-read rejection below, same as before this
		feature existed.

		A literal constant index out of [0, count) is rejected at compile
		time, same as tuple's own compile-time-constant index check - the
		one bit of free bounds checking available here (FixedArrayType's
		count, unlike a raw Ptr[T], is always known at compile time). A
		non-constant (runtime) index is otherwise unchecked, matching
		Ptr[T]/ConstPtr[T]'s own GetItem convention - this is a raw inline
		C array field, not a general-purpose bounds-checked container (see
		FixedArrayType's own docstring). '''
		array_type = self._static_field_type_or_none( attr_node )
		if not isinstance( array_type, FixedArrayType ):
			return None
		if ( isinstance( index_node, ast.Constant ) and isinstance( index_node.value, int )
				and not isinstance( index_node.value, bool ) and not ( 0 <= index_node.value < array_type.count )):
			self.lowering.discovery.fail(
				f'index {index_node.value} out of range for {array_type.qualname} (0..{array_type.count-1}): {ast.unparse(index_node)}',
				index_node,
			)
		if isinstance( attr_node, ast.Name ):
			root = self._fixed_array_root_operand( attr_node )
			attr = ''
		else:
			root = self._lower_expr( attr_node.value, None )
			attr = attr_node.attr
		index_type = self.lowering.discovery.get_intrinsics()['usize']
		index = self._lower_expr( index_node, index_type )
		return root, attr, array_type, index

	def _fixed_array_root_operand( self, name_node: ast.Name ) -> ir.Operand:
		''' the raw Operand for a bare local/parameter Name already known
		(by the caller) to be of FixedArrayType - used only by
		_fixed_array_index_target's Name branch and
		_lower_compiler_addrof's bare-Name branch, to get the array's own
		storage as an addressable root. Deliberately bypasses _expr_Name's
		ordinary "read this as a whole VALUE" path (which rejects
		FixedArrayType outright, the same as the Attribute/field-read case
		- see FixedArrayType's own docstring: it has no whole-value read),
		since indexing/addrof-ing the array itself isn't a whole-value
		read at all. '''
		name = self.lowering.discovery.find_name( name_node.id, name_node )
		if not isinstance( name, Variable ):
			self.lowering.discovery.fail( f'{name_node.id!r} is not a value, cannot use it as an expression', name_node )
		if not name.is_global and not self._cfg.is_live( name_node.id ):
			self.lowering.discovery.fail( f'{name_node.id!r} is not initialized on all code branches', name_node )
		self.lowering._ensure_resolved( name )
		return name

	def _existing_local_or_none( self, target_id: str, node: ast.AST, context: str ) -> Variable|None:
		''' is target_id ALREADY a genuine binding this assignment should
		reuse - fn.names (the CURRENT function's own scope) first, falling
		through to the enclosing MODULE's own top-level names (never an
		enclosing CLASS scope, and never builtins/intrinsics) if not found
		there. Treats a previously-BROKEN entry as if it weren't there at
		all - free to redeclare cleanly via _declare_local below, same as a
		genuinely first assignment, since nothing usable was ever produced
		for it. Shared by _stmt_Assign, _stmt_AnnAssign, _expr_NamedExpr
		(walrus), and _bind_loop_target - all four mirror the same "reuse
		existing, else declare fresh" rule (see their own comments).
		`context` only feeds the not-a-variable error message, which
		differs slightly per caller.

		This exact two-scope search (not the full find_name_or_none walk,
		and not fn.names alone) is shaped by three real, confirmed cases:

		1. fn.names alone breaks `global x; x = value` (or `x: T = value`)
		reassigning a MODULE-level global from inside a function -
		_stmt_Global is deliberately a no-op (see its own comment) that
		relies ENTIRELY on this kind of lookup falling through to module
		scope; skipping that fallback makes the reassignment look like a
		brand-new LOCAL instead, silently shadowing the real global. A
		local genuinely colliding with an unrelated MODULE-level name -
		e.g. `head` inside http.client._build_request_head, this module's
		OWN head() verb helper - is correctly rejected the same way (fixed
		by renaming the local, not by widening this lookup further).

		2. The full find_name_or_none walk (module_context's scope_stack,
		innermost first) ALSO includes the enclosing CLASS scope for a
		method body, and builtins/intrinsics at the very end - so a
		genuinely fresh local shadowing an unrelated CLASS MEMBER (e.g.
		`byte_len: usize = ...` as a local inside str._from_owned_cstr,
		despite str ALSO having an instance method literally named
		byte_len) or an intrinsic TYPE name (a local literally named `u64`
		or `u32`, deliberately exercised by emitter_c_test.py's own
		UnionReceiverDispatchCoercionTests) would incorrectly find that
		outer name and reject the shadowing declaration as "not a
		variable, cannot assign to it" - confirmed regressions once
		_stmt_AnnAssign started calling this at all (previously it never
		checked for an existing binding, so neither case ever reached
		here). module.names is NOT builtins/intrinsics (Module keeps all
		three as separate dicts - see its own fields) so this search
		reaches real module-level globals without ever touching those. '''
		existing = self._current_fn.names.get( target_id )
		if existing is None:
			# module_stack[-1], not _find_module_for(self._current_fn) - the
			# latter linear-scans every module by file match, and this is a
			# hot path (called on every bare-Name assignment); module_
			# context() already keeps the CURRENTLY active module on top of
			# this same stack for the whole time a statement is being
			# lowered, the identical source find_name_or_none itself reads
			existing = self.lowering.discovery.module_stack[-1].names.get( target_id )
		if existing is None or existing.broken:
			return None
		if not isinstance( existing, Variable ):
			self.lowering.discovery.fail( f'{target_id!r} is not a variable, {context}', node )
		return existing

	def _existing_loop_carried_or_none( self, target_id: str ) -> Variable | None:
		''' `del x; x = value` inside a loop body looked like a fresh
		declaration to _existing_local_or_none (del removes target_id from
		fn.names), so _declare_local minted a brand-new, uid-suffixed C
		variable for it - fine outside a loop, but the loop body is only ever
		lowered once and its back-edge goto still targets the ORIGINAL C
		variable, so the next iteration's own `del x` released that original
		(already released, now-abandoned) variable again: a real double free.
		cfg.py's loop_back_edge() didn't catch it either - entry and back-edge
		bindings for `x` agree on OwnState (OWNED either way), and it never
		compares Variable IDENTITY, only state.
		Only consulted when the ordinary fn.names/module lookup already
		missed: if target_id was ALSO bound at the innermost active loop's own
		entry snapshot, resurrect and reuse that SAME Variable (same C
		identifier) so the reassignment goes through the ordinary reuse path
		below instead of _declare_local. Re-registers it into fn.names too -
		unlike _declare_local, a plain reassign doesn't do that itself, and
		later code (same iteration or after the loop) needs to find it again.
		Only the innermost active loop is checked - safe regardless of
		nesting depth: a loop's own entry snapshot already inherits every
		binding live from its enclosing scopes, so del+reassign sharing any
		one (possibly nested) loop body always finds `target_id` via
		whichever loop directly encloses them. Splitting del and the
		reassignment across two different loop nesting levels (e.g. del
		inside an inner loop with the reassignment only after it exits)
		never reaches this fallback at all - loop_back_edge()'s own entry/
		back-edge stability check already rejects that shape as a hard
		compile error first (confirmed: "does not exist consistently across
		loop iterations"). '''
		if not self._loop_labels:
			return None
		entry_binding = self._loop_labels[-1].loop_snapshot.bindings.get( target_id )
		if entry_binding is None:
			return None
		var = entry_binding.operand
		self._current_fn.add_name( target_id, var )
		return var

	def _declare_local( self, target_id: str, node: ast.AST, lower_rhs: Callable[[Type|None],ir.Operand], *, default_type: Type|None = None ) -> tuple[Variable,ir.Operand]:
		''' the RHS is lowered BEFORE target_id is registered - not the
		other way around - because it may reference target_id itself: a
		bare Name target's own read/write desugaring (_stmt_AugAssign's
		`x += 1` -> `x = x + 1`, threaded straight through _stmt_Assign)
		relies on a genuinely-undeclared x's OWN read, inside that
		synthesized RHS, still failing "not defined" normally - registering
		x first would make that read find a freshly-declared, empty x
		instead. Only on FAILURE is a (broken) placeholder registered here,
		in the except clause - the actual fix for the gap that otherwise let
		a later, separate reference to target_id report a spurious "not
		defined" cascade on top of the real, original error (see
		Name.broken). Shared by the "no prior declaration" branches of
		_stmt_Assign, _expr_NamedExpr (walrus), and _bind_loop_target.
		`default_type` is only meaningful for _bind_loop_target's own
		fallback when value_expr has no type of its own to infer from (e.g.
		range()'s implicit literal start) - every other caller leaves it
		None, inferring purely from the RHS. '''
		reject_reserved_c_identifier( self.lowering.discovery.fail, target_id, node )
		fn = self._current_fn
		needs_uid_suffix = self._mark_fresh_local_declared( target_id )
		try:
			operand = lower_rhs( default_type )
		except CompileError:
			broken = Variable( stem = target_id, qualname = f'{fn.qualname}.{target_id}', file = fn.file, line = getattr( node, 'lineno', None ), type = None, broken = True, needs_uid_suffix = needs_uid_suffix )
			fn.add_name( broken.stem, broken )
			raise
		# NOTE: deliberately does NOT auto-.or_throw() an unannotated fresh
		# local just because its RHS happens to be Result-shaped - `r =
		# some_fallible_call()` (no annotation) has always captured the raw
		# Result, precisely so it can be inspected via .is_ok()/match right
		# after (confirmed by real fixtures: test_fallible_init_shape_has_
		# ok_err_branches, match's own __match_subj_N desugaring reaching
		# this SAME path). Case 1/2 of the general auto-or_throw() rule
		# (_coerce_or_check_operand, _stmt_Return, discard sites - see
		# _auto_or_throw's own docstring) still fire the moment `var` is
		# later used somewhere that actually wants its own T rather than
		# the whole Result - an unannotated arithmetic result that's never
		# inspected as a Result and only ever flows into a T-typed context
		# still ends up auto-consumed there, just one statement later than
		# it used to be eagerly consumed at its own declaration
		var = Variable( stem = target_id, qualname = f'{fn.qualname}.{target_id}', file = fn.file, line = getattr( node, 'lineno', None ), type = operand.type, needs_uid_suffix = needs_uid_suffix )
		fn.add_name( var.stem, var )
		self.lowering.schedule( var.type )
		return var, operand

	def _resolve_narrow_member( self, name: str, member_stem: str, node: ast.AST ) -> Variable:
		# type_resolver.py hands down only a STEM (see its own comment on
		# why - resolved against the TEXTUAL/abstract union at that pass,
		# T/E may still be bare TypeVars there) - re-resolve the real,
		# substituted member against `name`'s own already-monomorphized
		# type here, the same pattern _coerce_into_union already uses. A
		# parameter's own declared type (unlike a local var initialized
		# from a call's already-eagerly-monomorphized return type) stays a
		# genuine Specialization wrapping the ABSTRACT base - .base alone
		# isn't enough, has to go through monomorphize_class same as any
		# other generic-class use site, or the member's own .type resolves
		# to the unsubstituted TypeVar instead of the real leaf (str, not T).
		# Shared by _stmt_Assign's own is_narrowing_bind handling (match/if-
		# desugared narrowing) and _stmt_While's own exit-narrowing (Phase 7).
		subject_var = self.lowering.discovery.find_name( name, node )
		assert isinstance( subject_var, Variable )
		return self._resolve_narrow_member_of_type( subject_var.type, member_stem )

	def _resolve_narrow_member_of_type( self, subject_type: Type, member_stem: str ) -> Variable:
		# shared tail of _resolve_narrow_member (a local/param, looked up by
		# name) and _resolve_narrow_attr_member (a field, looked up via
		# _attr_lookup) - both need the same monomorphize-then-find-by-stem
		# once they have the subject's own type in hand.
		base = self.lowering.monomorphize_class( subject_type ) if isinstance( subject_type, Specialization ) else subject_type
		self.lowering._union_storage.get( base )
		member = next( ( attr for attr in base.attributes if attr.stem == member_stem ), None )
		assert member is not None
		return member

	def _attribute_chain_key( self, node: ast.expr ) -> str|None:
		# lowering.py's counterpart of type_resolver.py's identically-named
		# helper - given ANY Name/Attribute node, returns the same
		# '::'-joined synthetic key a narrowing construct would have used
		# to narrow it, or None if node isn't a plain Name-rooted
		# attribute chain. No per-hop field validation needed (unlike
		# type_resolver.py's _narrow_subject_key, which is building a NEW
		# marker) - a bogus key for a shape that was never actually
		# narrowable just never matches anything in cfg.py's own dict.
		if isinstance( node, ast.Name ):
			return node.id
		if isinstance( node, ast.Attribute ):
			base_key = self._attribute_chain_key( node.value )
			if base_key is not None:
				return f'{base_key}::{node.attr}'
		return None

	def _resolve_chain_owner_type( self, base_key: str, base_type: Type, attr_hops: list[str], node: ast.AST ) -> Type:
		# walks attr_hops one at a time off base_type, returning the type
		# reached after the LAST hop - shared by _resolve_narrow_attr_member
		# (walking to the container just before the field being narrowed)
		# and the read-side narrowed-chain resolution (_expr_Attribute/
		# _static_type_of_value_expr), which need the exact same walk to
		# resolve an attribute chain's OWNER type at any depth. At each
		# step, checks whether the ACCUMULATED prefix so far is itself
		# already narrowed (cfg.narrowed_member()) before falling through
		# to an ordinary _attr_lookup - this is what makes nested
		# narrowing compose (`if self.a is not None: if self.a.b is not
		# None: ...` resolves `.b` against self.a's PROVEN type, not its
		# raw declared union). If an intermediate hop is itself T|None but
		# not independently proven non-None, _attr_lookup on it fails (a
		# union has no plain named fields) - correct: you can't narrow
		# self.a.b.c if self.a might be None without its own proof.
		current_type = base_type
		accumulated_key = base_key
		for hop in attr_hops:
			accumulated_key = f'{accumulated_key}::{hop}'
			narrowed = self._cfg.narrowed_member( accumulated_key )
			current_type = narrowed.type if narrowed is not None else self.lowering._attr_lookup( current_type, hop, node ).type
		return current_type

	def _resolve_narrow_attr_member( self, attr_base: str, attr_hops: list[str], member_stem: str, node: ast.AST ) -> Variable:
		# attribute counterpart of _resolve_narrow_member - type_resolver.py's
		# visit_If (field-chain narrowing shape, `self.field`/`self.a.b.c`)
		# hands down the base local's name and the field CHAIN's own hop
		# names separately, since a field has no Variable of its own
		# reachable via find_name. Re-resolves the chain's REAL
		# (monomorphized, and narrowed-aware at every intermediate hop)
		# type off the base local first, the same reason
		# _resolve_narrow_member itself can't just trust type_resolver.py's
		# own (possibly still-abstract) textual type.
		base_var = self.lowering.discovery.find_name( attr_base, node )
		assert isinstance( base_var, Variable )
		self.lowering._ensure_resolved( base_var )
		base_type = self.lowering.monomorphize_class( base_var.type ) if isinstance( base_var.type, Specialization ) else base_var.type
		container_type = self._resolve_chain_owner_type( attr_base, base_type, attr_hops[:-1], node )
		field_var = self.lowering._attr_lookup( container_type, attr_hops[-1], node )
		return self._resolve_narrow_member_of_type( field_var.type, member_stem )

	def _stmt_Assign( self, node: ast.Assign ) -> None:
		if getattr( node, 'is_narrowing_bind', False ):
			# type_resolver.py's _match_pattern: `match x: case T(x):`
			# reusing the subject's own name - x's real Variable/storage is
			# untouched, this is a pure compile-time fact ("reads of x from
			# here until this scope's own restore() may read through the
			# union's own payload instead") - no IR at all, see cfg.py's
			# narrow()/_expr_Name's own comment for the read-side rewrite.
			target_name = node.targets[0]
			assert isinstance( target_name, ast.Name )
			attr_base = getattr( node, 'narrow_attr_base', None )
			stems: list[str] = node.narrows_member_stems
			if attr_base is not None:
				# field-chain narrowing (type_resolver.py's visit_If) -
				# target_name.id is a synthetic '::'-joined key, never a
				# real local; _expr_Attribute consults the identical key.
				members = [ self._resolve_narrow_attr_member( attr_base, node.narrow_attr_hops, stem, node ) for stem in stems ]
			else:
				members = [ self._resolve_narrow_member( target_name.id, stem, node ) for stem in stems ]
			if len( members ) == 1:
				self._cfg.narrow( target_name.id, members[0] )
			else:
				# a 3+-member union's `is None` guard (type_resolver.py's
				# visit_If) - the surviving path is proven to be ONE OF
				# these members, not a single leaf. See cfg.narrow_many's
				# own docstring.
				self._cfg.narrow_many( target_name.id, members )
			return
		if len( node.targets ) != 1:
			self.lowering.discovery.fail( f'multiple assignment targets not supported: {ast.unparse(node)}', node )
		target = node.targets[0]
		if isinstance( target, ast.Name ):
			existing = self._existing_local_or_none( target.id, node, 'cannot assign to it' )
			if existing is None:
				existing = self._existing_loop_carried_or_none( target.id )
			match_stmt_id = getattr( node, 'match_stmt_id', None )
			if (
				existing is not None and getattr( node, 'is_match_binding', False )
				and match_stmt_id is not None and self._match_binding_origin.get( target.id ) == match_stmt_id
			):
				# a SIBLING arm of the SAME match statement (type_resolver.py's
				# visit_Match tags every arm's own binding Assign with the same
				# id(the ast.Match)) rebinding this exact name - unlike an
				# ordinary reassignment or a genuinely separate later match
				# statement reusing the name (both still go through the
				# reuse-and-coerce path below, unchanged), sibling arms are
				# mutually exclusive by construction (exactly one ever runs)
				# and must NOT share one C-level slot - forcing that here was
				# the actual bug (a real compile error whenever the two arms'
				# payload types differed, even though they can never be
				# simultaneously live). Falling through to the "no prior
				# declaration" branch below mints this arm its OWN fresh
				# Variable; _mark_fresh_local_declared already gives it a
				# uid-suffixed, disambiguated C identifier (the same mechanism
				# a del-then-redeclare already relies on), so the two arms'
				# bindings never collide at the C level either.
				existing = None
			if existing is not None:
				# a module-level global's own Variable may not have had its
				# OWN .resolve run yet (its .type is None until then) if this
				# reassignment is the FIRST thing to touch it - see AugAssign's
				# identical _ensure_resolved(existing) call above for the full
				# reasoning
				self.lowering._ensure_resolved( existing )
				self._cfg.unnarrow( target.id ) # a real reassignment invalidates whatever this name was previously narrowed to - see cfg.py's own comment
				# every local (including a `case T(name):` match-arm binding -
				# see is_match_binding below) is function-scoped, no per-arm/
				# per-match scoping at all - reusing a binding name across two
				# SEPARATE, unrelated match statements is exactly as ordinary
				# as reusing a loop counter across two separate loops, and
				# works fine here whenever both sides happen to agree on the
				# payload type (an everyday reassignment, same as `x = 5`
				# then `x = 6`). Only a genuine type MISMATCH on such a reuse
				# needs a real diagnostic - and match_binding's own gets a
				# context naming the REAL cause instead of the bare "expected
				# X, got Y" ordinary reassignment already produces on its
				# own, which never explains where X even came from (nothing
				# in THIS statement's own source mentions it - it's a
				# leftover from wherever `target.id` was first bound).
				is_match_binding = getattr( node, 'is_match_binding', False )
				context = (
					f'{target.id!r} is already declared earlier in this function (e.g. by another match '
					f"arm's own binding) with an incompatible type"
				) if is_match_binding else None
				operand = self._lower_expr( node.value, existing.type, context = context )
				self._cfg_assign( existing, operand, is_alias = self.lowering._is_aliasing_expr( node.value, operand ), node = node )
			else:
				# first assignment to a name with no prior declaration - same
				# as an AnnAssign, but the type is inferred from the RHS
				# instead of coming from an explicit annotation
				var, operand = self._declare_local( target.id, node, lambda expected: self._lower_expr( node.value, expected ))
				# type_resolver.py's visit_Match desugars `match r:` into
				# `__match_subj_N = r; if ...` and marks the synthesized
				# Assign with these two attributes (see its own comment) -
				# is_match_subject means the fresh __match_subj_N temp must
				# never itself become a tracked obligation (the if-chain
				# below only does raw tag Compares, never is_ok()/is_err(),
				# so nothing would ever clear it); match_clears_name carries
				# the ORIGINAL name through when the subject was a bare Name
				# - ordinary aliasing assignment deliberately does NOT clear
				# the source (see cfg.py's "Independent tracking"), but a
				# match statement genuinely IS the inspection of its subject
				is_match_subject = getattr( node, 'is_match_subject', False )
				is_alias = self.lowering._is_aliasing_expr( node.value, operand )
				# when the subject is a bare Name (is_alias=True), the
				# ORIGINAL name already owns a live reference for the whole
				# (function-scoped) rest of its lifetime, so __match_subj_N
				# only needs a BORROW, not its own Incref/epilogue-Decref
				# pair - see cfg.py's assign() borrow= doc for the bug this
				# fixes (a real, always-unbalanced-until-function-exit
				# Incref that inflated every compiler.refcount() read taken
				# inside a match arm). A non-Name subject (e.g. `match
				# make():`) has no such original owner, so it keeps full
				# ownership tracking unchanged (borrow=False there).
				self._cfg_assign( var, operand, is_alias = is_alias, node = node, track_result = not is_match_subject, borrow = is_match_subject and is_alias )
				if getattr( node, 'is_match_binding', False ):
					if match_stmt_id is not None:
						# so a LATER sibling arm of this same match statement
						# reusing this name (the check just above, at this
						# branch's own top) recognizes it as a same-match
						# rebind needing its own fresh storage too, rather than
						# an ordinary reuse to coerce-or-reject
						self._match_binding_origin[ target.id ] = match_stmt_id
					# type_resolver.py's _match_pattern: a `case T(name):`
					# extracted payload. This language has no wildcard/discard
					# binding syntax (no Rust-style `case T(_):`), so a case
					# body that never reads `name` (`case Result.Err(e): pass`,
					# or one that builds a fresh, unrelated error instead of
					# reusing e) is entirely ordinary, expected code, not an
					# oversight - unlike a bare user-declared local's own
					# genuinely-forgotten unused value, there's no syntax the
					# user could have written instead to signal "discard this"
					# and silence a real warning. Mark it read unconditionally
					# (harmless when the arm DOES go on to use it - a real
					# later read isn't affected either way) rather than
					# leaving every such arm's own real, confirmed
					# -Wunused-variable/C4189 unfixable from the language side
					self._emit( ir.MarkUsed( operand = var ))
				match_clears_name = getattr( node, 'match_clears_name', None )
				if match_clears_name is not None:
					self._cfg.clear_result( match_clears_name )
		elif isinstance( target, ast.Attribute ):
			target_key = self._attribute_chain_key( target )
			if target_key is not None:
				# a real reassignment invalidates whatever this exact field
				# CHAIN (any depth, `self.a.b.c`) was previously narrowed to
				# - same reasoning the plain-Name branch's own unnarrow()
				# call has above, generalized to the '::'-joined key _expr_
				# Attribute/_stmt_Assign's is_narrowing_bind branch use.
				# cfg.py's own unnarrow() also purges any LONGER key sharing
				# this one as a prefix, so reassigning a SHORT prefix here
				# (`self.a = ...`) still correctly drops a narrow proven
				# about `self.a.b` too, not just an exact-key match.
				self._cfg.unnarrow( target_key )
			obj, writeback = self._lower_attr_target_obj( target.value )
			attr_var = self.lowering._attr_lookup( obj.type, target.attr, target )
			self._check_field_visibility( obj.type, attr_var, target.attr, target )
			if isinstance( attr_var.type, FixedArrayType ):
				# same restriction as the GetAttr (read) side - a bare C array
				# member is never assignable via `=` (only a whole containing
				# struct/union is, via the compound-literal construction path
				# _lower_allocate_fields already handles) - see
				# FixedArrayType's own docstring
				self.lowering.discovery.fail(
					f'{ast.unparse(target)}: {attr_var.type.qualname} fields cannot be assigned after construction '
					f'(no element-level array access is implemented)',
					target,
				)
			operand = self._lower_expr( node.value, attr_var.type )
			needs_field_lock = False
			is_construction_write = self._construction_self is not None and obj is self._construction_self
			if not is_construction_write:
				# PLAN_THREAD_SAFE_SHARED_STATE.md Cost mitigation #2 - mirrors
				# cfg.py's assign() flipping reassigned_outside_init for a
				# global the moment it's written from outside its own
				# initializer. `obj is self._construction_self` is the ONLY
				# "this write is __init__ constructing itself" shape (a write
				# to some OTHER already-published object, even from within a
				# different object's own __init__, is exactly the race this
				# flag exists to catch) - so anything else flips it, unconditionally.
				attr_var.field_reassigned_outside_init = True
			if is_construction_write:
				# self.<attr> = value, inside __init__ construction itself -
				# tracked for definite-assignment/self-escape purposes (see
				# RCCLASS ATTRIBUTE LIFETIME.md and cfg.attr_assign())
				for instr in self._cfg.attr_assign( attr_var, operand, is_alias = self.lowering._is_aliasing_expr( node.value, operand )):
					self._emit( instr )
			elif getattr( node, 'generator_first_rc_assign', False ):
				# PLAN_GENERATORS.md Phase 5 (roadmap Phase 5) - a generator's
				# own $$__next__ reassigning an RC-typed promoted local for
				# the FIRST time ever (type_resolver.py's _rename_and_track_
				# liveness splits every such reassignment into `if self.__
				# <stem>_live: <ordinary decref-old assign via attr_replace,
				# below> else: <THIS tagged assign>; self.__<stem>_live =
				# True`). The field's CURRENT value is still the construction-
				# time placeholder (a bare `0`, see _expr_Constant's own
				# generator_zero_rc_field exemption elsewhere in this file) -
				# reading it back and decref-ing it the ordinary way below
				# would compute &(NULL)->$header, a real, confirmed UBSan trap
				# (member access through a null pointer is UB even when the
				# member is at offset 0, and release_object's own runtime
				# null-check would otherwise make it harmless anyway) - so
				# there is no old value read/decref here at all, unlike both
				# branches above/below. Deliberately NOT routed through
				# attr_assign (unlike the construction-time branch just above)
				# even though the underlying need - "a fresh value, no prior
				# one to decref" - is the same: attr_assign also pushes a
				# 'self.<attr>'-keyed entry onto self.bindings/the epilogue
				# stack, a mechanism scoped to (and only ever reconciled
				# correctly by merge_if/cfg.py for) an actual __init__ under
				# construction - reusing it here, inside the ordinary if/else
				# type_resolver.py synthesizes around this branch, made
				# merge_if() see a 'self.<attr>' binding fresh on only one
				# branch and try to `del self._current_fn.names['self.<attr>']`
				# - a real KeyError, confirmed via a real repro, since no local
				# named 'self.<attr>' is ever registered in fn.names (that key
				# format is attr_assign's own bindings-dict convention, not a
				# real name lookup key). Only the two ordinary halves of what
				# a fresh RC value assignment needs are reproduced directly
				# instead, via the same public helpers _stmt_Return already
				# uses for an analogous "move ownership in, no bindings
				# tracking" need: incref the new value if it's an alias of an
				# existing tracked binding (mirrors attr_replace/attr_assign's
				# own is_alias branch), else untrack_temp() so the fresh temp's
				# own end-of-statement cleanup doesn't ALSO decref it now that
				# the field owns it.
				if self.lowering._is_aliasing_expr( node.value, operand ):
					for instr in self._cfg.incref( attr_var.type, operand ):
						self._emit( instr )
				else:
					self._cfg.untrack_temp( operand )
			elif cfg.rc_leaves( attr_var.type ):
				# ordinary SetAttr on an already-constructed instance -
				# "an RCClass is always complete, so setting an attribute
				# is always a replace" (RCCLASS ATTRIBUTE LIFETIME.md).
				# cfg.py doesn't track arbitrary struct instances' field
				# CONTENTS across statements (v1 scope cut - see cfg.py's
				# module docstring), so the current value is always read
				# fresh here rather than consulted from any tracked state.
				# PLAN_THREAD_SAFE_SHARED_STATE.md Part B: this whole
				# sequence (read-old, decref-old/incref-new, the SetAttr
				# below) is ONE critical section, the field-locking
				# counterpart of Part A's identical "the Assign has to stay
				# INSIDE the same critical section as the Acquire/decref"
				# rule - a concurrent reader/writer on the SAME object's
				# SAME field must never observe a state between the decref
				# of the old value and the store of the new one.
				needs_field_lock = self._is_real_field_receiver( obj.type )
				if needs_field_lock:
					self._emit( ir.AcquireFieldLock( obj = obj, field = attr_var ))
				old = self._new_temp( attr_var.type )
				self._emit( ir.GetAttr( dest = old, obj = obj, attr = target.attr ))
				for instr in self._cfg.attr_replace( attr_var.type, old, operand, is_alias = self.lowering._is_aliasing_expr( node.value, operand )):
					self._emit( instr )
			self._emit( ir.SetAttr( obj = obj, attr = target.attr, value = operand ))
			if needs_field_lock:
				self._emit( ir.ReleaseFieldLock( obj = obj, field = attr_var ))
			if writeback is not None:
				writeback( obj )
		elif isinstance( target, ast.Subscript ):
			if isinstance( target.value, ( ast.Attribute, ast.Name )):
				fixed = self._fixed_array_index_target( target.value, target.slice )
				if fixed is not None:
					root, attr, array_type, index = fixed
					operand = self._lower_expr( node.value, array_type.elem_type )
					self._emit( ir.SetAttrIndex( obj = root, attr = attr, index = index, value = operand ))
					return
			obj = self._lower_expr( target.value, None )
			setitem_fn = self._find_indexlike_setitem( obj.type, target.slice )
			if setitem_fn is None:
				# no real __setitem__ declared (raw pointers, or any other
				# type that doesn't define subscript assignment as a method)
				# - falls back to the flat SetItem opcode, unconditionally.
				# Both the index and the value need a real expected_type here,
				# same as _expr_Subscript's own raw-pointer GetItem fallback
				# already gives its index (this comment used to just claim to
				# mirror that, without actually doing so) - without it, a bare
				# literal on either side (`buf[0] = x`/`buf[i] = 0`) falls
				# through to whatever the bare-literal default happens to be,
				# instead of the pointer's own real index/element type.
				index_type = self.lowering.discovery.get_intrinsics()['usize']
				elem_type = None
				if isinstance( obj.type, Specialization ) and obj.type.pointer_stem() is not None:
					elem_type = obj.type.args[0]
				index = self._lower_expr( target.slice, index_type )
				operand = self._lower_expr( node.value, elem_type )
				self._emit( ir.SetItem( obj = obj, index = index, value = operand ))
			else:
				# a real __setitem__ - call it like any other method, then if
				# it returns Result[T,E], auto-consume it exactly like
				# _expr_Subscript's own __getitem__ call does: `obj[i] = v`
				# reads as sugar for `obj.__setitem__(i, v).or_return()`
				# whenever __setitem__ can fail
				self.lowering._ensure_resolved( setitem_fn )
				self.lowering.schedule( setitem_fn.return_type )
				index = self._lower_expr( target.slice, setitem_fn.parameters[0].type )
				operand = self._lower_expr( node.value, setitem_fn.parameters[1].type )
				if setitem_fn.return_type is self.lowering.discovery.get_none_type():
					# the ordinary/conventional case (matches Python's own
					# __setitem__ protocol, which always returns None) -
					# a real Temp dest here would try to assign C's void
					# return to a variable, which doesn't compile; no
					# Result to auto-consume either
					self._emit( ir.Call( dest = None, target = setitem_fn, receiver = obj, args = [ index, operand ], kwargs = {} ))
				else:
					call_dest = self._new_temp( setitem_fn.return_type )
					self._emit( ir.Call( dest = call_dest, target = setitem_fn, receiver = obj, args = [ index, operand ], kwargs = {} ))
					self._maybe_consume_result( node, call_dest, self.lowering._SUBSCRIPT_ALTERNATIVES )
		elif isinstance( target, ast.Tuple ):
			# `(a, b) = t` / `a, b = t` - both spellings parse to the same
			# ast.Assign(targets=[ast.Tuple(...)]) shape. node.value is
			# lowered exactly ONCE (not per-element) since it may be
			# side-effecting (sock.accept().or_return()) - each element is
			# then a raw GetAttr off the tuple's own _0/_1/... fields,
			# genuinely aliasing the tuple's own storage, the identical
			# shape _expr_Subscript's tuple-constant-index read already
			# established (and already fixed a real use-after-free for -
			# see its own is_tuple_element_read comment) - is_alias=True
			# unconditionally for a fresh declaration, re-derived from the
			# coercion result for a reassignment (a union-widening coerce
			# already increfs the leaf it wraps internally; treating that
			# as still-aliasing would double-incref).
			if any( isinstance( elt, ast.Starred ) for elt in target.elts ):
				self.lowering.discovery.fail( f'starred unpacking targets are not supported: {ast.unparse(node)}', node )
			if not all( isinstance( elt, ast.Name ) for elt in target.elts ):
				self.lowering.discovery.fail( f'unpacking targets must be plain names (nested tuple targets are not supported): {ast.unparse(node)}', node )
			fn = self._current_fn
			try:
				value = self._lower_expr( node.value, None )
				resolved_value_type = self.lowering._ensure_resolved( value.type )
				tuple_type = self.lowering._tuple_storage.tuple_type_for( resolved_value_type )
				if tuple_type is None:
					self.lowering.discovery.fail(
						f'cannot unpack a non-tuple value (got {resolved_value_type.qualname if resolved_value_type else "?"}): '
						f'{ast.unparse(node)}',
						node,
					)
				if len( tuple_type.elem_types ) != len( target.elts ):
					self.lowering.discovery.fail(
						f'unpacking target has {len(target.elts)} name(s), value has {len(tuple_type.elem_types)}: {ast.unparse(node)}',
						node,
					)
				for i, elt in enumerate( target.elts ):
					assert isinstance( elt, ast.Name )
					attr_var = self.lowering._attr_lookup( resolved_value_type, f'_{i}', node )
					elem = self._new_temp( attr_var.type )
					self._emit( ir.GetAttr( dest = elem, obj = value, attr = f'_{i}' ))
					existing = self._existing_local_or_none( elt.id, node, 'cannot assign to it' )
					if existing is not None:
						self.lowering._ensure_resolved( existing ) # see _stmt_Assign's identical call for why
						self._cfg.unnarrow( elt.id )
						final = self._coerce_or_check_operand( elem, existing.type, node )
						is_alias = not getattr( final, 'is_union_coerce_result', False )
						self._cfg_assign( existing, final, is_alias = is_alias, node = node )
					else:
						var = Variable( stem = elt.id, qualname = f'{fn.qualname}.{elt.id}', file = fn.file, line = getattr( node, 'lineno', None ), type = elem.type, needs_uid_suffix = self._mark_fresh_local_declared( elt.id ))
						fn.add_name( var.stem, var )
						self.lowering.schedule( var.type )
						self._cfg_assign( var, elem, is_alias = True, node = node )
			except CompileError:
				for elt in target.elts:
					if isinstance( elt, ast.Name ) and self.lowering.discovery.find_name_or_none( elt.id ) is None:
						broken = Variable( stem = elt.id, qualname = f'{fn.qualname}.{elt.id}', file = fn.file, line = getattr( node, 'lineno', None ), type = None, broken = True )
						fn.add_name( broken.stem, broken )
				raise
		else:
			self.lowering.discovery.fail( f'unsupported Assign target: {ast.unparse(node)}', node )

	def _stmt_AugAssign( self, node: ast.AugAssign ) -> None:
		# x += y desugars to x = x + y (reusing whatever arithmetic mode is
		# active, exactly like a hand-written x = x + y would). A bare Name
		# target reads/writes via the synthesized BinOp+Assign below - safe
		# because a Name lookup has no side effects of its own. Attribute/
		# Subscript targets can't use that same trick (their object/index
		# expression would be evaluated twice - once to read, once to
		# resolve the write - a real correctness risk: `get_obj().x += 1`
		# must only call get_obj() once), so those two branches lower the
		# target's object/index exactly once themselves, then read/compute/
		# write through the SAME already-lowered operand(s) - mirroring
		# _lower_attr_target_obj's own reasoning and _stmt_Assign's own
		# Attribute/Subscript branches, just fused with a read first.
		if isinstance( node.target, ast.Name ):
			existing = self._existing_local_or_none( node.target.id, node.target, 'cannot use it as an AugAssign target' )
			# try an __iadd__-family in-place operator FIRST, same reasoning
			# as the Attribute/Subscript branches below - but only ever
			# worth even looking for an RC-class target (is_rc_pointer());
			# lower the target's identity/type before the rvalue (mirroring
			# how Attribute/Subscript already resolve their own `old` before
			# lowering `right`) so `right`'s own type inference has the
			# target's type to hint against. _ensure_resolved(existing), NOT
			# existing.type - a genuine module-level global's own Variable
			# may not have had its OWN .resolve run yet at this point (its
			# .type is None until then, see Variable's own doc), unlike a
			# GetAttr-produced field/an already-lowered local, which are
			# always already resolved
			self.lowering._ensure_resolved( existing )
			if existing is not None and existing.type is not None and self.lowering._ensure_resolved( existing.type ).is_rc_pointer():
				# not existing.type itself - an RCClass target is never a Ptr
				# specialization, and __iadd__'s own parameter type is
				# independent of the receiver's type (e.g. Counter.__iadd__(
				# self, v: i32)), so hinting a bare literal toward
				# existing.type (an RCClass) would wrongly REJECT it outright
				# (_expr_Constant's own literal-vs-expected_type check fires
				# regardless of strict). Instead, peek at __iadd__'s own
				# declared parameter type (when unambiguous) and hint the
				# literal with THAT - without it, a bare literal locks in as
				# builtins.int before _find_iplace_dunder ever runs, missing
				# an i32-typed __iadd__ outright (see
				# _peek_single_dunder_param_type's own docstring)
				iplace_name = _IPLACE_BINOP_DUNDER.get( type( node.op ))
				right_hint = self._peek_single_dunder_param_type( existing.type, iplace_name ) if iplace_name is not None else None
				right = self._lower_expr( node.value, right_hint )
				iplace_method = self._find_iplace_dunder( existing.type, type( node.op ), right.type )
				if iplace_method is not None:
					self._emit_iplace_dunder_call( node, iplace_method, existing, right )
					return
				# no matching __iadd__ - reproduce `x = x + y` manually;
				# `right` is already lowered, so this must NOT delegate back
				# through a synthesized BinOp+_stmt_Assign (that would
				# re-lower node.value - a real double-evaluation risk this
				# file already guards against elsewhere, see the comment
				# above on Attribute/Subscript targets). is_alias=False
				# matches what the delegated path below would compute
				# anyway (_is_aliasing_expr never treats a bare BinOp as
				# aliasing, regardless of its operands)
				# _lower_binop_values is called directly here (not through
				# _lower_expr, which _expr_BinOp's own callers get for free) -
				# its own raw, possibly-still-Result-shaped return needs the
				# SAME case-2 auto-or_throw() coercion _lower_expr's tail
				# would otherwise apply, or an unconsumed checked-arithmetic
				# Result would get passed straight to _cfg_assign/ir.Assign
				# below - confirmed via a real repro (auto_or_throw_test.py's
				# own AugAssign coverage): `lst[0] += 10` compiled to invalid
				# C, passing a raw Result struct where __setitem__'s plain-T
				# parameter was declared
				result = self._coerce_or_check_operand( self._lower_binop_values( node, existing, right, existing.type ), existing.type, node )
				self._cfg.unnarrow( node.target.id )
				self._cfg_assign( existing, result, is_alias = False, node = node )
				return
			# not RC (or not yet declared) - unchanged: synthesized x = x + y
			read = ast.Name( id = node.target.id, ctx = ast.Load() )
			ast.copy_location( read, node.target )
			binop = ast.BinOp( left = read, op = node.op, right = node.value )
			ast.copy_location( binop, node )
			assign = ast.Assign( targets = [ node.target ], value = binop )
			ast.copy_location( assign, node )
			self._stmt_Assign( assign )
		elif isinstance( node.target, ast.Attribute ):
			obj, writeback = self._lower_attr_target_obj( node.target.value )
			attr_var = self.lowering._attr_lookup( obj.type, node.target.attr, node.target )
			self._check_field_visibility( obj.type, attr_var, node.target.attr, node.target )
			old = self._new_temp( attr_var.type )
			old_is_rc = bool( cfg.rc_leaves( attr_var.type )) and self._is_real_field_receiver( obj.type )
			if old_is_rc:
				# PLAN_THREAD_SAFE_SHARED_STATE.md Part B: `old` stays alive
				# (as the iplace-dunder receiver, or as an attr_replace/
				# _lower_binop_values operand) across evaluating node.value
				# below, which can run arbitrary code - a bare, unretained
				# GetAttr here would leave the exact "reader loads a pointer,
				# gets preempted before its own incref, writer frees it"
				# window A.3 describes, just for a field instead of a global.
				# Retained here, under its own critical section (never held
				# across node.value's own lowering - "lock the access, not
				# the statement"), and registered fresh_temp so it's treated
				# as an already-owned value the rest of this function - NOT
				# double-counted against attr_replace's own unconditional
				# decref of `old` below: that decref balances the FIELD's
				# original ownership, this incref adds a SEPARATE, temp-
				# owned reference that this statement's own pending_temps
				# cleanup balances at the end (see _new_temp/fresh_temp) -
				# two owners in, two decrefs out, whichever branch below runs.
				self._emit( ir.AcquireFieldLock( obj = obj, field = attr_var, exclusive = False )) # Cost mitigation #4: a read
				self._emit( ir.GetAttr( dest = old, obj = obj, attr = node.target.attr ))
				for instr in self._cfg.incref( attr_var.type, old ):
					self._emit( instr )
				self._emit( ir.ReleaseFieldLock( obj = obj, field = attr_var, exclusive = False ))
				self._cfg.fresh_temp( old, attr_var.type )
			else:
				self._emit( ir.GetAttr( dest = old, obj = obj, attr = node.target.attr ))
			field_is_rc = self.lowering._ensure_resolved( attr_var.type ).is_rc_pointer()
			if field_is_rc:
				# not attr_var.type itself - see the Name branch's identical
				# comment: a field's own RCClass type is never a sensible hint
				# for __iadd__'s independently-typed argument, and hinting a
				# bare literal toward it would wrongly REJECT it outright.
				# Peek at __iadd__'s own declared parameter type instead
				iplace_name = _IPLACE_BINOP_DUNDER.get( type( node.op ))
				right_hint = self._peek_single_dunder_param_type( attr_var.type, iplace_name ) if iplace_name is not None else None
				right = self._lower_expr( node.value, right_hint )
			else:
				usize_cls = self.lowering.discovery.get_intrinsics()['usize']
				right_hint = usize_cls if self.lowering._type_resolver._is_ptr_specialization( old.type ) else old.type
				right = self._lower_expr( node.value, right_hint )
			iplace_method = self._find_iplace_dunder( attr_var.type, type( node.op ), right.type ) if field_is_rc else None
			if iplace_method is not None:
				# old aliases the SAME heap object attr_var's own field
				# already points to (Part B's own temp-owned reference on
				# top, if old_is_rc - see its own comment above; the field's
				# own pointer value never changes) - mutating it in place via
				# __iadd__ already mutates the field's pointee, so SetAttr/
				# writeback would be redundant (and, for writeback in
				# particular, actively pointless - only the field's pointee
				# changed, not the struct holding the field). old's own
				# pending_temps cleanup (if old_is_rc) balances the extra
				# temp-owned reference automatically on this early return,
				# same as any other fresh_temp-registered value.
				self._emit_iplace_dunder_call( node, iplace_method, old, right )
				return
			# see the Name-target branch's identical comment on why this
			# needs its own explicit case-2 coercion
			result = self._coerce_or_check_operand( self._lower_binop_values( node, old, right, attr_var.type ), attr_var.type, node )
			needs_field_lock = False
			is_construction_write = self._construction_self is not None and obj is self._construction_self
			if not is_construction_write:
				# see _stmt_Assign's identical flip for why
				attr_var.field_reassigned_outside_init = True
			if is_construction_write:
				# self.<attr> += value, inside __init__ construction itself -
				# same definite-assignment/self-escape tracking an ordinary
				# self.<attr> = value gets in _stmt_Assign
				for instr in self._cfg.attr_assign( attr_var, result, is_alias = False ):
					self._emit( instr )
			elif cfg.rc_leaves( attr_var.type ):
				# ordinary SetAttr on an already-constructed instance - `old`
				# is exactly the CURRENT value _stmt_Assign's own Attribute
				# branch would otherwise re-read via its own fresh GetAttr;
				# reusing it here avoids a redundant third read. Its own
				# SEPARATE critical section, same reasoning _stmt_Assign's
				# identical branch already documents - not merged with the
				# read-side lock above (node.value's own lowering ran in
				# between, and could itself touch this SAME object's OTHER
				# fields - B.4's reentrancy hazard, avoided by never holding
				# the lock across anything but one field access).
				needs_field_lock = self._is_real_field_receiver( obj.type )
				if needs_field_lock:
					self._emit( ir.AcquireFieldLock( obj = obj, field = attr_var ))
				for instr in self._cfg.attr_replace( attr_var.type, old, result, is_alias = False ):
					self._emit( instr )
			self._emit( ir.SetAttr( obj = obj, attr = node.target.attr, value = result ))
			if needs_field_lock:
				self._emit( ir.ReleaseFieldLock( obj = obj, field = attr_var ))
			if writeback is not None:
				writeback( obj )
		elif isinstance( node.target, ast.Subscript ):
			# snapshotted before ANY of obj/index/old/right/result below are
			# built - same rationale as _stmt_Expr's own receiver_pending_
			# start (see self._discarded_call_pending_start's own comment):
			# both the get_dest and set_dest _auto_or_throw calls further
			# down can each early-return via an uncovered Err leaf, and their
			# own operand/argument temps (most commonly `result`, the fresh
			# combined value passed into the discarded __setitem__ call)
			# must be released on that path too, not just left pending for a
			# normal end-of-statement flush the early return skips right
			# past. Safe to snapshot this early because this whole branch IS
			# the full statement - no outer in-progress expression's own
			# pending temp could exist before this point to accidentally
			# flush (see _flush_new_pending_temps's own docstring on why
			# that would otherwise be unsafe).
			subscript_pending_start = len( self._pending_temps )
			obj = self._lower_expr( node.target.value, None )
			getitem_fn = self._find_indexlike_getitem( obj.type, node.target.slice )
			if getitem_fn is None:
				# no real __getitem__ declared (raw pointers, or any other
				# type that doesn't define subscript access as a method) -
				# mirrors _expr_Subscript's own raw-pointer GetItem fallback
				# and _stmt_Assign's own raw-pointer SetItem fallback, fused
				# around a single obj/index lowering
				if isinstance( obj.type, Specialization ) and obj.type.pointer_stem() is not None:
					elem_type = obj.type.args[0]
				else:
					self.lowering.discovery.fail( f'cannot infer the element type of {ast.unparse(node.target)} - no expected type available from context', node.target )
				index_type = self.lowering.discovery.get_intrinsics()['usize']
				index = self._lower_expr( node.target.slice, index_type )
				old = self._new_temp( elem_type )
				self._emit( ir.GetItem( dest = old, obj = obj, index = index ))
				right = self._lower_expr( node.value, old.type )
				# see the Name-target branch's identical comment on why this
				# needs its own explicit case-2 coercion
				result = self._coerce_or_check_operand( self._lower_binop_values( node, old, right, old.type ), old.type, node )
				self._emit( ir.SetItem( obj = obj, index = index, value = result ))
			else:
				# a real __getitem__/__setitem__ pair. Tries an __iadd__-
				# family in-place operator on the element type FIRST (RC-
				# class elements only - see _find_iplace_dunder): that path
				# only ever needs __getitem__'s own single Result consumed
				# and skips __setitem__ entirely. Falls back to the ordinary
				# get->combine->set sequence otherwise, with both Results'
				# coverage checked TOGETHER (_require_chained_result_return)
				# instead of two independent checks that could each fail at
				# a different time. index is lowered exactly once, shared by
				# both the get and (fallback) set calls.
				setitem_fn = self._find_indexlike_setitem( obj.type, node.target.slice )
				if setitem_fn is None:
					self.lowering.discovery.fail( f'{ast.unparse(node.target.value)} defines __getitem__ but not __setitem__ - cannot assign to {ast.unparse(node.target)}', node.target )
				self.lowering._ensure_resolved( getitem_fn )
				self.lowering.schedule( getitem_fn.return_type )
				self.lowering._ensure_resolved( setitem_fn )
				self.lowering.schedule( setitem_fn.return_type )
				index = self._lower_expr( node.target.slice, getitem_fn.parameters[0].type )
				# the element type - unwrapped from __getitem__'s own
				# Result[T,E] return type if fallible - is knowable purely
				# from getitem_fn's declared signature, before the get Call
				# is even emitted, which lets `right` (and the __iadd__
				# lookup, which needs right's type) be resolved BEFORE
				# deciding which path to take, exactly the same "lower the
				# target/its type first so the rvalue infers against it"
				# ordering the Attribute/Name branches already use
				get_shape = self.lowering._type_resolver._result_shape( getitem_fn.return_type )
				elem_type = get_shape[0] if get_shape is not None else getitem_fn.return_type
				elem_is_rc = self.lowering._ensure_resolved( elem_type ).is_rc_pointer()
				# elem_type itself is never the hint when the element is RC -
				# see the Attribute/Name branches' identical comment. Peek at
				# __iadd__'s own declared parameter type instead
				if elem_is_rc:
					iplace_name = _IPLACE_BINOP_DUNDER.get( type( node.op ))
					right_hint = self._peek_single_dunder_param_type( elem_type, iplace_name ) if iplace_name is not None else None
				else:
					right_hint = elem_type
				right = self._lower_expr( node.value, right_hint, strict = False )
				iplace_method = self._find_iplace_dunder( elem_type, type( node.op ), right.type ) if elem_is_rc else None
				get_dest = self._new_temp( getitem_fn.return_type )
				self._emit( ir.Call( dest = get_dest, target = getitem_fn, receiver = obj, args = [ index ], kwargs = {} ))
				if iplace_method is not None:
					old = self._maybe_consume_result( node.target, get_dest, self.lowering._SUBSCRIPT_ALTERNATIVES )
					# old already went through __getitem__'s own incref (see
					# lib/builtins/__list.py's __getitem__), so it's a
					# genuine extra owned reference to the SAME heap object
					# the container's slot stores - mutating it in place via
					# __iadd__ already mutates what the container holds.
					# __setitem__ is skipped entirely: this is the concrete
					# fix for needing two separately-covered Results (one
					# from __getitem__, one from __setitem__) for what's
					# conceptually one fallible operation.
					# When the Result was fallible, _maybe_consume_result's
					# own or_throw() extraction incref's this owned copy but
					# (unlike get_dest's Call dest, auto-registered by _emit)
					# never registers IT as a fresh_temp - it's normally handed
					# straight to a Variable's Assign, whose own lifetime
					# absorbs the ownership. Here it's only ever passed as the
					# iplace call's receiver, so without this explicit
					# registration nothing ever decrefs it - a leaked Counter
					# confirmed via the leak-check epilogue. Idempotent when
					# `old` is still just get_dest (non-fallible __getitem__),
					# already registered by its own Call emission.
					self._cfg.fresh_temp( old, elem_type )
					self._emit_iplace_dunder_call( node, iplace_method, old, right )
					return
				set_shape = self.lowering._type_resolver._result_shape( setitem_fn.return_type )
				error_classes = [ shape[1] for shape in ( get_shape, set_shape ) if shape is not None ]
				if error_classes:
					result_cls = self.lowering.discovery.find_name( 'Result', node.target )
					self.lowering._type_resolver._require_chained_result_return(
						node.target, result_cls, error_classes, self.lowering._SUBSCRIPT_ALTERNATIVES, fn = self._current_fn,
					)
				# pre_checked=True: coverage for BOTH steps' error types was
				# already validated together above via
				# _require_chained_result_return - _auto_or_throw must not
				# re-validate (and re-report) either one on its own
				old = (
					self._auto_or_throw(
						node.target, get_dest, self.lowering._SUBSCRIPT_ALTERNATIVES, want_result = True, pre_checked = True,
						receiver_pending_start = subscript_pending_start,
					)
					if get_shape is not None else get_dest
				)
				assert old is not None # want_result=True above guarantees this
				# see the Name-target branch's identical comment on why this
				# needs its own explicit case-2 coercion (setitem_fn's own
				# `val` parameter wants old.type directly, never a Result)
				result = self._coerce_or_check_operand( self._lower_binop_values( node, old, right, old.type ), old.type, node )
				if setitem_fn.return_type is self.lowering.discovery.get_none_type():
					self._emit( ir.Call( dest = None, target = setitem_fn, receiver = obj, args = [ index, result ], kwargs = {} ))
				else:
					set_dest = self._new_temp( setitem_fn.return_type )
					self._emit( ir.Call( dest = set_dest, target = setitem_fn, receiver = obj, args = [ index, result ], kwargs = {} ))
					if set_shape is not None:
						self._auto_or_throw(
							node.target, set_dest, self.lowering._SUBSCRIPT_ALTERNATIVES, want_result = False, pre_checked = True,
							receiver_pending_start = subscript_pending_start,
						)
		else:
			self.lowering.discovery.fail( f'unsupported AugAssign target: {ast.unparse(node)}', node )

	def _stmt_Expr( self, node: ast.Expr ) -> None:
		if self._super_init_shape( node.value ) is not None:
			# only ever consumed specially as literally __init__'s own first
			# statement (_lower_super_init_if_required, called BEFORE this
			# per-statement loop even starts) - reaching here at all means
			# it's either not statement 0, or this isn't even __init__, or
			# there's no base to chain to in the first place
			self.lowering.discovery.fail(
				f'super().__init__(...) is only allowed as the literal first statement of a subclass\'s own __init__: {ast.unparse(node)}',
				node,
			)
		defer_kind = self.lowering._defer_kind_of_call( node.value )
		if defer_kind is not None:
			if len( node.value.args ) != 1 or node.value.keywords:
				self.lowering.discovery.fail( f'{defer_kind}(...) takes exactly one argument: {ast.unparse(node)}', node )
			single_stmt = ast.Expr( value = node.value.args[0] )
			ast.copy_location( single_stmt, node )
			self._register_defer_block( is_err_only = ( defer_kind == 'errdefer' ), body = [ single_stmt ], node = node )
			return
		if isinstance( node.value, ast.Constant ) and isinstance( node.value.value, str ):
			return # a docstring (or any other bare string literal used as a statement) - a no-op, same as _stmt_Pass
		if self.lowering._is_compiler_call( node.value ) == 'early_return':
			self._lower_compiler_early_return( node.value )
			return
		if self.lowering._is_compiler_call( node.value ) == 'decref':
			self._lower_compiler_decref( node.value )
			return
		if self.lowering._is_compiler_call( node.value ) == '__internal_decref__':
			self._lower_compiler_internal_decref( node.value )
			return
		if self.lowering._is_compiler_call( node.value ) == 'incref':
			self._lower_compiler_incref( node.value )
			return
		if self.lowering._is_compiler_call( node.value ) == 'atomic_store':
			self._lower_compiler_atomic_store( node.value )
			return
		if self.lowering._is_compiler_call( node.value ) == 'decref_dynamic':
			self._lower_compiler_decref_dynamic( node.value )
			return
		if self.lowering._is_compiler_call( node.value ) == '__raw_free__':
			self._lower_compiler_raw_free( node.value )
			return
		if self.lowering._is_compiler_call( node.value ) == 'dump_live_objects':
			self._lower_compiler_dump_live_objects( node.value )
			return
		if self.lowering._is_compiler_call( node.value ) == '__debug_raw_track__':
			self._lower_compiler_debug_raw_track( node.value )
			return
		if self.lowering._is_compiler_call( node.value ) == '__debug_raw_untrack__':
			self._lower_compiler_debug_raw_untrack( node.value )
			return
		if self.lowering._is_compiler_call( node.value ) == '__debug_track_immortal_cache__':
			self._lower_compiler_debug_track_immortal_cache( node.value )
			return
		if isinstance( node.value, ast.Yield ):
			# PLAN_GENERATORS.md Phase F - a bare (statement-position)
			# `yield expr` inside a generator's $$__next__ body
			# (type_resolver.py only ever emits ast.Yield in a
			# is_generator_next function - see ensure_generator_
			# synthesized's own top-of-file docstring)
			self._lower_generator_yield( node.value )
			return
		if isinstance( node.value, ast.YieldFrom ):
			# PLAN_GENERATORS.md A.4a follow-up - type_resolver.py's
			# _desugar_generator_yield_from always rewrites a real `yield
			# from` into an ordinary for-loop before lowering ever runs,
			# so reaching HERE means the SAME "generator whose own
			# ensure_generator_synthesized run aborted partway through,
			# for an unrelated already-reported reason" situation _lower_
			# generator_yield's own identical check handles - see its
			# docstring for the full explanation. A graceful discovery.
			# fail() here too, instead of falling through to the generic
			# "unsupported expression statement" message below (which is
			# technically correct but confusingly generic for what's
			# really a cascading secondary error, not a new one).
			self.lowering.discovery.fail(
				f'{self._current_fn.qualname}: yield from reached outside a successfully-synthesized generator '
				f'(an earlier, already-reported error left this generator only partially built) - see PLAN_GENERATORS.md',
				node,
			)
			return
		if not isinstance( node.value, ast.Call ):
			self.lowering.discovery.fail( f'unsupported expression statement: {ast.unparse(node)}', node )
		# snapshotted BEFORE lowering the call at all (its own callee/args are
		# about to be built) - see self._discarded_call_pending_start's own
		# comment on why: a discarded Result-typed call whose Err leaf auto-
		# propagates needs its OWN argument temps released on that early-
		# return path too, exactly like an explicit .or_throw()/or_return()
		# call's receiver_pending_start already does for the receiver
		self._discarded_call_pending_start = len( self._pending_temps )
		self._lower_call( node.value, None, want_result = False )
		self._discarded_call_pending_start = None

	def _stmt_With( self, node: ast.With ) -> None:
		if len( node.items ) != 1:
			# NOT ast.unparse(node) here - that would dump this with-
			# statement's entire BODY into the message too (a real repro:
			# a multi-statement with-block produced a multi-line error
			# message that buried the actual problem)
			self.lowering.discovery.fail( f'unsupported with statement (only a single context manager is supported, got {len(node.items)})', node )
		item = node.items[0]
		context_expr = item.context_expr

		if item.optional_vars is None:
			defer_kind = self.lowering._defer_kind_of_with( context_expr )
			if defer_kind is not None:
				self._register_defer_block( is_err_only = ( defer_kind == 'errdefer' ), body = node.body, node = node )
				return

			attr = self.lowering._is_compiler_attr( context_expr )
			mode: arithmetic_mode.ArithmeticMode|None = None
			if attr == 'wrap_arithmetic':
				mode = arithmetic_mode.ArithmeticWrap()
			elif attr == 'saturate_arithmetic':
				mode = arithmetic_mode.ArithmeticSaturate()
			elif self.lowering._is_compiler_call( context_expr ) == 'panic_arithmetic':
				if len( context_expr.args ) != 1 or context_expr.keywords:
					self.lowering.discovery.fail( f'compiler.panic_arithmetic(...) takes exactly one argument: {ast.unparse(context_expr)}', node )
				str_cls = self.lowering.discovery.find_name( 'str', node )
				errmsg = self._lower_expr( context_expr.args[0], str_cls )
				mode = arithmetic_mode.ArithmeticPanic( errmsg )

			if mode is not None:
				self._arithmetic_mode.append( mode )
				try:
					for stmt in node.body:
						# same per-statement recovery boundary as the top-level loop
						# in lower_function - one bad statement inside the with-block
						# doesn't stop the rest of it from being lowered
						try:
							self._lower_stmt( stmt )
						except CompileError:
							continue
				finally:
					self._arithmetic_mode.pop()
				return

		# general context-manager form: with EXPR [as NAME]: BODY - EXPR's
		# type supplies __enter__(self)/__exit__(self), neither of the
		# special-cased shapes above (defer/errdefer, compiler.*_arithmetic)
		self._lower_with_context_manager( node, item )

	def _lower_with_context_manager( self, node: ast.With, item: ast.withitem ) -> None:
		''' `with EXPR [as NAME]: BODY` for a user-defined context manager -
		EXPR's type must supply __enter__(self)->T and __exit__(self)->None.
		Desugars to (as plain AST, fed back through the ordinary statement
		pipeline, same idiom type_resolver.py's own desugaring passes use):
			__with_ctx_N = EXPR
			[NAME = ]__with_ctx_N.__enter__()
			with defer: __with_ctx_N.__exit__()
			BODY
		reusing _register_defer_block for the guaranteed-once-per-entry,
		runs-on-every-exit-path contract - UNLIKE defer/errdefer, this is
		fine inside a loop as long as BODY always falls through to its own
		natural end: that path disarms the registered defer and calls
		__exit__() directly, once per iteration, right where written -
		there's no "runs once, ever" hazard because nothing here waits for
		the function's own eventual epilogue. The hazard - and the reason
		defer/errdefer itself bans loops outright (single armed/captured
		slot, not one per iteration) - only applies when BODY can leave via
		a Break/Continue that escapes to an enclosing loop (Return is fine -
		it ends the function outright, so there's no later iteration left to
		lose track of); see the loop check below and
		_body_may_break_or_continue_to_enclosing_loop. Also has the same
		"not inside a generator's own body" restriction (Mechanism 2's defer-replay is a
		SEPARATE, generator-specific path this doesn't integrate with yet -
		a clean rejection, not attempted here).
		__exit__ always runs unconditionally on every exit path - this
		compiler has no Python-style exception propagation for __exit__ to
		observe or suppress, so there's no exc_type/exc_value/traceback
		parameter, unlike Python's own protocol; a Result-returning
		__enter__/__exit__ works the same as any other bare/bound call
		(the ordinary "an unconsumed Result is a compile error" rule
		applies exactly as it would to hand-written code, nothing special
		here consumes or requires one).

		__with_ctx_N/NAME (`as NAME`) are ORDINARY, function-scoped locals,
		exactly like any other name introduced anywhere in this compiler
		(cfg.py's own module docstring: no block scoping at all) - NOT torn
		down early at this with-statement's own textual end. An earlier
		version of this feature DID scope them (reusing _stmt_If's own
		branch-confinement machinery, treating the whole with-block as an
		unconditionally-taken "if branch") - wrong, and reverted: `with`
		does not introduce a lifetime scope (real Python's own `with EXPR as
		NAME:` doesn't either - NAME stays bound and alive for the rest of
		the enclosing function/scope there too), and a with-statement's
		BODY - unlike an if-branch's body - always executes exactly once
		when reached, so ANY local BODY itself declares needs to survive
		past the with-statement's own end the same way it would if the same
		statements were written with no with-statement wrapping them at
		all. Confirmed via a real repro: `with Ctx(): x: i32 = 5` followed
		by `return x` wrongly reported `'x' is not initialized on all code
		branches` under the branch-confinement version - `merge_if`'s
		confinement applies to EVERY binding newly introduced inside the
		window, not just __with_ctx_N/NAME, so there was no way to confine
		only those two without also breaking every ordinary local BODY
		declares. `with x:` on an EXISTING object (not a fresh construction)
		is the other motivating case: __with_ctx_N then aliases x rather
		than owning a fresh construction, but ordinary aliasing assignment
		in this language (`ctx = x`, a plain Name read) still takes its own
		independent Incref - same as any other `y = x` - so it stays
		correctly balanced by its own (now function-scoped, not block-
		scoped) eventual release; nothing here needs to special-case that
		case, it just needs to NOT be forced into an artificial block scope
		that has no basis in either this language's own "no block scoping"
		design or real Python's own `with` semantics (which also introduces
		no new scope - NAME/x stay bound and alive for the rest of the
		enclosing scope there too). '''
		# neither message below unparses `node` itself - it's the whole
		# with-statement INCLUDING its body, which would dump the entire
		# block into the error message (a real repro: a multi-statement
		# with-body produced a multi-line error that buried the actual
		# problem); `node`'s own lineno (passed as the location) already
		# pinpoints the with-statement precisely enough
		if self._loop_depth > 0 and self._body_may_break_or_continue_to_enclosing_loop( node.body ):
			self.lowering.discovery.fail(
				'with-statement (context manager) is not allowed inside a loop when its body can break/continue out '
				'of that loop - call another function and use the with-statement inside that instead', node,
			)
		if self._current_fn.is_generator_next:
			self.lowering.discovery.fail(
				'with-statement (context manager) is not supported inside a generator body yet', node,
			)

		index = self._with_ctx_id
		self._with_ctx_id += 1
		ctx_name = f'__with_ctx_{index}'

		context_expr = item.context_expr
		ctx_assign = ast.Assign( targets = [ ast.Name( id = ctx_name, ctx = ast.Store() ) ], value = context_expr )
		ast.fix_missing_locations( ast.copy_location( ctx_assign, node ))
		self._lower_stmt( ctx_assign )

		ctx_var = self._existing_local_or_none( ctx_name, node, 'with-statement context expression' )
		assert ctx_var is not None # just declared immediately above - _lower_stmt would have raised on failure
		ctx_type = ctx_var.type

		if self.lowering._find_method( ctx_type, '__enter__' ) is None or self.lowering._find_method( ctx_type, '__exit__' ) is None:
			type_name = ctx_type.qualname if ctx_type is not None else '?'
			self.lowering.discovery.fail(
				f'with-statement requires {type_name} to define both __enter__(self) and __exit__(self): {ast.unparse(context_expr)}', node,
			)

		def _ctx_read() -> ast.Name:
			n = ast.Name( id = ctx_name, ctx = ast.Load() )
			ast.fix_missing_locations( ast.copy_location( n, node ))
			return n

		enter_call = ast.Call( func = ast.Attribute( value = _ctx_read(), attr = '__enter__', ctx = ast.Load() ), args = [], keywords = [] )
		ast.fix_missing_locations( ast.copy_location( enter_call, node ))
		enter_stmt: ast.stmt
		if item.optional_vars is not None:
			assert isinstance( item.optional_vars, ast.Name )
			enter_stmt = ast.Assign( targets = [ ast.Name( id = item.optional_vars.id, ctx = ast.Store() ) ], value = enter_call )
		else:
			enter_stmt = ast.Expr( value = enter_call )
		ast.fix_missing_locations( ast.copy_location( enter_stmt, node ))
		self._lower_stmt( enter_stmt )

		def _make_exit_stmt() -> ast.stmt:
			# a FRESH node every call, never reused across the two sites
			# below - lowering attaches mutable per-occurrence attributes to
			# a node as it processes it (same reason _build_defer_replay_
			# guards' own resume_call lambda in type_resolver.py rebuilds
			# fresh each time, not once and shared)
			call = ast.Call( func = ast.Attribute( value = _ctx_read(), attr = '__exit__', ctx = ast.Load() ), args = [], keywords = [] )
			ast.fix_missing_locations( ast.copy_location( call, node ))
			stmt = ast.Expr( value = call )
			ast.fix_missing_locations( ast.copy_location( stmt, node ))
			return stmt

		# _register_defer_block gives FUNCTION-scoped semantics (runs once,
		# from here to wherever the function actually ends, regardless of
		# what code follows this with-statement) - exactly right for an
		# early return/break/continue reached from INSIDE this with-block's
		# own body, but too broad on its own: a with-statement's __exit__
		# must run when THIS BLOCK is left, not merely "sometime before the
		# function ends". So: register it as a defer (covers every early-
		# exit path from inside the body below), then - only if the body
		# can actually fall through to its own natural end (_stmt_diverges:
		# false unless every path through the body already returns/breaks/
		# continues) - explicitly DISARM that defer's flag and call
		# __exit__() directly, right here, matching the block's real
		# lexical extent. Without the disarm, a with-statement followed by
		# more code in the same function would see __exit__ fire twice:
		# once here (if this were a plain second call with no disarm) AND
		# again later at the function's own eventual exit - confirmed via a
		# real compile+run repro (list.append() call counts, __exit__'s own
		# side effects observably running at the wrong point in the
		# program's actual output order, not just "eventually").
		# allow_inside_loop=True is safe here regardless of _loop_depth: the
		# rejection above already ran for the one case that would matter (BODY
		# can exit early and skip the direct disarm-and-call path below) - if
		# we get here inside a loop, BODY always falls through to its own
		# natural end, so this registered defer is always disarmed again a few
		# lines down and never actually replayed at the function's own epilogue
		exit_entry = self._register_defer_block( is_err_only = False, body = [ _make_exit_stmt() ], node = node, allow_inside_loop = True )

		for stmt in node.body:
			# same per-statement recovery boundary as the arithmetic-mode/
			# defer-body cases above
			try:
				self._lower_stmt( stmt )
			except CompileError:
				continue

		if not node.body or not self._stmt_diverges( node.body[-1] ):
			for instr in self._cfg.disarm_defer( exit_entry ):
				self._emit( instr )
			self._lower_stmt( _make_exit_stmt() )

	def _body_may_break_or_continue_to_enclosing_loop( self, body: list[ast.stmt] ) -> bool:
		''' true if any statement in `body` can leave it via a Break/
		Continue that targets an ENCLOSING loop rather than one `body`
		itself introduces - the only shape unsafe for
		_lower_with_context_manager to place inside a loop (see that
		function's own loop check for why). Return is deliberately NOT
		treated as a hazard here, even though it also skips the with-
		statement's direct disarm-and-call step: Return terminates the
		whole function immediately, so however many loop iterations already
		ran, there's no "next iteration" left to lose track of - the
		registered defer replays exactly once, correctly, against whichever
		iteration's own ctx is live at that point. That's the real
		difference from Break/Continue, which don't end the function and so
		can revisit this with-statement on a later iteration before the
		function ever truly ends - defer/errdefer's own "single armed slot,
		replayed once" contract can't represent more than one such visit.
		match statements are already desugared to chained ast.If by the
		time lowering.py runs (mirroring _stmt_diverges's own assumption).
		try/except (ast.Try) is handled by _stmt_may_break_or_continue
		itself, below. '''
		return any( self._stmt_may_break_or_continue( stmt, in_nested_loop = False ) for stmt in body )

	def _stmt_may_break_or_continue( self, stmt: ast.stmt, in_nested_loop: bool ) -> bool:
		if isinstance( stmt, ( ast.Break, ast.Continue )):
			return not in_nested_loop
		if isinstance( stmt, ( ast.For, ast.While )):
			return any( self._stmt_may_break_or_continue( s, in_nested_loop = True ) for s in stmt.body )
		if isinstance( stmt, ( ast.If, ast.With )):
			return any( self._stmt_may_break_or_continue( s, in_nested_loop ) for s in stmt.body )
		if isinstance( stmt, ast.Try ):
			# a break/continue anywhere in body/orelse/finalbody/any handler's
			# own body still belongs to the SAME enclosing loop - mirrors
			# _loop_has_reachable_break's own already-existing ast.Try walk
			lists = [ stmt.body, stmt.orelse, stmt.finalbody ] + [ h.body for h in stmt.handlers ]
			return any( self._stmt_may_break_or_continue( s, in_nested_loop ) for lst in lists for s in lst )
		return False

	def _body_contains_return( self, body: list[ast.stmt] ) -> bool:
		''' true if `return` appears anywhere in `body` (recursively, through
		nested if/while/for/with/try - match is already desugared to chained
		ast.If by the time lowering.py runs). Used to reject `return` inside a
		try-statement's own `finally:` body: `finally` is captured ONCE
		(_register_defer_block) and replayed via goto at every early-exit
		path the try/except construct has, PLUS inlined directly at its own
        normal-fallthrough point - a real `return` baked into that captured
		replay would fire from whichever of those unrelated call sites
		happens to replay it, discarding that path's own actual return value
		(Python's own well-known finally-return footgun, made structurally
		worse here since one `return` would silently execute at multiple,
		textually-unrelated points instead of just shadowing the one
		enclosing try/except it lexically appears in). '''
		return any( self._stmt_contains_return( stmt ) for stmt in body )

	def _stmt_contains_return( self, stmt: ast.stmt ) -> bool:
		if isinstance( stmt, ast.Return ):
			return True
		if isinstance( stmt, ast.With ):
			return any( self._stmt_contains_return( s ) for s in stmt.body )
		if isinstance( stmt, ( ast.For, ast.While, ast.If )):
			return any( self._stmt_contains_return( s ) for s in stmt.body + stmt.orelse )
		if isinstance( stmt, ast.Try ):
			lists = [ stmt.body, stmt.orelse, stmt.finalbody ] + [ h.body for h in stmt.handlers ]
			return any( self._stmt_contains_return( s ) for lst in lists for s in lst )
		return False

	def _static_type_of_value_expr( self, node: ast.expr ) -> Type|None:
		# compile-time-only: the static type of a value-shaped expression
		# (Name/Attribute) - no IR emitted, x itself is never evaluated or
		# lowered (unlike _lower_expr, which would emit a real GetAttr for
		# e.g. self.field, or even execute a call, just to inspect its
		# .type). Used by compiler.sizeof(x)'s value-argument fallback so
		# that e.g. compiler.sizeof(self) never turns self into a real
		# instruction operand - self is read only as self.type here, so
		# cfg.py's check_self_escape (which only inspects instruction
		# operands) never sees it, even mid-__init__ before construction
		# completes
		if isinstance( node, ast.Name ):
			name = self.lowering.discovery.find_name( node.id, node )
			if not isinstance( name, Variable ):
				return None
			member = self._cfg.narrowed_member( node.id )
			return member.type if member is not None else name.type
		if isinstance( node, ast.Attribute ):
			# checks THIS node's own full chain key (not just a single hop
			# off node.value) - mirrors the ast.Name branch's own
			# narrowed_member lookup above, and _expr_Attribute's identical
			# check (its own narrowed-read extraction comment has the full
			# reasoning). Without this, a narrowed field chain
			# (`self.a.b.c`, at ANY depth) reported its plain declared
			# (still-union) type here, which broke any FURTHER attribute
			# hop chained on top of it (`self.a.b.c.method()`): the
			# recursive call one level up would then _attr_lookup the next
			# name against the whole union instead of the narrowed leaf
			# and fail outright, instead of gracefully declining (returning
			# None) the way an unrelated non-callable shape does.
			chain_key = self._attribute_chain_key( node )
			if chain_key is not None:
				member = self._cfg.narrowed_member( chain_key )
				if member is not None:
					return member.type
			owner_type = self._static_type_of_value_expr( node.value )
			if owner_type is None:
				return None
			# a non-failing probe, not _attr_lookup - a @property getter has
			# no static type here (reading it needs a real call, which this
			# function never emits) and must decline (None) rather than
			# _attr_lookup's fatal "has no attribute" for a non-Variable
			# find. Confirmed by a real repro: `b.inner.go()` where b.inner
			# is a property failed to compile even though `x = b.inner;
			# x.go()` worked fine.
			field = self.lowering._find_field( owner_type, node.attr )
			return field.type if field is not None else None
		if ( isinstance( node, ast.Call ) and isinstance( node.func, ast.Attribute )
				and isinstance( node.func.value, ast.Name ) and node.func.value.id == 'compiler'
				and node.func.attr == 'cast' and len( node.args ) == 2 ):
			# compiler.cast(T, x) - the env-read rewrite
			# _rewrite_captures_into_env_reads produces for a captured name
			# inside a lambda/nested-def body (env.field becomes
			# compiler.cast(env_type, erased_env).field). Mirrors
			# _lower_compiler_cast's own target-type resolution, without ever
			# lowering node.args[1] - this node is never evaluated here, only
			# inspected, same as every other branch above.
			target_type = getattr( node.args[0], 'resolved_type', None )
			if target_type is None:
				target_type = self.lowering._try_resolve_namespace( node.args[0] )
			return target_type
		return None

	def _stmt_If( self, node: ast.If ) -> None:
		test = self._lower_branch_condition( node.test )
		else_label = self._new_label( 'if_else' )
		self._emit( ir.JumpIfFalse( cond = test, target = else_label ))

		# each branch is lowered into its OWN captured instruction list
		# (same technique _register_defer_block already uses) rather than
		# appended directly - a local confined to just one branch needs its
		# own Decref spliced into THAT branch's own code specifically
		# (before its own exit to the join point), never at the shared
		# join point both branches reach, since it only exists on that one
		# path. Known ahead of time only after BOTH branches have been
		# explored (merge_if, below), so neither branch's own instructions
		# can be emitted directly as they're lowered
		entry_snapshot = self._cfg.snapshot()
		fn = self._current_fn
		# fn.names is shared, mutable, function-wide state (no block scoping
		# - cfg.py's own module docstring) that _stmt_Delete mutates
		# directly (`del fn.names[x]`) with no snapshot/restore of its own,
		# unlike self._cfg's state just above. Without this, `del x` inside
		# ONE branch (even a TERMINATING one, which never reaches the join
		# at all) permanently removes x from fn.names for the rest of the
		# function - a sibling branch's own independent `del x`, or plain
		# code after the if that expects x to still be a real name, then
		# fails with a spurious "not a local variable"/"is not defined"
		# instead of the correct per-path "not initialized on all code
		# branches" _live already reports. Reconciled (both before lowering
		# the false branch, and again at the very end) the same "never let a
		# REMOVAL escape, only let a genuinely NEW name survive" way
		# merge_if()'s own `removed` already is for RC bindings (see below):
		# a name that predates the if simply reverts to its entry state
		# regardless of which branch(es) del'd it, but a name FRESHLY
		# declared inside one branch stays visible even to its own sibling -
		# unlike a del'd removal, this is a real, established, already-
		# tested shape (an explicit `x: T = ...` annotation is rejected as
		# "already declared" if a SECOND, mutually-exclusive branch also
		# declares it - deliberately, regardless of branch exclusivity, see
		# lowering_test.py's test_annotated_redeclaration_across_branches_
		# is_a_compile_error) that restoring fn.names wholesale back to
		# entry before the false branch would silently break, by making the
		# true branch's own fresh declaration invisible to the false
		# branch's own _existing_local_or_none check.
		entry_names = dict( fn.names )
		def _reconcile_names( *branch_ends: dict ) -> None:
			fn.names.clear()
			fn.names.update( entry_names )
			for branch_end in branch_ends:
				for k, v in branch_end.items():
					if k not in entry_names:
						fn.names[k] = v
		outer_instructions = self._instructions
		self._instructions = []
		# no special protection needed here for entries that SURVIVE this if
		# (index < entry_snapshot.stack_depth) even though both branches are
		# independently restore()'d back to this SAME entry_snapshot below -
		# cfg.py's own _neutralize() never mutates a shared Epilogue object
		# in place (see its own docstring), so a plain `if bad: compiler.
		# decref(g); return -1` correctly leaves g's entry untouched for the
		# false branch's own restore()'d view - the true branch's own
		# compiler.decref(g) builds a REPLACEMENT entry instead of touching
		# the shared one.
		# see cfg.py's CFGState.enter_branch's own docstring: lets
		# current_epilogue_label() recognize an RC entry pushed while
		# lowering THIS branch (e.g. a match arm's own payload binding) as
		# branch-confined - restore(), called once this branch's fully
		# lowered, silently drops it, so a `return` inside here must never
		# be handed that entry's own label as a shared jump target
		self._cfg.enter_branch( entry_snapshot.stack_depth )
		try:
			for stmt in node.body:
				try:
					self._lower_stmt( stmt )
				except CompileError:
					continue
		finally:
			self._cfg.exit_branch()
		true_captured = self._instructions
		true_end = dict( self._cfg.bindings )
		true_end_results = self._cfg.unchecked_results()
		true_end_narrowed = self._cfg.narrowed_snapshot()
		true_end_live = self._cfg.live_snapshot()
		true_names_end = dict( fn.names )
		# return/break/continue as a branch's own last statement means
		# that branch never reaches the if's join point at all - see
		# merge_if()'s own comment on why that has to be treated
		# differently from an ordinary falling-through branch (full
		# terminator/dead-code analysis for anything deeper - nested ifs
		# that both terminate, etc - is future work, not attempted here)
		true_terminates = bool( node.body ) and self._stmt_diverges( node.body[-1] )

		if node.orelse:
			self._cfg.restore( entry_snapshot )
			# only undoes a REMOVAL (del) the true branch made to a
			# pre-existing name - true's own NEW declarations stay visible
			# to the false branch (see entry_names' own comment above)
			_reconcile_names( true_names_end )
			self._instructions = []
			self._cfg.enter_branch( entry_snapshot.stack_depth )
			try:
				for stmt in node.orelse:
					try:
						self._lower_stmt( stmt )
					except CompileError:
						continue
			finally:
				self._cfg.exit_branch()
			false_captured = self._instructions
			false_end = dict( self._cfg.bindings )
			false_end_results = self._cfg.unchecked_results()
			false_end_narrowed = self._cfg.narrowed_snapshot()
			false_end_live = self._cfg.live_snapshot()
			false_names_end = dict( fn.names )
			false_terminates = bool( node.orelse ) and self._stmt_diverges( node.orelse[-1] )
		else:
			false_captured = []
			false_end = dict( entry_snapshot.bindings )
			false_end_results = set( entry_snapshot.results )
			false_end_narrowed = dict( entry_snapshot.narrowed )
			false_end_live = set( entry_snapshot.live )
			false_names_end = dict( entry_names )
			false_terminates = False

		self._cfg.restore( entry_snapshot )
		_reconcile_names( true_names_end, false_names_end )
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
		# `removed` only drives the RC Decref instructions above (already
		# spliced into true_extra/false_extra) - it must NOT also remove
		# these names from fn.names. This language has no block scoping (see
		# cfg.py's own module docstring): a name introduced anywhere in the
		# function body stays a real name for the rest of it. Whether a
		# later reference is actually valid is now the new _live mechanism's
		# job (_expr_Name's own liveness gate) - it correctly reports "not
		# initialized on all code branches" for exactly this case, instead
		# of the misleading "is not defined" this loop used to produce by
		# deleting the name out from under a later reference entirely.

		for instr in true_captured:
			self._emit_captured( instr )
		for instr in true_extra:
			self._emit( instr )
		if node.orelse:
			end_label = self._new_label( 'if_end' )
			self._emit( ir.Jump( target = end_label ))
			self._emit( ir.Label( name = else_label ))
			for instr in false_captured:
				self._emit_captured( instr )
			for instr in false_extra:
				self._emit( instr )
			self._emit( ir.Label( name = end_label ))
		elif false_extra:
			# false_extra used to always be empty here (merge_if's "fresh on
			# exactly one branch" case can only ever populate
			# false_instructions when a real ast.orelse existed, since
			# false_end is otherwise just entry_bindings copied verbatim -
			# see false_end's own fallback above) - but merge_if's
			# ownership-disagreement flag reconciliation (the "if x is
			# None: x = Owned(...)" idiom, no else needed) DOES need to land
			# a disarm Assign on exactly this implicit "condition was
			# false" path. Needs the SAME Jump-over-else the real-orelse
			# branch above already has: true_captured/true_extra fall
			# straight through into else_label with no separating Jump, so
			# WITHOUT one, this disarm ran on BOTH paths unconditionally -
			# not just the "condition was false" one it's actually meant
			# for - permanently disarming the flag regardless of which
			# branch ran, silently leaking whatever the if-branch's own
			# reassignment made owned (confirmed via a real repro: `if not
			# ys: ys = make()`, ys a parameter, leaked the list make()
			# returns on the path that reassigns it - flag defaults armed,
			# but this unconditional disarm cancelled it either way).
			end_label = self._new_label( 'if_end' )
			self._emit( ir.Jump( target = end_label ))
			self._emit( ir.Label( name = else_label ))
			for instr in false_extra:
				self._emit( instr )
			self._emit( ir.Label( name = end_label ))
		else:
			self._emit( ir.Label( name = else_label ))
