# stdlib imports:
import ast
import copy
from typing import Callable

# local imports:
import cfg
import ir
from errors import CompileError
from mpy_types import (
	Name, Type, Variable, Function, Overload, ClassLike, Module, Specialization, TaggedUnion, CStruct, CUnion, CEnum, TypeVar, Move, Copy, RCClass, Scalar, ClosureType, TupleType, FixedArrayType, int_stem_range,
)
from type_resolver import TypeResolver

from lowering_shared import _ALTERNATIVES_BY_ERROR, _BINOP_DUNDER, _REFLECTED_BINOP_DUNDER, _COMP_DUNDER, _UNARYOP_DUNDER, _FLOAT_UNSUPPORTED_BINOPS
from mpy_types import is_float_scalar as _is_float_scalar, is_signed_scalar as _is_signed_scalar

class ExprLoweringMixin:
	''' plain expression lowering (_expr_*), checked-arithmetic, and comparison/cast helpers - mixed into FunctionLowering (lowering.py), which
	see for the shared instance state (self._instructions, self._cfg, self.lowering,
	etc.) every method here reads and writes. Never instantiated on its own;
	split out of lowering.py purely to keep that file to a manageable size - see
	lowering.py's own class docstring and FunctionLowering's base-class list for
	the full set of sibling mixins this one is composed with. '''


	# --- expressions -----------------------------------------------------------

	def _lower_expr( self, node: ast.expr, expected_type: Type|None, *, strict: bool = True, context: str|None = None ) -> ir.Operand:
		''' `strict=False` (default True): `expected_type` here is a HINT for
		inference (e.g. _lower_binary_operands passing the left operand's own
		type down to help type an untyped literal, or to help a nested
		generic call bind its type params) rather than a real requirement the
		lowered operand must satisfy - skips the safe-scalar-widening
		coercion and the final _check_assignable rejection (both new; see
		their own comments below), but still applies the pre-existing,
		unconditionally-safe TaggedUnion/RCClass-upcast/pointer-cast
		coercions. Every ordinary call site (assignment, call argument,
		return, ...) leaves this at its default True.
		`context`, passed straight through to _coerce_or_check_operand/
		_check_assignable, only affects a real MISMATCH's own failure
		message (see _check_assignable's own doc) - every legitimate
		coercion still applies exactly the same either way. Almost every
		caller leaves this at its default None (a bare "expected X, got Y");
		_stmt_Assign's own match-arm-binding reassignment is the one caller
		that needs it, to name the REAL cause (a reused binding name, not an
		ordinary value mismatch) instead of a message that never explains
		where the "expected" type even came from. '''
		self._current_lineno = getattr( node, 'lineno', self._current_lineno )
		method = getattr( self, f'_expr_{node.__class__.__name__}', None )
		if method is None:
			self.lowering.discovery.fail( f'unsupported expression: {ast.unparse(node)}', node )
		operand = method( node, expected_type )
		return self._coerce_or_check_operand( operand, expected_type, node, strict = strict, context = context )

	def _coerce_or_check_operand( self, operand: ir.Operand, expected_type: Type|None, node: ast.AST, *, strict: bool = True, context: str|None = None ) -> ir.Operand:
		''' the shared post-dispatch tail: given an operand (freshly produced
		by one of the _expr_X dispatch methods above, OR - unlike every
		other caller - already-lowered and handed in directly, with no AST
		node of its own left to re-dispatch) and an expected_type, applies
		every legitimate coercion in turn and, if none apply and `strict`,
		rejects a genuine mismatch. Factored out of _lower_expr (which calls
		this immediately after dispatch, `node` there being the same node
		method() was just given) specifically so _lower_union_receiver_call
		can call this a SECOND time, once per union leaf, against an
		operand it already has - never re-lowering/re-evaluating the
		original argument expression (which would double its side effects
		once per leaf) while still getting the exact same coercion-or-
		rejection treatment an ordinary call argument gets. `context`, if
		given, only affects _check_assignable's own failure message (see
		its own docstring) - it plays no role in which coercion, if any,
		applies. '''
		# post-hoc, not a pre-emptive override of expected_type before
		# dispatch: a node kind that already produces the right union type
		# on its own (an explicit Result.Ok(x) call, a match-narrowed
		# union-typed value, ...) already satisfies operand.type is
		# expected_type and skips this entirely - only a genuine mismatch
		# (a plain leaf value where the union itself was expected) reaches
		# _coerce_into_union, see its own comment
		#
		# unwrap a Specialization first - a GENERIC union return/annotation
		# type (Result[T,E], or any user @union class Foo[T]:) is always a
		# Specialization wrapping the abstract TaggedUnion, never a bare
		# TaggedUnion instance itself, so a plain isinstance check here
		# (unlike is_rc/rc_leaves/is_result_type/_as_specialization elsewhere
		# in this codebase, all of which unwrap first) silently never fired
		# for one - a bare leaf value flowing into a generic union-typed
		# context (e.g. `return some_T` where T|None is declared) never got
		# coerced at all. Monomorphized (not the bare abstract class) once
		# it's actually needed below - the abstract class's own .attributes
		# hold bare TypeVars (T/E), never a real leaf's type, so
		# _coerce_into_union needs the CONCRETE, substituted union to have
		# any chance of matching operand.type - mirrors the identical
		# "monomorphize_class if Specialization else itself" pattern already
		# used elsewhere for this same reason (e.g. _expr_Name's own
		# union-narrowing branch)
		expected_union: TaggedUnion|None = None
		if operand.type is not expected_type: # cheap identity check first - skip monomorphize_class entirely on the common "already matches" path
			if isinstance( expected_type, TaggedUnion ):
				expected_union = expected_type
			elif isinstance( expected_type, Specialization ) and isinstance( expected_type.base, TaggedUnion ):
				expected_union = self.lowering.monomorphize_class( expected_type )
		if expected_union is not None and operand.type is not expected_union:
			# operand being ITSELF union-shaped does NOT automatically rule out
			# coercion - a nominal @union (e.g. HTTPError) is exactly as valid a
			# member of a WIDER union (OSError|HTTPError) as any plain leaf type
			# is. But it doesn't automatically qualify either: operand can ALSO be
			# a genuinely unrelated Result[T,OtherE] flowing into a Result[T,E]-
			# expected context (Result is itself a TaggedUnion-based
			# Specialization) - that shape is handled by a SEPARATE mechanism
			# entirely (_stmt_Return's own _maybe_widen_return_result, emitting
			# ir.WidenResult), which must get the first chance to run, not be
			# preempted by a hard failure here. So this only actually invokes
			# _coerce_into_union (which hard-fails on a genuine mismatch) when
			# operand's whole type already verbatim matches one of expected_
			# union's own leaf attribute types - the same check _coerce_into_
			# union would use to decide to wrap it anyway, just performed as a
			# non-failing probe first so a genuine non-member (e.g. that
			# unrelated Result[T,OtherE]) is left untouched for its own,
			# more-specific caller-side handling instead. Also probes via
			# _is_rcclass_upcast, not just _same_type: a subclass instance
			# (Sub) is just as valid a fit for a leaf declared as its OWN
			# base class (Base) as an exact-type match is - ordinary
			# derived->base substitution works everywhere else a Base-typed
			# parameter appears, so a Base|None-typed one shouldn't reject
			# it just because the match has to happen leaf-by-leaf here
			# instead of directly against expected_type itself.
			if any(
				self.lowering._type_resolver._same_type( attr.type, operand.type )
				or self._is_rcclass_upcast( operand.type, attr.type )
				for attr in expected_union.attributes
			):
				was_fresh = self._cfg.is_fresh_temp( operand )
				pre_coerce = operand
				operand = self._coerce_into_union( operand, expected_union, node )
				if was_fresh:
					# _coerce_into_union's own ctor call unconditionally
					# increfs whatever it wraps (needed when the wrapped
					# value is a BORROWED reference its own caller still
					# needs afterward) - but a value that was already a
					# fresh, solely-owned temp here (a Call/Allocate
					# result) doesn't need that extra reference kept
					# alive too: release it right here, in place, rather
					# than leaving it as a dangling pending-temp
					# obligation for whatever later flush would otherwise
					# decref it unconditionally. Confirmed as a real bug
					# via a ternary branch specifically (ASAN: SEGV
					# reading uninitialized stack memory on the OTHER
					# branch, where this temp was never even created) -
					# every other caller of this shared coercion tail has
					# the identical gap, just silently masked there by
					# dumb luck (a non-branching statement's own
					# unconditional flush still nets the right refcount
					# when there's no branch boundary for it to straddle).
					for instr in self._cfg.decref( pre_coerce.type, pre_coerce ):
						self._emit( instr )
					self._cfg.untrack_temp( pre_coerce )
			else:
				# operand's WHOLE type isn't one of expected_union's own
				# leaves (the check just above) - but operand can still be
				# ITSELF an anonymous union (e.g. bytes|bytearray) whose
				# INDIVIDUAL leaves are each covered by expected_union's own
				# wider leaf set (e.g. bytes|bytearray|memoryview) - real
				# Python has no static type system to hit this at all, but
				# this codebase's own callers regularly narrow-then-widen
				# (e.g. lib/zipfile.py's own bytes|bytearray-typed parameter
				# flowing into a bytes|bytearray|memoryview-typed one) -
				# confirmed as a real, previously-unsupported gap: silently
				# ACCEPTED at compile time with no coercion at all whenever
				# operand happened to already look union-shaped in some
				# OTHER, unrelated way, or rejected outright otherwise.
				# file is None restricts this to a synthesized ANONYMOUS
				# union specifically (discovery.py's _get_or_create_union) -
				# a NOMINAL `@union class Foo:` stays a single opaque leaf
				# instead (the check above already covers "Foo itself is one
				# of expected_union's own members verbatim"); exploding a
				# nominal union's own internal variants against an unrelated
				# wider union's leaf set would be wrong (same distinction
				# emitter_c.py's own _emit_widen_error already draws for the
				# identical reason, on the Result[T,E] error-widening side).
				operand_shape = self.lowering._type_resolver._tagged_union_shape( operand.type )
				if (
					operand_shape is not None and operand_shape[0].file is None
					and all(
						any(
							self.lowering._type_resolver._same_type( member.type, attr.type )
							or self._is_rcclass_upcast( member.type, attr.type )
							for attr in expected_union.attributes
						)
						for member in operand_shape[1]
					)
				):
					was_fresh = self._cfg.is_fresh_temp( operand )
					pre_coerce = operand
					operand = self._coerce_union_subset( operand, operand_shape, expected_union, node )
					if was_fresh:
						# same "cancel the ctor's own extra incref back out"
						# convention as the plain-leaf coercion case just
						# above - see its own comment for the full reasoning
						# (_coerce_union_subset's per-branch _coerce_into_
						# union call takes the identical incref).
						for instr in self._cfg.decref( pre_coerce.type, pre_coerce ):
							self._emit( instr )
						self._cfg.untrack_temp( pre_coerce )
		# a derived RCClass value flowing into a base-class context (arg, return,
		# assignment) is an upcast: struct Derived* -> struct Base*, which C
		# rejects without an explicit cast. A CastWrap is ordinarily a borrowed
		# reinterpret (its temp is never RC-registered for a borrowed operand -
		# see _lower_bound_method_closure's own note), so this adds no incref/decref,
		# exactly right for passing the SAME object under its base type.
		# Restricted to a genuine strict-subclass relationship so it never masks
		# an unrelated type mismatch.
		elif ( expected_type is not None and operand.type is not expected_type
				and self._is_rcclass_upcast( operand.type, expected_type ) ):
			was_fresh = self._cfg.is_fresh_temp( operand )
			if was_fresh:
				# ownership is moving into the CastWrap's own dest below -
				# untrack the PRE-cast temp here so its own end-of-statement
				# flush doesn't ALSO decref/free the very object the cast
				# just handed off, out from under it. Same "was_fresh" guard
				# the union-coercion branch above already needs, just
				# transferring ownership silently instead of decref'ing
				# (that branch fixes a double-incref/leak; this one fixes
				# the opposite - a temp left registered with nothing left to
				# consume it, later swept as if abandoned). Confirmed via a
				# real compile-and-run use-after-free: `o: Ops = RealOps()`
				# (Ops a base class, RealOps a subclass) segfaulted - the
				# fresh RealOps object was released() immediately after
				# construction, right after being upcast into the base-typed
				# local, while `o` still pointed at the same (now-freed)
				# memory.
				self._cfg.untrack_temp( operand )
			dest = self._new_temp( expected_type )
			self._emit( ir.CastWrap( dest = dest, operand = operand ) )
			if was_fresh:
				# re-register the ownership on `dest` (not `operand`, dropped
				# above) so it isn't just silently dropped when nothing
				# downstream (assign()/field_value()/move()) claims it - e.g.
				# a fresh upcast used directly as an ordinary (non-move[T])
				# call argument, never bound to a name at all (e.g.
				# `handler.setFormatter( TagFormatter() )`). Without this,
				# the single reference from construction ends up owned by
				# nobody: neither the pre-cast temp (untracked above) nor
				# dest (never registered) - a real leak, not just a
				# bookkeeping gap, confirmed via a debug-build live-object
				# report. A downstream consumer that DOES claim it still
				# untracks dest itself (assign()'s is_alias=False branch,
				# field_value(), etc - the same discipline as any other
				# fresh temp), so this is a no-op for that path.
				self._cfg.fresh_temp( dest, expected_type )
			operand = dest
		# a same-signedness, strictly-wider scalar (i32 -> i64, u8 -> u32,
		# f32 -> f64) is a safe, value-preserving widening - allowed
		# implicitly, same CastWrap shape as the RCClass upcast above, but
		# this one is a REAL value conversion (C's own sign-/zero-extension,
		# not a pointer reinterpret) - see _is_safe_scalar_widening's own
		# docstring for exactly which pairs qualify and why isize/usize are
		# deliberately excluded. Gated on `strict` (see _lower_expr's own
		# parameter comment) - _lower_binary_operands' own cross-operand
		# HINTING must never trigger this: `c: f64 = a + b` (a: f64, b: f32)
		# needs to keep hitting _lower_binop_values' own deliberately-
		# stricter "floating-point operation requires both operands to be
		# the SAME type, cast explicitly" rule, not have b silently widened
		# to f64 here first.
		elif ( strict and expected_type is not None and operand.type is not expected_type
				and self._is_safe_scalar_widening( operand.type, expected_type ) ):
			dest = self._new_temp( expected_type )
			self._emit( ir.CastWrap( dest = dest, operand = operand ) )
			operand = dest
		# interchangeable object-pointer types (Ptr[T]/ConstPtr[T], any T/U) -
		# generalized here from _expr_Name/_expr_Attribute's own narrower,
		# node-kind-specific calls to the same helper, so a pointer value
		# reaching this point through ANY expression kind (a Call result, a
		# ternary, ...) gets the identical treatment - _maybe_castwrap_pointer
		# re-checks its own guard and is a no-op once operand.type already
		# matches, so calling it unconditionally here is safe even when
		# _expr_Name/_expr_Attribute already tried it themselves. NOT gated
		# on `strict` - this coercion predates this change and is always
		# safe (any two object pointers share one C representation)
		elif expected_type is not None and operand.type is not expected_type:
			operand = self._maybe_castwrap_pointer( operand, expected_type )
		# the single choke point: every legitimate coercion above already had
		# its chance to rewrite `operand` into something matching expected_type -
		# anything still mismatched past this point is a genuine type error
		# (lowering.py's own longstanding gap: "a genuine argument-type
		# mismatch isn't checked anywhere yet" - see _check_assignable).
		# Skipped when strict=False (see _lower_expr's own `strict` param
		# comment) - a HINT passed down for inference purposes only, not a
		# real requirement the operand must satisfy; the caller (e.g.
		# _lower_binop_values' own float-same-type check) is responsible for
		# validating the ACTUAL requirement itself in that case.
		if strict:
			# case 2 of the general auto-or_throw() rule (see _auto_or_throw's
			# own docstring): a Result[T,E]-shaped operand flowing into a
			# context that wants T directly (not the whole Result) gets
			# implicitly .or_throw()'d, then the UNWRAPPED T is re-run
			# through this same coercion pipeline (it may still need e.g.
			# scalar widening against expected_type, now that its type is T
			# instead of Result[T,E]). Guarded (inside the shared probe) on
			# expected_type itself NOT also being Result-shaped - mirrors
			# _maybe_widen_return_result's own "op_shape/fn_shape both
			# Result" guard - so an explicit `x: Result[T,E] = a + b` (or a
			# Result-typed argument/return) keeps capturing the raw,
			# unconsumed Result exactly as written.
			consumed = self._maybe_auto_consume_result( node, operand, expected_type, self.lowering._AUTO_CONSUME_ALTERNATIVES, context = context )
			if consumed is not None:
				return consumed
			self._check_assignable( operand, expected_type, node, context = context )
		return operand

	def _is_rcclass_upcast( self, sub: Type|None, sup: Type|None ) -> bool:
		''' True if `sub` is a strict subclass (transitively) of `sup`, both being
		RCClasses (or specializations of one) - i.e. a derived->base upcast.

		Deliberately NOT expressed with Type.is_rc()/is_rc_pointer(): this asks
		about INHERITANCE, not reference counting, and it needs the real RCClass
		OBJECT to walk .base with. A tuple[T...] is every bit as much an RC
		pointer as an RCClass but has no inheritance chain at all, so widening
		this guard to is_rc_pointer() would let one into an upcast test it can
		never meaningfully participate in. The isinstance is the right check
		here - see mpy_types.Type's own note on the RC vs layout vs class-kind
		distinction. '''
		def rc_of( t: Type|None ) -> Type|None:
			base = t.base if isinstance( t, Specialization ) else t
			return base if isinstance( base, RCClass ) else None
		sub_c, sup_c = rc_of( sub ), rc_of( sup )
		if sub_c is None or sup_c is None:
			return False
		c = getattr( sub_c, 'base', None )
		while c is not None:
			if c is sup_c:
				return True
			c = getattr( c, 'base', None )
		return False

	# same-signedness scalar widening chains - isize/usize deliberately
	# excluded (their concrete width is target-dependent, see discovery.py's
	# get_intrinsics: sizeof = active_target['bits'] - so treating them as
	# interchangeable with a same-width fixed type would be non-portable
	# across targets; usize(x)/isize(x) stay explicit-cast-only), as are
	# bool/NoneType (never "numeric" for this purpose)
	_SIGNED_INT_WIDENING_ORDER = ( 'i8', 'i16', 'i32', 'i64', 'i128' )
	_UNSIGNED_INT_WIDENING_ORDER = ( 'u8', 'u16', 'u32', 'u64', 'u128' )
	_FLOAT_WIDENING_ORDER = ( 'f32', 'f64' )

	def _is_safe_scalar_widening( self, operand_type: Type|None, expected_type: Type|None ) -> bool:
		''' True if operand_type -> expected_type is a same-family (signed
		int / unsigned int / float), strictly-growing-width scalar
		conversion - i32->i64, u8->u32, f32->f64, etc. Narrowing, sign-
		changing (i32->u32), bool<->numeric, and anything involving isize/
		usize are all deliberately NOT safe widenings here - those require
		an explicit T(x) cast, matching this language's own "no implicit
		conversion" design (SYNTAX.md) everywhere else. `float`/`double` are
		the SAME Scalar object as f32/f64 (see discovery.py's get_intrinsics),
		so comparing by .stem already handles both spellings uniformly. '''
		if not isinstance( operand_type, Scalar ) or not isinstance( expected_type, Scalar ):
			return False
		for order in ( self._SIGNED_INT_WIDENING_ORDER, self._UNSIGNED_INT_WIDENING_ORDER, self._FLOAT_WIDENING_ORDER ):
			if operand_type.stem in order and expected_type.stem in order:
				return order.index( expected_type.stem ) > order.index( operand_type.stem )
		return False

	def _check_assignable( self, operand: ir.Operand, expected_type: Type|None, node: ast.AST, *, context: str|None = None ) -> None:
		''' the single choke point for lowering.py's own longstanding,
		self-documented gap ("a genuine argument-type mismatch isn't
		checked anywhere yet (no general type-checking pass exists)") -
		called last from _coerce_or_check_operand (in turn called from both
		_lower_expr and, a second time per leaf, _lower_union_receiver_call),
		after every legitimate coercion (TaggedUnion wrap, RCClass upcast,
		safe scalar widening, interchangeable pointer cast) already had its
		chance to rewrite `operand` into something matching expected_type.
		`context`, if given, is prefixed onto the failure message - used by
		union-receiver dispatch to name which leaf/parameter disagreed,
		since a bare "expected X, got Y" doesn't otherwise say WHICH of
		several call targets is the one that actually declared X. Uses
		_same_type,
		not raw `is`, for the equality check: a bare Specialization and its
		own already-monomorphized form (or a bare TupleType and its own
		resolved backing RCClass) are the SAME type reached through two
		different representations, not a real conflict - see
		TypeResolver._same_type's own docstring. Also exempts two shapes
		_expr_Constant's own pre-existing literal-kind check already treats
		as non-mismatches for the identical reason (same precedent, applied
		here too since this is the general case, not just the literal one):
		a still-unbound generic TypeVar as expected_type (e.g.
		_lower_allocate_fields lowering a generic no-__init__ field's own
		value against its still-abstract, unsubstituted field.type - nothing
		concrete exists yet to validate against, T is what's BEING inferred,
		not what to check the value against), and a CEnum on either side
		matched against its own declared underlying value_type (e.g.
		`x: builtins.OSError = ...; return x` from a function declared -> u32
		- "a CEnum has exactly the same runtime representation as its
		underlying type", per _try_lower_construct_call's own CEnum-
		construction comment, so the two are interchangeable both directions,
		not just at construction time). Also unwraps move[T]/copy[T] on the
		expected side: both are pure compile-time annotation wrappers (see
		Move/Copy's own docstrings) - a real call-site VALUE is always
		plain T, never literally typed as Move[T]/Copy[T] itself (that
		ownership transfer is tracked separately, by _apply_move_hook/cfg.py,
		not by the value's own .type). '''
		if expected_type is None or operand.type is None:
			return
		if self.lowering._type_resolver._same_type( operand.type, expected_type ):
			return
		if isinstance( expected_type, TypeVar ):
			return
		if isinstance( expected_type, CEnum ) and operand.type is expected_type.value_type:
			return
		if isinstance( operand.type, CEnum ) and expected_type is operand.type.value_type:
			return
		if isinstance( expected_type, ( Move, Copy )):
			self._check_assignable( operand, expected_type.inner, node, context = context )
			return
		prefix = f'{context}: ' if context is not None else ''
		self.lowering.discovery.fail(
			f'{prefix}{ast.unparse(node)}: expected {expected_type.qualname}, got {operand.type.qualname} - '
			f'these are different types; convert explicitly if this is intentional '
			f'(e.g. {expected_type.stem}(...) for a scalar target)',
			node,
		)

	def _maybe_castwrap_pointer( self, operand: ir.Operand, expected_type: Type|None ) -> ir.Operand:
		''' When a pointer-typed value flows into a context expecting a DIFFERENT
		pointer type (e.g. self.__metadata: Ptr[_FastListMetadata] passed to
		sys.memcpy's src: ConstPtr[u8]), insert a CastWrap - in C all object
		pointers share one representation, so this is safe, and without it GCC/
		clang reject the call as -Wincompatible-pointer-types. Function pointers
		(Ptr[Callable[...]]) are deliberately excluded: casting them interferes
		with generic-parameter inference through callable arguments, and object<->
		function pointer casts are not the interchangeable case above. This is the
		same coercion _expr_Name applies to bare locals, factored out so the
		attribute path (self.field reads) gets it too. '''
		tr = self.lowering._type_resolver
		if ( expected_type is not None and operand.type is not expected_type
				and tr._is_ptr_specialization( operand.type ) and tr._is_ptr_specialization( expected_type )
				and tr._callable_type_of( operand.type ) is None and tr._callable_type_of( expected_type ) is None ):
			dest = self._new_temp( expected_type )
			self._emit( ir.CastWrap( dest = dest, operand = operand ) )
			return dest
		return operand

	def _coerce_into_union( self, operand: ir.Operand, union: TaggedUnion, node: ast.AST ) -> ir.Operand:
		# operand's own type doesn't match the union it needs to become -
		# TODO.txt's own documented "opportunistic union emission" gap
		# (a: IntStr = 'foo' should become a = IntStr.v_str('foo')). If
		# operand.type is exactly one of the union's own leaves, wrap it
		# through that leaf's own UnionStorage-synthesized member
		# constructor - the SAME Function a real, explicit Result.Ok(x)
		# call already resolves to and calls via ordinary call resolution;
		# this just does that implicitly. A genuine mismatch (operand's
		# type isn't a member of the union at all) is a real compile
		# error, not silently passed through.
		self.lowering._union_storage.get( union ) # ensures union.names[leaf.stem] exists
		# _same_type, not raw `is` - a leaf's declared type (e.g. list[Op]
		# substituted into a generic union's own attributes) and operand's own
		# type can be two different Specialization objects for the identical
		# instantiation (one already-monomorphized, one freshly built from an
		# annotation) - see TypeResolver._same_type's own docstring, the exact
		# same duality _unify_type_param/_check_assignable already guard
		# against elsewhere. Without this, a bare `list[Op]` return against a
		# declared `list[Op]|None` return type wrongly fell through to the
		# "not one of its members" failure below.
		leaf = next( ( attr for attr in union.attributes if self.lowering._type_resolver._same_type( attr.type, operand.type ) ), None )
		if leaf is None:
			# no leaf matches operand's own type exactly - but a derived RCClass
			# (Sub) flowing into a leaf declared as one of ITS OWN base classes
			# (Base) is still a legitimate member, same as passing Sub() directly
			# to an ordinary Base-typed parameter works everywhere else. The
			# synthesized ctor below is declared to take exactly `leaf.type`
			# (Base), never whatever subclass actually flowed in, so operand
			# needs the identical CastWrap reinterpretation _coerce_or_check_
			# operand's own RCClass-upcast branch applies for a plain (non-union)
			# target - done here explicitly since this call bypasses that branch
			# entirely (this function's own caller only reaches it after already
			# deciding a coercion applies).
			leaf = next( ( attr for attr in union.attributes if self._is_rcclass_upcast( operand.type, attr.type ) ), None )
			if leaf is not None:
				cast_dest = self._new_temp( leaf.type )
				self._emit( ir.CastWrap( dest = cast_dest, operand = operand ) )
				operand = cast_dest
		if leaf is None:
			self.lowering.discovery.fail(
				f'{ast.unparse(node)}: expected {union.qualname}, got a type that is not one of its members',
				node,
			)
		ctor_fn = union.get_local_or_raise( leaf.stem )
		self.lowering.schedule( ctor_fn )
		self.lowering.schedule( ctor_fn.return_type )
		for p in ctor_fn.parameters or []:
			self.lowering.schedule( p.type )
		dest = self._new_temp( union )
		self._emit( ir.Call( dest = dest, target = ctor_fn, receiver = None, args = [ operand ], kwargs = {} ))
		# see _is_aliasing_expr's own comment on this flag: `dest` is a
		# fresh, already-Increfed Call result (the ctor's own body increfs
		# the leaf it wraps), never still-aliasing whatever `node` (the
		# original, pre-coercion expression) looked like to a caller that
		# only has the ast around, not this operand
		dest.is_union_coerce_result = True
		return dest

	def _coerce_union_subset( self, operand: ir.Operand, operand_shape: 'tuple[TaggedUnion,list[Variable]]', expected_union: TaggedUnion, node: ast.AST ) -> ir.Operand:
		''' operand is ITSELF an anonymous union (e.g. bytes|bytearray) whose
		every leaf also appears among expected_union's own leaves (e.g.
		bytes|bytearray|memoryview) - the general "narrower union flows into
		a wider superset union" coercion, distinct from _coerce_into_union's
		own "operand is a bare leaf, or IS one nominal member verbatim"
		cases just above (neither applies here: operand's WHOLE type isn't
		itself one of expected_union's leaves, its own INDIVIDUAL leaves
		are). Builds a runtime tag dispatch over operand's own members (the
		same GetAttr-tag/Cmp/JumpIfFalse shape _emit_eq_dispatch_tree already
		uses), extracting each member's payload as a bare borrow
		(_extract_union_payload) and re-wrapping it through
		_coerce_into_union AGAINST expected_union - reusing that method's
		own leaf-matching (incl. RCClass-upcast) and ctor-call machinery
		verbatim rather than duplicating it, so a subclass leaf or an
		RC/non-RC payload is handled identically to the ordinary bare-leaf
		coercion path. Every branch merges into one shared dest via
		_flush_branch_temps + Assign, the exact same multi-branch-to-one-
		dest pattern _emit_eq_dispatch_tree's own per-cell handling already
		uses - reused here rather than reinvented since getting temp
		lifetime right across branch boundaries is exactly what that
		pattern was hardened against (see _flush_branch_temps' own
		docstring: an earlier, ad hoc version of this shape crashed reading
		an uninitialized branch's own garbage tag/payload). RC ownership:
		operand itself is only ever READ here (bare GetAttr extraction,
		never decref'd/moved) - _coerce_into_union's own ctor call increfs
		each extracted payload independently, so operand keeps its own
		original reference throughout; the caller (_coerce_or_check_operand)
		already decrefs operand afterward when it was a fresh temp, the
		same "cancel the ctor's extra incref back out" convention the
		ordinary bare-leaf coercion case already relies on. '''
		op_base, op_members = operand_shape
		op_tag_attr, op_data_attr, op_payload_cls, op_tags = self.lowering._union_storage.get( op_base )
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		dest = self._new_temp( expected_union )
		end_label = self._new_label( 'union_widen_end' )
		for i, member in enumerate( op_members ):
			is_last = ( i == len( op_members ) - 1 )
			if not is_last:
				next_label = self._new_label( 'union_widen_next' )
				tag_dest = self._new_temp( op_tag_attr.type )
				self._emit( ir.GetAttr( dest = tag_dest, obj = operand, attr = op_tag_attr.stem ))
				match = self._new_temp( bool_cls )
				self._emit( ir.Cmp( dest = match, op = ir.CmpOp.EQ, left = tag_dest, right = ir.Const( type = op_tag_attr.type, value = op_tags[member.stem] )))
				self._emit( ir.JumpIfFalse( cond = match, target = next_label ))
			cell_start = len( self._pending_temps )
			payload = self._extract_union_payload( operand, op_data_attr, op_payload_cls, member )
			value = self._coerce_into_union( payload, expected_union, node )
			self._flush_branch_temps( cell_start, dest, value )
			self._emit( ir.Assign( dest = dest, src = value ))
			self._emit( ir.Jump( target = end_label ))
			if not is_last:
				self._emit( ir.Label( name = next_label ))
		self._emit( ir.Label( name = end_label ))
		self._cfg.fresh_temp( dest, expected_union )
		return dest

	def _expr_Name( self, node: ast.Name, expected_type: Type|None ) -> ir.Operand:
		name = self.lowering.discovery.find_name( node.id, node )
		# module-level `_x`/`__x` privacy (SYNTAX.md) - covers a bare-name
		# VALUE read (not a call - _resolve_callee's own hook covers that)
		# whose binding reached local scope WITHOUT ever going through
		# visit_ImportFrom's own check (discovery.py): a LOCAL, function-
		# body `from X import _y` (as opposed to a module-level one) is
		# bound directly into fn.names during LOWERING itself, never
		# visiting discovery.py's visit_ImportFrom at all - confirmed via a
		# real repro, lib/sys.py's own exit()'s `from crt import _exit;
		# _exit(code)`. Harmless (not double-erroring) for the ordinary
		# module-level-import case too - same pass/fail result either way,
		# since both checks compare the identical (defining, accessing)
		# module pair.
		if id( name ) not in self._inline_param_alias_ids:
			self.lowering.discovery.check_module_visibility( name, node, self._owning_module )
		if isinstance( name, Function ):
			return self._lower_function_ref( name, node )
		if not isinstance( name, Variable ):
			self.lowering.discovery.fail( f'{node.id!r} is not a value, cannot use it as an expression', node )
		# definite-assignment gate: a local (never a global - those aren't
		# tracked by this function's own _live set at all, see cfg.py's
		# is_live() docstring) that's in scope (fn.names, so find_name above
		# already succeeded) but not provably assigned on every path that
		# reaches here - e.g. a bare `x: T` declaration only assigned inside
		# one if-branch, then read unconditionally after it
		if not name.is_global and not self._cfg.is_live( node.id ):
			self.lowering.discovery.fail( f'{node.id!r} is not initialized on all code branches', node )
		self.lowering._ensure_resolved( name )
		if isinstance( name.type, FixedArrayType ):
			# same restriction as a FixedArrayType FIELD's whole-value read
			# (see _expr_Attribute's identical check, and FixedArrayType's
			# own docstring: a bare C array is never a loadable/assignable
			# VALUE, only element-indexed access or compiler.addrof() work).
			# Callers that legitimately need the array's own storage as an
			# addressable ROOT (indexing, addrof) use
			# _fixed_array_root_operand instead of the ordinary _lower_expr/
			# _expr_Name dispatch, so they never reach this rejection.
			self.lowering.discovery.fail(
				f'{node.id!r}: {name.type.qualname} locals cannot be read as a whole value '
				f'(no element-level array access is implemented)',
				node,
			)
		member = self._cfg.narrowed_member( node.id )
		# _same_type, not raw `is` - same PLAN_COMPILER_BUG_SWEEP.md audit
		# that found the other Shape 1 candidates flagged this escape-hatch
		# comparison too, on the theory that expected_type and name.type
		# could be two different objects for the identical union
		# instantiation. No repro could be constructed for it despite
		# several attempts (generic-substituted vs fresh-annotation
		# parameter types, local-variable-annotation vs fresh-annotation) -
		# unlike the OTHER unconfirmed Shape 1 candidates (gated by a
		# separate, already-known upstream bug), this one looks like it may
		# genuinely be unreachable: _get_or_create_union caches by a
		# qualname-text key (ARCHITECTURE.md), which is insensitive to
		# whether the union's own MEMBER Specializations are identical
		# objects, so two structurally-identical union ANNOTATIONS seem to
		# always land on the same cached union object regardless. Applied
		# anyway for consistency/defense-in-depth - strictly safer than
		# `is` (accepts everything `is` did, plus more), so this can only
		# widen when the escape hatch correctly fires, never narrow it.
		if member is not None and not self.lowering._type_resolver._same_type( expected_type, name.type ):
			# name is currently proven to hold this union member (cfg.py's
			# narrow(), from a `match x: case T(x):` arm reusing x's own
			# name) - read through the union's own payload instead of
			# returning the raw (still union-typed) operand, unless the
			# caller explicitly wants the whole union back (expected_type
			# is name.type exactly - a rare escape hatch, e.g. passing x
			# through to another T|None-typed parameter unchanged). Same
			# GetAttr(data).GetAttr(v_member) shape cfg.py's own
			# _extract_payload/lowering.py's _maybe_unwrap_union_arg
			# already use elsewhere for the identical operation - a pure
			# read, no incref needed here: name itself still owns the
			# whole union unconditionally the entire time, this is just
			# viewing one field of it (same as any other attribute read);
			# only actually ALIASING this returned operand into a NEW
			# binding needs its own incref, and that already happens the
			# ordinary way wherever this operand is next consumed. Same
			# Specialization gap as _stmt_Assign's own narrowing-bind
			# handling above: a parameter's declared type stays a genuine
			# Specialization (T/E still bare TypeVars) - .base alone would
			# hand _union_storage.get() the ABSTRACT, unsubstituted payload
			# shape instead of the real monomorphized one.
			base = self.lowering.monomorphize_class( name.type ) if isinstance( name.type, Specialization ) else name.type
			_tag_attr, data_attr, payload_cls, _tags = self.lowering._union_storage.get( base )
			if name.is_global:
				# PLAN_THREAD_SAFE_SHARED_STATE.md Part A: the "name itself
				# still owns the whole union unconditionally the entire time"
				# reasoning above does NOT hold for a protected global - a
				# concurrent thread can reassign its slot at any point after
				# this extraction, unlike a local/parameter no other thread
				# can touch. Extract AND retain atomically under one lock
				# instead of returning a bare (unowned) view, and hand back
				# an already-fresh, OWNED temp via fresh_temp() - cfg.py's
				# assign() (its own is_alias branch) and
				# _incref_aliasing_return both recognize an already-fresh
				# temp via is_fresh_temp() and skip incref'ing it again,
				# exactly like an ordinary Call/Allocate result. Confirmed
				# necessary via a real crash under concurrent stress: the
				# original "bare view, incref happens later, wherever this
				# operand is next consumed" design left an unprotected
				# window between this extraction and that later incref, in
				# which a concurrent writer could free the very object being
				# extracted.
				self._emit( ir.AcquireGlobalLock( var = name, exclusive = False )) # Cost mitigation #4: a read
				payload_dest = self._new_temp( payload_cls )
				self._emit( ir.GetAttr( dest = payload_dest, obj = name, attr = data_attr.stem ))
				leaf_dest = self._new_temp( member.type )
				self._emit( ir.GetAttr( dest = leaf_dest, obj = payload_dest, attr = f'v_{member.stem}' ))
				self._emit( ir.Incref( value = leaf_dest ))
				self._emit( ir.ReleaseGlobalLock( var = name, exclusive = False ))
				self._cfg.fresh_temp( leaf_dest, member.type )
				return leaf_dest
			payload_dest = self._new_temp( payload_cls )
			self._emit( ir.GetAttr( dest = payload_dest, obj = name, attr = data_attr.stem ))
			leaf_dest = self._new_temp( member.type )
			self._emit( ir.GetAttr( dest = leaf_dest, obj = payload_dest, attr = f'v_{member.stem}' ))
			return leaf_dest
		# a pointer-typed local flowing into a context expecting a
		# differently-typed pointer (e.g. return ptr where ptr: Ptr[u8] but
		# the function returns Ptr[T]) used to get a CastWrap inserted right
		# here - now handled uniformly by _lower_expr's own generalized
		# pointer-coercion branch instead (which correctly excludes function
		# pointers, unlike this now-removed inline copy did), so every
		# expression kind gets identical treatment, not just a bare Name
		return name

	def _expr_NamedExpr( self, node: ast.NamedExpr, expected_type: Type|None ) -> ir.Operand:
		''' walrus (`x := expr`): the same two ast.Name-target branches
		_stmt_Assign uses (reassignment vs first declaration - `target` is
		always a bare ast.Name per Python's own grammar), except this is an
		EXPRESSION, so it hands back the assigned operand as its own value
		instead of emitting a void statement. Doesn't thread expected_type
		into the RHS lowering below - _lower_expr's own wrapper already
		re-applies _coerce_or_check_operand to whatever this returns, so
		outer-context coercion (e.g. `x: i64 = (y := 5)`) happens for free,
		same as every other _expr_* method. All locals here are function-
		scoped unconditionally (not block-scoped), so a walrus-bound name
		stays visible after its enclosing if/while exactly like an ordinary
		preceding assignment would - no special escape-the-block handling
		needed, unlike real Python's own comprehension-scoping nuance
		(moot anyway - this language has no comprehensions). '''
		target = node.target
		assert isinstance( target, ast.Name )
		existing = self._existing_local_or_none( target.id, node, 'cannot assign to it' )
		if existing is None:
			existing = self._existing_loop_carried_or_none( target.id )
		if existing is not None:
			self.lowering._ensure_resolved( existing ) # see _stmt_Assign's identical call for why
			self._cfg.unnarrow( target.id )
			operand = self._lower_expr( node.value, existing.type )
			self._cfg_assign( existing, operand, is_alias = self.lowering._is_aliasing_expr( node.value, operand ), node = node )
			return existing
		var, operand = self._declare_local( target.id, node, lambda expected: self._lower_expr( node.value, expected ))
		is_alias = self.lowering._is_aliasing_expr( node.value, operand )
		self._cfg_assign( var, operand, is_alias = is_alias, node = node )
		return var

	def _expr_Constant( self, node: ast.Constant, expected_type: Type|None ) -> ir.Operand:
		# ArrType[N] field's own supported literal (see FixedArrayType's own
		# docstring): a bare `0` means "zero-fill the whole array" - the one
		# value this construction path knows how to emit (a C11 `{0}`
		# designated-initializer payload, valid only inside the class-body
		# compound-literal construction shape - see emitter_c.py's
		# _emit_const). Handled first/separately since none of the ordinary
		# scalar-literal validation below (int range checks, CEnum duality,
		# ...) applies to this type at all.
		if isinstance( expected_type, FixedArrayType ):
			if type( node.value ) is not int or node.value != 0:
				self.lowering.discovery.fail(
					f'{ast.unparse(node)}: {expected_type.qualname} only supports a 0 (zero-fill) literal here - '
					f'per-element array construction is not implemented',
					node,
				)
			return ir.Const( type = expected_type, value = 0 )
		# a floating-point literal can only be typed as a float. If context
		# hints it toward a non-float scalar (an integer), that's a silent-
		# truncation trap - reject it. Catches `i + 1.5` (the literal hinted to
		# the int operand's type by _lower_binary_operands), `x: i32 = 1.5`, and
		# `i32(1.5)`. A float target (f32/f64), no expected type (defaults f64),
		# and a union expected type all fall through unaffected.
		if isinstance( node.value, float ) and isinstance( expected_type, Scalar ) and not _is_float_scalar( expected_type ):
			self.lowering.discovery.fail(
				f'a floating-point literal cannot be used where {expected_type.qualname} is expected - '
				f'write an explicit cast (e.g. {expected_type.stem}(...)) or use an integer literal: {ast.unparse(node)}',
				node,
			)
		# builtins.int (arbitrary-precision) is the one RCClass a bare int
		# literal sugars into directly - `x: int = 0`/`count: int = 0`
		# (a class attribute default)/`return 0` (a function declared
		# -> int) are all extremely ordinary code, and int's own single-
		# i32-parameter __init__ makes "construct int from this literal"
		# unambiguous - unlike the general "b: Box = 0" RCClass case this
		# method deliberately keeps rejecting below (see the generator_
		# zero_rc_field comment), so this is scoped to int specifically,
		# not a blanket literal-into-any-RCClass relaxation. Also fires
		# when expected_type is None outright (a genuinely bare literal,
		# e.g. `x = 0`/`pos = j` where j turns out to be some other type
		# entirely) - int is the safe, no-silent-narrowing default for a
		# literal with no context at all, not i32 (see the sibling
		# expected_type-is-None branch further down, which still defaults
		# to i32 when the literal is only ambiguous within a TaggedUnion/
		# TypeVar, a narrower case where forcing RCClass int would break
		# ordinary scalar-union code like `x: str|i32 = 5`). Rewrites to
		# an ordinary `int(literal)` construction call and re-dispatches
		# through the general Call path - the SAME thing a user would
		# have to write by hand today, just implicit here. The nested
		# literal argument's own expected_type is i32 (int.__init__'s
		# declared parameter type, a Scalar), not this method's own
		# int-RCClass expected_type, so it falls through the ordinary
		# scalar-literal path below unaffected on its own re-entry - no
		# risk of this branch firing twice for the same value.
		if type( node.value ) is int:
			int_type = self.lowering.discovery.find_name_or_none( 'int' )
			if int_type is not None and ( expected_type is int_type or expected_type is None ):
				# int.__init__ only takes an i32 - fine for the common case
				# (int(literal)), but there's no reason a genuinely bigger
				# literal shouldn't just work too, this being arbitrary-
				# precision int, not a fixed-width scalar with a real range
				# limit. A literal outside i32's own range instead goes
				# through int.from_str(...) on the literal's own decimal
				# text - always succeeds (a Python int's own str() is always
				# a valid decimal string from_str accepts), hence the plain
				# .unwrap() rather than surfacing a Result.
				i32_type = self.lowering.discovery.get_intrinsics()['i32']
				lo, hi = int_stem_range( i32_type )
				if lo <= node.value <= hi:
					call_node = ast.Call( func = ast.Name( id = 'int', ctx = ast.Load() ), args = [ ast.Constant( value = node.value ) ], keywords = [] )
				else:
					call_node = ast.Call(
						func = ast.Attribute(
							value = ast.Call(
								func = ast.Attribute( value = ast.Name( id = 'int', ctx = ast.Load() ), attr = 'from_str', ctx = ast.Load() ),
								args = [ ast.Constant( value = str( node.value )) ], keywords = [],
							),
							attr = 'unwrap', ctx = ast.Load(),
						),
						args = [ ast.Constant( value = 'a compile-time int literal is always well-formed' ) ], keywords = [],
					)
				ast.copy_location( call_node, node )
				ast.fix_missing_locations( call_node )
				return self._lower_expr( call_node, int_type )
		# a literal being lowered against a CONCRETE, non-union expected type -
		# verify the literal's own Python value kind could plausibly represent
		# it at all (the same coarse stem-compatibility _LITERAL_COMPATIBLE_
		# STEMS already uses for overload-argument matching below). Without
		# this, expected_type was blindly trusted for anything other than the
		# None/TaggedUnion cases just below - `return 5` inside a function
		# declared -> str tagged the resulting Const as type=str while its own
		# .value stayed the Python int 5 (invalid/nonsensical downstream:
		# emitter_c.py's _emit_const dispatches purely on c.value's own Python
		# type, producing a bare `5` where a struct builtins$str* was
		# expected - confirmed compiling with zero errors, a real C compiler
		# then rejecting the mismatched return type outright). A union
		# expected_type (possibly Specialization-wrapped, e.g. Result[T,E])
		# is exempted here for the same reason the block below exempts it -
		# a literal is never itself union-shaped, _lower_expr's own post-hoc
		# coercion handles that case separately. A POINTER expected_type
		# (Ptr[T]/ConstPtr[T]) is also exempted - a bare int literal assigned
		# to one is a deliberate, pre-existing, well-defined bit-
		# reinterpretation idiom real WinAPI constants rely on (e.g.
		# lib/windows/kernel32.py's `INVALID_HANDLE_VALUE: HANDLE = -1`),
		# already handled downstream by emitter_c.py's own _emit_const
		# pointer-const branch - not a type error. A bare, still-unbound
		# TypeVar (a generic class's own T, not yet substituted with a
		# concrete type - e.g. a field=value construction argument for a
		# still-generic Box[T]) is exempted too: there's no real type here
		# yet to validate against at all, this literal's own natural type
		# is what will eventually get UNIFIED to solve T, not compared
		# against it. A CEnum expected_type validates against its OWN
		# underlying scalar's stem instead of being exempted outright
		# (cenum_value_type below) - "a CEnum has exactly the same
		# runtime representation as its underlying type" (EnumName(42)'s
		# own construction call, see _try_lower_construct_call's CEnum
		# branch, is exactly what this is for), but that's only true
		# when the literal is actually compatible with the underlying
		# SCALAR - a raw int literal for an i32-backed CEnum, not
		# literally anything. Unconditionally exempting every CEnum
		# expected_type from validation here (the original code) let a
		# kind-mismatched literal (e.g. a string) sail through
		# unchecked, tagging the resulting Const with the CEnum type
		# while its own .value stayed the mismatched Python value -
		# confirmed to crash emitter_c.py's _emit_const with an
		# uncaught Python NotImplementedError instead of a clean
		# CompileError (PLAN_COMPILER_BUG_SWEEP.md)
		cenum_value_type = expected_type.value_type if isinstance( expected_type, CEnum ) else None
		expected_base = expected_type.base if isinstance( expected_type, Specialization ) else expected_type
		if (
			expected_type is not None and not isinstance( expected_base, TaggedUnion )
			and not isinstance( expected_type, TypeVar )
			and not self.lowering._type_resolver._is_ptr_specialization( expected_type )
			# PLAN_GENERATORS.md Phase 5 (roadmap Phase 5) - a compiler-
			# synthesized `0` standing in for "this RC-typed generator
			# field isn't assigned yet" (type_resolver.py's
			# _build_generator_backing_class/_rewrite_generator_
			# constructor - a live-flag-gated field, never read before
			# its own first real assignment, so the actual zero bit
			# pattern here is never observed by anything but the
			# generator's own state/flag-gated destructor deciding NOT
			# to decref it). Deliberately NOT a general "int literal into
			# any RCClass" language relaxation (unlike the pre-existing
			# Ptr[T]/ConstPtr[T] exemption just above, which IS meant for
			# ordinary user code) - gated on this compiler-internal tag
			# only, exactly like resolved_type/resolved_callee elsewhere
			# in this codebase, so ordinary user code still can't write
			# `b: Box = 0` as a novel "null RCClass" idiom
			and not getattr( node, 'generator_zero_rc_field', False )
		):
			compatible_stems = self.lowering._LITERAL_COMPATIBLE_STEMS.get( type( node.value ) )
			expected_stem = cenum_value_type.stem if cenum_value_type is not None else getattr( expected_type, 'stem', None )
			# an int literal implicitly widening into a float scalar (`x: f64
			# = 1`, `f64(1)`) is a pre-existing, legitimate pattern - NOT
			# folded into _LITERAL_COMPATIBLE_STEMS[int] itself, since that
			# dict is shared with _lower_overload_arg's own OVERLOAD
			# resolution, where allowing int literals to also match a float
			# overload would introduce new ambiguity there. The reverse
			# (float literal -> int scalar) stays rejected by the dedicated
			# guard at the top of this method - this is one-directional
			is_int_into_float = type( node.value ) is int and expected_stem in ( 'f32', 'f64' )
			# a None literal is deliberately NOT a key in _LITERAL_COMPATIBLE_
			# STEMS at all - _lower_overload_arg's own identical dict lookup
			# relies on exactly that miss to skip straight to _lower_expr(expr,
			# None) for overload candidate matching (a None literal's "which
			# overload accepts it" question needs real per-candidate nullable-
			# type matching, not this coarse stem-list check) - adding a
			# NoneType entry there would silently break that. Validated here
			# instead, separately: only NoneType itself is a legitimate non-
			# pointer target (Ptr[T]/ConstPtr[T] is already exempted above -
			# the real, common nullable-pointer idiom, e.g. `p: Ptr[u8] = None`)
			if node.value is None and expected_stem != 'NoneType':
				self.lowering.discovery.fail(
					f'{ast.unparse(node)}: None cannot be used where {expected_type.qualname} is expected',
					node,
				)
			if compatible_stems is not None and expected_stem not in compatible_stems and not is_int_into_float:
				kind = type( node.value ).__name__
				article = 'an' if kind[0] in 'aeiou' else 'a'
				self.lowering.discovery.fail(
					f'{ast.unparse(node)}: {article} {kind} literal cannot be used where '
					f'{expected_type.qualname} is expected',
					node,
				)
			# a PLAIN literal (not the argument of an explicit T(...)/
			# compiler.cast(T,...) - see _lower_scalar_cast's own
			# _allow_literal_bit_reinterpret handling, which exempts THAT
			# case as deliberate bit-reinterpretation) flowing into a
			# concrete integer scalar type must actually FIT that type's
			# real range - compatible_stems above only checked the
			# literal's KIND (int vs float/str/...), never its magnitude
			if (
				not self._allow_literal_bit_reinterpret and type( node.value ) is int
				and ( isinstance( expected_type, Scalar ) or cenum_value_type is not None )
				and expected_stem in self.lowering._LITERAL_COMPATIBLE_STEMS[int]
			):
				lo, hi = int_stem_range( cenum_value_type if cenum_value_type is not None else expected_type )
				if not ( lo <= node.value <= hi ):
					self.lowering.discovery.fail(
						f'{node.value} is out of range for {expected_type.qualname} ({lo}..{hi}): {ast.unparse(node)}',
						node,
					)
		# expected_type being a TaggedUnion (e.g. str|None) is treated the
		# same as no expected_type at all: a literal's OWN Python type
		# always determines its natural type (bool/i32(*)/str/NoneType) -
		# blindly typing the Const as the whole union here would be wrong
		# (a literal is never itself union-shaped at the C level), and
		# _lower_expr's own post-hoc coercion (see its comment) is what
		# actually wraps this natural-typed Const into the union afterward.
		# A bare, still-unbound TypeVar is treated the same way, matching
		# the validation exemption above - the literal's own natural type
		# is what UNIFIES to solve T, so tagging the Const with the bare
		# TypeVar itself (leaving it unsubstituted downstream) is wrong.
		# (*) an int literal specifically only reaches here for these two
		# union/TypeVar cases - a truly bare expected_type is None already
		# returned above via the int(literal) construction rewrite.
		if expected_type is None or isinstance( expected_type, TaggedUnion ) or isinstance( expected_type, TypeVar ):
			if isinstance( node.value, bool ):
				expected_type = self.lowering.discovery.get_intrinsics()['bool']
			elif isinstance( node.value, float ):
				# float literals default to f64 when no contextual type is
				# available (bare `x = 3.14`) - matches Python, whose float is
				# 64-bit. Checked before int is unnecessary (float/int are
				# disjoint, unlike bool/int) but placed here for clarity. A
				# hinted literal (`y: f32 = 1.5`) never reaches here with
				# expected_type None, so this is purely the no-context default
				expected_type = self.lowering.discovery.get_intrinsics()['f64']
			elif isinstance( node.value, int ):
				# a genuinely bare literal (expected_type is None outright) is
				# already redirected to int(literal) above, before this block -
				# this only still runs for TaggedUnion/TypeVar, where the
				# literal's natural type has to fit a partially-known shape
				# (e.g. `x: str|i32 = 5`, a generic call unifying a bare `T`)
				# and forcing RCClass int would break that. Also the fallback
				# when builtins.int itself hasn't been discovered yet (early
				# bootstrap / a test stubbing its own minimal `class int:`).
				expected_type = self.lowering.discovery.get_intrinsics()['i32']
			elif isinstance( node.value, str ):
				expected_type = self.lowering.discovery.find_name_or_none( 'str' )
				if expected_type is None:
					self.lowering.discovery.fail(
						f'cannot infer the type of literal {node.value!r} - no str type available ({ast.unparse(node)})',
						node,
					)
			elif isinstance( node.value, bytes ):
				expected_type = self.lowering.discovery.find_name_or_none( 'bytes' )
				if expected_type is None:
					self.lowering.discovery.fail(
						f'cannot infer the type of literal {node.value!r} - no bytes type available ({ast.unparse(node)})',
						node,
					)
			elif node.value is None:
				expected_type = self.lowering.discovery.get_none_type()
			else:
				self.lowering.discovery.fail(
					f'cannot infer the type of literal {node.value!r} - no expected type available from context ({ast.unparse(node)})',
					node,
				)
		return ir.Const( type = expected_type, value = node.value )

	def _const_usize( self, value: int ) -> ir.Const:
		return ir.Const( type = self.lowering.discovery.get_intrinsics()['usize'], value = value )

	def _const_i32( self, value: int ) -> ir.Const:
		return ir.Const( type = self.lowering.discovery.get_intrinsics()['i32'], value = value )

	def _const_bool( self, value: bool ) -> ir.Const:
		return ir.Const( type = self.lowering.discovery.get_intrinsics()['bool'], value = value )

	def _expr_Attribute( self, node: ast.Attribute, expected_type: Type|None ) -> ir.Operand:
		# CEnum member VALUE expressions (OSError.FileNotFoundError used as
		# a runtime value) — the base is a class, not a runtime value, so
		# the normal _lower_expr path would reject it. Walk the namespace
		# chain through .names dicts (type_resolver.py already resolved
		# and scheduled every link) and fold the member to an ir.Const.
		chain = self.lowering.find_name_recursive( node )
		if chain is not None:
			obj, attr = chain
			if isinstance( obj, CEnum ):
				assert obj.resolve is None, (
					f'CEnum {obj.qualname} reached lowering unresolved — '
					f'type_resolver.py visit_Attribute should have resolved it'
				)
				value = obj.members.get( attr )
				if value is not None:
					# tag the Const with the CEnum's own nominal type, not its
					# underlying scalar, whenever the surrounding context already
					# expects exactly that CEnum (e.g. a generic type param
					# already bound to it by the enclosing return-type context -
					# see _unify_type_param, which has no CEnum<->value_type
					# exemption the way _check_assignable does at line ~4169/4171
					# and so would wrongly see this as a conflicting inference).
					# Falls back to the scalar for every other context (None, the
					# raw value_type itself, an unrelated/unbound TypeVar) -
					# _check_assignable's own bidirectional CEnum<->value_type
					# exemption already makes both spellings interchangeable
					# there, so this only changes behavior where the exemption
					# doesn't already exist.
					const_type = obj if expected_type is obj else obj.value_type
					return ir.Const( type = const_type, value = value )
			# scope-like terminal (Module, RCClass, etc.) — look up the
			# final attribute as a value directly, without recursing into
			# _lower_expr (which would fail for `sys` when the base is a
			# Module, since a Module is not a value expression). Covers
			# `sys.stdout`, `sys.free`, `builtins.int`, etc. — any
			# module-level Variable/Function/RCClass reached by dotted name.
			names = getattr( obj, 'names', None )
			if isinstance( names, dict ):
				name_obj = names.get( attr )
				if isinstance( name_obj, Variable ):
					# `Foo.flag` (obj resolved to the CLASS itself) where
					# `flag` is an ordinary per-instance field (is_global is
					# False for anything class-scoped, since scope is the
					# class, not the module - see is_global's assignment)
					# reaches this same names dict, so it must be excluded
					# here rather than at the isinstance(obj, RCClass, ...)
					# check below, which never runs otherwise: an instance
					# field's Variable lives on each OBJECT, never on the
					# class as a standalone value the way a real
					# module-level global does. Give a specific, actionable
					# message instead of falling through to _expr_Name's own
					# generic "'Foo' is not a value" (which still fires
					# below, but only names the CLASS, not the actual
					# attribute the user was trying to reach).
					if name_obj.is_global:
						# module-level `_x`/`__x` privacy (SYNTAX.md)
						if id( name_obj ) not in self._inline_param_alias_ids:
							self.lowering.discovery.check_module_visibility( name_obj, node, self._owning_module )
						self.lowering._ensure_resolved( name_obj )
						return name_obj
			if isinstance( obj, ( RCClass, CStruct, TaggedUnion, CUnion ) ):
				self.lowering.discovery.fail(
					f'{ast.unparse(node)}: class attribute access is not supported - read {attr!r} through an instance instead '
					f'(e.g. some_{obj.stem.lower()}.{attr})',
					node,
				)
		obj = self._lower_expr( node.value, None )
		# worker.run used as a VALUE (no call parens) - _attr_lookup below
		# only ever finds a Variable (a real field); a method is a
		# Function, which it rejects outright ("has no attribute"). Same
		# restrictions _lower_function_ref already enforces for a BARE
		# function reference (no receiver to bind there) minus the
		# receiver-less requirement itself, since THAT'S exactly what a
		# closure is for: a generic/overloaded/static method still has
		# nowhere natural to bind (no single fixed signature, or no
		# receiver at all - closures only wrap a REAL bound instance call)
		method = self.lowering._find_method( obj.type, node.attr )
		if method is None:
			# _find_method itself can't distinguish "no such attribute at
			# all" from "found a real Overload group, silently discarded
			# it" - see its own isinstance(found, Function) check. Re-probe
			# here (identical chain_lookup/names shape) specifically to
			# give a clear diagnostic for the latter case, which
			# _attr_lookup below would otherwise ALSO reject, but with the
			# misleading "no such attribute" message - an overloaded
			# method genuinely has nowhere to bind as a bare value (no
			# single fixed signature to close over), same restriction the
			# comment above already states for generic/overloaded methods;
			# this just reports it accurately instead of via the wrong
			# error text. Purely diagnostic - both paths were already a
			# compile error either way, nothing here changes what compiles
			resolved_owner = self.lowering._ensure_resolved( obj.type )
			if isinstance( resolved_owner, ( CStruct, RCClass )):
				maybe_overload = resolved_owner.chain_lookup( node.attr )
			else:
				owner_names = getattr( resolved_owner, 'names', None )
				maybe_overload = owner_names.get( node.attr ) if isinstance( owner_names, dict ) else None
			if isinstance( maybe_overload, Overload ):
				self.lowering.discovery.fail(
					f'{node.attr!r} is an overloaded method and cannot be referenced as a value - call it directly instead: {ast.unparse(node)}',
					node,
				)
		if (
			isinstance( method, Function ) and method.cls is not None
			and not method.is_static and not method.is_classmethod
			and not method.type_params and not method.is_overload
		):
			if method.is_property:
				# @property - `obj.attr` (no call parens) means "call this
				# zero-arg getter and use its result", not "bind a callable
				# closure to it" (the ordinary method-as-value meaning just
				# below, which a property never uses - there is no coherent
				# "callable referring to this property" the way there is for
				# an ordinary method). _ensure_resolved first - method.
				# return_type is still None until then (same reason the
				# dunder-dispatch BinOp/Compare paths above resolve their own
				# method before ever reading .return_type)
				self.lowering._ensure_resolved( method )
				# _lower_method_call's own `result_type` param TYPES dest
				# directly (`self._new_temp(result_type)`) rather than
				# checking it against anything - every other caller (f-string
				# dunder dispatch) always passes the callee's own known
				# return type, so that's safe there. Passing expected_type
				# straight through here instead was a real, confirmed type-
				# safety bug: `expected_type` is only a HINT from the
				# assignment context (e.g. `regs: list[tuple[i32,i32]] =
				# some_property_returning_list[tuple[isize,isize]]`), not a
				# fact about what the getter actually returns - dest ended up
				# typed (and emitted) as the WRONG C struct, silently
				# accepted by the C compiler as a mismatched pointer
				# assignment, corrupting every read through it. method.
				# return_type is always the real, correct type here; coerce
				# the result against expected_type afterward, the same way
				# an ordinary call's own shared tail does.
				result = self._lower_method_call( obj, node.attr, [], method.return_type, node )
				return self._coerce_or_check_operand( result, expected_type, node )
			return self._lower_bound_method_closure( node, obj, method, expected_type )
		attr_var = self.lowering._attr_lookup( obj.type, node.attr, node )
		self._check_field_visibility( obj.type, attr_var, node.attr, node )
		if isinstance( attr_var.type, FixedArrayType ):
			# a bare C array member isn't assignable via `=` at all (only a
			# whole containing struct/union is, or an explicit memcpy) - see
			# FixedArrayType's own docstring. Reading it out as an ordinary
			# value (`x = f.b`) would need ir.GetAttr's emission to do
			# something other than a plain `dest = (obj).field;` assignment,
			# which isn't implemented (no element-level access exists yet
			# either, the same gap this repo's own bytearray has today) -
			# rejected here with a clear message rather than silently
			# reaching emitter_c.py and producing invalid C.
			self.lowering.discovery.fail(
				f'{ast.unparse(node)}: {attr_var.type.qualname} fields cannot be read as a whole value yet '
				f'(no element-level array access is implemented)',
				node,
			)
		dest = self._new_temp( attr_var.type )
		# mirrors _expr_Name's own narrowed-read rewrite exactly (see its
		# comment for the full reasoning) - checks THIS node's own full
		# chain key (any depth, `self.a.b.c`), not just a single hop; a
		# chain that ISN'T itself narrowed but rests on a narrowed PREFIX
		# (`self.a` narrowed, `.b.c` chained on top) was already resolved
		# correctly by the recursive `obj = self._lower_expr(node.value,
		# ...)` call above, which narrows at whatever shallower level
		# actually matched.
		chain_key = self._attribute_chain_key( node )
		member = self._cfg.narrowed_member( chain_key ) if chain_key is not None else None
		is_narrowed = member is not None and not self.lowering._type_resolver._same_type( expected_type, attr_var.type )
		# PLAN_THREAD_SAFE_SHARED_STATE.md Part B: an RC-typed field read
		# (narrowed or not) needs the SAME "acquire / read+retain / release"
		# protection Part A already gives a global - a bare, unretained
		# GetAttr here is the exact "reader loads a pointer, gets preempted
		# before its own incref, a concurrent writer frees it" race A.3
		# describes, just for a field's OWN storage (obj->field) instead of
		# a global's. `obj` itself is never at risk (this thread already
		# holds its own live reference to obj, unlike a global's slot, which
		# nothing here owns) - only obj's FIELD needs protecting.
		attr_is_rc = bool( cfg.rc_leaves( attr_var.type )) and self._is_real_field_receiver( obj.type )
		if attr_is_rc:
			self._emit( ir.AcquireFieldLock( obj = obj, field = attr_var, exclusive = False )) # Cost mitigation #4: a read
		self._emit( ir.GetAttr( dest = dest, obj = obj, attr = node.attr ))
		if is_narrowed:
			base = self.lowering.monomorphize_class( attr_var.type ) if isinstance( attr_var.type, Specialization ) else attr_var.type
			_tag_attr, data_attr, payload_cls, _tags = self.lowering._union_storage.get( base )
			payload_dest = self._new_temp( payload_cls )
			self._emit( ir.GetAttr( dest = payload_dest, obj = dest, attr = data_attr.stem ))
			leaf_dest = self._new_temp( member.type )
			self._emit( ir.GetAttr( dest = leaf_dest, obj = payload_dest, attr = f'v_{member.stem}' ))
			if attr_is_rc:
				self._emit( ir.Incref( value = leaf_dest ))
				self._emit( ir.ReleaseFieldLock( obj = obj, field = attr_var, exclusive = False ))
				self._cfg.fresh_temp( leaf_dest, member.type )
			return leaf_dest
		if attr_is_rc:
			for instr in self._cfg.incref( attr_var.type, dest ):
				self._emit( instr )
			self._emit( ir.ReleaseFieldLock( obj = obj, field = attr_var, exclusive = False ))
			self._cfg.fresh_temp( dest, attr_var.type )
		# a pointer-typed field passed into a differently-typed pointer parameter
		# (e.g. sys.memcpy( ..., self.__metadata, ... ) where src is ConstPtr[u8])
		# needs the same CastWrap coercion _expr_Name does for bare locals.
		return self._maybe_castwrap_pointer( dest, expected_type )

	def _expr_Tuple( self, node: ast.Tuple, expected_type: Type|None ) -> ir.Operand:
		# `(a, b, c)` in value position (PLAN_TUPLE.md) - the first real
		# handling of ast.Tuple as a VALUE anywhere in this file (elsewhere
		# it only ever appears as an annotation-subscript shape, e.g. Dict
		# [K,V]'s own multi-arg slice). No synthesized __init__/construct-
		# call round-trip needed: this builds the backing RCClass directly
		# via ir.Allocate's own field=value shape, the exact same "no real
		# __init__" convention _lower_allocate_fields already applies to any
		# class that doesn't declare one, and the same direct-Allocate shape
		# _lower_bound_method_closure already uses to build a ClosureType
		# value with no __init__ of its own either.
		# arity 0 (`()`) and arity 1 (`(x,)`) are both unambiguous at the AST
		# level - Python's own parser never confuses either with a plain
		# parenthesized expression (`(x)` never becomes an ast.Tuple at all;
		# only a genuine trailing comma or empty parens do) - so both are
		# handled the same way as any other arity here, no special-casing
		# needed past this point.
		#
		# per-element expected types, threaded down the same way
		# _lower_allocate_fields threads field.type into each field's own
		# _lower_expr call - without this, a leaf value destined for a
		# union-typed element (e.g. `bytes` into a declared
		# tuple[bytes|None, str|None]) never goes through
		# _coerce_or_check_operand's union-coercion, AND the tuple type
		# inferred below from the elements' own NATURAL (uncoerced) types
		# would differ from expected_type - two distinct backing RCClasses
		# for what's supposed to be one tuple type, with only the natural
		# one's allocator actually scheduled (_schedule_rcclass_
		# construction below) while dest ends up typed as the OTHER
		# (expected) one - exactly the "call to undeclared function"/
		# "assigning incompatible type" emitter bug this comment is here
		# to prevent regressing. Only applied when arity matches - a
		# genuine arity mismatch is a real type error better left to
		# whatever assignment/return-type check already reports it
		# clearly, not guessed at here.
		#
		# expected_type isn't always the bare TupleType itself - a generic
		# call's own type-param inference (monomorphize.py's substitute_
		# type_params) eagerly resolves a TupleType bound to a TypeVar into
		# its backing RCClass before handing it down as an expected-type
		# hint (Result.Ok((a, b)) against a declared Result[tuple[T1|None,
		# T2|None],E] return type reaches here with expected_type already
		# the tuple's backing RCClass, not the bare TupleType) - falling
		# back to tuple_type_for's reverse lookup recovers the original
		# elem_types (with their union members) in that case too, instead
		# of silently skipping per-element coercion and inferring the
		# tuple's own NATURAL (non-union) type, which then disagrees with
		# the outer expected type and fails generic inference.
		expected_tuple_type = (
			expected_type if isinstance( expected_type, TupleType )
			else self.lowering._tuple_storage.tuple_type_for( expected_type )
		)
		expected_elem_types: list[Type]|None = (
			expected_tuple_type.elem_types
			if expected_tuple_type is not None and len( expected_tuple_type.elem_types ) == len( node.elts )
			else None
		)
		operands: list[ir.Operand] = []
		for i, elt in enumerate( node.elts ):
			elem_expected = expected_elem_types[i] if expected_elem_types is not None else None
			value = self._lower_expr( elt, elem_expected )
			# same per-field RC-retain emission _lower_allocate_fields's own
			# field-value loop uses for every other class's field=value
			# construction sugar - a fresh value (Allocate/Call/Constant)
			# needs no extra incref, an aliasing read of an existing
			# binding (Name/Attribute) does, since the tuple now
			# independently owns a reference alongside whatever binding the
			# element came from
			for instr in self._cfg.field_value( value.type, value, is_alias = self.lowering._is_aliasing_expr( elt, value )):
				self._emit( instr )
			operands.append( value )
		tt = self.lowering.discovery._get_or_create_tuple_type( [ op.type for op in operands ] )
		backing_cls = self.lowering._ensure_resolved( tt )
		# every constructed RCClass needs sys.alloc[T]/sys.free (and __del__,
		# if declared - not applicable here) scheduled at the CONSTRUCTION
		# site, same as every other ir.Allocate emission in this file -
		# _ensure_resolved(tt) alone only guarantees the backing class
		# itself is scheduled, not its allocator
		self.lowering._schedule_rcclass_construction( backing_cls, backing_cls )
		fields = { f'_{i}': op for i, op in enumerate( operands ) }
		# resolve expected_type too, not just tt - an annotated declaration
		# (`x: tuple[i32,str] = (1,"a")`) hands down the SAME bare,
		# unresolved TupleType discovery.py's visit_Subscript produced for
		# the annotation (interned - same object as tt above), not yet
		# swapped for backing_cls. Only trust it when it actually resolves to
		# THIS tuple's own backing class though - expected_type can just as
		# easily be an unrelated OUTER context (e.g. a `tuple[T,T]|None`
		# parameter's own union, handed down so _coerce_or_check_operand can
		# wrap the result into it afterward) rather than a description of the
		# tuple itself; interning guarantees identity in the genuine-match
		# case, so anything else must fall back to backing_cls, not be
		# trusted as dest's real type (a real bug: a tuple literal passed as
		# a `tuple[str,str]|None` argument used to set dest.type to the
		# UNION's own TaggedUnion, which isn't an RCClass, crashing emitter_c
		# .py's Allocate emission with `assert isinstance(concrete_cls,
		# RCClass)` since the union coercion never got a chance to run).
		resolved_expected = self.lowering._ensure_resolved( expected_type ) if expected_type is not None else None
		dest = self._new_temp( resolved_expected if resolved_expected is backing_cls else backing_cls )
		self._emit( ir.Allocate( dest = dest, cls = backing_cls, fields = fields ))
		return dest

	def _narrow_generic_container_expected_type( self, expected_type: Type|None, stem: str, node: ast.AST ) -> Type|None:
		''' expected_type itself if it already resolves to a stem[T]
		Specialization; otherwise, when expected_type is (or wraps) a
		TaggedUnion, that union's own stem[T] member - IF exactly one
		exists. Mirrors _coerce_or_check_operand's own union-unwrapping
		(monomorphize_class + "Specialization wrapping a TaggedUnion" check)
		so a list/set literal targeting a nullable/union-typed context
		(`list[T]|None`) resolves the same way an ordinary call argument
		already does, instead of every such literal needing an intermediate
		typed local first. Ambiguous (0 or 2+ matching members) is left
		alone, not guessed at - same "don't guess" precedent _expr_List/
		_expr_Set already document for their own element-type inference. '''
		spec = self.lowering._type_resolver._as_specialization( expected_type )
		if spec is not None and spec.base.stem == stem and len( spec.args ) == 1:
			return expected_type
		union: TaggedUnion|None = None
		if isinstance( expected_type, TaggedUnion ):
			union = expected_type
		elif isinstance( expected_type, Specialization ) and isinstance( expected_type.base, TaggedUnion ):
			union = self.lowering.monomorphize_class( expected_type )
		if union is None:
			return None
		matches = [
			leaf for leaf in union.leaves()
			if ( leaf_spec := self.lowering._type_resolver._as_specialization( leaf )) is not None
			and leaf_spec.base.stem == stem and len( leaf_spec.args ) == 1
		]
		return matches[0] if len( matches ) == 1 else None

	def _expr_List( self, node: ast.List, expected_type: Type|None ) -> ir.Operand:
		''' [a, b, c] - requires expected_type to already be a concrete
		list[T] Specialization (inferring T from the elements themselves
		when no annotation/return-type is available is deferred - every
		real site in lib/ already has one, matching _expr_Tuple's own
		precedent of deferring an unforced generalization (arity 0/1)
		rather than guessing). Builds one list[T] instance via
		_construct_generic_instance, then a real append(elt).unwrap(...)
		method-call chain per element - list[T] has a real __init__/append,
		unlike tuple, so this can't reuse _expr_Tuple's single-ir.Allocate
		shape. A wrong-typed element is rejected the ordinary way by the
		_lower_expr(elt, elem_type) call below - the general assignability
		check already covers it, nothing extra needed here. '''
		# _as_specialization, not a bare isinstance(expected_type,
		# Specialization) check - expected_type may already have been
		# eagerly monomorphized to the real list[T] RCClass by
		# TypeResolver.resolve_declared_types (a function's own declared
		# return type, a variable's own annotation, ...) by the time this
		# runs, same duality _same_type exists to handle elsewhere -
		# without this, a perfectly valid list[T]-typed context wrongly
		# fails with "needs a known list[T] target type"
		spec = self.lowering._type_resolver._as_specialization( expected_type )
		resolved = self.lowering._ensure_resolved( expected_type ) if expected_type is not None else None
		if not ( spec is not None and isinstance( resolved, RCClass )
				and spec.base.stem == 'list' and len( spec.args ) == 1 ):
			# expected_type may be a union NAMING list[T] as one of its
			# members (e.g. a `list[T]|None` parameter) rather than
			# resolving to list[T] directly - narrow to that member so the
			# literal builds as a plain list[T] here; _coerce_or_check_
			# operand's own post-dispatch tail (in _lower_expr, using the
			# ORIGINAL union expected_type, not this narrowed local) already
			# knows how to wrap a matching leaf value into the wider union
			# afterward, so no wrapping is needed here
			narrowed = self._narrow_generic_container_expected_type( expected_type, 'list', node )
			if narrowed is not None:
				expected_type = narrowed
				spec = self.lowering._type_resolver._as_specialization( expected_type )
				resolved = self.lowering._ensure_resolved( expected_type )
		if not ( spec is not None and isinstance( resolved, RCClass )
				and spec.base.stem == 'list' and len( spec.args ) == 1 ):
			self.lowering.discovery.fail(
				f'list literal needs a known list[T] target type from context (e.g. an annotation or return type): {ast.unparse(node)}',
				node,
			)
		elem_type = spec.args[0]
		dest = self._construct_generic_instance( expected_type, node )
		if not node.elts:
			return dest
		append_fn = self.lowering._find_method( dest.type, 'append' )
		assert append_fn is not None, 'internal compiler error: list[T] has no append method'
		self.lowering._ensure_resolved( append_fn )
		self.lowering.schedule( append_fn.return_type )
		for elt in node.elts:
			operand = self._lower_expr( elt, elem_type )
			# dest=None: append()'s return value (None) is never read, only
			# its side effect - mirrors _expr_Set's own add() handling below
			self._emit( ir.Call( dest = None, target = append_fn, receiver = dest, args = [ operand ], kwargs = {} ))
		return dest

	def _expr_Set( self, node: ast.Set, expected_type: Type|None ) -> ir.Operand:
		''' {a, b, c} - mirrors _expr_List's own shape (requires expected_type
		to already be a concrete set[T] Specialization - element-driven
		inference deferred, same precedent as list/tuple literals above).
		Builds one set[T] instance via _construct_generic_instance, then a
		real add(elt) call per element - unlike list[T].append, set[T].add
		returns plain None (no Result[None,OverflowError] to unwrap), so
		this skips _expr_List's errmsg/unwrap dance entirely. '''
		# _as_specialization, not a bare isinstance(expected_type,
		# Specialization) check - see _expr_List's own identical comment
		spec = self.lowering._type_resolver._as_specialization( expected_type )
		resolved = self.lowering._ensure_resolved( expected_type ) if expected_type is not None else None
		if not ( spec is not None and isinstance( resolved, RCClass )
				and spec.base.stem == 'set' and len( spec.args ) == 1 ):
			# see _expr_List's own identical union-narrowing comment
			narrowed = self._narrow_generic_container_expected_type( expected_type, 'set', node )
			if narrowed is not None:
				expected_type = narrowed
				spec = self.lowering._type_resolver._as_specialization( expected_type )
				resolved = self.lowering._ensure_resolved( expected_type )
		if not ( spec is not None and isinstance( resolved, RCClass )
				and spec.base.stem == 'set' and len( spec.args ) == 1 ):
			self.lowering.discovery.fail(
				f'set literal needs a known set[T] target type from context (e.g. an annotation or return type): {ast.unparse(node)}',
				node,
			)
		elem_type = spec.args[0]
		dest = self._construct_generic_instance( expected_type, node )
		if not node.elts:
			# the standard parser never actually produces an empty ast.Set
			# from source text (`{}` always parses as ast.Dict) - kept for
			# robustness against a synthetically-built empty node, same
			# defensive guard _expr_List keeps for its own analogous case
			return dest
		add_fn = self.lowering._find_method( dest.type, 'add' )
		assert add_fn is not None, 'internal compiler error: set[T] has no add method'
		self.lowering._ensure_resolved( add_fn )
		self.lowering.schedule( add_fn.return_type )
		for elt in node.elts:
			operand = self._lower_expr( elt, elem_type )
			# dest=None: add()'s return value (None) is never read, only its
			# side effect - same "dest=None for a call whose result isn't
			# used" convention _expr_List's own unwrap() call above relies on
			self._emit( ir.Call( dest = None, target = add_fn, receiver = dest, args = [ operand ], kwargs = {} ))
		return dest

	# every ordinary signed-int literal default (i32) and the usual
	# usize-returning sources (len(), .find(), another index computation)
	# a slice bound sees in practice - isize itself passes through
	# _lower_slice_bound's own `operand.type is isize_cls` short-circuit,
	# never reaching this set
	_SLICE_BOUND_WIDENABLE_STEMS = ( 'usize', 'i8', 'i16', 'i32', 'i64' )

	def _lower_slice_bound( self, expr: ast.expr, isize_cls: Type ) -> ir.Operand:
		''' one slice bound (x[EXPR:...] / x[...:EXPR]) - isize, but a plain
		usize expression (overwhelmingly the common case: len(), .find(),
		another slice/index computation, ...), or an ordinary signed-int
		literal (which defaults to i32, not isize - see _expr_Constant's own
		literal-inference rule), is widened to isize here implicitly, unlike
		everywhere else in this language (SYNTAX.md's "no implicit
		conversion" rule - see _is_safe_scalar_widening's own docstring for
		why isize/usize are deliberately EXCLUDED from the GENERAL safe-
		widening set: a usize value near u64::MAX would silently corrupt
		going through isize). A slice bound is different: real container/
		string lengths never approach that range in practice, and requiring
		every existing `s[nbytes:]`/`s[idx+9:]` call site (nbytes/idx already
		usize, e.g. from len()/.find()) to write `s[isize(nbytes):]` by hand
		would make ordinary, positive-only slicing needlessly verbose just to
		support the new negative-index case.

		Lowered with expected_type=None (natural type), NOT isize_cls -
		passing isize_cls through would make _lower_expr correctly infer a
		bare literal as isize, but it ALSO propagates into any @inline
		splice reached along the way (e.g. bare `len(t)` - len[T] is
		@inline), whose own internal return-expression lowering always
		uses strict=True regardless of what strict this call passed,
		rejecting t.__len__()'s natural usize return against isize before
		control ever returns here - confirmed via a real repro
		(mmap.mmap's own len(t) inside a slice bound). Lowering as the
		operand's own natural type sidesteps that entirely; the widening
		below (a manual CastWrap, not the general safe-widening mechanism -
		isize/usize aren't members of _SIGNED_INT_WIDENING_ORDER at all) is
		this method's own private coercion rule, scoped to slice bounds
		only. '''
		if isinstance( expr, ast.Constant ) and type( expr.value ) is int:
			# a bare literal bound (s[0:5]) has no @inline-splice hazard to
			# dodge (that only applies to a Call expression like len(t)) -
			# lower it directly against isize_cls so it doesn't fall through
			# to the bare-literal default (builtins.int, which is unrelated
			# to isize and isn't in _SLICE_BOUND_WIDENABLE_STEMS)
			return self._lower_expr( expr, isize_cls )
		operand = self._lower_expr( expr, None )
		if operand.type is isize_cls:
			return operand
		if isinstance( operand.type, Scalar ) and operand.type.stem in self._SLICE_BOUND_WIDENABLE_STEMS:
			dest = self._new_temp( isize_cls )
			self._emit( ir.CastWrap( dest = dest, operand = operand ) )
			return dest
		return self._coerce_or_check_operand( operand, isize_cls, expr )

	def _lower_slice_subscript( self, node: ast.Subscript, obj: ir.Operand ) -> ir.Operand:
		''' x[a:b] / x[:b] / x[a:] - dispatches through an ordinary
		__getitem__(slice) overload (slice: a start/stop range
		descriptor, lib/builtins/__init__.py), resolved via
		_find_dunder_for_arg - the caller here already knows the exact arg
		type (slice), exactly that helper's designed use case, unlike the
		plain-index path's _find_indexlike_getitem. Replaces the old
		hardcoded 3-type (str/bytearray/memoryview) own _byte_slice/
		_SLICE_LENGTH_METHOD dispatch - any type declaring a
		__getitem__(slice) overload now supports slice syntax generically
		(str/bytearray/memoryview keep byte-offset semantics via their own
		overload bodies; list[T] returns a fresh copy, UnsafeList[T] a
		borrowed slice[T] view - see their own __getitem__(slice)).
		stop's default (omitted upper bound) is left for the CALLEE's own
		overload body to resolve (slice.stop is nullable) rather than
		hardcoded here per type, since only the callee knows the right unit
		(str's real __len__() is a codepoint count, wrong for its own
		byte-offset slicing - see str.byte_len() vs str.__len__()). No
		special RC/aliasing tagging needed (unlike the tuple-index case's
		node.is_tuple_element_read) - this goes through an ordinary
		ir.Call, which the general Call-result convention already treats as
		a fresh, owned value by default. '''
		node_slice = node.slice
		assert isinstance( node_slice, ast.Slice )
		if node_slice.step is not None:
			self.lowering.discovery.fail( f'slice step is not supported: {ast.unparse(node)}', node )
		slice_cls = self.lowering._ensure_resolved( self.lowering.discovery.find_name( 'slice', node ))
		getitem_fn = self._find_dunder_for_arg( obj.type, '__getitem__', slice_cls )
		if getitem_fn is None:
			self.lowering.discovery.fail(
				f'slicing is not supported for {obj.type.qualname if obj.type else "?"} '
				f'(no __getitem__(slice) overload): {ast.unparse(node)}',
				node,
			)
		self.lowering._ensure_resolved( getitem_fn )
		self.lowering.schedule( getitem_fn.return_type )
		isize_cls = self.lowering.discovery.get_intrinsics()['isize']
		start_field = self.lowering._find_field( slice_cls, 'start' )
		stop_field = self.lowering._find_field( slice_cls, 'stop' )
		assert start_field is not None and stop_field is not None, 'internal compiler error: slice missing start/stop fields'
		if node_slice.lower is not None:
			start = self._lower_slice_bound( node_slice.lower, isize_cls )
		else:
			start = ir.Const( type = isize_cls, value = 0 )
		if node_slice.upper is not None:
			# lowered against the concrete leaf type (isize), not the union
			# (stop_field.type) directly - a bare int literal defaults to
			# i32 against a union expected_type (_expr_Constant's own
			# literal-vs-union exemption), which then fails to coerce into
			# isize|None (i32 isn't one of its leaves) - coerce the already
			# isize-typed operand into the union explicitly instead
			stop_value = self._lower_slice_bound( node_slice.upper, isize_cls )
			stop = self._coerce_or_check_operand( stop_value, stop_field.type, node_slice.upper )
		else:
			# no upper bound given - slice.stop is nullable specifically
			# so this "unbounded" state survives all the way into the
			# callee's own overload body, rather than being resolved here
			# against a hardcoded per-type length method
			none_node = ast.Constant( value = None )
			ast.copy_location( none_node, node )
			stop = self._lower_expr( none_node, stop_field.type )
		slice_dest = self._new_temp( slice_cls )
		self._emit( ir.Allocate( dest = slice_dest, cls = slice_cls, fields = { 'start': start, 'stop': stop } ))
		dest = self._new_temp( getitem_fn.return_type )
		self._emit( ir.Call( dest = dest, target = getitem_fn, receiver = obj, args = [ slice_dest ], kwargs = {} ))
		# plain sugar for obj.__getitem__(slice) - see _expr_Subscript's own
		# comment for why this doesn't auto-consume a fallible result
		return dest

	def _lower_tuple_slice( self, node: ast.Subscript, obj: ir.Operand, tuple_type: TupleType ) -> ir.Operand:
		''' t[a:b] on a homogeneous tuple[T,...] - unlike every other slice
		target (_lower_slice_subscript's generic __getitem__(slice) dispatch
		to a runtime method), a tuple's slice RESULT TYPE depends on the
		slice's own bounds (a DIFFERENT fixed-arity tuple[T,...] per distinct
		(start,stop) pair) - no ordinary method could express that with one
		fixed return type, so this requires compile-time-constant bounds and
		builds the result directly via field copies instead, the same "no
		real __init__, just field=value sugar" construction _expr_Tuple's own
		tuple-LITERAL handling already uses for its Allocate. Bounds are
		clamped exactly like every other slice's own out-of-range tolerance
		(_resolve_slice_bounds, lib/builtins/__init__.py) - never a compile
		error, an out-of-range/inverted bound just yields a shorter (possibly
		empty) result, same as real Python. Unlike the runtime slice cstruct
		(start/stop: isize, negative-index-aware), a negative bound here
		isn't supported: `-1` parses as ast.UnaryOp(USub,...), not
		ast.Constant, so it's rejected below as "not constant" - tuple
		slicing needs fully compile-time-constant bounds (the result's own
		TYPE depends on them), and const_bound() only ever unwraps a bare
		ast.Constant, never evaluates a general constant expression. '''
		node_slice = node.slice
		assert isinstance( node_slice, ast.Slice )
		if node_slice.step is not None:
			self.lowering.discovery.fail( f'slice step is not supported: {ast.unparse(node)}', node )
		n = len( tuple_type.elem_types )
		def const_bound( expr: ast.expr|None, default: int ) -> int:
			if expr is None:
				return default
			if not ( isinstance( expr, ast.Constant ) and isinstance( expr.value, int ) and not isinstance( expr.value, bool )):
				self.lowering.discovery.fail(
					f'tuple slicing requires compile-time-constant integer bounds: {ast.unparse(node)}', node,
				)
				return default
			return expr.value
		start = max( 0, min( const_bound( node_slice.lower, 0 ), n ))
		stop = max( 0, min( const_bound( node_slice.upper, n ), n ))
		if start > stop:
			stop = start
		elem_types = tuple_type.elem_types[ start:stop ]
		# arity 0/1 results are real, valid tuple types now (0/1-arity
		# tuple[...] annotations and literals both exist - see discovery.py's
		# visit_Subscript/_expr_Tuple) - no special-casing needed here either
		result_tt = self.lowering.discovery._get_or_create_tuple_type( elem_types )
		backing_cls = self.lowering._ensure_resolved( result_tt )
		self.lowering._schedule_rcclass_construction( backing_cls, backing_cls )
		resolved_obj_type = self.lowering._ensure_resolved( obj.type )
		fields: dict[str,ir.Operand] = {}
		for i, elem_type in enumerate( elem_types ):
			src_field = self.lowering._attr_lookup( resolved_obj_type, f'_{start + i}', node )
			elem_dest = self._new_temp( src_field.type )
			self._emit( ir.GetAttr( dest = elem_dest, obj = obj, attr = f'_{start + i}' ))
			# a GetAttr borrow of the source tuple's own field - genuinely
			# aliasing, same as _expr_Tuple's own per-element field_value call
			# for an aliasing Name/Attribute element
			for instr in self._cfg.field_value( src_field.type, elem_dest, is_alias = True ):
				self._emit( instr )
			fields[ f'_{i}' ] = elem_dest
		dest = self._new_temp( backing_cls )
		self._emit( ir.Allocate( dest = dest, cls = backing_cls, fields = fields ))
		return dest

	def _expr_Subscript( self, node: ast.Subscript, expected_type: Type|None ) -> ir.Operand:
		if isinstance( node.value, ( ast.Attribute, ast.Name )) and not isinstance( node.slice, ast.Slice ):
			fixed = self._fixed_array_index_target( node.value, node.slice )
			if fixed is not None:
				root, attr, array_type, index = fixed
				dest = self._new_temp( array_type.elem_type )
				self._emit( ir.GetAttrIndex( dest = dest, obj = root, attr = attr, index = index ))
				return self._maybe_castwrap_pointer( dest, expected_type )
		obj = self._lower_expr( node.value, None )
		if isinstance( node.slice, ast.Slice ):
			# a tuple's own slice needs its OWN handling (_lower_tuple_slice),
			# not the generic __getitem__(slice) dispatch below - the result
			# type is a DIFFERENT fixed-arity tuple[...] depending on the
			# slice's own compile-time bounds, which no ordinary runtime
			# __getitem__(slice) method could express (its return type is
			# fixed at declaration time) - see _lower_tuple_slice's own
			# docstring
			tuple_type = self.lowering._tuple_storage.tuple_type_for( self.lowering._ensure_resolved( obj.type ))
			if tuple_type is not None:
				return self._lower_tuple_slice( node, obj, tuple_type )
			return self._lower_slice_subscript( node, obj )
		# tuple_type_for checked BEFORE _find_indexlike_getitem, not after:
		# a HOMOGENEOUS tuple's backing class now also declares a real,
		# fallible __getitem__ (Sequence[T] conformance, for min()/iter()/
		# etc. - see tuple_storage.py), so _find_indexlike_getitem would
		# find it too - but t[0] (a compile-time-constant index) should
		# always prefer the cheap, infallible direct-field-access rewrite
		# below over a fallible method call, exactly as it already did for
		# a heterogeneous tuple (which has no __getitem__ at all). Checking
		# tuple_type_for first keeps that behavior unconditional, regardless
		# of whether this particular tuple happens to also conform to
		# Sequence[T] - confirmed necessary by a real repro/regression: `t
		# [0]` on tuple[str,str,str] started requiring the enclosing
		# function to return Result[_,IndexError] the moment homogeneous
		# tuples gained a real __getitem__, breaking many pre-existing
		# lib/ call sites that only ever used constant indices.
		resolved_obj_type = self.lowering._ensure_resolved( obj.type )
		tuple_type = self.lowering._tuple_storage.tuple_type_for( resolved_obj_type )
		getitem_fn = None if tuple_type is not None else self._find_indexlike_getitem( obj.type, node.slice )
		if getitem_fn is None:
			if tuple_type is not None:
				valid_index = (
					isinstance( node.slice, ast.Constant )
					and isinstance( node.slice.value, int )
					and not isinstance( node.slice.value, bool ) # bool is an int subclass in Python's own ast - not a legal tuple index
				)
				if not valid_index:
					self.lowering.discovery.fail(
						f'tuple element access requires a compile-time-constant integer index: {ast.unparse(node)}',
						node,
					)
				index = node.slice.value
				arity = len( tuple_type.elem_types )
				# real Python's own negative-index convention (t[-1] is the
				# last element) - resolved here, at compile time, same as
				# every other bound already is for this direct-field-access
				# rewrite (there's no runtime __getitem__ call here to
				# delegate the wraparound to - see _resolve_index's own
				# comment for the general runtime-container version of this
				# same rule)
				if index < 0:
					index += arity
				if not ( 0 <= index < arity ):
					self.lowering.discovery.fail(
						f'tuple index {node.slice.value} out of range for {resolved_obj_type.qualname} ({-arity}..{arity-1}): {ast.unparse(node)}',
						node,
					)
				attr_var = self.lowering._attr_lookup( resolved_obj_type, f'_{index}', node )
				dest = self._new_temp( attr_var.type )
				self._emit( ir.GetAttr( dest = dest, obj = obj, attr = f'_{index}' ))
				# genuinely aliasing (a GetAttr borrow of the tuple's own
				# field, NOT a fresh Call-owned value) - _is_aliasing_expr's
				# own comment already anticipated exactly this path
				# ("raw ir.GetItem/GetAttr, genuinely aliasing a container
				# element... revisit this once [an indexable container]
				# does [exist]") but was never revisited once tuples landed.
				# Tag the node here (same posture as node.resolved_callee/
				# node.is_narrowing_bind elsewhere) rather than have
				# _is_aliasing_expr re-inspect node.value's own type itself,
				# which is exactly the "risk re-resolving/double-evaluating
				# the receiver" its own comment already rules out - real
				# repro: `q: int = some_tuple[0]` (T RC) then reassigning an
				# existing local to another tuple-index-read, in a loop,
				# with the tuple itself released each iteration, silently
				# skipped this Incref, so the tuple's own destructor's
				# cascading decref of its OWN fields double-released the
				# SAME object the reassigned local still pointed to - a
				# real, confirmed (via generated-C inspection and repeated
				# real-compile-and-run trials) use-after-free, not a
				# hypothetical.
				node.is_tuple_element_read = True
				return dest
			# no real __getitem__ declared (raw pointers, or any other type
			# that doesn't define subscript access as a method) - falls
			# back to the flat GetItem opcode, unconditionally
			if expected_type is None:
				# for Ptr[T]/ConstPtr[T], the pointee type is the natural
				# result of a dereference; for any other type we can't guess
				if isinstance( obj.type, Specialization ) and obj.type.pointer_stem() is not None:
					expected_type = obj.type.args[0]
				else:
					self.lowering.discovery.fail( f'cannot infer the result type of {ast.unparse(node)} - no expected type available from context', node )
			# pointer subscript indices are always usize (pointer arithmetic
			# is defined in terms of the pointer's own element size, not the
			# index's runtime width) — give the index a concrete type so a
			# bare literal 0 in e.g. `ptr[0]` doesn't fail type inference
			index_type = self.lowering.discovery.get_intrinsics()['usize']
			index = self._lower_expr( node.slice, index_type )
			dest = self._new_temp( expected_type )
			self._emit( ir.GetItem( dest = dest, obj = obj, index = index ))
			return dest

		# a real __getitem__ - call it like any other method. `obj[i]` is
		# plain sugar for `obj.__getitem__(i)`, nothing more: if __getitem__
		# is fallible, the caller gets the raw Result[T,E] back and consumes
		# it explicitly (.unwrap()/.or_return()/match), exactly like a
		# fallible `==`/`!=` already does (see _lower_eq_or_ne) - no
		# auto-.or_return() sugar here (that's reserved for checked
		# arithmetic, where formulas get too deep to hand-unwrap every step)
		self.lowering._ensure_resolved( getitem_fn )
		self.lowering.schedule( getitem_fn.return_type )
		index = self._lower_expr( node.slice, getitem_fn.parameters[0].type )
		call_dest = self._new_temp( getitem_fn.return_type )
		self._emit( ir.Call( dest = call_dest, target = getitem_fn, receiver = obj, args = [ index ], kwargs = {} ))
		return call_dest

	def _lower_binary_operands( self, left_node: ast.expr, right_node: ast.expr, expected_type: Type|None, *, infer_right_from_left: bool = True, op: type|None = None ) -> tuple[ir.Operand,ir.Operand]:
		# shared by _expr_BinOp and _expr_Compare: a bare literal constant on
		# either side has no type of its own to offer, so the non-constant
		# side is lowered first and its own inferred type used as the
		# constant's expected_type instead. `infer_right_from_left` captures
		# the one real difference between the two callers when NEITHER side
		# is constant: _expr_BinOp still hints the right operand with the
		# left operand's own inferred type (expected_type or left.type) - but
		# _expr_Compare's expected_type is the comparison's own result type
		# (bool), unrelated to the operands, and never cross-hints one
		# operand from the other outside the constant branches above
		left_is_const = isinstance( left_node, ast.Constant )
		right_is_const = isinstance( right_node, ast.Constant )
		usize_cls = self.lowering.discovery.get_intrinsics()['usize']
		# expected_type is the OUTER statement's own target (e.g. `r:
		# Result[i32|int,E] = x + y`'s Result[...]) - hinting an OPERAND's
		# own lowering with it directly is wrong whenever expected_type is
		# a union: if the operand's OWN natural type happens to exactly
		# equal one of its leaves (x: i32|int here, matching Result's own
		# Ok leaf exactly), _coerce_or_check_operand's union-wrap coercion
		# silently wraps x into Result.Ok(x) BEFORE _lower_binop_values
		# ever runs - x's type is then Result[...], not i32|int, breaking
		# both ordinary dunder lookup and _lower_binop_dispatch's own
		# union-shape detection. Same bug CLASS _lower_eq_or_ne's own
		# right_hint fix already covers for left.type being a union - this
		# is the expected_type-cascades-from-the-caller counterpart,
		# confirmed via a real repro (`r: Result[i32|int,E] = x + y`,
		# x/y: i32|int). None here lets the operand infer its own natural
		# type instead, exactly as if no hint had been given at all.
		# _tagged_union_shape, not a raw isinstance check - expected_type
		# here can still be a Specialization wrapping a TaggedUnion base
		# (a generic Result[T,E] annotation not yet monomorphized) rather
		# than a bare TaggedUnion already - same Specialization-vs-plain
		# duality _tagged_union_shape already exists to paper over
		# everywhere else in this file
		operand_hint = expected_type if self.lowering._type_resolver._tagged_union_shape( expected_type ) is None else None
		# pointer arithmetic (ptr + offset) is never homogeneous the way
		# ordinary scalar +/- is - hinting the OTHER (non-pointer) side with
		# the pointer's own type here (as every branch below otherwise
		# does) would wrongly propagate into a nested BinOp too (e.g. `ptr +
		# idx * element_size` - the outer Add's own right-hint, if left
		# unguarded, leaks into the INNER Mult's result_type, producing a
		# nonsensical "checked pointer multiply"). usize is the natural
		# offset type - same reasoning _expr_Subscript's own pointer index
		# hint already uses
		# every _lower_expr call below is strict=False: these are all HINTS
		# (helping an untyped literal or a nested generic call infer its own
		# type), never a real requirement the operand must satisfy - the
		# real requirement, if any, is validated separately, either by _lower_
		# binop_values' own explicit checks (e.g. its float-same-type rule)
		# or by the OUTER _lower_expr call that invoked _expr_BinOp/_expr_
		# Compare in the first place, re-checking the whole expression's own
		# RESULT afterward. Without this, e.g. `c: f64 = a + b` (a: f64, b:
		# f32) would have b silently widened to f64 right here, bypassing
		# _lower_binop_values' own deliberately stricter "both operands must
		# already be the SAME float type, cast explicitly" rule entirely.
		if left_is_const and not right_is_const:
			right = self._lower_expr( right_node, operand_hint, strict = False )
			# only hint the literal toward right.type when that's actually a
			# meaningful target for a literal to become (a scalar, or the
			# existing Ptr-offset special case) - hinting toward an arbitrary
			# non-scalar CLASS (e.g. `5 + some_vector`, right.type=Vector) hits
			# _expr_Constant's own literal-compatibility check ("an int literal
			# cannot be used where Vector is expected") before this expression
			# ever reaches _lower_binop_values' own dunder/reflected-dunder
			# dispatch - confirmed via a real repro. Falls back to peeking at
			# right.type's own REFLECTED dunder (e.g. Vector.__radd__(other:
			# i32) for `5 + some_vector`) when op is known and unambiguous -
			# without it, the literal locks in as builtins.int before
			# _find_dunder_for_arg ever runs, missing an i32-typed __radd__
			# outright (see _peek_single_dunder_param_type's own docstring).
			# None (both here and as this whole peek's own miss/ambiguous
			# result) lets the literal infer its own natural type instead,
			# same as it would with no hint at all.
			if self.lowering._type_resolver._is_ptr_specialization( right.type ):
				left_hint = usize_cls
			elif isinstance( right.type, Scalar ):
				left_hint = right.type
			else:
				method_name = _BINOP_DUNDER.get( op ) if op is not None else None
				reflected_name = _REFLECTED_BINOP_DUNDER.get( method_name ) if method_name is not None else None
				left_hint = self._peek_single_dunder_param_type( right.type, reflected_name ) if reflected_name is not None else None
			left = self._lower_expr( left_node, left_hint, strict = False )
		elif right_is_const and not left_is_const:
			left = self._lower_expr( left_node, operand_hint, strict = False )
			if self.lowering._type_resolver._is_ptr_specialization( left.type ):
				right_hint = usize_cls
			elif isinstance( left.type, Scalar ):
				right_hint = left.type
			else:
				# forward dunder this time (e.g. Vector.__add__(other: i32)
				# for `some_vector + 5`) - see the left_is_const branch's
				# identical comment just above
				method_name = _BINOP_DUNDER.get( op ) if op is not None else None
				right_hint = self._peek_single_dunder_param_type( left.type, method_name ) if method_name is not None else None
			right = self._lower_expr( right_node, right_hint, strict = False )
		else:
			left = self._lower_expr( left_node, operand_hint, strict = False )
			if infer_right_from_left:
				if self.lowering._type_resolver._is_ptr_specialization( left.type ):
					right_hint = usize_cls
				elif operand_hint is not None:
					right_hint = operand_hint
				elif self.lowering._type_resolver._tagged_union_shape( left.type ) is not None:
					# same "don't hint an operand's own lowering with a
					# union" principle as operand_hint above, just via a
					# DIFFERENT path: left.type here is the fallback hint
					# for right when nothing else applies, but if left
					# happens to be a union operand (x: Vector|i32),
					# hinting right (a PLAIN Vector-typed operand, no
					# union of its own) with it wrongly wraps right into
					# THAT union too - widening the leaf-pair grid with a
					# cell the source never actually expressed. Confirmed
					# via a real repro (Boxed|int + Boxed wrongly widened
					# right into Boxed|int too, inventing an (int,int)
					# grid cell that doesn't exist in the source at all)
					right_hint = None
				else:
					right_hint = left.type
			else:
				right_hint = operand_hint
			right = self._lower_expr( right_node, right_hint, strict = False )
		return left, right

	def _expr_BinOp( self, node: ast.BinOp, expected_type: Type|None ) -> ir.Operand:
		left, right = self._lower_binary_operands( node.left, node.right, expected_type, op = type( node.op ))
		return self._lower_binop_values( node, left, right, expected_type )

	def _lower_binop_values( self, node: 'ast.BinOp|ast.AugAssign', left: ir.Operand, right: ir.Operand, expected_type: Type|None ) -> ir.Operand:
		# the dunder-dispatch/checked-arithmetic core of _expr_BinOp, split
		# out so _stmt_AugAssign's own Attribute/Subscript-target handling
		# can reuse it with operands it already lowered itself (reading the
		# target's object/index exactly once - see that method's own
		# comment) instead of going through _lower_binary_operands, which
		# always lowers both sides fresh from AST. `node` is only ever
		# read for its `.op` (ast.BinOp and ast.AugAssign both have one)
		# and as an error-reporting location - never for `.left`/`.right`.

		# either operand union-typed (`x: i32|int; y: i32|int; x + y`) - a
		# whole separate dispatch, mirroring _lower_eq_or_ne's identical
		# fork for ==/!=: a union operand has no dunder/scalar-arithmetic
		# shape of its OWN, only its individual LEAVES do, so every
		# (left leaf, right leaf) pairing needs its own classification -
		# see _lower_binop_dispatch's own docstring. Left completely
		# untouched below when neither operand is a union - existing,
		# already-verified dunder/reflected-dunder/scalar-arithmetic
		# tail keeps handling that case exactly as it does today.
		# Result[T,E] is itself a @union, so it would otherwise fall straight
		# into the union-leaf-pair dispatch below and get silently
		# decomposed into per-leaf (T, E) arithmetic instead of erroring -
		# an unconsumed Result operand here almost always means the user
		# forgot to .unwrap()/.or_return() a fallible x[i]/x.method() first
		left = self._reject_unconsumed_result_operand( node, left )
		right = self._reject_unconsumed_result_operand( node, right )

		left_shape = self.lowering._type_resolver._tagged_union_shape( left.type )
		right_shape = self.lowering._type_resolver._tagged_union_shape( right.type )
		if left_shape is not None or right_shape is not None:
			return self._lower_binop_dispatch( node, left, left_shape, right, right_shape )

		# try the dunder method (str.__add__, ...) first on left.type, then -
		# mirroring Python's real protocol - the REFLECTED, differently-named
		# dunder on right.type (str.__radd__, ...) if the forward one isn't
		# applicable. Unlike the old code, this ISN'T gated on left.type
		# being non-Scalar: a Scalar left operand has no dunder of its own to
		# try (skip straight to reflected), but that must NOT also skip
		# giving right.type's own reflected dunder a chance - e.g. `5 +
		# some_vector` needs some_vector's own __radd__(other: i32), the
		# exact same "Scalar operand can't have a forward dunder, but the
		# OTHER side's own dunder is still a real candidate" gap _lower_eq_
		# or_ne's own unconditional-Eq/NotEq-dispatch fix already closed for
		# ==/!=; this is the binop counterpart, with a differently-named
		# reflected method instead of the same name reflected.
		method_name = _BINOP_DUNDER.get( type( node.op ))
		if method_name is not None:
			# NOT gated on isinstance(left.type, Scalar)/isinstance(right.type,
			# Scalar) anymore - _find_dunder_for_arg already resolves a
			# scalar-registered dunder (i32.__add__ = ...; see lib/builtins)
			# identically to a real class's own method (both just read
			# .names - see _find_method's own owner-kind branch). A Scalar
			# with nothing registered under this name just misses, exactly
			# like a class that doesn't define the dunder at all - no special
			# case needed for "this operand has no dunder mechanism".
			for candidate in self._mode_qualified_dunder_names( method_name ):
				method = self._find_dunder_for_arg( left.type, candidate, right.type )
				if method is not None:
					return self._emit_binop_dunder_call( node, method, left, right, expected_type )
			reflected_name = _REFLECTED_BINOP_DUNDER.get( method_name )
			if reflected_name is not None:
				for candidate in self._mode_qualified_dunder_names( reflected_name ):
					reflected_method = self._find_dunder_for_arg( right.type, candidate, left.type )
					if reflected_method is not None:
						return self._emit_binop_dunder_call( node, reflected_method, right, left, expected_type )

		# a float on EITHER side gets two float-specific rejections dunder
		# dispatch above can't produce itself (it only ever MISSES silently,
		# never explains why): strict same-type (a bare literal on either
		# side has already been hinted to the other's type by _lower_binary_
		# operands, so `f + 1.5`/`f + 1` still work; only a float mixed with
		# an int VARIABLE, or two different float widths, reaches this error.
		# NOTE an int VARIABLE combined with a bare float LITERAL (`i + 1.5`)
		# is not caught here - the literal is hinted to the int's type and
		# truncated, an accepted first-pass edge), and bitwise/shift/floordiv/
		# mod, which have no floating-point meaning and no dunder at all.
		# Every remaining same-type float shape (+-*/`/`) already has a real
		# dunder (lib/builtins/__scalar_dunders.py's f_*_checked/wrapped/
		# saturated, __truediv__) and dispatched through it above - nothing
		# legitimate reaches past these two checks.
		if _is_float_scalar( left.type ) or _is_float_scalar( right.type ):
			if left.type is not right.type:
				float_type = left.type if _is_float_scalar( left.type ) else right.type
				self.lowering.discovery.fail(
					f'floating-point operation requires both operands to be the same type - '
					f'got {left.type.qualname if left.type else "?"} and {right.type.qualname if right.type else "?"}; cast one explicitly '
					f'(e.g. {float_type.stem}(x)): {ast.unparse(node)}',
					node,
				)
			bad = _FLOAT_UNSUPPORTED_BINOPS.get( type( node.op ))
			if bad is not None:
				self.lowering.discovery.fail( f'operator {bad!r} is not supported on floating-point values: {ast.unparse(node)}', node )

		# every operator this language actually supports has a dunder mapping
		# in _BINOP_DUNDER above (dispatched through it, or through the float
		# checks just above, before ever reaching here) - what's left is a
		# genuinely unsupported operator (`/` on an int - only `//` exists;
		# `**`/`@`, never mapped to anything at all), OR a supported operator
		# with no dunder registered for THIS particular pair of operand types
		# (e.g. `int + str`) - name both sides' types so it's clear which
		self.lowering.discovery.fail(
			f'unsupported binary operator between {left.type.qualname if left.type else "?"} and '
			f'{right.type.qualname if right.type else "?"}: {ast.unparse(node)}',
			node,
		)

	def _lower_arithmetic_op( self, node: ast.AST, opcode: type|None, extra: ir.Operand|None, result_type: Type, operand_kwargs: dict, kind: str ) -> ir.Operand:
		# shared by _lower_scalar_cast/_expr_BinOp/_expr_UnaryOp - each just
		# resolves its own (opcode, extra) via the active ArithmeticMode's
		# GetCast/GetBinOp/GetUnaryOp and hands them here along with its own
		# operand shape (cast/USub take a single `operand`, BinOp takes
		# `left`/`right`). `kind` is only used for the unsupported-operator
		# message below - _lower_scalar_cast's GetCast() never actually
		# returns None (every mode defines a cast opcode), so that branch is
		# unreachable from there, but harmless to share
		if opcode is None:
			self.lowering.discovery.fail( f'unsupported {kind} operator: {ast.unparse(node)}', node )
		if not opcode.checked_errors:
			# wrap/saturate, or no overflow concept at all (bitwise/Invert)
			dest = self._new_temp( result_type )
			self._emit( opcode( dest = dest, **operand_kwargs ))
			return dest
		# check mode (the default - see the class docstring): the op itself
		# produces Result[result_type,<error_type>], where <error_type> is a
		# single marker class (most ops) or the anonymous UNION of several
		# (signed Div/Mod -> ZeroDivisionError|OverflowError; checked float / ->
		# ZeroDivisionError|FloatingPointError). `with compiler.panic_
		# arithmetic(msg):` (extra is the lowered msg operand) still consumes
		# eagerly via Unwrap, which panics immediately and needs nowhere to
		# propagate to. The default (extra is None) does NOT eagerly consume
		# here anymore - the raw Result just flows out, picked up downstream
		# by the general case-1 (discarded statement)/case-2 (flowing into a
		# T-typed context) auto-or_throw() rule, exactly the same mechanism
		# .or_throw()/or_return() already use - see _auto_or_throw's own
		# docstring for why this is now the ONE place that logic lives.
		result_cls = self.lowering.discovery.find_name( 'Result', node )
		error_type, _alternatives = self._resolve_checked_error( node, opcode, result_type )
		return self._emit_checked_op( node, opcode, operand_kwargs, result_type, result_cls, error_type, extra )

	def _resolve_checked_error( self, node: ast.AST, opcode: type, result_type: Type ) -> tuple[ClassLike,str]:
		# the error TYPE a checked opcode's Result is against, plus the "how to
		# avoid needing to propagate it" message. opcode.checked_errors lists
		# the possible error class names; opcode.signed_only names the ones that
		# only apply to a signed integer result (OverflowError on Div/Mod's
		# INT_MIN/-1) - filtered out for unsigned operands. A single remaining
		# name -> that marker class (unchanged from before); several -> the
		# anonymous union of them (interned by _get_or_create_union, so a user's
		# own `A | B` annotation on the enclosing function's return type is the
		# SAME object - identity holds for _require_result_return's coverage
		# check and for OrReturn's error copy).
		signed = _is_signed_scalar( result_type )
		names = [ e for e in opcode.checked_errors if e not in opcode.signed_only or signed ]
		classes = [ self.lowering.discovery.find_name( name, node ) for name in names ]
		if len( classes ) == 1:
			# single-error path: preserve the exact per-error message (existing
			# tests assert e.g. 'Result[_,ZeroDivisionError]' + 'panic_arithmetic')
			return classes[0], _ALTERNATIVES_BY_ERROR[names[0]]
		error_union = self.lowering.discovery._get_or_create_union( classes )
		# the union's tag/data storage must exist by emit time (OrReturn's
		# widening and the division emitter both read the inner variant tag)
		self.lowering.schedule( error_union )
		self.lowering._union_storage.get( error_union )
		alternatives = (
			f'change the enclosing function to return Result[_,{" | ".join(sorted(c.stem for c in classes))}] '
			'(or a wider union covering those), or wrap this in `with compiler.panic_arithmetic(...):`'
		)
		return error_union, alternatives

	def _emit_checked_op( self, node: ast.AST, opcode: type, operand_kwargs: dict, result_type: Type, result_cls: ClassLike, error_cls: ClassLike, extra: ir.Operand|None ) -> ir.Temp:
		# shared by Check-mode binops (Add/Sub/Mult/Shl/Div/Mod), USub, and
		# scalar casts - operand_kwargs is however the specific opcode names
		# its operand(s) (left/right for a binop, operand for USub/cast)
		check_type = self.lowering.discovery._get_or_create_specialization( result_cls, [ result_type, error_cls ] )
		# the emitter declares a local variable of this Result type; the
		# struct definition must exist even though the Check op's result is
		# often consumed inline (Unwrap in panic mode, or_throw() downstream
		# otherwise) — schedule it now so monomorphize_class emits it into
		# compiler.tagged_unions
		self.lowering.schedule( check_type )
		check_dest = self._new_temp( check_type )
		self._emit( opcode( dest = check_dest, **operand_kwargs ))
		if extra is not None:
			# panic mode - still consumes eagerly, unaffected by the general
			# auto-or_throw() rule (see _lower_arithmetic_op's own comment)
			return self._consume_checked_result( node, check_dest, result_type, extra )
		# Check mode (the default) - the raw, unconsumed Result[result_type,
		# error_cls] flows straight out; case-1/case-2 auto-or_throw() picks
		# it up downstream (see _auto_or_throw)
		return check_dest

	def _consume_checked_result(
		self, node: ast.AST, check_dest: ir.Temp, result_type: Type, extra: ir.Operand|None, *, receiver_pending_start: int|None = None,
	) -> ir.Temp:
		# shared by both binop (AddCheck/.../Div/Mod) and unary (NegCheck)
		# Check-mode ops, _maybe_consume_result's __len__/__getitem__ auto-
		# unwrap, and _lower_or_return's own <result_expr>.or_return() - see
		# _expr_BinOp's own comment on the OrReturn/OrJump/Unwrap split.
		# check_dest is sometimes a real, named Variable (or_return()'s own
		# receiver) and sometimes a bare Temp (checked arithmetic, __len__/
		# __getitem__'s auto-unwrap) - isinstance covers both uniformly
		# the innermost active multi-statement @inline splice, if this
		# early-exit-shaped construct (.or_return(), checked arithmetic
		# under the default Check mode, or the __len__/__getitem__ auto-
		# consume path) is reached from one of a spliced body's own pre-
		# return statements (self._current_fn is briefly the caller during
		# this window too - see _splice_multi_statement_inline_body's own
		# comment). Left unredirected, the OrReturn/OrJump path below would
		# jump to/return from the CALLER's own real epilogue - a real
		# correctness bug (silently skipping the rest of THIS splice AND
		# the caller's own subsequent statements), not just an unsupported
		# case - so both branches below stow into the SPLICE's own result
		# var/exited flag instead of self._return_value_var/a real return
		# whenever this is set. The trailing return-EXPRESSION itself is
		# lowered with this restored to None first, so it's unaffected -
		# nothing of the splice remains after it to skip past there, so
		# jumping to the caller's own epilogue is already correct, exactly
		# as the single-statement case already relies on
		if self._in_inline_splice_prelude and not self._inline_scope_vars:
			# PLAN_RETURN_INFERENCE.md's own @inline variant reached here
			# with target.return_type still the "infer it" sentinel (see
			# _splice_multi_statement_inline_body's own top-of-function
			# comment) - no inline scope exists to redirect into (result_
			# var's type isn't known yet, by construction), so this narrow
			# combination stays rejected, exactly as the single, blanket
			# guard this method used to have always rejected every
			# multi-statement splice's own pre-return statements
			self.lowering.discovery.fail(
				f'@inline: .or_return()/checked arithmetic that could propagate an error is not yet supported before the '
				f'final return of a multi-statement body whose own return type is still being inferred: {ast.unparse(node)}',
				node,
			)
		inline_scope = self._inline_scope_vars[-1] if self._in_inline_splice_prelude and self._inline_scope_vars else None
		# see _flush_new_pending_temps's own docstring: check_dest/extra are
		# the only operands still needed below (by OrReturn/OrJump/Unwrap
		# themselves) - anything else pending SINCE receiver_pending_start
		# (i.e. only from building up check_dest's own receiver expression,
		# e.g. a chained field receiver's own Part B retain - NOT from
		# whatever outer expression check_dest itself is only part of) must
		# be released now, before the Err leg's own early exit can skip
		# right past it. Only the explicit .or_return() syntax currently
		# passes a real receiver_pending_start (see _lower_or_return) -
		# every other caller (checked arithmetic, __len__/__getitem__'s
		# auto-unwrap) leaves it None, unaffected, since their own operands
		# don't go through Part B's field-receiver retain shape in the same
		# way and haven't been confirmed to need this. Placed before
		# unwrapped's own creation so this never touches unwrapped itself
		# (nothing is written into it yet).
		if receiver_pending_start is not None:
			self._flush_new_pending_temps( receiver_pending_start, check_dest, extra )
		unwrapped = self._new_temp( result_type )
		# OrReturn/OrJump/Unwrap hand unwrapped a fresh, solely-owned copy of
		# the Ok payload (moved out of check_dest, no incref) - registering it
		# here, same as an ordinary Call/Allocate result, is what lets a
		# discarded `<result>.or_return()` bare statement (_lower_or_return's
		# own want_result=False path) actually release an RC-leaf payload
		# nobody ever reads, instead of leaking it. A no-op for non-RC
		# result_types. Every real consumer (assign/return_/field_value/
		# untrack_temp) already pops this out of tracking as part of its own
		# ordinary handling, so registering it doesn't disturb the "used
		# further" path.
		self._cfg.fresh_temp( unwrapped, result_type )
		if extra is None:
			if isinstance( check_dest, Variable ):
				self._cfg.clear_result( check_dest.stem ) # this call IS the inspection of check_dest - clear it before the exit-path check below, or it'd wrongly flag itself
			# the OrReturn/OrJump path below is a second, separate function-
			# exit point alongside plain `return` (see cfg.check_unchecked_
			# results' own docstring) - anything else still unchecked here
			# would otherwise be silently discarded exactly like falling off
			# the end unchecked would be
			try:
				self._cfg.check_unchecked_results( None )
			except CompileError as e:
				self.lowering.discovery.fail( str( e ), node )
			# pass check_dest (when it's a named, tracked Variable - e.g.
			# `x.or_return()`, as opposed to a bare Temp from `foo().or_return()`)
			# into current_epilogue_label(), mirroring _stmt_Return's identical
			# call for `return x` - this lets the identity-skip guard recognize
			# "the operand being propagated out IS itself one of the still-live
			# tracked bindings" and route through the safe inline-replay path
			# below instead of jumping into check_dest's OWN epilogue label (which
			# would double-decref its payload right after copying it into the
			# return value, unretained - a real, confirmed use-after-free: check_
			# dest's Err payload gets moved into the returned struct, then the
			# shared epilogue ladder at that SAME label unconditionally decrefs
			# check_dest's own copy of it too). A bare Temp has no epilogue entry
			# of its own, so this is a no-op for that case, matching today's
			# already-correct behavior exactly (identical to passing None).
			tracked_operand = check_dest if isinstance( check_dest, Variable ) else None
			label = self._cfg.current_epilogue_label( tracked_operand )
			if label is not None:
				if inline_scope is not None:
					result_var, exited_flag, _merge_label = inline_scope
					self._emit( ir.OrJump( dest = unwrapped, value = check_dest, target = label, return_slot = result_var, exited_flag = exited_flag ))
				else:
					self._emit( ir.OrJump( dest = unwrapped, value = check_dest, target = label, return_slot = self._return_value_var ))
			else:
				# either check_dest's own entry needed excluding (the bug above),
				# or (matching _stmt_Return's own inline path for the identical
				# reasons - a confined entry, or simply nothing pending) there's no
				# shared label to jump to at all - return_() handles both uniformly:
				# it replays every OTHER still-live binding/defer/errdefer
				# obligation inline, skipping only check_dest's own entry (a no-op
				# skip when check_dest has no entry in the first place). Its
				# DeclareTemp side effects (via the injected new_temp callback)
				# land unconditionally in the main instruction stream right here,
				# same as _stmt_Return's identical call - only the rest (the
				# actual conditional replay logic) is embedded below, to run
				# strictly inside the Err branch
				replay = self._cfg.return_( tracked_operand, lambda: self._build_is_err_check( node ))
				# PLAN_GENERATORS.md's defer/errdefer phase (Mechanism 2) -
				# ir.OrReturn.epilogue is ALREADY Err-branch-exclusive by
				# construction (unlike an ordinary function's own shared
				# epilogue, which is reached by success AND error paths
				# alike, needing its own is_err() guard) - so both `defer`
				# and `errdefer` sites currently armed at this fallible
				# operation's own position just get appended here, no extra
				# guard needed. See _build_generator_error_defer_replay's
				# own docstring for why this is a no-op outside a generator.
				# The pessimistic-done pin runs FIRST, ahead of any defer/
				# errdefer replay - both are no-ops outside a generator,
				# order between them doesn't affect correctness inside one
				# (see _build_generator_pessimistic_done_pin's own docstring)
				replay = replay + self._build_generator_pessimistic_done_pin() + self._build_generator_error_defer_replay()
				if inline_scope is not None:
					# always `goto`s merge_label directly (see emitter_c.py's
					# own ir.OrReturn.inline_exit handling) - same bypass
					# shape _stmt_Return's identical branch marks captured for
					self._cfg.mark_inline_scope_captured()
					self._emit( ir.OrReturn( dest = unwrapped, value = check_dest, epilogue = replay, inline_exit = inline_scope ))
				else:
					self._emit( ir.OrReturn( dest = unwrapped, value = check_dest, epilogue = replay ))
		else:
			panic_fn = self.lowering._type_resolver._resolve_sys_function( 'panic' )
			self.lowering.schedule( panic_fn )
			self._emit( ir.Unwrap( dest = unwrapped, value = check_dest, errmsg = extra, panic = panic_fn ))
		# OrReturn/OrJump/Unwrap all extract the Ok payload as a raw struct-
		# field copy (emitter_c.py's _emit_or_return/_emit_or_jump/ir.Unwrap
		# handling) - a BORROW of check_dest's own payload, not a fresh
		# reference, exactly like Result.unwrap()'s old bare `return
		# self.data.v_Ok` was. check_dest's own payload gets its OWN eventual
		# decref (it's an ordinary tracked temp/binding like any other Result
		# value - see cfg.py's rc_leaves()/lowering.py's CFGState resolve_type
		# wiring for the fix that makes that actually happen now), so without
		# this incref `unwrapped` and check_dest's own decref would fight over
		# the SAME single reference - confirmed with a real UAF repro
		# (int.__floordiv__'s `self.divmod(other).or_return()`, caught by
		# AddressSanitizer). A no-op for a non-RC result_type (plain
		# arithmetic's own scalar Check ops), so safe to call unconditionally
		# regardless of which of this function's three callers reached here.
		for instr in self._cfg.incref( unwrapped.type, unwrapped ):
			self._emit( instr )
		return unwrapped

	def _expr_UnaryOp( self, node: ast.UnaryOp, expected_type: Type|None ) -> ir.Operand:
		# expected_type is the OUTER expression's own target (e.g. `return
		# -i` from a function declared -> i32|None) - hinting node.operand's
		# own lowering with it directly is wrong whenever expected_type is a
		# union: _lower_expr's own union-wrap coercion would silently wrap
		# the OPERAND into Some(i) BEFORE the unary operator ever runs,
		# leaving `-`/`~`/`not` trying to operate on a union value instead
		# of the scalar it actually needs - confirmed via a real repro
		# (`return -i` from a function declared -> i32|None generated
		# invalid C, assigning a bare int into the union struct directly;
		# NegWrap's own dest/operand had both silently become union-typed).
		# Same fix _lower_binary_operands' own operand_hint already applies
		# for +-*/etc - None here lets node.operand infer its own natural
		# type instead, exactly as if no hint had been given at all; the
		# unary operator's own RESULT still gets coerced into expected_type
		# normally, by the ordinary _lower_expr call that invoked this
		# method in the first place.
		operand_hint = expected_type if self.lowering._type_resolver._tagged_union_shape( expected_type ) is None else None
		if isinstance( node.op, ast.Not ):
			# strict=False: expected_type here is only ever a hint for the
			# RARE case node.operand itself still needs inference (an
			# untyped literal/generic call), never a real requirement that
			# x must already BE expected_type's own type
			operand = self._lower_expr( node.operand, operand_hint, strict = False )
			# `not x` needs x's real TRUTHINESS, not a raw `!operand` C
			# negation - those coincide for a Scalar (both are "!= 0"), but
			# NOT for an arbitrary RC/union operand: `!ptr` in C is a null-
			# pointer check, which is always false for any live, non-null
			# object - silently wrong for e.g. `not ''` (an empty-but-non-
			# null str, correctly falsy per _truthiness_of_operand's own
			# str.__bool__ dispatch, but `!ptr` would say truthy since the
			# pointer itself isn't null). _truthiness_of_operand is the
			# single source of truth for what "truthy" means for any type
			# (Scalar/__bool__/union-tag-dispatch/default) - reused here so
			# `not x` can never disagree with `if x:`/BoolOp's own answer.
			truthiness = self._truthiness_of_operand( operand, node )
			# `not x`'s own result is ALWAYS bool, never expected_type itself
			# (which, same reasoning as operand_hint above, might be a union
			# wrapping bool - `return not flag` from a function declared ->
			# bool|None) - dest must stay plain bool here so ir.Not's own
			# `dest = !(operand);` codegen matches its declared C type; the
			# post-dispatch _coerce_or_check_operand tail (_lower_expr's own,
			# back in the caller) is what wraps this plain bool into the
			# union afterward, same as every other _expr_X method relies on
			# it to. Confirmed via a real repro: using expected_type directly
			# here made dest itself union-typed, which _coerce_or_check_
			# operand's own "operand.type is expected_type already" identity
			# check then wrongly treated as "nothing to coerce", leaving a
			# bare `!(flag)` assigned straight into a union struct in C
			bool_cls = self.lowering.discovery.find_name( 'bool', node )
			dest = self._new_temp( bool_cls )
			self._emit( ir.Not( dest = dest, operand = truthiness ))
			return dest
		operand = self._lower_expr( node.operand, operand_hint )

		# non-scalar operand — try the dunder method (int.__neg__, ...),
		# mirroring _expr_BinOp/_expr_Compare's identical dispatch
		if not isinstance( operand.type, Scalar ):
			method_name = _UNARYOP_DUNDER.get( type( node.op ))
			if method_name is not None:
				method = self.lowering._find_method( operand.type, method_name )
				if method is not None:
					self.lowering._ensure_resolved( method )
					self.lowering.schedule( method.return_type )
					for p in ( method.parameters or [] ):
						self.lowering.schedule( p.type )
					dest = self._new_temp( expected_type or method.return_type )
					self._emit( ir.Call( dest = dest, target = method, receiver = operand, args = [], kwargs = {} ))
					return dest

		result_type = expected_type if _is_float_scalar( expected_type ) else operand.type

		# a float operand takes the GetFloatUnaryOp path: unary `-` is plain
		# negation (never faults), `~` falls through to the unsupported-operator
		# error - see ArithmeticMode.GetFloatUnaryOp
		if _is_float_scalar( operand.type ):
			opcode, extra = self._arithmetic_mode[-1].GetFloatUnaryOp( node )
		else:
			opcode, extra = self._arithmetic_mode[-1].GetUnaryOp( node )
		return self._lower_arithmetic_op( node, opcode, extra, result_type, { 'operand': operand }, 'unary' )

	def _boolop_leaf_excluded( self, leaf_type: Type, is_and: bool ) -> bool:
		''' True if leaf_type can PROVABLY never be the value that survives
		into a BoolOp's dest at a NON-LAST operand position, given is_and -
		see _boolop_union_remap_shape's own comment for why this matters (`x:
		i32|None or y: str` must not force y to also agree with x's own
		None leaf, since a None x is always falsy and `or` never keeps a
		falsy non-last operand). Deliberately narrow: only NoneType counts,
		exactly the same "single, statically-fixed truthiness" class a
		literal constant belongs to (see _expr_BoolOp's own whole-operand
		skip) - a type with its own __bool__ (str, list, dict, set, ...) is
		genuinely runtime-dependent (an empty string is exactly as real a
		value as a non-empty one), so it's NEVER excluded either direction,
		matching _truthiness_of_operand's own identical distinction. `and`
		never excludes anything via this: None is a valid FALSY survivor
		for `and` (not excludable), and no OTHER type has a single fixed
		truthiness at the bare TYPE level - only a literal CONSTANT does,
		already handled separately. '''
		if not is_and:
			return leaf_type is self.lowering.discovery.get_none_type()
		return False

	def _boolop_union_remap_shape( self, operand_type: Type|None, dest_type: Type|None, is_last: bool, is_and: bool ) -> 'tuple[TaggedUnion,list[Variable],list[Variable]]|None':
		''' does a DECISIVE BoolOp operand need per-LEAF re-mapping into
		dest's own (possibly wider/differently-shaped) union type, rather
		than a single plain coercion of its WHOLE type? Two shapes need
		this:
		- a NON-LAST operand with at least one of its own leaves PROVABLY
		  excluded from ever surviving here (_boolop_leaf_excluded) - e.g.
		  `x: i32|None or y: str`: a None x is always falsy, so `or` never
		  keeps it - only the REMAINING (kept) leaves need to agree with
		  dest's type, not operand_type as a whole.
		- the LAST operand, whenever its own type is a union that doesn't
		  already exactly equal dest's (e.g. `z: str|i32|None = y or x`,
		  x: i32|None, y: str - x is unconditionally decisive here
		  regardless of its own truthiness, per real Python's `or`/`and`
		  keeping the last operand "as-is" even when falsy, so EVERY leaf
		  of x, not just some, needs its own chance to become a member of
		  dest's own wider union). Confirmed as a real, necessary case, not
		  just the non-last one: without this, `x` (i32|None) failed
		  _coerce_or_check_operand's plain, whole-type coercion outright -
		  that path only ever handles a bare LEAF value becoming a member
		  of a union, never one union's own leaves flowing into a
		  DIFFERENT, wider union.
		Returns None when there's nothing to remap: operand_type isn't a
		union at all, the two types already match exactly, dest_type isn't
		established yet AND this is the last operand (operand's own
		natural, WHOLE type just becomes dest's directly - no remapping
		needed, nothing to exclude for is_last anyway), or (non-last case
		only, degenerately) every leaf would be excluded - each of these
		is safest left to the ordinary, single coercion path instead of
		asserted against. A Result[T,E] is never remapped, same reasoning
		as _truthiness_of_operand's own identical exclusion: it must stay
		fully, explicitly checked (.unwrap()/.is_ok()/match), never
		implicitly unwrapped by a truthiness/last-operand position.
		dest_type is NOT established yet for a NON-last operand with a
		real exclusion (e.g. `tz or localtz()`, tz: ZoneInfo|None, no
		annotation) is still remapped: with exactly one surviving leaf,
		_emit_decisive_remapped's own per-leaf loop seeds dest's type
		directly from that ONE leaf (no union wrapping needed at all,
		since there's only one possible outcome); with two or more,
		_emit_decisive_remapped synthesizes/interns a real union of
		exactly the kept leaves first (discovery._get_or_create_union -
		the same interning every other anonymous union in this compiler
		already goes through), so a later operand still gets a real,
		properly excluded target to widen into instead of x's own full,
		unexcluded type. Confirmed as a real, necessary case, not just the
		"dest already established" one: skipping it here left dest seeded
		from tz's own UNEXCLUDED ZoneInfo|None (`(tz or localtz()).
		utcoffset(...)` then failed with "'utcoffset' is not callable on
		intrinsics.NoneType" - lib/datetime.py's own real usage of this
		exact idiom); the two-or-more-leaf case was ALSO confirmed broken
		on its own (`x: i32|str|None; y: bool; z = x or y` - a hard
		compile error, "expected str|None|i32, got bool", even though
		real Python's own `x or y` here unambiguously excludes x's None). '''
		shape = self.lowering._type_resolver._tagged_union_shape( operand_type )
		if shape is None:
			return None
		base, members = shape
		if base.is_result_type():
			return None
		if dest_type is not None and self.lowering._type_resolver._same_type( operand_type, dest_type ):
			return None
		if is_last:
			if dest_type is None:
				return None
			return base, members, members
		kept = [ m for m in members if not self._boolop_leaf_excluded( m.type, is_and ) ]
		if not kept:
			return None
		return base, members, kept

	def _emit_decisive_remapped( self, operand_start: int, operand: ir.Operand, remap: 'tuple[TaggedUnion,list[Variable],list[Variable]]', dest_ref: list, node: ast.AST ) -> None:
		''' _expr_BoolOp's emit_decisive, for a union-typed decisive
		operand that needs per-leaf re-mapping into dest's own type
		(_boolop_union_remap_shape) - since which of the KEPT leaves is
		actually active can't be known at compile time when more than one
		remains, this dispatches on the RUNTIME tag (mirrors
		_truthiness_of_union_operand's identical per-leaf tag-Cmp-
		JumpIfFalse-then-Jump-to-a-shared-end shape), doing the ordinary
		strict-coercion + aliasing-vs-fresh merge once PER SURVIVING LEAF
		instead of once for the whole operand - merging through one
		intermediate temp first (extract, THEN decide fresh-vs-borrowed
		once for the merged result) would blur whether that merged value
		is a fresh wrap (_coerce_into_union already increfs what it wraps)
		or a borrowed passthrough (still needs its own Incref here),
		corrupting refcounts either way; committing to a live
		is_union_coerce_result check per leaf (exactly what _is_aliasing_
		expr already relies on elsewhere for the identical reason) is what
		keeps this safe. `dest_ref` is a one-element list holding the
		enclosing _expr_BoolOp's own `dest` (a plain nonlocal isn't
		reachable from this method) - mutated in place the same way
		emit_decisive's own `nonlocal dest` would. `operand` itself (the
		ORIGINAL, wider union) is fully consumed by the end of this call -
		its own remaining temps (plus every tag/cmp/payload-extraction temp
		from every branch, taken or not) are flushed via the ordinary
		_flush_branch_temps, keeping only dest. '''
		base, _all_members, kept = remap
		tag_attr, _data_attr, _payload_cls, tags = self.lowering._union_storage.get( base )
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		end_label = self._new_label( 'boolop_narrow_end' )
		# dest not established yet and more than one leaf survives the
		# exclusion: its own type can't be seeded from whichever leaf
		# happens to be processed FIRST in the loop below (that would
		# arbitrarily drop the others) - synthesize/intern a real union of
		# exactly the kept leaves up front instead, mirroring
		# _emit_binop_fallible_dispatch's own identical success/error-
		# union collapsing (single leaf -> that leaf itself, 2+ ->
		# discovery._get_or_create_union, then schedule() + union_storage.
		# get() so it's actually usable in emitted IR, not just interned).
		if dest_ref[0] is None and len( kept ) > 1:
			narrowed = self.lowering.discovery._get_or_create_union( [ m.type for m in kept ] )
			self.lowering.schedule( narrowed )
			self.lowering._union_storage.get( narrowed )
			dest_ref[0] = self._new_temp( narrowed )
		for member in kept:
			next_label = self._new_label( 'boolop_narrow_next' )
			tag_dest = self._new_temp( tag_attr.type )
			self._emit( ir.GetAttr( dest = tag_dest, obj = operand, attr = tag_attr.stem ))
			cmp_dest = self._new_temp( bool_cls )
			self._emit( ir.Cmp( dest = cmp_dest, op = ir.CmpOp.EQ, left = tag_dest, right = ir.Const( type = tag_attr.type, value = tags[member.stem] )))
			self._emit( ir.JumpIfFalse( cond = cmp_dest, target = next_label ))
			# owning=False (default): a transient borrow out of `operand` -
			# same as _truthiness_of_union_operand's identical extraction.
			# The subsequent _coerce_or_check_operand call either wraps it
			# fresh (is_union_coerce_result=True, already Increfed by that
			# wrap's own constructor) or leaves it exactly as this borrowed
			# read (no wrap needed - payload.type already equals dest's own
			# type), in which case it still shares operand's own reference
			# and needs its own Incref below before becoming dest's
			# independent copy.
			payload = self._maybe_unwrap_union_arg( operand, member.type )
			dest = dest_ref[0]
			coerced = self._coerce_or_check_operand( payload, dest.type if dest is not None else None, node )
			if dest is None:
				dest = self._new_temp( coerced.type )
				dest_ref[0] = dest
			if getattr( coerced, 'is_union_coerce_result', False ):
				self._cfg.untrack_temp( coerced )
			else:
				for instr in self._cfg.incref( dest.type, coerced ):
					self._emit( instr )
			self._emit( ir.Assign( dest = dest, src = coerced ))
			self._emit( ir.Jump( target = end_label ))
			self._emit( ir.Label( name = next_label ))
		self._emit( ir.Label( name = end_label ))
		self._flush_branch_temps( operand_start, dest_ref[0] )

	def _expr_BoolOp( self, node: ast.BoolOp, expected_type: Type|None ) -> ir.Operand:
		# short-circuit and/or: evaluate operands left to right, stopping
		# early (jump to end) as soon as the result is already decided -
		# `and` stops on the first FALSY operand, `or` on the first
		# TRUTHY one - same short-circuit control flow as before. `dest`
		# now ends up holding the DECISIVE operand's own VALUE, not a
		# forced bool coercion (Python semantics: `5 and 10` -> 10, `0 or
		# 10` -> 10) - the decisive operand's truthiness (computed via
		# _truthiness_of_operand, without re-lowering it) only decides
		# WHETHER to jump, never what gets merged into dest. Needed by
		# match's nested pattern tests (an outer tag check AND, only if
		# that passes, an inner tag check on the payload - reading the
		# payload before confirming the outer tag would be reading the
		# wrong union member's storage).
		#
		# The first operand actually lowered as a real candidate fixes
		# dest's own type; every later such operand is checked/coerced
		# against that same type via the ordinary strict _lower_expr path
		# - same "operands share one common type" assumption _lower_
		# binary_operands already makes for +/-/* etc. A genuine mismatch
		# across the chain is a real compile error via the usual
		# _check_assignable message.
		#
		# A literal operand (ast.Constant) whose Python truthiness is
		# known at compile time is folded instead of given a runtime
		# check:
		# - non-decisive for this op (`0` in `0 or x`, `True` in `True
		#   and x`) can never end up in dest, so it's skipped outright -
		#   its own, possibly unrelated literal type never has to agree
		#   with the rest of the chain either. Needed for `0 or 'x'` ->
		#   'x': without this, 0's own int type would wrongly become
		#   dest's type before 'x' is ever reached, rejecting it as a
		#   mismatch.
		# - decisive for this op (`True or foo()`, `False and foo()`)
		#   makes dest = that literal and stops lowering the chain right
		#   there - every remaining operand is provably unreachable,
		#   exactly like Python's own bytecode never touching them. Not
		#   just an optimization: without it, foo()'s own return type
		#   would have to agree with the decisive literal's type too
		#   (None vs bool here), which real Python never requires since
		#   it never evaluates foo() at all.
		#
		# Each operand's own intermediate temps (e.g. `field.find(x)`'s
		# Result temp, consumed as `.is_ok()`'s receiver) are flushed via
		# _flush_branch_temps right after that operand's own code runs -
		# see that method's own comment for the confirmed crash this
		# fixes. A non-decisive operand's OWN value is flushed right
		# along with them (nothing kept but dest) since it never survives
		# into dest either.
		is_and = isinstance( node.op, ast.And )
		end_label = self._new_label( 'booland' if is_and else 'boolor' )
		# only reachable when some non-last operand takes the runtime-split
		# path below; an all-compile-time-decisive chain (e.g. `True and x`)
		# falls through to dest without ever jumping here, so unless we
		# track that and skip emitting the label, C sees a dead `L:;` -
		# unused-label warning.
		end_label_used = False
		dest = self._new_temp( expected_type ) if expected_type is not None else None

		def lower_operand( value_node: ast.expr ) -> tuple[int, ir.Operand]:
			# strict=False: the real, strict "must agree with dest's type"
			# check is deferred to emit_decisive below, not applied here -
			# a decisive operand whose type is a union needing per-leaf
			# RE-MAPPING into dest's own type (see
			# _boolop_union_remap_shape) needs its own REAL, natural type
			# intact to even detect that shape; forcing it through dest.type
			# here first would reject it outright (a whole union coercing
			# into a DIFFERENT union isn't a coercion this compiler supports
			# in general - only per-LEAF, which is exactly what remapping
			# does instead).
			operand_start = len( self._pending_temps )
			operand = self._lower_expr( value_node, dest.type if dest is not None else expected_type, strict = False )
			return operand_start, operand

		def emit_decisive( operand_start: int, value_node: ast.expr, operand: ir.Operand, is_last: bool ) -> None:
			# the deferred strict coercion/check lower_operand skipped -
			# same "operands share one common type" requirement as always,
			# just applied here, at the one choke point every decisive
			# operand (is_last, a decisive literal constant, or a non-last
			# operand that turned out decisive at runtime) passes through.
			# A union-typed operand that needs per-leaf re-mapping into
			# dest's own type (_boolop_union_remap_shape - covers BOTH a
			# non-last operand with some of its own leaves excluded, e.g.
			# `x: i32|None or y: str`, AND the last operand whenever its
			# own union differs from dest's, e.g. `z: str|i32|None = y or
			# x`) is delegated to _emit_decisive_remapped instead - plain
			# _coerce_or_check_operand below only ever handles a bare LEAF
			# value becoming a union member, never one union's own leaves
			# flowing into a DIFFERENT, wider union.
			nonlocal dest
			remap = self._boolop_union_remap_shape( operand.type, dest.type if dest is not None else None, is_last, is_and )
			if remap is not None:
				dest_ref = [ dest ]
				self._emit_decisive_remapped( operand_start, operand, remap, dest_ref, node )
				dest = dest_ref[0]
				return
			operand = self._coerce_or_check_operand( operand, dest.type if dest is not None else None, value_node )
			# mirrors _expr_IfExp's own per-branch RC bookkeeping: an
			# aliasing operand (existing Name/Attribute read) needs its
			# own Incref before merging into dest, a fresh one (Call/
			# Allocate result, INCLUDING a union-wrap _coerce_or_check_
			# operand just built above) just has its ownership moved via
			# untrack_temp - see _expr_IfExp's own comment for the
			# confirmed UAF/double-free this avoids. _is_aliasing_expr's
			# own is_union_coerce_result check (see its docstring) is what
			# makes this correct even though `operand` may no longer be
			# what `value_node` originally looked like.
			if dest is None:
				dest = self._new_temp( operand.type )
			if self.lowering._is_aliasing_expr( value_node, operand ):
				for instr in self._cfg.incref( dest.type, operand ):
					self._emit( instr )
			else:
				self._cfg.untrack_temp( operand )
			self._flush_branch_temps( operand_start, dest, operand )
			self._emit( ir.Assign( dest = dest, src = operand ))

		n = len( node.values )
		for i, value_node in enumerate( node.values ):
			is_last = i == n - 1
			static_truth = bool( value_node.value ) if isinstance( value_node, ast.Constant ) else None
			if static_truth is not None and not is_last:
				decisive = ( is_and and not static_truth ) or ( not is_and and static_truth )
				if not decisive:
					continue # provably never reaches dest - skip regardless of its own type
				operand_start, operand = lower_operand( value_node )
				emit_decisive( operand_start, value_node, operand, True )
				break # every remaining operand is provably unreachable
			operand_start, operand = lower_operand( value_node )
			if is_last:
				emit_decisive( operand_start, value_node, operand, True )
				break
			cond = self._truthiness_of_operand( operand, value_node )
			# _truthiness_of_operand's own default-truthy fallback (a bare,
			# no-__bool__, non-Scalar, non-union type - e.g. a plain
			# RCClass instance) returns a compile-time Const, not a real
			# runtime check - same "single, statically-fixed truthiness"
			# class a literal AST Constant belongs to (see the static_truth
			# handling above), just discovered from the operand's TYPE
			# instead of its own source syntax. Route it through the exact
			# same compile-time decisive-or-skip logic: without this, a
			# NON-decisive such operand (`f and 1`, f: a bare RCClass -
			# always truthy, so `and` never keeps it) would still generate
			# a real (if dead/unreachable) runtime jump PLUS a "decisive"
			# branch that tries to coerce f's own type into dest - which
			# can fail outright in a condition context (dest.type forced
			# to bool) even though real Python could never actually reach
			# that branch at all.
			const_truth = cond.value if isinstance( cond, ir.Const ) and cond.type is self.lowering.discovery.find_name( 'bool', node ) else None
			if const_truth is not None:
				decisive = ( is_and and not const_truth ) or ( not is_and and const_truth )
				if not decisive:
					self._flush_branch_temps( operand_start, dest )
					continue
				emit_decisive( operand_start, value_node, operand, False )
				break
			continue_label = self._new_label( 'booland_continue' if is_and else 'boolor_continue' )
			skip_opcode = ir.JumpIfTrue if is_and else ir.JumpIfFalse
			# snapshot _pending_temps HERE, before the branch split: operand
			# (and cond's own temp, and any of operand's own intermediate
			# temps) are shared, ALREADY-COMPUTED values whose FATE forks
			# into two mutually exclusive runtime paths below (kept by the
			# decisive branch, discarded by the continue branch) - but
			# _pending_temps is one flat, IN-PLACE-MUTATED list, not scoped
			# per branch the way _expr_IfExp's true/false branches are
			# (each of THOSE captures its own FRESH start point, AFTER the
			# other branch's own flush already trimmed the list, since each
			# branch lowers its own independent sub-expression from
			# scratch). Here, both branches share the SAME already-lowered
			# operand: emit_decisive's own flush call (for the decisive
			# branch, generated first in COMPILE-TIME instruction order)
			# permanently trims operand out of _pending_temps - by the time
			# the continue branch's OWN flush call below runs, it would
			# find nothing left to release, NEVER emitting operand's
			# DeleteTemp/Decref in EITHER branch's actual code. Confirmed
			# as a real, reproducible LEAK (not a crash -
			# `base.upper() and (tail + '')`'s discarded str never got
			# decref'd in the continue branch's own generated code), missed
			# by refcount()-based testing that only ever inspected the
			# SURVIVING value, never the discarded one. Restoring the
			# snapshot right before the continue branch gives it back its
			# own, independent view of what's pending, exactly mirroring
			# IfExp's own effect (a fresh, unclaimed view per branch).
			pending_snapshot = list( self._pending_temps )
			# same problem, one layer down: fresh_temp()/untrack_temp()
			# mutate cfg._temp_states in place too - the decisive branch's
			# own untrack_temp(operand) call (run first, compile-time-
			# sequentially) permanently un-registers operand's ownership,
			# so the continue branch's later delete_temp() on the SAME
			# operand would silently skip its Decref. See
			# snapshot_temp_states' own docstring for the confirmed leak
			# this fixes.
			temp_states_snapshot = self._cfg.snapshot_temp_states()
			self._emit( skip_opcode( cond = cond, target = continue_label ))
			emit_decisive( operand_start, value_node, operand, False )
			self._emit( ir.Jump( target = end_label ))
			end_label_used = True
			self._emit( ir.Label( name = continue_label ))
			self._pending_temps = pending_snapshot
			self._cfg.restore_temp_states( temp_states_snapshot )
			self._flush_branch_temps( operand_start, dest )
		if end_label_used:
			self._emit( ir.Label( name = end_label ))
		self._cfg.fresh_temp( dest, dest.type )
		return dest

	def _expr_IfExp( self, node: ast.IfExp, expected_type: Type|None ) -> ir.Operand:
		# ternary `x if cond else y` — both branches assign to the same
		# dest temp, then merge at end_label. Use JumpIfTrue so the true
		# branch (body) comes first, avoiding an extra negate.
		#
		# RC bookkeeping mirrors cfg.assign()'s own is_alias split, done
		# per-branch since node.body/node.orelse can differ in aliasing-ness
		# (e.g. `x if cond else str('literal')`): an ALIASING branch value
		# (a plain Name/GetAttr read of an already-live binding) needs its
		# own Incref before being merged into dest, since dest becomes an
		# independent, longer-lived holder of the same reference; a FRESH
		# branch value (a Call/Allocate result, already registered via
		# fresh_temp() by whatever lowered it) has its ownership MOVED into
		# dest via the plain ir.Assign below, so it must be untrack_temp()'d
		# - otherwise _flush_pending_temps' later decref of the branch's own
		# temp double-frees the exact same object dest (and whatever dest
		# gets assigned into) still holds. dest itself only becomes tracked
		# once, after both branches (fresh_temp() is idempotent per id) -
		# confirmed as a real, reproducible UAF/double-free via direct
		# testing (`str('-') if cond else str('+')` corrupted/crashed
		# before this fix), not just reasoning from the code shape.
		#
		# Each branch's own PURELY INTERMEDIATE temps (e.g. every temp a
		# chained `prefix + str('.') + k` concatenation DeclareTemp's along
		# the way, none of which is `true_val`/`false_val` itself) are
		# flushed inside that branch via _flush_branch_temps - see its own
		# comment for why: left to the enclosing statement's normal
		# end-of-statement flush, they leak past end_label and get
		# unconditionally decref'd even in the branch that never ran,
		# releasing an uninitialized C local - a real, reproducible stack-
		# overflow crash (release_object on stack garbage), not just a
		# leak/UAF, confirmed via direct testing.
		cond = self._lower_truth_test( node.test )
		else_label = self._new_label( 'ifexp_else' )
		end_label = self._new_label( 'ifexp_end' )
		dest = self._new_temp( expected_type ) if expected_type is not None else None
		self._emit( ir.JumpIfFalse( cond = cond, target = else_label ))
		# true branch
		true_branch_start = len( self._pending_temps )
		true_val = self._lower_expr( node.body, expected_type )
		if dest is None:
			dest = self._new_temp( true_val.type )
		if self.lowering._is_aliasing_expr( node.body, true_val ):
			for instr in self._cfg.incref( dest.type, true_val ):
				self._emit( instr )
		else:
			self._cfg.untrack_temp( true_val )
		self._flush_branch_temps( true_branch_start, dest, true_val )
		self._emit( ir.Assign( dest = dest, src = true_val ))
		self._emit( ir.Jump( target = end_label ))
		# false branch
		self._emit( ir.Label( name = else_label ))
		false_branch_start = len( self._pending_temps )
		false_val = self._lower_expr( node.orelse, dest.type )
		if self.lowering._is_aliasing_expr( node.orelse, false_val ):
			for instr in self._cfg.incref( dest.type, false_val ):
				self._emit( instr )
		else:
			self._cfg.untrack_temp( false_val )
		self._flush_branch_temps( false_branch_start, dest, false_val )
		self._emit( ir.Assign( dest = dest, src = false_val ))
		self._emit( ir.Label( name = end_label ))
		self._cfg.fresh_temp( dest, dest.type )
		return dest

	def _expr_Compare( self, node: ast.Compare, expected_type: Type|None ) -> ir.Operand:
		# ast.Is/IsNot ARE handled (see _lower_is_comparison) - identity
		# happens to coincide with value equality for every value kind this
		# language has today. ast.In/NotIn ARE ALSO handled (see
		# _lower_in_comparison) but needed their own dispatch method rather
		# than falling through _COMP_DUNDER below - see that method's own
		# comment for why
		if len( node.ops ) != 1 or len( node.comparators ) != 1:
			self.lowering.discovery.fail( f'chained comparisons are not yet supported: {ast.unparse(node)}', node )
		if isinstance( node.ops[0], ( ast.Is, ast.IsNot )):
			return self._lower_is_comparison( node, negate = isinstance( node.ops[0], ast.IsNot ))
		if isinstance( node.ops[0], ( ast.In, ast.NotIn )):
			return self._lower_in_comparison( node, negate = isinstance( node.ops[0], ast.NotIn ))

		if ( isinstance( node.left, ast.Constant ) and not isinstance( node.comparators[0], ast.Constant )
				and isinstance( node.comparators[0], ast.Name )):
			# a bare literal on the left needs a hint from the right side's
			# own type, or it locks in as builtins.int (this file's own
			# bare-literal default) before dispatch even looks for a
			# matching dunder - e.g. `1 < a` (a: i32) needs the literal to
			# become i32, not int, to find i32.__lt__. Only probed here for
			# a bare Name - provably side-effect-free to look up twice
			# (re-lowered again below/inside _lower_eq_or_ne exactly as it
			# always was) - a Call or other expression is left alone rather
			# than risk double-evaluating it.
			right_probe = self._lower_expr( node.comparators[0], None, strict = False )
			left_hint = right_probe.type if isinstance( right_probe.type, Scalar ) else None
			left = self._lower_expr( node.left, left_hint )
		else:
			left = self._lower_expr( node.left, None )
		if isinstance( node.ops[0], ( ast.Eq, ast.NotEq )):
			# Eq/NotEq get their OWN unified path (_lower_eq_or_ne), tried
			# BEFORE the scalar-vs-non-scalar fork below (unlike every other
			# operator) - a union can appear on EITHER side regardless of
			# whether the OTHER side happens to be scalar: `None == x` (a
			# bare None literal - NoneType is itself Scalar, see discovery.
			# py's get_none_type - has no dunder of its own at all) or `5 ==
			# n` (a bare int literal, also Scalar) both need the same
			# union-on-the-right handling as `"hi" == x`. The generic
			# method-lookup-on-non-scalar-left dispatch below can never
			# cover either shape, since it never even runs when left is
			# Scalar.
			return self._lower_eq_or_ne( node, left, expected_type, negate = isinstance( node.ops[0], ast.NotEq ))

		# try the dunder method (<, >, <=, >=, ...) - NOT gated on
		# isinstance(left.type, Scalar) anymore: _find_dunder_for_arg
		# already resolves a Scalar-registered dunder (i32.__lt__ = ...;
		# see lib/builtins/__scalar_dunders.py) identically to a real
		# class's own method (both just read .names). A Scalar with
		# nothing registered under this name (NoneType, in practice) just
		# misses, exactly like a class that doesn't define the dunder at
		# all - falls through to flat Cmp below either way, same
		# precedent as _lower_binop_values' own arithmetic dispatch
		method_name = _COMP_DUNDER.get( type( node.ops[0] ))
		if method_name is not None:
			# _find_dunder_for_arg, not the plain _find_method - see
			# its own docstring: right, lowered with strict=True just
			# below, is already guaranteed to end up exactly left.type
			# (coerced or rejected) before this dunder lookup even
			# matters, so the wanted implementation is whichever one
			# declares its own parameter as exactly left.type - same
			# "caller already knows the wanted arg type" shape as
			# _lower_eq_or_ne's own same-type fast path
			method = self._find_dunder_for_arg( left.type, method_name, left.type )
			if method is not None:
				right = self._lower_expr( node.comparators[0], left.type )
				# _emit_fallible_method_call, not a hand-rolled ir.Call -
				# a Scalar-registered dunder (method.cls is None) needs its
				# receiver threaded as a plain leading positional arg
				# instead of ir.Call.receiver (discovery.py never strips
				# "self" off a free function's own parameter list the way
				# it does for a real class method - same fix _lower_call/
				# _lower_method_call/_emit_binop_dunder_call already apply)
				return self._emit_fallible_method_call( node, method, left, [ right ], expected_type )
		# no matching dunder - a hard compile error, not a silent flat-Cmp
		# fallback (pointer identity comparison, or a plain scalar compare):
		# confirmed with the user - there's no sensible default for
		# comparing two arbitrary values (an RCClass's own == is meaningless
		# unless the class defines it), so every comparable type needs a
		# real dunder now, no exceptions. Scalars/Ptr[T]/ConstPtr[T]/CEnum
		# all have one (gen_scalar_dunders.py, lib/builtins/__ptr_arith.py,
		# discovery.py's _synthesize_cenum_comparison_methods) - this is
		# reached only by a genuine RCClass/CStruct/CUnion/TaggedUnion (or a
		# leftover corner) with no __<op>__ of its own.
		right = self._lower_expr( node.comparators[0], left.type )
		cmp_op = self.lowering._CMP_OPCODES.get( type( node.ops[0] ))
		if cmp_op is None:
			self.lowering.discovery.fail( f'unsupported comparison operator: {ast.unparse(node)}', node )
		type_name = left.type.qualname if left.type is not None else '?'
		self.lowering.discovery.fail(
			f'{type_name} has no {method_name}() defined - comparison requires an explicit dunder: {ast.unparse(node)}',
			node,
		)
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		dest = self._new_temp( bool_cls )
		self._emit( ir.Cmp( dest = dest, op = cmp_op, left = left, right = right ))
		return dest

	def _lower_operand_compare( self, left: ir.Operand, right: ir.Operand, negate: bool, node: ast.AST ) -> ir.Operand:
		''' Eq/NotEq between two ALREADY-LOWERED operands of the SAME
		(non-union) type - the same dunder-or-flat-Cmp choice _lower_eq_or_ne
		makes from AST nodes, reimplemented against operands directly since
		_build_union_leaf_eq's own narrowed payload has no AST node of its
		own to re-dispatch through (mirrors _coerce_or_check_operand's
		identical "no node to re-evaluate" posture). '''
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		method_name = '__ne__' if negate else '__eq__'
		# _find_dunder_for_arg, not the plain _find_method - both
		# operands are already known to share the SAME type (this
		# method's own docstring), so the wanted implementation is
		# whichever one declares its own parameter as exactly left.type.
		# NOT gated on isinstance(left.type, Scalar) anymore - a Scalar-
		# registered __eq__/__ne__ resolves identically here (see lib/
		# builtins/__scalar_dunders.py), same precedent as everywhere else
		# this file's comparison dispatch dropped that gate.
		method = self._find_dunder_for_arg( left.type, method_name, left.type )
		if method is not None:
			# _emit_fallible_method_call, not a hand-rolled ir.Call - see
			# _expr_Compare's own identical comment on why (Scalar-
			# registered dunder receiver threading)
			return self._emit_fallible_method_call( node, method, left, [ right ], None )
		# no matching dunder - a hard compile error, not a silent flat-Cmp
		# fallback - see _expr_Compare's own identical comment for the full
		# rationale (confirmed with the user: no sensible default exists for
		# comparing two arbitrary values). No source AST node for this
		# specific comparison (this method's own docstring - a narrowed
		# union-leaf payload has none of its own), so `node` here is
		# whatever the caller passed for error-reporting purposes only.
		type_name = left.type.qualname if left.type is not None else '?'
		self.lowering.discovery.fail( f'{type_name} has no {method_name}() defined - comparison requires an explicit dunder', node )
		dest = self._new_temp( bool_cls )
		self._emit( ir.Cmp( dest = dest, op = ir.CmpOp.NE if negate else ir.CmpOp.EQ, left = left, right = right ))
		return dest

	def _lower_is_comparison( self, node: ast.Compare, negate: bool ) -> ir.Operand:
		# `is`/`is not` mean real Python identity - for every value kind
		# this language has today (scalars, pointers, RC handles) identity
		# coincides with value equality, so this is plain Cmp EQ/NE...
		# UNLESS one side is a bare `None` literal being compared against a
		# TaggedUnion-typed value (T|None, e.g. sys._alloc()'s
		# Ptr[u8]|None) - there, "is None" means "the active member is
		# NoneType", which needs a tag check (the same UnionStorage.get
		# machinery match statements/conditional dispatch already use), not
		# a flat Cmp against a synthesized None operand of union type
		# (which wouldn't correspond to any real runtime representation)
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		cmp_op = ir.CmpOp.NE if negate else ir.CmpOp.EQ
		left_node, right_node = node.left, node.comparators[0]
		left_is_none = isinstance( left_node, ast.Constant ) and left_node.value is None
		right_is_none = isinstance( right_node, ast.Constant ) and right_node.value is None

		if left_is_none and right_is_none:
			return ir.Const( type = bool_cls, value = not negate ) # `None is None` / `None is not None` - degenerate, but not a crash

		if left_is_none or right_is_none:
			# a TaggedUnion-typed operand (T|None) never reaches here anymore -
			# type_resolver.py's _ReferenceResolver already rewrote that
			# shape into a plain tag Eq/NotEq Compare before lowering ever
			# saw this statement (see its own visit_Compare). What's left is
			# a flat Cmp against a real None-typed Const - e.g. a raw
			# Ptr[T]|None never actually applies (still a TaggedUnion), so in
			# practice this is for whatever non-union type this language
			# ever allows a bare `is None` against
			other = self._lower_expr( right_node if left_is_none else left_node, None )
			dest = self._new_temp( bool_cls )
			self._emit( ir.Cmp( dest = dest, op = cmp_op, left = other, right = ir.Const( type = other.type, value = None ) ))
			return dest

		left = self._lower_expr( left_node, None )
		right = self._lower_expr( right_node, left.type )
		dest = self._new_temp( bool_cls )
		self._emit( ir.Cmp( dest = dest, op = cmp_op, left = left, right = right ))
		return dest

	def _lower_in_comparison( self, node: ast.Compare, negate: bool ) -> ir.Operand:
		# `x in y` / `x not in y` mean `y.__contains__(x)` (negated for
		# NotIn) - the REVERSE of every other _COMP_DUNDER-driven comparison
		# (==, <, ...), where the LEFT operand is always the receiver. That
		# reversal is exactly why In/NotIn can't just be added as two more
		# _COMP_DUNDER entries and fall through the generic left-operand
		# dispatch above: this lowers the RIGHT operand first and dispatches
		# on ITS type instead.
		right = self._lower_expr( node.comparators[0], None )
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		if not isinstance( right.type, Scalar ):
			method = self.lowering._find_method( right.type, '__contains__' )
			if method is not None:
				self.lowering._ensure_resolved( method )
				self.lowering.schedule( method.return_type )
				for p in ( method.parameters or [] ):
					self.lowering.schedule( p.type )
				param_type = method.parameters[0].type if method.parameters else None
				left = self._lower_expr( node.left, param_type )
				call_dest = self._new_temp( method.return_type )
				self._emit( ir.Call( dest = call_dest, target = method, receiver = right, args = [ left ], kwargs = {} ))
				if not negate:
					return call_dest
				# NotIn: negate __contains__'s plain bool result - ir.Not
				# (same as _expr_UnaryOp's `not x`), NOT
				# _lower_is_comparison's tagged-union-aware EQ/NE flip,
				# which solves an unrelated problem (`is None` narrowing)
				dest = self._new_temp( bool_cls )
				self._emit( ir.Not( dest = dest, operand = call_dest ))
				return dest
		# no __contains__ on a non-scalar right operand, or a scalar right
		# operand entirely (e.g. `x in 5`) - unlike ==, there's no sane
		# degraded fallback (a raw pointer/value compare is never what `in`
		# means), so this is a hard error rather than a silent Cmp fallback
		self.lowering.discovery.fail(
			f'{"not " if negate else ""}in requires a __contains__ method on '
			f'{right.type.qualname if right.type else "?"}: {ast.unparse(node)}',
			node,
		)

	def _narrowed_type_of_name( self, var_id: str, declared_type: Type ) -> Type:
		''' declared_type, unless var_id is currently narrowed (cfg.py's
		narrow(), e.g. inside an `if x is not None:`/`match x: case _:` arm) -
		then the narrowed member's own type instead. Shared by
		_try_lower_indirect_call/_try_lower_closure_call so a Ptr[Callable[...]]
		|None or Closure[...]|None parameter is recognized as callable once
		narrowed to its non-None leaf, not just when declared bare - mirrors
		_static_type_of_value_expr's identical narrowed_member lookup. '''
		member = self._cfg.narrowed_member( var_id )
		return member.type if member is not None else declared_type
