# stdlib imports:
import ast
from dataclasses import replace
from typing import Callable, NoReturn

# local imports:
import cfg
import ir
from discovery import Discovery
from errors import CompileError
from mpy_types import (
	Name, Type, Variable, Parameter, Function, Overload, ClassLike, Specialization, TaggedUnion, CStruct, TypeVar, ConditionalDispatch, RCClass, Scalar, CallableType, ClosureType, TupleType, int_stem_range,
)
import overload_resolution
from union_storage import ReceiverDispatch as _ReceiverDispatch

from lowering_shared import _CHECKED_BINOP_OPCODES, _CHECKED_FLOAT_BINOP_OPCODES

class CallLoweringMixin:
	''' call resolution and dispatch (the core `_lower_call` machinery) - mixed into FunctionLowering (lowering.py), which
	see for the shared instance state (self._instructions, self._cfg, self.lowering,
	etc.) every method here reads and writes. Never instantiated on its own;
	split out of lowering.py purely to keep that file to a manageable size - see
	lowering.py's own class docstring and FunctionLowering's base-class list for
	the full set of sibling mixins this one is composed with. '''


	def _expr_Call( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		return self._lower_call( node, expected_type, want_result = True )

	def _resolve_callee( self, func_node: ast.expr ) -> tuple[Function|Overload|Specialization|_ReceiverDispatch,ir.Operand|None]:
		target = self.lowering._type_resolver._resolve_callee_target( func_node )
		if target is not None:
			# module-level `_x`/`__x` privacy (SYNTAX.md) - THIS is the one
			# real "about to actually call this" resolution site (unlike
			# _stmt_diverges's own unrelated re-probe of the identical
			# _resolve_callee_target/_try_resolve_namespace chain, purely to
			# ask "is this NoReturn-shaped" - see _try_resolve_namespace's
			# own comment on why enforcing THERE was a real false positive).
			# Unwrap a Specialization the same way _stmt_diverges does - a
			# private GENERIC function reached this way is still a Function
			# underneath, just monomorphization-wrapped.
			fn_target = target.base if isinstance( target, Specialization ) else target
			if id( fn_target ) not in self._inline_param_alias_ids:
				self.lowering.discovery.check_module_visibility( fn_target, func_node, self._owning_module )
			return target, None

		if not isinstance( func_node, ast.Attribute ):
			self.lowering.discovery.fail( f'cannot call {ast.unparse(func_node)}', func_node )
		receiver = self._lower_expr( func_node.value, None )
		shape = self.lowering._type_resolver._tagged_union_shape( receiver.type )
		if shape is not None:
			base, members = shape
			self.lowering._ensure_resolved( base )
			direct = base.get_local_or_raise( func_node.attr )
			if not isinstance( direct, ( Function, Overload )):
				# receiver's own VALUE/TYPE is deliberately left exactly as
				# lowered above (still the full, wide union) - only the SET OF
				# CANDIDATE LEAVES this dispatches over narrows, when the
				# receiver's own subject is proven (cfg.py's narrow_many(),
				# e.g. a 3+-member T|U|None union's `is None` guard) to be one
				# of fewer than the union's own full member list. Without
				# this, `x.value()` (a method T and U both define, but None
				# doesn't) unconditionally required EVERY member - including
				# a member already PROVEN impossible here - to define it,
				# rejecting an entirely reachable call. Any other read of the
				# same name (e.g. a `type(x) is T` tag comparison) is
				# untouched by this - it still sees receiver's own real,
				# unmodified operand.
				subject_key = self._attribute_chain_key( func_node.value )
				if subject_key is not None:
					narrowed = self._cfg.narrowed_members( subject_key )
					if narrowed is not None and 0 < len( narrowed ) < len( members ):
						members = narrowed
				return self.lowering._type_resolver._resolve_union_receiver_members( base, members, func_node.attr, func_node ), receiver
		target = self.lowering._type_resolver._attr_lookup_callable( receiver.type, func_node.attr, func_node )
		return target, self._maybe_deref_arrow_receiver( receiver, target )

	def _maybe_deref_arrow_receiver( self, receiver: ir.Operand, target: Function|Overload|Specialization|_ReceiverDispatch ) -> ir.Operand:
		''' `p.method()` where p: Ptr[T]/ConstPtr[T] is arrow-sugar -
		_attr_lookup_callable already redirected the NAME LOOKUP to T's own
		method, but `receiver` above is still evaluated against p's own,
		un-redirected Ptr[T] type. A plain (non-interface) CStruct or an
		RCClass method's self expects a real T value (T's own by-value
		struct, or the RCClass's own bare pointer respectively - see
		lower_function's own self_param construction), not the raw Ptr[T] -
		dereference it here, the same ir.GetItem the ordinary `p[0]`
		subscript uses (Ptr[T]/ConstPtr[T]'s own dereference convention).
		An @interface CStruct's self is ALWAYS Ptr[T] itself though - there
		the pointer already IS what self expects, so this leaves receiver
		untouched (confirmed: dereferencing there produced invalid C -
		passing a by-value struct where the generated prototype declares a
		pointer). '''
		if not self.lowering._type_resolver._is_ptr_specialization( receiver.type ):
			return receiver
		target_cls = getattr( target, 'cls', None )
		if target_cls is None or ( isinstance( target_cls, CStruct ) and target_cls.is_interface ):
			return receiver
		index_type = self.lowering.discovery.get_intrinsics()['usize']
		index = ir.Const( type = index_type, value = 0 )
		dest = self._new_temp( receiver.type.args[0] )
		self._emit( ir.GetItem( dest = dest, obj = receiver, index = index ))
		return dest

	def _apply_move_hook( self, param: Parameter, operand: ir.Operand, target_qualname: str, node: ast.AST ) -> None:
		# the semantic half of move[T] - _check_move_argument (run earlier,
		# inside _match_call_args) already validated the call-site syntax
		# agrees; this is where the argument's OWN ownership state actually
		# transitions, once its real Operand exists (needs the lowered
		# value, not just the AST expr) - shared by every _match_call_args
		# caller (plain calls, both generic call flavors, union-receiver
		# dispatch), called right after each argument is lowered.
		# cfg.py's move() raises a bare CompileError (it has no ErrorCollector
		# access - see its own module docstring), so it must be caught and
		# re-recorded here like every other cfg.py call site does - otherwise
		# it's an unrecorded exception, and FunctionLowering.run()'s per-
		# statement `except CompileError: continue` recovery silently
		# swallows it with no diagnostic at all, letting a move of a still-
		# BORROWED argument compile clean (this was a real bug - a bare
		# `raise CompileError` here, unlike every sibling cfg.py call site,
		# was never caught/re-recorded, so it vanished into that recovery
		# boundary; confirmed by a real use-after-free, see lib/ed25519.py's
		# history).
		if param.is_move:
			try:
				instructions = self._cfg.move( operand, target_qualname = target_qualname, param_stem = param.stem )
			except CompileError as e:
				self.lowering.discovery.fail( str( e ), node )
			for instr in instructions:
				self._emit( instr )

	def _lower_overload_arg( self, expr: ast.expr, position: int|None, kw_name: str|None, candidates: list[Function], node: ast.AST ) -> ir.Operand:
		if isinstance( expr, ast.Lambda ):
			return self._lower_overload_lambda_arg( expr, position, kw_name, candidates, node )
		if not isinstance( expr, ast.Constant ):
			return self._lower_expr( expr, None )
		compatible_stems = self.lowering._LITERAL_COMPATIBLE_STEMS.get( type( expr.value ) )
		if compatible_stems is None:
			return self._lower_expr( expr, None ) # a None literal, or something else - falls through to _expr_Constant's own "cannot infer" error, same as before

		candidate_types: list[Type] = []
		for fn in candidates:
			# a stub (fn.bound_to is not None) is "signature-only, never
			# actually called" (see mpy_types.Overload's own docstring) - the
			# real, emitted call always targets its bound_to implementation,
			# whose OWN parameter type is what the literal argument actually
			# needs to satisfy at the C level, not the stub's own (often
			# narrower) declared one. Without this redirect, a stub bound to
			# a WIDER implementation (e.g. Result[T,E].unwrap_or's own
			# `default: T` stub, bound to the plain `default: T|None = None`
			# impl) contributed its own narrower scalar type as a SEPARATE
			# candidate alongside the impl's own wider union type - two
			# genuinely different Type objects for what is, at the real call
			# site, the exact same parameter slot - which the magnitude-based
			# disambiguation just below (correctly designed for choosing
			# between truly independent overload arms, e.g. i8 vs i32) then
			# resolved by picking the narrower SCALAR one, since only a
			# Scalar passes its own isinstance(t, Scalar) filter. The literal
			# was then lowered against that narrower type while the actual,
			# real call still targets the wider union-typed implementation -
			# a genuine "passing 'int' to parameter of incompatible type
			# 'struct $__u$$...'" C mismatch, confirmed via a real compile of
			# Result[i32,str]'s own .unwrap_or(5). Redirecting first makes a
			# bound stub and its own implementation contribute the SAME
			# (impl's) type, deduping to one real candidate - matching what
			# actually gets called
			real_fn = fn.bound_to if fn.bound_to is not None else fn
			if real_fn.parameters is None:
				continue
			param = (
				real_fn.parameters[position] if position is not None and position < len( real_fn.parameters ) else
				next( ( p for p in real_fn.parameters if p.stem == kw_name ), None )
			)
			if param is None or param.type is None:
				continue
			if getattr( param.type, 'stem', None ) in compatible_stems:
				matched_type = param.type
			else:
				# not DIRECTLY a compatible scalar - but a union-typed param
				# (e.g. Result[T,E].unwrap_or's own `default: T|None`, once T
				# itself substitutes to a scalar) can still unambiguously
				# accept this literal, through exactly one of its own leaves.
				# Without this, a union-typed candidate was always silently
				# skipped here regardless of whether it fit, so a literal
				# argument to an overloaded call never got the union-
				# coercion _lower_expr's own expected_type machinery
				# (_coerce_into_union) already does correctly for an ORDINARY
				# (non-overloaded) call - the literal fell through to the
				# "no candidate's parameter type is even plausible" path
				# below, lowered with expected_type=None, and reached
				# emitter_c as a bare scalar handed to a C parameter whose
				# real type is the whole union struct: a genuine, confirmed
				# "passing 'int' to parameter of incompatible type 'struct
				# $__u$$...'" C compile error (Result[i32|None,str].
				# unwrap_or(42), a fallible generator's own g.__next__().
				# unwrap_or(default) - any T|None-shaped default param at
				# all). len(...)==1 (not >=1) mirrors this method's own
				# existing ambiguity discipline just below: a literal that
				# plausibly fits more than one leaf of the SAME union is
				# exactly as ambiguous as fitting more than one candidate
				# scalar directly would be, and is left for the ordinary
				# ambiguous-candidate error path rather than silently
				# guessing one.
				#
				# the matched LEAF itself (not the whole union) is what gets
				# added as this candidate's type below - lowering the literal
				# with the union as its expected_type would make the
				# resulting operand's own STATIC type the whole union (both
				# members "possible" as far as any TYPE-based reasoning can
				# tell), even though a fresh literal's value is obviously,
				# unambiguously never the None variant. overload_resolution.
				# resolve_call is a pure function of TYPES with no way to see
				# that extra fact - an operand whose type is the whole union
				# made it build a genuine, unnecessary RUNTIME conditional
				# dispatch (checking the literal's own freshly-constructed
				# tag at runtime, only ever one way) instead of a single
				# unconditional target, and that dispatch path (_emit_
				# dispatch_call) has no notion of a bound method RECEIVER at
				# all - confirmed via a real compile of Result[i32,str]'s own
				# .unwrap_or(5): a bound method call's own `self` argument
				# went missing from the emitted C call entirely. Using the
				# narrow leaf here instead keeps the operand's own static
				# type exactly as unambiguous as the literal itself is, and
				# the mismatch against the real (union-typed) parameter this
				# eventually needs to satisfy is resolved afterward, once the
				# winning target is actually known - see the Overload
				# dispatch's own post-resolution coercion step, below
				leaves = param.type.leaves()
				compatible_leaves = [ leaf for leaf in leaves if getattr( leaf, 'stem', None ) in compatible_stems ]
				if len( compatible_leaves ) != 1:
					continue
				matched_type = compatible_leaves[0]
			if not any( t is matched_type for t in candidate_types ):
				candidate_types.append( matched_type )

		if len( candidate_types ) > 1 and type( expr.value ) is int:
			# kind alone left more than one candidate (e.g. i8 AND i32 both
			# accept an int literal) - narrow further by whether the
			# literal's own MAGNITUDE actually fits each candidate's real
			# range (f(300) between f(x: i8)/f(x: i32) has only one answer,
			# not an ambiguity). Only ever NARROWS candidate_types when this
			# lands on exactly one match - if it eliminates every candidate,
			# or still leaves more than one (genuinely ambiguous even by
			# magnitude, e.g. two same-range types), candidate_types is left
			# untouched and the existing ambiguous/fallback paths below are
			# completely unaffected
			in_range = [ t for t in candidate_types if isinstance( t, Scalar ) and int_stem_range( t )[0] <= expr.value <= int_stem_range( t )[1] ]
			if len( in_range ) == 1:
				candidate_types = in_range
			elif expr.value >= 0 and { t.stem for t in in_range } == { 'usize', 'isize' }:
				# a non-negative literal fits BOTH usize and isize's own
				# real range (magnitude alone can never disambiguate this
				# one specific pair, unlike e.g. i8 vs i32) - real Python-
				# style negative-index overloads (see _resolve_index's own
				# comment) exist specifically so a NEGATIVE literal can
				# reach the isize leaf; a non-negative literal has no
				# reason to prefer it over the historical usize default,
				# so break the tie toward usize rather than surface a
				# confusing "ambiguous" error for the overwhelmingly
				# common case (every existing `x.__getitem__(0)`-shaped
				# call site with a literal 0..N index, e.g. lib/re.py's
				# own out.__getitem__(0)).
				candidate_types = [ t for t in in_range if t.stem == 'usize' ]
		if len( candidate_types ) == 1:
			return self._lower_expr( expr, candidate_types[0] )
		if len( candidate_types ) > 1:
			self.lowering.discovery.fail(
				f'ambiguous literal argument {ast.unparse(expr)} - matches more than one overload candidate type '
				f'({", ".join( t.qualname for t in candidate_types )}): {ast.unparse(node)}',
				node,
			)
		return self._lower_expr( expr, None ) # no candidate's parameter type is even plausible for this literal's kind - falls through to the existing error

	def _lower_overload_lambda_arg( self, expr: ast.Lambda, position: int|None, kw_name: str|None, candidates: list[Function], node: ast.AST ) -> ir.Operand:
		# mirrors _lower_overload_arg's own literal-argument handling above,
		# for the same reason: a bare `lambda ...: ...` has no type of its
		# own before a specific overload candidate is picked either -
		# _expr_Lambda needs a concrete Ptr[Callable[...]]/Closure[...]
		# expected_type up front to infer its parameter types. Scan
		# candidates for the ones whose declared parameter type at this
		# position is actually callable-shaped (the same two-check
		# CallableType-or-ClosureType test _expr_Lambda itself uses), and
		# use it unambiguously if exactly one candidate qualifies - e.g.
		# Result[T,E].unwrap's `errmsg: str` vs `errmsg: Ptr[Callable[[E],
		# str]]` candidates: only the latter is callable-shaped, so this
		# always resolves without ambiguity for that call
		def _callable_shape( t: Type ) -> Type|None:
			fn_type = self.lowering._type_resolver._callable_type_of( t )
			if fn_type is None and isinstance( t, ClosureType ):
				fn_type = t
			return fn_type

		candidate_types: list[Type] = []
		for fn in candidates:
			real_fn = fn.bound_to if fn.bound_to is not None else fn # same stub redirect as the literal path above - see its own comment
			if real_fn.parameters is None:
				continue
			param = (
				real_fn.parameters[position] if position is not None and position < len( real_fn.parameters ) else
				next( ( p for p in real_fn.parameters if p.stem == kw_name ), None )
			)
			if param is None or param.type is None:
				continue
			if _callable_shape( param.type ) is not None:
				matched_type = param.type
			else:
				# not directly callable-shaped - but a union-typed param
				# (e.g. Result[T,E].unwrap's own `errmsg: str|Ptr[Callable[
				# [E],str]]` implementation, once E substitutes concretely) can still
				# unambiguously accept a lambda through exactly one of its own
				# leaves, mirroring the literal path's identical union-leaf
				# handling above (same rationale: the lambda is lowered
				# against the matched LEAF's own narrow type, not the whole
				# union, so the resulting operand's static type stays exactly
				# as unambiguous as the lambda itself is)
				leaves = param.type.leaves()
				callable_leaves = [ leaf for leaf in leaves if _callable_shape( leaf ) is not None ]
				if len( callable_leaves ) != 1:
					continue
				matched_type = callable_leaves[0]
			if not any( t is matched_type for t in candidate_types ):
				candidate_types.append( matched_type )
		if len( candidate_types ) == 1:
			return self._lower_expr( expr, candidate_types[0] )
		if len( candidate_types ) > 1:
			self.lowering.discovery.fail(
				f'ambiguous lambda argument {ast.unparse(expr)} - matches more than one overload candidate type '
				f'({", ".join( t.qualname for t in candidate_types )}): {ast.unparse(node)}',
				node,
			)
		return self._lower_expr( expr, None ) # no candidate's parameter type is callable-shaped - falls through to _expr_Lambda's own "no expected Callable[...] context" error

	def _check_indirect_call_shape( self, node: ast.Call, fn_type: CallableType ) -> None:
		# shared *args/kwargs/arity validation for every _try_lower_indirect_
		# call callee shape below - split out so each shape's own branch can
		# call this at the point where it's already safe to fail loudly
		# (target may or may not be lowered yet, depending on the shape -
		# see each branch's own comment).
		if any( isinstance( a, ast.Starred ) for a in node.args ):
			self.lowering.discovery.fail( f'*args not supported yet: {ast.unparse(node)}', node )
		if node.keywords:
			self.lowering.discovery.fail( f'a Callable[...] call takes no keyword arguments: {ast.unparse(node)}', node )
		if len( node.args ) != len( fn_type.arg_types ):
			self.lowering.discovery.fail(
				f'{ast.unparse(node.func)}(...) takes {len(fn_type.arg_types)} argument(s), got {len(node.args)}: {ast.unparse(node)}',
				node,
			)

	def _try_lower_indirect_call( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand|None:
		# eq_fn(a, b) where eq_fn: Ptr[Callable[[A,B],R]] - a call THROUGH a
		# function-pointer VALUE, not a named Function/method lookup at all
		# (see PLAN_CALLABLE.md) - _resolve_callee has no way to express
		# this (it only ever returns a Function/Overload/Specialization/
		# _ReceiverDispatch, never an arbitrary Operand), so it's
		# recognized here instead, same "try a shape, None means try the
		# next one" convention as the construction recognizers above.
		#
		# Three callee shapes, in two different STRATEGIES:
		#
		# - Name (eq_fn(...)) and Attribute (obj.field(...), a Ptr[Callable
		#   [...]]-typed FIELD read off obj) determine "is this even
		#   callable" via a PURELY STATIC type lookup (no IR emitted)
		#   BEFORE ever lowering node.func for real - critical for the
		#   Attribute case specifically: _resolve_callee's own Attribute
		#   path (the fallback once every recognizer here returns None)
		#   lowers node.func.value ITSELF once it takes over, so lowering
		#   it here too and then bailing out on a non-match would double-
		#   evaluate a receiver with side effects.
		# - Everything else (a Call result, a Subscript result, ...) has no
		#   static type available without a real non-evaluating type-
		#   inference pass over arbitrary expressions - so instead this
		#   evaluates node.func ONCE, unconditionally, and inspects the
		#   REAL operand's type. This is still double-evaluation-safe: for
		#   any node.func shape other than Name/Attribute, _resolve_callee
		#   fails IMMEDIATELY, before evaluating anything at all (see its
		#   own `if not isinstance(func_node, ast.Attribute): fail(...)`
		#   guard) - so nothing downstream ever gets a second chance to
		#   evaluate the same expression, whether this turns out callable
		#   or not. A program that wasn't going to compile anyway (the
		#   non-callable case) doesn't need its abandoned evaluation to be
		#   free of side effects, since it never runs.
		if isinstance( node.func, ast.Name ):
			name = self.lowering.discovery.find_name_or_none( node.func.id )
			if not isinstance( name, Variable ):
				return None
			self.lowering._ensure_resolved( name )
			effective_type = self._narrowed_type_of_name( node.func.id, name.type )
			fn_type = self.lowering._type_resolver._callable_type_of( effective_type )
			if fn_type is None:
				return None
			self._check_indirect_call_shape( node, fn_type )
			target = self._lower_expr( node.func, None )
		elif isinstance( node.func, ast.Attribute ):
			receiver_type = self._static_type_of_value_expr( node.func.value )
			if receiver_type is None:
				return None
			if self.lowering._find_method( receiver_type, node.func.attr ) is not None:
				return None # a real method exists with this name - an ordinary method call, not a field call
			field = self.lowering._find_field( receiver_type, node.func.attr )
			if field is None:
				return None # no such field either - let _resolve_callee's own Attribute path give the accurate diagnostic
			fn_type = self.lowering._type_resolver._callable_type_of( field.type )
			if fn_type is None:
				return None
			self._check_indirect_call_shape( node, fn_type )
			target = self._lower_expr( node.func, None )
		else:
			# a Subscript here is NOT necessarily "index a runtime value" -
			# `some_generic_fn[T](...)`/`compiler.atomic_add[T](...)` is
			# namespace-resolved generic-call syntax (a Function/Specialization
			# looked up by NAME, exactly what _try_resolve_namespace already
			# recognizes for the construction-sugar recognizers above), not a
			# value to evaluate - confirmed by a real regression: evaluating
			# it here unconditionally hit _expr_Subscript's own "cannot take a
			# bare reference to a generic function" / "'sys' is not a value"
			# rejections for shapes that were never meant to reach _lower_expr
			# at all. Bail out (no evaluation attempted) whenever this static,
			# non-emitting lookup succeeds, leaving it for the SAME namespace-
			# based resolution _resolve_callee_target/the generic-call path
			# already handles, unchanged - only once it fails (this Subscript/
			# whatever really is a runtime value, e.g. t[0](...)/get_it()(...))
			# does evaluating it here become both correct and, per the same
			# reasoning as the other shapes above, double-evaluation-safe.
			if self.lowering._try_resolve_namespace( node.func ) is not None:
				return None
			candidate = self._lower_expr( node.func, None )
			fn_type = self.lowering._type_resolver._callable_type_of( candidate.type )
			if fn_type is None:
				return None
			self._check_indirect_call_shape( node, fn_type )
			target = candidate
		args = [ self._lower_expr( arg_node, arg_type ) for arg_node, arg_type in zip( node.args, fn_type.arg_types ) ]
		return self._emit_call_indirect( target, args, fn_type.return_type, expected_type )

	def _emit_call_indirect( self, target: ir.Operand, args: list[ir.Operand], return_type: Type, expected_type: Type|None ) -> ir.Operand:
		# shared by _try_lower_indirect_call/_try_lower_closure_call - a
		# NoneType-returning target (Callable[[...],None]/Closure[[...],
		# None]) needs dest=None in the emitted ir.CallIndirect (matching
		# ir.Call's own void-return convention - emitter_c.py's C
		# expression for the call itself has C type void there, and
		# `t = (void)(...)` is a real, confirmed compile error, not just
		# reasoning), but the CALLER here (a construction-sugar recognizer,
		# "None means try the next one") still needs to return a real,
		# non-None ir.Operand to signal "matched" - ir.Const(NoneType,
		# None) is a real value, the same representation an explicit
		# `x = None` already produces, distinct from Python's own None
		none_type = self.lowering.discovery.get_none_type()
		if return_type is none_type:
			self._emit( ir.CallIndirect( dest = None, target = target, args = args ))
			return ir.Const( type = none_type, value = None )
		dest = self._new_temp( expected_type or return_type )
		self._emit( ir.CallIndirect( dest = dest, target = target, args = args ))
		return dest

	def _lower_parameter_default( self, target: Function, param: Parameter, node: ast.Call ) -> ir.Operand:
		''' an omitted argument's default VALUE, lowered in the DEFINING
		function/class's own module/scope - not the caller's: a default
		expression can reference names visible where the function was
		DEFINED (matching _lower_allocate_fields's identical field-default
		pattern), and errors inside it should be located there too. Shared
		by every "fill in an omitted argument" call site (_lower_call_args,
		_fill_generic_call_defaults, the resolved-Overload-candidate tail)
		so they can't drift out of sync on this - confirmed as a real,
		pre-existing gap before this helper existed: the Overload-candidate
		tail was missing the module_context/scope_context switch entirely
		(only the other two had it), silently resolving a default value
		against the CALLER's own scope instead.

		Also updates self._owning_module for the duration - module_context
		is a genuine, correct switch to the callee's own module, but
		self._owning_module (what check_module_visibility's callers pass as
		the "who's really accessing this" module - see its own comment) is
		otherwise fixed for this WHOLE FunctionLowering pass and would
		still read as the CALLER's module here without this override.
		Confirmed as a real false positive: lib/pathlib.py's own `flavor:
		PathFlavor = _NATIVE_FLAVOR` parameter default, filled in at a
		call site living in a completely different module, got flagged as
		that OTHER module illegally reaching pathlib's own package-private
		constant - same bug shape as _lower_inline_call's own identical
		fix, just for default-argument filling instead of @inline
		splicing.

		node is the real CALL SITE (e.g. logger.debug("hi")) - needed so
		compiler.caller_line()/caller_file() (see _fold_caller_location) can
		capture ITS location rather than param.default's own. '''
		self.lowering._type_resolver.resolve_parameter_default( target, param )
		intrinsic = self.lowering._is_compiler_call( param.default )
		if intrinsic in ( 'caller_line', 'caller_file' ):
			return self._fold_caller_location( intrinsic, param, node )
		saved_owning_module = self._owning_module
		self._owning_module = self.lowering._find_module_for( target )
		# self._current_lineno gets the SAME "reads as the caller's" bug
		# self._owning_module's own comment above documents, but there's no
		# matching override to switch it TO here - it's only ever paired
		# with self._current_fn.file (Allocate.loc's own stamping, see
		# _emit), which never changes for the rest of this whole
		# FunctionLowering pass, so a default value's own line (target's
		# defining module, e.g. list[T]'s own `initial_capacity: usize = 8`)
		# permanently corrupts _current_lineno into a real line number from
		# a DIFFERENT file than self._current_fn.file names - confirmed via
		# a real repro: a bare list-literal's own $$__new__-bypassing
		# Allocate (_construct_generic_instance) reported "grap.mpy:408",
		# grap.mpy being the caller's file but 408 only a real line in
		# lib/builtins/__list.py (list[T].__init__'s own default value).
		# Save/restore around the SAME _lower_expr call as _owning_module.
		saved_lineno = self._current_lineno
		try:
			with self.lowering.discovery.module_context( self._owning_module ):
				with self.lowering.discovery.scope_context( target ):
					return self._lower_expr( param.default, param.type )
		finally:
			self._owning_module = saved_owning_module
			self._current_lineno = saved_lineno

	def _lower_call_args( self, target: Function, node: ast.Call, *, receiver_fills_first_param: bool = False ) -> tuple[list[ir.Operand],dict[str,ir.Operand]]:
		# shared by the plain call path (_lower_call's own else branch) and
		# _lower_generic_function_call: lowers positional/keyword args
		# straight against target's own already-concrete declared parameter
		# types, applying each param's move hook as it goes. NOT reused by
		# _lower_inferred_generic_call or _lower_class_generic_method_call -
		# both of those still need to INFER target's type params before a
		# parameter type is concrete enough to lower an argument against (in
		# _lower_inferred_generic_call's case, args are lowered with no
		# expected type at all, and move hooks apply in a separate pass
		# afterward instead), so forcing them through this helper would
		# change what expected_type each argument actually gets
		positional, keyword = self.lowering._match_call_args( target, node, receiver_fills_first_param = receiver_fills_first_param )
		args = []
		for param, expr in positional:
			operand = self._lower_expr( expr, param.type )
			self._apply_move_hook( param, operand, target.qualname, node )
			args.append( operand )
		kwargs = {}
		for param, expr in keyword:
			operand = self._lower_expr( expr, param.type )
			self._apply_move_hook( param, operand, target.qualname, node )
			kwargs[param.stem] = operand
		# fill in default values for any parameter that was not
		# explicitly provided by the call site (e.g. print(msg,
		# end='\n') called as print('hello') — end gets its
		# default lowered here as if the caller had passed it)
		given = { param.stem for param, _ in positional }
		given.update( kwargs.keys() )
		for param in target.parameters or []:
			if param.stem not in given and param.default is not None:
				# a construction call embedded in the default (`x: Foo =
				# Foo()`) needs the same eager __init__ pre-resolution an
				# ordinary body statement gets from _ReferenceResolver -
				# defaults live on fn.node.args, never walked by resolve_
				# function_body's fn.node.body loop, and are lowered here,
				# often before target's own turn on the compile queue ever
				# comes up - see _lower_parameter_default's own docstring
				kwargs[param.stem] = self._lower_parameter_default( target, param, node )
		return args, kwargs

	def _lower_inline_call( self, node: ast.Call, target: Function, receiver: ir.Operand|None, args: list[ir.Operand], kwargs: dict[str,ir.Operand], expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# PLAN_INLINE.md, generalized for multi-statement bodies - target.
		# is_inline: splice target's own body directly here instead of
		# ever emitting a real ir.Call. `args`/`kwargs` are already-lowered
		# operands (the caller already ran _lower_call_args, or the
		# interleaved generic lower_and_unify - same move-hook/argument-
		# lowering either way, only the tail differs). target may be a
		# plain Function, OR an already-monomorphized one (target.node was
		# deep-copied per Specialization by monomorphize.py - see its own
		# docstring), so target.node.body is always safe to read directly
		# here regardless of which caller reached this
		if id( target ) in self._inlining_stack:
			self.lowering.discovery.fail(
				f'@inline {target.qualname}: recursive inlining (directly or through another @inline function) is not supported: {ast.unparse(node)}',
				node,
			)
		# a discarded Result return (want_result=False, target.return_type
		# Result-shaped) is no longer rejected here - case 1 of the general
		# auto-or_throw() rule (see _auto_or_throw's own docstring) picks it
		# up at each of this method's own return points below instead
		# (_finish_call_result), same as the ordinary (non-inline) call
		# tails now do
		stmts = target.node.body
		if stmts and isinstance( stmts[0], ast.Expr ) and isinstance( stmts[0].value, ast.Constant ) and isinstance( stmts[0].value.value, str ):
			stmts = stmts[1:] # strip a leading docstring, same shape discovery.py's _is_inline_eligible_body already validated
		# the reentrancy guard wraps the WHOLE call - both branches below,
		# not just the single-expression case's own return-expression -
		# so a recursive @inline call reached from a pre-return statement
		# in the multi-statement path is caught identically
		self._inlining_stack.append( id( target ))
		# splicing target's own body statements directly into THIS
		# FunctionLowering instance's own pass (no separate FunctionLowering
		# instance/module_context push for target itself - that's the whole
		# point of @inline) means self._owning_module would otherwise still
		# read as the CALLER's own module for code lexically written in
		# target's module - wrong for check_module_visibility specifically:
		# inlining is a pure optimization and must stay transparent to
		# access control, exactly like a private method inlined into a
		# caller in any other language doesn't retroactively become public.
		# Confirmed as a real false positive, not just theory: Ptr.__str__
		# (an @inline dunder living in builtins) reaches builtins' own
		# _ptr_hex_digits from its spliced body - without this save/
		# restore, that got attributed to whatever unrelated module
		# happened to be calling str(ptr), e.g. __main__.
		saved_owning_module = self._owning_module
		self._owning_module = target.module
		# a failure while lowering target's own SPLICED body (below) is
		# reported at THAT body's own source location (target.file/whatever
		# node inside it failed) - correct for an ordinary function, but
		# actively misleading for @inline: the body isn't a separate call
		# frame, it's textually spliced in right here, so a caller has no
		# way to tell "which of possibly many len(...) call sites in my own
		# code triggered this" from the error alone (confirmed via a real
		# repro: a type mismatch inside builtins.len[T]'s own `return t.
		# __len__()` reported ONLY lib/builtins/__init__.py:<its own line>,
		# never grap.mpy's own call site, nor which concrete T). Appends one
		# extra, purely additive note (never replaces/edits the original
		# diagnostic) pointing at THIS call's own real site plus each
		# argument's own concrete type - the closest a generic @inline site
		# can get to "here's what T actually was" without full type-param-
		# binding plumbing, and enough to answer both complaints at once.
		errors_before = len( self.lowering.discovery.errors.errors )
		call_site_fn = self._current_fn
		try:
			if len( stmts ) > 1:
				# the splice's own pre-return statements bind self/params
				# (and declare their own alpha-renamed locals) directly on
				# the shared self._cfg - unlike provisional.names (a fresh,
				# single-use, discarded Function), self._cfg._live is NOT
				# reverted internally by _splice_multi_statement_inline_body
				# itself, so it's done here instead: wholesale snapshot/
				# restore around the whole call - see set_live()'s own
				# docstring for why a full revert is safe here
				saved_live = self._cfg.live_snapshot()
				try:
					return self._splice_multi_statement_inline_body( node, target, receiver, args, kwargs, expected_type, want_result, stmts )
				finally:
					self._cfg.set_live( saved_live )

			bindings: dict[str,ir.Operand] = {} if receiver is None else { 'self': receiver }
			for i, param in enumerate( target.parameters or [] ):
				bindings[param.stem] = args[i] if i < len( args ) else kwargs[param.stem]

			# each binding becomes a REAL local Variable, registered under its
			# ordinary name ('self', a parameter's own stem) directly into
			# target.names - not just an _expr_Name-level shortcut - because
			# discovery.find_name is reached from more than one place while
			# lowering a Call (e.g. _try_resolve_namespace, used by the
			# construction-call recognizers to probe whether `self.foo(...)`
			# might be construction sugar, BEFORE ordinary attribute/method
			# resolution ever runs) - anything less than a real registry entry
			# left those other paths seeing an unresolved 'self'/param name
			# (confirmed by a real repro, not just reasoning: self.__len__()
			# inside an inlined body failed exactly this way, from inside a
			# construction-sugar probe, not from _expr_Name at all).
			#
			# the Variable's own .stem (what emitter_c.py actually declares as
			# a C local, keyed by NAME not by object identity - see its own
			# "declared" set) is deliberately NOT 'self'/the parameter's own
			# stem - reusing those would silently collide with and overwrite
			# the ENCLOSING function's own real `self`/parameter of the same
			# name the moment one method's @inline body gets spliced into
			# another method's own body. _inline_binding_id makes every
			# splice's own bindings unique instead.
			#
			# no _cfg_assign/incref here, deliberately - this must behave
			# exactly like an ordinary (non-@move) function parameter already
			# does at a REAL call boundary: borrowed, no incref at the
			# boundary, no independent decref responsibility (the caller's own
			# argument operand keeps whatever cleanup it already had, e.g. an
			# argument Temp's own DeleteTemp - untouched by any of this). A
			# bare ir.Assign against a fresh Variable is exactly that: a named
			# alias for the call's own duration, nothing more.
			#
			# when the operand is ALREADY a Variable (by far the common case -
			# a bare-name receiver/argument, e.g. b.get_len()/some_result.
			# is_ok()), it's registered directly, no fresh copy and no Assign
			# at all - true zero overhead, and what makes the "compiles
			# identically to writing the callee's body directly at the call
			# site" guarantee exact, not just "close". Only a genuinely
			# computed operand (a Temp from a sub-expression like make_box().
			# get_len(), or a Const) needs the synthesized-local fallback -
			# both to give it a referenceable name at all (Temp/Const aren't
			# Name subtypes, discovery.find_name's registry requires one - see
			# above) and to guarantee it's evaluated exactly once even if the
			# spliced body references self/that parameter more than once
			saved: dict[str,object] = {}
			saved_live: dict[str,bool] = {}
			bound_ids: set[int] = set() # see _incref_aliasing_return's own `force` doc - every self/parameter binding here is always treated as borrowed
			for stem, operand in bindings.items():
				if isinstance( operand, Variable ):
					fresh = operand
				else:
					fresh = Variable(
						stem = f'$inline{self._inline_binding_id}${stem}',
						qualname = f'{target.qualname}$$inline{self._inline_binding_id}${stem}',
						file = target.file, line = target.line,
						type = operand.type,
					)
					self._inline_binding_id += 1
					self._emit( ir.Assign( dest = fresh, src = operand ))
				bound_ids.add( id( fresh ))
				saved[stem] = target.names.get( stem )
				target.names[stem] = fresh
				# liveness is keyed by `stem` (the literal 'self'/parameter
				# name the spliced body's own ast.Name nodes reference it
				# by, via target.names) NOT fresh.stem (which differs for
				# the synthesized-local fallback above, and even for the
				# zero-copy case is the CALLER's own variable's stem, e.g.
				# 'b' for a `b.get_len()` receiver, not 'self'). Bypasses
				# _cfg_assign like the rest of this binding deliberately
				# does (see this method's own "no _cfg_assign/incref here"
				# comment above) - still unconditionally bound right here.
				# Saved/restored the same shadow-and-restore way target.
				# names itself is, in case `stem` collides with an outer
				# name that wasn't actually live before this splice.
				saved_live[stem] = self._cfg.is_live( stem )
				self._cfg.mark_live( stem )

			self._inline_param_alias_ids.update( bound_ids )
			return_expr = stmts[-1].value
			module = self.lowering._find_module_for( target )
			try:
				with self.lowering.discovery.module_context( module ):
					with self.lowering.discovery.scope_context( target ):
						result = self._lower_expr( return_expr, expected_type or target.return_type )
						result = self._incref_aliasing_return( return_expr, result, force = id( result ) in bound_ids, wrap_fresh = True )
			finally:
				self._inline_param_alias_ids.difference_update( bound_ids )
				for stem, was_live in saved_live.items():
					if not was_live:
						self._cfg.unmark_live( stem )
				for stem, old in saved.items():
					if old is None:
						target.names.pop( stem, None )
					else:
						target.names[stem] = old
			return self._finish_call_result( node, result, want_result )
		except CompileError:
			if len( self.lowering.discovery.errors.errors ) > errors_before:
				self.lowering.discovery.errors.error(
					f'(the error above happened while inlining {target.qualname}({self._describe_inline_call_args( receiver, target, args, kwargs )}), '
					f'spliced in from this call)',
					call_site_fn.file if call_site_fn is not None else None, node.lineno,
				)
			raise
		finally:
			self._inlining_stack.pop()
			self._owning_module = saved_owning_module

	def _describe_inline_call_args( self, receiver: ir.Operand|None, target: Function, args: list[ir.Operand], kwargs: dict[str,ir.Operand] ) -> str:
		''' "self: SomeClass, t: SomeOtherClass" - each argument's own REAL,
		already-lowered concrete type, in target's own declared parameter
		order. For a generic target (e.g. builtins.len[T]) this is the
		closest available stand-in for "what did T resolve to" without full
		type-param-binding plumbing threaded all the way out here: T itself
		is never named directly, but the argument bound to a T-typed
		parameter shows its own concrete type, which is exactly what a
		reader needs to answer that question by inspection. '''
		parts: list[str] = []
		if receiver is not None:
			parts.append( f'self: {receiver.type.qualname if receiver.type else "?"}' )
		for i, param in enumerate( target.parameters or [] ):
			if i < len( args ):
				parts.append( f'{param.stem}: {args[i].type.qualname if args[i].type else "?"}' )
			elif param.stem in kwargs:
				parts.append( f'{param.stem}: {kwargs[param.stem].type.qualname if kwargs[param.stem].type else "?"}' )
		return ', '.join( parts )

	def _splice_multi_statement_inline_body( self, node: ast.Call, target: Function, receiver: ir.Operand|None, args: list[ir.Operand], kwargs: dict[str,ir.Operand], expected_type: Type|None, want_result: bool, stmts: list[ast.stmt] ) -> ir.Operand|None:
		# PLAN_INLINE.md multi-statement generalization - target's own
		# body (already docstring-stripped by _lower_inline_call, the only
		# caller) has more than the single `return <expr>` statement the
		# original @inline design handled. discovery.py's _is_inline_
		# eligible_body already guarantees `stmts` ends in exactly one,
		# un-nested `return <expr>`, no other Return anywhere else in it,
		# no defer/errdefer, and no reassignment of self/a parameter
		# anywhere among the pre-return statements - this method doesn't
		# re-check any of that. The reentrancy guard was already pushed by
		# _lower_inline_call, covering this whole splice.
		bindings: dict[str,ir.Operand] = {} if receiver is None else { 'self': receiver }
		for i, param in enumerate( target.parameters or [] ):
			bindings[param.stem] = args[i] if i < len( args ) else kwargs[param.stem]

		# a fresh, per-call-site provisional Function - independent deep-
		# copied .node, independent .names dict, never touching `target`
		# itself (unlike the single-statement path's target.names
		# monkeypatch above - no save/restore needed anywhere in this
		# path, provisional is single-use, discarded once this call
		# returns). type_params=[]/args=[] is a no-op substitution: target
		# is already type-parameter-free by the time it reaches here
		# regardless of whether it was originally generic (monomorphize.py
		# already did that substitution before _lower_inline_call was ever
		# reached - see PLAN_RETURN_INFERENCE.md/monomorphize_function)
		provisional = self.lowering._monomorphizer._build_monomorphized_function( target, [], [], target.qualname )

		# run BEFORE alpha-renaming: match-statement desugaring (rewrite 2
		# - there is no _stmt_Match anywhere in this file, so a match
		# statement in a spliced body can only ever lower after this runs)
		# needs to see the ORIGINAL names (the alpha-renamer below only
		# understands plain ast.Name, never match-pattern capture shapes);
		# generic-call tagging (rewrite 3) is genuinely per-copy already -
		# needed for a nested generic call inside the pre-return
		# statements to resolve against THIS call site's own bindings, not
		# some other call site's
		self.lowering._type_resolver.resolve_function_body( provisional )

		provisional_stmts = provisional.node.body
		if provisional_stmts and isinstance( provisional_stmts[0], ast.Expr ) and isinstance( provisional_stmts[0].value, ast.Constant ) and isinstance( provisional_stmts[0].value.value, str ):
			provisional_stmts = provisional_stmts[1:]
		pre_return_stmts = provisional_stmts[:-1]
		return_stmt = provisional_stmts[-1]

		# alpha-rename every local the pre-return statements themselves
		# declare (a Store-context Name that isn't already self/a
		# parameter - those are bound via the names-dict substitution
		# below instead, never renamed here, since that mechanism still
		# keys off the literal original name) to a fresh, globally-unique
		# name, reusing the same $inline{id}$stem convention self/param
		# bindings already use (so the two can never collide). No
		# shadowing subtlety needed: this language has no block scoping
		# (cfg.py's own docstring: "structural, not a reference scan"),
		# and a nested def/lambda's own free variables are already
		# rejected elsewhere (PLAN_LAMBDA.md) - a flat, uniform rename
		# across every occurrence, Store and Load alike, is exactly
		# correct here, not an approximation. In-place mutation of ast.
		# Name.id suffices (no NodeTransformer needed) since provisional.
		# node is already a private, freshly-deep-copied-per-call-site
		# object - this only ever changes a string field, never
		# restructures the tree
		fl = self
		rename_map: dict[str,str] = {}
		class _LocalCollector( ast.NodeVisitor ):
			def visit_FunctionDef( self, fd: ast.FunctionDef ) -> None:
				pass
			def visit_AsyncFunctionDef( self, fd: ast.AsyncFunctionDef ) -> None:
				pass
			def visit_Lambda( self, lam: ast.Lambda ) -> None:
				pass
			def visit_Name( self, n: ast.Name ) -> None:
				if isinstance( n.ctx, ast.Store ) and n.id not in bindings and n.id not in rename_map:
					rename_map[n.id] = f'$inline{fl._inline_binding_id}${n.id}'
					fl._inline_binding_id += 1
		collector = _LocalCollector()
		for stmt in pre_return_stmts:
			collector.visit( stmt )

		if rename_map:
			class _LocalRenamer( ast.NodeVisitor ):
				def visit_FunctionDef( self, fd: ast.FunctionDef ) -> None:
					pass
				def visit_AsyncFunctionDef( self, fd: ast.AsyncFunctionDef ) -> None:
					pass
				def visit_Lambda( self, lam: ast.Lambda ) -> None:
					pass
				def visit_Name( self, n: ast.Name ) -> None:
					if n.id in rename_map:
						n.id = rename_map[n.id]
			renamer = _LocalRenamer()
			for stmt in pre_return_stmts:
				renamer.visit( stmt )
			renamer.visit( return_stmt ) # a later pre-return statement, or the return-expression itself, may reference an earlier pre-return-declared local

		# bind self/params into the PROVISIONAL's own names dict - same
		# logic the single-statement path above uses for target.names,
		# just no save/restore needed (provisional is single-use)
		bound_ids: set[int] = set() # see _incref_aliasing_return's own `force` doc - every self/parameter binding here is always treated as borrowed
		for stem, operand in bindings.items():
			if isinstance( operand, Variable ):
				fresh = operand
			else:
				fresh = Variable(
					stem = f'$inline{self._inline_binding_id}${stem}',
					qualname = f'{target.qualname}$$inline{self._inline_binding_id}${stem}',
					file = target.file, line = target.line,
					type = operand.type,
				)
				self._inline_binding_id += 1
				self._emit( ir.Assign( dest = fresh, src = operand ))
			bound_ids.add( id( fresh ))
			provisional.names[stem] = fresh
			# liveness keyed by `stem` (see _lower_inline_call's own
			# identical single-statement-path comment) - the caller
			# (_lower_inline_call) reverts self._cfg's ENTIRE live set once
			# this whole splice returns, so no per-stem save/restore is
			# needed here, unlike provisional.names/that other path
			self._cfg.mark_live( stem )
		self._inline_param_alias_ids.update( bound_ids ) # see _lower_inline_call's own identical comment - removed again once this whole splice returns, in the try/finally around its own module_context below

		# early/nested-return + defer/errdefer/.or_return() generalization -
		# a splice-local "epilogue" scope for the pre-return statements: an
		# early return, or a .or_return()/checked-arithmetic early exit,
		# reached from one of them must never jump into/return from the
		# CALLER's own real epilogue - it needs its OWN local landing point.
		# result_var carries whichever value flowed through an early exit
		# (the splice-local analogue of self._return_value_var); exited_flag
		# (armed alongside it, same mechanism defer/errdefer's own flags
		# use) lets the tail below tell "early exit vs normal fallthrough"
		# apart once everything converges - see cfg.py's push_inline_scope()
		# and current_epilogue_label()/return_()'s own comments for the CFG
		# half of this, and ir.OrReturn.inline_exit/ir.OrJump.exited_flag
		# for how or_return()/checked-arithmetic feed into it
		# PLAN_RETURN_INFERENCE.md's own @inline variant (_infer_return_
		# only_type_params_inline) reaches here with target.return_type set
		# to Python None as a DELIBERATE SENTINEL (not none_type - the real
		# NoneType class), specifically so the trailing return-expression's
		# own _lower_expr(..., None) call can take its own natural type,
		# later read back via result.type to discover R. None of the new
		# early-exit machinery below can run in that state - result_var/
		# result would need a REAL type up front, which is exactly the one
		# thing not known yet. This is safe to skip entirely rather than
		# work around: _is_eager_return_inferable_body (the ONLY gate that
		# lets return-only inference even be attempted) already requires
		# EXACTLY ONE reachable return, so a body reaching here with this
		# sentinel can never have an early return to support in the first
		# place - only .or_return()/checked-arithmetic in a pre-return
		# statement remains a real (if narrow) hazard, still explicitly
		# rejected below, exactly as the single, blanket guard this
		# replaces always did for every multi-statement splice
		none_type = self.lowering.discovery.get_none_type()
		noreturn_type = self.lowering.discovery.get_intrinsics()['NoReturn']
		supports_early_exit = target.return_type is not None
		result_var: Variable|None = None
		exited_flag: Variable|None = None
		merge_label: str|None = None
		bool_cls: Type|None = None
		if supports_early_exit:
			result_var = (
				Variable(
					stem = f'$inline{self._inline_binding_id}$result', qualname = f'{target.qualname}$$inline{self._inline_binding_id}$result',
					file = target.file, line = target.line, type = target.return_type,
				)
				if target.return_type not in ( none_type, noreturn_type )
				else None
			)
			bool_cls = self.lowering.discovery.find_name( 'bool', node )
			exited_flag = Variable(
				stem = f'$inline{self._inline_binding_id}$exited', qualname = f'{target.qualname}$$inline{self._inline_binding_id}$exited',
				file = target.file, line = target.line, type = bool_cls,
			)
			self._inline_binding_id += 1
			merge_label = self._new_label( 'inline_merge' )

		module = self.lowering._find_module_for( target )
		try:
			with self.lowering.discovery.module_context( module ):
				with self.lowering.discovery.scope_context( provisional ):
					# the ONE place self._current_fn is ever reassigned in this
					# file - narrowly scoped to this window, restored in a
					# finally. Needed because _stmt_AnnAssign/_stmt_Assign's
					# fresh-declaration branch registers a new local into
					# self._current_fn (see their own code) while their
					# "already exists?" check instead goes through discovery.
					# find_name_or_none (walking discovery.scope_stack, which
					# module_context/scope_context above already point at
					# `provisional`) - without this reassignment those two
					# would disagree: a pre-return local would silently
					# register into the CALLER's own namespace (self._current_
					# fn, unless reassigned, stays whatever the caller's own
					# top-level function is - confirmed by grep, it's assigned
					# exactly once, in __init__, and never touched anywhere
					# else in this file), corrupting any later caller-side
					# reference to a same-named local; and reassigning that
					# same pre-return local a second time within the SAME
					# spliced body would fail to find its own first
					# registration, creating a second, independent binding
					# instead of a replace (a silent decref/leak, not just a
					# cosmetic issue - cfg.py's own fresh-vs-replace machinery
					# depends on finding the SAME Variable object both times).
					# Making self._current_fn and the active scope_context
					# point at the same `provisional` object for this whole
					# window fixes both at once.
					#
					# result_var/exited_flag are both given a real, flat,
					# unconditional declaration/init RIGHT HERE - before the
					# pre-return statements (and therefore before any .or_
					# return()/checked-arithmetic early exit nested inside
					# emitter_c.py's own hand-emitted C `{ }` blocks - see ir.
					# DeclareLocal's own docstring) could otherwise become
					# result_var's first, block-scoped-and-therefore-unsafe
					# write. exited_flag has a trivial default (False) an
					# ordinary ir.Assign already declares safely; result_var's
					# type has no generic default, hence DeclareLocal
					scope_label: str|None = None
					if supports_early_exit:
						assert exited_flag is not None and bool_cls is not None and merge_label is not None
						if result_var is not None:
							self._emit( ir.DeclareLocal( variable = result_var ))
						self._emit( ir.Assign( dest = exited_flag, src = ir.Const( type = bool_cls, value = False )))
						scope_label = self._cfg.push_inline_scope()
						self._inline_scope_vars.append(( result_var, exited_flag, merge_label ))
					outer_fn = self._current_fn
					outer_prelude = self._in_inline_splice_prelude
					self._current_fn = provisional
					self._in_inline_splice_prelude = True
					try:
						for stmt in pre_return_stmts:
							# mirrors FunctionLowering.run()'s own identical
							# per-statement recovery boundary - one bad
							# statement doesn't stop the rest of this splice
							# from being lowered (and error-collected)
							try:
								self._lower_stmt( stmt )
							except CompileError:
								continue
					finally:
						self._current_fn = outer_fn
						self._in_inline_splice_prelude = outer_prelude

					if not supports_early_exit:
						# PLAN_RETURN_INFERENCE.md's own @inline variant - see
						# this method's own top-of-function comment. No scope
						# was pushed, nothing to merge - the trailing return-
						# expression's own natural type IS the answer being
						# discovered here, exactly as the pre-existing
						# single-statement/original multi-statement code always
						# computed it
						result = self._lower_expr( return_stmt.value, expected_type )
						result = self._incref_aliasing_return( return_stmt.value, result, force = id( result ) in bound_ids, wrap_fresh = True )
						return self._finish_call_result( node, result, want_result )

					assert scope_label is not None and exited_flag is not None and merge_label is not None
					# current_epilogue_label()'s own fallback target once
					# nothing shallower within THIS splice qualified (push_
					# inline_scope()'s own label) - an inline-unwind return_()
					# call reached during the splice already replayed
					# everything itself and jumps straight past this, to
					# merge_label below (see _stmt_Return/_consume_checked_
					# result's own splice branches). Neither label is a real
					# jump target unless the splice body actually contained an
					# early exit reaching one of those two branches (a plain
					# multi-statement @inline body with none, e.g. Ptr.__str__,
					# never goes near either) - gated on InlineScope.captured,
					# same "don't declare a label nothing goto's" reasoning
					# build_epilogue_ladder() already uses for a real function's
					# own shared epilogue (a real, confirmed -Wunused-label
					# otherwise, suite-wide)
					if self._cfg.inline_scope_captured():
						self._emit( ir.Label( name = scope_label ))
					for instr in self._cfg.build_inline_scope_ladder( lambda: self._build_is_err_check( node )):
						self._emit( instr )
					was_captured = self._cfg.pop_inline_scope()
					self._inline_scope_vars.pop()

					# early exit vs normal fallthrough - both converge into ONE
					# result operand from here, same "shared dest temp, two
					# Assign sites, converge at one label" shape _expr_IfExp
					# already uses for Python's own ternary. self._current_fn/
					# _in_inline_splice_prelude are already restored to the
					# REAL caller above, before this point - the trailing
					# return-expression's own .or_return()/checked-arithmetic
					# behavior is therefore unchanged from the single-statement
					# case (validates and jumps against the CALLER's own
					# epilogue/return type, exactly as already tested), while
					# scope_context(provisional) stays active so it can still
					# resolve pre-return-declared locals it references
					if was_captured:
						self._emit( ir.Label( name = merge_label ))
					result = self._new_temp( target.return_type )
					normal_label = self._new_label( 'inline_normal' )
					converge_label = self._new_label( 'inline_converge' )
					self._emit( ir.JumpIfFalse( cond = exited_flag, target = normal_label ))
					if result_var is not None:
						self._emit( ir.Assign( dest = result, src = result_var ))
					self._emit( ir.Jump( target = converge_label ))
					self._emit( ir.Label( name = normal_label ))
					trailing_value = self._lower_expr( return_stmt.value, target.return_type )
					trailing_value = self._incref_aliasing_return( return_stmt.value, trailing_value, force = id( trailing_value ) in bound_ids, wrap_fresh = True )
					self._emit( ir.Assign( dest = result, src = trailing_value ))
					# trailing_value's own ownership (if it's a bare temp - e.g.
					# the Result.Ok(x) construction temp a trailing `return
					# Result.Ok(x)` produces) just transferred into `result`
					# above via the plain ir.Assign - untrack it, or whatever
					# later cleans up STILL-pending temps (_flush_pending_temps,
					# called by _lower_stmt's own post-statement wrapper once
					# this whole splice call returns) would emit a SECOND,
					# unconditional RC-check for it outside the "normal" arm's
					# own guard - reading trailing_value's memory even on the
					# early-exit path, where it was never assigned at all (a
					# real uninitialized-read bug, not just a redundant decref -
					# confirmed by a real repro under MSVC's /RTC1). Exactly the
					# same concern _stmt_Return's own identical transfer already
					# guards against via this same call
					self._cfg.untrack_temp( trailing_value )
					self._emit( ir.Label( name = converge_label ))
		finally:
			self._inline_param_alias_ids.difference_update( bound_ids )
		return self._finish_call_result( node, result, want_result )

	def _lower_call( self, node: ast.Call, expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		super_method_name = self._super_call_shape( node )
		if super_method_name is not None:
			return self._lower_super_call( node, super_method_name, expected_type, want_result )
		match self.lowering._is_compiler_call( node ):
			case 'caller_line' | 'caller_file' as name:
				self.lowering.discovery.fail(
					f"compiler.{name}() is only valid as a parameter's default value "
					f"(e.g. `def f(x: i32 = compiler.{name}()) -> None:`), not as a general expression: {ast.unparse(node)}",
					node,
				)

			case 'sizeof':
				result = self._lower_compiler_sizeof( node, expected_type )
				return result if want_result else None

			case 'is_rc':
				result = self._lower_compiler_is_rc( node, expected_type )
				return result if want_result else None

			case 'refcount':
				result = self._lower_compiler_refcount( node, expected_type )
				return result if want_result else None

			case '__debug_quarantine__':
				result = self._lower_compiler_debug_quarantine( node, expected_type )
				return result if want_result else None

			case 'error':
				self._lower_compiler_error( node, expected_type )

			case 'checked_add' | 'wrapped_add' | 'saturated_add' | 'checked_sub' | 'wrapped_sub' | 'saturated_sub' | \
				'checked_mul' | 'wrapped_mul' | 'saturated_mul' | 'checked_floordiv' | 'wrapped_floordiv' | 'saturated_floordiv' | \
				'checked_mod' | 'wrapped_mod' | 'saturated_mod' | 'checked_truediv' | 'wrapped_truediv' | \
				'checked_shl' | 'wrapped_shl' | 'saturated_shl':
				# every compiler.<mode>_<kind>(a, b) intrinsic shares one
				# lowering - see _lower_compiler_checked_binop and
				# _CHECKED_BINOP_OPCODES/_CHECKED_FLOAT_BINOP_OPCODES for how
				# (kind, mode) picks the actual opcode per operand type
				name = self.lowering._is_compiler_call( node )
				mode, _sep, kind = name.partition( '_' )
				result = self._lower_compiler_checked_binop( node, name, kind, mode, expected_type )
				return result if want_result else None

			case 'cast':
				result = self._lower_compiler_cast( node, expected_type )
				return result if want_result else None

			case 'checked_convert':
				result = self._lower_compiler_checked_convert( node, expected_type )
				return result if want_result else None

			case 'bitand' | 'bitor' | 'bitxor' | 'rshift':
				result = self._lower_compiler_bitwise( node, self.lowering._is_compiler_call( node ), expected_type )
				return result if want_result else None

			case 'cmp_eq' | 'cmp_ne' | 'cmp_lt' | 'cmp_le' | 'cmp_gt' | 'cmp_ge':
				result = self._lower_compiler_cmp( node, self.lowering._is_compiler_call( node ), expected_type )
				return result if want_result else None

			case 'checked_ptr_add' | 'wrapped_ptr_add' | 'saturated_ptr_add' | \
				'checked_ptr_sub' | 'wrapped_ptr_sub' | 'saturated_ptr_sub':
				name = self.lowering._is_compiler_call( node )
				mode, _sep, rest = name.partition( '_' )
				kind = 'add' if rest == 'ptr_add' else 'sub'
				result = self._lower_compiler_ptr_binop( node, name, kind, mode, expected_type )
				return result if want_result else None

			case 'ptr_sub_dist':
				result = self._lower_compiler_ptr_diff( node, expected_type )
				return result if want_result else None

			case '__raw_alloc__':
				result = self._lower_compiler_raw_alloc( node, expected_type )
				return result if want_result else None

			case 'addrof':
				result = self._lower_compiler_addrof( node, expected_type )
				return result if want_result else None

			case 'atomic_load':
				result = self._lower_compiler_atomic_load( node, expected_type )
				return result if want_result else None

			case 'atomic_add':
				result = self._lower_compiler_atomic_rmw( node, expected_type, ir.AtomicRMWOp.ADD, want_result = want_result )
				return result if want_result else None

			case 'atomic_sub':
				result = self._lower_compiler_atomic_rmw( node, expected_type, ir.AtomicRMWOp.SUB, want_result = want_result )
				return result if want_result else None

			case 'atomic_exchange':
				result = self._lower_compiler_atomic_rmw( node, expected_type, ir.AtomicRMWOp.EXCHANGE, want_result = want_result )
				return result if want_result else None

			case 'atomic_compare_exchange':
				result = self._lower_compiler_atomic_compare_exchange( node, expected_type )
				return result if want_result else None

			case 'cexpr':
				result = self.lowering._lower_compiler_cexpr( node, expected_type )
				return result if want_result else None

			case 'c_field':
				result = self._lower_compiler_c_field( node, expected_type )
				return result if want_result else None

			case 'c_field_set':
				self._lower_compiler_c_field_set( node, expected_type )
				return None

			case 'c_field_addr':
				result = self._lower_compiler_c_field_addr( node, expected_type )
				return result if want_result else None

			case 'fetch_unicode_table':
				result = self.lowering._lower_compiler_fetch_unicode_table( node )
				return result if want_result else None

			case 'fetch_windows_zones_table':
				result = self.lowering._lower_compiler_fetch_windows_zones_table( node )
				return result if want_result else None

			case 'format_f64':
				result = self._lower_compiler_format_f64( node, expected_type )
				return result if want_result else None

			case 'is_nan':
				result = self._lower_compiler_is_nan_or_inf( node, expected_type, 'is_nan' )
				return result if want_result else None

			case 'is_inf':
				result = self._lower_compiler_is_nan_or_inf( node, expected_type, 'is_inf' )
				return result if want_result else None

			case 'parse_f64':
				result = self._lower_compiler_parse_f64( node, expected_type )
				return result if want_result else None


		if isinstance( node.func, ast.Attribute ) and node.func.attr == 'or_return':
			# <result_expr>.or_return() - recognized by AST shape alone,
			# BEFORE _resolve_callee/_attr_lookup_callable ever look for a
			# real declared 'or_return' method on the receiver's class -
			# there is none to find (discovery.py's _parse_function now
			# rejects any user-written `def or_return(...)` outright, on
			# ANY class, since one could never actually be called - see its
			# own comment). Without this, a receiver whose class has no
			# such method (the ordinary, correct case - nobody is expected
			# to write one) failed to resolve at all ("'or_return' is not
			# callable on ..."), confirmed by a real repro: this bug
			# predates and is unrelated to that new rejection, which just
			# makes the fix here airtight instead of merely "the common
			# case works". _lower_or_return itself already validates the
			# receiver is actually Result[_,_]-shaped (and that no
			# arguments were given) - a non-Result receiver correctly still
			# fails there, with the same message as before.
			# snapshotted BEFORE lowering the receiver expression, not after -
			# _flush_new_pending_temps needs to know exactly which pending
			# temps belong to the RECEIVER's own build-up (from here onward)
			# versus an OUTER, still-in-progress expression this whole
			# `.or_return()` call is only PART of (e.g. the first half of a
			# chained `k + str(':') + v.or_return()` concatenation, whose own
			# still-needed pending temp must never be touched - see that
			# helper's own docstring for the real bug this guards against)
			receiver_pending_start = len( self._pending_temps )
			receiver = self._lower_expr( node.func.value, None )
			return self._lower_or_return( node, receiver, want_result, receiver_pending_start = receiver_pending_start )

		if isinstance( node.func, ast.Attribute ) and node.func.attr == 'or_throw':
			# <result_expr>.or_throw() - same recognition shape as or_return()
			# just above (see its own comment) - discovery.py rejects a
			# user-written `def or_throw(...)` outright, on any class, the
			# same way
			receiver_pending_start = len( self._pending_temps )
			receiver = self._lower_expr( node.func.value, None )
			return self._lower_or_throw( node, receiver, want_result, receiver_pending_start = receiver_pending_start )

		# each recognizer returns None (not an error) when this call doesn't
		# match its own construction-sugar shape at all, falling through to
		# the next; a real error inside a matched shape (e.g. a malformed
		# __allocate__ call) still raises/records normally
		construction_recognizers = (
			self._try_lower_generator_allocate_call,
			self._try_lower_allocate_call,
			self._try_lower_construct_call,
			self._try_lower_scalar_construct_call,
			self._try_lower_indirect_call,
			self._try_lower_closure_call,
		)
		for recognizer in construction_recognizers:
			allocate_dest = recognizer( node, expected_type )
			if allocate_dest is not None:
				return allocate_dest if want_result else None

		# type_resolver.py's own generic-call resolution
		# (_ReferenceResolver.visit_Call) may already have tagged this call
		# with its resolved, monomorphized callee (an ordinary, concrete
		# Function - never a Specialization) - when present, it's
		# authoritative and skips _resolve_callee/the Specialization
		# branches below entirely, so a call resolved there never touches
		# Specialization on this side at all. Absent (any call that pass
		# left untagged, generic or not) falls through to the exact same
		# resolution this always did
		resolved_callee = getattr( node, 'resolved_callee', None )
		if resolved_callee is not None:
			target, receiver = resolved_callee, None
		else:
			target, receiver = self._resolve_callee( node.func )
		if receiver is not None and isinstance( target, Function ) and ( target.is_static or target.is_classmethod ):
			# self.static_method(...) - _resolve_callee's own Attribute
			# fallback always computes a receiver for ANY dotted callee (it
			# has no way to know staticness before resolving the attribute
			# itself), but ir.Call.receiver's own docstring already says a
			# @staticmethod/@classmethod call takes none at all ("None for
			# a free function, staticmethod, or classmethod call") - this
			# is the one place that promise wasn't kept, and it showed up
			# as a real C compile error (an extra `self` argument at the
			# call site that the callee's own prototype never declared).
			# ClassName.static_method(...) never hits this: it already
			# resolves through _resolve_callee_target's namespace-lookup
			# path instead, which never computes a receiver in the first
			# place - only the self.-qualified spelling needs the null-out
			receiver = None
		if receiver is not None:
			self.lowering.schedule( receiver.type )

		if receiver is not None and isinstance( target, Function ) and target.is_move:
			# @move on a method means calling it consumes/invalidates self -
			# cfg.py's own move() (already the exact mechanism _apply_move_hook
			# uses for move[T] PARAMETER arguments) needs to run here too, for
			# the RECEIVER: nothing else ever transitions the CALLER's own
			# ownership-tracking state for a receiver on an @move call -
			# confirmed via a real double-free (bytearray.release(), called
			# through str.from_cstr's own move[bytearray] parameter: release()
			# only invalidates ITS OWN self.__data sentinel, guarding against
			# a double-free of the byte buffer, but does nothing about the
			# CALLER's own binding, which still got an ordinary Decref at
			# scope exit on top of that - two teardown paths for one struct).
			# Also correctly rejects calling an @move method through a merely
			# BORROWED receiver (move()'s own OWNED precondition), which
			# was never checked before either. Caught/re-recorded here for the
			# same reason _apply_move_hook does - move() raises a bare,
			# unrecorded CompileError (see its own comment).
			try:
				instructions = self._cfg.move( receiver, target_qualname = target.qualname, param_stem = 'self' )
			except CompileError as e:
				self.lowering.discovery.fail( str( e ), node )
			for instr in instructions:
				self._emit( instr )

		if isinstance( target, ( Function, Overload )) and target.stem in self.lowering._RESULT_CONSUMING_METHODS and isinstance( receiver, Variable ):
			# .is_ok()/.is_err()/.unwrap(msg)/.unwrap_or(default) - like
			# or_return() above, these aren't given their own IR shape;
			# they're ordinary method dispatch (unwrap_or in particular
			# resolves through the Overload branch below, for its `default:
			# T` stub), so recognition has to happen here by stem + class
			# identity rather than at a single shared call site the way
			# or_return's own _lower_or_return is. Placed before any of the
			# dispatch branches below (rather than only in the plain-
			# Function "final else" tail) so it applies uniformly regardless
			# of which branch actually ends up lowering the call
			target_cls_base = target.cls.base if isinstance( target.cls, Specialization ) else target.cls
			if target_cls_base is self.lowering.discovery.find_name( 'Result', node ):
				self._cfg.clear_result( receiver.stem )

		# NOTE: no receiver move for unwrap/unwrap_or here (this used to move a
		# Temp receiver - xs.__getitem__(0).unwrap(msg) - so its RC payload
		# wasn't decref'd twice, back when unwrap/unwrap_or returned
		# self.data.v_Ok as a BORROW, AND back when a temp Result's own payload
		# happened to never actually get tracked/decref'd at all - see cfg.py's
		# rc_leaves()/_refcount_instructions (the resolve_type wiring above,
		# in this file's own CFGState construction) for that separate,
		# previously-missing half. Both are fixed now: unwrap/unwrap_or COPY
		# the payload into a local before returning it (lib/builtins/
		# __init__.py), which increfs it, so the returned value is a
		# genuinely owned reference, and the
		# receiver temp's ordinary end-of-expression decref is exactly right -
		# moving it here would suppress that decref and leak the payload. The
		# same incref is also what fixes the persistent-Variable-receiver
		# double-free (g0 = xs.__getitem__(0); g0.unwrap(msg)) that the old
		# Temp-only move never covered - so both receiver shapes now balance.

		if isinstance( target, _ReceiverDispatch ):
			return self._lower_union_receiver_call( node, target, receiver, expected_type, want_result )

		# `.or_return()` no longer reaches here at all - it's recognized and
		# fully handled at the top of this method, before target/receiver
		# were even resolved (see that check's own comment for why)

		if isinstance( target, Specialization ) and isinstance( target.base, Function ):
			return self._lower_generic_function_call( node, target, receiver, expected_type, want_result )

		if isinstance( target, Function ) and target.type_params:
			return self._lower_inferred_generic_call( node, target, receiver, expected_type, want_result )

		# target.cls can legitimately BE a Specialization now (monomorphized_
		# function sets a monomorphized method's own .cls to one) - Specialization
		# has no .type_params of its own, so this must not read it directly;
		# getattr's default (None/falsy) correctly means "not this branch",
		# since a receiver that already pinned down concrete class args (the
		# only way target.cls ends up a Specialization here) already went
		# through _attr_lookup_callable's own substitution - nothing left to
		# infer
		if isinstance( target, Function ) and not target.type_params and target.cls is not None and getattr( target.cls, 'type_params', None ):
			return self._lower_class_generic_method_call( node, target, receiver, expected_type, want_result )

		# set only by the Overload branch below, when a winning stub's own
		# more specific return type applies to THIS call's real arguments
		# (see the Overload branch's own comment on stub_covers_call) - a
		# genuinely narrower Type than target.return_type's own real,
		# emittable return type. Applied AFTER the call is emitted (see the
		# shared tail below, "narrowed_return_type is not None"): the call
		# itself always still targets the ORIGINAL, shared Function (never a
		# replace()'d copy with a mismatched declared return type), and the
		# narrowing is realized by statically extracting the matching leaf
		# out of the call's real, wide-typed result - sound specifically
		# because stub_covers_call already proved this call's own arguments
		# can never actually produce the OTHER (narrowed-away) member at
		# runtime, so no tag check is needed, mirroring
		# _maybe_unwrap_union_arg's identical no-runtime-check extraction
		# for a union-typed CALL ARGUMENT already known to match one leaf
		narrowed_return_type: Type|None = None
		if isinstance( target, Overload ):
			# a bare literal argument has no type of its own before a
			# specific implementation is chosen - _lower_overload_arg tries
			# each candidate's declared parameter type at that position,
			# using it unambiguously if exactly one is even plausible for
			# the literal's own kind (a string literal never plausibly
			# matches an i32 parameter, etc.) and failing clearly rather
			# than guessing if more than one genuinely could
			candidates = [ *target.stubs, *target.implementations ]
			for fn in candidates:
				assert fn.resolve is None, f'internal compiler error - {fn.qualname} was not resolved before overload dispatch'
			if any( fn.is_move for fn in candidates ):
				# the receiver-move-hook (below, gated on isinstance(target,
				# Function)) never fires for an Overload target at all -
				# calling an @move-decorated overload alternative would
				# neither track receiver ownership correctly nor error, so
				# it's rejected outright, matching this same plan's
				# identical policy for a union-typed receiver
				self.lowering.discovery.fail(
					f'calling an @move-decorated overload of {target.qualname} is not supported: {ast.unparse(node)}',
					node,
				)

			def _peel_move( expr: ast.expr ) -> tuple[ast.expr,bool]:
				# move(...) sugar isn't a real name anywhere - _check_move_
				# argument (only reachable once a single concrete Function
				# target is already chosen, never for an Overload group)
				# already recognizes this exact shape; mirrored here so it
				# at least PARSES against an overload group too, before a
				# winning candidate is even known. Which positions/kwargs
				# were wrapped is remembered (moved_pos/moved_kw below) so
				# it can be validated/applied once resolve_call picks a
				# single concrete winner, below.
				if isinstance( expr, ast.Call ) and isinstance( expr.func, ast.Name ) and expr.func.id == 'move':
					if len( expr.args ) != 1 or expr.keywords:
						self.lowering.discovery.fail( f'move(...) takes exactly one argument: {ast.unparse(expr)}', node )
					return expr.args[0], True
				return expr, False

			peeled_args = [ _peel_move( a ) for a in node.args ]
			args = [ self._lower_overload_arg( e, i, None, candidates, node ) for i, ( e, _ ) in enumerate( peeled_args ) ]
			moved_pos = [ was_moved for _, was_moved in peeled_args ]
			if any( kw.arg is None for kw in node.keywords ):
				self.lowering.discovery.fail( f'**kwargs not supported yet: {ast.unparse(node)}', node )
			peeled_kwargs = { kw.arg: _peel_move( kw.value ) for kw in node.keywords }
			kwargs = { name: self._lower_overload_arg( e, None, name, candidates, node ) for name, ( e, _ ) in peeled_kwargs.items() }
			moved_kw = { name: was_moved for name, ( _, was_moved ) in peeled_kwargs.items() }
			arg_types = [ op.type for op in args ]
			kwarg_types = { name: op.type for name, op in kwargs.items() }
			call_slots: list[int|str] = [ *range( len( arg_types )), *kwarg_types.keys() ]
			arg_leaves: dict[int|str,tuple[Type,...]] = {
				**{ i: tuple( t.leaves() ) for i, t in enumerate( arg_types ) },
				**{ name: tuple( t.leaves() ) for name, t in kwarg_types.items() },
			}

			# an @overload group declared inside a generic CLASS (e.g.
			# Result[T,E].unwrap_or's `default: T` stub) is now pre-
			# substituted by monomorphize_class itself whenever `target`
			# was reached through a concrete class specialization - see
			# Monomorphizer._substituted_overload. target.stubs/
			# .implementations are ALREADY the correctly monomorphized
			# Functions in that case (each one's own .cls a Specialization,
			# the same convention monomorphized_function already uses for
			# a single non-overloaded generic method), so no per-call-site
			# substitution is needed here at all anymore - this used to
			# reconstruct that same substitution by hand from the
			# RECEIVER's own type instead (isinstance(receiver.type,
			# Specialization)), which only worked because the receiver
			# hadn't been resolved to its real ClassLike yet; detecting
			# "was this group substituted" now just reads the already-
			# substituted candidate's own .cls, the same signal
			# monomorphized_function already exposes everywhere else
			substituted = bool( candidates ) and isinstance( candidates[0].cls, Specialization )

			def _resolve_original( fn: Function ) -> Function:
				if not substituted:
					return fn
				# when a stub won the overload resolution, `fn` is the
				# stub's own `bound_to` (the real, already-monomorphized
				# implementation) - the stub has a more specific return
				# type than the impl (e.g. T vs T|None), so use the
				# stub's return type while still calling through to the impl.
				# `bound_to` is a static, unconditional relationship (one
				# stub always resolves to the same implementation), so it
				# alone can't tell whether THIS call's own arguments
				# actually matched the stub's narrower signature or fell
				# through to the implementation's own wider one (e.g.
				# unwrap_or()'s zero-argument form only ever matches the
				# plain `default: T|None = None` impl, never the `default:
				# T` stub bound to it) - stub_covers_call re-checks that
				# against this call's real argument types before narrowing
				winning_stub = next( ( s for s in target.stubs if s.bound_to is fn ), None )
				if (
					winning_stub is not None and winning_stub.return_type is not fn.return_type
					and overload_resolution.stub_covers_call(
						winning_stub, call_slots, arg_leaves, self.lowering._type_resolver._same_type,
					)
				):
					# only actually narrow (build a distinct replace()'d copy)
					# when the stub's own return type is genuinely a DIFFERENT
					# object from fn's own - when T is already itself Optional
					# (e.g. Result[i32|None,E].unwrap_or), the stub's `T` and
					# the impl's `T|None` substitute to the exact SAME interned
					# union object (discovery.py's _get_or_create_union
					# memoizes by flattened/deduped leaf set - see its own
					# flattening fix), so there's nothing to narrow. Skipping
					# the copy in that case matters for more than avoiding
					# useless work: the copy returned here is handed straight
					# to self.lowering._ensure_resolved()/schedule() below as
					# if it were its own real, independent compile unit - fully
					# separately LOWERED (type_resolver.py's
					# resolve_function_body + Lowering.lower_function) under
					# the narrowed return_type - but mangle_function_qualname
					# (emitter_c.py) mangles purely off fn.qualname +
					# fn.overload_group, with no notion of "this Function
					# object is a distinct return-type view of another one" -
					# so a call site combining a zero-argument call (schedules
					# the original, wide-return-type fn) with an explicit-
					# argument call to the same group previously always built
					# and scheduled a SEPARATE replace()'d copy, even on the
					# many calls where there was nothing left to actually
					# narrow, producing two independently-lowered Functions
					# sharing one mangled C symbol - a genuine duplicate-
					# definition/argument-type-mismatch at the real C compile
					# stage. Confirmed via a real compile of
					# Result[i32|None,str]'s own .unwrap_or(42): before this
					# fix, a fresh identical-looking copy was always built and
					# independently scheduled regardless, producing exactly
					# that redefinition.
					return replace( fn, return_type = winning_stub.return_type )
				return fn

			try:
				branches, resolved = overload_resolution.resolve_call(
				target.stubs, target.implementations, arg_types, kwarg_types,
				qualname = target.qualname, same_type = self.lowering._type_resolver._same_type,
				protocol_conforms = self.lowering._type_conforms_to_protocol,
			)
			except CompileError as e:
				# resolve_call is a pure function of types with no
				# AST/Discovery reference by design - it raises unrecorded,
				# this is where a location actually gets attached and it
				# lands in the collector
				self.lowering.discovery.fail( str( e ), node )
			if branches:
				if any( moved_pos ) or any( moved_kw.values() ):
					# which branch actually runs is a RUNTIME decision
					# (ConditionalDispatch) - move(...)'s ownership transfer
					# needs a single, statically-known target (matching this
					# plan's own policy on @move through a union receiver/
					# overload group elsewhere) - not attempted here
					self.lowering.discovery.fail(
						f'move(...) through a runtime-dispatched overload group is not supported: {ast.unparse(node)}',
						node,
					)
				branches = [ ConditionalDispatch( conditions = b.conditions, function = _resolve_original( b.function )) for b in branches ]
				resolved = _resolve_original( resolved )
				if resolved.type_params or any( b.function.type_params for b in branches ):
					# a GENERIC candidate as one branch (or the trailing
					# default) of a runtime-dispatched overload group: which
					# concrete C symbol to call has to be fixed at compile
					# time (this compiler has no vtable/runtime-polymorphic
					# dispatch concept anywhere) - reachable only when the
					# call's own union-typed argument has leaves routing to
					# more than one overload, at least one of them generic.
					#
					# Whatever leaf(s) of the union still reach one particular
					# generic branch/default get resolved STATICALLY, either
					# explicitly (a non-default branch always has its own
					# runtime condition(s) - see overload_resolution.
					# resolve_call) or by elimination (the trailing default:
					# the call's real leaves at that slot, minus whatever
					# every OTHER branch's own condition already claims
					# there). The common case is exactly ONE leaf - T is
					# knowable at compile time same as any other generic
					# call, and this branch is monomorphized in place before
					# ever reaching _lower_conditional_dispatch (which
					# unconditionally schedules every branch's target as an
					# ordinary, already-concrete compile unit).
					#
					# When 2+ leaves can still legitimately reach ONE generic
					# branch (e.g. a 3+-member union where only one member
					# has a concrete overload - every OTHER member falls
					# through to the SAME generic default, each needing its
					# own distinct monomorphization) - _expand_dispatch_target
					# splits that ONE branch into one new, individually-
					# concrete branch PER leaf, each with its own extra
					# runtime condition pinning exactly that leaf. Exactly
					# one overall entry (see `default_fn` below) stays the
					# trailing, unconditioned default - by construction, once
					# every OTHER entry's own condition has been tested and
					# excluded, only that one's own territory can remain, so
					# it never needs a check of its own either way.
					# every leaf already spoken for by a CONCRETE candidate,
					# keyed by the real call-site operand it came from - not
					# just whichever branches happen to carry an explicit
					# runtime condition: a concrete candidate that ends up as
					# the trailing default has its own condition computed
					# then discarded by resolve_call (see its own "own
					# leaves" comment - a default never needs one), so
					# reading conditions alone under-counts. A concrete
					# function's own declared parameter type unambiguously
					# IS the one leaf it handles, condition or not - reading
					# .parameters directly instead is both simpler and
					# correct for every concrete candidate, branch or default.
					claimed: dict[int,list[Type]] = {}
					for b in ( *branches, ConditionalDispatch( conditions = [], function = resolved )):
						if b.function.type_params:
							continue
						for p in b.function.parameters or []:
							if p.type is None:
								continue
							operand = self.lowering._dispatch_operand_for_param( node, b.function, p, args, kwargs )
							claimed.setdefault( id( operand ), [] ).append( p.type )
					new_branches: list[ConditionalDispatch] = []
					for b in branches:
						if not b.function.type_params:
							new_branches.append( b )
							continue
						original_params = b.function.parameters or []
						known = { id( p ): t for p, t in b.conditions }
						for extra_conditions, monomorphized in self._expand_dispatch_target( node, b.function, known, args, kwargs, claimed ):
							new_params = monomorphized.parameters or []
							remapped = self._remap_conditions( original_params, new_params, b.conditions )
							new_branches.append( ConditionalDispatch( conditions = remapped + extra_conditions, function = monomorphized ))
					branches = new_branches
					if resolved.type_params:
						# the LAST expansion becomes the new trailing default
						# (its own extra_conditions are dropped - see the
						# comment above); every OTHER expansion is a genuine
						# new conditioned branch, appended after the ones
						# above (lowest priority, matching resolve_call's own
						# "default is whatever's left once every real branch
						# is excluded" convention)
						*extra, ( _, default_fn ) = self._expand_dispatch_target( node, resolved, {}, args, kwargs, claimed )
						for extra_conditions, monomorphized in extra:
							new_branches.append( ConditionalDispatch( conditions = extra_conditions, function = monomorphized ))
						resolved = default_fn
				return self._lower_conditional_dispatch( node, branches, resolved, receiver, args, kwargs, expected_type, want_result )
			winning_stub = next( ( s for s in target.stubs if s.bound_to is resolved ), None )
			if (
				winning_stub is not None and winning_stub.return_type is not resolved.return_type
				and overload_resolution.stub_covers_call(
					winning_stub, call_slots, arg_leaves, self.lowering._type_resolver._same_type,
				)
			):
				narrowed_return_type = winning_stub.return_type
			target = resolved
			if target.type_params:
				# resolve_call picked a GENERIC candidate (see overload_
				# resolution.py's own wildcard/TypeVar handling) - target is
				# still the abstract, unspecialized Function here, never a
				# real compile unit of its own (only ITS monomorphized
				# Specialization ever is - see _emit_generic_call's own
				# scheduling). Deliberately skips _ensure_resolved(target)
				# below (unlike the concrete case) - resolve_call() already
				# resolved every group member internally, and
				# _ensure_resolved's own unconditional schedule() would
				# register this bare abstract Function as a real compile
				# unit, reaching the emitter with a still-bare TypeVar
				# parameter (confirmed via a real repro: emitter_c.py's
				# c_type() crashes with NotImplementedError on the TypeVar
				# itself). Route through the same inference+monomorphization
				# machinery a bare generic-function call uses instead of
				# falling into the rest of this branch, which assumes a
				# concrete target.parameters (union-coercion, move
				# validation, default-arg filling) and the shared call-
				# emission tail below, neither of which apply to an
				# unspecialized generic target.
				return self._lower_overload_generic_call( node, target, receiver, args, kwargs, expected_type, want_result )
			self.lowering._ensure_resolved( target ) # resolve_call() already resolved every group member internally - this just schedules the chosen one

			# args/kwargs were lowered by _lower_overload_arg BEFORE target was
			# known, each literal deliberately typed as narrowly as possible
			# (see _lower_overload_arg's own comment on why: so resolve_call's
			# own dispatch decision, above, sees the literal's true, single-
			# leaf type rather than a whole union "either member is possible"
			# type). Now that target is fixed, coerce any operand that's still
			# narrower than target's own real declared parameter type UP into
			# it - mirrors the identical coercion an ORDINARY, non-overloaded
			# call gets for free by lowering its arguments directly against
			# the (already statically known) target's parameter types; an
			# Overload target never goes through that path (see _lower_call_
			# args, only used by the plain-target `else` branch below), so it
			# needs this equivalent applied explicitly, once, here
			for i, param in enumerate( target.parameters or [] ):
				if i >= len( args ):
					break
				if args[i].type is not param.type and isinstance( param.type, TaggedUnion ):
					args[i] = self._coerce_into_union( args[i], param.type, node )
			for param in target.parameters or []:
				if param.stem in kwargs and kwargs[param.stem].type is not param.type and isinstance( param.type, TaggedUnion ):
					kwargs[param.stem] = self._coerce_into_union( kwargs[param.stem], param.type, node )

			# now that a single concrete winner is known, validate move(...)
			# usage against ITS OWN parameters (mirroring _check_move_
			# argument's identical checks) and actually transition
			# ownership (mirroring _apply_move_hook) - both were previously
			# unreachable for an Overload target, see this plan's own item 3
			for i, param in enumerate( target.parameters or [] ):
				if i >= len( args ):
					break
				was_moved = moved_pos[i] if i < len( moved_pos ) else False
				if param.is_move and not was_moved:
					self.lowering.discovery.fail(
						f"{target.qualname}: parameter {param.stem!r} is move[{param.type.qualname}] - "
						f"call site must pass move(...): {ast.unparse(node)}",
						node,
					)
				elif was_moved and not param.is_move:
					self.lowering.discovery.fail(
						f"{target.qualname}: parameter {param.stem!r} is not move[T] - "
						f"call site must not wrap it in move(...): {ast.unparse(node)}",
						node,
					)
				if param.is_move:
					self._apply_move_hook( param, args[i], target.qualname, node )
			for param in target.parameters or []:
				if param.stem not in moved_kw:
					continue
				was_moved = moved_kw[param.stem]
				if param.is_move and not was_moved:
					self.lowering.discovery.fail(
						f"{target.qualname}: parameter {param.stem!r} is move[{param.type.qualname}] - "
						f"call site must pass move(...): {ast.unparse(node)}",
						node,
					)
				elif was_moved and not param.is_move:
					self.lowering.discovery.fail(
						f"{target.qualname}: parameter {param.stem!r} is not move[T] - "
						f"call site must not wrap it in move(...): {ast.unparse(node)}",
						node,
					)
				if param.is_move:
					self._apply_move_hook( param, kwargs[param.stem], target.qualname, node )

			# fill in default values for any of target's OWN parameters the
			# call site didn't supply - mirrors _lower_call_args's identical
			# tail for the plain (non-Overload) path just below, which this
			# branch never goes through (an Overload target builds args/
			# kwargs itself, above, straight from node.args/node.keywords,
			# with no equivalent step). Without this, a zero-argument
			# unwrap_or() call (its own `default: T|None = None` impl
			# parameter never supplied) reached real emission with no
			# 'default' entry in instr.kwargs at all, crashing emitter_c.py's
			# _emit_call_args with a bare KeyError
			given = { p.stem for i, p in enumerate( target.parameters or [] ) if i < len( args ) }
			given.update( kwargs.keys() )
			for param in target.parameters or []:
				if param.stem not in given and param.default is not None:
					# see _lower_parameter_default's own docstring for why
					# this is needed - a construction call embedded in this
					# default otherwise never gets its __init__ eagerly
					# pre-resolved. This call site used to lower the default
					# expression directly, with no module_context/scope_
					# context switch at all (unlike its own "mirrors _lower_
					# call_args's identical tail" comment claimed) - a real,
					# pre-existing bug, not just missing the later privacy-
					# check fix: it silently resolved a default value against
					# the CALLER's own scope instead of the callee's.
					kwargs[param.stem] = self._lower_parameter_default( target, param, node )
		else:
			self.lowering._resolve_call_target( target )
			# a Scalar-registered method's receiver isn't threaded through
			# call.args at all - _match_call_args's own
			# receiver_fills_first_param excludes target's first positional
			# parameter from call-site matching accordingly (see its own
			# comment); the receiver is spliced back in as an ordinary
			# leading positional argument just below.
			receiver_fills_first_param = receiver is not None and isinstance( target, Function ) and target.cls is None
			args, kwargs = self._lower_call_args( target, node, receiver_fills_first_param = receiver_fills_first_param )

		if receiver is not None and isinstance( target, Function ) and target.cls is None:
			# ir.Call's own receiver field is for real bound-method calls
			# only (an RCClass/CStruct method with "self" already excluded
			# from .parameters) - a Scalar-attached free function (see the
			# comment above) takes the receiver as an ordinary LEADING
			# positional argument instead, confirmed by a real KeyError
			# crash on ordinary `f.__str__()` call syntax before this fix.
			# _lower_method_call above (used by f-string dunder-dispatch/
			# format-spec call sites) already carries this exact fix for
			# its own narrower set of callers; this is the same fix for the
			# general call-lowering path every other Scalar-attached-method
			# call site (including ordinary user-written `receiver.method()`
			# syntax) actually goes through.
			args = [ receiver ] + args
			receiver = None

		if isinstance( target, Function ) and target.is_inline:
			# PLAN_INLINE.md - reaches this shared tail from either the
			# plain (non-generic, non-Overload) `else` branch above, the
			# resolved_callee pre-tag (type_resolver.py's own generic-call
			# resolution, ~ this method's own top), or a receiver-based
			# generic-class method already monomorphized by _attr_lookup_
			# callable before dispatch even started (e.g. some_result.
			# is_ok()/.is_err() - see PLAN_INLINE.md's own "traced through
			# _attr_lookup_callable" note). Never reached with is_inline set
			# from the Overload branch above - @inline+@overload is
			# rejected at discovery time, so a resolved group member is
			# never is_inline
			return self._lower_inline_call( node, target, receiver, args, kwargs, expected_type, want_result )

		self.lowering.schedule( target.return_type )
		for param in target.parameters or []:
			self.lowering.schedule( param.type )

		# a bare `foo()` statement whose return value is a Result - _stmt_
		# Expr is the only caller that ever passes want_result=False for a
		# call used as a full statement (every other _lower_call caller
		# threads want_result through from ITS OWN caller instead), so this
		# is the "value produced, immediately discarded, never even bound to
		# a name" case from the plan's validation table - case 1 of the
		# general auto-or_throw() rule (_finish_call_result, below) now
		# picks it up instead of hard-rejecting; force_result routes this
		# call through the SAME dest-producing machinery `want_result=True`
		# already uses below, purely so there's something to auto-consume.
		# every other call-emission tail that can produce a Result from a
		# bare statement carries this same force_result copy:
		# _emit_generic_call (generic calls), _infer_return_only_type_params_
		# inline (return-only-inferred generics), _lower_inline_call/
		# _finish_call_result (@inline splices), and _lower_conditional_
		# dispatch/_lower_union_receiver_call (runtime union-argument/
		# receiver dispatch) - so this is no longer a gap, just this tail's
		# own copy of a check every call-lowering path shares
		force_result = not want_result and cfg.is_result_type( target.return_type )
		# same "produce a dest so there's something to auto-consume" idea,
		# but for a discarded NON-Result RC return (e.g. `xs.pop().unwrap(msg)
		# -> str` as a bare statement) - _finish_call_result's own release
		# path (see its docstring) needs a real dest to release; without
		# this the `else` branch below passes dest=None, so the callee still
		# constructs and returns the object, it's just never captured -
		# confirmed as a real leak via that exact repro.
		discard_rc = (
			not want_result and not force_result
			and target.return_type is not None and target.return_type.is_rc()
		)

		if want_result or force_result or discard_rc:
			target_return_type = target.return_type
			# a Specialization of a TaggedUnion base (e.g. an unmonomorphized
			# Result[usize,IndexError]) is just as "already the expected
			# union" as a TaggedUnion instance itself - Result's own class
			# body (discovery.py's _parse_ClassDef_TaggedUnion) makes its
			# base a TaggedUnion, so a bare isinstance( _, TaggedUnion )
			# check misses every generic-union return that hasn't been
			# individually monomorphized yet, which target_return_type
			# usually hasn't been at this point (nothing upstream forces it -
			# self.lowering.schedule() below only enqueues it for later
			# compilation)
			target_return_type_base = target_return_type.base if isinstance( target_return_type, Specialization ) else target_return_type
			if narrowed_return_type is not None:
				# the raw call always targets `target`'s own REAL, wide return
				# type (see narrowed_return_type's own comment, above where
				# it's declared) - sizing dest directly to expected_type/
				# narrowed_return_type here, the way the ordinary branches
				# below do, would declare dest with a C type the actual
				# callee never produces (its real, emitted C prototype still
				# returns the whole union struct). The narrowing is instead
				# realized AFTER the call: extract the matching leaf out of
				# the wide result via the SAME no-runtime-check extraction
				# _maybe_unwrap_union_arg already uses for a union-typed call
				# ARGUMENT known to match one leaf - sound here for the
				# identical reason (stub_covers_call already proved this
				# call's arguments can never produce the other member)
				dest = self._new_temp( target_return_type )
				self._emit( ir.Call( dest = dest, target = target, receiver = receiver, args = args, kwargs = kwargs ))
				return self._finish_call_result( node, self._maybe_unwrap_union_arg( dest, narrowed_return_type, owning = True ), want_result )
			if isinstance( expected_type, TaggedUnion ) and target_return_type is not None and not isinstance( target_return_type_base, ( TaggedUnion, TypeVar )):
				# a call whose own return type is a plain leaf (e.g. str)
				# flowing into a T|None-typed slot - dest must be typed as
				# target_return_type (what the emitted ir.Call's C signature
				# actually returns), not expected_type, or dest's C
				# declaration wouldn't match the value assigned into it.
				# _lower_expr's own post-hoc coercion (_coerce_into_union,
				# right after this call returns) then wraps it into the union
				# - see its own comment. Deliberately narrower than "prefer
				# target_return_type whenever it's concrete": when
				# target_return_type is ITSELF (a Specialization of) a
				# TaggedUnion (e.g. a Result[usize,IndexError]-returning call
				# assigned into an already Result[usize,IndexError]-typed
				# local), the two are the same union from context but not
				# necessarily the same object - Specialization instances for
				# one generic instantiation aren't interned across
				# independent resolutions, so forcing dest to expected_type
				# there (the `else` below, unchanged from before this fix)
				# keeps dest identity-compatible with whatever already
				# expects it, instead of tripping _coerce_into_union's
				# identity check with a whole (non-leaf) union value it
				# would wrongly treat as a leaf needing wrapping
				dest = self._new_temp( target_return_type )
			else:
				# a bare TypeVar expected_type is never a legitimate coercion
				# target here - by this point `target` is always already a
				# concrete, resolved Function (never itself mid-generic-
				# dispatch - see the "already tagged with its resolved,
				# monomorphized callee" comment above), so target_return_type
				# is the call's own real, concrete return type. A stray bare
				# TypeVar hint (e.g. _lower_allocate_fields threading a
				# generic field's own still-abstract declared type down as
				# the expected_type for lowering that field's VALUE
				# expression - _infer_allocate_type_args then needs that
				# value's own REAL type back, not this hint bounced straight
				# through) would otherwise silently override dest with a
				# meaningless, unresolvable type instead of what the callee
				# actually returns.
				#
				# Beyond the TypeVar case, expected_type is only safe to use
				# for dest when it's the SAME type as target_return_type (just
				# possibly a different Specialization/TupleType representation
				# of the identical instantiation - _same_type's own docstring)
				# - using it whenever merely non-None, regardless of whether it
				# actually matches, let a genuinely mismatched declared local
				# type (`x: i32 = a_result_returning_call()`) silently retype
				# dest to i32 while the callee's real C prototype still returns
				# the whole Result struct, producing invalid C AND skipping
				# _coerce_or_check_operand's own mismatch rejection below
				# entirely (operand.type came back already equal to
				# expected_type, so its "still mismatched past this point"
				# check never even saw a mismatch to catch). A genuine
				# mismatch here now falls through with dest correctly typed as
				# target_return_type instead, so the coercion-or-rejection
				# tail gets an honest look at it.
				if isinstance( expected_type, TypeVar ):
					dest = self._new_temp( target_return_type )
				elif expected_type is not None and target_return_type is not None and not self.lowering._type_resolver._same_type( target_return_type, expected_type ):
					dest = self._new_temp( target_return_type )
				else:
					dest = self._new_temp( expected_type or target_return_type )
			self._emit( ir.Call( dest = dest, target = target, receiver = receiver, args = args, kwargs = kwargs ))
			return self._finish_call_result( node, dest, want_result )
		else:
			self._emit( ir.Call( dest = None, target = target, receiver = receiver, args = args, kwargs = kwargs ))
			return None

	def _lower_conditional_dispatch( self, node: ast.Call, branches: list[ConditionalDispatch], default: Function, receiver: ir.Operand|None, args: list[ir.Operand], kwargs: dict[str,ir.Operand], expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# a union-typed argument's runtime tag decides which overload
		# implementation actually runs (e.g. len(copy_from) where
		# copy_from: bytes|bytearray resolves to two candidates, bytes and
		# bytearray). Reuses the same tag/data/v_<member> machinery match
		# statements use (UnionStorage.get) - branches are tried in
		# priority order, falling through to `default` (no test needed -
		# it's whatever's left once every more specific branch is excluded).
		# `receiver` is the SAME instance operand for every branch (an
		# Overload group is either entirely bound methods sharing one
		# receiver, or entirely receiver-less free functions/statics - never
		# a mix) - already scheduled/move-tracked by _lower_call before ever
		# reaching here, so it's just threaded through unchanged into each
		# branch's own ir.Call, same as `args`/`kwargs` already are.
		self.lowering._ensure_resolved( default )
		for branch in branches:
			self.lowering._ensure_resolved( branch.function )

		# case 1 of the general auto-or_throw() rule (_finish_call_result,
		# below) - this is its own call-emission tail (union-argument
		# runtime dispatch), never reaches _lower_call's shared tail, so
		# needs the identical force_result copy _emit_generic_call already
		# carries for the generic-call tail
		force_result = not want_result and cfg.is_result_type( default.return_type )
		# discarded plain (non-Result) RC return - see _lower_call's own
		# identical discard_rc for why this still needs a real dest
		discard_rc = not want_result and not force_result and default.return_type is not None and default.return_type.is_rc()
		produce_result = want_result or force_result or discard_rc
		dest = self._new_temp( expected_type or default.return_type ) if produce_result else None
		end_label = self._new_label( 'dispatch_end' )
		for branch in branches:
			next_label = self._new_label( 'dispatch_next' )
			self._lower_dispatch_tests( node, branch.function, branch.conditions, args, kwargs, next_label )
			self._emit_dispatch_call( branch.function, receiver, args, kwargs, dest, produce_result )
			self._emit( ir.Jump( target = end_label ))
			self._emit( ir.Label( name = next_label ))
		self._emit_dispatch_call( default, receiver, args, kwargs, dest, produce_result )
		self._emit( ir.Label( name = end_label ))
		return self._finish_call_result( node, dest, want_result )

	def _lower_dispatch_tests( self, node: ast.AST, target: Function, conditions: list[tuple[Parameter,Type]], args: list[ir.Operand], kwargs: dict[str,ir.Operand], next_label: str ) -> None:
		# a branch's conditions are ANDed together - emits one Cmp +
		# JumpIfFalse per condition, all targeting next_label, which is
		# already a short-circuit AND with no combined boolean value to
		# build at all (same trick _expr_BoolOp uses, just directly in IR
		# since these operands are already lowered)
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		for param, leaf_type in conditions:
			operand = self.lowering._dispatch_operand_for_param( node, target, param, args, kwargs )
			shape = self.lowering._type_resolver._tagged_union_shape( operand.type )
			if shape is None:
				self.lowering.discovery.fail( f'{target.qualname}: conditional dispatch on a non-union argument: {ast.unparse(node)}', node )
			base, members = shape
			# _same_type, not raw `is` - leaf_type (from overload_resolution.
			# py's own call-site-argument-derived condition) and a member's
			# own .type (re-derived here from operand.type via
			# _tagged_union_shape) can be two different Specialization
			# objects for the identical generic instantiation - same
			# duality _check_assignable/_unify_type_param/_coerce_into_union
			# already guard against elsewhere (see PLAN_COMPILER_BUG_SWEEP.md)
			member = next( ( attr for attr in members if self.lowering._type_resolver._same_type( attr.type, leaf_type ) ), None )
			if member is None:
				self.lowering.discovery.fail( f'{target.qualname}: {leaf_type.qualname if leaf_type else "?"} is not a member of {operand.type.qualname}', node )
			tag_attr, _data_attr, _payload_cls, tags = self.lowering._union_storage.get( base )
			tag_dest = self._new_temp( tag_attr.type )
			self._emit( ir.GetAttr( dest = tag_dest, obj = operand, attr = tag_attr.stem ))
			cmp_dest = self._new_temp( bool_cls )
			self._emit( ir.Cmp( dest = cmp_dest, op = ir.CmpOp.EQ, left = tag_dest, right = ir.Const( type = tag_attr.type, value = tags[member.stem] ) ))
			self._emit( ir.JumpIfFalse( cond = cmp_dest, target = next_label ))

	def _emit_dispatch_call( self, target: Function, receiver: ir.Operand|None, args: list[ir.Operand], kwargs: dict[str,ir.Operand], dest: ir.Temp|None, want_result: bool ) -> None:
		params = target.parameters or []
		unwrapped_args = [ self._maybe_unwrap_union_arg( a, p.type ) for a, p in zip( args, params ) ]
		unwrapped_kwargs = {
			name: self._maybe_unwrap_union_arg( value, next( p for p in params if p.stem == name ).type )
			for name, value in kwargs.items()
		}
		self.lowering.schedule( target.return_type )
		for p in params:
			self.lowering.schedule( p.type )
		self._emit( ir.Call( dest = dest if want_result else None, target = target, receiver = receiver, args = unwrapped_args, kwargs = unwrapped_kwargs ))

	def _maybe_unwrap_union_arg( self, operand: ir.Operand, target_type: Type|None, *, owning: bool = False ) -> ir.Operand:
		# a union-typed call-site argument (copy_from: bytes|bytearray)
		# must be unwrapped to the concrete leaf type the chosen branch's
		# parameter actually declares before it can be passed as a real
		# argument - mirrors match's own payload extraction
		if target_type is None or operand.type is target_type:
			return operand
		shape = self.lowering._type_resolver._tagged_union_shape( operand.type )
		if shape is None:
			return operand
		base, members = shape
		# _same_type, not raw `is` - same duality as _lower_dispatch_tests'
		# own identical fix just above (target_type and a member's own
		# .type can be two different Specialization objects for the same
		# generic instantiation). Silently returning operand UNCHANGED
		# when no member matches (rather than failing loudly) makes this
		# one worse than _lower_dispatch_tests' own version if it ever
		# misfires - a wrong, still-union-typed argument passed through
		# to a call expecting a concrete leaf, not a compile error
		member = next( ( attr for attr in members if self.lowering._type_resolver._same_type( attr.type, target_type ) ), None )
		if member is None:
			return operand
		tag_attr, data_attr, payload_cls, tags = self.lowering._union_storage.get( base )
		payload_dest = self._new_temp( payload_cls )
		self._emit( ir.GetAttr( dest = payload_dest, obj = operand, attr = data_attr.stem ))
		dest = self._new_temp( target_type )
		self._emit( ir.GetAttr( dest = dest, obj = payload_dest, attr = f'v_{member.stem}' ))
		if owning and target_type.is_rc():
			# unlike the call-ARGUMENT use of this helper (a transient
			# borrow: `operand`'s own union temp is still alive for the
			# duration of the call, and gets its own ordinary decref
			# afterward, which is all the extracted leaf needs), an
			# `owning=True` caller (the Overload branch's narrowed_return_
			# type tail) hands this value on as the CALL EXPRESSION's own
			# result, which outlives `operand` itself - `operand` (the
			# wide-typed temp holding the real call result) still gets its
			# own normal tag-checked decref at end-of-expression regardless
			# of this narrowing, so without a compensating incref here the
			# extracted leaf and `operand` end up sharing one single
			# reference between two independent owners, and whichever
			# decref runs first frees the payload out from under the
			# other - confirmed via a real repro (Result[str,E].unwrap_or(
			# explicit_default): the narrowed str result read as
			# use-after-free garbage the moment anything else allocated
			# over the freed bytes, invisible under a build that happened
			# to leave the freed memory untouched)
			self._emit( ir.Incref( value = dest ))
			# dest is now a genuinely owned reference (the Incref above), but
			# a bare GetAttr dest is never auto-registered as fresh (_emit's
			# own rule: only Call/Allocate results are) - without this, a
			# caller that consumes dest via assign()/return/etc. still works
			# (that machinery adopts ownership regardless of fresh_temp
			# registration), but one that uses it inline and unbound (e.g.
			# `x.unwrap_or(default) == y`) leaked this Incref's own reference
			# outright - nothing ever tracked it to flush/decref. Confirmed
			# via a real repro (str.__getitem__(i).unwrap_or('') == ':' leaked
			# a str every call).
			self._cfg.fresh_temp( dest, target_type )
		return dest

	def _lower_union_receiver_call( self, node: ast.Call, dispatch: _ReceiverDispatch, receiver: ir.Operand, expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# copy_from.get_const_ptr() where copy_from: bytes|bytearray - unlike
		# _lower_conditional_dispatch (one shared Function, a union-typed
		# ARGUMENT unwrapped per branch), each leaf here has its own
		# unrelated method under this name, so what's dispatched on is the
		# RECEIVER's own tag instead - same tag/data/v_<member> machinery
		# match statements and dispatch already use (UnionStorage.get),
		# just no shared target Function to reuse ConditionalDispatch with
		reference = dispatch.per_leaf[0][1]
		for _member, fn in dispatch.per_leaf:
			self.lowering._ensure_resolved( fn )
		positional, keyword = self.lowering._match_call_args( reference, node )
		args = []
		for param, expr in positional:
			operand = self._lower_expr( expr, param.type )
			self._apply_move_hook( param, operand, dispatch.union.qualname, node )
			args.append( operand )
		kwargs = {}
		for param, expr in keyword:
			operand = self._lower_expr( expr, param.type )
			self._apply_move_hook( param, operand, dispatch.union.qualname, node )
			kwargs[param.stem] = operand

		tag_attr, data_attr, payload_cls, tags = self.lowering._union_storage.get( dispatch.union )
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		# case 1 of the general auto-or_throw() rule - same force_result
		# copy _lower_conditional_dispatch's own identical fix needs, for
		# the union-RECEIVER dispatch tail
		force_result = not want_result and cfg.is_result_type( reference.return_type )
		# discarded plain (non-Result) RC return - see _lower_call's own
		# identical discard_rc for why this still needs a real dest
		discard_rc = not want_result and not force_result and reference.return_type is not None and reference.return_type.is_rc()
		produce_result = want_result or force_result or discard_rc
		dest = self._new_temp( expected_type or reference.return_type ) if produce_result else None
		end_label = self._new_label( 'recv_dispatch_end' )
		for i, ( member, fn ) in enumerate( dispatch.per_leaf ):
			is_last = i == len( dispatch.per_leaf ) - 1
			if not is_last:
				next_label = self._new_label( 'recv_dispatch_next' )
				tag_dest = self._new_temp( tag_attr.type )
				self._emit( ir.GetAttr( dest = tag_dest, obj = receiver, attr = tag_attr.stem ))
				cmp_dest = self._new_temp( bool_cls )
				self._emit( ir.Cmp( dest = cmp_dest, op = ir.CmpOp.EQ, left = tag_dest, right = ir.Const( type = tag_attr.type, value = tags[member.stem] )))
				self._emit( ir.JumpIfFalse( cond = cmp_dest, target = next_label ))
			payload_dest = self._new_temp( payload_cls )
			self._emit( ir.GetAttr( dest = payload_dest, obj = receiver, attr = data_attr.stem ))
			narrowed = self._new_temp( member.type )
			self._emit( ir.GetAttr( dest = narrowed, obj = payload_dest, attr = f'v_{member.stem}' ))
			self.lowering.schedule( fn.return_type )
			for p in fn.parameters or []:
				self.lowering.schedule( p.type )
			# per-leaf argument coercion/validation - `args`/`kwargs` above
			# were built ONCE, lowered against `reference`'s own declared
			# parameter types only; a leaf whose own parameter type
			# genuinely differs (Box[i32]|Box[u32]'s own two `set(x: T)`
			# instantiations) needs the SAME coercion-or-rejection chain
			# _lower_expr's own dispatch would already have given it, run
			# again here against THIS leaf's own type - reusing the already-
			# lowered operand (never re-lowering/re-evaluating the original
			# argument expression, which would double its side effects once
			# per leaf; see _coerce_or_check_operand's own docstring)
			leaf_temps_start = len( self._pending_temps )
			leaf_args = []
			for ( ref_param, expr ), operand in zip( positional, args ):
				leaf_param = self._corresponding_leaf_param( reference, fn, ref_param )
				context = f'{fn.qualname}(...): parameter {leaf_param.stem!r}'
				leaf_args.append( self._coerce_or_check_operand( operand, leaf_param.type, expr, context = context ))
			leaf_kwargs = {}
			for ref_param, expr in keyword:
				leaf_param = self._corresponding_leaf_param( reference, fn, ref_param )
				context = f'{fn.qualname}(...): parameter {leaf_param.stem!r}'
				leaf_kwargs[leaf_param.stem] = self._coerce_or_check_operand( kwargs[ref_param.stem], leaf_param.type, expr, context = context )
			self._emit( ir.Call( dest = dest, target = fn, receiver = narrowed, args = leaf_args, kwargs = leaf_kwargs ))
			# a leaf whose own parameter type needs real union-widening
			# coercion (not just a borrowed CastWrap - see _coerce_or_check_
			# operand's own comment) leaves a fresh, independently
			# fresh_temp()-tracked wrapped value in leaf_args/leaf_kwargs,
			# passed to the Call above as an ordinary BORROWED argument (no
			# ownership transfer, same convention every other call site
			# uses) - the caller still owns releasing it. In non-branching
			# code the enclosing statement's own end-of-statement flush does
			# that correctly; here this whole per-leaf block is only ONE
			# branch of a larger dispatch tree (skippable via an earlier
			# leaf's own tag match), so that flush fires unconditionally for
			# EVERY leaf regardless of which one's Call actually ran -
			# reading tag/payload data off an uninitialized C local for
			# whichever leaf never executed. Confirmed via a real repro
			# (union receiver dispatch, one leaf declaring a plain parameter
			# type, the other a wider union needing _coerce_into_union) -
			# same bug class _flush_branch_temps' own docstring documents
			# for _expr_BoolOp/_expr_IfExp, and _emit_eq_dispatch_tree's/
			# _coerce_or_check_operand's own hand-rolled decref+untrack
			# fixes cover elsewhere in this file. dest is excluded (it's
			# this whole call's own merge point, must survive to the next
			# leaf/end_label)
			self._flush_branch_temps( leaf_temps_start, *( [ dest ] if dest is not None else [] ))
			if not is_last:
				self._emit( ir.Jump( target = end_label ))
				self._emit( ir.Label( name = next_label ))
		self._emit( ir.Label( name = end_label ))
		return self._finish_call_result( node, dest, want_result )
