# stdlib imports:
import ast

# local imports:
import ir
from fstring_format_spec import FStringFormatSpec, FormatSpecError, parse_format_spec, validate_str_spec, validate_int_spec, validate_float_spec
from mpy_types import (
	Type, Function, ClassLike, Specialization, TaggedUnion,
)

class FStringLoweringMixin:
	''' f-string / format-spec lowering - mixed into FunctionLowering (lowering.py), which
	see for the shared instance state (self._instructions, self._cfg, self.lowering,
	etc.) every method here reads and writes. Never instantiated on its own;
	split out of lowering.py purely to keep that file to a manageable size - see
	lowering.py's own class docstring and FunctionLowering's base-class list for
	the full set of sibling mixins this one is composed with. '''


	def _is_literal_format_spec( self, format_spec: ast.JoinedStr ) -> bool:
		return all( isinstance( v, ast.Constant ) for v in format_spec.values )

	def _literal_format_spec_text( self, format_spec: ast.JoinedStr ) -> str:
		return ''.join( v.value for v in format_spec.values ) # each v is ast.Constant(str) - _is_literal_format_spec already confirmed this

	def _lower_fstring_part( self, node: 'ast.Constant|ast.FormattedValue', str_type: Type ) -> ir.Operand:
		# one element of an f-string's ast.JoinedStr.values - either a
		# literal text segment (ast.Constant, already merged by CPython's
		# own parser) or a {expr} interpolation (ast.FormattedValue).
		# Shared by _expr_JoinedStr's single-part short-circuit and its
		# N-part UnsafeList/slice/str.concat path below - both need the
		# same str-typed operand per element, just assembled differently
		# (PLAN_FSTRINGS.md).
		if isinstance( node, ast.Constant ):
			return self._lower_expr( node, str_type )
		# ast.FormattedValue
		parsed_spec = None
		if node.format_spec is not None:
			if not self._is_literal_format_spec( node.format_spec ):
				self.lowering.discovery.fail(
					f'f-string format specs must be a literal string for now - dynamic format specs are not supported yet: {ast.unparse(node)}',
					node,
				)
			try:
				parsed_spec = parse_format_spec( self._literal_format_spec_text( node.format_spec ))
			except FormatSpecError as e:
				self.lowering.discovery.fail( f'{e} ({ast.unparse(node)})', node )

		operand = self._lower_expr( node.value, None )

		if node.conversion == -1 and parsed_spec is not None:
			# no explicit !conversion - the format spec dispatches against
			# the value's OWN type directly (int's own radix/width/sign
			# handling, e.g.), matching Python's own format(x, spec) ==
			# type(x).__format__(x, spec) - as opposed to format(str(x),
			# spec) or format(repr(x), spec), which is what an EXPLICIT
			# !s/!r/!a conversion means instead (handled below)
			return self._lower_dispatch_format_spec( operand, parsed_spec, str_type, node )

		# conversion 114 == '!r' or 97 == '!a' (ascii wants a repr-shaped
		# text, ascii-escaped below via _lower_ascii_escape) both want
		# __repr__; -1 (none) and 115 ('!s') want __str__ - just an ordinary
		# method lookup, same as any other type: every fixed-width int
		# scalar (i8/u8/.../isize/usize) has a real __str__/__repr__
		# (lib/builtins/__scalar_dunders.py's i_str_signed/i_str_unsigned),
		# same mechanism f64/f32's own __str__/__repr__ use (__float.py) -
		# a scalar WITHOUT one (bool, currently) still fails cleanly here
		# with a plain "method not found" error rather than being auto-boxed.
		# str itself only short-circuits to identity for !s/no-conversion
		# (str.__str__ is itself an identity - see __init__.py); !r/!a on a
		# str operand must still go through str.__repr__() for real quoting/
		# escaping, so they're excluded from the identity shortcut here.
		if operand.type is str_type and node.conversion not in ( 114, 97 ):
			value_as_str = operand
		else:
			method_name = '__repr__' if node.conversion in ( 114, 97 ) else '__str__'
			value_as_str = self._lower_method_call( operand, method_name, [], str_type, node )
		if node.conversion == 97:
			value_as_str = self._lower_ascii_escape( value_as_str, str_type, node )

		if parsed_spec is not None:
			# an explicit !s/!r/!a conversion (or the operand's own type
			# needing __str__) already reduced the value to plain str - the
			# spec now formats THAT text (fill/align/width/precision-as-
			# truncation only, str's own branch below) rather than
			# dispatching against the original value's own type again
			return self._lower_dispatch_format_spec( value_as_str, parsed_spec, str_type, node )
		return value_as_str

	def _lower_ascii_escape( self, operand: ir.Operand, str_type: Type, node: ast.AST ) -> ir.Operand:
		# f-string !a conversion's second half - operand is already the
		# result of __repr__() (called above, same as !r): real quoting/
		# escaping already happened there, so this just needs to escape
		# any printable non-ASCII codepoints __repr__ left as literal
		# UTF-8 - str._ascii_escape() (lib/builtins/__init__.py) does
		# exactly that, leaving every ASCII byte (including the quotes/
		# backslashes __repr__ inserted) untouched.
		return self._lower_method_call( operand, '_ascii_escape', [], str_type, node )

	def _lower_dispatch_format_spec( self, operand: ir.Operand, spec: FStringFormatSpec, str_type: Type, node: ast.AST ) -> ir.Operand:
		if operand.type is str_type:
			return self._lower_str_format_spec( operand, spec, str_type, node )
		int_type = self.lowering.discovery.find_name_or_none( 'int' )
		if int_type is not None and operand.type is int_type:
			return self._lower_int_format_spec( operand, spec, str_type, node )
		intrinsics = self.lowering.discovery.get_intrinsics()
		if operand.type is intrinsics.get( 'f32' ) or operand.type is intrinsics.get( 'f64' ):
			return self._lower_float_format_spec( operand, spec, str_type, node )
		type_name = operand.type.qualname if operand.type is not None else '?'
		if spec.type in ( 'f', 'F', 'e', 'E', 'g', 'G', '%' ):
			self.lowering.discovery.fail(
				f"f-string format spec: {spec.type!r} needs a real float type with formatting support, which doesn't exist yet ({type_name}): {ast.unparse(node)}",
				node,
			)
		self.lowering.discovery.fail(
			f'f-string format spec: {type_name} does not support format specs yet (only str, int, and float do): {ast.unparse(node)}',
			node,
		)

	def _lower_pad_by_align( self, operand: ir.Operand, align: str, fill: str, width: int, str_type: Type, node: ast.AST ) -> ir.Operand:
		method_name = { '<': 'ljust', '>': 'rjust', '^': 'center' }[align]
		args = [ self._const_usize( width ), ir.Const( type = str_type, value = fill ) ]
		return self._lower_method_call( operand, method_name, args, str_type, node )

	def _lower_str_format_spec( self, operand: ir.Operand, spec: FStringFormatSpec, str_type: Type, node: ast.AST ) -> ir.Operand:
		try:
			validate_str_spec( spec )
		except FormatSpecError as e:
			self.lowering.discovery.fail( f'{e} ({ast.unparse(node)})', node )
		value = operand
		if spec.precision is not None:
			value = self._lower_method_call( value, '_truncate_codepoints', [ self._const_usize( spec.precision ) ], str_type, node )
		if spec.width is not None:
			value = self._lower_pad_by_align( value, spec.align or '<', spec.fill, spec.width, str_type, node ) # str's own default align is left, unlike numeric types' right
		return value

	_RADIX_BY_TYPE_CHAR = { 'b': 2, 'o': 8, 'x': 16, 'X': 16 }
	_RADIX_PREFIX_BY_TYPE_CHAR = { 'b': '0b', 'o': '0o', 'x': '0x', 'X': '0X' }

	def _lower_int_format_spec( self, operand: ir.Operand, spec: FStringFormatSpec, str_type: Type, node: ast.AST ) -> ir.Operand:
		try:
			validate_int_spec( spec )
		except FormatSpecError as e:
			self.lowering.discovery.fail( f'{e} ({ast.unparse(node)})', node )
		type_char = spec.type

		if type_char in ( 'b', 'o', 'x', 'X' ):
			base = self._RADIX_BY_TYPE_CHAR[type_char]
			uppercase = type_char == 'X'
			raw_digits = self._lower_method_call( operand, '_to_radix_digits', [ self._const_i32( base ), self._const_bool( uppercase ) ], str_type, node )
			prefix_text = self._RADIX_PREFIX_BY_TYPE_CHAR[type_char] if spec.alt else ''
			sep_text = '' # grouping is never valid for a radix type char (validate_int_spec)
		else:
			raw_digits = self._lower_method_call( operand, '_decimal_digits', [], str_type, node )
			prefix_text = ''
			sep_text = spec.grouping or ''
		sep = ir.Const( type = str_type, value = sep_text ) # '' still groups correctly - see str._insert_thousands_sep's own comment

		sign_char = self._lower_method_call( operand, '_sign_prefix', [ ir.Const( type = str_type, value = spec.sign ) ], str_type, node )
		if prefix_text:
			sign_and_prefix = self._lower_str_add( sign_char, ir.Const( type = str_type, value = prefix_text ), str_type, node )
		else:
			sign_and_prefix = sign_char

		if spec.width is not None and spec.align == '=':
			# the '0' shorthand - zero-padding goes BETWEEN sign/prefix and
			# digits, grouping-aware (str._pad_and_group_after_prefix - a
			# plain "group first, then _pad_after_prefix" two-step gives
			# the wrong answer once grouping is combined with zero-pad, see
			# its own comment) - needs the RAW, ungrouped digits, not the
			# _insert_thousands_sep'd ones the other two branches below want
			return self._lower_method_call(
				raw_digits, '_pad_and_group_after_prefix',
				[ sign_and_prefix, self._const_usize( spec.width ), ir.Const( type = str_type, value = spec.fill ), sep ],
				str_type, node,
			)
		digits = self._lower_method_call( raw_digits, '_insert_thousands_sep', [ sep ], str_type, node )
		if spec.width is None:
			return self._lower_str_add( sign_and_prefix, digits, str_type, node )
		body = self._lower_str_add( sign_and_prefix, digits, str_type, node )
		return self._lower_pad_by_align( body, spec.align or '>', spec.fill, spec.width, str_type, node ) # numeric types' own default align is right, unlike str's left

	def _lower_float_format_spec( self, operand: ir.Operand, spec: FStringFormatSpec, str_type: Type, node: ast.AST ) -> ir.Operand:
		# 'f'/'F'/'e'/'E'/'g'/'G'/'%' (PLAN_STR_FORMAT.md item 4 - every
		# float type char fstring_format_spec.FORMAT_SPEC_TYPE_CHARS
		# recognizes). Same sign+digits+pad assembly shape as
		# _lower_int_format_spec above (no radix/grouping prefix to worry
		# about here, so it's simpler), calling into lib/builtins/
		# __float.py's own _sign_prefix/_fixed_digits/_percent_digits
		# methods - real control flow lives there, not hand-built IR here,
		# matching int's own _sign_prefix/_to_radix_digits split.
		try:
			validate_float_spec( spec )
		except FormatSpecError as e:
			self.lowering.discovery.fail( f'{e} ({ast.unparse(node)})', node )
		precision = spec.precision if spec.precision is not None else 6 # Python's own f"{x:f}"/f"{x:e}"/f"{x:g}"/f"{x:%}" all share this default
		alt = self._const_bool( spec.alt )
		sep = ir.Const( type = str_type, value = spec.grouping or '' ) # '' still groups correctly - see str._insert_thousands_sep's own comment
		is_percent = spec.type == '%'
		# None type char WITH an explicit precision behaves like 'g' (plus
		# its own "always show a fractional digit in fixed form" tweak) -
		# real Python's own "None" presentation, not plain 'f' (see
		# validate_float_spec's own comment and lib/builtins/__float.py's
		# _none_type_digits_raw). None type char with NO precision either
		# (f"{x:10}") needs Python's real shortest-round-trip repr
		# algorithm instead - _repr_digits/_repr_digits_raw (lib/builtins/
		# __float.py), the same machinery bare f"{x}" uses via __str__/
		# __repr__ (_lower_fstring_part's own dispatch, unrelated to this
		# function - reached before a format spec is even considered).
		is_none_type_with_precision = spec.type is None and spec.precision is not None and not is_percent
		is_none_type_no_precision = spec.type is None and spec.precision is None and not is_percent
		type_char = (
			self._const_i32( ord( spec.type or 'f' ) )
			if not is_percent and not is_none_type_with_precision and not is_none_type_no_precision
			else None
		)
		sign_char = self._lower_method_call( operand, '_sign_prefix', [ ir.Const( type = str_type, value = spec.sign ) ], str_type, node )

		if is_percent:
			digits_method, digits_args = '_percent_digits', [ self._const_usize( precision ), alt ]
		elif is_none_type_with_precision:
			digits_method, digits_args = '_none_type_digits', [ self._const_usize( precision ), alt ]
		elif is_none_type_no_precision:
			digits_method, digits_args = '_repr_digits', []
		else:
			digits_method, digits_args = '_fixed_digits', [ self._const_usize( precision ), type_char, alt ]

		if spec.width is not None and spec.align == '=':
			# the '0' shorthand - zero-padding goes BETWEEN sign and
			# digits, grouping-aware AND special-value-aware (str._pad_
			# maybe_special - a plain "group first, then _pad_after_prefix"
			# two-step gives the wrong answer once grouping is combined
			# with zero-pad, and "nan"/"inf" text needs to skip grouping
			# entirely even when requested - see str._pad_and_group_after_
			# prefix's own comment and _pad_maybe_special's own comment) -
			# needs the RAW, ungrouped digits (the '_raw' variant of
			# whichever digits_method was picked above), not the already-
			# grouped ones the other branch below wants
			raw = self._lower_method_call( operand, digits_method + '_raw', digits_args, str_type, node )
			if is_percent:
				# str._pad_maybe_special has no notion of '%' - reserve 1
				# char of the nominal width for it here, then append it
				# after, the same "caller reserves room for what this
				# method doesn't know about" convention _pad_and_group_
				# before_dot's own comment documents
				inner_width = max( spec.width - 1, 0 )
				padded = self._lower_method_call(
					raw, '_pad_maybe_special',
					[ sign_char, self._const_usize( inner_width ), ir.Const( type = str_type, value = spec.fill ), sep ],
					str_type, node,
				)
				return self._lower_str_add( padded, ir.Const( type = str_type, value = '%' ), str_type, node )
			return self._lower_method_call(
				raw, '_pad_maybe_special',
				[ sign_char, self._const_usize( spec.width ), ir.Const( type = str_type, value = spec.fill ), sep ],
				str_type, node,
			)

		digits = self._lower_method_call( operand, digits_method, digits_args + [ sep ], str_type, node )
		if spec.width is None:
			return self._lower_str_add( sign_char, digits, str_type, node )
		body = self._lower_str_add( sign_char, digits, str_type, node )
		return self._lower_pad_by_align( body, spec.align or '>', spec.fill, spec.width, str_type, node ) # numeric types' own default align is right, unlike str's left

	def _lower_str_add( self, left: ir.Operand, right: ir.Operand, str_type: Type, node: ast.AST ) -> ir.Operand:
		return self._lower_method_call( left, '__add__', [ right ], str_type, node )

	def _expr_JoinedStr( self, node: ast.JoinedStr, expected_type: Type|None ) -> ir.Operand:
		# f-string (PLAN_FSTRINGS.md). A fully compile-time-known JoinedStr
		# never reaches here at all - compile_time_transformer.py's own
		# _ConstFolder.visit_JoinedStr already collapsed it to a plain
		# ast.Constant(str) before lowering.py ever sees the function body.
		# expected_type is deliberately never used to type the result here,
		# same reasoning _expr_Constant's own comment gives for its own
		# TaggedUnion case: str.concat's return is authoritatively str
		# either way, and _lower_expr's own post-hoc coercion is what wraps
		# a plain str into a wider union afterward, if one was asked for.
		str_type = self.lowering.discovery.find_name_or_none( 'str' )
		if str_type is None:
			self.lowering.discovery.fail( f'f-string requires the str type to be available: {ast.unparse(node)}', node )

		if len( node.values ) == 0:
			return ir.Const( type = str_type, value = '' )
		if len( node.values ) == 1:
			return self._lower_fstring_part( node.values[0], str_type )

		parts = [ self._lower_fstring_part( value, str_type ) for value in node.values ]
		n = len( parts )
		usize_cls = self.lowering.discovery.get_intrinsics()['usize']

		# UnsafeList[str](n) - the escape hatch lib/builtins/__list.py's own
		# module docstring names for exactly this: a fixed-capacity,
		# never-escaping, single-statement-lifetime scratch buffer, with no
		# lock overhead a real list[T] would pay for no reason here (n is
		# fixed at compile time - capacity never grows, so append() below
		# can never actually trigger RawList._grow() at all)
		# _get_or_create_specialization + _ensure_resolved gives back the
		# REAL, concrete, already-monomorphized UnsafeList[str] ClassLike
		# (not the Specialization wrapper - same "swap a Specialization for
		# its monomorphized form" ensure_resolved always does), exactly the
		# way _try_lower_construct_call's own "explicit ClassName[T](...)"
		# branch does before ITS target_cls.type_params check ever runs
		# (type_resolver.py's own _try_resolve_namespace pre-resolves a
		# Subscript callee's Specialization the same way). Using this
		# CONCRETE class from here on (not the abstract UnsafeList) matters
		# for real: its own .names are ALREADY-substituted (T=str bound)
		# methods, no separate per-method Specialization dance needed - and
		# _schedule_rcclass_construction below specifically REQUIRES a
		# concrete class (passing the still-generic abstract one there
		# schedules the ABSTRACT __del__ as a standalone compile unit, T
		# forever unbound - confirmed via a real repro: "compiler.is_rc(T)
		# requires a concrete type" - type_resolver.py's own
		# _schedule_rcclass_destructor_deps documents this exact hazard
		# and guards against it with a cls.type_params check; this is the
		# same hazard from the calling side instead).
		unsafelist_cls = self.lowering.discovery.find_name( 'UnsafeList', node )
		cls_spec = self.lowering.discovery._get_or_create_specialization( unsafelist_cls, [ str_type ])
		concrete_cls = self.lowering._ensure_resolved( cls_spec )

		init = concrete_cls.get_local_or_raise( '__init__' )
		self.lowering._ensure_resolved( init ) # schedules init ITSELF as a compile unit - monomorphize_class's own per-method substitution loop only builds+caches the substituted Function, it never schedules any of them for real emission on its own (confirmed via a real repro: an unscheduled monomorphized method compiles fine at the CALL SITE but is never actually emitted, producing a C "call to undeclared function" link-time-shaped error)
		self.lowering.schedule( init.return_type )
		for p in ( init.parameters or [] ):
			self.lowering.schedule( p.type )

		buf = self._new_temp( cls_spec )
		self.lowering._schedule_rcclass_construction( concrete_cls, cls_spec )
		self._emit( ir.Allocate( dest = buf, cls = concrete_cls, fields = {} ))
		n_const = ir.Const( type = usize_cls, value = n )
		self._emit( ir.Call( dest = None, target = init, receiver = buf, args = [ n_const ], kwargs = {} ))

		#none_type = self.lowering.discovery.get_none_type()
		#overflow_error_cls = self.lowering.discovery.find_name( 'OverflowError', node )
		append = concrete_cls.get_local_or_raise( 'append' )
		self.lowering._ensure_resolved( append ) # see init's own comment on why this is needed
		self.lowering.schedule( append.return_type )
		for p in ( append.parameters or [] ):
			self.lowering.schedule( p.type )
		for part in parts:
			#append_result = self._new_temp( append.return_type )
			#self._emit( ir.Call( dest = append_result, target = append, receiver = buf, args = [ part ], kwargs = {} ))
			#self._lower_unwrap_result(
			#	append_result, 'f-string: internal append failed (unreachable - buffer is pre-sized exactly)',
			#	none_type, overflow_error_cls, str_type, node, want_result = False,
			#)
			self._emit( ir.Call( dest = None, target = append, receiver = buf, args = [ part ], kwargs = {} ))

		# str.concat( buf ) directly - concat's own parameter type is
		# UnsafeList[str] (matches buf exactly), so no bridging step is
		# needed here at all (this used to build a slice[str] view over buf
		# first, back when concat took slice[str] - that view type is gone
		# now, see PLAN_STR_FORMAT.md/list.__getitem__(slice) history).
		concat = self.lowering._find_method( str_type, 'concat' )
		self.lowering._ensure_resolved( concat )
		self.lowering.schedule( concat.return_type )
		for p in ( concat.parameters or [] ):
			self.lowering.schedule( p.type )
		dest = self._new_temp( str_type )
		self._emit( ir.Call( dest = dest, target = concat, receiver = None, args = [ buf ], kwargs = {} ))
		return dest
