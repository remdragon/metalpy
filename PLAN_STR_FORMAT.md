str/f-string formatting: what's still deferred (str.format, dynamic specs,
float specs, __format__ protocol)

STATUS: the two real gaps from the original f-string pass (07cce54) - `!a`
conversion and literal `{expr:spec}` format specs for str/int - are now
implemented (fstring_format_spec.py, lowering.py's _lower_dispatch_format_spec/
_lower_str_format_spec/_lower_int_format_spec, compile_time_transformer.py's
visit_JoinedStr). That pass was deliberately narrow-scoped rather than
open-ended; this file tracks the pieces it left out on purpose, with enough
design texture that each is independently pickup-able later rather than a bare
TODO line. Item 4 (float format specs, 'f'/'F' only) has since landed too -
see its own STATUS note below.

Why this split happened

f-strings' `{expr}` is a real ast.FormattedValue - Python's own grammar
already parses it, no new language needed. A LITERAL format_spec (every
element of the FormattedValue's own format_spec JoinedStr is a plain
ast.Constant) can be parsed entirely in PYTHON, at compile time, by
lowering.py itself - fstring_format_spec.py's parse_format_spec() does
exactly this, mirroring CPython's own format-spec BNF. Everything below
needs something that ceiling doesn't cover: a REAL str->str spec parser
running at RUNTIME (in metalpy source, since no regex/parsing infra exists
in this language anywhere), a real `__format__` dispatch protocol, or a
type (float) that doesn't have string conversion at all yet.

Deferred items

1. Dynamic (non-literal) format specs - f"{x:{width}}"

   Rejected today at lowering.py:3705 ("f-string format specs must be a
   literal string for now") the moment _is_literal_format_spec() sees a
   real ast.FormattedValue inside the format_spec's own JoinedStr, before
   ever trying to parse joined text as a spec string.

   Needs: a real `str -> ParsedSpec`-equivalent parser written IN metalpy
   source (byte-scanning, mirroring fstring_format_spec.py's own
   left-to-right BNF walk, since no regex/parsing infra exists anywhere
   else in this language either), reachable at runtime - plus the
   `__format__` protocol below to actually dispatch it against a value
   whose type isn't known until then either (a truly dynamic spec can
   appear on any operand, not just str/int). This is the natural
   foundation str.format() itself would reuse (see #3).

2. str.format() - "{0:>10}, {name!r}".format(x, name=y)

   Not started. Its template string is an ordinary RUNTIME str value,
   parsed by str.format's own field-reference mini-scanner (`{0}`,
   `{name}`, `{0.attr[key]!r:spec}`) over the template text itself - this
   has no relationship to Python's AST at all, unlike f-strings' `{expr}`
   (already a real expression node). Needs the same runtime spec parser
   as #1, PLUS the separate field-reference scanner, PLUS positional/
   keyword argument-list handling (str.format takes *args/**kwargs -
   itself its own gap, see _match_call_args's existing *args/**kwargs
   rejection elsewhere in this compiler).

3. A general, user-overridable `__format__` dunder protocol

   Today's format-spec dispatch (lowering.py's _lower_dispatch_format_spec,
   line 3759) is NOT an extensible protocol - it's a fixed, compile-time
   `if operand.type is str_type / is int_type` chain calling specific,
   directly-known methods (_lower_str_format_spec/_lower_int_format_spec).
   This is fine for a literal spec (lowering.py already knows the operand's
   static type, so it can just call the right thing directly), but a
   dynamic spec (#1) or str.format() (#2) both need genuine runtime
   dispatch - some `T.__format__(self, spec: str) -> str` a user's own
   RCClass could define and override, resolved the way __str__/__repr__
   already are (_find_method). Arrives naturally alongside #1/#2, not
   worth building in isolation first.

4. Float format specs - f"{x:.2f}", f"{x:.3e}", f"{x:%}"

   STATUS: DONE - every float type char fstring_format_spec.
   FORMAT_SPEC_TYPE_CHARS recognizes now works: 'f'/'F' (fixed-point),
   'e'/'E' (exponential), 'g'/'G' (general), '%' (percent), plus no type
   char at all (a simplification - defers to 'f', not Python's real "None"
   presentation, which is closer to 'g' with its own tweaks; a known,
   separate, narrower gap than this item ever covered). Sign/width/align/
   the '0' zero-pad shorthand all work uniformly across every type char,
   same as int/str. validate_int_spec still gives the old "doesn't exist
   yet" message for a float type char reaching an INT operand - unrelated
   to this item, that type/operand mismatch is simply invalid.

   The old "doesn't exist yet" message for a float OPERAND is gone -
   validate_float_spec (fstring_format_spec.py) now only rejects a type
   char that's genuinely invalid for float (e.g. 'x', int's own radix
   char) with "is not valid for float". `#` and `,`/`_` grouping are both
   supported too now, for every type char (see below).

   float did NOT end up needing a boxed RCClass the way this item
   originally assumed. Scalar (mpy_types.py) already had a `.names` dict
   and a working `SomeScalar.method = some_function` registration sigil
   (discovery.py's visit_Assign, tested by discovery_test.py's
   ScalarMethodRegistrationTests) built for scalar-to-scalar casts like
   usize.__u32__ - lib/builtins/__float.py just registers ordinary free
   functions (_f64_sign_prefix/_f64_fixed_digits, plus f32 equivalents that
   widen and delegate) onto f64.names/f32.names the same way, and
   lowering.py's existing _find_method/_lower_method_call already dispatch
   to a Scalar receiver generically. One real gap surfaced by being the
   FIRST actual caller through this path (not just registration, which the
   existing test already covered): a Scalar-registered function is a
   genuine free Function, so discovery never strips a "self" off its
   .parameters the way it does for a real class body - _lower_method_call
   now special-cases `method.cls is None` to pass the receiver as an
   ordinary leading positional argument instead of via ir.Call's own
   receiver field, which assumes a receiver-stripped parameter list.
   __str__/__repr__ (bare f"{x}") remain deferred regardless - see #6;
   f"{x:.1f}" never reaches them at all (an explicit format spec dispatches
   straight against the operand's own type).

   _lower_float_format_spec() (lowering.py, alongside _lower_str_format_
   spec/_lower_int_format_spec) mirrors _lower_int_format_spec's own
   sign+digits+pad assembly shape, reusing _lower_pad_by_align verbatim for
   width/align, per this item's original guidance. The actual numeric-to-
   decimal-digits conversion (the genuinely new work this item always
   flagged) is NOT hand-rolled metalpy-source arithmetic - getting
   float-to-decimal rounding exactly right by hand is genuinely hard
   (naive fractional-digit extraction accumulates floating-point error) -
   it's a new `compiler.format_f64(buf, size, precision, type_char, value)`
   compiler intrinsic (lowering.py's _lower_compiler_format_f64/
   ir.FormatFloat, same shape as compiler.addrof/compiler.atomic_load),
   backed by a hand-written C helper in emitter_c.py's PROLOGUE that
   resolves and calls the real, always-present platform libc float
   formatter at runtime via a dynamically-built "%.*X" format string
   (X = type_char, one of 'f'/'F'/'e'/'E'/'g'/'G'): real snprintf on
   POSIX; on Windows, msvcrt.dll's own _snprintf, resolved dynamically via
   GetModuleHandleA/LoadLibraryA/GetProcAddress (kernel32) rather than
   statically linked, keeping no-crt Windows builds CRT-free. '%' has no
   printf equivalent - lib/builtins/__float.py's _f64_percent_digits
   handles it entirely in metalpy source instead (scale by 100, format as
   'f', append a literal '%' - Python's own exact definition of '%').
   This is NOT an ordinary @extern binding on either platform -
   deliberately, for two independent reasons found while building the
   'f'/'F' half of this: (1) emitter_c.py's extern codegen only emits
   fixed-arity C prototypes, an ABI hazard for a genuinely variadic callee
   on some argument shapes/platforms; (2) tagging it under the 'c' extern
   lib (the obvious alternative) would flip compiler.extern_libs and break
   float_test.py's own no_crt = 'c' not in compiler.extern_libs on Windows
   even though the C implementation never touches real msvcrt/ucrt
   statically. ntdll.dll's own exported _snprintf (same dynamic-resolution
   technique, and genuinely present per `dumpbin /exports ntdll.dll`) was
   tried FIRST and rejected once a real functional test showed it silently
   fails on any float conversion ("%.*f"/"%f" both just emit a stray "f",
   no digits) - apparently NT's kernel-adjacent runtime never needed float
   formatting internally. Buffer sizing in lib/builtins/__float.py is a
   generous fixed upper bound (a max-magnitude f64 needs at most 309
   integer digits), not tightly computed.

   A SECOND, Windows-only quirk was found adding 'e'/'E'/'g'/'G': legacy
   msvcrt.dll's own exponent is always padded to exactly 3 digits
   ("1.23e+003"), unlike Python/C99 (POSIX's real snprintf included),
   which use a 2-digit floor ("1.23e+03") - confirmed by a real test
   against this system's own msvcrt.dll, since dumpbin (symbol presence)
   can't show behavioral differences like this. emitter_c.py's PROLOGUE
   now also carries __metalpy_fixup_msvcrt_exponent, a small Windows-only
   post-processing pass that strips the extra leading zero from the
   exponent in place before returning - real f64 exponents are always
   <= 3 digits, so at most one zero is ever stripped in practice.

   `#` (always show the decimal point for 'f'/'F'/'e'/'E', keep trailing
   zeros for 'g'/'G') and `,`/`_` grouping (thousands separators in the
   integer part) are both DONE too now - validate_float_spec no longer
   rejects either, and both were confirmed to match real Python's own
   output exactly. compiler.format_f64 grew an `alt` (bool) parameter for
   '#', passed straight into the dynamically-built format string ("%#.*X")
   since real snprintf's own '#' flag already matches Python's semantics
   for every one of these type chars exactly - no post-processing needed,
   unlike grouping, which has NO printf equivalent at all: lib/builtins/
   __float.py's _group_integer_part is a real (small, self-contained)
   post-processing pass, applied to whatever snprintf/msvcrt already
   returned, inserting the separator every 3 digits into the text before
   the first '.' only - a correct no-op for 'e'/'E' and for 'g'/'G' in
   its own exponential form, since there's only ever one digit before the
   decimal point either way. No boxed float class was needed for any of
   this, confirming the note above - only Scalar's existing `.names`
   registration hook. Two more real bugs were found and fixed along the
   way, both only surfacing once this pass actually exercised paths the
   earlier ones hadn't:

   - _group_integer_part's own trivial "nothing to group" early return
     handed the caller back its own BORROWED `digits` parameter directly,
     without the explicit incref str.concat's own comment already
     documents as required for exactly this shape ("compiler.incref(part)
     right after the borrow gives part a real +1 of its own") - a real,
     reproducible over-release/use-after-free (confirmed by a garbage
     process exit code, the same "different wrong output on repeated
     runs" signature real UB produces), not just a style nit.
   - legacy msvcrt.dll's own _snprintf silently returns nothing at all for
     the uppercase 'F' conversion character specifically (confirmed by a
     real test against this system's own msvcrt.dll: 'E'/'G' both work
     correctly there, only 'F' doesn't - matching real printf history,
     since %F was only added in C99, after legacy msvcrt) - meaning
     f"{x:.1F}"-shaped specs had been silently broken since the very
     first 'f'/'F' pass landed, just never exercised by a real end-to-end
     test until now. emitter_c.py's PROLOGUE now substitutes lowercase
     'f' internally on Windows for this one conversion character - correct
     for every finite value, the only known remaining gap being inf/nan
     capitalization, which this compiler doesn't special-case either way
     yet (a separate, likely upcoming item once float inf/nan display
     itself is addressed).

   A THIRD bug - real, not float-specific, and not caught until a
   dedicated review of this whole area afterward - was found and fixed
   separately: grouping (,/_) COMBINED with the '0' zero-pad shorthand
   produced silently wrong output for BOTH int and float. Real Python
   groups the padding characters together with the original digits
   (f"{1234567:015,d}" == '000,001,234,567' - the padding zeros pick up
   their own comma separators too); the old code grouped the digits
   FIRST, then rjust-padded that already-grouped string with raw fill
   characters on top, giving '0000001,234,567' instead - the right VALUE,
   wrong grouping. No test anywhere (int or float) had ever combined the
   '0' shorthand with grouping. Fixed with three new shared str methods
   (lib/builtins/__init__.py, alongside _pad_after_prefix): _insert_
   thousands_sep (the grouping loop int._decimal_digits_with_grouping and
   float._group_integer_part each already had their own earlier, narrower
   inline copy of - not retrofitted onto either, to avoid touching
   already-shipped code for a pure refactor), _pad_and_group_after_prefix
   (pads RAW ungrouped digits, then groups the WHOLE padded field - int's
   own shape), and _pad_and_group_before_dot (float's wrapper - splits at
   the first '.', delegates the integer part to the previous method,
   reappends the untouched fractional/exponent suffix). Both int
   (_decimal_digits, lib/builtins/__int.py) and float (_fixed_digits_raw/
   _percent_digits_raw, lib/builtins/__float.py) gained raw (ungrouped)
   digit-producing variants alongside their existing grouped ones,
   specifically for this zero-pad path - lowering.py's _lower_int_format_
   spec/_lower_float_format_spec now route the '0'-shorthand branch
   through the raw+pad-and-group path, leaving every OTHER branch
   (non-zero-pad width, or no width at all) calling the original,
   unchanged, already-tested grouped-digit methods exactly as before.

5. `=` general sign-aware alignment

   Only the `0` zero-pad shorthand's own implicit `=` is supported
   (fstring_format_spec.py's parse_format_spec sets align='=' itself when
   it sees a bare `0` and no explicit align char was already given). An
   EXPLICIT `=` align character is rejected outright
   (fstring_format_spec.py:87, "explicit '=' alignment is not supported
   yet"): f"{-5:=8d}" (sign-aware padding with a non-zero width but no '0'
   shorthand) or f"{-5:*=8d}" (sign-aware padding with a custom fill
   character) both fail today, even though `_lower_int_format_spec`/
   `_lower_float_format_spec` already build sign-and-grouping-aware
   padding for the `0` shorthand case (str._pad_and_group_after_prefix/
   _pad_and_group_before_dot, item 4's own grouping+zero-pad bugfix) -
   both already take an arbitrary `fill` parameter (not hardcoded to
   '0'), so generalizing to an explicit `=` align character with a custom
   fill is a small, mostly-mechanical follow-up entirely within the
   existing design, not a new one. (str._pad_after_prefix, the older,
   non-grouping-aware helper these superseded for this exact call site,
   is unused by any format-spec path now, though still defined.)

6. Implicit scalar-to-str boxing - f"{i}" for i: i32

   Still a discretionary design choice, not a technical gap. Deliberately
   requires explicit `int(i)`/`float(f)` conversion today, matching
   print()'s own existing no-implicit-stringification convention
   (lowering.py:3735-3739's operand.type is str_type / else __str__/
   __repr__ dispatch only ever fires against a boxed value - a bare scalar
   has no __str__ of its own to find, so _find_method just fails with "f-
   string requires <type>.__str__() to be available"). Revisit only if
   this stops feeling like the right default.

7. `c` (int-as-codepoint) and `n` (locale-aware) type chars

   Deliberately excluded from FORMAT_SPEC_TYPE_CHARS
   (fstring_format_spec.py's own header comment). `n` specifically because
   nothing else in this compiler is locale-aware anywhere else either (no
   locale infra exists at all); `c` is rare enough not to be worth adding
   speculatively. Would need their own type-char branches in
   validate_int_spec + a new _lower_int_as_codepoint dispatch if ever
   picked up.

8. Unicode grapheme-cluster-aware width/alignment

   str.ljust/rjust/center (and therefore every format-spec width/align
   path built on them) are codepoint-count-aware only, not
   grapheme-cluster-aware - a multi-codepoint emoji or combining-mark
   sequence counts as 2+ "characters" for width purposes, same ceiling
   str's own pre-existing width methods already accepted before this pass
   touched them. Fixing this for real needs a real Unicode grapheme
   segmentation table/algorithm (UAX #29) that doesn't exist anywhere in
   this compiler yet - out of scope for anything format-spec-specific.

9. Nested f-strings inside a format_spec - f"{x:{f'{y}'}}"

   Moot while non-literal specs (#1) are rejected outright:
   _is_literal_format_spec disqualifies ANY format_spec containing a real
   ast.FormattedValue before ever looking at what's inside it, so a nested
   f-string never gets far enough to matter. Falls out automatically once
   #1 is implemented (a nested f-string is just another runtime
   expression the dynamic-spec parser would evaluate like any other).
