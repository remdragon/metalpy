str/f-string formatting: what's still deferred (str.format, dynamic specs,
float specs, __format__ protocol)

STATUS: the two real gaps from the original f-string pass (07cce54) - `!a`
conversion and literal `{expr:spec}` format specs for str/int - are now
implemented (fstring_format_spec.py, lowering.py's _lower_dispatch_format_spec/
_lower_str_format_spec/_lower_int_format_spec, compile_time_transformer.py's
visit_JoinedStr). That pass was deliberately narrow-scoped rather than
open-ended; this file tracks the pieces it left out on purpose, with enough
design texture that each is independently pickup-able later rather than a bare
TODO line.

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

   The float TYPE now exists (f32/f64, aliased float/double - landed
   separately, see arithmetic_mode.py/discovery.py/ir.py/emitter_c.py) but
   it's still a bare scalar intrinsic with no boxed class at all - no
   __str__, no __repr__. This means `f"{x}"` for a bare float doesn't even
   compile yet, same "no implicit stringification" ceiling that already
   applies to i32/bool/every other scalar (see #6) - let's not confuse
   that with a format-spec-specific gap.

   fstring_format_spec.py's parser already recognizes the float type chars
   (FORMAT_SPEC_TYPE_CHARS includes 'fFeEgG%') specifically so a clear,
   named error can be produced instead of a generic one -
   validate_int_spec() special-cases them ("needs a real float type with
   formatting support, which doesn't exist yet", fstring_format_spec.py:175)
   and _lower_dispatch_format_spec does the same for any other operand type
   (lowering.py:3766-3770). Once a boxed float class exists with a real
   __str__/__repr__ AND some way to get a fixed-precision digit string (not
   just a default-precision __str__ text), this needs:

   - validate_float_spec() in fstring_format_spec.py, alongside the
     existing validate_str_spec/validate_int_spec (precision means digit
     count here, not truncation like str's; sign/`#`/width/fill/align all
     reuse the existing FStringFormatSpec fields unchanged).
   - _lower_float_format_spec() in lowering.py, alongside
     _lower_str_format_spec/_lower_int_format_spec, wired into
     _lower_dispatch_format_spec's dispatch chain. Sign handling and
     width/align padding can reuse int._sign_prefix (lib/builtins/
     __int.py)/_lower_pad_by_align's existing shape verbatim (int's own
     design already pushed the
     conditional logic into plain metalpy source methods rather than
     hand-rolled IR specifically so this kind of reuse is easy - see the
     RC bug note in commit 52333fd for why that design choice was made:
     hand-rolling conditional logic directly in lowering.py's IR
     construction is where the tuple-aliasing bug slipped in). Only the
     actual numeric-to-decimal-digits conversion (fixed/exponential/
     general/percent, `#` meaning "always show the decimal point" for
     float specifically) is genuinely new work, and depends entirely on
     what the float class ends up exposing.
   - This may need to land as its own immediate follow-up right after the
     boxed float class itself, rather than bundled with whatever adds
     __str__/__repr__ to float, if that pass doesn't also expose precision
     control.

5. `=` general sign-aware alignment

   Only the `0` zero-pad shorthand's own implicit `=` is supported
   (fstring_format_spec.py's parse_format_spec sets align='=' itself when
   it sees a bare `0` and no explicit align char was already given). An
   EXPLICIT `=` align character is rejected outright
   (fstring_format_spec.py:87, "explicit '=' alignment is not supported
   yet"): f"{-5:=8d}" (sign-aware padding with a non-zero width but no '0'
   shorthand) or f"{-5:*=8d}" (sign-aware padding with a custom fill
   character) both fail today, even though `_lower_int_format_spec`
   already has to build sign-aware padding for the `0` shorthand case
   (via str._pad_after_prefix) - generalizing it to accept an arbitrary
   fill character instead of the hardcoded '0' is a small, mostly-
   mechanical follow-up entirely within the existing design, not a new
   one.

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
