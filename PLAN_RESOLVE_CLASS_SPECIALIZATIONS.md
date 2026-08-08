Goal

Extend the generic-FUNCTION-call work already in type_resolver.py (a call site gets
tagged `node.resolved_callee = <the monomorphized Function>`, so lowering.py never
has to build/unwrap a Specialization for a call it can resolve ahead of time) to
classes: RCClass/CStruct/CUnion/TaggedUnion specializations should resolve into
real, "normal-looking" objects before lowering.py ever sees them, the same way.

Current state (re-verified against the actual code, not memory of an earlier session)

- monomorphize.py's `Monomorphizer.monomorphize_class` already builds a real,
  substituted ClassLike (attributes + plain methods substituted into concrete
  types) - the mechanism exists. It's just invoked lazily, on demand, all over
  lowering.py, via `TypeResolver.ensure_resolved` (the one place a Specialization
  gets swapped for its monomorphized form).
- `lowering.py` still has ~25 `isinstance(x, Specialization)` sites. Most of the
  ones left are class-shaped: self-typing inside a generic class's own methods
  (`lower_function`'s self-synthesis), generic construction (`_lower_generic_
  construction_args`, fallible-construction Result-wrapping), overload-group
  receiver substitution, and Ptr[T]/ConstPtr[T] pointer checks.
- `_attr_lookup` already calls `_ensure_resolved(owner_type)` first thing, so
  ordinary field/method access through a Specialization-typed receiver is
  ALREADY basically fine at the point of use - the remaining problem is that the
  receiver's own declared `.type` is still a bare Specialization arriving there
  in the first place, not that `_attr_lookup` mishandles it.
- TODO.txt already documents two real, confirmed bugs this directly explains:
  - "a generic class's own __init__ is never monomorphized when reached through
    plain ClassName(...) construction inside a helper function"
  - a match `case` pattern's bound name, from a *generic* TaggedUnion member,
    gets the member's bare unsubstituted TypeVar as its type, because match
    resolves the pattern's owner class by name (never subscripted), not from the
    subject's own concrete Specialization.

Key insight

A Specialization isn't minted at one AST "call site" the way a function call's
target is - it shows up wherever a TYPE gets built: annotations (`x: Result[i32,
E]`), field/parameter types substituted through `monomorphize_class`/
`monomorphized_function`, and construction call sites. Chasing all ~25 lowering.py
read sites individually is the wrong shape of fix (there will always be a 26th).
Fix Specialization creation at its two SOURCES instead, once each - every
downstream consumer (attribute access, self-typing, isinstance-style queries)
inherits the fix for free, the same leverage `resolved_callee` got for calls
without lowering.py's attribute/receiver code needing to change at all.

Source 1 - substitution. `Monomorphizer.substitute_type_params` (monomorphize.py
:53-79) rebuilds a substituted field/parameter type as a bare Specialization even
when the result is now fully concrete - e.g. `Result[T,E]`'s own `data:
Result$data[T,E]` field, substituted for `Result[i32,MyError]`, becomes
`Specialization(Result$data, [i32,MyError])`: still needs a SECOND
`ensure_resolved` before it's a real CUnion. Fix: when the rebuilt Specialization
is fully concrete (no TypeVar anywhere in its args, recursively) and its base is
a ClassLike, monomorphize it immediately and return the real object instead. One
change, fixes every field type and every generic-method parameter/return type
that flows out of `monomorphize_class`/`monomorphized_function`.

Source 2 - plain annotations. `def f(x: Result[i32,E])`, a global/attribute `y:
Box[i32]` - these build their Specialization directly via discovery.py's
`_get_or_create_specialization`, with no monomorphize.py substitution step
involved at all. discovery.py can't depend on monomorphize.py (stage-1 vs
stage-1.5 layering), so this needs a NEW type_resolver.py step, run once per
Function/Variable right after its own `.resolve()` populates `.type`/
`.return_type`/`.parameters`, mirroring where `resolve_function_body` already
sits in compiler.py's dispatch. Proposed: `TypeResolver.resolve_declared_types
(unit)`, replacing the `for attr in unit.attributes: self.lowering._ensure_
resolved(attr)` loop already duplicated across 4 branches of `compiler.py._lower`,
and adding the equivalent step for `Function.parameters`/`.return_type`, which
doesn't exist today at all.

Together these two eliminate Specialization from every TYPE a Variable/
Parameter/Function can hold, for any concrete instantiation - attribute access,
self-typing, and isinstance-style Result/TaggedUnion shape checks all stop seeing
Specialization without any changes of their own, since they all read `.type`/
`.cls` fields that are already resolved by the time they run.

What this doesn't cover: construction call sites

`Foo(...)` / `Foo[i32,E](...)` - the point a NEW concrete instantiation is
actually decided, from an explicit subscript or inferred from the constructor's
own arguments/expected type - has no annotation to inherit a resolved type from.
It has to be resolved itself, the direct analogue of `_ReferenceResolver.
visit_Call`'s existing generic-function rewrite, as a new rewrite: "generic class
construction resolution". Detects `Foo(...)`/`Foo[T](...)` call sites, resolves
concrete args (explicit: read off the subscript, same as today; implicit: unify
`__init__`'s declared parameter types against argument types via the ALREADY-BUILT
`_unify_type_param`/`_type_of_expr` machinery, plus the surrounding AnnAssign's
own annotation as an expected-type hint - mirrors `_lower_generic_construction_
args`'s existing two-phase strategy), monomorphizes eagerly, and tags the Call
node (`node.resolved_construction = (concrete_cls, concrete_init)`), mirroring
`resolved_callee` exactly. lowering.py's construction-lowering entry point gets
ONE new short-circuit check, mirroring the `_lower_call` hook from last session.

Known hard edge case: fallible construction

`Foo(...)` where `__init__` returns `Result[None,E]` becomes `Result[Foo,E]` per
SYNTAX.md (`_emit_fallible_construction`) - genuinely more involved than the
plain case (needs an `is_err()` call, a branch, Ok/Err re-wrapping, all IR-level
today). Recommend scoping this OUT of the first pass: don't attempt to recognize
a fallible construction site as resolvable at all, so `visit_Call`'s own "any
doubt, bail" discipline naturally leaves it untouched, and lowering's existing,
correct `_lower_generic_construction_args`/`_emit_fallible_construction` keeps
handling it exactly as today - same safety property the function work already
relies on: a miss here is a missed optimization only, never lost coverage.

Phased implementation plan

1. monomorphize.py - `substitute_type_params` eager-monomorphize fix, plus a
   small `_is_concrete(t)` recursive helper (no TypeVar anywhere in a Type,
   recursively through Specialization args) - shared by this and step 4's
   construction-arg unification bail-out, same shape as the function work's own
   TypeVar check in `_try_resolve_generic_call`.
2. type_resolver.py - new `resolve_declared_types(unit: Function|ClassLike)`;
   wire it into compiler.py's dispatch, replacing the 4 duplicated attr-loops and
   adding the missing Function-parameter/return_type equivalent.
3. Re-run the full test suite. Steps 1+2 alone should already fix TODO.txt's
   match-pattern-substitution gap and shrink lowering.py's Specialization
   surface - self-typing and `_attr_lookup`'s own `ensure_resolved` call become
   no-ops in the common case, several `isinstance` branches become provably dead.
4. type_resolver.py - the new construction-call rewrite, non-fallible case only,
   explicit subscript first, implicit inference second (mirrors `visit_Call`'s
   own two-step shape for functions).
5. lowering.py - `resolved_construction` short-circuit hook at the construction-
   lowering entry point.
6. Test against TODO.txt's own confirmed repro (generic RCClass constructed via
   plain `ClassName(...)` inside a helper function) plus new type_resolver_test.py
   coverage mirroring the generic-function tests already there.
7. Cleanup pass - delete now-provably-dead Specialization branches in
   lowering.py, fix compiler.py:24's stale `CompileUnit` comment ("a class
   Specialization never reaches _lower directly" - it does, in the very next
   branch), audit whether `_tagged_union_shape`/`_result_shape`/`_is_RC` become
   simplifiable (probably not removable outright - a still-abstract Specialization
   inside a generic class's OWN body, e.g. `Result[T,E]`'s own `data: Result$data
   [T,E]` field before substitution, is legitimate and permanent, so these stay
   genuine "is this a Specialization, sometimes yes" queries, not a bug).

Explicitly out of scope / unaffected

- Ptr[T]/ConstPtr[T] - Scalar-based, and `Monomorphizer` has NO monomorphization
  support for Scalar at all (`ensure_resolved`'s own comment: ".names/.resolve
  stay raw passthroughs to the abstract base, same as always"). These stay
  Specialization forever - a real, permanent case, not a gap to close.
- Fallible construction (see above) - deferred, not blocked.
- CEnum specializations - CEnum has no `type_params` field in mpy_types.py at
  all (not declared generic-capable); not applicable.

Decisions

- Step 4 will cover implicit inference too (not just explicit Foo[i32,E](...)),
  since that's the shape TODO.txt's confirmed bug actually needs.
- Sequencing: land steps 1-3 (declared-type resolution) first, run the full
  suite, commit, and reassess before starting steps 4-7 (construction-call
  resolution) in a follow-up pass.

Status - step 1 done and verified; step 2/3 blocked on a real architectural
conflict, reverted

Step 1 (monomorphize.py's substitute_type_params eager-monomorphize fix, plus
_is_concrete) is implemented and passes the full suite (558 tests, same 2
pre-existing, environment-only failures - missing kernel32.lib/ntdll.lib on
this machine - as on a clean checkout). Along the way it surfaced a second,
narrower version of the SAME infinite-recursion hazard the eager step
introduces: a generic class's own method that returns a Specialization of its
OWN enclosing class (`Result[T,E]`'s own method returning `Result[T,E]`)
substitutes, for a concrete instantiation, back to that SAME spec -
recursing into monomorphize_class for it again before the first call
finishes building it. Fixed with a `self._building: set[int]` guard (id(spec)
currently mid-construction) that substitute_type_params checks before
eagerly recursing - falls back to handing back the bare (but real, and
eventually-resolved-by-the-in-progress-call) Specialization for that one
self-referential case only.

Step 2/3 (`TypeResolver.resolve_declared_types`, wired into compiler.py to
patch `Function.return_type`/`.parameters[*].type`, `Variable.type`, and
class `.attributes[*].type` in place after their own `.resolve()` populates
them from a plain source annotation) was implemented, then REVERTED after it
broke 18 tests. Root cause, confirmed by tracing one failure
(test_binop_check_mode_emits_or_return) down to `TypeResolver.
_require_result_return`:

    ok = (
        fn is not None
        and isinstance( return_type, Specialization )
        and return_type.base is result_cls
        and len( return_type.args ) == 2
        and return_type.args[1] is error_cls
    )

This is a PERVASIVE idiom throughout lowering.py/type_resolver.py, not an
isolated case - at least 4 more confirmed sites do the identical thing
(lowering.py:381 errdefer's own is_err_fn monomorphization - the exact
"incomplete type" bug its own comment says was already fixed once before;
lowering.py:2269/2290 field-substitution's fn_cls check; lowering.py:2537
construction's pinning_type check; type_resolver.py's `_result_shape`). Once
`resolve_declared_types` swaps a Function's `return_type` (or a Variable's
`type`) from `Specialization(Result,[i32,E])` to the monomorphized CStruct
directly, EVERY one of these checks silently stops recognizing "this is a
Result[T,E] instantiated with args [i32,E])" - `dataclasses.replace(base,
...)` (what monomorphize_class actually builds the concrete object from)
does NOT retain any back-reference to the Specialization it came from, so
`.args`/`.base` become unrecoverable from the object itself. The checked-
arithmetic/or_return/errdefer/generic-construction/generic-method-dispatch
test failures are all downstream of this one gap.

Source 1's fix (substitute_type_params) did NOT hit this - it only touches
fields/parameters being substituted from WITHIN an already-generic
class/method's own body, a narrower surface that apparently doesn't cross
paths with these particular shape-checks in the current test suite. Source
2 (plain declared annotations - the actual Function.return_type/Variable.type
a real Result[i32,E]-returning function or variable carries) is exactly what
these shape-checks are built to recognize, so it collides directly.

What a real fix needs: a monomorphized ClassLike has to stay recoverably
"shaped like Result[i32,E]" even once it's no longer wrapped in a
Specialization - e.g. a new `origin: Specialization|None = None` field on
RCClass/CStruct/CUnion/TaggedUnion, set to `spec` inside monomorphize_class,
plus updating every one of the sites above (and probably more not yet
surfaced - this list came from one failing-test trace, not an exhaustive
audit) to check `t if isinstance(t, Specialization) else getattr(t, 'origin',
None)` instead of `isinstance(t, Specialization)` alone. That's a real,
possibly sizeable audit across lowering.py + type_resolver.py, not a small
addition - bigger than what "land declared-type resolution first, then stop"
was scoped for. compiler.py and type_resolver.py have been reverted back to
their pre-session state (monomorphize.py's step 1 fix is the only change
currently in the working tree); nothing broken is left uncommitted.

Working through the isinstance(Specialization) sites one at a time

Rather than the origin-field audit (which touches every site at once),
decided to work through the ~12 confirmed sites individually, starting with
the ones that don't need it at all.

lowering.py:3033 - overload-group receiver substitution - FIXED, no origin
field needed

This one turned out to be self-contained: monomorphize_class's method-
substitution loop only ever handled plain Function members, explicitly
skipping Overload groups ("a separate, bigger piece of work, out of scope
here" per its own original comment) - every specialization of a generic
class shared the SAME abstract Overload object, so _lower_call had to
reconstruct a substituted copy by hand, per call site, using the RECEIVER's
own (still-Specialization) type as the only place the concrete args were
available. Extended monomorphize_class to also substitute Overload members
(Monomorphizer._substituted_overload) the same way it already substitutes
plain methods - once, up front, memoized - so target.stubs/.implementations
arrive at _lower_call already concrete. Detecting "was this group
substituted" no longer needs the receiver at all: it reads the already-
substituted candidate's own .cls (a Specialization set by monomorphized_
function, the SAME convention already used everywhere else for a single
generic method), so lowering.py's Overload branch shrank from the
cls_args/_for_matching/original_by_id apparatus down to a single `bool(
candidates) and isinstance(candidates[0].cls, Specialization)` check.

Along the way, hit and fixed a real, PRE-EXISTING bug in discovery.py's
_get_or_create_specialization, unrelated to the origin-tracking gap: its
cache key is `base.qualname[args...]`, and a stub and the plain
implementation it binds to share the exact same .qualname (both just
called e.g. `make` - _get_qualname has no notion of "which overload
candidate"). Monomorphizing both through the shared cache made whichever
one got built first get silently handed back for the other too. Only
implementations go through the shared monomorphized_function cache now
(they're real, independently-reachable compile units); stubs - never
independently scheduled/compiled, dispatch-only, no real body - are
substituted directly instead, sidestepping the collision entirely rather
than fixing the cache key itself (which would need spec.qualname to stay
unique per candidate too, for C symbol naming, not just the cache).

Full suite: 558 tests, same 2 pre-existing environment-only failures.
monomorphize_test.py's test_monomorphize_class_leaves_overload_group_
untouched renamed to test_monomorphize_class_substitutes_overload_group
and rewritten for the new (intended) behavior.

Next site to look at: TBD.
