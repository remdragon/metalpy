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

lowering.py:2420 - _try_lower_allocate_call's private-access check - FIXED,
no origin field needed (this one isn't a type-shape query at all)

This site was never actually about recovering a Specialization's args -
it's a pure identity/access-control check ("is the class attached to the
CURRENTLY EXECUTING method the same class __allocate__ is being called
on"), which reads clearer as a named predicate than an inline isinstance-
and-unwrap. Added ScopeMixin.in_private_scope(scope) (mpy_types.py) - `self`
is the target class, `scope` is whatever class the current function
belongs to (Lowering._current_fn.cls, possibly a Specialization for a
monomorphized generic-class method); unwraps scope to its own abstract
base and compares identity. Landed on ScopeMixin rather than RCClass alone
after confirming target_cls at this call site is typed as ClassLike (not
RCClass-only) and that TaggedUnion.__allocate__ genuinely goes through this
exact check in production: every @union member constructor's synthesized
body calls the outer union's own __allocate__ this way (union_storage.py's
_build_member_constructor) - RCClass/CStruct/CUnion/TaggedUnion share no
common base class to hang a single method off other than ScopeMixin, which
all of them (plus CEnum/Function/Module) already use.
_try_lower_allocate_call's own check collapses to `not target_cls.
in_private_scope(fn.cls)`, no local fn_cls/fn_base_cls unwrapping left
inline.

Full suite: 558 tests, same 2 pre-existing environment-only failures.
Confirmed test_allocate_external_call_is_rejected (a @cstruct, not RCClass)
still passes, exercising the ClassLike-generality directly.

lowering.py:2498/2831 - _lower_generic_construction_args and
_lower_class_generic_method_call - not a Specialization gap, but a real
duplication found while reviewing the site

While confirming _lower_generic_construction_args (Box(...) construction
inference) doesn't have the overload-style gap - it doesn't; __init__ is a
plain Function, already correctly substituted by monomorphize_class's
existing method loop, unaffected by and unrelated to the overload fix -
noticed its own "match args against the callee's still-abstract
parameters, lower each with a partial-binding-derived expected-type hint,
apply move hooks, unify to refine bindings" block was duplicated near-
verbatim in _lower_class_generic_method_call (Result.Ok(val) static/
classmethod dispatch - same class-type-param inference problem, just
reached without a receiver instead of via a constructor). Extracted into
Lowering._lower_and_infer_call_args(node, callee, type_params, bindings,
qualname), called from both. Only real behavior change: _lower_generic_
construction_args's move-hook calls used to tag CFG move-tracking with
init.qualname while its own unify calls used target_cls.qualname (an
inconsistency within that one function) - now both use target_cls.qualname
uniformly, matching what _lower_class_generic_method_call (and every other
message in that same function) already did throughout.

Full suite: 558/558 passing (the 2 usually-failing kernel32.lib/ntdll.lib
link tests passed this run too - environment flake, unrelated to this
change).

Next site to look at: TBD.

Step 4 implemented: generic construction call resolution

Went ahead and implemented the deferred "Step 4" (moving construction-call
resolution into type_resolver.py, mirroring the generic-function-call
rewrite) after discussing _lower_generic_construction_args's own `bindings`
mechanism - it isn't a Specialization-caching gap like the overload case,
it's genuine inference (there's no Specialization to preprocess against
until this function decides what it is), but the SAME "redo the inference
at every call site, never pre-resolved before lowering" cost the function-
call work already eliminated for calls applies here too.

Added TypeResolver._try_resolve_generic_construction (type_resolver.py),
called from visit_Call whenever _try_resolve_generic_call itself doesn't
match: detects Foo(...) where Foo is a generic RCClass with a plain,
non-fallible __init__ and no base class, infers the class's own type
params via the SAME _infer_generic_args/_unify_type_param machinery
already built for function calls (generalized to take an explicit
type_params list rather than reading target.type_params, since here it's
the CLASS's params being solved via __init__'s parameters, not the
function's own), and tags the call node - node.resolved_construction =
(concrete_cls, concrete_init) - both ordinary, already-monomorphized
objects, never a Specialization, same discipline as node.resolved_callee.
lowering.py's _try_lower_construct_call gets one new short-circuit branch;
neither concrete_cls nor concrete_init is scheduled by type_resolver.py
itself - that happens the ordinary way when lowering.py actually reaches
the tagged call, exactly mirroring how a resolved_callee Function gets
scheduled by lowering's own ordinary call path rather than by the tagging
pass.

Deliberately out of scope, matching the earlier decision: fallible
__init__ (Result[None,E] return, SYNTAX.md's Result[Foo,E]-wrapping),
explicit Box[i32](...) syntax (not even resolvable by name lookup today -
_try_resolve_callable_namespace has no Subscript-over-a-class handling),
overloaded/subclassed __init__, and bare field=value construction sugar
(no __init__ at all) - all left untouched, falling through to lowering's
unchanged, still fully correct machinery.

Real bug caught and fixed during implementation, not just a missed
optimization: unlike a function call, _lower_generic_construction_args
ALSO pins from an expected_type when one's available (b: Box[i32] =
Box(1) binds T=i32 before the literal argument 1 is even lowered, giving
it an i32 hint directly) - this pass has no expected-type context
threaded through it at all, so trusting a bare literal argument's own
context-free default type (_type_of_expr's Constant handling, i.e.
builtins.int) would silently infer the WRONG type param whenever it
disagrees with what the surrounding annotation actually says. Confirmed
via a real repro (not just reasoning): Box(1) under b: Box[i32] = ...
built and compiled a spurious extra Box[builtins.int] specialization
alongside the correct Box[intrinsics.i32] one, before the fix. Fixed by
adding trust_literals=False to _infer_generic_args (shared with the
function-call path, which keeps trust_literals=True - a bare function
call has no equivalent expected_type-pinning to disagree with, since
Lowering._lower_inferred_generic_call itself never uses expected_type
either) - a class type param only ever inferable from a literal argument
now simply falls through to lowering's own, correct, expected_type-aware
pass, same as any other case this pass can't confidently resolve.

Verified against TODO.txt's own confirmed repro (generic RCClass
constructed via plain ClassName(...) inside a helper function) directly,
plus 3 new type_resolver_test.py tests (successful inference via an
argument's real type, literal-argument bail-out, fallible-init bail-out).

Full suite: 561/561 passing (558 + 3 new).

Fallible __init__() support - DONE, plus a newly-surfaced pre-existing bug
flagged (not fixed)

Extended _try_resolve_generic_construction to also accept a __init__
returning Result[None,_] (fallible, SYNTAX.md), not just plain None -
turned out to need almost nothing beyond widening the bail condition to
`init.return_type is none_type OR _result_shape(init.return_type)[0] is
none_type`: type-param inference only ever looks at __init__'s own
PARAMETERS, never its return type, so fallibility doesn't change how the
class's type args get inferred at all - only what lowering.py does with
the result afterward (Lowering._init_fallibility/_emit_fallible_
construction, both unchanged, still own that decision entirely, reading
it off the CONCRETE, already-substituted init this pass hands them).
test_generic_construction_with_fallible_init_is_left_untagged rewritten
to test_generic_construction_with_fallible_init_is_tagged for the new
behavior.

While verifying, found (but did NOT fix - out of scope, pre-existing, and
not introduced by this session's construction work) a real bug in
_result_shape (type_resolver.py) that the eager-monomorphization fix
(step 1, commit 28e96fa) introduced: _result_shape only recognizes a
Specialization (`isinstance(t, Specialization) and t.base is result_cls`).
If a fallible __init__'s error type is itself the class's OWN type param
(`def __init__(self, v: T) -> Result[None,T]:` - unusual, but legal) then
substitute_type_params's rebuild of Result[None,T] against a concrete T
actually changes something, which step 1's own eager-monomorphize step
then immediately resolves into a real, concrete TaggedUnion/CStruct
object - losing the .args a Specialization would have carried, and with
it _result_shape's only way to recognize "this is Result[None,SomeError]"
at all. Confirmed via direct repro, and confirmed NOT specific to this
session's tagging work: the identical error ("must return None or
Result[None,_], got builtins.Result[...]") fires through the OLD,
untagged Lowering._lower_generic_construction_args path too, by forcing a
literal argument to make trust_literals bail construction resolution.
Every real test/library case uses a FIXED, unrelated error class (Result
[None,MyError]) - substitute_type_params's own early-return ("if all(sa
is a...): return t" - unchanged, no eager-monomorphize triggered)
protects that entirely, which is why nothing has hit this in practice.
The real fix is the "origin" tracking this whole plan doc originally
proposed and then deferred (see the very first sections above) - probably
smaller in scope than first imagined (Monomorphizer could track id
(monomorphized) -> Specialization in an internal dict, set inside
monomorphize_class itself, letting _result_shape/_require_result_return/
_emit_fallible_construction recover the origin without adding a field to
every ClassLike dataclass) - but touches enough call sites (at least
lowering.py:1384's own `value.type.base`, type_resolver.py's
_require_result_return, _emit_fallible_construction's `init.return_type.
args[1]`) that it deserves its own pass rather than folding into this one.

Origin-tracking - DONE

Implemented as scoped down: Monomorphizer gained a plain `self._origins:
dict[int,Specialization]` (id(monomorphized ClassLike) -> the Specialization
it came from), populated the moment monomorphize_class finishes building
one, plus `origin_of(t)` to look it up. No new field on any mpy_types.py
dataclass. TypeResolver.`_as_specialization(t)` wraps it (t itself if
already a Specialization, else origin_of(t)) as the one shared "get me a
Specialization's .base/.args from this, whichever representation it's
currently in" primitive, used by:

- `_result_shape`/`_require_result_return` (type_resolver.py) - the
  original trigger
- `Lowering._unify_type_param`'s recursive Specialization-vs-Specialization
  branch (both copies, lowering.py and type_resolver.py) - a SEPARATE bug
  found while testing: a monomorphized method's own return type (Result.
  Ok's declared Result[T,E], unified against an already-substituted
  Result[None,i32]) hit the identical gap
- a third helper, `_same_type(a,b)`, added for _unify_type_param's
  CONFLICT check specifically (`existing is not actual` used bare identity
  - two argument positions revealing the SAME specialization through two
  DIFFERENT representations - one still a bare Specialization, one already
  monomorphized - triggered a false "inferred as both X and X" - same
  qualname printed twice, since both representations share it)

Two call sites (lowering.py:1384's `value.type.base`, and the identical
`receiver.type.base` in or_return()'s own check) turned out not to need
origin-tracking at all - they were only ever using .base to recover "the
abstract Result class" from an already-Result-shaped value, which
`self.discovery.find_name('Result', node)` gets directly and unconditionally,
sidestepping the representation question entirely. `_emit_fallible_
construction`'s `init.return_type.args[1]` now goes through `_result_shape`
instead.

Verified against the exact repro that started this (a fallible __init__
whose error type is the class's own T, both via type_resolver.py's new
tagging path and forced through the old untagged lowering.py fallback via
a literal argument) - 2 new emitter_c_test.py tests, one per path.

Full suite: 563/563 passing (561 + 2 new).

Source 2 (eager-monomorphize a Function's OWN declared parameter/return
type, not just a substituted one) - DONE, narrow slice

Re-attempted per a direct user prompt after PLAN_COMPILER_BUG_SWEEP.md's
overload_resolution.py fix ("I worked hard to get Specialization out of
lowering.py... how can we make sure that monomorphizer can find the
actual RCClass for list[i32] instead of the Specialization object?") -
that fix had papered over the real gap with an injected `same_type`
predicate inside overload_resolution.py rather than closing it at the
source this section originally proposed.

Landed as `TypeResolver.resolve_declared_types(fn)`: resolves `fn` (if
not already), then runs every concrete, ClassLike-based Specialization
directly typing one of `fn`'s own declared parameters or its return type
through `monomorphize_class`, mutating `param.type`/`fn.return_type` in
place. Wired into the two real production callers of
`overload_resolution.resolve_call` (`TypeResolver._resolve_callable` and
_ReferenceResolver's `_overload_call_return_type`) - NOT the full "every
declared type everywhere" sweep the original plan scoped; that stays
separate, bigger work if it's ever wanted.

Fallout, once a plain declared type could ALSO show up already-
monomorphized (previously only a SUBSTITUTED one could):

- `Compiler._virtual_signatures_match` compared `a.return_type is
  b.return_type` (and per-parameter) by raw identity - an override and
  its base method can end up with one side monomorphized and the other
  not, depending on which was resolved through a real call first. Fixed
  via `_same_type`.
- `Lowering._expr_List`/`_expr_Set` required `isinstance(expected_type,
  Specialization)` directly to read `.args[0]` (the element type) - a
  function's own declared return type used as the list/set literal's
  target type could now already be the real RCClass. Fixed via
  `_as_specialization`.
- The bigger one: `_ReferenceResolver.visit_Match` and every sibling
  narrowing helper in the same class (`_rewrite_tagged_union_truthiness`,
  the `type(x) is T` rewrite, `_try_desugar_type_is_if`'s shape helper,
  `_match_pattern`'s `case None:`/leaf-class branches, and
  `_resolved_union_members` itself) all computed `base = t.base if
  isinstance(t, Specialization) else t` - correct ONLY when a
  non-Specialization `t` means "genuinely non-generic union", which
  stopped being true the moment a GENERIC union's own return type could
  now arrive pre-monomorphized. Using the concrete union as `base`
  directly breaks `_resolve_case_member`'s `owner is not base` check,
  since a case pattern's Owner (`Result.Ok`) is always resolved by NAME
  against the ABSTRACT class - `owner` (abstract) is never identical to
  `base` (now concrete), so EVERY case in the match silently fails to
  match its own pattern. Confirmed via a real, minimal-looking repro that
  took a while to pin down: `csv_reader_test.py`'s
  `test_programs_compile_and_run` merges several independent programs
  that all call `csv.reader()` (whose return type is `Result[csv.Reader,
  csv.Error]`) into one compiled unit; the FIRST program to call it
  triggered the eager monomorphization (fine, its own match had already
  been desugared against the still-abstract type), but every SUBSEQUENT
  program's `match csv.reader(path): case Result.Ok(rr): r = rr ...`
  silently dropped the Ok arm's entire body - `r` never got assigned, so
  a LATER `r.__next__()` failed with a confusing "name 'r' is not
  defined" instead of any error pointing at the match statement itself.
  Fixed every site the same way: `spec = self.resolver._as_specialization
  (t); base = spec.base if spec is not None else t`.

Verified: the new csv_reader_test failure (which single-handedly caught
this) now passes, plus `lowering_test.py`'s
`test_or_return_call_expands_to_or_return_ir_at_call_site` (a pre-existing
test whose OWN fixture built a throwaway Specialization via
`_get_or_create_specialization` for comparison, instead of reading the
real callee's now-possibly-monomorphized `.return_type` - fixed in the
test, not the compiler). Full suite green across 20+ consecutive runs
(this class of bug is inherently order-dependent - which call site
resolves a shared generic callee FIRST determines whether every OTHER
call site sees a Specialization or the real class - so a single green run
proves little on its own).
