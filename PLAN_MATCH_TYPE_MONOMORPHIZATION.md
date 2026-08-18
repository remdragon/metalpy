# Scope: `match type(x): case ConcreteT(...): ...` over a generic method's own type-parameter-typed value

**STATUS: implemented** (Direction A below), on this same worktree
(`generic-type-match-dispatch`), after this document's original
investigation - see "Recommendation" at the bottom for why: the user asked
for it explicitly, overriding the "wait for the Overload+TypeVar task"
default recommendation this doc originally landed on. `type_resolver.py`'s
`_try_fold_match_type` (sibling of the pre-existing `_try_fold_is_rc_if`)
implements it; `emitter_c_test.py`'s `GenericMatchTypeMonomorphizationRealCompileTests`
and `lowering_test.py`'s `test_match_type_*` tests cover it. Verified against
MSVC/clang/gcc(WSL) - full suite, 1298 tests, 0 failures on all three.

## Context

Motivating case: the originally-proposed generic form of `str.__call__`:

```python
class str:
	@staticmethod
	def __call__[T]( other: T ) -> str:
		match type( other ):
			case str( other ):
				return other
			case _:
				return other.__str__()
```

Only the plain, non-generic `str.__call__(x: str) -> str: return x` overload
is shipping in the `call-dunder-dispatch` worktree's current task. This
document investigates whether/how to add the generic, `type()`-branching
form as a follow-up, per the request that spawned it.

**The gap, confirmed by reading the code (not just the existing test
docstring that motivated this investigation):**
`type(x) is T` / `match x: case T(...):` exist today ONLY as textual sugar
over the pre-existing tagged-union-narrowing machinery -
`type_resolver.py`'s `_rewrite_type_is_comparison`/`_type_is_shape`/
`visit_Match` all require x's *static* type to already be a `TaggedUnion`
(or `Specialization` of one) before any of this fires. A bare generic
`other: T` parameter is neither - at the point `match type(other):` is
written, `T` is just a `TypeVar`, not a union - so none of this sugar
applies, and (per `visit_Match`'s own union-only logic) the snippet above
would not compile as written.

`type_resolver.py` (~line 4090-4119) already has a comment documenting a
deliberately-deferred, differently-scoped extension point here: making
`type(x)` usable as a general value-producing expression *outside* an
is/instanceof comparison. That's not what this document is about - the need
here is narrower (compile-time branch selection during monomorphization),
and the existing note doesn't cover it.

## Overlap with the Overload+TypeVar-dispatch task - **read this first**

There's a separately-spawned task, worktree `overload-typevar-dispatch`,
titled "Add TypeVar-matching + monomorphization to Overload dispatch." As of
this investigation that worktree is still sitting at the same commit as
`master` (no new commits, no dedicated PLAN doc yet) - it hadn't produced a
result to build on, but its *charter* matters a great deal to this task's
own scope.

The exact motivating case above - "one concrete fast path (str-to-str
identity) plus a generic fallback (call `.__str__()` on anything else)" - is
naturally expressible as two ordinary `@overload` candidates instead of one
generic method with an internal `match type(...)`:

```python
class str:
	@overload
	@staticmethod
	def __call__( other: str ) -> str: ...
	@overload
	@staticmethod
	def __call__[T]( other: T ) -> str: ...

	@staticmethod
	def __call__( other: str ) -> str:
		return other
	@staticmethod
	def __call__[T]( other: T ) -> str:
		return other.__str__()
```

**Confirmed by reading `overload_resolution.py` directly**: this does NOT
work today, and for a very specific, checkable reason - `Type.leaves()`
(the base case, `mpy_types.py:50-55`) returns `[self]`, so a `TypeVar`-typed
parameter's own "required leaves" is literally `[T]`, the bare `TypeVar`
object. `_contains`/`_intersect`/`_subtract` in `overload_resolution.py`
compare leaves via `same_type` (identity, or `TypeResolver._same_type` in
production) - a real call's argument type is never identical to the
abstract `TypeVar` object `T`, so a `TypeVar`-typed candidate can never
`_contains`-match any real argument slice. It just silently never wins,
for any call, ever. This is a genuine, confirmed gap in
`overload_resolution.py` today, not a hypothetical - and it is *exactly*
the gap the `overload-typevar-dispatch` task exists to close (treating an
unmatched `TypeVar`-typed candidate as the catch-all for whatever leaf
types no concrete candidate covers, then monomorphizing it against
whichever leaf type it actually ends up dispatched with per call site).

**Conclusion**: once that sibling task ships, it covers this document's
motivating use case *more generally* than match-type-pruning would (N
concrete overloads + one generic fallback, resolved per call site via the
existing, well-tested overload-ranking machinery) and *more idiomatically*
(ordinary static dispatch, no new "prune a match statement's arms during
monomorphization" mechanism needed at all). Match-type-pruning is not
needed for this specific motivating case if Overload+TypeVar dispatch
lands.

**Recommendation: do not implement match-type-pruning now.** Revisit only
if a future concrete need arises that Overload+TypeVar dispatch genuinely
can't express (see "Recommendation" below for what that would look like).
The design below is written up in full so it doesn't need re-deriving from
scratch if that happens.

## Design for a future match-type-pruning feature, if one is ever needed

### Direction A (recommended): monomorphization-time AST pruning, sibling of `_try_fold_is_rc_if`

`type_resolver.py` already has a directly-analogous, shipped mechanism:
`_try_fold_is_rc_if` (~line 4438) folds `if compiler.is_rc(T): A else: B`
(T a generic class's own type param) to just A's or B's statements at
compile time, keyed on a monomorphization's own concrete type-parameter
binding. It works by relying on a discipline `resolve_function_body`'s own
docstring documents: this whole AST-rewrite pass runs *twice* per generic
method - once against the shared, abstract body (T still an unbound
`TypeVar`, the fold declines and leaves the `if` untouched) and once more
against each monomorphized copy's own independently deep-copied body (built
by `Monomorphizer._build_monomorphized_function`, which substitutes T into
`.names` before this pass ever runs on it) - where T is now concrete and
the fold actually fires. This is precisely the mechanism the requesting
task described as "genuinely new" - it isn't; it already exists for `if`,
just not yet for `match`.

**Proposed shape** - a new `_try_fold_match_type`, hooked into `visit_Match`
the same way `_try_fold_is_rc_if` is hooked into `visit_If`
(`folded = self._try_fold_is_rc_if(node); if folded is not None: return folded`
runs *before* `visit_If`'s own union-narrowing logic; `visit_Match` needs
the identical early-exit, since its existing body otherwise assumes the
subject's `base` is a `TaggedUnion` and will mishandle - most likely
mis-report as "not a union type" - a bare-TypeVar subject):

1. **Detect the shape**: `match type(<Name>):` where `<Name>`'s own
   *declared* type is a bare, single type parameter of the enclosing
   generic function/class (not embedded inside a larger `Specialization`,
   not `type(x.attr)` or any other non-Name expression). On the abstract
   body this always declines (the type is still a `TypeVar`); on a
   monomorphized copy, resolve it via the same `self.locals`/
   `_try_resolve_namespace` machinery `_try_fold_is_rc_if` already uses.
   Any doubt at all - decline silently, exactly like every other rewrite in
   this class - and fall through to today's ordinary (TaggedUnion-only)
   match handling, which reports its own correct error for a genuinely
   unsupported subject.
2. **Resolve each arm**: for a monomorphized (concrete-T) body, each
   `case ConcreteClass(binding):` arm's class name resolves via
   `_try_resolve_namespace`, compared against the concrete bound type with
   `TypeResolver._same_type` (identity-safe against the eager-monomorphize
   duality `overload_resolution.py`'s own module comment already documents
   for the identical reason).
3. **Fold**: exactly one arm should structurally win (or the wildcard
   `case _:`). Since the subject was never a real union, no runtime branch
   is emitted at all - `visit_Match` is replaced wholesale by only that
   arm's own visited statements, the same "return folded statements, no
   `ast.If`/`Cmp` left behind" shape `_try_fold_is_rc_if` already uses.
4. **Binding is simpler than union-case narrowing, not harder**: the
   winning arm's single-name capture (`case str(other):`) becomes a plain
   rebind, `other = <original subject expr>` - there's no `.data.v_X`
   payload to extract (the subject already *is* exactly that concrete
   type, not a union leaf), so this skips the extraction step
   `_match_union_member` needs entirely.
5. **No match, no wildcard**: a hard compile error at monomorphization time
   ("no match arm covers the concrete type X for this instantiation"), not
   a silent fallthrough - matches this class's posture elsewhere (decline
   silently only while genuinely unsure the shape applies; once the shape
   is confirmed and resolved, fail loudly rather than swallow it).
6. **Carve-outs**, mirroring `_try_fold_is_rc_if`'s own restrictions to keep
   this contained: only a bare single-name-capture class pattern per arm
   (`case str(other):`, not deeper positional/keyword deconstruction of a
   plain leaf type - that's a separate, unrelated feature); only a bare
   `type(<Name>)` subject, not any other expression shape.

Rough size: comparable to `_try_fold_is_rc_if` itself (~50-70 lines) plus a
3-line `visit_Match` hook, plus real-compile tests (per this repo's
`RealCompileMixin` convention) covering: the str-fallback arm, a non-str
concrete arm, a wildcard-only arm covering everything, the missing-coverage
compile error, and confirming the still-abstract body correctly declines
(doesn't fold prematurely). Verification would need MSVC + clang + gcc
(WSL), matching this repo's mandatory multi-compiler policy for anything
touching `type_resolver.py`'s codegen-affecting rewrite passes.

### Direction B (rejected): synthesize a throwaway single-member union per instantiation, reuse the existing TaggedUnion-only machinery unchanged

The alternative floated when this task was spawned: for each concrete T a
generic method gets monomorphized against, synthesize a one-off
single-member "type-erased" `TaggedUnion` wrapping just that T, coerce the
subject into it, and let the *existing*, unmodified `type(x) is T`/`match`
rewrite machinery run against it as-is.

Rejected:
- It requires real per-instantiation type synthesis - a fresh
  `UnionStorage` entry, a real tag+payload C struct, a real construction
  call - to answer a question Direction A resolves as a pure compile-time
  identity comparison with **zero** runtime representation. Strictly more
  machinery for strictly less information (a single-member union's tag
  check is always trivially true).
- It doesn't naturally extend to multiple concrete-type arms
  (`case str(...): case int(...): case _:`) without synthesizing a real
  *multi*-member union per instantiation and reimplementing this
  document's own case-resolution logic on top of it anyway - at which
  point it has all of Direction A's complexity plus the synthesized-union
  overhead, for no benefit.
- The existing `type_resolver.py` extension-point comment this direction
  was inspired by (~line 4090-4119) is scoped differently: making
  `type(x)` a general value-producing expression usable *outside*
  is/instanceof comparisons. That's a different problem from "prune a
  match statement's arms during monomorphization," and reusing it here
  would be forcing an unrelated mechanism to fit.

## Recommendation (original) / Outcome (actual)

The original recommendation from this investigation was to **not implement
match-type-pruning yet**, on the reasoning that the motivating use case is
better served by `overload-typevar-dispatch`, and implementing this first
risked shipping a narrower mechanism that task would make redundant.

The user, informed of that reasoning, asked for it to be implemented
anyway. That's a legitimate call the requester gets to make even when a
"wait and see" default was the more conservative option - Direction A was
always the correct design if/when it was going to be built (small,
well-contained, sibling of an already-shipped mechanism, not new compiler
architecture), so there was no unsound-implementation risk in proceeding.
**Direction A is implemented, tested, and merged into this worktree** (see
the STATUS note at the top of this document). `overload-typevar-dispatch`
remains worth pursuing independently - it still covers a broader set of
cases (N concrete overloads, not just one match's worth of arms) - but the
two are no longer mutually exclusive; a future `lib/` author can reach for
whichever shape fits the specific method better.

### Implementation notes (found only by actually building it)

The design above turned out to need one correction once real code exercised
it: falling through to visit_Match's ordinary (TaggedUnion-only) handling
when the subject's type was still an abstract `TypeVar` was WRONG - that
handling calls `discovery.fail()` unconditionally the moment it can't
resolve a union, which would have recorded a permanent, spurious compile
error on the still-abstract body of every generic method using this shape,
even though the statement resolves cleanly once monomorphized. The shipped
`_try_fold_match_type` instead returns the match statement completely
unvisited (`[node]`) specifically when the subject is confirmed to be a
still-unbound `TypeVar` - holding it for the monomorphized copy's own
second pass - while every OTHER kind of "can't fold this" (not a `type(
Name)` subject at all, an undeterminable type, or a real TaggedUnion)
declines with a plain `None` and reproduces today's pre-existing error for
that shape, unchanged. See the method's own docstring in `type_resolver.py`
for the full reasoning; this distinction is the one part of the design that
wasn't obvious from `_try_fold_is_rc_if`'s own precedent (that fold's
decline path is harmless because nothing else in this pass attaches any
meaning to `compiler.is_rc(T)` outside the fold itself - `match` has no
equivalent safety net).
