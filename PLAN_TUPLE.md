tuple[T0, T1, ...] - heterogeneous fixed-arity value groups

Why

This compiler has no way to group values of different types into one thing
today, and there are two concrete, already-written pieces of stdlib code
hacked around that gap, not just an abstract nice-to-have:

- lib/builtins/__int.py's int.divmod() (line 497) returns a hand-rolled
  `@cstruct class DivMod: quotient: int; remainder: int` (line 60) instead
  of `Result[(int,int), IntError]` - its own comment says exactly why:
  "this language has no tuple type at all (confirmed directly: a bare
  `(int, int)` return-type annotation crashes discovery.py outright,
  AttributeError on a None .qualname, since no visit_Tuple exists anywhere
  to handle one - tuples are simply never a supported type here)."
  __floordiv__/__mod__ (lines 557-563) both call divmod() and immediately
  destructure `.quotient`/`.remainder` off the result.
- str.partition() (TODO.txt line 468, "missing helper methods") is being
  implemented in a separate, concurrent session and needs to return a
  3-way split (before, separator, after) - Python's own str.partition()
  returns a 3-tuple; without tuple here it would otherwise need its own
  one-off @cstruct exactly like DivMod's, duplicating this same workaround
  a second time.

ast.Tuple is only ever parsed as an ANNOTATION-subscript shape (Dict[K,V]/
Callable[[...]]'s multi-arg slice) today - never lowered as a VALUE
anywhere in lowering.py. Both known consumers above happen to be
HOMOGENEOUS (int,int) / (str,str,str) - worth noting since it means the
absolute minimum viable slice could theoretically special-case "same type
repeated," but the type-system cost of supporting genuinely heterogeneous
elem_types is identical (TupleType.elem_types is a list either way - see
Implementation), so there's no reason to build the narrower thing.

The core design problem tuple poses that list[T]/dict[K,V] never did:
list[T] and dict[K,V] are ordinary fixed-arity generics (type_params=[T] /
[K,V]), matched 1:1 against a subscript by discovery.py's visit_Subscript
and monomorphized by substituting into one shared attribute layout
(monomorphize.py's monomorphize_class). tuple[int, str, bool] is variadic
arity AND heterogeneous - every distinct element-type LIST needs its own,
differently-shaped backing layout. There is no existing "arity-N generic"
mechanism to reuse; the one existing precedent for a variadic type-LIST
shape is Callable[[Arg1,Arg2,...], Ret]/Closure[[...],...], and that one
is deliberately NOT a real ClassLike with members - it's a bare, receiver-
less function-pointer SHAPE (mpy_types.CallableType), nothing is ever
"constructed" as one. tuple needs the opposite: a real, constructible,
destructible, RC-aware value - closer in spirit to TaggedUnion's own
synthesized `data` payload (one field per member, built lazily on first
use by union_storage.py's UnionStorage) than to Callable[...].

Scope for this pass

In scope:
1. tuple[T0, T1, ..., Tn] (n >= 1, i.e. arity >= 2) resolves as a real type
   - mpy_types.TupleType, interned by element-type list the same way
   CallableType is interned by (arg_types, return_type).
2. A synthesized backing RCClass (fields _0.._n, a synthesized __init__)
   built lazily on first use, memoized per distinct TupleType - so
   construction, RC-field retain/release, and destruction all fall out of
   EXISTING RCClass machinery (type_resolver.py's
   _synthesize_rcclass_destructor) with zero new emitter code for the
   struct itself. This mirrors UnionStorage.get()'s "tag/data synthesized
   lazily, memoized, scheduled separately" shape exactly.

   RCClass, not CStruct, is a correctness choice, not a style one:
   cfg.py's own header comment states "v1 deliberately doesn't reach into
   struct/union FIELDS ... a binding whose type has no RC leaves at all
   (CStruct/CUnion/CEnum/Scalar, or a TaggedUnion with no RC leaves) is
   invisible here", and cfg.py's rc_leaves()/is_rc() confirm it structurally
   - a CStruct is never itself treated as RC-carrying, regardless of its
   own field types. DivMod (int.divmod()'s hand-rolled @cstruct, both
   fields typed `int` - and `int` is a plain, undecorated `class int:` in
   __int.py, i.e. a real heap-boxed RCClass, not a value type) is a live
   example of exactly this gap: a DivMod-typed binding going out of scope
   is invisible to the CFG's decref walk, so its two `int` fields are never
   released there. A CStruct-backed tuple would inherit this same silent-
   leak behavior for any RC element (str, list[T], dict[K,V], another
   tuple, ...); an RCClass-backed tuple does not, because the CFG already
   treats a whole RCClass binding as one RC leaf to decref, cascading into
   the synthesized destructor that correctly tears down every field. This
   also means migrating int.divmod() to return tuple[int,int] (a real
   candidate follow-up once this lands, not part of this pass) would fix a
   latent leak as a side effect, not just remove a workaround.
3. Tuple literal syntax in value position: `t = (1, "hello", true)` -
   the first real handling of ast.Tuple as a VALUE anywhere in lowering.py.
   Lowered by inferring each element's own type, interning the resulting
   TupleType, and rewriting to a synthesized construct-call against the
   backing class (reuses _try_lower_construct_call, the same "real __init__
   path" every other class construction already goes through - no new
   Allocate/field-assign IR needed).
4. Element access via a COMPILE-TIME-CONSTANT integer index only: `t[0]`,
   `t[1]`, ... rewritten to plain attribute access (`t._0`) at lowering
   time. A non-constant index (`t[i]`) or an out-of-range constant index is
   a compile error, not a runtime one - a heterogeneous tuple has no single
   type to give a runtime __getitem__, so this can't be the ordinary
   __getitem__ dispatch list[T]/dict[K,V] use.

Deferred (flagged, not attempted this pass):
- Multi-assignment / unpacking (`a, b = t`) - needs new ast.Assign target-
  shape handling (a tuple/list target), orthogonal to tuple's own type and
  not needed for tuple VALUES to exist and be constructed/read. Natural
  follow-on once tuple values exist to unpack from, same relationship
  PLAN_CALLABLE.md's own "lambda is sugar once function-references-as-
  values exist" deferred item had to Callable[...].
- `return a, b` multi-return sugar - mechanically "ast.Tuple in return
  position", likely falls out for free once tuple literals exist AND the
  function's declared return type is a matching tuple[...] annotation, but
  NOT assumed here - needs its own explicit verification once (3) lands,
  not bundled into this pass's initial implementation.
- Iteration (`for x in some_tuple`) - lowering.py's existing
  _lower_for_over_indexable duck-types on __len__ + a single-return-type
  __getitem__; a heterogeneous tuple structurally cannot satisfy a single-
  return-type __getitem__. Likely permanently N/A for heterogeneous tuples,
  not attempted here even for the homogeneous case.
- Runtime/variable index into a tuple (`t[i]`) - statically unsound for
  heterogeneous elements (mypy rejects this too); not attempted, not even
  for a homogeneous special case.
- __eq__/__hash__/__lt__/other comparisons, __repr__/str() conversion,
  bool(t) truthiness - deferred to a follow-up once construction/access
  land and something in lib/ actually needs one of these (e.g. tuple as a
  dict key needs __hash__ specifically - no forcing case for it yet).
- Zero-element `tuple[]` / one-element `tuple[T]` - Python's own ast.Tuple
  parse requires a trailing comma to disambiguate a 1-tuple literal `(x,)`
  from a plain parenthesized expression `(x)`; arity-0/1 edge cases are
  flagged to confirm during implementation, not pre-solved here. This pass
  targets arity >= 2 only.
- Any FastLock-style thread-safety wrapper - NOT applicable. list[T]/
  dict[K,V] need one because they're mutable after construction; a tuple's
  fields are only ever written once, by its own synthesized __init__, so
  (per PLAN_LIST_T.md's own reasoning) atomic refcounting alone already
  makes a constructed tuple safely shareable across threads with no lock.
- A hand-written lib/builtins/__tuple.py "RawTuple" shared core - there is
  no shared core to factor out. RawList/RawDict exist because every list[T]
  /dict[K,V] instantiation shares ONE erased layout; every distinct
  tuple[...] element-type list is a genuinely different layout, so there is
  nothing type-erased in common across instantiations to compile once. This
  is a real, surprising-relative-to-every-other-container difference, worth
  stating explicitly so nobody goes looking for a RawTuple that shouldn't
  exist.

Precedent reused, not invented fresh

- union_storage.py's UnionStorage.get() - "synthesize a real ClassLike,
  memoize by id() on first use, schedule its payload separately from the
  outer type" is the exact shape a new TupleStorage copies structurally,
  including UnionStorage's own stated reason for standing apart from
  Lowering ("meant to be usable standalone").
- union_storage.py's _build_member_constructor - the technique for
  synthesizing a real ast.FunctionDef body (fix_missing_locations, wrapped
  in a real Function, registered into .names under a synthesized `$`-
  prefixed name findable only via that class's own scope) is exactly what
  TupleStorage needs for its synthesized __init__(self, _0: T0, ..., _n:
  Tn) -> None.
- discovery.py's visit_Subscript textual special-case for Callable[...]/
  Closure[...] (recognized ahead of the ordinary type_params-arity-checked
  generic path, by bare name, before falling through to `base =
  self.visit(node.value)`) - tuple[...] gets the same kind of special case,
  keyed on `node.value.id == 'tuple'` (lowercase, matching list[T]/dict
  [K,V]'s own established casing rather than Callable/Closure's
  capitalized compiler-syntax convention, since from a user's perspective
  tuple reads as an ordinary builtin, not compiler magic).
- discovery.py's _get_or_create_callable_type interning cache - the
  template for a new _get_or_create_tuple_type, keyed on the element type
  list's own qualnames the same way _get_or_create_union canonicalizes its
  key.
- type_resolver.py's _synthesize_rcclass_destructor - TupleStorage's
  synthesized backing RCClass gets correct per-field RC teardown for free,
  the same way list[T]'s __inner/__lock fields and dict[K,V]'s own fields
  need no hand-written __del__ today.
- lowering.py's _lower_bound_method_closure (building a ClosureType value)
  - NOT _try_lower_construct_call/_build_member_constructor as originally
  planned (see the IMPLEMENTATION DEVIATION note above): it already
  establishes the simpler precedent of building a synthesized RCClass
  instance directly via ir.Allocate's own field=value shape, no __init__
  Function/ast.FunctionDef/call-lowering round-trip needed - tuple literal
  lowering (_expr_Tuple) copies that shape exactly, one step simpler than
  the plan originally called for.
- type_resolver.py's ensure_resolved - the SINGLE place a Specialization
  already gets swapped for its real, substituted form, used unconditionally
  by ~15+ existing call sites throughout lowering.py/type_resolver.py.
  TupleType reuses this EXACT chokepoint (a sibling `isinstance(obj,
  TupleType)` branch) rather than inventing a second "resolve me" protocol
  - every existing caller that already knows how to treat "whatever
  _ensure_resolved handed back" as the real type transparently receives
  the backing RCClass, with no changes needed at any of those call sites.

Implementation

1. mpy_types.py: new TupleType(Type) - elem_types: list[Type]. Not a
   ScopeMixin (mirrors CallableType - no members of its own on the TYPE;
   the members live on the synthesized backing RCClass instead). Add a
   self-caching `backing: RCClass|None = None` slot, same spirit as
   Specialization.monomorphized, populated by TupleStorage.get() the first
   time this exact TupleType is touched.
2. tuple_storage.py (new file, mirrors union_storage.py structurally):
   TupleStorage(discovery, schedule). get(tt: TupleType) -> RCClass builds
   (memoized on TupleType.backing itself, not a separate id(tt)-keyed side
   table - see mpy_types.py's own comment on that field) a synthesized
   RCClass named via a Specialization-style bracketed qualname (f'tuple[{
   ",".join(t.qualname for t in elem_types)}]', mirrors _get_or_create_
   specialization's own convention - emitter_c.py's mangle_qualname already
   turns '.','[',',',']' into legal C identifier fragments generically, no
   emitter changes needed) and attributes _0.._n (Variable per elem_types
   [i]). Also keeps a reverse map (id(backing) -> TupleType) for lowering.
   py's _expr_Subscript to recognize a tuple-backed receiver. Schedules the
   backing class itself (same schedule-the-payload-separately discipline
   UnionStorage.get() uses for its own CUnion).

   IMPLEMENTATION DEVIATION (found building this): no synthesized __init__
   after all. _lower_bound_method_closure (building a ClosureType value)
   was found to already establish a simpler precedent than union_storage.
   py's own _build_member_constructor - it builds its RCClass instance
   directly via ir.Allocate's own field=value shape, with no __init__
   Function/ast.FunctionDef synthesis at all. tuple's own construction
   (_expr_Tuple, below) copies THAT shape instead: lower each element,
   emit ir.Allocate(cls=backing, fields={'_0':...,...}) directly. This
   means TupleStorage.get() never builds an ast.FunctionDef/Function at
   all - simpler than originally planned, and correctly so (a real
   __init__ would only be needed to support `tuple[i32,str](1,"a")` bare-
   call construction syntax, which is out of scope for this pass - see
   Scope's own construction-via-literal-only framing).

   IMPLEMENTATION DEVIATION #2 (found while smoke-testing): the backing
   RCClass cannot use file=None/line=None the way UnionStorage's own
   synthesized CUnion payload does. type_resolver.py's _synthesize_
   rcclass_destructor copies cls.file/.line directly onto the synthesized
   $$__destructor__ Function it builds for every concrete RCClass, and
   lowering.py's _find_module_for looks up the owning module by matching
   that file exactly against a real Module - file=None crashes ("no
   module found owning ...$$__destructor__ (file=None)") the first time a
   tuple is actually constructed, since no Module has file=None. Fixed by
   using discovery.module_stack[-1].file/.line instead, the same "whichever
   module is currently active" convention discovery.py's own _get_or_
   create_closure_type already uses for exactly this reason. UnionStorage's
   own CUnion payload gets away with None because a CUnion is never RCClass
   and therefore never reaches _synthesize_rcclass_destructor at all - this
   distinction wasn't obvious until it crashed a real end-to-end run.
3. discovery.py: _get_or_create_tuple_type(elem_types: list[Type]) ->
   TupleType interning cache (mirrors _get_or_create_callable_type).
   visit_Subscript gets a new special case ahead of the ordinary generic
   path: `node.value.id == 'tuple'` with `isinstance(node.slice, ast.Tuple)`
   and `len(node.slice.elts) >= 2` (arity 0/1 rejected with a clear error
   for this pass, per Scope above), visiting each slice element and
   returning _get_or_create_tuple_type(elem_types).
4. lowering.py:
   - New _expr_Tuple, dispatched for ast.Tuple in value position (net new -
     today ast.Tuple only appears in annotation-subscript position). Lowers
     each element expression first (each field's own RC-retain emission via
     self._cfg.field_value(...), the exact per-field loop _lower_allocate_
     fields already uses for every other class's field=value construction
     sugar), interns the TupleType via discovery._get_or_create_tuple_type
     from the elements' own real lowered types, resolves it to the backing
     RCClass via _ensure_resolved, schedules sys.alloc[T]/sys.free/__del__
     via _schedule_rcclass_construction (same as every other ir.Allocate
     construction site), then emits ir.Allocate(cls=backing, fields={'_0':
     ...,...}) directly - see the IMPLEMENTATION DEVIATION note under (2)
     above for why this needs no synthesized __init__ or call-lowering
     round-trip at all.
   - Wherever `obj[i]` subscript-in-value-position is currently dispatched
     (_expr_Subscript): new branch checked BEFORE the ordinary Ptr/ConstPtr
     GetItem fallback (inside the existing `getitem_fn is None` branch,
     since a tuple-backed receiver has no real __getitem__ either) - if
     obj.type (resolved) is a TupleStorage-backed RCClass (found via
     TupleStorage's own reverse lookup) and the index is ast.Constant(int)
     in [0, n) (bool explicitly excluded - it's an int subclass in
     Python's own ast), rewrite to a direct _attr_lookup + ir.GetAttr(obj,
     f'_{index}') using the ALREADY-lowered receiver operand (never a
     freshly re-lowered/re-evaluated ast.Attribute node - re-lowering
     node.value a second time would double-evaluate any side-effecting
     subexpression, e.g. `f()[0]` calling f() twice); otherwise (non-
     constant index, or out-of-range constant, on a tuple-backed receiver)
     a clear compile error naming the constant-index-only requirement.
   - Lowering.__init__ (alongside _union_storage/_monomorphizer, already
     borrowed from TypeResolver rather than constructed locally): borrow
     type_resolver.tuple_storage the same way.
   - type_resolver.py: TypeResolver.__init__ constructs self.tuple_storage
     = TupleStorage(discovery, self.schedule), alongside union_storage/
     monomorphizer (which Lowering already borrows rather than builds
     itself - tuple_storage follows that same ownership, not a new pattern
     invented for this). TypeResolver.ensure_resolved (the SAME central
     "swap for the real thing" chokepoint the existing Specialization
     branch already uses, called from ~15+ sites across lowering.py/type_
     resolver.py) gets a sibling branch: `if isinstance(obj, TupleType):
     return self.tuple_storage.get(obj)`. This was found to be the key
     simplification versus the original plan - every EXISTING _ensure_
     resolved call site (parameter/return-type scheduling, _attr_lookup,
     _find_method, etc.) transparently receives the backing RCClass
     instead of the bare TupleType with ZERO changes needed at any of
     those call sites, exactly mirroring how a bare Specialization is
     already swapped for its monomorphized form everywhere, completely
     unremarked-on by 99% of the code that touches it.
5. No lib/builtins/*.py changes needed for tuple's own implementation - see
   Scope's "no RawTuple" item above for why. (lib/builtins/__init__.py may
   later gain real stdlib code that USES tuple, e.g. dict.items(), but
   that's separate follow-on work, not part of landing tuple itself.)

Verification

- lowering_test.py: tuple[i32,str] annotation resolves to a TupleType,
  interned (two annotations spelling the same element list share one
  object, mirrors CallableType's own interning test); tuple literal
  `(1, "a")` lowers to a construct-call against the synthesized backing
  RCClass; `t[0]`/`t[1]` constant-index lowers to GetAttr; `t[2]` (out of
  range) and `t[i]` (non-constant index) are both rejected with clear
  errors; two structurally different tuple[...] types never share a
  backing class.
- emitter_c_test.py real compile-and-run: new TupleTests(CompilerTestCase)
  mirroring ListGenericTests'/DictTests' exact pattern (construct source as
  a triple-quoted string, assert no discovery errors, _assert_compiles_and_
  runs, distinguish pass/fail by return code). Covers: construct
  tuple[i32,str,bool], read back each field by constant index, verify
  values; an RC element (tuple[str,i32]) constructed then dropped - no
  leak/double-free (destructor cascade via _synthesize_rcclass_destructor,
  same style PLAN_LIST_T.md's own thread-safety section verifies RC-field
  teardown); nested tuple (tuple[tuple[i32,i32], str]) as a stretch goal,
  not required for this pass to land.
- str.partition() (the concurrent session's own pending work) needs exactly
  what (3)/(4) provide: construct tuple[str,str,str], read back each of the
  3 fields by constant index. Worth a direct sanity check against that
  session's actual signature once both land, since it's the nearest real
  consumer.
- Full python3 tests.py green after landing, count cited at the landing
  point (baseline to be recorded when work starts).

Coordination: str.partition() is being implemented in a separate,
concurrent session right now and needs tuple to land underneath it rather
than invent its own one-off @cstruct workaround. Worth flagging to that
session once this plan is underway so it can land its own return type as
tuple[str,str,str] directly instead of a throwaway struct that would need
migrating later.

STATUS: landed. All "In scope" items (1-4) implemented as described above,
with the two IMPLEMENTATION DEVIATIONs noted under (2) (no synthesized
__init__; module_stack[-1] file/line instead of None) - both found by
actually running end-to-end smoke tests, not anticipated up front. Baseline
was 759 passing (python3 tests.py, before this work started). Landed with
771 passing: +8 lowering_test.py tests (interning identity, literal ->
Allocate with positional fields, arity-2-minimum on both the literal and
the annotation, constant-index -> GetAttr, out-of-range/non-constant index
rejection, two distinct shapes never sharing a backing class) and +4
emitter_c_test.py TupleTests real compile-and-run tests (heterogeneous
i32/str/bool construct-and-read-back with deliberately-wrong-value checks
to rule out a vacuous pass, unannotated-local type inference, an RC
element constructed and dropped without crashing, two distinct tuple
shapes constructed and read independently in the same function). Full
suite green throughout, no regressions.

Not yet done (real candidate follow-ups, deliberately not bundled into
this pass): int.divmod() -> tuple[int,int] migration (would also fix the
latent DivMod CStruct RC-field-teardown gap described above); str.
partition() itself (the concurrent session's own work, unblocked by this
landing); everything under "Deferred" above remains deferred.

UPDATE (int.divmod() migration): done - lib/builtins/__int.py's divmod()
now returns Result[tuple[int,int], IntError] directly, DivMod (the @cstruct
workaround) is gone, __floordiv__/__mod__ read result[0]/result[1].
int_test.py/lib/builtins/__init__.py updated to match.

This migration surfaced a real, previously-latent gap in tuple's OWN
implementation (this pass's own "landed" claim above was true only for the
shapes actually exercised by TupleTests - a bare local declared straight
from a tuple literal in the same statement. divmod's own shape - T bound to
tuple[int,int] through GENERIC type inference, then read back through a
chain of Result[T,E].unwrap() - is structurally different, and broke
immediately): a bare, unresolved TupleType can reach emitter_c.py in at
least three ways this pass never exercised, each fixed as found, not
anticipated:

1. monomorphize.py's substitute_type_params (the TypeVar branch) returned
   whatever `arg` a type param was bound to completely unprocessed - fine
   for an already-concrete type, but a bare TupleType bound to T (e.g. from
   Result.Ok((q,r))'s own generic-call inference) needs the exact same
   eager-resolve treatment the Specialization branch two lines down already
   gives itself, and for the identical reason ("the single highest-leverage
   fix point... every future reader would otherwise need its OWN
   ensure_resolved call to unwrap"). Fixed by resolving through tuple_
   storage.get() right there - Monomorphizer now takes a TupleStorage too
   (threaded from type_resolver.py, constructed before Monomorphizer since
   it's now a dependency, same relationship union_storage already has).
2. A side effect of chasing (1): TWO different Specialization objects can
   exist for what looks like "the same" Result[tuple[int,int],IntError] -
   one from divmod's own declared return-type annotation (bare TupleType,
   discovery.py never resolves an annotation on parse), one from generic-
   call inference at a construction site (resolved, since _expr_Tuple
   defensively resolves expected_type). _get_or_create_specialization
   interns by qualname STRING, and TupleType.qualname == backing.qualname
   by construction, so lowering.py's _unify_type_param's own conflict
   check (_same_type) needed a TupleType<->backing duality branch too,
   mirroring the EXISTING Specialization<->monomorphized-form duality it
   already had for the identical reason (two representations of the same
   type reaching from two different argument positions).
3. Even after (1)/(2), a PLAIN LOCAL/PARAMETER/GLOBAL declared straight
   from a tuple[...] annotation (`dm1: tuple[int,int] = ...`) is NEVER
   independently re-resolved anywhere in the general case - only ever
   fixed up when the RHS happens to be a tuple LITERAL in the SAME
   statement (_expr_Tuple's own defensive resolve of expected_type, added
   during THIS pass's original landing - divmod's dm1 is assigned from a
   method CALL, not a literal, so that fix never applied). Rather than
   chase every remaining place resolution could theoretically be lost (a
   genuinely open-ended search - found by a real hang during THIS one
   migration, not by inspection), fixed architecturally instead: gave
   TupleType the same "directly emittable without prior resolution"
   treatment Specialization already has everywhere in emitter_c.py -
   c_type()/_value_spelling()/_member_access_operator() each got a
   TupleType branch alongside their existing RCClass one (always a
   pointer, always `->`, mangled via its own qualname - guaranteed to
   match whatever concrete backing eventually gets emitted, since
   TupleType.qualname == backing.qualname by construction). Also found:
   cfg.py's is_rc() (ownership-tracking - decides whether a binding needs
   scope-exit decref at all) had the identical gap - unlike an ordinary
   generic, where the ABSTRACT TEMPLATE class alone already answers "is
   this RC" without monomorphizing, TupleType has no template to check;
   fixed by returning True unconditionally for any TupleType (a tuple's
   backing is ALWAYS RCClass by construction, a structural guarantee, not
   something that needs .backing to already be populated).

None of this was caught by this pass's own TupleTests, despite one of them
being an "RC element constructed and dropped without crashing" test -
every original test happened to construct straight into an annotated local
in the same statement, the one shape _expr_Tuple's own defensive resolve
already covered. is_rc() returning False for an unresolved TupleType is a
silent LEAK, not a crash, so a real-compile-and-run test that only checks
the exit code (this whole file's own convention, no ASan integration - see
this plan's own Verification section) would never have caught it either.
Added directly: emitter_c_test.py's TupleTests gained test_tuple_bound_
through_generic_inference_not_a_bare_literal (a self-contained generic
function returning Result[tuple[i32,i32],E], independent of int itself) -
guards the fix, not just the workaround.

Full suite: 771 -> 772 passing throughout this migration (four
monomorphize_test.py call sites needed updating for Monomorphizer's new
tuple_storage constructor argument, not new tests, just fixed call sites;
int_test.py's own divmod coverage already existed and needed only its
DivMod->tuple[int,int] read-back updated in place; +1 new test, the
generic-inference regression case above). No other regressions.
