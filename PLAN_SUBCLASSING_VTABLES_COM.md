Goal

Implement subclassing (single inheritance) and virtual method dispatch
(vtables) for CStruct, laid out to be COM-compatible: a metalpy CStruct
value can be handed to a real COM client as a valid interface pointer,
and metalpy code can call through a foreign COM interface pointer using
the same mechanism.

RCClass is explicitly OUT of scope for this round - see "Why RCClass is
deferred, not abandoned" below.

Decisions made so far (see the conversation this plan came out of)

- Native virtual dispatch, laid out COM-compliant (IUnknown-shaped vtable
  prefix, HRESULT, etc.) - not a narrow "call into existing COM objects
  only" shim. Consuming foreign COM objects and exposing metalpy-
  implemented ones both fall out of the same mechanism once the layout
  is right.
- Single inheritance, one vtable per interface. QueryInterface (once it's
  real, see Phase 3) walks one linear chain up to IUnknown - not full
  multi-interface COM (an object answering to several unrelated
  interfaces via QueryInterface, C++-multiple-inheritance-shaped). Real,
  deferred scope, not a simplification that blocks adding it later - the
  layout here doesn't preclude it.
- Scoped to CStruct only, not RCClass - see below. This was a
  correction made mid-design, after the original RCClass-inclusive draft
  surfaced a real problem: making metalpy's own AddRef/Release (i.e.
  Incref/Decref, called automatically at nearly every scope exit
  throughout the whole language) go through a vtable indirection instead
  of a direct field/passed-in destructor pointer would impose a real,
  global performance cost on ordinary metalpy code that has nothing to
  do with COM, with no data yet on how much it'd actually cost.
- A method only enters the vtable if it's explicitly marked @virtual -
  a class having a vtable at all (@interface, or subclassing one) does
  NOT make every one of its methods virtual by default. @virtual is
  deliberately a shared, class-kind-agnostic concept (not CStruct-
  specific), meant to apply the same way to RCClass methods once that
  work resumes - one consistent decorator/rule across both, not two
  independently-invented ones.

REVISION (post Phase 1+2): @interface CStructs are never a plain value
type

After Phase 1+2 shipped (self by value, trampolines narrowing a pointer
down to a value, construction as a stack compound literal - all
documented below and now superseded), a real problem surfaced: passing
a derived CStruct VALUE where a base CStruct VALUE was expected
type-checked but emitted invalid C (different struct types aren't
implicitly convertible by value, even sharing a layout prefix). The fix
isn't a narrower type-check - it's that COM identity fundamentally
doesn't work by value at all. An @interface CStruct was redesigned to
never be a plain value type anywhere: not a local, not a parameter, not
a return type, not self. Concretely:

- self is Ptr[T], for EVERY method on an @interface CStruct - virtual or
  not. Consistency was chosen deliberately over a narrower rule (self by
  value for ordinary methods, Ptr[T] only for @virtual ones): one
  calling convention, not two, and it eliminates the trampoline/value-
  narrowing-cast machinery Phase 2 originally needed (a plain pointer-
  to-pointer cast replaces both - see "Dispatch mechanism" below).
- Construction (`FooImpl(x=1, y=2)`) heap-allocates via sys.alloc[T] -
  the same real allocation path RCClass's own `ClassName(...)` already
  uses - and produces a Ptr[T], never a bare T. Unlike RCClass, this is
  an EXPLICIT Ptr[T] in the metalpy type system, not an invisible-
  pointer convention (c_type(cls) itself is untouched - a bare CStruct
  is still a plain value everywhere c_type sees it bare; the pointer-ness
  is added explicitly only where self/construction need it - see
  emitter_c.py's _self_c_type).
- The Ptr[T]/ConstPtr[T] dot-operator: `.attr`/`.method()` on ANY
  Ptr[T]/ConstPtr[T] (not just @interface CStruct - this is a general
  language feature) redirects name lookup to the pointee and emits a
  real `->`, matching what `[0].attr` already did for lookup but,
  unlike `[0].attr`, without copying the pointee into a value first -
  so `self.x`/`p.x = 5` read like ordinary Python attribute access
  despite self/p being genuine pointers. Implemented in lowering.py's
  _attr_lookup/_attr_lookup_callable (redirect to the pointee's type,
  leaving the operand itself untouched) and emitter_c.py's
  _member_access_operator (Ptr/ConstPtr now emits -> like RCClass
  already does). Bonus: this incidentally fixes the pre-existing
  Ptr[T][0].field = value write-through bug for the `.field` spelling
  (real arrow-write, not deref-into-a-temp-and-discard) - see
  emitter_c_test.py's test_ptr_dot_operator_write_through_on_plain_cstruct.
  The explicit-index spelling (`p[idx].field = value`, idx != 0) still
  has the old bug, flagged separately.
- No exhaustive rejection of bare-value type annotations was added
  everywhere (parameters/locals/fields) - construction simply never
  produces a bare value anymore, which covers the intended usage.
  `p[0]` still technically produces a bare value as an escape hatch (an
  existing, general Ptr[T] capability, not special-cased away) - a
  function declaring a bare-typed parameter is syntactically legal but
  uncallable with any value produced the ordinary way.
- Lifetime management is fully manual, matching real COM (see "AddRef/
  Release are the user's own virtual methods" below) - not wired into
  metalpy's automatic Incref/Decref. A simple, NOT YET BUILT pattern for
  users who want deterministic cleanup anyway: wrap a Ptr[T] COM object
  inside a small RCClass whose __del__ calls Release() - gets automatic
  decref-driven cleanup for free from the existing RC machinery, no
  compiler changes needed. Worth a small stdlib helper (something like
  ComPtr[T]) once there's real usage to shape it against - not part of
  this plan yet.

Why RCClass is deferred, not abandoned

The original draft of this plan included RCClass, on the reasoning that
release_object's `destructor` parameter is resolved from a value's
STATIC declared type today, not looked up dynamically - so decref-ing a
value through a base-typed reference whose actual object is a subclass
with its own additional owned fields would call the wrong (base)
destructor and leak them. Fixing that needs SOME dynamic dispatch
reachable from the object itself, which is what a vtable is for - and
once AddRef/Release are virtual for that reason, they're already COM-
shaped. That reasoning is still correct: RCClass subclassing genuinely
can't ship soundly without dynamic dispatch for destruction. But "correct
fix eventually" isn't the same as "pay the cost everywhere today, sight
unseen" - Incref/Decref sit on a hot path (nearly every RC-typed
variable's scope exit, every parameter pass with a move/borrow, every
field write), and virtualizing them is a global, code-shape change, not
a narrow one. That deserves real profiling before committing to it, not
a decision made in a design doc. RCClass subclassing (and the destructor-
soundness fix it needs) is real future work, sequenced AFTER this CStruct
work ships and after there's actual performance data on what virtual
Incref/Decref costs in practice - not scrapped.

Architectural framing: CStruct as interface, not as "a smaller RCClass"

A COM interface pointer, in C, is `{ const XVtbl* lpVtbl; }` - a vtable
pointer and nothing else. The real, backing object (with actual data)
is a separate struct that EMBEDS that shape as its first member, plus
private fields after it. This maps onto CStruct directly, in two flavors
that need no new keyword to distinguish - "pure interface" is just the
degenerate case of an implementation with zero fields of its own:

- A pure interface declaration (e.g. a hand-declared IUnknown, or a
  foreign interface like IShellFolder): a CStruct with no fields of its
  own beyond the synthesized vtable pointer, only virtual method
  signatures. No implementation, just a calling shape.
- An implementation: a CStruct subclassing an interface CStruct, adding
  its own private fields and real method bodies. RCClass's existing
  field-flattening (base-first, most-derived-last, already implemented
  in emitter_c.py's emit_rcclass) generalizes directly to this - the
  same walk, just reused for CStruct's own base chain instead of being
  RCClass-only.

Critically: an implementation CStruct's own AddRef/Release, if it has
them, are ordinary user-written virtual methods, like any other method
on the interface. They are NOT tied into metalpy's own automatic Incref/
Decref/scope-exit RC machinery at all - a metalpy program using a CStruct-
based COM object is responsible for calling AddRef/Release itself,
exactly like any other COM client in any other language. This is a
deliberate non-goal, not an oversight: teaching the compiler's own
automatic RC to understand CStruct lifetime is real, separate work that
this plan does not attempt, and matches genuine COM semantics anyway
(COM was never garbage-collected/automatically-refcounted by design -
every real COM client calls AddRef/Release by hand).

Given that, lifetime/allocation for a CStruct-based COM object is the
program's own responsibility too (no automatic Incref/Decref). Per the
REVISION above, an @interface CStruct is now heap-allocated by
construction itself (never a plain stack value at all, unlike a plain
@cstruct) - so the "needs a stable address to hand out as a pointer"
concern that originally motivated a separate Ptr[CStruct] investigation
is now just how construction always works, not a follow-on step.

Design

1. Layout

    typedef struct IFooVtbl IFooVtbl;
    struct IFoo {
        const IFooVtbl* $vtable;   // the ONLY member of a pure interface
    };

    struct FooImpl {              // subclasses IFoo
        const IFooVtbl* $vtable;   // inherited - same first-member position,
                                    // exactly like RCClass's base-fields-
                                    // first convention today
        i32 x;                     // FooImpl's own fields
        i32 y;
    };

$vtable is the literal first member (COM's one hard ABI requirement) -
this is CStruct's equivalent of RCClass's existing $header convention,
just without the refcount field RCClass's ObjectHeader also carries
(CStruct has nothing to refcount automatically - see above).

2. Heap allocation via Ptr[CStruct]

A stack-local CStruct value is no good for handing out a COM interface
pointer - it has to outlive the function that constructed it, often
crossing into another process entirely (out-of-process COM, or just
handing a pointer to a foreign caller). This needs Ptr[T] to work for a
CStruct T, paired with some way to heap-allocate one and get a Ptr[T]
back.

CONFIRMED (Phase 1, not just read - actually compiled and run):
sys.alloc[SomeInterfaceCStruct](1) needs no new code at all. The C-level
type mapping was already general enough (c_type's own Specialization
branch for Ptr[T]/ConstPtr[T] calls _value_spelling(inner_type) on
whatever T is, no RCClass-specific restriction), and sys.alloc[T] itself
(lib/sys.py) was never the gap either - it's a plain generic function,
sizeof(T) * count bytes from the raw _alloc allocator, no RCClass
special-casing in its own body. There's real prior art too: lib/builtins/
__list.py's own RawList already heap-allocates Ptr[_ListMetadata] (a
plain @cstruct) via sys.alloc[_ListMetadata] and indexes/reads/writes
through it - a working, in-production example of exactly this pattern
that predates this plan entirely. Whole-struct heap alloc + index +
whole-value read/write (`p[0] = FooImpl(x=5)`, `p[0].do_thing(...)`) is
verified working end-to-end (real C compile + run, see
InterfaceCStructLayoutTests.test_ptr_interface_cstruct_heap_roundtrip in
emitter_c_test.py). One caveat found along the way, NOT part of this
gap: `p[0].field = value` (a FIELD-level write THROUGH a dereferenced
pointer index, as opposed to replacing the whole value) silently doesn't
write back to the pointee - lowering copies `*p` into a local temp,
mutates the temp, and discards it. Confirmed pre-existing and unrelated
to CStruct/@interface specifically (reproduces identically for a plain,
non-@interface @cstruct too) - a general Ptr[T] indexed-lvalue lowering
gap, flagged separately, not this plan's problem to fix.

(For context on why this looked riskier before checking: RCClass's OWN
construction path - _schedule_rcclass_construction, ir.Allocate,
ObjectHeader initialization - IS RCClass-shaped, but that machinery only
runs for RCClass's `ClassName(...)` heap-construction syntax. CStruct's
`ClassName(...)` is plain stack/value construction (a C compound
literal, confirmed in emitter_c_test.py) and never touches ir.Allocate
at all - the two constructor syntaxes were never on the same code path
to begin with, which is why sys.alloc[T] alone was already enough.)

Status: DONE, no code changes needed - Phase 1 for this section is
complete.

3. Vtable struct + static instance, per concrete interface/implementation

    typedef struct IFooVtbl {
        // IUnknown's 3 - COM ABI requirement, MUST be exactly these 3,
        // in exactly this order, first, on every interface
        HRESULT (*QueryInterface)( IFoo* self, const IID* riid, void** out );
        uint32_t (*AddRef)( IFoo* self );
        uint32_t (*Release)( IFoo* self );
        // then IFoo's own virtual methods, in declaration order
        ReturnType (*SomeMethod)( IFoo* self, ArgTypes... );
    } IFooVtbl;

    static const IFooVtbl __main__$FooImpl$$vtable = {
        .QueryInterface = __main__$FooImpl$$QueryInterface,
        .AddRef = __main__$FooImpl$$AddRef,
        .Release = __main__$FooImpl$$Release,
        .SomeMethod = __main__$FooImpl$SomeMethod,
    };

  No __stdcall anywhere - 32-bit Windows isn't a real target, and on x64
  there's exactly one calling convention (stdcall and cdecl are the same
  ABI there), so the platform default is already COM-correct. If 32-bit
  ever becomes a real target, this is a contained, mechanical change
  (add __stdcall to these typedefs and the corresponding Function.node
  emission), not a design change - not worth the complexity now.

Slot assignment: the interface that first declares a virtual method owns
its slot index for the whole hierarchy below it; a subclass's own vtable
places an override at the SAME slot its base declared, and appends any
new virtual methods it introduces itself after everything inherited - no
slot ever gets renumbered, so a caller holding a base-typed pointer can
always call through the base's own known slot index regardless of the
actual concrete implementation.

4. Dispatch mechanism

A new ir.Instruction (VirtualCall, or an extra field on ir.Call - naming
TBD during implementation, affects emitter_c.py enough to decide early)
emits as `(( const IFooVtbl* )obj->$vtable)->SomeMethod( obj, args... )`
instead of ir.Call's current direct `mangled_name( args... )`. Only
CStructs that actually have a vtable (see opt-in below) ever go through
this - a plain, non-interface CStruct is entirely unaffected, same
direct-call behavior as today.

5. Opt-in, not default - both at the class level AND per method

Every existing CStruct in the language stays exactly as it is today -
plain value construction, no vtable, no layout change. A CStruct gets a
vtable AT ALL when it either (a) declares a base CStruct that's itself
an interface, or (b) is explicitly declared as a root interface (a new
decorator - `@interface` or similar).

That alone doesn't make every method on it virtual, though: a method
only goes into the vtable if it's explicitly marked `@virtual`. A method
without `@virtual` on an @interface CStruct is an ordinary, statically-
dispatched method (ir.Call, direct C symbol, no vtable indirection) even
though its OWNER has a vtable - the same "be explicit, don't infer it"
posture as @interface itself not inheriting implicitly. This gives a
class both overridable (@virtual) and non-overridable (plain) methods
side by side, and means adding a vtable to a class doesn't silently make
every existing method slower/indirect - only the ones actually meant to
be polymorphic.

@virtual is deliberately NOT scoped to CStruct specifically - it's the
same decorator, same validation rules, meant to apply to RCClass methods
too once that work resumes (see "Why RCClass is deferred" above) - one
consistent "this method is in the vtable" concept across both class
kinds, not two independently-invented ones. Concretely: @virtual's own
implementation (recognizing the decorator, recording which methods are
virtual on a Function, assigning/inheriting vtable slots) should live
somewhere shared (discovery.py's own decorator handling, mpy_types.py's
Function gaining an is_virtual flag) rather than folded into CStruct-
specific code paths, even though only CStruct actually builds a vtable
from it right now.

6. Subclassing mechanics

- CStruct gains a .base field, mirroring RCClass's own.
- _attr_lookup/_find_method need to walk a CStruct's own base chain, the
  same single-inheritance linear walk RCClass subclassing would have
  needed (self, then base, then base.base, ... until None) - today
  neither RCClass nor CStruct's own lookup does this at all. DONE
  (Phase 1) - CStruct.chain_lookup in mpy_types.py, wired into
  _attr_lookup/_attr_lookup_callable/_find_method.
- SUPERSEDED by the REVISION above (kept for history): calling an
  inherited, non-@virtual method on a DERIVED instance originally failed
  to compile as real C, because CStruct methods took self BY VALUE, and
  a value-narrowing cast (`*(const AncestorType*)&derived_value`) was
  the first fix. Once self became Ptr[T] uniformly for every @interface
  method (see the REVISION), the fix simplified to a plain pointer-to-
  pointer cast - `(AncestorType*)derived_ptr` - no address-of, no
  dereference, no value copy. Still in emitter_c.py's
  _emit_self_operand. Verified end-to-end in emitter_c_test.py's
  test_inherited_method_callable_through_subclass_instance.
- No construction-chaining story is needed here the way RCClass
  subclassing would need one (base __init__ before derived fields, etc.)
  - an implementation CStruct's own fields (vtable pointer plus whatever
  it adds) are just ordinary field-value construction into the heap-
  allocated instance, unaffected by inheritance depth. This is one of
  the concrete ways scoping to CStruct is smaller, not just "RCClass's
  plan with fewer steps."
- A CStruct subclassing an @interface CStruct MUST itself be declared
  @interface too - @interface-ness is NOT inherited implicitly. A
  subclass that omits it is a compile error, not a silent "data-only
  CStruct that happens to also carry an inherited $vtable pointer".
  Deliberately conservative (explicit choice over implicit inheritance,
  same reasoning as the single-vtable-not-multi-interface decision
  above) - relaxing this later (e.g. letting @interface-ness inherit
  automatically) is easy; tightening it after code already depends on
  the looser behavior is not.
- Whether a PLAIN (non-@interface) CStruct can subclass another plain
  CStruct at all - ordinary field inheritance, no vtable involved, fully
  unrelated to COM - stays out of scope for this plan (see below);
  CStruct.base as introduced here only has defined behavior for the
  @interface case.
- Override validation: strict signature matching to start (same
  parameter/return types, no covariance/contravariance) - matches how
  overload matching elsewhere in this codebase already stays
  conservative rather than guessing at variance rules. A subclass
  overriding an inherited @virtual method MUST repeat @virtual on its
  own re-declaration - matching an existing virtual slot's name+
  signature alone is not enough. Decided: same "explicit at every
  declaration site" reasoning @interface's own non-inheritance already
  follows: no method is virtual, anywhere, without @virtual written on
  that exact declaration.
- Unimplemented @virtual methods (an interface method with no real
  body) need no separate `@abstract` decorator - a `@virtual` method
  whose body is a stub (`...`, the same shape `@overload` stubs already
  use and `_is_stub_body` already recognizes) IS the "must be overridden"
  declaration. This falls out of CStruct's narrower role here: an
  @interface CStruct's own methods exist only to be filled in by a
  concrete implementation subclassing it - nothing ever constructs a
  bare interface directly - so every stub-bodied @virtual method is
  already, by construction, unimplemented-until-overridden. No new AST
  convention, no instantiability tracking, just reusing the existing
  stub-body check in the one place it's actually ambiguous (building a
  vtable/static instance for a class that still has a stub-bodied slot
  is the compile error). A general-purpose `@abstract` decorator (for
  RCClass hierarchies where abstract methods can appear at any depth
  mixed with fields and constructors, not just at an interface root) is
  a separate, deferred question - see "Why RCClass is deferred" above;
  not part of this CStruct-scoped plan.
- Multiple inheritance stays a compile error, unchanged.

7. COM specifics

- A built-in IUnknown CStruct (or a documented pattern for declaring an
  equivalent one) with QueryInterface/AddRef/Release, each marked
  @virtual (they're COM ABI requirements - always in slots 0/1/2, never
  optional the way a class's own additional methods are) - every other
  interface's own base, directly or transitively.
- QueryInterface/AddRef/Release are hand-written by the user, like any
  other @virtual method - no compiler-synthesized IUnknown boilerplate in
  this plan. Once this ships and there's real usage to learn from, an
  opt-in automatic-IUnknown convenience (synthesizing all three, the same
  spirit as _synthesize_rcclass_destructor already does for RCClass
  destructors) is worth reconsidering - genuinely open to it being the
  better default, just not committing to it sight unseen, same posture
  as the RCClass-deferral reasoning above. See Phase 4.
- GUID/IID type: lib/guid.py, a `GUID` class (@cstruct, {u32,u16,u16,
  u8x8} fixed layout) constructible from a standard hyphenated hex string
  - `GUID('deadbeef-dead-beef-dead-beefdeadbeef')` - matching the usual
  8-4-4-4-12 IID/CLSID text form. Needs confirming this language has
  fixed-size array support for Data4 (or model it as 8 separate u8
  fields if not), AND confirming metalpy's own str type currently
  supports whatever parsing GUID.__init__ needs (splitting on '-',
  parsing hex substrings to integers) - str.split() is specifically
  listed as a missing method in TODO.txt today, so this may be a real,
  small dependency to land first, not just "write the constructor".
- A real QueryInterface needs a way to associate an interface/
  implementation with the ID(s) it answers to - a class-level constant or
  decorator argument, e.g. `@interface(iid = GUID('...'))`.
- HRESULT: a library type (lib/windows/com.py?) with the usual SEVERITY/
  FACILITY/CODE bit layout and named constants (S_OK, E_NOINTERFACE,
  E_FAIL, ...) - a library addition, not a compiler change.
- Calling convention: none needed - see the layout section above (32-bit
  Windows isn't a real target; platform default is already COM-correct
  on x64).

Phased implementation plan

1. DONE. Type model + layout. CStruct.base, the @interface decorator
   (@interface doesn't inherit implicitly, enforced), $vtable as the
   synthesized first member, base-chain field flattening, _attr_lookup/
   _find_method walking a CStruct's own base chain (CStruct.chain_lookup),
   and Ptr[CStruct] heap allocation (confirmed zero new code needed -
   sys.alloc[T] already worked). Committed.
2. DONE. Dispatch. Landed without a new IR instruction - emission reads
   Function.is_virtual directly off the existing ir.Call.target and
   changes the emitted C shape (see the ir.Call branch's is_virtual check
   in emitter_c.py), no lowering.py/ir.py changes needed for the call
   site itself. Delivered: full Vtbl struct body + slot ordering
   (CStruct.virtual_slots), a static const vtable instance per fully-
   fulfilled class (emit_interface_vtable_instance - see
   CStruct/_interface_fulfilled_slot_impls) wiring each slot DIRECTLY to
   its real implementing function via a function-pointer cast (no
   trampoline - see the REVISION above, self being uniformly Ptr[T] made
   the trampoline layer unnecessary), $vtable wired at construction time,
   slot-introduction/override-signature validation (compiler.py's
   _validate_interface_vtable - only the root may introduce a new slot,
   overrides must match strictly), construction-time fulfillment
   validation (lowering.py - constructing an interface with any
   unfulfilled/stub slot is a compile error). Tested end-to-end: real
   vtable dispatch (not a direct call) reaching an override, confirmed
   via both the generated C shape and actual execution - see
   emitter_c_test.py's
   test_virtual_dispatch_calls_the_override_not_the_stub.

   One bug found and fixed along the way (in this phase's own new code,
   not pre-existing): the Vtbl struct's own `self` parameter, the first
   mention of an @interface CStruct as a pointer, hit the same "function
   prototype scope" trap RCClass's forward tags exist to avoid - every
   @interface CStruct now gets the same `struct X;` forward tag RCClass
   already got.

   A second thing found here turned out to be a symptom of a bigger
   design gap, not a narrow bug: passing a derived CStruct value where an
   ORDINARY (non-self) parameter expected a base CStruct type type-
   checked but produced invalid C. Chasing this led to the REVISION above
   (self by value, and CStruct-as-value generally, was the wrong model
   for a COM-identity type) rather than a local patch. A general CStruct
   subtype-coercion gap can still resurface for the same reason it did
   here IF a bare (non-Ptr[T]) @interface CStruct type annotation is used
   anywhere - see the REVISION's own note on why that's not exhaustively
   guarded against, and the flagged follow-up task for the remaining
   `p[0]`-escape-hatch case.
3. PAUSED - blocked on a prerequisite. COM specifics: lib/guid.py's GUID
   type (plus whatever str support it needs), HRESULT library type, a
   worked example consuming a real foreign Windows COM interface end to
   end (a simple, easily-testable one - TBD which), and a worked example
   exposing a metalpy-implemented interface to a hand-written C caller -
   both with hand-written QueryInterface/AddRef/Release, matching the
   "hand-rolling first" decision above.

   Blocker found while starting this phase: GUID's constructor needs
   str.split('-') to parse a hyphenated hex string, but list[T] (str.
   split()'s natural return type) turned out to be completely broken -
   `list[i32]()` fails with "cannot call list[i32]", a `.cast[T]()`
   failure, and a missing arithmetic-mode wrapper in list.__del__, for
   ANY type parameter, not just RC types. Confirmed list[T] has never
   actually been compiled anywhere in the test suite before (only
   Python-level `list[str]` type hints in test harness code, unrelated).
   Also found and fixed along the way (kept, unrelated to the list[T]
   bug itself): compiler.py's _trigger_name called `getattr(unit,
   'qualname', str(unit))`, which Python evaluates eagerly regardless of
   whether qualname exists - harmless normally, but a genuine infinite
   hang the moment the reachable object graph has a real cycle (which
   list[T] apparently triggers) since dataclass __repr__ has no cycle
   detection. That fix is committed on its own.

   Decided: fix list[T] properly as its own, separate piece of work
   (not a narrower split() workaround) - and while there, implement
   str.find()/str.index() as real general-purpose methods (both already
   listed missing in TODO.txt) rather than embedding one-off byte-
   scanning logic inside split() itself. Flagged as a background task;
   this phase resumes once that lands.
4. Stretch/optional, not committed: an opt-in automatic-IUnknown
   convenience (compiler-synthesized QueryInterface/AddRef/Release,
   `_synthesize_rcclass_destructor`-style) - revisit once Phase 1-3 are
   proven and there's real usage to judge whether hand-rolling is
   actually a burden worth automating, and if so, what a correct default
   QueryInterface (single-interface, walks to IUnknown) should look like.

RCClass subclassing/vtables (deferred, future plan, not written yet):
once this ships and there's real profiling data on what a virtual-call
Incref/Decref costs on representative metalpy programs, revisit the
original RCClass-inclusive design (destructor-soundness fix via vtable-
dispatched AddRef/Release, base-chain field flattening already mostly
in place, full subclassing mechanics including constructor chaining
across RCCLASS ATTRIBUTE LIFETIME's existing protocol). Also worth
designing then, not now: a general-purpose `@abstract` decorator.
CStruct doesn't need one - see "Unimplemented @virtual methods" above,
stub bodies already cover it - but RCClass's richer model (arbitrary-
depth hierarchies, abstract methods possibly mixed with real fields and
constructors, not just declared at an interface root) may warrant an
explicit marker rather than reusing the stub-body convention.

Explicitly out of scope

- RCClass subclassing/vtables - deferred per above, not this plan's job.
- 32-bit Windows / __stdcall - not a real target; a contained, mechanical
  change to revisit if that ever changes, not a design concern now.
- Full COM activation/registration (IClassFactory, DllGetClassObject,
  registry entries, type libraries/IDL generation) - "a COM-ABI-shaped
  value metalpy can construct and hand a pointer to" is the target, not
  "a DLL a `regsvr32`-style tool can register."
- Multiple interface implementation / full QueryInterface semantics (an
  object answering to several unrelated interfaces) - deferred per the
  interface-model decision above.
- Covariant/contravariant override signatures - strict exact-match only.
- Automatic (compiler-managed) refcounting for CStruct-based COM objects
  - AddRef/Release are the user's own virtual methods; metalpy's
  automatic Incref/Decref/scope-exit RC machinery is not extended to
  CStruct at all in this plan.
- Compiler-synthesized QueryInterface/AddRef/Release - hand-rolled only
  for now; automatic synthesis is Phase 4, not committed to.
- Plain (non-@interface) CStruct subclassing plain CStruct for ordinary
  field inheritance, unrelated to vtables/COM - CStruct.base as
  introduced here only has defined behavior for the @interface case.

Implementation details (none blocking - resolve inline during the
relevant phase rather than pre-designing here)

- Exact shape of the new virtual-dispatch IR instruction: not designed
  up front - Phase 2 figures this out from what the dispatch codegen
  actually needs, same way other IR instructions in this codebase were
  added.
- Ptr[CStruct] heap allocation: confirmed not a heavy lift - build it in
  Phase 1 (see "Heap allocation via Ptr[CStruct]" above for what's
  already generic vs what's RCClass-shaped today).
- GUID.__init__ needs to parse a hyphenated hex string, which needs
  str.split() - confirmed missing (per TODO.txt) and confirmed simple
  to add. Lands as its own small prerequisite piece of work ahead of
  Phase 3 (lib/guid.py depends on it), not a design risk to the plan.
