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
program's own responsibility too: CStructs have no heap allocation today
(emitter_c.py: "CStruct/CUnion - plain value construction, no header/no
heap allocation at all"). An implementation that needs a stable address
to hand out as a pointer (the common case - an interface pointer handed
to a foreign caller, or to another process, has to outlive the call that
produced it) needs Ptr[CStruct] to work - see "Heap allocation via
Ptr[CStruct]" below.

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

The C-level type mapping already looks general enough to not need
changes: c_type's own Specialization branch for Ptr[T]/ConstPtr[T] calls
_value_spelling(inner_type) on whatever T is, with no RCClass-specific
restriction visible in that code path - worth confirming directly rather
than assuming, but it doesn't look like new emitter work. sys.alloc[T]
itself (lib/sys.py) isn't the gap either - it's a plain generic function,
sizeof(T) * count bytes from the raw _alloc allocator, no RCClass
special-casing in its own body. The real gap is one level up: how a
constructed RCClass gets from that raw allocation to a live object is
orchestrated by lowering.py (_schedule_rcclass_construction schedules
sys.alloc[concrete_type] as a real compile unit, then _lower_allocate_
fields/_try_lower_construct_call emit ir.Allocate) and finished by
emitter_c.py's ir.Allocate handling, which assumes ObjectHeader
initialization follows the raw allocation. There's no existing path to
"heap-allocate a bare value type and hand back a pointer to it, no
header, no refcount field, just the raw bytes" - that's the piece to
add. Needs its own primitive/code path, sized during Phase 1 once the
exact gap is confirmed by trying it directly.

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
  neither RCClass nor CStruct's own lookup does this at all.
- No construction-chaining story is needed here the way RCClass
  subclassing would need one (base __init__ before derived fields, etc.)
  - CStruct construction is already "plain value construction", and an
  implementation CStruct's own fields (vtable pointer plus whatever it
  adds) are just ordinary field-value construction, unaffected by
  inheritance depth. This is one of the concrete ways scoping to CStruct
  is smaller, not just "RCClass's plan with fewer steps."
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

1. Type model + layout. CStruct.base, the @interface decorator (or
   equivalent, with the "@interface doesn't inherit implicitly" rule
   enforced), $vtable as the synthesized first member, base-chain field
   flattening generalized from emit_rcclass's existing walk so both
   RCClass and CStruct share it, _attr_lookup/_find_method walking a
   CStruct's own base chain, and Ptr[CStruct] heap allocation (confirm
   the C-level type mapping directly, build whatever allocation
   primitive turns out to be missing). No dispatch yet - this phase is
   pure type-model + layout, testable by inspecting emitted struct/
   vtable-typedef shapes and a round-tripped Ptr[SomeCStruct] allocation
   directly, same spirit as the existing emitter_c_test.py style.
2. Dispatch. The new virtual-call IR instruction + emission, per-concrete-
   interface vtable struct + static instance generation, slot assignment
   and override resolution across a hierarchy. A hand-written IUnknown-
   shaped interface with one extra virtual method, called both through a
   base-typed and a derived-typed reference, is the natural end-to-end
   test here - QueryInterface/AddRef/Release bodies are hand-written in
   this test too, same as real usage will be (see COM specifics above).
3. COM specifics. lib/guid.py's GUID type (plus whatever str support it
   needs), HRESULT library type, a worked example consuming a real
   foreign Windows COM interface end to end (a simple, easily-testable
   one - TBD which), and a worked example exposing a metalpy-implemented
   interface to a hand-written C caller - both with hand-written
   QueryInterface/AddRef/Release, matching the "hand-rolling first"
   decision above.
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
