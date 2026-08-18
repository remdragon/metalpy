This is a complete redesign of the metalpy compiler from the ground up.

the compiler needs to operate in distinct stages with output that can be unit-tested at each stage. There needs to be clear areas of responsibility for each stage

stage 1 - discovery:
	parse source modules and imports and generate type information
	resolve all type references to their definitions
	
	anonymous unions will be detected here and created here, but only so that
	type information is correctly constructed, scheduling anonymous unions has
	to wait for stage 2.
	
	important note: type references are objects, not strings
		we need to store metadata about types
			is it ref-counted?
			etc
	
	Here's is a simple proposal of the Type objects needed:
	
	class Type: # base class
		source: tuple[str,int]|None = None # e.x. ( 'builtins/_int.py', 42 )
		name: str|None = None # e.x. 'builtins.int' # NOTE: this is also it's key into a global type registry if necessary
	
	class ScalarType( Type ):
		'''
		maybe could usage a better name, but this could cover any of the following:
			i32, u32, isize, usize
			Ptr[...]
			ConstPtr[...]
		'''
	
	class UnresolvedType( Type ):
		'''
		this is a placeholder for a name we've come across that hasn't been defined yet
		'''
		resolve_to: Type|None = None
	
	class CUnion( Type ):
		'''
		devolves to union{ ... } in C, multiple interpretations of the same memory
		
		@cunion
		class Foo:
			foo: int
			bar: str
		'''
	
	class TaggedUnion( Type ):
		'''
		implements a tagged union like int|str. see also builtins.Result
		
		@union
		class IntStr:
			v_int: int
			v_str: str
		'''
	
	class CStruct( Type ):
		'''
		implements a struct in C
		
		@cstruct
		class POINT:
			x: i32
			y: i32
		'''
	
	class CEnum( Type ):
		'''
		implements a simple enum (not load bearing like @union)
		
		@enum
		class Foo:
			Bar = 1
			Baz = 2
		'''
	
	class RCClass( Type ):
		''' a ref-counted class Foo (without @enum, @cunion, @union, or @cstruct) '''
		...
	
	class Function( Type ):
		'''
		holds information about a function, including its parameters and the AST for its body
		'''
	
	class Variable( Type ):
		'''
		holds information about a Variable. This can include global variables, function parameters, locals, or temporaries (i.e. registers)
		'''
	
	testable output:
		type information with metadata
		
		compile:
			def foo( x: Foo ) -> None:
				pass
		
		test:
			self.assertEqual(
				compiler.get_type( '__main__.Foo' ),
				UnresolvedType(),
			)
		
		compile:
			class Foo:
				pass
		
		test:
			mod = compiler.discover_source( code )
			Foo = mod.get_local( 'Foo' )
			self.assertTrue( isinstance( Foo, Class )) # not a Struct, Function, UnresolvedType, etc
			self.assertEqual( Foo.name, '__main__.Foo' )
			self.assertEqual( Foo.base, None ) # no base class
			self.assertEqual( Foo.attributes, [] ) # no attributes
			self.assertEqual( Foo.methods, [] ) # no methods

stage 2 - type resolution
	* walk the tree from main and determine everything touched by main,
		directly or indirectly, and finishing type resolution for
		anything relevant to the final product
	* convert anonymous unions into TaggedUnion operations
		in order to look up an anonymous union, we have to canonicalize it to avoid conflicts and duplicates.
			canonicalization is a two-pass effort:
			1) lookup each name in its namespace and translate that to the qualname
			2) sort all qualnames asciibetically
				str|int -> builtins.int|builtins.str
		schedule any anonymous unions that get touched by other scheduled artifacts
		because we will be attaching anonymous union type information
			by reference, not by name, we can cnames for the union type as they are created.
			builtins.int|builtins.str --> $builtins$int$$builtins$str
				$ prefix to distinguish it symbols coming from builtins.py
		automatically generate and queue union function bodies into work queue as they are detected
	
	future work:
		* move arithmetic mode logic here (transform ast tree)
		* convert Specializations into full plain RCClasses

stage 3 - lowering - convert function bodies into IR:
	process every item emitted from stage 2 type resolution
		decompose python AST into low-level IR
			emit INCREF/DECREF where necessary
			defer/errdefer scheduling
	
	testable output:
		IR sequences from functions
		
		compile:
			def foo( x: i32 ) -> None:
				with compiler.wrap_arithmetic:
					print( x + 1 )
		
		test output:
			mod = compiler.compile_source( ... )
			foo = mod.get_local( 'foo' ) # look up name from the module's local namespace
			x = foo.get_local( 'x' ) # look up the variable 'x' from the function's local namespace
			print = compiler.get_global( 'builtins.print' )
			i32 = compiler.get_intrinsic( 'i32' ) # I don't know if this is what we want exactly but intrinsics need to have Type objects too
			t0 = foo.get_local( 't0' ) # look up the variable 't0' that we expected to get created
			
			FUNC_START( 'foo', [ x ], None ) # verify that the FUNC_START takes the single parameter x and has no return value
			DECLARE_TEMP( t0, i32 )
			ADD_WRAP( t0, x, 1 )
			CALL( print, [ t0 ])
			DELETE_TEMP( t0 ) # t0 is no longer valid after this IR
			RETURN
			FUNC_END( 'foo' )
		
		NOTE: the point of this design is that once IR is generated, we are done parsing python source

IR:
	the IR will target a virtual machine with infinite registers/temporaries
	the instruction set will be based on the 3AC/SSA standard
	it will be the IR generator's responsibility to generate DELETE statements
		against registers/temporaries. This is so the emitter doesn't have to
		try to figure that out later.

design decision: range() stays a compiler intrinsic, not a real generator

	PLAN_GENERATORS.md's own generator-function work (real `yield`,
	state-machine transform, real user-authored iterators) does NOT extend to
	range() itself. `for i in range(n):` is recognized textually
	(lowering.py's _is_range_call) and lowered directly to a plain counting
	loop (_lower_for_range) - no heap allocation, no backing RCClass, no
	refcounting, arithmetic proven safe by construction (target < stop
	strictly before every increment) so the increment bypasses the ordinary
	checked-arithmetic policy entirely.

	This is deliberate, not a gap waiting to be closed. range() is the single
	most common loop-counting construct in any real program - every one of
	those call sites would pay a real allocation + refcount-churn cost for
	zero functional benefit if range() were reimplemented as an ordinary
	generator function (PLAN_GENERATORS.md's own machinery: a synthesized
	RCClass, sys.alloc, an ObjectHeader, atomic refcount ops on every
	__next__() call). A user-authored generator that NEEDS real yield/resume
	semantics still works fully (including `for x in range(n): yield x*2`
	inside one - PLAN_GENERATORS.md's Phase 4, a for-loop over range()
	containing yield is desugared into the equivalent while-loop shape before
	lowering) - only range() ITSELF stays intrinsic. Confirmed with the user
	(2026-08-15): do not convert range() into a real generator.

stage 4 - IR optimization (map/reduce):
	NOTE: all concept of @union is gone here, IR operates on values and pointers
	
	because low-level IR can sometimes generate suboptimal sequences, this stage
	looks for opportunities to optimize the low-level IR.
	
	(this stage could end up being a noop, but I want to define it just in case)
	
	testable output:
		simplified IR output from known detectable scenarios

stage 5 - emitter:
	this module is strictly responsible for emitting the generated IR
	to C or ASM or whatever other backend we choose with simple sugar to make it compile.
	
	testable output:
		generated C output

stage 6 - linker:
	this module is responsible for dealing with the inconsistencies between
	different compilers to generate the final executable

design decision: scalar T(x)/.to_T() are two separate operations, not one

	SYNTAX.md's own "Scalar Conversions" section covers the user-facing
	rules. This is the internal rationale, for anyone touching either
	mechanism.

	`T(x)` (construct-cast syntax, `_lower_scalar_cast` in lowering.py) is
	purely a bit-width question: same-width or widening always succeeds,
	unconditionally, in every arithmetic mode (a bare width comparison
	against `Scalar.sizeof`, then either `ir.CastWrap` directly or the
	existing mode-selected Cast opcode for a genuine narrowing). `x.to_T()`
	(`compiler.checked_convert`/`ir.ConvertCheck`, emitter_c.py's
	`_emit_convert_check`) is a genuine numeric VALUE-range check against
	the target's own [MIN,MAX], independent of width - it can fail for a
	same-width conversion T(x) never would (`i8(-1).to_u8()` fails;
	`u8(i8(-1))` never does). These were deliberately kept as two
	independent opcodes/emitters, not a shared implementation, even though
	the underlying comparison arithmetic looks similar (both reuse
	`_int_min_max_bit_pattern`'s MIN/MAX and the same wideint/u128/MSVC-
	fallback edge cases) - confirmed with the user across several rounds of
	clarification, since an earlier draft of this design conflated the two
	(T(x) as sugar over .to_T()) before a concrete counterexample
	(`u8(i8(-1))` succeeding vs `i8(-1).to_u8()` failing) settled it.

	.to_T() has exactly one variant - no wrapped/saturated flavor - since
	there's no meaningful "wrapped value-range check" the way there's a
	meaningful wrapped ADD; ambient arithmetic mode only governs how the
	resulting Result gets consumed at the call site (propagate under
	checked/wrap/saturate, panic under panic mode), mirroring how
	`int.__floordiv__`/`__mod__`'s own single ZeroDivisionError check
	already works.

design decision: generic functions specializing onto Scalar.method sigils

	lib/builtins/__scalar_arith.py (both the arithmetic dunders and the
	.to_T() conversions) is generated by gen_scalar_arith.py as ONE real
	generic function body per (operator, mode) shape, specialized per
	concrete scalar type via `Scalar.method = generic_fn[ConcreteType]` -
	not one hand-duplicated function per type the way this file's
	predecessor worked. This is real language-level generics (`def f[T](...)`,
	backed by monomorphize.py's Monomorphizer), reused here for the first
	time on a bare, still-abstract type parameter whose own operand type
	drives arithmetic-opcode selection - previously every generic in this
	codebase (`sys.alloc[T]`, `list[T]`, ...) only ever used its type param
	for container/pointer shape, never as the direct operand of `+`/a
	checked-arithmetic intrinsic.

	Mechanically: discovery.py's `visit_Assign` (the `TypeName.method = ...`
	sigil recognizer) accepts a `Specialization` RHS in addition to a plain
	`Function`, storing it RAW, unmonomorphized, directly in
	`Scalar.names` - eager monomorphization right there isn't possible, since
	`Monomorphizer` (and the `schedule`/`union_storage`/`tuple_storage` it
	depends on) doesn't exist yet at discovery time, only from
	`TypeResolver.__init__` onward. Every reader of `Scalar.names` therefore
	monomorphizes on first read instead: lowering.py's
	`Lowering._resolve_scalar_name` (used by `_find_method`/
	`_find_dunder_for_arg`) and type_resolver.py's `_attr_lookup_callable`
	(the ordinary `x.method(...)` call-resolution path, confirmed via a real
	spike to already reach `Scalar.names` generically for any name, not
	just dunders - no new dispatch mechanism was needed for `.to_T()` to be
	callable as an ordinary method). Once monomorphized, a Scalar-registered
	generic instantiation behaves identically to a hand-written one
	everywhere downstream - same `is_inline`/`is_fallible_arithmetic` flags
	(carried through by `dataclasses.replace` in
	`_build_monomorphized_function`), same zero-overhead `@inline` splicing.

	Finding this mechanism actually usable this way surfaced one real,
	independent, pre-existing bug along the way (not caused by genericity
	itself, confirmed via the completely unmodified, non-generic shipped
	file): `_find_dunder_for_arg`'s own dunder-matching check required
	exactly one declared parameter, correct for a real class method (`self`
	already stripped by discovery) but wrong for a Scalar-registered dunder
	(a free function whose receiver is never stripped - always two
	parameters) - silently rejecting EVERY Scalar-registered dunder since
	`@fallible_arithmetic` first shipped, undetected because the fallback
	path it silently fell through to happened to produce correct results
	for every previously-tested call shape. See memory file
	scalar_dunder_dispatch_never_matched_bug_fixed.md for the full story.

	`_emit_fallible_method_call` (lowering.py, extracted from the binop-
	specific `_emit_binop_dunder_call`) is the shared "resolve+schedule,
	splice-or-Call, consume via ambient mode if @fallible_arithmetic" tail
	both binop dunder dispatch and `.to_T()` method-call dispatch reuse -
	generalized specifically so a second call shape (one operand, no second
	argument) didn't need to duplicate that logic.
