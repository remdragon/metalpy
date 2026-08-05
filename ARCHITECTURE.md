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
