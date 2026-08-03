thoughts on setting attributes of an RCClass:

I believe we can assume in most cases that an RCClass is complete, so setting
an attribute is always a replace.

deleting an attribute is a compile error.

The one exception to this rule that I can think of is __init__().
__init__() is going to require a slight tweak to the CFG. The CFG
needs to treat each of the class's attributes as uninitialized locals.

Also, even though we haven't implemented subclassing, we need to spec
some things here because they will affect the design

If __init__() calls super.__init__(), any base-class attributes will
be considered initialized because its __init__() function has the same
initialization requirements. only newly defined attributes in the
subclass itself will be uninitialized.

if __init__() does *not* call super.__init__(), then it is responsible
for initializing all base ( and base of base, etc ) attributes itself.

Any uninitialized variable when __init__() returns would be a compile error.

Now, with a fallible __init__(), I think this should be a simple matter of
treating the class attributes like local variables and clean up any that
were initialized up to that point.

Some examples:

class Foo:
	a: int
	b: int
	
	def __init__( self ):
		self.a = 0
		# compile error - b not initialized

class Foo1:
	a: int
	
	def __init__( self ):
		self.a = 0

class Foo2( Foo1 ):
	b: int

	def __init__( self ):
		super.__init__()
		self.b = 0
		# compiles because super.__init__() marks a initialized

class Foo3( Foo1 ):
	b: int
	def __init__( self ):
		self.b = 0
		# compile error - a not initialized (call super.__init__() or initialize a directly)

This begs an interesting question, what if the base class defines private attributes?

To me the answer is simple, super.__init__() must be called


default values should be captured as expressions and emitted __init__()'s prologue

class Foo:
	a: int = 0
	
	def __init__( self ):
		# prologue: a = (default value expression above)
		self.a = 1 # overwrites the default value

also...

class Foo:
	a: int = 0

this is also okay, because the default __init__() just binds to __allocate__() and the default
should be processed there too
