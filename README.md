Welcome to MetalPy

MetalPy is something I've wanted to build for years. It is a love
letter to the Python programming language. It compiles Python source
(with a few significant exceptions) down to C source code and generates
an executable.

As I said, I love the Python programming language. I can express ideas
and implement them very rapidly.

There's a few reasons for MetalPy, a few things about Python that are
less than ideal:

	* it's hard to know what Exceptions might interrupt your code
	* performance is rather poor.
	* building and distributing a hello world windows executable is 10's of megabytes.

In order to make something that fixes these issues, but remains as similar
to Python as possible, here are the solutions that I came up with:

	* Exceptions are out
		- Result objects that can't be ignored, inspired by Rust
		- defer and errdefer for resource cleanup, inspired by Zig
	* "everything is an object" is not supported
	* Multiple Inheritance is not supported
		- very hard to get right for very little benefit

Features that exist now:

	* memory-management through automatic ref-counting
	* unbounded int class
	* threadsafe list
	* threadsafe dict
	* tuple
	* anonymous unions like int|str with both compile time and runtime type narrowing
	* subclasses
	* file i/o
	* generic functions and classes
	* threading, non-reentrant Lock, atomic primitives
	* same-named functions chosen at either compile-time or run-time based on parameters
	* ability to create actual C structs (can interact directly with COM interfaces)
	* on Windows, the stdlib avoids linking against msvcrt. However, use of any
		feature that requires the C runtime will bring it in.
	* lazily scan source file and only compile things actually needed

Features that are being scoped and built right now:
	* f-strings
	* global object initialization on startup
	* inline functions
	* generators

Features that are planned but not built yet:
	* threadsafe Queue
	* socket library
	* http client/server classes
	* smtp library
	* email parsing
	* re library
	* json library
	* parser library

Features that would be nice to have:
	* pip-like package manager

Python functionality that I have no plans to implement:
	* async/await
	* multiple inheritance
	* everything is an object
	* monkey-patching
	* Exceptions

Other thoughts:

Right now, metalpy uses Python's own AST parser and uses a C compiler to generate
the actual executable. This gives MetalPy a lot of portability. While I plan to build
a parser library, I don't have any plan to make MetalPy self-hosting. There are some
small things about MetalPy's syntax that could be made nicer by stepping away from
Python's parser, but restricting to Python's AST parser opens up a lot of opportunities
to use existing tools for source code analysis.

