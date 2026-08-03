The purpose of this file is to explore how function-level resource cleanup
needs to work.

We are going to replace _needs_epilogue and
_epilogue_label with a single variable:
	
	@dataclass
	class Epilogue:
		name: str
		instructions: list[ast.AST] # I don't know if this is the right type...
		emitted: bool = False
	
	_epilogue_stack: list[Epilogue]

Any current references to _needs_epilogue can be replaced with bool(_epilogue_stack)

Any current references to _epilogue_label can be replaced with _epilogue_stack[-1].name

I'm typing up a question to you regarding how hard is it to scope a variable's lifetime
automatically based on del x or whether it never gets referenced outside a certain
context. I'm typing up this example based on the answer being that it is feasible.

```metalpy pseudo-code
class Foo:
	pass

def bar( a: Foo, b: Foo ) -> Result[_,OverflowError]:
	# begin prologue:
	# if any RC parameters:
	# 	create initial epilogue stack entry
	# 	for each RC parameter:
	# 		notify CFG of RC variable and its epilogue stack entry
	# >>> inject defer flag setup here <<<
	# end prologue
	
	defer( a.close() )
	# create new epilogue stack entry and register defer body to it
	
	if a == b:
		return
		# if any entries in epilogue_stack:
		# 	emit goto epilogue_stack[-1]
		# else:
		# 	emit ir.Return
	
	c = a
	# create new epilogue stack entry and register DecRef to it
	
	# begin epilogue
	# 	for each item in epilogue stack (in reverse order):
	# 		emit ir.Label
	# 		emit instructions
	# 		pop item off epilogue stack
	# 	emit ir.Return
	# end epilogue
```
