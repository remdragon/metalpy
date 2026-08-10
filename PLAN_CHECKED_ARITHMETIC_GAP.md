If a user actually wants the Result from an integer operation,
current the only way to acquire that is to return it from a function:

```
def foo() -> Result[i32,OverflowError]:
	return 1+2
```

but sometimes that may be inconvenient. There should be a way
for users to capture the Result of a checked operation without
being required to put it in a separate function.

Maybe something like this even though its ugly:

```
1.checked_add(2)
```

