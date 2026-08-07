Current state of list[T]

The implementation is built on a sophisticated RawList core with stable ID mapping and O(1) swap-and-pop erasure. The list[T] wrapper handles RC management. But it's a
non-Python-like API — uses stable IDs instead of positional indices for most operations.

Gaps vs Python list

Core missing methods (must-have)

┌──────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Method                       │ Notes                                                                                                                                         │
├──────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ `__getitem__`                │ Current takes stable ID; needs Python-style positional index (0-based, negative indices). Also should not return `Result` — should            │
│                              │ panic/raise                                                                                                                                   │
│ `__setitem__`                │ `lst[i] = val` — doesn't exist                                                                                                                │
│ `__delitem__`                │ `del lst[i]` — doesn't exist. Could reuse `erase_at`                                                                                          │
│ `__iter__` / for-loop        │ Needs `__len__` + `__getitem__` for lowering's current for-loop desugaring                                                                    │
│ support                      │                                                                                                                                               │
│ `pop([i])`                   │ Remove and return item at position                                                                                                            │
│ `insert(i, val)`             │ Insert at position (expensive — shift elements right)                                                                                         │
│ `remove(val)`                │ Find first occurrence and remove                                                                                                              │
│ `index(val)`                 │ Find index of first occurrence                                                                                                                │
│ `count(val)`                 │ Count occurrences                                                                                                                             │
│ `reverse()`                  │ Reverse in-place                                                                                                                              │
│ `extend(iterable)`           │ Append from another iterable                                                                                                                  │
└──────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

Nice-to-have

┌─────────────────────┬───────────────────────────────────────────────────┐
│ Method              │ Notes                                             │
├─────────────────────┼───────────────────────────────────────────────────┤
│ `sort()`            │ Requires comparison operators on `T`. Big project │
│ `copy()`            │ Shallow copy — simple with raw memcpy + incref    │
│ `__eq__` / `__ne__` │ Structural equality                               │
│ `__contains__`      │ `x in lst`                                        │
└─────────────────────┴───────────────────────────────────────────────────┘

Thread-safety

Metalpy RCClasses use _Atomic int32_t for refcounting, so individual list objects can be safely shared across threads without use-after-free. But the list's mutable state has no
synchronization — concurrent append/erase from two threads is a data race.

Real Python avoids this via the GIL. Metalpy is a systems language — this is the same model as C++ std::vector. The fix is either:
• Document list as not thread-safe for concurrent mutation (like C++ std::vector)
• Add an optional lock — list[T](threadsafe=True) that wraps every mutation in a mutex

Testing approach

1. Correctness: Create list[i32], exercise every method, verify state
2. RC correctness: Create list[SomeRCClass], verify destructor runs correct number of times (incref/decref counts)
3. Memory leaks: Create/destroy lists with RC elements; verify all __del__ fire
4. No existing tests: There are zero tests for list[T] today

Implementation plan

1. Fix __getitem__ to use positional indices (with negative index support) — this is the blocker for for-loop iteration and the biggest API gap
2. Add __setitem__ — lst[i] = val (decref old, incref+store new)
3. Add pop(), insert(), reverse(), extend(), index(), count(), remove()
4. Write comprehensive correctness tests
5. Write RC-leak tests (create list[Foo], append/remove, verify destructor counts)
6. Document thread-safety (non-thread-safe for mutations, like C++ std::vector)
