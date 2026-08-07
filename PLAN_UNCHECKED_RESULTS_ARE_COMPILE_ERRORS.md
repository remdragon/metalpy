Proposal: Unchecked Result compile error

Goal

A Result[T,E] value bound to a local variable must be inspected before the variable goes out of scope, is overwritten, or reaches a merge point where the other branch didn't also
inspect it. The compiler should reject programs that silently discard errors.

What "inspected" means

A Result is consumed (cleared from tracking) when:

┌────────────────────────────────────┬─────────────────────┐
│ Call                               │ Effect              │
├────────────────────────────────────┼─────────────────────┤
│ `r.is_ok()`                        │ `clear_result('r')` │
│ `r.is_err()`                       │ `clear_result('r')` │
│ `r.or_return()`                    │ `clear_result('r')` │
│ `r.unwrap(msg)`                    │ `clear_result('r')` │
│ `r.unwrap_or(default)`             │ `clear_result('r')` │
│ `match r:` (at least one case arm) │ `clear_result('r')` │
└────────────────────────────────────┴─────────────────────┘

A Result is tracked (added to the set) when a Result-typed expression is assigned to a variable:

┌───────────────────────────────────────┬─────────────────────┐
│ Event                                 │ Effect              │
├───────────────────────────────────────┼─────────────────────┤
│ `x = <Result-expr>`                   │ `track_result('x')` │
│ `x: Result[...] = <expr>` (AnnAssign) │ `track_result('x')` │
└───────────────────────────────────────┴─────────────────────┘

Independent tracking

Aliases are tracked independently. y = r tracks y but does NOT clear r. Each binding must be checked on its own. This is simpler than cross-binding tracking and avoids subtle
bugs.

Validation errors

┌───────────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ When                                                                  │ Error                                                                                                │
├───────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Function exits (return / fall-off)                                    │ `Result value 'r' was never inspected — use .is_ok(), .is_err(), .or_return(), .unwrap(msg), or      │
│                                                                       │ match`                                                                                               │
│ Overwrite `r = <anything>`                                            │ `Result value 'r' is discarded — it was never inspected: ...`                                        │
│ `del r`                                                               │ `Result value 'r' is discarded via del — it was never inspected: ...`                                │
│ One if-branch checks `r` but the other doesn't (and neither           │ `Result 'r' was inspected on one branch but not the other`                                           │
│ terminates)                                                           │                                                                                                      │
└───────────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────┘

Scope cut (v1)

• return r transfers obligation to caller — caller's problem, not checked across function boundaries
• Passing a Result as a call argument — callee's problem
• Result-typed struct fields — same v1 scope cut as RC fields in cfg.py
• The above are compile-time errors to be added later; v1 covers the common local-variable case

Implementation

All state lives in cfg.CFGState as self._unchecked_results: set[str], snapshotted/restored alongside self.bindings:

**cfg.py changes:**
• Add is_result_type() helper (already done — survives the revert)
• Add _unchecked_results: set[str] to CFGState.__init__
• Add results: set[str] field to _Snapshot
• Update snapshot()/restore() to include results
• Add track_result(name), clear_result(name), is_unchecked(name) methods
• In assign(): error if overwriting an unchecked result; clear if non-Result type
• In deleted(): error if deleting an unchecked result
• In merge_if(): compare results consistency across branches
• In return_(): error if any live binding has unchecked results

**lowering.py changes:**
• In _stmt_Assign / _stmt_AnnAssign: call track_result when binding a Result value
• In _lower_call / _lower_or_return: call clear_result when is_ok/is_err/or_return/unwrap/unwrap_or is called on a Result binding
• In _lower_call final else block: error when discarding a Result-returning call result
• In _stmt_Match (type_resolver's rewrite): call clear_result when matching a Result binding

Files affected

┌──────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ File         │ Change                                                                                                                                                        │
├──────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ `cfg.py`     │ `_unchecked_results`, `_Snapshot.results`, snapshot/restore, `track_result`/`clear_result`/`is_unchecked`, validation in `assign()`, `deleted()`,             │
│              │ `merge_if()`, `return_()`                                                                                                                                     │
│ `lowering.py │ `track_result`/`clear_result` calls at bind/consume sites, discarded-call check                                                                               │
│ `            │                                                                                                                                                               │
│ `cfg_test.py │ New tests for the Result checking mechanics                                                                                                                   │
│ `            │                                                                                                                                                               │
└──────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
