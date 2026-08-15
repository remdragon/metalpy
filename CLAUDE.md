# Working in this repo

## Always use a fresh worktree

For any work beyond a trivial read-only question, create a **new** worktree
with `EnterWorktree` at the start of the session — even if the request names
an existing worktree path (e.g. "the worktree at `.claude/worktrees/foo`") or
describes where some in-progress work or a bug currently lives. Naming an
existing path is not an instruction to work there; it's just describing
context. Create a new worktree regardless, and port over only the specific
files/diff actually needed.

**Why:** Multiple sessions run against this repo concurrently, including
against the same named worktree. Reusing a worktree a request happens to
mention has caused real collisions: another session's merge landed mid-task,
files were caught in a half-edited state, and test failures showed up and
then silently vanished as the other session's own edits moved past them.

**Never work directly in the shared checkout** (`C:\cvs\metalpy` itself,
outside `.claude/worktrees/`) for anything that edits files — it's actively
committed to by other sessions and uncommitted work there can be silently
reverted.

If you do end up pointed at an existing worktree (e.g. mid-session, after
being told about it), and later need to make the work durable/uncommitted-
safe, migrate the diff to a fresh worktree of your own rather than leaving it
only in the shared one.

## Exception: merging a branch into master

The shared checkout is where `master` (and possibly other long-lived
branches) live, since git only allows one worktree per branch — there's no
way to get a second checkout of `master` to merge into. When the user
explicitly asks to merge a finished branch into master, it's fine to `cd`
into the shared checkout and run the merge there, e.g.:

```
cd C:\cvs\metalpy
git status                    # confirm clean before touching anything
git merge <branch-to-merge>
```

This exception covers only the merge itself (and the `git status` check
first) — not general file editing, not other commits, not resolving
unrelated conflicts by rewriting code. If `git status` shows anything
uncommitted or in-progress that isn't yours, stop and flag it rather than
merging over it. If the merge is a clean fast-forward, prefer that over a
merge commit.
