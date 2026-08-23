# Working in this repo

## Always consult your memory

The user's primary development machine has access to 3 compilers — 2 on
Windows and 1 on Linux via WSL. Check memory for how to invoke each one so
code changes can be tested thoroughly.

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
explicitly asks to merge a finished branch into master, it's fine to exit
your worktree and `cd` into the shared checkout and run the merge there,
e.g.:

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

## Working a task

MetalPy is still under active development. It is not uncommon to come across
missing features that block the task at hand. Strongly prefer fixing bugs or
missing features immediately (in a separate session or background task if
necessary) instead of working around them.

When working on existing code, especially in the lib folder, keep an eye out for
code that is ugly or is working around current or past bugs and clean them up
if you can, otherwise draw my attention to them.

## Comments

Comments should be as concise as possible and not explain things that should
be obvious to a reasonably qualified reader. History lessons aren't needed in
comments, just the lessons learned from them if necessary, e.g. "asserting
instead of calling resolve here because all objects are supposed to be resolved
by now" instead of a verbose multiline explanation of a bug that had to be
hunted down to this point.

## Finishing a task

When summarizing a finished task, always include a list of any unresolved issues
such as bugs or gaps discovered or anything that deserves the user's attention.
