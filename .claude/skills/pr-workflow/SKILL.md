---
name: pr-workflow
description: Use when creating PRs, reviewing PRs, checking CI status, or when user mentions pull request, PR, merge, code review workflow, or branch strategy. Guides the full pull request lifecycle from branch creation through review to merge. Language-neutral.
---

# PR Workflow

## Branch Strategy

### Naming Convention

```
<agent-or-author>/<task-id>-<short-description>
```

Examples:
- `core/T12-cache-invalidation`
- `feature/T08-scoring-algorithm`
- `qa/T15-coverage-gaps`

### Branch Rules

- `main` is always deployable -- all tests pass, benchmarks meet baselines
- Feature branches diverge from `main` and merge back to `main`
- One logical change per branch (maps 1:1 to a TASKS.md task)
- Delete branches after merge

## Creating a PR

### Pre-PR Checklist

Before opening a PR, verify locally:

```bash
# 1. All tests pass
<test fast command from Tooling table>

# 2. Lint is clean
<lint command from Tooling table>

# 3. No unintended files
git status
git diff --stat main...HEAD

# 4. Commits are clean (one logical change per commit)
git log --oneline main...HEAD
```

### PR Structure

```markdown
## Summary
- <1-3 bullet points describing WHAT changed and WHY>

## Changes
- <List of files/modules modified and the nature of each change>

## Test plan
- [ ] Unit tests added/updated for new behavior
- [ ] Integration tests pass
- [ ] Benchmark results (if performance-sensitive):
  - Before: <metric>
  - After: <metric>
  - Seeds used: <seed list>

## Task reference
Closes T<number> from TASKS.md
```

### Opening the PR

```bash
# Push branch and create PR
git push -u origin <branch-name>

gh pr create \
  --title "Short description (under 70 chars)" \
  --body "$(cat <<'EOF'
## Summary
- <what and why>

## Test plan
- [ ] Tests pass
- [ ] Lint clean
EOF
)"
```

## Reviewing a PR

### Self-Review (before requesting review)

Read your own diff as if you're seeing it for the first time:

```bash
# View the full diff against main
gh pr diff <pr-number>

# Or locally
git diff main...HEAD
```

### Review Checklist

**Correctness:**
- [ ] Does the code do what the PR description says?
- [ ] Are edge cases handled?
- [ ] Are there off-by-one errors or boundary issues?

**Quality:**
- [ ] Follows project code quality standards (CLAUDE.md Hard Limits)?
- [ ] No magic numbers (all in constants file)?
- [ ] Type annotations on public APIs?
- [ ] No swallowed errors or bare except/catch?

**Testing:**
- [ ] New public methods have tests?
- [ ] Tests are deterministic (fixed seeds)?
- [ ] Tests follow Arrange-Act-Assert?

**Safety:**
- [ ] No unbounded loops or recursion?
- [ ] No secrets committed (.env, API keys)?
- [ ] Validation at system boundaries?

**Scope:**
- [ ] Changes are focused on one task?
- [ ] No unrelated refactoring mixed in?
- [ ] No unnecessary formatting changes?

## CI Verification

### Checking CI Status

```bash
# Check PR status and CI checks
gh pr checks <pr-number>

# Watch CI in real-time
gh pr checks <pr-number> --watch

# View CI logs if a check failed
gh run view <run-id> --log-failed
```

### When CI Fails

1. Read the failed check output
2. Fix locally, push new commit (do NOT force-push)
3. Wait for CI to re-run
4. If the failure is flaky/unrelated, note it in the PR. To determine if a failure is flaky: (1) check if the failing test is related to your changes — if not, it may be pre-existing, (2) re-run the CI pipeline once — if it passes on retry, it's flaky, (3) check recent CI history on the base branch — if the same test fails there too, it's a pre-existing issue. Only mark as flaky after at least one of these checks.

## Merge Strategy

### When to Merge

- All CI checks pass
- At least one approval (if team project)
- No unresolved review comments
- Branch is up to date with `main`

### How to Merge

```bash
# Squash merge (preferred -- clean history)
gh pr merge <pr-number> --squash

# Or merge commit (when individual commits are meaningful)
gh pr merge <pr-number> --merge

# Never rebase merge (rewrites history, breaks references)
```

### Post-Merge Cleanup

```bash
# Delete the remote branch (gh does this automatically with --delete-branch)
gh pr merge <pr-number> --squash --delete-branch

# Update local main
git checkout main
git pull

# Delete local branch
git branch -d <branch-name>

# Update TASKS.md
# Set task status to `done`, add result note
```

## Handling Common Situations

### PR Has Merge Conflicts

```bash
# Update your branch with main
git checkout <branch-name>
git merge main
# Resolve conflicts, then:
git add <resolved-files>
git commit -m "Merge main and resolve conflicts"
git push
```

### PR Needs Changes After Review

```bash
# Make fixes in new commits (don't amend/squash during review)
git add <files>
git commit -m "Address review: <what changed>"
git push
```

### PR Is Too Large

Split it:
1. Identify independent pieces (e.g., refactor vs. feature)
2. Create separate branches from main for each piece
3. Open smaller, focused PRs
4. Reference the original task in each PR

**Rule of thumb**: If a PR exceeds 10 files or 300 lines of diff, actively look for a natural split point (e.g., refactoring vs. feature, backend vs. frontend). If all changes are tightly coupled and splitting would require duplicating context, proceed as one PR but note the size in the description. Trivial changes (import reordering, formatting) don't count toward the limit.

## Multi-Agent PR Workflow

When agents work in parallel via worktrees:

1. Each agent works on a separate branch (one task per branch)
2. Agents push and open PRs independently
3. QA agent reviews PRs before merge
4. Lead agent resolves conflicts between agent PRs
5. Merge order: dependencies first, then dependents

```
Agent A: core/T12-cache  ─── PR #1 ──┐
Agent B: feature/T08-scoring ─ PR #2 ─┼── QA reviews ── Lead merges
Agent C: qa/T15-coverage  ─── PR #3 ──┘
```

## Anti-Patterns

| Anti-Pattern | Fix |
|--------------|-----|
| Giant PRs (1000+ lines) | Split into focused PRs |
| Mixing refactoring with features | Separate PRs for each |
| Force-pushing during review | Push new commits |
| Merging with failing CI | Fix CI first |
| "LGTM" without reading diff | Use the review checklist |
| Rebasing public branches | Use merge to update from main |
| Committing review fixes as amends | New commits preserve review context |

## Gotchas

- **Describing WHAT changed but not WHY**: "Updated user.py" tells the reviewer nothing. The PR description should explain the motivation: what problem this solves and why this approach was chosen.
- **Including unrelated changes**: A PR that fixes a bug AND refactors an unrelated module is hard to review and risky to merge. One logical change per PR — split if you notice scope creep.
- **Not running tests locally before pushing**: Pushing and waiting for CI is slower than running tests locally first. Catch failures before they block the review cycle.

## Checklist

- [ ] Branch named correctly (`<author>/<task-id>-<description>`)
- [ ] One logical change per PR
- [ ] Self-review completed before requesting review
- [ ] CI passes (lint, test, security)
- [ ] PR description has summary, test plan, and task reference
- [ ] Review checklist completed
- [ ] Squash-merged after approval
- [ ] Branch deleted after merge
- [ ] TASKS.md updated with result
