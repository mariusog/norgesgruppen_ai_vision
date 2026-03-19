---
name: prd-to-issues
description: Break a PRD into independently-grabbable GitHub issues using tracer-bullet vertical slices. Use when the user wants to convert a PRD to issues, create implementation tickets, or break down a PRD into work items.
---

# PRD to Issues

Break a PRD into independently-grabbable GitHub issues using vertical slices (tracer bullets).

## Process

### 1. Locate the PRD

Ask the user for the PRD GitHub issue number (or URL).

If the PRD is not already in your context window, fetch it with `gh issue view <number>`.

### 2. Explore the Codebase (optional)

If you have not already explored the codebase, do so to understand the current state of the code.

### 3. Draft Vertical Slices

Break the PRD into **tracer bullet** issues. Each issue is a thin vertical slice that cuts through ALL integration layers end-to-end, NOT a horizontal slice of one layer.

Slices may be **HITL** or **AFK**:
- **HITL** (human-in-the-loop): requires human interaction, such as an architectural decision or a design review
- **AFK**: can be implemented and merged without human interaction

Prefer AFK over HITL where possible.

**Vertical slice rules:**
- Each slice delivers a narrow but COMPLETE path through every layer (schema, API, UI, tests)
- A completed slice is demoable or verifiable on its own
- Prefer many thin slices over few thick ones

### 4. Quiz the User

Present the proposed breakdown as a numbered list. For each slice, show:

- **Title**: short descriptive name
- **Type**: HITL / AFK
- **Blocked by**: which other slices (if any) must complete first
- **User stories covered**: which user stories from the PRD this addresses

Ask the user:

- Does the granularity feel right? (too coarse / too fine)
- Are the dependency relationships correct?
- Should any slices be merged or split further?
- Are the correct slices marked as HITL and AFK?

Iterate until the user approves the breakdown.

### 5. Create the GitHub Issues

For each approved slice, create a GitHub issue using `gh issue create`. Create issues in dependency order (blockers first) so you can reference real issue numbers in the "Blocked by" field.

Use this issue body template:

```markdown
## Parent PRD

#<prd-issue-number>

## What to build

A concise description of this vertical slice. Describe the end-to-end behavior, not layer-by-layer implementation. Reference specific sections of the parent PRD rather than duplicating content.

## Acceptance criteria

- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## Blocked by

- Blocked by #<issue-number> (if any)

Or "None - can start immediately" if no blockers.

## User stories addressed

Reference by number from the parent PRD:

- User story 3
- User story 7
```

Do NOT close or modify the parent PRD issue.

## Gotchas

- **Creating issues that are too large**: An issue should be completable in a single focused session. If the description has more than 5-7 checklist items, split it into smaller issues.
- **Missing acceptance criteria**: Every issue must have a clear "how to verify this is done" section. Without it, there's no way to confirm completion and no basis for code review.
- **Ignoring blocking dependencies**: If issue B requires issue A's output, the dependency must be explicit. Untracked dependencies cause agents to stall or build on incomplete foundations.
