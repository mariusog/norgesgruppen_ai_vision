---
name: write-a-prd
description: Create a PRD through user interview, codebase exploration, and module design, then submit as a GitHub issue. Use when the user wants to write a PRD, create a product requirements document, or plan a new feature.
---

# Write a PRD

Create a Product Requirements Document through structured interview and codebase exploration.

## Process

Steps 1 (context) and 5 (write PRD) are always required. Steps 2-4 can be shortened or skipped: skip Step 2 (codebase exploration) if the feature is standalone with no existing code to integrate with. Skip Step 3 (interview) if the user has already provided detailed requirements. Skip Step 4 (module design) if the feature is small enough to fit in a single module.

### 1. Gather the Problem

Ask the user for a long, detailed description of the problem they want to solve and any potential ideas for solutions.

### 2. Explore the Codebase

Explore the repo to verify their assertions and understand the current state of the codebase.

### 3. Interview

Interview the user relentlessly about every aspect of this plan until you reach a shared understanding. Walk down each branch of the design tree, resolving dependencies between decisions one-by-one.

### 4. Sketch Modules

Sketch out the major modules you will need to build or modify to complete the implementation. Actively look for opportunities to extract **deep modules** that can be tested in isolation.

A deep module (as opposed to a shallow module) is one which encapsulates a lot of functionality in a simple, testable interface which rarely changes.

Check with the user that these modules match their expectations. Check with the user which modules they want tests written for.

### 5. Write the PRD

Once you have a complete understanding of the problem and solution, write the PRD using the template below. The PRD should be submitted as a GitHub issue using `gh issue create`.

## PRD Template

```markdown
## Problem Statement

The problem that the user is facing, from the user's perspective.

## Solution

The solution to the problem, from the user's perspective.

## User Stories

A LONG, numbered list of user stories. Each user story should be in the format of:

1. As an <actor>, I want a <feature>, so that <benefit>

This list should be extremely extensive and cover all aspects of the feature.

## Implementation Decisions

A list of implementation decisions that were made. This can include:

- The modules that will be built/modified
- The interfaces of those modules that will be modified
- Technical clarifications from the developer
- Architectural decisions
- Schema changes
- API contracts
- Specific interactions

Do NOT include specific file paths or code snippets. They may end up being outdated very quickly.

## Testing Decisions

A list of testing decisions that were made. Include:

- A description of what makes a good test (only test external behavior, not implementation details)
- Which modules will be tested
- Prior art for the tests (i.e. similar types of tests in the codebase)

## Out of Scope

A description of the things that are out of scope for this PRD.

## Further Notes

Any further notes about the feature.
```

## Gotchas

- **Writing solutions instead of problems**: A PRD that specifies "build a REST API with endpoints X, Y, Z" has already decided the solution. Start with the problem: what user need is unmet? The implementation approach should emerge from the requirements, not the other way around.
- **Skipping the interview and assuming requirements**: Even when you think you understand what's needed, the interview step surfaces hidden constraints, edge cases, and priorities. Never skip it for non-trivial features.
- **Making the PRD too long**: A PRD over 2 pages is rarely read in full. Keep it focused: problem, key decisions, acceptance criteria. Move detailed specs to linked documents if needed.
