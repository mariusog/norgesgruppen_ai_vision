---
name: prd-to-plan
description: Break a PRD into a phased implementation plan using vertical slices (tracer bullets), output as a local Markdown file. Use when the user wants to create an implementation plan from a PRD without creating GitHub issues.
---

# PRD to Plan

Break a PRD into a phased implementation plan using vertical slices (tracer bullets). Output is a Markdown file in `./plans/`.

## Process

### 1. Confirm the PRD Is in Context

The PRD should already be in the conversation. If it isn't, ask the user to paste it or point you to the file.

### 2. Explore the Codebase

If you have not already explored the codebase, do so to understand the current architecture, existing patterns, and integration layers.

### 3. Identify Durable Architectural Decisions

Before slicing, identify high-level decisions that are unlikely to change throughout implementation:

- Route structures / URL patterns
- Database schema shape
- Key data models
- Authentication / authorization approach
- Third-party service boundaries

These go in the plan header so every phase can reference them.

### 4. Draft Vertical Slices

Break the PRD into **tracer bullet** phases. Each phase is a thin vertical slice that cuts through ALL integration layers end-to-end, NOT a horizontal slice of one layer.

**Vertical slice rules:**
- Each slice delivers a narrow but COMPLETE path through every layer (schema, API, UI, tests)
- A completed slice is demoable or verifiable on its own
- Prefer many thin slices over few thick ones
- Do NOT include specific file names, function names, or implementation details that are likely to change as later phases are built
- DO include durable decisions: route paths, schema shapes, data model names

### 5. Quiz the User

Present the proposed breakdown as a numbered list. For each phase show:

- **Title**: short descriptive name
- **User stories covered**: which user stories from the PRD this addresses

Ask the user:

- Does the granularity feel right? (too coarse / too fine)
- Should any phases be merged or split further?

Iterate until the user approves the breakdown.

### 6. Write the Plan File

Create `./plans/` if it doesn't exist. Write the plan as a Markdown file named after the feature (e.g. `./plans/user-onboarding.md`).

Use this template:

```markdown
# Plan: <Feature Name>

> Source PRD: <brief identifier or link>

## Architectural decisions

Durable decisions that apply across all phases:

- **Routes**: ...
- **Schema**: ...
- **Key models**: ...
- (add/remove sections as appropriate)

---

## Phase 1: <Title>

**User stories**: <list from PRD>

### What to build

A concise description of this vertical slice. Describe the end-to-end behavior, not layer-by-layer implementation.

### Acceptance criteria

- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

---

## Phase 2: <Title>

**User stories**: <list from PRD>

### What to build

...

### Acceptance criteria

- [ ] ...

<!-- Repeat for each phase -->
```

## Gotchas

- **Including implementation details that will change**: File paths, function names, and specific class structures don't belong in a plan — they'll change during implementation. Focus on responsibilities, interfaces, and data flow.
- **Planning too many phases**: More than 3-4 phases usually means the plan is too granular. Each phase should deliver a working vertical slice. If phases depend on each other linearly, collapse them.
- **Skipping the tracer bullet**: The first phase must be a minimal end-to-end slice that proves the architecture works. Don't plan Phase 1 as "set up infrastructure" — plan it as "one user can do one thing, end to end."
