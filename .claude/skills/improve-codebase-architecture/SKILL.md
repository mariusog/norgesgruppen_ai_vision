---
name: improve-codebase-architecture
description: Explore a codebase to find opportunities for architectural improvement, focusing on making the codebase more testable by deepening shallow modules. Use when the user wants to improve architecture, find refactoring opportunities, consolidate tightly-coupled modules, or make a codebase more AI-navigable.
---

# Improve Codebase Architecture

Explore a codebase organically, surface architectural friction, discover opportunities for improving testability, and propose module-deepening refactors as GitHub issue RFCs.

A **deep module** (John Ousterhout, "A Philosophy of Software Design") has a small interface hiding a large implementation. Deep modules are more testable, more AI-navigable, and let you test at the boundary instead of inside.

## Process

### 1. Explore the Codebase

Use the Agent tool with subagent_type=Explore to navigate the codebase naturally. Do NOT follow rigid heuristics -- explore organically and note where you experience friction:

- Where does understanding one concept require bouncing between many small files?
- Where are modules so shallow that the interface is nearly as complex as the implementation?
- Where have pure functions been extracted just for testability, but the real bugs hide in how they're called?
- Where do tightly-coupled modules create integration risk in the seams between them?
- Which parts of the codebase are untested, or hard to test?

The friction you encounter IS the signal.

Spend no more than 10-15 minutes on exploration. You're looking for friction, not building a complete map. If you haven't found friction after reading 10-15 files, the codebase may already be well-structured — report that as a finding.

### 2. Present Candidates

Present 3-5 candidates, ranked by impact (most coupling/friction first). If you found more than 5, include only the top 5 — the user can always ask for more.

Present a numbered list of deepening opportunities. For each candidate, show:

- **Cluster**: Which modules/concepts are involved
- **Why they're coupled**: Shared types, call patterns, co-ownership of a concept
- **Dependency category**: See the Dependency Categories section below
- **Test impact**: What existing tests would be replaced by boundary tests

Do NOT propose interfaces yet. Ask the user: "Which of these would you like to explore?"

### 3. User Picks a Candidate

### 4. Frame the Problem Space

Before spawning sub-agents, write a user-facing explanation of the problem space for the chosen candidate:

- The constraints any new interface would need to satisfy
- The dependencies it would need to rely on
- A rough illustrative code sketch to make the constraints concrete -- this is not a proposal, just a way to ground the constraints

Show this to the user, then immediately proceed to Step 5. The user reads and thinks about the problem while the sub-agents work in parallel.

### 5. Design Multiple Interfaces

Spawn 3+ sub-agents in parallel using the Agent tool. Each must produce a **radically different** interface for the deepened module.

Prompt each sub-agent with a separate technical brief (file paths, coupling details, dependency category, what's being hidden). Give each agent a different design constraint:

- Agent 1: "Minimize the interface -- aim for 1-3 entry points max"
- Agent 2: "Maximize flexibility -- support many use cases and extension"
- Agent 3: "Optimize for the most common caller -- make the default case trivial"
- Agent 4 (if applicable): "Design around the ports & adapters pattern for cross-boundary dependencies"

Each sub-agent outputs:

1. Interface signature (types, methods, params)
2. Usage example showing how callers use it
3. What complexity it hides internally
4. Dependency strategy (how deps are handled -- see Dependency Categories below)
5. Trade-offs

Present designs sequentially, then compare them in prose.

After comparing, give your own recommendation: which design you think is strongest and why. If elements from different designs would combine well, propose a hybrid. Be opinionated -- the user wants a strong read, not just a menu.

### 6. User Picks an Interface (or Accepts Recommendation)

### 7. Create GitHub Issue

Create a refactor RFC as a GitHub issue using `gh issue create`. Do NOT ask the user to review before creating -- just create it and share the URL. Use the RFC template below.

## Dependency Categories

When assessing a candidate for deepening, classify its dependencies:

### 1. In-process

Pure computation, in-memory state, no I/O. Always deepenable -- just merge the modules and test directly.

### 2. Local-substitutable

Dependencies that have local test stand-ins (e.g., PGLite for Postgres, in-memory filesystem). Deepenable if the test substitute exists. The deepened module is tested with the local stand-in running in the test suite.

### 3. Remote but Owned (Ports & Adapters)

Your own services across a network boundary (microservices, internal APIs). Define a port (interface) at the module boundary. The deep module owns the logic; the transport is injected. Tests use an in-memory adapter. Production uses the real HTTP/gRPC/queue adapter.

### 4. True External (Mock)

Third-party services (Stripe, Twilio, etc.) you don't control. Mock at the boundary. The deepened module takes the external dependency as an injected port, and tests provide a mock implementation.

## Testing Strategy

The core principle: **replace, don't layer.**

- Old unit tests on shallow modules are waste once boundary tests exist -- delete them
- Write new tests at the deepened module's interface boundary
- Tests assert on observable outcomes through the public interface, not internal state
- Tests should survive internal refactors -- they describe behavior, not implementation

## RFC Issue Template

```markdown
## Problem

Describe the architectural friction:

- Which modules are shallow and tightly coupled
- What integration risk exists in the seams between them
- Why this makes the codebase harder to navigate and maintain

## Proposed Interface

The chosen interface design:

- Interface signature (types, methods, params)
- Usage example showing how callers use it
- What complexity it hides internally

## Dependency Strategy

Which category applies and how dependencies are handled:

- **In-process**: merged directly
- **Local-substitutable**: tested with [specific stand-in]
- **Ports & adapters**: port definition, production adapter, test adapter
- **Mock**: mock boundary for external services

## Testing Strategy

- **New boundary tests to write**: describe the behaviors to verify at the interface
- **Old tests to delete**: list the shallow module tests that become redundant
- **Test environment needs**: any local stand-ins or adapters required

## Implementation Recommendations

Durable architectural guidance that is NOT coupled to current file paths:

- What the module should own (responsibilities)
- What it should hide (implementation details)
- What it should expose (the interface contract)
- How callers should migrate to the new interface
```

## Gotchas

- **Proposing refactors that break all existing tests**: A "better" architecture that requires rewriting every test is a net negative. Propose changes that can be migrated incrementally, with existing tests still passing at each step.
- **Over-abstracting with one implementation**: Creating an interface or abstraction layer when only one concrete implementation exists adds complexity without benefit. Abstract when you have 2+ real implementations, not before.
- **Refactoring stable, working code**: Code that works, is tested, and rarely changes doesn't need architectural improvement regardless of how it looks. Focus on code that is actively causing friction.
