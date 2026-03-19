---
name: readme-standards
description: Use when creating a new project README, reviewing documentation quality, or when user mentions README, project documentation, or onboarding docs. Guidelines for writing clear, complete README files. Language-neutral.
---

# README Standards

## Purpose

A README is the entry point for anyone (human or AI agent) encountering a project. It should answer three questions in under 60 seconds:

1. **What** does this project do?
2. **How** do I get it running?
3. **Where** do I find what I need?

## Required Sections

Every project README must include these sections, in this order:

### 1. Title and One-Line Description

```markdown
# Project Name

One sentence explaining what this project does and who it's for.
```

- No badges, logos, or decorations above the title
- The one-liner should be concrete, not marketing language
- Good: "A CLI tool that converts CSV files to typed Python dataclasses"
- Bad: "A next-generation data transformation framework for the modern developer"

### 2. Quick Start

```markdown
## Quick Start

\```bash
# Install
<install command>

# Run
<example usage command>
\```
```

- Maximum 5 commands from zero to working
- Show the most common use case, not every option
- If setup takes more than 5 steps, something is wrong with the setup

### 3. What It Does (optional for small projects)

```markdown
## Overview

- Reads CSV files and infers column types
- Generates Python dataclass definitions
- Supports nullable fields and custom type mappings
```

- Bullet points, not paragraphs
- Focus on capabilities, not implementation details
- Skip this section if the one-liner already says everything

### 4. Usage

```markdown
## Usage

\```bash
# Basic usage
myproject input.csv

# With options
myproject input.csv --output models.py --nullable
\```
```

- Show 2-3 real commands that cover the main use cases
- Include expected output if it's not obvious
- Link to detailed docs if the API is large

### 5. Project Structure (for repos with multiple files)

```markdown
## Project Structure

\```
src/           # Source code
tests/         # Test suite
docs/          # Generated reports
logs/          # Runtime logs
\```
```

- Only show top-level directories with one-line descriptions
- Don't list every file -- that's what `ls` is for
- Focus on where to find things, not what every file does

### 6. Development

```markdown
## Development

\```bash
# Setup
<language-specific setup commands>

# Test
<test command from CLAUDE.md Tooling table>

# Lint
<lint command from CLAUDE.md Tooling table>
\```
```

- How to set up a dev environment
- How to run tests
- How to run linting/formatting
- Keep it to the essential commands (reference CLAUDE.md for full details if applicable)

### 7. License

```markdown
## License

MIT License. See [LICENSE](LICENSE) for details.
```

## Optional Sections

Include these only when relevant:

| Section | When to Include |
|---------|-----------------|
| **Configuration** | Project has config files or environment variables |
| **API Reference** | Library with public API (link to generated docs) |
| **Contributing** | Open source project accepting contributions |
| **Changelog** | Project with versioned releases |
| **Architecture** | Complex project where understanding the design helps contributors |
| **FAQ / Troubleshooting** | Known issues that users hit repeatedly |

## Writing Style

### Do

- Use short, direct sentences
- Lead with the action: "Run `pytest`" not "You can run the tests by executing `pytest`"
- Use code blocks for anything the reader will type or read as output
- Keep the total README under 200 lines (link to detailed docs for more)
- Use tables for structured comparisons
- Update the README when the interface changes

### Don't

- Don't pad with badges, shields, or decorative images at the top
- Don't explain what a README is
- Don't include auto-generated API docs inline (link to them)
- Don't write paragraphs when bullet points work
- Don't include roadmap items or aspirational features
- Don't duplicate content from other docs (CLAUDE.md, CONTRIBUTING.md)
- Don't use "we" or "our" -- just describe what the project does

## AI-Agent Considerations

When an AI agent reads your README, it needs:

- **Concrete commands** it can run (not prose descriptions of how to set things up)
- **File paths** it can navigate to (not vague references like "the config")
- **Structured data** it can parse (tables and code blocks, not paragraphs)
- **Brevity** -- every extra line costs tokens

If the project has a `CLAUDE.md`, the README should focus on human onboarding and link to `CLAUDE.md` for AI-agent-specific instructions. Don't duplicate between the two.

## Scoring (0-100)

Score each area. Award the indicated points when the criteria are fully met, partial credit for partial coverage:

| Area | Points | Criteria for full points |
|------|--------|--------------------------|
| Title + one-line description | 15 | Concrete, not marketing language |
| Quick start | 20 | Zero to working in ≤5 commands; commands actually work |
| Usage examples | 15 | 2-3 real examples covering main use cases |
| Project structure | 10 | Top-level dirs with one-line descriptions |
| Development setup | 15 | Setup, test, lint commands that work |
| License | 5 | Stated clearly |
| Brevity | 10 | Under 200 lines, no padding or duplication |
| Accuracy | 10 | All commands work, all paths exist, no stale info |

| Score | Interpretation |
|-------|---------------|
| 90-100 | Complete -- all required sections present and well-written |
| 70-89 | Good -- most sections present, minor gaps |
| 50-69 | Incomplete -- missing key sections or stale content |
| 0-49 | Insufficient -- major sections missing or misleading |

## Gotchas

- **Setup instructions that only work on your machine**: "Run `make start`" fails if the reader doesn't have Make installed. List prerequisites explicitly and test instructions on a clean environment.
- **Documenting aspirational features as if they exist**: If a feature is planned but not built, don't put it in the README. Readers will try to use it and get confused. Mark future work clearly or omit it.
- **Letting the README grow past 200 lines**: A long README means information is buried. Move detailed content to separate docs and keep the README focused on quick start and orientation.

## Checklist

- [ ] Title + one-line description (concrete, not marketing)
- [ ] Quick start: zero to working in 5 commands or fewer
- [ ] Usage: 2-3 real examples of the main use cases
- [ ] Project structure: top-level directories with one-line descriptions
- [ ] Development: setup, test, lint commands
- [ ] License stated
- [ ] Under 200 lines total
- [ ] No stale information (commands actually work, paths exist)
- [ ] Code blocks for every command and file path
