---
name: security-scan
description: Run security scans for code vulnerabilities and dependency audits. Use when the user asks to check for security issues, run security scans, or audit dependencies.
---

# Security Scan Skill

Scan for security vulnerabilities in code and dependencies.

## Scope

This skill scans the entire project for security issues.
It is not limited to changed files since vulnerabilities can exist anywhere.

## Step 1: Static Analysis

Use the **Security scan** command from the CLAUDE.md Tooling table.

### Common Issues to Fix (all languages)

**Hardcoded Passwords/Secrets**
- Move to environment variables or config files
- Use `.env` files (excluded via `.gitignore`)

**Unsafe Deserialization**
- Avoid deserializing untrusted data with unsafe methods
- Prefer JSON for data exchange

**Command Injection**
- Never pass user input directly to shell commands
- Use parameterized/safe command execution

**SQL Injection**
- Use parameterized queries, never string formatting

**Unsafe Config Loading**
- Use safe parsing modes (e.g., `yaml.safe_load`, not `yaml.load`)

### Language-Specific Tools

| Language | Static Analysis | Dependency Audit |
|----------|----------------|-----------------|
| Python | `bandit -r . -ll` | `pip-audit` |
| TypeScript/JS | `npm audit` | `npm audit` |
| Go | `gosec ./...` | `govulncheck ./...` |
| Rust | `cargo clippy` | `cargo audit` |

### Handling Warnings
- Fix the code properly
- Do NOT suppress with ignore comments unless absolutely necessary
- If a false positive, document why in a comment

## Step 2: Dependency Audit

Check for known vulnerabilities in dependencies. Update vulnerable packages if patched versions exist. If no patch available, add a comment in the dependency file (e.g., `requirements.txt`, `package.json`) explaining the vulnerability and why no update is possible, and note it in the Completion report.

## Step 3: Additional Checks

### Check for Hardcoded Secrets
Search for: API keys, tokens, passwords, private keys, connection strings.

Use grep to scan for common secret patterns:
```sh
grep -rn --include='*.py' --include='*.ts' --include='*.go' --include='*.rs' --include='*.rb' --include='*.yml' --include='*.json' \
  -iE '(password|secret|token|api_key|apikey|private_key)\s*[:=]\s*["'"'"'][^"'"'"']{8,}' . 2>/dev/null | grep -v node_modules | grep -v .git
```

Also check for base64-encoded strings and high-entropy strings in config files.

### Check .gitignore
Ensure sensitive files are excluded:
- `.env`, `.env.*`
- `*.pem`, `*.key`
- `credentials.*`, `secrets.*`

## Step 4: Verify Clean

Re-run all scans to confirm issues are resolved.

## Gotchas

- **Grep patterns miss obfuscated secrets**: Base64-encoded tokens, split strings, and environment variable names that don't match common patterns (`MY_TOKEN` vs `password`) evade regex-based scanning. Check config files manually.
- **Dismissing warnings as false positives too quickly**: If a security tool flags something, investigate before dismissing. Many "false positives" are actually real issues in edge cases.
- **Ignoring transitive dependency vulnerabilities**: `pip-audit` and `npm audit` may flag vulnerabilities in deep dependencies you don't directly control. These still need assessment — the vulnerability is in your runtime regardless of who wrote it.

## Scoring (0-100)

Start at 100. Deduct points for each **unresolved** finding (after fixes in Steps 1-3):

| Severity | Deduction per finding |
|----------|----------------------|
| Critical (RCE, injection, auth bypass) | -20 |
| High (hardcoded secrets, unsafe deserialization) | -10 |
| Medium (missing input validation, weak crypto) | -5 |
| Low (informational, best-practice deviation) | -2 |
| Vulnerable dependency (unpatched) | -5 |

| Score | Interpretation |
|-------|---------------|
| 90-100 | Secure -- no critical or high findings |
| 70-89 | Acceptable -- no critical findings, some medium issues |
| 50-69 | At risk -- high-severity findings or multiple medium |
| 0-49 | Unsafe -- critical vulnerabilities present |

## Completion

Report:
- Number of warnings found and fixed
- Number of vulnerable dependencies found and updated
- Any remaining issues that need attention
- **Score: X/100** (deductions itemized by severity)
