---
name: browser-testing
description: Use when building end-to-end browser tests, debugging UI issues, setting up Playwright/Cypress, or when user mentions browser testing, e2e tests, visual testing, page objects, or screenshot debugging. Covers MCP tool setup, test patterns, and product verification flows; examples use Playwright but principles apply to any browser automation framework.
---

# Browser Testing Skill

End-to-end browser testing patterns, MCP tool setup for visual debugging, and product verification flows. Examples use Playwright but principles are framework-neutral.

## Step 1: Choose Your Setup

### MCP Tools (for Claude-assisted debugging)

Configure MCP servers so Claude can see browser state directly:

| Tool | Purpose | Setup |
|------|---------|-------|
| [Playwright MCP](https://github.com/microsoft/playwright-mcp) | Browser automation, screenshots, DOM inspection | `claude mcp add playwright -- npx @anthropic-ai/playwright-mcp` |
| [Chrome DevTools MCP](https://developer.chrome.com/blog/chrome-devtools-mcp) | Console logs, network tab, performance | `claude mcp add chrome-devtools -- npx @anthropic-ai/chrome-devtools-mcp` |

With MCP configured, Claude can take screenshots, read console errors, and inspect DOM elements without you manually copying output.

### Test Framework Selection

| Framework | Best for | Languages |
|-----------|----------|-----------|
| Playwright | Cross-browser, modern APIs, auto-wait | Python, JS/TS, .NET, Java |
| Cypress | Component testing, fast feedback | JavaScript/TypeScript |
| Selenium | Legacy projects, widest browser support | All major languages |

When starting fresh, prefer Playwright — it has the best auto-wait behavior and cross-browser support.

## Step 2: Project Structure

```
tests/
+-- e2e/
|   +-- pages/              # Page object models
|   |   +-- login_page.py
|   |   +-- dashboard_page.py
|   +-- flows/              # Multi-page user flows
|   |   +-- test_signup_flow.py
|   |   +-- test_checkout_flow.py
|   +-- conftest.py         # Browser fixtures, base URL config
```

Separate e2e tests from unit tests. E2e tests are slow — mark them accordingly (e.g., `@pytest.mark.slow` or a dedicated test directory).

## Step 3: Page Object Model

Encapsulate page interactions behind a clean API. Tests should read like user stories, not DOM manipulation scripts. Create a page object for each major page or component your tests interact with. You're done with this step when at least one e2e test runs end-to-end through page objects without direct selector manipulation in the test body.

```python
# pages/login_page.py
class LoginPage:
    def __init__(self, page):
        self.page = page

    async def navigate(self):
        await self.page.goto("/login")

    async def login(self, email: str, password: str):
        await self.page.fill("[data-testid=email]", email)
        await self.page.fill("[data-testid=password]", password)
        await self.page.click("[data-testid=submit]")

    async def get_error_message(self) -> str:
        return await self.page.text_content("[data-testid=error]")
```

```python
# flows/test_login_flow.py
async def test_login_with_valid_credentials(login_page, dashboard_page):
    await login_page.navigate()
    await login_page.login("user@example.com", "valid-password")
    assert await dashboard_page.is_visible()

async def test_login_with_invalid_password(login_page):
    await login_page.navigate()
    await login_page.login("user@example.com", "wrong-password")
    assert "Invalid credentials" in await login_page.get_error_message()
```

### Selector Strategy

Prefer selectors in this order (most stable to least stable):

1. `data-testid` attributes — immune to styling and text changes
2. ARIA roles — `role=button`, `role=navigation`
3. Text content — `text="Submit"` (breaks on i18n changes)
4. CSS classes — fragile, break on styling refactors
5. XPath — last resort, unreadable and brittle

Add `data-testid` attributes to key interactive elements in the application code. This is a one-time investment that makes all e2e tests more stable.

## Step 4: Fixtures and Configuration

```python
# conftest.py
import pytest
from playwright.async_api import async_playwright

@pytest.fixture(scope="session")
def base_url():
    """Override with --base-url flag or BASE_URL env var."""
    import os
    return os.environ.get("BASE_URL", "http://localhost:3000")

@pytest.fixture
async def browser():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        yield browser
        await browser.close()

@pytest.fixture
async def page(browser, base_url):
    context = await browser.new_context(base_url=base_url)
    page = await context.new_page()
    yield page
    await context.close()

@pytest.fixture
async def authenticated_page(browser, base_url):
    """Page with a logged-in user session. Adapt login flow to your app."""
    context = await browser.new_context(base_url=base_url)
    page = await context.new_page()
    await page.goto("/login")
    await page.fill("[data-testid=email]", "test@example.com")
    await page.fill("[data-testid=password]", "test-password")
    await page.click("[data-testid=submit]")
    await page.wait_for_url("**/dashboard")
    yield page
    await context.close()
```

### Environment Configuration

| Environment | Base URL | Notes |
|-------------|----------|-------|
| Local dev | `http://localhost:3000` | Default, fastest feedback |
| CI | `http://localhost:3000` | Start app in CI before tests |
| Staging | `https://staging.example.com` | Run subset of critical paths |
| Production | `https://example.com` | Smoke tests only, read-only flows |

Never run destructive tests (account deletion, data mutation) against production.

## Step 5: Wait Strategies

The #1 source of flaky browser tests is improper waiting. Never use `sleep()`.

```python
# BAD — arbitrary sleep
await asyncio.sleep(2)
await page.click("#submit")

# GOOD — wait for specific condition
await page.wait_for_selector("[data-testid=submit]", state="visible")
await page.click("[data-testid=submit]")

# GOOD — wait for navigation
async with page.expect_navigation():
    await page.click("[data-testid=submit]")

# GOOD — wait for network idle (after SPA transitions)
await page.wait_for_load_state("networkidle")
```

Playwright auto-waits for elements by default — use its built-in waiting rather than adding manual waits.

## Step 6: Visual Debugging Workflow

When a test fails or a UI issue is reported:

### With MCP Tools (Claude can see the browser)

1. Claude navigates to the failing page via Playwright MCP
2. Claude takes a screenshot and inspects the DOM
3. Claude reads console errors directly
4. Claude identifies the issue and proposes a fix

### Without MCP Tools (manual workflow)

1. Take a screenshot of the issue and share it with Claude
2. Copy any browser console errors
3. Share the relevant HTML/component code
4. Claude diagnoses from the visual + code context

### Screenshot on Failure (automated)

```python
@pytest.fixture(autouse=True)
async def screenshot_on_failure(request, page):
    yield
    if request.node.rep_call and request.node.rep_call.failed:
        name = request.node.name.replace("/", "_")
        await page.screenshot(path=f"logs/screenshots/{name}.png")
```

Save failure screenshots to `logs/` (gitignored). This gives you a visual record of every failure without manual intervention.

## Step 7: Product Verification Flows

For critical business paths, build reusable verification skills that can run as smoke tests:

```python
# flows/test_critical_paths.py
"""
Critical path smoke tests. Run these:
- Before every deploy to production
- After infrastructure changes
- As the first check when "something is wrong"
"""
import time

async def test_signup_flow(page, base_url):
    """New user can sign up, verify email, and land on dashboard."""
    await page.goto(f"{base_url}/signup")
    await page.fill("[data-testid=email]", f"test+{int(time.time())}@example.com")
    await page.fill("[data-testid=password]", "SecurePass123!")
    await page.click("[data-testid=submit]")
    await page.wait_for_url("**/dashboard")
    assert await page.is_visible("[data-testid=welcome-banner]")

async def test_checkout_flow(page, authenticated_page):
    """Authenticated user can add item to cart and reach payment."""
    await authenticated_page.goto("/products")
    await authenticated_page.click("[data-testid=add-to-cart]:first-of-type")
    await authenticated_page.click("[data-testid=cart-icon]")
    await authenticated_page.click("[data-testid=checkout]")
    assert await authenticated_page.is_visible("[data-testid=payment-form]")
```

These are your highest-value tests. Invest time in making them stable and fast.

## Gotchas

- **Flaky selectors**: Tests that use CSS classes or DOM structure break on every styling change. Use `data-testid` attributes exclusively for test selectors. If the app doesn't have them, add them — it's the single highest-ROI investment for e2e stability.
- **Testing against live APIs**: E2e tests that hit real external APIs (payment processors, email services) are slow, flaky, and expensive. Mock external services at the network level or use sandbox/test modes.
- **Headless/headed behavior differences**: Some UI behaviors (hover states, focus management, file upload dialogs) work differently in headless mode. If a test passes headed but fails headless, the test is likely relying on a visual trigger — switch to a programmatic interaction.
- **Ignoring console errors**: A passing test with console errors is a false positive. Check `page.on("console")` and `page.on("pageerror")` for unexpected errors — these often indicate real bugs that the happy path doesn't surface.
- **Slow test suites killing feedback loops**: E2e tests are inherently slower than unit tests. Run only critical path tests in CI on every PR; run the full suite on a schedule (nightly) or before deploys.

## Completion

Report:
- Number of e2e tests written or updated
- Critical paths covered (list the flows)
- Flaky tests identified and stabilized
- MCP tools configured (if applicable)
- Console errors found during test runs
