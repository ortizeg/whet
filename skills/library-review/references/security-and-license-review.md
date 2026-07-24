# Security and License Review

Scope: criteria 5 and 6 of the evaluation checklist — license compatibility and security
track record — plus the CI automation and recurring audit schedule that keep them true
after adoption.

## Contents

- [5. License Compatibility](#5-license-compatibility)
- [6. Security History](#6-security-history)
- [Automated Checks in CI](#automated-checks-in-ci)
- [Regular Review Schedule](#regular-review-schedule)

## 5. License Compatibility

Verify the license is compatible with your project's license.

| License | Compatible with MIT | Compatible with Apache 2.0 | Notes |
|---------|--------------------|----|-------|
| MIT | Yes | Yes | Most permissive |
| Apache 2.0 | Yes | Yes | Patent grant |
| BSD 2/3 | Yes | Yes | Permissive |
| LGPL | Yes (dynamic linking) | Yes (dynamic linking) | Must not modify |
| GPL | No | No | Copyleft, viral |
| AGPL | No | No | Copyleft, network use |

```bash
# Check license via GitHub API
gh api repos/OWNER/REPO --jq '.license.spdx_id'

# Check license via pip
pip show PACKAGE_NAME | grep License
```

## 6. Security History

Review the library's security track record.

```bash
# Check for known vulnerabilities
pip-audit --requirement requirements.txt

# Check GitHub security advisories
gh api repos/OWNER/REPO/security-advisories --jq '.[].summary'

# Check the Safety database
safety check --full-report
```

| Signal | Green | Yellow | Red |
|--------|-------|--------|-----|
| CVE history | None or quickly patched | Few, patched within weeks | Multiple, slow patches |
| Dependency audit | Clean | Minor issues | Critical vulnerabilities |
| Security policy | Published SECURITY.md | Informal process | None |

## Automated Checks in CI

Integrate dependency review into CI so a risky transitive dependency fails the PR rather
than landing unnoticed.

```yaml
# .github/workflows/dependency-review.yml
name: Dependency Review

on:
  pull_request:

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Dependency Review
        uses: actions/dependency-review-action@v4
        with:
          fail-on-severity: high
          deny-licenses: GPL-3.0, AGPL-3.0

      - name: Audit dependencies
        run: |
          pip install pip-audit
          pip-audit --requirement requirements.txt
```

## Regular Review Schedule

Dependencies should be reviewed periodically, not just at adoption:

```bash
# Monthly: Check for outdated packages
pip list --outdated

# Monthly: Run security audit
pip-audit

# Quarterly: Review dependency tree for unused packages
pipdeptree --warn silence | grep -E "^\w"

# Annually: Full evaluation of all dependencies against the checklist
```
