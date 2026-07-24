# PR and Issue Templates

Scope: the `.github/PULL_REQUEST_TEMPLATE.md` and the YAML issue forms for bug reports
and feature requests.

## Contents

- [Pull Request Template](#pull-request-template)
- [Bug Report Template](#bug-report-template)
- [Feature Request Template](#feature-request-template)

## Pull Request Template

Standardize PR descriptions to capture context for reviewers.

```markdown
<!-- .github/PULL_REQUEST_TEMPLATE.md -->

## Summary
<!-- Brief description of changes -->

## Changes
-

## Test Plan
- [ ] `pixi run test` passes
- [ ] `pixi run lint` passes
- [ ] `pixi run typecheck` passes
- [ ] `pixi run docs-build` succeeds

## Checklist
- [ ] Tests added/updated for new functionality
- [ ] Documentation updated if needed
- [ ] No new type errors introduced
```

## Bug Report Template

```yaml
# .github/ISSUE_TEMPLATE/bug_report.yml
name: Bug Report
description: Report a bug or unexpected behavior
labels: ["bug"]
body:
  - type: textarea
    id: description
    attributes:
      label: Description
      description: Clear description of the bug
    validations:
      required: true
  - type: textarea
    id: reproduction
    attributes:
      label: Steps to Reproduce
      description: Minimal steps to reproduce the issue
      value: |
        1. ...
        2. ...
        3. ...
    validations:
      required: true
  - type: textarea
    id: expected
    attributes:
      label: Expected Behavior
      description: What you expected to happen
    validations:
      required: true
  - type: input
    id: version
    attributes:
      label: Version
      description: Package version or commit hash
    validations:
      required: true
  - type: dropdown
    id: os
    attributes:
      label: Operating System
      options:
        - Linux
        - macOS
        - Windows
    validations:
      required: true
```

## Feature Request Template

```yaml
# .github/ISSUE_TEMPLATE/feature_request.yml
name: Feature Request
description: Suggest a new feature or improvement
labels: ["enhancement"]
body:
  - type: textarea
    id: problem
    attributes:
      label: Problem Statement
      description: What problem does this solve?
    validations:
      required: true
  - type: textarea
    id: solution
    attributes:
      label: Proposed Solution
      description: How should this work?
    validations:
      required: true
  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives Considered
      description: Other approaches you've considered
    validations:
      required: false
```
