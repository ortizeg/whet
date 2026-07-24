# Technology Radar Classification

Scope: the full Adopt / Trial / Assess / Hold criteria and the written review template used
to propose a new dependency.

## Contents

- [Adopt](#adopt)
- [Trial](#trial)
- [Assess](#assess)
- [Hold](#hold)
- [Review Template](#review-template)

## Adopt

The library is proven and recommended for use across the project.

**Criteria:**
- All checklist items are green or yellow.
- Used in production by the team or well-known organizations.
- Active maintenance with responsive maintainers.
- Good type support and documentation.
- Compatible license.

**Examples:** PyTorch, Pydantic, Ruff, pytest, Pillow, NumPy.

## Trial

The library shows promise and should be tested in a non-critical part of the project.

**Criteria:**
- Most checklist items are green or yellow.
- Relatively new but gaining traction quickly.
- Team member willing to champion and maintain the integration.
- Wrapped behind a project interface to limit exposure.

**Examples:** A new image augmentation library, a faster data loader, a specialized metric library.

## Assess

The library is interesting but needs further evaluation before any use.

**Criteria:**
- Mixed signals on the checklist.
- No team member has hands-on experience.
- Potential overlap with existing tools.

**Action:** Assign a team member to create a proof-of-concept in a branch.

## Hold

The library should not be adopted at this time.

**Criteria:**
- Multiple red signals on the checklist.
- Abandoned or poorly maintained.
- License incompatibility.
- Better alternatives exist.
- Significant security concerns.

## Review Template

Use this template when proposing a new dependency:

```markdown
## Library Review: [Library Name]

### Basic Information
- **Name**: [library-name]
- **Version**: [x.y.z]
- **License**: [MIT/Apache/etc.]
- **PyPI**: [link]
- **Repository**: [link]
- **Documentation**: [link]

### Purpose
[Why do we need this library? What problem does it solve?]

### Alternatives Considered
| Library | Pros | Cons |
|---------|------|------|

### Evaluation Checklist
Score each criterion above (maintenance, community, docs, types, license,
security, performance, integration) green/yellow/red.

### Decision
- [ ] Adopt  - [ ] Trial  - [ ] Assess  - [ ] Hold

### Wrapping Plan
[How will this library be wrapped behind a project interface?]

### Reviewer
[Name, Date]
```
