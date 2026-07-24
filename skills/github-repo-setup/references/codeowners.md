# CODEOWNERS

Scope: routing code review to the right team with a `.github/CODEOWNERS` file, and the
pattern syntax it accepts.

## Example CODEOWNERS

Define ownership for code review routing.

```
# .github/CODEOWNERS
# Default owner for everything
* @org/ml-team

# Specific ownership
/skills/         @org/ml-team
/agents/         @org/ml-team
/archetypes/     @org/ml-team
/docs/           @org/docs-team
/.github/        @org/devops-team
pixi.toml        @org/devops-team
pyproject.toml   @org/ml-team
```

Later rules win: the last matching pattern in the file determines the owner, so put the
broad `*` default first and narrow rules after it.

## CODEOWNERS Syntax

| Pattern | Meaning |
|---------|---------|
| `*` | Everything (default) |
| `/docs/` | Only the top-level `docs/` directory |
| `*.py` | All Python files anywhere |
| `/src/models/` | Specific subdirectory |
| `@user` | Individual GitHub user |
| `@org/team` | GitHub team |

CODEOWNERS only enforces review when branch protection sets
`require_code_owner_reviews: true`; otherwise it merely suggests reviewers.
