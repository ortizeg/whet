# Verification and Error Handling

Scope: the script that verifies a freshly generated project, and what to do when a step of
initialization fails.

## Verification Script

```bash
# Run after project generation to verify everything works
cd {{project_slug}}

# Check no template variables remain
grep -r '{{' . --include='*.py' --include='*.toml' --include='*.yaml' --include='*.md' && echo "FAIL: Template variables remain" || echo "PASS: All variables substituted"

# Install and verify
pixi install
pixi run lint
pixi run typecheck
pixi run test

# Initialize git
git init
pixi run pre-commit install
git add .
git commit -m "Initial project from {{archetype}} archetype"
```

## Error Handling

If any step fails during initialization:

1. **Missing user input**: Re-prompt with sensible defaults
2. **Invalid archetype**: Show list of valid archetypes and ask again
3. **Template substitution failure**: Report which files still contain `{{` markers
4. **Pixi install failure**: Check pixi.toml channels and dependency compatibility
5. **Lint/typecheck failure**: Fix generated code before completing initialization

The master skill never leaves the project in a broken state. If initialization cannot be
completed cleanly, it reports exactly what went wrong and what manual steps are needed.
