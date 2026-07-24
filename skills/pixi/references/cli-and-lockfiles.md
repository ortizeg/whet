# Pixi CLI and Lock Files

Scope: the day-to-day pixi command reference and the lock-file policy for applications
versus libraries.

## Common Commands

```bash
# Install dependencies (creates/updates lock file)
pixi install

# Add a conda dependency
pixi add numpy ">=1.26"

# Add a PyPI dependency
pixi add --pypi torch ">=2.2"

# Add a dependency to a specific feature
pixi add --feature dev pytest ">=7.4"

# Remove a dependency
pixi remove numpy

# List all installed packages
pixi list

# Show environment info
pixi info

# Run a task
pixi run test
pixi run lint
pixi run quality

# Run a shell command in the environment
pixi shell

# Clean the environment (remove .pixi/)
pixi clean
```

## Lock File Management

Pixi generates `pixi.lock` which pins exact versions of all dependencies. This file
should be committed to version control for applications but excluded for libraries.

```gitignore
# For applications (training projects, services): commit pixi.lock
# Do NOT add pixi.lock to .gitignore

# For libraries (published packages): ignore pixi.lock
pixi.lock
```

In CI, run `pixi install` (which honours the committed lock file) rather than resolving
fresh, so the pipeline installs byte-identical environments to developer machines.
