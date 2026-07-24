# Ruff Rule Selection

Scope: why each Ruff rule family in the standard config is enabled, and how the
per-file-ignore exceptions are scoped.

## Rule selection rationale

| Rule | Why It Matters |
|------|---------------|
| `E` | Basic Python style errors. Catches obvious formatting issues. |
| `F` | Unused imports and variables. Dead code creates confusion. |
| `I` | Import ordering. Consistent imports improve readability. |
| `N` | Naming conventions. `snake_case` for functions, `PascalCase` for classes. |
| `UP` | Modern Python syntax. Use `dict` not `Dict`, `X \| Y` not `Union[X, Y]`. |
| `S` | Security issues. SQL injection, hardcoded passwords, unsafe YAML loading. |
| `B` | Common bugs. Mutable default arguments, unused loop variables. |
| `A` | Shadowing builtins. Never name a variable `input`, `list`, `type`, etc. |
| `C4` | Comprehension style. Use list comprehensions over `list(map(...))`. |
| `T20` | No print statements. Use `logging` instead. Print in prod code is a bug. |
| `SIM` | Simplification. Collapse nested ifs, use ternary where clearer. |
| `TCH` | Type-checking imports. Move type-only imports behind `TYPE_CHECKING`. |
| `RUF` | Ruff-specific. Catches additional patterns other tools miss. |
| `PTH` | Pathlib. Use `Path` not `os.path`. Modern Python file handling. |
| `ERA` | Dead code. Commented-out code should be deleted, not committed. |

## Scoped exceptions

Two rules are ignored globally, both for a specific reason:

- `E501` (line length) — the formatter owns line length; leaving the lint rule on
  double-reports the same issue.
- `S101` (`assert`) — disallowed in source, but re-allowed for tests via per-file-ignores.

Everything else is narrowed with `per-file-ignores` rather than turned off project-wide:

```toml
[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["S101", "S105", "S106"]   # allow assert and hardcoded passwords in tests
"scripts/**/*.py" = ["T20"]                   # allow print in scripts
"notebooks/**/*.py" = ["T20", "E402"]         # allow print and late imports in notebooks
```

The `pep8-naming` section teaches `N` about Pydantic's validator decorators so
`field_validator`/`model_validator` methods are not flagged as badly named classmethods:

```toml
[tool.ruff.lint.pep8-naming]
classmethod-decorators = ["pydantic.field_validator", "pydantic.model_validator"]
```

## Running Ruff

```bash
ruff check .            # lint
ruff check . --fix      # lint + auto-fix
ruff format .           # format (add --check or --diff for CI)
```
