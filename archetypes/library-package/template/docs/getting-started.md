# Getting Started

## Install for use

```bash
pip install ${project_slug}
```

## Install for development

The canonical development environment is [Pixi](https://pixi.sh/).

```bash
pixi install                 # create the environment from pixi.toml
pixi run test                # pytest
pixi run quality             # lint + format-check + typecheck + test
```

Add a dependency with `pixi add <name>` (conda) or `pixi add --pypi <name>`, and
mirror runtime dependencies into `[project.dependencies]` in `pyproject.toml` —
that table is what users of the published wheel actually install.

## The registry

`Registry` turns a configuration string into an object without the call site
importing the concrete implementation.

```python
from ${package_name} import Registry, RegistryError

encoders: Registry[str] = Registry("encoders")


@encoders.register("upper")
def upper_encoder(text: str) -> str:
    return text.upper()


encoders.names()  # ["upper"]
encoders.create("UPPER", "hi")  # "HI" — keys are case-insensitive

try:
    encoders.create("missing", "hi")
except RegistryError as exc:
    print(exc)  # lists the known keys
```

Registering the same key twice raises `RegistryError` rather than silently
shadowing the first entry. Pass `override=True` when replacement is intended.

## Configuration

`LibraryConfig` is a frozen Pydantic V2 model. `from_env()` reads each field
from an environment variable named `ENV_PREFIX + FIELD_NAME.upper()`, where
`ENV_PREFIX` is the package name upper-cased with a trailing underscore. Run
`${project_slug} info` to print the exact names rather than guessing:

```console
$ ${project_slug} info
...
  "env_vars": {
    "log_level": "<PREFIX>LOG_LEVEL",
    "max_items": "<PREFIX>MAX_ITEMS",
    "strict": "<PREFIX>STRICT"
  }
```

Set them like any other environment variable, then:

```python
from ${package_name} import ENV_PREFIX, LibraryConfig

ENV_PREFIX  # e.g. "MYLIB_"
config = LibraryConfig.from_env()
config.log_level  # "DEBUG"
config.max_items  # 64
```

Unknown keys are rejected, so a typo fails loudly at load time instead of being
silently ignored.

## The CLI

```bash
${project_slug} --version
${project_slug} info          # version + resolved config as JSON
${project_slug} components    # keys in the package-level registry
```

JSON payloads go to stdout; log records go to stderr, so `${project_slug} info |
jq` works.

## Releasing

1. Move the `[Unreleased]` entries in `CHANGELOG.md` under a new version heading.
2. Bump `__version__` in `src/${package_name}/__init__.py` — it is the single
   source of truth, read at build time by `[tool.hatch.version]`.
3. Tag and push:

```bash
git tag v0.2.0
git push origin v0.2.0
```

4. Publish the GitHub Release; the `publish` workflow builds the sdist and wheel
   and uploads them to PyPI via trusted publishing.
