# ${project_name}

${description}

`${project_slug}` is a small, typed, dependency-light Python library. It ships a
`py.typed` marker, so everything documented here is type-checked for downstream
consumers too.

## Install

```bash
pip install ${project_slug}
```

## The public API

Everything re-exported from `${package_name}/__init__.py` is public and covered
by semantic versioning:

| Name | What it is |
|---|---|
| `Registry` | A typed, name-keyed registry of factories |
| `RegistryError` | Raised on unknown or duplicate registry keys |
| `component_registry` | A package-level `Registry` for plugins to populate |
| `LibraryConfig` | Frozen Pydantic V2 settings model |
| `ENV_PREFIX` | Prefix for the environment variables `LibraryConfig` reads |
| `__version__` | The installed version string |

## Thirty-second tour

```python
from ${package_name} import LibraryConfig, Registry

encoders: Registry[str] = Registry("encoders")


@encoders.register("upper")
def upper_encoder(text: str) -> str:
    return text.upper()


encoders.create("upper", "hello")  # -> "HELLO"

config = LibraryConfig.from_env()
config.max_items  # -> 1024 unless overridden in the environment
```

## Logging

The library logs through [Loguru](https://loguru.readthedocs.io/) and never
installs a sink of its own — an imported library has no business rerouting your
application's logs. Configure Loguru in your application and the library's
records will follow.
