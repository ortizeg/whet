# ${project_name}

${description}

A small, typed, dependency-light Python library. Ships `py.typed`, so consumers
get the type hints. Built with [hatchling](https://hatch.pypa.io/), developed
with [Pixi](https://pixi.sh/), published to PyPI.

## Install

```bash
pip install ${project_slug}
```

## Quick start

```python
from ${package_name} import LibraryConfig, Registry

encoders: Registry[str] = Registry("encoders")


@encoders.register("upper")
def upper_encoder(text: str) -> str:
    return text.upper()


encoders.names()  # ["upper"]
encoders.create("UPPER", "hi")  # "HI" — keys are case-insensitive

config = LibraryConfig.from_env()
config.max_items  # 1024 unless overridden in the environment
```

## Public API

Everything re-exported from `${package_name}/__init__.py` is public and covered
by semantic versioning; anything else is an implementation detail.

| Name | What it is |
|---|---|
| `Registry` | A typed, name-keyed registry of factories |
| `RegistryError` | Raised on unknown or duplicate registry keys |
| `component_registry` | A package-level `Registry` for plugins to populate |
| `LibraryConfig` | Frozen Pydantic V2 settings model with `from_env()` |
| `ENV_PREFIX` | Prefix of the environment variables `LibraryConfig` reads |
| `__version__` | The installed version string |

## CLI

```bash
${project_slug} --version
${project_slug} info          # version + resolved config as JSON on stdout
${project_slug} components    # keys in the package-level registry
```

## Development

Pixi is the canonical environment manager for this project.

```bash
pixi install                  # create the environment
pixi add <name>               # add a conda dependency
pixi add --pypi <name>        # add a PyPI dependency
```

Runtime dependencies must also be listed in `[project.dependencies]` in
`pyproject.toml` — that table is what installs alongside the published wheel.

Inside the environment (`pixi shell`, or via `pixi run <task>`):

```bash
pytest                        # test suite
pytest --cov                  # with coverage
ruff check .                  # lint
ruff format .                 # format
mypy src/ --strict            # type check
```

Pixi tasks wrap the same commands: `pixi run test`, `pixi run lint`,
`pixi run typecheck`, `pixi run quality`.

Install the git hooks once with `pre-commit install`.

## Documentation

```bash
pixi run -e docs docs-serve   # live reload at http://127.0.0.1:8000
pixi run -e docs docs-build   # strict build into site/
```

`docs.yml` deploys to GitHub Pages on every push to `main`.

## Logging

Logging goes through [Loguru](https://loguru.readthedocs.io/). The library
installs **no sink** — an imported library has no business rerouting your
application's logs. Configure Loguru in your application and these records will
follow. Only `cli.py` adds a sink, and only when the CLI actually runs.

## Layout

```
${project_slug}/
├── .github/workflows/{ci,docs,publish}.yml
├── docs/{index,getting-started,api}.md
├── examples/basic_usage.py
├── src/${package_name}/
│   ├── __init__.py      # the public API
│   ├── cli.py           # console script
│   ├── config.py        # LibraryConfig
│   ├── core.py          # Registry
│   └── py.typed
├── tests/
├── CHANGELOG.md
├── LICENSE.txt
├── mkdocs.yml
├── pixi.toml
└── pyproject.toml
```

## Releasing

1. Move the `[Unreleased]` entries in `CHANGELOG.md` under a new version heading.
2. Bump `__version__` in `src/${package_name}/__init__.py`. It is the single
   source of truth — `[tool.hatch.version]` reads it at build time.
3. Tag and push:

```bash
git tag v0.2.0
git push origin v0.2.0
```

4. Publish the GitHub Release. The `publish` workflow builds with
   `python -m build`, verifies with `twine check`, and uploads to PyPI using
   trusted publishing (no stored API token).

To build and upload by hand:

```bash
python -m build
twine check dist/*
twine upload --repository testpypi dist/*   # staging
twine upload dist/*                         # production
```

## License

MIT — see [LICENSE.txt](LICENSE.txt).
