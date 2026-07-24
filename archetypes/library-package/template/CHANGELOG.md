# Changelog

All notable changes to ${project_name} are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The public API is defined as everything re-exported from
`src/${package_name}/__init__.py`. Changes to anything else are not breaking
changes.

## [Unreleased]

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [0.1.0] - 1970-01-01

### Added

- `Registry` — a typed, name-keyed registry of factories, with a decorator API,
  case-insensitive keys, and duplicate-registration protection.
- `RegistryError` — raised on unknown or duplicate keys; subclasses `KeyError`.
- `component_registry` — a package-level registry downstream code can populate.
- `LibraryConfig` — a frozen Pydantic V2 settings model with `from_env()`
  loading from environment variables prefixed with the upper-cased package name.
- `${project_slug}` console script with `info` and `components` subcommands.
- `py.typed` marker so downstream consumers get the shipped type hints.
