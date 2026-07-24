# Project Structure and Archetype Extensions

Scope: the src-layout rationale, the package and test scaffolding files, template-variable
derivation rules, and the directories each archetype adds on top of the standard layout.

## Contents

- [Why src-layout?](#why-src-layout)
- [Scaffolding Files](#scaffolding-files)
- [Variable Derivation Rules](#variable-derivation-rules)
- [Archetype-Specific Extensions](#archetype-specific-extensions)

## Why src-layout?

The `src/` layout is mandatory across all archetypes. It prevents accidental imports of the
local package during testing and enforces proper installation. The `py.typed` marker
enables downstream consumers to benefit from your type annotations.

Without `src/`, `pytest` imports the package directly from the working directory, so tests
can pass against code that was never installed correctly — a class of failure that only
appears after publishing.

## Scaffolding Files

```python
# src/{{package_name}}/__init__.py
"""{{description}}."""

__version__ = "{{version}}"
```

```python
# tests/conftest.py
"""Shared test fixtures for {{project_name}}."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_data_dir(tmp_path):
    """Provide a temporary directory pre-populated with test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir
```

`src/{{package_name}}/py.typed` is an empty file — the PEP 561 marker that tells type
checkers this package ships inline annotations.

## Variable Derivation Rules

```python
# project_slug is derived from project_name
project_slug = project_name.lower().replace(" ", "-").replace("_", "-")
# Example: "Face Detection System" -> "face-detection-system"

# package_name is derived from project_slug
package_name = project_slug.replace("-", "_")
# Example: "face-detection-system" -> "face_detection_system"
```

## Archetype-Specific Extensions

Each archetype adds directories and files on top of the standard structure. Here is what
each archetype adds:

### pytorch-training-project

```
configs/
├── experiment/
│   └── default.yaml
├── model/
│   └── default.yaml
└── data/
    └── default.yaml
src/{{package_name}}/
├── models/
│   └── __init__.py
├── data/
│   └── __init__.py
├── transforms/
│   └── __init__.py
└── train.py
scripts/
└── train.sh
```

### cv-inference-service

```
src/{{package_name}}/
├── api/
│   ├── __init__.py
│   └── routes.py
├── models/
│   └── __init__.py
└── inference/
    └── __init__.py
docker/
├── Dockerfile
└── docker-compose.yml
```

### research-notebook

```
notebooks/
├── 01_exploration.ipynb
└── 02_experiments.ipynb
data/
├── raw/.gitkeep
└── processed/.gitkeep
```

The remaining archetypes — `library-package`, `data-processing-pipeline`, and `model-zoo` —
extend the standard structure with their own directories; consult the archetype's own
`README.md` and `template/` directory for the exact tree.
