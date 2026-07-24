"""${project_name} — ${description}

The public API of this package is exactly what is re-exported here. Anything
reachable only through a submodule is an implementation detail and may change
without a major version bump.
"""

from ${package_name}.config import ENV_PREFIX, LibraryConfig
from ${package_name}.core import Registry, RegistryError, component_registry

__version__ = "0.1.0"

__all__ = [
    "ENV_PREFIX",
    "LibraryConfig",
    "Registry",
    "RegistryError",
    "__version__",
    "component_registry",
]
