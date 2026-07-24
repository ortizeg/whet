"""Allow ``python -m ${package_name} ...``."""

from ${package_name}.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
