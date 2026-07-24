"""Doctor command — health check for whet installation."""

from __future__ import annotations

import typer
from rich.console import Console

from whet.adapters.detect import detect_platform
from whet.cli import app
from whet.core.config import PLATFORM_PATHS, WhetConfig
from whet.registry.loader import discover_skills

console = Console()


def _check_mark(ok: bool) -> str:
    return "[green]✓[/green]" if ok else "[red]✗[/red]"


def _warn_mark() -> str:
    return "[yellow]![/yellow]"


@app.command()
def doctor() -> None:
    """Run health checks on your whet installation."""
    console.print("[bold]whet doctor[/bold]\n")
    issues = 0
    warnings = 0

    cfg = WhetConfig.load()

    # Check 1: Skills directory
    if cfg.skills_dir.is_dir():
        skills = discover_skills(cfg.skills_dir)
        n = len(skills)
        console.print(f"  {_check_mark(True)} Skills directory: {cfg.skills_dir} ({n} skills)")
    else:
        console.print(f"  {_check_mark(False)} Skills directory not found: {cfg.skills_dir}")
        issues += 1
        skills = []

    # Check 2: Archetypes directory
    if cfg.archetypes_dir.is_dir():
        archetypes = [
            d.name
            for d in cfg.archetypes_dir.iterdir()
            if d.is_dir() and (d / "README.md").exists()
        ]
        n_arch = len(archetypes)
        console.print(
            f"  {_check_mark(True)} Archetypes directory: "
            f"{cfg.archetypes_dir} ({n_arch} archetypes)"
        )
    else:
        console.print(f"  {_warn_mark()} Archetypes directory not found: {cfg.archetypes_dir}")
        warnings += 1

    # Check 3: Platform detection
    detected = detect_platform()
    if detected:
        console.print(f"  {_check_mark(True)} Platform detected: {detected.value}")
    else:
        console.print(f"  {_warn_mark()} No platform auto-detected in current directory")
        warnings += 1

    # Check 4: Installed skills
    if detected:
        from whet.cli.skills import _get_adapter

        adapter = _get_adapter(detected)
        paths = PLATFORM_PATHS[detected]

        lc = len(adapter.list_installed(paths.local_dir))
        gc = len(adapter.list_installed(paths.global_dir))

        if lc > 0 or gc > 0:
            console.print(f"  {_check_mark(True)} Installed: {lc} local, {gc} global")
        else:
            console.print(f"  {_warn_mark()} No skills installed for {detected.value}")
            warnings += 1

    # Check 5: SKILL.md frontmatter
    if skills:
        missing_fm = [s for s in skills if not s.description]
        if missing_fm:
            n_fm = len(missing_fm)
            console.print(f"  {_warn_mark()} {n_fm} skills missing YAML frontmatter")
            for s in missing_fm[:5]:
                console.print(f"    - {s.name}")
            warnings += 1
        else:
            console.print(f"  {_check_mark(True)} All skills have YAML frontmatter")

    # Check 6: skill.toml files
    if skills:
        missing_toml = [s for s in skills if not s.has_toml]
        if missing_toml:
            n_toml = len(missing_toml)
            console.print(f"  {_warn_mark()} {n_toml} skills missing skill.toml")
            for s in missing_toml[:5]:
                console.print(f"    - {s.name}")
            warnings += 1
        else:
            console.print(f"  {_check_mark(True)} All skills have skill.toml metadata")

    # Check 7: Settings template
    from whet.settings.engine import get_settings_target, get_template_path

    plat = detected.value if detected else cfg.target.value
    template_path = get_template_path(plat)
    if template_path.exists():
        console.print(f"  {_check_mark(True)} Settings template: {template_path.name}")

        # Check if settings are applied
        local_target = get_settings_target(plat, "local")
        if local_target.exists():
            console.print(f"  {_check_mark(True)} Local settings: {local_target}")
        else:
            console.print(
                f"  {_warn_mark()} No local settings. Run: [bold]whet settings apply[/bold]"
            )
            warnings += 1
    else:
        console.print(f"  {_warn_mark()} No settings template for {plat}")
        warnings += 1

    # Check 8: Dependency consistency
    if skills:
        available = {s.name for s in skills}
        broken_deps = []
        for skill in skills:
            if skill.metadata and skill.metadata.dependencies.requires:
                for dep in skill.metadata.dependencies.requires:
                    if dep not in available:
                        broken_deps.append((skill.name, dep))

        if broken_deps:
            n_bd = len(broken_deps)
            console.print(f"  {_warn_mark()} {n_bd} broken dependency reference(s)")
            for skill_name, dep in broken_deps[:5]:
                console.print(f"    - {skill_name} requires '{dep}' (not found)")
            warnings += 1
        else:
            console.print(f"  {_check_mark(True)} All skill dependencies resolve")

    # Summary
    console.print()
    if issues:
        console.print(f"[red]{issues} error(s), {warnings} warning(s)[/red]")
        raise typer.Exit(code=1)
    elif warnings:
        console.print(f"[yellow]{warnings} warning(s), 0 errors[/yellow]")
    else:
        console.print("[bold green]All checks passed.[/bold green]")
