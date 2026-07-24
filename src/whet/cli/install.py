"""Install command — bulk install skills and settings to a platform."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from whet.adapters.base import PlatformAdapter
from whet.cli import app
from whet.core.config import Platform, WhetConfig
from whet.core.manifest import find_orphans, read_manifest, write_manifest
from whet.core.skill import Skill
from whet.registry.loader import discover_skills

console = Console()


@app.command()
def install(
    scope_global: bool = typer.Option(False, "--global", "-g", help="Install to global directory."),
    scope_local: bool = typer.Option(False, "--local", "-l", help="Install to local project."),
    category: str | None = typer.Option(None, "--category", "--cat", help="Only install category."),
    include_extras: bool = typer.Option(
        False, "--include-extras", help="Also install opt-in 'extra' tier skills."
    ),
    prune: bool = typer.Option(
        False, "--prune", help="Also remove whet-installed skills that no longer exist upstream."
    ),
    with_settings: bool = typer.Option(
        False, "--with-settings", "-s", help="Also apply settings template."
    ),
) -> None:
    """Install skills and optionally settings."""
    if not scope_global and not scope_local:
        scope_local = True

    cfg = _get_config()
    adapter = _get_adapter(cfg.target)
    paths = cfg.get_platform_paths()

    source_skills = discover_skills(cfg.skills_dir)
    # Every skill that still exists upstream, before any filtering. Prune compares
    # against this so a skill merely skipped by --category/--include-extras is never
    # mistaken for one that was deleted.
    available = {s.name for s in source_skills}

    all_skills = source_skills
    if category:
        all_skills = [s for s in all_skills if s.category == category]

    if not include_extras:
        skipped = [s for s in all_skills if s.tier == "extra"]
        all_skills = [s for s in all_skills if s.tier != "extra"]
        if skipped:
            names = ", ".join(s.name for s in skipped)
            console.print(
                f"[dim]Skipping {len(skipped)} extra skills ({names}). "
                f"Use --include-extras to install them.[/dim]"
            )

    if not all_skills:
        console.print("[yellow]No skills found to install.[/yellow]")
        raise typer.Exit(code=1)

    if scope_global:
        _install_to(adapter, all_skills, paths.global_dir, "global", available, prune)
        if with_settings:
            _apply_settings(cfg.target.value, "global")

    if scope_local:
        _install_to(adapter, all_skills, paths.local_dir, "local", available, prune)
        if with_settings:
            _apply_settings(cfg.target.value, "local")


def _install_to(
    adapter: PlatformAdapter,
    items: list[Skill],
    target_dir: Path,
    scope_label: str,
    available: set[str],
    prune: bool,
) -> None:
    """Install skills to a target directory, optionally pruning deleted ones."""
    console.print(f"\n[bold]Installing {len(items)} skills ({scope_label})...[/bold]")

    for item in items:
        adapter.install_skill(item, target_dir)
        console.print(f"  [green]✓[/green] {item.name}")

    console.print(f"\n[bold green]✓ Installed {len(items)} skills to {target_dir}[/bold green]")

    previously_installed = read_manifest(target_dir)
    orphans = find_orphans(previously_installed, available)

    if prune and orphans:
        console.print(f"\n[bold]Pruning {len(orphans)} removed skills...[/bold]")
        for name in orphans:
            if adapter.remove_skill(name, target_dir):
                console.print(f"  [yellow]−[/yellow] {name}")
    elif orphans:
        names = ", ".join(orphans)
        console.print(
            f"[dim]{len(orphans)} installed skills no longer exist upstream ({names}). "
            f"Use --prune to remove them.[/dim]"
        )

    # Record ownership: everything whet has installed here, minus anything just pruned.
    owned = set(previously_installed) | {item.name for item in items}
    if prune:
        owned -= set(orphans)
    write_manifest(target_dir, sorted(owned))


def _apply_settings(platform: str, scope: str) -> None:
    """Apply settings template with merge support."""
    from whet.settings.engine import (
        get_settings_target,
        get_template_path,
        load_existing,
        load_template,
        merge_settings,
        write_settings,
    )

    template_path = get_template_path(platform)
    settings_path = get_settings_target(platform, scope)
    template = load_template(template_path)
    existing = load_existing(settings_path)

    if existing:
        merged = merge_settings(existing, template)
        write_settings(settings_path, merged)
        console.print(f"\n[bold green]✓ Merged settings into {settings_path}[/bold green]")
    else:
        write_settings(settings_path, template)
        console.print(f"\n[bold green]✓ Applied settings to {settings_path}[/bold green]")


def _get_config() -> WhetConfig:
    return WhetConfig.load()


def _get_adapter(platform: Platform) -> PlatformAdapter:
    from whet.adapters.antigravity import AntigravityAdapter
    from whet.adapters.claude import ClaudeAdapter
    from whet.adapters.copilot import CopilotAdapter
    from whet.adapters.cursor import CursorAdapter

    adapters: dict[Platform, type[PlatformAdapter]] = {
        Platform.CLAUDE: ClaudeAdapter,
        Platform.ANTIGRAVITY: AntigravityAdapter,
        Platform.CURSOR: CursorAdapter,
        Platform.COPILOT: CopilotAdapter,
    }
    return adapters[platform]()
