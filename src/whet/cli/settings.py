"""Settings commands — generate, apply, and diff platform settings."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from whet.cli import app
from whet.core.config import Platform, WhetConfig
from whet.settings.engine import (
    diff_settings,
    get_settings_target,
    get_template_path,
    load_existing,
    load_template,
    merge_settings,
    write_settings,
)

console = Console()

settings_app = typer.Typer(
    name="settings",
    help="Manage platform settings (permissions, allowed commands).",
    no_args_is_help=True,
)
app.add_typer(settings_app)


@settings_app.command()
def generate(
    platform: str | None = typer.Option(  # noqa: UP007
        None, "--platform", "-p", help="Target platform."
    ),
) -> None:
    """Generate optimized settings for a platform."""
    cfg = WhetConfig.load()
    plat = platform or cfg.target.value

    try:
        Platform(plat)
    except ValueError as err:
        valid = ", ".join(p.value for p in Platform)
        console.print(f"[red]Invalid platform '{plat}'. Valid: {valid}[/red]")
        raise typer.Exit(code=1) from err

    template_path = get_template_path(plat)

    if not template_path.exists():
        console.print(f"[yellow]No settings template for '{plat}'.[/yellow]")
        console.print("[dim]Available templates:[/dim]")
        templates_dir = template_path.parent
        for f in sorted(templates_dir.glob("*.json")):
            console.print(f"  {f.stem}")
        raise typer.Exit(code=1)

    template = load_template(template_path)
    n = len(template.permissions.get("allow", []))

    console.print(f"[bold]Generated settings for {plat}[/bold]\n")
    console.print(f"  Permissions: {n} allow rules")
    console.print(f"  Template:    {template_path}")
    console.print("\n  Run [bold]whet settings diff[/bold] to review")
    console.print("  Run [bold]whet settings apply[/bold] to write")


@settings_app.command()
def diff(
    platform: str | None = typer.Option(  # noqa: UP007
        None, "--platform", "-p", help="Target platform."
    ),
    scope: str = typer.Option("local", "--scope", "-s", help="Scope: local or global."),
) -> None:
    """Preview changes between current settings and template."""
    cfg = WhetConfig.load()
    plat = platform or cfg.target.value

    template_path = get_template_path(plat)
    if not template_path.exists():
        console.print(f"[red]No template for '{plat}'.[/red]")
        raise typer.Exit(code=1)

    template = load_template(template_path)
    target_path = get_settings_target(plat, scope)
    existing = load_existing(target_path)

    new_entries, kept_entries, all_merged = diff_settings(existing, template)

    console.print(f"[bold]Settings diff for {plat} ({scope})[/bold]\n")
    console.print(f"  Target: {target_path}")

    if existing:
        console.print(f"  Current: {len(kept_entries)} permissions")
    else:
        console.print("  Current: [dim]no file[/dim]")

    console.print(f"  Template: {len(template.permissions.get('allow', []))} permissions")
    console.print(f"  Result: {len(all_merged)} permissions\n")

    if new_entries:
        table = Table(title="New permissions to add")
        table.add_column("Permission", style="green")
        for entry in new_entries:
            table.add_row(f"+ {entry}")
        console.print(table)
    else:
        console.print("  [dim]No new permissions — settings are up to date.[/dim]")


@settings_app.command()
def apply(
    platform: str | None = typer.Option(  # noqa: UP007
        None, "--platform", "-p", help="Target platform."
    ),
    scope_global: bool = typer.Option(False, "--global", "-g", help="Apply to global settings."),
    scope_local: bool = typer.Option(
        False, "--local", "-l", help="Apply to local project settings."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing settings entirely."
    ),
) -> None:
    """Apply settings template to platform configuration.

    By default merges with existing settings (keeps current permissions,
    adds new ones from template). Use --force to replace entirely.
    """
    if not scope_global and not scope_local:
        scope_local = True

    cfg = WhetConfig.load()
    plat = platform or cfg.target.value

    template_path = get_template_path(plat)
    if not template_path.exists():
        console.print(f"[red]No template for '{plat}'.[/red]")
        raise typer.Exit(code=1)

    template = load_template(template_path)

    scopes = []
    if scope_global:
        scopes.append("global")
    if scope_local:
        scopes.append("local")

    for scope in scopes:
        target_path = get_settings_target(plat, scope)
        existing = load_existing(target_path)

        if existing and not force:
            merged = merge_settings(existing, template)
            new_entries, _, _ = diff_settings(existing, template)
            write_settings(target_path, merged)
            n_new = len(new_entries)
            n_total = len(merged.permissions.get("allow", []))
            console.print(
                f"  [green]✓[/green] Merged {scope}: +{n_new} new, {n_total} total → {target_path}"
            )
        elif existing and force:
            # Back up existing
            backup = target_path.with_suffix(".json.bak")
            backup.write_text(target_path.read_text())
            write_settings(target_path, template)
            n = len(template.permissions.get("allow", []))
            console.print(f"  [green]✓[/green] Replaced {scope}: {n} permissions → {target_path}")
            console.print(f"    [dim]Backup: {backup}[/dim]")
        else:
            write_settings(target_path, template)
            n = len(template.permissions.get("allow", []))
            console.print(f"  [green]✓[/green] Created {scope}: {n} permissions → {target_path}")
