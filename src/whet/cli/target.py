"""Target command — set and show the active platform."""

from __future__ import annotations

import typer
from rich.console import Console

from whet.adapters.detect import detect_platform
from whet.cli import app
from whet.core.config import PLATFORM_PATHS, Platform, WhetConfig

console = Console()


@app.command()
def target(
    platform: Platform | None = typer.Argument(None, help="Platform to target."),  # noqa: UP007
) -> None:
    """Set or show the target platform.

    Without arguments, shows the current target and auto-detected platform.
    """
    if platform:
        cfg = WhetConfig.load()
        cfg = cfg.model_copy(update={"target": platform})
        cfg.save()
        paths = PLATFORM_PATHS[platform]
        console.print(f"[bold green]✓ Target set to {platform.value}[/bold green]")
        console.print(f"  Global: {paths.global_dir}")
        console.print(f"  Local:  {paths.local_dir}")
        return

    # Show current state
    cfg = WhetConfig.load()
    detected = detect_platform()
    console.print(f"[bold]Current target:[/bold] [cyan]{cfg.target.value}[/cyan]\n")
    console.print("[bold]Platform Detection[/bold]\n")

    if detected:
        console.print(f"  Auto-detected: [cyan]{detected.value}[/cyan]")
    else:
        console.print("  Auto-detected: [dim]none[/dim]")

    console.print("\n[bold]Available Platforms[/bold]\n")
    for plat, paths in PLATFORM_PATHS.items():
        marker = " [green]✓[/green]" if plat == detected else ""
        console.print(f"  [cyan]{plat.value}[/cyan]{marker}")
        console.print(f"    Global: {paths.global_dir}")
        console.print(f"    Local:  {paths.local_dir}")
