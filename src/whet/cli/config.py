"""Config command — get/set whet configuration."""

from __future__ import annotations

import typer
from rich.console import Console

from whet.cli import app
from whet.core.config import Platform, WhetConfig

console = Console()


@app.command()
def config(
    key: str | None = typer.Argument(None, help="Configuration key."),  # noqa: UP007
    value: str | None = typer.Argument(None, help="Value to set."),  # noqa: UP007
) -> None:
    """Get or set whet configuration.

    Without arguments, shows all configuration.
    With a key, shows that value. With key and value, sets it.
    """
    cfg = WhetConfig.load()

    if key is None:
        console.print("[bold]whet configuration[/bold]\n")
        console.print(f"  target:         {cfg.target.value}")
        console.print(f"  skills_dir:     {cfg.skills_dir}")
        console.print(f"  archetypes_dir: {cfg.archetypes_dir}")
        return

    if key == "target":
        if value:
            try:
                platform = Platform(value)
            except ValueError as err:
                valid = ", ".join(p.value for p in Platform)
                console.print(f"[red]Invalid platform '{value}'. Valid: {valid}[/red]")
                raise typer.Exit(code=1) from err
            cfg = cfg.model_copy(update={"target": platform})
            cfg.save()
            console.print(f"[green]✓[/green] target = {platform.value}")
        else:
            console.print(f"target = {cfg.target.value}")
    else:
        console.print(f"[yellow]Unknown config key: {key}[/yellow]")
        raise typer.Exit(code=1)
