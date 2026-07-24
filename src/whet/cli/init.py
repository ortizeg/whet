"""Init command — scaffold a new project from an archetype."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from whet.cli import app
from whet.core.config import WhetConfig
from whet.scaffold.engine import (
    ScaffoldContext,
    discover_archetypes,
    load_archetype,
    render_template,
)

console = Console()


@app.command(name="init")
def init_project(
    archetype_name: str | None = typer.Argument(  # noqa: UP007
        None, help="Archetype to scaffold from."
    ),
    output_dir: str | None = typer.Option(  # noqa: UP007
        None, "--output", "-o", help="Output directory (default: current dir)."
    ),
    project_name: str | None = typer.Option(  # noqa: UP007
        None, "--name", "-n", help="Project name."
    ),
    author: str | None = typer.Option(  # noqa: UP007
        None, "--author", "-a", help="Author name."
    ),
    with_recommended: bool = typer.Option(
        False, "--with-recommended", help="Also install the archetype's recommended skills."
    ),
    no_skills: bool = typer.Option(
        False, "--no-skills", help="Scaffold files only; do not install the archetype's skills."
    ),
) -> None:
    """Scaffold a new project from an archetype template.

    Installs the archetype's required skills into the new project so it is usable
    immediately. Without arguments, lists available archetypes.
    """
    cfg = WhetConfig.load()

    if archetype_name is None:
        _list_archetypes(cfg)
        return

    archetype = load_archetype(cfg.archetypes_dir / archetype_name)
    if not archetype:
        console.print(f"[red]Archetype '{archetype_name}' not found.[/red]")
        console.print("[dim]Run 'whet init' to see available archetypes.[/dim]")
        raise typer.Exit(code=1)

    name = project_name or archetype_name
    dest = Path(output_dir) if output_dir else Path.cwd() / name

    if dest.exists() and any(dest.iterdir()):
        console.print(f"[red]Directory '{dest}' already exists and is not empty.[/red]")
        raise typer.Exit(code=1)

    context = ScaffoldContext(
        project_name=name,
        description=archetype.metadata.description,
        author=author or "",
    )

    console.print(f"[bold]Scaffolding: {archetype.metadata.name}[/bold]\n")
    console.print(f"  Project:   {context.project_name}")
    console.print(f"  Package:   {context.package_name}")
    console.print(f"  Directory: {dest}")
    console.print()

    render_template(archetype, dest, context)

    console.print(f"[bold green]✓ Project scaffolded at {dest}[/bold green]\n")

    wanted = list(archetype.skills.required)
    if with_recommended:
        wanted += [s for s in archetype.skills.recommended if s not in wanted]

    if no_skills:
        if wanted:
            console.print("[bold]Skills for this archetype (not installed):[/bold]")
            for skill in wanted:
                console.print(f"  - {skill}")
            console.print(f"\nRun: [bold]cd {dest.name} && whet add {' '.join(wanted)}[/bold]")
    elif wanted:
        _install_skills(cfg, wanted, dest)

    if archetype.skills.recommended and not with_recommended:
        console.print("\n[bold]Recommended skills[/bold] [dim](--with-recommended)[/dim]")
        for skill in archetype.skills.recommended:
            console.print(f"  - {skill}")


def _install_skills(cfg: WhetConfig, names: list[str], dest: Path) -> None:
    """Install the archetype's skills into the scaffolded project.

    An archetype's value over a plain folder copy is the skill set it composes, so
    scaffolding without installing them left the most important part as homework.
    """
    from whet.cli.skills import _get_adapter
    from whet.core.config import PLATFORM_PATHS
    from whet.registry.loader import load_skill

    adapter = _get_adapter(cfg.target)
    target_dir = dest / PLATFORM_PATHS[cfg.target].local_dir

    console.print(f"[bold]Installing {len(names)} skills...[/bold]")
    missing: list[str] = []
    for name in names:
        skill = load_skill(cfg.skills_dir, name)
        if skill is None:
            missing.append(name)
            continue
        adapter.install_skill(skill, target_dir)
        console.print(f"  [green]✓[/green] {name}")

    if missing:
        # A dangling name means archetype.toml drifted from the skill library.
        console.print(f"  [yellow]![/yellow] not found: {', '.join(missing)}")

    console.print(f"\n[bold green]✓ Installed {len(names) - len(missing)} skills[/bold green]")


def _list_archetypes(cfg: WhetConfig) -> None:
    """List available archetypes."""
    archetypes = discover_archetypes(cfg.archetypes_dir)

    if not archetypes:
        console.print("[dim]No archetypes found.[/dim]")
        raise typer.Exit()

    table = Table(title="Available Archetypes")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Category", style="green")
    table.add_column("Description")
    table.add_column("Template", style="dim")

    for arch in archetypes:
        has_tmpl = "yes" if arch.has_template else "minimal"
        table.add_row(
            arch.metadata.name,
            arch.metadata.category,
            arch.metadata.description,
            has_tmpl,
        )

    console.print(table)
    console.print("\n[dim]Usage: whet init <archetype-name>[/dim]")
