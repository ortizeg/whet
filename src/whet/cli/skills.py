"""Skill management commands: add, remove, list, search, info."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from whet.cli import app
from whet.core.config import Platform, WhetConfig
from whet.registry.loader import discover_skills, load_skill
from whet.registry.resolver import resolve_dependencies
from whet.registry.search import search_skills

console = Console()


def _get_config() -> WhetConfig:
    return WhetConfig.load()


def _get_adapter(platform: Platform):  # type: ignore[no-untyped-def]
    from whet.adapters.antigravity import AntigravityAdapter
    from whet.adapters.claude import ClaudeAdapter
    from whet.adapters.copilot import CopilotAdapter
    from whet.adapters.cursor import CursorAdapter

    adapters = {
        Platform.CLAUDE: ClaudeAdapter,
        Platform.ANTIGRAVITY: AntigravityAdapter,
        Platform.CURSOR: CursorAdapter,
        Platform.COPILOT: CopilotAdapter,
    }
    return adapters[platform]()


@app.command()
def add(
    skills: list[str] = typer.Argument(help="Skill name(s) to add."),
    scope: bool = typer.Option(False, "--global", "-g", help="Install globally."),
) -> None:
    """Add skill(s) to the current project or globally."""
    cfg = _get_config()
    adapter = _get_adapter(cfg.target)
    all_skills = discover_skills(cfg.skills_dir)
    available = {s.name: s for s in all_skills}

    # Resolve dependencies
    to_install = resolve_dependencies(skills, available)

    paths = cfg.get_platform_paths()
    target_dir = paths.global_dir if scope else paths.local_dir

    installed_count = 0
    for name in to_install:
        skill = available.get(name)
        if not skill:
            console.print(f"[yellow]Warning:[/yellow] Skill '{name}' not found, skipping.")
            continue
        adapter.install_skill(skill, target_dir)
        extra = " (dependency)" if name not in skills else ""
        console.print(f"  [green]✓[/green] {name}{extra}")
        installed_count += 1

    console.print(f"\n[bold]Installed {installed_count} skill(s) to {target_dir}[/bold]")


@app.command()
def remove(
    skill_name: str = typer.Argument(help="Skill name to remove."),
    scope: bool = typer.Option(False, "--global", "-g", help="Remove from global installation."),
) -> None:
    """Remove a skill from the current project or global installation."""
    cfg = _get_config()
    adapter = _get_adapter(cfg.target)
    paths = cfg.get_platform_paths()
    target_dir = paths.global_dir if scope else paths.local_dir

    if adapter.remove_skill(skill_name, target_dir):
        console.print(f"  [green]✓[/green] Removed {skill_name}")
    else:
        console.print(f"  [red]✗[/red] Skill '{skill_name}' not found in {target_dir}")
        raise typer.Exit(code=1)


@app.command(name="list")
def list_skills(
    installed: bool = typer.Option(False, "--installed", "-i", help="Show only installed skills."),
    category: str | None = typer.Option(None, "--category", "--cat", help="Filter by category."),  # noqa: UP007
) -> None:
    """Browse available skills."""
    cfg = _get_config()

    if installed:
        adapter = _get_adapter(cfg.target)
        paths = cfg.get_platform_paths()
        # Check both local and global
        local_installed = adapter.list_installed(paths.local_dir)
        global_installed = adapter.list_installed(paths.global_dir)
        all_installed = sorted(set(local_installed + global_installed))

        if not all_installed:
            console.print("[dim]No skills installed.[/dim]")
            raise typer.Exit()

        table = Table(title="Installed Skills")
        table.add_column("Name", style="cyan")
        table.add_column("Scope", style="green")

        for name in all_installed:
            scope_label = []
            if name in local_installed:
                scope_label.append("local")
            if name in global_installed:
                scope_label.append("global")
            table.add_row(name, ", ".join(scope_label))

        console.print(table)
        return

    all_skills = discover_skills(cfg.skills_dir)
    if category:
        all_skills = [s for s in all_skills if s.category == category]

    if not all_skills:
        console.print("[dim]No skills found.[/dim]")
        raise typer.Exit()

    table = Table(title="Available Skills")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Category", style="green")
    table.add_column("Description")

    for skill in all_skills:
        desc = skill.description[:80] + "..." if len(skill.description) > 80 else skill.description
        table.add_row(skill.name, skill.category, desc)

    console.print(table)
    console.print(f"\n[dim]{len(all_skills)} skills available[/dim]")


@app.command()
def search(
    query: str = typer.Argument(help="Search query (matches name, description, tags)."),
) -> None:
    """Search skills by name, description, or tags."""
    cfg = _get_config()
    all_skills = discover_skills(cfg.skills_dir)
    results = search_skills(all_skills, query=query)

    if not results:
        console.print(f"[dim]No skills matching '{query}'[/dim]")
        raise typer.Exit()

    table = Table(title=f"Search results for '{query}'")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Category", style="green")
    table.add_column("Tags", style="dim")
    table.add_column("Description")

    for skill in results:
        tags = ", ".join(skill.tags[:4])
        desc = skill.description[:60] + "..." if len(skill.description) > 60 else skill.description
        table.add_row(skill.name, skill.category, tags, desc)

    console.print(table)


@app.command()
def info(
    skill_name: str = typer.Argument(help="Skill name."),
) -> None:
    """Show detailed information about a skill."""
    cfg = _get_config()
    skill = load_skill(cfg.skills_dir, skill_name)

    if not skill:
        console.print(f"[red]Skill '{skill_name}' not found.[/red]")
        raise typer.Exit(code=1)

    console.print(f"\n[bold cyan]{skill.name}[/bold cyan]")
    console.print(f"[dim]{skill.description}[/dim]\n")

    if skill.metadata:
        m = skill.metadata
        console.print(f"  [bold]Category:[/bold] {m.category}")
        console.print(f"  [bold]Version:[/bold]  {m.version}")
        console.print(f"  [bold]Tags:[/bold]     {', '.join(m.tags)}")

        if m.dependencies.requires:
            console.print(f"  [bold]Requires:[/bold] {', '.join(m.dependencies.requires)}")
        if m.dependencies.recommends:
            console.print(f"  [bold]Recommends:[/bold] {', '.join(m.dependencies.recommends)}")

        if m.compatibility.libraries:
            console.print("  [bold]Libraries:[/bold]")
            for lib, ver in m.compatibility.libraries.items():
                console.print(f"    {lib} {ver}")

    console.print(f"\n  [dim]Path: {skill.path}[/dim]")
