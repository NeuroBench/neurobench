from rich.layout import Layout
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.panel import Panel
from rich.columns import Columns


def make_layout(verbose: bool = False) -> Layout:
    """Create a layout for the benchmark runner."""
    layout = Layout()
    if verbose:
        layout.split(
            Layout(name="progress", size=3),
            Layout(name="content"),
        )
    else:
        layout.split(
            Layout(name="progress", size=3),
        )

    return layout


def create_progress_bar(total: int) -> Progress:
    """Create a progress bar."""

    progress = Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[bold cyan]Processing[/bold cyan]"),
        BarColumn(bar_width=40, style="magenta", complete_style="cyan"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[bold blue]{task.completed}/{task.total}[/bold blue] batches"),
        TextColumn("•"),
        TimeRemainingColumn(),
        expand=True,
        transient=False,
    )

    return progress


def create_content_panel(progress, batch_results, verbose):
    panels = []
    if verbose:
        for key, value in batch_results.items():
            if isinstance(value, dict):
                sub_panels = [
                    Panel(
                        f"Value: {sub_value}",
                        title=sub_key,
                        border_style="green",
                    )
                    for sub_key, sub_value in value.items()
                ]
                panels.append(
                    Panel(
                        Columns(sub_panels),
                        title=key,
                        border_style="blue",
                    )
                )
            else:
                panels.append(
                    Panel(
                        f"Value: {value}",
                        title=key,
                        border_style="blue",
                    )
                )
    # Create a structured panel for the content section
    content_panel = Panel(
        Columns(panels),
        title=f"Processing batch {progress.tasks[0].completed}/{progress.tasks[0].total}",
        title_align="left",
        border_style="cyan",
    )

    # Update the content section with the history of panels
    return content_panel
