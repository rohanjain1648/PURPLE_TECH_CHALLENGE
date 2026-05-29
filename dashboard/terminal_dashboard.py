"""
Live terminal dashboard — proof that the detection pipeline and API are genuinely connected.
Polls the API every 2 seconds and renders a Rich live display.

Usage:
    python -m dashboard.terminal_dashboard \\
        --store-id STORE_BLR_002 \\
        --api-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from typing import Optional

import requests
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()
API_POLL_INTERVAL = 2.0   # seconds


def _get(api_url: str, path: str, timeout: int = 5) -> Optional[dict]:
    try:
        r = requests.get(f"{api_url}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"_error": str(exc)}


def _metrics_panel(data: Optional[dict]) -> Panel:
    if not data or "_error" in data:
        err = (data or {}).get("_error", "API unavailable")
        return Panel(Text(f"[red]{err}[/red]"), title="[bold]Store Metrics[/bold]", border_style="red")

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold white")

    visitors = data.get("unique_visitors", 0)
    active = data.get("active_visitors", 0)
    conv = data.get("conversion_rate", 0.0)
    queue = data.get("current_queue_depth", 0)
    abandon = data.get("abandonment_rate", 0.0)

    table.add_row("Unique Visitors Today", str(visitors))
    table.add_row("Active In-Store", f"[green]{active}[/green]" if active else "0")
    table.add_row("Conversion Rate", f"[green]{conv:.1%}[/green]" if conv > 0.1 else f"[yellow]{conv:.1%}[/yellow]")
    table.add_row("Billing Queue Depth", f"[red]{queue}[/red]" if queue > 4 else str(queue))
    table.add_row("Queue Abandonment", f"{abandon:.1%}")

    return Panel(table, title="[bold cyan]Store Metrics[/bold cyan]", border_style="cyan")


def _heatmap_panel(data: Optional[dict]) -> Panel:
    if not data or "_error" in data:
        return Panel(Text("[red]No data[/red]"), title="[bold]Zone Heatmap[/bold]", border_style="red")

    zones = data.get("zones", [])
    table = Table(box=box.SIMPLE_HEAD, show_header=True, padding=(0, 1))
    table.add_column("Zone", style="cyan", min_width=14)
    table.add_column("Visits", justify="right")
    table.add_column("Avg Dwell", justify="right")
    table.add_column("Heat", justify="left", min_width=20)

    for z in sorted(zones, key=lambda x: x.get("normalized_score", 0), reverse=True):
        score = z.get("normalized_score", 0)
        filled = int(score / 5)   # 20 blocks max
        bar_color = "red" if score > 75 else "yellow" if score > 40 else "green"
        bar = f"[{bar_color}]{'█' * filled}{'░' * (20 - filled)}[/{bar_color}]"
        dwell_s = z.get("avg_dwell_ms", 0) / 1000
        confidence = "" if z.get("data_confidence") == "HIGH" else " [dim](low data)[/dim]"
        table.add_row(z["zone_id"], str(z.get("visit_count", 0)), f"{dwell_s:.0f}s{confidence}", bar)

    return Panel(table, title="[bold magenta]Zone Heatmap[/bold magenta]", border_style="magenta")


def _funnel_panel(data: Optional[dict]) -> Panel:
    if not data or "_error" in data:
        return Panel(Text("[red]No data[/red]"), title="[bold]Conversion Funnel[/bold]", border_style="red")

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Stage", style="cyan", min_width=16)
    table.add_column("Count", justify="right")
    table.add_column("Drop-off", justify="right", style="red")

    for stage in data.get("stages", []):
        drop = f"-{stage['drop_off_pct']:.1f}%" if stage['drop_off_pct'] > 0 else ""
        table.add_row(stage["stage"], str(stage["count"]), drop)

    return Panel(table, title="[bold yellow]Conversion Funnel[/bold yellow]", border_style="yellow")


def _anomalies_panel(data: Optional[dict]) -> Panel:
    if not data or "_error" in data:
        return Panel(Text("[dim]No anomaly data[/dim]"), title="[bold]Anomalies[/bold]", border_style="dim")

    anomalies = data.get("active_anomalies", [])
    if not anomalies:
        return Panel(
            Text("[green]No active anomalies[/green]", justify="center"),
            title="[bold green]Anomalies[/bold green]",
            border_style="green",
        )

    table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
    table.add_column("Type", style="bold")
    table.add_column("Sev")
    table.add_column("Action", max_width=50, overflow="fold")

    sev_colors = {"CRITICAL": "red", "WARN": "yellow", "INFO": "blue"}
    for a in anomalies:
        sev = a.get("severity", "INFO")
        color = sev_colors.get(sev, "white")
        table.add_row(a["anomaly_type"], f"[{color}]{sev}[/{color}]", a.get("suggested_action", ""))

    return Panel(table, title="[bold red]Active Anomalies[/bold red]", border_style="red")


def _health_panel(data: Optional[dict]) -> Panel:
    if not data or "_error" in data:
        return Panel(Text("[red]API unreachable[/red]"), title="Health", border_style="red")

    status = data.get("status", "unknown")
    color = {"healthy": "green", "degraded": "yellow", "unhealthy": "red"}.get(status, "white")
    db = data.get("database", "?")
    uptime = data.get("uptime_seconds", 0)
    lines = [
        f"Status: [{color}]{status.upper()}[/{color}]",
        f"DB: {db}  Uptime: {uptime:.0f}s",
    ]
    for s in data.get("stores", []):
        icon = "✓" if s["status"] == "HEALTHY" else "!"
        lines.append(f"  {icon} {s['store_id']}: {s['status']}")

    return Panel("\n".join(lines), title="[bold]Health[/bold]", border_style=color)


def run_dashboard(store_id: str, api_url: str) -> None:
    console.print(f"[bold cyan]Store Intelligence Dashboard[/bold cyan] — {store_id}")
    console.print(f"[dim]API: {api_url}  |  Press Ctrl+C to exit[/dim]\n")

    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            metrics = _get(api_url, f"/stores/{store_id}/metrics")
            heatmap = _get(api_url, f"/stores/{store_id}/heatmap")
            funnel = _get(api_url, f"/stores/{store_id}/funnel")
            anomalies = _get(api_url, f"/stores/{store_id}/anomalies")
            health = _get(api_url, "/health")

            now_str = datetime.now().strftime("%H:%M:%S")
            title = f"[bold]Store Intelligence — {store_id}[/bold]  [dim]{now_str}[/dim]"

            layout = Layout()
            layout.split_column(
                Layout(Panel(title, border_style="dim"), size=3),
                Layout(name="top", size=14),
                Layout(name="bottom"),
            )
            layout["top"].split_row(
                Layout(_metrics_panel(metrics), name="metrics"),
                Layout(_funnel_panel(funnel), name="funnel"),
                Layout(_health_panel(health), name="health", minimum_size=28),
            )
            layout["bottom"].split_row(
                Layout(_heatmap_panel(heatmap), name="heatmap"),
                Layout(_anomalies_panel(anomalies), name="anomalies"),
            )

            live.update(layout)
            time.sleep(API_POLL_INTERVAL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Store Intelligence Terminal Dashboard")
    parser.add_argument("--store-id", default="STORE_BLR_002")
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()
    try:
        run_dashboard(args.store_id, args.api_url)
    except KeyboardInterrupt:
        console.print("\n[dim]Dashboard stopped.[/dim]")


if __name__ == "__main__":
    main()
