"""Render the architecture diagram to ``docs/diagram.png`` (reproducible).

A small, dependency-light matplotlib drawing of the layered architecture and the
agent's critique→revise graph, so the diagram in ``docs/architecture.md`` is
regenerated from code rather than hand-edited in an image tool.

Usage (from the repo root):

    python scripts/gen_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must precede the pyplot import.

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "docs" / "diagram.png"

# (label, sub-label) for each architecture layer, top to bottom.
_LAYERS: list[tuple[str, str]] = [
    ("USER", "a person submitting a task"),
    ("UI  ·  ui/", "Streamlit — submit task, view v1→v2, metrics; read-only demo"),
    ("APPLICATION  ·  app/", "FastAPI — /health · /run · /results (thin boundary)"),
    ("AGENT  ·  agents/", "LangGraph: retrieve → generate → critique → revise; retrieval tool"),
    ("EVALUATION  ·  evaluators/", "deterministic checks + sampled LLM judge + rubric + analysis"),
    ("FEEDBACK  ·  feedback/", "EvalResult → structured critique → drives the revision"),
    ("STORAGE  ·  storage/", "SQLite — tasks · responses · traces · evals · experiments"),
    ("REPORTING  ·  reports/", "pandas metrics + matplotlib charts (read-only over storage)"),
]

_LAYER_COLORS = [
    "#e9ecef", "#d8e2dc", "#cfe1f2", "#bcd4e6",
    "#a9c5db", "#cdb4db", "#ffd6a5", "#caf0c8",
]


def _layer_box(ax: plt.Axes, x: float, y: float, w: float, h: float,
               title: str, sub: str, color: str) -> None:
    """Draw one rounded layer box with a title and sub-label."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2, edgecolor="#343a40", facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + 0.12, y + h * 0.62, title, fontsize=11, fontweight="bold",
            va="center", ha="left")
    ax.text(x + 0.12, y + h * 0.26, sub, fontsize=8.5, va="center", ha="left",
            color="#343a40")


def _down_arrow(ax: plt.Axes, x: float, y_top: float, y_bot: float) -> None:
    """Draw a downward connector arrow between two stacked layers."""
    ax.add_patch(FancyArrowPatch(
        (x, y_top), (x, y_bot),
        arrowstyle="-|>", mutation_scale=14, linewidth=1.1, color="#495057",
    ))


def render(out_path: Path = OUT_PATH) -> Path:
    """Render the architecture diagram to ``out_path`` and return it."""
    fig, (ax_layers, ax_graph) = plt.subplots(
        1, 2, figsize=(15, 8.5), gridspec_kw={"width_ratios": [1.45, 1.0]}
    )

    # --- Left: layered architecture ------------------------------------
    ax_layers.set_title("Layered architecture", fontsize=13, fontweight="bold", pad=12)
    n = len(_LAYERS)
    box_w, box_h, gap = 8.6, 0.86, 0.30
    x0 = 0.4
    top = n * (box_h + gap)
    for i, (title, sub) in enumerate(_LAYERS):
        y = top - (i + 1) * (box_h + gap)
        _layer_box(ax_layers, x0, y, box_w, box_h, title, sub, _LAYER_COLORS[i])
        if i < n - 1:
            _down_arrow(ax_layers, x0 + box_w / 2, y, y - gap + 0.04)
    ax_layers.text(
        x0 + box_w / 2, -0.55,
        "Storage is the integration boundary: every layer depends on the schema, "
        "not on each other.",
        fontsize=8.5, style="italic", ha="center", color="#495057",
    )
    ax_layers.set_xlim(0, box_w + 0.8)
    ax_layers.set_ylim(-0.9, top + 0.2)
    ax_layers.axis("off")

    # --- Right: the critique→revise graph ------------------------------
    ax_graph.set_title("Agent graph (critique → revise)", fontsize=13,
                       fontweight="bold", pad=12)
    nodes = {
        "retrieve":      (0.5, 9.2),
        "generate":      (0.5, 8.0),
        "evaluate_v1":   (0.5, 6.8),
        "feedback":      (0.5, 5.4),
        "revise":        (0.5, 4.2),
        "evaluate_v2":   (0.5, 3.0),
        "carry_forward": (2.4, 5.4),
        "END":           (1.45, 1.4),
    }
    node_w, node_h = 1.7, 0.7
    free = {"retrieve", "evaluate_v1", "feedback", "evaluate_v2", "carry_forward"}
    for name, (cx, cy) in nodes.items():
        color = "#caf0c8" if name in free else ("#dee2e6" if name == "END" else "#bcd4e6")
        box = FancyBboxPatch(
            (cx - node_w / 2, cy - node_h / 2), node_w, node_h,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            linewidth=1.2, edgecolor="#343a40", facecolor=color,
        )
        ax_graph.add_patch(box)
        ax_graph.text(cx, cy, name, fontsize=9, fontweight="bold",
                      ha="center", va="center")

    def edge(a: str, b: str, label: str = "", *, color: str = "#495057") -> None:
        (ax_, ay), (bx, by) = nodes[a], nodes[b]
        ax_graph.add_patch(FancyArrowPatch(
            (ax_, ay - node_h / 2) if ay > by else (ax_, ay + node_h / 2),
            (bx, by + node_h / 2) if ay > by else (bx, by - node_h / 2),
            arrowstyle="-|>", mutation_scale=13, linewidth=1.1,
            color=color, connectionstyle="arc3,rad=0.0",
        ))
        if label:
            ax_graph.text((ax_ + bx) / 2 + 0.15, (ay + by) / 2, label,
                          fontsize=7.5, color=color, ha="left", va="center")

    edge("retrieve", "generate")
    edge("generate", "evaluate_v1")
    edge("evaluate_v1", "feedback", "fail &\nbudget left", color="#c1121f")
    edge("feedback", "revise")
    edge("revise", "evaluate_v2")
    edge("evaluate_v2", "END")
    # branch to carry_forward (pass / budget exhausted) and on to END
    ax_graph.add_patch(FancyArrowPatch(
        (nodes["evaluate_v1"][0] + node_w / 2, nodes["evaluate_v1"][1]),
        (nodes["carry_forward"][0], nodes["carry_forward"][1] + node_h / 2),
        arrowstyle="-|>", mutation_scale=13, linewidth=1.1, color="#2a9d8f",
        connectionstyle="arc3,rad=-0.3",
    ))
    ax_graph.text(2.55, 6.4, "pass / budget\nexhausted", fontsize=7.5,
                  color="#2a9d8f", ha="left", va="center")
    ax_graph.add_patch(FancyArrowPatch(
        (nodes["carry_forward"][0], nodes["carry_forward"][1] - node_h / 2),
        (nodes["END"][0] + 0.3, nodes["END"][1] + node_h / 2),
        arrowstyle="-|>", mutation_scale=13, linewidth=1.1, color="#495057",
        connectionstyle="arc3,rad=0.25",
    ))

    ax_graph.text(
        1.45, 0.4,
        "green = free deterministic step · blue = model call · "
        "the paid judge is NEVER in the loop",
        fontsize=7.5, style="italic", ha="center", color="#495057",
    )
    ax_graph.set_xlim(-0.7, 3.6)
    ax_graph.set_ylim(0, 10.0)
    ax_graph.axis("off")

    fig.suptitle("eval-loop — architecture", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    print(f"Wrote {render()}")
