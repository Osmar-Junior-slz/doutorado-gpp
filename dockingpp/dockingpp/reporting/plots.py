"""Plots mínimos para relatórios de docking."""

from __future__ import annotations


def _plt():
    import matplotlib.pyplot as plt

    return plt


def plot_search_reduction(omega_full: float, omega_reduced: float, omega_ratio: float):
    plt = _plt()
    fig, ax = plt.subplots()
    ax.bar(["full", "reduced"], [omega_full, omega_reduced])
    ax.set_title(f"Redução do espaço (ratio={omega_ratio:.3f})")
    return fig


def plot_convergence(steps: list[int], best_cheap: list[float | None]):
    plt = _plt()
    fig, ax = plt.subplots()
    y = [v for v in best_cheap if v is not None]
    x = steps[: len(y)]
    ax.plot(x, y, marker="o")
    ax.set_title("Convergência por iteração")
    return fig


def plot_cost_vs_quality(n_evals_cheap: int, n_evals_expensive: int, best_score_cheap: float | None, best_score_expensive: float | None):
    plt = _plt()
    fig, ax = plt.subplots()
    ax.scatter([n_evals_cheap, n_evals_expensive], [best_score_cheap or 0.0, best_score_expensive or 0.0])
    ax.set_title("Custo vs qualidade")
    return fig


def plot_trigger_timeline(trigger_steps: list[int]):
    plt = _plt()
    fig, ax = plt.subplots()
    if trigger_steps:
        ax.vlines(trigger_steps, ymin=0, ymax=1)
    ax.set_title("Timeline de triggers do score caro")
    return fig


def plot_pockets_total_vs_selected(n_pockets_total: int, n_pockets_selected: int):
    plt = _plt()
    fig, ax = plt.subplots()
    ax.bar(["total", "selecionados"], [n_pockets_total, n_pockets_selected])
    ax.set_title("Pockets totais vs selecionados")
    return fig


def plot_confidence(confidence: float | None):
    plt = _plt()
    fig, ax = plt.subplots()
    if confidence is None:
        ax.text(0.5, 0.5, "calibração ausente", ha="center", va="center")
    else:
        ax.bar(["confidence"], [confidence])
    ax.set_title("Confiança/Calibração")
    return fig
