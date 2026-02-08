"""Gráficos para relatórios (PT-BR)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


def plot_omega_reduction(series: dict[str, Any], out_png: str | Path) -> None:
    """Plota redução do espaço de busca Ω -> Ω'."""

    plt = _carregar_pyplot()
    xs = series.get("iter", list(range(_comprimento_base(series))))
    n_eval_total = series.get("n_eval_total", [])
    n_filtered = series.get("n_filtered", [])
    n_selected = series.get("n_selected", [])

    fig, ax = plt.subplots()
    _plot_linha(ax, xs, n_eval_total, "Ω amostrado (n_eval_total)")
    _plot_linha(ax, xs, n_filtered, "Descartadas (n_filtered)")
    _plot_linha(ax, xs, n_selected, "Ω' selecionado (n_selected)")

    kept_ratio = _calcular_ratio(n_selected, n_eval_total)
    if kept_ratio:
        ax2 = ax.twinx()
        _plot_linha(ax2, xs, kept_ratio, "kept_ratio", linestyle="--", color="tab:green")
        ax2.set_ylabel("kept_ratio")

    ax.set_title("Redução do espaço de busca")
    ax.set_xlabel("Iteração")
    ax.set_ylabel("Quantidade")
    ax.legend(loc="upper left")
    _salvar_figura(fig, out_png)


def plot_cost_quality(series: dict[str, Any], out_png: str | Path) -> None:
    """Plota custo (tempo/iter) vs qualidade (best scores)."""

    plt = _carregar_pyplot()
    fig, ax = plt.subplots()
    for label, dados in _normalizar_para_comparacao(series):
        xs = _series_x(dados)
        best_cheap = dados.get("best_cheap", [])
        best_expensive = dados.get("best_expensive", [])
        _plot_linha(ax, xs, best_cheap, f"{label} - Cheap")
        if any(valor is not None for valor in best_expensive):
            _plot_linha(ax, xs, best_expensive, f"{label} - Expensive", linestyle="--")

    ax.set_title("Custo x Qualidade")
    ax.set_xlabel("Runtime (s) ou Iteração")
    ax.set_ylabel("Best score")
    ax.legend()
    _salvar_figura(fig, out_png)


def plot_score_stability(series: dict[str, Any], out_png: str | Path) -> None:
    """Plota estabilidade do best score ao longo do tempo."""

    plt = _carregar_pyplot()
    xs = _series_x(series)
    ys = series.get("best_cheap", [])
    fig, ax = plt.subplots()
    _plot_linha(ax, xs, ys, "Best cheap")

    marcadores = _pontos_expensive(series, xs)
    if marcadores:
        ax.scatter(marcadores, _filtrar_por_x(xs, ys, marcadores), color="tab:red", label="Expensive ran")

    ax.set_title("Estabilidade do score")
    ax.set_xlabel("Runtime (s) ou Iteração")
    ax.set_ylabel("Best score (cheap)")
    ax.legend()
    _salvar_figura(fig, out_png)


def plot_pocket_rank_effect(events_or_series: list[dict[str, Any]] | dict[str, Any], out_png: str | Path) -> None:
    """Agrega efeito do ranking de pockets."""

    plt = _carregar_pyplot()
    eventos = events_or_series if isinstance(events_or_series, list) else []
    agregados = _agregar_por_pocket(eventos)
    pockets = list(agregados.keys())
    n_eval = [agregados[pocket]["n_eval_total"] for pocket in pockets]
    best_cheap = [agregados[pocket]["best_cheap"] for pocket in pockets]

    fig, ax = plt.subplots()
    ax.bar(pockets, n_eval, label="n_eval_total", alpha=0.6)
    ax.set_xlabel("Pocket rank/id")
    ax.set_ylabel("n_eval_total")

    ax2 = ax.twinx()
    ax2.plot(pockets, best_cheap, color="tab:orange", marker="o", label="best_cheap")
    ax2.set_ylabel("Best cheap")

    ax.set_title("Efeito do pocket ranking")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    _salvar_figura(fig, out_png)


def plot_filter_distribution(events: list[dict[str, Any]], out_png: str | Path) -> bool:
    """Plota distribuição pré/pós filtro quando houver dados suficientes."""

    plt = _carregar_pyplot()
    dados_pre, dados_pos = _buscar_distribuicoes(events)
    if not dados_pre or not dados_pos:
        return False

    fig, ax = plt.subplots()
    ax.boxplot([dados_pre, dados_pos], labels=["Pré-filtro", "Pós-filtro"])
    ax.set_title("Distribuição pré vs pós filtro")
    ax.set_ylabel("Score")
    _salvar_figura(fig, out_png)
    return True


def _normalizar_para_comparacao(series: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    if "full" in series and "reduced" in series:
        return [("Completo", series["full"]), ("Reduzido", series["reduced"])]
    return [("Execução", series)]


def _series_x(series: dict[str, Any]) -> list[Any]:
    runtime = series.get("runtime_s", [])
    if any(valor is not None for valor in runtime):
        return runtime
    return series.get("iter", list(range(_comprimento_base(series))))


def _comprimento_base(series: dict[str, Any]) -> int:
    for chave, valores in series.items():
        if chave == "missing":
            continue
        if isinstance(valores, Iterable):
            return len(list(valores))
    return 0


def _plot_linha(ax: "plt.Axes", xs: list[Any], ys: list[Any], label: str, **kwargs: Any) -> None:
    dados = [(x, y) for x, y in zip(xs, ys, strict=False) if y is not None and x is not None]
    if not dados:
        return
    xs_validos, ys_validos = zip(*dados)
    ax.plot(xs_validos, ys_validos, label=label, **kwargs)


def _calcular_ratio(numerador: list[Any], denominador: list[Any]) -> list[float]:
    ratio: list[float] = []
    for num, den in zip(numerador, denominador, strict=False):
        if num is None or den is None:
            ratio.append(None)  # type: ignore[arg-type]
            continue
        den_val = float(den)
        ratio.append(float(num) / max(den_val, 1.0))
    return ratio


def _pontos_expensive(series: dict[str, Any], xs: list[Any]) -> list[Any]:
    expensive = series.get("expensive_ran", [])
    marcadores: list[Any] = []
    anterior: float | None = None
    for idx, valor in enumerate(expensive):
        if valor is None:
            continue
        atual = float(valor)
        if anterior is not None and atual > anterior and idx < len(xs):
            marcadores.append(xs[idx])
        anterior = atual
    return marcadores


def _filtrar_por_x(xs: list[Any], ys: list[Any], xs_alvo: list[Any]) -> list[Any]:
    mapa = {x: y for x, y in zip(xs, ys, strict=False)}
    return [mapa.get(x) for x in xs_alvo]


def _agregar_por_pocket(eventos: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    agregados: dict[str, dict[str, float]] = {}
    for evento in eventos:
        pocket = _buscar_valor(evento, ["pocket_rank", "rank", "pocket_id", "pocket"])
        if pocket is None:
            continue
        pocket_key = str(pocket)
        registro = agregados.setdefault(pocket_key, {"n_eval_total": 0.0, "best_cheap": float("-inf")})
        n_eval = _buscar_valor(evento, ["n_eval_total", "eval_total", "n_scored", "cheap_evals"])
        if n_eval is None and evento.get("name") in ["n_eval_total", "eval_total", "n_scored", "cheap_evals"]:
            n_eval = evento.get("value")
        if n_eval is not None:
            registro["n_eval_total"] = max(registro["n_eval_total"], float(n_eval))
        best = _buscar_valor(evento, ["best_score_cheap", "best_cheap", "best"])
        if best is None and evento.get("name") in ["best_score_cheap", "best_cheap", "best"]:
            best = evento.get("value")
        if best is not None:
            registro["best_cheap"] = max(registro["best_cheap"], float(best))
    for pocket, registro in agregados.items():
        if registro["best_cheap"] == float("-inf"):
            registro["best_cheap"] = 0.0
    return agregados


def _buscar_distribuicoes(events: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    chaves_pre = ["scores_pre", "scores_before", "scores_raw", "pre_filter_scores"]
    chaves_pos = ["scores_post", "scores_after", "scores_filtered", "post_filter_scores"]
    for evento in events:
        pre = _buscar_lista(evento, chaves_pre)
        pos = _buscar_lista(evento, chaves_pos)
        if pre and pos:
            return pre, pos
    return [], []


def _buscar_lista(evento: dict[str, Any], chaves: list[str]) -> list[float]:
    for chave in chaves:
        valor = evento.get(chave)
        if isinstance(valor, list) and valor:
            return [float(item) for item in valor if isinstance(item, (int, float))]
    return []


def _buscar_valor(evento: dict[str, Any], chaves: list[str]) -> Any:
    for chave in chaves:
        if evento.get(chave) is not None:
            return evento.get(chave)
    extras = evento.get("extras")
    if isinstance(extras, dict):
        for chave in chaves:
            if extras.get(chave) is not None:
                return extras.get(chave)
    return None


def _salvar_figura(fig: "plt.Figure", out_png: str | Path) -> None:
    caminho = Path(out_png)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, bbox_inches="tight")
    _carregar_pyplot().close(fig)


def _carregar_pyplot() -> "plt":
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depende do ambiente
        raise RuntimeError("matplotlib não disponível para gerar gráficos.") from exc
    return plt
