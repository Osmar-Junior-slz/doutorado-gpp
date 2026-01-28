"""Modelo stub de priors para bolsões."""

# PT-BR: este módulo mantém compatibilidade com a API antiga de ranqueamento
# de bolsões, agora delegando para o núcleo do pipeline PhD.

from __future__ import annotations

import numpy as np

from dockingpp.core.ranqueamento_bolsoes import ranquear_bolsoes
from dockingpp.data.structs import Pocket


def rank_pockets(receptor: object, pockets: list[Pocket], peptide: object | None = None) -> list[tuple[Pocket, float]]:
    """Ranqueia bolsões via heurística simples (alias retrocompatível)."""

    return ranquear_bolsoes(receptor, pockets, peptide=peptide)


class PriorNetPocket:
    """Rede de prior stub para pontuar bolsões."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Retorna scores de prior para bolsões (stub)."""

        return np.zeros(features.shape[0], dtype=float)
