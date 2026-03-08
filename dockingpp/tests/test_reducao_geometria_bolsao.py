import math

import numpy as np

from dockingpp.data.structs import Pocket
from dockingpp.reducao.geometria_bolsao import descrever_geometria_bolsao
from dockingpp.reducao.modelos import GeometriaBolsao


def test_descrever_geometria_bolsao_simples():
    coords = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, -0.25],
            [0.0, 0.0, 0.25],
        ],
        dtype=float,
    )
    bolsao = Pocket(
        id="b1",
        center=np.zeros(3, dtype=float),
        radius=2.0,
        coords=coords,
    )

    geometria = descrever_geometria_bolsao(bolsao)

    assert isinstance(geometria, GeometriaBolsao)
    assert geometria.id_bolsao == "b1"
    assert math.isclose(geometria.comprimento_util, 2.0, rel_tol=1e-8, abs_tol=1e-8)
    assert math.isclose(geometria.largura_util, 1.0, rel_tol=1e-8, abs_tol=1e-8)
    assert math.isclose(geometria.profundidade_util, 0.5, rel_tol=1e-8, abs_tol=1e-8)
    assert 0.0 <= geometria.continuidade_superficial <= 1.0
    assert 0.0 <= geometria.exposicao_superficial <= 1.0
    assert geometria.volume_estimado > 0.0


def test_descrever_geometria_bolsao_com_dados_incompletos_sem_coords():
    bolsao = Pocket(
        id="b2",
        center=np.array([1.0, 1.0, 1.0], dtype=float),
        radius=3.0,
        coords=np.zeros((0, 3), dtype=float),
    )

    geometria = descrever_geometria_bolsao(bolsao)

    assert geometria.id_bolsao == "b2"
    assert geometria.comprimento_util == 6.0
    assert geometria.largura_util == 6.0
    assert geometria.profundidade_util == 6.0
    assert geometria.continuidade_superficial == 0.0
    assert geometria.exposicao_superficial == 0.5
    assert geometria.volume_estimado > 0.0


def test_descrever_geometria_bolsao_retorno_estavel():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    bolsao = Pocket(id="b3", center=np.zeros(3, dtype=float), radius=2.0, coords=coords)

    g1 = descrever_geometria_bolsao(bolsao)
    g2 = descrever_geometria_bolsao(bolsao)

    assert g1 == g2


def test_descrever_geometria_bolsao_usa_meta_quando_coords_ausente():
    bolsao = Pocket(
        id="b4",
        center=np.zeros(3, dtype=float),
        radius=1.5,
        coords=np.zeros((0, 3), dtype=float),
        meta={"coords": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)},
    )

    geometria = descrever_geometria_bolsao(bolsao)

    assert geometria.id_bolsao == "b4"
    assert math.isclose(geometria.comprimento_util, 1.0, rel_tol=1e-8, abs_tol=1e-8)
    assert math.isclose(geometria.largura_util, 0.0, rel_tol=1e-8, abs_tol=1e-8)
    assert math.isclose(geometria.profundidade_util, 0.0, rel_tol=1e-8, abs_tol=1e-8)
    assert geometria.volume_estimado == 0.0
