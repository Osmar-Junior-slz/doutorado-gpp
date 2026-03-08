import math

import numpy as np

from dockingpp.reducao.modelos import PerfilPeptideo
from dockingpp.reducao.perfil_peptideo import construir_perfil_peptideo


def test_construir_perfil_retorna_tipo_esperado():
    peptideo = {"coords": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)}

    perfil = construir_perfil_peptideo(peptideo)

    assert isinstance(perfil, PerfilPeptideo)


def test_construir_perfil_filtra_coordenadas_invalidas_e_mantem_valores():
    peptideo = {
        "coords": np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [np.nan, 0.0, 0.0],
            ],
            dtype=float,
        )
    }

    perfil = construir_perfil_peptideo(peptideo)

    assert math.isclose(perfil.comprimento_efetivo, 2.0, rel_tol=1e-8, abs_tol=1e-8)
    assert math.isclose(perfil.largura_efetiva, 0.0, rel_tol=1e-8, abs_tol=1e-8)
    assert math.isclose(perfil.espessura_efetiva, 0.0, rel_tol=1e-8, abs_tol=1e-8)
    assert math.isclose(perfil.extensao_maxima, 2.0, rel_tol=1e-8, abs_tol=1e-8)


def test_construir_perfil_caso_degenerado_vazio():
    perfil = construir_perfil_peptideo({"coords": np.zeros((0, 3), dtype=float)})

    assert perfil == PerfilPeptideo(
        comprimento_efetivo=0.0,
        largura_efetiva=0.0,
        espessura_efetiva=0.0,
        extensao_maxima=0.0,
        raio_giro=0.0,
        indice_flexibilidade=0.0,
    )


def test_construir_perfil_caso_degenerado_ponto_unico():
    perfil = construir_perfil_peptideo(np.array([[1.0, 2.0, 3.0]], dtype=float))

    assert perfil.comprimento_efetivo == 0.0
    assert perfil.largura_efetiva == 0.0
    assert perfil.espessura_efetiva == 0.0
    assert perfil.extensao_maxima == 0.0
    assert perfil.raio_giro == 0.0
    assert perfil.indice_flexibilidade == 0.0


def test_construir_perfil_caso_simples_em_linha_reta():
    peptideo = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    perfil = construir_perfil_peptideo(peptideo)

    assert math.isclose(perfil.comprimento_efetivo, 4.0, rel_tol=1e-8, abs_tol=1e-8)
    assert math.isclose(perfil.largura_efetiva, 0.0, rel_tol=1e-8, abs_tol=1e-8)
    assert math.isclose(perfil.espessura_efetiva, 0.0, rel_tol=1e-8, abs_tol=1e-8)
    assert math.isclose(perfil.extensao_maxima, 4.0, rel_tol=1e-8, abs_tol=1e-8)
    assert math.isclose(perfil.raio_giro, math.sqrt(8.0 / 3.0), rel_tol=1e-8, abs_tol=1e-8)
    assert math.isclose(perfil.indice_flexibilidade, 0.0, rel_tol=1e-8, abs_tol=1e-8)
