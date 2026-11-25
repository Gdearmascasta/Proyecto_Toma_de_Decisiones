"""Pruebas de las funciones de valoración que comparten el notebook y el tablero."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))

from calculos import (cobertura_optima, comparar_instrumentos, cop, curva_coberturas, evaluar_estrategia, millones,
                      pct, precio_forward, prima_call, prima_put, strike_put_collar, tornado, trm_simulada,
                      vol_simulada)

ARGUMENTOS = dict(monto_usd=100_000, presupuesto=4_000.0, margen_objetivo=0.25, margen_minimo=0.15)


@pytest.fixture
def trm_final():
    generador = np.random.default_rng(0)
    return 4_000 * np.exp(generador.normal(0, 0.08, 20_000))


def test_forward_cumple_paridad_de_tasas():
    f = precio_forward(4_000, 0.10, 0.04, 12)
    assert f == pytest.approx(4_000 * 1.10 / 1.04)
    assert precio_forward(4_000, 0.10, 0.04, 0) == pytest.approx(4_000)


def test_forward_aplica_recargo_del_banco():
    assert precio_forward(4_000, 0.10, 0.04, 6, 0.002) == pytest.approx(precio_forward(4_000, 0.10, 0.04, 6) * 1.002)


def test_trm_simulada_sin_retornos_ni_deriva_es_el_spot():
    assert np.allclose(trm_simulada(np.zeros(5), 4_000, 0.0, 60), 4_000)


def test_cobertura_total_elimina_la_varianza(trm_final):
    m = evaluar_estrategia(trm_final, 4_100, 1.0, **ARGUMENTOS)
    assert m["desv_costo"] == pytest.approx(0, abs=1e-6)
    assert m["costo_esperado"] == pytest.approx(100_000 * 4_100)
    assert m["costo_p95"] == pytest.approx(m["costo_p5"])


def test_sin_cobertura_el_costo_es_el_de_la_trm_final(trm_final):
    m = evaluar_estrategia(trm_final, 4_100, 0.0, **ARGUMENTOS)
    assert m["costo_esperado"] == pytest.approx(100_000 * trm_final.mean())
    assert m["cvar95"] >= m["costo_p95"] >= m["costo_esperado"]


def test_cobertura_optima_respeta_la_tolerancia(trm_final):
    curva = curva_coberturas(trm_final, 4_100, **ARGUMENTOS)
    assert len(curva) == 21
    optima = cobertura_optima(curva, 0.05)
    assert optima["prob_margen_bajo"] <= 0.05
    factibles = curva[curva["prob_margen_bajo"] <= 0.05]
    assert optima["costo_esperado"] == factibles["costo_esperado"].min()


def test_cobertura_optima_sin_factibles_devuelve_la_de_menor_riesgo(trm_final):
    curva = curva_coberturas(trm_final, 4_100, **ARGUMENTOS)
    optima = cobertura_optima(curva, -1)
    assert optima["prob_margen_bajo"] == curva["prob_margen_bajo"].min()


def test_tornado_ordena_por_rango_y_la_base_coincide(trm_final):
    tabla, base = tornado(4_000, 0.04, 0.06, 4_000, trm_final, 100_000, 0.0015, 0.25, 0.5, 6)
    assert list(tabla["rango"]) == sorted(tabla["rango"])
    assert len(tabla) == 6
    f = precio_forward(4_000, 0.10, 0.04, 6, 0.0015)
    esperado = 100_000 * 4_000 / 0.75 - 100_000 * (0.5 * f + 0.5 * np.median(trm_final))
    assert base == pytest.approx(esperado)


def test_paridad_put_call():
    c, p = prima_call(4_100, 4_000, 0.15, 6), prima_put(4_100, 4_000, 0.15, 6)
    assert c - p == pytest.approx(4_100 - 4_000)


def test_prima_call_sin_volatilidad_es_el_valor_intrinseco():
    assert prima_call(4_100, 4_000, 0.0, 6) == pytest.approx(100)
    assert prima_call(4_100, 4_200, 0.0, 6) == pytest.approx(0)


def test_prima_call_crece_con_la_volatilidad():
    assert prima_call(4_100, 4_100, 0.20, 6) > prima_call(4_100, 4_100, 0.10, 6) > 0


def test_collar_de_costo_cero_iguala_las_primas():
    strike_call = 4_100 * 1.03
    strike_put = strike_put_collar(4_100, strike_call, 0.15, 6)
    assert strike_put < 4_100
    assert prima_put(4_100, strike_put, 0.15, 6) == pytest.approx(prima_call(4_100, strike_call, 0.15, 6), abs=1e-6)


def test_collar_con_strike_en_el_forward_equivale_al_forward():
    assert strike_put_collar(4_100, 4_100, 0.15, 6) == pytest.approx(4_100, abs=1e-6)


def test_vol_simulada_anualiza_la_desviacion():
    r = np.array([-0.1, 0.1] * 500)
    assert vol_simulada(r, 3) == pytest.approx(r.std() / np.sqrt(0.25))


def test_comparar_instrumentos(trm_final):
    tabla = comparar_instrumentos(trm_final, 4_100, 0.15, 6, 1.0, 0.03, **ARGUMENTOS).set_index("instrumento")
    assert list(tabla.index) == ["Sin cubrir", "Forward", "Call", "Collar"]
    forward = tabla.loc["Forward"]
    assert forward["costo_esperado"] == pytest.approx(100_000 * 4_100)
    call = tabla.loc["Call"]
    techo = 100_000 * (4_100 * 1.03 + call["prima"])
    assert call["costo_p95"] <= techo + 1e-6
    collar = tabla.loc["Collar"]
    assert collar["prima"] == 0
    costo_collar = 100_000 * np.clip(trm_final, collar["strike_put"], collar["strike_call"])
    assert collar["costo_esperado"] == pytest.approx(costo_collar.mean())
    assert collar["cvar95"] == pytest.approx(costo_collar[costo_collar >= np.percentile(costo_collar, 95)].mean())
    sin = tabla.loc["Sin cubrir"]
    assert sin["costo_esperado"] == pytest.approx(evaluar_estrategia(trm_final, 4_100, 0.0, **ARGUMENTOS)["costo_esperado"])


def test_comparar_instrumentos_con_cobertura_parcial_mezcla_con_la_trm(trm_final):
    tabla = comparar_instrumentos(trm_final, 4_100, 0.15, 6, 0.5, 0.03, **ARGUMENTOS).set_index("instrumento")
    assert tabla.loc["Forward", "costo_esperado"] == pytest.approx(100_000 * (0.5 * 4_100 + 0.5 * trm_final.mean()))


def test_formatos_colombianos():
    assert cop(1234567.891, 2) == "$ 1.234.567,89"
    assert millones(1_260_000) == "$ 1,3 M"
    assert pct(0.1234) == "12,3 %"
