"""Funciones de valoración y simulación que usa el tablero (las mismas del notebook)."""

import math

import numpy as np
import pandas as pd


def precio_forward(spot, tasa_cop, tasa_usd, meses, margen_banco=0.0):
    """Forward teórico por paridad cubierta de tasas más el recargo del banco."""
    return spot * ((1 + tasa_cop) / (1 + tasa_usd)) ** (meses / 12) * (1 + margen_banco)


def trm_simulada(retornos, spot, deriva, pasos):
    """TRM al vencimiento a partir de los retornos simulados con el GJR-GARCH."""
    return spot * np.exp(np.asarray(retornos) + deriva * pasos)


def evaluar_estrategia(trm_final, forward, cobertura, monto_usd, presupuesto, margen_objetivo, margen_minimo):
    """Métricas de costo, riesgo y margen de una estrategia de cobertura."""
    costo = monto_usd * (cobertura * forward + (1 - cobertura) * trm_final)
    return {"cobertura": cobertura, **metricas_costo(costo, monto_usd, presupuesto, margen_objetivo, margen_minimo)}


def metricas_costo(costo, monto_usd, presupuesto, margen_objetivo, margen_minimo):
    """Métricas de un vector de costos simulados en pesos (P95 es el CFaR y cvar95 el promedio del peor 5 %)."""
    costo_presupuestado = monto_usd * presupuesto
    ventas = costo_presupuestado / (1 - margen_objetivo)
    margen = 1 - costo / ventas
    p95 = np.percentile(costo, 95)
    return {
        "costo_esperado": costo.mean(), "costo_p5": np.percentile(costo, 5),
        "costo_p95": p95, "cvar95": costo[costo >= p95].mean(), "desv_costo": costo.std(),
        "desviacion_presupuesto": costo.mean() - costo_presupuestado,
        "prob_superar_presupuesto": (costo > costo_presupuestado * (1 + 1e-9)).mean(),
        "margen_esperado": margen.mean(), "margen_p5": np.percentile(margen, 5),
        "prob_margen_bajo": (margen < margen_minimo).mean(),
        "utilidad_esperada": (ventas - costo).mean(), "prob_perdida": (ventas - costo < 0).mean(),
    }


def curva_coberturas(trm_final, forward, **argumentos):
    """Métricas para coberturas de 0 % a 100 % en pasos de 5 %."""
    coberturas = np.round(np.arange(0, 1.0001, 0.05), 2)
    return pd.DataFrame([evaluar_estrategia(trm_final, forward, h, **argumentos) for h in coberturas])


def cobertura_optima(tabla, tolerancia):
    """Menor costo esperado sujeto a P(margen < mínimo) <= tolerancia."""
    factibles = tabla[tabla["prob_margen_bajo"] <= tolerancia]
    if factibles.empty:
        return tabla.loc[tabla["prob_margen_bajo"].idxmin()]
    return factibles.loc[factibles["costo_esperado"].idxmin()]


def tornado(spot, tasa_usd, diferencial, presupuesto, trm_final, monto, margen_banco, margen_objetivo, cobertura, meses):
    """Utilidad bruta al mover una variable a la vez entre su mínimo y su máximo."""
    valores_trm = {"min": np.percentile(trm_final, 5), "base": np.median(trm_final), "max": np.percentile(trm_final, 95)}
    base = dict(trm=valores_trm["base"], monto=monto, diferencial=diferencial, margen_banco=margen_banco,
                margen_objetivo=margen_objetivo, cobertura=cobertura)

    def utilidad(**cambios):
        p = {**base, **cambios}
        f = precio_forward(spot, tasa_usd + p["diferencial"], tasa_usd, meses, p["margen_banco"])
        costo = p["monto"] * (p["cobertura"] * f + (1 - p["cobertura"]) * p["trm"])
        return p["monto"] * presupuesto / (1 - p["margen_objetivo"]) - costo

    escenarios = {
        "TRM al vencimiento (P5 a P95)": ("trm", valores_trm["min"], valores_trm["max"]),
        "Margen objetivo (−5 / +5 p.p.)": ("margen_objetivo", margen_objetivo - 0.05, margen_objetivo + 0.05),
        "Monto de la compra (±20 %)": ("monto", monto * 0.8, monto * 1.2),
        "Porcentaje cubierto (0 % a 100 %)": ("cobertura", 0.0, 1.0),
        "Diferencial de tasas (±2 p.p.)": ("diferencial", diferencial - 0.02, diferencial + 0.02),
        "Recargo del banco (0 % a 0,40 %)": ("margen_banco", 0.0, 0.004),
    }
    filas = []
    for nombre, (param, bajo, alto) in escenarios.items():
        u_bajo, u_alto = utilidad(**{param: bajo}), utilidad(**{param: alto})
        filas.append({"variable": nombre, "utilidad_bajo": u_bajo, "utilidad_alto": u_alto, "rango": abs(u_alto - u_bajo)})
    return pd.DataFrame(filas).sort_values("rango").reset_index(drop=True), utilidad()


def _normal(x):
    """Distribución normal estándar acumulada (sin scipy, que no se instala en Posit)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _d1_d2(forward, strike, vol, meses):
    raiz = vol * math.sqrt(meses / 12)
    d1 = (math.log(forward / strike) + raiz ** 2 / 2) / raiz
    return d1, d1 - raiz


def prima_call(forward, strike, vol, meses):
    """Prima de una call sobre el dólar por Black-76, expresada en pesos al vencimiento.

    Sobre el forward del banco equivale a Garman-Kohlhagen con el recargo incluido; llevar la prima al
    vencimiento deja todas las estrategias comparables en la misma fecha de pago.
    """
    if vol <= 0 or meses <= 0:
        return max(forward - strike, 0.0)
    d1, d2 = _d1_d2(forward, strike, vol, meses)
    return forward * _normal(d1) - strike * _normal(d2)


def prima_put(forward, strike, vol, meses):
    """Prima de una put sobre el dólar por Black-76, en pesos al vencimiento."""
    return prima_call(forward, strike, vol, meses) - (forward - strike)


def strike_put_collar(forward, strike_call, vol, meses):
    """Strike de la put vendida que financia exactamente la call comprada (collar de costo cero)."""
    objetivo = prima_call(forward, strike_call, vol, meses)
    bajo, alto = forward * 1e-3, forward
    for _ in range(200):
        medio = (bajo + alto) / 2
        if prima_put(forward, medio, vol, meses) < objetivo:
            bajo = medio
        else:
            alto = medio
    return (bajo + alto) / 2


def vol_simulada(retornos, meses):
    """Volatilidad anualizada implícita en los retornos simulados con el GJR-GARCH al plazo dado."""
    return float(np.std(retornos) / np.sqrt(meses / 12))


def comparar_instrumentos(trm_final, forward, vol, meses, cobertura, holgura_call, **argumentos):
    """Métricas de no cubrir, forward, call y collar de costo cero sobre los mismos escenarios.

    holgura_call es cuánto por encima del forward queda el strike de la call (y el techo del collar).
    """
    strike_call = forward * (1 + holgura_call)
    prima = prima_call(forward, strike_call, vol, meses)
    strike_put = strike_put_collar(forward, strike_call, vol, meses)
    por_dolar = {
        "Sin cubrir": (trm_final, np.nan, np.nan, 0.0),
        "Forward": (np.full_like(trm_final, forward, dtype=float), np.nan, np.nan, 0.0),
        "Call": (np.minimum(trm_final, strike_call) + prima, strike_call, np.nan, prima),
        "Collar": (np.clip(trm_final, strike_put, strike_call), strike_call, strike_put, 0.0),
    }
    filas = []
    for nombre, (cubierto, k_call, k_put, p) in por_dolar.items():
        h = 0.0 if nombre == "Sin cubrir" else cobertura
        costo = argumentos["monto_usd"] * (h * cubierto + (1 - h) * trm_final)
        filas.append({"instrumento": nombre, "strike_call": k_call, "strike_put": k_put, "prima": p,
                      **metricas_costo(costo, **argumentos)})
    return pd.DataFrame(filas)


def cop(valor, decimales=0):
    """Formato de pesos colombianos con punto como separador de miles."""
    texto = f"{valor:,.{decimales}f}"
    return "$ " + texto.replace(",", "X").replace(".", ",").replace("X", ".")


def millones(valor, decimales=1):
    """Cifra en millones de pesos con formato colombiano."""
    texto = f"{valor / 1e6:,.{decimales}f}"
    return "$ " + texto.replace(",", "X").replace(".", ",").replace("X", ".") + " M"


def pct(valor, decimales=1):
    """Porcentaje con coma decimal."""
    return f"{valor * 100:.{decimales}f} %".replace(".", ",")
