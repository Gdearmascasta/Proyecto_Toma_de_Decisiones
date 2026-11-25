"""Tablero interactivo de cobertura cambiaria USD/COP."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

from calculos import (cobertura_optima, comparar_instrumentos, cop, curva_coberturas, evaluar_estrategia, millones,
                      pct, precio_forward, tornado, trm_simulada, vol_simulada)

# Carga de datos
DATOS = Path(__file__).resolve().parent / "datos"
serie = pd.read_csv(DATOS / "serie_diaria.csv", parse_dates=["fecha"]).set_index("fecha")
macro = pd.read_csv(DATOS / "macro_mensual.csv", parse_dates=["fecha"]).set_index("fecha")
semanal = pd.read_csv(DATOS / "retornos_semanales.csv", parse_dates=["fecha"]).set_index("fecha")
simulaciones = pd.read_csv(DATOS / "simulaciones.csv")
historico = pd.read_csv(DATOS / "backtest.csv", parse_dates=["fecha"])
with open(DATOS / "parametros.json", encoding="utf-8") as archivo:
    P = json.load(archivo)

M = P["mercado"]
S = P["supuestos"]
SPOT = M["spot"]
RETORNOS = {3: simulaciones["r_3m"].values, 6: simulaciones["r_6m"].values}
PASOS = {3: M["pasos_3"], 6: M["pasos_6"]}
DERIVAS = {
    "aleatorio": 0.0,
    "historica": M["deriva_historica"],
    "paridad": np.log((1 + M["tasa_cop"]) / (1 + M["tasa_usd"])) / M["obs_anuales"],
}
DIAS_DATOS_VIEJOS = 14
ANIO_MIN, ANIO_MAX = serie.index.year.min(), serie.index.year.max()
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

# Paletas de color para el modo oscuro y el modo claro
PALETAS = {
    "oscuro": dict(azul="#7CC6FE", turquesa="#2EE6C5", ambar="#FFC857", rojo="#FF6B81", lila="#B79CFF",
                   texto="#EAF2FB", tenue="rgba(234,242,251,0.62)", rejilla="rgba(255,255,255,0.08)",
                   linea="rgba(255,255,255,0.3)", fondo_hover="rgba(12,24,48,0.92)"),
    "claro": dict(azul="#1F6FEB", turquesa="#0E9F8E", ambar="#D98E04", rojo="#D1495B", lila="#7C5CD6",
                  texto="#10233F", tenue="rgba(16,35,63,0.62)", rejilla="rgba(16,35,63,0.08)",
                  linea="rgba(16,35,63,0.35)", fondo_hover="rgba(255,255,255,0.96)"),
}


def colores(tema):
    """Colores de la paleta del tema elegido."""
    c = PALETAS.get(tema, PALETAS["oscuro"])
    return c["azul"], c["turquesa"], c["ambar"], c["rojo"], c["lila"], c["texto"], c["tenue"], c["rejilla"], c["linea"]


AZUL, TURQUESA, AMBAR, ROJO, LILA, TEXTO, TENUE, REJILLA, LINEA = colores("oscuro")


def estilo(fig, alto=340, leyenda=True, tema="oscuro"):
    """Aplica el estilo de vidrio a una figura de Plotly según el tema."""
    c = PALETAS.get(tema, PALETAS["oscuro"])
    TEXTO, TENUE, REJILLA = c["texto"], c["tenue"], c["rejilla"]
    fig.update_layout(
        height=alto, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, Segoe UI, sans-serif", color=TEXTO, size=12),
        hovermode="x unified", showlegend=leyenda,
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)", font=dict(color=TENUE)),
        hoverlabel=dict(bgcolor=c["fondo_hover"], bordercolor=c["linea"], font=dict(color=TEXTO)),
        separators=",.",
    )
    fig.update_xaxes(gridcolor=REJILLA, zeroline=False, color=TENUE, linecolor=REJILLA)
    fig.update_yaxes(gridcolor=REJILLA, zeroline=False, color=TENUE)
    return fig


def tarjeta(titulo, valor, detalle="", tono=""):
    """Tarjeta de KPI."""
    return html.Div(className=f"kpi {tono}", children=[
        html.Span(titulo, className="kpi-titulo"),
        html.Span(valor, className="kpi-valor"),
        html.Span(detalle, className="kpi-detalle"),
    ])


def panel(titulo, subtitulo, contenido, clase="", ayuda=""):
    """Contenedor de vidrio con título y recuadro flotante que explica la gráfica."""
    boton = html.Div(className="ayuda", tabIndex="0", children=[
        html.Span("i", className="ayuda-icono"),
        html.Div(className="ayuda-texto", children=[html.Strong("Cómo leer esta gráfica"), html.P(ayuda)]),
    ]) if ayuda else None
    return html.Div(className=f"vidrio panel {clase}", children=[
        html.Div(className="panel-cabecera", children=[
            html.Div(children=[html.H3(titulo), html.P(subtitulo)]), boton,
        ]),
        contenido,
    ])


AYUDAS = {
    "trm": "Muestra la TRM diaria con su media móvil de 60 días. La franja sombreada cubre dos desviaciones alrededor de la media y, cuando la TRM se sale de ella, hay un movimiento fuera de lo normal. La línea naranja es el Brent, que suele moverse en sentido contrario al dólar.",
    "macro": "Las líneas sólidas son las tasas de política del Banco de la República y de la Fed. La distancia entre ellas es el diferencial que encarece el forward. Las líneas punteadas muestran la inflación anual de cada país.",
    "caja_anual": "Cada caja resume los valores diarios de la TRM en un año. La línea central es la mediana, la caja contiene la mitad de los días y los bigotes llegan a los valores habituales más extremos. Una caja alta indica un año en el que el dólar recorrió mucho terreno.",
    "vol": "La línea gris es la volatilidad calculada con los últimos 21 días. La roja es la que estima el modelo GJR-GARCH día a día. Los picos son episodios de estrés, como marzo de 2020, que luego se disipan poco a poco.",
    "dispersion": "Cada punto es una semana. Si la nube baja hacia la derecha, cuando el Brent sube la TRM tiende a bajar. La pendiente de la recta y la correlación ρ resumen la fuerza de esa relación en el periodo elegido.",
    "corr": "Correlación de la TRM con cada mercado calculada con el último año de datos. Por debajo de cero se mueven en sentido contrario. Que las líneas cambien tanto muestra que estas relaciones no son estables.",
    "hist": "Las barras muestran cuántas semanas tuvieron cada retorno y la curva naranja es la normal con la misma media y desviación. Hay más semanas extremas de las que predice la normal, por eso la simulación usa una t de Student.",
    "caja_mes": "Cada caja reúne el retorno de un mismo mes en todos los años del periodo. Si un mes tuviera una caja claramente más alta convendría evitar pagar en él. Las cajas se superponen bastante, así que no aparece un mes que sea sistemáticamente más barato.",
    "calor": "Cada celda es la correlación entre los retornos semanales de dos mercados. Los tonos turquesa son relaciones positivas y los rojos negativas. La fila de la TRM indica qué mercados acompañan al dólar y cuáles van en contra.",
    "caja_retornos": "Cada caja resume los retornos diarios de la TRM en un año. Una caja más ancha y bigotes más largos significan un año más volátil y, por tanto, más riesgo para quien paga en dólares.",
    "costos": "Compara los 10.000 costos simulados sin cobertura y con el porcentaje elegido. Cubrir estrecha la distribución y recorta la cola derecha, que son los escenarios en los que la compra sale más cara.",
    "curva": "Para cada porcentaje cubierto se ven el costo esperado, el percentil 95 y la franja entre los percentiles 5 y 95. El punto marca la cobertura más barata que cumple la tolerancia de riesgo.",
    "tornado": "Cada barra muestra cuánto cambia la utilidad bruta al mover una sola variable entre su valor bajo y su valor alto, con las demás fijas. El rojo es el escenario desfavorable y el turquesa el favorable. Las barras más largas, arriba, son las que más pesan en el resultado.",
    "caja_costos": "Cada caja resume el costo simulado de la compra con un porcentaje cubierto distinto. A medida que se cubre más, la caja se encoge y se acerca al precio del forward. La línea amarilla es el presupuesto.",
    "backtest": "Cada barra es un mes desde 2015. En verde, cubrir con forward habría salido más barato que pagar a la TRM del vencimiento, y en rojo habría salido más caro. Con la estrategia escalonada el pago se cubre en tres tramos iguales a lo largo del plazo. Al pasar el cursor se ven la TRM inicial, el forward y la TRM final.",
    "instrumentos": "Compara no cubrir con cubrir el porcentaje elegido mediante un forward, una call o un collar de costo cero, sobre los mismos 10.000 escenarios. El P95 es el costo que solo se supera en el 5 % de los escenarios y el CVaR es el promedio de ese peor 5 %. La call pone un techo a la tasa y conserva la ganancia si el dólar baja, a cambio de una prima. El collar fija la tasa entre un piso y un techo sin pagar prima.",
}


def control(etiqueta, componente, ayuda=""):
    """Bloque de control con etiqueta."""
    return html.Div(className="control", children=[
        html.Label(etiqueta), componente, html.Small(ayuda) if ayuda else None,
    ])


# KPI de contexto calculados con los datos
variacion_30 = SPOT / serie["trm"].asof(serie.index.max() - pd.Timedelta(days=30)) - 1
forward_6 = precio_forward(SPOT, M["tasa_cop"], M["tasa_usd"], 6, S["margen_banco"])
fecha_texto = pd.Timestamp(M["fecha_decision"]).strftime("%d/%m/%Y")

app = Dash(__name__, title="Cobertura del riesgo cambiario", update_title=None,
           external_stylesheets=["https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"])
server = app.server



def cabecera():
    """Encabezado con la antigüedad de los datos calculada en cada carga de la página."""
    dias = (pd.Timestamp.today().normalize() - pd.Timestamp(M["fecha_decision"])).days
    viejos = dias > DIAS_DATOS_VIEJOS
    antiguedad = "Actualizados hoy" if dias <= 0 else f"Hace {dias} día" + ("s" if dias != 1 else "")
    return html.Header(className="cabecera", children=[
        html.Div(children=[
            html.Span("USD / COP · Importadora de Cartagena", className="sobretitulo"),
            html.H1("Cobertura del riesgo cambiario en importaciones"),
            html.P("Evaluación de contratos forward a 3 y 6 meses para una importadora de Cartagena."),
        ]),
        html.Div(className="cabecera-derecha", children=[
            html.Button(id="boton-tema", className="boton-tema vidrio", n_clicks=0, title="Cambiar entre modo claro y oscuro",
                        children=[html.Span("☀", className="icono-sol"), html.Span("☾", className="icono-luna"),
                                  html.Span(id="texto-tema", children="Modo claro")]),
            html.Div(className="sello vidrio" + (" viejo" if viejos else ""), title=(
                "Los datos tienen más de dos semanas: conviene volver a descargarlos y ejecutar el notebook" if viejos else None),
                children=[
                    html.Span("Datos al", className="kpi-titulo"), html.Strong(fecha_texto),
                    html.Span(f"TRM {cop(SPOT, 2)}", className="kpi-detalle"),
                    html.Span(antiguedad + (" · desactualizados" if viejos else ""), className="antiguedad"),
                ]),
        ]),
    ])


contexto_caja = html.Div(className="vidrio contexto", children=[
    html.Div(className="contexto-icono", children="i"),
    html.Div(children=[
        html.H4("Contexto de la problemática"),
        html.P([
            "La empresa fija hoy sus precios de venta en pesos, pero paga la mercancía en dólares meses después. "
            "Si el dólar sube en ese tiempo, el margen se reduce. Un ", html.B("forward"),
            " permite fijar hoy la tasa de pago, pero cuesta el diferencial de tasas entre Colombia y EE. UU. "
            "Este tablero va del contexto al riesgo y termina en la decisión, con 10.000 escenarios simulados del dólar.",
        ]),
    ]),
])

kpis_contexto = html.Div(className="kpis", children=[
    tarjeta("TRM vigente", cop(SPOT, 2), f"{pct(variacion_30)} en 30 días", "sube" if variacion_30 > 0 else "baja"),
    tarjeta("Presupuesto", cop(M["presupuesto"], 2), "Promedio TRM 12 meses"),
    tarjeta("Forward 6 meses", cop(forward_6, 2), f"{pct(forward_6 / SPOT - 1)} sobre la TRM"),
    tarjeta("Volatilidad GARCH", pct(P["garch"]["vol_actual"]), f"Histórica {pct(P['eda']['vol_anual'])}"),
    tarjeta("Diferencial de tasas", f"{(M['tasa_cop'] - M['tasa_usd']) * 100:.2f} p.p.".replace(".", ",", 1), f"BanRep {pct(M['tasa_cop'], 2)} · Fed {pct(M['tasa_usd'], 2)}"),
    tarjeta("Brent", f"US$ {M['brent']:.2f}".replace(".", ","), f"Correlación semanal {P['eda']['corr_brent']:.2f}".replace(".", ",")),
])

filtro_fechas = html.Div(className="vidrio filtro", children=[
    html.Span("Periodo de las gráficas de contexto y riesgo", className="filtro-etiqueta"),
    dcc.RangeSlider(id="rango-anios", min=ANIO_MIN, max=ANIO_MAX, step=1, value=[ANIO_MIN, ANIO_MAX],
                    marks={a: str(a) for a in range(ANIO_MIN, ANIO_MAX + 1)}, allowCross=False, allow_direct_input=False),
])

pestana_contexto = html.Div(children=[
    html.Div(className="rejilla dos", children=[
        panel("TRM diaria y Brent", "Media móvil de 60 días, banda de dos desviaciones y Brent en el eje derecho",
              html.Div(children=[
                  dcc.Checklist(id="ver-brent", options=[{"label": " Mostrar Brent", "value": "si"}], value=["si"],
                                className="interruptor"),
                  dcc.Graph(id="g-trm", config={"displayModeBar": False}),
              ]), "ancho", AYUDAS["trm"]),
        panel("Tasas e inflación", "Tasas de política e inflación anual de Colombia y EE. UU.",
              dcc.Graph(id="g-macro", config={"displayModeBar": False}), ayuda=AYUDAS["macro"]),
    ]),
    panel("Distribución anual de la TRM", "TRM diaria por año, el año en curso en ámbar",
          dcc.Graph(id="g-caja-anual", config={"displayModeBar": False}), ayuda=AYUDAS["caja_anual"]),
])

pestana_riesgo = html.Div(children=[
    html.Div(className="rejilla tres", children=[
        panel("Volatilidad anualizada", "Móvil de 21 días y GJR-GARCH condicional",
              dcc.Graph(id="g-vol", config={"displayModeBar": False}), "ancho", AYUDAS["vol"]),
        panel("TRM frente al Brent", "Retornos semanales y recta de regresión",
              dcc.Graph(id="g-dispersion", config={"displayModeBar": False}), ayuda=AYUDAS["dispersion"]),
    ]),
    html.Div(className="rejilla tres", children=[
        panel("Correlación móvil de 52 semanas", "TRM frente a Brent, índice dólar y S&P 500",
              dcc.Graph(id="g-corr", config={"displayModeBar": False}), "ancho", AYUDAS["corr"]),
        panel("Distribución de retornos semanales", "Retornos de la TRM frente a una normal",
              dcc.Graph(id="g-hist", config={"displayModeBar": False}), ayuda=AYUDAS["hist"]),
    ]),
    html.Div(className="rejilla tres", children=[
        panel("Retorno mensual por mes del año", "Retornos mensuales de la TRM agrupados por mes",
              dcc.Graph(id="g-caja-mes", config={"displayModeBar": False}), "ancho", AYUDAS["caja_mes"]),
        panel("Correlación entre mercados", "Retornos semanales del periodo elegido",
              dcc.Graph(id="g-calor", config={"displayModeBar": False}), ayuda=AYUDAS["calor"]),
    ]),
    panel("Retornos diarios por año", "Distribución de los retornos diarios de la TRM",
          dcc.Graph(id="g-caja-retornos", config={"displayModeBar": False}), ayuda=AYUDAS["caja_retornos"]),
])

opciones_deriva = [
    {"label": "Paseo aleatorio", "value": "aleatorio"},
    {"label": "Tendencia histórica", "value": "historica"},
    {"label": "Paridad de tasas", "value": "paridad"},
]

controles = html.Div(className="vidrio controles", children=[
    html.H3("Supuestos de la operación"),
    control("Plazo del pago", dcc.RadioItems(id="plazo", options=[{"label": "3 meses", "value": 3}, {"label": "6 meses", "value": 6}],
                                              value=6, className="pastillas", inline=True)),
    control("Porcentaje cubierto", dcc.Slider(id="cobertura", min=0, max=100, step=5, value=round(P["optimas"]["6"]["cobertura"] * 100),
                                              marks={0: "0 %", 25: "25 %", 50: "50 %", 75: "75 %", 100: "100 %"},
                                              tooltip={"placement": "bottom", "always_visible": False, "transform": "formatoPct"}, allow_direct_input=False)),
    control("Monto de la compra (USD)", dcc.Slider(id="monto", min=50_000, max=1_000_000, step=10_000, value=S["monto_usd"],
                                                   marks={50_000: "50 mil", 250_000: "250 mil", 500_000: "500 mil", 1_000_000: "1 millón"},
                                                   tooltip={"placement": "bottom", "always_visible": False, "transform": "formatoUSD"}, allow_direct_input=False)),
    control("TRM presupuestada", dcc.Input(id="presupuesto", type="number", value=round(M["presupuesto"], 2), step=0.01,
                                           className="entrada"), "Por defecto, promedio de los últimos 12 meses"),
    html.Div(className="par", children=[
        control("Tasa COP (%)", dcc.Input(id="tasa-cop", type="number", value=round(M["tasa_cop"] * 100, 3), step=0.001, className="entrada")),
        control("Tasa USD (%)", dcc.Input(id="tasa-usd", type="number", value=round(M["tasa_usd"] * 100, 3), step=0.001, className="entrada")),
    ]),
    control("Recargo del banco (%)", dcc.Slider(id="recargo", min=0, max=0.6, step=0.05, value=S["margen_banco"] * 100,
                                                marks={0: "0", 0.15: "0,15", 0.3: "0,30", 0.6: "0,60"}, allow_direct_input=False)),
    html.Div(className="par", children=[
        control("Margen objetivo (%)", dcc.Input(id="margen-obj", type="number", value=S["margen_objetivo"] * 100, step=1, className="entrada")),
        control("Margen mínimo (%)", dcc.Input(id="margen-min", type="number", value=S["margen_minimo"] * 100, step=1, className="entrada")),
    ]),
    control("Tolerancia al riesgo (%)", dcc.Slider(id="tolerancia", min=1, max=20, step=1, value=S["tolerancia_riesgo"] * 100,
                                                   marks={1: "1", 5: "5", 10: "10", 20: "20"}, allow_direct_input=False),
            "Probabilidad máxima aceptada de quedar bajo el margen mínimo"),
    control("Qué esperar del dólar", dcc.RadioItems(id="deriva", options=opciones_deriva, value="aleatorio", className="pastillas apiladas")),
    control("Strike de la call (% sobre el forward)", dcc.Slider(id="holgura", min=0, max=10, step=0.5,
                                                                value=P["opciones"]["holgura_call"] * 100,
                                                                marks={0: "0", 3: "3", 5: "5", 10: "10"}, allow_direct_input=False),
            "También es el techo del collar de costo cero"),
    html.Button("Descargar escenario (CSV)", id="boton-descarga", className="boton-descarga vidrio", n_clicks=0),
    dcc.Download(id="descarga"),
])

pestana_decision = html.Div(className="decision", children=[
    controles,
    html.Div(className="resultados", children=[
        html.Div(id="kpis-decision", className="kpis"),
        html.Div(id="recomendacion", className="vidrio recomendacion"),
        html.Div(className="rejilla dos-iguales", children=[
            panel("Distribución del costo", "Sin cobertura y con la cobertura elegida",
                  dcc.Graph(id="g-costos", config={"displayModeBar": False}), ayuda=AYUDAS["costos"]),
            panel("Costo y riesgo por nivel de cobertura", "Costo esperado, percentil 95 y cobertura sugerida",
                  dcc.Graph(id="g-curva", config={"displayModeBar": False}), ayuda=AYUDAS["curva"]),
        ]),
        html.Div(className="rejilla dos-iguales", children=[
            panel("Análisis de sensibilidad", "Utilidad bruta al mover una variable a la vez",
                  dcc.Graph(id="g-tornado", config={"displayModeBar": False}), ayuda=AYUDAS["tornado"]),
            panel("Costo por nivel de cobertura", "10.000 escenarios por cada porcentaje cubierto",
                  dcc.Graph(id="g-caja-costos", config={"displayModeBar": False}), ayuda=AYUDAS["caja_costos"]),
        ]),
        panel("Forward, call y collar de costo cero", "Costo esperado, P95 y CVaR 95 % con el porcentaje cubierto elegido",
              html.Div(children=[html.P(id="texto-instrumentos", className="nota"),
                                 dcc.Graph(id="g-instrumentos", config={"displayModeBar": False})]),
              ayuda=AYUDAS["instrumentos"]),
    ]),
])

pestana_backtest = html.Div(children=[
    html.Div(className="vidrio filtro", children=[
        html.Span("Plazo", className="filtro-etiqueta"),
        dcc.RadioItems(id="plazo-bt", options=[{"label": "3 meses", "value": 3}, {"label": "6 meses", "value": 6}],
                       value=6, className="pastillas", inline=True),
        html.Span("Estrategia", className="filtro-etiqueta"),
        dcc.RadioItems(id="estrategia-bt", options=[{"label": "Todo al inicio", "value": "todo"},
                                                    {"label": "Escalonada en 3 tramos", "value": "escalonada"}],
                       value="todo", className="pastillas", inline=True),
    ]),
    html.Div(id="kpis-backtest", className="kpis"),
    panel("Backtest de la cobertura, 2015-2026",
          "Ahorro por dólar de cubrir al 100 % frente a pagar a la TRM del vencimiento (verde = cubrir salió más barato)",
          dcc.Graph(id="g-backtest", config={"displayModeBar": False}), ayuda=AYUDAS["backtest"]),
])



def construir_layout():
    """Página completa; Dash la reconstruye en cada carga para que la antigüedad de los datos esté al día."""
    return html.Div(id="raiz", className="fondo", children=[
        dcc.Store(id="tema", storage_type="local", data="oscuro"),
        html.Div(className="mancha m1"), html.Div(className="mancha m2"), html.Div(className="mancha m3"),
        html.Main(className="contenedor", children=[
            cabecera(),
            contexto_caja,
            kpis_contexto,
            filtro_fechas,
            dcc.Tabs(id="pestanas", value="decision", className="pestanas", children=[
                dcc.Tab(label="1 · Contexto", value="contexto", className="pestana", selected_className="pestana-activa",
                        children=[pestana_contexto]),
                dcc.Tab(label="2 · Riesgo", value="riesgo", className="pestana", selected_className="pestana-activa",
                        children=[pestana_riesgo]),
                dcc.Tab(label="3 · Decisión", value="decision", className="pestana", selected_className="pestana-activa",
                        children=[pestana_decision]),
                dcc.Tab(label="4 · Backtest", value="backtest", className="pestana", selected_className="pestana-activa",
                        children=[pestana_backtest]),
            ]),
            html.Footer(className="pie", children=[
                html.P("Fuentes: Superintendencia Financiera (TRM, datos.gov.co), Yahoo Finance (Brent, índice dólar, S&P 500), "
                       "Banco de Pagos Internacionales (tasas de política e inflación). Simulación GJR-GARCH con 10.000 escenarios y semilla fija."),
            ]),
        ]),
    ])


app.layout = construir_layout


@app.callback(Output("tema", "data"), Input("boton-tema", "n_clicks"), State("tema", "data"), prevent_initial_call=True)
def cambiar_tema(clics, tema):
    """Alterna entre el modo oscuro y el modo claro."""
    return "claro" if tema == "oscuro" else "oscuro"


@app.callback(Output("raiz", "className"), Output("texto-tema", "children"), Input("tema", "data"))
def aplicar_tema(tema):
    """Aplica la clase del tema a toda la página."""
    if tema == "claro":
        return "fondo claro", "Modo oscuro"
    return "fondo", "Modo claro"


def filtrar(df, rango):
    """Recorta una tabla al rango de años elegido."""
    return df.loc[str(rango[0]):str(rango[1])]


@app.callback(Output("g-trm", "figure"), Output("g-macro", "figure"), Output("g-caja-anual", "figure"),
              Input("rango-anios", "value"), Input("ver-brent", "value"), Input("tema", "data"))
def actualizar_contexto(rango, ver_brent, tema="oscuro"):
    """Gráficas de la pestaña de contexto."""
    AZUL, TURQUESA, AMBAR, ROJO, LILA, TEXTO, TENUE, REJILLA, LINEA = colores(tema)
    d = filtrar(serie, rango)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["banda_superior"], line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=d.index, y=d["banda_inferior"], fill="tonexty", fillcolor="rgba(46,230,197,0.12)",
                             line=dict(width=0), name="Banda ±2σ", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d.index, y=d["media_movil"], name="Media 60 días", line=dict(color=TURQUESA, width=1.5),
                             hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=d.index, y=d["trm"], name="TRM", line=dict(color=AZUL, width=1.8), hovertemplate="$ %{y:,.2f}"))
    if "si" in (ver_brent or []):
        fig.add_trace(go.Scatter(x=d.index, y=d["brent"], name="Brent (US$)", yaxis="y2",
                                 line=dict(color=AMBAR, width=0.9), opacity=0.5, hovertemplate="US$ %{y:.1f}"))
    fig.update_layout(yaxis=dict(title="COP por USD", tickformat=",.0f"),
                      yaxis2=dict(overlaying="y", side="right", showgrid=False, color=AMBAR, title="US$ por barril", tickformat=".0f", nticks=6))
    estilo(fig, 380, tema=tema)

    m = filtrar(macro, rango)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=m.index, y=m["tasa_banrep"], name="Tasa BanRep", line=dict(color=TURQUESA, width=2, shape="hv")))
    fig2.add_trace(go.Scatter(x=m.index, y=m["tasa_fed"], name="Tasa Fed", line=dict(color=LILA, width=2, shape="hv")))
    fig2.add_trace(go.Scatter(x=m.index, y=m["inflacion_col"], name="Inflación Col.", line=dict(color=AMBAR, width=1.3, dash="dot")))
    fig2.add_trace(go.Scatter(x=m.index, y=m["inflacion_eeuu"], name="Inflación EE. UU.", line=dict(color=ROJO, width=1.3, dash="dot")))
    fig2.update_yaxes(ticksuffix=" %")
    fig2.update_traces(hovertemplate="%{y:.2f} %")

    fig3 = go.Figure()
    for anio, grupo in d.groupby(d.index.year):
        color = AMBAR if anio == ANIO_MAX else TURQUESA
        fig3.add_trace(go.Box(y=grupo["trm"], name=str(anio), marker=dict(color=color, size=3), line=dict(width=1.4),
                              boxpoints="outliers",
                              hovertemplate="%{y:,.0f}<extra></extra>"))
    fig3.update_traces(fillcolor="rgba(46,230,197,0.18)", selector=dict(type="box"))
    fig3.update_traces(fillcolor="rgba(255,200,87,0.25)", selector=dict(name=str(ANIO_MAX)))
    fig3.update_yaxes(title="COP por USD", tickformat=",.0f")
    fig3.update_layout(hovermode="closest")
    return fig, estilo(fig2, 380, tema=tema), estilo(fig3, 320, leyenda=False, tema=tema)


@app.callback(Output("g-vol", "figure"), Output("g-dispersion", "figure"), Output("g-corr", "figure"), Output("g-hist", "figure"),
              Output("g-caja-mes", "figure"), Output("g-calor", "figure"), Output("g-caja-retornos", "figure"),
              Input("rango-anios", "value"), Input("tema", "data"))
def actualizar_riesgo(rango, tema="oscuro"):
    """Gráficas de la pestaña de riesgo."""
    AZUL, TURQUESA, AMBAR, ROJO, LILA, TEXTO, TENUE, REJILLA, LINEA = colores(tema)
    d = filtrar(serie, rango)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["vol_21d"], name="Móvil 21 días", line=dict(color=TENUE, width=1)))
    fig.add_trace(go.Scatter(x=d.index, y=d["vol_garch"], name="GJR-GARCH", line=dict(color=ROJO, width=1.8),
                             fill="tozeroy", fillcolor="rgba(255,107,129,0.10)"))
    fig.update_yaxes(tickformat=".0%")
    fig.update_traces(hovertemplate="%{y:.1%}")

    w = filtrar(semanal, rango)
    pendiente, intercepto = np.polyfit(w["brent"], w["trm"], 1)
    x = np.linspace(w["brent"].min(), w["brent"].max(), 50)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=w["brent"], y=w["trm"], mode="markers", name="Semanas",
                              marker=dict(color=AZUL, size=6, opacity=0.55, line=dict(width=0)),
                              hovertemplate="Brent %{x:.1%}<br>TRM %{y:.1%}<extra></extra>"))
    fig2.add_trace(go.Scatter(x=x, y=intercepto + pendiente * x, name=f"Pendiente {pendiente:.3f}".replace(".", ","),
                              line=dict(color=AMBAR, width=2.5), hoverinfo="skip"))
    fig2.update_xaxes(tickformat=".0%", title="Retorno Brent")
    fig2.update_yaxes(tickformat=".0%", title="Retorno TRM")
    fig2.update_layout(hovermode="closest")
    corr_texto = f"ρ = {w['trm'].corr(w['brent']):.2f}".replace(".", ",")
    fig2.add_annotation(text=corr_texto, xref="paper", yref="paper", x=0.98, y=0.95, showarrow=False,
                        font=dict(size=15, color=AMBAR))

    c = pd.DataFrame({"Brent": semanal["trm"].rolling(52).corr(semanal["brent"]),
                      "Índice dólar": semanal["trm"].rolling(52).corr(semanal["indice_dolar"]),
                      "S&P 500": semanal["trm"].rolling(52).corr(semanal["sp500"])}).dropna()
    c = filtrar(c, rango)
    fig3 = go.Figure()
    for nombre, color in zip(c.columns, (AMBAR, TURQUESA, LILA)):
        fig3.add_trace(go.Scatter(x=c.index, y=c[nombre], name=nombre, line=dict(color=color, width=1.8), hovertemplate="%{y:.2f}"))
    fig3.add_hline(y=0, line=dict(color=LINEA, width=1))

    r = w["trm"]
    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(x=r, nbinsx=50, histnorm="probability density", name="Observado",
                                marker=dict(color=AZUL, line=dict(width=0)), opacity=0.75))
    xs = np.linspace(r.min(), r.max(), 200)
    normal = np.exp(-0.5 * ((xs - r.mean()) / r.std()) ** 2) / (r.std() * np.sqrt(2 * np.pi))
    fig4.add_trace(go.Scatter(x=xs, y=normal, name="Normal", line=dict(color=AMBAR, width=2)))
    fig4.update_xaxes(tickformat=".0%")
    fig4.update_layout(hovermode="closest", bargap=0.03)

    mensual = np.log(d["trm"].resample("ME").last()).diff().dropna()
    fig5 = go.Figure()
    for numero, nombre in enumerate(MESES, start=1):
        valores = mensual[mensual.index.month == numero]
        fig5.add_trace(go.Box(y=valores, name=nombre, marker=dict(color=AZUL, size=4), line=dict(width=1.4, color=AZUL),
                              fillcolor="rgba(124,198,254,0.18)", boxpoints="all", jitter=0.4, pointpos=0,
                              customdata=valores.index.year, hovertemplate="%{customdata}: %{y:.1%}<extra></extra>"))
    fig5.add_hline(y=0, line=dict(color=LINEA, width=1))
    fig5.update_yaxes(tickformat=".0%", title="Retorno mensual")
    fig5.update_layout(hovermode="closest")

    nombres = {"trm": "TRM", "brent": "Brent", "wti": "WTI", "indice_dolar": "Índice dólar",
               "ecopetrol_adr": "Ecopetrol", "sp500": "S&P 500", "oro": "Oro"}
    matriz = w[list(nombres)].rename(columns=nombres).corr()
    fig6 = go.Figure(go.Heatmap(z=matriz.values, x=matriz.columns, y=matriz.index, zmin=-1, zmax=1,
                                colorscale=[[0, ROJO], [0.5, REJILLA], [1, TURQUESA]],
                                texttemplate="%{z:.2f}", textfont=dict(size=11, color=TEXTO), xgap=3, ygap=3,
                                hovertemplate="%{y} y %{x}: %{z:.2f}<extra></extra>", colorbar=dict(thickness=10, outlinewidth=0)))
    fig6.update_yaxes(autorange="reversed", showgrid=False)
    fig6.update_xaxes(showgrid=False, tickangle=-35)
    fig6.update_layout(hovermode="closest")

    diarios = np.log(d["trm"]).diff().dropna()
    fig7 = go.Figure()
    for anio, grupo in diarios.groupby(diarios.index.year):
        fig7.add_trace(go.Box(y=grupo, name=str(anio), marker=dict(color=LILA, size=3), line=dict(width=1.3, color=LILA),
                              fillcolor="rgba(183,156,255,0.18)", boxpoints="outliers", hovertemplate="%{y:.2%}<extra></extra>"))
    fig7.update_yaxes(tickformat=".1%", title="Retorno diario")
    fig7.update_layout(hovermode="closest")
    return (estilo(fig, 320, tema=tema), estilo(fig2, 320, tema=tema), estilo(fig3, 300, tema=tema), estilo(fig4, 300, tema=tema),
            estilo(fig5, 330, leyenda=False, tema=tema), estilo(fig6, 330, leyenda=False, tema=tema), estilo(fig7, 320, leyenda=False, tema=tema))


ENTRADAS_DECISION = ["plazo", "cobertura", "monto", "presupuesto", "tasa-cop", "tasa-usd", "recargo", "margen-obj",
                     "margen-min", "tolerancia", "deriva", "holgura"]


def escenario(plazo, cobertura, monto, presupuesto, tasa_cop, tasa_usd, recargo, margen_obj, margen_min, tolerancia, deriva,
              holgura):
    """Supuestos normalizados y cálculos de la pestaña de decisión, compartidos por las gráficas y la descarga."""
    presupuesto = presupuesto or M["presupuesto"]
    tasa_cop = (tasa_cop if tasa_cop is not None else M["tasa_cop"] * 100) / 100
    tasa_usd = (tasa_usd if tasa_usd is not None else M["tasa_usd"] * 100) / 100
    margen_obj = min(max((margen_obj or 25) / 100, 0.01), 0.9)
    margen_min = min(max((margen_min or 15) / 100, 0.0), margen_obj)
    h = cobertura / 100
    forward = precio_forward(SPOT, tasa_cop, tasa_usd, plazo, recargo / 100)
    trm_final = trm_simulada(RETORNOS[plazo], SPOT, DERIVAS[deriva], PASOS[plazo])
    argumentos = dict(monto_usd=monto, presupuesto=presupuesto, margen_objetivo=margen_obj, margen_minimo=margen_min)
    curva = curva_coberturas(trm_final, forward, **argumentos)
    return dict(
        plazo=plazo, h=h, monto=monto, presupuesto=presupuesto, tasa_cop=tasa_cop, tasa_usd=tasa_usd, recargo=recargo / 100,
        margen_obj=margen_obj, margen_min=margen_min, tolerancia=tolerancia / 100, deriva=deriva, holgura=holgura / 100,
        forward=forward, trm_final=trm_final, vol=vol_simulada(RETORNOS[plazo], plazo),
        elegida=evaluar_estrategia(trm_final, forward, h, **argumentos),
        sin_cubrir=evaluar_estrategia(trm_final, forward, 0.0, **argumentos),
        curva=curva, optima=cobertura_optima(curva, tolerancia / 100),
        instrumentos=comparar_instrumentos(trm_final, forward, vol_simulada(RETORNOS[plazo], plazo), plazo, h, holgura / 100,
                                           **argumentos),
    )


@app.callback(
    Output("kpis-decision", "children"), Output("recomendacion", "children"), Output("g-costos", "figure"),
    Output("g-curva", "figure"), Output("g-tornado", "figure"), Output("g-caja-costos", "figure"),
    Output("g-instrumentos", "figure"), Output("texto-instrumentos", "children"),
    *[Input(nombre, "value") for nombre in ENTRADAS_DECISION], Input("tema", "data"),
)
def actualizar_decision(plazo, cobertura, monto, presupuesto, tasa_cop, tasa_usd, recargo, margen_obj, margen_min, tolerancia, deriva,
                        holgura, tema="oscuro"):
    """Simulación, métricas y recomendación para los supuestos elegidos."""
    AZUL, TURQUESA, AMBAR, ROJO, LILA, TEXTO, TENUE, REJILLA, LINEA = colores(tema)
    e = escenario(plazo, cobertura, monto, presupuesto, tasa_cop, tasa_usd, recargo, margen_obj, margen_min, tolerancia, deriva,
                  holgura)
    presupuesto, tasa_cop, tasa_usd, margen_obj, margen_min = (e["presupuesto"], e["tasa_cop"], e["tasa_usd"], e["margen_obj"],
                                                               e["margen_min"])
    h, forward, trm_final, curva, optima = e["h"], e["forward"], e["trm_final"], e["curva"], e["optima"]
    elegida, sin_cubrir, instrumentos = e["elegida"], e["sin_cubrir"], e["instrumentos"]

    ahorro = sin_cubrir["costo_esperado"] - elegida["costo_esperado"]
    call = instrumentos.set_index("instrumento").loc["Call"]
    kpis = [
        tarjeta(f"Forward {plazo} meses", cop(forward, 2), f"{pct(forward / SPOT - 1)} sobre la TRM"),
        tarjeta("Costo esperado", millones(elegida["costo_esperado"]), f"Sin cubrir {millones(sin_cubrir['costo_esperado'])}"),
        tarjeta("Costo P95 (CFaR)", millones(elegida["costo_p95"]), f"Sin cubrir {millones(sin_cubrir['costo_p95'])}", "alerta"),
        tarjeta("CVaR 95 %", millones(elegida["cvar95"]), f"Promedio del peor 5 %; sin cubrir {millones(sin_cubrir['cvar95'])}",
                "alerta"),
        tarjeta("P(superar presupuesto)", pct(elegida["prob_superar_presupuesto"]), f"Sin cubrir {pct(sin_cubrir['prob_superar_presupuesto'])}"),
        tarjeta("P(margen < mínimo)", pct(elegida["prob_margen_bajo"]), f"Tolerancia {tolerancia:.0f} %",
                "baja" if elegida["prob_margen_bajo"] <= tolerancia / 100 else "sube"),
        tarjeta("Ahorro esperado vs no cubrir", millones(ahorro), "Negativo = prima pagada por cubrir",
                "baja" if ahorro >= 0 else "neutro"),
        tarjeta("Prima de la call", cop(call["prima"], 1), f"Por dólar, strike {cop(call['strike_call'], 0)}"),
    ]

    opt_h = optima["cobertura"]
    cumple = optima["prob_margen_bajo"] <= tolerancia / 100
    texto = [
        html.Div(className="reco-numero", children=[html.Span("Cobertura sugerida"), html.Strong(pct(opt_h, 0))]),
        html.Div(className="reco-texto", children=[
            html.H4("Interpretación de los resultados"),
            html.P(
                f"Con pago a {plazo} meses, la regla busca la cobertura más barata que deje la probabilidad de un margen "
                f"inferior al {pct(margen_min, 0)} en {tolerancia:.0f} % o menos. "
                + (f"Cubrir el {pct(opt_h, 0)} lleva esa probabilidad a {pct(optima['prob_margen_bajo'])} "
                   f"(sin cubrir es {pct(sin_cubrir['prob_margen_bajo'])}) y recorta el costo en el peor 5 % de los escenarios de "
                   f"{millones(sin_cubrir['costo_p95'])} a {millones(optima['costo_p95'])}, a cambio de "
                   f"{millones(optima['costo_esperado'] - sin_cubrir['costo_esperado'])} más de costo esperado."
                   if cumple and opt_h > 0 else
                   f"Sin cubrir esa probabilidad ya es de {pct(sin_cubrir['prob_margen_bajo'])}, dentro de la tolerancia, "
                   f"así que no hace falta cubrir. Cada punto cubierto agrega costo esperado porque el forward "
                   f"está {pct(forward / SPOT - 1)} por encima de la TRM de hoy."
                   if cumple else
                   f"Ninguna cobertura cumple la tolerancia. La que más reduce el riesgo es {pct(opt_h, 0)}.")
            ),
        ]),
    ]

    costo_sin = monto * trm_final
    costo_con = monto * (h * forward + (1 - h) * trm_final)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=costo_sin / 1e6, name="Sin cubrir", nbinsx=70, marker=dict(color=ROJO, line=dict(width=0)), opacity=0.55))
    fig.add_trace(go.Histogram(x=costo_con / 1e6, name=f"Cubierto {cobertura} %", nbinsx=70, marker=dict(color=TURQUESA, line=dict(width=0)), opacity=0.7))
    fig.add_vline(x=monto * presupuesto / 1e6, line=dict(color=AMBAR, width=2, dash="dash"),
                  annotation_text="Presupuesto", annotation_font_color=AMBAR)
    fig.update_layout(barmode="overlay", hovermode="closest", bargap=0.02)
    fig.update_xaxes(title="Costo de la compra (millones de COP)", tickformat=",.0f")
    fig.update_yaxes(title="Escenarios")

    fig2 = go.Figure()
    x = curva["cobertura"] * 100
    fig2.add_trace(go.Scatter(x=x, y=curva["costo_p95"] / 1e6, line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig2.add_trace(go.Scatter(x=x, y=curva["costo_p5"] / 1e6, fill="tonexty", fillcolor="rgba(124,198,254,0.12)",
                              line=dict(width=0), name="Rango P5-P95", hoverinfo="skip"))
    fig2.add_trace(go.Scatter(x=x, y=curva["costo_esperado"] / 1e6, name="Costo esperado", line=dict(color=AZUL, width=3),
                              hovertemplate="%{y:,.1f} M"))
    fig2.add_trace(go.Scatter(x=x, y=curva["costo_p95"] / 1e6, name="P95", line=dict(color=ROJO, width=2, dash="dash"),
                              hovertemplate="%{y:,.1f} M"))
    fig2.add_hline(y=monto * presupuesto / 1e6, line=dict(color=AMBAR, width=1.5))
    fig2.add_trace(go.Scatter(x=[opt_h * 100], y=[optima["costo_p95"] / 1e6], mode="markers", name="Sugerida",
                              marker=dict(color=TURQUESA, size=15, line=dict(color="white", width=2)), hovertemplate="%{x:.0f} %"))
    fig2.update_xaxes(title="Porcentaje cubierto", ticksuffix=" %")
    fig2.update_yaxes(title="Millones de COP", tickformat=",.0f")

    tabla, base = tornado(SPOT, tasa_usd, tasa_cop - tasa_usd, presupuesto, trm_final, monto, recargo / 100, margen_obj, h, plazo)
    fig3 = go.Figure()
    menores = tabla[["utilidad_bajo", "utilidad_alto"]].min(axis=1)
    mayores = tabla[["utilidad_bajo", "utilidad_alto"]].max(axis=1)
    fig3.add_trace(go.Bar(y=tabla["variable"], x=(base - menores) / 1e6, base=menores / 1e6, orientation="h", name="Escenario desfavorable",
                          marker=dict(color=ROJO), hovertemplate="%{base:,.1f} M<extra></extra>"))
    fig3.add_trace(go.Bar(y=tabla["variable"], x=(mayores - base) / 1e6, base=base / 1e6, orientation="h", name="Escenario favorable",
                          marker=dict(color=TURQUESA), hovertemplate="%{x:,.1f} M sobre la base<extra></extra>"))
    fig3.add_vline(x=base / 1e6, line=dict(color=TEXTO, width=1.5))
    fig3.update_layout(barmode="overlay", hovermode="closest")
    fig3.update_xaxes(title="Utilidad bruta (millones de COP)", tickformat=",.0f")

    fig4 = go.Figure()
    niveles = sorted(set([0, 25, 50, 75, 100, cobertura]))
    for nivel in niveles:
        elegido = nivel == cobertura
        color = TURQUESA if elegido else AZUL
        costo = monto * (nivel / 100 * forward + (1 - nivel / 100) * trm_final) / 1e6
        fig4.add_trace(go.Box(y=costo, name=f"{nivel} %" + (" ✓" if elegido else ""), boxpoints=False,
                              marker=dict(color=color), line=dict(color=color, width=2 if elegido else 1.3),
                              fillcolor="rgba(46,230,197,0.28)" if elegido else "rgba(124,198,254,0.12)",
                              hovertemplate="%{y:,.1f} M<extra></extra>"))
    fig4.add_hline(y=monto * presupuesto / 1e6, line=dict(color=AMBAR, width=1.5, dash="dash"),
                   annotation_text="Presupuesto", annotation_font_color=AMBAR, annotation_position="top left")
    fig4.update_yaxes(title="Costo (millones de COP)", tickformat=",.0f")
    fig4.update_xaxes(title="Porcentaje cubierto")
    fig4.update_layout(hovermode="closest")

    fig5 = go.Figure()
    for metrica, nombre, color in (("costo_esperado", "Costo esperado", AZUL), ("costo_p95", "P95 (CFaR)", AMBAR),
                                   ("cvar95", "CVaR 95 %", ROJO)):
        fig5.add_trace(go.Scatter(x=instrumentos["instrumento"], y=instrumentos[metrica] / 1e6, name=nombre, mode="markers",
                                  marker=dict(color=color, size=14, line=dict(color=TEXTO, width=1)),
                                  hovertemplate="%{y:,.1f} M<extra>" + nombre + "</extra>"))
    fig5.add_hline(y=monto * presupuesto / 1e6, line=dict(color=TURQUESA, width=1.5, dash="dash"),
                   annotation_text="Presupuesto", annotation_font_color=TURQUESA, annotation_position="top left")
    fig5.update_yaxes(title="Millones de COP", tickformat=",.0f")
    fig5.update_layout(hovermode="x unified")
    collar = instrumentos.set_index("instrumento").loc["Collar"]
    texto_instrumentos = (
        f"Cubriendo el {cobertura} % a {plazo} meses con volatilidad simulada de {pct(e['vol'])}: la call fija un techo de "
        f"{cop(call['strike_call'])} por {cop(call['prima'], 1)} de prima por dólar, y el collar deja la tasa entre "
        f"{cop(collar['strike_put'])} y {cop(collar['strike_call'])} sin prima. Primas teóricas sin el diferencial del banco."
    )
    return (kpis, texto, estilo(fig, 330, tema=tema), estilo(fig2, 330, tema=tema), estilo(fig3, 330, leyenda=False, tema=tema),
            estilo(fig4, 330, leyenda=False, tema=tema), estilo(fig5, 340, tema=tema), texto_instrumentos)


@app.callback(Output("descarga", "data"), Input("boton-descarga", "n_clicks"),
              *[State(nombre, "value") for nombre in ENTRADAS_DECISION], prevent_initial_call=True)
def descargar_escenario(clics, *valores):
    """CSV con los supuestos elegidos, la curva por nivel de cobertura y la comparación de instrumentos."""
    e = escenario(*valores)
    supuestos = pd.DataFrame([
        ("Fecha de los datos", M["fecha_decision"]), ("TRM vigente", SPOT), ("Plazo (meses)", e["plazo"]),
        ("Porcentaje cubierto", e["h"]), ("Monto (USD)", e["monto"]), ("TRM presupuestada", e["presupuesto"]),
        ("Tasa COP", e["tasa_cop"]), ("Tasa USD", e["tasa_usd"]), ("Recargo del banco", e["recargo"]),
        ("Margen objetivo", e["margen_obj"]), ("Margen mínimo", e["margen_min"]), ("Tolerancia al riesgo", e["tolerancia"]),
        ("Deriva supuesta", e["deriva"]), ("Strike de la call sobre el forward", e["holgura"]), ("Forward", e["forward"]),
        ("Cobertura sugerida", e["optima"]["cobertura"]),
    ], columns=["supuesto", "valor"])
    supuestos["valor"] = [str(v).replace(".", ",") if isinstance(v, (int, float)) else v for v in supuestos["valor"]]
    bloques = [("Supuestos", supuestos), ("Métricas por nivel de cobertura con forward", e["curva"]),
               ("Comparación de instrumentos", e["instrumentos"])]
    texto = "\n".join(f"{titulo}\n{tabla.to_csv(sep=';', decimal=',', index=False)}" for titulo, tabla in bloques)
    nombre = f"escenario_cobertura_{e['plazo']}m_{round(e['h'] * 100)}pct.csv"
    return dcc.send_string("﻿" + texto, nombre)


@app.callback(Output("kpis-backtest", "children"), Output("g-backtest", "figure"), Input("plazo-bt", "value"),
              Input("estrategia-bt", "value"), Input("tema", "data"))
def actualizar_backtest(plazo, estrategia="todo", tema="oscuro"):
    """Resultado histórico de cubrir cada mes desde 2015, todo al inicio o en tres tramos."""
    AZUL, TURQUESA, AMBAR, ROJO, LILA, TEXTO, TENUE, REJILLA, LINEA = colores(tema)
    d = historico[historico["plazo"] == plazo]
    escalonada = estrategia == "escalonada"
    precio = d["forward_escalonado"] if escalonada else d["forward"]
    a = d["ahorro_escalonado_pct"] if escalonada else d["ahorro_cubrir_pct"]
    var_sin = (d["trm_final"] / d["trm_inicial"] - 1).std()
    var_con = (precio / d["trm_inicial"] - 1).std()
    kpis = [
        tarjeta("Meses evaluados", f"{len(d)}", f"{d['fecha'].min():%m/%Y} a {d['fecha'].max():%m/%Y}"),
        tarjeta("Cubrir salió más barato", pct((a > 0).mean()), "Proporción de meses", "baja"),
        tarjeta("Costo medio de cubrir", pct(-a.mean(), 2), "Por dólar, sobre la TRM inicial"),
        tarjeta("Peor caso sin cubrir", pct(a.max()), "Sobrecosto frente a la cobertura", "alerta"),
        tarjeta("Dispersión del costo", f"{pct(var_sin)} → {pct(var_con)}", "Sin cubrir → cubriendo"),
    ]
    if escalonada:
        mejor = (d["ahorro_escalonado_pct"] > d["ahorro_cubrir_pct"]).mean()
        kpis.append(tarjeta("Escalonada vs todo al inicio", pct(mejor), "Meses en que los tramos salieron más baratos", "neutro"))
    fig = go.Figure(go.Bar(x=d["fecha"], y=a, marker=dict(color=np.where(a > 0, TURQUESA, ROJO)),
                           customdata=np.stack([d["trm_inicial"], precio, d["trm_final"]], axis=1),
                           hovertemplate="%{x|%m/%Y}<br>Ahorro %{y:.1%}<br>TRM inicial %{customdata[0]:,.0f}"
                                         "<br>Precio cubierto %{customdata[1]:,.0f}<br>TRM final %{customdata[2]:,.0f}<extra></extra>"))
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(hovermode="closest")
    return kpis, estilo(fig, 380, leyenda=False, tema=tema)


if __name__ == "__main__":
    app.run(debug=False, port=8050)
