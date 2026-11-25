# Cobertura del riesgo cambiario en importaciones

Evaluación de contratos forward a 3 y 6 meses para una importadora de Cartagena.

Proyecto final de Toma de decisiones basadas en datos (UTB).
Integrantes: Ángel Luis Vega Rodríguez (T00068186), Germán Eduardo De Armas Castaño (T00068765), María Valentina Serna (T00067756), Luis Mario Díaz (T00069220) y Daniel Castillo (T00091828).

La pregunta es si una importadora que paga en dólares a 3 o 6 meses debe fijar hoy la tasa con un forward y qué porcentaje del pago conviene cubrir.

Tablero publicado: https://01a2057d-f00e-0325-72e3-2329520e3b87.share.connect.posit.cloud/

## Estructura

```
reproducir.py             regenera todo con un solo comando (ver abajo)
datos/
  descargar_datos.py      descarga TRM, Yahoo Finance, BIS y FRED a CSV
  crudos/                 copias de las descargas
  limpios/                paneles diario y mensual después del ETL
notebook/
  analisis_cobertura.ipynb   ETL, EDA, regresión, GJR-GARCH, Monte Carlo, opciones (call y collar),
                             tornado, backtest (todo al inicio y escalonado) y recomendación
resultados/
  resultados.json         todas las cifras que usan el documento y el tablero
documento/
  documento_tecnico.qmd   documento técnico (portada, índice y 4 páginas de contenido)
  estilo.typ, logos/      diseño del PDF y logos de la UTB y la Escuela
  documento_tecnico.pdf   versión compilada
  figuras/                gráficas exportadas por el notebook
dashboard/
  app.py, calculos.py     tablero en Dash (modo claro y oscuro, descarga del escenario en CSV)
  assets/estilos.css      estilo visual
  datos/                  archivos que exporta el notebook para el tablero
  requirements.txt        dependencias para Posit Connect Cloud
presentacion/             diapositivas de la sustentación (18, con notas del orador)
tests/                    pruebas de las fórmulas de calculos.py, que usan el notebook y el tablero
.github/workflows/        pruebas en cada push y actualización semanal de datos por pull request
```

## Cómo reproducir todo

Crear el entorno virtual dentro del proyecto (la carpeta `venv/` está en `.gitignore`):

```bash
python -m venv venv
venv/bin/python -m pip install -r requirements-notebook.txt      # Linux y macOS
venv\Scripts\python.exe -m pip install -r requirements-notebook.txt   # Windows
```

En Windows, si el proyecto está en OneDrive y la ruta es demasiado larga para instalar Jupyter, crear el entorno en una carpeta corta (por ejemplo `C:\venvs\cobertura`) y usar ese Python en los comandos.

Luego, con el Python del entorno:

```bash
python reproducir.py               # notebook, pruebas y PDF con los CSV ya guardados
python reproducir.py --descargar   # además descarga datos nuevos primero
python reproducir.py --sin-pdf     # sin compilar el documento (si no está Quarto)
python app.py                      # tablero en http://localhost:8050
python -m pytest tests             # solo las pruebas
```

`reproducir.py` ejecuta en orden `datos/descargar_datos.py` (opcional), el notebook completo, las pruebas y `quarto render documento/documento_tecnico.qmd`. Se detiene en el primer paso que falle.

La simulación usa 10.000 escenarios y semilla 42, así que los resultados son idénticos en cada ejecución en un mismo equipo. Entre equipos con otro sistema operativo o versión de Python, el ajuste del GJR-GARCH puede variar en la quinta cifra significativa, sin cambiar ninguna cifra redondeada del documento.

El tablero muestra junto a la fecha de los datos cuántos días tienen, y la marca en ámbar cuando pasan de dos semanas. La Action `Actualizar datos` corre cada lunes (o a mano desde la pestaña Actions de GitHub), descarga datos nuevos, ejecuta el notebook y abre un pull request. No publica sola, porque el documento y las diapositivas citan cifras fijas.

## Despliegue en Posit Connect Cloud

1. Subir el proyecto a un repositorio de GitHub (la carpeta `dashboard/` debe incluir `datos/` y `assets/`).
2. En connect.posit.cloud elegir **Publish → Dash**, seleccionar el repositorio y la rama.
3. Archivo principal `app.py` (en la raíz, carga el tablero de `dashboard/`). Posit instala las librerías de `requirements.txt`.
4. Publicar y compartir el enlace.

## Despliegue en Vercel (alternativa)

`vercel_entrada.py` expone la aplicación Flask del tablero como `app`, y `pyproject.toml` le indica a Vercel que use ese archivo (`tool.vercel.entrypoint`). `.vercelignore` deja fuera el notebook, el documento, la presentación y los datos crudos, que el tablero no necesita.

1. En vercel.com elegir **Add New → Project** e importar el repositorio de GitHub.
2. Dejar *Framework Preset* en **Other** o como lo detecte Vercel (Python), sin comando de build.
3. Publicar. Vercel instala las dependencias de `pyproject.toml`, que son las mismas de `requirements.txt`.

Si se cambia una dependencia, hay que actualizarla en los dos archivos. La primera visita después de un rato sin uso tarda unos segundos mientras la función arranca.
