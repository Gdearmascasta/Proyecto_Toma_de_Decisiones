"""Regenera todo el proyecto con un solo comando: datos, notebook, pruebas y documento técnico.

Uso:
    python reproducir.py               notebook, pruebas y PDF con los CSV ya guardados
    python reproducir.py --descargar   además descarga datos nuevos antes del notebook
    python reproducir.py --sin-pdf     omite el documento (por ejemplo, si Quarto no está instalado)

Se ejecuta con el Python del entorno virtual del proyecto (ver LEEME.md).
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent


def paso(titulo, comando, **opciones):
    """Ejecuta un paso y detiene todo si falla."""
    print(f"\n==> {titulo}\n    {' '.join(str(c) for c in comando)}", flush=True)
    resultado = subprocess.run(comando, cwd=RAIZ, **opciones)
    if resultado.returncode != 0:
        sys.exit(f"Falló el paso «{titulo}» (código {resultado.returncode}).")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--descargar", action="store_true", help="descargar datos nuevos antes de ejecutar el notebook")
    parser.add_argument("--sin-pdf", action="store_true", help="no compilar el documento técnico")
    args = parser.parse_args()
    python = sys.executable
    # El kernel de Jupyter se lanza como «python»: con el entorno primero en el PATH usa sus librerías
    os.environ["PATH"] = str(Path(python).parent) + os.pathsep + os.environ.get("PATH", "")

    if args.descargar:
        paso("Descarga de datos", [python, "datos/descargar_datos.py"])
    paso("Notebook de principio a fin", [python, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                                         "--ExecutePreprocessor.kernel_name=python3", "--ExecutePreprocessor.timeout=1800",
                                         "notebook/analisis_cobertura.ipynb"])
    paso("Pruebas de las funciones de cálculo", [python, "-m", "pytest", "-q", "tests"])
    if args.sin_pdf:
        print("\nDocumento técnico omitido (--sin-pdf).")
    elif shutil.which("quarto") is None:
        sys.exit("No se encontró Quarto. Instálalo desde quarto.org o usa --sin-pdf.")
    else:
        paso("Documento técnico", ["quarto", "render", "documento/documento_tecnico.qmd"],
             env={**os.environ, "QUARTO_PYTHON": python})
    print("\nListo. Para abrir el tablero: python app.py y entrar a http://localhost:8050")


if __name__ == "__main__":
    main()
