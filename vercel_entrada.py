"""Punto de entrada para Vercel: expone como `app` la aplicación Flask que sirve el tablero.

Posit Connect Cloud sigue usando app.py, donde `app` es el objeto Dash; Vercel necesita la
aplicación WSGI de Flask, que es `server`. pyproject.toml apunta aquí con tool.vercel.entrypoint.
"""

from app import server as app  # noqa: F401
