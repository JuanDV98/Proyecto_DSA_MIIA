#!/usr/bin/env bash
# Arranca el tablero Dash con Gunicorn en el puerto definido
gunicorn app:app --bind 0.0.0.0:$PORT
