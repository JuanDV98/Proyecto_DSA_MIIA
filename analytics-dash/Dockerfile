# Imagen base ligera de Python
FROM python:3.10-slim

# 1. Crear un usuario sin privilegios root
RUN adduser --disabled-password --gecos '' dash-user

# 2. Definir directorio de trabajo
WORKDIR /opt/analytics-dash

# 3. Copiar dependencias e instalar
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 4. Copiar el resto del código (app.py, modelos, run.sh)
COPY . .

# 5. Dar permiso de ejecución al script de arranque
RUN chmod +x ./run.sh

# 6. Ajustar permisos al usuario dash-user
RUN chown -R dash-user:dash-user /opt/analytics-dash

# 7. Cambiar a usuario no-root
USER dash-user

# 8. Puerto por defecto de Dash
EXPOSE 8050

# 9. Arrancar el tablero con el script
CMD ["bash", "./run.sh"]