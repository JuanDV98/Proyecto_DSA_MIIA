# 0. Imagen base
FROM python:3.10-slim

# 1. Crear un usuario sin root para correr la app
RUN adduser --disabled-password --gecos '' api-user

# 2. Definir el directorio de trabajo
WORKDIR /opt/analytics-api

# 3. Copiar todo el código y artefactos
COPY . /opt/analytics-api/

# 4. Actualizar pip e instalar dependencias
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Dar permiso de ejecución al script de arranque
RUN chmod +x ./run.sh

# 6. Ceder propiedad de los ficheros al usuario no-root
RUN chown -R api-user:api-user /opt/analytics-api

# 7. Cambiar a usuario no-root para mayor seguridad
USER api-user

# 8. Exponer el puerto que usará la API
EXPOSE 8001

# 9. Comando que se ejecuta al iniciar el contenedor
CMD ["bash", "./run.sh"]
