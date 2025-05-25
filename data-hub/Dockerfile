# Usa una imagen base de Ubuntu 24.10
FROM ubuntu:24.10

# Configuración de variables de entorno
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV DEBIAN_FRONTEND=noninteractive

# Crear directorio de trabajo
RUN mkdir /code
WORKDIR /code

# Instalar Python 3.13 y paquetes básicos
RUN apt-get update \
    && apt-get install -y software-properties-common wget build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get install -y python3.13 python3.13-venv python3.13-dev


# Crear un entorno virtual
RUN python3.13 -m venv /opt/venv

# Activar el entorno virtual y actualizar pip
RUN . /opt/venv/bin/activate \
    && /opt/venv/bin/python -m pip install --upgrade pip

# Configurar PATH para usar el entorno virtual por defecto
ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get install -y gdal-bin
RUN apt-get install -y libgdal-dev
RUN apt-get install -y python3-gdal
RUN apt-get install -y libpq-dev
RUN apt-get install -y ffmpeg
RUN apt-get clean

# Agregar el repositorio oficial de PostgreSQL
RUN apt install -y postgresql-common
RUN apt-get install -y postgresql-client 


# Configurar alias para Python3
RUN ln -s /usr/bin/python3.13 /usr/local/bin/python3

# Configurar variables de entorno para GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Crear usuario Celery
RUN groupadd -g 1999 celery \
    && useradd -r -u 1999 -g celery celery

# Copiar el archivo de requisitos y realizar la instalación
ADD requirements.txt /code/
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
ADD . /code/

# Configurar el punto de entrada y el comando predeterminado
ENTRYPOINT ["/bin/bash", "/code/entrypoint_generated.sh"]
CMD ""
