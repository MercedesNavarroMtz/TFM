# TFM
Repositorio del Trabajo Fin de Master del Máster de Ciencia de Datos de la UOC de Mercedes Navarro Martínez donde se guarda el código de análisis de datos y de la base de datos.


<hr><h3 align="center">Plataforma de datos marinos para el análisis y predicción del estado de infraestructuras acuícolas. </h3><hr>
Este repositorio contiene el sistema de modelado predictivo desarrollado para el análisis de datos procedentes de sensores en instalaciones acuícolas. Incluye herramientas para el procesamiento, modelado y análisis de datos, así como una base de datos estructurada con Django y soporte completo para contenedores Docker.


## Estructura general del proyecto

├── models.py # Script principal de modelado predictivo

├── utils.py # Funciones auxiliares utilizadas en models.py

├── requirements.txt # Dependencias necesarias scripts de preprocesado y modelado

├── data-hub/ # Proyecto Django con base de datos y API

│ ├── cages/ # App proporcionada por CTN

│ ├── aquamore/ # App desarrollada por el autor para datos del proyecto AQUAMOORE

│ ├── station/ # App desarrollada por el autor para el proyecto SIOM

│ ├── Dockerfile # Dockerfile para contenerización del entorno

│ ├── docker-compose.yml

| ├── requeriments.txt # Dependencias necesarias para la base de datos

| |── ....

│ └── README.md # Instrucciones para desplegar el contenedor


---

## Instalación y uso

### Requisitos previos
- Python 3.8+
- Git
- Docker

### Clonar el repositorio

git clone https://github.com/MercedesNavarroMtz/TFM.git

cd TFM

pip install -r requirements.txt

python models.py

## Para la base de datos y el uso de Docker 
El directorio data-hub/ contiene la base de datos desarrollada en Django, organizada en tres aplicaciones: cages (proporcionada por el CTN), y aquamore y station, desarrolladas por el autor. Todo el sistema está preparado para ejecutarse en un entorno Docker.

### Instrucciones para el despliegue
Dentro del directorio data-hub, encontrarás un archivo README.md con las instrucciones necesarias para levantar los servicios con Docker y acceder a la base de datos.

cd data-hub

docker-compose up --build

## Notas
Las aplicaciones aquamore y station fueron desarrolladas a partir de la estructura (cages) proporcionada por el CTN.


