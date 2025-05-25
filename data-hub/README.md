<hr><h3 align="center">INSTALACIÓN</h3><hr>

#### Primera ejecución
1.	Instalar docker.
2.	Si se está ejecutando Docker en un equipo Windows, los archivos Entrypoint.sh, Docker-manager y .env se deben editar. Sus finales de línea deben estar según el estándar de Linux (LF). 
3.	Desde línea de comandos ir al directorio donde está el respositorio.
4.	Crear archivo .env, con las variables de entrorno necesarias para este proyecto.
5.	Ejecutar “**sh docker-manager build**” (Construye el contenedor).
6.	Ejecutar “**sh docker-manager up --initial**" 


> **Nota**
> <br> Cuando se añade el modificador “--initial”, el script entrypoint.sh ejecuta el bloque de código que se encarga de hacer migraciones y fixtures, es decir: de crear la BBDD. Normalmente sólo la primera vez que se levanta el contenedor, o después de modificar el modelo de datos.


<hr><h3 align="center">USO</h3><hr>

#### Ejecución
1. **sh docker-manager up**

#### Detener contenedores
- CNTRL+C si se tiene una terminal con el proyecto corriendo.
- **sh docker-manager stop**. Detiene los contenedores.
- **sh docker-manager down**. Detiene los contenedores y los limpia, o borra la información generada en ellos.


#### Otros comandos
- **docker ps** -> Lista contenedores, sus ID's y los puertos que están a la escucha.
- **docker exec -it {id_contenedor} bash** -> Entra al contenedor (Se sale con “exit”).

<br>
