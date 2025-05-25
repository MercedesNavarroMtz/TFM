import psycopg2
from django.db import connection
from psycopg2 import sql
import logging


logger = logging.getLogger(__name__)

class CitusManager:
    def __init__(self, db_name, user, password, host="localhost", port=5432):
        """
        Constructor para inicializar la conexión a la base de datos.

        :param db_name: Nombre de la base de datos.
        :param user: Usuario de la base de datos.
        :param password: Contraseña del usuario.
        :param host: Dirección del host (por defecto: localhost).
        :param port: Puerto de la base de datos (por defecto: 5432).
        """
    
        self.db_name = db_name
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.connection = None

    def connect(self):
        """
        Conecta a la base de datos PostgreSQL.
        """
        try:
            self.connection = psycopg2.connect(
                dbname=self.db_name,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            print("Conexión a la base de datos exitosa.")
        except Exception as e:
            print(f"Error al conectar a la base de datos: {e}")

    def close_connection(self):
        """
        Cierra la conexión a la base de datos.
        """
        if self.connection:
            self.connection.close()
            print("Conexión a la base de datos cerrada.")

    def execute_query(self, query, params=None):
        """
        Ejecuta una consulta SQL.

        :param query: Consulta SQL a ejecutar.
        :param params: Parámetros para la consulta (por defecto: None).
        :return: Resultado de la consulta si aplica.
        """
        if not self.connection:
            print("No hay conexión activa a la base de datos.")
            return

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                if str(query).strip().upper().startswith("SQL('SELECT"):
                    result = cursor.fetchall()
                    return result
                self.connection.commit()
                print("Consulta ejecutada con éxito.")
        except Exception as e:
            print(f"Error al ejecutar la consulta: {e}")

    def check_if_exist(self, table_name):
        """
        Comprueba si una tabla existe.

        :param table_name: Nombre de la tabla.
        """
        query = sql.SQL("SELECT EXISTS (SELECT 1 FROM pg_tables WHERE tablename = %s)")
        return self.execute_query(query, [table_name])[0][0]
    
    def check_if_distributed(self, table_name):
        """
        Comprueba si una tabla esta distribuida.

        :param table_name: Nombre de la tabla distribuida.
        """
        query = sql.SQL("SELECT EXISTS (SELECT 1 FROM pg_dist_partition WHERE logicalrelid = %s::regclass);")
        return self.execute_query(query, [table_name])[0][0]
    
    def check_if_reference(self, table_name):
        """
        Comprueba si una tabla esta replicada.

        :param table_name: Nombre de la tabla replicada.
        """
        query = sql.SQL("SELECT EXISTS (SELECT 1 FROM pg_dist_partition WHERE logicalrelid = %s::regclass AND partmethod = 'r');")
        return self.execute_query(query, [table_name])[0][0]

    
    def create_distributed_table(self, table_name,table_type, partition_column):
        """
        Crea una tabla distribuida o replicada en Citus.

        :param table_name: Nombre de la tabla a distribuir.
        :param table_type: Columna que indica si la tabla será distribuida o replicada
        :param partition_column: Columna de partición para distribuir la tabla.
        """
        try:
            query = ''
            if  not self.check_if_exist(table_name):
                print(f"La tabla '{table_name}' no existe. No se puede distribuir.")
                return
            
            if table_type == 'distributed':
                if self.check_if_distributed(table_name):
                    print(f"La tabla '{table_name}' ya está distribuida. Saltando.")
                    return
                query = sql.SQL("SELECT create_distributed_table(%s, %s);")
                self.execute_query(query,[table_name,partition_column])

            elif table_type == 'reference':
                if self.check_if_distributed(table_name):
                    print(f"La tabla '{table_name}' ya está distribuida. Saltando.")
                    return
                query = sql.SQL("SELECT create_reference_table(%s);")
                self.execute_query(query,[table_name])
                
            else:
                print("Tipo de tabla inválido")
                return

            print(f"Tabla {table_name} distribuida o replicada exitosamente.")
        except Exception as e:
            print(f"Error al distribuir o replicar la tabla {table_name}: {e}")

