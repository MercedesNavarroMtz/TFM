import os
from django.db.models.signals import post_save
from django.dispatch import receiver
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.db.models.signals import post_migrate
from django.apps import apps

from data_hub.citus_manager import CitusManager  # Importa tu clase CitusManager

# Configuración para conectarse a la base de datos
citus_manager = CitusManager(
    db_name=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host="db",
    port=5432
)

citus_manager.connect()


#@receiver(post_save)
@receiver(post_migrate)
def distribute_table(sender, **kwargs):
    """
    Señal que se ejecuta después de guardar un modelo.
    Distribuye automáticamente las tablas asociadas a modelos.
    """

    # Evitar que Django llame a esta señal para modelos internos
    # if not apps.is_installed(sender._meta.app_label):
    #     return

    # Iterar por todos los modelos registrados en la aplicación
    for model in apps.get_models():
        # Verificar si el modelo tiene la etiqueta `table_type='distributed'`
        table_type = getattr(model, 'table_type', None)

        if table_type == 'distributed' or table_type == 'reference':
            table_name = model._meta.db_table  # Nombre de la tabla en la base de datos
            partition_column = getattr(model, 'partition_column', None) # Columna de partición por defecto (puedes personalizar esto)
            print(table_name)

            try:
                citus_manager.create_distributed_table(table_name, table_type, partition_column)
            except Exception as e:
                print(f"Error al distribuir la tabla {table_name}: {e}")


# @receiver(post_save, sender=Node)
# def send_node_update_notification(sender, instance, created, **kwargs):
#     """
#     Signal que se dispara cuando se guarda un objeto Node.
#     """
#     # Determina si fue una creación o una actualización
#     event_type = "created" if created else "updated"

#     if event_type != 'updated':
#         return

#     current_time = datetime.datetime.now()


#     # Crear el mensaje
#     message = {
#         "type": "send.message",
#         "date_time": str(current_time),  
#         "node_name": instance.name,  # Asegúrate de que `Node` tenga un campo `name`
#         "msg_type": event_type,
#         "msg": f"{instance.get_status_display()}",
#     }

#     print(message)

#     # Envía el mensaje a través del WebSocket
#     channel_layer = get_channel_layer()
#     async_to_sync(channel_layer.group_send)(
#         "nodes_control",  # Nombre del grupo
#         message,
#     )
