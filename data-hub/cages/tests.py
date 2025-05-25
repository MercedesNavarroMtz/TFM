import os
from django.test import TestCase
from django.urls import reverse
from django.conf import settings
from rest_framework import status
from rest_framework.test import APITestCase
from django.utils import timezone
from django.db.models import Q
from .serializers import *
from .helper import *
from .models import *
# from data_hub.helper import getKeycloakToken
from rest_framework.test import APIClient
from django.core.files.uploadedfile import SimpleUploadedFile
### python3 manage.py test


# class CagesTests(APITestCase):
#     def setUp(self):
#         self.client = APIClient()
#         # Autenticar al usuario
#         self.access_token = getKeycloakToken()
#         self.username = os.getenv("KEYCLOAK_USER_ADMIN")

#         # Crear un nodo para las pruebas
#         self.node_1 = Node.objects.create(
#              type="A", name='Node_test_1', status= 0, location= "aa", min_accuracy= 0.7, images_n= 2000, images_interval= 1
#         )

#         # Crear runs para las pruebas
#         run_test_data = {
#             "name": "run_test_1",
#             "type": "Ejemplo",
#             #"initial_accuracy": 0.85
#         }

#         file_path = "management/resources/model_test.pkl"
#         with open(file_path, 'rb') as file:
#             file_data = file.read()
#             file_obj = SimpleUploadedFile("model_test_1.pkl", file_data, content_type="application/octet-stream")

#         run_test_data['model_path'] = file_obj

#         # Realiza la petición POST para crear el nuevo Store_run
#         self.client.post('/management/store-run/', run_test_data, HTTP_AUTHORIZATION=f'Bearer {self.access_token}',format='multipart')

#         self.status_run_1 = Status_run.objects.create(
#              node=self.node_1, store_run=Store_run.objects.last(), status= 0
#         )

#     def tearDown(self):
#         # Borra solo el archivo llamado model_test.pkl después de que se complete la prueba
#         media_path_model = os.path.join(settings.MEDIA_ROOT, 'model')
#         file_path_1 = os.path.join(media_path_model, "model_test_1.pkl")
#         file_path_2 = os.path.join(media_path_model, "model_test_2.pkl")
#         if os.path.exists(file_path_1):
#             os.remove(file_path_1)
#         if os.path.exists(file_path_2):
#             os.remove(file_path_2)

#         media_path_images = os.path.join(settings.MEDIA_ROOT,'images')
#         for folder_name in os.listdir(media_path_images):
#             folder_path = os.path.join(media_path_images, folder_name)
#             if os.path.isdir(folder_path) and folder_name.startswith(self.node_1.name):
#                 for file_name in os.listdir(folder_path):
#                     file_path = os.path.join(folder_path, file_name)
#                     os.remove(file_path)
#                 os.rmdir(folder_path)
        

#     def test_create_store_run(self):
#         # Datos de ejemplo para crear un nuevo Store_run
#         data = {
#             "name": "run test 2",
#             "type": "Ejemplo",
#             #"initial_accuracy": 0.85  # Ajusta según sea necesario
#         }

#         # Simula la carga del archivo model_test.pkl
#         file_path = "management/resources/model_test.pkl"
#         with open(file_path, 'rb') as file:
#             file_data = file.read()
#             file_obj = SimpleUploadedFile("model_test_2.pkl", file_data, content_type="application/octet-stream")

#         data['model_path'] = file_obj

#         # Realiza la petición POST para crear el nuevo Store_run
#         response = self.client.post('/management/store-run/', data, HTTP_AUTHORIZATION=f'Bearer {self.access_token}',format='multipart')
#         # Verifica que la petición haya sido exitosa (código de estado HTTP 201)
#         self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        
#         # Obtén el objeto recién creado desde la base de datos
#         store_run = Store_run.objects.last()
#         # Verifica que los datos del objeto coincidan con los datos que enviaste en la petición POST
#         self.assertEqual(store_run.name, data['name'])
#         self.assertEqual(store_run.type, data['type'])
#         #self.assertEqual(store_run.initial_accuracy, data['initial_accuracy'])

#         # Verifica que el campo model_path no esté vacío
#         self.assertIsNotNone(store_run.model_path)


#     def test_get_status_run(self):

#         response = self.client.get('/management/status-run/?status=0&node__name='+self.node_1.name , HTTP_AUTHORIZATION=f'Bearer {self.access_token}')
#         # Verifica que la petición haya sido exitosa 
#         self.assertEqual(response.status_code, status.HTTP_200_OK)

#         status_run = Status_run.objects.last()
#         # Verifica que los datos del objeto coincidan con los datos que enviaste en la petición POST
#         self.assertEqual(status_run.node, self.status_run_1.node)
#         self.assertEqual(status_run.store_run, self.status_run_1.store_run)
#         self.assertEqual(status_run.status, self.status_run_1.status)


#     def test_image_upload(self):
#         data = {
#             "node_name": self.node_1.name,
#         }

#         # Simula la carga del archivo model_test.pkl
#         file_path = "management/resources/images_raw.zip"
#         with open(file_path, 'rb') as file:
#             file_data = file.read()
#             file_obj = SimpleUploadedFile("images_raw.zip", file_data, content_type="application/octet-stream")

#         data['file'] = file_obj

#         # Realiza la petición POST para crear el nuevo Store_run
#         response = self.client.post('/management/image-upload/', data, HTTP_AUTHORIZATION=f'Bearer {self.access_token}',format='multipart')
#         # Verifica que la petición haya sido exitosa (código de estado HTTP 201)
#         self.assertEqual(response.status_code, status.HTTP_201_CREATED)

