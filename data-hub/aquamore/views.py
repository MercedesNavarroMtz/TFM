from django.shortcuts import render, redirect
import pandas as pd
import io
import os
from rest_framework import viewsets,status, permissions
from rest_framework.parsers import MultiPartParser, FormParser
from .models import *
from .serializers import *
from rest_framework.response import Response
from datetime import datetime
from django.db import transaction
from rest_framework.decorators import action
from django_filters.rest_framework import DjangoFilterBackend, FilterSet, DateTimeFilter, NumberFilter, CharFilter
from django.views import View
from django.http import HttpResponse
from django.template.loader import render_to_string
from django.http import StreamingHttpResponse
import csv




# Create your views here.

class InfrastructureViewSet(viewsets.ModelViewSet):
    queryset = Infrastructure.objects.all()        # Para obtener todas las infraestructuras de la base de datos
    # permission_classes = (permissions.IsAuthenticated,)
    serializer_class = InfrastructureSerializer    # Para transformar los datos con el serializer
    # filter_backends = (filters.DjangoFilterBackend,)


class SensorViewSet(viewsets.ModelViewSet):
    queryset = Sensor.objects.all() # Para obtener todos los sensores de la base de datos
    permission_classes = (permissions.AllowAny,)
    serializer_class = SensorSerializer # Transformaciones con el serializador


class SensorLoadFilter(FilterSet):
    sensor_id = NumberFilter(field_name="sensor__id", lookup_expr='exact')
    sensor_name = CharFilter(field_name="sensor__name", lookup_expr='icontains')
    date_from = DateTimeFilter(field_name="date_time", lookup_expr='gte')
    date_to = DateTimeFilter(field_name="date_time", lookup_expr='lte')
    
    class Meta:
        model = SensorLoad
        fields = ['sensor_id', 'sensor_name', 'date_from', 'date_to']



class SensorLoadViewSet(viewsets.ModelViewSet):
    queryset = SensorLoad.objects.all()
    permission_classes = (permissions.AllowAny,)
    serializer_class = SensorLoadSerializer
    filter_backends = [DjangoFilterBackend]
    filterset_class = SensorLoadFilter
    # parser_classes = [MultiPartParser, FormParser]  # Permite procesar archivos


    @action(detail=False, methods=['post', 'get'], url_path='update-csv', permission_classes=[permissions.AllowAny])
    def update_csv(self, request):

        """ Función que va a permitir subir los archivos CSV y procesarlos"""

        if request.method == 'GET':
            return Response(template_name='upload_form.html') 
        
        print(" Recibida solicitud POST en /update-csv/")
        print(" request.FILES:", request.FILES)

        serializer = FileUploadSerializer(data=request.data) 
        
        if not serializer.is_valid():
            print(" Errores en el serializer:", serializer.errors)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST) 
            
        file = request.FILES.get('file')
        
        if not file:
            print(" No se recibió un archivo")
            return Response({"error": "No se proporcionó ningún archivo"}, status=status.HTTP_400_BAD_REQUEST)

        filename = os.path.splitext(file.name)[0]

        try:
            # Extraer el nombre del sensor del formato "sensor_nu2_3_4_24"
            parts = filename.split('_')
            if len(parts) < 2:
                raise ValueError("Formato de nombre de archivo incorrecto")
                
            sensor_name = parts[1]
            
        except ValueError as e:
            print(f" Error: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        sensor = Sensor.objects.filter(name=sensor_name).first()

        if not sensor:
            print(f" No se encontró un sensor con el nombre '{sensor_name}'")
            return Response({"error": f"No se encontró un sensor con el nombre '{sensor_name}'"}, status=status.HTTP_400_BAD_REQUEST)

        decoded_file = file.read().decode('latin-1')
        contenido = decoded_file.strip().splitlines()
        cleaned_lines = [line for line in contenido if len(line.split(',')) == 2]
        try:
            df = pd.read_csv(io.StringIO("\n".join(cleaned_lines)), delimiter=",", header=None, names=['date_time', 'weight'])
        except Exception as e:
            print(f"Error al leer el CSV con pandas: {e}")
            return Response({"error": "No se pudo leer el archivo CSV. Verifique su formato."}, status=status.HTTP_400_BAD_REQUEST)
    
        meses = {
                'ene.': '01', 'feb.': '02', 'mar.': '03', 'abr.': '04',
                'may.': '05', 'jun.': '06', 'jul.': '07', 'ago.': '08',
                'sep.': '09', 'oct.': '10', 'nov.': '11', 'dic.': '12'
            }

        def convertir_fecha(fecha_str):
            """Transforma la fecha al formato correcto"""
            for mes_es, mes_num in meses.items():
                if mes_es in fecha_str:
                    fecha_str = fecha_str.replace(mes_es, mes_num)
                    break
            try:
                return datetime.strptime(fecha_str, "%d-%m-%Y %H:%M:%S.%f")
            except ValueError:
                try:
                    return datetime.strptime(fecha_str, "%d-%m-%Y %H:%M:%S")
                except ValueError:
                    return None

        df['date_time'] = df['date_time'].astype(str).apply(convertir_fecha)

        df.dropna(subset=['date_time'], inplace=True)
        df['date_time'] = df['date_time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))

        print(f"Primeras filas después de corregir fechas:\n{df.head()}")

        df['sensor_id'] = sensor.id  

        data_list = []

        for index, row in df.iterrows():

            data = {
                'sensor': sensor,
                'date_time': row['date_time'],  
                'weight': row['weight'],
            }
            data_list.append(data)

            if index < 5:
                print(f" Fila {index + 1} procesada: {data}")

        with transaction.atomic():
            objects = [SensorLoad(**data) for data in data_list]
            SensorLoad.objects.bulk_create(objects, batch_size=3000)

        print(f"{len(data_list)} registros guardados en la base de datos para el sensor {sensor.id}")

        return Response({"message": f"Datos actualizados correctamente para sensor {sensor.id}"}, status=status.HTTP_200_OK)

class SensorSclFilter(FilterSet): # Filtros 
    sensor_id = NumberFilter(field_name="sensor__id", lookup_expr='exact')
    sensor_name = CharFilter(field_name="sensor__name", lookup_expr='icontains')
    date_from = DateTimeFilter(field_name="date_time", lookup_expr='gte')
    date_to = DateTimeFilter(field_name="date_time", lookup_expr='lte')
    
    class Meta:
        model = SensorSCLyFlot
        fields = ['sensor_id', 'sensor_name', 'date_from', 'date_to']



class SensorSCLyFlotViewSet(viewsets.ModelViewSet):
    queryset = SensorSCLyFlot.objects.all()
    permission_classes = (permissions.AllowAny,)
    serializer_class = SensorSCLyFlotSerializer
    parser_classes = (MultiPartParser, FormParser) # Define los tipos de datos que puede recibir, multipartparser peticion multipart/form-data (imagenes,audios,docs)
    filter_backends = [DjangoFilterBackend]
    filterset_class = SensorSclFilter                                              # Recibe datos enviados en formularios

    @action(detail=False, methods=['post'], url_path='update-csv', permission_classes=[permissions.AllowAny])
    def update_csv(self, request):

        """ Función que va a permitir subir los archivos CSV y procesarlos"""

        print(" Recibida solicitud POST en /update-csv/")
        print(" request.FILES:", request.FILES)

        serializer = FileUploadSerializer(data=request.data)  

        if not serializer.is_valid():
            print(" Errores en el serializer:", serializer.errors)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)  

        file = request.FILES.get('file')

        if not file:
            print(" No se recibió un archivo")
            return Response({"error": "No se proporcionó ningún archivo"}, status=status.HTTP_400_BAD_REQUEST)

        filename = os.path.splitext(file.name)[0]

        try:
            sensor_id = int(filename[-2:])
            print(f" ID de sensor extraído del nombre del archivo: {sensor_id}")
        except ValueError:
            print(" Error: El nombre del archivo debe terminar en dos números")
            return Response({"error": "El nombre del archivo debe terminar en dos números"}, status=status.HTTP_400_BAD_REQUEST)

        # Buscar el sensor cuyo nombre termine con esos dos dígitos
        sensor = Sensor.objects.filter(name__endswith=str(sensor_id)).first()

        if not sensor:
            print(f" No se encontró un sensor cuyo nombre termine en '{sensor_id}'")
            return Response({"error": f"No se encontró un sensor cuyo nombre termine en '{sensor_id}'"}, status=status.HTTP_400_BAD_REQUEST)

        print(f" Sensor encontrado: {sensor.name} (ID: {sensor.id})")

        decoded_file = file.read().decode('latin-1')
        df = pd.read_csv(io.StringIO(decoded_file), delimiter=',', quotechar='"')

        df['sensor_id'] = sensor.id  

        data_list = []

        for index, row in df.iterrows(): 
            data = {
                'sensor': sensor,
                'temperature': str(row['Temp(Â°C)']),  
                'depth': str(row['Depth(m)']),
                'pitch': row['Pitch(Â°)'],
                'roll': row['Roll(Â°)'],
                'head': row['Head(Â°)'],
                'four_p': row['4p(Âº)'],
                'date_time': row['Date_Time'],  
            }
            data_list.append(data)

            if index < 5:
                print(f" Fila {index + 1} procesada: {data}")

        with transaction.atomic():
            objects = [SensorSCLyFlot(**data) for data in data_list]
            SensorSCLyFlot.objects.bulk_create(objects, batch_size=3000)

        print(f"{len(data_list)} registros guardados en la base de datos para el sensor {sensor.id}")

        return Response({"message": f"Datos actualizados correctamente para sensor {sensor.id}"}, status=status.HTTP_200_OK)
    
class SensorCorrFilter(FilterSet):
    sensor_id = NumberFilter(field_name="sensor__id", lookup_expr='exact')
    sensor_name = CharFilter(field_name="sensor__name", lookup_expr='icontains')
    date_from = DateTimeFilter(field_name="date_time", lookup_expr='gte')
    date_to = DateTimeFilter(field_name="date_time", lookup_expr='lte')
    
    class Meta:
        model = SensorCorr
        fields = ['sensor_id', 'sensor_name', 'date_from', 'date_to']


class SensorCorrViewSet(viewsets.ModelViewSet):
    queryset = SensorCorr.objects.all()
    permission_classes = (permissions.AllowAny,)
    serializer_class = SensorCorrSerializer
    filter_backends = [DjangoFilterBackend]
    filterset_class = SensorCorrFilter  

    @action(detail=False, methods=['post'], url_path='update-csv', permission_classes=[permissions.AllowAny])
    def update_csv(self, request):

        """ Función que va a permitir subir los archivos CSV y procesarlos"""

        print(" Recibida solicitud POST en /update-csv/")
        print(" request.FILES:", request.FILES)

        serializer = FileUploadSerializer(data=request.data)  

        if not serializer.is_valid():
            print(" Errores en el serializer:", serializer.errors)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)  

        file = request.FILES.get('file')

        if not file:
            print(" No se recibió un archivo")
            return Response({"error": "No se proporcionó ningún archivo"}, status=status.HTTP_400_BAD_REQUEST)

        filename = os.path.splitext(file.name)[0]

        try:
            sensor_id = filename.split('_')[0]  # Esto se queda con 'AQUA101'
            print(f" ID de sensor extraído del nombre del archivo: {sensor_id}")
        except ValueError:
            print(" Error: El nombre del archivo debe terminar en dos números")
            return Response({"error": "El nombre del archivo debe terminar en dos números"}, status=status.HTTP_400_BAD_REQUEST)

        # Buscar el sensor cuyo nombre termine con esos dos dígitos
        sensor = Sensor.objects.filter(name=sensor_id).first()

        if not sensor:
            print(f" No se encontró un sensor cuyo nombre termine en '{sensor_id}'")
            return Response({"error": f"No se encontró un sensor cuyo nombre termine en '{sensor_id}'"}, status=status.HTTP_400_BAD_REQUEST)

        print(f" Sensor encontrado: {sensor.name} (ID: {sensor.id})")

        decoded_file = file.read().decode('latin-1')
        df = pd.read_csv(io.StringIO(decoded_file), delimiter=';')

        print(f"Primeras filas del archivo CSV:\n{df.head()}")

        df['sensor_id'] = sensor.id  

        data_list = []

        for index, row in df.iterrows():
            date_time = pd.to_datetime(row['DateTime'], errors='coerce', dayfirst=True)

            data = {
                'sensor': sensor,
                'date_time': date_time,  
                'heading': str(row['Heading']),
                'pitch': row['Pitch'],
                'roll': row['Roll'],
                'pressure': row['Pressure'],
                'speed1': row['Speed#1(2.50m)'],
                'direction1': row['Dir#1(2.50m)'], 
                'speed2': row['Speed#2(4.50m)'],
                'direction2': row['Dir#2(4.50m)'],  
                'speed3': row['Speed#3(6.50m)'],
                'direction3': row['Dir#3(6.50m)'],  
                'speed4': row['Speed#4(8.50m)'],
                'direction4': row['Dir#4(8.50m)'],  
                'speed5': row['Speed#5(10.50m)'],
                'direction5': row['Dir#5(10.50m)'],
                'speed6': row['Speed#6(12.50m)'],
                'direction6': row['Dir#6(12.50m)'],
                'speed7': row['Speed#7(14.50m)'],
                'direction7': row['Dir#7(14.50m)'],
                'speed8': row['Speed#8(16.50m)'],
                'direction8': row['Dir#8(16.50m)'],    
                'speed9': row['Speed#9(18.50m)'],
                'direction9': row['Dir#9(18.50m)'], 
                'speed10': row['Speed#10(20.50m)'],
                'direction10': row['Dir#10(20.50m)'],  
                'speed11': row['Speed#11(22.50m)'],
                'direction11': row['Dir#11(22.50m)'],   
                'speed12': row['Speed#12(24.50m)'],
                'direction12': row['Dir#12(24.50m)'],
                'speed13': row['Speed#13(26.50m)'],
                'direction13': row['Dir#13(26.50m)'], 
                'speed14': row['Speed#14(28.50m)'],
                'direction14': row['Dir#14(28.50m)'],      
                'speed15': row['Speed#15(30.50m)'],
                'direction15': row['Dir#15(30.50m)'],   
                'speed16': row['Speed#16(32.50m)'],
                'direction16': row['Dir#16(32.50m)'],   
                'speed17': row['Speed#17(34.50m)'],
                'direction17': row['Dir#17(34.50m)'],   
                'speed18': row['Speed#18(36.50m)'],
                'direction18': row['Dir#18(36.50m)'],   
                'speed19': row['Speed#19(38.50m)'],
                'direction19': row['Dir#19(38.50m)'],   
                'speed20': row['Speed#20(40.50m)'],
                'direction20': row['Dir#20(40.50m)'],   
                
            }

            data_list.append(data)

            if index < 5:
                print(f" Fila {index + 1} procesada: {data}")

        with transaction.atomic():
            objects = [SensorCorr(**data) for data in data_list]
            SensorCorr.objects.bulk_create(objects, batch_size=3000)

        return Response({"message": f"Datos actualizados correctamente para sensor {sensor.id}"}, status=status.HTTP_200_OK)
    

# Aqui creamos las vistas para el pequeño front
class SensorListView(View):
    def get(self, request):
        sensores = Sensor.objects.all()
        return render(request, 'sensores/lista.html', {'sensores': sensores})



def descargar_datos_sensor(request, sensor_id):

    """ Función que va a permitir la descarga de archivos de datos en el front"""


    sensor = Sensor.objects.get(id=sensor_id)
    datos = SensorLoad.objects.filter(sensor=sensor)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{sensor.name}_data.csv"'

    writer = csv.writer(response)
    writer.writerow(['Fecha', 'Peso'])

    for dato in datos:
        writer.writerow([dato.date_time, dato.weight])

    return response





class ProyectoSelectorView(View):
    def get(self, request):
        proyectos = ['aquamore', 'station']
        return render(request, 'proyectos/select.html', {'proyectos': proyectos})

def ver_proyecto(request):

    """ Función que va a mostrar todos los sensores y estaciones en el front"""


    proyecto = request.GET.get('proyecto')

    if proyecto == 'aquamore':
        return redirect('aquamore:lista_sensores')

    elif proyecto == 'station':
        return redirect('station:lista_sensores')

    else:
        return redirect('aquamore:select_proyecto')
