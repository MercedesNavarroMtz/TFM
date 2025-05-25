from django.shortcuts import render
from rest_framework import viewsets,status, permissions
import pandas as pd
from django_filters import rest_framework as filters
import io
import os
from rest_framework.decorators import action
from .models import *
from .serializers import *
from rest_framework.response import Response
from datetime import datetime
from django.db import transaction
from rest_framework.decorators import action
from django_filters.rest_framework import DjangoFilterBackend, FilterSet, DateTimeFilter, NumberFilter, CharFilter
import csv
from .models import *
from .serializers import *

# Create your views here.
class EstacionViewSet(viewsets.ModelViewSet):
    queryset = Estacion.objects.all()
    permission_classes = (permissions.AllowAny,)
    serializer_class = EstacionSerializer
    # filter_backends = (filters.DjangoFilterBackend,)

    @action(detail=False, methods=['post'], url_path='upload-csv')
    def upload_csv(self, request):
        file = request.FILES.get('file')
        if not file:
            return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)

        try:

            filename = file.name
            print(f"Nombre del archivo recibido: {filename}")

            entity_prefix = os.path.splitext(filename)[0].split('_')[0].upper()
            print(f"Entidad detectada: {entity_prefix}")

            try:
                measure_entity = Entidad[entity_prefix].value
            except KeyError:
                return Response({'error': f'Entidad desconocida: {entity_prefix}'}, status=status.HTTP_400_BAD_REQUEST)

            decoded_file = file.read().decode('latin-1')
            io_string = io.StringIO(decoded_file)
            reader = csv.DictReader(io_string, delimiter=';')
            if reader.fieldnames and reader.fieldnames[0].startswith('ï»¿'):
                reader.fieldnames[0] = reader.fieldnames[0].replace('ï»¿', '')
            estaciones_creadas = 0
            estaciones_vistas = set()
            print("Columnas detectadas en el CSV:", reader.fieldnames)


            for row in reader:

                station_name = row.get('Estacion')
                if not station_name or station_name in estaciones_vistas:
                    continue

                estaciones_vistas.add(station_name)

                latitude = row.get('Cordenada X')
                longitude = row.get('Cordenada Y')
                location = row.get('Localizacion_de_la_estacion')

                # Comprobamos si ya existe en la base de datos
                if not Estacion.objects.filter(station=station_name).exists():
                    Estacion.objects.create(
                        station=station_name,
                        latitude=int(latitude),
                        longitude=int(longitude),
                        measure_entity=measure_entity,
                        location=location,
                    )
                    estaciones_creadas += 1

            return Response({'created': estaciones_creadas}, status=status.HTTP_201_CREATED)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        



def convertir_fecha(fecha_string):
    """Convierte una fecha de formato 'DD/MM/YYYY HH:MM' a 'YYYY-MM-DD HH:MM'"""
    if not fecha_string or fecha_string.strip() == '':
        return None
    
    try:
        try:
            fecha_obj = datetime.strptime(fecha_string, '%d/%m/%Y %H:%M:%S')
        except ValueError:
            try:
                fecha_obj = datetime.strptime(fecha_string, '%d/%m/%Y %H:%M')
            except ValueError:
                # Intentar un formato adicional común
                fecha_obj = datetime.strptime(fecha_string, '%d/%m/%Y')
            
        # Devolver la fecha en formato string ISO
        return fecha_obj.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"Error al convertir la fecha: {e} - Fecha original: '{fecha_string}'")
        return None



class ValuesFilter(filters.FilterSet):
    station_id = filters.NumberFilter(field_name='station__id', lookup_expr='exact')
    station_name = filters.CharFilter(field_name='station__station', lookup_expr='icontains')
    entity = filters.NumberFilter(field_name='station__measure_entity', lookup_expr='exact')
    entity_name = filters.CharFilter(method='filter_entity_name')
    
    class Meta:
        model = Values
        fields = ['station_id', 'station_name', 'entity', 'entity_name']

    def filter_entity_name(self, queryset, name, value):

        try:
            # Buscar el valor entero correspondiente al nombre de la entidad
            entity_value = Entidad[value.upper()].value
            return queryset.filter(station__measure_entity=entity_value)
        except (KeyError, AttributeError):
            # Si el nombre no es válido, devolver queryset vacío
            return queryset.none()

class ValuesViewSet(viewsets.ModelViewSet):
    queryset = Values.objects.all()
    permission_classes = (permissions.AllowAny,)
    serializer_class = ValuesSerializer
    filter_backends = (filters.DjangoFilterBackend,)
    filterset_class = ValuesFilter

    @action(detail=False, methods=['post'], url_path='upload-csv')
    def upload_csv(self, request):
        file = request.FILES.get('file')
        if not file:
            return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            decoded_file = file.read().decode('latin-1')
            io_string = io.StringIO(decoded_file)
            reader = csv.DictReader(io_string, delimiter=';')
            print(reader.fieldnames)
            if reader.fieldnames and reader.fieldnames[0].startswith('ï»¿'):
                reader.fieldnames[0] = reader.fieldnames[0].replace('ï»¿', '')
            print(f"Columnas detectadas en el CSV: {reader.fieldnames}")
            values_created = 0

            for i, row in enumerate(reader):
                # Leer los campos de la estación y otros datos
                station_name = row.get('Estacion')
                location = row.get('Localizacion_de_la_estacion')
                latitude = row.get('Cordenada X')
                longitude = row.get('Cordenada Y')
                fecha = row.get('Fecha')
                parametro = row.get('Parametro')
                value = row.get('Valor')
                unidad = row.get('Unidad')
                medida = row.get('Medida')
                fecha_final = row.get('Fecha_fin')
                metodo = row.get('Metodo')
                replica = row.get('Replica')
                profundidad = row.get('Profundidad')

                try:
                    latitude_int = int(float(latitude)) if latitude else None
                    longitude_int = int(float(longitude)) if longitude else None
                except ValueError:
                    print(" → Coordenadas inválidas:", latitude, longitude)
                    continue

                if fecha:  # Verificar que fecha no esté vacía o sea solo espacios
                    fecha_convertida = convertir_fecha(fecha)
                    if not fecha_convertida:
                        # print(f" → Fecha inválida en la fila {i}: '{fecha}', se omite.")
                        continue
                    fecha = fecha_convertida
                else:
                    # print(f" → Fecha vacía en la fila {i}, se omite.")
                    continue


                # if fecha_final: # Verificar que fecha no esté vacía o sea solo espacios
                #     fecha_convertida = convertir_fecha(fecha_final)
                #     if not fecha_convertida:
                #         print(f" → Fecha inválida en la fila {i}: '{fecha_final}', se omite.")
                #         continue
                #     fecha_final = fecha_convertida
                # else:
                #     print(f" → Fecha vacía en la fila {i}, se omite.")
                #     continue

            # Convertir el valor (value) a número
                if value:
                    try:
                        value = value.replace(',', '.')  # Reemplazar coma por punto
                        value = float(value)  # Convertir a float
                    except ValueError:
                        print(f" → Valor inválido en la fila {i}, se omite.")
                        continue

                # Validar el parámetro
                if parametro not in Parametro.values:
                    print(f" → Parámetro desconocido: {parametro}. Se omite.")
                    continue

                # Comprobar si la estación ya existe en la base de datos
                estacion = Estacion.objects.filter(station=station_name).first()
                if not estacion:
                    print(f" → Estación {station_name} no encontrada en la base de datos.")
                    continue

                # Crear o actualizar los datos relacionados en el modelo Values
                Values.objects.create(
                    station=estacion,
                    datetime=fecha,
                    parameter=parametro,
                    value=value,
                    units=unidad if unidad else None,
                    measure=medida if medida else None,
                    final_datetime=fecha_final if fecha_final else None,
                    method=metodo if metodo else None,
                    replica=replica if replica else None,
                    depth=profundidad if profundidad else None
                )
                values_created += 1

            print(f"\nTotal valores creados: {values_created}")
            return Response({'created': values_created}, status=status.HTTP_201_CREATED)

        except Exception as e:
            print("Error al procesar el CSV:", e)
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        
from django.views import View
from django.shortcuts import render
from django.http import HttpResponse

class EstacionListView(View):
    def get(self, request):
        estaciones = Estacion.objects.all()
        return render(request, 'station/lista_estaciones.html', {'estaciones': estaciones})


def descargar_valores_estacion(request, estacion_id):
    estacion = Estacion.objects.get(id=estacion_id)
    datos = Values.objects.filter(station=estacion)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{estacion.station}_valores.csv"'

    writer = csv.writer(response)
    writer.writerow(['Fecha', 'Parámetro', 'Valor', 'Unidad'])

    for d in datos:
        writer.writerow([d.datetime, d.parameter, d.value, d.units])

    return response
