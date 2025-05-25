from rest_framework import serializers
from rest_framework.exceptions import ValidationError

from .models import *

#Los serializers convierten los datos de las tablas en JSON para que puedan las aplicaciones
#usar la información.

class InfrastructureSerializer(serializers.ModelSerializer):
    class Meta:
        model = Infrastructure # Modelo que va a convertir
        fields = '__all__'     # Todos los campos del modelo

class SensorSerializer(serializers.ModelSerializer):
    infrastructure = serializers.PrimaryKeyRelatedField(queryset=Infrastructure.objects.all())
    class Meta:
        model = Sensor
        fields = '__all__'

class SensorLoadSerializer(serializers.ModelSerializer):
    sensor = serializers.PrimaryKeyRelatedField(queryset=Sensor.objects.all())
    class Meta:
        model = SensorLoad
        fields = '__all__'

class SensorSCLyFlotSerializer(serializers.ModelSerializer):
    sensor = serializers.PrimaryKeyRelatedField(queryset=Sensor.objects.all())
    class Meta:
        model = SensorSCLyFlot
        fields = '__all__'

class SensorCorrSerializer(serializers.ModelSerializer):
    sensor = serializers.PrimaryKeyRelatedField(queryset=Sensor.objects.all())
    class Meta:
        model = SensorCorr
        fields = '__all__'

class FileUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    class Meta:
        fields = ['file']
