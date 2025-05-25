from rest_framework import serializers
from .models import *

class EstacionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Estacion
        fields = '__all__'

class ValuesSerializer(serializers.ModelSerializer):
    station  = serializers.PrimaryKeyRelatedField(queryset=Estacion.objects.all())
    class Meta:
        model = Values
        fields = '__all__'


class FileUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    class Meta:
        fields = ['file']