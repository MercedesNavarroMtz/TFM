from django.db import models

# Create your models here.

class Entidad(models.IntegerChoices):
    AEV = 0, ('AEV')
    APC = 1, ('APC')
    CTD = 2, ('CTD')
    DMA = 3, ('DMA')
    PVA = 4, ('PVA')

class Parametro(models.TextChoices):
    OXIGENO = 'OXIGENO', 'Oxigeno'
    CLOROFILA = 'CLOROFILA', 'Clorofila'
    SALINIDAD = 'SALINIDAD', 'Salinidad'
    TURBIDEZ = 'TURBIDEZ', 'Turbidez'
    TRANSPARENCIA = 'TRANSPARENCIA', 'Transparencia'
    TEMPERATURA = 'TEMPERATURA', 'Temperatura'

class Estacion(models.Model):
    measure_entity = models.IntegerField(choices=Entidad.choices)   
    station = models.CharField(null=True)  
    latitude = models.IntegerField()
    longitude = models.IntegerField()
    location = models.CharField(max_length=255, null=True)

    def __str__(self):
        return self.station

class Values(models.Model):
    station = models.ForeignKey(Estacion, on_delete=models.CASCADE)
    datetime = models.DateTimeField()
    parameter = models.TextField(max_length=255, choices=Parametro)
    value = models.FloatField()
    units = models.CharField(max_length=10, null= True)
    measure =  models.CharField(max_length=10, null=True) 
    final_datetime = models.DateTimeField(null=True)
    method = models.CharField(null=True, max_length=255)
    replica = models.IntegerField(null=True)
    
    PROFUNDIDAD = [
        ('FONDO', 'FONDO'),
        ('SUPERFICIE', 'SUPERFICIE'),
    ]           
    depth = models.CharField(null= True, max_length=20, choices=PROFUNDIDAD)

