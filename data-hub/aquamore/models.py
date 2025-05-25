from django.db import models

# Create your models here.

class SensorType(models.IntegerChoices):
    load = 0, ('Load')
    scl = 1, ('SCL')
    flot = 2, ('Floatability')
    current = 3, ('Currentmeter')

class InfrastructureType(models.IntegerChoices):
    cage = 0, ('Cage')
    buoy = 1, ('Buoy')
    longline = 2, ('LongLine')


class Infrastructure(models.Model):
    type = models.IntegerField(choices=InfrastructureType.choices)
    name = models.CharField(max_length=256)
    location = models.CharField(max_length=256, blank=True, null=True)
    user = models.CharField(max_length=256, blank=True, null=True)

    # table_type = 'reference'
    def __str__(self):
        return self.name

class Sensor(models.Model):
    infrastructure = models.ForeignKey(Infrastructure, on_delete=models.CASCADE)
    name = models.CharField(max_length=256)
    sensor_type = models.IntegerField(choices=SensorType.choices)

    # table_type = 'reference'
    def __str__(self):
        return self.name

class SensorLoad(models.Model):
    sensor = models.ForeignKey(Sensor, on_delete=models.CASCADE)
    date_time = models.DateTimeField()
    weight = models.FloatField(null=True)

class SensorSCLyFlot(models.Model):
    sensor = models.ForeignKey(Sensor, on_delete=models.CASCADE)
    date_time = models.DateTimeField()
    temperature =  models.FloatField(null=True)
    depth = models.FloatField(null=True)
    pitch = models.IntegerField(null=True)
    roll = models.IntegerField(null=True)
    head = models.IntegerField(null=True)
    four_p = models.IntegerField(null=True)
   
class SensorCorr(models.Model):
    sensor = models.ForeignKey(Sensor, on_delete=models.CASCADE)
    date_time = models.DateTimeField()
    heading = models.FloatField(null=True)
    pitch = models.FloatField(null=True)
    roll = models.FloatField(null=True)
    pressure = models.FloatField(null=True)
    temperature = models.FloatField(null=True)
    speed1 = models.FloatField(null=True)
    direction1 = models.FloatField(null=True)
    speed2 = models.FloatField(null=True)
    direction2 = models.FloatField(null=True)
    speed3 = models.FloatField(null=True)
    direction3 = models.FloatField(null=True)
    speed4 = models.FloatField(null=True)
    direction4 = models.FloatField(null=True)
    speed5 = models.FloatField(null=True)
    direction5 = models.FloatField(null=True)
    speed6 = models.FloatField(null=True)
    direction6 = models.FloatField(null=True)
    speed7 = models.FloatField(null=True)
    direction7 =models.FloatField(null=True)
    speed8 = models.FloatField(null=True)
    direction8 = models.FloatField(null=True)
    speed9 = models.FloatField(null=True)
    direction9 = models.FloatField(null=True)
    speed10 = models.FloatField(null=True)
    direction10 = models.FloatField(null=True)
    speed11 = models.FloatField(null=True)
    direction11 = models.FloatField(null=True)
    speed12 = models.FloatField(null=True)
    direction12 = models.FloatField(null=True)
    speed13 = models.FloatField(null=True)
    direction13 = models.FloatField(null=True)
    speed14 = models.FloatField(null=True)
    direction14 = models.FloatField(null=True)
    speed15 = models.FloatField(null=True)
    direction15 = models.FloatField(null=True)
    speed16 = models.FloatField(null=True)
    direction16 = models.FloatField(null=True)
    speed17 = models.FloatField(null=True)
    direction17 = models.FloatField(null=True)
    speed18 = models.FloatField(null=True)
    direction18 = models.FloatField(null=True)
    speed19 = models.FloatField(null=True)
    direction19 = models.FloatField(null=True)
    speed20 = models.FloatField(null=True)
    direction20 = models.FloatField(null=True)  

    
