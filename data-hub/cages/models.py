from django.contrib.gis.db import models
from django.utils.translation import gettext_lazy as _


class SENSOR_TYPE(models.IntegerChoices):
    load = 0, _('Load')
    scl = 1, _('SCL')

class Cage(models.Model):
    name = models.CharField(max_length=256)
    location = models.CharField(max_length=256, blank=True, null=True)
    user = models.CharField(max_length=256, blank=True, null=True)

    table_type = 'reference'


class Sensor(models.Model):
    cage = models.ForeignKey(Cage, on_delete=models.CASCADE)
    name = models.CharField(max_length=256)
    sensor_type = models.IntegerField(default=SENSOR_TYPE.load, choices=SENSOR_TYPE.choices)

    table_type = 'reference'

class SensorLoad(models.Model):
    sensor = models.ForeignKey(Sensor, on_delete=models.CASCADE)
    date_time = models.DateTimeField()
    weight = models.FloatField(null=True)
    
    table_type = 'distributed'
    partition_column = 'sensor_id'

    class Meta:
        unique_together = [('sensor', 'date_time')] 
        # indexes = [
        #     models.Index(fields=['sensor'], name='cages_sensor_idx'),
        #     models.Index(fields=['date_time'], name='cages_sensor_date_time_idx'),
        # ]
    




