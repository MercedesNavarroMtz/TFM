from rest_framework import routers
from django.urls import include, path
from .views import SensorListView, descargar_datos_sensor, ProyectoSelectorView, ver_proyecto
from . import views

app_name = 'aquamore'

router = routers.DefaultRouter()
router.register(r'infrastructures', views.InfrastructureViewSet, basename='infrastructure')
router.register(r'sensors', views.SensorViewSet, basename='sensors')
router.register(r'load_sensors', views.SensorLoadViewSet, basename='load_sensors')
router.register(r'sclfloat_sensors', views.SensorSCLyFlotViewSet, basename='sclfloat_sensors')
router.register(r'corr_sensor', views.SensorCorrViewSet, basename='corr_sensor')

urlpatterns = [
    path('', include(router.urls)),

    path('sensores/', SensorListView.as_view(), name='lista_sensores'),
    path('sensores/<int:sensor_id>/descargar/', descargar_datos_sensor, name='descargar_datos_sensor'),

    path('seleccionar/', ProyectoSelectorView.as_view(), name='select_proyecto'),
    path('ver-proyecto/', ver_proyecto, name='ver_proyecto'),
]
