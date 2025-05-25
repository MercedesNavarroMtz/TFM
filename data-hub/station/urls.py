from rest_framework import routers
from django.urls import include, path
from .import views
from .views import EstacionListView, descargar_valores_estacion # o la vista equivalente



app_name = 'cages_aquamore'

router = routers.DefaultRouter()
router.register(r'estacion', views.EstacionViewSet, basename = 'estacion')
router.register(r'values', views.ValuesViewSet, basename = 'values')



urlpatterns = [
    path('', include(router.urls)),
    path('sensores/', EstacionListView.as_view(), name='lista_sensores'),
    path('sensores/<int:estacion_id>/descargar/', descargar_valores_estacion, name='descargar_valores_estacion'),

    # path('injured-img/', views.InjuredView.as_view(), name='injured-img'),
    # path('ship-impact-zone/', views.getShipInfoImpactZoneViewSet.as_view(), name = 'ship-impact-zone'),
    # path('users-list/', views.keycloakInfoViewSet.as_view(), name = 'users-list'),


 ]