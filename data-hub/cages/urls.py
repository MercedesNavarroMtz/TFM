from rest_framework import routers
from django.urls import include, path
from .import views


app_name = 'cages'

router = routers.DefaultRouter()
#router.register(r'node', views.NodeViewSet, basename = 'node')



urlpatterns = [
    path('', include(router.urls)),
    # path('injured-img/', views.InjuredView.as_view(), name='injured-img'),
    # path('ship-impact-zone/', views.getShipInfoImpactZoneViewSet.as_view(), name = 'ship-impact-zone'),
    # path('users-list/', views.keycloakInfoViewSet.as_view(), name = 'users-list'),


 ]