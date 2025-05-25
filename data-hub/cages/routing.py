from django.urls import re_path

from . import consumers

websocket_urlpatterns = [
    # re_path(r'ws/node/(?P<room_name>\w+)/', consumers.NodesConsumer.as_asgi()), 
    re_path(r'ws/nodes/$', consumers.NodesConsumer.as_asgi()), 
]

