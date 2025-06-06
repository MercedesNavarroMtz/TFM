"""
ASGI config for modem project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.0/howto/deployment/asgi/
"""

import os
import django
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from channels.security.websocket import AllowedHostsOriginValidator
from django.core.asgi import get_asgi_application


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'data_hub.settings')
django.setup()

from cages.routing import websocket_urlpatterns

application = ProtocolTypeRouter({
        'http': get_asgi_application(),    
        "websocket": AuthMiddlewareStack(URLRouter(websocket_urlpatterns)), 
        # "websocket": AllowedHostsOriginValidator(
        #     AuthMiddlewareStack(URLRouter(websocket_urlpatterns))
        # ),
    })



