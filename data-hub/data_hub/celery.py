import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'data_hub.settings')

app = Celery('data_hub')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

# 'redis://:supersecretpassword@redis:6379/1'
app.conf.result_backend = os.environ.get('REDIS_CHANNEL_URL')
app.conf.broker_connection_retry_on_startup = True