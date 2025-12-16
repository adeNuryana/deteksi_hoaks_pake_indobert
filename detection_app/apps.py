# detection_app/apps.py
from django.apps import AppConfig

class DetectionAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detection_app'
    
    # def ready(self):
    #     # Import signal handlers
    #     import detection_app.signals
