# detection_app/context_processors.py
from django.conf import settings

def site_info(request):
    """Context processor untuk informasi site"""
    return {
        'SITE_NAME': 'Hoax Detector ID',
        'SITE_DESCRIPTION': 'Sistem Deteksi Berita Hoax Indonesia',
        'SITE_VERSION': '1.0.0',
        'MODEL_NAME': 'IndoBERT Base P2',
        'DEBUG': settings.DEBUG,
    }