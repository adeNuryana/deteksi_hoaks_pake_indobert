# detection_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # Public pages
    path('', views.home, name='home'),
    path('detect/', views.detect_news, name='detect'),
    path('detect/url/', views.detect_from_url, name='detect_url'),
    path('detect/batch/', views.batch_detect, name='batch_detect'),
    path('about/', views.about, name='about'),
    path('api/docs/', views.api_docs, name='api_docs'),
    
    # Auth
    path('register/', views.register, name='register'),
    
    # User features (require login)
    path('history/', views.detection_history, name='detection_history'),
    path('history/<int:detection_id>/', views.detection_detail, name='detection_detail'),
    path('history/export/<str:format_type>/', views.export_history, name='export_history'),
    
    # API endpoints
    path('api/detect/', views.api_detect, name='api_detect'),
    path('api/stats/', views.api_stats, name='api_stats'),
]