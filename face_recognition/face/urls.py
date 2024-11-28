from django.contrib import admin
from django.urls import path
from .views import first, capture_face_data
from . import views
urlpatterns = [
    path('', first),
    path('capture_face_data/', capture_face_data, name='capture_face_data'),

]
