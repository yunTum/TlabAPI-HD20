# coding: utf-8

from rest_framework import routers
from django.urls import path
from .views import *

router = routers.DefaultRouter()

urlpatterns = [
    path('', test, name='test'),
    
    path('aichat/', AIchatList.as_view()),
    path('aichat/<int:pk>/', AIchatDetail.as_view()),
    ]