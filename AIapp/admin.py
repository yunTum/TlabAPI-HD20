from django.contrib import admin
from .models import *

@admin.register(AIchat)
class AIchat(admin.ModelAdmin):
    pass