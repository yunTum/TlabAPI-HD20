# coding: utf-8

from rest_framework import serializers
from .models import *

class AIchatSerializer(serializers.ModelSerializer):
    class Meta:
        model = AIchat
        fields = '__all__'

class AILearningSerializer(serializers.ModelSerializer):
    class Meta:
        model = LearningData
        fields = '__all__'