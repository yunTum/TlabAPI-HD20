from django.db import models

class APITest(models.Model):
    InputText = models.CharField(max_length=128)
    OutputText = models.CharField(max_length=128)
    created_at = models.DateTimeField(auto_now_add=True)

class AIchat(models.Model):
    InputText = models.CharField(max_length=128,default="")
    OutputText = models.CharField(max_length=128,default="")
    created_at = models.DateTimeField(auto_now_add=True)

class LearningData(models.Model):
    epochtime = models.IntegerField()
    txtfile = models.FileField()