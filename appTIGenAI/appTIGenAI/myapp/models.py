from django.db import models

# Create your models here.

from django.db import models

class Image(models.Model):
    title = models.CharField(max_length=255)
    image = models.ImageField(upload_to='static/images/')

    class Meta:
        app_label = "myapp"

