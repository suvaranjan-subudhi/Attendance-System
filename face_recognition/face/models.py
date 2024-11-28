from django.db import models

# Create your models here.
from django.db import models

class FaceData(models.Model):
    employee_id = models.CharField(max_length=255)  # Adjust max_length as needed
    face_encoding = models.BinaryField()  # Stores the serialized face data

    created_at = models.DateTimeField(auto_now_add=True)  # Automatically add timestamp

    def __str__(self):
        return f"FaceData for Employee ID: {self.employee_id}"
