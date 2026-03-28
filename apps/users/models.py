from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    class Roles(models.TextChoices):
        CLIENT = "CLIENT", "Client"
        FREELANCER = "FREELANCER", "Freelancer"

    role = models.CharField(max_length=20, choices=Roles.choices, default=Roles.FREELANCER)


class FreelancerProfile(models.Model):
    user = models.OneToOneField("users.User", on_delete=models.CASCADE, related_name="freelancer_profile")
    bio = models.TextField(blank=True)


class ClientProfile(models.Model):
    user = models.OneToOneField("users.User", on_delete=models.CASCADE, related_name="client_profile")
    company_name = models.CharField(max_length=255, blank=True)
