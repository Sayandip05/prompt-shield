"""
Seed command: 2 DevOps/Infrastructure-focused clients.
Run: python manage.py seed_clients3

Clients created:
  cl5@ff.dev  / pass: Test@1234  – CloudNova (SaaS, needs AWS + CI/CD + K8s)
  cl6@ff.dev  / pass: Test@1234  – SecureBase (cybersecurity firm, needs infra hardening)
"""

import logging

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.db import transaction

from apps.users.models import ClientProfile

User = get_user_model()
logger = logging.getLogger(__name__)

PASSWORD = "Test@1234"

CLIENTS = [
    {
        "email": "cl5@ff.dev",
        "first_name": "Sanjay",
        "last_name": "Malhotra",
        "company_name": "CloudNova Systems",
        "total_spent": "61000.00",
        "average_rating": "4.85",
        "total_reviews": 29,
    },
    {
        "email": "cl6@ff.dev",
        "first_name": "Ritu",
        "last_name": "Saxena",
        "company_name": "SecureBase Infosec",
        "total_spent": "38000.00",
        "average_rating": "4.90",
        "total_reviews": 16,
    },
]


class Command(BaseCommand):
    help = "Seeds 2 DevOps/infrastructure-focused clients (batch 3)."

    @transaction.atomic
    def handle(self, *args, **options):
        self.stdout.write(self.style.MIGRATE_HEADING("Seeding clients batch 3 (DevOps/infra focus)..."))
        created = 0
        for data in CLIENTS:
            if User.objects.filter(email=data["email"]).exists():
                self.stdout.write(f"  [SKIP] {data['email']} already exists.")
                continue

            user = User(
                email=data["email"],
                first_name=data["first_name"],
                last_name=data["last_name"],
                role=User.Roles.CLIENT,
            )
            user.set_password(PASSWORD)
            user.save()

            profile, _ = ClientProfile.objects.get_or_create(user=user)
            profile.company_name = data["company_name"]
            profile.total_spent = data["total_spent"]
            profile.average_rating = data["average_rating"]
            profile.total_reviews = data["total_reviews"]
            profile.save(update_fields=["company_name", "total_spent", "average_rating", "total_reviews"])
            self.stdout.write(self.style.SUCCESS(
                f"  [OK] {data['email']} — {data['company_name']}"
            ))
            created += 1

        self.stdout.write(self.style.SUCCESS(f"\nDone. {created} clients created."))
        self.stdout.write("Password for all: Test@1234")
