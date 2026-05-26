"""
Seed command: 2 Backend/Node.js-focused clients.
Run: python manage.py seed_clients2

Clients created:
  cl3@ff.dev  / pass: Test@1234  – FinLedger (fintech, wants APIs & microservices)
  cl4@ff.dev  / pass: Test@1234  – LogiTrack (logistics SaaS, needs Node.js backend)
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
        "email": "cl3@ff.dev",
        "first_name": "Amit",
        "last_name": "Khanna",
        "company_name": "FinLedger Solutions",
        "total_spent": "45000.00",
        "average_rating": "4.60",
        "total_reviews": 18,
    },
    {
        "email": "cl4@ff.dev",
        "first_name": "Deepa",
        "last_name": "Srinivasan",
        "company_name": "LogiTrack India Pvt Ltd",
        "total_spent": "27500.00",
        "average_rating": "4.75",
        "total_reviews": 14,
    },
]


class Command(BaseCommand):
    help = "Seeds 2 backend/Node.js-focused clients (batch 2)."

    @transaction.atomic
    def handle(self, *args, **options):
        self.stdout.write(self.style.MIGRATE_HEADING("Seeding clients batch 2 (backend/Node focus)..."))
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
