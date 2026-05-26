"""
Seed command: 2 Frontend-focused clients.
Run: python manage.py seed_clients1

Clients created:
  cl1@ff.dev  / pass: Test@1234  – PixelCraft Studio (product startup, heavy UI work)
  cl2@ff.dev  / pass: Test@1234  – ShopSprint (e-commerce, wants storefronts + dashboards)
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
        "email": "cl1@ff.dev",
        "first_name": "Rahul",
        "last_name": "Gupta",
        "company_name": "PixelCraft Studio",
        "total_spent": "18500.00",
        "average_rating": "4.80",
        "total_reviews": 12,
    },
    {
        "email": "cl2@ff.dev",
        "first_name": "Neha",
        "last_name": "Agarwal",
        "company_name": "ShopSprint Technologies",
        "total_spent": "32000.00",
        "average_rating": "4.70",
        "total_reviews": 21,
    },
]


class Command(BaseCommand):
    help = "Seeds 2 frontend-focused clients (batch 1)."

    @transaction.atomic
    def handle(self, *args, **options):
        self.stdout.write(self.style.MIGRATE_HEADING("Seeding clients batch 1 (frontend focus)..."))
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
