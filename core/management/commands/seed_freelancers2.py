"""
Seed command: 5 Backend-specialist freelancers.
Run: python manage.py seed_freelancers2

Freelancers created:
  fl6@ff.dev  / pass: Test@1234  – Django/Python expert (PRO, veteran)
  fl7@ff.dev  / pass: Test@1234  – Node.js/Express + MongoDB (PRO)
  fl8@ff.dev  / pass: Test@1234  – FastAPI + async Python (PRO)
  fl9@ff.dev  / pass: Test@1234  – Go microservices (FREE, new)
  fl10@ff.dev / pass: Test@1234  – Ruby on Rails full-stack (PRO)
"""

import logging

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.db import transaction

from apps.users.models import FreelancerProfile

User = get_user_model()
logger = logging.getLogger(__name__)

PASSWORD = "Test@1234"

FREELANCERS = [
    {
        "email": "fl6@ff.dev",
        "first_name": "Vikram",
        "last_name": "Rao",
        "bio": (
            "Django & Python veteran with 8 years of production experience. "
            "Designed REST APIs for fintech startups serving 1M+ users. "
            "Expert in Celery, Redis, PostgreSQL query optimisation, and DRF."
        ),
        "skills": ["Python", "Django", "DRF", "Celery", "Redis", "PostgreSQL", "Docker"],
        "hourly_rate": "85.00",
        "tier": FreelancerProfile.SubscriptionTier.PRO,
        "total_earned": "47000.00",
        "average_rating": "4.92",
        "total_reviews": 63,
        "is_available": True,
    },
    {
        "email": "fl7@ff.dev",
        "first_name": "Aditya",
        "last_name": "Verma",
        "bio": (
            "Node.js and Express developer with deep MongoDB expertise. "
            "Built real-time apps using Socket.IO and WebSockets. "
            "Passionate about clean API design and microservice decomposition."
        ),
        "skills": ["Node.js", "Express", "MongoDB", "Socket.IO", "JWT", "AWS Lambda"],
        "hourly_rate": "65.00",
        "tier": FreelancerProfile.SubscriptionTier.PRO,
        "total_earned": "21500.00",
        "average_rating": "4.78",
        "total_reviews": 38,
        "is_available": True,
    },
    {
        "email": "fl8@ff.dev",
        "first_name": "Meera",
        "last_name": "Iyer",
        "bio": (
            "FastAPI and async Python specialist. I build blazing-fast APIs with "
            "full OpenAPI documentation, Pydantic validation, and SQLAlchemy async. "
            "Also experienced with Kafka and event-driven architectures."
        ),
        "skills": ["FastAPI", "Python", "SQLAlchemy", "Pydantic", "Kafka", "PostgreSQL"],
        "hourly_rate": "72.00",
        "tier": FreelancerProfile.SubscriptionTier.PRO,
        "total_earned": "18900.00",
        "average_rating": "4.83",
        "total_reviews": 29,
        "is_available": True,
    },
    {
        "email": "fl9@ff.dev",
        "first_name": "Rahul",
        "last_name": "Bose",
        "bio": (
            "Go developer transitioning from Java backend. Building microservices "
            "with gRPC and REST. New to freelancing but strong fundamentals — "
            "3 years at a Bengaluru SaaS company."
        ),
        "skills": ["Go", "gRPC", "REST", "PostgreSQL", "Docker", "Kubernetes"],
        "hourly_rate": "40.00",
        "tier": FreelancerProfile.SubscriptionTier.FREE,
        "total_earned": "1800.00",
        "average_rating": "4.50",
        "total_reviews": 4,
        "is_available": True,
    },
    {
        "email": "fl10@ff.dev",
        "first_name": "Tanvi",
        "last_name": "Joshi",
        "bio": (
            "Ruby on Rails developer with a full-stack bent. 5 years building "
            "B2B SaaS platforms. Love convention-over-configuration and shipping "
            "features fast. Also comfortable with Hotwire and Turbo."
        ),
        "skills": ["Ruby on Rails", "Ruby", "PostgreSQL", "Hotwire", "Sidekiq", "RSpec"],
        "hourly_rate": "68.00",
        "tier": FreelancerProfile.SubscriptionTier.PRO,
        "total_earned": "22300.00",
        "average_rating": "4.75",
        "total_reviews": 33,
        "is_available": False,
    },
]


class Command(BaseCommand):
    help = "Seeds 5 backend-specialist freelancers (batch 2)."

    @transaction.atomic
    def handle(self, *args, **options):
        self.stdout.write(self.style.MIGRATE_HEADING("Seeding freelancers batch 2 (backend)..."))
        created = 0
        for data in FREELANCERS:
            if User.objects.filter(email=data["email"]).exists():
                self.stdout.write(f"  [SKIP] {data['email']} already exists.")
                continue

            user = User(
                email=data["email"],
                first_name=data["first_name"],
                last_name=data["last_name"],
                role=User.Roles.FREELANCER,
            )
            user.set_password(PASSWORD)
            user.save()

            profile, _ = FreelancerProfile.objects.get_or_create(user=user)
            profile.bio = data["bio"]
            profile.skills = data["skills"]
            profile.hourly_rate = data["hourly_rate"]
            profile.subscription_tier = data["tier"]
            profile.total_earned = data["total_earned"]
            profile.average_rating = data["average_rating"]
            profile.total_reviews = data["total_reviews"]
            profile.is_available = data["is_available"]
            profile.save(update_fields=[
                "bio", "skills", "hourly_rate", "subscription_tier",
                "total_earned", "average_rating", "total_reviews", "is_available",
            ])
            self.stdout.write(self.style.SUCCESS(f"  [OK] {data['email']} — {data['first_name']} {data['last_name']}"))
            created += 1

        self.stdout.write(self.style.SUCCESS(f"\nDone. {created} freelancers created."))
        self.stdout.write("Password for all: Test@1234")
