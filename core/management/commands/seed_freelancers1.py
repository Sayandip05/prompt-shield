"""
Seed command: 5 Frontend-specialist freelancers.
Run: python manage.py seed_freelancers1

Freelancers created:
  fl1@ff.dev  / pass: Test@1234  – React/Next.js expert (PRO, high earnings)
  fl2@ff.dev  / pass: Test@1234  – Vue.js + Tailwind (PRO, mid earnings)
  fl3@ff.dev  / pass: Test@1234  – Angular enterprise dev (FREE, new)
  fl4@ff.dev  / pass: Test@1234  – Mobile React Native (PRO)
  fl5@ff.dev  / pass: Test@1234  – UI/UX + Figma (PRO, top rated)
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
        "email": "fl1@ff.dev",
        "first_name": "Arjun",
        "last_name": "Mehta",
        "bio": (
            "Senior React & Next.js developer with 6 years of experience building "
            "high-performance SaaS frontends. Obsessed with performance budgets, "
            "Core Web Vitals, and component architecture. Open-source contributor."
        ),
        "skills": ["React", "Next.js", "TypeScript", "GraphQL", "Tailwind CSS", "Jest"],
        "hourly_rate": "75.00",
        "tier": FreelancerProfile.SubscriptionTier.PRO,
        "total_earned": "28500.00",
        "average_rating": "4.90",
        "total_reviews": 41,
        "is_available": True,
    },
    {
        "email": "fl2@ff.dev",
        "first_name": "Priya",
        "last_name": "Nair",
        "bio": (
            "Vue.js and Nuxt specialist who loves building accessible, "
            "beautifully animated interfaces. 4 years shipping production apps "
            "for e-commerce and fintech clients globally."
        ),
        "skills": ["Vue.js", "Nuxt.js", "Tailwind CSS", "GSAP", "Pinia", "Cypress"],
        "hourly_rate": "60.00",
        "tier": FreelancerProfile.SubscriptionTier.PRO,
        "total_earned": "14200.00",
        "average_rating": "4.85",
        "total_reviews": 28,
        "is_available": True,
    },
    {
        "email": "fl3@ff.dev",
        "first_name": "Rohan",
        "last_name": "Das",
        "bio": (
            "Angular developer with enterprise background. Worked at TCS for 3 years "
            "building large-scale internal tools. Now freelancing and transitioning to "
            "smaller, more impactful projects."
        ),
        "skills": ["Angular", "RxJS", "NgRx", "TypeScript", "Material UI", "Jasmine"],
        "hourly_rate": "45.00",
        "tier": FreelancerProfile.SubscriptionTier.FREE,
        "total_earned": "2100.00",
        "average_rating": "4.60",
        "total_reviews": 6,
        "is_available": True,
    },
    {
        "email": "fl4@ff.dev",
        "first_name": "Sneha",
        "last_name": "Kulkarni",
        "bio": (
            "React Native developer specialising in cross-platform mobile apps. "
            "Delivered 8 apps to App Store & Play Store. Expert in Expo, push "
            "notifications, and offline-first architectures."
        ),
        "skills": ["React Native", "Expo", "TypeScript", "Redux", "Firebase", "Detox"],
        "hourly_rate": "70.00",
        "tier": FreelancerProfile.SubscriptionTier.PRO,
        "total_earned": "19800.00",
        "average_rating": "4.88",
        "total_reviews": 34,
        "is_available": False,
    },
    {
        "email": "fl5@ff.dev",
        "first_name": "Kavya",
        "last_name": "Sharma",
        "bio": (
            "Top-rated UI/UX designer and frontend developer. I bridge design and "
            "code — starting from Figma wireframes to pixel-perfect React implementations. "
            "Focused on micro-interactions and design systems."
        ),
        "skills": ["Figma", "React", "CSS Animations", "Storybook", "Framer Motion", "A11y"],
        "hourly_rate": "80.00",
        "tier": FreelancerProfile.SubscriptionTier.PRO,
        "total_earned": "34000.00",
        "average_rating": "4.97",
        "total_reviews": 52,
        "is_available": True,
    },
]


class Command(BaseCommand):
    help = "Seeds 5 frontend-specialist freelancers (batch 1)."

    @transaction.atomic
    def handle(self, *args, **options):
        self.stdout.write(self.style.MIGRATE_HEADING("Seeding freelancers batch 1 (frontend)..."))
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
