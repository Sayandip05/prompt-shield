"""
Seed command: 5 DevOps/Cloud/AI-specialist freelancers.
Run: python manage.py seed_freelancers3

Freelancers created:
  fl11@ff.dev / pass: Test@1234  – AWS/Terraform DevOps (PRO)
  fl12@ff.dev / pass: Test@1234  – Kubernetes & Docker (PRO)
  fl13@ff.dev / pass: Test@1234  – AI/ML Python engineer (PRO, top earner)
  fl14@ff.dev / pass: Test@1234  – Data engineer / dbt (FREE, new)
  fl15@ff.dev / pass: Test@1234  – Security / penetration tester (PRO)
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
        "email": "fl11@ff.dev",
        "first_name": "Karthik",
        "last_name": "Subramanian",
        "bio": (
            "AWS Certified Solutions Architect with 7 years of DevOps experience. "
            "Expert in Terraform, CloudFormation, ECS Fargate, RDS, and cost optimisation. "
            "Helped 15+ startups migrate to AWS with zero-downtime deployments."
        ),
        "skills": ["AWS", "Terraform", "CloudFormation", "ECS", "GitHub Actions", "Python"],
        "hourly_rate": "90.00",
        "tier": FreelancerProfile.SubscriptionTier.PRO,
        "total_earned": "52000.00",
        "average_rating": "4.94",
        "total_reviews": 57,
        "is_available": True,
    },
    {
        "email": "fl12@ff.dev",
        "first_name": "Ananya",
        "last_name": "Pillai",
        "bio": (
            "Kubernetes and Docker specialist. I set up production-grade k8s clusters, "
            "Helm chart deployments, Prometheus/Grafana monitoring stacks, and "
            "cost-efficient auto-scaling policies for mid-size engineering teams."
        ),
        "skills": ["Kubernetes", "Docker", "Helm", "Prometheus", "Grafana", "ArgoCD"],
        "hourly_rate": "82.00",
        "tier": FreelancerProfile.SubscriptionTier.PRO,
        "total_earned": "38600.00",
        "average_rating": "4.89",
        "total_reviews": 44,
        "is_available": True,
    },
    {
        "email": "fl13@ff.dev",
        "first_name": "Siddharth",
        "last_name": "Ghosh",
        "bio": (
            "AI/ML engineer and LLM integrator. I fine-tune models, build RAG pipelines, "
            "and ship LangChain/LangGraph agents to production. Former ML engineer at "
            "a Bengaluru AI startup. Top 5% earner on the platform."
        ),
        "skills": ["Python", "LangChain", "LangGraph", "OpenAI", "HuggingFace", "FastAPI", "Pinecone"],
        "hourly_rate": "110.00",
        "tier": FreelancerProfile.SubscriptionTier.PRO,
        "total_earned": "71000.00",
        "average_rating": "4.96",
        "total_reviews": 48,
        "is_available": True,
    },
    {
        "email": "fl14@ff.dev",
        "first_name": "Pooja",
        "last_name": "Reddy",
        "bio": (
            "Data engineer with 2 years of industry experience in ETL pipelines, "
            "dbt transformations, and Airflow orchestration. "
            "Learning Spark and looking for interesting data projects to grow my portfolio."
        ),
        "skills": ["dbt", "Airflow", "Python", "SQL", "BigQuery", "Snowflake"],
        "hourly_rate": "42.00",
        "tier": FreelancerProfile.SubscriptionTier.FREE,
        "total_earned": "3200.00",
        "average_rating": "4.65",
        "total_reviews": 7,
        "is_available": True,
    },
    {
        "email": "fl15@ff.dev",
        "first_name": "Nikhil",
        "last_name": "Tiwari",
        "bio": (
            "Offensive security engineer and certified ethical hacker (CEH, OSCP). "
            "I perform web app penetration testing, code audits, and security architecture reviews. "
            "Helped 20+ companies find and fix critical vulnerabilities before attackers did."
        ),
        "skills": ["Penetration Testing", "OWASP", "Burp Suite", "Python", "Linux", "AWS Security"],
        "hourly_rate": "95.00",
        "tier": FreelancerProfile.SubscriptionTier.PRO,
        "total_earned": "43500.00",
        "average_rating": "4.91",
        "total_reviews": 39,
        "is_available": False,
    },
]


class Command(BaseCommand):
    help = "Seeds 5 DevOps/AI-specialist freelancers (batch 3)."

    @transaction.atomic
    def handle(self, *args, **options):
        self.stdout.write(self.style.MIGRATE_HEADING("Seeding freelancers batch 3 (DevOps/AI)..."))
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
