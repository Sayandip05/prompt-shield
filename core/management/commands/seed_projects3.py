"""
Seed command: 5 COMPLETED DevOps projects (full lifecycle — released payments, deliverables).
Run: python manage.py seed_projects3

Depends on: seed_clients3, seed_freelancers3
Projects (all COMPLETED with released payments and approved deliverables):
  1. AWS ECS Production Deployment      – cl5 + fl11 (COMPLETED, payment RELEASED)
  2. Kubernetes Cluster Setup           – cl5 + fl12 (COMPLETED, payment RELEASED)
  3. AI Chatbot Integration (LangChain) – cl6 + fl13 (COMPLETED, payment RELEASED)
  4. Penetration Test & Security Audit  – cl6 + fl15 (COMPLETED, payment RELEASED)
  5. CI/CD Pipeline (GitHub Actions)    – cl5 + fl11 (COMPLETED, payment RELEASED)
"""

import logging
import uuid
from datetime import date, timedelta

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from apps.bidding.models import Bid, Contract
from apps.messaging.models import Conversation, Message
from apps.notifications.models import Notification
from apps.payments.models import Escrow, Payment, PlatformEarning
from apps.projects.models import Project, ProjectSkill
from apps.worklogs.models import Deliverable, DeliveryProof, WorkLog

User = get_user_model()
logger = logging.getLogger(__name__)


def _build_completed_project(client, freelancer, title, description, budget,
                              skills, days_ago_started, days_duration, stdout):
    """
    Create a fully completed project: project → bid (accepted) → contract (closed)
    → payment (RELEASED) → escrow (released) → worklogs (approved) → deliverables (approved)
    → delivery proof.
    """
    now = timezone.now()

    if Project.objects.filter(title=title).exists():
        stdout.write(f"  [SKIP] '{title}' already exists.")
        return

    started = now - timedelta(days=days_ago_started)
    ended = started + timedelta(days=days_duration)

    project = Project.objects.create(
        client=client,
        title=title,
        description=description,
        budget=budget,
        deadline=(started + timedelta(days=days_duration)).date(),
        status=Project.Status.COMPLETED,
    )
    for skill in skills:
        ProjectSkill.objects.get_or_create(project=project, skill_name=skill)

    bid = Bid.objects.create(
        project=project,
        freelancer=freelancer,
        amount=budget,
        cover_letter=f"I am the best fit for '{title}'. Delivered on time, every time.",
        status=Bid.Status.ACCEPTED,
    )

    contract = Contract.objects.create(
        bid=bid,
        agreed_amount=budget,
        end_date=ended,
        is_active=False,
    )

    payment = Payment.objects.create(
        contract=contract,
        total_amount=budget,
        status=Payment.Status.RELEASED,
        razorpay_order_id=f"order_done_{project.id:04d}",
        razorpay_payment_id=f"pay_done_{project.id:04d}",
        razorpay_payout_id=f"payout_done_{project.id:04d}",
    )
    platform_cut = round(float(budget) * 0.10, 2)
    escrow = Escrow.objects.create(payment=payment, held_amount=budget, released_at=ended)
    PlatformEarning.objects.create(
        payment=payment,
        cut_percentage="10.00",
        cut_amount=str(platform_cut),
    )

    # Worklogs — one per day of the contract
    log_descriptions = [
        "Kick-off call, requirements gathering, and environment setup.",
        "Architecture design and tech stack decisions documented.",
        "Core implementation started — scaffolding and base configuration.",
        "Main feature development — first milestone complete.",
        "Testing, bug fixes, and performance review.",
        "Final integration, documentation, and client handover walkthrough.",
    ]
    wl_count = min(days_duration, len(log_descriptions))
    total_hours = 0
    for i in range(wl_count):
        log_date = started.date() + timedelta(days=i)
        if WorkLog.objects.filter(contract=contract, date=log_date).exists():
            continue
        wl = WorkLog.objects.create(
            contract=contract,
            freelancer=freelancer,
            date=log_date,
            description=log_descriptions[i],
            hours_worked="8.00",
            status=WorkLog.Status.APPROVED,
            approved_at=now,
            approved_by=client,
        )
        total_hours += 8

    # Deliverables (3 approved)
    deliverable_data = [
        ("Requirements & Architecture Document",
         "Delivered a detailed architecture doc covering infra design, tech choices, and API contracts."),
        ("Implementation — Core Feature",
         "All core features implemented, tested, and passing CI pipeline."),
        ("Final Delivery & Handover",
         "Full documentation, runbook, and live walkthrough with client team."),
    ]
    approved_count = 0
    for d_title, d_desc in deliverable_data:
        Deliverable.objects.create(
            contract=contract,
            freelancer=freelancer,
            title=d_title,
            description=d_desc,
            status=Deliverable.Status.APPROVED,
            submitted_at=ended - timedelta(days=1),
            reviewed_at=ended,
            reviewed_by=client,
            client_feedback="Excellent work — exactly what we needed.",
            hours_logged="16.00",
            payment_released=True,
        )
        approved_count += 1

    # Delivery proof
    DeliveryProof.objects.create(
        contract=contract,
        pdf_url=f"https://s3.example.com/proofs/{project.id}/delivery_proof.pdf",
        total_hours=str(total_hours),
        total_logs_count=wl_count,
        total_deliverables=len(deliverable_data),
        approved_deliverables=approved_count,
        report_id=str(uuid.uuid4()).replace("-", "")[:24],
    )

    # Conversation
    conv, _ = Conversation.objects.get_or_create(contract=contract)
    chat = [
        (client, f"Welcome! Let's get '{title}' started."),
        (freelancer, "Thanks! I'll begin with the architecture doc and share it by EOD."),
        (client, "Architecture looks great — approved. Please proceed."),
        (freelancer, "Core implementation complete. Opening PR for review."),
        (client, "PR merged and tested. Looks excellent. Ready for final delivery?"),
        (freelancer, "Yes! Final delivery doc and runbook are attached. Great working with you!"),
        (client, "Amazing work. Releasing payment now. Will definitely work together again!"),
    ]
    for sender, content in chat:
        Message.objects.create(conversation=conv, sender=sender, content=content, is_read=True)

    # Notifications
    Notification.objects.create(
        recipient=freelancer,
        title="Payment Released",
        body=f"INR {budget} has been released to your account for '{title}'.",
        type=Notification.Type.PAYMENT_RELEASED,
    )
    Notification.objects.create(
        recipient=client,
        title="Project Completed",
        body=f"'{title}' has been completed. Delivery proof is available.",
        type=Notification.Type.PROOF_READY,
    )

    stdout(
        f"  [OK] COMPLETED '{title}' — INR {budget} RELEASED, "
        f"{wl_count} logs, {approved_count} deliverables"
    )


class Command(BaseCommand):
    help = "Seeds 5 COMPLETED DevOps projects with full payment lifecycle (batch 3)."

    @transaction.atomic
    def handle(self, *args, **options):
        self.stdout.write(self.style.MIGRATE_HEADING(
            "Seeding projects batch 3 (DevOps / COMPLETED)..."
        ))

        try:
            cl5 = User.objects.get(email="cl5@ff.dev")
            cl6 = User.objects.get(email="cl6@ff.dev")
            fl11 = User.objects.get(email="fl11@ff.dev")
            fl12 = User.objects.get(email="fl12@ff.dev")
            fl13 = User.objects.get(email="fl13@ff.dev")
            fl15 = User.objects.get(email="fl15@ff.dev")
        except User.DoesNotExist as exc:
            self.stderr.write(self.style.ERROR(
                f"Required user not found: {exc}. "
                "Run seed_clients3 and seed_freelancers3 first."
            ))
            return

        PROJECTS = [
            dict(
                client=cl5, freelancer=fl11,
                title="AWS ECS Production Deployment",
                description=(
                    "CloudNova needed a zero-downtime production deployment on AWS ECS Fargate. "
                    "Included ECR image registry, ALB with HTTPS, RDS Aurora, secrets in Parameter Store, "
                    "CloudWatch dashboards, and auto-scaling policies."
                ),
                budget="11000.00",
                skills=["AWS", "ECS Fargate", "Terraform", "RDS", "CloudWatch", "ALB"],
                days_ago_started=60, days_duration=21,
            ),
            dict(
                client=cl5, freelancer=fl12,
                title="Kubernetes Cluster — Production Setup",
                description=(
                    "CloudNova needed a production-grade Kubernetes cluster: EKS provisioning, "
                    "Helm deployments, Prometheus + Grafana monitoring, cert-manager for TLS, "
                    "and horizontal pod autoscaling. Full runbook delivered."
                ),
                budget="13500.00",
                skills=["Kubernetes", "EKS", "Helm", "Prometheus", "Grafana", "ArgoCD"],
                days_ago_started=45, days_duration=18,
            ),
            dict(
                client=cl6, freelancer=fl13,
                title="AI Chatbot Integration (LangChain + RAG)",
                description=(
                    "SecureBase needed an internal AI chatbot for security policy Q&A. "
                    "Built with LangChain, Pinecone vector store, and Groq LLM. "
                    "Includes document ingestion pipeline and conversation memory."
                ),
                budget="9800.00",
                skills=["Python", "LangChain", "Pinecone", "FastAPI", "Groq", "OpenAI"],
                days_ago_started=30, days_duration=14,
            ),
            dict(
                client=cl6, freelancer=fl15,
                title="Full Web Application Penetration Test",
                description=(
                    "SecureBase commissioned a full black-box pentest of their client portal. "
                    "Included OWASP Top 10 assessment, API security review, and a detailed "
                    "remediation report with CVSS scores and proof-of-concept exploits."
                ),
                budget="7500.00",
                skills=["Penetration Testing", "OWASP", "Burp Suite", "Python", "Linux"],
                days_ago_started=20, days_duration=7,
            ),
            dict(
                client=cl5, freelancer=fl11,
                title="GitHub Actions CI/CD Pipeline",
                description=(
                    "CloudNova needed a full CI/CD pipeline: lint → test → build Docker image → "
                    "push to ECR → deploy to ECS with blue/green strategy, Slack deployment "
                    "notifications, and automatic rollback on health check failure."
                ),
                budget="4500.00",
                skills=["GitHub Actions", "Docker", "AWS ECR", "ECS", "Bash", "Python"],
                days_ago_started=15, days_duration=6,
            ),
        ]

        for p in PROJECTS:
            _build_completed_project(stdout=self.stdout.write, **p)

        self.stdout.write(self.style.SUCCESS(
            "\nDone. 5 COMPLETED DevOps projects seeded with full payment lifecycle."
        ))
