"""
Seed command: 5 backend projects (IN_PROGRESS — contracts, worklogs, messages).
Run: python manage.py seed_projects2

Depends on: seed_clients2, seed_freelancers2
Projects (all IN_PROGRESS with active contracts and work activity):
  1. Payments Microservice (Node.js)    – cl3 + fl7
  2. REST API for Logistics Platform    – cl4 + fl6
  3. Real-time Notifications Service    – cl3 + fl8
  4. Admin Dashboard Backend (Django)   – cl4 + fl6
  5. Webhook Integration System         – cl3 + fl7
"""

import logging
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
from apps.worklogs.models import Deliverable, WorkLog

User = get_user_model()
logger = logging.getLogger(__name__)


def _create_project_with_contract(client, freelancer, title, description, budget,
                                   skills, deadline_days, stdout):
    """Helper: project → accepted bid → contract → payment (ESCROWED)."""
    now = timezone.now()

    if Project.objects.filter(title=title).exists():
        stdout.write(f"  [SKIP] '{title}' already exists.")
        return None, None

    project = Project.objects.create(
        client=client,
        title=title,
        description=description,
        budget=budget,
        deadline=now.date() + timedelta(days=deadline_days),
        status=Project.Status.IN_PROGRESS,
    )
    for skill in skills:
        ProjectSkill.objects.get_or_create(project=project, skill_name=skill)

    bid = Bid.objects.create(
        project=project,
        freelancer=freelancer,
        amount=budget,
        cover_letter=f"I am the best fit for '{title}'. Let's build something great.",
        status=Bid.Status.ACCEPTED,
    )
    contract = Contract.objects.create(
        bid=bid,
        agreed_amount=budget,
        end_date=now + timedelta(days=deadline_days),
        is_active=True,
    )
    payment = Payment.objects.create(
        contract=contract,
        total_amount=budget,
        status=Payment.Status.ESCROWED,
        razorpay_order_id=f"order_seed_{project.id:04d}",
        razorpay_payment_id=f"pay_seed_{project.id:04d}",
    )
    platform_cut = float(budget) * 0.10
    Escrow.objects.create(payment=payment, held_amount=budget)
    PlatformEarning.objects.create(
        payment=payment,
        cut_percentage="10.00",
        cut_amount=str(round(platform_cut, 2)),
    )
    Notification.objects.create(
        recipient=freelancer,
        title="Bid Accepted — Contract Created",
        body=f"Your bid on '{title}' was accepted. Contract is now active.",
        type=Notification.Type.BID_ACCEPTED,
    )
    Notification.objects.create(
        recipient=client,
        title="Escrow Created",
        body=f"INR {budget} has been escrowed for '{title}'.",
        type=Notification.Type.ESCROW_CREATED,
    )
    stdout.write(f"  [OK] Project+Contract: '{title}'")
    return project, contract


def _seed_worklogs(contract, freelancer, client, days_back, stdout):
    """Seed daily worklogs for the last N days."""
    today = date.today()
    descriptions = [
        ("Initial project setup and repo scaffolding",
         "Set up project repo, configured linting, CI pipeline, and base folder structure."),
        ("API schema design and OpenAPI spec",
         "Designed all REST endpoints, wrote OpenAPI 3.0 spec, reviewed with client."),
        ("Core business logic implementation",
         "Implemented main service layer with full unit test coverage (85%+)."),
        ("Database migrations and seed data",
         "Created all migrations, wrote seed fixtures, verified integrity constraints."),
        ("Integration with third-party payment gateway",
         "Integrated Razorpay checkout flow with webhook verification and idempotency."),
        ("Bug fixes from client review",
         "Fixed 12 issues raised in client review session. All tests passing."),
        ("Performance optimisation — DB queries",
         "Reduced N+1 queries using select_related/prefetch. API response time down 60%."),
    ]
    for i, (log_title, log_desc) in enumerate(descriptions[:days_back]):
        log_date = today - timedelta(days=days_back - i)
        if WorkLog.objects.filter(contract=contract, date=log_date).exists():
            continue
        status = WorkLog.Status.APPROVED if i < days_back - 2 else WorkLog.Status.PENDING_APPROVAL
        wl = WorkLog.objects.create(
            contract=contract,
            freelancer=freelancer,
            date=log_date,
            description=log_desc,
            hours_worked="7.50",
            status=status,
            reference_url="https://github.com/freelanceflow/project/pull/" + str(i + 1),
        )
        if status == WorkLog.Status.APPROVED:
            wl.approved_at = timezone.now() - timedelta(days=days_back - i - 1)
            wl.approved_by = client
            wl.save(update_fields=["approved_at", "approved_by"])
        Notification.objects.create(
            recipient=client,
            title="Work Log Submitted",
            body=f"Work log for {log_date} submitted: {log_title}",
            type=Notification.Type.LOG_SUBMITTED,
        )
    stdout.write(f"       \\- {days_back} worklogs seeded")


def _seed_conversation(contract, client, freelancer, messages, stdout):
    """Seed a conversation with messages for a contract."""
    conv, _ = Conversation.objects.get_or_create(contract=contract)
    for sender_key, content in messages:
        sender = client if sender_key == "client" else freelancer
        Message.objects.create(
            conversation=conv,
            sender=sender,
            content=content,
            is_read=True,
        )
    stdout.write(f"       |- {len(messages)} messages seeded")


class Command(BaseCommand):
    help = "Seeds 5 IN_PROGRESS backend projects with contracts, worklogs & messages (batch 2)."

    @transaction.atomic
    def handle(self, *args, **options):
        self.stdout.write(self.style.MIGRATE_HEADING(
            "Seeding projects batch 2 (backend / IN_PROGRESS)..."
        ))

        try:
            cl3 = User.objects.get(email="cl3@ff.dev")
            cl4 = User.objects.get(email="cl4@ff.dev")
            fl6 = User.objects.get(email="fl6@ff.dev")
            fl7 = User.objects.get(email="fl7@ff.dev")
            fl8 = User.objects.get(email="fl8@ff.dev")
        except User.DoesNotExist as exc:
            self.stderr.write(self.style.ERROR(
                f"Required user not found: {exc}. "
                "Run seed_clients2 and seed_freelancers2 first."
            ))
            return

        # Project 1
        proj1, cont1 = _create_project_with_contract(
            client=cl3, freelancer=fl7,
            title="Payments Microservice (Node.js + Razorpay)",
            description=(
                "FinLedger needs a standalone Node.js microservice to handle all payment operations "
                "— Razorpay order creation, webhook verification, idempotent event processing, "
                "and payout scheduling. Must include full integration tests and OpenAPI docs."
            ),
            budget="8500.00",
            skills=["Node.js", "Express", "MongoDB", "Razorpay", "Jest", "Docker"],
            deadline_days=25,
            stdout=self.stdout,
        )
        if cont1:
            _seed_worklogs(cont1, fl7, cl3, 6, self.stdout)
            _seed_conversation(cont1, cl3, fl7, [
                ("client", "Welcome aboard! Please start with the Razorpay order creation flow."),
                ("freelancer", "Thanks! I've reviewed the Razorpay docs. Starting with the order endpoint today."),
                ("freelancer", "Order creation and webhook verification are done. Moving to payout scheduling."),
                ("client", "Great progress! Can you add idempotency for webhooks?"),
                ("freelancer", "Already handled — using razorpay_event_id as unique key in MongoDB."),
                ("client", "Perfect. Please open a PR when payout scheduling is ready."),
            ], self.stdout)

        # Project 2
        proj2, cont2 = _create_project_with_contract(
            client=cl4, freelancer=fl6,
            title="Logistics REST API — Django + PostgreSQL",
            description=(
                "LogiTrack needs a comprehensive REST API for fleet management: vehicle tracking, "
                "route optimisation, driver assignment, and delivery status updates. "
                "Django + DRF + PostgreSQL. Must handle 10K+ daily requests with sub-200ms response times."
            ),
            budget="12000.00",
            skills=["Python", "Django", "DRF", "PostgreSQL", "Celery", "Redis"],
            deadline_days=35,
            stdout=self.stdout,
        )
        if cont2:
            _seed_worklogs(cont2, fl6, cl4, 7, self.stdout)
            _seed_conversation(cont2, cl4, fl6, [
                ("client", "Hi Vikram, glad to have you. Can we jump on a call to discuss the schema?"),
                ("freelancer", "Absolutely — I've drafted the ER diagram already, let me share it."),
                ("client", "Schema looks solid. One addition — add a 'priority' field to deliveries."),
                ("freelancer", "Done in migration 0003. I also added an index for faster priority queries."),
                ("freelancer", "Vehicle tracking endpoint is live. Response time is 140ms under load test."),
                ("client", "140ms is excellent! When can we get the route optimisation API?"),
                ("freelancer", "By Thursday. I'm integrating with Google Maps Distance Matrix API."),
            ], self.stdout)

        # Project 3
        proj3, cont3 = _create_project_with_contract(
            client=cl3, freelancer=fl8,
            title="Real-time Notification Service (FastAPI + WebSockets)",
            description=(
                "FinLedger needs a real-time notification service: WebSocket connections for browser, "
                "FCM for mobile push, and email fallback for offline users. "
                "Built with FastAPI and Redis pub/sub. Must support 5000 concurrent connections."
            ),
            budget="7200.00",
            skills=["FastAPI", "Python", "Redis", "WebSockets", "Firebase FCM", "PostgreSQL"],
            deadline_days=20,
            stdout=self.stdout,
        )
        if cont3:
            _seed_worklogs(cont3, fl8, cl3, 5, self.stdout)
            _seed_conversation(cont3, cl3, fl8, [
                ("client", "Meera, please start with the WebSocket connection manager."),
                ("freelancer", "On it! I'll use Redis pub/sub so we can scale horizontally."),
                ("freelancer", "WebSocket manager done — tested with 5200 concurrent connections via k6."),
                ("client", "Impressive! Now add FCM push for mobile clients."),
                ("freelancer", "FCM integration complete with topic-based subscriptions. Testing fallback now."),
            ], self.stdout)

        # Project 4
        proj4, cont4 = _create_project_with_contract(
            client=cl4, freelancer=fl6,
            title="Admin Dashboard Backend — Django + REST",
            description=(
                "LogiTrack needs an admin backend for internal operations: driver management, "
                "vehicle maintenance scheduling, analytics reports, and CSV export. "
                "Role-based access control with audit logging required."
            ),
            budget="6800.00",
            skills=["Python", "Django", "DRF", "PostgreSQL", "Celery", "JWT"],
            deadline_days=30,
            stdout=self.stdout,
        )
        if cont4:
            _seed_worklogs(cont4, fl6, cl4, 4, self.stdout)
            _seed_conversation(cont4, cl4, fl6, [
                ("client", "Vikram, can we have RBAC with 3 roles: admin, manager, viewer?"),
                ("freelancer", "Yes — I'll use Django's permission system with custom role middleware."),
                ("freelancer", "RBAC implemented. Audit log model also done — tracks all write operations."),
                ("client", "The audit log is exactly what we needed. Can you add CSV export next?"),
            ], self.stdout)

        # Project 5
        proj5, cont5 = _create_project_with_contract(
            client=cl3, freelancer=fl7,
            title="Third-party Webhook Integration Hub",
            description=(
                "FinLedger receives webhooks from 6 providers (Razorpay, Stripe, Zoho, etc.) and "
                "needs a unified integration hub: signature verification, event routing, retry logic "
                "with exponential backoff, and a dead-letter queue for failed events."
            ),
            budget="5500.00",
            skills=["Node.js", "Express", "MongoDB", "Redis", "Bull Queue", "Docker"],
            deadline_days=18,
            stdout=self.stdout,
        )
        if cont5:
            _seed_worklogs(cont5, fl7, cl3, 3, self.stdout)
            _seed_conversation(cont5, cl3, fl7, [
                ("client", "Aditya, start with Razorpay and Stripe webhook handlers."),
                ("freelancer", "Both done with HMAC signature verification. Moving to retry logic with Bull."),
                ("client", "Can the dead-letter queue send Slack alerts?"),
                ("freelancer", "Yes — I'll add a Slack webhook notifier for DLQ events. Done by EOD."),
            ], self.stdout)

        self.stdout.write(self.style.SUCCESS(
            "\nDone. 5 IN_PROGRESS backend projects seeded with contracts, worklogs & messages."
        ))
