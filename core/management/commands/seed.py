"""
Management command to seed the database with realistic test data.

Credentials are printed to stdout at the end for quick reference.

Usage:
    python manage.py seed
    python manage.py seed --no-clear   # keep existing data
"""

import logging
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from apps.bidding.models import Bid, Contract
from apps.messaging.models import Conversation, Message
from apps.projects.models import Project, ProjectSkill
from apps.users.models import ClientProfile, FreelancerProfile

User = get_user_model()
logger = logging.getLogger(__name__)


def _is_elasticsearch_reachable() -> bool:
    """Return True if Elasticsearch is reachable, False otherwise."""
    try:
        from django.conf import settings
        import urllib.request
        import urllib.error

        es_url = getattr(settings, "ELASTICSEARCH_URL", "http://localhost:9200")
        url = f"{es_url.rstrip('/')}/_cluster/health"
        with urllib.request.urlopen(url, timeout=3) as resp:  # noqa: S310
            return resp.status == 200
    except Exception:
        return False


class Command(BaseCommand):
    help = "Seeds the database with realistic test data for development."

    def add_arguments(self, parser):
        parser.add_argument(
            "--no-clear",
            action="store_true",
            default=False,
            help="Skip clearing existing non-superuser data before seeding.",
        )

    @transaction.atomic
    def handle(self, *args, **options):
        # Disconnect all Elasticsearch-bound signals so seeding works without
        # a running ES instance. The search.signals module registers plain Django
        # signals that call ES directly; we disconnect them here and reconnect
        # after the seed completes.
        self._disconnect_es_signals()

        try:
            self._run_seed(options)
        finally:
            self._reconnect_es_signals()

    def _disconnect_es_signals(self):
        """Temporarily disconnect all Elasticsearch-syncing signals."""
        # Step 1: Set the AUTOSYNC flag to False — this is checked by registry.update()
        # before making any ES calls, so even if signals fire they become no-ops.
        from django.conf import settings as dj_settings
        dj_settings.ELASTICSEARCH_DSL_AUTOSYNC = False
        self._original_autosync = True  # remember to restore

        # Step 2: Fully teardown the DSL signal processor (disconnects bound methods)
        try:
            from django.apps import apps as django_apps
            ded_config = django_apps.get_app_config("django_elasticsearch_dsl")
            if ded_config.signal_processor:
                ded_config.signal_processor.teardown()
                self._dsl_processor = ded_config.signal_processor
            self.stdout.write("  [INFO] DSL signal processor torn down.")
        except Exception as exc:
            self.stdout.write(f"  [WARN] DSL teardown failed: {exc}")

        # Step 3: Disconnect the custom search app signals
        try:
            from django.db.models.signals import post_save, post_delete
            from apps.projects.models import Project, ProjectSkill
            from apps.users.models import FreelancerProfile
            from apps.search import signals as search_signals

            post_save.disconnect(search_signals.update_project_document, sender=Project)
            post_delete.disconnect(search_signals.delete_project_document, sender=Project)
            post_save.disconnect(search_signals.update_project_document_on_skill_change, sender=ProjectSkill)
            post_delete.disconnect(search_signals.delete_project_document_on_skill_delete, sender=ProjectSkill)
            post_save.disconnect(search_signals.update_freelancer_document, sender=FreelancerProfile)
            post_delete.disconnect(search_signals.delete_freelancer_document, sender=FreelancerProfile)
            self.stdout.write("  [INFO] Custom search app signals disconnected.")
        except Exception as exc:
            self.stdout.write(f"  [WARN] Could not disconnect custom ES signals: {exc}")

    def _reconnect_es_signals(self):
        """
        Reconnect Elasticsearch-syncing signals after seeding — but only if
        Elasticsearch is actually reachable.  If it is not, signals remain
        detached (they are safe — see _safe_es_update in search/signals.py).
        The operator is instructed to run `search_index --rebuild` once ES is up.
        """
        if not _is_elasticsearch_reachable():
            self.stdout.write(
                self.style.WARNING(
                    "\n  [WARN] Elasticsearch is not reachable — ES signals stay "
                    "detached for this session.\n"
                    "  Once ES is healthy run:\n"
                    "    docker compose exec web python manage.py search_index --rebuild -f\n"
                )
            )
            return

        # ES is up — restore AUTOSYNC and reconnect signals.
        from django.conf import settings as dj_settings
        dj_settings.ELASTICSEARCH_DSL_AUTOSYNC = True

        try:
            dsl_processor = getattr(self, "_dsl_processor", None)
            if dsl_processor:
                dsl_processor.setup()
        except Exception:
            pass

        try:
            from django.db.models.signals import post_save, post_delete
            from apps.projects.models import Project, ProjectSkill
            from apps.users.models import FreelancerProfile
            from apps.search import signals as search_signals

            post_save.connect(search_signals.update_project_document, sender=Project)
            post_delete.connect(search_signals.delete_project_document, sender=Project)
            post_save.connect(search_signals.update_project_document_on_skill_change, sender=ProjectSkill)
            post_delete.connect(search_signals.delete_project_document_on_skill_delete, sender=ProjectSkill)
            post_save.connect(search_signals.update_freelancer_document, sender=FreelancerProfile)
            post_delete.connect(search_signals.delete_freelancer_document, sender=FreelancerProfile)
            self.stdout.write("  [INFO] ES signals reconnected.")
        except Exception as exc:
            self.stdout.write(self.style.WARNING(f"  [WARN] Could not reconnect ES signals: {exc}"))

    @transaction.atomic
    def _run_seed(self, options):
        self.stdout.write(self.style.MIGRATE_HEADING("Starting database seed..."))

        if not options["no_clear"]:
            self._clear_data()

        # ── Users ─────────────────────────────────────────────────────────────
        client1, _ = self._create_client(
            email="client@example.com",
            password="password123",
            first_name="Acme",
            last_name="Corp",
            company_name="Acme Corporation",
        )
        client2, _ = self._create_client(
            email="client2@example.com",
            password="password123",
            first_name="TechVista",
            last_name="Inc",
            company_name="TechVista Inc.",
        )

        freelancer1, _ = self._create_freelancer(
            email="freelancer@example.com",
            password="password123",
            first_name="John",
            last_name="Doe",
            bio="Experienced full-stack developer with 5+ years building SaaS products.",
            skills=["Python", "Django", "React", "PostgreSQL", "Docker"],
            hourly_rate=50.00,
            tier=FreelancerProfile.SubscriptionTier.PRO,
            total_earned=12500.00,
            average_rating=4.80,
            total_reviews=23,
        )
        freelancer2, _ = self._create_freelancer(
            email="freelancer2@example.com",
            password="password123",
            first_name="Jane",
            last_name="Smith",
            bio="UI/UX designer and frontend specialist. Pixel-perfect interfaces.",
            skills=["Figma", "React", "Tailwind CSS", "JavaScript", "HTML/CSS"],
            hourly_rate=65.00,
            tier=FreelancerProfile.SubscriptionTier.PRO,
            total_earned=9800.00,
            average_rating=4.95,
            total_reviews=18,
        )

        # ── Projects ──────────────────────────────────────────────────────────
        projects_data = [
            {
                "client": client1,
                "title": "E-commerce Website – Full Stack",
                "description": (
                    "We need an experienced full-stack developer to build a modern "
                    "e-commerce platform with product listings, cart, Razorpay checkout, "
                    "and an admin dashboard."
                ),
                "budget": 5000.00,
                "status": Project.Status.IN_PROGRESS,
                "days_offset": 30,
                "skills": ["React", "Django", "PostgreSQL", "Razorpay"],
            },
            {
                "client": client1,
                "title": "Django REST API for Mobile App",
                "description": (
                    "Looking for a Django expert to build a RESTful API with JWT auth, "
                    "push notifications, and a robust test suite for our iOS app."
                ),
                "budget": 3500.00,
                "status": Project.Status.OPEN,
                "days_offset": 45,
                "skills": ["Python", "Django REST Framework", "Docker", "PostgreSQL"],
            },
            {
                "client": client1,
                "title": "Fix Responsive Bugs on Landing Page",
                "description": (
                    "Small task: fix mobile-responsive issues on our marketing landing "
                    "page. Pixel-perfect match to Figma designs required."
                ),
                "budget": 250.00,
                "status": Project.Status.COMPLETED,
                "days_offset": -10,
                "skills": ["HTML", "CSS", "Tailwind CSS"],
            },
            {
                "client": client2,
                "title": "SaaS Dashboard UI Redesign",
                "description": (
                    "Our existing dashboard looks dated. We need a modern redesign "
                    "with dark-mode support, data visualisations, and a component library."
                ),
                "budget": 4200.00,
                "status": Project.Status.OPEN,
                "days_offset": 60,
                "skills": ["Figma", "React", "Tailwind CSS", "Chart.js"],
            },
            {
                "client": client2,
                "title": "CI/CD Pipeline Setup on AWS",
                "description": (
                    "Set up GitHub Actions CI/CD that deploys to ECS Fargate with "
                    "automated testing, Docker image builds, and Slack notifications."
                ),
                "budget": 1800.00,
                "status": Project.Status.OPEN,
                "days_offset": 21,
                "skills": ["AWS", "Docker", "GitHub Actions", "Terraform"],
            },
        ]

        created_projects = []
        for p_data in projects_data:
            project = Project.objects.create(
                client=p_data["client"],
                title=p_data["title"],
                description=p_data["description"],
                budget=p_data["budget"],
                deadline=timezone.now().date() + timedelta(days=p_data["days_offset"]),
                status=p_data["status"],
            )
            for skill in p_data["skills"]:
                ProjectSkill.objects.get_or_create(project=project, skill_name=skill)
            created_projects.append(project)
            self.stdout.write(self.style.SUCCESS(f"  [OK] Project: {project.title}"))

        # ── Bids ──────────────────────────────────────────────────────────────
        in_progress_project = created_projects[0]

        # freelancer1 bid – accepted
        bid_accepted = Bid.objects.create(
            project=in_progress_project,
            freelancer=freelancer1,
            amount=4500.00,
            cover_letter=(
                "Hi! I have built three similar e-commerce platforms in the past two "
                "years. I can deliver a production-ready solution within 4 weeks, "
                "including full test coverage and deployment documentation."
            ),
            status=Bid.Status.ACCEPTED,
        )

        # freelancer2 bid – rejected
        Bid.objects.create(
            project=in_progress_project,
            freelancer=freelancer2,
            amount=4800.00,
            cover_letter=(
                "I specialise in beautiful frontends. Happy to collaborate on the "
                "React side and deliver pixel-perfect components."
            ),
            status=Bid.Status.REJECTED,
        )

        # freelancer1 bids on client2 dashboard – pending
        Bid.objects.create(
            project=created_projects[3],
            freelancer=freelancer1,
            amount=3900.00,
            cover_letter=(
                "I have built multiple SaaS dashboards with Recharts and Tailwind. "
                "I can deliver a full component library and dark-mode toggle."
            ),
            status=Bid.Status.PENDING,
        )

        # freelancer2 bids on client2 dashboard – pending
        Bid.objects.create(
            project=created_projects[3],
            freelancer=freelancer2,
            amount=4100.00,
            cover_letter=(
                "UI/UX is my speciality. I'll start with a Figma prototype for approval "
                "then implement in React with Storybook documentation."
            ),
            status=Bid.Status.PENDING,
        )

        self.stdout.write(self.style.SUCCESS("  [OK] Created 4 bids"))

        # ── Contract ──────────────────────────────────────────────────────────
        contract = Contract.objects.create(
            bid=bid_accepted,
            agreed_amount=4500.00,
            end_date=timezone.now() + timedelta(days=30),
            is_active=True,
        )
        self.stdout.write(
            self.style.SUCCESS(f"  [OK] Contract created for: {in_progress_project.title}")
        )

        # ── Conversation & Messages ────────────────────────────────────────────
        conversation = Conversation.objects.create(contract=contract)
        chat_messages = [
            (client1, "Hi John, great to have you on board! When can you start?"),
            (
                freelancer1,
                "Hello! Thanks for accepting my bid. I can start tomorrow. "
                "Could you share the Figma designs and the existing codebase access?",
            ),
            (
                client1,
                "Perfect. I've sent the Figma invite to your email. The repo is on "
                "GitHub – I'll add you now. Please push a project setup PR by Friday.",
            ),
            (
                freelancer1,
                "Got the Figma invite, thank you. I'll review the designs today and "
                "have the initial project scaffold up by Thursday. I'll keep you posted!",
            ),
            (
                client1,
                "Sounds great. Feel free to ask if you need anything from our side.",
            ),
        ]
        for sender, content in chat_messages:
            Message.objects.create(
                conversation=conversation,
                sender=sender,
                content=content,
                is_read=True,
            )
        self.stdout.write(
            self.style.SUCCESS(f"  [OK] Conversation seeded with {len(chat_messages)} messages")
        )

        # ── Summary ───────────────────────────────────────────────────────────
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Database seeded successfully!"))
        self.stdout.write(self.style.MIGRATE_HEADING("--- Login Credentials ---"))
        self.stdout.write(
            self.style.SUCCESS("  CLIENT 1     :  client@example.com      /  password123")
        )
        self.stdout.write(
            self.style.SUCCESS("  CLIENT 2     :  client2@example.com     /  password123")
        )
        self.stdout.write(
            self.style.SUCCESS("  FREELANCER 1 :  freelancer@example.com  /  password123")
        )
        self.stdout.write(
            self.style.SUCCESS("  FREELANCER 2 :  freelancer2@example.com /  password123")
        )
        self.stdout.write(self.style.MIGRATE_HEADING("------------------------"))

    # ── Private helpers ───────────────────────────────────────────────────────

    def _clear_data(self):
        self.stdout.write("  Clearing existing data...")
        Message.objects.all().delete()
        Conversation.objects.all().delete()
        Contract.objects.all().delete()
        Bid.objects.all().delete()
        Project.objects.all().delete()
        # Delete profiles explicitly before users to avoid constraint conflicts
        ClientProfile.objects.all().delete()
        FreelancerProfile.objects.all().delete()
        User.objects.filter(is_superuser=False).delete()
        self.stdout.write(self.style.SUCCESS("  [OK] Cleared."))

    def _create_client(self, *, email, password, first_name, last_name, company_name):
        user = User(
            email=email,
            first_name=first_name,
            last_name=last_name,
            role=User.Roles.CLIENT,
        )
        user.set_password(password)
        user.save()
        # The post_save signal auto-creates the ClientProfile; use get_or_create
        # and then update with richer seed data.
        profile, _ = ClientProfile.objects.get_or_create(user=user)
        profile.company_name = company_name
        profile.total_spent = 5000.00
        profile.save(update_fields=["company_name", "total_spent"])
        self.stdout.write(self.style.SUCCESS(f"  [OK] Client: {email}"))
        return user, profile

    def _create_freelancer(
        self,
        *,
        email,
        password,
        first_name,
        last_name,
        bio,
        skills,
        hourly_rate,
        tier,
        total_earned,
        average_rating,
        total_reviews,
    ):
        user = User(
            email=email,
            first_name=first_name,
            last_name=last_name,
            role=User.Roles.FREELANCER,
        )
        user.set_password(password)
        user.save()
        # The post_save signal auto-creates the FreelancerProfile; use get_or_create
        # and then update with richer seed data.
        profile, _ = FreelancerProfile.objects.get_or_create(user=user)
        profile.bio = bio
        profile.skills = skills
        profile.hourly_rate = hourly_rate
        profile.subscription_tier = tier
        profile.total_earned = total_earned
        profile.average_rating = average_rating
        profile.total_reviews = total_reviews
        profile.is_available = True
        profile.save(
            update_fields=[
                "bio", "skills", "hourly_rate", "subscription_tier",
                "total_earned", "average_rating", "total_reviews", "is_available",
            ]
        )
        self.stdout.write(self.style.SUCCESS(f"  [OK] Freelancer: {email}"))
        return user, profile
