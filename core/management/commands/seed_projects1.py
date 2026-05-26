"""
Seed command: 5 frontend projects (OPEN status, multiple bids, conversations).
Run: python manage.py seed_projects1

Depends on: seed_clients1, seed_freelancers1
Projects:
  1. React Dashboard Redesign           – cl1, bids from fl1+fl2+fl5  (OPEN)
  2. Next.js Marketing Site             – cl1, bids from fl1+fl5       (OPEN)
  3. E-commerce Storefront (React)      – cl2, bids from fl2+fl4       (OPEN)
  4. Mobile App UI (React Native)       – cl2, bids from fl4+fl5       (OPEN)
  5. Component Library + Storybook      – cl1, bids from fl1+fl2+fl5   (OPEN)
"""

import logging
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from apps.bidding.models import Bid
from apps.messaging.models import Conversation, Message
from apps.notifications.models import Notification
from apps.projects.models import Project, ProjectSkill

User = get_user_model()
logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Seeds 5 OPEN frontend projects with bids and conversations (batch 1)."

    @transaction.atomic
    def handle(self, *args, **options):
        self.stdout.write(self.style.MIGRATE_HEADING("Seeding projects batch 1 (frontend / OPEN)..."))

        # ── Resolve users (must have run seed_clients1 & seed_freelancers1 first)
        try:
            cl1 = User.objects.get(email="cl1@ff.dev")
            cl2 = User.objects.get(email="cl2@ff.dev")
            fl1 = User.objects.get(email="fl1@ff.dev")
            fl2 = User.objects.get(email="fl2@ff.dev")
            fl4 = User.objects.get(email="fl4@ff.dev")
            fl5 = User.objects.get(email="fl5@ff.dev")
        except User.DoesNotExist as exc:
            self.stderr.write(self.style.ERROR(
                f"Required user not found: {exc}. "
                "Run seed_clients1, seed_freelancers1 first."
            ))
            return

        now = timezone.now()

        PROJECTS = [
            {
                "client": cl1,
                "title": "React Dashboard Redesign",
                "description": (
                    "PixelCraft Studio needs a full redesign of our internal analytics dashboard. "
                    "We want dark-mode support, chart components (Recharts), and a polished "
                    "component library built in React + TypeScript. Figma designs will be provided."
                ),
                "budget": "6500.00",
                "deadline_days": 45,
                "skills": ["React", "TypeScript", "Tailwind CSS", "Recharts", "Figma"],
                "bids": [
                    {
                        "freelancer": fl1,
                        "amount": "6000.00",
                        "cover_letter": (
                            "I have built 5 similar dashboards with Recharts and dark mode. "
                            "I can start immediately and deliver a Figma-to-code pixel-perfect result. "
                            "Happy to share relevant portfolio links."
                        ),
                        "status": Bid.Status.PENDING,
                    },
                    {
                        "freelancer": fl2,
                        "amount": "6200.00",
                        "cover_letter": (
                            "Vue.js is my primary stack but I'm very proficient in React too. "
                            "I bring strong design sensibility and will match your Figma exactly."
                        ),
                        "status": Bid.Status.PENDING,
                    },
                    {
                        "freelancer": fl5,
                        "amount": "6400.00",
                        "cover_letter": (
                            "UI/UX is my core speciality. I'll first review your Figma, suggest "
                            "improvements, then implement with full Storybook documentation."
                        ),
                        "status": Bid.Status.PENDING,
                    },
                ],
            },
            {
                "client": cl1,
                "title": "Next.js Marketing Site with CMS",
                "description": (
                    "We need a high-performance marketing site built with Next.js 14 (App Router), "
                    "Sanity CMS integration, and deployed to Vercel. Must score 95+ on Lighthouse. "
                    "Includes blog, landing pages, and a contact form with email notifications."
                ),
                "budget": "4200.00",
                "deadline_days": 30,
                "skills": ["Next.js", "TypeScript", "Sanity CMS", "Vercel", "Tailwind CSS"],
                "bids": [
                    {
                        "freelancer": fl1,
                        "amount": "3900.00",
                        "cover_letter": (
                            "Next.js App Router is my daily driver. I have a Sanity CMS starter "
                            "kit ready. I guarantee 95+ Lighthouse scores with ISR and image optimisation."
                        ),
                        "status": Bid.Status.PENDING,
                    },
                    {
                        "freelancer": fl5,
                        "amount": "4100.00",
                        "cover_letter": (
                            "I specialise in marketing sites that convert. I'll ensure the design "
                            "is pixel-perfect and the CMS schema is clean for non-technical editors."
                        ),
                        "status": Bid.Status.PENDING,
                    },
                ],
            },
            {
                "client": cl2,
                "title": "E-commerce Storefront in React",
                "description": (
                    "ShopSprint needs a modern storefront built in React with product listing, "
                    "filters, cart with local persistence, and Razorpay checkout. "
                    "Backend API is already built — you'll be consuming REST endpoints only."
                ),
                "budget": "5500.00",
                "deadline_days": 40,
                "skills": ["React", "Redux Toolkit", "Razorpay", "Tailwind CSS", "Axios"],
                "bids": [
                    {
                        "freelancer": fl2,
                        "amount": "5200.00",
                        "cover_letter": (
                            "I have built 3 e-commerce frontends integrating Razorpay. "
                            "My cart implementations are snappy and handle edge cases like "
                            "stock depletion and payment failures gracefully."
                        ),
                        "status": Bid.Status.PENDING,
                    },
                    {
                        "freelancer": fl4,
                        "amount": "5400.00",
                        "cover_letter": (
                            "While my speciality is React Native, I have extensive React web "
                            "experience and can make this storefront feel native-app-smooth."
                        ),
                        "status": Bid.Status.PENDING,
                    },
                ],
            },
            {
                "client": cl2,
                "title": "iOS & Android Shopping App (React Native)",
                "description": (
                    "ShopSprint needs a cross-platform mobile app for iOS and Android. "
                    "Features: product browsing, wishlist, cart, push notifications, "
                    "Razorpay payment, and order tracking. Design system provided in Figma."
                ),
                "budget": "9000.00",
                "deadline_days": 60,
                "skills": ["React Native", "Expo", "Redux", "Firebase", "Razorpay", "Figma"],
                "bids": [
                    {
                        "freelancer": fl4,
                        "amount": "8500.00",
                        "cover_letter": (
                            "Mobile is my full-time speciality. I have shipped 8 production apps "
                            "with Expo, including Razorpay integration and Firebase push notifications. "
                            "I can match your Figma exactly on both platforms."
                        ),
                        "status": Bid.Status.PENDING,
                    },
                    {
                        "freelancer": fl5,
                        "amount": "8800.00",
                        "cover_letter": (
                            "I'll handle the design system implementation end-to-end — "
                            "from Figma component audit to production-ready React Native screens."
                        ),
                        "status": Bid.Status.PENDING,
                    },
                ],
            },
            {
                "client": cl1,
                "title": "Design System & Component Library",
                "description": (
                    "PixelCraft Studio wants a shared component library (React + Storybook) "
                    "that all our products will consume via npm. Must include tokens for color, "
                    "spacing, typography, and 30+ components with full accessibility compliance."
                ),
                "budget": "7800.00",
                "deadline_days": 55,
                "skills": ["React", "Storybook", "TypeScript", "Radix UI", "A11y", "Figma"],
                "bids": [
                    {
                        "freelancer": fl1,
                        "amount": "7400.00",
                        "cover_letter": (
                            "I built a design system from scratch for a B2B SaaS — same "
                            "Storybook + Radix approach. I take accessibility seriously and "
                            "write full ARIA compliance documentation."
                        ),
                        "status": Bid.Status.PENDING,
                    },
                    {
                        "freelancer": fl2,
                        "amount": "7500.00",
                        "cover_letter": (
                            "Design systems are my favourite type of project. I'll audit your "
                            "Figma tokens first, then align the code perfectly."
                        ),
                        "status": Bid.Status.PENDING,
                    },
                    {
                        "freelancer": fl5,
                        "amount": "7700.00",
                        "cover_letter": (
                            "As a designer-developer, I ensure tokens are consistent "
                            "between Figma and code. I'll also write the contribution guide."
                        ),
                        "status": Bid.Status.PENDING,
                    },
                ],
            },
        ]

        for p_data in PROJECTS:
            if Project.objects.filter(title=p_data["title"]).exists():
                self.stdout.write(f"  [SKIP] Project '{p_data['title']}' already exists.")
                continue

            project = Project.objects.create(
                client=p_data["client"],
                title=p_data["title"],
                description=p_data["description"],
                budget=p_data["budget"],
                deadline=now.date() + timedelta(days=p_data["deadline_days"]),
                status=Project.Status.OPEN,
            )
            for skill in p_data["skills"]:
                ProjectSkill.objects.get_or_create(project=project, skill_name=skill)

            for bid_data in p_data["bids"]:
                bid = Bid.objects.create(
                    project=project,
                    freelancer=bid_data["freelancer"],
                    amount=bid_data["amount"],
                    cover_letter=bid_data["cover_letter"],
                    status=bid_data["status"],
                )
                # Notify client about each bid
                Notification.objects.create(
                    recipient=p_data["client"],
                    title="New Bid Received",
                    body=(
                        f"{bid_data['freelancer'].get_full_name()} submitted a bid of "
                        f"INR {bid_data['amount']} on '{project.title}'."
                    ),
                    type=Notification.Type.BID_SUBMITTED,
                )

            self.stdout.write(self.style.SUCCESS(
                f"  [OK] '{project.title}' — {len(p_data['bids'])} bids"
            ))

        self.stdout.write(self.style.SUCCESS("\nDone. 5 OPEN frontend projects seeded."))
