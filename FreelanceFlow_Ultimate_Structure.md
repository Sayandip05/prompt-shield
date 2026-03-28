# FreelanceFlow — Ultimate Complete Folder Structure
## Modular Monolith · Layered Architecture · Full Stack · Production Ready

```
freelanceflow/                                     ← Git repository root
│
│  ══════════════════════════════════════════════
│  BACKEND — Django (Python)
│  ══════════════════════════════════════════════
│
├── config/                                        ← Django project config (NOT an app)
│   ├── settings/
│   │   ├── __init__.py
│   │   ├── base.py                                ← All shared settings
│   │   ├── local.py                               ← DEBUG=True, console email, no S3
│   │   └── production.py                          ← S3, Sentry, HTTPS, HSTS
│   ├── urls.py                                    ← Root URL dispatcher
│   ├── celery.py                                  ← Celery app + Beat schedule
│   ├── asgi.py                                    ← ASGI entry (Daphne + Channels)
│   └── wsgi.py                                    ← WSGI entry (Gunicorn)
│
├── apps/                                          ← All Django domain apps
│   ├── __init__.py
│   │
│   ├── users/                                     ← Auth, profiles, roles
│   │   ├── migrations/
│   │   │   └── __init__.py
│   │   ├── tests/
│   │   │   ├── __init__.py
│   │   │   ├── test_models.py
│   │   │   ├── test_services.py
│   │   │   ├── test_selectors.py
│   │   │   └── test_views.py
│   │   ├── __init__.py
│   │   ├── apps.py                                ← name = 'apps.users'
│   │   ├── models.py                              ← User, FreelancerProfile, ClientProfile
│   │   ├── serializers.py                         ← Register, Login, Profile serializers
│   │   ├── services.py                            ← create_user(), update_profile()
│   │   ├── selectors.py                           ← get_user_by_id(), get_profile()
│   │   ├── views.py                               ← Thin HTTP handlers only
│   │   ├── urls.py
│   │   ├── permissions.py                         ← IsFreelancer, IsClient, IsOwner
│   │   ├── signals.py                             ← auto-create profile on register
│   │   ├── tasks.py                               ← send_welcome_email_task()
│   │   └── admin.py                               ← UserAdmin, ProfileAdmin
│   │
│   ├── projects/                                  ← Client posts, Freelancer browses
│   │   ├── migrations/
│   │   ├── tests/
│   │   │   ├── test_models.py
│   │   │   ├── test_services.py
│   │   │   └── test_views.py
│   │   ├── __init__.py
│   │   ├── apps.py                                ← name = 'apps.projects'
│   │   ├── models.py                              ← Project, ProjectSkill
│   │   ├── serializers.py                         ← ProjectSerializer, CreateProjectSerializer
│   │   ├── services.py                            ← create_project(), close_project()
│   │   ├── selectors.py                           ← get_open_projects(), get_client_projects()
│   │   ├── views.py
│   │   ├── urls.py
│   │   ├── permissions.py                         ← IsProjectOwner
│   │   └── admin.py
│   │
│   ├── bidding/                                   ← Bid submit, accept (race condition solved)
│   │   ├── migrations/
│   │   ├── tests/
│   │   │   ├── test_services.py                   ← test select_for_update race condition
│   │   │   └── test_views.py
│   │   ├── __init__.py
│   │   ├── apps.py                                ← name = 'apps.bidding'
│   │   ├── models.py                              ← Bid, Contract
│   │   ├── serializers.py
│   │   ├── services.py                            ← submit_bid(), accept_bid() ← select_for_update
│   │   ├── selectors.py                           ← get_bids_for_project(), get_freelancer_bids()
│   │   ├── views.py
│   │   ├── urls.py
│   │   ├── permissions.py                         ← IsBidOwner, IsProjectClient
│   │   └── admin.py
│   │
│   ├── payments/                                  ← Escrow in → platform cut → release
│   │   ├── migrations/
│   │   ├── tests/
│   │   │   ├── test_services.py                   ← test atomic payment flow
│   │   │   ├── test_views.py
│   │   │   └── test_webhook.py                    ← test HMAC Stripe verification
│   │   ├── __init__.py
│   │   ├── apps.py                                ← name = 'apps.payments'
│   │   ├── models.py                              ← Payment, Escrow, PlatformEarning, PaymentEvent
│   │   ├── serializers.py
│   │   ├── services.py                            ← create_escrow(), release_payment(), calculate_cut()
│   │   ├── selectors.py                           ← get_payment_for_contract(), get_platform_revenue()
│   │   ├── views.py
│   │   ├── urls.py
│   │   ├── tasks.py                               ← process_stripe_webhook_task()
│   │   ├── permissions.py                         ← IsPaymentClient
│   │   └── admin.py                               ← PaymentAdmin, PlatformEarningAdmin
│   │
│   ├── worklogs/                                  ← THE UNIQUE FEATURE
│   │   ├── migrations/
│   │   ├── tests/
│   │   │   ├── test_models.py
│   │   │   ├── test_services.py
│   │   │   ├── test_ai_service.py                 ← mock Anthropic API
│   │   │   ├── test_pdf_service.py
│   │   │   └── test_views.py
│   │   ├── templates/
│   │   │   └── worklogs/
│   │   │       ├── weekly_report.html             ← WeasyPrint PDF: weekly report
│   │   │       └── delivery_proof.html            ← WeasyPrint PDF: final proof
│   │   ├── __init__.py
│   │   ├── apps.py                                ← name = 'apps.worklogs'
│   │   ├── models.py                              ← WorkLog, WeeklyReport, DeliveryProof
│   │   ├── serializers.py
│   │   ├── services.py                            ← create_log(), trigger_report(), generate_proof()
│   │   ├── selectors.py                           ← get_logs_for_week(), get_reports_for_contract()
│   │   ├── views.py
│   │   ├── urls.py
│   │   ├── tasks.py                               ← generate_ai_report_task(), generate_pdf_task()
│   │   ├── ai_service.py                          ← build_prompt() + call Anthropic API (isolated)
│   │   ├── pdf_service.py                         ← WeasyPrint → PDF bytes → upload to S3
│   │   ├── permissions.py                         ← IsContractFreelancer, IsContractClient
│   │   └── admin.py
│   │
│   ├── messaging/                                 ← Real-time chat per contract
│   │   ├── migrations/
│   │   ├── tests/
│   │   │   ├── test_consumers.py                  ← WebSocket consumer tests
│   │   │   └── test_views.py
│   │   ├── __init__.py
│   │   ├── apps.py                                ← name = 'apps.messaging'
│   │   ├── models.py                              ← Conversation, Message
│   │   ├── serializers.py
│   │   ├── services.py                            ← create_conversation(), send_message()
│   │   ├── selectors.py                           ← get_messages(), get_conversations()
│   │   ├── consumers.py                           ← AsyncWebsocketConsumer
│   │   ├── routing.py                             ← WebSocket URL patterns
│   │   ├── views.py                               ← REST: history, list
│   │   ├── urls.py
│   │   └── admin.py
│   │
│   ├── notifications/                             ← In-app + email alerts
│   │   ├── migrations/
│   │   ├── tests/
│   │   │   └── test_services.py
│   │   ├── __init__.py
│   │   ├── apps.py                                ← name = 'apps.notifications'
│   │   ├── models.py                              ← Notification
│   │   ├── serializers.py
│   │   ├── services.py                            ← create_notification(), mark_read()
│   │   ├── selectors.py                           ← get_unread_notifications()
│   │   ├── views.py
│   │   ├── urls.py
│   │   ├── tasks.py                               ← send_email_notification_task()
│   │   └── admin.py
│   │
│   └── search/                                    ← Elasticsearch project search
│       ├── tests/
│       │   └── test_views.py
│       ├── __init__.py
│       ├── apps.py                                ← name = 'apps.search'
│       ├── documents.py                           ← ProjectDocument (django-elasticsearch-dsl)
│       ├── serializers.py
│       ├── views.py                               ← SearchView
│       ├── urls.py
│       └── signals.py                             ← sync Project → ES via on_commit()
│
├── core/                                          ← Shared utilities, zero business logic
│   ├── __init__.py
│   ├── permissions.py                             ← IsOwnerOrAdmin, base classes
│   ├── pagination.py                              ← StandardResultsPagination (20/page)
│   ├── exceptions.py                              ← custom DRF exception handler
│   ├── throttles.py                               ← TieredRateThrottle (Free/Pro)
│   └── utils.py                                   ← calculate_platform_cut(), format_currency()
│
│  ══════════════════════════════════════════════
│  FRONTEND — React (JavaScript)
│  ══════════════════════════════════════════════
│
├── frontend/
│   ├── public/
│   │   ├── index.html
│   │   └── favicon.ico
│   │
│   ├── src/
│   │   │
│   │   ├── api/                                   ← All backend API calls (axios)
│   │   │   ├── auth.js                            ← login(), register(), refreshToken()
│   │   │   ├── projects.js                        ← getProjects(), createProject()
│   │   │   ├── bids.js                            ← submitBid(), acceptBid()
│   │   │   ├── payments.js                        ← createEscrow(), releasePayment()
│   │   │   ├── worklogs.js                        ← submitLog(), getReports()
│   │   │   ├── messages.js                        ← getMessages()
│   │   │   ├── notifications.js                   ← getNotifications(), markRead()
│   │   │   └── search.js                          ← searchProjects()
│   │   │
│   │   ├── components/                            ← Reusable UI components
│   │   │   ├── common/
│   │   │   │   ├── Navbar.jsx
│   │   │   │   ├── Sidebar.jsx
│   │   │   │   ├── NotificationBell.jsx           ← real-time unread count
│   │   │   │   ├── LoadingSpinner.jsx
│   │   │   │   ├── ErrorMessage.jsx
│   │   │   │   └── ConfirmModal.jsx
│   │   │   │
│   │   │   ├── projects/
│   │   │   │   ├── ProjectCard.jsx
│   │   │   │   ├── ProjectForm.jsx                ← create/edit project
│   │   │   │   └── ProjectStatusBadge.jsx
│   │   │   │
│   │   │   ├── bids/
│   │   │   │   ├── BidCard.jsx
│   │   │   │   └── BidForm.jsx
│   │   │   │
│   │   │   ├── worklogs/
│   │   │   │   ├── WorkLogForm.jsx                ← THE CORE: daily log submit
│   │   │   │   ├── WorkLogCard.jsx
│   │   │   │   ├── WeeklyReportCard.jsx           ← shows AI report + PDF download
│   │   │   │   └── DeliveryProofBanner.jsx        ← final proof download
│   │   │   │
│   │   │   ├── payments/
│   │   │   │   ├── EscrowStatus.jsx
│   │   │   │   └── PaymentHistory.jsx
│   │   │   │
│   │   │   └── chat/
│   │   │       ├── ChatWindow.jsx                 ← WebSocket consumer
│   │   │       ├── MessageBubble.jsx
│   │   │       └── ChatInput.jsx
│   │   │
│   │   ├── pages/                                 ← Route-level page components
│   │   │   │
│   │   │   ├── auth/
│   │   │   │   ├── LoginPage.jsx
│   │   │   │   ├── RegisterPage.jsx               ← choose CLIENT or FREELANCER
│   │   │   │   └── GoogleCallbackPage.jsx
│   │   │   │
│   │   │   ├── client/                            ← CLIENT DASHBOARD
│   │   │   │   ├── ClientOverviewPage.jsx         ← /client/overview/
│   │   │   │   ├── ClientProjectsPage.jsx         ← /client/projects/
│   │   │   │   ├── ClientProjectDetailPage.jsx    ← /client/projects/{id}/
│   │   │   │   ├── ClientPaymentsPage.jsx         ← /client/payments/
│   │   │   │   └── ClientMessagesPage.jsx         ← /client/messages/
│   │   │   │
│   │   │   ├── freelancer/                        ← FREELANCER DASHBOARD
│   │   │   │   ├── FreelancerOverviewPage.jsx     ← /freelancer/overview/
│   │   │   │   ├── FreelancerBrowsePage.jsx       ← /freelancer/browse/ (search)
│   │   │   │   ├── FreelancerBidsPage.jsx         ← /freelancer/bids/
│   │   │   │   ├── FreelancerContractsPage.jsx    ← /freelancer/contracts/
│   │   │   │   ├── FreelancerContractDetailPage.jsx ← submit log, view reports, PDF
│   │   │   │   ├── FreelancerWorklogsPage.jsx     ← /freelancer/worklogs/
│   │   │   │   ├── FreelancerEarningsPage.jsx     ← /freelancer/earnings/
│   │   │   │   └── FreelancerMessagesPage.jsx     ← /freelancer/messages/
│   │   │   │
│   │   │   └── shared/
│   │   │       ├── NotFoundPage.jsx
│   │   │       └── UnauthorizedPage.jsx
│   │   │
│   │   ├── hooks/                                 ← Custom React hooks
│   │   │   ├── useAuth.js                         ← JWT token management
│   │   │   ├── useWebSocket.js                    ← WebSocket connection + reconnect
│   │   │   ├── useNotifications.js                ← real-time notification polling
│   │   │   └── usePagination.js
│   │   │
│   │   ├── context/
│   │   │   ├── AuthContext.jsx                    ← user, role, token
│   │   │   └── NotificationContext.jsx
│   │   │
│   │   ├── utils/
│   │   │   ├── axiosInstance.js                   ← base axios + JWT interceptor
│   │   │   ├── formatCurrency.js
│   │   │   └── formatDate.js
│   │   │
│   │   ├── routes/
│   │   │   ├── AppRouter.jsx                      ← all routes defined here
│   │   │   ├── ClientRoute.jsx                    ← guard: role must be CLIENT
│   │   │   └── FreelancerRoute.jsx                ← guard: role must be FREELANCER
│   │   │
│   │   ├── App.jsx
│   │   └── main.jsx
│   │
│   ├── .env                                       ← VITE_API_BASE_URL=http://localhost:8000
│   ├── .env.production                            ← VITE_API_BASE_URL=https://yourdomain.com
│   ├── vite.config.js
│   ├── package.json
│   └── index.html
│
│  ══════════════════════════════════════════════
│  INFRASTRUCTURE & DEPLOYMENT
│  ══════════════════════════════════════════════
│
├── deployment/
│   ├── nginx/
│   │   └── freelanceflow.conf                     ← HTTP + WebSocket routing to Gunicorn/Daphne
│   └── scripts/
│       ├── deploy.sh                              ← git pull + docker-compose up -d
│       └── backup_db.sh                           ← pg_dump → S3 daily backup
│
├── .github/
│   └── workflows/
│       ├── ci.yml                                 ← pytest + flake8 on every push/PR
│       └── deploy.yml                             ← SSH into server + run deploy.sh on main
│
├── docker-compose.yml                             ← LOCAL DEV: all services
├── docker-compose.prod.yml                        ← PRODUCTION: all services
├── Dockerfile                                     ← Multi-stage: builder + production
│
│  ══════════════════════════════════════════════
│  PROJECT ROOT FILES
│  ══════════════════════════════════════════════
│
├── requirements/
│   ├── base.txt                                   ← django, drf, celery, channels, etc.
│   ├── local.txt                                  ← debug-toolbar, pytest, factory-boy
│   └── production.txt                             ← gunicorn, sentry-sdk, whitenoise
│
├── .env.example                                   ← All variables documented, no real values
├── .env                                           ← NEVER committed (in .gitignore)
├── .dockerignore                                  ← excludes .env, venv/, __pycache__/
├── .gitignore
├── pytest.ini                                     ← testpaths = apps, DJANGO_SETTINGS_MODULE
├── setup.cfg                                      ← flake8 + isort config
└── manage.py
```

---

## The Layered Rule — How Every App Is Built Internally

```
HTTP Request arrives
       │
       ▼
   urls.py          ROUTING ONLY. Zero logic.
       │
       ▼
   views.py          HTTP ONLY. Parse request → call service/selector → return Response.
       │                        No ORM. No business logic. No AI calls.
       ├─── READ ───► selectors.py    ALL ORM read queries live here only.
       │
       └─── WRITE ──► services.py    ALL business logic + transaction.atomic()
                            │                          + on_commit() + Celery dispatch
                            ▼
                       models.py     PURE data definitions. No logic whatsoever.
```

---

## Services Each Docker Container Runs

```yaml
LOCAL (docker-compose.yml)          PRODUCTION (docker-compose.prod.yml)
─────────────────────────           ────────────────────────────────────
db          PostgreSQL 15            db           PostgreSQL 15
redis       Redis 7                  redis        Redis 7
web         Django runserver         web          Gunicorn (3 workers)
daphne      Daphne ASGI              daphne       Daphne ASGI (WebSockets)
celery      Celery worker            celery       Celery worker (urgent + background queues)
celery-beat Celery Beat              celery-beat  Celery Beat (weekly AI report scheduler)
flower      Celery monitoring        nginx        Nginx (reverse proxy + SSL)
elastic     Elasticsearch 8         elastic      Elasticsearch 8
```

---

## The 3 Dashboards — Where Each Lives

```
ROLE          URL PREFIX           Source Files
──────────────────────────────────────────────────────────
Client        /client/*            frontend/src/pages/client/
Freelancer    /freelancer/*        frontend/src/pages/freelancer/
Super Admin   /admin/              Django Admin (built-in, customized in each apps/*/admin.py)
```

---

## What Each requirements/base.txt Contains

```
# Core
django==4.2.*
djangorestframework
djangorestframework-simplejwt
django-allauth
django-cors-headers
django-environ
django-filter

# Database
psycopg2
django-redis

# Async / Real-time
channels
channels-redis
daphne

# Task Queue
celery
django-celery-beat
flower

# Search
elasticsearch-dsl
django-elasticsearch-dsl

# Storage
django-storages
boto3

# AI
anthropic

# PDF
weasyprint

# Payments
stripe

# Security
django-axes

# API Docs
drf-spectacular
```

---

## The One Golden Rule for This Project

```
views.py     → knows about HTTP only
services.py  → knows about business logic only
selectors.py → knows about read queries only
models.py    → knows about data structure only
tasks.py     → knows about async work only
ai_service.py → knows about AI only (worklogs app only)
pdf_service.py → knows about PDF only (worklogs app only)
```

**No layer ever does another layer's job.**
**No file ever imports from a file at the same or higher layer.**
