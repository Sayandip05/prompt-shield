# FreelanceFlow Backend - Complete Interview Guide

This document provides a comprehensive overview of the FreelanceFlow Django backend project for interview preparation. It covers every file, model, view, service, and key concept.

---

## Table of Contents
1. [Project Architecture Overview](#1-project-architecture-overview)
2. [Core Module (core/)](#2-core-module-core)
3. [Users App](#3-users-app)
4. [Projects App](#4-projects-app)
5. [Bidding App](#5-bidding-app)
6. [Payments App](#6-payments-app)
7. [Worklogs App](#7-worklogs-app)
8. [Messaging App](#8-messaging-app)
9. [Notifications App](#9-notifications-app)
10. [Search App](#10-search-app)
11. [Configuration Files](#11-configuration-files)
12. [Key Interview Topics](#12-key-interview-topics)

---

## 1. Project Architecture Overview

### Project Structure
```
FreelanceFlow/
├── apps/           # Django applications
│   ├── users/      # User authentication & profiles
│   ├── projects    # Project management
│   ├── bidding     # Bidding & contracts
│   ├── payments    # Payment processing (Razorpay)
│   ├── worklogs    # Work logging & reporting
│   ├── messaging   # Real-time chat (WebSocket)
│   ├── notifications # In-app notifications
│   └── search      # Elasticsearch integration
├── config/         # Django settings & configuration
│   ├── settings/  # Environment-specific settings
│   ├── urls.py    # Root URL configuration
│   ├── celery.py  # Celery configuration
│   ├── asgi.py    # ASGI application (WebSocket)
│   └── wsgi.py    # WSGI application
├── core/          # Shared utilities & middleware
└── manage.py      # Django management script
```

### Technology Stack
- **Framework**: Django 5.x with Django REST Framework
- **Authentication**: JWT (djangorestframework-simplejwt)
- **Database**: PostgreSQL (production), SQLite (development)
- **Cache**: Redis (Upstash) - for caching & rate limiting
- **Task Queue**: Celery with Redis broker
- **Real-time**: Django Channels with WebSocket
- **Search**: Elasticsearch with django-elasticsearch-dsl
- **Payments**: Razorpay (Indian payment gateway)
- **AI Integration**: Groq API for AI-powered reports

### Installed Apps (base.py:58-92)
```python
DJANGO_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
]

THIRD_PARTY_APPS = [
    "rest_framework",
    "rest_framework_simplejwt",
    "rest_framework_simplejwt.token_blacklist",
    "corsheaders",
    "django_filters",
    "channels",
    "django_celery_beat",
    "django_elasticsearch_dsl",
    "django_extensions",
    "axes",  # Brute force protection
]

LOCAL_APPS = [
    "core",
    "apps.users",
    "apps.projects",
    "apps.bidding",
    "apps.payments",
    "apps.worklogs",
    "apps.messaging",
    "apps.notifications",
    "apps.search",
]
```

---

## 2. Core Module (core/)

The core module contains shared utilities, middleware, permissions, and custom implementations used across all apps.

### 2.1 core/utils.py

**Purpose**: Common utility functions used throughout the project.

**Key Functions**:
- `calculate_platform_cut(amount, percentage=10)` - Calculates platform fee (10% default) and freelancer's net amount
- `generate_report_id()` - Generates unique UUID for reports
- `format_currency(amount, currency="$")` - Formats decimal as currency string
- `truncate_text(text, max_length=100, suffix="...")` - Truncates text with ellipsis

**Imports Used**: `uuid`, `decimal.Decimal`, `ROUND_HALF_UP`

---

### 2.2 core/permissions.py

**Purpose**: Custom permission classes for role-based access control.

**Classes**:
1. `IsOwnerOrAdmin` - Allows object owners or admin users
2. `BaseRolePermission` - Base class for role-based permissions
3. `IsClient` - Only clients can access (role="CLIENT")
4. `IsFreelancer` - Only freelancers can access (role="FREELANCER")
5. `IsOwner` - Only object owner can access

**Key Pattern**: Uses `has_object_permission()` and `has_permission()` methods

---

### 2.3 core/pagination.py

**Purpose**: Custom pagination for API responses.

**Class**: `StandardResultsPagination(PageNumberPagination)`
- Default: 20 items per page
- Max: 100 items per page (`?page_size=`)
- Returns format: `{count, next, previous, results}`

**Import**: `rest_framework.pagination.PageNumberPagination`, `rest_framework.response.Response`

---

### 2.4 core/exceptions.py

**Purpose**: Custom exception handling and business logic exceptions.

**Functions**:
- `custom_exception_handler(exc, context)` - Returns consistent error format: `{error, code, field}`

**Exception Classes**:
1. `BusinessError` - Base exception with message and code
2. `PermissionDeniedError` - Code: "permission_denied"
3. `NotFoundError` - Code: "not_found"
4. `ValidationError` - Code: "validation_error" with optional field

**Pattern**: All custom exceptions inherit from `BusinessError(Exception)` with message and code attributes

---

### 2.5 core/middleware.py

**Purpose**: Custom middleware for request/response processing.

**Classes**:
1. **RequestLoggingMiddleware** - Logs every API request with method, path, status, duration_ms, user_id, IP
2. **SecurityHeadersMiddleware** - Adds X-Content-Type-Options, X-XSS-Protection, Referrer-Policy headers
3. **RateLimitMiddleware** - Simple Redis-based rate limiting (100/hour anonymous, 1000/hour authenticated)
4. **CacheControlMiddleware** - Adds cache headers, excludes /api/, /admin/, /ws/
5. **CORSCustomMiddleware** - Handles CORS with custom configuration

**Key Pattern**: Each middleware implements `__init__(get_response)` and `__call__(request)`

---

### 2.6 core/decorators.py

**Purpose**: Custom decorators for views.

**Decorators**:
- `@api_csrf_exempt` - Exempts API views from CSRF validation (for token auth)

**Import**: `django.views.decorators.csrf.csrf_exempt`

---

### 2.7 core/throttles.py

**Purpose**: Rate limiting based on subscription tier.

**Classes**:
1. **TieredRateThrottle** - Different rates for FREE vs PRO users
   - FREE: 30 requests/minute
   - PRO: 300 requests/minute
2. **LoginRateThrottle** - Strict 5/minute for login attempts

---

### 2.8 core/cache.py

**Purpose**: Centralized cache configuration (Redis-based)

---

### 2.9 core/sanitizers.py

**Purpose**: Input sanitization for security

---

### 2.10 core/middleware_shutdown.py

**Purpose**: Graceful shutdown handling for Django

---

### 2.11 core/health.py

**Purpose**: Health check endpoint for monitoring

---

### 2.12 core/__init__.py

**Purpose**: Contains AppConfig for core module

---

## 3. Users App

**Purpose**: Handles user authentication, registration, profiles, and account management.

### 3.1 apps/users/models.py

**Models**:

#### User (AbstractUser subclass)
- **Fields**:
  - `email` - Unique email (replaces username as USERNAME_FIELD)
  - `role` - CHOICES: CLIENT, FREELANCER
  - `is_deactivated` - Boolean for soft delete
  - `deactivated_at` - DateTime when deactivated
- **Custom Manager**: `UserManager` with `create_user()` and `create_superuser()`
- **Key Method**: `full_name` property

#### FreelancerProfile (OneToOne to User)
- **Fields**:
  - `bio` - Text field
  - `skills` - JSONField (list of skills)
  - `hourly_rate` - DecimalField
  - `subscription_tier` - CHOICES: FREE, PRO
  - `total_earned` - DecimalField
  - `avatar` - URLField
  - `is_available` - Boolean for availability toggle
  - `average_rating`, `total_reviews` - Rating fields
  - `razorpay_fund_account_id` - For payouts

#### ClientProfile (OneToOne to User)
- **Fields**:
  - `company_name` - CharField
  - `total_spent` - DecimalField
  - `avatar`, `average_rating`, `total_reviews` - Similar to freelancer

---

### 3.2 apps/users/signals.py

**Purpose**: Auto-creates profiles when User is created.

**Signal Handlers**:
1. `post_save` receiver `create_user_profile` - Creates FreelancerProfile or ClientProfile based on role
2. `post_save` receiver `save_user_profile` - Ensures profile exists on save

**Key Pattern**: Uses `transaction.on_commit()` to send welcome email async

---

### 3.3 apps/users/serializers.py

**Key Serializers**:
- `FreelancerProfileSerializer` - Freelancer profile fields
- `ClientProfileSerializer` - Client profile fields
- `UserSerializer` - Full user with nested profiles
- `UserRegistrationSerializer` - Registration with password validation
- `UserProfileUpdateSerializer` - Updates user + profile fields
- `ChangePasswordSerializer` - Password change validation
- `PasswordResetRequestSerializer`, `PasswordResetConfirmSerializer` - Password reset
- `AvatarUploadSerializer` - Avatar URL upload
- `AvailabilityToggleSerializer` - Freelancer availability toggle
- `AccountDeactivationSerializer` - Soft delete with "DEACTIVATE" confirmation

---

### 3.4 apps/users/services.py

**Key Functions**:
1. `create_user(email, password, role, first_name, last_name)` - Creates user with profile via signal
2. `update_profile(user, data)` - Updates user and profile based on role
3. `change_password(user, old_password, new_password)` - Validates old password, sets new
4. `send_password_reset_email(email)` - Sends password reset email (doesn't reveal if email exists)
5. `reset_password(uid, token, new_password)` - Validates token, resets password
6. `send_verification_email(user)` - Sends email verification
7. `verify_email(uid, token)` - Verifies email with token
8. `update_avatar(user, avatar_url)` - Updates avatar based on role
9. `toggle_freelancer_availability(user, is_available)` - Sets freelancer availability
10. `deactivate_account(user, password)` - Soft delete with password confirmation
11. `reactivate_account(user)` - Reactivates deactivated account

**Key Pattern**: Uses `transaction.atomic()` and raises `ValidationError` from core.exceptions

---

### 3.5 apps/users/views.py

**Key Views** (all inherit from `generics` or extend ViewSets):

1. **RegisterView** (CreateAPIView) - POST /register/
2. **LoginView** (TokenObtainPairView) - POST /login/ - JWT token generation
3. **ProfileView** (RetrieveUpdateAPIView) - GET/PATCH /me/
4. **ChangePasswordView** (GenericAPIView) - POST /change-password/
5. **UserDetailView** (RetrieveAPIView) - GET /<id>/
6. **PasswordResetRequestView** - POST /password-reset/
7. **PasswordResetConfirmView** - POST /password-reset/confirm/
8. **EmailVerificationView** - POST /verify-email/
9. **ResendVerificationEmailView** - POST /resend-verification/
10. **UpdateAvatarView** - POST /avatar/
11. **ToggleAvailabilityView** - POST /availability/
12. **DeactivateAccountView** - POST /deactivate/
13. **ReactivateAccountView** - POST /reactivate/

**Throttling**: Auth endpoints use `AuthRateThrottle` (5/minute)

---

### 3.6 apps/users/selectors.py

**Functions**:
- `get_user_by_id(user_id)` - Returns User or 404
- `get_user_by_email(email)` - Returns User or None
- `get_freelancer_profile(user)` - Returns FreelancerProfile or None
- `get_client_profile(user)` - Returns ClientProfile or None
- `list_freelancers(skills, limit)` - Lists freelancers with optional skill filter (uses PostgreSQL `__overlap`)
- `list_clients(limit)` - Lists clients

---

### 3.7 apps/users/urls.py

**URL Patterns**:
```
register/, login/, token/refresh/,
me/, change-password/, avatar/, availability/,
password-reset/, password-reset/confirm/,
verify-email/, resend-verification/,
deactivate/, reactivate/,
<int:pk>/
```

---

### 3.8 apps/users/tasks.py

**Key Tasks** (Celery):
- `send_welcome_email_task(user_id)` - Async welcome email

---

### 3.9 apps/users/permissions.py

**Purpose**: User-specific permission classes (extends core/permissions.py)

---

### 3.10 apps/users/admin.py

**Purpose**: Django admin configuration for User, FreelancerProfile, ClientProfile

---

## 4. Projects App

**Purpose**: Manages client-posted projects that freelancers can bid on.

### 4.1 apps/projects/models.py

**Models**:

#### Project
- **Fields**:
  - `client` - ForeignKey to User (CLIENT role only)
  - `title` - CharField
  - `description` - TextField
  - `budget` - DecimalField
  - `deadline` - DateField (optional)
  - `status` - CHOICES: OPEN, IN_PROGRESS, COMPLETED, CANCELLED
  - `created_at`, `updated_at` - DateTime fields
- **Ordering**: `-created_at`

#### ProjectSkill
- **Fields**:
  - `project` - ForeignKey to Project
  - `skill_name` - CharField
- **Constraint**: unique_together(project, skill_name)

---

### 4.2 apps/projects/services.py

**Key Functions**:
- `create_project(client, title, description, budget, deadline, skills)` - Creates project with skills
- `update_project(project, data)` - Updates project fields
- `mark_project_completed(project)` - Changes status to COMPLETED

---

### 4.3 apps/projects/serializers.py

**Key Serializers**:
- `ProjectSerializer` - Full project with skills
- `ProjectCreateSerializer` - For creating projects
- `ProjectListSerializer` - For listing projects

---

### 4.4 apps/projects/views.py

**ViewSet**: `ProjectViewSet` with standard CRUD operations

---

### 4.5 apps/projects/selectors.py

**Functions**:
- `get_project_by_id(project_id)` - Returns Project or 404
- `get_client_projects(client)` - Returns client's projects
- `get_open_projects()` - Returns projects with OPEN status

---

## 5. Bidding App

**Purpose**: Handles freelancer bids on projects and contract management.

### 5.1 apps/bidding/models.py

**Models**:

#### Bid
- **Fields**:
  - `project` - ForeignKey to Project
  - `freelancer` - ForeignKey to User (FREELANCER role only)
  - `amount` - DecimalField (bid amount)
  - `cover_letter` - TextField (proposal)
  - `status` - CHOICES: PENDING, ACCEPTED, REJECTED, WITHDRAWN
- **Constraint**: unique_together(project, freelancer)
- **Ordering**: `-created_at`

#### Contract
- **Fields**:
  - `bid` - OneToOneField to Bid
  - `agreed_amount` - DecimalField
  - `start_date` - DateTime (auto_now_add)
  - `end_date` - DateTime (nullable)
  - `is_active` - Boolean (default True)
- **Properties**: `project`, `freelancer`, `client` (via bid relationship)

---

### 5.2 apps/bidding/services.py

**Key Functions**:
1. `create_bid(freelancer, project, amount, cover_letter)` - Creates bid
2. `accept_bid(bid, client)` - Accepts bid, creates Contract
3. `reject_bid(bid, client)` - Rejects bid
4. `withdraw_bid(bid, freelancer)` - Withdraws bid
5. `complete_contract(contract)` - Marks contract as completed

---

### 5.3 apps/bidding/views.py

**ViewSets**:
- `BidViewSet` - CRUD for bids
- `ContractViewSet` - CRUD for contracts

---

### 5.4 apps/bidding/serializers.py

- `BidSerializer` - Full bid with project info
- `BidCreateSerializer` - For creating bids
- `ContractSerializer` - Full contract with bid info
- `ContractCreateSerializer` - Auto-creates from accepted bid

---

### 5.5 Additional Bidding Services (extended)

- `services_counter_offer.py` - Counter offer functionality
- `services_retraction.py` - Bid retraction
- `services_amendment.py` - Contract amendments
- `services_termination.py` - Contract termination
- `services_review.py` - Review/rating system
- `services_worklog_approval.py` - Worklog approval workflow

---

## 6. Payments App

**Purpose**: Handles payment processing via Razorpay, including escrow, payouts, and refunds.

### 6.1 apps/payments/models.py

**Models**:

#### Payment
- **Fields**:
  - `contract` - OneToOneField to Contract
  - `total_amount` - DecimalField
  - `status` - CHOICES: PENDING, ESCROWED, PAYOUT_PENDING, RELEASED, PAYOUT_FAILED, REFUNDED
  - `razorpay_order_id` - CharField (Razorpay Order ID)
  - `razorpay_payment_id` - CharField (Payment ID)
  - `razorpay_payout_id` - CharField (Payout ID for freelancer)
  - `payout_error` - TextField (error message if payout fails)
  - `razorpay_refund_id`, `refund_amount` - For refunds
- **Ordering**: `-created_at`

#### Escrow
- **Fields**:
  - `payment` - OneToOneField to Payment
  - `held_amount` - DecimalField
  - `released_at` - DateTime (nullable)
  - `refund_amount` - For partial refunds

#### PlatformEarning
- **Fields**:
  - `payment` - ForeignKey to Payment
  - `cut_percentage` - DecimalField (e.g., 10%)
  - `cut_amount` - DecimalField (actual platform fee)
- **Purpose**: Revenue tracking for platform

#### PaymentEvent
- **Fields**:
  - `payment` - ForeignKey to Payment
  - `razorpay_event_id` - CharField (unique)
  - `event_type` - CharField (e.g., "payment.captured")
  - `processed_at` - DateTime
- **Purpose**: Webhook idempotency - prevents duplicate processing

---

### 6.2 apps/payments/services.py

**Key Functions**:

1. **Escrow Creation**:
   - `create_escrow(contract, client)` - Creates Razorpay order and Payment record
   - Converts amount to paise (multiply by 100)
   - Raises `PermissionDeniedError` if not client
   - Raises `ValidationError` if payment already exists

2. **Payment Confirmation**:
   - `confirm_escrow_payment(razorpay_order_id, razorpay_payment_id)` - Updates status to ESCROWED, creates Escrow record

3. **Payment Release**:
   - `release_payment(contract, client)` - Initiates payout to freelancer
   - Calculates platform cut using `core.utils.calculate_platform_cut()`
   - Schedules async task for RazorpayX payout
   - Requires freelancer's `razorpay_fund_account_id`

4. **Razorpay Integration**:
   - `verify_razorpay_signature(order_id, payment_id, signature)` - HMAC-SHA256 verification
   - `process_razorpay_webhook(payload, raw_body, signature, event_id)` - Processes webhooks async
   - `has_payment_event_been_processed(event_id)` - Idempotency check
   - `record_payment_event(payment, event_id, event_type)` - Records processed events

5. **Refunds**:
   - `process_refund(payment_id, refund_amount, reason)` - Processes Razorpay refund
   - `process_contract_termination_payment(payment, refund_percentage)` - Handles partial refund + payout

6. **Disputes**:
   - `initiate_payment_dispute(payment_id, disputer, reason, description)` - Creates dispute, notifies parties

---

### 6.3 apps/payments/views.py

**Views**:

#### PaymentViewSet (ReadOnlyModelViewSet)
- `list` - GET /payments/ - Lists user's payments (filtered by role)
- `retrieve` - GET /payments/{id}/ - Get payment detail
- `escrow` - POST /payments/escrow/ - Create escrow payment
- `release` - POST /payments/release/ - Release payment to freelancer
- `history` - GET /payments/history/ - Payment history summary

**Permissions**: `IsPaymentParticipant` - Only contract participants can view

**Standalone Views**:
- `verify_payment` - POST /payments/verify/ - Client verifies payment after frontend completion
- `razorpay_webhook` - POST /payments/webhook/ - Razorpay webhook endpoint (AllowAny)

---

### 6.4 apps/payments/tasks.py

**Celery Tasks** (async processing):

1. **`process_razorpay_webhook_task(event_id, event_type, event_data)`**
   - Processes Razorpay webhook events asynchronously
   - Handles: `payment.captured`, `payment.failed`
   - Calls `confirm_escrow_payment()` and `record_payment_event()`

2. **`razorpay_transfer_to_freelancer_task(payment_id, amount)`**
   - Creates RazorpayX payout to freelancer's fund account
   - Uses `IMPS` mode for transfer
   - On success: Updates status to RELEASED, creates PlatformEarning, ends contract, generates delivery proof
   - On failure: Updates status to PAYOUT_FAILED with error message

3. **`process_razorpay_refund_task(payment_id, refund_amount)`**
   - Processes refund asynchronously

---

### 6.5 apps/payments/selectors.py

**Functions**:
- `get_payment_by_id(payment_id)` - Returns Payment or 404
- `get_payment_by_contract(contract_id)` - Returns Payment or None
- `get_client_payment_history(client)` - All payments made by client
- `get_freelancer_earnings(freelancer)` - Released payments to freelancer
- `get_freelancer_total_earned(freelancer)` - Sum of all earnings
- `get_client_total_spent(client)` - Sum of all spent (escrowed + released)
- `get_platform_total_earnings()` - Total platform cut

---

### 6.6 apps/payments/serializers.py

- `PaymentSerializer` - Full payment details
- `PaymentListSerializer` - List view with less detail
- `CreateEscrowSerializer` - Validates contract_id
- `ReleasePaymentSerializer` - Validates contract_id for release
- `PaymentHistorySerializer` - Summary stats

---

### 6.7 apps/payments/admin.py

**Purpose**: Django admin for Payment, Escrow, PlatformEarning, PaymentEvent

---

### 6.8 Additional Payment Models

- `models_milestone.py` - Milestone-based payments
- `models_dispute.py` - PaymentDispute model
- `models_extended.py` - Extended payment features
- `services_invoice.py` - Invoice generation
- `services_currency.py` - Currency conversion
- `services_tax.py` - Tax calculation
- `services_milestone.py` - Milestone payment logic

---

## 7. Worklogs App

**Purpose**: Daily work logging, weekly AI reports, deliverables, and delivery proof generation.

### 7.1 apps/worklogs/models.py

**Models**:

#### WorkLog
- **Fields**:
  - `contract` - ForeignKey to Contract
  - `freelancer` - ForeignKey to User
  - `date` - DateField
  - `description` - TextField
  - `hours_worked` - DecimalField (0.1-24, validated)
  - `screenshot` - ImageField (upload to S3)
  - `screenshot_url`, `reference_url` - URLFields
  - `status` - CHOICES: DRAFT, PENDING_APPROVAL, APPROVED, REJECTED
  - `ai_generated_summary` - TextField (from AI chat)
  - `client_notes` - TextField (feedback)
  - `approved_at`, `approved_by` - Approval tracking
- **Constraint**: unique_together(contract, date)

#### WeeklyReport
- **Fields**:
  - `contract` - ForeignKey to Contract
  - `week_start`, `week_end` - DateField
  - `ai_summary` - TextField (AI-generated)
  - `pdf_url` - URLField (S3)
  - `sent_to_client_at` - DateTime
- **Property**: `total_hours` - Sum of worklogs for the week

#### Deliverable
- **Fields**:
  - `contract`, `freelancer` - ForeignKeys
  - `title`, `description` - CharField/TextField
  - `ai_chat_transcript` - JSONField (full chat with AI)
  - `ai_generated_report` - TextField
  - `attached_files` - JSONField
  - `status` - CHOICES: DRAFT, SUBMITTED, UNDER_REVIEW, APPROVED, REJECTED, REVISION_REQUESTED
  - `submitted_at`, `reviewed_at`, `reviewed_by`
  - `client_feedback`, `revision_notes` - TextFields
  - `hours_logged` - DecimalField
  - `payment_released` - Boolean

#### DeliveryProof
- **Fields**:
  - `contract` - OneToOneField to Contract
  - `pdf_url` - URLField (S3)
  - `generated_at` - DateTime
  - `total_hours`, `total_logs_count` - Statistics
  - `total_deliverables`, `approved_deliverables` - Counts
  - `report_id` - Unique CharField (tamper-evident)

---

### 7.2 apps/worklogs/services.py

**Key Functions**:

1. **WorkLog Management**:
   - `create_worklog(freelancer, contract_id, log_date, description, hours_worked, ...)` - Creates daily log
   - `update_worklog(log, freelancer, ...)` - Updates log
   - `delete_worklog(log, freelancer)` - Deletes log

2. **Delivery Proof**:
   - `generate_delivery_proof(contract_id)` - Creates DeliveryProof with PDF
   - Uses `report_id` from `core.utils.generate_report_id()`

3. **AI Integration** (via `groq_service.py`):
   - Uses Groq API for AI-powered summaries
   - LangSmith integration for tracing

---

### 7.3 apps/worklogs/tasks.py

**Celery Tasks**:

1. **`generate_pdf_task`** - Low priority, generates PDF reports
2. **`generate_ai_report_task`** - Generates weekly AI reports
3. **`generate_weekly_reports_for_all_contracts`** - Bulk report generation
4. **`notify_client_log_submitted`** - Notifies client of new work log

---

### 7.4 apps/worklogs/views.py

**ViewSets**:
- `WorkLogViewSet` - CRUD for work logs
- `WeeklyReportViewSet` - Weekly reports
- `DeliverableViewSet` - Deliverables management

---

### 7.5 apps/worklogs/pdf_service.py

**Purpose**: PDF generation for delivery proof and weekly reports using reportlab or similar

---

## 8. Messaging App

**Purpose**: Real-time chat between client and freelancer per contract using WebSockets.

### 8.1 apps/messaging/models.py

**Models**:

#### Conversation
- **Fields**:
  - `contract` - OneToOneField to Contract
  - `created_at`, `updated_at` - DateTime
- **Ordering**: `-updated_at`

#### Message
- **Fields**:
  - `conversation` - ForeignKey to Conversation
  - `sender` - ForeignKey to User
  - `content` - TextField
  - `attachments` - JSONField (list of attachment URLs with metadata)
  - `is_read` - Boolean
  - `created_at` - DateTime

---

### 8.2 apps/messaging/services.py

**Functions**:

1. `get_or_create_conversation(contract_id)` - Returns Conversation, creates if not exists
2. `send_message(sender, conversation_id, content)` - Creates Message, updates conversation timestamp
3. `mark_messages_as_read(conversation_id, user)` - Marks unread messages as read (excludes user's own messages)

---

### 8.3 apps/messaging/views.py

**ViewSets**:

#### ConversationViewSet (ReadOnlyModelViewSet)
- `list` - GET /messaging/conversations/
- `retrieve` - GET /messaging/conversations/{id}/
- `messages` - GET /messaging/conversations/{id}/messages/ (paginated)
- `send` - POST /messaging/conversations/{id}/send/
- `mark_read` - POST /messaging/conversations/{id}/mark_read/

#### MessageViewSet (ReadOnlyModelViewSet)
- List messages filtered by conversation

---

### 8.4 apps/messaging/consumers.py (WebSocket)

**Class**: `ChatConsumer(AsyncWebsocketConsumer)`

**WebSocket Flow**:
1. `connect()` - Authenticates via JWT token in query string, validates contract participant
2. `disconnect()` - Leaves room group
3. `receive(text_data)` - Parses JSON, saves message, broadcasts to room group
4. `chat_message(event)` - Sends message to WebSocket

**Room Pattern**: `chat_{contract_id}` - Group name for contract-specific chat

**Authentication**:
- JWT token from query string: `ws://host/ws/chat/{contract_id}/?token={jwt}`
- Validates user is part of contract (either client or freelancer)

---

### 8.5 apps/messaging/routing.py

**URL Patterns**: WebSocket URL routes for chat

```python
websocket_urlpatterns = [
    path("ws/chat/<int:contract_id>/", ChatConsumer.as_asgi()),
]
```

---

### 8.6 apps/messaging/serializers.py

- `ConversationSerializer` - Full conversation with contract info
- `MessageSerializer` - Full message with sender info
- `SendMessageSerializer` - Validates content field

---

### 8.7 apps/messaging/selectors.py

**Functions**:
- `get_user_conversations(user)` - Returns conversations where user is participant
- `get_conversation_messages(conversation_id)` - Returns messages for conversation
- `get_conversation_by_contract(contract_id)` - Returns conversation or None

---

## 9. Notifications App

**Purpose**: In-app notifications for various events.

### 9.1 apps/notifications/models.py

**Model**: Notification
- **Fields**:
  - `recipient` - ForeignKey to User
  - `title` - CharField
  - `body` - TextField
  - `type` - CHOICES: BID_SUBMITTED, BID_ACCEPTED, ESCROW_CREATED, LOG_SUBMITTED, REPORT_READY, PAYMENT_RELEASED, PROOF_READY, MESSAGE_RECEIVED
  - `is_read` - Boolean
  - `created_at` - DateTime
- **Indexes**: Index on (recipient, is_read) for performance

---

### 9.2 apps/notifications/services.py

**Functions**:
- `create_notification(recipient, title, body, notification_type, data)` - Creates notification

---

### 9.3 apps/notifications/tasks.py

**Celery Tasks** for sending notifications (email, push, etc.)

---

### 9.4 apps/notifications/views.py

**ViewSet**: `NotificationViewSet` for CRUD operations

---

## 10. Search App

**Purpose**: Elasticsearch integration for searching projects and freelancers.

### 10.1 apps/search/documents.py

**Elasticsearch Documents**:

#### ProjectDocument
- Index: "projects"
- Fields: id, title, description, budget, deadline, client_name, client_email, skills, status
- Related models: User

#### FreelancerDocument
- Index: "freelancers"
- Fields: id, bio, hourly_rate, subscription_tier, total_earned, email, full_name, skills
- Related models: User

**Key Pattern**: Uses `django_elasticsearch_dsl` with `@registry.register_document` decorator

---

### 10.2 apps/search/services.py

**Functions**:
- `search_projects(query, filters)` - Elasticsearch search for projects
- `search_freelancers(query, filters)` - Elasticsearch search for freelancers

---

### 10.3 apps/search/views.py

**ViewSet**: `SearchViewSet` with search endpoints

---

## 11. Configuration Files

### 11.1 config/settings/base.py

**Key Settings**:

1. **Environment Variables** (via `django-environ`):
   - `SECRET_KEY`, `DEBUG`, `ALLOWED_HOSTS`
   - `DATABASE_URL` - PostgreSQL connection string
   - `REDIS_URL` - Redis/Upstash connection
   - `RAZORPAY_KEY_ID`, `RAZORPAY_KEY_SECRET`, `RAZORPAY_WEBHOOK_SECRET`, `RAZORPAY_ACCOUNT_NUMBER`
   - `AWS_*` - S3 configuration
   - `GROQ_API_KEY` - AI service
   - `ELASTICSEARCH_URL`
   - `PLATFORM_CUT_PERCENTAGE` (default: 10)
   - `SENTRY_DSN` - Error tracking

2. **Django REST Framework**:
   - JWT authentication: `rest_framework_simplejwt.authentication.JWTAuthentication`
   - Custom exception handler: `core.exceptions.custom_exception_handler`
   - Custom pagination: `core.pagination.StandardResultsPagination`
   - Throttle classes: AnonRateThrottle, UserRateThrottle
   - Filter backends: DjangoFilterBackend, SearchFilter, OrderingFilter

3. **JWT Settings**:
   - Access token: 60 minutes
   - Refresh token: 7 days
   - Blacklist after rotation enabled

4. **Channels** (WebSocket):
   - Backend: `channels_redis.core.RedisChannelLayer`
   - Redis URL from settings

5. **Celery**:
   - Broker: Redis
   - Result backend: Redis
   - Task serializer: JSON
   - Beat scheduler: `django_celery_beat.schedulers.DatabaseScheduler`

6. **CORS**:
   - `CORS_ALLOWED_ORIGINS` from env
   - `CORS_ALLOW_CREDENTIALS = True`

7. **Caches**:
   - Default: Redis cache (django_redis)
   - Fallback: LocMemCache for Axes

8. **Axes** (Brute Force Protection):
   - Database handler
   - 5 attempts, 5 minute cooldown

---

### 11.2 config/settings/local.py

**Purpose**: Development settings
- DEBUG = True
- Uses SQLite by default
- Allowed hosts: localhost

---

### 11.3 config/settings/production.py

**Purpose**: Production settings
- DEBUG = False
- Security settings (SSL, HSTS, secure cookies)
- S3 storage for static/media files
- PostgreSQL database
- Sentry integration for error tracking

---

### 11.4 config/urls.py

**Root URL Configuration**:
```python
urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/users/", include("apps.users.urls")),
    path("api/projects/", include("apps.projects.urls")),
    path("api/bidding/", include("apps.bidding.urls")),
    path("api/payments/", include("apps.payments.urls")),
    path("api/worklogs/", include("apps.worklogs.urls")),
    path("api/messaging/", include("apps.messaging.urls")),
    path("api/notifications/", include("apps.notifications.urls")),
    path("api/search/", include("apps.search.urls")),
]
```

---

### 11.5 config/celery.py

**Purpose**: Celery application configuration

**Key Configurations**:
1. **Queue Isolation**: Separate queues for different task priorities
   - `freelanceflow` - Default
   - `freelanceflow_high_priority` - Payments, webhooks
   - `freelanceflow_low_priority` - Reports, PDFs

2. **Task Routing**:
   - Payment tasks → high priority
   - PDF/report tasks → low priority

3. **Worker Settings**:
   - Max tasks per child: 1000
   - Soft time limit: 5 minutes
   - Hard time limit: 10 minutes
   - Acknowledge late: True

4. **Graceful Shutdown**:
   - `@worker_shutting_down.connect` - Closes DB and cache connections

---

### 11.6 config/asgi.py

**Purpose**: ASGI application for both HTTP and WebSocket

**Components**:
- `ProtocolTypeRouter` - Routes to HTTP or WebSocket
- `AuthMiddlewareStack` - JWT authentication for WebSocket
- `URLRouter` - Maps WebSocket URLs to consumers

**Graceful Shutdown**:
- `cleanup_on_shutdown()` async function
- Closes channel layer, DB connections, cache connections

---

### 11.7 config/wsgi.py

**Purpose**: WSGI application for traditional HTTP serving (Gunicorn)

---

### 11.8 config/signals.py

**Purpose**: Django signal configuration for apps (similar to ready() in AppConfig)

---

## 12. Key Interview Topics

### 12.1 Django & DRF Core

1. **Custom User Model**
   - Why use AbstractUser? What's the alternative?
   - What changes are needed in settings for custom user? (`AUTH_USER_MODEL`)
   - How to handle email as username?

2. **JWT Authentication**
   - How does djangorestframework-simplejwt work?
   - What are access and refresh tokens?
   - How to implement token blacklist for logout?

3. **Serializer Validation**
   - Field-level validation (`validate_<field>`)
   - Object-level validation (`validate()`)
   - Custom validators

4. **ViewSet vs Generic Views**
   - When to use each?
   - How does Router work?

5. **Custom Exception Handling**
   - Why custom exception handler?
   - How to return consistent error format?

---

### 12.2 Payment Integration

1. **Razorpay Flow**
   - Order creation → Payment → Webhook → Confirmation
   - Why use webhooks? (async nature of payment processing)

2. **Escrow System**
   - How does escrow work?
   - What's the payment lifecycle? (PENDING → ESCROWED → RELEASED)

3. **Payout to Freelancer**
   - RazorpayX payouts
   - What's a fund account?
   - Why use async tasks for payouts?

4. **Idempotency**
   - Why is it important for webhooks?
   - How is it implemented here? (`PaymentEvent` model)

---

### 12.3 Real-Time Features

1. **Django Channels**
   - How does WebSocket work with Django?
   - What is a channel layer?
   - How is authentication handled?

2. **WebSocket Consumer**
   - `AsyncWebsocketConsumer` vs `SyncConsumer`
   - Room groups for broadcasting

---

### 12.4 Task Queue & Celery

1. **Why Celery?**
   - Async processing for long-running tasks
   - Decouple from HTTP request/response

2. **Celery Architecture**
   - Broker (Redis)
   - Worker
   - Result backend

3. **Task Design**
   - When to use `@shared_task`?
   - How to handle retries?
   - What is `transaction.on_commit()`?

---

### 12.5 Database & Performance

1. **Query Optimization**
   - `select_related()` - ForeignKey/OneToOne
   - `prefetch_related()` - Reverse ForeignKey/M2M
   - Database indexes

2. **Caching**
   - Redis cache configuration
   - Cache invalidation strategies

3. **Rate Limiting**
   - How does Axes work?
   - Why tiered rate limiting?

---

### 12.6 Security

1. **CSRF**
   - Why exempt API from CSRF with JWT?
   - Proper headers

2. **CORS**
   - How to configure?
   - Why `AllowCredentials` requires specific origins?

3. **Brute Force Protection**
   - How does Axes work?

---

### 12.7 Architecture Patterns

1. **Service Layer Pattern**
   - Why separate services from views?
   - What goes in services vs views?

2. **Selector Pattern**
   - Why selectors instead of just models?
   - When to use?

3. **Repository Pattern** (implied)
   - How does selector act as repository?

---

### 12.8 Common Interview Questions

1. **Walk through the payment flow**
   - Client creates escrow → Razorpay order → Payment → Webhook → Release → Payout

2. **How does real-time messaging work?**
   - WebSocket connection → JWT auth → Room group → Broadcast

3. **Why use signals for profile creation?**
   - Automatic profile creation on user registration

4. **How to handle concurrent webhook events?**
   - Idempotency with PaymentEvent

5. **What's the platform cut calculation?**
   - `calculate_platform_cut()` function

6. **How to ensure security?**
   - CSRF exemption for token auth
   - Custom permissions
   - Rate limiting
   - Input validation

7. **Database relationships used?**
   - OneToOne (User↔Profile, Bid↔Contract)
   - ForeignKey (Project→User, WorkLog→Contract)
   - Many-to-many via through models

8. **Elasticsearch integration?**
   - django-elasticsearch-dsl documents
   - Signals for auto-indexing

---

### 12.9 Code Organization Questions

1. **Why separate services, selectors, serializers?**
   - Separation of concerns
   - Testability
   - Reusability

2. **What's the purpose of core module?**
   - Shared utilities
   - Cross-app concerns

3. **Why config as separate directory?**
   - Environment-specific settings
   - Clean separation

---

This comprehensive guide covers every aspect of the FreelanceFlow backend. Study each section, understand the relationships between components, and be prepared to discuss architecture decisions and trade-offs in interviews.