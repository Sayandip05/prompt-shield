# FreelanceFlow Backend Implementation Plan

## Phase 1: Foundation Setup

### Task 1.1: Core Settings & Configuration
- Update `config/settings/base.py` with all required settings (DRF, JWT, CORS, Channels, Celery, Elasticsearch, S3, etc.)
- Create `config/settings/local.py` with DEBUG=True and local overrides
- Create `config/settings/production.py` with security settings
- Create `.env.example` with all environment variables

### Task 1.2: Core Module Utilities
- Implement `core/exceptions.py` - custom exception handler for consistent API error responses
- Implement `core/pagination.py` - StandardResultsPagination class
- Implement `core/permissions.py` - IsOwnerOrAdmin, BaseRolePermission base classes
- Implement `core/throttles.py` - TieredRateThrottle for rate limiting
- Implement `core/utils.py` - calculate_platform_cut, generate_report_id, format_currency

### Task 1.3: Celery & ASGI Configuration
- Complete `config/celery.py` with task discovery and queue configuration
- Complete `config/asgi.py` for WebSocket support with Channels

---

## Phase 2: Users App (Authentication & Profiles)

### Task 2.1: Models (`apps/users/models.py`)
- Complete User model (email as username, role field already exists)
- Complete FreelancerProfile (skills, hourly_rate, bio, subscription_tier)
- Complete ClientProfile (company_name, total_spent)

### Task 2.2: Selectors (`apps/users/selectors.py`)
- get_user_by_id(id)
- get_freelancer_profile(user)
- get_client_profile(user)
- list_freelancers(filters)

### Task 2.3: Services (`apps/users/services.py`)
- create_user(email, password, role, **extra_fields)
- update_profile(user, data)
- update_subscription_tier(user, tier)

### Task 2.4: Signals (`apps/users/signals.py`)
- Auto-create FreelancerProfile or ClientProfile on User creation

### Task 2.5: Serializers (`apps/users/serializers.py`)
- UserRegistrationSerializer
- UserSerializer (read)
- FreelancerProfileSerializer
- ClientProfileSerializer

### Task 2.6: Permissions (`apps/users/permissions.py`)
- IsFreelancer
- IsClient
- IsOwner

### Task 2.7: Views & URLs (`apps/users/views.py`, `apps/users/urls.py`)
- RegisterView, LoginView (JWT)
- ProfileView (GET/PATCH)
- UserDetailView

### Task 2.8: Tasks (`apps/users/tasks.py`)
- send_welcome_email_task(user_id)

### Task 2.9: Tests (`apps/users/tests/`)
- test_models.py, test_services.py, test_views.py

---

## Phase 3: Projects App

### Task 3.1: Models (`apps/projects/models.py`)
- Project (client FK, title, description, budget, deadline, status)
- ProjectSkill (project FK, skill_name)

### Task 3.2: Selectors (`apps/projects/selectors.py`)
- get_open_projects(filters)
- get_client_projects(client)
- get_project_by_id(id)

### Task 3.3: Services (`apps/projects/services.py`)
- create_project(client, data)
- update_project(project, data)
- close_project(project)

### Task 3.4: Permissions (`apps/projects/permissions.py`)
- IsProjectOwner

### Task 3.5: Serializers (`apps/projects/serializers.py`)
- ProjectCreateSerializer
- ProjectSerializer (read)
- ProjectListSerializer

### Task 3.6: Views & URLs (`apps/projects/views.py`, `apps/projects/urls.py`)
- ProjectViewSet (CRUD for projects)

### Task 3.7: Tests (`apps/projects/tests/`)
- test_models.py, test_services.py, test_views.py

---

## Phase 4: Bidding App (Bids & Contracts)

### Task 4.1: Models (`apps/bidding/models.py`)
- Bid (project FK, freelancer FK, amount, cover_letter, status)
- Contract (bid OneToOne, start_date, end_date, agreed_amount)

### Task 4.2: Selectors (`apps/bidding/selectors.py`)
- get_project_bids(project)
- get_freelancer_bids(freelancer)
- get_contract_by_id(id)
- get_active_contracts(freelancer)

### Task 4.3: Services (`apps/bidding/services.py`)
- submit_bid(freelancer, project, amount, cover_letter) - with validation
- accept_bid(bid_id, client) - **with select_for_update and transaction.atomic**
- reject_bid(bid)
- withdraw_bid(bid, freelancer)

### Task 4.4: Permissions (`apps/bidding/permissions.py`)
- IsBidOwner
- IsProjectClient

### Task 4.5: Serializers (`apps/bidding/serializers.py`)
- BidCreateSerializer
- BidSerializer
- ContractSerializer

### Task 4.6: Views & URLs (`apps/bidding/views.py`, `apps/bidding/urls.py`)
- BidViewSet
- ContractViewSet

### Task 4.7: Tests (`apps/bidding/tests/`)
- Focus on race condition test for accept_bid

---

## Phase 5: Payments App (Escrow & Stripe)

### Task 5.1: Models (`apps/payments/models.py`)
- Payment (contract FK, total_amount, status, stripe_payment_intent_id)
- Escrow (payment OneToOne, held_amount, released_at)
- PlatformEarning (payment FK, cut_percentage, cut_amount)
- PaymentEvent (payment FK, stripe_event_id) - idempotency

### Task 5.2: Selectors (`apps/payments/selectors.py`)
- get_payment_by_contract(contract)
- get_client_payment_history(client)
- get_freelancer_earnings(freelancer)

### Task 5.3: Services (`apps/payments/services.py`)
- create_escrow(contract, client) - Stripe PaymentIntent creation
- release_payment(contract, client) - platform cut + Stripe Transfer
- process_webhook_event(payload, sig_header) - HMAC verification

### Task 5.4: Permissions (`apps/payments/permissions.py`)
- IsPaymentParticipant

### Task 5.5: Serializers (`apps/payments/serializers.py`)
- EscrowCreateSerializer
- PaymentSerializer
- ReleasePaymentSerializer

### Task 5.6: Views & URLs (`apps/payments/views.py`, `apps/payments/urls.py`)
- EscrowView, ReleaseView
- WebhookView (Stripe)
- PaymentHistoryView

### Task 5.7: Tasks (`apps/payments/tasks.py`)
- process_stripe_webhook_task
- stripe_transfer_to_freelancer_task

### Task 5.8: Tests (`apps/payments/tests/`)
- test_services.py, test_views.py, test_webhook.py

---

## Phase 6: Worklogs App (Daily Logs, AI Reports, PDFs)

### Task 6.1: Models (`apps/worklogs/models.py`)
- WorkLog (contract FK, freelancer FK, date, description, hours_worked, screenshot_url, reference_url)
- WeeklyReport (contract FK, week_start, week_end, ai_summary, pdf_url)
- DeliveryProof (contract FK, pdf_url, generated_at, total_hours, total_logs_count)

### Task 6.2: Selectors (`apps/worklogs/selectors.py`)
- get_contract_logs(contract, date_range)
- get_weekly_reports(contract)
- get_delivery_proof(contract)

### Task 6.3: Services (`apps/worklogs/services.py`)
- create_log(freelancer, contract_id, data) - one log per day validation
- trigger_weekly_report(contract_id, week_start, week_end)
- generate_delivery_proof(contract_id)

### Task 6.4: Permissions (`apps/worklogs/permissions.py`)
- IsContractFreelancer
- IsContractParticipant

### Task 6.5: Serializers (`apps/worklogs/serializers.py`)
- WorkLogCreateSerializer
- WorkLogSerializer
- WeeklyReportSerializer
- DeliveryProofSerializer

### Task 6.6: AI Service (`apps/worklogs/ai_service.py`)
- build_prompt(contract_id, week_start, week_end)
- call_ai(prompt) - Anthropic API call

### Task 6.7: PDF Service (`apps/worklogs/pdf_service.py`)
- generate_weekly_report_pdf(report_id) - WeasyPrint + S3 upload
- generate_delivery_proof_pdf(proof_id)

### Task 6.8: Templates (`apps/worklogs/templates/worklogs/`)
- weekly_report.html
- delivery_proof.html

### Task 6.9: Views & URLs (`apps/worklogs/views.py`, `apps/worklogs/urls.py`)
- WorkLogViewSet
- WeeklyReportViewSet
- DeliveryProofView

### Task 6.10: Tasks (`apps/worklogs/tasks.py`)
- generate_ai_report_task
- generate_pdf_task
- check_if_week_complete

### Task 6.11: Tests (`apps/worklogs/tests/`)
- test_models.py, test_services.py, test_ai_service.py, test_pdf_service.py, test_views.py

---

## Phase 7: Messaging App (WebSocket Chat)

### Task 7.1: Models (`apps/messaging/models.py`)
- Conversation (contract OneToOne)
- Message (conversation FK, sender FK, content, created_at, is_read)

### Task 7.2: Selectors (`apps/messaging/selectors.py`)
- get_conversation(contract_id)
- get_messages(conversation, pagination)

### Task 7.3: Services (`apps/messaging/services.py`)
- create_conversation(contract)
- send_message(conversation, sender, content)
- mark_messages_read(conversation, user)

### Task 7.4: Consumer (`apps/messaging/consumers.py`)
- ChatConsumer (connect, receive, disconnect)
- JWT validation from query string
- Channel group joining

### Task 7.5: Routing (`apps/messaging/routing.py`)
- WebSocket URL routing

### Task 7.6: Serializers (`apps/messaging/serializers.py`)
- MessageSerializer
- ConversationSerializer

### Task 7.7: Views & URLs (`apps/messaging/views.py`, `apps/messaging/urls.py`)
- ConversationViewSet
- MessageListView

### Task 7.8: Tests (`apps/messaging/tests/`)
- test_consumers.py, test_views.py

---

## Phase 8: Notifications App

### Task 8.1: Models (`apps/notifications/models.py`)
- Notification (recipient FK, title, body, type, is_read, created_at)

### Task 8.2: Selectors (`apps/notifications/selectors.py`)
- get_user_notifications(user, unread_only)
- get_unread_count(user)

### Task 8.3: Services (`apps/notifications/services.py`)
- create_notification(recipient, title, body, type)
- mark_as_read(notification_id, user)
- mark_all_read(user)

### Task 8.4: Serializers (`apps/notifications/serializers.py`)
- NotificationSerializer

### Task 8.5: Views & URLs (`apps/notifications/views.py`, `apps/notifications/urls.py`)
- NotificationViewSet
- MarkAllReadView

### Task 8.6: Tasks (`apps/notifications/tasks.py`)
- create_notification_task
- send_email_notification_task

### Task 8.7: Tests (`apps/notifications/tests/`)
- test_services.py

---

## Phase 9: Search App (Elasticsearch)

### Task 9.1: Documents (`apps/search/documents.py`)
- ProjectDocument (Elasticsearch mapping)

### Task 9.2: Signals (`apps/search/signals.py`)
- Sync project to Elasticsearch on save (via Celery task)

### Task 9.3: Views & URLs (`apps/search/views.py`, `apps/search/urls.py`)
- ProjectSearchView (fuzzy matching, filters)

### Task 9.4: Tests (`apps/search/tests/`)
- test_views.py

---

## Phase 10: Admin Customization

### Task 10.1: Admin for Each App
- users/admin.py - User, FreelancerProfile, ClientProfile admin
- projects/admin.py - Project admin with skills inline
- bidding/admin.py - Bid, Contract admin
- payments/admin.py - Payment, Escrow, PlatformEarning admin
- worklogs/admin.py - WorkLog, WeeklyReport, DeliveryProof admin
- messaging/admin.py - Conversation, Message admin
- notifications/admin.py - Notification admin

---

## Phase 11: Final Integration & Testing

### Task 11.1: Integration Tests
- Full flow test: register -> create project -> bid -> contract -> escrow -> logs -> release

### Task 11.2: Celery Beat Configuration
- Weekly report generation schedule (Sunday 11:59 PM)

### Task 11.3: Docker Configuration
- Verify docker-compose.yml for local development
- Verify docker-compose.prod.yml for production

---

## Implementation Order Summary

```
1. Foundation (settings, core module)
2. Users (auth, profiles) ← foundation for all
3. Projects (depends on users)
4. Bidding (depends on projects, users)
5. Payments (depends on bidding/contracts)
6. Worklogs (depends on contracts, unique AI feature)
7. Messaging (depends on contracts)
8. Notifications (used by all)
9. Search (depends on projects)
10. Admin customization
11. Integration testing
```

## Environment Variables Required

```bash
# Core
SECRET_KEY=your-secret-key
DEBUG=True/False
DATABASE_URL=postgres://...
REDIS_URL=redis://...

# External Services (can be placeholders for dev)
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_STORAGE_BUCKET_NAME=...
ANTHROPIC_API_KEY=sk-ant-...
ELASTICSEARCH_URL=http://localhost:9200
```