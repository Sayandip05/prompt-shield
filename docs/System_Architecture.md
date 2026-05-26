# FreelanceFlow - System Architecture

**Version:** 4.0.0  
**Last Updated:** May 1, 2026

---

## 1. Full Architecture Diagram

``````
                        USERS (Browsers)
                              |
                              v
                    +------------------+
                    |  Nginx (Port 443)|
                    |  - Rate Limiting |
                    |  - SSL/TLS       |
                    +--------+---------+
                             |
              +--------------+-------------+
              |              |             |
              v              v             v
      +--------------+ +----------+ +--------------+
      |  Gunicorn    | |  Daphne  | |    Celery    |
      |  (REST API)  | |(WebSocket)| |   Workers    |
      |  Port 8000   | |Port 8001 | |              |
      +------+-------+ +----+-----+ +------+-------+
             |              |              |
             +--------------+--------------+
                            |
        +-------------------+-------------------+
        |                   |                   |
        v                   v                   v
+--------------+    +--------------+    +--------------+
| PostgreSQL   |    |    Redis     |    |Elasticsearch |
|   (Data)     |    |(Cache/Broker)|    |   (Search)   |
+--------------+    +--------------+    +--------------+
        |
        v
+------------------------------------------------------+
|              External Services                        |
|  - Razorpay (Payments)                               |
|  - Groq AI (LLM)                                     |
|  - AWS S3 (Storage)                                  |
|  - SMTP (Email)                                      |
+------------------------------------------------------+
``````

---

## 2. Service Connections

``````
Frontend (React)
    | HTTPS
Nginx
    |-> Gunicorn (Django REST API)
    |   |-> PostgreSQL (Database)
    |   |-> Redis (Cache)
    |   |-> Elasticsearch (Search)
    |   |-> Razorpay API (Payments)
    |   |-> Groq API (AI)
    |   +-> S3 (File Storage)
    |
    +-> Daphne (WebSocket)
        +-> Redis (Channel Layer)

Celery Workers
    |-> Redis (Broker)
    |-> PostgreSQL (Database)
    |-> Razorpay (Payouts)
    +-> SMTP (Emails)
``````

---

## 3. Key Request/Response Flows

### 3.1 User Login

``````
1. POST /api/auth/login/ {email, password}
2. Django validates credentials
3. Generate JWT tokens (access + refresh)
4. Return {access, refresh, user}
5. Frontend stores tokens in localStorage
``````

### 3.2 Project Search

``````
1. GET /api/projects/search/?q=react&skills=javascript
2. Django queries Elasticsearch
3. Elasticsearch returns matching projects
4. Django enriches with PostgreSQL data
5. Return paginated results
``````

### 3.3 Real-time Chat

``````
1. WebSocket: ws://domain/ws/chat/123/?token=jwt
2. Daphne authenticates JWT
3. Join Redis channel group chat_123
4. User sends message -> Save to PostgreSQL
5. Broadcast via Redis to all group members
6. Other users receive message instantly
``````

### 3.4 Payment Flow

``````
1. Client: POST /api/payments/escrow/ {contract_id}
2. Django -> Razorpay: Create order
3. Frontend: Razorpay modal -> User pays
4. Frontend: POST /api/payments/verify/ {payment_id, signature}
5. Django: Verify HMAC signature
6. Update Payment status: PENDING -> ESCROWED
7. [Work completed]
8. Client: POST /api/payments/release/ {contract_id}
9. Celery task -> RazorpayX payout to freelancer
10. Webhook confirms -> Update status: RELEASED
``````

### 3.5 AI Worklog Generation

``````
1. POST /api/worklogs/ai-chat/message/ {message, contract_id}
2. Django -> Groq API (Llama 3.3 70B via LangChain)
3. AI responds with questions or generates JSON report
4. Return {message, report_ready, report_data}
5. If report ready -> POST /api/worklogs/deliverables/
6. Create Deliverable with AI-generated summary
``````

---

## 4. Authentication Flow

``````
JWT Authentication

LOGIN:
  POST /api/auth/login/ {email, password}
    |
  Validate credentials
    |
  Generate JWT tokens:
    - Access token (60 min)
    - Refresh token (7 days)
    |
  Return {access, refresh, user}

AUTHENTICATED REQUEST:
  GET /api/projects/
  Header: Authorization: Bearer <access_token>
    |
  Extract & verify JWT signature
    |
  Check expiration & blacklist
    |
  Load user from database
    |
  Check permissions
    |
  Execute request

TOKEN REFRESH:
  POST /api/auth/refresh/ {refresh}
    |
  Verify refresh token
    |
  Generate new access + refresh tokens
    |
  Blacklist old refresh token
    |
  Return new tokens

LOGOUT:
  POST /api/auth/logout/ {refresh}
    |
  Add refresh token to Redis blacklist
    |
  Return 204 No Content
``````

---

## 5. Background Job Flow (Celery)

``````
Celery Task Architecture

Django App
    |
task.delay(args) -> Redis Broker -> Task Queue
                                      |
                              Celery Workers (1-N)
                                      |
                              Execute task
                                      |
                              Redis Result Backend
``````

### Task Types

**1. Email Notifications**
``````
Trigger: User action (bid accepted, payment released)
Task: send_email_notification.delay(user_id, template)
Execution: Render template -> Send via SMTP
Retry: 3 attempts with exponential backoff
``````

**2. Payment Payouts**
``````
Trigger: Client releases payment
Task: razorpay_transfer_to_freelancer_task.delay(payment_id, amount)
Execution: Call RazorpayX API -> Update payment status
Priority: High
Retry: 5 attempts
``````

**3. Weekly Reports**
``````
Trigger: Celery Beat (Monday 9 AM)
Task: generate_weekly_reports.delay()
Execution: For each contract -> Call Groq AI -> Create report
``````

**4. Webhook Processing**
``````
Trigger: Razorpay webhook received
Task: process_razorpay_webhook_task.delay(event_id, payload)
Execution: Parse event -> Update payment -> Send notifications
Idempotency: Check PaymentEvent table
``````

### Periodic Tasks (Celery Beat)

| Task | Schedule | Purpose |
|------|----------|---------|
| generate_weekly_reports | Monday 9 AM | AI weekly summaries |
| send_notification_digest | Daily 8 AM | Email digests |
| cleanup_old_notifications | Daily 2 AM | Delete old notifications |
| sync_razorpay_payouts | Every 15 min | Check payout status |

---

## 6. External Integrations

### 6.1 Razorpay (Payments)

**Purpose**: Payment gateway + payouts

**APIs Used**:
- Orders API: Create escrow orders
- Payments API: Process payments
- Payouts API (RazorpayX): Transfer to freelancers
- Webhooks: Payment status updates

**Flow**:
``````
Django -> Razorpay Orders API -> Create order
Frontend -> Razorpay Checkout -> User pays
Razorpay -> Webhook -> Django verifies signature
Django -> RazorpayX Payouts API -> Transfer to freelancer
``````

**Security**:
- API Key authentication (Basic Auth)
- Webhook HMAC SHA256 signature verification
- Idempotency via PaymentEvent model

### 6.2 Groq AI (LLM)

**Purpose**: AI worklog generation

**Model**: Llama 3.3 70B Versatile

**Framework**: LangChain + LangGraph (stateful conversations)

**Flow**:
``````
Django -> LangGraph workflow -> Groq API
    |
AI generates response or JSON report
    |
LangSmith tracing (monitoring)
``````

**Fallback**: Direct Groq API if LangGraph fails

### 6.3 AWS S3 (Storage)

**Purpose**: File storage

**Buckets**:
- freelanceflow-static-prod: CSS, JS, images
- freelanceflow-media-prod: User uploads (screenshots, attachments)
- freelanceflow-backups-prod: Database backups

**Flow**:
``````
User uploads file -> Django validates
    |
Upload to S3 via boto3
    |
Store URL in PostgreSQL
    |
Generate signed URL (for private files)
``````

### 6.4 Email (SMTP)

**Purpose**: Transactional emails

**Provider**: AWS SES or Gmail SMTP

**Types**: Welcome, bid notifications, payment confirmations, digests

**Flow**:
``````
Event triggers -> Queue Celery task
    |
Celery worker renders template
    |
Send via SMTP
    |
Retry 3 times on failure
``````

---

## 7. Security Decisions

### 7.1 JWT over Sessions

**Why**:
- Stateless (scalable across servers)
- Mobile-friendly
- Includes user info (reduces DB queries)

**Implementation**:
- Access: 60 min, Refresh: 7 days
- Token rotation on refresh
- Redis blacklist for logout

### 7.2 Escrow Payments

**Why**:
- Protects both parties
- Reduces disputes
- Platform controls release

**Implementation**:
- Razorpay holds funds
- Client approval triggers release
- 10% platform cut deducted automatically

### 7.3 Rate Limiting

**Why**:
- Prevents abuse
- DDoS protection

**Limits**:
- Anonymous: 100/hour
- Authenticated: 1000/hour
- Auth endpoints: 5/min

### 7.4 Password Security

**Implementation**: bcrypt with 12 rounds (Django default)

### 7.5 Brute Force Protection

**Implementation**: Django Axes - 5 failed attempts -> 5 min lockout

### 7.6 SQL Injection Prevention

**Implementation**: Django ORM only (automatic parameterization)

### 7.7 XSS Prevention

**Implementation**: Django template auto-escaping + CSP headers

---

## 8. Why Modular Monolith over Microservices

### Decision: Modular Monolith

**Structure**:
``````
FreelanceFlow (Single Codebase)
|-- apps/users
|-- apps/projects
|-- apps/bidding
|-- apps/payments
|-- apps/worklogs
|-- apps/messaging
|-- apps/notifications
+-- apps/search
``````

### Advantages

| Benefit | Explanation |
|---------|-------------|
| **Simpler Deployment** | Single codebase, one pipeline, easier rollbacks |
| **Faster Development** | Shared code, no API contracts, easier refactoring |
| **Lower Complexity** | One server to monitor, simpler debugging |
| **Cost Effective** | Single server vs multiple, lower infrastructure costs |
| **Data Consistency** | ACID transactions, no eventual consistency issues |
| **Performance** | No network latency, in-process function calls |

### Trade-offs

| Limitation | Mitigation |
|------------|------------|
| Cannot scale modules independently | Horizontal scaling with load balancer |
| Single language/framework | Django is versatile enough |
| Shared codebase coordination | Clear module boundaries |
| One module crash affects all | Proper error handling, monitoring |

### When to Consider Microservices

**Future triggers**:
- Team size > 50 developers
- Need to scale specific modules independently
- Different SLAs for different features
- Polyglot requirements

**Current scale**: 1-10 developers -> Modular monolith is optimal

### Module Communication

**Direct function call (tight coupling)**:
``````python
from apps.payments.services import create_escrow
payment = create_escrow(contract, client)
``````

**Django signals (loose coupling)**:
``````python
from django.dispatch import Signal
bid_accepted = Signal()

# Sender
bid_accepted.send(sender=Bid, contract=contract)

# Receiver
@receiver(bid_accepted)
def create_escrow_on_bid_accepted(sender, contract, **kwargs):
    create_escrow(contract, contract.client)
``````

---

**Document Status**: Complete  
**Last Updated**: May 1, 2026