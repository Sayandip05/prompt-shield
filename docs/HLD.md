# FreelanceFlow - High-Level Design (HLD)

**Version:** 4.0.0  
**Last Updated:** May 1, 2026  
**Status:** Production Ready

---

## 1. Problem Statement

**The Problem**: Freelance work suffers from trust issues, payment disputes, and lack of transparency. Clients worry about paying upfront without guaranteed delivery. Freelancers fear non-payment after completing work. Both parties struggle with unclear work documentation and dispute resolution.

**Who Are the Users**:
- **Clients**: Businesses and individuals who need to hire freelancers for projects
- **Freelancers**: Independent professionals offering services (development, design, writing, etc.)

**Pain Points Solved**:
- ✅ **Payment Security**: Escrow-based payments protect both parties
- ✅ **Work Transparency**: AI-powered worklog documentation provides clear audit trails
- ✅ **Dispute Prevention**: Structured deliverable approval workflow reduces conflicts
- ✅ **Communication**: Real-time chat keeps everyone aligned
- ✅ **Trust Building**: Rating system and delivery proofs establish credibility

---

## 2. System Overview

FreelanceFlow is a **full-stack freelance marketplace platform** that connects clients with freelancers through a secure, transparent workflow powered by AI-assisted work documentation and escrow-based payments.

**What the System Does**:
1. **Project Posting**: Clients post projects with budget, skills, and deadlines
2. **Bidding**: Freelancers submit proposals with their rates and approach
3. **Contract Management**: Automated contract creation when bids are accepted
4. **Escrow Payments**: Secure payment holding until work is approved
5. **AI Work Documentation**: Groq-powered chat converts conversations into structured worklogs
6. **Real-time Communication**: WebSocket-based messaging between parties
7. **Deliverable Approval**: Structured review and approval workflow
8. **Automated Payouts**: Instant transfers to freelancers upon approval
9. **Proof of Delivery**: Tamper-evident delivery certificates

---

## 3. Major Components & Responsibilities

### 3.1 Backend Services (Django + DRF)

| Component | Responsibility | Key Features |
|-----------|---------------|--------------|
| **User Management** | Authentication, profiles, roles | JWT auth, 2FA, activity logging, online status |
| **Project Service** | Project CRUD, search, categorization | Elasticsearch integration, skill matching, bookmarks |
| **Bidding Service** | Bid submission, acceptance, contracts | Counter-offers, retractions, amendments |
| **Payment Service** | Escrow, payouts, disputes | Razorpay integration, signature verification, refunds |
| **Worklog Service** | Daily logs, AI chat, deliverables | Groq AI integration, approval workflow, proof generation |
| **Messaging Service** | Real-time chat | WebSocket (Django Channels), typing indicators |
| **Notification Service** | In-app, email, push | Event-driven, digest emails, announcements |
| **Search Service** | Full-text search | Elasticsearch, filters, autocomplete |

### 3.2 Frontend Application (React + Vite)

| Component | Responsibility |
|-----------|---------------|
| **Authentication** | Login, registration, role selection |
| **Client Dashboard** | Project management, bid review, payments |
| **Freelancer Dashboard** | Project browsing, bid submission, work tracking |
| **AI Chat Interface** | Natural language worklog creation |
| **Real-time Chat** | WebSocket messaging with file attachments |
| **Payment UI** | Razorpay checkout, escrow management |

### 3.3 Infrastructure Components

| Component | Purpose |
|-----------|---------|
| **PostgreSQL** | Primary database (ACID compliance) |
| **Redis** | Cache, session store, Celery broker, Channels layer |
| **Elasticsearch** | Full-text search engine |
| **Celery** | Async task processing (emails, payouts, reports) |
| **Nginx** | Reverse proxy, SSL termination, static files |
| **AWS S3** | Media file storage (screenshots, attachments) |
| **Razorpay** | Payment gateway and payout processing |
| **Groq AI** | LLM for worklog generation (Llama 3.3 70B) |

---

## 4. Data Flow - Key User Journeys

### 4.1 Client Hiring Flow

```
Client registers → Creates project → Publishes
    ↓
Freelancers submit bids
    ↓
Client reviews bids → Accepts one bid
    ↓
System creates Contract → Notifies freelancer
    ↓
Client creates Escrow payment
    ↓
Razorpay order created → Client pays → Funds held in escrow
    ↓
Freelancer works → Submits deliverables
    ↓
Client reviews → Approves work
    ↓
System releases payment → Transfers to freelancer (minus 10% platform cut)
    ↓
Both parties leave reviews → Project completed
```

### 4.2 Freelancer Work Flow

```
Freelancer registers → Sets up profile
    ↓
Browses projects (Elasticsearch search)
    ↓
Submits bid with cover letter
    ↓
Bid accepted → Contract created → Notified
    ↓
Opens AI chat → Describes work in natural language
    ↓
AI (Groq) asks clarifying questions → Generates structured report
    ↓
Freelancer reviews AI report → Submits as deliverable
    ↓
Client approves → WorkLog created → Payment released
    ↓
Freelancer receives funds in Razorpay account → Withdraws to bank
```

### 4.3 Payment Flow (Detailed)

```
Client clicks "Create Escrow"
    ↓
Backend: create_escrow() → Razorpay order created
    ↓
Frontend: Razorpay modal opens → Client enters card details
    ↓
Razorpay processes payment → Returns payment_id + signature
    ↓
Frontend: verify_payment() → Backend verifies HMAC signature
    ↓
Backend: Payment status PENDING → ESCROWED
    ↓
Funds held in Razorpay escrow account
    ↓
[Work happens, deliverable approved]
    ↓
Client clicks "Release Payment"
    ↓
Backend: release_payment() → Calculates platform cut (10%)
    ↓
Celery task: razorpay_transfer_to_freelancer_task()
    ↓
RazorpayX payout initiated → Webhook confirms
    ↓
Payment status: ESCROWED → RELEASED
    ↓
Freelancer receives funds in bank account
```

### 4.4 AI Worklog Generation Flow

```
Freelancer opens AI chat interface
    ↓
Types: "I built a React authentication component today"
    ↓
Frontend: POST /api/worklogs/ai-chat/message/
    ↓
Backend: groq_service.chat() → LangChain + LangGraph workflow
    ↓
AI (Groq Llama 3.3 70B): "What libraries did you use? How long did it take?"
    ↓
Freelancer: "Used JWT and Axios, took about 4 hours"
    ↓
AI continues conversation → Extracts details
    ↓
Freelancer: "Generate report"
    ↓
AI generates JSON:
{
  "title": "React Authentication Component",
  "description": "Built authentication component with JWT...",
  "hours_worked": 4.0,
  "tasks_completed": ["JWT integration", "Axios setup", "Login form"],
  "technologies_used": ["React", "JWT", "Axios"]
}
    ↓
Frontend displays structured report → Freelancer reviews
    ↓
Freelancer clicks "Submit Deliverable"
    ↓
Backend: create_deliverable() → Stores AI transcript + report
    ↓
Client receives notification → Reviews deliverable
    ↓
Client approves → WorkLog created with AI summary
```

---

## 5. Technology Stack & Rationale

### 5.1 Backend Stack

| Technology | Why Chosen |
|------------|------------|
| **Django 4.x** | Mature framework, batteries-included, excellent ORM, security features |
| **Django REST Framework** | Industry standard for REST APIs, serializers, authentication |
| **PostgreSQL 15** | ACID compliance, JSON support, full-text search, reliability |
| **Redis 7** | Multi-purpose (cache, sessions, Celery broker, Channels layer) |
| **Celery** | Distributed task queue, retry logic, scheduling, monitoring |
| **Django Channels** | Native WebSocket support, Redis integration, scalable |
| **Elasticsearch 8** | Full-text search, filtering, highlighting, fast queries |
| **Razorpay** | India-focused, escrow support, webhook reliability, instant payouts |
| **Groq API** | Fast LLM inference (Llama 3.3 70B), cost-effective, reliable |
| **LangChain + LangGraph** | Stateful AI conversations, structured output, monitoring |
| **LangSmith** | AI tracing, debugging, performance monitoring |

### 5.2 Frontend Stack

| Technology | Why Chosen |
|------------|------------|
| **React 18** | Component-based, large ecosystem, excellent developer experience |
| **Vite** | Fast build tool, HMR, modern development experience |
| **Tailwind CSS** | Utility-first, responsive design, rapid development |
| **Axios** | HTTP client with interceptors, request/response transformation |
| **React Router v6** | Client-side routing, nested routes, protected routes |
| **Context API** | State management without external dependencies |
| **WebSocket Client** | Real-time messaging, reconnection logic |

### 5.3 Infrastructure

| Technology | Why Chosen |
|------------|------------|
| **Docker** | Containerization, consistent environments, easy deployment |
| **Nginx** | Reverse proxy, SSL termination, static file serving, load balancing |
| **Gunicorn** | WSGI server, worker management, graceful shutdown |
| **AWS S3** | Scalable storage, CDN integration, reliability |
| **AWS EC2** | Flexible compute, auto-scaling, cost-effective |
| **CloudWatch** | Logging, monitoring, alerting for AWS services |

---

## 6. Architecture Diagram (C4 Level 1-2)

### Level 1: System Context

```
┌─────────────────────────────────────────────────────────────────┐
│                         Internet Users                           │
│                                                                  │
│         ┌──────────────┐              ┌──────────────┐          │
│         │   Clients    │              │ Freelancers  │          │
│         │ (Hire work)  │              │ (Do work)    │          │
│         └──────┬───────┘              └──────┬───────┘          │
│                │                             │                   │
│                └─────────────┬───────────────┘                   │
│                              │                                   │
└──────────────────────────────┼───────────────────────────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │                              │
                │      FreelanceFlow           │
                │   (Marketplace Platform)     │
                │                              │
                │  - Project Management        │
                │  - Escrow Payments           │
                │  - AI Work Documentation     │
                │  - Real-time Chat            │
                │                              │
                └──────────────┬───────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐    ┌──────────────────┐    ┌──────────────┐
│   Razorpay    │    │    Groq AI       │    │   AWS S3     │
│  (Payments)   │    │  (LLM Service)   │    │  (Storage)   │
└───────────────┘    └──────────────────┘    └──────────────┘
```

### Level 2: Container Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FreelanceFlow System                         │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Frontend (React SPA)                         │ │
│  │  - Client Dashboard    - Freelancer Dashboard                  │ │
│  │  - AI Chat UI          - Real-time Messaging                   │ │
│  │  - Payment Checkout    - Project Search                        │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                  │                                   │
│                                  │ HTTPS/WSS                         │
│                                  ▼                                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Nginx (Reverse Proxy)                        │ │
│  │  - SSL Termination     - Rate Limiting                         │ │
│  │  - Static Files        - Load Balancing                        │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                  │                                   │
│         ┌────────────────────────┼────────────────────────┐         │
│         │                        │                        │         │
│         ▼                        ▼                        ▼         │
│  ┌─────────────┐        ┌──────────────┐        ┌──────────────┐  │
│  │  Gunicorn   │        │   Daphne     │        │    Celery    │  │
│  │  (WSGI)     │        │ (WebSocket)  │        │   Workers    │  │
│  │             │        │              │        │              │  │
│  │ Django REST │        │   Channels   │        │ Async Tasks  │  │
│  │     API     │        │   Consumer   │        │              │  │
│  └─────────────┘        └──────────────┘        └──────────────┘  │
│         │                        │                        │         │
│         └────────────────────────┼────────────────────────┘         │
│                                  │                                   │
│  ┌───────────────────────────────┼──────────────────────────────┐  │
│  │                    Data & Service Layer                       │  │
│  │                                                               │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │  │
│  │  │ PostgreSQL   │  │    Redis     │  │Elasticsearch │      │  │
│  │  │  (Primary)   │  │ (Cache/Msg)  │  │   (Search)   │      │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘      │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐        ┌──────────────┐        ┌──────────────┐
│   Razorpay    │        │   Groq AI    │        │   AWS S3     │
│               │        │              │        │              │
│ - Orders      │        │ - LangChain  │        │ - Media      │
│ - Payments    │        │ - LangGraph  │        │ - Static     │
│ - Payouts     │        │ - LangSmith  │        │ - Backups    │
│ - Webhooks    │        │              │        │              │
└───────────────┘        └──────────────┘        └──────────────┘
```

---

## 7. Database Schema (Core Models)

### 7.1 Entity Relationships

```
User (1) ──→ (1) FreelancerProfile
User (1) ──→ (1) ClientProfile
User (1) ──→ (M) Project (as Client)
User (1) ──→ (M) Bid (as Freelancer)

Project (1) ──→ (M) Bid
Bid (1) ──→ (1) Contract (when accepted)

Contract (1) ──→ (1) Payment
Contract (1) ──→ (M) WorkLog
Contract (1) ──→ (M) Deliverable
Contract (1) ──→ (1) Conversation

Payment (1) ──→ (1) Escrow
Payment (1) ──→ (M) PlatformEarning

Conversation (1) ──→ (M) Message
```

### 7.2 Core Models

| Model | Key Fields | Purpose |
|-------|-----------|---------|
| **User** | email, role (CLIENT/FREELANCER), is_active | Authentication & authorization |
| **Project** | title, description, budget, status, deadline | Client job postings |
| **Bid** | amount, cover_letter, status, created_at | Freelancer proposals |
| **Contract** | agreed_amount, start_date, end_date, is_active | Accepted work agreement |
| **Payment** | total_amount, status, razorpay_order_id | Payment tracking |
| **Escrow** | held_amount, released_at, refund_amount | Funds held securely |
| **WorkLog** | date, hours_worked, description, screenshot_url | Daily work documentation |
| **Deliverable** | ai_chat_transcript, ai_generated_report, status | AI-powered work submission |
| **Message** | content, attachments, is_read, timestamp | Real-time chat messages |
| **Notification** | title, body, type, is_read | User notifications |

---

## 8. API Architecture

### 8.1 REST API Endpoints (115+ total)

| Module | Endpoints | Authentication |
|--------|-----------|----------------|
| **Authentication** | `/api/auth/register/`, `/api/auth/login/`, `/api/auth/refresh/` | Public + JWT |
| **Projects** | `/api/projects/`, `/api/projects/{id}/`, `/api/projects/search/` | JWT Required |
| **Bidding** | `/api/bids/`, `/api/bids/{id}/accept/`, `/api/bids/{id}/reject/` | JWT Required |
| **Payments** | `/api/payments/escrow/`, `/api/payments/release/`, `/api/payments/webhook/` | JWT + Signature |
| **Worklogs** | `/api/worklogs/`, `/api/worklogs/deliverables/`, `/api/worklogs/ai-chat/` | JWT Required |
| **Messaging** | `/api/messages/conversations/`, `/api/messages/` | JWT Required |
| **Notifications** | `/api/notifications/`, `/api/notifications/{id}/mark-read/` | JWT Required |

### 8.2 WebSocket Endpoints

| Endpoint | Purpose | Authentication |
|----------|---------|----------------|
| `/ws/chat/{contract_id}/` | Real-time messaging | JWT via query param |

---

## 9. Security Architecture

### 9.1 Authentication & Authorization

- **JWT Tokens**: Access (60 min) + Refresh (7 days)
- **Token Rotation**: New refresh token on each refresh
- **Token Blacklist**: Invalidated tokens stored in Redis
- **Role-Based Access**: CLIENT vs FREELANCER permissions
- **Object-Level Permissions**: IsProjectOwner, IsContractParticipant

### 9.2 Payment Security

- **HMAC Signature Verification**: All Razorpay webhooks verified
- **Idempotency**: PaymentEvent model prevents duplicate processing
- **Escrow Protection**: Funds never transferred directly
- **Atomic Transactions**: Database locks prevent race conditions

### 9.3 API Security

- **Rate Limiting**: 100/hour (anon), 1000/hour (authenticated)
- **CORS**: Whitelist frontend URL only
- **CSRF Protection**: Tokens for state-changing operations
- **Brute Force Protection**: Django Axes (5 attempts → 5 min lockout)
- **Input Validation**: DRF serializers + custom validators

---

## 10. AI Integration Architecture

### 10.1 Groq AI Service

**Model**: Llama 3.3 70B Versatile  
**Framework**: LangChain + LangGraph  
**Monitoring**: LangSmith tracing

**Workflow**:
```
User Message
    ↓
LangGraph State Machine
    ↓
process_message_node → Calls Groq LLM
    ↓
check_intent_node → Determines if report ready
    ↓
Conditional routing (continue/generate_report/end)
    ↓
Extract JSON report from AI response
    ↓
Return structured data to frontend
```

**Features**:
- Conversational worklog creation
- Automatic hour estimation
- Technology extraction
- Task breakdown
- Professional report generation
- Fallback to direct API if LangGraph unavailable

---

## 11. Scalability & Performance

### 11.1 Caching Strategy

- **Redis Cache**: User profiles, project details (5 min TTL)
- **Static Files**: Nginx caching (1 year)
- **Database Query Optimization**: select_related, prefetch_related
- **Elasticsearch**: Offload search queries from PostgreSQL

### 11.2 Async Processing

- **Celery Tasks**: Email sending, payment processing, report generation
- **Task Retry**: Exponential backoff for failed tasks
- **Task Monitoring**: Flower dashboard

### 11.3 Load Balancing

- **Nginx**: Round-robin across multiple Gunicorn workers
- **AWS ALB**: Multi-AZ deployment across EC2 instances
- **Database**: RDS Multi-AZ with read replicas

---

## 12. Scope Definition

### ✅ In Scope (Implemented)

**Core Features**:
- User registration and authentication (JWT)
- Project posting and management
- Bid submission and acceptance
- Contract creation and management
- Escrow-based payments (Razorpay)
- Payment verification and webhooks
- Automated payouts to freelancers
- Real-time chat (WebSocket)
- AI-powered worklog generation (Groq)
- Deliverable approval workflow
- Weekly report generation
- Delivery proof documents
- In-app and email notifications
- Full-text search (Elasticsearch)
- 2FA authentication
- Activity logging
- Payment disputes
- Contract amendments
- Milestone-based payments
- Multi-currency support

**Infrastructure**:
- Docker containerization
- Nginx reverse proxy
- Celery task queue
- Redis caching
- AWS S3 storage
- Graceful shutdown
- Health checks
- Rate limiting
- CORS configuration

### 🚧 Partially Implemented

- Admin dashboard (basic Django admin)
- Analytics (logging only, no visualization)
- Review system (models exist, limited UI)

### ❌ Out of Scope (Future)

- Video call integration
- Mobile apps (iOS/Android)
- Blockchain proof of work
- Advanced analytics dashboard
- Automated dispute resolution (AI mediation)
- Subscription tiers (PRO features)
- Referral program
- External tool integrations (GitHub, Figma)
- Automated invoice generation
- Time tracking desktop app
- Skill verification/certifications
- Portfolio showcase
- Automated matching algorithm
- Escrow insurance
- Multi-language support
- Advanced fraud detection

---

## 13. Deployment Architecture

### 13.1 Production Environment (AWS EC2)

```
Route 53 DNS
    ↓
CloudFront CDN (Static Assets)
    ↓
Application Load Balancer (SSL Termination)
    ↓
EC2 Instances (3x, Multi-AZ)
    - Nginx
    - Gunicorn (Django)
    - Daphne (WebSocket)
    - Celery Workers
    ↓
RDS PostgreSQL (Multi-AZ)
ElastiCache Redis (Cluster Mode)
Elasticsearch Service
S3 Buckets (Media, Static, Backups)
```

### 13.2 Monitoring & Observability

**Planned**:
- Grafana dashboards (application, infrastructure, business metrics)
- Prometheus metrics collection
- Loki log aggregation
- AlertManager notifications
- CloudWatch integration

**Current**:
- LangSmith (AI tracing)
- Sentry (error tracking)
- Nginx access logs
- Django logging

---

## 14. Key Metrics & KPIs

### Business Metrics
- Total projects posted
- Total bids submitted
- Contract acceptance rate
- Average project value
- Platform revenue (10% cut)
- User retention rate

### Technical Metrics
- API response time (p50, p95, p99)
- Error rate (4xx, 5xx)
- Database query performance
- Cache hit ratio
- WebSocket connection stability
- AI response time

### User Experience Metrics
- Time to first bid
- Time to contract acceptance
- Deliverable approval rate
- Payment release time
- User satisfaction score

---

**Document Status**: ✅ Complete  
**Last Review**: May 1, 2026  
**Next Review**: June 1, 2026
