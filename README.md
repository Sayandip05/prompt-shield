# FreelanceFlow

> A production-ready full-stack freelance marketplace built with Django and React, featuring secure escrow payments, real-time chat, and AI-powered worklogs.

## Overview

FreelanceFlow is a two-sided freelance marketplace platform where:
- **Clients** post projects, receive bids, hire freelancers, and manage payments
- **Freelancers** browse projects, submit bids, track work, and receive payments
- **Platform** holds funds in escrow, takes a cut on release, and uses AI to auto-generate weekly progress reports

### Key Features

- **Role-Based Dashboards**: Separate dashboards for Clients and Freelancers
- **AI-Powered Worklogs**: Auto-generated weekly reports using Groq API + WeasyPrint PDFs
- **Secure Escrow Payments**: Razorpay integration with locked funds and safe releases
- **Real-Time Messaging**: WebSocket chat via Django Channels
- **Full-Text Search**: Elasticsearch-powered project discovery
- **Two-Factor Authentication**: TOTP-based 2FA for enhanced security
- **Payment Milestones**: Break payments into stages for better project management
- **Worklog Approval**: Client approval workflow for freelancer work
- **Counter-Offers**: Negotiate bid amounts between clients and freelancers
- **Activity Logging**: Complete audit trail for security and compliance

### Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Django 4.2, Django REST Framework |
| Database | PostgreSQL |
| Cache & Broker | Redis |
| Async Tasks | Celery + Django Celery Beat |
| WebSocket | Daphne + Channels |
| Search | Elasticsearch |
| AI | Groq API (Llama 3.3 70B) |
| PDF | WeasyPrint |
| Payments | Razorpay |
| Monitoring | LangSmith |
| Frontend | React 18 (Vite), Tailwind CSS |
| Architecture | Layered Modular Monolith |

---

## Quick Start (Local Development)

### Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.12+ |
| Node.js | 18+ |
| PostgreSQL | 14+ |
| Redis | 7+ |

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/FreelanceFlow.git
cd FreelanceFlow
```

### 2. Backend Setup

#### Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

#### Install Dependencies

```bash
pip install -r requirements/local.txt
```

#### Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Django (Required)
SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/freelanceflow

# Redis
REDIS_URL=redis://localhost:6379/0
```

#### Run Migrations

```bash
python manage.py migrate
```

#### Start Backend Services

```bash
# Terminal 1: Django server
python manage.py runserver

# Terminal 2: Celery worker
celery -A config worker -l info

# Terminal 3: Celery beat (scheduler)
celery -A config beat -l info
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### 4. Access the Application

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Django API | http://localhost:8000 |
| Admin Panel | http://localhost:8000/admin/ |

---

## Seed Data (Create Test Users)

```bash
python manage.py shell
```

```python
from django.contrib.auth import get_user_model
from apps.projects.models import Project, ProjectSkill
from apps.bidding.models import Bid
from apps.users.models import FreelancerProfile, ClientProfile

User = get_user_model()

# Create Client
client = User.objects.create_user(
    email='client@example.com',
    password='testpass123',
    first_name='John',
    last_name='Doe',
    role='CLIENT'
)
ClientProfile.objects.create(user=client, company_name='Acme Inc')

# Create Freelancer
freelancer = User.objects.create_user(
    email='freelancer@example.com',
    password='testpass123',
    first_name='Jane',
    last_name='Smith',
    role='FREELANCER'
)
FreelancerProfile.objects.create(
    user=freelancer,
    skills=['Python', 'Django', 'React'],
    hourly_rate=75.00
)

# Create Sample Project
project = Project.objects.create(
    client=client,
    title='E-commerce Website',
    description='Build a full-stack e-commerce website with Django and React',
    budget=5000.00
)
ProjectSkill.objects.create(project=project, skill_name='Django')
ProjectSkill.objects.create(project=project, skill_name='React')

print("✓ Seed data created!")
print("  Client: client@example.com / testpass123")
print("  Freelancer: freelancer@example.com / testpass123")
```

---

## Environment Variables Reference

### Required Variables

```env
# Django
SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
DATABASE_URL=postgresql://postgres:password@localhost:5432/freelanceflow
REDIS_URL=redis://localhost:6379/0
```

### Optional Variables (Feature Flags)

```env
# Razorpay (Payments)
RAZORPAY_KEY_ID=rzp_test_...
RAZORPAY_KEY_SECRET=...
RAZORPAY_WEBHOOK_SECRET=whsec_...

# Groq (AI Reports)
GROQ_API_KEY=gsk_...

# LangSmith (Monitoring)
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=freelanceflow

# AWS S3 (File Storage)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_STORAGE_BUCKET_NAME=freelanceflow-files
AWS_S3_REGION_NAME=us-east-1

# Elasticsearch (Search)
ELASTICSEARCH_URL=http://localhost:9200

# Email
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password
DEFAULT_FROM_EMAIL=noreply@freelanceflow.com

# Platform Settings
PLATFORM_CUT_PERCENTAGE=10
```

---

## Project Structure

```
FreelanceFlow/
├── apps/                    # Django applications (8 domain apps)
│   ├── users/              # Authentication & profiles
│   ├── projects/           # Project management
│   ├── bidding/            # Bids & contracts
│   ├── payments/           # Stripe escrow & releases
│   ├── worklogs/            # Time tracking & AI reports
│   ├── messaging/           # Real-time chat (WebSocket)
│   ├── notifications/       # In-app notifications
│   └── search/              # Elasticsearch integration
├── config/                  # Django configuration
│   ├── settings/            # base.py, local.py, production.py
│   ├── urls.py             # URL routing
│   ├── asgi.py             # ASGI + Channels
│   └── celery.py           # Celery configuration
├── core/                    # Shared utilities
│   ├── permissions.py       # Custom permissions
│   ├── pagination.py        # Pagination classes
│   ├── exceptions.py       # Error handling
│   └── utils.py            # Utilities
├── frontend/                # React frontend
│   ├── src/
│   │   ├── api/            # API client
│   │   ├── components/     # Reusable UI components
│   │   ├── context/        # React contexts
│   │   ├── hooks/          # Custom hooks
│   │   ├── pages/          # Page components
│   │   ├── routes/         # Route protection
│   │   └── utils/          # Utilities
│   └── package.json
├── deployment/              # Deployment configs
├── docs/                    # Documentation
├── requirements/            # Python dependencies
├── docker-compose.yml       # Local dev stack
├── docker-compose.prod.yml  # Production stack
└── Dockerfile
```

### Architecture Pattern: Layered Modular Monolith

Each app follows the same internal structure:

```
models.py      → What data exists (schema only)
selectors.py   → How to READ data (ORM queries)
services.py    → How to CHANGE data (business logic)
serializers.py → How data looks over API
views.py       → HTTP request/response
urls.py        → URL routing
permissions.py → Who can do what
tasks.py       → Background jobs (Celery)
```

**Golden Rule:** Views never query the database directly. Services never handle HTTP. Models never call external APIs.

---

## API Endpoints Summary

**Total Endpoints: 115+**

| App | Prefix | Key Endpoints |
|-----|---------|---------------|
| Users | `/api/users/` | Register, Login, Profile, 2FA, Activity, Status |
| Projects | `/api/projects/` | CRUD, List, Bookmarks |
| Bidding | `/api/bidding/` | Bids, Accept/Reject, Contracts, Counter-Offers, Retraction, Worklog Approval |
| Payments | `/api/payments/` | Create Escrow, Release, Webhook, Milestones, Disputes |
| Worklogs | `/api/worklogs/` | Logs, Reports, AI Generation |
| Messaging | `/api/messaging/` | Conversations, Messages |
| Notifications | `/api/notifications/` | List, Mark Read |
| Search | `/api/search/` | Project Search, Autocomplete, History |
| Reviews | `/api/reviews/` | Create, List, Respond |

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for complete endpoint list.

---

## Running Tests

```bash
# Run all tests
pytest

# Run specific app
pytest apps/projects/

# Run with coverage
pytest --cov=. --cov-report=html
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and commit
4. Push to branch: `git push origin feature/your-feature`
5. Submit a Pull Request

### Code Style

- **Python**: Follow PEP 8, use Black formatter
- **JavaScript**: Follow ESLint rules

```bash
# Format Python
black .

# Format JS
cd frontend && npm run lint
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [API Documentation](API_DOCUMENTATION.md) | Complete API reference (115+ endpoints) |
| [System Architecture](docs/System_architechture.md) | Technical design, components, data flow |
| [HLD](docs/HLD.md) | High-level overview, APIs, user journeys |
| [LLD](docs/LLD.md) | Database schema, sequence diagrams, errors |
| [Deployment](docs/deployment.md) | Docker, CI/CD, production setup |
| [Commit History](docs/commit-history.md) | Development history |
| [Implementation Memory](memory.md) | Complete implementation details |
| [Feature Status](missing.md) | All features status (100% complete) |

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## The Platform Flow

```
1. Client posts project → Title, description, budget, deadline
2. Freelancers browse and submit bids (amount + proposal)
3. Client can make counter-offer → Negotiate price and timeline
4. Client accepts bid → Contract created (with optional milestones)
5. Client funds escrow → Money held securely via Razorpay
6. Freelancer submits daily work logs
7. Client approves worklogs → Approval workflow
8. Weekly: AI generates progress report → PDF created using Groq
9. Client approves work → Contract marked complete
10. Platform takes cut (10%) → Releases remainder to freelancer
11. Final delivery proof PDF generated
12. Both parties can leave reviews → Rating system
```

---

## Production Readiness: 85%

### ✅ Complete
- 50+ database models
- 115+ API endpoints
- Complete service layer (24 files)
- Essential extended features (8 features)
- Security fixes (9/9)
- Email notifications
- AI-powered reports
- Payment processing

### 🔄 Optional (Can use Django Admin)
- 14 nice-to-have features available via admin panel
- Advanced analytics
- OAuth integration

See [missing.md](missing.md) for complete feature status.
