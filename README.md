# FreelanceFlow

**AI-powered freelance marketplace connecting clients with skilled freelancers**

---

## 🎯 Problem It Solves

Traditional freelance platforms lack transparency in work tracking and payment security. FreelanceFlow solves this with AI-powered worklog generation, escrow-based payments, and real-time collaboration tools—ensuring trust, accountability, and seamless project delivery.

---

## 🚀 Live Demo

- **Frontend**: [https://freelanceflow.vercel.app](https://freelanceflow.vercel.app) *(placeholder)*
- **Backend API**: [https://api.freelanceflow.com](https://api.freelanceflow.com) *(placeholder)*
- **API Docs**: [https://api.freelanceflow.com/swagger](https://api.freelanceflow.com/swagger) *(placeholder)*

---

## 📸 Screenshot

![FreelanceFlow Dashboard](./docs/assets/screenshot.png) *(Add your screenshot here)*

---

## 🛠️ Tech Stack

**Backend**
- Django 5.0 (Modular Monolith)
- PostgreSQL (Supabase)
- Redis (Upstash)
- Celery + Celery Beat
- Django Channels (WebSocket)
- Elasticsearch

**Frontend**
- React 18
- Vite
- TailwindCSS
- React Query

**AI & Integrations**
- Groq API (LLM)
- LangChain + LangGraph
- Razorpay (Payments)
- AWS S3 (Storage)
- WeasyPrint (PDF)

**DevOps**
- Docker + Docker Compose
- GitHub Actions (CI/CD)
- Nginx
- AWS EC2 (Production)

---

## ✨ Key Features

- 🤖 **AI Worklog Generation** - Chat with AI to auto-generate daily work reports
- 💰 **Escrow Payments** - Secure milestone-based payments with 10% platform fee
- 🔍 **Smart Search** - Elasticsearch-powered project and freelancer discovery
- 💬 **Real-time Chat** - WebSocket messaging per contract
- 📊 **Weekly Reports** - AI-generated project summaries
- 🔐 **2FA & RBAC** - Two-factor auth with role-based access control

---

## 🚀 Quick Local Setup

### Backend

```bash
# Clone repository
git clone https://github.com/yourusername/freelanceflow.git
cd freelanceflow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your credentials

# Run migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Run development server
python manage.py runserver
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Setup environment
cp .env.example .env
# Edit .env with backend API URL

# Run development server
npm run dev
```

### Docker (Full Stack)

```bash
# Build and run all services
docker-compose up --build

# Backend: http://localhost:8000
# Frontend: http://localhost:5173
# PostgreSQL: localhost:5432
# Redis: localhost:6379
```

---

## 📐 Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   React     │─────▶│    Django    │─────▶│ PostgreSQL  │
│  Frontend   │      │   Backend    │      │ (Supabase)  │
└─────────────┘      └──────────────┘      └─────────────┘
                            │
                ┌───────────┼───────────┐
                │           │           │
         ┌──────▼────┐ ┌───▼────┐ ┌───▼─────┐
         │   Redis   │ │  Groq  │ │   S3    │
         │ (Upstash) │ │   AI   │ │ Storage │
         └───────────┘ └────────┘ └─────────┘
```

**Architecture Type**: Modular Monolith (8 Django apps)

---

## 📚 Documentation

For detailed documentation, see the `/docs` folder:

- **[HLD.md](./docs/HLD.md)** - High-Level Design (system overview, architecture, tech choices)
- **[LLD.md](./docs/LLD.md)** - Low-Level Design (database schema, API endpoints, algorithms)
- **[System_Architecture.md](./docs/System_Architecture.md)** - Service connections, flows, integrations
- **[deployment.md](./docs/deployment.md)** - Deployment guide (Render, Railway, AWS EC2)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Your Name**

- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

**Built with ❤️ using Django, React, and AI**
