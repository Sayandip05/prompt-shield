# FreelanceFlow

> A production-ready full-stack freelance marketplace built with Django and React, featuring secure escrow payments, real-time chat, and AI-powered worklogs.

## Overview

FreelanceFlow is a comprehensive platform designed for freelancers and clients to interoperate seamlessly. From bidding and contracting to final delivery proof and payment release, FreelanceFlow is built as a robust modular monolith. 

## Key Features

- **Role-Based Dashboards**: Real-time hubs specifically crafted for Clients and Freelancers.
- **AI-Powered Worklogs**: Auto-generated structured weekly work reports using Anthropic API and WeasyPrint PDF backups.
- **Secure Escrow Payments**: Stripe integration ensuring locked-in funds and safe transaction releases.
- **Real-Time Bidding & Messaging**: Handled via Django Channels for instant communication and bid flow.

## Tech Stack

- **Backend**: Django, Django REST Framework, PostgreSQL, Redis, Celery, Daphne (ASGI).
- **Frontend**: React (Vite).
- **Architecture**: Layered Modular Monolith (Views → Services → Selectors → Models).

---
*For a complete map of the project configuration, refer to [`FreelanceFlow_Ultimate_Structure.md`](./FreelanceFlow_Ultimate_Structure.md).*
