# FreelanceFlow - Deployment Guide

**Version:** 4.0.0  
**Last Updated:** May 1, 2026

---

## 1. Deployment Overview

### Initial Deployment (Quick Start)

| Service | Platform | URL/Instance |
|---------|----------|--------------|
| **Frontend** | Vercel | https://freelanceflow.vercel.app |
| **Backend** | Render/Railway | https://freelanceflow-api.onrender.com |
| **Database** | Render PostgreSQL | freelanceflow-db |
| **Redis** | Upstash Redis | freelanceflow-redis |
| **Storage** | AWS S3 | freelanceflow-media-prod |

### Production Deployment (Scalable)

| Service | Platform | Details |
|---------|----------|---------|
| **Frontend** | Vercel/CloudFront | CDN + React SPA |
| **Backend** | AWS EC2 + Nginx | Multi-instance with ALB |
| **Database** | AWS RDS PostgreSQL | Multi-AZ, automated backups |
| **Redis** | AWS ElastiCache | Cluster mode, 3 shards |
| **Search** | AWS Elasticsearch | 3 data nodes |
| **Storage** | AWS S3 + CloudFront | Static + media files |

---

## 2. Environment Variables

### Backend (.env)

**Django Core**:
- `SECRET_KEY`
- `DEBUG`
- `DJANGO_ENV`
- `ALLOWED_HOSTS`
- `DJANGO_SETTINGS_MODULE`

**Database**:
- `DATABASE_URL`
- `DATABASE_NAME`
- `DATABASE_USER`
- `DATABASE_PASSWORD`
- `DATABASE_HOST`
- `DATABASE_PORT`

**Redis**:
- `REDIS_URL`
- `CELERY_BROKER_URL`
- `CELERY_RESULT_BACKEND`

**Payments (Razorpay)**:
- `RAZORPAY_KEY_ID`
- `RAZORPAY_KEY_SECRET`
- `RAZORPAY_WEBHOOK_SECRET`

**Storage (AWS S3)**:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_STORAGE_BUCKET_NAME`
- `AWS_S3_REGION_NAME`
- `AWS_CLOUDFRONT_DOMAIN`

**AI (Groq + LangSmith)**:
- `GROQ_API_KEY`
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT`
- `LANGSMITH_TRACING`

**Search (Elasticsearch)**:
- `ELASTICSEARCH_URL`

**Email (SMTP)**:
- `EMAIL_HOST`
- `EMAIL_PORT`
- `EMAIL_HOST_USER`
- `EMAIL_HOST_PASSWORD`
- `DEFAULT_FROM_EMAIL`

**Frontend & CORS**:
- `FRONTEND_URL`
- `CORS_ALLOWED_ORIGINS`

**Business**:
- `PLATFORM_CUT_PERCENTAGE`

**Monitoring (Optional)**:
- `SENTRY_DSN`

### Frontend (.env)

- `VITE_API_URL`
- `VITE_WS_URL`
- `VITE_RAZORPAY_KEY_ID`

---

## 3. Quick Deploy Steps

### Option A: Render (Recommended for MVP)

**Backend Deployment**:
``````bash
1. Push code to GitHub
2. Go to Render Dashboard
3. New Web Service -> Connect GitHub repo
4. Settings:
   - Build Command: pip install -r requirements/production.txt
   - Start Command: gunicorn config.wsgi:application
   - Add environment variables
5. Create PostgreSQL database (same dashboard)
6. Auto-deploys on git push
``````

**Frontend Deployment**:
``````bash
1. Push code to GitHub
2. Go to Vercel Dashboard
3. Import GitHub repo
4. Settings:
   - Framework: Vite
   - Build Command: npm run build
   - Output Directory: dist
   - Add environment variables
5. Auto-deploys on git push
``````

### Option B: Railway (Alternative)

``````bash
1. Push code to GitHub
2. Railway Dashboard -> New Project
3. Deploy from GitHub repo
4. Add PostgreSQL + Redis from Railway marketplace
5. Set environment variables
6. Auto-deploys on git push
``````

### Option C: AWS EC2 (Production)

See `deployment/AWS_EC2_DEPLOYMENT.md` for complete guide.

---

## 4. CI/CD Pipeline

### Current Setup (GitHub Actions - Planned)

**Trigger**: Push to `main` branch

**Pipeline Steps**:
``````
1. Run Tests
   - pytest (backend)
   - npm test (frontend)

2. Security Scan
   - Dependency vulnerabilities
   - Code quality checks

3. Build Docker Images
   - Backend image
   - Tag with commit SHA

4. Deploy to Staging
   - Update Render/Railway service
   - Run migrations
   - Health check

5. Manual Approval (for production)

6. Deploy to Production
   - Blue-green deployment
   - Health check
   - Rollback on failure

7. Notifications
   - Slack/Discord alert
   - Deployment status
``````

**Configuration**: See `deployment/cicd/github-actions/`

### Auto-Deploy (Current)

**Render/Railway**:
- Watches `main` branch
- Auto-deploys on push
- Runs build command
- Restarts service

**Vercel**:
- Watches `main` branch
- Auto-builds and deploys
- Preview deployments for PRs

---

## 5. Health Check

### Endpoints

**Backend Health**:
``````
GET https://freelanceflow-api.onrender.com/health/

Response:
{
  "status": "healthy",
  "timestamp": "2026-05-01T12:00:00Z",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "elasticsearch": "healthy"
  }
}
``````

**Frontend Health**:
``````
GET https://freelanceflow.vercel.app/

Status: 200 OK (React app loads)
``````

### Monitoring

- **Render**: Built-in metrics dashboard
- **Vercel**: Analytics + Web Vitals
- **Uptime**: UptimeRobot or Pingdom
- **Errors**: Sentry integration

---

## 6. Run Locally

### Backend

``````bash
# 1. Clone repository
git clone https://github.com/yourusername/freelanceflow.git
cd freelanceflow

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements/local.txt

# 4. Setup environment
cp .env.example .env
# Edit .env with your local values

# 5. Run migrations
python manage.py migrate

# 6. Create superuser
python manage.py createsuperuser

# 7. Run development server
python manage.py runserver

# Backend: http://localhost:8000
# Admin: http://localhost:8000/admin/
``````

### Frontend

``````bash
# 1. Navigate to frontend
cd frontend

# 2. Install dependencies
npm install

# 3. Setup environment
cp .env.example .env
# Edit .env with backend URL

# 4. Run development server
npm run dev

# Frontend: http://localhost:3000
``````

### With Docker (Full Stack)

``````bash
# 1. Start all services
docker-compose up -d

# 2. Run migrations
docker-compose exec web python manage.py migrate

# 3. Create superuser
docker-compose exec web python manage.py createsuperuser

# Services:
# - Backend: http://localhost:8000
# - Frontend: http://localhost:3000
# - PostgreSQL: localhost:5432
# - Redis: localhost:6379
# - Elasticsearch: localhost:9200

# 4. Stop services
docker-compose down
``````

---

## 7. Common Deployment Issues

### Issue 1: Database Connection Failed

**Symptoms**: `OperationalError: could not connect to server`

**Fix**:
``````bash
# Check DATABASE_URL format
DATABASE_URL=postgresql://user:password@host:5432/dbname

# Verify database is running
# Render: Check database status in dashboard
# Local: docker-compose ps
``````

### Issue 2: Static Files Not Loading

**Symptoms**: 404 errors for CSS/JS files

**Fix**:
``````bash
# Collect static files
python manage.py collectstatic --noinput

# Check AWS S3 settings
AWS_STORAGE_BUCKET_NAME=your-bucket
AWS_S3_REGION_NAME=us-east-1

# Verify CORS on S3 bucket
``````

### Issue 3: Celery Workers Not Running

**Symptoms**: Background tasks not executing

**Fix**:
``````bash
# Start Celery worker
celery -A config worker -l info

# Start Celery beat (for scheduled tasks)
celery -A config beat -l info

# Check Redis connection
redis-cli ping  # Should return PONG
``````

### Issue 4: WebSocket Connection Failed

**Symptoms**: Real-time chat not working

**Fix**:
``````bash
# Check Daphne is running
daphne -b 0.0.0.0 -p 8001 config.asgi:application

# Verify Redis channel layer
python manage.py shell
>>> from channels.layers import get_channel_layer
>>> channel_layer = get_channel_layer()
>>> await channel_layer.send("test", {"type": "test.message"})

# Check WebSocket URL in frontend
WS_URL=wss://your-domain.com/ws/
``````

### Issue 5: Razorpay Webhook Not Working

**Symptoms**: Payments stuck in PENDING status

**Fix**:
``````bash
# Verify webhook URL in Razorpay dashboard
https://your-domain.com/api/payments/webhook/

# Check webhook secret
RAZORPAY_WEBHOOK_SECRET=your_secret

# Test webhook locally with ngrok
ngrok http 8000
# Use ngrok URL in Razorpay dashboard
``````

### Issue 6: Build Failed on Render/Railway

**Symptoms**: Deployment fails during build

**Fix**:
``````bash
# Check build logs
# Common issues:
# - Missing dependencies in requirements.txt
# - Python version mismatch
# - Environment variables not set

# Specify Python version
# Create runtime.txt:
python-3.11.0

# Check requirements file
pip freeze > requirements/production.txt
``````

### Issue 7: Out of Memory (OOM)

**Symptoms**: Service crashes, 502 errors

**Fix**:
``````bash
# Reduce Gunicorn workers
gunicorn config.wsgi:application --workers 2

# Upgrade Render/Railway plan
# Or optimize queries:
# - Add select_related/prefetch_related
# - Add database indexes
# - Enable query caching
``````

### Issue 8: CORS Errors

**Symptoms**: `Access-Control-Allow-Origin` errors in browser

**Fix**:
``````python
# Update CORS_ALLOWED_ORIGINS in .env
CORS_ALLOWED_ORIGINS=https://your-frontend.vercel.app,https://www.your-domain.com

# Check settings/base.py
CORS_ALLOWED_ORIGINS = env('CORS_ALLOWED_ORIGINS')
CORS_ALLOW_CREDENTIALS = True
``````

---

## 8. Deployment Checklist

### Pre-Deployment

- [ ] All tests passing
- [ ] Environment variables configured
- [ ] Database migrations created
- [ ] Static files collected
- [ ] Security settings enabled (DEBUG=False)
- [ ] ALLOWED_HOSTS configured
- [ ] CORS origins whitelisted
- [ ] API keys rotated (if needed)

### Post-Deployment

- [ ] Health check returns 200
- [ ] Database migrations applied
- [ ] Static files loading
- [ ] WebSocket connections working
- [ ] Celery workers running
- [ ] Scheduled tasks executing
- [ ] Email sending working
- [ ] Payment flow tested
- [ ] Error tracking active (Sentry)
- [ ] Monitoring dashboards configured

---

## 9. Rollback Procedure

### Render/Railway

``````bash
# 1. Go to deployment history
# 2. Click on previous successful deployment
# 3. Click "Redeploy"
# 4. Verify health check
``````

### Vercel

``````bash
# 1. Go to deployments page
# 2. Find previous deployment
# 3. Click "Promote to Production"
``````

### AWS EC2 (Blue-Green)

``````bash
# 1. Switch ALB target group to previous version
# 2. Verify health checks
# 3. Monitor error rates
# 4. If stable, terminate new instances
``````

---

## 10. Scaling Guide

### Vertical Scaling (Render/Railway)

``````
Starter: 512 MB RAM, 0.5 CPU
Standard: 2 GB RAM, 1 CPU
Pro: 4 GB RAM, 2 CPU
``````

### Horizontal Scaling (AWS)

``````
Auto Scaling Group:
- Min instances: 2
- Max instances: 10
- Scale on: CPU > 70% or Request count > 1000/min
``````

### Database Scaling

``````
Read Replicas: For read-heavy workloads
Connection Pooling: PgBouncer
Caching: Redis for frequent queries
``````

---

## 11. Backup Strategy

### Database Backups

**Render**: Automatic daily backups (7-day retention)

**AWS RDS**: 
- Automated backups (7-day retention)
- Manual snapshots before major changes
- Point-in-time recovery enabled

### Media Files Backups

**S3 Versioning**: Enabled on media bucket

**Lifecycle Policy**: 
- Keep versions for 90 days
- Move to Glacier after 30 days

---

## 12. Monitoring & Alerts

### Uptime Monitoring

- **UptimeRobot**: Check every 5 minutes
- **Alert**: Email + Slack on downtime

### Error Tracking

- **Sentry**: Real-time error alerts
- **Threshold**: Alert on > 10 errors/min

### Performance Monitoring

- **Render Metrics**: CPU, Memory, Response time
- **Vercel Analytics**: Web Vitals, Page load time

### Log Aggregation

- **Render Logs**: Last 7 days
- **AWS CloudWatch**: Long-term storage

---

## 13. Security Checklist

- [ ] HTTPS enabled (SSL certificate)
- [ ] Environment variables secured (not in code)
- [ ] Database password rotated
- [ ] API keys rotated
- [ ] CORS properly configured
- [ ] Rate limiting enabled
- [ ] Brute force protection active
- [ ] SQL injection prevention (ORM only)
- [ ] XSS prevention (template escaping)
- [ ] CSRF protection enabled
- [ ] Security headers configured (HSTS, CSP)
- [ ] Webhook signatures verified
- [ ] File upload validation
- [ ] Input sanitization

---

## 14. Cost Optimization

### Render/Railway (Initial)

``````
Backend: ~$7-25/month
Database: ~$7-15/month
Redis: Free tier (Upstash)
Total: ~$15-40/month
``````

### AWS (Production)

``````
EC2 (3x t3.medium): ~$75/month
RDS (db.t3.medium): ~$60/month
ElastiCache (cache.t3.medium): ~$50/month
S3 + CloudFront: ~$10-30/month
Total: ~$200-250/month
``````

**Optimization Tips**:
- Use Reserved Instances (save 30-50%)
- Enable auto-scaling (scale down during low traffic)
- Use S3 lifecycle policies
- Optimize database queries
- Enable caching aggressively

---

**For detailed AWS deployment**: See `deployment/AWS_EC2_DEPLOYMENT.md`  
**For CI/CD setup**: See `deployment/cicd/README.md`  
**For monitoring setup**: See `deployment/monitoring/README.md`

---

**Document Status**: Complete  
**Last Updated**: May 1, 2026