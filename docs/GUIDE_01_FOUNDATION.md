# FreelanceFlow — Backend Interview Guide
# Part 1: Foundation Layer
> **Order to read**: This file first → `GUIDE_02_BUSINESS_LOGIC.md` → `GUIDE_03_ADVANCED.md`

---

## What is FreelanceFlow?

A **freelance marketplace backend** built with Django + DRF. Clients post projects, freelancers bid, a contract is created when a bid is accepted, work is logged daily, payments are escrowed via Razorpay, and real-time chat runs over WebSockets. Think Upwork — built from scratch.

---

## Part 1 covers: Entry points → Config → Core utilities → Users & Auth

```
Start here:
manage.py
  └── config/settings/   (base → local → production)
  └── config/urls.py     (all API routes)
  └── config/asgi.py     (WebSocket entry point)
  └── config/celery.py   (background task broker)
  └── config/signals.py  (graceful shutdown)

Then:
core/                    (shared utilities used by ALL apps)
  ├── middleware.py
  ├── exceptions.py
  ├── permissions.py
  ├── pagination.py
  ├── cache.py
  ├── sanitizers.py
  └── throttles.py

Then:
apps/users/              (auth, profiles — everything else depends on this)
  ├── models.py
  ├── signals.py
  ├── services.py
  ├── serializers.py
  ├── views.py
  └── urls.py
```

---

## Step 1 — `manage.py`

**What it does:** Django's CLI entry point. Sets `DJANGO_SETTINGS_MODULE` to `config.settings.local`.

**Interview point:** "Why do we have separate settings files?"
- `base.py` — shared across all environments
- `local.py` — overrides for dev (console email, no HTTPS)
- `production.py` — strict security, S3 storage, Sentry

---

## Step 2 — `config/settings/base.py`

**Open this file and explain in order:**

### 2a. Environment variables via `django-environ`
```python
env = environ.Env(
    DATABASE_URL=(str, "sqlite:///db.sqlite3"),
    REDIS_URL=(str, "redis://localhost:6379/0"),
    ...
)
environ.Env.read_env(os.path.join(BASE_DIR, ".env"))
```
**Why:** All secrets come from `.env`, never hardcoded. The `(type, default)` tuple means the app can boot even if a variable is missing — important for CI environments.

### 2b. INSTALLED_APPS split into 3 groups
```python
DJANGO_APPS = [...]      # django.contrib.*
THIRD_PARTY_APPS = [...]  # DRF, JWT, channels, axes, etc.
LOCAL_APPS = [...]        # our apps/
```
**Why:** Clean separation. Makes it obvious which apps are ours vs framework vs packages.

### 2c. Database — single line
```python
DATABASES = {"default": env.db_url("DATABASE_URL", default="sqlite:///db.sqlite3")}
```
`django-environ` parses the full `postgresql://user:pass@host/db` URL into the Django dict format.

### 2d. CHANNEL_LAYERS — WebSocket backbone
```python
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {"hosts": [REDIS_URL]},
    }
}
```
**Why Redis:** WebSocket messages need to broadcast across multiple server processes. Redis pub/sub enables that. Without this, two users on different Daphne workers can't communicate.

### 2e. Celery queues — 3 priority levels
```python
task_queues=(
    Queue('freelanceflow'),               # default
    Queue('freelanceflow_high_priority'), # payments, webhooks
    Queue('freelanceflow_low_priority'),  # PDF/AI reports
)
```
**Interview point:** "Why separate queues?" — Payment tasks must not be blocked by a slow PDF generation job. High-priority queue runs on a dedicated worker.

### 2f. CACHES — Redis with `IGNORE_EXCEPTIONS=True`
```python
"OPTIONS": {
    "IGNORE_EXCEPTIONS": True,  # Don't crash if Redis is down
}
```
**Why:** Cache is not critical path. If Redis is down, the app still serves requests from the DB — just slower.

### 2g. django-axes (brute force protection)
```python
AXES_LOGIN_FAILURE_LIMIT = 5
AXES_COOLOFF_TIME = 300  # 5 minutes lockout
AXES_HANDLER = "axes.handlers.cache.AxesCacheHandler"
```
Tracks failed login attempts in Redis (cache backend). After 5 failures from the same IP, locks out for 5 minutes.

---

## Step 3 — `config/urls.py`

```python
urlpatterns = [
    path("api/users/",         include("apps.users.urls")),
    path("api/projects/",      include("apps.projects.urls")),
    path("api/bidding/",       include("apps.bidding.urls")),
    path("api/payments/",      include("apps.payments.urls")),
    path("api/worklogs/",      include("apps.worklogs.urls")),
    path("api/messaging/",     include("apps.messaging.urls")),
    path("api/notifications/", include("apps.notifications.urls")),
    path("api/search/",        include("apps.search.urls")),
]
```
All 8 apps mounted under `/api/`. Clean, predictable namespace.

---

## Step 4 — `config/asgi.py`

**The WebSocket entry point.**

```python
application = ProtocolTypeRouter({
    "http": django_asgi_app,          # normal HTTP → Django
    "websocket": AuthMiddlewareStack(
        URLRouter(routing.websocket_urlpatterns)
    ),
})
```

**Interview points:**
- `ProtocolTypeRouter` — routes by protocol (HTTP vs WS)
- `AuthMiddlewareStack` — wraps Django's session auth around WS connections
- JWT auth is done manually in `ChatConsumer.get_user_from_token()` via query string `?token=<JWT>`
- **Graceful shutdown:** SIGTERM handler calls `channel_layer.flush()` then closes DB/cache connections before exit

---

## Step 5 — `config/celery.py`

```python
app = Celery("freelanceflow")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
```

**Key configurations:**
| Setting | Value | Why |
|---|---|---|
| `task_acks_late=True` | Ack after execution | Task re-queued if worker crashes mid-task |
| `worker_prefetch_multiplier=1` | No prefetch | Fair distribution for long-running tasks |
| `task_reject_on_worker_lost=True` | Re-queue on crash | Zero message loss |
| `worker_max_tasks_per_child=1000` | Restart worker | Prevents memory leaks |

**SIGTERM handler** → triggers graceful drain (finish current task, reject new ones, close DB connections).

---

## Step 6 — `config/signals.py` (GracefulShutdown class)

```python
class GracefulShutdown:
    def shutdown(self, signum, frame):
        self._close_database_connections()
        self._close_cache_connections()
        self._close_elasticsearch_connections()
        logging.shutdown()
        sys.exit(0)
```
**Pattern:** Singleton instance registered to SIGTERM/SIGINT. Prevents double-shutdown with `is_shutting_down` flag. Closes all resources in correct order before exit.

---

## Step 7 — `core/middleware.py` (read top-to-bottom)

4 middleware classes, applied in order via `MIDDLEWARE` in `base.py`:

### `GracefulShutdownMiddleware` (first in chain)
Rejects new requests when `is_shutting_down()` returns True. Returns 503 immediately — in-flight requests drain, new ones are refused.

### `RequestLoggingMiddleware`
```python
duration = (time.perf_counter() - start_time) * 1000
log_data = {"method", "path", "status", "duration_ms", "user_id", "ip"}
response["X-Response-Time"] = f"{duration}ms"
```
Every request gets a structured JSON log. Status ≥500 → ERROR, ≥400 → WARNING, else INFO. Adds `X-Response-Time` header for client debugging.

### `SecurityHeadersMiddleware`
Adds `X-Content-Type-Options: nosniff`, `X-XSS-Protection`, `Referrer-Policy`. CORS origin from `CORS_ALLOWED_ORIGINS` in prod.

### `CORSCustomMiddleware`
Custom CORS — reads `HTTP_ORIGIN`, only sets `Access-Control-Allow-Origin` if origin is in the allowlist (or DEBUG mode).

---

## Step 8 — `core/exceptions.py`

### Custom exception hierarchy
```python
BusinessError              # base
  ├── PermissionDeniedError
  ├── NotFoundError
  └── ValidationError(field=None)  # field name for form errors
```

### `custom_exception_handler`
DRF calls this instead of its default handler. Normalises ALL errors into:
```json
{ "error": "Human readable", "code": "error_code", "field": "field_name" }
```
**Why:** Without this, DRF returns `{"email": ["This field is required."]}`. With it, frontend always gets a consistent shape regardless of error source.

---

## Step 9 — `core/permissions.py`

```python
class IsClient(BasePermission):
    def has_permission(self, request, view):
        return request.user.role == "CLIENT"

class IsFreelancer(BasePermission):
    def has_permission(self, request, view):
        return request.user.role == "FREELANCER"
```
Role-based guards reused across all views. Applied as `permission_classes = [IsAuthenticated, IsClient]`.

---

## Step 10 — `core/cache.py`

Helper wrappers around Django's cache:
```python
def cache_get_or_set(key, callable, timeout):
    value = cache.get(key)
    if value is None:
        value = callable()
        cache.set(key, value, timeout)
    return value
```
Used in selectors (read layer) to cache expensive DB queries. Cache keys are namespaced by `KEY_PREFIX = "freelanceflow"`.

---

## Step 11 — `apps/users/models.py`

### `User` (AbstractUser subclass)
```python
class User(AbstractUser):
    username = None          # removed — email is the login
    email = models.EmailField(unique=True)
    role = models.CharField(choices=[CLIENT, FREELANCER])
    is_deactivated = models.BooleanField(default=False)  # soft delete
    USERNAME_FIELD = "email"
```

### `FreelancerProfile` (OneToOne → User)
Key fields: `skills` (JSONField list), `hourly_rate`, `subscription_tier` (FREE/PRO), `total_earned`, `average_rating`, `is_available`.

### `ClientProfile` (OneToOne → User)
Key fields: `company_name`, `total_spent`, `average_rating`.

**Interview point:** "Why separate profile models instead of one big User?"
- Single Responsibility — User handles auth, Profile handles domain data
- Easy to extend one without touching the other
- `OneToOneField` gives O(1) lookup: `user.freelancer_profile`

---

## Step 12 — `apps/users/signals.py`

```python
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        if instance.role == FREELANCER:
            FreelancerProfile.objects.create(user=instance)
        elif instance.role == CLIENT:
            ClientProfile.objects.create(user=instance)
        # Send welcome email AFTER transaction commits (not during)
        transaction.on_commit(
            lambda: send_welcome_email_task.delay(instance.id)
        )
```

**Key pattern — `transaction.on_commit`:** The Celery task is dispatched only after the DB transaction commits. If we called `.delay()` directly inside the signal, the task might run before the user row is visible to other DB connections (race condition).

---

## Step 13 — `apps/users/services.py` (business logic layer)

Each function has a single responsibility. Open and walk through:

| Function | What it does |
|---|---|
| `create_user()` | Validates email uniqueness, password strength, creates user in `transaction.atomic()` |
| `update_profile()` | Updates User + Profile fields based on role |
| `change_password()` | Verifies old password, validates new one, saves |
| `send_password_reset_email()` | Generates `uid` + token, builds frontend link, sends email. Never reveals if email exists. |
| `reset_password()` | Decodes uid, verifies token (time-limited), sets new password |
| `deactivate_account()` | Soft delete — sets `is_deactivated=True`, `is_active=False`, sends confirmation email |
| `toggle_freelancer_availability()` | Flips `is_available` on FreelancerProfile |

**Architecture pattern (Service Layer):**
- Views → call service functions → services contain ALL business logic
- Views never touch models directly
- Services raise `ValidationError`/`BusinessError` → view catches and returns 400/403

---

## Step 14 — `apps/users/serializers.py`

Split into `serializers.py` (core) and `serializers_extended.py` (extras).

Key serializers:
- `RegisterSerializer` — validates role, calls `create_user()` service
- `UserProfileSerializer` — read-only nested profile (role-aware: shows FreelancerProfile or ClientProfile fields)
- `LoginSerializer` — returns `access` + `refresh` JWT tokens via `simplejwt`

---

## Step 15 — `apps/users/views.py`

Views are thin — they delegate to services:
```python
class RegisterView(CreateAPIView):
    permission_classes = [AllowAny]

    def create(self, request):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = create_user(**serializer.validated_data)  # ← calls service
        return Response(UserSerializer(user).data, status=201)
```

JWT auth endpoints via `simplejwt`:
- `POST /api/users/login/` → access + refresh token
- `POST /api/users/token/refresh/` → new access token
- `POST /api/users/logout/` → blacklists refresh token

---

## Key Concepts to Articulate

### "Explain the request lifecycle"
1. Request hits Nginx → Gunicorn (HTTP) or Daphne (WS)
2. Passes through middleware stack (logging → security headers → CORS → auth)
3. JWT decoded by `JWTAuthentication` — sets `request.user`
4. `permission_classes` check role
5. Serializer validates input
6. View calls service function
7. Service validates business rules, writes DB in `transaction.atomic()`
8. Signal fires (e.g., send email via Celery `on_commit`)
9. Serializer formats response
10. `RequestLoggingMiddleware` logs duration

### "Why Service Layer instead of fat models/views?"
- **Testable** — services are plain Python functions, easy to unit test
- **Reusable** — management commands, Celery tasks, and views all call the same service
- **Separation** — views handle HTTP, services handle business rules, models handle persistence

### "How does JWT token blacklisting work?"
`simplejwt` with `BLACKLIST_AFTER_ROTATION=True`. On logout, the refresh token's `jti` (JWT ID) is stored in `OutstandingToken` table. On next refresh attempt, it's rejected. `ROTATE_REFRESH_TOKENS=True` means every refresh returns a new refresh token too.

---

> **Next:** Open `GUIDE_02_BUSINESS_LOGIC.md` → Projects → Bidding → Payments
