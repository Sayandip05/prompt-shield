# FreelanceFlow — Backend Interview Guide
# Part 3: Advanced Layer
> **Order:** `GUIDE_01_FOUNDATION.md` → `GUIDE_02_BUSINESS_LOGIC.md` → **This file**

---

## What this file covers

```
apps/worklogs/    — daily work logs, AI reports (Groq LLM), PDF generation
apps/messaging/   — real-time WebSocket chat (Django Channels + Redis)
apps/notifications/ — in-app + email + push notifications
apps/search/      — Elasticsearch full-text search (projects + freelancers)
```

---

## apps/worklogs — File reading order

```
models.py              → WorkLog, WeeklyReport, Deliverable, DeliveryProof
models_extended.py     → TimeOffRequest
selectors.py           → query helpers (approved logs, pending logs, etc.)
services.py            → submit_log, approve_log, reject_log, generate_proof
ai_service.py          → LLM summarisation using LangGraph
groq_service.py        → Groq API integration (LLM calls)
pdf_service.py         → PDF generation from templates
tasks.py               → Celery tasks for AI + PDF (low priority queue)
views.py               → all worklog API endpoints
urls.py
```

---

## Step 27 — `apps/worklogs/models.py`

### `WorkLog`
```python
class WorkLog(models.Model):
    class Status(TextChoices):
        DRAFT            = "DRAFT"
        PENDING_APPROVAL = "PENDING_APPROVAL"
        APPROVED         = "APPROVED"
        REJECTED         = "REJECTED"

    contract      = ForeignKey(Contract, related_name="work_logs")
    freelancer    = ForeignKey(User)
    date          = DateField()
    description   = TextField()
    hours_worked  = DecimalField(validators=[MinValueValidator(0.1), MaxValueValidator(24)])
    screenshot_url = URLField(blank=True)
    reference_url  = URLField(blank=True)   # GitHub PR, Figma link, etc.
    status        = CharField(default=Status.DRAFT)
    approved_at   = DateTimeField(null=True)
    approved_by   = ForeignKey(User, null=True, related_name="approved_work_logs")

    class Meta:
        unique_together = ["contract", "date"]  # one log per day per contract
```

`unique_together = ["contract", "date"]` — **prevents duplicate logs for the same day**. A freelancer can't submit Monday's log twice.

### `Deliverable`
Represents a discrete piece of work with an AI-generated report:
```python
class Deliverable(models.Model):
    class Status(TextChoices):
        DRAFT              = "DRAFT"
        SUBMITTED          = "SUBMITTED"
        UNDER_REVIEW       = "UNDER_REVIEW"
        APPROVED           = "APPROVED"
        REJECTED           = "REJECTED"
        REVISION_REQUESTED = "REVISION_REQUESTED"

    ai_chat_transcript    = JSONField(default=list)  # full LLM conversation
    ai_generated_report   = TextField(blank=True)    # AI summary
    hours_logged          = DecimalField(default=0)
    payment_released      = BooleanField(default=False)
```

### `DeliveryProof`
Generated at project completion — a tamper-evident audit document:
```python
class DeliveryProof(models.Model):
    contract              = OneToOneField(Contract)
    pdf_url               = URLField()             # S3 URL
    total_hours           = DecimalField(...)
    total_logs_count      = IntegerField()
    total_deliverables    = IntegerField()
    approved_deliverables = IntegerField()
    report_id             = CharField(unique=True) # UUID-based, tamper-evident
```

**Interview point:** "What is a DeliveryProof and why does it have a unique report_id?"
It's the final certificate of work. The `report_id` is a UUID-derived unique string — if someone modifies the PDF, the ID no longer matches the DB record, proving tampering.

---

## Step 28 — `apps/worklogs/services.py` (worklog lifecycle)

### `submit_log()`
```python
@transaction.atomic
def submit_log(freelancer, contract_id, date, description, hours, reference_url):
    contract = get_object_or_404(Contract, id=contract_id, is_active=True)
    if contract.freelancer != freelancer:
        raise PermissionDeniedError("Not your contract.")
    if WorkLog.objects.filter(contract=contract, date=date).exists():
        raise ValidationError("Log for this date already submitted.")

    log = WorkLog.objects.create(
        contract=contract, freelancer=freelancer,
        date=date, description=description,
        hours_worked=hours, reference_url=reference_url,
        status=WorkLog.Status.PENDING_APPROVAL,
    )
    Notification.objects.create(recipient=contract.client, ...)
    return log
```

### `approve_log()` / `reject_log()`
```python
def approve_log(client, log_id):
    log = get_object_or_404(WorkLog, id=log_id, contract__bid__project__client=client)
    if log.status != WorkLog.Status.PENDING_APPROVAL:
        raise BusinessError("Log is not pending approval.")
    log.status = WorkLog.Status.APPROVED
    log.approved_at = timezone.now()
    log.approved_by = client
    log.save()
    Notification.objects.create(recipient=log.freelancer, ...)
```

### `generate_delivery_proof()`
```python
@transaction.atomic
def generate_delivery_proof(client, contract_id):
    contract = get_object_or_404(Contract, id=contract_id)
    approved_logs = WorkLog.objects.filter(contract=contract, status="APPROVED")
    total_hours = approved_logs.aggregate(Sum('hours_worked'))['hours_worked__sum']

    # Trigger async PDF generation
    generate_pdf_task.delay(contract_id)

    return DeliveryProof.objects.create(
        contract=contract,
        total_hours=total_hours,
        total_logs_count=approved_logs.count(),
        report_id=uuid.uuid4().hex[:24],
        pdf_url="",  # filled by Celery task after PDF uploaded to S3
    )
```

---

## Step 29 — `apps/worklogs/groq_service.py` (AI integration)

This is the AI brain of the worklog system. Uses **Groq** (fast LLM inference) to:
1. Generate daily worklog summaries
2. Produce weekly progress reports
3. Power the AI chat assistant for deliverable creation

```python
from groq import Groq

client = Groq(api_key=settings.GROQ_API_KEY)

def generate_worklog_summary(description: str, hours: float) -> str:
    response = client.chat.completions.create(
        model="llama3-8b-8192",   # Groq's fast Llama 3 model
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Hours: {hours}\nWork done: {description}"},
        ],
        max_tokens=500,
        temperature=0.3,  # low temperature = factual, consistent output
    )
    return response.choices[0].message.content
```

**Interview points:**
- `temperature=0.3` — lower = more deterministic. For work summaries we want facts, not creativity.
- Model `llama3-8b-8192` — 8B parameter model via Groq's fast inference. 8192 token context window.
- Response is stored in `WorkLog.ai_generated_summary` for the weekly PDF.

---

## Step 30 — `apps/worklogs/ai_service.py` (LangGraph agent)

For the deliverable AI chat flow — a **multi-turn conversation agent**:

```python
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

# State machine for the deliverable creation conversation
class DeliverableState(TypedDict):
    messages: List[BaseMessage]
    deliverable_data: dict
    is_complete: bool

def build_deliverable_agent():
    graph = StateGraph(DeliverableState)
    graph.add_node("gather_info", gather_info_node)
    graph.add_node("generate_report", generate_report_node)
    graph.add_node("confirm", confirm_node)

    graph.add_edge("gather_info", "generate_report")
    graph.add_conditional_edges(
        "confirm",
        lambda state: END if state["is_complete"] else "gather_info"
    )
    return graph.compile()
```

**Interview point:** "Why LangGraph instead of a simple LLM call?"
Multi-turn deliverable creation needs **state** — the agent asks questions, remembers previous answers, and only generates the final report when it has enough info. LangGraph models this as a state machine graph. Each node is a step; edges define the flow.

---

## Step 31 — `apps/worklogs/tasks.py` (Celery background tasks)

```python
@shared_task(queue='freelanceflow_low_priority')
def generate_ai_report_task(worklog_id):
    """Generate AI summary for a submitted worklog."""
    log = WorkLog.objects.get(id=worklog_id)
    summary = groq_service.generate_worklog_summary(log.description, log.hours_worked)
    log.ai_generated_summary = summary
    log.save(update_fields=["ai_generated_summary"])

@shared_task(queue='freelanceflow_low_priority')
def generate_pdf_task(contract_id):
    """Generate PDF delivery proof and upload to S3."""
    from .pdf_service import generate_proof_pdf
    pdf_url = generate_proof_pdf(contract_id)
    DeliveryProof.objects.filter(contract_id=contract_id).update(pdf_url=pdf_url)
    Notification.objects.create(...)

@shared_task(queue='freelanceflow_low_priority')
def generate_weekly_reports_for_all_contracts():
    """Runs weekly via Celery Beat. Generates reports for all active contracts."""
    active_contracts = Contract.objects.filter(is_active=True)
    for contract in active_contracts:
        # Aggregate last 7 days of approved logs
        ...
```

**Why low-priority queue?** PDF generation and AI calls are slow (1–5 seconds). They must not block payment webhook processing (which takes <100ms).

---

## apps/messaging — File reading order

```
models.py          → Conversation (OneToOne Contract), Message
models_extended.py → MessageReaction, ThreadedReply
consumers.py       → ChatConsumer (WebSocket handler — READ THIS CAREFULLY)
routing.py         → WebSocket URL patterns
services.py        → send_message, mark_as_read
services_search.py → search messages by content
services_typing.py → broadcast typing indicators
selectors.py       → get messages, conversations
views.py           → REST endpoints (history, mark read)
urls.py
```

---

## Step 32 — `apps/messaging/consumers.py` (WebSocket — the most interview-heavy file)

```python
class ChatConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        self.contract_id = self.scope['url_route']['kwargs']['contract_id']
        self.room_group_name = f'chat_{self.contract_id}'

        # Step 1: Authenticate via JWT in query string
        self.user = await self.get_user_from_token()
        if isinstance(self.user, AnonymousUser):
            await self.close()
            return

        # Step 2: Authorization — only contract participants
        if not await self.is_contract_participant():
            await self.close()
            return

        # Step 3: Join Redis channel group
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

    async def receive(self, text_data):
        data = json.loads(text_data)
        # Save to DB (sync → async bridge)
        message = await database_sync_to_async(send_message)(
            sender=self.user,
            conversation_id=...,
            content=data['message']
        )
        # Broadcast to all participants in this contract's channel
        await self.channel_layer.group_send(
            self.room_group_name,
            {'type': 'chat_message', 'message': {...}}
        )

    async def chat_message(self, event):
        # Called by group_send — sends to this specific WebSocket connection
        await self.send(text_data=json.dumps(event['message']))

    @database_sync_to_async
    def get_user_from_token(self):
        query = self.scope.get('query_string', b'').decode()
        if 'token=' in query:
            token = query.split('token=')[1].split('&')[0]
            try:
                access_token = AccessToken(token)
                return User.objects.get(id=access_token['user_id'])
            except Exception:
                pass
        return AnonymousUser()
```

**Interview: Walk through what happens when a message is sent**
1. Frontend sends JSON over WS: `{"message": "Hello!"}`
2. `receive()` fires
3. `database_sync_to_async(send_message)()` — saves to `Message` table (sync DB call wrapped for async)
4. `channel_layer.group_send()` — publishes to Redis pub/sub for `chat_{contract_id}`
5. Redis delivers to ALL connected consumers in that group
6. Each consumer's `chat_message()` fires → `self.send()` pushes to that browser's WebSocket

**Interview: Why `database_sync_to_async`?**
Django ORM is synchronous. Django Channels consumers are async (`asyncio`). Calling a sync function from async context blocks the event loop. The wrapper runs the sync code in a thread pool, freeing the event loop.

**Interview: What is a "channel group"?**
A named pub/sub topic in Redis. `chat_42` = the group for contract #42. Any consumer that calls `group_add("chat_42", channel_name)` will receive messages sent to `group_send("chat_42", ...)`. This works across multiple Daphne processes.

**Interview: How is JWT auth done for WebSockets?**
HTTP headers can't be set on browser WebSocket connections. Instead, the token is passed as a query param: `ws://api/ws/chat/42/?token=<JWT>`. `get_user_from_token()` parses it from `self.scope['query_string']`.

---

## Step 33 — `apps/messaging/routing.py`

```python
websocket_urlpatterns = [
    re_path(r'ws/chat/(?P<contract_id>\d+)/$', ChatConsumer.as_asgi()),
]
```

Mounted in `config/asgi.py` via `URLRouter`. Pattern: `/ws/chat/{contract_id}/`.

---

## apps/notifications — File reading order

```
models.py               → Notification (in-app)
models_extended.py      → NotificationPreference (user settings)
models_push.py          → PushSubscription (browser push via FCM)
services.py             → create_notification, bulk_notify
services_push.py        → send FCM push notification
services_digest.py      → send daily digest email
services_announcement.py → platform-wide announcements
email_service.py        → transactional email templates
tasks.py                → Celery async email/push tasks
views.py                → list, mark_read, mark_all_read
urls.py
```

---

## Step 34 — `apps/notifications/models.py`

```python
class Notification(models.Model):
    class Type(TextChoices):
        BID_SUBMITTED    = "BID_SUBMITTED"
        BID_ACCEPTED     = "BID_ACCEPTED"
        ESCROW_CREATED   = "ESCROW_CREATED"
        LOG_SUBMITTED    = "LOG_SUBMITTED"
        REPORT_READY     = "REPORT_READY"
        PAYMENT_RELEASED = "PAYMENT_RELEASED"
        PROOF_READY      = "PROOF_READY"
        MESSAGE_RECEIVED = "MESSAGE_RECEIVED"

    recipient = ForeignKey(User, related_name="notifications")
    title     = CharField(max_length=255)
    body      = TextField()
    type      = CharField(choices=Type.choices)
    is_read   = BooleanField(default=False)

    class Meta:
        indexes = [models.Index(fields=["recipient", "is_read"])]
```

**Composite index on `(recipient, is_read)`** — The most common query is `WHERE recipient_id = X AND is_read = false` (unread count badge). This index makes it O(log n).

---

## Step 35 — `apps/notifications/services.py`

```python
def create_notification(recipient, title, body, notif_type):
    notification = Notification.objects.create(
        recipient=recipient, title=title, body=body, type=notif_type
    )
    # Check user preferences
    prefs = NotificationPreference.objects.get(user=recipient)
    if prefs.email_enabled:
        send_notification_email_task.delay(notification.id)
    if prefs.push_enabled:
        send_push_notification_task.delay(notification.id)
    return notification
```

**Three delivery channels:**
1. **In-app** — stored in `Notification` table, polled by frontend
2. **Email** — Celery task → `email_service.py` → Django `send_mail`
3. **Push** — Celery task → `services_push.py` → Firebase FCM API

---

## apps/search — File reading order

```
documents.py           → ES index definitions (ProjectDocument, FreelancerDocument)
signals.py             → sync ES on model save/delete
services.py            → search_projects(), search_freelancers()
services_autocomplete.py → typeahead suggestions
services_saved.py      → saved searches
services_history.py    → search history tracking
selectors.py           → query helpers
views.py               → search API endpoints
urls.py
```

---

## Step 36 — `apps/search/documents.py`

```python
from django_elasticsearch_dsl import Document, fields
from django_elasticsearch_dsl.registries import registry

@registry.register_document
class ProjectDocument(Document):
    skills = fields.NestedField(properties={
        'skill_name': fields.KeywordField(),
    })
    client_name = fields.TextField()
    status      = fields.KeywordField()
    budget      = fields.FloatField()

    class Index:
        name = 'projects'
        settings = {'number_of_shards': 1, 'number_of_replicas': 0}

    class Django:
        model = Project
        fields = ['title', 'description', 'created_at']
        related_models = [ProjectSkill]

@registry.register_document
class FreelancerDocument(Document):
    skills         = fields.ListField(fields.KeywordField())
    hourly_rate    = fields.FloatField()
    average_rating = fields.FloatField()
    is_available   = fields.BooleanField()

    class Index:
        name = 'freelancers'

    class Django:
        model = FreelancerProfile
        fields = ['bio']
```

**Interview points:**
- `NestedField` for skills — allows querying inside the nested object: `skills.skill_name = "React"`
- `KeywordField` vs `TextField` — Keywords are exact match (filter/aggregate), TextFields are full-text searched (analyzed)
- `number_of_replicas: 0` in dev — no replica needed for single-node local ES

---

## Step 37 — `apps/search/signals.py`

```python
from django.db.models.signals import post_save, post_delete

@receiver(post_save, sender=Project)
def update_project_document(sender, instance, **kwargs):
    _safe_es_update(lambda: ProjectDocument().update(instance))

@receiver(post_delete, sender=Project)
def delete_project_document(sender, instance, **kwargs):
    _safe_es_update(lambda: ProjectDocument().get(id=instance.id).delete())

def _safe_es_update(fn):
    """Never crash the app if ES is down."""
    try:
        fn()
    except Exception as exc:
        logger.warning(f"ES sync failed (non-critical): {exc}")
```

**`_safe_es_update` pattern** — ES is a search cache, not the source of truth. If ES is down, the write still succeeds to Postgres. The `WARNING` log alerts ops. ES is rebuilt with `manage.py search_index --rebuild`.

**Disconnected during seeding** — The seed command calls `ELASTICSEARCH_DSL_AUTOSYNC = False` and tears down the signal processor before bulk inserting data, reconnecting only if ES is reachable after. This prevents ES errors from blocking development seeding.

---

## Step 38 — `apps/search/services.py`

```python
def search_projects(query=None, skills=None, min_budget=None, max_budget=None,
                    status="OPEN", page=1, page_size=20):
    s = ProjectDocument.search()

    if query:
        s = s.query("multi_match", query=query,
                    fields=["title^3", "description"],  # title boosted 3x
                    fuzziness="AUTO")

    if skills:
        s = s.filter("nested", path="skills",
                     query=Q("terms", skills__skill_name=skills))

    if min_budget:
        s = s.filter("range", budget={"gte": min_budget})
    if max_budget:
        s = s.filter("range", budget={"lte": max_budget})

    s = s.filter("term", status=status)
    s = s[((page-1) * page_size):(page * page_size)]
    return s.execute()
```

**Interview points:**
- `multi_match` — searches across multiple fields in one query
- `^3` boost — title matches are 3x more relevant than description matches
- `fuzziness="AUTO"` — typo tolerance (1–2 character edits allowed)
- `nested` filter — required for querying NestedField documents
- Pagination via ES slice notation `s[from:to]`

---

## Full System Flow — End to End

### "Walk me through everything that happens when a freelancer submits a worklog"

```
1. POST /api/worklogs/  → WorkLogCreateView
2. JWTAuthentication decodes token → request.user = freelancer
3. IsFreelancer permission check passes
4. WorkLogSerializer validates (hours 0.1–24, required fields)
5. services.submit_log() called:
   a. Verifies contract is active and belongs to freelancer
   b. Checks unique_together → no duplicate date
   c. Creates WorkLog(status=PENDING_APPROVAL)
   d. Creates Notification for client (in-app)
   e. transaction.on_commit → generate_ai_report_task.delay(log.id)
6. HTTP 201 returned to frontend
7. [Async] generate_ai_report_task runs on low-priority queue:
   a. Calls groq_service.generate_worklog_summary()
   b. Stores AI summary in WorkLog.ai_generated_summary
8. Client sees notification → reviews log → calls approve_log()
9. Log.status = APPROVED, approved_by = client
10. [Later] generate_weekly_reports_for_all_contracts Celery Beat task
    aggregates approved logs → generates PDF → uploads to S3 → stores URL
```

---

## Final Interview Cheat Sheet

| Concept | Implementation |
|---|---|
| Auth | JWT via `simplejwt` — access (60min) + refresh (7d) + blacklist on logout |
| Role-based access | `IsClient` / `IsFreelancer` permission classes + `limit_choices_to` on FK |
| No race conditions | `select_for_update()` + `transaction.atomic()` on payment release |
| Webhook idempotency | `PaymentEvent.razorpay_event_id` unique — duplicate = IntegrityError |
| WebSocket auth | JWT in query string `?token=` parsed in `get_user_from_token()` |
| Real-time delivery | Redis Channel Layer pub/sub via `group_send` |
| Background jobs | Celery with 3 priority queues (high=payments, low=AI/PDF, default=rest) |
| Graceful shutdown | SIGTERM → drain tasks → close DB/cache/ES connections → exit |
| ES failure tolerance | `_safe_es_update` — catches all ES exceptions, logs WARNING, never crashes |
| Brute force | `django-axes` — 5 failures → 5min lockout via Redis cache |
| Audit trail | `DeliveryProof` with unique `report_id` — tamper-evident delivery certificate |
| AI reports | Groq LLM (Llama 3) via async Celery task — non-blocking |
| Agent flow | LangGraph state machine for multi-turn deliverable creation chat |
| Search | Elasticsearch with `multi_match`, nested skills filter, budget range, fuzzy |
| Notifications | 3 channels — in-app (DB), email (Celery), push (FCM) based on user prefs |
| N+1 prevention | `select_related` / `prefetch_related` in all selector functions |
| Soft delete | `is_deactivated` + `is_active=False` on User — data retained, login blocked |
