# FreelanceFlow — Backend Interview Guide
# Part 2: Business Logic Layer
> **Order:** `GUIDE_01_FOUNDATION.md` → **This file** → `GUIDE_03_ADVANCED.md`

---

## What this file covers

```
apps/projects/    — project posting, skills, drafts, bookmarks
apps/bidding/     — bid lifecycle, contracts, amendments, reviews, terminations
apps/payments/    — Razorpay escrow, milestones, invoices, disputes
```

These three apps form the **core business flow**. Every feature in the rest of the app depends on this chain:

```
Project (OPEN)
  └── Bid (PENDING → ACCEPTED)
        └── Contract (active)
              ├── Payment (ESCROWED → RELEASED)
              ├── WorkLog  (→ GUIDE_03)
              ├── Conversation (→ GUIDE_03)
              └── Notification (→ GUIDE_03)
```

---

## apps/projects — File reading order

```
models.py              → Project + ProjectSkill (the core data shapes)
models_extended.py     → ProjectBookmark, ProjectDraft, ProjectCategory, ProjectShare
signals.py             → none (ES sync happens in apps/search/signals.py)
selectors.py           → DB query functions (read layer)
services.py            → create/update/close project (write layer)
services_bookmark.py   → bookmark toggle
services_draft.py      → save/publish draft
services_category.py   → category management
services_share.py      → share link generation
serializers.py         → ProjectSerializer, ProjectListSerializer
serializers_extended.py → BriefProjectSerializer (for lists)
views.py               → CRUD endpoints
views_extended.py      → bookmarks, drafts, share endpoints
urls.py + urls_extended.py
```

---

## Step 16 — `apps/projects/models.py`

```python
class Project(models.Model):
    class Status(TextChoices):
        OPEN = "OPEN"
        IN_PROGRESS = "IN_PROGRESS"
        COMPLETED = "COMPLETED"
        CANCELLED = "CANCELLED"

    client = ForeignKey(User, limit_choices_to={'role': 'CLIENT'})
    title  = CharField(max_length=255)
    budget = DecimalField(max_digits=12, decimal_places=2)
    deadline = DateField(null=True, blank=True)
    status = CharField(choices=Status.choices, default=Status.OPEN)
```

`limit_choices_to={'role': 'CLIENT'}` — DB-level guard. Only CLIENT users can own projects.

```python
class ProjectSkill(models.Model):
    project    = ForeignKey(Project, related_name="skills")
    skill_name = CharField(max_length=100)

    class Meta:
        unique_together = ["project", "skill_name"]  # no duplicates
```

**Interview point:** "Why is ProjectSkill a separate model instead of a JSONField?"
- Enables filtering: `Project.objects.filter(skills__skill_name="React")`
- Elasticsearch indexes individual skills for faceted search
- `unique_together` enforces integrity at DB level

---

## Step 17 — `apps/projects/selectors.py`

```python
def get_projects_for_client(client_user, status=None):
    qs = Project.objects.filter(client=client_user).select_related("client")
    if status:
        qs = qs.filter(status=status)
    return qs.prefetch_related("skills")

def get_open_projects(filters=None):
    qs = Project.objects.filter(status="OPEN")
    ...
    return qs
```

**Pattern:** Selectors are **pure read functions** — no business logic, just queries. They always use `select_related` / `prefetch_related` to avoid N+1.

**Interview point:** "What is N+1 and how do you avoid it?"
Without `prefetch_related("skills")`, accessing `project.skills.all()` in a loop for 20 projects = 21 queries. With it = 2 queries total.

---

## Step 18 — `apps/projects/services.py`

```python
@transaction.atomic
def create_project(client: User, title, description, budget, skills, deadline=None):
    if client.role != User.Roles.CLIENT:
        raise PermissionDeniedError("Only clients can post projects.")

    project = Project.objects.create(
        client=client, title=title, description=description,
        budget=budget, deadline=deadline
    )
    for skill in skills:
        ProjectSkill.objects.get_or_create(project=project, skill_name=skill.strip())
    return project
```

**Pattern:**
1. Guard clause (role check) first — fail fast
2. Wrap in `transaction.atomic()` — project + skills created atomically
3. `get_or_create` on skills — idempotent, safe to call multiple times

---

## apps/bidding — File reading order

```
models.py              → Bid + Contract (core)
models_amendment.py    → ContractAmendment (scope change requests)
models_extended.py     → CounterOffer
models_review.py       → ContractReview (ratings after completion)
models_termination.py  → ContractTermination (dispute/early end)
selectors.py           → query functions
services.py            → submit_bid, accept_bid, reject_bid, withdraw_bid
services_amendment.py  → request/approve scope changes
services_counter_offer.py → negotiate bid amount
services_retraction.py → retract a bid before acceptance
services_review.py     → submit/respond to reviews, update ratings
services_termination.py → terminate contract with reason + evidence
services_worklog_approval.py → client approves/rejects daily work logs
serializers.py + serializers_extended.py + serializers_review.py
views.py + views_extended.py + views_review.py
urls.py + urls_extended.py + urls_review.py
```

---

## Step 19 — `apps/bidding/models.py`

### `Bid`
```python
class Bid(models.Model):
    class Status(TextChoices):
        PENDING   = "PENDING"
        ACCEPTED  = "ACCEPTED"
        REJECTED  = "REJECTED"
        WITHDRAWN = "WITHDRAWN"

    project    = ForeignKey(Project, related_name="bids")
    freelancer = ForeignKey(User, limit_choices_to={'role': 'FREELANCER'})
    amount     = DecimalField(max_digits=12, decimal_places=2)
    cover_letter = TextField()

    class Meta:
        unique_together = ["project", "freelancer"]  # one bid per freelancer per project
```

### `Contract`
```python
class Contract(models.Model):
    bid          = OneToOneField(Bid, related_name="contract")
    agreed_amount = DecimalField(...)
    end_date     = DateTimeField(null=True)
    is_active    = BooleanField(default=True)

    @property
    def project(self):   return self.bid.project
    @property
    def freelancer(self): return self.bid.freelancer
    @property
    def client(self):    return self.bid.project.client
```

**Interview points:**
- `OneToOneField` on Bid → Contract means one bid can produce exactly one contract
- Properties `project`, `freelancer`, `client` avoid deep chaining in views: `contract.client` vs `contract.bid.project.client`
- `unique_together = ["project", "freelancer"]` prevents a freelancer bidding twice on the same project

---

## Step 20 — `apps/bidding/services.py` (the lifecycle)

### `submit_bid()`
```python
def submit_bid(freelancer, project_id, amount, cover_letter):
    project = get_object_or_404(Project, id=project_id, status="OPEN")
    if Bid.objects.filter(project=project, freelancer=freelancer).exists():
        raise ValidationError("You already bid on this project.")
    bid = Bid.objects.create(project=project, freelancer=freelancer,
                              amount=amount, cover_letter=cover_letter)
    Notification.objects.create(recipient=project.client, ...)
    return bid
```

### `accept_bid()` — the most important function
```python
@transaction.atomic
def accept_bid(client, bid_id):
    bid = get_object_or_404(Bid, id=bid_id, project__client=client)
    if bid.status != Bid.Status.PENDING:
        raise BusinessError("Only pending bids can be accepted.")

    bid.status = Bid.Status.ACCEPTED
    bid.save()

    # Reject all other bids on the same project
    Bid.objects.filter(project=bid.project).exclude(id=bid.id).update(
        status=Bid.Status.REJECTED
    )

    # Create contract
    contract = Contract.objects.create(
        bid=bid,
        agreed_amount=bid.amount,
        end_date=timezone.now() + timedelta(days=30),
    )

    # Change project status
    bid.project.status = Project.Status.IN_PROGRESS
    bid.project.save()

    # Notify freelancer
    Notification.objects.create(recipient=bid.freelancer, ...)
    return contract
```

**Why `transaction.atomic()`?** If anything fails (e.g., contract creation), bid status rollback automatically. Database stays consistent.

**Bulk reject with `.update()`** — one SQL `UPDATE` instead of N individual saves.

---

## Step 21 — `apps/bidding/services_review.py`

After contract completion, both parties leave reviews:

```python
def submit_review(reviewer, contract_id, rating, comment):
    contract = get_object_or_404(Contract, id=contract_id, is_active=False)
    # Determine role
    if reviewer == contract.client:
        reviewee = contract.freelancer
    else:
        reviewee = contract.client

    review = ContractReview.objects.create(
        contract=contract, reviewer=reviewer,
        reviewee=reviewee, rating=rating, comment=comment
    )
    # Recalculate average rating
    _update_average_rating(reviewee)
    return review

def _update_average_rating(user):
    reviews = ContractReview.objects.filter(reviewee=user)
    avg = reviews.aggregate(avg=Avg('rating'))['avg'] or 0
    if user.role == FREELANCER:
        user.freelancer_profile.average_rating = avg
        user.freelancer_profile.save()
```

**Interview point:** "How do you keep denormalized data (average_rating) consistent?"
Recalculate from source of truth every time a review is added/edited. Never trust cached values for financial or trust-score data.

---

## Step 22 — `apps/bidding/services_termination.py`

Handles early contract termination:
```python
@transaction.atomic
def terminate_contract(initiator, contract_id, reason, evidence_urls):
    contract = get_object_or_404(Contract, id=contract_id, is_active=True)
    # Both parties can initiate
    if initiator not in [contract.client, contract.freelancer]:
        raise PermissionDeniedError()

    termination = ContractTermination.objects.create(
        contract=contract, initiated_by=initiator,
        reason=reason, evidence=evidence_urls
    )
    # Mark contract inactive
    contract.is_active = False
    contract.save()
    # Trigger dispute resolution flow
    _notify_support_team(contract, termination)
    return termination
```

---

## apps/payments — File reading order

```
models.py              → Payment, Escrow, PlatformEarning, PaymentEvent
models_milestone.py    → Milestone (phase-based payments)
models_dispute.py      → PaymentDispute
models_extended.py     → InvoiceRecord, TaxRecord
selectors.py           → query helpers
services.py            → create_payment_order, verify_payment, release_payment
services_milestone.py  → create/fund/release milestones
services_invoice.py    → generate invoice PDF
services_currency.py   → INR/USD conversion
services_tax.py        → GST/TDS calculation
tasks.py               → Celery tasks for webhook processing + payouts
views.py + views_extended.py
urls.py + urls_extended.py
```

---

## Step 23 — `apps/payments/models.py`

```python
class Payment(models.Model):
    class Status(TextChoices):
        PENDING        = "PENDING"
        ESCROWED       = "ESCROWED"       # client paid, held in escrow
        PAYOUT_PENDING = "PAYOUT_PENDING" # work approved, initiating payout
        RELEASED       = "RELEASED"       # freelancer received funds
        PAYOUT_FAILED  = "PAYOUT_FAILED"  # RazorpayX payout failed
        REFUNDED       = "REFUNDED"       # client refunded

    contract           = OneToOneField(Contract)
    total_amount       = DecimalField(...)
    razorpay_order_id  = CharField(...)  # from Razorpay order creation
    razorpay_payment_id = CharField(...) # from Razorpay payment capture
    razorpay_payout_id = CharField(...)  # from RazorpayX payout

class Escrow(models.Model):
    payment     = OneToOneField(Payment)
    held_amount = DecimalField(...)
    released_at = DateTimeField(null=True)  # set when payment released

class PlatformEarning(models.Model):
    payment        = ForeignKey(Payment)
    cut_percentage = DecimalField(...)  # 10%
    cut_amount     = DecimalField(...)  # computed amount

class PaymentEvent(models.Model):
    """Idempotency log for Razorpay webhooks."""
    payment           = ForeignKey(Payment)
    razorpay_event_id = CharField(unique=True)  # prevents double processing
    event_type        = CharField(...)
    processed_at      = DateTimeField(auto_now_add=True)
```

**Interview point:** "How do you handle duplicate webhooks?"
`PaymentEvent.razorpay_event_id` has `unique=True`. If Razorpay sends the same event twice, the second `INSERT` raises `IntegrityError` → caught → return 200 (idempotent response). Razorpay stops retrying.

---

## Step 24 — `apps/payments/services.py` (the full payment flow)

### Phase 1 — Client pays into escrow
```python
def create_payment_order(contract_id, client):
    contract = get_object_or_404(Contract, id=contract_id, is_active=True)
    # Create Razorpay order
    client = razorpay.Client(auth=(KEY_ID, KEY_SECRET))
    order = client.order.create({
        "amount": int(contract.agreed_amount * 100),  # in paise
        "currency": "INR",
        "receipt": f"contract_{contract.id}",
    })
    payment = Payment.objects.create(
        contract=contract,
        total_amount=contract.agreed_amount,
        razorpay_order_id=order['id'],
        status=Payment.Status.PENDING,
    )
    return payment, order
```

### Phase 2 — Razorpay webhook confirms payment
```python
def verify_and_escrow_payment(razorpay_order_id, razorpay_payment_id, signature):
    # HMAC-SHA256 signature verification
    params = f"{razorpay_order_id}|{razorpay_payment_id}"
    expected = hmac.new(KEY_SECRET.encode(), params.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise ValidationError("Invalid payment signature.")

    payment = Payment.objects.get(razorpay_order_id=razorpay_order_id)
    payment.razorpay_payment_id = razorpay_payment_id
    payment.status = Payment.Status.ESCROWED
    payment.save()
    Escrow.objects.create(payment=payment, held_amount=payment.total_amount)
```

**Security:** `hmac.compare_digest()` prevents timing attacks. Never use `==` for signature comparison.

### Phase 3 — Client approves work → release payment
```python
@transaction.atomic
def release_payment(client, payment_id):
    payment = Payment.objects.select_for_update().get(id=payment_id)
    if payment.status != Payment.Status.ESCROWED:
        raise BusinessError("Payment is not in escrow.")

    platform_cut = payment.total_amount * Decimal("0.10")
    freelancer_amount = payment.total_amount - platform_cut

    PlatformEarning.objects.create(
        payment=payment, cut_percentage=10, cut_amount=platform_cut
    )
    payment.status = Payment.Status.PAYOUT_PENDING
    payment.save()

    # Async Celery task for actual RazorpayX payout
    razorpay_transfer_to_freelancer_task.delay(payment.id, str(freelancer_amount))
```

**`select_for_update()`** — row-level DB lock. If two requests try to release the same payment simultaneously, one waits. Prevents double-payout.

---

## Step 25 — `apps/payments/tasks.py` (Celery background tasks)

```python
@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def razorpay_transfer_to_freelancer_task(self, payment_id, amount):
    try:
        payment = Payment.objects.get(id=payment_id)
        # Call RazorpayX API
        ...
        payment.status = Payment.Status.RELEASED
        payment.save()
    except razorpay.errors.BadRequestError as exc:
        payment.status = Payment.Status.PAYOUT_FAILED
        payment.payout_error = str(exc)
        payment.save()
        raise self.retry(exc=exc)  # retry up to 3 times
```

**Routed to `freelanceflow_high_priority` queue** — payment tasks must never wait behind slow PDF jobs.

**`bind=True`** — `self` is the task instance, needed for `self.retry()`.

**Interview point:** "What happens if the payout fails after 3 retries?"
`PAYOUT_FAILED` status is set. Admin sees it in Django admin. Support team manually resolves via Razorpay dashboard. `payout_error` field stores the error message for debugging.

---

## Step 26 — `apps/payments/services_milestone.py`

For large projects, payment is split into milestones:

```python
def create_milestone(contract, title, amount, due_date):
    total_milestones = Milestone.objects.filter(contract=contract)
    total_allocated = total_milestones.aggregate(Sum('amount'))['amount__sum'] or 0
    if total_allocated + amount > contract.agreed_amount:
        raise ValidationError("Milestone amounts exceed contract total.")
    return Milestone.objects.create(...)

def release_milestone(client, milestone_id):
    milestone = Milestone.objects.select_for_update().get(id=milestone_id)
    # Same ESCROWED → RELEASED flow as Payment, but per milestone
```

---

## Core Architecture Pattern Summary

```
Request
  ↓
View  (thin — validates HTTP, delegates to service)
  ↓
Serializer  (validates input shape)
  ↓
Service  (ALL business logic, raises BusinessError/ValidationError)
  ↓
Selector  (read) or Model.objects  (write) — inside transaction.atomic()
  ↓
Signal  (side effects — notifications, emails via Celery on_commit)
  ↓
Celery Task  (async — emails, payouts, PDFs)
```

**Why this matters:**
- Views never contain `if/else` business logic
- Services are testable without HTTP context
- Signals fire AFTER transaction commits (no race conditions)
- Celery tasks are retryable (no money lost on API failures)

---

## Key Interview Questions for This Section

**Q: How does the escrow flow work end-to-end?**
1. Client calls `create_payment_order` → Razorpay returns `order_id`
2. Client pays on frontend using Razorpay checkout
3. Razorpay sends webhook → `verify_and_escrow_payment` → `Payment.ESCROWED`
4. Freelancer does work, client approves worklogs
5. Client calls `release_payment` → `select_for_update` lock → `PAYOUT_PENDING`
6. Celery task calls RazorpayX API → `Payment.RELEASED`

**Q: What prevents a client from releasing the same payment twice?**
- `select_for_update()` — DB row lock during status check + update
- Status guard: `if payment.status != ESCROWED: raise BusinessError`
- Both checks inside `transaction.atomic()` — atomically checked and updated

**Q: How are duplicate Razorpay webhooks handled?**
`PaymentEvent` table with `unique=True` on `razorpay_event_id`. Second webhook → `IntegrityError` → caught → return HTTP 200 (idempotent).

---

> **Next:** Open `GUIDE_03_ADVANCED.md` → Worklogs → AI/WebSocket → Search → Celery
