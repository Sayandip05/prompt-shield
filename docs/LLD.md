# FreelanceFlow - Low-Level Design (LLD)

**Version:** 4.0.0  
**Last Updated:** May 1, 2026

---

## 1. Database Schema

### Core Tables

**Users & Auth**
- `users_user`: User accounts (email, password, role)
- `users_freelancerprofile`: Freelancer details (skills, rate, rating)
- `users_clientprofile`: Client details (company, spending)
- `users_twofactorauth`: 2FA settings
- `users_activitylog`: Audit trail
- `users_useronlinestatus`: Online/offline tracking

**Projects & Bidding**
- `projects_project`: Client job postings
- `projects_projectskill`: Required skills (M2M)
- `projects_projectcategory`: Categories
- `projects_projectbookmark`: User favorites
- `bidding_bid`: Freelancer proposals
- `bidding_contract`: Accepted bids
- `bidding_counteroffer`: Client counter-proposals
- `bidding_contractamendment`: Scope changes

**Payments**
- `payments_payment`: Payment records
- `payments_escrow`: Held funds
- `payments_platformearning`: Platform revenue (10%)
- `payments_paymentevent`: Webhook idempotency
- `payments_paymentmilestone`: Milestone-based payments
- `payments_paymentdispute`: Dispute resolution

**Work Tracking**
- `worklogs_worklog`: Daily work logs
- `worklogs_weeklyreport`: AI-generated summaries
- `worklogs_deliverable`: Submitted work items
- `worklogs_deliveryproof`: Final proof documents
- `worklogs_timeoff`: Leave tracking

**Communication**
- `messaging_conversation`: Chat per contract
- `messaging_message`: Individual messages
- `notifications_notification`: In-app notifications

**Search**
- `search_searchhistory`: User search history
- `search_savedsearch`: Saved searches

### Key Relationships

```
User (1) -> (1) FreelancerProfile
User (1) -> (1) ClientProfile
User (1) -> (M) Project (as client)
User (1) -> (M) Bid (as freelancer)

Project (1) -> (M) Bid
Bid (1) -> (1) Contract (when accepted)

Contract (1) -> (1) Payment
Contract (1) -> (M) WorkLog
Contract (1) -> (M) Deliverable
Contract (1) -> (1) Conversation

Payment (1) -> (1) Escrow
Payment (1) -> (M) PlatformEarning

Conversation (1) -> (M) Message
```

### Important Indexes

```python
# High-traffic queries
Project: ['status', 'created_at'], ['client', 'status']
Bid: ['project', 'status'], ['freelancer', 'status']
Contract: ['bid', 'is_active'], ['client', 'is_active']
Payment: ['contract'], ['status', 'created_at']
WorkLog: ['contract', 'date'], ['freelancer', 'date']
Message: ['conversation', 'created_at']
Notification: ['recipient', 'is_read']
```

---

## 2. API Endpoints

### Authentication (`/api/users/`)
- `POST /register/` - User registration
- `POST /login/` - Login (returns JWT)
- `POST /token/refresh/` - Refresh access token
- `GET /me/` - Get current user profile
- `POST /change-password/` - Change password
- `POST /password-reset/` - Request password reset
- `POST /verify-email/` - Verify email address

### Projects (`/api/projects/`)
- `GET /` - List projects (with filters)
- `POST /` - Create project
- `GET /{id}/` - Get project details
- `PATCH /{id}/` - Update project
- `DELETE /{id}/` - Delete project
- `POST /{id}/publish/` - Publish project

### Bidding (`/api/bidding/`)
- `GET /bids/` - List bids
- `POST /bids/` - Submit bid
- `GET /bids/{id}/` - Get bid details
- `POST /bids/{id}/accept/` - Accept bid (creates contract)
- `POST /bids/{id}/reject/` - Reject bid
- `DELETE /bids/{id}/` - Withdraw bid
- `GET /contracts/` - List contracts
- `GET /contracts/{id}/` - Get contract details
- `POST /contracts/{id}/complete/` - Mark contract complete

### Payments (`/api/payments/`)
- `GET /` - List payments
- `POST /escrow/` - Create escrow payment
- `POST /release/` - Release payment to freelancer
- `POST /verify/` - Verify Razorpay payment signature
- `POST /webhook/` - Razorpay webhook handler
- `GET /disputes/` - List disputes
- `POST /disputes/` - Create dispute

### Worklogs (`/api/worklogs/`)
- `GET /` - List worklogs
- `POST /` - Create worklog
- `GET /{id}/` - Get worklog details
- `PATCH /{id}/` - Update worklog
- `POST /deliverables/` - Create deliverable
- `POST /deliverables/{id}/submit/` - Submit for review
- `POST /deliverables/{id}/approve/` - Client approves
- `POST /deliverables/{id}/reject/` - Client rejects
- `POST /ai-chat/message/` - Send message to AI
- `POST /ai-chat/generate-deliverable/` - Generate from chat

### Messaging (`/api/messaging/`)
- `GET /conversations/` - List conversations
- `GET /conversations/{id}/` - Get conversation with messages
- `POST /messages/` - Send message
- `WebSocket /ws/chat/{contract_id}/` - Real-time chat

### Notifications (`/api/notifications/`)
- `GET /` - List notifications
- `PATCH /{id}/mark-read/` - Mark as read
- `DELETE /{id}/` - Delete notification

### Search (`/api/search/`)
- `GET /projects/` - Search projects (Elasticsearch)
- `GET /freelancers/` - Search freelancers

---

## 3. Key Modules & Responsibilities

### Service Layer Pattern

Each app follows this structure:

```
apps/{app_name}/
├── models.py          # Database models
├── serializers.py     # API serializers (validation)
├── views.py           # API views (thin controllers)
├── services.py        # Business logic (core)
├── selectors.py       # Read operations (queries)
├── permissions.py     # Authorization logic
├── tasks.py           # Celery async tasks
└── urls.py            # URL routing
```

**Responsibilities**:

| Module | Responsibility |
|--------|---------------|
| **models.py** | Database schema, relationships, constraints |
| **serializers.py** | Input validation, output formatting |
| **views.py** | HTTP handling, delegate to services |
| **services.py** | Business logic, transactions, side effects |
| **selectors.py** | Optimized read queries (select_related, prefetch_related) |
| **permissions.py** | Object-level authorization |
| **tasks.py** | Background jobs (emails, payouts, reports) |

---

## 4. Important Algorithms & Logic

### A. Payment Escrow Flow

```python
def create_escrow(contract, client):
    # 1. Verify client owns contract
    if contract.client != client:
        raise PermissionDeniedError()
    
    # 2. Create Razorpay order
    order = razorpay_client.order.create({
        'amount': int(contract.agreed_amount * 100),  # paise
        'currency': 'INR'
    })
    
    # 3. Create payment record (atomic)
    with transaction.atomic():
        payment = Payment.objects.create(
            contract=contract,
            total_amount=contract.agreed_amount,
            status='PENDING',
            razorpay_order_id=order['id']
        )
    
    return payment

def release_payment(contract, client):
    # 1. Verify and lock payment
    with transaction.atomic():
        payment = Payment.objects.select_for_update().get(
            contract=contract
        )
        
        if payment.status != 'ESCROWED':
            raise ValidationError()
        
        # 2. Calculate platform cut
        platform_cut = payment.total_amount * 0.10
        freelancer_amount = payment.total_amount - platform_cut
        
        # 3. Record platform earning
        PlatformEarning.objects.create(
            payment=payment,
            cut_amount=platform_cut
        )
        
        # 4. Update status
        payment.status = 'RELEASED'
        payment.save()
    
    # 5. Async payout to freelancer
    razorpay_transfer_to_freelancer_task.delay(
        payment.id, 
        freelancer_amount
    )
```

### B. Bid Acceptance Logic

```python
def accept_bid(bid, client):
    # 1. Verify client owns project
    if bid.project.client != client:
        raise PermissionDeniedError()
    
    # 2. Atomic transaction
    with transaction.atomic():
        # Lock bid
        bid = Bid.objects.select_for_update().get(id=bid.id)
        
        if bid.status != 'PENDING':
            raise ValidationError("Bid already processed")
        
        # Accept this bid
        bid.status = 'ACCEPTED'
        bid.save()
        
        # Reject all other bids
        Bid.objects.filter(
            project=bid.project,
            status='PENDING'
        ).update(status='REJECTED')
        
        # Create contract
        contract = Contract.objects.create(
            bid=bid,
            agreed_amount=bid.amount,
            start_date=timezone.now(),
            is_active=True
        )
        
        # Update project status
        bid.project.status = 'IN_PROGRESS'
        bid.project.save()
    
    # 3. Send notifications (async)
    notify_freelancer_bid_accepted.delay(contract.id)
    notify_other_freelancers_bid_rejected.delay(bid.project.id)
    
    return contract
```

### C. AI Worklog Generation (LangGraph)

```python
def generate_worklog_with_ai(messages, project_name):
    # 1. Create LangGraph workflow
    workflow = StateGraph(ChatAgentState)
    
    # 2. Add nodes
    workflow.add_node("process_message", process_message_node)
    workflow.add_node("check_intent", check_intent_node)
    
    # 3. Define flow
    workflow.set_entry_point("process_message")
    workflow.add_edge("process_message", "check_intent")
    workflow.add_conditional_edges(
        "check_intent",
        lambda state: state["next_action"],
        {
            "continue": END,
            "generate_report": END
        }
    )
    
    # 4. Execute
    graph = workflow.compile()
    result = graph.invoke({
        "messages": messages,
        "project_name": project_name,
        "report_ready": False
    })
    
    # 5. Extract structured data
    if result["report_ready"]:
        return result["report_data"]  # JSON with hours, tasks, etc.
```

### D. Webhook Idempotency

```python
def process_razorpay_webhook(payload, signature):
    # 1. Verify signature
    razorpay_client.utility.verify_webhook_signature(
        payload, 
        signature, 
        WEBHOOK_SECRET
    )
    
    event_id = payload['event']
    
    # 2. Check if already processed (idempotency)
    if PaymentEvent.objects.filter(
        razorpay_event_id=event_id
    ).exists():
        return  # Already processed, skip
    
    # 3. Process event
    with transaction.atomic():
        payment = Payment.objects.get(
            razorpay_payment_id=payload['payment_id']
        )
        
        # Update payment status
        payment.status = 'COMPLETED'
        payment.save()
        
        # Record event (prevents duplicate processing)
        PaymentEvent.objects.create(
            payment=payment,
            razorpay_event_id=event_id,
            event_type=payload['event']
        )
```

---

## 5. Design Patterns Used

### A. Service Layer Pattern

**Why**: Separate business logic from HTTP handling

```python
# views.py (thin controller)
class ProjectCreateView(APIView):
    def post(self, request):
        serializer = ProjectSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # Delegate to service
        project = create_project(
            client=request.user,
            **serializer.validated_data
        )
        
        return Response(
            ProjectSerializer(project).data,
            status=201
        )

# services.py (business logic)
def create_project(client, **data):
    with transaction.atomic():
        project = Project.objects.create(
            client=client,
            **data
        )
        
        # Side effects
        index_project_in_elasticsearch(project)
        notify_matching_freelancers(project)
        
        return project
```

### B. Repository Pattern (Selectors)

**Why**: Optimize read queries, separate from write operations

```python
# selectors.py
def get_project_with_bids(project_id):
    return Project.objects.select_related(
        'client',
        'category'
    ).prefetch_related(
        'skills',
        'bids__freelancer__profile'
    ).get(id=project_id)

def get_active_contracts_for_freelancer(freelancer):
    return Contract.objects.filter(
        bid__freelancer=freelancer,
        is_active=True
    ).select_related(
        'bid__project__client'
    ).order_by('-start_date')
```

### C. Strategy Pattern (Payment Gateways)

**Why**: Support multiple payment providers

```python
class PaymentGateway(ABC):
    @abstractmethod
    def create_order(self, amount, currency):
        pass
    
    @abstractmethod
    def verify_payment(self, payment_id, signature):
        pass

class RazorpayGateway(PaymentGateway):
    def create_order(self, amount, currency):
        return razorpay_client.order.create(...)
    
    def verify_payment(self, payment_id, signature):
        return verify_razorpay_signature(...)

# Usage
gateway = RazorpayGateway()  # Can swap with StripeGateway
order = gateway.create_order(5000, 'INR')
```

### D. Observer Pattern (Django Signals)

**Why**: Loose coupling between modules

```python
# bidding/signals.py
bid_accepted = Signal()

@receiver(bid_accepted)
def create_escrow_on_bid_accepted(sender, contract, **kwargs):
    # Automatically create escrow when bid accepted
    create_escrow(contract, contract.client)

@receiver(bid_accepted)
def notify_freelancer(sender, contract, **kwargs):
    send_notification(
        contract.freelancer,
        "Your bid was accepted!"
    )

# Usage in service
bid_accepted.send(sender=Bid, contract=contract)
```

### E. Factory Pattern (Serializers)

**Why**: Dynamic serializer selection based on context

```python
def get_serializer_class(self):
    if self.action == 'list':
        return ProjectListSerializer  # Minimal fields
    elif self.action == 'retrieve':
        return ProjectDetailSerializer  # Full details
    return ProjectSerializer
```

---

## 6. Component Diagram (C4 Level 3)

```
+----------------------------------------------------------+
|                    Django Application                     |
+----------------------------------------------------------+
|                                                          |
|  +----------------------------------------------------+  |
|  |                  API Layer (Views)                 |  |
|  |  - Authentication (JWT)                            |  |
|  |  - Request validation                              |  |
|  |  - Response formatting                             |  |
|  +----------------------------------------------------+  |
|                          |                               |
|                          v                               |
|  +----------------------------------------------------+  |
|  |              Service Layer (Business Logic)        |  |
|  |                                                    |  |
|  |  +---------------+  +---------------+              |  |
|  |  | User Service  |  |Project Service|              |  |
|  |  | - Register    |  | - Create      |              |  |
|  |  | - Login       |  | - Publish     |              |  |
|  |  +---------------+  +---------------+              |  |
|  |                                                    |  |
|  |  +---------------+  +---------------+              |  |
|  |  | Bid Service   |  |Payment Service|              |  |
|  |  | - Submit      |  | - Escrow      |              |  |
|  |  | - Accept      |  | - Release     |              |  |
|  |  +---------------+  +---------------+              |  |
|  |                                                    |  |
|  |  +---------------+  +---------------+              |  |
|  |  |Worklog Service|  |Message Service|              |  |
|  |  | - Create      |  | - Send        |              |  |
|  |  | - AI Generate |  | - Broadcast   |              |  |
|  |  +---------------+  +---------------+              |  |
|  +----------------------------------------------------+  |
|                          |                               |
|                          v                               |
|  +----------------------------------------------------+  |
|  |              Data Access Layer (ORM)               |  |
|  |  - Models                                          |  |
|  |  - Selectors (optimized queries)                  |  |
|  |  - Transactions                                    |  |
|  +----------------------------------------------------+  |
|                          |                               |
+----------------------------------------------------------+
                           |
        +------------------+------------------+
        |                  |                  |
        v                  v                  v
+---------------+  +---------------+  +---------------+
|  PostgreSQL   |  |     Redis     |  |Elasticsearch  |
|  - User data  |  |  - Cache      |  |  - Projects   |
|  - Projects   |  |  - Sessions   |  |  - Freelancers|
|  - Payments   |  |  - Celery     |  |  - Full-text  |
+---------------+  +---------------+  +---------------+
```

---

## 7. Edge Cases Handled

### A. Concurrency Issues

**Problem**: Two clients accept same bid simultaneously

**Solution**: Database row locking
```python
with transaction.atomic():
    bid = Bid.objects.select_for_update().get(id=bid_id)
    if bid.status != 'PENDING':
        raise ValidationError("Already processed")
    # Process...
```

### B. Payment Idempotency

**Problem**: Webhook received multiple times

**Solution**: Event tracking
```python
if PaymentEvent.objects.filter(
    razorpay_event_id=event_id
).exists():
    return  # Skip duplicate
```

### C. Partial Failures

**Problem**: Payment released but notification fails

**Solution**: Async tasks with retry
```python
transaction.on_commit(lambda: 
    notify_freelancer.apply_async(
        args=[freelancer_id],
        retry=True,
        max_retries=3
    )
)
```

### D. Race Conditions

**Problem**: Multiple bids accepted on same project

**Solution**: Atomic transaction + status check
```python
with transaction.atomic():
    # Lock project
    project = Project.objects.select_for_update().get(id=project_id)
    
    if project.status != 'OPEN':
        raise ValidationError("Project no longer open")
    
    # Accept bid and update project
    # ...
```

### E. Stale Data

**Problem**: User sees outdated project status

**Solution**: Cache invalidation
```python
def update_project(project_id, **data):
    project = Project.objects.get(id=project_id)
    project.update(**data)
    
    # Invalidate cache
    cache.delete(f'project:{project_id}')
    
    # Reindex in Elasticsearch
    index_project(project)
```

### F. Large File Uploads

**Problem**: Memory issues with large screenshots

**Solution**: Streaming upload to S3
```python
def upload_screenshot(file):
    # Stream directly to S3 (no memory buffering)
    s3_client.upload_fileobj(
        file,
        bucket_name,
        key,
        Config=TransferConfig(
            multipart_threshold=5 * 1024 * 1024  # 5MB
        )
    )
```

### G. Timezone Issues

**Problem**: Worklog dates in different timezones

**Solution**: Store UTC, convert on display
```python
# Always store in UTC
worklog.date = timezone.now()

# Convert to user timezone on serialization
class WorkLogSerializer(serializers.ModelSerializer):
    date = serializers.DateTimeField(
        default_timezone=user.timezone
    )
```

### H. Soft Deletes

**Problem**: Need to preserve data for audit

**Solution**: Soft delete pattern
```python
class Project(models.Model):
    is_deleted = models.BooleanField(default=False)
    deleted_at = models.DateTimeField(null=True)
    
    objects = ProjectManager()  # Excludes deleted
    all_objects = models.Manager()  # Includes deleted

def delete_project(project):
    project.is_deleted = True
    project.deleted_at = timezone.now()
    project.save()
```

---

**Document Status**: Complete  
**Last Updated**: May 1, 2026