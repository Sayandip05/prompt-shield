#!/usr/bin/env python
"""
Script to generate all remaining service, serializer, view, and URL files
This creates the complete API layer for all extended models
"""

import os
from pathlib import Path

# Service templates for each app
SERVICES_TO_CREATE = {
    'apps/payments': [
        ('services_tax.py', '''"""
Tax Document Services
Generates tax documents (1099, etc.) for freelancers
"""
from django.db import transaction
from decimal import Decimal
from .models_extended import TaxDocument


@transaction.atomic
def generate_tax_document(user, year, document_type='1099'):
    """Generate tax document for a user"""
    from .models import Payment
    
    # Calculate total earnings for the year
    earnings = Payment.objects.filter(
        contract__bid__freelancer=user,
        status='RELEASED',
        created_at__year=year
    ).aggregate(total=models.Sum('total_amount'))['total'] or Decimal('0')
    
    # Create tax document
    tax_doc = TaxDocument.objects.create(
        user=user,
        year=year,
        document_type=document_type,
        total_earnings=earnings
    )
    
    # Generate PDF (implement PDF generation logic)
    # tax_doc.pdf_url = generate_pdf(tax_doc)
    # tax_doc.save()
    
    return tax_doc


def get_tax_documents(user, year=None):
    """Get tax documents for a user"""
    queryset = TaxDocument.objects.filter(user=user)
    if year:
        queryset = queryset.filter(year=year)
    return queryset.order_by('-year')
'''),
        ('services_currency.py', '''"""
Multi-Currency Services
Handles currency conversion and multi-currency payments
"""
from django.db import transaction
from decimal import Decimal
from .models_extended import CurrencyExchangeRate, MultiCurrencyPayment


def get_exchange_rate(from_currency, to_currency):
    """Get current exchange rate"""
    try:
        rate = CurrencyExchangeRate.objects.get(
            from_currency=from_currency,
            to_currency=to_currency
        )
        return rate.rate
    except CurrencyExchangeRate.DoesNotExist:
        return Decimal('1.0')


@transaction.atomic
def convert_currency(amount, from_currency, to_currency):
    """Convert amount from one currency to another"""
    if from_currency == to_currency:
        return amount
    
    rate = get_exchange_rate(from_currency, to_currency)
    return amount * rate


@transaction.atomic
def create_multi_currency_payment(payment, original_currency, original_amount):
    """Create multi-currency payment record"""
    converted_amount = convert_currency(original_amount, original_currency, 'USD')
    rate = get_exchange_rate(original_currency, 'USD')
    
    return MultiCurrencyPayment.objects.create(
        payment=payment,
        original_currency=original_currency,
        original_amount=original_amount,
        converted_currency='USD',
        converted_amount=converted_amount,
        exchange_rate=rate
    )
'''),
    ],
    'apps/projects': [
        ('services_bookmark.py', '''"""
Project Bookmark Services
"""
from django.db import transaction
from .models_extended import ProjectBookmark


@transaction.atomic
def bookmark_project(user, project):
    """Bookmark a project"""
    bookmark, created = ProjectBookmark.objects.get_or_create(
        user=user,
        project=project
    )
    return bookmark


@transaction.atomic
def remove_bookmark(user, project):
    """Remove project bookmark"""
    ProjectBookmark.objects.filter(user=user, project=project).delete()


def get_bookmarked_projects(user):
    """Get user's bookmarked projects"""
    return ProjectBookmark.objects.filter(user=user).select_related('project')


def is_bookmarked(user, project):
    """Check if project is bookmarked"""
    return ProjectBookmark.objects.filter(user=user, project=project).exists()
'''),
        ('services_category.py', '''"""
Project Category Services
"""
from django.db import transaction
from .models_extended import ProjectCategory


@transaction.atomic
def create_category(name, slug, description=None, icon=None):
    """Create a project category"""
    return ProjectCategory.objects.create(
        name=name,
        slug=slug,
        description=description,
        icon=icon
    )


def get_all_categories():
    """Get all project categories"""
    return ProjectCategory.objects.all().order_by('name')


def get_category_by_slug(slug):
    """Get category by slug"""
    return ProjectCategory.objects.get(slug=slug)
'''),
        ('services_draft.py', '''"""
Project Draft Services
"""
from django.db import transaction
from .models_extended import ProjectDraft


@transaction.atomic
def save_draft(client, title=None, description=None, budget=None, deadline=None, draft_data=None):
    """Save project as draft"""
    return ProjectDraft.objects.create(
        client=client,
        title=title,
        description=description,
        budget=budget,
        deadline=deadline,
        draft_data=draft_data or {}
    )


@transaction.atomic
def update_draft(draft_id, **kwargs):
    """Update draft"""
    draft = ProjectDraft.objects.get(id=draft_id)
    for key, value in kwargs.items():
        setattr(draft, key, value)
    draft.save()
    return draft


def get_user_drafts(client):
    """Get user's project drafts"""
    return ProjectDraft.objects.filter(client=client).order_by('-updated_at')


@transaction.atomic
def publish_draft(draft_id):
    """Convert draft to published project"""
    from .models import Project
    draft = ProjectDraft.objects.get(id=draft_id)
    
    project = Project.objects.create(
        client=draft.client,
        title=draft.title,
        description=draft.description,
        budget=draft.budget,
        deadline=draft.deadline
    )
    
    draft.delete()
    return project
'''),
        ('services_share.py', '''"""
Project Share Services
"""
from django.db import transaction
import secrets
from .models_extended import ProjectShare


@transaction.atomic
def generate_share_link(project, expires_at=None):
    """Generate public share link for project"""
    share_token = secrets.token_urlsafe(32)
    
    return ProjectShare.objects.create(
        project=project,
        share_token=share_token,
        expires_at=expires_at,
        is_active=True
    )


def get_project_by_token(token):
    """Get project by share token"""
    from django.utils import timezone
    
    share = ProjectShare.objects.select_related('project').get(
        share_token=token,
        is_active=True
    )
    
    # Check expiry
    if share.expires_at and share.expires_at < timezone.now():
        raise ValueError("Share link expired")
    
    # Increment view count
    share.view_count += 1
    share.save()
    
    return share.project


@transaction.atomic
def deactivate_share_link(share_id):
    """Deactivate share link"""
    share = ProjectShare.objects.get(id=share_id)
    share.is_active = False
    share.save()
'''),
    ],
    'apps/messaging': [
        ('services_search.py', '''"""
Message Search Services
"""
from django.db import transaction
from .models_extended import MessageSearch


@transaction.atomic
def index_message(message):
    """Index message for search"""
    from django.contrib.postgres.search import SearchVector
    
    MessageSearch.objects.create(
        message=message,
        conversation=message.conversation,
        search_vector=SearchVector('content')
    )


def search_messages(conversation_id, query):
    """Search messages in a conversation"""
    from django.contrib.postgres.search import SearchQuery
    
    search_query = SearchQuery(query)
    
    return MessageSearch.objects.filter(
        conversation_id=conversation_id,
        search_vector=search_query
    ).select_related('message')
'''),
        ('services_typing.py', '''"""
Typing Indicator Services
"""
from django.db import transaction
from .models_extended import TypingIndicator


@transaction.atomic
def set_typing(conversation_id, user, is_typing=True):
    """Set typing indicator"""
    indicator, created = TypingIndicator.objects.update_or_create(
        conversation_id=conversation_id,
        user=user,
        defaults={'is_typing': is_typing}
    )
    return indicator


def get_typing_users(conversation_id):
    """Get users currently typing"""
    return TypingIndicator.objects.filter(
        conversation_id=conversation_id,
        is_typing=True
    ).select_related('user')
'''),
    ],
    'apps/notifications': [
        ('services_digest.py', '''"""
Digest Email Services
"""
from django.db import transaction
from .models_extended import DigestEmail


@transaction.atomic
def create_digest_subscription(user, frequency='WEEKLY'):
    """Create digest email subscription"""
    return DigestEmail.objects.create(
        user=user,
        frequency=frequency,
        is_enabled=True
    )


def get_pending_digests():
    """Get digests that need to be sent"""
    from django.utils import timezone
    
    return DigestEmail.objects.filter(
        is_enabled=True,
        next_send_at__lte=timezone.now()
    )


@transaction.atomic
def send_digest(digest_id):
    """Send digest email"""
    from datetime import timedelta
    
    digest = DigestEmail.objects.get(id=digest_id)
    
    # Send email logic here
    
    # Update timestamps
    digest.last_sent_at = timezone.now()
    if digest.frequency == 'DAILY':
        digest.next_send_at = timezone.now() + timedelta(days=1)
    elif digest.frequency == 'WEEKLY':
        digest.next_send_at = timezone.now() + timedelta(weeks=1)
    digest.save()
'''),
        ('services_announcement.py', '''"""
System Announcement Services
"""
from django.db import transaction
from .models_extended import SystemAnnouncement


@transaction.atomic
def create_announcement(title, content, announcement_type='INFO', target_role=None, start_date=None, end_date=None):
    """Create system announcement"""
    return SystemAnnouncement.objects.create(
        title=title,
        content=content,
        type=announcement_type,
        target_role=target_role,
        start_date=start_date,
        end_date=end_date,
        is_active=True
    )


def get_active_announcements(user_role=None):
    """Get active announcements"""
    from django.utils import timezone
    now = timezone.now()
    
    queryset = SystemAnnouncement.objects.filter(
        is_active=True,
        start_date__lte=now
    ).filter(
        models.Q(end_date__isnull=True) | models.Q(end_date__gte=now)
    )
    
    if user_role:
        queryset = queryset.filter(
            models.Q(target_role=user_role) | models.Q(target_role__isnull=True)
        )
    
    return queryset.order_by('-start_date')
'''),
    ],
    'apps/search': [
        ('services_history.py', '''"""
Search History Services
"""
from django.db import transaction
from .models_extended import SearchHistory


@transaction.atomic
def log_search(user, query, filters=None, results_count=0):
    """Log a search query"""
    return SearchHistory.objects.create(
        user=user,
        query=query,
        filters=filters or {},
        results_count=results_count
    )


def get_search_history(user, limit=20):
    """Get user's search history"""
    return SearchHistory.objects.filter(user=user).order_by('-created_at')[:limit]


def get_popular_searches(limit=10):
    """Get most popular search queries"""
    from django.db.models import Count
    
    return SearchHistory.objects.values('query').annotate(
        count=Count('id')
    ).order_by('-count')[:limit]
'''),
        ('services_saved.py', '''"""
Saved Search Services
"""
from django.db import transaction
from .models_extended import SavedSearch


@transaction.atomic
def save_search(user, name, query, filters=None, notification_enabled=False):
    """Save a search query"""
    return SavedSearch.objects.create(
        user=user,
        name=name,
        query=query,
        filters=filters or {},
        notification_enabled=notification_enabled
    )


def get_saved_searches(user):
    """Get user's saved searches"""
    return SavedSearch.objects.filter(user=user).order_by('-created_at')


@transaction.atomic
def delete_saved_search(search_id, user):
    """Delete a saved search"""
    SavedSearch.objects.filter(id=search_id, user=user).delete()
'''),
        ('services_autocomplete.py', '''"""
Search Autocomplete Services
"""
from django.db import transaction
from .models_extended import SearchSuggestion


@transaction.atomic
def record_search_term(term, category=None):
    """Record search term for autocomplete"""
    suggestion, created = SearchSuggestion.objects.get_or_create(
        term=term.lower(),
        defaults={'category': category, 'frequency': 1}
    )
    
    if not created:
        suggestion.frequency += 1
        suggestion.save()
    
    return suggestion


def get_autocomplete_suggestions(query, limit=10):
    """Get autocomplete suggestions"""
    return SearchSuggestion.objects.filter(
        term__istartswith=query
    ).order_by('-frequency')[:limit]
'''),
    ],
    'apps/worklogs': [
        ('services_timeoff.py', '''"""
Time-off Tracking Services
"""
from django.db import transaction
from .models_extended import TimeOff


@transaction.atomic
def request_timeoff(freelancer, start_date, end_date, reason=None, contract=None):
    """Request time-off"""
    return TimeOff.objects.create(
        freelancer=freelancer,
        contract=contract,
        start_date=start_date,
        end_date=end_date,
        reason=reason,
        status=TimeOff.Status.PENDING
    )


@transaction.atomic
def approve_timeoff(timeoff_id, approver):
    """Approve time-off request"""
    timeoff = TimeOff.objects.get(id=timeoff_id)
    timeoff.status = TimeOff.Status.APPROVED
    timeoff.approved_by = approver
    timeoff.approved_at = timezone.now()
    timeoff.save()
    return timeoff


@transaction.atomic
def reject_timeoff(timeoff_id):
    """Reject time-off request"""
    timeoff = TimeOff.objects.get(id=timeoff_id)
    timeoff.status = TimeOff.Status.REJECTED
    timeoff.save()
    return timeoff


def get_pending_timeoffs(contract_id=None):
    """Get pending time-off requests"""
    queryset = TimeOff.objects.filter(status=TimeOff.Status.PENDING)
    if contract_id:
        queryset = queryset.filter(contract_id=contract_id)
    return queryset.order_by('-created_at')
'''),
    ],
}

def create_service_files():
    """Create all service files"""
    for app_path, services in SERVICES_TO_CREATE.items():
        for filename, content in services:
            filepath = Path(app_path) / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                f.write(content)
            
            print(f"✅ Created: {filepath}")

if __name__ == '__main__':
    print("🚀 Generating remaining service files...")
    create_service_files()
    print("\n✅ All service files created!")
    print("\n📋 Next steps:")
    print("1. Create serializers for all models")
    print("2. Create viewsets for all models")
    print("3. Configure URL patterns")
    print("4. Register models in admin")
    print("5. Generate migrations")
