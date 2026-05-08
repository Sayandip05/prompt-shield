from django.contrib import admin
from apps.notifications.models import Notification


@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    """Admin configuration for Notification model."""
    
    list_display = [
        "id",
        "recipient",
        "type",
        "title_preview",
        "is_read",
        "created_at"
    ]
    list_filter = ["type", "is_read", "created_at"]
    search_fields = ["recipient__email", "title", "body"]
    readonly_fields = ["created_at"]
    date_hierarchy = "created_at"
    
    fieldsets = (
        ("Notification Details", {
            "fields": ("recipient", "type", "title", "body")
        }),
        ("Status", {
            "fields": ("is_read",)
        }),
        ("Timestamps", {
            "fields": ("created_at",),
            "classes": ("collapse",)
        }),
    )
    
    def title_preview(self, obj):
        """Display truncated title."""
        return obj.title[:40] + "..." if len(obj.title) > 40 else obj.title
    title_preview.short_description = "Title"
