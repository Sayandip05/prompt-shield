from django.db import models
from django.conf import settings

from apps.bidding.models import Contract


class WorkLog(models.Model):
    """
    Daily work log submitted by freelancers.
    """
    contract = models.ForeignKey(
        Contract,
        on_delete=models.CASCADE,
        related_name="work_logs"
    )
    freelancer = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="work_logs"
    )
    date = models.DateField()
    description = models.TextField(help_text="What was done today")
    hours_worked = models.DecimalField(
        max_digits=4,
        decimal_places=2,
        help_text="Hours worked (e.g., 7.5)"
    )
    screenshot_url = models.URLField(
        blank=True,
        help_text="Optional screenshot proof"
    )
    reference_url = models.URLField(
        blank=True,
        help_text="Optional reference link (GitHub, Figma, etc.)"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = "work_logs"
        ordering = ["-date", "-created_at"]
        unique_together = ["contract", "date"]
    
    def __str__(self):
        return f"Log for {self.contract.bid.project.title} on {self.date}"


class WeeklyReport(models.Model):
    """
    AI-generated weekly progress report.
    """
    contract = models.ForeignKey(
        Contract,
        on_delete=models.CASCADE,
        related_name="weekly_reports"
    )
    week_start = models.DateField()
    week_end = models.DateField()
    ai_summary = models.TextField(help_text="AI-generated report content")
    pdf_url = models.URLField(
        blank=True,
        help_text="S3 URL to generated PDF"
    )
    sent_to_client_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = "weekly_reports"
        ordering = ["-week_start"]
        unique_together = ["contract", "week_start"]
    
    def __str__(self):
        return f"Report for week {self.week_start} to {self.week_end}"
    
    @property
    def total_hours(self):
        """Calculate total hours for this week."""
        from django.db.models import Sum
        result = WorkLog.objects.filter(
            contract=self.contract,
            date__range=[self.week_start, self.week_end]
        ).aggregate(total=Sum('hours_worked'))
        return result['total'] or 0


class DeliveryProof(models.Model):
    """
    Final proof of delivery document generated at project completion.
    """
    contract = models.OneToOneField(
        Contract,
        on_delete=models.CASCADE,
        related_name="delivery_proof"
    )
    pdf_url = models.URLField(
        help_text="S3 URL to generated PDF"
    )
    generated_at = models.DateTimeField(auto_now_add=True)
    total_hours = models.DecimalField(max_digits=10, decimal_places=2)
    total_logs_count = models.IntegerField()
    report_id = models.CharField(
        max_length=100,
        unique=True,
        help_text="Unique tamper-evident report ID"
    )
    
    class Meta:
        db_table = "delivery_proofs"
    
    def __str__(self):
        return f"Proof for {self.contract.bid.project.title}"
