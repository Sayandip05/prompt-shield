from django.db.models import QuerySet, Sum
from django.shortcuts import get_object_or_404
from datetime import date, timedelta

from .models import WorkLog, WeeklyReport, DeliveryProof


def get_worklog_by_id(log_id: int) -> WorkLog:
    """Get work log by ID."""
    return get_object_or_404(WorkLog, id=log_id)


def get_contract_worklogs(
    contract_id: int,
    start_date: date | None = None,
    end_date: date | None = None
) -> QuerySet[WorkLog]:
    """
    Get work logs for a contract with optional date filtering.
    """
    queryset = WorkLog.objects.filter(
        contract_id=contract_id
    ).select_related('freelancer', 'contract__bid__project')
    
    if start_date:
        queryset = queryset.filter(date__gte=start_date)
    if end_date:
        queryset = queryset.filter(date__lte=end_date)
    
    return queryset


def get_freelancer_worklogs(freelancer, contract_id: int | None = None) -> QuerySet[WorkLog]:
    """Get work logs for a freelancer."""
    queryset = WorkLog.objects.filter(freelancer=freelancer)
    
    if contract_id:
        queryset = queryset.filter(contract_id=contract_id)
    
    return queryset.select_related('contract__bid__project')


def get_weekly_report_by_id(report_id: int) -> WeeklyReport:
    """Get weekly report by ID."""
    return get_object_or_404(WeeklyReport, id=report_id)


def get_contract_weekly_reports(contract_id: int) -> QuerySet[WeeklyReport]:
    """Get all weekly reports for a contract."""
    return WeeklyReport.objects.filter(
        contract_id=contract_id
    ).select_related('contract__bid__project')


def get_delivery_proof_by_contract(contract_id: int) -> DeliveryProof | None:
    """Get delivery proof for a contract."""
    try:
        return DeliveryProof.objects.get(contract_id=contract_id)
    except DeliveryProof.DoesNotExist:
        return None


def get_total_hours_for_contract(contract_id: int) -> float:
    """Get total hours logged for a contract."""
    result = WorkLog.objects.filter(
        contract_id=contract_id
    ).aggregate(total=Sum('hours_worked'))
    return result['total'] or 0


def get_week_logs(contract_id: int, week_start: date) -> QuerySet[WorkLog]:
    """Get logs for a specific week."""
    week_end = week_start + timedelta(days=6)
    return WorkLog.objects.filter(
        contract_id=contract_id,
        date__range=[week_start, week_end]
    ).order_by('date')


def get_total_hours_for_week(contract_id: int, week_start: date) -> float:
    """Get total hours for a specific week."""
    week_end = week_start + timedelta(days=6)
    result = WorkLog.objects.filter(
        contract_id=contract_id,
        date__range=[week_start, week_end]
    ).aggregate(total=Sum('hours_worked'))
    return result['total'] or 0


def has_log_for_date(contract_id: int, log_date: date) -> bool:
    """Check if a log already exists for a date."""
    return WorkLog.objects.filter(
        contract_id=contract_id,
        date=log_date
    ).exists()
