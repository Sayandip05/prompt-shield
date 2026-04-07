from rest_framework import serializers

from .models import WorkLog, WeeklyReport, DeliveryProof
from apps.users.serializers import UserSerializer
from apps.bidding.serializers import ContractSerializer


class WorkLogSerializer(serializers.ModelSerializer):
    """Serializer for work logs."""
    freelancer = UserSerializer(read_only=True)
    contract = ContractSerializer(read_only=True)
    
    class Meta:
        model = WorkLog
        fields = [
            'id',
            'contract',
            'freelancer',
            'date',
            'description',
            'hours_worked',
            'screenshot_url',
            'reference_url',
            'created_at',
            'updated_at',
        ]


class WorkLogCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating work logs."""
    
    class Meta:
        model = WorkLog
        fields = [
            'date',
            'description',
            'hours_worked',
            'screenshot_url',
            'reference_url',
        ]
    
    def validate_hours_worked(self, value):
        if value <= 0 or value > 24:
            raise serializers.ValidationError("Hours must be between 0 and 24.")
        return value
    
    def validate_description(self, value):
        if len(value.strip()) < 10:
            raise serializers.ValidationError(
                "Description must be at least 10 characters."
            )
        return value


class WorkLogUpdateSerializer(serializers.ModelSerializer):
    """Serializer for updating work logs."""
    
    class Meta:
        model = WorkLog
        fields = [
            'description',
            'hours_worked',
            'screenshot_url',
            'reference_url',
        ]
    
    def validate_hours_worked(self, value):
        if value is not None and (value <= 0 or value > 24):
            raise serializers.ValidationError("Hours must be between 0 and 24.")
        return value


class WeeklyReportSerializer(serializers.ModelSerializer):
    """Serializer for weekly reports."""
    contract = ContractSerializer(read_only=True)
    total_hours = serializers.DecimalField(
        max_digits=6,
        decimal_places=2,
        read_only=True
    )
    
    class Meta:
        model = WeeklyReport
        fields = [
            'id',
            'contract',
            'week_start',
            'week_end',
            'ai_summary',
            'pdf_url',
            'total_hours',
            'sent_to_client_at',
            'created_at',
        ]


class DeliveryProofSerializer(serializers.ModelSerializer):
    """Serializer for delivery proofs."""
    contract = ContractSerializer(read_only=True)
    
    class Meta:
        model = DeliveryProof
        fields = [
            'id',
            'contract',
            'pdf_url',
            'generated_at',
            'total_hours',
            'total_logs_count',
            'report_id',
        ]
