from rest_framework import status, viewsets, permissions
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import WorkLog, WeeklyReport, DeliveryProof
from .serializers import (
    WorkLogSerializer,
    WorkLogCreateSerializer,
    WorkLogUpdateSerializer,
    WeeklyReportSerializer,
    DeliveryProofSerializer,
)
from .services import (
    create_worklog,
    update_worklog,
    delete_worklog,
    generate_delivery_proof,
)
from .selectors import (
    get_worklog_by_id,
    get_contract_worklogs,
    get_contract_weekly_reports,
    get_delivery_proof_by_contract,
)
from .permissions import (
    IsWorkLogFreelancer,
    IsContractParticipant,
    IsContractFreelancer,
)
from apps.bidding.models import Contract
from core.exceptions import ValidationError


class WorkLogViewSet(viewsets.ModelViewSet):
    """
    ViewSet for WorkLog operations.
    
    Endpoints:
    - GET /api/worklogs/ - List work logs
    - POST /api/worklogs/ - Create work log (freelancer only)
    - GET /api/worklogs/{id}/ - Get work log detail
    - PATCH /api/worklogs/{id}/ - Update work log (freelancer only)
    - DELETE /api/worklogs/{id}/ - Delete work log (freelancer only)
    """
    
    def get_queryset(self):
        user = self.request.user
        contract_id = self.request.query_params.get('contract')
        
        if contract_id:
            return get_contract_worklogs(contract_id)
        
        if user.role == 'FREELANCER':
            return WorkLog.objects.filter(freelancer=user)
        
        return WorkLog.objects.filter(
            contract__bid__project__client=user
        )
    
    def get_serializer_class(self):
        if self.action == 'create':
            return WorkLogCreateSerializer
        elif self.action in ['update', 'partial_update']:
            return WorkLogUpdateSerializer
        return WorkLogSerializer
    
    def get_permissions(self):
        if self.action == 'create':
            return [permissions.IsAuthenticated()]
        elif self.action in ['update', 'partial_update', 'destroy']:
            return [permissions.IsAuthenticated(), IsWorkLogFreelancer()]
        return [permissions.IsAuthenticated(), IsContractParticipant()]
    
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        contract_id = request.query_params.get('contract')
        if not contract_id:
            return Response(
                {"error": "Contract ID required.", "code": "validation_error"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        
        try:
            log = create_worklog(
                freelancer=request.user,
                contract_id=int(contract_id),
                log_date=serializer.validated_data['date'],
                description=serializer.validated_data['description'],
                hours_worked=serializer.validated_data['hours_worked'],
                screenshot_url=serializer.validated_data.get('screenshot_url', ''),
                reference_url=serializer.validated_data.get('reference_url', ''),
            )
            
            return Response(
                WorkLogSerializer(log).data,
                status=status.HTTP_201_CREATED,
            )
        except ValidationError as e:
            return Response(
                {"error": e.message, "code": e.code, "field": e.field},
                status=status.HTTP_400_BAD_REQUEST,
            )
    
    def update(self, request, *args, **kwargs):
        partial = kwargs.pop('partial', False)
        log = self.get_object()
        serializer = self.get_serializer(data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        
        try:
            updated_log = update_worklog(
                log=log,
                freelancer=request.user,
                description=serializer.validated_data.get('description'),
                hours_worked=serializer.validated_data.get('hours_worked'),
                screenshot_url=serializer.validated_data.get('screenshot_url'),
                reference_url=serializer.validated_data.get('reference_url'),
            )
            
            return Response(
                WorkLogSerializer(updated_log).data,
                status=status.HTTP_200_OK,
            )
        except ValidationError as e:
            return Response(
                {"error": e.message, "code": e.code, "field": e.field},
                status=status.HTTP_400_BAD_REQUEST,
            )
    
    def destroy(self, request, *args, **kwargs):
        log = self.get_object()
        
        try:
            delete_worklog(log, request.user)
            return Response(
                {"message": "Work log deleted successfully."},
                status=status.HTTP_200_OK,
            )
        except ValidationError as e:
            return Response(
                {"error": e.message, "code": e.code, "field": e.field},
                status=status.HTTP_400_BAD_REQUEST,
            )


class WeeklyReportViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for WeeklyReport operations (read-only).
    """
    serializer_class = WeeklyReportSerializer
    permission_classes = [permissions.IsAuthenticated, IsContractParticipant]
    
    def get_queryset(self):
        contract_id = self.request.query_params.get('contract')
        
        if contract_id:
            return get_contract_weekly_reports(contract_id)
        
        user = self.request.user
        
        if user.role == 'FREELANCER':
            return WeeklyReport.objects.filter(
                contract__bid__freelancer=user
            )
        
        return WeeklyReport.objects.filter(
            contract__bid__project__client=user
        )


class DeliveryProofViewSet(viewsets.ViewSet):
    """
    ViewSet for DeliveryProof operations.
    """
    permission_classes = [permissions.IsAuthenticated, IsContractParticipant]
    
    def retrieve(self, request, pk=None):
        """Get delivery proof for a contract."""
        proof = get_delivery_proof_by_contract(pk)
        
        if not proof:
            return Response(
                {"error": "Delivery proof not found.", "code": "not_found"},
                status=status.HTTP_404_NOT_FOUND,
            )
        
        serializer = DeliveryProofSerializer(proof)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def generate(self, request, pk=None):
        """Generate delivery proof for a contract."""
        try:
            proof = generate_delivery_proof(pk)
            return Response(
                DeliveryProofSerializer(proof).data,
                status=status.HTTP_201_CREATED,
            )
        except ValidationError as e:
            return Response(
                {"error": e.message, "code": e.code, "field": e.field},
                status=status.HTTP_400_BAD_REQUEST,
            )
