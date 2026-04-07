from django.db.models import Sum, Q
from rest_framework import status, viewsets, permissions
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.response import Response

from .models import Payment
from .serializers import (
    PaymentSerializer,
    PaymentListSerializer,
    CreateEscrowSerializer,
    ReleasePaymentSerializer,
    PaymentHistorySerializer,
)
from .services import create_escrow, release_payment, process_stripe_webhook
from .selectors import (
    get_payment_by_id,
    get_payment_by_contract,
    get_client_payment_history,
    get_freelancer_earnings,
    get_client_total_spent,
    get_freelancer_total_earned,
)
from .permissions import IsPaymentParticipant, IsPaymentClient
from apps.bidding.models import Contract
from core.exceptions import ValidationError


class PaymentViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for Payment operations.
    
    Endpoints:
    - GET /api/payments/ - List user's payments
    - GET /api/payments/{id}/ - Get payment detail
    - POST /api/payments/escrow/ - Create escrow (client only)
    - POST /api/payments/release/ - Release payment (client only)
    """
    
    def get_queryset(self):
        user = self.request.user
        
        if user.role == 'CLIENT':
            return get_client_payment_history(user)
        else:
            return get_freelancer_earnings(user)
    
    def get_serializer_class(self):
        if self.action == 'list':
            return PaymentListSerializer
        return PaymentSerializer
    
    permission_classes = [permissions.IsAuthenticated, IsPaymentParticipant]
    
    @action(detail=False, methods=['post'])
    def escrow(self, request):
        """Create escrow for a contract (client only)."""
        serializer = CreateEscrowSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        try:
            contract = Contract.objects.get(
                id=serializer.validated_data['contract_id']
            )
        except Contract.DoesNotExist:
            return Response(
                {"error": "Contract not found.", "code": "not_found"},
                status=status.HTTP_404_NOT_FOUND,
            )
        
        try:
            payment = create_escrow(contract, request.user)
            return Response(
                {
                    "message": "Escrow created successfully.",
                    "payment": PaymentSerializer(payment).data,
                    "client_secret": payment.stripe_payment_intent_id,
                },
                status=status.HTTP_201_CREATED,
            )
        except ValidationError as e:
            return Response(
                {"error": e.message, "code": e.code, "field": e.field},
                status=status.HTTP_400_BAD_REQUEST,
            )
    
    @action(detail=False, methods=['post'])
    def release(self, request):
        """Release payment to freelancer (client only)."""
        serializer = ReleasePaymentSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        try:
            contract = Contract.objects.get(
                id=serializer.validated_data['contract_id']
            )
        except Contract.DoesNotExist:
            return Response(
                {"error": "Contract not found.", "code": "not_found"},
                status=status.HTTP_404_NOT_FOUND,
            )
        
        try:
            payment = release_payment(contract, request.user)
            return Response(
                {
                    "message": "Payment released successfully.",
                    "payment": PaymentSerializer(payment).data,
                },
                status=status.HTTP_200_OK,
            )
        except ValidationError as e:
            return Response(
                {"error": e.message, "code": e.code, "field": e.field},
                status=status.HTTP_400_BAD_REQUEST,
            )
    
    @action(detail=False, methods=['get'])
    def history(self, request):
        """Get payment history summary."""
        user = request.user
        
        if user.role == 'CLIENT':
            total_spent = get_client_total_spent(user)
            pending_escrow = Payment.objects.filter(
                contract__bid__project__client=user,
                status=Payment.Status.ESCROWED
            ).aggregate(total=Sum('total_amount'))['total'] or 0
            
            data = {
                'total_spent': total_spent,
                'total_earned': 0,
                'pending_escrow': pending_escrow,
            }
        else:
            total_earned = get_freelancer_total_earned(user)
            
            data = {
                'total_spent': 0,
                'total_earned': total_earned,
                'pending_escrow': 0,
            }
        
        serializer = PaymentHistorySerializer(data)
        return Response(serializer.data)


@api_view(['POST'])
@permission_classes([permissions.AllowAny])
def stripe_webhook(request):
    """
    Stripe webhook endpoint for payment events.
    """
    payload = request.body
    sig_header = request.headers.get('Stripe-Signature')
    
    try:
        process_stripe_webhook(payload, sig_header)
        return Response({"status": "success"}, status=status.HTTP_200_OK)
    except ValidationError as e:
        return Response(
            {"error": e.message, "code": e.code},
            status=status.HTTP_400_BAD_REQUEST,
        )
    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_400_BAD_REQUEST,
        )
