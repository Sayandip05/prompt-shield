import stripe
from django.db import transaction
from django.conf import settings
from django.utils import timezone

from .models import Payment, Escrow, PlatformEarning, PaymentEvent
from apps.bidding.models import Contract
from apps.projects.services import mark_project_completed
from core.exceptions import ValidationError, PermissionDeniedError, NotFoundError
from core.utils import calculate_platform_cut


stripe.api_key = settings.STRIPE_SECRET_KEY


def create_escrow(contract: Contract, client) -> Payment:
    """
    Create escrow payment for a contract.
    Client pays full amount which is held in escrow.
    
    Args:
        contract: Contract instance
        client: User instance (must be contract client)
    
    Returns:
        Created Payment instance
    """
    if contract.client != client:
        raise PermissionDeniedError("Only the client can create escrow.")
    
    if hasattr(contract, 'payment'):
        raise ValidationError("Payment already exists for this contract.")
    
    with transaction.atomic():
        # Create Stripe PaymentIntent
        try:
            intent = stripe.PaymentIntent.create(
                amount=int(contract.agreed_amount * 100),  # Convert to cents
                currency='usd',
                metadata={
                    'contract_id': contract.id,
                    'project_title': contract.bid.project.title,
                },
            )
        except stripe.error.StripeError as e:
            raise ValidationError(f"Payment processing error: {str(e)}")
        
        # Create payment record
        payment = Payment.objects.create(
            contract=contract,
            total_amount=contract.agreed_amount,
            status=Payment.Status.PENDING,
            stripe_payment_intent_id=intent.id,
        )
        
        return payment


def confirm_escrow_payment(payment_intent_id: str) -> Payment:
    """
    Confirm that escrow payment has been received (called by webhook).
    
    Args:
        payment_intent_id: Stripe PaymentIntent ID
    
    Returns:
        Updated Payment instance
    """
    with transaction.atomic():
        try:
            payment = Payment.objects.select_for_update().get(
                stripe_payment_intent_id=payment_intent_id
            )
        except Payment.DoesNotExist:
            raise NotFoundError("Payment not found.")
        
        if payment.status != Payment.Status.PENDING:
            raise ValidationError("Payment is not pending.")
        
        # Update payment status
        payment.status = Payment.Status.ESCROWED
        payment.save()
        
        # Create escrow record
        Escrow.objects.create(
            payment=payment,
            held_amount=payment.total_amount,
        )
        
        return payment


def release_payment(contract: Contract, client) -> Payment:
    """
    Release payment to freelancer (minus platform cut).
    
    Args:
        contract: Contract instance
        client: User instance (must be contract client)
    
    Returns:
        Updated Payment instance
    """
    from apps.notifications.tasks import (
        notify_freelancer_payment_released,
        generate_delivery_proof,
    )
    
    if contract.client != client:
        raise PermissionDeniedError("Only the client can release payment.")
    
    with transaction.atomic():
        try:
            payment = Payment.objects.select_for_update().get(contract=contract)
        except Payment.DoesNotExist:
            raise NotFoundError("Payment not found.")
        
        if payment.status != Payment.Status.ESCROWED:
            raise ValidationError("Payment is not in escrow.")
        
        # Calculate platform cut
        cut_info = calculate_platform_cut(
            payment.total_amount,
            settings.PLATFORM_CUT_PERCENTAGE
        )
        
        # Create platform earning record
        PlatformEarning.objects.create(
            payment=payment,
            cut_percentage=cut_info['cut_percentage'],
            cut_amount=cut_info['cut_amount'],
        )
        
        # Update payment status
        payment.status = Payment.Status.RELEASED
        payment.save()
        
        # Update escrow record
        escrow = payment.escrow
        escrow.released_at = timezone.now()
        escrow.save()
        
        # Complete the contract
        contract.is_active = False
        contract.end_date = timezone.now()
        contract.save()
        
        # Mark project as completed
        mark_project_completed(contract.bid.project)
        
        # Schedule post-release tasks
        transaction.on_commit(lambda: [
            stripe_transfer_to_freelancer_task.delay(
                payment.id,
                float(cut_info['freelancer_amount'])
            ),
            notify_freelancer_payment_released.delay(contract.id),
            generate_delivery_proof.delay(contract.id),
        ])
        
        return payment


def process_stripe_webhook(payload: dict, sig_header: str) -> bool:
    """
    Process Stripe webhook event with HMAC verification.
    
    Args:
        payload: Request body
        sig_header: Stripe signature header
    
    Returns:
        True if processed successfully
    """
    from .tasks import process_stripe_webhook_task
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise ValidationError("Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise PermissionDeniedError("Invalid signature")
    
    # Check idempotency
    if has_payment_event_been_processed(event.id):
        return True
    
    # Process event asynchronously
    process_stripe_webhook_task.delay(event.id, event.type, event.data.object)
    
    return True


def has_payment_event_been_processed(stripe_event_id: str) -> bool:
    """Check if event has been processed."""
    return PaymentEvent.objects.filter(stripe_event_id=stripe_event_id).exists()


def record_payment_event(payment: Payment, stripe_event_id: str, event_type: str):
    """Record processed payment event for idempotency."""
    PaymentEvent.objects.create(
        payment=payment,
        stripe_event_id=stripe_event_id,
        event_type=event_type,
    )
