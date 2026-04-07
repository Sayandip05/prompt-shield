from celery import shared_task
import stripe
from django.conf import settings

from .models import Payment, PaymentEvent
from .services import confirm_escrow_payment, record_payment_event

stripe.api_key = settings.STRIPE_SECRET_KEY


@shared_task
def process_stripe_webhook_task(event_id: str, event_type: str, event_data: dict):
    """
    Process Stripe webhook event asynchronously.
    """
    from .services import has_payment_event_been_processed
    
    # Double-check idempotency
    if has_payment_event_been_processed(event_id):
        return
    
    if event_type == 'payment_intent.succeeded':
        payment_intent_id = event_data.get('id')
        try:
            payment = confirm_escrow_payment(payment_intent_id)
            record_payment_event(payment, event_id, event_type)
        except Exception as e:
            # Log error but don't raise (webhook should return 200)
            print(f"Error processing payment_intent.succeeded: {e}")
    
    elif event_type == 'payment_intent.payment_failed':
        # Handle failed payment
        payment_intent_id = event_data.get('id')
        # TODO: Notify client of failed payment
        pass


@shared_task
def stripe_transfer_to_freelancer_task(payment_id: int, amount: float):
    """
    Transfer funds to freelancer's Stripe Connect account.
    
    Args:
        payment_id: Payment ID
        amount: Amount to transfer
    """
    from apps.bidding.models import Contract
    
    try:
        payment = Payment.objects.get(id=payment_id)
        contract = payment.contract
        freelancer = contract.bid.freelancer
        
        # TODO: Get freelancer's Stripe Connect account ID
        # This requires Stripe Connect setup
        # For now, just log the transfer
        print(f"Would transfer ${amount} to freelancer {freelancer.email}")
        
        # Example transfer code (requires Stripe Connect):
        # stripe.Transfer.create(
        #     amount=int(amount * 100),
        #     currency='usd',
        #     destination=freelancer_stripe_account_id,
        #     transfer_group=f'contract_{contract.id}',
        # )
        
    except Payment.DoesNotExist:
        print(f"Payment {payment_id} not found")
    except Exception as e:
        print(f"Error transferring funds: {e}")
