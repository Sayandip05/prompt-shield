from celery import shared_task
import razorpay
from django.conf import settings

from .models import Payment, PaymentEvent
from .services import confirm_escrow_payment, record_payment_event

razorpay_client = razorpay.Client(auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET))


@shared_task
def process_razorpay_webhook_task(event_id: str, event_type: str, event_data: dict):
    """
    Process Razorpay webhook event asynchronously.
    """
    from .services import has_payment_event_been_processed
    
    # Double-check idempotency
    if has_payment_event_been_processed(event_id):
        return
    
    if event_type == 'payment.captured':
        payment_entity = event_data.get('payment', {}).get('entity', {})
        razorpay_order_id = payment_entity.get('order_id')
        razorpay_payment_id = payment_entity.get('id')
        
        try:
            payment = confirm_escrow_payment(razorpay_order_id, razorpay_payment_id)
            record_payment_event(payment, event_id, event_type)
        except Exception as e:
            # Log error but don't raise (webhook should return 200)
            print(f"Error processing payment.captured: {e}")
    
    elif event_type == 'payment.failed':
        # Handle failed payment
        payment_entity = event_data.get('payment', {}).get('entity', {})
        razorpay_order_id = payment_entity.get('order_id')
        # TODO: Notify client of failed payment
        print(f"Payment failed for order: {razorpay_order_id}")


@shared_task
def razorpay_transfer_to_freelancer_task(payment_id: int, amount: float):
    """
    Transfer funds to freelancer using Razorpay Route/Payout.
    
    Args:
        payment_id: Payment ID
        amount: Amount to transfer
    """
    from apps.bidding.models import Contract
    
    try:
        payment = Payment.objects.get(id=payment_id)
        contract = payment.contract
        freelancer = contract.bid.freelancer
        
        # TODO: Implement Razorpay Route or Payout API
        # This requires freelancer's bank account details or UPI
        # For now, just log the transfer
        print(f"Would transfer ₹{amount} to freelancer {freelancer.email}")
        
        # Example payout code (requires Razorpay X account):
        # payout = razorpay_client.payout.create({
        #     'account_number': settings.RAZORPAY_ACCOUNT_NUMBER,
        #     'amount': int(amount * 100),
        #     'currency': 'INR',
        #     'mode': 'IMPS',
        #     'purpose': 'payout',
        #     'fund_account_id': freelancer_fund_account_id,
        #     'queue_if_low_balance': True,
        #     'reference_id': f'contract_{contract.id}',
        # })
        
    except Payment.DoesNotExist:
        print(f"Payment {payment_id} not found")
    except Exception as e:
        print(f"Error transferring funds: {e}")
