from django.db import transaction

from .models import Conversation, Message
from apps.bidding.models import Contract
from core.exceptions import ValidationError, PermissionDeniedError, NotFoundError


def get_or_create_conversation(contract_id: int) -> Conversation:
    """Get or create conversation for a contract."""
    try:
        contract = Contract.objects.get(id=contract_id)
    except Contract.DoesNotExist:
        raise NotFoundError("Contract not found.")
    
    conversation, created = Conversation.objects.get_or_create(
        contract=contract
    )
    
    return conversation


def send_message(
    sender,
    conversation_id: int,
    content: str
) -> Message:
    """
    Send a message in a conversation.
    """
    if not content or not content.strip():
        raise ValidationError("Message content is required.", field="content")
    
    try:
        conversation = Conversation.objects.get(id=conversation_id)
    except Conversation.DoesNotExist:
        raise NotFoundError("Conversation not found.")
    
    # Verify sender is part of the conversation
    contract = conversation.contract
    if sender not in [contract.bid.freelancer, contract.bid.project.client]:
        raise PermissionDeniedError(
            "You are not a participant in this conversation."
        )
    
    with transaction.atomic():
        message = Message.objects.create(
            conversation=conversation,
            sender=sender,
            content=content.strip(),
        )
        
        # Update conversation timestamp
        conversation.save()
        
        return message


def mark_messages_as_read(conversation_id: int, user) -> int:
    """
    Mark all unread messages in a conversation as read for a user.
    """
    count = Message.objects.filter(
        conversation_id=conversation_id,
        is_read=False
    ).exclude(sender=user).update(is_read=True)
    
    return count
