"""
Structured logging for the Messaging app.
"""
import logging

logger = logging.getLogger("apps.messaging")


def log_conversation_created(conversation):
    logger.info(
        "Conversation created: id=%s contract=%s",
        conversation.id, conversation.contract_id,
    )


def log_message_sent(message):
    logger.info(
        "Message sent: id=%s conversation=%s sender=%s",
        message.id, message.conversation_id, message.sender_id,
    )


def log_websocket_connected(user_id, conversation_id):
    logger.info("WebSocket connected: user=%s conversation=%s", user_id, conversation_id)


def log_websocket_disconnected(user_id, conversation_id):
    logger.info("WebSocket disconnected: user=%s conversation=%s", user_id, conversation_id)


def log_websocket_error(user_id, error):
    logger.error("WebSocket error: user=%s error=%s", user_id, str(error))
