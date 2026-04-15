"""
Structured logging for the Notifications app.
"""
import logging

logger = logging.getLogger("apps.notifications")


def log_notification_created(notification):
    logger.info(
        "Notification created: id=%s type=%s recipient=%s",
        notification.id, notification.type, notification.recipient_id,
    )


def log_notification_read(notification):
    logger.info("Notification read: id=%s recipient=%s", notification.id, notification.recipient_id)


def log_notification_email_sent(notification):
    logger.info(
        "Notification email sent: id=%s type=%s recipient=%s",
        notification.id, notification.type, notification.recipient_id,
    )


def log_notification_email_failed(notification, error):
    logger.error(
        "Notification email failed: id=%s recipient=%s error=%s",
        notification.id, notification.recipient_id, str(error),
    )


def log_bulk_notifications_created(count, notification_type):
    logger.info("Bulk notifications created: count=%d type=%s", count, notification_type)
