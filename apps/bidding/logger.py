"""
Structured logging for the Bidding app.
"""
import logging

logger = logging.getLogger("apps.bidding")


def log_bid_submitted(bid):
    logger.info(
        "Bid submitted: id=%s project=%s freelancer=%s amount=%s",
        bid.id, bid.project_id, bid.freelancer_id, bid.amount,
    )


def log_bid_accepted(bid):
    logger.info(
        "Bid accepted: id=%s project=%s freelancer=%s",
        bid.id, bid.project_id, bid.freelancer_id,
    )


def log_bid_rejected(bid):
    logger.info(
        "Bid rejected: id=%s project=%s freelancer=%s",
        bid.id, bid.project_id, bid.freelancer_id,
    )


def log_contract_created(contract):
    logger.info(
        "Contract created: id=%s bid=%s status=%s",
        contract.id, contract.bid_id, contract.status,
    )


def log_contract_status_changed(contract, old_status, new_status):
    logger.info(
        "Contract status changed: id=%s '%s' -> '%s'",
        contract.id, old_status, new_status,
    )
