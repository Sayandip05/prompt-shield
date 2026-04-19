"""
Structured logging for the Worklogs app.
"""
import logging

logger = logging.getLogger("apps.worklogs")


def log_worklog_created(worklog):
    logger.info(
        "Worklog created: id=%s contract=%s hours=%s date=%s",
        worklog.id, worklog.contract_id, worklog.hours, worklog.date,
    )


def log_worklog_submitted(worklog):
    logger.info("Worklog submitted: id=%s contract=%s", worklog.id, worklog.contract_id)


def log_worklog_approved(worklog):
    logger.info("Worklog approved: id=%s contract=%s", worklog.id, worklog.contract_id)


def log_worklog_rejected(worklog, reason=None):
    logger.warning(
        "Worklog rejected: id=%s contract=%s reason=%s",
        worklog.id, worklog.contract_id, reason,
    )


def log_report_generated(report):
    logger.info(
        "Weekly report generated: id=%s contract=%s week=%s",
        report.id, report.contract_id, report.week_start,
    )


def log_ai_report_started(contract_id, week_start):
    logger.info("AI report generation started: contract=%s week=%s", contract_id, week_start)


def log_ai_report_failed(contract_id, error):
    logger.error("AI report generation failed: contract=%s error=%s", contract_id, str(error))


def log_delivery_proof_uploaded(proof):
    logger.info("Delivery proof uploaded: id=%s worklog=%s", proof.id, proof.worklog_id)
