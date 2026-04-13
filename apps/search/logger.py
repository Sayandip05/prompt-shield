"""
Structured logging for the Search app.
"""
import logging

logger = logging.getLogger("apps.search")


def log_search_query(query, search_type, results_count):
    logger.info(
        "Search query: q='%s' type=%s results=%d",
        query, search_type, results_count,
    )


def log_index_updated(document_type, document_id):
    logger.info("ES index updated: type=%s id=%s", document_type, document_id)


def log_index_deleted(document_type, document_id):
    logger.info("ES index deleted: type=%s id=%s", document_type, document_id)


def log_reindex_started(index_name):
    logger.info("ES reindex started: index=%s", index_name)


def log_reindex_completed(index_name, doc_count):
    logger.info("ES reindex completed: index=%s docs=%d", index_name, doc_count)


def log_reindex_failed(index_name, error):
    logger.error("ES reindex failed: index=%s error=%s", index_name, str(error))


def log_search_error(query, error):
    logger.error("Search failed: q='%s' error=%s", query, str(error))
