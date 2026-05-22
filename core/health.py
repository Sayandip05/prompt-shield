"""
Health check endpoint for infrastructure monitoring.

Usage:
    Add to config/urls.py:
        path("health/", include("core.health")),
"""
import logging
from datetime import datetime

from django.db import connection
from django.http import JsonResponse
from django.urls import path
from django.conf import settings

logger = logging.getLogger(__name__)


def health_check(request):
    """
    Basic health check returning service status.
    Checks database connectivity, Redis, Celery, and Elasticsearch availability.
    """
    health = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": getattr(settings, "APP_VERSION", "1.0.0"),
        "checks": {},
    }

    # Database check
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        health["checks"]["database"] = {"status": "up"}
    except Exception as e:
        logger.error("Health check - database failed: %s", str(e))
        health["checks"]["database"] = {"status": "down", "error": str(e)}
        health["status"] = "unhealthy"

    # Redis check
    try:
        from django_redis import get_redis_connection
        redis_conn = get_redis_connection("default")
        redis_conn.ping()
        health["checks"]["redis"] = {"status": "up"}
    except Exception as e:
        logger.warning("Health check - redis failed: %s", str(e))
        health["checks"]["redis"] = {"status": "down", "error": str(e)}
        health["status"] = "degraded"

    # Celery check (lightweight — just check broker reachable)
    try:
        from config.celery import app as celery_app
        inspect = celery_app.control.inspect(timeout=2.0)
        ping = inspect.ping()
        if ping:
            health["checks"]["celery"] = {"status": "up", "workers": len(ping)}
        else:
            health["checks"]["celery"] = {"status": "no_workers"}
            health["status"] = "degraded"
    except Exception as e:
        logger.warning("Health check - celery failed: %s", str(e))
        health["checks"]["celery"] = {"status": "down", "error": str(e)}

    # Elasticsearch check — only degrades, never marks unhealthy.
    # The app continues to work without ES (signals are resilient).
    try:
        import urllib.request
        es_url = getattr(settings, "ELASTICSEARCH_URL", "http://localhost:9200")
        url = f"{es_url.rstrip('/')}/_cluster/health"
        with urllib.request.urlopen(url, timeout=3) as resp:  # noqa: S310
            if resp.status == 200:
                health["checks"]["elasticsearch"] = {"status": "up"}
            else:
                health["checks"]["elasticsearch"] = {"status": "starting"}
                if health["status"] == "healthy":
                    health["status"] = "degraded"
    except Exception as e:
        logger.warning("Health check - elasticsearch unavailable: %s", str(e))
        health["checks"]["elasticsearch"] = {"status": "down", "error": str(e)}
        if health["status"] == "healthy":
            health["status"] = "degraded"

    # 200 for healthy/degraded; 503 only when database (critical) is down
    status_code = 200 if health["status"] in ("healthy", "degraded") else 503
    return JsonResponse(health, status=status_code)


def readiness_check(request):
    """
    Readiness probe — checks if the application is ready to serve traffic.
    More strict than health check.
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        return JsonResponse({"ready": True}, status=200)
    except Exception:
        return JsonResponse({"ready": False}, status=503)


def liveness_check(request):
    """
    Liveness probe — just confirms the process is running.
    """
    return JsonResponse({"alive": True}, status=200)


urlpatterns = [
    path("", health_check, name="health-check"),
    path("ready/", readiness_check, name="readiness-check"),
    path("live/", liveness_check, name="liveness-check"),
]
