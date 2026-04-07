from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import WorkLogViewSet, WeeklyReportViewSet, DeliveryProofViewSet


router = DefaultRouter()
router.register(r'logs', WorkLogViewSet, basename='worklog')
router.register(r'reports', WeeklyReportViewSet, basename='weeklyreport')

urlpatterns = [
    path('', include(router.urls)),
    path('proofs/<int:pk>/', DeliveryProofViewSet.as_view({
        'get': 'retrieve',
        'post': 'generate',
    }), name='deliveryproof'),
]
