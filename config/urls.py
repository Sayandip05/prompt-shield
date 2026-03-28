from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/users/", include("apps.users.urls")),
    path("api/projects/", include("apps.projects.urls")),
    path("api/bidding/", include("apps.bidding.urls")),
    path("api/payments/", include("apps.payments.urls")),
    path("api/worklogs/", include("apps.worklogs.urls")),
    path("api/messaging/", include("apps.messaging.urls")),
    path("api/notifications/", include("apps.notifications.urls")),
    path("api/search/", include("apps.search.urls")),
]
