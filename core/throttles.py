from rest_framework.throttling import UserRateThrottle


class TieredRateThrottle(UserRateThrottle):
    scope = "tiered"
