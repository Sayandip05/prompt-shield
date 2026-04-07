from rest_framework import permissions

from .models import WorkLog, WeeklyReport, DeliveryProof


class IsWorkLogFreelancer(permissions.BasePermission):
    """
    Permission that allows only the freelancer who created the log.
    """
    
    def has_object_permission(self, request, view, obj):
        return obj.freelancer == request.user


class IsContractParticipant(permissions.BasePermission):
    """
    Permission that allows either freelancer or client of the contract.
    """
    
    def has_object_permission(self, request, view, obj):
        user = request.user
        if isinstance(obj, WorkLog):
            return (
                obj.freelancer == user or
                obj.contract.bid.project.client == user
            )
        elif isinstance(obj, WeeklyReport):
            return (
                obj.contract.bid.freelancer == user or
                obj.contract.bid.project.client == user
            )
        elif isinstance(obj, DeliveryProof):
            return (
                obj.contract.bid.freelancer == user or
                obj.contract.bid.project.client == user
            )
        return False


class IsContractFreelancer(permissions.BasePermission):
    """
    Permission that allows only the contract's freelancer.
    """
    
    def has_object_permission(self, request, view, obj):
        if isinstance(obj, WorkLog):
            return obj.freelancer == request.user
        return obj.contract.bid.freelancer == request.user


class IsContractClient(permissions.BasePermission):
    """
    Permission that allows only the contract's client.
    """
    
    def has_object_permission(self, request, view, obj):
        if isinstance(obj, WorkLog):
            return obj.contract.bid.project.client == request.user
        return obj.contract.bid.project.client == request.user
