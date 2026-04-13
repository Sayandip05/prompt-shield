from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from apps.projects.models import Project, ProjectSkill
from apps.users.models import FreelancerProfile
from apps.search.documents import ProjectDocument, FreelancerDocument


@receiver(post_save, sender=Project)
def update_project_document(sender, instance, **kwargs):
    """Update project document in Elasticsearch when project is saved."""
    ProjectDocument().update(instance)


@receiver(post_delete, sender=Project)
def delete_project_document(sender, instance, **kwargs):
    """Delete project document from Elasticsearch when project is deleted."""
    ProjectDocument().delete(instance, ignore=404)


@receiver(post_save, sender=ProjectSkill)
def update_project_document_on_skill_change(sender, instance, **kwargs):
    """Update project document when skills are modified."""
    ProjectDocument().update(instance.project)


@receiver(post_delete, sender=ProjectSkill)
def delete_project_document_on_skill_delete(sender, instance, **kwargs):
    """Update project document when skills are deleted."""
    ProjectDocument().update(instance.project)


@receiver(post_save, sender=FreelancerProfile)
def update_freelancer_document(sender, instance, **kwargs):
    """Update freelancer document in Elasticsearch when profile is saved."""
    FreelancerDocument().update(instance)


@receiver(post_delete, sender=FreelancerProfile)
def delete_freelancer_document(sender, instance, **kwargs):
    """Delete freelancer document from Elasticsearch when profile is deleted."""
    FreelancerDocument().delete(instance, ignore=404)
