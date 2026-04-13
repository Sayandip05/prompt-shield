"""
Search app admin configuration.

The Search app uses Elasticsearch documents (ProjectDocument, FreelancerDocument),
not Django ORM models, so there are no models to register in admin.

ES index management is done via management commands:
    python manage.py search_index --rebuild
    python manage.py search_index --populate
"""
