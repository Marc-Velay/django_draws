from django.apps import AppConfig

from .classifier import load_model


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    def ready(self):
        load_model()