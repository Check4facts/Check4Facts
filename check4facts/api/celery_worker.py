from check4facts.api import app, celery_init_app

celery_app = celery_init_app(app)
