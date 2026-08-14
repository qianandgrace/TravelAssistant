python -m celery -A utils.tasks.celery_app worker --loglevel=info --pool=solo
python server.py
python webapp.py
