from celery import shared_task


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology.tree', ignore_result=False)
def get_ontology_tree_task(self, ontology):
    return {'child_to_parent': ontology.categories_categories.to_dict(orient='records')}


@shared_task(bind=True,autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology.whatever', ignore_result=False)
def get_whatever(self):
    return 1