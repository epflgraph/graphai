from celery import shared_task
from graphai.api.common.ontology import ontology


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.tree', ignore_result=False, ontology_obj=ontology)
def get_ontology_tree_task(self):
    return {'child_to_parent': self.ontology_obj.get_predefined_tree()}


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.parent', ignore_result=False, ontology_obj=ontology)
def get_category_parent_task(self, child_id):
    return {'child_to_parent': self.ontology_obj.get_category_parent(child_id)}


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.children', ignore_result=False, ontology_obj=ontology)
def get_category_children_task(self, parent_id):
    return {'child_to_parent': self.ontology_obj.get_category_children(parent_id)}
