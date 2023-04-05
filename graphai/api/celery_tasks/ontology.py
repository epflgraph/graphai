from celery import shared_task
from graphai.api.common.ontology import ontology


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology.tree', ignore_result=False, ontology_obj=ontology)
def get_ontology_tree_task(self):
    return {'child_to_parent': self.ontology_obj.get_predefined_tree()}


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology.parent', ignore_result=False, ontology_obj=ontology)
def get_category_parent_task(self, child_id):
    return {'child_to_parent': self.ontology_obj.get_category_parent(child_id)}



@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology.children', ignore_result=False, ontology_obj=ontology)
def get_category_children_task(self, parent_id):
    return {'child_to_parent': self.ontology_obj.get_category_children(parent_id)}


def get_ontology_tree_master():
    task = (get_ontology_tree_task.s()).apply_async(priority=6)
    task_id = task.id
    try:
        results = task.get(timeout=10)
    except TimeoutError as e:
        print(e)
        results = None
    return {'id': task_id, 'results': results}


def get_category_parent_master(child_id):
    task = (get_category_parent_task.s(child_id)).apply_async(priority=6)
    task_id = task.id
    try:
        results = task.get(timeout=10)
    except TimeoutError as e:
        print(e)
        results = None
    return {'id': task_id, 'results': results}


def get_category_children_master(child_id):
    task = (get_category_children_task.s(child_id)).apply_async(priority=6)
    task_id = task.id
    try:
        results = task.get(timeout=10)
    except TimeoutError as e:
        print(e)
        results = None
    return {'id': task_id, 'results': results}