from celery import shared_task
from graphai.api.common.ontology import ontology
from .common import compile_task_results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology.tree', ignore_result=False, ontology_obj=ontology)
def get_ontology_tree_task(self):
    print(self.ontology_obj.category_ids)
    return {'child_to_parent': self.ontology_obj.categories_categories.to_dict(orient='records')}


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology.parent', ignore_result=False, ontology_obj=ontology)
def get_category_parent_task(self, child_id):
    if child_id not in self.ontology_obj.category_parents.index:
        results = None
    else:
        results = [{
                    'ParentCategoryID': self.ontology_obj.category_parents[child_id],
                    'ChildCategoryID': child_id
                }]
    return {'child_to_parent': results}



@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology.children', ignore_result=False, ontology_obj=ontology)
def get_category_children_task(self, parent_id):
    if parent_id not in self.ontology_obj.category_ids:
        results = None
    else:
        cat_to_cat = self.ontology_obj.categories_categories
        results = cat_to_cat.loc[cat_to_cat['ParentCategoryID']==parent_id].to_dict(orient='records')
    return {'child_to_parent': results}


def get_ontology_tree_master():
    task = (get_ontology_tree_task.s()).apply_async(priority=6)
    task_id = task.id
    results = task.get(timeout=10)
    return {'id': task_id, 'results': results}


def get_category_parent_master(child_id):
    task = (get_category_parent_task.s(child_id)).apply_async(priority=6)
    task_id = task.id
    results = task.get(timeout=10)
    return {'id': task_id, 'results': results}


def get_category_children_master(child_id):
    task = (get_category_children_task.s(child_id)).apply_async(priority=6)
    task_id = task.id
    results = task.get(timeout=10)
    return {'id': task_id, 'results': results}