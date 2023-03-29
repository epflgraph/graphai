from celery import shared_task


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology.tree', ignore_result=False)
def get_ontology_tree_task(self, ontology):
    return {'child_to_parent': ontology.categories_categories.to_dict(orient='records')}


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology.parent', ignore_result=False)
def get_category_parent_task(self, ontology, child_id):
    if child_id not in ontology.category_parents.index:
        return None
    return {'child_to_parent':
                [{
                    'ParentCategoryID': ontology.category_parents[child_id],
                    'ChildCategoryID': child_id
                }]
            }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology.children', ignore_result=False)
def get_category_children_task(self, ontology, parent_id):
    if parent_id not in ontology.category_ids:
        return None
    cat_to_cat = ontology.categories_categories
    return {'child_to_parent': cat_to_cat.loc[cat_to_cat['ParentCategoryID']==parent_id].to_dict(orient='records')}
