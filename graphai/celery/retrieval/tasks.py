from celery import shared_task
from graphai.core.retrieval.retrieval_utils import (
    retrieve_from_es,
    chunk_text
)
from graphai.core.retrieval.anonymization import (
    AnonymizerModels,
    ANONYMIZER_UNLOAD_WAITING_PERIOD,
    anonymize_text
)


anonymizer_model = AnonymizerModels()


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='rag.retrieve', ignore_result=False)
def retrieve_from_es_task(self, embedding_results, text, index_to_search_in,
                          filters=None, limit=10, return_scores=False, filter_by_date=False):
    return retrieve_from_es(embedding_results, text, index_to_search_in,
                            filters, limit, return_scores, filter_by_date)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='rag.chunk', ignore_result=False)
def chunk_text_task(self, text, chunk_size=400, chunk_overlap=100,
                    one_chunk_per_page=False, one_chunk_per_doc=False):
    return chunk_text(text, chunk_size, chunk_overlap, one_chunk_per_page, one_chunk_per_doc)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='rag.anonymize', anonymization_obj=anonymizer_model, ignore_result=False)
def anonymize_text_task(self, text, lang):
    return anonymize_text(self.anonymization_obj, text, lang)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='rag.clean_up_anonymizer_object', anonymization_obj=anonymizer_model, ignore_result=False)
def cleanup_anonymizer_object_task(self):
    """Periodic task that releases the presidio/GLiNER anonymizer stack."""
    return self.anonymization_obj.unload_model(ANONYMIZER_UNLOAD_WAITING_PERIOD)
