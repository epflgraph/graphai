import time
import gc
import copy
from itertools import chain

import numpy as np
import json
from sentence_transformers import SentenceTransformer

from graphai.core.common.caching import (
    EmbeddingDBCachingManager,
    token_based_text_lookup,
    fingerprint_based_text_lookup,
    database_callback_generic
)
from graphai.core.common.common_utils import get_current_datetime
from graphai.core.common.config import config
from graphai.core.common.fingerprinting import (
    md5_text,
    perceptual_hash_text
)
import torch
from multiprocessing import Lock


MODEL_TYPES = {
    'all-MiniLM-L12-v2': 'sentence-transformers/all-MiniLM-L12-v2',
    'Solon-embeddings-large-0.1': 'OrdalieTech/Solon-embeddings-large-0.1'
}
# 3 hours
EMBEDDING_UNLOAD_WAITING_PERIOD = 3 * 3600.0
LONG_TEXT_ERROR = "Text over token limit for selected model (%d)."


def embedding_to_json(v):
    v_list = v.tolist()
    v_list = [round(x, 9) for x in v_list]
    return json.dumps(v_list)


def embedding_from_json(s):
    return np.array(json.loads(s))


def generate_embedding_text_token(s, model_type):
    """
    Generates an md5-based token for a string
    Args:
        s: The string
        model_type: Type of embedding model

    Returns:
        Token
    """
    assert isinstance(s, str)
    return md5_text(s) + '_' + model_type


def get_text_token_count_using_model(model_tokenizer, text):
    input_ids = model_tokenizer.tokenizer(
        text, return_attention_mask=False, return_token_type_ids=False
    )['input_ids']
    if isinstance(text, str):
        return len(input_ids)
    else:
        return sum([len(x) for x in input_ids])


class EmbeddingModels:
    def __init__(self):
        self.models = None
        self.load_lock = Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.last_heavy_model_use = time.time()
        try:
            print("Reading HuggingFace model path from config")
            self.cache_dir = config['huggingface']['model_path']
            if self.cache_dir == '':
                self.cache_dir = None
        except Exception:
            print(
                "The HuggingFace dl path could not be found in the config file, using default (~/.cache/huggingface). "
                "To use a different one, make sure to add a [huggingface] section with the model_path parameter."
            )
            self.cache_dir = None

    def get_device(self):
        return self.device

    def load_models(self, load_heavies=True):
        """
        Loads sentence transformers model
        Returns:
            None
        """
        with self.load_lock:
            if self.models is None:
                self.models = dict()
                self.models['all-MiniLM-L12-v2'] = SentenceTransformer(
                    MODEL_TYPES['all-MiniLM-L12-v2'],
                    device=self.device,
                    cache_folder=self.cache_dir
                )
            if load_heavies:
                if 'Solon-embeddings-large-0.1' not in self.models:
                    self.models['Solon-embeddings-large-0.1'] = SentenceTransformer(
                        MODEL_TYPES['Solon-embeddings-large-0.1'],
                        device=self.device,
                        cache_folder=self.cache_dir
                    )
                self.last_heavy_model_use = time.time()

    def load_model(self, model_type):
        if model_type not in MODEL_TYPES.keys():
            return False
        with self.load_lock:
            if self.models is None:
                self.models = dict()
            if model_type not in self.models.keys():
                self.models[model_type] = SentenceTransformer(
                    MODEL_TYPES[model_type],
                    device=self.device,
                    cache_folder=self.cache_dir
                )
            if model_type != 'all-MiniLM-L12-v2':
                self.last_heavy_model_use = time.time()
        return True

    def model_loaded(self, model_type):
        return self.models is not None and model_type in self.models.keys()

    def _get_tokenizer(self, model_type):
        return self.models[model_type][0]

    def get_tokenizer(self, model_type):
        loaded = self.load_model(model_type)
        if not loaded:
            return None
        return self._get_tokenizer(model_type)

    def set_tokenizer(self, model_type, tokenizer):
        loaded = self.load_model(model_type)
        if not loaded:
            return False
        new_models = {
            m: self.models[m] for m in self.models if m != model_type
        }
        new_models[model_type] = SentenceTransformer(
            modules=[tokenizer, self.models[model_type][1], self.models[model_type][2]]
        )
        self.models = new_models
        return True

    def get_last_usage(self):
        return self.last_heavy_model_use

    def unload_model(self, unload_period=EMBEDDING_UNLOAD_WAITING_PERIOD):
        """
        Unloads all models except the light, default model.
        Args:
            unload_period: Minimum time that needs to have passed since last use of heavy model to qualify it for
            unloading. If set to 0, forces an unloading.

        Returns:
            None if not enough time has passed since last use, a list of unloaded models otherwise.
        """
        deleted_models = None
        with self.load_lock:
            if time.time() - self.get_last_usage() > unload_period:
                deleted_models = list()
                if self.models is None:
                    return deleted_models
                heavy_model_keys = set(MODEL_TYPES.keys()).difference({'all-MiniLM-L12-v2'})
                for key in heavy_model_keys:
                    if key in self.models:
                        deleted_models.append(key)
                        del self.models[key]
                if len(deleted_models) > 0:
                    gc.collect()
        return deleted_models

    @staticmethod
    def _get_token_count(model, text):
        model_tokenizer = model[0]
        return get_text_token_count_using_model(model_tokenizer, text)

    @staticmethod
    def _get_model_max_tokens(model):
        return model.max_seq_length

    def get_token_count(self, text, model_type):
        if text is None or len(text) == 0:
            return 0
        self.load_model(model_type)
        if model_type not in self.models.keys():
            raise NotImplementedError(f"Selected model type not implemented: {model_type}")
        model_to_use = self.models[model_type]
        return self._get_token_count(model_to_use, text)

    def get_max_tokens(self, model_type):
        self.load_model(model_type)
        if model_type not in self.models.keys():
            raise NotImplementedError(f"Selected model type not implemented: {model_type}")
        model_to_use = self.models[model_type]
        return self._get_model_max_tokens(model_to_use)

    def _get_model_output(self, model, text):
        try:
            print(self.device)
            model_max_tokens = self._get_model_max_tokens(model)
            n_tokens = self._get_token_count(model, text)
            if n_tokens > model_max_tokens:
                return None
            return model.encode(text)
        except IndexError as e:
            print(e)
            return None

    def _embed(self, model, text):
        text_too_large = False
        result = self._get_model_output(model, text)
        if result is None:
            text_too_large = True
        return result, text_too_large

    def embed(self, text, model_type='all-MiniLM-L12-v2'):
        if text is None or len(text) == 0:
            return None
        self.load_model(model_type)
        if model_type not in self.models.keys():
            raise NotImplementedError(f"Selected model type not implemented: {model_type}")
        model = self.models[model_type]
        max_tokens = self._get_model_max_tokens(model)
        results, text_too_large = self._embed(model, text)
        return results, text_too_large, max_tokens


def copy_embedding_object(embedding_obj, model_type):
    tokenizer = embedding_obj.get_tokenizer(model_type)
    if tokenizer is None:
        raise NotImplementedError(f"Model type {model_type} cannot be found!")
    current_embedding_obj = copy.copy(embedding_obj)
    current_tokenizer = copy.deepcopy(tokenizer)
    current_embedding_obj.set_tokenizer(model_type, current_tokenizer)
    return current_embedding_obj, current_tokenizer


def compute_embedding_text_fingerprint_callback(results, text, model_type):
    # This task does not have the condition of the 'fresh' flag being True because text fingerprinting never fails
    token = results['token']
    values_dict = {
        'fingerprint': results['result'],
        'source': text,
        'model_type': model_type,
        'date_added': get_current_datetime()
    }
    database_callback_generic(token, EmbeddingDBCachingManager(), values_dict, False, False)
    return results


def token_based_embedding_lookup(token, model_type):
    return token_based_text_lookup(token, EmbeddingDBCachingManager(), 'embedding', model_type=model_type)


def fingerprint_based_embedding_lookup(token, fp, model_type):
    return fingerprint_based_text_lookup(token, fp, EmbeddingDBCachingManager(),
                                         main_col='embedding', extra_cols=['model_type'],
                                         equality_conditions={'model_type': model_type},
                                         model_type=model_type)


def embed_text(models, text, model_type):
    try:
        embedding, text_too_large, model_max_tokens = models.embed(text, model_type)
        success = embedding is not None
        if text_too_large:
            embedding = LONG_TEXT_ERROR % model_max_tokens
    except NotImplementedError as e:
        print(e)
        embedding = str(e)
        success = False
        text_too_large = False

    return {
        'result': embedding,
        'successful': success,
        'text_too_large': text_too_large,
        'fresh': success,
        'model_type': model_type,
        'device': models.get_device()
    }


def insert_embedding_into_db(results, token, text, model_type, force=False):
    db_manager = EmbeddingDBCachingManager()
    if results['fresh']:
        values_dict = {
            'source': text,
            'embedding': embedding_to_json(results['result']),
            'model_type': model_type
        }
        existing = db_manager.get_details(token, ['date_added'], using_most_similar=False)[0]
        if existing is None or existing['date_added'] is None:
            current_datetime = get_current_datetime()
            values_dict['date_added'] = current_datetime
        # Inserting values for original token
        db_manager.insert_or_update_details(
            token, values_dict
        )
        if not force:
            # Inserting the same values for closest token if different than original token
            # Only happens if the other token has been fingerprinted first without having its embedding calculated.
            closest = db_manager.get_closest_match(token)
            if closest is not None and closest != token:
                db_manager.insert_or_update_details(
                    closest, values_dict
                )
        # If the computation is fresh, we need to JSONify the resulting numpy array.
        # Non-fresh successful computation results come from cache hits, and those are already in JSON.
        # Non-successful computation results are normal strings.
        results['result'] = embedding_to_json(results['result'])
    elif not results['successful']:
        # in case we fingerprinted something and then failed to embed it, we delete its cache row
        db_manager.delete_cache_rows([token])
    return results


def embedding_text_list_fingerprint_parallel(tokens, text_list, i, n):
    start_index = int(i * len(tokens) / n)
    end_index = int((i + 1) * len(tokens) / n)
    if start_index == end_index:
        return []
    tokens = tokens[start_index:end_index]
    text_list = text_list[start_index:end_index]
    results = list()
    db_manager = EmbeddingDBCachingManager()
    for j in range(len(tokens)):
        existing = db_manager.get_details(tokens[j], cols=['fingerprint'])[0]
        if existing is not None and existing['fingerprint'] is not None:
            results.append(
                {
                    'token': tokens[j],
                    'fp': existing['fingerprint'],
                    'text': text_list[j],
                    'fresh': False
                }
            )
        else:
            results.append(
                {
                    'token': tokens[j],
                    'fp': perceptual_hash_text(text_list[j]),
                    'text': text_list[j],
                    'fresh': True
                }
            )
    return results


def embedding_text_list_fingerprint_callback(results, model_type):
    db_manager = EmbeddingDBCachingManager()
    all_results = list(chain.from_iterable(results))
    for result in all_results:
        if result['fresh']:
            values_dict = {
                'fingerprint': result['fp'],
                'source': result['text'],
                'model_type': model_type,
                'date_added': get_current_datetime()
            }
            db_manager.insert_or_update_details(result['token'], values_dict)
    return all_results


def embedding_text_list_embed_parallel(input_list, embedding_obj, model_type, i, n, force=False):
    start_index = int(i * len(input_list) / n)
    end_index = int((i + 1) * len(input_list) / n)
    if start_index == end_index:
        return []
    input_list = input_list[start_index:end_index]
    input_list = dict(enumerate(input_list))
    results_dict = dict()
    if not force:
        for ind in input_list:
            current_dict = input_list[ind]
            current_results = token_based_embedding_lookup(current_dict['token'], model_type)
            if current_results is None:
                current_results = fingerprint_based_embedding_lookup(
                    current_dict['token'], current_dict['fp'], model_type
                )
            if current_results is not None:
                current_results['id_token'] = current_dict['token']
                current_results['source'] = current_dict['text']
                results_dict[ind] = current_results
    remaining_indices = list(set(input_list.keys()).difference(set(results_dict.keys())))
    if len(remaining_indices) > 0:
        current_embedding_obj, tokenizer = copy_embedding_object(embedding_obj, model_type)
        token_counts = [get_text_token_count_using_model(tokenizer, input_list[ind]['text'])
                        for ind in remaining_indices]
        model_max_tokens = current_embedding_obj.get_max_tokens(model_type)
        i = 0
        while i < len(remaining_indices):
            current_token_count_sum = token_counts[i]
            j = i + 1
            while current_token_count_sum < model_max_tokens and j < len(remaining_indices):
                current_token_count_sum += token_counts[j]
                j += 1
            if current_token_count_sum >= model_max_tokens:
                j -= 1
            if j == i:
                j = i + 1
            current_results_list = embed_text(current_embedding_obj,
                                              [input_list[remaining_indices[k]]['text'] for k in range(i, j)],
                                              model_type)
            for k in range(i, j):
                current_results = {
                    'result': current_results_list['result'][k - i]
                    if not isinstance(current_results_list['result'], str)
                    else current_results_list['result'],
                    'successful': current_results_list['successful'],
                    'text_too_large': current_results_list['text_too_large'],
                    'fresh': current_results_list['fresh'],
                    'model_type': current_results_list['model_type'],
                    'device': current_results_list['device'],
                    'id_token': input_list[remaining_indices[k]]['token'],
                    'source': input_list[remaining_indices[k]]['text']
                }
                results_dict[remaining_indices[k]] = current_results
            i = j
    sorted_indices = sorted(results_dict.keys())
    current_embedding_obj = None
    tokenizer = None
    return [results_dict[i] for i in sorted_indices]


def embedding_text_list_embed_callback(results, model_type, force):
    all_results = list(chain.from_iterable(results))
    new_results = list()
    for result in all_results:
        new_result = insert_embedding_into_db(result, result['id_token'], result['source'], model_type, force)
        del new_result['id_token']
        del new_result['source']
        new_results.append(new_result)
    gc.collect()
    return new_results
