import time
import gc

import numpy as np
import json
from sentence_transformers import SentenceTransformer
from graphai.core.common.config import config
from graphai.core.common.fingerprinting import md5_text
import torch
from multiprocessing import Lock


MODEL_TYPES = {
    'all-MiniLM-L12-v2': 'sentence-transformers/all-MiniLM-L12-v2',
    'Solon-embeddings-large-0.1': 'OrdalieTech/Solon-embeddings-large-0.1'
}
# 3 hours
EMBEDDING_UNLOAD_WAITING_PERIOD = 3 * 3600.0


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
