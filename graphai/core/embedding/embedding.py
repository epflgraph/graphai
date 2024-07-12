import time

import numpy as np
import json
from sentence_transformers import SentenceTransformer
from graphai.core.interfaces.config import config
from graphai.core.common.fingerprinting import md5_text
from graphai.core.common.common_utils import convert_list_to_text, convert_text_back_to_list
import torch


MODEL_TYPES = {
    'all-MiniLM-L12-v2': 'sentence-transformers/all-MiniLM-L12-v2',
    'Solon-embeddings-large-0.1': 'OrdalieTech/Solon-embeddings-large-0.1'
}


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
    assert isinstance(s, str) or isinstance(s, list)
    if isinstance(s, str):
        return md5_text(s) + '_' + model_type
    else:
        return md5_text(convert_list_to_text(s)) + '_' + model_type


def get_model_max_tokens(model):
    return model.max_seq_length


class EmbeddingModels:
    def __init__(self):
        self.models = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.last_heavy_model_use = 0
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

    def get_last_usage(self):
        return self.last_heavy_model_use

    def unload_heavy_models(self):
        deleted_models = list()
        if self.models is None:
            return deleted_models
        heavy_model_keys = set(MODEL_TYPES.keys()).difference({'all-MiniLM-L12-v2'})
        for key in heavy_model_keys:
            if key in self.models:
                deleted_models.append(key)
                del self.models[key]
        return deleted_models

    def _get_model_output(self, model, text):
        try:
            print(self.device)
            model_tokenizer = model[0]
            model_max_tokens = get_model_max_tokens(model)
            input_ids = model_tokenizer.tokenizer(
                text, return_attention_mask=False, return_token_type_ids=False
            )['input_ids']
            if len(input_ids) > model_max_tokens:
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
        load_heavies = model_type != 'all-MiniLM-L12-v2'
        self.load_models(load_heavies=load_heavies)
        if model_type not in self.models.keys():
            raise NotImplementedError("Selected model type not implemented")
        results, text_too_large = self._embed(self.models[model_type], text)
        return results, text_too_large, get_model_max_tokens(self.models[model_type])
