import numpy as np
import json
from sentence_transformers import SentenceTransformer
from graphai.core.interfaces.config import config
from graphai.core.common.fingerprinting import md5_text
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
    assert isinstance(s, str)
    return md5_text(s) + '_' + model_type


class EmbeddingModels:
    def __init__(self):
        self.models = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
            self.models['Solon-embeddings-large-0.1'] = SentenceTransformer(
                MODEL_TYPES['Solon-embeddings-large-0.1'],
                device=self.device,
                cache_folder=self.cache_dir
            )

    def embed(self, text, model_type='all-MiniLM-L12-v2'):
        if text is None or len(text) == 0:
            return None
        load_heavies = model_type != 'all-MiniLM-L12-v2'
        self.load_models(load_heavies=load_heavies)
        if model_type not in self.models.keys():
            raise NotImplementedError("Selected model type not implemented")
        return self.models[model_type].encode(text)
