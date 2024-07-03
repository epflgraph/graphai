from sentence_transformers import SentenceTransformer
from graphai.core.interfaces.config import config
from graphai.core.common.fingerprinting import md5_text
import torch


MODEL_TYPES = {
    'light': 'sentence-transformers/all-MiniLM-L12-v2',
}


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
            print("Reading embedding model path from config")
            self.cache_dir = config['embedding']['model_path']
            if self.cache_dir == '':
                self.cache_dir = None
        except Exception:
            print(
                "The embedding dl path could not be found in the config file, using default. "
                "To use a different one, make sure to add a [embedding] section with the model_path parameter."
            )
            self.cache_dir = None

    def get_device(self):
        return self.device

    def load_models(self):
        """
        Loads sentence transformers model
        Returns:
            None
        """
        if self.models is None:
            self.models = dict()
            self.models['light'] = SentenceTransformer(MODEL_TYPES['light'],
                                                       device=self.device,
                                                       cache_folder=self.cache_dir)

    def embed(self, text, model_type='light'):
        self.load_models()
        if model_type not in self.models.keys():
            raise NotImplementedError("Selected model type not implemented")
        if text is None or len(text) == 0:
            return None
        return self.models[model_type].encode(text)
