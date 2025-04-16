from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NerModelConfiguration, TransformersNlpEngine
from presidio_anonymizer import AnonymizerEngine
from graphai.core.common.config import config
import torch
from multiprocessing import Lock
import time


# Transformer model config
model_config = [
    {
        "lang_code": "en",
        "model_name": {
            "spacy": "en_core_web_sm",  # for tokenization, lemmatization
            "transformers": "Davlan/distilbert-base-multilingual-cased-ner-hrl"  # for NER
        }
    },
    {
        "lang_code": "fr",
        "model_name": {
            "spacy": "fr_core_news_sm",  # for tokenization, lemmatization
            "transformers": "Davlan/distilbert-base-multilingual-cased-ner-hrl"  # for NER
        }
    }
]
# No need for a mapping since the default works well
mapping = None
labels_to_ignore = ["O"]


class AnonymizerModels:
    def __init__(self):
        self.models = None
        self.load_lock = Lock()
        self.last_model_use = time.time()
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

    def load_models(self):
        with self.load_lock:
            if self.models is None:
                self.models = dict()
                print('Loading analyzer and anonymizer')
                ner_model_configuration = NerModelConfiguration(
                    model_to_presidio_entity_mapping=mapping,
                    alignment_mode="expand",  # "strict", "contract", "expand"
                    aggregation_strategy="max",  # "simple", "first", "average", "max"
                    labels_to_ignore=labels_to_ignore)

                transformers_nlp_engine = TransformersNlpEngine(
                    models=model_config,
                    ner_model_configuration=ner_model_configuration)

                # Transformer-based analyzer
                analyzer = AnalyzerEngine(
                    nlp_engine=transformers_nlp_engine,
                    supported_languages=["en", "fr"]
                )
                self.models['analyzer'] = analyzer
                self.models['anonymizer'] = AnonymizerEngine()

    def anonymize(self, text, lang):
        self.load_models()
        if lang not in ['en', 'fr']:
            raise NotImplementedError("Only English and French are implemented at the moment.")
        analyzer_results = self.models['analyzer'].analyze(text=text, language=lang)
        anonymized = self.models['anonymizer'].anonymize(text, analyzer_results=analyzer_results)
        return anonymized.text


def anonymize_text(anonymizer_model, text, lang):
    try:
        result = anonymizer_model.anonymize(text, lang)
    except NotImplementedError:
        return {
            "result": "Language not supported",
            "successful": False
        }
    return {
        "result": result,
        "successful": True
    }
