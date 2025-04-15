from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NerModelConfiguration, TransformersNlpEngine
from presidio_anonymizer import AnonymizerEngine
from transformers import AutoTokenizer, AutoModelForTokenClassification
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
            "transformers": "StanfordAIMI/stanford-deidentifier-base"  # for NER
        }
    }
]

# Entity mappings between the model's and Presidio's
mapping = dict(
    PER="PERSON",
    LOC="LOCATION",
    ORG="ORGANIZATION",
    AGE="AGE",
    ID="ID",
    EMAIL="EMAIL",
    DATE="DATE_TIME",
    PHONE="PHONE_NUMBER",
    PERSON="PERSON",
    LOCATION="LOCATION",
    GPE="LOCATION",
    ORGANIZATION="ORGANIZATION",
    NORP="NRP",
    PATIENT="PERSON",
    STAFF="PERSON",
    HOSP="LOCATION",
    PATORG="ORGANIZATION",
    TIME="DATE_TIME",
    HCW="PERSON",
    HOSPITAL="LOCATION",
    FACILITY="LOCATION",
    VENDOR="ORGANIZATION",
)

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
                self.models['analyzers'] = dict()
                print('Loading EN analyzer')
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
                    supported_languages=["en"]
                )
                self.models['analyzers']['en'] = analyzer
                self.models['anonymizer'] = AnonymizerEngine()

    def anonymize(self, text, lang):
        self.load_models()
        if lang not in self.models['analyzers'].keys():
            print(f"Language {lang} not implemented, falling back to en...")
            lang = 'en'
        analyzer_results = self.models['analyzers'][lang].analyze(text=text, language=lang)
        anonymized = self.models['anonymizer'].anonymize(text, analyzer_results=analyzer_results)
        return anonymized.text
