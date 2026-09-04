from importlib.util import find_spec
import gc
from multiprocessing import Lock
import time

import torch

from graphai.core.common.common_utils import strtobool
from graphai.core.common.config import config

# Presidio classes are imported lazily on first use so that workers which never
# anonymize do not pay the ~600 MiB import cost. These module-level placeholders
# are kept so that tests and callers can monkeypatch them before the first load.
AnalyzerEngine = None
AnonymizerEngine = None
GLiNERRecognizer = None
NerModelConfiguration = None
TransformersNlpEngine = None

# Unload the presidio/GLiNER anonymizer stack after this many seconds of inactivity.
ANONYMIZER_UNLOAD_WAITING_PERIOD = 3 * 3600.0


# Transformer model config
supported_languages = ["en", "fr"]
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
# Davlan/distilbert-base-multilingual-cased-ner-hrl emits BIO labels. Presidio
# requires an explicit model-to-Presidio mapping for NLP engine entities.
mapping = {
    "B-DATE": "DATE_TIME",
    "I-DATE": "DATE_TIME",
    "DATE": "DATE_TIME",
    "B-PER": "PERSON",
    "I-PER": "PERSON",
    "PER": "PERSON",
    "PERSON": "PERSON",
    "B-ORG": "ORGANIZATION",
    "I-ORG": "ORGANIZATION",
    "ORG": "ORGANIZATION",
    "ORGANIZATION": "ORGANIZATION",
    "B-LOC": "LOCATION",
    "I-LOC": "LOCATION",
    "LOC": "LOCATION",
    "LOCATION": "LOCATION",
    "GPE": "LOCATION",
}
labels_to_ignore = ["O"]

DEFAULT_GLINER_MODEL = "urchade/gliner_multi_pii-v1"

# GLiNER accepts natural-language labels, which lets us ask for broader PII
# classes than the Davlan BIO model can emit. Values are the replacement tags
# returned by Presidio's anonymizer.
gliner_entity_mapping = {
    "person": "PERSON",
    "name": "PERSON",
    "organization": "ORGANIZATION",
    "company": "ORGANIZATION",
    "location": "LOCATION",
    "address": "STREET_ADDRESS",
    "full address": "STREET_ADDRESS",
    "postal code": "POSTAL_CODE",
    "date": "DATE_TIME",
    "date of birth": "DATE_OF_BIRTH",
    "phone number": "PHONE_NUMBER",
    "mobile phone number": "PHONE_NUMBER",
    "landline phone number": "PHONE_NUMBER",
    "fax number": "PHONE_NUMBER",
    "email": "EMAIL_ADDRESS",
    "email address": "EMAIL_ADDRESS",
    "ip address": "IP_ADDRESS",
    "url": "URL",
    "credit card number": "CREDIT_CARD",
    "credit card expiration date": "CREDIT_CARD_EXPIRATION",
    "credit card brand": "CREDIT_CARD_BRAND",
    "cvv": "CVV",
    "cvc": "CVV",
    "bank account number": "BANK_ACCOUNT_NUMBER",
    "iban": "IBAN_CODE",
    "social security number": "US_SSN",
    "social_security_number": "US_SSN",
    "passport number": "PASSPORT_NUMBER",
    "passport_number": "PASSPORT_NUMBER",
    "passport expiration date": "PASSPORT_EXPIRATION",
    "driver's license number": "DRIVER_LICENSE_NUMBER",
    "driver licence": "DRIVER_LICENSE_NUMBER",
    "national id number": "NATIONAL_ID",
    "identity card number": "IDENTITY_CARD_NUMBER",
    "identity document number": "IDENTITY_DOCUMENT_NUMBER",
    "tax identification number": "TAX_ID",
    "health insurance number": "HEALTH_INSURANCE_NUMBER",
    "health insurance id number": "HEALTH_INSURANCE_NUMBER",
    "national health insurance number": "HEALTH_INSURANCE_NUMBER",
    "insurance number": "INSURANCE_NUMBER",
    "registration number": "REGISTRATION_NUMBER",
    "student id number": "STUDENT_ID",
    "flight number": "FLIGHT_NUMBER",
    "reservation number": "RESERVATION_NUMBER",
    "train ticket number": "TRAIN_TICKET_NUMBER",
    "transaction number": "TRANSACTION_NUMBER",
    "license plate number": "LICENSE_PLATE",
    "vehicle registration number": "VEHICLE_REGISTRATION",
    "serial number": "SERIAL_NUMBER",
    "digital signature": "DIGITAL_SIGNATURE",
    "username": "USERNAME",
    "social media handle": "USERNAME",
    "password": "PASSWORD",
}


def _anonymization_config():
    return config.get("anonymization", {})


def _get_anonymization_bool(key, default):
    value = _anonymization_config().get(key, default)
    return strtobool(str(value))


def _get_anonymization_float(key, default):
    value = _anonymization_config().get(key, default)
    return float(value)


def _is_gliner_available():
    return find_spec("gliner") is not None


def build_gliner_recognizers(device=None, cache_dir=None):
    if not _get_anonymization_bool("gliner_enabled", "true"):
        return []

    if not _is_gliner_available():
        print(
            "GLiNER anonymization recognizer is enabled but the 'gliner' package is not installed. "
            "Install presidio_analyzer[gliner] or gliner to enable broad PII detection."
        )
        return []

    global GLiNERRecognizer
    if GLiNERRecognizer is None:
        from presidio_analyzer.predefined_recognizers import GLiNERRecognizer

    anonymization_config = _anonymization_config()
    model_name = anonymization_config.get("gliner_model_name", DEFAULT_GLINER_MODEL)
    threshold = _get_anonymization_float("gliner_threshold", "0.30")
    flat_ner = _get_anonymization_bool("gliner_flat_ner", "false")
    multi_label = _get_anonymization_bool("gliner_multi_label", "true")
    map_location = anonymization_config.get("gliner_device") or device
    model_kwargs = {"cache_dir": cache_dir} if cache_dir else {}

    return [
        GLiNERRecognizer(
            model_name=model_name,
            entity_mapping=gliner_entity_mapping,
            supported_language=language,
            threshold=threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
            map_location=map_location,
            **model_kwargs,
        )
        for language in supported_languages
    ]


class AnonymizerModels:
    def __init__(self):
        self.models = None
        self.load_lock = Lock()
        self.last_model_use = time.time()
        self._device = None
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

    @property
    def device(self):
        # Defer CUDA probing until the device is actually needed. This keeps
        # CPU-only workers from opening the NVIDIA driver at import time.
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def load_models(self):
        with self.load_lock:
            if self.models is None:
                print('Loading analyzer and anonymizer')
                # Import presidio lazily so that workers which never anonymize
                # (e.g. video, text, image) do not pay the ~600 MiB baseline.
                global AnalyzerEngine, AnonymizerEngine, NerModelConfiguration, TransformersNlpEngine
                if AnalyzerEngine is None:
                    from presidio_analyzer import AnalyzerEngine
                if AnonymizerEngine is None:
                    from presidio_anonymizer import AnonymizerEngine
                if NerModelConfiguration is None or TransformersNlpEngine is None:
                    from presidio_analyzer.nlp_engine import NerModelConfiguration, TransformersNlpEngine

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
                    supported_languages=supported_languages
                )
                for recognizer in build_gliner_recognizers(device=self.device, cache_dir=self.cache_dir):
                    analyzer.registry.add_recognizer(recognizer)
                self.models = {
                    'analyzer': analyzer,
                    'anonymizer': AnonymizerEngine(),
                }
                self.last_model_use = time.time()

    def anonymize(self, text, lang):
        self.load_models()
        if lang not in supported_languages:
            raise NotImplementedError("Only English and French are implemented at the moment.")
        analyzer_results = self.models['analyzer'].analyze(text=text, language=lang)
        anonymized = self.models['anonymizer'].anonymize(text, analyzer_results=analyzer_results)
        self.last_model_use = time.time()
        return anonymized.text

    def unload_model(self, unload_period=ANONYMIZER_UNLOAD_WAITING_PERIOD):
        """
        Unloads the presidio/GLiNER stack if it has not been used recently.
        Returns the list of unloaded model names, or an empty list.
        """
        unloaded = []
        with self.load_lock:
            if time.time() - self.last_model_use > unload_period:
                self.models = None
                gc.collect()
                unloaded = ['analyzer', 'anonymizer']
        return unloaded


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
