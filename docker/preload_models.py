import argparse
import os

from sentence_transformers import SentenceTransformer
from transformers import MarianMTModel, MarianTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
from presidio_analyzer.nlp_engine import NerModelConfiguration, TransformersNlpEngine

try:
    from gliner import GLiNER
except ImportError:
    GLiNER = None


FALLBACK_DEFAULT_GLINER_MODEL = "urchade/gliner_multi_pii-v1"
FALLBACK_PRESIDIO_MODEL_MAPPING = {
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


def get_presidio_model_settings() -> tuple[str, dict[str, str]]:
    try:
        from graphai.core.retrieval.anonymization import (
            DEFAULT_GLINER_MODEL,
            mapping as presidio_model_mapping,
        )
    except ImportError:
        print("graphai package not available during preload; using built-in NER defaults")
        return FALLBACK_DEFAULT_GLINER_MODEL, FALLBACK_PRESIDIO_MODEL_MAPPING

    return DEFAULT_GLINER_MODEL, presidio_model_mapping


def preload_sentence_models(cache_dir: str) -> None:
    sentence_models = [
        "sentence-transformers/all-MiniLM-L12-v2",
        "OrdalieTech/Solon-embeddings-large-0.1",
    ]
    for model in sentence_models:
        print(f"Preloading sentence model: {model}")
        SentenceTransformer(model, cache_folder=cache_dir)


def preload_translation_models(cache_dir: str) -> None:
    translation_models = [
        "Helsinki-NLP/opus-mt-tc-big-en-fr",
        "Helsinki-NLP/opus-mt-tc-big-fr-en",
        "Helsinki-NLP/opus-mt-de-en",
        "Helsinki-NLP/opus-mt-it-en",
        "Helsinki-NLP/opus-mt-en-de",
        "Helsinki-NLP/opus-mt-en-it",
    ]
    for model in translation_models:
        print(f"Preloading translation model: {model}")
        if "tc-big" in model:
            MarianTokenizer.from_pretrained(model, cache_dir=cache_dir)
            MarianMTModel.from_pretrained(model, cache_dir=cache_dir)
        else:
            AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
            AutoModelForSeq2SeqLM.from_pretrained(model, cache_dir=cache_dir)


def preload_ner_models(cache_dir: str) -> None:
    _, presidio_model_mapping = get_presidio_model_settings()

    nlp_models = [
        {
            "lang_code": "en",
            "model_name": {
                "spacy": "en_core_web_sm",
                "transformers": "Davlan/distilbert-base-multilingual-cased-ner-hrl",
            },
        },
        {
            "lang_code": "fr",
            "model_name": {
                "spacy": "fr_core_news_sm",
                "transformers": "Davlan/distilbert-base-multilingual-cased-ner-hrl",
            },
        },
    ]
    print("Preloading Presidio / NER models")
    TransformersNlpEngine(
        models=nlp_models,
        ner_model_configuration=NerModelConfiguration(
            model_to_presidio_entity_mapping=presidio_model_mapping,
            labels_to_ignore=["O"],
        ),
    )


def preload_pii_models(cache_dir: str) -> None:
    default_gliner_model, _ = get_presidio_model_settings()

    if GLiNER is None:
        print("Skipping GLiNER PII model preload because gliner is not installed")
        return

    print(f"Preloading GLiNER PII model: {default_gliner_model}")
    GLiNER.from_pretrained(default_gliner_model, cache_dir=cache_dir)


def preload_all(cache_dir: str) -> None:
    preload_sentence_models(cache_dir)
    preload_translation_models(cache_dir)
    preload_ner_models(cache_dir)
    preload_pii_models(cache_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preload GraphAI Hugging Face models.")
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_HOME", "/opt/models/huggingface"),
        help="Target cache directory for Hugging Face and sentence-transformers models.",
    )
    parser.add_argument(
        "--group",
        choices=["all", "sentence", "translation", "ner", "pii"],
        default="all",
        help="Which model group to preload.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.group == "sentence":
        preload_sentence_models(args.cache_dir)
    elif args.group == "translation":
        preload_translation_models(args.cache_dir)
    elif args.group == "ner":
        preload_ner_models(args.cache_dir)
    elif args.group == "pii":
        preload_pii_models(args.cache_dir)
    else:
        preload_all(args.cache_dir)

    print(f"Hugging Face model preload finished for group: {args.group}")
