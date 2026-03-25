import argparse
import os

from sentence_transformers import SentenceTransformer
from transformers import MarianTokenizer, MarianMTModel, AutoTokenizer, AutoModelForSeq2SeqLM
from presidio_analyzer.nlp_engine import NerModelConfiguration, TransformersNlpEngine


def preload_huggingface(cache_dir: str = "/opt/models/huggingface") -> None:
    sentence_models = [
        "sentence-transformers/all-MiniLM-L12-v2",
        "OrdalieTech/Solon-embeddings-large-0.1",
    ]
    for model in sentence_models:
        SentenceTransformer(model, cache_folder=cache_dir)

    translation_models = [
        "Helsinki-NLP/opus-mt-tc-big-en-fr",
        "Helsinki-NLP/opus-mt-tc-big-fr-en",
        "Helsinki-NLP/opus-mt-de-en",
        "Helsinki-NLP/opus-mt-it-en",
    ]
    for model in translation_models:
        if "tc-big" in model:
            MarianTokenizer.from_pretrained(model, cache_dir=cache_dir)
            MarianMTModel.from_pretrained(model, cache_dir=cache_dir)
        else:
            AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
            AutoModelForSeq2SeqLM.from_pretrained(model, cache_dir=cache_dir)

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
    TransformersNlpEngine(
        models=nlp_models,
        ner_model_configuration=NerModelConfiguration(
            model_to_presidio_entity_mapping=None,
            labels_to_ignore=["O"],
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preload GraphAI Hugging Face models.")
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_HOME", "/opt/models/huggingface"),
        help="Target cache directory for Hugging Face and sentence-transformers models.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preload_huggingface(args.cache_dir)
    print("Hugging Face model preload finished")
