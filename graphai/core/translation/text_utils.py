from itertools import chain

import langdetect
from ftfy import fix_encoding
import numpy as np
import pysbd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from multiprocessing import Lock
import time
import gc

from graphai.core.common.common_utils import convert_list_to_text, convert_text_back_to_list
from graphai.core.common.fingerprinting import md5_text
from graphai.core.common.config import config

HUGGINGFACE_MAX_TOKENS = 512
# 3 hours
HUGGINGFACE_UNLOAD_WAITING_PERIOD = 3 * 3600.0


def generate_src_tgt_dict(src, tgt):
    """
    Creates a source language and target language dictionary for translation
    Args:
        src: Source lang
        tgt: Target lang

    Returns:
        dict
    """
    return {'source_lang': src, 'target_lang': tgt}


def generate_translation_text_token(s, src, tgt):
    """
    Generates an md5-based token for a string
    Args:
        s: The string
        src: Source lang
        tgt: Target lang

    Returns:
        Token
    """
    assert isinstance(s, str) or isinstance(s, list)
    if isinstance(s, str):
        return md5_text(s) + '_' + src + '_' + tgt
    else:
        return md5_text(convert_list_to_text(s)) + '_' + src + '_' + tgt


def detect_text_language(s):
    """
    Detects the language of the provided string
    Args:
        s: String to detect language for

    Returns:
        Language of the string
    """
    if s is None or s == '':
        return None
    try:
        return langdetect.detect(s)
    except langdetect.lang_detect_exception.LangDetectException:
        return None


def compute_slide_tfidf_scores(list_of_sets, min_freq=1):
    sep = ' [{!!}] '
    list_of_strings = [sep.join(s) for s in list_of_sets]
    vectorizer = TfidfVectorizer(analyzer=lambda x: x.split(sep), norm=None, min_df=min_freq)
    tfidf_matrix = vectorizer.fit_transform(list_of_strings)
    scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
    scores = scores.tolist()
    words_kept = set(vectorizer.vocabulary_.keys())
    return scores, words_kept


def find_set_cover(list_of_sets, coverage=1.0, scores=None):
    assert isinstance(list_of_sets, list) and all(isinstance(s, set) for s in list_of_sets)
    elements = set(chain.from_iterable(list_of_sets))
    covered = set()
    cover = list()
    if scores is None:
        scores = [1] * len(list_of_sets)
    current_coverage = 0
    while current_coverage < coverage:
        print(len(covered), len(elements))
        subset = max(list_of_sets, key=lambda s: len(s - covered) * scores[list_of_sets.index(s)])
        cover.append(subset)
        covered |= subset
        new_coverage = len(covered) / len(elements)
        # If the selection of this subset has not increased coverage, then we're stuck in a loop with all remaining
        # scores being equal to 0, and we should stop the algorithm.
        if new_coverage == current_coverage:
            print(f'Max coverage reached at {new_coverage}')
            break
        current_coverage = new_coverage

    cover_indices = [list_of_sets.index(s) for s in cover]
    return cover, cover_indices


def find_best_slide_subset(slides_and_concepts, coverage=1.0, priorities=True, min_freq=2):
    if priorities:
        scores, words_to_keep = compute_slide_tfidf_scores(slides_and_concepts, min_freq)
        slides_and_concepts = [{w for w in s if w in words_to_keep} for s in slides_and_concepts]
    else:
        scores = None
    return find_set_cover(slides_and_concepts, coverage, scores)


class TranslationModels:
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
        """
        Loads Huggingface translation and tokenization models plus a pysbd segmenter
        Returns:
            None
        """
        with self.load_lock:
            if self.models is None:
                self.models = dict()
                print('Loading EN-FR')
                self.models['en-fr'] = dict()
                self.models['en-fr']['tokenizer'] = MarianTokenizer.from_pretrained(
                    "Helsinki-NLP/opus-mt-tc-big-en-fr",
                    cache_dir=self.cache_dir)
                self.models['en-fr']['model'] = MarianMTModel.from_pretrained(
                    "Helsinki-NLP/opus-mt-tc-big-en-fr",
                    cache_dir=self.cache_dir).to(self.device)
                self.models['en-fr']['segmenter'] = pysbd.Segmenter(language='en', clean=False)
                print('Loading FR-EN')
                self.models['fr-en'] = dict()
                self.models['fr-en']['tokenizer'] = MarianTokenizer.from_pretrained(
                    "Helsinki-NLP/opus-mt-tc-big-fr-en",
                    cache_dir=self.cache_dir)
                self.models['fr-en']['model'] = MarianMTModel.from_pretrained(
                    "Helsinki-NLP/opus-mt-tc-big-fr-en",
                    cache_dir=self.cache_dir).to(self.device)
                self.models['fr-en']['segmenter'] = pysbd.Segmenter(language='fr', clean=False)
                print('Loading DE-EN')
                self.models['de-en'] = dict()
                self.models['de-en']['tokenizer'] = AutoTokenizer.from_pretrained(
                    "Helsinki-NLP/opus-mt-de-en",
                    cache_dir=self.cache_dir)
                self.models['de-en']['model'] = AutoModelForSeq2SeqLM.from_pretrained(
                    "Helsinki-NLP/opus-mt-de-en",
                    cache_dir=self.cache_dir).to(self.device)
                self.models['de-en']['segmenter'] = pysbd.Segmenter(language='de', clean=False)
                print('Loading IT-EN')
                self.models['it-en'] = dict()
                self.models['it-en']['tokenizer'] = AutoTokenizer.from_pretrained(
                    "Helsinki-NLP/opus-mt-it-en",
                    cache_dir=self.cache_dir)
                self.models['it-en']['model'] = AutoModelForSeq2SeqLM.from_pretrained(
                    "Helsinki-NLP/opus-mt-it-en",
                    cache_dir=self.cache_dir).to(self.device)
                self.models['it-en']['segmenter'] = pysbd.Segmenter(language='it', clean=False)
            self.last_model_use = time.time()

    def get_device(self):
        return self.device

    def get_last_usage(self):
        return self.last_model_use

    def unload_model(self, unload_period=HUGGINGFACE_UNLOAD_WAITING_PERIOD):
        deleted_models = None
        with self.load_lock:
            if time.time() - self.get_last_usage() > unload_period:
                if self.models is None:
                    deleted_models = list()
                else:
                    deleted_models = list(self.models.keys())
                    self.models = None
                    gc.collect()
        return deleted_models

    def _tokenize_and_get_model_output(self, sentence, tokenizer, model):
        """
        Internal method. Translates one single sentence.
        Args:
            sentence: Sentence to translate
            tokenizer: Tokenizer model
            model: Translation model

        Returns:
            Translation result or None if translation fails
        """
        try:
            print(self.device)
            input_ids = tokenizer.encode(sentence, return_tensors="pt", padding=True)
            if input_ids.shape[1] > HUGGINGFACE_MAX_TOKENS:
                return None
            input_ids = input_ids.to(self.device)
            outputs = model.generate(input_ids, max_length=512)
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return decoded
        except IndexError as e:
            print(e)
            return None

    def _translate(self, text, tokenizer, model, segmenter):
        """
        Internal method. Translates entire text, sentence by sentence.
        Args:
            text: Text to translate
            tokenizer: Tokenizer model
            model: Translation model
            segmenter: Sentence segmenter

        Returns:
            Translated text, plus a flag denoting whether any of the sentences was too long and unpunctuated
        """
        sentences = segmenter.segment(text.replace('\n', '. '))
        sentences = [sentence.strip() for sentence in sentences]
        full_result = ''
        for sentence in sentences:
            print(sentence)
            if len(sentence) == 0:
                continue
            if len(sentence) < 4:
                full_result += ' ' + sentence
                continue
            decoded = self._tokenize_and_get_model_output(sentence, tokenizer, model)
            print(decoded)
            if decoded is None:
                return None, True
            full_result += ' ' + decoded

        return fix_encoding(full_result), False

    def translate(self, text, how='en-fr'):
        """
        Translates provided text
        Args:
            text: Text to translate
            how: source-target language

        Returns:
            Translated text and 'unpunctuated text too long' flag
        """
        self.load_models()
        if how not in self.models.keys():
            raise NotImplementedError("Source or target language not implemented")
        if text is None or len(text) == 0:
            return None, False
        tokenizer = self.models[how]['tokenizer']
        model = self.models[how]['model']
        segmenter = self.models[how]['segmenter']
        text = convert_text_back_to_list(text, return_list=True)
        results = [self._translate(current_text, tokenizer, model, segmenter) for current_text in text]
        return convert_list_to_text([x[0] for x in results]), any([x[1] for x in results]), [x[1] for x in results]
