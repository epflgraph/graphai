import hashlib
import json
from bisect import bisect
from itertools import chain

import fingerprint
import langdetect
import numpy as np
import openai
import pysbd
import tiktoken
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from graphai.core.common.config import config
from graphai.core.common.json_repair.json_repair import repair_json
from graphai.core.common.gpt_message_presets import (
    generate_lecture_summary_message,
    generate_generic_summary_message,
    generate_academic_entity_summary_message,
)

TRANSLATION_LIST_SEPARATOR = ' [{[!!SEP!!]}] '
CHATGPT_COSTS_PER_1K = {
    'gpt-3.5-turbo-1106': {
        'prompt_tokens': 0.001,
        'completion_tokens': 0.002
    }
}


def translation_list_to_text(str_or_list):
    if not isinstance(str_or_list, list):
        return str_or_list
    str_or_list = [x if x is not None else '' for x in str_or_list]
    return TRANSLATION_LIST_SEPARATOR.join(str_or_list)


def translation_text_back_to_list(s, return_list=False):
    results = s.split(TRANSLATION_LIST_SEPARATOR)
    if len(results) == 1 and not return_list:
        return results[0]
    return results


def md5_text(s):
    """
    Computes the md5 hash of a string
    Args:
        s: The string

    Returns:
        MD5 hash
    """
    return hashlib.md5(s.encode('utf8')).hexdigest()


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
        return md5_text(translation_list_to_text(s)) + '_' + src + '_' + tgt


def perceptual_hash_text(s):
    """
    Computes the perceptual hash of a strong
    Args:
        s: String to hash
        min_window_length: Minimum window length for k-grams
        max_window_length: Maximum window length for k-grams
        hash_len: Length of the hash

    Returns:
        Perceptual hash of string
    """
    hash_len = 32
    string_length = len(s)
    window_lengths = [1, 4, 10, 20, 50]
    kgram_lengths = [int(np.ceil(x / 2)) for x in window_lengths]
    length_index = bisect(window_lengths, int(np.ceil(string_length / hash_len))) - 1
    window_length = window_lengths[length_index]
    kgram_length = kgram_lengths[length_index]

    fprinter = fingerprint.Fingerprint(kgram_len=kgram_length, window_len=window_length, base=10, modulo=256)
    try:
        hash_numbers = fprinter.generate(str=s)
        if len(hash_numbers) > hash_len:
            sample_indices = np.linspace(start=0, stop=len(hash_numbers) - 1, num=hash_len - 1, endpoint=False).tolist()
            sample_indices.append(len(hash_numbers) - 1)
            sample_indices = [int(x) for x in sample_indices]
            hash_numbers = [hash_numbers[i] for i in sample_indices]
        elif len(hash_numbers) < hash_len:
            hash_numbers = hash_numbers + [(0, 0)] * (32 - len(hash_numbers))
        fp_result = ''.join([f"{n[0]:02x}" for n in hash_numbers])
    except fingerprint.FingerprintException:
        fp_result = ''.join(['0'] * 64)
    return "%s_%02d_%02d" % (fp_result, window_length, kgram_length)


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


def count_tokens_for_openai(text, model="cl100k_base"):
    """
    Counts the number of tokens in a given text for a given OpenAI model
    Args:
        text: The text to tokenize
        model: The OpenAI model to use

    Returns:
        Number of tokens
    """
    encoding = tiktoken.get_encoding(model)
    return len(encoding.encode(text))


def force_dict_to_text(t):
    if isinstance(t, dict):
        return json.dumps(t)
    return t


def generate_summary_text_token(text, text_type='text', summary_type='summary'):
    assert summary_type in ['summary', 'cleanup']
    if summary_type == 'summary':
        assert text_type in ['lecture', 'academic_entity', 'text']

    text = force_dict_to_text(text)
    token = md5_text(text) + '_' + text_type + '_' + summary_type
    return token


def generate_completion_type_dict(text_type, completion_type):
    return {
        'input_type': text_type,
        'completion_type': completion_type
    }


def convert_text_or_dict_to_text(text_or_dict):
    if isinstance(text_or_dict, dict):
        text = "\n\n".join([f"{k}: {v}" for k, v in text_or_dict.items()])
    else:
        text = f"Text: {text_or_dict}"
    return text


def update_token_count(old_token_counts=None, token_counts=None):
    if token_counts is None or len(token_counts) == 0:
        return old_token_counts
    if old_token_counts is None or len(old_token_counts) == 0:
        return token_counts
    return {
        k: token_counts[k] + old_token_counts[k]
        for k in token_counts
    }


def compute_chatgpt_request_cost(token_counts, model_type):
    return sum([token_counts[x] * CHATGPT_COSTS_PER_1K[model_type][x] / 1000
                for x in CHATGPT_COSTS_PER_1K[model_type]])


class ChatGPTSummarizer:
    def __init__(self):
        try:
            print("Reading OpenAI API key from config")
            self.api_key = config['openai']['api_key']
        except Exception:
            self.api_key = None

        if self.api_key is None:
            print(
                "The OpenAI API key could not be found in the config file. "
                "Make sure to add a [openai] section with the api_key parameter. "
                "OpenAI API endpoints cannot be used as there is no default API key."
            )

    def establish_connection(self):
        """
        Ensures that an API key exists and sets it as the OpenAI key
        Returns:
            Boolean indicating whether an API key was found
        """
        if self.api_key is not None:
            openai.api_key = self.api_key
            return True
        else:
            return False

    def _generate_completion(self, text, system_message, max_len=None, temperature=1.0, top_p=1.0,
                             simulate=False, verbose=False, timeout=60):
        """
        Internal method, generates a chat completion, which is the OpenAI API endpoint for ChatGPT interactions
        Args:
            text: The text to be provided in the "user" role, i.e. the text that is to be processed by ChatGPT
                  If this argument is a list, its elements are assumed to be, alternately, user and assistant
                  messages. This enables the handling of a back-and-forth conversation.
            system_message: The text to be provided in the "system" role, which provides directives to ChatGPT
            max_len: Approximate maximum length of the response in words

        Returns:
            Results returned by ChatGPT, a flag that is True if there were too many tokens, and the total # of tokens
            if (and only if) the request is successful (0 otherwise).
            A (None, True, 0) result means that the completion failed because the message had too many tokens,
            while a (None, False, 0) result indicates a different error (e.g. failed connection).
        """
        assert isinstance(text, str) or (isinstance(text, list) and all(isinstance(t, str) for t in text)) \
               or (isinstance(text, list) and all(isinstance(t, dict) for t in text))
        if isinstance(text, str):
            text = [{'role': 'user', 'content': text}]
        elif isinstance(text[0], str):
            text = [{"role": "user", "content": text[i]} if i % 2 == 0
                    else {"role": "assistant", "content": text[i]}
                    for i in range(len(text))]
        has_api_key = self.establish_connection()
        # If there's no api key, only simulate mode is possible
        if not has_api_key and not simulate:
            return None, text, False, None
        concatenated_text = '\n'.join([t['content'] for t in text])
        text_token_count = count_tokens_for_openai(concatenated_text)
        system_token_count = count_tokens_for_openai(system_message)
        if max_len is None:
            max_len = 3 * text_token_count
        else:
            max_len = int(2 * max_len)
        # We count the approximate number of tokens in order to choose the right model (i.e. context size)
        approx_token_count = text_token_count + system_token_count + max_len
        print(approx_token_count)
        if approx_token_count < 16384:
            # This is now the 16K context model for 3.5
            model_type = 'gpt-3.5-turbo-1106'
        else:
            # If the token count is above 16384, the text is too large and we can't summarize it
            return None, text, True, None
        if not simulate:
            messages = [{"role": "system", "content": system_message}]
            messages += text
            try:
                # Generate the completion
                client = openai.OpenAI(api_key=config['openai']['api_key'])
                completion = client.chat.completions.create(
                    model=model_type,
                    messages=messages,
                    max_tokens=max_len,
                    temperature=temperature,
                    top_p=top_p,
                    timeout=timeout
                )
                if verbose:
                    print(completion)
            except openai.OpenAIError as e:
                # We check to see if the exception was caused by too many tokens in the input
                print(e)
                if "This model's maximum context length is" in str(e):
                    return None, text, True, None
                else:
                    return None, text, False, None
            except Exception as e:
                # Any error other than "too many tokens" is dealt with here
                print(e)
                return None, text, False, None
            token_count_dict = dict(completion.usage)
            final_result = completion.choices[0].message.content
        else:
            input_token_estimate = system_token_count + text_token_count
            output_token_estimate = max_len
            token_count_dict = {
                'prompt_tokens': input_token_estimate,
                'completion_tokens': output_token_estimate,
                'total_tokens': input_token_estimate + output_token_estimate
            }
            final_result = None
        cost = compute_chatgpt_request_cost(token_count_dict, model_type)
        token_count_dict['cost'] = cost
        return final_result, text, False, token_count_dict

    def _summarize(self, text, system_message, temperature=1.0, top_p=0.3, simulate=False, max_len=None):
        results, message_chain, too_many_tokens, n_total_tokens = \
            self._generate_completion(text, system_message, temperature=temperature, top_p=top_p, simulate=simulate,
                                      timeout=60, max_len=max_len)
        token_count = n_total_tokens
        if results is None:
            # This includes simulate==True
            return None, system_message, too_many_tokens, token_count

        # Making sure the results are in a valid JSON format
        results, message_chain, too_many_tokens, n_total_tokens = \
            self.make_sure_json_is_valid(results, message_chain, system_message,
                                         temperature=temperature, top_p=top_p)
        token_count = update_token_count(token_count, token_count)

        return results, system_message, too_many_tokens, token_count

    def summarize_academic_entity(self, input_dict, long_len=200, short_len=32, title_len=10,
                                  temperature=1.0, top_p=0.3, simulate=False):
        entity = input_dict['entity']
        name = input_dict['name']
        subtype = input_dict['subtype']
        possible_subtypes = input_dict['possible_subtypes']
        text = input_dict['text']
        categories = input_dict['categories']
        user_message = f'[entity] = {entity}\n[name] = {name}\n' \
                       f'[subtype] = {subtype}\n[possible subtypes] = {possible_subtypes}\n' \
                       f'[text] = "{text}"\n[categories] = {categories}\n' \
                       f'[n words (long)] = {long_len}\n[n words (short)] = {short_len}\n' \
                       f'[n words (title)] = {title_len}\n'
        system_message, assistant_message = generate_academic_entity_summary_message()
        message_chain = [
            {'role': 'user', 'content': user_message},
            {'role': 'assistant', 'content': assistant_message}
        ]
        max_len = int(1.5 * (long_len + short_len))
        return self._summarize(message_chain, system_message,
                               temperature, top_p, simulate, max_len)

    def summarize_generic(self, text, long_len=200, short_len=32, title_len=10,
                          temperature=1.0, top_p=0.3, simulate=False):
        assert isinstance(text, str) or isinstance(text, dict)
        system_message = generate_generic_summary_message(long_len, short_len, title_len)
        text = convert_text_or_dict_to_text(text)
        max_len = int(1.5 * (long_len + short_len + title_len))
        return self._summarize(text, system_message, temperature, top_p, simulate, max_len)

    def summarize_lecture(self, slide_to_concepts, long_len=200, short_len=32, title_len=10,
                          temperature=1.0, top_p=0.3, simulate=False):
        assert isinstance(slide_to_concepts, dict)
        system_message = generate_lecture_summary_message(long_len, short_len, title_len)
        slide_to_concepts = {k: v for k, v in slide_to_concepts.items() if len(v) > 0}
        slide_numbers_sorted = sorted(list(slide_to_concepts.keys()))
        text = '\n\n'.join([f'Slide {i}: ' + '; '.join(slide_to_concepts[i]) for i in slide_numbers_sorted])
        max_len = int(1.5 * (long_len + short_len + title_len))
        return self._summarize(text, system_message, temperature, top_p, simulate, max_len)

    def make_sure_json_is_valid(self, results, messages, system_message, temperature=1.0, top_p=0.3,
                                n_retries=1):
        # Trying to directly parse the JSON...
        assert isinstance(messages, list) and all(isinstance(message, dict) for message in messages)
        try:
            print('Parsing JSON...')
            prev_results_json = json.loads(results)
            print('JSON parsed successfully')
            return prev_results_json, (messages + [{'role': 'assistant', 'content': results}]), False, dict()
        except json.JSONDecodeError:
            print(results)
            print('Results not in JSON, retrying')
        # Trying to fix the JSON algorithmically...
        try:
            print('Trying to algorithmically fix the JSON...')
            repaired_json = repair_json(results)
            prev_results_json = json.loads(repaired_json)
            print('JSON parsed successfully')
            return prev_results_json, (messages + [{'role': 'assistant', 'content': repaired_json}]), False, dict()
        except Exception:
            print('Algorithmic JSON fix unsuccessful, retrying')
        # Trying to fix the JSON by asking ChatGPT again...
        messages = messages + [{'role': 'assistant', 'content': results}]
        retried = 0
        while retried < n_retries:
            try:
                correction_message = [
                    {
                        'role': 'user',
                        'content': 'The results were not in a JSON format. Improve your previous response by '
                                   'making sure the results are in JSON. Be sure to escape double quotes and '
                                   'backslashes. Double quotes should be escaped to \\", and backslashes to \\\\.'
                                   ' Do NOT return the exact same results as the input: the input is not a valid '
                                   'JSON and the results must be a valid JSON.'
                    }
                ]
                results, message_chain, too_many_tokens, n_total_tokens = \
                    self._generate_completion(
                        messages + correction_message,
                        system_message, temperature=temperature, top_p=top_p,
                        timeout=20
                    )
                if results is None:
                    return None, message_chain, too_many_tokens, n_total_tokens
                print('Parsing JSON...')
                repaired_json = repair_json(results)
                results_json = json.loads(repaired_json)
                print('JSON parsed successfully')
                return results_json, (message_chain + [{'role': 'assistant', 'content': repaired_json}]), \
                    too_many_tokens, n_total_tokens
            except json.JSONDecodeError:
                retried += 1
        raise Exception(f"Could not get ChatGPT to produce a JSON result, "
                        f"here are the final results produced: {results}")

    def cleanup_text(self, text_or_dict, text_type='slide', handwriting=True, temperature=1, top_p=0.3,
                     simulate=False):
        """
        Cleans up dirty text that contains typos and unnecessary line breaks.
        Args:
            text_or_dict: Input, can be one string or a dictionary of strings
            text_type: Source of the input, 'slide' by default
            handwriting: Whether the text comes from OCR on handwriting
            temperature: Temperature for GPT, determines creativity/determinism
            top_p: Top P for GPT, determines which parts of the distribution are considered.
            simulate: Whether to just simulate the costs

        Returns:
            Results, system message sent to GPT, whether there were too many tokens, and # of tokens/cost
        """
        if handwriting:
            system_message = "You will be given the contents of a %s, " \
                             "which result from optical character recognition " \
                             "and therefore are very messy. " % text_type
        else:
            system_message = "You will be given the contents of a %s, which contain typos. " % text_type
        text = convert_text_or_dict_to_text(text_or_dict)
        system_message += "Your task is to clean up the contents. " \
                          "The text could potentially contain typos, incorrect grammar, scrambled sentences, " \
                          "and mathematical notation. " \
                          "Return the results in the following JSON format:\n" \
                          "```\n" \
                          "{\"language\": [LANGUAGE],\n" \
                          "\"subject\": [SUBJECT MATTER],\n" \
                          "\"cleaned\": [CLEANED UP TEXT]}.\n" \
                          "```\n" \
                          "DO NOT provide any explanations as to how you performed the cleanup. " \
                          "DO NOT translate or summarize the text.\n" \
                          "Clean the text up by performing the following steps, in order:\n" \
                          "1. Detect the language of the text: [LANGUAGE].\n" \
                          "2. Find the Wikipedia article that best describes " \
                          "the subject matter of the text: [SUBJECT MATTER].\n" \
                          "3. Some sentences may have been split by line breaks into separate lines. Try to " \
                          "detect and fix them, making sure that the entire sentence is in one line. " \
                          "Put punctuation marks at the end of every detected sentence.\n" \
                          "4. Correct ALL typos. Treat words that exist neither in [LANGUAGE] nor in English " \
                          "as typos. Treat words that are completely unrelated to the [SUBJECT MATTER] or " \
                          "the other words in the text as typos. Correct every single typo to the closest word " \
                          "in [LANGUAGE] (or failing that, English) that fits within the [SUBJECT MATTER]. " \
                          "If you encounter a non-word and cannot correct it to anything meaningful, remove " \
                          "it from the text entirely.\n" \
                          "5. Remove all mathematical formulae.\n" \
                          "Make sure the results are in JSON, and that double quotes and backslashes are escaped " \
                          "properly (double quotes should be \\\" and backslashes should be \\\\)."

        # Make a call to ChatGPT to generate initial results. These will still be full of typos.
        text = text + "\n\nBe sure to respond in JSON format!"
        results, message_chain, too_many_tokens, n_total_tokens = \
            self._generate_completion(text, system_message, temperature=temperature, top_p=top_p, simulate=simulate,
                                      timeout=20)
        print('FIRST')
        token_count = n_total_tokens
        if not simulate:
            if results is None:
                return None, system_message, too_many_tokens, token_count

            results, message_chain, too_many_tokens, n_total_tokens = \
                self.make_sure_json_is_valid(results, message_chain, system_message,
                                             temperature=temperature, top_p=top_p)
            token_count = update_token_count(token_count, n_total_tokens)
            if results is None:
                return None, system_message, too_many_tokens, token_count
        else:
            # If we're simulating the run, then there's no need to double check the JSON
            # We don't count potential JSON corrections in our estimation because they are quite rare anyway
            # Since no actual result has been generated, we assume that the result is roughly the same as the
            # original input, since token-wise they should actually be pretty close.
            message_chain = [{'role': 'user', 'content': text}, {'role': 'assistant', 'content': text}]

        # Now make a second call to ChatGPT, asking it to improve its initial results.
        correction_message = \
            [{
                'role': 'user',
                'content': 'There are still typos in the text. Try to improve your previous response by correcting '
                           'more typos. Do not provide any explanations on how you fix typos. Make sure the results '
                           'are in JSON format.'
            }]
        results, message_chain, too_many_tokens, n_total_tokens = \
            self._generate_completion(
                message_chain + correction_message, system_message, max_len=len(text) if simulate else None,
                temperature=temperature, top_p=top_p, simulate=simulate, timeout=20)
        print('SECOND')
        token_count = update_token_count(token_count, n_total_tokens)

        if results is None or simulate:
            # If simulate is True, the results will always be None, but this if is as explicit as possible
            # for the sake of readability
            # Again, simulate=True means that we do not try to fix the JSON
            return None, system_message, too_many_tokens, token_count

        results, message_chain, too_many_tokens, n_total_tokens = \
            self.make_sure_json_is_valid(results, message_chain,
                                         system_message,
                                         temperature=temperature, top_p=top_p)
        print(results)
        token_count = update_token_count(token_count, n_total_tokens)
        if results is None:
            return None, system_message, too_many_tokens, token_count

        # Return the results of the second call
        return results, system_message, too_many_tokens, token_count


class TranslationModels:
    def __init__(self):
        self.models = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_models(self):
        """
        Loads Huggingface translation and tokenization models plus a pysbd segmenter
        Returns:
            None
        """
        if self.models is None:
            self.models = dict()
            print('Loading EN-FR')
            self.models['en-fr'] = dict()
            self.models['en-fr']['tokenizer'] = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
            self.models['en-fr']['model'] = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr"). \
                to(self.device)
            self.models['en-fr']['segmenter'] = pysbd.Segmenter(language='en', clean=False)
            print('Loading FR-EN')
            self.models['fr-en'] = dict()
            self.models['fr-en']['tokenizer'] = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
            self.models['fr-en']['model'] = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en"). \
                to(self.device)
            self.models['fr-en']['segmenter'] = pysbd.Segmenter(language='fr', clean=False)

    def get_device(self):
        return self.device

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
            input_ids = tokenizer.encode(sentence, return_tensors="pt")
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

        return full_result, False

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
        text = translation_text_back_to_list(text, return_list=True)
        results = [self._translate(current_text, tokenizer, model, segmenter) for current_text in text]
        return translation_list_to_text([x[0] for x in results]), any([x[1] for x in results])
