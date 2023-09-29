import configparser
import hashlib
import json
from bisect import bisect

import fingerprint
import langdetect
import numpy as np
import openai
import pysbd
import tiktoken
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from graphai.definitions import CONFIG_DIR


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
    return md5_text(s) + '_' + src + '_' + tgt


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


def generate_summary_text_token(text, text_type='text', summary_type='summary', len_class='normal', tone='info'):
    assert text_type in ['person', 'unit', 'concept', 'course', 'lecture', 'MOOC', 'publication', 'text']
    assert summary_type in ['summary', 'title']
    assert len_class in ['vshort', 'short', 'normal']
    assert tone in ['info', 'promo']

    text = force_dict_to_text(text)
    token = md5_text(text) + '_' + text_type + '_' + summary_type + '_' + len_class + '_' + tone
    return token


def generate_summary_type_dict(text_type, summary_type, len_class, tone):
    return {
        'input_type': text_type,
        'summary_type': summary_type,
        'summary_len_class': len_class,
        'summary_tone': tone
    }


class ChatGPTSummarizer:
    def __init__(self):
        config_contents = configparser.ConfigParser()
        try:
            print('Reading ChatGPT API key from file')
            config_contents.read(f'{CONFIG_DIR}/models.ini')
            self.api_key = config_contents['CHATGPT'].get('api_key', fallback=None)
        except Exception:
            self.api_key = None
        if self.api_key is None:
            print(f'Could not read file {CONFIG_DIR}/models.ini or '
                  f'file does not have section [CHATGPT], ChatGPT API '
                  f'endpoints cannot be used as there is no '
                  f'default API key.')

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

    def _generate_completion(self, text, system_message, max_len):
        """
        Internal method, generates a chat completion, which is the OpenAI API endpoint for ChatGPT interactions
        Args:
            text: The text to be provided in the "user" role, i.e. the text that is to be processed by ChatGPT
            system_message: The text to be provided in the "system" role, which provides directives to ChatGPT
            max_len: Approximate maximum length of the response in words

        Returns:
            Results returned by ChatGPT, a flag that is True if there were too many tokens, and the total # of tokens
            if (and only if) the request is successful (0 otherwise).
            A (None, True, 0) result means that the completion failed because the message had too many tokens,
            while a (None, False, 0) result indicates a different error (e.g. failed connection).
        """
        has_api_key = self.establish_connection()
        if not has_api_key:
            return None, False, 0
        # We count the approximate number of tokens in order to choose the right model (i.e. context size)
        approx_token_count = count_tokens_for_openai(text) + count_tokens_for_openai(system_message) + int(2 * max_len)
        if approx_token_count < 4096:
            model_type = 'gpt-3.5-turbo'
        elif 4096 < approx_token_count < 16384:
            model_type = 'gpt-3.5-turbo-16k'
        else:
            # If the token count is above 16384, the text is too large and we can't summarize it
            return None, True, 0
        try:
            # Generate the completion
            completion = openai.ChatCompletion.create(
                model=model_type,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": text}
                ],
                max_tokens=int(2 * max_len)
            )
            print(completion)
        except openai.error.InvalidRequestError as e:
            # We check to see if the exception was caused by too many tokens in the input
            print(e)
            if "This model's maximum context length is" in e:
                return None, True, 0
            else:
                return None, False, 0
        except Exception as e:
            # Any error other than "too many tokens" is dealt with here
            print(e)
            return None, False, 0
        return completion.choices[0].message.content, False, completion.usage.total_tokens

    def generate_summary(self, text_or_dict, text_type='lecture', summary_type='summary',
                         len_class='normal', tone='info', max_normal_len=100, max_short_len=40):
        """
        Generates a summary or a title for the provided text
        Args:
            text_or_dict: String or dictionary containing all the text to summarize and synthesize into one summary
            text_type: Type of text, e.g. "lecture", "course". Useful for summaries.
            summary_type: The type of the summary to be produced, either 'title' or 'summary' (default)
            len_class: Whether there's a constraint on the number of sentences ('vshort' for 1, 'short' for 2)
                       or not ('normal', default).
            tone: Whether to use a marketing tone ('promo') or an informative tone ('info', default)
            max_normal_len: Approximate maximum length of result (in words)

        Returns:
            Result of summarization, plus a flag indicating whether there were too many tokens
        """
        if isinstance(text_or_dict, dict):
            text_dict_fields = [x.lower() for x in text_or_dict.keys()]
        else:
            if text_type != "text":
                text_dict_fields = ["text"]
            else:
                text_dict_fields = list()

        # Telling the API what information is being provided on the entity
        if len(text_dict_fields) > 0:
            system_message = f"You will be given the {', '.join(text_dict_fields)} of a {text_type}."
        else:
            system_message = f"You will be given a {text_type}."

        # We have certain constraints on the length of the response.
        n_sentences = None
        max_len = max_normal_len
        if len_class == 'vshort':
            n_sentences = 1
            max_len = max_short_len / 2
        elif len_class == 'short':
            n_sentences = 2
            max_len = max_short_len
        if n_sentences is not None:
            sentences = f" {n_sentences}-sentence"
        else:
            sentences = ""
        max_len_str = f" with under {max_len} words."

        # Based on the text_type, we may have additional constraints.
        # This section should be expanded based on feedback
        exclude_name = (summary_type == 'summary' and n_sentences == 1) or \
                       (summary_type == 'title' and n_sentences is not None)
        additional_constraints = ""
        if text_type == "person":
            additional_constraints = " INCLUDE their job title and place of work in the response (if available)."
            if exclude_name:
                additional_constraints += " EXCLUDE the name of the person from the response."
        elif text_type == "unit":
            additional_constraints = " INCLUDE the institution that it is part of in the response (if available)."
            if exclude_name:
                additional_constraints += " EXCLUDE the name of the unit from the response."
        elif text_type == 'concept':
            if exclude_name:
                additional_constraints += " EXCLUDE the name of the concept from the response."
        elif text_type == 'course':
            additional_constraints = " INCLUDE the name of the professor teaching it (if available)."
            if exclude_name:
                additional_constraints += " EXCLUDE the name of the course from the response."
        elif text_type == 'MOOC':
            additional_constraints = " INCLUDE the name of the professor teaching it (if available)."
            if exclude_name:
                additional_constraints += " EXCLUDE the name of the MOOC from the response."
        elif text_type == 'lecture':
            if exclude_name:
                additional_constraints += " EXCLUDE the name of the lecture from the response."
        elif text_type == 'publication':
            additional_constraints += " EXCLUDE the paper's name from the response."
            if exclude_name:
                additional_constraints += " EXCLUDE the names of the authors from the response."
            else:
                additional_constraints = " INCLUDE the names of the first few authors in the response."

        # This is the main part that determines whether we get a title or a summary
        if summary_type == 'title':
            system_message += f" Generate a title for the {text_type}{max_len_str}."
        else:
            system_message += f" Generate a{sentences} summary for the " \
                              f"{text_type}{max_len_str}."

        # Adding the additional constraints
        system_message += additional_constraints

        if tone == 'promo':
            system_message += " Write in a promotional tone."
        else:
            system_message += " Write in a neutral, informative tone."

        # Now we compile the response format
        response_format = f"\"{summary_type}: "
        sample_response = ""
        # This section should also be expanded based on feedback
        if text_type == 'person':
            if n_sentences == 1:
                response_format += "[BRIEF DESCRIPTION OF CURRENT JOB]\""
                sample_response = \
                    f"\"{summary_type}: Associate Professor at EPFL working on social network analysis"
            elif n_sentences == 2:
                response_format += "[BRIEF DESCRIPTION OF CURRENT JOB], [BRIEF DESCRIPTION OF INTERESTS].\""
                sample_response = \
                    f"\"{summary_type}: Associate Professor at EPFL working on social network analysis, " \
                    f"with contributions to graph theory and graph neural networks\""
            else:
                response_format += "[RESPONSE]\""
        elif text_type == 'unit':
            if n_sentences is not None:
                response_format += "[BRIEF DESCRIPTION OF RESEARCH OR DEVELOPMENT AREAS]\""
                sample_response = \
                    f"\"{summary_type}: Laboratory at EPFL working on social network analysis and graph neural networks"
            else:
                response_format += "[RESPONSE]\""
        elif text_type == 'concept':
            if n_sentences is not None:
                response_format += "[BRIEF EXPLANATION OF THE CONCEPT]\""
                sample_response = \
                    f"\"{summary_type}: Algorithm in graph theory that finds a minimum cut in a given graph"
            else:
                response_format += "[RESPONSE]\""
        elif text_type == 'course':
            if n_sentences is not None:
                response_format += "[BRIEF DESCRIPTION OF COURSE CONTENTS]\""
                sample_response = \
                    f"\"{summary_type}: Course on graph theory and graph algorithms for BSc Computer Science students"
            else:
                response_format += "[RESPONSE]\""
        elif text_type == 'MOOC':
            if n_sentences is not None:
                response_format += "[BRIEF DESCRIPTION OF MOOC CONTENTS]\""
                sample_response = \
                    f"\"{summary_type}: Introductory-level MOOC on graph theory and graph algorithms"
            else:
                response_format += "[RESPONSE]\""
        elif text_type == 'lecture':
            if n_sentences is not None:
                response_format += "[BRIEF DESCRIPTION OF LECTURE CONTENTS]\""
                sample_response = \
                    f"\"{summary_type}: Lecture introducing directed acyclic graphs and presenting a number of theorems"
            else:
                response_format += "[RESPONSE]\""
        elif text_type == 'publication':
            if n_sentences is not None:
                response_format += "[BRIEF DESCRIPTION OF PUBLICATION CONTENTS]\""
                sample_response = \
                    f"\"{summary_type}: Paper by Dillenbourg et al. introducing the concept of collaborative learning"
            else:
                response_format += "[RESPONSE]\""
        else:
            response_format += "[RESPONSE]\""
        system_message += f"Give your response in the form: {response_format}"
        if sample_response != "":
            system_message += f"\nHere's an example of an acceptable response: {sample_response}"

        if isinstance(text_or_dict, dict):
            text = "\n\n".join([f"{k}: {v}" for k, v in text_or_dict.items()])
        else:
            text = f"Text: {text_or_dict}"

        results, too_many_tokens, n_total_tokens = self._generate_completion(text, system_message, max_normal_len)
        # Now we remove the "Title:" or "Summary:" at the beginning
        results = ':'.join(results.split(':')[1:]).strip().strip('"')
        return results, too_many_tokens, n_total_tokens


class TranslationModels:
    def __init__(self):
        self.models = None

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
            self.models['en-fr']['model'] = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
            self.models['en-fr']['segmenter'] = pysbd.Segmenter(language='en', clean=False)
            print('Loading FR-EN')
            self.models['fr-en'] = dict()
            self.models['fr-en']['tokenizer'] = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
            self.models['fr-en']['model'] = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
            self.models['fr-en']['segmenter'] = pysbd.Segmenter(language='fr', clean=False)

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
            input_ids = tokenizer.encode(sentence, return_tensors="pt")
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
        if text is None or text == '':
            return None, False
        tokenizer = self.models[how]['tokenizer']
        model = self.models[how]['model']
        segmenter = self.models[how]['segmenter']
        return self._translate(text, tokenizer, model, segmenter)
