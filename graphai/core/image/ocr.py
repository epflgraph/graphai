import io
import json
import time
from multiprocessing import Lock

from abc import ABC, abstractmethod

import pdf2image
import pytesseract
from google.cloud import vision
from openai import OpenAI

from graphai.core.common.common_utils import file_exists

import base64


OPENAI_OCR_PROMPT = """
    You are to extract the text contents of the following image. Formulae (if any) are to be extracted as valid LaTeX.
    Output your response as a valid JSON with two fields:
    1. "text": Containing ONLY the extracted text and formulae (if applicable). Do not include ANY extra explanations.
    2. "keywords": A list of at least 1 and at most 10 keywords that describe the contents of the image.
    If any LaTeX is present in the "text" field, ensure that it is valid and that it'll compile using XeLaTeX.
"""


class ImgToBase64Converter:

    def __init__(self, image_path):
        with open(image_path, "rb") as image_file:
            self.base64 = base64.b64encode(image_file.read()).decode('utf-8')

    def get_base64(self):
        return self.base64


class AbstractOCRModel(ABC):
    def __init__(self, api_key, model_class, model_name):
        self.api_key = api_key
        self.model_type = model_class
        self.model_name = model_name
        self.model_params = None

        if self.api_key is None:
            print(
                f"No {model_name} API key was provided. "
                f"{model_name} API endpoints cannot be used as there is no default API key."
            )
        self.model = None
        self.load_lock = Lock()

    def establish_connection(self):
        """
        Lazily connects to the OCR API
        Returns:
            True if a connection already exists or if a new connection is successfully established, False otherwise
        """
        with self.load_lock:
            if self.model is None:
                if self.model_params is not None and isinstance(self.model_params, dict):
                    print(f'Establishing {self.model_name} API connection...')
                    try:
                        self.model = self.model_type(**self.model_params)
                        return True
                    except Exception:
                        print(f'Failed to connect to {self.model_name} API!')
                        return False
                else:
                    print('No API key provided!')
                    return False
            else:
                return True

    @abstractmethod
    def perform_ocr(self, input_filename_with_path):
        pass


class GoogleOCRModel(AbstractOCRModel):
    def __init__(self, api_key):
        super().__init__(api_key, vision.ImageAnnotatorClient, 'Google')
        self.model_params = dict(
            client_options={"api_key": self.api_key}
        )

    def perform_ocr(self, input_filename_with_path):
        """
        Performs OCR with two methods (text_detection and document_text_detection)
        Args:
            input_filename_with_path: Full path of the input image file

        Returns:
            Text results of the two OCR methods
        """
        model_loaded = self.establish_connection()
        if not model_loaded:
            return None, None
        # Loading the image
        if not file_exists(input_filename_with_path):
            print(f'Error: File {input_filename_with_path} does not exist')
            return None, None
        with io.open(input_filename_with_path, 'rb') as image_file:
            image_content = image_file.read()
        g_image_obj = vision.Image(content=image_content)
        # Waiting for results (accounting for possible failures)
        results_1 = self.wait_for_ocr_results(image_object=g_image_obj, method='dtd')
        results_2 = self.wait_for_ocr_results(image_object=g_image_obj, method='td')
        return results_1, results_2

    def wait_for_ocr_results(self, image_object, method='dtd', retries=6):
        """
        Makes call to Google OCR API and waits for the results
        Args:
            image_object: Image object for the Google API
            method: Method to use, 'td' for text detection and 'dtd' for document text detection
            retries: Number of retries to perform in case of failure

        Returns:
            OCR results
        """
        assert method in ['dtd', 'td']
        results = None
        for i in range(retries):
            try:
                if method == 'dtd':
                    results = self.model.document_text_detection(image=image_object)
                else:
                    results = self.model.text_detection(image=image_object)
                break
            except Exception:
                print('Failed to call OCR engine. Trying again in 60 seconds ...')
                time.sleep(5)
        if results is not None:
            results = results.full_text_annotation.text
        return results


class OpenAIOCRModel(AbstractOCRModel):
    def __init__(self, api_key):
        super().__init__(api_key, OpenAI, "OpenAI")
        self.model_params = dict(
            api_key=self.api_key
        )

    def perform_ocr(self, input_filename_with_path):
        model_loaded = self.establish_connection()
        if not model_loaded:
            return None
        img_b64_str = ImgToBase64Converter(input_filename_with_path).get_base64()
        img_type = f'image/{input_filename_with_path.split(".")[-1]}'
        try:
            response = self.model.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": OPENAI_OCR_PROMPT
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{img_type};base64,{img_b64_str}"},
                            },
                        ],
                    }
                ],
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            print(e)
            return None


def get_ocr_colnames(method):
    if method == 'tesseract':
        return ['ocr_tesseract_results']
    elif method == 'google':
        return ['ocr_google_1_results', 'ocr_google_2_results']
    else:
        return ['ocr_openai_results']


def perform_tesseract_ocr_on_pdf(pdf_path, language=None, in_pages=True):
    """
    Performs OCR using tesseract on a pdf file
    Args:
        pdf_path: Path to the PDF file
        language: Language of the PDF file
        in_pages: Whether to return the results as a separate pages (in a JSON string) or as a singular string.

    Returns:
        String containing the entire PDF file's extracted contents
    """
    if language is None:
        language = 'enfr'
    if not file_exists(pdf_path):
        print(f'Error: File {pdf_path} does not exist')
        return None
    pdf_imageset = pdf2image.convert_from_path(pdf_path)
    results = [
        pytesseract.image_to_string(
            img,
            lang={'en': 'eng', 'fr': 'fra', 'enfr': 'eng+fra', 'eneq': 'eng+equ', 'freq': 'fra+equ'}[language]
        )
        for img in pdf_imageset
    ]
    if not in_pages:
        return '\n'.join(results)
    else:
        return json.dumps(
            [
                {
                    'page': i + 1,
                    'content': results[i]
                }
                for i in range(len(results))
            ]
        )
