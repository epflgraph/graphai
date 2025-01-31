import io
import time
from multiprocessing import Lock

import pdf2image
import pytesseract
from google.cloud import vision

from graphai.core.common.common_utils import file_exists


class GoogleOCRModel:
    def __init__(self, api_key):
        self.api_key = api_key

        if self.api_key is None:
            print(
                "No Google API key was provided. "
                "Google API endpoints cannot be used as there is no default API key."
            )
        self.model = None
        self.load_lock = Lock()

    def establish_connection(self):
        """
        Lazily connects to the Google API
        Returns:
            True if a connection already exists or if a new connection is successfully established, False otherwise
        """
        with self.load_lock:
            if self.model is None:
                if self.api_key is not None:
                    print('Establishing Google API connection...')
                    try:
                        self.model = vision.ImageAnnotatorClient(client_options={"api_key": self.api_key})
                        return True
                    except Exception:
                        print('Failed to connect to Google API!')
                        return False
                else:
                    print('No API key provided!')
                    return False
            else:
                return True

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


def get_ocr_colnames(method):
    if method == 'tesseract':
        return ['ocr_tesseract_results']
    else:
        return ['ocr_google_1_results', 'ocr_google_2_results']


def perform_tesseract_ocr_on_pdf(pdf_path, language=None):
    """
    Performs OCR using tesseract on a pdf file
    Args:
        pdf_path: Path to the PDF file
        language: Language of the PDF file

    Returns:
        String containing the entire PDF file's extracted contents
    """
    if language is None:
        language = 'enfr'
    if not file_exists(pdf_path):
        print(f'Error: File {pdf_path} does not exist')
        return None
    pdf_imageset = pdf2image.convert_from_path(pdf_path)
    return '\n'.join(
        pytesseract.image_to_string(img, lang={'en': 'eng', 'fr': 'fra', 'enfr': 'eng+fra'}[language])
        for img in pdf_imageset
    )
