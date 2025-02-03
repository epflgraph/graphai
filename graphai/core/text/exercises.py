import requests

from graphai.core.common.config import config


def generate_exercise(data):
    """
    Makes request to Chatbot API to generate an exercise.
    """

    try:
        url = f"http://{config['chatbot']['host']}:{config['chatbot']['port']}"
    except Exception:
        print("Warning: chatbot configuration not available in file config.ini, defaulting to localhost:5100. Add the variables to the file to suppress this warning.")
        url = 'http://localhost:5100'

    url += '/generate_exercise'

    payload = {
        'description': data.description,
        'bloom_level': data.bloom_level,
        'include_solution': data.include_solution,
        'output_format': data.output_format,
        'llm_model': data.llm_model,
        'openai_api_key': data.openai_api_key,
    }

    try:
        payload['text'] = data.text
    except Exception:
        payload['lecture_id'] = data.lecture_id

    try:
        response = requests.post(url, json=payload).json()
    except Exception as e:
        msg = f"Request to chatbot API failed. Make sure it is available at {url}."
        print(msg, e)
        return {'error': msg}

    return response
