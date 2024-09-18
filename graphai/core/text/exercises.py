import requests

from graphai.core.common.config import config


def generate_lecture_exercise(lecture_id, description, include_solution):
    """
    Makes request to Chatbot API to generate an exercise for a given lecture.
    """

    try:
        url = f"{config['chatbot']['host']}:{config['chatbot']['port']}/generate_lecture_exercise"
    except Exception:
        print("Warning: chatbot configuration not available in file config.ini, defaulting to localhost:5100. Add the variables to the file to suppress this warning.")
        url = 'localhost:5100'

    try:
        response = requests.post(url, params={'lecture_id': lecture_id, 'description': description, 'include_solution': include_solution}).json()
    except Exception as e:
        msg = f"Request to chatbot API failed. Make sure it is available at {url}."
        print(msg, e)
        return {'error': msg}

    return response
