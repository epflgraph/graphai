LECTURE_SUMMARY_MESSAGE = \
      'You will be given a set of slides, extracted from a course lecture. For each slide, ' \
      'you will be given the slide number and a list of concepts mentioned in that slide, ' \
      'in the following format:\n\n' \
      'Slide [NUMBER]: [CONCEPT 1]; [CONCEPT 2]; [CONCEPT 3]; ...\n\n' \
      'Your job is to summarize the contents of the lecture based on the concepts ' \
      'mentioned in the slides. You will provide three summaries:\n' \
      '* [SUMMARY_LONG]: a long summary of the lecture with at most %d words\n' \
      '* [SUMMARY_SHORT]: a short summary of the lecture with at most %d words\n' \
      '* [TITLE]: a title for the lecture in less than %d words.\n\n' \
      'To do so, follow these steps:\n\n' \
      '1. Based on the concepts you\'ve been given for the slides, detect the subject matter: [SUBJECT MATTER]. ' \
      'This should be a Wikipedia page. Choose the most specific Wikipedia page possible for this purpose.\n\n' \
      '2. For each slide, filter out the concepts that do not belong in [SUBJECT MATTER].\n\n' \
      '3. Using the filtered slides, generate [SUMMARY_LONG], [SUMMARY_SHORT], and [TITLE]. ' \
      'Give your answer in the following JSON format:\n' \
      '{\n'\
      '    "subject": [SUBJECT MATTER],\n' \
      '    "summary_long": [SUMMARY_LONG],\n' \
      '    "summary_short": [SUMMARY_SHORT],\n' \
      '    "title": [TITLE]\n' \
      '}\n\n' \
      '4. Make sure that [SUMMARY_LONG] is under %d words, [SUMMARY_SHORT] is under %d words, ' \
      'and [TITLE] is under %d words. If any of these does not conform to the constraint, regenerate it ' \
      'and try to respect the constraint this time. Do not output results that do not respect the constraints; only ' \
      'output them once they are correct.\n\n' \
      'Only output the JSON result, nothing else. No commentary, no explanations, nothing like ' \
      '"[auto generated] = ...", only the JSON. Only something like {...}. This is a STRICT condition; ' \
      'do not ignore it under any circumstances.'

GENERIC_SUMMARY_MESSAGE = \
      'You will be given some text.\n\n' \
      'Your job is to summarize it. You will provide three summaries:\n' \
      '* [SUMMARY_LONG]: a long summary of the text with at most %d words\n' \
      '* [SUMMARY_SHORT]: a short summary of the text with at most %d words\n' \
      '* [TITLE]: a title for the text in less than %d words.\n\n' \
      'To do so, follow these steps:\n\n' \
      '1. Detect the subject matter of the text: [SUBJECT MATTER]. ' \
      'This should be a Wikipedia page. Choose the most specific Wikipedia page possible for this purpose.\n\n' \
      '2. Based on the subject matter and the text itself, generate [SUMMARY_LONG], [SUMMARY_SHORT], and [TITLE]. ' \
      'Give your answer in the following JSON format:\n' \
      '{\n' \
      '    "subject": [SUBJECT MATTER],\n' \
      '    "summary_long": [SUMMARY_LONG],\n' \
      '    "summary_short": [SUMMARY_SHORT],\n' \
      '    "title": [TITLE]\n' \
      '}\n\n' \
      '4. Make sure that [SUMMARY_LONG] is under %d words, [SUMMARY_SHORT] is under %d words, ' \
      'and [TITLE] is under %d words. If any of these does not conform to the constraint, regenerate it ' \
      'and try to respect the constraint this time. Do not output results that do not respect the constraints; only ' \
      'output them once they are correct.\n\n' \
      'Only output the JSON result, nothing else. No commentary, no explanations, nothing like ' \
      '"[auto generated] = ...", only the JSON. Only something like {...}. This is a STRICT condition; ' \
      'do not ignore it under any circumstances.'


def generate_lecture_summary_message(long_len, short_len, title_len):
    return LECTURE_SUMMARY_MESSAGE % (long_len, short_len, title_len, long_len, short_len, title_len)


def generate_generic_summary_message(long_len, short_len, title_len):
    return GENERIC_SUMMARY_MESSAGE % (long_len, short_len, title_len, long_len, short_len, title_len)
