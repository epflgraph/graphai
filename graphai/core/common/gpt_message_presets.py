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


ACADEMIC_ENTITY_SUMMARY_MESSAGE = \
      '# Description of input variables related to an academic entity\n\n# [entity]: Type ' \
      'of academic entity.\n# [name]: Name of [entity].\n# [subtype]: Subtype of [' \
      'entity].\n# [possible subtypes]: List of possible values that [subtype] can ' \
      'have.\n# [text]: Non-curated raw text describing [entity], scraped from [' \
      'entity]\'s website.\n# [categories]: List of research categories extracted from ' \
      'scientific publications authored by people related to [entity]. The categories are ' \
      'ranked in descending order of importance.\n# [n words (long)]: Maximum number of ' \
      'words in long description.\n# [n words (short)]: Maximum number of words in short ' \
      'summary.\n[n words (title)]: Maximum number of words in title\n\n' \
      '# Scripted tasks\n\nIF [subtype] IS null\n    [inferred subtype] = ' \
      'Infer the [entity]\'s subtype from [text]. Only use values from [possible ' \
      'subtypes].\n    ASSERT [inferred subtype] IS IN [possible subtypes]\nELSE\n    [' \
      'inferred subtype] = [subtype]\nEND IF\nASSERT [inferred subtype] IS NOT null\n\n[' \
      'top 3] = An estimation of the top 3 categories, generated as a combination, ' \
      'variation, or aggregation of [text] and [categories]. You should prioritise the ' \
      'content of [text], and use [categories] only as additional information.\nASSERT ' \
      'TYPE([top 3]) IS LIST\nASSERT SIZE([top 3]) = 3\nASSERT [top 3] IS LOWER CASE\n\n[' \
      'auto generated] = false\n\n[long description] = Using [text] and [categories], ' \
      'generate a new text that respects the following conditions:\n    - is a ' \
      'description of [entity];\n    - is primarily faithful to the facts stated in [' \
      'text], and uses [categories] only as support information (eg, if [text] is null or ' \
      'useless);\n    - if [text] is null or useless, uses [categories] as the primary ' \
      'source of information (in which case, set [auto generated] to true);\n    - is ' \
      'written in 3rd person;\n    - does not include contact information, such as email ' \
      'addresses, phone numbers, or postal addresses;\n    - is not limited to a list of ' \
      'facts, but has a proper narrative;\n    - refrains from adding any of your own ' \
      'adjectives that imply a value judgement or provide a qualitative evaluation of the ' \
      'entity\'s virtues, achievements, or capabilities - unless directly cited in [' \
      'text]. This includes words or phrases like \'noteworthy\', \'profound\', ' \
      '\'significant\', \'remarkable\', \'seemingly\', etc.\nASSERT LENGTH OF [long ' \
      'description] <= [n words (long)] WORDS\n\nIF LENGTH OF [long description] > [n ' \
      'words (long)] WORDS\n    REGENERATE DIFFERENT [long description]\n    ASSERT ' \
      'LENGTH OF [long description] <= [n words (long)] WORDS\nEND IF\n\n[short ' \
      'description] =  one sentence derived from [long description]\nASSERT [short ' \
      'description] IS DERIVED FROM [long description] AND RESPECTS SAME CONDITIONS AS [' \
      'long description]\nASSERT LENGTH OF [short description] <= [n words (long)] ' \
      'WORDS\n\nIF LENGTH OF [short description] > [n words (short)] WORDS\n    ' \
      'REGENERATE DIFFERENT [short description]\n' \
      'IF LENGTH OF [title] > [n words (title)] WORDS\n    ' \
      'REGENERATE DIFFERENT [title]\n' \
      'ASSERT LENGTH OF [short description] ' \
      '<= [n words (short)] WORDS\nEND IF\n\nYour task, Mr. ChatGPT, is to execute this ' \
      'script based on the input variables I will provide next.\n\nOutput the following ' \
      'JSON format:\n {\n    \"is_auto_generated\": [auto generated],' \
      '\n    \"inferred_subtype\": [inferred subtype],\n    \"top_3_categories\": [top ' \
      '3],\n    \"summary_short\": [short description],\n    \"summary_long\": [' \
      'long description]\n    \"title\": [title]\n}\n\nOnly output the JSON result, nothing else. ' \
      'No commentary, no explanations, nothing like "[auto generated] = ...", only the JSON. Only ' \
      'something like {...}. This is a STRICT condition; do not ignore it under any ' \
      'circumstances.'


SUMMARY_ASSISTANT_MESSAGE = 'Output the following JSON format:\n {\n    \'is_auto_generated\': [auto generated],' \
                            '\n    \'inferred_subtype\': [inferred subtype],\n    \'top_3_categories\': [top 3],' \
                            '\n    \'short_description\': [short description],\n    \'long_description\': [long ' \
                            'description]\n}\n\nOnly output the JSON result, nothing else. No comentary, ' \
                            'no explanations, nothing like "[auto generated] = ...", only the JSON. Only something ' \
                            'like {...}. This is a STRICT condition; do not ignore it under any circumstances. '


def generate_lecture_summary_message(long_len, short_len, title_len):
    return LECTURE_SUMMARY_MESSAGE % (long_len, short_len, title_len, long_len, short_len, title_len)


def generate_generic_summary_message(long_len, short_len, title_len):
    return GENERIC_SUMMARY_MESSAGE % (long_len, short_len, title_len, long_len, short_len, title_len)


def generate_academic_entity_summary_message():
    return ACADEMIC_ENTITY_SUMMARY_MESSAGE, SUMMARY_ASSISTANT_MESSAGE
