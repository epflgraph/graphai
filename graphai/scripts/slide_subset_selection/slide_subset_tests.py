import json

from graphai.core.common.text_utils import find_best_slide_subset, ChatGPTSummarizer, convert_text_or_dict_to_text
from graphai.core.common.caching import SlideDBCachingManager
from graphai.core.common.common_utils import make_sure_path_exists
import requests
import argparse


API_BASE = 'http://0.0.0.0:28801'


def get_video_slides(token):
    db_manager = SlideDBCachingManager()
    slide_list = db_manager.get_details_using_origin(token, ['timestamp', 'slide_number', 'ocr_google_1_results'])
    return slide_list


def detect_concepts(list_of_slides_text):
    results_list = list()
    counter = 0
    for slide_text in list_of_slides_text:
        if slide_text is None:
            results_list.append(set())
        url = API_BASE + '/text/wikify'
        data = json.dumps({"raw_text": slide_text})
        response = requests.post(url, data).json()
        try:
            current_results = sorted([(c['PageTitle'], c['MixedScore'], c['LevenshteinScore']) for c in response],
                                     key=lambda x: -x[1])
            results_list.append({c[0] for c in current_results if c[1] > 0.7 and c[2] > 0.1})
        except Exception as e:
            print('An error occurred, here is the response')
            print(response)
            results_list.append(set())
        counter += 1
        if counter % 10 == 0:
            print(counter)
    return results_list


def clean_slides_up(slides, retries=3):
    results = list()
    c = ChatGPTSummarizer()
    for slide_content in slides:
        current_results = ''
        for retry in range(retries):
            cleaned, _, _, _ = c.cleanup_text(slide_content)
            if cleaned is not None:
                current_results = cleaned['cleaned']
                break
        if current_results is None:
            print(f'REQUEST TIMED OUT AFTER {retries} RETRIES!')
        results.append(current_results)
    return results


def make_slide_summarization_request(slides):
    system_message = "You will be given a set of slides, extracted from a course lecture. " \
                     "For each slide, you will be given the slide number and a list of concepts " \
                     "mentioned in that slide, in the following format:\n" \
                     "Slide [NUMBER]: [CONCEPT 1]; [CONCEPT 2]; [CONCEPT 3]; ...\n" \
                     "Your job is to summarize the contents of the lecture based on the concepts mentioned " \
                     "in the slides. You will provide three summaries: [SUMMARY_LONG], which is longer and " \
                     "has at most 200 words; [SUMMARY_SHORT], which is shorter and has at most 50 words; " \
                     "and [TITLE], which is a title for the lecture in less than 10 words. To do so, follow " \
                     "these steps:\n" \
                     "1. Based on the concepts you've been given for the slides, " \
                     "detect the subject matter: [SUBJECT MATTER]. This should be a Wikipedia page. Choose the " \
                     "most specific Wikipedia page possible for this purpose.\n" \
                     "2. For each slide, filter out the concepts that do not belong in [SUBJECT MATTER].\n" \
                     "3. Using the filtered slides, generate [SUMMARY_LONG], [SUMMARY_SHORT], and [TITLE]." \
                     "Give your answer in the following JSON format:\n" \
                     "{\n" \
                     "    \"subject\": [SUBJECT MATTER],\n" \
                     "    \"summary_long\": [SUMMARY_LONG],\n" \
                     "    \"summary_short\": [SUMMARY_SHORT],\n" \
                     "    \"title\": [TITLE]\n" \
                     "}\n" \
                     "Only provide the JSON response. No explanations. The respones must conform to the " \
                     "length requiements provided above."
    c = ChatGPTSummarizer()
    results, _, cost = c._generate_completion(slides, system_message)
    return results, cost


def generate_results_path(token):
    path = './results/' + token + '_results.json'
    make_sure_path_exists(path, file_at_the_end=True)
    return path


def save_results(token, slides_list, concepts_raw, content_clean, concepts_clean):
    results_path = generate_results_path(token)
    full_dict = {
        i: {
            'slide_raw': slides_list[i][0],
            'slide_number': slides_list[i][1],
            'timestamp': slides_list[i][2],
            'concepts_raw': list(concepts_raw[i]),
            'slide_clean': content_clean[i],
            'concepts_clean': list(concepts_clean[i])
        }
        for i in range(len(slides_list))
    }
    with open(results_path, 'w') as f:
        json.dump(full_dict, f)


def load_results(token):
    results_path = generate_results_path(token)
    try:
        with open(results_path, 'r') as f:
            res = json.load(f)
            return res
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, help='Token of the video to do the test on', required=True)
    parser.add_argument('--coverage', type=float, help='Coverage of the concepts', default=0.9)
    parser.add_argument('--min_freq', type=int,
                        help='Minimum frequency that a concept must have to not get filtered out', default=2)
    args = parser.parse_args()
    token = args.token
    coverage = args.coverage
    min_freq = args.min_freq

    print('Looking for cached results...')
    existing_results = load_results(token)
    if existing_results is None:
        print('Results not found, starting retrieval...')
        # Retrieve the slides of the video
        slides_list = get_video_slides(token)
        print(len(slides_list))
        slides_list = [(v['ocr_google_1_results'], v['slide_number'], v['timestamp']) for v in slides_list]
        slides_list = sorted(slides_list, key=lambda x: x[1])
        slides_text_list = [s[0] for s in slides_list]

        # Detect concepts on every slide's raw OCR
        slides_concepts_raw = detect_concepts(slides_text_list)
        slides_cleaned_up = clean_slides_up(slides_text_list)
        slides_concepts_cleaned = detect_concepts(slides_cleaned_up)

        # Store the results
        print('Saving...')
        save_results(token, slides_list, slides_concepts_raw, slides_cleaned_up, slides_concepts_cleaned)
        print('Processing...')
    else:
        print('Results loaded, processing...')
        keys = list(existing_results.keys())
        slides_concepts_raw = [existing_results[k]['concepts_raw'] for k in keys]
        slides_concepts_cleaned = [existing_results[k]['concepts_clean'] for k in keys]

    priorities = True
    print(f"Computing results for coverage={coverage}, priorities={priorities}")
    # Choose the optimal subset
    best_subset, best_indices = find_best_slide_subset(slides_concepts_raw, coverage, priorities, min_freq=min_freq)
    best_indices_sorted = sorted(best_indices)
    print(len(best_indices_sorted))
    print(best_indices_sorted)
    # Summarize using raw concepts
    slides_for_summarization_no_cleanup = {
        f'Slide {i+1}': slides_concepts_raw[i]
        for i in best_indices_sorted
    }
    print(slides_for_summarization_no_cleanup)
    nocleanup_results, nocleanup_cost = \
        make_slide_summarization_request(convert_text_or_dict_to_text(slides_for_summarization_no_cleanup))
    print('********WITHOUT CLEANUP********')
    print('Results:')
    print(nocleanup_results)
    print('Cost:')
    print(nocleanup_cost)
    # Summarize using clean concepts
    slides_for_summarization_with_cleanup = {
        f'Slide {i+1}': slides_concepts_cleaned[i]
        for i in best_indices_sorted
    }
    slides_for_summarization_with_cleanup = {k: v for k, v in slides_for_summarization_with_cleanup.items()
                                             if len(v) > 0}
    print(slides_for_summarization_with_cleanup)
    results, cost = make_slide_summarization_request(
        convert_text_or_dict_to_text(slides_for_summarization_with_cleanup)
    )
    print('********WITH CLEANUP********')
    print('Results:')
    print(results)
    print('Cost:')
    print(cost)


if __name__ == '__main__':
    main()
