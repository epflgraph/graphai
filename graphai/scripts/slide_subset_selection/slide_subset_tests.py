import json

from graphai.core.common.text_utils import find_best_slide_subset, ChatGPTSummarizer, convert_text_or_dict_to_text
from graphai.core.common.caching import SlideDBCachingManager
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
        url = API_BASE + '/text/wikify'
        data = json.dumps({"raw_text": slide_text})
        response = requests.post(url, data).json()
        current_results = sorted([(c['PageTitle'], c['MixedScore']) for c in response], key=lambda x: -x[1])
        results_list.append({c[0] for c in current_results if c[1] > 0.5})
        counter += 1
        if counter % 20 == 0:
            print(response)
    return results_list


def clean_slides_up(slides):
    results = list()
    c = ChatGPTSummarizer()
    for slide_content in slides:
        cleaned, _, _, _ = c.cleanup_text(slide_content)
        if cleaned is not None:
            results.append(cleaned['cleaned'])
        else:
            print('REQUEST TIMED OUT!')
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
                     "    \"summary_long\": [SUMMARY_LONG],\n" \
                     "    \"summary_short\": [SUMMARY_SHORT],\n" \
                     "    \"title\": [TITLE]\n" \
                     "}\n" \
                     "Only provide the JSON response. No explanations. The respones must conform to the " \
                     "length requiements provided above."
    c = ChatGPTSummarizer()
    results, _, cost = c._generate_completion(slides, system_message)
    return results, cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, help='Token of the video to do the test on.', required=True)
    args = parser.parse_args()

    # Retrieve the slides of the video
    slides_list = get_video_slides(args.token)
    print(len(slides_list))
    slides_list = [(v['ocr_google_1_results'], v['slide_number'], v['timestamp']) for v in slides_list]
    slides_list = sorted(slides_list, key=lambda x: x[1])
    slides_text_list = [s[0] for s in slides_list]

    # Detect concepts on every slide's raw OCR
    slides_concepts_raw = detect_concepts(slides_text_list)

    # Now we compute the results for multiple coverage values
    priorities = True
    for coverage in [0.7, 0.8]:
        print(f"Computing results for coverage={coverage}, priorities={priorities}")
        # Choose the optimal subset
        best_subset, best_indices = find_best_slide_subset(slides_concepts_raw, coverage, priorities, min_freq=2)
        best_indices_sorted = sorted(best_indices)
        print(len(best_indices_sorted))
        print(best_indices_sorted)
        slides_for_summarization = [slides_text_list[i] for i in best_indices_sorted]
        slides_for_summarization = clean_slides_up(slides_for_summarization)
        slides_for_summarization = detect_concepts(slides_for_summarization)
        slides_for_summarization = {f'Slide {best_indices_sorted[i]+1}': slides_for_summarization[i]
                                    for i in range(len(slides_for_summarization))}
        results, cost = make_slide_summarization_request(convert_text_or_dict_to_text(slides_for_summarization))
        print('Results:')
        print(results)
        print('Cost:')
        print(cost)


if __name__ == '__main__':
    main()
