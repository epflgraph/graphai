from graphai_client.client import login
from graphai_client.client import translate_text, get_video_token, extract_slides
from argparse import ArgumentParser
import time
import multiprocessing
import random


text_list = """This session will be all about an extended example program dealing with discrete event simulation of 
digital circuits. This example will show interesting ways how one can combine assignments and higher order 
functions. Our task will be to construct a digital circuit simulator. There are a number of things that we need 
to do. We need to define a way to specify a digital circuit. We need to define a way to how to run simulations 
and then we have to put everything together so that the circuits can actually drive the simulator. This is also a 
great example that shows how to build programs that do discrete event simulation. So we start with a small 
description language for digital circuits. A digital circuit is composed of wires and the functional components. 
Wires transport signals and components transform signals. We represent signals using Booleans True and False. So 
it's a digital circuit simulator and analog one. A signal is either true or false and not but not something in 
between. Their base components are also called gates. They are the inverter whose output is the inverse of its 
input. The AND gate whose output is the conjunction, logical AND of its inputs and the OR gate whose output is 
the disjunction, logical OR of its inputs. Once we have these three, we can construct the other components by 
combining the base components.""".split('. ')


video_url_list = [
    "https://api.cast.switch.ch/p/113/sp/11300/playManifest/entryId/0_i8zqj20g/format/"
    "download/protocol/https/flavorParamIds/0",
    "https://api.cast.switch.ch/p/113/sp/11300/playManifest/entryId/0_191runee/format/"
    "download/protocol/https/flavorParamIds/0",
    "https://api.cast.switch.ch/p/113/sp/11300/serveFlavor/entryId/0_bpewzi0w/v/2/ev/7/"
    "flavorId/0_7doco2q2/fileName/Heure_de_Contact_6_:_26_10_2021_(Source).webm/forceproxy/true/name/a.webm",
    "https://api.cast.switch.ch/p/113/sp/11300/playManifest/entryId/0_00gdquzv/format/"
    "download/protocol/https/flavorParamIds/0"
]

TIMEOUTS = {
    'translate': 20,
    'slides': 600
}


def stress_test_worker(login_config, what='translate'):
    assert what in ['translate', 'slides']
    if what == 'translate':
        index = random.randint(0, len(text_list) - 1)
        results = translate_text(text_list[index], 'en', 'fr', login_config, force=True)
        if results is not None:
            return True
    else:
        index = random.randint(0, len(video_url_list) - 1)
        results_video = get_video_token(video_url_list[index], login_config, force=True)
        results_slides = extract_slides(results_video, login_config, force=True)
        if results_slides is not None:
            return True
    return False


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--requests', type=int, default=10)
    parser.add_argument('--timeout_translate', type=int, default=None)
    parser.add_argument('--timeout_slides', type=int, default=None)
    args = parser.parse_args()

    config_file = args.config
    n_requests = args.requests
    timeout_values = TIMEOUTS.copy()
    if args.timeout_translate and args.timeout_translate > 0:
        timeout_values['translate'] = args.timeout_translate
    if args.timeout_slides and args.timeout_slides > 0:
        timeout_values['slides'] = args.timeout_slides

    login_config = login(config_file)

    print('STARTING TASKS...')
    start_time = time.time()
    with multiprocessing.Pool(processes=min([n_requests, 10])) as pool:
        what = ['slides' if i % 5 == 4 else 'translate' for i in range(n_requests)]
        tasks = {
            i: {
                'task': pool.apply_async(stress_test_worker, (login_config, what[i])),
                'timeout': timeout_values[what[i]]
            } for i in range(n_requests)
        }
        result = all([tasks[i]['task'].get(timeout=tasks[i]['timeout']) for i in tasks])
        if result:
            print('All tasks succeeded')
        else:
            print('Some tasks failed')

    finish_time = time.time()
    print(f'TIME ELAPSED: {finish_time - start_time}')


if __name__ == '__main__':
    main()
