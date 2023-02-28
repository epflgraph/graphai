import logging

from core.utils.time.date import now

# Get uvicorn logger so we can write on it
logger = logging.getLogger('uvicorn.error')


def log(msg, seconds=None, total=False, length=64):
    if seconds is None:
        logger.info(f'[{now()}] {msg}')
    else:
        padding_length = length - len(msg)
        if padding_length > 0:
            padding = '.' * padding_length
        else:
            padding = ''

        if total:
            time_msg = f'Elapsed total time: {seconds}s.'
        else:
            time_msg = f'Elapsed time: {seconds}s.'

        logger.info(f'[{now()}] {msg}{padding} {time_msg}')
