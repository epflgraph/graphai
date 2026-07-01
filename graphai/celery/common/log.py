from graphai.core.common.logging import get_logger

logger = get_logger('graphai.celery')


def log(msg, seconds=None, total=False, length=64):
    """Lightweight helper for startup / lifecycle log lines.

    Uses the shared structlog logger so the output follows the same colourful,
    emoji-friendly format as the rest of the service.
    """
    if seconds is None:
        logger.info(msg)
    else:
        padding_length = length - len(msg)
        padding = '.' * padding_length if padding_length > 0 else ''
        time_msg = f'Elapsed total time: {seconds}s.' if total else f'Elapsed time: {seconds}s.'
        logger.info(f'{msg}{padding} {time_msg}')
