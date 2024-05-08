from pkg_resources import get_distribution, DistributionNotFound
import logging


try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "UNKNOWN"


class DisableLogger:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


log = logging.getLogger("jang")
log.setLevel(logging.DEBUG)
logger_ch = logging.StreamHandler()
logger_ch.setLevel(logging.DEBUG)
logger_ch.setFormatter(logging.Formatter("[%(asctime)s:%(name)s:%(levelname)s] %(message)s"))
log.addHandler(logger_ch)
