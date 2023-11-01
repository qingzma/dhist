import logging
import sys
import time

logging.basicConfig(
    level=logging.DEBUG,
    # [%(threadName)-12.12s]
    format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler(
            "logs/{}_{}.log".format('log', time.strftime("%Y%m%d-%H%M%S"))),
        logging.StreamHandler()
    ])
logger = logging.getLogger(__name__)
# disable matplotlib logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger('numexpr').setLevel(logging.WARNING)

# stream_handler = logging.StreamHandler(sys.stdout)
# logger.addHandler(stream_handler)
