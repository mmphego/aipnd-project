import signal
import sys

import json

from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import requests


DELAY = INTERVAL = 4 * 60  # interval time in seconds
MIN_DELAY = MIN_INTERVAL = 2 * 60
KEEPALIVE_URL = "https://nebula.udacity.com/api/v1/remote/keep-alive"
TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token"
TOKEN_HEADERS = {"Metadata-Flavor": "Google"}


def _request_handler(headers):
    def _handler(signum, frame):
        requests.request("POST", KEEPALIVE_URL, headers=headers)

    return _handler


@contextmanager
def active_session(delay=DELAY, interval=INTERVAL):
    """
    Example:

    from workspace_utils import active session

    with active_session():
        # do long-running work here
    """
    token = requests.request("GET", TOKEN_URL, headers=TOKEN_HEADERS).text
    headers = {"Authorization": "STAR " + token}
    delay = max(delay, MIN_DELAY)
    interval = max(interval, MIN_INTERVAL)
    original_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _request_handler(headers))
        signal.setitimer(signal.ITIMER_REAL, delay, interval)
        yield
    finally:
        signal.signal(signal.SIGALRM, original_handler)
        signal.setitimer(signal.ITIMER_REAL, 0)


def keep_awake(iterable, delay=DELAY, interval=INTERVAL):
    """
    Example:

    from workspace_utils import keep_awake

    for i in keep_awake(range(5)):
        # do iteration with lots of work here
    """
    with active_session(delay, interval):
        yield from iterable


def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.set_xticklabels("")
    ax.set_yticklabels("")

    return ax


class ProgressBar:
    current_step = 0
    max_step = 0
    bar_length = 40

    def __init__(self, max_step):
        self.max_step = max_step

    def show_progress(self):
        # Every time you call this function, the progress bar will be updated by one step
        self.current_step += 1

        # The percentage information
        info_percent = f"{str(int(self.current_step / float(self.max_step) * 100))}%"
        # The progress bar graph
        cnt_current_block = int(
            (self.current_step / float(self.max_step)) * self.bar_length
        )
        info_current_block = ["█"] * cnt_current_block
        info_rest_block = [" "] * (self.bar_length - cnt_current_block)
        # The step information
        info_count = f"{str(self.current_step)}/{str(self.max_step)}"

        sys.stdout.write(
            f"{info_percent}|{''.join(info_current_block)}{''.join(info_rest_block)}"
            f"|{info_count}\r"
        )
        sys.stdout.flush()

    def end(self):
        # When you finish your job, call this function to start with a new line
        sys.stdout.write("\n")
        sys.stdout.flush()

# Label mapping
def get_names(file_name: str) -> dict:
    """Read json file to dict."""
    with open(file_name, "r") as f:
        names = json.load(f)
    return names
