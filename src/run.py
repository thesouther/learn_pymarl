import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from os.path import dirname, abspath

from utils.logging import Logger
from utils.timehelper import time_left, time_str
from logging import getLevelName, getLogger


def run(_run, _config, _log):
    pass