import copy
import torch as th
from torch.optim import RMSprop
from components.episode_buffer import EpisodeBatch
from modules.critics.coma import COMACritic


class COMALearner:
    def __init__(self) -> None:
        pass