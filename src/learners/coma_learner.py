import copy
import torch as th
from torch.optim import RMSprop
from components.episode_buffer import EpisodeBatch
from modules.critics.coma import COMACritic
from utils.rl_utils import build_td_lambda_targets


class COMALearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic = COMACritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.params = self.agent_params + self.critic_params

        self.agent_optimiser = RMSprop(params=self.agent_params,
                                       lr=args.lr,
                                       alpha=args.optim_alpha,
                                       eps=args.optim_eps)
        self.critic_optimiser = RMSprop(params=self.critic_params,
                                        lr=args.critic_lr,
                                        alpha=args.optim_alpha,
                                        eps=args.optim_eps)

    def train(self):
        pass