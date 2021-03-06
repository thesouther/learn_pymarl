import datetime
import os
import os.path as osp
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from os.path import dirname, abspath
import runners

from utils.logging import Logger
from utils.timehelper import time_left, time_str

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRy
from components.transforms import OneHot
from components.episode_buffer import ReplayBuffer


def run(_run, _config, _log):
    # print(_config)
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters: ")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure trensoboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = osp.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = osp.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")


def run_sequential(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    # {'obs_shape': 80, 'state_shape': 65, 'n_actions': 11, 'n_agents': 5, 'episode_limit': 120}
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {
            "vshape": env_info["state_shape"]
        },
        "obs": {
            "vshape": env_info["obs_shape"],
            "group": "agents"
        },
        "actions": {
            "vshape": (1, ),
            "group": "agents",
            "dtype": th.long
        },
        "avail_actions": {
            "vshape": (env_info["n_actions"], ),
            "group": "agents",
            "dtype": th.int
        },
        "reward": {
            "vshape": (1, )
        },
        "terminated": {
            "vshape": (1, ),
            "dtype": th.uint8
        },
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onthot", [OneHot(out_dim=args.n_actions)])}

    buffer = ReplayBuffer(scheme,
                          groups,
                          args.buffer_size,
                          env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRy[args.mac](buffer.scheme, groups, args)


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] //
                                   config["batch_size_run"]) * config["batch_size_run"]

    return config
