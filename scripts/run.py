from rlpyt.experiments.configs.atari.dqn.atari_dqn import configs
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.atari.atari_env import AtariTrajInfo
from rlpyt.utils.logging.context import logger_context

import wandb
import torch
import torch.multiprocessing as mp
import numpy as np
import os
import copy
from itertools import product as cartesian_product

from src.models import SPRCatDqnModel
from src.rlpyt_utils import OneToOneSerialEvalCollector, SerialSampler, MinibatchRlEvalWandb
from src.algos import SPRCategoricalDQN
from src.agent import SPRAgent
from src.planner_helper import PlanningSPRAgent
from src.rlpyt_atari_env import AtariEnv
from src.utils import set_config


def train_single_run(game, seed, cuda_idx, args, group_name):
    """单个 (game, seed) 组合的训练，跑在独立进程中。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    run_name = f"{game}_seed{seed}"
    wandb_kwargs = dict(
        config=vars(args),
        tags=[args.tag] if args.tag else None,
        dir=args.wandb_dir if args.wandb_dir else None,
        group=group_name,
        name=run_name,
        reinit=True,
    )
    if args.public:
        wandb.init(anonymous="allow", **wandb_kwargs)
    else:
        wandb.init(project=args.project, entity=args.entity, **wandb_kwargs)
    wandb.config.update({"seed": seed, "game": game}, allow_val_change=True)

    env = AtariEnv
    args_copy = copy.deepcopy(args)
    args_copy.seed = seed
    config = set_config(args_copy, game)

    sampler = SerialSampler(
        EnvCls=env,
        TrajInfoCls=AtariTrajInfo,
        env_kwargs=config["env"],
        eval_env_kwargs=config["eval_env"],
        batch_T=config["sampler"]["batch_T"],
        batch_B=config["sampler"]["batch_B"],
        max_decorrelation_steps=0,
        eval_CollectorCls=OneToOneSerialEvalCollector,
        eval_n_envs=config["sampler"]["eval_n_envs"],
        eval_max_steps=config["sampler"]["eval_max_steps"],
        eval_max_trajectories=config["sampler"]["eval_max_trajectories"],
    )

    args_copy.discount = config["algo"]["discount"]

    algo = SPRCategoricalDQN(
        optim_kwargs=config["optim"], jumps=args_copy.jumps, **config["algo"]
    )
    # agent = SPRAgent(
    #     ModelCls=SPRCatDqnModel, model_kwargs=config["model"], **config["agent"]
    # )

    agent = PlanningSPRAgent(
        ModelCls=SPRCatDqnModel,
        model_kwargs=config["model"],
        planning_horizon=3,         # rollout 几步，别超过 jumps
        planning_top_k=5,           # Q 值筛出 5 个候选再 rollout
        planning_warmup_itrs=5000,  # 前 5000 itr 不用 planning
        **config["agent"],
    )

    wandb.config.update(config, allow_val_change=True)

    runner = MinibatchRlEvalWandb(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=args_copy.n_steps,
        affinity=dict(cuda_idx=cuda_idx),
        log_interval_steps=args_copy.n_steps // args_copy.num_logs,
        seed=seed,
        final_eval_only=args_copy.final_eval_only,
    )

    config_log = dict(game=game)
    name = f"dqn_{game}"
    log_dir = os.path.join("logs", game, f"seed_{seed}")

    with logger_context(log_dir, seed, name, config_log, snapshot_mode="last"):
        runner.train()

    wandb.finish()


def worker_fn(rank, tasks, cuda_idx, args, group_name):
    """mp.spawn 入口：rank 索引到 tasks 列表中的 (game, seed)。"""
    game, seed = tasks[rank]
    train_single_run(game, seed, cuda_idx, args, group_name)


def build_and_train_parallel(games, cuda_idx, args):
    # 笛卡尔积: games × seeds
    tasks = list(cartesian_product(games, args.seeds))
    n_tasks = len(tasks)

    group_name = "multi_game_multi_seed"
    if args.tag:
        group_name += f"_{args.tag}"

    print(f"=== 启动 {n_tasks} 个并行训练进程 (GPU {cuda_idx}) ===")
    for i, (g, s) in enumerate(tasks):
        print(f"  [{i}] game={g}, seed={s}")

    if n_tasks == 1:
        game, seed = tasks[0]
        train_single_run(game, seed, cuda_idx, args, group_name)
    else:
        mp.spawn(
            worker_fn,
            args=(tasks, cuda_idx, args, group_name),
            nprocs=n_tasks,
            join=True,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # ---------- 核心参数：多游戏 + 多 seed ----------
    parser.add_argument(
        "--games", type=str, nargs="+", default=["private_eye"],
        # , "breakout", "boxing", , "asterix", "alien"
        help="List of Atari games, e.g. --games ms_pacman pong breakout",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
        help="List of random seeds, e.g. --seeds 0 1 2 3",
    )
    # ------------------------------------------------
    parser.add_argument("--grayscale", type=int, default=1)
    parser.add_argument("--framestack", type=int, default=4)
    parser.add_argument("--imagesize", type=int, default=84)
    parser.add_argument("--n-steps", type=int, default=100000)
    parser.add_argument("--dqn-hidden-size", type=int, default=256)
    parser.add_argument("--target-update-interval", type=int, default=1)
    parser.add_argument("--target-update-tau", type=float, default=1.0)
    parser.add_argument("--momentum-tau", type=float, default=0.01)
    parser.add_argument("--batch-b", type=int, default=1)
    parser.add_argument("--batch-t", type=int, default=1)
    parser.add_argument("--beluga", action="store_true")
    parser.add_argument("--jumps", type=int, default=5)
    parser.add_argument("--num-logs", type=int, default=10)
    parser.add_argument("--renormalize", type=int, default=1)
    parser.add_argument("--dueling", type=int, default=1)
    parser.add_argument("--replay-ratio", type=int, default=64)
    parser.add_argument("--dynamics-blocks", type=int, default=0)
    parser.add_argument("--residual-tm", type=int, default=0.0)
    parser.add_argument("--n-step", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tag", type=str, default="", help="Tag for wandb run.")
    parser.add_argument("--wandb-dir", type=str, default="", help="Directory for wandb files.")
    parser.add_argument("--norm-type", type=str, default="bn", choices=["bn", "ln", "in", "none"])
    parser.add_argument("--aug-prob", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--spr", type=int, default=1)
    parser.add_argument("--distributional", type=int, default=1)
    parser.add_argument("--delta-clip", type=float, default=1.0)
    parser.add_argument("--prioritized-replay", type=int, default=1)
    parser.add_argument("--momentum-encoder", type=int, default=1)
    parser.add_argument("--shared-encoder", type=int, default=0)
    parser.add_argument("--local-spr", type=int, default=0)
    parser.add_argument("--global-spr", type=int, default=1)
    parser.add_argument("--noisy-nets", type=int, default=1)
    parser.add_argument("--noisy-nets-std", type=float, default=0.5)
    parser.add_argument("--classifier", type=str, default="q_l1", choices=["mlp", "bilinear", "q_l1", "q_l2", "none"])
    parser.add_argument("--final-classifier", type=str, default="linear", choices=["mlp", "linear", "none"])
    parser.add_argument("--augmentation", type=str, default=["shift", "intensity"], nargs="+",
                        choices=["none", "rrc", "affine", "crop", "blur", "shift", "intensity"])
    parser.add_argument("--q-l1-type", type=str, default=["value", "advantage"], nargs="+",
                        choices=["noisy", "value", "advantage", "relu"])
    parser.add_argument("--target-augmentation", type=int, default=1)
    parser.add_argument("--eval-augmentation", type=int, default=0)
    parser.add_argument("--reward-loss-weight", type=float, default=0.0)
    parser.add_argument("--model-rl-weight", type=float, default=0.0)
    parser.add_argument("--model-spr-weight", type=float, default=5.0)
    parser.add_argument("--t0-spr-loss-weight", type=float, default=0.0)
    parser.add_argument("--eps-steps", type=int, default=2001)
    parser.add_argument("--min-steps-learn", type=int, default=2000)
    parser.add_argument("--eps-init", type=float, default=1.0)
    parser.add_argument("--eps-final", type=float, default=0.0)
    parser.add_argument("--final-eval-only", type=int, default=1)
    parser.add_argument("--time-offset", type=int, default=0)
    parser.add_argument("--project", type=str, default="mpr")
    parser.add_argument("--entity", type=str, default="abs-world-models")
    parser.add_argument("--cuda_idx", help="gpu to use", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()

    args.seed = args.seeds[0]
    args.game = args.games[0]

    mp.set_start_method("spawn", force=True)

    build_and_train_parallel(
        games=args.games,
        cuda_idx=args.cuda_idx,
        args=args,
    )