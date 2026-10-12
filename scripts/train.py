"""Trains a PPO agent on the slime hunt task, one game process per environment.

uv run scripts/train.py --timesteps 1000000 --envs 8
uv run tensorboard --logdir runs
"""

import argparse
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from towerfall_rl.envs import SIMPLE_REWARDS, Rewards, SlimeHuntEnv

EPISODE_STATS = ("won", "died", "kills", "stomps")
REWARDS = {"shaped": Rewards(), "simple": SIMPLE_REWARDS}


def make_env(enemy_count: int, rewards: str, rank: int) -> Monitor:
    env = SlimeHuntEnv(enemy_count=enemy_count, rewards=REWARDS[rewards], headless_core=rank)
    return Monitor(env, info_keywords=EPISODE_STATS)


class EpisodeStatsCallback(BaseCallback):
    """Logs win rate, death rate and kills over the last 100 episodes next to the reward."""

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        episodes = self.model.ep_info_buffer
        if episodes:
            for key in EPISODE_STATS:
                self.logger.record(f"rollout/ep_{key}_mean", np.mean([ep[key] for ep in episodes]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--envs", type=int, default=8, help="number of games run in parallel")
    parser.add_argument("--enemies", type=int, default=3)
    parser.add_argument(
        "--rewards",
        choices=REWARDS,
        default="shaped",
        help="shaped: the Rewards defaults; simple: only kills, death and time",
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda for the neural network")
    parser.add_argument("--name", default=datetime.now().strftime("slime-hunt-%Y%m%d-%H%M%S"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path("runs") / args.name
    env = SubprocVecEnv(
        [partial(make_env, args.enemies, args.rewards, rank) for rank in range(args.envs)]
    )
    try:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=2048 // args.envs,
            batch_size=256,
            ent_coef=0.01,
            device=args.device,
            seed=args.seed,
            tensorboard_log=str(run_dir),
            verbose=1,
        )
        callbacks = [
            EpisodeStatsCallback(),
            CheckpointCallback(
                save_freq=max(100_000 // args.envs, 1), save_path=str(run_dir / "checkpoints")
            ),
        ]
        start = time.perf_counter()
        model.learn(total_timesteps=args.timesteps, callback=callbacks, tb_log_name="tb")
        elapsed = time.perf_counter() - start
        model.save(run_dir / "model")
        print(
            f"Saved {run_dir / 'model.zip'} after {elapsed / 60:.1f} min "
            f"({model.num_timesteps / elapsed:.0f} steps/s)"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
