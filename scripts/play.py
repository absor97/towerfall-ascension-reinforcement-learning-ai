"""Runs a trained agent on the slime hunt task, to watch it or to measure its win rate.

uv run scripts/play.py runs/<name>/model.zip                      # watch at normal speed
uv run scripts/play.py runs/<name>/model.zip --fps 0 --episodes 300   # measure
"""

import argparse

import numpy as np
from stable_baselines3 import PPO

from towerfall_rl.envs import SlimeHuntEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="path to a saved model .zip")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--enemies", type=int, default=3)
    parser.add_argument("--fps", type=int, default=60, help="game speed cap, 0 = uncapped")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="always take the most likely action instead of sampling like in training",
    )
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    env = SlimeHuntEnv(enemy_count=args.enemies, fps=args.fps)
    results = []
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=episode)
            done = False
            while not done:
                action, _ = model.predict(observation, deterministic=args.deterministic)
                observation, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            results.append(info)
            outcome = "won" if info["won"] else "died" if info["died"] else "timeout"
            print(f"episode {episode + 1}: {outcome}, {info['kills']} kills")
    finally:
        env.close()

    print(
        f"\n{len(results)} episodes: won {np.mean([r['won'] for r in results]):.0%}, "
        f"died {np.mean([r['died'] for r in results]):.0%}, "
        f"mean kills {np.mean([r['kills'] for r in results]):.2f} "
        f"({np.mean([r['stomps'] for r in results]):.2f} by stomping)"
    )


if __name__ == "__main__":
    main()
