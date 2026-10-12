"""Smoke test and speed benchmark: two random agents in a 1v1 versus match.

uv run scripts/random_agents.py --frames 3000 --fps 0
"""

import argparse
import random
import time

from towerfall_rl.client import AgentConnection, Button, Towerfall, read_updates


def random_buttons(rng: random.Random) -> str:
    buttons = rng.choice(["", Button.LEFT, Button.RIGHT]) + rng.choice(["", Button.UP, Button.DOWN])
    for button, probability in ((Button.JUMP, 0.1), (Button.DASH, 0.05), (Button.SHOOT, 0.3)):
        if rng.random() < probability:
            buttons += button
    return buttons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=3000)
    parser.add_argument("--fps", type=int, default=0, help="simulation speed cap, 0 = uncapped")
    parser.add_argument("--level", type=int, default=1, help="versus tower, 1-12")
    parser.add_argument("--kill", action="store_true", help="close the game when done")
    args = parser.parse_args()

    rng = random.Random(0)
    config = {
        "mode": "versus",
        "level": args.level,
        "fps": args.fps,
        "agentTimeout": "00:00:10",
        "agents": [
            {"type": "remote", "archer": "green", "team": "blue"},
            {"type": "remote", "archer": "red", "team": "red"},
        ],
    }
    game = Towerfall(config)
    print(f"Connected to TowerFall pid={game.pid} port={game.port}")
    agents: list[AgentConnection] = [game.join() for _ in config["agents"]]

    entity_types: set[str] = set()
    frames = 0
    start = time.perf_counter()
    try:
        while frames < args.frames:
            # Every agent has to answer every frame, or the game stalls until agentTimeout.
            updates = read_updates(agents)
            for agent in agents:
                agent.send_actions(random_buttons(rng))
            entity_types.update(entity["type"] for entity in updates[0]["entities"])
            frames += 1
            if frames == 1:
                # Level loading happens before the first update; keep it out of the timing.
                start = time.perf_counter()
    except TimeoutError:
        print("No update within the timeout: the match probably ended.")
    finally:
        elapsed = time.perf_counter() - start
        for agent in agents:
            agent.close()
        game.close(kill=args.kill)

    print(
        f"{frames} frames in {elapsed:.1f}s = {frames / elapsed:.0f} fps "
        f"({frames / elapsed / 60:.1f}x real time)"
    )
    print(f"Agent indices: {[agent.index for agent in agents]}")
    print(f"Entity types seen: {sorted(entity_types)}")


if __name__ == "__main__":
    main()
