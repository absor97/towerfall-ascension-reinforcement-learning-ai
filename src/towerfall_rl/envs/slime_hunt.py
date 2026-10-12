"""First training task: one archer on a flat floor has to kill the slimes before they reach it."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from towerfall_rl.client import WIDTH, Message, Towerfall, TowerfallError

from .actions import ACTION_SPACE, to_buttons
from .observation import GRID_SHAPE, ObservationEncoder, wrapped_offset

# Rows from the top of the screen; the floor's surface ends up at y=100 in game coordinates.
_FLOOR_ROW = 14
_SOLIDS = [[int(row == _FLOOR_ROW)] * GRID_SHAPE[0] for row in range(GRID_SHAPE[1])]
_SPAWN_Y = 110


@dataclass(frozen=True)
class Rewards:
    """Points the agent earns or loses, sized relative to one kill."""

    kill: float = 1.0
    # Extra for a kill made by landing on the slime instead of shooting it.
    stomp_bonus: float = 0.5
    death: float = 2.0
    # Worth of holding at least one arrow: gained when picking one up with an empty quiver and
    # lost again when firing the last one, so shooting and re-collecting an arrow earns nothing.
    armed: float = 0.2
    # For starting a dodge within `close_dodge_radius` pixels of a living slime. Limited per
    # episode, otherwise standing next to a slime and dodging would pay better than killing it.
    close_dodge: float = 0.1
    close_dodge_radius: float = 40.0
    close_dodge_limit: int = 3
    # Lost every game frame, so that finishing quickly beats waiting.
    frame: float = 0.0005


# The plain objective without any hints, as a baseline to compare shaped rewards against.
SIMPLE_REWARDS = Rewards(stomp_bonus=0.0, death=1.0, armed=0.0, close_dodge=0.0)


def is_stomp(archer: Message, slime: Message) -> bool:
    """Whether a slime that just died was killed by this archer landing on it.

    At the frame of a stomp the archer's centre is 12px above the slime's and at most 12px to
    the side. For an arrow kill the archer is either level with the slime or far away.
    """
    dx, dy = wrapped_offset(slime, archer)
    return abs(dx) <= 14 and 6 <= dy <= 18


class SlimeHuntEnv(gym.Env):
    """Sandbox episode with random spawns, scored by `Rewards`.

    The episode ends when the archer dies or every slime is dead, and is truncated after
    `max_episode_frames`. Each `step` holds the chosen buttons for `frame_skip` game frames.
    `fps` caps the game speed: 0 for training, 60 to watch. `headless_core` turns drawing off,
    hides the window and gives the game that CPU core; pass a different number to every parallel
    environment.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        # Three slimes against a three-arrow quiver: random play wins about 44% of episodes.
        enemy_count: int = 3,
        frame_skip: int = 4,
        max_episode_frames: int = 1200,
        spawn_distance: tuple[float, float] = (50.0, 140.0),
        rewards: Rewards | None = None,
        fps: int = 0,
        headless_core: int | None = None,
        game_dir: str | Path | None = None,
    ) -> None:
        self.enemy_count = enemy_count
        self.frame_skip = frame_skip
        self.max_episode_frames = max_episode_frames
        self.spawn_distance = spawn_distance
        self.rewards = rewards or Rewards()

        self._encoder = ObservationEncoder(max_enemies=enemy_count)
        self.observation_space = self._encoder.space
        self.action_space = ACTION_SPACE

        config = {
            "mode": "sandbox",
            "level": 2,
            "fps": fps,
            # The game waits this long for our next action; learning updates take a while.
            "agentTimeout": "12:00:00",
            "solids": _SOLIDS,
            "noGraphics": headless_core is not None,
            "agents": [{"type": "remote", "archer": "green"}],
        }
        self._game = Towerfall(config, game_dir=game_dir)
        if headless_core is not None:
            self._game.run_headless(headless_core)
        self._agent = self._game.join(timeout=60.0)

        self._update: Message | None = None
        self._grid = np.zeros(GRID_SHAPE, dtype=np.float32)
        self._scenario: Message | None = None
        self._alive_enemies: set[int] = set()
        self._held_buttons = ""
        self._frames = 0
        self._kills = 0
        self._stomps = 0
        self._close_dodges = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        # Seeding the game as well makes an episode replay exactly from the env's seed.
        self._game.reset(self._spawn(), seed=int(self.np_random.integers(2**31)))
        if self._update is not None:
            # The game is still waiting for an answer to the last frame of the old episode.
            self._agent.send_actions()
        self._update = self._agent.read_update()
        if self._update["id"] != 0:
            raise TowerfallError(f"Expected a fresh episode, got frame {self._update['id']}")
        if self._agent.scenario is not self._scenario:
            self._scenario = self._agent.scenario
            self._grid = np.asarray(self._scenario["grid"], dtype=np.float32)

        self._alive_enemies = self._enemies_alive(self._update["entities"])
        self._held_buttons = ""
        self._frames = 0
        self._kills = 0
        self._stomps = 0
        self._close_dodges = 0
        return self._observation(), {}

    def step(self, action: Sequence[int]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._held_buttons = to_buttons(action)
        reward = 0.0
        died = False
        terminated = False
        for _ in range(self.frame_skip):
            archer_before = self._archer()
            self._agent.send_actions(self._held_buttons)
            self._update = self._agent.read_update()
            self._frames += 1

            frame_reward, died = self._score_frame(archer_before)
            reward += frame_reward
            terminated = died or not self._alive_enemies
            if terminated:
                break

        truncated = not terminated and self._frames >= self.max_episode_frames
        info = {
            "kills": self._kills,
            "stomps": self._stomps,
            "died": died,
            "won": not died and not self._alive_enemies,
        }
        return self._observation(), reward, terminated, truncated, info

    def _score_frame(self, archer_before: Message | None) -> tuple[float, bool]:
        """Reward for the frame that just arrived, and whether the archer died in it."""
        rewards = self.rewards
        archer = self._archer()
        enemies = {e["id"]: e for e in self._update["entities"] if e["isEnemy"]}
        alive = {enemy_id for enemy_id, enemy in enemies.items() if not enemy["isDead"]}
        reward = -rewards.frame

        for enemy_id in self._alive_enemies - alive:
            self._kills += 1
            reward += rewards.kill
            slime = enemies.get(enemy_id)
            if archer and slime and is_stomp(archer, slime):
                self._stomps += 1
                reward += rewards.stomp_bonus
        self._alive_enemies = alive

        if archer is None:
            return reward - rewards.death, True
        if archer_before is None:
            return reward, False

        reward += rewards.armed * (bool(archer["arrows"]) - bool(archer_before["arrows"]))

        started_dodge = archer["state"] == "dodging" and archer_before["state"] != "dodging"
        if started_dodge and self._close_dodges < rewards.close_dodge_limit:
            offsets = (wrapped_offset(archer, enemies[enemy_id]) for enemy_id in alive)
            if any(math.hypot(dx, dy) <= rewards.close_dodge_radius for dx, dy in offsets):
                self._close_dodges += 1
                reward += rewards.close_dodge
        return reward, False

    def close(self) -> None:
        self._agent.close()
        self._game.close(kill=True)

    def _spawn(self) -> list[Message]:
        archer_x = self.np_random.uniform(30, WIDTH - 30)
        entities: list[Message] = [{"type": "archer", "pos": {"x": archer_x, "y": _SPAWN_Y}}]
        for _ in range(self.enemy_count):
            side = self.np_random.choice((-1, 1))
            x = (archer_x + side * self.np_random.uniform(*self.spawn_distance)) % WIDTH
            entities.append(
                {"type": "slime", "facing": int(-side), "pos": {"x": float(x), "y": _SPAWN_Y}}
            )
        return entities

    def _archer(self) -> Message | None:
        for entity in self._update["entities"]:
            if entity["type"] == "archer" and entity["playerIndex"] == self._agent.index:
                return None if entity["dead"] else entity
        return None

    @staticmethod
    def _enemies_alive(entities: Sequence[Message]) -> set[int]:
        return {entity["id"] for entity in entities if entity["isEnemy"] and not entity["isDead"]}

    def _observation(self) -> np.ndarray:
        return self._encoder.encode(
            self._archer(), self._update["entities"], self._grid, self._held_buttons
        )
