"""Turns the mod's per-frame entity list into a fixed-size observation vector.

Game coordinates are pixels with the origin at the bottom left and y pointing up. The playfield
wraps around on both axes, so relative positions use the shortest wrapped offset.
"""

from collections.abc import Sequence

import numpy as np
from gymnasium import spaces

from towerfall_rl.client import HEIGHT, WIDTH, Button, Message

GRID_SHAPE = (32, 24)
CELL_SIZE = 10

_SPEED_SCALE = 10.0
_FULL_QUIVER = 6
_FLYING_ARROW_STATES = frozenset({"shooting", "gravity", "drilling", "falling"})

_SELF_FEATURES = 15
_ENEMY_FEATURES = 5
_ARROW_FEATURES = 6


def wrapped_offset(source: Message, target: Message) -> tuple[float, float]:
    """Shortest (dx, dy) from `source` to `target` on the wrapping playfield."""
    dx = (target["pos"]["x"] - source["pos"]["x"] + WIDTH / 2) % WIDTH - WIDTH / 2
    dy = (target["pos"]["y"] - source["pos"]["y"] + HEIGHT / 2) % HEIGHT - HEIGHT / 2
    return dx, dy


class ObservationEncoder:
    """Encodes what one archer knows: itself, nearby enemies and arrows, surrounding terrain.

    Layout of the vector, all values in [-1, 1]:
      - own state: position, velocity, facing, on ground, on wall, aim, arrows held, dodging,
        ducking, and whether jump / dash / shoot are currently held
      - `max_enemies` slots, nearest first: present, offset, velocity
      - `max_arrows` slots, nearest first: present, offset, velocity, can be picked up
      - solid cells in a window of `terrain_radius` cells around the archer
    """

    def __init__(
        self,
        max_enemies: int = 3,
        max_arrows: int = 4,
        terrain_radius: tuple[int, int] = (5, 4),
    ) -> None:
        self.max_enemies = max_enemies
        self.max_arrows = max_arrows
        self.terrain_radius = terrain_radius
        self._terrain_size = (2 * terrain_radius[0] + 1) * (2 * terrain_radius[1] + 1)
        self.size = (
            _SELF_FEATURES
            + max_enemies * _ENEMY_FEATURES
            + max_arrows * _ARROW_FEATURES
            + self._terrain_size
        )
        self.space = spaces.Box(-1.0, 1.0, shape=(self.size,), dtype=np.float32)

    def encode(
        self,
        archer: Message | None,
        entities: Sequence[Message],
        grid: np.ndarray,
        held_buttons: str = "",
    ) -> np.ndarray:
        """`grid` is the scenario grid as a (32, 24) array; a dead archer sees all zeros."""
        observation = np.zeros(self.size, dtype=np.float32)
        if archer is None:
            return observation

        observation[:_SELF_FEATURES] = (
            archer["pos"]["x"] / WIDTH * 2 - 1,
            archer["pos"]["y"] / HEIGHT * 2 - 1,
            archer["vel"]["x"] / _SPEED_SCALE,
            archer["vel"]["y"] / _SPEED_SCALE,
            archer["facing"],
            archer["onGround"],
            archer["onWall"],
            archer["aimDirection"]["x"],
            archer["aimDirection"]["y"],
            len(archer["arrows"]) / _FULL_QUIVER,
            archer["state"] == "dodging",
            archer["state"] == "ducking",
            Button.JUMP in held_buttons,
            Button.DASH in held_buttons,
            Button.SHOOT in held_buttons,
        )
        offset = _SELF_FEATURES

        enemies = [e for e in entities if e["isEnemy"] and not e["isDead"]]
        for enemy, (dx, dy) in self._nearest(archer, enemies, self.max_enemies):
            observation[offset : offset + _ENEMY_FEATURES] = (
                1,
                *self._relative(enemy, dx, dy),
            )
            offset += _ENEMY_FEATURES
        offset = _SELF_FEATURES + self.max_enemies * _ENEMY_FEATURES

        arrows = [e for e in entities if e["type"] == "arrow"]
        for arrow, (dx, dy) in self._nearest(archer, arrows, self.max_arrows):
            observation[offset : offset + _ARROW_FEATURES] = (
                1,
                *self._relative(arrow, dx, dy),
                arrow["state"] not in _FLYING_ARROW_STATES,
            )
            offset += _ARROW_FEATURES

        observation[-self._terrain_size :] = self._terrain(archer, grid).ravel()
        return np.clip(observation, -1.0, 1.0, out=observation)

    @staticmethod
    def _nearest(
        archer: Message, entities: Sequence[Message], count: int
    ) -> list[tuple[Message, tuple[float, float]]]:
        with_offsets = [(entity, wrapped_offset(archer, entity)) for entity in entities]
        with_offsets.sort(key=lambda item: item[1][0] ** 2 + item[1][1] ** 2)
        return with_offsets[:count]

    @staticmethod
    def _relative(entity: Message, dx: float, dy: float) -> tuple[float, float, float, float]:
        return (
            dx / (WIDTH / 2),
            dy / (HEIGHT / 2),
            entity["vel"]["x"] / _SPEED_SCALE,
            entity["vel"]["y"] / _SPEED_SCALE,
        )

    def _terrain(self, archer: Message, grid: np.ndarray) -> np.ndarray:
        cell_x = int(archer["pos"]["x"] // CELL_SIZE)
        cell_y = int(archer["pos"]["y"] // CELL_SIZE)
        radius_x, radius_y = self.terrain_radius
        columns = np.arange(cell_x - radius_x, cell_x + radius_x + 1) % GRID_SHAPE[0]
        rows = np.arange(cell_y - radius_y, cell_y + radius_y + 1) % GRID_SHAPE[1]
        return grid[np.ix_(columns, rows)]
