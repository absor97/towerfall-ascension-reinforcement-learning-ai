"""Finds or launches a modded TowerFall.exe and drives its management API."""

import os
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

import psutil

from .connection import AgentConnection, Connection, Message
from .window import minimize_windows

DEFAULT_GAME_DIR = Path(r"C:\Program Files (x86)\Steam\steamapps\common\TowerFall")
GAME_DIR_ENV_VAR = "TOWERFALL_PATH"
EXE_NAME = "TowerFall.exe"
# With this file holding the Steam app id (251470) the game starts directly instead of asking
# Steam to relaunch it, which Steam refuses for a second copy.
STEAM_APPID_FILE = "steam_appid.txt"
_OWNED_MESSAGE = "Game is currently owned by a different client."

# Playfield size in game pixels; the scenario grid is 32x24 cells of 10px.
WIDTH = 320
HEIGHT = 240


class Button(StrEnum):
    LEFT = "l"
    RIGHT = "r"
    UP = "u"
    DOWN = "d"
    JUMP = "j"
    DASH = "z"
    SHOOT = "s"


class TowerfallError(RuntimeError):
    pass


def find_game_dir(game_dir: str | Path | None = None) -> Path:
    """Resolves the game folder: explicit argument, then $TOWERFALL_PATH, then Steam's default."""
    path = Path(game_dir or os.environ.get(GAME_DIR_ENV_VAR) or DEFAULT_GAME_DIR)
    if not (path / EXE_NAME).is_file():
        raise TowerfallError(
            f"{EXE_NAME} not found in {path}. Set {GAME_DIR_ENV_VAR} to the game folder."
        )
    return path


def is_mod_installed(game_dir: Path) -> bool:
    """True when the patcher has run: it keeps the vanilla exe under Patcher/Original."""
    return (game_dir / "TowerfallAiMod.dll").is_file() and (
        game_dir / "Patcher" / "Original" / EXE_NAME
    ).is_file()


def replay_mode(game_dir: Path) -> str | None:
    """The `ReplayMode` option stored in the game's save file: UseGPU, UseCPU or Off."""
    try:
        save_data = (game_dir / "tf_saveData").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    match = re.search(r"<ReplayMode>(\w+)</ReplayMode>", save_data)
    return match.group(1) if match else None


class Towerfall:
    """Owns one TowerFall.exe process through its management connection.

    A game that already runs with the mod and has no owner is reused; otherwise a new one is
    launched. `config` follows the mod's config reference, for example::

        {"mode": "versus", "level": 1, "fps": 60,
         "agents": [{"type": "remote", "team": "blue"}, {"type": "remote", "team": "red"}]}

    `fps` caps the simulation speed; 0 removes the cap.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        game_dir: str | Path | None = None,
        timeout: float = 10.0,
        launch_timeout: float = 60.0,
    ) -> None:
        self.game_dir = find_game_dir(game_dir)
        if not is_mod_installed(self.game_dir):
            raise TowerfallError(f"The AI mod is not installed in {self.game_dir}.")
        if replay_mode(self.game_dir) not in (None, "Off"):
            # The end-of-round replay freezes the agents and crashes the game when the
            # simulation runs faster than real time (IndexOutOfRange in ReplayViewer.Finish).
            raise TowerfallError(
                f"Replays are on. Set <ReplayMode>Off</ReplayMode> in "
                f"{self.game_dir / 'tf_saveData'} while the game is closed."
            )
        self.timeout = timeout
        self._pool_dir = self.game_dir / "aimod" / "pools" / "default"
        self.config = config
        self.pid, self.port, self._management = self._acquire_game(config, launch_timeout)

    def configure(self, config: Mapping[str, Any]) -> None:
        """Starts a new session. Every remote agent has to `join` again afterwards."""
        self._check(self._management.request({"type": "config", "config": config}), "configure")
        self.config = config

    def join(self, timeout: float | None = None) -> AgentConnection:
        """Takes the next free remote agent slot. The match starts once all slots are taken."""
        connection = Connection(self.port, timeout=timeout or self.timeout)
        self._check(connection.request({"type": "join"}), "join")
        return AgentConnection(connection)

    def reset(
        self, entities: Sequence[Mapping[str, Any]] | None = None, seed: int | None = None
    ) -> None:
        """Sandbox mode only: recreates the given entities, one `archer` per agent required.

        `seed` restarts the game's random numbers, so the same seed and the same button presses
        replay the same episode.
        """
        message: Message = {"type": "reset", "entities": entities}
        if seed is not None:
            message["seed"] = seed
        self._check(self._management.request(message), "reset")

    def run_headless(self, core: int) -> None:
        """Speeds the game up for training: no drawing, and a CPU core of its own.

        Games numbered 0, 1, 2... each get one physical core with all of its hardware threads.
        Left to the scheduler, several games run at a third of the speed. Cores are handed out
        from the last one down, because the rest of the system keeps the first ones busy.
        """
        minimize_windows(self.pid)
        physical = psutil.cpu_count(logical=False)
        threads_per_core = psutil.cpu_count() // physical
        first = (physical - 1 - core % physical) * threads_per_core
        psutil.Process(self.pid).cpu_affinity(list(range(first, first + threads_per_core)))

    def close(self, kill: bool = False) -> None:
        """Releases the game so another client can own it, or terminates it with `kill`."""
        self._management.close()
        if kill:
            try:
                psutil.Process(self.pid).terminate()
            except psutil.NoSuchProcess:
                pass

    def __enter__(self) -> "Towerfall":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _acquire_game(
        self, config: Mapping[str, Any], launch_timeout: float
    ) -> tuple[int, int, Connection]:
        """Claims a free game by configuring it, launching one if there is none.

        Clients that start together race for the same games; the loser moves on to the next.
        """
        launched = False
        deadline = time.monotonic() + launch_timeout
        while time.monotonic() < deadline:
            game = self._find_free_game()
            if game is None:
                if not launched:
                    exe = self.game_dir / EXE_NAME
                    subprocess.Popen([str(exe), "--aimod"], cwd=self.game_dir)
                    launched = True
                time.sleep(0.5)
                continue
            pid, port = game
            try:
                management = Connection(port, timeout=self.timeout)
                response = management.request({"type": "config", "config": config})
            except OSError:
                time.sleep(0.2)
                continue
            if response.get("success"):
                return pid, port, management
            management.close()
            if response.get("message") != _OWNED_MESSAGE:
                self._check(response, "configure")
            time.sleep(0.2)
        raise TowerfallError(
            f"No free {EXE_NAME} within {launch_timeout:.0f}s. Check that Steam is running, that "
            f"{STEAM_APPID_FILE} exists in the game folder when running several games, and see "
            f"the logs in {self.game_dir / 'aimod'}."
        )

    def _find_free_game(self) -> tuple[int, int] | None:
        """The mod keeps a file named after its pid in the pool while no client owns the game.

        The file also holds the port, but it must not be opened: the game deletes and rewrites
        it without sharing, and its server thread dies if that fails. The port is looked up
        from the process instead.
        """
        if not self._pool_dir.is_dir():
            return None
        for entry in self._pool_dir.iterdir():
            try:
                process = psutil.Process(int(entry.name))
                if process.name() != EXE_NAME:
                    raise psutil.NoSuchProcess(process.pid)
                for connection in process.net_connections(kind="tcp"):
                    if connection.status == psutil.CONN_LISTEN:
                        return process.pid, connection.laddr.port
            except (ValueError, psutil.NoSuchProcess):
                try:
                    entry.unlink(missing_ok=True)
                except OSError:
                    pass
            except psutil.Error:
                continue
        return None

    @staticmethod
    def _check(response: Message, action: str) -> None:
        if response.get("type") != "result" or not response.get("success"):
            raise TowerfallError(f"TowerFall refused to {action}: {response.get('message')}")
