"""Socket connections to the TowerFall AI mod (our fork in mod/ of
https://github.com/TowerfallAi/towerfall-ai).

Wire format: a 2-byte big-endian payload length followed by the payload as ASCII JSON.
"""

import json
import socket
from collections.abc import Mapping, Sequence
from typing import Any

_HEADER_SIZE = 2
_MAX_MESSAGE_SIZE = (1 << 16) - 1

# The fork's version. The upstream release (0.1.1) lacks the events, the seed and the fields
# this client relies on.
MOD_VERSION = "0.2.0"

Message = dict[str, Any]


class Connection:
    """A framed JSON connection to the game."""

    def __init__(self, port: int, host: str = "127.0.0.1", timeout: float | None = None) -> None:
        self._socket = socket.create_connection((host, port), timeout=timeout)
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def send(self, message: Mapping[str, Any]) -> None:
        payload = json.dumps(message, separators=(",", ":")).encode("ascii")
        if len(payload) > _MAX_MESSAGE_SIZE:
            raise ValueError(f"Message of {len(payload)} bytes exceeds {_MAX_MESSAGE_SIZE} bytes")
        # Header and payload go out in one call: the mod reads the payload with a single
        # Receive and would truncate a message that arrives split across segments.
        self._socket.sendall(len(payload).to_bytes(_HEADER_SIZE, "big") + payload)

    def recv(self) -> Message:
        size = int.from_bytes(self._recv_exact(_HEADER_SIZE), "big")
        return json.loads(self._recv_exact(size))

    def request(self, message: Mapping[str, Any]) -> Message:
        self.send(message)
        return self.recv()

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> "Connection":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _recv_exact(self, size: int) -> bytes:
        buffer = bytearray(size)
        view = memoryview(buffer)
        received = 0
        while received < size:
            count = self._socket.recv_into(view[received:])
            if count == 0:
                raise ConnectionError("TowerFall closed the connection")
            received += count
        return bytes(buffer)


class AgentConnection:
    """The agent side of a joined connection: reads game state, sends button presses."""

    def __init__(self, connection: Connection) -> None:
        self.connection = connection
        self.index: int | None = None
        self.scenario: Message | None = None
        self._update_id: int | None = None

    def poll(self) -> Message | None:
        """Handles one message: returns an `update` frame, acknowledges anything else.

        The game resends `init` and `scenario` every time a level loads, so they can show up
        between any two updates.
        """
        message = self.connection.recv()
        match message["type"]:
            case "init":
                if message["version"] != MOD_VERSION:
                    raise RuntimeError(
                        f"TowerFall runs version {message['version']} of the mod, but this client "
                        f"needs {MOD_VERSION}. Run mod/install.ps1 with the game closed."
                    )
                self.index = message["index"]
                self._ack()
            case "scenario":
                self.scenario = message
                self._ack()
            case "update":
                self._update_id = message["id"]
                return message
            case other:
                raise ValueError(f"Unexpected message type from TowerFall: {other}")
        return None

    def read_update(self) -> Message:
        """Blocks until the next `update` frame. With several agents use `read_updates`."""
        while True:
            update = self.poll()
            if update is not None:
                return update

    def send_actions(self, buttons: str = "", frames: int = 1) -> None:
        """Replies to the last update with the buttons to hold (see `Button`).

        The game holds them for `frames` frames and only then sends the next update, which saves
        the round trip for the frames in between. With several agents the smallest value counts.
        """
        message = {"type": "actions", "actions": buttons, "id": self._update_id}
        if frames > 1:
            message["frames"] = frames
        self.connection.send(message)

    def close(self) -> None:
        self.connection.close()

    def _ack(self) -> None:
        self.connection.send({"type": "result", "success": True})


def read_updates(agents: Sequence[AgentConnection]) -> list[Message]:
    """Reads the next `update` frame for every agent of one game.

    The game sends each of `init`, `scenario` and `update` to all agents and waits for all of
    them to answer before moving on, so the agents have to be served one message at a time.
    """
    updates: list[Message | None] = [None] * len(agents)
    while any(update is None for update in updates):
        for i, agent in enumerate(agents):
            if updates[i] is None:
                updates[i] = agent.poll()
    return updates
