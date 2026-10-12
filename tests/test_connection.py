"""Protocol tests against a fake game served from a local socket; TowerFall is not needed."""

import json
import socket
import threading
from collections.abc import Callable

import pytest

from towerfall_rl.client import MOD_VERSION, AgentConnection, Connection, read_updates


def frame(message: dict) -> bytes:
    payload = json.dumps(message).encode("ascii")
    return len(payload).to_bytes(2, "big") + payload


def read_frame(sock: socket.socket) -> dict:
    size = int.from_bytes(sock.recv(2), "big")
    return json.loads(sock.recv(size))


@pytest.fixture
def serve():
    """Runs `handler(server_sockets)` in a thread once `count` clients have connected."""
    listener = socket.create_server(("127.0.0.1", 0))
    threads: list[threading.Thread] = []

    def start(handler: Callable[[list[socket.socket]], None], count: int = 1) -> int:
        def run() -> None:
            clients = [listener.accept()[0] for _ in range(count)]
            try:
                handler(clients)
            finally:
                for client in clients:
                    client.close()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        threads.append(thread)
        return listener.getsockname()[1]

    yield start
    for thread in threads:
        thread.join(timeout=5)
    listener.close()


def test_request_round_trip(serve):
    def echo(clients: list[socket.socket]) -> None:
        clients[0].sendall(
            frame({"type": "result", "success": True, "echo": read_frame(clients[0])})
        )

    with Connection(serve(echo), timeout=5) as connection:
        response = connection.request({"type": "join"})

    assert response == {"type": "result", "success": True, "echo": {"type": "join"}}


def test_recv_reassembles_fragmented_message(serve):
    message = {"type": "update", "id": 7, "entities": [{"type": "archer"}] * 50}

    def trickle(clients: list[socket.socket]) -> None:
        data = frame(message)
        for i in range(0, len(data), 97):
            clients[0].sendall(data[i : i + 97])

    with Connection(serve(trickle), timeout=5) as connection:
        assert connection.recv() == message


def test_recv_raises_when_game_closes_connection(serve):
    with Connection(serve(lambda clients: None), timeout=5) as connection:
        with pytest.raises(ConnectionError):
            connection.recv()


def test_read_updates_handshakes_all_agents_in_lockstep(serve):
    def game(clients: list[socket.socket]) -> None:
        # Like the mod: send a stage to every agent, then wait for every agent to answer.
        for stage in ("init", "scenario"):
            for index, client in enumerate(clients):
                client.sendall(frame({"type": stage, "index": index, "version": MOD_VERSION}))
            for client in clients:
                assert read_frame(client) == {"type": "result", "success": True}
        for client in clients:
            client.sendall(frame({"type": "update", "id": 3, "entities": []}))
        for client in clients:
            assert read_frame(client) == {"type": "actions", "actions": "jr", "id": 3}

    port = serve(game, count=2)
    agents = [AgentConnection(Connection(port, timeout=5)) for _ in range(2)]

    updates = read_updates(agents)
    for agent in agents:
        agent.send_actions("jr")

    assert [update["id"] for update in updates] == [3, 3]
    assert [agent.index for agent in agents] == [0, 1]
    assert all(agent.scenario is not None for agent in agents)
    for agent in agents:
        agent.close()


def test_init_from_another_mod_version_is_rejected(serve):
    def game(clients: list[socket.socket]) -> None:
        clients[0].sendall(frame({"type": "init", "index": 0, "version": "0.1.1"}))

    agent = AgentConnection(Connection(serve(game), timeout=5))

    with pytest.raises(RuntimeError, match="0.1.1"):
        agent.poll()


def test_send_actions_asks_to_hold_buttons_only_for_several_frames(serve):
    received: list[dict] = []

    def game(clients: list[socket.socket]) -> None:
        for update_id in (0, 4):
            clients[0].sendall(frame({"type": "update", "id": update_id, "entities": []}))
            received.append(read_frame(clients[0]))

    agent = AgentConnection(Connection(serve(game), timeout=5))
    agent.read_update()
    agent.send_actions("r", frames=4)
    agent.read_update()
    agent.send_actions("r")
    agent.close()

    assert received == [
        {"type": "actions", "actions": "r", "id": 0, "frames": 4},
        {"type": "actions", "actions": "r", "id": 4},
    ]
