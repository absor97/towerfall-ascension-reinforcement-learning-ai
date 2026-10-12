import numpy as np
import pytest

from towerfall_rl.envs import (
    ACTION_SPACE,
    ObservationEncoder,
    is_stomp,
    to_buttons,
    wrapped_offset,
)


def entity(kind: str, x: float, y: float, **extra) -> dict:
    return {
        "type": kind,
        "pos": {"x": x, "y": y},
        "vel": {"x": 0.0, "y": 0.0},
        "isEnemy": False,
        "isDead": False,
        "facing": 1,
        **extra,
    }


def archer(x: float = 100.0, y: float = 107.0, arrows: int = 3) -> dict:
    return entity(
        "archer",
        x,
        y,
        onGround=True,
        onWall=False,
        aimDirection={"x": 1.0, "y": 0.0},
        arrows=["normal"] * arrows,
        state="normal",
    )


def slime(x: float, y: float = 105.0, dead: bool = False) -> dict:
    return entity("slime", x, y, isEnemy=True, isDead=dead)


@pytest.fixture
def flat_floor() -> np.ndarray:
    grid = np.zeros((32, 24), dtype=np.float32)
    grid[:, 9] = 1  # Solid from y=90 to y=100.
    return grid


def test_wrapped_offset_takes_the_short_way_around():
    assert wrapped_offset(entity("a", 10, 100), entity("b", 300, 100)) == (-30, 0)
    assert wrapped_offset(entity("a", 100, 10), entity("b", 100, 230)) == (0, -20)
    assert wrapped_offset(entity("a", 100, 100), entity("b", 160, 130)) == (60, 30)


def test_dead_archer_sees_zeros(flat_floor):
    encoder = ObservationEncoder()
    observation = encoder.encode(None, [slime(200)], flat_floor)
    assert observation.shape == encoder.space.shape
    assert not observation.any()


def test_enemy_slots_hold_the_nearest_living_enemies(flat_floor):
    encoder = ObservationEncoder(max_enemies=2)
    me = archer(x=100)
    entities = [me, slime(260), slime(110, dead=True), slime(300), slime(60)]

    observation = encoder.encode(me, entities, flat_floor)

    first, second = observation[15:20], observation[20:25]
    assert first[0] == 1 and first[1] == pytest.approx(-40 / 160)
    # x=300 is 120px away going left across the screen edge, nearer than x=260.
    assert second[0] == 1 and second[1] == pytest.approx(-120 / 160)


def test_empty_slots_stay_zero_and_arrows_report_pickup(flat_floor):
    encoder = ObservationEncoder(max_enemies=1, max_arrows=2)
    me = archer(arrows=1)
    stuck = entity("arrow", 180, 101, state="stuck")

    observation = encoder.encode(me, [me, stuck], flat_floor, held_buttons="rs")

    assert observation[9] == pytest.approx(1 / 6)
    assert list(observation[12:15]) == [0, 0, 1]  # jump, dash, shoot held
    assert not observation[15:20].any()
    assert observation[20] == 1 and observation[21] == pytest.approx(0.5)
    assert observation[25] == 1  # can be picked up
    assert not observation[26:32].any()


def test_terrain_window_is_centred_on_the_archer(flat_floor):
    encoder = ObservationEncoder(terrain_radius=(1, 2))
    me = archer(x=5, y=107)  # Cell (0, 10): the window wraps past the left edge.

    window = encoder.encode(me, [me], flat_floor)[-15:].reshape(3, 5)

    # Rows 8..12 of each column; only the floor row 9 is solid.
    assert (window == np.array([0, 1, 0, 0, 0], dtype=np.float32)).all()


def test_observation_stays_inside_its_space(flat_floor):
    encoder = ObservationEncoder()
    me = archer(x=-3, y=250)
    me["vel"] = {"x": 40.0, "y": -40.0}
    observation = encoder.encode(me, [me, slime(200)], flat_floor)
    assert encoder.space.contains(observation)


def test_is_stomp_needs_the_archer_right_above_the_slime():
    # Offsets measured in the game at the frame of the kill.
    assert is_stomp(archer(x=164.6, y=117.0), slime(160, 105.0))
    assert is_stomp(archer(x=171.2, y=117.0), slime(160, 105.0))
    assert is_stomp(archer(x=2.0, y=117.0), slime(316, 105.0))  # across the screen edge
    assert not is_stomp(archer(x=106.1, y=107.0), slime(160, 105.0))  # arrow from the floor
    assert not is_stomp(archer(x=141.6, y=105.2), slime(160, 105.0))  # point blank arrow
    assert not is_stomp(archer(x=126.8, y=118.5), slime(160, 105.0))  # shot while jumping


def test_to_buttons():
    assert to_buttons([0, 0, 0, 0, 0]) == ""
    assert to_buttons([1, 2, 1, 0, 1]) == "lujs"
    assert to_buttons([2, 1, 0, 1, 0]) == "rdz"
    assert all(set(to_buttons(ACTION_SPACE.sample())) <= set("lrudjzs") for _ in range(50))
