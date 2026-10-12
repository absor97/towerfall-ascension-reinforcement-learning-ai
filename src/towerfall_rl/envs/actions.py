"""The agent's action space and its translation to controller buttons."""

from collections.abc import Sequence

from gymnasium import spaces

from towerfall_rl.client import Button

# Horizontal axis, vertical axis, jump, dash, shoot. Buttons behave like a real controller:
# jump and dash trigger on the press, an arrow is fired when shoot is released, and the axes
# both move the archer and aim the bow.
ACTION_SPACE = spaces.MultiDiscrete([3, 3, 2, 2, 2])

_HORIZONTAL = ("", Button.LEFT, Button.RIGHT)
_VERTICAL = ("", Button.DOWN, Button.UP)
_TOGGLES = (Button.JUMP, Button.DASH, Button.SHOOT)


def to_buttons(action: Sequence[int]) -> str:
    """Converts a sample of `ACTION_SPACE` to the mod's string of held buttons."""
    buttons = _HORIZONTAL[action[0]] + _VERTICAL[action[1]]
    for button, pressed in zip(_TOGGLES, action[2:], strict=True):
        if pressed:
            buttons += button
    return buttons
