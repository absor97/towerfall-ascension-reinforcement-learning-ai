"""Win32 access to the game's window."""

import ctypes
from ctypes import wintypes

_SW_SHOWMINNOACTIVE = 7


def minimize_windows(pid: int) -> int:
    """Minimizes every visible top-level window of a process and returns how many it found.

    A minimized game skips drawing entirely, which is most of the cost of a frame.
    """
    user32 = ctypes.windll.user32
    count = 0

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd: int, _: int) -> bool:
        nonlocal count
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            user32.ShowWindow(hwnd, _SW_SHOWMINNOACTIVE)
            count += 1
        return True

    user32.EnumWindows(visit, 0)
    return count
