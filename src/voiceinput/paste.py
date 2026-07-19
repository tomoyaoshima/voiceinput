import time

import pyperclip
import pync
from pynput.keyboard import Controller, Key

_kbd = Controller()


def copy_to_clipboard(text: str) -> None:
    pyperclip.copy(text)


def paste_active_app() -> None:
    time.sleep(0.05)
    with _kbd.pressed(Key.cmd):
        _kbd.press("v")
        _kbd.release("v")


def notify(title: str, message: str, *, enabled: bool = True) -> None:
    if not enabled:
        return
    pync.notify(message, title=title)
