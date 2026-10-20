# Windows / IDE console UTF-8 handling — prevents mojibake in training logs
from __future__ import annotations

import os
import sys


def _console_code_page() -> int | None:
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        return int(ctypes.windll.kernel32.GetConsoleOutputCP())  # type: ignore[attr-defined]
    except (AttributeError, OSError, ValueError):
        return None


def _try_set_utf8_code_page() -> bool:
    if sys.platform != "win32":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
        return _console_code_page() == 65001
    except (AttributeError, OSError, ValueError):
        return False


def _normalize_encoding(name: str | None) -> str:
    if not name:
        return ""
    return name.lower().replace("-", "")


def _resolve_stdio_encoding() -> str:
    if sys.platform != "win32":
        return "utf-8"

    orig = _normalize_encoding(getattr(sys.stdout, "encoding", None))
    _try_set_utf8_code_page()
    cp = _console_code_page()

    if cp == 65001 or orig in ("utf8",):
        return "utf-8"
    if cp in (936, 54936) or orig in ("cp936", "gbk", "gb2312", "mbcs"):
        return "gbk"
    return "utf-8"


def configure_console_utf8() -> None:
    """Call early in script to normalise stdout / stderr encoding."""
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    enc = _resolve_stdio_encoding()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(
                encoding=enc,
                errors="replace",
                line_buffering=True,
                write_through=True,
            )
        except (AttributeError, OSError, ValueError, TypeError):
            try:
                stream.reconfigure(encoding=enc, errors="replace")
            except (AttributeError, OSError, ValueError, TypeError):
                pass
