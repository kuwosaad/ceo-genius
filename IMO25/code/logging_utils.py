from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, TextIO

_log_file: Optional[TextIO] = None
_log_directory = "../logs"
_log_counter_file = os.path.join(_log_directory, "log_counter.txt")
_log_number: Optional[int] = None
_verbose_mode = False
_original_print = print


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_print(*args, **kwargs) -> None:
    timestamp = get_timestamp()
    message = " ".join(str(arg) for arg in args)
    timestamped = f"[{timestamp}] {message}"
    _original_print(timestamped, **kwargs)
    if _log_file is not None:
        _log_file.write(timestamped + "\n")
        _log_file.flush()


def debug_print(*args, **kwargs) -> None:
    if _verbose_mode:
        log_print("[DEBUG]", *args, **kwargs)


def set_verbose_mode(value: bool) -> None:
    global _verbose_mode
    _verbose_mode = value


def get_next_log_number() -> int:
    global _log_counter_file, _log_number
    if _log_number is not None:
        return _log_number
    os.makedirs(os.path.dirname(_log_counter_file), exist_ok=True)
    counter = 1
    try:
        if os.path.exists(_log_counter_file):
            with open(_log_counter_file, "r") as f:
                counter = int(f.read().strip())
    except Exception:
        counter = 1
    try:
        with open(_log_counter_file, "w") as f:
            f.write(str(counter + 1))
    except Exception:
        pass
    _log_number = counter
    return counter


def initialize_logging(log_directory: str = "logs") -> str:
    global _log_directory
    _log_directory = log_directory
    os.makedirs(_log_directory, exist_ok=True)
    log_number = get_next_log_number()
    return os.path.join(_log_directory, f"IMO{log_number}.log")


def set_log_file(log_file_path: str) -> bool:
    global _log_file
    if log_file_path:
        try:
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            _log_file = open(log_file_path, "w", encoding="utf-8")
            return True
        except Exception as e:
            log_print(f"Error opening log file {log_file_path}: {e}")
            return False
    return True


def close_log_file() -> None:
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None
