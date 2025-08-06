from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, TextIO

# Internal logging state
_log_file: Optional[TextIO] = None
_log_directory = "../logs"  # Consolidate logs to project root
_log_counter_file = os.path.join(_log_directory, "log_counter.txt")
_log_number: Optional[int] = None
_verbose_mode = False
original_print = print


def set_verbose_mode(flag: bool) -> None:
    """Enable or disable verbose debugging output."""
    global _verbose_mode
    _verbose_mode = flag


def get_timestamp() -> str:
    """Get current timestamp for logging."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_print(*args, **kwargs) -> None:
    """Custom print that writes to stdout and log file with timestamps."""
    timestamp = get_timestamp()
    message = " ".join(str(arg) for arg in args)
    timestamped_message = f"[{timestamp}] {message}"
    original_print(timestamped_message, **kwargs)
    if _log_file is not None:
        _log_file.write(timestamped_message + "\n")
        _log_file.flush()


def debug_print(*args, **kwargs) -> None:
    """Print debug messages only when verbose mode is enabled."""
    if _verbose_mode:
        log_print("[DEBUG]", *args, **kwargs)


def get_next_log_number() -> int:
    """Get the next sequential log number and increment the counter."""
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
    """Initialize the logging directory and return a numbered log file path."""
    global _log_directory, _log_counter_file
    _log_directory = log_directory
    os.makedirs(_log_directory, exist_ok=True)
    _log_counter_file = os.path.join(_log_directory, "log_counter.txt")
    log_number = get_next_log_number()
    return os.path.join(_log_directory, f"IMO{log_number}.log")


def get_log_directory() -> str:
    """Return the active log directory."""
    return _log_directory


def set_log_file(log_file_path: str) -> bool:
    """Set the log file for output."""
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
    """Close the log file if it's open."""
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None
