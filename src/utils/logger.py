"""
logger.py — File-backed logger for Bird vs Drone Detector training runs.

Creates a timestamped log file and mirrors every message to Python's
standard logging infrastructure so other modules can use standard
``logging.getLogger()`` calls alongside this utility.
"""

from __future__ import annotations

import datetime
import logging
import os


class Logger:
    """Dual-output logger: writes to a timestamped file *and* the Python
    ``logging`` module simultaneously.

    Parameters
    ----------
    log_dir : str, optional
        Directory where log files are created.  Created automatically if it
        does not exist (default ``'logs'``).
    name : str, optional
        Base name for the logger and the log file (default ``'training'``).

    Examples
    --------
    >>> log = Logger(log_dir='logs', name='training')
    >>> log.log('Epoch 1 — loss: 0.342')
    >>> log.close()
    """

    def __init__(self, log_dir: str = "logs", name: str = "training") -> None:
        self._name    = name
        self._log_dir = log_dir
        self._closed  = False

        # Create log directory if needed.
        os.makedirs(log_dir, exist_ok=True)

        # Build timestamped filename: training_20240115_143022.txt
        timestamp      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename       = f"{name}_{timestamp}.txt"
        self._log_path = os.path.join(log_dir, filename)

        # Open file handle (UTF-8, line-buffered).
        self._file = open(self._log_path, "w", encoding="utf-8", buffering=1)  # noqa: WPS515

        # Configure Python logger – use a unique name to avoid collisions when
        # multiple Logger instances are created in one process.
        logger_name    = f"BirdDrone.{name}.{timestamp}"
        self._logger   = logging.getLogger(logger_name)
        self._logger.setLevel(logging.DEBUG)

        if not self._logger.handlers:
            handler   = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
                datefmt="%H:%M:%S",
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

        # Write a header line so the file is self-describing.
        header = (
            f"# BirdDrone Logger | name={name} | started={timestamp}\n"
            f"# Log path: {os.path.abspath(self._log_path)}\n"
        )
        self._file.write(header)
        self._logger.debug("Log file opened: %s", self._log_path)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def log(self, msg: str) -> None:
        """Write *msg* to the log file and the Python logger.

        Parameters
        ----------
        msg : str
            The message to record.  A timestamp prefix is prepended
            automatically in the file.

        Raises
        ------
        RuntimeError
            If :meth:`close` has already been called.
        """
        if self._closed:
            raise RuntimeError(
                "Logger.log() called after close().  Create a new Logger instance."
            )

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line      = f"[{timestamp}] {msg}\n"

        self._file.write(line)
        self._logger.info(msg)

    def get_log_path(self) -> str:
        """Return the absolute path of the current log file.

        Returns
        -------
        str
            Absolute filesystem path to the log file created during
            ``__init__``.
        """
        return os.path.abspath(self._log_path)

    def close(self) -> None:
        """Flush and close the underlying file handle.

        Calling :meth:`close` more than once is safe (idempotent).
        After closing, calls to :meth:`log` will raise ``RuntimeError``.
        """
        if self._closed:
            return

        self._file.flush()
        self._file.close()
        self._closed = True
        self._logger.debug("Log file closed: %s", self._log_path)

    # ------------------------------------------------------------------
    # Context-manager support  (``with Logger(...) as log:``)
    # ------------------------------------------------------------------

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        # Do not suppress exceptions.
        return False

    def __repr__(self) -> str:
        state = "closed" if self._closed else "open"
        return f"Logger(name={self._name!r}, path={self._log_path!r}, {state})"
