"""Centralized logging for sangam.

All modules should use ``init_logger(__name__)`` instead of
``logging.getLogger(__name__)``.  This keeps every logger under the
``"sangam"`` root so that a single ``logging.basicConfig`` in the
entry-point configures the whole tree.
"""

import logging
import sys


class NewLineFormatter(logging.Formatter):
    """Formatter that inserts a newline between the header and multi-line messages."""

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\n" + parts[0])
        return msg


_ROOT_LOGGER = logging.getLogger("sangam")
_DEFAULT_HANDLER: logging.Handler | None = None
_DEFAULT_LOG_LEVEL = logging.INFO
_LOG_FORMAT = "%(levelname)s %(asctime)s %(name)s: %(message)s"
_LOG_DATE_FORMAT = "%m-%d %H:%M:%S"


def _configure_root_logger() -> None:
    global _DEFAULT_HANDLER
    if _DEFAULT_HANDLER is not None:
        return

    _DEFAULT_HANDLER = logging.StreamHandler(sys.stdout)
    _DEFAULT_HANDLER.flush = sys.stdout.flush  # type: ignore[attr-defined]
    _DEFAULT_HANDLER.setLevel(_DEFAULT_LOG_LEVEL)

    fmt = NewLineFormatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)
    _DEFAULT_HANDLER.setFormatter(fmt)
    _ROOT_LOGGER.addHandler(_DEFAULT_HANDLER)


def init_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``"sangam"`` root.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` that inherits the root's handlers.
    """
    _configure_root_logger()
    logger = logging.getLogger(name)
    logger.setLevel(_DEFAULT_LOG_LEVEL)
    logger.propagate = False
    if _DEFAULT_HANDLER not in logger.handlers:
        logger.addHandler(_DEFAULT_HANDLER)
    return logger
