"""Central logging configuration for the application."""

import logging

LOGGER_NAME = "smart_waste_multimodal"

DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
DEFAULT_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a named logger without configuring global logging."""
    if not name:
        return logging.getLogger(LOGGER_NAME)

    normalized_name = name.strip(".")
    if not normalized_name:
        return logging.getLogger(LOGGER_NAME)

    if normalized_name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)

    if normalized_name.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(normalized_name)

    return logging.getLogger(f"{LOGGER_NAME}.{normalized_name}")


def configure_logging(
    level: int = logging.INFO,
    *,
    log_format: str = DEFAULT_LOG_FORMAT,
) -> None:
    """Configure standard application logging explicitly.

    Existing root handlers are preserved to avoid duplicate handlers
    and to remain compatible with test runners and hosting environments.
    """
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=DEFAULT_DATE_FORMAT,
    )
    logging.getLogger(LOGGER_NAME).setLevel(level)
