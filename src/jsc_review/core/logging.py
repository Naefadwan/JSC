"""Application logging configuration."""

import logging


def configure_logging(level: str) -> None:
    """Configure concise local process logging."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
