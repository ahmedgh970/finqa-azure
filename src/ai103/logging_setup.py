"""Logs locaux ; les journaux HTTP détaillés des SDK restent désactivés."""

import logging


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("azure").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
