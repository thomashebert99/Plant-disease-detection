import sys

from loguru import logger


def setup_logger(log_level: str = "INFO", log_file: str = "logs/app.log") -> None:
    """
    Configure Loguru pour le projet.
    - Console : logs colorés avec niveau INFO minimum
    - Fichier : rotation quotidienne, conservation 7 jours, compression gzip
    """
    # Supprimer le handler par défaut
    logger.remove()

    # Handler console — format lisible avec couleurs
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>",
        colorize=True,
    )

    # Handler fichier — rotation quotidienne, rétention 7 jours
    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="1 day",
        retention="7 days",
        compression="gz",
        encoding="utf-8",
    )
