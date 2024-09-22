"""Utility functions.

This module contains any generic logic that is used in the project.
"""

import structlog

def get_logger(model_name: str) -> structlog.stdlib.BoundLogger:
    """Get a logger instance.

    Args:
        model_name (str): Name of the model

    Returns:
        structlog.stdlib.BoundLogger: Logger instance
    """
    logger = structlog.get_logger()
    logger = logger.bind(model_name=model_name)
    logger.warning(f"Logger is active for the model: {model_name}.")
    return logger





def log_model_evaluation(logger: structlog.stdlib.BoundLogger, metrics: dict) -> None:
    """Log the evaluation metrics.

    Args:
        logger (structlog.stdlib.BoundLogger): Logger instance
        metrics (dict): Evaluation metrics
    """
    logger.info("Model evaluation metrics", **metrics)
    logger.warning("Model evaluation metrics have been successfully logged.")

# End of Python Script
