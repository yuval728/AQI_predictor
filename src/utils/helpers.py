import logging
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    logger.info(f"Logging configured with level: {log_level}")


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
    """
    Save data to JSON file.
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Data saved to JSON: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        return False


def load_json(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load data from JSON file.
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"JSON file not found: {file_path}")
            return None
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Data loaded from JSON: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None


def save_pickle(obj: Any, file_path: Union[str, Path]) -> bool:
    """
    Save object to pickle file.
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f"Object saved to pickle: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving pickle to {file_path}: {e}")
        return False


def load_pickle(file_path: Union[str, Path]) -> Optional[Any]:
    """
    Load object from pickle file.
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"Pickle file not found: {file_path}")
            return None
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"Object loaded from pickle: {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Error loading pickle from {file_path}: {e}")
        return None




