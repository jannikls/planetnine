import os
import yaml
from pathlib import Path
from loguru import logger
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

for dir_path in [DATA_DIR, CONFIG_DIR, LOGS_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

logger.add(
    LOGS_DIR / "planetnine_{time}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
)

def load_config(config_name: str = "survey_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = CONFIG_DIR / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config

config = load_config()

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CANDIDATES_DIR = DATA_DIR / "candidates"

for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CANDIDATES_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)