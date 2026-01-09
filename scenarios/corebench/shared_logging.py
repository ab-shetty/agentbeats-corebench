"""
Shared logging configuration for CoreBench
Both evaluator and agent write to the same timestamped log file
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
import os

# Global log file path - set once at startup
_LOG_FILE = None
_LOG_INITIALIZED = False


def get_log_file():
    """Get or create the shared log file path"""
    global _LOG_FILE
    if _LOG_FILE is None:
        log_dir = Path(os.getenv("COREBENCH_LOG_DIR", "logs"))
        log_dir.mkdir(exist_ok=True)
        # Use daily timestamp so all runs on same day share the same file
        timestamp = datetime.now().strftime("%Y%m%d")
        _LOG_FILE = log_dir / f"corebench_run_{timestamp}.log"
    return _LOG_FILE


def setup_logging(component_name: str):
    """
    Set up logging for a component (evaluator or agent).
    All components write to the same shared log file.
    
    Args:
        component_name: Name of component (e.g., 'evaluator', 'purple_agent')
    """
    global _LOG_INITIALIZED
    
    log_file = get_log_file()
    
    # Get root logger
    root_logger = logging.getLogger()
    
    # Only configure handlers once
    if not _LOG_INITIALIZED:
        root_logger.setLevel(logging.DEBUG)
        root_logger.handlers.clear()
        
        # Detailed file logging
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-20s:%(lineno)-4d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Console logging
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        _LOG_INITIALIZED = True
        
        root_logger.info("=" * 100)
        root_logger.info(f"COREBENCH RUN - Unified log file: {log_file}")
        root_logger.info("=" * 100)
    
    # Log component startup
    logger = logging.getLogger(component_name)
    logger.info(f"{'=' * 100}")
    logger.info(f"{component_name.upper()} COMPONENT STARTING")
    logger.info(f"{'=' * 100}")
    
    return log_file