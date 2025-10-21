"""
Logging utilities for VessQC

Provides centralized logging to both console and file.

Exports
-------
setup_logging, debug_print, info_print, error_print
"""

# Copyright © Peter Lampen, Lennart Kowitz, ISAS Dortmund, 2025

import logging
from pathlib import Path

# Setup logging
LOG_FILE = Path.home() / '.vessqc_debug.log'

def setup_logging():
    """Setup logging to file and console"""
    # Clear any existing handlers
    logger = logging.getLogger('vessqc')
    logger.handlers.clear()
    
    # Set level
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

def debug_print(msg: str):
    """Print and log debug message"""
    print(msg)
    logger.debug(msg)

def info_print(msg: str):
    """Print and log info message"""
    print(msg)
    logger.info(msg)

def error_print(msg: str):
    """Print and log error message"""
    print(msg)
    logger.error(msg)

def get_log_file() -> Path:
    """Get the path to the log file"""
    return LOG_FILE
