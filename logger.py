import logging
import logging.config
import sys
from config import LOG_CONFIG

def setup_logger(name):
    """Set up logger with the specified configuration"""
    logger = logging.getLogger(name)
    
    # Configure logging using dictConfig
    logging.config.dictConfig(LOG_CONFIG)
    
    return logger 