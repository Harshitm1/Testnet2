"""
Configuration file for the Delta Exchange Trading Bot
"""

import os

# API Configuration
API_CONFIG = {
    'base_url': 'https://cdn-ind.testnet.deltaex.org',  # India Testnet REST API URL
    'api_key': 'x8gvyWA4xp2bvNFU8jRjQICgb2Xn3D',  # Testnet API Key
    'api_secret': 'oUl3B4nAHRL00VKWyAcpKeOHBSyxSL1z2Jem9esrHFbZQ6mjbTvpb5HVG6d1'  # Testnet API Secret
}

# Trading Configuration
TRADING_CONFIG = {
    'symbol': 'ETHUSD',  # Trading pair
    'product_id': 1699,  # Specific product ID for ETHUSD
    'timeframe': '1m',  # 1-minute timeframe for high-frequency trading
    'initial_capital': 100,  # Initial trading capital in USD
    'leverage': 1,  # Trading leverage (1x)
    
    # Strategy parameters
    'sensitivity': 0.005,  # 0.5% price movement threshold (more sensitive for 1m)
    'min_volume_percentile': 30,  # Lower volume requirement for faster trading
    'trend_period': 10,  # Shorter trend period for 1-minute
    'min_trades_distance': 5,  # Shorter distance between trades for 1-minute
    
    # Risk management
    'stop_loss_pct': 0.02,  # 2% stop loss
    'trailing_stop_pct': 0.015,  # 1.5% trailing stop
    'max_position_size': 1.0  # Maximum position size as a fraction of capital
}

# Strategy configuration
STRATEGY_CONFIG = {
    'sensitivity': 0.015,  # 1.5% price movement threshold
    'min_volume_percentile': 50,  # Volume above 50th percentile
    'trend_period': 20,  # 20-period SMA for trend
    'min_trades_distance': 10  # Minimum 10 candles between trades
}

# Telegram configuration
TELEGRAM_CONFIG = {
    'bot_token': '7203764992:AAEW7YINK48mYlNRodV_jqpP33LlD2uqOLg',
    'chat_id': '1099769493'
}

# Delta Exchange API configuration
DELTA_CONFIG = {
    'api_key': 'x8gvyWA4xp2bvNFU8jRjQICgb2Xn3D',
    'api_secret': 'oUl3B4nAHRL00VKWyAcpKeOHBSyxSL1z2Jem9esrHFbZQ6mjbTvpb5HVG6d1'
}

# Logging Configuration
LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'filename': 'trading_1min.log',
            'mode': 'a'
        }
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
} 