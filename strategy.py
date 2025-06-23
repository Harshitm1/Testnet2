import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
from config import STRATEGY_CONFIG, TRADING_CONFIG
from logger import setup_logger

logger = setup_logger('strategy')

class OrderBlocks:
    def __init__(self, sensitivity=0.015, ob_mitigation='Close', min_volume_percentile=50, trend_period=20, min_trades_distance=10):
        # Initialize with configuration parameters
        self.sensitivity = sensitivity  # 1.5% price movement threshold
        self.ob_mitigation = ob_mitigation
        self.buy_alert = True
        self.sell_alert = True
        self.bear_boxes = []
        self.bull_boxes = []
        self.min_volume_percentile = min_volume_percentile
        self.trend_period = trend_period
        self.min_trades_distance = min_trades_distance
        self.last_trade_index = -self.min_trades_distance
        
        # Initialize state variables
        self.current_position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = None
        self.stop_loss_price = None
        self.trailing_stop_price = None
        self.position_size = 0
        self.capital = TRADING_CONFIG['initial_capital']

    def calc_indicators(self, df):
        """Calculate technical indicators for analysis"""
        try:
            # Calculate percentage change over 4 bars
            df = df.copy()
            df.loc[:, 'pc'] = (df['open'] - df['open'].shift(4)) / df['open'].shift(4) * 100
            
            # Calculate volume metrics
            df.loc[:, 'volume_ma'] = df['volume'].rolling(window=20).mean()
            df.loc[:, 'volume_percentile'] = df['volume'].rolling(window=50).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else np.nan
            )
            
            # Calculate trend indicators
            df.loc[:, 'sma20'] = df['close'].rolling(window=self.trend_period).mean()
            df.loc[:, 'sma50'] = df['close'].rolling(window=50).mean()
            
            # Calculate ATR for volatility filtering
            df.loc[:, 'tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            df.loc[:, 'atr'] = df['tr'].rolling(window=14).mean()
            
            # Momentum indicator
            df.loc[:, 'roc'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
            
            # Add swing high/low detection (fix ambiguity)
            df.loc[:, 'swing_high'] = ((df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(2)) &
                                 (df['high'] > df['high'].shift(-1)) & (df['high'] > df['high'].shift(-2))).astype(int)
            df.loc[:, 'swing_low'] = ((df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(2)) &
                                (df['low'] < df['low'].shift(-1)) & (df['low'] < df['low'].shift(-2))).astype(int)
            
            logger.info("Successfully calculated technical indicators")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return df

    def is_valid_trade_condition(self, df, idx, trade_type='long'):
        """Check if trading conditions are met"""
        try:
            # Check if enough distance from last trade
            if idx - self.last_trade_index < self.min_trades_distance:
                logger.debug(f"Trade rejected: Too close to last trade (idx={idx}, last_trade_index={self.last_trade_index}, min_trades_distance={self.min_trades_distance})")
                print(f"Trade rejected: Too close to last trade (idx={idx}, last_trade_index={self.last_trade_index}, min_trades_distance={self.min_trades_distance})")
                return False
            # Check volume conditions
            volume_percentile = df['volume_percentile'].iloc[idx]
            if not isinstance(volume_percentile, (int, float)) or volume_percentile < self.min_volume_percentile:
                logger.debug(f"Trade rejected: Insufficient volume (volume_percentile={volume_percentile}, min={self.min_volume_percentile})")
                print(f"Trade rejected: Insufficient volume (volume_percentile={volume_percentile}, min={self.min_volume_percentile})")
                return False
            # Check trend conditions
            sma20 = df['sma20'].iloc[idx]
            sma50 = df['sma50'].iloc[idx]
            roc = df['roc'].iloc[idx]
            if not all(isinstance(x, (int, float)) for x in [sma20, sma50, roc]):
                logger.debug(f"Trade rejected: Invalid indicator values (sma20={sma20}, sma50={sma50}, roc={roc})")
                print(f"Trade rejected: Invalid indicator values (sma20={sma20}, sma50={sma50}, roc={roc})")
                return False
            if trade_type == 'long':
                if not (sma20 > sma50 and roc > 0):
                    logger.debug(f"Trade rejected: Trend conditions not met for long (sma20={sma20}, sma50={sma50}, roc={roc})")
                    print(f"Trade rejected: Trend conditions not met for long (sma20={sma20}, sma50={sma50}, roc={roc})")
                    return False
            else:  # short
                if not (sma20 < sma50 and roc < 0):
                    logger.debug(f"Trade rejected: Trend conditions not met for short (sma20={sma20}, sma50={sma50}, roc={roc})")
                    print(f"Trade rejected: Trend conditions not met for short (sma20={sma20}, sma50={sma50}, roc={roc})")
                    return False
            # Volatility check
            current_atr = df['atr'].iloc[idx]
            avg_atr = df['atr'].rolling(window=20).mean().iloc[idx]
            if not all(isinstance(x, (int, float)) for x in [current_atr, avg_atr]):
                logger.debug(f"Trade rejected: Invalid ATR values (current_atr={current_atr}, avg_atr={avg_atr})")
                print(f"Trade rejected: Invalid ATR values (current_atr={current_atr}, avg_atr={avg_atr})")
                return False
            if current_atr > avg_atr * 1.5:
                logger.debug(f"Trade rejected: Volatility too high (current_atr={current_atr}, avg_atr={avg_atr})")
                print(f"Trade rejected: Volatility too high (current_atr={current_atr}, avg_atr={avg_atr})")
                return False
            print(f"Trade conditions met for {trade_type} at idx={idx}")
            return True
        except Exception as e:
            logger.error(f"Error checking trade conditions: {str(e)}")
            print(f"Error checking trade conditions: {str(e)}")
            return False

    def find_order_blocks(self, df):
        try:
            df = self.calc_indicators(df)
            current_position = 0  # Track current position: 0 = no position, 1 = long, -1 = short
            
            for idx in range(4, len(df)):
                pc = df['pc'].iloc[idx]
                prev_pc = df['pc'].iloc[idx-1]
                
                # Skip if pc or prev_pc is not a valid number
                if not isinstance(pc, (int, float)) or not isinstance(prev_pc, (int, float)):
                    continue
                    
                # Check for bearish order block
                if (prev_pc > -self.sensitivity and pc <= -self.sensitivity and 
                    not any(idx - b['start_idx'] <= 5 for b in self.bear_boxes) and
                    current_position != -1):  # Only enter short if not already in a short position
                    
                    # Only add if trade conditions are met
                    if self.is_valid_trade_condition(df, idx, 'short'):
                        for i in range(idx-4, max(idx-16, -1), -1):
                            close_price = df['close'].iloc[i]
                            open_price = df['open'].iloc[i]
                            
                            if not isinstance(close_price, (int, float)) or not isinstance(open_price, (int, float)):
                                continue
                                
                            if close_price > open_price:
                                high_price = df['high'].iloc[i]
                                low_price = df['low'].iloc[i]
                                
                                if not isinstance(high_price, (int, float)) or not isinstance(low_price, (int, float)):
                                    continue
                                    
                                self.bear_boxes.append({
                                    'start_idx': i,
                                    'top': high_price,
                                    'bot': low_price
                                })
                                self.last_trade_index = idx
                                current_position = -1  # Update current position to short
                                break

                # Check for bullish order block
                if (prev_pc < self.sensitivity and pc >= self.sensitivity and 
                    not any(idx - b['start_idx'] <= 5 for b in self.bull_boxes) and
                    current_position != 1):  # Only enter long if not already in a long position
                    
                    # Only add if trade conditions are met
                    if self.is_valid_trade_condition(df, idx, 'long'):
                        for i in range(idx-4, max(idx-16, -1), -1):
                            close_price = df['close'].iloc[i]
                            open_price = df['open'].iloc[i]
                            
                            if not isinstance(close_price, (int, float)) or not isinstance(open_price, (int, float)):
                                continue
                                
                            if close_price < open_price:
                                high_price = df['high'].iloc[i]
                                low_price = df['low'].iloc[i]
                                
                                if not isinstance(high_price, (int, float)) or not isinstance(low_price, (int, float)):
                                    continue
                                    
                                self.bull_boxes.append({
                                    'start_idx': i,
                                    'top': high_price,
                                    'bot': low_price
                                })
                                self.last_trade_index = idx
                                current_position = 1  # Update current position to long
                                break
                
                # Update position based on exit conditions
                if current_position != 0:
                    current_price = df['close'].iloc[idx]
                    
                    if not isinstance(current_price, (int, float)):
                        continue
                        
                    # Check for stop loss or opposite signal exit
                    if current_position == 1:  # Long position
                        stop_price = current_price * (1 - 0.02)  # 2% stop loss
                        if current_price <= stop_price or (prev_pc > -self.sensitivity and pc <= -self.sensitivity):
                            current_position = 0
                    else:  # Short position
                        stop_price = current_price * (1 + 0.02)  # 2% stop loss
                        if current_price >= stop_price or (prev_pc < self.sensitivity and pc >= self.sensitivity):
                            current_position = 0
            
            return df
            
        except Exception as e:
            logger.error(f"Error analyzing market: {str(e)}")
            return df

    def update_position(self, signal):
        """Update internal position tracking based on signal"""
        try:
            if signal['type'] == 'entry':
                self.current_position = 1 if signal['direction'] == 'long' else -1
                self.entry_price = signal['price']
                self.stop_loss_price = signal['stop_loss']
                self.trailing_stop_price = signal['trailing_stop']
                logger.info(f"Entered {signal['direction']} position at {signal['price']}")
            
            elif signal['type'] == 'exit':
                self.current_position = 0
                self.entry_price = None
                self.stop_loss_price = None
                self.trailing_stop_price = None
                logger.info(f"Exited position at {signal['price']} due to {signal['reason']}")
                
        except Exception as e:
            logger.error(f"Error updating position: {str(e)}")

def fetch_market_data(symbol=TRADING_CONFIG['symbol'], timeframe=TRADING_CONFIG['timeframe']):
    """Fetch market data from Delta Exchange"""
    try:
        # Calculate timestamps
        end_time = int(time.time())
        start_time = end_time - (60 * 24 * 60 * 60)  # Last 60 days
        
        timeframe_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        
        resolution = timeframe_map.get(timeframe, '1m')
        
        logger.info(f"Fetching {symbol} data for {timeframe} timeframe")
        
        response = requests.get(
            'https://api.india.delta.exchange/v2/history/candles',
            params={
                'resolution': resolution,
                'symbol': symbol,
                'start': str(start_time),
                'end': str(end_time)
            },
            headers={'Accept': 'application/json'}
        )
        
        response.raise_for_status()
        data = response.json()
        
        if not data or 'result' not in data:
            raise ValueError("No data received from API")
            
        df = pd.DataFrame(data['result'])
        df = df.rename(columns={
            'time': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.sort_values('timestamp')
        
        logger.info(f"Successfully fetched {len(df)} candles")
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        return pd.DataFrame() 