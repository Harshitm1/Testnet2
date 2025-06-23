import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategy import OrderBlocks
from delta_rest_client import DeltaRestClient
from config import TRADING_CONFIG, API_CONFIG
from logger import setup_logger
import traceback
from telegram_bot import TelegramBot, send_telegram_message
import requests
import matplotlib.pyplot as plt

logger = setup_logger('trading_bot')

class TradingBot:
    def __init__(self):
        # Initialize Delta Exchange client
        self.client = DeltaRestClient(
            base_url=API_CONFIG['base_url'],
            api_key=API_CONFIG['api_key'],
            api_secret=API_CONFIG['api_secret']
        )
        
        # Initialize strategy
        self.strategy = OrderBlocks(
            sensitivity=TRADING_CONFIG['sensitivity'],
            ob_mitigation='Close',
            min_volume_percentile=TRADING_CONFIG['min_volume_percentile'],
            trend_period=TRADING_CONFIG['trend_period'],
            min_trades_distance=TRADING_CONFIG['min_trades_distance']
        )
        
        # Store product ID
        self.product_id = TRADING_CONFIG['product_id']
        logger.info(f"Successfully initialized trading for product ID: {self.product_id}")
        
        # Send Telegram notification on bot start
        start_msg = (
            f"🚀 Trading Bot Started!\n"
            f"Product ID: {self.product_id}\n"
            f"Initial Capital: ${TRADING_CONFIG['initial_capital']}\n"
            f"Leverage: {TRADING_CONFIG['leverage']}x\n"
            f"Order Block Strategy Params:\n"
            f"- Candle TF: 1m\n"
            f"- Price Move Threshold: {TRADING_CONFIG['sensitivity']*100:.2f}%\n"
            f"- Min Volume Percentile: {TRADING_CONFIG['min_volume_percentile']}\n"
            f"- SMA Trend: 20/50\n"
            f"- Min Candles Between Trades: {TRADING_CONFIG['min_trades_distance']}\n"
            f"- Stop Loss: {TRADING_CONFIG['stop_loss_pct']*100:.2f}%\n"
            f"- Trailing Stop: {TRADING_CONFIG['trailing_stop_pct']*100:.2f}%\n"
        )
        send_telegram_message(start_msg)
        
        # Trading state
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = None
        self.stop_loss = None
        self.trailing_stop = None
        self.position_size = 0
        self.capital = TRADING_CONFIG['initial_capital']
        
        # Performance tracking
        self.trades = []
        self.equity = []
        
        # Initialize TelegramBot
        self.telegram = TelegramBot()
        
    def rate_limit_aware_call(self, func, *args, **kwargs):
        """Call an API function, handle 429 errors and sleep as needed."""
        while True:
            try:
                return func(*args, **kwargs)
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    reset_ms = int(e.response.headers.get('X-RATE-LIMIT-RESET', '1000'))
                    reset_sec = max(reset_ms / 1000, 1)
                    logger.warning(f"Rate limit hit. Sleeping for {reset_sec:.1f} seconds.")
                    time.sleep(reset_sec)
                else:
                    raise

    def fetch_latest_data(self):
        """Fetch historical 1m candles for the strategy and latest ticker for current price."""
        try:
            candles = self.rate_limit_aware_call(
                self.client.get_history_candles,
                TRADING_CONFIG['symbol'],
                resolution=TRADING_CONFIG['timeframe'],
                limit=200
            )
            if not candles or len(candles) == 0:
                raise ValueError("No candle data received from API")
            print('DEBUG CANDLES:', candles[:2])  # Print first two for inspection
            df = pd.DataFrame(candles)
            # Use the correct key for timestamp
            if 'time' in df.columns:
                df = df.rename(columns={'time': 'timestamp'})
            # Standardize column names
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.sort_values('timestamp').reset_index(drop=True)
            logger.info(f"Fetched {len(df)} candles for strategy.")
            return df
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            return None
            
    def place_order(self, side, size, price=None, order_type='market'):
        """Place an order on Delta Exchange"""
        try:
            # Calculate actual size based on position_size and leverage
            actual_size = size * TRADING_CONFIG['leverage']
            
            response = self.client.place_order(
                symbol=TRADING_CONFIG['symbol'],
                side=side,
                size=actual_size,
                price=price,
                order_type=order_type
            )
            
            if response and 'result' in response:
                logger.info(f"Successfully placed {side} order for {actual_size} contracts at {price}")
                return response['result']
            else:
                logger.error(f"Failed to place order: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None
            
    def update_position(self, current_price):
        """Update position and check for exits"""
        if self.position != 0:
            try:
                # Calculate unrealized PnL
                if self.position == 1:  # Long position
                    unrealized_pnl = (current_price - self.entry_price) * self.position_size
                    # Update trailing stop
                    new_trailing_stop = current_price * (1 - TRADING_CONFIG['trailing_stop_pct'])
                    if new_trailing_stop > self.trailing_stop:
                        self.trailing_stop = new_trailing_stop
                        logger.info(f"Updated long trailing stop to ${self.trailing_stop:.2f}")
                else:  # Short position
                    unrealized_pnl = (self.entry_price - current_price) * self.position_size
                    # Update trailing stop
                    new_trailing_stop = current_price * (1 + TRADING_CONFIG['trailing_stop_pct'])
                    if new_trailing_stop < self.trailing_stop:
                        self.trailing_stop = new_trailing_stop
                        logger.info(f"Updated short trailing stop to ${self.trailing_stop:.2f}")
                
                # Check for stop loss or trailing stop hit
                if ((self.position == 1 and current_price <= self.trailing_stop) or 
                    (self.position == -1 and current_price >= self.trailing_stop)):
                    # Close position
                    order = {
                        'product_id': self.product_id,
                        'size': self.position_size,
                        'side': 'sell' if self.position == 1 else 'buy',
                        'order_type': 'market_order',
                        'reduce_only': 'true'
                    }
                    if self.client.create_order(order):
                        # Update capital and record trade
                        self.capital += unrealized_pnl
                        self.trades.append({
                            'exit_price': current_price,
                            'exit_time': datetime.now(),
                            'pnl': unrealized_pnl,
                            'exit_reason': 'trailing_stop'
                        })
                        logger.info(f"Closed position at ${current_price:.2f}, PnL: ${unrealized_pnl:.2f}")
                        # Reset position
                        self.position = 0
                        self.entry_price = None
                        self.stop_loss = None
                        self.trailing_stop = None
                        self.position_size = 0
                        
            except Exception as e:
                logger.error(f"Error updating position: {str(e)}")
            
    def execute_trades(self, df):
        """Execute trades based on strategy signals"""
        try:
            # Calculate indicators and order blocks
            df = self.strategy.find_order_blocks(df)
            # Get current price from the latest data
            current_price = float(df['close'].iloc[-1])
            
            # Update existing position if any
            self.update_position(current_price)
            
            # Get order book to check liquidity
            orderbook = self.client.get_l2_orderbook(str(self.product_id))
            print('DEBUG ORDERBOOK:', orderbook)  # Debug print
            if not orderbook:
                logger.error("Could not fetch orderbook")
                return
                
            # Calculate available liquidity using correct key
            def get_size(level):
                if isinstance(level, dict):
                    return float(level.get('size', 0))
                elif isinstance(level, (list, tuple)) and len(level) > 1:
                    return float(level[1])
                return 0
            bid_liquidity = sum([get_size(level) for level in orderbook['buy']])
            ask_liquidity = sum([get_size(level) for level in orderbook['sell']])
            logger.info(f"Bid liquidity: {bid_liquidity}, Ask liquidity: {ask_liquidity}")
            
            # Only trade if there's sufficient liquidity
            min_liquidity = self.capital * 10  # Require 10x our position size in liquidity
            
            if self.position == 0:  # Only enter if not in a position
                # Calculate position size (in contracts)
                position_size = self.capital / current_price
                
                # Check for long entry
                if (bid_liquidity > min_liquidity and 
                    self.strategy.is_valid_trade_condition(df, len(df)-1, 'long')):
                    # Enter long position
                    order = {
                        'product_id': self.product_id,
                        'size': int(position_size),
                        'side': 'buy',
                        'order_type': 'market_order'
                    }
                    if self.client.create_order(order):
                        self.position = 1
                        self.entry_price = current_price
                        self.position_size = position_size
                        self.stop_loss = current_price * (1 - TRADING_CONFIG['stop_loss_pct'])
                        self.trailing_stop = current_price * (1 - TRADING_CONFIG['trailing_stop_pct'])
                        self.trades.append({
                            'type': 'long',
                            'entry_price': current_price,
                            'entry_time': datetime.now(),
                            'position_size': self.position_size,
                            'stop_loss': self.stop_loss,
                            'trailing_stop': self.trailing_stop
                        })
                        logger.info(f"Entered long position at ${current_price:.2f}")
                        self.telegram.send_trade_alert('buy', TRADING_CONFIG['symbol'], current_price, int(position_size))
                
                # Check for short entry
                elif (ask_liquidity > min_liquidity and 
                      self.strategy.is_valid_trade_condition(df, len(df)-1, 'short')):
                    # Enter short position
                    order = {
                        'product_id': self.product_id,
                        'size': int(position_size),
                        'side': 'sell',
                        'order_type': 'market_order'
                    }
                    if self.client.create_order(order):
                        self.position = -1
                        self.entry_price = current_price
                        self.position_size = position_size
                        self.stop_loss = current_price * (1 + TRADING_CONFIG['stop_loss_pct'])
                        self.trailing_stop = current_price * (1 + TRADING_CONFIG['trailing_stop_pct'])
                        self.trades.append({
                            'type': 'short',
                            'entry_price': current_price,
                            'entry_time': datetime.now(),
                            'position_size': self.position_size,
                            'stop_loss': self.stop_loss,
                            'trailing_stop': self.trailing_stop
                        })
                        logger.info(f"Entered short position at ${current_price:.2f}")
                        self.telegram.send_trade_alert('sell', TRADING_CONFIG['symbol'], current_price, int(position_size))
            
            # Update equity curve
            equity = self.capital
            if self.position != 0:
                if self.position == 1:
                    unrealized_pnl = (current_price - self.entry_price) * self.position_size
                else:
                    unrealized_pnl = (self.entry_price - current_price) * self.position_size
                equity += unrealized_pnl
            self.equity.append({
                'timestamp': datetime.now(),
                'equity': equity
            })
            
        except Exception as e:
            logger.error(f"Error executing trades: {str(e)}\n{traceback.format_exc()}")
            self.telegram.send_error_alert(str(e))
            
    def print_wallet_balance(self):
        try:
            for asset_id in ['USDT', 'ETH']:
                balance = self.rate_limit_aware_call(self.client.get_balances, asset_id)
                print(f'WALLET DEBUG RESPONSE for {asset_id}:', balance)
                if balance:
                    print(f"Wallet Balance ({asset_id}): {balance['balance']}")
                else:
                    print(f"Could not fetch wallet balance for {asset_id}")
        except Exception as e:
            print(f"Error fetching wallet balance: {e}")
            
    def run(self):
        """Main trading loop - match backtest.py logic exactly for trading."""
        logger.info("Starting trading bot...")
        last_processed_time = None
        position = 0
        entry_price = None
        position_size = 0
        stop_loss = None
        trailing_stop = None
        capital = self.capital
        leverage = TRADING_CONFIG['leverage']
        while True:
            try:
                self.print_wallet_balance()
                df = self.fetch_latest_data()
                if df is not None and not df.empty:
                    if last_processed_time is not None:
                        new_candles = df[df['timestamp'] > last_processed_time]
                    else:
                        new_candles = df[-2:]
                    if not new_candles.empty:
                        df = self.strategy.find_order_blocks(df)
                        for i in new_candles.index:
                            current_price = df['close'].iloc[i]
                            timestamp = df['timestamp'].iloc[i]
                            # Print latest indicator values
                            print(f"\nCandle @ {timestamp} | O:{df['open'].iloc[i]} H:{df['high'].iloc[i]} L:{df['low'].iloc[i]} C:{current_price} V:{df['volume'].iloc[i]}")
                            print(f"Indicators: pc={df['pc'].iloc[i]:.4f}, vol_pct={df['volume_percentile'].iloc[i]:.2f}, sma20={df['sma20'].iloc[i]:.2f}, sma50={df['sma50'].iloc[i]:.2f}, roc={df['roc'].iloc[i]:.2f}, atr={df['atr'].iloc[i]:.2f}")
                            # Print order block status
                            bull_blocks = [b['start_idx'] for b in self.strategy.bull_boxes]
                            bear_blocks = [b['start_idx'] for b in self.strategy.bear_boxes]
                            print(f"Bull OBs: {bull_blocks[-5:]}")
                            print(f"Bear OBs: {bear_blocks[-5:]}")
                            # --- Exit logic ---
                            if position != 0:
                                if position == 1:
                                    trail_price = current_price * (1 - TRADING_CONFIG['trailing_stop_pct'])
                                    if trailing_stop is None or trail_price > trailing_stop:
                                        trailing_stop = trail_price
                                    if current_price <= trailing_stop:
                                        order = {
                                            'product_id': self.product_id,
                                            'size': int(position_size),
                                            'side': 'sell',
                                            'order_type': 'market_order',
                                            'reduce_only': 'true'
                                        }
                                        self.rate_limit_aware_call(self.client.create_order, order)
                                        self.telegram.send_trade_alert('sell', TRADING_CONFIG['symbol'], current_price, int(position_size))
                                        pnl = (current_price - entry_price) * position_size
                                        capital += pnl
                                        logger.info(f"Closed LONG at {current_price}, PnL: {pnl}")
                                        position = 0
                                        entry_price = None
                                        position_size = 0
                                        stop_loss = None
                                        trailing_stop = None
                                else:
                                    trail_price = current_price * (1 + TRADING_CONFIG['trailing_stop_pct'])
                                    if trailing_stop is None or trail_price < trailing_stop:
                                        trailing_stop = trail_price
                                    if current_price >= trailing_stop:
                                        order = {
                                            'product_id': self.product_id,
                                            'size': int(position_size),
                                            'side': 'buy',
                                            'order_type': 'market_order',
                                            'reduce_only': 'true'
                                        }
                                        self.rate_limit_aware_call(self.client.create_order, order)
                                        self.telegram.send_trade_alert('buy', TRADING_CONFIG['symbol'], current_price, int(position_size))
                                        pnl = (entry_price - current_price) * position_size
                                        capital += pnl
                                        logger.info(f"Closed SHORT at {current_price}, PnL: {pnl}")
                                        position = 0
                                        entry_price = None
                                        position_size = 0
                                        stop_loss = None
                                        trailing_stop = None
                            for bull_box in self.strategy.bull_boxes:
                                if i == bull_box['start_idx']:
                                    print(f"Bull OB detected at {i} (candle {timestamp})")
                                    if position == 0:
                                        position = 1
                                        entry_price = current_price
                                        position_size = (capital * leverage) / entry_price
                                        stop_loss = entry_price * (1 - TRADING_CONFIG['stop_loss_pct'])
                                        trailing_stop = entry_price * (1 - TRADING_CONFIG['trailing_stop_pct'])
                                        order = {
                                            'product_id': self.product_id,
                                            'size': int(position_size),
                                            'side': 'buy',
                                            'order_type': 'market_order'
                                        }
                                        self.rate_limit_aware_call(self.client.create_order, order)
                                        self.telegram.send_trade_alert('buy', TRADING_CONFIG['symbol'], current_price, int(position_size))
                                        logger.info(f"Opened LONG at {current_price}")
                                    elif position == -1:
                                        order = {
                                            'product_id': self.product_id,
                                            'size': int(position_size),
                                            'side': 'buy',
                                            'order_type': 'market_order',
                                            'reduce_only': 'true'
                                        }
                                        self.rate_limit_aware_call(self.client.create_order, order)
                                        self.telegram.send_trade_alert('buy', TRADING_CONFIG['symbol'], current_price, int(position_size))
                                        pnl = (entry_price - current_price) * position_size
                                        capital += pnl
                                        logger.info(f"Closed SHORT at {current_price}, PnL: {pnl}")
                                        position = 0
                                        entry_price = None
                                        position_size = 0
                                        stop_loss = None
                                        trailing_stop = None
                            for bear_box in self.strategy.bear_boxes:
                                if i == bear_box['start_idx']:
                                    print(f"Bear OB detected at {i} (candle {timestamp})")
                                    if position == 0:
                                        position = -1
                                        entry_price = current_price
                                        position_size = (capital * leverage) / entry_price
                                        stop_loss = entry_price * (1 + TRADING_CONFIG['stop_loss_pct'])
                                        trailing_stop = entry_price * (1 + TRADING_CONFIG['trailing_stop_pct'])
                                        order = {
                                            'product_id': self.product_id,
                                            'size': int(position_size),
                                            'side': 'sell',
                                            'order_type': 'market_order'
                                        }
                                        self.rate_limit_aware_call(self.client.create_order, order)
                                        self.telegram.send_trade_alert('sell', TRADING_CONFIG['symbol'], current_price, int(position_size))
                                        logger.info(f"Opened SHORT at {current_price}")
                                    elif position == 1:
                                        order = {
                                            'product_id': self.product_id,
                                            'size': int(position_size),
                                            'side': 'sell',
                                            'order_type': 'market_order',
                                            'reduce_only': 'true'
                                        }
                                        self.rate_limit_aware_call(self.client.create_order, order)
                                        self.telegram.send_trade_alert('sell', TRADING_CONFIG['symbol'], current_price, int(position_size))
                                        pnl = (current_price - entry_price) * position_size
                                        capital += pnl
                                        logger.info(f"Closed LONG at {current_price}, PnL: {pnl}")
                                        position = 0
                                        entry_price = None
                                        position_size = 0
                                        stop_loss = None
                                        trailing_stop = None
                        last_processed_time = new_candles['timestamp'].max()
                        # Update equity curve for plotting
                        self.equity.append({'timestamp': timestamp, 'equity': capital})
                    else:
                        logger.info("No new candles to process.")
                    # Plot after each loop
                    self.plot_live_results(df, self.strategy.bull_boxes, self.strategy.bear_boxes, self.equity)
                print("\nTrading Status:")
                print(f"Current Capital: ${capital:.2f}")
                print(f"Position: {'Long' if position == 1 else 'Short' if position == -1 else 'None'}")
                if position != 0:
                    print(f"Entry Price: ${entry_price:.2f}")
                    print(f"Current Price: ${df['close'].iloc[-1]:.2f}")
                    print(f"Stop Loss: ${stop_loss:.2f}")
                    print(f"Trailing Stop: ${trailing_stop:.2f}")
                time.sleep(60)
            except KeyboardInterrupt:
                logger.info("Stopping trading bot...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                time.sleep(60)

    def test_proxy_order(self):
        print("\n--- Proxy Buy Order Test ---")
        order = {
            'product_id': self.product_id,
            'size': 1,
            'side': 'buy',
            'order_type': 'market_order'
        }
        result = self.client.create_order(order)
        print("Buy order result:", result)
        self.telegram.send_trade_alert('buy', TRADING_CONFIG['symbol'], 'TEST', 1)
        print("\n--- Proxy Sell Order Test ---")
        order = {
            'product_id': self.product_id,
            'size': 1,
            'side': 'sell',
            'order_type': 'market_order'
        }
        result = self.client.create_order(order)
        print("Sell order result:", result)
        self.telegram.send_trade_alert('sell', TRADING_CONFIG['symbol'], 'TEST', 1)

    def plot_live_results(self, df, bull_boxes, bear_boxes, equity_curve):
        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
        ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        ax1.plot(df['timestamp'], df['close'], label='Price', color='blue', alpha=0.5)
        for b in bear_boxes:
            ax1.axvspan(df['timestamp'].iloc[b['start_idx']], df['timestamp'].iloc[-1],
                        ymin=(b['bot'] - df['low'].min()) / (df['high'].max() - df['low'].min()),
                        ymax=(b['top'] - df['low'].min()) / (df['high'].max() - df['low'].min()),
                        color='#506CD3', alpha=0.33)
        for b in bull_boxes:
            ax1.axvspan(df['timestamp'].iloc[b['start_idx']], df['timestamp'].iloc[-1],
                        ymin=(b['bot'] - df['low'].min()) / (df['high'].max() - df['low'].min()),
                        ymax=(b['top'] - df['low'].min()) / (df['high'].max() - df['low'].min()),
                        color='#64C4AC', alpha=0.33)
        ax1.set_title('ETH-USD Price and Order Blocks (Live)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        # Equity curve
        if equity_curve:
            times = [e['timestamp'] for e in equity_curve]
            values = [e['equity'] for e in equity_curve]
            ax2.plot(times, values, label='Equity', color='purple', linewidth=2)
        ax2.set_title('Equity Curve (Live)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Equity ($)')
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        plt.savefig('live_trading_ob_equity.png')
        plt.close()

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
    # bot.test_proxy_order() 
