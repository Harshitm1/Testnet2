import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from strategy import OrderBlocks, fetch_market_data
from config import TRADING_CONFIG

def calculate_compounding_pnl(df, ob, initial_capital=100):
    """Calculate PnL with full compounding - each trade uses the entire current capital"""
    position = 0
    capital = initial_capital
    trades = []
    df = df.copy()  # Create a copy to avoid modifying the original DataFrame
    df['position'] = 0
    df['equity'] = initial_capital
    
    # Risk management parameters
    max_loss_per_trade = 0.02  # 2% max loss per trade
    trailing_stop = 0.015  # 1.5% trailing stop
    leverage = TRADING_CONFIG['leverage']  # Use configured leverage
    
    for i in range(1, len(df)):
        current_price = df['close'].iloc[i]
        
        # Update trailing stop if in position
        if position != 0:
            if position == 1:  # Long position
                # Calculate current position value
                current_value = position_size * current_price
                unrealized_pnl = current_value - (position_size * entry_price)
                
                # Update trailing stop
                trail_price = current_price * (1 - trailing_stop)
                if trail_price > trades[-1].get('stop_loss', 0):
                    trades[-1]['stop_loss'] = trail_price
                
                # Check for stop loss
                if current_price <= trades[-1]['stop_loss']:
                    exit_price = current_price
                    pnl = unrealized_pnl
                    capital += pnl
                    position = 0
                    trades[-1].update({
                        'exit_price': exit_price,
                        'exit_time': df['timestamp'].iloc[i],
                        'pnl': pnl,
                        'exit_capital': capital,
                        'exit_reason': 'stop_loss',
                        'return_pct': (pnl / trades[-1]['entry_capital']) * 100
                    })
                    
            else:  # Short position
                # Calculate current position value
                current_value = position_size * entry_price
                unrealized_pnl = current_value - (position_size * current_price)
                
                # Update trailing stop
                trail_price = current_price * (1 + trailing_stop)
                if trail_price < trades[-1].get('stop_loss', float('inf')):
                    trades[-1]['stop_loss'] = trail_price
                
                # Check for stop loss
                if current_price >= trades[-1]['stop_loss']:
                    exit_price = current_price
                    pnl = unrealized_pnl
                    capital += pnl
                    position = 0
                    trades[-1].update({
                        'exit_price': exit_price,
                        'exit_time': df['timestamp'].iloc[i],
                        'pnl': pnl,
                        'exit_capital': capital,
                        'exit_reason': 'stop_loss',
                        'return_pct': (pnl / trades[-1]['entry_capital']) * 100
                    })

        # Check for new trade entry or opposite signal exit
        for bull_box in ob.bull_boxes:
            if i == bull_box['start_idx']:
                if position == 0:  # New long entry
                    position = 1
                    entry_price = current_price
                    # Calculate position size using current capital
                    position_size = (capital * leverage) / entry_price
                    stop_loss = entry_price * (1 - max_loss_per_trade)
                    trades.append({
                        'type': 'long',
                        'entry_price': entry_price,
                        'entry_time': df['timestamp'].iloc[i],
                        'entry_capital': capital,
                        'position_size': position_size,
                        'stop_loss': stop_loss
                    })
                elif position == -1:  # Exit short on opposite signal
                    exit_price = current_price
                    # Calculate PnL based on position size
                    current_value = position_size * entry_price
                    pnl = current_value - (position_size * exit_price)
                    capital += pnl
                    position = 0
                    trades[-1].update({
                        'exit_price': exit_price,
                        'exit_time': df['timestamp'].iloc[i],
                        'pnl': pnl,
                        'exit_capital': capital,
                        'exit_reason': 'opposite_signal',
                        'return_pct': (pnl / trades[-1]['entry_capital']) * 100
                    })

        for bear_box in ob.bear_boxes:
            if i == bear_box['start_idx']:
                if position == 0:  # New short entry
                    position = -1
                    entry_price = current_price
                    # Calculate position size using current capital
                    position_size = (capital * leverage) / entry_price
                    stop_loss = entry_price * (1 + max_loss_per_trade)
                    trades.append({
                        'type': 'short',
                        'entry_price': entry_price,
                        'entry_time': df['timestamp'].iloc[i],
                        'entry_capital': capital,
                        'position_size': position_size,
                        'stop_loss': stop_loss
                    })
                elif position == 1:  # Exit long on opposite signal
                    exit_price = current_price
                    # Calculate PnL based on position size
                    current_value = position_size * exit_price
                    pnl = current_value - (position_size * entry_price)
                    capital += pnl
                    position = 0
                    trades[-1].update({
                        'exit_price': exit_price,
                        'exit_time': df['timestamp'].iloc[i],
                        'pnl': pnl,
                        'exit_capital': capital,
                        'exit_reason': 'opposite_signal',
                        'return_pct': (pnl / trades[-1]['entry_capital']) * 100
                    })

        # Update equity curve
        if position == 0:
            df.loc[df.index[i], 'equity'] = capital
        else:
            # Calculate unrealized PnL
            if position == 1:
                current_value = position_size * current_price
                unrealized_pnl = current_value - (position_size * entry_price)
            else:
                current_value = position_size * entry_price
                unrealized_pnl = current_value - (position_size * current_price)
            df.loc[df.index[i], 'equity'] = capital + unrealized_pnl
        
        df.loc[df.index[i], 'position'] = position

    # Close any open position at the end
    if position != 0:
        exit_price = df['close'].iloc[-1]
        if position == 1:
            current_value = position_size * exit_price
            pnl = current_value - (position_size * entry_price)
        else:
            current_value = position_size * entry_price
            pnl = current_value - (position_size * exit_price)
        capital += pnl
        trades[-1].update({
            'exit_price': exit_price,
            'exit_time': df['timestamp'].iloc[-1],
            'pnl': pnl,
            'exit_capital': capital,
            'exit_reason': 'end_of_data',
            'return_pct': (pnl / trades[-1]['entry_capital']) * 100
        })

    trades_df = pd.DataFrame(trades)
    return df, capital, trades_df

def print_compounding_stats(trades_df, initial_capital, final_capital):
    """Print detailed statistics for compounding returns"""
    if not trades_df.empty:
        completed_trades = trades_df[trades_df['pnl'].notna()]
        if not completed_trades.empty:
            win_rate = len(completed_trades[completed_trades['pnl'] > 0]) / len(completed_trades) * 100
            avg_win = completed_trades[completed_trades['pnl'] > 0]['pnl'].mean() if not completed_trades[completed_trades['pnl'] > 0].empty else 0
            avg_loss = completed_trades[completed_trades['pnl'] < 0]['pnl'].mean() if not completed_trades[completed_trades['pnl'] < 0].empty else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            print("\nCompounding Performance Metrics:")
            print(f"Initial Capital: ${initial_capital:.2f}")
            print(f"Final Capital: ${final_capital:.2f}")
            print(f"Total Return: {((final_capital - initial_capital) / initial_capital * 100):.2f}%")
            print(f"Number of Trades: {len(completed_trades)}")
            print(f"Win Rate: {win_rate:.1f}%")
            print(f"Average Win: ${avg_win:.2f}")
            print(f"Average Loss: ${avg_loss:.2f}")
            print(f"Profit Factor: {profit_factor:.2f}")
            print(f"Average Trade Return: {(completed_trades['return_pct'].mean()):.2f}%")
            print(f"Largest Win: ${completed_trades['pnl'].max():.2f}")
            print(f"Largest Loss: ${completed_trades['pnl'].min():.2f}")
            print(f"Average Capital Per Trade: ${completed_trades['entry_capital'].mean():.2f}")
            
            # Calculate drawdown metrics
            if 'cumulative_capital' in completed_trades.columns:
                rolling_max = completed_trades['cumulative_capital'].expanding().max()
                drawdown = (completed_trades['cumulative_capital'] - rolling_max) / rolling_max * 100
                max_drawdown = drawdown.min()
                print(f"Maximum Drawdown: {max_drawdown:.2f}%")
            
            # Print detailed trade list
            print("\nDetailed Trade List:")
            print("=" * 100)
            print(f"{'Trade #':<7} {'Type':<6} {'Entry Time':<20} {'Exit Time':<20} {'Entry':<10} {'Exit':<10} {'PnL':<10} {'Return %':<10}")
            print("-" * 100)
            
            for idx, trade in completed_trades.iterrows():
                print(f"{idx + 1:<7} {trade['type']:<6} {trade['entry_time'].strftime('%Y-%m-%d %H:%M'):<20} "
                      f"{trade['exit_time'].strftime('%Y-%m-%d %H:%M'):<20} "
                      f"${trade['entry_price']:<9.2f} ${trade['exit_price']:<9.2f} "
                      f"${trade['pnl']:<9.2f} {trade['return_pct']:<9.2f}%")
            print("=" * 100)

def plot_results(df, trades_df, ob, period):
    """Plot backtest results including price, order blocks, trades, and equity curve"""
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
    ax1, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    ax1.plot(df['timestamp'], df['close'], label='Price', color='blue', alpha=0.5)

    for b in ob.bear_boxes:
        ax1.axvspan(df['timestamp'].iloc[b['start_idx']], df['timestamp'].iloc[-1],
                    ymin=(b['bot'] - df['low'].min()) / (df['high'].max() - df['low'].min()),
                    ymax=(b['top'] - df['low'].min()) / (df['high'].max() - df['low'].min()),
                    color='#506CD3', alpha=0.33)
    for b in ob.bull_boxes:
        ax1.axvspan(df['timestamp'].iloc[b['start_idx']], df['timestamp'].iloc[-1],
                    ymin=(b['bot'] - df['low'].min()) / (df['high'].max() - df['low'].min()),
                    ymax=(b['top'] - df['low'].min()) / (df['high'].max() - df['low'].min()),
                    color='#64C4AC', alpha=0.33)

    if not trades_df.empty:
        ax1.scatter(trades_df[trades_df['type'] == 'long']['entry_time'],
                    trades_df[trades_df['type'] == 'long']['entry_price'],
                    marker='^', color='green', label='Long Entry', s=100)
        ax1.scatter(trades_df[trades_df['type'] == 'short']['entry_time'],
                    trades_df[trades_df['type'] == 'short']['entry_price'],
                    marker='v', color='red', label='Short Entry', s=100)

    ax1.set_title(f'ETH-USD Price and Order Blocks ({period})')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(df['timestamp'], df['equity'], label='Equity', color='purple', linewidth=2)
    ax2.set_title('Equity Curve')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Equity ($)')
    ax2.legend()
    ax2.grid(True)

    if not trades_df.empty and 'pnl' in trades_df.columns:
        completed = trades_df[trades_df['pnl'].notna()]
        if not completed.empty:
            ax3.hist(completed['pnl'], bins=20, color='purple', alpha=0.7)
            ax3.axvline(x=0, color='red', linestyle='--', label='Break Even')
            win_rate = len(completed[completed['pnl'] > 0]) / len(completed) * 100
            avg_win = completed[completed['pnl'] > 0]['pnl'].mean() if not completed[completed['pnl'] > 0].empty else 0
            avg_loss = completed[completed['pnl'] < 0]['pnl'].mean() if not completed[completed['pnl'] < 0].empty else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            stats_text = f'Win Rate: {win_rate:.1f}%\nAvg Win: ${avg_win:.2f}\nAvg Loss: ${avg_loss:.2f}\nProfit Factor: {profit_factor:.2f}'
            ax3.set_title('Trade PnL Distribution')
            ax3.set_xlabel('PnL ($)')
            ax3.set_ylabel('Frequency')
            ax3.text(0.02, 0.95, stats_text, transform=ax3.transAxes,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax3.legend()
            ax3.grid(True)

    plt.tight_layout()
    plt.savefig(f'ethusd_ob_results_{period}.png')
    plt.close()

if __name__ == "__main__":
    # Fetch ETH/USD data
    print("Fetching ETH/USD data...")
    df = fetch_market_data(timeframe='1m')
    
    # Initialize OrderBlocks with default parameters
    ob = OrderBlocks()
    
    # Find order blocks
    print("\nAnalyzing order blocks...")
    df = ob.find_order_blocks(df)
    
    # Calculate compounding PnL with initial capital of 100 USD
    print("\nCalculating compounding PnL...")
    initial_capital = TRADING_CONFIG['initial_capital']
    df, final_capital, trades_df = calculate_compounding_pnl(df, ob, initial_capital)
    
    # Print statistics
    print("\nTrading Statistics:")
    print_compounding_stats(trades_df, initial_capital, final_capital)
    
    # Plot results
    print("\nPlotting results...")
    plot_results(df, trades_df, ob, '1m') 