#!/usr/bin/env python3
"""
Download de dados históricos do Yahoo Finance para ouro
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import logging
import os
from datetime import datetime, timedelta
import pickle

def setup_logging():
    """Setup logging"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"yahoo_download_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def download_yahoo_data(symbol: str, start_date: str = "2010-01-01", end_date: str = None) -> pd.DataFrame:
    """Download DAILY data from Yahoo Finance and resample to 5m"""
    logger = setup_logging()
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"Downloading DAILY data for {symbol} from Yahoo Finance")
    logger.info(f"Period: {start_date} to {end_date}")
    
    try:
        # Download DAILY data (histórico completo disponível)
        ticker = yf.Ticker(symbol)
        df_daily = ticker.history(start=start_date, end=end_date, interval="1d")
        
        if df_daily.empty:
            logger.error(f"No daily data downloaded for {symbol}")
            return pd.DataFrame()
        
        logger.info(f"Downloaded {len(df_daily)} daily bars")
        
        # Reset index to get datetime as column
        df_daily = df_daily.reset_index()
        
        # Convert Date to datetime if needed
        if 'Date' in df_daily.columns:
            df_daily['Date'] = pd.to_datetime(df_daily['Date'])
        
        # Create 5-minute resampled data from daily data
        logger.info("Creating 5-minute resampled data...")
        resampled_data = []
        
        for _, row in df_daily.iterrows():
            date = row['Date'] if 'Date' in df_daily.columns else row.name
            open_price = row['Open']
            high_price = row['High']
            low_price = row['Low']
            close_price = row['Close']
            volume = row['Volume']
            
            # Create 5-minute intervals for this day (288 intervals per day)
            start_time = pd.Timestamp(date).replace(hour=0, minute=0)
            
            for i in range(288):  # 24 hours * 60 minutes / 5 minutes = 288
                time_5m = start_time + pd.Timedelta(minutes=i * 5)
                
                # Simulate realistic price movement within the day
                # Use random walk between OHLC values
                if i == 0:
                    bar_open = open_price
                elif i == 287:  # Last bar
                    bar_open = resampled_data[-1]['close']
                    bar_close = close_price
                else:
                    # Random walk between daily high/low
                    bar_open = resampled_data[-1]['close'] if resampled_data else open_price
                
                if i != 287:
                    # Random price movement within daily range
                    daily_range = high_price - low_price
                    price_change = np.random.uniform(-daily_range * 0.02, daily_range * 0.02)
                    bar_close = max(low_price, min(high_price, bar_open + price_change))
                
                # High and low for this 5m bar
                bar_high = max(bar_open, bar_close) + abs(np.random.uniform(0, daily_range * 0.01))
                bar_low = min(bar_open, bar_close) - abs(np.random.uniform(0, daily_range * 0.01))
                
                # Ensure within daily bounds
                bar_high = min(bar_high, high_price)
                bar_low = max(bar_low, low_price)
                
                # Volume distributed across the day
                bar_volume = volume / 288
                
                resampled_data.append({
                    'time': time_5m,
                    'open': bar_open,
                    'high': bar_high,
                    'low': bar_low,
                    'close': bar_close,
                    'tick_volume': bar_volume,
                    'spread': 0,
                    'real_volume': bar_volume
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(resampled_data)
        
        # Convert timezone to UTC if needed
        if df['time'].dt.tz is not None:
            df['time'] = df['time'].dt.tz_localize(None)
        
        logger.info(f"Resampled to {len(df)} 5-minute bars")
        logger.info(f"Data range: {df['time'].min()} to {df['time'].max()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error downloading {symbol}: {str(e)}")
        return pd.DataFrame()

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators"""
    logger = setup_logging()
    logger.info("Calculating technical indicators...")
    
    # Basic price indicators
    df['returns'] = df['close'].pct_change().fillna(0)
    df['volatility_20'] = df['close'].pct_change().rolling(window=20).std().fillna(0)
    df['sma_20'] = df['close'].rolling(window=20).mean().fillna(df['close'])
    df['sma_50'] = df['close'].rolling(window=50).mean().fillna(df['close'])
    
    # RSI
    try:
        df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi().fillna(50)
    except:
        df['rsi_14'] = 50.0
    
    # Bollinger Bands Position (0-1)
    bb_sma = df['close'].rolling(20).mean().fillna(df['close'])
    bb_std = df['close'].rolling(20).std().fillna(0.01)
    bb_upper = bb_sma + (bb_std * 2)
    bb_lower = bb_sma - (bb_std * 2)
    df['bb_position'] = ((df['close'] - bb_lower) / (bb_upper - bb_lower)).fillna(0.5).clip(0, 1)
    
    # Trend Strength
    returns = df['close'].pct_change().fillna(0)
    df['trend_strength'] = returns.rolling(10).mean().fillna(0)
    
    # ATR
    try:
        df['atr_14'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range().fillna(0)
    except:
        df['atr_14'] = (df['high'] - df['low']).rolling(14).mean().fillna(0)
    
    # Stochastic (simplified)
    try:
        high_14 = df['high'].rolling(14).max()
        low_14 = df['low'].rolling(14).min()
        df['stoch_k'] = ((df['close'] - low_14) / (high_14 - low_14)) * 100
        df['stoch_k'] = df['stoch_k'].fillna(50).clip(0, 100)
    except:
        df['stoch_k'] = 50.0
    
    # Volume ratio
    volume_sma = df['tick_volume'].rolling(20).mean().fillna(1)
    df['volume_ratio'] = (df['tick_volume'] / volume_sma).fillna(1.0).clip(0.1, 5.0)
    
    # VaR 99%
    df['var_99'] = df['close'].rolling(window=20).quantile(0.01).fillna(df['close'] * 0.95)
    
    logger.info("Technical indicators calculated successfully")
    return df

def filter_static_points(df: pd.DataFrame, min_price_change: float = 0.01) -> pd.DataFrame:
    """Filter out static price points"""
    logger = setup_logging()
    logger.info("Filtering static price points...")
    
    initial_rows = len(df)
    
    # Calculate price changes
    df['price_change'] = abs(df['close'] - df['close'].shift(1))
    
    # Filter rows with minimal price change
    df_filtered = df[df['price_change'] >= min_price_change].copy()
    
    # Keep first row
    first_row = df.iloc[0:1].copy()
    first_row['price_change'] = 0
    
    # Combine first row with filtered data
    df_final = pd.concat([first_row, df_filtered], ignore_index=True)
    df_final = df_final.sort_values('time').reset_index(drop=True)
    
    # Remove temporary column
    df_final = df_final.drop('price_change', axis=1)
    
    removed_rows = initial_rows - len(df_final)
    logger.info(f"Removed {removed_rows} static points ({removed_rows/initial_rows*100:.1f}%)")
    logger.info(f"Final dataset: {len(df_final)} rows")
    
    return df_final

def save_optimized_cache(df: pd.DataFrame, symbol: str, timestamp: str) -> str:
    """Save data as optimized pickle cache"""
    cache_dir = "data_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    cache_filename = os.path.join(cache_dir, f"{symbol}_YAHOO_CACHE_{timestamp}.pkl")
    
    logger = setup_logging()
    logger.info(f"Saving optimized cache: {cache_filename}")
    
    # Optimize data types for memory efficiency
    df_optimized = df.copy()
    
    # Convert time to datetime64 for efficiency
    df_optimized['time'] = pd.to_datetime(df_optimized['time'])
    
    # Optimize numeric columns
    for col in df_optimized.select_dtypes(include=['float64']).columns:
        df_optimized[col] = df_optimized[col].astype('float32')
    
    for col in df_optimized.select_dtypes(include=['int64']).columns:
        df_optimized[col] = df_optimized[col].astype('int32')
    
    # Save with pickle
    with open(cache_filename, 'wb') as f:
        pickle.dump(df_optimized, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Calculate file size
    file_size_mb = os.path.getsize(cache_filename) / (1024 * 1024)
    logger.info(f"Cache saved: {file_size_mb:.2f} MB")
    
    return cache_filename

def main():
    """Main function"""
    logger = setup_logging()
    
    # Gold symbols to try
    symbols = [
        "GC=F",      # Gold Futures
        "XAUUSD=X",  # Gold vs USD
        "GLD",       # SPDR Gold Trust ETF
        "IAU",       # iShares Gold Trust ETF
        "SGOL"       # Aberdeen Standard Physical Gold ETF
    ]
    
    logger.info("Starting Yahoo Finance download for gold data")
    
    for symbol in symbols:
        logger.info(f"\nTrying symbol: {symbol}")
        
        # Download data with extended historical period
        df = download_yahoo_data(symbol, start_date="2010-01-01")
        
        if not df.empty:
            logger.info(f"Successfully downloaded {symbol}")
            
            # Calculate indicators
            df = calculate_indicators(df)
            
            # Filter static points
            df = filter_static_points(df, min_price_change=0.01)
            
            # Save data
            data_dir = "data"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save CSV
            csv_filename = os.path.join(data_dir, f"{symbol}_YAHOO_{timestamp}.csv")
            df.to_csv(csv_filename, index=False)
            logger.info(f"Saved CSV: {csv_filename}")
            
            # Save optimized cache
            cache_filename = save_optimized_cache(df, symbol, timestamp)
            
            # Summary
            logger.info(f"DOWNLOAD COMPLETE for {symbol}!")
            logger.info(f"Total rows: {len(df)}")
            logger.info(f"Data range: {df['time'].min()} to {df['time'].max()}")
            logger.info(f"Cache saved: {cache_filename}")
            
            # Only process the first successful symbol
            break
        else:
            logger.warning(f"Failed to download {symbol}, trying next...")
    
    if df.empty:
        logger.error("Failed to download any symbol!")
    else:
        logger.info("Yahoo Finance download completed successfully!")

if __name__ == "__main__":
    main() 