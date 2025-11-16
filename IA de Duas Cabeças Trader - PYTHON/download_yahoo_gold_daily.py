#!/usr/bin/env python3
"""
Download de dados históricos DIÁRIOS do Yahoo Finance para ouro
e conversão para 5 minutos via resample
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
    log_file = os.path.join(log_dir, f"yahoo_daily_download_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def download_yahoo_daily_data(symbol: str, start_date: str = "1990-01-01", end_date: str = None) -> pd.DataFrame:
    """Download daily data from Yahoo Finance"""
    logger = setup_logging()
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"Downloading {symbol} daily data from Yahoo Finance")
    logger.info(f"Period: {start_date} to {end_date}")
    
    try:
        # Download daily data
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval="1d")
        
        if df.empty:
            logger.error(f"No data downloaded for {symbol}")
            return pd.DataFrame()
        
        # Reset index to get datetime as column
        df = df.reset_index()
        
        # Rename columns to match MT5 format
        df = df.rename(columns={
            'Date': 'time',
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'tick_volume'
        })
        
        # Add missing columns
        df['spread'] = 0  # Yahoo doesn't provide spread
        df['real_volume'] = df['tick_volume']
        
        # Convert timezone to UTC if needed
        if df['time'].dt.tz is not None:
            df['time'] = df['time'].dt.tz_localize(None)
        
        logger.info(f"Downloaded {len(df)} daily rows")
        logger.info(f"Data range: {df['time'].min()} to {df['time'].max()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error downloading {symbol}: {str(e)}")
        return pd.DataFrame()

def resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily data to 5-minute intervals"""
    logger = setup_logging()
    logger.info("Resampling daily data to 5-minute intervals...")
    
    # Set time as index
    df = df.set_index('time')
    
    # Create 5-minute resampled data
    df_5min = pd.DataFrame()
    
    for _, row in df.iterrows():
        # Create 5-minute intervals for this day (assuming 24-hour market for simplicity)
        day_start = row.name.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        # Create 5-minute intervals (288 bars per day)
        time_range = pd.date_range(start=day_start, end=day_end, freq='5min')[:-1]  # Exclude last to avoid overlap
        
        # Create OHLC data for each 5-minute interval
        for time in time_range:
            # Add some noise to make it more realistic
            noise_factor = 0.0001  # 0.01% noise
            
            open_price = row['open'] * (1 + np.random.normal(0, noise_factor))
            high_price = max(row['high'], open_price) * (1 + np.random.normal(0, noise_factor))
            low_price = min(row['low'], open_price) * (1 + np.random.normal(0, noise_factor))
            close_price = row['close'] * (1 + np.random.normal(0, noise_factor))
            
            # Ensure OHLC relationship
            high_price = max(open_price, high_price, close_price)
            low_price = min(open_price, low_price, close_price)
            
            df_5min = pd.concat([df_5min, pd.DataFrame({
                'open': [open_price],
                'high': [high_price],
                'low': [low_price],
                'close': [close_price],
                'tick_volume': [row['tick_volume'] / 288],  # Distribute volume across 288 5-min bars
                'spread': [row['spread']],
                'real_volume': [row['real_volume'] / 288]
            }, index=[time])])
    
    # Reset index to get time as column
    df_5min = df_5min.reset_index()
    df_5min = df_5min.rename(columns={'index': 'time'})
    
    logger.info(f"Resampled to {len(df_5min)} 5-minute bars")
    return df_5min

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
    
    cache_filename = os.path.join(cache_dir, f"{symbol}_YAHOO_DAILY_CACHE_{timestamp}.pkl")
    
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
    
    # Gold symbols to try (daily data)
    symbols = [
        "GC=F",      # Gold Futures
        "GLD",       # SPDR Gold Trust ETF
        "IAU",       # iShares Gold Trust ETF
        "SGOL",      # Aberdeen Standard Physical Gold ETF
        "XAUUSD=X"   # Gold vs USD (try again)
    ]
    
    logger.info("Starting Yahoo Finance daily download for gold data")
    
    for symbol in symbols:
        logger.info(f"\nTrying symbol: {symbol}")
        
        # Download daily data
        df = download_yahoo_daily_data(symbol, start_date="2010-01-01")
        
        if not df.empty:
            logger.info(f"Successfully downloaded {symbol} daily data")
            
            # Resample to 5-minute intervals
            df_5min = resample_to_5min(df)
            
            # Calculate indicators
            df_5min = calculate_indicators(df_5min)
            
            # Filter static points
            df_5min = filter_static_points(df_5min, min_price_change=0.001)  # Lower threshold for resampled data
            
            # Save data
            data_dir = "data"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save CSV
            csv_filename = os.path.join(data_dir, f"{symbol}_YAHOO_DAILY_5MIN_{timestamp}.csv")
            df_5min.to_csv(csv_filename, index=False)
            logger.info(f"Saved CSV: {csv_filename}")
            
            # Save optimized cache
            cache_filename = save_optimized_cache(df_5min, symbol, timestamp)
            
            # Summary
            logger.info(f"DOWNLOAD COMPLETE for {symbol}!")
            logger.info(f"Original daily rows: {len(df)}")
            logger.info(f"Resampled 5-min rows: {len(df_5min)}")
            logger.info(f"Data range: {df_5min['time'].min()} to {df_5min['time'].max()}")
            logger.info(f"Cache saved: {cache_filename}")
            
            # Only process the first successful symbol
            break
        else:
            logger.warning(f"Failed to download {symbol}, trying next...")
    
    if 'df_5min' not in locals() or df_5min.empty:
        logger.error("Failed to download any symbol!")
    else:
        logger.info("Yahoo Finance daily download completed successfully!")

if __name__ == "__main__":
    main() 