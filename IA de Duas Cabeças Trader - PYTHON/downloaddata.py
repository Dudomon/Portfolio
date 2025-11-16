import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import ta
import logging
import os
import pickle
import time
from typing import Dict, List, Tuple

def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"data_download_{timestamp}.log")
    
    # Fix encoding issues with emojis
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def initialize_mt5():
    """Initialize MT5 connection with retry mechanism"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if not mt5.initialize():
                logger.error(f"Failed to initialize MT5 (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return False
            
            # Log MT5 version and connection info
            logger.info(f"MT5 version: {mt5.version()}")
            logger.info(f"Connected to: {mt5.terminal_info().name}")
            return True
            
        except Exception as e:
            logger.error(f"MT5 initialization error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    return False

def get_maximum_available_data(symbol: str, timeframe: int) -> Tuple[datetime, datetime]:
    """Get the maximum available data range for a symbol and timeframe"""
    try:
        # Try to get data from the very beginning
        # Start from 1990 (before gold trading was common)
        start_date = datetime(1990, 1, 1)
        end_date = datetime.now()
        
        # Try to get the first available data by requesting a large number of bars
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 10000)
        if rates is not None and len(rates) > 0:
            first_time = pd.to_datetime(rates[0]['time'], unit='s')
            last_time = pd.to_datetime(rates[-1]['time'], unit='s')
            logger.info(f"First available data for {symbol} {timeframe}: {first_time}")
            logger.info(f"Last available data for {symbol} {timeframe}: {last_time}")
            start_date = first_time
            end_date = last_time
        
        return start_date, end_date
        
    except Exception as e:
        logger.error(f"Error getting data range: {e}")
        # Fallback to reasonable defaults
        return datetime(2010, 1, 1), datetime.now()

def download_data(symbol: str, timeframe: int, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Download data from MT5 for a specific timeframe with optimized chunking and retry mechanism"""
    timezone = pytz.timezone("America/Sao_Paulo")
    start_date = timezone.localize(start_date)
    end_date = timezone.localize(end_date)
    
    logger.info(f"DOWNLOADING {symbol} data for timeframe {timeframe}")
    logger.info(f"Period: {start_date} to {end_date}")
    
    # Calculate optimal chunk size based on timeframe
    if timeframe == mt5.TIMEFRAME_M5:
        chunk_size = timedelta(days=7)  # 7 days for 5m (very granular)
    elif timeframe == mt5.TIMEFRAME_M15:
        chunk_size = timedelta(days=14)  # 14 days for 15m
    elif timeframe == mt5.TIMEFRAME_H4:
        chunk_size = timedelta(days=90)  # 90 days for 4h
    else:
        chunk_size = timedelta(days=30)  # Default
    
    all_data = []
    current_start = start_date
    total_chunks = 0
    successful_chunks = 0
    
    logger.info(f"Estimated chunks: {int((end_date - start_date) / chunk_size) + 1}")
    
    while current_start < end_date:
        current_end = min(current_start + chunk_size, end_date)
        total_chunks += 1
        
        logger.info(f"Chunk {total_chunks}: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
        
        # Retry mechanism for each chunk
        max_retries = 3
        chunk_success = False
        
        for retry in range(max_retries):
            try:
                rates = mt5.copy_rates_range(symbol, timeframe, current_start, current_end)
                if rates is not None and len(rates) > 0:
                    df_chunk = pd.DataFrame(rates)
                    df_chunk['time'] = pd.to_datetime(df_chunk['time'], unit='s')
                    all_data.append(df_chunk)
                    successful_chunks += 1
                    chunk_success = True
                    logger.info(f"Downloaded {len(rates)} rows for chunk {total_chunks}")
                    break
                else:
                    logger.warning(f"No data available for chunk {total_chunks}")
                    break
            except Exception as e:
                logger.error(f"Error downloading chunk {total_chunks} (retry {retry + 1}/{max_retries}): {str(e)}")
                if retry < max_retries - 1:
                    time.sleep(1)
        
        if not chunk_success:
            logger.error(f"Failed to download chunk {total_chunks} after {max_retries} retries")
        
        current_start = current_end
        
        # Progress update every 10 chunks
        if total_chunks % 10 == 0:
            logger.info(f"Progress: {total_chunks} chunks processed, {successful_chunks} successful")
    
    if not all_data:
        logger.error(f"Failed to download any data for {symbol} {timeframe}")
        return pd.DataFrame()
    
    # Combine all chunks
    logger.info("Combining all chunks...")
    df = pd.concat(all_data, ignore_index=True)
    df = df.drop_duplicates(subset=['time'])
    df = df.sort_values('time')
    
    logger.info(f"SUCCESS: Downloaded {len(df)} total rows for {symbol} {timeframe}")
    logger.info(f"Data range: {df['time'].min()} to {df['time'].max()}")
    logger.info(f"Success rate: {successful_chunks}/{total_chunks} chunks ({successful_chunks/total_chunks*100:.1f}%)")
    
    return df

def filter_static_points(df: pd.DataFrame, min_price_change: float = 0.01) -> pd.DataFrame:
    """Filter out static price points to reduce noise"""
    logger.info("Filtering static price points...")
    
    initial_rows = len(df)
    
    # Calculate price changes
    df['price_change'] = abs(df['close'] - df['close'].shift(1))
    
    # Filter rows with minimal price change
    df_filtered = df[df['price_change'] >= min_price_change].copy()
    
    # Keep first row (no previous price to compare)
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

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators"""
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
    
    # Volume ratio (if volume exists)
    if 'tick_volume' in df.columns:
        volume_sma = df['tick_volume'].rolling(20).mean().fillna(1)
        df['volume_ratio'] = (df['tick_volume'] / volume_sma).fillna(1.0).clip(0.1, 5.0)
    else:
        df['volume_ratio'] = 1.0
    
    # VaR 99%
    df['var_99'] = df['close'].rolling(window=20).quantile(0.01).fillna(df['close'] * 0.95)
    
    logger.info("Technical indicators calculated successfully")
    return df

def save_optimized_cache(df: pd.DataFrame, symbol: str, timestamp: str) -> str:
    """Save data as optimized pickle cache"""
    cache_dir = "data_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    cache_filename = os.path.join(cache_dir, f"{symbol}_CACHE_{timestamp}.pkl")
    
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

def load_optimized_cache(cache_filename: str) -> pd.DataFrame:
    """Load data from optimized pickle cache"""
    logger.info(f"Loading cache: {cache_filename}")
    
    with open(cache_filename, 'rb') as f:
        df = pickle.load(f)
    
    logger.info(f"Cache loaded: {len(df)} rows, {len(df.columns)} columns")
    return df

def process_timeframes(symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
    """Download and process data for different timeframes"""
    # Define timeframes with their MT5 constants
    timeframes = {
        '5m': mt5.TIMEFRAME_M5,
        '15m': mt5.TIMEFRAME_M15,
        '4h': mt5.TIMEFRAME_H4
    }
    
    processed_data = {}
    
    # Verify symbol exists
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error(f"Symbol {symbol} not found")
        return processed_data
    
    logger.info(f"Processing symbol: {symbol}")
    logger.info(f"Available timeframes: {list(timeframes.keys())}")
    
    for tf_name, tf in timeframes.items():
        try:
            logger.info(f"Starting download for {symbol} {tf_name}...")
            
            # Get maximum available data range for this timeframe
            tf_start, tf_end = get_maximum_available_data(symbol, tf)
            logger.info(f"{tf_name} data range: {tf_start} to {tf_end}")
            
            df = download_data(symbol, tf, tf_start, tf_end)
            
            if not df.empty:
                logger.info(f"Calculating indicators for {tf_name}")
                df = calculate_indicators(df)
                
                # Filter static points
                logger.info(f"Filtering static points for {tf_name}")
                df = filter_static_points(df, min_price_change=0.01)
                
                # Rename columns to include timeframe
                rename_cols = {}
                for col in df.columns:
                    if col not in ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']:
                        rename_cols[col] = f"{col}_{tf_name}"
                
                df = df.rename(columns=rename_cols)
                processed_data[tf_name] = df
                logger.info(f"Successfully processed {len(df)} rows for {tf_name}")
            else:
                logger.error(f"No data downloaded for {tf_name}")
                
        except Exception as e:
            logger.error(f"Error processing {tf_name}: {str(e)}")
            continue
    
    # Verify we have data for all timeframes
    missing_timeframes = set(timeframes.keys()) - set(processed_data.keys())
    if missing_timeframes:
        logger.warning(f"Missing data for timeframes: {missing_timeframes}")
    
    return processed_data

def merge_timeframes(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all timeframes into a single DataFrame"""
    logger.info("Merging all timeframes...")
    
    if not data:
        logger.error("No data to merge")
        return pd.DataFrame()
    
    # Start with the highest frequency data (5m)
    if '5m' in data:
        merged_df = data['5m'].copy()
        logger.info(f"Base timeframe: 5m with {len(merged_df)} rows")
    else:
        logger.error("No 5m data available for merging")
        return pd.DataFrame()
    
    # Merge other timeframes using forward fill
    for tf_name in ['15m', '4h']:
        if tf_name in data:
            tf_df = data[tf_name].copy()
            tf_df = tf_df.set_index('time')
            
            # Resample to 5m frequency and forward fill
            tf_df_resampled = tf_df.resample('5T').ffill()
            
            # Get only the feature columns (not OHLCV)
            feature_cols = [col for col in tf_df_resampled.columns if not col in ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]
            
            # Merge with main dataframe
            for col in feature_cols:
                merged_df[col] = tf_df_resampled[col].values[:len(merged_df)]
            
            logger.info(f"Merged {tf_name} features: {len(feature_cols)} columns")
    
    logger.info(f"Final merged dataset: {len(merged_df)} rows, {len(merged_df.columns)} columns")
    return merged_df

def main():
    # Initialize MT5
    if not initialize_mt5():
        return
    
    try:
        # Set parameters for gold
        symbol = "GOLD"  # Gold
        
        logger.info(f"Starting COMPLETE data download for {symbol}")
        logger.info(f"Target: MAXIMUM available historical data")
        
        # Download and process data for all timeframes
        data = process_timeframes(symbol, datetime(1990, 1, 1), datetime.now())
        
        if not data:
            logger.error("No data downloaded")
            return
        
        # Merge all timeframes
        merged_data = merge_timeframes(data)
        
        if merged_data.empty:
            logger.error("Failed to merge data")
            return
        
        # Save processed data
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual timeframes
        for tf_name, df in data.items():
            if not df.empty:
                filename = os.path.join(data_dir, f"{symbol}_{tf_name}_{timestamp}.csv")
                df.to_csv(filename, index=False)
                logger.info(f"Saved {filename} with {len(df)} rows")
        
        # Save merged dataset
        merged_filename = os.path.join(data_dir, f"{symbol}_COMPLETE_{timestamp}.csv")
        merged_data.to_csv(merged_filename, index=False)
        logger.info(f"Saved merged dataset: {merged_filename} with {len(merged_data)} rows")
        
        # Save optimized cache
        cache_filename = save_optimized_cache(merged_data, symbol, timestamp)
        
        # Summary
        logger.info("DOWNLOAD COMPLETE!")
        logger.info(f"Total timeframes processed: {len(data)}")
        logger.info(f"Total rows in merged dataset: {len(merged_data)}")
        logger.info(f"Data range: {merged_data['time'].min()} to {merged_data['time'].max()}")
        logger.info(f"Optimized cache saved: {cache_filename}")
        
        # Verify all timeframes were downloaded
        expected_timeframes = {'5m', '15m', '4h'}
        downloaded_timeframes = set(data.keys())
        if downloaded_timeframes == expected_timeframes:
            logger.info("All timeframes downloaded successfully!")
        else:
            missing = expected_timeframes - downloaded_timeframes
            logger.warning(f"Missing timeframes: {missing}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main() 