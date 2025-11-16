#!/usr/bin/env python3
"""
🚀 Download Yahoo Finance Gold Data - 1 MINUTE CURRICULUM LEARNING
Script específico para dados de 1 minuto para curriculum learning

OBJETIVO: Criar dataset base para treinar modelo com timeframes progressivos
- Dados 1m para micro-scalping inicial
- Base para expandir para 5m, 15m gradualmente
"""

import yfinance as yf
import pandas as pd
import os
import logging
from datetime import datetime, timedelta
import numpy as np

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def download_gold_1m_data(symbol="GC=F", days_back=7):
    """
    Download 1-minute gold data for curriculum learning
    
    Args:
        symbol: Yahoo symbol for gold futures (GC=F)
        days_back: How many days back to download (max 7 for 1m data)
    """
    logger = setup_logging()
    
    # Yahoo Finance limitation: 1m data only available for last 7-8 days
    if days_back > 7:
        logger.warning("Yahoo Finance 1m data limited to 7 days. Setting to 7.")
        days_back = 7
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    logger.info(f"🎯 CURRICULUM LEARNING - Downloading 1m data")
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"Expected bars: ~{days_back * 24 * 60} (24h * 60min)")
    
    try:
        # Download 1-minute data
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval="1m")
        
        if df.empty:
            logger.error(f"No 1m data downloaded for {symbol}")
            return pd.DataFrame()
        
        # Reset index to get datetime as column
        df = df.reset_index()
        
        # Rename columns to match MT5 format
        df = df.rename(columns={
            'Datetime': 'time',
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Add timeframe suffix for consistency
        df = df.rename(columns={
            'open': 'open_1m',
            'high': 'high_1m',
            'low': 'low_1m', 
            'close': 'close_1m',
            'volume': 'volume_1m'
        })
        
        # Calculate basic technical indicators for 1m
        df = add_technical_indicators_1m(df)
        
        # Filter only market hours (optional - keep all for now)
        # df = filter_market_hours(df)
        
        logger.info(f"✅ Downloaded {len(df)} bars of 1m data")
        logger.info(f"📊 Date range: {df['time'].min()} to {df['time'].max()}")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error downloading data: {e}")
        return pd.DataFrame()

def add_technical_indicators_1m(df):
    """Add technical indicators optimized for 1-minute timeframe"""
    logger = logging.getLogger(__name__)
    
    try:
        # Basic price features
        df['returns_1m'] = df['close_1m'].pct_change().fillna(0)
        
        # Very short-term indicators for 1m scalping
        
        # RSI with shorter period for 1m
        df['rsi_5_1m'] = calculate_rsi(df['close_1m'], period=5)   # 5-minute RSI
        df['rsi_14_1m'] = calculate_rsi(df['close_1m'], period=14) # 14-minute RSI
        
        # Very short moving averages
        df['sma_5_1m'] = df['close_1m'].rolling(5).mean()   # 5-minute SMA
        df['sma_20_1m'] = df['close_1m'].rolling(20).mean() # 20-minute SMA
        df['ema_9_1m'] = df['close_1m'].ewm(span=9).mean()  # 9-minute EMA
        
        # Bollinger Bands with short period
        bb_period = 10  # 10-minute Bollinger
        bb_std = df['close_1m'].rolling(bb_period).std()
        bb_sma = df['close_1m'].rolling(bb_period).mean()
        df['bb_upper_1m'] = bb_sma + (bb_std * 2)
        df['bb_lower_1m'] = bb_sma - (bb_std * 2)
        df['bb_position_1m'] = (df['close_1m'] - df['bb_lower_1m']) / (df['bb_upper_1m'] - df['bb_lower_1m'])
        df['bb_position_1m'] = df['bb_position_1m'].clip(0, 1).fillna(0.5)
        
        # Volatility measures
        df['volatility_10_1m'] = df['returns_1m'].rolling(10).std() * 100  # 10-minute volatility
        df['atr_10_1m'] = calculate_atr(df, period=10)  # 10-minute ATR
        
        # Momentum indicators
        df['momentum_5_1m'] = df['close_1m'] / df['close_1m'].shift(5) - 1  # 5-minute momentum
        
        # Volume indicators
        df['volume_sma_10_1m'] = df['volume_1m'].rolling(10).mean()
        df['volume_ratio_1m'] = df['volume_1m'] / df['volume_sma_10_1m'].fillna(1)
        
        # Trend strength (very short term)
        df['trend_strength_1m'] = df['returns_1m'].rolling(5).mean()  # 5-minute trend
        
        logger.info("✅ Technical indicators added for 1m timeframe")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error adding technical indicators: {e}")
        return df

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high_low = df[f'high_1m'] - df[f'low_1m']
    high_close = np.abs(df[f'high_1m'] - df[f'close_1m'].shift())
    low_close = np.abs(df[f'low_1m'] - df[f'close_1m'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr.fillna(0)

def save_dataset(df, filename_suffix=""):
    """Save dataset to multiple formats"""
    logger = logging.getLogger(__name__)
    
    if df.empty:
        logger.error("❌ Cannot save empty dataset")
        return
    
    # Create data directory if not exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"GOLD_1M_CURRICULUM{filename_suffix}_{timestamp}"
    
    # Save as CSV (human readable)
    csv_path = os.path.join(data_dir, f"{base_filename}.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"💾 CSV saved: {csv_path}")
    
    # Save as PKL (optimized for loading)
    pkl_path = os.path.join(data_dir, f"{base_filename}.pkl")
    df.to_pickle(pkl_path)
    logger.info(f"💾 PKL saved: {pkl_path}")
    
    # Print summary
    logger.info(f"📊 Dataset Summary:")
    logger.info(f"   Total bars: {len(df)}")
    logger.info(f"   Date range: {df['time'].min()} to {df['time'].max()}")
    logger.info(f"   Columns: {len(df.columns)}")
    logger.info(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    return pkl_path

def main():
    """Main execution function"""
    logger = setup_logging()
    
    logger.info("🚀 CURRICULUM LEARNING - 1M Dataset Creation")
    logger.info("=" * 60)
    
    # Try multiple gold symbols for 1m data
    symbols_to_try = ["GOLD", "GLD", "IAU", "GC=F"]
    
    df = pd.DataFrame()
    for symbol in symbols_to_try:
        logger.info(f"🔍 Trying symbol: {symbol}")
        df = download_gold_1m_data(symbol=symbol, days_back=7)
        if not df.empty:
            logger.info(f"✅ Success with symbol: {symbol}")
            break
        else:
            logger.warning(f"❌ Failed with symbol: {symbol}")
    
    if df.empty:
        logger.error("❌ All symbols failed. Trying with 5 days...")
        # Last resort: try with even shorter period
        for symbol in symbols_to_try:
            df = download_gold_1m_data(symbol=symbol, days_back=5)
            if not df.empty:
                break
    
    if not df.empty:
        # Save dataset
        pkl_path = save_dataset(df, "_BASE")
        
        logger.info("✅ 1M Dataset created successfully!")
        logger.info(f"📁 Ready for curriculum learning training")
        logger.info(f"🎯 Next step: Modify ppov1.py or headv6.py to use 1m data")
        
        # Display first few rows
        print("\n📊 Sample of 1M data:")
        print(df[['time', 'open_1m', 'high_1m', 'low_1m', 'close_1m', 'rsi_14_1m', 'bb_position_1m']].head(10))
        
    else:
        logger.error("❌ Failed to create 1M dataset")

if __name__ == "__main__":
    main()