#!/usr/bin/env python3
"""
🚀 Download FRESH Yahoo Finance Gold Data - 1 MINUTE (1 MÊS+)
Otimizado para obter máximo de dados frescos possível
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

def download_fresh_gold_data():
    """Download maximum fresh 1m gold data available"""
    logger = setup_logging()

    logger.info("🚀 DOWNLOAD DADOS FRESCOS 1MIN - MÁXIMO PERÍODO")
    logger.info("=" * 60)

    # Estratégia: Tentar diferentes períodos e símbolos
    symbols = ["GC=F", "GOLD", "GLD"]  # Gold futures, spot, ETF
    periods = ["7d", "5d", "2d", "1d"]  # Yahoo 1m limit

    best_df = pd.DataFrame()
    best_symbol = ""
    best_period = ""

    for symbol in symbols:
        for period in periods:
            try:
                logger.info(f"🔍 Testando: {symbol} período {period}")

                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval="1m")

                if not df.empty:
                    logger.info(f"✅ {symbol} {period}: {len(df)} bars")

                    # Manter o maior dataset
                    if len(df) > len(best_df):
                        best_df = df.copy()
                        best_symbol = symbol
                        best_period = period

                else:
                    logger.warning(f"❌ {symbol} {period}: Sem dados")

            except Exception as e:
                logger.warning(f"❌ {symbol} {period}: Erro - {e}")

    if best_df.empty:
        logger.error("❌ Nenhum dado 1m obtido!")
        return pd.DataFrame()

    logger.info(f"🏆 MELHOR: {best_symbol} {best_period} = {len(best_df)} bars")

    # Processar dados
    df = best_df.reset_index()

    # Padronizar colunas
    df = df.rename(columns={
        'Datetime': 'timestamp',
        'Open': 'open_1m',
        'High': 'high_1m',
        'Low': 'low_1m',
        'Close': 'close_1m',
        'Volume': 'volume_1m'
    })

    # Adicionar indicadores básicos
    df = add_basic_indicators(df)

    # Salvar
    save_path = save_fresh_dataset(df, best_symbol, best_period)

    # Estatísticas finais
    logger.info("📊 DATASET FRESCO CRIADO:")
    logger.info(f"   Símbolo: {best_symbol}")
    logger.info(f"   Período: {best_period}")
    logger.info(f"   Bars: {len(df)}")
    logger.info(f"   Range: {df['timestamp'].min()} até {df['timestamp'].max()}")
    logger.info(f"   Arquivo: {save_path}")

    return df

def add_basic_indicators(df):
    """Adicionar indicadores essenciais"""

    # Returns
    df['returns_1m'] = df['close_1m'].pct_change().fillna(0)

    # RSI rápido para 1m
    df['rsi_14_1m'] = calculate_rsi(df['close_1m'], 14)

    # SMA curta
    df['sma_20_1m'] = df['close_1m'].rolling(20).mean()

    # Bollinger position
    bb_sma = df['close_1m'].rolling(20).mean()
    bb_std = df['close_1m'].rolling(20).std()
    df['bb_upper_1m'] = bb_sma + (bb_std * 2)
    df['bb_lower_1m'] = bb_sma - (bb_std * 2)
    df['bb_position_1m'] = ((df['close_1m'] - df['bb_lower_1m']) /
                           (df['bb_upper_1m'] - df['bb_lower_1m'])).clip(0, 1).fillna(0.5)

    # Volatilidade
    df['volatility_20_1m'] = df['returns_1m'].rolling(20).std() * 100

    return df

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def save_fresh_dataset(df, symbol, period):
    """Salvar dataset fresco"""

    # Criar diretório
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Nome do arquivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"GOLD_1M_FRESH_{symbol}_{period}_{timestamp}.pkl"
    save_path = os.path.join(data_dir, filename)

    # Salvar
    df.to_pickle(save_path)

    return save_path

if __name__ == "__main__":
    df = download_fresh_gold_data()

    if not df.empty:
        print("\n🎯 PRIMEIRAS 10 LINHAS:")
        print(df[['timestamp', 'open_1m', 'high_1m', 'low_1m', 'close_1m', 'rsi_14_1m']].head(10))
        print(f"\n✅ {len(df)} bars de dados frescos salvos!")
    else:
        print("❌ Falha ao obter dados frescos")