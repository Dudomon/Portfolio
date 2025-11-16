#!/usr/bin/env python3
"""
🚀 Download MT5 Gold Data - 25 SEMANAS de dados 1MIN
Usando MetaTrader5 para obter histórico extenso (~6 meses)
"""

import MetaTrader5 as mt5
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

def download_mt5_gold_25weeks():
    """Download 25 semanas de dados 1min do MT5 (~6 meses)"""
    logger = setup_logging()

    logger.info("🚀 DOWNLOAD MT5 GOLD - 25 SEMANAS DE DADOS 1MIN")
    logger.info("=" * 60)

    # Conectar MT5
    if not mt5.initialize():
        logger.error("❌ Falha ao conectar MT5")
        logger.error(f"Erro: {mt5.last_error()}")
        return pd.DataFrame()

    logger.info("✅ MT5 conectado com sucesso")

    # Símbolos de ouro para tentar
    gold_symbols = ["GOLD", "XAUUSD", "GOLD#", "GOLDUSD", "XAU/USD"]

    # Período: 25 semanas atrás até hoje (~6 meses)
    end_date = datetime.now()
    start_date = end_date - timedelta(weeks=25)  # 25 semanas = ~175 dias

    logger.info(f"📅 Período: {start_date.strftime('%Y-%m-%d')} até {end_date.strftime('%Y-%m-%d')}")

    best_df = pd.DataFrame()
    best_symbol = ""

    for symbol in gold_symbols:
        try:
            logger.info(f"🔍 Testando símbolo: {symbol}")

            # Verificar se símbolo existe
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.warning(f"❌ {symbol}: Símbolo não encontrado")
                continue

            if not symbol_info.visible:
                logger.info(f"🔧 {symbol}: Ativando símbolo")
                if not mt5.symbol_select(symbol, True):
                    logger.warning(f"❌ {symbol}: Falha ao ativar")
                    continue

            # Baixar dados 1min
            rates = mt5.copy_rates_range(
                symbol,
                mt5.TIMEFRAME_M1,
                start_date,
                end_date
            )

            if rates is None or len(rates) == 0:
                logger.warning(f"❌ {symbol}: Nenhum dado obtido")
                continue

            # Converter para DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')

            logger.info(f"✅ {symbol}: {len(df)} barras obtidas")
            logger.info(f"   Range: {df['time'].min()} até {df['time'].max()}")

            # Manter o maior dataset
            if len(df) > len(best_df):
                best_df = df.copy()
                best_symbol = symbol

        except Exception as e:
            logger.error(f"❌ {symbol}: Erro - {e}")
            continue

    # Finalizar MT5
    mt5.shutdown()

    if best_df.empty:
        logger.error("❌ Nenhum dado MT5 obtido!")
        return pd.DataFrame()

    logger.info(f"🏆 MELHOR: {best_symbol} = {len(best_df)} barras")

    # Processar dados
    df = process_mt5_data(best_df, best_symbol)

    # Salvar
    save_path = save_mt5_dataset(df, best_symbol)

    # Estatísticas finais
    duration_days = (df['timestamp'].max() - df['timestamp'].min()).days
    duration_weeks = round(duration_days / 7, 1)
    logger.info("📊 DATASET MT5 CRIADO:")
    logger.info(f"   Símbolo: {best_symbol}")
    logger.info(f"   Bars: {len(df):,}")
    logger.info(f"   Duração: {duration_days} dias ({duration_weeks} semanas)")
    logger.info(f"   Range: {df['timestamp'].min()} até {df['timestamp'].max()}")
    logger.info(f"   Arquivo: {save_path}")

    return df

def process_mt5_data(df, symbol):
    """Processar dados MT5 para formato padrão"""

    # Renomear colunas para padrão
    df = df.rename(columns={
        'time': 'timestamp',
        'open': 'open_1m',
        'high': 'high_1m',
        'low': 'low_1m',
        'close': 'close_1m',
        'tick_volume': 'volume_1m'
    })

    # Adicionar indicadores básicos
    df = add_mt5_indicators(df)

    return df

def add_mt5_indicators(df):
    """Adicionar indicadores essenciais para dados MT5"""

    # Returns
    df['returns_1m'] = df['close_1m'].pct_change().fillna(0)

    # RSI
    df['rsi_14_1m'] = calculate_rsi(df['close_1m'], 14)

    # SMA
    df['sma_20_1m'] = df['close_1m'].rolling(20).mean()
    df['sma_50_1m'] = df['close_1m'].rolling(50).mean()

    # Bollinger Bands
    bb_sma = df['close_1m'].rolling(20).mean()
    bb_std = df['close_1m'].rolling(20).std()
    df['bb_upper_1m'] = bb_sma + (bb_std * 2)
    df['bb_lower_1m'] = bb_sma - (bb_std * 2)
    df['bb_position_1m'] = ((df['close_1m'] - df['bb_lower_1m']) /
                           (df['bb_upper_1m'] - df['bb_lower_1m'])).clip(0, 1).fillna(0.5)

    # ATR
    df['atr_14_1m'] = calculate_atr_mt5(df, 14)

    # Volatilidade
    df['volatility_20_1m'] = df['returns_1m'].rolling(20).std() * 100

    # Volume ratio
    df['volume_sma_20_1m'] = df['volume_1m'].rolling(20).mean()
    df['volume_ratio_1m'] = df['volume_1m'] / df['volume_sma_20_1m'].fillna(1)

    return df

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_atr_mt5(df, period=14):
    """Calculate ATR para dados MT5"""
    high_low = df['high_1m'] - df['low_1m']
    high_close = np.abs(df['high_1m'] - df['close_1m'].shift())
    low_close = np.abs(df['low_1m'] - df['close_1m'].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr.fillna(0)

def save_mt5_dataset(df, symbol):
    """Salvar dataset MT5"""

    # Criar diretório
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Nome do arquivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"GOLD_1M_MT5_{symbol}_25WEEKS_{timestamp}.pkl"
    save_path = os.path.join(data_dir, filename)

    # Salvar
    df.to_pickle(save_path)

    return save_path

if __name__ == "__main__":
    df = download_mt5_gold_25weeks()

    if not df.empty:
        print(f"\n🎯 DADOS MT5 OBTIDOS:")
        print(f"Total: {len(df):,} barras")
        duration_days = (df['timestamp'].max() - df['timestamp'].min()).days
        duration_weeks = round(duration_days / 7, 1)
        print(f"Duração: {duration_days} dias ({duration_weeks} semanas)")
        print(f"\n📊 PRIMEIRAS 5 LINHAS:")
        print(df[['timestamp', 'open_1m', 'high_1m', 'low_1m', 'close_1m', 'rsi_14_1m']].head())
        print(f"\n✅ Dados MT5 salvos com sucesso!")
    else:
        print("❌ Falha ao obter dados MT5")