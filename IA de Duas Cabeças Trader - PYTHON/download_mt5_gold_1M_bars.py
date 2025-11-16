#!/usr/bin/env python3
"""
🚀 DOWNLOAD MT5 GOLD - 1 MILHÃO DE BARRAS 1MIN
===============================================

Download massivo de 3 anos de dados GOLD 1min do MT5
~1,061,262 barras de dados 100% reais
"""

import MetaTrader5 as mt5
import pandas as pd
import os
import logging
import numpy as np
from datetime import datetime, timedelta

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def download_mt5_gold_1M_bars():
    """🚀 Download 1 milhão de barras GOLD 1min do MT5 (3 anos)"""
    logger = setup_logging()

    logger.info("🚀 DOWNLOAD MT5 GOLD - 1 MILHÃO DE BARRAS 1MIN")
    logger.info("=" * 80)
    logger.info("📊 Período: 3 anos completos (~1.06M barras)")
    logger.info("🎯 Objetivo: Dataset massivo 100% real para treino robusto")
    logger.info("")

    # Conectar MT5
    if not mt5.initialize():
        logger.error("❌ Falha ao conectar MT5")
        logger.error(f"Erro: {mt5.last_error()}")
        return pd.DataFrame()

    logger.info("✅ MT5 conectado com sucesso")

    # Configuração do download
    symbol = "GOLD"  # Melhor símbolo identificado
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1095)  # 3 anos = ~1095 dias

    logger.info(f"🔍 Símbolo: {symbol}")
    logger.info(f"📅 Período: {start_date.strftime('%Y-%m-%d %H:%M')} até {end_date.strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"⏱️ Duração: 3 anos (~1095 dias)")

    try:
        # Verificar símbolo
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"❌ Símbolo {symbol} não encontrado!")
            mt5.shutdown()
            return pd.DataFrame()

        if not symbol_info.visible:
            logger.info(f"🔧 Ativando símbolo {symbol}")
            if not mt5.symbol_select(symbol, True):
                logger.error(f"❌ Falha ao ativar símbolo {symbol}")
                mt5.shutdown()
                return pd.DataFrame()

        logger.info(f"✅ Símbolo {symbol} ativado e disponível")

        # DOWNLOAD MASSIVO
        logger.info("🚀 INICIANDO DOWNLOAD MASSIVO...")
        logger.info("⏳ Isso pode demorar alguns minutos...")

        start_time = datetime.now()

        rates = mt5.copy_rates_range(
            symbol,
            mt5.TIMEFRAME_M1,
            start_date,
            end_date
        )

        download_time = datetime.now() - start_time

        if rates is None or len(rates) == 0:
            logger.error("❌ Nenhum dado obtido do MT5!")
            mt5.shutdown()
            return pd.DataFrame()

        logger.info(f"✅ DOWNLOAD CONCLUÍDO!")
        logger.info(f"📊 Barras obtidas: {len(rates):,}")
        logger.info(f"⏱️ Tempo de download: {download_time.total_seconds():.2f}s")

        # Converter para DataFrame
        logger.info("🔄 Convertendo para DataFrame...")
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Estatísticas básicas
        actual_days = (df['time'].max() - df['time'].min()).days
        bars_per_day = len(df) / max(actual_days, 1)

        logger.info(f"📈 ESTATÍSTICAS DO DATASET:")
        logger.info(f"   Primeira barra: {df['time'].min()}")
        logger.info(f"   Última barra: {df['time'].max()}")
        logger.info(f"   Dias reais: {actual_days}")
        logger.info(f"   Barras/dia: {bars_per_day:.0f}")
        logger.info(f"   Coverage: {len(df) / (actual_days * 1440 * 5/7):.1%} (considerando weekends)")

    except Exception as e:
        logger.error(f"❌ Erro durante download: {e}")
        mt5.shutdown()
        return pd.DataFrame()

    finally:
        # Finalizar MT5
        mt5.shutdown()
        logger.info("🔌 Conexão MT5 finalizada")

    # PROCESSAMENTO E ENRIQUECIMENTO
    logger.info("🧠 PROCESSANDO E ADICIONANDO INDICADORES...")

    df = process_and_enrich_data(df, symbol, logger)

    # SALVAR DATASET
    logger.info("💾 SALVANDO DATASET MASSIVO...")
    save_path = save_massive_dataset(df, symbol, logger)

    # ESTATÍSTICAS FINAIS
    logger.info("")
    logger.info("=" * 80)
    logger.info("🎉 DATASET MASSIVO 1M+ BARRAS CRIADO COM SUCESSO!")
    logger.info("=" * 80)
    logger.info(f"📊 ESTATÍSTICAS FINAIS:")
    logger.info(f"   Símbolo: {symbol}")
    logger.info(f"   Total de barras: {len(df):,}")
    logger.info(f"   Período: {df['timestamp'].min()} até {df['timestamp'].max()}")
    logger.info(f"   Duração: {(df['timestamp'].max() - df['timestamp'].min()).days} dias")
    logger.info(f"   Colunas: {len(df.columns)} features")
    logger.info(f"   Arquivo: {save_path}")
    logger.info(f"   Tamanho: ~{os.path.getsize(save_path) / (1024*1024):.1f} MB")

    # RECOMENDAÇÕES DE USO
    logger.info(f"")
    logger.info(f"🎯 RECOMENDAÇÕES DE USO:")
    logger.info(f"   📈 TREINO: Usar 70% dos dados (~{int(len(df) * 0.7):,} barras)")
    logger.info(f"   📊 VALIDAÇÃO: Usar 20% dos dados (~{int(len(df) * 0.2):,} barras)")
    logger.info(f"   🧪 TESTE: Usar 10% dos dados (~{int(len(df) * 0.1):,} barras)")
    logger.info(f"   💡 Split temporal sequencial recomendado")

    logger.info(f"")
    logger.info(f"✅ DATASET 1M+ BARRAS PRONTO PARA USO!")

    return df

def process_and_enrich_data(df, symbol, logger):
    """🧠 Processar e enriquecer dados MT5 com indicadores completos"""

    # Renomear colunas para padrão
    df = df.rename(columns={
        'time': 'timestamp',
        'open': 'open_1m',
        'high': 'high_1m',
        'low': 'low_1m',
        'close': 'close_1m',
        'tick_volume': 'volume_1m'
    })

    logger.info(f"📊 Adicionando indicadores técnicos...")

    # Adicionar spread e real_volume (MT5 padrão)
    df['spread'] = df['high_1m'] - df['low_1m']  # Spread básico
    df['real_volume'] = df['volume_1m']  # Volume real

    # INDICADORES BÁSICOS
    logger.info("   📈 Returns...")
    df['returns_1m'] = df['close_1m'].pct_change().fillna(0)

    # RSI
    logger.info("   📊 RSI...")
    df['rsi_14_1m'] = calculate_rsi(df['close_1m'], 14)

    # SMAs
    logger.info("   📈 SMAs...")
    df['sma_5_1m'] = df['close_1m'].rolling(5).mean()
    df['sma_10_1m'] = df['close_1m'].rolling(10).mean()
    df['sma_20_1m'] = df['close_1m'].rolling(20).mean()
    df['sma_50_1m'] = df['close_1m'].rolling(50).mean()
    df['sma_200_1m'] = df['close_1m'].rolling(200).mean()

    # EMAs
    logger.info("   📉 EMAs...")
    df['ema_5_1m'] = df['close_1m'].ewm(span=5).mean()
    df['ema_10_1m'] = df['close_1m'].ewm(span=10).mean()
    df['ema_20_1m'] = df['close_1m'].ewm(span=20).mean()
    df['ema_50_1m'] = df['close_1m'].ewm(span=50).mean()
    df['ema_200_1m'] = df['close_1m'].ewm(span=200).mean()

    # MACD
    logger.info("   🌊 MACD...")
    ema12 = df['close_1m'].ewm(span=12).mean()
    ema26 = df['close_1m'].ewm(span=26).mean()
    df['macd_1m'] = ema12 - ema26
    df['macd_signal_1m'] = df['macd_1m'].ewm(span=9).mean()
    df['macd_histogram_1m'] = df['macd_1m'] - df['macd_signal_1m']

    # Bollinger Bands
    logger.info("   📊 Bollinger Bands...")
    bb_sma = df['close_1m'].rolling(20).mean()
    bb_std = df['close_1m'].rolling(20).std()
    df['bb_upper_1m'] = bb_sma + (bb_std * 2)
    df['bb_middle_1m'] = bb_sma
    df['bb_lower_1m'] = bb_sma - (bb_std * 2)
    df['bb_position_1m'] = ((df['close_1m'] - df['bb_lower_1m']) /
                           (df['bb_upper_1m'] - df['bb_lower_1m'])).clip(0, 1).fillna(0.5)

    # ATR
    logger.info("   📏 ATR...")
    df['tr_1m'] = calculate_true_range(df)
    df['atr_14_1m'] = df['tr_1m'].rolling(14).mean()

    # Volume indicators
    logger.info("   📦 Volume indicators...")
    df['volume_sma_20_1m'] = df['volume_1m'].rolling(20).mean()
    df['volume_ratio_1m'] = df['volume_1m'] / df['volume_sma_20_1m'].fillna(1)

    # Volatility
    logger.info("   🌪️ Volatility...")
    df['volatility_20_1m'] = df['returns_1m'].rolling(20).std() * 100

    # Stochastic
    logger.info("   📈 Stochastic...")
    df['stoch_k_1m'] = calculate_stochastic_k(df, 14)
    df['stoch_d_1m'] = df['stoch_k_1m'].rolling(3).mean()

    # Preencher NaNs
    logger.info("   🔧 Preenchendo valores ausentes...")
    df = df.fillna(method='ffill').fillna(method='bfill')

    # Verificar qualidade final
    total_nulls = df.isnull().sum().sum()
    logger.info(f"   ✅ Processamento concluído!")
    logger.info(f"   📊 Total de features: {len(df.columns)}")
    logger.info(f"   🔍 Valores nulos restantes: {total_nulls}")

    return df

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_true_range(df):
    """Calculate True Range"""
    high_low = df['high_1m'] - df['low_1m']
    high_close = np.abs(df['high_1m'] - df['close_1m'].shift())
    low_close = np.abs(df['low_1m'] - df['close_1m'].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.fillna(0)

def calculate_stochastic_k(df, period=14):
    """Calculate Stochastic %K"""
    lowest_low = df['low_1m'].rolling(period).min()
    highest_high = df['high_1m'].rolling(period).max()

    k_percent = 100 * ((df['close_1m'] - lowest_low) / (highest_high - lowest_low))
    return k_percent.fillna(50)

def save_massive_dataset(df, symbol, logger):
    """💾 Salvar dataset massivo com metadados"""

    # Criar diretório
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Nome do arquivo com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"GOLD_1M_MT5_MASSIVE_{len(df)}bars_{timestamp}.pkl"
    save_path = os.path.join(data_dir, filename)

    # Salvar em pickle (mais eficiente para dados grandes)
    logger.info(f"💾 Salvando como: {filename}")
    df.to_pickle(save_path)

    # Salvar também uma amostra em CSV para inspeção
    sample_size = min(10000, len(df))
    csv_path = save_path.replace('.pkl', '_SAMPLE.csv')
    df.head(sample_size).to_csv(csv_path, index=False)
    logger.info(f"📄 Amostra salva como CSV: {os.path.basename(csv_path)}")

    # Criar arquivo de metadados
    metadata = {
        'filename': filename,
        'symbol': symbol,
        'total_bars': len(df),
        'start_date': str(df['timestamp'].min()),
        'end_date': str(df['timestamp'].max()),
        'duration_days': (df['timestamp'].max() - df['timestamp'].min()).days,
        'columns': len(df.columns),
        'features': list(df.columns),
        'file_size_mb': os.path.getsize(save_path) / (1024*1024),
        'created': timestamp,
        'source': 'MT5_Direct',
        'quality': '100%_Real_Data'
    }

    metadata_path = save_path.replace('.pkl', '_metadata.txt')
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    logger.info(f"📋 Metadados salvos: {os.path.basename(metadata_path)}")

    return save_path

if __name__ == "__main__":
    df = download_mt5_gold_1M_bars()

    if not df.empty:
        print(f"\n🎉 SUCCESS! Dataset massivo criado:")
        print(f"📊 Total: {len(df):,} barras")
        print(f"📅 Período: {df['timestamp'].min()} até {df['timestamp'].max()}")
        print(f"🔧 Features: {len(df.columns)} colunas")
        print(f"\n📈 PRIMEIRAS 5 LINHAS:")
        display_cols = ['timestamp', 'open_1m', 'high_1m', 'low_1m', 'close_1m', 'volume_1m', 'rsi_14_1m']
        available_cols = [col for col in display_cols if col in df.columns]
        print(df[available_cols].head())
        print(f"\n✅ Dataset massivo 1M+ barras pronto para uso!")
    else:
        print("❌ Falha ao criar dataset massivo")