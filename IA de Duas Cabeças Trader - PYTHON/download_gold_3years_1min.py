#!/usr/bin/env python3
"""
🚀 DOWNLOAD OURO 3 ANOS - 1 MINUTO TIMEFRAME
Download de dados históricos de 3 anos do ouro no Yahoo Finance
com timeframe de 1 minuto para treinamento intensivo
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import logging
import os
from datetime import datetime, timedelta
import pickle
import time

def setup_logging():
    """Setup logging avançado"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"gold_3years_1min_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def download_gold_1min_chunks(symbol: str = "GC=F", years_back: int = 3) -> pd.DataFrame:
    """
    Download dados de 1 minuto em chunks (Yahoo limita período para 1min)
    """
    logger = setup_logging()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)
    
    logger.info(f"🚀 DOWNLOAD INICIADO: {symbol}")
    logger.info(f"📅 Período: {start_date.strftime('%Y-%m-%d')} até {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"⏰ Timeframe: 1 minuto (dados brutos)")
    
    # Yahoo Finance limita dados de 1min a ~7 dias por vez
    # Vamos baixar em chunks de 7 dias
    chunk_days = 7
    all_data = []
    
    current_date = start_date
    chunk_count = 0
    
    while current_date < end_date:
        chunk_end = min(current_date + timedelta(days=chunk_days), end_date)
        chunk_count += 1
        
        logger.info(f"📦 Chunk {chunk_count}: {current_date.strftime('%Y-%m-%d')} → {chunk_end.strftime('%Y-%m-%d')}")
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Download com retry em caso de falha
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    df_chunk = ticker.history(
                        start=current_date.strftime('%Y-%m-%d'),
                        end=chunk_end.strftime('%Y-%m-%d'),
                        interval="1m"
                    )
                    
                    if not df_chunk.empty:
                        # Reset index e ajustar colunas
                        df_chunk = df_chunk.reset_index()
                        df_chunk = df_chunk.rename(columns={
                            'Datetime': 'time',
                            'Open': 'open',
                            'High': 'high',
                            'Low': 'low', 
                            'Close': 'close',
                            'Volume': 'tick_volume'
                        })
                        
                        # Adicionar colunas faltantes
                        df_chunk['spread'] = 0
                        df_chunk['real_volume'] = df_chunk['tick_volume']
                        
                        # Converter timezone para UTC se necessário
                        if df_chunk['time'].dt.tz is not None:
                            df_chunk['time'] = df_chunk['time'].dt.tz_convert('UTC').dt.tz_localize(None)
                        
                        all_data.append(df_chunk)
                        logger.info(f"✅ Chunk {chunk_count}: {len(df_chunk)} barras baixadas")
                        break
                    else:
                        logger.warning(f"⚠️ Chunk {chunk_count}: Dados vazios (tentativa {attempt+1}/{max_retries})")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Chunk {chunk_count} falhou (tentativa {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)  # Wait before retry
                    
            # Rate limiting para evitar bloqueio do Yahoo
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"❌ Erro no chunk {chunk_count}: {e}")
        
        current_date = chunk_end
    
    if not all_data:
        logger.error("❌ FALHA TOTAL: Nenhum dado foi baixado!")
        return pd.DataFrame()
    
    # Combinar todos os chunks
    df_combined = pd.concat(all_data, ignore_index=True)
    df_combined = df_combined.sort_values('time').reset_index(drop=True)
    
    # Remover duplicatas por timestamp
    df_combined = df_combined.drop_duplicates(subset=['time'], keep='first')
    
    logger.info(f"🎉 DOWNLOAD COMPLETO!")
    logger.info(f"📊 Total de barras: {len(df_combined):,}")
    logger.info(f"📅 Período real: {df_combined['time'].min()} → {df_combined['time'].max()}")
    
    return df_combined

def calculate_enhanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcular indicadores técnicos avançados para 1min"""
    logger = setup_logging()
    logger.info("🔧 Calculando indicadores técnicos avançados...")
    
    # Indicadores básicos
    df['returns'] = df['close'].pct_change().fillna(0)
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    
    # Volatilidades múltiplas
    df['volatility_10'] = df['returns'].rolling(window=10).std().fillna(0)
    df['volatility_30'] = df['returns'].rolling(window=30).std().fillna(0)
    df['volatility_60'] = df['returns'].rolling(window=60).std().fillna(0)
    
    # Médias móveis múltiplas 
    df['sma_10'] = df['close'].rolling(window=10).mean().fillna(df['close'])
    df['sma_30'] = df['close'].rolling(window=30).mean().fillna(df['close'])
    df['sma_60'] = df['close'].rolling(window=60).mean().fillna(df['close'])
    df['ema_10'] = df['close'].ewm(span=10).mean().fillna(df['close'])
    df['ema_30'] = df['close'].ewm(span=30).mean().fillna(df['close'])
    
    # RSI múltiplos
    try:
        df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi().fillna(50)
        df['rsi_30'] = ta.momentum.RSIIndicator(df['close'], window=30).rsi().fillna(50)
    except:
        df['rsi_14'] = 50.0
        df['rsi_30'] = 50.0
    
    # MACD
    try:
        macd_indicator = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
        df['macd'] = macd_indicator.macd().fillna(0)
        df['macd_signal'] = macd_indicator.macd_signal().fillna(0)
        df['macd_diff'] = macd_indicator.macd_diff().fillna(0)
    except:
        df['macd'] = 0.0
        df['macd_signal'] = 0.0 
        df['macd_diff'] = 0.0
    
    # Bollinger Bands
    bb_sma = df['close'].rolling(20).mean().fillna(df['close'])
    bb_std = df['close'].rolling(20).std().fillna(0.01)
    df['bb_upper'] = bb_sma + (bb_std * 2)
    df['bb_lower'] = bb_sma - (bb_std * 2)
    df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])).fillna(0.5).clip(0, 1)
    df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / bb_sma).fillna(0)
    
    # ATR
    try:
        df['atr_14'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range().fillna(0)
        df['atr_30'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=30).average_true_range().fillna(0)
    except:
        df['atr_14'] = (df['high'] - df['low']).rolling(14).mean().fillna(0)
        df['atr_30'] = (df['high'] - df['low']).rolling(30).mean().fillna(0)
    
    # Stochastic
    try:
        high_14 = df['high'].rolling(14).max()
        low_14 = df['low'].rolling(14).min()
        df['stoch_k'] = ((df['close'] - low_14) / (high_14 - low_14)) * 100
        df['stoch_k'] = df['stoch_k'].fillna(50).clip(0, 100)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean().fillna(50)
    except:
        df['stoch_k'] = 50.0
        df['stoch_d'] = 50.0
    
    # Volume indicators
    volume_sma_20 = df['tick_volume'].rolling(20).mean().fillna(1)
    df['volume_ratio'] = (df['tick_volume'] / volume_sma_20).fillna(1.0).clip(0.1, 5.0)
    df['volume_sma'] = volume_sma_20
    
    # Price patterns
    df['high_low_ratio'] = (df['high'] / df['low']).fillna(1.0)
    df['open_close_ratio'] = (df['open'] / df['close']).fillna(1.0)
    
    # Momentum indicators
    df['momentum_10'] = (df['close'] / df['close'].shift(10)).fillna(1.0)
    df['momentum_30'] = (df['close'] / df['close'].shift(30)).fillna(1.0)
    
    # Rate of Change
    df['roc_10'] = df['close'].pct_change(periods=10).fillna(0) * 100
    df['roc_30'] = df['close'].pct_change(periods=30).fillna(0) * 100
    
    logger.info("✅ Indicadores calculados com sucesso!")
    return df

def filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Filtrar apenas horários de mercado (trading hours)"""
    logger = setup_logging()
    logger.info("⏰ Filtrando horários de mercado...")
    
    initial_rows = len(df)
    
    # Adicionar colunas de tempo
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['weekday'] = df['time'].dt.weekday  # 0=Monday, 6=Sunday
    
    # Filtrar apenas horários de mercado do ouro (24h, mas excluir finais de semana)
    df_filtered = df[
        (df['weekday'] < 5) |  # Segunda a Sexta
        ((df['weekday'] == 6) & (df['hour'] >= 18)) |  # Domingo após 18h
        ((df['weekday'] == 4) & (df['hour'] < 22))     # Sexta até 22h
    ].copy()
    
    # Remover colunas temporárias
    df_filtered = df_filtered.drop(['hour', 'minute', 'weekday'], axis=1)
    
    removed_rows = initial_rows - len(df_filtered)
    logger.info(f"📊 Removidas {removed_rows:,} barras fora do horário ({removed_rows/initial_rows*100:.1f}%)")
    logger.info(f"📊 Dataset final: {len(df_filtered):,} barras")
    
    return df_filtered

def save_comprehensive_dataset(df: pd.DataFrame, symbol: str, timeframe: str) -> tuple:
    """Salvar dataset completo em múltiplos formatos"""
    logger = setup_logging()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Criar diretórios
    for dir_name in ['data', 'data_cache']:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    
    # Nomes dos arquivos
    base_name = f"{symbol}_3YEARS_{timeframe}_{timestamp}"
    csv_file = os.path.join('data', f"{base_name}.csv")
    pkl_file = os.path.join('data_cache', f"{base_name}.pkl")
    
    logger.info(f"💾 Salvando dataset: {base_name}")
    
    # Otimizar tipos de dados
    df_optimized = df.copy()
    df_optimized['time'] = pd.to_datetime(df_optimized['time'])
    
    # Float64 -> Float32 (economia de memória)
    float_cols = df_optimized.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df_optimized[col] = df_optimized[col].astype('float32')
    
    # Int64 -> Int32
    int_cols = df_optimized.select_dtypes(include=['int64']).columns
    for col in int_cols:
        df_optimized[col] = df_optimized[col].astype('int32')
    
    # Salvar CSV
    df_optimized.to_csv(csv_file, index=False)
    csv_size_mb = os.path.getsize(csv_file) / (1024 * 1024)
    logger.info(f"✅ CSV salvo: {csv_file} ({csv_size_mb:.2f} MB)")
    
    # Salvar Pickle otimizado
    with open(pkl_file, 'wb') as f:
        pickle.dump(df_optimized, f, protocol=pickle.HIGHEST_PROTOCOL)
    pkl_size_mb = os.path.getsize(pkl_file) / (1024 * 1024) 
    logger.info(f"✅ Pickle salvo: {pkl_file} ({pkl_size_mb:.2f} MB)")
    
    return csv_file, pkl_file

def main():
    """Função principal para download dos 3 anos de dados de 1 minuto"""
    logger = setup_logging()
    
    logger.info("🚀 INICIANDO DOWNLOAD DE OURO - 3 ANOS - 1 MINUTO")
    logger.info("="*60)
    
    # Símbolos a tentar (ordem de prioridade)
    symbols = [
        "GC=F",      # Gold Futures (melhor para 1min)
        "XAUUSD=X",  # Gold vs USD
        "GLD",       # SPDR Gold Trust ETF
        "IAU",       # iShares Gold Trust ETF
    ]
    
    for symbol in symbols:
        logger.info(f"🎯 Tentando símbolo: {symbol}")
        
        # Download dos dados
        df = download_gold_1min_chunks(symbol, years_back=3)
        
        if not df.empty:
            logger.info(f"✅ Download bem-sucedido para {symbol}")
            logger.info(f"📊 Barras baixadas: {len(df):,}")
            
            # Filtrar horários de mercado
            df = filter_market_hours(df)
            
            # Calcular indicadores
            df = calculate_enhanced_indicators(df)
            
            # Salvar dataset
            csv_file, pkl_file = save_comprehensive_dataset(df, symbol, "1MIN")
            
            # Estatísticas finais
            logger.info("="*60)
            logger.info("🎉 DOWNLOAD COMPLETO!")
            logger.info(f"📊 Símbolo: {symbol}")
            logger.info(f"📊 Total de barras: {len(df):,}")
            logger.info(f"📅 Período: {df['time'].min()} → {df['time'].max()}")
            logger.info(f"💰 Preço inicial: ${df['close'].iloc[0]:.2f}")
            logger.info(f"💰 Preço final: ${df['close'].iloc[-1]:.2f}")
            logger.info(f"📈 Variação total: {((df['close'].iloc[-1]/df['close'].iloc[0])-1)*100:+.2f}%")
            logger.info(f"📁 Arquivos salvos:")
            logger.info(f"   📄 CSV: {csv_file}")
            logger.info(f"   🗃️ Pickle: {pkl_file}")
            logger.info("="*60)
            
            # Dataset criado com sucesso, interromper loop
            return
            
        else:
            logger.warning(f"❌ Falha no download para {symbol}, tentando próximo...")
    
    logger.error("❌ FALHA TOTAL: Não foi possível baixar dados de nenhum símbolo!")

if __name__ == "__main__":
    main()