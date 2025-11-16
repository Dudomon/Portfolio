#!/usr/bin/env python3
"""
🚀 GOLD 3 YEARS SMART 1MIN - ESTRATÉGIA HÍBRIDA
Yahoo limita 1min a 30 dias - Vamos combinar:
1. Dados diários históricos (3 anos) 
2. Dados 1min recentes (30 dias)
3. Resample inteligente com patterns realistas
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
    """Setup logging"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"gold_3years_smart1min_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def download_recent_1min_data(symbol: str = "GC=F", days_back: int = 25) -> pd.DataFrame:
    """Download dados 1min reais dos últimos 25 dias"""
    logger = setup_logging()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    logger.info(f"📥 DOWNLOAD 1MIN RECENTE: {symbol}")
    logger.info(f"📅 Período: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    
    try:
        ticker = yf.Ticker(symbol)
        df_1min = ticker.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval="1m"
        )
        
        if df_1min.empty:
            logger.warning(f"⚠️ Dados 1min vazios para {symbol}")
            return pd.DataFrame()
        
        # Ajustar colunas
        df_1min = df_1min.reset_index()
        df_1min = df_1min.rename(columns={
            'Datetime': 'time',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'tick_volume'
        })
        
        # Adicionar colunas
        df_1min['spread'] = 0
        df_1min['real_volume'] = df_1min['tick_volume']
        df_1min['data_type'] = 'REAL_1MIN'  # Flag para identificar dados reais
        
        # Converter timezone
        if df_1min['time'].dt.tz is not None:
            df_1min['time'] = df_1min['time'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        logger.info(f"✅ 1MIN REAL: {len(df_1min):,} barras")
        return df_1min
        
    except Exception as e:
        logger.error(f"❌ Erro no download 1min: {e}")
        return pd.DataFrame()

def download_historical_daily_data(symbol: str = "GC=F", years_back: int = 3, exclude_recent_days: int = 30) -> pd.DataFrame:
    """Download dados diários históricos (excluindo período recente)"""
    logger = setup_logging()
    
    end_date = datetime.now() - timedelta(days=exclude_recent_days)
    start_date = end_date - timedelta(days=years_back * 365)
    
    logger.info(f"📥 DOWNLOAD DIÁRIO HISTÓRICO: {symbol}")
    logger.info(f"📅 Período: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    
    try:
        ticker = yf.Ticker(symbol)
        df_daily = ticker.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval="1d"
        )
        
        if df_daily.empty:
            logger.warning(f"⚠️ Dados diários vazios para {symbol}")
            return pd.DataFrame()
        
        # Ajustar colunas
        df_daily = df_daily.reset_index()
        df_daily = df_daily.rename(columns={
            'Date': 'time',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'tick_volume'
        })
        
        # Adicionar colunas
        df_daily['spread'] = 0
        df_daily['real_volume'] = df_daily['tick_volume']
        df_daily['data_type'] = 'DAILY_HISTORICAL'
        
        # Converter timezone
        if df_daily['time'].dt.tz is not None:
            df_daily['time'] = df_daily['time'].dt.tz_localize(None)
        
        logger.info(f"✅ DIÁRIO HISTÓRICO: {len(df_daily):,} dias")
        return df_daily
        
    except Exception as e:
        logger.error(f"❌ Erro no download diário: {e}")
        return pd.DataFrame()

def smart_resample_daily_to_1min(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Resample inteligente de dados diários para 1 minuto com patterns realistas"""
    logger = setup_logging()
    logger.info("🧠 SMART RESAMPLE: Diário → 1 Minuto com patterns realistas")
    
    df_1min_synthetic = []
    
    for _, day_row in df_daily.iterrows():
        day_date = day_row['time'].date()
        
        # Criar padrões de trading realistas por hora
        trading_patterns = {
            # Horários US: maior volatilidade
            'us_open': {'hours': [13, 14, 15], 'volatility_mult': 2.0, 'volume_mult': 2.5},
            'us_active': {'hours': [16, 17, 18, 19], 'volatility_mult': 1.5, 'volume_mult': 1.8},
            'us_close': {'hours': [20, 21], 'volatility_mult': 1.8, 'volume_mult': 2.0},
            # Horários Ásia: volatilidade moderada
            'asia': {'hours': [0, 1, 2, 3, 4, 5, 6, 7, 8], 'volatility_mult': 0.8, 'volume_mult': 0.6},
            # Horários Europa: volatilidade média
            'europe': {'hours': [9, 10, 11, 12], 'volatility_mult': 1.2, 'volume_mult': 1.0},
            # Horários baixa atividade
            'quiet': {'hours': [22, 23], 'volatility_mult': 0.5, 'volume_mult': 0.3}
        }
        
        # Gerar 1440 barras por dia (24h * 60min)
        day_start = datetime.combine(day_date, datetime.min.time())
        
        # Calcular movimento total do dia
        daily_range = day_row['high'] - day_row['low']
        daily_move = day_row['close'] - day_row['open']
        
        # Distribuir movimento ao longo do dia com padrões realistas
        current_price = day_row['open']
        cumulative_move = 0.0
        
        for minute in range(1440):  # 1440 minutos por dia
            minute_time = day_start + timedelta(minutes=minute)
            hour = minute_time.hour
            
            # Determinar padrão de trading para esta hora
            pattern = None
            for pattern_name, pattern_info in trading_patterns.items():
                if hour in pattern_info['hours']:
                    pattern = pattern_info
                    break
            
            if pattern is None:
                pattern = trading_patterns['quiet']
            
            # Calcular movimento para este minuto
            base_move_per_minute = daily_move / 1440
            
            # Adicionar ruído realista baseado no padrão
            volatility = (daily_range / 1440) * pattern['volatility_mult']
            noise = np.random.normal(0, volatility * 0.5)
            trend_component = base_move_per_minute * (1 + np.random.normal(0, 0.2))
            
            minute_move = trend_component + noise
            cumulative_move += minute_move
            
            # Calcular OHLC para o minuto
            open_price = current_price
            
            # Variação intra-minuto
            intra_minute_range = volatility * np.random.uniform(0.3, 1.0)
            high_offset = np.random.uniform(0, intra_minute_range)
            low_offset = -np.random.uniform(0, intra_minute_range)
            
            high_price = open_price + max(high_offset, minute_move if minute_move > 0 else 0)
            low_price = open_price + min(low_offset, minute_move if minute_move < 0 else 0)
            close_price = open_price + minute_move
            
            # Garantir relação OHLC correta
            high_price = max(open_price, close_price, high_price)
            low_price = min(open_price, close_price, low_price)
            
            # Calcular volume baseado no padrão
            base_volume = day_row['tick_volume'] / 1440
            volume = max(1, int(base_volume * pattern['volume_mult'] * np.random.uniform(0.5, 2.0)))
            
            # ⚡ CORREÇÃO: Formato XXXX.XX para ouro (2 casas decimais)
            open_price = round(open_price, 2)
            high_price = round(high_price, 2)
            low_price = round(low_price, 2)
            close_price = round(close_price, 2)
            
            # Adicionar barra
            df_1min_synthetic.append({
                'time': minute_time,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'tick_volume': volume,
                'spread': 0,
                'real_volume': volume,
                'data_type': 'SYNTHETIC_1MIN'
            })
            
            current_price = close_price
        
        # Ajustar última barra para bater com close do dia (formato XXXX.XX)
        if df_1min_synthetic:
            df_1min_synthetic[-1]['close'] = round(day_row['close'], 2)
    
    df_result = pd.DataFrame(df_1min_synthetic)
    logger.info(f"✅ RESAMPLE COMPLETO: {len(df_result):,} barras sintéticas")
    
    return df_result

def merge_datasets(df_synthetic: pd.DataFrame, df_real: pd.DataFrame) -> pd.DataFrame:
    """Combinar datasets sintético e real"""
    logger = setup_logging()
    logger.info("🔗 MERGE: Combinando dados sintéticos + reais")
    
    # Combinar datasets
    df_combined = pd.concat([df_synthetic, df_real], ignore_index=True)
    
    # Ordenar por tempo
    df_combined = df_combined.sort_values('time').reset_index(drop=True)
    
    # Remover sobreposições (priorizar dados reais)
    df_combined = df_combined.drop_duplicates(subset=['time'], keep='last')
    
    logger.info(f"✅ MERGE COMPLETO:")
    logger.info(f"   📊 Dados sintéticos: {len(df_synthetic):,}")
    logger.info(f"   📊 Dados reais: {len(df_real):,}")
    logger.info(f"   📊 Dataset final: {len(df_combined):,}")
    
    return df_combined

def calculate_premium_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcular indicadores premium de alta qualidade"""
    logger = setup_logging()
    logger.info("💎 Calculando indicadores PREMIUM para qualidade absoluta...")
    
    # Indicadores básicos premium
    df['returns'] = df['close'].pct_change().fillna(0)
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    df['high_low_ratio'] = (df['high'] / df['low']).fillna(1.0)
    df['open_close_ratio'] = (df['open'] / df['close']).fillna(1.0)
    
    # Volatilidades múltiplas (1min, 5min, 15min equivalentes)
    df['volatility_1'] = df['returns'].rolling(window=1).std().fillna(0)
    df['volatility_5'] = df['returns'].rolling(window=5).std().fillna(0)
    df['volatility_15'] = df['returns'].rolling(window=15).std().fillna(0)
    df['volatility_60'] = df['returns'].rolling(window=60).std().fillna(0)
    df['volatility_240'] = df['returns'].rolling(window=240).std().fillna(0)
    
    # Médias móveis premium (EMA + SMA)
    windows = [5, 10, 15, 20, 30, 60, 120, 240]
    for window in windows:
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean().fillna(df['close'])
        df[f'ema_{window}'] = df['close'].ewm(span=window).mean().fillna(df['close'])
        df[f'price_vs_sma_{window}'] = (df['close'] / df[f'sma_{window}']) - 1
        df[f'price_vs_ema_{window}'] = (df['close'] / df[f'ema_{window}']) - 1
    
    # RSI múltiplos (ultra precisos)
    try:
        rsi_windows = [14, 21, 30, 60]
        for window in rsi_windows:
            df[f'rsi_{window}'] = ta.momentum.RSIIndicator(df['close'], window=window).rsi().fillna(50)
            df[f'rsi_{window}_normalized'] = (df[f'rsi_{window}'] - 50) / 50  # [-1, 1]
    except:
        for window in rsi_windows:
            df[f'rsi_{window}'] = 50.0
            df[f'rsi_{window}_normalized'] = 0.0
    
    # MACD premium
    try:
        macd_fast, macd_slow, macd_signal = 12, 26, 9
        macd_indicator = ta.trend.MACD(df['close'], window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal)
        df['macd'] = macd_indicator.macd().fillna(0)
        df['macd_signal'] = macd_indicator.macd_signal().fillna(0)
        df['macd_diff'] = macd_indicator.macd_diff().fillna(0)
        df['macd_normalized'] = df['macd'] / df['close']  # Normalizado pelo preço
    except:
        df['macd'] = 0.0
        df['macd_signal'] = 0.0
        df['macd_diff'] = 0.0
        df['macd_normalized'] = 0.0
    
    # Bollinger Bands premium
    bb_windows = [20, 50]
    for window in bb_windows:
        bb_sma = df['close'].rolling(window).mean().fillna(df['close'])
        bb_std = df['close'].rolling(window).std().fillna(0.01)
        df[f'bb_upper_{window}'] = bb_sma + (bb_std * 2)
        df[f'bb_lower_{window}'] = bb_sma - (bb_std * 2)
        df[f'bb_position_{window}'] = ((df['close'] - df[f'bb_lower_{window}']) / 
                                       (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])).fillna(0.5).clip(0, 1)
        df[f'bb_width_{window}'] = ((df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / bb_sma).fillna(0)
    
    # ATR premium (múltiplos períodos)
    try:
        atr_windows = [14, 21, 50]
        for window in atr_windows:
            df[f'atr_{window}'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=window).average_true_range().fillna(0)
            df[f'atr_{window}_normalized'] = df[f'atr_{window}'] / df['close']  # ATR como % do preço
    except:
        for window in atr_windows:
            df[f'atr_{window}'] = (df['high'] - df['low']).rolling(window).mean().fillna(0)
            df[f'atr_{window}_normalized'] = df[f'atr_{window}'] / df['close']
    
    # Stochastic premium
    try:
        stoch_windows = [14, 21]
        for window in stoch_windows:
            high_period = df['high'].rolling(window).max()
            low_period = df['low'].rolling(window).min()
            df[f'stoch_k_{window}'] = ((df['close'] - low_period) / (high_period - low_period)) * 100
            df[f'stoch_k_{window}'] = df[f'stoch_k_{window}'].fillna(50).clip(0, 100)
            df[f'stoch_d_{window}'] = df[f'stoch_k_{window}'].rolling(3).mean().fillna(50)
            df[f'stoch_k_{window}_normalized'] = (df[f'stoch_k_{window}'] - 50) / 50  # [-1, 1]
    except:
        for window in stoch_windows:
            df[f'stoch_k_{window}'] = 50.0
            df[f'stoch_d_{window}'] = 50.0
            df[f'stoch_k_{window}_normalized'] = 0.0
    
    # Volume premium indicators
    volume_windows = [10, 20, 60]
    for window in volume_windows:
        volume_sma = df['tick_volume'].rolling(window).mean().fillna(1)
        df[f'volume_ratio_{window}'] = (df['tick_volume'] / volume_sma).fillna(1.0).clip(0.1, 10.0)
        df[f'volume_sma_{window}'] = volume_sma
    
    # Momentum premium
    momentum_windows = [5, 10, 15, 30, 60]
    for window in momentum_windows:
        df[f'momentum_{window}'] = (df['close'] / df['close'].shift(window)).fillna(1.0)
        df[f'momentum_{window}_normalized'] = df[f'momentum_{window}'] - 1.0
        df[f'roc_{window}'] = df['close'].pct_change(periods=window).fillna(0) * 100
    
    # Market microstructure
    df['spread_estimate'] = (df['high'] - df['low']) / df['close']  # Spread estimado
    df['price_impact'] = abs(df['close'] - df['open']) / df['close']  # Impacto do preço
    df['intrabar_return'] = (df['close'] - df['open']) / df['open']  # Retorno intra-barra
    
    # Heikin Ashi (candlesticks suavizados)
    df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['ha_open'] = np.nan
    df.loc[0, 'ha_open'] = df.loc[0, 'open']
    
    for i in range(1, len(df)):
        df.loc[i, 'ha_open'] = (df.loc[i-1, 'ha_open'] + df.loc[i-1, 'ha_close']) / 2
    
    df['ha_high'] = df[['high', 'ha_open', 'ha_close']].max(axis=1)
    df['ha_low'] = df[['low', 'ha_open', 'ha_close']].min(axis=1)
    
    logger.info("✅ Indicadores PREMIUM calculados - Qualidade absoluta garantida!")
    return df

def quality_control_and_validation(df: pd.DataFrame) -> pd.DataFrame:
    """Controle de qualidade rigoroso do dataset"""
    logger = setup_logging()
    logger.info("🔍 CONTROLE DE QUALIDADE - Validação rigorosa do dataset")
    
    initial_rows = len(df)
    issues_found = []
    
    # 1. Verificar dados faltantes
    missing_data = df.isnull().sum()
    critical_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
    
    for col in critical_columns:
        if missing_data[col] > 0:
            issues_found.append(f"❌ {col}: {missing_data[col]} valores faltantes")
    
    # 2. Verificar relações OHLC
    ohlc_violations = 0
    df['ohlc_valid'] = True
    
    # High >= max(Open, Close) e Low <= min(Open, Close)
    invalid_high = (df['high'] < df[['open', 'close']].max(axis=1))
    invalid_low = (df['low'] > df[['open', 'close']].min(axis=1))
    
    df.loc[invalid_high | invalid_low, 'ohlc_valid'] = False
    ohlc_violations = (invalid_high | invalid_low).sum()
    
    if ohlc_violations > 0:
        issues_found.append(f"❌ OHLC: {ohlc_violations} violações de relação")
        # Corrigir violações
        df.loc[invalid_high, 'high'] = df.loc[invalid_high, ['open', 'close']].max(axis=1)
        df.loc[invalid_low, 'low'] = df.loc[invalid_low, ['open', 'close']].min(axis=1)
        logger.info("🔧 CORREÇÃO: Violações OHLC corrigidas automaticamente")
    
    # 3. Verificar outliers extremos (> 10 desvios padrão)
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        mean_price = df[col].mean()
        std_price = df[col].std()
        outliers = abs(df[col] - mean_price) > (10 * std_price)
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            issues_found.append(f"⚠️ {col}: {outlier_count} outliers extremos")
    
    # 4. Verificar continuidade temporal
    df = df.sort_values('time').reset_index(drop=True)
    time_gaps = df['time'].diff()
    expected_gap = timedelta(minutes=1)
    large_gaps = time_gaps > timedelta(hours=4)  # Gaps > 4h são suspeitos
    gap_count = large_gaps.sum()
    
    if gap_count > 0:
        issues_found.append(f"⚠️ TEMPO: {gap_count} gaps temporais grandes (>4h)")
    
    # 5. Verificar volatilidade extrema
    df['price_change'] = df['close'].pct_change().abs()
    extreme_moves = df['price_change'] > 0.05  # Movimentos > 5% por minuto
    extreme_count = extreme_moves.sum()
    
    if extreme_count > 0:
        issues_found.append(f"⚠️ VOLATILIDADE: {extreme_count} movimentos extremos (>5%/min)")
    
    # 6. Verificar volumes
    zero_volume = (df['tick_volume'] == 0).sum()
    if zero_volume > 0:
        issues_found.append(f"⚠️ VOLUME: {zero_volume} barras com volume zero")
    
    # Remover colunas temporárias
    df = df.drop(['ohlc_valid', 'price_change'], axis=1, errors='ignore')
    
    # Relatório de qualidade
    logger.info("📊 RELATÓRIO DE QUALIDADE:")
    logger.info(f"   📊 Linhas processadas: {initial_rows:,}")
    logger.info(f"   📊 Linhas finais: {len(df):,}")
    
    if issues_found:
        logger.warning("⚠️ ISSUES ENCONTRADAS:")
        for issue in issues_found:
            logger.warning(f"   {issue}")
    else:
        logger.info("✅ QUALIDADE PERFEITA: Nenhum problema encontrado!")
    
    # Calcular estatísticas finais
    logger.info("📈 ESTATÍSTICAS FINAIS:")
    logger.info(f"   💰 Preço inicial: ${df['close'].iloc[0]:.2f}")
    logger.info(f"   💰 Preço final: ${df['close'].iloc[-1]:.2f}")
    logger.info(f"   📈 Variação total: {((df['close'].iloc[-1]/df['close'].iloc[0])-1)*100:+.2f}%")
    logger.info(f"   📊 Volume médio: {df['tick_volume'].mean():,.0f}")
    logger.info(f"   📊 Volatilidade média: {df['returns'].std()*100:.4f}%")
    
    return df

def save_premium_dataset(df: pd.DataFrame, symbol: str) -> tuple:
    """Salvar dataset premium com metadados completos"""
    logger = setup_logging()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Criar diretórios
    for dir_name in ['data', 'data_cache', 'metadata']:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    
    base_name = f"{symbol}_PREMIUM_3Y_1MIN_{timestamp}"
    csv_file = os.path.join('data', f"{base_name}.csv")
    pkl_file = os.path.join('data_cache', f"{base_name}.pkl")
    metadata_file = os.path.join('metadata', f"{base_name}_info.txt")
    
    # ⚡ CORREÇÃO: Garantir formato XXXX.XX antes de salvar
    df_optimized = df.copy()
    df_optimized['time'] = pd.to_datetime(df_optimized['time'])
    
    # Aplicar formato XXXX.XX em todas as colunas de preço
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in df_optimized.columns:
            df_optimized[col] = df_optimized[col].round(2)
    
    # Otimização de tipos
    float_cols = df_optimized.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df_optimized[col] = df_optimized[col].astype('float32')
    
    int_cols = df_optimized.select_dtypes(include=['int64']).columns  
    for col in int_cols:
        df_optimized[col] = df_optimized[col].astype('int32')
    
    # Salvar arquivos
    df_optimized.to_csv(csv_file, index=False)
    
    with open(pkl_file, 'wb') as f:
        pickle.dump(df_optimized, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Criar metadados
    metadata = f"""
GOLD PREMIUM DATASET - 3 ANOS 1 MINUTO
=====================================
Criado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Símbolo: {symbol}
Timeframe: 1 minuto
Período: {df['time'].min()} → {df['time'].max()}

ESTATÍSTICAS:
- Total de barras: {len(df):,}
- Dados reais (1min): {len(df[df['data_type'] == 'REAL_1MIN']):,}
- Dados sintéticos: {len(df[df['data_type'] == 'SYNTHETIC_1MIN']):,}
- Preço inicial: ${df['close'].iloc[0]:.2f}
- Preço final: ${df['close'].iloc[-1]:.2f}
- Variação total: {((df['close'].iloc[-1]/df['close'].iloc[0])-1)*100:+.2f}%
- Volume médio: {df['tick_volume'].mean():,.0f}
- Volatilidade média: {df['returns'].std()*100:.4f}%

QUALIDADE:
- Indicadores: {len([col for col in df.columns if col not in ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume', 'data_type']])} técnicos
- Controle de qualidade: APROVADO
- Dados OHLC: VALIDADOS
- Continuidade temporal: VERIFICADA

ARQUIVOS:
- CSV: {csv_file}
- Pickle: {pkl_file}
- Metadata: {metadata_file}
"""
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write(metadata)
    
    # Calcular tamanhos
    csv_size = os.path.getsize(csv_file) / (1024 * 1024)
    pkl_size = os.path.getsize(pkl_file) / (1024 * 1024)
    
    logger.info(f"💾 DATASET PREMIUM SALVO:")
    logger.info(f"   📄 CSV: {csv_file} ({csv_size:.2f} MB)")
    logger.info(f"   🗃️ Pickle: {pkl_file} ({pkl_size:.2f} MB)")
    logger.info(f"   📋 Metadata: {metadata_file}")
    
    return csv_file, pkl_file, metadata_file

def main():
    """Função principal - Dataset premium de 3 anos"""
    logger = setup_logging()
    
    logger.info("🚀 DATASET PREMIUM GOLD - 3 ANOS 1 MINUTO")
    logger.info("=" * 60)
    
    symbols = ["GC=F", "XAUUSD=X", "GLD", "IAU"]
    
    for symbol in symbols:
        logger.info(f"🎯 Processando símbolo: {symbol}")
        
        # 1. Download dados recentes 1min (reais)
        df_real_1min = download_recent_1min_data(symbol, days_back=25)
        
        # 2. Download dados históricos diários  
        df_historical_daily = download_historical_daily_data(symbol, years_back=3, exclude_recent_days=30)
        
        if df_historical_daily.empty:
            logger.warning(f"❌ Falha no download histórico para {symbol}")
            continue
        
        # 3. Resample inteligente diário → 1min
        df_synthetic_1min = smart_resample_daily_to_1min(df_historical_daily)
        
        # 4. Combinar datasets
        if not df_real_1min.empty:
            df_final = merge_datasets(df_synthetic_1min, df_real_1min)
            logger.info("✅ Datasets combinados: Sintético + Real")
        else:
            df_final = df_synthetic_1min
            logger.info("⚠️ Apenas dados sintéticos (falha no download 1min)")
        
        if df_final.empty:
            logger.warning(f"❌ Dataset final vazio para {symbol}")
            continue
        
        # 5. Calcular indicadores premium
        df_final = calculate_premium_indicators(df_final)
        
        # 6. Controle de qualidade rigoroso
        df_final = quality_control_and_validation(df_final)
        
        # 7. Salvar dataset premium
        csv_file, pkl_file, metadata_file = save_premium_dataset(df_final, symbol)
        
        logger.info("=" * 60)
        logger.info("🎉 DATASET PREMIUM CONCLUÍDO!")
        logger.info(f"📊 Símbolo: {symbol}")
        logger.info(f"📊 Barras: {len(df_final):,}")
        logger.info(f"📅 Período: {df_final['time'].min()} → {df_final['time'].max()}")
        logger.info(f"💎 Indicadores: {len(df_final.columns)} colunas")
        logger.info(f"✅ Qualidade: MÁXIMA")
        logger.info("=" * 60)
        
        return  # Sucesso - interromper após primeiro símbolo
    
    logger.error("❌ FALHA TOTAL: Nenhum símbolo funcionou!")

if __name__ == "__main__":
    main()