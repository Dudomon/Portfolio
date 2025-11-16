#!/usr/bin/env python3
"""
🚀 REALISTIC GOLD DATASET V2 - Algoritmo Melhorado
Combina dados reais com sintéticos ultra-realistas
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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def download_real_5m_data(symbol: str = "GC=F", months_back: int = 6) -> pd.DataFrame:
    """Download dados reais de 5 minutos (até 2 anos disponíveis)"""
    logger = setup_logging()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back * 30)
    
    logger.info(f"📥 DOWNLOAD REAL 5MIN: {symbol}")
    logger.info(f"📅 Período: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval="5m")
        
        if df.empty:
            logger.warning(f"⚠️ Sem dados 5m para {symbol}")
            return pd.DataFrame()
        
        # Preparar formato padrão
        df = df.reset_index()
        df = df.rename(columns={
            'Datetime': 'time',
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'tick_volume'
        })
        
        df['spread'] = 0
        df['real_volume'] = df['tick_volume']
        df['data_type'] = 'REAL_5MIN'
        
        # Garantir formato XXXX.XX
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df[col] = df[col].round(2)
        
        # Converter timezone para UTC se necessário
        if df['time'].dt.tz is not None:
            df['time'] = df['time'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        logger.info(f"✅ REAL 5MIN: {len(df):,} barras baixadas")
        return df
        
    except Exception as e:
        logger.error(f"❌ Erro no download 5m: {e}")
        return pd.DataFrame()

def interpolate_5m_to_1m_realistic(df_5m: pd.DataFrame) -> pd.DataFrame:
    """Interpolar dados 5m para 1m de forma ultra-realista"""
    logger = setup_logging()
    logger.info("🧮 INTERPOLAÇÃO REALISTA: 5m → 1m")
    
    df_1m = []
    
    for i in range(len(df_5m)):
        bar_5m = df_5m.iloc[i]
        base_time = pd.to_datetime(bar_5m['time'])
        
        # DADOS DA BARRA 5M
        open_5m = bar_5m['open']
        high_5m = bar_5m['high']
        low_5m = bar_5m['low'] 
        close_5m = bar_5m['close']
        volume_5m = bar_5m['tick_volume']
        
        # ESTRATÉGIA: Criar path realista Open→Close respeitando H/L
        total_move = close_5m - open_5m
        volatility = (high_5m - low_5m) / open_5m
        
        # 1. GERAR PATH BROWNIANO BASE
        price_path = generate_realistic_price_path(
            start_price=open_5m,
            end_price=close_5m, 
            high_limit=high_5m,
            low_limit=low_5m,
            volatility=volatility,
            steps=5
        )
        
        # 2. DISTRIBUIR VOLUME REALISTICAMENTE
        volume_distribution = generate_realistic_volume_pattern(volume_5m, volatility)
        
        # 3. CRIAR 5 BARRAS DE 1 MINUTO
        for minute_idx in range(5):
            minute_time = base_time + timedelta(minutes=minute_idx)
            
            # Preços da barra 1m
            open_1m = price_path[minute_idx]
            close_1m = price_path[minute_idx + 1]
            
            # High/Low com ruído intra-minuto realista
            intra_volatility = volatility * np.random.uniform(0.2, 0.8)
            noise_range = open_1m * intra_volatility * 0.1
            
            high_1m = max(open_1m, close_1m) + abs(np.random.normal(0, noise_range))
            low_1m = min(open_1m, close_1m) - abs(np.random.normal(0, noise_range))
            
            # Garantir limites globais da barra 5m
            high_1m = min(high_1m, high_5m)
            low_1m = max(low_1m, low_5m)
            
            # Formato XXXX.XX
            open_1m = round(open_1m, 2)
            high_1m = round(high_1m, 2)
            low_1m = round(low_1m, 2)
            close_1m = round(close_1m, 2)
            
            df_1m.append({
                'time': minute_time,
                'open': open_1m,
                'high': high_1m,
                'low': low_1m,
                'close': close_1m,
                'tick_volume': int(volume_distribution[minute_idx]),
                'spread': 0,
                'real_volume': int(volume_distribution[minute_idx]),
                'data_type': 'INTERPOLATED_1MIN'
            })
    
    df_result = pd.DataFrame(df_1m)
    logger.info(f"✅ INTERPOLAÇÃO: {len(df_result):,} barras 1m criadas")
    return df_result

def generate_realistic_price_path(start_price, end_price, high_limit, low_limit, volatility, steps=5):
    """Gerar path de preços realista usando Brownian Bridge"""
    
    # Brownian Bridge: garante start→end respeitando limites
    total_move = end_price - start_price
    
    # Gerar path base
    path = [start_price]
    
    for i in range(1, steps):
        # Progresso linear base
        linear_progress = start_price + (total_move * i / steps)
        
        # Adicionar ruído browniano
        noise_factor = volatility * np.sqrt(i * (steps - i) / steps) * 0.3
        noise = np.random.normal(0, start_price * noise_factor)
        
        # Aplicar ruído com limites
        new_price = linear_progress + noise
        new_price = max(low_limit, min(high_limit, new_price))
        
        path.append(new_price)
    
    # Garantir end price
    path.append(end_price)
    
    return path

def generate_realistic_volume_pattern(total_volume, volatility):
    """Distribuir volume de forma realista (mais volume em barras mais voláteis)"""
    
    # Padrão base: volume mais concentrado no meio (barras 2-3)
    base_pattern = np.array([0.15, 0.2, 0.3, 0.25, 0.1])  # Soma = 1.0
    
    # Ajustar baseado na volatilidade
    volatility_factor = min(volatility * 10, 2.0)  # Limitar fator
    
    # Mais volatilidade = mais volume no início/fim (momentum)
    if volatility_factor > 1.0:
        base_pattern = np.array([0.25, 0.2, 0.2, 0.2, 0.15])
    
    # Adicionar ruído
    noise = np.random.dirichlet([1] * 5) * 0.3  # Ruído que soma 0.3
    final_pattern = base_pattern * 0.7 + noise  # 70% padrão + 30% ruído
    
    # Garantir soma = 1
    final_pattern = final_pattern / final_pattern.sum()
    
    # Distribuir volume
    volume_distribution = final_pattern * total_volume
    
    # Garantir mínimo de volume
    volume_distribution = np.maximum(volume_distribution, 1)
    
    return volume_distribution

def enhance_synthetic_algorithm(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Melhorar algoritmo sintético com realismo extremo"""
    logger = setup_logging()
    logger.info("🧠 ALGORITMO SINTÉTICO V2: Ultra-realista")
    
    df_1m_synthetic = []
    
    for _, day_row in df_daily.iterrows():
        day_start = pd.to_datetime(day_row['time'])
        
        # Padrões de volatilidade intraday realistas
        volatility_hourly = generate_hourly_volatility_pattern()
        
        # Preço inicial do dia
        current_price = day_row['open']
        daily_move = day_row['close'] - day_row['open']
        
        # Gerar 1440 barras (24h * 60min) com padrões realistas
        for minute in range(1440):
            minute_time = day_start + timedelta(minutes=minute)
            hour = minute_time.hour
            
            # 1. VOLATILIDADE DINÂMICA POR HORÁRIO
            base_volatility = (day_row['high'] - day_row['low']) / day_row['open'] / 1440
            hourly_mult = volatility_hourly[hour]
            current_volatility = base_volatility * hourly_mult
            
            # 2. MOVIMENTO BROWNIANO COM DRIFT
            daily_progress = minute / 1440.0
            target_price = day_row['open'] + (daily_move * daily_progress)
            drift = (target_price - current_price) * 0.01  # Gentle pull toward target
            
            # 3. RUÍDO REALISTA
            noise = np.random.normal(0, current_price * current_volatility)
            
            # 4. MOMENTUM E REVERSÃO
            if minute > 0:
                last_move = df_1m_synthetic[-1]['close'] - df_1m_synthetic[-1]['open']
                momentum = last_move * np.random.uniform(-0.3, 0.7)  # Momentum com reversão
            else:
                momentum = 0
            
            # 5. CALCULAR PREÇO FINAL
            price_change = drift + noise + momentum
            new_price = current_price + price_change
            
            # Garantir limites do dia
            new_price = max(day_row['low'], min(day_row['high'], new_price))
            new_price = round(new_price, 2)
            
            # 6. GERAR OHLC INTRA-MINUTO
            open_price = current_price
            close_price = new_price
            
            # High/Low com micro-movimentos
            intra_range = abs(price_change) * np.random.uniform(0.5, 2.0)
            high_price = max(open_price, close_price) + intra_range * np.random.uniform(0, 1)
            low_price = min(open_price, close_price) - intra_range * np.random.uniform(0, 1)
            
            # Limites e formato
            high_price = round(min(high_price, day_row['high']), 2)
            low_price = round(max(low_price, day_row['low']), 2)
            open_price = round(open_price, 2)
            close_price = round(close_price, 2)
            
            # 7. VOLUME REALISTA
            base_volume = day_row['tick_volume'] / 1440
            volume_mult = hourly_mult * np.random.uniform(0.3, 3.0)
            volume = max(1, int(base_volume * volume_mult))
            
            df_1m_synthetic.append({
                'time': minute_time,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'tick_volume': volume,
                'spread': 0,
                'real_volume': volume,
                'data_type': 'SYNTHETIC_V2_1MIN'
            })
            
            current_price = new_price
        
        # Ajustar última barra para bater close do dia
        if df_1m_synthetic:
            df_1m_synthetic[-1]['close'] = round(day_row['close'], 2)
    
    df_result = pd.DataFrame(df_1m_synthetic)
    logger.info(f"✅ SINTÉTICO V2: {len(df_result):,} barras ultra-realistas")
    return df_result

def generate_hourly_volatility_pattern():
    """Padrão realista de volatilidade por hora (baseado em padrões de ouro reais)"""
    # Ouro tem padrões específicos: mais ativo durante abertura asiática/europeia/americana
    hourly_mults = {
        0: 0.4, 1: 0.3, 2: 0.3, 3: 0.4,  # Madrugada quieta
        4: 0.6, 5: 0.8, 6: 1.2, 7: 1.4,  # Abertura asiática
        8: 1.8, 9: 2.0, 10: 1.6, 11: 1.3, # Europa ativa
        12: 1.0, 13: 0.9, 14: 1.1, 15: 1.4, # Almoço Europa
        16: 1.8, 17: 2.2, 18: 1.9, 19: 1.5, # Abertura América
        20: 1.2, 21: 1.0, 22: 0.8, 23: 0.6  # Fechamento
    }
    return hourly_mults

def download_daily_historical(symbol: str = "GC=F", years_back: int = 3) -> pd.DataFrame:
    """Download dados históricos diários para base sintética"""
    logger = setup_logging()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)
    
    logger.info(f"📥 DOWNLOAD HISTÓRICO DIÁRIO: {symbol}")
    logger.info(f"📅 Período: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval="1d")
        
        if df.empty:
            logger.warning(f"⚠️ Sem dados diários para {symbol}")
            return pd.DataFrame()
        
        # Preparar formato
        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'time',
            'Open': 'open',
            'High': 'high',
            'Low': 'low', 
            'Close': 'close',
            'Volume': 'tick_volume'
        })
        
        # Formato XXXX.XX
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df[col] = df[col].round(2)
        
        logger.info(f"✅ HISTÓRICO DIÁRIO: {len(df):,} dias")
        return df
        
    except Exception as e:
        logger.error(f"❌ Erro no download histórico: {e}")
        return pd.DataFrame()

def calculate_premium_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcular indicadores premium otimizados"""
    logger = setup_logging()
    logger.info("💎 Calculando indicadores premium...")
    
    try:
        # Basic features
        df['returns'] = df['close'].pct_change().fillna(0)
        df['high_low_ratio'] = (df['high'] / df['low']).fillna(1.0)
        df['open_close_ratio'] = (df['open'] / df['close']).fillna(1.0)
        
        # Moving averages (otimizadas para 1min)
        for period in [5, 10, 20, 60]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean().fillna(df['close'])
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean().fillna(df['close'])
        
        # RSI otimizado
        try:
            df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi().fillna(50)
            df['rsi_30'] = ta.momentum.RSIIndicator(df['close'], window=30).rsi().fillna(50)
        except:
            df['rsi_14'] = 50.0
            df['rsi_30'] = 50.0
        
        # MACD
        try:
            macd = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
            df['macd'] = macd.macd().fillna(0)
            df['macd_signal'] = macd.macd_signal().fillna(0)
            df['macd_diff'] = macd.macd_diff().fillna(0)
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
        
        # ATR
        try:
            df['atr_14'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range().fillna(0)
        except:
            df['atr_14'] = (df['high'] - df['low']).rolling(14).mean().fillna(0)
        
        # Volatility
        df['volatility_60'] = df['returns'].rolling(60).std().fillna(0) * 100
        
        # Volume features
        df['volume_sma_20'] = df['tick_volume'].rolling(20).mean().fillna(1)
        df['volume_ratio'] = (df['tick_volume'] / df['volume_sma_20']).fillna(1.0).clip(0.1, 5.0)
        
        logger.info("✅ Indicadores premium calculados")
        return df
        
    except Exception as e:
        logger.error(f"❌ Erro nos indicadores: {e}")
        return df

def save_hybrid_dataset(df: pd.DataFrame, symbol: str) -> str:
    """Salvar dataset híbrido V2"""
    logger = setup_logging()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Criar diretórios
    for dir_name in ['data', 'data_cache', 'metadata']:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    
    base_name = f"{symbol}_HYBRID_V2_3Y_1MIN_{timestamp}"
    csv_file = os.path.join('data', f"{base_name}.csv")
    pkl_file = os.path.join('data_cache', f"{base_name}.pkl")
    metadata_file = os.path.join('metadata', f"{base_name}_info.txt")
    
    # Otimizar dataset
    df_optimized = df.copy()
    df_optimized['time'] = pd.to_datetime(df_optimized['time'])
    
    # Garantir formato XXXX.XX final
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in df_optimized.columns:
            df_optimized[col] = df_optimized[col].round(2)
    
    # Otimizar tipos
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
    
    # Estatísticas
    real_count = len(df[df['data_type'].str.contains('REAL', na=False)])
    synthetic_count = len(df[df['data_type'].str.contains('SYNTHETIC', na=False)])
    interpolated_count = len(df[df['data_type'].str.contains('INTERPOLATED', na=False)])
    
    # Criar metadata
    metadata = f"""
GOLD HYBRID DATASET V2 - ULTRA-REALISTA
======================================
Criado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Símbolo: {symbol}
Timeframe: 1 minuto
Período: {df['time'].min()} → {df['time'].max()}

ESTATÍSTICAS:
- Total de barras: {len(df):,}
- Dados reais (interpolados): {real_count + interpolated_count:,}
- Dados sintéticos V2: {synthetic_count:,}
- Preço inicial: ${df['close'].iloc[0]:.2f}
- Preço final: ${df['close'].iloc[-1]:.2f}
- Variação total: {((df['close'].iloc[-1]/df['close'].iloc[0])-1)*100:+.2f}%
- Volume médio: {df['tick_volume'].mean():.0f}

QUALIDADE V2:
- Algoritmo: Brownian Bridge + Volatilidade horária
- Interpolação: 5m→1m realista
- Formato: XXXX.XX garantido
- Indicadores: Premium otimizados
- Diversidade esperada: >80%

ARQUIVOS:
- CSV: {csv_file}
- Pickle: {pkl_file}
- Metadata: {metadata_file}
"""
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write(metadata)
    
    logger.info(f"💾 Dataset V2 salvo: {csv_file}")
    return csv_file

def main():
    """Criar dataset híbrido ultra-realista V2"""
    logger = setup_logging()
    
    logger.info("🚀 CRIANDO DATASET HÍBRIDO V2 ULTRA-REALISTA")
    logger.info("="*60)
    
    # 1. DOWNLOAD DADOS REAIS 5M (últimos 6 meses)
    logger.info("📥 FASE 1: Dados reais 5m")
    df_real_5m = download_real_5m_data("GC=F", months_back=6)
    
    if not df_real_5m.empty:
        # 2. INTERPOLAR 5M → 1M
        logger.info("🧮 FASE 2: Interpolação 5m→1m")
        df_real_1m = interpolate_5m_to_1m_realistic(df_real_5m)
    else:
        df_real_1m = pd.DataFrame()
    
    # 3. DADOS SINTÉTICOS HISTÓRICOS (V2 melhorado)
    logger.info("🧠 FASE 3: Dados sintéticos V2")
    df_daily = download_daily_historical("GC=F", years_back=3)
    
    if not df_daily.empty:
        df_synthetic_1m = enhance_synthetic_algorithm(df_daily)
    else:
        logger.error("❌ Falha no download histórico")
        return
    
    # 4. COMBINAR DATASETS
    logger.info("🔗 FASE 4: Combinação final")
    
    datasets = []
    if not df_synthetic_1m.empty:
        datasets.append(df_synthetic_1m)
    if not df_real_1m.empty:
        datasets.append(df_real_1m)
    
    if datasets:
        df_final = pd.concat(datasets, ignore_index=True)
        df_final = df_final.sort_values('time').reset_index(drop=True)
        df_final = df_final.drop_duplicates(subset=['time'], keep='last')
        
        # 5. ADICIONAR INDICADORES PREMIUM
        logger.info("💎 FASE 5: Indicadores premium")
        df_final = calculate_premium_indicators(df_final)
        
        # 6. SALVAR DATASET FINAL
        save_path = save_hybrid_dataset(df_final, "GC=F")
        
        logger.info("🎉 DATASET HÍBRIDO V2 CRIADO COM SUCESSO!")
        logger.info(f"📊 Total: {len(df_final):,} barras")
        logger.info(f"📁 Arquivo: {save_path}")
        
        return save_path
        
    else:
        logger.error("❌ Falha na criação do dataset")
        return None

if __name__ == "__main__":
    main()