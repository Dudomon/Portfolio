#!/usr/bin/env python3
"""
🔄 CRIAR DATASET 1M SINTÉTICO A PARTIR DE 5M
Transformar 1.1M barras de 5m em ~5.5M barras de 1m sintéticas

ESTRATÉGIA:
- Cada barra 5m vira 5 barras 1m
- Interpolação inteligente usando padrões realistas
- Preservar características técnicas importantes
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
import random

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def load_5m_dataset():
    """Carregar o dataset massivo de 5m existente"""
    logger = logging.getLogger(__name__)
    
    # Procurar pelo dataset de 5m (completo agora que sabemos que funciona)
    data_files = [
        "data_cache/GC=F_YAHOO_DAILY_CACHE_20250711_041924.pkl",  # Dataset completo 1.1M
        "data/test_5m_5000bars.pkl",  # Dataset teste 5k barras como fallback
        "data_cache/GC=F_YAHOO_DAILY_CACHE_20250711_041924_BACKUP.pkl"
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            logger.info(f"📊 Carregando dataset 5m: {file_path}")
            if file_path.endswith('.pkl'):
                df = pd.read_pickle(file_path)
            else:
                df = pd.read_csv(file_path)
            logger.info(f"✅ Carregado: {len(df)} barras 5m")
            return df
    
    raise FileNotFoundError("❌ Dataset 5m não encontrado!")

def create_1m_from_5m_bar(row_5m, bar_index):
    """
    Criar 5 barras de 1m a partir de 1 barra de 5m
    Usando interpolação inteligente + ruído realista
    """
    
    # Extrair dados da barra 5m
    time_5m = row_5m['time'] if 'time' in row_5m else row_5m.name
    open_5m = row_5m['open']
    high_5m = row_5m['high'] 
    low_5m = row_5m['low']
    close_5m = row_5m['close']
    volume_5m = row_5m.get('tick_volume', row_5m.get('volume', 1000))
    
    # Gerar 5 timestamps de 1m
    base_time = pd.to_datetime(time_5m)
    timestamps_1m = [base_time + timedelta(minutes=i) for i in range(5)]
    
    # 🎯 ESTRATÉGIA DE INTERPOLAÇÃO INTELIGENTE
    # Criar path realista de preços dentro da barra 5m
    
    # Definir pontos de controle (open, high, low, close)
    price_points = [open_5m]
    
    # Decidir quando hit high e low (padrões aleatórios realistas)
    pattern = random.choice([
        'up_then_down',     # Sobe depois desce
        'down_then_up',     # Desce depois sobe  
        'trending_up',      # Tendência de alta
        'trending_down',    # Tendência de baixa
        'choppy'           # Lateral com volatilidade
    ])
    
    if pattern == 'up_then_down':
        price_points.extend([high_5m * 0.8 + open_5m * 0.2, high_5m, low_5m * 0.6 + high_5m * 0.4, close_5m])
    elif pattern == 'down_then_up': 
        price_points.extend([low_5m * 0.7 + open_5m * 0.3, low_5m, high_5m * 0.5 + low_5m * 0.5, close_5m])
    elif pattern == 'trending_up':
        mid1 = open_5m + (close_5m - open_5m) * 0.3
        mid2 = open_5m + (close_5m - open_5m) * 0.7
        price_points.extend([mid1, mid2, high_5m, close_5m])
    elif pattern == 'trending_down':
        mid1 = open_5m + (close_5m - open_5m) * 0.3  
        mid2 = open_5m + (close_5m - open_5m) * 0.7
        price_points.extend([mid1, low_5m, mid2, close_5m])
    else:  # choppy
        mid = (high_5m + low_5m) / 2
        price_points.extend([mid * 1.02, mid * 0.98, mid * 1.01, close_5m])
    
    # Garantir que high/low sejam respeitados
    max_price = max(price_points)
    min_price = min(price_points)
    
    if max_price < high_5m:
        # Ajustar um ponto para hit o high
        max_idx = price_points.index(max_price)
        price_points[max_idx] = high_5m
    
    if min_price > low_5m:
        # Ajustar um ponto para hit o low  
        min_idx = price_points.index(min_price)
        price_points[min_idx] = low_5m
    
    # Criar barras 1m
    bars_1m = []
    
    for i in range(5):
        # Preços OHLC para esta barra 1m
        open_1m = price_points[i]
        close_1m = price_points[i + 1] if i < 4 else close_5m
        
        # High/Low com pequenas variações aleatórias
        spread = abs(close_1m - open_1m)
        noise_factor = 0.3  # 30% de ruído
        
        high_1m = max(open_1m, close_1m) + spread * noise_factor * random.random()
        low_1m = min(open_1m, close_1m) - spread * noise_factor * random.random()
        
        # Garantir que não ultrapasse limites da barra 5m
        high_1m = min(high_1m, high_5m)
        low_1m = max(low_1m, low_5m)
        
        # Volume distribuído
        volume_1m = volume_5m / 5 * (0.7 + 0.6 * random.random())  # Variação 70-130%
        
        # Criar registro da barra 1m
        bar_1m = {
            'time': timestamps_1m[i],
            'open_1m': round(open_1m, 2),
            'high_1m': round(high_1m, 2), 
            'low_1m': round(low_1m, 2),
            'close_1m': round(close_1m, 2),
            'volume_1m': int(volume_1m),
            'source_bar_5m': bar_index,  # Rastreabilidade
            'pattern': pattern
        }
        
        bars_1m.append(bar_1m)
    
    return bars_1m

def add_technical_indicators_1m_fast(df):
    """Adicionar indicadores técnicos otimizados para 1m"""
    logger = logging.getLogger(__name__)
    
    try:
        # Returns
        df['returns_1m'] = df['close_1m'].pct_change().fillna(0)
        
        # RSI rápido (períodos menores para 1m)
        df['rsi_7_1m'] = calculate_rsi_fast(df['close_1m'], 7)
        df['rsi_14_1m'] = calculate_rsi_fast(df['close_1m'], 14)
        
        # Médias móveis curtas
        df['sma_5_1m'] = df['close_1m'].rolling(5).mean()
        df['sma_20_1m'] = df['close_1m'].rolling(20).mean()
        df['ema_9_1m'] = df['close_1m'].ewm(span=9).mean()
        
        # Bollinger Bands rápido
        bb_period = 12
        bb_sma = df['close_1m'].rolling(bb_period).mean()
        bb_std = df['close_1m'].rolling(bb_period).std()
        df['bb_upper_1m'] = bb_sma + (bb_std * 2)
        df['bb_lower_1m'] = bb_sma - (bb_std * 2)
        df['bb_position_1m'] = ((df['close_1m'] - df['bb_lower_1m']) / 
                               (df['bb_upper_1m'] - df['bb_lower_1m'])).clip(0, 1).fillna(0.5)
        
        # Volatilidade
        df['volatility_10_1m'] = df['returns_1m'].rolling(10).std() * 100
        
        # Trend strength
        df['trend_strength_1m'] = df['returns_1m'].rolling(10).mean()
        
        # Momentum
        df['momentum_5_1m'] = df['close_1m'] / df['close_1m'].shift(5) - 1
        
        logger.info("✅ Indicadores técnicos 1m adicionados")
        return df
        
    except Exception as e:
        logger.error(f"❌ Erro ao calcular indicadores: {e}")
        return df

def calculate_rsi_fast(prices, period):
    """RSI otimizado para performance"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def create_massive_1m_dataset():
    """Criar dataset massivo de 1m a partir de dados 5m"""
    logger = setup_logging()
    
    logger.info("🔄 CRIANDO DATASET MASSIVO 1M A PARTIR DE 5M")
    logger.info("=" * 60)
    
    # Carregar dataset 5m
    df_5m = load_5m_dataset()
    logger.info(f"📊 Dataset 5m carregado: {len(df_5m)} barras")
    
    # Estimar resultado
    estimated_1m_bars = len(df_5m) * 5
    estimated_size_mb = estimated_1m_bars * 25 * 8 / (1024 * 1024)  # ~25 cols, 8 bytes cada
    logger.info(f"🎯 Resultado esperado: ~{estimated_1m_bars:,} barras 1m (~{estimated_size_mb:.1f} MB)")
    
    # Processar em chunks para evitar memory overflow
    chunk_size = 10000  # Processar 10k barras 5m por vez
    all_bars_1m = []
    
    for chunk_start in range(0, len(df_5m), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(df_5m))
        chunk_df = df_5m.iloc[chunk_start:chunk_end]
        
        logger.info(f"🔄 Processando chunk {chunk_start:,}-{chunk_end:,} ({len(chunk_df)} barras 5m)")
        
        # Processar cada barra 5m
        chunk_bars_1m = []
        for idx, row in chunk_df.iterrows():
            try:
                bars_1m = create_1m_from_5m_bar(row, idx)
                chunk_bars_1m.extend(bars_1m)
            except Exception as e:
                logger.warning(f"⚠️ Erro processando barra {idx}: {e}")
                continue
        
        all_bars_1m.extend(chunk_bars_1m)
        
        if len(all_bars_1m) % 50000 == 0:
            logger.info(f"✅ Progresso: {len(all_bars_1m):,} barras 1m criadas")
    
    # Converter para DataFrame
    logger.info("🔄 Convertendo para DataFrame...")
    df_1m = pd.DataFrame(all_bars_1m)
    
    # Ordenar por tempo
    df_1m = df_1m.sort_values('time').reset_index(drop=True)
    
    # Adicionar indicadores técnicos
    logger.info("🔄 Calculando indicadores técnicos...")
    df_1m = add_technical_indicators_1m_fast(df_1m)
    
    # Salvar dataset
    logger.info("💾 Salvando dataset...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Salvar como PKL (comprimido)
    pkl_path = f"data/GOLD_1M_MASSIVE_SYNTHETIC_{timestamp}.pkl"
    df_1m.to_pickle(pkl_path)
    
    # Estatísticas finais
    actual_size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
    
    logger.info("🎉 DATASET MASSIVO 1M CRIADO COM SUCESSO!")
    logger.info(f"📊 Total de barras: {len(df_1m):,}")
    logger.info(f"📅 Período: {df_1m['time'].min()} até {df_1m['time'].max()}")
    logger.info(f"📁 Arquivo: {pkl_path}")
    logger.info(f"💾 Tamanho: {actual_size_mb:.1f} MB")
    logger.info(f"🎯 Multiplicação: {len(df_1m) / len(df_5m):.1f}x")
    
    return df_1m, pkl_path

def main():
    try:
        df_1m, pkl_path = create_massive_1m_dataset()
        
        print(f"\n✅ SUCESSO! Dataset massivo 1m criado:")
        print(f"📊 {len(df_1m):,} barras de 1 minuto")
        print(f"📁 Salvo em: {pkl_path}")
        print(f"🎯 Pronto para curriculum learning!")
        
        # Sample dos dados
        print(f"\n📈 Amostra dos dados:")
        print(df_1m[['time', 'open_1m', 'close_1m', 'rsi_14_1m', 'bb_position_1m']].head(10))
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()