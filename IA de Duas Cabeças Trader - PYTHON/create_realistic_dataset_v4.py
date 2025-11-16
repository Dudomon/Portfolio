#!/usr/bin/env python3
"""
🚀 DATASET V4 - NOVA ABORDAGEM REALISTA
Interpolação inteligente de dados reais ao invés de geração sintética
"""

import os
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def create_realistic_dataset_v4():
    """
    NOVA ABORDAGEM V4: Usar dados REAIS como base principal
    """
    logger = setup_logging()
    logger.info("🚀 DATASET V4 - NOVA ABORDAGEM REALISTA")
    logger.info("="*70)
    logger.info("🎯 ESTRATÉGIA V4: Dados reais + interpolação inteligente")
    logger.info("   - Base: Dados reais 5min do Yahoo Finance")
    logger.info("   - Método: Interpolação realista para 1min")
    logger.info("   - Controle: Sem forçar movimentos artificiais")
    
    # 1. Download dados reais 5 minutos (mais confiáveis que 1min)
    logger.info("📥 DOWNLOAD DADOS REAIS: GC=F 5min")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)  # 60 dias de dados reais
    logger.info(f"📅 Período: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    
    ticker = yf.Ticker("GC=F")
    df_5min = ticker.history(start=start_date, end=end_date, interval="5m")
    df_5min.reset_index(inplace=True)
    df_5min.rename(columns={
        'Datetime': 'time',
        'Open': 'open',
        'High': 'high',
        'Low': 'low', 
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)
    
    logger.info(f"✅ DADOS REAIS 5min: {len(df_5min):,} barras")
    
    if len(df_5min) < 1000:
        logger.error("❌ Dados insuficientes do Yahoo Finance")
        return None
    
    # 2. Expandir dados reais para 3 anos (replicar padrões)
    logger.info("🔄 EXPANDINDO PARA 3 ANOS COM PADRÕES REAIS")
    df_expanded = expand_real_patterns(df_5min, logger)
    
    # 3. Interpolação inteligente 5min -> 1min
    logger.info("🧠 INTERPOLAÇÃO INTELIGENTE 5min → 1min")
    df_1min = intelligent_interpolation_v4(df_expanded, logger)
    
    # 4. Adicionar indicadores
    logger.info("📊 INDICADORES TÉCNICOS V4")
    df_with_indicators = add_indicators_v4(df_1min, logger)
    
    # 5. Salvar dataset
    logger.info("💾 SALVANDO DATASET V4")
    csv_path = save_dataset_v4(df_with_indicators, "GC=F")
    
    logger.info(f"🎉 DATASET V4 CRIADO COM SUCESSO!")
    logger.info(f"📁 Arquivo: {csv_path}")
    return csv_path

def expand_real_patterns(df_5min: pd.DataFrame, logger):
    """
    Expandir 60 dias de dados reais para 3 anos replicando padrões
    """
    logger.info("🔄 EXPANDINDO PADRÕES REAIS...")
    
    # Calcular estatísticas dos dados reais
    df_5min['returns'] = df_5min['close'].pct_change().fillna(0)
    real_vol = df_5min['returns'].std()
    real_drift = df_5min['returns'].mean()
    
    logger.info(f"   Volatilidade real: {real_vol*100:.4f}%")
    logger.info(f"   Drift real: {real_drift*100:.6f}%")
    
    # Replicar padrões com variações
    df_expanded = []
    base_price = df_5min['close'].iloc[-1]  # Último preço real
    
    # Replicar padrões múltiplas vezes com variações
    for cycle in range(18):  # 18 ciclos de 60 dias = ~3 anos
        cycle_multiplier = 1.0 + np.random.uniform(-0.2, 0.2)  # ±20% variação
        vol_multiplier = 1.0 + np.random.uniform(-0.3, 0.5)   # Volatilidade variável
        
        logger.info(f"   Ciclo {cycle+1}: mult_preço={cycle_multiplier:.3f}, mult_vol={vol_multiplier:.3f}")
        
        for i, row in df_5min.iterrows():
            # Replicar estrutura OHLC com variações
            time_offset = timedelta(days=60 * cycle)
            new_time = row['time'] + time_offset
            
            # Escalar preços mantendo estrutura
            price_ratio = base_price / df_5min['close'].iloc[0]
            
            new_open = row['open'] * price_ratio * cycle_multiplier
            new_high = row['high'] * price_ratio * cycle_multiplier  
            new_low = row['low'] * price_ratio * cycle_multiplier
            new_close = row['close'] * price_ratio * cycle_multiplier
            
            # Adicionar ruído controlado
            noise = np.random.normal(0, new_close * real_vol * vol_multiplier * 0.1)
            
            df_expanded.append({
                'time': new_time,
                'open': round(new_open + noise, 2),
                'high': round(new_high + abs(noise), 2),
                'low': round(new_low - abs(noise), 2),
                'close': round(new_close + noise, 2),
                'volume': int(row['volume'] * np.random.uniform(0.5, 2.0)),
                'cycle': cycle,
                'vol_mult': vol_multiplier
            })
        
        base_price *= cycle_multiplier  # Atualizar preço base
    
    df_result = pd.DataFrame(df_expanded)
    logger.info(f"✅ DADOS EXPANDIDOS: {len(df_result):,} barras 5min")
    
    return df_result

def intelligent_interpolation_v4(df_5min: pd.DataFrame, logger):
    """
    Interpolação inteligente 5min -> 1min baseada em padrões reais
    """
    logger.info("🧠 INTERPOLAÇÃO INTELIGENTE 5min → 1min")
    
    df_1min = []
    
    for i, row in df_5min.iterrows():
        base_time = pd.to_datetime(row['time'])
        
        # Dados da barra 5min
        open_price = row['open']
        high_price = row['high']
        low_price = row['low']
        close_price = row['close']
        volume_5min = row['volume']
        
        # Calcular movimento total da barra
        total_move = close_price - open_price
        price_range = high_price - low_price
        
        # Gerar 5 barras de 1 minuto
        for minute in range(5):
            minute_time = base_time + timedelta(minutes=minute)
            
            # Progressão temporal dentro da barra 5min
            progress = (minute + 1) / 5.0
            
            # INTERPOLAÇÃO REALISTA baseada em microestrutura
            
            # 1. Preço base (interpolação linear suavizada)
            linear_price = open_price + (total_move * progress)
            
            # 2. Adicionar padrão de alta/baixa realista
            # Simular que high/low acontecem em momentos aleatórios
            high_moment = np.random.uniform(0, 1)
            low_moment = np.random.uniform(0, 1)
            
            # 3. Ajustar baseado em onde estamos na progressão
            if progress <= high_moment:
                high_influence = (progress / high_moment) if high_moment > 0 else 0
                price_bias = (high_price - linear_price) * high_influence * 0.3
            else:
                price_bias = 0
                
            if progress <= low_moment:
                low_influence = (progress / low_moment) if low_moment > 0 else 0
                price_bias += (low_price - linear_price) * low_influence * 0.3
            
            # 4. Micro-volatilidade realista
            micro_vol = price_range * 0.1  # 10% do range como micro-volatilidade
            micro_noise = np.random.normal(0, micro_vol)
            
            # 5. Preço final da barra 1min
            if minute == 4:  # Última barra deve fechar no close correto
                bar_close = close_price
            else:
                bar_close = linear_price + price_bias + micro_noise
                bar_close = max(low_price, min(high_price, bar_close))  # Dentro do range
            
            # 6. Gerar OHLC para barra 1min
            if minute == 0:
                bar_open = open_price
            else:
                bar_open = df_1min[-1]['close']  # Close da barra anterior
                
            # High/Low da barra 1min
            intrabar_range = price_range * 0.2 * np.random.uniform(0.5, 1.5)  # Variação
            bar_high = max(bar_open, bar_close) + abs(np.random.normal(0, intrabar_range * 0.5))
            bar_low = min(bar_open, bar_close) - abs(np.random.normal(0, intrabar_range * 0.5))
            
            # Garantir que high/low não saem do range 5min
            bar_high = min(bar_high, high_price)
            bar_low = max(bar_low, low_price)
            
            # Garantir consistência OHLC
            bar_high = max(bar_high, bar_open, bar_close)
            bar_low = min(bar_low, bar_open, bar_close)
            
            # 7. Volume distribuído
            volume_1min = int(volume_5min / 5 * np.random.uniform(0.3, 2.0))
            
            # 8. Arredondar preços
            df_1min.append({
                'time': minute_time,
                'open': round(bar_open, 2),
                'high': round(bar_high, 2),
                'low': round(bar_low, 2),
                'close': round(bar_close, 2),
                'tick_volume': max(1, volume_1min),
                'spread': 0,
                'real_volume': max(1, volume_1min),
                'data_type': 'REAL_INTERPOLATED_V4'
            })
    
    df_result = pd.DataFrame(df_1min)
    logger.info(f"✅ INTERPOLAÇÃO CONCLUÍDA: {len(df_result):,} barras 1min")
    
    # Validar resultado
    validate_v4_quality(df_result, logger)
    
    return df_result

def validate_v4_quality(df: pd.DataFrame, logger):
    """Validar qualidade do dataset V4"""
    logger.info("🔍 VALIDANDO QUALIDADE V4...")
    
    df['returns'] = df['close'].pct_change().fillna(0)
    returns_clean = df['returns'][df['returns'] != 0]
    
    static_ratio = (df['returns'] == 0).mean()
    vol_minute = returns_clean.std()
    extreme_ratio = (np.abs(df['returns']) > 0.001).mean()
    
    logger.info(f"📊 QUALIDADE V4:")
    logger.info(f"   Barras estáticas: {static_ratio*100:.1f}%")
    logger.info(f"   Vol/minuto: {vol_minute*100:.4f}%") 
    logger.info(f"   Movimentos >0.1%: {extreme_ratio*100:.2f}%")
    
    # Analisar sequências estáticas
    sequences = []
    current_len = 0
    for ret in df['returns']:
        if ret == 0:
            current_len += 1
        else:
            if current_len > 0:
                sequences.append(current_len)
            current_len = 0
    
    if sequences:
        max_seq = max(sequences)
        avg_seq = np.mean(sequences)
        logger.info(f"   Seq estática máx: {max_seq}")
        logger.info(f"   Seq estática média: {avg_seq:.1f}")

def add_indicators_v4(df: pd.DataFrame, logger):
    """Adicionar indicadores técnicos V4"""
    logger.info("📈 CALCULANDO INDICADORES V4...")
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = (100 - (100 / (1 + rs))).fillna(50)
    
    # Médias móveis essenciais
    for period in [5, 10, 20, 50, 200]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean().fillna(df['close'])
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean().fillna(df['close'])
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = (ema12 - ema26).fillna(0)
    df['macd_signal'] = df['macd'].ewm(span=9).mean().fillna(0)
    df['macd_histogram'] = (df['macd'] - df['macd_signal']).fillna(0)
    
    # Bollinger Bands
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_upper'] = (sma20 + 2 * std20).fillna(df['close'])
    df['bb_middle'] = sma20.fillna(df['close'])
    df['bb_lower'] = (sma20 - 2 * std20).fillna(df['close'])
    
    # ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr_14'] = df['tr'].rolling(14).mean().fillna(df['tr'])
    
    # Volume
    df['volume_sma_20'] = df['tick_volume'].rolling(20).mean().fillna(df['tick_volume'])
    df['volume_ratio'] = (df['tick_volume'] / df['volume_sma_20']).fillna(1.0)
    
    # Volatilidade
    df['volatility_20'] = df['close'].pct_change().rolling(20).std().fillna(0) * 100
    
    logger.info("✅ Indicadores V4 calculados")
    return df

def save_dataset_v4(df: pd.DataFrame, symbol: str):
    """Salvar dataset V4"""
    logger = setup_logging()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs('data', exist_ok=True)
    
    base_name = f"{symbol}_REALISTIC_V4_{timestamp}"
    csv_file = os.path.join('data', f"{base_name}.csv")
    
    # Otimizar tipos
    df_opt = df.copy()
    df_opt['time'] = pd.to_datetime(df_opt['time'])
    
    # Arredondar preços
    for col in ['open', 'high', 'low', 'close']:
        if col in df_opt.columns:
            df_opt[col] = df_opt[col].round(2)
    
    df_opt.to_csv(csv_file, index=False)
    
    # Stats finais
    returns = df['close'].pct_change().dropna()
    static_ratio = (returns == 0).mean()
    vol_minute = returns.std()
    
    logger.info(f"💾 DATASET V4 SALVO: {csv_file}")
    logger.info(f"📊 STATS FINAIS:")
    logger.info(f"   Total barras: {len(df):,}")
    logger.info(f"   Estáticas: {static_ratio*100:.1f}%")
    logger.info(f"   Vol/min: {vol_minute*100:.4f}%")
    
    return csv_file

if __name__ == "__main__":
    dataset_path = create_realistic_dataset_v4()
    print(f"\n🏁 DATASET V4 CRIADO: {dataset_path}")