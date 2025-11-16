#!/usr/bin/env python3
"""
🚀 DATASET V3 REALISTA - VERSÃO CORRIGIDA
Criação de dataset realisticamente calibrado - BUG FREE
"""

import os
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from typing import List, Dict, Any

def setup_logging():
    """Setup logging para debug"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def create_realistic_dataset_v3_fixed():
    """
    Criar dataset V3 realista - VERSÃO CORRIGIDA
    """
    logger = setup_logging()
    logger.info("🚀 CRIANDO DATASET V3 FIXED - REALISTA")
    logger.info("="*70)
    
    # Download dados históricos diários
    logger.info("📥 DOWNLOAD HISTÓRICO DIÁRIO: GC=F")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)  # 3 anos
    logger.info(f"📅 Período: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    
    ticker = yf.Ticker("GC=F")
    df_daily = ticker.history(start=start_date, end=end_date, interval="1d")
    df_daily.reset_index(inplace=True)
    df_daily.rename(columns={
        'Date': 'time',
        'Open': 'open',
        'High': 'high', 
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)
    
    logger.info(f"✅ HISTÓRICO DIÁRIO: {len(df_daily)} dias")
    
    # Algoritmo V3 CORRIGIDO
    logger.info("🧠 FASE: Algoritmo V3 CORRIGIDO")
    df_1m_fixed = create_realistic_algorithm_v3_fixed(df_daily, logger)
    
    # Adicionar indicadores
    logger.info("📊 FASE: Indicadores técnicos premium")
    df_with_indicators = add_premium_indicators_v3(df_1m_fixed, logger)
    
    # Salvar dataset
    logger.info("💾 FASE: Salvando dataset V3 FIXED")
    csv_path = save_realistic_dataset_v3_fixed(df_with_indicators, "GC=F")
    
    logger.info(f"🎉 DATASET V3 FIXED CRIADO COM SUCESSO!")
    logger.info(f"📁 Arquivo: {csv_path}")
    return csv_path

def create_realistic_algorithm_v3_fixed(df_daily: pd.DataFrame, logger):
    """
    Algoritmo V3 CORRIGIDO - sem bugs de loop infinito
    """
    logger.info("🚀 ALGORITMO V3: CORRIGIDO E CALIBRADO")
    
    # PARÂMETROS REALISTAS
    PARAMS = {
        'target_minute_vol': 0.00026352,  # Volatilidade realista por minuto
        'max_static_ratio': 0.08,         # Máximo 8% estáticas
        'extreme_move_ratio': 0.012,      # 1.2% movimentos >0.1%
        'max_static_sequence': 45,        # Máximo 45 barras estáticas consecutivas
        'tick_size': 0.01,                # Tick size ouro
        'force_move_probability': 0.02,   # 2% chance movimento forçado por minuto
    }
    
    logger.info(f"🎯 PARÂMETROS V3 FIXED:")
    logger.info(f"   Vol/min: {PARAMS['target_minute_vol']*100:.4f}%")
    logger.info(f"   Static max: {PARAMS['max_static_ratio']*100:.1f}%")
    logger.info(f"   Extreme: {PARAMS['extreme_move_ratio']*100:.2f}%")
    logger.info(f"   Max seq: {PARAMS['max_static_sequence']}")
    
    df_1m_result = []
    
    for day_idx, day_row in df_daily.iterrows():
        day_start = pd.to_datetime(day_row['time'])
        daily_range = day_row['high'] - day_row['low']
        daily_vol = daily_range / day_row['open'] if day_row['open'] > 0 else 0.01
        
        # Escalar volatilidade para nível realista
        target_daily_vol = PARAMS['target_minute_vol'] * np.sqrt(1440)
        vol_multiplier = max(0.5, min(3.0, target_daily_vol / daily_vol)) if daily_vol > 0 else 1.0
        
        # Regime de volatilidade
        vol_regime = np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
        vol_regime_mult = {'low': 0.6, 'medium': 1.0, 'high': 1.8}[vol_regime]
        
        logger.info(f"Dia {day_idx+1}: Vol regime={vol_regime} (mult={vol_regime_mult:.1f}x)")
        
        # Estado para controle de sequências
        current_price = day_row['open']
        static_sequence_length = 0
        last_movement_was_forced = False
        
        # Gerar 1440 barras por dia
        for minute in range(1440):
            minute_time = day_start + timedelta(minutes=minute)
            hour = minute_time.hour
            
            # Volatilidade dinâmica por hora (simplificada)
            hour_multiplier = 1.0 + 0.3 * np.sin(2 * np.pi * hour / 24)  # Padrão senoidal
            
            # Volatilidade atual
            base_vol = daily_vol / 1440 * vol_multiplier * vol_regime_mult
            current_vol = base_vol * hour_multiplier
            
            # GERAÇÃO DE MOVIMENTO DE PREÇO
            
            # 1. Drift para target do dia
            daily_progress = minute / 1440.0
            target_price = day_row['open'] + (day_row['close'] - day_row['open']) * daily_progress
            drift = (target_price - current_price) * 0.001  # Mean reversion fraco
            
            # 2. Componente browniano
            brownian = np.random.normal(0, current_price * current_vol)
            
            # 3. Jumps ocasionais
            jump_size = 0
            if np.random.random() < PARAMS['extreme_move_ratio'] / 1440:
                jump_size = np.random.choice([-1, 1]) * current_price * np.random.uniform(0.001, 0.005)
                logger.info(f"   JUMP! Dia {day_idx+1}, min {minute}: {jump_size/current_price*100:.3f}%")
            
            # 4. Movimento total inicial
            price_change = drift + brownian + jump_size
            
            # 5. CONTROLE INTELIGENTE DE SEQUÊNCIAS ESTÁTICAS
            force_movement = False
            
            # Verificar se precisa forçar movimento
            if static_sequence_length >= PARAMS['max_static_sequence'] and not last_movement_was_forced:
                force_movement = True
                # Movimento mínimo garantido
                min_move = PARAMS['tick_size'] * np.random.choice([-1, 1])
                price_change = min_move
                logger.info(f"   FORÇA! Dia {day_idx+1}, seq {static_sequence_length}: {min_move:.3f}")
            
            # 6. Anti-estático ocasional (baixa probabilidade)
            elif abs(price_change) < PARAMS['tick_size'] * 0.5 and np.random.random() < PARAMS['force_move_probability']:
                min_move = PARAMS['tick_size'] * np.random.choice([-1, 1])
                price_change = min_move
            
            # 7. Aplicar movimento
            new_price = current_price + price_change
            
            # 8. Limitações realistas
            day_low = day_row['low'] - daily_range * 0.1
            day_high = day_row['high'] + daily_range * 0.1
            new_price = max(day_low, min(day_high, new_price))
            
            # 9. Tick size enforcement
            new_price = round(new_price / PARAMS['tick_size']) * PARAMS['tick_size']
            new_price = round(new_price, 2)
            
            # 10. GERAR OHLC
            open_price = current_price
            close_price = new_price
            
            # High/Low com micro-volatilidade
            if abs(price_change) > PARAMS['tick_size']:
                intra_vol = current_vol * np.random.uniform(0.5, 1.5)
                high_extra = abs(np.random.normal(0, current_price * intra_vol * 0.5))
                low_extra = abs(np.random.normal(0, current_price * intra_vol * 0.5))
            else:
                high_extra = low_extra = PARAMS['tick_size'] * np.random.uniform(0, 0.3)
            
            high_price = max(open_price, close_price) + high_extra
            low_price = min(open_price, close_price) - low_extra
            
            # Garantir limites
            high_price = round(min(high_price, day_high), 2)
            low_price = round(max(low_price, day_low), 2)
            
            # 11. Volume realista
            base_volume = max(1, int(day_row.get('volume', 1000) / 1440))
            volume_mult = hour_multiplier * np.random.uniform(0.3, 3.0)
            volume = max(1, int(base_volume * volume_mult))
            
            # 12. ATUALIZAR CONTROLE DE SEQUÊNCIAS
            movement_size = abs(close_price - current_price)
            
            if movement_size < PARAMS['tick_size'] * 0.5:
                # Movimento insignificante - incrementar contador
                static_sequence_length += 1
            else:
                # Movimento significativo - resetar contador
                static_sequence_length = 0
            
            # Marcar se movimento foi forçado
            last_movement_was_forced = force_movement
            
            # 13. Adicionar barra ao resultado
            df_1m_result.append({
                'time': minute_time,
                'open': round(open_price, 2),
                'high': round(high_price, 2), 
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'tick_volume': volume,
                'spread': 0,
                'real_volume': volume,
                'data_type': 'REALISTIC_V3_FIXED',
                'vol_regime': vol_regime,
                'static_seq': static_sequence_length
            })
            
            current_price = close_price
        
        # Garantir close do dia correto
        if df_1m_result:
            df_1m_result[-1]['close'] = round(day_row['close'], 2)
    
    df_result = pd.DataFrame(df_1m_result)
    logger.info(f"✅ ALGORITMO V3 FIXED: {len(df_result):,} barras criadas")
    
    # Validar resultado
    validate_fixed_dataset(df_result, PARAMS, logger)
    
    return df_result

def validate_fixed_dataset(df: pd.DataFrame, params: dict, logger):
    """Validar dataset V3 FIXED"""
    logger.info("🔍 VALIDANDO DATASET V3 FIXED...")
    
    # Calcular estatísticas
    df['returns'] = df['close'].pct_change().fillna(0)
    returns_clean = df['returns'][df['returns'] != 0]
    
    static_ratio = (df['returns'] == 0).mean()
    vol_minute = returns_clean.std()
    extreme_ratio = (np.abs(df['returns']) > 0.001).mean()
    
    # Sequências estáticas
    max_seq = df['static_seq'].max()
    
    logger.info(f"📊 RESULTADOS VALIDAÇÃO:")
    logger.info(f"   Barras estáticas: {static_ratio*100:.1f}% (target: <{params['max_static_ratio']*100:.1f}%)")
    logger.info(f"   Vol/minuto: {vol_minute*100:.4f}% (target: {params['target_minute_vol']*100:.4f}%)")
    logger.info(f"   Movimentos >0.1%: {extreme_ratio*100:.2f}% (target: {params['extreme_move_ratio']*100:.2f}%)")
    logger.info(f"   Seq máxima: {max_seq} (limit: {params['max_static_sequence']})")
    
    # Avaliar qualidade
    issues = []
    if static_ratio > params['max_static_ratio'] * 1.5:
        issues.append("Muitas barras estáticas")
    if vol_minute < params['target_minute_vol'] * 0.5:
        issues.append("Volatilidade muito baixa")
    if extreme_ratio < params['extreme_move_ratio'] * 0.5:
        issues.append("Poucos movimentos extremos")
    if max_seq > params['max_static_sequence'] * 2:
        issues.append("Sequências muito longas")
    
    if issues:
        logger.warning(f"⚠️ PROBLEMAS: {', '.join(issues)}")
    else:
        logger.info(f"✅ QUALIDADE: Dataset aprovado!")

def add_premium_indicators_v3(df: pd.DataFrame, logger):
    """Adicionar indicadores técnicos premium V3"""
    logger.info("📈 CALCULANDO INDICADORES PREMIUM V3...")
    
    try:
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = (100 - (100 / (1 + rs))).fillna(50)
        
        # Médias móveis
        for period in [5, 10, 20, 50, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean().fillna(df['close'])
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean().fillna(df['close'])
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = (ema_12 - ema_26).fillna(0)
        df['macd_signal'] = df['macd'].ewm(span=9).mean().fillna(0)
        df['macd_histogram'] = (df['macd'] - df['macd_signal']).fillna(0)
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['bb_upper'] = (sma_20 + 2 * std_20).fillna(df['close'])
        df['bb_middle'] = sma_20.fillna(df['close'])
        df['bb_lower'] = (sma_20 - 2 * std_20).fillna(df['close'])
        
        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr_14'] = df['tr'].rolling(14).mean().fillna(df['tr'])
        
        # Volume indicators
        df['volume_sma_20'] = df['tick_volume'].rolling(20).mean().fillna(df['tick_volume'])
        df['volume_ratio'] = (df['tick_volume'] / df['volume_sma_20']).fillna(1.0).clip(0.1, 5.0)
        
        # Volatilidade
        df['volatility_20'] = df['close'].pct_change().rolling(20).std().fillna(0) * 100
        
        logger.info("✅ Indicadores premium V3 calculados")
        return df
        
    except Exception as e:
        logger.error(f"❌ Erro nos indicadores V3: {e}")
        return df

def save_realistic_dataset_v3_fixed(df: pd.DataFrame, symbol: str) -> str:
    """Salvar dataset V3 FIXED"""
    logger = setup_logging()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Criar diretórios
    for dir_name in ['data', 'data_cache', 'metadata']:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    
    base_name = f"{symbol}_REALISTIC_V3_FIXED_{timestamp}"
    csv_file = os.path.join('data', f"{base_name}.csv")
    
    # Otimizar dataset
    df_optimized = df.copy()
    df_optimized['time'] = pd.to_datetime(df_optimized['time'])
    
    # Garantir formato XXXX.XX
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in df_optimized.columns:
            df_optimized[col] = df_optimized[col].round(2)
    
    # Salvar CSV
    df_optimized.to_csv(csv_file, index=False)
    
    # Estatísticas finais
    returns = df['close'].pct_change().dropna()
    static_ratio = (returns == 0).mean()
    vol_per_minute = returns.std()
    extreme_ratio = (np.abs(returns) > 0.001).mean()
    
    logger.info(f"💾 Dataset V3 FIXED salvo: {csv_file}")
    logger.info(f"📊 ESTATÍSTICAS FINAIS:")
    logger.info(f"   Barras: {len(df):,}")
    logger.info(f"   Estáticas: {static_ratio*100:.1f}%")
    logger.info(f"   Vol/min: {vol_per_minute*100:.4f}%")
    logger.info(f"   Extremos: {extreme_ratio*100:.2f}%")
    
    return csv_file

if __name__ == "__main__":
    dataset_path = create_realistic_dataset_v3_fixed()
    print(f"\n🏁 DATASET V3 FIXED CRIADO: {dataset_path}")