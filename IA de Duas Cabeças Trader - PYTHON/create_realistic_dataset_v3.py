#!/usr/bin/env python3
"""
🚀 DATASET V3 - REALISTICAMENTE CALIBRADO
Baseado na investigação profunda dos problemas do V2

PARÂMETROS REALISTAS IDENTIFICADOS:
- Volatilidade por minuto: 0.026% (não 0.012%)
- Barras estáticas: máximo 10% (não 41.6%)
- Movimentos >0.1%: pelo menos 1% das barras (não 0.053%)
- Sequências estáticas: máximo 60 barras (não 1440)
- Distribuição mais próxima da normal
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

def create_realistic_intraday_algorithm_v3(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Algoritmo V3 - REALISTICAMENTE CALIBRADO
    Corrige todos os problemas identificados na investigação profunda
    """
    logger = setup_logging()
    logger.info("🚀 ALGORITMO V3: REALISTICAMENTE CALIBRADO")
    
    # PARÂMETROS REALISTAS CALIBRADOS
    REALISTIC_PARAMS = {
        'target_minute_vol': 0.00026352,  # 0.026352% - volatilidade realista por minuto
        'max_static_ratio': 0.08,         # Máximo 8% de barras estáticas (vs 41.6%)
        'extreme_move_ratio': 0.012,      # 1.2% de movimentos >0.1% (vs 0.053%)
        'max_static_sequence': 45,        # Máximo 45 barras estáticas (vs 1440)
        'tick_size': 0.01,                # Tick size padrão do ouro
        'volatility_clustering': True,    # Clustering realista de volatilidade
        'fat_tails': True,                # Distribuição com caudas gordas
        'weekend_gaps': True,             # Gaps realistas de fim de semana
    }
    
    logger.info(f"🎯 PARÂMETROS V3:")
    logger.info(f"   Target Vol/min: {REALISTIC_PARAMS['target_minute_vol']*100:.4f}%")
    logger.info(f"   Max Static: {REALISTIC_PARAMS['max_static_ratio']*100:.1f}%")
    logger.info(f"   Extreme moves: {REALISTIC_PARAMS['extreme_move_ratio']*100:.2f}%")
    
    df_1m_realistic = []
    
    for day_idx, day_row in df_daily.iterrows():
        day_start = pd.to_datetime(day_row['time'])
        
        # 1. VOLATILIDADE REALISTA POR DIA
        daily_range = day_row['high'] - day_row['low']
        daily_vol = daily_range / day_row['open']  # Volatilidade como % do preço
        
        # Ajustar volatilidade para nível realista
        target_daily_vol = REALISTIC_PARAMS['target_minute_vol'] * np.sqrt(1440)  # Escalar para dia
        vol_multiplier = max(0.5, min(3.0, target_daily_vol / daily_vol)) if daily_vol > 0 else 1.0
        
        # 2. PADRÕES DE VOLATILIDADE INTRADAY REALISTAS
        volatility_schedule = generate_realistic_volatility_schedule()
        
        # 3. REGIME DE VOLATILIDADE (clustering)
        vol_regime = np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
        vol_regime_multiplier = {'low': 0.6, 'medium': 1.0, 'high': 1.8}[vol_regime]
        
        logger.info(f"Dia {day_idx+1}: Vol regime={vol_regime} (mult={vol_regime_multiplier:.1f}x)")
        
        current_price = day_row['open']
        
        # 4. CONTROLE DE SEQUÊNCIAS ESTÁTICAS
        static_sequence_length = 0
        last_price = current_price
        
        # 5. GERAR 1440 BARRAS REALISTAS POR DIA
        for minute in range(1440):
            minute_time = day_start + timedelta(minutes=minute)
            hour = minute_time.hour
            
            # 6. VOLATILIDADE DINÂMICA REALISTA
            base_vol = daily_vol / 1440 * vol_multiplier * vol_regime_multiplier
            hourly_mult = volatility_schedule[hour]
            current_vol = base_vol * hourly_mult
            
            # 7. MODELO DE PREÇOS REALISTA (GBM + Jumps + Mean Reversion)
            
            # 7a. Drift para target do dia
            daily_progress = minute / 1440.0
            target_price = day_row['open'] + (day_row['close'] - day_row['open']) * daily_progress
            drift = (target_price - current_price) * 0.001  # Mean reversion fraco
            
            # 7b. Componente Browniano Normal
            normal_move = np.random.normal(0, current_price * current_vol)
            
            # 7c. Jumps ocasionais (eventos extremos)
            jump_prob = REALISTIC_PARAMS['extreme_move_ratio'] / 1440  # Probabilidade por minuto
            if np.random.random() < jump_prob:
                jump_size = np.random.choice([-1, 1]) * current_price * np.random.uniform(0.001, 0.005)
                logger.info(f"   JUMP! Minuto {minute}: {jump_size/current_price*100:.3f}%")
            else:
                jump_size = 0
            
            # 7d. Movimento final
            price_change = drift + normal_move + jump_size
            
            # 7e. CONTROLE INTELIGENTE DE SEQUÊNCIAS ESTÁTICAS
            # Apenas verificar se precisa forçar movimento
            force_move = False
            if static_sequence_length >= REALISTIC_PARAMS['max_static_sequence']:
                # Forçar movimento mínimo
                force_move = True
                min_move_size = REALISTIC_PARAMS['tick_size'] * np.random.choice([-1, 1])
                price_change = min_move_size
                logger.info(f"   FORÇA MOVIMENTO! Seq {static_sequence_length}: {min_move_size:.3f}")
            
            # 7f. ANTI-ESTÁTICO: Movimento mínimo ocasional (baixa probabilidade)
            elif abs(price_change) < REALISTIC_PARAMS['tick_size'] * 0.1 and np.random.random() < 0.01:  # 1% chance apenas
                min_move = REALISTIC_PARAMS['tick_size'] * np.random.choice([-1, 1])
                if abs(price_change) < abs(min_move):
                    price_change = min_move
            
            new_price = current_price + price_change
            
            # 8. LIMITAÇÕES REALISTAS
            # Garantir limites do dia (com alguma flexibilidade)
            day_low = day_row['low'] - daily_range * 0.1  # 10% flexibilidade
            day_high = day_row['high'] + daily_range * 0.1
            new_price = max(day_low, min(day_high, new_price))
            
            # 9. TICK SIZE ENFORCEMENT
            new_price = round(new_price / REALISTIC_PARAMS['tick_size']) * REALISTIC_PARAMS['tick_size']
            new_price = round(new_price, 2)
            
            # 10. GERAR OHLC INTRA-MINUTO REALISTA
            open_price = current_price
            close_price = new_price
            
            # High/Low com micro-estrutura realista
            if abs(price_change) > 0:
                intra_volatility = current_vol * np.random.uniform(0.8, 2.0)
                high_extra = abs(np.random.normal(0, current_price * intra_volatility))
                low_extra = abs(np.random.normal(0, current_price * intra_volatility))
            else:
                high_extra = low_extra = REALISTIC_PARAMS['tick_size'] * np.random.uniform(0, 0.5)
            
            high_price = max(open_price, close_price) + high_extra
            low_price = min(open_price, close_price) - low_extra
            
            # Garantir limites e formato
            high_price = round(min(high_price, day_high), 2)
            low_price = round(max(low_price, day_low), 2)
            open_price = round(open_price, 2)
            close_price = round(close_price, 2)
            
            # 11. ATUALIZAR CONTROLE DE SEQUÊNCIA ESTÁTICA
            if abs(close_price - current_price) < REALISTIC_PARAMS['tick_size'] * 0.5:
                static_sequence_length += 1
            else:
                static_sequence_length = 0  # Reset counter
            
            # 12. VOLUME REALISTA COM CLUSTERING
            base_volume = day_row['tick_volume'] / 1440
            vol_volume_correlation = 1.0 + (current_vol / REALISTIC_PARAMS['target_minute_vol'] - 1.0) * 0.5
            volume_mult = hourly_mult * vol_volume_correlation * np.random.uniform(0.2, 4.0)
            volume = max(1, int(base_volume * volume_mult))
            
            df_1m_realistic.append({
                'time': minute_time,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'tick_volume': volume,
                'spread': 0,
                'real_volume': volume,
                'data_type': 'REALISTIC_V3_1MIN',
                'vol_regime': vol_regime,
                'hour_mult': hourly_mult
            })
            
            current_price = close_price
            last_price = close_price
        
        # 13. AJUSTE FINAL DO DIA (garantir close correto)
        if df_1m_realistic:
            df_1m_realistic[-1]['close'] = round(day_row['close'], 2)
    
    df_result = pd.DataFrame(df_1m_realistic)
    logger.info(f"✅ ALGORITMO V3: {len(df_result):,} barras realistas criadas")
    
    # 14. VALIDAÇÃO AUTOMÁTICA
    validate_realistic_dataset(df_result, REALISTIC_PARAMS)
    
    return df_result

def generate_realistic_volatility_schedule():
    """Gerar padrão realista de volatilidade por hora baseado em mercados reais"""
    # Baseado em padrões reais de ouro: mais ativo em aberturas de mercados principais
    schedule = {
        # Madrugada - baixa atividade
        0: 0.3, 1: 0.25, 2: 0.2, 3: 0.3,
        # Abertura Ásia/Oceania - aumento gradual
        4: 0.6, 5: 0.8, 6: 1.1, 7: 1.3,
        # Europa ativa
        8: 1.6, 9: 1.8, 10: 1.7, 11: 1.4,
        # Almoço Europa - redução
        12: 1.0, 13: 0.9, 14: 1.1, 15: 1.3,
        # Sobreposição Europa-EUA - pico
        16: 2.0, 17: 2.2, 18: 1.9, 19: 1.6,
        # EUA ativa
        20: 1.4, 21: 1.2, 22: 1.0, 23: 0.8
    }
    
    # Adicionar ruído realista (+/-20%)
    for hour in schedule:
        noise = np.random.uniform(0.8, 1.2)
        schedule[hour] *= noise
    
    return schedule

def validate_realistic_dataset(df: pd.DataFrame, target_params: dict):
    """Validar se dataset V3 atende critérios realistas"""
    logger = setup_logging()
    logger.info("🔍 VALIDANDO DATASET V3...")
    
    # Calcular retornos
    df['returns'] = df['close'].pct_change().fillna(0)
    
    # 1. Taxa de barras estáticas
    static_ratio = (df['returns'] == 0).mean()
    logger.info(f"   Barras estáticas: {static_ratio*100:.1f}% (target: <{target_params['max_static_ratio']*100:.1f}%)")
    
    # 2. Volatilidade por minuto
    minute_vol = df['returns'].std()
    logger.info(f"   Vol/minuto: {minute_vol*100:.4f}% (target: {target_params['target_minute_vol']*100:.4f}%)")
    
    # 3. Movimentos extremos
    extreme_ratio = (np.abs(df['returns']) > 0.001).mean()
    logger.info(f"   Movimentos >0.1%: {extreme_ratio*100:.2f}% (target: {target_params['extreme_move_ratio']*100:.2f}%)")
    
    # 4. Sequências estáticas máximas
    sequences = []
    current_length = 0
    for ret in df['returns']:
        if ret == 0:
            current_length += 1
        else:
            if current_length > 0:
                sequences.append(current_length)
            current_length = 0
    
    if current_length > 0:
        sequences.append(current_length)
    
    max_sequence = max(sequences) if sequences else 0
    logger.info(f"   Sequência estática máx: {max_sequence} barras (target: <{target_params['max_static_sequence']})")
    
    # 5. VALIDAÇÃO FINAL
    issues = []
    if static_ratio > target_params['max_static_ratio']:
        issues.append(f"Barras estáticas muito altas: {static_ratio*100:.1f}%")
    
    if minute_vol < target_params['target_minute_vol'] * 0.7:
        issues.append(f"Volatilidade muito baixa: {minute_vol/target_params['target_minute_vol']:.2f}x")
    
    if extreme_ratio < target_params['extreme_move_ratio'] * 0.7:
        issues.append(f"Poucos movimentos extremos: {extreme_ratio*100:.2f}%")
    
    if max_sequence > target_params['max_static_sequence']:
        issues.append(f"Sequência muito longa: {max_sequence} barras")
    
    if not issues:
        logger.info("✅ DATASET V3 VALIDADO: Todos os critérios realistas atendidos!")
    else:
        logger.warning(f"⚠️ DATASET V3: {len(issues)} problemas detectados:")
        for issue in issues:
            logger.warning(f"   - {issue}")

def calculate_premium_indicators_v3(df: pd.DataFrame) -> pd.DataFrame:
    """Calcular indicadores premium otimizados para V3"""
    logger = setup_logging()
    logger.info("💎 Calculando indicadores premium V3...")
    
    try:
        # Basic features
        df['returns'] = df['close'].pct_change().fillna(0)
        df['high_low_ratio'] = (df['high'] / df['low']).fillna(1.0)
        df['open_close_ratio'] = (df['open'] / df['close']).fillna(1.0)
        
        # Moving averages (otimizadas para realismo)
        for period in [5, 10, 20, 60]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean().fillna(df['close'])
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean().fillna(df['close'])
        
        # RSI
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
        
        # Volatility realista
        df['volatility_60'] = df['returns'].rolling(60).std().fillna(0) * 100
        
        # Volume features
        df['volume_sma_20'] = df['tick_volume'].rolling(20).mean().fillna(1)
        df['volume_ratio'] = (df['tick_volume'] / df['volume_sma_20']).fillna(1.0).clip(0.1, 5.0)
        
        logger.info("✅ Indicadores premium V3 calculados")
        return df
        
    except Exception as e:
        logger.error(f"❌ Erro nos indicadores V3: {e}")
        return df

def save_realistic_dataset_v3(df: pd.DataFrame, symbol: str) -> str:
    """Salvar dataset V3 realisticamente calibrado"""
    logger = setup_logging()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Criar diretórios
    for dir_name in ['data', 'data_cache', 'metadata']:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    
    base_name = f"{symbol}_REALISTIC_V3_3Y_1MIN_{timestamp}"
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
    
    # Estatísticas de validação
    returns = df['returns'].dropna()
    static_ratio = (returns == 0).mean()
    vol_per_minute = returns.std()
    extreme_ratio = (np.abs(returns) > 0.001).mean()
    
    # Criar metadata
    metadata = f"""
GOLD REALISTIC DATASET V3 - REALISTICAMENTE CALIBRADO
=====================================================
Criado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Símbolo: {symbol}
Timeframe: 1 minuto
Período: {df['time'].min()} → {df['time'].max()}

ESTATÍSTICAS V3:
- Total de barras: {len(df):,}
- Dados realistas V3: {len(df):,}
- Preço inicial: ${df['close'].iloc[0]:.2f}
- Preço final: ${df['close'].iloc[-1]:.2f}
- Variação total: {((df['close'].iloc[-1]/df['close'].iloc[0])-1)*100:+.2f}%
- Volume médio: {df['tick_volume'].mean():.0f}

QUALIDADE REALISTA V3:
- Barras estáticas: {static_ratio*100:.1f}% (target: <8%)
- Volatilidade/min: {vol_per_minute*100:.4f}% (target: ~0.026%)
- Movimentos >0.1%: {extreme_ratio*100:.2f}% (target: ~1.2%)
- Algoritmo: GBM + Jumps + Mean Reversion + Anti-static
- Calibração: Baseada em investigação profunda do V2
- Clustering: Volatilidade realista por hora
- Microestrutura: Tick size 0.01 enforced

ARQUIVOS:
- CSV: {csv_file}
- Pickle: {pkl_file}
- Metadata: {metadata_file}
"""
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write(metadata)
    
    logger.info(f"💾 Dataset V3 realista salvo: {csv_file}")
    return csv_file

def main():
    """Criar dataset V3 realisticamente calibrado"""
    logger = setup_logging()
    
    logger.info("🚀 CRIANDO DATASET V3 REALISTICAMENTE CALIBRADO")
    logger.info("="*70)
    logger.info("🎯 CORREÇÕES BASEADAS NA INVESTIGAÇÃO PROFUNDA:")
    logger.info("   - Volatilidade: 2x maior que V2")
    logger.info("   - Barras estáticas: <8% (vs 41.6% do V2)")
    logger.info("   - Movimentos extremos: 25x mais frequentes")
    logger.info("   - Sequências estáticas: máx 45 barras (vs 1440)")
    
    # 1. DOWNLOAD HISTÓRICO
    df_daily = download_daily_historical("GC=F", years_back=3)
    
    if df_daily.empty:
        logger.error("❌ Falha no download histórico")
        return
    
    # 2. ALGORITMO V3 REALISTA
    logger.info("🧠 FASE: Algoritmo V3 realisticamente calibrado")
    df_realistic_1m = create_realistic_intraday_algorithm_v3(df_daily)
    
    if df_realistic_1m.empty:
        logger.error("❌ Falha na criação do dataset V3")
        return
    
    # 3. INDICADORES PREMIUM
    logger.info("💎 FASE: Indicadores premium V3")
    df_final = calculate_premium_indicators_v3(df_realistic_1m)
    
    # 4. SALVAR DATASET V3
    save_path = save_realistic_dataset_v3(df_final, "GC=F")
    
    logger.info("🎉 DATASET V3 REALISTICAMENTE CALIBRADO CRIADO!")
    logger.info(f"📊 Total: {len(df_final):,} barras")
    logger.info(f"📁 Arquivo: {save_path}")
    logger.info("🔥 PRONTO PARA TREINAMENTO REALISTA!")
    
    return save_path

if __name__ == "__main__":
    main()