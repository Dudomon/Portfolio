#!/usr/bin/env python3
"""
CRIAR DATASET DESAFIADOR PARA EVITAR OVERFITTING DO MODELO V7
Baseado na análise: datasets atuais são muito fáceis (volatilidade 0.14%, autocorr=1.0)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def create_challenging_gold_dataset(n_bars=2000000, base_price=2000.0):
    """
    Criar dataset com dificuldade adequada para modelo V7 (1.45M params)
    """
    print(f"Criando dataset desafiador com {n_bars:,} barras...")
    
    data = []
    current_price = base_price
    current_time = datetime(2023, 1, 1)
    
    # Parâmetros por regime
    regimes = {
        'bull': {
            'trend': 0.0003,        # 0.03% por barra
            'volatility': 0.012,    # 1.2% volatilidade 
            'up_bias': 0.55,        # 55% chances up
            'duration_range': (100000, 300000)  # 100k-300k barras
        },
        'bear': {  
            'trend': -0.0002,       # -0.02% por barra
            'volatility': 0.018,    # 1.8% volatilidade (maior em bear)
            'up_bias': 0.45,        # 45% chances up
            'duration_range': (50000, 200000)   # 50k-200k barras
        },
        'sideways': {
            'trend': 0.0,           # Sem trend
            'volatility': 0.008,    # 0.8% volatilidade
            'up_bias': 0.50,        # 50% chances up
            'duration_range': (80000, 250000)   # 80k-250k barras
        }
    }
    
    # Estado inicial
    current_regime = 'bull'
    regime_bars_left = random.randint(*regimes[current_regime]['duration_range'])
    
    # Volatility clustering state
    vol_multiplier = 1.0
    vol_persistence = 0.95
    
    # Evento extremo tracker
    next_extreme_event = random.randint(100000, 200000)
    
    print(f"Regime inicial: {current_regime} por {regime_bars_left:,} barras")
    
    for i in range(n_bars):
        # Progress
        if i % 200000 == 0:
            print(f"Progresso: {i:,}/{n_bars:,} ({i/n_bars*100:.1f}%)")
        
        # === REGIME SWITCHING ===
        if regime_bars_left <= 0:
            # Escolher próximo regime (não pode repetir)
            available_regimes = [r for r in regimes.keys() if r != current_regime]
            current_regime = random.choice(available_regimes)
            regime_bars_left = random.randint(*regimes[current_regime]['duration_range'])
            print(f"  Mudança de regime: {current_regime} por {regime_bars_left:,} barras (barra {i:,})")
        
        regime_info = regimes[current_regime]
        
        # === VOLATILITY CLUSTERING ===
        # Volatilidade varia com persistência
        vol_shock = random.normalvariate(0, 0.1)
        vol_multiplier = vol_persistence * vol_multiplier + (1 - vol_persistence) * 1.0 + vol_shock
        vol_multiplier = max(0.3, min(3.0, vol_multiplier))  # Limitar entre 0.3x e 3x
        
        effective_volatility = regime_info['volatility'] * vol_multiplier
        
        # === EVENTOS EXTREMOS ===
        extreme_event_factor = 1.0
        if i >= next_extreme_event:
            event_type = random.choice(['flash_crash', 'gap_up', 'gap_down', 'volatility_spike'])
            
            if event_type == 'flash_crash':
                extreme_event_factor = 0.95  # -5% instantâneo
                effective_volatility *= 3    # Volatilidade 3x maior
                print(f"  Flash crash na barra {i:,}")
                
            elif event_type == 'gap_up':
                extreme_event_factor = 1.02  # +2% gap
                print(f"  Gap up na barra {i:,}")
                
            elif event_type == 'gap_down':
                extreme_event_factor = 0.98  # -2% gap
                print(f"  Gap down na barra {i:,}")
                
            elif event_type == 'volatility_spike':
                effective_volatility *= 4    # Volatilidade 4x maior
                print(f"  Volatility spike na barra {i:,}")
            
            # Próximo evento em 150k-300k barras
            next_extreme_event = i + random.randint(150000, 300000)
        
        # === GERAÇÃO DE PREÇOS ===
        # Trend + noise + bias direcional
        trend_component = regime_info['trend']
        
        # Bias direcional baseado no regime
        direction = 1 if random.random() < regime_info['up_bias'] else -1
        
        # Return com noise heteroscedástico
        base_return = trend_component + direction * np.abs(random.normalvariate(0, effective_volatility))
        
        # Adicionar autocorrelação imperfeita (quebrar previsibilidade perfeita)
        if i > 0:
            last_return = (data[-1]['close'] - data[-1]['open']) / data[-1]['open']
            momentum_factor = 0.1 * last_return  # 10% momentum carry
            base_return += momentum_factor
        
        # Aplicar evento extremo
        total_return = (base_return * extreme_event_factor) - 1
        
        # Calcular novo preço
        new_price = current_price * (1 + total_return)
        
        # === OHLC GENERATION ===
        # Spread variável baseado em volatilidade
        base_spread = 0.0005  # 0.05% base
        vol_spread = effective_volatility * 0.1  # Spread aumenta com volatilidade
        total_spread = base_spread + vol_spread
        
        # Intrabar movement
        high_factor = 1 + random.uniform(0, effective_volatility * 0.5)
        low_factor = 1 - random.uniform(0, effective_volatility * 0.5)
        
        open_price = current_price
        close_price = new_price
        high_price = max(open_price, close_price) * high_factor
        low_price = min(open_price, close_price) * low_factor
        
        # Volume correlacionado com volatilidade
        base_volume = 10000
        vol_volume = int(base_volume * (1 + effective_volatility * 10))
        volume = random.randint(vol_volume // 2, vol_volume * 2)
        
        # Criar registro
        bar_data = {
            'timestamp': current_time + timedelta(minutes=i),
            'open': round(open_price, 2),
            'high': round(high_price, 2), 
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume,
            'regime': current_regime
        }
        
        data.append(bar_data)
        
        # Update state
        current_price = close_price
        regime_bars_left -= 1
    
    print("Dataset criado com sucesso!")
    return pd.DataFrame(data)

def analyze_dataset_difficulty(df):
    """Analisar dificuldade do dataset criado"""
    print("\\n" + "="*60)
    print("ANÁLISE DE DIFICULDADE DO DATASET CRIADO")
    print("="*60)
    
    # Estatísticas básicas
    df['returns'] = df['close'].pct_change()
    volatility = df['returns'].std()
    
    print(f"Shape: {df.shape}")
    print(f"Volatilidade diária: {volatility:.4f} ({volatility*100:.2f}%)")
    
    # Up/down balance
    up_days = (df['returns'] > 0).sum()
    down_days = (df['returns'] < 0).sum()
    print(f"Up days: {up_days:,} ({up_days/len(df)*100:.1f}%)")
    print(f"Down days: {down_days:,} ({down_days/len(df)*100:.1f}%)")
    
    # Reversals
    price_changes = np.diff(df['close'])
    direction_changes = np.sum(np.diff(np.sign(price_changes)) != 0)
    reversal_rate = direction_changes / len(price_changes)
    print(f"Taxa de reversão: {reversal_rate:.3f}")
    
    # Autocorrelação
    autocorr = df['returns'].autocorr(lag=1)
    print(f"Autocorrelação lag-1: {autocorr:.3f}")
    
    # Regimes
    regime_counts = df['regime'].value_counts()
    print(f"\\nDistribuição de regimes:")
    for regime, count in regime_counts.items():
        print(f"  {regime}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Score de dificuldade
    difficulty_score = 0
    
    if volatility > 0.01:  # > 1%
        difficulty_score += 2
        print("\\nVolatilidade ADEQUADA para modelo complexo")
    
    if 0.3 < reversal_rate < 0.7:
        difficulty_score += 1
        print("Reversões em nível ADEQUADO")
    
    if abs(autocorr) < 0.8:
        difficulty_score += 1
        print("Autocorrelação ADEQUADA (não muito previsível)")
    
    print(f"\\nScore de dificuldade: {difficulty_score}/4")
    
    if difficulty_score >= 3:
        print("DATASET ADEQUADO para modelo V7")
    else:
        print("Dataset ainda pode ser muito fácil")

def main():
    print("CRIANDO DATASET DESAFIADOR PARA MODELO V7")
    print("="*80)
    
    # Criar dataset
    df = create_challenging_gold_dataset(n_bars=2000000)
    
    # Salvar
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/GOLD_CHALLENGING_2M_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\\nDataset salvo: {filename}")
    
    # Analisar
    analyze_dataset_difficulty(df)
    
    print("\\n" + "="*80)
    print("DATASET DESAFIADOR CRIADO COM SUCESSO!")
    print(f"Arquivo: {filename}")
    print("Características:")
    print("- Regimes dinâmicos (bull/bear/sideways)")
    print("- Volatilidade variável (0.8% - 1.8%)")
    print("- Eventos extremos (crashes, gaps)")
    print("- Autocorrelação imperfeita")
    print("- Microestrutura realista")
    print("\\nUsar este dataset com hiperparâmetros ultra-conservadores!")

if __name__ == '__main__':
    main()