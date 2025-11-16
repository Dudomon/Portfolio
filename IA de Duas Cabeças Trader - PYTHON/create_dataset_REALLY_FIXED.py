#!/usr/bin/env python3
"""
DATASET DESAFIADOR - VERSÃO REALMENTE CORRIGIDA
Problema anterior: preços cresciam exponencialmente
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def create_dataset_smart(n_bars=2000000, base_price=2000.0):
    """
    Criar dataset com lógica CORRETA - preços controlados
    """
    print(f"Criando dataset com lógica CORRIGIDA: {n_bars:,} barras...")
    
    data = []
    current_time = datetime(2023, 1, 1)
    
    # USAR RANDOM WALK COM MEAN REVERSION - não explosão exponencial
    prices = [base_price]
    
    # Parâmetros do modelo
    mean_price = base_price  # Preço médio de longo prazo
    mean_reversion_speed = 0.001  # Velocidade de reversão à média
    
    # Estados dos regimes
    regimes = ['bull', 'bear', 'sideways']
    current_regime = 'bull'
    regime_duration = random.randint(100000, 300000)
    regime_counter = 0
    
    # Parâmetros por regime
    regime_params = {
        'bull': {'drift': 0.0001, 'vol': 0.012},
        'bear': {'drift': -0.0001, 'vol': 0.018}, 
        'sideways': {'drift': 0.0, 'vol': 0.008}
    }
    
    print(f"Regime inicial: {current_regime} por {regime_duration:,} barras")
    
    for i in range(1, n_bars):
        if i % 200000 == 0:
            print(f"Progresso: {i:,}/{n_bars:,} ({i/n_bars*100:.1f}%) - Preço: {prices[-1]:.2f}")
        
        # Mudança de regime
        regime_counter += 1
        if regime_counter >= regime_duration:
            # Escolher novo regime
            available = [r for r in regimes if r != current_regime]
            current_regime = random.choice(available)
            regime_duration = random.randint(50000, 250000)
            regime_counter = 0
            print(f"  Regime: {current_regime} por {regime_duration:,} barras (barra {i:,})")
        
        # Parâmetros do regime atual
        params = regime_params[current_regime]
        
        # MEAN REVERSION: força que puxa preço para média
        current_price = prices[-1]
        mean_reversion_force = mean_reversion_speed * (mean_price - current_price) / mean_price
        
        # Componente drift do regime
        regime_drift = params['drift']
        
        # Componente aleatório
        random_shock = np.random.normal(0, params['vol'])
        
        # Return total com mean reversion
        total_return = regime_drift + mean_reversion_force + random_shock
        
        # LIMITAR RETURN para evitar explosão
        total_return = np.clip(total_return, -0.03, 0.03)  # Máximo ±3%
        
        # Novo preço
        new_price = current_price * (1 + total_return)
        
        # FORÇA mean reversion se preço fugir muito
        if new_price > mean_price * 2.0:  # Se 2x maior que média
            new_price = mean_price * 1.8  # Forçar volta
        elif new_price < mean_price * 0.5:  # Se metade da média
            new_price = mean_price * 0.6  # Forçar volta
        
        prices.append(new_price)
    
    print("Preços gerados, criando OHLCV...")
    
    # Gerar OHLCV a partir dos preços
    for i in range(n_bars):
        if i == 0:
            open_price = close_price = prices[i]
        else:
            open_price = prices[i-1]  # Open = close anterior
            close_price = prices[i]
        
        # High/Low baseado em volatilidade intrabar
        price_change = abs(close_price - open_price)
        intrabar_vol = max(price_change * 0.5, close_price * 0.005)  # Min 0.5% do preço
        
        high_price = max(open_price, close_price) + random.uniform(0, intrabar_vol)
        low_price = min(open_price, close_price) - random.uniform(0, intrabar_vol)
        
        # Volume baseado em movimento
        vol_factor = 1 + (price_change / open_price * 10)  # Mais volume = mais movimento
        volume = int(15000 * vol_factor * random.uniform(0.8, 1.2))
        volume = max(5000, min(50000, volume))
        
        # Determinar regime baseado na posição
        regime_idx = (i // 100000) % 3
        regime_name = regimes[regime_idx]
        
        bar_data = {
            'timestamp': current_time + timedelta(minutes=i),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2), 
            'close': round(close_price, 2),
            'volume': volume,
            'regime': regime_name
        }
        
        data.append(bar_data)
    
    print("Dataset criado!")
    return pd.DataFrame(data)

def validate_dataset_simple(df):
    """Validação simples sem emojis problemáticos"""
    print("\\nVALIDACAO DO DATASET")
    print("="*40)
    
    # NaN check
    nan_count = df.isnull().sum().sum()
    print(f"NaN values: {nan_count}")
    
    # Price range
    min_price = df[['open', 'high', 'low', 'close']].min().min()
    max_price = df[['open', 'high', 'low', 'close']].max().max()
    print(f"Price range: {min_price:.2f} - {max_price:.2f}")
    
    # Returns
    returns = df['close'].pct_change()
    vol = returns.std()
    max_ret = returns.max()
    min_ret = returns.min()
    
    print(f"Volatility: {vol:.4f} ({vol*100:.2f}%)")
    print(f"Max return: {max_ret:.4f} ({max_ret*100:.2f}%)")
    print(f"Min return: {min_ret:.4f} ({min_ret*100:.2f}%)")
    
    # Check sanity
    sane = True
    if nan_count > 0:
        print("FAIL: Has NaN values")
        sane = False
    if min_price <= 0:
        print("FAIL: Negative prices")
        sane = False
    if abs(max_ret) > 0.1 or abs(min_ret) > 0.1:
        print("FAIL: Extreme returns")
        sane = False
    if vol > 0.05:
        print("FAIL: Too volatile")
        sane = False
    
    if sane:
        print("PASS: Dataset is valid")
    
    return sane

def main():
    print("CRIANDO DATASET COM LOGICA CORRIGIDA")
    print("="*50)
    
    # Criar dataset
    df = create_dataset_smart(n_bars=2000000)
    
    # Validar
    if not validate_dataset_simple(df):
        print("\\nDataset failed validation!")
        return
    
    # Salvar
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/GOLD_FIXED_2M_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\\nDataset saved: {filename}")
    
    print("\\nCaracteristicas:")
    print("- Mean reversion (precos nao explodem)")
    print("- Volatilidade controlada")
    print("- Returns limitados a ±3%") 
    print("- Regimes dinamicos")
    print("- Validacao passou")
    print("\\nEste dataset e desafiador MAS seguro!")

if __name__ == '__main__':
    main()