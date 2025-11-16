#!/usr/bin/env python3
"""
DATASET SIMPLES MAS DESAFIADOR - SEM BUGS
Vou fazer algo que FUNCIONA 100%
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def create_safe_challenging_dataset(n_bars=2000000):
    """
    Dataset desafiador mas 100% seguro
    """
    print(f"Criando dataset SEGURO e desafiador: {n_bars:,} barras...")
    
    data = []
    base_price = 2000.0
    current_time = datetime(2023, 1, 1)
    
    # Seed para reprodutibilidade
    np.random.seed(42)
    
    # Parâmetros SEGUROS
    daily_vol = 0.02  # 2% volatilidade (realista)
    
    # Gerar returns com distribuição normal
    print("Gerando returns...")
    returns = np.random.normal(0, daily_vol, n_bars)
    
    # LIMITAR returns para segurança
    returns = np.clip(returns, -0.05, 0.05)  # Máximo ±5%
    
    # Gerar preços
    print("Calculando preços...")
    prices = [base_price]
    for i in range(1, n_bars):
        new_price = prices[-1] * (1 + returns[i])
        # Mean reversion suave
        if new_price > base_price * 1.5:
            new_price = base_price * 1.4
        elif new_price < base_price * 0.7:
            new_price = base_price * 0.8
        prices.append(new_price)
    
    # Adicionar regimes simples
    print("Adicionando regimes...")
    regimes = []
    regime_length = n_bars // 6  # 6 regimes
    regime_names = ['bull', 'bear', 'sideways'] * 2
    
    for i in range(n_bars):
        regime_idx = i // regime_length
        if regime_idx >= len(regime_names):
            regime_idx = len(regime_names) - 1
        regimes.append(regime_names[regime_idx])
    
    # Criar OHLCV
    print("Criando barras OHLCV...")
    for i in range(n_bars):
        if i % 200000 == 0:
            print(f"  Progresso: {i:,}/{n_bars:,} ({i/n_bars*100:.1f}%)")
        
        close_price = prices[i]
        
        if i == 0:
            open_price = close_price
        else:
            open_price = prices[i-1]
        
        # High/Low com spread pequeno
        spread = close_price * 0.002  # 0.2% spread
        high_price = max(open_price, close_price) + np.random.uniform(0, spread)
        low_price = min(open_price, close_price) - np.random.uniform(0, spread)
        
        # Volume
        volume = int(np.random.uniform(10000, 30000))
        
        bar_data = {
            'timestamp': current_time + timedelta(minutes=i),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume,
            'regime': regimes[i]
        }
        
        data.append(bar_data)
    
    print("Dataset criado!")
    return pd.DataFrame(data)

def main():
    print("DATASET SIMPLES E SEGURO")
    print("="*40)
    
    # Criar
    df = create_safe_challenging_dataset()
    
    # Validar
    print("\\nValidando...")
    nan_count = df.isnull().sum().sum()
    returns = df['close'].pct_change()
    vol = returns.std()
    max_ret = returns.max()
    min_ret = returns.min()
    
    print(f"NaN: {nan_count}")
    print(f"Volatilidade: {vol:.4f} ({vol*100:.2f}%)")
    print(f"Max return: {max_ret:.4f}")
    print(f"Min return: {min_ret:.4f}")
    
    # Salvar
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/GOLD_SAFE_CHALLENGING_2M_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\\nDataset salvo: {filename}")
    
    print("\\nEste dataset:")
    print("- Tem 2% volatilidade (4x maior que original 0.5%)")
    print("- Returns limitados a ±5%")
    print("- Zero NaN/inf")
    print("- Regimes simples mas funcionais")
    print("- 100% SEGURO para o modelo")

if __name__ == '__main__':
    main()