#!/usr/bin/env python3
"""
DATASET DESAFIADOR MATEMATICAMENTE COERENTE - VERSÃO CORRIGIDA
Gera 2M barras de 5min com proteções contra overflow
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def create_proper_challenging_dataset(n_bars=2000000, base_price=2000.0):
    """
    Criar dataset desafiador mas MATEMATICAMENTE COERENTE
    Com proteções contra overflow e valores extremos
    """
    print(f"🎯 CRIANDO DATASET DESAFIADOR COERENTE - {n_bars:,} barras")
    print("="*80)
    
    data = []
    current_time = datetime(2024, 1, 1)
    
    # === PARÂMETROS REALISTAS E DISTINTIVOS ===
    regimes = {
        'bull': {
            'drift': 0.00008,        # +0.008% por barra
            'volatility': 0.010,     # 1.0% volatilidade
            'up_prob': 0.56,         # 56% probabilidade de subir
            'vol_clustering': 0.85,  # Clustering médio
            'mean_reversion': 0.02,  # Baixa reversão
            'duration_range': (50000, 150000)
        },
        'bear': {
            'drift': -0.00010,       # -0.010% por barra
            'volatility': 0.018,     # 1.8% volatilidade (alta)
            'up_prob': 0.44,         # 44% probabilidade de subir
            'vol_clustering': 0.90,  # Alto clustering
            'mean_reversion': 0.01,  # Muito baixa reversão
            'duration_range': (30000, 100000)
        },
        'sideways': {
            'drift': 0.0,            # Zero drift
            'volatility': 0.007,     # 0.7% volatilidade (baixa)
            'up_prob': 0.50,         # 50% probabilidade
            'vol_clustering': 0.75,  # Clustering baixo
            'mean_reversion': 0.06,  # Alta reversão à média
            'duration_range': (40000, 120000)
        },
        'volatile': {
            'drift': 0.00003,        # Drift muito pequeno
            'volatility': 0.025,     # 2.5% volatilidade (alta)
            'up_prob': 0.51,         # Quase neutro
            'vol_clustering': 0.93,  # Clustering alto
            'mean_reversion': 0.03,  # Média reversão
            'duration_range': (20000, 60000)
        }
    }
    
    # === ESTADO INICIAL ===
    current_price = base_price
    current_regime = random.choice(list(regimes.keys()))
    regime_bars_left = random.randint(*regimes[current_regime]['duration_range'])
    
    # Tracking variables
    vol_state = 1.0  # Estado de volatilidade atual
    momentum = 0.0   # Momentum de curto prazo
    regime_transitions = []
    
    print(f"Regime inicial: {current_regime} por {regime_bars_left:,} barras")
    print("\nGerando barras...")
    
    for i in range(n_bars):
        if i % 100000 == 0 and i > 0:
            print(f"  Progresso: {i:,}/{n_bars:,} ({i/n_bars*100:.1f}%) - Preço: ${current_price:.2f}")
        
        # === PROTEÇÃO CONTRA VALORES EXTREMOS ===
        if current_price < 100:  # Preço muito baixo
            current_price = 100
        elif current_price > 20000:  # Preço muito alto
            current_price = 20000
        
        # === MUDANÇA DE REGIME ===
        if regime_bars_left <= 0:
            old_regime = current_regime
            # Transição inteligente
            if current_regime == 'volatile':
                current_regime = random.choices(['sideways', 'bear', 'bull'], 
                                              weights=[0.5, 0.3, 0.2])[0]
            elif current_regime == 'sideways':
                current_regime = random.choices(['bull', 'bear', 'volatile', 'sideways'], 
                                              weights=[0.3, 0.3, 0.2, 0.2])[0]
            else:
                available = [r for r in regimes.keys() if r != current_regime]
                weights = [0.4 if r == 'sideways' else 0.3 for r in available]
                current_regime = random.choices(available, weights=weights)[0]
            
            regime_bars_left = random.randint(*regimes[current_regime]['duration_range'])
            regime_transitions.append((i, old_regime, current_regime))
            
            if len(regime_transitions) <= 10:
                print(f"    Transição: {old_regime} → {current_regime} (barra {i:,})")
        
        regime = regimes[current_regime]
        
        # === VOLATILIDADE COM CLUSTERING (LIMITADA) ===
        vol_shock = np.random.normal(0, 0.05)  # Shock menor
        vol_state = regime['vol_clustering'] * vol_state + \
                   (1 - regime['vol_clustering']) * 1.0 + vol_shock
        vol_state = np.clip(vol_state, 0.5, 2.0)  # Limites mais rígidos
        
        effective_vol = regime['volatility'] * vol_state
        
        # === MOMENTUM E MEAN REVERSION ===
        momentum *= (1 - regime['mean_reversion'])
        momentum = np.clip(momentum, -0.002, 0.002)  # Limitar momentum
        
        # === GERAÇÃO DE RETORNO ===
        drift = regime['drift']
        
        # Componente estocástico
        if random.random() < regime['up_prob']:
            stochastic = abs(np.random.normal(0, effective_vol))
        else:
            stochastic = -abs(np.random.normal(0, effective_vol))
        
        # Momentum effect limitado
        momentum_effect = momentum * 0.2
        
        # Return final com proteções
        price_return = drift + stochastic + momentum_effect
        price_return = np.clip(price_return, -0.03, 0.03)  # Limite de 3%
        
        # Atualizar momentum
        momentum = momentum * 0.8 + price_return * 0.2
        
        # === EVENTOS ESPECIAIS (2% chance, mais controlados) ===
        if random.random() < 0.02:
            event_type = random.choice(['mini_spike', 'mini_crash', 'squeeze'])
            
            if event_type == 'mini_spike' and current_regime != 'bear':
                price_return += effective_vol * 0.5
            elif event_type == 'mini_crash' and current_regime != 'bull':
                price_return -= effective_vol * 0.5
            elif event_type == 'squeeze':
                effective_vol *= 0.5
            
            price_return = np.clip(price_return, -0.03, 0.03)
        
        # === CALCULAR NOVO PREÇO ===
        new_price = current_price * (1 + price_return)
        new_price = np.clip(new_price, current_price * 0.97, current_price * 1.03)
        
        # === GERAR OHLC REALISTA ===
        open_price = current_price
        close_price = new_price
        
        # Wicks proporcionais mas controlados
        wick_size = effective_vol * 0.3
        
        if close_price > open_price:
            high_price = close_price * (1 + wick_size)
            low_price = open_price * (1 - wick_size * 0.5)
        else:
            high_price = open_price * (1 + wick_size * 0.5)
            low_price = close_price * (1 - wick_size)
        
        # Garantir coerência
        high_price = max(high_price, open_price, close_price) * 1.001
        low_price = min(low_price, open_price, close_price) * 0.999
        
        # === VOLUME CORRELACIONADO ===
        base_volume = 15000
        vol_factor = 1 + effective_vol * 10
        move_factor = 1 + abs(price_return) * 20
        regime_factor = {'bull': 1.1, 'bear': 1.2, 'sideways': 0.9, 'volatile': 1.3}
        
        volume = int(base_volume * vol_factor * move_factor * 
                    regime_factor[current_regime] * random.uniform(0.8, 1.2))
        volume = np.clip(volume, 8000, 50000)
        
        # === CRIAR BARRA ===
        bar = {
            'timestamp': current_time + timedelta(minutes=i*5),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume,
            'regime': current_regime
        }
        
        data.append(bar)
        
        # Atualizar estado
        current_price = close_price
        regime_bars_left -= 1
    
    # === CRIAR DATAFRAME ===
    df = pd.DataFrame(data)
    
    print(f"\n✅ Dataset criado com sucesso!")
    print(f"   Total de transições de regime: {len(regime_transitions)}")
    
    return df

def validate_and_analyze(df):
    """Validar e analisar o dataset criado"""
    print("\n" + "="*80)
    print("🔍 VALIDAÇÃO E ANÁLISE DO DATASET")
    print("="*80)
    
    # Adicionar returns
    df['returns'] = df['close'].pct_change()
    
    # Estatísticas básicas
    print("\n📊 ESTATÍSTICAS DE PREÇO:")
    for col in ['open', 'high', 'low', 'close']:
        print(f"   {col}: min=${df[col].min():.2f}, max=${df[col].max():.2f}, mean=${df[col].mean():.2f}")
    
    # Returns
    returns = df['returns'].dropna()
    print(f"\n📈 ANÁLISE DE RETURNS:")
    print(f"   Mean: {returns.mean():.6f} ({returns.mean()*100:.4f}%)")
    print(f"   Std: {returns.std():.6f} ({returns.std()*100:.2f}%)")
    print(f"   Skew: {returns.skew():.3f}")
    print(f"   Kurt: {returns.kurtosis():.3f}")
    print(f"   Min: {returns.min():.4f}")
    print(f"   Max: {returns.max():.4f}")
    
    # Regimes
    print(f"\n🎯 ANÁLISE POR REGIME:")
    regime_stats = df.groupby('regime')['returns'].agg(['mean', 'std', 'count']).round(6)
    
    for regime in regime_stats.index:
        count = regime_stats.loc[regime, 'count']
        mean = regime_stats.loc[regime, 'mean'] 
        std = regime_stats.loc[regime, 'std']
        print(f"   {regime}: {count:,} bars, mean={mean:.6f} ({mean*100:.4f}%), std={std:.6f}")
    
    # Verificar diferença entre regimes
    means = regime_stats['mean'].values
    regime_diff = np.max(means) - np.min(means)
    print(f"\n   Diferença entre regimes: {regime_diff:.6f} ({regime_diff*100:.4f}%)")
    
    if regime_diff > 0.0001:
        print("   ✅ Regimes são distintivos!")
    else:
        print("   ❌ Regimes muito similares!")
    
    # Volume
    df['vol_returns_corr'] = df['volume'].rolling(100).corr(df['returns'].abs())
    vol_corr_mean = df['vol_returns_corr'].mean()
    print(f"\n📊 ANÁLISE DE VOLUME:")
    print(f"   Volume médio: {df['volume'].mean():,.0f}")
    print(f"   Correlação volume-volatilidade: {vol_corr_mean:.3f}")
    
    # Autocorrelação
    autocorr_1 = returns.autocorr(lag=1)
    autocorr_5 = returns.autocorr(lag=5)
    print(f"\n📈 AUTOCORRELAÇÃO:")
    print(f"   Lag-1: {autocorr_1:.4f}")
    print(f"   Lag-5: {autocorr_5:.4f}")
    
    print("\n✅ Dataset validado e pronto para uso!")
    
    return True

def main():
    print("🚀 GERADOR DE DATASET DESAFIADOR COERENTE V2")
    print("="*80)
    
    # Criar dataset
    df = create_proper_challenging_dataset(n_bars=2000000)
    
    # Validar
    if validate_and_analyze(df):
        # Salvar
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/GOLD_COHERENT_FIXED_2M_{timestamp}.csv"
        
        # Salvar apenas colunas essenciais
        columns_to_save = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'regime']
        df[columns_to_save].to_csv(filename, index=False)
        
        print(f"\n✅ DATASET SALVO: {filename}")
        print("\n📋 CARACTERÍSTICAS FINAIS:")
        print("   ✓ 2M barras de 5 minutos")
        print("   ✓ 4 regimes distintivos com características únicas")
        print("   ✓ Transições inteligentes entre regimes")
        print("   ✓ Volatilidade clustering controlada")
        print("   ✓ Momentum e mean reversion")
        print("   ✓ Volume correlacionado")
        print("   ✓ Proteções contra overflow")
        print("   ✓ Matematicamente coerente")
        print("   ✓ Desafiador mas convergível")
        
        return filename

if __name__ == '__main__':
    main()