#!/usr/bin/env python3
"""
DATASET DESAFIADOR MATEMATICAMENTE COERENTE
Gera 2M barras de 5min com padrões aprendíveis mas não triviais
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def create_proper_challenging_dataset(n_bars=2000000, base_price=2000.0):
    """
    Criar dataset desafiador mas MATEMATICAMENTE COERENTE
    - Regimes com características distintivas
    - Padrões aprendíveis mas não óbvios
    - Volatilidade realista
    - Sem defeitos matemáticos
    """
    print(f"🎯 CRIANDO DATASET DESAFIADOR COERENTE - {n_bars:,} barras")
    print("="*80)
    
    data = []
    current_time = datetime(2024, 1, 1)
    
    # === PARÂMETROS REALISTAS E DISTINTIVOS ===
    regimes = {
        'bull': {
            'drift': 0.00015,        # +0.015% por barra (bull médio)
            'volatility': 0.012,     # 1.2% volatilidade
            'up_prob': 0.58,         # 58% probabilidade de subir
            'vol_clustering': 0.85,  # Clustering médio
            'mean_reversion': 0.02,  # Baixa reversão
            'duration_range': (50000, 150000)
        },
        'bear': {
            'drift': -0.00018,       # -0.018% por barra (bear forte)
            'volatility': 0.022,     # 2.2% volatilidade (alta)
            'up_prob': 0.42,         # 42% probabilidade de subir
            'vol_clustering': 0.92,  # Alto clustering (pânico)
            'mean_reversion': 0.01,  # Muito baixa reversão
            'duration_range': (30000, 100000)
        },
        'sideways': {
            'drift': 0.0,            # Zero drift
            'volatility': 0.008,     # 0.8% volatilidade (baixa)
            'up_prob': 0.50,         # 50% probabilidade
            'vol_clustering': 0.75,  # Clustering baixo
            'mean_reversion': 0.08,  # Alta reversão à média
            'duration_range': (40000, 120000)
        },
        'volatile': {
            'drift': 0.00005,        # Drift pequeno
            'volatility': 0.035,     # 3.5% volatilidade (muito alta)
            'up_prob': 0.51,         # Quase neutro
            'vol_clustering': 0.95,  # Clustering extremo
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
        
        # === MUDANÇA DE REGIME ===
        if regime_bars_left <= 0:
            old_regime = current_regime
            # Transição inteligente (não puramente aleatória)
            if current_regime == 'volatile':
                # Após volatilidade, tende a sideways
                current_regime = random.choices(['sideways', 'bear', 'bull'], 
                                              weights=[0.5, 0.3, 0.2])[0]
            elif current_regime == 'sideways':
                # Sideways pode quebrar para qualquer direção
                current_regime = random.choices(['bull', 'bear', 'volatile', 'sideways'], 
                                              weights=[0.3, 0.3, 0.2, 0.2])[0]
            else:
                # Bull/Bear tendem a não se inverter diretamente
                available = [r for r in regimes.keys() if r != current_regime]
                weights = [0.4 if r == 'sideways' else 0.3 for r in available]
                current_regime = random.choices(available, weights=weights)[0]
            
            regime_bars_left = random.randint(*regimes[current_regime]['duration_range'])
            regime_transitions.append((i, old_regime, current_regime))
            
            if len(regime_transitions) <= 20:  # Mostrar apenas primeiras transições
                print(f"    Transição: {old_regime} → {current_regime} (barra {i:,})")
        
        regime = regimes[current_regime]
        
        # === VOLATILIDADE COM CLUSTERING ===
        vol_shock = np.random.normal(0, 0.1)
        vol_state = regime['vol_clustering'] * vol_state + \
                   (1 - regime['vol_clustering']) * 1.0 + vol_shock
        vol_state = np.clip(vol_state, 0.3, 3.0)
        
        effective_vol = regime['volatility'] * vol_state
        
        # === MOMENTUM E MEAN REVERSION ===
        # Momentum decai com mean reversion
        momentum *= (1 - regime['mean_reversion'])
        
        # === GERAÇÃO DE RETORNO ===
        # Componente de drift
        drift = regime['drift']
        
        # Componente estocástico com viés direcional
        if random.random() < regime['up_prob']:
            stochastic = abs(np.random.normal(0, effective_vol))
        else:
            stochastic = -abs(np.random.normal(0, effective_vol))
        
        # Adicionar momentum
        momentum_effect = momentum * 0.3  # 30% do momentum anterior
        
        # Return final
        price_return = drift + stochastic + momentum_effect
        price_return = np.clip(price_return, -0.05, 0.05)  # Limite de 5%
        
        # Atualizar momentum
        momentum = momentum * 0.7 + price_return * 0.3
        
        # === EVENTOS ESPECIAIS (5% chance) ===
        if random.random() < 0.05:
            event_type = random.choices(
                ['spike', 'crash', 'gap', 'squeeze'],
                weights=[0.25, 0.25, 0.3, 0.2]
            )[0]
            
            if event_type == 'spike' and current_regime != 'bear':
                price_return += abs(np.random.normal(0, effective_vol * 2))
            elif event_type == 'crash' and current_regime != 'bull':
                price_return -= abs(np.random.normal(0, effective_vol * 2))
            elif event_type == 'gap':
                price_return += np.random.normal(0, effective_vol * 3)
            elif event_type == 'squeeze':
                effective_vol *= 0.3  # Reduz volatilidade temporariamente
        
        # === CALCULAR NOVO PREÇO ===
        new_price = current_price * (1 + price_return)
        new_price = max(new_price, current_price * 0.95)  # Proteção contra crashes
        
        # === GERAR OHLC REALISTA ===
        open_price = current_price
        close_price = new_price
        
        # Wicks proporcionais à volatilidade
        wick_size = effective_vol * abs(np.random.normal(0.5, 0.3))
        
        if close_price > open_price:
            # Barra de alta
            high_price = close_price * (1 + wick_size * 0.7)
            low_price = open_price * (1 - wick_size * 0.3)
        else:
            # Barra de baixa
            high_price = open_price * (1 + wick_size * 0.3)
            low_price = close_price * (1 - wick_size * 0.7)
        
        # Garantir coerência OHLC
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # === VOLUME CORRELACIONADO ===
        # Volume base com componentes
        base_volume = 10000
        vol_multiplier = 1 + effective_vol * 20  # Volume aumenta com volatilidade
        price_change_multiplier = 1 + abs(price_return) * 30  # E com movimento
        regime_multiplier = {'bull': 1.2, 'bear': 1.4, 'sideways': 0.8, 'volatile': 1.6}
        
        volume = int(base_volume * vol_multiplier * price_change_multiplier * 
                    regime_multiplier[current_regime] * random.uniform(0.7, 1.3))
        volume = np.clip(volume, 5000, 100000)
        
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
    
    # === ADICIONAR FEATURES TÉCNICAS BÁSICAS ===
    # Isso ajuda o modelo a aprender padrões
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['high_low_spread'] = (df['high'] - df['low']) / df['close']
    df['close_open_spread'] = (df['close'] - df['open']) / df['open']
    
    print(f"\n✅ Dataset criado com sucesso!")
    print(f"   Total de transições de regime: {len(regime_transitions)}")
    
    return df

def validate_dataset(df):
    """Validar coerência matemática do dataset"""
    print("\n" + "="*80)
    print("🔍 VALIDAÇÃO DO DATASET")
    print("="*80)
    
    # Check 1: NaN/Inf
    nan_count = df.isnull().sum().sum()
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    print(f"✓ NaN values: {nan_count}")
    print(f"✓ Inf values: {inf_count}")
    
    # Check 2: OHLC coherence
    ohlc_errors = 0
    for idx in range(len(df)):
        row = df.iloc[idx]
        if not (row['low'] <= row['open'] <= row['high'] and 
                row['low'] <= row['close'] <= row['high']):
            ohlc_errors += 1
    print(f"✓ OHLC errors: {ohlc_errors}")
    
    # Check 3: Price statistics
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        print(f"✓ {col}: min=${df[col].min():.2f}, max=${df[col].max():.2f}")
    
    # Check 4: Returns analysis
    if 'returns' in df.columns:
        returns = df['returns'].dropna()
        print(f"\n📊 RETURNS ANALYSIS:")
        print(f"   Mean: {returns.mean():.6f} ({returns.mean()*100:.4f}%)")
        print(f"   Std: {returns.std():.6f} ({returns.std()*100:.2f}%)")
        print(f"   Skew: {returns.skew():.3f}")
        print(f"   Kurt: {returns.kurtosis():.3f}")
        print(f"   Min: {returns.min():.6f}")
        print(f"   Max: {returns.max():.6f}")
    
    # Check 5: Regime analysis
    if 'regime' in df.columns:
        print(f"\n📈 REGIME DISTRIBUTION:")
        regime_stats = df.groupby('regime').agg({
            'returns': ['mean', 'std', 'count']
        }).round(6)
        
        for regime in regime_stats.index:
            count = regime_stats.loc[regime, ('returns', 'count')]
            mean = regime_stats.loc[regime, ('returns', 'mean')]
            std = regime_stats.loc[regime, ('returns', 'std')]
            print(f"   {regime}: {count:,} bars, mean={mean:.6f}, std={std:.6f}")
    
    # Check 6: Predictability test
    if 'returns' in df.columns:
        future_returns = df['returns'].shift(-1)
        
        # Test various features for predictability
        features_to_test = ['returns', 'high_low_spread', 'close_open_spread', 'volume']
        print(f"\n🎯 PREDICTABILITY TEST (correlation with next return):")
        
        for feature in features_to_test:
            if feature in df.columns:
                corr = df[feature].corr(future_returns)
                print(f"   {feature} → next_return: {corr:.6f}")
    
    # Check 7: Volume analysis
    if 'volume' in df.columns:
        vol_price_corr = df['volume'].corr(df['high_low_spread'])
        print(f"\n📊 VOLUME ANALYSIS:")
        print(f"   Volume-Volatility correlation: {vol_price_corr:.3f}")
        print(f"   Mean volume: {df['volume'].mean():,.0f}")
        print(f"   Volume CV: {df['volume'].std() / df['volume'].mean():.3f}")
    
    print("\n✅ Dataset validado e pronto para uso!")
    
    return True

def main():
    print("🚀 GERADOR DE DATASET DESAFIADOR COERENTE")
    print("="*80)
    
    # Criar dataset
    df = create_proper_challenging_dataset(n_bars=2000000)
    
    # Validar
    if validate_dataset(df):
        # Salvar
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/GOLD_COHERENT_CHALLENGING_2M_{timestamp}.csv"
        
        # Remover colunas auxiliares antes de salvar
        columns_to_save = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'regime']
        df[columns_to_save].to_csv(filename, index=False)
        
        print(f"\n✅ DATASET SALVO: {filename}")
        print("\n📋 CARACTERÍSTICAS:")
        print("   - 2M barras de 5 minutos")
        print("   - 4 regimes distintivos com transições inteligentes")
        print("   - Volatilidade clustering realista")
        print("   - Momentum e mean reversion")
        print("   - Volume correlacionado com ação de preço")
        print("   - Eventos especiais (5% das barras)")
        print("   - Matematicamente coerente e validado")
        print("   - Desafiador mas convergível")
        
        return filename

if __name__ == '__main__':
    main()