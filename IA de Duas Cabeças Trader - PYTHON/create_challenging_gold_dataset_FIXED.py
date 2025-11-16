#!/usr/bin/env python3
"""
DATASET DESAFIADOR CORRIGIDO - SEM BUGS DE NaN
O anterior tinha lógica falha na geração de preços
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def create_robust_challenging_dataset(n_bars=2000000, base_price=2000.0):
    """
    Criar dataset desafiador MAS ESTÁVEL - sem NaN ou valores impossíveis
    """
    print(f"Criando dataset desafiador CORRIGIDO com {n_bars:,} barras...")
    
    data = []
    current_price = base_price
    current_time = datetime(2023, 1, 1)
    
    # Regimes com parâmetros REALISTAS
    regimes = {
        'bull': {
            'drift': 0.00008,           # 0.008% drift por barra (realista)
            'volatility': 0.015,        # 1.5% volatilidade 
            'up_prob': 0.52,            # 52% chances up (ligeiro bias)
            'duration_range': (100000, 300000)
        },
        'bear': {  
            'drift': -0.00005,          # -0.005% drift por barra
            'volatility': 0.022,        # 2.2% volatilidade (maior em bear)
            'up_prob': 0.48,            # 48% chances up
            'duration_range': (50000, 200000)
        },
        'sideways': {
            'drift': 0.0,               # Zero drift
            'volatility': 0.008,        # 0.8% volatilidade baixa
            'up_prob': 0.50,            # 50% chances up
            'duration_range': (80000, 250000)
        }
    }
    
    # Estado inicial
    current_regime = 'bull'
    regime_bars_left = random.randint(*regimes[current_regime]['duration_range'])
    
    # Volatility clustering
    vol_multiplier = 1.0
    vol_persistence = 0.98  # Alta persistência
    
    # Extreme events
    next_extreme_event = random.randint(200000, 400000)
    extreme_event_cooldown = 0
    
    print(f"Regime inicial: {current_regime} por {regime_bars_left:,} barras")
    
    for i in range(n_bars):
        if i % 200000 == 0:
            print(f"Progresso: {i:,}/{n_bars:,} ({i/n_bars*100:.1f}%) - Preço atual: {current_price:.2f}")
        
        # === REGIME SWITCHING ===
        if regime_bars_left <= 0:
            # Transição suave entre regimes
            available_regimes = [r for r in regimes.keys() if r != current_regime]
            current_regime = random.choice(available_regimes)
            regime_bars_left = random.randint(*regimes[current_regime]['duration_range'])
            print(f"  Regime: {current_regime} por {regime_bars_left:,} barras (barra {i:,}, preço {current_price:.2f})")
        
        regime_info = regimes[current_regime]
        
        # === VOLATILITY CLUSTERING ===
        vol_shock = np.random.normal(0, 0.05)  # Shock na volatilidade
        vol_multiplier = vol_persistence * vol_multiplier + (1 - vol_persistence) * 1.0 + vol_shock
        vol_multiplier = np.clip(vol_multiplier, 0.5, 2.5)  # Limitar entre 0.5x e 2.5x
        
        effective_volatility = regime_info['volatility'] * vol_multiplier
        
        # === EVENTOS EXTREMOS ===
        extreme_multiplier = 1.0
        if i >= next_extreme_event and extreme_event_cooldown <= 0:
            event_type = random.choice(['flash_crash', 'gap_up', 'gap_down', 'vol_spike'])
            
            if event_type == 'flash_crash':
                extreme_multiplier = 0.97  # -3% máximo
                effective_volatility *= 2.0
                extreme_event_cooldown = 50000  # Cooldown
                print(f"  Flash crash na barra {i:,} (preço: {current_price:.2f})")
                
            elif event_type == 'gap_up':
                extreme_multiplier = 1.015  # +1.5% gap
                extreme_event_cooldown = 30000
                print(f"  Gap up na barra {i:,}")
                
            elif event_type == 'gap_down':
                extreme_multiplier = 0.985  # -1.5% gap
                extreme_event_cooldown = 30000
                print(f"  Gap down na barra {i:,}")
                
            elif event_type == 'vol_spike':
                effective_volatility *= 2.5
                extreme_event_cooldown = 20000
                print(f"  Volatility spike na barra {i:,}")
            
            next_extreme_event = i + random.randint(200000, 500000)
        
        if extreme_event_cooldown > 0:
            extreme_event_cooldown -= 1
        
        # === GERAÇÃO DE PREÇOS ROBUSTA ===
        # Drift do regime
        drift_component = regime_info['drift']
        
        # Componente aleatório baseado em probabilidade
        random_component = 0
        if random.random() < regime_info['up_prob']:
            # Movimento para cima
            random_component = np.abs(np.random.normal(0, effective_volatility))
        else:
            # Movimento para baixo  
            random_component = -np.abs(np.random.normal(0, effective_volatility))
        
        # Return total (sem valores extremos)
        total_return = drift_component + random_component
        total_return = np.clip(total_return, -0.05, 0.05)  # Clip em ±5% máximo
        
        # Aplicar evento extremo
        if extreme_multiplier != 1.0:
            total_return = (extreme_multiplier - 1.0)
        
        # Calcular novo preço (SEMPRE POSITIVO)
        new_price = current_price * (1 + total_return)
        new_price = max(new_price, current_price * 0.95)  # Nunca cair mais que 5%
        new_price = min(new_price, current_price * 1.05)  # Nunca subir mais que 5%
        
        # === OHLC GENERATION ROBUSTA ===
        open_price = current_price
        close_price = new_price
        
        # High/Low com lógica correta
        if close_price >= open_price:
            # Barra verde
            high_multiplier = 1 + random.uniform(0, effective_volatility * 0.3)
            low_multiplier = 1 - random.uniform(0, effective_volatility * 0.2)
            high_price = close_price * high_multiplier
            low_price = open_price * low_multiplier
        else:
            # Barra vermelha
            high_multiplier = 1 + random.uniform(0, effective_volatility * 0.2) 
            low_multiplier = 1 - random.uniform(0, effective_volatility * 0.3)
            high_price = open_price * high_multiplier
            low_price = close_price * low_multiplier
        
        # Garantir que high >= max(open, close) e low <= min(open, close)
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Volume correlacionado com volatilidade e movimento
        price_change_abs = abs(close_price - open_price) / open_price
        base_volume = 15000
        vol_factor = 1 + (effective_volatility * 10) + (price_change_abs * 5)
        volume = int(base_volume * vol_factor * random.uniform(0.7, 1.3))
        volume = max(5000, min(100000, volume))  # Limitar volume
        
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

def validate_dataset(df):
    """Validar dataset antes de usar - ZERO TOLERÂNCIA para NaN/inf"""
    print("\\n" + "="*60)
    print("VALIDAÇÃO RIGOROSA DO DATASET")
    print("="*60)
    
    # Check 1: NaN values
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        print(f"❌ FALHA: {nan_count} valores NaN encontrados!")
        return False
    else:
        print("✅ Zero valores NaN")
    
    # Check 2: Infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    if inf_count > 0:
        print(f"❌ FALHA: {inf_count} valores infinitos encontrados!")
        return False
    else:
        print("✅ Zero valores infinitos")
    
    # Check 3: Negative prices
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in df.columns:
            min_price = df[col].min()
            if min_price <= 0:
                print(f"❌ FALHA: Preços negativos em {col}: min={min_price}")
                return False
    print("✅ Todos os preços são positivos")
    
    # Check 4: OHLC logic
    invalid_ohlc = 0
    for idx in range(len(df)):
        row = df.iloc[idx]
        if not (row['low'] <= row['open'] <= row['high'] and 
                row['low'] <= row['close'] <= row['high']):
            invalid_ohlc += 1
    
    if invalid_ohlc > 0:
        print(f"❌ FALHA: {invalid_ohlc} barras com OHLC inválido!")
        return False
    else:
        print("✅ Lógica OHLC correta")
    
    # Check 5: Returns extremos
    df['returns'] = df['close'].pct_change()
    max_return = df['returns'].max()
    min_return = df['returns'].min()
    volatility = df['returns'].std()
    
    print(f"\\nESTATÍSTICAS:")
    print(f"  Volatilidade: {volatility:.4f} ({volatility*100:.2f}%)")
    print(f"  Max return: {max_return:.4f} ({max_return*100:.2f}%)")
    print(f"  Min return: {min_return:.4f} ({min_return*100:.2f}%)")
    
    if abs(max_return) > 0.1 or abs(min_return) > 0.1:
        print("❌ FALHA: Returns extremos (>10%) que podem causar NaN!")
        return False
    else:
        print("✅ Returns dentro de limites seguros")
    
    # Check 6: Regime distribution
    regime_counts = df['regime'].value_counts()
    print(f"\\nREGIMES:")
    for regime, count in regime_counts.items():
        print(f"  {regime}: {count:,} ({count/len(df)*100:.1f}%)")
    
    print(f"\\n✅ DATASET VALIDADO - SEGURO PARA USO!")
    print(f"Shape: {df.shape}")
    print(f"Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
    
    return True

def main():
    print("CRIANDO DATASET DESAFIADOR CORRIGIDO")
    print("="*80)
    
    # Criar dataset
    df = create_robust_challenging_dataset(n_bars=2000000)
    
    # Validar ANTES de salvar
    if not validate_dataset(df):
        print("\\n❌ DATASET FALHOU NA VALIDAÇÃO - NÃO SERÁ SALVO!")
        return
    
    # Salvar apenas se passou na validação
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/GOLD_CHALLENGING_SAFE_2M_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\\n✅ Dataset seguro salvo: {filename}")
    
    print("\\n" + "="*80)
    print("DATASET DESAFIADOR MAS SEGURO CRIADO!")
    print(f"Arquivo: {filename}")
    print("Características:")
    print("- Regimes dinâmicos com transições suaves")
    print("- Volatilidade clustering realista")
    print("- Eventos extremos controlados")
    print("- Validação rigorosa (zero NaN/inf)")
    print("- OHLC logic perfeita")
    print("- Returns limitados (máx ±5%)")
    print("\\nEste dataset é DESAFIADOR mas não vai quebrar o modelo!")

if __name__ == '__main__':
    main()