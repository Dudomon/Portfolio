#!/usr/bin/env python3
"""
🔍 ANÁLISE DE PADRÕES REAIS DO OURO
Extrair características autênticas para o dataset sintético
"""

import numpy as np
import pandas as pd
import os
from scipy import stats
import pickle

def analyze_real_gold_data():
    """Analisar dados reais do ouro para extrair padrões"""
    print("ANALISE DE PADROES REAIS DO OURO")
    print("="*50)
    
    # Tentar carregar dados reais
    data_files = [
        "data/GC=F_YAHOO_DAILY_5MIN_20250711_041924.csv",
        "data/GC=F_YAHOO_DAILY_5MIN_20250704_142845.csv", 
        "data/GOLD_5m_20250704_122438.csv"
    ]
    
    df = None
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"Carregando: {data_file}")
            df = pd.read_csv(data_file)
            break
    
    if df is None:
        print("ERRO: Nenhum arquivo de dados encontrado!")
        return None
    
    # Encontrar coluna de preço
    price_col = None
    for col in ['close', 'Close', 'close_5m', 'CLOSE']:
        if col in df.columns:
            price_col = col
            break
    
    if price_col is None:
        print("ERRO: Coluna de preço não encontrada!")
        print("Colunas disponíveis:", list(df.columns))
        return None
    
    prices = df[price_col].values
    print(f"Dados carregados: {len(prices)} barras")
    print(f"Período: ${prices[0]:.2f} - ${prices[-1]:.2f}")
    
    # ANÁLISE 1: DISTRIBUIÇÃO DE RETORNOS
    print(f"\n1. DISTRIBUICAO DE RETORNOS:")
    print("-" * 30)
    
    returns = np.diff(prices) / prices[:-1]
    returns_pct = returns * 100
    
    print(f"Retorno médio: {np.mean(returns_pct):.6f}%")
    print(f"Volatilidade: {np.std(returns_pct):.4f}%")
    print(f"Skewness: {stats.skew(returns_pct):.4f}")
    print(f"Kurtosis: {stats.kurtosis(returns_pct):.4f}")
    
    # ANÁLISE 2: REGIMES DE VOLATILIDADE
    print(f"\n2. REGIMES DE VOLATILIDADE:")
    print("-" * 30)
    
    # Volatilidade rolling
    vol_window = 50
    rolling_vol = pd.Series(returns_pct).rolling(vol_window).std()
    
    # Definir thresholds baseados nos dados reais
    vol_25 = np.percentile(rolling_vol.dropna(), 25)
    vol_50 = np.percentile(rolling_vol.dropna(), 50) 
    vol_75 = np.percentile(rolling_vol.dropna(), 75)
    vol_90 = np.percentile(rolling_vol.dropna(), 90)
    
    print(f"Vol 25%: {vol_25:.4f}%")
    print(f"Vol 50%: {vol_50:.4f}%") 
    print(f"Vol 75%: {vol_75:.4f}%")
    print(f"Vol 90%: {vol_90:.4f}%")
    
    # Classificar regimes
    low_vol_count = np.sum(rolling_vol <= vol_25)
    med_vol_count = np.sum((rolling_vol > vol_25) & (rolling_vol <= vol_75))
    high_vol_count = np.sum((rolling_vol > vol_75) & (rolling_vol <= vol_90))
    extreme_vol_count = np.sum(rolling_vol > vol_90)
    
    total_valid = len(rolling_vol.dropna())
    
    print(f"\nDistribuição real observada:")
    print(f"Baixa vol: {low_vol_count/total_valid*100:.1f}%")
    print(f"Média vol: {med_vol_count/total_valid*100:.1f}%")
    print(f"Alta vol: {high_vol_count/total_valid*100:.1f}%")
    print(f"Extrema vol: {extreme_vol_count/total_valid*100:.1f}%")
    
    # ANÁLISE 3: PADRÕES INTRADAY
    print(f"\n3. PADROES INTRADAY:")
    print("-" * 30)
    
    # Assumir que temos timestamp ou criar um sintético
    if 'timestamp' in df.columns or 'date' in df.columns or 'datetime' in df.columns:
        # Usar timestamp real se disponível
        pass
    else:
        # Criar timestamps sintéticos (5min bars)
        df['synthetic_time'] = pd.date_range('2023-01-01', periods=len(df), freq='5min')
    
    # Análise por hora do dia
    df['hour'] = pd.to_datetime(df.get('synthetic_time', df.index)).dt.hour
    hourly_vol = df.groupby('hour')[price_col].pct_change().std() * 100
    
    print("Volatilidade por hora:")
    for hour in range(0, 24, 2):
        if hour in hourly_vol.index:
            print(f"  {hour:02d}h: {hourly_vol[hour]:.4f}%")
    
    # ANÁLISE 4: SUPORTE E RESISTÊNCIA
    print(f"\n4. NIVEIS DE SUPORTE/RESISTENCIA:")
    print("-" * 30)
    
    # Encontrar níveis psicológicos
    price_levels = []
    price_min, price_max = prices.min(), prices.max()
    
    # Níveis round numbers
    for level in range(int(price_min//50)*50, int(price_max//50)*50 + 100, 50):
        if price_min <= level <= price_max:
            # Contar quantas vezes o preço "tocou" este nível (±2$)
            touches = np.sum(np.abs(prices - level) <= 2)
            if touches >= 5:  # Mínimo 5 toques
                price_levels.append({'level': level, 'touches': touches})
    
    print("Principais níveis psicológicos:")
    for level_info in sorted(price_levels, key=lambda x: x['touches'], reverse=True)[:5]:
        print(f"  ${level_info['level']}: {level_info['touches']} toques")
    
    # ANÁLISE 5: PADRÕES DE MOVIMENTO
    print(f"\n5. PADROES DE MOVIMENTO:")
    print("-" * 30)
    
    # Sequências de alta/baixa
    price_direction = np.sign(np.diff(prices))
    
    # Contar sequências
    sequences = []
    current_seq = 1
    current_dir = price_direction[0]
    
    for direction in price_direction[1:]:
        if direction == current_dir:
            current_seq += 1
        else:
            sequences.append(current_seq)
            current_seq = 1
            current_dir = direction
    sequences.append(current_seq)
    
    avg_sequence = np.mean(sequences)
    max_sequence = max(sequences)
    
    print(f"Sequência média: {avg_sequence:.1f} barras")
    print(f"Sequência máxima: {max_sequence} barras")
    
    # EXTRAIR PARÂMETROS PARA SÍNTESE
    print(f"\n6. PARAMETROS EXTRAIDOS PARA SINTESE:")
    print("-" * 30)
    
    synthetic_params = {
        'base_price': np.mean(prices),
        'daily_drift': np.mean(returns_pct) / 100,
        'base_volatility': np.std(returns_pct) / 100,
        'skewness': stats.skew(returns_pct),
        'kurtosis': stats.kurtosis(returns_pct),
        'vol_regimes': {
            'low': {'threshold': vol_25/100, 'prob': low_vol_count/total_valid},
            'medium': {'threshold': vol_75/100, 'prob': med_vol_count/total_valid}, 
            'high': {'threshold': vol_90/100, 'prob': high_vol_count/total_valid},
            'extreme': {'threshold': vol_90/100*2, 'prob': extreme_vol_count/total_valid}
        },
        'intraday_pattern': hourly_vol.to_dict(),
        'sequence_stats': {
            'avg_sequence': avg_sequence,
            'max_sequence': max_sequence
        },
        'support_resistance': price_levels,
        'price_range': {'min': float(price_min), 'max': float(price_max)}
    }
    
    print("Parâmetros extraídos e salvos!")
    
    # Salvar parâmetros
    with open('real_gold_analysis_params.pkl', 'wb') as f:
        pickle.dump(synthetic_params, f)
    
    return synthetic_params

if __name__ == "__main__":
    params = analyze_real_gold_data()
    if params:
        print(f"\nParametros salvos em: real_gold_analysis_params.pkl")
        print("Pronto para gerar dataset sintético baseado em dados reais!")