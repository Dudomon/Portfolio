#!/usr/bin/env python3
"""
INVESTIGAÇÃO PRISTINA DO DATASET
Verificar se o dataset é realmente desafiador ou se tem padrões óbvios
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def investigate_dataset_deep():
    """Investigação profunda do dataset"""
    dataset_path = 'data/GOLD_SAFE_CHALLENGING_2M_20250801_203251.csv'
    
    print("INVESTIGACAO PRISTINA DO DATASET")
    print("="*60)
    print(f"Arquivo: {dataset_path}")
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"\\nDADOS BASICOS:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # === TESTE 1: QUALIDADE DOS PRECOS ===
        print(f"\\n=== TESTE 1: QUALIDADE DOS PRECOS ===")
        
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                min_price = df[col].min()
                max_price = df[col].max()
                mean_price = df[col].mean()
                std_price = df[col].std()
                
                print(f"  {col}: min={min_price:.2f}, max={max_price:.2f}, mean={mean_price:.2f}, std={std_price:.2f}")
                
                # Verificar range realista
                if min_price < 100 or max_price > 10000:
                    print(f"    AVISO: Range de preço suspeito para {col}")
        
        # === TESTE 2: LOGICA OHLC ===
        print(f"\\n=== TESTE 2: LOGICA OHLC ===")
        
        ohlc_violations = 0
        for i in range(len(df)):
            row = df.iloc[i]
            # High deve ser >= max(open, close)
            # Low deve ser <= min(open, close)
            if not (row['low'] <= min(row['open'], row['close']) and 
                    row['high'] >= max(row['open'], row['close'])):
                ohlc_violations += 1
        
        print(f"  Violacoes OHLC: {ohlc_violations:,} ({ohlc_violations/len(df)*100:.3f}%)")
        if ohlc_violations == 0:
            print("  ✅ OHLC logic perfeita")
        else:
            print("  ❌ OHLC logic com problemas")
        
        # === TESTE 3: ANALISE DE RETURNS ===
        print(f"\\n=== TESTE 3: ANALISE DE RETURNS ===")
        
        df['returns'] = df['close'].pct_change()
        
        # Estatísticas básicas
        vol_daily = df['returns'].std()
        vol_annualized = vol_daily * np.sqrt(252 * 1440)  # 1440 minutos/dia
        skewness = df['returns'].skew()
        kurtosis = df['returns'].kurtosis()
        
        print(f"  Volatilidade diaria: {vol_daily:.4f} ({vol_daily*100:.2f}%)")
        print(f"  Volatilidade anualizada: {vol_annualized:.4f} ({vol_annualized*100:.1f}%)")
        print(f"  Skewness: {skewness:.3f}")
        print(f"  Kurtosis: {kurtosis:.3f}")
        
        # Verificar distribuição normal
        from scipy import stats
        ks_stat, ks_p_value = stats.kstest(df['returns'].dropna(), 'norm')
        print(f"  Teste normalidade (KS): stat={ks_stat:.4f}, p-value={ks_p_value:.4f}")
        
        if ks_p_value < 0.05:
            print("  ✅ Distribuição NÃO normal (realista)")
        else:
            print("  ❌ Distribuição muito normal (suspeito)")
        
        # === TESTE 4: AUTOCORRELACAO ===
        print(f"\\n=== TESTE 4: AUTOCORRELACAO ===")
        
        # Autocorrelação de returns (deve ser próxima de zero)
        autocorr_1 = df['returns'].autocorr(lag=1)
        autocorr_5 = df['returns'].autocorr(lag=5)
        autocorr_20 = df['returns'].autocorr(lag=20)
        
        print(f"  Autocorr lag-1: {autocorr_1:.4f}")
        print(f"  Autocorr lag-5: {autocorr_5:.4f}")
        print(f"  Autocorr lag-20: {autocorr_20:.4f}")
        
        if abs(autocorr_1) < 0.05:
            print("  ✅ Baixa autocorrelação (imprevisível)")
        else:
            print("  ❌ Alta autocorrelação (padrões óbvios)")
        
        # === TESTE 5: ANALISE DE REGIMES ===
        print(f"\\n=== TESTE 5: ANALISE DE REGIMES ===")
        
        if 'regime' in df.columns:
            regime_counts = df['regime'].value_counts()
            regime_transitions = 0
            
            for i in range(1, len(df)):
                if df.iloc[i]['regime'] != df.iloc[i-1]['regime']:
                    regime_transitions += 1
            
            print(f"  Distribuição de regimes:")
            for regime, count in regime_counts.items():
                pct = count / len(df) * 100
                print(f"    {regime}: {count:,} ({pct:.1f}%)")
            
            print(f"  Transições de regime: {regime_transitions:,}")
            print(f"  Frequência de transição: {regime_transitions/len(df)*100:.3f}%")
            
            # Analisar performance por regime
            regime_stats = df.groupby('regime')['returns'].agg(['mean', 'std']).round(6)
            print(f"  Performance por regime:")
            print(regime_stats)
        
        # === TESTE 6: PADRÕES SUSPEITOS ===
        print(f"\\n=== TESTE 6: PADROES SUSPEITOS ===")
        
        # Verificar sequências repetitivas
        price_diffs = df['close'].diff()
        
        # Contar valores zero consecutivos
        zero_runs = []
        current_run = 0
        for diff in price_diffs:
            if abs(diff) < 1e-10:  # Praticamente zero
                current_run += 1
            else:
                if current_run > 0:
                    zero_runs.append(current_run)
                current_run = 0
        
        if zero_runs:
            max_zero_run = max(zero_runs)
            avg_zero_run = sum(zero_runs) / len(zero_runs)
            print(f"  Sequências de preços iguais: {len(zero_runs)} sequências")
            print(f"  Maior sequência: {max_zero_run} barras")
            print(f"  Média sequência: {avg_zero_run:.1f} barras")
        else:
            print("  ✅ Sem sequências de preços iguais")
        
        # === TESTE 7: MICROESTRUTURA ===
        print(f"\\n=== TESTE 7: MICROESTRUTURA ===")
        
        # Spreads
        df['spread'] = (df['high'] - df['low']) / df['close']
        avg_spread = df['spread'].mean()
        std_spread = df['spread'].std()
        
        print(f"  Spread médio: {avg_spread:.4f} ({avg_spread*100:.2f}%)")
        print(f"  Spread std: {std_spread:.4f} ({std_spread*100:.2f}%)")
        
        # Volume
        if 'volume' in df.columns:
            vol_mean = df['volume'].mean()
            vol_std = df['volume'].std()
            vol_cv = vol_std / vol_mean  # Coefficient of variation
            
            print(f"  Volume médio: {vol_mean:,.0f}")
            print(f"  Volume CV: {vol_cv:.3f}")
            
            if vol_cv < 0.3:
                print("  ❌ Volume muito constante (suspeito)")
            else:
                print("  ✅ Volume com variação realista")
        
        # === TESTE 8: DIFICULDADE PARA RL ===
        print(f"\\n=== TESTE 8: DIFICULDADE PARA RL ===")
        
        # Calcular Sharpe ratio do buy-and-hold
        total_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
        sharpe_bh = (df['returns'].mean() / df['returns'].std()) * np.sqrt(252 * 1440)
        
        print(f"  Buy-and-hold return: {total_return:.4f} ({total_return*100:.2f}%)")
        print(f"  Buy-and-hold Sharpe: {sharpe_bh:.3f}")
        
        # Calcular máximo drawdown
        cumulative = (1 + df['returns']).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative / rolling_max) - 1
        max_dd = drawdown.min()
        
        print(f"  Máximo drawdown: {max_dd:.4f} ({max_dd*100:.2f}%)")
        
        # Dificuldade score
        difficulty_score = 0
        
        if vol_daily > 0.01:  # > 1% volatilidade
            difficulty_score += 1
        if abs(autocorr_1) < 0.05:  # Baixa previsibilidade
            difficulty_score += 1
        if abs(sharpe_bh) < 1.0:  # Buy-and-hold não muito bom
            difficulty_score += 1
        if abs(max_dd) > 0.1:  # Drawdown significativo
            difficulty_score += 1
        if vol_cv > 0.3:  # Volume variável
            difficulty_score += 1
        
        print(f"\\n=== SCORE DE DIFICULDADE: {difficulty_score}/5 ===")
        
        if difficulty_score >= 4:
            print("✅ DATASET DESAFIADOR - Adequado para RL")
        elif difficulty_score >= 2:
            print("⚠️ DATASET MODERADO - Pode ser muito fácil")
        else:
            print("❌ DATASET FÁCIL - Modelo vai overfit rapidamente")
        
        return difficulty_score >= 3
        
    except Exception as e:
        print(f"ERRO na investigação: {e}")
        return False

def main():
    print("INVESTIGACAO COMPLETA DO DATASET")
    print("Verificando se dataset é PRISTINO e desafiador...")
    
    is_challenging = investigate_dataset_deep()
    
    print(f"\\n" + "="*60)
    print("VEREDICTO FINAL")
    print("="*60)
    
    if is_challenging:
        print("✅ DATASET É ADEQUADO")
        print("  - Qualidade pristina")
        print("  - Dificuldade adequada para RL")
        print("  - Sem padrões óbvios")
    else:
        print("❌ DATASET PRECISA MELHORAR")
        print("  - Pode ser muito fácil")
        print("  - Padrões previsíveis")
        print("  - Modelo não vai aprender direito")
    
    print(f"\\nSe dataset está adequado, problema pode ser:")
    print("- Hiperparâmetros ainda muito conservadores")
    print("- Modelo já inicializado em estado 'bom'")
    print("- Normalização muito agressiva")

if __name__ == '__main__':
    main()