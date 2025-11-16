#!/usr/bin/env python3
"""
Análise Detalhada do CSV de Treinamento SILUS
==============================================
Identificar problemas específicos no arquivo de 9.4MB
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_training_csv():
    """Análise profunda do CSV de treinamento"""
    
    # Arquivo correto de 9.4MB
    csv_path = Path("D:/Projeto/Otimizacao/treino_principal/models/SILUS/SILUS_training_metrics_20250827_220321.csv")
    
    print("="*80)
    print("🔍 ANÁLISE DETALHADA DO CSV DE TREINAMENTO SILUS")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print(f"\n📂 Arquivo: {csv_path.name}")
    print(f"   Tamanho: {csv_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Carregar CSV
    print("\n⏳ Carregando dados...")
    df = pd.read_csv(csv_path)
    
    print(f"\n📊 Dados carregados:")
    print(f"   Total de linhas: {len(df):,}")
    print(f"   Total de colunas: {len(df.columns)}")
    print(f"   Período: Step {df['step'].min()} até {df['step'].max()}")
    
    # Análise de colunas
    print(f"\n📋 Colunas disponíveis:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    
    # Estatísticas básicas
    print("\n="*80)
    print("📈 ESTATÍSTICAS GERAIS")
    print("="*80)
    
    key_metrics = {
        'Episode Rewards': 'episode_reward',
        'Sharpe Ratio': 'sharpe_ratio',
        'Portfolio Value': 'portfolio_value',
        'Win Rate': 'win_rate',
        'Policy Loss': 'policy_loss',
        'Value Loss': 'value_loss',
        'Entropy Loss': 'entropy_loss',
        'Learning Rate': 'learning_rate',
        'Clip Fraction': 'clip_fraction',
        'Gradient Norm': 'gradient_norm'
    }
    
    for name, col in key_metrics.items():
        if col in df.columns:
            print(f"\n{name}:")
            print(f"   Min: {df[col].min():.6f}")
            print(f"   Max: {df[col].max():.6f}")
            print(f"   Mean: {df[col].mean():.6f}")
            print(f"   Std: {df[col].std():.6f}")
            
            # Verificar zeros
            zeros = (df[col] == 0).sum()
            if zeros > 0:
                print(f"   ⚠️ Zeros: {zeros} ({zeros/len(df)*100:.1f}%)")
    
    # Análise de problemas
    print("\n="*80)
    print("🚨 PROBLEMAS IDENTIFICADOS")
    print("="*80)
    
    # 1. Episode rewards zerados
    if 'episode_reward' in df.columns:
        zero_rewards = (df['episode_reward'] == 0).sum()
        print(f"\n1️⃣ Episode Rewards Zerados: {zero_rewards}/{len(df)} ({zero_rewards/len(df)*100:.1f}%)")
        if zero_rewards == len(df):
            print("   ❌ CRÍTICO: Todos os episode rewards são ZERO!")
            print("   → Episodes não estão terminando corretamente")
    
    # 2. Sharpe sempre zero
    if 'sharpe_ratio' in df.columns:
        non_zero_sharpe = (df['sharpe_ratio'] != 0).sum()
        print(f"\n2️⃣ Sharpe Ratio Não-Zero: {non_zero_sharpe}/{len(df)} ({non_zero_sharpe/len(df)*100:.1f}%)")
        if non_zero_sharpe == 0:
            print("   ❌ CRÍTICO: Sharpe Ratio sempre ZERO!")
            print("   → Não está sendo calculado corretamente")
    
    # 3. Entropia colapsando
    if 'entropy_loss' in df.columns:
        entropy_start = df['entropy_loss'].iloc[:100].mean()
        entropy_end = df['entropy_loss'].iloc[-100:].mean()
        print(f"\n3️⃣ Colapso de Entropia:")
        print(f"   Início: {entropy_start:.2f}")
        print(f"   Final: {entropy_end:.2f}")
        print(f"   Mudança: {(entropy_end - entropy_start):.2f}")
        if entropy_end < -10:
            print("   ⚠️ Entropia muito negativa - política determinística demais")
    
    # 4. Win rate degradando
    if 'win_rate' in df.columns:
        wr_start = df['win_rate'].iloc[:1000].mean()
        wr_end = df['win_rate'].iloc[-1000:].mean()
        print(f"\n4️⃣ Degradação de Win Rate:")
        print(f"   Início: {wr_start:.1f}%")
        print(f"   Final: {wr_end:.1f}%")
        print(f"   Mudança: {(wr_end - wr_start):.1f}%")
    
    # 5. Portfolio resets
    if 'portfolio_value' in df.columns:
        resets = (df['portfolio_value'] == 500).sum()
        print(f"\n5️⃣ Portfolio Resets (valor=500): {resets}")
        print(f"   Frequência: a cada {len(df)/resets:.1f} steps")
    
    # Análise temporal
    print("\n="*80)
    print("📊 EVOLUÇÃO TEMPORAL")
    print("="*80)
    
    # Dividir em quartis
    quartiles = np.array_split(df, 4)
    
    for i, q in enumerate(quartiles, 1):
        print(f"\n🔸 Quartil {i} (Steps {q['step'].min()}-{q['step'].max()}):")
        
        if 'episode_reward' in q.columns:
            print(f"   Episode Reward: {q['episode_reward'].mean():.4f}")
        if 'sharpe_ratio' in q.columns:
            print(f"   Sharpe Ratio: {q['sharpe_ratio'].mean():.4f}")
        if 'win_rate' in q.columns:
            print(f"   Win Rate: {q['win_rate'].mean():.1f}%")
        if 'portfolio_value' in q.columns:
            print(f"   Portfolio: {q['portfolio_value'].mean():.2f}")
        if 'policy_loss' in q.columns:
            print(f"   Policy Loss: {q['policy_loss'].mean():.4f}")
        if 'entropy_loss' in q.columns:
            print(f"   Entropy Loss: {q['entropy_loss'].mean():.2f}")
    
    # Salvar subset para análise manual
    print("\n💾 Salvando subset para análise...")
    subset = df[['step', 'episode_reward', 'sharpe_ratio', 'win_rate', 
                 'portfolio_value', 'policy_loss', 'value_loss', 
                 'entropy_loss']].iloc[::100]  # A cada 100 steps
    
    subset_path = Path("D:/Projeto/avaliacao/silus_training_subset.csv")
    subset.to_csv(subset_path, index=False)
    print(f"   Salvo em: {subset_path}")
    
    # Conclusões
    print("\n="*80)
    print("💡 CONCLUSÕES E RECOMENDAÇÕES")
    print("="*80)
    
    print("""
PROBLEMAS PRINCIPAIS:
1. Episodes não terminam → rewards sempre zero
2. Sharpe não calculado → sem otimização risk-adjusted
3. Entropia colapsou → política determinística
4. Win rate degradou → overfitting ou reward inadequado

SOLUÇÕES URGENTES:
1. Corrigir lógica de episódios no silus.py
2. Implementar cálculo correto de Sharpe
3. Aumentar coeficiente de entropia (ent_coef)
4. Revisar sistema de rewards V4 INNO

PARÂMETROS SUGERIDOS:
- ent_coef: 0.01 → 0.05
- learning_rate: 8e-5 → 3e-5
- clip_range: 0.2 → 0.1
- Confidence threshold: 0.6 → 0.75
""")
    
    return df

if __name__ == "__main__":
    df = analyze_training_csv()