#!/usr/bin/env python3
"""
🔍 ANÁLISE COMPLETA DO TREINAMENTO SILUS - DIAGNÓSTICO DE PERFORMANCE
======================================================================

OBJETIVO: Entender por que Sharpe 0.23 é baixo e como melhorar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
import os

# Configuração de visualização
plt.style.use('dark_background')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)

def load_training_metrics():
    """Carregar todos os CSVs de métricas de treinamento"""
    
    # Pegar o CSV principal que cobre até 5M steps
    main_csv = "D:/Projeto/Otimizacao/treino_principal/models/SILUS/SILUS_training_metrics_20250828_091114.csv"
    
    print(f"📊 Carregando métricas de treinamento: {os.path.basename(main_csv)}")
    
    try:
        df = pd.read_csv(main_csv)
        print(f"✅ Carregado: {len(df)} linhas de dados")
        print(f"📅 Steps: {df['step'].min()} até {df['step'].max()}")
        return df
    except Exception as e:
        print(f"❌ Erro ao carregar: {e}")
        return None

def load_convergence_analysis():
    """Carregar análise de convergência"""
    
    conv_csv = "D:/Projeto/Otimizacao/treino_principal/models/SILUS/convergence_analysis_20250828_091114.csv"
    
    try:
        df = pd.read_csv(conv_csv)
        print(f"✅ Convergence analysis: {len(df)} linhas")
        return df
    except Exception as e:
        print(f"❌ Erro ao carregar convergence: {e}")
        return None

def load_reward_analysis():
    """Carregar análise de rewards"""
    
    reward_csv = "D:/Projeto/Otimizacao/treino_principal/models/SILUS/reward_analysis_20250828_091114.csv"
    
    try:
        df = pd.read_csv(reward_csv)
        print(f"✅ Reward analysis: {len(df)} linhas")
        return df
    except Exception as e:
        print(f"❌ Erro ao carregar rewards: {e}")
        return None

def load_trading_performance():
    """Carregar performance de trading"""
    
    trading_csv = "D:/Projeto/Otimizacao/treino_principal/models/SILUS/trading_performance_20250828_091114.csv"
    
    try:
        df = pd.read_csv(trading_csv)
        print(f"✅ Trading performance: {len(df)} linhas")
        return df
    except Exception as e:
        print(f"❌ Erro ao carregar trading: {e}")
        return None

def analyze_training_metrics(df):
    """Análise detalhada das métricas de treinamento"""
    
    print("\n" + "="*80)
    print("📊 ANÁLISE DE MÉTRICAS DE TREINAMENTO")
    print("="*80)
    
    # Verificar colunas disponíveis
    print(f"\n📋 Colunas disponíveis: {df.columns.tolist()}")
    
    # Estatísticas básicas em pontos-chave
    key_steps = [500000, 1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 3900000, 4000000, 4500000, 5000000]
    
    print("\n🎯 MÉTRICAS EM PONTOS-CHAVE:")
    print("-" * 80)
    
    metrics_summary = []
    
    for step in key_steps:
        # Pegar linha mais próxima do step
        closest_idx = (df['step'] - step).abs().idxmin()
        row = df.iloc[closest_idx]
        actual_step = row['step']
        
        # Coletar métricas principais
        metrics = {
            'Step': f"{actual_step/1e6:.2f}M",
            'Loss': row.get('loss', np.nan),
            'Value Loss': row.get('value_loss', np.nan),
            'Policy Loss': row.get('policy_loss', np.nan),
            'Entropy': row.get('entropy_loss', np.nan),
            'KL Div': row.get('approx_kl', np.nan),
            'Clip Frac': row.get('clip_fraction', np.nan),
            'Explained Var': row.get('explained_variance', np.nan),
            'Learning Rate': row.get('learning_rate', np.nan)
        }
        
        metrics_summary.append(metrics)
        
        # Print resumido
        if step in [1000000, 2000000, 3000000, 3900000, 4000000, 5000000]:
            print(f"\n📍 {metrics['Step']} steps:")
            print(f"   Loss: {metrics['Loss']:.4f}")
            print(f"   Value Loss: {metrics['Value Loss']:.4f}")
            print(f"   Policy Loss: {metrics['Policy Loss']:.4f}")
            print(f"   Entropy: {metrics['Entropy']:.4f}")
            print(f"   KL Divergence: {metrics['KL Div']:.6f}")
            print(f"   Clip Fraction: {metrics['Clip Frac']:.3f}")
            print(f"   Explained Var: {metrics['Explained Var']:.3f}")
            print(f"   LR: {metrics['Learning Rate']:.2e}")
    
    # Criar DataFrame resumido
    summary_df = pd.DataFrame(metrics_summary)
    
    # Análise de tendências
    print("\n" + "="*80)
    print("📈 ANÁLISE DE TENDÊNCIAS")
    print("="*80)
    
    # 1. Tendência do Loss
    print("\n1️⃣ TENDÊNCIA DO LOSS:")
    loss_start = df[df['step'] < 1000000]['loss'].mean()
    loss_mid = df[(df['step'] >= 2000000) & (df['step'] < 3000000)]['loss'].mean()
    loss_end = df[df['step'] > 4000000]['loss'].mean()
    
    print(f"   Início (< 1M): {loss_start:.4f}")
    print(f"   Meio (2-3M): {loss_mid:.4f}")
    print(f"   Fim (> 4M): {loss_end:.4f}")
    print(f"   Redução total: {(loss_start - loss_end)/loss_start*100:.1f}%")
    
    # 2. Estabilidade do Explained Variance
    print("\n2️⃣ EXPLAINED VARIANCE (CRÍTICO!):")
    ev_early = df[df['step'] < 1000000]['explained_variance'].mean()
    ev_sweet = df[(df['step'] >= 3800000) & (df['step'] <= 4000000)]['explained_variance'].mean()
    ev_late = df[df['step'] > 4500000]['explained_variance'].mean()
    
    print(f"   Early (< 1M): {ev_early:.3f}")
    print(f"   Sweet Spot (3.8-4M): {ev_sweet:.3f}")
    print(f"   Late (> 4.5M): {ev_late:.3f}")
    
    if ev_sweet > 0:
        print(f"   ⚠️ ALERTA: Explained Variance positivo no sweet spot!")
        print(f"   → Indica possível reward hacking ou overfitting ao reward")
    
    # 3. KL Divergence e Clip Fraction
    print("\n3️⃣ KL DIVERGENCE & CLIP FRACTION:")
    kl_avg = df['approx_kl'].mean()
    kl_std = df['approx_kl'].std()
    clip_avg = df['clip_fraction'].mean()
    clip_std = df['clip_fraction'].std()
    
    print(f"   KL Divergence médio: {kl_avg:.6f} (±{kl_std:.6f})")
    print(f"   Clip Fraction médio: {clip_avg:.3f} (±{clip_std:.3f})")
    
    if kl_avg < 0.01:
        print(f"   ⚠️ KL muito baixo - política mudando muito devagar!")
    if clip_avg > 0.2:
        print(f"   ⚠️ Clip fraction alto - mudanças muito agressivas!")
    
    # 4. Entropy Analysis
    print("\n4️⃣ ENTROPY (EXPLORAÇÃO):")
    entropy_start = df[df['step'] < 500000]['entropy_loss'].mean()
    entropy_end = df[df['step'] > 4500000]['entropy_loss'].mean()
    
    print(f"   Início: {entropy_start:.4f}")
    print(f"   Fim: {entropy_end:.4f}")
    print(f"   Redução: {(entropy_start - entropy_end)/entropy_start*100:.1f}%")
    
    if entropy_end < 0.01:
        print(f"   ⚠️ Entropy muito baixa - pouca exploração!")
    
    return summary_df

def analyze_convergence(df):
    """Análise de convergência"""
    
    if df is None:
        return
    
    print("\n" + "="*80)
    print("🎯 ANÁLISE DE CONVERGÊNCIA")
    print("="*80)
    
    # Verificar colunas
    print(f"\n📋 Colunas: {df.columns.tolist()}")
    
    # Estatísticas de convergência
    if 'gradient_norm' in df.columns:
        grad_norm_avg = df['gradient_norm'].mean()
        grad_norm_std = df['gradient_norm'].std()
        print(f"\n📊 Gradient Norm: {grad_norm_avg:.6f} (±{grad_norm_std:.6f})")
        
        if grad_norm_avg < 0.001:
            print("   ⚠️ Gradientes muito pequenos - aprendizado lento!")
    
    if 'policy_stability' in df.columns:
        stability = df['policy_stability'].mean()
        print(f"📊 Policy Stability: {stability:.3f}")
        
        if stability < 0.8:
            print("   ⚠️ Política instável!")

def analyze_rewards(df):
    """Análise de rewards"""
    
    if df is None:
        return
    
    print("\n" + "="*80)
    print("💰 ANÁLISE DE REWARDS")
    print("="*80)
    
    # Estatísticas de reward
    if 'mean_reward' in df.columns:
        reward_trend = df.groupby(df.index // 100)['mean_reward'].mean()
        
        print(f"\n📊 Reward médio geral: {df['mean_reward'].mean():.4f}")
        print(f"📊 Reward máximo: {df['mean_reward'].max():.4f}")
        print(f"📊 Reward mínimo: {df['mean_reward'].min():.4f}")
        
        # Tendência
        early_reward = df.iloc[:len(df)//3]['mean_reward'].mean()
        late_reward = df.iloc[-len(df)//3:]['mean_reward'].mean()
        
        print(f"\n📈 Tendência:")
        print(f"   Early: {early_reward:.4f}")
        print(f"   Late: {late_reward:.4f}")
        print(f"   Melhoria: {(late_reward - early_reward)/abs(early_reward)*100:.1f}%")

def analyze_trading(df):
    """Análise de trading performance"""
    
    if df is None:
        return
    
    print("\n" + "="*80)
    print("💹 ANÁLISE DE TRADING PERFORMANCE")
    print("="*80)
    
    # Verificar colunas
    print(f"\n📋 Colunas: {df.columns.tolist()}")
    
    # Estatísticas de trading
    if 'win_rate' in df.columns:
        wr_avg = df['win_rate'].mean()
        print(f"\n📊 Win Rate médio: {wr_avg:.1f}%")
    
    if 'sharpe_ratio' in df.columns:
        sharpe_avg = df['sharpe_ratio'].mean()
        sharpe_max = df['sharpe_ratio'].max()
        print(f"📊 Sharpe médio: {sharpe_avg:.3f}")
        print(f"📊 Sharpe máximo: {sharpe_max:.3f}")
        
        if sharpe_avg < 0.5:
            print("   ⚠️ Sharpe muito baixo para trading real!")
    
    if 'profit_factor' in df.columns:
        pf_avg = df['profit_factor'].mean()
        print(f"📊 Profit Factor médio: {pf_avg:.2f}")

def provide_recommendations():
    """Recomendações para melhorar performance"""
    
    print("\n" + "="*80)
    print("🚀 RECOMENDAÇÕES PARA MELHORAR PERFORMANCE")
    print("="*80)
    
    recommendations = """
    
1️⃣ PROBLEMA: Sharpe Ratio baixo (0.23)
   
   CAUSAS PROVÁVEIS:
   • Reward system não otimizado para Sharpe
   • Excesso de trades (overtrading)
   • Gestão de risco inadequada
   
   SOLUÇÕES:
   ✅ Implementar Sharpe-based reward: reward = returns / std(returns)
   ✅ Penalizar número excessivo de trades
   ✅ Aumentar filtro de confiança para 0.7-0.8
   ✅ Implementar position sizing dinâmico baseado em volatilidade

2️⃣ PROBLEMA: Explained Variance positivo (possível reward hacking)
   
   SOLUÇÕES:
   ✅ Reduzir amplificação do reward (de 4x para 2x)
   ✅ Adicionar regularização L2 no critic
   ✅ Implementar reward clipping mais agressivo
   ✅ Usar GAE lambda = 0.9 (ao invés de 0.95)

3️⃣ PROBLEMA: KL Divergence muito baixo
   
   SOLUÇÕES:
   ✅ Aumentar learning rate para 1e-4
   ✅ Reduzir clip_range para 0.1
   ✅ Aumentar n_epochs para 20
   ✅ Usar adaptive KL penalty

4️⃣ PROBLEMA: Entropy muito baixa (pouca exploração)
   
   SOLUÇÕES:
   ✅ Aumentar ent_coef para 0.1-0.15
   ✅ Implementar entropy scheduling (decay mais lento)
   ✅ Adicionar noise injection na política
   ✅ Usar curiosity-driven exploration

5️⃣ MELHORIAS NO AMBIENTE:
   
   ✅ Implementar custos de transação realistas (spread + comissão)
   ✅ Adicionar slippage dinâmico baseado em volume
   ✅ Simular impacto de mercado para orders grandes
   ✅ Implementar latência e delays de execução

6️⃣ MELHORIAS NO TREINAMENTO:
   
   ✅ Usar curriculum learning (começar com mercados mais fáceis)
   ✅ Implementar meta-learning para adaptação rápida
   ✅ Treinar ensemble de modelos (3-5 modelos)
   ✅ Usar adversarial training para robustez

7️⃣ VALIDAÇÃO MAIS RIGOROSA:
   
   ✅ Walk-forward optimization
   ✅ Monte Carlo simulation com diferentes seeds
   ✅ Stress testing em períodos de crise
   ✅ Out-of-sample testing em outros ativos

8️⃣ ARQUITETURA:
   
   ✅ Aumentar hidden_size para 512
   ✅ Adicionar attention mechanism
   ✅ Usar transformer architecture
   ✅ Implementar memory replay buffer maior
    """
    
    print(recommendations)
    
    print("\n" + "="*80)
    print("🎯 PRÓXIMOS PASSOS PRIORITÁRIOS:")
    print("="*80)
    print("""
    1. IMEDIATO: Aumentar filtro de confiança para 0.75
    2. IMEDIATO: Reduzir reward amplification para 2x
    3. CURTO PRAZO: Implementar Sharpe-based reward
    4. MÉDIO PRAZO: Adicionar custos de transação realistas
    5. LONGO PRAZO: Migrar para transformer architecture
    """)

def main():
    """Executar análise completa"""
    
    print("="*80)
    print("🔍 ANÁLISE COMPLETA DO TREINAMENTO SILUS")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Carregar dados
    print("\n📂 CARREGANDO DADOS...")
    training_df = load_training_metrics()
    convergence_df = load_convergence_analysis()
    reward_df = load_reward_analysis()
    trading_df = load_trading_performance()
    
    # Análises
    if training_df is not None:
        summary_df = analyze_training_metrics(training_df)
    
    analyze_convergence(convergence_df)
    analyze_rewards(reward_df)
    analyze_trading(trading_df)
    
    # Recomendações
    provide_recommendations()
    
    print("\n" + "="*80)
    print("✅ ANÁLISE CONCLUÍDA")
    print("="*80)

if __name__ == "__main__":
    main()