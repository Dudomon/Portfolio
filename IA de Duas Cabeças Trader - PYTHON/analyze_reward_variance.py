#!/usr/bin/env python3
"""
Script para análise de variância de rewards nos dados de avaliação
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import statistics

def analyze_reward_variance():
    """Analisa variância de rewards nos arquivos JSONL"""
    
    # Arquivos para análise
    files_to_analyze = [
        "D:/Projeto/avaliacoes/rewards_20250804_094339.jsonl",
        "D:/Projeto/avaliacoes/training_20250804_094339.jsonl"
    ]
    
    reward_data = []
    training_data = []
    
    print("🔍 ANÁLISE DE VARIÂNCIA DE REWARDS")
    print("=" * 60)
    
    # Análise de dados de rewards
    try:
        with open(files_to_analyze[0], 'r') as f:
            for line_num, line in enumerate(f):
                if line_num == 0:  # Skip header
                    continue
                if line_num > 10000:  # Limit for performance
                    break
                    
                try:
                    data = json.loads(line.strip())
                    if data.get('type') == 'reward_info':
                        reward_data.append({
                            'step': data.get('step', 0),
                            'total_reward': data.get('total_reward', 0),
                            'portfolio_value': data.get('portfolio_value', 0),
                            'current_drawdown': data.get('current_drawdown', 0),
                            'win_rate': data.get('win_rate', 0),
                            'total_pnl': data.get('total_pnl', 0),
                            'trades_count': data.get('trades_count', 0),
                            'gaming_penalty': data.get('reward_components', {}).get('gaming_penalty', 0)
                        })
                except json.JSONDecodeError:
                    continue
                    
    except FileNotFoundError:
        print("❌ Arquivo de rewards não encontrado")
        return
        
    # Análise de dados de training
    try:
        with open(files_to_analyze[1], 'r') as f:
            for line_num, line in enumerate(f):
                if line_num == 0:  # Skip header
                    continue
                if line_num > 10000:  # Limit for performance
                    break
                    
                try:
                    data = json.loads(line.strip())
                    if data.get('type') == 'training_step':
                        training_data.append({
                            'step': data.get('step', 0),
                            'loss': data.get('loss', 0),
                            'value_loss': data.get('value_loss', 0),
                            'entropy_loss': data.get('entropy_loss', 0),
                            'clip_fraction': data.get('clip_fraction', 0),
                            'explained_variance': data.get('explained_variance', 0)
                        })
                except json.JSONDecodeError:
                    continue
                    
    except FileNotFoundError:
        print("❌ Arquivo de training não encontrado")
        
    if not reward_data:
        print("❌ Nenhum dado de reward encontrado")
        return
        
    # Converter para DataFrames
    df_rewards = pd.DataFrame(reward_data)
    df_training = pd.DataFrame(training_data) if training_data else pd.DataFrame()
    
    print(f"📊 Dados coletados:")
    print(f"   - Rewards: {len(df_rewards)} registros")
    print(f"   - Training: {len(df_training)} registros")
    print()
    
    # ANÁLISE 1: Variância dos rewards ao longo do tempo
    print("1️⃣ VARIÂNCIA DOS REWARDS AO LONGO DO TEMPO")
    print("-" * 50)
    
    rewards = df_rewards['total_reward'].values
    rewards_non_zero = rewards[rewards != 0]
    
    print(f"Total de registros: {len(rewards)}")
    print(f"Registros não-zero: {len(rewards_non_zero)}")
    print(f"Registros zero: {len(rewards) - len(rewards_non_zero)} ({((len(rewards) - len(rewards_non_zero))/len(rewards)*100):.1f}%)")
    print()
    
    if len(rewards_non_zero) > 0:
        print(f"Estatísticas dos rewards não-zero:")
        print(f"   Média: {np.mean(rewards_non_zero):.4f}")
        print(f"   Desvio Padrão: {np.std(rewards_non_zero):.4f}")
        print(f"   Variância: {np.var(rewards_non_zero):.4f}")
        print(f"   Mínimo: {np.min(rewards_non_zero):.4f}")
        print(f"   Máximo: {np.max(rewards_non_zero):.4f}")
        print(f"   Mediana: {np.median(rewards_non_zero):.4f}")
        
        # Análise de instabilidade
        print(f"\n📈 ANÁLISE DE INSTABILIDADE:")
        reward_changes = np.diff(rewards_non_zero)
        if len(reward_changes) > 0:
            print(f"   Variação média entre steps: {np.mean(np.abs(reward_changes)):.4f}")
            print(f"   Variação máxima: {np.max(np.abs(reward_changes)):.4f}")
            print(f"   Coeficiente de variação: {(np.std(rewards_non_zero)/np.mean(rewards_non_zero)):.4f}")
    else:
        print("❌ Todos os rewards são zero - possível problema no sistema")
    print()
    
    # ANÁLISE 2: Correlação com explained_variance
    print("2️⃣ CORRELAÇÃO REWARD VARIANCE vs EXPLAINED VARIANCE")
    print("-" * 50)
    
    if not df_training.empty and 'explained_variance' in df_training.columns:
        # Alinhar dados por step
        merged = pd.merge(df_rewards, df_training, on='step', how='inner')
        
        if len(merged) > 0:
            reward_var = merged['total_reward'].values
            explained_var = merged['explained_variance'].values
            
            # Remove valores zero/NaN
            valid_mask = (reward_var != 0) & (~np.isnan(explained_var)) & (explained_var != 0)
            
            if np.sum(valid_mask) > 10:
                reward_var_clean = reward_var[valid_mask]
                explained_var_clean = explained_var[valid_mask]
                
                correlation = np.corrcoef(reward_var_clean, explained_var_clean)[0, 1]
                print(f"Correlação Reward Variance vs Explained Variance: {correlation:.4f}")
                
                print(f"Explained Variance Stats:")
                print(f"   Média: {np.mean(explained_var_clean):.4f}")
                print(f"   Desvio Padrão: {np.std(explained_var_clean):.4f}")
                print(f"   Mínimo: {np.min(explained_var_clean):.4f}")
                print(f"   Máximo: {np.max(explained_var_clean):.4f}")
            else:
                print("❌ Dados insuficientes para correlação")
        else:
            print("❌ Não foi possível alinhar dados de reward e training")
    else:
        print("❌ Dados de training não disponíveis ou incompletos")
    print()
    
    # ANÁLISE 3: Padrões de instabilidade
    print("3️⃣ PADRÕES DE INSTABILIDADE NOS REWARDS")
    print("-" * 50)
    
    # Análise de gaming penalties
    gaming_penalties = df_rewards['gaming_penalty'].values
    gaming_penalties_non_zero = gaming_penalties[gaming_penalties != 0]
    
    print(f"Gaming Penalties:")
    print(f"   Total de penalidades: {len(gaming_penalties_non_zero)}")
    print(f"   Percentual de steps com penalidade: {(len(gaming_penalties_non_zero)/len(gaming_penalties)*100):.1f}%")
    
    if len(gaming_penalties_non_zero) > 0:
        print(f"   Penalidade média: {np.mean(gaming_penalties_non_zero):.4f}")
        print(f"   Penalidade máxima: {np.min(gaming_penalties_non_zero):.4f}")  # min porque são valores negativos
    
    # Análise de drawdown
    drawdowns = df_rewards['current_drawdown'].values
    drawdowns_non_zero = drawdowns[drawdowns > 0]
    
    print(f"\nDrawdown Analysis:")
    if len(drawdowns_non_zero) > 0:
        print(f"   Drawdown médio: {np.mean(drawdowns_non_zero):.2f}%")
        print(f"   Drawdown máximo: {np.max(drawdowns_non_zero):.2f}%")
        print(f"   Steps em drawdown: {len(drawdowns_non_zero)} ({(len(drawdowns_non_zero)/len(drawdowns)*100):.1f}%)")
    print()
    
    # ANÁLISE 4: Clipping frequency
    print("4️⃣ FREQUÊNCIA DE CLIPPING")
    print("-" * 50)
    
    if not df_training.empty and 'clip_fraction' in df_training.columns:
        clip_fractions = df_training['clip_fraction'].values
        clip_fractions_non_zero = clip_fractions[clip_fractions > 0]
        
        print(f"Clip Fraction Stats:")
        print(f"   Registros com clipping: {len(clip_fractions_non_zero)} de {len(clip_fractions)}")
        print(f"   Percentual com clipping: {(len(clip_fractions_non_zero)/len(clip_fractions)*100):.1f}%")
        
        if len(clip_fractions_non_zero) > 0:
            print(f"   Clip fraction média: {np.mean(clip_fractions_non_zero):.4f}")
            print(f"   Clip fraction máxima: {np.max(clip_fractions_non_zero):.4f}")
            
            # Análise de clipping excessivo
            high_clip = clip_fractions_non_zero[clip_fractions_non_zero > 0.3]
            if len(high_clip) > 0:
                print(f"   ⚠️ Steps com clipping alto (>30%): {len(high_clip)} ({(len(high_clip)/len(clip_fractions)*100):.1f}%)")
            
            very_high_clip = clip_fractions_non_zero[clip_fractions_non_zero > 0.5]
            if len(very_high_clip) > 0:
                print(f"   🚨 Steps com clipping muito alto (>50%): {len(very_high_clip)} ({(len(very_high_clip)/len(clip_fractions)*100):.1f}%)")
    else:
        print("❌ Dados de clip_fraction não disponíveis")
    print()
    
    # SUMÁRIO FINAL
    print("📋 SUMÁRIO DA ANÁLISE")
    print("=" * 60)
    
    issues_found = []
    
    if len(rewards_non_zero) == 0:
        issues_found.append("🚨 CRÍTICO - Todos os rewards são zero")
    elif len(rewards_non_zero) / len(rewards) < 0.1:
        issues_found.append("⚠️ ALTA - Mais de 90% dos rewards são zero")
    
    if len(rewards_non_zero) > 0:
        cv = np.std(rewards_non_zero) / np.abs(np.mean(rewards_non_zero))
        if cv > 2.0:
            issues_found.append(f"⚠️ ALTA - Coeficiente de variação muito alto ({cv:.2f})")
    
    if len(gaming_penalties_non_zero) / len(gaming_penalties) > 0.3:
        issues_found.append("⚠️ MÉDIA - Muitas penalidades por gaming (>30%)")
    
    if not df_training.empty and len(clip_fractions_non_zero) > 0:
        high_clip_pct = len(clip_fractions_non_zero[clip_fractions_non_zero > 0.5]) / len(clip_fractions) * 100
        if high_clip_pct > 10:
            issues_found.append(f"⚠️ MÉDIA - Clipping excessivo em {high_clip_pct:.1f}% dos steps")
    
    if issues_found:
        print("PROBLEMAS IDENTIFICADOS:")
        for issue in issues_found:
            print(f"   {issue}")
    else:
        print("✅ Nenhum problema crítico identificado nos dados analisados")
    
    print(f"\n📊 Análise concluída - {len(df_rewards)} registros de rewards processados")

if __name__ == "__main__":
    analyze_reward_variance()