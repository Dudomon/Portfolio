#!/usr/bin/env python3
"""
Análise completa da distribuição de rewards para otimização do range de clipping.
Analisa o arquivo de rewards mais recente para determinar distribuição e recomendar range ótimo.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def load_rewards_data(filepath):
    """Carrega dados de rewards do arquivo JSONL."""
    rewards = []
    reward_components = defaultdict(list)
    metadata = []
    
    print(f"Carregando dados de: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 10000 == 0:
                print(f"Processando linha {line_num:,}...")
            
            try:
                data = json.loads(line.strip())
                
                if data.get('type') == 'reward_info':
                    total_reward = data.get('total_reward')
                    if total_reward is not None:
                        rewards.append(total_reward)
                        
                        # Componentes do reward
                        components = data.get('reward_components', {})
                        for key, value in components.items():
                            if value is not None:
                                reward_components[key].append(value)
                        
                        # Metadata adicional
                        metadata.append({
                            'step': data.get('step'),
                            'portfolio_value': data.get('portfolio_value'),
                            'win_rate': data.get('win_rate'),
                            'trades_count': data.get('trades_count'),
                            'current_drawdown': data.get('current_drawdown'),
                            'total_pnl': data.get('total_pnl'),
                            'gaming_penalty': data.get('gaming_detection', {}).get('gaming_penalty', 0),
                            'overtrading_penalty': data.get('gaming_detection', {}).get('overtrading_penalty', 0)
                        })
                        
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Erro na linha {line_num}: {e}")
                continue
    
    print(f"Carregados {len(rewards):,} registros de rewards")
    return np.array(rewards), reward_components, metadata

def analyze_distribution(rewards):
    """Análise estatística completa da distribuição de rewards."""
    print("\n" + "="*80)
    print("ANÁLISE COMPLETA DA DISTRIBUIÇÃO DE REWARDS")
    print("="*80)
    
    # Estatísticas básicas
    print(f"\n📊 ESTATÍSTICAS BÁSICAS:")
    print(f"Total de amostras: {len(rewards):,}")
    print(f"Média: {np.mean(rewards):.6f}")
    print(f"Mediana: {np.median(rewards):.6f}")
    print(f"Desvio padrão: {np.std(rewards):.6f}")
    print(f"Variância: {np.var(rewards):.6f}")
    print(f"Skewness: {stats.skew(rewards):.6f}")
    print(f"Kurtosis: {stats.kurtosis(rewards):.6f}")
    
    # Valores extremos
    print(f"\n🔥 VALORES EXTREMOS:")
    print(f"Mínimo ABSOLUTO: {np.min(rewards):.6f}")
    print(f"Máximo ABSOLUTO: {np.max(rewards):.6f}")
    print(f"Range total: {np.max(rewards) - np.min(rewards):.6f}")
    
    # Percentis críticos
    percentiles = [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 99.5, 99.9]
    print(f"\n📈 PERCENTIS DETALHADOS:")
    for p in percentiles:
        value = np.percentile(rewards, p)
        print(f"P{p:4.1f}%: {value:8.6f}")
    
    return {
        'mean': np.mean(rewards),
        'median': np.median(rewards),
        'std': np.std(rewards),
        'min': np.min(rewards),
        'max': np.max(rewards),
        'percentiles': {p: np.percentile(rewards, p) for p in percentiles}
    }

def analyze_clipping_impact(rewards, current_range=(-1, 1)):
    """Analisa o impacto do clipping atual e sugere ranges otimizados."""
    print(f"\n" + "="*80)
    print("ANÁLISE DE IMPACTO DO CLIPPING")
    print("="*80)
    
    min_val, max_val = current_range
    
    # Valores perdidos pelo clipping atual
    clipped_low = rewards < min_val
    clipped_high = rewards > max_val
    
    total_clipped = np.sum(clipped_low) + np.sum(clipped_high)
    pct_clipped = (total_clipped / len(rewards)) * 100
    
    print(f"\n🔍 IMPACTO DO CLIPPING ATUAL [{min_val}, {max_val}]:")
    print(f"Valores < {min_val}: {np.sum(clipped_low):,} ({np.sum(clipped_low)/len(rewards)*100:.2f}%)")
    print(f"Valores > {max_val}: {np.sum(clipped_high):,} ({np.sum(clipped_high)/len(rewards)*100:.2f}%)")
    print(f"Total clippado: {total_clipped:,} ({pct_clipped:.2f}%)")
    print(f"Informação preservada: {100-pct_clipped:.2f}%")
    
    # Distribuição dos valores clippados
    if np.sum(clipped_low) > 0:
        clipped_low_values = rewards[clipped_low]
        print(f"\nValores BAIXOS clippados:")
        print(f"  Menor valor: {np.min(clipped_low_values):.6f}")
        print(f"  Maior valor clippado: {np.max(clipped_low_values):.6f}")
        print(f"  Média dos clippados: {np.mean(clipped_low_values):.6f}")
    
    if np.sum(clipped_high) > 0:
        clipped_high_values = rewards[clipped_high]
        print(f"\nValores ALTOS clippados:")
        print(f"  Menor valor clippado: {np.min(clipped_high_values):.6f}")
        print(f"  Maior valor: {np.max(clipped_high_values):.6f}")
        print(f"  Média dos clippados: {np.mean(clipped_high_values):.6f}")
    
    return {
        'current_range': current_range,
        'total_clipped': total_clipped,
        'pct_clipped': pct_clipped,
        'clipped_low_count': np.sum(clipped_low),
        'clipped_high_count': np.sum(clipped_high)
    }

def recommend_optimal_ranges(rewards):
    """Recomenda ranges ótimos baseados em diferentes critérios."""
    print(f"\n" + "="*80)
    print("RECOMENDAÇÕES DE RANGES ÓTIMOS")
    print("="*80)
    
    ranges_to_test = [
        # Baseado em percentis
        (np.percentile(rewards, 1), np.percentile(rewards, 99)),    # 98% preservado
        (np.percentile(rewards, 2), np.percentile(rewards, 98)),    # 96% preservado
        (np.percentile(rewards, 5), np.percentile(rewards, 95)),    # 90% preservado
        
        # Baseado em desvios padrão
        (np.mean(rewards) - 2*np.std(rewards), np.mean(rewards) + 2*np.std(rewards)),  # 2 sigma
        (np.mean(rewards) - 3*np.std(rewards), np.mean(rewards) + 3*np.std(rewards)),  # 3 sigma
        
        # Ranges simétricos
        (-2, 2),
        (-3, 3),
        (-5, 5),
        (-10, 10),
        
        # Range atual para comparação
        (-1, 1)
    ]
    
    recommendations = []
    
    print(f"\n📋 COMPARAÇÃO DE RANGES:")
    print(f"{'Range':<20} {'Clippados':<12} {'% Perdido':<12} {'% Preservado':<15} {'Recomendação'}")
    print("-" * 80)
    
    for min_val, max_val in ranges_to_test:
        clipped_low = np.sum(rewards < min_val)
        clipped_high = np.sum(rewards > max_val)
        total_clipped = clipped_low + clipped_high
        pct_lost = (total_clipped / len(rewards)) * 100
        pct_preserved = 100 - pct_lost
        
        # Classificação da recomendação
        if pct_preserved >= 98:
            rec = "🟢 EXCELENTE"
        elif pct_preserved >= 95:
            rec = "🟡 BOA"
        elif pct_preserved >= 90:
            rec = "🟠 ACEITÁVEL"
        else:
            rec = "🔴 RUIM"
        
        print(f"[{min_val:6.2f}, {max_val:6.2f}] {total_clipped:8,} {pct_lost:8.2f}% {pct_preserved:11.2f}% {rec}")
        
        recommendations.append({
            'range': (min_val, max_val),
            'clipped_count': total_clipped,
            'pct_lost': pct_lost,
            'pct_preserved': pct_preserved,
            'recommendation': rec
        })
    
    # Encontrar o range ótimo
    print(f"\n🎯 RECOMENDAÇÃO FINAL:")
    
    # Range que preserva 95-98% da informação com menor range possível
    optimal_candidates = [r for r in recommendations if 95 <= r['pct_preserved'] <= 98]
    if optimal_candidates:
        # Escolher o com menor range (mais restritivo mas ainda bom)
        optimal = min(optimal_candidates, key=lambda x: x['range'][1] - x['range'][0])
        print(f"Range ÓTIMO: [{optimal['range'][0]:.3f}, {optimal['range'][1]:.3f}]")
        print(f"Preserva: {optimal['pct_preserved']:.2f}% da informação")
        print(f"Perde apenas: {optimal['pct_lost']:.2f}% dos valores")
    else:
        # Fallback para o que preserva mais informação
        optimal = max(recommendations, key=lambda x: x['pct_preserved'])
        print(f"Range RECOMENDADO: [{optimal['range'][0]:.3f}, {optimal['range'][1]:.3f}]")
        print(f"Preserva: {optimal['pct_preserved']:.2f}% da informação")
    
    return recommendations

def analyze_reward_components(reward_components):
    """Analisa a distribuição dos componentes individuais do reward."""
    print(f"\n" + "="*80)
    print("ANÁLISE DOS COMPONENTES DE REWARD")
    print("="*80)
    
    for component, values in reward_components.items():
        if len(values) > 0:
            values_array = np.array(values)
            non_zero = values_array[values_array != 0]
            
            print(f"\n🔧 {component.upper()}:")
            print(f"  Total valores: {len(values_array):,}")
            print(f"  Valores não-zero: {len(non_zero):,} ({len(non_zero)/len(values_array)*100:.1f}%)")
            
            if len(non_zero) > 0:
                print(f"  Min: {np.min(non_zero):.6f}")
                print(f"  Max: {np.max(non_zero):.6f}")
                print(f"  Média: {np.mean(non_zero):.6f}")
                print(f"  P5: {np.percentile(non_zero, 5):.6f}")
                print(f"  P95: {np.percentile(non_zero, 95):.6f}")

def main():
    """Função principal."""
    # Encontrar arquivo de rewards mais recente
    avaliacoes_dir = Path("avaliacoes")
    reward_files = list(avaliacoes_dir.glob("rewards_*.jsonl"))
    
    if not reward_files:
        print("❌ Nenhum arquivo de rewards encontrado!")
        return
    
    # Usar o mais recente
    latest_file = max(reward_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 Arquivo mais recente: {latest_file}")
    
    # Carregar e analisar dados
    rewards, reward_components, metadata = load_rewards_data(latest_file)
    
    if len(rewards) == 0:
        print("❌ Nenhum dado de reward encontrado!")
        return
    
    # Análises
    stats = analyze_distribution(rewards)
    clipping_impact = analyze_clipping_impact(rewards)
    recommendations = recommend_optimal_ranges(rewards)
    analyze_reward_components(reward_components)
    
    # Salvar relatório resumido
    report_file = f"reward_distribution_analysis_{latest_file.stem.split('_')[-1]}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("RELATÓRIO DE ANÁLISE DE DISTRIBUIÇÃO DE REWARDS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Arquivo analisado: {latest_file}\n")
        f.write(f"Total de amostras: {len(rewards):,}\n\n")
        
        f.write("ESTATÍSTICAS BÁSICAS:\n")
        f.write(f"Média: {stats['mean']:.6f}\n")
        f.write(f"Desvio padrão: {stats['std']:.6f}\n")
        f.write(f"Mínimo: {stats['min']:.6f}\n")
        f.write(f"Máximo: {stats['max']:.6f}\n\n")
        
        f.write("PERCENTIS CRÍTICOS:\n")
        for p in [1, 5, 95, 99]:
            f.write(f"P{p}%: {stats['percentiles'][p]:.6f}\n")
        
        f.write(f"\nCLIPPING ATUAL [-1, 1]:\n")
        f.write(f"Informação perdida: {clipping_impact['pct_clipped']:.2f}%\n")
        f.write(f"Valores clippados: {clipping_impact['total_clipped']:,}\n\n")
        
        # Melhor recomendação
        best_rec = max(recommendations, key=lambda x: x['pct_preserved'] if x['pct_preserved'] <= 98 else 0)
        f.write("RECOMENDAÇÃO ÓTIMA:\n")
        f.write(f"Range: [{best_rec['range'][0]:.3f}, {best_rec['range'][1]:.3f}]\n")
        f.write(f"Informação preservada: {best_rec['pct_preserved']:.2f}%\n")
    
    print(f"\n✅ Relatório salvo em: {report_file}")
    print(f"\n🎯 RESUMO EXECUTIVO:")
    print(f"   • Total de rewards analisados: {len(rewards):,}")
    print(f"   • Range atual [-1, 1] perde: {clipping_impact['pct_clipped']:.2f}% da informação")
    print(f"   • Valores extremos: [{stats['min']:.3f}, {stats['max']:.3f}]")
    
    # Recomendação final destacada
    optimal_95_98 = [r for r in recommendations if 95 <= r['pct_preserved'] <= 98]
    if optimal_95_98:
        best = min(optimal_95_98, key=lambda x: x['range'][1] - x['range'][0])
        print(f"   • RANGE ÓTIMO RECOMENDADO: [{best['range'][0]:.3f}, {best['range'][1]:.3f}]")
        print(f"     (Preserva {best['pct_preserved']:.1f}% da informação)")

if __name__ == "__main__":
    main()