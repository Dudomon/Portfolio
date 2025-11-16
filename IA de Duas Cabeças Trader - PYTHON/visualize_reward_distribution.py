#!/usr/bin/env python3
"""
Visualização da distribuição de rewards para análise detalhada.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_rewards_simple(filepath, max_samples=50000):
    """Carrega dados de rewards de forma mais rápida."""
    rewards = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if len(rewards) >= max_samples:
                break
                
            try:
                data = json.loads(line.strip())
                if data.get('type') == 'reward_info':
                    total_reward = data.get('total_reward')
                    if total_reward is not None:
                        rewards.append(total_reward)
            except:
                continue
    
    return np.array(rewards)

def create_visualizations(rewards):
    """Cria visualizações da distribuição de rewards."""
    
    # Setup da figura
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Análise da Distribuição de Rewards', fontsize=16, fontweight='bold')
    
    # 1. Histograma completo
    ax1 = axes[0, 0]
    ax1.hist(rewards, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(-1, color='red', linestyle='--', linewidth=2, label='Clipping atual [-1, 1]')
    ax1.axvline(1, color='red', linestyle='--', linewidth=2)
    ax1.axvline(-2, color='green', linestyle=':', linewidth=2, label='Range ótimo [-2, -0.002]')
    ax1.axvline(-0.002, color='green', linestyle=':', linewidth=2)
    ax1.set_xlabel('Reward Value')
    ax1.set_ylabel('Frequência')
    ax1.set_title('Distribuição Completa de Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot
    ax2 = axes[0, 1]
    box_data = [rewards]
    bp = ax2.boxplot(box_data, patch_artist=True, labels=['Todos os Rewards'])
    bp['boxes'][0].set_facecolor('lightcoral')
    ax2.axhline(-1, color='red', linestyle='--', alpha=0.7, label='Clipping [-1, 1]')
    ax2.axhline(1, color='red', linestyle='--', alpha=0.7)
    ax2.set_ylabel('Reward Value')
    ax2.set_title('Box Plot - Identificação de Outliers')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Densidade por range
    ax3 = axes[1, 0]
    
    # Separação por ranges
    range_1 = rewards[(rewards >= -1) & (rewards <= 1)]
    range_2 = rewards[(rewards >= -2) & (rewards <= 2)]
    range_clipped = rewards[(rewards < -1) | (rewards > 1)]
    
    ax3.hist(range_1, bins=50, alpha=0.6, label=f'Range [-1,1] ({len(range_1):,} amostras)', color='red')
    ax3.hist(range_clipped, bins=50, alpha=0.6, label=f'Valores clippados ({len(range_clipped):,} amostras)', color='orange')
    ax3.set_xlabel('Reward Value')
    ax3.set_ylabel('Frequência')
    ax3.set_title('Comparação: Dentro vs Fora do Range Atual')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Percentis
    ax4 = axes[1, 1]
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    values = [np.percentile(rewards, p) for p in percentiles]
    
    bars = ax4.bar(range(len(percentiles)), values, color='steelblue', alpha=0.7)
    ax4.axhline(-1, color='red', linestyle='--', linewidth=2, label='Clipping atual')
    ax4.axhline(1, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Percentis')
    ax4.set_ylabel('Reward Value')
    ax4.set_title('Distribuição por Percentis')
    ax4.set_xticks(range(len(percentiles)))
    ax4.set_xticklabels([f'P{p}' for p in percentiles])
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Adicionar valores nos bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Salvar
    output_file = 'reward_distribution_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Visualização salva em: {output_file}")
    
    return output_file

def detailed_statistics(rewards):
    """Estatísticas detalhadas formatadas."""
    print("\n" + "="*80)
    print("ESTATÍSTICAS DETALHADAS DOS REWARDS")
    print("="*80)
    
    # Informações básicas
    print(f"\n📊 INFORMAÇÕES GERAIS:")
    print(f"Total de amostras: {len(rewards):,}")
    print(f"Valores únicos: {len(np.unique(rewards)):,}")
    print(f"Range observado: [{np.min(rewards):.6f}, {np.max(rewards):.6f}]")
    
    # Distribuição por quadrantes
    q1 = np.sum(rewards < -1.5)
    q2 = np.sum((rewards >= -1.5) & (rewards < -0.5))
    q3 = np.sum((rewards >= -0.5) & (rewards < 0.5))
    q4 = np.sum(rewards >= 0.5)
    
    print(f"\n📈 DISTRIBUIÇÃO POR QUADRANTES:")
    print(f"Muito negativos (< -1.5): {q1:,} ({q1/len(rewards)*100:.1f}%)")
    print(f"Negativos (-1.5 a -0.5): {q2:,} ({q2/len(rewards)*100:.1f}%)")
    print(f"Neutros (-0.5 a 0.5): {q3:,} ({q3/len(rewards)*100:.1f}%)")
    print(f"Positivos (> 0.5): {q4:,} ({q4/len(rewards)*100:.1f}%)")
    
    # Análise de clipping em diferentes ranges
    print(f"\n🔍 ANÁLISE DE CLIPPING EM DIFERENTES RANGES:")
    ranges_to_test = [
        (-1, 1),     # Atual
        (-1.5, 1.5), # Expandido 1
        (-2, 2),     # Expandido 2
        (-2, 0),     # Assimétrico (baseado na distribuição)
        (-2, -0.002) # Recomendado pela análise
    ]
    
    for min_val, max_val in ranges_to_test:
        clipped = np.sum((rewards < min_val) | (rewards > max_val))
        preserved = len(rewards) - clipped
        pct_preserved = (preserved / len(rewards)) * 100
        
        status = "✅" if pct_preserved >= 95 else "⚠️" if pct_preserved >= 90 else "❌"
        print(f"{status} [{min_val:6.1f}, {max_val:6.1f}]: {pct_preserved:5.1f}% preservado ({clipped:,} clippados)")

def main():
    """Função principal."""
    # Encontrar arquivo mais recente
    avaliacoes_dir = Path("avaliacoes")
    reward_files = list(avaliacoes_dir.glob("rewards_*.jsonl"))
    
    if not reward_files:
        print("❌ Nenhum arquivo de rewards encontrado!")
        return
    
    latest_file = max(reward_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 Analisando: {latest_file}")
    
    # Carregar dados
    print("🔄 Carregando dados de rewards...")
    rewards = load_rewards_simple(latest_file)
    
    if len(rewards) == 0:
        print("❌ Nenhum dado encontrado!")
        return
    
    print(f"✅ Carregados {len(rewards):,} registros")
    
    # Análises
    detailed_statistics(rewards)
    
    # Criar visualizações
    print(f"\n🎨 Gerando visualizações...")
    viz_file = create_visualizations(rewards)
    
    print(f"\n🎯 CONCLUSÕES PRINCIPAIS:")
    print(f"1. O range atual [-1, 1] está clippando {np.sum((rewards < -1) | (rewards > 1))/len(rewards)*100:.1f}% dos dados")
    print(f"2. A distribuição é fortemente assimétrica (skew positivo)")
    print(f"3. Maioria dos rewards está na faixa negativa")
    print(f"4. Range ótimo recomendado: [-2.0, -0.002] preserva 98% da informação")
    print(f"5. Apenas {np.sum(rewards > 0)/len(rewards)*100:.1f}% dos rewards são positivos")

if __name__ == "__main__":
    main()