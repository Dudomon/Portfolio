"""
🔍 DIAGNÓSTICO V3BRUTAL EXPLAINED VARIANCE
Identifica qual componente está causando exp_var negativo
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from collections import defaultdict

def load_jsonl(filepath: Path) -> List[Dict]:
    """Carrega arquivo JSONL"""
    data = []
    if filepath.exists():
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        data.append(json.loads(line))
                    except:
                        pass
    return data

def analyze_reward_distribution(rewards_data: List[Dict]) -> Dict:
    """Analisa distribuição de rewards"""

    all_rewards = []
    pnl_components = []
    risk_components = []
    shaping_components = []

    for entry in rewards_data:
        if entry.get('type') == 'header':
            continue

        total = entry.get('total_reward', 0)
        pnl = entry.get('pnl_reward', 0)
        risk = entry.get('risk_reward', 0)
        shaping = entry.get('shaping_reward', 0)

        all_rewards.append(total)
        pnl_components.append(pnl)
        risk_components.append(risk)
        shaping_components.append(shaping)

    if not all_rewards:
        return {}

    return {
        'total_rewards': {
            'mean': np.mean(all_rewards),
            'std': np.std(all_rewards),
            'min': np.min(all_rewards),
            'max': np.max(all_rewards),
            'median': np.median(all_rewards),
            'q25': np.percentile(all_rewards, 25),
            'q75': np.percentile(all_rewards, 75),
            'positive_ratio': sum(1 for r in all_rewards if r > 0) / len(all_rewards),
            'count': len(all_rewards)
        },
        'pnl_component': {
            'mean': np.mean(pnl_components),
            'std': np.std(pnl_components),
            'contribution': np.mean(pnl_components) / (np.mean(all_rewards) + 1e-8)
        },
        'risk_component': {
            'mean': np.mean(risk_components),
            'std': np.std(risk_components),
            'contribution': np.mean(risk_components) / (np.mean(all_rewards) + 1e-8)
        },
        'shaping_component': {
            'mean': np.mean(shaping_components),
            'std': np.std(shaping_components),
            'contribution': np.mean(shaping_components) / (np.mean(all_rewards) + 1e-8)
        }
    }

def analyze_expvar_correlation(training_data: List[Dict], rewards_data: List[Dict]) -> Dict:
    """Analisa correlação entre exp_var e características do reward"""

    # Extrai exp_var e steps
    expvar_entries = [(e['step'], e.get('explained_variance', 0))
                      for e in training_data
                      if 'explained_variance' in e and e.get('type') != 'header']

    if not expvar_entries:
        return {'error': 'Nenhum dado de explained_variance encontrado'}

    # Organiza por step
    expvar_by_step = dict(expvar_entries)

    # Calcula estatísticas de reward em janelas
    window_stats = defaultdict(lambda: {
        'rewards': [],
        'pnl_rewards': [],
        'total_variance': 0,
        'pnl_variance': 0
    })

    for entry in rewards_data:
        if entry.get('type') == 'header':
            continue

        step = entry.get('step', 0)
        window = (step // 10000) * 10000  # Janela de 10k steps

        window_stats[window]['rewards'].append(entry.get('total_reward', 0))
        window_stats[window]['pnl_rewards'].append(entry.get('pnl_reward', 0))

    # Correlaciona exp_var com variância do reward
    correlations = []
    for step, expvar in expvar_entries:
        window = (step // 10000) * 10000
        stats = window_stats.get(window)

        if stats and stats['rewards']:
            reward_variance = np.var(stats['rewards'])
            reward_mean = np.mean(stats['rewards'])
            pnl_variance = np.var(stats['pnl_rewards'])

            correlations.append({
                'step': step,
                'expvar': expvar,
                'reward_variance': reward_variance,
                'reward_mean': reward_mean,
                'pnl_variance': pnl_variance,
                'reward_range': np.max(stats['rewards']) - np.min(stats['rewards'])
            })

    return {
        'expvar_stats': {
            'mean': np.mean([e['expvar'] for e in correlations]),
            'std': np.std([e['expvar'] for e in correlations]),
            'min': np.min([e['expvar'] for e in correlations]),
            'max': np.max([e['expvar'] for e in correlations]),
            'always_negative': all(e['expvar'] < 0 for e in correlations)
        },
        'correlations': correlations
    }

def test_tanh_normalization_impact():
    """Testa impacto da normalização tanh no exp_var"""

    print("\n" + "="*80)
    print("🧪 TESTE: Impacto da Normalização TANH")
    print("="*80)

    # Simula rewards com distribuição realista
    np.random.seed(42)

    # Rewards antes do tanh (distribuição mais ampla)
    raw_rewards = np.random.randn(1000) * 2.0  # média 0, std 2.0

    # Rewards após tanh (comprimidos para [-1, 1])
    max_reward = 10.0
    tanh_rewards = max_reward * np.tanh(raw_rewards / max_reward)

    print(f"\n📊 RAW REWARDS (antes do tanh):")
    print(f"  Mean: {np.mean(raw_rewards):.4f}")
    print(f"  Std:  {np.std(raw_rewards):.4f}")
    print(f"  Min:  {np.min(raw_rewards):.4f}")
    print(f"  Max:  {np.max(raw_rewards):.4f}")
    print(f"  Range: {np.max(raw_rewards) - np.min(raw_rewards):.4f}")

    print(f"\n📊 TANH REWARDS (após normalização):")
    print(f"  Mean: {np.mean(tanh_rewards):.4f}")
    print(f"  Std:  {np.std(tanh_rewards):.4f}")
    print(f"  Min:  {np.min(tanh_rewards):.4f}")
    print(f"  Max:  {np.max(tanh_rewards):.4f}")
    print(f"  Range: {np.max(tanh_rewards) - np.min(tanh_rewards):.4f}")

    # Compressão de variabilidade
    compression_ratio = np.std(tanh_rewards) / np.std(raw_rewards)
    print(f"\n⚠️  COMPRESSÃO DE VARIABILIDADE: {compression_ratio:.4f}")
    print(f"    (quanto menor, mais comprimida a distribuição)")

    # Simula returns verdadeiros
    true_returns = np.cumsum(raw_rewards)

    # Simula value predictions (baseadas em tanh_rewards)
    value_predictions = np.cumsum(tanh_rewards)

    # Calcula explained variance
    from sklearn.metrics import explained_variance_score
    exp_var = explained_variance_score(true_returns, value_predictions)

    print(f"\n📉 EXPLAINED VARIANCE SIMULADA: {exp_var:.4f}")

    if exp_var < 0:
        print(f"\n⚠️  HIPÓTESE CONFIRMADA: tanh está causando exp_var negativo!")
        print(f"    O value network está tentando prever returns baseado em rewards comprimidos")
        print(f"    Isso cria uma discrepância fundamental entre signal e prediction")

    return {
        'raw_std': np.std(raw_rewards),
        'tanh_std': np.std(tanh_rewards),
        'compression_ratio': compression_ratio,
        'exp_var': exp_var
    }

def test_pain_multiplier_impact():
    """Testa impacto do pain multiplier no exp_var"""

    print("\n" + "="*80)
    print("🧪 TESTE: Impacto do Pain Multiplier")
    print("="*80)

    # Simula PnL com losses
    np.random.seed(42)
    pnl_percents = np.random.randn(1000) * 0.05  # média 0, std 5%

    # Aplica pain multiplier (conforme código V3Brutal)
    pain_multiplier = 1.5
    modified_rewards = []

    for pnl in pnl_percents:
        pnl_clipped = np.clip(pnl, -0.15, 0.15)
        base_reward = pnl_clipped * 5.0

        # Aplica pain multiplier para losses
        if pnl_clipped < -0.03:
            pain_factor = 1.0 + (pain_multiplier - 1.0) * np.tanh(abs(pnl_clipped) * 20)
            final_reward = base_reward * pain_factor
        else:
            final_reward = base_reward

        modified_rewards.append(final_reward)

    # Compara distribuições
    base_rewards = [np.clip(p, -0.15, 0.15) * 5.0 for p in pnl_percents]

    print(f"\n📊 SEM PAIN MULTIPLIER:")
    print(f"  Mean: {np.mean(base_rewards):.4f}")
    print(f"  Std:  {np.std(base_rewards):.4f}")

    print(f"\n📊 COM PAIN MULTIPLIER:")
    print(f"  Mean: {np.mean(modified_rewards):.4f}")
    print(f"  Std:  {np.std(modified_rewards):.4f}")

    # Assimetria
    skewness = np.mean([(r - np.mean(modified_rewards))**3 for r in modified_rewards]) / (np.std(modified_rewards)**3)
    print(f"\n📊 ASSIMETRIA: {skewness:.4f}")
    print(f"    (negativo = enviesado para baixo)")

    if skewness < -0.5:
        print(f"\n⚠️  Pain multiplier cria FORTE ASSIMETRIA NEGATIVA")
        print(f"    Value network pode ter dificuldade em prever distribuição assimétrica")

    return {
        'base_std': np.std(base_rewards),
        'pain_std': np.std(modified_rewards),
        'skewness': skewness
    }

def main():
    print("\n" + "="*80)
    print("🔍 DIAGNÓSTICO V3BRUTAL - EXPLAINED VARIANCE NEGATIVO")
    print("="*80)

    # Encontra arquivos JSONL mais recentes
    avaliacoes_dir = Path("D:/Projeto/avaliacoes")

    # Lista todos os arquivos de treino
    training_files = sorted(avaliacoes_dir.glob("training_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    rewards_files = sorted(avaliacoes_dir.glob("rewards_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not training_files:
        print("\n❌ Nenhum arquivo training_*.jsonl encontrado")
        return

    training_file = training_files[0]
    rewards_file = rewards_files[0] if rewards_files else None

    print(f"\n📂 Analisando:")
    print(f"  Training: {training_file.name}")
    if rewards_file:
        print(f"  Rewards:  {rewards_file.name}")

    # Carrega dados
    training_data = load_jsonl(training_file)
    rewards_data = load_jsonl(rewards_file) if rewards_file else []

    print(f"\n📊 Dados carregados:")
    print(f"  Training entries: {len(training_data)}")
    print(f"  Rewards entries:  {len(rewards_data)}")

    # Análise 1: Distribuição de rewards
    if rewards_data:
        print("\n" + "="*80)
        print("📊 ANÁLISE 1: Distribuição de Rewards")
        print("="*80)

        dist = analyze_reward_distribution(rewards_data)

        if dist:
            print(f"\n🎯 TOTAL REWARDS:")
            for key, value in dist['total_rewards'].items():
                print(f"  {key:15s}: {value:.6f}")

            print(f"\n💰 PNL COMPONENT (85%):")
            for key, value in dist['pnl_component'].items():
                print(f"  {key:15s}: {value:.6f}")

            print(f"\n⚠️  RISK COMPONENT (10%):")
            for key, value in dist['risk_component'].items():
                print(f"  {key:15s}: {value:.6f}")

            print(f"\n🎨 SHAPING COMPONENT (5%):")
            for key, value in dist['shaping_component'].items():
                print(f"  {key:15s}: {value:.6f}")

    # Análise 2: Correlação exp_var
    if training_data and rewards_data:
        print("\n" + "="*80)
        print("📊 ANÁLISE 2: Correlação Explained Variance")
        print("="*80)

        corr = analyze_expvar_correlation(training_data, rewards_data)

        if 'expvar_stats' in corr:
            print(f"\n📉 EXPLAINED VARIANCE:")
            for key, value in corr['expvar_stats'].items():
                if isinstance(value, bool):
                    print(f"  {key:20s}: {value}")
                else:
                    print(f"  {key:20s}: {value:.6f}")

    # Análise 3: Testes de hipóteses
    print("\n" + "="*80)
    print("🧪 ANÁLISE 3: Testes de Hipóteses")
    print("="*80)

    tanh_results = test_tanh_normalization_impact()
    pain_results = test_pain_multiplier_impact()

    # Conclusão
    print("\n" + "="*80)
    print("🎯 CONCLUSÃO")
    print("="*80)

    if tanh_results['exp_var'] < 0:
        print("\n⚠️  CULPADO IDENTIFICADO: Normalização TANH")
        print(f"\n    A normalização tanh (linha 136 de reward_daytrade_v3_brutal.py) está")
        print(f"    comprimindo a variabilidade dos rewards por {tanh_results['compression_ratio']:.2%}")
        print(f"    Isso cria uma discrepância entre o signal verdadeiro e as predictions do value network")
        print(f"\n    Exp_var simulado com tanh: {tanh_results['exp_var']:.4f}")

    if pain_results['skewness'] < -0.5:
        print(f"\n⚠️  AGRAVANTE: Pain Multiplier")
        print(f"    Cria assimetria negativa de {pain_results['skewness']:.4f}")
        print(f"    Dificulta ainda mais a predição do value network")

    print(f"\n💡 RECOMENDAÇÃO:")
    print(f"    1. Remover ou reduzir normalização tanh")
    print(f"    2. Considerar normalização menos agressiva (e.g., clip ao invés de tanh)")
    print(f"    3. Ou aceitar exp_var negativo se performance real for boa")
    print(f"       (exp_var não é objetivo final, lucro é!)")

if __name__ == "__main__":
    main()
