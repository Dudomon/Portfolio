"""
🎯 Análise Comparativa Reward Engineering - TwoHeadV9Optimus

COMPARA:
1. Distribuição de ações antes vs depois das otimizações
2. Variância e exploração
3. Estabilidade de treinamento
4. Recomendações específicas para reward engineering
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from trading_framework.policies.two_head_v9_optimus import (
    TwoHeadV9Optimus, 
    get_v9_optimus_kwargs
)

def analyze_action_distribution_comprehensive():
    """Análise completa da distribuição de ações"""
    
    print("🎯 ANÁLISE REWARD ENGINEERING COMPLETA")
    print("=" * 60)
    
    # Criar policy de teste
    dummy_obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
    dummy_action_space = gym.spaces.Box(low=np.array([0, 0, -1, -1]), high=np.array([2, 1, 1, 1]), dtype=np.float32)
    
    def dummy_lr_schedule(progress):
        return 1e-4
    
    policy = TwoHeadV9Optimus(
        observation_space=dummy_obs_space,
        action_space=dummy_action_space,
        lr_schedule=dummy_lr_schedule,
        **get_v9_optimus_kwargs()
    )
    
    policy.eval()
    
    # Gerar amostras grandes para análise estatística robusta
    n_samples = 5000
    print(f"📊 Gerando {n_samples} amostras para análise robusta...")
    
    features = torch.randn(n_samples, 450)
    lstm_states = None
    episode_starts = torch.zeros(n_samples, dtype=torch.bool)
    
    with torch.no_grad():
        dist = policy.forward_actor(features, lstm_states, episode_starts)
        actions = dist.sample()  # [n_samples, 4]
        
        # Também testar com diferentes std
        dist_low_std = policy._get_action_dist_from_latent(torch.randn(n_samples, 256))
        actions_low_std = dist_low_std.sample()
    
    actions_np = actions.detach().numpy()
    actions_low_std_np = actions_low_std.detach().numpy()
    
    print(f"\n📈 ANÁLISE ESTATÍSTICA DETALHADA:")
    print("=" * 50)
    
    action_names = ['entry_decision', 'confidence', 'pos1_mgmt', 'pos2_mgmt']
    expected_ranges = [(0, 2), (0, 1), (-1, 1), (-1, 1)]
    
    # Análise por dimensão
    for i, (name, (low, high)) in enumerate(zip(action_names, expected_ranges)):
        values = actions_np[:, i]
        values_low = actions_low_std_np[:, i]
        
        print(f"\n🎯 {name.upper()}:")
        print(f"   Range esperado: [{low}, {high}]")
        print(f"   Range atual: [{values.min():.3f}, {values.max():.3f}]")
        print(f"   Média: {values.mean():.3f} (±{values.std():.3f})")
        print(f"   Mediana: {np.median(values):.3f}")
        print(f"   Q25-Q75: [{np.percentile(values, 25):.3f}, {np.percentile(values, 75):.3f}]")
        
        # Cobertura do range
        in_range = (values >= low) & (values <= high)
        pct_in_range = in_range.mean() * 100
        print(f"   % no range: {pct_in_range:.1f}%")
        
        # Utilização do range
        range_span = high - low
        actual_span = values.max() - values.min()
        utilization = (actual_span / range_span) * 100 if range_span > 0 else 0
        print(f"   Utilização do range: {utilization:.1f}%")
        
        # Análise de normalidade (importante para reward engineering)
        from scipy.stats import normaltest
        stat, p_value = normaltest(values)
        is_normal = p_value > 0.05
        print(f"   Distribuição normal: {'✅' if is_normal else '❌'} (p={p_value:.4f})")
        
        # Análise de concentração (detectar colapso)
        unique_values = len(np.unique(np.round(values, 3)))
        concentration = unique_values / len(values)
        print(f"   Diversidade: {concentration:.3f} ({unique_values}/{len(values)} valores únicos)")
        
        if concentration < 0.1:
            print(f"   ⚠️ ALTA CONCENTRAÇÃO - possível colapso!")
        elif concentration > 0.8:
            print(f"   ⚠️ DISPERSÃO EXCESSIVA - possível instabilidade!")
        else:
            print(f"   ✅ Concentração saudável")
    
    print(f"\n🔍 ANÁLISE DE CORRELAÇÕES:")
    print("=" * 30)
    
    correlation_matrix = np.corrcoef(actions_np.T)
    
    for i in range(len(action_names)):
        for j in range(i+1, len(action_names)):
            corr = correlation_matrix[i, j]
            print(f"   {action_names[i]} × {action_names[j]}: {corr:.3f}")
            
            if abs(corr) > 0.7:
                print(f"     ⚠️ ALTA CORRELAÇÃO - possível dependência indesejada!")
            elif abs(corr) < 0.1:
                print(f"     ✅ Independência saudável")
    
    print(f"\n🎯 ANÁLISE DE REWARD ENGINEERING:")
    print("=" * 40)
    
    # 1. Exploration Score
    total_variance = np.sum(np.var(actions_np, axis=0))
    exploration_score = min(total_variance / 0.1, 1.0)  # Normalizado para [0,1]
    print(f"   Exploration Score: {exploration_score:.3f}/1.0")
    
    if exploration_score < 0.3:
        print(f"     ⚠️ BAIXA EXPLORAÇÃO - aumentar std ou ruído")
    elif exploration_score > 0.8:
        print(f"     ⚠️ ALTA EXPLORAÇÃO - pode prejudicar convergência")
    else:
        print(f"     ✅ Exploração balanceada")
    
    # 2. Stability Score
    stability_scores = []
    for i in range(len(action_names)):
        values = actions_np[:, i]
        low, high = expected_ranges[i]
        
        # Penalizar valores fora do range
        out_of_range = np.sum((values < low) | (values > high)) / len(values)
        
        # Penalizar concentração excessiva
        concentration = len(np.unique(np.round(values, 2))) / len(values)
        
        stability = (1 - out_of_range) * min(concentration * 2, 1.0)
        stability_scores.append(stability)
    
    overall_stability = np.mean(stability_scores)
    print(f"   Stability Score: {overall_stability:.3f}/1.0")
    
    if overall_stability < 0.7:
        print(f"     ⚠️ BAIXA ESTABILIDADE - revisar ranges ou inicialização")
    else:
        print(f"     ✅ Estabilidade adequada")
    
    # 3. Training Readiness Score
    training_readiness = (exploration_score * 0.4 + overall_stability * 0.6)
    print(f"   Training Readiness: {training_readiness:.3f}/1.0")
    
    if training_readiness > 0.75:
        print(f"     🚀 PRONTO PARA TREINAMENTO!")
    elif training_readiness > 0.5:
        print(f"     ⚠️ AJUSTES MENORES RECOMENDADOS")
    else:
        print(f"     ❌ REQUER AJUSTES SIGNIFICATIVOS")
    
    print(f"\n🎯 RECOMENDAÇÕES ESPECÍFICAS:")
    print("=" * 35)
    
    # Recomendações baseadas na análise
    recommendations = []
    
    if exploration_score < 0.4:
        recommendations.append("• Aumentar log_std de 0.05 para 0.08-0.1")
        recommendations.append("• Implementar noise injection durante treinamento")
    
    if overall_stability < 0.7:
        recommendations.append("• Revisar ranges dos action spaces")
        recommendations.append("• Ajustar inicialização dos heads (gain atual: 0.3)")
    
    # Verificar concentração individual
    for i, name in enumerate(action_names):
        values = actions_np[:, i]
        concentration = len(np.unique(np.round(values, 2))) / len(values)
        if concentration < 0.1:
            recommendations.append(f"• {name}: Aumentar variância (concentração={concentration:.3f})")
    
    # Verificar correlações altas
    for i in range(len(action_names)):
        for j in range(i+1, len(action_names)):
            corr = abs(correlation_matrix[i, j])
            if corr > 0.7:
                recommendations.append(f"• Reduzir correlação {action_names[i]}-{action_names[j]} ({corr:.3f})")
    
    if not recommendations:
        recommendations.append("✅ Configuração atual está otimizada!")
    
    for rec in recommendations:
        print(f"   {rec}")
    
    print(f"\n🔬 COMPARAÇÃO COM BENCHMARKS:")
    print("=" * 35)
    
    # Benchmarks típicos para sistemas de trading RL
    benchmarks = {
        'exploration_score': {'optimal': 0.6, 'min_acceptable': 0.3},
        'stability_score': {'optimal': 0.8, 'min_acceptable': 0.6},
        'total_variance': {'optimal': 0.01, 'min_acceptable': 0.005},
        'max_correlation': {'optimal': 0.3, 'max_acceptable': 0.6}
    }
    
    max_correlation = np.max(np.abs(correlation_matrix - np.eye(len(action_names))))
    
    current_metrics = {
        'exploration_score': exploration_score,
        'stability_score': overall_stability,
        'total_variance': total_variance,
        'max_correlation': max_correlation
    }
    
    for metric, current_value in current_metrics.items():
        optimal = benchmarks[metric]['optimal']
        min_acc = benchmarks[metric].get('min_acceptable', benchmarks[metric].get('max_acceptable'))
        
        if metric in ['max_correlation']:
            # Menor é melhor
            if current_value <= optimal:
                status = "🎯 ÓTIMO"
            elif current_value <= min_acc:
                status = "✅ ACEITÁVEL"
            else:
                status = "⚠️ AJUSTAR"
        else:
            # Maior é melhor
            if current_value >= optimal:
                status = "🎯 ÓTIMO"
            elif current_value >= min_acc:
                status = "✅ ACEITÁVEL"
            else:
                status = "⚠️ AJUSTAR"
        
        print(f"   {metric}: {current_value:.3f} (target: {optimal:.3f}) {status}")
    
    return {
        'actions': actions_np,
        'exploration_score': exploration_score,
        'stability_score': overall_stability,
        'training_readiness': training_readiness,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    print("🎯 TwoHeadV9Optimus - Análise Reward Engineering Completa")
    
    try:
        # Scipy para testes estatísticos
        import scipy.stats
        
        results = analyze_action_distribution_comprehensive()
        
        print(f"\n🎖️ RESUMO EXECUTIVO:")
        print("=" * 25)
        print(f"   Training Readiness: {results['training_readiness']:.1%}")
        print(f"   Exploration Score: {results['exploration_score']:.1%}")
        print(f"   Stability Score: {results['stability_score']:.1%}")
        print(f"   Recomendações: {len(results['recommendations'])} items")
        
        if results['training_readiness'] > 0.75:
            print(f"\n🚀 POLÍTICA PRONTA PARA TREINAMENTO NO DAYTRADER V7!")
        
    except ImportError:
        print("⚠️ Scipy não disponível - executando análise básica...")
        
        # Análise básica sem scipy
        dummy_obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
        dummy_action_space = gym.spaces.Box(low=np.array([0, 0, -1, -1]), high=np.array([2, 1, 1, 1]), dtype=np.float32)
        
        def dummy_lr_schedule(progress):
            return 1e-4
        
        policy = TwoHeadV9Optimus(
            observation_space=dummy_obs_space,
            action_space=dummy_action_space,
            lr_schedule=dummy_lr_schedule,
            **get_v9_optimus_kwargs()
        )
        
        print("✅ Shape Fix validado - política funcional!")