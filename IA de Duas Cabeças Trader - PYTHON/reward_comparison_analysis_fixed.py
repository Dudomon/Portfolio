#!/usr/bin/env python3
"""
📊 ANÁLISE COMPARATIVA: V2 vs V3 BRUTAL
Comparação matemática entre o sistema atual e o novo sistema brutal
"""

import numpy as np
import matplotlib.pyplot as plt

def analyze_reward_systems():
    """
    Análise matemática comparativa entre sistemas de reward
    """
    
    # Cenários de teste (PnL em % do balance)
    pnl_scenarios = np.linspace(-0.20, 0.20, 41)  # -20% a +20%
    
    # Sistema V2 (atual - simulado baseado na análise)
    def v2_reward(pnl_pct):
        # Baseado na análise: peso dilufado, múltiplos componentes
        pnl_component = pnl_pct * 5.0  # Peso 5.0 mas diluído
        
        if pnl_pct > 0:
            bonus = 0.2 * pnl_pct  # Win bonus
            total = pnl_component + bonus
        else:
            penalty = -0.3 * abs(pnl_pct)  # Loss penalty (PEQUENA!)
            total = pnl_component + penalty
        
        # Normalização que mata o reward
        return total / 5.0
    
    # Sistema V3 Brutal (novo)
    def v3_brutal_reward(pnl_pct):
        base_reward = pnl_pct * 100.0  # Amplificação direta
        
        if pnl_pct < -0.05:  # Perdas > 5%
            # PAIN MULTIPLICATION
            base_reward *= 4.0
        elif pnl_pct > 0.03:  # Lucros > 3%
            # Bonus pequeno
            base_reward *= 1.2
            
        return np.clip(base_reward, -50, 50)
    
    # Calcular rewards para todos os cenários
    v2_rewards = [v2_reward(pnl) for pnl in pnl_scenarios]
    v3_rewards = [v3_brutal_reward(pnl) for pnl in pnl_scenarios]
    
    # Análise matemática
    print("🔥 ANÁLISE MATEMÁTICA COMPARATIVA")
    print("=" * 60)
    
    # Casos específicos
    test_cases = [-0.10, -0.05, -0.02, 0.00, 0.02, 0.05, 0.10]
    
    print("\n📊 COMPARAÇÃO DIRETA:")
    print("PnL%     V2 Reward    V3 Brutal    Ratio (V3/V2)")
    print("-" * 50)
    
    for pnl in test_cases:
        v2_r = v2_reward(pnl)
        v3_r = v3_brutal_reward(pnl)
        ratio = v3_r / v2_r if v2_r != 0 else "INF"
        
        print(f"{pnl*100:+5.0f}%    {v2_r:+8.2f}    {v3_r:+8.2f}    {ratio}")
    
    # Análise de assimetria gain/loss
    print("\n🎯 ANÁLISE DE ASSIMETRIA GAIN/LOSS:")
    
    gain_5pct_v2 = v2_reward(0.05)
    loss_5pct_v2 = v2_reward(-0.05)
    ratio_v2 = abs(gain_5pct_v2 / loss_5pct_v2) if loss_5pct_v2 != 0 else 0
    
    gain_5pct_v3 = v3_brutal_reward(0.05)
    loss_5pct_v3 = v3_brutal_reward(-0.05)
    ratio_v3 = abs(gain_5pct_v3 / loss_5pct_v3) if loss_5pct_v3 != 0 else 0
    
    print(f"V2 - Ganho 5%: {gain_5pct_v2:+.2f}, Perda 5%: {loss_5pct_v2:+.2f}, Ratio: {ratio_v2:.2f}")
    print(f"V3 - Ganho 5%: {gain_5pct_v3:+.2f}, Perda 5%: {loss_5pct_v3:+.2f}, Ratio: {ratio_v3:.2f}")
    
    print(f"\n🔥 PAIN AMPLIFICATION:")
    print(f"V2: Perda 10% = {v2_reward(-0.10):+.2f}")
    print(f"V3: Perda 10% = {v3_brutal_reward(-0.10):+.2f}")
    print(f"V3 aplica {abs(v3_brutal_reward(-0.10) / v2_reward(-0.10)):.1f}x mais DOR!")
    
    # Análise de gradiente (derivada)
    print(f"\n📈 ANÁLISE DE GRADIENTE (incentivo para melhoria):")
    
    def gradient_v2(pnl):
        delta = 0.001
        return (v2_reward(pnl + delta) - v2_reward(pnl - delta)) / (2 * delta)
    
    def gradient_v3(pnl):
        delta = 0.001  
        return (v3_brutal_reward(pnl + delta) - v3_brutal_reward(pnl - delta)) / (2 * delta)
    
    for pnl in [-0.05, 0.00, 0.05]:
        grad_v2 = gradient_v2(pnl)
        grad_v3 = gradient_v3(pnl)
        print(f"PnL {pnl*100:+3.0f}% - Gradiente V2: {grad_v2:+6.1f}, V3: {grad_v3:+6.1f} (ratio: {grad_v3/grad_v2:.1f}x)")
    
    return pnl_scenarios, v2_rewards, v3_rewards

def plot_comparison(pnl_scenarios, v2_rewards, v3_rewards):
    """
    Plotar comparação visual dos sistemas
    """
    plt.figure(figsize=(12, 8)
    
    # Converter PnL para percentual
    pnl_pct = pnl_scenarios * 100
    
    plt.subplot(2, 1, 1)
    plt.plot(pnl_pct, v2_rewards, 'b-', label='V2 (Atual - Dilufado)', linewidth=2)
    plt.plot(pnl_pct, v3_rewards, 'r-', label='V3 (Brutal)', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('PnL (%)')
    plt.ylabel('Reward')
    plt.title('Comparação: Sistema V2 vs V3 Brutal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Ratio
    plt.subplot(2, 1, 2)
    ratios = [v3/v2 if v2 != 0 else 0 for v2, v3 in zip(v2_rewards, v3_rewards)]
    plt.plot(pnl_pct, ratios, 'g-', linewidth=2)
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Paridade')
    plt.xlabel('PnL (%)')
    plt.ylabel('Ratio V3/V2')
    plt.title('Amplificação do V3 vs V2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('D:/Projeto/reward_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    pnl_scenarios, v2_rewards, v3_rewards = analyze_reward_systems()
    
    try:
        plot_comparison(pnl_scenarios, v2_rewards, v3_rewards)
        print(f"\n📊 Gráfico salvo em: D:/Projeto/reward_comparison.png")
    except Exception as e:
        print(f"Erro ao plotar (normal em servidor): {e}")
    
    print(f"\n🎯 CONCLUSÃO:")
    print(f"V3 Brutal fornece:")
    print(f"✅ PAIN REAL para perdas grandes (4x amplificação)")
    print(f"✅ Incentivo direto para lucro (sem diluição)")
    print(f"✅ Gradientes mais fortes para aprendizado")
    print("✅ Simplicidade (200 linhas vs 1400)")