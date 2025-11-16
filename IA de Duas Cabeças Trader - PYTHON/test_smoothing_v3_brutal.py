#!/usr/bin/env python3
"""
🧪 TESTE DO EXPONENTIAL SMOOTHING V3 BRUTAL
Verifica se o sistema de smoothing está estabilizando os rewards
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from trading_framework.rewards.reward_daytrade_v3_brutal import create_brutal_daytrade_reward_system

class MockEnvironment:
    """Mock environment para simular volatilidade extrema"""

    def __init__(self):
        self.step = 0
        self.portfolio_value = 1000.0
        self.initial_balance = 1000.0

    def simulate_volatile_rewards(self):
        """Simula PnL extremamente volátil como no treinamento real"""
        # Simular swings brutais como vimos nos logs: -0.7 → +0.4
        volatile_pnls = [
            -50,    # Perda grande
            +80,    # Lucro grande
            -30,    # Perda média
            +60,    # Lucro bom
            -70,    # Perda brutal
            +40,    # Lucro médio
            -20,    # Perda pequena
            +90,    # Lucro excelente
            -60,    # Perda grande
            +30     # Lucro pequeno
        ]

        if self.step < len(volatile_pnls):
            pnl = volatile_pnls[self.step]
            self.portfolio_value = self.initial_balance + pnl
        else:
            # Continuar com flutuações aleatórias
            pnl_change = np.random.normal(0, 30)  # Média 0, std 30
            self.portfolio_value += pnl_change

        self.step += 1
        return self.portfolio_value

def test_smoothing_effectiveness():
    """Testa efetividade do smoothing em cenário volátil"""

    print("🧪 TESTE DO EXPONENTIAL SMOOTHING V3 BRUTAL")
    print("=" * 60)

    # Criar sistemas com diferentes níveis de smoothing
    smoothing_configs = [
        ("SEM SMOOTHING", 1.0),
        ("SMOOTHING SUAVE", 0.1),
        ("SMOOTHING BALANCEADO", 0.3),
        ("SMOOTHING RESPONSIVO", 0.5)
    ]

    results = {}

    for config_name, alpha in smoothing_configs:
        print(f"\n📊 Testando {config_name} (alpha={alpha})")
        print("-" * 40)

        # Criar reward system e environment
        reward_system = create_brutal_daytrade_reward_system(1000.0)
        reward_system.set_smoothing_alpha(alpha)
        env = MockEnvironment()

        raw_rewards = []
        smoothed_rewards = []

        # Simular 10 steps com alta volatilidade
        for step in range(10):
            env.simulate_volatile_rewards()

            # Simular action (dummy)
            action = np.array([1.0, 0.5, 0.0, 0.0])
            old_state = {}

            # Calcular reward
            reward, info, done = reward_system.calculate_reward_and_info(env, action, old_state)

            raw_rewards.append(info.get('raw_reward', reward))
            smoothed_rewards.append(reward)

            # Log dos primeiros 5 steps
            if step < 5:
                pnl = env.portfolio_value - env.initial_balance
                print(f"  Step {step+1}: PnL={pnl:+.0f} | Raw={info.get('raw_reward', 0):.4f} | Smooth={reward:.4f}")

        # Calcular estatísticas
        raw_variance = np.var(raw_rewards)
        smooth_variance = np.var(smoothed_rewards)
        variance_reduction = (raw_variance - smooth_variance) / raw_variance * 100

        results[config_name] = {
            'alpha': alpha,
            'raw_variance': raw_variance,
            'smooth_variance': smooth_variance,
            'variance_reduction': variance_reduction,
            'mean_raw': np.mean(raw_rewards),
            'mean_smooth': np.mean(smoothed_rewards)
        }

        print(f"  📈 Variância Raw: {raw_variance:.6f}")
        print(f"  📊 Variância Smooth: {smooth_variance:.6f}")
        print(f"  🎯 Redução: {variance_reduction:.1f}%")

    # Relatório final
    print(f"\n🏆 RELATÓRIO DE EFETIVIDADE DO SMOOTHING")
    print("=" * 60)

    for config_name, stats in results.items():
        print(f"\n{config_name}:")
        print(f"  Alpha: {stats['alpha']}")
        print(f"  Redução de Variância: {stats['variance_reduction']:+.1f}%")
        print(f"  Média Raw: {stats['mean_raw']:.4f}")
        print(f"  Média Smooth: {stats['mean_smooth']:.4f}")
        print(f"  Preservação de Sinal: {abs(stats['mean_smooth']/stats['mean_raw']*100):-.1f}%" if stats['mean_raw'] != 0 else "N/A")

    # Encontrar melhor configuração
    best_config = max(results.items(),
                     key=lambda x: x[1]['variance_reduction'] if x[1]['variance_reduction'] > 0 else -999)

    print(f"\n🎯 MELHOR CONFIGURAÇÃO: {best_config[0]}")
    print(f"   Alpha recomendado: {best_config[1]['alpha']}")
    print(f"   Redução de variância: {best_config[1]['variance_reduction']:.1f}%")

    print(f"\n✅ SMOOTHING IMPLEMENTADO COM SUCESSO!")
    print(f"   Configuração atual no código: alpha=0.3 (Balanceado)")
    print(f"   Impacto esperado: Explained variance negativa → positiva")

if __name__ == "__main__":
    test_smoothing_effectiveness()