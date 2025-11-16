#!/usr/bin/env python3
"""
🧪 TESTE DE BALANCEAMENTO V3 ELEGANCE
Analisa balance entre componentes e signal-to-noise ratio
"""

import sys
import os
sys.path.append("D:/Projeto")

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from trading_framework.rewards.reward_daytrade_v3_elegance import ElegantDayTradingRewardV3, V3EleganceConfig

class MockTradingEnv:
    """Mock environment para testes controlados"""
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.portfolio_value = initial_balance
        self.peak_portfolio_value = initial_balance
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.positions = []
        self.trades = []
        self.current_step = 0
        self.positive_days_streak = 1
        self.last_confidence_score = 0.5
        
    def update_portfolio(self, pnl_percent: float):
        """Atualizar portfolio com PnL específico"""
        self.portfolio_value = self.initial_balance * (1 + pnl_percent)
        
        # Atualizar peak e drawdown
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
            
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
    
    def add_trade(self, pnl_usd: float):
        """Adicionar trade ao histórico"""
        trade = {
            'pnl_usd': pnl_usd,
            'entry_step': self.current_step - 10,
            'exit_step': self.current_step,
            'duration': 10
        }
        self.trades.append(trade)
    
    def set_positions(self, total_risk_percent: float):
        """Simular posições abertas com risco específico"""
        if total_risk_percent > 0:
            self.positions = [
                {'volume': total_risk_percent * self.portfolio_value, 'entry_step': self.current_step - 5}
            ]
        else:
            self.positions = []

def test_component_balance():
    """🧪 Teste de balanceamento entre componentes"""
    
    print("🧪 TESTE DE BALANCEAMENTO V3 ELEGANCE")
    print("=" * 60)
    
    # Configurar reward system
    config = V3EleganceConfig()
    reward_system = ElegantDayTradingRewardV3(config)
    
    # Cenários de teste balanceados
    test_scenarios = [
        # (pnl_percent, risk_percent, overtrading, description)
        (0.05, 0.02, 10, "Lucro 5% - Low Risk"),
        (0.10, 0.03, 15, "Lucro 10% - Normal Risk"),
        (0.03, 0.08, 25, "Lucro 3% - High Risk"),
        (-0.05, 0.02, 10, "Loss 5% - Low Risk"), 
        (-0.10, 0.03, 15, "Loss 10% - Normal Risk"),
        (-0.03, 0.15, 35, "Loss 3% - Extreme Risk"),
        (0.25, 0.02, 60, "Lucro 25% - Overtrading"),
        (-0.25, 0.12, 45, "Loss 25% - Multi Risk"),
        (0.01, 0.01, 5, "Neutro - Minimal Activity"),
        (0.0, 0.0, 0, "Zero - No Activity")
    ]
    
    results = []
    
    for pnl_pct, risk_pct, trades_count, description in test_scenarios:
        # Criar ambiente mock
        env = MockTradingEnv()
        reward_system.reset_episode(env.initial_balance)
        
        # Configurar cenário
        env.update_portfolio(pnl_pct)
        env.set_positions(risk_pct)
        
        # Simular overtrading
        for i in range(trades_count):
            env.add_trade(pnl_pct * env.initial_balance / max(trades_count, 1))
            env.current_step += 1
        
        # Calcular reward
        dummy_action = np.array([0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0])
        reward, info = reward_system.calculate_final_reward(env, dummy_action, {})
        
        # Análise de componentes
        pnl_component = info.get('pnl_component', 0.0)
        risk_component = info.get('risk_component', 0.0)
        
        # Calcular contribuição relativa
        total_abs = abs(pnl_component) + abs(risk_component)
        pnl_contribution = abs(pnl_component) / max(total_abs, 0.001) * 100
        risk_contribution = abs(risk_component) / max(total_abs, 0.001) * 100
        
        result = {
            'scenario': description,
            'pnl_percent': pnl_pct * 100,
            'risk_percent': risk_pct * 100,
            'trades': trades_count,
            'final_reward': reward,
            'pnl_component': pnl_component,
            'risk_component': risk_component,
            'pnl_contribution_pct': pnl_contribution,
            'risk_contribution_pct': risk_contribution,
            'reward_regime': info.get('regime', 'unknown')
        }
        results.append(result)
        
        print(f"\n📊 {description}")
        print(f"   PnL: {pnl_pct*100:+.1f}% | Risk: {risk_pct*100:.1f}% | Trades: {trades_count}")
        print(f"   Final Reward: {reward:+.3f}")
        print(f"   ├─ PnL Component: {pnl_component:+.3f} ({pnl_contribution:.0f}%)")
        print(f"   └─ Risk Component: {risk_component:+.3f} ({risk_contribution:.0f}%)")
    
    return results

def analyze_signal_noise_ratio(results: List[Dict]):
    """📊 Análise de signal-to-noise ratio"""
    
    print(f"\n🔍 ANÁLISE SIGNAL-TO-NOISE RATIO")
    print("=" * 50)
    
    # Extrair dados
    rewards = [r['final_reward'] for r in results]
    pnl_components = [r['pnl_component'] for r in results]
    risk_components = [r['risk_component'] for r in results]
    pnl_percents = [r['pnl_percent'] for r in results]
    
    # Calcular correlações
    pnl_corr = np.corrcoef(pnl_percents, rewards)[0, 1] if len(set(pnl_percents)) > 1 else 0
    
    # Signal strength (correlação PnL vs Reward)
    signal_strength = abs(pnl_corr)
    
    # Noise analysis (variabilidade não explicada por PnL)
    pnl_rewards = [r['pnl_component'] for r in results]
    total_rewards = [r['final_reward'] for r in results]
    
    # Noise = std(total_reward - pnl_reward) / std(total_reward)
    residuals = np.array(total_rewards) - np.array(pnl_rewards)
    noise_ratio = np.std(residuals) / max(np.std(total_rewards), 0.001)
    
    # Component balance analysis
    pnl_contributions = [r['pnl_contribution_pct'] for r in results]
    risk_contributions = [r['risk_contribution_pct'] for r in results]
    
    avg_pnl_contrib = np.mean(pnl_contributions)
    avg_risk_contrib = np.mean(risk_contributions)
    
    # Balance consistency
    pnl_std = np.std(pnl_contributions)
    risk_std = np.std(risk_contributions)
    balance_consistency = 100 - max(pnl_std, risk_std)
    
    print(f"🎯 SIGNAL STRENGTH:")
    print(f"   PnL-Reward Correlation: {pnl_corr:.3f}")
    print(f"   Signal Strength: {signal_strength:.3f} ({'FORTE' if signal_strength > 0.8 else 'MÉDIO' if signal_strength > 0.5 else 'FRACO'})")
    
    print(f"\n🔊 NOISE ANALYSIS:")
    print(f"   Noise Ratio: {noise_ratio:.3f} ({'BAIXO' if noise_ratio < 0.2 else 'MÉDIO' if noise_ratio < 0.5 else 'ALTO'})")
    print(f"   Signal-to-Noise: {1/max(noise_ratio, 0.001):.2f}:1")
    
    print(f"\n⚖️ COMPONENT BALANCE:")
    print(f"   PnL Contribution: {avg_pnl_contrib:.1f}% ± {pnl_std:.1f}%")
    print(f"   Risk Contribution: {avg_risk_contrib:.1f}% ± {risk_std:.1f}%")
    print(f"   Balance Consistency: {balance_consistency:.1f}%")
    
    # Avaliação geral
    print(f"\n📋 AVALIAÇÃO GERAL:")
    
    # Signal quality
    if signal_strength > 0.8:
        signal_grade = "A"
    elif signal_strength > 0.6:
        signal_grade = "B"
    elif signal_strength > 0.4:
        signal_grade = "C"
    else:
        signal_grade = "D"
    
    # Noise quality
    if noise_ratio < 0.2:
        noise_grade = "A"
    elif noise_ratio < 0.4:
        noise_grade = "B"
    elif noise_ratio < 0.6:
        noise_grade = "C"
    else:
        noise_grade = "D"
    
    # Balance quality
    if balance_consistency > 80:
        balance_grade = "A"
    elif balance_consistency > 60:
        balance_grade = "B"
    elif balance_consistency > 40:
        balance_grade = "C"
    else:
        balance_grade = "D"
    
    print(f"   Signal Quality: {signal_grade} ({signal_strength:.3f})")
    print(f"   Noise Control: {noise_grade} ({noise_ratio:.3f})")
    print(f"   Balance Consistency: {balance_grade} ({balance_consistency:.1f}%)")
    
    # Overall grade
    grades = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
    overall_score = (grades[signal_grade] + grades[noise_grade] + grades[balance_grade]) / 3
    
    if overall_score >= 3.5:
        overall_grade = "A"
        quality = "EXCELENTE"
    elif overall_score >= 2.5:
        overall_grade = "B" 
        quality = "BOM"
    elif overall_score >= 1.5:
        overall_grade = "C"
        quality = "MÉDIO"
    else:
        overall_grade = "D"
        quality = "RUIM"
    
    print(f"\n🎯 QUALIDADE GERAL: {overall_grade} - {quality}")
    
    return {
        'signal_strength': signal_strength,
        'noise_ratio': noise_ratio,
        'signal_to_noise': 1/max(noise_ratio, 0.001),
        'pnl_contribution': avg_pnl_contrib,
        'risk_contribution': avg_risk_contrib,
        'balance_consistency': balance_consistency,
        'overall_grade': overall_grade,
        'quality': quality
    }

def test_edge_cases():
    """🧪 Teste de casos extremos"""
    
    print(f"\n🚨 TESTE DE CASOS EXTREMOS")
    print("=" * 40)
    
    config = V3EleganceConfig()
    reward_system = ElegantDayTradingRewardV3(config)
    
    extreme_cases = [
        (0.5, 0.0, 0, "MEGA LUCRO +50%"),
        (-0.5, 0.0, 0, "MEGA LOSS -50%"), 
        (0.1, 0.5, 0, "Lucro + EXTREME RISK"),
        (0.0, 0.0, 200, "EXTREME OVERTRADING"),
        (0.001, 0.001, 1, "MICRO MOVEMENT"),
        (1.0, 1.0, 500, "TUDO NO MÁXIMO")
    ]
    
    for pnl_pct, risk_pct, trades, description in extreme_cases:
        env = MockTradingEnv()
        reward_system.reset_episode(env.initial_balance)
        
        env.update_portfolio(pnl_pct)
        env.set_positions(risk_pct)
        
        for i in range(trades):
            env.add_trade(pnl_pct * env.initial_balance / max(trades, 1))
        
        dummy_action = np.array([0.5] * 8)
        reward, info = reward_system.calculate_final_reward(env, dummy_action, {})
        
        print(f"   {description}: {reward:+.3f}")
    
    print(f"\n✅ Edge cases testados - sistema mantém estabilidade")

def generate_balance_report():
    """📊 Gerar relatório completo de balanceamento"""
    
    print(f"\n📋 EXECUTANDO ANÁLISE COMPLETA...")
    
    # Executar testes
    balance_results = test_component_balance()
    noise_analysis = analyze_signal_noise_ratio(balance_results)
    test_edge_cases()
    
    # Salvar relatório
    report_file = "D:/Projeto/avaliacoes/v3_elegance_balance_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("🚀 RELATÓRIO DE BALANCEAMENTO V3 ELEGANCE\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("📊 RESULTADOS PRINCIPAIS:\n")
        f.write(f"   Signal Strength: {noise_analysis['signal_strength']:.3f}\n")
        f.write(f"   Noise Ratio: {noise_analysis['noise_ratio']:.3f}\n") 
        f.write(f"   Signal-to-Noise: {noise_analysis['signal_to_noise']:.2f}:1\n")
        f.write(f"   PnL Contribution: {noise_analysis['pnl_contribution']:.1f}%\n")
        f.write(f"   Risk Contribution: {noise_analysis['risk_contribution']:.1f}%\n")
        f.write(f"   Balance Consistency: {noise_analysis['balance_consistency']:.1f}%\n")
        f.write(f"   Overall Grade: {noise_analysis['overall_grade']} - {noise_analysis['quality']}\n\n")
        
        f.write("📋 CENÁRIOS TESTADOS:\n")
        for result in balance_results:
            f.write(f"   {result['scenario']}: {result['final_reward']:+.3f}\n")
            f.write(f"     └─ PnL: {result['pnl_contribution_pct']:.0f}% | Risk: {result['risk_contribution_pct']:.0f}%\n")
    
    print(f"💾 Relatório salvo: {report_file}")
    
    return noise_analysis

if __name__ == "__main__":
    print("🚀 V3 ELEGANCE BALANCE ANALYZER")
    print("Testando balanceamento de componentes e signal-to-noise ratio")
    
    final_analysis = generate_balance_report()
    
    print(f"\n🎯 CONCLUSÃO FINAL:")
    print(f"V3 Elegance - Qualidade: {final_analysis['quality']}")
    print(f"Signal-to-Noise: {final_analysis['signal_to_noise']:.2f}:1")
    print(f"Balance: PnL {final_analysis['pnl_contribution']:.0f}% | Risk {final_analysis['risk_contribution']:.0f}%")