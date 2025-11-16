#!/usr/bin/env python3
"""
🧪 TESTE DE BALANCEAMENTO REWARD V3 BRUTAL
Análise detalhada dos componentes e balanceamento do sistema
"""

import numpy as np
import sys
sys.path.append("D:/Projeto")

from trading_framework.rewards.reward_daytrade_v3_brutal import BrutalMoneyReward

class MockTradingEnv:
    """Ambiente mock para testes controlados"""
    
    def __init__(self, scenario_name, **kwargs):
        self.scenario_name = scenario_name
        
        # Valores padrão
        defaults = {
            'total_realized_pnl': 0.0,
            'total_unrealized_pnl': 0.0,
            'portfolio_value': 10000.0,
            'peak_portfolio_value': 10000.0,
            'position_count': 0,
            'total_trades': 0,
            'win_trades': 0,
            'loss_trades': 0,
            'current_drawdown': 0.0
        }
        
        # Aplicar valores específicos do cenário
        for key, value in defaults.items():
            setattr(self, key, kwargs.get(key, value))

def test_component_breakdown():
    """
    Testa o breakdown detalhado dos componentes do reward
    """
    
    print("🧪 TESTE DE COMPONENTES DO REWARD V3 BRUTAL")
    print("=" * 80)
    
    reward_system = BrutalMoneyReward(initial_balance=10000)
    
    # Cenários específicos para testar cada componente
    scenarios = [
        # Cenário 1: Lucro pequeno sem risk
        {
            'name': "💰 Lucro pequeno (2%)",
            'kwargs': {
                'total_realized_pnl': 200,
                'total_unrealized_pnl': 0,
                'portfolio_value': 10200,
                'peak_portfolio_value': 10200
            }
        },
        
        # Cenário 2: Lucro médio com bonus
        {
            'name': "💰 Lucro médio (4% - com bonus)",
            'kwargs': {
                'total_realized_pnl': 400,
                'total_unrealized_pnl': 0,
                'portfolio_value': 10400,
                'peak_portfolio_value': 10400
            }
        },
        
        # Cenário 3: Perda pequena
        {
            'name': "📉 Perda pequena (3%)",
            'kwargs': {
                'total_realized_pnl': -300,
                'total_unrealized_pnl': 0,
                'portfolio_value': 9700,
                'peak_portfolio_value': 10000
            }
        },
        
        # Cenário 4: Perda grande com PAIN
        {
            'name': "💥 Perda grande (8% - PAIN ativado)",
            'kwargs': {
                'total_realized_pnl': -800,
                'total_unrealized_pnl': 0,
                'portfolio_value': 9200,
                'peak_portfolio_value': 10000
            }
        },
        
        # Cenário 5: Drawdown severo
        {
            'name': "⚠️ Drawdown severo (25%)",
            'kwargs': {
                'total_realized_pnl': 0,
                'total_unrealized_pnl': -2500,
                'portfolio_value': 7500,
                'peak_portfolio_value': 10000
            }
        },
        
        # Cenário 6: Mix realizado + não realizado
        {
            'name': "🔄 Mix: +3% realizado, -1% não realizado",
            'kwargs': {
                'total_realized_pnl': 300,
                'total_unrealized_pnl': -100,
                'portfolio_value': 10200,
                'peak_portfolio_value': 10300
            }
        },
        
        # Cenário 7: Break even
        {
            'name': "⚖️ Break even",
            'kwargs': {
                'total_realized_pnl': 0,
                'total_unrealized_pnl': 0,
                'portfolio_value': 10000,
                'peak_portfolio_value': 10000
            }
        },
    ]
    
    print("\n📊 ANÁLISE DETALHADA POR CENÁRIO:")
    print("-" * 80)
    
    for scenario in scenarios:
        env = MockTradingEnv(scenario['name'], **scenario['kwargs'])
        
        # Calcular reward e componentes
        reward, info, done = reward_system.calculate_reward_and_info(
            env, np.zeros(8), {}
        )
        
        print(f"\n{scenario['name']}")
        print(f"  📈 Realized PnL: ${info.get('realized_pnl', 0):+.0f}")
        print(f"  📊 Unrealized PnL: ${info.get('unrealized_pnl', 0):+.0f}")
        print(f"  💰 Total PnL: ${info.get('total_pnl', 0):+.0f} ({info.get('pnl_percent', 0):+.1f}%)")
        print(f"  📉 Drawdown: {info.get('portfolio_drawdown', 0):.1f}%")
        print(f"  ⚡ PAIN Applied: {'YES' if info.get('pain_applied', False) else 'NO'}")
        print(f"  🎯 PnL Reward: {info.get('pnl_reward', 0):+.2f}")
        print(f"  ⚠️ Risk Penalty: {info.get('risk_penalty', 0):+.2f}")
        print(f"  🏆 TOTAL REWARD: {reward:+.2f}")
        print(f"  🛑 Episode Done: {'YES' if done else 'NO'}")

def test_balance_analysis():
    """
    Testa o balanceamento entre PnL e Risk components
    """
    
    print(f"\n\n🎯 ANÁLISE DE BALANCEAMENTO PNL vs RISK")
    print("=" * 80)
    
    reward_system = BrutalMoneyReward(initial_balance=10000)
    
    # Teste de ranges para análise
    pnl_ranges = [-0.20, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.20]
    risk_ranges = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    
    print("\n📊 MATRIZ DE BALANCEAMENTO (PnL% vs Drawdown%):")
    print("PnL%\\DD%", end="")
    for risk in risk_ranges:
        print(f"    {risk*100:4.0f}%", end="")
    print()
    
    for pnl in pnl_ranges:
        print(f"{pnl*100:+4.0f}%   ", end="")
        
        for risk in risk_ranges:
            # Criar ambiente para este caso
            portfolio_value = 10000 * (1 + pnl)
            peak_value = 10000 if risk == 0 else portfolio_value / (1 - risk)
            
            env = MockTradingEnv(
                f"test_{pnl}_{risk}",
                total_realized_pnl=pnl * 10000,
                portfolio_value=portfolio_value,
                peak_portfolio_value=peak_value
            )
            
            reward, info, done = reward_system.calculate_reward_and_info(
                env, np.zeros(8), {}
            )
            
            print(f"  {reward:+5.1f}", end="")
        print()

def test_pain_thresholds():
    """
    Testa os thresholds de PAIN multiplication
    """
    
    print(f"\n\n💥 ANÁLISE DOS THRESHOLDS DE PAIN")
    print("=" * 80)
    
    reward_system = BrutalMoneyReward(initial_balance=10000)
    
    # Teste em torno do threshold de 5%
    pnl_test_points = np.linspace(-0.15, 0.15, 31)
    
    print("PnL%     Reward    Pain?   Multiplier")
    print("-" * 40)
    
    for pnl in pnl_test_points:
        env = MockTradingEnv(
            f"pain_test_{pnl}",
            total_realized_pnl=pnl * 10000,
            portfolio_value=10000 * (1 + pnl),
            peak_portfolio_value=10000 if pnl <= 0 else 10000 * (1 + pnl)
        )
        
        reward, info, done = reward_system.calculate_reward_and_info(
            env, np.zeros(8), {}
        )
        
        # Calcular multiplier efetivo
        base_reward = pnl * 100
        actual_pnl_reward = info.get('pnl_reward', 0)
        multiplier = abs(actual_pnl_reward / base_reward) if base_reward != 0 else 1.0
        
        pain_marker = "💥" if info.get('pain_applied', False) else "  "
        
        print(f"{pnl*100:+5.1f}%   {reward:+6.1f}   {pain_marker}   {multiplier:5.1f}x")

def test_risk_thresholds():
    """
    Testa os thresholds de risk management
    """
    
    print(f"\n\n⚠️ ANÁLISE DOS THRESHOLDS DE RISK")
    print("=" * 80)
    
    reward_system = BrutalMoneyReward(initial_balance=10000)
    
    # Teste em torno do threshold de 15%
    risk_test_points = np.linspace(0.0, 0.40, 21)
    
    print("Drawdown%    Risk Penalty    Threshold Breach")
    print("-" * 50)
    
    for risk in risk_test_points:
        portfolio_value = 10000 * (1 - risk)
        peak_value = 10000
        
        env = MockTradingEnv(
            f"risk_test_{risk}",
            portfolio_value=portfolio_value,
            peak_portfolio_value=peak_value
        )
        
        reward, info, done = reward_system.calculate_reward_and_info(
            env, np.zeros(8), {}
        )
        
        breach_marker = "🚨" if info.get('risk_threshold_breached', False) else "  "
        
        print(f"   {risk*100:5.1f}%      {info.get('risk_penalty', 0):+8.2f}        {breach_marker}")

def test_early_termination():
    """
    Testa o sistema de early termination
    """
    
    print(f"\n\n🛑 ANÁLISE DE EARLY TERMINATION")
    print("=" * 80)
    
    reward_system = BrutalMoneyReward(initial_balance=10000)
    
    # Teste de diferentes níveis de perda
    loss_levels = np.linspace(0.0, 0.80, 17)
    
    print("Portfolio Loss%    Episode Done?    Reason")
    print("-" * 50)
    
    for loss in loss_levels:
        portfolio_value = 10000 * (1 - loss)
        
        env = MockTradingEnv(
            f"termination_test_{loss}",
            portfolio_value=portfolio_value,
            peak_portfolio_value=10000
        )
        
        reward, info, done = reward_system.calculate_reward_and_info(
            env, np.zeros(8), {}
        )
        
        reason = "Catastrophic loss (>50%)" if done else "Continue"
        done_marker = "🛑" if done else "✅"
        
        print(f"     {loss*100:5.1f}%         {done_marker}         {reason}")

if __name__ == "__main__":
    test_component_breakdown()
    test_balance_analysis()
    test_pain_thresholds()
    test_risk_thresholds()
    test_early_termination()
    
    print(f"\n\n🎯 RESUMO DO BALANCEAMENTO:")
    print("=" * 80)
    print("✅ PnL Component: 90% do reward (dominante)")
    print("✅ Risk Component: 10% do reward (apenas para casos extremos)")
    print("💥 PAIN Threshold: Perdas > 5% recebem 4x amplificação")
    print("⚠️ Risk Threshold: Drawdown > 15% recebe penalty severa")
    print("🛑 Termination: Portfolio loss > 50% termina episódio")
    print("🎯 Focus: 100% alinhado com fazer DINHEIRO, não métricas acadêmicas")