#!/usr/bin/env python3
"""
🧪 TESTE DE BALANCEAMENTO - REWARD SYSTEM V2
Detectar problemas que podem confundir o crítico
"""

import sys
sys.path.append("D:/Projeto")
import numpy as np
from typing import Dict, List, Tuple

def create_mock_env(trades: List[Dict], portfolio_value: float = 1000.0):
    """Criar mock environment para testes"""
    class MockEnv:
        def __init__(self):
            self.trades = trades
            self.portfolio_value = portfolio_value
            self.initial_balance = 1000.0
            self.realized_balance = portfolio_value
            self.balance = portfolio_value
            self.current_step = len(trades) * 10  # Simular steps
            self.peak_portfolio = max(1000.0, portfolio_value)
            self.current_drawdown = max(0, (self.peak_portfolio - portfolio_value) / self.peak_portfolio)
            self.max_drawdown = self.current_drawdown
            self.current_positions = 0
            self.reward_history_size = 100
            self.recent_rewards = []
            
    return MockEnv()

def test_reward_balance():
    print("🧪 TESTE DE BALANCEAMENTO - REWARD SYSTEM V2")
    print("=" * 60)
    
    try:
        from trading_framework.rewards.reward_daytrade_v2 import BalancedDayTradingRewardCalculator
        reward_calc = BalancedDayTradingRewardCalculator()
        
        print("✅ Reward calculator importado com sucesso")
        print(f"   Tipo: {type(reward_calc).__name__}")
        
    except Exception as e:
        print(f"❌ Erro na importação: {e}")
        return False
    
    # CENÁRIOS DE TESTE
    test_scenarios = [
        {
            'name': 'Scenario 1: Pure Profit',
            'trades': [
                {'pnl_usd': 50, 'duration_steps': 20, 'position_size': 0.02},
                {'pnl_usd': 30, 'duration_steps': 15, 'position_size': 0.02},
                {'pnl_usd': 75, 'duration_steps': 25, 'position_size': 0.03}
            ],
            'portfolio': 1155,
            'expected_trend': 'positive'
        },
        {
            'name': 'Scenario 2: Pure Loss', 
            'trades': [
                {'pnl_usd': -40, 'duration_steps': 12, 'position_size': 0.02},
                {'pnl_usd': -25, 'duration_steps': 8, 'position_size': 0.01},
                {'pnl_usd': -35, 'duration_steps': 18, 'position_size': 0.02}
            ],
            'portfolio': 900,
            'expected_trend': 'negative'
        },
        {
            'name': 'Scenario 3: Mixed (Break-even)',
            'trades': [
                {'pnl_usd': 60, 'duration_steps': 22, 'position_size': 0.03},
                {'pnl_usd': -35, 'duration_steps': 10, 'position_size': 0.02},
                {'pnl_usd': -25, 'duration_steps': 15, 'position_size': 0.01}
            ],
            'portfolio': 1000,
            'expected_trend': 'neutral'
        },
        {
            'name': 'Scenario 4: Micro-trading (Many small)',
            'trades': [{'pnl_usd': 2, 'duration_steps': 3, 'position_size': 0.005} for _ in range(50)],
            'portfolio': 1100,
            'expected_trend': 'micro_farming'
        },
        {
            'name': 'Scenario 5: Quality trading (Few large)',
            'trades': [
                {'pnl_usd': 120, 'duration_steps': 45, 'position_size': 0.03},
                {'pnl_usd': 95, 'duration_steps': 38, 'position_size': 0.025},
                {'pnl_usd': -15, 'duration_steps': 12, 'position_size': 0.01}
            ],
            'portfolio': 1200,
            'expected_trend': 'quality'
        }
    ]
    
    print("\n📊 EXECUTANDO CENÁRIOS DE TESTE")
    print("=" * 60)
    
    results = []
    balance_issues = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🎯 {scenario['name']}:")
        print("-" * 40)
        
        # Criar environment
        env = create_mock_env(scenario['trades'], scenario['portfolio'])
        
        # Testar diferentes ações
        actions_to_test = [
            np.array([0.0, 0.0, 0, 0, 0, 0, 0, 0]),  # HOLD
            np.array([0.6, 0.2, 0, 0, 0, 0, 0, 0]),  # BUY  
            np.array([0.0, 0.0, 1, 0, 0, 0, 0, 0]),  # SELL
        ]
        
        scenario_results = {}
        
        for j, action in enumerate(actions_to_test):
            action_name = ['HOLD', 'BUY', 'SELL'][j]
            old_state = {'trades_count': len(scenario['trades']) - 1}
            
            try:
                reward, info, done = reward_calc.calculate_reward_and_info(env, action, old_state)
                components = info.get('reward_components', {})
                
                scenario_results[action_name] = {
                    'reward': reward,
                    'components': components
                }
                
                # Análise dos componentes
                total_component_magnitude = sum(abs(v) for v in components.values())
                pnl_component = abs(components.get('pnl', 0))
                pnl_dominance = (pnl_component / total_component_magnitude) * 100 if total_component_magnitude > 0 else 0
                
                print(f"   {action_name:4}: Reward={reward:>8.4f}, PnL dominance={pnl_dominance:>5.1f}%")
                
                # Detectar problemas de balanceamento
                if scenario['expected_trend'] == 'positive' and reward < 0:
                    balance_issues.append(f"{scenario['name']} ({action_name}): Reward negativo em cenário lucrativo")
                elif scenario['expected_trend'] == 'negative' and reward > 0:
                    balance_issues.append(f"{scenario['name']} ({action_name}): Reward positivo em cenário de prejuízo")
                elif pnl_dominance < 50 and scenario['expected_trend'] != 'micro_farming':
                    balance_issues.append(f"{scenario['name']} ({action_name}): PnL não dominante ({pnl_dominance:.1f}%)")
                
            except Exception as e:
                print(f"   ❌ ERRO em {action_name}: {e}")
                balance_issues.append(f"{scenario['name']} ({action_name}): Erro no cálculo - {str(e)[:50]}")
        
        results.append({
            'scenario': scenario['name'],
            'results': scenario_results,
            'trades_count': len(scenario['trades']),
            'total_pnl': sum(t['pnl_usd'] for t in scenario['trades']),
            'portfolio': scenario['portfolio']
        })
    
    print("\n🔍 ANÁLISE DE BALANCEAMENTO")
    print("=" * 60)
    
    # Análise comparativa
    comparison_tests = [
        {
            'name': 'Profit vs Loss Consistency',
            'test': lambda: compare_profit_vs_loss(results)
        },
        {
            'name': 'Action Invariance',
            'test': lambda: check_action_invariance(results)
        },
        {
            'name': 'Component Balance',
            'test': lambda: analyze_component_balance(results)
        },
        {
            'name': 'Micro vs Quality Reward',
            'test': lambda: compare_micro_vs_quality(results)
        }
    ]
    
    for test_info in comparison_tests:
        print(f"\n📋 {test_info['name']}:")
        try:
            test_result = test_info['test']()
            if isinstance(test_result, list):
                for result_line in test_result:
                    print(f"   {result_line}")
            else:
                print(f"   {test_result}")
        except Exception as e:
            print(f"   ❌ Erro no teste: {e}")
    
    print(f"\n⚠️  PROBLEMAS DE BALANCEAMENTO DETECTADOS:")
    print("=" * 60)
    
    if balance_issues:
        for issue in balance_issues:
            print(f"   🚨 {issue}")
    else:
        print("   ✅ Nenhum problema crítico detectado")
    
    print(f"\n📊 RESUMO:")
    print(f"   Cenários testados: {len(test_scenarios)}")
    print(f"   Problemas detectados: {len(balance_issues)}")
    print(f"   Taxa de sucesso: {((len(test_scenarios)*3 - len(balance_issues))/(len(test_scenarios)*3))*100:.1f}%")
    
    return len(balance_issues) == 0

def compare_profit_vs_loss(results):
    """Comparar consistência entre cenários de lucro e prejuízo"""
    profit_scenario = next((r for r in results if 'Pure Profit' in r['scenario']), None)
    loss_scenario = next((r for r in results if 'Pure Loss' in r['scenario']), None)
    
    if not profit_scenario or not loss_scenario:
        return "❌ Cenários de comparação não encontrados"
    
    issues = []
    
    for action in ['HOLD', 'BUY', 'SELL']:
        if action in profit_scenario['results'] and action in loss_scenario['results']:
            profit_reward = profit_scenario['results'][action]['reward']
            loss_reward = loss_scenario['results'][action]['reward']
            
            if profit_reward <= loss_reward:
                issues.append(f"❌ {action}: Profit reward ({profit_reward:.4f}) ≤ Loss reward ({loss_reward:.4f})")
            else:
                issues.append(f"✅ {action}: Profit reward > Loss reward (Δ={profit_reward-loss_reward:.4f})")
    
    return issues

def check_action_invariance(results):
    """Verificar se diferentes ações dão rewards similares (não deveria)"""
    issues = []
    
    for result in results:
        scenario_name = result['scenario']
        scenario_results = result['results']
        
        if len(scenario_results) >= 3:
            rewards = [scenario_results[action]['reward'] for action in ['HOLD', 'BUY', 'SELL'] if action in scenario_results]
            
            if len(rewards) >= 2:
                reward_variance = np.var(rewards)
                reward_range = max(rewards) - min(rewards)
                
                if reward_variance < 1e-6 and reward_range < 0.001:
                    issues.append(f"⚠️ {scenario_name}: Actions têm rewards muito similares (var={reward_variance:.2e})")
                else:
                    issues.append(f"✅ {scenario_name}: Actions têm rewards distintos (range={reward_range:.4f})")
    
    return issues

def analyze_component_balance(results):
    """Analisar balanceamento dos componentes"""
    issues = []
    
    for result in results:
        scenario_name = result['scenario']
        
        # Pegar componentes da ação BUY (mais representativa)
        if 'BUY' in result['results']:
            components = result['results']['BUY']['components']
            
            total_magnitude = sum(abs(v) for v in components.values())
            if total_magnitude > 0:
                component_percentages = {k: (abs(v) / total_magnitude) * 100 for k, v in components.items()}
                
                # Verificar dominância do PnL
                pnl_dominance = component_percentages.get('pnl', 0)
                
                if pnl_dominance < 40:
                    issues.append(f"⚠️ {scenario_name}: PnL não dominante ({pnl_dominance:.1f}%)")
                else:
                    issues.append(f"✅ {scenario_name}: PnL dominante ({pnl_dominance:.1f}%)")
                
                # Listar top 3 componentes
                top_components = sorted(component_percentages.items(), key=lambda x: x[1], reverse=True)[:3]
                top_3_str = ", ".join([f"{k}:{v:.1f}%" for k, v in top_components])
                issues.append(f"   Top 3: {top_3_str}")
    
    return issues

def compare_micro_vs_quality(results):
    """Comparar micro-trading vs quality trading"""
    micro_result = next((r for r in results if 'Micro-trading' in r['scenario']), None)
    quality_result = next((r for r in results if 'Quality trading' in r['scenario']), None)
    
    if not micro_result or not quality_result:
        return "❌ Cenários de micro vs quality não encontrados"
    
    issues = []
    
    # Comparar reward per dollar
    micro_pnl = micro_result['total_pnl']
    quality_pnl = quality_result['total_pnl'] 
    
    if 'BUY' in micro_result['results'] and 'BUY' in quality_result['results']:
        micro_reward = micro_result['results']['BUY']['reward']
        quality_reward = quality_result['results']['BUY']['reward']
        
        micro_ratio = micro_reward / micro_pnl if micro_pnl != 0 else 0
        quality_ratio = quality_reward / quality_pnl if quality_pnl != 0 else 0
        
        if quality_ratio > micro_ratio:
            ratio_improvement = quality_ratio / micro_ratio if micro_ratio != 0 else float('inf')
            issues.append(f"✅ Quality trading é {ratio_improvement:.1f}x melhor que micro-trading")
        else:
            issues.append(f"❌ Micro-trading tem melhor reward/dollar que quality trading")
        
        issues.append(f"   Micro: {micro_ratio:.6f} R/$, Quality: {quality_ratio:.6f} R/$")
        issues.append(f"   Micro: {micro_result['trades_count']} trades, Quality: {quality_result['trades_count']} trades")
    
    return issues

if __name__ == "__main__":
    success = test_reward_balance()
    print(f"\n🏆 RESULTADO FINAL: {'✅ APROVADO' if success else '❌ PROBLEMAS DETECTADOS'}")
    exit(0 if success else 1)