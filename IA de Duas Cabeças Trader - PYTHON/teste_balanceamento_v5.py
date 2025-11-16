"""
🎯 TESTE DE BALANCEAMENTO V5 SHARPE REWARD SYSTEM
Verifica se os componentes do reward V5 estão balanceados corretamente
"""

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import os

# Add path para imports
sys.path.append('.')

from trading_framework.rewards import create_reward_system

def simular_trading_scenarios():
    """Simula diferentes cenários de trading para testar balanceamento"""
    
    print("🎯 TESTE DE BALANCEAMENTO V5 SHARPE REWARD SYSTEM")
    print("=" * 60)
    
    # Criar reward system V5
    reward_system = create_reward_system("v5_sharpe", initial_balance=1000.0)
    
    if reward_system is None:
        print("❌ ERRO: Não foi possível criar V5 reward system")
        return
    
    print("✅ V5 Reward System criado com sucesso")
    print(f"📊 Configuração:")
    print(f"   Sharpe Weight: {reward_system.sharpe_weight}")
    print(f"   Risk Weight: {reward_system.risk_weight}")
    print(f"   Volatility Weight: {reward_system.volatility_weight}")
    print(f"   Target Volatility: {reward_system.target_volatility}")
    print()
    
    # Simular diferentes cenários de trading
    scenarios = {
        "🟢 TRADING PERFEITO": {
            "returns": [0.02, 0.015, 0.01, 0.025, 0.03, 0.02, 0.01, 0.015, 0.02, 0.025] * 3,  # 30 returns consistentes
            "description": "Returns consistentes e positivos - Alto Sharpe esperado"
        },
        "🔴 TRADING ERRÁTICO": {
            "returns": [0.05, -0.08, 0.12, -0.15, 0.08, -0.05, 0.2, -0.18, 0.1, -0.12] * 3,  # 30 returns voláteis
            "description": "Returns muito voláteis - Baixo Sharpe esperado"
        },
        "🟡 TRADING MEDIANO": {
            "returns": [0.01, 0.0, -0.005, 0.015, 0.008, -0.002, 0.012, 0.003, -0.001, 0.01] * 3,  # 30 returns medianos
            "description": "Returns pequenos e mistos - Sharpe mediano esperado"
        },
        "🔵 GRANDE GANHO FINAL": {
            "returns": [0.001] * 25 + [0.15, 0.12, 0.08, 0.1, 0.05],  # 30 returns com grande ganho no final
            "description": "Pequenos gains + grande ganho final"
        }
    }
    
    results = {}
    
    for scenario_name, scenario_data in scenarios.items():
        print(f"\n{scenario_name}")
        print("-" * 40)
        print(f"📝 {scenario_data['description']}")
        
        # Reset reward system para novo cenário
        reward_system.reset()
        
        rewards = []
        portfolio_values = []
        sharpe_components = []
        risk_components = []
        volatility_components = []
        
        current_portfolio = 1000.0
        
        for i, return_pct in enumerate(scenario_data['returns']):
            # Simular mudança no portfolio
            current_portfolio *= (1 + return_pct)
            
            # Mock trading info
            info = {
                'portfolio_value': current_portfolio,
                'total_pnl': current_portfolio - 1000.0,
                'unrealized_pnl': 0.0,
                'position': 1 if return_pct > 0 else -1 if return_pct < 0 else 0,
                'action': 1 if return_pct > 0 else 2 if return_pct < 0 else 0,
                'price_change': return_pct,
                'volatility': abs(return_pct),
                'step': i
            }
            
            # Mock environment para V5 
            class MockEnv:
                def __init__(self, portfolio_value, total_pnl, unrealized_pnl):
                    self.portfolio_value = portfolio_value
                    self.total_realized_pnl = total_pnl
                    self.total_unrealized_pnl = unrealized_pnl
                    self.peak_portfolio_value = max(1000.0, portfolio_value)
                    self.positions = []
                    self.trades = []
                    self.current_step = i
            
            mock_env = MockEnv(current_portfolio, current_portfolio - 1000.0, 0.0)
            
            # Calcular reward usando V5 method
            reward, reward_info, done = reward_system.calculate_reward_and_info(
                mock_env, 
                np.array([0.5]),  # mock action
                {}  # mock old_state
            )
            
            rewards.append(reward)
            portfolio_values.append(current_portfolio)
            
            # Capturar componentes internos se disponível
            if hasattr(reward_system, 'last_sharpe_component'):
                sharpe_components.append(reward_system.last_sharpe_component)
            if hasattr(reward_system, 'last_risk_component'):
                risk_components.append(reward_system.last_risk_component)
            if hasattr(reward_system, 'last_volatility_component'):
                volatility_components.append(reward_system.last_volatility_component)
        
        # Calcular métricas finais
        total_return = (current_portfolio - 1000.0) / 1000.0 * 100
        returns_array = np.array(scenario_data['returns'])
        actual_sharpe = np.mean(returns_array) / np.std(returns_array) if np.std(returns_array) > 0 else 0
        avg_reward = np.mean(rewards)
        reward_std = np.std(rewards)
        
        results[scenario_name] = {
            'total_return': total_return,
            'actual_sharpe': actual_sharpe,
            'avg_reward': avg_reward,
            'reward_std': reward_std,
            'final_portfolio': current_portfolio,
            'rewards': rewards,
            'portfolio_values': portfolio_values
        }
        
        print(f"📈 Return Total: {total_return:.2f}%")
        print(f"📊 Sharpe Real: {actual_sharpe:.3f}")
        print(f"🎯 Reward Médio: {avg_reward:.4f}")
        print(f"📉 Reward StdDev: {reward_std:.4f}")
        print(f"💰 Portfolio Final: ${current_portfolio:.2f}")
        
        if sharpe_components:
            print(f"🎯 Componente Sharpe Médio: {np.mean(sharpe_components):.4f}")
        if risk_components:
            print(f"⚠️  Componente Risk Médio: {np.mean(risk_components):.4f}")
        if volatility_components:
            print(f"📊 Componente Volatility Médio: {np.mean(volatility_components):.4f}")
    
    # Análise comparativa
    print("\n" + "="*60)
    print("🔍 ANÁLISE COMPARATIVA DE BALANCEAMENTO")
    print("="*60)
    
    # Verificar se rewards estão correlacionados com Sharpe
    sharpe_values = [results[scenario]['actual_sharpe'] for scenario in results]
    reward_values = [results[scenario]['avg_reward'] for scenario in results]
    
    correlation = np.corrcoef(sharpe_values, reward_values)[0, 1]
    print(f"🔗 Correlação Sharpe vs Reward: {correlation:.3f}")
    
    if correlation > 0.5:
        print("✅ EXCELENTE: Rewards bem correlacionados com Sharpe")
    elif correlation > 0.2:
        print("🟡 RAZOÁVEL: Correlação moderada com Sharpe")
    else:
        print("❌ PROBLEMA: Rewards mal correlacionados com Sharpe")
    
    # Verificar se trading perfeito tem melhor reward
    perfect_reward = results["🟢 TRADING PERFEITO"]['avg_reward']
    erratic_reward = results["🔴 TRADING ERRÁTICO"]['avg_reward']
    
    print(f"\n📊 TRADING PERFEITO reward: {perfect_reward:.4f}")
    print(f"📊 TRADING ERRÁTICO reward: {erratic_reward:.4f}")
    
    if perfect_reward > erratic_reward:
        print("✅ CORRETO: Trading consistente recompensado mais que errático")
    else:
        print("❌ PROBLEMA: Trading errático sendo mais recompensado")
    
    # Verificar se rewards não são extremos
    all_rewards = []
    for scenario in results:
        all_rewards.extend(results[scenario]['rewards'])
    
    reward_range = max(all_rewards) - min(all_rewards)
    print(f"\n📈 Range de Rewards: {reward_range:.4f}")
    print(f"📈 Reward Mínimo: {min(all_rewards):.4f}")
    print(f"📈 Reward Máximo: {max(all_rewards):.4f}")
    
    if reward_range < 10:
        print("✅ BOM: Range de rewards controlado")
    else:
        print("⚠️  ATENÇÃO: Range de rewards muito amplo")
    
    # Salvar resultados detalhados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"teste_balanceamento_v5_{timestamp}.txt", "w", encoding='utf-8') as f:
        f.write("🎯 TESTE DE BALANCEAMENTO V5 SHARPE REWARD SYSTEM\n")
        f.write("=" * 60 + "\n\n")
        
        for scenario_name, result in results.items():
            f.write(f"{scenario_name}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Return: {result['total_return']:.2f}%\n")
            f.write(f"Actual Sharpe: {result['actual_sharpe']:.3f}\n")
            f.write(f"Avg Reward: {result['avg_reward']:.4f}\n")
            f.write(f"Reward StdDev: {result['reward_std']:.4f}\n")
            f.write(f"Final Portfolio: ${result['final_portfolio']:.2f}\n\n")
        
        f.write(f"Correlação Sharpe vs Reward: {correlation:.3f}\n")
        f.write(f"Range de Rewards: {reward_range:.4f}\n")
    
    print(f"\n💾 Resultados salvos em: teste_balanceamento_v5_{timestamp}.txt")
    print("\n🎯 TESTE DE BALANCEAMENTO CONCLUÍDO!")
    
    return results

def test_reward_components():
    """Testa componentes individuais do reward system"""
    
    print("\n🔧 TESTE DE COMPONENTES INDIVIDUAIS")
    print("="*50)
    
    reward_system = create_reward_system("v5_sharpe", initial_balance=1000.0)
    
    if not reward_system:
        print("❌ Não foi possível criar reward system para teste de componentes")
        return
    
    # Teste com cenário controlado
    test_scenarios = [
        {"returns": [0.01] * 30, "name": "Returns Estáveis"},
        {"returns": [0.05, -0.05] * 15, "name": "Returns Alternados"},
        {"returns": [0.0] * 30, "name": "Returns Zero"},
        {"returns": np.random.normal(0.01, 0.02, 30), "name": "Returns Aleatórios"}
    ]
    
    for scenario in test_scenarios:
        print(f"\n🧪 {scenario['name']}")
        print("-" * 30)
        
        reward_system.reset()
        rewards = []
        portfolio = 1000.0
        
        for i, ret in enumerate(scenario['returns']):
            portfolio *= (1 + ret)
            
            info = {
                'portfolio_value': portfolio,
                'total_pnl': portfolio - 1000.0,
                'unrealized_pnl': 0.0,
                'position': 1 if ret > 0 else -1 if ret < 0 else 0,
                'action': 1 if ret > 0 else 2 if ret < 0 else 0,
                'price_change': ret,
                'volatility': abs(ret),
                'step': i
            }
            
            # Mock environment para V5
            class MockEnvComp:
                def __init__(self, portfolio_value, total_pnl, unrealized_pnl):
                    self.portfolio_value = portfolio_value
                    self.total_realized_pnl = total_pnl
                    self.total_unrealized_pnl = unrealized_pnl
                    self.peak_portfolio_value = max(1000.0, portfolio_value)
                    self.positions = []
                    self.trades = []
                    self.current_step = i
            
            mock_env_comp = MockEnvComp(portfolio, portfolio - 1000.0, 0.0)
            
            reward, reward_info, done = reward_system.calculate_reward_and_info(
                mock_env_comp, 
                np.array([0.5]),  # mock action
                {}  # mock old_state
            )
            rewards.append(reward)
        
        print(f"Reward Médio: {np.mean(rewards):.4f}")
        print(f"Reward StdDev: {np.std(rewards):.4f}")
        print(f"Portfolio Final: ${portfolio:.2f}")

if __name__ == "__main__":
    try:
        results = simular_trading_scenarios()
        test_reward_components()
    except Exception as e:
        print(f"❌ ERRO durante teste: {e}")
        import traceback
        traceback.print_exc()