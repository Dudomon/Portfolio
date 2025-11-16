#!/usr/bin/env python3
"""
🎯 TESTE ESPECÍFICO: Reward System durante treinamento simulado
Simular exatamente o que acontece durante o treinamento do daytrader
"""

import sys
sys.path.append("D:/Projeto")
import numpy as np
from typing import Dict

def test_reward_system_training_simulation():
    print("🎯 SIMULAÇÃO DE TREINAMENTO - REWARD SYSTEM V3.0")
    print("=" * 60)
    
    # 1. IMPORTAR O REWARD SYSTEM EXATAMENTE COMO O DAYTRADER FAZ
    print("1️⃣ IMPORTANDO REWARD SYSTEM COMO O DAYTRADER...")
    
    try:
        from trading_framework.rewards.reward_daytrade_v2 import create_balanced_daytrading_reward_system
        
        # Criar exatamente como o daytrader cria
        initial_balance = 1000.0
        reward_system = create_balanced_daytrading_reward_system(initial_balance)
        
        print(f"   ✅ Reward system criado: {type(reward_system).__name__}")
        print(f"   ✅ Initial balance: ${initial_balance}")
        
    except Exception as e:
        print(f"   ❌ Falha na importação: {e}")
        return False
    
    # 2. SIMULAR ENVIRONMENT DE TREINAMENTO
    print("\n2️⃣ SIMULANDO ENVIRONMENT DE TREINAMENTO...")
    
    class MockTradingEnv:
        """Mock do TradingEnv usado pelo daytrader"""
        def __init__(self):
            self.trades = []
            self.current_step = 0
            self.balance = 1000.0
            self.realized_balance = 1000.0
            self.portfolio_value = 1000.0
            self.initial_balance = 1000.0
            self.peak_portfolio = 1000.0
            self.current_drawdown = 0.0
            self.max_drawdown = 0.0
            self.current_positions = 0
            self.reward_history_size = 100
            self.recent_rewards = []
            
        def add_trade(self, pnl_usd, duration_steps=10):
            """Simular um trade completo"""
            trade = {
                'pnl_usd': pnl_usd,
                'pnl': pnl_usd,
                'duration_steps': duration_steps,
                'position_size': 0.02,
                'entry_price': 2000.0,
                'exit_price': 2000.0 + pnl_usd,
                'side': 'long' if pnl_usd > 0 else 'short',
                'exit_reason': 'manual'
            }
            self.trades.append(trade)
            self.portfolio_value += pnl_usd
            self.realized_balance += pnl_usd
            
            # Update drawdown
            if self.portfolio_value > self.peak_portfolio:
                self.peak_portfolio = self.portfolio_value
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.peak_portfolio - self.portfolio_value) / self.peak_portfolio
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
                
            self.current_step += duration_steps
            
    env = MockTradingEnv()
    print("   ✅ Mock environment criado")
    
    # 3. SIMULAR SEQUÊNCIA DE TREINAMENTO REALISTA
    print("\n3️⃣ SIMULANDO SEQUÊNCIA DE TREINAMENTO...")
    
    training_scenarios = [
        # Cenário 1: Início do treinamento - trades aleatórios
        {
            'name': 'Início do treino (random)',
            'trades': [
                {'pnl': 15, 'duration': 8},
                {'pnl': -5, 'duration': 3},
                {'pnl': -12, 'duration': 15},
                {'pnl': 8, 'duration': 5}
            ]
        },
        
        # Cenário 2: Modelo aprendendo - alguns bons trades
        {
            'name': 'Modelo aprendendo',
            'trades': [
                {'pnl': 35, 'duration': 12},
                {'pnl': -8, 'duration': 4},
                {'pnl': 42, 'duration': 20},
                {'pnl': 18, 'duration': 7},
                {'pnl': -15, 'duration': 6}
            ]
        },
        
        # Cenário 3: Micro-farming (problema que queremos evitar)
        {
            'name': 'Micro-farming (deveria ser penalizado)',
            'trades': [
                {'pnl': 2, 'duration': 2} for _ in range(25)  # 25 trades de $2
            ]
        },
        
        # Cenário 4: Quality trading (o que queremos)
        {
            'name': 'Quality trading (deveria ser premiado)',
            'trades': [
                {'pnl': 65, 'duration': 25},
                {'pnl': 48, 'duration': 18},
                {'pnl': 55, 'duration': 22},
                {'pnl': -20, 'duration': 8},
                {'pnl': 72, 'duration': 30}
            ]
        }
    ]
    
    results = []
    
    for scenario in training_scenarios:
        print(f"\n   🧪 {scenario['name']}:")
        
        # Reset environment
        test_env = MockTradingEnv()
        
        # Executar trades do cenário
        for trade_data in scenario['trades']:
            test_env.add_trade(trade_data['pnl'], trade_data['duration'])
        
        # Calcular reward como seria no treinamento
        action = np.array([0.6, 0.2, 0, 0, 0, 0, 0, 0])  # Ação típica
        old_state = {'trades_count': len(test_env.trades) - 1}
        
        try:
            reward, info, done = reward_system.calculate_reward_and_info(test_env, action, old_state)
            components = info.get('reward_components', {})
            
            total_pnl = sum(t['pnl'] for t in scenario['trades'])
            reward_per_dollar = reward / total_pnl if total_pnl > 0 else 0
            
            result = {
                'scenario': scenario['name'],
                'trades_count': len(scenario['trades']),
                'total_pnl': total_pnl,
                'reward': reward,
                'reward_per_dollar': reward_per_dollar,
                'components': components
            }
            results.append(result)
            
            print(f"     Trades: {len(scenario['trades'])}")
            print(f"     Total PnL: ${total_pnl:.1f}")
            print(f"     Reward: {reward:.6f}")
            print(f"     R/$ ratio: {reward_per_dollar:.8f}")
            
            # Mostrar top 3 componentes
            sorted_components = sorted(components.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            print(f"     Top components:")
            for comp_name, comp_value in sorted_components:
                if abs(comp_value) > 0.001:
                    print(f"       {comp_name}: {comp_value:.6f}")
                    
        except Exception as e:
            print(f"     ❌ Erro no cálculo de reward: {e}")
            return False
    
    # 4. ANÁLISE DOS RESULTADOS
    print(f"\n4️⃣ ANÁLISE DOS RESULTADOS DE TREINAMENTO:")
    print("=" * 50)
    
    if len(results) >= 4:
        micro_farming = results[2]  # Cenário 3
        quality_trading = results[3]  # Cenário 4
        
        # Verificar se quality trading é melhor que micro-farming
        quality_reward_ratio = quality_trading['reward_per_dollar']
        micro_reward_ratio = micro_farming['reward_per_dollar']
        
        ratio_improvement = quality_reward_ratio / micro_reward_ratio if micro_reward_ratio > 0 else float('inf')
        
        print(f"   📊 COMPARAÇÃO CRÍTICA:")
        print(f"     Micro-farming R/$:  {micro_reward_ratio:.8f}")
        print(f"     Quality trading R/$: {quality_reward_ratio:.8f}")
        print(f"     Quality é {ratio_improvement:.0f}x melhor")
        
        # Validações críticas
        validations = []
        
        if ratio_improvement > 50:
            validations.append("✅ Quality trading significativamente melhor")
        else:
            validations.append(f"❌ Quality trading só {ratio_improvement:.1f}x melhor (deveria ser >50x)")
        
        # Verificar se micro-farming tem activity bonus = 0
        micro_activity = micro_farming['components'].get('activity_bonus', -999)
        if micro_activity == 0.0:
            validations.append("✅ Micro-farming sem activity bonus")
        else:
            validations.append(f"❌ Micro-farming ainda tem activity bonus: {micro_activity}")
        
        # Verificar se PnL é dominante em quality trading
        quality_components = quality_trading['components']
        total_abs = sum(abs(v) for v in quality_components.values() if abs(v) > 0.001)
        pnl_contribution = abs(quality_components.get('pnl', 0))
        pnl_percentage = (pnl_contribution / total_abs * 100) if total_abs > 0 else 0
        
        if pnl_percentage > 50:
            validations.append(f"✅ PnL dominante no quality trading ({pnl_percentage:.1f}%)")
        else:
            validations.append(f"❌ PnL não dominante ({pnl_percentage:.1f}%)")
        
        print(f"\n   🎯 VALIDAÇÕES:")
        for validation in validations:
            print(f"     {validation}")
        
        success_count = len([v for v in validations if v.startswith("✅")])
        print(f"\n   SUCCESS RATE: {success_count}/{len(validations)} ({success_count/len(validations)*100:.0f}%)")
        
        return success_count >= 2  # Pelo menos 2/3 validações devem passar
    
    return False

def main():
    print("🧪 TESTE DE INTEGRAÇÃO REWARD SYSTEM NO TREINAMENTO")
    print("=" * 70)
    
    success = test_reward_system_training_simulation()
    
    print(f"\n🏆 RESULTADO FINAL:")
    if success:
        print("   🟢 REWARD SYSTEM FUNCIONANDO CORRETAMENTE NO TREINAMENTO")
        print("   ✅ Daytrader pode treinar com segurança usando o reward V3.0")
    else:
        print("   🔴 PROBLEMAS DETECTADOS NO REWARD SYSTEM")
        print("   ❌ Corrigir problemas antes do treinamento")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)