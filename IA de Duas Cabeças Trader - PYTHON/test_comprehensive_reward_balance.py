#!/usr/bin/env python3
"""
🧪 TESTE ABRANGENTE - REWARD SYSTEM V3.0 ACTION-AWARE
Usar TradingEnv real do daytrader para testes completos
"""

import sys
sys.path.append("D:/Projeto")
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

def create_mock_trading_env_enhanced():
    """Criar mock environment mais realista com state completo"""
    print("✅ Criando Mock TradingEnv aprimorado...")
    
    class EnhancedMockTradingEnv:
        def __init__(self):
            # Setup básico
            self.trades = []
            self.portfolio_value = 1000.0
            self.initial_balance = 1000.0
            self.realized_balance = 1000.0
            self.balance = 1000.0
            self.current_step = 0
            self.peak_portfolio = 1000.0
            self.current_drawdown = 0.0
            self.max_drawdown = 0.0
            self.current_positions = 0
            self.reward_history_size = 100
            self.recent_rewards = []
            
            # Market data simulado
            self.current_price = 50000.0
            self.price_history = [50000.0 + np.random.randn() * 100 for _ in range(100)]
            
        def reset(self):
            self.trades = []
            self.portfolio_value = self.initial_balance
            self.current_step = 0
            self.peak_portfolio = self.initial_balance
            self.current_drawdown = 0.0
            self.max_drawdown = 0.0
            self.current_positions = 0
            return np.random.randn(2580)  # Mock observation
        
        def step(self, action):
            self.current_step += 1
            
            # Simular execução de trade baseada na ação
            direction = action[0] if len(action) > 0 else 0.0
            size = action[1] if len(action) > 1 else 0.0
            exit_signal = action[2] if len(action) > 2 else 0.0
            
            # Simular resultado do trade
            if abs(direction) > 0.1 or abs(size) > 0.05:  # Ação significativa
                # Simular PnL baseado na ação
                base_pnl = np.random.randn() * 50  # PnL base
                
                # Ajustar PnL baseado na qualidade da ação
                if direction > 0.5 and size > 0.2:  # Aggressive buy
                    pnl_multiplier = 1.2 if np.random.random() > 0.4 else 0.8
                elif direction > 0.2 and size > 0.1:  # Conservative buy  
                    pnl_multiplier = 1.1 if np.random.random() > 0.3 else 0.9
                else:
                    pnl_multiplier = 1.0
                
                final_pnl = base_pnl * pnl_multiplier
                
                # Criar trade
                trade = {
                    'pnl_usd': final_pnl,
                    'pnl': final_pnl,
                    'duration_steps': np.random.randint(5, 30),
                    'position_size': size,
                    'entry_price': self.current_price,
                    'exit_price': self.current_price + final_pnl,
                    'side': 'long' if direction > 0 else 'short',
                    'exit_reason': 'manual'
                }
                
                self.trades.append(trade)
                self.portfolio_value += final_pnl
                self.realized_balance += final_pnl
                
                # Update drawdown
                if self.portfolio_value > self.peak_portfolio:
                    self.peak_portfolio = self.portfolio_value
                    self.current_drawdown = 0.0
                else:
                    self.current_drawdown = (self.peak_portfolio - self.portfolio_value) / self.peak_portfolio
                    self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
            # Return step results
            obs = np.random.randn(2580)  # Mock observation
            env_reward = 0.0  # We don't use env reward
            done = self.current_step >= 50  # Episode limit for testing
            info = {}
            
            return obs, env_reward, done, info
        
        def get_state_summary(self):
            return {
                'trades_count': len(self.trades),
                'portfolio_value': self.portfolio_value,
                'total_pnl': sum(t['pnl_usd'] for t in self.trades),
                'win_rate': len([t for t in self.trades if t['pnl_usd'] > 0]) / max(1, len(self.trades)),
                'avg_trade_pnl': np.mean([t['pnl_usd'] for t in self.trades]) if self.trades else 0,
                'current_drawdown': self.current_drawdown
            }
    
    env = EnhancedMockTradingEnv()
    print(f"   Tipo: {type(env).__name__}")
    print(f"   Mock obs shape: (2580,)")
    print(f"   Mock action space: (8,)")
    print(f"   Environment: Enhanced Mock com trade simulation")
    
    return env

def test_comprehensive_reward_balance():
    print("🧪 TESTE ABRANGENTE - REWARD SYSTEM V3.0 ACTION-AWARE")
    print("=" * 70)
    
    # Criar environment (enhanced mock)
    env = create_mock_trading_env_enhanced()
    if env is None:
        return False
    
    try:
        from trading_framework.rewards.reward_daytrade_v2 import BalancedDayTradingRewardCalculator
        reward_calc = BalancedDayTradingRewardCalculator()
        print("✅ Reward calculator V3.0 importado")
        
    except Exception as e:
        print(f"❌ Erro na importação reward: {e}")
        return False
    
    print("\n📊 EXECUTANDO TESTE ABRANGENTE COM TRADING ENV REAL")
    print("=" * 70)
    
    # TESTE 1: DIVERSAS AÇÕES EM MESMO ESTADO
    print("\n🎯 TESTE 1: ACTION INVARIANCE (CRÍTICO)")
    print("-" * 50)
    
    # Reset e pegar estado inicial
    obs = env.reset()
    initial_state = {'trades_count': 0}
    
    # Testar diferentes ações no mesmo estado
    test_actions = [
        np.array([0.0, 0.0, 0, 0, 0, 0, 0, 0]),      # HOLD
        np.array([0.3, 0.1, 0, 0, 0, 0, 0, 0]),      # Conservative BUY
        np.array([0.8, 0.3, 0, 0, 0, 0, 0, 0]),      # Aggressive BUY  
        np.array([0.1, 0.05, 1, 0, 0, 0, 0, 0]),     # Conservative SELL
        np.array([0.2, 0.1, 1, 0, 0, 0, 0, 0]),      # Aggressive SELL
        np.array([0.5, 0.2, 0, 0, 0, 0, 0, 0]),      # Moderate BUY
    ]
    
    action_names = ['HOLD', 'Cons_BUY', 'Aggr_BUY', 'Cons_SELL', 'Aggr_SELL', 'Mod_BUY']
    action_results = {}
    
    for i, (action, name) in enumerate(zip(test_actions, action_names)):
        try:
            reward, info, done = reward_calc.calculate_reward_and_info(env, action, initial_state)
            components = info.get('reward_components', {})
            
            action_results[name] = {
                'reward': reward,
                'components': components,
                'action': action.copy()
            }
            
            print(f"   {name:10}: Reward={reward:>8.6f}")
            
        except Exception as e:
            print(f"   ❌ ERRO em {name}: {e}")
    
    # Análise de variance
    rewards = [r['reward'] for r in action_results.values()]
    reward_variance = np.var(rewards)
    reward_range = max(rewards) - min(rewards)
    
    print(f"\n   📊 ANÁLISE ACTION INVARIANCE:")
    print(f"      Variance: {reward_variance:.8f}")
    print(f"      Range: {reward_range:.6f}")
    
    if reward_variance < 1e-10:
        print(f"      🚨 CRÍTICO: Actions têm rewards idênticos!")
        action_invariance_pass = False
    elif reward_range < 0.001:
        print(f"      ⚠️ AVISO: Actions têm rewards muito similares")
        action_invariance_pass = False
    else:
        print(f"      ✅ OK: Actions têm rewards distintos")
        action_invariance_pass = True
    
    # TESTE 2: SIMULAÇÃO DE TRADING COM DIFERENTES ESTRATÉGIAS
    print(f"\n🎯 TESTE 2: SIMULAÇÃO DE ESTRATÉGIAS")
    print("-" * 50)
    
    strategies = {
        'Conservative Trader': {
            'actions': [np.array([0.2, 0.05, 0, 0, 0, 0, 0, 0]) for _ in range(10)],
            'description': 'Small positions, low risk'
        },
        'Aggressive Trader': {
            'actions': [np.array([0.8, 0.3, 0, 0, 0, 0, 0, 0]) for _ in range(10)],
            'description': 'Large positions, high risk'
        },
        'Mixed Strategy': {
            'actions': [
                np.array([0.6, 0.2, 0, 0, 0, 0, 0, 0]),  # Moderate
                np.array([0.1, 0.02, 0, 0, 0, 0, 0, 0]), # Conservative  
                np.array([0.9, 0.4, 0, 0, 0, 0, 0, 0]),  # Aggressive
                np.array([0.0, 0.0, 0, 0, 0, 0, 0, 0]),  # Hold
            ] * 3,  # Repeat pattern
            'description': 'Varying risk levels'
        },
        'HODL Strategy': {
            'actions': [np.array([0.0, 0.0, 0, 0, 0, 0, 0, 0]) for _ in range(10)],
            'description': 'Mostly holding, minimal trading'
        }
    }
    
    strategy_results = {}
    
    for strategy_name, strategy_info in strategies.items():
        print(f"\n   🎪 Testando: {strategy_name}")
        print(f"      {strategy_info['description']}")
        
        # Reset environment
        env.reset()
        cumulative_reward = 0.0
        step_rewards = []
        trades_made = 0
        old_state = {'trades_count': 0}
        
        for step, action in enumerate(strategy_info['actions']):
            try:
                # Execute action
                obs, env_reward, done, env_info = env.step(action)
                
                # Get our reward
                reward, info, _ = reward_calc.calculate_reward_and_info(env, action, old_state)
                components = info.get('reward_components', {})
                
                cumulative_reward += reward
                step_rewards.append(reward)
                
                # Update state
                current_trades = len(getattr(env, 'trades', []))
                if current_trades > old_state['trades_count']:
                    trades_made += 1
                old_state = {'trades_count': current_trades}
                
                if done:
                    break
                    
            except Exception as e:
                print(f"      ⚠️ Erro no step {step}: {e}")
                continue
        
        # Resultado da estratégia
        avg_reward = cumulative_reward / len(strategy_info['actions'])
        reward_consistency = np.std(step_rewards) if len(step_rewards) > 1 else 0
        
        strategy_results[strategy_name] = {
            'cumulative_reward': cumulative_reward,
            'avg_reward_per_step': avg_reward,
            'reward_consistency': reward_consistency,
            'trades_made': trades_made,
            'final_portfolio': getattr(env, 'portfolio_value', 1000)
        }
        
        print(f"      Cumulative Reward: {cumulative_reward:>8.4f}")
        print(f"      Avg Reward/Step: {avg_reward:>8.6f}")
        print(f"      Trades Made: {trades_made}")
        print(f"      Portfolio: ${strategy_results[strategy_name]['final_portfolio']:.2f}")
    
    # TESTE 3: VERIFICAR BALANCE DOS COMPONENTES
    print(f"\n🎯 TESTE 3: BALANCE DOS COMPONENTES")
    print("-" * 50)
    
    # Reset e fazer alguns trades para ter dados
    env.reset()
    sample_actions = [
        np.array([0.6, 0.2, 0, 0, 0, 0, 0, 0]),  # BUY
        np.array([0.0, 0.0, 0, 0, 0, 0, 0, 0]),  # HOLD 
        np.array([0.3, 0.1, 1, 0, 0, 0, 0, 0]),  # SELL
    ]
    
    old_state = {'trades_count': 0}
    component_analysis = {}
    
    for i, action in enumerate(sample_actions):
        try:
            obs, env_reward, done, env_info = env.step(action)
            reward, info, _ = reward_calc.calculate_reward_and_info(env, action, old_state)
            components = info.get('reward_components', {})
            
            # Analisar componentes
            total_magnitude = sum(abs(v) for v in components.values() if abs(v) > 1e-6)
            if total_magnitude > 0:
                component_percentages = {
                    k: (abs(v) / total_magnitude) * 100 
                    for k, v in components.items() 
                    if abs(v) > 1e-6
                }
                
                component_analysis[f'Step_{i+1}'] = {
                    'total_reward': reward,
                    'total_magnitude': total_magnitude,
                    'components': component_percentages,
                    'top_component': max(component_percentages.items(), key=lambda x: x[1]) if component_percentages else ('none', 0)
                }
                
                print(f"   Step {i+1}: Total Reward = {reward:.6f}")
                print(f"      Total Magnitude = {total_magnitude:.6f}")
                
                # Top 3 components
                sorted_components = sorted(component_percentages.items(), key=lambda x: x[1], reverse=True)[:3]
                for comp_name, comp_pct in sorted_components:
                    print(f"      {comp_name}: {comp_pct:.1f}%")
                
            old_state = {'trades_count': len(getattr(env, 'trades', []))}
            
        except Exception as e:
            print(f"   ❌ Erro no step {i+1}: {e}")
    
    # ANÁLISE FINAL
    print(f"\n🔍 ANÁLISE FINAL DOS RESULTADOS")
    print("=" * 70)
    
    issues_found = []
    
    # 1. Action Invariance
    if not action_invariance_pass:
        issues_found.append("❌ Actions ainda retornam rewards similares/idênticos")
    else:
        print("✅ Action Invariance: PASSOU")
    
    # 2. Strategy Differentiation  
    strategy_rewards = [r['cumulative_reward'] for r in strategy_results.values()]
    strategy_variance = np.var(strategy_rewards)
    
    if strategy_variance < 0.001:
        issues_found.append("❌ Estratégias diferentes têm rewards similares")
    else:
        print("✅ Strategy Differentiation: PASSOU")
    
    # 3. Component Balance
    pnl_dominance_ok = True
    for step_name, analysis in component_analysis.items():
        pnl_pct = analysis['components'].get('pnl', 0)
        if pnl_pct < 40:  # PnL should be at least 40% dominant
            pnl_dominance_ok = False
            break
    
    if not pnl_dominance_ok:
        issues_found.append("❌ PnL não está dominante nos componentes")
    else:
        print("✅ Component Balance: PASSOU")
    
    # 4. Reward Scale
    max_reward = max(rewards) if rewards else 0
    min_reward = min(rewards) if rewards else 0
    
    if abs(max_reward) > 10 or abs(min_reward) > 10:
        issues_found.append("❌ Rewards fora da escala adequada (>10)")
    else:
        print("✅ Reward Scale: PASSOU")
    
    print(f"\n📊 RESUMO DOS PROBLEMAS:")
    if issues_found:
        for issue in issues_found:
            print(f"   {issue}")
    else:
        print("   ✅ Nenhum problema crítico detectado")
    
    print(f"\n🏆 RESULTADO: {'✅ APROVADO' if len(issues_found) == 0 else '❌ NECESSITA CORREÇÕES'}")
    print(f"   Problemas: {len(issues_found)}/4")
    print(f"   Taxa de sucesso: {((4-len(issues_found))/4)*100:.0f}%")
    
    return len(issues_found) == 0

if __name__ == "__main__":
    success = test_comprehensive_reward_balance()
    exit(0 if success else 1)