#!/usr/bin/env python3
"""
🍒 BATERIA COMPLETA DE TESTES - REWARD SYSTEM CHERRY
=====================================================

Testes comprehensivos:
1. Balance dos componentes de reward
2. Correlação matemática entre rewards
3. Distribuição por tipo de ação  
4. Consistência temporal
5. Incentivos para trades lucrativos
6. Análise de gradientes de reward
7. Detecção de loops ou feedback negativo
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from collections import defaultdict
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

sys.path.append("D:/Projeto")

class RewardSystemTester:
    """Tester completo para sistema de rewards"""
    
    def __init__(self):
        self.results = {}
        self.test_data = []
        self.env = None
        self.reward_calculator = None
        
    def setup_environment(self):
        """Setup environment Cherry para testes"""
        print("🔧 Configurando ambiente Cherry para testes...")
        
        original_cwd = os.getcwd()
        os.chdir("D:/Projeto")
        
        try:
            from cherry import load_optimized_data_original, TradingEnv
            from trading_framework.rewards.reward_system_simple import SimpleRewardCalculator
            
            # Dados pequenos para testes rápidos
            data = load_optimized_data_original()
            data = data.iloc[-3000:].reset_index(drop=True)
            
            # Ambiente
            self.env = TradingEnv(
                df=data,
                window_size=20,
                is_training=True,
                initial_balance=500.0
            )
            
            # Reward calculator
            self.reward_calculator = SimpleRewardCalculator(initial_balance=500.0)
            
            print(f"✅ Ambiente configurado: {len(data)} barras")
            return True
            
        except Exception as e:
            print(f"❌ Erro setup: {e}")
            return False
        finally:
            os.chdir(original_cwd)
    
    def test_1_component_balance(self):
        """Teste 1: Balance dos componentes de reward"""
        print("\n🔍 TESTE 1: BALANCE DOS COMPONENTES DE REWARD")
        print("=" * 60)
        
        # Simular diferentes cenários de trade
        scenarios = [
            # [action, portfolio_before, portfolio_after, trades_count, positions]
            {"name": "Trade Lucrativo", "action": [1.0, 0.8, 0, 0], "portfolio_change": 50, "trade_success": True},
            {"name": "Trade Perdedor", "action": [1.5, 0.7, 0, 0], "portfolio_change": -30, "trade_success": True},
            {"name": "HOLD Inteligente", "action": [0.2, 0.5, 0, 0], "portfolio_change": 0, "trade_success": False},
            {"name": "Overtrading", "action": [1.8, 0.9, 0, 0], "portfolio_change": -10, "trade_success": True},
            {"name": "Gestão Posição", "action": [0.1, 0.3, 0.8, 0], "portfolio_change": 15, "trade_success": False},
        ]
        
        component_analysis = {}
        
        for scenario in scenarios:
            print(f"\n📊 Cenário: {scenario['name']}")
            
            # Reset environment
            obs = self.env.reset()
            old_state = {
                "portfolio_total_value": self.env.portfolio_value,
                "trades_count": len(self.env.trades)
            }
            
            # Simular mudança de portfolio
            if scenario['trade_success']:
                # Simular trade fechado
                trade_data = {
                    'pnl_usd': scenario['portfolio_change'],
                    'type': 'long' if scenario['action'][0] < 1.5 else 'short',
                    'entry_price': 2000.0,
                    'exit_price': 2000.0 + scenario['portfolio_change']/0.02/100,
                    'sl_points': 15,
                    'tp_points': 25,
                    'duration_hours': 2.5
                }
                self.env.trades.append(trade_data)
            
            self.env.portfolio_value += scenario['portfolio_change']
            
            # Calcular reward
            action = np.array(scenario['action'])
            reward, info, _ = self.reward_calculator.calculate_reward_and_info(
                self.env, action, old_state
            )
            
            components = info.get('components', {})
            
            print(f"  Reward Total: {reward:.3f}")
            print(f"  Componentes:")
            
            scenario_components = {}
            for comp_name, comp_value in components.items():
                if comp_value != 0:
                    print(f"    {comp_name}: {comp_value:.3f}")
                    scenario_components[comp_name] = comp_value
            
            component_analysis[scenario['name']] = {
                'total_reward': reward,
                'components': scenario_components
            }
        
        # Análise de balance
        print(f"\n📈 ANÁLISE DE BALANCE:")
        
        all_components = set()
        for scenario_data in component_analysis.values():
            all_components.update(scenario_data['components'].keys())
        
        component_ranges = {}
        for comp in all_components:
            values = [scenario_data['components'].get(comp, 0) 
                     for scenario_data in component_analysis.values()]
            component_ranges[comp] = {
                'min': min(values),
                'max': max(values), 
                'range': max(values) - min(values),
                'std': np.std(values)
            }
        
        print(f"Componentes com maior variação:")
        sorted_components = sorted(component_ranges.items(), 
                                 key=lambda x: x[1]['range'], reverse=True)
        
        for comp_name, stats in sorted_components[:5]:
            print(f"  {comp_name}: range={stats['range']:.3f}, std={stats['std']:.3f}")
        
        self.results['component_balance'] = {
            'scenarios': component_analysis,
            'component_stats': component_ranges,
            'balance_score': 1.0 - (np.std([s['total_reward'] for s in component_analysis.values()]) / 10)
        }
        
        print(f"Balance Score: {self.results['component_balance']['balance_score']:.3f}")
        
    def test_2_mathematical_correlation(self):
        """Teste 2: Correlação matemática entre rewards e performance"""
        print("\n🔍 TESTE 2: CORRELAÇÃO MATEMÁTICA")
        print("=" * 60)
        
        # Gerar múltiplas simulações
        simulation_data = []
        
        for i in range(100):
            # Reset
            obs = self.env.reset()
            self.reward_calculator.reset()
            
            episode_rewards = []
            episode_pnls = []
            episode_actions = []
            
            # Simular episódio curto
            for step in range(20):
                # Action aleatória
                action = np.array([
                    np.random.uniform(0, 2),      # entry_decision
                    np.random.uniform(0, 1),      # confidence
                    np.random.uniform(-1, 1),     # pos1_mgmt  
                    np.random.uniform(-1, 1)      # pos2_mgmt
                ])
                
                old_state = {
                    "portfolio_total_value": self.env.portfolio_value,
                    "trades_count": len(self.env.trades)
                }
                
                # Executar step
                obs, reward, done, info = self.env.step(action)
                
                episode_rewards.append(reward)
                episode_pnls.append(self.env.portfolio_value - 500.0)  # PnL from initial
                episode_actions.append(action.copy())
                
                if done:
                    break
            
            # Métricas do episódio
            total_reward = sum(episode_rewards)
            final_pnl = self.env.portfolio_value - 500.0
            total_trades = len(self.env.trades)
            
            simulation_data.append({
                'episode': i,
                'total_reward': total_reward,
                'final_pnl': final_pnl,
                'total_trades': total_trades,
                'avg_reward_per_step': np.mean(episode_rewards),
                'reward_volatility': np.std(episode_rewards),
                'actions': episode_actions
            })
        
        # Análise de correlação
        rewards = [d['total_reward'] for d in simulation_data]
        pnls = [d['final_pnl'] for d in simulation_data]
        trades = [d['total_trades'] for d in simulation_data]
        
        # Correlação Reward vs PnL
        corr_reward_pnl, p_reward_pnl = pearsonr(rewards, pnls)
        
        # Correlação Reward vs Trades
        corr_reward_trades, p_reward_trades = pearsonr(rewards, trades)
        
        # Correlação PnL vs Trades  
        corr_pnl_trades, p_pnl_trades = pearsonr(pnls, trades)
        
        print(f"Correlação Reward vs PnL: {corr_reward_pnl:.3f} (p={p_reward_pnl:.3f})")
        print(f"Correlação Reward vs Trades: {corr_reward_trades:.3f} (p={p_reward_trades:.3f})")
        print(f"Correlação PnL vs Trades: {corr_pnl_trades:.3f} (p={p_pnl_trades:.3f})")
        
        # Qualidade das correlações
        correlation_quality = "EXCELENTE" if abs(corr_reward_pnl) > 0.7 else \
                             "BOM" if abs(corr_reward_pnl) > 0.5 else \
                             "FRACO" if abs(corr_reward_pnl) > 0.3 else "RUIM"
        
        print(f"Qualidade Correlação Reward-PnL: {correlation_quality}")
        
        self.results['mathematical_correlation'] = {
            'reward_pnl_correlation': corr_reward_pnl,
            'reward_trades_correlation': corr_reward_trades,
            'pnl_trades_correlation': corr_pnl_trades,
            'correlation_quality': correlation_quality,
            'simulation_data': simulation_data[:10]  # Primeiros 10 para análise
        }
    
    def test_3_action_distribution(self):
        """Teste 3: Distribuição de rewards por tipo de ação"""
        print("\n🔍 TESTE 3: DISTRIBUIÇÃO POR TIPO DE AÇÃO")
        print("=" * 60)
        
        action_rewards = {'HOLD': [], 'LONG': [], 'SHORT': []}
        action_success = {'HOLD': 0, 'LONG': 0, 'SHORT': 0}
        action_count = {'HOLD': 0, 'LONG': 0, 'SHORT': 0}
        
        # Testar actions específicas
        test_actions = [
            # HOLDs
            [0.1, 0.3, 0, 0], [0.2, 0.4, 0, 0], [0.32, 0.5, 0, 0],
            # LONGs
            [0.4, 0.6, 0, 0], [0.5, 0.7, 0, 0], [0.66, 0.8, 0, 0],
            # SHORTs
            [0.7, 0.6, 0, 0], [1.0, 0.7, 0, 0], [1.5, 0.8, 0, 0], [2.0, 0.9, 0, 0]
        ]
        
        for action_vals in test_actions:
            # Reset
            obs = self.env.reset()
            
            # Determinar tipo de ação
            if action_vals[0] < 0.33:
                action_type = 'HOLD'
            elif action_vals[0] < 0.67:
                action_type = 'LONG' 
            else:
                action_type = 'SHORT'
            
            action_count[action_type] += 1
            
            old_state = {
                "portfolio_total_value": self.env.portfolio_value,
                "trades_count": len(self.env.trades)
            }
            
            # Executar action
            action = np.array(action_vals)
            obs, reward, done, info = self.env.step(action)
            
            action_rewards[action_type].append(reward)
            
            # Verificar se teve sucesso (trade executado ou reward positivo)
            if info.get('trade_executed', False) or reward > 5.0:
                action_success[action_type] += 1
        
        # Análise por tipo
        print("Análise por tipo de ação:")
        for action_type in ['HOLD', 'LONG', 'SHORT']:
            if action_rewards[action_type]:
                avg_reward = np.mean(action_rewards[action_type])
                std_reward = np.std(action_rewards[action_type])
                success_rate = action_success[action_type] / action_count[action_type] * 100
                
                print(f"  {action_type:5}: Avg Reward={avg_reward:6.2f}, Std={std_reward:6.2f}, Success={success_rate:5.1f}%")
        
        # Teste ANOVA - diferenças significativas entre grupos
        try:
            f_stat, p_value = stats.f_oneway(
                action_rewards['HOLD'], 
                action_rewards['LONG'], 
                action_rewards['SHORT']
            )
            print(f"ANOVA F-statistic: {f_stat:.3f}, p-value: {p_value:.3f}")
            
            significant_diff = p_value < 0.05
            print(f"Diferenças significativas entre tipos: {'SIM' if significant_diff else 'NÃO'}")
            
        except Exception as e:
            print(f"ANOVA falhou: {e}")
            significant_diff = None
            
        self.results['action_distribution'] = {
            'rewards_by_type': action_rewards,
            'success_by_type': action_success,
            'count_by_type': action_count,
            'significant_differences': significant_diff
        }
    
    def test_4_temporal_consistency(self):
        """Teste 4: Consistência temporal dos rewards"""
        print("\n🔍 TESTE 4: CONSISTÊNCIA TEMPORAL") 
        print("=" * 60)
        
        # Executar sequência longa com mesma action
        obs = self.env.reset()
        
        consistent_action = np.array([1.0, 0.7, 0, 0])  # LONG consistent
        temporal_rewards = []
        temporal_pnls = []
        
        print("Executando sequência consistente...")
        for step in range(50):
            old_portfolio = self.env.portfolio_value
            
            obs, reward, done, info = self.env.step(consistent_action)
            
            temporal_rewards.append(reward)
            temporal_pnls.append(self.env.portfolio_value - 500.0)
            
            if step % 10 == 0:
                print(f"  Step {step}: Reward={reward:.3f}, PnL=${self.env.portfolio_value-500:.2f}")
            
            if done:
                break
        
        # Análise de consistência
        reward_volatility = np.std(temporal_rewards)
        reward_trend = np.polyfit(range(len(temporal_rewards)), temporal_rewards, 1)[0]
        
        # Auto-correlação (reward atual vs reward anterior)
        if len(temporal_rewards) > 1:
            autocorr = np.corrcoef(temporal_rewards[:-1], temporal_rewards[1:])[0,1]
        else:
            autocorr = 0
        
        # Drift detection (tendência consistente)
        reward_drift = abs(temporal_rewards[-1] - temporal_rewards[0]) / len(temporal_rewards)
        
        print(f"\nAnálise Temporal:")
        print(f"  Volatilidade: {reward_volatility:.3f}")
        print(f"  Tendência: {reward_trend:.5f}")
        print(f"  Auto-correlação: {autocorr:.3f}")
        print(f"  Drift: {reward_drift:.5f}")
        
        consistency_score = max(0, 1.0 - reward_volatility/5.0 - abs(reward_drift)*10)
        print(f"  Score Consistência: {consistency_score:.3f}")
        
        self.results['temporal_consistency'] = {
            'volatility': reward_volatility,
            'trend': reward_trend,
            'autocorrelation': autocorr,
            'drift': reward_drift,
            'consistency_score': consistency_score,
            'temporal_data': temporal_rewards[:20]  # Primeiros 20
        }
    
    def test_5_profit_incentives(self):
        """Teste 5: Incentivos corretos para trades lucrativos"""
        print("\n🔍 TESTE 5: INCENTIVOS PARA TRADES LUCRATIVOS")
        print("=" * 60)
        
        profit_scenarios = [
            {'pnl': 100, 'description': 'Grande Lucro'},
            {'pnl': 50, 'description': 'Lucro Médio'},
            {'pnl': 10, 'description': 'Pequeno Lucro'},
            {'pnl': 0, 'description': 'Breakeven'},
            {'pnl': -10, 'description': 'Pequena Perda'},
            {'pnl': -50, 'description': 'Perda Média'},
            {'pnl': -100, 'description': 'Grande Perda'},
        ]
        
        incentive_analysis = {}
        
        for scenario in profit_scenarios:
            obs = self.env.reset()
            
            # Simular trade fechado com PnL específico
            trade_data = {
                'pnl_usd': scenario['pnl'],
                'type': 'long',
                'entry_price': 2000.0,
                'exit_price': 2000.0 + scenario['pnl']/0.02/100,
                'sl_points': 15,
                'tp_points': 25,
                'duration_hours': 2.0
            }
            
            old_state = {
                "portfolio_total_value": self.env.portfolio_value,
                "trades_count": len(self.env.trades)
            }
            
            # Adicionar trade
            self.env.trades.append(trade_data)
            self.env.portfolio_value += scenario['pnl']
            
            # Calcular reward
            action = np.array([1.0, 0.8, 0, 0])
            reward, info, _ = self.reward_calculator.calculate_reward_and_info(
                self.env, action, old_state
            )
            
            incentive_analysis[scenario['description']] = {
                'pnl': scenario['pnl'],
                'reward': reward,
                'reward_per_dollar': reward / abs(scenario['pnl']) if scenario['pnl'] != 0 else reward
            }
            
            print(f"  {scenario['description']:15}: PnL=${scenario['pnl']:4}, Reward={reward:7.3f}")
        
        # Verificar se incentivos estão alinhados
        profits = [s['pnl'] for s in incentive_analysis.values()]
        rewards = [s['reward'] for s in incentive_analysis.values()]
        
        incentive_correlation, _ = pearsonr(profits, rewards)
        
        print(f"\nCorrelação PnL vs Reward: {incentive_correlation:.3f}")
        
        alignment_quality = "EXCELENTE" if incentive_correlation > 0.8 else \
                           "BOM" if incentive_correlation > 0.6 else \
                           "FRACO" if incentive_correlation > 0.3 else "RUIM"
        
        print(f"Alinhamento de Incentivos: {alignment_quality}")
        
        self.results['profit_incentives'] = {
            'scenarios': incentive_analysis,
            'correlation': incentive_correlation,
            'alignment_quality': alignment_quality
        }
    
    def test_6_reward_gradients(self):
        """Teste 6: Análise de gradientes de reward"""
        print("\n🔍 TESTE 6: GRADIENTES DE REWARD")
        print("=" * 60)
        
        # Testar sensibilidade do reward a mudanças pequenas
        base_action = np.array([1.0, 0.8, 0, 0])
        obs = self.env.reset()
        
        # Reward baseline
        old_state = {
            "portfolio_total_value": self.env.portfolio_value,
            "trades_count": len(self.env.trades)
        }
        
        obs, base_reward, done, info = self.env.step(base_action)
        
        # Testar pequenas variações
        variations = [
            {'param': 'entry_decision', 'delta': 0.1, 'index': 0},
            {'param': 'confidence', 'delta': 0.1, 'index': 1},
            {'param': 'pos1_mgmt', 'delta': 0.1, 'index': 2},
            {'param': 'pos2_mgmt', 'delta': 0.1, 'index': 3},
        ]
        
        gradients = {}
        
        for var in variations:
            obs = self.env.reset()
            
            # Action com pequena mudança
            modified_action = base_action.copy()
            modified_action[var['index']] += var['delta']
            
            old_state = {
                "portfolio_total_value": self.env.portfolio_value,
                "trades_count": len(self.env.trades)
            }
            
            obs, modified_reward, done, info = self.env.step(modified_action)
            
            # Calcular gradiente
            gradient = (modified_reward - base_reward) / var['delta']
            gradients[var['param']] = gradient
            
            print(f"  Gradiente {var['param']:15}: {gradient:8.3f}")
        
        # Análise de estabilidade dos gradientes
        gradient_stability = np.std(list(gradients.values()))
        
        print(f"\nEstabilidade dos Gradientes: {gradient_stability:.3f}")
        
        stability_quality = "EXCELENTE" if gradient_stability < 1.0 else \
                           "BOM" if gradient_stability < 5.0 else \
                           "FRACO" if gradient_stability < 10.0 else "RUIM"
        
        print(f"Qualidade da Estabilidade: {stability_quality}")
        
        self.results['reward_gradients'] = {
            'base_reward': base_reward,
            'gradients': gradients,
            'stability': gradient_stability,
            'stability_quality': stability_quality
        }
    
    def generate_comprehensive_report(self):
        """Gerar relatório completo da análise"""
        print("\n📊 RELATÓRIO COMPLETO - REWARD SYSTEM ANALYSIS")
        print("=" * 80)
        
        # Calcular score geral
        scores = {
            'balance': self.results.get('component_balance', {}).get('balance_score', 0),
            'correlation': abs(self.results.get('mathematical_correlation', {}).get('reward_pnl_correlation', 0)),
            'consistency': self.results.get('temporal_consistency', {}).get('consistency_score', 0),
            'incentives': abs(self.results.get('profit_incentives', {}).get('correlation', 0)),
            'stability': 1.0 - min(1.0, self.results.get('reward_gradients', {}).get('stability', 10) / 10)
        }
        
        overall_score = np.mean(list(scores.values()))
        
        print(f"SCORES INDIVIDUAIS:")
        for test_name, score in scores.items():
            quality = "EXCELENTE" if score > 0.8 else "BOM" if score > 0.6 else "MÉDIO" if score > 0.4 else "RUIM"
            print(f"  {test_name.capitalize():15}: {score:.3f} ({quality})")
        
        print(f"\n🎯 SCORE GERAL: {overall_score:.3f}")
        
        overall_quality = "EXCELENTE" if overall_score > 0.8 else \
                         "BOM" if overall_score > 0.6 else \
                         "MÉDIO" if overall_score > 0.4 else "RUIM"
        
        print(f"🏆 QUALIDADE GERAL: {overall_quality}")
        
        # Recomendações
        print(f"\n💡 RECOMENDAÇÕES:")
        
        if scores['balance'] < 0.6:
            print("  - ⚠️ Rebalancear pesos dos componentes de reward")
        
        if scores['correlation'] < 0.5:
            print("  - ⚠️ Melhorar correlação reward-performance")
        
        if scores['consistency'] < 0.6:
            print("  - ⚠️ Reduzir volatilidade temporal dos rewards")
        
        if scores['incentives'] < 0.7:
            print("  - ⚠️ Alinhar melhor incentivos com lucratividadade")
        
        if scores['stability'] < 0.6:
            print("  - ⚠️ Estabilizar gradientes de reward")
        
        if overall_score > 0.8:
            print("  - ✅ Sistema de rewards em excelente estado!")
        
        # Salvar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reward_analysis_cherry_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'overall_score': overall_score,
                'individual_scores': scores,
                'detailed_results': self.results
            }, f, indent=2, default=str)
        
        print(f"\n💾 Relatório salvo: {filename}")
        
        return overall_score

def run_complete_reward_test():
    """Executar bateria completa de testes"""
    print("🍒 BATERIA COMPLETA DE TESTES - REWARD SYSTEM CHERRY")
    print("=" * 80)
    
    tester = RewardSystemTester()
    
    # Setup
    if not tester.setup_environment():
        print("❌ Falha no setup")
        return
    
    # Executar todos os testes
    try:
        tester.test_1_component_balance()
        tester.test_2_mathematical_correlation() 
        tester.test_3_action_distribution()
        tester.test_4_temporal_consistency()
        tester.test_5_profit_incentives()
        tester.test_6_reward_gradients()
        
        # Relatório final
        overall_score = tester.generate_comprehensive_report()
        
        print(f"\n🎯 TESTE COMPLETO FINALIZADO!")
        print(f"📊 Score Final: {overall_score:.3f}")
        
        return overall_score
        
    except Exception as e:
        print(f"❌ Erro durante testes: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

if __name__ == "__main__":
    score = run_complete_reward_test()
    print(f"\n✅ BATERIA DE TESTES CONCLUÍDA - Score: {score:.3f}")