#!/usr/bin/env python3
"""
🔬 ANÁLISE COMPLETA DO REWARD SYSTEM
Teste matemático rigoroso de balanceamento, correlações e viéses
"""

import sys
import importlib
sys.path.append('D:\\Projeto')

# Force reload para garantir valores atuais
if 'trading_framework.rewards.reward_daytrade_v2' in sys.modules:
    importlib.reload(sys.modules['trading_framework.rewards.reward_daytrade_v2'])

import numpy as np
import pandas as pd
from scipy import stats
from trading_framework.rewards.reward_daytrade_v2 import BalancedDayTradingRewardCalculator
import logging
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo
import matplotlib.pyplot as plt

# Configurar logging
logging.basicConfig(level=logging.WARNING)

class AdvancedMockEnv:
    """Environment avançado para testes rigorosos"""
    def __init__(self, initial_balance=1000.0):
        self.trades = []
        self.balance = initial_balance
        self.realized_balance = initial_balance
        self.initial_balance = initial_balance
        self.current_step = 0
        self.positions = []
        self.current_positions = 0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.daily_pnl = []
        self.last_action = None
        
        # Portfolio metrics
        self.portfolio_value = initial_balance
        self.peak_portfolio = initial_balance
        
        # Performance tracking
        self.recent_rewards = []
        self.reward_history_size = 100
        
        # Estado observável simulado
        self.current_price = 2000.0
        self.price_history = [2000.0]
        
    def simulate_price_movement(self, volatility=0.01):
        """Simular movimento de preço"""
        change = np.random.normal(0, volatility) * self.current_price
        self.current_price += change
        self.price_history.append(self.current_price)
        
    def add_trade(self, side='long', pnl=None, duration=None, exit_reason='manual'):
        """Adicionar trade com parâmetros específicos"""
        if pnl is None:
            pnl = np.random.normal(5, 30)  # PnL aleatório
        if duration is None:
            duration = np.random.randint(5, 50)
            
        entry_price = self.current_price - pnl if side == 'long' else self.current_price + pnl
        
        trade = {
            'pnl_usd': pnl,
            'pnl': pnl,
            'duration_steps': duration,
            'position_size': 0.02,
            'entry_price': entry_price,
            'exit_price': self.current_price,
            'side': side,
            'exit_reason': exit_reason,
            'entry_step': max(0, self.current_step - duration),
            'exit_step': self.current_step
        }
        
        self.trades.append(trade)
        self.realized_balance += pnl
        self.portfolio_value += pnl
        
        # Calcular drawdown
        if self.portfolio_value > self.peak_portfolio:
            self.peak_portfolio = self.portfolio_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_portfolio - self.portfolio_value) / self.peak_portfolio
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        return trade

class RewardSystemAnalyzer:
    """Analisador matemático do reward system"""
    
    def __init__(self):
        self.test_results = {}
        self.statistical_tests = {}
        
    def test_component_weights(self) -> Dict:
        """Teste 1: Verificar pesos dos componentes"""
        print("🔬 TESTE 1: PESOS DOS COMPONENTES")
        print("=" * 50)
        
        reward_calc = BalancedDayTradingRewardCalculator(enable_curiosity=False)
        
        # Verificar pesos internos
        weights = {
            'activity_bonus_weight': reward_calc.activity_bonus_weight,
            'inactivity_penalty_weight': reward_calc.inactivity_penalty_weight,
            'target_activity_rate': reward_calc.target_activity_rate,
        }
        
        # Verificar se há base_weights
        if hasattr(reward_calc, 'base_weights'):
            weights.update(reward_calc.base_weights)
            
        print("📊 PESOS INTERNOS:")
        for name, weight in weights.items():
            print(f"   {name}: {weight}")
            
        # Teste com múltiplos cenários
        scenarios = self._generate_test_scenarios()
        component_impacts = {}
        
        for scenario_name, (env, action, old_state) in scenarios.items():
            reward_calc.reset()
            reward, info, done = reward_calc.calculate_reward_and_info(env, action, old_state)
            components = info.get('reward_components', {})
            
            for comp_name, comp_value in components.items():
                if comp_name not in component_impacts:
                    component_impacts[comp_name] = []
                component_impacts[comp_name].append(abs(comp_value))
        
        # Calcular impacto médio de cada componente
        avg_impacts = {}
        for comp_name, values in component_impacts.items():
            avg_impacts[comp_name] = np.mean(values) if values else 0.0
        
        total_impact = sum(avg_impacts.values())
        
        print(f"\n📈 IMPACTO MÉDIO DOS COMPONENTES:")
        proportions = {}
        for comp_name, impact in avg_impacts.items():
            proportion = (impact / total_impact * 100) if total_impact > 0 else 0
            proportions[comp_name] = proportion
            print(f"   {comp_name}: {impact:.6f} ({proportion:.1f}%)")
        
        # Análise de balanceamento
        activity_prop = proportions.get('activity_bonus', 0)
        base_prop = proportions.get('base_reward', 0)
        
        balance_status = "✅ BALANCEADO"
        if activity_prop > 50:
            balance_status = "❌ ACTIVITY DOMINA"
        elif activity_prop < 5:
            balance_status = "⚠️ ACTIVITY MUITO FRACO"
        elif base_prop < 30:
            balance_status = "⚠️ BASE REWARD FRACO"
            
        print(f"\n🎯 STATUS BALANCEAMENTO: {balance_status}")
        
        self.test_results['component_weights'] = {
            'weights': weights,
            'proportions': proportions,
            'status': balance_status,
            'total_impact': total_impact
        }
        
        return self.test_results['component_weights']
    
    def test_reward_correlation_with_performance(self) -> Dict:
        """Teste 2: Correlação reward vs performance"""
        print(f"\n🔬 TESTE 2: CORRELAÇÃO REWARD vs PERFORMANCE")
        print("=" * 50)
        
        reward_calc = BalancedDayTradingRewardCalculator(enable_curiosity=False)
        
        # Gerar diferentes níveis de performance
        performance_scenarios = {
            'very_profitable': {'win_rate': 0.8, 'avg_pnl': 50, 'trades': 20},
            'profitable': {'win_rate': 0.65, 'avg_pnl': 25, 'trades': 15},
            'break_even': {'win_rate': 0.5, 'avg_pnl': 0, 'trades': 10},
            'losing': {'win_rate': 0.35, 'avg_pnl': -15, 'trades': 12},
            'very_losing': {'win_rate': 0.2, 'avg_pnl': -40, 'trades': 8}
        }
        
        correlations = {}
        performance_data = []
        reward_data = []
        
        for scenario_name, params in performance_scenarios.items():
            # Simular múltiplas sessões
            session_rewards = []
            session_pnls = []
            
            for session in range(10):  # 10 sessões por cenário
                env = AdvancedMockEnv()
                reward_calc.reset()
                
                total_reward = 0
                total_pnl = 0
                
                # Simular trades baseados nos parâmetros
                for trade_idx in range(params['trades']):
                    # Determinar se é win ou loss
                    is_win = np.random.random() < params['win_rate']
                    
                    if is_win:
                        pnl = abs(params['avg_pnl']) + np.random.normal(0, 10)
                    else:
                        pnl = -abs(params['avg_pnl']) + np.random.normal(0, 10)
                    
                    trade = env.add_trade(pnl=pnl)
                    total_pnl += pnl
                    
                    # Simular step de reward
                    action = np.array([0.6, 0.2])  # Trade action
                    old_state = {'trades_count': len(env.trades) - 1}
                    reward, info, done = reward_calc.calculate_reward_and_info(env, action, old_state)
                    total_reward += reward
                
                session_rewards.append(total_reward)
                session_pnls.append(total_pnl)
            
            avg_reward = np.mean(session_rewards)
            avg_pnl = np.mean(session_pnls)
            
            performance_data.extend(session_pnls)
            reward_data.extend(session_rewards)
            
            print(f"   {scenario_name}: PnL={avg_pnl:.2f}, Reward={avg_reward:.4f}")
        
        # Calcular correlação
        correlation, p_value = stats.pearsonr(performance_data, reward_data)
        spearman_corr, spearman_p = stats.spearmanr(performance_data, reward_data)
        
        print(f"\n📊 CORRELAÇÕES:")
        print(f"   Pearson: {correlation:.4f} (p={p_value:.4f})")
        print(f"   Spearman: {spearman_corr:.4f} (p={spearman_p:.4f})")
        
        # Interpretação
        if correlation > 0.7:
            corr_status = "✅ FORTE CORRELAÇÃO"
        elif correlation > 0.4:
            corr_status = "⚠️ CORRELAÇÃO MODERADA"
        elif correlation > 0.1:
            corr_status = "❌ CORRELAÇÃO FRACA"
        else:
            corr_status = "❌ SEM CORRELAÇÃO"
            
        print(f"   Status: {corr_status}")
        
        self.test_results['performance_correlation'] = {
            'pearson': correlation,
            'spearman': spearman_corr,
            'p_value': p_value,
            'status': corr_status,
            'sample_size': len(performance_data)
        }
        
        return self.test_results['performance_correlation']
    
    def test_bias_detection(self) -> Dict:
        """Teste 3: Detecção de viéses"""
        print(f"\n🔬 TESTE 3: DETECÇÃO DE VIÉSES")
        print("=" * 50)
        
        reward_calc = BalancedDayTradingRewardCalculator(enable_curiosity=False)
        biases = {}
        
        # Viés 1: Activity Bias (HOLD vs TRADE)
        print("🎯 Testando Activity Bias...")
        hold_rewards = []
        trade_rewards = []
        
        for test in range(50):
            env = AdvancedMockEnv()
            reward_calc.reset()
            
            # HOLD action
            action_hold = np.array([0.0, 0.0])
            old_state = {'trades_count': 0}
            reward_hold, _, _ = reward_calc.calculate_reward_and_info(env, action_hold, old_state)
            hold_rewards.append(reward_hold)
            
            # Reset para trade action
            reward_calc.reset()
            env = AdvancedMockEnv()
            
            # Trade action
            action_trade = np.array([0.7, 0.3])
            reward_trade, _, _ = reward_calc.calculate_reward_and_info(env, action_trade, old_state)
            trade_rewards.append(reward_trade)
        
        hold_mean = np.mean(hold_rewards)
        trade_mean = np.mean(trade_rewards)
        activity_bias = trade_mean - hold_mean
        
        # Teste estatístico
        activity_ttest = stats.ttest_ind(trade_rewards, hold_rewards)
        
        print(f"   HOLD média: {hold_mean:.6f}")
        print(f"   TRADE média: {trade_mean:.6f}")
        print(f"   Diferença: {activity_bias:.6f}")
        print(f"   T-test p-value: {activity_ttest.pvalue:.6f}")
        
        biases['activity_bias'] = {
            'difference': activity_bias,
            'p_value': activity_ttest.pvalue,
            'significant': activity_ttest.pvalue < 0.05
        }
        
        # Viés 2: Size Bias (trades grandes vs pequenos)
        print(f"\n🎯 Testando Size Bias...")
        small_trade_rewards = []
        large_trade_rewards = []
        
        for test in range(30):
            # Small trades
            env_small = AdvancedMockEnv()
            reward_calc.reset()
            
            env_small.add_trade(pnl=5)  # Trade pequeno
            action = np.array([0.6, 0.2])
            old_state = {'trades_count': 0}
            reward_small, _, _ = reward_calc.calculate_reward_and_info(env_small, action, old_state)
            small_trade_rewards.append(reward_small)
            
            # Large trades
            env_large = AdvancedMockEnv()
            reward_calc.reset()
            
            env_large.add_trade(pnl=100)  # Trade grande
            reward_large, _, _ = reward_calc.calculate_reward_and_info(env_large, action, old_state)
            large_trade_rewards.append(reward_large)
        
        small_mean = np.mean(small_trade_rewards)
        large_mean = np.mean(large_trade_rewards)
        size_bias = large_mean - small_mean
        
        size_ttest = stats.ttest_ind(large_trade_rewards, small_trade_rewards)
        
        print(f"   Trade pequeno média: {small_mean:.6f}")
        print(f"   Trade grande média: {large_mean:.6f}")
        print(f"   Diferença: {size_bias:.6f}")
        print(f"   T-test p-value: {size_ttest.pvalue:.6f}")
        
        biases['size_bias'] = {
            'difference': size_bias,
            'p_value': size_ttest.pvalue,
            'significant': size_ttest.pvalue < 0.05
        }
        
        # Viés 3: Temporal Bias (início vs fim de episódio)
        print(f"\n🎯 Testando Temporal Bias...")
        early_rewards = []
        late_rewards = []
        
        for test in range(30):
            env = AdvancedMockEnv()
            reward_calc.reset()
            
            # Early episode
            reward_calc.step_count = 10
            action = np.array([0.6, 0.2])
            old_state = {'trades_count': 0}
            reward_early, _, _ = reward_calc.calculate_reward_and_info(env, action, old_state)
            early_rewards.append(reward_early)
            
            # Late episode
            reward_calc.step_count = 1800
            reward_late, _, _ = reward_calc.calculate_reward_and_info(env, action, old_state)
            late_rewards.append(reward_late)
        
        early_mean = np.mean(early_rewards)
        late_mean = np.mean(late_rewards)
        temporal_bias = late_mean - early_mean
        
        temporal_ttest = stats.ttest_ind(late_rewards, early_rewards)
        
        print(f"   Início episódio média: {early_mean:.6f}")
        print(f"   Fim episódio média: {late_mean:.6f}")
        print(f"   Diferença: {temporal_bias:.6f}")
        print(f"   T-test p-value: {temporal_ttest.pvalue:.6f}")
        
        biases['temporal_bias'] = {
            'difference': temporal_bias,
            'p_value': temporal_ttest.pvalue,
            'significant': temporal_ttest.pvalue < 0.05
        }
        
        self.test_results['biases'] = biases
        return biases
    
    def test_mathematical_properties(self) -> Dict:
        """Teste 4: Propriedades matemáticas"""
        print(f"\n🔬 TESTE 4: PROPRIEDADES MATEMÁTICAS")
        print("=" * 50)
        
        reward_calc = BalancedDayTradingRewardCalculator(enable_curiosity=False)
        properties = {}
        
        # Propriedade 1: Boundedness (limitação)
        print("🎯 Testando Boundedness...")
        rewards = []
        
        for test in range(100):
            env = AdvancedMockEnv()
            reward_calc.reset()
            
            # Cenários extremos
            if test < 25:
                # Muito lucrativo
                for _ in range(5):
                    env.add_trade(pnl=1000)
            elif test < 50:
                # Muito perdedor
                for _ in range(5):
                    env.add_trade(pnl=-1000)
            elif test < 75:
                # Muitos trades pequenos
                for _ in range(50):
                    env.add_trade(pnl=1)
            
            action = np.array([0.6, 0.2])
            old_state = {'trades_count': 0}
            reward, _, _ = reward_calc.calculate_reward_and_info(env, action, old_state)
            rewards.append(reward)
        
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)
        std_reward = np.std(rewards)
        
        print(f"   Min reward: {min_reward:.6f}")
        print(f"   Max reward: {max_reward:.6f}")
        print(f"   Std reward: {std_reward:.6f}")
        
        # Verificar se está boundado adequadamente
        is_bounded = abs(min_reward) < 1000 and abs(max_reward) < 1000
        
        properties['boundedness'] = {
            'min': min_reward,
            'max': max_reward,
            'std': std_reward,
            'is_bounded': is_bounded
        }
        
        # Propriedade 2: Monotonicity (monotonicidade com performance)
        print(f"\n🎯 Testando Monotonicity...")
        pnl_levels = [-100, -50, -10, 0, 10, 50, 100]
        monotonic_rewards = []
        
        for pnl in pnl_levels:
            env = AdvancedMockEnv()
            reward_calc.reset()
            
            env.add_trade(pnl=pnl)
            action = np.array([0.6, 0.2])
            old_state = {'trades_count': 0}
            reward, _, _ = reward_calc.calculate_reward_and_info(env, action, old_state)
            monotonic_rewards.append(reward)
        
        # Verificar se há tendência crescente
        correlation_with_pnl, _ = stats.pearsonr(pnl_levels, monotonic_rewards)
        is_monotonic = correlation_with_pnl > 0.3
        
        print(f"   Correlação PnL-Reward: {correlation_with_pnl:.4f}")
        print(f"   É monotônico: {is_monotonic}")
        
        properties['monotonicity'] = {
            'correlation': correlation_with_pnl,
            'is_monotonic': is_monotonic,
            'rewards': monotonic_rewards,
            'pnl_levels': pnl_levels
        }
        
        # Propriedade 3: Stability (estabilidade)
        print(f"\n🎯 Testando Stability...")
        stability_rewards = []
        
        # Múltiplas execuções do mesmo cenário
        base_env_setup = lambda: AdvancedMockEnv()
        
        for run in range(50):
            env = base_env_setup()
            reward_calc.reset()
            
            # Cenário padrão
            env.add_trade(pnl=25)
            action = np.array([0.6, 0.2])
            old_state = {'trades_count': 0}
            reward, _, _ = reward_calc.calculate_reward_and_info(env, action, old_state)
            stability_rewards.append(reward)
        
        stability_std = np.std(stability_rewards)
        stability_mean = np.mean(stability_rewards)
        cv = stability_std / abs(stability_mean) if stability_mean != 0 else float('inf')
        
        is_stable = cv < 0.5  # Coeficiente de variação < 50%
        
        print(f"   Desvio padrão: {stability_std:.6f}")
        print(f"   Coeficiente de variação: {cv:.4f}")
        print(f"   É estável: {is_stable}")
        
        properties['stability'] = {
            'std': stability_std,
            'mean': stability_mean,
            'cv': cv,
            'is_stable': is_stable
        }
        
        self.test_results['mathematical_properties'] = properties
        return properties
    
    def _generate_test_scenarios(self) -> Dict:
        """Gerar cenários de teste padronizados"""
        scenarios = {}
        
        # Cenário 1: HOLD
        env1 = AdvancedMockEnv()
        action1 = np.array([0.0, 0.0])
        old_state1 = {'trades_count': 0}
        scenarios['hold'] = (env1, action1, old_state1)
        
        # Cenário 2: Trade ativo
        env2 = AdvancedMockEnv()
        action2 = np.array([0.8, 0.3])
        old_state2 = {'trades_count': 0}
        scenarios['active_trading'] = (env2, action2, old_state2)
        
        # Cenário 3: Com trades lucrativos
        env3 = AdvancedMockEnv()
        env3.add_trade(pnl=50)
        env3.add_trade(pnl=30)
        action3 = np.array([0.6, 0.2])
        old_state3 = {'trades_count': 0}
        scenarios['profitable'] = (env3, action3, old_state3)
        
        # Cenário 4: Com trades perdedores
        env4 = AdvancedMockEnv()
        env4.add_trade(pnl=-40)
        env4.add_trade(pnl=-20)
        action4 = np.array([0.6, 0.2])
        old_state4 = {'trades_count': 0}
        scenarios['losing'] = (env4, action4, old_state4)
        
        return scenarios
    
    def generate_comprehensive_report(self) -> str:
        """Gerar relatório completo"""
        report = []
        report.append("🔬 RELATÓRIO COMPLETO DE ANÁLISE DO REWARD SYSTEM")
        report.append("=" * 70)
        
        # Resumo executivo
        report.append("\n📊 RESUMO EXECUTIVO:")
        
        # Component weights
        if 'component_weights' in self.test_results:
            cw = self.test_results['component_weights']
            report.append(f"   Balanceamento: {cw['status']}")
            activity_prop = cw['proportions'].get('activity_bonus', 0)
            report.append(f"   Activity representa: {activity_prop:.1f}% do impacto")
        
        # Performance correlation
        if 'performance_correlation' in self.test_results:
            pc = self.test_results['performance_correlation']
            report.append(f"   Correlação com performance: {pc['status']}")
            report.append(f"   Pearson: {pc['pearson']:.4f}")
        
        # Biases
        if 'biases' in self.test_results:
            biases = self.test_results['biases']
            significant_biases = [name for name, bias in biases.items() if bias['significant']]
            if significant_biases:
                report.append(f"   ⚠️ Viéses detectados: {', '.join(significant_biases)}")
            else:
                report.append("   ✅ Nenhum viés significativo detectado")
        
        # Mathematical properties
        if 'mathematical_properties' in self.test_results:
            mp = self.test_results['mathematical_properties']
            bounded = mp['boundedness']['is_bounded']
            monotonic = mp['monotonicity']['is_monotonic']
            stable = mp['stability']['is_stable']
            
            report.append(f"   Propriedades: Bounded={bounded}, Monotonic={monotonic}, Stable={stable}")
        
        # Recomendações
        report.append("\n🛠️ RECOMENDAÇÕES:")
        
        # Baseado nos resultados
        if 'component_weights' in self.test_results:
            activity_prop = self.test_results['component_weights']['proportions'].get('activity_bonus', 0)
            if activity_prop > 40:
                report.append("   🔧 CRÍTICO: Reduzir activity_bonus_weight")
            elif activity_prop < 10:
                report.append("   ⚖️ Considerar aumentar activity_bonus_weight")
            else:
                report.append("   ✅ Activity weight adequado")
        
        if 'performance_correlation' in self.test_results:
            corr = self.test_results['performance_correlation']['pearson']
            if corr < 0.4:
                report.append("   📈 Melhorar correlação reward-performance")
            else:
                report.append("   ✅ Correlação reward-performance adequada")
        
        if 'biases' in self.test_results:
            biases = self.test_results['biases']
            if biases.get('activity_bias', {}).get('significant', False):
                diff = biases['activity_bias']['difference']
                if abs(diff) > 0.1:
                    report.append("   ⚖️ Ajustar viés de atividade")
        
        # Status final
        report.append("\n🎯 STATUS GERAL:")
        
        issues = 0
        if 'component_weights' in self.test_results:
            if 'DOMINA' in self.test_results['component_weights']['status']:
                issues += 1
        
        if 'performance_correlation' in self.test_results:
            if 'FRACA' in self.test_results['performance_correlation']['status'] or 'SEM' in self.test_results['performance_correlation']['status']:
                issues += 1
        
        if issues == 0:
            report.append("   ✅ SISTEMA REWARD BALANCEADO E FUNCIONAL")
        elif issues == 1:
            report.append("   ⚠️ SISTEMA COM PROBLEMAS MENORES - AJUSTES RECOMENDADOS")
        else:
            report.append("   ❌ SISTEMA COM PROBLEMAS CRÍTICOS - CORREÇÃO NECESSÁRIA")
        
        return "\n".join(report)

def main():
    """Executar análise completa"""
    print("🔬 INICIANDO ANÁLISE COMPLETA DO REWARD SYSTEM")
    print("=" * 70)
    
    analyzer = RewardSystemAnalyzer()
    
    try:
        # Executar todos os testes
        print("⏳ Executando testes... (pode levar alguns minutos)")
        
        analyzer.test_component_weights()
        analyzer.test_reward_correlation_with_performance()
        analyzer.test_bias_detection()
        analyzer.test_mathematical_properties()
        
        # Gerar relatório final
        report = analyzer.generate_comprehensive_report()
        print(f"\n{report}")
        
        # Salvar resultados
        import json
        with open('reward_analysis_results.json', 'w') as f:
            # Converter numpy arrays para listas para JSON
            serializable_results = {}
            for key, value in analyzer.test_results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            serializable_results[key][k] = v.tolist()
                        elif isinstance(v, (np.integer, np.floating)):
                            serializable_results[key][k] = float(v)
                        else:
                            serializable_results[key][k] = v
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Resultados salvos em: reward_analysis_results.json")
        
    except Exception as e:
        print(f"❌ ERRO durante análise: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()