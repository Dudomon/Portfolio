#!/usr/bin/env python3
"""
🔬 BATERIA COMPLETA DE TESTES - REWARD SYSTEM V3.0
VALIDAÇÃO FINAL ANTES DO PRÓXIMO TREINAMENTO

OBJETIVO: GARANTIR que não vai dar merda no milésimo retreino
"""

import sys
sys.path.append("D:/Projeto")
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from trading_framework.rewards.reward_daytrade_v2 import BalancedDayTradingRewardCalculator
import json
from datetime import datetime

class ComprehensiveRewardValidator:
    """Validador completo do sistema de reward antes do treinamento"""
    
    def __init__(self):
        self.calc = BalancedDayTradingRewardCalculator()
        self.test_results = {}
        self.critical_failures = []
        
    def create_mock_env(self, trades_data: List[Dict] = None):
        """Criar environment mockado"""
        class MockEnv:
            def __init__(self, trades=None):
                self.trades = trades or []
                self.current_step = 0
                self.balance = 1000
                self.realized_balance = 1000 + sum(t.get('pnl_usd', 0) for t in self.trades)
                self.portfolio_value = self.realized_balance
                self.initial_balance = 1000
                self.peak_portfolio = max(1000, self.portfolio_value)
                self.current_drawdown = 0.0
                self.max_drawdown = 0.0
                self.current_positions = 0
                self.reward_history_size = 100
                self.recent_rewards = []
                
        return MockEnv(trades_data)
        
    def test_1_no_activity_bonus(self) -> Dict:
        """🚨 TESTE CRÍTICO 1: Confirmar ZERO activity bonus"""
        print("\n🚨 TESTE CRÍTICO 1: VERIFICAÇÃO ACTIVITY BONUS = 0")
        print("=" * 60)
        
        test_scenarios = [
            ("HOLD 100 steps", []),
            ("1 trade pequeno", [{'pnl_usd': 5}]),
            ("10 trades médios", [{'pnl_usd': 20} for _ in range(10)]),
            ("50 trades micro", [{'pnl_usd': 1} for _ in range(50)]),
            ("100 trades overtrading", [{'pnl_usd': 2} for _ in range(100)])
        ]
        
        activity_results = {}
        
        for scenario_name, trades in test_scenarios:
            env = self.create_mock_env(trades)
            action = np.array([0.6, 0.2, 0, 0, 0, 0, 0, 0])
            old_state = {'trades_count': 0}
            
            reward, info, _ = self.calc.calculate_reward_and_info(env, action, old_state)
            activity_bonus = info.get('reward_components', {}).get('activity_bonus', -999)
            
            activity_results[scenario_name] = activity_bonus
            
            if activity_bonus != 0.0:
                self.critical_failures.append(f"❌ FALHA CRÍTICA: {scenario_name} tem activity_bonus = {activity_bonus}")
                print(f"   ❌ {scenario_name}: activity_bonus = {activity_bonus} (DEVERIA SER 0)")
            else:
                print(f"   ✅ {scenario_name}: activity_bonus = {activity_bonus}")
        
        all_zero = all(bonus == 0.0 for bonus in activity_results.values())
        
        print(f"\n🎯 RESULTADO: {'✅ ACTIVITY BONUS ELIMINADO' if all_zero else '❌ AINDA EXISTE ACTIVITY BONUS'}")
        
        return {
            'passed': all_zero,
            'results': activity_results,
            'critical': not all_zero
        }
    
    def test_2_anti_microfarming_effectiveness(self) -> Dict:
        """🚨 TESTE CRÍTICO 2: Eficácia anti-micro-farming"""
        print("\n🚨 TESTE CRÍTICO 2: ANTI-MICRO-FARMING EFFECTIVENESS")
        print("=" * 60)
        
        scenarios = {
            "micro_extreme": [{'pnl_usd': 0.5} for _ in range(200)],  # 200 trades de $0.50
            "micro_moderate": [{'pnl_usd': 2.0} for _ in range(100)], # 100 trades de $2.00
            "quality_small": [{'pnl_usd': 50.0} for _ in range(5)],   # 5 trades de $50
            "quality_large": [{'pnl_usd': 150.0} for _ in range(3)],  # 3 trades de $150
            "balanced": [{'pnl_usd': 30.0} for _ in range(10)]        # 10 trades de $30
        }
        
        results = {}
        
        for name, trades in scenarios.items():
            env = self.create_mock_env(trades)
            action = np.array([0.6, 0.2, 0, 0, 0, 0, 0, 0])
            old_state = {'trades_count': 0}
            
            reward, info, _ = self.calc.calculate_reward_and_info(env, action, old_state)
            
            total_pnl = sum(t['pnl_usd'] for t in trades)
            trade_count = len(trades)
            reward_per_dollar = reward / total_pnl if total_pnl > 0 else 0
            
            results[name] = {
                'reward': reward,
                'total_pnl': total_pnl,
                'trade_count': trade_count,
                'reward_per_dollar': reward_per_dollar,
                'avg_trade_size': total_pnl / trade_count if trade_count > 0 else 0
            }
            
            print(f"   {name:15}: R={reward:.6f}, $/trade=${total_pnl/trade_count:.1f}, R/$ ratio={reward_per_dollar:.6f}")
        
        # VALIDAÇÕES CRÍTICAS
        critical_checks = []
        
        # 1. Quality deve ser > Micro
        if results['quality_large']['reward'] > results['micro_extreme']['reward']:
            print(f"   ✅ Quality Large > Micro Extreme")
        else:
            critical_checks.append("Quality Large ≤ Micro Extreme")
            
        # 2. Micro-farming deve ter reward/dollar BAIXÍSSIMO
        if results['micro_extreme']['reward_per_dollar'] < 0.00001:
            print(f"   ✅ Micro extreme severely penalized (R/$={results['micro_extreme']['reward_per_dollar']:.8f})")
        else:
            critical_checks.append(f"Micro extreme not penalized enough (R/$={results['micro_extreme']['reward_per_dollar']:.8f})")
            
        # 3. Quality deve ter reward/dollar SUPERIOR
        quality_ratio = results['quality_large']['reward_per_dollar']
        micro_ratio = results['micro_extreme']['reward_per_dollar']
        
        if quality_ratio > micro_ratio * 100:  # Quality deve ser 100x+ melhor
            print(f"   ✅ Quality is {quality_ratio/micro_ratio:.0f}x better reward/dollar than micro")
        else:
            critical_checks.append(f"Quality only {quality_ratio/micro_ratio:.1f}x better (should be 100x+)")
        
        self.critical_failures.extend(critical_checks)
        
        return {
            'passed': len(critical_checks) == 0,
            'results': results,
            'critical_failures': critical_checks
        }
    
    def test_3_reward_component_balance(self) -> Dict:
        """🚨 TESTE CRÍTICO 3: Balanceamento de componentes"""
        print("\n🚨 TESTE CRÍTICO 3: BALANCEAMENTO DE COMPONENTES")
        print("=" * 60)
        
        # Cenário padrão para análise
        standard_trades = [{'pnl_usd': 40.0} for _ in range(8)]
        env = self.create_mock_env(standard_trades)
        action = np.array([0.6, 0.2, 0, 0, 0, 0, 0, 0])
        old_state = {'trades_count': 0}
        
        reward, info, _ = self.calc.calculate_reward_and_info(env, action, old_state)
        components = info.get('reward_components', {})
        
        print("   COMPONENTES ATIVOS:")
        active_components = {}
        total_contribution = 0
        
        for comp_name, comp_value in components.items():
            if abs(comp_value) > 0.001:  # Apenas componentes significativos
                active_components[comp_name] = comp_value
                total_contribution += abs(comp_value)
                print(f"     {comp_name:20}: {comp_value:.6f}")
        
        print(f"\n   TOTAL ABSOLUTE CONTRIBUTION: {total_contribution:.6f}")
        
        # VALIDAÇÕES CRÍTICAS
        critical_checks = []
        
        # 1. PnL deve ser o componente dominante (>50% do total)
        pnl_contribution = abs(components.get('pnl', 0))
        pnl_percentage = (pnl_contribution / total_contribution * 100) if total_contribution > 0 else 0
        
        if pnl_percentage > 50:
            print(f"   ✅ PnL is dominant component ({pnl_percentage:.1f}%)")
        else:
            critical_checks.append(f"PnL not dominant ({pnl_percentage:.1f}%, should be >50%)")
        
        # 2. Activity bonus deve ser EXATAMENTE 0
        activity_bonus = components.get('activity_bonus', -999)
        if activity_bonus == 0.0:
            print(f"   ✅ Activity bonus eliminated (0.0)")
        else:
            critical_checks.append(f"Activity bonus still exists ({activity_bonus})")
        
        # 3. Curiosity deve estar desabilitado
        curiosity = components.get('curiosity', -999)
        if curiosity == 0.0:
            print(f"   ✅ Curiosity disabled (0.0)")
        else:
            critical_checks.append(f"Curiosity still active ({curiosity})")
        
        # 4. Gaming penalty deve estar desabilitado (temporário)
        gaming_penalty = components.get('gaming_penalty', -999)
        if gaming_penalty == 0.0:
            print(f"   ✅ Gaming penalty disabled (0.0)")
        else:
            critical_checks.append(f"Gaming penalty active ({gaming_penalty})")
        
        self.critical_failures.extend(critical_checks)
        
        return {
            'passed': len(critical_checks) == 0,
            'components': components,
            'pnl_percentage': pnl_percentage,
            'critical_failures': critical_checks
        }
    
    def test_4_mathematical_properties(self) -> Dict:
        """🚨 TESTE CRÍTICO 4: Propriedades matemáticas"""
        print("\n🚨 TESTE CRÍTICO 4: PROPRIEDADES MATEMÁTICAS")
        print("=" * 60)
        
        # Teste de monotonicidade: PnL crescente deve gerar reward crescente
        pnl_levels = [-50, -20, -5, 0, 5, 20, 50, 100, 200]
        rewards = []
        
        for pnl in pnl_levels:
            env = self.create_mock_env([{'pnl_usd': pnl}])
            action = np.array([0.6, 0.2, 0, 0, 0, 0, 0, 0])
            old_state = {'trades_count': 0}
            
            reward, info, _ = self.calc.calculate_reward_and_info(env, action, old_state)
            rewards.append(reward)
        
        # Verificar monotonicidade
        monotonic_increases = 0
        for i in range(1, len(rewards)):
            if rewards[i] > rewards[i-1]:
                monotonic_increases += 1
        
        monotonic_ratio = monotonic_increases / (len(rewards) - 1)
        
        print(f"   PnL Levels: {pnl_levels}")
        print(f"   Rewards:    {[f'{r:.4f}' for r in rewards]}")
        print(f"   Monotonic increases: {monotonic_increases}/{len(rewards)-1} ({monotonic_ratio:.1%})")
        
        # VALIDAÇÕES CRÍTICAS
        critical_checks = []
        
        if monotonic_ratio >= 0.8:  # 80%+ dos casos deve ser monotônico
            print(f"   ✅ Reward is mostly monotonic with PnL ({monotonic_ratio:.1%})")
        else:
            critical_checks.append(f"Reward not monotonic enough ({monotonic_ratio:.1%}, should be ≥80%)")
        
        # Teste de boundedness: rewards não devem explodir
        max_reward = max(rewards)
        min_reward = min(rewards)
        
        if abs(max_reward) < 10 and abs(min_reward) < 10:
            print(f"   ✅ Rewards are bounded (max={max_reward:.4f}, min={min_reward:.4f})")
        else:
            critical_checks.append(f"Rewards not properly bounded (max={max_reward:.4f}, min={min_reward:.4f})")
        
        self.critical_failures.extend(critical_checks)
        
        return {
            'passed': len(critical_checks) == 0,
            'monotonic_ratio': monotonic_ratio,
            'reward_range': (min_reward, max_reward),
            'pnl_reward_pairs': list(zip(pnl_levels, rewards)),
            'critical_failures': critical_checks
        }
    
    def test_5_edge_cases(self) -> Dict:
        """🚨 TESTE CRÍTICO 5: Casos extremos"""
        print("\n🚨 TESTE CRÍTICO 5: CASOS EXTREMOS")
        print("=" * 60)
        
        edge_cases = {
            "empty_trades": [],
            "single_huge_win": [{'pnl_usd': 1000}],
            "single_huge_loss": [{'pnl_usd': -1000}],
            "mixed_realistic": [
                {'pnl_usd': 50}, {'pnl_usd': -30}, {'pnl_usd': 80},
                {'pnl_usd': -20}, {'pnl_usd': 100}, {'pnl_usd': -40}
            ],
            "all_losses": [{'pnl_usd': -10} for _ in range(10)],
            "tiny_profits": [{'pnl_usd': 0.01} for _ in range(1000)]  # Extreme micro-farming
        }
        
        results = {}
        critical_checks = []
        
        for case_name, trades in edge_cases.items():
            try:
                env = self.create_mock_env(trades)
                action = np.array([0.6, 0.2, 0, 0, 0, 0, 0, 0])
                old_state = {'trades_count': 0}
                
                reward, info, _ = self.calc.calculate_reward_and_info(env, action, old_state)
                
                results[case_name] = {
                    'reward': reward,
                    'trade_count': len(trades),
                    'total_pnl': sum(t.get('pnl_usd', 0) for t in trades),
                    'components': info.get('reward_components', {})
                }
                
                print(f"   {case_name:20}: R={reward:.6f}, trades={len(trades)}")
                
                # Validações específicas
                if case_name == "empty_trades" and abs(reward) > 0.001:
                    critical_checks.append(f"Empty trades should give ~0 reward, got {reward:.6f}")
                
                if case_name == "tiny_profits" and reward > 0.01:
                    critical_checks.append(f"Extreme micro-farming should be heavily penalized, got {reward:.6f}")
                
                if abs(reward) > 100:  # Reward exploding
                    critical_checks.append(f"{case_name}: Reward exploded to {reward:.6f}")
                    
            except Exception as e:
                critical_checks.append(f"{case_name}: Exception {str(e)}")
                results[case_name] = {'error': str(e)}
        
        self.critical_failures.extend(critical_checks)
        
        return {
            'passed': len(critical_checks) == 0,
            'results': results,
            'critical_failures': critical_checks
        }
    
    def test_6_consistency_across_episodes(self) -> Dict:
        """🚨 TESTE CRÍTICO 6: Consistência entre episódios"""
        print("\n🚨 TESTE CRÍTICO 6: CONSISTÊNCIA ENTRE EPISÓDIOS")
        print("=" * 60)
        
        # Mesmo cenário testado 20 vezes
        standard_scenario = [{'pnl_usd': 25} for _ in range(8)]
        rewards = []
        
        for episode in range(20):
            # Reset calculator para simular novo episódio
            calc = BalancedDayTradingRewardCalculator()
            
            env = self.create_mock_env(standard_scenario)
            action = np.array([0.6, 0.2, 0, 0, 0, 0, 0, 0])
            old_state = {'trades_count': 0}
            
            reward, info, _ = calc.calculate_reward_and_info(env, action, old_state)
            rewards.append(reward)
        
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        cv = std_reward / abs(mean_reward) if mean_reward != 0 else float('inf')
        
        print(f"   Mean reward: {mean_reward:.6f}")
        print(f"   Std reward:  {std_reward:.6f}")
        print(f"   CV:          {cv:.4f}")
        
        critical_checks = []
        
        # Reward deve ser consistente (CV < 5%)
        if cv < 0.05:
            print(f"   ✅ Reward is consistent (CV={cv:.4f})")
        else:
            critical_checks.append(f"Reward inconsistent across episodes (CV={cv:.4f}, should be <0.05)")
        
        self.critical_failures.extend(critical_checks)
        
        return {
            'passed': len(critical_checks) == 0,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'cv': cv,
            'all_rewards': rewards,
            'critical_failures': critical_checks
        }
    
    def run_full_validation(self) -> Dict:
        """Executar validação completa"""
        print("🔬 INICIANDO VALIDAÇÃO COMPLETA DO REWARD SYSTEM V3.0")
        print("🎯 OBJETIVO: GARANTIR funcionamento antes do próximo treinamento")
        print("=" * 80)
        
        # Executar todos os testes
        test_results = {}
        
        test_results['test_1_no_activity_bonus'] = self.test_1_no_activity_bonus()
        test_results['test_2_anti_microfarming'] = self.test_2_anti_microfarming_effectiveness()
        test_results['test_3_component_balance'] = self.test_3_reward_component_balance()
        test_results['test_4_mathematical_properties'] = self.test_4_mathematical_properties()
        test_results['test_5_edge_cases'] = self.test_5_edge_cases()
        test_results['test_6_consistency'] = self.test_6_consistency_across_episodes()
        
        # RESULTADO FINAL
        print("\n" + "="*80)
        print("🏆 RESULTADO FINAL DA VALIDAÇÃO")
        print("="*80)
        
        passed_tests = 0
        total_tests = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
            print(f"   {test_name:30}: {status}")
            if result.get('passed', False):
                passed_tests += 1
        
        success_rate = passed_tests / total_tests * 100
        
        print(f"\n📊 TAXA DE SUCESSO: {success_rate:.0f}% ({passed_tests}/{total_tests})")
        
        if len(self.critical_failures) > 0:
            print(f"\n❌ FALHAS CRÍTICAS DETECTADAS ({len(self.critical_failures)}):")
            for i, failure in enumerate(self.critical_failures, 1):
                print(f"   {i}. {failure}")
        else:
            print(f"\n✅ NENHUMA FALHA CRÍTICA DETECTADA")
        
        # Recomendação final
        if success_rate >= 90 and len(self.critical_failures) == 0:
            recommendation = "🟢 SAFE TO TRAIN - Sistema aprovado para treinamento"
        elif success_rate >= 75:
            recommendation = "🟡 CAUTION - Treinar com cuidado, monitorar closely"
        else:
            recommendation = "🔴 DO NOT TRAIN - Corrigir problemas antes do treinamento"
        
        print(f"\n🎯 RECOMENDAÇÃO: {recommendation}")
        
        # Salvar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"reward_validation_report_{timestamp}.json"
        
        report_data = {
            'timestamp': timestamp,
            'success_rate': success_rate,
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'critical_failures': self.critical_failures,
            'recommendation': recommendation,
            'test_results': test_results
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n💾 Relatório salvo em: {report_file}")
        
        return report_data

def main():
    validator = ComprehensiveRewardValidator()
    results = validator.run_full_validation()
    
    # Return exit code based on results
    if results['success_rate'] >= 90 and len(results['critical_failures']) == 0:
        return 0  # Success
    else:
        return 1  # Failure

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)