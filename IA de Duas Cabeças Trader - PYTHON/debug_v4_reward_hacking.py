"""
🔍 V4 INNO REWARD HACKING DETECTOR
Investigar explained variance positivo anômalo e possível reward hacking
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import sys
import os

# Adicionar path para importar reward system
sys.path.append('D:/Projeto/trading_framework/rewards/')
from reward_daytrade_v4_inno import InnovativeMoneyReward
from reward_daytrade_v3_brutal import BrutalMoneyReward

class RewardHackingDetector:
    """Detecta possíveis problemas no V4 INNO reward system"""
    
    def __init__(self):
        self.v3_system = BrutalMoneyReward(initial_balance=10000)
        self.v4_system = InnovativeMoneyReward(initial_balance=10000)
        
    def create_mock_env(self, realized_pnl, unrealized_pnl, portfolio_value, peak_value, 
                       trades=None, positions=None, current_step=0):
        """Criar ambiente mock para testes"""
        class MockEnv:
            def __init__(self, realized_pnl, unrealized_pnl, portfolio_value, peak_value, trades, positions, current_step):
                self.total_realized_pnl = realized_pnl
                self.total_unrealized_pnl = unrealized_pnl
                self.portfolio_value = portfolio_value
                self.peak_portfolio_value = peak_value
                self.trades = trades or []
                self.positions = positions or []
                self.current_step = current_step
                
        return MockEnv(realized_pnl, unrealized_pnl, portfolio_value, peak_value, trades, positions, current_step)
    
    def test_reward_scaling_linearity(self):
        """🔍 TESTE 1: Verificar se rewards escalam linearmente com PnL"""
        print("🔍 TESTE 1: LINEARIDADE DOS REWARDS")
        print("=" * 60)
        
        pnl_values = [-1000, -500, -100, 0, 100, 500, 1000, 2000]
        action = np.array([0.5, 0.3, 0.2, 0.1])
        
        results = []
        
        for pnl in pnl_values:
            env = self.create_mock_env(pnl, 0, 10000 + pnl, 10000, [], [], 100)
            
            v3_reward, v3_info, _ = self.v3_system.calculate_reward_and_info(env, action, {})
            v4_reward, v4_info, _ = self.v4_system.calculate_reward_and_info(env, action, {})
            
            results.append({
                'pnl': pnl,
                'pnl_percent': pnl / 10000 * 100,
                'v3_reward': v3_reward,
                'v4_reward': v4_reward,
                'v4_pnl_component': v4_info.get('pure_pnl_component', 0),
                'v4_activity_component': v4_info.get('activity_component', 0),
                'v4_shaping_component': v4_info.get('shaping_component', 0)
            })
        
        # Analisar linearidade
        print(f"{'PnL %':<8} {'V3 Reward':<12} {'V4 Reward':<12} {'V4 PnL':<12} {'V4 Act':<12} {'V4 Shp':<12} {'Ratio':<8}")
        print("-" * 80)
        
        for r in results:
            ratio = r['v4_reward'] / r['v3_reward'] if r['v3_reward'] != 0 else float('inf')
            print(f"{r['pnl_percent']:+6.1f}%   {r['v3_reward']:+8.4f}    {r['v4_reward']:+8.4f}    "
                  f"{r['v4_pnl_component']:+8.4f}    {r['v4_activity_component']:+8.4f}    "
                  f"{r['v4_shaping_component']:+8.4f}    {ratio:6.2f}x")
        
        # Verificar se ratio é constante (indicaria linearidade)
        ratios = [r['v4_reward'] / r['v3_reward'] for r in results if r['v3_reward'] != 0]
        ratio_std = np.std(ratios)
        ratio_mean = np.mean(ratios)
        
        print(f"\n📊 ANÁLISE LINEARIDADE:")
        print(f"   Ratio médio V4/V3: {ratio_mean:.2f}x")
        print(f"   Desvio padrão: {ratio_std:.3f}")
        print(f"   Linearidade: {'✅ BOA' if ratio_std < 0.5 else '⚠️ PROBLEMÁTICA'}")
        
        return results
    
    def test_activity_bonus_exploitation(self):
        """🔍 TESTE 2: Detectar exploração do activity bonus"""
        print("\n🔍 TESTE 2: EXPLORAÇÃO DO ACTIVITY BONUS")
        print("=" * 60)
        
        action = np.array([0.5, 0.3, 0.2, 0.1])
        scenarios = []
        
        # Diferentes combinações de trades e PnL
        test_cases = [
            # (trades_count, total_pnl, scenario_name)
            (0, 0, "Sem trades"),
            (1, -100, "1 trade perdedor"),
            (1, 100, "1 trade vencedor"),
            (5, 0, "5 trades break-even"),
            (5, 500, "5 trades lucrativos"),
            (5, -500, "5 trades perdedores"),
            (15, 300, "15 trades (overtrading) lucrativos"),
            (15, -300, "15 trades (overtrading) perdedores"),
            (50, 100, "50 trades (extreme overtrading)"),
        ]
        
        for trades_count, total_pnl, scenario in test_cases:
            # Criar trades artificiais
            if trades_count > 0:
                pnl_per_trade = total_pnl / trades_count
                trades = [{'pnl_usd': pnl_per_trade} for _ in range(trades_count)]
            else:
                trades = []
            
            env = self.create_mock_env(total_pnl, 0, 10000 + total_pnl, 10000, trades, [], 500)
            
            v4_reward, v4_info, _ = self.v4_system.calculate_reward_and_info(env, action, {})
            
            scenarios.append({
                'scenario': scenario,
                'trades_count': trades_count,
                'total_pnl': total_pnl,
                'pnl_per_trade': total_pnl / trades_count if trades_count > 0 else 0,
                'total_reward': v4_reward,
                'pnl_component': v4_info.get('pure_pnl_component', 0),
                'activity_component': v4_info.get('activity_component', 0),
                'activity_status': v4_info.get('activity_status', 'N/A'),
                'trades_per_episode': v4_info.get('trades_per_episode', 0)
            })
        
        print(f"{'Scenario':<25} {'Trades':<8} {'PnL/Trade':<10} {'Total R':<10} {'PnL R':<10} {'Act R':<10} {'Status':<15}")
        print("-" * 100)
        
        for s in scenarios:
            print(f"{s['scenario']:<25} {s['trades_count']:<8} {s['pnl_per_trade']:+8.1f}   "
                  f"{s['total_reward']:+8.4f} {s['pnl_component']:+8.4f} {s['activity_component']:+8.4f} "
                  f"{s['activity_status']:<15}")
        
        # Detectar possível hacking
        print(f"\n🚨 DETECÇÃO DE HACKING:")
        
        # Caso suspeito: muitos trades pequenos vs poucos trades grandes
        case_many_small = next(s for s in scenarios if s['trades_count'] == 15 and s['total_pnl'] > 0)
        case_few_large = next(s for s in scenarios if s['trades_count'] == 5 and s['total_pnl'] > 0)
        
        if case_many_small['total_reward'] > case_few_large['total_reward']:
            reward_diff = case_many_small['total_reward'] - case_few_large['total_reward']
            print(f"   ⚠️ POSSÍVEL HACKING: Overtrading recompensado!")
            print(f"      15 trades = {case_many_small['total_reward']:+.4f}")
            print(f"      5 trades  = {case_few_large['total_reward']:+.4f}")
            print(f"      Diferença = {reward_diff:+.4f} (favorece overtrading)")
        else:
            print(f"   ✅ Activity bonus funciona corretamente")
        
        return scenarios
    
    def test_unrealized_pnl_hacking(self):
        """🔍 TESTE 3: Detectar exploração do unrealized PnL"""
        print("\n🔍 TESTE 3: EXPLORAÇÃO DO UNREALIZED PNL")
        print("=" * 60)
        
        action = np.array([0.5, 0.3, 0.2, 0.1])
        
        test_cases = [
            # (realized, unrealized, scenario)
            (0, 500, "Apenas unrealized +5%"),
            (500, 0, "Apenas realized +5%"),
            (250, 250, "50/50 realized/unrealized"),
            (100, 400, "80% unrealized"),
            (400, 100, "80% realized"),
            (0, 1000, "Unrealized extremo +10%"),
            (0, -500, "Unrealized negativo -5%"),
            (-500, 0, "Realized negativo -5%"),
        ]
        
        results = []
        
        for realized, unrealized, scenario in test_cases:
            total_portfolio = 10000 + realized + unrealized
            
            # Criar posição se há unrealized
            positions = []
            if unrealized != 0:
                positions = [{'id': 'pos1', 'unrealized_pnl': unrealized}]
            
            env = self.create_mock_env(realized, unrealized, total_portfolio, 10000, [], positions, 500)
            
            v3_reward, v3_info, _ = self.v3_system.calculate_reward_and_info(env, action, {})
            v4_reward, v4_info, _ = self.v4_system.calculate_reward_and_info(env, action, {})
            
            results.append({
                'scenario': scenario,
                'realized': realized,
                'unrealized': unrealized,
                'total_pnl': realized + unrealized,
                'v3_reward': v3_reward,
                'v4_reward': v4_reward,
                'v3_total_pnl': v3_info.get('total_pnl', 0),
                'v4_total_pnl': v4_info.get('total_pnl', 0),
                'v4_unrealized_factor': v4_info.get('v4_unrealized_factor', 0.5)
            })
        
        print(f"{'Scenario':<25} {'Realized':<10} {'Unreal':<8} {'V3 R':<10} {'V4 R':<10} {'V3 PnL':<8} {'V4 PnL':<8}")
        print("-" * 90)
        
        for r in results:
            print(f"{r['scenario']:<25} {r['realized']:+8.0f}   {r['unrealized']:+6.0f}   "
                  f"{r['v3_reward']:+8.4f} {r['v4_reward']:+8.4f} {r['v3_total_pnl']:+6.0f}   {r['v4_total_pnl']:+6.0f}")
        
        # Verificar se valorização do unrealized está causando distorção
        unrealized_only = next(r for r in results if r['realized'] == 0 and r['unrealized'] == 500)
        realized_only = next(r for r in results if r['realized'] == 500 and r['unrealized'] == 0)
        
        unrealized_advantage = unrealized_only['v4_reward'] - unrealized_only['v3_reward']
        realized_advantage = realized_only['v4_reward'] - realized_only['v3_reward']
        
        print(f"\n🎯 ANÁLISE UNREALIZED:")
        print(f"   Unrealized boost V4 vs V3: {unrealized_advantage:+.4f}")
        print(f"   Realized boost V4 vs V3: {realized_advantage:+.4f}")
        print(f"   Bias para unrealized: {unrealized_advantage - realized_advantage:+.4f}")
        
        if abs(unrealized_advantage - realized_advantage) > 0.1:
            print(f"   ⚠️ POSSÍVEL HACKING: Unrealized PnL super-valorizado")
        else:
            print(f"   ✅ Valorização balanceada")
        
        return results
    
    def test_critic_confusion_patterns(self):
        """🔍 TESTE 4: Detectar padrões que confundem o critic"""
        print("\n🔍 TESTE 4: PADRÕES QUE CONFUNDEM O CRITIC")
        print("=" * 60)
        
        action = np.array([0.5, 0.3, 0.2, 0.1])
        
        # Padrões que podem confundir explained variance
        patterns = [
            # Recompensas que não refletem valor real
            ("High reward, low value", 200, 0, 10200, [{'pnl_usd': 50} for _ in range(4)]),  # 4 trades pequenos
            ("Low reward, high value", 1000, 0, 11000, [{'pnl_usd': 1000}]),  # 1 trade grande
            ("Inconsistent rewards", 500, 0, 10500, [{'pnl_usd': 200}, {'pnl_usd': 300}]),  # Trades inconsistentes
            ("Activity vs PnL conflict", 100, 0, 10100, [{'pnl_usd': 10} for _ in range(10)]),  # Muitos trades minúsculos
            ("Shaping dominance", 50, 450, 10500, [{'pnl_usd': 50}]),  # Unrealized alto, realized baixo
        ]
        
        results = []
        
        for pattern_name, realized, unrealized, portfolio, trades in patterns:
            positions = []
            if unrealized > 0:
                positions = [{'id': 'pos1', 'unrealized_pnl': unrealized}]
            
            env = self.create_mock_env(realized, unrealized, portfolio, 10000, trades, positions, 500)
            
            reward, info, _ = self.v4_system.calculate_reward_and_info(env, action, {})
            
            # Calcular "true value" baseado apenas no PnL total
            true_value = (portfolio - 10000) / 10000  # % gain
            
            # Reward breakdown
            pnl_component = info.get('pure_pnl_component', 0)
            activity_component = info.get('activity_component', 0)
            shaping_component = info.get('shaping_component', 0)
            
            results.append({
                'pattern': pattern_name,
                'total_reward': reward,
                'true_value': true_value,
                'pnl_component': pnl_component,
                'activity_component': activity_component,
                'shaping_component': shaping_component,
                'reward_value_ratio': reward / true_value if true_value != 0 else float('inf'),
                'non_pnl_ratio': (activity_component + shaping_component) / reward if reward != 0 else 0
            })
        
        print(f"{'Pattern':<20} {'Total R':<10} {'True V':<10} {'R/V Ratio':<10} {'Non-PnL%':<10}")
        print("-" * 70)
        
        for r in results:
            print(f"{r['pattern']:<20} {r['total_reward']:+8.4f} {r['true_value']:+8.4f} "
                  f"{r['reward_value_ratio']:+8.2f} {r['non_pnl_ratio']*100:+7.1f}%")
        
        # Detectar inconsistências
        print(f"\n🎯 DETECÇÃO DE INCONSISTÊNCIAS:")
        
        high_ratio_cases = [r for r in results if abs(r['reward_value_ratio']) > 50]
        high_nonpnl_cases = [r for r in results if r['non_pnl_ratio'] > 0.3]
        
        if high_ratio_cases:
            print(f"   ⚠️ REWARD/VALUE DESBALANCEADO:")
            for case in high_ratio_cases:
                print(f"      {case['pattern']}: Ratio = {case['reward_value_ratio']:+.1f}")
        
        if high_nonpnl_cases:
            print(f"   ⚠️ COMPONENTES NÃO-PNL DOMINANTES:")
            for case in high_nonpnl_cases:
                print(f"      {case['pattern']}: {case['non_pnl_ratio']*100:.1f}% não-PnL")
        
        return results
    
    def run_full_analysis(self):
        """Executar análise completa"""
        print("🔍 V4 INNO REWARD HACKING DETECTION")
        print("=" * 80)
        print("Investigando explained variance positivo anômalo...")
        print("=" * 80)
        
        # Executar todos os testes
        linearity_results = self.test_reward_scaling_linearity()
        activity_results = self.test_activity_bonus_exploitation()
        unrealized_results = self.test_unrealized_pnl_hacking()
        critic_results = self.test_critic_confusion_patterns()
        
        # Resumo final
        print("\n" + "=" * 80)
        print("📋 RESUMO EXECUTIVO - POSSÍVEIS CAUSAS DO EXPLAINED VARIANCE POSITIVO")
        print("=" * 80)
        
        issues_found = []
        
        # Check linearity
        ratios = [r['v4_reward'] / r['v3_reward'] for r in linearity_results if r['v3_reward'] != 0]
        if np.std(ratios) > 0.5:
            issues_found.append("⚠️ LINEARIDADE: Rewards não escalam proporcionalmente")
        
        # Check activity exploitation
        overtrading_case = next((s for s in activity_results if s['trades_count'] == 15 and s['total_pnl'] > 0), None)
        normal_case = next((s for s in activity_results if s['trades_count'] == 5 and s['total_pnl'] > 0), None)
        if overtrading_case and normal_case and overtrading_case['total_reward'] > normal_case['total_reward']:
            issues_found.append("⚠️ ACTIVITY HACKING: Overtrading sendo recompensado")
        
        # Check unrealized bias
        unrealized_adv = next((r for r in unrealized_results if r['realized'] == 0 and r['unrealized'] == 500), None)
        realized_adv = next((r for r in unrealized_results if r['realized'] == 500 and r['unrealized'] == 0), None)
        if unrealized_adv and realized_adv:
            bias = (unrealized_adv['v4_reward'] - unrealized_adv['v3_reward']) - (realized_adv['v4_reward'] - realized_adv['v3_reward'])
            if abs(bias) > 0.1:
                issues_found.append("⚠️ UNREALIZED BIAS: Posições abertas super-valorizadas")
        
        # Check critic confusion
        high_nonpnl = [r for r in critic_results if r['non_pnl_ratio'] > 0.3]
        if high_nonpnl:
            issues_found.append("⚠️ CRITIC CONFUSION: Componentes não-PnL dominando rewards")
        
        if issues_found:
            print("🚨 PROBLEMAS DETECTADOS:")
            for issue in issues_found:
                print(f"   {issue}")
            print(f"\n💡 POSSÍVEL CAUSA DO EXPLAINED VARIANCE POSITIVO:")
            print(f"   O critic pode estar aprendendo a explorar esses padrões anômalos")
            print(f"   ao invés de aprender o valor real dos estados")
        else:
            print("✅ Nenhum problema crítico detectado nos rewards V4")
            print("   Explained variance positivo pode ter outras causas")
        
        return {
            'linearity': linearity_results,
            'activity': activity_results, 
            'unrealized': unrealized_results,
            'critic': critic_results,
            'issues': issues_found
        }

if __name__ == "__main__":
    detector = RewardHackingDetector()
    results = detector.run_full_analysis()