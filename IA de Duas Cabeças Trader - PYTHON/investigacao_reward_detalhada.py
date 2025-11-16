#!/usr/bin/env python3
"""
🔍 INVESTIGAÇÃO SISTEMÁTICA E MINUCIOSA DO REWARD SYSTEM
Análise granular de cada componente para sistema impecável
"""

import sys
import os
sys.path.append("D:/Projeto")

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from trading_framework.rewards.reward_daytrade_v2 import BalancedDayTradingRewardCalculator

class RewardSystemForensics:
    """Investigação forense do sistema de rewards"""
    
    def __init__(self):
        self.calculator = BalancedDayTradingRewardCalculator(enable_curiosity=False)
        
    def analyze_individual_components(self):
        """Análise granular de cada componente individualmente"""
        print("🔬 ANÁLISE FORENSE - COMPONENTES INDIVIDUAIS")
        print("=" * 60)
        
        # Cenários de teste específicos
        test_scenarios = [
            {"name": "Trade Lucrativo 1%", "pnl": 0.01, "win": True},
            {"name": "Trade Lucrativo 2%", "pnl": 0.02, "win": True}, 
            {"name": "Trade Lucrativo 5%", "pnl": 0.05, "win": True},
            {"name": "Trade com Perda 1%", "pnl": -0.01, "win": False},
            {"name": "Trade com Perda 2%", "pnl": -0.02, "win": False},
            {"name": "Trade com Perda 5%", "pnl": -0.05, "win": False},
            {"name": "Trade Neutro", "pnl": 0.0, "win": False}
        ]
        
        weights = self.calculator.base_weights
        print("📊 PESOS CONFIGURADOS:")
        for comp, weight in weights.items():
            print(f"   {comp}: {weight}")
        
        print("\n🧮 CÁLCULO DETALHADO POR CENÁRIO:")
        
        for scenario in test_scenarios:
            print(f"\n📋 {scenario['name']} (PnL: {scenario['pnl']*100:.1f}%)")
            print("-" * 40)
            
            # Calcular cada componente manualmente
            pnl = scenario['pnl']
            
            # 1. PnL Direto
            pnl_direct = pnl * weights['pnl_direct']
            print(f"   PnL Direct: {pnl:.3f} × {weights['pnl_direct']:.1f} = {pnl_direct:.6f}")
            
            # 2. Win/Loss Bonus
            if scenario['win']:
                win_bonus = weights['win_bonus']
                loss_penalty = 0
                print(f"   Win Bonus: {weights['win_bonus']:.1f} (APLICADO)")
                print(f"   Loss Penalty: 0 (não aplicado)")
            else:
                win_bonus = 0
                loss_penalty = weights['loss_penalty'] if pnl < 0 else 0
                print(f"   Win Bonus: 0 (não aplicado)")
                print(f"   Loss Penalty: {weights['loss_penalty']:.1f} = {loss_penalty:.3f}")
            
            # Total do trade
            total = pnl_direct + win_bonus + loss_penalty
            print(f"   TOTAL TRADE: {total:.6f}")
            
            # Contribuições relativas
            if abs(total) > 0.0001:
                contrib_pnl = abs(pnl_direct) / abs(total) * 100
                contrib_win = abs(win_bonus) / abs(total) * 100
                contrib_loss = abs(loss_penalty) / abs(total) * 100
                
                print(f"   CONTRIBUIÇÕES:")
                print(f"     PnL Direct: {contrib_pnl:.1f}%")
                print(f"     Win Bonus: {contrib_win:.1f}%") 
                print(f"     Loss Penalty: {contrib_loss:.1f}%")
                
                # Identificar problemas
                if contrib_pnl < 50:
                    print(f"   ⚠️ PnL Direct BAIXO! Deveria ser >50%")
                if contrib_win > 30 or contrib_loss > 30:
                    print(f"   ⚠️ Win/Loss dominando! Deveria ser <30%")
    
    def test_win_loss_correlation(self):
        """Testar correlação entre win_bonus e loss_penalty"""
        print("\n🔍 INVESTIGAÇÃO: CORRELAÇÃO WIN_BONUS ↔ LOSS_PENALTY")
        print("=" * 60)
        
        scenarios = []
        for i in range(1000):
            pnl = np.random.uniform(-0.05, 0.05)
            win = pnl > 0
            
            scenario = {
                'pnl': pnl,
                'win': win,
                'win_bonus': self.calculator.base_weights['win_bonus'] if win else 0,
                'loss_penalty': self.calculator.base_weights['loss_penalty'] if not win and pnl < 0 else 0
            }
            scenarios.append(scenario)
        
        df = pd.DataFrame(scenarios)
        correlation = df['win_bonus'].corr(df['loss_penalty'])
        
        print(f"📊 CORRELAÇÃO WIN_BONUS ↔ LOSS_PENALTY: {correlation:.3f}")
        
        if abs(correlation) > 0.7:
            print("🚨 PROBLEMA: Correlação muito alta!")
            print("   Causa: Win_bonus e Loss_penalty são aplicados de forma mutuamente exclusiva")
            print("   Solução: Reformular para serem independentes do sinal do PnL")
        
        # Análise de quando são aplicados
        only_win = df[(df['win_bonus'] > 0) & (df['loss_penalty'] == 0)]
        only_loss = df[(df['win_bonus'] == 0) & (df['loss_penalty'] < 0)]
        both_zero = df[(df['win_bonus'] == 0) & (df['loss_penalty'] == 0)]
        
        print(f"   Apenas Win Bonus: {len(only_win)} casos ({len(only_win)/len(df)*100:.1f}%)")
        print(f"   Apenas Loss Penalty: {len(only_loss)} casos ({len(only_loss)/len(df)*100:.1f}%)")
        print(f"   Ambos Zero: {len(both_zero)} casos ({len(both_zero)/len(df)*100:.1f}%)")
        
        return correlation
    
    def investigate_pnl_direct_weakness(self):
        """Investigar por que PnL direto tem contribuição baixa"""
        print("\n🔍 INVESTIGAÇÃO: PnL DIRETO BAIXA CONTRIBUIÇÃO")
        print("=" * 60)
        
        # Testar diferentes magnitudes de PnL
        pnl_values = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
        
        print("📊 ANÁLISE DE MAGNITUDE:")
        print("PnL%     | PnL_Direct | Win_Bonus | Total    | %PnL_Direct")
        print("-" * 55)
        
        for pnl in pnl_values:
            # Calcular componentes
            pnl_direct = pnl * self.calculator.base_weights['pnl_direct']  # 3.0
            win_bonus = self.calculator.base_weights['win_bonus']  # 0.5
            total = pnl_direct + win_bonus
            
            pnl_percentage = pnl_direct / total * 100 if total > 0 else 0
            
            print(f"{pnl*100:5.1f}%   | {pnl_direct:8.4f}   | {win_bonus:7.3f}   | {total:6.4f}   | {pnl_percentage:7.1f}%")
            
        print("\n🔍 DIAGNÓSTICO:")
        
        # Para trades pequenos (0.1% - 1%), win_bonus domina
        small_pnl = 0.001
        small_pnl_direct = small_pnl * 3.0  # 0.003
        small_win_bonus = 0.5
        small_total = small_pnl_direct + small_win_bonus  # 0.503
        small_contrib = small_pnl_direct / small_total * 100  # 0.6%
        
        print(f"   Trade pequeno (0.1%): PnL_direct contribui apenas {small_contrib:.1f}%")
        print(f"   Causa: Win_bonus fixo (0.5) >> PnL_direct variável (0.003)")
        
        # Para trades médios (2%)
        med_pnl = 0.02
        med_pnl_direct = med_pnl * 3.0  # 0.06
        med_win_bonus = 0.5
        med_total = med_pnl_direct + med_win_bonus  # 0.56
        med_contrib = med_pnl_direct / med_total * 100  # 10.7%
        
        print(f"   Trade médio (2%): PnL_direct contribui {med_contrib:.1f}%")
        
        # Para trades grandes (5%)
        big_pnl = 0.05
        big_pnl_direct = big_pnl * 3.0  # 0.15
        big_win_bonus = 0.5
        big_total = big_pnl_direct + big_win_bonus  # 0.65
        big_contrib = big_pnl_direct / big_total * 100  # 23.1%
        
        print(f"   Trade grande (5%): PnL_direct contribui {big_contrib:.1f}%")
        
        print("\n💡 CONCLUSÃO:")
        print("   🚨 PROBLEMA: Win_bonus/Loss_penalty são FIXOS, PnL_direct é VARIÁVEL")
        print("   📈 Para trades pequenos: bonus fixo domina")
        print("   📈 Para trades grandes: PnL_direct ganha relevância")
        print("   🎯 SOLUÇÃO: Tornar win_bonus/loss_penalty PROPORCIONAIS ao PnL")
        
        return {
            'small_contrib': small_contrib,
            'med_contrib': med_contrib, 
            'big_contrib': big_contrib
        }
    
    def test_alternative_configurations(self):
        """Testar configurações alternativas"""
        print("\n💡 TESTE DE CONFIGURAÇÕES ALTERNATIVAS")
        print("=" * 60)
        
        configs = [
            {
                "name": "ATUAL",
                "pnl_direct": 3.0,
                "win_bonus": 0.5,
                "loss_penalty": -0.5
            },
            {
                "name": "PROPORCIONAL V1", 
                "pnl_direct": 3.0,
                "win_bonus": "proportional_0.2",  # 20% do PnL como bonus
                "loss_penalty": "proportional_-0.2"  # 20% do PnL como penalty
            },
            {
                "name": "PROPORCIONAL V2",
                "pnl_direct": 4.0,
                "win_bonus": "proportional_0.1", 
                "loss_penalty": "proportional_-0.1"
            },
            {
                "name": "APENAS PnL",
                "pnl_direct": 4.0,
                "win_bonus": 0.0,
                "loss_penalty": 0.0
            }
        ]
        
        test_pnls = [0.001, 0.01, 0.02, 0.05, -0.001, -0.01, -0.02, -0.05]
        
        for config in configs:
            print(f"\n📋 CONFIGURAÇÃO: {config['name']}")
            print("PnL%     | PnL_Direct | Bonus/Pen | Total    | %PnL_Direct")
            print("-" * 55)
            
            total_pnl_contrib = []
            
            for pnl in test_pnls:
                # PnL Direct
                pnl_direct = pnl * config['pnl_direct']
                
                # Bonus/Penalty
                if isinstance(config['win_bonus'], str) and "proportional" in config['win_bonus']:
                    factor = float(config['win_bonus'].split('_')[1])
                    bonus_penalty = abs(pnl) * factor if pnl > 0 else 0
                elif isinstance(config['loss_penalty'], str) and "proportional" in config['loss_penalty']:
                    factor = float(config['loss_penalty'].split('_')[1])
                    bonus_penalty = abs(pnl) * factor if pnl < 0 else 0
                else:
                    if pnl > 0:
                        bonus_penalty = config['win_bonus']
                    elif pnl < 0:
                        bonus_penalty = config['loss_penalty']
                    else:
                        bonus_penalty = 0
                
                total = pnl_direct + bonus_penalty
                pnl_contrib = abs(pnl_direct) / abs(total) * 100 if abs(total) > 0.0001 else 0
                total_pnl_contrib.append(pnl_contrib)
                
                print(f"{pnl*100:5.1f}%   | {pnl_direct:8.4f}   | {bonus_penalty:7.3f}   | {total:6.4f}   | {pnl_contrib:7.1f}%")
            
            avg_pnl_contrib = np.mean(total_pnl_contrib)
            print(f"   MÉDIA %PnL_Direct: {avg_pnl_contrib:.1f}%")
            
            if avg_pnl_contrib > 60:
                print("   ✅ EXCELENTE: PnL domina o sistema")
            elif avg_pnl_contrib > 40:
                print("   ✅ BOM: PnL tem peso significativo") 
            elif avg_pnl_contrib > 20:
                print("   ⚠️ FRACO: PnL tem peso baixo")
            else:
                print("   ❌ CRÍTICO: PnL quase irrelevante")
    
    def generate_perfect_config_recommendation(self):
        """Gerar recomendação de configuração perfeita"""
        print("\n🎯 RECOMENDAÇÃO DE CONFIGURAÇÃO PERFEITA")
        print("=" * 60)
        
        print("📋 ANÁLISE DOS PROBLEMAS IDENTIFICADOS:")
        print("   1. ❌ Win_bonus/Loss_penalty FIXOS dominam trades pequenos")
        print("   2. ❌ Correlação perfeita (100%) entre win_bonus e loss_penalty")
        print("   3. ❌ PnL_direct tem contribuição baixa (0.1-23%)")
        print("   4. ❌ Sistema não é proporcional ao risco/magnitude")
        
        print("\n💡 SOLUÇÃO SISTEMÁTICA:")
        
        # Configuração recomendada
        recommended_config = {
            "pnl_direct": 5.0,  # Aumentado para dominar
            "win_bonus_factor": 0.1,  # 10% do PnL absoluto como bonus
            "loss_penalty_factor": -0.1,  # 10% do PnL absoluto como penalty
            "position_sizing_bonus": 0.2,  # Reduzido
            "drawdown_penalty": -0.1,  # Reduzido
            "sharpe_ratio_bonus": 0.3,  # Reduzido
            "win_rate_bonus": 0.2,  # Reduzido
            "consistency_bonus": 0.2   # Reduzido
        }
        
        print("🎯 CONFIGURAÇÃO PERFEITA RECOMENDADA:")
        for comp, value in recommended_config.items():
            print(f"   {comp}: {value}")
        
        print("\n📊 EXEMPLO DE CÁLCULO COM NOVA CONFIGURAÇÃO:")
        print("   Trade 2% lucro:")
        pnl = 0.02
        pnl_direct = pnl * 5.0  # 0.10
        win_bonus = abs(pnl) * 0.1  # 0.002
        total = pnl_direct + win_bonus  # 0.102
        pnl_contrib = pnl_direct / total * 100  # 98%
        
        print(f"     PnL_direct: {pnl_direct:.4f} ({pnl_contrib:.1f}%)")
        print(f"     Win_bonus: {win_bonus:.4f} ({win_bonus/total*100:.1f}%)")
        print(f"     ✅ PnL domina com {pnl_contrib:.1f}%!")
        
        print("\n   Trade 2% perda:")
        pnl = -0.02
        pnl_direct = pnl * 5.0  # -0.10
        loss_penalty = abs(pnl) * -0.1  # -0.002
        total = pnl_direct + loss_penalty  # -0.102
        pnl_contrib = abs(pnl_direct) / abs(total) * 100  # 98%
        
        print(f"     PnL_direct: {pnl_direct:.4f} ({pnl_contrib:.1f}%)")
        print(f"     Loss_penalty: {loss_penalty:.4f} ({abs(loss_penalty)/abs(total)*100:.1f}%)")
        print(f"     ✅ PnL domina com {pnl_contrib:.1f}%!")
        
        return recommended_config
    
    def run_complete_investigation(self):
        """Executar investigação completa"""
        print("🔬 INVESTIGAÇÃO SISTEMÁTICA E MINUCIOSA")
        print("🎯 OBJETIVO: SISTEMA DE REWARDS IMPECÁVEL")
        print("=" * 80)
        
        # 1. Análise individual de componentes
        self.analyze_individual_components()
        
        # 2. Investigação de correlação
        correlation = self.test_win_loss_correlation()
        
        # 3. Investigação PnL direto
        pnl_analysis = self.investigate_pnl_direct_weakness()
        
        # 4. Teste configurações alternativas
        self.test_alternative_configurations()
        
        # 5. Recomendação final
        perfect_config = self.generate_perfect_config_recommendation()
        
        print(f"\n🏆 INVESTIGAÇÃO COMPLETA FINALIZADA")
        print(f"📄 Relatório detalhado gerado")
        print(f"💡 Configuração perfeita recomendada")
        print(f"🚀 Próximo passo: Implementar correções")
        
        return {
            'correlation_analysis': correlation,
            'pnl_analysis': pnl_analysis,
            'perfect_config': perfect_config
        }

def main():
    """Executar investigação completa"""
    investigator = RewardSystemForensics()
    results = investigator.run_complete_investigation()
    
    # Salvar resultados
    import json
    with open('investigacao_reward_completa.json', 'w') as f:
        # Converter numpy types para JSON
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                   for k, v in value.items()}
            else:
                json_results[key] = float(value) if isinstance(value, (np.floating, np.integer)) else value
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Resultados salvos em 'investigacao_reward_completa.json'")

if __name__ == "__main__":
    main()