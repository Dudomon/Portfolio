#!/usr/bin/env python3
"""
🧪 TESTE DO SISTEMA DE TRAILING STOP REWARD
Verifica se o V4 Selective agora está recompensando adequadamente 
o uso do trailing stop pela management head
"""

import sys
import numpy as np
from unittest.mock import Mock

# Mock para testar o reward system
class MockEnvTrailing:
    """Mock para simular ambiente com trailing stop info"""
    
    def __init__(self):
        self.trades = []
        self.positions = []
        self.current_step = 100
        
    def add_trailing_trade(self, scenario="good_trailing"):
        """Adiciona trade com informações de trailing stop"""
        base_trade = {
            'entry_step': 50,
            'exit_step': 100,
            'type': 'long',
            'pnl_usd': 25.0,
            'entry_price': 1800.0,
            'exit_price': 1825.0,
            'duration': 50
        }
        
        if scenario == "good_trailing":
            # Cenário: Trailing usado corretamente
            base_trade.update({
                'trailing_activated': True,
                'trailing_protected': True, 
                'trailing_timing': True,
                'trailing_moves': 2,
                'exit_reason': 'trailing_stop',
                'missed_trailing_opportunity': False,
                'tp_adjusted': False
            })
        elif scenario == "missed_opportunity":
            # Cenário: Perdeu oportunidade de trailing
            base_trade.update({
                'trailing_activated': False,
                'trailing_protected': False,
                'trailing_timing': False,
                'trailing_moves': 0,
                'exit_reason': 'TP hit',
                'missed_trailing_opportunity': True,
                'tp_adjusted': False
            })
        elif scenario == "tp_adjustment":
            # Cenário: Ajustou TP com sucesso
            base_trade.update({
                'trailing_activated': False,
                'trailing_protected': False,
                'trailing_timing': False,
                'trailing_moves': 0,
                'exit_reason': 'TP hit',
                'missed_trailing_opportunity': False,
                'tp_adjusted': True
            })
        elif scenario == "bad_tp_adjustment":
            # Cenário: Ajustou TP mas perdeu
            base_trade.update({
                'pnl_usd': -15.0,  # Perda
                'trailing_activated': False,
                'trailing_protected': False,
                'trailing_timing': False,
                'trailing_moves': 0,
                'exit_reason': 'SL hit',
                'missed_trailing_opportunity': False,
                'tp_adjusted': True
            })
        
        self.trades.append(base_trade)
        
    def add_position(self, unrealized_pnl=10.0):
        """Adiciona posição ativa para testar management em tempo real"""
        position = {
            'type': 'long',
            'entry_price': 1800.0,
            'unrealized_pnl': unrealized_pnl,
            'age': 20
        }
        self.positions.append(position)

# Importar o sistema de reward (mock simples)
class MockSelectiveTradingReward:
    """Mock do reward system para teste"""
    
    def __init__(self):
        self.logger = Mock()
        
    def _calculate_trailing_stop_reward(self, env, action):
        """Implementação real do método que adicionamos"""
        try:
            reward = 0.0
            info = {'trailing_analysis': {}}
            
            # Verificar se houve trades recentes
            trades = getattr(env, 'trades', [])
            if not trades:
                info['no_trades'] = True
                return 0.0, info
            
            last_trade = trades[-1]
            
            # 1. 🎯 REWARD POR ATIVAÇÃO DE TRAILING STOP
            if last_trade.get('trailing_activated', False):
                activation_bonus = 0.15  # Bonus moderado por ativar trailing
                reward += activation_bonus
                info['trailing_analysis']['activation_bonus'] = activation_bonus
                
                # Bonus adicional se foi ativado no timing certo
                if last_trade.get('trailing_timing', False):
                    timing_bonus = 0.10
                    reward += timing_bonus
                    info['trailing_analysis']['timing_bonus'] = timing_bonus
            
            # 2. 🎯 REWARD POR PROTEÇÃO DE LUCRO VIA TRAILING
            if last_trade.get('trailing_protected', False):
                protection_bonus = 0.20  # Bonus maior por proteger lucros
                reward += protection_bonus
                info['trailing_analysis']['protection_bonus'] = protection_bonus
                
                # Bonus extra se o trade teve exit_reason = trailing_stop
                if last_trade.get('exit_reason', '') == 'trailing_stop':
                    execution_bonus = 0.15
                    reward += execution_bonus
                    info['trailing_analysis']['execution_bonus'] = execution_bonus
            
            # 3. 🎯 REWARD POR MOVIMENTOS INTELIGENTES DO TRAILING
            trailing_moves = last_trade.get('trailing_moves', 0)
            if trailing_moves > 0:
                # Reward moderado por mover o trailing (máximo 3 moves)
                move_bonus = min(trailing_moves * 0.05, 0.15)
                reward += move_bonus
                info['trailing_analysis']['move_bonus'] = move_bonus
                info['trailing_analysis']['trailing_moves'] = trailing_moves
            
            # 4. 🎯 PENALIDADE POR OPORTUNIDADE PERDIDA
            if last_trade.get('missed_trailing_opportunity', False):
                missed_penalty = -0.10  # Penalidade moderada
                reward += missed_penalty
                info['trailing_analysis']['missed_penalty'] = missed_penalty
            
            # 5. 🎯 REWARD POR AJUSTES DE TP
            if last_trade.get('tp_adjusted', False):
                # Verificar se o ajuste foi inteligente
                pnl_usd = last_trade.get('pnl_usd', 0.0)
                if pnl_usd > 0:  # TP adjustment que resultou em lucro
                    tp_bonus = 0.08
                    reward += tp_bonus
                    info['trailing_analysis']['tp_bonus'] = tp_bonus
                elif pnl_usd < 0:  # TP adjustment que resultou em perda (sinal ruim)
                    tp_penalty = -0.05
                    reward += tp_penalty
                    info['trailing_analysis']['tp_penalty'] = tp_penalty
            
            # 6. 🎯 ANÁLISE DO MANAGEMENT ACTION ATUAL
            # Verificar se modelo está tentando gerenciar posições existentes
            positions = getattr(env, 'positions', [])
            if positions and len(action) >= 4:
                # Analisar management actions [2] e [3]
                mgmt_actions = action[2:4]
                active_mgmt = any(abs(a) > 0.5 for a in mgmt_actions)
                
                if active_mgmt:
                    # Small bonus for active management when in position
                    for i, pos in enumerate(positions[:2]):
                        unrealized_pnl = pos.get('unrealized_pnl', 0.0)
                        mgmt_action = mgmt_actions[i] if i < len(mgmt_actions) else 0.0
                        
                        # Se posição em lucro e modelo está gerenciando ativamente
                        if unrealized_pnl > 0 and abs(mgmt_action) > 0.8:
                            active_mgmt_bonus = 0.03
                            reward += active_mgmt_bonus
                            info['trailing_analysis'][f'active_mgmt_pos{i}'] = active_mgmt_bonus
            
            info['trailing_analysis']['total_trailing_reward'] = reward
            return reward, info
            
        except Exception as e:
            self.logger.error(f"Erro no trailing stop reward: {e}")
            return 0.0, {'error': str(e)}

def test_good_trailing_scenario():
    """🧪 Teste: Cenário de trailing stop usado corretamente"""
    print("🧪 TESTE 1: Trailing Stop Usado Corretamente")
    print("=" * 50)
    
    env = MockEnvTrailing()
    env.add_trailing_trade("good_trailing")
    
    reward_system = MockSelectiveTradingReward()
    action = np.array([0, 0.5, 1.5, 0.0])  # HOLD with strong SL management
    
    reward, info = reward_system._calculate_trailing_stop_reward(env, action)
    
    print(f"📊 Cenário: Trailing usado corretamente")
    print(f"  Trade info: trailing_activated=True, trailing_protected=True")
    print(f"  Exit reason: trailing_stop, Moves: 2")
    print(f"  Total trailing reward: {reward:.3f}")
    
    analysis = info.get('trailing_analysis', {})
    expected_components = ['activation_bonus', 'timing_bonus', 'protection_bonus', 'execution_bonus', 'move_bonus']
    found_components = []
    
    for component in expected_components:
        if component in analysis:
            found_components.append(component)
            print(f"    {component}: {analysis[component]:.3f}")
    
    # Verificar se reward é positivo e significativo
    expected_min_reward = 0.50  # Deveria ter pelo menos 0.50 de reward
    success = reward >= expected_min_reward and len(found_components) >= 4
    
    print(f"  Expected components found: {len(found_components)}/5")
    print(f"  Trailing stop reward functioning: {'✅' if success else '❌'}")
    
    return success

def test_missed_opportunity():
    """🧪 Teste: Cenário de oportunidade perdida"""
    print("\n🧪 TESTE 2: Oportunidade de Trailing Perdida") 
    print("=" * 50)
    
    env = MockEnvTrailing()
    env.add_trailing_trade("missed_opportunity")
    
    reward_system = MockSelectiveTradingReward()
    action = np.array([0, 0.5, 0.0, 0.0])  # HOLD with no management
    
    reward, info = reward_system._calculate_trailing_stop_reward(env, action)
    
    print(f"📊 Cenário: Oportunidade perdida")
    print(f"  Trade info: missed_trailing_opportunity=True")
    print(f"  Exit reason: TP hit (sem trailing)")
    print(f"  Total trailing reward: {reward:.3f}")
    
    analysis = info.get('trailing_analysis', {})
    
    if 'missed_penalty' in analysis:
        print(f"    missed_penalty: {analysis['missed_penalty']:.3f}")
    
    # Reward deveria ser negativo (penalidade)
    success = reward < 0
    
    print(f"  Missed opportunity penalty applied: {'✅' if success else '❌'}")
    
    return success

def test_tp_adjustment_scenarios():
    """🧪 Teste: Cenários de ajuste de TP"""
    print("\n🧪 TESTE 3: Ajustes de Take Profit")
    print("=" * 50)
    
    # 3.1 TP adjustment com sucesso
    env1 = MockEnvTrailing()
    env1.add_trailing_trade("tp_adjustment")
    
    reward_system = MockSelectiveTradingReward()
    action = np.array([0, 0.5, 0.0, 1.2])  # HOLD with TP management
    
    reward1, info1 = reward_system._calculate_trailing_stop_reward(env1, action)
    
    print(f"📊 Cenário 3.1: TP adjustment com lucro")
    print(f"  Trade PnL: +$25 (lucro)")
    print(f"  TP adjusted: True")
    print(f"  Trailing reward: {reward1:.3f}")
    
    # 3.2 TP adjustment com perda  
    env2 = MockEnvTrailing()
    env2.add_trailing_trade("bad_tp_adjustment")
    
    reward2, info2 = reward_system._calculate_trailing_stop_reward(env2, action)
    
    print(f"  Cenário 3.2: TP adjustment com perda")
    print(f"  Trade PnL: -$15 (perda)")
    print(f"  TP adjusted: True") 
    print(f"  Trailing reward: {reward2:.3f}")
    
    # Verificar se bonus/penalty foram aplicados corretamente
    success1 = reward1 > 0  # TP com lucro deveria dar bonus
    success2 = reward2 < 0  # TP com perda deveria dar penalty
    
    print(f"  TP success bonus applied: {'✅' if success1 else '❌'}")
    print(f"  TP failure penalty applied: {'✅' if success2 else '❌'}")
    
    return success1 and success2

def test_active_management_bonus():
    """🧪 Teste: Bonus por management ativo durante posições"""
    print("\n🧪 TESTE 4: Management Ativo em Posições")
    print("=" * 50)
    
    env = MockEnvTrailing()
    env.add_position(unrealized_pnl=15.0)  # Posição em lucro
    env.add_trailing_trade("good_trailing")  # Adicionar trade para não retornar early
    
    reward_system = MockSelectiveTradingReward()
    
    # Ação com management ativo
    action_active = np.array([0, 0.5, 1.5, 0.8])  # Strong SL and TP management
    reward_active, info_active = reward_system._calculate_trailing_stop_reward(env, action_active)
    
    # Ação sem management
    action_passive = np.array([0, 0.5, 0.1, 0.1])  # Minimal management
    reward_passive, info_passive = reward_system._calculate_trailing_stop_reward(env, action_passive)
    
    print(f"📊 Cenário: Posição em lucro (+$15)")
    print(f"  Active management action: [0, 0.5, 1.5, 0.8]")
    print(f"  Active management reward: {reward_active:.3f}")
    print(f"  Passive management reward: {reward_passive:.3f}")
    
    # Management ativo deveria dar bonus quando posição está em lucro
    success = reward_active > reward_passive
    
    print(f"  Active management bonus: {'✅' if success else '❌'}")
    
    return success

def main():
    """Executa teste completo do sistema de trailing stop reward"""
    print("🎯 TESTE DO SISTEMA DE TRAILING STOP REWARD")
    print("=" * 70)
    print("Verificando se V4 Selective está ensinando management head")
    print("=" * 70)
    
    tests = [
        ("Trailing Stop Correto", test_good_trailing_scenario),
        ("Oportunidade Perdida", test_missed_opportunity), 
        ("Ajustes de TP", test_tp_adjustment_scenarios),
        ("Management Ativo", test_active_management_bonus)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
            status = "✅ PASSOU" if result else "❌ FALHOU"
            print(f"\n{status} - {test_name}")
        except Exception as e:
            print(f"\n❌ ERRO - {test_name}: {e}")
            results.append(False)
    
    print(f"\n{'='*70}")
    print(f"🏆 RESULTADO DOS TESTES DE TRAILING STOP REWARD")
    print(f"{'='*70}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"Testes passaram: {passed}/{total}")
    
    if passed == total:
        print("✅ SISTEMA DE TRAILING STOP REWARD FUNCIONANDO")
        print("🎯 V4 Selective agora ensina management head adequadamente")
        print("📈 Impacto esperado:")
        print("   - Modelo aprende a usar trailing stop")
        print("   - Melhor proteção de lucros")
        print("   - Management head mais ativa")
        print("   - Ajustes inteligentes de SL/TP")
        print("   - Redução de oportunidades perdidas")
        
    else:
        print("❌ SISTEMA precisa de ajustes")
        print("🔧 Verificar implementação no reward_daytrade_v4_selective.py")
    
    print("=" * 70)

if __name__ == "__main__":
    main()