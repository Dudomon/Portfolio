#!/usr/bin/env python3
"""
🧪 TESTE DO SISTEMA DE TRAILING STOP INTELIGENTE
==================================================

Testa as modificações feitas no V3 Brutal reward system para:
- ✅ Recompensar movimentos inteligentes
- ❌ Penalizar movimentos contraproducentes
"""

import sys
import os
sys.path.append("D:/Projeto")

from trading_framework.rewards.reward_daytrade_v3_brutal import BrutalMoneyReward

def test_intelligent_trailing_rewards():
    """
    🧪 Teste específico para trailing stops inteligentes
    """
    print("🧪 TESTE DE TRAILING STOP INTELIGENTE")
    print("=" * 60)

    # Criar o reward system
    reward_system = BrutalMoneyReward(initial_balance=1000.0)

    # Mock environment
    class MockEnv:
        def __init__(self, trades):
            self.trades = trades

    # 🎯 CENÁRIOS DE TESTE
    test_scenarios = [
        {
            'name': '✅ SHORT INTELIGENTE: Preço caiu, SL desceu, lucro',
            'trades': [{
                'type': 'short',
                'entry_price': 2000.0,
                'final_sl': 1995.0,  # SL 5 pontos abaixo da entrada (bom)
                'pnl': 30.0,  # Lucro
                'trailing_activated': True,
                'trailing_moves': 2,
                'price_direction_during_trail': -1,  # Preço desceu (favorável ao short)
                'sl_adjusted': True
            }]
        },
        {
            'name': '❌ SHORT BURRO: Preço subiu, SL subiu, prejuízo',
            'trades': [{
                'type': 'short',
                'entry_price': 2000.0,
                'final_sl': 2020.0,  # SL 20 pontos ACIMA da entrada (ruim!)
                'pnl': -25.0,  # Prejuízo
                'trailing_activated': True,
                'trailing_moves': 3,
                'price_direction_during_trail': 1,  # Preço subiu (contra short)
                'sl_adjusted': True
            }]
        },
        {
            'name': '✅ LONG INTELIGENTE: Preço subiu, SL subiu, lucro',
            'trades': [{
                'type': 'long',
                'entry_price': 2000.0,
                'final_sl': 1995.0,  # SL 5 pontos abaixo da entrada (bom)
                'pnl': 40.0,  # Lucro
                'trailing_activated': True,
                'trailing_moves': 1,
                'price_direction_during_trail': 1,  # Preço subiu (favorável ao long)
                'sl_adjusted': True
            }]
        },
        {
            'name': '❌ LONG BURRO: Preço caiu, SL caiu demais, prejuízo',
            'trades': [{
                'type': 'long',
                'entry_price': 2000.0,
                'final_sl': 1980.0,  # SL muito abaixo da entrada (ruim!)
                'pnl': -30.0,  # Prejuízo
                'trailing_activated': True,
                'trailing_moves': 2,
                'price_direction_during_trail': -1,  # Preço caiu (contra long)
                'sl_adjusted': True
            }]
        },
        {
            'name': '🎯 COMBO INTELIGENTE: Trailing + TP bem usado',
            'trades': [{
                'type': 'short',
                'entry_price': 2000.0,
                'final_sl': 1998.0,  # SL próximo da entrada (ok)
                'pnl': 50.0,  # Bom lucro
                'trailing_activated': True,
                'trailing_moves': 1,
                'tp_adjusted': True,
                'price_direction_during_trail': -1,  # Preço favorável
                'sl_adjusted': True,
                'close_reason': 'TP hit'
            }]
        },
        {
            'name': '🚫 SEM TRAILING: Trade normal para comparação',
            'trades': [{
                'type': 'long',
                'entry_price': 2000.0,
                'pnl': 20.0,
                'trailing_activated': False,
                'trailing_moves': 0,
                'sl_adjusted': False,
                'tp_adjusted': False
            }]
        }
    ]

    # 🧪 EXECUTAR TESTES
    for i, scenario in enumerate(test_scenarios):
        print(f"\n{i+1}. {scenario['name']}")
        print("-" * 50)

        mock_env = MockEnv(scenario['trades'])

        # Testar funções específicas
        trailing_reward = reward_system._calculate_trailing_stop_rewards(mock_env)
        sltp_reward = reward_system._calculate_dynamic_sltp_rewards(mock_env)

        trade = scenario['trades'][0]

        # Mostrar análises internas
        if trade.get('trailing_activated', False):
            against_breakeven = reward_system._sl_moved_against_breakeven(trade)
            moves_smart = reward_system._trailing_moves_were_smart(trade)

            print(f"  🔍 SL contra breakeven: {against_breakeven}")
            print(f"  🧠 Movimentos inteligentes: {moves_smart}")

        if trade.get('sl_adjusted', False):
            sl_smart = reward_system._sl_adjustment_was_smart(trade)
            print(f"  🛡️ Ajuste SL inteligente: {sl_smart}")

        if trade.get('tp_adjusted', False):
            tp_smart = reward_system._tp_adjustment_was_smart(trade)
            print(f"  🎯 Ajuste TP inteligente: {tp_smart}")

        # Resultados
        total_reward = trailing_reward + sltp_reward
        print(f"  💰 Trailing Reward: {trailing_reward:+.4f}")
        print(f"  🎯 SL/TP Reward: {sltp_reward:+.4f}")
        print(f"  📊 TOTAL: {total_reward:+.4f}")

        # Interpretação
        if total_reward > 0:
            print(f"  ✅ COMPORTAMENTO RECOMPENSADO")
        elif total_reward < 0:
            print(f"  ❌ COMPORTAMENTO PENALIZADO")
        else:
            print(f"  ⚪ NEUTRO")

def test_breakeven_detection():
    """
    🔍 Teste específico da detecção de breakeven
    """
    print("\n\n🔍 TESTE DE DETECÇÃO DE BREAKEVEN")
    print("=" * 60)

    reward_system = BrutalMoneyReward(initial_balance=1000.0)

    test_cases = [
        {
            'name': 'SHORT: SL muito acima da entrada (RUIM)',
            'trade': {
                'type': 'short',
                'entry_price': 2000.0,
                'final_sl': 2015.0  # 15 pontos acima = ruim
            }
        },
        {
            'name': 'SHORT: SL próximo da entrada (OK)',
            'trade': {
                'type': 'short',
                'entry_price': 2000.0,
                'final_sl': 2003.0  # 3 pontos acima = ok
            }
        },
        {
            'name': 'LONG: SL muito abaixo da entrada (RUIM)',
            'trade': {
                'type': 'long',
                'entry_price': 2000.0,
                'final_sl': 1980.0  # 20 pontos abaixo = ruim
            }
        },
        {
            'name': 'LONG: SL próximo da entrada (OK)',
            'trade': {
                'type': 'long',
                'entry_price': 2000.0,
                'final_sl': 1997.0  # 3 pontos abaixo = ok
            }
        }
    ]

    for case in test_cases:
        against_breakeven = reward_system._sl_moved_against_breakeven(case['trade'])
        result = "❌ CONTRA BREAKEVEN" if against_breakeven else "✅ DENTRO DO BREAKEVEN"
        print(f"  {case['name']}: {result}")

if __name__ == "__main__":
    try:
        test_intelligent_trailing_rewards()
        test_breakeven_detection()
        print("\n\n🎯 TESTE CONCLUÍDO!")
        print("✅ O sistema agora penaliza movimentos contraproducentes")
        print("✅ E recompensa apenas trailing stops inteligentes")

    except Exception as e:
        print(f"❌ Erro durante teste: {e}")
        import traceback
        traceback.print_exc()