#!/usr/bin/env python3
"""
🔬 TESTE DO FLUXO DE AÇÃO - Testar Entry: 0.57, Confidence: 1.00
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

sys.path.append("Robo Ander")
from Robot_1min import TradingRobotV7
import numpy as np

def test_action_flow():
    """Testar o fluxo completo de ação"""
    print("🔬 INICIANDO TESTE DO FLUXO DE AÇÃO")
    print("=" * 50)
    
    # Criar robot
    robot = TradingRobotV7()
    robot.mt5_connected = True  # Simular MT5 conectado
    robot.max_positions = 2
    
    # Simular predição problemática: Entry: 0.57, Confidence: 1.00
    test_action = np.array([0.57, 1.00, -1.00, 1.00])
    
    print(f"📊 AÇÃO DE TESTE: {test_action}")
    print(f"   Entry: {test_action[0]}")
    print(f"   Confidence: {test_action[1]}")
    print(f"   Pos1_Mgmt: {test_action[2]}")
    print(f"   Pos2_Mgmt: {test_action[3]}")
    print()
    
    # Teste 1: _process_v7_action
    print("🧪 TESTE 1: _process_v7_action()")
    try:
        result = robot._process_v7_action(test_action)
        print(f"✅ Processamento OK!")
        print(f"   entry_decision: {result['entry_decision']}")
        print(f"   action_name: {result['action_name']}")
        print(f"   entry_confidence: {result['entry_confidence']}")
        print()
        
        # Verificar mapeamento
        raw_decision = test_action[0]
        expected_decision = 1 if 0.33 <= raw_decision < 0.67 else (0 if raw_decision < 0.33 else 2)
        print(f"🎯 VERIFICAÇÃO MAPEAMENTO:")
        print(f"   Raw Decision: {raw_decision}")
        print(f"   Esperado: {expected_decision} ({'HOLD' if expected_decision == 0 else 'LONG' if expected_decision == 1 else 'SHORT'})")
        print(f"   Obtido: {result['entry_decision']} ({result['action_name']})")
        
        if result['entry_decision'] == expected_decision:
            print("   ✅ Mapeamento CORRETO!")
        else:
            print("   ❌ Mapeamento INCORRETO!")
        print()
        
    except Exception as e:
        print(f"❌ ERRO no _process_v7_action: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Teste 2: Simular _execute_v7_decision
    print("🧪 TESTE 2: _execute_v7_decision() (simulado)")
    try:
        action_name = result['action_name']
        entry_confidence = result['entry_confidence']
        
        print(f"   Recebendo: action_name='{action_name}', confidence={entry_confidence}")
        
        if action_name in ['BUY', 'SELL']:
            print(f"   ✅ Deveria executar {action_name}!")
            print(f"   📋 Logs esperados:")
            print(f"      🧠 [V7-DECISION] {action_name} | Confidence: {entry_confidence:.3f}")
            print(f"      🎯 [V7-ATTEMPT] Tentando executar {action_name}")
            print(f"      🟢 [SINAL COMPRA] ou 🔴 [SINAL VENDA]")
        else:
            print(f"   ⭕ HOLD - não executará trade")
        print()
            
    except Exception as e:
        print(f"❌ ERRO na simulação _execute_v7_decision: {e}")
        return
    
    # Teste 3: Verificar filtros
    print("🧪 TESTE 3: Verificação de filtros")
    
    # Filtro de confidence
    confidence_pass = entry_confidence >= 0.8
    print(f"   🎯 Filtro Confidence (≥0.8): {entry_confidence} -> {'✅ PASS' if confidence_pass else '❌ FAIL'}")
    
    # Simular posições (assumindo sem posições no início)
    positions_count = 0
    max_positions = 2
    position_limit_pass = positions_count < max_positions
    print(f"   🔒 Limite Posições ({positions_count}/{max_positions}): {'✅ PASS' if position_limit_pass else '❌ FAIL'}")
    
    # Conclusão
    print()
    print("🎯 CONCLUSÃO:")
    if result['action_name'] == 'BUY' and confidence_pass and position_limit_pass:
        print("   ✅ DEVERIA EXECUTAR COMPRA!")
        print("   ❌ Se não executou no robô real, há bug adicional!")
    elif result['action_name'] == 'HOLD':
        print("   ⭕ Processamento gerou HOLD - verificar mapeamento")
    else:
        print("   ❌ Bloqueado por filtros")
    
    print("=" * 50)

if __name__ == "__main__":
    test_action_flow()