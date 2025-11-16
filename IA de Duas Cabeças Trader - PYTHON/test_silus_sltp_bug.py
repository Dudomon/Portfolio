#!/usr/bin/env python3
"""
🔬 TESTE CONTROLADO: Verificar se modelo aprendeu ajustes SL/TP
Objetivo: Confirmar se bug no ambiente impediu aprendizado de ajustes dinâmicos
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from silus import TradingEnv
import numpy as np
import pandas as pd

def test_sltp_adjustment_bug():
    """🧪 Teste controlado para verificar bug de ajustes SL/TP"""

    print("🔬 [TESTE] Iniciando teste controlado de ajustes SL/TP...")
    print("🎯 [OBJETIVO] Verificar se bug no threshold impediu aprendizado\n")

    # Simular dados mínimos (apenas para testar lógica)
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
        'open_1m': np.random.uniform(3650, 3670, 100),
        'high_1m': np.random.uniform(3651, 3671, 100),
        'low_1m': np.random.uniform(3649, 3669, 100),
        'close_1m': np.random.uniform(3650, 3670, 100),
        'volume_1m': np.random.uniform(1000, 5000, 100)
    })

    # Criar ambiente de teste
    env = TradingEnv(test_data)

    # Setup ambiente
    env.current_step = 50
    env.balance = 10000
    env.positions = []

    print("📊 [SETUP] Ambiente configurado com dados sintéticos")

    # ==================== TESTE 1: CRIAR POSIÇÃO MANUALMENTE ====================
    print("\n🧪 [TESTE 1] Criando posição manual para testar ajustes...")

    # Simular posição existente
    test_position = {
        'entry_step': 45,
        'entry_price': 3660.0,
        'type': 'long',
        'lot_size': 0.1,  # Adicionado lot_size que estava faltando
        'sl': 3650.0,  # SL inicial a 10 pontos
        'tp': 3670.0,  # TP inicial a 10 pontos
        'trailing_activated': False,
        'tp_adjusted': False
    }

    env.positions = [test_position]
    current_price = 3665.0  # Posição em lucro

    print(f"✅ [POSIÇÃO] Criada: Entry={test_position['entry_price']}, SL={test_position['sl']}, TP={test_position['tp']}")
    print(f"💰 [PREÇO] Atual: {current_price} (lucro de +5 pontos)")

    # ==================== TESTE 2: TESTAR FUNÇÃO PROCESS_DYNAMIC_TRAILING_STOP ====================
    print("\n🧪 [TESTE 2] Testando função _process_dynamic_trailing_stop do silus...")

    # Simular ajustes do modelo
    sl_adjust = 0.5   # Modelo quer ajustar SL
    tp_adjust = -0.5  # Modelo quer ajustar TP

    print(f"🤖 [MODELO] Produz: sl_adjust={sl_adjust}, tp_adjust={tp_adjust}")

    # Testar função real do ambiente
    try:
        result = env._process_dynamic_trailing_stop(
            test_position, sl_adjust, tp_adjust, current_price, 0
        )

        print(f"✅ [SILUS] Função executou com sucesso!")
        print(f"🎯 [RESULTADO] tp_adjusted: {result.get('tp_adjusted', False)}")
        print(f"📊 [DETALHES] action_taken: {result.get('action_taken', False)}")

        if result.get('tp_adjusted', False):
            tp_info = result.get('tp_info', {})
            print(f"🎯 [TP INFO] {tp_info}")

    except Exception as e:
        print(f"❌ [ERRO] Função falhou: {e}")

    # ==================== TESTE 3: COMPARAR ANTES E DEPOIS ====================
    print("\n🧪 [TESTE 3] Comparando comportamento antes/depois da correção...")

    # Simular threshold original (bugado)
    bug_threshold = abs(tp_adjust) > 0.5  # FALSE para ±0.5
    fixed_threshold = abs(tp_adjust) >= 0.5  # TRUE para ±0.5

    print(f"❌ [ANTES] Threshold > 0.5: {bug_threshold} (BLOQUEAVA)")
    print(f"✅ [DEPOIS] Threshold >= 0.5: {fixed_threshold} (PERMITE)")

    # ==================== TESTE 4: VERIFICAR ESTADO DA POSIÇÃO ====================
    print("\n🧪 [TESTE 4] Verificando se ajustes foram aplicados...")

    print(f"📍 [POSIÇÃO FINAL]:")
    print(f"   SL: {test_position.get('sl', 'N/A')}")
    print(f"   TP: {test_position.get('tp', 'N/A')}")
    print(f"   TP Adjusted: {test_position.get('tp_adjusted', False)}")

    # ==================== RESULTADO FINAL ====================
    print("\n" + "="*60)
    print("🎯 [CONCLUSÃO DO TESTE CONTROLADO]")
    print("="*60)

    # Determinar se correção funcionou
    silus_fixed = result.get('tp_adjusted', False) if 'result' in locals() else False

    if silus_fixed:
        print("✅ SUCESSO: Silus corrigido funciona corretamente!")
        print("🧠 PRÓXIMO TREINO: Modelo poderá aprender ajustes dinâmicos")
        print("🎯 AMBIENTE: Pronto para treinar funcionalidade SL/TP")
        conclusion = {
            'bug_fixed': True,
            'silus_functional': True,
            'robot_aligned': True,
            'ready_for_retraining': True
        }
    else:
        print("❌ PROBLEMA: Ainda há issues no ambiente")
        print("🔧 NECESSÁRIO: Investigação adicional")
        conclusion = {
            'bug_fixed': False,
            'silus_functional': False,
            'robot_aligned': True,
            'ready_for_retraining': False
        }

    return conclusion

if __name__ == "__main__":
    result = test_sltp_adjustment_bug()

    print(f"\n🔬 [TESTE COMPLETO] Resultado: {result}")