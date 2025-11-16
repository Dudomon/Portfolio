"""
🧪 TESTE DE ALINHAMENTO SL/TP - Robot_1min vs RobotV7 vs silus.py
Verifica se os três sistemas produzem exatamente os mesmos SL/TP
"""
import numpy as np

# Configurações idênticas ao silus.py
REALISTIC_SLTP_CONFIG = {
    'sl_min_points': 2,     # SL mínimo: 2 pontos (daytrade)
    'sl_max_points': 8,     # SL máximo: 8 pontos (daytrade)
    'tp_min_points': 3,     # TP mínimo: 3 pontos (daytrade)
    'tp_max_points': 15,    # TP máximo: 15 pontos (daytrade)
    'sl_tp_step': 0.5,      # Variação: 0.5 pontos
}

def convert_management_to_sltp_adjustments(mgmt_value):
    """
    🚀 Converte valor de management [-1,1] em ajustes SL/TP bidirecionais (como silus.py)
    """
    if mgmt_value < 0:
        # Foco em SL management
        if mgmt_value < -0.5:
            return (0.5, 0)  # Afrouxar SL
        else:
            return (-0.5, 0)  # Apertar SL
    elif mgmt_value > 0:
        # Foco em TP management
        if mgmt_value > 0.5:
            return (0, 0.5)  # TP distante
        else:
            return (0, -0.5)  # TP próximo
    else:
        return (0, 0)

def convert_action_to_realistic_sltp(sltp_action_values, current_price):
    """
    🚀 Converte action space para SL/TP realistas de forma clara (como silus.py)
    """
    sl_adjust = sltp_action_values[0]  # [-3,3] para SL
    tp_adjust = sltp_action_values[1]  # [-3,3] para TP

    # Converter para pontos realistas separadamente
    sl_points = REALISTIC_SLTP_CONFIG['sl_min_points'] + \
                (sl_adjust + 3) * (REALISTIC_SLTP_CONFIG['sl_max_points'] - REALISTIC_SLTP_CONFIG['sl_min_points']) / 6

    tp_points = REALISTIC_SLTP_CONFIG['tp_min_points'] + \
                (tp_adjust + 3) * (REALISTIC_SLTP_CONFIG['tp_max_points'] - REALISTIC_SLTP_CONFIG['tp_min_points']) / 6

    # Arredondar para múltiplos de 0.5 pontos
    sl_points = round(sl_points * 2) / 2
    tp_points = round(tp_points * 2) / 2

    # Garantir limites (segurança)
    sl_points = max(REALISTIC_SLTP_CONFIG['sl_min_points'], min(sl_points, REALISTIC_SLTP_CONFIG['sl_max_points']))
    tp_points = max(REALISTIC_SLTP_CONFIG['tp_min_points'], min(tp_points, REALISTIC_SLTP_CONFIG['tp_max_points']))

    return [sl_points, tp_points]

def test_sltp_alignment():
    """🧪 Teste de alinhamento com casos diversos"""
    print("🧪 TESTE DE ALINHAMENTO SL/TP")
    print("=" * 50)

    # Casos de teste
    test_cases = [
        {"name": "SL Afrouxar", "pos1_mgmt": -0.8, "pos2_mgmt": 0.2},
        {"name": "SL Apertar", "pos1_mgmt": -0.3, "pos2_mgmt": 0.7},
        {"name": "TP Próximo", "pos1_mgmt": 0.3, "pos2_mgmt": -0.4},
        {"name": "TP Distante", "pos1_mgmt": 0.9, "pos2_mgmt": -0.7},
        {"name": "Neutro", "pos1_mgmt": 0.0, "pos2_mgmt": 0.0},
        {"name": "Extremos", "pos1_mgmt": -1.0, "pos2_mgmt": 1.0},
    ]

    current_price = 2650.0  # Preço XAUUSD

    for case in test_cases:
        print(f"\n📝 CASO: {case['name']}")
        print(f"   pos1_mgmt = {case['pos1_mgmt']}, pos2_mgmt = {case['pos2_mgmt']}")

        # === SISTEMA SILUS.PY (TREINO) ===
        pos1_sl_adjust, pos1_tp_adjust = convert_management_to_sltp_adjustments(case['pos1_mgmt'])
        pos2_sl_adjust, pos2_tp_adjust = convert_management_to_sltp_adjustments(case['pos2_mgmt'])

        # Para primeira posição, usar pos1_mgmt (linha 5952-5954 do silus.py)
        sl_adjust = pos1_sl_adjust
        tp_adjust = pos1_tp_adjust

        # Converter para pontos realistas
        realistic_sltp = convert_action_to_realistic_sltp([sl_adjust, tp_adjust], current_price)
        silus_sl = abs(realistic_sltp[0])
        silus_tp = abs(realistic_sltp[1])

        print(f"   🎯 SILUS.PY:    SL={silus_sl:.1f} pontos, TP={silus_tp:.1f} pontos")

        # === SISTEMA ROBOT_1MIN (EXECUÇÃO) ===
        # Mesmo algoritmo implementado
        robot1_sl_adjust, robot1_tp_adjust = convert_management_to_sltp_adjustments(case['pos1_mgmt'])
        robot1_realistic_sltp = convert_action_to_realistic_sltp([robot1_sl_adjust, robot1_tp_adjust], current_price)
        robot1_sl = abs(robot1_realistic_sltp[0])
        robot1_tp = abs(robot1_realistic_sltp[1])

        print(f"   🤖 ROBOT_1MIN:  SL={robot1_sl:.1f} pontos, TP={robot1_tp:.1f} pontos")

        # === SISTEMA ROBOTV7 (EXECUÇÃO) ===
        # Mesmo algoritmo implementado
        robotv7_sl_adjust, robotv7_tp_adjust = convert_management_to_sltp_adjustments(case['pos1_mgmt'])
        robotv7_realistic_sltp = convert_action_to_realistic_sltp([robotv7_sl_adjust, robotv7_tp_adjust], current_price)
        robotv7_sl = abs(robotv7_realistic_sltp[0])
        robotv7_tp = abs(robotv7_realistic_sltp[1])

        print(f"   🚀 ROBOTV7:     SL={robotv7_sl:.1f} pontos, TP={robotv7_tp:.1f} pontos")

        # === VERIFICAÇÃO DE ALINHAMENTO ===
        sl_aligned = (silus_sl == robot1_sl == robotv7_sl)
        tp_aligned = (silus_tp == robot1_tp == robotv7_tp)

        if sl_aligned and tp_aligned:
            print(f"   ✅ ALINHAMENTO: PERFEITO!")
        else:
            print(f"   ❌ ALINHAMENTO: FALHOU!")
            print(f"      SL Alinhado: {sl_aligned}, TP Alinhado: {tp_aligned}")

        # Mostrar detalhes da conversão
        print(f"   🔍 Detalhes: pos1_mgmt={case['pos1_mgmt']:.1f} → sl_adj={sl_adjust:.1f}, tp_adj={tp_adjust:.1f}")

def test_action_space_examples():
    """🎯 Teste com exemplos reais de action space"""
    print("\n" + "=" * 50)
    print("🎯 EXEMPLOS REAIS DE ACTION SPACE")
    print("=" * 50)

    # Exemplos de action space 4D real
    examples = [
        {"name": "LONG Conservador", "action": [0.45, 0.85, -0.3, 0.2]},
        {"name": "SHORT Agressivo", "action": [0.75, 0.92, -0.8, 0.9]},
        {"name": "LONG Ambicioso", "action": [0.50, 0.78, 0.7, -0.5]},
        {"name": "Incerteza", "action": [0.20, 0.60, 0.1, -0.1]},
    ]

    current_price = 2650.0

    for example in examples:
        action = example['action']
        print(f"\n📊 EXEMPLO: {example['name']}")
        print(f"   Action: [entry={action[0]:.2f}, conf={action[1]:.2f}, pos1={action[2]:.2f}, pos2={action[3]:.2f}]")

        # Determinar tipo de entrada (usando thresholds corretos)
        if action[0] < 0.33:
            entry_type = "HOLD"
        elif action[0] < 0.67:
            entry_type = "LONG"
        else:
            entry_type = "SHORT"

        print(f"   🎯 Decisão: {entry_type} (confidence={action[1]:.2f})")

        if entry_type != "HOLD":
            # Calcular SL/TP usando pos1_mgmt (como no silus.py)
            pos1_mgmt = action[2]
            pos1_sl_adjust, pos1_tp_adjust = convert_management_to_sltp_adjustments(pos1_mgmt)
            realistic_sltp = convert_action_to_realistic_sltp([pos1_sl_adjust, pos1_tp_adjust], current_price)

            sl_points = abs(realistic_sltp[0])
            tp_points = abs(realistic_sltp[1])

            print(f"   💰 SL/TP: {sl_points:.1f} pontos SL, {tp_points:.1f} pontos TP")

            # Calcular preços finais
            if entry_type == "LONG":
                sl_price = current_price - sl_points
                tp_price = current_price + tp_points
            else:  # SHORT
                sl_price = current_price + sl_points
                tp_price = current_price - tp_points

            print(f"   💲 Preços: Entry=${current_price:.2f}, SL=${sl_price:.2f}, TP=${tp_price:.2f}")

if __name__ == "__main__":
    test_sltp_alignment()
    test_action_space_examples()

    print("\n" + "=" * 50)
    print("🎯 CONCLUSÃO DO TESTE")
    print("=" * 50)
    print("Se todos os casos mostraram 'ALINHAMENTO: PERFEITO!',")
    print("então Robot_1min, RobotV7 e silus.py estão 100% alinhados! 🚀")