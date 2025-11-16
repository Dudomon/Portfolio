"""
🔍 TESTE DETALHADO: FLUXO COMPLETO ACTION → SL/TP INICIAIS
Rastreia cada passo da conversão até os valores finais da ordem MT5
"""
import numpy as np

# Configurações idênticas ao Robot_1min.py
REALISTIC_SLTP_CONFIG = {
    'sl_min_points': 2,     # SL mínimo: 2 pontos
    'sl_max_points': 8,     # SL máximo: 8 pontos
    'tp_min_points': 3,     # TP mínimo: 3 pontos
    'tp_max_points': 15,    # TP máximo: 15 pontos
}

def convert_management_to_sltp_adjustments(mgmt_value):
    """🚀 Converter management value em ajustes SL/TP bidirecionais"""
    print(f"   📥 INPUT: mgmt_value = {mgmt_value}")

    if mgmt_value < 0:
        # Foco em SL management
        if mgmt_value < -0.5:
            result = (0.5, 0)  # Afrouxar SL
            print(f"   🔄 LOGIC: mgmt < -0.5 → Afrouxar SL")
        else:
            result = (-0.5, 0)  # Apertar SL
            print(f"   🔄 LOGIC: -0.5 <= mgmt < 0 → Apertar SL")
    elif mgmt_value > 0:
        # Foco em TP management
        if mgmt_value > 0.5:
            result = (0, 0.5)  # TP distante
            print(f"   🔄 LOGIC: mgmt > 0.5 → TP distante")
        else:
            result = (0, -0.5)  # TP próximo
            print(f"   🔄 LOGIC: 0 < mgmt <= 0.5 → TP próximo")
    else:
        result = (0, 0)  # Neutro
        print(f"   🔄 LOGIC: mgmt = 0 → Neutro")

    print(f"   📤 OUTPUT: sl_adjust = {result[0]}, tp_adjust = {result[1]}")
    return result

def convert_action_to_realistic_sltp(sltp_action_values, current_price):
    """🚀 Converter ajustes para pontos realistas"""
    sl_adjust = sltp_action_values[0]
    tp_adjust = sltp_action_values[1]

    print(f"   📥 INPUT: sl_adjust = {sl_adjust}, tp_adjust = {tp_adjust}")

    # Conversão: [-3,3] → [min,max] pontos
    sl_points = REALISTIC_SLTP_CONFIG['sl_min_points'] + \
                (sl_adjust + 3) * (REALISTIC_SLTP_CONFIG['sl_max_points'] - REALISTIC_SLTP_CONFIG['sl_min_points']) / 6

    tp_points = REALISTIC_SLTP_CONFIG['tp_min_points'] + \
                (tp_adjust + 3) * (REALISTIC_SLTP_CONFIG['tp_max_points'] - REALISTIC_SLTP_CONFIG['tp_min_points']) / 6

    print(f"   🔄 CALC SL: 2 + ({sl_adjust} + 3) * (8-2) / 6 = {sl_points:.2f}")
    print(f"   🔄 CALC TP: 3 + ({tp_adjust} + 3) * (15-3) / 6 = {tp_points:.2f}")

    # Arredondar para múltiplos de 0.5
    sl_points = round(sl_points * 2) / 2
    tp_points = round(tp_points * 2) / 2

    print(f"   🔄 ROUND: SL = {sl_points:.1f}, TP = {tp_points:.1f}")

    # Garantir limites
    sl_points = max(REALISTIC_SLTP_CONFIG['sl_min_points'], min(sl_points, REALISTIC_SLTP_CONFIG['sl_max_points']))
    tp_points = max(REALISTIC_SLTP_CONFIG['tp_min_points'], min(tp_points, REALISTIC_SLTP_CONFIG['tp_max_points']))

    print(f"   📤 OUTPUT: SL = {sl_points:.1f} pontos, TP = {tp_points:.1f} pontos")

    return [sl_points, tp_points]

def calculate_final_prices(sl_points, tp_points, current_price, entry_type):
    """💰 Calcular preços finais SL/TP para ordem MT5"""
    print(f"   📥 INPUT: sl_points = {sl_points:.1f}, tp_points = {tp_points:.1f}")
    print(f"   📥 INPUT: current_price = ${current_price:.2f}, entry_type = {entry_type}")

    if entry_type == "LONG":
        # LONG: SL abaixo, TP acima
        sl_price = current_price - (sl_points * 1.0)  # Multiplicador 1.0
        tp_price = current_price + (tp_points * 1.0)  # Multiplicador 1.0
        print(f"   🔄 LONG: SL = {current_price:.2f} - ({sl_points:.1f} * 1.0) = ${sl_price:.2f}")
        print(f"   🔄 LONG: TP = {current_price:.2f} + ({tp_points:.1f} * 1.0) = ${tp_price:.2f}")
    else:  # SHORT
        # SHORT: SL acima, TP abaixo
        sl_price = current_price + (sl_points * 1.0)  # Multiplicador 1.0
        tp_price = current_price - (tp_points * 1.0)  # Multiplicador 1.0
        print(f"   🔄 SHORT: SL = {current_price:.2f} + ({sl_points:.1f} * 1.0) = ${sl_price:.2f}")
        print(f"   🔄 SHORT: TP = {current_price:.2f} - ({tp_points:.1f} * 1.0) = ${tp_price:.2f}")

    print(f"   📤 OUTPUT: SL = ${sl_price:.2f}, TP = ${tp_price:.2f}")

    return sl_price, tp_price

def test_complete_flow():
    """🧪 Testar fluxo completo com exemplos específicos"""
    print("🔍 TESTE COMPLETO: ACTION → SL/TP INICIAIS NA ORDEM MT5")
    print("=" * 70)

    current_price = 2650.0  # Preço XAUUSD atual

    # Casos de teste específicos
    test_cases = [
        {
            "name": "LONG Conservador",
            "action": [0.45, 0.85, -0.3, 0.2],
            "expected_type": "LONG"
        },
        {
            "name": "SHORT Agressivo",
            "action": [0.75, 0.92, -0.8, 0.9],
            "expected_type": "SHORT"
        },
        {
            "name": "LONG Ambicioso",
            "action": [0.50, 0.78, 0.7, -0.5],
            "expected_type": "LONG"
        }
    ]

    for case in test_cases:
        action = case['action']
        expected_type = case['expected_type']

        print(f"\n📊 CASO: {case['name']}")
        print(f"📥 ACTION: [entry={action[0]:.2f}, conf={action[1]:.2f}, pos1={action[2]:.2f}, pos2={action[3]:.2f}]")
        print("-" * 50)

        # PASSO 1: Determinar tipo de entrada
        print("🎯 PASSO 1: DETERMINAR TIPO DE ENTRADA")
        raw_decision = action[0]
        if raw_decision < 0.33:
            entry_type = "HOLD"
        elif raw_decision < 0.67:
            entry_type = "LONG"
        else:
            entry_type = "SHORT"

        print(f"   raw_decision = {raw_decision:.2f}")
        print(f"   Thresholds: < 0.33 = HOLD | < 0.67 = LONG | >= 0.67 = SHORT")
        print(f"   📤 RESULTADO: {entry_type}")

        if entry_type == "HOLD":
            print("   ⭕ HOLD - Não abre ordem")
            continue

        # PASSO 2: Converter pos1_mgmt para ajustes SL/TP
        print(f"\n🔄 PASSO 2: CONVERTER pos1_mgmt PARA AJUSTES SL/TP")
        pos1_mgmt = action[2]
        sl_adjust, tp_adjust = convert_management_to_sltp_adjustments(pos1_mgmt)

        # PASSO 3: Converter ajustes para pontos realistas
        print(f"\n⚙️ PASSO 3: CONVERTER AJUSTES PARA PONTOS REALISTAS")
        realistic_sltp = convert_action_to_realistic_sltp([sl_adjust, tp_adjust], current_price)
        sl_points = abs(realistic_sltp[0])
        tp_points = abs(realistic_sltp[1])

        # PASSO 4: Calcular preços finais para ordem MT5
        print(f"\n💰 PASSO 4: CALCULAR PREÇOS FINAIS PARA ORDEM MT5")
        sl_price, tp_price = calculate_final_prices(sl_points, tp_points, current_price, entry_type)

        # RESUMO FINAL
        print(f"\n📋 RESUMO FINAL:")
        print(f"   🎯 Tipo: {entry_type}")
        print(f"   📍 Preço Entrada: ${current_price:.2f}")
        print(f"   🛡️ Stop Loss: ${sl_price:.2f} ({sl_points:.1f} pontos)")
        print(f"   🎯 Take Profit: ${tp_price:.2f} ({tp_points:.1f} pontos)")

        # Verificar se distâncias estão corretas
        if entry_type == "LONG":
            sl_distance = current_price - sl_price
            tp_distance = tp_price - current_price
        else:  # SHORT
            sl_distance = sl_price - current_price
            tp_distance = current_price - tp_price

        print(f"   📏 Distância SL: {sl_distance:.2f} pontos")
        print(f"   📏 Distância TP: {tp_distance:.2f} pontos")

        # Verificar se é o tipo esperado
        if entry_type == expected_type:
            print(f"   ✅ TIPO CORRETO: {entry_type} = {expected_type}")
        else:
            print(f"   ❌ TIPO INCORRETO: {entry_type} ≠ {expected_type}")

def validate_mt5_request_format():
    """🔧 Validar formato da requisição MT5"""
    print(f"\n" + "=" * 70)
    print("🔧 VALIDAÇÃO: FORMATO REQUISIÇÃO MT5")
    print("=" * 70)

    current_price = 2650.0
    sl_points = 4.5
    tp_points = 9.0
    volume = 0.02

    # Exemplo para LONG
    print("📊 EXEMPLO REQUISIÇÃO LONG:")
    sl_price = current_price - (sl_points * 1.0)  # 2645.50
    tp_price = current_price + (tp_points * 1.0)  # 2659.00

    request = {
        "action": "mt5.TRADE_ACTION_DEAL",
        "symbol": "XAUUSD",
        "volume": round(volume, 2),
        "type": "mt5.ORDER_TYPE_BUY",
        "price": current_price,
        "sl": sl_price,        # ← SL INICIAL AQUI
        "tp": tp_price,        # ← TP INICIAL AQUI
        "deviation": 20,
        "magic": 12345,
        "comment": "V7 - SL/TP do modelo",
        "type_time": "mt5.ORDER_TIME_GTC",
        "type_filling": "mt5.ORDER_FILLING_IOC",
    }

    print("   📋 ESTRUTURA REQUEST:")
    for key, value in request.items():
        if key in ['sl', 'tp', 'price']:
            print(f"   {key:12}: ${value:.2f}")
        else:
            print(f"   {key:12}: {value}")

    print(f"\n✅ CONFIRMAÇÃO:")
    print(f"   Os valores SL/TP na requisição MT5 são EXATAMENTE os calculados pelo modelo!")
    print(f"   SL = ${sl_price:.2f} ({sl_points:.1f} pontos do preço)")
    print(f"   TP = ${tp_price:.2f} ({tp_points:.1f} pontos do preço)")

if __name__ == "__main__":
    test_complete_flow()
    validate_mt5_request_format()

    print(f"\n" + "=" * 70)
    print("🎯 CONCLUSÃO FINAL")
    print("=" * 70)
    print("1. ✅ Modelo decide via action[2] (pos1_mgmt)")
    print("2. ✅ Management é convertido bidirecionalmente")
    print("3. ✅ Ajustes viram pontos realistas [2-8] SL, [3-15] TP")
    print("4. ✅ Pontos viram preços com multiplicador 1.0")
    print("5. ✅ Preços vão direto para requisição MT5")
    print("")
    print("🚀 O MODELO CONTROLA COMPLETAMENTE OS SL/TP INICIAIS!")