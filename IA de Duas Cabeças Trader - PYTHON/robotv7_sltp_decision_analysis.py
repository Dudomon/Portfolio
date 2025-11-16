"""
🔍 ANÁLISE DECISÃO SL/TP INICIAIS NO ROBOTV7
Como exatamente o modelo decide os valores iniciais de SL e TP
"""

def analyze_robotv7_sltp_decision():
    """📊 Análise completa do processo de decisão SL/TP no RobotV7"""

    print("🔍 ANÁLISE: COMO O MODELO DECIDE SL/TP INICIAIS NO ROBOTV7")
    print("=" * 70)

    print("📋 FLUXO COMPLETO DE DECISÃO:")
    print()

    steps = [
        {
            "step": 1,
            "title": "Action Space do Modelo",
            "description": "Modelo fornece action 4D: [entry_decision, entry_confidence, pos1_mgmt, pos2_mgmt]",
            "example": "action = [0.45, 0.85, -0.3, 0.2]",
            "location": "RobotV7.py:linha ~950"
        },
        {
            "step": 2,
            "title": "Extração dos Valores",
            "description": "pos1_mgmt e pos2_mgmt são extraídos e clampados para [-1,1]",
            "example": "pos1_mgmt = -0.3, pos2_mgmt = 0.2",
            "location": "RobotV7.py:linha ~1009-1011"
        },
        {
            "step": 3,
            "title": "Conversão Bidireional",
            "description": "Management values são convertidos em ajustes SL/TP bidirecionais",
            "example": "pos1_mgmt=-0.3 → sl_adjust=-0.5, tp_adjust=0",
            "location": "_convert_management_to_sltp_adjustments()"
        },
        {
            "step": 4,
            "title": "Seleção de Posição",
            "description": "Para SL/TP iniciais, usa SEMPRE pos1_mgmt (primeira posição)",
            "example": "sl_adjust = pos1_sl_adjust = -0.5, tp_adjust = pos1_tp_adjust = 0",
            "location": "RobotV7.py:linha ~1014-1015"
        },
        {
            "step": 5,
            "title": "Conversão para Pontos",
            "description": "Ajustes [-0.5,0.5] são convertidos para pontos realistas [2-8] SL, [3-15] TP",
            "example": "sl_adjust=-0.5 → 4.5 pontos, tp_adjust=0 → 9.0 pontos",
            "location": "_convert_action_to_realistic_sltp()"
        },
        {
            "step": 6,
            "title": "Aplicação nos Preços",
            "description": "Pontos são aplicados ao preço atual com multiplicador 1.0x",
            "example": "LONG: SL=$2645.50 (2650-4.5), TP=$2659.00 (2650+9.0)",
            "location": "_execute_v7_unified_trade()"
        },
        {
            "step": 7,
            "title": "Ordem MT5",
            "description": "Preços SL/TP são incluídos diretamente na requisição MT5",
            "example": 'request = {"sl": 2645.50, "tp": 2659.00, ...}',
            "location": "RobotV7.py:linha ~1093-1106"
        }
    ]

    for step_info in steps:
        print(f"🎯 PASSO {step_info['step']}: {step_info['title']}")
        print(f"   📝 {step_info['description']}")
        print(f"   💡 Exemplo: {step_info['example']}")
        print(f"   📍 Local: {step_info['location']}")
        print()

def analyze_conversion_logic():
    """🔧 Análise detalhada da lógica de conversão"""

    print("🔧 LÓGICA DE CONVERSÃO DETALHADA")
    print("=" * 70)

    print("📊 ETAPA 1: Management → Ajustes Bidirecionais")
    print("Função: _convert_management_to_sltp_adjustments(mgmt_value)")
    print()

    conversions = [
        {"mgmt": -0.8, "condition": "< -0.5", "result": "(0.5, 0)", "meaning": "Afrouxar SL"},
        {"mgmt": -0.3, "condition": ">= -0.5 e < 0", "result": "(-0.5, 0)", "meaning": "Apertar SL"},
        {"mgmt": 0.0, "condition": "== 0", "result": "(0, 0)", "meaning": "Neutro"},
        {"mgmt": 0.3, "condition": "> 0 e <= 0.5", "result": "(0, -0.5)", "meaning": "TP próximo"},
        {"mgmt": 0.7, "condition": "> 0.5", "result": "(0, 0.5)", "meaning": "TP distante"}
    ]

    for conv in conversions:
        print(f"   mgmt_value = {conv['mgmt']} ({conv['condition']}) → {conv['result']} = {conv['meaning']}")

    print()
    print("📊 ETAPA 2: Ajustes → Pontos Realistas")
    print("Função: _convert_action_to_realistic_sltp([sl_adjust, tp_adjust])")
    print()
    print("   Configuração:")
    print("   - SL: [2-8] pontos (daytrade)")
    print("   - TP: [3-15] pontos (daytrade)")
    print("   - Variação: 0.5 pontos")
    print()
    print("   Fórmula SL: 2 + (sl_adjust + 3) * (8-2) / 6")
    print("   Fórmula TP: 3 + (tp_adjust + 3) * (15-3) / 6")
    print()

    examples = [
        {"adjust": (-0.5, 0), "sl_calc": "2 + (-0.5+3) * 6/6 = 4.5", "tp_calc": "3 + (0+3) * 12/6 = 9.0"},
        {"adjust": (0.5, 0), "sl_calc": "2 + (0.5+3) * 6/6 = 5.5", "tp_calc": "3 + (0+3) * 12/6 = 9.0"},
        {"adjust": (0, -0.5), "sl_calc": "2 + (0+3) * 6/6 = 5.0", "tp_calc": "3 + (-0.5+3) * 12/6 = 8.0"},
        {"adjust": (0, 0.5), "sl_calc": "2 + (0+3) * 6/6 = 5.0", "tp_calc": "3 + (0.5+3) * 12/6 = 10.0"}
    ]

    for ex in examples:
        print(f"   Ajustes {ex['adjust']}:")
        print(f"   - SL: {ex['sl_calc']} pontos")
        print(f"   - TP: {ex['tp_calc']} pontos")
        print()

def test_complete_examples():
    """🧪 Exemplos completos de decisão"""

    print("🧪 EXEMPLOS COMPLETOS DE DECISÃO")
    print("=" * 70)

    examples = [
        {
            "name": "LONG Conservador",
            "action": [0.45, 0.85, -0.3, 0.2],
            "current_price": 2650.0
        },
        {
            "name": "SHORT Agressivo",
            "action": [0.75, 0.92, -0.8, 0.9],
            "current_price": 2650.0
        },
        {
            "name": "LONG Ambicioso",
            "action": [0.50, 0.78, 0.7, -0.5],
            "current_price": 2650.0
        }
    ]

    for example in examples:
        action = example['action']
        current_price = example['current_price']

        print(f"\n📊 EXEMPLO: {example['name']}")
        print(f"   Action: {action}")
        print(f"   Preço atual: ${current_price:.2f}")
        print()

        # Passo 1: Determinar tipo
        raw_decision = action[0]
        if raw_decision < 0.33:
            entry_type = "HOLD"
        elif raw_decision < 0.67:
            entry_type = "LONG"
        else:
            entry_type = "SHORT"
        print(f"   1️⃣ Tipo entrada: {entry_type} (decision={raw_decision:.2f})")

        if entry_type == "HOLD":
            print("   ⭕ HOLD - Sem SL/TP")
            continue

        # Passo 2: Extrair management
        pos1_mgmt = action[2]
        print(f"   2️⃣ pos1_mgmt: {pos1_mgmt}")

        # Passo 3: Conversão bidireational
        if pos1_mgmt < 0:
            if pos1_mgmt < -0.5:
                sl_adjust, tp_adjust = (0.5, 0)
                meaning = "Afrouxar SL"
            else:
                sl_adjust, tp_adjust = (-0.5, 0)
                meaning = "Apertar SL"
        elif pos1_mgmt > 0:
            if pos1_mgmt > 0.5:
                sl_adjust, tp_adjust = (0, 0.5)
                meaning = "TP distante"
            else:
                sl_adjust, tp_adjust = (0, -0.5)
                meaning = "TP próximo"
        else:
            sl_adjust, tp_adjust = (0, 0)
            meaning = "Neutro"

        print(f"   3️⃣ Ajustes: sl={sl_adjust}, tp={tp_adjust} ({meaning})")

        # Passo 4: Conversão para pontos
        sl_points = 2 + (sl_adjust + 3) * (8-2) / 6
        tp_points = 3 + (tp_adjust + 3) * (15-3) / 6

        # Arredondar para 0.5
        sl_points = round(sl_points * 2) / 2
        tp_points = round(tp_points * 2) / 2

        # Garantir limites
        sl_points = max(2, min(sl_points, 8))
        tp_points = max(3, min(tp_points, 15))

        print(f"   4️⃣ Pontos: SL={sl_points:.1f}, TP={tp_points:.1f}")

        # Passo 5: Aplicar nos preços
        if entry_type == "LONG":
            sl_price = current_price - sl_points
            tp_price = current_price + tp_points
        else:  # SHORT
            sl_price = current_price + sl_points
            tp_price = current_price - tp_points

        print(f"   5️⃣ Preços finais: SL=${sl_price:.2f}, TP=${tp_price:.2f}")
        print(f"   6️⃣ MT5 Request: {{'sl': {sl_price:.2f}, 'tp': {tp_price:.2f}}}")

def analyze_key_points():
    """🎯 Pontos chave da decisão"""

    print("\n🎯 PONTOS CHAVE DA DECISÃO SL/TP")
    print("=" * 70)

    key_points = [
        "✅ CONTROLE TOTAL: O modelo controla 100% dos SL/TP iniciais",
        "✅ DECISÃO ÚNICA: pos1_mgmt determina ambos SL e TP (bidirecionalmente)",
        "✅ LÓGICA CLARA: < 0 = foco SL, > 0 = foco TP, threshold em ±0.5",
        "✅ RANGES FIXOS: SL sempre [2-8] pontos, TP sempre [3-15] pontos",
        "✅ RESOLUÇÃO: Múltiplos de 0.5 pontos para precisão",
        "✅ APLICAÇÃO DIRETA: Multiplicador 1.0x (sem conversões adicionais)",
        "✅ MT5 DIRETO: Valores vão direto para requisição MT5 sem filtros"
    ]

    for point in key_points:
        print(f"   {point}")

    print()
    print("🔧 DIFERENÇAS vs TRAILING STOP:")
    print("   - SL/TP INICIAIS: Decididos na abertura da ordem")
    print("   - TRAILING STOP: Ajustes dinâmicos após abertura")
    print("   - INICIAIS: Uma decisão por ordem")
    print("   - TRAILING: Múltiplas decisões durante vida da posição")

if __name__ == "__main__":
    analyze_robotv7_sltp_decision()
    analyze_conversion_logic()
    test_complete_examples()
    analyze_key_points()

    print("\n" + "=" * 70)
    print("🎯 RESUMO EXECUTIVO")
    print("=" * 70)
    print("O modelo decide SL/TP iniciais através de um processo determinístico:")
    print()
    print("1. 📊 action[2] (pos1_mgmt) é o valor decisório único")
    print("2. 🔄 Conversão bidireacional: < 0 afeta SL, > 0 afeta TP")
    print("3. 📏 Mapeamento para ranges realistas: SL[2-8], TP[3-15] pontos")
    print("4. 💰 Aplicação direta no preço com multiplicador 1.0x")
    print("5. 📤 Inclusão imediata na requisição MT5")
    print()
    print("✅ O modelo tem CONTROLE TOTAL sobre os valores iniciais de SL e TP!")
    print("✅ Processo 100% determinístico e alinhado com silus.py!")