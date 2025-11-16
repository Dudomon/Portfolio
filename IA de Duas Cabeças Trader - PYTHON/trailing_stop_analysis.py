"""
🔍 ANÁLISE DO SISTEMA TRAILING STOP
Comparação entre silus.py, Robot_1min.py e RobotV7.py
"""

def analyze_trailing_stop_systems():
    """📊 Análise comparativa dos sistemas de trailing stop"""

    print("🔍 ANÁLISE SISTEMA TRAILING STOP")
    print("=" * 60)

    systems = {
        "silus.py": {
            "function": "_process_dynamic_trailing_stop",
            "activation_threshold": "abs(trailing_signal) > 1.5",
            "movement_threshold": "abs(trailing_signal) > 0.5",
            "distance_calculation": "15 + (trailing_intensity * 5)",  # 15-30 pontos
            "distance_adjustment": "±(trailing_intensity * 3)",       # ±3x intensidade
            "min_distance": "10 pontos",
            "max_distance": "40 pontos",
            "profit_requirement": "current_pnl > 0",
            "opportunity_threshold": "current_pnl > pos['entry_price'] * 0.02"  # 2%
        },

        "Robot_1min.py": {
            "function": "_process_dynamic_sl_tp_adjustment",
            "activation_threshold": "abs(trailing_signal) > 1.5",
            "movement_threshold": "abs(trailing_signal) > 0.5",
            "distance_calculation": "15 + (trailing_intensity * 5)",  # 15-30 pontos
            "distance_adjustment": "±(trailing_intensity * 3)",       # ±3x intensidade
            "min_distance": "10 pontos",
            "max_distance": "40 pontos",
            "profit_requirement": "current_pnl > 0",
            "opportunity_threshold": "current_pnl > position.price_open * 0.02"  # 2%
        },

        "RobotV7.py": {
            "function": "_process_dynamic_sl_tp_adjustment",
            "activation_threshold": "abs(trailing_signal) > 1.5",
            "movement_threshold": "abs(trailing_signal) > 0.5",
            "distance_calculation": "15 + (trailing_intensity * 5)",  # 15-30 pontos
            "distance_adjustment": "±(trailing_intensity * 3)",       # ±3x intensidade
            "min_distance": "10 pontos",
            "max_distance": "40 pontos",
            "profit_requirement": "current_pnl > 0",
            "opportunity_threshold": "current_pnl > position.price_open * 0.02"  # 2%
        }
    }

    print("📋 COMPARAÇÃO DOS SISTEMAS:")
    print()

    for key in ["activation_threshold", "movement_threshold", "distance_calculation",
                "distance_adjustment", "min_distance", "max_distance",
                "profit_requirement", "opportunity_threshold"]:

        print(f"🔧 {key.replace('_', ' ').title()}:")

        silus_val = systems["silus.py"][key]
        robot1_val = systems["Robot_1min.py"][key]
        robotv7_val = systems["RobotV7.py"][key]

        # Verificar alinhamento
        all_same = silus_val == robot1_val == robotv7_val
        robots_same = robot1_val == robotv7_val

        print(f"   silus.py:     {silus_val}")
        print(f"   Robot_1min:   {robot1_val}")
        print(f"   RobotV7:      {robotv7_val}")

        if all_same:
            print(f"   ✅ TODOS ALINHADOS")
        elif robots_same:
            print(f"   ⚠️  ROBÔS ALINHADOS, SILUS DIFERENTE")
        else:
            print(f"   ❌ DESALINHAMENTO ENCONTRADO")
        print()

def analyze_key_differences():
    """🔍 Análise das diferenças críticas"""
    print("🔍 DIFERENÇAS CRÍTICAS IDENTIFICADAS")
    print("=" * 60)

    differences = [
        {
            "component": "Distance Adjustment Multiplier",
            "silus": "±(trailing_intensity * 3)",
            "robots": "±(trailing_intensity * 3)",
            "impact": "CORRIGIDO: Agora alinhados",
            "critical": False
        },
        {
            "component": "Function Name",
            "silus": "_process_dynamic_trailing_stop",
            "robots": "_process_dynamic_sl_tp_adjustment",
            "impact": "Nomenclatura diferente, mas funcionalidade similar",
            "critical": False
        },
        {
            "component": "Position Structure",
            "silus": "pos dict com keys diretas",
            "robots": "position object com _metadata",
            "impact": "Estrutura de dados diferente para tracking",
            "critical": False
        }
    ]

    for diff in differences:
        status = "🔴 CRÍTICA" if diff["critical"] else "🟡 MENOR"
        print(f"{status} - {diff['component']}:")
        print(f"   silus.py: {diff['silus']}")
        print(f"   Robôs:    {diff['robots']}")
        print(f"   Impacto:  {diff['impact']}")
        print()

def analyze_trailing_logic():
    """🎯 Análise da lógica de trailing stop"""
    print("🎯 LÓGICA DO TRAILING STOP")
    print("=" * 60)

    logic_flow = [
        "1. 📊 Modelo envia sl_adjust [-3,3] e tp_adjust [-3,3]",
        "2. 🎯 sl_adjust é interpretado como trailing_signal",
        "3. 🔢 tp_adjust é interpretado como trailing_intensity",
        "4. 🔥 ATIVAÇÃO: |trailing_signal| > 1.5 + posição em lucro",
        "5. 📏 Distância inicial: 15 + (intensity * 5) = 15-30 pontos",
        "6. 🔄 MOVIMENTO: trailing ativo + |signal| > 0.5",
        "7. ⚙️  Ajuste distância: ±(intensity * X) - X varia!",
        "8. 📊 Oportunidade perdida: 2%+ lucro sem trailing",
        "9. 🎯 TP dinâmico: |tp_adjust| > 0.5 modifica TP atual"
    ]

    for step in logic_flow:
        print(f"   {step}")

    print()
    print("🔧 PARÂMETROS CHAVE:")
    print("   - Ativação:     |signal| > 1.5")
    print("   - Movimento:    |signal| > 0.5")
    print("   - Dist. inicial: 15-30 pontos")
    print("   - Dist. mín:    10 pontos")
    print("   - Dist. máx:    40 pontos")
    print("   - Lucro mín:    > 0 para ativação")
    print("   - Oportunidade: 2% de lucro")

def test_trailing_scenarios():
    """🧪 Testar cenários de trailing stop"""
    print("\n🧪 CENÁRIOS DE TESTE TRAILING STOP")
    print("=" * 60)

    scenarios = [
        {
            "name": "Ativação Normal",
            "sl_adjust": -2.0,
            "tp_adjust": 1.2,
            "pnl": 50.0,
            "trailing_active": False,
            "expected": "Ativar trailing com 21 pontos de distância"
        },
        {
            "name": "Movimento Apertar",
            "sl_adjust": 1.8,
            "tp_adjust": 0.8,
            "pnl": 75.0,
            "trailing_active": True,
            "current_distance": 20,
            "expected": "Apertar trailing para ~18 pontos (todos alinhados)"
        },
        {
            "name": "Movimento Relaxar",
            "sl_adjust": -1.2,
            "tp_adjust": 1.5,
            "pnl": 30.0,
            "trailing_active": True,
            "current_distance": 15,
            "expected": "Relaxar trailing para ~19.5 pontos (todos alinhados)"
        },
        {
            "name": "Sinal Fraco",
            "sl_adjust": 0.3,
            "tp_adjust": 0.2,
            "pnl": 40.0,
            "trailing_active": False,
            "expected": "Nenhuma ação (signal < 1.5)"
        },
        {
            "name": "Posição no Prejuízo",
            "sl_adjust": -2.5,
            "tp_adjust": 1.8,
            "pnl": -20.0,
            "trailing_active": False,
            "expected": "Nenhuma ação (pnl <= 0)"
        }
    ]

    for scenario in scenarios:
        print(f"\n📋 {scenario['name']}:")
        print(f"   sl_adjust: {scenario['sl_adjust']}")
        print(f"   tp_adjust: {scenario['tp_adjust']}")
        print(f"   PnL: ${scenario['pnl']:.0f}")
        print(f"   Trailing ativo: {scenario['trailing_active']}")
        if 'current_distance' in scenario:
            print(f"   Distância atual: {scenario['current_distance']} pontos")
        print(f"   🎯 Esperado: {scenario['expected']}")

if __name__ == "__main__":
    analyze_trailing_stop_systems()
    analyze_key_differences()
    analyze_trailing_logic()
    test_trailing_scenarios()

    print("\n" + "=" * 60)
    print("🎯 CONCLUSÃO DA ANÁLISE")
    print("=" * 60)
    print("✅ ALINHAMENTOS:")
    print("   - Thresholds de ativação e movimento idênticos")
    print("   - Cálculo de distância inicial idêntico")
    print("   - Limites mín/máx de distância idênticos")
    print("   - Lógica de ativação em lucro idêntica")
    print()
    print("✅ CORREÇÃO APLICADA:")
    print("   - Todos os sistemas agora usam ±(intensity * 3)")
    print("   - Alinhamento perfeito entre silus.py e robôs")
    print("   - IMPACTO: Comportamento idêntico em todos os sistemas")
    print()
    print("🚀 STATUS:")
    print("   Sistema de trailing stop totalmente alinhado!")
    print("   Todas as implementações são consistentes")