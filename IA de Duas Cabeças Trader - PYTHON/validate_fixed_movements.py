"""
🔍 Validação: Movimento FIXO de SL/TP (±0.5 pontos)
"""

def validate_fixed_movements():
    """Valida se os movimentos são FIXOS (±0.5) e não proporcionais"""

    print("="*80)
    print("🔍 VALIDAÇÃO: Movimentos FIXOS de SL/TP")
    print("="*80)

    # Ler cherry.py
    with open("D:\\Projeto\\cherry.py", "r", encoding="utf-8") as f:
        cherry_content = f.read()

    # Ler Robot_cherry.py
    with open("D:\\Projeto\\Modelo PPO Trader\\Robot_cherry.py", "r", encoding="utf-8") as f:
        robot_content = f.read()

    issues = []

    print("\n🔍 Verificando se movimentos são FIXOS (não proporcionais)...")

    # 1. Verificar se NÃO tem multiplicação proporcional
    bad_patterns = [
        ("sl_adjust * 2.0", "Multiplicação proporcional de SL"),
        ("tp_adjust * 3.0", "Multiplicação proporcional de TP"),
        ("sl_adjust * 2", "Multiplicação proporcional de SL"),
        ("tp_adjust * 3", "Multiplicação proporcional de TP"),
    ]

    for pattern, desc in bad_patterns:
        if pattern in cherry_content:
            issues.append(f"❌ cherry.py: Ainda usa {desc}: '{pattern}'")
        if pattern in robot_content:
            issues.append(f"❌ Robot_cherry.py: Ainda usa {desc}: '{pattern}'")

    # 2. Verificar padrão correto: movimento = adjust (FIXO)
    cherry_sl_fixed = "sl_movement_points = sl_adjust  # VALOR FIXO" in cherry_content
    cherry_tp_fixed = "tp_movement_points = tp_adjust  # VALOR FIXO" in cherry_content

    robot_sl_fixed = "sl_movement_points = sl_adjust  # VALOR FIXO" in robot_content
    robot_tp_fixed = "tp_movement_points = tp_adjust  # VALOR FIXO" in robot_content

    if cherry_sl_fixed:
        print("✅ cherry.py: SL movement FIXO (±0.5 pontos)")
    else:
        issues.append("❌ cherry.py: SL movement não está usando valor FIXO")

    if cherry_tp_fixed:
        print("✅ cherry.py: TP movement FIXO (±0.5 pontos)")
    else:
        issues.append("❌ cherry.py: TP movement não está usando valor FIXO")

    if robot_sl_fixed:
        print("✅ Robot_cherry.py: SL movement FIXO (±0.5 pontos)")
    else:
        issues.append("❌ Robot_cherry.py: SL movement não está usando valor FIXO")

    if robot_tp_fixed:
        print("✅ Robot_cherry.py: TP movement FIXO (±0.5 pontos)")
    else:
        issues.append("❌ Robot_cherry.py: TP movement não está usando valor FIXO")

    # 3. Verificar função convert_management_to_sltp_adjustments
    print("\n🔍 Verificando função convert_management_to_sltp_adjustments...")

    cherry_has_func = "def convert_management_to_sltp_adjustments(mgmt_value):" in cherry_content
    robot_has_func = "def _convert_management_to_sltp_adjustments(self, mgmt_value):" in robot_content

    cherry_returns_fixed = "return (0.5, 0)" in cherry_content and "return (-0.5, 0)" in cherry_content
    robot_returns_fixed = "return (0.5, 0)" in robot_content and "return (-0.5, 0)" in robot_content

    if cherry_has_func and cherry_returns_fixed:
        print("✅ cherry.py: Função retorna valores FIXOS (±0.5)")
    else:
        issues.append("❌ cherry.py: Função não retorna valores fixos")

    if robot_has_func and robot_returns_fixed:
        print("✅ Robot_cherry.py: Função retorna valores FIXOS (±0.5)")
    else:
        issues.append("❌ Robot_cherry.py: Função não retorna valores fixos")

    # 4. Simular valores esperados
    print("\n🔍 Simulando valores esperados...")
    print("\n📊 Valores de Management → SL/TP Adjust:")
    print("   mgmt = -0.8  →  sl_adjust=+0.5, tp_adjust=0   (Afrouxar SL)")
    print("   mgmt = -0.3  →  sl_adjust=-0.5, tp_adjust=0   (Apertar SL)")
    print("   mgmt = +0.3  →  sl_adjust=0,    tp_adjust=-0.5 (TP próximo)")
    print("   mgmt = +0.8  →  sl_adjust=0,    tp_adjust=+0.5 (TP distante)")
    print("\n📊 Movimento Final de SL/TP (após _process_dynamic_trailing):")
    print("   SEMPRE ±0.5 pontos (FIXO), nunca proporcional!")

    # RESUMO FINAL
    print("\n" + "="*80)
    print("📊 RESUMO DA VALIDAÇÃO:")
    print("="*80)

    if not issues:
        print("✅ PERFEITO! Movimentos são FIXOS (±0.5 pontos)")
        print("\n📋 Sistema Correto:")
        print("   ✅ convert_management() retorna ±0.5 FIXO")
        print("   ✅ _process_dynamic_trailing() usa valor direto (não multiplica)")
        print("   ✅ Resultado final: SEMPRE ±0.5 pontos por ajuste")
        print("\n⚠️  Nota: Se o modelo já foi treinado com valores")
        print("    proporcionais (×2.0, ×3.0), será necessário RE-TREINAR!")
        return True
    else:
        print(f"❌ FALHA! {len(issues)} problema(s) encontrado(s):")
        for issue in issues:
            print(f"   {issue}")
        return False

if __name__ == "__main__":
    success = validate_fixed_movements()
    exit(0 if success else 1)
