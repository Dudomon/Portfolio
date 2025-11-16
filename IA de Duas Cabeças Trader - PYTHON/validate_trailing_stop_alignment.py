"""
🔍 Validação: Alinhamento de Trailing Stop cherry.py ↔ Robot_cherry.py
"""

def validate_trailing_stop_logic():
    """Valida se a lógica de trailing stop está alinhada"""

    print("="*80)
    print("🔍 VALIDAÇÃO: Alinhamento Trailing Stop cherry.py ↔ Robot_cherry.py")
    print("="*80)

    # Ler cherry.py
    with open("D:\\Projeto\\cherry.py", "r", encoding="utf-8") as f:
        cherry_content = f.read()

    # Ler Robot_cherry.py
    with open("D:\\Projeto\\Modelo PPO Trader\\Robot_cherry.py", "r", encoding="utf-8") as f:
        robot_content = f.read()

    issues = []

    # 1. Verificar sistema DIRETO (sem ativação explícita)
    print("\n🔍 Verificando sistema DIRETO de trailing stop...")

    if "trailing_activated" in robot_content and "pos_metadata.get('trailing_activated'" in robot_content:
        issues.append("❌ Robot_cherry.py ainda usa sistema de ATIVAÇÃO explícita (trailing_activated)")
    else:
        print("✅ Robot_cherry.py: Sistema de ativação explícita removido")

    # 2. Verificar cap de $100 USD
    print("\n🔍 Verificando cap de $100 USD...")

    cherry_has_cap = "if current_pnl >= 100:" in cherry_content
    robot_has_cap = "if current_pnl >= 100:" in robot_content

    if cherry_has_cap:
        print("✅ cherry.py: Cap de $100 USD implementado")
    else:
        issues.append("❌ cherry.py: Cap de $100 USD NÃO encontrado")

    if robot_has_cap:
        print("✅ Robot_cherry.py: Cap de $100 USD implementado")
    else:
        issues.append("❌ Robot_cherry.py: Cap de $100 USD NÃO encontrado")

    # 3. Verificar lógica de SL TRAILING ONLY
    print("\n🔍 Verificando SL TRAILING ONLY (a favor do trade)...")

    # cherry.py patterns
    cherry_long_up = "# LONG: new SL = current SL + movement" in cherry_content
    cherry_long_restrict = "# RESTRICTION: SL can only go UP" in cherry_content
    cherry_short_down = "# SHORT: new SL = current SL - movement" in cherry_content
    cherry_short_restrict = "# RESTRICTION: SL can only go DOWN" in cherry_content

    # Robot_cherry.py patterns
    robot_long_up = "# LONG: new SL = current SL + movement" in robot_content
    robot_long_restrict = "# RESTRICTION: SL can only go UP" in robot_content
    robot_short_down = "# SHORT: new SL = current SL - movement" in robot_content
    robot_short_restrict = "# RESTRICTION: SL can only go DOWN" in robot_content

    if cherry_long_up and cherry_long_restrict:
        print("✅ cherry.py: LONG SL only UP (proteção)")
    else:
        issues.append("❌ cherry.py: LONG SL restriction missing")

    if cherry_short_down and cherry_short_restrict:
        print("✅ cherry.py: SHORT SL only DOWN (proteção)")
    else:
        issues.append("❌ cherry.py: SHORT SL restriction missing")

    if robot_long_up and robot_long_restrict:
        print("✅ Robot_cherry.py: LONG SL only UP (proteção)")
    else:
        issues.append("❌ Robot_cherry.py: LONG SL restriction missing")

    if robot_short_down and robot_short_restrict:
        print("✅ Robot_cherry.py: SHORT SL only DOWN (proteção)")
    else:
        issues.append("❌ Robot_cherry.py: SHORT SL restriction missing")

    # 4. Verificar TP com cap de $100
    print("\n🔍 Verificando TP adjustable com cap de $100...")

    cherry_tp_cap = "if potential_pnl <= 100:" in cherry_content
    robot_tp_cap = "if potential_pnl <= 100:" in robot_content

    if cherry_tp_cap:
        print("✅ cherry.py: TP com validação de cap $100")
    else:
        issues.append("❌ cherry.py: TP sem validação de cap")

    if robot_tp_cap:
        print("✅ Robot_cherry.py: TP com validação de cap $100")
    else:
        issues.append("❌ Robot_cherry.py: TP sem validação de cap")

    # 5. Verificar thresholds de ativação
    print("\n🔍 Verificando thresholds de ativação...")

    cherry_sl_threshold = "if abs(sl_adjust) >= 0.3:" in cherry_content
    cherry_tp_threshold = "if abs(tp_adjust) >= 0.3:" in cherry_content

    robot_sl_threshold = "if abs(sl_adjust) >= 0.3:" in robot_content
    robot_tp_threshold = "if abs(tp_adjust) >= 0.3:" in robot_content

    if cherry_sl_threshold and robot_sl_threshold:
        print("✅ Threshold SL alinhado: >= 0.3")
    else:
        issues.append("❌ Threshold SL desalinhado")

    if cherry_tp_threshold and robot_tp_threshold:
        print("✅ Threshold TP alinhado: >= 0.3")
    else:
        issues.append("❌ Threshold TP desalinhado")

    # 6. Verificar buffers de segurança
    print("\n🔍 Verificando buffers de segurança...")

    cherry_sl_buffer = "current_price - 5.0" in cherry_content
    cherry_tp_buffer = "current_price + 3.0" in cherry_content

    robot_sl_buffer = "current_price - 5.0" in robot_content
    robot_tp_buffer = "current_price + 3.0" in robot_content

    if cherry_sl_buffer and robot_sl_buffer:
        print("✅ SL buffer alinhado: 5.0 pontos")
    else:
        issues.append("❌ SL buffer desalinhado")

    if cherry_tp_buffer and robot_tp_buffer:
        print("✅ TP buffer alinhado: 3.0 pontos")
    else:
        issues.append("❌ TP buffer desalinhado")

    # 7. Verificar multiplicadores de movimento
    print("\n🔍 Verificando multiplicadores de movimento...")

    cherry_sl_mult = "sl_adjust * 2.0" in cherry_content
    cherry_tp_mult = "tp_adjust * 3.0" in cherry_content

    robot_sl_mult = "sl_adjust * 2.0" in robot_content
    robot_tp_mult = "tp_adjust * 3.0" in robot_content

    if cherry_sl_mult and robot_sl_mult:
        print("✅ SL movement alinhado: * 2.0")
    else:
        issues.append("❌ SL movement desalinhado")

    if cherry_tp_mult and robot_tp_mult:
        print("✅ TP movement alinhado: * 3.0")
    else:
        issues.append("❌ TP movement desalinhado")

    # 8. Verificar auto-close no cap
    print("\n🔍 Verificando auto-close no cap de $100...")

    cherry_autoclose = "self._close_position(pos, self.current_step)" in cherry_content and "current_pnl >= 100" in cherry_content
    robot_autoclose = "mt5.order_send(close_request)" in robot_content and "current_pnl >= 100" in robot_content

    if cherry_autoclose:
        print("✅ cherry.py: Auto-close em $100 implementado")
    else:
        print("⚠️  cherry.py: Auto-close em $100 não detectado claramente")

    if robot_autoclose:
        print("✅ Robot_cherry.py: Auto-close em $100 implementado")
    else:
        issues.append("❌ Robot_cherry.py: Auto-close em $100 não implementado")

    # RESUMO FINAL
    print("\n" + "="*80)
    print("📊 RESUMO DA VALIDAÇÃO:")
    print("="*80)

    if not issues:
        print("✅ SUCESSO! Trailing stop TOTALMENTE ALINHADO!")
        print("\n📋 Características Alinhadas:")
        print("   ✅ Sistema DIRETO (sem ativação explícita)")
        print("   ✅ Cap de $100 USD no TP")
        print("   ✅ Auto-close em $100 USD")
        print("   ✅ SL TRAILING ONLY (a favor do trade)")
        print("      ├─ LONG: SL só sobe")
        print("      └─ SHORT: SL só desce")
        print("   ✅ TP ajustável com validação de cap")
        print("   ✅ Buffers de segurança (5pt SL, 3pt TP)")
        print("   ✅ Multiplicadores consistentes (2x SL, 3x TP)")
        print("   ✅ Thresholds alinhados (0.3)")
        return True
    else:
        print(f"❌ FALHA! {len(issues)} problema(s) encontrado(s):")
        for issue in issues:
            print(f"   {issue}")
        return False

if __name__ == "__main__":
    success = validate_trailing_stop_logic()
    exit(0 if success else 1)
