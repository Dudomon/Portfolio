"""
🧪 TESTE: Migração Completa V8 - DAYTRADER + AVALIAR

Verifica se:
1. daytrader8dim.py está configurado para V8 Elegance
2. avaliar_v8.py foi criado e configurado corretamente
3. Chamadas no daytrader foram alteradas para V8
"""

import re
import os

def test_daytrader_v8_migration():
    """Testa migração do daytrader8dim.py"""
    
    print("🧪 TESTANDO MIGRAÇÃO DAYTRADER8DIM.PY → V8")
    print("="*55)
    
    daytrader_path = "D:/Projeto/daytrader8dim.py"
    
    with open(daytrader_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = [
        ("V8 Import", "from trading_framework.policies.two_head_v8_elegance import"),
        ("V8 Policy", "TwoHeadV8Elegance"),
        ("V8 Kwargs", "get_v8_elegance_kwargs()"),
        ("V8 Validation", "validate_v8_elegance_policy"),
        ("V8 Banner", "V8 ELEGANCE OPTIMIZED"),
        ("V8 Eval Call", "_run_avaliar_v8_evaluation"),
        ("V8 Eval Comment", "AVALIAR_V8.PY"),
    ]
    
    results = []
    for check_name, pattern in checks:
        found = pattern in content
        results.append((check_name, found))
        status = "✅" if found else "❌"
        print(f"   {status} {check_name}: {pattern[:40]}...")
    
    return all(found for _, found in results)

def test_avaliar_v8_creation():
    """Testa criação e configuração do avaliar_v8.py"""
    
    print("\n🧪 TESTANDO AVALIAR_V8.PY")
    print("="*55)
    
    avaliar_path = "D:/Projeto/avaliacao/avaliar_v8.py"
    
    if not os.path.exists(avaliar_path):
        print("❌ avaliar_v8.py não existe!")
        return False
    
    with open(avaliar_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = [
        ("V8 Title", "V8ELEGANCE"),
        ("V8 Function", "test_v8_elegance_trading"),
        ("V8 Import", "two_head_v8_elegance"),
        ("V8 Kwargs", "get_v8_elegance_kwargs"),
        ("V8 Checkpoint", "find_v8_checkpoint"),
        ("V8 Main Call", "test_v8_elegance_trading()"),
    ]
    
    results = []
    for check_name, pattern in checks:
        found = pattern in content
        results.append((check_name, found))
        status = "✅" if found else "❌"
        print(f"   {status} {check_name}: {pattern}")
    
    return all(found for _, found in results)

def test_v8_integration_summary():
    """Resumo da integração V8"""
    
    print("\n📊 RESUMO DA INTEGRAÇÃO V8")
    print("="*55)
    
    # Test components
    daytrader_ok = test_daytrader_v8_migration()
    avaliar_ok = test_avaliar_v8_creation()
    
    print(f"\n📋 RESULTADOS:")
    print(f"   🏗️ daytrader8dim.py → V8: {'✅ OK' if daytrader_ok else '❌ FALHOU'}")
    print(f"   🧪 avaliar_v8.py: {'✅ OK' if avaliar_ok else '❌ FALHOU'}")
    
    if daytrader_ok and avaliar_ok:
        print(f"\n🎉 MIGRAÇÃO V8 COMPLETA!")
        print(f"   🚀 TwoHeadV8Elegance ativa no sistema")
        print(f"   🧪 Avaliação automática V8 configurada")
        print(f"   ⚡ Sistema pronto para treinamento V8")
        return True
    else:
        print(f"\n❌ MIGRAÇÃO V8 INCOMPLETA!")
        print(f"   🔧 Verificar componentes falharam")
        return False

if __name__ == "__main__":
    success = test_v8_integration_summary()
    
    if success:
        print(f"\n✅ V8 ELEGANCE INTEGRADA COM SUCESSO!")
        print(f"   Pronta para usar em produção")
    else:
        print(f"\n❌ PROBLEMAS NA INTEGRAÇÃO V8")
        print(f"   Verificar configurações")
    
    print("\n" + "="*55)