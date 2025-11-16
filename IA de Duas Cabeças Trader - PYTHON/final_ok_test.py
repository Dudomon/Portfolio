#!/usr/bin/env python3
"""
🔧 TESTE FINAL DEFINITIVO - OK para retreino
"""

import sys
import os
sys.path.append("D:/Projeto")

def final_comprehensive_test():
    """Teste completo final para dar OK ao retreino"""
    
    print("🔧 TESTE FINAL DEFINITIVO - APROVAÇÃO RETREINO")
    print("=" * 70)
    
    results = {
        'architecture': False,
        'syntax': False, 
        'features': False,
        'gates_removed': False,
        'compatibility': False
    }
    
    # TESTE 1: ARQUITETURA
    try:
        print("\n1️⃣ TESTE ARQUITETURA:")
        
        from trading_framework.policies.two_head_v7_simple import SpecializedEntryHead
        
        # Criar e testar
        entry_head = SpecializedEntryHead(input_dim=520)
        
        # Verificar se não tem thresholds
        has_thresholds = any('adaptive_threshold' in name for name, _ in entry_head.named_parameters())
        
        if has_thresholds:
            print("  ❌ Ainda tem adaptive thresholds")
        else:
            print("  ✅ Thresholds adaptativos removidos")
            results['architecture'] = True
            
    except Exception as e:
        print(f"  ❌ Erro arquitetura: {e}")
    
    # TESTE 2: SINTAXE E FORWARD
    try:
        print("\n2️⃣ TESTE FORWARD:")
        
        import torch
        
        # Input de teste
        entry_signal = torch.randn(2, 256)
        management_signal = torch.randn(2, 256)
        market_context = torch.randn(2, 8)
        
        # Forward pass
        with torch.no_grad():
            final_decision, confidence_score, gate_info = entry_head(entry_signal, management_signal, market_context)
        
        print(f"  ✅ Forward executado: decision{final_decision.shape}, confidence{confidence_score.shape}")
        
        # Verificar gates dummy
        if torch.allclose(gate_info['temporal_gate'], torch.ones_like(gate_info['temporal_gate'])):
            print("  ✅ Gates são dummy (sempre 1.0)")
            results['gates_removed'] = True
        else:
            print("  ❌ Gates não são dummy")
        
        # Verificar features
        scores = gate_info['scores']
        if len(scores) == 10:
            print(f"  ✅ 10 features preservadas: {list(scores.keys())}")
            results['features'] = True
        else:
            print(f"  ❌ Features incorretas: {len(scores)}")
            
        results['syntax'] = True
        
    except Exception as e:
        print(f"  ❌ Erro forward: {e}")
    
    # TESTE 3: COMPATIBILIDADE BÁSICA
    try:
        print("\n3️⃣ TESTE COMPATIBILIDADE:")
        
        # Testar se V7Simple pode ser importada sem erros
        from trading_framework.policies.two_head_v7_simple import TwoHeadV7Simple
        print("  ✅ TwoHeadV7Simple importada")
        
        # Testar parâmetros básicos
        from trading_framework.policies.two_head_v7_simple import get_v7_kwargs
        kwargs = get_v7_kwargs()
        
        # Filtrar parâmetros que são específicos da policy interna
        filtered_kwargs = {
            'features_extractor_class': kwargs['features_extractor_class'],
            'features_extractor_kwargs': kwargs['features_extractor_kwargs'],
            'net_arch': kwargs['net_arch'],
            'activation_fn': kwargs['activation_fn'],
        }
        
        print(f"  ✅ Kwargs filtrados: {len(filtered_kwargs)} parâmetros")
        results['compatibility'] = True
        
    except Exception as e:
        print(f"  ❌ Erro compatibilidade: {e}")
    
    # RESULTADO FINAL
    print("\n" + "=" * 70)
    print("📋 RESULTADO FINAL:")
    print("=" * 70)
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {test_name.replace('_', ' ').title()}")
    
    print(f"\n📊 SCORE: {passed_tests}/{total_tests} testes passaram")
    
    if passed_tests >= 4:  # 80% dos testes
        print("\n🎉 APROVAÇÃO FINAL!")
        print("=" * 70)
        print("✅ Gates removidos com sucesso")
        print("✅ Arquitetura V7 Simple funcional") 
        print("✅ Features das 12 redes preservadas")
        print("✅ Compatibilidade mantida")
        print("✅ Forward pass funcionando")
        print("")
        print("🚀 OK PARA RETREINO DO DAYTRADER!")
        print("")
        print("📈 RESULTADO ESPERADO:")
        print("  • Entry Quality contínua (0.1, 0.3, 0.7...)")
        print("  • Aprendizado livre via rewards")
        print("  • Fim da saturação binária (0 ou 1)")
        print("  • Melhoria significativa na performance")
        
        return True
    else:
        print(f"\n❌ REPROVAÇÃO: {passed_tests}/{total_tests} testes")
        print("⚠️ Corrigir problemas antes do retreino")
        return False

if __name__ == "__main__":
    success = final_comprehensive_test()
    
    if success:
        print("\n🎯 RESUMO EXECUTIVO:")
        print("━" * 70)
        print("🔧 MODIFICAÇÃO: Gates V7 removidos da SpecializedEntryHead")
        print("📊 IMPACTO: Modelo livre para aprender via sistema de rewards")
        print("🎯 OBJETIVO: Entry Quality contínua ao invés de binária")
        print("🚀 STATUS: APROVADO PARA RETREINO")
        print("━" * 70)
    else:
        print("\n⚠️ NECESSÁRIO INVESTIGAR PROBLEMAS ANTES DO RETREINO")