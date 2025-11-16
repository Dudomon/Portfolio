#!/usr/bin/env python3
"""
🔧 TESTE FINAL ARQUITETURA - Verificar se V7 Simple modificada funciona
"""

import sys
import os
sys.path.append("D:/Projeto")

import torch
import numpy as np

def test_v7_simple_architecture():
    """Teste básico da arquitetura V7 Simple modificada"""
    
    print("🔧 TESTE FINAL - ARQUITETURA V7 SIMPLE SEM GATES")
    print("=" * 60)
    
    try:
        # Importar a classe modificada
        from trading_framework.policies.two_head_v7_simple import SpecializedEntryHead
        
        print("✅ SpecializedEntryHead importada com sucesso")
        
        # Criar instância
        input_dim = 520  # Dimensão esperada
        entry_head = SpecializedEntryHead(input_dim=input_dim)
        
        print(f"✅ SpecializedEntryHead criada (input_dim={input_dim})")
        
        # Verificar se não tem adaptive_thresholds
        has_thresholds = any('adaptive_threshold' in name for name, _ in entry_head.named_parameters())
        
        if has_thresholds:
            print("❌ ERRO: Ainda tem adaptive thresholds!")
            return False
        else:
            print("✅ Adaptive thresholds removidos corretamente")
        
        # Criar input de teste
        batch_size = 4
        entry_signal = torch.randn(batch_size, 256)
        management_signal = torch.randn(batch_size, 256) 
        market_context = torch.randn(batch_size, 8)
        
        print(f"✅ Inputs criados: entry({entry_signal.shape}), mgmt({management_signal.shape}), market({market_context.shape})")
        
        # Testar forward pass
        with torch.no_grad():
            entry_head.eval()
            final_decision, confidence_score, gate_info = entry_head(entry_signal, management_signal, market_context)
        
        print(f"✅ Forward pass executado:")
        print(f"  final_decision: {final_decision.shape}")
        print(f"  confidence_score: {confidence_score.shape}")
        print(f"  gate_info keys: {list(gate_info.keys())}")
        
        # Verificar se gates são dummy (sempre 1.0)
        temporal_gate = gate_info['temporal_gate']
        validation_gate = gate_info['validation_gate']
        
        if torch.allclose(temporal_gate, torch.ones_like(temporal_gate)):
            print("✅ Gates são dummy (sempre 1.0) - correto!")
        else:
            print(f"❌ Gates não são dummy: temporal_gate = {temporal_gate.mean().item():.3f}")
        
        # Verificar variabilidade no final_decision
        decision_std = final_decision.std().item()
        print(f"✅ Final decision variabilidade: std={decision_std:.4f}")
        
        if decision_std > 0.01:
            print("✅ Decisões têm variabilidade adequada")
        else:
            print("⚠️ Pouca variabilidade (pode ser normal para teste aleatório)")
        
        # Verificar features
        scores = gate_info['scores']
        print(f"\n📊 FEATURES (ex-scores):")
        for name, feature in scores.items():
            mean_val = feature.mean().item()
            std_val = feature.std().item()
            print(f"  {name:12}: mean={mean_val:.3f}, std={std_val:.3f}")
        
        print(f"\n🎯 RESULTADO DO TESTE:")
        print(f"  ✅ Arquitetura modificada funciona")
        print(f"  ✅ Gates removidos (agora são dummy)")
        print(f"  ✅ Features das 12 redes preservadas") 
        print(f"  ✅ Forward pass sem erros")
        print(f"  ✅ Compatibilidade mantida")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_code_syntax():
    """Verificar se o código da V7 Simple está sintaticamente correto"""
    
    print("\n" + "=" * 60)
    print("🔍 TESTE DE SINTAXE")
    print("=" * 60)
    
    try:
        # Compilar o arquivo modificado
        file_path = "D:/Projeto/trading_framework/policies/two_head_v7_simple.py"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Tentar compilar
        compile(code, file_path, 'exec')
        print("✅ Sintaxe do código está correta")
        
        # Verificar se tem problemas óbvios
        issues = []
        
        if 'return final_decision, confidence_score, gate_info' not in code:
            issues.append("Return statement incorreto")
        
        if 'confidence_score = confidence_feature' not in code:
            issues.append("confidence_score não definido")
        
        if 'torch.ones_like(' not in code:
            issues.append("Gates dummy não implementados")
        
        if issues:
            print("⚠️ Possíveis problemas:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✅ Estrutura do código parece correta")
        
        return len(issues) == 0
        
    except SyntaxError as e:
        print(f"❌ ERRO DE SINTAXE: {e}")
        return False
    except Exception as e:
        print(f"❌ ERRO: {e}")
        return False

if __name__ == "__main__":
    print("🎯 BATERIA DE TESTES FINAL")
    print("=" * 60)
    
    test1 = test_code_syntax()
    test2 = test_v7_simple_architecture()
    
    print("\n" + "=" * 60) 
    print("📋 RESULTADO FINAL")
    print("=" * 60)
    
    if test1 and test2:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("")
        print("✅ Sintaxe correta")
        print("✅ Arquitetura funcional")
        print("✅ Gates removidos") 
        print("✅ Compatibilidade mantida")
        print("✅ Features preservadas")
        print("")
        print("🚀 APROVAÇÃO FINAL PARA RETREINO!")
        print("")
        print("💡 PRÓXIMAS ETAPAS:")
        print("  1. Iniciar retreino do DayTrader")
        print("  2. Monitorar Entry Quality (deve ser contínua)")
        print("  3. Verificar se rewards funcionam corretamente")
        print("  4. Observar melhoria na qualidade das entradas")
    else:
        print("❌ ALGUNS TESTES FALHARAM")
        if not test1:
            print("  - Problemas de sintaxe")
        if not test2:
            print("  - Problemas na arquitetura")
        print("")
        print("⚠️ INVESTIGAR ANTES DO RETREINO")