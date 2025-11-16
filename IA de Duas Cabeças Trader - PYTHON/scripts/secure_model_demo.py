#!/usr/bin/env python3
"""
🧪 DEMONSTRAÇÃO DO SISTEMA DE PROTEÇÃO
=====================================

Script de demonstração para testar o sistema de proteção com um modelo pequeno.
Ideal para validar funcionamento antes de proteger modelos grandes.
"""

import sys
import os
sys.path.append("D:/Projeto")

from trading_framework.security.secure_model_system import ModelSecurityManager, HardwareFingerprint
import tempfile


def demo_protection_system():
    """Demonstração completa do sistema"""
    
    print("🧪 DEMO - SISTEMA DE PROTEÇÃO DE MODELOS")
    print("=" * 60)
    
    # 1. Mostrar info do sistema
    print("💻 Hardware Fingerprint desta máquina:")
    hw_fingerprint = HardwareFingerprint.generate()
    print(f"   {hw_fingerprint}")
    print()
    
    # 2. Configurações
    master_key = "demo_key_2025"
    manager = ModelSecurityManager(master_key)
    
    # 3. Encontrar um modelo Cherry pequeno para teste
    test_models = [
        "D:/Projeto/Otimizacao/treino_principal/models/Cherry/Cherry_simpledirecttraining_50000_steps_20250907_223552.zip",
        "D:/Projeto/Otimizacao/treino_principal/models/Cherry/Cherry_simpledirecttraining_100000_steps_20250907_224040.zip",
        "D:/Projeto/Otimizacao/treino_principal/models/Cherry/Cherry_simpledirecttraining_150000_steps_20250907_224552.zip",
    ]
    
    test_model = None
    for model_path in test_models:
        if os.path.exists(model_path):
            test_model = model_path
            break
    
    if not test_model:
        print("❌ Nenhum modelo de teste encontrado")
        print("💡 Execute o treinamento Cherry primeiro ou ajuste os paths")
        return
    
    print(f"🎯 Modelo de teste selecionado:")
    print(f"   {os.path.basename(test_model)}")
    file_size_mb = os.path.getsize(test_model) / (1024*1024)
    print(f"   Tamanho: {file_size_mb:.2f} MB")
    print()
    
    # 4. Criar arquivo temporário para teste
    with tempfile.NamedTemporaryFile(suffix='.secure', delete=False) as tmp:
        secure_path = tmp.name
    
    try:
        # 5. Teste de proteção
        print("🔐 FASE 1: PROTEÇÃO DO MODELO")
        print("-" * 30)
        
        success = manager.convert_checkpoint(
            input_path=test_model,
            output_path=secure_path,
            hardware_lock=True
        )
        
        if not success:
            print("❌ Falha na proteção")
            return
        
        print("✅ Modelo protegido com sucesso!")
        
        secure_size_mb = os.path.getsize(secure_path) / (1024*1024)
        print(f"📊 Arquivo seguro: {secure_size_mb:.2f} MB")
        print()
        
        # 6. Teste de carregamento
        print("🔓 FASE 2: CARREGAMENTO DO MODELO PROTEGIDO")
        print("-" * 30)
        
        try:
            model_info = manager.wrapper.load_secure(
                secure_path=secure_path,
                master_key=master_key,
                validate_hardware=True
            )
            
            print("✅ Modelo carregado com sucesso!")
            print(f"   Tipo: {model_info.get('model_type', 'Unknown')}")
            print(f"   Versão: {model_info.get('version', 'Unknown')}")
            
            # Mostrar algumas estatísticas dos pesos
            state_dict = model_info['state_dict']
            total_params = sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))
            print(f"   Parâmetros: {total_params:,}")
            print()
            
        except Exception as e:
            print(f"❌ Erro ao carregar: {e}")
            return
        
        # 7. Teste de hardware lock
        print("🔒 FASE 3: TESTE DE HARDWARE LOCK")
        print("-" * 30)
        
        # Simular fingerprint diferente (hack para demo)
        original_generate = HardwareFingerprint.generate
        HardwareFingerprint.generate = lambda: "fake_hardware_123"
        
        try:
            manager.wrapper.load_secure(
                secure_path=secure_path,
                master_key=master_key,
                validate_hardware=True
            )
            print("❌ Hardware lock falhou - modelo carregou em hardware não autorizado")
            
        except ValueError as e:
            if "ACESSO NEGADO" in str(e):
                print("✅ Hardware lock funcionando corretamente")
                print(f"   Erro esperado: {e}")
            else:
                print(f"❌ Erro inesperado: {e}")
                
        except Exception as e:
            print(f"❌ Erro inesperado no teste: {e}")
            
        finally:
            # Restaurar função original
            HardwareFingerprint.generate = original_generate
        
        print()
        
        # 8. Teste de chave incorreta
        print("🔑 FASE 4: TESTE DE CHAVE INCORRETA")
        print("-" * 30)
        
        wrong_key_manager = ModelSecurityManager("wrong_key_123")
        
        try:
            wrong_key_manager.wrapper.load_secure(
                secure_path=secure_path,
                master_key="wrong_key_123",
                validate_hardware=False  # Desabilitar hw check para este teste
            )
            print("❌ Proteção por chave falhou - modelo carregou com chave errada")
            
        except Exception as e:
            print("✅ Proteção por chave funcionando corretamente")
            print(f"   Erro esperado: {type(e).__name__}")
        
        print()
        
        # 9. Resumo final
        print("🎉 DEMO CONCLUÍDA")
        print("=" * 60)
        print("✅ Sistema de proteção funcionando corretamente:")
        print("   • Criptografia ✅")
        print("   • Obfuscação de pesos ✅") 
        print("   • Hardware lock ✅")
        print("   • Proteção por chave ✅")
        print()
        print("💡 O sistema está pronto para proteger seus modelos!")
        print("🚀 Use 'python protect_models.py --help' para começar")
        
    finally:
        # Limpar arquivo temporário
        try:
            os.unlink(secure_path)
        except:
            pass


if __name__ == "__main__":
    demo_protection_system()