#!/usr/bin/env python3
"""
🧪 TESTE TRAINING V3 BRUTAL
Validação com 1000 steps de training real
"""

import sys
sys.path.append("D:/Projeto")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configurações para teste rápido
QUICK_TEST = True
TEST_STEPS = 1000

if QUICK_TEST:
    # Override das configurações para teste rápido
    os.environ["DAYTRADER_QUICK_TEST"] = "1"
    os.environ["DAYTRADER_TEST_STEPS"] = str(TEST_STEPS)

import daytrader

def run_v3_test():
    print("🧪 INICIANDO TESTE V3 BRUTAL - 1000 STEPS")
    print("=" * 60)
    
    try:
        # Executar com parâmetros de teste
        if hasattr(daytrader, 'main'):
            daytrader.main()
        else:
            print("⚠️ Função main não encontrada, importando classe diretamente")
            
    except KeyboardInterrupt:
        print("\n🛑 Teste interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro durante teste: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_v3_test()