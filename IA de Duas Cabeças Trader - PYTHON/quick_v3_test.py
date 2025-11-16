#!/usr/bin/env python3
"""
🧪 TESTE RÁPIDO V3 BRUTAL - 1000 STEPS
Modifica temporariamente as configurações para teste rápido
"""

import sys
sys.path.append("D:/Projeto")

# Importar e patch das configurações antes do daytrader
import daytrader

# Fazer backup e modificar configuração temporariamente
original_config = daytrader.TRAINING_CONFIG.copy()

# Configurações para teste rápido
daytrader.TRAINING_CONFIG.update({
    "total_timesteps": 1000,  # Apenas 1000 steps para teste
    "max_dataset_bars": 10000,  # Dataset menor
})

print("🧪 CONFIGURAÇÃO TEMPORÁRIA PARA TESTE V3:")
print(f"✅ Total timesteps: {daytrader.TRAINING_CONFIG['total_timesteps']}")
print(f"✅ Max dataset bars: {daytrader.TRAINING_CONFIG['max_dataset_bars']}")

def quick_test():
    try:
        print("\n🧪 INICIANDO TESTE RÁPIDO V3 BRUTAL...")
        print("=" * 60)
        
        # Executar main com configuração modificada
        daytrader.main()
        
        print("\n🎯 TESTE V3 BRUTAL CONCLUÍDO COM SUCESSO! 🚀")
        
    except KeyboardInterrupt:
        print("\n🛑 Teste interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro durante teste: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restaurar configuração original
        daytrader.TRAINING_CONFIG.update(original_config)
        print(f"\n🔄 Configuração original restaurada")

if __name__ == "__main__":
    quick_test()