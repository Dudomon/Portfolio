#!/usr/bin/env python3
"""
🧪 TESTE de integração do V3 Brutal no sistema de rewards
"""

import sys
import os
sys.path.append('.')

from trading_framework.rewards import create_reward_system

def main():
    """Testar se V3 Brutal está disponível no sistema"""
    print("🚀 TESTE DE INTEGRAÇÃO V3 BRUTAL")
    print("=" * 50)
    
    # Tentar criar o reward system V3 Brutal
    try:
        reward_system = create_reward_system("v3_brutal", initial_balance=1000.0)
        
        if reward_system is not None:
            print("✅ V3 Brutal criado com sucesso!")
            print(f"   Tipo: {type(reward_system).__name__}")
            print(f"   Balance inicial: ${reward_system.initial_balance}")
            
            # Testar função de teste
            reward_system.test_trailing_sltp_rewards()
            
        else:
            print("❌ Falha ao criar V3 Brutal - reward_system é None")
            
    except Exception as e:
        print(f"❌ Erro ao criar V3 Brutal: {e}")
        return False
        
    print("\n🎯 INTEGRAÇÃO COMPLETA!")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ V3 Brutal está pronto para uso no SILUS!")
    else:
        print("❌ Problema na integração - verificar imports")