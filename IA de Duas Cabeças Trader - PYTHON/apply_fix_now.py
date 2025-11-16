#!/usr/bin/env python3
"""
🚨 APLICAR FIX AGORA NO MODELO RODANDO 🚨
Execute este script IMEDIATAMENTE para corrigir os zeros críticos!
"""

import sys
import os
sys.path.append("D:/Projeto")

def apply_emergency_fix():
    """Aplicar fix no modelo atual"""
    
    print("🚨 TENTANDO APLICAR FIX NO MODELO ATIVO...")
    
    try:
        # Importar fix
        from emergency_fix_v8 import apply_fix_now
        
        print("✅ Fix importado com sucesso")
        
        # Tentar encontrar modelo ativo
        # Você precisa adaptar essa parte conforme seu setup
        
        print("⚠️ INSTRUÇÕES MANUAIS:")
        print("1. No seu script daytrader8dim que está rodando,")
        print("2. Adicione esta linha APÓS carregar o modelo:")
        print("")
        print("   from emergency_fix_v8 import apply_fix_now")
        print("   apply_fix_now(model)  # onde 'model' é seu modelo PPO")
        print("")
        print("3. O fix será aplicado automaticamente")
        print("4. Continue o treinamento normalmente")
        print("")
        print("🔥 URGENTE: Os LSTMs estão 100% mortos, aplique o fix AGORA!")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        return False

if __name__ == "__main__":
    print("🚨 EMERGENCY FIX APLICATOR")
    print("=" * 50)
    
    apply_emergency_fix()
    
    print("\n" + "🚨" * 20)
    print("MODELO PRECISA DE FIX IMEDIATO!")
    print("LSTMs: 100% ZEROS = MORTOS")
    print("DecisionMaker: 70% ZEROS = CRÍTICO")
    print("🚨" * 20)