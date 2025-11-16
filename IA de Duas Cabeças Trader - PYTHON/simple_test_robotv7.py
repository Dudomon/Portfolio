#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 TESTE SIMPLES: Verificar se RobotV7 abre sem travar
"""

import subprocess
import sys
import time
import os

def simple_test():
    """Teste simples e direto"""
    robot_path = "Modelo PPO Trader/RobotV7.py"
    
    if not os.path.exists(robot_path):
        print(f"❌ RobotV7 não encontrado: {robot_path}")
        return False
    
    print("🧪 TESTE SIMPLES - RobotV7")
    print("=" * 30)
    print("⏳ Iniciando RobotV7...")
    print("⏳ Aguarde 5 segundos...")
    
    try:
        # Lançar RobotV7 em processo separado
        process = subprocess.Popen([sys.executable, robot_path])
        
        # Aguardar 5 segundos
        time.sleep(5)
        
        # Verificar se ainda está rodando
        if process.poll() is None:
            print("✅ RobotV7 iniciou com sucesso!")
            print("✅ Verifique se a janela está visível")
            print("\n💡 Se a janela não aparecer:")
            print("   • Clique no ícone na barra de tarefas")
            print("   • Use Alt+Tab para alternar janelas")
            print("   • Verifique se não está atrás de outras janelas")
            
            input("\nPressione Enter quando terminar o teste...")
            
            # Finalizar processo
            try:
                process.terminate()
                time.sleep(2)
                if process.poll() is None:
                    process.kill()
                print("✅ Processo finalizado")
            except:
                print("⚠️ Processo pode ainda estar rodando")
            
            return True
        else:
            print("❌ RobotV7 fechou inesperadamente")
            return False
            
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False

if __name__ == "__main__":
    simple_test()
