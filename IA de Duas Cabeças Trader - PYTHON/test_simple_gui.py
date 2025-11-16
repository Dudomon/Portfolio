#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 TESTE: GUI Simples do RobotV7
Testar se a nova GUI simples funciona sem travamentos
"""

import subprocess
import sys
import time
import os

def test_simple_gui():
    """Testar GUI simples"""
    robot_path = "Modelo PPO Trader/RobotV7.py"
    
    if not os.path.exists(robot_path):
        print(f"❌ RobotV7 não encontrado: {robot_path}")
        return False
    
    print("🧪 TESTE: GUI Simples")
    print("=" * 30)
    print("⏳ Iniciando RobotV7 com GUI simples...")
    
    try:
        # Lançar RobotV7
        process = subprocess.Popen([sys.executable, robot_path])
        
        # Aguardar inicialização
        time.sleep(5)
        
        # Verificar se ainda está rodando
        if process.poll() is None:
            print("✅ RobotV7 com GUI simples iniciado!")
            print()
            print("📋 TESTE DE MINIMIZAÇÃO:")
            print("1. Verifique se a janela apareceu")
            print("2. Minimize a janela")
            print("3. Clique na barra de tarefas para restaurar")
            print("4. Repita várias vezes")
            print()
            
            # Aguardar teste
            input("Pressione Enter quando terminar o teste...")
            
            # Perguntar resultado
            result = input("\nA GUI simples funcionou sem travamentos? (s/N): ")
            
            # Finalizar
            try:
                process.terminate()
                time.sleep(2)
                if process.poll() is None:
                    process.kill()
                print("✅ Processo finalizado")
            except:
                pass
            
            return result.lower() == 's'
        else:
            print("❌ RobotV7 fechou inesperadamente")
            return False
            
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_gui()
    
    if success:
        print("🎉 SUCESSO! GUI simples funciona corretamente!")
    else:
        print("❌ Ainda há problemas com a GUI")
    
    input("\nPressione Enter para sair...")
