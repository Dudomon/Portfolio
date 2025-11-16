#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚨 LAUNCHER DE EMERGÊNCIA - RobotV7
Use este script se o RobotV7 estiver travando
"""

import os
import sys
import subprocess
import time

def kill_existing_robots():
    """Matar processos RobotV7 existentes"""
    try:
        # Windows
        subprocess.run(['taskkill', '/F', '/IM', 'python.exe'], 
                      capture_output=True, shell=True)
        subprocess.run(['taskkill', '/F', '/IM', 'pythonw.exe'], 
                      capture_output=True, shell=True)
        time.sleep(2)
        print("✅ Processos Python anteriores finalizados")
    except Exception as e:
        print(f"⚠️ Erro ao finalizar processos: {e}")

def launch_robot_safe():
    """Lançar RobotV7 de forma segura"""
    robot_path = "Modelo PPO Trader/RobotV7.py"
    
    if not os.path.exists(robot_path):
        print(f"❌ RobotV7 não encontrado: {robot_path}")
        return False
    
    print("🚀 Iniciando RobotV7 em modo seguro...")
    
    try:
        # Lançar em processo separado
        if sys.platform == "win32":
            subprocess.Popen([sys.executable, robot_path], 
                           creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            subprocess.Popen([sys.executable, robot_path])
        
        print("✅ RobotV7 iniciado com sucesso!")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao iniciar RobotV7: {e}")
        return False

if __name__ == "__main__":
    print("🔧 LAUNCHER DE EMERGÊNCIA - RobotV7")
    print("=" * 50)
    
    # Opção de matar processos existentes
    response = input("Finalizar processos Python existentes? (S/n): ")
    if response.lower() != 'n':
        kill_existing_robots()
    
    # Lançar RobotV7
    if launch_robot_safe():
        print("\n✅ RobotV7 deve estar rodando agora")
        print("Se ainda estiver travado, verifique a barra de tarefas")
    else:
        print("\n❌ Falha ao iniciar RobotV7")
        input("Pressione Enter para sair...")
