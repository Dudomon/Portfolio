#!/usr/bin/env python3
"""
🤖 ROBOTV7 SIMPLE - EXECUTÁVEL DIRETO
===================================

Executa robotlogin.py diretamente sem launcher intermediário
"""

import sys
import os
import subprocess

def main():
    """Executar robotlogin diretamente"""
    print("RobotV7 - Starting...")
    
    # Caminho para robotlogin.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    robotlogin_path = os.path.join(current_dir, "robotlogin.py")
    
    if not os.path.exists(robotlogin_path):
        print(f"ERRO: robotlogin.py não encontrado em: {robotlogin_path}")
        input("Pressione Enter para sair...")
        return
    
    print(f"Executando: {robotlogin_path}")
    
    # Executar robotlogin.py diretamente
    try:
        subprocess.run([sys.executable, robotlogin_path], 
                      cwd=current_dir,
                      check=True)
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar robotlogin.py: {e}")
        input("Pressione Enter para sair...")
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário")

if __name__ == "__main__":
    main()