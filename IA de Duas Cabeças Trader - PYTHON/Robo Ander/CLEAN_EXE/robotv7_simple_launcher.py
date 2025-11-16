#!/usr/bin/env python3
import sys
import os
import subprocess

def main():
    # Detectar se estamos rodando como executável ou script
    if getattr(sys, 'frozen', False):
        # Executável PyInstaller - usar diretório do exe
        base_dir = os.path.dirname(sys.executable)
    else:
        # Script Python normal - usar diretório do script
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Tentar encontrar robotlogin.pyc em locais possíveis
    possible_paths = [
        os.path.join(base_dir, "robotlogin.pyc"),
        os.path.join(base_dir, "__pycache__", "robotlogin.pyc"),
        os.path.join(os.path.dirname(base_dir), "__pycache__", "robotlogin.pyc"),
        os.path.join(base_dir, "..", "__pycache__", "robotlogin.pyc")
    ]
    
    robotlogin_path = None
    for path in possible_paths:
        if os.path.exists(path):
            robotlogin_path = path
            break
    
    if not robotlogin_path:
        print("ERRO: robotlogin.pyc nao encontrado!")
        print("Procurado em:")
        for path in possible_paths:
            print(f"  {path}")
        input("Pressione Enter para sair...")
        return
    
    print(f"Encontrado: {robotlogin_path}")
    print("Iniciando RobotV7...")
    
    # Executar robotlogin.pyc
    try:
        subprocess.run([sys.executable, robotlogin_path], check=True)
    except Exception as e:
        print(f"Erro ao executar: {e}")
        input("Pressione Enter para sair...")

if __name__ == "__main__":
    main()