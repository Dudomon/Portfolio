#!/usr/bin/env python3
import sys
import os
import subprocess

# Configurar path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("🚀 RobotV7 Legion - Iniciando...")

try:
    # Executar o Robot_1min.pyc diretamente
    python_cmd = [sys.executable, os.path.join(current_dir, "Robot_1min.pyc")]
    subprocess.run(python_cmd)
    
except Exception as e:
    print(f"❌ Erro: {e}")
    input("Pressione Enter para sair...")