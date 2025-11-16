#!/usr/bin/env python3
import sys
import os

# Configurar path
if getattr(sys, 'frozen', False):
    current_dir = os.path.dirname(sys.executable)
else:
    current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, current_dir)

print("🚀 RobotV7 Legion - Carregando...")

try:
    # Usar python.exe local ou de sistema
    python_exe = "python"  # Assumir que Python está no PATH
    
    # Importar diretamente (sem subprocess) - Robot_1min com sistema de login
    import Robot_1min
    
except Exception as e:
    print(f"❌ Erro: {e}")
    input("Pressione Enter para sair...")