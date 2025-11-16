#!/usr/bin/env python3
import sys
import os

# Configurar path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("🚀 RobotV7 Legion - Iniciando...")

# Executar Robot_1min diretamente (sem subprocess, sem main())
try:
    # Robot_1min.py executa automaticamente quando importado
    exec(open(os.path.join(current_dir, "Robot_1min.py")).read())
    
except Exception as e:
    print(f"❌ Erro: {e}")
    import traceback
    traceback.print_exc()
    input("Pressione Enter para sair...")