#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys

print("🚀 Iniciando auto-py-to-exe...")
print("📋 Instruções:")
print("1. Selecione: Robo Ander/RoboAnder.py")
print("2. Escolha: One File")
print("3. Escolha: Console Based")
print("4. Adicione arquivos:")
print("   - Robo Ander/Modelo Ander")
print("   - Robo Ander/enhanced_normalizer_final.pkl")
print("   - Robo Ander/online_login_ander.py")
print("   - Robo Ander/gerenciar_usuarios_online.py")
print("   - Robo Ander/online_system_real.py")
print("   - trading_framework")
print("   - data_cache")
print("   - logs")
print("5. Clique em CONVERT")

try:
    subprocess.check_call([sys.executable, "-m", "auto_py_to_exe"])
except Exception as e:
    print(f"❌ Erro: {e}")
