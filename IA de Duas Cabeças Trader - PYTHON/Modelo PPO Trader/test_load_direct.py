import os
import sys

# Adicionar path para encontrar trading_framework
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)

from sb3_contrib import RecurrentPPO

# Teste direto do ZIP
zip_path = os.path.join(os.getcwd(), "modelo daytrade", "Legion daytrade.zip")
print(f"Carregando: {zip_path}")

try:
    model = RecurrentPPO.load(zip_path)
    print(f"Sucesso! Modelo: {type(model).__name__}")
    print(f"Policy: {type(model.policy).__name__}")
except Exception as e:
    print(f"Erro: {e}")
    import traceback
    traceback.print_exc()