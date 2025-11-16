"""
🧪 TESTE: Verificar se checkpoint saving está funcionando corretamente
"""
import os
import sys
sys.path.append("D:/Projeto")

from sb3_contrib import RecurrentPPO
from cherry import setup_gpu, TradingEnv
import pandas as pd
import numpy as np

print("=" * 60)
print("🧪 TESTE DE SALVAMENTO DE CHECKPOINT")
print("=" * 60)

# 1. Setup GPU
print("\n1️⃣ Configurando GPU...")
setup_gpu()

# 2. Criar environment simples
print("\n2️⃣ Criando environment de teste...")
data = pd.DataFrame({
    'open_1m': np.random.randn(1000) * 10 + 2000,
    'high_1m': np.random.randn(1000) * 10 + 2010,
    'low_1m': np.random.randn(1000) * 10 + 1990,
    'close_1m': np.random.randn(1000) * 10 + 2000,
    'volume_1m': np.random.randint(100, 1000, 1000),
})

env = TradingEnv(
    df=data,
    window_size=20,
    is_training=True,
    initial_balance=500.0
)

print(f"✅ Environment criado: {len(data)} steps")

# 3. Criar modelo
print("\n3️⃣ Criando modelo RecurrentPPO...")
model = RecurrentPPO(
    "MlpLstmPolicy",
    env,
    learning_rate=0.0001,
    n_steps=128,
    batch_size=64,
    verbose=1,
    device="cuda"
)
print("✅ Modelo criado")

# 4. Treinar por 256 steps apenas (2 rollouts)
print("\n4️⃣ Treinando 256 steps...")
model.learn(total_timesteps=256)
print("✅ Treino concluído")

# 5. Testar salvamento
print("\n5️⃣ Testando salvamento de checkpoint...")
test_path = "D:/Projeto/test_checkpoint_save"

try:
    # Remover logger temporariamente (igual ao cherry.py corrigido)
    logger_backup = None
    if hasattr(model, 'logger') and model.logger:
        logger_backup = model.logger
        model.logger = None

    # Salvar
    model.save(test_path)

    # Restaurar logger
    if logger_backup:
        model.logger = logger_backup

    print(f"✅ Checkpoint salvo: {test_path}.zip")

    # Verificar se arquivo existe e não está vazio
    if os.path.exists(f"{test_path}.zip"):
        size = os.path.getsize(f"{test_path}.zip")
        print(f"✅ Arquivo existe: {size:,} bytes")

        if size > 0:
            print("✅ Arquivo não está vazio!")

            # Tentar carregar de volta
            print("\n6️⃣ Testando carregamento...")
            loaded_model = RecurrentPPO.load(test_path)
            print("✅ Modelo carregado com sucesso!")

            print("\n" + "=" * 60)
            print("🎉 TESTE PASSOU! Checkpoint saving está funcionando!")
            print("=" * 60)
        else:
            print("❌ ERRO: Arquivo está vazio (0 bytes)")
    else:
        print("❌ ERRO: Arquivo não foi criado")

except Exception as e:
    print(f"❌ ERRO ao salvar: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Limpar arquivo de teste
    if os.path.exists(f"{test_path}.zip"):
        os.remove(f"{test_path}.zip")
        print(f"\n🧹 Arquivo de teste removido")
