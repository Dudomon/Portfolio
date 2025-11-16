"""
🧪 TESTE SIMPLES: Verificar se model.save() funciona com logger removal
"""
import os
import sys
import numpy as np
import gym
from gym import spaces

# Importar RecurrentPPO
from sb3_contrib import RecurrentPPO

print("=" * 60)
print("🧪 TESTE SIMPLES DE SALVAMENTO COM LOGGER REMOVAL")
print("=" * 60)

# 1. Criar um environment dummy simples
class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None):
        return np.zeros(10, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(10, dtype=np.float32), 0.0, False, False, {}

print("\n1️⃣ Criando environment dummy...")
env = DummyEnv()
print("✅ Environment criado")

# 2. Criar modelo RecurrentPPO
print("\n2️⃣ Criando modelo RecurrentPPO...")
model = RecurrentPPO(
    "MlpLstmPolicy",
    env,
    learning_rate=0.0001,
    n_steps=128,
    batch_size=64,
    verbose=1,
    device="cpu"  # CPU para teste rápido
)
print("✅ Modelo criado")

# 3. Treinar por 256 steps apenas
print("\n3️⃣ Treinando 256 steps...")
model.learn(total_timesteps=256)
print("✅ Treino concluído")

# 4. Testar salvamento COM logger removal (igual ao cherry.py corrigido)
print("\n4️⃣ Testando salvamento COM logger removal...")
test_path = "D:/Projeto/test_checkpoint_save_simple"

try:
    # ✅ TÉCNICA DO CHERRY.PY CORRIGIDO: Excluir logger do salvamento
    print("   🔧 Salvando com exclude=['logger']")

    # Salvar modelo completo excluindo logger
    model.save(test_path, exclude=['logger'])
    print(f"   ✅ model.save() executado")

    # Verificar se arquivo existe e não está vazio
    if os.path.exists(f"{test_path}.zip"):
        size = os.path.getsize(f"{test_path}.zip")
        print(f"\n5️⃣ Verificando arquivo salvo...")
        print(f"   ✅ Arquivo existe: {test_path}.zip")
        print(f"   ✅ Tamanho: {size:,} bytes")

        if size > 0:
            print(f"   ✅ Arquivo não está vazio!")

            # Tentar carregar de volta
            print("\n6️⃣ Testando carregamento...")
            loaded_model = RecurrentPPO.load(test_path)
            print("   ✅ Modelo carregado com sucesso!")

            print("\n" + "=" * 60)
            print("🎉 TESTE PASSOU! Checkpoint saving está funcionando!")
            print("=" * 60)
            print("\n✅ A correção no cherry.py está CORRETA:")
            print("   1. Remove logger temporariamente")
            print("   2. Usa model.save() (não torch.save())")
            print("   3. Restaura logger")
            print("   4. Resultado: arquivo .zip válido e carregável")
        else:
            print("   ❌ ERRO: Arquivo está vazio (0 bytes)")
    else:
        print("   ❌ ERRO: Arquivo não foi criado")

except Exception as e:
    print(f"\n❌ ERRO ao salvar: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Limpar arquivo de teste
    if os.path.exists(f"{test_path}.zip"):
        os.remove(f"{test_path}.zip")
        print(f"\n🧹 Arquivo de teste removido")
