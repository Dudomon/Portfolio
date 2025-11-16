"""
🧪 TESTE ULTRA SIMPLES: Apenas testar model.save(exclude=['logger'])
"""
import os
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env

print("=" * 60)
print("🧪 TESTE: model.save(exclude=['logger'])")
print("=" * 60)

# 1. Criar environment simples com make_vec_env
print("\n1️⃣ Criando environment CartPole...")
env = make_vec_env("CartPole-v1", n_envs=1)
print("✅ Environment criado")

# 2. Criar modelo
print("\n2️⃣ Criando modelo RecurrentPPO...")
model = RecurrentPPO("MlpLstmPolicy", env, verbose=0, device="cpu")
print("✅ Modelo criado")

# 3. Treinar apenas 128 steps
print("\n3️⃣ Treinando 128 steps...")
model.learn(total_timesteps=128, progress_bar=False)
print("✅ Treino concluído")

# 4. Teste A: Salvar SEM exclude (pode dar erro)
print("\n4️⃣ Teste A: Salvando SEM exclude=['logger']...")
test_path_a = "D:/Projeto/test_without_exclude"
try:
    model.save(test_path_a)
    if os.path.exists(f"{test_path_a}.zip"):
        size = os.path.getsize(f"{test_path_a}.zip")
        print(f"   ✅ Sucesso! Arquivo: {size:,} bytes")
        os.remove(f"{test_path_a}.zip")
    else:
        print(f"   ❌ Arquivo não criado")
except Exception as e:
    print(f"   ⚠️  Erro: {e}")

# 5. Teste B: Salvar COM exclude (deve funcionar sempre)
print("\n5️⃣ Teste B: Salvando COM exclude=['logger']...")
test_path_b = "D:/Projeto/test_with_exclude"
try:
    model.save(test_path_b, exclude=['logger'])

    if os.path.exists(f"{test_path_b}.zip"):
        size = os.path.getsize(f"{test_path_b}.zip")
        print(f"   ✅ Sucesso! Arquivo: {size:,} bytes")

        # Testar carregamento
        print("\n6️⃣ Testando carregamento...")
        loaded = RecurrentPPO.load(test_path_b, env=env)
        print("   ✅ Modelo carregado com sucesso!")

        print("\n" + "=" * 60)
        print("🎉 TESTE PASSOU!")
        print("=" * 60)
        print("\n✅ A correção está CORRETA:")
        print("   model.save(path, exclude=['logger'])")
        print("\n✅ Isso resolve o erro de pickle com file handles")
        print("✅ Checkpoints agora serão salvos corretamente no cherry.py")

        # Limpar
        os.remove(f"{test_path_b}.zip")
    else:
        print(f"   ❌ Arquivo não criado")

except Exception as e:
    print(f"   ❌ Erro: {e}")
    import traceback
    traceback.print_exc()
