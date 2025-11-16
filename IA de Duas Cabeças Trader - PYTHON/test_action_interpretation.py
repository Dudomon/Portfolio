"""
🧪 TESTE: Validar interpretação do action space

Verificar se quando modelo prediz ação SHORT, realmente executa SHORT
"""

import sys
import os
import numpy as np
sys.path.insert(0, 'D:/Projeto')

from cherry import TradingEnv, load_1m_dataset, ACTION_THRESHOLD_LONG, ACTION_THRESHOLD_SHORT
from stable_baselines3 import PPO

def test_action_interpretation():
    print("\n" + "="*80)
    print("🧪 TESTE: INTERPRETAÇÃO DO ACTION SPACE")
    print("="*80)

    checkpoint_path = "D:/Projeto/Otimizacao/treino_principal/models/Oitavo/Oitavo_simpledirecttraining_4700000_steps_20251008_042753.zip"

    print(f"\n📊 Thresholds configurados:")
    print(f"   ACTION_THRESHOLD_LONG:  {ACTION_THRESHOLD_LONG}")
    print(f"   ACTION_THRESHOLD_SHORT: {ACTION_THRESHOLD_SHORT}")
    print(f"\n   Interpretação:")
    print(f"   < {ACTION_THRESHOLD_LONG}  → HOLD")
    print(f"   {ACTION_THRESHOLD_LONG} a {ACTION_THRESHOLD_SHORT} → LONG")
    print(f"   >= {ACTION_THRESHOLD_SHORT} → SHORT")

    # Carregar
    df = load_1m_dataset()
    env = TradingEnv(df=df, window_size=10)
    model = PPO.load(checkpoint_path, env=env)
    print("\n✅ Modelo carregado")

    # Testar 50 ações
    print(f"\n📊 Testando 50 ações do modelo:")
    print(f"{'Step':<6} {'Raw[0]':<10} {'Interpretado':<12} {'Esperado':<12} {'Match':<6}")
    print("="*60)

    obs = env.reset()
    matches = 0
    total = 0

    for step in range(50):
        action, _states = model.predict(obs, deterministic=False)

        raw_value = action[0]

        # INTERPRETAÇÃO CORRETA
        if raw_value < ACTION_THRESHOLD_LONG:
            expected = "HOLD"
        elif raw_value < ACTION_THRESHOLD_SHORT:
            expected = "LONG"
        else:
            expected = "SHORT"

        # Executar ação e ver o que env faz
        obs, reward, done, info = env.step(action)

        # Verificar se criou trade
        if hasattr(env, 'trades') and len(env.trades) > total:
            # Novo trade criado
            last_trade = env.trades[-1]
            actual = last_trade.get('type', 'unknown').upper()
            match = (actual == expected)
            matches += (1 if match else 0)
            total += 1

            status = "✅" if match else "🚨"
            print(f"{step:<6} {raw_value:<10.4f} {actual:<12} {expected:<12} {status}")

        if done:
            obs = env.reset()

    print("="*80)
    if total > 0:
        accuracy = matches / total * 100
        print(f"\n📊 RESULTADO: {matches}/{total} matches ({accuracy:.1f}%)")

        if accuracy > 95:
            print("✅ Interpretação CORRETA - Model e Env estão alinhados")
            return True
        else:
            print("🚨 INTERPRETAÇÃO ERRADA - Model e Env estão DESALINHADOS!")
            return False
    else:
        print("\n⚠️ Nenhum trade executado para validar")
        return None

if __name__ == "__main__":
    try:
        result = test_action_interpretation()
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
