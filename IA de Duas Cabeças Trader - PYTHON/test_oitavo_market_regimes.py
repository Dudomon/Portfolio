"""
🧪 TESTE: Viés SHORT em diferentes regimes de mercado - Oitavo 4.7M

Testa se o modelo adapta estratégia conforme regime:
- Bull market (alta) → deveria preferir LONG
- Bear market (baixa) → deveria preferir SHORT
- Sideways (lateral) → deveria ser balanceado

Se modelo tem VIÉS SHORT: fará mais SHORTs mesmo em bull market
Se modelo é INTELIGENTE: adaptará estratégia ao regime
"""

import sys
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Importar ambiente
sys.path.insert(0, 'D:/Projeto')
from cherry import TradingEnv, load_1m_dataset

def identify_market_regime(df, start_idx, window=100):
    """
    Identifica regime de mercado baseado em retornos

    Returns:
        regime (str): 'BULL', 'BEAR', 'SIDEWAYS'
        trend_strength (float): força da tendência (0-1)
    """
    if start_idx + window > len(df):
        window = len(df) - start_idx

    prices = df['close_1m'].iloc[start_idx:start_idx+window]

    # Calcular retorno total e volatilidade
    total_return = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
    volatility = prices.pct_change().std()

    # Calcular % de closes subindo vs descendo
    returns = prices.pct_change().dropna()
    up_pct = (returns > 0).sum() / len(returns)

    # Classificar regime
    if total_return > 0.005 and up_pct > 0.55:  # >0.5% retorno e >55% closes positivos
        regime = 'BULL'
        trend_strength = min(abs(total_return) * 100, 1.0)
    elif total_return < -0.005 and up_pct < 0.45:  # <-0.5% retorno e <45% closes positivos
        regime = 'BEAR'
        trend_strength = min(abs(total_return) * 100, 1.0)
    else:
        regime = 'SIDEWAYS'
        trend_strength = 0.3  # Mercado lateral tem baixa tendência

    return regime, trend_strength, total_return

def find_market_regimes(df, num_samples=5, window=200):
    """
    Encontra períodos representativos de cada regime no dataset

    Returns:
        dict: {regime: [(start_idx, end_idx, return, strength), ...]}
    """
    regimes = {'BULL': [], 'BEAR': [], 'SIDEWAYS': []}

    # Buscar em intervalos de 1000 barras
    search_step = 1000
    max_idx = len(df) - window

    print(f"\n🔍 Buscando regimes de mercado no dataset ({len(df):,} barras)...")

    for start_idx in range(100, max_idx, search_step):
        regime, strength, ret = identify_market_regime(df, start_idx, window)

        # Adicionar apenas se for representativo
        if len(regimes[regime]) < num_samples:
            regimes[regime].append((start_idx, start_idx + window, ret, strength))
            print(f"   ✅ {regime} encontrado: idx {start_idx:,} | Return: {ret*100:.2f}% | Strength: {strength:.2f}")

        # Parar quando tiver amostras suficientes de todos
        if all(len(samples) >= num_samples for samples in regimes.values()):
            break

    return regimes

def test_regime(model, env, regime_name, start_idx, end_idx, steps=100):
    """
    Testa modelo em um regime específico

    Returns:
        dict: estatísticas do teste
    """
    # Reset environment para período específico
    env.current_step = start_idx
    obs = env._get_observation()

    entry_long = 0
    entry_short = 0
    hold_actions = 0

    for step in range(min(steps, end_idx - start_idx)):
        action, _states = model.predict(obs, deterministic=False)

        # Analisar ação - ACTION SPACE 4D:
        # [0] entry_decision: [-1,1] (< -0.33=hold, -0.33 a 0.33=long, > 0.33=short)
        entry_decision = action[0]

        # Detectar tipo de ação
        if entry_decision < -0.33:  # HOLD
            hold_actions += 1
        elif -0.33 <= entry_decision <= 0.33:  # LONG
            entry_long += 1
        else:  # entry_decision > 0.33 = SHORT
            entry_short += 1

        obs, reward, done, info = env.step(action)

        if done or env.current_step >= end_idx:
            break

    total_entries = entry_long + entry_short

    return {
        'regime': regime_name,
        'long': entry_long,
        'short': entry_short,
        'hold': hold_actions,
        'total_entries': total_entries,
        'long_pct': entry_long / (total_entries + 0.001) * 100,
        'short_pct': entry_short / (total_entries + 0.001) * 100,
        'ratio': entry_long / (entry_short + 0.001)
    }

def test_market_regimes_oitavo():
    """Testa viés SHORT do Oitavo 4.7M em diferentes regimes"""
    print("\n" + "=" * 80)
    print("🧪 TESTE DE VIÉS POR REGIME DE MERCADO - OITAVO 4.7M")
    print("=" * 80)

    checkpoint_path = "D:/Projeto/Otimizacao/treino_principal/models/Oitavo/Oitavo_simpledirecttraining_4700000_steps_20251008_042753.zip"

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint não encontrado: {checkpoint_path}")
        return False

    print(f"\n📦 Carregando modelo: {checkpoint_path}")

    # Carregar dados
    print("📊 Carregando dados históricos...")
    df = load_1m_dataset()
    print(f"✅ Dados carregados: {len(df):,} barras")

    # Encontrar regimes
    regimes_samples = find_market_regimes(df, num_samples=3, window=200)

    # Criar ambiente
    env = TradingEnv(df=df, window_size=10)

    # Carregar modelo
    try:
        model = PPO.load(checkpoint_path, env=env)
        print("\n✅ Modelo carregado com sucesso\n")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return False

    # Testar cada regime
    print("=" * 80)
    print("📊 TESTANDO COMPORTAMENTO EM CADA REGIME (100 steps cada)")
    print("=" * 80)

    all_results = []

    for regime_name in ['BULL', 'SIDEWAYS', 'BEAR']:
        samples = regimes_samples[regime_name]

        if not samples:
            print(f"\n⚠️ Regime {regime_name} não encontrado no dataset")
            continue

        print(f"\n{'='*80}")
        print(f"🎯 REGIME: {regime_name}")
        print(f"{'='*80}")

        regime_results = []

        for i, (start_idx, end_idx, ret, strength) in enumerate(samples, 1):
            print(f"\n📍 Amostra {i}/{len(samples)} - Período: {start_idx:,} a {end_idx:,}")
            print(f"   Return: {ret*100:.2f}% | Strength: {strength:.2f}")

            result = test_regime(model, env, regime_name, start_idx, end_idx, steps=100)
            regime_results.append(result)

            print(f"   🟢 LONG: {result['long']} ({result['long_pct']:.1f}%)")
            print(f"   🔴 SHORT: {result['short']}) ({result['short_pct']:.1f}%)")
            print(f"   ⏸️  HOLD: {result['hold']}")
            print(f"   📊 Ratio L/S: {result['ratio']:.2f}")

        # Agregar resultados do regime
        total_long = sum(r['long'] for r in regime_results)
        total_short = sum(r['short'] for r in regime_results)
        total_hold = sum(r['hold'] for r in regime_results)
        total_entries = total_long + total_short

        regime_summary = {
            'regime': regime_name,
            'long': total_long,
            'short': total_short,
            'hold': total_hold,
            'total_entries': total_entries,
            'long_pct': total_long / (total_entries + 0.001) * 100,
            'short_pct': total_short / (total_entries + 0.001) * 100,
            'ratio': total_long / (total_short + 0.001)
        }

        all_results.append(regime_summary)

        print(f"\n{'─'*80}")
        print(f"📊 RESUMO {regime_name}:")
        print(f"   🟢 LONG Total: {total_long} ({regime_summary['long_pct']:.1f}%)")
        print(f"   🔴 SHORT Total: {total_short} ({regime_summary['short_pct']:.1f}%)")
        print(f"   📊 Ratio L/S: {regime_summary['ratio']:.2f}")

    # Análise final
    print("\n" + "=" * 80)
    print("🔍 ANÁLISE FINAL: VIÉS SHORT vs INTELIGÊNCIA DE MERCADO")
    print("=" * 80)

    # Procurar resultados de cada regime
    bull_result = next((r for r in all_results if r['regime'] == 'BULL'), None)
    bear_result = next((r for r in all_results if r['regime'] == 'BEAR'), None)
    side_result = next((r for r in all_results if r['regime'] == 'SIDEWAYS'), None)

    has_short_bias = False
    is_intelligent = False

    print("\n📊 RESUMO POR REGIME:")
    print(f"{'Regime':<15} {'LONG%':<10} {'SHORT%':<10} {'Ratio':<10} {'Diagnóstico'}")
    print("─" * 80)

    for result in all_results:
        regime = result['regime']
        ratio = result['ratio']

        # Diagnóstico
        if regime == 'BULL':
            expected = "LONG > SHORT"
            diagnosis = "✅ CORRETO" if ratio > 1.2 else "⚠️ SUSPEITO"
            if ratio < 0.8:
                diagnosis = "🚨 VIÉS SHORT!"
                has_short_bias = True
            elif ratio > 1.2:
                is_intelligent = True
        elif regime == 'BEAR':
            expected = "SHORT > LONG"
            diagnosis = "✅ CORRETO" if ratio < 0.8 else "⚠️ SUSPEITO"
            if ratio > 1.2:
                diagnosis = "⚠️ Não adapta"
        else:  # SIDEWAYS
            expected = "BALANCEADO"
            diagnosis = "✅ CORRETO" if 0.7 <= ratio <= 1.3 else "⚠️ DESBALANCEADO"

        print(f"{regime:<15} {result['long_pct']:<10.1f} {result['short_pct']:<10.1f} {ratio:<10.2f} {diagnosis}")

    print("\n" + "=" * 80)
    print("🎯 DIAGNÓSTICO FINAL:")
    print("=" * 80)

    # Análise contextual
    if has_short_bias:
        print("🚨 VIÉS SHORT DETECTADO!")
        print("   Modelo faz mais SHORTs mesmo em BULL market")
        print("   ⚠️ Isso indica viés nos rewards, não inteligência")
        return True
    elif is_intelligent:
        if bull_result and bear_result:
            adapts = bull_result['ratio'] > 1.2 and bear_result['ratio'] < 0.8
            if adapts:
                print("✅ MODELO INTELIGENTE - Adapta estratégia ao regime!")
                print("   🟢 BULL market → prefere LONG")
                print("   🔴 BEAR market → prefere SHORT")
                print("   📊 Sem viés - comportamento correto")
            else:
                print("⚠️ MODELO PARCIALMENTE ADAPTATIVO")
                print("   Prefere LONG em bull, mas não inverte em bear")
                print("   Pode ter leve viés LONG (não necessariamente ruim)")
        else:
            print("✅ SEM VIÉS SHORT EVIDENTE")
            print("   Modelo prefere LONG em bull market (comportamento esperado)")
        return False
    else:
        print("⚠️ COMPORTAMENTO AMBÍGUO")
        print("   Não há viés SHORT claro, mas adaptação limitada")
        return False

if __name__ == "__main__":
    try:
        has_bias = test_market_regimes_oitavo()

        print("\n" + "=" * 80)
        print("🎯 RESULTADO FINAL:")
        if has_bias:
            print("❌ VIÉS SHORT DETECTADO - Modelo não adapta a regimes")
            sys.exit(1)
        else:
            print("✅ SEM VIÉS SHORT - Modelo adapta ou tem comportamento normal")
            sys.exit(0)

    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
