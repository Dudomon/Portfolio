#!/usr/bin/env python3
"""
🎯 TESTE SIMPLES DE ATIVIDADE - SILUS 450K
Teste rápido para verificar quantos trades o modelo realmente executa
"""

import sys
import os
sys.path.append("D:/Projeto")

import numpy as np
import torch
from stable_baselines3 import PPO

def test_activity_450k():
    """
    Teste simples da atividade do SILUS 450K
    """
    print("🎯 TESTE DE ATIVIDADE - LEGION V1")
    print("=" * 50)
    
    # Carregar modelo
    model_path = "D:/Projeto/Modelos para testar/Legion V1.zip"
    
    try:
        model = PPO.load(model_path)
        print("✅ Modelo Legion V1 carregado com sucesso")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return
    
    # Setup ambiente simples
    from silus import load_optimized_data_original, TradingEnv
    
    print("📊 Carregando dados...")
    data = load_optimized_data_original()
    print(f"✅ Dados carregados: {len(data)} barras")
    
    # Criar ambiente
    env = TradingEnv(
        df=data,
        window_size=20,
        is_training=False,
        initial_balance=500.0
    )
    
    # 🎯 CARREGAR NORMALIZER IGUAL AO ROBOTV7
    print("📊 Carregando normalizer...")
    normalizer = None
    try:
        import pickle
        normalizer_path = "D:/Projeto/enhanced_normalizer_final.pkl"
        with open(normalizer_path, 'rb') as f:
            normalizer = pickle.load(f)
        print("✅ Normalizer carregado (igual ao RobotV7)")
    except Exception as e:
        print(f"⚠️ Erro ao carregar normalizer: {e}")
        normalizer = None
    
    print("\\n🔄 Executando teste de 5 episódios...")
    
    total_trades = 0
    total_steps = 0
    episodes_with_trades = 0
    
    for episode in range(5):
        print(f"\\n📈 Episódio {episode + 1}/5")
        
        obs = env.reset()
        episode_trades = 0
        episode_steps = 0
        
        # Rodar por 500 steps (verificação rápida)
        for step in range(500):
            # 🎯 APLICAR NORMALIZER IGUAL AO ROBOTV7
            obs_for_model = obs.copy()
            if normalizer is not None:
                try:
                    obs_for_model = normalizer.normalize_obs(obs_for_model)
                except Exception as e:
                    print(f"    ⚠️ Erro na normalização: {e}")
            
            # Predict (stochastic para ter variação)
            action, _ = model.predict(obs_for_model, deterministic=False)
            
            # Step
            obs, reward, done, info = env.step(action)
            episode_steps += 1
            total_steps += 1
            
            # Contar trades
            if hasattr(env, 'positions') and len(env.positions) > 0:
                episode_trades = len(env.positions)
            
            # Check for trade info
            if 'trade_executed' in info and info['trade_executed']:
                episode_trades += 1
                print(f"    Step {step}: TRADE EXECUTADO!")
            
            if done:
                break
        
        total_trades += episode_trades
        if episode_trades > 0:
            episodes_with_trades += 1
        
        print(f"    Trades: {episode_trades}, Steps: {episode_steps}")
    
    # Resultados
    print("\\n📊 RESULTADOS:")
    print("-" * 30)
    print(f"Total de episódios: 5")
    print(f"Total de trades: {total_trades}")
    print(f"Episódios com trades: {episodes_with_trades}")
    print(f"Taxa de atividade: {episodes_with_trades/5*100:.1f}%")
    print(f"Trades por episódio: {total_trades/5:.2f}")
    print(f"Total steps: {total_steps}")
    print(f"Trades por 100 steps: {total_trades/total_steps*100:.2f}")
    
    # Análise
    print("\\n🔍 ANÁLISE:")
    if total_trades == 0:
        print("❌ PROBLEMA: Nenhum trade executado!")
        print("   Possíveis causas:")
        print("   - Confidence threshold muito alto")
        print("   - Modelo muito conservador") 
        print("   - Ambiente configurado incorretamente")
    elif episodes_with_trades < 3:
        print("⚠️ BAIXA ATIVIDADE: Poucos episódios ativos")
        print("   O modelo pode estar muito seletivo")
    else:
        print("✅ ATIVIDADE NORMAL: Modelo está operando")
    
    print(f"\\n🎯 COMPARAÇÃO COM PRODUÇÃO:")
    print(f"   Esperado: ~10-20 trades/dia")
    print(f"   Observado: {total_trades/5*288/500:.1f} trades/dia (estimativa)")

if __name__ == "__main__":
    test_activity_450k()