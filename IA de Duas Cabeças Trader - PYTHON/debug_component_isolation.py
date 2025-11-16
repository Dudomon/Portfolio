#!/usr/bin/env python3
"""
🔍 DEBUG COMPONENT ISOLATION
Script para testar cada componente do reward system individualmente
"""

import sys
import os
import numpy as np
import pandas as pd
sys.path.append("D:/Projeto")

from daytrader import TradingEnv

def test_component_isolation():
    """🔍 Testar cada componente individualmente"""
    
    print("🔍 TESTE DE ISOLAMENTO DE COMPONENTES - REWARD SYSTEM")
    print("=" * 70)
    
    # Carregar dataset pequeno
    dataset_path = "D:/Projeto/data/GC_YAHOO_ENHANCED_V3_BALANCED_20250804_192226.csv"
    df = pd.read_csv(dataset_path)
    
    # Processar dataset
    if 'time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['time'])
        df.set_index('timestamp', inplace=True)
        df.drop('time', axis=1, inplace=True)
    
    df = df.rename(columns={
        'open': 'open_5m',
        'high': 'high_5m', 
        'low': 'low_5m',
        'close': 'close_5m',
        'tick_volume': 'volume_5m'
    })
    
    # Usar apenas 50 barras para teste rápido
    test_df = df.head(50).copy()
    
    # Criar ambiente
    trading_params = {
        'base_lot_size': 0.02,
        'max_lot_size': 0.03,
        'initial_balance': 500.0,
        'target_trades_per_day': 18,
        'stop_loss_range': (2.0, 8.0),
        'take_profit_range': (3.0, 15.0)
    }
    
    env = TradingEnv(
        test_df,
        window_size=20,
        is_training=False,
        initial_balance=500.0,
        trading_params=trading_params
    )
    
    reward_system = env.reward_system
    print(f"✅ Sistema preparado para testes")
    print(f"🔍 Fase atual: {reward_system.current_phase}")
    
    # Reset
    obs = env.reset()
    
    # Fazer algumas ações HOLD para simular cenário normal
    action_hold = np.array([0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    old_state = {
        "portfolio_total_value": env.portfolio_value,
        "current_drawdown": env.current_drawdown,
        "trades_count": len(env.trades)
    }
    
    print(f"\n🎮 TESTE BASELINE - AÇÃO HOLD SEM TRADES")
    print("=" * 50)
    
    # Executar step simples
    obs, reward_full, done, info = env.step(action_hold)
    
    print(f"   Portfolio: ${env.portfolio_value:.2f}")
    print(f"   Trades: {len(env.trades)}")
    print(f"   Reward TOTAL: {reward_full:.6f}")
    
    print(f"\n🔬 AGORA TESTANDO CADA COMPONENTE INDIVIDUAL...")
    print("=" * 60)
    
    # 1. TESTAR ANTI-GAMING ISOLADO
    print(f"\n1. 🛡️ TESTANDO ANTI-GAMING SYSTEM:")
    gaming_penalty = reward_system._calculate_gaming_penalties(env)
    print(f"   Gaming Penalty: {gaming_penalty:.6f}")
    print(f"   Gaming Detection: {reward_system.gaming_detection}")
    
    # 2. TESTAR CONSISTENCY ISOLADO  
    print(f"\n2. 📊 TESTANDO CONSISTENCY SYSTEM:")
    consistency_reward = reward_system._calculate_consistency_rewards(env)
    print(f"   Consistency Reward: {consistency_reward:.6f}")
    
    # 3. TESTAR PERFORMANCE CORRELATION ISOLADO
    print(f"\n3. 📈 TESTANDO PERFORMANCE CORRELATION:")
    performance_bonus = reward_system._calculate_performance_correlation_bonus(env)
    print(f"   Performance Bonus: {performance_bonus:.6f}")
    
    # 4. TESTAR ACTIVITY ENHANCEMENT ISOLADO
    print(f"\n4. 🎯 TESTANDO ACTIVITY ENHANCEMENT:")
    activity_reward = reward_system._calculate_activity_enhancement_reward(env, action_hold)
    print(f"   Activity Reward: {activity_reward:.6f}")
    print(f"   Consecutive Holds: {reward_system.consecutive_holds}")
    
    # 5. TESTAR CONTINUOUS FEEDBACK ISOLADO
    print(f"\n5. 🔄 TESTANDO CONTINUOUS FEEDBACK:")
    continuous_components = reward_system._calculate_continuous_feedback_components(env)
    continuous_total = sum(continuous_components.values())
    print(f"   Continuous Components: {continuous_components}")
    print(f"   Continuous Total: {continuous_total:.6f}")
    
    # 6. SIMULAÇÃO CUMULATIVA - ADICIONAR UM POR VEZ
    print(f"\n🧮 SIMULAÇÃO CUMULATIVA - ADICIONANDO COMPONENTE POR VEZ:")
    print("=" * 60)
    
    base_reward = continuous_total
    print(f"   BASE (continuous): {base_reward:.6f}")
    
    # Adicionar Gaming
    cumulative = base_reward + gaming_penalty
    print(f"   + Gaming: {base_reward:.6f} + {gaming_penalty:.6f} = {cumulative:.6f}")
    
    # Adicionar Consistency  
    cumulative += consistency_reward
    print(f"   + Consistency: {cumulative - consistency_reward:.6f} + {consistency_reward:.6f} = {cumulative:.6f}")
    
    # Adicionar Performance
    cumulative += performance_bonus
    print(f"   + Performance: {cumulative - performance_bonus:.6f} + {performance_bonus:.6f} = {cumulative:.6f}")
    
    # Adicionar Activity
    cumulative += activity_reward
    print(f"   + Activity: {cumulative - activity_reward:.6f} + {activity_reward:.6f} = {cumulative:.6f}")
    
    print(f"\n📊 COMPARAÇÃO FINAL:")
    print(f"   Raw Reward Calculado: {cumulative:.6f}")
    print(f"   Raw Reward Real: {reward_full * 5.0:.6f} (reward * 5)")  # Desfazer divisão por 5
    print(f"   Reward Final: {reward_full:.6f}")
    print(f"   Após Normalização (/5): {cumulative / 5.0:.6f}")
    
    # 7. IDENTIFICAR O CULPADO
    print(f"\n🚨 IDENTIFICAÇÃO DO CULPADO:")
    print("=" * 40)
    
    components = {
        'Gaming': gaming_penalty,
        'Consistency': consistency_reward, 
        'Performance': performance_bonus,
        'Activity': activity_reward,
        'Continuous': continuous_total
    }
    
    # Ordenar por valor (mais negativo primeiro)
    sorted_components = sorted(components.items(), key=lambda x: x[1])
    
    print("   Componentes ordenados (mais negativo primeiro):")
    for name, value in sorted_components:
        if value < -0.01:
            print(f"   🔴 {name}: {value:.6f} (PROBLEMA!)")
        elif value < 0:
            print(f"   🟡 {name}: {value:.6f} (suspeito)")
        elif value > 0.01:
            print(f"   🟢 {name}: {value:.6f} (positivo)")
        else:
            print(f"   ⚪ {name}: {value:.6f} (neutro)")
    
    # TESTE ADICIONAL: Simular múltiplos HOLDs para ver activity punishment
    print(f"\n🎯 TESTE ESPECIAL: MÚLTIPLOS HOLDS CONSECUTIVOS")
    print("=" * 50)
    
    for i in range(20):
        obs, reward, done, info = env.step(action_hold)
        activity_now = reward_system._calculate_activity_enhancement_reward(env, action_hold)
        
        if i % 5 == 0:  # Log a cada 5 steps
            print(f"   Step {i+1}: Consecutive HOLDs: {reward_system.consecutive_holds}, Activity: {activity_now:.6f}")
    
    return True

if __name__ == "__main__":
    test_component_isolation()