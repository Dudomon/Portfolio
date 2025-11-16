#!/usr/bin/env python3
"""
🔍 ANÁLISE DA ESCALA DOS COMPONENTES
Script para entender por que divide por 5 e qual seria a escala real
"""

import sys
import os
import numpy as np
sys.path.append("D:/Projeto")

def analyze_component_scaling():
    """🔍 Analisar escala teórica vs real dos componentes"""
    
    print("🔍 ANÁLISE DA ESCALA DOS COMPONENTES DO REWARD SYSTEM")
    print("=" * 60)
    
    print("\n📊 ESCALA TEÓRICA MÁXIMA DE CADA COMPONENTE:")
    print("=" * 50)
    
    # Analisar base_weights para entender escala máxima
    base_weights = {
        # PnL BALANCEADO (40% do sistema)
        "pnl_direct": 1.0,              
        "win_bonus": 0.5,               
        "loss_penalty": -1.0,           
        
        # GESTÃO DE RISCO (30% do sistema)
        "risk_reward_bonus": 2.0,       
        "position_sizing_bonus": 1.5,   
        "max_loss_penalty": -3.0,       
        "drawdown_penalty": -2.0,       
        "risk_management_bonus": 1.0,   
        
        # CONSISTÊNCIA E PERFORMANCE (20% do sistema)
        "sharpe_ratio_bonus": 1.5,      
        "win_rate_bonus": 1.0,          
        "consistency_bonus": 0.8,       
        "streak_bonus": 0.6,            
        
        # TIMING E EXECUÇÃO (10% do sistema)
        "execution_bonus": 0.5,         
        "optimal_duration": 0.3,        
        "timing_bonus": 0.2,            
    }
    
    # 1. PnL Components
    print("1. 💰 PnL COMPONENTS:")
    pnl_max = 0
    print(f"   pnl_direct: ±{base_weights['pnl_direct']}")
    print(f"   win_bonus: +{base_weights['win_bonus']}")  
    print(f"   loss_penalty: {base_weights['loss_penalty']}")
    pnl_max = base_weights['pnl_direct'] + base_weights['win_bonus']  # Cenário mais positivo
    pnl_min = base_weights['pnl_direct'] + base_weights['loss_penalty']  # Cenário mais negativo
    print(f"   📊 RANGE PnL: [{pnl_min:.1f}, {pnl_max:.1f}]")
    
    # 2. Risk Management Components  
    print(f"\n2. 🛡️ RISK MANAGEMENT COMPONENTS:")
    risk_components = ["risk_reward_bonus", "position_sizing_bonus", "max_loss_penalty", 
                      "drawdown_penalty", "risk_management_bonus"]
    risk_max = sum(base_weights[k] for k in risk_components if base_weights[k] > 0)
    risk_min = sum(base_weights[k] for k in risk_components if base_weights[k] < 0)
    for comp in risk_components:
        print(f"   {comp}: {base_weights[comp]:+.1f}")
    print(f"   📊 RANGE Risk: [{risk_min:.1f}, {risk_max:.1f}]")
    
    # 3. Consistency Components
    print(f"\n3. 📊 CONSISTENCY COMPONENTS:")
    consistency_components = ["sharpe_ratio_bonus", "win_rate_bonus", "consistency_bonus", "streak_bonus"]
    consistency_max = sum(base_weights[k] for k in consistency_components)
    consistency_min = 0  # Todos são positivos ou zero
    for comp in consistency_components:
        print(f"   {comp}: +{base_weights[comp]}")
    print(f"   📊 RANGE Consistency: [0.0, {consistency_max:.1f}]")
    
    # 4. Timing Components
    print(f"\n4. ⚡ TIMING COMPONENTS:")
    timing_components = ["execution_bonus", "optimal_duration", "timing_bonus"]
    timing_max = sum(base_weights[k] for k in timing_components)
    timing_min = 0  # Todos são positivos ou zero
    for comp in timing_components:
        print(f"   {comp}: +{base_weights[comp]}")
    print(f"   📊 RANGE Timing: [0.0, {timing_max:.1f}]")
    
    # 5. Activity Enhancement (o culpado!)
    print(f"\n5. 🎯 ACTIVITY ENHANCEMENT:")
    print(f"   inactivity_penalty: -5.0 (máximo)")
    print(f"   trade_bonus: +0.5 (máximo)")
    print(f"   attempt_bonus: +0.02")
    activity_max = 0.5 + 0.02
    activity_min = -5.0
    print(f"   📊 RANGE Activity: [{activity_min:.1f}, {activity_max:.1f}]")
    
    # 6. Gaming Penalties
    print(f"\n6. 🛡️ ANTI-GAMING:")
    print(f"   micro_trades_penalty: -0.2")
    print(f"   uniform_trades_penalty: -0.1") 
    print(f"   overtrading_penalty: -0.001 por trade extra")
    print(f"   gaming_repetido_penalty: -0.3")
    gaming_min = -0.2 - 0.1 - 0.3  # Pior cenário
    gaming_max = 0
    print(f"   📊 RANGE Gaming: [{gaming_min:.1f}, {gaming_max:.1f}]")
    
    # CÁLCULO DA ESCALA TOTAL TEÓRICA
    print(f"\n🧮 ESCALA TOTAL TEÓRICA:")
    print("=" * 30)
    
    total_max = pnl_max + risk_max + consistency_max + timing_max + activity_max + gaming_max
    total_min = pnl_min + risk_min + consistency_min + timing_min + activity_min + gaming_min
    
    print(f"   Componente         | Min    | Max")
    print(f"   -------------------|--------|--------")
    print(f"   PnL                | {pnl_min:+6.1f} | {pnl_max:+6.1f}")
    print(f"   Risk Management    | {risk_min:+6.1f} | {risk_max:+6.1f}")
    print(f"   Consistency        | {consistency_min:+6.1f} | {consistency_max:+6.1f}")
    print(f"   Timing             | {timing_min:+6.1f} | {timing_max:+6.1f}")
    print(f"   Activity           | {activity_min:+6.1f} | {activity_max:+6.1f}")
    print(f"   Gaming             | {gaming_min:+6.1f} | {gaming_max:+6.1f}")
    print(f"   -------------------|--------|--------")
    print(f"   TOTAL TEÓRICO      | {total_min:+6.1f} | {total_max:+6.1f}")
    
    # COMPARAR COM DIVISÃO POR 5
    print(f"\n🔢 ANÁLISE DA DIVISÃO POR 5:")
    print("=" * 30)
    print(f"   Range teórico: [{total_min:.1f}, {total_max:.1f}]")
    print(f"   Amplitude: {total_max - total_min:.1f}")
    print(f"   Divisão por 5 resulta em: [{total_min/5:.2f}, {total_max/5:.2f}]")
    print(f"   Amplitude normalizada: {(total_max - total_min)/5:.2f}")
    
    # EXPLICAR PROBLEMA
    print(f"\n🚨 ANÁLISE DO PROBLEMA:")
    print("=" * 25)
    print(f"   1. Escala teórica máxima: ~{total_max:.1f}")
    print(f"   2. Escala teórica mínima: ~{total_min:.1f}")
    print(f"   3. Activity Enhancement sozinho: {activity_min:.1f} a {activity_max:.1f}")
    print(f"   4. Activity representa {abs(activity_min)/abs(total_min)*100:.1f}% do problema!")
    
    # SIMULAÇÃO DE CENÁRIOS REAIS
    print(f"\n📋 SIMULAÇÃO DE CENÁRIOS REAIS:")
    print("=" * 35)
    
    scenarios = {
        "HOLD Simples (sem trades)": {
            "pnl": 0, "risk": 0, "consistency": 0, "timing": 0, 
            "activity": 0, "gaming": 0
        },
        "HOLD com 20 steps consecutivos": {
            "pnl": 0, "risk": 0, "consistency": 0, "timing": 0,
            "activity": -(20-15)*0.1, "gaming": 0  # Activity penalty!
        },
        "Trade lucrativo simples": {
            "pnl": 0.002, "risk": 0.1, "consistency": 0, "timing": 0.1,
            "activity": 0.02, "gaming": 0
        },
        "Overtrading (50 trades)": {
            "pnl": 0.01, "risk": 0.2, "consistency": 0.5, "timing": 0.3,
            "activity": 0.5, "gaming": -0.6  # Gaming penalty!
        }
    }
    
    for scenario_name, values in scenarios.items():
        total = sum(values.values())
        normalized = total / 5.0
        print(f"\n   {scenario_name}:")
        print(f"      Raw total: {total:+.3f}")
        print(f"      Normalizado (/5): {normalized:+.3f}")
        print(f"      Componentes: {values}")
    
    # RECOMENDAÇÃO
    print(f"\n💡 RECOMENDAÇÕES:")
    print("=" * 20)
    print("   1. 🎯 Activity Enhancement é o MAIOR problema (-5.0 sozinho!)")
    print("   2. 🔢 Divisão por 5 faz sentido TEORICAMENTE")  
    print("   3. 🚨 Mas na prática Activity domina o signal")
    print("   4. ✅ Desabilitar ou reduzir drasticamente Activity")
    print("   5. 🔄 Talvez dividir por 2-3 seja mais apropriado na prática")

if __name__ == "__main__":
    analyze_component_scaling()