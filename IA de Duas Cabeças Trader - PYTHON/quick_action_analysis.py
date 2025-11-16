#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 ANÁLISE RÁPIDA DE AÇÕES - DIRETO AO PONTO
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import glob
import os
from collections import Counter

print("🔍 ANÁLISE RÁPIDA - DETECTIVE DE ACTIONS")
print("=" * 60)

def analyze_recent_logs():
    """Analisar logs recentes para encontrar padrões de ação"""
    
    # Buscar arquivos de log
    log_files = []
    patterns = [
        "*.txt",
        "debug_*.txt", 
        "training_*.log",
        "*.log"
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern)
        log_files.extend(files)
    
    print(f"📁 Encontrados {len(log_files)} arquivos de log")
    
    # Analisar conteúdo dos logs
    action_mentions = []
    trading_mentions = []
    
    for file in log_files[:10]:  # Apenas primeiros 10
        try:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Procurar menções de ações
                if 'action' in content.lower():
                    lines = content.split('\n')
                    for line in lines:
                        if 'action' in line.lower() and ('hold' in line.lower() or 'long' in line.lower() or 'short' in line.lower()):
                            action_mentions.append(line.strip())
                
                # Procurar atividade de trading
                if 'trade' in content.lower():
                    lines = content.split('\n')
                    for line in lines:
                        if ('trades/dia' in line.lower() or 'trading' in line.lower()) and any(c.isdigit() for c in line):
                            trading_mentions.append(line.strip())
                            
        except Exception as e:
            continue
    
    print(f"\n📊 MENÇÕES DE AÇÕES ENCONTRADAS ({len(action_mentions)}):")
    for mention in action_mentions[-5:]:  # Últimas 5
        print(f"  {mention}")
    
    print(f"\n📈 MENÇÕES DE TRADING ENCONTRADAS ({len(trading_mentions)}):")
    for mention in trading_mentions[-5:]:  # Últimas 5
        print(f"  {mention}")

def check_model_action_distribution():
    """Verificar distribuição de ações sem carregar modelo completo"""
    
    print(f"\n🎯 SIMULAÇÃO DE DISTRIBUIÇÃO DE AÇÕES:")
    
    # Simular diferentes cenários de confiança
    scenarios = [
        ("Modelo Conservador", [0.85, 0.10, 0.05]),  # 85% hold, 10% long, 5% short
        ("Modelo Balanceado", [0.60, 0.25, 0.15]),   # 60% hold, 25% long, 15% short  
        ("Modelo Agressivo", [0.30, 0.40, 0.30]),    # 30% hold, 40% long, 30% short
        ("Modelo com Hold Bias", [0.95, 0.03, 0.02]) # 95% hold - PROBLEMA!
    ]
    
    for name, distribution in scenarios:
        hold_pct, long_pct, short_pct = distribution
        trades_per_day = (long_pct + short_pct) * 288  # 288 steps por dia (5min)
        
        print(f"\n  {name}:")
        print(f"    Hold: {hold_pct*100:.1f}% | Long: {long_pct*100:.1f}% | Short: {short_pct*100:.1f}%")
        print(f"    Trades/Dia esperados: {trades_per_day:.1f}")
        
        if trades_per_day < 10:
            print(f"    🔴 MUITO BAIXA ATIVIDADE!")
        elif trades_per_day > 50:
            print(f"    🔴 ATIVIDADE EXCESSIVA!")
        else:
            print(f"    ✅ Atividade OK")

def analyze_action_space():
    """Analisar espaço de ações teoricamente"""
    
    print(f"\n🏗️ ANÁLISE DO ACTION SPACE:")
    
    # Action space esperado: [11 dimensões]
    action_info = [
        "[0] entry_decision: 0=hold, 1=long, 2=short",
        "[1] entry_confidence: [0,1] Confiança da entrada", 
        "[2] temporal_signal: [-1,1] Sinal temporal",
        "[3] risk_appetite: [0,1] Apetite ao risco",
        "[4] market_regime_bias: [-1,1] Viés do regime",
        "[5-10] sl/tp adjustments: [-3,3] Ajustes SL/TP"
    ]
    
    for info in action_info:
        print(f"  {info}")
    
    print(f"\n🎯 PROBLEMAS POTENCIAIS:")
    problems = [
        "1. Action[0] pode estar sempre outputando ~0 (hold bias)",
        "2. Confidence muito baixa impedindo trades mesmo com entry_decision > 0",
        "3. Gradientes zerados fazendo action head 'travar' em hold",
        "4. Softmax/Sigmoid mal configurado gerando sempre mesmo output",
        "5. Learning rate muito baixo não atualizando action weights"
    ]
    
    for problem in problems:
        print(f"  {problem}")

def suggest_debug_steps():
    """Sugerir próximos passos de debug"""
    
    print(f"\n💡 PRÓXIMOS PASSOS PARA DEBUG:")
    
    steps = [
        "1. Adicionar print(action) no daytrader.py step() para ver outputs reais",
        "2. Verificar se action[0] está sempre entre -0.1 e 0.1 (hold bias)",
        "3. Testar modelo com actions manuais: [1, 0.8, ...] para forçar trades",
        "4. Verificar gradientes da action_net no modelo",
        "5. Comparar pesos da action_net entre checkpoints antigos/novos",
        "6. Adicionar logging detalhado na TwoHeadV7Intuition",
        "7. Verificar se reward system está penalizando trades demais"
    ]
    
    for step in steps:
        print(f"  {step}")
    
    print(f"\n🔧 COMANDOS RÁPIDOS PARA TESTAR:")
    commands = [
        "# No daytrader.py, linha ~4715, adicionar:",
        "print(f'Action raw: {action}')",
        "print(f'Entry decision: {int(action[0])}')",
        "print(f'Entry confidence: {float(action[1]):.3f}')",
        "",
        "# Para forçar um trade manual:",
        "action = [1.0, 0.8, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
    ]
    
    for cmd in commands:
        print(f"  {cmd}")

def main():
    """Função principal - análise rápida"""
    
    # 1. Analisar logs existentes
    analyze_recent_logs()
    
    # 2. Verificar distribuições teóricas
    check_model_action_distribution()
    
    # 3. Analisar action space
    analyze_action_space()
    
    # 4. Sugerir debug steps
    suggest_debug_steps()
    
    print(f"\n" + "🎯" * 60)
    print("🎯 RESUMO EXECUTIVO")
    print("🎯" * 60)
    print("🔴 SUSPEITA PRINCIPAL: Hold Bias no Action Head")
    print("🔴 Model está outputando action[0] ≈ 0 sempre")
    print("🔴 Mesmo com composite_score > 60%, entry_decision = 0")
    print("")
    print("🚀 SOLUÇÃO RÁPIDA: Adicionar logging detalhado das actions")
    print("💡 Verificar se action[0] varia ou está 'travado' em 0")
    print("✅ Com isso descobriremos se é problema de:")
    print("   - Arquitetura neural (hold bias)")
    print("   - Gradientes (zeros extremos)")  
    print("   - Configuração (thresholds)")
    print("   - Reward system (penalização excessiva)")

if __name__ == "__main__":
    main()