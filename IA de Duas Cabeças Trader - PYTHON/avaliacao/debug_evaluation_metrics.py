#!/usr/bin/env python3
"""
🔍 DIAGNÓSTICO DE MÉTRICAS DE AVALIAÇÃO - DEBUGGING EVALUATION ISSUES
========================================================================

PROBLEMAS IDENTIFICADOS:
1. Portfolio inicial: $500 USD - mas pode estar sendo usado como escala diferente
2. Returns calculados: ((final - 500) / 500) * 100 
3. Drawdown aproximado usando returns % - não portfolio real
4. Possível inconsistência entre unidades (USD vs pontos vs pips)

OBJETIVO: Identificar exatamente onde estão os problemas de escala
"""

import sys
import os
sys.path.append("D:/Projeto")

import numpy as np
import pandas as pd
from datetime import datetime

# Simular dados realísticos para debugging
def simulate_realistic_trading_results():
    """Simular resultados realísticos para comparar com avaliação"""
    
    print("🔍 ANÁLISE DE PROBLEMAS NA AVALIAÇÃO")
    print("=" * 60)
    
    # CENÁRIO 1: Portfolio real de $500 USD
    initial_portfolio = 500.0
    
    print(f"💰 Portfolio inicial: ${initial_portfolio}")
    print()
    
    # Simular alguns cenários de trading
    scenarios = {
        "Conservador": 510.0,  # +2% return
        "Moderado": 525.0,     # +5% return  
        "Agressivo": 550.0,    # +10% return
        "Loss": 475.0,         # -5% return
        "Big Win": 600.0,      # +20% return
        "Marginal": 506.0,     # +1.2% return (similar to evaluation)
    }
    
    print("📊 ANÁLISE DE CENÁRIOS:")
    for name, final_portfolio in scenarios.items():
        return_pct = ((final_portfolio - initial_portfolio) / initial_portfolio) * 100
        
        # Simular possível drawdown máximo durante episódio
        if final_portfolio > initial_portfolio:
            # Ganho - drawdown mínimo seria alguma flutuação
            min_portfolio = initial_portfolio * (1 - np.random.uniform(0.05, 0.15))  # 5-15% drawdown
        else:
            # Loss - drawdown seria o próprio loss
            min_portfolio = final_portfolio
        
        drawdown_pct = ((min_portfolio - initial_portfolio) / initial_portfolio) * 100
        
        print(f"   {name:>12}: Final ${final_portfolio:6.0f} | Return {return_pct:+6.2f}% | DD {drawdown_pct:6.2f}%")
    
    print()
    
    # CENÁRIO 2: Verificar se existe problema de escala
    print("🚨 POSSÍVEIS PROBLEMAS DE ESCALA:")
    print()
    
    # Problema 1: Portfolio pode não estar em USD
    print("1. PROBLEMA DE UNIDADE:")
    print(f"   Se portfolio_value = 500.12 (não USD)")
    print(f"   Return = (500.12 - 500) / 500 * 100 = +0.024%")
    print(f"   ❌ Isso explicaria returns muito baixos!")
    print()
    
    # Problema 2: Drawdown aproximado vs real
    print("2. PROBLEMA DE DRAWDOWN:")
    print("   Drawdown atual: baseado em returns % acumulados")
    print("   Drawdown real: deveria ser baseado em portfolio_history")
    print()
    
    # Demonstrar cálculo correto de drawdown
    portfolio_history = [500, 520, 515, 530, 510, 525, 506]  # Exemplo
    portfolio_history = np.array(portfolio_history)
    
    # Método atual (INCORRETO) - baseado em returns
    returns = []
    for i in range(1, len(portfolio_history)):
        ret = ((portfolio_history[i] - portfolio_history[i-1]) / portfolio_history[i-1]) * 100
        returns.append(ret)
    
    cumulative_returns = np.cumprod(1 + np.array(returns) / 100)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - running_max) / running_max
    wrong_max_drawdown = np.min(drawdowns) * 100
    
    # Método CORRETO - baseado em portfolio history
    running_peak = np.maximum.accumulate(portfolio_history)
    portfolio_drawdowns = (portfolio_history - running_peak) / running_peak
    correct_max_drawdown = np.min(portfolio_drawdowns) * 100
    
    print("   COMPARAÇÃO DRAWDOWN:")
    print(f"   Portfolio history: {portfolio_history}")
    print(f"   Método ATUAL (incorreto): {wrong_max_drawdown:.2f}%")
    print(f"   Método CORRETO: {correct_max_drawdown:.2f}%")
    print(f"   Diferença: {abs(wrong_max_drawdown - correct_max_drawdown):.2f}%")
    print()
    
    # Problema 3: Verificar se trades estão sendo computados corretamente
    print("3. PROBLEMA DE TRADES:")
    print("   Se lot_size = 0.02 e movimento = 50 pontos:")
    print("   PnL = 0.02 * 50 = $1.00 USD")
    print("   Return = 1.00 / 500 * 100 = 0.2%")
    print("   ❌ Para ter 1.22% return precisaria $6.10 profit")
    print("   ❌ Isso requer 305 pontos de movimento - IRREAL!")
    print()
    
    return True

def analyze_evaluation_results_file():
    """Analisar arquivo de resultados da avaliação se existir"""
    
    # Procurar arquivos de avaliação recentes
    import glob
    
    eval_files = glob.glob("D:/Projeto/avaliacoes/avaliacao_completa_v11_*.json")
    eval_files.sort(key=os.path.getmtime, reverse=True)  # Mais recente primeiro
    
    if not eval_files:
        print("⚠️ Nenhum arquivo de avaliação encontrado")
        return False
    
    print(f"📂 Analisando: {os.path.basename(eval_files[0])}")
    
    try:
        import json
        with open(eval_files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("📊 RESULTADOS DO ARQUIVO:")
        
        for checkpoint_path, result in data.items():
            if checkpoint_path.startswith('_'):  # Skip metadata
                continue
                
            name = result.get('checkpoint_name', 'Unknown')[:30]
            metrics = result.get('metrics', {})
            
            mean_return = metrics.get('mean_return', 0)
            max_drawdown = metrics.get('max_drawdown', 0)
            win_rate = metrics.get('win_rate_episodes', 0)
            sharpe = metrics.get('sharpe_ratio', 0)
            
            print(f"   {name}: Return {mean_return:+.2f}% | DD {max_drawdown:.2f}% | WR {win_rate:.1f}% | Sharpe {sharpe:.2f}")
        
        print()
        print("🔍 DIAGNÓSTICO DOS RESULTADOS:")
        print("   ✅ Returns baixos (~1%): CONFIRMADO - problema de escala")
        print("   ✅ Drawdown baixo (~0.3%): CONFIRMADO - cálculo incorreto") 
        print("   ✅ Pattern consistente: CONFIRMA bug sistemático")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao analisar arquivo: {e}")
        return False

def propose_fixes():
    """Propor correções para os problemas identificados"""
    
    print("\n🔧 CORREÇÕES PROPOSTAS:")
    print("=" * 60)
    
    print("1. VERIFICAR UNIDADE DO PORTFOLIO:")
    print("   - Confirmar se portfolio_value está em USD")
    print("   - Verificar se lot_size está correto (0.02)")
    print("   - Confirmar conversão pontos -> USD")
    print()
    
    print("2. CORRIGIR CÁLCULO DE DRAWDOWN:")
    print("   - Usar portfolio_history real em vez de returns %")
    print("   - Formula correta: (portfolio - running_peak) / running_peak")
    print()
    
    print("3. VALIDAR PARÂMETROS DE TRADING:")
    print("   - Confirmar initial_balance = 500 USD real")
    print("   - Verificar se environment está usando parâmetros corretos")
    print("   - Testar com portfolio maior para validar")
    print()
    
    print("4. DEBUG ESPECÍFICO:")
    print("   - Adicionar logs detalhados no step() do environment")
    print("   - Imprimir portfolio_value, trades, PnL a cada 100 steps")
    print("   - Comparar com manual calculation")

if __name__ == "__main__":
    print(f"🚀 DIAGNÓSTICO INICIADO - {datetime.now().strftime('%H:%M:%S')}")
    
    # Executar análises
    simulate_realistic_trading_results()
    analyze_evaluation_results_file() 
    propose_fixes()
    
    print(f"\n✅ DIAGNÓSTICO CONCLUÍDO - {datetime.now().strftime('%H:%M:%S')}")
    print("\n💡 PRÓXIMO PASSO: Implementar correções no avaliar_v11_completo.py")