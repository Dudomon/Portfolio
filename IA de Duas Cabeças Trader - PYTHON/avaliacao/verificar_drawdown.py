#!/usr/bin/env python3
"""
Verificar Drawdown nos Dados de Treinamento
==========================================
Analisar se o drawdown realmente atinge 50%
"""

import pandas as pd
import numpy as np

def analyze_drawdown():
    """Analisar drawdown nos dados de treinamento"""
    
    print("="*80)
    print("🔍 ANÁLISE DE DRAWDOWN - DADOS DE TREINAMENTO")
    print("="*80)
    
    # Carregar dados
    csv_path = "D:/Projeto/Otimizacao/treino_principal/models/SILUS/SILUS_training_metrics_20250827_220321.csv"
    print(f"\n📂 Carregando: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"   Dados carregados: {len(df)} linhas")
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return
    
    # Verificar colunas
    print(f"\n📋 Colunas disponíveis:")
    for col in df.columns:
        print(f"   - {col}")
    
    # Analisar portfolio_value e drawdown
    if 'portfolio_value' in df.columns:
        portfolio_values = df['portfolio_value']
        
        print(f"\n📊 ANÁLISE DO PORTFOLIO:")
        print(f"   Portfolio inicial: 500.00")
        print(f"   Portfolio min: {portfolio_values.min():.2f}")
        print(f"   Portfolio max: {portfolio_values.max():.2f}")
        print(f"   Portfolio médio: {portfolio_values.mean():.2f}")
        
        # Calcular drawdown manual
        initial_balance = 500.0
        running_drawdown = []
        peak_value = initial_balance
        
        for value in portfolio_values:
            if value > peak_value:
                peak_value = value
            
            # Drawdown em percentual
            drawdown_pct = (peak_value - value) / peak_value if peak_value > 0 else 0
            running_drawdown.append(drawdown_pct)
        
        running_drawdown = np.array(running_drawdown)
        
        print(f"\n📈 DRAWDOWN CALCULADO:")
        print(f"   Drawdown min: {running_drawdown.min():.4f} (0%)")
        print(f"   Drawdown max: {running_drawdown.max():.4f} ({running_drawdown.max()*100:.2f}%)")
        print(f"   Drawdown médio: {running_drawdown.mean():.4f} ({running_drawdown.mean()*100:.2f}%)")
        
        # Verificar se algum drawdown passou de 50%
        extreme_drawdowns = running_drawdown > 0.5
        count_extreme = extreme_drawdowns.sum()
        
        print(f"\n🚨 DRAWDOWNS > 50%:")
        print(f"   Ocorrências: {count_extreme} ({count_extreme/len(running_drawdown)*100:.2f}%)")
        
        if count_extreme > 0:
            print(f"   ✅ CONFIRMADO: Drawdown >50% está causando episódios prematuros!")
            
            # Mostrar alguns casos
            extreme_indices = np.where(extreme_drawdowns)[0][:10]
            print(f"\n   Primeiros 10 casos de drawdown >50%:")
            for i in extreme_indices:
                step = df.iloc[i]['step']
                portfolio = portfolio_values.iloc[i]
                dd = running_drawdown[i]
                print(f"     Step {step}: Portfolio ${portfolio:.2f}, Drawdown {dd*100:.1f}%")
        else:
            print(f"   ❌ NENHUM drawdown >50% encontrado!")
            print(f"   O problema NÃO é early termination por drawdown")
    
    # Verificar coluna drawdown se existir
    if 'drawdown' in df.columns:
        drawdown_col = df['drawdown']
        
        print(f"\n📊 COLUNA DRAWDOWN ORIGINAL:")
        print(f"   Drawdown min: {drawdown_col.min():.4f}")
        print(f"   Drawdown max: {drawdown_col.max():.4f}")
        print(f"   Drawdown médio: {drawdown_col.mean():.4f}")
        
        # Verificar se drawdown original >50%
        extreme_original = drawdown_col > 0.5
        count_original = extreme_original.sum()
        print(f"   Drawdowns >50% (original): {count_original}")
        
        # Comparar com calculado
        if len(running_drawdown) == len(drawdown_col):
            correlation = np.corrcoef(running_drawdown, drawdown_col)[0, 1]
            print(f"   Correlação calculado vs original: {correlation:.4f}")
    
    # Analisar resets de portfolio
    portfolio_resets = (portfolio_values == 500.0).sum()
    print(f"\n🔄 RESETS DE PORTFOLIO:")
    print(f"   Portfolio = 500.0: {portfolio_resets} vezes")
    print(f"   Frequência: a cada {len(df)/portfolio_resets:.1f} steps")
    
    # Verificar se há padrão nos resets
    reset_indices = np.where(portfolio_values == 500.0)[0]
    if len(reset_indices) > 1:
        reset_intervals = np.diff(reset_indices)
        print(f"   Intervalo médio entre resets: {reset_intervals.mean():.1f} steps")
        print(f"   Intervalo min: {reset_intervals.min()} steps")
        print(f"   Intervalo max: {reset_intervals.max()} steps")
        
        # Mostrar primeiros resets
        print(f"\n   Primeiros 10 resets:")
        for i, idx in enumerate(reset_indices[:10]):
            step = df.iloc[idx]['step']
            print(f"     Reset {i+1}: Step {step}")
    
    return {
        'max_drawdown': running_drawdown.max() if 'portfolio_value' in df.columns else 0,
        'extreme_drawdowns': count_extreme if 'portfolio_value' in df.columns else 0,
        'portfolio_resets': portfolio_resets if 'portfolio_value' in df.columns else 0
    }

if __name__ == "__main__":
    results = analyze_drawdown()
    
    print(f"\n" + "="*80)
    print("💡 CONCLUSÃO")
    print("="*80)
    
    if results['extreme_drawdowns'] > 0:
        print(f"""
❌ PROBLEMA CONFIRMADO: DRAWDOWN >50% CAUSANDO EPISÓDIOS PREMATUROS

- Drawdown máximo: {results['max_drawdown']*100:.1f}%
- Ocorrências >50%: {results['extreme_drawdowns']}
- Portfolio resets: {results['portfolio_resets']}

SOLUÇÃO: Ajustar limite de early termination no reward system V4 INNO
Trocar de 50% para 70-80% ou desabilitar completamente.
""")
    else:
        print(f"""
✅ DRAWDOWN NÃO É O PROBLEMA

- Drawdown máximo: {results['max_drawdown']*100:.1f}%
- Nenhum caso >50% encontrado
- Portfolio resets: {results['portfolio_resets']} (outro motivo)

INVESTIGAR OUTRAS CAUSAS:
1. Condições de done no silus.py
2. current_step >= len(df) - 1
3. episode_steps >= MAX_STEPS
4. Alguma outra condição oculta
""")