#!/usr/bin/env python3
"""
🔍 ANÁLISE PROFUNDA DA PERFORMANCE SILUS - DIAGNÓSTICO COMPLETO
================================================================

Sharpe 0.23 é MUITO baixo para trading real. Vamos entender o porquê.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
import glob
import os

def analyze_evaluation_results():
    """Analisar resultados de avaliação detalhadamente"""
    
    print("="*80)
    print("📊 ANÁLISE DOS RESULTADOS DE AVALIAÇÃO")
    print("="*80)
    
    # Carregar último resultado de avaliação
    eval_files = glob.glob("D:/Projeto/avaliacoes/avaliacao_completa_v11_*.json")
    eval_files.sort(key=os.path.getmtime, reverse=True)
    
    if not eval_files:
        print("❌ Nenhum arquivo de avaliação encontrado")
        return
    
    latest_file = eval_files[0]
    print(f"\n📂 Analisando: {os.path.basename(latest_file)}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Analisar cada checkpoint
    all_results = []
    
    for checkpoint_path, result in data.items():
        if checkpoint_path.startswith('_'):
            continue
        
        metrics = result.get('metrics', {})
        name = result.get('checkpoint_name', 'Unknown')
        
        # Extrair steps do nome
        steps_str = [s for s in name.split('_') if 'steps' in s]
        if steps_str:
            try:
                steps_num = steps_str[0].replace('steps', '').replace('k', '000')
                if steps_num:
                    steps = int(steps_num)
                else:
                    steps = 0
            except:
                steps = 0
        else:
            steps = 0
        
        result_data = {
            'steps': steps,
            'name': name[:40],
            'mean_return': metrics.get('mean_return', 0),
            'std_return': metrics.get('std_return', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'win_rate': metrics.get('win_rate_episodes', 0),
            'total_trades': metrics.get('total_trades', 0),
            'profit_factor': metrics.get('profit_factor', 0),
            'calmar_ratio': metrics.get('calmar_ratio_REAL', 0),
        }
        
        all_results.append(result_data)
    
    # Criar DataFrame e ordenar
    df = pd.DataFrame(all_results)
    df = df[df['steps'] > 0].sort_values('steps')
    
    print(f"\n📊 RESUMO DA PERFORMANCE POR CHECKPOINT:")
    print("-"*80)
    
    # Mostrar pontos-chave
    key_checkpoints = df[df['steps'].isin([1500000, 2450000, 2550000, 2600000, 2750000, 
                                           3250000, 3850000, 3900000, 3950000, 4000000, 
                                           4050000, 4100000, 4250000, 4500000])]
    
    for _, row in key_checkpoints.iterrows():
        steps_m = row['steps'] / 1e6
        print(f"\n🔸 {steps_m:.2f}M steps:")
        print(f"   Sharpe: {row['sharpe_ratio']:.3f}")
        print(f"   Return: {row['mean_return']:+.2f}% (±{row['std_return']:.2f}%)")
        print(f"   Drawdown: {row['max_drawdown']:.2f}%")
        print(f"   Win Rate: {row['win_rate']:.1f}%")
        print(f"   Trades: {row['total_trades']:.0f}")
        print(f"   Profit Factor: {row['profit_factor']:.2f}")
    
    return df

def diagnose_problems(df):
    """Diagnosticar problemas específicos"""
    
    print("\n" + "="*80)
    print("🔍 DIAGNÓSTICO DE PROBLEMAS")
    print("="*80)
    
    # 1. Análise do Sharpe
    best_sharpe = df['sharpe_ratio'].max()
    best_checkpoint = df.loc[df['sharpe_ratio'].idxmax(), 'steps'] / 1e6
    
    print(f"\n1️⃣ PROBLEMA DO SHARPE RATIO:")
    print(f"   Melhor Sharpe: {best_sharpe:.3f} ({best_checkpoint:.2f}M steps)")
    print(f"   ❌ MUITO BAIXO para trading real (deveria ser > 1.0)")
    
    # Decomposição do Sharpe
    best_row = df.loc[df['sharpe_ratio'].idxmax()]
    mean_ret = best_row['mean_return']
    std_ret = best_row['std_return']
    
    print(f"\n   Decomposição:")
    print(f"   • Return médio: {mean_ret:.2f}%")
    print(f"   • Desvio padrão: {std_ret:.2f}%")
    print(f"   • Sharpe = {mean_ret}/{std_ret} = {mean_ret/std_ret if std_ret > 0 else 0:.3f}")
    
    if std_ret > 5:
        print(f"   ⚠️ VOLATILIDADE MUITO ALTA ({std_ret:.1f}%)")
    if mean_ret < 2:
        print(f"   ⚠️ RETURN MUITO BAIXO ({mean_ret:.1f}%)")
    
    # 2. Análise de Win Rate
    print(f"\n2️⃣ ANÁLISE DE WIN RATE:")
    avg_wr = df['win_rate'].mean()
    print(f"   Win Rate médio: {avg_wr:.1f}%")
    
    if avg_wr < 50:
        print(f"   ⚠️ Win rate abaixo de 50% - sistema perdedor na maioria")
    
    # 3. Análise de Overtrading
    print(f"\n3️⃣ ANÁLISE DE OVERTRADING:")
    avg_trades = df['total_trades'].mean()
    print(f"   Média de trades: {avg_trades:.1f}")
    
    if avg_trades > 30:
        print(f"   ⚠️ Possível overtrading - muitos trades para período curto")
    
    # 4. Profit Factor
    print(f"\n4️⃣ ANÁLISE DE PROFIT FACTOR:")
    avg_pf = df['profit_factor'].mean()
    best_pf = df['profit_factor'].max()
    print(f"   Profit Factor médio: {avg_pf:.2f}")
    print(f"   Melhor Profit Factor: {best_pf:.2f}")
    
    if avg_pf < 1.5:
        print(f"   ⚠️ Profit Factor baixo - sistema pouco lucrativo")

def analyze_root_causes():
    """Análise das causas raízes"""
    
    print("\n" + "="*80)
    print("🔬 ANÁLISE DE CAUSAS RAÍZES")
    print("="*80)
    
    causes = """
    
🔴 CAUSA 1: REWARD SYSTEM INADEQUADO
    
    • V4 INNO com amplificação 4x está criando sinais muito fortes
    • Foco em PnL de curto prazo ao invés de risk-adjusted returns
    • Não considera Sharpe Ratio diretamente no reward
    • Activity bonus incentiva overtrading
    
🔴 CAUSA 2: FILTRO DE CONFIANÇA MUITO BAIXO (0.6)
    
    • Permite muitos trades de baixa qualidade
    • Aumenta volatilidade sem aumentar returns proporcionalmente
    • Degrada o Sharpe Ratio significativamente
    
🔴 CAUSA 3: GESTÃO DE RISCO INADEQUADA
    
    • Stop Loss fixo (2-8 pontos) não se adapta à volatilidade
    • Position sizing fixo (0.02 lot) não considera risco
    • Sem trailing stop ou gestão dinâmica
    
🔴 CAUSA 4: AMBIENTE DE TREINAMENTO IRREAL
    
    • Sem custos de transação (spread/comissão)
    • Sem slippage
    • Sem impacto de mercado
    • Execução instantânea perfeita
    
🔴 CAUSA 5: MÉTRICA DE OTIMIZAÇÃO ERRADA
    
    • Treinando para maximizar reward total
    • Deveria otimizar para Sharpe Ratio ou Calmar Ratio
    • Não penaliza volatilidade adequadamente
    """
    
    print(causes)

def propose_concrete_solutions():
    """Soluções concretas e implementáveis"""
    
    print("\n" + "="*80)
    print("💡 SOLUÇÕES CONCRETAS PARA MELHORAR PERFORMANCE")
    print("="*80)
    
    solutions = """
    
✅ SOLUÇÃO 1: NOVO REWARD SYSTEM (PRIORIDADE MÁXIMA)
    
    Implementar Sharpe-based reward:
    ```python
    def calculate_sharpe_reward(returns_window):
        if len(returns_window) < 20:
            return 0
        mean_return = np.mean(returns_window)
        std_return = np.std(returns_window)
        if std_return < 0.001:
            std_return = 0.001
        sharpe = mean_return / std_return
        return np.tanh(sharpe * 0.5)  # Normalizar entre -1 e 1
    ```
    
✅ SOLUÇÃO 2: AJUSTAR FILTROS E PARÂMETROS
    
    • Aumentar MIN_CONFIDENCE_THRESHOLD para 0.75
    • Reduzir reward amplification de 4x para 2x
    • Aumentar cooldown entre trades de 7 para 15 steps
    • Limitar max trades por episódio para 10
    
✅ SOLUÇÃO 3: POSITION SIZING DINÂMICO
    
    ```python
    def calculate_position_size(confidence, volatility, portfolio):
        base_size = 0.02
        confidence_mult = confidence  # 0.75 a 1.0
        vol_mult = 1.0 / (1 + volatility * 10)  # Reduz em alta vol
        kelly_fraction = 0.25  # Conservative Kelly
        
        position_size = base_size * confidence_mult * vol_mult * kelly_fraction
        return np.clip(position_size, 0.01, 0.03)
    ```
    
✅ SOLUÇÃO 4: CUSTOS REALISTAS
    
    • Adicionar spread: 0.5 pontos
    • Comissão: $2 por trade
    • Slippage: 0.1-0.3 pontos baseado em volatilidade
    • Delay de execução: 1-2 steps
    
✅ SOLUÇÃO 5: RETREINAR COM NOVO OBJETIVO
    
    ```python
    training_params = {
        'learning_rate': 5e-5,  # Reduzir
        'ent_coef': 0.05,  # Reduzir exploração
        'clip_range': 0.1,  # Mais conservador
        'n_epochs': 15,  # Mais epochs
        'batch_size': 256,  # Maior batch
        'gae_lambda': 0.90,  # Menos bias
        'target_kl': 0.02,  # Early stopping
    }
    ```
    
✅ SOLUÇÃO 6: VALIDAÇÃO RIGOROSA
    
    • Walk-forward optimization com janelas de 3 meses
    • Out-of-sample test em XAUUSD, EURUSD, SPX
    • Monte Carlo com 100 simulações
    • Stress test em crashes (2020, 2008)
    """
    
    print(solutions)
    
    print("\n" + "="*80)
    print("🎯 PLANO DE AÇÃO IMEDIATO")
    print("="*80)
    
    action_plan = """
    
📅 HOJE:
    1. Alterar MIN_CONFIDENCE_THRESHOLD para 0.75 no silus.py
    2. Reduzir reward amplification para 2x
    3. Aumentar cooldown para 15 steps
    
📅 AMANHÃ:
    4. Implementar Sharpe-based reward
    5. Adicionar custos de transação
    6. Implementar position sizing dinâmico
    
📅 ESTA SEMANA:
    7. Retreinar modelo com novos parâmetros
    8. Validar em out-of-sample data
    9. Comparar com benchmark Buy&Hold
    
🎯 META: Alcançar Sharpe > 1.0 em 7 dias
    """
    
    print(action_plan)

def calculate_realistic_expectations():
    """Calcular expectativas realistas"""
    
    print("\n" + "="*80)
    print("📊 EXPECTATIVAS REALISTAS APÓS MELHORIAS")
    print("="*80)
    
    expectations = """
    
COM AS MELHORIAS PROPOSTAS, ESPERAMOS:
    
📈 MÉTRICAS ALVO (REALISTAS):
    • Sharpe Ratio: 0.8 - 1.2 (atual: 0.23)
    • Win Rate: 55-60% (atual: 48%)
    • Profit Factor: 1.8 - 2.2 (atual: 1.7)
    • Return médio: 2-3% mensal (atual: 1.5%)
    • Max Drawdown: < 10% (atual: 15-16%)
    • Trades por mês: 20-30 (atual: ~50)
    
⚠️ REALIDADE DO MERCADO:
    • Sharpe > 1.5 é MUITO raro em trading real
    • Sharpe > 2.0 é quase impossível sustentavelmente
    • Hedge funds profissionais: Sharpe 0.8-1.2
    • Top quant funds: Sharpe 1.5-2.0 (com bilhões em infra)
    
✅ OBJETIVO REALISTA:
    Sharpe 1.0 com consistência é EXCELENTE para retail
    """
    
    print(expectations)

def main():
    """Executar análise completa"""
    
    print("="*80)
    print("🔍 ANÁLISE PROFUNDA DA PERFORMANCE SILUS")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Análise dos resultados
    df = analyze_evaluation_results()
    
    if df is not None and len(df) > 0:
        diagnose_problems(df)
    
    # Análise de causas
    analyze_root_causes()
    
    # Soluções
    propose_concrete_solutions()
    
    # Expectativas
    calculate_realistic_expectations()
    
    print("\n" + "="*80)
    print("✅ ANÁLISE CONCLUÍDA - AÇÃO IMEDIATA NECESSÁRIA!")
    print("="*80)

if __name__ == "__main__":
    main()