#!/usr/bin/env python3
"""
🎯 COMPARAÇÃO DAYTRADER: 8M vs FINAL (10.8M)
Análise para verificar se o treinamento adicional melhorou ou degradou
"""

import sys
import os
import traceback
from datetime import datetime
sys.path.append("D:/Projeto")

import numpy as np
import pandas as pd
import torch

# Configuração dos checkpoints DayTrader
DAYTRADER_8M = "D:/Projeto/Otimizacao/treino_principal/models/DAYTRADER/DAYTRADER_phase4stresstesting_8000000_steps_20250808_173027.zip"
DAYTRADER_FINAL = "D:/Projeto/Otimizacao/treino_principal/models/DAYTRADER/FINAL_ABSOLUTE_10836000_steps_20250808_234703.zip"

INITIAL_PORTFOLIO = 500.0
BASE_LOT_SIZE = 0.02
MAX_LOT_SIZE = 0.03
TEST_STEPS = 5000

def test_daytrader_checkpoint(checkpoint_path, checkpoint_name):
    """Testa um checkpoint específico do DayTrader"""
    
    print(f"\n🔥 TESTANDO {checkpoint_name}")
    print("=" * 50)
    
    try:
        # Imports
        from sb3_contrib import RecurrentPPO
        from daytrader import TradingEnv
        
        # Dataset real - usar SEMPRE a mesma seção
        dataset_path = "D:/Projeto/data/GC_YAHOO_ENHANCED_V3_BALANCED_20250804_192226.csv"
        
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset não encontrado: {dataset_path}")
            return None
            
        df = pd.read_csv(dataset_path)
        
        # Processar dataset
        if 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'])
            df.set_index('timestamp', inplace=True)
            df.drop('time', axis=1, inplace=True)
        
        # Renomear colunas
        df = df.rename(columns={
            'open': 'open_5m',
            'high': 'high_5m',
            'low': 'low_5m', 
            'close': 'close_5m',
            'tick_volume': 'volume_5m'
        })
        
        # Usar seção fixa para comparação justa
        total_len = len(df)
        start_idx = total_len // 4  # Quarto inicial
        end_idx = start_idx + 8000   # 8k barras
        test_df = df.iloc[start_idx:end_idx]
        
        print(f"📊 Dataset: {len(test_df):,} barras ({test_df.index.min()} - {test_df.index.max()})")
        
        # Criar ambiente
        trading_params = {
            'base_lot_size': BASE_LOT_SIZE,
            'max_lot_size': MAX_LOT_SIZE,
            'initial_balance': INITIAL_PORTFOLIO,
            'target_trades_per_day': 18,
            'stop_loss_range': (2.0, 8.0),
            'take_profit_range': (3.0, 15.0)
        }
        
        env = TradingEnv(
            test_df,
            window_size=20,
            is_training=False,
            initial_balance=INITIAL_PORTFOLIO,
            trading_params=trading_params
        )
        
        # Carregar modelo
        print(f"🤖 Carregando {checkpoint_name}...")
        model = RecurrentPPO.load(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu')
        model.policy.set_training_mode(False)
        
        # EXECUTAR 5 RUNS para estabilidade
        all_results = []
        
        for run in range(5):
            print(f"  🔄 Run {run+1}/5...")
            
            obs = env.reset()
            lstm_states = None
            done = False
            step = 0
            
            # Tracking por run
            portfolio_history = [INITIAL_PORTFOLIO]
            trades_log = []
            entry_qualities = []
            actions_log = []
            
            while not done and step < TEST_STEPS:
                # Predição não-determinística
                action, lstm_states = model.predict(obs, state=lstm_states, deterministic=False)
                
                obs, reward, done, info = env.step(action)
                
                entry_quality = float(action[1])
                entry_qualities.append(entry_quality)
                
                actions_log.append({
                    'entry_decision': int(action[0]),
                    'entry_quality': entry_quality
                })
                
                # Log trades
                if hasattr(info, 'get') and info.get('trade_closed', False):
                    trades_log.append({
                        'pnl': info.get('trade_pnl', 0),
                        'type': info.get('trade_type', 'unknown'),
                        'entry_quality': entry_quality
                    })
                
                portfolio_history.append(env.portfolio_value)
                step += 1
            
            # Métricas do run
            final_portfolio = env.portfolio_value
            total_return = ((final_portfolio - INITIAL_PORTFOLIO) / INITIAL_PORTFOLIO) * 100
            
            avg_entry_quality = np.mean(entry_qualities)
            median_entry_quality = np.median(entry_qualities)
            std_entry_quality = np.std(entry_qualities)
            
            # Distribuição Entry Quality
            eq_array = np.array(entry_qualities)
            zeros_pct = (eq_array == 0.0).sum() / len(eq_array) * 100
            ones_pct = (eq_array == 1.0).sum() / len(eq_array) * 100
            middle_pct = 100 - zeros_pct - ones_pct
            
            # Trading stats
            total_trades = len(trades_log)
            win_rate = 0
            avg_profit = 0
            avg_loss = 0
            
            if total_trades > 0:
                profitable_trades = [t for t in trades_log if t['pnl'] > 0]
                losing_trades = [t for t in trades_log if t['pnl'] < 0]
                
                win_rate = (len(profitable_trades) / total_trades) * 100
                avg_profit = np.mean([t['pnl'] for t in profitable_trades]) if profitable_trades else 0
                avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            # Ações
            decisions = [a['entry_decision'] for a in actions_log]
            hold_pct = (sum(1 for d in decisions if d == 0) / len(decisions)) * 100
            long_pct = (sum(1 for d in decisions if d == 1) / len(decisions)) * 100
            short_pct = (sum(1 for d in decisions if d == 2) / len(decisions)) * 100
            
            # Drawdown
            portfolio_array = np.array(portfolio_history)
            running_max = np.maximum.accumulate(portfolio_array)
            drawdown = (portfolio_array - running_max) / running_max * 100
            max_drawdown = np.min(drawdown)
            
            run_result = {
                'run': run + 1,
                'final_portfolio': final_portfolio,
                'total_return': total_return,
                'avg_entry_quality': avg_entry_quality,
                'median_entry_quality': median_entry_quality,
                'std_entry_quality': std_entry_quality,
                'zeros_pct': zeros_pct,
                'ones_pct': ones_pct,
                'middle_pct': middle_pct,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'hold_pct': hold_pct,
                'long_pct': long_pct,
                'short_pct': short_pct,
                'max_drawdown': max_drawdown
            }
            
            all_results.append(run_result)
        
        # Calcular médias e desvios
        metrics = {}
        for key in all_results[0].keys():
            if key != 'run':
                values = [r[key] for r in all_results]
                metrics[f'{key}_avg'] = np.mean(values)
                metrics[f'{key}_std'] = np.std(values)
                metrics[f'{key}_min'] = np.min(values)
                metrics[f'{key}_max'] = np.max(values)
        
        metrics['checkpoint_name'] = checkpoint_name
        metrics['checkpoint_path'] = checkpoint_path
        metrics['individual_runs'] = all_results
        
        print(f"✅ {checkpoint_name} testado com sucesso!")
        return metrics
        
    except Exception as e:
        print(f"❌ ERRO testando {checkpoint_name}: {e}")
        print(f"Detalhes: {traceback.format_exc()}")
        return None

def compare_daytrader_checkpoints(dt_8m_metrics, dt_final_metrics):
    """Comparação detalhada entre 8M e Final"""
    
    print("\n" + "=" * 80)
    print("🏆 DAYTRADER: 8M vs FINAL (10.8M) - ANÁLISE COMPLETA")
    print("=" * 80)
    
    print(f"\n📊 PERFORMANCE FINANCEIRA:")
    print(f"{'Métrica':<20} {'8M Steps':<15} {'Final 10.8M':<15} {'Diferença':<15} {'Melhoria':<10}")
    print("-" * 75)
    
    dt8_return = dt_8m_metrics['total_return_avg']
    dtf_return = dt_final_metrics['total_return_avg']
    return_diff = dtf_return - dt8_return
    return_improvement = "✅" if return_diff > 0 else "❌" if return_diff < -1 else "➖"
    
    dt8_portfolio = dt_8m_metrics['final_portfolio_avg']
    dtf_portfolio = dt_final_metrics['final_portfolio_avg']
    portfolio_diff = dtf_portfolio - dt8_portfolio
    portfolio_improvement = "✅" if portfolio_diff > 0 else "❌" if portfolio_diff < -5 else "➖"
    
    dt8_dd = dt_8m_metrics['max_drawdown_avg']
    dtf_dd = dt_final_metrics['max_drawdown_avg']
    dd_diff = dtf_dd - dt8_dd
    dd_improvement = "✅" if dd_diff > -1 else "❌" if dd_diff < -5 else "➖"
    
    print(f"{'Retorno %':<20} {dt8_return:>+7.2f}%      {dtf_return:>+7.2f}%      {return_diff:>+7.2f}%      {return_improvement}")
    print(f"{'Portfolio Final':<20} ${dt8_portfolio:>7.2f}      ${dtf_portfolio:>7.2f}      ${portfolio_diff:>+7.2f}      {portfolio_improvement}")
    print(f"{'Max Drawdown':<20} {dt8_dd:>7.2f}%      {dtf_dd:>7.2f}%      {dd_diff:>+7.2f}%      {dd_improvement}")
    
    print(f"\n🎯 ENTRY QUALITY EVOLUTION:")
    print(f"{'Métrica':<20} {'8M Steps':<15} {'Final 10.8M':<15} {'Diferença':<15} {'Melhoria':<10}")
    print("-" * 75)
    
    dt8_eq = dt_8m_metrics['avg_entry_quality_avg']
    dtf_eq = dt_final_metrics['avg_entry_quality_avg']
    eq_diff = dtf_eq - dt8_eq
    eq_improvement = "✅" if eq_diff > 0.005 else "❌" if eq_diff < -0.005 else "➖"
    
    dt8_med = dt_8m_metrics['median_entry_quality_avg']
    dtf_med = dt_final_metrics['median_entry_quality_avg']
    med_diff = dtf_med - dt8_med
    med_improvement = "✅" if med_diff > 0.01 else "❌" if med_diff < -0.01 else "➖"
    
    dt8_std = dt_8m_metrics['std_entry_quality_avg']
    dtf_std = dt_final_metrics['std_entry_quality_avg']
    std_diff = dtf_std - dt8_std
    std_improvement = "✅" if abs(std_diff) < 0.01 else "❌"  # Menor std é melhor
    
    print(f"{'EQ Média':<20} {dt8_eq:>7.3f}        {dtf_eq:>7.3f}        {eq_diff:>+7.3f}        {eq_improvement}")
    print(f"{'EQ Mediana':<20} {dt8_med:>7.3f}        {dtf_med:>7.3f}        {med_diff:>+7.3f}        {med_improvement}")
    print(f"{'EQ Std Dev':<20} {dt8_std:>7.3f}        {dtf_std:>7.3f}        {std_diff:>+7.3f}        {std_improvement}")
    
    print(f"\n📊 DISTRIBUIÇÃO ENTRY QUALITY:")
    print(f"{'Métrica':<20} {'8M Steps':<15} {'Final 10.8M':<15} {'Diferença':<15} {'Melhoria':<10}")
    print("-" * 75)
    
    dt8_zeros = dt_8m_metrics['zeros_pct_avg']
    dtf_zeros = dt_final_metrics['zeros_pct_avg']
    zeros_diff = dtf_zeros - dt8_zeros
    zeros_improvement = "✅" if zeros_diff < -1 else "❌" if zeros_diff > 1 else "➖"
    
    dt8_ones = dt_8m_metrics['ones_pct_avg']
    dtf_ones = dt_final_metrics['ones_pct_avg']
    ones_diff = dtf_ones - dt8_ones
    ones_improvement = "✅" if ones_diff < -1 else "❌" if ones_diff > 1 else "➖"
    
    dt8_middle = dt_8m_metrics['middle_pct_avg']
    dtf_middle = dt_final_metrics['middle_pct_avg']
    middle_diff = dtf_middle - dt8_middle
    middle_improvement = "✅" if middle_diff > 1 else "❌" if middle_diff < -1 else "➖"
    
    print(f"{'% Zeros':<20} {dt8_zeros:>7.1f}%        {dtf_zeros:>7.1f}%        {zeros_diff:>+7.1f}%        {zeros_improvement}")
    print(f"{'% Ones':<20} {dt8_ones:>7.1f}%        {dtf_ones:>7.1f}%        {ones_diff:>+7.1f}%        {ones_improvement}")
    print(f"{'% Middle':<20} {dt8_middle:>7.1f}%        {dtf_middle:>7.1f}%        {middle_diff:>+7.1f}%        {middle_improvement}")
    
    print(f"\n💼 TRADING BEHAVIOR:")
    print(f"{'Métrica':<20} {'8M Steps':<15} {'Final 10.8M':<15} {'Diferença':<15} {'Melhoria':<10}")
    print("-" * 75)
    
    dt8_trades = dt_8m_metrics['total_trades_avg']
    dtf_trades = dt_final_metrics['total_trades_avg']
    trades_diff = dtf_trades - dt8_trades
    trades_improvement = "✅" if trades_diff > 0.5 else "❌" if trades_diff < -0.5 else "➖"
    
    dt8_wr = dt_8m_metrics['win_rate_avg']
    dtf_wr = dt_final_metrics['win_rate_avg']
    wr_diff = dtf_wr - dt8_wr
    wr_improvement = "✅" if wr_diff > 2 else "❌" if wr_diff < -2 else "➖"
    
    dt8_hold = dt_8m_metrics['hold_pct_avg']
    dtf_hold = dt_final_metrics['hold_pct_avg']
    hold_diff = dtf_hold - dt8_hold
    hold_improvement = "✅" if abs(hold_diff) < 2 else "❌"  # Hold estável é bom
    
    print(f"{'Total Trades':<20} {dt8_trades:>7.1f}        {dtf_trades:>7.1f}        {trades_diff:>+7.1f}        {trades_improvement}")
    print(f"{'Win Rate %':<20} {dt8_wr:>7.1f}%        {dtf_wr:>7.1f}%        {wr_diff:>+7.1f}%        {wr_improvement}")
    print(f"{'% Hold':<20} {dt8_hold:>7.1f}%        {dtf_hold:>7.1f}%        {hold_diff:>+7.1f}%        {hold_improvement}")
    
    # ANÁLISE DE PROGRESSO
    print(f"\n🔍 ANÁLISE DE PROGRESSO (8M → 10.8M):")
    print("=" * 50)
    
    improvements = 0
    degradations = 0
    neutral = 0
    
    metrics_analysis = [
        ("Retorno", return_diff, return_improvement),
        ("Entry Quality", eq_diff, eq_improvement),
        ("EQ Mediana", med_diff, med_improvement),
        ("% Zeros", zeros_diff, zeros_improvement),
        ("% Middle", middle_diff, middle_improvement),
        ("Drawdown", dd_diff, dd_improvement)
    ]
    
    for metric_name, diff_value, improvement_status in metrics_analysis:
        if improvement_status == "✅":
            improvements += 1
        elif improvement_status == "❌":
            degradations += 1
        else:
            neutral += 1
    
    print(f"✅ Melhorias: {improvements}")
    print(f"❌ Degradações: {degradations}")  
    print(f"➖ Neutras: {neutral}")
    
    # VEREDICTO FINAL
    print(f"\n🏆 VEREDICTO FINAL:")
    print("=" * 30)
    
    if improvements > degradations:
        if eq_diff > 0.01:  # Entry Quality melhorou significativamente
            verdict = "🟢 TREINAMENTO ADICIONAL VALEU A PENA"
            recommendation = "✅ Modelo final (10.8M) é superior ao 8M"
        else:
            verdict = "🟡 LEVE MELHORIA"
            recommendation = "🤔 Melhoria marginal - ambos são similares"
    elif improvements < degradations:
        verdict = "🔴 TREINAMENTO ADICIONAL FOI PREJUDICIAL" 
        recommendation = "⚠️ Modelo 8M pode ser melhor que o final"
    else:
        verdict = "🟡 SEM DIFERENÇA SIGNIFICATIVA"
        recommendation = "➖ Ambos modelos têm performance similar"
    
    print(f"{verdict}")
    print(f"💡 {recommendation}")
    
    # Análise específica Entry Quality
    print(f"\n📊 FOCO EM ENTRY QUALITY:")
    target_original = 0.488
    
    dt8_vs_target = dt8_eq - target_original
    dtf_vs_target = dtf_eq - target_original
    
    print(f"🎯 vs Target Original (0.488):")
    print(f"   8M Steps: {dt8_eq:.3f} ({dt8_vs_target:+.3f})")
    print(f"   Final: {dtf_eq:.3f} ({dtf_vs_target:+.3f})")
    
    if dtf_eq > dt8_eq and dtf_eq > target_original:
        eq_verdict = "✅ FINAL é melhor (Entry Quality mais alta)"
    elif dt8_eq > dtf_eq and dt8_eq > target_original:
        eq_verdict = "⚠️ 8M era melhor (degradou no final)"
    else:
        eq_verdict = "➖ Ambos similares ou abaixo do target"
    
    print(f"🔍 {eq_verdict}")

def main():
    """Função principal"""
    
    print("🔥 COMPARAÇÃO DAYTRADER: 8M vs FINAL")
    print("=" * 50)
    print("📊 Análise: Treinamento adicional melhorou?")
    print("🎯 Foco: Entry Quality evolution")
    print("💰 Portfolio: $500 inicial")
    print("🔄 Runs: 5 runs x 5000 steps")
    print("=" * 50)
    
    # Testar DayTrader 8M
    dt_8m_metrics = test_daytrader_checkpoint(DAYTRADER_8M, "DAYTRADER 8M")
    
    # Testar DayTrader Final
    dt_final_metrics = test_daytrader_checkpoint(DAYTRADER_FINAL, "DAYTRADER FINAL")
    
    if dt_8m_metrics and dt_final_metrics:
        # Comparação detalhada
        compare_daytrader_checkpoints(dt_8m_metrics, dt_final_metrics)
    else:
        print("❌ Falha ao testar um ou ambos os checkpoints")

if __name__ == "__main__":
    print(f"🚀 INICIANDO COMPARAÇÃO 8M vs FINAL - {datetime.now().strftime('%H:%M:%S')}")
    main()
    print(f"\n✅ COMPARAÇÃO CONCLUÍDA - {datetime.now().strftime('%H:%M:%S')}")